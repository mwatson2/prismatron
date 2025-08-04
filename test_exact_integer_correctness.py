#!/usr/bin/env python3
"""
Test 8-frame WMMA kernel correctness with exact integer arithmetic.

Uses small integer values that can be exactly represented in TF32 to eliminate
floating-point precision errors and test pure algorithmic correctness.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np

try:
    import cupy

    print(f"CUDA available: {cupy.cuda.runtime.runtimeGetVersion()}")
    GPU_AVAILABLE = True
except ImportError:
    print("CUDA not available")
    GPU_AVAILABLE = False
    sys.exit(1)


def create_integer_ata_matrix(led_count=64, channels=3, num_diagonals=1, max_value=7):
    """
    Create ATA matrix with small integer values for exact TF32 representation.

    Args:
        led_count: Number of LEDs (must be multiple of 32)
        channels: Number of channels
        num_diagonals: Number of diagonals to include (1=main only, 2=main+1st, etc.)
        max_value: Maximum absolute integer value

    Returns:
        Tuple of (block_data, block_offsets, dense_matrix_for_reference)
    """
    np.random.seed(42)  # Reproducible results

    led_blocks = led_count // 16

    # Create block offsets (0, 1, 2, ..., num_diagonals-1)
    block_offsets = np.arange(num_diagonals, dtype=np.int32)

    # Create block data with small integers
    block_data = np.zeros((channels, num_diagonals, 16, 16), dtype=np.float32)

    # Create dense reference matrix for comparison
    dense_matrices = []

    for c in range(channels):
        dense_matrix = np.zeros((led_count, led_count), dtype=np.float32)

        for d, offset in enumerate(block_offsets):
            # Create integer values in range [-max_value, max_value]
            block_values = np.random.randint(-max_value, max_value + 1, (16, 16)).astype(np.float32)

            # For main diagonal, make it symmetric and positive definite-ish
            if offset == 0:
                block_values = (block_values + block_values.T) / 2  # Make symmetric
                block_values += np.eye(16) * (max_value + 1)  # Add to diagonal for stability

            block_data[c, d] = block_values

            # Fill dense matrix for all block positions on this diagonal
            for i in range(led_blocks):
                j = i + offset
                if j < led_blocks:
                    # Upper triangular
                    row_start, row_end = i * 16, (i + 1) * 16
                    col_start, col_end = j * 16, (j + 1) * 16
                    dense_matrix[row_start:row_end, col_start:col_end] = block_values

                    # Symmetric lower triangular (if not main diagonal)
                    if offset > 0:
                        dense_matrix[col_start:col_end, row_start:row_end] = block_values.T

        dense_matrices.append(dense_matrix)

    return block_data, block_offsets, dense_matrices


def create_integer_input_batch(batch_size=8, channels=3, led_count=64, max_value=15):
    """Create input batch with small integer values."""
    np.random.seed(123)  # Different seed from matrix

    # Create integer values in range [0, max_value]
    input_batch = np.random.randint(0, max_value + 1, (batch_size, channels, led_count)).astype(np.float32)

    return input_batch


def test_integer_correctness(num_diagonals=1, test_name=""):
    """Test WMMA kernel against dense numpy with exact integer arithmetic."""
    print(f"\n=== Testing {test_name} ({num_diagonals} diagonal{'s' if num_diagonals > 1 else ''}) ===")

    try:
        from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix

        # Test parameters
        led_count = 64  # Must be multiple of 32
        channels = 3
        batch_size = 8
        max_matrix_value = 7  # Small integers for exact representation
        max_input_value = 15

        # Create batch matrix
        batch_matrix = BatchSymmetricDiagonalATAMatrix(led_count=led_count, batch_size=batch_size)

        print(f"Matrix size: {led_count}x{led_count}")
        print(f"Block structure: {batch_matrix.led_blocks}x{batch_matrix.led_blocks}")
        print(f"Diagonals: {num_diagonals}")

        # Create integer ATA matrix
        block_data, block_offsets, dense_matrices = create_integer_ata_matrix(
            led_count, channels, num_diagonals, max_matrix_value
        )

        print(f"Block data shape: {block_data.shape}")
        print(f"Block offsets: {block_offsets}")
        print(f"Matrix value range: [{-max_matrix_value}, {max_matrix_value + 8}] (integers)")

        # Load into batch matrix
        batch_matrix.block_data_gpu = cupy.array(block_data)
        batch_matrix.block_offsets_upper = cupy.array(block_offsets, dtype=cupy.int32)
        batch_matrix.block_diag_count = len(block_offsets)

        # Create integer input batch
        input_batch = create_integer_input_batch(batch_size, channels, led_count, max_input_value)
        input_batch_gpu = cupy.array(input_batch)

        print(f"Input batch shape: {input_batch.shape}")
        print(f"Input value range: [0, {max_input_value}] (integers)")
        print(f"Sample input values: {input_batch[0, 0, :5]}")

        # Method 1: WMMA kernel
        print("\nComputing with 8-frame WMMA kernel...")
        result_wmma = batch_matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=False, debug_logging=False)
        result_wmma_cpu = cupy.asnumpy(result_wmma)

        # Method 2: Dense numpy reference
        print("Computing with dense numpy reference...")
        result_reference = np.zeros_like(input_batch)

        for b in range(batch_size):
            for c in range(channels):
                # Dense matrix multiplication: ATA @ input_vector
                result_reference[b, c] = dense_matrices[c] @ input_batch[b, c]

        # Compare results
        print("\n=== Exact Integer Correctness Analysis ===")

        abs_diff = np.abs(result_wmma_cpu - result_reference)
        max_abs_error = np.max(abs_diff)
        mean_abs_error = np.mean(abs_diff)

        # Calculate relative errors (avoid division by zero)
        rel_diff = abs_diff / (np.abs(result_reference) + 1e-10)
        max_rel_error = np.max(rel_diff)
        mean_rel_error = np.mean(rel_diff)

        print(f"Maximum absolute error: {max_abs_error:.6f}")
        print(f"Mean absolute error: {mean_abs_error:.6f}")
        print(f"Maximum relative error: {max_rel_error:.6f} ({max_rel_error*100:.4f}%)")
        print(f"Mean relative error: {mean_rel_error:.6f} ({mean_rel_error*100:.4f}%)")

        # Show sample comparisons
        print("\nSample comparisons (first 5 elements, batch 0, channel 0):")
        for i in range(5):
            wmma_val = result_wmma_cpu[0, 0, i]
            ref_val = result_reference[0, 0, i]
            error = abs(wmma_val - ref_val)
            rel_err = error / (abs(ref_val) + 1e-10) * 100
            print(f"  LED {i}: WMMA={wmma_val:8.3f}, Reference={ref_val:8.3f}, Error={error:.6f} ({rel_err:.4f}%)")

        # Check magnitudes
        wmma_magnitude = np.sqrt(np.mean(result_wmma_cpu**2))
        ref_magnitude = np.sqrt(np.mean(result_reference**2))
        print("\nResult magnitudes:")
        print(f"  WMMA: {wmma_magnitude:.6f}")
        print(f"  Reference: {ref_magnitude:.6f}")

        # Assess correctness with strict tolerance for integer arithmetic
        # Since we're using small integers, errors should be minimal
        SUCCESS_THRESHOLD_ABS = 0.01  # Very strict absolute error
        SUCCESS_THRESHOLD_REL = 0.001  # 0.1% relative error

        success = max_abs_error < SUCCESS_THRESHOLD_ABS and max_rel_error < SUCCESS_THRESHOLD_REL

        if success:
            print("\n✅ INTEGER CORRECTNESS TEST PASSED!")
            print(f"   Max abs error {max_abs_error:.6f} < {SUCCESS_THRESHOLD_ABS}")
            print(f"   Max rel error {max_rel_error*100:.4f}% < {SUCCESS_THRESHOLD_REL*100:.1f}%")
            return True
        else:
            print("\n❌ INTEGER CORRECTNESS TEST FAILED!")
            print(f"   Max abs error {max_abs_error:.6f} >= {SUCCESS_THRESHOLD_ABS}")
            print(f"   Max rel error {max_rel_error*100:.4f}% >= {SUCCESS_THRESHOLD_REL*100:.1f}%")

            # Show worst errors for debugging
            max_error_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            b_max, c_max, i_max = max_error_idx
            print(f"\nWorst error at [batch={b_max}, channel={c_max}, led={i_max}]:")
            print(f"  WMMA: {result_wmma_cpu[b_max, c_max, i_max]:.6f}")
            print(f"  Reference: {result_reference[b_max, c_max, i_max]:.6f}")
            print(f"  Error: {abs_diff[b_max, c_max, i_max]:.6f}")

            return False

    except Exception as e:
        print(f"❌ Error during integer correctness test: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_integer_tests():
    """Run integer correctness tests with different diagonal patterns."""
    print("🧮 Testing 8-Frame WMMA Kernel with Exact Integer Arithmetic")
    print("=" * 60)

    test_cases = [
        (1, "Main Diagonal Only"),
        (2, "Main + 1st Super-diagonal"),
        (3, "Main + 2 Super-diagonals"),
        (4, "Main + 3 Super-diagonals"),
    ]

    results = []

    for num_diagonals, test_name in test_cases:
        success = test_integer_correctness(num_diagonals, test_name)
        results.append((test_name, success))

    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGER CORRECTNESS TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n🎉 ALL INTEGER CORRECTNESS TESTS PASSED!")
        print("   8-frame WMMA kernel is algorithmically correct!")
    else:
        print("\n💥 SOME INTEGER CORRECTNESS TESTS FAILED!")
        print("   Kernel needs debugging for exact arithmetic!")

    return all_passed


if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("GPU required for testing")
        sys.exit(1)

    success = run_all_integer_tests()
    sys.exit(0 if success else 1)
