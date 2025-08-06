"""
Test correctness of 8-frame vertical pair WMMA against individual diagonal ATA matvec operations.

This test compares the new 32x8x16 WMMA vertical pair approach against 8 individual
diagonal ATA matrix-vector multiplications. We expect small differences due to TF32
vs FP32 precision differences in tensor cores.
"""

import sys
from pathlib import Path

# Add project root to path for tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pytest

try:
    import cupy

    print(f"CUDA available: {cupy.cuda.runtime.runtimeGetVersion()}")
    GPU_AVAILABLE = True
except ImportError:
    print("CUDA not available")
    GPU_AVAILABLE = False
    # Don't exit in test mode
    cupy = None


def create_test_diagonal_ata_matrix(led_count: int, channels: int = 3):
    """Create a test diagonal ATA matrix with known properties."""
    from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix

    # Create batch matrix
    batch_matrix = BatchSymmetricDiagonalATAMatrix(led_count=led_count, batch_size=8)

    # Create synthetic ATA data - diagonal dominant for stability
    block_diag_count = 5  # Small number of diagonals for testing

    # Create block offsets (main diagonal + a few upper diagonals)
    block_offsets = np.array([0, 1, 2, 3, 4], dtype=np.int32)  # Main + 4 upper diagonals

    # Create block data with realistic values
    np.random.seed(42)  # Reproducible results
    block_data = np.zeros((channels, block_diag_count, 16, 16), dtype=np.float32)

    for c in range(channels):
        for d in range(block_diag_count):
            # Create positive definite-like blocks
            if d == 0:  # Main diagonal - stronger
                block_data[c, d] = np.eye(16, dtype=np.float32) * 2.0 + np.random.normal(0, 0.1, (16, 16)).astype(
                    np.float32
                )
            else:  # Off-diagonals - weaker
                block_data[c, d] = np.random.normal(0, 0.05, (16, 16)).astype(np.float32)

            # Make symmetric for diagonal blocks
            if d == 0:
                block_data[c, d] = (block_data[c, d] + block_data[c, d].T) / 2

    return batch_matrix, block_data, block_offsets


def compute_individual_matvec(block_data, block_offsets, input_vectors, led_blocks, channels):
    """Compute ATA @ input_vector for each input vector individually using FP32."""
    batch_size = input_vectors.shape[0]
    padded_leds = input_vectors.shape[2]

    results = []

    for batch_idx in range(batch_size):
        input_vec = input_vectors[batch_idx]  # (channels, padded_leds)
        output_vec = np.zeros_like(input_vec)

        for c in range(channels):
            for d, offset in enumerate(block_offsets):
                for i in range(led_blocks):
                    j = i + offset
                    if j >= led_blocks:
                        continue

                    # Get block
                    block = block_data[c, d]  # (16, 16)

                    # Get input slice
                    input_slice = input_vec[c, j * 16 : (j + 1) * 16]  # (16,)

                    # Matrix-vector multiply: block @ input_slice
                    output_slice = block @ input_slice  # (16,)

                    # Add to output
                    output_vec[c, i * 16 : (i + 1) * 16] += output_slice

                    # Symmetric contribution (if off-diagonal)
                    if offset > 0:
                        # A^T @ input (transpose contribution)
                        input_slice_transpose = input_vec[c, i * 16 : (i + 1) * 16]
                        output_slice_transpose = block.T @ input_slice_transpose
                        output_vec[c, j * 16 : (j + 1) * 16] += output_slice_transpose

        results.append(output_vec)

    return np.stack(results, axis=0)  # (batch_size, channels, padded_leds)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA not available")
def test_8frame_correctness():
    """Test 8-frame vertical pair WMMA correctness."""
    print("\n=== Testing 8-Frame Vertical Pair WMMA Correctness ===")

    # Test parameters
    led_count = 64  # Small enough for detailed debugging
    channels = 3
    batch_size = 8

    try:
        # Create test matrix and data
        batch_matrix, block_data, block_offsets = create_test_diagonal_ata_matrix(led_count, channels)

        print(f"Test matrix: {led_count} LEDs, {channels} channels, {batch_size} frames")
        print(f"Block dimensions: {batch_matrix.led_blocks} x {batch_matrix.led_blocks}")
        print(f"Padded LED count: {batch_matrix.padded_led_count}")
        print(f"Block diagonals: {len(block_offsets)}")

        # Load matrix data directly (for testing)
        batch_matrix.block_data_gpu = cupy.array(block_data)
        batch_matrix.block_offsets_upper = cupy.array(block_offsets, dtype=cupy.int32)
        batch_matrix.block_diag_count = len(block_offsets)

        # Create test input vectors
        np.random.seed(123)
        input_batch = np.random.normal(0, 1, (batch_size, channels, batch_matrix.padded_led_count)).astype(np.float32)
        input_batch_gpu = cupy.array(input_batch)

        print(f"Input batch shape: {input_batch.shape}")

        # Method 1: 8-frame vertical pair WMMA (TF32 precision)
        print("\nComputing with 8-frame vertical pair WMMA...")
        result_wmma = batch_matrix.multiply_batch8_3d(input_batch_gpu)
        result_wmma_cpu = cupy.asnumpy(result_wmma)

        # Method 2: Individual diagonal ATA matvec operations (FP32 precision)
        print("Computing with individual diagonal ATA matvec operations...")
        result_individual = compute_individual_matvec(
            block_data, block_offsets, input_batch, batch_matrix.led_blocks, channels
        )

        # Compare results
        print("\n=== Correctness Analysis ===")

        # Calculate various error metrics
        abs_diff = np.abs(result_wmma_cpu - result_individual)
        rel_diff = abs_diff / (np.abs(result_individual) + 1e-8)

        max_abs_error = np.max(abs_diff)
        mean_abs_error = np.mean(abs_diff)
        max_rel_error = np.max(rel_diff)
        mean_rel_error = np.mean(rel_diff)

        print(f"Maximum absolute error: {max_abs_error:.2e}")
        print(f"Mean absolute error: {mean_abs_error:.2e}")
        print(f"Maximum relative error: {max_rel_error:.2e} ({max_rel_error*100:.4f}%)")
        print(f"Mean relative error: {mean_rel_error:.2e} ({mean_rel_error*100:.4f}%)")

        # Check magnitudes
        wmma_magnitude = np.sqrt(np.mean(result_wmma_cpu**2))
        individual_magnitude = np.sqrt(np.mean(result_individual**2))
        print(f"WMMA result magnitude: {wmma_magnitude:.6f}")
        print(f"Individual result magnitude: {individual_magnitude:.6f}")

        # Assess correctness (expect TF32 vs FP32 differences)
        # TF32 typically gives errors in the range of 1e-3 to 1e-4 relative error
        SUCCESS_THRESHOLD_REL = 1e-2  # 1% relative error threshold
        SUCCESS_THRESHOLD_ABS = 1e-4  # Absolute error threshold

        if max_rel_error < SUCCESS_THRESHOLD_REL and mean_rel_error < SUCCESS_THRESHOLD_REL / 10:
            print("\n✅ CORRECTNESS TEST PASSED!")
            print("   Errors are within expected range for TF32 vs FP32 differences")
            print(f"   Max relative error {max_rel_error*100:.4f}% < {SUCCESS_THRESHOLD_REL*100:.1f}% threshold")
            return True
        else:
            print("\n❌ CORRECTNESS TEST FAILED!")
            print("   Errors exceed expected range for TF32 vs FP32 differences")
            print(f"   Max relative error {max_rel_error*100:.4f}% >= {SUCCESS_THRESHOLD_REL*100:.1f}% threshold")

            # Show some sample comparisons for debugging
            print("\nSample comparisons (first 5 elements of first frame, first channel):")
            for i in range(min(5, len(result_wmma_cpu[0, 0]))):
                wmma_val = result_wmma_cpu[0, 0, i]
                indiv_val = result_individual[0, 0, i]
                diff = abs(wmma_val - indiv_val)
                rel_err = diff / (abs(indiv_val) + 1e-8) * 100
                print(f"  [{i}] WMMA: {wmma_val:.6f}, Individual: {indiv_val:.6f}, Error: {diff:.2e} ({rel_err:.3f}%)")

            return False

    except Exception as e:
        print(f"❌ Error during correctness test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("GPU required for testing")
        sys.exit(1)

    success = test_8frame_correctness()

    if success:
        print("\n🎉 8-Frame Vertical Pair WMMA Correctness Test PASSED!")
        print("   Implementation correctly uses 32x8x16 WMMA operations with vertical block pairs")
        sys.exit(0)
    else:
        print("\n💥 8-Frame Vertical Pair WMMA Correctness Test FAILED!")
        print("   Need to debug kernel implementation")
        sys.exit(1)
