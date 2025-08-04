#!/usr/bin/env python3
"""
Debug script to test 8-frame CUDA kernel with random input patterns.

This specifically tests the pattern-dependent bug where the kernel works 
with simple constant patterns but fails with random input.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    import cupy
    CUDA_AVAILABLE = True
    print(f"✓ CUDA available: {cupy.cuda.runtime.runtimeGetVersion()}")
except ImportError:
    print("✗ CUDA not available")
    sys.exit(1)

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix


def create_identity_matrix(led_count: int, batch_size: int):
    """Create identity matrix for testing."""
    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count,
        crop_size=64,
        batch_size=batch_size,
        output_dtype=cupy.float32
    )

    # Create identity matrix (A*x = x)
    dia_offsets = np.array([0], dtype=np.int32)  # Only main diagonal
    dia_data = np.ones((3, 1, led_count), dtype=np.float32)  # Identity

    dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
    matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

    return matrix


def test_pattern_dependency():
    """Test the pattern-dependent bug."""
    print("\n" + "="*60)
    print("TESTING PATTERN-DEPENDENT BUG")
    print("="*60)

    led_count = 32
    batch_size = 8

    # Create identity matrix
    matrix = create_identity_matrix(led_count, batch_size)

    print(f"Testing {led_count} LEDs with batch_size={batch_size}")
    print(f"Matrix info: {matrix.get_info()}")

    # Test 1: Simple constant pattern (should work)
    print("\n=== Test 1: Simple Constant Pattern ===")

    input_simple = np.zeros((batch_size, 3, led_count), dtype=np.float32)
    for batch_idx in range(batch_size):
        for channel in range(3):
            # Constant value per batch/channel combination
            input_simple[batch_idx, channel, :] = (batch_idx + 1) + channel * 0.1

    input_simple_gpu = cupy.asarray(input_simple, dtype=cupy.float32)
    expected_simple = input_simple.copy()  # Identity: A*x = x

    result_simple = matrix.multiply_batch8_3d(input_simple_gpu, optimized_kernel=False, debug_logging=True)

    max_diff_simple = cupy.max(cupy.abs(result_simple - cupy.asarray(expected_simple)))
    relative_error_simple = max_diff_simple / (cupy.max(cupy.abs(result_simple)) + 1e-10)

    print(f"Simple pattern - Max diff: {max_diff_simple:.6f}, Relative error: {relative_error_simple:.6f}")

    # Test 2: Random pattern (likely to fail)
    print("\n=== Test 2: Random Pattern ===")

    np.random.seed(42)  # Reproducible random pattern
    input_random = np.random.randn(batch_size, 3, led_count).astype(np.float32)
    input_random_gpu = cupy.asarray(input_random, dtype=cupy.float32)
    expected_random = input_random.copy()  # Identity: A*x = x

    result_random = matrix.multiply_batch8_3d(input_random_gpu, optimized_kernel=False, debug_logging=True)

    max_diff_random = cupy.max(cupy.abs(result_random - cupy.asarray(expected_random)))
    relative_error_random = max_diff_random / (cupy.max(cupy.abs(result_random)) + 1e-10)

    print(f"Random pattern - Max diff: {max_diff_random:.6f}, Relative error: {relative_error_random:.6f}")

    # Test 3: Varying pattern within batch (another pattern type)
    print("\n=== Test 3: Varying Pattern Within Batch ===")

    input_varying = np.zeros((batch_size, 3, led_count), dtype=np.float32)
    for batch_idx in range(batch_size):
        for channel in range(3):
            for led_idx in range(led_count):
                # Different value for each LED position
                input_varying[batch_idx, channel, led_idx] = (batch_idx + 1) + channel * 0.1 + led_idx * 0.01

    input_varying_gpu = cupy.asarray(input_varying, dtype=cupy.float32)
    expected_varying = input_varying.copy()  # Identity: A*x = x

    result_varying = matrix.multiply_batch8_3d(input_varying_gpu, optimized_kernel=False, debug_logging=True)

    max_diff_varying = cupy.max(cupy.abs(result_varying - cupy.asarray(expected_varying)))
    relative_error_varying = max_diff_varying / (cupy.max(cupy.abs(result_varying)) + 1e-10)

    print(f"Varying pattern - Max diff: {max_diff_varying:.6f}, Relative error: {relative_error_varying:.6f}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Simple constant pattern: {relative_error_simple:.1e} relative error {'✅ PASS' if relative_error_simple < 0.01 else '❌ FAIL'}")
    print(f"Random pattern:          {relative_error_random:.1e} relative error {'✅ PASS' if relative_error_random < 0.01 else '❌ FAIL'}")
    print(f"Varying pattern:         {relative_error_varying:.1e} relative error {'✅ PASS' if relative_error_varying < 0.01 else '❌ FAIL'}")

    # Analyze the failures
    if relative_error_random > 0.01:
        print(f"\n❌ PATTERN-DEPENDENT BUG CONFIRMED: Random input fails with {relative_error_random:.1e} error")

        # Show detailed comparison for first few values
        print("Detailed comparison (Random pattern, Batch 0, Channel 0, first 8 LEDs):")
        expected_sample = expected_random[0, 0, :8]
        actual_sample = cupy.asnumpy(result_random[0, 0, :8])

        print("LED | Expected  | Actual    | Diff      | Ratio")
        print("----|-----------|-----------|-----------|----------")
        for i in range(8):
            diff = actual_sample[i] - expected_sample[i]
            ratio = actual_sample[i] / expected_sample[i] if abs(expected_sample[i]) > 1e-10 else float('inf')
            print(f"{i:3d} | {expected_sample[i]:9.6f} | {actual_sample[i]:9.6f} | {diff:9.6f} | {ratio:8.3f}")

    if relative_error_varying > 0.01:
        print(f"\n❌ LED-VARYING BUG CONFIRMED: Varying LED values fail with {relative_error_varying:.1e} error")

    # Return success status
    success = (relative_error_simple < 0.01 and
              relative_error_random < 0.01 and
              relative_error_varying < 0.01)

    return success


def main():
    """Main debug function."""
    print("8-Frame Pattern Dependency Debug Tool")
    print("====================================")

    success = test_pattern_dependency()

    if success:
        print("\n✅ All patterns work correctly - no pattern dependency bug!")
    else:
        print("\n❌ Pattern-dependent bug confirmed - kernel fails with certain input patterns")
        print("\nNext debugging steps:")
        print("1. Check CUDA kernel tensor indexing for different input patterns")
        print("2. Verify atomic operations don't have race conditions")
        print("3. Check if WMMA fragment loading handles varying values correctly")
        print("4. Validate shared memory layout for non-constant patterns")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
