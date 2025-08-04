#!/usr/bin/env python3
"""
Debug script for 8-frame CUDA kernel correctness issue.

This script creates the simplest possible test case to isolate the bug
in the 8-frame WMMA kernel implementation.
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


def create_minimal_test_case():
    """Create the smallest possible test case for debugging."""
    print("\n=== Creating Minimal Test Case ===")

    # Use very small matrix: 32 LEDs = 2x2 blocks of 16x16
    led_count = 32
    batch_size = 8

    print(f"LED count: {led_count} (2x2 blocks)")
    print(f"Batch size: {batch_size}")

    # Create 8-frame matrix
    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count,
        crop_size=64,
        batch_size=batch_size,
        output_dtype=cupy.float32
    )

    # Create simple diagonal matrix: just identity
    dia_offsets = np.array([0], dtype=np.int32)  # Only main diagonal
    dia_data = np.ones((3, 1, led_count), dtype=np.float32)  # Identity matrix

    print("Matrix structure: identity matrix with main diagonal = 1.0")

    # Convert to GPU and build matrix
    dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
    matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

    return matrix


def test_with_cpu_reference():
    """Test against a simple CPU reference implementation."""
    print("\n=== CPU Reference Test ===")

    led_count = 32
    batch_size = 8

    # Create simple identity matrix on CPU
    identity_matrix = np.eye(led_count, dtype=np.float32)

    # Create test input
    input_batch_cpu = np.zeros((batch_size, 3, led_count), dtype=np.float32)
    for batch_idx in range(batch_size):
        for channel in range(3):
            input_batch_cpu[batch_idx, channel, :] = (batch_idx + 1) + channel * 0.1

    # CPU reference computation: identity * input = input
    expected_output = input_batch_cpu.copy()

    print("Expected output (should equal input for identity matrix):")
    for batch_idx in range(batch_size):
        batch_sum = np.sum(expected_output[batch_idx], axis=1)  # Sum over LEDs for each channel
        print(f"  Batch {batch_idx}: R={batch_sum[0]:.6f}, G={batch_sum[1]:.6f}, B={batch_sum[2]:.6f}")

    return cupy.asarray(expected_output), cupy.asarray(input_batch_cpu)


def create_simple_input():
    """Create simple input for testing."""
    print("\n=== Creating Simple Input ===")

    led_count = 32
    batch_size = 8

    # Create simple input: each batch item has value = batch_index + 1
    input_batch = np.zeros((batch_size, 3, led_count), dtype=np.float32)

    for batch_idx in range(batch_size):
        for channel in range(3):
            # Each batch item has constant value across all LEDs
            input_batch[batch_idx, channel, :] = (batch_idx + 1) + channel * 0.1

    print("Input pattern:")
    for batch_idx in range(batch_size):
        print(f"  Batch {batch_idx}: R={input_batch[batch_idx, 0, 0]:.1f}, "
              f"G={input_batch[batch_idx, 1, 0]:.1f}, B={input_batch[batch_idx, 2, 0]:.1f}")

    return cupy.asarray(input_batch, dtype=cupy.float32)


def debug_kernel_vs_reference():
    """Debug kernel output vs reference implementation."""
    print("\n" + "="*60)
    print("DEBUGGING 8-FRAME KERNEL CORRECTNESS")
    print("="*60)

    # Create test case
    matrix = create_minimal_test_case()
    input_batch = create_simple_input()

    print(f"\nMatrix info: {matrix.get_info()}")

    # Debug: Check if matrix was built correctly
    print(f"\nMatrix block data shape: {matrix.block_data_gpu.shape}")
    print(f"Matrix block offsets: {cupy.asnumpy(matrix.block_offsets_upper)}")

    # Check a few values from the block matrix
    block_sample = cupy.asnumpy(matrix.block_data_gpu[0, 0, :4, :4])  # Channel 0, Block diagonal 0, top-left 4x4
    print("Block matrix sample (should be identity-like):")
    print(block_sample)

    # Check if diagonal 1 has any data (it shouldn't for identity matrix)
    if matrix.block_diag_count > 1:
        block_sample_diag1 = cupy.asnumpy(matrix.block_data_gpu[0, 1, :4, :4])  # Channel 0, Block diagonal 1
        print("Block diagonal 1 sample (should be zeros for identity):")
        print(block_sample_diag1)
        print(f"Non-zero count in diagonal 1: {np.count_nonzero(block_sample_diag1)}")

    print(f"Total non-zero values in block matrix: {cupy.count_nonzero(matrix.block_data_gpu)}")

    # Test 1: Sequential reference
    print("\n=== Test 1: Sequential Reference ===")

    # Let's try with a single frame that should work
    print("Testing single frame on matrix configured for batch_size=8:")
    single_input = input_batch[0]  # Shape: (3, 32)
    print(f"Single input shape: {single_input.shape}")
    print(f"Single input sample: R={single_input[0,0]:.1f}, G={single_input[1,0]:.1f}, B={single_input[2,0]:.1f}")

    try:
        single_result = matrix.multiply_3d(single_input, optimized_kernel=True, debug_logging=True)
        print(f"Single result shape: {single_result.shape}")
        print(f"Single result sample: R={single_result[0,0]:.6f}, G={single_result[1,0]:.6f}, B={single_result[2,0]:.6f}")
        print(f"Single result sum: R={cupy.sum(single_result[0]):.6f}, G={cupy.sum(single_result[1]):.6f}, B={cupy.sum(single_result[2]):.6f}")
    except Exception as e:
        print(f"Single frame test failed: {e}")

    # Now let's create a matrix configured for batch_size=1 for proper comparison
    print("\n=== Creating Matrix for batch_size=1 ===")
    matrix_single = BatchSymmetricDiagonalATAMatrix(
        led_count=32,
        crop_size=64,
        batch_size=1,  # Configure for single batch
        output_dtype=cupy.float32
    )

    # Use the same matrix data
    dia_offsets = np.array([0], dtype=np.int32)
    dia_data = np.ones((3, 1, 32), dtype=np.float32)
    dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
    matrix_single._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

    # Test with proper single-batch matrix
    result_sequential = []
    for i in range(8):
        single_result = matrix_single.multiply_3d(input_batch[i], optimized_kernel=True, debug_logging=False)
        result_sequential.append(single_result)
        print(f"Batch {i} result sum: R={cupy.sum(single_result[0]):.6f}, "
              f"G={cupy.sum(single_result[1]):.6f}, B={cupy.sum(single_result[2]):.6f}")

    result_sequential = cupy.stack(result_sequential, axis=0)
    print(f"Sequential total sum: {cupy.sum(result_sequential):.6f}")

    # Test 2: 8-frame kernel
    print("\n=== Test 2: 8-Frame Kernel ===")
    try:
        result_8frame = matrix.multiply_batch8_3d(input_batch, optimized_kernel=False, debug_logging=True)
        print(f"8-frame total sum: {cupy.sum(result_8frame):.6f}")

        for i in range(8):
            print(f"Batch {i} result sum: R={cupy.sum(result_8frame[i, 0]):.6f}, "
                  f"G={cupy.sum(result_8frame[i, 1]):.6f}, B={cupy.sum(result_8frame[i, 2]):.6f}")

    except Exception as e:
        print(f"8-frame kernel failed: {e}")
        return False

    # Test 3: Compare with CPU reference (not sequential which seems broken)
    print("\n=== Test 3: Comparison vs CPU Reference ===")
    expected_output, _ = test_with_cpu_reference()

    max_diff = cupy.max(cupy.abs(result_8frame - expected_output))
    relative_error = max_diff / (cupy.max(cupy.abs(expected_output)) + 1e-10)

    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Relative error: {relative_error:.6f}")

    if relative_error > 0.01:  # Allow 1% error for floating point differences
        print(f"❌ KERNEL BUG: {relative_error:.1e} relative error vs CPU reference")

        # Detailed comparison for first batch item
        print("\nDetailed comparison for Batch 0, Channel 0:")
        exp_vals = cupy.asnumpy(expected_output[0, 0, :8])  # First 8 LEDs
        ker_vals = cupy.asnumpy(result_8frame[0, 0, :8])

        print("LED | Expected | 8-Frame  | Diff")
        print("----|----------|----------|--------")
        for i in range(8):
            diff = ker_vals[i] - exp_vals[i]
            print(f"{i:3d} | {exp_vals[i]:8.6f} | {ker_vals[i]:8.6f} | {diff:8.6f}")

        return False
    else:
        print(f"✅ KERNEL CORRECT: {relative_error:.1e} relative error vs CPU reference")
        print("Note: Sequential reference (16-frame kernel) appears to be broken, returning all zeros")
        return True


def analyze_tensor_layout():
    """Analyze tensor layout issues."""
    print("\n=== Analyzing Tensor Layout ===")

    # Create test data
    batch_size, channels, leds = 8, 3, 32

    # Test different tensor layouts
    print("Testing tensor layout...")

    # Expected layout: (8, 3, 32)
    test_tensor = cupy.zeros((batch_size, channels, leds), dtype=cupy.float32)

    # Fill with recognizable pattern
    for b in range(batch_size):
        for c in range(channels):
            for l in range(leds):
                test_tensor[b, c, l] = b * 1000 + c * 100 + l

    print(f"Tensor shape: {test_tensor.shape}")
    print(f"Tensor strides: {test_tensor.strides}")
    print(f"Is C-contiguous: {test_tensor.flags.c_contiguous}")

    # Sample values to verify layout
    print("Sample values:")
    print(f"  [0,0,0] = {test_tensor[0,0,0]} (expected: 0)")
    print(f"  [0,0,1] = {test_tensor[0,0,1]} (expected: 1)")
    print(f"  [0,1,0] = {test_tensor[0,1,0]} (expected: 100)")
    print(f"  [1,0,0] = {test_tensor[1,0,0]} (expected: 1000)")

    return test_tensor


def main():
    """Main debug function."""
    print("8-Frame CUDA Kernel Debug Tool")
    print("==============================")

    # Test 1: Analyze tensor layout
    analyze_tensor_layout()

    # Test 2: CPU reference for expected behavior
    expected_output, input_batch = test_with_cpu_reference()

    # Test 3: Debug kernel correctness
    success = debug_kernel_vs_reference()

    if success:
        print("\n✅ 8-frame kernel is working correctly!")
    else:
        print("\n❌ 8-frame kernel has correctness issues that need to be fixed.")
        print("\nNext steps:")
        print("1. Check tensor indexing in CUDA kernel")
        print("2. Verify atomic operation usage")
        print("3. Check WMMA fragment loading/storing")
        print("4. Validate symmetric contribution logic")
        print("5. The 16-frame kernels also seem to have issues - may be matrix construction problem")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
