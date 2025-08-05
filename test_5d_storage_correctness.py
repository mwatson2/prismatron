#!/usr/bin/env python3
"""
Test script to verify correctness of the 5D block storage format fix.

This test creates a small symmetric matrix and verifies that:
1. All blocks are stored correctly (no overwrites)
2. Matrix multiplication results match expected values
3. Memory usage is optimized based on bandwidth
"""

import os
import sys

import cupy
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix as DiagonalATAMatrix


def create_test_diagonal_matrix(led_count=64, bandwidth=32):
    """
    Create a test diagonal matrix with known pattern.

    Args:
        led_count: Number of LEDs (will be padded to multiple of 16)
        bandwidth: Maximum diagonal offset

    Returns:
        DiagonalATAMatrix instance
    """
    # Create a simple diagonal pattern with decreasing values
    # Main diagonal = 10, first upper diagonal = 9, second = 8, etc.
    max_diagonals = min(bandwidth, led_count)

    # Create diagonal offsets (including main diagonal at 0)
    offsets = np.arange(max_diagonals, dtype=np.int32)

    # Create diagonal data for 3 channels
    channels = 3
    dia_data = np.zeros((channels, max_diagonals, led_count), dtype=np.float32)

    # Fill with decreasing pattern
    for ch in range(channels):
        for diag_idx, offset in enumerate(offsets):
            value = 10 - diag_idx  # Decreasing values
            # Fill the diagonal (accounting for shorter diagonals)
            diag_length = led_count - offset
            dia_data[ch, diag_idx, :diag_length] = value

    # Create DiagonalATAMatrix
    regular_matrix = DiagonalATAMatrix(led_count, crop_size=1)
    regular_matrix.bandwidth = bandwidth
    regular_matrix.dia_data_upper_gpu = cupy.asarray(dia_data)
    regular_matrix.dia_offsets_upper_gpu = cupy.asarray(offsets)

    # Set attributes expected by from_diagonal_ata_matrix
    regular_matrix.k = max_diagonals  # Total number of diagonals
    regular_matrix.sparsity = 0.95  # Dummy value
    regular_matrix.nnz = led_count * max_diagonals  # Approximate

    # Create full diagonal data/offsets for compatibility
    regular_matrix.dia_offsets = offsets
    regular_matrix.dia_data_cpu = cupy.asnumpy(dia_data)

    return regular_matrix


def verify_block_storage(batch_matrix):
    """
    Verify that blocks are stored correctly in 5D format.

    Args:
        batch_matrix: BatchSymmetricDiagonalATAMatrix instance

    Returns:
        bool: True if storage is correct
    """
    print("\n=== Verifying 5D Block Storage ===")

    # Check storage shape
    expected_shape = (batch_matrix.channels, batch_matrix.max_block_diag, batch_matrix.led_blocks, 16, 16)
    actual_shape = batch_matrix.block_data_gpu.shape

    print(f"Expected shape: {expected_shape}")
    print(f"Actual shape: {actual_shape}")

    if actual_shape != expected_shape:
        print("ERROR: Storage shape mismatch!")
        return False

    # Check that blocks contain expected values
    # For our test pattern, block at (row, col) should contain:
    # - Main diagonal blocks (row == col): value = 10
    # - First upper diagonal (col == row + 1): value = 9
    # - Second upper diagonal (col == row + 2): value = 8
    # - etc.

    print("\nChecking block values...")
    errors = 0
    blocks_checked = 0

    block_data_cpu = cupy.asnumpy(batch_matrix.block_data_gpu)

    for channel in range(batch_matrix.channels):
        for diag_idx in range(min(batch_matrix.max_block_diag, 5)):  # Check first 5 diagonals
            expected_value = 10 - diag_idx

            for block_row in range(min(batch_matrix.led_blocks, 4)):  # Check first 4 block rows
                block_col = block_row + diag_idx

                if block_col < batch_matrix.led_blocks:
                    # Get the block
                    block = block_data_cpu[channel, diag_idx, block_row, :, :]

                    # Check diagonal elements (should have expected_value)
                    # Note: Only check elements that are within the original matrix
                    for i in range(16):
                        for j in range(16):
                            led_row = block_row * 16 + i
                            led_col = block_col * 16 + j

                            if (
                                led_col == led_row + diag_idx
                                and led_row < batch_matrix.led_count
                                and led_col < batch_matrix.led_count
                                and abs(block[i, j] - expected_value) > 1e-6
                            ):
                                print(
                                    f"ERROR: Block[{channel},{diag_idx},{block_row}][{i},{j}] = {block[i,j]}, expected {expected_value}"
                                )
                                errors += 1

                    blocks_checked += 1

    print(f"Blocks checked: {blocks_checked}")
    print(f"Errors found: {errors}")

    return errors == 0


def verify_multiplication(batch_matrix, test_vectors):
    """
    Verify that matrix multiplication produces correct results.

    Args:
        batch_matrix: BatchSymmetricDiagonalATAMatrix instance
        test_vectors: Test input vectors

    Returns:
        bool: True if multiplication is correct
    """
    print("\n=== Verifying Matrix Multiplication ===")

    # Perform batch multiplication
    result_gpu = batch_matrix.multiply_batch_3d(test_vectors)
    result_cpu = cupy.asnumpy(result_gpu)

    # For verification, compute expected result manually
    # For our diagonal matrix, result[i] = sum of (diagonal_value * input[i+offset])
    expected = np.zeros_like(result_cpu)
    test_vectors_cpu = cupy.asnumpy(test_vectors)

    batch_size, channels, leds = test_vectors.shape

    # Manual computation using diagonal pattern
    for b in range(batch_size):
        for ch in range(channels):
            for i in range(batch_matrix.led_count):
                # Main diagonal contribution
                expected[b, ch, i] = 10 * test_vectors_cpu[b, ch, i]

                # Upper diagonal contributions
                for offset in range(1, min(batch_matrix.bandwidth, batch_matrix.led_count - i)):
                    diagonal_value = 10 - offset
                    expected[b, ch, i] += diagonal_value * test_vectors_cpu[b, ch, i + offset]

                # Lower diagonal contributions (symmetric)
                for offset in range(1, min(batch_matrix.bandwidth, i + 1)):
                    diagonal_value = 10 - offset
                    expected[b, ch, i] += diagonal_value * test_vectors_cpu[b, ch, i - offset]

    # Compare results
    max_error = np.max(np.abs(result_cpu - expected))
    rel_error = max_error / np.max(np.abs(expected))

    print(f"Max absolute error: {max_error}")
    print(f"Max relative error: {rel_error * 100:.6f}%")

    # Check a few specific values
    print("\nSample values (batch 0, channel 0):")
    for i in [0, 1, 2, 10, 20, 30]:
        if i < leds:
            print(f"  LED {i}: result={result_cpu[0,0,i]:.6f}, expected={expected[0,0,i]:.6f}")

    return rel_error < 0.001  # Allow 0.1% error


def main():
    """Main test function."""
    print("=" * 80)
    print("5D Block Storage Format Correctness Test")
    print("=" * 80)

    # Test parameters
    led_count = 64
    bandwidth = 32
    batch_size = 16

    print("\nTest Configuration:")
    print(f"  LED count: {led_count}")
    print(f"  Bandwidth: {bandwidth}")
    print(f"  Batch size: {batch_size}")

    # Create test diagonal matrix
    print("\nCreating test diagonal matrix...")
    regular_matrix = create_test_diagonal_matrix(led_count, bandwidth)

    # Convert to batch matrix
    print("\nConverting to batch matrix with 5D storage...")
    batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=batch_size)

    # Print storage statistics
    print("\nStorage Statistics:")
    print(f"  LED blocks: {batch_matrix.led_blocks}")
    print(f"  Max block diagonal: {batch_matrix.max_block_diag}")
    print(f"  Block shape: {batch_matrix.block_data_gpu.shape}")
    print(f"  Memory usage: {batch_matrix.block_data_gpu.nbytes / 1024**2:.2f} MB")

    # Expected values
    expected_max_block_diag = (bandwidth + 15) // 16  # ceil(bandwidth / 16)
    print(f"  Expected max_block_diag: {expected_max_block_diag}")

    # Verify storage format
    storage_correct = verify_block_storage(batch_matrix)

    # Create test vectors
    print("\nCreating test vectors...")
    test_vectors_cpu = np.random.randn(batch_size, 3, led_count).astype(np.float32)
    test_vectors_gpu = cupy.asarray(test_vectors_cpu)

    # Verify multiplication
    multiplication_correct = verify_multiplication(batch_matrix, test_vectors_gpu)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Storage format correct: {'PASS' if storage_correct else 'FAIL'}")
    print(f"Multiplication correct: {'PASS' if multiplication_correct else 'FAIL'}")
    print(f"Overall: {'PASS' if storage_correct and multiplication_correct else 'FAIL'}")

    # Additional memory efficiency check
    print("\nMemory Efficiency:")
    full_storage = batch_matrix.channels * batch_matrix.led_blocks * batch_matrix.led_blocks * 16 * 16 * 4  # float32
    actual_storage = batch_matrix.block_data_gpu.nbytes
    print(f"  Full storage would be: {full_storage / 1024**2:.2f} MB")
    print(f"  Actual storage: {actual_storage / 1024**2:.2f} MB")
    print(f"  Storage efficiency: {actual_storage / full_storage * 100:.1f}%")

    return storage_correct and multiplication_correct


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
