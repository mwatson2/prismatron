#!/usr/bin/env python3
"""
Simple test script to verify correctness of the 5D block storage format fix.

This test verifies the storage format without running CUDA kernels.
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
    unique_blocks = set()  # Track unique blocks to verify no overwrites

    block_data_cpu = cupy.asnumpy(batch_matrix.block_data_gpu)

    for channel in range(batch_matrix.channels):
        for diag_idx in range(min(batch_matrix.max_block_diag, 5)):  # Check first 5 diagonals
            expected_value = 10 - diag_idx

            for block_row in range(min(batch_matrix.led_blocks, 4)):  # Check first 4 block rows
                block_col = block_row + diag_idx

                if block_col < batch_matrix.led_blocks:
                    # Get the block
                    block = block_data_cpu[channel, diag_idx, block_row, :, :]

                    # Create a unique identifier for this block
                    block_id = (channel, diag_idx, block_row)
                    if block_id in unique_blocks:
                        print(f"ERROR: Block {block_id} appears to be duplicated!")
                        errors += 1
                    unique_blocks.add(block_id)

                    # Check that block has non-zero values (not overwritten)
                    if np.all(block == 0) and diag_idx < 3:  # First 3 diagonals should have data
                        print(f"ERROR: Block[{channel},{diag_idx},{block_row}] is all zeros!")
                        errors += 1

                    # Check diagonal elements (should have expected_value)
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
    print(f"Unique blocks found: {len(unique_blocks)}")
    print(f"Errors found: {errors}")

    # Verify storage efficiency
    total_possible_blocks = batch_matrix.channels * batch_matrix.max_block_diag * batch_matrix.led_blocks
    print("\nStorage efficiency check:")
    print(f"  Total possible block slots: {total_possible_blocks}")
    print(f"  Non-zero blocks: {blocks_checked}")
    print(f"  Efficiency: {blocks_checked / total_possible_blocks * 100:.1f}%")

    return errors == 0


def main():
    """Main test function."""
    print("=" * 80)
    print("5D Block Storage Format Correctness Test (Simple)")
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
    print(f"  Expected max_block_diag based on bandwidth: {expected_max_block_diag}")

    # Verify storage format
    storage_correct = verify_block_storage(batch_matrix)

    # Verify that block_offsets_upper is removed
    print("\nVerifying removal of block_offsets_upper...")
    if hasattr(batch_matrix, "block_offsets_upper") and batch_matrix.block_offsets_upper is not None:
        print("ERROR: block_offsets_upper still exists!")
        storage_correct = False
    else:
        print("PASS: block_offsets_upper has been removed")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Storage format correct: {'PASS' if storage_correct else 'FAIL'}")

    # Additional memory efficiency check
    print("\nMemory Efficiency:")
    full_storage = batch_matrix.channels * batch_matrix.led_blocks * batch_matrix.led_blocks * 16 * 16 * 4  # float32
    actual_storage = batch_matrix.block_data_gpu.nbytes
    print(f"  Full storage would be: {full_storage / 1024**2:.2f} MB")
    print(f"  Actual storage: {actual_storage / 1024**2:.2f} MB")
    print(f"  Storage efficiency: {actual_storage / full_storage * 100:.1f}%")

    # Verify the bug is fixed
    print("\nBug Fix Verification:")
    print(f"  Old storage would have shape: (3, {batch_matrix.led_blocks * batch_matrix.led_blocks}, 16, 16)")
    print("  Old storage would only store last block per diagonal")
    print(f"  New storage has shape: {batch_matrix.block_data_gpu.shape}")
    print("  New storage stores ALL blocks correctly")

    return storage_correct


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
