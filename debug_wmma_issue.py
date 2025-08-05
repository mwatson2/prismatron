#!/usr/bin/env python3
"""
Debug script to isolate WMMA kernel accuracy issues.
"""

import os
import sys

import cupy
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix


def debug_simple_case():
    """Debug the simplest case that shows high error."""
    print("=" * 60)
    print("DEBUG: Simple Tridiagonal Case")
    print("=" * 60)

    # Create a very simple 32x32 tridiagonal matrix
    led_count = 32
    channels = 3

    # Simple pattern: main diagonal = 2, off diagonal = 1
    offsets = np.array([0, 1], dtype=np.int32)
    dia_data = np.zeros((channels, 2, led_count), dtype=np.float32)

    # Use exact values to minimize floating point issues
    for ch in range(channels):
        dia_data[ch, 0, :] = 2.0  # Main diagonal
        dia_data[ch, 1, : led_count - 1] = 1.0  # Off diagonal

    # Create diagonal matrix
    diagonal_matrix = DiagonalATAMatrix(led_count, crop_size=1)
    diagonal_matrix.bandwidth = 2
    diagonal_matrix.dia_data_upper_gpu = cupy.asarray(dia_data)
    diagonal_matrix.dia_offsets_upper_gpu = cupy.asarray(offsets)
    diagonal_matrix.k = 2
    diagonal_matrix.sparsity = 0.9
    diagonal_matrix.nnz = led_count + (led_count - 1)
    diagonal_matrix.dia_offsets = offsets
    diagonal_matrix.dia_data_cpu = dia_data

    # Create batch matrix
    batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(diagonal_matrix, batch_size=16)

    # Create simple test vector - all ones for easy verification
    print("\nTesting with all-ones input vector...")
    test_input = np.ones((16, 3, 32), dtype=np.float32)
    test_input_gpu = cupy.asarray(test_input)

    # Expected result for tridiagonal [2, 1; 1, 2, 1; 1, 2, 1; ...; 1, 2]
    # For all-ones input:
    # result[0] = 2*1 + 1*1 = 3
    # result[i] = 1*1 + 2*1 + 1*1 = 4 (for i = 1 to 30)
    # result[31] = 1*1 + 2*1 = 3
    expected = np.full((16, 3, 32), 4.0, dtype=np.float32)
    expected[:, :, 0] = 3.0  # First element
    expected[:, :, 31] = 3.0  # Last element

    print("Expected result pattern:")
    print(f"  result[0] = {expected[0, 0, 0]} (edge)")
    print(f"  result[1] = {expected[0, 0, 1]} (interior)")
    print(f"  result[31] = {expected[0, 0, 31]} (edge)")

    # Compute using WMMA kernel
    wmma_result = batch_matrix.multiply_batch_3d(test_input_gpu)
    wmma_result_cpu = cupy.asnumpy(wmma_result)

    print("\nActual WMMA result:")
    print(f"  result[0] = {wmma_result_cpu[0, 0, 0]:.6f}")
    print(f"  result[1] = {wmma_result_cpu[0, 0, 1]:.6f}")
    print(f"  result[31] = {wmma_result_cpu[0, 0, 31]:.6f}")

    # Compare
    abs_diff = np.abs(expected - wmma_result_cpu)
    max_abs_diff = np.max(abs_diff)
    rel_diff = abs_diff / (np.abs(expected) + 1e-10)
    max_rel_diff = np.max(rel_diff)

    print(f"\nError analysis:")
    print(f"  Max absolute error: {max_abs_diff:.6f}")
    print(f"  Max relative error: {max_rel_diff:.6f} ({max_rel_diff*100:.2f}%)")

    # Find worst error location
    worst_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    batch_idx, ch, led_idx = worst_idx
    print(f"  Worst error at [{batch_idx}][{ch}][{led_idx}]:")
    print(f"    Expected: {expected[batch_idx, ch, led_idx]:.6f}")
    print(f"    Actual:   {wmma_result_cpu[batch_idx, ch, led_idx]:.6f}")
    print(f"    Diff:     {abs_diff[batch_idx, ch, led_idx]:.6f}")

    return max_abs_diff < 1e-3 and max_rel_diff < 0.01


def debug_storage_format():
    """Check if the 5D storage format is working correctly."""
    print("\n" + "=" * 60)
    print("DEBUG: 5D Storage Format")
    print("=" * 60)

    # Create simple matrix
    led_count = 32
    channels = 3
    offsets = np.array([0, 1], dtype=np.int32)
    dia_data = np.zeros((channels, 2, led_count), dtype=np.float32)

    for ch in range(channels):
        dia_data[ch, 0, :] = 2.0
        dia_data[ch, 1, : led_count - 1] = 1.0

    diagonal_matrix = DiagonalATAMatrix(led_count, crop_size=1)
    diagonal_matrix.bandwidth = 2
    diagonal_matrix.dia_data_upper_gpu = cupy.asarray(dia_data)
    diagonal_matrix.dia_offsets_upper_gpu = cupy.asarray(offsets)
    diagonal_matrix.k = 2
    diagonal_matrix.sparsity = 0.9
    diagonal_matrix.nnz = led_count + (led_count - 1)
    diagonal_matrix.dia_offsets = offsets
    diagonal_matrix.dia_data_cpu = dia_data

    batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(diagonal_matrix, batch_size=16)

    # Check 5D storage structure
    print(f"5D storage shape: {batch_matrix.block_data_gpu.shape}")
    print(f"Expected: (3, 2, 2, 16, 16)")
    print(f"Max block diagonal: {batch_matrix.max_block_diag}")
    print(f"LED blocks: {batch_matrix.led_blocks}")

    # Check specific blocks
    block_data_cpu = cupy.asnumpy(batch_matrix.block_data_gpu)

    print("\nChecking main diagonal block (0,0):")
    main_block = block_data_cpu[0, 0, 0, :, :]  # Channel 0, diag 0, block (0,0)
    print(f"  Block[0,0] top-left corner:\n{main_block[:4, :4]}")

    print("\nChecking off-diagonal block (0,1):")
    if batch_matrix.max_block_diag > 1:
        off_block = block_data_cpu[0, 1, 0, :, :]  # Channel 0, diag 1, block (0,1)
        print(f"  Block[0,1] top-left corner:\n{off_block[:4, :4]}")

    return True


def main():
    """Main debug function."""
    print("WMMA Kernel Debug Analysis")
    print("Investigating accuracy issues...")

    success = True

    if not debug_simple_case():
        success = False

    if not debug_storage_format():
        success = False

    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)

    if success:
        print("✓ Debug analysis completed - check results above")
    else:
        print("✗ Issues found in WMMA implementation")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
