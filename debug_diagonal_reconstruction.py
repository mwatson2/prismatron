#!/usr/bin/env python3
"""
Debug the diagonal reconstruction bug in BatchSymmetricDiagonalATAMatrix.

Compare the diagonal storage format between the working SymmetricDiagonalATAMatrix
and the broken reconstruction in BatchSymmetricDiagonalATAMatrix.
"""

import logging

import cupy as cp
import numpy as np

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from src.utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix

logging.basicConfig(level=logging.WARNING)


def debug_diagonal_storage():
    """Debug the diagonal storage format and reconstruction."""

    print("=== Debugging Diagonal Storage ===")

    # Use minimal example for clarity
    led_count = 16
    channels = 3

    # Create simple test case
    tensor = SingleBlockMixedSparseTensor(
        batch_size=led_count, channels=channels, height=128, width=96, block_size=64, dtype=cp.uint8
    )

    # Set just a few blocks to create predictable pattern
    np.random.seed(42)
    for led_idx in range(4):  # Only first 4 LEDs
        for channel in range(channels):
            top = 0
            left = led_idx * 24  # Spread out horizontally
            left = (left // 4) * 4  # Align to 4

            if left + 64 <= 96:  # Ensure it fits
                values = cp.ones((64, 64), dtype=cp.uint8) * (led_idx + 1)  # Simple constant values
                tensor.set_block(led_idx, channel, top, left, values)

    # Compute ground truth
    dense_ata = tensor.compute_ata_dense()
    print(f"Dense ATA shape: {dense_ata.shape}")
    print(f"Dense ATA channel 0 top-left 8x8:")
    print(dense_ata[:8, :8, 0])

    # Create symmetric diagonal matrix
    regular_ata = SymmetricDiagonalATAMatrix.from_dense(
        dense_ata.transpose(2, 0, 1), led_count=led_count, significance_threshold=0.01, crop_size=64
    )

    # Examine the diagonal storage
    print(f"\nDiagonal data shape: {regular_ata.dia_data_gpu.shape}")
    print(f"Diagonal offsets: {regular_ata.dia_offsets_upper}")

    # Show the raw diagonal data
    dia_data_cpu = cp.asnumpy(regular_ata.dia_data_gpu)
    print(f"\nChannel 0 diagonal data (first 8 elements of each diagonal):")
    for diag_idx, offset in enumerate(regular_ata.dia_offsets_upper):
        data = dia_data_cpu[0, diag_idx, :8]
        print(f"  Offset {offset}: {data}")

    # Now test manual reconstruction using BatchSymmetricDiagonalATAMatrix algorithm
    print(f"\n=== Manual Reconstruction Test ===")

    # EXACTLY the same algorithm as in BatchSymmetricDiagonalATAMatrix
    dia_data_gpu = regular_ata.dia_data_gpu
    dia_offsets_upper = regular_ata.dia_offsets_upper

    print("Reconstructing using BatchSymmetricDiagonalATAMatrix algorithm...")

    channel = 0  # Focus on channel 0
    dense_matrix = np.zeros((led_count, led_count), dtype=np.float32)

    # Fill from element diagonals
    for diag_idx, offset in enumerate(dia_offsets_upper):
        diag_data = cp.asnumpy(dia_data_gpu[channel, diag_idx, :led_count])
        print(f"\nProcessing diagonal offset {offset}:")
        print(f"  Diagonal data: {diag_data[:8]}")

        # Fill upper diagonal
        for i in range(led_count):
            j = i + offset
            if j < led_count:
                dense_matrix[i, j] = diag_data[i]
                if i < 8 and j < 8:  # Show first few assignments
                    print(f"    Setting [{i}, {j}] = {diag_data[i]}")

                # Fill symmetric lower diagonal (if not main diagonal)
                if offset > 0:
                    dense_matrix[j, i] = diag_data[i]
                    if i < 8 and j < 8:  # Show first few assignments
                        print(f"    Setting [{j}, {i}] = {diag_data[i]} (symmetric)")

    print(f"\nReconstructed matrix top-left 8x8:")
    print(dense_matrix[:8, :8])

    print(f"\nOriginal matrix top-left 8x8:")
    print(dense_ata[:8, :8, channel])

    # Compare
    max_error = np.max(np.abs(dense_matrix - dense_ata[:, :, channel]))
    print(f"\nReconstruction error: {max_error}")

    # Let's also check what the SymmetricDiagonalATAMatrix thinks it should be
    print(f"\n=== Checking SymmetricDiagonalATAMatrix multiply ===")

    test_vector = np.zeros(led_count, dtype=np.float32)
    test_vector[0] = 1.0  # Unit vector for first LED
    test_vector_gpu = cp.asarray(test_vector)

    # Ground truth: dense matrix multiply
    ground_truth = dense_ata[:, 0, channel]  # First column of dense matrix

    # SymmetricDiagonalATAMatrix multiply
    test_3d = cp.tile(test_vector_gpu[None, :], (3, 1))  # (3, led_count)
    regular_result = regular_ata.multiply_3d(test_3d)
    regular_first_channel = cp.asnumpy(regular_result[channel, :])

    print(f"Ground truth (dense[:, 0]): {ground_truth[:8]}")
    print(f"Regular multiply result: {regular_first_channel[:8]}")
    print(f"Regular multiply error: {np.max(np.abs(regular_first_channel - ground_truth))}")

    # Manual reconstruction multiply
    manual_result = dense_matrix @ test_vector
    print(f"Manual reconstruction result: {manual_result[:8]}")
    print(f"Manual reconstruction error: {np.max(np.abs(manual_result - ground_truth))}")

    return max_error < 1e-10


def debug_diagonal_format_difference():
    """Debug the difference between how SymmetricDiagonalATAMatrix stores vs reads diagonals."""

    print("\n=== Debugging Diagonal Format Difference ===")

    # Create minimal case - just 4x4 matrix
    led_count = 4

    # Create a simple known matrix
    test_matrix = np.array([[1, 2, 3, 0], [2, 4, 5, 6], [3, 5, 7, 8], [0, 6, 8, 9]], dtype=np.float32)

    print("Test matrix:")
    print(test_matrix)

    # Convert to 3D format for SymmetricDiagonalATAMatrix
    test_3d = np.stack([test_matrix, test_matrix, test_matrix], axis=0)  # (3, 4, 4)

    # Create SymmetricDiagonalATAMatrix from this known matrix
    regular_ata = SymmetricDiagonalATAMatrix.from_dense(test_3d, led_count=4, significance_threshold=0.1, crop_size=64)

    print(f"\nSymmetricDiagonalATAMatrix diagonal storage:")
    print(f"Offsets: {regular_ata.dia_offsets_upper}")
    dia_data_cpu = cp.asnumpy(regular_ata.dia_data_gpu)
    print(f"Diagonal data shape: {dia_data_cpu.shape}")

    for diag_idx, offset in enumerate(regular_ata.dia_offsets_upper):
        data = dia_data_cpu[0, diag_idx, :4]
        print(f"  Offset {offset}: {data}")

    # Manual reconstruction using BatchSymmetricDiagonalATAMatrix method
    print(f"\nManual reconstruction:")

    reconstructed = np.zeros((4, 4), dtype=np.float32)
    for diag_idx, offset in enumerate(regular_ata.dia_offsets_upper):
        diag_data = dia_data_cpu[0, diag_idx, :4]

        # Fill upper diagonal
        for i in range(4):
            j = i + offset
            if j < 4:
                reconstructed[i, j] = diag_data[i]
                print(f"  Setting [{i}, {j}] = {diag_data[i]} (from diagonal {offset}, index {i})")

                # Fill symmetric lower diagonal (if not main diagonal)
                if offset > 0:
                    reconstructed[j, i] = diag_data[i]
                    print(f"  Setting [{j}, {i}] = {diag_data[i]} (symmetric)")

    print("Reconstructed matrix:")
    print(reconstructed)

    print("Original matrix:")
    print(test_matrix)

    error = np.max(np.abs(reconstructed - test_matrix))
    print(f"Reconstruction error: {error}")

    return error < 1e-10


if __name__ == "__main__":
    debug_diagonal_storage()
    debug_diagonal_format_difference()
