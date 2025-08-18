#!/usr/bin/env python3
"""
Debug the conversion process from SymmetricDiagonalATAMatrix to BatchSymmetricDiagonalATAMatrix.

This script validates the element diagonal to block diagonal conversion step by step,
comparing results against known-good references.
"""

import logging

import cupy as cp
import numpy as np

from src.utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from src.utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix

logging.basicConfig(level=logging.WARNING)  # Reduce verbosity


def test_conversion_accuracy():
    """Test the accuracy of the element diagonal to block diagonal conversion."""

    print("=== Testing Conversion Process ===")

    # Use small aligned problem for detailed analysis
    led_count = 16  # Multiple of 16 for tensor cores
    channels = 3
    height, width = 128, 96
    block_size = 64

    print(f"LEDs: {led_count}, Image: {height}x{width}")

    # Create sparse tensor with fixed seed for reproducibility
    tensor = SingleBlockMixedSparseTensor(
        batch_size=led_count, channels=channels, height=height, width=width, block_size=block_size, dtype=cp.uint8
    )

    # Set identical blocks with fixed random seed
    np.random.seed(42)
    rows = int(np.sqrt(led_count))
    cols = int(np.ceil(led_count / rows))

    for led_idx in range(led_count):
        row = led_idx // cols
        col = led_idx % cols

        for channel in range(channels):
            top = row * (height // rows)
            left = col * (width // cols)
            left = (left // 4) * 4  # Align to 4

            if top + block_size <= height and left + block_size <= width:
                values = cp.random.randint(0, 256, (block_size, block_size), dtype=cp.uint8)
                tensor.set_block(led_idx, channel, top, left, values)

    # Compute dense ATA as ground truth
    print("\nComputing ground truth dense ATA...")
    dense_ata = tensor.compute_ata_dense()  # (led_count, led_count, 3)

    print(f"Dense ATA shape: {dense_ata.shape}")
    print(f"Dense ATA range: [{dense_ata.min():.6f}, {dense_ata.max():.6f}]")

    # Create SymmetricDiagonalATAMatrix
    print("\nCreating SymmetricDiagonalATAMatrix...")
    regular_ata = SymmetricDiagonalATAMatrix.from_dense(
        dense_ata.transpose(2, 0, 1), led_count=led_count, significance_threshold=0.01, crop_size=block_size
    )

    # Test single multiplication first
    print("\nTesting SymmetricDiagonalATAMatrix single multiplication...")
    np.random.seed(123)
    test_input = np.random.randn(3, led_count).astype(np.float32)
    test_input_gpu = cp.asarray(test_input)

    # Ground truth: direct dense multiplication
    ground_truth_result = np.zeros((3, led_count), dtype=np.float32)
    for channel in range(3):
        ground_truth_result[channel] = dense_ata[:, :, channel] @ test_input[channel]

    # Regular matrix result
    regular_result = regular_ata.multiply_3d(test_input_gpu)
    regular_cpu = cp.asnumpy(regular_result)

    # Compare regular vs ground truth
    max_diff_regular = np.max(np.abs(regular_cpu - ground_truth_result))
    rms_diff_regular = np.sqrt(np.mean((regular_cpu - ground_truth_result) ** 2))
    rms_truth = np.sqrt(np.mean(ground_truth_result**2))

    print("Regular vs Ground Truth:")
    print(f"  Max difference: {max_diff_regular:.8f}")
    print(f"  RMS difference: {rms_diff_regular:.8f}")
    print(f"  Relative error: {rms_diff_regular/rms_truth*100:.6f}%")

    # Now test the conversion process
    print("\nTesting conversion to BatchSymmetricDiagonalATAMatrix...")
    batch_ata = BatchSymmetricDiagonalATAMatrix.from_symmetric_diagonal_matrix(regular_ata, batch_size=8)

    # Test batch multiplication (single input repeated 8 times)
    batch_input = cp.tile(test_input_gpu[None, :, :], (8, 1, 1))  # (8, 3, led_count)
    batch_result = batch_ata.multiply_batch8_3d(batch_input)
    batch_cpu = cp.asnumpy(batch_result[0])  # First frame

    # Compare batch vs ground truth
    max_diff_batch = np.max(np.abs(batch_cpu - ground_truth_result))
    rms_diff_batch = np.sqrt(np.mean((batch_cpu - ground_truth_result) ** 2))

    print("Batch vs Ground Truth:")
    print(f"  Max difference: {max_diff_batch:.8f}")
    print(f"  RMS difference: {rms_diff_batch:.8f}")
    print(f"  Relative error: {rms_diff_batch/rms_truth*100:.6f}%")

    # Compare batch vs regular
    max_diff_batch_reg = np.max(np.abs(batch_cpu - regular_cpu))
    rms_diff_batch_reg = np.sqrt(np.mean((batch_cpu - regular_cpu) ** 2))
    rms_regular = np.sqrt(np.mean(regular_cpu**2))

    print("Batch vs Regular:")
    print(f"  Max difference: {max_diff_batch_reg:.8f}")
    print(f"  RMS difference: {rms_diff_batch_reg:.8f}")
    print(f"  Relative error: {rms_diff_batch_reg/rms_regular*100:.6f}%")

    # Detailed analysis of the conversion
    print("\n=== Conversion Analysis ===")

    # Extract and verify the dense matrix reconstruction
    print("\nAnalyzing dense matrix reconstruction...")

    # Get diagonal data from regular matrix
    dia_data_gpu = regular_ata.dia_data_gpu
    dia_offsets_upper = regular_ata.dia_offsets_upper

    print(f"Diagonal data shape: {dia_data_gpu.shape}")
    print(f"Diagonal offsets: {dia_offsets_upper}")

    # Manually reconstruct dense matrix using the same algorithm as the conversion
    reconstructed_matrices = []
    for channel in range(channels):
        dense_matrix = np.zeros((led_count, led_count), dtype=np.float32)

        # Fill from element diagonals
        for diag_idx, offset in enumerate(dia_offsets_upper):
            diag_data = cp.asnumpy(dia_data_gpu[channel, diag_idx, :led_count])

            # Fill upper diagonal
            for i in range(led_count):
                j = i + offset
                if j < led_count:
                    dense_matrix[i, j] = diag_data[i]

                    # Fill symmetric lower diagonal (if not main diagonal)
                    if offset > 0:
                        dense_matrix[j, i] = diag_data[i]

        reconstructed_matrices.append(dense_matrix)

    # Compare reconstructed vs original dense matrix
    max_reconstruction_error = 0
    for channel in range(channels):
        error = np.max(np.abs(reconstructed_matrices[channel] - dense_ata[:, :, channel]))
        max_reconstruction_error = max(max_reconstruction_error, error)

        print(f"Channel {channel} reconstruction error: {error:.10f}")

    print(f"Maximum reconstruction error: {max_reconstruction_error:.10f}")

    # If reconstruction is perfect, the bug is in the block storage or WMMA kernel
    if max_reconstruction_error < 1e-10:
        print("✓ Dense matrix reconstruction is PERFECT")
        print("✗ Bug must be in block storage or WMMA kernel")
    else:
        print("✗ Dense matrix reconstruction has errors")
        print("✓ Found the conversion bug!")

    # Summary
    print("\n=== SUMMARY ===")
    regular_ok = rms_diff_regular / rms_truth < 0.001  # 0.1% tolerance
    batch_ok = rms_diff_batch / rms_truth < 0.001
    conversion_ok = max_reconstruction_error < 1e-10

    print(f"Regular matrix: {'✓ PASS' if regular_ok else '✗ FAIL'} ({rms_diff_regular/rms_truth*100:.6f}% error)")
    print(f"Batch matrix: {'✓ PASS' if batch_ok else '✗ FAIL'} ({rms_diff_batch/rms_truth*100:.6f}% error)")
    print(f"Dense reconstruction: {'✓ PASS' if conversion_ok else '✗ FAIL'} ({max_reconstruction_error:.2e} max error)")

    if conversion_ok and not batch_ok:
        print("Bug is in block storage or WMMA kernel implementation")
    elif not conversion_ok:
        print("Bug is in dense matrix reconstruction from diagonals")
    else:
        print("All components working correctly - investigate test setup")


if __name__ == "__main__":
    test_conversion_accuracy()
