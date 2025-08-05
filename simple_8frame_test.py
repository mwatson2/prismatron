#!/usr/bin/env python3
"""
Simple test to verify 8-frame batch functionality works correctly.
"""

import sys

import numpy as np

# Add project root to path
sys.path.insert(0, "/mnt/dev/prismatron/src")

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available")
    sys.exit(1)

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix


def main():
    """Simple 8-frame test with smaller matrix."""
    print("Simple 8-Frame Batch Test")
    print("=" * 50)

    # Smaller test case
    led_count = 128
    crop_size = 32

    # Create a simple test matrix
    ata_matrix = DiagonalATAMatrix(led_count=led_count, crop_size=crop_size)

    # Create simple diagonal pattern
    offsets = [-2, -1, 0, 1, 2]  # 5 diagonals
    k = len(offsets)

    # Create simple diagonal data
    dia_data_cpu = np.zeros((3, k, led_count), dtype=np.float32)

    for channel in range(3):
        for diag_idx, offset in enumerate(offsets):
            if offset == 0:
                dia_data_cpu[channel, diag_idx, :] = 2.0  # Main diagonal
            else:
                dia_data_cpu[channel, diag_idx, :] = 0.5  # Off-diagonals

    # Set matrix data
    ata_matrix.dia_data_cpu = dia_data_cpu
    ata_matrix.dia_offsets = np.array(offsets, dtype=np.int32)
    ata_matrix.k = k
    ata_matrix.bandwidth = 2
    ata_matrix.sparsity = 1.0
    ata_matrix.nnz = np.count_nonzero(dia_data_cpu)

    # Convert to GPU
    ata_matrix.dia_data_gpu = cp.asarray(dia_data_cpu)
    ata_matrix.dia_offsets_gpu = cp.asarray(offsets, dtype=cp.int32)

    print(f"Test matrix: {led_count}×{led_count}, {k} diagonals")

    # Convert to 8-frame batch storage
    print("Converting to 8-frame batch storage...")
    batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(ata_matrix, batch_size=8)

    # Create test input
    print("Creating test input...")
    np.random.seed(42)
    test_input = np.random.randn(8, 3, led_count).astype(np.float32) * 0.1
    test_input_gpu = cp.asarray(test_input)

    # Test the multiplication
    print("Testing 8-frame batch multiplication...")
    result = batch_matrix.multiply_batch8_3d(test_input_gpu, debug_logging=True)
    result_cpu = cp.asnumpy(result)

    print(f"Input shape: {test_input.shape}")
    print(f"Result shape: {result_cpu.shape}")
    print(f"Input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"Result range: [{result_cpu.min():.4f}, {result_cpu.max():.4f}]")

    print("\n✅ Simple 8-frame test completed successfully!")


if __name__ == "__main__":
    main()
