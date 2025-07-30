#!/usr/bin/env python3
"""
Debug tool to compare DIA and Dense ATA matrices element by element.
"""

import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.dense_ata_matrix import DenseATAMatrix
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compare_matrices_elementwise(file_path: str):
    """Compare DIA and Dense matrices element by element."""

    logger.info(f"Loading matrices from: {file_path}")
    data = np.load(file_path, allow_pickle=True)

    # Load both matrices
    dia_data = data["dia_matrix"].item()
    dense_data = data["dense_ata_matrix"].item()

    dia_matrix = DiagonalATAMatrix.from_dict(dia_data)
    dense_matrix = DenseATAMatrix.from_dict(dense_data)

    logger.info(f"DIA matrix: {dia_matrix.led_count} LEDs, k={dia_matrix.k} diagonals")
    logger.info(f"Dense matrix: {dense_matrix.led_count} LEDs, {dense_matrix.memory_mb:.1f}MB")

    # Compare basic properties
    print("\n=== Matrix Properties ===")
    print(f"LED count - DIA: {dia_matrix.led_count}, Dense: {dense_matrix.led_count}")
    print(f"Channels - DIA: {dia_matrix.channels}, Dense: {dense_matrix.channels}")
    print(f"Storage dtype - DIA: {dia_matrix.storage_dtype}, Dense: {dense_matrix.storage_dtype}")

    # Compare raw matrix data for first channel
    channel = 0
    print(f"\n=== Channel {channel} Element Comparison ===")

    # Get DIA matrix as dense for comparison
    dia_scipy = dia_matrix.get_channel_dia_matrix(channel)
    dia_dense = dia_scipy.toarray()

    # Get dense matrix data
    dense_cpu = dense_matrix.dense_matrices_cpu[channel]

    print(f"DIA dense shape: {dia_dense.shape}")
    print(f"Dense matrix shape: {dense_cpu.shape}")

    # Sample some elements for comparison
    print("\n=== Sample Element Values (before any operations) ===")
    sample_indices = [(0, 0), (0, 1), (1, 0), (1, 1), (100, 100), (500, 500)]

    for i, j in sample_indices:
        if i < dia_dense.shape[0] and j < dia_dense.shape[1]:
            dia_val = dia_dense[i, j]
            dense_val = dense_cpu[i, j]
            ratio = dia_val / dense_val if dense_val != 0 else "inf"
            print(f"  [{i:3d}, {j:3d}]: DIA={dia_val:12.6e}, Dense={dense_val:12.6e}, Ratio={ratio}")

    # Check matrix statistics
    print(f"\n=== Matrix Statistics ===")
    print(f"DIA matrix - min: {dia_dense.min():.6e}, max: {dia_dense.max():.6e}, mean: {dia_dense.mean():.6e}")
    print(f"Dense matrix - min: {dense_cpu.min():.6e}, max: {dense_cpu.max():.6e}, mean: {dense_cpu.mean():.6e}")

    # Check if it's a simple scaling factor
    nonzero_mask = (dia_dense != 0) & (dense_cpu != 0)
    if np.any(nonzero_mask):
        ratios = dia_dense[nonzero_mask] / dense_cpu[nonzero_mask]
        unique_ratios = np.unique(np.round(ratios, 10))
        print(f"\nRatio analysis (non-zero elements):")
        print(f"  Unique ratios count: {len(unique_ratios)}")
        if len(unique_ratios) <= 10:
            print(f"  Unique ratios: {unique_ratios}")
        else:
            print(f"  Sample ratios: {unique_ratios[:10]}")
        print(f"  Ratio range: {ratios.min():.6e} to {ratios.max():.6e}")
        print(f"  Ratio std: {ratios.std():.6e}")

    # Test with a simple vector multiplication
    print(f"\n=== Vector Multiplication Test ===")
    test_vector = cp.ones((3, dia_matrix.led_count), dtype=cp.float32) * 0.5

    # DIA result
    dia_result = dia_matrix.multiply_3d(test_vector)
    dia_result_cpu = cp.asnumpy(dia_result)

    # Dense result
    dense_result = dense_matrix.multiply_vector(test_vector)
    dense_result_cpu = cp.asnumpy(dense_result)

    print(f"Test vector: shape {test_vector.shape}, all values = 0.5")
    print(
        f"DIA result - min: {dia_result_cpu.min():.6e}, max: {dia_result_cpu.max():.6e}, mean: {dia_result_cpu.mean():.6e}"
    )
    print(
        f"Dense result - min: {dense_result_cpu.min():.6e}, max: {dense_result_cpu.max():.6e}, mean: {dense_result_cpu.mean():.6e}"
    )

    # Sample result values
    print(f"\nSample result values (channel 0):")
    for i in [0, 1, 10, 100]:
        if i < dia_result_cpu.shape[1]:
            dia_val = dia_result_cpu[0, i]
            dense_val = dense_result_cpu[0, i]
            ratio = dia_val / dense_val if dense_val != 0 else "inf"
            print(f"  LED {i:3d}: DIA={dia_val:12.6e}, Dense={dense_val:12.6e}, Ratio={ratio}")


if __name__ == "__main__":
    compare_matrices_elementwise("diffusion_patterns/capture-0728-01-combined.npz")
