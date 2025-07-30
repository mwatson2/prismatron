#!/usr/bin/env python3
"""
Debug the ATA computation to find where DIA and Dense differ.
"""

import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def debug_ata_computation():
    """Debug the ATA computation process step by step."""

    # Load the original capture data
    logger.info("Loading capture data...")
    data = np.load("diffusion_patterns/capture-0728-01.npz", allow_pickle=True)

    # Load mixed tensor
    mixed_tensor_data = data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_data)

    logger.info(f"Mixed tensor: {mixed_tensor.image_height}x{mixed_tensor.image_width}, {mixed_tensor.led_count} LEDs")

    # Build CSC matrix (same as used in both DIA and Dense)
    logger.info("Building CSC matrix...")
    csc_matrix = mixed_tensor.to_csc_matrix()

    logger.info(f"CSC matrix: {csc_matrix.shape}, {csc_matrix.nnz} non-zeros")

    # Test with just first channel and small subset
    channel = 0
    led_count = mixed_tensor.led_count  # Full LED count

    start_col = channel * led_count
    end_col = (channel + 1) * led_count

    logger.info(f"Extracting channel {channel}: columns {start_col}:{end_col}")

    # Method 1: DIA approach (what works)
    logger.info("\n=== Method 1: DIA Approach (Direct CSC A^T @ A) ===")

    # Convert to CuPy and extract channel
    csc_gpu = cp.sparse.csc_matrix(csc_matrix)
    A_channel_gpu = csc_gpu[:, start_col:end_col]

    # Convert to CPU CSR and compute A^T @ A
    A_channel_csr = A_channel_gpu.tocsr()
    A_channel_scipy = sp.csr_matrix(
        (cp.asnumpy(A_channel_csr.data), cp.asnumpy(A_channel_csr.indices), cp.asnumpy(A_channel_csr.indptr)),
        shape=A_channel_csr.shape,
    )

    logger.info("Computing A^T @ A using scipy...")
    ata_dia = A_channel_scipy.T @ A_channel_scipy

    logger.info(f"DIA result: {ata_dia.shape}, {ata_dia.nnz} non-zeros")
    logger.info(
        f"DIA stats: min={ata_dia.data.min():.6e}, max={ata_dia.data.max():.6e}, mean={ata_dia.data.mean():.6e}"
    )

    # Convert to dense for comparison
    ata_dia_dense = ata_dia.toarray()

    # Method 2: Dense approach (what we implemented)
    logger.info("\n=== Method 2: Dense Approach (Current Implementation) ===")

    # Same as in dense_ata_matrix.py
    A_channel_csr2 = A_channel_gpu.tocsr()
    A_channel_scipy2 = sp.csr_matrix(
        (cp.asnumpy(A_channel_csr2.data), cp.asnumpy(A_channel_csr2.indices), cp.asnumpy(A_channel_csr2.indptr)),
        shape=A_channel_csr2.shape,
    )

    ata_dense_sparse = A_channel_scipy2.T @ A_channel_scipy2
    ata_dense_result = ata_dense_sparse.toarray().astype(np.float32)

    logger.info(f"Dense result: {ata_dense_result.shape}")
    logger.info(
        f"Dense stats: min={ata_dense_result.min():.6e}, max={ata_dense_result.max():.6e}, mean={ata_dense_result.mean():.6e}"
    )

    # Method 3: Verify matrices are identical
    logger.info("\n=== Method 3: Direct Comparison ===")

    # Compare the sparse results
    diff_sparse = (ata_dia - ata_dense_sparse).data
    if len(diff_sparse) == 0:
        logger.info("Sparse matrices are identical!")
    else:
        logger.info(f"Sparse difference: max={np.abs(diff_sparse).max():.6e}")

    # Compare dense results
    diff_dense = ata_dia_dense - ata_dense_result
    logger.info(f"Dense difference: max={np.abs(diff_dense).max():.6e}, mean={np.abs(diff_dense).mean():.6e}")

    # Sample some values
    print("\nSample comparisons:")
    sample_indices = [(0, 0), (0, 1), (1, 1), (100, 100)]
    for i, j in sample_indices:
        dia_val = ata_dia_dense[i, j]
        dense_val = ata_dense_result[i, j]
        print(f"  [{i:3d}, {j:3d}]: DIA={dia_val:.6e}, Dense={dense_val:.6e}, Diff={dia_val-dense_val:.6e}")

    # Check if scaling is the issue
    logger.info("\n=== Scaling Analysis ===")
    scaling_factor = 0.065705  # From logs
    ata_scaling = scaling_factor**2
    logger.info(f"Expected ATA scaling: {ata_scaling:.6f}")

    # Apply scaling to see final values
    ata_dia_scaled = ata_dia_dense * ata_scaling
    ata_dense_scaled = ata_dense_result * ata_scaling

    logger.info(
        f"DIA after scaling: min={ata_dia_scaled.min():.6e}, max={ata_dia_scaled.max():.6e}, mean={ata_dia_scaled.mean():.6e}"
    )
    logger.info(
        f"Dense after scaling: min={ata_dense_scaled.min():.6e}, max={ata_dense_scaled.max():.6e}, mean={ata_dense_scaled.mean():.6e}"
    )


if __name__ == "__main__":
    debug_ata_computation()
