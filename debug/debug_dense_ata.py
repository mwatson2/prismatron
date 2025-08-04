#!/usr/bin/env python3
"""
Debug Dense ATA Matrix Issue.

This script tests the DenseATAMatrix multiply_vector method to identify
why optimization is producing all-zero LED values.
"""

import logging

# Set up path
import sys
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.dense_ata_matrix import DenseATAMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_dense_ata_multiply():
    """Test dense ATA matrix multiply_vector method."""

    # Load pattern file
    pattern_file = "diffusion_patterns/capture-0728-01-dense_fixed.npz"
    if not Path(pattern_file).exists():
        logger.error(f"Pattern file not found: {pattern_file}")
        return

    logger.info(f"Loading pattern file: {pattern_file}")
    data = np.load(pattern_file, allow_pickle=True)

    # Check what's in the file
    logger.info(f"Available keys: {list(data.keys())}")

    # Load dense ATA matrix if available
    if "dense_ata_matrix" not in data:
        logger.error("No dense_ata_matrix found in pattern file")
        return

    dense_ata_data = data["dense_ata_matrix"].item()
    logger.info(f"Dense ATA matrix format: {dense_ata_data.get('format', 'unknown')}")

    # Load the dense ATA matrix
    dense_ata = DenseATAMatrix.from_dict(dense_ata_data)
    logger.info(f"Loaded dense ATA matrix: {dense_ata}")

    # Create test LED values
    led_count = dense_ata.led_count
    logger.info(f"LED count: {led_count}")

    # Test with simple non-zero values
    test_values = cp.ones((3, led_count), dtype=cp.float32) * 0.5
    logger.info(
        f"Test values shape: {test_values.shape}, range: [{cp.min(test_values):.3f}, {cp.max(test_values):.3f}]"
    )

    # Test multiply_vector
    logger.info("Testing dense ATA multiply_vector...")
    try:
        result = dense_ata.multiply_vector(test_values)
        logger.info(f"Result shape: {result.shape}")
        logger.info(f"Result range: [{cp.min(result):.6f}, {cp.max(result):.6f}]")
        logger.info(f"Result mean: {cp.mean(result):.6f}")
        logger.info(f"Result std: {cp.std(result):.6f}")

        # Check if all zeros
        if cp.max(cp.abs(result)) == 0:
            logger.error("PROBLEM: multiply_vector returned all zeros!")

            # Check the matrices themselves
            if dense_ata.dense_matrices_gpu is None:
                logger.info("Dense matrices not on GPU, loading...")
                dense_ata.dense_matrices_gpu = cp.asarray(dense_ata.dense_matrices_cpu, dtype=dense_ata.storage_dtype)

            for channel in range(3):
                matrix = dense_ata.dense_matrices_gpu[channel]
                matrix_min = cp.min(matrix)
                matrix_max = cp.max(matrix)
                matrix_mean = cp.mean(matrix)
                matrix_nonzero = cp.count_nonzero(matrix)
                logger.info(
                    f"Channel {channel} matrix: range=[{matrix_min:.6f}, {matrix_max:.6f}], mean={matrix_mean:.6f}, nonzero={matrix_nonzero}/{matrix.size}"
                )

                if matrix_max == 0:
                    logger.error(f"Channel {channel} matrix is all zeros!")
        else:
            logger.info("SUCCESS: multiply_vector returned non-zero values")

    except Exception as e:
        logger.error(f"Error in multiply_vector: {e}")
        import traceback

        traceback.print_exc()


def test_ata_values():
    """Test the actual ATA matrix values to understand the scaling issue."""

    pattern_file = "diffusion_patterns/capture-0728-01-dense_fixed.npz"
    if not Path(pattern_file).exists():
        logger.error(f"Pattern file not found: {pattern_file}")
        return

    logger.info(f"Loading pattern file: {pattern_file}")
    data = np.load(pattern_file, allow_pickle=True)

    # Load dense ATA matrix if available
    if "dense_ata_matrix" not in data:
        logger.error("No dense_ata_matrix found in pattern file")
        return

    dense_ata_data = data["dense_ata_matrix"].item()
    dense_ata = DenseATAMatrix.from_dict(dense_ata_data)
    logger.info(f"Loaded dense ATA matrix: {dense_ata}")

    # Examine the actual matrix values
    logger.info("Analyzing ATA matrix values...")

    if dense_ata.dense_matrices_gpu is None:
        dense_ata.dense_matrices_gpu = cp.asarray(dense_ata.dense_matrices_cpu, dtype=dense_ata.storage_dtype)

    for channel in range(3):
        matrix = dense_ata.dense_matrices_gpu[channel]

        # Basic statistics
        matrix_min = float(cp.min(matrix))
        matrix_max = float(cp.max(matrix))
        matrix_mean = float(cp.mean(matrix))
        matrix_std = float(cp.std(matrix))
        matrix_median = float(cp.median(matrix))
        matrix_nonzero = int(cp.count_nonzero(matrix))

        logger.info(f"Channel {channel} ATA matrix statistics:")
        logger.info(f"  Range: [{matrix_min:.3e}, {matrix_max:.3e}]")
        logger.info(f"  Mean: {matrix_mean:.3e}, Std: {matrix_std:.3e}")
        logger.info(f"  Median: {matrix_median:.3e}")
        logger.info(f"  Non-zero elements: {matrix_nonzero}/{matrix.size} ({100*matrix_nonzero/matrix.size:.1f}%)")

        # Check diagonal values (should be largest)
        diagonal = cp.diag(matrix)
        diag_min = float(cp.min(diagonal))
        diag_max = float(cp.max(diagonal))
        diag_mean = float(cp.mean(diagonal))

        logger.info(f"  Diagonal range: [{diag_min:.3e}, {diag_max:.3e}], mean: {diag_mean:.3e}")

        # Sample some off-diagonal values for comparison
        matrix_copy = matrix.copy()
        cp.fill_diagonal(matrix_copy, 0)  # Remove diagonal
        if cp.count_nonzero(matrix_copy) > 0:
            off_diag_max = float(cp.max(matrix_copy))
            off_diag_mean = float(cp.mean(matrix_copy[matrix_copy != 0]))
            logger.info(f"  Off-diagonal max: {off_diag_max:.3e}, mean (non-zero): {off_diag_mean:.3e}")

        # Check for any abnormally large values
        large_values = matrix > 1e6
        if cp.any(large_values):
            num_large = int(cp.sum(large_values))
            logger.warning(f"  Found {num_large} values > 1e6! This suggests scaling issues.")

        # Compare with typical expected ranges
        if matrix_max > 1e6:
            logger.error(f"  PROBLEM: Matrix values are too large (max={matrix_max:.3e})")
            logger.error("  Expected ATA matrices to have values roughly in [0, 1000] range")
            logger.error("  Large values will cause LED optimization to fail")

        if matrix_max < 1e-6:
            logger.error(f"  PROBLEM: Matrix values are too small (max={matrix_max:.3e})")
            logger.error("  This suggests the matrix is nearly singular")


if __name__ == "__main__":
    logger.info("=== Testing Dense ATA Matrix ===")
    test_dense_ata_multiply()

    logger.info("\n=== Analyzing ATA Matrix Values ===")
    test_ata_values()
