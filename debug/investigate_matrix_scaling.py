#!/usr/bin/env python3
"""
Investigate Matrix Scaling in Synthetic Pattern Generation.

This script investigates how the synthetic pattern generation tool creates
diffusion matrices and ATA matrices to understand the correct scaling.
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
from tools.generate_synthetic_patterns import SyntheticPatternGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_synthetic_pattern_scaling():
    """Test how synthetic patterns are scaled in different scenarios."""

    # Test 1: Generate patterns with float32 format
    logger.info("=== Test 1: Float32 Synthetic Patterns ===")

    generator_fp32 = SyntheticPatternGenerator(
        frame_width=800, frame_height=480, seed=42, block_size=64, use_uint8=False  # Float32 format
    )

    # Generate a small pattern set
    diffusion_matrix_fp32, led_mapping = generator_fp32.generate_sparse_patterns_chunked(led_count=100, chunk_size=50)

    sparse_csc_fp32 = diffusion_matrix_fp32.to_csc_matrix()
    logger.info(f"Float32 diffusion matrix: shape={sparse_csc_fp32.shape}, nnz={sparse_csc_fp32.nnz}")
    logger.info(f"Float32 diffusion data range: [{sparse_csc_fp32.data.min():.6f}, {sparse_csc_fp32.data.max():.6f}]")

    # Generate DenseATAMatrix from float32 diffusion matrix
    dense_ata_fp32 = DenseATAMatrix(led_count=100, channels=3, storage_dtype=cp.float32, output_dtype=cp.float32)
    dense_ata_fp32.build_from_diffusion_matrix(sparse_csc_fp32)

    # Test multiply_vector
    test_values = cp.ones((3, 100), dtype=cp.float32) * 0.5
    result_fp32 = dense_ata_fp32.multiply_vector(test_values)
    logger.info(f"Float32 ATA result range: [{cp.min(result_fp32):.6f}, {cp.max(result_fp32):.6f}]")

    # Test 2: Generate patterns with uint8 format
    logger.info("\n=== Test 2: Uint8 Synthetic Patterns ===")

    generator_uint8 = SyntheticPatternGenerator(
        frame_width=800, frame_height=480, seed=42, block_size=64, use_uint8=True  # Uint8 format
    )

    # Generate the same pattern set
    diffusion_matrix_uint8, led_mapping = generator_uint8.generate_sparse_patterns_chunked(led_count=100, chunk_size=50)

    sparse_csc_uint8 = diffusion_matrix_uint8.to_csc_matrix()
    logger.info(f"Uint8 diffusion matrix: shape={sparse_csc_uint8.shape}, nnz={sparse_csc_uint8.nnz}")
    logger.info(f"Uint8 diffusion data range: [{sparse_csc_uint8.data.min():.6f}, {sparse_csc_uint8.data.max():.6f}]")

    # The uint8 diffusion matrix should have values in [0, 1] range because the
    # sparse matrix is created with float32 values, but the mixed tensor stores uint8 values

    # Generate DenseATAMatrix from uint8 diffusion matrix
    dense_ata_uint8 = DenseATAMatrix(led_count=100, channels=3, storage_dtype=cp.float32, output_dtype=cp.float32)
    dense_ata_uint8.build_from_diffusion_matrix(sparse_csc_uint8)

    # Test multiply_vector
    result_uint8 = dense_ata_uint8.multiply_vector(test_values)
    logger.info(f"Uint8 ATA result range: [{cp.min(result_uint8):.6f}, {cp.max(result_uint8):.6f}]")

    # Test 3: Check the mixed tensor values directly
    logger.info("\n=== Test 3: Mixed Tensor Values ===")

    # Load the mixed tensors
    mixed_tensor_fp32 = generator_fp32._generate_mixed_tensor_format(sparse_csc_fp32)
    mixed_tensor_uint8 = generator_uint8._generate_mixed_tensor_format(sparse_csc_uint8)

    logger.info(f"Float32 mixed tensor dtype: {mixed_tensor_fp32.dtype}")
    logger.info(f"Uint8 mixed tensor dtype: {mixed_tensor_uint8.dtype}")

    # Sample some block values
    if mixed_tensor_fp32.sparse_values is not None:
        fp32_block_sample = mixed_tensor_fp32.sparse_values[0, 0]  # First channel, first LED
        logger.info(f"Float32 block sample range: [{cp.min(fp32_block_sample):.6f}, {cp.max(fp32_block_sample):.6f}]")

    if mixed_tensor_uint8.sparse_values is not None:
        uint8_block_sample = mixed_tensor_uint8.sparse_values[0, 0]  # First channel, first LED
        logger.info(f"Uint8 block sample range: [{cp.min(uint8_block_sample):.6f}, {cp.max(uint8_block_sample):.6f}]")
        logger.info(f"Uint8 block sample type: {type(cp.asnumpy(uint8_block_sample).flat[0])}")

    # Test 4: Check what happens when we do A^T @ b calculation
    logger.info("\n=== Test 4: A^T @ b Calculation ===")

    # Create a test target frame
    test_frame = cp.ones((3, 480, 800), dtype=cp.uint8) * 128  # Mid-gray

    # Do A^T @ b with uint8 mixed tensor (this uses the kernel that applies scaling)
    result_atb_uint8 = mixed_tensor_uint8.transpose_dot_product_3d(test_frame, planar_output=True)
    logger.info(f"Uint8 A^T @ b result range: [{cp.min(result_atb_uint8):.6f}, {cp.max(result_atb_uint8):.6f}]")

    # Do A^T @ b with float32 mixed tensor
    test_frame_fp32 = test_frame.astype(cp.float32) / 255.0  # Normalize to [0,1]
    result_atb_fp32 = mixed_tensor_fp32.transpose_dot_product_3d(test_frame_fp32, planar_output=True)
    logger.info(f"Float32 A^T @ b result range: [{cp.min(result_atb_fp32):.6f}, {cp.max(result_atb_fp32):.6f}]")

    # Compare scaling
    ratio = cp.max(result_atb_uint8) / cp.max(result_atb_fp32) if cp.max(result_atb_fp32) > 0 else 0
    logger.info(f"Uint8 vs Float32 A^T @ b ratio: {ratio:.2f}")


if __name__ == "__main__":
    test_synthetic_pattern_scaling()
