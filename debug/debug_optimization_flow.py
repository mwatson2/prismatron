#!/usr/bin/env python3
"""
Debug Optimization Flow.

Debug the optimization step by step to see where it's failing.
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


def debug_optimization_step_by_step():
    """Debug each step of the optimization process."""

    # Use the corrected file
    pattern_file = "diffusion_patterns/capture-0728-01-dense_fixed.npz"

    if not Path(pattern_file).exists():
        logger.error(f"Fixed pattern file not found: {pattern_file}")
        return

    logger.info(f"Loading fixed pattern file: {pattern_file}")
    data = np.load(pattern_file, allow_pickle=True)

    # Load components
    mixed_tensor_data = data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_data)
    logger.info(f"Loaded mixed tensor: {mixed_tensor}")

    dense_ata_data = data["dense_ata_matrix"].item()
    dense_ata = DenseATAMatrix.from_dict(dense_ata_data)
    logger.info(f"Loaded dense ATA matrix: {dense_ata}")

    # Load ATA inverse for initialization
    ata_inverse = data["ata_inverse"]
    logger.info(f"Loaded ATA inverse: shape={ata_inverse.shape}")

    # Create a simple test frame
    test_frame = cp.ones((3, 480, 800), dtype=cp.uint8) * 128  # Mid-gray
    logger.info(f"Created test frame: shape={test_frame.shape}, dtype={test_frame.dtype}")

    # Step 1: Calculate A^T @ b
    logger.info("\n=== Step 1: Calculate A^T @ b ===")
    result_atb = mixed_tensor.transpose_dot_product_3d(test_frame, planar_output=True)
    logger.info(f"A^T @ b result: shape={result_atb.shape}, range=[{cp.min(result_atb):.6f}, {cp.max(result_atb):.6f}]")

    # Step 2: Initialize using ATA inverse
    logger.info("\n=== Step 2: Initialize using ATA inverse ===")
    ata_inverse_gpu = cp.asarray(ata_inverse)
    led_values_init_raw = cp.einsum("ijk,ik->ij", ata_inverse_gpu, result_atb)
    logger.info(
        f"Initial LED values (raw): shape={led_values_init_raw.shape}, range=[{cp.min(led_values_init_raw):.6f}, {cp.max(led_values_init_raw):.6f}]"
    )

    # Step 3: Clip to valid range
    led_values_init = cp.clip(led_values_init_raw, 0.0, 1.0)
    logger.info(f"Initial LED values (clipped): range=[{cp.min(led_values_init):.6f}, {cp.max(led_values_init):.6f}]")

    # Step 4: Test dense ATA multiply
    logger.info("\n=== Step 3: Test Dense ATA Multiply ===")
    ata_result = dense_ata.multiply_vector(led_values_init)
    logger.info(f"ATA @ x result: range=[{cp.min(ata_result):.6f}, {cp.max(ata_result):.6f}]")

    # Step 5: Compute gradient
    logger.info("\n=== Step 4: Compute Gradient ===")
    gradient = ata_result - result_atb
    logger.info(f"Gradient: range=[{cp.min(gradient):.6f}, {cp.max(gradient):.6f}]")
    logger.info(f"Gradient magnitude: {cp.sqrt(cp.sum(gradient**2)):.6f}")

    # Step 6: Compute step size
    logger.info("\n=== Step 5: Compute Step Size ===")
    g_dot_g = cp.sum(gradient * gradient)
    ata_gradient = dense_ata.multiply_vector(gradient)
    g_dot_ata_g = cp.sum(gradient * ata_gradient)

    logger.info(f"g^T @ g: {g_dot_g:.6f}")
    logger.info(f"g^T @ ATA @ g: {g_dot_ata_g:.6f}")

    if g_dot_ata_g > 0:
        step_size = float(0.9 * g_dot_g / g_dot_ata_g)
        logger.info(f"Step size: {step_size:.6f}")

        # Step 7: Take gradient step
        logger.info("\n=== Step 6: Take Gradient Step ===")
        led_values_new_raw = led_values_init - step_size * gradient
        led_values_new = cp.clip(led_values_new_raw, 0.0, 1.0)

        logger.info(f"New LED values (raw): range=[{cp.min(led_values_new_raw):.6f}, {cp.max(led_values_new_raw):.6f}]")
        logger.info(f"New LED values (clipped): range=[{cp.min(led_values_new):.6f}, {cp.max(led_values_new):.6f}]")

        # Check if values changed
        diff = cp.max(cp.abs(led_values_new - led_values_init))
        logger.info(f"Max change in LED values: {diff:.6f}")

        if diff < 1e-6:
            logger.warning("LED values barely changed - optimization may be stuck!")

    else:
        logger.error(f"Invalid step size calculation: g^T @ ATA @ g = {g_dot_ata_g:.6f} <= 0")

    # Step 8: Check if everything is working with the corrected matrix
    logger.info("\n=== Summary ===")
    logger.info(f"A^T @ b range: [{cp.min(result_atb):.3f}, {cp.max(result_atb):.3f}]")
    logger.info(
        f"ATA matrix max value (should be < 5000): {cp.max(dense_ata.dense_matrices_gpu) if dense_ata.dense_matrices_gpu is not None else 'Unknown'}"
    )
    logger.info(f"Initial LED values non-zero: {cp.count_nonzero(led_values_init)}/{led_values_init.size}")


if __name__ == "__main__":
    debug_optimization_step_by_step()
