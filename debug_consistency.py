#!/usr/bin/env python3
"""
Debug consistency between SingleBlockMixedSparseTensor and CSC matrix approaches.

This script creates small, controlled test cases to identify where the
two implementations differ and why.
"""

import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_simple_test_case():
    """Create a minimal test case for debugging."""
    logger.info("Creating simple test case...")

    # Very small dimensions for easy debugging
    batch_size = 2  # 2 LEDs
    channels = 2  # 2 channels
    height = 16  # 16x16 image
    width = 16
    block_size = 4  # 4x4 blocks

    tensor = SingleBlockMixedSparseTensor(
        batch_size, channels, height, width, block_size
    )

    # Set specific blocks with known values
    # LED 0, Channel 0: all 1.0s at position (2, 3)
    tensor.set_block(0, 0, 2, 3, cp.ones((4, 4), dtype=cp.float32))

    # LED 0, Channel 1: all 2.0s at position (8, 9)
    tensor.set_block(0, 1, 8, 9, cp.ones((4, 4), dtype=cp.float32) * 2.0)

    # LED 1, Channel 0: all 3.0s at position (0, 0)
    tensor.set_block(1, 0, 0, 0, cp.ones((4, 4), dtype=cp.float32) * 3.0)

    # LED 1, Channel 1: all 4.0s at position (10, 5)
    tensor.set_block(1, 1, 10, 5, cp.ones((4, 4), dtype=cp.float32) * 4.0)

    logger.info(f"Set {cp.sum(tensor.blocks_set)} blocks")
    return tensor


def create_csc_from_tensor(tensor):
    """Create CSC matrix from tensor for exact comparison."""
    logger.info("Converting tensor to CSC format...")

    # Combined matrix: [A_r 0; 0 A_g] layout for 2 channels
    # Shape: (pixels * channels, leds * channels)
    pixels = tensor.height * tensor.width
    matrix_shape = (pixels * tensor.channels, tensor.batch_size * tensor.channels)

    rows = []
    cols = []
    data = []

    # Convert each block to CSC entries
    for batch_idx in range(tensor.batch_size):
        for channel_idx in range(tensor.channels):
            if not tensor.blocks_set[batch_idx, channel_idx]:
                continue

            # Get block data and position
            block_values = cp.asnumpy(tensor.sparse_values[batch_idx, channel_idx])
            top_row = int(tensor.block_positions[batch_idx, channel_idx, 0])
            top_col = int(tensor.block_positions[batch_idx, channel_idx, 1])

            logger.info(
                f"Block ({batch_idx}, {channel_idx}): position=({top_row}, {top_col}), value={block_values[0,0]}"
            )

            # Convert block to sparse entries
            for i in range(tensor.block_size):
                for j in range(tensor.block_size):
                    value = block_values[i, j]
                    if abs(value) > 1e-10:
                        # Global pixel position
                        pixel_row = top_row + i
                        pixel_col = top_col + j
                        pixel_idx = pixel_row * tensor.width + pixel_col

                        # CSC matrix position (channel-specific offset)
                        matrix_row = channel_idx * pixels + pixel_idx
                        matrix_col = channel_idx * tensor.batch_size + batch_idx

                        rows.append(matrix_row)
                        cols.append(matrix_col)
                        data.append(value)

    # Create CSC matrix
    A_combined = sp.csc_matrix(
        (data, (rows, cols)), shape=matrix_shape, dtype=np.float32
    )

    logger.info(f"CSC matrix: {A_combined.shape}, {A_combined.nnz} non-zeros")

    # Transfer to GPU
    from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix

    A_combined_gpu = cupy_csc_matrix(A_combined)

    return A_combined_gpu


def create_test_target():
    """Create a simple test target image."""
    logger.info("Creating test target image...")

    # 16x16 image with known values
    target = cp.zeros((16, 16), dtype=cp.float32)

    # Set specific regions to known values
    target[2:6, 3:7] = 10.0  # Overlaps with LED 0, Channel 0 (1.0 * 10.0 * 16 = 160)
    target[8:12, 9:13] = 5.0  # Overlaps with LED 0, Channel 1 (2.0 * 5.0 * 16 = 160)
    target[0:4, 0:4] = 2.0  # Overlaps with LED 1, Channel 0 (3.0 * 2.0 * 16 = 96)
    target[10:14, 5:9] = 1.0  # Overlaps with LED 1, Channel 1 (4.0 * 1.0 * 16 = 64)

    logger.info(f"Target image shape: {target.shape}")
    logger.info(f"Target value ranges: min={target.min()}, max={target.max()}")

    return target


def compute_csc_result(A_combined_gpu, target):
    """Compute A^T @ b using CSC matrix."""
    logger.info("Computing CSC result...")

    # Convert target to combined vector format [R_pixels; G_pixels]
    pixels = target.size
    channels = 2
    target_combined = cp.zeros(pixels * channels, dtype=cp.float32)
    target_flat = target.ravel()

    for c in range(channels):
        start_idx = c * pixels
        end_idx = (c + 1) * pixels
        target_combined[start_idx:end_idx] = target_flat

    logger.info(f"Target combined shape: {target_combined.shape}")
    logger.info(f"Target combined first few values: {target_combined[:8]}")

    # Compute A^T @ b
    result_combined = A_combined_gpu.T @ target_combined

    logger.info(f"Result combined shape: {result_combined.shape}")
    logger.info(f"Result combined values: {result_combined}")

    # Convert back to (batch_size, channels) format
    batch_size = 2
    result = cp.zeros((batch_size, channels), dtype=cp.float32)
    for c in range(channels):
        start_idx = c * batch_size
        end_idx = (c + 1) * batch_size
        result[:, c] = result_combined[start_idx:end_idx]

    return result


def compute_tensor_result(tensor, target):
    """Compute A^T @ b using SingleBlockMixedSparseTensor."""
    logger.info("Computing tensor result...")

    result = tensor.transpose_dot_product(target)
    logger.info(f"Tensor result shape: {result.shape}")
    logger.info(f"Tensor result values:\n{result}")

    return result


def debug_tensor_computation(tensor, target):
    """Debug the tensor computation step by step."""
    logger.info("Debugging tensor computation...")

    # Manual computation for verification
    result_manual = cp.zeros((tensor.batch_size, tensor.channels), dtype=cp.float32)

    for batch_idx in range(tensor.batch_size):
        for channel_idx in range(tensor.channels):
            if not tensor.blocks_set[batch_idx, channel_idx]:
                continue

            # Get block position and values
            top_row = int(tensor.block_positions[batch_idx, channel_idx, 0])
            top_col = int(tensor.block_positions[batch_idx, channel_idx, 1])
            block_values = tensor.sparse_values[batch_idx, channel_idx]

            # Extract target region
            target_region = target[
                top_row : top_row + tensor.block_size,
                top_col : top_col + tensor.block_size,
            ]

            # Compute dot product
            dot_product = cp.sum(block_values * target_region)
            result_manual[batch_idx, channel_idx] = dot_product

            logger.info(f"Block ({batch_idx}, {channel_idx}):")
            logger.info(f"  Position: ({top_row}, {top_col})")
            logger.info(f"  Block values[0,0]: {block_values[0,0]}")
            logger.info(f"  Target region[0,0]: {target_region[0,0]}")
            logger.info(f"  Dot product: {dot_product}")

    logger.info(f"Manual result:\n{result_manual}")
    return result_manual


def main():
    """Run the consistency debugging."""
    logger.info("Debugging SingleBlockMixedSparseTensor vs CSC consistency")
    logger.info("=" * 60)

    # Create test case
    tensor = create_simple_test_case()
    target = create_test_target()

    # Compute results using both methods
    tensor_result = compute_tensor_result(tensor, target)
    manual_result = debug_tensor_computation(tensor, target)

    # Create CSC matrix and compute result
    csc_matrix = create_csc_from_tensor(tensor)
    csc_result = compute_csc_result(csc_matrix, target)

    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 60)

    logger.info(f"Tensor result:\n{tensor_result}")
    logger.info(f"Manual result:\n{manual_result}")
    logger.info(f"CSC result:\n{csc_result}")

    # Check tensor vs manual
    tensor_manual_diff = cp.max(cp.abs(tensor_result - manual_result))
    logger.info(f"\nTensor vs Manual max diff: {tensor_manual_diff}")

    # Check tensor vs CSC
    tensor_csc_diff = cp.max(cp.abs(tensor_result - csc_result))
    logger.info(f"Tensor vs CSC max diff: {tensor_csc_diff}")

    # Check manual vs CSC
    manual_csc_diff = cp.max(cp.abs(manual_result - csc_result))
    logger.info(f"Manual vs CSC max diff: {manual_csc_diff}")

    # Expected values (calculated by hand)
    expected = cp.array(
        [
            [160.0, 160.0],  # LED 0: (1.0*10.0*16, 2.0*5.0*16)
            [96.0, 64.0],  # LED 1: (3.0*2.0*16, 4.0*1.0*16)
        ],
        dtype=cp.float32,
    )

    logger.info(f"Expected result:\n{expected}")

    expected_tensor_diff = cp.max(cp.abs(tensor_result - expected))
    expected_csc_diff = cp.max(cp.abs(csc_result - expected))

    logger.info(f"\nTensor vs Expected max diff: {expected_tensor_diff}")
    logger.info(f"CSC vs Expected max diff: {expected_csc_diff}")

    # Determine which approach is correct
    if expected_tensor_diff < 1e-5:
        logger.info("✓ Tensor approach matches expected values")
    else:
        logger.warning("⚠ Tensor approach differs from expected")

    if expected_csc_diff < 1e-5:
        logger.info("✓ CSC approach matches expected values")
    else:
        logger.warning("⚠ CSC approach differs from expected")


if __name__ == "__main__":
    main()
