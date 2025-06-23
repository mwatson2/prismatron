#!/usr/bin/env python3
"""
Debug medium-scale case to identify where CSC and SingleBlock differ.
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_medium_tensor():
    """Create medium-sized tensor for debugging."""
    logger.info("Creating medium tensor: 10 LEDs, 3 channels, 64x64 image, 8x8 blocks")

    batch_size = 10
    channels = 3
    height = 64
    width = 64
    block_size = 8

    tensor = SingleBlockMixedSparseTensor(
        batch_size, channels, height, width, block_size
    )

    # Set blocks with simple patterns
    np.random.seed(42)  # For reproducibility

    for led_id in range(batch_size):
        for channel in range(channels):
            # Simple pattern: just put the LED/channel ID as the value
            value = float(led_id * channels + channel + 1)

            # Random position
            top_row = np.random.randint(0, height - block_size)
            top_col = np.random.randint(0, width - block_size)

            pattern = cp.ones((block_size, block_size), dtype=cp.float32) * value
            tensor.set_block(led_id, channel, top_row, top_col, pattern)

    logger.info(f"Created tensor with {cp.sum(tensor.blocks_set)} blocks")
    return tensor


def create_csc_matrix(tensor):
    """Create CSC matrix with detailed logging."""
    logger.info("Creating CSC matrix...")

    pixels = tensor.height * tensor.width
    matrix_shape = (pixels * tensor.channels, tensor.batch_size * tensor.channels)

    rows = []
    cols = []
    data = []

    # Track what we're adding for debugging
    entries_per_led_channel = {}

    for batch_idx in range(tensor.batch_size):
        for channel_idx in range(tensor.channels):
            if not tensor.blocks_set[batch_idx, channel_idx]:
                continue

            block_values = cp.asnumpy(tensor.sparse_values[batch_idx, channel_idx])
            top_row = int(tensor.block_positions[batch_idx, channel_idx, 0])
            top_col = int(tensor.block_positions[batch_idx, channel_idx, 1])

            key = (batch_idx, channel_idx)
            entries_per_led_channel[key] = []

            for i in range(tensor.block_size):
                for j in range(tensor.block_size):
                    value = block_values[i, j]
                    if abs(value) > 1e-10:
                        pixel_row = top_row + i
                        pixel_col = top_col + j
                        pixel_idx = pixel_row * tensor.width + pixel_col

                        # CSC matrix position
                        matrix_row = channel_idx * pixels + pixel_idx
                        matrix_col = channel_idx * tensor.batch_size + batch_idx

                        rows.append(matrix_row)
                        cols.append(matrix_col)
                        data.append(value)

                        entries_per_led_channel[key].append(
                            (matrix_row, matrix_col, value)
                        )

    # Log first few entries for debugging
    logger.info("First few CSC entries:")
    for i in range(min(10, len(data))):
        logger.info(f"  ({rows[i]}, {cols[i]}) = {data[i]}")

    A_combined = sp.csc_matrix(
        (data, (rows, cols)), shape=matrix_shape, dtype=np.float32
    )

    from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix

    A_combined_gpu = cupy_csc_matrix(A_combined)

    logger.info(f"CSC matrix: {A_combined.shape}, {A_combined.nnz} non-zeros")
    return A_combined_gpu


def compute_csc_result(A_combined_gpu, target, channels):
    """Compute CSC result with logging."""
    logger.info("Computing CSC result...")

    pixels = target.size
    target_combined = cp.zeros(pixels * channels, dtype=cp.float32)
    target_flat = target.ravel()

    logger.info(f"Target flat first 5 values: {target_flat[:5]}")

    for c in range(channels):
        start_idx = c * pixels
        end_idx = (c + 1) * pixels
        target_combined[start_idx:end_idx] = target_flat

    logger.info(f"Target combined first 10 values: {target_combined[:10]}")

    # Compute A^T @ b
    result_combined = A_combined_gpu.T @ target_combined
    logger.info(f"Result combined: {result_combined}")

    # Convert back to (batch_size, channels) format
    batch_size = A_combined_gpu.shape[1] // channels
    result = cp.zeros((batch_size, channels), dtype=cp.float32)
    for c in range(channels):
        start_idx = c * batch_size
        end_idx = (c + 1) * batch_size
        result[:, c] = result_combined[start_idx:end_idx]

    return result


def manual_computation(tensor, target):
    """Manual computation for verification."""
    logger.info("Manual computation...")

    result = cp.zeros((tensor.batch_size, tensor.channels), dtype=cp.float32)

    for batch_idx in range(tensor.batch_size):
        for channel_idx in range(tensor.channels):
            if not tensor.blocks_set[batch_idx, channel_idx]:
                continue

            top_row = int(tensor.block_positions[batch_idx, channel_idx, 0])
            top_col = int(tensor.block_positions[batch_idx, channel_idx, 1])
            block_values = tensor.sparse_values[batch_idx, channel_idx]

            target_region = target[
                top_row : top_row + tensor.block_size,
                top_col : top_col + tensor.block_size,
            ]

            dot_product = cp.sum(block_values * target_region)
            result[batch_idx, channel_idx] = dot_product

            logger.info(
                f"LED {batch_idx}, Ch {channel_idx}: pos=({top_row},{top_col}), "
                f"block_val={block_values[0,0]}, target_avg={cp.mean(target_region):.3f}, "
                f"result={dot_product:.3f}"
            )

    return result


def main():
    """Run medium-scale debugging."""
    logger.info("Medium-scale debugging of CSC vs SingleBlock")

    # Create test case
    tensor = create_medium_tensor()

    # Create simple target
    target = cp.random.rand(64, 64).astype(cp.float32)
    logger.info(f"Target range: [{target.min():.3f}, {target.max():.3f}]")

    # Compute using all three methods
    tensor_result = tensor.transpose_dot_product(target)
    manual_result = manual_computation(tensor, target)

    csc_matrix = create_csc_matrix(tensor)
    csc_result = compute_csc_result(csc_matrix, target, tensor.channels)

    # Compare
    logger.info("\n=== COMPARISON ===")
    logger.info(f"Tensor result:\n{tensor_result}")
    logger.info(f"Manual result:\n{manual_result}")
    logger.info(f"CSC result:\n{csc_result}")

    tensor_manual_diff = cp.max(cp.abs(tensor_result - manual_result))
    tensor_csc_diff = cp.max(cp.abs(tensor_result - csc_result))
    manual_csc_diff = cp.max(cp.abs(manual_result - csc_result))

    logger.info(f"\nTensor vs Manual: {tensor_manual_diff:.6f}")
    logger.info(f"Tensor vs CSC: {tensor_csc_diff:.6f}")
    logger.info(f"Manual vs CSC: {manual_csc_diff:.6f}")

    if tensor_manual_diff < 1e-5:
        logger.info("✓ Tensor matches manual")
    if tensor_csc_diff < 1e-5:
        logger.info("✓ Tensor matches CSC")
    if manual_csc_diff < 1e-5:
        logger.info("✓ Manual matches CSC")


if __name__ == "__main__":
    main()
