#!/usr/bin/env python3
"""
Validate mixed tensor einsum operation by hand calculation.

This test manually computes the expected A^T @ b result and compares it
with the CUDA kernel output to ensure correctness.
"""

import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_einsum_by_hand():
    """Test einsum by manually calculating expected results."""
    logger.info("=== Testing Einsum by Hand Calculation ===")
    
    # Small test case: 2 LEDs, 3 channels, 20x20 image, 4x4 blocks
    batch_size, channels = 2, 3
    height, width = 20, 20
    block_size = 4
    
    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size)
    
    # Set specific known patterns
    # LED 0, Red channel at (2,3): all 2.0s
    led0_r_values = cp.ones((4, 4), dtype=cp.float32) * 2.0
    tensor.set_block(0, 0, 2, 3, led0_r_values)
    
    # LED 0, Green channel at (8,5): all 3.0s  
    led0_g_values = cp.ones((4, 4), dtype=cp.float32) * 3.0
    tensor.set_block(0, 1, 8, 5, led0_g_values)
    
    # LED 0, Blue channel at (12,10): all 4.0s
    led0_b_values = cp.ones((4, 4), dtype=cp.float32) * 4.0
    tensor.set_block(0, 2, 12, 10, led0_b_values)
    
    # LED 1, Red channel at (1,15): all 5.0s (moved to avoid overlap)
    led1_r_values = cp.ones((4, 4), dtype=cp.float32) * 5.0
    tensor.set_block(1, 0, 1, 15, led1_r_values)
    
    # LED 1, Green channel at (6,12): all 6.0s (moved to avoid overlap)
    led1_g_values = cp.ones((4, 4), dtype=cp.float32) * 6.0
    tensor.set_block(1, 1, 6, 12, led1_g_values)
    
    # LED 1, Blue channel at (16,16): all 7.0s (moved to avoid overlap)
    led1_b_values = cp.ones((4, 4), dtype=cp.float32) * 7.0
    tensor.set_block(1, 2, 16, 16, led1_b_values)
    
    # Create target image with NON-OVERLAPPING known values
    target = cp.zeros((height, width), dtype=cp.float32)
    target[2:6, 3:7] = 10.0     # LED 0 Red at (2,3)
    target[8:12, 5:9] = 20.0    # LED 0 Green at (8,5)  
    target[12:16, 10:14] = 30.0 # LED 0 Blue at (12,10)
    target[1:5, 15:19] = 40.0   # LED 1 Red at (1,15)
    target[6:10, 12:16] = 60.0  # LED 1 Green at (6,12) - moved to avoid overlap
    target[16:20, 16:20] = 50.0 # LED 1 Blue at (16,16) - moved to avoid overlap
    
    # DEBUG: Check actual overlaps
    logger.info(f"LED 0 Red block at (2,3): target values = {target[2:6, 3:7].sum()}")
    logger.info(f"LED 0 Green block at (8,5): target values = {target[8:12, 5:9].sum()}")  
    logger.info(f"LED 0 Blue block at (12,10): target values = {target[12:16, 10:14].sum()}")
    logger.info(f"LED 1 Red block at (1,15): target values = {target[1:5, 15:19].sum()}")
    logger.info(f"LED 1 Green block at (6,12): target values = {target[6:10, 12:16].sum()}")
    logger.info(f"LED 1 Blue block at (16,16): target values = {target[16:20, 16:20].sum()}")
    
    # Check for overlaps between target regions
    logger.info(f"LED 0 Red region overlaps with LED 1 Blue: {target[2:6, 3:7][target[1:5, 1:5] > 0].sum()}")
    logger.info(f"Any overlap between (1:5,1:5) and (8:12,5:9): {np.any(target[1:5, 1:5] * target[8:12, 5:9])}")
    
    logger.info(f"Target shape: {target.shape}")
    logger.info(f"Target non-zero regions: {cp.count_nonzero(target)} pixels")
    
    # Compute using mixed tensor
    result = tensor.transpose_dot_product(target)
    logger.info(f"Mixed tensor result shape: {result.shape}")
    logger.info(f"Mixed tensor result:\n{result}")
    
    # Manual calculation for verification
    # LED 0, Red: 2.0 * 10.0 * 16 = 320.0
    led0_r_expected = 2.0 * 10.0 * 16
    
    # LED 0, Green: 3.0 * 20.0 * 16 = 960.0
    led0_g_expected = 3.0 * 20.0 * 16
    
    # LED 0, Blue: 4.0 * 30.0 * 16 = 1920.0
    led0_b_expected = 4.0 * 30.0 * 16
    
    # LED 1, Red: 5.0 * 40.0 * 16 = 3200.0
    led1_r_expected = 5.0 * 40.0 * 16
    
    # LED 1, Green: 6.0 * 60.0 * 16 = 5760.0
    led1_g_expected = 6.0 * 60.0 * 16
    
    # LED 1, Blue: 7.0 * 50.0 * 16 = 5600.0 (at position 16,16)
    led1_b_expected = 7.0 * 50.0 * 16
    
    expected_manual = cp.array([
        [led0_r_expected, led0_g_expected, led0_b_expected],  # LED 0: [320, 960, 1920]
        [led1_r_expected, led1_g_expected, led1_b_expected],  # LED 1: [3200, 5760, 5600]
    ], dtype=cp.float32)
    
    logger.info(f"Manual calculation expected:\n{expected_manual}")
    
    # Compare results
    if cp.allclose(result, expected_manual, rtol=1e-5):
        logger.info("✅ Mixed tensor einsum matches manual calculation!")
        return True
    else:
        logger.error(f"❌ Mismatch! Difference:\n{result - expected_manual}")
        return False


def test_einsum_with_numpy_reference():
    """Test einsum against numpy reference implementation."""
    logger.info("=== Testing Against NumPy Einsum Reference ===")
    
    # Create small test case
    batch_size, channels = 3, 3
    height, width = 16, 16  
    block_size = 4
    
    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size)
    
    # Create random sparse patterns
    np.random.seed(42)
    cp.random.seed(42)
    
    for led in range(batch_size):
        for channel in range(channels):
            # Random block position
            max_row = height - block_size
            max_col = width - block_size
            row = np.random.randint(0, max_row)
            col = np.random.randint(0, max_col)
            
            # Random block values
            values = cp.random.randn(block_size, block_size).astype(cp.float32)
            tensor.set_block(led, channel, row, col, values)
    
    # Create random target
    target = cp.random.randn(height, width).astype(cp.float32)
    
    # Compute using mixed tensor
    result_mixed = tensor.transpose_dot_product(target)
    
    # Compute using numpy einsum reference
    # Convert mixed tensor to dense format for numpy - use new channels-first layout
    dense_tensor = cp.zeros((channels, batch_size, height, width), dtype=cp.float32)
    
    for channel in range(channels):
        for led in range(batch_size):
            if tensor.blocks_set[channel, led]:  # New indexing
                row, col = tensor.block_positions[channel, led]  # New indexing
                values = tensor.sparse_values[channel, led]  # New indexing
                dense_tensor[channel, led, row:row+block_size, col:col+block_size] = values
    
    # NumPy einsum: 'ijkl,kl->ij' means sum over k,l dimensions
    # With channels-first layout: (channels, leds, height, width) -> (channels, leds)
    result_numpy_channels_first = cp.einsum('ijkl,kl->ij', dense_tensor, target)
    # Transpose to match expected output format: (channels, leds) -> (leds, channels)
    result_numpy = result_numpy_channels_first.T
    
    logger.info(f"Mixed tensor result shape: {result_mixed.shape}")
    logger.info(f"NumPy einsum result shape: {result_numpy.shape}")
    logger.info(f"Results close: {cp.allclose(result_mixed, result_numpy, rtol=1e-5)}")
    
    if cp.allclose(result_mixed, result_numpy, rtol=1e-5):
        logger.info("✅ Mixed tensor matches NumPy einsum reference!")
        return True
    else:
        logger.error(f"❌ Mismatch! Max difference: {cp.abs(result_mixed - result_numpy).max()}")
        logger.error(f"Mixed tensor result:\n{result_mixed}")
        logger.error(f"NumPy result:\n{result_numpy}")
        return False


if __name__ == "__main__":
    success1 = test_einsum_by_hand()
    success2 = test_einsum_with_numpy_reference()
    
    if success1 and success2:
        logger.info("🎉 All einsum validation tests passed!")
        sys.exit(0)
    else:
        logger.error("💥 Some einsum validation tests failed!")
        sys.exit(1)