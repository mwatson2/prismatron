#!/usr/bin/env python3
"""
Test that clipped CSC patterns and mixed tensor produce identical A^T@b results.

This test uses the new clipping-equivalent patterns to verify mathematical equivalence.
"""

import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.const import FRAME_HEIGHT, FRAME_WIDTH
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_clipped_equivalence():
    """Test that clipped CSC and mixed tensor produce identical A^T@b results."""
    logger.info("=== Testing Clipped Equivalence ===")
    
    # Load pattern data with built-in clipping equivalence
    data = np.load('diffusion_patterns/test_clipped_100.npz')
    logger.info(f"Loaded pattern file: {data.files}")
    
    # Reconstruct clipped CSC matrix (already equivalent to mixed tensor)
    matrix_shape = tuple(data['matrix_shape'])
    matrix_data = data['matrix_data']
    matrix_indices = data['matrix_indices'] 
    matrix_indptr = data['matrix_indptr']
    
    clipped_csc = sp.csc_matrix((matrix_data, matrix_indices, matrix_indptr), 
                                shape=matrix_shape)
    logger.info(f"Loaded clipped CSC matrix shape: {clipped_csc.shape}")
    logger.info(f"Clipped CSC NNZ: {clipped_csc.nnz}")
    
    # Extract RGB channel matrices from clipped CSC
    logger.info("Extracting RGB channels from clipped CSC...")
    A_r_clipped = clipped_csc[:, 0::3]  # Red channels
    A_g_clipped = clipped_csc[:, 1::3]  # Green channels
    A_b_clipped = clipped_csc[:, 2::3]  # Blue channels
    
    # Create test target image
    np.random.seed(42)
    target_image = np.random.randn(FRAME_HEIGHT, FRAME_WIDTH, 3).astype(np.float32)
    target_flat = target_image.reshape(-1, 3)  # (pixels, 3)
    
    # Test A^T@b calculation with clipped CSC
    logger.info("Computing A^T@b with clipped CSC...")
    atb_r_clipped = A_r_clipped.T @ target_flat[:, 0]
    atb_g_clipped = A_g_clipped.T @ target_flat[:, 1]
    atb_b_clipped = A_b_clipped.T @ target_flat[:, 2]
    result_clipped = np.column_stack([atb_r_clipped, atb_g_clipped, atb_b_clipped])
    
    # Load and test mixed tensor
    logger.info("Loading mixed tensor...")
    led_count = int(data['mixed_tensor_led_count'])
    channels = int(data['mixed_tensor_channels'])
    height = int(data['mixed_tensor_height'])
    width = int(data['mixed_tensor_width'])
    block_size = int(data['mixed_tensor_block_size'])
    
    mixed_tensor = SingleBlockMixedSparseTensor(led_count, channels, height, width, block_size)
    
    # Load data with transpose for new channels-first layout
    old_values = data['mixed_tensor_values']  # (LEDs, channels, H, W)
    old_positions = data['mixed_tensor_positions']  # (LEDs, channels, 2)
    old_blocks_set = data['mixed_tensor_blocks_set']  # (LEDs, channels)
    
    # Transpose to channels-first layout
    mixed_tensor.sparse_values = cp.asarray(old_values.transpose(1, 0, 2, 3))  # (channels, LEDs, H, W)
    mixed_tensor.block_positions = cp.asarray(old_positions.transpose(1, 0, 2))  # (channels, LEDs, 2)
    mixed_tensor.blocks_set = cp.asarray(old_blocks_set.transpose(1, 0))  # (channels, LEDs)
    
    logger.info("Computing A^T@b with mixed tensor...")
    
    # Test each color channel separately and extract diagonal
    target_gpu = cp.asarray(target_image[:, :, 0])  # Test with red channel
    result_mixed_r = mixed_tensor.transpose_dot_product(target_gpu)
    
    target_gpu = cp.asarray(target_image[:, :, 1])  # Test with green channel
    result_mixed_g = mixed_tensor.transpose_dot_product(target_gpu)
    
    target_gpu = cp.asarray(target_image[:, :, 2])  # Test with blue channel
    result_mixed_b = mixed_tensor.transpose_dot_product(target_gpu)
    
    # Extract diagonal elements to match CSC approach
    result_mixed = np.zeros((led_count, 3), dtype=np.float32)
    result_mixed[:, 0] = result_mixed_r[:, 0].get()  # Red LED values from red channel
    result_mixed[:, 1] = result_mixed_g[:, 1].get()  # Green LED values from green channel
    result_mixed[:, 2] = result_mixed_b[:, 2].get()  # Blue LED values from blue channel
    
    # Compare results
    logger.info("=== Comparison Results ===")
    logger.info(f"Clipped CSC result shape: {result_clipped.shape}")
    logger.info(f"Mixed tensor result shape: {result_mixed.shape}")
    
    # Check if clipped CSC matches mixed tensor (allow for GPU/CPU precision differences)
    if np.allclose(result_clipped, result_mixed, rtol=1e-4, atol=1e-4):
        logger.info("✅ SUCCESS: Clipped CSC matches mixed tensor perfectly!")
        logger.info(f"Max difference: {np.abs(result_clipped - result_mixed).max():.2e}")
        logger.info(f"Mean difference: {np.abs(result_clipped - result_mixed).mean():.2e}")
        return True
    else:
        logger.error("❌ MISMATCH: Clipped CSC does not match mixed tensor")
        logger.error(f"Max difference: {np.abs(result_clipped - result_mixed).max()}")
        logger.error(f"Mean difference: {np.abs(result_clipped - result_mixed).mean()}")
        
        # Show some sample differences
        logger.error("Sample differences (first 5 LEDs):")
        for i in range(min(5, led_count)):
            logger.error(f"LED {i}: Clipped={result_clipped[i]}, Mixed={result_mixed[i]}")
        
        return False


if __name__ == "__main__":
    success = test_clipped_equivalence()
    sys.exit(0 if success else 1)