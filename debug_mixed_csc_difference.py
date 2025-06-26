#!/usr/bin/env python3
"""
Debug why mixed tensor and CSC produce different results.
"""

import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse as sp
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.const import FRAME_HEIGHT, FRAME_WIDTH
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_atb_calculation():
    """Debug A^T@b calculation differences between mixed tensor and CSC."""
    logger.info("=== Debugging A^T@b Calculation Differences ===")
    
    # Load clipped pattern data
    data = np.load('diffusion_patterns/synthetic_1000_with_ata_clipped.npz', allow_pickle=True)
    
    # Load CSC matrix
    matrix_shape = tuple(data['matrix_shape'])
    matrix_data = data['matrix_data']
    matrix_indices = data['matrix_indices'] 
    matrix_indptr = data['matrix_indptr']
    clipped_csc = sp.csc_matrix((matrix_data, matrix_indices, matrix_indptr), shape=matrix_shape)
    
    # Load mixed tensor
    led_count = int(data['mixed_tensor_led_count'])
    channels = int(data['mixed_tensor_channels'])
    height = int(data['mixed_tensor_height'])
    width = int(data['mixed_tensor_width'])
    block_size = int(data['mixed_tensor_block_size'])
    
    mixed_tensor = SingleBlockMixedSparseTensor(led_count, channels, height, width, block_size)
    
    # Load with channels-first layout
    old_values = data['mixed_tensor_values']
    old_positions = data['mixed_tensor_positions']
    old_blocks_set = data['mixed_tensor_blocks_set']
    
    mixed_tensor.sparse_values = cp.asarray(old_values.transpose(1, 0, 2, 3))
    mixed_tensor.block_positions = cp.asarray(old_positions.transpose(1, 0, 2))
    mixed_tensor.blocks_set = cp.asarray(old_blocks_set.transpose(1, 0))
    
    # Load same test image that the optimizer would use
    img = Image.open('env/lib/python3.10/site-packages/sklearn/datasets/images/flower.jpg')
    img = img.resize((800, 480))  # Match FRAME_WIDTH x FRAME_HEIGHT
    target_image = np.array(img).astype(np.float32) / 255.0
    
    logger.info(f"Target image shape: {target_image.shape}")
    logger.info(f"CSC matrix shape: {clipped_csc.shape}")
    logger.info(f"Mixed tensor shape: ({led_count}, {channels}, {height}, {width})")
    
    # CSC A^T@b calculation (RGB channels)
    A_r = clipped_csc[:, 0::3]  # Red channels
    A_g = clipped_csc[:, 1::3]  # Green channels
    A_b = clipped_csc[:, 2::3]  # Blue channels
    
    target_flat = target_image.reshape(-1, 3)
    atb_r_csc = A_r.T @ target_flat[:, 0]
    atb_g_csc = A_g.T @ target_flat[:, 1] 
    atb_b_csc = A_b.T @ target_flat[:, 2]
    result_csc = np.column_stack([atb_r_csc, atb_g_csc, atb_b_csc])
    
    logger.info(f"CSC A^T@b result shape: {result_csc.shape}")
    logger.info(f"CSC A^T@b sample: {result_csc[:3]}")
    
    # Mixed tensor A^T@b calculation (channel separation like in the optimizer)
    target_r = cp.asarray(target_image[:, :, 0])
    target_g = cp.asarray(target_image[:, :, 1])
    target_b = cp.asarray(target_image[:, :, 2])
    
    result_r = mixed_tensor.transpose_dot_product_cuda_high_performance(target_r)
    result_g = mixed_tensor.transpose_dot_product_cuda_high_performance(target_g)
    result_b = mixed_tensor.transpose_dot_product_cuda_high_performance(target_b)
    
    # Extract diagonal elements (this is the key issue!)
    result_mixed = np.zeros((led_count, 3), dtype=np.float32)
    result_mixed[:, 0] = result_r[:, 0].get()  # Red LED values from red channel
    result_mixed[:, 1] = result_g[:, 1].get()  # Green LED values from green channel
    result_mixed[:, 2] = result_b[:, 2].get()  # Blue LED values from blue channel
    
    logger.info(f"Mixed tensor A^T@b result shape: {result_mixed.shape}")
    logger.info(f"Mixed tensor A^T@b sample: {result_mixed[:3]}")
    
    # Compare results
    diff = np.abs(result_csc - result_mixed)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    logger.info(f"Max difference: {max_diff}")
    logger.info(f"Mean difference: {mean_diff}")
    logger.info(f"Relative difference: {max_diff / np.abs(result_csc).max():.6f}")
    
    # Check if the issue is in the diagonal extraction
    logger.info("\n=== Checking Mixed Tensor Full Results ===")
    logger.info(f"Red channel result shape: {result_r.shape}")
    logger.info(f"Red channel result sample:\n{result_r[:3].get()}")
    
    # The issue might be that we should NOT be extracting diagonal elements
    # Instead, we should be processing all channels together
    logger.info("\n=== Testing Full Channel Mixed Tensor ===")
    
    # Try processing the full RGB image at once (this might be the correct approach)
    target_rgb_flat = cp.asarray(target_image.reshape(height, width, channels))
    
    # Process each color channel separately but keep full results
    result_mixed_full = np.zeros((led_count, 3), dtype=np.float32)
    
    for c in range(3):
        target_channel = target_rgb_flat[:, :, c]
        result_channel = mixed_tensor.transpose_dot_product_cuda_high_performance(target_channel)
        result_mixed_full[:, c] = result_channel[:, c].get()  # Take the c-th channel result
    
    logger.info(f"Full mixed tensor result shape: {result_mixed_full.shape}")
    logger.info(f"Full mixed tensor result sample: {result_mixed_full[:3]}")
    
    # Compare this with CSC
    diff_full = np.abs(result_csc - result_mixed_full)
    max_diff_full = diff_full.max()
    mean_diff_full = diff_full.mean()
    
    logger.info(f"Full approach - Max difference: {max_diff_full}")
    logger.info(f"Full approach - Mean difference: {mean_diff_full}")
    
    if max_diff_full < max_diff:
        logger.info("✅ Full channel approach is better!")
        return True
    else:
        logger.error("❌ Still significant differences")
        return False


if __name__ == "__main__":
    success = debug_atb_calculation()
    sys.exit(0 if success else 1)