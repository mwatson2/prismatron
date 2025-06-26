#!/usr/bin/env python3
"""
Debug what the mixed tensor transpose_dot_product actually computes.
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


def debug_mixed_tensor_computation():
    """Debug what mixed tensor actually computes vs CSC."""
    logger.info("=== Debugging Mixed Tensor Computation ===")
    
    # Load pattern data
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
    
    # Create a simple test target: single pixel at (100, 100) with value [1, 0, 0] (red)
    test_target = np.zeros((height, width, 3), dtype=np.float32)
    test_target[100, 100, 0] = 1.0  # Red pixel
    
    logger.info(f"Test target: Single red pixel at (100, 100)")
    
    # CSC approach
    A_r = clipped_csc[:, 0::3]  # Red channels
    A_g = clipped_csc[:, 1::3]  # Green channels  
    A_b = clipped_csc[:, 2::3]  # Blue channels
    
    target_flat = test_target.reshape(-1, 3)
    atb_r_csc = A_r.T @ target_flat[:, 0]  # Red pixels to red LEDs
    atb_g_csc = A_g.T @ target_flat[:, 1]  # Green pixels to green LEDs
    atb_b_csc = A_b.T @ target_flat[:, 2]  # Blue pixels to blue LEDs
    result_csc = np.column_stack([atb_r_csc, atb_g_csc, atb_b_csc])
    
    logger.info(f"CSC result for red pixel: shape {result_csc.shape}")
    logger.info(f"CSC red channel (nonzero): {np.count_nonzero(atb_r_csc)} values")
    logger.info(f"CSC green channel (nonzero): {np.count_nonzero(atb_g_csc)} values") 
    logger.info(f"CSC blue channel (nonzero): {np.count_nonzero(atb_b_csc)} values")
    
    # Show CSC results for LEDs that should be affected
    pixel_idx = 100 * width + 100  # Flat index of test pixel
    logger.info(f"Test pixel flat index: {pixel_idx}")
    
    # Find which LEDs are affected by this pixel
    affected_leds_r = []
    affected_leds_g = []
    affected_leds_b = []
    
    for led in range(led_count):
        if A_r[pixel_idx, led] > 0:
            affected_leds_r.append((led, A_r[pixel_idx, led]))
        if A_g[pixel_idx, led] > 0:
            affected_leds_g.append((led, A_g[pixel_idx, led]))  
        if A_b[pixel_idx, led] > 0:
            affected_leds_b.append((led, A_b[pixel_idx, led]))
    
    logger.info(f"LEDs affected by red pixel (red channel): {len(affected_leds_r)}")
    logger.info(f"LEDs affected by red pixel (green channel): {len(affected_leds_g)}")
    logger.info(f"LEDs affected by red pixel (blue channel): {len(affected_leds_b)}")
    
    if affected_leds_r:
        led_id, weight = affected_leds_r[0]
        logger.info(f"Sample affected LED {led_id} (red): weight={weight:.3f}, CSC result={result_csc[led_id, 0]:.3f}")
    
    # Mixed tensor approach  
    target_r = cp.asarray(test_target[:, :, 0])
    target_g = cp.asarray(test_target[:, :, 1]) 
    target_b = cp.asarray(test_target[:, :, 2])
    
    result_r = mixed_tensor.transpose_dot_product_cuda_high_performance(target_r)
    result_g = mixed_tensor.transpose_dot_product_cuda_high_performance(target_g)
    result_b = mixed_tensor.transpose_dot_product_cuda_high_performance(target_b)
    
    logger.info(f"Mixed tensor results: R={result_r.shape}, G={result_g.shape}, B={result_b.shape}")
    logger.info(f"Mixed tensor red result (nonzero): {cp.count_nonzero(result_r).get()} values")
    logger.info(f"Mixed tensor green result (nonzero): {cp.count_nonzero(result_g).get()} values")
    logger.info(f"Mixed tensor blue result (nonzero): {cp.count_nonzero(result_b).get()} values")
    
    # Compare the results
    mixed_diagonal = np.zeros((led_count, 3), dtype=np.float32)
    mixed_diagonal[:, 0] = result_r[:, 0].get()  # Red pixel to red LED channels
    mixed_diagonal[:, 1] = result_g[:, 1].get()  # Green pixel to green LED channels  
    mixed_diagonal[:, 2] = result_b[:, 2].get()  # Blue pixel to blue LED channels
    
    logger.info("=== Single Pixel Test Results ===")
    logger.info(f"CSC result nonzeros: {np.count_nonzero(result_csc)}")
    logger.info(f"Mixed diagonal nonzeros: {np.count_nonzero(mixed_diagonal)}")
    
    diff = np.abs(result_csc - mixed_diagonal)
    logger.info(f"Difference: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    # Check if the affected LEDs match
    if affected_leds_r:
        led_id, weight = affected_leds_r[0]
        logger.info(f"LED {led_id}: CSC={result_csc[led_id, 0]:.6f}, Mixed={mixed_diagonal[led_id, 0]:.6f}, diff={diff[led_id, 0]:.6f}")
    
    # Let's also check what happens when we use ALL result channels
    mixed_full = np.zeros((led_count, 3), dtype=np.float32)
    mixed_full[:, 0] = result_r[:, 0].get()  # Red pixel affects all LEDs' red channels
    mixed_full[:, 1] = result_r[:, 1].get()  # Red pixel affects all LEDs' green channels
    mixed_full[:, 2] = result_r[:, 2].get()  # Red pixel affects all LEDs' blue channels
    
    diff_full = np.abs(result_csc - mixed_full)
    logger.info(f"Full extraction - Difference: max={diff_full.max():.6f}, mean={diff_full.mean():.6f}")
    
    return True


if __name__ == "__main__":
    success = debug_mixed_tensor_computation()
    sys.exit(0 if success else 1)