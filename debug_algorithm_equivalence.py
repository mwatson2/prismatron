#!/usr/bin/env python3
"""
Step-by-step debugging of algorithm equivalence between CSC and mixed tensor.
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


def debug_algorithm_step_by_step():
    """Debug the algorithm step by step to find the difference."""
    logger.info("=== Step-by-Step Algorithm Equivalence Debugging ===")
    
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
    
    # Load test image
    img = Image.open('env/lib/python3.10/site-packages/sklearn/datasets/images/flower.jpg')
    img = img.resize((800, 480))
    target_image = np.array(img).astype(np.float32) / 255.0
    
    logger.info(f"Target image shape: {target_image.shape}")
    logger.info(f"CSC matrix shape: {clipped_csc.shape}")
    logger.info(f"Mixed tensor blocks stored: {cp.sum(mixed_tensor.blocks_set).get()}")
    
    # === STEP 1: Data Structure Analysis ===
    logger.info("\n=== STEP 1: Data Structure Analysis ===")
    
    # CSC structure: (pixels, led*3) where columns are [LED0_R, LED0_G, LED0_B, LED1_R, LED1_G, LED1_B, ...]
    logger.info(f"CSC matrix shape: {clipped_csc.shape}")
    logger.info(f"CSC matrix columns: {clipped_csc.shape[1]} (should be {led_count * 3})")
    
    # Mixed tensor structure: channels-first (3, 1000, H, W)
    logger.info(f"Mixed tensor shape: ({channels}, {led_count}, {height}, {width})")
    logger.info(f"Mixed tensor blocks: {mixed_tensor.sparse_values.shape}")
    
    # === STEP 2: A^T@b Calculation Comparison ===
    logger.info("\n=== STEP 2: A^T@b Calculation Comparison ===")
    
    # CSC approach: Extract RGB channel matrices
    A_r = clipped_csc[:, 0::3]  # Red channels: columns [0, 3, 6, 9, ...]
    A_g = clipped_csc[:, 1::3]  # Green channels: columns [1, 4, 7, 10, ...]
    A_b = clipped_csc[:, 2::3]  # Blue channels: columns [2, 5, 8, 11, ...]
    
    logger.info(f"CSC channel matrices: A_r={A_r.shape}, A_g={A_g.shape}, A_b={A_b.shape}")
    
    # Target processing
    target_flat = target_image.reshape(-1, 3)  # (384000, 3)
    
    # CSC A^T@b calculation
    atb_r_csc = A_r.T @ target_flat[:, 0]  # (1000,)
    atb_g_csc = A_g.T @ target_flat[:, 1]  # (1000,)  
    atb_b_csc = A_b.T @ target_flat[:, 2]  # (1000,)
    result_csc = np.column_stack([atb_r_csc, atb_g_csc, atb_b_csc])  # (1000, 3)
    
    logger.info(f"CSC A^T@b result shape: {result_csc.shape}")
    logger.info(f"CSC A^T@b sample: {result_csc[:3]}")
    
    # Mixed tensor A^T@b calculation
    target_r = cp.asarray(target_image[:, :, 0])
    target_g = cp.asarray(target_image[:, :, 1])
    target_b = cp.asarray(target_image[:, :, 2])
    
    result_r = mixed_tensor.transpose_dot_product(target_r)  # (1000, 3)
    result_g = mixed_tensor.transpose_dot_product(target_g)  # (1000, 3)
    result_b = mixed_tensor.transpose_dot_product(target_b)  # (1000, 3)
    
    logger.info(f"Mixed tensor results shapes: R={result_r.shape}, G={result_g.shape}, B={result_b.shape}")
    logger.info(f"Mixed tensor result_r sample: {result_r[:3].get()}")
    logger.info(f"Mixed tensor result_g sample: {result_g[:3].get()}")
    logger.info(f"Mixed tensor result_b sample: {result_b[:3].get()}")
    
    # === STEP 3: The Key Question - How to Extract Results ===
    logger.info("\n=== STEP 3: Result Extraction Analysis ===")
    
    # Current implementation extracts diagonal:
    result_mixed_diagonal = np.zeros((led_count, 3), dtype=np.float32)
    result_mixed_diagonal[:, 0] = result_r[:, 0].get()  # Red LED values from red channel
    result_mixed_diagonal[:, 1] = result_g[:, 1].get()  # Green LED values from green channel
    result_mixed_diagonal[:, 2] = result_b[:, 2].get()  # Blue LED values from blue channel
    
    logger.info(f"Mixed diagonal extraction: {result_mixed_diagonal[:3]}")
    
    # Alternative: What if we need ALL results from each channel?
    result_mixed_full = np.zeros((led_count, 3), dtype=np.float32)
    result_mixed_full[:, 0] = result_r[:, 0].get()  # All LEDs driven by red pixels -> red LED values
    result_mixed_full[:, 1] = result_g[:, 1].get()  # All LEDs driven by green pixels -> green LED values  
    result_mixed_full[:, 2] = result_b[:, 2].get()  # All LEDs driven by blue pixels -> blue LED values
    
    logger.info(f"Mixed full extraction: {result_mixed_full[:3]}")
    
    # === STEP 4: Matrix Structure Deep Dive ===
    logger.info("\n=== STEP 4: Matrix Structure Deep Dive ===")
    
    # Let's understand what the mixed tensor is actually storing
    logger.info("Analyzing mixed tensor block structure...")
    
    # Check a specific LED's diffusion patterns
    test_led = 0
    for c in range(3):
        if mixed_tensor.blocks_set[c, test_led]:
            pos = mixed_tensor.block_positions[c, test_led].get()
            values = mixed_tensor.sparse_values[c, test_led].get()
            logger.info(f"LED {test_led}, channel {c}: position {pos}, values range [{values.min():.3f}, {values.max():.3f}]")
    
    # Check the corresponding CSC columns
    for c in range(3):
        col_idx = test_led * 3 + c
        col_data = clipped_csc[:, col_idx].toarray().flatten()
        nonzero_count = np.count_nonzero(col_data)
        if nonzero_count > 0:
            logger.info(f"CSC LED {test_led}, channel {c}: {nonzero_count} non-zeros, range [{col_data.min():.3f}, {col_data.max():.3f}]")
    
    # === STEP 5: Compare Differences ===
    logger.info("\n=== STEP 5: Difference Analysis ===")
    
    diff_diagonal = np.abs(result_csc - result_mixed_diagonal)
    diff_full = np.abs(result_csc - result_mixed_full)
    
    logger.info(f"CSC vs Mixed (diagonal) - Max diff: {diff_diagonal.max():.6f}, Mean diff: {diff_diagonal.mean():.6f}")
    logger.info(f"CSC vs Mixed (full) - Max diff: {diff_full.max():.6f}, Mean diff: {diff_full.mean():.6f}")
    
    # === STEP 6: Block Diagonal Theory Check ===
    logger.info("\n=== STEP 6: Block Diagonal Theory Check ===")
    
    # The mixed tensor should be equivalent to a block diagonal CSC matrix
    # Let's verify this by creating a block diagonal version manually
    
    # Create block diagonal matrix from mixed tensor
    total_pixels = height * width
    block_diag_data = []
    block_diag_row = []
    block_diag_col = []
    
    for led in range(led_count):
        for c in range(3):
            if mixed_tensor.blocks_set[c, led]:
                # Get block data and position
                block_values = mixed_tensor.sparse_values[c, led].get()
                top_row, top_col = mixed_tensor.block_positions[c, led].get()
                
                # Convert block to pixel indices
                for r in range(block_size):
                    for col in range(block_size):
                        if top_row + r < height and top_col + col < width:
                            pixel_idx = (top_row + r) * width + (top_col + col)
                            col_idx = led * 3 + c
                            value = block_values[r, col]
                            
                            if abs(value) > 1e-10:
                                block_diag_data.append(value)
                                block_diag_row.append(pixel_idx)
                                block_diag_col.append(col_idx)
    
    block_diag_matrix = sp.coo_matrix(
        (block_diag_data, (block_diag_row, block_diag_col)),
        shape=(total_pixels, led_count * 3),
        dtype=np.float32
    ).tocsc()
    
    logger.info(f"Block diagonal matrix shape: {block_diag_matrix.shape}")
    logger.info(f"Block diagonal NNZ: {block_diag_matrix.nnz}")
    logger.info(f"Original CSC NNZ: {clipped_csc.nnz}")
    
    # Compare matrices
    if block_diag_matrix.shape == clipped_csc.shape:
        matrix_diff = (block_diag_matrix - clipped_csc).toarray()
        max_matrix_diff = np.abs(matrix_diff).max()
        logger.info(f"Matrix difference (block diag vs CSC): {max_matrix_diff:.2e}")
        
        if max_matrix_diff < 1e-6:
            logger.info("✅ Matrices are identical - issue is in A^T@b calculation")
        else:
            logger.error("❌ Matrices differ - issue is in data storage")
            
            # Find where they differ
            diff_locations = np.where(np.abs(matrix_diff) > 1e-6)
            if len(diff_locations[0]) > 0:
                sample_row = diff_locations[0][0]
                sample_col = diff_locations[1][0]
                logger.error(f"Sample difference at ({sample_row}, {sample_col}): "
                           f"Block={block_diag_matrix[sample_row, sample_col]:.6f}, "
                           f"CSC={clipped_csc[sample_row, sample_col]:.6f}")
    
    return True


if __name__ == "__main__":
    success = debug_algorithm_step_by_step()
    sys.exit(0 if success else 1)