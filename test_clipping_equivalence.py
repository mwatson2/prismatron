#!/usr/bin/env python3
"""
Test clipping equivalence between CSC and mixed tensor formats.

This test applies the same 96x96 clipping to CSC matrices as used in mixed tensor
to verify that both approaches compute exactly the same A^T@b result.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_mixed_tensor_clipping_to_csc(sparse_matrix_csc, mixed_tensor_data):
    """
    Apply the same 96x96 clipping to CSC matrix as used in mixed tensor.
    
    Args:
        sparse_matrix_csc: Original CSC matrix (384000, 3000)
        mixed_tensor_data: Mixed tensor data from patterns file
        
    Returns:
        Clipped CSC matrix with same effective data as mixed tensor
    """
    logger.info("Applying mixed tensor clipping to CSC matrix...")
    
    # Get mixed tensor metadata
    positions = mixed_tensor_data['mixed_tensor_positions']  # (1000, 3, 2)
    blocks_set = mixed_tensor_data['mixed_tensor_blocks_set']  # (1000, 3)
    block_size = int(mixed_tensor_data['mixed_tensor_block_size'])
    
    # Create a copy of the CSC matrix to modify
    clipped_csc = sparse_matrix_csc.copy()
    clipped_csc = clipped_csc.tolil()  # Convert to LIL for efficient modification
    
    logger.info(f"Original CSC matrix shape: {clipped_csc.shape}")
    logger.info(f"Block size: {block_size}")
    
    # Process each LED and channel
    led_count = sparse_matrix_csc.shape[1] // 3
    modified_count = 0
    
    for led_id in range(led_count):
        for channel in range(3):
            col_idx = led_id * 3 + channel  # Column in interleaved format
            
            if not blocks_set[led_id, channel]:
                # No block set for this LED/channel - zero out the entire column
                clipped_csc[:, col_idx] = 0
                continue
                
            # Get the 96x96 block position used in mixed tensor
            top_row, top_col = positions[led_id, channel]
            
            # Define the 96x96 clipping region
            bottom_row = top_row + block_size
            right_col = top_col + block_size
            
            # More efficient: get all non-zero entries for this column and filter them
            col_data = clipped_csc[:, col_idx]
            col_indices = col_data.nonzero()[0]
            
            if len(col_indices) > 0:
                # Convert linear indices to 2D coordinates
                row_coords = col_indices // FRAME_WIDTH
                col_coords = col_indices % FRAME_WIDTH
                
                # Find indices outside the 96x96 block
                outside_mask = ~((top_row <= row_coords) & (row_coords < bottom_row) & 
                               (top_col <= col_coords) & (col_coords < right_col))
                
                # Zero out pixels outside the block
                outside_indices = col_indices[outside_mask]
                for idx in outside_indices:
                    clipped_csc[idx, col_idx] = 0
            
            modified_count += 1
            
            if modified_count % 100 == 0:
                logger.info(f"Processed {modified_count}/{led_count * 3} columns")
    
    # Convert back to CSC format
    clipped_csc = clipped_csc.tocsc()
    clipped_csc.eliminate_zeros()
    
    logger.info(f"Clipped CSC matrix shape: {clipped_csc.shape}")
    logger.info(f"Original NNZ: {sparse_matrix_csc.nnz}, Clipped NNZ: {clipped_csc.nnz}")
    
    return clipped_csc


def test_clipping_equivalence():
    """Test that clipped CSC and mixed tensor produce identical A^T@b results."""
    logger.info("=== Testing Clipping Equivalence ===")
    
    # Load pattern data with clipping equivalence
    data = np.load('diffusion_patterns/test_clipped_100.npz')
    
    # Reconstruct original CSC matrix
    matrix_shape = tuple(data['matrix_shape'])
    matrix_data = data['matrix_data']
    matrix_indices = data['matrix_indices'] 
    matrix_indptr = data['matrix_indptr']
    
    sparse_matrix_csc = sp.csc_matrix((matrix_data, matrix_indices, matrix_indptr), 
                                      shape=matrix_shape)
    logger.info(f"Loaded CSC matrix shape: {sparse_matrix_csc.shape}")
    
    # Apply clipping to match mixed tensor
    clipped_csc = apply_mixed_tensor_clipping_to_csc(sparse_matrix_csc, data)
    
    # Extract RGB channel matrices from both original and clipped
    logger.info("Extracting RGB channels from original CSC...")
    A_r_orig = sparse_matrix_csc[:, 0::3]  # Red channels
    A_g_orig = sparse_matrix_csc[:, 1::3]  # Green channels  
    A_b_orig = sparse_matrix_csc[:, 2::3]  # Blue channels
    
    logger.info("Extracting RGB channels from clipped CSC...")
    A_r_clipped = clipped_csc[:, 0::3]  # Red channels
    A_g_clipped = clipped_csc[:, 1::3]  # Green channels
    A_b_clipped = clipped_csc[:, 2::3]  # Blue channels
    
    # Create test target image
    np.random.seed(42)
    target_image = np.random.randn(FRAME_HEIGHT, FRAME_WIDTH, 3).astype(np.float32)
    target_flat = target_image.reshape(-1, 3)  # (pixels, 3)
    
    # Test A^T@b calculation with both approaches
    logger.info("Computing A^T@b with original CSC...")
    atb_r_orig = A_r_orig.T @ target_flat[:, 0]
    atb_g_orig = A_g_orig.T @ target_flat[:, 1]
    atb_b_orig = A_b_orig.T @ target_flat[:, 2]
    result_orig = np.column_stack([atb_r_orig, atb_g_orig, atb_b_orig])
    
    logger.info("Computing A^T@b with clipped CSC...")
    atb_r_clipped = A_r_clipped.T @ target_flat[:, 0]
    atb_g_clipped = A_g_clipped.T @ target_flat[:, 1]
    atb_b_clipped = A_b_clipped.T @ target_flat[:, 2]
    result_clipped = np.column_stack([atb_r_clipped, atb_g_clipped, atb_b_clipped])
    
    # Load and test mixed tensor
    logger.info("Loading mixed tensor...")
    from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
    
    led_count = int(data['mixed_tensor_led_count'])
    channels = int(data['mixed_tensor_channels'])
    height = int(data['mixed_tensor_height'])
    width = int(data['mixed_tensor_width'])
    block_size = int(data['mixed_tensor_block_size'])
    
    mixed_tensor = SingleBlockMixedSparseTensor(led_count, channels, height, width, block_size)
    mixed_tensor.sparse_values = cp.asarray(data['mixed_tensor_values'])
    mixed_tensor.block_positions = cp.asarray(data['mixed_tensor_positions'])
    mixed_tensor.blocks_set = cp.asarray(data['mixed_tensor_blocks_set'])
    
    logger.info("Computing A^T@b with mixed tensor...")
    target_gpu = cp.asarray(target_image[:, :, 0])  # Test with red channel
    result_mixed_r = mixed_tensor.transpose_dot_product_cuda_high_performance(target_gpu)
    
    target_gpu = cp.asarray(target_image[:, :, 1])  # Test with green channel
    result_mixed_g = mixed_tensor.transpose_dot_product_cuda_high_performance(target_gpu)
    
    target_gpu = cp.asarray(target_image[:, :, 2])  # Test with blue channel
    result_mixed_b = mixed_tensor.transpose_dot_product_cuda_high_performance(target_gpu)
    
    # Extract diagonal elements to match CSC approach
    result_mixed = np.zeros((led_count, 3), dtype=np.float32)
    result_mixed[:, 0] = result_mixed_r[:, 0].get()  # Red LED values from red channel
    result_mixed[:, 1] = result_mixed_g[:, 1].get()  # Green LED values from green channel
    result_mixed[:, 2] = result_mixed_b[:, 2].get()  # Blue LED values from blue channel
    
    # Compare results
    logger.info("=== Comparison Results ===")
    logger.info(f"Original CSC result shape: {result_orig.shape}")
    logger.info(f"Clipped CSC result shape: {result_clipped.shape}")
    logger.info(f"Mixed tensor result shape: {result_mixed.shape}")
    
    # Check if clipped CSC matches mixed tensor
    if np.allclose(result_clipped, result_mixed, rtol=1e-5, atol=1e-6):
        logger.info("✅ SUCCESS: Clipped CSC matches mixed tensor!")
        return True
    else:
        logger.error("❌ MISMATCH: Clipped CSC does not match mixed tensor")
        logger.error(f"Max difference: {np.abs(result_clipped - result_mixed).max()}")
        logger.error(f"Mean difference: {np.abs(result_clipped - result_mixed).mean()}")
        
        # Show some sample differences
        logger.error("Sample differences (first 5 LEDs):")
        for i in range(5):
            logger.error(f"LED {i}: Clipped={result_clipped[i]}, Mixed={result_mixed[i]}")
        
        return False


if __name__ == "__main__":
    success = test_clipping_equivalence()
    sys.exit(0 if success else 1)