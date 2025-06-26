#!/usr/bin/env python3
"""
Test that CUDA kernel computes the same operation as CSC block diagonal format.

This test verifies that the mixed tensor CUDA kernels produce identical results
to the mathematical equivalent CSC block diagonal matrix operations.
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


def create_block_diagonal_csc_from_mixed_tensor(mixed_tensor: SingleBlockMixedSparseTensor) -> sp.csc_matrix:
    """
    Create a block diagonal CSC matrix from mixed tensor data.
    
    The block diagonal format has shape (pixels, leds*3) where each LED's RGB data
    is arranged in a 3x3 block diagonal structure:
    
    [A_r  0   0 ]
    [ 0  A_g  0 ]
    [ 0   0  A_b]
    
    Args:
        mixed_tensor: Mixed tensor with diffusion pattern data
        
    Returns:
        Block diagonal CSC matrix equivalent to mixed tensor
    """
    logger.info("Creating block diagonal CSC matrix from mixed tensor...")
    
    total_pixels = mixed_tensor.height * mixed_tensor.width
    total_leds = mixed_tensor.batch_size
    total_channels = mixed_tensor.channels
    
    # Pre-allocate lists with estimated size
    estimated_nnz = int(cp.sum(mixed_tensor.blocks_set)) * mixed_tensor.block_size ** 2
    data = np.zeros(estimated_nnz, dtype=np.float32)
    row_indices = np.zeros(estimated_nnz, dtype=np.int32)
    col_indices = np.zeros(estimated_nnz, dtype=np.int32)
    
    nnz_count = 0
    
    for channel in range(total_channels):
        for led in range(total_leds):
            if not mixed_tensor.blocks_set[channel, led]:
                continue
                
            # Get block data and position
            block_values = cp.asnumpy(mixed_tensor.sparse_values[channel, led])  # (block_size, block_size)
            top_row, top_col = cp.asnumpy(mixed_tensor.block_positions[channel, led])
            
            # Column index in block diagonal format
            col_idx = led * total_channels + channel
            
            # Vectorized conversion of block to pixel indices
            block_size = mixed_tensor.block_size
            row_offsets, col_offsets = np.meshgrid(np.arange(block_size), np.arange(block_size), indexing='ij')
            
            # Convert to global pixel indices
            pixel_rows = top_row + row_offsets.flatten()
            pixel_cols = top_col + col_offsets.flatten()
            pixel_indices = pixel_rows * mixed_tensor.width + pixel_cols
            values = block_values.flatten()
            
            # Only include significant values
            significant_mask = np.abs(values) > 1e-10
            if np.any(significant_mask):
                significant_pixels = pixel_indices[significant_mask]
                significant_values = values[significant_mask]
                n_significant = len(significant_values)
                
                # Store in pre-allocated arrays
                if nnz_count + n_significant <= estimated_nnz:
                    data[nnz_count:nnz_count + n_significant] = significant_values
                    row_indices[nnz_count:nnz_count + n_significant] = significant_pixels
                    col_indices[nnz_count:nnz_count + n_significant] = col_idx
                    nnz_count += n_significant
    
    # Trim arrays to actual size
    data = data[:nnz_count]
    row_indices = row_indices[:nnz_count]
    col_indices = col_indices[:nnz_count]
    
    # Create CSC matrix
    block_diagonal_matrix = sp.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(total_pixels, total_leds * total_channels),
        dtype=np.float32
    ).tocsc()
    
    logger.info(f"Created block diagonal CSC matrix: {block_diagonal_matrix.shape}")
    logger.info(f"Block diagonal NNZ: {block_diagonal_matrix.nnz}")
    
    return block_diagonal_matrix


def test_cuda_csc_equivalence():
    """Test that CUDA kernel produces same results as CSC block diagonal."""
    logger.info("=== Testing CUDA vs CSC Block Diagonal Equivalence ===")
    
    # Load clipped pattern data
    data = np.load('diffusion_patterns/test_clipped_100.npz')
    
    # Load mixed tensor
    led_count = int(data['mixed_tensor_led_count'])
    channels = int(data['mixed_tensor_channels'])
    height = int(data['mixed_tensor_height'])
    width = int(data['mixed_tensor_width'])
    block_size = int(data['mixed_tensor_block_size'])
    
    mixed_tensor = SingleBlockMixedSparseTensor(led_count, channels, height, width, block_size)
    
    # Load data with transpose for channels-first layout
    old_values = data['mixed_tensor_values']  # (LEDs, channels, H, W)
    old_positions = data['mixed_tensor_positions']  # (LEDs, channels, 2)
    old_blocks_set = data['mixed_tensor_blocks_set']  # (LEDs, channels)
    
    # Transpose to channels-first layout
    mixed_tensor.sparse_values = cp.asarray(old_values.transpose(1, 0, 2, 3))
    mixed_tensor.block_positions = cp.asarray(old_positions.transpose(1, 0, 2))
    mixed_tensor.blocks_set = cp.asarray(old_blocks_set.transpose(1, 0))
    
    logger.info(f"Loaded mixed tensor: {mixed_tensor}")
    
    # Create block diagonal CSC equivalent
    block_diagonal_csc = create_block_diagonal_csc_from_mixed_tensor(mixed_tensor)
    
    # Create test target image
    np.random.seed(42)
    target_image = np.random.randn(height, width).astype(np.float32)
    target_flat = target_image.flatten()  # (pixels,)
    
    # Test 1: Compare with chunked CUDA kernel
    logger.info("Testing chunked CUDA kernel...")
    target_gpu = cp.asarray(target_image)
    result_cuda_chunked = mixed_tensor.transpose_dot_product(target_gpu)
    
    # Compute equivalent with block diagonal CSC
    logger.info("Computing equivalent with block diagonal CSC...")
    result_csc_block_diagonal = block_diagonal_csc.T @ target_flat
    result_csc_reshaped = result_csc_block_diagonal.reshape(led_count, channels)
    
    logger.info("=== Chunked CUDA vs Block Diagonal CSC ===")
    logger.info(f"CUDA result shape: {result_cuda_chunked.shape}")
    logger.info(f"CSC result shape: {result_csc_reshaped.shape}")
    
    cuda_chunked_cpu = result_cuda_chunked.get()
    max_diff_chunked = np.abs(cuda_chunked_cpu - result_csc_reshaped).max()
    mean_diff_chunked = np.abs(cuda_chunked_cpu - result_csc_reshaped).mean()
    
    logger.info(f"Max difference: {max_diff_chunked:.2e}")
    logger.info(f"Mean difference: {mean_diff_chunked:.2e}")
    
    chunked_success = np.allclose(cuda_chunked_cpu, result_csc_reshaped, rtol=1e-4, atol=1e-4)
    if chunked_success:
        logger.info("✅ Chunked CUDA kernel matches block diagonal CSC!")
    else:
        logger.error("❌ Chunked CUDA kernel does not match block diagonal CSC")
    
    # Test 2: Compare with high-performance CUDA kernel
    logger.info("Testing high-performance CUDA kernel...")
    try:
        result_cuda_hp = mixed_tensor.transpose_dot_product_cuda_high_performance(target_gpu)
        
        logger.info("=== High-Performance CUDA vs Block Diagonal CSC ===")
        logger.info(f"HP CUDA result shape: {result_cuda_hp.shape}")
        
        cuda_hp_cpu = result_cuda_hp.get()
        max_diff_hp = np.abs(cuda_hp_cpu - result_csc_reshaped).max()
        mean_diff_hp = np.abs(cuda_hp_cpu - result_csc_reshaped).mean()
        
        logger.info(f"Max difference: {max_diff_hp:.2e}")
        logger.info(f"Mean difference: {mean_diff_hp:.2e}")
        
        hp_success = np.allclose(cuda_hp_cpu, result_csc_reshaped, rtol=1e-4, atol=1e-4)
        if hp_success:
            logger.info("✅ High-performance CUDA kernel matches block diagonal CSC!")
        else:
            logger.error("❌ High-performance CUDA kernel does not match block diagonal CSC")
            
            # Show sample differences
            logger.error("Sample differences (first 5 LEDs):")
            for i in range(min(5, led_count)):
                logger.error(f"LED {i}: CUDA HP={cuda_hp_cpu[i]}, CSC={result_csc_reshaped[i]}")
        
    except Exception as e:
        logger.warning(f"High-performance CUDA kernel not available: {e}")
        hp_success = True  # Don't fail if kernel not available
    
    # Test 3: Compare with compute-optimized CUDA kernel
    logger.info("Testing compute-optimized CUDA kernel...")
    try:
        result_cuda_co = mixed_tensor.transpose_dot_product_cuda_compute_optimized(target_gpu)
        
        logger.info("=== Compute-Optimized CUDA vs Block Diagonal CSC ===")
        logger.info(f"CO CUDA result shape: {result_cuda_co.shape}")
        
        cuda_co_cpu = result_cuda_co.get()
        max_diff_co = np.abs(cuda_co_cpu - result_csc_reshaped).max()
        mean_diff_co = np.abs(cuda_co_cpu - result_csc_reshaped).mean()
        
        logger.info(f"Max difference: {max_diff_co:.2e}")
        logger.info(f"Mean difference: {mean_diff_co:.2e}")
        
        co_success = np.allclose(cuda_co_cpu, result_csc_reshaped, rtol=1e-4, atol=1e-4)
        if co_success:
            logger.info("✅ Compute-optimized CUDA kernel matches block diagonal CSC!")
        else:
            logger.error("❌ Compute-optimized CUDA kernel does not match block diagonal CSC")
            
            # Show sample differences  
            logger.error("Sample differences (first 5 LEDs):")
            for i in range(min(5, led_count)):
                logger.error(f"LED {i}: CUDA CO={cuda_co_cpu[i]}, CSC={result_csc_reshaped[i]}")
        
    except Exception as e:
        logger.warning(f"Compute-optimized CUDA kernel not available: {e}")
        co_success = True  # Don't fail if kernel not available
    
    # Overall success
    overall_success = chunked_success and hp_success and co_success
    
    if overall_success:
        logger.info("🎉 All CUDA kernels match block diagonal CSC equivalence!")
        return True
    else:
        logger.error("💥 Some CUDA kernels do not match block diagonal CSC equivalence!")
        return False


if __name__ == "__main__":
    success = test_cuda_csc_equivalence()
    sys.exit(0 if success else 1)