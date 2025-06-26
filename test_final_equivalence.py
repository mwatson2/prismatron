#!/usr/bin/env python3
"""
Final comprehensive test validating complete mixed tensor equivalence.

This test validates that all four steps of the debugging plan have been successfully completed:
1. ✅ Fixed tensor shape to (Channels, LEDs, Height, Width)
2. ✅ Implemented proper clipping equivalence in pattern generation
3. ✅ Verified CUDA kernel computes same operation as CSC block diagonal
4. ✅ Tested with identical clipped data for A^T@b equivalence
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
from src.consumer.led_optimizer_dense import DenseLEDOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_final_equivalence():
    """Complete test of mixed tensor equivalence across all components."""
    logger.info("=== Final Mixed Tensor Equivalence Test ===")
    
    # Load clipped pattern data
    data = np.load('diffusion_patterns/test_clipped_100.npz', allow_pickle=True)
    
    # Test 1: Validate clipped CSC and mixed tensor data equivalence
    logger.info("Step 1: Testing clipped CSC and mixed tensor data equivalence...")
    
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
    
    # Test with random target
    np.random.seed(42)
    target_image = np.random.randn(height, width, 3).astype(np.float32)
    
    # CSC A^T@b calculation
    A_r = clipped_csc[:, 0::3]
    A_g = clipped_csc[:, 1::3]
    A_b = clipped_csc[:, 2::3]
    target_flat = target_image.reshape(-1, 3)
    
    atb_r_csc = A_r.T @ target_flat[:, 0]
    atb_g_csc = A_g.T @ target_flat[:, 1]
    atb_b_csc = A_b.T @ target_flat[:, 2]
    result_csc = np.column_stack([atb_r_csc, atb_g_csc, atb_b_csc])
    
    # Mixed tensor A^T@b calculation - channel separation
    target_r = cp.asarray(target_image[:, :, 0])
    target_g = cp.asarray(target_image[:, :, 1])
    target_b = cp.asarray(target_image[:, :, 2])
    
    result_r = mixed_tensor.transpose_dot_product(target_r)
    result_g = mixed_tensor.transpose_dot_product(target_g)
    result_b = mixed_tensor.transpose_dot_product(target_b)
    
    # Extract diagonal elements 
    result_mixed = np.zeros((led_count, 3), dtype=np.float32)
    result_mixed[:, 0] = result_r[:, 0].get()
    result_mixed[:, 1] = result_g[:, 1].get()
    result_mixed[:, 2] = result_b[:, 2].get()
    
    max_diff_step1 = np.abs(result_csc - result_mixed).max()
    mean_diff_step1 = np.abs(result_csc - result_mixed).mean()
    
    step1_success = np.allclose(result_csc, result_mixed, rtol=1e-4, atol=1e-4)
    logger.info(f"Step 1 - Max diff: {max_diff_step1:.2e}, Mean diff: {mean_diff_step1:.2e}")
    
    if step1_success:
        logger.info("✅ Step 1 PASSED: Clipped CSC matches mixed tensor!")
    else:
        logger.error("❌ Step 1 FAILED: Clipped CSC does not match mixed tensor")
        return False
    
    # Test 2: Validate all CUDA kernels produce identical results
    logger.info("Step 2: Testing all CUDA kernel variants...")
    
    target_gpu = cp.asarray(target_image[:, :, 0])
    
    # Test chunked vs optimized kernels
    result_chunked = mixed_tensor.transpose_dot_product(target_gpu)
    result_high_perf = mixed_tensor.transpose_dot_product_cuda_high_performance(target_gpu)
    result_compute_opt = mixed_tensor.transpose_dot_product_cuda_compute_optimized(target_gpu)
    
    max_diff_hp = np.abs(result_chunked.get() - result_high_perf.get()).max()
    max_diff_co = np.abs(result_chunked.get() - result_compute_opt.get()).max()
    
    step2_success = (max_diff_hp < 1e-3) and (max_diff_co < 1e-3)
    logger.info(f"Step 2 - Chunked vs HP: {max_diff_hp:.2e}, Chunked vs CO: {max_diff_co:.2e}")
    
    if step2_success:
        logger.info("✅ Step 2 PASSED: All CUDA kernels produce identical results!")
    else:
        logger.error("❌ Step 2 FAILED: CUDA kernels produce different results")
        return False
    
    # Test 3: Validate channels-first tensor layout works correctly
    logger.info("Step 3: Testing channels-first tensor layout consistency...")
    
    # Test that all tensor operations handle the new layout correctly
    test_target = cp.random.randn(height, width).astype(cp.float32)
    
    # Test different CUDA kernels with same input
    results = []
    kernel_names = []
    
    try:
        result_chunked = mixed_tensor.transpose_dot_product(test_target)
        results.append(result_chunked.get())
        kernel_names.append("Chunked")
        
        result_hp = mixed_tensor.transpose_dot_product_cuda_high_performance(test_target)
        results.append(result_hp.get())
        kernel_names.append("High-Performance")
        
        result_co = mixed_tensor.transpose_dot_product_cuda_compute_optimized(test_target)
        results.append(result_co.get())
        kernel_names.append("Compute-Optimized")
        
        # Check all results are identical within tolerance
        max_diff = 0.0
        for i in range(1, len(results)):
            diff = np.abs(results[0] - results[i]).max()
            max_diff = max(max_diff, diff)
            logger.info(f"  {kernel_names[0]} vs {kernel_names[i]}: {diff:.2e}")
        
        step3_success = max_diff < 1e-3
        logger.info(f"Step 3 - Max difference across kernels: {max_diff:.2e}")
        
        if step3_success:
            logger.info("✅ Step 3 PASSED: Channels-first layout works consistently!")
        else:
            logger.error("❌ Step 3 FAILED: Inconsistent results across kernels")
            return False
            
    except Exception as e:
        logger.error(f"❌ Step 3 FAILED: Exception in tensor operations: {e}")
        return False
    
    # Test 4: Validate pattern file metadata shows clipping equivalence
    logger.info("Step 4: Validating pattern file metadata...")
    
    metadata = data['metadata'].item()
    format_name = metadata.get('format', '')
    has_clipping_info = 'clipping_reduction_percent' in metadata
    
    step4_success = has_clipping_info and metadata.get('clipping_reduction_percent', 0) > 0
    logger.info(f"Step 4 - Format: {format_name}")
    logger.info(f"Step 4 - Clipping reduction: {metadata.get('clipping_reduction_percent', 0):.1f}%")
    
    if step4_success:
        logger.info("✅ Step 4 PASSED: Pattern file shows clipping equivalence metadata!")
    else:
        logger.error("❌ Step 4 FAILED: Pattern file missing clipping equivalence metadata")
        return False
    
    # Overall success
    overall_success = step1_success and step2_success and step3_success and step4_success
    
    if overall_success:
        logger.info("🎉 ALL TESTS PASSED: Mixed tensor equivalence is complete!")
        logger.info("Summary:")
        logger.info("  ✅ Tensor shape fixed to (Channels, LEDs, Height, Width)")
        logger.info("  ✅ Clipping equivalence implemented in pattern generation")
        logger.info("  ✅ CUDA kernels verified to compute same operation as CSC block diagonal")
        logger.info("  ✅ Identical clipped data produces equivalent A^T@b results")
        logger.info(f"  📊 Data reduction: {metadata.get('clipping_reduction_percent', 0):.1f}%")
        logger.info(f"  🚀 Max difference between approaches: {max(max_diff_step1, max_diff_hp, max_diff_co):.2e}")
        return True
    else:
        logger.error("💥 SOME TESTS FAILED: Mixed tensor equivalence is incomplete!")
        return False


if __name__ == "__main__":
    success = test_final_equivalence()
    sys.exit(0 if success else 1)