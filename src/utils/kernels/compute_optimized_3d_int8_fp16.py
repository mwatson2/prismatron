"""
Compute-optimized 3D CUDA kernel for int8 inputs with FP16 output.

This module implements the int8 version of the optimized CUDA kernel for LED optimization operations
with FP16 output for memory efficiency, specifically the A^T @ b matrix multiplication for mixed sparse tensors.
"""

import logging
from typing import Tuple

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)

# 3D Multi-channel CUDA kernel for int8 inputs with fp16 output (experimental aligned version)
COMPUTE_OPTIMIZED_3D_INT8_EXPERIMENTAL_FP16_KERNEL = r"""
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
using namespace cooperative_groups;

__device__ int dot_int8(uchar4 a, uchar4 b) {
    // Compute dot product of two uchar4 vectors as integers
    // Shape: (4,) + (4,) -> scalar
    return (int)a.x * (int)b.x + (int)a.y * (int)b.y + (int)a.z * (int)b.z + (int)a.w * (int)b.w;
}

extern "C" __global__
void compute_optimized_3d_transpose_dot_product_int8_experimental_fp16_kernel(
    const unsigned char* sparse_values,  // (channels, batch_size, block_size, block_size) - int8
    const int* block_positions,          // Shape: (channels, batch_size, 2)
    const unsigned char* target_3d,      // Shape: (channels, height, width) - int8 planar form
    __half* result,                      // Shape: (batch_size, channels) - fp16 output
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int block_size
) {
    // EXPERIMENTAL: Aligned uchar4 loads for both sparse and target data
    // Requirements:
    // - block_size % 4 == 0 (for vectorized uchar4 access)
    // - LED x-positions must be multiples of 4 for aligned target access
    // Grid: (batch_size, channels) - one block per (LED, channel) combination
    // Block: (32 threads, 1, 1) - each thread processes block_size/32 complete rows

    auto warp = tiled_partition<32>(this_thread_block());

    int led_id = blockIdx.x;
    int channel_id = blockIdx.y;

    // Bounds check
    if (led_id >= batch_size || channel_id >= channels) return;

    int idx = led_id * channels + channel_id;

    // Get block position for this LED/channel from (channels, batch_size, 2) layout
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Row-based processing: each thread handles exactly rows_per_thread rows
    int rows_per_thread = block_size / 32;  // e.g., 64/32 = 2 rows per thread
    int block_elements = block_size * block_size;  // 4096 for 64x64

    // Direct computation with row-based iteration (eliminates bounds checks)
    // Use int for accumulation to avoid overflow, then convert to fp16
    int thread_sum_int = 0;

    // Calculate base offset for sparse image: (top left)
    int sparse_offset = channel_id * (batch_size * block_elements) + led_id * block_elements;

    // Calculate base offset for target image (top left)
    int target_offset = channel_id * height * width + top_row * width + top_col;

    // Initialize row variables outside loop for efficiency
    int thread_row = threadIdx.x * rows_per_thread;     // Thread's starting row
    int max_thread_row = thread_row + rows_per_thread;  // End row for this thread
    int sparse_idx = sparse_offset + thread_row * block_size; // Starting index for first row
    int target_idx = target_offset + thread_row * width;

    // Each thread processes rows_per_thread complete rows
    for (   ;
            thread_row < max_thread_row;
            thread_row++, target_idx += width - block_size) {
        // Process all columns in this row with aligned vectorized int8 operations
        for (int col = 0;
             col < block_size;
             col += 4, target_idx += 4, sparse_idx += 4) {
            uchar4 sparse_vec = *reinterpret_cast<const uchar4*>(
                &sparse_values[sparse_idx]
            );
            uchar4 target_vec = *reinterpret_cast<const uchar4*>(
                &target_3d[target_idx]
            );
            thread_sum_int += dot_int8(sparse_vec, target_vec);
        }
    }

    // Convert to float and perform warp reduction
    float thread_sum = (float)thread_sum_int;
    float sum = reduce(warp, thread_sum, plus<float>());

    if (warp.thread_rank() == 0) {
        // Normalize by (255 * 255) to align with fp32 equivalent operation
        // int8 range [0,255] × [0,255] -> fp32 range [0,1]
        // Convert to FP16 for output
        float normalized_sum = sum / (255.0f * 255.0f);
        result[idx] = __float2half(normalized_sum);
    }
}
"""

# 3D Multi-channel CUDA kernel for int8 inputs with fp16 output (aligned optimized version)
COMPUTE_OPTIMIZED_3D_INT8_FP16_KERNEL = r"""
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
using namespace cooperative_groups;

__device__ int dot_int8(uchar4 a, uchar4 b) {
    // Compute dot product of two uchar4 vectors as integers
    // Shape: (4,) + (4,) -> scalar
    return (int)a.x * (int)b.x + (int)a.y * (int)b.y + (int)a.z * (int)b.z + (int)a.w * (int)b.w;
}

extern "C" __global__
void compute_optimized_3d_transpose_dot_product_int8_fp16_kernel(
    const unsigned char* sparse_values,  // (channels, batch_size, block_size, block_size) - int8
    const int* block_positions,          // Shape: (channels, batch_size, 2)
    const unsigned char* target_3d,      // Shape: (channels, height, width) - int8 planar form
    __half* result,                      // Shape: (batch_size, channels) - fp16 output
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int block_size
) {
    // Row-based processing optimization with aligned uchar4 loads
    // Requires:
    // - block_size % 4 == 0 (for vectorized uchar4 access)
    // - LED x-positions should be multiples of 4 for optimal performance
    // Grid: (batch_size, channels) - one block per (LED, channel) combination
    // Block: (32 threads, 1, 1) - each thread processes block_size/32 complete rows

    auto warp = tiled_partition<32>(this_thread_block());

    int led_id = blockIdx.x;
    int channel_id = blockIdx.y;

    // Bounds check
    if (led_id >= batch_size || channel_id >= channels) return;

    int idx = led_id * channels + channel_id;

    // Get block position for this LED/channel from (channels, batch_size, 2) layout
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Row-based processing: each thread handles exactly rows_per_thread rows
    int rows_per_thread = block_size / 32;  // e.g., 64/32 = 2 rows per thread
    int block_elements = block_size * block_size;  // 4096 for 64x64

    // Direct computation with row-based iteration (eliminates bounds checks)
    // Use int for accumulation to avoid overflow, then convert to fp16
    int thread_sum_int = 0;

    // Calculate base offset for sparse image: (top left)
    int sparse_offset = channel_id * (batch_size * block_elements) + led_id * block_elements;

    // Calculate base offset for target image (top left)
    int target_offset = channel_id * height * width + top_row * width + top_col;

    // Initialize row variables outside loop for efficiency
    int thread_row = threadIdx.x * rows_per_thread;     // Thread's starting row
    int max_thread_row = thread_row + rows_per_thread;  // End row for this thread
    int sparse_idx = sparse_offset + thread_row * block_size; // Starting index for first row
    int target_idx = target_offset + thread_row * width;

    // Each thread processes rows_per_thread complete rows
    for (   ;
            thread_row < max_thread_row;
            thread_row++, target_idx += width - block_size) {
        // Process all columns in this row with aligned vectorized int8 operations
        for (int col = 0;
             col < block_size;
             col += 4, target_idx += 4, sparse_idx += 4) {
            uchar4 sparse_vec = *reinterpret_cast<const uchar4*>(
                &sparse_values[sparse_idx]
            );
            uchar4 target_vec = *reinterpret_cast<const uchar4*>(
                &target_3d[target_idx]
            );
            thread_sum_int += dot_int8(sparse_vec, target_vec);
        }
    }

    // Convert to float and perform warp reduction
    float thread_sum = (float)thread_sum_int;
    float sum = reduce(warp, thread_sum, plus<float>());

    if (warp.thread_rank() == 0) {
        // Normalize by (255 * 255) to align with fp32 equivalent operation
        // int8 range [0,255] × [0,255] -> fp32 range [0,1]
        // Convert to FP16 for output
        float normalized_sum = sum / (255.0f * 255.0f);
        result[idx] = __float2half(normalized_sum);
    }
}
"""

# Compile the kernel once on module import
_kernel_cache = {}


def get_compute_optimized_3d_int8_fp16_kernel():
    """Get compiled compute-optimized 3D int8 CUDA kernel with aligned optimization and FP16 output."""
    if "compute_optimized_3d_int8_aligned_fp16" not in _kernel_cache:
        logger.info("Compiling aligned compute-optimized 3D int8 FP16 CUDA kernel...")

        # Compile the kernel
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_3D_INT8_FP16_KERNEL,
            "compute_optimized_3d_transpose_dot_product_int8_fp16_kernel",
        )
        _kernel_cache["compute_optimized_3d_int8_aligned_fp16"] = kernel

        logger.info("Aligned compute-optimized 3D int8 FP16 CUDA kernel compiled successfully")

    return _kernel_cache["compute_optimized_3d_int8_aligned_fp16"]


def get_compute_optimized_3d_int8_experimental_fp16_kernel():
    """Get the compiled experimental compute-optimized 3D int8 CUDA kernel with aligned loads and FP16 output."""
    if "compute_optimized_3d_int8_experimental_fp16" not in _kernel_cache:
        logger.info("Compiling experimental compute-optimized 3D int8 FP16 CUDA kernel...")

        # Compile the experimental kernel
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_3D_INT8_EXPERIMENTAL_FP16_KERNEL,
            "compute_optimized_3d_transpose_dot_product_int8_experimental_fp16_kernel",
        )
        _kernel_cache["compute_optimized_3d_int8_experimental_fp16"] = kernel

        logger.info("Experimental compute-optimized 3D int8 FP16 CUDA kernel compiled successfully")

    return _kernel_cache["compute_optimized_3d_int8_experimental_fp16"]


def cuda_transpose_dot_product_3d_compute_optimized_int8_fp16(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    target_3d: cp.ndarray,
    batch_size: int,
    channels: int,
    block_size: int,
) -> cp.ndarray:
    """
    3D Compute-optimized CUDA kernel wrapper for A^T @ b operation with int8 inputs and FP16 output.

    Uses row-based processing optimization with 2D grid launch for optimal performance.
    Processes int8 sparse values and target images, performs integer arithmetic,
    then converts to fp16 with proper scaling to match fp32 equivalent results.

    Args:
        sparse_values: Dense blocks, shape (channels, batch_size, block_size, block_size), uint8
        block_positions: Block positions, shape (channels, batch_size, 2)
        target_3d: Target image in planar form, shape (channels, height, width), dtype uint8
        batch_size: Number of LEDs
        channels: Number of channels
        block_size: Size of square blocks (must be multiple of 4)

    Returns:
        Result of A^T @ b, shape (batch_size, channels), dtype float16
        Values are normalized by (255*255) to align with fp32 range [0,1]
    """
    channels_input, height, width = target_3d.shape

    if channels_input != channels:
        raise ValueError(f"Input channels {channels_input} != expected {channels}")

    if sparse_values.dtype != cp.uint8:
        raise ValueError(f"sparse_values must be uint8, got {sparse_values.dtype}")

    if target_3d.dtype != cp.uint8:
        raise ValueError(f"target_3d must be uint8, got {target_3d.dtype}")

    if block_size % 4 != 0:
        raise ValueError(f"block_size {block_size} must be multiple of 4 for vectorization")

    # Prepare output array with FP16 dtype
    result = cp.zeros((batch_size, channels), dtype=cp.float16)

    # Get compiled kernel
    kernel = get_compute_optimized_3d_int8_fp16_kernel()

    # 2D Grid configuration: one block per (LED, channel) combination
    grid_size = (batch_size, channels)  # 2D grid: (batch_size, channels)
    block_size_1d = (32,)  # 32 threads per block for optimal warp utilization

    logger.debug("3D Compute-optimized int8 FP16 CUDA kernel launch (2D Grid):")
    logger.debug(f"  Grid: {grid_size} (batch_size={batch_size}, channels={channels})")
    logger.debug(f"  Block: {block_size_1d} (32 threads per block)")
    logger.debug(f"  Target shape: {target_3d.shape} (planar int8)")
    logger.debug(f"  Total blocks: {batch_size * channels:,}")
    logger.debug(f"  Total threads: {batch_size * channels * 32:,}")
    logger.debug("  Output dtype: FP16")

    # Calculate shared memory size needed
    # Only reduction workspace (32 floats) - no sparse block caching
    shared_mem_size = 32 * 4  # 4 bytes per float = 128 bytes

    logger.debug(f"  Shared memory per block: {shared_mem_size} bytes ({shared_mem_size / 1024:.3f}KB)")

    # Single kernel launch with 2D grid - let GPU scheduler handle work distribution
    kernel(
        grid_size,
        block_size_1d,
        (
            sparse_values.ravel(),  # Flatten to 1D uint8
            block_positions.ravel(),  # Flatten to 1D int32
            target_3d.ravel(),  # Flatten 3D planar int8 input to 1D
            result.ravel(),  # Flatten result to 1D fp16
            batch_size,
            channels,
            height,
            width,
            block_size,
        ),
        shared_mem=shared_mem_size,
    )

    return result


def cuda_transpose_dot_product_3d_compute_optimized_int8_experimental_fp16(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    target_3d: cp.ndarray,
    batch_size: int,
    channels: int,
    block_size: int,
) -> cp.ndarray:
    """
    Experimental 3D Compute-optimized CUDA kernel wrapper for A^T @ b operation with int8 inputs and FP16 output.

    This experimental version uses aligned uchar4 loads for both sparse and target data,
    requiring LED x-positions to be multiples of 4 for optimal memory access patterns.

    Args:
        sparse_values: Dense blocks, shape (channels, batch_size, block_size, block_size), uint8
        block_positions: Block positions, shape (channels, batch_size, 2)
        target_3d: Target image in planar form, shape (channels, height, width), dtype uint8
        batch_size: Number of LEDs
        channels: Number of channels
        block_size: Size of square blocks (must be multiple of 4)

    Returns:
        Result of A^T @ b, shape (batch_size, channels), dtype float16
        Values are normalized by (255*255) to align with fp32 range [0,1]
    """
    channels_input, height, width = target_3d.shape

    if channels_input != channels:
        raise ValueError(f"Input channels {channels_input} != expected {channels}")

    if sparse_values.dtype != cp.uint8:
        raise ValueError(f"sparse_values must be uint8, got {sparse_values.dtype}")

    if target_3d.dtype != cp.uint8:
        raise ValueError(f"target_3d must be uint8, got {target_3d.dtype}")

    if block_size % 4 != 0:
        raise ValueError(f"block_size {block_size} must be multiple of 4 for vectorization")

    # Prepare output array with FP16 dtype
    result = cp.zeros((batch_size, channels), dtype=cp.float16)

    # Get compiled experimental kernel
    kernel = get_compute_optimized_3d_int8_experimental_fp16_kernel()

    # 2D Grid configuration: one block per (LED, channel) combination
    grid_size = (batch_size, channels)  # 2D grid: (batch_size, channels)
    block_size_1d = (32,)  # 32 threads per block for optimal warp utilization

    logger.debug("Experimental 3D Compute-optimized int8 FP16 CUDA kernel launch (2D Grid):")
    logger.debug(f"  Grid: {grid_size} (batch_size={batch_size}, channels={channels})")
    logger.debug(f"  Block: {block_size_1d} (32 threads per block)")
    logger.debug(f"  Target shape: {target_3d.shape} (planar int8)")
    logger.debug(f"  Total blocks: {batch_size * channels:,}")
    logger.debug(f"  Total threads: {batch_size * channels * 32:,}")
    logger.debug("  Output dtype: FP16")

    # Calculate shared memory size needed
    # Only reduction workspace (32 floats) - no sparse block caching
    shared_mem_size = 32 * 4  # 4 bytes per float = 128 bytes

    logger.debug(f"  Shared memory per block: {shared_mem_size} bytes ({shared_mem_size / 1024:.3f}KB)")

    # Single kernel launch with 2D grid - let GPU scheduler handle work distribution
    kernel(
        grid_size,
        block_size_1d,
        (
            sparse_values.ravel(),  # Flatten to 1D uint8
            block_positions.ravel(),  # Flatten to 1D int32
            target_3d.ravel(),  # Flatten 3D planar int8 input to 1D
            result.ravel(),  # Flatten result to 1D fp16
            batch_size,
            channels,
            height,
            width,
            block_size,
        ),
        shared_mem=shared_mem_size,
    )

    return result
