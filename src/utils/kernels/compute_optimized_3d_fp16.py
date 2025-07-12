"""
Compute-optimized 3D CUDA kernel with FP16 output for SingleBlockMixedSparseTensor operations.

This module implements the FP16 version of the optimized CUDA kernel for LED optimization operations,
specifically the A^T @ b matrix multiplication for mixed sparse tensors with FP16 output.
"""

import logging
from typing import Tuple

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)

# 3D Multi-channel CUDA kernel with FP16 output - 2D grid single launch optimization
COMPUTE_OPTIMIZED_3D_FP16_KERNEL = r"""
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
using namespace cooperative_groups;

__device__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ float4 load_float4_unaligned(const float* ptr) {
    // This may generate less efficient code but handles any alignment
    return make_float4(ptr[0], ptr[1], ptr[2], ptr[3]);
}

extern "C" __global__
void compute_optimized_3d_transpose_dot_product_fp16_kernel(
    const float* sparse_values,     // Shape: (channels, batch_size, block_size, block_size) - FP32 input
    const int* block_positions,     // Shape: (channels, batch_size, 2)
    const float* target_3d,         // Shape: (channels, height, width) - planar form, FP32 input
    __half* result,                 // Shape: (batch_size, channels) if interleaved, (channels, batch_size) if planar - FP16 output
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int block_size,
    const bool interleaved          // true: (batch_size, channels), false: (channels, batch_size)
) {
    // Row-based processing optimization: each thread processes whole rows
    // Requires: block_size % 32 == 0 (e.g., 64x64 blocks with 32 threads = 2 rows per thread)
    // Grid: (batch_size, channels) - one block per (LED, channel) combination
    // Block: (32 threads, 1, 1) - each thread processes block_size/32 complete rows

    auto warp = tiled_partition<32>(this_thread_block());

    int led_id = blockIdx.x;
    int channel_id = blockIdx.y;

    // Bounds check
    if (led_id >= batch_size || channel_id >= channels) return;

    // Calculate output index based on desired memory layout
    // interleaved=true:  (batch_size, channels) layout - idx = led_id * channels + channel_id
    // interleaved=false: (channels, batch_size) layout - idx = channel_id * batch_size + led_id
    int idx = interleaved ? led_id * channels + channel_id : channel_id * batch_size + led_id;

    // Get block position for this LED/channel from (channels, batch_size, 2) layout
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Row-based processing: each thread handles exactly rows_per_thread rows
    int rows_per_thread = block_size / 32;  // e.g., 64/32 = 2 rows per thread
    int block_elements = block_size * block_size;  // 4096 for 64x64

    // Direct computation with row-based iteration (eliminates bounds checks)
    float thread_sum = 0.0f;

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
        // Process all columns in this row
        for (int col = 0;
             col < block_size;
             col += 4, target_idx += 4, sparse_idx += 4) {
            float4 sparse_vec = *reinterpret_cast<const float4*>(&sparse_values[sparse_idx]);
            float4 target_vec = load_float4_unaligned(&target_3d[target_idx]);
            thread_sum += dot(sparse_vec, target_vec);
        }
    }

    float sum = reduce(warp, thread_sum, plus<float>());

    if (warp.thread_rank() == 0) {
        // Convert FP32 result to FP16 and store
        result[idx] = __float2half(sum);
    }
}
"""

# Compile the kernel once on module import
_kernel_cache = {}


def get_compute_optimized_3d_fp16_kernel():
    """Get the compiled compute-optimized 3D CUDA kernel with FP16 output."""
    if "compute_optimized_3d_fp16" not in _kernel_cache:
        logger.info("Compiling compute-optimized 3D FP16 CUDA kernel...")

        # Compile the kernel
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_3D_FP16_KERNEL,
            "compute_optimized_3d_transpose_dot_product_fp16_kernel",
        )
        _kernel_cache["compute_optimized_3d_fp16"] = kernel

        logger.info("Compute-optimized 3D FP16 CUDA kernel compiled successfully")

    return _kernel_cache["compute_optimized_3d_fp16"]


def cuda_transpose_dot_product_3d_compute_optimized_fp16(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    target_3d: cp.ndarray,
    batch_size: int,
    channels: int,
    block_size: int,
    interleaved: bool = True,
) -> cp.ndarray:
    """
    3D Compute-optimized CUDA kernel wrapper for A^T @ b operation with FP16 output.

    Uses row-based processing optimization with 2D grid launch for optimal performance.
    Each thread processes complete rows (block_size/32 rows per thread) with direct memory
    addressing. Eliminates conditional branches and reduces mathematical operations per element.
    Output is converted to FP16 for memory efficiency.

    Args:
        sparse_values: Dense blocks, shape (channels, batch_size, block_size, block_size), FP32
        block_positions: Block positions, shape (channels, batch_size, 2)
        target_3d: Target image in planar form, shape (channels, height, width), FP32
        batch_size: Number of LEDs
        channels: Number of channels
        block_size: Size of square blocks (must be multiple of 32)
        interleaved: If True, return (batch_size, channels). If False, return (channels, batch_size).

    Returns:
        Result of A^T @ b, shape (batch_size, channels) if interleaved=True,
        (channels, batch_size) if interleaved=False, FP16
    """
    channels_input, height, width = target_3d.shape

    if channels_input != channels:
        raise ValueError(f"Input channels {channels_input} != expected {channels}")

    # Prepare output array with correct shape based on layout preference
    if interleaved:
        result = cp.zeros((batch_size, channels), dtype=cp.float16)
    else:
        result = cp.zeros((channels, batch_size), dtype=cp.float16)

    # Get compiled kernel
    kernel = get_compute_optimized_3d_fp16_kernel()

    # 2D Grid configuration: one block per (LED, channel) combination
    grid_size = (batch_size, channels)  # 2D grid: (batch_size, channels)
    block_size_1d = (32,)  # 32 threads per block for optimal warp utilization

    logger.debug("3D Compute-optimized FP16 CUDA kernel launch (2D Grid):")
    logger.debug(f"  Grid: {grid_size} (batch_size={batch_size}, channels={channels})")
    logger.debug(f"  Block: {block_size_1d} (32 threads per block)")
    logger.debug(f"  Target shape: {target_3d.shape} (planar)")
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
            sparse_values.ravel(),  # Flatten to 1D
            block_positions.ravel(),  # Flatten to 1D
            target_3d.ravel(),  # Flatten 3D planar input to 1D
            result.ravel(),  # Flatten result to 1D (FP16)
            batch_size,
            channels,
            height,
            width,
            block_size,
            interleaved,  # New parameter for output layout control
        ),
        shared_mem=shared_mem_size,
    )

    return result
