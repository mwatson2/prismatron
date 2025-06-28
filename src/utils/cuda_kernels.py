"""
Custom CUDA kernels for SingleBlockMixedSparseTensor operations.

This module implements optimized CUDA kernels for LED optimization operations,
specifically the A^T @ b matrix multiplication for mixed sparse tensors.
"""

import logging
from typing import Tuple

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)

# 3D Multi-channel CUDA kernel - 2D grid single launch optimization (banked 12.8x improvement)
COMPUTE_OPTIMIZED_3D_KERNEL = r"""
extern "C" __global__
void compute_optimized_3d_transpose_dot_product_kernel(
    const float* sparse_values,     // Shape: (channels, batch_size, block_size, block_size)
    const int* block_positions,     // Shape: (channels, batch_size, 2)
    const float* target_3d,         // Shape: (channels, height, width) - planar form
    float* result,                  // Shape: (batch_size, channels)
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int block_size
) {
    // Row-based processing optimization: each thread processes whole rows
    // Requires: block_size % 32 == 0 (e.g., 64x64 blocks with 32 threads = 2 rows per thread)
    // Grid: (batch_size, channels) - one block per (LED, channel) combination
    // Block: (32 threads, 1, 1) - each thread processes block_size/32 complete rows

    int led_id = blockIdx.x;
    int channel_id = blockIdx.y;

    // Bounds check
    if (led_id >= batch_size || channel_id >= channels) return;

    int idx = led_id * channels + channel_id;

    // Get block position for this LED/channel from (channels, batch_size, 2) layout
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Shared memory optimization: only reduction workspace
    extern __shared__ float shared_mem[];
    float* reduction_workspace = shared_mem; // 32 floats (128 bytes)

    // Row-based processing: each thread handles exactly rows_per_thread rows
    int rows_per_thread = block_size / 32;  // e.g., 64/32 = 2 rows per thread
    int block_elements = block_size * block_size;  // 4096 for 64x64

    // Direct computation with row-based iteration (eliminates bounds checks)
    float thread_sum = 0.0f;

    // Calculate base offset for sparse values: (channels, batch, block, block) layout
    int sparse_offset = channel_id * (batch_size * block_elements) + led_id * block_elements;

    // Calculate target channel offset for planar access
    int target_offset = top_col + channel_id * height * width;

    // Initialize row variables outside loop for efficiency
    int thread_row = threadIdx.x * rows_per_thread;  // Thread's starting row
    int thread_row_offset = ( top_row + thread_row ) * width;  // Global row index in planar form
    int sparse_idx = sparse_offset + thread_row * block_size; // Starting index for first row

    // Each thread processes rows_per_thread complete rows
    for (int row_offset = 0; row_offset < rows_per_thread;
            row_offset++, thread_row++, thread_row_offset += width) {
        // Initialize column variables outside loop for efficiency
        int target_idx = target_offset;

        // Process all columns in this row
        for (int block_col = 0; block_col < block_size;
                block_col++, target_idx++, sparse_idx++) {
            // Direct read from sparse values (no bounds checks needed - block_size % 32 == 0)
            float sparse_val = sparse_values[sparse_idx];

            // Direct read from target image (planar access)
            float target_val = target_3d[target_idx + thread_row_offset];

            thread_sum += sparse_val * target_val;
        }
    }

    // Store in shared memory for reduction
    reduction_workspace[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction: 32 → 1 using tree reduction
    // Optimized for 32 threads (single warp, no sync needed within warp)
    if (threadIdx.x < 16) {
        reduction_workspace[threadIdx.x] += reduction_workspace[threadIdx.x + 16];
    }
    __syncthreads();

    if (threadIdx.x < 8) {
        reduction_workspace[threadIdx.x] += reduction_workspace[threadIdx.x + 8];
    }
    __syncthreads();

    if (threadIdx.x < 4) {
        reduction_workspace[threadIdx.x] += reduction_workspace[threadIdx.x + 4];
    }
    __syncthreads();

    if (threadIdx.x < 2) {
        reduction_workspace[threadIdx.x] += reduction_workspace[threadIdx.x + 2];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        result[idx] = reduction_workspace[0] + reduction_workspace[1];
    }
}
"""

# Compile the kernel once on module import
_kernel_cache = {}


def get_compute_optimized_3d_kernel():
    """Get the compiled compute-optimized 3D CUDA kernel for planar input."""
    if "compute_optimized_3d" not in _kernel_cache:
        logger.info("Compiling compute-optimized 3D CUDA kernel...")

        # Compile the kernel
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_3D_KERNEL,
            "compute_optimized_3d_transpose_dot_product_kernel",
        )
        _kernel_cache["compute_optimized_3d"] = kernel

        logger.info("Compute-optimized 3D CUDA kernel compiled successfully")

    return _kernel_cache["compute_optimized_3d"]


def cuda_transpose_dot_product_3d_compute_optimized(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    target_3d: cp.ndarray,
    batch_size: int,
    channels: int,
    block_size: int,
) -> cp.ndarray:
    """
    3D Compute-optimized CUDA kernel wrapper for A^T @ b operation with planar input.

    Uses row-based processing optimization with 2D grid launch for optimal performance.
    Each thread processes complete rows (block_size/32 rows per thread) with direct memory
    addressing. Eliminates conditional branches and reduces mathematical operations per element.

    Args:
        sparse_values: Dense blocks, shape (channels, batch_size, block_size, block_size)
        block_positions: Block positions, shape (channels, batch_size, 2)
        target_3d: Target image in planar form, shape (channels, height, width)
        batch_size: Number of LEDs
        channels: Number of channels
        block_size: Size of square blocks (must be multiple of 32)

    Returns:
        Result of A^T @ b, shape (batch_size, channels)
    """
    channels_input, height, width = target_3d.shape

    if channels_input != channels:
        raise ValueError(f"Input channels {channels_input} != expected {channels}")

    # Prepare output array
    result = cp.zeros((batch_size, channels), dtype=cp.float32)

    # Get compiled kernel
    kernel = get_compute_optimized_3d_kernel()

    # 2D Grid configuration: one block per (LED, channel) combination (banked 12.8x improvement)
    grid_size = (batch_size, channels)  # 2D grid: (batch_size, channels)
    block_size_1d = (32,)  # 32 threads per block for optimal warp utilization

    logger.debug(
        f"3D Compute-optimized CUDA kernel launch (2D Grid - banked 12.8x improvement):"
    )
    logger.debug(f"  Grid: {grid_size} (batch_size={batch_size}, channels={channels})")
    logger.debug(f"  Block: {block_size_1d} (32 threads per block)")
    logger.debug(f"  Target shape: {target_3d.shape} (planar)")
    logger.debug(f"  Total blocks: {batch_size * channels:,}")
    logger.debug(f"  Total threads: {batch_size * channels * 32:,}")

    # Calculate shared memory size needed
    # Only reduction workspace (32 floats) - no sparse block caching
    shared_mem_size = 32 * 4  # 4 bytes per float = 128 bytes

    logger.debug(
        f"  Shared memory per block: {shared_mem_size} bytes ({shared_mem_size/1024:.3f}KB)"
    )

    # Single kernel launch with 2D grid - let GPU scheduler handle work distribution
    kernel(
        grid_size,
        block_size_1d,
        (
            sparse_values.ravel(),  # Flatten to 1D
            block_positions.ravel(),  # Flatten to 1D
            target_3d.ravel(),  # Flatten 3D planar input to 1D
            result.ravel(),  # Flatten result to 1D
            batch_size,
            channels,
            height,
            width,
            block_size,
        ),
        shared_mem=shared_mem_size,
    )

    return result


# Experimental kernel - row-based processing optimization
EXPERIMENTAL_COMPUTE_OPTIMIZED_3D_KERNEL = r"""
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

__device__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ float4 load_float4_unaligned(const float* ptr) {
    // This may generate less efficient code but handles any alignment
    return make_float4(ptr[0], ptr[1], ptr[2], ptr[3]);
}

extern "C" __global__
void experimental_compute_optimized_3d_transpose_dot_product_kernel(
    const float* sparse_values,     // Shape: (channels, batch_size, block_size, block_size)
    const int* block_positions,     // Shape: (channels, batch_size, 2)
    const float* target_3d,         // Shape: (channels, height, width) - planar form
    float* result,                  // Shape: (batch_size, channels)
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int block_size
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

    int idx = led_id * channels + channel_id;

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
        result[idx] = sum;
    }
}
"""


def get_experimental_compute_optimized_3d_kernel():
    """Get the compiled experimental compute-optimized 3D CUDA kernel for planar input."""
    if "experimental_compute_optimized_3d" not in _kernel_cache:
        logger.info("Compiling experimental compute-optimized 3D CUDA kernel...")

        # Compile the kernel
        kernel = cp.RawKernel(
            EXPERIMENTAL_COMPUTE_OPTIMIZED_3D_KERNEL,
            "experimental_compute_optimized_3d_transpose_dot_product_kernel",
        )
        _kernel_cache["experimental_compute_optimized_3d"] = kernel

        logger.info(
            "Experimental compute-optimized 3D CUDA kernel compiled successfully"
        )

    return _kernel_cache["experimental_compute_optimized_3d"]


def cuda_transpose_dot_product_3d_compute_optimized_experimental(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    target_3d: cp.ndarray,
    batch_size: int,
    channels: int,
    block_size: int,
) -> cp.ndarray:
    """
    Experimental 3D Compute-optimized CUDA kernel wrapper for A^T @ b operation with planar input.

    Experimental optimization: Row-based processing to reduce operations and eliminate bounds
    checks.
    Each thread processes complete rows (block_size/32 rows per thread) with direct memory
    addressing. Eliminates conditional branches and reduces mathematical operations per element.

    Args:
        sparse_values: Dense blocks, shape (channels, batch_size, block_size, block_size)
        block_positions: Block positions, shape (channels, batch_size, 2)
        target_3d: Target image in planar form, shape (channels, height, width)
        batch_size: Number of LEDs
        channels: Number of channels
        block_size: Size of square blocks

    Returns:
        Result of A^T @ b, shape (batch_size, channels)
    """
    channels_input, height, width = target_3d.shape

    if channels_input != channels:
        raise ValueError(f"Input channels {channels_input} != expected {channels}")

    # Prepare output array
    result = cp.zeros((batch_size, channels), dtype=cp.float32)

    # Get compiled experimental kernel
    kernel = get_experimental_compute_optimized_3d_kernel()

    # 2D Grid configuration: one block per (LED, channel) combination
    grid_size = (batch_size, channels)  # 2D grid: (batch_size, channels)
    block_size_1d = (32,)  # 32 threads per block for optimal warp utilization

    logger.debug(f"Experimental 3D Compute-optimized CUDA kernel launch (2D Grid):")
    logger.debug(f"  Grid: {grid_size} (batch_size={batch_size}, channels={channels})")
    logger.debug(f"  Block: {block_size_1d} (32 threads per block)")
    logger.debug(f"  Target shape: {target_3d.shape} (planar)")
    logger.debug(f"  Total blocks: {batch_size * channels:,}")
    logger.debug(f"  Total threads: {batch_size * channels * 32:,}")

    # Calculate shared memory size needed
    # Only reduction workspace (32 floats) - no sparse block caching in experimental kernel
    shared_mem_size = 32 * 4  # 4 bytes per float = 128 bytes

    logger.debug(
        f"  Shared memory per block: {shared_mem_size} bytes ({shared_mem_size/1024:.3f}KB)"
    )

    # Single kernel launch with 2D grid - let GPU scheduler handle work distribution
    kernel(
        grid_size,
        block_size_1d,
        (
            sparse_values.ravel(),  # Flatten to 1D
            block_positions.ravel(),  # Flatten to 1D
            target_3d.ravel(),  # Flatten 3D planar input to 1D
            result.ravel(),  # Flatten result to 1D
            batch_size,
            channels,
            height,
            width,
            block_size,
        ),
        shared_mem=shared_mem_size,
    )

    return result
