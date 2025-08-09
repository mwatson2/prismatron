"""
Compute-optimized 3D batch CUDA kernel for SingleBlockMixedSparseTensor operations.

This module implements the FP32 batch version of the optimized CUDA kernel for LED optimization operations,
specifically the batched A^T @ B matrix multiplication where B contains multiple frames.
"""

import logging
from typing import Tuple

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)

# 3D batch CUDA kernel with shared memory optimization for LED patterns
COMPUTE_OPTIMIZED_3D_BATCH_KERNEL = r"""
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
void compute_optimized_3d_batch_transpose_dot_product_kernel(
    const float* sparse_values,        // (channels, batch_size, block_size, block_size)
    const int* block_positions,        // (channels, batch_size, 2)
    const float* target_batch,         // (batch_frames, channels, height, width)
    float* result,                     // (batch_frames, batch_size, channels) or (batch_frames, channels, batch_size)
    const int batch_size,              // Number of LEDs
    const int channels,                // Number of color channels
    const int batch_frames,            // Number of input frames
    const int height,
    const int width,
    const int block_size,
    const bool interleaved             // Output layout control
) {
    // 3D grid coordinates
    int led_id = blockIdx.x;           // LED index [0, batch_size)
    int channel_id = blockIdx.y;       // Channel index [0, channels)
    int frame_id = blockIdx.z;         // Frame index [0, batch_frames)

    // Bounds checking
    if (led_id >= batch_size || channel_id >= channels || frame_id >= batch_frames)
        return;

    // **KEY OPTIMIZATION: Load LED pattern into shared memory**
    extern __shared__ float led_pattern[];  // Dynamic shared memory allocation

    // Cooperative loading of LED pattern by all 32 threads
    int elements_per_thread = (block_size * block_size + 31) / 32;
    int sparse_base_offset = channel_id * (batch_size * block_size * block_size) +
                            led_id * block_size * block_size;

    for (int i = 0; i < elements_per_thread; i++) {
        int element_idx = threadIdx.x * elements_per_thread + i;
        if (element_idx < block_size * block_size) {
            led_pattern[element_idx] = sparse_values[sparse_base_offset + element_idx];
        }
    }
    __syncthreads();  // Ensure all threads have loaded their part

    // Get LED block position
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Process assigned rows (same logic as single-frame version)
    int rows_per_thread = block_size / 32;
    float thread_sum = 0.0f;

    // Calculate target frame base offset
    int target_frame_offset = frame_id * (channels * height * width) +
                             channel_id * (height * width) +
                             top_row * width + top_col;

    // Each thread processes rows_per_thread rows using shared memory LED pattern
    int thread_start_row = threadIdx.x * rows_per_thread;
    for (int row_offset = 0; row_offset < rows_per_thread; row_offset++) {
        int current_row = thread_start_row + row_offset;
        int target_row_offset = target_frame_offset + current_row * width;
        int led_row_offset = current_row * block_size;

        // Process all columns in this row using vectorized loads
        for (int col = 0; col < block_size; col += 4) {
            // Load 4 elements from target frame
            float4 target_vec = load_float4_unaligned(
                &target_batch[target_row_offset + col]);

            // Load 4 elements from shared memory LED pattern
            float4 led_vec = *reinterpret_cast<const float4*>(
                &led_pattern[led_row_offset + col]);

            // Compute dot product
            thread_sum += dot(led_vec, target_vec);
        }
    }

    // Warp reduction
    auto warp = tiled_partition<32>(this_thread_block());
    float sum = reduce(warp, thread_sum, plus<float>());

    // Write result with correct indexing
    if (warp.thread_rank() == 0) {
        int result_idx;
        if (interleaved) {
            // (batch_frames, batch_size, channels)
            result_idx = frame_id * (batch_size * channels) + led_id * channels + channel_id;
        } else {
            // (batch_frames, channels, batch_size)
            result_idx = frame_id * (channels * batch_size) + channel_id * batch_size + led_id;
        }
        result[result_idx] = sum;
    }
}
"""

# Compile the kernel once on module import
_kernel_cache = {}


def get_compute_optimized_3d_batch_kernel():
    """Get the compiled compute-optimized 3D batch CUDA kernel for FP32 inputs."""
    if "compute_optimized_3d_batch" not in _kernel_cache:
        logger.info("Compiling compute-optimized 3D batch CUDA kernel...")

        # Compile the kernel
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_3D_BATCH_KERNEL,
            "compute_optimized_3d_batch_transpose_dot_product_kernel",
        )
        _kernel_cache["compute_optimized_3d_batch"] = kernel

        logger.info("Compute-optimized 3D batch CUDA kernel compiled successfully")

    return _kernel_cache["compute_optimized_3d_batch"]


def cuda_transpose_dot_product_3d_batch_compute_optimized(
    sparse_values: cp.ndarray,  # (channels, batch_size, block_size, block_size)
    block_positions: cp.ndarray,  # (channels, batch_size, 2)
    target_batch: cp.ndarray,  # (batch_frames, channels, height, width)
    batch_size: int,
    channels: int,
    batch_frames: int,
    block_size: int,
    interleaved: bool = True,
) -> cp.ndarray:
    """
    3D Batch Compute-optimized CUDA kernel wrapper for batched A^T @ B operation.

    Processes multiple input frames simultaneously using shared memory optimization
    for LED patterns. Each CUDA block loads one LED pattern into shared memory
    and reuses it across all computations within that block.

    Args:
        sparse_values: Dense LED blocks, shape (channels, batch_size, block_size, block_size)
        block_positions: Block positions, shape (channels, batch_size, 2)
        target_batch: Target frames, shape (batch_frames, channels, height, width)
        batch_size: Number of LEDs
        channels: Number of color channels
        batch_frames: Number of input frames to process
        block_size: Size of square LED blocks (must be multiple of 32)
        interleaved: If True, return (batch_frames, batch_size, channels).
                    If False, return (batch_frames, channels, batch_size).

    Returns:
        Result of batched A^T @ B, shape (batch_frames, batch_size, channels) if interleaved=True,
        (batch_frames, channels, batch_size) if interleaved=False, dtype float32
    """
    batch_frames_input, channels_input, height, width = target_batch.shape

    if batch_frames_input != batch_frames:
        raise ValueError(f"Input batch_frames {batch_frames_input} != expected {batch_frames}")

    if channels_input != channels:
        raise ValueError(f"Input channels {channels_input} != expected {channels}")

    if block_size % 32 != 0:
        raise ValueError(f"block_size {block_size} must be multiple of 32 for row-based processing")

    # Prepare output array with correct shape based on layout preference
    if interleaved:
        result = cp.zeros((batch_frames, batch_size, channels), dtype=cp.float32)
    else:
        result = cp.zeros((batch_frames, channels, batch_size), dtype=cp.float32)

    # Get compiled kernel
    kernel = get_compute_optimized_3d_batch_kernel()

    # 3D Grid configuration: (batch_size, channels, batch_frames)
    grid_size = (batch_size, channels, batch_frames)
    block_size_1d = (32,)  # 32 threads per block for optimal warp utilization

    logger.debug("3D Batch Compute-optimized CUDA kernel launch:")
    logger.debug(f"  Grid: {grid_size} (batch_size={batch_size}, channels={channels}, batch_frames={batch_frames})")
    logger.debug(f"  Block: {block_size_1d} (32 threads per block)")
    logger.debug(f"  Target batch shape: {target_batch.shape}")
    logger.debug(f"  Total blocks: {batch_size * channels * batch_frames:,}")
    logger.debug(f"  Total threads: {batch_size * channels * batch_frames * 32:,}")

    # Calculate shared memory size needed for LED pattern storage
    # Each block stores one LED pattern (block_size x block_size floats)
    shared_mem_size = block_size * block_size * 4  # 4 bytes per float

    logger.debug(f"  Shared memory per block: {shared_mem_size} bytes ({shared_mem_size / 1024:.3f}KB)")

    if shared_mem_size > 48 * 1024:  # 48KB shared memory limit
        raise ValueError(f"Shared memory requirement {shared_mem_size/1024:.1f}KB exceeds 48KB limit")

    # Launch kernel with 3D grid and dynamic shared memory
    kernel(
        grid_size,
        block_size_1d,
        (
            sparse_values.ravel(),  # Flatten to 1D
            block_positions.ravel(),  # Flatten to 1D
            target_batch.ravel(),  # Flatten 4D batch input to 1D
            result.ravel(),  # Flatten result to 1D
            batch_size,
            channels,
            batch_frames,
            height,
            width,
            block_size,
            interleaved,
        ),
        shared_mem=shared_mem_size,
    )

    return result
