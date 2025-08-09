"""
Optimized V2 3D batch CUDA kernel for SingleBlockMixedSparseTensor operations.

This module implements an improved batch kernel that processes multiple frames per block
for better GPU occupancy and performance.
"""

import logging
from typing import Tuple

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)

# Improved 3D batch kernel with better thread utilization
COMPUTE_OPTIMIZED_3D_BATCH_V2_KERNEL = r"""
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

__device__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ float4 load_float4_unaligned(const float* ptr) {
    return make_float4(ptr[0], ptr[1], ptr[2], ptr[3]);
}

extern "C" __global__
void compute_optimized_3d_batch_v2_kernel(
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
    // **KEY CHANGE: Process multiple frames per block**
    // Grid: (batch_size, channels)
    // Each block processes ALL frames for one (LED, channel) pair

    int led_id = blockIdx.x;
    int channel_id = blockIdx.y;

    if (led_id >= batch_size || channel_id >= channels)
        return;

    // Each thread processes one or more frames
    int frames_per_thread = (batch_frames + blockDim.x - 1) / blockDim.x;
    int thread_start_frame = threadIdx.x * frames_per_thread;
    int thread_end_frame = min(thread_start_frame + frames_per_thread, batch_frames);

    // Load LED pattern into shared memory once
    extern __shared__ float shared_data[];
    float* led_pattern = shared_data;

    // Cooperative loading of LED pattern (all threads participate)
    int sparse_base_offset = channel_id * (batch_size * block_size * block_size) +
                            led_id * block_size * block_size;
    int elements_per_thread = (block_size * block_size + blockDim.x - 1) / blockDim.x;

    for (int i = 0; i < elements_per_thread; i++) {
        int element_idx = threadIdx.x + i * blockDim.x;
        if (element_idx < block_size * block_size) {
            led_pattern[element_idx] = sparse_values[sparse_base_offset + element_idx];
        }
    }
    __syncthreads();

    // Get LED block position (same for all frames)
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Process assigned frames
    for (int frame_id = thread_start_frame; frame_id < thread_end_frame; frame_id++) {
        float sum = 0.0f;

        // Calculate target frame base offset
        int target_frame_offset = frame_id * (channels * height * width) +
                                 channel_id * (height * width) +
                                 top_row * width + top_col;

        // Compute dot product between LED pattern and target region
        // Vectorized processing for efficiency
        for (int row = 0; row < block_size; row++) {
            int target_row_offset = target_frame_offset + row * width;
            int led_row_offset = row * block_size;

            // Process columns in groups of 4
            for (int col = 0; col < block_size; col += 4) {
                if (col + 3 < block_size) {
                    // Load 4 elements from target
                    float4 target_vec = load_float4_unaligned(&target_batch[target_row_offset + col]);

                    // Load 4 elements from shared memory
                    float4 led_vec = *reinterpret_cast<const float4*>(&led_pattern[led_row_offset + col]);

                    sum += dot(led_vec, target_vec);
                } else {
                    // Handle remaining elements
                    for (int c = col; c < block_size; c++) {
                        sum += led_pattern[led_row_offset + c] * target_batch[target_row_offset + c];
                    }
                }
            }
        }

        // Write result directly (no reduction needed since each thread handles complete frames)
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

# Alternative kernel with warp-level frame processing
COMPUTE_OPTIMIZED_3D_BATCH_WARP_KERNEL = r"""
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

__device__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ float4 load_float4_unaligned(const float* ptr) {
    return make_float4(ptr[0], ptr[1], ptr[2], ptr[3]);
}

extern "C" __global__
void compute_optimized_3d_batch_warp_kernel(
    const float* sparse_values,        // (channels, batch_size, block_size, block_size)
    const int* block_positions,        // (channels, batch_size, 2)
    const float* target_batch,         // (batch_frames, channels, height, width)
    float* result,                     // Output
    const int batch_size,
    const int channels,
    const int batch_frames,
    const int height,
    const int width,
    const int block_size,
    const bool interleaved
) {
    // **WARP-LEVEL APPROACH: Each warp (32 threads) processes one frame**
    // Grid: (batch_size, channels, (batch_frames + 7) / 8)
    // Block: 256 threads = 8 warps = 8 frames processed in parallel

    int led_id = blockIdx.x;
    int channel_id = blockIdx.y;
    int frame_block_id = blockIdx.z;

    if (led_id >= batch_size || channel_id >= channels)
        return;

    // Determine which frame this warp processes
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int frame_id = frame_block_id * 8 + warp_id;

    if (frame_id >= batch_frames)
        return;

    // Load LED pattern into shared memory (all threads participate)
    extern __shared__ float led_pattern[];

    int sparse_base_offset = channel_id * (batch_size * block_size * block_size) +
                            led_id * block_size * block_size;

    // Each thread loads multiple elements
    int total_elements = block_size * block_size;
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        led_pattern[i] = sparse_values[sparse_base_offset + i];
    }
    __syncthreads();

    // Get LED block position
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Each thread in the warp processes a portion of the dot product
    float thread_sum = 0.0f;

    // Calculate target frame base offset
    int target_frame_offset = frame_id * (channels * height * width) +
                             channel_id * (height * width) +
                             top_row * width + top_col;

    // Distribute work across the warp
    // For block_size=32, each thread gets 1 row
    // For block_size=64, each thread gets 2 rows
    int rows_per_thread = (block_size + 31) / 32;
    int thread_start_row = lane_id * rows_per_thread;
    int thread_end_row = min(thread_start_row + rows_per_thread, block_size);

    // Handle case where lane_id >= block_size (for small blocks)
    if (thread_start_row >= block_size) {
        thread_start_row = block_size;
        thread_end_row = block_size;
    }

    for (int row = thread_start_row; row < thread_end_row; row++) {
        int target_row_offset = target_frame_offset + row * width;
        int led_row_offset = row * block_size;

        // Process all columns for this row
        for (int col = 0; col < block_size; col += 4) {
            if (col + 3 < block_size) {
                float4 target_vec = load_float4_unaligned(&target_batch[target_row_offset + col]);
                float4 led_vec = *reinterpret_cast<const float4*>(&led_pattern[led_row_offset + col]);
                thread_sum += dot(led_vec, target_vec);
            } else {
                for (int c = col; c < block_size; c++) {
                    thread_sum += led_pattern[led_row_offset + c] * target_batch[target_row_offset + c];
                }
            }
        }
    }

    // Warp reduction
    auto warp = tiled_partition<32>(this_thread_block());
    float sum = reduce(warp, thread_sum, plus<float>());

    // First thread in warp writes result
    if (lane_id == 0) {
        int result_idx;
        if (interleaved) {
            result_idx = frame_id * (batch_size * channels) + led_id * channels + channel_id;
        } else {
            result_idx = frame_id * (channels * batch_size) + channel_id * batch_size + led_id;
        }
        result[result_idx] = sum;
    }
}
"""

_kernel_cache = {}


def get_compute_optimized_3d_batch_v2_kernel():
    """Get the compiled V2 batch kernel."""
    if "batch_v2" not in _kernel_cache:
        logger.info("Compiling optimized V2 batch CUDA kernel...")
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_3D_BATCH_V2_KERNEL,
            "compute_optimized_3d_batch_v2_kernel",
        )
        _kernel_cache["batch_v2"] = kernel
        logger.info("V2 batch kernel compiled successfully")
    return _kernel_cache["batch_v2"]


def get_compute_optimized_3d_batch_warp_kernel():
    """Get the compiled warp-based batch kernel."""
    if "batch_warp" not in _kernel_cache:
        logger.info("Compiling warp-based batch CUDA kernel...")
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_3D_BATCH_WARP_KERNEL,
            "compute_optimized_3d_batch_warp_kernel",
        )
        _kernel_cache["batch_warp"] = kernel
        logger.info("Warp-based batch kernel compiled successfully")
    return _kernel_cache["batch_warp"]


def cuda_transpose_dot_product_3d_batch_v2(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    target_batch: cp.ndarray,
    batch_size: int,
    channels: int,
    batch_frames: int,
    block_size: int,
    interleaved: bool = True,
) -> cp.ndarray:
    """
    V2 optimized batch kernel with better thread utilization.

    Each block processes all frames for one (LED, channel) pair,
    improving data reuse and reducing kernel launch overhead.
    """
    batch_frames_input, channels_input, height, width = target_batch.shape

    if batch_frames_input != batch_frames:
        raise ValueError(f"Input batch_frames {batch_frames_input} != expected {batch_frames}")

    if channels_input != channels:
        raise ValueError(f"Input channels {channels_input} != expected {channels}")

    # Prepare output array
    if interleaved:
        result = cp.zeros((batch_frames, batch_size, channels), dtype=cp.float32)
    else:
        result = cp.zeros((batch_frames, channels, batch_size), dtype=cp.float32)

    # Get compiled kernel
    kernel = get_compute_optimized_3d_batch_v2_kernel()

    # Grid: one block per (LED, channel) pair
    grid_size = (batch_size, channels)

    # Block: enough threads to process all frames efficiently
    # Use 128 or 256 threads for good occupancy
    block_size_1d = min(256, max(32, batch_frames))

    # Calculate shared memory size
    shared_mem_size = block_size * block_size * 4  # 4 bytes per float

    if shared_mem_size > 48 * 1024:
        raise ValueError(f"Shared memory requirement {shared_mem_size/1024:.1f}KB exceeds 48KB limit")

    # Launch kernel
    kernel(
        grid_size,
        (block_size_1d,),
        (
            sparse_values.ravel(),
            block_positions.ravel(),
            target_batch.ravel(),
            result.ravel(),
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


def cuda_transpose_dot_product_3d_batch_warp(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    target_batch: cp.ndarray,
    batch_size: int,
    channels: int,
    batch_frames: int,
    block_size: int,
    interleaved: bool = True,
) -> cp.ndarray:
    """
    Warp-based batch kernel where each warp processes one frame.

    This approach provides good parallelism for moderate batch sizes
    while maintaining efficient shared memory usage.
    """
    batch_frames_input, channels_input, height, width = target_batch.shape

    if batch_frames_input != batch_frames:
        raise ValueError(f"Input batch_frames {batch_frames_input} != expected {batch_frames}")

    if channels_input != channels:
        raise ValueError(f"Input channels {channels_input} != expected {channels}")

    # Prepare output array
    if interleaved:
        result = cp.zeros((batch_frames, batch_size, channels), dtype=cp.float32)
    else:
        result = cp.zeros((batch_frames, channels, batch_size), dtype=cp.float32)

    # Get compiled kernel
    kernel = get_compute_optimized_3d_batch_warp_kernel()

    # Grid: (batch_size, channels, ceil(batch_frames / 8))
    # Each block processes up to 8 frames (8 warps * 1 frame per warp)
    frames_per_block = 8
    frame_blocks = (batch_frames + frames_per_block - 1) // frames_per_block
    grid_size = (batch_size, channels, frame_blocks)

    # Block: 256 threads = 8 warps
    block_size_1d = 256

    # Calculate shared memory size
    shared_mem_size = block_size * block_size * 4

    if shared_mem_size > 48 * 1024:
        raise ValueError(f"Shared memory requirement {shared_mem_size/1024:.1f}KB exceeds 48KB limit")

    # Launch kernel
    kernel(
        grid_size,
        (block_size_1d,),
        (
            sparse_values.ravel(),
            block_positions.ravel(),
            target_batch.ravel(),
            result.ravel(),
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
