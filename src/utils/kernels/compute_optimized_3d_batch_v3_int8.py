"""
V3 Memory-optimized batch CUDA kernel for uint8 inputs with FP32 output.

This implements the production-ready uint8 version where both LED patterns and
target images are uint8, with proper uint32 accumulation and 255² scaling.
"""

import logging

import cupy as cp

logger = logging.getLogger(__name__)

# V3 uint8 frame-parallel kernel with proper vectorization
COMPUTE_OPTIMIZED_3D_BATCH_V3_INT8_KERNEL = r"""
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

__device__ int dot_int8(uchar4 a, uchar4 b) {
    // Compute dot product of two uchar4 vectors as integers
    // Each multiplication gives [0,255] × [0,255] = [0,65025]
    // Sum of 4 gives max 4×65025 = 260,100 (fits in int32)
    return (int)a.x * (int)b.x + (int)a.y * (int)b.y + (int)a.z * (int)b.z + (int)a.w * (int)b.w;
}

extern "C" __global__
void compute_optimized_3d_batch_v3_int8_frames_kernel(
    const unsigned char* sparse_values,  // (channels, batch_size, block_size, block_size) - uint8
    const int* block_positions,          // (channels, batch_size, 2) - int32
    const unsigned char* target_batch,   // (batch_frames, channels, height, width) - uint8
    float* result,                       // Output - fp32
    const int batch_size,
    const int channels,
    const int batch_frames,
    const int height,
    const int width,
    const int block_size,
    const bool interleaved
) {
    // **FRAME-PARALLEL APPROACH with uint8**
    // Grid: (batch_size, channels)
    // Block: batch_frames warps (32 threads each)

    int led_id = blockIdx.x;
    int channel_id = blockIdx.y;
    int frame_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (led_id >= batch_size || channel_id >= channels || frame_id >= batch_frames)
        return;

    // Load LED pattern into shared memory (all threads collaborate)
    // Use uint8 storage for patterns
    extern __shared__ unsigned char led_pattern[];

    int sparse_offset = channel_id * (batch_size * block_size * block_size) +
                       led_id * block_size * block_size;

    // Each thread loads part of pattern
    int total_threads = blockDim.x;
    for (int i = threadIdx.x; i < block_size * block_size; i += total_threads) {
        led_pattern[i] = sparse_values[sparse_offset + i];
    }
    __syncthreads();

    // Get LED position
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Compute for this frame using uint32 accumulation
    unsigned int thread_sum_uint = 0;  // Use uint32 for accumulation

    int target_frame_base = frame_id * (channels * height * width) +
                           channel_id * (height * width) +
                           top_row * width + top_col;

    // Distribute work across warp
    int rows_per_thread = (block_size + 31) / 32;
    int start_row = lane_id * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, block_size);

    // Process assigned rows with vectorized uint8 operations
    for (int row = start_row; row < end_row; row++) {
        int target_row_offset = target_frame_base + row * width;
        int pattern_row_offset = row * block_size;

        // Process columns in groups of 4 using uchar4 vectors
        for (int col = 0; col < block_size; col += 4) {
            // Load 4 uint8 elements from target image
            uchar4 target_vec = *reinterpret_cast<const uchar4*>(&target_batch[target_row_offset + col]);

            // Load 4 uint8 elements from LED pattern in shared memory
            uchar4 pattern_vec = *reinterpret_cast<const uchar4*>(&led_pattern[pattern_row_offset + col]);

            // Compute dot product (returns int, max 4×255² = 260,100)
            thread_sum_uint += (unsigned int)dot_int8(pattern_vec, target_vec);
        }
    }

    // Warp reduction of uint32 values
    auto warp = tiled_partition<32>(this_thread_block());
    unsigned int sum_uint = reduce(warp, thread_sum_uint, plus<unsigned int>());

    // Convert to float and scale by 1/(255²) = 1/65025
    // This gives output in range [0, (block_size²)/65025] ≈ [0, 0.063] for 64×64 blocks
    if (lane_id == 0) {
        float sum_normalized = (float)sum_uint / 65025.0f;  // Divide by 255²

        int result_idx;
        if (interleaved) {
            result_idx = frame_id * (batch_size * channels) + led_id * channels + channel_id;
        } else {
            result_idx = frame_id * (channels * batch_size) + channel_id * batch_size + led_id;
        }
        result[result_idx] = sum_normalized;
    }
}

// Alternative kernel without 255² scaling for direct uint32 output
extern "C" __global__
void compute_optimized_3d_batch_v3_int8_raw_kernel(
    const unsigned char* sparse_values,
    const int* block_positions,
    const unsigned char* target_batch,
    unsigned int* result,               // uint32 output (no scaling)
    const int batch_size,
    const int channels,
    const int batch_frames,
    const int height,
    const int width,
    const int block_size,
    const bool interleaved
) {
    // Same structure as above but with uint32 output
    int led_id = blockIdx.x;
    int channel_id = blockIdx.y;
    int frame_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (led_id >= batch_size || channel_id >= channels || frame_id >= batch_frames)
        return;

    extern __shared__ unsigned char led_pattern[];

    int sparse_offset = channel_id * (batch_size * block_size * block_size) +
                       led_id * block_size * block_size;

    int total_threads = blockDim.x;
    for (int i = threadIdx.x; i < block_size * block_size; i += total_threads) {
        led_pattern[i] = sparse_values[sparse_offset + i];
    }
    __syncthreads();

    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    unsigned int thread_sum_uint = 0;

    int target_frame_base = frame_id * (channels * height * width) +
                           channel_id * (height * width) +
                           top_row * width + top_col;

    int rows_per_thread = (block_size + 31) / 32;
    int start_row = lane_id * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, block_size);

    for (int row = start_row; row < end_row; row++) {
        int target_row_offset = target_frame_base + row * width;
        int pattern_row_offset = row * block_size;

        for (int col = 0; col < block_size; col += 4) {
            uchar4 target_vec = *reinterpret_cast<const uchar4*>(&target_batch[target_row_offset + col]);
            uchar4 pattern_vec = *reinterpret_cast<const uchar4*>(&led_pattern[pattern_row_offset + col]);
            thread_sum_uint += (unsigned int)dot_int8(pattern_vec, target_vec);
        }
    }

    auto warp = tiled_partition<32>(this_thread_block());
    unsigned int sum_uint = reduce(warp, thread_sum_uint, plus<unsigned int>());

    if (lane_id == 0) {
        int result_idx;
        if (interleaved) {
            result_idx = frame_id * (batch_size * channels) + led_id * channels + channel_id;
        } else {
            result_idx = frame_id * (channels * batch_size) + channel_id * batch_size + led_id;
        }
        result[result_idx] = sum_uint;  // Raw uint32 output
    }
}
"""

_kernel_cache = {}


def get_compute_optimized_3d_batch_v3_int8_frames_kernel():
    """Get V3 uint8 frame-parallel kernel with fp32 output."""
    if "batch_v3_int8_frames" not in _kernel_cache:
        logger.info("Compiling V3 uint8 frame-parallel batch CUDA kernel...")
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_3D_BATCH_V3_INT8_KERNEL,
            "compute_optimized_3d_batch_v3_int8_frames_kernel",
        )
        _kernel_cache["batch_v3_int8_frames"] = kernel
        logger.info("V3 uint8 frame-parallel kernel compiled successfully")
    return _kernel_cache["batch_v3_int8_frames"]


def get_compute_optimized_3d_batch_v3_int8_raw_kernel():
    """Get V3 uint8 frame-parallel kernel with uint32 output."""
    if "batch_v3_int8_raw" not in _kernel_cache:
        logger.info("Compiling V3 uint8 raw batch CUDA kernel...")
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_3D_BATCH_V3_INT8_KERNEL,
            "compute_optimized_3d_batch_v3_int8_raw_kernel",
        )
        _kernel_cache["batch_v3_int8_raw"] = kernel
        logger.info("V3 uint8 raw kernel compiled successfully")
    return _kernel_cache["batch_v3_int8_raw"]


def cuda_transpose_dot_product_3d_batch_v3_int8(
    sparse_values: cp.ndarray,  # uint8
    block_positions: cp.ndarray,  # int32
    target_batch: cp.ndarray,  # uint8
    batch_size: int,
    channels: int,
    batch_frames: int,
    block_size: int,
    interleaved: bool = True,
    raw_output: bool = False,
) -> cp.ndarray:
    """
    V3 uint8 batch kernel with proper scaling.

    Args:
        raw_output: If True, return uint32 without 255² scaling.
                   If False, return fp32 with 255² scaling applied.
    """
    batch_frames_input, channels_input, height, width = target_batch.shape

    if batch_frames_input != batch_frames:
        raise ValueError(f"Input batch_frames {batch_frames_input} != expected {batch_frames}")

    if channels_input != channels:
        raise ValueError(f"Input channels {channels_input} != expected {channels}")

    if sparse_values.dtype != cp.uint8:
        raise ValueError(f"sparse_values must be uint8, got {sparse_values.dtype}")

    if target_batch.dtype != cp.uint8:
        raise ValueError(f"target_batch must be uint8, got {target_batch.dtype}")

    # Choose output type and kernel
    if raw_output:
        # uint32 output without scaling
        if interleaved:
            result = cp.zeros((batch_frames, batch_size, channels), dtype=cp.uint32)
        else:
            result = cp.zeros((batch_frames, channels, batch_size), dtype=cp.uint32)
        kernel = get_compute_optimized_3d_batch_v3_int8_raw_kernel()
    else:
        # fp32 output with 255² scaling
        if interleaved:
            result = cp.zeros((batch_frames, batch_size, channels), dtype=cp.float32)
        else:
            result = cp.zeros((batch_frames, channels, batch_size), dtype=cp.float32)
        kernel = get_compute_optimized_3d_batch_v3_int8_frames_kernel()

    # Grid: one block per (LED, channel)
    grid_size = (batch_size, channels)

    # Block: batch_frames warps (32 threads each)
    threads_per_frame = 32
    block_size_1d = batch_frames * threads_per_frame

    if block_size_1d > 1024:  # CUDA block size limit
        raise ValueError(f"Too many frames ({batch_frames}) for single kernel block. Max ~32 frames.")

    # Shared memory: one LED pattern (uint8)
    shared_mem_size = block_size * block_size  # 1 byte per element for uint8

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
