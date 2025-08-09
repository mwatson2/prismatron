"""
V3 Memory-optimized batch CUDA kernel with proper vectorization and coalescing.
"""

import logging

import cupy as cp

logger = logging.getLogger(__name__)

# V3 kernel with proper memory coalescing and spatial optimization
COMPUTE_OPTIMIZED_3D_BATCH_V3_KERNEL = r"""
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

__device__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ float4 load_float4_aligned(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

extern "C" __global__
void compute_optimized_3d_batch_v3_kernel(
    const float* sparse_values,        // (channels, batch_size, block_size, block_size)
    const int* block_positions,        // (channels, batch_size, 2)
    const float* target_batch,         // (batch_frames, channels, height, width)
    float* result,                     // Output array
    const int batch_size,
    const int channels,
    const int batch_frames,
    const int height,
    const int width,
    const int block_size,
    const bool interleaved
) {
    // **SPATIAL-AWARE APPROACH**
    // Grid: (ceil(batch_size/8), channels)
    // Block: 256 threads = 8 warps = 8 LEDs per block
    // Each block processes 8 LEDs for one channel across all frames

    int led_block_id = blockIdx.x;
    int channel_id = blockIdx.y;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp handles one LED
    int led_id = led_block_id * 8 + warp_id;

    if (led_id >= batch_size || channel_id >= channels)
        return;

    // Load LED pattern into shared memory (all warps collaborate)
    extern __shared__ float shared_data[];
    float* led_patterns = shared_data;  // [8][64*64] - one pattern per LED

    // Get LED block position
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Load this LED's pattern to shared memory
    int pattern_offset = warp_id * block_size * block_size;  // Offset in shared memory
    int sparse_offset = channel_id * (batch_size * block_size * block_size) +
                       led_id * block_size * block_size;

    // Each thread in warp loads part of the pattern
    int elements_per_thread = (block_size * block_size + 31) / 32;
    for (int i = 0; i < elements_per_thread; i++) {
        int element_idx = lane_id + i * 32;
        if (element_idx < block_size * block_size) {
            led_patterns[pattern_offset + element_idx] = sparse_values[sparse_offset + element_idx];
        }
    }
    __syncthreads();

    // Process all frames for this LED
    for (int frame_id = 0; frame_id < batch_frames; frame_id++) {
        float thread_sum = 0.0f;

        // Calculate base offset for this frame and channel
        int target_frame_base = frame_id * (channels * height * width) +
                               channel_id * (height * width) +
                               top_row * width + top_col;

        // **VECTORIZED ACCESS PATTERN**
        // Each thread processes 2 rows (64 elements total)
        int rows_per_thread = (block_size + 31) / 32;
        int start_row = lane_id * rows_per_thread;
        int end_row = min(start_row + rows_per_thread, block_size);

        for (int row = start_row; row < end_row; row++) {
            int target_row_offset = target_frame_base + row * width;
            int pattern_row_offset = pattern_offset + row * block_size;

            // Process entire row with vectorized loads (16 float4 loads for 64 elements)
            for (int col = 0; col < block_size; col += 4) {
                float4 target_vec = *reinterpret_cast<const float4*>(&target_batch[target_row_offset + col]);
                float4 pattern_vec = *reinterpret_cast<const float4*>(&led_patterns[pattern_row_offset + col]);
                thread_sum += dot(pattern_vec, target_vec);
            }
        }

        // Warp reduction
        auto warp = tiled_partition<32>(this_thread_block());
        float sum = reduce(warp, thread_sum, plus<float>());

        // Write result (first thread in warp)
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
}

// Alternative: Super-optimized kernel with frame-level parallelism
extern "C" __global__
void compute_optimized_3d_batch_v3_frames_kernel(
    const float* sparse_values,
    const int* block_positions,
    const float* target_batch,
    float* result,
    const int batch_size,
    const int channels,
    const int batch_frames,
    const int height,
    const int width,
    const int block_size,
    const bool interleaved
) {
    // **FRAME-PARALLEL APPROACH**
    // Grid: (batch_size, channels)
    // Block: batch_frames * 32 threads (one warp per frame)

    int led_id = blockIdx.x;
    int channel_id = blockIdx.y;
    int frame_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (led_id >= batch_size || channel_id >= channels || frame_id >= batch_frames)
        return;

    // Load LED pattern into shared memory (all threads collaborate)
    extern __shared__ float led_pattern[];

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

    // Compute for this frame
    float thread_sum = 0.0f;

    int target_frame_base = frame_id * (channels * height * width) +
                           channel_id * (height * width) +
                           top_row * width + top_col;

    // Distribute work across warp
    int rows_per_thread = (block_size + 31) / 32;
    int start_row = lane_id * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, block_size);

    for (int row = start_row; row < end_row; row++) {
        int target_row_offset = target_frame_base + row * width;
        int pattern_row_offset = row * block_size;

        for (int col = 0; col < block_size; col += 4) {
            float4 target_vec = *reinterpret_cast<const float4*>(&target_batch[target_row_offset + col]);
            float4 pattern_vec = *reinterpret_cast<const float4*>(&led_pattern[pattern_row_offset + col]);
            thread_sum += dot(pattern_vec, target_vec);
        }
    }

    // Warp reduction
    auto warp = tiled_partition<32>(this_thread_block());
    float sum = reduce(warp, thread_sum, plus<float>());

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


def get_compute_optimized_3d_batch_v3_kernel():
    """Get V3 spatial-aware kernel."""
    if "batch_v3" not in _kernel_cache:
        logger.info("Compiling V3 memory-optimized batch CUDA kernel...")
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_3D_BATCH_V3_KERNEL,
            "compute_optimized_3d_batch_v3_kernel",
        )
        _kernel_cache["batch_v3"] = kernel
        logger.info("V3 kernel compiled successfully")
    return _kernel_cache["batch_v3"]


def get_compute_optimized_3d_batch_v3_frames_kernel():
    """Get V3 frame-parallel kernel."""
    if "batch_v3_frames" not in _kernel_cache:
        logger.info("Compiling V3 frame-parallel batch CUDA kernel...")
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_3D_BATCH_V3_KERNEL,
            "compute_optimized_3d_batch_v3_frames_kernel",
        )
        _kernel_cache["batch_v3_frames"] = kernel
        logger.info("V3 frame-parallel kernel compiled successfully")
    return _kernel_cache["batch_v3_frames"]


def cuda_transpose_dot_product_3d_batch_v3(
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
    V3 memory-optimized batch kernel with spatial awareness.

    Processes 8 LEDs per block to improve shared memory efficiency
    and reduce total number of blocks for better occupancy.
    """
    batch_frames_input, channels_input, height, width = target_batch.shape

    if batch_frames_input != batch_frames:
        raise ValueError(f"Input batch_frames {batch_frames_input} != expected {batch_frames}")

    if channels_input != channels:
        raise ValueError(f"Input channels {channels_input} != expected {channels}")

    # Output array
    if interleaved:
        result = cp.zeros((batch_frames, batch_size, channels), dtype=cp.float32)
    else:
        result = cp.zeros((batch_frames, channels, batch_size), dtype=cp.float32)

    # Get kernel
    kernel = get_compute_optimized_3d_batch_v3_kernel()

    # Grid: ceil(batch_size/8) blocks × channels
    leds_per_block = 8
    led_blocks = (batch_size + leds_per_block - 1) // leds_per_block
    grid_size = (led_blocks, channels)

    # Block: 256 threads = 8 warps (one per LED)
    block_size_1d = 256

    # Shared memory: 8 LED patterns
    shared_mem_size = leds_per_block * block_size * block_size * 4

    if shared_mem_size > 96 * 1024:  # Use extended shared memory if available
        raise ValueError(f"Shared memory requirement {shared_mem_size/1024:.1f}KB exceeds limit")

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


def cuda_transpose_dot_product_3d_batch_v3_frames(
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
    V3 frame-parallel batch kernel.

    Each block processes one LED across all frames simultaneously.
    """
    batch_frames_input, channels_input, height, width = target_batch.shape

    if batch_frames_input != batch_frames:
        raise ValueError(f"Input batch_frames {batch_frames_input} != expected {batch_frames}")

    if channels_input != channels:
        raise ValueError(f"Input channels {channels_input} != expected {channels}")

    # Output array
    if interleaved:
        result = cp.zeros((batch_frames, batch_size, channels), dtype=cp.float32)
    else:
        result = cp.zeros((batch_frames, channels, batch_size), dtype=cp.float32)

    # Get kernel
    kernel = get_compute_optimized_3d_batch_v3_frames_kernel()

    # Grid: one block per (LED, channel)
    grid_size = (batch_size, channels)

    # Block: batch_frames warps (32 threads each)
    threads_per_frame = 32
    block_size_1d = batch_frames * threads_per_frame

    if block_size_1d > 1024:  # CUDA block size limit
        # Fall back to V3 spatial kernel
        return cuda_transpose_dot_product_3d_batch_v3(
            sparse_values, block_positions, target_batch, batch_size, channels, batch_frames, block_size, interleaved
        )

    # Shared memory: one LED pattern
    shared_mem_size = block_size * block_size * 4

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
