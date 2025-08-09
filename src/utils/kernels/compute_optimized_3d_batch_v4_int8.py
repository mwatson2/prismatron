"""
V4 High-performance batch CUDA kernel optimized for uint8 3008 LEDs.

This version uses a different grid/block strategy optimized for production scale:
- Fewer, larger blocks for better occupancy
- LED-parallel processing within each block
- Coalesced memory access patterns
"""

import logging

import cupy as cp

logger = logging.getLogger(__name__)

# V4 uint8 kernel optimized for 3008 LEDs with better occupancy
COMPUTE_OPTIMIZED_3D_BATCH_V4_INT8_KERNEL = r"""
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

__device__ int dot_int8(uchar4 a, uchar4 b) {
    return (int)a.x * (int)b.x + (int)a.y * (int)b.y + (int)a.z * (int)b.z + (int)a.w * (int)b.w;
}

extern "C" __global__
void compute_optimized_3d_batch_v4_int8_kernel(
    const unsigned char* sparse_values,  // (channels, batch_size, block_size, block_size)
    const int* block_positions,          // (channels, batch_size, 2)
    const unsigned char* target_batch,   // (batch_frames, channels, height, width)
    float* result,                       // (batch_frames, batch_size, channels) or (batch_frames, channels, batch_size)
    const int batch_size,
    const int channels,
    const int batch_frames,
    const int height,
    const int width,
    const int block_size,
    const bool interleaved
) {
    // **V4 ARCHITECTURE: LED-parallel blocks for better occupancy**
    // Grid: (num_led_blocks, channels, batch_frames)
    // Block: 256 threads processing multiple LEDs

    const int LEDS_PER_BLOCK = 4;  // Each block processes 4 LEDs
    const int BLOCK_THREADS = 256;

    int led_block_id = blockIdx.x;
    int channel_id = blockIdx.y;
    int frame_id = blockIdx.z;
    int tid = threadIdx.x;

    if (channel_id >= channels || frame_id >= batch_frames)
        return;

    int led_base = led_block_id * LEDS_PER_BLOCK;
    int led_end = min(led_base + LEDS_PER_BLOCK, batch_size);

    // Shared memory for multiple LED patterns (4 LEDs × 4KB each = 16KB total)
    extern __shared__ unsigned char shared_patterns[];

    // Load LED patterns for this block into shared memory
    for (int led_offset = 0; led_offset < LEDS_PER_BLOCK; led_offset++) {
        int led_id = led_base + led_offset;
        if (led_id >= batch_size) break;

        int sparse_offset = channel_id * (batch_size * block_size * block_size) +
                           led_id * block_size * block_size;
        int shared_offset = led_offset * block_size * block_size;

        // All threads collaborate to load this LED's pattern
        for (int i = tid; i < block_size * block_size; i += BLOCK_THREADS) {
            shared_patterns[shared_offset + i] = sparse_values[sparse_offset + i];
        }
    }
    __syncthreads();

    // Process each LED in this block
    for (int led_offset = 0; led_offset < LEDS_PER_BLOCK; led_offset++) {
        int led_id = led_base + led_offset;
        if (led_id >= batch_size) break;

        // Get LED position
        int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
        int top_row = block_positions[pos_idx + 0];
        int top_col = block_positions[pos_idx + 1];

        // Compute dot product for this LED and frame
        unsigned int thread_sum = 0;

        int target_frame_base = frame_id * (channels * height * width) +
                               channel_id * (height * width) +
                               top_row * width + top_col;

        int shared_pattern_offset = led_offset * block_size * block_size;

        // Distribute 64×64 work across 256 threads
        int pixels_per_thread = (block_size * block_size + BLOCK_THREADS - 1) / BLOCK_THREADS;
        int pixel_start = tid * pixels_per_thread;
        int pixel_end = min(pixel_start + pixels_per_thread, block_size * block_size);

        // Process pixels with vectorized uint8 operations
        for (int pixel_idx = pixel_start; pixel_idx < pixel_end; pixel_idx += 4) {
            int remaining = min(4, pixel_end - pixel_idx);
            if (remaining == 4) {
                // Full vectorized operation
                int row = pixel_idx / block_size;
                int col = pixel_idx % block_size;

                if (col <= block_size - 4) {  // Can do full uchar4 load
                    int target_offset = target_frame_base + row * width + col;
                    uchar4 target_vec = *reinterpret_cast<const uchar4*>(&target_batch[target_offset]);
                    uchar4 pattern_vec = *reinterpret_cast<const uchar4*>(&shared_patterns[shared_pattern_offset + pixel_idx]);
                    thread_sum += dot_int8(pattern_vec, target_vec);
                } else {
                    // Handle edge case with scalar operations
                    for (int i = 0; i < 4 && pixel_idx + i < pixel_end; i++) {
                        int curr_pixel = pixel_idx + i;
                        int curr_row = curr_pixel / block_size;
                        int curr_col = curr_pixel % block_size;
                        int target_offset = target_frame_base + curr_row * width + curr_col;

                        thread_sum += (unsigned int)target_batch[target_offset] *
                                     (unsigned int)shared_patterns[shared_pattern_offset + curr_pixel];
                    }
                }
            } else {
                // Handle remaining pixels with scalar operations
                for (int i = 0; i < remaining; i++) {
                    int curr_pixel = pixel_idx + i;
                    int curr_row = curr_pixel / block_size;
                    int curr_col = curr_pixel % block_size;
                    int target_offset = target_frame_base + curr_row * width + curr_col;

                    thread_sum += (unsigned int)target_batch[target_offset] *
                                 (unsigned int)shared_patterns[shared_pattern_offset + curr_pixel];
                }
            }
        }

        // Block-wide reduction for this LED
        __shared__ unsigned int reduction_buffer[BLOCK_THREADS];
        reduction_buffer[tid] = thread_sum;
        __syncthreads();

        // Tree reduction
        for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction_buffer[tid] += reduction_buffer[tid + stride];
            }
            __syncthreads();
        }

        // Write result
        if (tid == 0) {
            float sum_normalized = (float)reduction_buffer[0] / 65025.0f;  // Scale by 1/255²

            int result_idx;
            if (interleaved) {
                result_idx = frame_id * (batch_size * channels) + led_id * channels + channel_id;
            } else {
                result_idx = frame_id * (channels * batch_size) + channel_id * batch_size + led_id;
            }
            result[result_idx] = sum_normalized;
        }
        __syncthreads();  // Ensure reduction buffer is ready for next LED
    }
}
"""

_kernel_cache = {}


def get_compute_optimized_3d_batch_v4_int8_kernel():
    """Get the compiled V4 uint8 batch kernel."""
    if "v4_int8" not in _kernel_cache:
        logger.info("Compiling V4 uint8 production-optimized batch CUDA kernel...")
        try:
            kernel = cp.RawKernel(
                COMPUTE_OPTIMIZED_3D_BATCH_V4_INT8_KERNEL, "compute_optimized_3d_batch_v4_int8_kernel"
            )
            _kernel_cache["v4_int8"] = kernel
            logger.info("V4 uint8 production kernel compiled successfully")
        except Exception as e:
            logger.error(f"Failed to compile V4 uint8 kernel: {e}")
            raise
    return _kernel_cache["v4_int8"]


def cuda_transpose_dot_product_3d_batch_v4_int8(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    target_batch: cp.ndarray,
    batch_size: int,
    channels: int,
    batch_frames: int,
    block_size: int,
    interleaved: bool = True,
    raw_output: bool = False,
) -> cp.ndarray:
    """
    Compute A^T @ B for multiple frames using V4 uint8 production-optimized kernel.

    Args:
        sparse_values: LED patterns (channels, batch_size, block_size, block_size) uint8
        block_positions: Position data (channels, batch_size, 2) int32
        target_batch: Input frames (batch_frames, channels, height, width) uint8
        batch_size: Number of LEDs
        channels: Number of color channels (typically 3)
        batch_frames: Number of frames to process
        block_size: Size of LED patterns (typically 64)
        interleaved: Output format (True for interleaved channels)
        raw_output: If True, return uint32 without scaling

    Returns:
        Result tensor (batch_frames, batch_size, channels) or (batch_frames, channels, batch_size)
    """
    if raw_output:
        raise NotImplementedError("V4 kernel only supports scaled float32 output")

    _, height, width = target_batch.shape[1], target_batch.shape[2], target_batch.shape[3]

    # Output tensor
    if interleaved:
        result_shape = (batch_frames, batch_size, channels)
    else:
        result_shape = (batch_frames, channels, batch_size)

    result = cp.zeros(result_shape, dtype=cp.float32)

    # Grid configuration: fewer, larger blocks
    LEDS_PER_BLOCK = 4
    num_led_blocks = (batch_size + LEDS_PER_BLOCK - 1) // LEDS_PER_BLOCK
    grid = (num_led_blocks, channels, batch_frames)
    block = (256,)  # 256 threads per block for good occupancy

    # Shared memory: 4 LEDs × 64×64 bytes = 16KB
    shared_mem = LEDS_PER_BLOCK * block_size * block_size

    kernel = get_compute_optimized_3d_batch_v4_int8_kernel()

    kernel(
        grid,
        block,
        (
            sparse_values,
            block_positions,
            target_batch,
            result,
            batch_size,
            channels,
            batch_frames,
            height,
            width,
            block_size,
            interleaved,
        ),
        shared_mem=shared_mem,
    )

    return result
