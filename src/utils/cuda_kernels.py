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

# 3D Multi-channel CUDA kernel - based on compute-optimized kernel for planar input
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
    const int block_size,
    const int iteration_offset      // Which set of 8 blocks this iteration processes
) {
    // Architecture-matched design: 8 SMs, 32 cores per SM
    // Grid: (8 blocks, 1, 1) - one block per SM
    // Block: (32 threads, 1, 1) - one thread per core

    // Which (LED, channel) combination this SM is processing
    int led_channel_id = iteration_offset * 8 + blockIdx.x;

    if (led_channel_id >= batch_size * channels) return;

    int led_id = led_channel_id / channels;
    int channel_id = led_channel_id % channels;
    int idx = led_id * channels + channel_id;

    // Get block position for this LED/channel from (channels, batch_size, 2) layout
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Shared memory optimized for GPU limits (max 48KB per block)
    // Load only sparse block into shared memory, access target directly
    extern __shared__ float shared_mem[];
    float* sparse_block = shared_mem;                    // block_size * block_size floats (36KB for 96x96)
    float* reduction_workspace = &shared_mem[block_size * block_size]; // 32 floats (128 bytes)
    // Total: 36KB + 128 bytes = ~36.1KB (well under 48KB limit)

    int block_elements = block_size * block_size;  // 9216 for 96x96
    int elements_per_thread = (block_elements + 31) / 32;  // 288 elements per thread

    // Cooperative loading: 32 threads load sparse block into shared memory
    // Calculate offset for (channels, batch_size, block_size, block_size) layout
    int sparse_offset = channel_id * (batch_size * block_elements) + led_id * block_elements;
    for (int i = 0; i < elements_per_thread; i++) {
        int element_idx = threadIdx.x * elements_per_thread + i;
        if (element_idx < block_elements) {
            sparse_block[element_idx] = sparse_values[sparse_offset + element_idx];
        }
    }

    __syncthreads();  // Ensure sparse block is loaded

    // Compute phase: each thread processes 288 multiply-adds
    // Access 3D target directly (channel_id, spatial) for planar layout
    float thread_sum = 0.0f;

    // Calculate target channel offset for planar access
    int channel_offset = channel_id * height * width;

    for (int i = 0; i < elements_per_thread; i++) {
        int element_idx = threadIdx.x * elements_per_thread + i;
        if (element_idx < block_elements) {
            // Calculate target image coordinates on-the-fly
            int block_row = element_idx / block_size;
            int block_col = element_idx % block_size;
            int global_row = top_row + block_row;
            int global_col = top_col + block_col;

            // Bounds check and compute with 3D planar access
            if (global_row < height && global_col < width) {
                float sparse_val = sparse_block[element_idx];
                // Planar access: target_3d[channel_id, global_row, global_col]
                float target_val = target_3d[channel_offset + global_row * width + global_col];
                thread_sum += sparse_val * target_val;
            }
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

# DEPRECATED: Compute-optimized CUDA kernel - 8-way parallelism matching SM architecture
# Use compute_optimized_3d_transpose_dot_product_kernel for new implementations
COMPUTE_OPTIMIZED_KERNEL = r"""
extern "C" __global__
void compute_optimized_transpose_dot_product_kernel(
    const float* sparse_values,     // Shape: (batch_size, channels, block_size, block_size)
    const int* block_positions,     // Shape: (batch_size, channels, 2)
    const bool* blocks_set,         // Shape: (batch_size, channels)
    const float* target_image,      // Shape: (height, width)
    float* result,                  // Shape: (batch_size, channels)
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int block_size,
    const int iteration_offset      // Which set of 8 blocks this iteration processes
) {
    // Architecture-matched design: 8 SMs, 32 cores per SM
    // Grid: (8 blocks, 1, 1) - one block per SM
    // Block: (32 threads, 1, 1) - one thread per core

    // Which (LED, channel) combination this SM is processing
    int led_channel_id = iteration_offset * 8 + blockIdx.x;

    if (led_channel_id >= batch_size * channels) return;

    int led_id = led_channel_id / channels;
    int channel_id = led_channel_id % channels;
    int idx = led_id * channels + channel_id;

    // Check if this block is set
    if (!blocks_set[idx]) {
        if (threadIdx.x == 0) {
            result[idx] = 0.0f;
        }
        return;
    }

    // Get block position
    int top_row = block_positions[idx * 2 + 0];
    int top_col = block_positions[idx * 2 + 1];

    // Shared memory optimized for GPU limits (max 48KB per block)
    // Load only sparse block into shared memory, access target directly
    extern __shared__ float shared_mem[];
    float* sparse_block = shared_mem;                    // block_size * block_size floats (36KB for 96x96)
    // 32 floats (128 bytes)
    float* reduction_workspace = &shared_mem[block_size * block_size];
    // Total: 36KB + 128 bytes = ~36.1KB (well under 48KB limit)

    int block_elements = block_size * block_size;  // 9216 for 96x96
    int elements_per_thread = (block_elements + 31) / 32;  // 288 elements per thread

    // Cooperative loading: 32 threads load sparse block into shared memory
    int sparse_offset = idx * block_elements;
    for (int i = 0; i < elements_per_thread; i++) {
        int element_idx = threadIdx.x * elements_per_thread + i;
        if (element_idx < block_elements) {
            sparse_block[element_idx] = sparse_values[sparse_offset + element_idx];
        }
    }

    __syncthreads();  // Ensure sparse block is loaded

    // Compute phase: each thread processes 288 multiply-adds
    // Access target image directly (relies on L2 cache for efficiency)
    float thread_sum = 0.0f;

    for (int i = 0; i < elements_per_thread; i++) {
        int element_idx = threadIdx.x * elements_per_thread + i;
        if (element_idx < block_elements) {
            // Calculate target image coordinates on-the-fly
            int block_row = element_idx / block_size;
            int block_col = element_idx % block_size;
            int global_row = top_row + block_row;
            int global_col = top_col + block_col;

            // Bounds check and compute
            if (global_row < height && global_col < width) {
                float sparse_val = sparse_block[element_idx];
                float target_val = target_image[global_row * width + global_col];
                thread_sum += sparse_val * target_val;
            }
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

# High-parallelism CUDA kernel - many threads per LED computation
HIGH_PARALLELISM_KERNEL = r"""
extern "C" __global__
void high_parallelism_transpose_dot_product_kernel(
    const float* sparse_values,     // Shape: (batch_size, channels, block_size, block_size)
    const int* block_positions,     // Shape: (batch_size, channels, 2)
    const bool* blocks_set,         // Shape: (batch_size, channels)
    const float* target_image,      // Shape: (height, width)
    float* result,                  // Shape: (batch_size, channels)
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int block_size
) {
    // Thread block configuration: THREADS_PER_LED threads per (LED, channel)
    const int THREADS_PER_LED = 256;  // 256 threads collaborate on each LED computation

    // Which (LED, channel) combination this thread block is working on
    int led_channel_id = blockIdx.x;
    int led_id = led_channel_id / channels;
    int channel_id = led_channel_id % channels;

    if (led_id >= batch_size) return;

    int idx = led_id * channels + channel_id;

    // Check if this block is set
    if (!blocks_set[idx]) {
        if (threadIdx.x == 0) {
            result[idx] = 0.0f;
        }
        return;
    }

    // Get block position
    int top_row = block_positions[idx * 2 + 0];
    int top_col = block_positions[idx * 2 + 1];

    // Each thread processes multiple elements for parallel reduction
    int block_elements = block_size * block_size;
    int elements_per_thread = (block_elements + THREADS_PER_LED - 1) / THREADS_PER_LED;

    // Thread-local accumulator
    float thread_sum = 0.0f;

    // Each thread processes its assigned elements
    int sparse_offset = idx * block_elements;
    for (int i = 0; i < elements_per_thread; i++) {
        int element_idx = threadIdx.x * elements_per_thread + i;
        if (element_idx >= block_elements) break;

        // Calculate target image coordinates
        int block_row = element_idx / block_size;
        int block_col = element_idx % block_size;
        int global_row = top_row + block_row;
        int global_col = top_col + block_col;

        // Bounds check and compute
        if (global_row < height && global_col < width) {
            float sparse_val = sparse_values[sparse_offset + element_idx];
            float target_val = target_image[global_row * width + global_col];
            thread_sum += sparse_val * target_val;
        }
    }

    // Parallel reduction in shared memory
    __shared__ float shared_sums[256];  // THREADS_PER_LED = 256
    shared_sums[threadIdx.x] = thread_sum;
    __syncthreads();

    // Tree reduction
    for (int stride = THREADS_PER_LED / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result
    if (threadIdx.x == 0) {
        result[idx] = shared_sums[0];
    }
}
"""

# Corrected kernel with proper parallelism
CORRECTED_TRANSPOSE_DOT_PRODUCT_KERNEL = r"""
extern "C" __global__
void corrected_transpose_dot_product_kernel(
    const float* sparse_values,     // Shape: (batch_size, channels, block_size, block_size)
    const int* block_positions,     // Shape: (batch_size, channels, 2)
    const bool* blocks_set,         // Shape: (batch_size, channels)
    const float* target_image,      // Shape: (height, width)
    float* result,                  // Shape: (batch_size, channels)
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int block_size
) {
    // Grid: (batch_size * channels, 1, 1) - one block per (LED, channel)
    // Block: (256, 1, 1) - 256 threads collaborate on each dot product

    int led_channel_idx = blockIdx.x;
    if (led_channel_idx >= batch_size * channels) return;

    int led_id = led_channel_idx / channels;
    int channel_id = led_channel_idx % channels;

    // Check if this block is set
    if (!blocks_set[led_channel_idx]) {
        if (threadIdx.x == 0) {
            result[led_channel_idx] = 0.0f;
        }
        return;
    }

    // Get block position for this (LED, channel)
    int top_row = block_positions[led_channel_idx * 2 + 0];
    int top_col = block_positions[led_channel_idx * 2 + 1];

    // Shared memory for parallel reduction
    __shared__ float partial_sums[256];  // One per thread

    int block_elements = block_size * block_size;
    int elements_per_thread = (block_elements + blockDim.x - 1) / blockDim.x;

    // Each thread processes a subset of the 9216 dot product elements
    float thread_sum = 0.0f;
    int sparse_offset = led_channel_idx * block_elements;

    for (int elem = 0; elem < elements_per_thread; elem++) {
        int element_idx = threadIdx.x * elements_per_thread + elem;
        if (element_idx >= block_elements) break;

        // Calculate target image coordinates for this element
        int block_row = element_idx / block_size;
        int block_col = element_idx % block_size;
        int global_row = top_row + block_row;
        int global_col = top_col + block_col;

        // Bounds check and compute
        if (global_row >= 0 && global_row < height &&
            global_col >= 0 && global_col < width) {
            float sparse_val = sparse_values[sparse_offset + element_idx];
            float target_val = target_image[global_row * width + global_col];
            thread_sum += sparse_val * target_val;
        }
    }

    // Store partial sum in shared memory
    partial_sums[threadIdx.x] = thread_sum;
    __syncthreads();

    // Parallel reduction: 256 -> 1
    // Tree reduction for efficiency
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result
    if (threadIdx.x == 0) {
        result[led_channel_idx] = partial_sums[0];
    }
}
"""

# ARCHIVED: Original kernel - INCORRECT AND DEPRECATED
#
# This kernel has a fundamental shared memory race condition where multiple threads
# overwrite the same shared memory with different target blocks. While it sometimes
# produces reasonable results due to cache effects, it is mathematically incorrect.
#
# Key problems:
# 1. Shared memory race: threads load different spatial regions into same memory
# 2. Last-writer-wins behavior creates non-deterministic results
# 3. Works accidentally due to spatial coherence in LED patterns
#
# Archived on 2024-06-23. Use corrected_transpose_dot_product_kernel instead.
DEPRECATED_TRANSPOSE_DOT_PRODUCT_KERNEL = r"""
extern "C" __global__
void deprecated_transpose_dot_product_kernel(
    const float* sparse_values,
    const int* block_positions,
    const bool* blocks_set,
    const float* target_image,
    float* result,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int block_size
) {
    // DEPRECATED: This kernel has shared memory race conditions
    // DO NOT USE - kept for reference only

    int led_id = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (led_id >= batch_size || channel_id >= channels) {
        return;
    }

    int idx = led_id * channels + channel_id;

    if (!blocks_set[idx]) {
        result[idx] = 0.0f;
        return;
    }

    int top_row = block_positions[idx * 2 + 0];
    int top_col = block_positions[idx * 2 + 1];

    // RACE CONDITION: Multiple threads overwrite same shared memory
    __shared__ float target_block[96 * 96];

    int threads_per_block = blockDim.x * blockDim.y;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int block_elements = block_size * block_size;

    // INCORRECT: Each thread loads from different (top_row, top_col)
    for (int i = thread_id; i < block_elements; i += threads_per_block) {
        int block_row = i / block_size;
        int block_col = i % block_size;
        int global_row = top_row + block_row;  // Different per thread!
        int global_col = top_col + block_col;  // Different per thread!

        if (global_row < height && global_col < width) {
            target_block[i] = target_image[global_row * width + global_col];
        } else {
            target_block[i] = 0.0f;
        }
    }

    __syncthreads();

    float sum = 0.0f;
    int sparse_offset = idx * block_size * block_size;

    for (int i = 0; i < block_elements; i++) {
        sum += sparse_values[sparse_offset + i] * target_block[i];
    }

    result[idx] = sum;
}
"""

# Compile the kernel once on module import
_kernel_cache = {}


def get_corrected_transpose_dot_product_kernel():
    """Get the compiled corrected CUDA kernel for transpose dot product."""
    if "corrected_transpose_dot_product" not in _kernel_cache:
        logger.info("Compiling corrected CUDA kernel for transpose_dot_product...")

        # Compile the kernel
        kernel = cp.RawKernel(
            CORRECTED_TRANSPOSE_DOT_PRODUCT_KERNEL,
            "corrected_transpose_dot_product_kernel",
        )
        _kernel_cache["corrected_transpose_dot_product"] = kernel

        logger.info("Corrected CUDA kernel compiled successfully")

    return _kernel_cache["corrected_transpose_dot_product"]


def get_deprecated_transpose_dot_product_kernel():
    """DEPRECATED: Get the original flawed kernel (for reference only)."""
    logger.warning("Using DEPRECATED kernel with race conditions - for reference only!")
    if "deprecated_transpose_dot_product" not in _kernel_cache:
        kernel = cp.RawKernel(
            DEPRECATED_TRANSPOSE_DOT_PRODUCT_KERNEL,
            "deprecated_transpose_dot_product_kernel",
        )
        _kernel_cache["deprecated_transpose_dot_product"] = kernel

    return _kernel_cache["deprecated_transpose_dot_product"]


def get_high_parallelism_kernel():
    """Get the compiled high-parallelism CUDA kernel."""
    if "high_parallelism" not in _kernel_cache:
        logger.info("Compiling high-parallelism CUDA kernel...")

        # Compile the kernel
        kernel = cp.RawKernel(
            HIGH_PARALLELISM_KERNEL, "high_parallelism_transpose_dot_product_kernel"
        )
        _kernel_cache["high_parallelism"] = kernel

        logger.info("High-parallelism CUDA kernel compiled successfully")

    return _kernel_cache["high_parallelism"]


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


def get_compute_optimized_kernel():
    """DEPRECATED: Get the compiled compute-optimized CUDA kernel (2D only)."""
    logger.warning(
        "get_compute_optimized_kernel is DEPRECATED - use get_compute_optimized_3d_kernel for new implementations"
    )
    if "compute_optimized" not in _kernel_cache:
        logger.info("Compiling compute-optimized CUDA kernel...")

        # Compile the kernel
        kernel = cp.RawKernel(
            COMPUTE_OPTIMIZED_KERNEL, "compute_optimized_transpose_dot_product_kernel"
        )
        _kernel_cache["compute_optimized"] = kernel

        logger.info("Compute-optimized CUDA kernel compiled successfully")

    return _kernel_cache["compute_optimized"]


def cuda_transpose_dot_product_corrected(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    blocks_set: cp.ndarray,
    target_image: cp.ndarray,
    batch_size: int,
    channels: int,
    block_size: int,
) -> cp.ndarray:
    """
    Corrected CUDA kernel wrapper for A^T @ b operation with proper parallelism.

    Args:
        sparse_values: Dense blocks, shape (batch_size, channels, block_size, block_size)
        block_positions: Block positions, shape (batch_size, channels, 2)
        blocks_set: Block set flags, shape (batch_size, channels)
        target_image: Target image, shape (height, width)
        batch_size: Number of LEDs
        channels: Number of channels
        block_size: Size of square blocks

    Returns:
        Result of A^T @ b, shape (batch_size, channels)
    """
    height, width = target_image.shape

    # Prepare output array
    result = cp.zeros((batch_size, channels), dtype=cp.float32)

    # Get compiled kernel
    kernel = get_corrected_transpose_dot_product_kernel()

    # Configure grid and block dimensions
    # Grid: one block per (LED, channel) combination
    # Block: 256 threads collaborating on each dot product
    total_led_channels = batch_size * channels
    threads_per_block = 256

    grid_size = (total_led_channels,)
    block_size_1d = (threads_per_block,)

    logger.debug(
        f"Corrected CUDA kernel launch: grid={grid_size}, block={block_size_1d}"
    )
    logger.debug(
        f"Total blocks: {total_led_channels}, Threads per block: {threads_per_block}"
    )

    # Launch kernel
    kernel(
        grid_size,
        block_size_1d,
        (
            sparse_values.ravel(),  # Flatten to 1D
            block_positions.ravel(),  # Flatten to 1D
            blocks_set.ravel(),  # Flatten to 1D
            target_image,  # 2D target image
            result.ravel(),  # Flatten result to 1D
            batch_size,
            channels,
            height,
            width,
            block_size,
        ),
    )

    return result


# High-performance kernel targeting 20+ GFLOPS for 2600+ LEDs
HIGH_PERFORMANCE_TRANSPOSE_DOT_PRODUCT_KERNEL = r"""
extern "C" __global__
void high_performance_transpose_dot_product_kernel(
    const float* sparse_values,     // Shape: (batch_size, channels, block_size, block_size)
    const int* block_positions,     // Shape: (batch_size, channels, 2)
    const bool* blocks_set,         // Shape: (batch_size, channels)
    const float* target_image,      // Shape: (height, width)
    float* result,                  // Shape: (batch_size, channels)
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int block_size
) {
    // High-performance design: Each thread block processes multiple (LED, channel) pairs
    // Grid: (total_pairs / PAIRS_PER_BLOCK, 1, 1)
    // Block: (512, 1, 1) - large blocks for better occupancy

    const int THREADS_PER_BLOCK = 512;
    // Process 1 (LED, channel) pair per thread block for memory efficiency
    const int PAIRS_PER_BLOCK = 1;

    // Shared memory within 48KB limit: 1 target block + reduction array
    __shared__ float target_block[96 * 96];     // 1 block * 96*96 * 4 bytes = 36KB
    __shared__ float partial_results[512];     // Parallel reduction array = 2KB
    // Total: 38KB (well within 48KB limit)

    int pair_idx = blockIdx.x;  // One pair per block
    int thread_id = threadIdx.x;

    if (pair_idx >= batch_size * channels) {
        return;
    }

    int led_id = pair_idx / channels;
    int channel_id = pair_idx % channels;

    // Check if this block is set
    if (!blocks_set[pair_idx]) {
        if (thread_id == 0) {
            result[pair_idx] = 0.0f;
        }
        return;
    }

    // Get block position
    int top_row = block_positions[pair_idx * 2 + 0];
    int top_col = block_positions[pair_idx * 2 + 1];

    // Cooperatively load target block into shared memory
    int block_elements = block_size * block_size;
    int elements_per_thread = (block_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int elem = 0; elem < elements_per_thread; elem++) {
        int element_idx = thread_id * elements_per_thread + elem;
        if (element_idx < block_elements) {
            int block_row = element_idx / block_size;
            int block_col = element_idx % block_size;
            int global_row = top_row + block_row;
            int global_col = top_col + block_col;

            if (global_row >= 0 && global_row < height &&
                global_col >= 0 && global_col < width) {
                target_block[element_idx] = target_image[global_row * width + global_col];
            } else {
                target_block[element_idx] = 0.0f;
            }
        }
    }

    __syncthreads();  // Ensure target block is loaded

    // Compute dot product with parallel reduction
    float thread_sum = 0.0f;

    for (int elem = 0; elem < elements_per_thread; elem++) {
        int element_idx = thread_id * elements_per_thread + elem;
        if (element_idx < block_elements) {
            // Calculate 4D sparse array index:
            // [led_id][channel_id][block_row][block_col]
            int block_row = element_idx / block_size;
            int block_col = element_idx % block_size;
            int sparse_4d_idx = ((led_id * channels + channel_id) * block_size + block_row) * block_size + block_col;

            float sparse_val = sparse_values[sparse_4d_idx];
            float target_val = target_block[element_idx];
            thread_sum += sparse_val * target_val;
        }
    }

    // Store partial sum
    partial_results[thread_id] = thread_sum;
    __syncthreads();

    // Parallel reduction: 512 -> 1
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
        if (thread_id < stride) {
            partial_results[thread_id] += partial_results[thread_id + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (thread_id == 0) {
        result[pair_idx] = partial_results[0];
    }
}
"""


def get_high_performance_kernel():
    """Get the compiled high-performance CUDA kernel targeting 20+ GFLOPS."""
    if "high_performance_transpose_dot_product" not in _kernel_cache:
        logger.info(
            "Compiling high-performance CUDA kernel for transpose_dot_product..."
        )

        # Compile the kernel
        kernel = cp.RawKernel(
            HIGH_PERFORMANCE_TRANSPOSE_DOT_PRODUCT_KERNEL,
            "high_performance_transpose_dot_product_kernel",
        )
        _kernel_cache["high_performance_transpose_dot_product"] = kernel

        logger.info("High-performance CUDA kernel compiled successfully")

    return _kernel_cache["high_performance_transpose_dot_product"]


def cuda_transpose_dot_product_high_performance(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    blocks_set: cp.ndarray,
    target_image: cp.ndarray,
    batch_size: int,
    channels: int,
    block_size: int,
) -> cp.ndarray:
    """
    High-performance CUDA kernel wrapper targeting 20+ GFLOPS for 2600+ LEDs.

    Args:
        sparse_values: Dense blocks, shape (batch_size, channels, block_size, block_size)
        block_positions: Block positions, shape (batch_size, channels, 2)
        blocks_set: Block set flags, shape (batch_size, channels)
        target_image: Target image, shape (height, width)
        batch_size: Number of LEDs
        channels: Number of channels
        block_size: Size of square blocks

    Returns:
        Result of A^T @ b, shape (batch_size, channels)
    """
    height, width = target_image.shape

    # Prepare output array
    result = cp.zeros((batch_size, channels), dtype=cp.float32)

    # Get compiled kernel
    kernel = get_high_performance_kernel()

    # Configure grid and block dimensions for high performance
    total_pairs = batch_size * channels
    pairs_per_block = 1  # Must match kernel constant
    threads_per_block = 512  # Must match kernel constant

    num_blocks = total_pairs  # One block per (LED, channel) pair

    grid_size = (num_blocks,)
    block_size_1d = (threads_per_block,)

    # Calculate shared memory size
    # 1 target block (96x96) + 1 reduction array (512)
    shared_mem_size = (
        block_size * block_size + threads_per_block
    ) * 4  # 4 bytes per float

    logger.debug(f"High-performance CUDA kernel launch:")
    logger.debug(f"  Grid: {grid_size}, Block: {block_size_1d}")
    logger.debug(f"  Processing {total_pairs} (LED, channel) pairs")
    logger.debug(f"  Shared memory: {shared_mem_size/1024:.1f}KB")

    # Launch kernel (shared memory is statically allocated in kernel)
    kernel(
        grid_size,
        block_size_1d,
        (
            sparse_values.ravel(),  # Flatten to 1D
            block_positions.ravel(),  # Flatten to 1D
            blocks_set.ravel(),  # Flatten to 1D
            target_image,  # 2D target image
            result.ravel(),  # Flatten result to 1D
            batch_size,
            channels,
            height,
            width,
            block_size,
        ),
    )

    return result


# DEPRECATED: Remove these old 2D CUDA functions - use 3D versions
def cuda_transpose_dot_product(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    blocks_set: cp.ndarray,
    target_image: cp.ndarray,
    batch_size: int,
    channels: int,
    block_size: int,
) -> cp.ndarray:
    """
    DEPRECATED: Use cuda_transpose_dot_product_3d_compute_optimized instead.

    This function uses a flawed kernel with race conditions and requires blocks_set parameter.
    """
    logger.warning(
        "cuda_transpose_dot_product is DEPRECATED - use cuda_transpose_dot_product_3d_compute_optimized"
    )
    # Fall back to corrected version
    return cuda_transpose_dot_product_corrected(
        sparse_values,
        block_positions,
        blocks_set,
        target_image,
        batch_size,
        channels,
        block_size,
    )


def cuda_transpose_dot_product_high_parallelism(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    blocks_set: cp.ndarray,
    target_image: cp.ndarray,
    batch_size: int,
    channels: int,
    block_size: int,
) -> cp.ndarray:
    """
    High-parallelism CUDA kernel wrapper for A^T @ b operation.

    Uses 256 threads per LED computation for much higher GPU utilization.

    Args:
        sparse_values: Dense blocks, shape (batch_size, channels, block_size, block_size)
        block_positions: Block positions, shape (batch_size, channels, 2)
        blocks_set: Block set flags, shape (batch_size, channels)
        target_image: Target image, shape (height, width)
        batch_size: Number of LEDs
        channels: Number of channels
        block_size: Size of square blocks

    Returns:
        Result of A^T @ b, shape (batch_size, channels)
    """
    height, width = target_image.shape

    # Prepare output array
    result = cp.zeros((batch_size, channels), dtype=cp.float32)

    # Get compiled kernel
    kernel = get_high_parallelism_kernel()

    # Configure grid and block dimensions for high parallelism
    # Each thread block handles one (LED, channel) combination with 256 threads
    total_led_channels = batch_size * channels
    threads_per_block = 256  # THREADS_PER_LED = 256
    num_blocks = total_led_channels

    grid_size = (num_blocks,)
    block_size_1d = (threads_per_block,)

    logger.debug(
        f"High-parallelism CUDA kernel launch: grid={grid_size}, block={block_size_1d}"
    )
    logger.debug(f"Total threads: {num_blocks * threads_per_block:,}")

    # Launch kernel
    kernel(
        grid_size,
        block_size_1d,
        (
            sparse_values.ravel(),  # Flatten to 1D
            block_positions.ravel(),  # Flatten to 1D
            blocks_set.ravel(),  # Flatten to 1D
            target_image,  # 2D target image
            result.ravel(),  # Flatten result to 1D
            batch_size,
            channels,
            height,
            width,
            block_size,
        ),
    )

    return result


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

    Uses 8-way parallelism matching SM architecture with optimized memory access patterns.
    Processes 3D planar input (channels, height, width) in one operation.

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

    # Get compiled kernel
    kernel = get_compute_optimized_3d_kernel()

    # Architecture-matched configuration: 8 SMs, 32 cores per SM
    total_led_channels = batch_size * channels
    sms_count = 8
    cores_per_sm = 32

    # Calculate number of iterations needed to process all blocks
    blocks_per_iteration = sms_count  # 8 blocks processed in parallel
    num_iterations = (
        total_led_channels + blocks_per_iteration - 1
    ) // blocks_per_iteration

    grid_size = (sms_count,)  # 8 blocks, one per SM
    block_size_1d = (cores_per_sm,)  # 32 threads, one per core

    logger.debug(f"3D Compute-optimized CUDA kernel launch:")
    logger.debug(f"  Grid: {grid_size}, Block: {block_size_1d}")
    logger.debug(f"  Iterations: {num_iterations}")
    logger.debug(f"  Target shape: {target_3d.shape} (planar)")
    logger.debug(f"  Total threads per iteration: {sms_count * cores_per_sm}")
    logger.debug(
        f"  Total compute threads: {num_iterations * sms_count * cores_per_sm:,}"
    )

    # Calculate shared memory size needed
    # 1 sparse block + 32 reduction values (target accessed directly for better L2 cache usage)
    shared_mem_size = (block_size * block_size + 32) * 4  # 4 bytes per float

    logger.debug(
        f"  Shared memory per block: {shared_mem_size} bytes ({shared_mem_size/1024:.1f}KB)"
    )

    # Launch kernel for each iteration
    for iteration in range(num_iterations):
        iteration_offset = iteration

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
                iteration_offset,
            ),
            shared_mem=shared_mem_size,
        )

        # Synchronize between iterations to ensure correct ordering
        cp.cuda.Device().synchronize()

    return result


def cuda_transpose_dot_product_compute_optimized(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    blocks_set: cp.ndarray,
    target_image: cp.ndarray,
    batch_size: int,
    channels: int,
    block_size: int,
) -> cp.ndarray:
    """
    DEPRECATED: Compute-optimized CUDA kernel wrapper for A^T @ b operation (2D only).

    Uses 8-way parallelism matching SM architecture with optimized memory access patterns.
    Targets ~14 GFLOPS performance by maximizing compute throughput.

    DEPRECATED: Use cuda_transpose_dot_product_3d_compute_optimized for new implementations.

    Args:
        sparse_values: Dense blocks, shape (batch_size, channels, block_size, block_size)
        block_positions: Block positions, shape (batch_size, channels, 2)
        blocks_set: Block set flags, shape (batch_size, channels)
        target_image: Target image, shape (height, width)
        batch_size: Number of LEDs
        channels: Number of channels
        block_size: Size of square blocks

    Returns:
        Result of A^T @ b, shape (batch_size, channels)
    """
    logger.warning(
        "cuda_transpose_dot_product_compute_optimized is DEPRECATED - use cuda_transpose_dot_product_3d_compute_optimized"
    )

    height, width = target_image.shape

    # Prepare output array
    result = cp.zeros((batch_size, channels), dtype=cp.float32)

    # Get compiled kernel
    kernel = get_compute_optimized_kernel()

    # Architecture-matched configuration: 8 SMs, 32 cores per SM
    total_led_channels = batch_size * channels
    sms_count = 8
    cores_per_sm = 32

    # Calculate number of iterations needed to process all blocks
    blocks_per_iteration = sms_count  # 8 blocks processed in parallel
    num_iterations = (
        total_led_channels + blocks_per_iteration - 1
    ) // blocks_per_iteration

    grid_size = (sms_count,)  # 8 blocks, one per SM
    block_size_1d = (cores_per_sm,)  # 32 threads, one per core

    logger.debug(f"Compute-optimized CUDA kernel launch:")
    logger.debug(f"  Grid: {grid_size}, Block: {block_size_1d}")
    logger.debug(f"  Iterations: {num_iterations}")
    logger.debug(f"  Total threads per iteration: {sms_count * cores_per_sm}")
    logger.debug(
        f"  Total compute threads: {num_iterations * sms_count * cores_per_sm:,}"
    )

    # Calculate shared memory size needed
    # 1 sparse block + 32 reduction values (target accessed directly for better L2 cache usage)
    shared_mem_size = (block_size * block_size + 32) * 4  # 4 bytes per float

    logger.debug(
        f"  Shared memory per block: {shared_mem_size} bytes ({shared_mem_size/1024:.1f}KB)"
    )

    # Launch kernel for each iteration
    for iteration in range(num_iterations):
        iteration_offset = iteration

        kernel(
            grid_size,
            block_size_1d,
            (
                sparse_values.ravel(),  # Flatten to 1D
                block_positions.ravel(),  # Flatten to 1D
                blocks_set.ravel(),  # Flatten to 1D
                target_image,  # 2D target image
                result.ravel(),  # Flatten result to 1D
                batch_size,
                channels,
                height,
                width,
                block_size,
                iteration_offset,
            ),
            shared_mem=shared_mem_size,
        )

        # Synchronize between iterations to ensure correct ordering
        cp.cuda.Device().synchronize()

    return result


def benchmark_cuda_kernel(
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    blocks_set: cp.ndarray,
    target_images: list,
    batch_size: int,
    channels: int,
    block_size: int,
) -> Tuple[float, cp.ndarray]:
    """
    Benchmark the CUDA kernel implementation.

    Args:
        sparse_values: Dense blocks
        block_positions: Block positions
        blocks_set: Block set flags
        target_images: List of target images for benchmarking
        batch_size: Number of LEDs
        channels: Number of channels
        block_size: Block size

    Returns:
        Tuple of (average_time_seconds, sample_result)
    """
    # Warm up
    _ = cuda_transpose_dot_product(
        sparse_values,
        block_positions,
        blocks_set,
        target_images[0],
        batch_size,
        channels,
        block_size,
    )
    cp.cuda.Device().synchronize()

    # Benchmark runs
    times = []
    for i in range(1, len(target_images)):
        cp.cuda.Device().synchronize()
        start_time = cp.cuda.Event()
        end_time = cp.cuda.Event()

        start_time.record()
        result = cuda_transpose_dot_product(
            sparse_values,
            block_positions,
            blocks_set,
            target_images[i],
            batch_size,
            channels,
            block_size,
        )
        end_time.record()

        cp.cuda.Device().synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_time, end_time)
        times.append(elapsed_ms / 1000.0)  # Convert to seconds

    avg_time = np.mean(times)
    return avg_time, result
