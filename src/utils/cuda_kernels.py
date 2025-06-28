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

# 3D Multi-channel CUDA kernel - direct memory access without sparse block caching
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
    // Architecture-matched design: 8 SMs, 128 cores per SM
    // Grid: (8 blocks, 1, 1) - one block per SM
    // Block: (32 threads, 1, 1) - four threads per core

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

    // Shared memory optimization: only reduction workspace (no sparse block cache)
    // Since we read each sparse/target value exactly once, caching provides no benefit
    extern __shared__ float shared_mem[];
    float* reduction_workspace = shared_mem; // 32 floats (128 bytes) - much smaller footprint
    // Total shared memory: 128 bytes (vs 36KB+ in original)

    int block_elements = block_size * block_size;  // 4096 for 64x64
    int elements_per_thread = (block_elements + 31) / 32;  // 128 elements per thread for 64x64

    // Direct computation: read sparse and target values on-demand
    // This optimizes for memory bandwidth rather than reuse
    float thread_sum = 0.0f;

    // Calculate base offset for sparse values: (channels, batch, block, block) layout
    int sparse_offset = channel_id * (batch_size * block_elements) + led_id * block_elements;

    // Calculate target channel offset for planar access
    int channel_offset = channel_id * height * width;

    // Each thread processes elements_per_thread multiply-adds directly from global memory
    for (int i = 0; i < elements_per_thread; i++) {
        int element_idx = threadIdx.x * elements_per_thread + i;
        if (element_idx < block_elements) {
            // Calculate target image coordinates on-the-fly
            int block_row = element_idx / block_size;
            int block_col = element_idx % block_size;
            int global_row = top_row + block_row;
            int global_col = top_col + block_col;

            // Bounds check and compute with direct global memory access
            if (global_row < height && global_col < width) {
                // Direct read from sparse values (no caching)
                float sparse_val = sparse_values[sparse_offset + element_idx];
                // Direct read from target image (planar access)
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

    # Architecture-matched configuration: 8 SMs, 128 cores per SM
    total_led_channels = batch_size * channels
    sms_count = 8
    cores_per_sm = 128  # 8 SMs with 128 cores each

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
    # Only reduction workspace (32 floats) - no sparse block caching
    shared_mem_size = 32 * 4  # 4 bytes per float = 128 bytes

    logger.debug(
        f"  Shared memory per block: {shared_mem_size} bytes ({shared_mem_size/1024:.3f}KB)"
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


# Experimental kernel - 2D grid single launch optimization
EXPERIMENTAL_COMPUTE_OPTIMIZED_3D_KERNEL = r"""
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
    // 2D Grid approach: blockIdx.x = led_id, blockIdx.y = channel_id
    // Grid: (batch_size, channels) - one block per (LED, channel) combination
    // Block: (32 threads, 1, 1) - optimal for warp-level parallelism

    int led_id = blockIdx.x;
    int channel_id = blockIdx.y;

    // Bounds check
    if (led_id >= batch_size || channel_id >= channels) return;

    int idx = led_id * channels + channel_id;

    // Get block position for this LED/channel from (channels, batch_size, 2) layout
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Shared memory optimization: only reduction workspace (no sparse block cache)
    // Since we read each sparse/target value exactly once, caching provides no benefit
    extern __shared__ float shared_mem[];
    float* reduction_workspace = shared_mem; // 32 floats (128 bytes) - much smaller footprint
    // Total shared memory: 128 bytes (vs 36KB+ in original)

    int block_elements = block_size * block_size;  // 4096 for 64x64
    int elements_per_thread = (block_elements + 31) / 32;  // 128 elements per thread for 64x64

    // Direct computation: read sparse and target values on-demand
    // This optimizes for memory bandwidth rather than reuse
    float thread_sum = 0.0f;

    // Calculate base offset for sparse values: (channels, batch, block, block) layout
    int sparse_offset = channel_id * (batch_size * block_elements) + led_id * block_elements;

    // Calculate target channel offset for planar access
    int channel_offset = channel_id * height * width;

    // Each thread processes elements_per_thread multiply-adds directly from global memory
    for (int i = 0; i < elements_per_thread; i++) {
        int element_idx = threadIdx.x * elements_per_thread + i;
        if (element_idx < block_elements) {
            // Calculate target image coordinates on-the-fly
            int block_row = element_idx / block_size;
            int block_col = element_idx % block_size;
            int global_row = top_row + block_row;
            int global_col = top_col + block_col;

            // Bounds check and compute with direct global memory access
            if (global_row < height && global_col < width) {
                // Direct read from sparse values (no caching)
                float sparse_val = sparse_values[sparse_offset + element_idx];
                // Direct read from target image (planar access)
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

    Experimental optimization: 2D grid single launch approach for better GPU scheduling.
    Uses a 2D grid (batch_size, channels) to eliminate multiple kernel launches and let
    the GPU scheduler handle work distribution directly. Each block processes one
    (LED, channel) combination with 32 threads collaborating on the dot product.

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
