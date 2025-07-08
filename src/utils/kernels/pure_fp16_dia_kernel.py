#!/usr/bin/env python3
"""
Pure FP16 CUDA kernel for 3D DIA matrix-vector multiplication.

This kernel operates entirely in FP16 precision for maximum performance:
- Matrix data: FP16
- Input vectors: FP16
- Output vectors: FP16
- Computation: FP16 with potential tensor core utilization

No hidden conversions - all inputs must be the exact expected types.
"""

from typing import Tuple

import cupy
import numpy as np

# Pure FP16 3D DIA kernel - all operations in FP16
PURE_FP16_3D_DIA_KERNEL = r"""
#include <cuda_fp16.h>

extern "C" __global__
void pure_fp16_3d_dia_kernel(
    const __half* __restrict__ data,     // 3D DIA matrix data: shape (channels, num_bands, n) - FP16
    const int* __restrict__ offsets,     // Band offsets: shape (num_bands,) - shared across channels
    const __half* __restrict__ x,        // Input vectors: shape (channels, n) - FP16
    __half* __restrict__ y,              // Output vectors: shape (channels, n) - FP16
    const int n,                         // Matrix dimension (number of LEDs)
    const int num_bands,                 // Number of diagonal bands
    const int channels                   // Number of channels (3 for RGB)
) {
    // 2D grid: blockIdx.x covers LED indices, blockIdx.y covers channels
    const int led_idx = blockIdx.x * blockDim.x + threadIdx.x;  // LED index [0, n)
    const int channel = blockIdx.y;                             // Channel index [0, channels)

    // Bounds checking
    if (led_idx >= n || channel >= channels) return;

    // Use FP16 accumulation for maximum performance
    __half sum = __float2half(0.0f);

    // Calculate base pointer for this channel's matrix data
    // data layout: (channels, num_bands, n)
    const __half* data_channel = data + channel * num_bands * n;
    const __half* x_channel = x + channel * n;

    // Iterate through all diagonal bands
    for (int band = 0; band < num_bands; band++) {
        const int offset = offsets[band];
        const int j = led_idx + offset;  // Column index

        // Check bounds for this diagonal element
        if (j >= 0 && j < n) {
            // 3D DIA format: A[channel,led_idx,j] stored at data_channel[band * n + j]
            const __half matrix_val = data_channel[band * n + j];

            // Skip zero elements (use direct comparison)
            if (__hne(matrix_val, __float2half(0.0f))) {
                const __half x_val = x_channel[j];
                sum = __hadd(sum, __hmul(matrix_val, x_val));
            }
        }
    }

    // Write result: y[channel, led_idx] = sum (already FP16)
    y[channel * n + led_idx] = sum;
}
"""

# Optimized version with shared memory and FP16 operations
PURE_FP16_3D_DIA_OPTIMIZED_KERNEL = r"""
#include <cuda_fp16.h>

extern "C" __global__
void pure_fp16_3d_dia_optimized_kernel(
    const __half* __restrict__ data,     // 3D DIA matrix data: shape (channels, num_bands, n) - FP16
    const int* __restrict__ offsets,     // Band offsets: shape (num_bands,) - shared across channels
    const __half* __restrict__ x,        // Input vectors: shape (channels, n) - FP16
    __half* __restrict__ y,              // Output vectors: shape (channels, n) - FP16
    const int n,                         // Matrix dimension (number of LEDs)
    const int num_bands,                 // Number of diagonal bands
    const int channels                   // Number of channels (3 for RGB)
) {
    extern __shared__ __half shared_x[];  // Shared memory for FP16 vector caching

    const int tid = threadIdx.x;
    const int bid_x = blockIdx.x;
    const int block_size = blockDim.x;
    const int led_idx = bid_x * block_size + tid;
    const int channel = blockIdx.y;

    // Bounds checking for channel
    if (channel >= channels) return;

    // Calculate shared memory layout per channel
    const int max_band_offset = 1000;  // Conservative estimate for bandwidth
    const int shared_size = block_size + 2 * max_band_offset;
    const int shared_start = bid_x * block_size - max_band_offset;

    // Load vector elements into shared memory for this channel (FP16)
    const __half* x_channel = x + channel * n;

    // Cooperative loading with bounds checking - load FP16 directly
    for (int i = tid; i < shared_size; i += block_size) {
        const int global_idx = shared_start + i;
        if (global_idx >= 0 && global_idx < n) {
            shared_x[i] = x_channel[global_idx];
        } else {
            shared_x[i] = __float2half(0.0f);  // Pad with zeros
        }
    }

    __syncthreads();  // Ensure all threads have loaded their data

    // Bounds checking for LED index
    if (led_idx >= n) return;

    // Use FP16 accumulation
    __half sum = __float2half(0.0f);

    // Calculate base pointer for this channel's matrix data
    const __half* data_channel = data + channel * num_bands * n;

    // Iterate through all diagonal bands
    for (int band = 0; band < num_bands; band++) {
        const int offset = offsets[band];
        const int j = led_idx + offset;  // Column index

        // Check bounds for this diagonal element
        if (j >= 0 && j < n) {
            // 3D DIA format: A[channel,led_idx,j] stored at data_channel[band * n + j]
            const __half matrix_val = data_channel[band * n + j];

            if (__hne(matrix_val, __float2half(0.0f))) {
                // Use shared memory for vector access (FP16)
                const int shared_idx = j - shared_start;
                __half x_val;
                if (shared_idx >= 0 && shared_idx < shared_size) {
                    x_val = shared_x[shared_idx];
                } else {
                    // Fallback to global memory (should be rare)
                    x_val = x_channel[j];
                }
                sum = __hadd(sum, __hmul(matrix_val, x_val));
            }
        }
    }

    // Write result: y[channel, led_idx] = sum (already FP16)
    y[channel * n + led_idx] = sum;
}
"""


class PureFP16DIA3DKernel:
    """
    Pure FP16 3D DIA matrix-vector multiplication kernel.

    Requirements:
    - Matrix data must be cupy.ndarray with dtype=cupy.float16
    - Input vectors must be cupy.ndarray with dtype=cupy.float16
    - Offsets must be cupy.ndarray with dtype=cupy.int32
    - All arrays must already be on GPU

    No hidden conversions or type coercion.
    """

    def __init__(self, use_optimized: bool = True):
        """Initialize the pure FP16 kernel.

        Args:
            use_optimized: Use optimized kernel with shared memory
        """
        self.use_optimized = use_optimized

        # Compile kernels
        if use_optimized:
            self.kernel = cupy.RawKernel(PURE_FP16_3D_DIA_OPTIMIZED_KERNEL, "pure_fp16_3d_dia_optimized_kernel")
        else:
            self.kernel = cupy.RawKernel(PURE_FP16_3D_DIA_KERNEL, "pure_fp16_3d_dia_kernel")

    def __call__(
        self,
        dia_data_3d: cupy.ndarray,  # Must be (channels, num_bands, n) FP16
        dia_offsets: cupy.ndarray,  # Must be (num_bands,) int32
        x: cupy.ndarray,  # Must be (channels, n) FP16
    ) -> cupy.ndarray:
        """
        Perform pure FP16 3D DIA matrix-vector multiplication.

        Args:
            dia_data_3d: 3D DIA matrix data, dtype=cupy.float16, shape=(channels, num_bands, n)
            dia_offsets: Band offsets, dtype=cupy.int32, shape=(num_bands,)
            x: Input vectors, dtype=cupy.float16, shape=(channels, n)

        Returns:
            Result vectors y = A @ x, dtype=cupy.float16, shape=(channels, n)

        Raises:
            TypeError: If input arrays don't have expected dtypes
            ValueError: If input arrays don't have expected shapes
        """
        # Strict type and shape validation - no hidden conversions
        channels, num_bands, n = dia_data_3d.shape

        # Type assertions
        if dia_data_3d.dtype != cupy.float16:
            raise TypeError(f"dia_data_3d must be cupy.float16, got {dia_data_3d.dtype}")
        if dia_offsets.dtype != cupy.int32:
            raise TypeError(f"dia_offsets must be cupy.int32, got {dia_offsets.dtype}")
        if x.dtype != cupy.float16:
            raise TypeError(f"x must be cupy.float16, got {x.dtype}")

        # Device assertions
        if not isinstance(dia_data_3d, cupy.ndarray):
            raise TypeError(f"dia_data_3d must be cupy.ndarray, got {type(dia_data_3d)}")
        if not isinstance(dia_offsets, cupy.ndarray):
            raise TypeError(f"dia_offsets must be cupy.ndarray, got {type(dia_offsets)}")
        if not isinstance(x, cupy.ndarray):
            raise TypeError(f"x must be cupy.ndarray, got {type(x)}")

        # Shape assertions
        if dia_offsets.shape != (num_bands,):
            raise ValueError(f"dia_offsets shape mismatch: expected ({num_bands},), got {dia_offsets.shape}")
        if x.shape != (channels, n):
            raise ValueError(f"x shape mismatch: expected ({channels}, {n}), got {x.shape}")

        # Allocate FP16 output
        y = cupy.zeros((channels, n), dtype=cupy.float16)

        # Launch configuration for 2D grid
        block_size = 256
        grid_x = (n + block_size - 1) // block_size  # LED dimension
        grid_y = channels  # Channel dimension

        if self.use_optimized:
            # Calculate shared memory size for FP16 data
            max_band_offset = 1000  # Conservative estimate
            shared_size = block_size + 2 * max_band_offset
            shared_mem_bytes = shared_size * 2  # 2 bytes per FP16

            # Launch optimized kernel with shared memory
            self.kernel(
                (grid_x, grid_y),  # 2D grid
                (block_size,),  # 1D block
                (dia_data_3d, dia_offsets, x, y, n, num_bands, channels),
                shared_mem=shared_mem_bytes,
            )
        else:
            # Launch basic kernel
            self.kernel(
                (grid_x, grid_y),  # 2D grid
                (block_size,),  # 1D block
                (dia_data_3d, dia_offsets, x, y, n, num_bands, channels),
            )

        return y


def get_kernel_info() -> dict:
    """Get information about the pure FP16 kernel implementation."""
    return {
        "name": "Pure FP16 3D DIA Kernel",
        "precision": "float16",
        "input_types": {"matrix": "cupy.float16", "vectors": "cupy.float16", "offsets": "cupy.int32"},
        "output_type": "cupy.float16",
        "features": [
            "No hidden type conversions",
            "Pure FP16 arithmetic",
            "Shared memory optimization available",
            "2D GPU grid for channel parallelism",
            "Strict type validation",
        ],
        "memory_bandwidth_savings": "50% vs FP32",
        "potential_speedup": "Up to 2x on tensor core GPUs",
    }
