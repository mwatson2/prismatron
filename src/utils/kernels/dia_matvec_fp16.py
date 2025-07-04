"""
Custom CUDA kernel for optimized DIA (diagonal) matrix-vector multiplication with FP16 output.

This module implements FP16 versions of the high-performance CUDA kernels specifically designed for
banded matrices in DIA format, providing memory efficiency for the A^T A matrix structure found
in the LED diffusion optimization problem.
"""

import time
from typing import Optional, Tuple

import cupy
import cupyx.scipy.sparse as cusp
import numpy as np

# CUDA kernel for DIA matrix-vector multiplication with FP16 output
DIA_MATVEC_FP16_KERNEL = r"""
#include <cuda_fp16.h>

extern "C" __global__
void dia_matvec_fp16_kernel(
    const float* __restrict__ data,      // DIA matrix data: shape (num_bands, n)
    const int* __restrict__ offsets,     // Band offsets: shape (num_bands,)
    const float* __restrict__ x,         // Input vector: shape (n,)
    __half* __restrict__ y,              // Output vector: shape (n,) - FP16
    const int n,                         // Matrix dimension
    const int num_bands                  // Number of diagonal bands
) {
    // Thread mapping: each thread computes one output element y[i]
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    float sum = 0.0f;

    // Iterate through all diagonal bands
    for (int band = 0; band < num_bands; band++) {
        const int offset = offsets[band];
        const int j = i + offset;  // Column index: A[i,j] where j = i + offset

        // Check bounds for this diagonal element
        if (j >= 0 && j < n) {
            // DIA format: A[i,j] is stored at data[band, j] (indexed by column j)
            // Memory layout: data is stored as (num_bands, n) in row-major order
            const float matrix_val = data[band * n + j];
            if (matrix_val != 0.0f) {  // Skip explicit zeros
                sum += matrix_val * x[j];
            }
        }
    }

    // Convert to FP16 and store
    y[i] = __float2half(sum);
}
"""

# Optimized kernel with shared memory for better memory bandwidth and FP16 output
DIA_MATVEC_OPTIMIZED_FP16_KERNEL = r"""
#include <cuda_fp16.h>

extern "C" __global__
void dia_matvec_optimized_fp16_kernel(
    const float* __restrict__ data,      // DIA matrix data: shape (num_bands, n)
    const int* __restrict__ offsets,     // Band offsets: shape (num_bands,)
    const float* __restrict__ x,         // Input vector: shape (n,)
    __half* __restrict__ y,              // Output vector: shape (n,) - FP16
    const int n,                         // Matrix dimension
    const int num_bands                  // Number of diagonal bands
) {
    extern __shared__ float shared_x[];  // Shared memory for vector caching

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int i = bid * block_size + tid;

    // Load vector elements into shared memory with overlap for band access
    const int shared_size = block_size + 2 * 50;  // +/- 50 max band offset
    const int shared_start = bid * block_size - 50;

    // Cooperative loading of vector elements
    for (int s = tid; s < shared_size; s += block_size) {
        const int global_idx = shared_start + s;
        if (global_idx >= 0 && global_idx < n) {
            shared_x[s] = x[global_idx];
        } else {
            shared_x[s] = 0.0f;
        }
    }

    __syncthreads();

    if (i >= n) return;

    float sum = 0.0f;

    // Iterate through all diagonal bands
    for (int band = 0; band < num_bands; band++) {
        const int offset = offsets[band];
        const int j = i + offset;  // Column index: A[i,j] where j = i + offset

        // Check bounds for this diagonal element
        if (j >= 0 && j < n) {
            // DIA format: A[i,j] is stored at data[band, j] (indexed by column j)
            const float matrix_val = data[band * n + j];
            if (matrix_val != 0.0f) {  // Skip explicit zeros
                // Use shared memory if j is in cached range
                const int shared_idx = j - shared_start;
                float x_val;
                if (shared_idx >= 0 && shared_idx < shared_size) {
                    x_val = shared_x[shared_idx];
                } else {
                    x_val = x[j];  // Fallback to global memory
                }
                sum += matrix_val * x_val;
            }
        }
    }

    // Convert to FP16 and store
    y[i] = __float2half(sum);
}
"""

# 3D DIA kernel for multi-channel matrix-vector multiplication with FP16 output
DIA_MATVEC_3D_FP16_KERNEL = r"""
#include <cuda_fp16.h>

extern "C" __global__
void dia_matvec_3d_fp16_kernel(
    const float* __restrict__ data,      // 3D DIA matrix data: shape (channels, num_bands, n)
    const int* __restrict__ offsets,     // Band offsets: shape (num_bands,) - shared across channels
    const float* __restrict__ x,         // Input vectors: shape (channels, n) - channel-major layout
    __half* __restrict__ y,              // Output vectors: shape (channels, n) - channel-major layout, FP16
    const int n,                         // Matrix dimension (number of LEDs)
    const int num_bands,                 // Number of diagonal bands
    const int channels                   // Number of channels (3 for RGB)
) {
    // 2D grid: blockIdx.x covers LED indices, blockIdx.y covers channels
    const int led_idx = blockIdx.x * blockDim.x + threadIdx.x;  // LED index [0, n)
    const int channel = blockIdx.y;                             // Channel index [0, channels)

    // Bounds checking
    if (led_idx >= n || channel >= channels) return;

    float sum = 0.0f;

    // Calculate base pointers for this channel
    // data: (channels, num_bands, n) -> channel * (num_bands * n) + band * n + led
    // x: (channels, n) -> channel * n + led
    const float* data_channel = data + channel * num_bands * n;  // Points to start of this channel's data
    const float* x_channel = x + channel * n;                   // Points to start of this channel's input vector

    // Iterate through all diagonal bands for this channel
    for (int band = 0; band < num_bands; band++) {
        const int offset = offsets[band];
        const int j = led_idx + offset;  // Column index: A[led_idx,j] where j = led_idx + offset

        // Check bounds for this diagonal element
        if (j >= 0 && j < n) {
            // 3D DIA format: A[channel,led_idx,j] is stored at data_channel[band * n + j]
            // Memory layout: data_channel points to (num_bands, n) for this channel
            const float matrix_val = data_channel[band * n + j];
            if (matrix_val != 0.0f) {  // Skip explicit zeros
                sum += matrix_val * x_channel[j];
            }
        }
    }

    // Write result: y[channel, led_idx] = sum (convert to FP16)
    // y: (channels, n) -> channel * n + led_idx
    y[channel * n + led_idx] = __float2half(sum);
}
"""

# Optimized 3D DIA kernel with shared memory and FP16 output
DIA_MATVEC_3D_OPTIMIZED_FP16_KERNEL = r"""
#include <cuda_fp16.h>

extern "C" __global__
void dia_matvec_3d_optimized_fp16_kernel(
    const float* __restrict__ data,      // 3D DIA matrix data: shape (channels, num_bands, n)
    const int* __restrict__ offsets,     // Band offsets: shape (num_bands,) - shared across channels
    const float* __restrict__ x,         // Input vectors: shape (channels, n) - channel-major layout
    __half* __restrict__ y,              // Output vectors: shape (channels, n) - channel-major layout, FP16
    const int n,                         // Matrix dimension (number of LEDs)
    const int num_bands,                 // Number of diagonal bands
    const int channels                   // Number of channels (3 for RGB)
) {
    extern __shared__ float shared_mem[];  // Shared memory for vector caching

    // 2D grid: blockIdx.x covers LED indices, blockIdx.y covers channels
    const int tid = threadIdx.x;
    const int bid_x = blockIdx.x;
    const int channel = blockIdx.y;
    const int block_size = blockDim.x;
    const int led_idx = bid_x * block_size + tid;

    // Bounds checking for channel
    if (channel >= channels) return;

    // Calculate shared memory layout per channel
    // Each channel gets its own section of shared memory for vector caching
    const int max_band_offset = 1000;  // Estimate based on typical bandwidth
    const int shared_size = block_size + 2 * max_band_offset;  // +/- max_band_offset
    const int shared_start = bid_x * block_size - max_band_offset;

    // Load vector elements into shared memory for this channel
    const float* x_channel = x + channel * n;
    float* shared_x = shared_mem;  // Each block uses shared memory independently

    // Cooperative loading with bounds checking
    for (int i = tid; i < shared_size; i += block_size) {
        const int global_idx = shared_start + i;
        if (global_idx >= 0 && global_idx < n) {
            shared_x[i] = x_channel[global_idx];
        } else {
            shared_x[i] = 0.0f;  // Pad with zeros
        }
    }

    __syncthreads();  // Ensure all threads have loaded their data

    // Bounds checking for LED index
    if (led_idx >= n) return;

    float sum = 0.0f;

    // Calculate base pointer for this channel's matrix data
    const float* data_channel = data + channel * num_bands * n;

    // Iterate through all diagonal bands
    for (int band = 0; band < num_bands; band++) {
        const int offset = offsets[band];
        const int j = led_idx + offset;  // Column index

        // Check bounds for this diagonal element
        if (j >= 0 && j < n) {
            // 3D DIA format: A[channel,led_idx,j] stored at data_channel[band * n + j]
            const float matrix_val = data_channel[band * n + j];
            if (matrix_val != 0.0f) {
                // Use shared memory for vector access
                const int shared_idx = j - shared_start;
                if (shared_idx >= 0 && shared_idx < shared_size) {
                    sum += matrix_val * shared_x[shared_idx];
                } else {
                    // Fallback to global memory (should be rare)
                    sum += matrix_val * x_channel[j];
                }
            }
        }
    }

    // Write result: y[channel, led_idx] = sum (convert to FP16)
    y[channel * n + led_idx] = __float2half(sum);
}
"""


class CustomDIAMatVecFP16:
    """Custom CUDA implementation for DIA matrix-vector multiplication with FP16 output."""

    def __init__(self, use_optimized: bool = True):
        """Initialize the custom DIA FP16 kernel.

        Args:
            use_optimized: Use optimized kernel with shared memory
        """
        self.use_optimized = use_optimized

        # Compile kernels
        if use_optimized:
            self.kernel = cupy.RawKernel(DIA_MATVEC_OPTIMIZED_FP16_KERNEL, "dia_matvec_optimized_fp16_kernel")
        else:
            self.kernel = cupy.RawKernel(DIA_MATVEC_FP16_KERNEL, "dia_matvec_fp16_kernel")

    def __call__(self, dia_matrix, x: cupy.ndarray) -> cupy.ndarray:
        """Perform DIA matrix-vector multiplication using custom kernel with FP16 output.

        Args:
            dia_matrix: DIA format sparse matrix (CuPy or SciPy)
            x: Input vector (CuPy array)

        Returns:
            Result vector y = A @ x (FP16)
        """
        # Convert to CuPy DIA format if needed
        if not isinstance(dia_matrix, cusp.dia_matrix):
            if hasattr(dia_matrix, "tocupy"):
                dia_matrix = dia_matrix.tocupy()
            else:
                dia_matrix = cusp.dia_matrix(dia_matrix)

        n = dia_matrix.shape[0]
        num_bands = len(dia_matrix.offsets)

        # Prepare data arrays
        data = cupy.asarray(dia_matrix.data, dtype=cupy.float32)  # Shape: (num_bands, n)
        offsets = cupy.asarray(dia_matrix.offsets, dtype=cupy.int32)  # Shape: (num_bands,)
        x_input = cupy.asarray(x, dtype=cupy.float32)  # Shape: (n,)
        y_output = cupy.zeros(n, dtype=cupy.float16)  # Shape: (n,) - FP16 output

        # Launch configuration
        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        if self.use_optimized:
            # Shared memory size: (block_size + 2*50) * sizeof(float)
            shared_mem_size = (block_size + 100) * 4

            self.kernel(
                (grid_size,),
                (block_size,),
                (data, offsets, x_input, y_output, n, num_bands),
                shared_mem=shared_mem_size,
            )
        else:
            self.kernel(
                (grid_size,),
                (block_size,),
                (data, offsets, x_input, y_output, n, num_bands),
            )

        return y_output


class CustomDIA3DMatVecFP16:
    """Custom CUDA implementation for 3D DIA matrix-vector multiplication with FP16 output.

    Handles multi-channel (RGB) DIA matrices efficiently using 2D GPU grid:
    - blockIdx.x: LED indices
    - blockIdx.y: Channel indices
    """

    def __init__(self, use_optimized: bool = True):
        """Initialize the custom 3D DIA FP16 kernel.

        Args:
            use_optimized: Use optimized kernel with shared memory
        """
        self.use_optimized = use_optimized

        # Compile 3D kernels
        if use_optimized:
            self.kernel = cupy.RawKernel(
                DIA_MATVEC_3D_OPTIMIZED_FP16_KERNEL,
                "dia_matvec_3d_optimized_fp16_kernel",
            )
        else:
            self.kernel = cupy.RawKernel(DIA_MATVEC_3D_FP16_KERNEL, "dia_matvec_3d_fp16_kernel")

    def __call__(
        self,
        dia_data_3d: cupy.ndarray,  # Shape: (channels, num_bands, n)
        dia_offsets: cupy.ndarray,  # Shape: (num_bands,)
        x: cupy.ndarray,  # Shape: (channels, n)
    ) -> cupy.ndarray:
        """Perform 3D DIA matrix-vector multiplication using custom kernel with FP16 output.

        Args:
            dia_data_3d: 3D DIA matrix data (channels, num_bands, n)
            dia_offsets: Band offsets shared across channels (num_bands,)
            x: Input vectors for each channel (channels, n)

        Returns:
            Result vectors y = A @ x for each channel (channels, n) in FP16
        """
        channels, num_bands, n = dia_data_3d.shape
        assert dia_offsets.shape == (num_bands,), f"Offsets shape mismatch: {dia_offsets.shape}"
        assert x.shape == (channels, n), f"Input shape mismatch: {x.shape}"

        # Prepare GPU arrays with proper types
        data_gpu = cupy.asarray(dia_data_3d, dtype=cupy.float32)  # Shape: (channels, num_bands, n)
        offsets_gpu = cupy.asarray(dia_offsets, dtype=cupy.int32)  # Shape: (num_bands,)
        x_gpu = cupy.asarray(x, dtype=cupy.float32)  # Shape: (channels, n)
        y_gpu = cupy.zeros((channels, n), dtype=cupy.float16)  # Shape: (channels, n) - FP16 output

        # Launch configuration for 2D grid
        block_size = 256
        grid_x = (n + block_size - 1) // block_size  # LED dimension
        grid_y = channels  # Channel dimension

        if self.use_optimized:
            # Calculate shared memory size for optimized kernel
            max_band_offset = 1000  # Conservative estimate
            shared_size = block_size + 2 * max_band_offset
            shared_mem_bytes = shared_size * 4  # 4 bytes per float32

            # Launch optimized kernel with shared memory
            self.kernel(
                (grid_x, grid_y),
                (block_size,),
                (
                    data_gpu,  # const float* data (channels, num_bands, n)
                    offsets_gpu,  # const int* offsets (num_bands,)
                    x_gpu,  # const float* x (channels, n)
                    y_gpu,  # __half* y (channels, n) - FP16 output
                    n,  # int n (LED count)
                    num_bands,  # int num_bands
                    channels,  # int channels
                ),
                shared_mem=shared_mem_bytes,
            )
        else:
            # Launch basic kernel without shared memory
            self.kernel(
                (grid_x, grid_y),
                (block_size,),
                (
                    data_gpu,  # const float* data (channels, num_bands, n)
                    offsets_gpu,  # const int* offsets (num_bands,)
                    x_gpu,  # const float* x (channels, n)
                    y_gpu,  # __half* y (channels, n) - FP16 output
                    n,  # int n (LED count)
                    num_bands,  # int num_bands
                    channels,  # int channels
                ),
            )

        return y_gpu
