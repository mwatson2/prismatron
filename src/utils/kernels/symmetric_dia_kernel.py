"""
Symmetric CUDA kernel for optimized DIA (diagonal) matrix-vector multiplication.

This module implements high-performance CUDA kernels for symmetric A^T A matrices
using symmetric storage that only stores the main diagonal and upper diagonals.
"""

import cupy
import numpy as np

# Symmetric 3D DIA kernel for multi-channel matrix-vector multiplication
SYMMETRIC_DIA_MATVEC_3D_KERNEL = r"""
extern "C" __global__
void symmetric_dia_matvec_3d_kernel(
    const float* __restrict__ data,      // 3D DIA matrix data: shape (channels, num_upper_bands, n) - only main + upper diagonals
    const int* __restrict__ offsets,     // Upper band offsets: shape (num_upper_bands,) - only non-negative offsets
    const float* __restrict__ x,         // Input vectors: shape (channels, n) - channel-major layout
    float* __restrict__ y,               // Output vectors: shape (channels, n) - channel-major layout
    const int n,                         // Matrix dimension (number of LEDs)
    const int num_upper_bands,           // Number of upper diagonal bands (including main diagonal)
    const int channels                   // Number of channels (3 for RGB)
) {
    // 2D grid: blockIdx.x covers LED indices, blockIdx.y covers channels
    const int led_idx = blockIdx.x * blockDim.x + threadIdx.x;  // LED index [0, n)
    const int channel = blockIdx.y;                             // Channel index [0, channels)

    // Bounds checking
    if (led_idx >= n || channel >= channels) return;

    float sum = 0.0f;

    // Calculate base pointers for this channel
    // data: (channels, num_upper_bands, n) -> channel * (num_upper_bands * n) + band * n + led
    // x: (channels, n) -> channel * n + led
    const float* data_channel = data + channel * num_upper_bands * n;  // Points to start of this channel's data
    const float* x_channel = x + channel * n;                          // Points to start of this channel's input vector

    // Iterate through all upper diagonal bands (including main diagonal)
    for (int band = 0; band < num_upper_bands; band++) {
        const int offset = offsets[band];  // This offset is >= 0 (main and upper diagonals only)

        // Upper diagonal contribution: A[led_idx, led_idx + offset] * x[led_idx + offset]
        const int j_upper = led_idx + offset;
        if (j_upper >= 0 && j_upper < n) {
            // DIA format: A[led_idx, j_upper] is stored at data_channel[band * n + j_upper] (indexed by column j_upper)
            const float matrix_val_upper = data_channel[band * n + j_upper];
            if (matrix_val_upper != 0.0f) {
                sum += matrix_val_upper * x_channel[j_upper];
            }
        }

        // Symmetric (lower diagonal) contribution: A[led_idx, led_idx - offset] * x[led_idx - offset]
        // Since A is symmetric: A[led_idx, led_idx - offset] = A[led_idx - offset, led_idx]
        if (offset > 0) {  // Skip main diagonal to avoid double counting
            const int j_lower = led_idx - offset;
            if (j_lower >= 0 && j_lower < n) {
                // A[led_idx, j_lower] = A[j_lower, led_idx] (symmetry)
                // A[j_lower, led_idx] is stored at data_channel[band * n + led_idx] (because for A[j_lower, led_idx], led_idx = j_lower + offset)
                const float matrix_val_lower = data_channel[band * n + led_idx];
                if (matrix_val_lower != 0.0f) {
                    sum += matrix_val_lower * x_channel[j_lower];
                }
            }
        }
    }

    // Write result: y[channel, led_idx] = sum
    // y: (channels, n) -> channel * n + led_idx
    y[channel * n + led_idx] = sum;
}
"""

# Optimized symmetric 3D DIA kernel with shared memory
SYMMETRIC_DIA_MATVEC_3D_OPTIMIZED_KERNEL = r"""
extern "C" __global__
void symmetric_dia_matvec_3d_optimized_kernel(
    const float* __restrict__ data,      // 3D DIA matrix data: shape (channels, num_upper_bands, n) - only main + upper diagonals
    const int* __restrict__ offsets,     // Upper band offsets: shape (num_upper_bands,) - only non-negative offsets
    const float* __restrict__ x,         // Input vectors: shape (channels, n) - channel-major layout
    float* __restrict__ y,               // Output vectors: shape (channels, n) - channel-major layout
    const int n,                         // Matrix dimension (number of LEDs)
    const int num_upper_bands,           // Number of upper diagonal bands (including main diagonal)
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
    const float* data_channel = data + channel * num_upper_bands * n;

    // Iterate through all upper diagonal bands (including main diagonal)
    for (int band = 0; band < num_upper_bands; band++) {
        const int offset = offsets[band];  // This offset is >= 0 (main and upper diagonals only)

        // Upper diagonal contribution: A[led_idx, led_idx + offset] * x[led_idx + offset]
        const int j_upper = led_idx + offset;
        if (j_upper >= 0 && j_upper < n) {
            // DIA format: A[led_idx, j_upper] is stored at data_channel[band * n + j_upper] (indexed by column j_upper)
            const float matrix_val_upper = data_channel[band * n + j_upper];
            if (matrix_val_upper != 0.0f) {
                // Use shared memory for vector access
                const int shared_idx_upper = j_upper - shared_start;
                if (shared_idx_upper >= 0 && shared_idx_upper < shared_size) {
                    sum += matrix_val_upper * shared_x[shared_idx_upper];
                } else {
                    // Fallback to global memory (should be rare)
                    sum += matrix_val_upper * x_channel[j_upper];
                }
            }
        }

        // Symmetric (lower diagonal) contribution: A[led_idx, led_idx - offset] * x[led_idx - offset]
        // Since A is symmetric: A[led_idx, led_idx - offset] = A[led_idx - offset, led_idx]
        if (offset > 0) {  // Skip main diagonal to avoid double counting
            const int j_lower = led_idx - offset;
            if (j_lower >= 0 && j_lower < n) {
                // A[led_idx, j_lower] = A[j_lower, led_idx] (symmetry)
                // A[j_lower, led_idx] is stored at data_channel[band * n + led_idx] (because for A[j_lower, led_idx], led_idx = j_lower + offset)
                const float matrix_val_lower = data_channel[band * n + led_idx];
                if (matrix_val_lower != 0.0f) {
                    // Use shared memory for vector access
                    const int shared_idx_lower = j_lower - shared_start;
                    if (shared_idx_lower >= 0 && shared_idx_lower < shared_size) {
                        sum += matrix_val_lower * shared_x[shared_idx_lower];
                    } else {
                        // Fallback to global memory (should be rare)
                        sum += matrix_val_lower * x_channel[j_lower];
                    }
                }
            }
        }
    }

    // Write result: y[channel, led_idx] = sum
    y[channel * n + led_idx] = sum;
}
"""


class SymmetricCustomDIA3DMatVec:
    """Custom CUDA implementation for symmetric 3D DIA matrix-vector multiplication.

    Handles multi-channel (RGB) symmetric DIA matrices efficiently using 2D GPU grid
    and symmetric storage that only stores main diagonal + upper diagonals.
    """

    def __init__(self, use_optimized: bool = True):
        """Initialize the symmetric 3D DIA kernel.

        Args:
            use_optimized: Use optimized kernel with shared memory
        """
        self.use_optimized = use_optimized

        # Compile symmetric 3D kernels
        if use_optimized:
            self.kernel = cupy.RawKernel(
                SYMMETRIC_DIA_MATVEC_3D_OPTIMIZED_KERNEL, "symmetric_dia_matvec_3d_optimized_kernel"
            )
        else:
            self.kernel = cupy.RawKernel(SYMMETRIC_DIA_MATVEC_3D_KERNEL, "symmetric_dia_matvec_3d_kernel")

    def __call__(
        self,
        dia_data_3d: cupy.ndarray,  # Shape: (channels, num_upper_bands, n) - only main + upper diagonals
        dia_offsets_upper: cupy.ndarray,  # Shape: (num_upper_bands,) - only non-negative offsets
        x: cupy.ndarray,  # Shape: (channels, n)
    ) -> cupy.ndarray:
        """Perform symmetric 3D DIA matrix-vector multiplication using custom kernel.

        Args:
            dia_data_3d: 3D DIA matrix data for upper diagonals only (channels, num_upper_bands, n)
            dia_offsets_upper: Upper band offsets (non-negative only) (num_upper_bands,)
            x: Input vectors for each channel (channels, n)

        Returns:
            Result vectors y = A @ x for each channel (channels, n)
        """
        channels, num_upper_bands, n = dia_data_3d.shape
        assert dia_offsets_upper.shape == (num_upper_bands,), f"Offsets shape mismatch: {dia_offsets_upper.shape}"
        assert x.shape == (channels, n), f"Input shape mismatch: {x.shape}"

        # Validate that all offsets are non-negative (upper diagonals only)
        assert cupy.all(dia_offsets_upper >= 0), "Symmetric kernel expects only non-negative offsets (upper diagonals)"

        # Prepare GPU arrays with proper types
        data_gpu = cupy.asarray(dia_data_3d, dtype=cupy.float32)  # Shape: (channels, num_upper_bands, n)
        offsets_gpu = cupy.asarray(dia_offsets_upper, dtype=cupy.int32)  # Shape: (num_upper_bands,)
        x_gpu = cupy.asarray(x, dtype=cupy.float32)  # Shape: (channels, n)
        y_gpu = cupy.zeros((channels, n), dtype=cupy.float32)  # Shape: (channels, n)

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
                    data_gpu,  # const float* data (channels, num_upper_bands, n)
                    offsets_gpu,  # const int* offsets (num_upper_bands,)
                    x_gpu,  # const float* x (channels, n)
                    y_gpu,  # float* y (channels, n)
                    n,  # int n (LED count)
                    num_upper_bands,  # int num_upper_bands
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
                    data_gpu,  # const float* data (channels, num_upper_bands, n)
                    offsets_gpu,  # const int* offsets (num_upper_bands,)
                    x_gpu,  # const float* x (channels, n)
                    y_gpu,  # float* y (channels, n)
                    n,  # int n (LED count)
                    num_upper_bands,  # int num_upper_bands
                    channels,  # int channels
                ),
            )

        return y_gpu
