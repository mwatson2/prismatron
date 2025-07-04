"""
Custom CUDA kernel for optimized DIA (diagonal) matrix-vector multiplication.

This module implements a high-performance CUDA kernel specifically designed for
banded matrices in DIA format, optimizing for the A^T A matrix structure found
in the LED diffusion optimization problem.
"""

import time
from typing import Optional, Tuple

import cupy
import cupyx.scipy.sparse as cusp
import numpy as np

# CUDA kernel for DIA matrix-vector multiplication
DIA_MATVEC_KERNEL = r"""
extern "C" __global__
void dia_matvec_kernel(
    const float* __restrict__ data,      // DIA matrix data: shape (num_bands, n)
    const int* __restrict__ offsets,     // Band offsets: shape (num_bands,)
    const float* __restrict__ x,         // Input vector: shape (n,)
    float* __restrict__ y,               // Output vector: shape (n,)
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

    y[i] = sum;
}
"""

# Optimized kernel with shared memory for better memory bandwidth
DIA_MATVEC_OPTIMIZED_KERNEL = r"""
extern "C" __global__
void dia_matvec_optimized_kernel(
    const float* __restrict__ data,      // DIA matrix data: shape (num_bands, n)
    const int* __restrict__ offsets,     // Band offsets: shape (num_bands,)
    const float* __restrict__ x,         // Input vector: shape (n,)
    float* __restrict__ y,               // Output vector: shape (n,)
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

    y[i] = sum;
}
"""

# 3D DIA kernel for multi-channel matrix-vector multiplication
DIA_MATVEC_3D_KERNEL = r"""
extern "C" __global__
void dia_matvec_3d_kernel(
    const float* __restrict__ data,      // 3D DIA matrix data: shape (channels, num_bands, n)
    const int* __restrict__ offsets,     // Band offsets: shape (num_bands,) - shared across channels
    const float* __restrict__ x,         // Input vectors: shape (channels, n) - channel-major layout
    float* __restrict__ y,               // Output vectors: shape (channels, n) - channel-major layout
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

    // Write result: y[channel, led_idx] = sum
    // y: (channels, n) -> channel * n + led_idx
    y[channel * n + led_idx] = sum;
}
"""

# Optimized 3D DIA kernel with shared memory
DIA_MATVEC_3D_OPTIMIZED_KERNEL = r"""
extern "C" __global__
void dia_matvec_3d_optimized_kernel(
    const float* __restrict__ data,      // 3D DIA matrix data: shape (channels, num_bands, n)
    const int* __restrict__ offsets,     // Band offsets: shape (num_bands,) - shared across channels
    const float* __restrict__ x,         // Input vectors: shape (channels, n) - channel-major layout
    float* __restrict__ y,               // Output vectors: shape (channels, n) - channel-major layout
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

    // Write result: y[channel, led_idx] = sum
    y[channel * n + led_idx] = sum;
}
"""


class CustomDIAMatVec:
    """Custom CUDA implementation for DIA matrix-vector multiplication."""

    def __init__(self, use_optimized: bool = True):
        """Initialize the custom DIA kernel.

        Args:
            use_optimized: Use optimized kernel with shared memory
        """
        self.use_optimized = use_optimized

        # Compile kernels
        if use_optimized:
            self.kernel = cupy.RawKernel(DIA_MATVEC_OPTIMIZED_KERNEL, "dia_matvec_optimized_kernel")
        else:
            self.kernel = cupy.RawKernel(DIA_MATVEC_KERNEL, "dia_matvec_kernel")

    def __call__(self, dia_matrix, x: cupy.ndarray) -> cupy.ndarray:
        """Perform DIA matrix-vector multiplication using custom kernel.

        Args:
            dia_matrix: DIA format sparse matrix (CuPy or SciPy)
            x: Input vector (CuPy array)

        Returns:
            Result vector y = A @ x
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
        y_output = cupy.zeros(n, dtype=cupy.float32)  # Shape: (n,)

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


class CustomDIA3DMatVec:
    """Custom CUDA implementation for 3D DIA matrix-vector multiplication.

    Handles multi-channel (RGB) DIA matrices efficiently using 2D GPU grid:
    - blockIdx.x: LED indices
    - blockIdx.y: Channel indices
    """

    def __init__(self, use_optimized: bool = True):
        """Initialize the custom 3D DIA kernel.

        Args:
            use_optimized: Use optimized kernel with shared memory
        """
        self.use_optimized = use_optimized

        # Compile 3D kernels
        if use_optimized:
            self.kernel = cupy.RawKernel(DIA_MATVEC_3D_OPTIMIZED_KERNEL, "dia_matvec_3d_optimized_kernel")
        else:
            self.kernel = cupy.RawKernel(DIA_MATVEC_3D_KERNEL, "dia_matvec_3d_kernel")

    def __call__(
        self,
        dia_data_3d: cupy.ndarray,  # Shape: (channels, num_bands, n)
        dia_offsets: cupy.ndarray,  # Shape: (num_bands,)
        x: cupy.ndarray,  # Shape: (channels, n)
    ) -> cupy.ndarray:
        """Perform 3D DIA matrix-vector multiplication using custom kernel.

        Args:
            dia_data_3d: 3D DIA matrix data (channels, num_bands, n)
            dia_offsets: Band offsets shared across channels (num_bands,)
            x: Input vectors for each channel (channels, n)

        Returns:
            Result vectors y = A @ x for each channel (channels, n)
        """
        channels, num_bands, n = dia_data_3d.shape
        assert dia_offsets.shape == (num_bands,), f"Offsets shape mismatch: {dia_offsets.shape}"
        assert x.shape == (channels, n), f"Input shape mismatch: {x.shape}"

        # Prepare GPU arrays with proper types
        data_gpu = cupy.asarray(dia_data_3d, dtype=cupy.float32)  # Shape: (channels, num_bands, n)
        offsets_gpu = cupy.asarray(dia_offsets, dtype=cupy.int32)  # Shape: (num_bands,)
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
                    data_gpu,  # const float* data (channels, num_bands, n)
                    offsets_gpu,  # const int* offsets (num_bands,)
                    x_gpu,  # const float* x (channels, n)
                    y_gpu,  # float* y (channels, n)
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
                    y_gpu,  # float* y (channels, n)
                    n,  # int n (LED count)
                    num_bands,  # int num_bands
                    channels,  # int channels
                ),
            )

        return y_gpu


def create_test_dia_matrix(n: int = 3000, num_bands: int = 101) -> Tuple[cusp.dia_matrix, np.ndarray]:
    """Create a test DIA matrix with realistic band structure.

    Args:
        n: Matrix dimension
        num_bands: Number of diagonal bands

    Returns:
        Tuple of (DIA matrix, dense equivalent for verification)
    """
    # Create band offsets: symmetric around main diagonal
    max_offset = (num_bands - 1) // 2
    offsets = list(range(-max_offset, max_offset + 1))

    # Create data array: shape (num_bands, n)
    data = np.zeros((num_bands, n), dtype=np.float32)

    np.random.seed(42)

    for i, offset in enumerate(offsets):
        # Band density decreases with distance from main diagonal
        density = max(0.6, 1.0 - abs(offset) / max_offset * 0.4)

        # Fill band with random values
        for j in range(n):
            if np.random.rand() < density:
                # Higher values near main diagonal
                intensity = np.exp(-abs(offset) / 10) * (np.random.rand() * 500 + 100)
                data[i, j] = intensity

    # Create DIA matrix
    dia_matrix = cusp.dia_matrix((data, offsets), shape=(n, n))

    # Create dense equivalent for verification
    dense_matrix = dia_matrix.toarray()

    return dia_matrix, dense_matrix


def benchmark_dia_kernels(dia_matrix, dense_matrix, num_trials: int = 20, num_warmup: int = 5):
    """Benchmark different DIA kernel implementations."""

    n = dia_matrix.shape[0]
    print(f"Benchmarking DIA kernels for {n}x{n} matrix with {len(dia_matrix.offsets)} bands")
    print(f"Matrix nnz: {dia_matrix.nnz}, sparsity: {dia_matrix.nnz / (n * n) * 100:.2f}%")

    # Create test vectors
    test_vectors = [cupy.random.randn(n, dtype=cupy.float32) for _ in range(num_trials + num_warmup)]

    results = {}

    # Custom kernel (basic)
    print("  Benchmarking Custom DIA kernel (basic)...")
    custom_basic = CustomDIAMatVec(use_optimized=False)

    # Warm-up
    for i in range(num_warmup):
        _ = custom_basic(dia_matrix, test_vectors[i])
    cupy.cuda.Stream.null.synchronize()

    # Timing
    times = []
    for i in range(num_warmup, num_warmup + num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        y = custom_basic(dia_matrix, test_vectors[i])
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times.append(elapsed_ms / 1000.0)

    results["custom_basic"] = {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "flops": 2 * dia_matrix.nnz,
    }

    # Custom kernel (optimized)
    print("  Benchmarking Custom DIA kernel (optimized)...")
    custom_opt = CustomDIAMatVec(use_optimized=True)

    # Warm-up
    for i in range(num_warmup):
        _ = custom_opt(dia_matrix, test_vectors[i])
    cupy.cuda.Stream.null.synchronize()

    # Timing
    times = []
    for i in range(num_warmup, num_warmup + num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        y = custom_opt(dia_matrix, test_vectors[i])
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times.append(elapsed_ms / 1000.0)

    results["custom_optimized"] = {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "flops": 2 * dia_matrix.nnz,
    }

    # CuPy built-in DIA
    print("  Benchmarking CuPy DIA @ operator...")

    # Warm-up
    for i in range(num_warmup):
        _ = dia_matrix @ test_vectors[i]
    cupy.cuda.Stream.null.synchronize()

    # Timing
    times = []
    for i in range(num_warmup, num_warmup + num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        y = dia_matrix @ test_vectors[i]
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times.append(elapsed_ms / 1000.0)

    results["cupy_dia"] = {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "flops": 2 * dia_matrix.nnz,
    }

    # Dense GPU baseline
    print("  Benchmarking Dense GPU baseline...")
    dense_gpu = cupy.asarray(dense_matrix, dtype=cupy.float32)

    # Warm-up
    for i in range(num_warmup):
        _ = dense_gpu @ test_vectors[i]
    cupy.cuda.Stream.null.synchronize()

    # Timing
    times = []
    for i in range(num_warmup, num_warmup + num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        y = dense_gpu @ test_vectors[i]
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times.append(elapsed_ms / 1000.0)

    results["dense_gpu"] = {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "flops": 2 * n * n,
    }

    return results


def verify_kernel_correctness(dia_matrix, dense_matrix, tolerance: float = 1e-2):
    """Verify that custom kernels produce correct results."""
    print("Verifying kernel correctness...")

    n = dia_matrix.shape[0]

    # Test with simple case first for detailed debugging
    print("  Testing with simple 3x3 matrix first...")
    import scipy.sparse as sp

    simple_dense = cupy.array([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]], dtype=cupy.float32)

    simple_dia_scipy = sp.dia_matrix(cupy.asnumpy(simple_dense))
    simple_dia_cupy = cusp.dia_matrix(simple_dia_scipy)

    x_simple = cupy.array([1.0, 1.0, 1.0], dtype=cupy.float32)
    y_ref_simple = simple_dense @ x_simple

    print(f"    Simple DIA offsets: {simple_dia_cupy.offsets}")
    print(f"    Simple DIA data shape: {simple_dia_cupy.data.shape}")
    print(f"    Simple DIA data:\n{simple_dia_cupy.data}")

    custom_basic = CustomDIAMatVec(use_optimized=False)
    y_basic_simple = custom_basic(simple_dia_cupy, x_simple)
    y_cupy_simple = simple_dia_cupy @ x_simple

    print(f"    Reference result: {y_ref_simple}")
    print(f"    Custom kernel:    {y_basic_simple}")
    print(f"    CuPy DIA:         {y_cupy_simple}")
    print(f"    Custom error:     {cupy.max(cupy.abs(y_basic_simple - y_ref_simple)):.6f}")
    print(f"    CuPy error:       {cupy.max(cupy.abs(y_cupy_simple - y_ref_simple)):.6f}")

    # If simple case fails, debug step by step
    if cupy.max(cupy.abs(y_basic_simple - y_ref_simple)) > tolerance:
        print("    DEBUGGING: Custom kernel fails on simple case!")
        print("    Manual verification:")

        # Manual calculation
        n_simple = 3
        offsets = cupy.asnumpy(simple_dia_cupy.offsets)
        data = cupy.asnumpy(simple_dia_cupy.data)
        x_cpu = cupy.asnumpy(x_simple)

        for i in range(n_simple):
            manual_sum = 0.0
            print(f"      y[{i}] calculation:")

            for band in range(len(offsets)):
                offset = offsets[band]
                j = i + offset

                if 0 <= j < n_simple:
                    # CORRECTED: A[i,j] is stored at data[band, j] for DIA format (indexed by column)
                    matrix_val = data[band, j]
                    contribution = matrix_val * x_cpu[j]
                    manual_sum += contribution

                    print(f"        Band {band} (offset {offset}): A[{i},{j}] = data[{band},{j}] = {matrix_val:.1f}")
                    print(f"          Contribution: {matrix_val:.1f} * x[{j}] = {contribution:.1f}")

            print(f"      y[{i}] = {manual_sum:.1f} (expected {y_ref_simple[i]:.1f})")

        return False

    # Test with larger matrix if simple case passes
    print("  Simple case passed, testing with larger matrix...")
    x = cupy.random.randn(n, dtype=cupy.float32)

    # Reference result (dense)
    y_ref = cupy.asarray(dense_matrix, dtype=cupy.float32) @ x

    # Custom basic kernel
    y_basic = custom_basic(dia_matrix, x)

    # Custom optimized kernel
    custom_opt = CustomDIAMatVec(use_optimized=True)
    y_opt = custom_opt(dia_matrix, x)

    # CuPy DIA
    y_cupy = dia_matrix @ x

    # Check errors
    error_basic = cupy.max(cupy.abs(y_basic - y_ref))
    error_opt = cupy.max(cupy.abs(y_opt - y_ref))
    error_cupy = cupy.max(cupy.abs(y_cupy - y_ref))

    print(f"  Max error (custom basic): {float(error_basic):.2e}")
    print(f"  Max error (custom optimized): {float(error_opt):.2e}")
    print(f"  Max error (CuPy DIA): {float(error_cupy):.2e}")

    all_correct = error_basic < tolerance and error_opt < tolerance and error_cupy < tolerance
    print(f"  Basic kernel correct: {error_basic < tolerance}")
    print(f"  Optimized kernel correct: {error_opt < tolerance}")
    print(f"  CuPy DIA correct: {error_cupy < tolerance}")

    return all_correct


def main():
    """Main function to test and benchmark custom DIA kernels."""
    print("=== Custom DIA Kernel Development ===\n")

    # Create test matrix with realistic band structure
    print("Creating test DIA matrix...")
    dia_matrix, dense_matrix = create_test_dia_matrix(n=3000, num_bands=101)

    print(f"DIA matrix shape: {dia_matrix.shape}")
    print(f"Number of bands: {len(dia_matrix.offsets)}")
    print(f"Band offsets: [{dia_matrix.offsets[0]}, ..., {dia_matrix.offsets[-1]}]")
    print(f"Matrix nnz: {dia_matrix.nnz}")
    print(f"Sparsity: {dia_matrix.nnz / (dia_matrix.shape[0] * dia_matrix.shape[1]) * 100:.2f}%")
    print(f"DIA memory: {dia_matrix.data.nbytes / 1024**2:.1f} MB")
    print(f"Dense memory: {dense_matrix.nbytes / 1024**2:.1f} MB")
    print(f"Memory reduction: {dense_matrix.nbytes / dia_matrix.data.nbytes:.1f}x\n")

    # Verify correctness
    if not verify_kernel_correctness(dia_matrix, dense_matrix):
        print("ERROR: Kernel verification failed!")
        return

    print()

    # Benchmark performance
    results = benchmark_dia_kernels(dia_matrix, dense_matrix)

    # Print results
    print("\nPerformance Results:")
    print("Kernel               | Time (ms)      | GFLOPS    | Speedup")
    print("---------------------|----------------|-----------|--------")

    baseline_time = results["dense_gpu"]["mean_time"]

    for name, data in results.items():
        mean_ms = data["mean_time"] * 1000
        std_ms = data["std_time"] * 1000
        gflops = data["flops"] / data["mean_time"] / 1e9
        speedup = baseline_time / data["mean_time"]

        print(f"{name:19s}  | {mean_ms:6.2f}±{std_ms:5.2f} | {gflops:8.2f}  | {speedup:6.2f}x")

    print("\n=== Custom Kernel Development Complete ===")


if __name__ == "__main__":
    main()
