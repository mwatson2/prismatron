#!/usr/bin/env python3
"""
Custom CUDA kernel for optimized DIA (diagonal) matrix-vector multiplication.

This module implements a high-performance CUDA kernel specifically designed for
banded matrices in DIA format, optimizing for the A^T A matrix structure found
in the LED diffusion optimization problem.

Key optimizations:
1. Coalesced memory access patterns for diagonal bands
2. Shared memory for vector reuse across threads
3. Optimized thread mapping for band structure
4. Reduced memory bandwidth vs dense matrix operations
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
            self.kernel = cupy.RawKernel(
                DIA_MATVEC_OPTIMIZED_KERNEL, "dia_matvec_optimized_kernel"
            )
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
        data = cupy.asarray(
            dia_matrix.data, dtype=cupy.float32
        )  # Shape: (num_bands, n)
        offsets = cupy.asarray(
            dia_matrix.offsets, dtype=cupy.int32
        )  # Shape: (num_bands,)
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


def create_test_dia_matrix(
    n: int = 3000, num_bands: int = 101
) -> Tuple[cusp.dia_matrix, np.ndarray]:
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


def benchmark_dia_kernels(
    dia_matrix, dense_matrix, num_trials: int = 20, num_warmup: int = 5
):
    """Benchmark different DIA kernel implementations."""

    n = dia_matrix.shape[0]
    print(
        f"Benchmarking DIA kernels for {n}x{n} matrix with {len(dia_matrix.offsets)} bands"
    )
    print(f"Matrix nnz: {dia_matrix.nnz}, sparsity: {dia_matrix.nnz/(n*n)*100:.2f}%")

    # Create test vectors
    test_vectors = [
        cupy.random.randn(n, dtype=cupy.float32) for _ in range(num_trials + num_warmup)
    ]

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

    simple_dense = cupy.array(
        [[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]], dtype=cupy.float32
    )

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
    print(
        f"    Custom error:     {cupy.max(cupy.abs(y_basic_simple - y_ref_simple)):.6f}"
    )
    print(
        f"    CuPy error:       {cupy.max(cupy.abs(y_cupy_simple - y_ref_simple)):.6f}"
    )

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

                    print(
                        f"        Band {band} (offset {offset}): A[{i},{j}] = data[{band},{j}] = {matrix_val:.1f}"
                    )
                    print(
                        f"          Contribution: {matrix_val:.1f} * x[{j}] = {contribution:.1f}"
                    )

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

    all_correct = (
        error_basic < tolerance and error_opt < tolerance and error_cupy < tolerance
    )
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
    print(
        f"Sparsity: {dia_matrix.nnz / (dia_matrix.shape[0] * dia_matrix.shape[1]) * 100:.2f}%"
    )
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

        print(
            f"{name:19s}  | {mean_ms:6.2f}±{std_ms:5.2f} | {gflops:8.2f}  | {speedup:6.2f}x"
        )

    print("\n=== Custom Kernel Development Complete ===")


if __name__ == "__main__":
    main()
