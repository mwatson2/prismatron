#!/usr/bin/env python3
"""
Standalone tests for 3D DIA CUDA kernel implementation.

Tests the CustomDIA3DMatVec class against reference implementations to ensure
correctness and measure performance improvements.
"""

# Add src to path for imports
import sys
import time
from pathlib import Path
from typing import Tuple

import cupy
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.custom_dia_kernel import CustomDIA3DMatVec


def create_test_3d_dia_matrix(
    n: int = 1000, num_bands: int = 101, channels: int = 3, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a test 3D DIA matrix with realistic band structure.

    Args:
        n: Matrix dimension (LED count)
        num_bands: Number of diagonal bands
        channels: Number of channels (RGB)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (dia_data_3d, dia_offsets, dense_matrices_3d)
    """
    np.random.seed(seed)

    # Create band offsets: symmetric around main diagonal
    max_offset = (num_bands - 1) // 2
    offsets = list(range(-max_offset, max_offset + 1))

    # Initialize 3D DIA data: (channels, num_bands, n)
    dia_data_3d = np.zeros((channels, num_bands, n), dtype=np.float32)

    # Create dense matrices for verification: (channels, n, n)
    dense_matrices_3d = np.zeros((channels, n, n), dtype=np.float32)

    for channel in range(channels):
        for i, offset in enumerate(offsets):
            # Band density decreases with distance from main diagonal
            density = max(0.6, 1.0 - abs(offset) / max_offset * 0.4)

            # Fill band with random values
            for j in range(n):
                if np.random.rand() < density:
                    # Higher values near main diagonal, different per channel
                    base_intensity = np.exp(-abs(offset) / 10) * (
                        np.random.rand() * 500 + 100
                    )
                    channel_factor = (
                        channel + 1
                    ) * 0.8  # Different intensity per channel
                    intensity = base_intensity * channel_factor

                    dia_data_3d[channel, i, j] = intensity

                    # Also fill dense matrix for verification
                    row_idx = (
                        j - offset
                    )  # In DIA: A[row,col] stored at data[band, col] where col = row + offset
                    if 0 <= row_idx < n:
                        dense_matrices_3d[channel, row_idx, j] = intensity

    return dia_data_3d, np.array(offsets, dtype=np.int32), dense_matrices_3d


def test_3d_dia_kernel_correctness():
    """Test 3D DIA kernel correctness against dense matrix reference."""
    print("=== Testing 3D DIA Kernel Correctness ===")

    # Create test matrices
    n = 500  # Smaller for faster testing
    num_bands = 51
    channels = 3

    dia_data_3d, dia_offsets, dense_matrices_3d = create_test_3d_dia_matrix(
        n, num_bands, channels
    )

    # Create test input vectors
    np.random.seed(123)
    x_test = np.random.randn(channels, n).astype(np.float32)

    print(f"Matrix size: {n}x{n} with {num_bands} bands, {channels} channels")
    print(f"3D DIA data shape: {dia_data_3d.shape}")
    print(f"Input vectors shape: {x_test.shape}")

    # Reference result using dense matrices
    print("Computing reference result with dense matrices...")
    y_reference = np.zeros((channels, n), dtype=np.float32)
    for channel in range(channels):
        y_reference[channel] = dense_matrices_3d[channel] @ x_test[channel]

    # Test basic 3D kernel
    print("Testing basic 3D DIA kernel...")
    kernel_basic = CustomDIA3DMatVec(use_optimized=False)

    # Convert to GPU
    dia_data_gpu = cupy.asarray(dia_data_3d)
    dia_offsets_gpu = cupy.asarray(dia_offsets)
    x_gpu = cupy.asarray(x_test)

    y_basic_gpu = kernel_basic(dia_data_gpu, dia_offsets_gpu, x_gpu)
    y_basic = cupy.asnumpy(y_basic_gpu)

    # Test optimized 3D kernel
    print("Testing optimized 3D DIA kernel...")
    kernel_opt = CustomDIA3DMatVec(use_optimized=True)
    y_opt_gpu = kernel_opt(dia_data_gpu, dia_offsets_gpu, x_gpu)
    y_opt = cupy.asnumpy(y_opt_gpu)

    # Check correctness
    def check_result(y_result, name):
        error = np.abs(y_result - y_reference)
        max_error = np.max(error)
        mean_error = np.mean(error)
        rel_error = max_error / (np.max(np.abs(y_reference)) + 1e-10)

        print(f"  {name}:")
        print(f"    Max absolute error: {max_error:.6e}")
        print(f"    Mean absolute error: {mean_error:.6e}")
        print(f"    Relative error: {rel_error:.6e}")

        success = rel_error < 1e-5
        print(f"    Status: {'✓ PASS' if success else '✗ FAIL'}")
        return success

    success_basic = check_result(y_basic, "Basic 3D kernel")
    success_opt = check_result(y_opt, "Optimized 3D kernel")

    if success_basic and success_opt:
        print("✓ All 3D DIA kernel tests PASSED")
        return True
    else:
        print("✗ Some 3D DIA kernel tests FAILED")
        return False


def benchmark_3d_dia_kernel():
    """Benchmark 3D DIA kernel performance."""
    print("\n=== Benchmarking 3D DIA Kernel Performance ===")

    # Test parameters
    n = 1000  # LED count (realistic size)
    num_bands = 1985  # Realistic band count from performance comparison
    channels = 3
    num_trials = 10
    num_warmup = 3

    dia_data_3d, dia_offsets, dense_matrices_3d = create_test_3d_dia_matrix(
        n, num_bands, channels
    )

    print(f"Matrix size: {n}x{n} with {num_bands} bands, {channels} channels")
    print(f"Total matrix elements: {channels * n * n:,}")
    print(f"3D DIA nnz: {np.count_nonzero(dia_data_3d):,}")

    # Create test vectors
    np.random.seed(456)
    test_vectors = [
        np.random.randn(channels, n).astype(np.float32)
        for _ in range(num_trials + num_warmup)
    ]

    # Convert to GPU
    dia_data_gpu = cupy.asarray(dia_data_3d)
    dia_offsets_gpu = cupy.asarray(dia_offsets)
    test_vectors_gpu = [cupy.asarray(v) for v in test_vectors]

    results = {}

    # Test dense matrix baseline (reference)
    print("  Benchmarking dense matrix reference...")
    dense_gpu = [cupy.asarray(dense_matrices_3d[c]) for c in range(channels)]

    # Warmup
    for i in range(num_warmup):
        x_gpu = test_vectors_gpu[i]
        for c in range(channels):
            _ = dense_gpu[c] @ x_gpu[c]
    cupy.cuda.Stream.null.synchronize()

    # Timing
    times = []
    for i in range(num_warmup, num_warmup + num_trials):
        x_gpu = test_vectors_gpu[i]

        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        for c in range(channels):
            _ = dense_gpu[c] @ x_gpu[c]
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times.append(elapsed_ms / 1000.0)

    results["dense_reference"] = {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "flops": 2 * np.count_nonzero(dia_data_3d),  # Approximate
    }

    # Test basic 3D DIA kernel
    print("  Benchmarking basic 3D DIA kernel...")
    kernel_basic = CustomDIA3DMatVec(use_optimized=False)

    # Warmup
    for i in range(num_warmup):
        _ = kernel_basic(dia_data_gpu, dia_offsets_gpu, test_vectors_gpu[i])
    cupy.cuda.Stream.null.synchronize()

    # Timing
    times = []
    for i in range(num_warmup, num_warmup + num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        _ = kernel_basic(dia_data_gpu, dia_offsets_gpu, test_vectors_gpu[i])
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times.append(elapsed_ms / 1000.0)

    results["3d_basic"] = {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "flops": 2 * np.count_nonzero(dia_data_3d),
    }

    # Test optimized 3D DIA kernel
    print("  Benchmarking optimized 3D DIA kernel...")
    kernel_opt = CustomDIA3DMatVec(use_optimized=True)

    # Warmup
    for i in range(num_warmup):
        _ = kernel_opt(dia_data_gpu, dia_offsets_gpu, test_vectors_gpu[i])
    cupy.cuda.Stream.null.synchronize()

    # Timing
    times = []
    for i in range(num_warmup, num_warmup + num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        _ = kernel_opt(dia_data_gpu, dia_offsets_gpu, test_vectors_gpu[i])
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times.append(elapsed_ms / 1000.0)

    results["3d_optimized"] = {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "flops": 2 * np.count_nonzero(dia_data_3d),
    }

    # Print results
    print("\nPerformance Results:")
    print("Method               | Time (ms)      | GFLOPS    | Speedup")
    print("---------------------|----------------|-----------|--------")

    baseline_time = results["dense_reference"]["mean_time"]

    for name, data in results.items():
        mean_ms = data["mean_time"] * 1000
        std_ms = data["std_time"] * 1000
        gflops = data["flops"] / data["mean_time"] / 1e9
        speedup = baseline_time / data["mean_time"]

        print(
            f"{name:19s}  | {mean_ms:6.2f}±{std_ms:5.2f} | {gflops:8.2f}  | {speedup:6.2f}x"
        )

    return results


def main():
    """Run all 3D DIA kernel tests."""
    print("3D DIA CUDA Kernel Test Suite")
    print("=" * 50)

    try:
        # Test correctness first
        correctness_passed = test_3d_dia_kernel_correctness()

        if correctness_passed:
            # Run performance benchmarks
            benchmark_results = benchmark_3d_dia_kernel()

            # Check if 3D kernels are faster than fallback would be
            opt_time_ms = benchmark_results["3d_optimized"]["mean_time"] * 1000
            if opt_time_ms < 10.0:  # Target: <10ms for 1000 LEDs
                print(
                    f"\n✓ 3D DIA kernel performance target met: {opt_time_ms:.2f}ms < 10ms"
                )
            else:
                print(
                    f"\n⚠ 3D DIA kernel slower than target: {opt_time_ms:.2f}ms > 10ms"
                )

            print("\n=== 3D DIA Kernel Implementation Complete ===")
            return 0
        else:
            print("\n✗ Correctness tests failed - not running benchmarks")
            return 1

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
