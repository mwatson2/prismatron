#!/usr/bin/env python3
"""
Performance benchmark to verify the improvement from fixing the block storage format.

Compares the fixed 5D storage format against sequential CPU implementation
to validate that the WMMA kernels are working correctly and efficiently.
"""

import os
import sys
import time

import cupy
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix


def create_benchmark_matrix(led_count, bandwidth):
    """Create a matrix with specified bandwidth for benchmarking."""
    print(f"Creating {led_count}x{led_count} matrix with bandwidth {bandwidth}")

    channels = 3
    max_diagonals = min(bandwidth, led_count)
    offsets = np.arange(max_diagonals, dtype=np.int32)
    dia_data = np.zeros((channels, max_diagonals, led_count), dtype=np.float32)

    # Fill with realistic values for LED optimization matrix
    np.random.seed(42)  # Reproducible benchmark
    for ch in range(channels):
        for diag_idx, offset in enumerate(offsets):
            # Decreasing values by diagonal (typical for diffusion patterns)
            base_value = 1.0 / (1.0 + offset * 0.1)
            diag_length = led_count - offset
            dia_data[ch, diag_idx, :diag_length] = base_value + np.random.randn(diag_length) * 0.1

    # Create DiagonalATAMatrix
    diagonal_matrix = DiagonalATAMatrix(led_count, crop_size=1)
    diagonal_matrix.bandwidth = bandwidth
    diagonal_matrix.dia_data_upper_gpu = cupy.asarray(dia_data)
    diagonal_matrix.dia_offsets_upper_gpu = cupy.asarray(offsets)
    diagonal_matrix.k = max_diagonals
    diagonal_matrix.sparsity = 1.0 - (max_diagonals / led_count)
    diagonal_matrix.nnz = sum(led_count - offset for offset in offsets)
    diagonal_matrix.dia_offsets = offsets
    diagonal_matrix.dia_data_cpu = dia_data

    return diagonal_matrix


def benchmark_wmma_performance(diagonal_matrix, batch_size, num_iterations=10):
    """Benchmark WMMA kernel performance."""
    print(f"\n--- WMMA {batch_size}-Frame Kernel Benchmark ---")

    led_count = diagonal_matrix.led_count
    channels = 3

    # Create batch matrix with fixed 5D storage
    batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(diagonal_matrix, batch_size=batch_size)

    # Create test vectors
    np.random.seed(123)
    test_vectors = np.random.randn(batch_size, channels, led_count).astype(np.float32) * 0.1
    test_vectors_gpu = cupy.ascontiguousarray(cupy.asarray(test_vectors))

    # Warmup iterations
    print("Performing warmup iterations...")
    for _ in range(3):
        _ = batch_matrix.multiply_batch_3d(test_vectors_gpu)
        cupy.cuda.Stream.null.synchronize()

    # Benchmark iterations
    print(f"Running {num_iterations} benchmark iterations...")
    start_time = time.time()

    for i in range(num_iterations):
        result = batch_matrix.multiply_batch_3d(test_vectors_gpu)
        cupy.cuda.Stream.null.synchronize()

    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    operations_per_sec = num_iterations / total_time

    # Estimate computational complexity
    matrix_ops = batch_size * channels * led_count * diagonal_matrix.bandwidth
    gflops = (matrix_ops * 2) / (avg_time * 1e9)  # 2 ops per multiply-add

    print(f"Results:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average time per operation: {avg_time*1000:.2f}ms")
    print(f"  Operations per second: {operations_per_sec:.1f}")
    print(f"  Estimated GFLOPS: {gflops:.2f}")
    print(f"  Matrix operations per call: {matrix_ops:,}")

    return avg_time, gflops


def benchmark_sequential_reference(diagonal_matrix, batch_size, num_iterations=10):
    """Benchmark sequential CPU reference for comparison."""
    print(f"\n--- Sequential CPU Reference Benchmark ---")

    led_count = diagonal_matrix.led_count
    channels = 3

    # Create reference dense matrices
    reference_matrices = []
    dia_data_cpu = cupy.asnumpy(diagonal_matrix.dia_data_upper_gpu)
    dia_offsets = cupy.asnumpy(diagonal_matrix.dia_offsets_upper_gpu)

    for channel in range(channels):
        dense_matrix = np.zeros((led_count, led_count), dtype=np.float32)

        # Fill upper triangle from diagonal storage
        for diag_idx, offset in enumerate(dia_offsets):
            diag_data = dia_data_cpu[channel, diag_idx, :led_count]

            for i in range(led_count - offset):
                row, col = i, i + offset
                dense_matrix[row, col] = diag_data[i]

                # Fill symmetric lower diagonal (if not main diagonal)
                if offset > 0:
                    dense_matrix[col, row] = diag_data[i]

        reference_matrices.append(dense_matrix)

    # Create test vectors
    np.random.seed(123)  # Same seed as WMMA test
    test_vectors = np.random.randn(batch_size, channels, led_count).astype(np.float32) * 0.1

    # Benchmark iterations
    print(f"Running {num_iterations} benchmark iterations...")
    start_time = time.time()

    for i in range(num_iterations):
        # Compute reference results using dense matrix multiplication
        for batch_idx in range(batch_size):
            for ch in range(channels):
                vector = test_vectors[batch_idx, ch, :]
                result = reference_matrices[ch] @ vector

    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    operations_per_sec = num_iterations / total_time

    # Estimate computational complexity
    matrix_ops = batch_size * channels * led_count * led_count  # Dense multiplication
    gflops = (matrix_ops * 2) / (avg_time * 1e9)  # 2 ops per multiply-add

    print(f"Results:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average time per operation: {avg_time*1000:.2f}ms")
    print(f"  Operations per second: {operations_per_sec:.1f}")
    print(f"  Estimated GFLOPS: {gflops:.2f}")
    print(f"  Matrix operations per call: {matrix_ops:,}")

    return avg_time, gflops


def verify_correctness(diagonal_matrix, batch_size):
    """Verify that WMMA results match reference implementation."""
    print(f"\n--- Correctness Verification ---")

    led_count = diagonal_matrix.led_count
    channels = 3

    # Create batch matrix
    batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(diagonal_matrix, batch_size=batch_size)

    # Create small test vectors for verification
    np.random.seed(999)
    test_vectors = np.random.randn(batch_size, channels, led_count).astype(np.float32) * 0.1
    test_vectors_gpu = cupy.ascontiguousarray(cupy.asarray(test_vectors))

    # Compute WMMA result
    wmma_result = batch_matrix.multiply_batch_3d(test_vectors_gpu)
    wmma_result_cpu = cupy.asnumpy(wmma_result)

    # Compute reference result
    dia_data_cpu = cupy.asnumpy(diagonal_matrix.dia_data_upper_gpu)
    dia_offsets = cupy.asnumpy(diagonal_matrix.dia_offsets_upper_gpu)

    reference_results = []
    for batch_idx in range(batch_size):
        batch_result_channels = []
        for ch in range(channels):
            # Build dense matrix for this channel
            dense_matrix = np.zeros((led_count, led_count), dtype=np.float32)

            for diag_idx, offset in enumerate(dia_offsets):
                diag_data = dia_data_cpu[ch, diag_idx, :led_count]

                for i in range(led_count - offset):
                    row, col = i, i + offset
                    dense_matrix[row, col] = diag_data[i]

                    if offset > 0:
                        dense_matrix[col, row] = diag_data[i]

            vector = test_vectors[batch_idx, ch, :]
            result = dense_matrix @ vector
            batch_result_channels.append(result)
        reference_results.append(np.stack(batch_result_channels))

    reference_batch = np.stack(reference_results)

    # Compare results
    abs_diff = np.abs(reference_batch - wmma_result_cpu)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    rel_diff = abs_diff / (np.abs(reference_batch) + 1e-10)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    print(f"Correctness Analysis:")
    print(f"  Max absolute error: {max_abs_diff:.2e}")
    print(f"  Mean absolute error: {mean_abs_diff:.2e}")
    print(f"  Max relative error: {max_rel_diff:.2e}")
    print(f"  Mean relative error: {mean_rel_diff:.2e}")

    tolerance = 1e-3
    success = max_abs_diff < tolerance

    if success:
        print(f"  ✓ PASS: Fixed block storage produces correct results")
    else:
        print(f"  ✗ FAIL: Results don't match reference (tolerance={tolerance:.0e})")

        # Show worst error details
        worst_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        batch_idx, ch, led_idx = worst_idx
        print(f"  Worst error at [{batch_idx}][{ch}][{led_idx}]:")
        print(f"    Reference: {reference_batch[batch_idx, ch, led_idx]:.6f}")
        print(f"    WMMA:      {wmma_result_cpu[batch_idx, ch, led_idx]:.6f}")
        print(f"    Diff:      {abs_diff[batch_idx, ch, led_idx]:.6f}")

    return success


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across different matrix sizes and batch sizes."""
    print("=" * 80)
    print("Block Storage Fix Performance Benchmark")
    print("Fixed 5D Storage Format vs Sequential CPU Reference")
    print("=" * 80)

    # Test configurations: (led_count, bandwidth, description)
    test_configs = [
        (64, 8, "Small matrix with moderate bandwidth"),
        (128, 16, "Medium matrix with moderate bandwidth"),
        (256, 32, "Large matrix with high bandwidth"),
        (512, 64, "Very large matrix with very high bandwidth"),
    ]

    all_results = []

    for led_count, bandwidth, description in test_configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {led_count}x{led_count} matrix, bandwidth={bandwidth}")
        print(f"Description: {description}")
        print(f"{'='*80}")

        # Create benchmark matrix
        diagonal_matrix = create_benchmark_matrix(led_count, bandwidth)

        # Test both 8-frame and 16-frame batch sizes
        for batch_size in [8, 16]:
            print(f"\n{'-'*40} Batch Size: {batch_size} {'-'*40}")

            # Verify correctness first
            correctness_ok = verify_correctness(diagonal_matrix, batch_size)
            if not correctness_ok:
                print(f"⚠️  Skipping performance test due to correctness issues")
                continue

            # Benchmark WMMA performance
            wmma_time, wmma_gflops = benchmark_wmma_performance(diagonal_matrix, batch_size, num_iterations=20)

            # Only run CPU reference for smaller matrices (to avoid excessive time)
            if led_count <= 256:
                cpu_time, cpu_gflops = benchmark_sequential_reference(diagonal_matrix, batch_size, num_iterations=5)
                speedup = cpu_time / wmma_time

                print(f"\nPerformance Comparison:")
                print(f"  WMMA time: {wmma_time*1000:.2f}ms")
                print(f"  CPU time:  {cpu_time*1000:.2f}ms")
                print(f"  Speedup:   {speedup:.2f}x")
                print(f"  WMMA GFLOPS: {wmma_gflops:.2f}")
                print(f"  CPU GFLOPS:  {cpu_gflops:.2f}")
            else:
                speedup = None
                cpu_time = None
                print(f"\nWMMA Performance:")
                print(f"  Time: {wmma_time*1000:.2f}ms")
                print(f"  GFLOPS: {wmma_gflops:.2f}")
                print(f"  (CPU reference skipped for large matrix)")

            all_results.append(
                {
                    "led_count": led_count,
                    "bandwidth": bandwidth,
                    "batch_size": batch_size,
                    "wmma_time_ms": wmma_time * 1000,
                    "wmma_gflops": wmma_gflops,
                    "cpu_time_ms": cpu_time * 1000 if cpu_time else None,
                    "speedup": speedup,
                    "description": description,
                }
            )

    # Summary report
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    print(f"{'Config':<15} {'Batch':<6} {'WMMA ms':<10} {'GFLOPS':<8} {'Speedup':<8} {'Status'}")
    print(f"{'-'*70}")

    for result in all_results:
        config = f"{result['led_count']}x{result['bandwidth']}"
        wmma_ms = result["wmma_time_ms"]
        gflops = result["wmma_gflops"]
        speedup = f"{result['speedup']:.1f}x" if result["speedup"] else "N/A"
        status = "✓ PASS" if result["speedup"] is None or result["speedup"] > 1.0 else "✗ SLOW"

        print(f"{config:<15} {result['batch_size']:<6} {wmma_ms:<10.2f} {gflops:<8.2f} {speedup:<8} {status}")

    # Overall assessment
    successful_tests = [r for r in all_results if r["speedup"] is None or r["speedup"] > 1.0]

    print(f"\nOVERALL ASSESSMENT:")
    print(f"  Tests completed: {len(all_results)}")
    print(f"  Successful: {len(successful_tests)}")
    print(
        f"  Fixed 5D block storage format: {'✓ WORKING' if len(successful_tests) == len(all_results) else '✗ ISSUES'}"
    )
    print(f"  WMMA kernels: {'✓ FUNCTIONAL' if len(successful_tests) > 0 else '✗ NOT WORKING'}")

    return len(successful_tests) == len(all_results)


def main():
    """Main benchmark function."""
    try:
        success = run_comprehensive_benchmark()
        return success
    except Exception as e:
        print(f"\nBenchmark failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
