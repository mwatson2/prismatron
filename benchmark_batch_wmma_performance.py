#!/usr/bin/env python3
"""
Performance benchmark comparing BatchSymmetricDiagonalATAMatrix vs SymmetricDiagonalATAMatrix.

This benchmark tests the new WMMA tensor core batch implementation (16 frames at once)
against the existing implementation (16 sequential calls) using realistic parameters:
- Matrix size: 2624×2624 (target LED count)
- Diagonals: 841 non-empty diagonals
- Batch size: 16 frames
"""

import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, "/mnt/dev/prismatron/src")

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available")
    sys.exit(1)

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix


def create_realistic_2624_matrix():
    """Create a realistic 2624×2624 matrix with 841 non-empty diagonals."""
    print("Creating realistic 2624×2624 matrix with 841 diagonals...")

    led_count = 2624
    crop_size = 64

    # Create DiagonalATAMatrix
    ata_matrix = DiagonalATAMatrix(led_count=led_count, crop_size=crop_size)

    # Create diagonal pattern with 841 diagonals (same as debug script for consistency)
    max_offset = 420  # Half of 841 to get symmetric pattern
    offsets = list(range(-max_offset, max_offset + 1))  # Total: 841 diagonals
    k = len(offsets)

    print(f"  Creating matrix with {k} diagonals (offsets {min(offsets)} to {max(offsets)})")

    # Create stable diagonal data (consistent with debug script)
    dia_data_cpu = np.zeros((3, k, led_count), dtype=np.float32)

    for channel in range(3):
        for diag_idx, offset in enumerate(offsets):
            abs_offset = abs(offset)
            if abs_offset == 0:
                # Main diagonal: strong self-coupling
                dia_data_cpu[channel, diag_idx, :] = 5.0
            else:
                # Off-diagonals: stable decay pattern
                decay_factor = np.exp(-abs_offset / 10.0)  # Exponential decay
                dia_data_cpu[channel, diag_idx, :] = decay_factor

    # Set matrix data
    ata_matrix.dia_data_cpu = dia_data_cpu
    ata_matrix.dia_offsets = np.array(offsets, dtype=np.int32)
    ata_matrix.k = k
    ata_matrix.bandwidth = max_offset
    ata_matrix.sparsity = np.count_nonzero(dia_data_cpu) / dia_data_cpu.size
    ata_matrix.nnz = np.count_nonzero(dia_data_cpu)

    # Convert to GPU
    ata_matrix.dia_data_gpu = cp.asarray(dia_data_cpu)
    ata_matrix.dia_offsets_gpu = cp.asarray(offsets, dtype=cp.int32)

    print(f"  Matrix: {led_count}×{led_count}")
    print(f"  Diagonals: {k}")
    print(f"  Bandwidth: {ata_matrix.bandwidth}")
    print(f"  Sparsity: {ata_matrix.sparsity:.3f}")
    print(f"  Non-zeros: {ata_matrix.nnz:,}")
    print(f"  Memory: {ata_matrix.dia_data_gpu.nbytes / 1024**2:.1f} MB")

    return ata_matrix


def create_test_inputs(batch_size=16, led_count=2624, num_trials=50):
    """Create test input batches for benchmarking."""
    print(f"Creating {num_trials + 10} test input batches...")

    # Create random test inputs (batch_size, channels, led_count)
    test_inputs = []
    for i in range(num_trials + 10):  # Extra for warmup
        # Realistic input values (not too large to avoid overflow)
        batch_input = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.1
        test_inputs.append(cp.asarray(batch_input))

    print(f"  Created {len(test_inputs)} batches of shape ({batch_size}, 3, {led_count})")
    print(f"  Input range: [{test_inputs[0].min():.3f}, {test_inputs[0].max():.3f}]")

    return test_inputs


def benchmark_sequential_symmetric(symmetric_matrix, test_inputs, num_warmup=10, num_trials=50):
    """Benchmark sequential calls to SymmetricDiagonalATAMatrix (16 calls)."""
    print("\n" + "=" * 60)
    print("Benchmarking Sequential SymmetricDiagonalATAMatrix")
    print("=" * 60)

    batch_size = test_inputs[0].shape[0]

    # Warmup
    print(f"Running {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        batch_input = test_inputs[i]  # Shape: (16, 3, 2624)

        # Process each frame sequentially
        results = []
        for frame_idx in range(batch_size):
            frame_input = batch_input[frame_idx]  # Shape: (3, 2624)
            result = symmetric_matrix.multiply_3d(frame_input, use_custom_kernel=True, optimized_kernel=True)
            results.append(result)

        # Combine results (not timed)
        batch_result = cp.stack(results, axis=0)  # Shape: (16, 3, 2624)

    cp.cuda.Stream.null.synchronize()
    print("Warmup completed.")

    # Benchmark trials
    print(f"Running {num_trials} benchmark trials...")
    times = []

    for i in range(num_warmup, num_warmup + num_trials):
        batch_input = test_inputs[i]

        # Time the sequential processing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()

        # Process each frame sequentially
        results = []
        for frame_idx in range(batch_size):
            frame_input = batch_input[frame_idx]  # Shape: (3, 2624)
            result = symmetric_matrix.multiply_3d(frame_input, use_custom_kernel=True, optimized_kernel=True)
            results.append(result)

        # Combine results
        batch_result = cp.stack(results, axis=0)  # Shape: (16, 3, 2624)

        end_event.record()
        end_event.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        times.append(elapsed_ms)

    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\nSequential Results ({num_trials} trials):")
    print(f"  Mean time: {mean_time:.3f} ± {std_time:.3f} ms")
    print(f"  Min time:  {min_time:.3f} ms")
    print(f"  Max time:  {max_time:.3f} ms")
    print(f"  Throughput: {batch_size / (mean_time / 1000):.1f} frames/sec")

    return {
        "mean_ms": mean_time,
        "std_ms": std_time,
        "min_ms": min_time,
        "max_ms": max_time,
        "throughput_fps": batch_size / (mean_time / 1000),
        "times": times,
    }


def benchmark_batch_wmma(batch_matrix, test_inputs, num_warmup=10, num_trials=50):
    """Benchmark BatchSymmetricDiagonalATAMatrix (16 frames at once)."""
    print("\n" + "=" * 60)
    print("Benchmarking Batch WMMA BatchSymmetricDiagonalATAMatrix")
    print("=" * 60)

    batch_size = test_inputs[0].shape[0]

    # Warmup
    print(f"Running {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        batch_input = test_inputs[i]  # Shape: (16, 3, 2624)
        result = batch_matrix.multiply_batch_3d(
            batch_input, optimized_kernel=False, debug_logging=False  # Use basic WMMA kernel
        )

    cp.cuda.Stream.null.synchronize()
    print("Warmup completed.")

    # Benchmark trials
    print(f"Running {num_trials} benchmark trials...")
    times = []

    for i in range(num_warmup, num_warmup + num_trials):
        batch_input = test_inputs[i]

        # Time the batch processing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()

        # Process entire batch at once with WMMA tensor cores
        batch_result = batch_matrix.multiply_batch_3d(
            batch_input, optimized_kernel=False, debug_logging=False  # Use basic WMMA kernel
        )

        end_event.record()
        end_event.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        times.append(elapsed_ms)

    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\nBatch WMMA Results ({num_trials} trials):")
    print(f"  Mean time: {mean_time:.3f} ± {std_time:.3f} ms")
    print(f"  Min time:  {min_time:.3f} ms")
    print(f"  Max time:  {max_time:.3f} ms")
    print(f"  Throughput: {batch_size / (mean_time / 1000):.1f} frames/sec")

    return {
        "mean_ms": mean_time,
        "std_ms": std_time,
        "min_ms": min_time,
        "max_ms": max_time,
        "throughput_fps": batch_size / (mean_time / 1000),
        "times": times,
    }


def verify_correctness(symmetric_matrix, batch_matrix, test_input):
    """Verify that both methods produce the same results."""
    print("\n" + "=" * 60)
    print("Verifying Correctness")
    print("=" * 60)

    batch_size = test_input.shape[0]

    # Sequential processing
    print("Computing sequential results...")
    sequential_results = []
    for frame_idx in range(batch_size):
        frame_input = test_input[frame_idx]  # Shape: (3, 2624)
        result = symmetric_matrix.multiply_3d(frame_input, use_custom_kernel=True, optimized_kernel=True)
        sequential_results.append(result)

    sequential_batch = cp.stack(sequential_results, axis=0)  # Shape: (16, 3, 2624)

    # Batch processing
    print("Computing batch WMMA results...")
    batch_result = batch_matrix.multiply_batch_3d(test_input, optimized_kernel=False, debug_logging=False)

    # Compare results
    print("Comparing results...")
    sequential_cpu = cp.asnumpy(sequential_batch)
    batch_cpu = cp.asnumpy(batch_result)

    diff = np.abs(sequential_cpu - batch_cpu)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = max_diff / np.max(np.abs(sequential_cpu))

    print(f"  Sequential result range: [{sequential_cpu.min():.6f}, {sequential_cpu.max():.6f}]")
    print(f"  Batch result range:      [{batch_cpu.min():.6f}, {batch_cpu.max():.6f}]")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Relative difference:     {rel_diff:.6f} ({rel_diff*100:.3f}%)")

    # Success criteria
    tolerance = 1e-2  # Lenient for large matrix with FP16 operations
    if max_diff < tolerance:
        print(f"  ✅ CORRECTNESS VERIFIED (within {tolerance})")
        return True
    else:
        print(f"  ❌ CORRECTNESS FAILED (exceeds {tolerance})")
        return False


def main():
    """Main benchmark function."""
    print("WMMA Tensor Core Batch Performance Benchmark")
    print("=" * 80)
    print("Target: 2624×2624 matrix, 841 diagonals, 16-frame batches")
    print("=" * 80)

    if not CUPY_AVAILABLE:
        print("CuPy not available - cannot run benchmark")
        return

    # Parameters
    batch_size = 16
    led_count = 2624
    num_warmup = 10
    num_trials = 50

    print("Benchmark parameters:")
    print(f"  LED count: {led_count}")
    print(f"  Batch size: {batch_size}")
    print(f"  Warmup iterations: {num_warmup}")
    print(f"  Benchmark trials: {num_trials}")

    try:
        # Create realistic matrix
        print("\n" + "=" * 60)
        print("Creating Test Matrix")
        print("=" * 60)
        regular_matrix = create_realistic_2624_matrix()

        # Convert to symmetric storage
        print("\nConverting to symmetric storage...")
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        # Convert to batch storage with WMMA
        print("\nConverting to batch WMMA storage...")
        batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=batch_size)

        # Create test inputs
        test_inputs = create_test_inputs(batch_size, led_count, num_trials)

        # Verify correctness first
        correctness_ok = verify_correctness(symmetric_matrix, batch_matrix, test_inputs[0])
        if not correctness_ok:
            print("\n❌ Correctness verification failed - stopping benchmark")
            return

        # Benchmark sequential approach
        sequential_results = benchmark_sequential_symmetric(symmetric_matrix, test_inputs, num_warmup, num_trials)

        # Benchmark batch WMMA approach
        batch_results = benchmark_batch_wmma(batch_matrix, test_inputs, num_warmup, num_trials)

        # Performance comparison
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)

        speedup = sequential_results["mean_ms"] / batch_results["mean_ms"]
        throughput_improvement = batch_results["throughput_fps"] / sequential_results["throughput_fps"]

        print(f"Sequential (16 calls):     {sequential_results['mean_ms']:.3f} ± {sequential_results['std_ms']:.3f} ms")
        print(f"Batch WMMA (1 call):       {batch_results['mean_ms']:.3f} ± {batch_results['std_ms']:.3f} ms")
        print(f"Speedup:                   {speedup:.2f}x")
        print()
        print(f"Sequential throughput:     {sequential_results['throughput_fps']:.1f} frames/sec")
        print(f"Batch WMMA throughput:     {batch_results['throughput_fps']:.1f} frames/sec")
        print(f"Throughput improvement:    {throughput_improvement:.2f}x")
        print()
        print("Matrix operations per second:")
        print(f"  Sequential: {sequential_results['throughput_fps']:.1f} ops/sec")
        print(f"  Batch WMMA: {batch_results['throughput_fps']:.1f} ops/sec")

        # Memory and computational analysis
        matrix_ops_per_frame = led_count * symmetric_matrix.nnz  # Approximate
        print("\nComputational analysis:")
        print(f"  Matrix size: {led_count}×{led_count}")
        print(f"  Non-zeros: {symmetric_matrix.nnz:,}")
        print(f"  Approx ops per frame: {matrix_ops_per_frame:,}")
        print(f"  Batch WMMA GFLOPS: {batch_results['throughput_fps'] * matrix_ops_per_frame / 1e9:.2f}")

        print("\n" + "=" * 80)
        if speedup > 1.5:
            print(f"🚀 EXCELLENT: {speedup:.2f}x speedup with batch WMMA tensor cores!")
        elif speedup > 1.1:
            print(f"✅ GOOD: {speedup:.2f}x speedup with batch WMMA tensor cores")
        else:
            print(f"⚠️  MINIMAL: Only {speedup:.2f}x speedup - optimization needed")
        print("=" * 80)

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
