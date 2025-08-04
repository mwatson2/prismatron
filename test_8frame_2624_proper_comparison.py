#!/usr/bin/env python3
"""
Proper performance comparison for 8-frame batch vs true sequential processing at 2624 LEDs.

This test compares against actual sequential matrix-vector multiplications,
not the broken GPU sequential fallback that returns zeros.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    import cupy
    CUDA_AVAILABLE = True
    print(f"✓ CUDA available: {cupy.cuda.runtime.runtimeGetVersion()}")
except ImportError:
    print("✗ CUDA not available")
    sys.exit(1)

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix


def create_identity_matrix_2624():
    """Create identity matrix for fair comparison."""
    led_count = 2624
    batch_size = 8

    print(f"Creating identity matrix: {led_count} LEDs, batch_size={batch_size}")

    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count,
        crop_size=64,
        batch_size=batch_size,
        output_dtype=cupy.float32
    )

    # Create identity matrix for fair comparison (A*x = x)
    dia_offsets = np.array([0], dtype=np.int32)  # Only main diagonal
    dia_data = np.ones((3, 1, led_count), dtype=np.float32)  # Identity

    dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
    matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

    print(f"Matrix info: {matrix.get_info()}")

    return matrix


def create_single_frame_matrix():
    """Create single-frame matrix for sequential comparison."""
    led_count = 2624
    batch_size = 1

    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count,
        crop_size=64,
        batch_size=batch_size,
        output_dtype=cupy.float32
    )

    # Same identity matrix
    dia_offsets = np.array([0], dtype=np.int32)
    dia_data = np.ones((3, 1, led_count), dtype=np.float32)
    dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
    matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

    return matrix


def test_production_scale_comparison():
    """Proper performance comparison at 2624 LEDs."""
    print("\n" + "="*80)
    print("PRODUCTION SCALE PROPER PERFORMANCE COMPARISON: 2624 LEDs")
    print("="*80)

    led_count = 2624
    batch_size = 8
    num_trials = 5  # Reduced for large matrix
    num_warmup = 2

    print(f"LED count: {led_count}")
    print(f"Batch size: {batch_size}")
    print(f"Matrix size: {led_count}x{led_count} = {led_count**2:,} elements")

    # Create matrices
    matrix_8frame = create_identity_matrix_2624()
    matrix_single = create_single_frame_matrix()

    # Create test input
    np.random.seed(42)
    input_batch = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.1
    input_batch_gpu = cupy.asarray(input_batch)

    print(f"Input batch shape: {input_batch_gpu.shape}")
    print(f"Input memory: {input_batch_gpu.nbytes / (1024*1024):.1f} MB")

    # Memory analysis
    matrix_memory = matrix_8frame.block_data_gpu.nbytes / (1024*1024)
    print(f"Matrix memory: {matrix_memory:.1f} MB")

    # Test 1: 8-frame batch processing
    print("\n=== 8-Frame Batch Processing ===")

    # Warmup
    for i in range(num_warmup):
        result_8frame = matrix_8frame.multiply_batch8_3d(input_batch_gpu, optimized_kernel=False)
        print(f"Warmup {i+1}: result sum {cupy.sum(cupy.abs(result_8frame)):.1f}")

    cupy.cuda.Stream.null.synchronize()

    # Timing
    times_8frame = []
    for trial in range(num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        result_8frame = matrix_8frame.multiply_batch8_3d(input_batch_gpu, optimized_kernel=False)
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times_8frame.append(elapsed_ms)
        print(f"Trial {trial+1}: {elapsed_ms:.2f}ms")

    time_8frame_mean = np.mean(times_8frame)
    time_8frame_std = np.std(times_8frame)

    # Test 2: True sequential processing
    print("\n=== True Sequential Processing (8 separate matrix-vector ops) ===")

    # Warmup sequential
    for i in range(num_warmup):
        results_seq = []
        for frame_idx in range(batch_size):
            single_input = input_batch_gpu[frame_idx:frame_idx+1]  # Keep batch dimension
            result_single = matrix_single.multiply_batch_3d(single_input, optimized_kernel=False)
            results_seq.append(result_single[0])  # Remove batch dimension
        result_sequential = cupy.stack(results_seq, axis=0)
        print(f"Warmup {i+1}: result sum {cupy.sum(cupy.abs(result_sequential)):.1f}")

    cupy.cuda.Stream.null.synchronize()

    # Timing sequential
    times_sequential = []
    for trial in range(num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()

        # Process each frame separately
        results_seq = []
        for frame_idx in range(batch_size):
            single_input = input_batch_gpu[frame_idx:frame_idx+1]
            result_single = matrix_single.multiply_batch_3d(single_input, optimized_kernel=False)
            results_seq.append(result_single[0])
        result_sequential = cupy.stack(results_seq, axis=0)

        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times_sequential.append(elapsed_ms)
        print(f"Trial {trial+1}: {elapsed_ms:.2f}ms")

    time_sequential_mean = np.mean(times_sequential)
    time_sequential_std = np.std(times_sequential)

    # Verify correctness
    print("\n=== Correctness Verification ===")
    max_diff = cupy.max(cupy.abs(result_8frame - result_sequential))
    relative_error = max_diff / (cupy.max(cupy.abs(result_8frame)) + 1e-10)
    print(f"Max difference: {max_diff:.6f}")
    print(f"Relative error: {relative_error:.6f}")

    if relative_error < 0.01:
        print("✅ Results match within acceptable tolerance")
    else:
        print("❌ Results differ significantly - potential correctness issue")

    # Performance analysis
    speedup = time_sequential_mean / time_8frame_mean

    print("\n" + "="*80)
    print("PRODUCTION PERFORMANCE RESULTS")
    print("="*80)
    print(f"Matrix size:            {led_count} LEDs ({led_count}x{led_count} matrix)")
    print(f"Batch size:             {batch_size} frames")
    print(f"Matrix memory:          {matrix_memory:.1f} MB")
    print(f"Input memory:           {input_batch_gpu.nbytes/(1024*1024):.1f} MB")
    print()
    print(f"8-frame batch time:     {time_8frame_mean:.2f} ± {time_8frame_std:.2f} ms")
    print(f"True sequential time:   {time_sequential_mean:.2f} ± {time_sequential_std:.2f} ms")
    print(f"Speedup:                {speedup:.2f}x")
    print()

    # Throughput analysis (approximate FLOPs)
    # For symmetric matrix-vector: roughly led_count^2 operations per frame
    operations = led_count * led_count * batch_size * 3  # 3 channels
    throughput_8frame = operations / (time_8frame_mean / 1000) / 1e9  # GFLOPS
    throughput_sequential = operations / (time_sequential_mean / 1000) / 1e9

    print(f"8-frame throughput:     {throughput_8frame:.1f} GFLOPS")
    print(f"Sequential throughput:  {throughput_sequential:.1f} GFLOPS")
    print()

    # Real-world implications
    fps_8frame = 1000 / time_8frame_mean
    fps_sequential = 1000 / time_sequential_mean

    print(f"Processing rate (8-frame):    {fps_8frame:.1f} batches/sec = {fps_8frame * batch_size:.1f} frames/sec")
    print(f"Processing rate (sequential): {fps_sequential:.1f} batches/sec = {fps_sequential * batch_size:.1f} frames/sec")
    print()

    # Performance assessment
    if speedup > 3.0:
        print(f"🎉 EXCELLENT: {speedup:.1f}x speedup at production scale!")
    elif speedup > 1.5:
        print(f"✅ GOOD: {speedup:.1f}x speedup at production scale")
    elif speedup > 1.0:
        print(f"✅ MODEST: {speedup:.1f}x speedup at production scale")
    else:
        print(f"⚠️  SLOWER: {speedup:.1f}x - sequential processing is faster")

    if time_8frame_mean < 10:
        print(f"⚡ VERY FAST: {time_8frame_mean:.1f}ms enables high-performance real-time processing")
    elif time_8frame_mean < 33:
        print(f"⚡ FAST: {time_8frame_mean:.1f}ms enables 30+ FPS real-time processing")
    elif time_8frame_mean < 100:
        print(f"✅ ACCEPTABLE: {time_8frame_mean:.1f}ms suitable for real-time processing")
    else:
        print(f"⚠️  SLOW: {time_8frame_mean:.1f}ms may limit real-time performance")

    return {
        'led_count': led_count,
        'time_8frame': time_8frame_mean,
        'time_sequential': time_sequential_mean,
        'speedup': speedup,
        'relative_error': relative_error,
        'throughput_8frame': throughput_8frame
    }


def main():
    """Main performance test."""
    print("8-Frame vs True Sequential Performance Test at Production Scale")
    print("=" * 70)

    try:
        results = test_production_scale_comparison()

        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"✅ Production scale test completed ({results['led_count']} LEDs)")
        print(f"⚡ 8-frame performance: {results['time_8frame']:.1f}ms per 8-frame batch")
        print(f"🏃 Sequential performance: {results['time_sequential']:.1f}ms per 8-frame batch")
        print(f"📈 Speedup: {results['speedup']:.2f}x")
        print(f"🎯 Accuracy: {results['relative_error']:.1e} relative error")
        print(f"💪 Throughput: {results['throughput_8frame']:.1f} GFLOPS")

        if results['speedup'] > 1.0:
            print(f"🎉 8-frame batch processing provides {results['speedup']:.1f}x speedup at production scale!")
        else:
            print(f"📊 At production scale, sequential processing is {1/results['speedup']:.1f}x faster")
            print("   This suggests overhead dominates at large matrix sizes for current implementation")

        return True

    except Exception as e:
        print(f"\n❌ Production test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
