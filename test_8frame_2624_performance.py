#!/usr/bin/env python3
"""
Performance test for 8-frame batch WMMA implementation at production scale (2624 LEDs).

Tests the performance characteristics of the 8-frame batch processing system
at the target production scale of 2624 LEDs.
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


def create_production_matrix(led_count: int, batch_size: int):
    """Create production-scale test matrix."""
    print(f"Creating {led_count} LED matrix with batch_size={batch_size}")

    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count,
        crop_size=64,  # Standard crop size
        batch_size=batch_size,
        output_dtype=cupy.float32
    )

    # Create realistic diagonal pattern (not just identity for production test)
    # Use multiple diagonals to simulate real diffusion patterns
    num_diagonals = min(10, led_count // 64)  # Reasonable number of diagonals
    dia_offsets = np.arange(num_diagonals, dtype=np.int32)
    dia_data = np.zeros((3, num_diagonals, led_count), dtype=np.float32)

    # Main diagonal
    dia_data[:, 0, :] = 1.0

    # Additional diagonals with decreasing weights
    for i in range(1, num_diagonals):
        weight = 1.0 / (i + 1)  # Decreasing weight
        dia_data[:, i, :-i] = weight

    print(f"Matrix diagonal pattern: {num_diagonals} diagonals")

    # Convert to GPU and build matrix
    dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
    matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

    print(f"Matrix built: {matrix.get_info()}")

    return matrix


def create_test_input(led_count: int, batch_size: int):
    """Create realistic test input patterns."""
    # Use random input to simulate real LED patterns
    np.random.seed(42)  # Reproducible results
    input_batch = np.random.randn(batch_size, 3, led_count).astype(np.float32)

    # Scale to reasonable LED brightness range
    input_batch = np.clip(input_batch * 0.5 + 0.5, 0.0, 1.0)

    return cupy.asarray(input_batch)


def test_production_performance():
    """Test performance at production scale (2624 LEDs)."""
    print("\n" + "="*80)
    print("PRODUCTION SCALE PERFORMANCE TEST: 2624 LEDs")
    print("="*80)

    led_count = 2624
    batch_size = 8
    num_trials = 10
    num_warmup = 3

    print(f"LED count: {led_count}")
    print(f"Batch size: {batch_size}")
    print(f"Matrix size: {math.ceil(led_count/16)}x{math.ceil(led_count/16)} blocks = {led_count}x{led_count} elements")
    print(f"Block count: {math.ceil(led_count/16)**2} blocks")

    # Create production matrix
    matrix = create_production_matrix(led_count, batch_size)
    input_batch = create_test_input(led_count, batch_size)

    print(f"Input batch shape: {input_batch.shape}")
    print(f"Input memory: {input_batch.nbytes / (1024*1024):.1f} MB")

    # Memory usage analysis
    matrix_memory = matrix.block_data_gpu.nbytes / (1024*1024)
    print(f"Matrix memory: {matrix_memory:.1f} MB")
    print(f"Total GPU memory: {(matrix_memory + input_batch.nbytes/(1024*1024)):.1f} MB")

    # Warmup runs
    print(f"\nWarming up with {num_warmup} runs...")
    for i in range(num_warmup):
        result = matrix.multiply_batch8_3d(input_batch, optimized_kernel=False, debug_logging=False)
        print(f"  Warmup {i+1}: result shape {result.shape}, sum {cupy.sum(cupy.abs(result)):.1f}")

    cupy.cuda.Stream.null.synchronize()

    # Performance measurement
    print(f"\nPerformance measurement with {num_trials} trials...")

    times_8frame = []
    for trial in range(num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        result = matrix.multiply_batch8_3d(input_batch, optimized_kernel=False, debug_logging=False)
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times_8frame.append(elapsed_ms)

        if trial < 3:  # Show first few results
            print(f"  Trial {trial+1}: {elapsed_ms:.2f}ms, result sum {cupy.sum(cupy.abs(result)):.1f}")

    # Sequential comparison (using fallback)
    print(f"\nSequential reference measurement with {num_trials} trials...")
    times_sequential = []
    for trial in range(num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        result_seq = matrix._sequential_8frame_fallback(input_batch, debug_logging=False)
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times_sequential.append(elapsed_ms)

        if trial < 3:
            print(f"  Trial {trial+1}: {elapsed_ms:.2f}ms, result sum {cupy.sum(cupy.abs(result_seq)):.1f}")

    # Calculate statistics
    time_8frame_mean = np.mean(times_8frame)
    time_8frame_std = np.std(times_8frame)
    time_sequential_mean = np.mean(times_sequential)
    time_sequential_std = np.std(times_sequential)

    speedup = time_sequential_mean / time_8frame_mean

    # Results
    print("\n" + "="*60)
    print("PRODUCTION PERFORMANCE RESULTS")
    print("="*60)
    print(f"Matrix size:           {led_count} LEDs ({math.ceil(led_count/16)}x{math.ceil(led_count/16)} blocks)")
    print(f"Batch size:            {batch_size} frames")
    print(f"Memory usage:          {matrix_memory:.1f} MB matrix + {input_batch.nbytes/(1024*1024):.1f} MB input")
    print()
    print(f"8-frame batch time:    {time_8frame_mean:.2f} ± {time_8frame_std:.2f} ms")
    print(f"Sequential time:       {time_sequential_mean:.2f} ± {time_sequential_std:.2f} ms")
    print(f"Speedup:               {speedup:.2f}x")
    print()

    # Performance analysis
    elements_processed = led_count * led_count * batch_size * 3  # 3 channels
    throughput_8frame = elements_processed / (time_8frame_mean / 1000) / 1e9  # GFLOPS
    throughput_sequential = elements_processed / (time_sequential_mean / 1000) / 1e9

    print(f"8-frame throughput:    {throughput_8frame:.2f} GFLOPS")
    print(f"Sequential throughput: {throughput_sequential:.2f} GFLOPS")
    print()

    # Real-world implications
    fps_8frame = 1000 / time_8frame_mean
    fps_sequential = 1000 / time_sequential_mean

    print(f"Potential FPS (8-frame):    {fps_8frame:.1f} FPS")
    print(f"Potential FPS (sequential): {fps_sequential:.1f} FPS")
    print()

    if speedup > 3.0:
        print(f"🎉 EXCELLENT: {speedup:.1f}x speedup achieved at production scale!")
    elif speedup > 2.0:
        print(f"✅ GOOD: {speedup:.1f}x speedup achieved at production scale")
    else:
        print(f"⚠️  MODEST: {speedup:.1f}x speedup achieved at production scale")

    if time_8frame_mean < 50:
        print(f"⚡ FAST: {time_8frame_mean:.1f}ms enables real-time processing")
    elif time_8frame_mean < 100:
        print(f"✅ ACCEPTABLE: {time_8frame_mean:.1f}ms suitable for near-real-time processing")
    else:
        print(f"⚠️  SLOW: {time_8frame_mean:.1f}ms may limit real-time performance")

    return {
        'led_count': led_count,
        'batch_size': batch_size,
        'time_8frame': time_8frame_mean,
        'time_sequential': time_sequential_mean,
        'speedup': speedup,
        'throughput_8frame': throughput_8frame,
        'memory_mb': matrix_memory + input_batch.nbytes/(1024*1024)
    }


def main():
    """Main performance test function."""
    print("8-Frame Batch WMMA Production Performance Test")
    print("=" * 50)

    try:
        results = test_production_performance()

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"✅ 8-frame implementation tested at production scale ({results['led_count']} LEDs)")
        print(f"⚡ Performance: {results['time_8frame']:.1f}ms per batch ({results['speedup']:.1f}x speedup)")
        print(f"💾 Memory usage: {results['memory_mb']:.1f}MB total GPU memory")
        print(f"🎯 Throughput: {results['throughput_8frame']:.1f} GFLOPS")

        return True

    except Exception as e:
        print(f"\n❌ Production test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
