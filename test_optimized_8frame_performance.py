#!/usr/bin/env python3
"""
Test the optimized 8-frame kernel performance at production scale.
"""

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


def create_production_matrix():
    """Create production-scale matrix."""
    led_count = 2624
    batch_size = 8

    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count,
        crop_size=64,
        batch_size=batch_size,
        output_dtype=cupy.float32
    )

    # Identity matrix for consistent results
    dia_offsets = np.array([0], dtype=np.int32)
    dia_data = np.ones((3, 1, led_count), dtype=np.float32)
    dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
    matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

    return matrix


def test_optimized_vs_basic_performance():
    """Compare optimized vs basic kernel performance."""
    print("\n" + "="*80)
    print("OPTIMIZED VS BASIC KERNEL PERFORMANCE COMPARISON")
    print("="*80)

    led_count = 2624
    batch_size = 8
    num_trials = 10
    num_warmup = 3

    # Create matrix and input
    matrix = create_production_matrix()

    np.random.seed(42)
    input_batch = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.01
    input_batch_gpu = cupy.asarray(input_batch)

    print(f"Matrix: {led_count} LEDs ({matrix.led_blocks}x{matrix.led_blocks} blocks)")
    print(f"Input: {input_batch_gpu.shape}")
    print(f"Memory: {matrix.block_data_gpu.nbytes/(1024*1024):.1f}MB matrix + {input_batch_gpu.nbytes/(1024*1024):.1f}MB input")

    # Test 1: Basic kernel
    print("\n=== Basic Kernel (32 threads/block) ===")

    # Warmup
    for i in range(num_warmup):
        result_basic = matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=False, debug_logging=False)
        if i == 0:
            print(f"Basic result sum: {cupy.sum(cupy.abs(result_basic)):.1f}")
    cupy.cuda.Stream.null.synchronize()

    # Timing
    times_basic = []
    for trial in range(num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        result_basic = matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=False, debug_logging=False)
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times_basic.append(elapsed_ms)
        if trial < 3:
            print(f"Trial {trial+1}: {elapsed_ms:.2f}ms")

    time_basic_mean = np.mean(times_basic)
    time_basic_std = np.std(times_basic)

    # Test 2: Optimized kernel
    print("\n=== Optimized Kernel (128 threads/block) ===")

    # Warmup
    for i in range(num_warmup):
        result_optimized = matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=True, debug_logging=False)
        if i == 0:
            print(f"Optimized result sum: {cupy.sum(cupy.abs(result_optimized)):.1f}")
    cupy.cuda.Stream.null.synchronize()

    # Timing
    times_optimized = []
    for trial in range(num_trials):
        start_event = cupy.cuda.Event()
        end_event = cupy.cuda.Event()

        start_event.record()
        result_optimized = matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=True, debug_logging=False)
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cupy.cuda.get_elapsed_time(start_event, end_event)
        times_optimized.append(elapsed_ms)
        if trial < 3:
            print(f"Trial {trial+1}: {elapsed_ms:.2f}ms")

    time_optimized_mean = np.mean(times_optimized)
    time_optimized_std = np.std(times_optimized)

    # Correctness check
    print("\n=== Correctness Verification ===")
    max_diff = cupy.max(cupy.abs(result_basic - result_optimized))
    relative_error = max_diff / (cupy.max(cupy.abs(result_basic)) + 1e-10)
    print(f"Max difference: {max_diff:.9f}")
    print(f"Relative error: {relative_error:.9f}")

    if relative_error < 0.01:
        print("✅ Results match within acceptable tolerance")
    else:
        print("❌ Results differ significantly")

    # Performance analysis
    speedup = time_basic_mean / time_optimized_mean

    print("\n" + "="*80)
    print("PERFORMANCE RESULTS")
    print("="*80)
    print(f"Matrix size:          {led_count} LEDs")
    print(f"Batch size:           {batch_size} frames")
    print()
    print(f"Basic kernel:         {time_basic_mean:.2f} ± {time_basic_std:.2f} ms")
    print(f"Optimized kernel:     {time_optimized_mean:.2f} ± {time_optimized_std:.2f} ms")
    print(f"Speedup:              {speedup:.2f}x")
    print()

    # Throughput calculation
    operations = led_count * led_count * batch_size * 3  # Approximate FLOPS
    throughput_basic = operations / (time_basic_mean / 1000) / 1e9  # GFLOPS
    throughput_optimized = operations / (time_optimized_mean / 1000) / 1e9  # GFLOPS

    print(f"Basic throughput:     {throughput_basic:.1f} GFLOPS")
    print(f"Optimized throughput: {throughput_optimized:.1f} GFLOPS")
    print()

    # Tensor core efficiency analysis
    peak_tensor_flops = 165e12  # Approximate for RTX 4090/A100 class GPU (165 TFLOPS)
    efficiency_basic = (throughput_basic * 1e9) / peak_tensor_flops * 100
    efficiency_optimized = (throughput_optimized * 1e9) / peak_tensor_flops * 100

    print("Tensor core efficiency:")
    print(f"  Basic:              {efficiency_basic:.3f}% of peak")
    print(f"  Optimized:          {efficiency_optimized:.3f}% of peak")
    print()

    # Assessment
    if speedup > 2.0:
        print(f"🎉 EXCELLENT: {speedup:.1f}x speedup with optimized kernel!")
    elif speedup > 1.2:
        print(f"✅ GOOD: {speedup:.1f}x speedup with optimized kernel")
    elif speedup > 1.0:
        print(f"📈 MODEST: {speedup:.1f}x speedup with optimized kernel")
    else:
        print(f"⚠️ SLOWER: Optimized kernel is {1/speedup:.1f}x slower")

    if throughput_optimized > 100:
        print(f"⚡ HIGH THROUGHPUT: {throughput_optimized:.0f} GFLOPS achieved")
    elif throughput_optimized > 50:
        print(f"✅ GOOD THROUGHPUT: {throughput_optimized:.0f} GFLOPS achieved")
    else:
        print(f"⚠️ LOW THROUGHPUT: Only {throughput_optimized:.0f} GFLOPS achieved")

    return {
        'basic_time': time_basic_mean,
        'optimized_time': time_optimized_mean,
        'speedup': speedup,
        'throughput_optimized': throughput_optimized,
        'relative_error': relative_error
    }


def main():
    """Main performance test."""
    print("Optimized 8-Frame Kernel Performance Test")
    print("=========================================")

    try:
        results = test_optimized_vs_basic_performance()

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("✅ Optimized kernel test completed")
        print(f"⚡ Basic performance: {results['basic_time']:.1f}ms")
        print(f"🚀 Optimized performance: {results['optimized_time']:.1f}ms")
        print(f"📈 Speedup: {results['speedup']:.2f}x")
        print(f"💪 Throughput: {results['throughput_optimized']:.0f} GFLOPS")
        print(f"🎯 Accuracy: {results['relative_error']:.1e} relative error")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
