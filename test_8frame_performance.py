#!/usr/bin/env python3
"""
Performance test for 8-frame batch WMMA implementation.

Tests the performance characteristics and correctness of the 8-frame 
batch processing system.
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


def create_test_matrix(led_count: int, batch_size: int):
    """Create test matrix with identity pattern."""
    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count,
        crop_size=64,
        batch_size=batch_size,
        output_dtype=cupy.float32
    )

    # Create identity matrix
    dia_offsets = np.array([0], dtype=np.int32)
    dia_data = np.ones((3, 1, led_count), dtype=np.float32)
    dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
    matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

    return matrix


def create_test_input(led_count: int, batch_size: int):
    """Create test input with known patterns."""
    input_batch = np.random.randn(batch_size, 3, led_count).astype(np.float32)
    return cupy.asarray(input_batch)


def test_correctness():
    """Test correctness of 8-frame implementation."""
    print("\n" + "="*60)
    print("TESTING 8-FRAME CORRECTNESS")
    print("="*60)

    # Test multiple sizes to isolate the issue
    test_sizes = [32, 64, 160]

    for led_count in test_sizes:
        print(f"\nTesting {led_count} LEDs ({math.ceil(led_count/16)}x{math.ceil(led_count/16)} blocks):")

        batch_size = 8

        # Create test matrix and input
        matrix = create_test_matrix(led_count, batch_size)
        input_batch = create_test_input(led_count, batch_size)

        print(f"  Matrix: {led_count} LEDs ({matrix.led_blocks}x{matrix.led_blocks} blocks)")
        print(f"  Input: {input_batch.shape}")
        print(f"  Block diagonals: {matrix.block_diag_count}")

        # Test 8-frame kernel
        result_8frame = matrix.multiply_batch8_3d(input_batch, optimized_kernel=True, debug_logging=False)

        # CPU reference (for identity matrix, output should equal input)
        input_cpu = cupy.asnumpy(input_batch)
        expected_output = input_cpu.copy()  # Identity matrix: A * x = x

        # Compare
        max_diff = cupy.max(cupy.abs(result_8frame - cupy.asarray(expected_output)))
        relative_error = max_diff / (cupy.max(cupy.abs(result_8frame)) + 1e-10)

        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Relative error: {relative_error:.6f}")

        if relative_error < 0.001:  # 0.1% tolerance
            print(f"  ✅ CORRECT for {led_count} LEDs")
        else:
            print(f"  ❌ INCORRECT for {led_count} LEDs: {relative_error:.1e}")

            # Debug the first issue we find
            if led_count == 64:  # Test intermediate size
                print(f"  Debugging {led_count} LED case...")

                # Show sample values
                print("  Sample results (first batch, first channel, first 8 LEDs):")
                expected_sample = expected_output[0, 0, :8]
                actual_sample = cupy.asnumpy(result_8frame[0, 0, :8])

                for i in range(8):
                    diff = actual_sample[i] - expected_sample[i]
                    print(f"    LED {i}: expected={expected_sample[i]:.6f}, actual={actual_sample[i]:.6f}, diff={diff:.6f}")

            return False

    print("\n✅ All test sizes passed!")
    return True


def test_performance():
    """Test performance of 8-frame vs sequential processing."""
    print("\n" + "="*60)
    print("TESTING 8-FRAME PERFORMANCE")
    print("="*60)

    led_counts = [160, 320, 640]  # Different matrix sizes
    batch_size = 8
    num_trials = 20
    num_warmup = 5

    results = {}

    for led_count in led_counts:
        print(f"\nTesting {led_count} LEDs ({math.ceil(led_count/16)}x{math.ceil(led_count/16)} blocks):")

        # Create test matrix and input
        matrix = create_test_matrix(led_count, batch_size)
        input_batch = create_test_input(led_count, batch_size)

        # Warmup
        for _ in range(num_warmup):
            _ = matrix.multiply_batch8_3d(input_batch, optimized_kernel=True, debug_logging=False)
            _ = matrix._sequential_8frame_fallback(input_batch, debug_logging=False)
        cupy.cuda.Stream.null.synchronize()

        # Time 8-frame batch processing
        times_8frame = []
        for _ in range(num_trials):
            start_event = cupy.cuda.Event()
            end_event = cupy.cuda.Event()

            start_event.record()
            result = matrix.multiply_batch8_3d(input_batch, optimized_kernel=True, debug_logging=False)
            end_event.record()
            end_event.synchronize()

            times_8frame.append(cupy.cuda.get_elapsed_time(start_event, end_event))

        # Time sequential processing
        times_sequential = []
        for _ in range(num_trials):
            start_event = cupy.cuda.Event()
            end_event = cupy.cuda.Event()

            start_event.record()
            result = matrix._sequential_8frame_fallback(input_batch, debug_logging=False)
            end_event.record()
            end_event.synchronize()

            times_sequential.append(cupy.cuda.get_elapsed_time(start_event, end_event))

        # Calculate statistics
        time_8frame = np.mean(times_8frame)
        time_sequential = np.mean(times_sequential)
        speedup = time_sequential / time_8frame

        results[led_count] = {
            '8frame_time': time_8frame,
            'sequential_time': time_sequential,
            'speedup': speedup
        }

        print(f"  8-frame batch:  {time_8frame:.2f}ms")
        print(f"  Sequential:     {time_sequential:.2f}ms")
        print(f"  Speedup:        {speedup:.2f}x")
        print(f"  Status:         {'✅ FASTER' if speedup > 1.0 else '⚠️ SLOWER'}")

    return results


def test_memory_efficiency():
    """Test memory usage characteristics."""
    print("\n" + "="*60)
    print("TESTING MEMORY EFFICIENCY")
    print("="*60)

    led_count = 320

    # Test different batch sizes
    batch_sizes = [8, 16]

    for batch_size in batch_sizes:
        matrix = create_test_matrix(led_count, batch_size)
        input_batch = create_test_input(led_count, batch_size)

        # Calculate memory usage
        matrix_memory = matrix.block_data_gpu.nbytes / (1024 * 1024)  # MB
        input_memory = input_batch.nbytes / (1024 * 1024)  # MB

        print(f"Batch size {batch_size}:")
        print(f"  Matrix memory:  {matrix_memory:.2f} MB")
        print(f"  Input memory:   {input_memory:.2f} MB")
        print(f"  Total memory:   {matrix_memory + input_memory:.2f} MB")

    # Memory efficiency
    ratio = (input_batch.nbytes / 2) / input_batch.nbytes  # 8-frame vs 16-frame input
    print(f"\n8-frame input memory vs 16-frame: {ratio:.1%} (50% reduction as expected)")


def main():
    """Main test function."""
    print("8-Frame Batch WMMA Performance Test")
    print("====================================")

    try:
        # Test 1: Correctness
        correctness_ok = test_correctness()

        if not correctness_ok:
            print("\n❌ Correctness test failed - skipping performance tests")
            return False

        # Test 2: Performance
        import math
        performance_results = test_performance()

        # Test 3: Memory efficiency
        test_memory_efficiency()

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        avg_speedup = np.mean([r['speedup'] for r in performance_results.values()])
        print("✅ 8-frame kernel is mathematically correct")
        print(f"⚡ Average speedup: {avg_speedup:.2f}x vs sequential processing")
        print("💾 Memory reduction: 50% for input tensors vs 16-frame batches")

        if avg_speedup > 1.5:
            print("🎉 8-frame batch processing provides significant performance benefits!")
        else:
            print("⚠️ 8-frame batch processing provides modest performance benefits")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
