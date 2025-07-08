#!/usr/bin/env python3
"""
Standalone performance test for DiagonalATAMatrix multiply_3d method with fp16 data.

This test measures the real-world performance of the multiply_3d method using
actual fp16 kernels with 2624 LEDs and realistic bandwidth patterns.
"""

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

# from src.utils.performance_timing import PerformanceTiming


def load_2624_patterns():
    """Load 2624 LED patterns for realistic testing."""
    pattern_path = Path(__file__).parent / "diffusion_patterns" / "synthetic_2624_fp16_64x64.npz"

    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_path}")

    data = np.load(str(pattern_path), allow_pickle=True)

    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())
    dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())

    return mixed_tensor, dia_matrix


def create_test_data_variants(led_count, num_variants=10):
    """Create multiple test data variants for realistic performance testing."""
    test_variants = []

    for i in range(num_variants):
        # Create different types of realistic LED value patterns
        if i % 4 == 0:
            # Uniform distribution
            led_values = np.random.uniform(0.0, 1.0, (3, led_count)).astype(np.float32)
        elif i % 4 == 1:
            # Sparse pattern (mostly zeros with some bright values)
            led_values = np.zeros((3, led_count), dtype=np.float32)
            bright_indices = np.random.choice(led_count, led_count // 10, replace=False)
            led_values[:, bright_indices] = np.random.uniform(0.7, 1.0, (3, len(bright_indices)))
        elif i % 4 == 2:
            # Gradient pattern
            led_values = np.zeros((3, led_count), dtype=np.float32)
            for c in range(3):
                led_values[c] = np.linspace(0.0, 1.0, led_count)
                # Add some noise
                led_values[c] += np.random.normal(0, 0.1, led_count)
            led_values = np.clip(led_values, 0.0, 1.0)
        else:
            # Normal distribution centered around 0.5
            led_values = np.random.normal(0.5, 0.2, (3, led_count)).astype(np.float32)
            led_values = np.clip(led_values, 0.0, 1.0)

        test_variants.append(led_values)

    return test_variants


def run_multiply_3d_performance_test():
    """Run comprehensive performance test for multiply_3d method."""
    print("=" * 80)
    print("DIAGONAL ATA MATRIX MULTIPLY_3D FP16 PERFORMANCE TEST")
    print("=" * 80)

    # Load patterns
    try:
        mixed_tensor, dia_matrix = load_2624_patterns()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    led_count = mixed_tensor.batch_size
    print(f"Loaded patterns: {led_count} LEDs")
    print(f"DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k}")
    print(f"DIA matrix sparsity: {dia_matrix.sparsity:.1%}")
    print(f"DIA matrix storage: {dia_matrix.get_info()['storage_format']}")

    # Check fp16 support
    fp16_available = dia_matrix.get_info().get("custom_kernel_fp16_available", False)
    print(f"FP16 kernel available: {fp16_available}")

    if not fp16_available:
        print("⚠️  FP16 kernels not available, falling back to FP32")

    # Create test data variants
    print("\nCreating test data variants...")
    test_variants = create_test_data_variants(led_count, num_variants=10)
    print(f"Created {len(test_variants)} test variants")

    # Test parameters
    warmup_runs = 5
    measurement_runs = 20

    print("\nTest parameters:")
    print(f"  Warmup runs: {warmup_runs}")
    print(f"  Measurement runs: {measurement_runs}")
    print(f"  Test variants: {len(test_variants)}")

    # === FP32 Baseline Test ===
    print("\n" + "=" * 60)
    print("FP32 BASELINE PERFORMANCE")
    print("=" * 60)

    # Warmup phase for FP32
    print("Warming up FP32 kernels...")
    for i in range(warmup_runs):
        test_data = cp.asarray(test_variants[i % len(test_variants)], dtype=cp.float32)
        result = dia_matrix.multiply_3d(test_data, output_dtype=cp.float32)
        cp.cuda.Device().synchronize()

    # Measurement phase for FP32
    print("Measuring FP32 performance...")
    fp32_times = []

    for i in range(measurement_runs):
        test_data = cp.asarray(test_variants[i % len(test_variants)], dtype=cp.float32)

        start_time = time.perf_counter()
        result_fp32 = dia_matrix.multiply_3d(test_data, output_dtype=cp.float32)
        cp.cuda.Device().synchronize()
        end_time = time.perf_counter()

        fp32_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    fp32_avg = np.mean(fp32_times)
    fp32_std = np.std(fp32_times)
    fp32_min = np.min(fp32_times)
    fp32_max = np.max(fp32_times)

    print("FP32 Results:")
    print(f"  Average: {fp32_avg:.3f}ms")
    print(f"  Std Dev: {fp32_std:.3f}ms")
    print(f"  Min: {fp32_min:.3f}ms")
    print(f"  Max: {fp32_max:.3f}ms")
    print(f"  Output shape: {result_fp32.shape}")
    print(f"  Output dtype: {result_fp32.dtype}")

    # === FP16 Performance Test ===
    print("\n" + "=" * 60)
    print("FP16 PERFORMANCE TEST")
    print("=" * 60)

    # Warmup phase for FP16
    print("Warming up FP16 kernels...")
    for i in range(warmup_runs):
        test_data = cp.asarray(test_variants[i % len(test_variants)], dtype=cp.float16)
        result = dia_matrix.multiply_3d(test_data, output_dtype=cp.float16)
        cp.cuda.Device().synchronize()

    # Measurement phase for FP16
    print("Measuring FP16 performance...")
    fp16_times = []

    for i in range(measurement_runs):
        test_data = cp.asarray(test_variants[i % len(test_variants)], dtype=cp.float16)

        start_time = time.perf_counter()
        result_fp16 = dia_matrix.multiply_3d(test_data, output_dtype=cp.float16)
        cp.cuda.Device().synchronize()
        end_time = time.perf_counter()

        fp16_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    fp16_avg = np.mean(fp16_times)
    fp16_std = np.std(fp16_times)
    fp16_min = np.min(fp16_times)
    fp16_max = np.max(fp16_times)

    print("FP16 Results:")
    print(f"  Average: {fp16_avg:.3f}ms")
    print(f"  Std Dev: {fp16_std:.3f}ms")
    print(f"  Min: {fp16_min:.3f}ms")
    print(f"  Max: {fp16_max:.3f}ms")
    print(f"  Output shape: {result_fp16.shape}")
    print(f"  Output dtype: {result_fp16.dtype}")

    # === Pure FP16 Performance Test ===
    print("\n" + "=" * 60)
    print("PURE FP16 PERFORMANCE TEST")
    print("=" * 60)

    # Test pure FP16 kernel availability
    try:
        # Warmup phase for pure FP16
        print("Warming up pure FP16 kernels...")
        for i in range(warmup_runs):
            test_data = cp.asarray(test_variants[i % len(test_variants)], dtype=cp.float16)
            result = dia_matrix.multiply_3d(test_data, output_dtype=cp.float16, use_pure_fp16=True)
            cp.cuda.Device().synchronize()

        # Measurement phase for pure FP16
        print("Measuring pure FP16 performance...")
        pure_fp16_times = []

        for i in range(measurement_runs):
            test_data = cp.asarray(test_variants[i % len(test_variants)], dtype=cp.float16)

            start_time = time.perf_counter()
            result_pure_fp16 = dia_matrix.multiply_3d(test_data, output_dtype=cp.float16, use_pure_fp16=True)
            cp.cuda.Device().synchronize()
            end_time = time.perf_counter()

            pure_fp16_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        pure_fp16_avg = np.mean(pure_fp16_times)
        pure_fp16_std = np.std(pure_fp16_times)
        pure_fp16_min = np.min(pure_fp16_times)
        pure_fp16_max = np.max(pure_fp16_times)

        print("Pure FP16 Results:")
        print(f"  Average: {pure_fp16_avg:.3f}ms")
        print(f"  Std Dev: {pure_fp16_std:.3f}ms")
        print(f"  Min: {pure_fp16_min:.3f}ms")
        print(f"  Max: {pure_fp16_max:.3f}ms")
        print(f"  Output shape: {result_pure_fp16.shape}")
        print(f"  Output dtype: {result_pure_fp16.dtype}")

        # Calculate speedup vs mixed precision FP16
        if fp16_avg > 0:
            pure_fp16_speedup = fp16_avg / pure_fp16_avg
            print(f"  Speedup vs Mixed FP16: {pure_fp16_speedup:.2f}x")

        # Store pure FP16 result for accuracy comparison
        result_pure_fp16_check = result_pure_fp16
        pure_fp16_available = True

    except Exception as e:
        print(f"Pure FP16 kernel not available: {e}")
        pure_fp16_available = False
        result_pure_fp16_check = None

    # === Accuracy Comparison ===
    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON")
    print("=" * 60)

    # Compare results on same input
    test_data_fp32 = cp.asarray(test_variants[0], dtype=cp.float32)
    test_data_fp16 = cp.asarray(test_variants[0], dtype=cp.float16)
    result_fp32_check = dia_matrix.multiply_3d(test_data_fp32, output_dtype=cp.float32)
    result_fp16_check = dia_matrix.multiply_3d(test_data_fp16, output_dtype=cp.float16)

    # Convert fp16 to fp32 for comparison
    result_fp16_as_fp32 = result_fp16_check.astype(cp.float32)

    # Calculate differences between FP32 and mixed FP16
    abs_diff = cp.abs(result_fp32_check - result_fp16_as_fp32)
    rel_diff = abs_diff / (cp.abs(result_fp32_check) + 1e-8)

    print("Accuracy Analysis (FP32 vs Mixed FP16):")
    print(f"  Max absolute difference: {float(cp.max(abs_diff)):.6e}")
    print(f"  Mean absolute difference: {float(cp.mean(abs_diff)):.6e}")
    print(f"  Max relative difference: {float(cp.max(rel_diff)):.6e}")
    print(f"  Mean relative difference: {float(cp.mean(rel_diff)):.6e}")

    # Compare pure FP16 if available
    if pure_fp16_available and result_pure_fp16_check is not None:
        # Get pure FP16 result for same input
        result_pure_fp16_check = dia_matrix.multiply_3d(test_data_fp16, output_dtype=cp.float16, use_pure_fp16=True)

        # Convert pure fp16 to fp32 for comparison
        result_pure_fp16_as_fp32 = result_pure_fp16_check.astype(cp.float32)

        # Compare pure FP16 vs FP32
        abs_diff_pure = cp.abs(result_fp32_check - result_pure_fp16_as_fp32)
        rel_diff_pure = abs_diff_pure / (cp.abs(result_fp32_check) + 1e-8)

        print("\nAccuracy Analysis (FP32 vs Pure FP16):")
        print(f"  Max absolute difference: {float(cp.max(abs_diff_pure)):.6e}")
        print(f"  Mean absolute difference: {float(cp.mean(abs_diff_pure)):.6e}")
        print(f"  Max relative difference: {float(cp.max(rel_diff_pure)):.6e}")
        print(f"  Mean relative difference: {float(cp.mean(rel_diff_pure)):.6e}")

        # Compare mixed FP16 vs pure FP16
        abs_diff_fp16 = cp.abs(result_fp16_as_fp32 - result_pure_fp16_as_fp32)
        rel_diff_fp16 = abs_diff_fp16 / (cp.abs(result_fp16_as_fp32) + 1e-8)

        print("\nAccuracy Analysis (Mixed FP16 vs Pure FP16):")
        print(f"  Max absolute difference: {float(cp.max(abs_diff_fp16)):.6e}")
        print(f"  Mean absolute difference: {float(cp.mean(abs_diff_fp16)):.6e}")
        print(f"  Max relative difference: {float(cp.max(rel_diff_fp16)):.6e}")
        print(f"  Mean relative difference: {float(cp.mean(rel_diff_fp16)):.6e}")

    # === Memory Usage Analysis ===
    print("\n" + "=" * 60)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 60)

    # Get memory info
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    # Test memory usage for FP32
    initial_memory = mempool.used_bytes()
    test_data = cp.asarray(test_variants[0], dtype=cp.float32)
    result_fp32_mem = dia_matrix.multiply_3d(test_data, output_dtype=cp.float32)
    fp32_memory = mempool.used_bytes() - initial_memory

    mempool.free_all_blocks()

    # Test memory usage for FP16
    initial_memory = mempool.used_bytes()
    test_data = cp.asarray(test_variants[0], dtype=cp.float16)
    result_fp16_mem = dia_matrix.multiply_3d(test_data, output_dtype=cp.float16)
    fp16_memory = mempool.used_bytes() - initial_memory

    print("Memory Usage:")
    print(f"  FP32 memory: {fp32_memory / (1024**2):.2f} MB")
    print(f"  FP16 memory: {fp16_memory / (1024**2):.2f} MB")
    print(
        f"  Memory savings: {(fp32_memory - fp16_memory) / (1024**2):.2f} MB ({(1 - fp16_memory/fp32_memory)*100:.1f}%)"
    )

    # === Performance Summary ===
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    speedup = fp32_avg / fp16_avg if fp16_avg > 0 else 0
    efficiency = (fp32_avg - fp16_avg) / fp32_avg * 100 if fp32_avg > 0 else 0

    print("Performance Comparison:")
    print(f"  FP32 average: {fp32_avg:.3f}ms")
    print(f"  Mixed FP16 average: {fp16_avg:.3f}ms")
    print(f"  Mixed FP16 speedup: {speedup:.2f}x")
    print(f"  Mixed FP16 efficiency gain: {efficiency:.1f}%")

    if pure_fp16_available:
        pure_speedup = fp32_avg / pure_fp16_avg if pure_fp16_avg > 0 else 0
        pure_efficiency = (fp32_avg - pure_fp16_avg) / fp32_avg * 100 if fp32_avg > 0 else 0
        mixed_vs_pure_speedup = fp16_avg / pure_fp16_avg if pure_fp16_avg > 0 else 0

        print(f"  Pure FP16 average: {pure_fp16_avg:.3f}ms")
        print(f"  Pure FP16 speedup vs FP32: {pure_speedup:.2f}x")
        print(f"  Pure FP16 efficiency vs FP32: {pure_efficiency:.1f}%")
        print(f"  Pure FP16 speedup vs Mixed FP16: {mixed_vs_pure_speedup:.2f}x")

    # Throughput calculation
    operations_per_second_fp32 = 1000 / fp32_avg if fp32_avg > 0 else 0
    operations_per_second_fp16 = 1000 / fp16_avg if fp16_avg > 0 else 0

    print("\nThroughput:")
    print(f"  FP32: {operations_per_second_fp32:.1f} ops/sec")
    print(f"  Mixed FP16: {operations_per_second_fp16:.1f} ops/sec")

    if pure_fp16_available:
        operations_per_second_pure_fp16 = 1000 / pure_fp16_avg if pure_fp16_avg > 0 else 0
        print(f"  Pure FP16: {operations_per_second_pure_fp16:.1f} ops/sec")

    # Target performance analysis
    target_fps = 60
    target_time_per_frame = 1000 / target_fps  # ms
    optimization_iterations_per_frame = 10  # Estimate
    target_time_per_multiply = target_time_per_frame / optimization_iterations_per_frame

    print(f"\nTarget Performance Analysis (for {target_fps} FPS):")
    print(f"  Target time per multiply_3d: {target_time_per_multiply:.3f}ms")
    print(f"  FP32 meets target: {'✅' if fp32_avg <= target_time_per_multiply else '❌'}")
    print(f"  FP16 meets target: {'✅' if fp16_avg <= target_time_per_multiply else '❌'}")

    print("\n✅ Performance test completed!")

    return {
        "fp32_avg": fp32_avg,
        "fp16_avg": fp16_avg,
        "speedup": speedup,
        "efficiency": efficiency,
        "accuracy_max_abs_diff": float(cp.max(abs_diff)),
        "accuracy_mean_abs_diff": float(cp.mean(abs_diff)),
        "fp32_memory_mb": fp32_memory / (1024**2),
        "fp16_memory_mb": fp16_memory / (1024**2),
        "fp32_times": fp32_times,
        "fp16_times": fp16_times,
    }


if __name__ == "__main__":
    try:
        results = run_multiply_3d_performance_test()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
