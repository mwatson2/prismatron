#!/usr/bin/env python3
"""
Unit tests for Pure FP16 DIA kernel.

This test suite validates:
1. Correctness against reference implementations
2. Performance compared to mixed-precision kernels
3. Memory efficiency
4. Type safety and error handling
"""

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.kernels.dia_matvec_fp16 import CustomDIA3DMatVecFP16
from src.utils.kernels.pure_fp16_dia_kernel import PureFP16DIA3DKernel, get_kernel_info


def create_test_dia_matrix(led_count: int, bandwidth: int, channels: int = 3) -> tuple:
    """
    Create a test 3D DIA matrix with realistic structure.

    Returns:
        tuple: (dia_data_fp32, dia_data_fp16, dia_offsets, reference_dense)
    """
    print(f"Creating test DIA matrix: {led_count} LEDs, bandwidth={bandwidth}, channels={channels}")

    # Create realistic diagonal pattern
    max_offset = bandwidth // 2
    offsets = list(range(-max_offset, max_offset + 1, 4))  # Every 4th diagonal
    num_bands = len(offsets)

    print(f"  Diagonal bands: {num_bands}")
    print(f"  Offset range: [{min(offsets)}, {max(offsets)}]")

    # Create random but structured data
    np.random.seed(42)  # For reproducibility

    dia_data_fp32 = np.zeros((channels, num_bands, led_count), dtype=np.float32)
    reference_dense = []

    for channel in range(channels):
        # Create per-channel DIA matrix
        for i, offset in enumerate(offsets):
            # Add some structure - stronger diagonal, weaker off-diagonals
            if offset == 0:
                # Main diagonal - stronger values
                dia_data_fp32[channel, i, :] = np.random.uniform(0.8, 1.2, led_count)
            else:
                # Off-diagonals - weaker values, more sparse
                values = np.random.uniform(0.1, 0.4, led_count)
                # Make sparse - zero out 70% of values
                mask = np.random.random(led_count) < 0.3
                dia_data_fp32[channel, i, :] *= mask

        # Create reference dense matrix for this channel
        dense_matrix = np.zeros((led_count, led_count), dtype=np.float32)
        for i, offset in enumerate(offsets):
            for j in range(led_count):
                row = j
                col = j + offset
                if 0 <= col < led_count:
                    dense_matrix[row, col] = dia_data_fp32[channel, i, j]

        reference_dense.append(dense_matrix)

    # Convert to FP16
    dia_data_fp16 = dia_data_fp32.astype(np.float16)

    # Create offsets array
    dia_offsets = np.array(offsets, dtype=np.int32)

    # Convert to GPU
    dia_data_fp32_gpu = cp.asarray(dia_data_fp32)
    dia_data_fp16_gpu = cp.asarray(dia_data_fp16)
    dia_offsets_gpu = cp.asarray(dia_offsets)

    return dia_data_fp32_gpu, dia_data_fp16_gpu, dia_offsets_gpu, reference_dense


def create_test_vectors(led_count: int, channels: int = 3, num_vectors: int = 5) -> tuple:
    """Create test input vectors with different patterns."""
    print(f"Creating {num_vectors} test vectors: {channels} channels, {led_count} LEDs each")

    test_vectors = []

    for i in range(num_vectors):
        if i == 0:
            # All ones
            vector = np.ones((channels, led_count), dtype=np.float32)
        elif i == 1:
            # Random uniform
            vector = np.random.uniform(0.0, 1.0, (channels, led_count)).astype(np.float32)
        elif i == 2:
            # Sparse pattern
            vector = np.zeros((channels, led_count), dtype=np.float32)
            indices = np.random.choice(led_count, led_count // 10, replace=False)
            vector[:, indices] = np.random.uniform(0.5, 1.0, (channels, len(indices)))
        elif i == 3:
            # Gradient pattern
            vector = np.zeros((channels, led_count), dtype=np.float32)
            for c in range(channels):
                vector[c] = np.linspace(0.0, 1.0, led_count)
        else:
            # Normal distribution
            vector = np.random.normal(0.5, 0.2, (channels, led_count)).astype(np.float32)
            vector = np.clip(vector, 0.0, 1.0)

        test_vectors.append(vector)

    return test_vectors


def test_correctness():
    """Test kernel correctness against reference implementation."""
    print("=" * 80)
    print("CORRECTNESS TESTING")
    print("=" * 80)

    # Test parameters
    led_count = 500  # Manageable size for dense reference
    bandwidth = 100
    channels = 3

    # Create test data
    dia_data_fp32, dia_data_fp16, dia_offsets, reference_dense = create_test_dia_matrix(led_count, bandwidth, channels)

    test_vectors = create_test_vectors(led_count, channels, num_vectors=3)

    # Initialize kernels
    pure_fp16_kernel = PureFP16DIA3DKernel(use_optimized=False)
    pure_fp16_optimized = PureFP16DIA3DKernel(use_optimized=True)
    mixed_precision_kernel = CustomDIA3DMatVecFP16(use_optimized=False)

    print(f"\nTesting {len(test_vectors)} different input patterns...")

    max_errors = []
    mean_errors = []

    for i, test_vector in enumerate(test_vectors):
        print(f"\n--- Test Vector {i+1} ---")

        # Convert test vector to appropriate types
        test_vector_fp32 = cp.asarray(test_vector, dtype=cp.float32)
        test_vector_fp16 = cp.asarray(test_vector, dtype=cp.float16)

        # Reference result using dense matrix multiplication
        reference_results = []
        for c in range(channels):
            dense_result = reference_dense[c] @ test_vector[c]
            reference_results.append(dense_result)
        reference_result = cp.asarray(np.array(reference_results), dtype=cp.float32)

        # Pure FP16 kernel results
        result_pure_fp16 = pure_fp16_kernel(dia_data_fp16, dia_offsets, test_vector_fp16)
        result_pure_fp16_opt = pure_fp16_optimized(dia_data_fp16, dia_offsets, test_vector_fp16)

        # Mixed precision result for comparison
        result_mixed = mixed_precision_kernel(dia_data_fp32, dia_offsets, test_vector_fp32)

        # Convert FP16 results to FP32 for comparison
        result_pure_fp16_fp32 = result_pure_fp16.astype(cp.float32)
        result_pure_fp16_opt_fp32 = result_pure_fp16_opt.astype(cp.float32)
        result_mixed_fp32 = result_mixed.astype(cp.float32)

        # Calculate errors vs reference
        error_pure = cp.abs(reference_result - result_pure_fp16_fp32)
        error_pure_opt = cp.abs(reference_result - result_pure_fp16_opt_fp32)
        error_mixed = cp.abs(reference_result - result_mixed_fp32)

        # Calculate relative errors
        ref_magnitude = cp.abs(reference_result) + 1e-8
        rel_error_pure = error_pure / ref_magnitude
        rel_error_pure_opt = error_pure_opt / ref_magnitude
        rel_error_mixed = error_mixed / ref_magnitude

        print(
            f"  Pure FP16 (basic):     max_abs_err={float(cp.max(error_pure)):.6e}, max_rel_err={float(cp.max(rel_error_pure)):.6e}"
        )
        print(
            f"  Pure FP16 (optimized): max_abs_err={float(cp.max(error_pure_opt)):.6e}, max_rel_err={float(cp.max(rel_error_pure_opt)):.6e}"
        )
        print(
            f"  Mixed precision:       max_abs_err={float(cp.max(error_mixed)):.6e}, max_rel_err={float(cp.max(rel_error_mixed)):.6e}"
        )

        max_errors.append(float(cp.max(rel_error_pure)))
        mean_errors.append(float(cp.mean(rel_error_pure)))

        # Verify basic and optimized give same results
        kernel_diff = cp.abs(result_pure_fp16 - result_pure_fp16_opt)
        print(f"  Basic vs Optimized:    max_diff={float(cp.max(kernel_diff)):.6e}")

        # Check if results are reasonable
        assert float(cp.max(rel_error_pure)) < 1e-3, f"Pure FP16 error too large: {float(cp.max(rel_error_pure))}"
        assert float(cp.max(kernel_diff)) < 1e-6, f"Basic vs optimized mismatch: {float(cp.max(kernel_diff))}"

    print("\n✅ Correctness test passed!")
    print(f"   Overall max relative error: {max(max_errors):.6e}")
    print(f"   Overall mean relative error: {np.mean(mean_errors):.6e}")

    return max(max_errors)


def test_type_safety():
    """Test that kernel properly rejects incorrect input types."""
    print("\n" + "=" * 80)
    print("TYPE SAFETY TESTING")
    print("=" * 80)

    # Create test data with correct types
    led_count = 100
    channels = 3
    num_bands = 10

    dia_data_fp16 = cp.random.random((channels, num_bands, led_count), dtype=cp.float32).astype(cp.float16)
    dia_offsets_int32 = cp.arange(num_bands, dtype=cp.int32)
    x_fp16 = cp.random.random((channels, led_count), dtype=cp.float32).astype(cp.float16)

    kernel = PureFP16DIA3DKernel()

    # Test correct types work
    try:
        result = kernel(dia_data_fp16, dia_offsets_int32, x_fp16)
        print("✅ Correct types accepted")
        assert result.dtype == cp.float16, f"Output should be FP16, got {result.dtype}"
        assert result.shape == (channels, led_count), f"Output shape wrong: {result.shape}"
    except Exception as e:
        print(f"❌ Correct types rejected: {e}")
        raise

    # Test wrong matrix dtype
    try:
        dia_data_fp32 = dia_data_fp16.astype(cp.float32)
        kernel(dia_data_fp32, dia_offsets_int32, x_fp16)
        print("❌ Wrong matrix dtype accepted (should fail)")
        raise AssertionError("Should have rejected FP32 matrix data")
    except TypeError as e:
        print(f"✅ Wrong matrix dtype rejected: {e}")

    # Test wrong input vector dtype
    try:
        x_fp32 = x_fp16.astype(cp.float32)
        kernel(dia_data_fp16, dia_offsets_int32, x_fp32)
        print("❌ Wrong input dtype accepted (should fail)")
        raise AssertionError("Should have rejected FP32 input vectors")
    except TypeError as e:
        print(f"✅ Wrong input dtype rejected: {e}")

    # Test wrong offsets dtype
    try:
        dia_offsets_int64 = dia_offsets_int32.astype(cp.int64)
        kernel(dia_data_fp16, dia_offsets_int64, x_fp16)
        print("❌ Wrong offsets dtype accepted (should fail)")
        raise AssertionError("Should have rejected int64 offsets")
    except TypeError as e:
        print(f"✅ Wrong offsets dtype rejected: {e}")

    # Test numpy arrays (should fail)
    try:
        dia_data_numpy = cp.asnumpy(dia_data_fp16)
        kernel(dia_data_numpy, dia_offsets_int32, x_fp16)
        print("❌ Numpy array accepted (should fail)")
        raise AssertionError("Should have rejected numpy array")
    except TypeError as e:
        print(f"✅ Numpy array rejected: {e}")

    print("✅ Type safety tests passed!")


def test_performance():
    """Test performance against mixed precision kernel."""
    print("\n" + "=" * 80)
    print("PERFORMANCE TESTING")
    print("=" * 80)

    # Test with realistic 2624 LED problem size
    led_count = 2624
    bandwidth = 400
    channels = 3

    # Create test data
    dia_data_fp32, dia_data_fp16, dia_offsets, _ = create_test_dia_matrix(led_count, bandwidth, channels)

    # Create test vectors for performance testing
    test_vectors = create_test_vectors(led_count, channels, num_vectors=10)

    # Initialize kernels
    pure_fp16_basic = PureFP16DIA3DKernel(use_optimized=False)
    pure_fp16_optimized = PureFP16DIA3DKernel(use_optimized=True)
    mixed_precision = CustomDIA3DMatVecFP16(use_optimized=True)

    # Convert test vectors
    test_vectors_fp32 = [cp.asarray(v, dtype=cp.float32) for v in test_vectors]
    test_vectors_fp16 = [cp.asarray(v, dtype=cp.float16) for v in test_vectors]

    # Warmup
    print("Warming up kernels...")
    for i in range(3):
        pure_fp16_basic(dia_data_fp16, dia_offsets, test_vectors_fp16[0])
        pure_fp16_optimized(dia_data_fp16, dia_offsets, test_vectors_fp16[0])
        mixed_precision(dia_data_fp32, dia_offsets, test_vectors_fp32[0])
        cp.cuda.Device().synchronize()

    # Benchmark each kernel
    kernels = [
        (
            "Pure FP16 (basic)",
            lambda: pure_fp16_basic(dia_data_fp16, dia_offsets, test_vectors_fp16[i % len(test_vectors_fp16)]),
        ),
        (
            "Pure FP16 (optimized)",
            lambda: pure_fp16_optimized(dia_data_fp16, dia_offsets, test_vectors_fp16[i % len(test_vectors_fp16)]),
        ),
        (
            "Mixed precision",
            lambda: mixed_precision(dia_data_fp32, dia_offsets, test_vectors_fp32[i % len(test_vectors_fp32)]),
        ),
    ]

    results = {}
    num_trials = 20

    for kernel_name, kernel_func in kernels:
        print(f"\nBenchmarking {kernel_name}...")
        times = []

        for i in range(num_trials):
            start_time = time.perf_counter()
            result = kernel_func()
            cp.cuda.Device().synchronize()
            end_time = time.perf_counter()

            times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        results[kernel_name] = {"avg": avg_time, "std": std_time, "min": min_time, "max": max_time, "times": times}

        print(f"  Average: {avg_time:.3f}ms")
        print(f"  Std Dev: {std_time:.3f}ms")
        print(f"  Range: [{min_time:.3f}, {max_time:.3f}]ms")

    # Performance comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    mixed_avg = results["Mixed precision"]["avg"]
    pure_basic_avg = results["Pure FP16 (basic)"]["avg"]
    pure_opt_avg = results["Pure FP16 (optimized)"]["avg"]

    print(f"Mixed precision:       {mixed_avg:.3f}ms")
    print(f"Pure FP16 (basic):     {pure_basic_avg:.3f}ms")
    print(f"Pure FP16 (optimized): {pure_opt_avg:.3f}ms")
    print()
    print(f"Speedup (basic vs mixed):     {mixed_avg / pure_basic_avg:.2f}x")
    print(f"Speedup (optimized vs mixed): {mixed_avg / pure_opt_avg:.2f}x")
    print(f"Optimized vs basic:           {pure_basic_avg / pure_opt_avg:.2f}x")

    # Memory usage analysis
    print("\n" + "=" * 60)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 60)

    # Calculate theoretical memory usage
    matrix_elements = channels * dia_offsets.shape[0] * led_count
    vector_elements = channels * led_count

    fp32_matrix_mb = matrix_elements * 4 / (1024**2)
    fp16_matrix_mb = matrix_elements * 2 / (1024**2)
    fp32_vector_mb = vector_elements * 4 / (1024**2)
    fp16_vector_mb = vector_elements * 2 / (1024**2)

    mixed_total_mb = fp32_matrix_mb + fp32_vector_mb + fp16_vector_mb  # FP32 in, FP16 out
    pure_fp16_total_mb = fp16_matrix_mb + fp16_vector_mb + fp16_vector_mb  # All FP16

    print("Matrix memory:")
    print(f"  FP32: {fp32_matrix_mb:.2f} MB")
    print(f"  FP16: {fp16_matrix_mb:.2f} MB")
    print("Vector memory (input + output):")
    print(f"  Mixed precision: {fp32_vector_mb + fp16_vector_mb:.2f} MB")
    print(f"  Pure FP16: {fp16_vector_mb * 2:.2f} MB")
    print("Total memory footprint:")
    print(f"  Mixed precision: {mixed_total_mb:.2f} MB")
    print(f"  Pure FP16: {pure_fp16_total_mb:.2f} MB")
    print(f"  Memory savings: {(1 - pure_fp16_total_mb/mixed_total_mb)*100:.1f}%")

    return results


def main():
    """Run all tests."""
    print("🚀 Starting Pure FP16 DIA Kernel Test Suite")
    print("=" * 80)

    # Print kernel info
    info = get_kernel_info()
    print(f"Kernel: {info['name']}")
    print(f"Precision: {info['precision']}")
    print(f"Features: {', '.join(info['features'])}")
    print()

    try:
        # Run tests
        max_error = test_correctness()
        test_type_safety()
        perf_results = test_performance()

        # Summary
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)

        print(f"✅ Correctness: Max relative error {max_error:.6e}")
        print("✅ Type Safety: All type checks passed")

        mixed_time = perf_results["Mixed precision"]["avg"]
        pure_time = perf_results["Pure FP16 (optimized)"]["avg"]
        speedup = mixed_time / pure_time

        print(f"✅ Performance: {speedup:.2f}x speedup over mixed precision")

        if speedup > 1.5:
            print("🏆 Excellent performance improvement!")
        elif speedup > 1.2:
            print("👍 Good performance improvement!")
        else:
            print("⚠️  Modest performance improvement - may need further optimization")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
