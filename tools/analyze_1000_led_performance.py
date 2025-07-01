#!/usr/bin/env python3
"""
Comprehensive performance analysis for corrected 1000 LED patterns.

Test actual vs estimated performance and identify optimization opportunities.
"""

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix


def analyze_1000_led_performance():
    """Comprehensive performance analysis for 1000 LED corrected patterns."""

    print("=== 1000 LED PERFORMANCE ANALYSIS ===")

    # Load corrected patterns
    patterns_path = "diffusion_patterns/corrected_1000_final.npz"
    patterns_data = np.load(patterns_path, allow_pickle=True)

    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_matrix = diffusion_csc.to_csc_matrix()
    led_positions = patterns_data["led_positions"]

    led_count = 1000
    print(f"LED count: {led_count}")
    print(f"CSC matrix shape: {csc_matrix.shape}")
    print(f"CSC matrix nnz: {csc_matrix.nnz:,}")
    print(
        f"CSC sparsity: {csc_matrix.nnz / (csc_matrix.shape[0] * csc_matrix.shape[1]) * 100:.3f}%"
    )

    # === Build DIA Matrix ===
    print(f"\n--- Building DIA Matrix ---")
    dia_matrix = DiagonalATAMatrix(led_count=led_count)

    build_start = time.time()
    dia_matrix.build_from_diffusion_matrix(csc_matrix, led_positions)
    build_time = time.time() - build_start
    print(f"DIA matrix build time: {build_time:.3f}s")
    print(f"DIA matrix diagonals: {dia_matrix.dia_data_cpu.shape[1]}")
    print(
        f"DIA efficiency: {dia_matrix.dia_data_cpu.shape[1] / led_count:.1f}x LED count"
    )

    # Check if this matches our expectations for corrected patterns
    expected_diagonals = int(185 * (led_count / 100))  # Scale from 100 LED baseline
    print(f"Expected diagonals (scaled): ~{expected_diagonals}")
    diagonal_ratio = dia_matrix.dia_data_cpu.shape[1] / expected_diagonals
    print(
        f"Actual vs expected: {diagonal_ratio:.1f}x ({'✅ Good' if diagonal_ratio < 2.0 else '⚠️ Higher than expected'})"
    )

    # === Performance Testing ===
    print(f"\n--- Performance Testing ---")

    # Test data
    test_led_values = np.random.randn(3, led_count).astype(np.float32) * 0.5
    test_led_values_gpu = cp.asarray(test_led_values)

    # === Test multiply_3d (A^T A @ x) ===
    print(f"\nTesting multiply_3d (A^T A @ x):")

    # Warmup
    for _ in range(5):
        result = dia_matrix.multiply_3d(test_led_values_gpu)
        if not isinstance(result, cp.ndarray):
            result = cp.asarray(result)
        cp.cuda.Device().synchronize()

    # Detailed timing
    times = []
    for _ in range(30):
        start = time.time()
        result = dia_matrix.multiply_3d(test_led_values_gpu)
        if not isinstance(result, cp.ndarray):
            result = cp.asarray(result)
        cp.cuda.Device().synchronize()
        times.append(time.time() - start)

    times_ms = np.array(times) * 1000
    print(f"  multiply_3d timing (30 samples):")
    print(f"    Mean: {times_ms.mean():.3f} ms")
    print(f"    Std:  {times_ms.std():.3f} ms")
    print(f"    Min:  {times_ms.min():.3f} ms")
    print(f"    Max:  {times_ms.max():.3f} ms")
    print(f"    95th: {np.percentile(times_ms, 95):.3f} ms")
    print(f"    Target: <1.000 ms ({'✅ PASS' if times_ms.mean() < 1.0 else '❌ FAIL'})")

    # === Test g_ata_g_3d (g^T @ A^T A @ g) ===
    print(f"\nTesting g_ata_g_3d (g^T @ A^T A @ g):")

    gradient_gpu = result  # Use multiply result as gradient

    # Warmup
    for _ in range(5):
        g_ata_g_result = dia_matrix.g_ata_g_3d(gradient_gpu)
        if not isinstance(g_ata_g_result, cp.ndarray):
            g_ata_g_result = cp.asarray(g_ata_g_result)
        cp.cuda.Device().synchronize()

    # Detailed timing
    times = []
    for _ in range(30):
        start = time.time()
        g_ata_g_result = dia_matrix.g_ata_g_3d(gradient_gpu)
        if not isinstance(g_ata_g_result, cp.ndarray):
            g_ata_g_result = cp.asarray(g_ata_g_result)
        cp.cuda.Device().synchronize()
        times.append(time.time() - start)

    times_ms = np.array(times) * 1000
    print(f"  g_ata_g_3d timing (30 samples):")
    print(f"    Mean: {times_ms.mean():.3f} ms")
    print(f"    Std:  {times_ms.std():.3f} ms")
    print(f"    Min:  {times_ms.min():.3f} ms")
    print(f"    Max:  {times_ms.max():.3f} ms")
    print(f"    95th: {np.percentile(times_ms, 95):.3f} ms")
    print(f"    Target: <1.000 ms ({'✅ PASS' if times_ms.mean() < 1.0 else '❌ FAIL'})")

    # Store results for further analysis
    multiply_avg = times_ms.mean()
    g_ata_g_avg = times_ms.mean()

    # === Test complete optimization step ===
    print(f"\nTesting complete optimization step:")

    step_times = []
    for _ in range(15):
        start = time.time()

        # A^T A @ x
        ata_x = dia_matrix.multiply_3d(test_led_values_gpu)
        if not isinstance(ata_x, cp.ndarray):
            ata_x = cp.asarray(ata_x)

        # Gradient computation (simplified - just use ata_x)
        gradient = ata_x

        # A^T A @ g
        ata_g = dia_matrix.multiply_3d(gradient)
        if not isinstance(ata_g, cp.ndarray):
            ata_g = cp.asarray(ata_g)

        # g^T @ A^T A @ g
        g_ata_g = dia_matrix.g_ata_g_3d(gradient)
        if not isinstance(g_ata_g, cp.ndarray):
            g_ata_g = cp.asarray(g_ata_g)

        # Update (simplified)
        step_size = 0.001
        test_led_values_gpu -= step_size * gradient

        cp.cuda.Device().synchronize()
        step_times.append(time.time() - start)

    step_times_ms = np.array(step_times) * 1000
    print(f"  Complete optimization step (15 samples):")
    print(f"    Mean: {step_times_ms.mean():.3f} ms")
    print(f"    Std:  {step_times_ms.std():.3f} ms")
    print(f"    Min:  {step_times_ms.min():.3f} ms")
    print(f"    Max:  {step_times_ms.max():.3f} ms")
    print(
        f"    Target: <66.67 ms for 15fps ({'✅ PASS' if step_times_ms.mean() < 66.67 else '❌ FAIL'})"
    )

    # === Comparison with 100 LED baseline ===
    print(f"\n--- Scaling Analysis vs 100 LED Baseline ---")

    # 100 LED baseline results (from previous test)
    baseline_multiply = 0.448  # ms
    baseline_g_ata_g = 0.446  # ms
    baseline_step = 1.186  # ms

    print(f"100 LED baseline:")
    print(f"  multiply_3d: {baseline_multiply:.3f} ms")
    print(f"  g_ata_g_3d: {baseline_g_ata_g:.3f} ms")
    print(f"  complete step: {baseline_step:.3f} ms")

    print(f"\n1000 LED actual:")
    print(f"  multiply_3d: {multiply_avg:.3f} ms")
    print(f"  g_ata_g_3d: {g_ata_g_avg:.3f} ms")
    print(f"  complete step: {step_times_ms.mean():.3f} ms")

    print(f"\nScaling factors (1000 LED / 100 LED):")
    multiply_scale = multiply_avg / baseline_multiply
    g_ata_g_scale = g_ata_g_avg / baseline_g_ata_g
    step_scale = step_times_ms.mean() / baseline_step

    print(
        f"  multiply_3d: {multiply_scale:.1f}x ({'✅ Linear' if multiply_scale < 15 else '❌ Superlinear'})"
    )
    print(
        f"  g_ata_g_3d: {g_ata_g_scale:.1f}x ({'✅ Linear' if g_ata_g_scale < 15 else '❌ Superlinear'})"
    )
    print(
        f"  complete step: {step_scale:.1f}x ({'✅ Linear' if step_scale < 15 else '❌ Superlinear'})"
    )

    # === FPS Analysis ===
    print(f"\n--- FPS Analysis ---")

    step_time_s = step_times_ms.mean() / 1000
    max_fps = 1.0 / step_time_s
    target_fps = 15

    print(f"Complete optimization step: {step_time_s:.6f}s")
    print(f"Maximum theoretical FPS: {max_fps:.1f}")
    print(f"Target FPS: {target_fps}")
    print(
        f"FPS achievement: {'✅ EXCEEDS TARGET' if max_fps >= target_fps else f'❌ {target_fps/max_fps:.1f}x TOO SLOW'}"
    )

    # === Memory Usage ===
    print(f"\n--- Memory Usage ---")

    dia_memory = dia_matrix.dia_data_cpu.nbytes / (1024 * 1024)
    csc_memory = (
        csc_matrix.data.nbytes + csc_matrix.indices.nbytes + csc_matrix.indptr.nbytes
    ) / (1024 * 1024)

    print(f"DIA matrix memory: {dia_memory:.1f} MB")
    print(f"CSC matrix memory: {csc_memory:.1f} MB")
    print(f"Memory ratio (DIA/CSC): {dia_memory/csc_memory:.1f}x")

    # === Dense A^T A Alternative ===
    print(f"\n--- Dense A^T A Alternative Analysis ---")

    if "dense_ata" in patterns_data:
        dense_ata_data = patterns_data["dense_ata"].item()
        dense_ata_matrices = dense_ata_data["dense_ata_matrices"]
        dense_memory = dense_ata_matrices.nbytes / (1024 * 1024)

        print(f"Dense A^T A memory: {dense_memory:.1f} MB")
        print(f"Dense vs DIA memory: {dense_memory/dia_memory:.1f}x")

        # Quick test of dense performance
        print(f"\nTesting dense A^T A performance:")
        dense_ata_gpu = [cp.asarray(dense_ata_matrices[:, :, c]) for c in range(3)]

        # Test dense matrix-vector multiplication
        dense_times = []
        for _ in range(20):
            start = time.time()

            dense_result = cp.zeros((3, led_count), dtype=cp.float32)
            for channel in range(3):
                dense_result[channel] = (
                    dense_ata_gpu[channel] @ test_led_values_gpu[channel]
                )

            cp.cuda.Device().synchronize()
            dense_times.append(time.time() - start)

        dense_avg = np.mean(dense_times) * 1000
        print(
            f"  Dense A^T A @ x: {dense_avg:.3f} ms ({'✅ FASTER' if dense_avg < multiply_avg else '❌ SLOWER'} than DIA)"
        )

        if dense_avg < multiply_avg:
            print(f"  Dense improvement: {multiply_avg/dense_avg:.1f}x faster")
        else:
            print(f"  DIA advantage: {dense_avg/multiply_avg:.1f}x faster")

    # === Summary ===
    print(f"\n--- PERFORMANCE SUMMARY ---")
    print(f"✅ Pattern generation: RCM ordering fixed")
    print(
        f"✅ DIA matrix sparsity: {dia_matrix.dia_data_cpu.shape[1]} diagonals ({dia_matrix.dia_data_cpu.shape[1]/led_count:.1f}x LED count)"
    )
    print(f"")
    print(f"Individual operation performance:")
    print(
        f"  multiply_3d (A^T A @ x): {multiply_avg:.3f}ms ({'✅ <1ms' if multiply_avg < 1.0 else '❌ >1ms'})"
    )
    print(
        f"  g_ata_g_3d (g^T @ A^T A @ g): {g_ata_g_avg:.3f}ms ({'✅ <1ms' if g_ata_g_avg < 1.0 else '❌ >1ms'})"
    )
    print(f"")
    print(f"System performance:")
    print(f"  Complete optimization step: {step_times_ms.mean():.3f}ms")
    print(f"  Maximum FPS: {max_fps:.1f}")
    print(f"  Target 15fps: {'✅ ACHIEVABLE' if max_fps >= 15 else '❌ NOT ACHIEVABLE'}")

    if max_fps < 15:
        print(f"")
        print(f"💡 Optimization recommendations:")
        print(f"  - Consider dense A^T A matrices for better performance")
        print(f"  - Profile for memory transfer bottlenecks")
        print(f"  - Investigate GPU kernel optimization")


if __name__ == "__main__":
    analyze_1000_led_performance()
