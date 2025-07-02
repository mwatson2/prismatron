#!/usr/bin/env python3
"""
Measure detailed optimization timing for DIA matrix operations.

Focus on individual operation timing to identify bottlenecks.
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


def measure_optimization_timing():
    """Measure detailed timing of optimization operations."""

    print("=== DETAILED OPTIMIZATION TIMING ANALYSIS ===")

    # Load patterns
    patterns_path = "diffusion_patterns/corrected_100.npz"
    patterns_data = np.load(patterns_path, allow_pickle=True)

    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_matrix = diffusion_csc.to_csc_matrix()
    led_positions = patterns_data["led_positions"]

    led_count = 100
    dia_matrix = DiagonalATAMatrix(led_count=led_count)

    print("Building DIA matrix...")
    build_start = time.time()
    dia_matrix.build_from_diffusion_matrix(csc_matrix, led_positions)
    build_time = time.time() - build_start
    print(f"DIA matrix build time: {build_time:.3f}s")
    print(f"DIA matrix diagonals: {dia_matrix.dia_data_cpu.shape[1]}")

    # Test data
    test_led_values = np.random.randn(3, led_count).astype(np.float32) * 0.5
    test_led_values_gpu = cp.asarray(test_led_values)

    print("\n--- Individual Operation Timing ---")

    # === Test multiply_3d (A^T A @ x) ===
    print("\nTesting multiply_3d (A^T A @ x):")

    # Warmup
    for _ in range(5):
        result = dia_matrix.multiply_3d(test_led_values_gpu)
        if not isinstance(result, cp.ndarray):
            result = cp.asarray(result)
        cp.cuda.Device().synchronize()

    # Detailed timing
    times = []
    for _ in range(50):  # More samples for precise measurement
        start = time.time()
        result = dia_matrix.multiply_3d(test_led_values_gpu)
        if not isinstance(result, cp.ndarray):
            result = cp.asarray(result)
        cp.cuda.Device().synchronize()
        times.append(time.time() - start)

    times_ms = np.array(times) * 1000
    print("  multiply_3d timing (50 samples):")
    print(f"    Mean: {times_ms.mean():.3f} ms")
    print(f"    Std:  {times_ms.std():.3f} ms")
    print(f"    Min:  {times_ms.min():.3f} ms")
    print(f"    Max:  {times_ms.max():.3f} ms")
    print(f"    95th: {np.percentile(times_ms, 95):.3f} ms")
    print(f"    Target: <1.000 ms ({'✅ PASS' if times_ms.mean() < 1.0 else '❌ FAIL'})")

    # === Test g_ata_g_3d (g^T @ A^T A @ g) ===
    print("\nTesting g_ata_g_3d (g^T @ A^T A @ g):")

    gradient_gpu = result  # Use multiply result as gradient

    # Warmup
    for _ in range(5):
        g_ata_g_result = dia_matrix.g_ata_g_3d(gradient_gpu)
        if not isinstance(g_ata_g_result, cp.ndarray):
            g_ata_g_result = cp.asarray(g_ata_g_result)
        cp.cuda.Device().synchronize()

    # Detailed timing
    times = []
    for _ in range(50):
        start = time.time()
        g_ata_g_result = dia_matrix.g_ata_g_3d(gradient_gpu)
        if not isinstance(g_ata_g_result, cp.ndarray):
            g_ata_g_result = cp.asarray(g_ata_g_result)
        cp.cuda.Device().synchronize()
        times.append(time.time() - start)

    times_ms = np.array(times) * 1000
    print("  g_ata_g_3d timing (50 samples):")
    print(f"    Mean: {times_ms.mean():.3f} ms")
    print(f"    Std:  {times_ms.std():.3f} ms")
    print(f"    Min:  {times_ms.min():.3f} ms")
    print(f"    Max:  {times_ms.max():.3f} ms")
    print(f"    95th: {np.percentile(times_ms, 95):.3f} ms")
    print(f"    Target: <1.000 ms ({'✅ PASS' if times_ms.mean() < 1.0 else '❌ FAIL'})")

    # === Test complete optimization step ===
    print("\nTesting complete optimization step:")

    # Simulate one optimization iteration
    # Step 1: A^T A @ x
    # Step 2: Compute gradient
    # Step 3: A^T A @ g
    # Step 4: g^T @ A^T A @ g
    # Step 5: Update x

    step_times = []
    for _ in range(20):
        start = time.time()

        # A^T A @ x
        ata_x = dia_matrix.multiply_3d(test_led_values_gpu)
        if not isinstance(ata_x, cp.ndarray):
            ata_x = cp.asarray(ata_x)

        # Simple gradient computation (ata_x - atb, but using ata_x as placeholder)
        gradient = ata_x

        # A^T A @ g
        ata_g = dia_matrix.multiply_3d(gradient)
        if not isinstance(ata_g, cp.ndarray):
            ata_g = cp.asarray(ata_g)

        # g^T @ A^T A @ g
        g_ata_g = dia_matrix.g_ata_g_3d(gradient)
        if not isinstance(g_ata_g, cp.ndarray):
            g_ata_g = cp.asarray(g_ata_g)

        # Simple update (just for timing)
        step_size = 0.001
        test_led_values_gpu -= step_size * gradient

        cp.cuda.Device().synchronize()
        step_times.append(time.time() - start)

    step_times_ms = np.array(step_times) * 1000
    print("  Complete optimization step (20 samples):")
    print(f"    Mean: {step_times_ms.mean():.3f} ms")
    print(f"    Std:  {step_times_ms.std():.3f} ms")
    print(f"    Min:  {step_times_ms.min():.3f} ms")
    print(f"    Max:  {step_times_ms.max():.3f} ms")
    print(f"    Target: <10.000 ms for 15fps ({'✅ PASS' if step_times_ms.mean() < 10.0 else '❌ FAIL'})")

    # === Scaling analysis ===
    print("\n--- Scaling Analysis ---")
    print(f"LED count: {led_count}")
    print(f"DIA diagonals: {dia_matrix.dia_data_cpu.shape[1]}")
    print(f"DIA efficiency: {dia_matrix.dia_data_cpu.shape[1] / led_count:.1f}x LED count")
    print(f"Matrix density: {dia_matrix.dia_data_cpu.shape[1] / led_count * 100:.1f}%")

    # Estimate for target LED counts
    target_leds = [500, 1000, 3200]
    for target in target_leds:
        # Assume similar diagonal density
        estimated_diagonals = dia_matrix.dia_data_cpu.shape[1] * (target / led_count)
        estimated_multiply_time = times_ms.mean() * (target / led_count) ** 1.5  # Rough scaling
        estimated_g_ata_g_time = times_ms.mean() * (target / led_count) ** 1.5
        estimated_total = estimated_multiply_time + estimated_g_ata_g_time

        print(f"\nEstimated performance for {target} LEDs:")
        print(f"  Est. diagonals: {estimated_diagonals:.0f}")
        print(f"  Est. multiply_3d: {estimated_multiply_time:.3f} ms")
        print(f"  Est. g_ata_g_3d: {estimated_g_ata_g_time:.3f} ms")
        print(f"  Est. total: {estimated_total:.3f} ms")
        print(f"  Target check: {'✅ FEASIBLE' if estimated_total < 10.0 else '❌ TOO SLOW'}")


if __name__ == "__main__":
    measure_optimization_timing()
