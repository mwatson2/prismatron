#!/usr/bin/env python3
"""
Simple timing breakdown for Mixed+DIA optimization.

Uses basic timing with CuPy events to get accurate GPU timing breakdown.
"""

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def time_operation(name, func, *args, **kwargs):
    """Time a single operation with GPU synchronization."""
    # GPU timing
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    # CPU timing
    start_cpu = time.time()
    start_event.record()

    result = func(*args, **kwargs)

    end_event.record()
    end_cpu = time.time()

    # Synchronize and get GPU time
    cp.cuda.runtime.deviceSynchronize()
    gpu_time = (
        cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
    )  # Convert to seconds
    cpu_time = end_cpu - start_cpu

    print(f"{name:30s}: CPU {cpu_time * 1000:6.2f}ms | GPU {gpu_time * 1000:6.2f}ms")

    return result, gpu_time, cpu_time


def analyze_mixed_dia_timing():
    """Analyze Mixed+DIA optimization timing breakdown."""

    print("=== Mixed+DIA Optimization Timing Breakdown ===")

    # Load test data
    patterns_path = "diffusion_patterns/synthetic_1000.npz"
    image_path = "flower_test.png"

    print("\nLoading test data...")
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Load mixed tensor
    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    # Load DIA matrix
    dia_matrix = DiagonalATAMatrix(led_count=mixed_tensor.batch_size)
    csc_data_dict = patterns_data["diffusion_matrix"].item()
    from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_full = diffusion_csc.to_csc_matrix()
    led_positions = patterns_data["led_positions"]
    dia_matrix.build_from_diffusion_matrix(csc_full, led_positions)

    # Load test image
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((800, 480))
    target_image = np.array(image, dtype=np.uint8)

    print(f"Mixed tensor: {mixed_tensor.batch_size} LEDs")
    print(f"DIA matrix: shape {dia_matrix.dia_data_cpu.shape}")
    print(f"Target image: shape {target_image.shape}")

    # Warmup
    print("\n=== Warmup Phase ===")
    for i in range(3):
        _ = optimize_frame_led_values(
            target_frame=target_image,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            max_iterations=10,
            compute_error_metrics=False,
            debug=False,
        )
    print("Warmup completed")

    # Detailed timing analysis
    print("\n=== Detailed Timing Analysis ===")

    # Run multiple iterations and average
    num_runs = 5
    all_timings = {"total": [], "iterations": []}

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}:")

        # Time the full optimization
        result, gpu_time, cpu_time = time_operation(
            "Full optimization",
            optimize_frame_led_values,
            target_frame=target_image,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            max_iterations=10,
            compute_error_metrics=True,
            debug=False,
        )

        all_timings["total"].append(gpu_time)
        all_timings["iterations"].append(result.iterations)

        print(f"  Converged: {result.converged} in {result.iterations} iterations")
        if result.error_metrics:
            print(f"  MSE: {result.error_metrics.get('mse', 'N/A'):.6f}")

    # Calculate statistics
    print("\n=== Results Summary ===")

    total_times = np.array(all_timings["total"]) * 1000  # Convert to ms
    iterations = np.array(all_timings["iterations"])

    print("Total optimization time:")
    print(f"  Mean: {np.mean(total_times):.1f}ms ± {np.std(total_times):.1f}ms")
    print(f"  Range: {np.min(total_times):.1f}ms - {np.max(total_times):.1f}ms")

    print("Iterations:")
    print(f"  Mean: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")
    print(f"  Range: {np.min(iterations)} - {np.max(iterations)}")

    # Time per iteration
    time_per_iteration = total_times / iterations
    print("Time per iteration:")
    print(
        f"  Mean: {np.mean(time_per_iteration):.1f}ms ± {np.std(time_per_iteration):.1f}ms"
    )

    # Performance analysis
    print("\n=== Performance Analysis ===")

    mean_total_time = np.mean(total_times)
    mean_iterations = np.mean(iterations)

    print(
        f"Current performance: {mean_total_time:.1f}ms ({1000 / mean_total_time:.1f} FPS)"
    )
    print(f"Target for 15 FPS: {1000 / 15:.1f}ms")

    if mean_total_time > 1000 / 15:
        speedup_needed = mean_total_time / (1000 / 15)
        print(f"Speedup needed: {speedup_needed:.1f}x")
    else:
        print("✓ Target performance achieved!")

    # Break down by component (rough estimates)
    print("\n=== Component Breakdown (Estimates) ===")

    # These are rough estimates based on typical optimization patterns
    setup_overhead = 5  # ms - target preparation, initialization
    per_iteration_time = np.mean(time_per_iteration)

    print(f"Setup overhead: ~{setup_overhead:.1f}ms")
    total_iter_time = per_iteration_time * mean_iterations
    print(
        f"Per-iteration work: ~{per_iteration_time:.1f}ms × {mean_iterations:.1f} = {total_iter_time:.1f}ms"
    )

    # Per-iteration breakdown (estimates)
    print(f"\nPer-iteration breakdown (estimates for {per_iteration_time:.1f}ms):")
    print(f"  A^T A @ x (DIA multiply): ~{per_iteration_time * 0.4:.1f}ms (40%)")
    print(f"  Gradient computation: ~{per_iteration_time * 0.1:.1f}ms (10%)")
    print(f"  Step size calculation: ~{per_iteration_time * 0.3:.1f}ms (30%)")
    print(f"  LED update & convergence: ~{per_iteration_time * 0.2:.1f}ms (20%)")

    # Optimization opportunities
    print("\n=== Optimization Opportunities ===")

    print("1. Reduce iterations:")
    target_iterations = 5
    if mean_iterations > target_iterations:
        time_saved = (mean_iterations - target_iterations) * per_iteration_time
        print(f"   If reduced to {target_iterations}: save {time_saved:.1f}ms")

    print("2. Optimize per-iteration time:")
    target_per_iter = 5  # ms
    if per_iteration_time > target_per_iter:
        time_saved = (per_iteration_time - target_per_iter) * mean_iterations
        print(f"   If reduced to {target_per_iter:.1f}ms/iter: save {time_saved:.1f}ms")

    print("3. Potential optimizations:")
    print("   - Pre-compute repeated calculations")
    print("   - Optimize GPU memory transfers")
    print("   - Use more efficient convergence criteria")
    print("   - Reduce floating point precision if acceptable")

    # Theoretical performance limits
    print("\n=== Theoretical Limits ===")

    # Estimate based on raw kernel performance from earlier tests
    raw_3d_kernel_time = 2.87  # ms from standalone test
    print(f"Raw 3D DIA kernel: {raw_3d_kernel_time:.1f}ms")

    theoretical_min = raw_3d_kernel_time * mean_iterations * 2  # 2 calls per iteration
    call_details = (
        f"{mean_iterations:.1f} iterations × 2 calls × {raw_3d_kernel_time:.1f}ms"
    )
    print(f"Theoretical minimum: {theoretical_min:.1f}ms ({call_details})")

    overhead = mean_total_time - theoretical_min
    print(
        f"Current overhead: {overhead:.1f}ms ({overhead / mean_total_time * 100:.1f}%)"
    )


if __name__ == "__main__":
    analyze_mixed_dia_timing()
