#!/usr/bin/env python3
"""
Comprehensive ATA inverse performance test for frame optimizer.

This script tests the frame optimizer with and without ATA inverse initialization
using the 2600 LED pattern file to collect detailed performance data.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_2600_led_patterns_with_ata_inverse():
    """Load 2600 LED patterns with ATA inverse matrices."""
    pattern_path = Path(__file__).parent / "diffusion_patterns" / "synthetic_2600_64x64_v7.npz"

    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_path}")

    print(f"Loading 2600 LED patterns from: {pattern_path}")
    data = np.load(str(pattern_path), allow_pickle=True)

    # Load mixed tensor
    mixed_tensor_dict = data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    # Load DIA matrix
    dia_dict = data["dia_matrix"].item()
    dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)

    # Load ATA inverse
    ata_inverse = None
    if "ata_inverse" in data:
        ata_inverse = data["ata_inverse"]
        print(f"✅ ATA inverse loaded: shape={ata_inverse.shape}")
    else:
        print("❌ No ATA inverse found in pattern file")

    # Load metadata
    ata_inverse_metadata = None
    if "ata_inverse_metadata" in data:
        ata_inverse_metadata = data["ata_inverse_metadata"].item()
        print(f"✅ ATA inverse metadata: computation_time={ata_inverse_metadata['computation_time']:.1f}s")

    print(f"Mixed tensor: {mixed_tensor.batch_size} LEDs, {mixed_tensor.height}x{mixed_tensor.width}")
    print(f"DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k} diagonals")

    return mixed_tensor, dia_matrix, ata_inverse, ata_inverse_metadata


def create_test_frame():
    """Create a test frame with realistic content."""
    height, width = 480, 800
    frame = np.zeros((3, height, width), dtype=np.uint8)

    # Create some realistic patterns
    # Red stripe
    frame[0, 100:150, :] = 255

    # Green circle
    center_y, center_x = height // 2, width // 2
    radius = 80
    y, x = np.ogrid[:height, :width]
    circle_mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2
    frame[1, circle_mask] = 200

    # Blue gradient
    frame[2, :, :] = (np.arange(width) / width * 255).astype(np.uint8)

    # Add some noise for realism
    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame


def run_performance_comparison():
    """Run comprehensive performance comparison with and without ATA inverse."""
    print("=" * 80)
    print("ATA INVERSE PERFORMANCE COMPARISON - 2600 LEDs")
    print("=" * 80)

    # Load patterns
    try:
        mixed_tensor, dia_matrix, ata_inverse, ata_inverse_metadata = load_2600_led_patterns_with_ata_inverse()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    if ata_inverse is None:
        print("❌ ATA inverse not available - run compute_ata_inverse.py first")
        return

    led_count = mixed_tensor.batch_size

    # Test parameters
    max_iterations = 20
    num_warmup_runs = 3
    num_test_runs = 10
    convergence_threshold = 1e-2  # More relaxed for better convergence

    # Create test frame
    target_frame = create_test_frame()
    print(f"Test frame shape: {target_frame.shape}")

    print("\nTest parameters:")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Warmup runs: {num_warmup_runs}")
    print(f"  Test runs: {num_test_runs}")
    print(f"  LED count: {led_count}")

    # Warmup runs
    print("\n=== WARMUP PHASE ===")
    for i in range(num_warmup_runs):
        print(f"Warmup run {i + 1}/{num_warmup_runs}...")
        _ = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            max_iterations=5,
            debug=False,
        )
    print("Warmup completed")

    # Test WITHOUT ATA inverse
    print("\n=== WITHOUT ATA INVERSE ===")
    results_default = []

    for run in range(num_test_runs):
        print(f"Run {run + 1}/{num_test_runs}...", end="", flush=True)

        start_time = time.perf_counter()
        result = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            debug=False,
            enable_timing=True,
        )
        total_time = time.perf_counter() - start_time

        results_default.append(
            {
                "total_time": total_time,
                "iterations": result.iterations,
                "converged": result.converged,
                "timing_data": result.timing_data,
            }
        )

        print(f" {total_time:.3f}s, {result.iterations} iters")

    # Test WITH ATA inverse
    print("\n=== WITH ATA INVERSE ===")
    results_ata_inv = []

    for run in range(num_test_runs):
        print(f"Run {run + 1}/{num_test_runs}...", end="", flush=True)

        start_time = time.perf_counter()
        result = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            ATA_inverse=ata_inverse,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            debug=False,
            enable_timing=True,
        )
        total_time = time.perf_counter() - start_time

        results_ata_inv.append(
            {
                "total_time": total_time,
                "iterations": result.iterations,
                "converged": result.converged,
                "timing_data": result.timing_data,
            }
        )

        print(f" {total_time:.3f}s, {result.iterations} iters")

    # Analyze results
    print("\n=== PERFORMANCE ANALYSIS ===")

    # Overall metrics
    avg_time_default = np.mean([r["total_time"] for r in results_default])
    std_time_default = np.std([r["total_time"] for r in results_default])
    avg_iters_default = np.mean([r["iterations"] for r in results_default])
    convergence_rate_default = np.mean([float(r["converged"]) for r in results_default]) * 100

    avg_time_ata_inv = np.mean([r["total_time"] for r in results_ata_inv])
    std_time_ata_inv = np.std([r["total_time"] for r in results_ata_inv])
    avg_iters_ata_inv = np.mean([r["iterations"] for r in results_ata_inv])
    convergence_rate_ata_inv = np.mean([float(r["converged"]) for r in results_ata_inv]) * 100

    print("\nOverall Performance:")
    print("  DEFAULT initialization:")
    print(f"    Total time: {avg_time_default:.4f}±{std_time_default:.4f}s")
    print(f"    Iterations: {avg_iters_default:.1f}")
    print(f"    Convergence rate: {convergence_rate_default:.0f}%")
    print(f"    Time per iteration: {avg_time_default / avg_iters_default:.4f}s")

    print("  ATA INVERSE initialization:")
    print(f"    Total time: {avg_time_ata_inv:.4f}±{std_time_ata_inv:.4f}s")
    print(f"    Iterations: {avg_iters_ata_inv:.1f}")
    print(f"    Convergence rate: {convergence_rate_ata_inv:.0f}%")
    print(f"    Time per iteration: {avg_time_ata_inv / avg_iters_ata_inv:.4f}s")

    # Performance improvement
    speedup = avg_time_default / avg_time_ata_inv
    iteration_reduction = avg_iters_default - avg_iters_ata_inv
    iteration_reduction_pct = (iteration_reduction / avg_iters_default) * 100

    print("\nPerformance Improvement:")
    print(f"  Total speedup: {speedup:.2f}x")
    print(f"  Iteration reduction: {iteration_reduction:.1f} ({iteration_reduction_pct:.1f}%)")
    print(f"  Time savings: {(avg_time_default - avg_time_ata_inv) * 1000:.1f}ms per frame")

    # Detailed timing breakdown
    if results_default[0]["timing_data"] and results_ata_inv[0]["timing_data"]:
        print("\n=== DETAILED TIMING BREAKDOWN ===")

        # Average timing data across runs
        def average_timing_data(results):
            timing_keys = results[0]["timing_data"].keys()
            avg_timings = {}
            for key in timing_keys:
                times = [r["timing_data"][key] for r in results if key in r["timing_data"]]
                avg_timings[key] = np.mean(times)
            return avg_timings

        avg_timing_default = average_timing_data(results_default)
        avg_timing_ata_inv = average_timing_data(results_ata_inv)

        print("Section-wise comparison (average per run):")

        # Core optimization sections
        core_sections = [
            "ata_multiply",
            "gradient_calculation",
            "gradient_step",
            "convergence_check",
            "convergence_and_updates",
        ]

        for section in core_sections:
            if section in avg_timing_default and section in avg_timing_ata_inv:
                default_time = avg_timing_default[section]
                ata_inv_time = avg_timing_ata_inv[section]
                improvement = (default_time - ata_inv_time) / default_time * 100
                print(f"  {section:20s}: {default_time:.4f}s → {ata_inv_time:.4f}s ({improvement:+.1f}%)")

        # Initialization section (only in ATA inverse version)
        if "ata_inverse_initialization" in avg_timing_ata_inv:
            init_time = avg_timing_ata_inv["ata_inverse_initialization"]
            print(f"  {'ata_inverse_init':20s}: {init_time:.4f}s (ATA inverse only)")

    # Frame rate potential
    print("\n=== FRAME RATE POTENTIAL ===")
    fps_default = 1.0 / avg_time_default
    fps_ata_inv = 1.0 / avg_time_ata_inv

    print(f"  Default initialization: {fps_default:.1f} FPS")
    print(f"  ATA inverse initialization: {fps_ata_inv:.1f} FPS")
    print(f"  FPS improvement: {fps_ata_inv - fps_default:.1f} FPS (+{(fps_ata_inv / fps_default - 1) * 100:.1f}%)")

    # Memory usage summary
    print("\n=== MEMORY SUMMARY ===")
    if ata_inverse_metadata:
        ata_inv_memory = ata_inverse.nbytes / (1024 * 1024)
        print(f"  ATA inverse memory: {ata_inv_memory:.1f}MB")
        print(f"  Precomputation time: {ata_inverse_metadata['computation_time']:.1f}s (one-time cost)")
        print(f"  Condition numbers: {[f'{cn:.2e}' for cn in ata_inverse_metadata['condition_numbers']]}")

    print("\n✅ Performance analysis completed!")


if __name__ == "__main__":
    run_performance_comparison()
