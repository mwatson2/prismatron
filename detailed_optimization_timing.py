#!/usr/bin/env python3
"""
Detailed timing breakdown for Mixed A^T + DIA A^T A optimization.

Uses PerformanceTiming class to identify bottlenecks and repeated work
that could be optimized. Runs warmup iterations then averages timing
across multiple iterations.
"""

import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.performance_timing import PerformanceTiming
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def create_detailed_timing_optimization():
    """Create detailed timing breakdown of Mixed+DIA optimization."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    print("=== Detailed Mixed+DIA Optimization Timing Analysis ===")

    # Load test data
    patterns_path = "diffusion_patterns/synthetic_1000.npz"
    image_path = "flower_test.png"

    print(f"Loading patterns from: {patterns_path}")
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Initialize matrices (one-time setup)
    setup_timer = PerformanceTiming("Setup", enable_gpu_timing=True)

    with setup_timer.section("load_mixed_tensor"):
        mixed_tensor_dict = patterns_data["mixed_tensor"].item()
        mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
        print(f"Mixed tensor: {mixed_tensor.batch_size} LEDs")

    with setup_timer.section("load_dia_matrix"):
        dia_matrix = DiagonalATAMatrix(led_count=mixed_tensor.batch_size)
        # Build DIA matrix from CSC patterns
        csc_data_dict = patterns_data["diffusion_matrix"].item()
        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

        diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
        csc_full = diffusion_csc.to_csc_matrix()
        led_positions = patterns_data["led_positions"]
        dia_matrix.build_from_diffusion_matrix(csc_full, led_positions)
        print(f"DIA matrix: shape {dia_matrix.dia_data_cpu.shape}")

    with setup_timer.section("load_test_image"):
        from PIL import Image

        image = Image.open(image_path).convert("RGB").resize((800, 480))
        target_image = np.array(image, dtype=np.uint8)
        print(f"Test image: shape {target_image.shape}")

    # Log setup timing
    print(f"\nSetup completed in {sum(s.duration for s in setup_timer._sections.values()):.3f}s")
    setup_timer.log(logger, include_percentages=True, sort_by="time")

    # Warmup runs
    print("\n=== Warmup Phase ===")
    warmup_iterations = 3
    for i in range(warmup_iterations):
        print(f"Warmup {i + 1}/{warmup_iterations}...")
        _ = optimize_frame_led_values(
            target_frame=target_image,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            max_iterations=10,
            compute_error_metrics=False,
            debug=False,
        )
    print("Warmup completed")

    # Detailed timing runs
    print("\n=== Detailed Timing Phase ===")
    timing_iterations = 5
    max_opt_iterations = 10

    all_timers = []

    for run_idx in range(timing_iterations):
        print(f"\nTiming run {run_idx + 1}/{timing_iterations}...")

        # Create fresh timer for this run
        timer = PerformanceTiming(f"OptimizationRun_{run_idx + 1}", enable_gpu_timing=True)

        # Use the existing frame optimizer function with detailed timing
        with timer.section("full_optimization", use_gpu_events=True):
            result = optimize_frame_led_values(
                target_frame=target_image,
                AT_matrix=mixed_tensor,
                ATA_matrix=dia_matrix,
                max_iterations=max_opt_iterations,
                compute_error_metrics=True,
                debug=False,
            )

        # Store timer for later analysis
        all_timers.append(timer)

        # Log this run's timing
        run_total_time = sum(s.duration for s in timer._sections.values())
        print(f"Run {run_idx + 1} total time: {run_total_time:.3f}s")

    # Analyze all runs
    print(f"\n=== Timing Analysis Across {timing_iterations} Runs ===")

    # Calculate average timings
    avg_timings = {}
    section_names = set()

    for timer in all_timers:
        for name, section in timer._sections.items():
            section_names.add(name)

    for section_name in section_names:
        durations = []
        gpu_durations = []

        for timer in all_timers:
            if section_name in timer._sections:
                section = timer._sections[section_name]
                durations.append(section.duration)
                if section.gpu_duration is not None:
                    gpu_durations.append(section.gpu_duration)

        if durations:
            avg_timings[section_name] = {
                "avg_duration": np.mean(durations),
                "std_duration": np.std(durations),
                "min_duration": np.min(durations),
                "max_duration": np.max(durations),
                "count": len(durations),
                "avg_gpu_duration": np.mean(gpu_durations) if gpu_durations else None,
                "total_time_across_runs": np.sum(durations),
            }

    # Sort by average duration
    sorted_sections = sorted(avg_timings.items(), key=lambda x: x[1]["avg_duration"], reverse=True)

    # Print detailed breakdown
    print(f"\nDetailed Timing Breakdown (averaged over {timing_iterations} runs):")
    print("=" * 80)
    print(f"{'Section':<35} {'Avg (ms)':<10} {'Std (ms)':<10} {'Count':<8} {'Total %':<10}")
    print("-" * 80)

    total_avg_time = sum(data["avg_duration"] for _, data in sorted_sections if "full_optimization" in _)
    if total_avg_time == 0:
        total_avg_time = sum(data["avg_duration"] for _, data in sorted_sections)

    for section_name, data in sorted_sections:
        avg_ms = data["avg_duration"] * 1000
        std_ms = data["std_duration"] * 1000
        percentage = (data["avg_duration"] / total_avg_time * 100) if total_avg_time > 0 else 0

        print(f"{section_name:<35} {avg_ms:<10.2f} {std_ms:<10.2f} {data['count']:<8} {percentage:<10.1f}")

        # Show GPU timing if available
        if data["avg_gpu_duration"] is not None:
            gpu_ms = data["avg_gpu_duration"] * 1000
            print(f"  └─ GPU: {gpu_ms:.2f}ms")

    # Identify optimization opportunities
    print("\n=== Optimization Opportunities ===")

    # Find sections that could be moved to setup
    setup_candidates = []
    per_iteration_work = []

    for section_name, data in sorted_sections:
        if "initialization" in section_name or "preparation" in section_name:
            setup_candidates.append((section_name, data["avg_duration"] * 1000))
        elif "iteration_" in section_name:
            per_iteration_work.append((section_name, data["avg_duration"] * 1000))

    print("\n1. Setup optimization candidates:")
    for name, time_ms in setup_candidates:
        print(f"   - {name}: {time_ms:.2f}ms (could be done once)")

    print("\n2. Per-iteration work (most expensive):")
    iteration_sections = [
        (name, data) for name, data in sorted_sections if "iteration" in name or "dia_" in name or "gradient" in name
    ]
    for name, data in iteration_sections[:5]:
        avg_ms = data["avg_duration"] * 1000
        print(f"   - {name}: {avg_ms:.2f}ms")

    # Calculate potential savings
    setup_savings = sum(time_ms for _, time_ms in setup_candidates)
    print(f"\n3. Potential savings from setup optimization: {setup_savings:.2f}ms")

    # Memory transfer analysis
    print("\n=== Memory Transfer Analysis ===")
    total_transfers = 0
    total_transfer_size = 0

    for timer in all_timers:
        for section in timer._sections.values():
            if section.memory_transfers:
                total_transfers += len(section.memory_transfers)
                total_transfer_size += sum(t["size_mb"] for t in section.memory_transfers)

    if total_transfers > 0:
        print(f"Total memory transfers: {total_transfers}")
        print(f"Total transfer size: {total_transfer_size:.1f}MB")
        print(f"Average transfer size: {total_transfer_size / total_transfers:.1f}MB")
    else:
        print("No explicit memory transfers recorded")

    # Final summary
    overall_avg = np.mean([sum(s.duration for s in timer._sections.values()) for timer in all_timers])
    overall_std = np.std([sum(s.duration for s in timer._sections.values()) for timer in all_timers])

    print("\n=== Final Summary ===")
    print(f"Overall optimization time: {overall_avg * 1000:.1f}ms ± {overall_std * 1000:.1f}ms")
    print(f"Estimated FPS: {1 / overall_avg:.1f}")
    print(f"Target 15 FPS requires: {1000 / 15:.1f}ms (current: {overall_avg * 1000:.1f}ms)")

    if overall_avg * 1000 > 1000 / 15:
        speedup_needed = (overall_avg * 1000) / (1000 / 15)
        print(f"Speedup needed: {speedup_needed:.1f}x")
    else:
        print("✓ Target performance met!")


if __name__ == "__main__":
    create_detailed_timing_optimization()
