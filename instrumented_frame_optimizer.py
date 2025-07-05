#!/usr/bin/env python3
"""
Instrumented frame optimizer with detailed timing breakdown.

This creates a timing-instrumented version of the frame optimizer that provides
detailed breakdowns of where time is spent during Mixed+DIA optimization.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import cupy as cp
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import FrameOptimizationResult
from src.utils.performance_timing import PerformanceTiming
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def instrumented_optimize_frame_led_values(
    target_frame: np.ndarray,
    at_matrix,
    ata_matrix,
    timer: PerformanceTiming,
    initial_values: Optional[np.ndarray] = None,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-6,
    step_size_scaling: float = 0.8,
    compute_error_metrics: bool = True,
    debug: bool = False,
) -> FrameOptimizationResult:
    """
    Instrumented version of optimize_frame_led_values with detailed timing.

    Args:
        target_frame: Target frame (H, W, 3) uint8 or (3, H, W) uint8
        at_matrix: A^T matrix (Mixed tensor or CSC)
        ata_matrix: A^T A matrix (DIA or dense)
        timer: PerformanceTiming instance
        (other args same as original function)

    Returns:
        FrameOptimizationResult with detailed timing
    """

    with timer.section("frame_preparation", use_gpu_events=True):
        # Input validation and format conversion
        with timer.section("input_validation"):
            if target_frame.dtype != np.int8 and target_frame.dtype != np.uint8:
                raise ValueError(f"Target frame must be int8 or uint8, got {target_frame.dtype}")

        with timer.section("format_conversion"):
            # Handle both planar (3, H, W) and standard (H, W, 3) formats
            if target_frame.shape == (3, 480, 800):
                target_planar = target_frame.astype(np.float32) / 255.0
            elif target_frame.shape == (480, 800, 3):
                target_normalized = target_frame.astype(np.float32) / 255.0
                target_planar = target_normalized.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
            else:
                raise ValueError(f"Unsupported frame shape {target_frame.shape}")

    with timer.section("atb_computation", use_gpu_events=True):
        # Compute A^T @ b using the AT matrix
        if isinstance(at_matrix, SingleBlockMixedSparseTensor):
            with timer.section("mixed_tensor_atb"):
                ATb = at_matrix.transpose_dot_product_3d(cp.asarray(target_planar))
        else:
            # Handle CSC matrix case
            with timer.section("csc_matrix_atb"):
                target_flattened = target_planar.reshape(3, -1)  # Shape: (3, pixels)
                csc_matrix = at_matrix.to_csc_matrix()  # Shape: (pixels, led_count*3)

                ATb_channels = []
                for channel in range(3):
                    channel_cols = np.arange(channel, csc_matrix.shape[1], 3)
                    AT_channel = csc_matrix[:, channel_cols]  # Shape: (pixels, led_count)
                    ATb_channel = AT_channel.T @ target_flattened[channel]  # Shape: (led_count,)
                    ATb_channels.append(ATb_channel)

                ATb = cp.stack(ATb_channels, axis=0)  # Shape: (3, led_count)

    with timer.section("led_initialization"):
        # Initialize LED values
        led_count = ATb.shape[1]
        print(f"DEBUG: ATb shape: {ATb.shape}, led_count: {led_count}")
        if initial_values is not None:
            led_values_gpu = cp.asarray(initial_values, dtype=cp.float32)
        else:
            led_values_gpu = cp.full((3, led_count), 0.5, dtype=cp.float32)
        print(f"DEBUG: led_values_gpu shape: {led_values_gpu.shape}")

    # Optimization loop with detailed timing
    step_sizes = [] if debug else None
    converged = False

    with timer.section("optimization_iterations", use_gpu_events=True):
        for iteration in range(max_iterations):
            with timer.section(f"iteration_{iteration:02d}", use_gpu_events=True):
                # Compute A^T A @ x
                with timer.section("ata_multiply", use_gpu_events=True):
                    if isinstance(ata_matrix, DiagonalATAMatrix):
                        with timer.section("dia_3d_kernel"):
                            ATA_x = ata_matrix.multiply_3d(
                                led_values_gpu,
                                use_custom_kernel=True,
                                optimized_kernel=True,
                            )
                            if not isinstance(ATA_x, cp.ndarray):
                                ATA_x = cp.asarray(ATA_x)
                    else:
                        # Dense matrix computation
                        with timer.section("dense_einsum"):
                            ATA_x = cp.einsum("ijc,cj->ci", cp.asarray(ata_matrix), led_values_gpu)

                # Compute gradient
                with timer.section("gradient_computation"):
                    gradient = ATA_x - ATb  # Shape: (3, led_count)

                # Compute step size
                with timer.section("step_size_computation", use_gpu_events=True):
                    with timer.section("gradient_dot_gradient"):
                        g_dot_g = cp.sum(gradient * gradient)

                    with timer.section("gradient_ata_gradient", use_gpu_events=True):
                        if isinstance(ata_matrix, DiagonalATAMatrix):
                            with timer.section("dia_quadratic_form"):
                                g_dot_ATA_g_per_channel = ata_matrix.g_ata_g_3d(
                                    gradient,
                                    use_custom_kernel=True,
                                    optimized_kernel=True,
                                )
                                if not isinstance(g_dot_ATA_g_per_channel, cp.ndarray):
                                    g_dot_ATA_g_per_channel = cp.asarray(g_dot_ATA_g_per_channel)
                                g_dot_ATA_g = cp.sum(g_dot_ATA_g_per_channel)
                        else:
                            with timer.section("dense_quadratic_form"):
                                g_dot_ATA_g = cp.einsum(
                                    "ci,ijc,cj->",
                                    gradient,
                                    cp.asarray(ata_matrix),
                                    gradient,
                                )

                    with timer.section("step_size_calculation"):
                        if g_dot_ATA_g > 0:
                            step_size = float(step_size_scaling * g_dot_g / g_dot_ATA_g)
                        else:
                            step_size = 0.01  # Fallback

                        if debug and step_sizes is not None:
                            step_sizes.append(step_size)

                # Update LED values
                with timer.section("led_update"):
                    led_values_new = cp.clip(led_values_gpu - step_size * gradient, 0, 1)

                # Check convergence
                with timer.section("convergence_check"):
                    delta = cp.linalg.norm(led_values_new - led_values_gpu)
                    if delta < convergence_threshold:
                        converged = True
                        if debug:
                            print(f"Converged after {iteration + 1} iterations, delta: {delta:.6f}")
                        break

                # Update values for next iteration
                with timer.section("values_update"):
                    led_values_gpu = led_values_new

    # Final processing
    with timer.section("result_processing"):
        with timer.section("gpu_to_cpu_transfer"):
            led_values_cpu = cp.asnumpy(led_values_gpu)

        with timer.section("output_scaling"):
            led_values_output = (led_values_cpu * 255.0).astype(np.uint8)

        # Compute error metrics if requested
        error_metrics = None
        if compute_error_metrics:
            with timer.section("error_metrics_computation"):
                # Simplified error computation for timing purposes
                residual = cp.linalg.norm(ATA_x - ATb)
                error_metrics = {
                    "residual_norm": float(residual),
                    "mse": float(residual**2 / (3 * led_count)),
                }

    # Create result
    result = FrameOptimizationResult(
        led_values=led_values_output,
        converged=converged,
        iterations=max_iterations if not converged else iteration + 1,
        error_metrics=error_metrics,
        step_sizes=np.array(step_sizes) if step_sizes else None,
    )

    return result


def run_detailed_timing_analysis():
    """Run detailed timing analysis using instrumented optimizer."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    print("=== Detailed Mixed+DIA Optimization Timing Analysis ===")

    # Load test data (one-time setup)
    patterns_path = "diffusion_patterns/synthetic_1000.npz"
    image_path = "flower_test.png"

    print(f"Loading patterns from: {patterns_path}")
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Load mixed tensor
    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
    print(f"Mixed tensor: {mixed_tensor.batch_size} LEDs")

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

    # Warmup runs
    print("\n=== Warmup Phase ===")
    warmup_iterations = 3
    for i in range(warmup_iterations):
        print(f"Warmup {i + 1}/{warmup_iterations}...")
        dummy_timer = PerformanceTiming("Warmup", enable_gpu_timing=False)
        _ = instrumented_optimize_frame_led_values(
            target_frame=target_image,
            at_matrix=mixed_tensor,
            ata_matrix=dia_matrix,
            timer=dummy_timer,
            max_iterations=10,
            compute_error_metrics=False,
            debug=False,
        )
    print("Warmup completed")

    # Detailed timing runs
    print("\n=== Detailed Timing Phase ===")
    timing_iterations = 5
    all_timers = []

    for run_idx in range(timing_iterations):
        print(f"Timing run {run_idx + 1}/{timing_iterations}...")

        # Create fresh timer for this run
        timer = PerformanceTiming(f"Run_{run_idx + 1}", enable_gpu_timing=True)

        # Run instrumented optimization
        result = instrumented_optimize_frame_led_values(
            target_frame=target_image,
            at_matrix=mixed_tensor,
            ata_matrix=dia_matrix,
            timer=timer,
            max_iterations=10,
            compute_error_metrics=True,
            debug=False,
        )

        all_timers.append(timer)

        # Print quick summary
        total_time = sum(s.duration for s in timer._sections.values())
        print(f"  Run {run_idx + 1} total: {total_time * 1000:.1f}ms")

    # Analyze results
    print(f"\n=== Analysis Across {timing_iterations} Runs ===")

    # Calculate average timings for each section
    section_stats = {}
    all_section_names = set()

    for timer in all_timers:
        for name in timer._sections:
            all_section_names.add(name)

    for section_name in all_section_names:
        durations = []
        gpu_durations = []

        for timer in all_timers:
            if section_name in timer._sections:
                section = timer._sections[section_name]
                durations.append(section.duration)
                if section.gpu_duration is not None:
                    gpu_durations.append(section.gpu_duration)

        if durations:
            section_stats[section_name] = {
                "mean": np.mean(durations),
                "std": np.std(durations),
                "min": np.min(durations),
                "max": np.max(durations),
                "gpu_mean": np.mean(gpu_durations) if gpu_durations else None,
                "count": len(durations),
            }

    # Sort by mean duration
    sorted_sections = sorted(section_stats.items(), key=lambda x: x[1]["mean"], reverse=True)

    # Print detailed breakdown
    print(f"\nDetailed Timing Breakdown (averaged over {timing_iterations} runs):")
    print("=" * 90)
    print(f"{'Section':<40} {'Mean (ms)':<12} {'Std (ms)':<12} {'GPU (ms)':<12} {'Count':<8}")
    print("-" * 90)

    total_time = section_stats.get("optimization_iterations", {}).get(
        "mean", sum(stats["mean"] for stats in section_stats.values())
    )

    for section_name, stats in sorted_sections:
        mean_ms = stats["mean"] * 1000
        std_ms = stats["std"] * 1000
        gpu_ms = stats["gpu_mean"] * 1000 if stats["gpu_mean"] else 0
        count = stats["count"]

        print(f"{section_name:<40} {mean_ms:<12.2f} {std_ms:<12.2f} {gpu_ms:<12.2f} {count:<8}")

    # Identify bottlenecks and optimization opportunities
    print("\n=== Performance Analysis ===")

    # Find the most expensive operations
    print("\n1. Most expensive operations:")
    for section_name, stats in sorted_sections[:10]:
        mean_ms = stats["mean"] * 1000
        percentage = (stats["mean"] / total_time * 100) if total_time > 0 else 0
        print(f"   {section_name}: {mean_ms:.2f}ms ({percentage:.1f}%)")

    # Look for repeated work that could be optimized
    print("\n2. Repeated operations (potential for optimization):")
    iteration_sections = [(name, stats) for name, stats in sorted_sections if "iteration_" in name]
    if iteration_sections:
        total_iteration_time = sum(stats["mean"] for _, stats in iteration_sections)
        print(f"   Total iteration time: {total_iteration_time * 1000:.2f}ms")
        for name, stats in iteration_sections[:5]:
            print(f"   {name}: {stats['mean'] * 1000:.2f}ms (×{stats['count']})")

    # Memory transfer analysis
    print("\n3. GPU operations:")
    gpu_sections = [(name, stats) for name, stats in sorted_sections if stats["gpu_mean"] is not None]
    if gpu_sections:
        total_gpu_time = sum(stats["gpu_mean"] for _, stats in gpu_sections)
        print(f"   Total GPU time: {total_gpu_time * 1000:.2f}ms")
        for name, stats in gpu_sections[:5]:
            print(f"   {name}: {stats['gpu_mean'] * 1000:.2f}ms")

    # Final summary
    overall_mean = np.mean([sum(s.duration for s in timer._sections.values()) for timer in all_timers])
    overall_std = np.std([sum(s.duration for s in timer._sections.values()) for timer in all_timers])

    print("\n=== Final Summary ===")
    print(f"Overall optimization time: {overall_mean * 1000:.1f}ms ± {overall_std * 1000:.1f}ms")
    print(f"Estimated FPS: {1 / overall_mean:.1f}")
    print(f"Target 15 FPS requires: {1000 / 15:.1f}ms")

    target_time_ms = 1000 / 15
    if overall_mean * 1000 > target_time_ms:
        speedup_needed = (overall_mean * 1000) / target_time_ms
        print(f"Speedup needed: {speedup_needed:.1f}x")
    else:
        print("✓ Target performance achieved!")

    # Export detailed data to CSV for further analysis
    export_filename = "mixed_dia_timing_analysis.csv"
    print(f"\nExporting detailed timing data to: {export_filename}")

    # Combine all timer data for export
    combined_timer = PerformanceTiming("CombinedAnalysis")
    for timer in all_timers:
        for name, section in timer._sections.items():
            if name not in combined_timer._sections:
                combined_timer._sections[name] = section
            else:
                # Average the durations
                existing = combined_timer._sections[name]
                existing.duration = (existing.duration + section.duration) / 2

    combined_timer.export_csv(export_filename)


if __name__ == "__main__":
    run_detailed_timing_analysis()
