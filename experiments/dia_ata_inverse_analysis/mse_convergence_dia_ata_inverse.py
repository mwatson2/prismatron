#!/usr/bin/env python3
"""
MSE Convergence Analysis for DIA ATA Inverse Matrices

Analyzes MSE convergence for different DIA ATA inverse diagonal factors:
- Diagonal factors: 1.0, 1.4, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0
- Compares DIA ATA inverse vs dense ATA inverse performance
- Tests with uint8 A-matrix (available diagonal factor files)

The diagonal factor determines how many diagonals to keep in the ATA inverse
as a fraction of the number of diagonals in the original ATA matrix.

GPU Memory Management Notes:
- CuPy maintains a memory pool that doesn't automatically release memory back to OS
- Use cupy.cuda.Device().synchronize() to ensure operations complete
- Use cupy.get_default_memory_pool().free_all_blocks() to force memory release
- GPU operations are asynchronous - memory can persist after Python objects are deleted
- Always combine: del objects → synchronize() → free_all_blocks() → gc.collect()
- Critical for preventing OOM errors in iterative testing scenarios
"""

# Add src to path
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cupy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent / "src"))

from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.frame_optimizer import optimize_frame_led_values
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_flower_image() -> np.ndarray:
    """Load and prepare the flower image for testing."""
    flower_path = Path("source/images/flower")

    # Try common image extensions
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        if (flower_path.parent / f"{flower_path.name}{ext}").exists():
            flower_path = flower_path.parent / f"{flower_path.name}{ext}"
            break

    if not flower_path.exists():
        # Fallback: create a synthetic flower-like test pattern
        print("Warning: Flower image not found, creating synthetic test pattern")
        frame = np.zeros((480, 800, 3), dtype=np.uint8)

        # Create flower-like pattern
        center_y, center_x = 240, 400
        for y in range(480):
            for x in range(800):
                dx, dy = x - center_x, y - center_y
                dist = np.sqrt(dx * dx + dy * dy)
                angle = np.arctan2(dy, dx)

                # Petals pattern
                petal_intensity = (np.sin(6 * angle) + 1) / 2
                if dist < 150:
                    frame[y, x, 0] = int(255 * petal_intensity * (1 - dist / 150))  # Red petals
                    frame[y, x, 1] = int(128 * (1 - dist / 200))  # Green center
                    frame[y, x, 2] = int(64 * petal_intensity)  # Blue accent

        return frame.transpose(2, 0, 1)  # Convert to (3, H, W)

    # Load real flower image
    image = Image.open(flower_path)
    image = image.convert("RGB")
    image = image.resize((800, 480), Image.Resampling.LANCZOS)

    # Convert to numpy array and transpose to planar format
    frame = np.array(image).transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    return frame.astype(np.uint8)


def load_dia_ata_inverse_patterns(
    diagonal_factor: float,
) -> Tuple[SingleBlockMixedSparseTensor, DiagonalATAMatrix, DiagonalATAMatrix, np.ndarray]:
    """Load patterns with DIA ATA inverse for specified diagonal factor."""

    pattern_file = f"diffusion_patterns/synthetic_2624_uint8_dia_{diagonal_factor}.npz"
    pattern_path = Path(pattern_file)

    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_file}")

    print(f"Loading DIA ATA inverse patterns from: {pattern_path}")
    data = np.load(str(pattern_path), allow_pickle=True)

    # Load mixed tensor
    mixed_tensor_dict = data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    # Load DIA matrix
    dia_dict = data["dia_matrix"].item()
    dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)

    # Load DIA ATA inverse
    ata_inverse_dia_dict = data["ata_inverse_dia"].item()
    ata_inverse_dia = DiagonalATAMatrix.from_dict(ata_inverse_dia_dict)

    # Load dense ATA inverse for comparison
    ata_inverse_dense = data["ata_inverse"]

    print(f"  LED count: {mixed_tensor.batch_size}")
    print(f"  Mixed tensor: {mixed_tensor.dtype}")
    print(f"  DIA matrix: {dia_matrix.storage_dtype}")
    print(f"  DIA ATA inverse: bandwidth={ata_inverse_dia.bandwidth}, k={ata_inverse_dia.k}")
    print(f"  Dense ATA inverse: shape={ata_inverse_dense.shape}, dtype={ata_inverse_dense.dtype}")

    return mixed_tensor, dia_matrix, ata_inverse_dia, ata_inverse_dense


def benchmark_matrix_multiply_performance(
    dia_matrix: DiagonalATAMatrix,
    ata_inverse_dense: np.ndarray,
    led_count: int,
    warmup_runs: int = 2,
    timing_runs: int = 5,
) -> Tuple[float, float]:
    """
    Benchmark DIA vs dense matrix multiply performance with proper warmup.

    Timing includes all memory access (matrix and vector) but excludes other warmup.
    Each run uses different random input vectors to avoid cache effects.

    Args:
        dia_matrix: DIA format matrix
        ata_inverse_dense: Dense format matrix (3, led_count, led_count)
        led_count: Number of LEDs
        warmup_runs: Number of warmup iterations
        timing_runs: Number of timed iterations

    Returns:
        (dia_time_ms, dense_time_ms): Average times in milliseconds
    """
    print(f"  Benchmarking matrix multiply performance...")
    print(f"    Warmup runs: {warmup_runs}, timing runs: {timing_runs}")

    try:
        # Pre-generate all test vectors to ensure different inputs for each run
        # Using numpy random state for reproducibility
        rng = np.random.RandomState(42)

        # Create test vectors on GPU for DIA operations
        test_vectors_gpu = []
        for i in range(warmup_runs + timing_runs):
            # Create different random vectors each time to avoid cache effects
            vec_cpu = rng.randn(3, led_count).astype(np.float32)
            vec_gpu = cupy.asarray(vec_cpu)
            test_vectors_gpu.append(vec_gpu)

        # Create test vectors on CPU for dense operations (different from GPU ones)
        test_vectors_cpu = []
        for i in range(warmup_runs + timing_runs):
            # Create different random vectors each time
            vec = rng.randn(3, led_count).astype(np.float32)
            test_vectors_cpu.append(vec)

        # Convert dense matrix to GPU for fair comparison
        ata_inverse_dense_gpu = cupy.asarray(ata_inverse_dense)

        # === DIA Performance Testing ===

        # Warmup runs for DIA
        print(f"    DIA warmup ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            result = dia_matrix.multiply_3d(test_vectors_gpu[i])
            cupy.cuda.Device().synchronize()  # Ensure completion

        # Clear any remaining operations before timing
        cupy.cuda.Device().synchronize()

        # Timed runs for DIA
        print(f"    DIA timing ({timing_runs} runs)...")
        dia_times = []
        for i in range(warmup_runs, warmup_runs + timing_runs):
            # Start timing - includes all memory access for matrix and vector
            start_time = time.perf_counter()
            result = dia_matrix.multiply_3d(test_vectors_gpu[i])
            cupy.cuda.Device().synchronize()  # Ensure operation completion
            end_time = time.perf_counter()
            dia_times.append((end_time - start_time) * 1000)  # Convert to ms

        # === Dense Performance Testing ===

        # Warmup runs for dense (CPU)
        print(f"    Dense CPU warmup ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            # Dense matrix multiply: (3, led_count, led_count) @ (3, led_count, 1) -> (3, led_count, 1)
            vec_reshaped = test_vectors_cpu[i][:, :, np.newaxis]  # (3, led_count, 1)
            result = np.matmul(ata_inverse_dense, vec_reshaped)

        # Timed runs for dense (CPU)
        print(f"    Dense CPU timing ({timing_runs} runs)...")
        dense_cpu_times = []
        for i in range(warmup_runs, warmup_runs + timing_runs):
            # Start timing - includes all memory access for matrix and vector
            start_time = time.perf_counter()
            vec_reshaped = test_vectors_cpu[i][:, :, np.newaxis]  # (3, led_count, 1)
            result = np.matmul(ata_inverse_dense, vec_reshaped)
            end_time = time.perf_counter()
            dense_cpu_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Warmup runs for dense (GPU)
        print(f"    Dense GPU warmup ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            vec_reshaped = test_vectors_gpu[i][:, :, cupy.newaxis]  # (3, led_count, 1)
            result = cupy.matmul(ata_inverse_dense_gpu, vec_reshaped)
            cupy.cuda.Device().synchronize()

        # Clear any remaining operations before timing
        cupy.cuda.Device().synchronize()

        # Timed runs for dense (GPU)
        print(f"    Dense GPU timing ({timing_runs} runs)...")
        dense_gpu_times = []
        for i in range(warmup_runs, warmup_runs + timing_runs):
            # Start timing - includes all memory access for matrix and vector
            start_time = time.perf_counter()
            vec_reshaped = test_vectors_gpu[i][:, :, cupy.newaxis]  # (3, led_count, 1)
            result = cupy.matmul(ata_inverse_dense_gpu, vec_reshaped)
            cupy.cuda.Device().synchronize()  # Ensure operation completion
            end_time = time.perf_counter()
            dense_gpu_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate averages and statistics
        dia_avg_ms = np.mean(dia_times)
        dia_std_ms = np.std(dia_times)

        dense_cpu_avg_ms = np.mean(dense_cpu_times)
        dense_cpu_std_ms = np.std(dense_cpu_times)

        dense_gpu_avg_ms = np.mean(dense_gpu_times)
        dense_gpu_std_ms = np.std(dense_gpu_times)

        # Use the better dense implementation for comparison
        if dense_gpu_avg_ms < dense_cpu_avg_ms:
            dense_avg_ms = dense_gpu_avg_ms
            dense_std_ms = dense_gpu_std_ms
            dense_platform = "GPU"
        else:
            dense_avg_ms = dense_cpu_avg_ms
            dense_std_ms = dense_cpu_std_ms
            dense_platform = "CPU"

        print(f"    DIA average: {dia_avg_ms:.3f} ± {dia_std_ms:.3f} ms")
        print(f"    Dense CPU average: {dense_cpu_avg_ms:.3f} ± {dense_cpu_std_ms:.3f} ms")
        print(f"    Dense GPU average: {dense_gpu_avg_ms:.3f} ± {dense_gpu_std_ms:.3f} ms")
        print(f"    Best dense platform: {dense_platform}")
        print(f"    Speedup ratio (DIA vs best dense): {dense_avg_ms/dia_avg_ms:.2f}x")

        return dia_avg_ms, dense_avg_ms

    finally:
        # Clean up all GPU memory allocations
        try:
            # Clear test vectors
            for vec in test_vectors_gpu:
                del vec
            del test_vectors_gpu

            for vec in test_vectors_cpu:
                del vec
            del test_vectors_cpu

            # Clear GPU matrix
            del ata_inverse_dense_gpu

            # Force GPU synchronization and memory cleanup
            cupy.cuda.Device().synchronize()
            cupy.get_default_memory_pool().free_all_blocks()

            # Python garbage collection
            gc.collect()

        except Exception as cleanup_error:
            print(f"    Warning: Memory cleanup error: {cleanup_error}")


def run_mse_convergence_test(
    case_name: str,
    mixed_tensor: SingleBlockMixedSparseTensor,
    dia_matrix: DiagonalATAMatrix,
    ata_inverse: DiagonalATAMatrix,
    target_frame: np.ndarray,
    max_iterations: int = 8,
) -> Tuple[np.ndarray, Dict]:
    """Run MSE convergence test for DIA ATA inverse case."""
    print(f"\n=== {case_name} ===")

    start_time = time.perf_counter()
    result = optimize_frame_led_values(
        target_frame=target_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dia_matrix,
        ata_inverse=ata_inverse,
        max_iterations=max_iterations,
        compute_error_metrics=True,
        debug=False,
        enable_timing=False,
        track_mse_per_iteration=True,
    )
    end_time = time.perf_counter()

    mse_values = result.mse_per_iteration

    print(f"  Iterations: {result.iterations}")
    print(f"  Time: {end_time - start_time:.3f}s")
    print(f"  Initial MSE: {mse_values[0]:.6f}")
    print(f"  Final MSE: {mse_values[-1]:.6f}")
    print(f"  MSE reduction: {(mse_values[0] - mse_values[-1]) / mse_values[0] * 100:.1f}%")

    if result.error_metrics:
        print(f"  Final PSNR: {result.error_metrics.get('psnr', 'N/A'):.2f} dB")

    metrics = {
        "iterations": result.iterations,
        "time": end_time - start_time,
        "initial_mse": float(mse_values[0]),
        "final_mse": float(mse_values[-1]),
        "mse_reduction_pct": float((mse_values[0] - mse_values[-1]) / mse_values[0] * 100),
        "final_psnr": result.error_metrics.get("psnr", 0.0) if result.error_metrics else 0.0,
    }

    # Clear any GPU memory used during optimization
    try:
        cupy.cuda.Device().synchronize()
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:  # nosec B110 - experimental code, safe to ignore cleanup errors
        pass

    return mse_values, metrics


def main():
    """Run MSE convergence analysis for different DIA ATA inverse diagonal factors."""
    print("MSE Convergence Analysis: DIA ATA Inverse Diagonal Factors")
    print("=" * 70)

    # Load flower image
    target_frame = load_flower_image()
    print(f"Target frame shape: {target_frame.shape}")

    # Test diagonal factors between 2.0 and 4.0 (available pattern files) - subset for faster testing
    diagonal_factors = [2.0, 2.4, 2.8, 3.0, 4.0]  # Key factors in 2.0-4.0 range

    # Results storage
    mse_results = {}
    metrics_summary = {}
    memory_analysis = {}
    timing_analysis = {}

    # Test dense ATA inverse as baseline (using factor 1.0 file)
    try:
        print(f"\n--- Loading dense baseline ---")
        mixed_tensor, dia_matrix, ata_inverse_dia, ata_inverse_dense = load_dia_ata_inverse_patterns(1.0)

        # Calculate memory usage for dense case
        dense_memory_mb = ata_inverse_dense.nbytes / (1024 * 1024)
        print(f"Dense ATA inverse memory: {dense_memory_mb:.1f} MB")
        memory_analysis["dense_baseline"] = dense_memory_mb

        # Benchmark performance (store dense timing for comparison)
        dia_time_ms, dense_time_ms = benchmark_matrix_multiply_performance(
            ata_inverse_dia, ata_inverse_dense, mixed_tensor.batch_size
        )
        timing_analysis["dense_baseline"] = {"dense_time_ms": dense_time_ms, "dia_time_ms": dia_time_ms}

        mse_values, metrics = run_mse_convergence_test(
            "Dense ATA inverse (baseline)", mixed_tensor, dia_matrix, ata_inverse_dense, target_frame
        )
        mse_results["dense_baseline"] = mse_values
        metrics_summary["dense_baseline"] = metrics

        # Clear references to save memory
        del ata_inverse_dense, ata_inverse_dia, dia_matrix, mixed_tensor

        # Force GPU memory cleanup
        cupy.cuda.Device().synchronize()
        cupy.get_default_memory_pool().free_all_blocks()
        gc.collect()

    except Exception as e:
        print(f"Skipping dense baseline: {e}")

    # Test each diagonal factor (load one at a time to save memory)
    for factor in diagonal_factors:
        try:
            print(f"\n--- Loading factor {factor} ---")
            mixed_tensor, dia_matrix, ata_inverse_dia, ata_inverse_dense = load_dia_ata_inverse_patterns(factor)

            # Calculate memory usage for DIA case
            dia_memory_mb = ata_inverse_dia.dia_data_cpu.nbytes / (1024 * 1024)
            print(f"DIA ATA inverse memory: {dia_memory_mb:.1f} MB (bandwidth={ata_inverse_dia.bandwidth})")
            memory_analysis[f"dia_factor_{factor}"] = dia_memory_mb

            # Benchmark performance
            dia_time_ms, dense_time_ms = benchmark_matrix_multiply_performance(
                ata_inverse_dia, ata_inverse_dense, mixed_tensor.batch_size
            )
            timing_analysis[f"dia_factor_{factor}"] = {"dense_time_ms": dense_time_ms, "dia_time_ms": dia_time_ms}

            mse_values, metrics = run_mse_convergence_test(
                f"DIA ATA inverse (factor {factor})", mixed_tensor, dia_matrix, ata_inverse_dia, target_frame
            )
            mse_results[f"dia_factor_{factor}"] = mse_values
            metrics_summary[f"dia_factor_{factor}"] = metrics

            # Clear references immediately after use to save memory
            del ata_inverse_dense, ata_inverse_dia, dia_matrix, mixed_tensor

            # Force GPU memory cleanup
            cupy.cuda.Device().synchronize()
            cupy.get_default_memory_pool().free_all_blocks()
            gc.collect()
            print(f"--- Completed factor {factor}, memory cleared ---")

        except Exception as e:
            print(f"Skipping factor {factor}: {e}")

            # Force GPU memory cleanup even on error
            try:
                cupy.cuda.Device().synchronize()
                cupy.get_default_memory_pool().free_all_blocks()
            except Exception:  # nosec B110 - experimental code, safe to ignore cleanup errors
                pass
            gc.collect()

    # Generate convergence plot
    if mse_results:
        print(f"\n=== Generating Convergence Plot ===")
        plt.figure(figsize=(14, 10))

        # Define colors and line styles
        colors = ["black", "blue", "green", "orange", "red", "purple", "brown"]
        line_styles = ["-", "--", "-.", ":", "-", "--", "-."]

        # Plot dense baseline first
        if "dense_baseline" in mse_results:
            mse_values = mse_results["dense_baseline"]
            iterations = np.arange(len(mse_values))
            plt.semilogy(
                iterations,
                mse_values,
                color="black",
                linewidth=3,
                linestyle="-",
                label="Dense ATA inverse (baseline)",
                marker="s",
                markersize=6,
            )

        # Plot DIA factors
        factor_cases = [key for key in mse_results.keys() if key.startswith("dia_factor_")]
        factor_cases.sort(key=lambda x: float(x.split("_")[-1]))  # Sort by factor value

        for i, case in enumerate(factor_cases):
            if case in mse_results:
                factor = case.split("_")[-1]
                mse_values = mse_results[case]
                iterations = np.arange(len(mse_values))

                color_idx = (i + 1) % len(colors)
                style_idx = (i + 1) % len(line_styles)

                plt.semilogy(
                    iterations,
                    mse_values,
                    color=colors[color_idx],
                    linewidth=2,
                    linestyle=line_styles[style_idx],
                    label=f"DIA factor {factor}",
                    marker="o",
                    markersize=4,
                )

        plt.xlabel("Iteration")
        plt.ylabel("MSE (log scale)")
        plt.title("MSE Convergence Analysis: DIA ATA Inverse Diagonal Factors\nFlower Image with uint8 A-matrix")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = "mse_convergence_dia_ata_inverse.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Convergence plot saved to: {plot_path}")

        # Show plot
        plt.show()

    # Print summary table
    print(f"\n=== Summary ===")
    print("Case                              | Factor | Init MSE  | Final MSE | Reduction | Time (s) | PSNR (dB)")
    print("-" * 100)

    # Dense baseline first
    if "dense_baseline" in metrics_summary:
        metrics = metrics_summary["dense_baseline"]
        print(
            f"{'Dense ATA inverse (baseline)':33} | {'N/A':6} | {metrics['initial_mse']:8.6f} | {metrics['final_mse']:8.6f} | "
            f"{metrics['mse_reduction_pct']:6.1f}%  | {metrics['time']:7.3f} | {metrics['final_psnr']:8.2f}"
        )

    # DIA factors
    factor_cases = [key for key in metrics_summary.keys() if key.startswith("dia_factor_")]
    factor_cases.sort(key=lambda x: float(x.split("_")[-1]))

    for case in factor_cases:
        if case in metrics_summary:
            factor = case.split("_")[-1]
            metrics = metrics_summary[case]
            name = f"DIA ATA inverse (factor {factor})"
            print(
                f"{name:33} | {factor:6} | {metrics['initial_mse']:8.6f} | {metrics['final_mse']:8.6f} | "
                f"{metrics['mse_reduction_pct']:6.1f}%  | {metrics['time']:7.3f} | {metrics['final_psnr']:8.2f}"
            )

    # Memory analysis
    print(f"\n=== Memory Analysis ===")
    if memory_analysis:
        print("Memory Usage Comparison:")
        print("Case                              | Memory (MB) | vs Dense")
        print("-" * 65)

        dense_memory = memory_analysis.get("dense_baseline", 0)
        if dense_memory > 0:
            print(f"{'Dense ATA inverse (baseline)':33} | {dense_memory:10.1f} | {'1.0x':>8}")

        # DIA factors
        factor_cases = [key for key in memory_analysis.keys() if key.startswith("dia_factor_")]
        factor_cases.sort(key=lambda x: float(x.split("_")[-1]))

        for case in factor_cases:
            if case in memory_analysis:
                factor = case.split("_")[-1]
                dia_memory = memory_analysis[case]
                memory_ratio = dia_memory / dense_memory if dense_memory > 0 else 0
                name = f"DIA ATA inverse (factor {factor})"
                print(f"{name:33} | {dia_memory:10.1f} | {memory_ratio:7.2f}x")

        print(f"\nMemory savings with DIA format:")
        if dense_memory > 0:
            for case in factor_cases:
                if case in memory_analysis:
                    factor = case.split("_")[-1]
                    dia_memory = memory_analysis[case]
                    savings_pct = (1 - dia_memory / dense_memory) * 100
                    print(f"  Factor {factor}: {savings_pct:.1f}% memory reduction")

    # Performance timing analysis
    print(f"\n=== Performance Timing Analysis ===")
    if timing_analysis:
        print("Matrix Multiply Performance (averaged over multiple runs):")
        print("Case                              | DIA (ms) | Dense (ms) | Speedup")
        print("-" * 70)

        # Dense baseline first
        if "dense_baseline" in timing_analysis:
            timing = timing_analysis["dense_baseline"]
            dia_time = timing["dia_time_ms"]
            dense_time = timing["dense_time_ms"]
            speedup = dense_time / dia_time
            print(f"{'Dense ATA inverse (baseline)':33} | {dia_time:7.2f} | {dense_time:9.2f} | {speedup:6.2f}x")

        # DIA factors
        factor_cases = [key for key in timing_analysis.keys() if key.startswith("dia_factor_")]
        factor_cases.sort(key=lambda x: float(x.split("_")[-1]))

        for case in factor_cases:
            if case in timing_analysis:
                factor = case.split("_")[-1]
                timing = timing_analysis[case]
                dia_time = timing["dia_time_ms"]
                dense_time = timing["dense_time_ms"]
                speedup = dense_time / dia_time
                name = f"DIA ATA inverse (factor {factor})"
                print(f"{name:33} | {dia_time:7.2f} | {dense_time:9.2f} | {speedup:6.2f}x")

        print(f"\nTiming insights:")
        if len(factor_cases) >= 1:
            # Find fastest DIA case
            dia_times = [timing_analysis[case]["dia_time_ms"] for case in factor_cases if case in timing_analysis]
            factors = [float(case.split("_")[-1]) for case in factor_cases if case in timing_analysis]

            if dia_times:
                fastest_idx = np.argmin(dia_times)
                fastest_factor = factors[fastest_idx]
                fastest_time = dia_times[fastest_idx]
                print(f"  Fastest DIA factor: {fastest_factor} ({fastest_time:.2f} ms)")

                if "dense_baseline" in timing_analysis:
                    baseline_dense_time = timing_analysis["dense_baseline"]["dense_time_ms"]
                    speedup_vs_dense = baseline_dense_time / fastest_time
                    print(f"  Best DIA speedup vs dense: {speedup_vs_dense:.2f}x")

    # Analysis insights
    print(f"\n=== Analysis Insights ===")
    if len(factor_cases) >= 2:
        # Compare performance vs diagonal factor
        factors = [float(case.split("_")[-1]) for case in factor_cases if case in metrics_summary]
        final_mses = [metrics_summary[f"dia_factor_{factor}"]["final_mse"] for factor in factors]
        final_psnrs = [metrics_summary[f"dia_factor_{factor}"]["final_psnr"] for factor in factors]

        best_factor_idx = np.argmin(final_mses)
        best_factor = factors[best_factor_idx]
        best_mse = final_mses[best_factor_idx]
        best_psnr = final_psnrs[best_factor_idx]

        print(f"  Best performing factor: {best_factor} (MSE: {best_mse:.6f}, PSNR: {best_psnr:.2f} dB)")

        if "dense_baseline" in metrics_summary:
            dense_mse = metrics_summary["dense_baseline"]["final_mse"]
            dense_psnr = metrics_summary["dense_baseline"]["final_psnr"]
            mse_ratio = best_mse / dense_mse
            psnr_diff = best_psnr - dense_psnr

            print(f"  vs Dense baseline: MSE ratio {mse_ratio:.2f}x, PSNR diff {psnr_diff:+.2f} dB")

            if mse_ratio < 2.0:
                print(f"  ✅ DIA approximation is quite good (within 2x MSE of dense)")
            elif mse_ratio < 5.0:
                print(f"  ⚠️  DIA approximation is reasonable (within 5x MSE of dense)")
            else:
                print(f"  ❌ DIA approximation may be too coarse (>{mse_ratio:.1f}x MSE of dense)")


if __name__ == "__main__":
    main()
