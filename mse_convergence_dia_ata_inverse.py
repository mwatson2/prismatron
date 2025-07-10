#!/usr/bin/env python3
"""
MSE Convergence Analysis for DIA ATA Inverse Matrices

Analyzes MSE convergence for different DIA ATA inverse diagonal factors:
- Diagonal factors: 1.0, 1.2, 1.4, 1.6, 1.8, 2.0
- Compares DIA ATA inverse vs dense ATA inverse performance
- Tests with uint8 A-matrix (available diagonal factor files)

The diagonal factor determines how many diagonals to keep in the ATA inverse
as a fraction of the number of diagonals in the original ATA matrix.
"""

# Add src to path
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

    return mse_values, metrics


def main():
    """Run MSE convergence analysis for different DIA ATA inverse diagonal factors."""
    print("MSE Convergence Analysis: DIA ATA Inverse Diagonal Factors")
    print("=" * 70)

    # Load flower image
    target_frame = load_flower_image()
    print(f"Target frame shape: {target_frame.shape}")

    # Test diagonal factors available (reduced set for speed)
    diagonal_factors = [1.0, 1.4, 2.0]

    # Results storage
    mse_results = {}
    metrics_summary = {}

    # Test dense ATA inverse as baseline (using factor 1.0 file)
    try:
        mixed_tensor, dia_matrix, ata_inverse_dia, ata_inverse_dense = load_dia_ata_inverse_patterns(1.0)

        mse_values, metrics = run_mse_convergence_test(
            "Dense ATA inverse (baseline)", mixed_tensor, dia_matrix, ata_inverse_dense, target_frame
        )
        mse_results["dense_baseline"] = mse_values
        metrics_summary["dense_baseline"] = metrics

    except Exception as e:
        print(f"Skipping dense baseline: {e}")

    # Test each diagonal factor
    for factor in diagonal_factors:
        try:
            mixed_tensor, dia_matrix, ata_inverse_dia, ata_inverse_dense = load_dia_ata_inverse_patterns(factor)

            mse_values, metrics = run_mse_convergence_test(
                f"DIA ATA inverse (factor {factor})", mixed_tensor, dia_matrix, ata_inverse_dia, target_frame
            )
            mse_results[f"dia_factor_{factor}"] = mse_values
            metrics_summary[f"dia_factor_{factor}"] = metrics

        except Exception as e:
            print(f"Skipping factor {factor}: {e}")

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
