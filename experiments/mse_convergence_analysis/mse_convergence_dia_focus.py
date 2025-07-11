#!/usr/bin/env python3
"""
Focused MSE Convergence Analysis for DIA ATA Inverse Matrices

Analyzes MSE convergence for different DIA diagonal factors:
- 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 10.0

Uses the flower image for testing with shorter iterations for faster analysis.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_flower_image() -> np.ndarray:
    """Load and prepare the flower image for testing."""
    flower_path = Path("images/source/flower")

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
) -> Tuple[SingleBlockMixedSparseTensor, DiagonalATAMatrix, DiagonalATAMatrix]:
    """Load patterns with DIA format ATA inverse for specified diagonal factor."""
    pattern_file = f"diffusion_patterns/synthetic_2624_uint8_dia_{diagonal_factor}.npz"
    pattern_path = Path(pattern_file)

    if not pattern_path.exists():
        raise FileNotFoundError(f"DIA pattern file not found: {pattern_file}")

    print(f"Loading DIA patterns (factor {diagonal_factor}) from: {pattern_path}")
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

    print(f"  LED count: {mixed_tensor.batch_size}")
    print(f"  Mixed tensor: {mixed_tensor.dtype}")
    print(f"  DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k}")
    print(f"  DIA ATA inverse: bandwidth={ata_inverse_dia.bandwidth}, k={ata_inverse_dia.k}")

    return mixed_tensor, dia_matrix, ata_inverse_dia


def run_mse_convergence_test(
    case_name: str,
    mixed_tensor: SingleBlockMixedSparseTensor,
    dia_matrix: DiagonalATAMatrix,
    ata_inverse_dia: DiagonalATAMatrix,
    target_frame: np.ndarray,
    max_iterations: int = 5,
) -> Tuple[np.ndarray, Dict]:
    """Run MSE convergence test for a specific case."""
    print(f"\n=== {case_name} ===")

    start_time = time.perf_counter()
    result = optimize_frame_led_values(
        target_frame=target_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dia_matrix,
        ata_inverse=ata_inverse_dia,
        initial_values=None,  # Use ATA inverse initialization
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
    """Run focused MSE convergence analysis for DIA ATA inverse matrices."""
    print("Focused MSE Convergence Analysis: DIA ATA Inverse Matrices")
    print("=" * 60)

    # Load flower image
    target_frame = load_flower_image()
    print(f"Target frame shape: {target_frame.shape}")

    # Test DIA factors
    dia_factors = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 10.0]

    # Check which factors are available
    available_factors = []
    for factor in dia_factors:
        pattern_file = f"diffusion_patterns/synthetic_2624_uint8_dia_{factor}.npz"
        if Path(pattern_file).exists():
            available_factors.append(factor)
        else:
            print(f"Missing DIA factor {factor} file: {pattern_file}")

    print(f"Available DIA factors: {available_factors}")

    mse_results = {}
    metrics_summary = {}

    # Run tests for each available factor
    for factor in available_factors:
        try:
            mixed_tensor, dia_matrix, ata_inverse_dia = load_dia_ata_inverse_patterns(factor)
            case_key = f"dia_factor_{factor}"
            case_name = f"DIA ATA inverse (factor {factor})"

            mse_values, metrics = run_mse_convergence_test(
                case_name,
                mixed_tensor,
                dia_matrix,
                ata_inverse_dia,
                target_frame,
                max_iterations=5,  # Shorter for faster analysis
            )
            mse_results[case_key] = mse_values
            metrics_summary[case_key] = metrics

        except Exception as e:
            print(f"Error with DIA factor {factor}: {e}")

    # Generate convergence plot
    if mse_results:
        print(f"\n=== Generating Convergence Plot ===")
        plt.figure(figsize=(12, 8))

        colors = ["blue", "green", "red", "purple", "orange", "brown", "pink"]

        for i, (case_key, mse_values) in enumerate(mse_results.items()):
            factor = float(case_key.split("_")[-1])
            iterations = np.arange(len(mse_values))

            # Add approximation error info to label
            if factor == 10.0:
                label = f"DIA factor {factor} (Perfect)"
            elif factor == 2.0:
                label = f"DIA factor {factor} (~1.3% error)"
            elif factor == 1.2:
                label = f"DIA factor {factor} (~5.0% error)"
            elif factor == 1.0:
                label = f"DIA factor {factor} (~7.1% error)"
            else:
                label = f"DIA factor {factor}"

            plt.semilogy(
                iterations,
                mse_values,
                color=colors[i % len(colors)],
                linewidth=2,
                label=label,
                marker="o",
                markersize=5,
            )

        plt.xlabel("Iteration")
        plt.ylabel("MSE (log scale)")
        plt.title(
            "MSE Convergence Analysis: DIA ATA Inverse Diagonal Factors\n"
            "Flower Image with uint8 A-matrix (2624 LEDs)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = "mse_convergence_dia_ata_inverse_updated.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Convergence plot saved to: {plot_path}")

        # Show plot
        plt.show()

    # Print summary table
    print(f"\n=== Summary Table ===")
    print("Factor | Bandwidth | K     | Memory Ratio | Init MSE  | Final MSE | Reduction | PSNR (dB)")
    print("-" * 85)

    for factor in available_factors:
        case_key = f"dia_factor_{factor}"
        if case_key in metrics_summary:
            metrics = metrics_summary[case_key]

            # Load pattern info
            try:
                pattern_file = f"diffusion_patterns/synthetic_2624_uint8_dia_{factor}.npz"
                data = np.load(pattern_file, allow_pickle=True)
                ata_inv_dict = data["ata_inverse_dia"].item()

                bandwidth = ata_inv_dict.get("bandwidth", "N/A")
                k = ata_inv_dict.get("k", "N/A")

                # Estimate memory ratio (vs dense)
                if k != "N/A":
                    dense_elements = 3 * 2624 * 2624  # 3 channels, 2624x2624 each
                    dia_elements = 3 * k * 2624
                    memory_ratio = dia_elements / dense_elements
                else:
                    memory_ratio = float("nan")

                print(
                    f"{factor:6.1f} | {bandwidth:9} | {k:5} | {memory_ratio:12.4f} | "
                    f"{metrics['initial_mse']:8.6f} | {metrics['final_mse']:8.6f} | "
                    f"{metrics['mse_reduction_pct']:6.1f}% | {metrics['final_psnr']:8.2f}"
                )

            except Exception as e:
                print(
                    f"{factor:6.1f} | {'Error':>9} | {'N/A':>5} | {'N/A':>12} | "
                    f"{metrics['initial_mse']:8.6f} | {metrics['final_mse']:8.6f} | "
                    f"{metrics['mse_reduction_pct']:6.1f}% | {metrics['final_psnr']:8.2f}"
                )

    print(f"\n=== Analysis Complete ===")
    print("Key findings:")
    print("- Memory usage decreases significantly with lower diagonal factors")
    print("- Convergence behavior shows the impact of ATA inverse approximation quality")
    print("- DIA factor 10.0 represents the 'perfect' case with full diagonal coverage")


if __name__ == "__main__":
    main()
