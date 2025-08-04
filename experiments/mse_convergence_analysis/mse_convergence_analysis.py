#!/usr/bin/env python3
"""
MSE Convergence Analysis Script

Analyzes MSE convergence for different precision combinations:
1. fp16 ATA + fp16 ATA inverse with fp32 A-matrix + ATA-inverse initialization
2. fp32 ATA + fp32 ATA inverse with fp32 A-matrix + ATA-inverse initialization
3. fp32 A-matrix without ATA-inverse initialization (random init)
4. uint8 A-matrix with fp32 ATA + fp32 ATA inverse + ATA-inverse initialization

Uses the flower image from source/images/flower for testing.
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


def load_patterns_and_ata_inverse(precision: str) -> Tuple[SingleBlockMixedSparseTensor, DiagonalATAMatrix, np.ndarray]:
    """Load diffusion patterns and ATA inverse for specified precision."""
    # Try to load patterns with ATA inverse
    pattern_files = [
        f"diffusion_patterns/synthetic_2624_{precision}.npz",
        f"diffusion_patterns/synthetic_1000_{precision}.npz",
        f"diffusion_patterns/synthetic_500_{precision}.npz",
    ]

    for pattern_file in pattern_files:
        pattern_path = Path(pattern_file)
        if pattern_path.exists():
            print(f"Loading {precision} patterns from: {pattern_path}")
            data = np.load(str(pattern_path), allow_pickle=True)

            # Load mixed tensor
            mixed_tensor_dict = data["mixed_tensor"].item()
            mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

            # Load DIA matrix
            if "dia_matrix" in data:
                dia_dict = data["dia_matrix"].item()
                dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)
            else:
                raise ValueError(f"No DIA matrix found in {pattern_file}")

            # Load ATA inverse
            if "ata_inverse" in data:
                ata_inverse = data["ata_inverse"]
                print(f"  ATA inverse: shape={ata_inverse.shape}, dtype={ata_inverse.dtype}")
            else:
                raise ValueError(f"No ATA inverse found in {pattern_file}")

            print(f"  LED count: {mixed_tensor.batch_size}")
            print(f"  Mixed tensor: {mixed_tensor.dtype}")
            print(f"  DIA matrix: {dia_matrix.storage_dtype}")

            return mixed_tensor, dia_matrix, ata_inverse

    raise FileNotFoundError(f"No suitable {precision} diffusion patterns found")


def run_mse_convergence_test(
    case_name: str,
    mixed_tensor: SingleBlockMixedSparseTensor,
    dia_matrix: DiagonalATAMatrix,
    ata_inverse: np.ndarray,
    target_frame: np.ndarray,
    use_ata_inverse_init: bool = True,
    max_iterations: int = 10,
) -> Tuple[np.ndarray, Dict]:
    """Run MSE convergence test for a specific case."""
    print(f"\n=== {case_name} ===")

    initial_values = (
        None if use_ata_inverse_init else np.random.rand(3, mixed_tensor.batch_size).astype(np.float32) * 0.1
    )

    start_time = time.perf_counter()
    result = optimize_frame_led_values(
        target_frame=target_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dia_matrix,
        ata_inverse=ata_inverse,
        initial_values=initial_values,
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
    """Run MSE convergence analysis for all 4 cases."""
    print("MSE Convergence Analysis")
    print("=" * 50)

    # Load flower image
    target_frame = load_flower_image()
    print(f"Target frame shape: {target_frame.shape}")

    # Test cases
    cases = []
    mse_results = {}
    metrics_summary = {}

    # Case 1: fp16 ATA + fp16 ATA inverse with fp32 A-matrix + ATA-inverse initialization
    try:
        mixed_tensor_fp16, dia_matrix_fp16, ata_inverse_fp16 = load_patterns_and_ata_inverse("fp16")
        mse_values, metrics = run_mse_convergence_test(
            "Case 1: fp16 ATA + fp16 ATA inverse + ATA-inverse init",
            mixed_tensor_fp16,
            dia_matrix_fp16,
            ata_inverse_fp16,
            target_frame,
            use_ata_inverse_init=True,
        )
        mse_results["fp16_with_ata_init"] = mse_values
        metrics_summary["fp16_with_ata_init"] = metrics
        cases.append("fp16_with_ata_init")
    except Exception as e:
        print(f"Skipping fp16 case: {e}")

    # Case 2: fp32 ATA + fp32 ATA inverse with fp32 A-matrix + ATA-inverse initialization
    try:
        mixed_tensor_fp32, dia_matrix_fp32, ata_inverse_fp32 = load_patterns_and_ata_inverse("fp32")
        mse_values, metrics = run_mse_convergence_test(
            "Case 2: fp32 ATA + fp32 ATA inverse + ATA-inverse init",
            mixed_tensor_fp32,
            dia_matrix_fp32,
            ata_inverse_fp32,
            target_frame,
            use_ata_inverse_init=True,
        )
        mse_results["fp32_with_ata_init"] = mse_values
        metrics_summary["fp32_with_ata_init"] = metrics
        cases.append("fp32_with_ata_init")
    except Exception as e:
        print(f"Skipping fp32 case: {e}")

    # Case 3: fp32 A-matrix without ATA-inverse initialization (random init)
    if "fp32_with_ata_init" in mse_results:
        mse_values, metrics = run_mse_convergence_test(
            "Case 3: fp32 A-matrix without ATA-inverse init (random)",
            mixed_tensor_fp32,
            dia_matrix_fp32,
            ata_inverse_fp32,
            target_frame,
            use_ata_inverse_init=False,
        )
        mse_results["fp32_random_init"] = mse_values
        metrics_summary["fp32_random_init"] = metrics
        cases.append("fp32_random_init")

    # Case 4: uint8 A-matrix with fp32 ATA + ATA-inverse initialization
    try:
        mixed_tensor_uint8, dia_matrix_uint8, ata_inverse_uint8 = load_patterns_and_ata_inverse("uint8")
        mse_values, metrics = run_mse_convergence_test(
            "Case 4: uint8 A-matrix + fp32 ATA + ATA-inverse init",
            mixed_tensor_uint8,
            dia_matrix_uint8,
            ata_inverse_uint8,
            target_frame,
            use_ata_inverse_init=True,
        )
        mse_results["uint8_with_ata_init"] = mse_values
        metrics_summary["uint8_with_ata_init"] = metrics
        cases.append("uint8_with_ata_init")
    except Exception as e:
        print(f"Skipping uint8 case: {e}")

    # Generate convergence plot
    if mse_results:
        print("\n=== Generating Convergence Plot ===")
        plt.figure(figsize=(12, 8))

        colors = ["blue", "red", "green", "orange", "purple"]
        labels = {
            "fp16_with_ata_init": "FP16 ATA + ATA-inverse init",
            "fp32_with_ata_init": "FP32 ATA + ATA-inverse init",
            "fp32_random_init": "FP32 ATA + Random init",
            "uint8_with_ata_init": "uint8 A-matrix + FP32 ATA + ATA-inverse init",
        }

        for i, case in enumerate(cases):
            if case in mse_results:
                mse_values = mse_results[case]
                iterations = np.arange(len(mse_values))
                plt.semilogy(
                    iterations,
                    mse_values,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    label=labels.get(case, case),
                    marker="o",
                    markersize=4,
                )

        plt.xlabel("Iteration")
        plt.ylabel("MSE (log scale)")
        plt.title("MSE Convergence Analysis: Flower Image\nDifferent Precision and Initialization Methods")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = "mse_convergence_analysis_flower.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Convergence plot saved to: {plot_path}")

        # Show plot
        plt.show()

    # Print summary table
    print("\n=== Summary ===")
    print("Case                              | Init MSE  | Final MSE | Reduction | Time (s) | PSNR (dB)")
    print("-" * 85)

    case_names = {
        "fp16_with_ata_init": "FP16 + ATA-inverse init",
        "fp32_with_ata_init": "FP32 + ATA-inverse init",
        "fp32_random_init": "FP32 + Random init",
        "uint8_with_ata_init": "uint8 A-matrix + FP32 ATA + ATA-inverse",
    }

    for case in cases:
        if case in metrics_summary and case in case_names:
            metrics = metrics_summary[case]
            name = case_names[case]
            print(
                f"{name:33} | {metrics['initial_mse']:8.6f} | {metrics['final_mse']:8.6f} | "
                f"{metrics['mse_reduction_pct']:6.1f}%  | {metrics['time']:7.3f} | {metrics['final_psnr']:8.2f}"
            )


if __name__ == "__main__":
    main()
