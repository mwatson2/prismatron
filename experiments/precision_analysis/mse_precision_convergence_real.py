#!/usr/bin/env python3

import sys

sys.path.append("/mnt/dev/prismatron/src")

import logging
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pattern_data(pattern_path: str):
    """Load pattern data from file."""
    logger.info(f"Loading pattern data from {pattern_path}")

    data = np.load(pattern_path, allow_pickle=True)

    # Load mixed tensor (A^T matrix)
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())

    # Load DIA matrix (A^T A matrix)
    dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())

    # Load ATA inverse
    ata_inverse = None
    if "ata_inverse" in data:
        ata_inverse = data["ata_inverse"]
        logger.info("Using dense format ATA inverse")

    return mixed_tensor, dia_matrix, ata_inverse


def create_test_frame(height: int = 480, width: int = 800) -> np.ndarray:
    """Create a simple test frame for consistent testing."""
    # Create a gradient pattern
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Red gradient from left to right
    frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)

    # Green gradient from top to bottom
    frame[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)

    # Blue checkerboard pattern
    checker_size = 32
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    checker_pattern = ((y_indices // checker_size) + (x_indices // checker_size)) % 2
    frame[:, :, 2] = checker_pattern * 255

    return frame


def run_convergence_test(mixed_tensor, dia_matrix, ata_inverse, test_name, max_iterations=20):
    """Run convergence test and track MSE over iterations."""
    logger.info(f"Running {test_name}")

    # Create test frame
    test_frame = create_test_frame()

    try:
        # Run optimization with full iterations and track MSE per iteration
        result = optimize_frame_led_values(
            target_frame=test_frame,
            at_matrix=mixed_tensor,
            ata_matrix=dia_matrix,
            ata_inverse=ata_inverse,
            initial_values=cp.full((3, mixed_tensor.batch_size), 0.5, dtype=cp.float32),  # Fixed initialization
            max_iterations=max_iterations,
            track_mse_per_iteration=True,
            debug=False,
        )

        # Get MSE history from result
        if hasattr(result, "mse_per_iteration") and result.mse_per_iteration is not None:
            mse_history = result.mse_per_iteration.tolist()
            logger.info(f"{test_name} MSE progression: {mse_history}")
        else:
            logger.warning(f"No MSE per iteration data for {test_name}")
            mse_history = []

        # Clear GPU memory
        cp.get_default_memory_pool().free_all_blocks()

    except Exception as e:
        logger.error(f"Error in {test_name}: {e}")
        mse_history = []

    return mse_history


def main():
    """Run MSE convergence comparison for different precision combinations."""

    # Test configurations using different pattern files for different dtypes
    configs = [
        ("fp32 A + fp32 ATA", "/mnt/dev/prismatron/diffusion_patterns/synthetic_2624_fp32.npz"),
        ("uint8 A + fp32 ATA", "/mnt/dev/prismatron/diffusion_patterns/synthetic_2624_uint8.npz"),
        ("fp16 A + fp16 ATA", "/mnt/dev/prismatron/diffusion_patterns/synthetic_2624_fp16.npz"),
    ]

    results = {}

    for test_name, pattern_path in configs:
        if not Path(pattern_path).exists():
            logger.error(f"Pattern file not found: {pattern_path}")
            continue

        try:
            # Load pattern data for this specific dtype
            mixed_tensor, dia_matrix, ata_inverse = load_pattern_data(pattern_path)

            mse_history = run_convergence_test(
                mixed_tensor, dia_matrix, ata_inverse, test_name, max_iterations=10  # Fewer iterations for memory
            )
            results[test_name] = mse_history

        except Exception as e:
            logger.error(f"Error with {test_name}: {e}")
            results[test_name] = None

    # Plot results
    plt.figure(figsize=(12, 8))

    colors = ["blue", "red", "green", "orange"]
    linestyles = ["-", "--", "-.", ":"]

    for i, (test_name, mse_history) in enumerate(results.items()):
        if mse_history is not None and len(mse_history) > 0:
            iterations = range(1, len(mse_history) + 1)
            plt.plot(
                iterations,
                mse_history,
                color=colors[i],
                linestyle=linestyles[i],
                marker="o",
                markersize=4,
                linewidth=2,
                label=test_name,
            )

    plt.xlabel("LSQR Iterations")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE Convergence: Matrix Precision Effects (Frame Optimizer)\n(Fixed Initialization = 0.5)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # Save plot
    output_file = "mse_precision_convergence_real.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    logger.info(f"Plot saved as: {output_file}")

    # Print summary
    print("\nFinal MSE Values:")
    for test_name, mse_history in results.items():
        if mse_history is not None and len(mse_history) > 0:
            final_mse = mse_history[-1]
            initial_mse = mse_history[0]
            print(f"  {test_name:20s}: Initial = {initial_mse:.6f}, Final = {final_mse:.6f}")
            if final_mse > initial_mse:
                print(f"    WARNING: MSE increased by {(final_mse/initial_mse - 1)*100:.1f}%")
            else:
                print(f"    MSE decreased by {(1 - final_mse/initial_mse)*100:.1f}%")


if __name__ == "__main__":
    main()
