#!/usr/bin/env python3
"""
Test program to analyze MSE convergence behavior comparing different ATA inverse formats.

This program loads different compressed DIA ATA inverse matrices and compares
how many additional iterations they require to reach the same MSE as the
uncompressed ATA inverse.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import load_ata_inverse_from_pattern, optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pattern_data(
    pattern_path: str, use_dia_ata_inverse: bool = False
) -> Tuple[SingleBlockMixedSparseTensor, DiagonalATAMatrix, Dict]:
    """Load mixed tensor, DIA matrix, and ATA inverse from pattern file."""
    logger.info(f"Loading pattern data from {pattern_path} (use_dia_ata_inverse={use_dia_ata_inverse})")

    data = np.load(pattern_path, allow_pickle=True)

    # Load mixed tensor (A^T matrix)
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())

    # Load DIA matrix (A^T A matrix)
    dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())

    # Load ATA inverse - choose format based on parameter
    ata_inverse = None
    if use_dia_ata_inverse and "ata_inverse_dia" in data:
        # Load the unified DIA format and convert to the format expected by frame optimizer
        ata_inverse_dia_dict = data["ata_inverse_dia"].item()
        ata_inverse_dia_matrix = DiagonalATAMatrix.from_dict(ata_inverse_dia_dict)
        ata_inverse = ata_inverse_dia_matrix  # Pass the DiagonalATAMatrix object directly
        logger.info("Using DIA format ATA inverse")
    elif "ata_inverse" in data:
        ata_inverse = data["ata_inverse"]
        logger.info("Using dense format ATA inverse")
    else:
        raise ValueError("No suitable ATA inverse found")

    return mixed_tensor, dia_matrix, ata_inverse


def create_test_frame(height: int = 480, width: int = 800) -> np.ndarray:
    """Create a simple test frame for consistent testing."""
    # Create a gradient pattern that's challenging to optimize
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Red gradient from left to right
    frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)

    # Green gradient from top to bottom
    frame[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)

    # Blue checkerboard pattern
    checker_size = 64
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    checker_pattern = ((y_indices // checker_size) + (x_indices // checker_size)) % 2
    frame[:, :, 2] = checker_pattern * 255

    return frame


def run_mse_convergence_test(max_iterations: int = 5) -> Dict[str, List[float]]:
    """Run MSE convergence test comparing different ATA inverse formats."""

    # Get pattern files
    pattern_dir = Path("diffusion_patterns")

    # Base pattern (uncompressed)
    base_pattern = pattern_dir / "synthetic_2624_uint8.npz"

    # DIA compressed patterns (test all factors)
    dia_patterns = [
        ("DIA 1.0", pattern_dir / "synthetic_2624_uint8_dia_1.0.npz"),
        ("DIA 1.2", pattern_dir / "synthetic_2624_uint8_dia_1.2.npz"),
        ("DIA 1.4", pattern_dir / "synthetic_2624_uint8_dia_1.4.npz"),
        ("DIA 1.6", pattern_dir / "synthetic_2624_uint8_dia_1.6.npz"),
        ("DIA 1.8", pattern_dir / "synthetic_2624_uint8_dia_1.8.npz"),
        ("DIA 2.0", pattern_dir / "synthetic_2624_uint8_dia_2.0.npz"),
    ]

    # Create test frame
    test_frame = create_test_frame()
    logger.info(f"Created test frame with shape {test_frame.shape}")

    # Store MSE results for each pattern
    mse_results = {}

    # Test base pattern (uncompressed ATA inverse)
    logger.info("Testing base pattern (uncompressed ATA inverse)")
    try:
        mixed_tensor, dia_matrix, ata_inverse = load_pattern_data(str(base_pattern), use_dia_ata_inverse=False)

        result = optimize_frame_led_values(
            target_frame=test_frame,
            at_matrix=mixed_tensor,
            ata_matrix=dia_matrix,
            ata_inverse=ata_inverse,
            max_iterations=max_iterations,
            track_mse_per_iteration=True,
            debug=False,
        )

        mse_results["Uncompressed"] = result.mse_per_iteration.tolist()
        logger.info(f"Uncompressed MSE progression: {result.mse_per_iteration}")

    except Exception as e:
        logger.error(f"Error testing base pattern: {e}")
        return {}

    # Test each DIA compressed pattern
    for pattern_name, pattern_path in dia_patterns:
        if not pattern_path.exists():
            logger.warning(f"Pattern file not found: {pattern_path}")
            continue

        logger.info(f"Testing {pattern_name}")
        try:
            mixed_tensor, dia_matrix, ata_inverse = load_pattern_data(str(pattern_path), use_dia_ata_inverse=True)

            result = optimize_frame_led_values(
                target_frame=test_frame,
                at_matrix=mixed_tensor,
                ata_matrix=dia_matrix,
                ata_inverse=ata_inverse,
                max_iterations=max_iterations,
                track_mse_per_iteration=True,
                debug=False,
            )

            mse_results[pattern_name] = result.mse_per_iteration.tolist()
            logger.info(f"{pattern_name} MSE progression: {result.mse_per_iteration}")

        except Exception as e:
            logger.error(f"Error testing {pattern_name}: {e}")
            continue

    return mse_results


def analyze_convergence_cost(mse_results: Dict[str, List[float]]) -> None:
    """Analyze how many additional iterations each compressed format requires."""

    if "Uncompressed" not in mse_results:
        logger.error("No uncompressed baseline found")
        return

    uncompressed_mse = mse_results["Uncompressed"]
    final_uncompressed_mse = uncompressed_mse[-1]

    logger.info(f"\nConvergence Analysis:")
    logger.info(f"Uncompressed final MSE: {final_uncompressed_mse:.6f}")
    logger.info(f"Uncompressed MSE progression: {uncompressed_mse}")

    for pattern_name, mse_values in mse_results.items():
        if pattern_name == "Uncompressed":
            continue

        # Find how many iterations it takes to reach the same final MSE
        iterations_needed = len(mse_values)
        achieved_final_mse = mse_values[-1]

        # Find the iteration where we first get close to the uncompressed final MSE
        target_mse = final_uncompressed_mse
        iterations_to_target = iterations_needed

        for i, mse in enumerate(mse_values):
            if mse <= target_mse * 1.01:  # Within 1% of target
                iterations_to_target = i + 1
                break

        additional_iterations = max(0, iterations_to_target - len(uncompressed_mse))

        logger.info(f"\n{pattern_name}:")
        logger.info(f"  Final MSE: {achieved_final_mse:.6f}")
        logger.info(f"  MSE progression: {mse_values}")
        logger.info(f"  Iterations to reach target MSE: {iterations_to_target}")
        logger.info(f"  Additional iterations vs uncompressed: {additional_iterations}")


def plot_mse_convergence(mse_results: Dict[str, List[float]]) -> None:
    """Plot MSE convergence curves for visual analysis."""

    plt.figure(figsize=(12, 8))

    for pattern_name, mse_values in mse_results.items():
        iterations = list(range(len(mse_values)))
        plt.plot(iterations, mse_values, marker="o", label=pattern_name, linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("MSE Convergence Comparison: Uncompressed vs DIA Compressed ATA Inverse")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # Log scale for better visualization

    # Save plot
    output_file = "mse_convergence_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {output_file}")

    plt.show()


def main():
    """Main function to run the MSE convergence test."""
    logger.info("Starting MSE convergence analysis")

    # Run the test with enough iterations to see convergence
    max_iterations = 5
    mse_results = run_mse_convergence_test(max_iterations)

    if not mse_results:
        logger.error("No results obtained")
        return

    # Analyze the results
    analyze_convergence_cost(mse_results)

    # Create visualization
    plot_mse_convergence(mse_results)

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
