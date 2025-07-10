#!/usr/bin/env python3
"""
Comprehensive MSE convergence comparison test program.

This program creates a detailed table of MSE values by iteration for different
ATA inverse formats, allowing analysis of convergence behavior and additional
iteration costs for compressed formats.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_pattern_components(
    pattern_path: str, use_dia_ata_inverse: bool = False
) -> Tuple[SingleBlockMixedSparseTensor, DiagonalATAMatrix, any]:
    """Load pattern components with error handling."""
    try:
        data = np.load(pattern_path, allow_pickle=True)

        # Load mixed tensor (A^T matrix)
        mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())

        # Load DIA matrix (A^T A matrix)
        dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())

        # Load ATA inverse
        ata_inverse = None
        if use_dia_ata_inverse and "ata_inverse_dia" in data:
            ata_inverse = data["ata_inverse_dia"].item()
            logger.info(f"Loaded DIA format ATA inverse from {pattern_path}")
        elif "ata_inverse" in data:
            ata_inverse = data["ata_inverse"]
            logger.info(f"Loaded dense format ATA inverse from {pattern_path}")
        else:
            raise ValueError(f"No ATA inverse found in {pattern_path}")

        return mixed_tensor, dia_matrix, ata_inverse

    except Exception as e:
        logger.error(f"Error loading pattern {pattern_path}: {e}")
        raise


def create_challenging_test_frame(height: int = 480, width: int = 800, frame_type: str = "gradient") -> np.ndarray:
    """Create various types of challenging test frames."""

    if frame_type == "gradient":
        # Multi-directional gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Red: left-to-right gradient
        frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)

        # Green: top-to-bottom gradient
        frame[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)

        # Blue: radial gradient from center
        y_center, x_center = height // 2, width // 2
        y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        distances = np.sqrt((y_indices - y_center) ** 2 + (x_indices - x_center) ** 2)
        max_distance = np.sqrt(y_center**2 + x_center**2)
        frame[:, :, 2] = (255 * distances / max_distance).astype(np.uint8)

    elif frame_type == "checkerboard":
        # High-frequency checkerboard pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        checker_size = 32
        y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        # Different checkerboard patterns for each channel
        for channel in range(3):
            size = checker_size * (2**channel)  # Different frequencies
            checker = ((y_indices // size) + (x_indices // size)) % 2
            frame[:, :, channel] = checker * 255

    elif frame_type == "stripes":
        # Vertical stripes with different frequencies per channel
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for channel in range(3):
            stripe_width = 16 * (2**channel)
            stripes = (np.arange(width) // stripe_width) % 2
            frame[:, :, channel] = stripes * 255

    else:  # random
        # High-contrast random noise
        frame = np.random.choice([0, 255], size=(height, width, 3), p=[0.3, 0.7]).astype(np.uint8)

    return frame


def run_mse_comparison_study(max_iterations: int = 10, num_test_frames: int = 3) -> pd.DataFrame:
    """Run comprehensive MSE comparison study."""

    logger.info(f"Starting MSE comparison study with {max_iterations} iterations and {num_test_frames} test frames")

    # Define pattern configurations to test
    pattern_configs = [
        {
            "name": "Dense ATA⁻¹",
            "file": "diffusion_patterns/synthetic_2624_uint8.npz",
            "use_dia": False,
            "description": "Uncompressed dense ATA inverse",
        }
    ]

    # Add DIA compressed patterns if they exist
    dia_compressions = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    for compression in dia_compressions:
        pattern_file = f"diffusion_patterns/synthetic_2624_uint8_dia_{compression}.npz"
        if Path(pattern_file).exists():
            pattern_configs.append(
                {
                    "name": f"DIA {compression}x",
                    "file": pattern_file,
                    "use_dia": True,
                    "description": f"DIA compressed ATA inverse (factor {compression})",
                }
            )

    logger.info(f"Testing {len(pattern_configs)} pattern configurations")

    # Test frame types
    frame_types = ["gradient", "checkerboard", "stripes"]

    # Collect all results
    results = []

    for frame_idx, frame_type in enumerate(frame_types[:num_test_frames]):
        logger.info(f"Testing frame type: {frame_type} ({frame_idx + 1}/{num_test_frames})")

        # Create test frame
        test_frame = create_challenging_test_frame(frame_type=frame_type)

        for config in pattern_configs:
            logger.info(f"  Testing {config['name']}...")

            try:
                # Load pattern components
                mixed_tensor, dia_matrix, ata_inverse = load_pattern_components(
                    config["file"], use_dia_ata_inverse=config["use_dia"]
                )

                # Run optimization with MSE tracking
                result = optimize_frame_led_values(
                    target_frame=test_frame,
                    at_matrix=mixed_tensor,
                    ata_matrix=dia_matrix,
                    ata_inverse=ata_inverse,
                    max_iterations=max_iterations,
                    track_mse_per_iteration=True,
                    debug=False,
                )

                # Store results for each iteration
                mse_values = result.mse_per_iteration
                if mse_values is not None:
                    for iteration, mse in enumerate(mse_values):
                        results.append(
                            {
                                "frame_type": frame_type,
                                "frame_idx": frame_idx,
                                "pattern_name": config["name"],
                                "pattern_file": config["file"],
                                "compression_type": "DIA" if config["use_dia"] else "Dense",
                                "iteration": iteration,
                                "mse": float(mse),
                                "converged": result.converged,
                                "total_iterations": result.iterations,
                            }
                        )

                    logger.info(f"    ✅ {config['name']}: Final MSE = {mse_values[-1]:.6f}")
                else:
                    logger.warning(f"    ❌ {config['name']}: No MSE values returned")

            except Exception as e:
                logger.error(f"    ❌ {config['name']}: Error - {e}")
                continue

    # Convert to DataFrame
    df = pd.DataFrame(results)
    logger.info(f"Collected {len(df)} data points")

    return df


def analyze_convergence_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze additional iteration costs for compressed formats."""

    if df.empty:
        logger.warning("No data to analyze")
        return pd.DataFrame()

    analysis_results = []

    # Group by frame type
    for frame_type in df["frame_type"].unique():
        frame_data = df[df["frame_type"] == frame_type]

        # Find baseline (dense) performance
        dense_data = frame_data[frame_data["compression_type"] == "Dense"]
        if dense_data.empty:
            logger.warning(f"No dense baseline found for frame type {frame_type}")
            continue

        # Get final MSE from dense method
        dense_final_mse = dense_data[dense_data["iteration"] == dense_data["iteration"].max()]["mse"].iloc[0]
        target_mse_threshold = dense_final_mse * 1.01  # Within 1% of dense final MSE

        logger.info(
            f"Frame type '{frame_type}': Dense final MSE = {dense_final_mse:.6f}, target threshold = {target_mse_threshold:.6f}"
        )

        # Analyze each compression method
        for pattern_name in frame_data["pattern_name"].unique():
            pattern_data = frame_data[frame_data["pattern_name"] == pattern_name]

            if pattern_name.startswith("Dense"):
                # Baseline case
                dense_iterations_to_target = len(dense_data)
                analysis_results.append(
                    {
                        "frame_type": frame_type,
                        "pattern_name": pattern_name,
                        "final_mse": dense_final_mse,
                        "iterations_to_target": dense_iterations_to_target,
                        "additional_iterations": 0,
                        "convergence_cost": 0.0,
                    }
                )

            else:
                # Compressed case - find iterations needed to reach target MSE
                iterations_to_target = len(pattern_data)  # Default to max iterations
                final_mse = pattern_data[pattern_data["iteration"] == pattern_data["iteration"].max()]["mse"].iloc[0]

                # Find first iteration where MSE <= target threshold
                target_reached = pattern_data[pattern_data["mse"] <= target_mse_threshold]
                if not target_reached.empty:
                    iterations_to_target = target_reached["iteration"].iloc[0] + 1  # +1 because iteration is 0-indexed

                additional_iterations = max(0, iterations_to_target - dense_iterations_to_target)
                convergence_cost = (
                    additional_iterations / dense_iterations_to_target if dense_iterations_to_target > 0 else 0
                )

                analysis_results.append(
                    {
                        "frame_type": frame_type,
                        "pattern_name": pattern_name,
                        "final_mse": final_mse,
                        "iterations_to_target": iterations_to_target,
                        "additional_iterations": additional_iterations,
                        "convergence_cost": convergence_cost,
                    }
                )

    analysis_df = pd.DataFrame(analysis_results)
    return analysis_df


def create_visualizations(df: pd.DataFrame, analysis_df: pd.DataFrame, output_dir: str = "."):
    """Create comprehensive visualizations."""

    if df.empty:
        logger.warning("No data for visualization")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. MSE convergence curves by pattern type
    fig, axes = plt.subplots(1, len(df["frame_type"].unique()), figsize=(15, 5))
    if len(df["frame_type"].unique()) == 1:
        axes = [axes]

    for idx, frame_type in enumerate(df["frame_type"].unique()):
        frame_data = df[df["frame_type"] == frame_type]

        ax = axes[idx]
        for pattern_name in frame_data["pattern_name"].unique():
            pattern_data = frame_data[frame_data["pattern_name"] == pattern_name].sort_values("iteration")
            ax.plot(
                pattern_data["iteration"],
                pattern_data["mse"],
                marker="o",
                label=pattern_name,
                linewidth=2,
                markersize=4,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("MSE")
        ax.set_title(f"MSE Convergence: {frame_type}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path / "mse_convergence_by_frame_type.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 2. Additional iterations cost analysis (if we have analysis data)
    if not analysis_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Additional iterations bar plot
        patterns = analysis_df["pattern_name"].unique()
        x_pos = np.arange(len(patterns))

        for frame_type in analysis_df["frame_type"].unique():
            frame_analysis = analysis_df[analysis_df["frame_type"] == frame_type]
            additional_iters = [
                (
                    frame_analysis[frame_analysis["pattern_name"] == p]["additional_iterations"].iloc[0]
                    if not frame_analysis[frame_analysis["pattern_name"] == p].empty
                    else 0
                )
                for p in patterns
            ]

            ax1.bar(
                x_pos + 0.1 * list(analysis_df["frame_type"].unique()).index(frame_type),
                additional_iters,
                width=0.25,
                label=frame_type,
                alpha=0.8,
            )

        ax1.set_xlabel("ATA Inverse Method")
        ax1.set_ylabel("Additional Iterations Needed")
        ax1.set_title("Additional Iterations to Reach Dense MSE Quality")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(patterns, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Convergence cost percentage
        for frame_type in analysis_df["frame_type"].unique():
            frame_analysis = analysis_df[analysis_df["frame_type"] == frame_type]
            convergence_costs = [
                (
                    frame_analysis[frame_analysis["pattern_name"] == p]["convergence_cost"].iloc[0] * 100
                    if not frame_analysis[frame_analysis["pattern_name"] == p].empty
                    else 0
                )
                for p in patterns
            ]

            ax2.bar(
                x_pos + 0.1 * list(analysis_df["frame_type"].unique()).index(frame_type),
                convergence_costs,
                width=0.25,
                label=frame_type,
                alpha=0.8,
            )

        ax2.set_xlabel("ATA Inverse Method")
        ax2.set_ylabel("Convergence Cost (%)")
        ax2.set_title("Relative Convergence Cost vs Dense Method")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(patterns, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "convergence_cost_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()


def print_summary_table(df: pd.DataFrame, analysis_df: pd.DataFrame):
    """Print comprehensive summary tables."""

    if df.empty:
        logger.warning("No data for summary")
        return

    print("\n" + "=" * 80)
    print("MSE CONVERGENCE COMPARISON SUMMARY")
    print("=" * 80)

    # Table 1: Final MSE values
    print("\n📊 Table 1: Final MSE Values by Method and Frame Type")
    print("-" * 60)

    final_mse_table = df[df["iteration"] == df.groupby(["frame_type", "pattern_name"])["iteration"].transform("max")]
    pivot_final = final_mse_table.pivot(index="pattern_name", columns="frame_type", values="mse")

    if not pivot_final.empty:
        print(pivot_final.round(6).to_string())

    # Table 2: MSE by iteration (first frame type only for space)
    if len(df["frame_type"].unique()) > 0:
        first_frame_type = df["frame_type"].unique()[0]
        frame_data = df[df["frame_type"] == first_frame_type]

        print(f"\n📈 Table 2: MSE by Iteration ({first_frame_type} frame)")
        print("-" * 60)

        pivot_iterations = frame_data.pivot(index="iteration", columns="pattern_name", values="mse")
        if not pivot_iterations.empty:
            print(pivot_iterations.round(6).to_string())

    # Table 3: Convergence cost analysis
    if not analysis_df.empty:
        print("\n⚡ Table 3: Convergence Cost Analysis")
        print("-" * 60)

        summary_cols = [
            "frame_type",
            "pattern_name",
            "final_mse",
            "iterations_to_target",
            "additional_iterations",
            "convergence_cost",
        ]
        print(analysis_df[summary_cols].round(4).to_string(index=False))

    print("\n" + "=" * 80)


def main():
    """Main function to run the MSE comparison study."""

    logger.info("🚀 Starting MSE Convergence Comparison Study")

    # Configuration
    max_iterations = 8  # Enough to see convergence trends
    num_test_frames = 2  # Test multiple frame types

    try:
        # Run the comparison study
        df = run_mse_comparison_study(max_iterations=max_iterations, num_test_frames=num_test_frames)

        if df.empty:
            logger.error("No data collected. Check pattern files and configurations.")
            return

        # Analyze convergence costs
        analysis_df = analyze_convergence_cost(df)

        # Print summary tables
        print_summary_table(df, analysis_df)

        # Create visualizations
        create_visualizations(df, analysis_df)

        # Save results to CSV
        df.to_csv("mse_convergence_results.csv", index=False)
        if not analysis_df.empty:
            analysis_df.to_csv("convergence_cost_analysis.csv", index=False)

        logger.info("✅ MSE comparison study completed successfully!")
        logger.info("📁 Results saved to: mse_convergence_results.csv")
        if not analysis_df.empty:
            logger.info("📁 Analysis saved to: convergence_cost_analysis.csv")

    except Exception as e:
        logger.error(f"❌ Study failed: {e}")
        raise


if __name__ == "__main__":
    main()
