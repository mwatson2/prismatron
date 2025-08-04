#!/usr/bin/env python3
"""
Simple MSE convergence comparison test program without external dependencies.

This program creates a detailed table of MSE values by iteration for different
ATA inverse formats, allowing analysis of convergence behavior.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

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


def create_test_frames(num_frames: int = 3) -> List[Tuple[str, np.ndarray]]:
    """Create various challenging test frames."""

    frames = []
    height, width = 480, 800

    # Frame 1: Multi-directional gradient
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    frame1[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)  # Red: left-to-right
    frame1[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)  # Green: top-to-bottom

    # Blue: radial gradient
    y_center, x_center = height // 2, width // 2
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    distances = np.sqrt((y_indices - y_center) ** 2 + (x_indices - x_center) ** 2)
    max_distance = np.sqrt(y_center**2 + x_center**2)
    frame1[:, :, 2] = (255 * distances / max_distance).astype(np.uint8)
    frames.append(("gradient", frame1))

    if num_frames > 1:
        # Frame 2: Checkerboard pattern
        frame2 = np.zeros((height, width, 3), dtype=np.uint8)
        checker_size = 32
        for channel in range(3):
            size = checker_size * (2**channel)
            checker = ((y_indices // size) + (x_indices // size)) % 2
            frame2[:, :, channel] = checker * 255
        frames.append(("checkerboard", frame2))

    if num_frames > 2:
        # Frame 3: High-contrast noise
        frame3 = np.random.choice([0, 255], size=(height, width, 3), p=[0.3, 0.7]).astype(np.uint8)
        frames.append(("noise", frame3))

    return frames[:num_frames]


def run_mse_comparison(max_iterations: int = 8) -> Dict:
    """Run MSE comparison and return results dictionary."""

    logger.info(f"Starting MSE comparison with {max_iterations} iterations")

    # Define pattern configurations
    pattern_configs = [
        {
            "name": "Dense_ATA_Inverse",
            "file": "diffusion_patterns/synthetic_2624_uint8.npz",
            "use_dia": False,
            "description": "Uncompressed dense ATA inverse",
        }
    ]

    # Add available DIA patterns
    dia_compressions = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    for compression in dia_compressions:
        pattern_file = f"diffusion_patterns/synthetic_2624_uint8_dia_{compression}.npz"
        if Path(pattern_file).exists():
            pattern_configs.append(
                {
                    "name": f"DIA_{compression}x",
                    "file": pattern_file,
                    "use_dia": True,
                    "description": f"DIA compressed ATA inverse (factor {compression})",
                }
            )

    logger.info(f"Found {len(pattern_configs)} pattern configurations")

    # Create test frames
    test_frames = create_test_frames(num_frames=2)  # Limit to 2 for faster execution

    # Results storage
    results = {
        "patterns": [config["name"] for config in pattern_configs],
        "frames": {},
        "iterations": list(range(max_iterations + 1)),  # Include initial MSE
    }

    for frame_name, test_frame in test_frames:
        logger.info(f"Testing with {frame_name} frame...")

        frame_results = {}
        baseline_mse = None

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

                if result.mse_per_iteration is not None:
                    mse_values = result.mse_per_iteration.tolist()
                    frame_results[config["name"]] = {
                        "mse_values": mse_values,
                        "final_mse": mse_values[-1],
                        "iterations_run": len(mse_values) - 1,
                        "converged": result.converged,
                    }

                    # Store baseline for comparison
                    if "Dense" in config["name"]:
                        baseline_mse = mse_values[-1]

                    logger.info(f"    ✅ Final MSE: {mse_values[-1]:.6f}")
                else:
                    logger.warning("    ❌ No MSE values returned")

            except Exception as e:
                logger.error(f"    ❌ Error: {e}")
                frame_results[config["name"]] = {"error": str(e)}

        # Add convergence analysis
        if baseline_mse is not None:
            for pattern_name, pattern_data in frame_results.items():
                if "mse_values" in pattern_data and "Dense" not in pattern_name:
                    mse_values = pattern_data["mse_values"]
                    target_threshold = baseline_mse * 1.01  # Within 1% of baseline

                    # Find iterations to reach target
                    iterations_to_target = len(mse_values)
                    for i, mse in enumerate(mse_values):
                        if mse <= target_threshold:
                            iterations_to_target = i
                            break

                    pattern_data["iterations_to_target"] = iterations_to_target
                    pattern_data["additional_iterations"] = max(
                        0,
                        iterations_to_target
                        - (len(frame_results.get("Dense_ATA_Inverse", {}).get("mse_values", [])) - 1),
                    )

        results["frames"][frame_name] = frame_results

    return results


def print_mse_table(results: Dict):
    """Print formatted MSE table."""

    print("\n" + "=" * 100)
    print("MSE CONVERGENCE COMPARISON TABLE")
    print("=" * 100)

    patterns = results["patterns"]

    for frame_name, frame_data in results["frames"].items():
        print(f"\n📊 FRAME TYPE: {frame_name.upper()}")
        print("-" * 80)

        # Print header
        header = f"{'Iteration':<10}"
        for pattern in patterns:
            header += f"{pattern:<15}"
        print(header)
        print("-" * len(header))

        # Find max iterations across all patterns
        max_iters = 0
        for pattern in patterns:
            if pattern in frame_data and "mse_values" in frame_data[pattern]:
                max_iters = max(max_iters, len(frame_data[pattern]["mse_values"]))

        # Print MSE values by iteration
        for iteration in range(max_iters):
            row = f"{iteration:<10}"
            for pattern in patterns:
                if pattern in frame_data and "mse_values" in frame_data[pattern]:
                    mse_values = frame_data[pattern]["mse_values"]
                    if iteration < len(mse_values):
                        row += f"{mse_values[iteration]:<15.6f}"
                    else:
                        row += f"{'---':<15}"
                else:
                    row += f"{'ERROR':<15}"
            print(row)

        # Print summary
        print("\n📈 SUMMARY:")
        for pattern in patterns:
            if pattern in frame_data and "mse_values" in frame_data[pattern]:
                data = frame_data[pattern]
                print(f"  {pattern}:")
                print(f"    Final MSE: {data['final_mse']:.6f}")
                print(f"    Iterations: {data['iterations_run']}")
                if "iterations_to_target" in data:
                    print(f"    Iterations to baseline MSE: {data['iterations_to_target']}")
                if "additional_iterations" in data:
                    print(f"    Additional iterations vs dense: {data['additional_iterations']}")


def create_plot(results: Dict):
    """Create MSE convergence plot."""

    try:
        plt.figure(figsize=(12, 8))

        # Plot for first frame type
        frame_name = list(results["frames"].keys())[0]
        frame_data = results["frames"][frame_name]

        for pattern in results["patterns"]:
            if pattern in frame_data and "mse_values" in frame_data[pattern]:
                mse_values = frame_data[pattern]["mse_values"]
                iterations = list(range(len(mse_values)))
                plt.plot(iterations, mse_values, marker="o", label=pattern, linewidth=2, markersize=4)

        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.title(f"MSE Convergence Comparison ({frame_name} frame)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        # Save plot
        plt.savefig("mse_convergence_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

        logger.info("📊 Plot saved as mse_convergence_comparison.png")

    except Exception as e:
        logger.error(f"Error creating plot: {e}")


def save_results_csv(results: Dict):
    """Save results to CSV format."""

    try:
        with open("mse_results_table.csv", "w") as f:
            # Write header
            f.write("frame_type,pattern_name,iteration,mse,final_mse,additional_iterations\n")

            # Write data
            for frame_name, frame_data in results["frames"].items():
                for pattern in results["patterns"]:
                    if pattern in frame_data and "mse_values" in frame_data[pattern]:
                        data = frame_data[pattern]
                        mse_values = data["mse_values"]
                        final_mse = data["final_mse"]
                        additional_iters = data.get("additional_iterations", 0)

                        for iteration, mse in enumerate(mse_values):
                            f.write(
                                f"{frame_name},{pattern},{iteration},{mse:.6f},{final_mse:.6f},{additional_iters}\n"
                            )

        logger.info("📁 Results saved to mse_results_table.csv")

    except Exception as e:
        logger.error(f"Error saving CSV: {e}")


def main():
    """Main function."""

    logger.info("🚀 Starting MSE Convergence Table Generation")

    try:
        # Run comparison
        results = run_mse_comparison(max_iterations=8)

        # Print table
        print_mse_table(results)

        # Create visualization
        create_plot(results)

        # Save results
        save_results_csv(results)

        logger.info("✅ MSE comparison completed successfully!")

    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()
