#!/usr/bin/env python3
"""
Matrix Performance Comparison Tool

Tests all four combinations of A^T and A^T A matrix formats:
1. CSC A^T + DIA A^T A (sparse-sparse)
2. CSC A^T + Dense A^T A (sparse-dense)
3. Mixed A^T + DIA A^T A (mixed-sparse)
4. Mixed A^T + Dense A^T A (mixed-dense)

Excludes initialization and cache warming times for fair comparison.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import FrameOptimizationResult, optimize_frame_led_values
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


class MatrixPerformanceComparison:
    """Performance comparison tool for different matrix formats."""

    def __init__(self, patterns_path: str):
        """Initialize with patterns file."""
        self.patterns_path = patterns_path
        logger.info(f"Loading patterns from: {patterns_path}")

        # Load all data once
        self.patterns_data = np.load(patterns_path, allow_pickle=True)

        # Initialize all matrix formats
        self._initialize_matrices()

    def _initialize_matrices(self):
        """Initialize all matrix formats during setup (not counted in timing)."""
        logger.info("Initializing all matrix formats...")

        # 1. CSC A^T matrix
        csc_data_dict = self.patterns_data["diffusion_matrix"].item()
        self.csc_matrix = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
        logger.info(f"CSC matrix: {self.csc_matrix.led_count} LEDs")

        # 2. Mixed tensor A^T matrix
        mixed_tensor_dict = self.patterns_data["mixed_tensor"].item()
        self.mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
        logger.info(f"Mixed tensor: {self.mixed_tensor.batch_size} LEDs")

        # 3. DIA A^T A matrix
        self.dia_matrix = DiagonalATAMatrix(led_count=self.csc_matrix.led_count)
        csc_full = self.csc_matrix.to_csc_matrix()
        led_positions = self.patterns_data["led_positions"]
        self.dia_matrix.build_from_diffusion_matrix(csc_full, led_positions)
        logger.info(f"DIA matrix: shape {self.dia_matrix.dia_data_cpu.shape}")

        # 4. Dense A^T A matrix
        dense_ata_dict = self.patterns_data["dense_ata"].item()
        self.dense_ata = dense_ata_dict["dense_ata_matrices"]
        logger.info(f"Dense ATA matrix: shape {self.dense_ata.shape}")

        self.led_count = self.csc_matrix.led_count
        logger.info("All matrices initialized successfully")

    def load_test_image(self, image_path: str) -> np.ndarray:
        """Load and prepare test image."""
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image, dtype=np.uint8)
        logger.info(f"Loaded test image: {image_array.shape}")
        return image_array

    def run_optimization(
        self,
        target_image: np.ndarray,
        at_format: str,
        ata_format: str,
        max_iterations: int = 10,
        warmup_runs: int = 2,
        timing_runs: int = 3,
    ) -> Tuple[FrameOptimizationResult, Dict[str, float]]:
        """
        Run optimization with specified matrix formats.

        Args:
            target_image: Target image for optimization
            at_format: "csc" or "mixed" for A^T matrix
            ata_format: "dia" or "dense" for A^T A matrix
            max_iterations: Number of optimization iterations
            warmup_runs: Number of warmup runs (not timed)
            timing_runs: Number of timed runs for averaging

        Returns:
            (result, timing_stats)
        """
        # Select matrices based on format
        if at_format == "csc":
            AT_matrix = self.csc_matrix
        elif at_format == "mixed":
            AT_matrix = self.mixed_tensor
        else:
            raise ValueError(f"Invalid AT format: {at_format}")

        if ata_format == "dia":
            ATA_matrix = self.dia_matrix
        elif ata_format == "dense":
            ATA_matrix = self.dense_ata
        else:
            raise ValueError(f"Invalid ATA format: {ata_format}")

        logger.info(f"Testing {at_format.upper()} A^T + {ata_format.upper()} A^T A")

        # Warmup runs (not timed)
        logger.info(f"Running {warmup_runs} warmup iterations...")
        for i in range(warmup_runs):
            _ = optimize_frame_led_values(
                target_frame=target_image,
                AT_matrix=AT_matrix,
                ATA_matrix=ATA_matrix,
                max_iterations=max_iterations,
                compute_error_metrics=False,
                debug=False,
            )
            logger.info(f"  Warmup {i+1}/{warmup_runs} completed")

        # Timed runs
        logger.info(f"Running {timing_runs} timed iterations...")
        timing_results = []
        final_result = None

        for i in range(timing_runs):
            start_time = time.time()

            result = optimize_frame_led_values(
                target_frame=target_image,
                AT_matrix=AT_matrix,
                ATA_matrix=ATA_matrix,
                max_iterations=max_iterations,
                compute_error_metrics=True,  # Only compute for final measurement
                debug=False,
            )

            end_time = time.time()
            timing_results.append(end_time - start_time)
            final_result = result

            logger.info(f"  Timed run {i+1}/{timing_runs}: {timing_results[-1]:.3f}s")

        # Calculate timing statistics
        timing_stats = {
            "mean_time": np.mean(timing_results),
            "std_time": np.std(timing_results),
            "min_time": np.min(timing_results),
            "max_time": np.max(timing_results),
            "all_times": timing_results,
        }

        logger.info(
            f"Average time: {timing_stats['mean_time']:.3f}s ± {timing_stats['std_time']:.3f}s"
        )

        return final_result, timing_stats

    def run_full_comparison(
        self,
        image_path: str,
        max_iterations: int = 10,
        warmup_runs: int = 2,
        timing_runs: int = 3,
    ) -> Dict[str, Dict]:
        """Run full comparison across all four matrix combinations."""
        logger.info("=== Starting Full Matrix Performance Comparison ===")

        # Load test image once
        target_image = self.load_test_image(image_path)

        # Define all combinations
        combinations = [
            ("csc", "dia", "CSC A^T + DIA A^T A (sparse-sparse)"),
            ("csc", "dense", "CSC A^T + Dense A^T A (sparse-dense)"),
            ("mixed", "dia", "Mixed A^T + DIA A^T A (mixed-sparse)"),
            ("mixed", "dense", "Mixed A^T + Dense A^T A (mixed-dense)"),
        ]

        results = {}

        for at_format, ata_format, description in combinations:
            logger.info(f"\n=== Testing: {description} ===")

            try:
                result, timing_stats = self.run_optimization(
                    target_image=target_image,
                    at_format=at_format,
                    ata_format=ata_format,
                    max_iterations=max_iterations,
                    warmup_runs=warmup_runs,
                    timing_runs=timing_runs,
                )

                combo_key = f"{at_format}_{ata_format}"
                results[combo_key] = {
                    "description": description,
                    "at_format": at_format,
                    "ata_format": ata_format,
                    "result": result,
                    "timing": timing_stats,
                    "success": True,
                }

                logger.info(f"✓ {description} completed successfully")

            except Exception as e:
                logger.error(f"✗ {description} failed: {e}")
                combo_key = f"{at_format}_{ata_format}"
                results[combo_key] = {
                    "description": description,
                    "at_format": at_format,
                    "ata_format": ata_format,
                    "success": False,
                    "error": str(e),
                }

        return results

    def print_comparison_report(self, results: Dict[str, Dict]):
        """Print detailed comparison report."""
        logger.info("\n" + "=" * 80)
        logger.info("MATRIX PERFORMANCE COMPARISON REPORT")
        logger.info("=" * 80)

        successful_results = {
            k: v for k, v in results.items() if v.get("success", False)
        }

        if not successful_results:
            logger.error("No successful runs to compare!")
            return

        logger.info(f"Test Configuration:")
        logger.info(f"  LED Count: {self.led_count}")
        logger.info(
            f"  Successful combinations: {len(successful_results)}/{len(results)}"
        )

        # Performance comparison table
        logger.info(f"\nPerformance Results:")
        logger.info(
            f"{'Combination':<25} {'Mean Time (s)':<15} {'Std (s)':<10} {'FPS':<8} {'MSE':<12} {'Converged':<10}"
        )
        logger.info("-" * 80)

        fastest_time = float("inf")
        fastest_combo = None
        best_mse = float("inf")
        best_mse_combo = None

        for combo_key, data in successful_results.items():
            result = data["result"]
            timing = data["timing"]

            mean_time = timing["mean_time"]
            std_time = timing["std_time"]
            fps = 1.0 / mean_time if mean_time > 0 else 0
            mse = result.error_metrics.get("mse", float("inf"))
            converged = "Yes" if result.converged else "No"

            # Track best performance
            if mean_time < fastest_time:
                fastest_time = mean_time
                fastest_combo = data["description"]

            if mse < best_mse:
                best_mse = mse
                best_mse_combo = data["description"]

            logger.info(
                f"{combo_key:<25} {mean_time:<15.3f} {std_time:<10.3f} {fps:<8.1f} {mse:<12.6f} {converged:<10}"
            )

        # Summary
        logger.info(f"\nSummary:")
        logger.info(f"  Fastest: {fastest_combo} ({fastest_time:.3f}s)")
        logger.info(f"  Best MSE: {best_mse_combo} (MSE: {best_mse:.6f})")

        # Detailed timing breakdown
        logger.info(f"\nDetailed Timing (all runs):")
        for combo_key, data in successful_results.items():
            if "timing" in data:
                times = data["timing"]["all_times"]
                logger.info(f"  {combo_key}: {[f'{t:.3f}' for t in times]}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Matrix Performance Comparison Tool")
    parser.add_argument(
        "--patterns", required=True, help="Path to diffusion patterns file"
    )
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Optimization iterations per run"
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs (not timed)")
    parser.add_argument(
        "--timing-runs", type=int, default=3, help="Timed runs for averaging"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # Run comparison
        comparison = MatrixPerformanceComparison(args.patterns)
        results = comparison.run_full_comparison(
            image_path=args.image,
            max_iterations=args.iterations,
            warmup_runs=args.warmup,
            timing_runs=args.timing_runs,
        )

        # Print report
        comparison.print_comparison_report(results)

        return 0

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
