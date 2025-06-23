#!/usr/bin/env python3
"""
Performance comparison script for Mixed Tensor vs CSC optimization.

This script performs detailed performance analysis comparing mixed tensor
and CSC optimization approaches using the synthetic_1000 diffusion patterns.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tools.standalone_optimizer import StandaloneOptimizer

logger = logging.getLogger(__name__)


class PerformanceComparison:
    """Performance comparison between Mixed Tensor and CSC optimization."""

    def __init__(self, patterns_path: str, test_image_path: str):
        """Initialize performance comparison."""
        self.patterns_path = patterns_path
        self.test_image_path = test_image_path
        self.results = {}

    def run_comparison(self, iterations: int = 3) -> Dict[str, Any]:
        """Run performance comparison between mixed tensor and CSC."""
        logger.info("Starting Mixed Tensor vs CSC Performance Comparison")
        logger.info(f"Patterns: {self.patterns_path}")
        logger.info(f"Test image: {self.test_image_path}")
        logger.info(f"Iterations per test: {iterations}")

        # Test CSC (sparse) optimizer
        logger.info("\n=== Testing CSC (Sparse) Optimizer ===")
        csc_results = self._test_optimizer("sparse", iterations)

        # Test Mixed Tensor optimizer
        logger.info("\n=== Testing Mixed Tensor Optimizer ===")
        mixed_results = self._test_optimizer("mixed", iterations)

        # Compare results
        comparison = self._analyze_comparison(csc_results, mixed_results)

        self.results = {
            "csc_results": csc_results,
            "mixed_results": mixed_results,
            "comparison": comparison,
        }

        return self.results

    def _test_optimizer(self, optimizer_type: str, iterations: int) -> Dict[str, Any]:
        """Test a specific optimizer type."""
        results = {
            "optimizer_type": optimizer_type,
            "iterations": iterations,
            "timings": [],
            "optimization_times": [],
            "total_times": [],
            "flop_info": [],
            "error_metrics": [],
        }

        for i in range(iterations):
            logger.info(f"  Iteration {i+1}/{iterations}")

            try:
                # Create optimizer
                optimizer = StandaloneOptimizer(
                    diffusion_patterns_path=self.patterns_path,
                    optimizer_type=optimizer_type,
                )

                # Run optimization
                start_time = time.time()
                result, target_image = optimizer.run(self.test_image_path)
                total_time = time.time() - start_time

                # Collect results
                if hasattr(result, "timing_breakdown") and result.timing_breakdown:
                    results["timings"].append(result.timing_breakdown)
                    results["optimization_times"].append(
                        result.timing_breakdown["optimize_time"]
                    )
                else:
                    results["optimization_times"].append(result.optimization_time)

                results["total_times"].append(total_time)

                if hasattr(result, "flop_info") and result.flop_info:
                    results["flop_info"].append(result.flop_info)

                if hasattr(result, "error_metrics") and result.error_metrics:
                    results["error_metrics"].append(result.error_metrics)

                # Log iteration results
                opt_time = (
                    result.optimization_time
                    if hasattr(result, "optimization_time")
                    else 0
                )
                logger.info(f"    Optimization time: {opt_time:.3f}s")
                if hasattr(result, "flop_info") and result.flop_info:
                    gflops_sec = result.flop_info.get("gflops_per_second", 0)
                    logger.info(f"    GFLOPS/sec: {gflops_sec:.2f}")

            except Exception as e:
                logger.error(f"    Failed iteration {i+1}: {e}")
                continue

        # Calculate statistics
        if results["optimization_times"]:
            results["stats"] = {
                "mean_optimization_time": np.mean(results["optimization_times"]),
                "std_optimization_time": np.std(results["optimization_times"]),
                "min_optimization_time": np.min(results["optimization_times"]),
                "max_optimization_time": np.max(results["optimization_times"]),
                "mean_total_time": np.mean(results["total_times"]),
            }

            # FLOP statistics
            if results["flop_info"]:
                gflops_per_sec = [
                    f.get("gflops_per_second", 0) for f in results["flop_info"]
                ]
                total_flops = [f.get("total_flops", 0) for f in results["flop_info"]]

                results["stats"].update(
                    {
                        "mean_gflops_per_second": np.mean(gflops_per_sec),
                        "max_gflops_per_second": np.max(gflops_per_sec),
                        "mean_total_flops": np.mean(total_flops),
                    }
                )

        return results

    def _analyze_comparison(
        self, csc_results: Dict[str, Any], mixed_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze comparison between CSC and Mixed Tensor results."""
        comparison = {
            "csc_available": bool(csc_results.get("stats")),
            "mixed_available": bool(mixed_results.get("stats")),
        }

        if not (comparison["csc_available"] and comparison["mixed_available"]):
            comparison["error"] = "Missing results for comparison"
            return comparison

        csc_stats = csc_results["stats"]
        mixed_stats = mixed_results["stats"]

        # Performance comparison
        comparison.update(
            {
                "speed_improvement": {
                    "optimization_time_ratio": csc_stats["mean_optimization_time"]
                    / mixed_stats["mean_optimization_time"],
                    "total_time_ratio": csc_stats["mean_total_time"]
                    / mixed_stats["mean_total_time"],
                },
                "gflops_comparison": {
                    "csc_gflops_per_sec": csc_stats.get("mean_gflops_per_second", 0),
                    "mixed_gflops_per_sec": mixed_stats.get(
                        "mean_gflops_per_second", 0
                    ),
                    "mixed_advantage": mixed_stats.get("mean_gflops_per_second", 0)
                    / max(csc_stats.get("mean_gflops_per_second", 1), 1),
                },
                "timing_summary": {
                    "csc_mean_opt_time": csc_stats["mean_optimization_time"],
                    "mixed_mean_opt_time": mixed_stats["mean_optimization_time"],
                    "absolute_speedup": csc_stats["mean_optimization_time"]
                    - mixed_stats["mean_optimization_time"],
                },
            }
        )

        return comparison

    def print_results(self) -> None:
        """Print detailed comparison results."""
        if not self.results:
            logger.error("No results to print. Run comparison first.")
            return

        print("\n" + "=" * 80)
        print("MIXED TENSOR vs CSC PERFORMANCE COMPARISON RESULTS")
        print("=" * 80)

        # CSC Results
        if self.results["csc_results"].get("stats"):
            csc_stats = self.results["csc_results"]["stats"]
            print(f"\nCSC (Sparse) Optimizer Results:")
            print(
                f"  Mean optimization time: {csc_stats['mean_optimization_time']:.3f}s ± {csc_stats['std_optimization_time']:.3f}s"
            )
            print(
                f"  Range: {csc_stats['min_optimization_time']:.3f}s - {csc_stats['max_optimization_time']:.3f}s"
            )
            if "mean_gflops_per_second" in csc_stats:
                print(f"  Mean GFLOPS/sec: {csc_stats['mean_gflops_per_second']:.2f}")
                print(f"  Peak GFLOPS/sec: {csc_stats['max_gflops_per_second']:.2f}")

        # Mixed Tensor Results
        if self.results["mixed_results"].get("stats"):
            mixed_stats = self.results["mixed_results"]["stats"]
            print(f"\nMixed Tensor Optimizer Results:")
            print(
                f"  Mean optimization time: {mixed_stats['mean_optimization_time']:.3f}s ± {mixed_stats['std_optimization_time']:.3f}s"
            )
            print(
                f"  Range: {mixed_stats['min_optimization_time']:.3f}s - {mixed_stats['max_optimization_time']:.3f}s"
            )
            if "mean_gflops_per_second" in mixed_stats:
                print(f"  Mean GFLOPS/sec: {mixed_stats['mean_gflops_per_second']:.2f}")
                print(f"  Peak GFLOPS/sec: {mixed_stats['max_gflops_per_second']:.2f}")

        # Comparison
        comparison = self.results["comparison"]
        if comparison.get("speed_improvement"):
            speed = comparison["speed_improvement"]
            gflops = comparison["gflops_comparison"]
            timing = comparison["timing_summary"]

            print(f"\nPerformance Comparison:")
            print(
                f"  Optimization time speedup: {speed['optimization_time_ratio']:.2f}x"
            )
            print(f"  Total time speedup: {speed['total_time_ratio']:.2f}x")
            print(
                f"  Absolute time savings: {timing['absolute_speedup']:.3f}s per frame"
            )

            if gflops["mixed_gflops_per_sec"] > 0:
                print(
                    f"  GFLOPS advantage (Mixed/CSC): {gflops['mixed_advantage']:.2f}x"
                )
                print(
                    f"  Mixed tensor GFLOPS/sec: {gflops['mixed_gflops_per_sec']:.2f}"
                )
                print(f"  CSC GFLOPS/sec: {gflops['csc_gflops_per_sec']:.2f}")

            # Performance interpretation
            print(f"\nInterpretation:")
            if speed["optimization_time_ratio"] > 1.1:
                print(
                    f"  ✓ Mixed tensor is {speed['optimization_time_ratio']:.1f}x faster than CSC"
                )
            elif speed["optimization_time_ratio"] < 0.9:
                print(
                    f"  ⚠ CSC is {1/speed['optimization_time_ratio']:.1f}x faster than mixed tensor"
                )
            else:
                print(f"  ≈ Similar performance between mixed tensor and CSC")

        print("\n" + "=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Mixed Tensor vs CSC Performance Comparison"
    )
    parser.add_argument(
        "--patterns",
        "-p",
        default="diffusion_patterns/synthetic_1000",
        help="Diffusion patterns file (without .npz extension)",
    )
    parser.add_argument(
        "--image", "-i", default="test_image.jpg", help="Test image path"
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=3,
        help="Number of iterations per test (default: 3)",
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

    # Validate inputs
    patterns_file = f"{args.patterns}.npz"
    if not Path(patterns_file).exists():
        logger.error(f"Patterns file not found: {patterns_file}")
        return 1

    if not Path(args.image).exists():
        logger.error(f"Test image not found: {args.image}")
        return 1

    try:
        # Run comparison
        comparison = PerformanceComparison(args.patterns, args.image)
        comparison.run_comparison(iterations=args.iterations)
        comparison.print_results()

        return 0

    except Exception as e:
        logger.error(f"Performance comparison failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
