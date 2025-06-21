#!/usr/bin/env python3
"""
Comprehensive LED Optimization Performance Comparison Tool.

This tool provides accurate performance comparison between sparse and dense
LED optimizers with proper GPU synchronization, timing methodology, and
FLOPS analysis.

Usage:
    python compare_optimization_performance.py --patterns diffusion_patterns/synthetic_1000
    python compare_optimization_performance.py --patterns diffusion_patterns/synthetic_1000 \\
        --iterations 20 --runs 5
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cupy as cp
import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.const import FRAME_HEIGHT, FRAME_WIDTH

# Add archive directory to path for sparse optimizer
archive_path = str(Path(__file__).parent.parent / "archive")
sys.path.insert(0, archive_path)
from led_optimizer_sparse import LEDOptimizer

from src.consumer.led_optimizer_dense import DenseLEDOptimizer

logger = logging.getLogger(__name__)


class PerformanceTimer:
    """High-precision timing with GPU synchronization support."""

    def __init__(self, use_gpu_timing: bool = True):
        self.use_gpu_timing = use_gpu_timing
        self.start_time = None
        self.start_event = None
        self.end_event = None

    def start(self):
        """Start timing with optional GPU synchronization."""
        if self.use_gpu_timing:
            # Ensure all pending GPU operations complete
            cp.cuda.Stream.null.synchronize()
            self.start_event = cp.cuda.Event()
            self.end_event = cp.cuda.Event()
            self.start_event.record()

        self.start_time = time.time()

    def stop(self) -> Tuple[float, float]:
        """Stop timing and return (cpu_time, gpu_time)."""
        cpu_time = time.time() - self.start_time

        if self.use_gpu_timing and self.start_event and self.end_event:
            self.end_event.record()
            cp.cuda.runtime.deviceSynchronize()
            gpu_time = (
                cp.cuda.get_elapsed_time(self.start_event, self.end_event) / 1000.0
            )
            return cpu_time, gpu_time
        else:
            return cpu_time, cpu_time


class OptimizerBenchmark:
    """Comprehensive benchmark for LED optimizers."""

    def __init__(self, patterns_path: str):
        self.patterns_path = patterns_path
        self.test_image = self._create_test_image()

        # Initialize optimizers
        logger.info("Initializing sparse optimizer...")
        self.sparse_optimizer = LEDOptimizer(diffusion_patterns_path=patterns_path)
        if not self.sparse_optimizer.initialize():
            raise RuntimeError("Failed to initialize sparse optimizer")

        logger.info("Initializing dense optimizer...")
        self.dense_optimizer = DenseLEDOptimizer(diffusion_patterns_path=patterns_path)
        if not self.dense_optimizer.initialize():
            raise RuntimeError("Failed to initialize dense optimizer")

    def _create_test_image(self) -> np.ndarray:
        """Create a test image with gradients and patterns."""
        image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # Create horizontal gradient in red channel
        for x in range(FRAME_WIDTH):
            image[:, x, 0] = int(255 * x / FRAME_WIDTH)

        # Create vertical gradient in green channel
        for y in range(FRAME_HEIGHT):
            image[y, :, 1] = int(255 * y / FRAME_HEIGHT)

        # Create circular pattern in blue channel
        center_x, center_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
        for y in range(FRAME_HEIGHT):
            for x in range(FRAME_WIDTH):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                image[y, x, 2] = int(255 * (1 - dist / max_dist))

        return image

    def _measure_optimization_time(self, optimizer, iterations: int, runs: int) -> Dict:
        """Measure optimization timing with multiple runs."""
        cpu_times = []
        gpu_times = []
        results = []

        optimizer_name = "dense" if hasattr(optimizer, "_ATA_gpu") else "sparse"
        logger.info(
            f"Benchmarking {optimizer_name} optimizer: {runs} runs × {iterations} iterations"
        )

        # Warm-up run
        logger.info("Warm-up run...")
        # Use debug version for benchmarking to get error metrics
        if hasattr(optimizer, "optimize_frame_with_debug"):
            _ = optimizer.optimize_frame_with_debug(
                self.test_image, max_iterations=iterations
            )
        else:
            _ = optimizer.optimize_frame(self.test_image, max_iterations=iterations)

        for run in range(runs):
            logger.info(f"Run {run + 1}/{runs}")

            # Use precise timing with GPU synchronization
            timer = PerformanceTimer(use_gpu_timing=True)
            timer.start()

            # Use debug version for benchmarking to get error metrics
            if hasattr(optimizer, "optimize_frame_with_debug"):
                result = optimizer.optimize_frame_with_debug(
                    self.test_image, max_iterations=iterations
                )
            else:
                result = optimizer.optimize_frame(
                    self.test_image, max_iterations=iterations
                )

            cpu_time, gpu_time = timer.stop()

            cpu_times.append(cpu_time)
            gpu_times.append(gpu_time)
            results.append(result)

            # Brief pause between runs
            time.sleep(0.1)

        # Calculate statistics
        avg_cpu_time = np.mean(cpu_times)
        std_cpu_time = np.std(cpu_times)
        min_cpu_time = np.min(cpu_times)
        max_cpu_time = np.max(cpu_times)

        avg_gpu_time = np.mean(gpu_times)
        std_gpu_time = np.std(gpu_times)
        min_gpu_time = np.min(gpu_times)
        max_gpu_time = np.max(gpu_times)

        # Use the best result for final metrics
        best_idx = np.argmin(cpu_times)
        best_result = results[best_idx]

        return {
            "optimizer_name": optimizer_name,
            "cpu_times": cpu_times,
            "gpu_times": gpu_times,
            "avg_cpu_time": avg_cpu_time,
            "std_cpu_time": std_cpu_time,
            "min_cpu_time": min_cpu_time,
            "max_cpu_time": max_cpu_time,
            "avg_gpu_time": avg_gpu_time,
            "std_gpu_time": std_gpu_time,
            "min_gpu_time": min_gpu_time,
            "max_gpu_time": max_gpu_time,
            "fps_cpu": 1.0 / avg_cpu_time,
            "fps_gpu": 1.0 / avg_gpu_time,
            "result": best_result,
            "iterations": iterations,
            "runs": runs,
        }

    def _analyze_flops(self, optimizer, result, timing_stats: Dict) -> Dict:
        """Analyze FLOPS performance for the optimizer."""
        optimizer_name = timing_stats["optimizer_name"]

        if hasattr(result, "flop_info") and result.flop_info:
            # Dense optimizer has built-in FLOP analysis
            flop_info = result.flop_info.copy()

            # Recalculate based on our accurate timing
            actual_time = timing_stats["avg_gpu_time"]
            flop_info["actual_gflops_per_second"] = flop_info["total_flops"] / (
                actual_time * 1e9
            )
            flop_info["timing_source"] = "gpu_synchronized"

        else:
            # Sparse optimizer: estimate FLOPs
            optimizer_stats = optimizer.get_optimizer_stats()

            # Get matrix information
            if (
                hasattr(optimizer, "_A_combined_csr_cpu")
                and optimizer._A_combined_csr_cpu is not None
            ):
                matrix = optimizer._A_combined_csr_cpu
                nnz = matrix.nnz
                led_count = optimizer._actual_led_count

                # Estimate FLOPs for sparse operations per iteration:
                # 1. Sparse matrix-vector multiply: 2 * nnz (multiply + add per non-zero)
                # 2. Vector operations: ~led_count operations
                # 3. Other operations (gradients, etc.): ~2 * led_count

                flops_per_iteration = (
                    2 * nnz
                    + 4  # Sparse matrix-vector multiply
                    * led_count  # Vector operations and overhead
                )

                total_flops = flops_per_iteration * timing_stats["iterations"]
                actual_time = timing_stats["avg_gpu_time"]

                flop_info = {
                    "total_flops": int(total_flops),
                    "flops_per_iteration": int(flops_per_iteration),
                    "gflops": total_flops / 1e9,
                    "actual_gflops_per_second": total_flops / (actual_time * 1e9),
                    "matrix_nnz": nnz,
                    "sparsity_percent": (1 - nnz / (matrix.shape[0] * matrix.shape[1]))
                    * 100,
                    "timing_source": "gpu_synchronized",
                    "estimation_method": "sparse_matrix_analysis",
                }
            else:
                flop_info = {
                    "total_flops": 0,
                    "flops_per_iteration": 0,
                    "gflops": 0.0,
                    "actual_gflops_per_second": 0.0,
                    "timing_source": "gpu_synchronized",
                    "estimation_method": "unavailable",
                }

        return flop_info

    def run_comparison(self, iterations: int = 10, runs: int = 3) -> Dict:
        """Run comprehensive performance comparison."""
        logger.info("=" * 60)
        logger.info("LED OPTIMIZATION PERFORMANCE COMPARISON")
        logger.info("=" * 60)
        logger.info(f"Test image: {self.test_image.shape}")
        logger.info(f"Iterations per run: {iterations}")
        logger.info(f"Number of runs: {runs}")
        logger.info(f"Patterns: {self.patterns_path}")

        # Get optimizer information
        sparse_stats = self.sparse_optimizer.get_optimizer_stats()
        dense_stats = self.dense_optimizer.get_optimizer_stats()

        logger.info(f"Sparse LED count: {sparse_stats.get('led_count', 'N/A')}")
        logger.info(f"Dense LED count: {dense_stats.get('led_count', 'N/A')}")

        # Benchmark sparse optimizer
        logger.info("\n" + "=" * 40)
        logger.info("SPARSE OPTIMIZER BENCHMARK")
        logger.info("=" * 40)
        sparse_timing = self._measure_optimization_time(
            self.sparse_optimizer, iterations, runs
        )
        sparse_flops = self._analyze_flops(
            self.sparse_optimizer, sparse_timing["result"], sparse_timing
        )

        # Benchmark dense optimizer
        logger.info("\n" + "=" * 40)
        logger.info("DENSE OPTIMIZER BENCHMARK")
        logger.info("=" * 40)
        dense_timing = self._measure_optimization_time(
            self.dense_optimizer, iterations, runs
        )
        dense_flops = self._analyze_flops(
            self.dense_optimizer, dense_timing["result"], dense_timing
        )

        return {
            "sparse": {
                "timing": sparse_timing,
                "flops": sparse_flops,
                "optimizer_stats": sparse_stats,
            },
            "dense": {
                "timing": dense_timing,
                "flops": dense_flops,
                "optimizer_stats": dense_stats,
            },
            "test_config": {
                "iterations": iterations,
                "runs": runs,
                "patterns_path": self.patterns_path,
                "test_image_shape": self.test_image.shape,
            },
        }

    def print_detailed_report(self, comparison_results: Dict):
        """Print comprehensive performance report."""
        sparse = comparison_results["sparse"]
        dense = comparison_results["dense"]
        config = comparison_results["test_config"]

        print("\n" + "=" * 80)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 80)

        print(f"Test Configuration:")
        print(f"  • Patterns: {config['patterns_path']}")
        print(f"  • Image size: {config['test_image_shape']}")
        print(f"  • Iterations per run: {config['iterations']}")
        print(f"  • Number of runs: {config['runs']}")

        # Timing comparison
        print(f"\n📊 TIMING RESULTS (GPU-synchronized):")
        print(f"  Sparse Optimizer:")
        print(
            f"    • Average time: {sparse['timing']['avg_gpu_time']:.3f}s ± "
            f"{sparse['timing']['std_gpu_time']:.3f}s"
        )
        print(
            f"    • Range: {sparse['timing']['min_gpu_time']:.3f}s - "
            f"{sparse['timing']['max_gpu_time']:.3f}s"
        )
        print(f"    • Estimated FPS: {sparse['timing']['fps_gpu']:.1f}")

        print(f"  Dense Optimizer:")
        print(
            f"    • Average time: {dense['timing']['avg_gpu_time']:.3f}s ± "
            f"{dense['timing']['std_gpu_time']:.3f}s"
        )
        print(
            f"    • Range: {dense['timing']['min_gpu_time']:.3f}s - "
            f"{dense['timing']['max_gpu_time']:.3f}s"
        )
        print(f"    • Estimated FPS: {dense['timing']['fps_gpu']:.1f}")

        # Performance comparison
        speedup = sparse["timing"]["avg_gpu_time"] / dense["timing"]["avg_gpu_time"]
        fps_improvement = dense["timing"]["fps_gpu"] / sparse["timing"]["fps_gpu"]

        print(f"\n🚀 PERFORMANCE COMPARISON:")
        print(f"    • Dense speedup: {speedup:.2f}x faster")
        print(f"    • FPS improvement: {fps_improvement:.2f}x")
        print(f"    • Time reduction: {(1 - 1/speedup) * 100:.1f}%")

        # FLOPS analysis
        print(f"\n⚡ FLOPS ANALYSIS:")
        print(f"  Sparse Optimizer:")
        print(f"    • Total FLOPs: {sparse['flops']['total_flops']:,}")
        print(f"    • FLOPs/iteration: {sparse['flops']['flops_per_iteration']:,}")
        print(f"    • GFLOPS/second: {sparse['flops']['actual_gflops_per_second']:.1f}")
        if "sparsity_percent" in sparse["flops"]:
            print(f"    • Matrix sparsity: {sparse['flops']['sparsity_percent']:.1f}%")

        print(f"  Dense Optimizer:")
        print(f"    • Total FLOPs: {dense['flops']['total_flops']:,}")
        print(
            f"    • FLOPs/iteration (dense loop): {dense['flops']['flops_per_iteration']:,}"
        )
        if "atb_flops_per_frame" in dense["flops"]:
            print(f"    • A^T*b FLOPs/frame: {dense['flops']['atb_flops_per_frame']:,}")
            print(f"    • Dense loop FLOPs: {dense['flops']['dense_loop_flops']:,}")
        print(f"    • GFLOPS/second: {dense['flops']['actual_gflops_per_second']:.1f}")

        # Detailed timing breakdown for dense optimizer
        if "detailed_timing" in dense["flops"]:
            timing = dense["flops"]["detailed_timing"]
            print(f"    • Detailed timing breakdown:")
            atb_time = timing.get("atb_time", 0)
            total_time = timing.get("total_optimization_time", 1)
            print(
                f"      - A^T*b calculation: {atb_time:.4f}s "
                f"({atb_time/total_time*100:.1f}%)"
            )
            loop_time = timing.get("loop_time", 0)
            print(
                f"      - Optimization loop: {loop_time:.4f}s "
                f"({loop_time/total_time*100:.1f}%)"
            )
            einsum_time = timing.get("einsum_time", 0)
            print(
                f"        • Gradient einsum (ATA@x): {einsum_time:.4f}s "
                f"({einsum_time/total_time*100:.1f}%)"
            )
            step_size_time = timing.get("step_size_time", 0)
            print(
                f"        • Step size total: {step_size_time:.4f}s "
                f"({step_size_time/total_time*100:.1f}%)"
            )
            if "step_einsum_time" in timing:
                step_einsum_time = timing.get("step_einsum_time", 0)
                print(
                    f"          - Step einsum (g^T@ATA@g): {step_einsum_time:.4f}s "
                    f"({step_einsum_time/total_time*100:.1f}%)"
                )
                step_overhead = timing.get("step_size_time", 0) - timing.get(
                    "step_einsum_time", 0
                )
                print(
                    f"          - Step overhead: {step_overhead:.4f}s "
                    f"({step_overhead/total_time*100:.1f}%)"
                )
            print(f"      - Iterations: {timing.get('iterations_completed', 'N/A')}")

        # Efficiency comparison
        flops_efficiency = (
            dense["flops"]["actual_gflops_per_second"]
            / sparse["flops"]["actual_gflops_per_second"]
        )
        print(f"\n    • FLOPS efficiency improvement: {flops_efficiency:.2f}x")

        # Quality comparison
        print(f"\n🎯 OPTIMIZATION QUALITY:")
        sparse_mse = sparse["timing"]["result"].error_metrics.get("mse", 0)
        dense_mse = dense["timing"]["result"].error_metrics.get("mse", 0)

        print(f"  Sparse MSE: {sparse_mse:.6f}")
        print(f"  Dense MSE: {dense_mse:.6f}")
        print(
            f"  Quality difference: {abs(sparse_mse - dense_mse):.2e} (lower is better)"
        )

        # Hardware utilization
        print(f"\n💻 HARDWARE INFORMATION:")
        device_info_sparse = sparse["optimizer_stats"].get("device_info", {})
        device_info_dense = dense["optimizer_stats"].get("device_info", {})

        if device_info_sparse:
            print(f"  GPU: {device_info_sparse.get('gpu_name', 'Unknown')}")
            if "memory_info" in device_info_sparse:
                mem_info = device_info_sparse["memory_info"]
                print(f"  GPU Memory: {mem_info.get('total_mb', 0):.0f}MB total")

        # Memory usage comparison
        if "ata_memory_mb" in dense["optimizer_stats"]:
            print(
                f"  Dense ATA memory: {dense['optimizer_stats']['ata_memory_mb']:.1f}MB"
            )

        # Real-time performance assessment
        print(f"\n🎮 REAL-TIME PERFORMANCE ASSESSMENT:")
        target_fps = 15
        sparse_realtime = (
            "✅ YES" if sparse["timing"]["fps_gpu"] >= target_fps else "❌ NO"
        )
        dense_realtime = "✅ YES" if dense["timing"]["fps_gpu"] >= target_fps else "❌ NO"

        print(f"  Target: {target_fps} FPS for real-time operation")
        print(
            f"  Sparse achieves target: {sparse_realtime} ({sparse['timing']['fps_gpu']:.1f} FPS)"
        )
        print(
            f"  Dense achieves target: {dense_realtime} ({dense['timing']['fps_gpu']:.1f} FPS)"
        )

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="LED Optimization Performance Comparison"
    )
    parser.add_argument(
        "--patterns",
        "-p",
        required=True,
        help="Diffusion patterns file path (without .npz extension)",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="Optimization iterations per run (default: 10)",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)",
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

    # Validate patterns file
    patterns_file = f"{args.patterns}.npz"
    if not Path(patterns_file).exists():
        logger.error(f"Patterns file not found: {patterns_file}")
        return 1

    try:
        # Run comprehensive benchmark
        benchmark = OptimizerBenchmark(args.patterns)
        results = benchmark.run_comparison(iterations=args.iterations, runs=args.runs)
        benchmark.print_detailed_report(results)

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
