"""
Performance benchmark for batch transpose dot product operations.

This script compares the performance of batched A^T @ B operations vs repeated
single-frame calls to measure the speedup achieved by the shared memory optimization.
"""

import logging
import time
from typing import Any, Dict, List

import cupy as cp
import numpy as np

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


class BatchTransposeDotProductBenchmark:
    """Benchmark suite for batch transpose dot product operations."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.results = []

        # Benchmark configurations - realistic LED system sizes
        self.benchmark_configs = [
            {
                "name": "Small_System_128x128",
                "batch_size": 64,  # 64 LEDs
                "channels": 3,  # RGB
                "height": 128,  # Small frame
                "width": 128,
                "block_size": 32,  # 32x32 LED patterns
                "batch_frames": [1, 2, 4, 8, 16, 32],  # Test different batch sizes
                "dtype": cp.float32,
            },
            {
                "name": "Medium_System_256x256",
                "batch_size": 256,  # 256 LEDs
                "channels": 3,  # RGB
                "height": 256,  # Medium frame
                "width": 256,
                "block_size": 64,  # 64x64 LED patterns
                "batch_frames": [1, 2, 4, 8, 16],  # Fewer batch sizes for larger system
                "dtype": cp.float32,
            },
            {
                "name": "Large_System_512x512",
                "batch_size": 512,  # 512 LEDs
                "channels": 3,  # RGB
                "height": 512,  # Large frame
                "width": 512,
                "block_size": 64,  # 64x64 LED patterns
                "batch_frames": [1, 2, 4, 8],  # Even fewer for very large system
                "dtype": cp.float32,
            },
            {
                "name": "uint8_Medium_System",
                "batch_size": 256,  # 256 LEDs
                "channels": 3,  # RGB
                "height": 256,  # Medium frame
                "width": 256,
                "block_size": 64,  # 64x64 LED patterns
                "batch_frames": [1, 2, 4, 8, 16],  # Test uint8 performance
                "dtype": cp.uint8,
            },
        ]

    def create_benchmark_tensor(self, config: Dict[str, Any]) -> SingleBlockMixedSparseTensor:
        """Create a test tensor for benchmarking."""
        tensor = SingleBlockMixedSparseTensor(
            batch_size=config["batch_size"],
            channels=config["channels"],
            height=config["height"],
            width=config["width"],
            block_size=config["block_size"],
            dtype=config["dtype"],
            output_dtype=cp.float32,
        )

        # Set random blocks at random positions (optimized for reproducibility)
        np.random.seed(12345)  # Fixed seed for consistent benchmarks
        for led_idx in range(config["batch_size"]):
            for channel_idx in range(config["channels"]):
                # Generate random position ensuring block fits in frame
                max_row = config["height"] - config["block_size"]
                max_col = config["width"] - config["block_size"]
                row = np.random.randint(0, max_row)
                col = np.random.randint(0, max_col)

                # Align positions for vectorization (multiple of 4)
                col = (col // 4) * 4

                # Generate random block data
                if config["dtype"] == cp.float32:
                    block_data = (
                        cp.random.rand(config["block_size"], config["block_size"], dtype=cp.float32) * 0.8 + 0.1
                    )
                else:  # uint8
                    block_data = cp.random.randint(0, 256, (config["block_size"], config["block_size"]), dtype=cp.uint8)

                tensor.set_block(led_idx, channel_idx, row, col, block_data)

        return tensor

    def create_benchmark_target_batch(self, config: Dict[str, Any], batch_frames: int) -> cp.ndarray:
        """Create target frames for benchmarking."""
        np.random.seed(54321)  # Different fixed seed for target data

        if config["dtype"] == cp.float32:
            target_batch = (
                cp.random.rand(batch_frames, config["channels"], config["height"], config["width"], dtype=cp.float32)
                * 0.8
                + 0.1
            )
        else:  # uint8
            target_batch = cp.random.randint(
                0, 256, (batch_frames, config["channels"], config["height"], config["width"]), dtype=cp.uint8
            )

        return target_batch

    def warm_up_gpu(self, config: Dict[str, Any]):
        """Warm up GPU with a few operations to ensure consistent timing."""
        logger.info(f"Warming up GPU for {config['name']}...")

        tensor = self.create_benchmark_tensor(config)
        target_batch = self.create_benchmark_target_batch(config, 2)  # Small warm-up batch

        # Perform a few operations to warm up
        for _ in range(3):
            _ = tensor.transpose_dot_product_3d_batch(target_batch)
            _ = tensor.transpose_dot_product_3d(target_batch[0])

        # Ensure all operations complete
        cp.cuda.Device().synchronize()

    def benchmark_single_vs_batch(
        self, config: Dict[str, Any], batch_frames: int, num_trials: int = 5
    ) -> Dict[str, Any]:
        """Benchmark single operations vs batch operations."""
        logger.info(f"Benchmarking {config['name']} with {batch_frames} frames...")

        # Create test data
        tensor = self.create_benchmark_tensor(config)
        target_batch = self.create_benchmark_target_batch(config, batch_frames)

        # Ensure data is on GPU and ready
        cp.cuda.Device().synchronize()

        # Benchmark batch operation
        batch_times = []
        for trial in range(num_trials):
            start_time = time.perf_counter()

            batch_result = tensor.transpose_dot_product_3d_batch(target_batch, planar_output=False)
            cp.cuda.Device().synchronize()  # Ensure kernel completion

            end_time = time.perf_counter()
            batch_times.append(end_time - start_time)

        # Benchmark repeated single operations
        single_times = []
        for trial in range(num_trials):
            start_time = time.perf_counter()

            individual_results = []
            for frame_idx in range(batch_frames):
                frame_result = tensor.transpose_dot_product_3d(target_batch[frame_idx], planar_output=False)
                individual_results.append(frame_result)
            cp.cuda.Device().synchronize()  # Ensure all kernels complete

            end_time = time.perf_counter()
            single_times.append(end_time - start_time)

        # Calculate statistics
        batch_mean = np.mean(batch_times)
        batch_std = np.std(batch_times)
        single_mean = np.mean(single_times)
        single_std = np.std(single_times)

        speedup = single_mean / batch_mean if batch_mean > 0 else 0

        # Calculate throughput (frames per second)
        batch_fps = batch_frames / batch_mean
        single_fps = batch_frames / single_mean

        return {
            "config_name": config["name"],
            "batch_frames": batch_frames,
            "batch_size": config["batch_size"],
            "channels": config["channels"],
            "frame_size": f"{config['height']}x{config['width']}",
            "block_size": f"{config['block_size']}x{config['block_size']}",
            "dtype": str(config["dtype"]),
            "batch_time_mean": batch_mean,
            "batch_time_std": batch_std,
            "single_time_mean": single_mean,
            "single_time_std": single_std,
            "speedup": speedup,
            "batch_fps": batch_fps,
            "single_fps": single_fps,
            "num_trials": num_trials,
        }

    def run_benchmarks(self):
        """Run all benchmark configurations."""
        logger.info("Starting batch transpose dot product benchmarks...")

        for config in self.benchmark_configs:
            logger.info(f"\n=== Benchmarking {config['name']} ===")

            # Warm up GPU for this configuration
            self.warm_up_gpu(config)

            # Test each batch size
            for batch_frames in config["batch_frames"]:
                try:
                    result = self.benchmark_single_vs_batch(config, batch_frames)
                    self.results.append(result)

                    logger.info(
                        f"  Batch {batch_frames:2d}: {result['speedup']:.2f}x speedup "
                        f"({result['batch_time_mean']*1000:.1f}ms vs {result['single_time_mean']*1000:.1f}ms)"
                    )

                except Exception as e:
                    logger.error(f"  Failed batch {batch_frames}: {e}")

    def print_detailed_results(self):
        """Print detailed benchmark results."""
        print("\n" + "=" * 120)
        print("DETAILED BATCH TRANSPOSE DOT PRODUCT BENCHMARK RESULTS")
        print("=" * 120)

        print(
            f"{'Config':<25} {'Frames':<7} {'LEDs':<5} {'Frame Size':<10} {'Block':<8} {'Dtype':<8} {'Speedup':<8} {'Batch FPS':<10} {'Single FPS':<11} {'Batch Time':<12} {'Single Time':<12}"
        )
        print("-" * 120)

        for result in self.results:
            print(
                f"{result['config_name']:<25} "
                f"{result['batch_frames']:<7} "
                f"{result['batch_size']:<5} "
                f"{result['frame_size']:<10} "
                f"{result['block_size']:<8} "
                f"{result['dtype']:<8} "
                f"{result['speedup']:<8.2f} "
                f"{result['batch_fps']:<10.1f} "
                f"{result['single_fps']:<11.1f} "
                f"{result['batch_time_mean']*1000:<12.1f} "
                f"{result['single_time_mean']*1000:<12.1f}"
            )

    def print_summary_analysis(self):
        """Print summary analysis of benchmark results."""
        print("\n" + "=" * 80)
        print("SUMMARY ANALYSIS")
        print("=" * 80)

        # Group results by configuration
        configs = {}
        for result in self.results:
            config_name = result["config_name"]
            if config_name not in configs:
                configs[config_name] = []
            configs[config_name].append(result)

        for config_name, results in configs.items():
            print(f"\n{config_name}:")

            # Find best speedup
            best_speedup = max(r["speedup"] for r in results)
            best_result = next(r for r in results if r["speedup"] == best_speedup)

            print(f"  Best speedup: {best_speedup:.2f}x at {best_result['batch_frames']} frames")
            print(f"  Peak batch throughput: {max(r['batch_fps'] for r in results):.1f} FPS")

            # Analyze scaling
            batch_1_fps = next((r["batch_fps"] for r in results if r["batch_frames"] == 1), None)
            if batch_1_fps:
                max_batch_fps = max(r["batch_fps"] for r in results if r["batch_frames"] > 1)
                throughput_improvement = max_batch_fps / batch_1_fps
                print(f"  Throughput scaling: {throughput_improvement:.2f}x from batch=1 to optimal batch size")

        # Overall statistics
        all_speedups = [r["speedup"] for r in self.results]
        print(f"\nOverall Statistics:")
        print(f"  Average speedup: {np.mean(all_speedups):.2f}x")
        print(f"  Maximum speedup: {np.max(all_speedups):.2f}x")
        print(f"  Minimum speedup: {np.min(all_speedups):.2f}x")
        print(f"  Speedup std dev: {np.std(all_speedups):.2f}x")


def main():
    """Run the benchmark suite."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Initialize and run benchmarks
    benchmark = BatchTransposeDotProductBenchmark()
    benchmark.run_benchmarks()

    # Print results
    benchmark.print_detailed_results()
    benchmark.print_summary_analysis()


if __name__ == "__main__":
    main()
