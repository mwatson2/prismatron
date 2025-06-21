#!/usr/bin/env python3
"""
Compare CuPy vs PyTorch sparse LED optimization performance.

This tool runs the same optimization problem with both implementations
and compares their performance characteristics.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.const import FRAME_HEIGHT, FRAME_WIDTH
from src.consumer.led_optimizer import LEDOptimizer as CuPyOptimizer
from src.consumer.led_optimizer_pytorch import PyTorchLEDOptimizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_image(seed: int = 42) -> np.ndarray:
    """Create a deterministic test image for consistent comparison."""
    np.random.seed(seed)

    # Create a test image with some structure
    image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    # Add some gradients and patterns
    y, x = np.ogrid[:FRAME_HEIGHT, :FRAME_WIDTH]

    # Red gradient
    image[:, :, 0] = (x / FRAME_WIDTH * 255).astype(np.uint8)

    # Green gradient
    image[:, :, 1] = (y / FRAME_HEIGHT * 255).astype(np.uint8)

    # Blue checkerboard
    checker = ((x // 50) + (y // 50)) % 2
    image[:, :, 2] = (checker * 255).astype(np.uint8)

    # Add some noise
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def run_cupy_optimization(
    patterns_path: str, test_image: np.ndarray, num_runs: int = 3
):
    """Run CuPy sparse optimization and return timing results."""
    logger.info("=== Testing CuPy Sparse Optimizer ===")

    optimizer = CuPyOptimizer(diffusion_patterns_path=patterns_path)

    if not optimizer.initialize():
        logger.error("Failed to initialize CuPy optimizer")
        return None

    stats = optimizer.get_optimizer_stats()
    logger.info(f"CuPy matrix shape: {stats.get('matrix_shape', 'N/A')}")
    logger.info(f"CuPy nnz: {stats.get('matrix_nnz', 'N/A'):,}")

    # Warm up
    logger.info("CuPy warm-up run...")
    optimizer.optimize_frame(test_image, max_iterations=5)

    # Timed runs
    times = []
    results = []

    for i in range(num_runs):
        logger.info(f"CuPy run {i+1}/{num_runs}")
        start_time = time.time()
        result = optimizer.optimize_frame(test_image, max_iterations=10)
        end_time = time.time()

        times.append(end_time - start_time)
        results.append(result)

        logger.info(f"  Time: {times[-1]:.3f}s")
        logger.info(f"  MSE: {result.error_metrics.get('mse', 'N/A'):.6f}")
        logger.info(
            f"  GFLOPS/s: {result.flop_info.get('gflops_per_second', 'N/A'):.1f}"
        )

    avg_time = np.mean(times)
    avg_gflops = np.mean([r.flop_info.get("gflops_per_second", 0) for r in results])
    avg_mse = np.mean([r.error_metrics.get("mse", float("inf")) for r in results])

    logger.info(
        f"CuPy Average: {avg_time:.3f}s, {avg_gflops:.1f} GFLOPS/s, MSE: {avg_mse:.6f}"
    )

    return {
        "optimizer_type": "CuPy",
        "times": times,
        "average_time": avg_time,
        "average_gflops": avg_gflops,
        "average_mse": avg_mse,
        "results": results,
        "stats": stats,
    }


def run_pytorch_optimization(
    patterns_path: str, test_image: np.ndarray, num_runs: int = 3
):
    """Run PyTorch sparse optimization and return timing results."""
    logger.info("=== Testing PyTorch Sparse Optimizer ===")

    optimizer = PyTorchLEDOptimizer(
        diffusion_patterns_path=patterns_path, device="cuda"
    )

    if not optimizer.initialize():
        logger.error("Failed to initialize PyTorch optimizer")
        return None

    stats = optimizer.get_optimizer_stats()
    logger.info(
        f"PyTorch tensor shape per channel: {stats.get('tensor_shape_per_channel', 'N/A')}"
    )
    nnz_per_channel = stats.get("nnz_per_channel", "N/A")
    total_nnz = stats.get("total_nnz", "N/A")
    logger.info(
        f"PyTorch nnz per channel: {nnz_per_channel:,}"
        if isinstance(nnz_per_channel, int)
        else f"PyTorch nnz per channel: {nnz_per_channel}"
    )
    logger.info(
        f"PyTorch total nnz: {total_nnz:,}"
        if isinstance(total_nnz, int)
        else f"PyTorch total nnz: {total_nnz}"
    )

    # Warm up
    logger.info("PyTorch warm-up run...")
    optimizer.optimize_frame(test_image, max_iterations=5)

    # Timed runs
    times = []
    results = []

    for i in range(num_runs):
        logger.info(f"PyTorch run {i+1}/{num_runs}")
        start_time = time.time()
        result = optimizer.optimize_frame(test_image, max_iterations=10)
        end_time = time.time()

        times.append(end_time - start_time)
        results.append(result)

        logger.info(f"  Time: {times[-1]:.3f}s")
        logger.info(f"  MSE: {result.error_metrics.get('mse', 'N/A'):.6f}")
        logger.info(
            f"  GFLOPS/s: {result.flop_info.get('gflops_per_second', 'N/A'):.1f}"
        )

    avg_time = np.mean(times)
    avg_gflops = np.mean([r.flop_info.get("gflops_per_second", 0) for r in results])
    avg_mse = np.mean([r.error_metrics.get("mse", float("inf")) for r in results])

    logger.info(
        f"PyTorch Average: {avg_time:.3f}s, {avg_gflops:.1f} GFLOPS/s, MSE: {avg_mse:.6f}"
    )

    return {
        "optimizer_type": "PyTorch",
        "times": times,
        "average_time": avg_time,
        "average_gflops": avg_gflops,
        "average_mse": avg_mse,
        "results": results,
        "stats": stats,
    }


def compare_results(cupy_results, pytorch_results):
    """Compare and summarize the optimization results."""
    logger.info("=== Performance Comparison ===")

    if cupy_results is None or pytorch_results is None:
        logger.error("Cannot compare - one or both optimizers failed")
        return

    cupy_time = cupy_results["average_time"]
    pytorch_time = pytorch_results["average_time"]

    cupy_gflops = cupy_results["average_gflops"]
    pytorch_gflops = pytorch_results["average_gflops"]

    cupy_mse = cupy_results["average_mse"]
    pytorch_mse = pytorch_results["average_mse"]

    speedup = cupy_time / pytorch_time if pytorch_time > 0 else float("inf")
    gflops_improvement = (
        pytorch_gflops / cupy_gflops if cupy_gflops > 0 else float("inf")
    )

    logger.info(f"Timing Results:")
    logger.info(f"  CuPy:     {cupy_time:.3f}s")
    logger.info(f"  PyTorch:  {pytorch_time:.3f}s")
    logger.info(
        f"  Speedup:  {speedup:.2f}x {'(PyTorch faster)' if speedup > 1 else '(CuPy faster)'}"
    )

    logger.info(f"GFLOPS Performance:")
    logger.info(f"  CuPy:     {cupy_gflops:.1f} GFLOPS/s")
    logger.info(f"  PyTorch:  {pytorch_gflops:.1f} GFLOPS/s")
    logger.info(
        f"  Improvement: {gflops_improvement:.2f}x "
        f"{'(PyTorch better)' if gflops_improvement > 1 else '(CuPy better)'}"
    )

    logger.info(f"Optimization Quality:")
    logger.info(f"  CuPy MSE:     {cupy_mse:.6f}")
    logger.info(f"  PyTorch MSE:  {pytorch_mse:.6f}")
    logger.info(
        f"  Quality ratio: {cupy_mse/pytorch_mse:.3f} "
        f"{'(PyTorch better)' if pytorch_mse < cupy_mse else '(CuPy better)'}"
    )

    # Frame rate estimates
    cupy_fps = 1.0 / cupy_time
    pytorch_fps = 1.0 / pytorch_time

    logger.info(f"Estimated Frame Rates:")
    logger.info(f"  CuPy:     {cupy_fps:.1f} FPS")
    logger.info(f"  PyTorch:  {pytorch_fps:.1f} FPS")


def main():
    """Main comparison function."""
    patterns_path = "diffusion_patterns/synthetic_1000"

    if not Path(f"{patterns_path}.npz").exists():
        logger.error(f"Patterns file not found: {patterns_path}.npz")
        logger.error(
            "Generate patterns first with: python tools/generate_synthetic_patterns.py"
        )
        return 1

    # Create test image
    logger.info("Creating test image...")
    test_image = create_test_image()
    logger.info(f"Test image shape: {test_image.shape}")

    # Run both optimizers
    cupy_results = run_cupy_optimization(patterns_path, test_image, num_runs=3)
    pytorch_results = run_pytorch_optimization(patterns_path, test_image, num_runs=3)

    # Compare results
    compare_results(cupy_results, pytorch_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
