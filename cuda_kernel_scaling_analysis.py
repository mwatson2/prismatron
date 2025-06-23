#!/usr/bin/env python3
"""
CUDA Kernel Scaling Analysis.

Analyzes the performance characteristics and scaling behavior of the
corrected CUDA kernel compared to the original flawed kernel.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cupy as cp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_gflops(
    led_count: int, channels: int, block_size: int, time_seconds: float
) -> float:
    """Calculate GFLOPS for the A^T @ b operation."""
    # Each (LED, channel) performs a dot product of block_size^2 elements
    # Each element: 1 multiply + 1 add = 2 FLOPs
    total_flops = led_count * channels * block_size * block_size * 2
    gflops = total_flops / (time_seconds * 1e9)
    return gflops


def create_synthetic_tensor(
    led_count: int, block_size: int = 96
) -> SingleBlockMixedSparseTensor:
    """Create synthetic tensor for performance testing."""
    logger.info(
        f"Creating synthetic tensor: {led_count} LEDs, {block_size}x{block_size} blocks..."
    )

    # Use realistic image size
    height, width, channels = 480, 800, 3

    tensor = SingleBlockMixedSparseTensor(
        led_count, channels, height, width, block_size
    )

    # Generate patterns with realistic sparsity
    np.random.seed(42)  # Reproducible

    for led_id in range(led_count):
        for channel in range(channels):
            # Random position
            top_row = np.random.randint(0, height - block_size)
            top_col = np.random.randint(0, width - block_size)

            # Create Gaussian diffusion pattern
            y, x = np.meshgrid(
                np.arange(block_size), np.arange(block_size), indexing="ij"
            )
            center_y, center_x = block_size // 2, block_size // 2

            sigma = np.random.uniform(15.0, 25.0)
            intensity = np.random.uniform(0.5, 1.0)

            pattern = intensity * np.exp(
                -((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * sigma**2)
            )

            # Apply sparsity (keep top 40%)
            threshold = np.percentile(pattern, 60)
            pattern[pattern < threshold] = 0

            tensor.set_block(
                led_id,
                channel,
                top_row,
                top_col,
                cp.asarray(pattern.astype(np.float32)),
            )

    memory_info = tensor.memory_info()
    logger.info(
        f"  Created {memory_info['blocks_stored']} blocks, {memory_info['total_mb']:.1f}MB"
    )

    return tensor


def benchmark_kernel_performance(
    tensor: SingleBlockMixedSparseTensor, num_runs: int = 20
) -> Dict:
    """Benchmark different kernel implementations."""
    led_count = tensor.batch_size
    logger.info(f"Benchmarking {led_count} LEDs with {num_runs} runs...")

    # Create test targets
    targets = []
    for _ in range(num_runs):
        target = cp.random.rand(tensor.height, tensor.width).astype(cp.float32)
        targets.append(target)

    results = {"led_count": led_count, "block_size": tensor.block_size}

    # Test chunked implementation
    try:
        # Warm up
        _ = tensor.transpose_dot_product(targets[0])
        cp.cuda.Device().synchronize()

        # Benchmark
        times = []
        for i in range(1, num_runs):
            cp.cuda.Device().synchronize()
            start_time = time.time()
            _ = tensor.transpose_dot_product(targets[i])
            cp.cuda.Device().synchronize()
            times.append(time.time() - start_time)

        chunked_time = np.mean(times)
        chunked_gflops = calculate_gflops(
            led_count, tensor.channels, tensor.block_size, chunked_time
        )

        results.update(
            {
                "chunked_available": True,
                "chunked_time_ms": chunked_time * 1000,
                "chunked_gflops": chunked_gflops,
                "chunked_fps": 1.0 / chunked_time,
            }
        )

        logger.info(
            f"  Chunked: {chunked_time*1000:.2f}ms, {chunked_gflops:.2f} GFLOPS"
        )

    except Exception as e:
        logger.warning(f"  Chunked failed: {e}")
        results["chunked_available"] = False

    # Test original CUDA kernel
    try:
        # Warm up
        _ = tensor.transpose_dot_product_cuda(targets[0])
        cp.cuda.Device().synchronize()

        # Benchmark
        times = []
        for i in range(1, num_runs):
            cp.cuda.Device().synchronize()
            start_time = time.time()
            _ = tensor.transpose_dot_product_cuda(targets[i])
            cp.cuda.Device().synchronize()
            times.append(time.time() - start_time)

        original_time = np.mean(times)
        original_gflops = calculate_gflops(
            led_count, tensor.channels, tensor.block_size, original_time
        )

        results.update(
            {
                "original_available": True,
                "original_time_ms": original_time * 1000,
                "original_gflops": original_gflops,
                "original_fps": 1.0 / original_time,
            }
        )

        if results["chunked_available"]:
            results["original_speedup"] = chunked_time / original_time

        logger.info(
            f"  Original: {original_time*1000:.2f}ms, {original_gflops:.2f} GFLOPS"
        )

    except Exception as e:
        logger.warning(f"  Original CUDA failed: {e}")
        results["original_available"] = False

    # Test corrected CUDA kernel
    try:
        # Warm up
        _ = tensor.transpose_dot_product_cuda_corrected(targets[0])
        cp.cuda.Device().synchronize()

        # Benchmark
        times = []
        for i in range(1, num_runs):
            cp.cuda.Device().synchronize()
            start_time = time.time()
            _ = tensor.transpose_dot_product_cuda_corrected(targets[i])
            cp.cuda.Device().synchronize()
            times.append(time.time() - start_time)

        corrected_time = np.mean(times)
        corrected_gflops = calculate_gflops(
            led_count, tensor.channels, tensor.block_size, corrected_time
        )

        results.update(
            {
                "corrected_available": True,
                "corrected_time_ms": corrected_time * 1000,
                "corrected_gflops": corrected_gflops,
                "corrected_fps": 1.0 / corrected_time,
            }
        )

        if results["chunked_available"]:
            results["corrected_speedup"] = chunked_time / corrected_time

        if results["original_available"]:
            results["corrected_vs_original"] = original_time / corrected_time

        logger.info(
            f"  Corrected: {corrected_time*1000:.2f}ms, {corrected_gflops:.2f} GFLOPS"
        )

    except Exception as e:
        logger.warning(f"  Corrected CUDA failed: {e}")
        results["corrected_available"] = False

    return results


def analyze_gpu_utilization(led_count: int, block_size: int = 96):
    """Analyze theoretical GPU utilization for the corrected kernel."""
    logger.info(f"\\nGPU UTILIZATION ANALYSIS for {led_count} LEDs:")

    # GPU specs (typical for NVIDIA Jetson/RTX)
    sms = 8  # Streaming Multiprocessors
    cores_per_sm = 32
    total_cores = sms * cores_per_sm

    # Corrected kernel design
    total_blocks = led_count * 3  # 3 channels
    threads_per_block = 256
    total_threads = total_blocks * threads_per_block

    # Calculate utilization
    theoretical_occupancy = min(1.0, total_threads / total_cores)
    blocks_per_sm = total_blocks / sms

    logger.info(f"  Total thread blocks: {total_blocks:,}")
    logger.info(f"  Total threads: {total_threads:,}")
    logger.info(f"  Blocks per SM: {blocks_per_sm:.1f}")
    logger.info(f"  Theoretical occupancy: {theoretical_occupancy:.1%}")

    # Work analysis
    work_per_thread = (block_size * block_size + 255) // 256  # Ceiling division
    total_operations = (
        led_count * 3 * block_size * block_size * 2
    )  # 2 FLOPs per element

    logger.info(f"  Work per thread: {work_per_thread} elements")
    logger.info(f"  Total operations: {total_operations:,} FLOPs")

    # Memory analysis
    sparse_memory = led_count * 3 * block_size * block_size * 4  # 4 bytes per float
    target_memory = 480 * 800 * 4  # Target image

    logger.info(f"  Sparse pattern memory: {sparse_memory / 1024**2:.1f}MB")
    logger.info(f"  Target image memory: {target_memory / 1024**2:.1f}MB")


def main():
    """Run comprehensive scaling analysis."""
    logger.info("CUDA Kernel Scaling Analysis")
    logger.info("=" * 60)

    # Test different LED counts with realistic block size
    led_counts = [50, 100, 250, 500, 750, 1000, 1500, 2000, 3000]
    block_size = 96  # Production block size

    all_results = []

    for led_count in led_counts:
        logger.info(f"\\n{'='*50}")
        logger.info(f"TESTING {led_count} LEDs")
        logger.info(f"{'='*50}")

        # Create synthetic tensor
        tensor = create_synthetic_tensor(led_count, block_size)

        # Analyze GPU utilization
        analyze_gpu_utilization(led_count, block_size)

        # Benchmark performance
        results = benchmark_kernel_performance(tensor)
        all_results.append(results)

    # Summary analysis
    logger.info(f"\\n{'='*60}")
    logger.info("SCALING ANALYSIS SUMMARY")
    logger.info(f"{'='*60}")

    # Performance table
    logger.info("\\nPERFORMANCE SCALING TABLE:")
    logger.info(
        "LEDs  | Chunked (ms) | Original (ms) | Corrected (ms) | Corrected GFLOPS | vs Original"
    )
    logger.info("-" * 90)

    for r in all_results:
        led_count = r["led_count"]

        chunked_str = (
            f"{r['chunked_time_ms']:.1f}" if r.get("chunked_available") else "N/A"
        )
        original_str = (
            f"{r['original_time_ms']:.1f}" if r.get("original_available") else "N/A"
        )
        corrected_str = (
            f"{r['corrected_time_ms']:.1f}" if r.get("corrected_available") else "N/A"
        )
        gflops_str = (
            f"{r['corrected_gflops']:.1f}" if r.get("corrected_available") else "N/A"
        )

        if r.get("corrected_vs_original"):
            if r["corrected_vs_original"] >= 1:
                vs_original = f"{r['corrected_vs_original']:.2f}x faster"
            else:
                vs_original = f"{1/r['corrected_vs_original']:.2f}x slower"
        else:
            vs_original = "N/A"

        logger.info(
            f"{led_count:4d} | {chunked_str:>11s} | {original_str:>12s} | "
            f"{corrected_str:>13s} | {gflops_str:>15s} | {vs_original}"
        )

    # Analyze trends
    corrected_results = [r for r in all_results if r.get("corrected_available")]

    if len(corrected_results) >= 2:
        gflops_values = [r["corrected_gflops"] for r in corrected_results]
        led_counts_values = [r["led_count"] for r in corrected_results]

        logger.info(f"\\nCORRECTED KERNEL SCALING ANALYSIS:")
        logger.info(
            f"  GFLOPS range: {min(gflops_values):.1f} - {max(gflops_values):.1f}"
        )
        peak_led_count = led_counts_values[gflops_values.index(max(gflops_values))]
        logger.info(
            f"  Peak performance: {max(gflops_values):.1f} GFLOPS at {peak_led_count} LEDs"
        )

        # Performance per LED
        gflops_per_led = [g / l for g, l in zip(gflops_values, led_counts_values)]
        logger.info(
            f"  GFLOPS per LED: {min(gflops_per_led):.4f} - {max(gflops_per_led):.4f}"
        )

        # Identify performance sweet spot
        sweet_spot_idx = gflops_values.index(max(gflops_values))
        sweet_spot_leds = led_counts_values[sweet_spot_idx]
        logger.info(f"  Performance sweet spot: {sweet_spot_leds} LEDs")

        # FPS analysis for real-time requirements
        logger.info(f"\\nREAL-TIME PERFORMANCE (15 FPS target):")
        for r in corrected_results:
            fps = r.get("corrected_fps", 0)
            real_time = "✅" if fps >= 15 else "❌"
            logger.info(f"  {r['led_count']:4d} LEDs: {fps:.1f} FPS {real_time}")

    # Memory scaling analysis
    logger.info(f"\\nMEMORY SCALING:")
    for r in all_results:
        led_count = r["led_count"]
        memory_mb = (
            led_count * 3 * block_size * block_size * 4 / (1024**2)
        )  # Sparse patterns
        logger.info(f"  {led_count:4d} LEDs: {memory_mb:.1f}MB sparse patterns")

    return 0


if __name__ == "__main__":
    sys.exit(main())
