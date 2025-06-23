#!/usr/bin/env python3
"""
High-Parallelism CUDA Kernel Performance Test.

Compares three approaches:
1. Chunked tensor (CPU-like parallelism)
2. Standard CUDA kernel (low parallelism)
3. High-parallelism CUDA kernel (maximum GPU utilization)
"""

import logging
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_tensor(
    led_count: int = 1000, block_size: int = 96
) -> SingleBlockMixedSparseTensor:
    """Create test tensor with realistic patterns."""
    logger.info(f"Creating test tensor for {led_count} LEDs...")

    height, width, channels = 480, 800, 3

    tensor = SingleBlockMixedSparseTensor(
        led_count, channels, height, width, block_size
    )

    # Generate realistic patterns
    np.random.seed(42)  # Reproducible results

    for led_id in range(led_count):
        for channel in range(channels):
            # Random position
            top_row = np.random.randint(0, height - block_size)
            top_col = np.random.randint(0, width - block_size)

            # Create Gaussian pattern
            y, x = np.meshgrid(
                np.arange(block_size), np.arange(block_size), indexing="ij"
            )
            center_y, center_x = block_size // 2, block_size // 2

            sigma = np.random.uniform(15.0, 25.0)
            intensity = np.random.uniform(0.5, 1.0)

            pattern = intensity * np.exp(
                -((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * sigma**2)
            )

            # Apply sparsity
            threshold = np.percentile(pattern, 60)  # Keep top 40%
            pattern[pattern < threshold] = 0

            tensor.set_block(
                led_id,
                channel,
                top_row,
                top_col,
                cp.asarray(pattern.astype(np.float32)),
            )

    memory_info = tensor.memory_info()
    logger.info(f"  Created {memory_info['blocks_stored']} blocks")
    logger.info(f"  Memory: {memory_info['total_mb']:.1f}MB")

    return tensor


def benchmark_approach(
    tensor: SingleBlockMixedSparseTensor,
    method_name: str,
    method_func,
    num_runs: int = 10,
):
    """Benchmark a specific approach."""
    logger.info(f"\n=== Benchmarking {method_name} ===")

    # Create test targets
    targets = []
    for _ in range(num_runs):
        target = cp.random.rand(tensor.height, tensor.width).astype(cp.float32)
        targets.append(target)

    # Test availability
    try:
        result = method_func(targets[0])
        available = True
        logger.info(f"  {method_name} available: Yes")
    except Exception as e:
        logger.warning(f"  {method_name} available: No - {e}")
        return {"available": False}

    # Warm up
    cp.cuda.Device().synchronize()

    # Benchmark
    times = []
    for i in range(1, num_runs):
        cp.cuda.Device().synchronize()
        start_time = time.time()

        result = method_func(targets[i])

        cp.cuda.Device().synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate theoretical GFLOPS
    # For 96x96 blocks with ~40% sparsity: ~3686 operations per block
    # 1000 LEDs × 3 channels × 3686 ops = ~11M operations
    estimated_ops = (
        tensor.batch_size
        * tensor.channels
        * (tensor.block_size * tensor.block_size * 0.4)
    )
    gflops = (2 * estimated_ops) / (avg_time * 1e9)  # 2 FLOPs per multiply-add

    logger.info(f"  Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    logger.info(f"  Min/Max time: {min_time*1000:.2f}ms / {max_time*1000:.2f}ms")
    logger.info(f"  GFLOPS: {gflops:.1f}")
    logger.info(f"  FPS: {1/avg_time:.1f}")

    return {
        "available": True,
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "gflops": gflops,
        "fps": 1 / avg_time,
        "result_shape": result.shape,
    }


def calculate_gpu_utilization_metrics(tensor: SingleBlockMixedSparseTensor):
    """Calculate expected GPU utilization for each approach."""
    logger.info(f"\n=== GPU UTILIZATION ANALYSIS ===")

    total_led_channels = tensor.batch_size * tensor.channels
    block_elements = tensor.block_size * tensor.block_size

    logger.info(f"Problem size:")
    logger.info(f"  LEDs: {tensor.batch_size}")
    logger.info(f"  Channels: {tensor.channels}")
    logger.info(f"  Total (LED, channel) pairs: {total_led_channels}")
    logger.info(
        f"  Block size: {tensor.block_size}×{tensor.block_size} = {block_elements} elements"
    )
    logger.info(
        f"  Total operations: {total_led_channels * block_elements * 2:,} FLOPs"
    )

    # Chunked approach
    logger.info(f"\nChunked Approach:")
    logger.info(f"  Parallelism: CPU-based chunking")
    logger.info(f"  GPU utilization: Low (sequential processing)")

    # Standard CUDA kernel
    logger.info(f"\nStandard CUDA Kernel:")
    threads_per_block_std = 32 * 3  # From original kernel config
    num_blocks_std = (tensor.batch_size + 31) // 32
    total_threads_std = threads_per_block_std * num_blocks_std
    logger.info(f"  Thread blocks: {threads_per_block_std} threads/block")
    logger.info(f"  Grid size: {num_blocks_std} blocks")
    logger.info(f"  Total threads: {total_threads_std:,}")
    logger.info(f"  Work per thread: {block_elements:,} operations")

    # High-parallelism CUDA kernel
    logger.info(f"\nHigh-Parallelism CUDA Kernel:")
    threads_per_block_hp = 256
    num_blocks_hp = total_led_channels
    total_threads_hp = threads_per_block_hp * num_blocks_hp
    work_per_thread_hp = (block_elements + 255) // 256  # Ceiling division
    logger.info(f"  Thread blocks: {threads_per_block_hp} threads/block")
    logger.info(f"  Grid size: {num_blocks_hp} blocks")
    logger.info(f"  Total threads: {total_threads_hp:,}")
    logger.info(f"  Work per thread: ~{work_per_thread_hp} operations")

    # Compute-optimized CUDA kernel
    logger.info(f"\nCompute-Optimized CUDA Kernel (Architecture-Matched):")
    sms_count = 8
    cores_per_sm = 32
    iterations_needed = (total_led_channels + 7) // 8
    total_threads_co = sms_count * cores_per_sm * iterations_needed
    work_per_thread_co = (block_elements + 31) // 32  # 288 for 96x96
    logger.info(f"  SMs utilized: {sms_count} (matches hardware)")
    logger.info(f"  Cores per SM: {cores_per_sm} (matches hardware)")
    logger.info(f"  Iterations: {iterations_needed}")
    logger.info(f"  Threads per iteration: {sms_count * cores_per_sm}")
    logger.info(f"  Total effective threads: {total_threads_co:,}")
    logger.info(f"  Work per thread: {work_per_thread_co} operations")
    logger.info(f"  Shared memory per SM: 72KB (sparse + target blocks)")

    logger.info(f"\nParallelism Comparison:")
    logger.info(
        f"  High-parallelism vs Standard: {total_threads_hp / total_threads_std:.1f}x more threads"
    )
    logger.info(
        f"  Compute-optimized vs Standard: {total_threads_co / total_threads_std:.1f}x effective threads"
    )
    logger.info(
        f"  Compute-optimized vs High-parallelism: {total_threads_co / total_threads_hp:.2f}x threads"
    )
    logger.info(
        f"  Work distribution (Compute-optimized): {block_elements / work_per_thread_co:.1f}x more parallel than standard"
    )


def main():
    """Run comprehensive performance comparison."""
    logger.info("High-Parallelism CUDA Kernel Performance Test")
    logger.info(
        "Comparing chunked, standard CUDA, and high-parallelism CUDA approaches"
    )
    logger.info("=" * 70)

    # Test different LED counts
    led_counts = [500, 1000, 2000]

    for led_count in led_counts:
        logger.info(f"\n{'='*70}")
        logger.info(f"TESTING {led_count} LEDs")
        logger.info(f"{'='*70}")

        # Create test tensor
        tensor = create_test_tensor(led_count)

        # Calculate theoretical metrics
        calculate_gpu_utilization_metrics(tensor)

        # Benchmark all approaches
        results = {}

        # 1. Chunked approach
        results["chunked"] = benchmark_approach(
            tensor,
            "Chunked Tensor",
            lambda target: tensor.transpose_dot_product(target),
        )

        # 2. Standard CUDA kernel
        results["standard_cuda"] = benchmark_approach(
            tensor,
            "Standard CUDA Kernel",
            lambda target: tensor.transpose_dot_product_cuda(target),
        )

        # 3. High-parallelism CUDA kernel
        results["high_parallelism"] = benchmark_approach(
            tensor,
            "High-Parallelism CUDA Kernel",
            lambda target: tensor.transpose_dot_product_cuda_high_parallelism(target),
        )

        # 4. Compute-optimized CUDA kernel
        results["compute_optimized"] = benchmark_approach(
            tensor,
            "Compute-Optimized CUDA Kernel",
            lambda target: tensor.transpose_dot_product_cuda_compute_optimized(target),
        )

        # Performance comparison
        logger.info(f"\n=== PERFORMANCE SUMMARY FOR {led_count} LEDs ===")

        available_results = {
            k: v for k, v in results.items() if v.get("available", False)
        }

        if len(available_results) >= 2:
            # Create comparison table
            logger.info(
                f"{'Method':<25} | {'Time (ms)':<10} | {'GFLOPS':<8} | {'FPS':<6} | {'Speedup':<8}"
            )
            logger.info("-" * 70)

            baseline_time = (
                available_results["chunked"]["avg_time_ms"]
                if "chunked" in available_results
                else None
            )

            for method, data in available_results.items():
                speedup = baseline_time / data["avg_time_ms"] if baseline_time else 1.0
                speedup_str = f"{speedup:.2f}x" if baseline_time else "baseline"

                logger.info(
                    f"{method:<25} | {data['avg_time_ms']:>8.2f} | {data['gflops']:>6.1f} | {data['fps']:>4.1f} | {speedup_str:>7s}"
                )

            # Highlight best performer
            best_method = min(
                available_results.keys(),
                key=lambda k: available_results[k]["avg_time_ms"],
            )
            best_gflops = available_results[best_method]["gflops"]

            logger.info(f"\nBest performer: {best_method}")
            logger.info(f"Peak GFLOPS: {best_gflops:.1f}")

            # Check if high-parallelism is available and beneficial
            if (
                "high_parallelism" in available_results
                and "standard_cuda" in available_results
            ):
                hp_improvement = (
                    available_results["standard_cuda"]["avg_time_ms"]
                    / available_results["high_parallelism"]["avg_time_ms"]
                )
                logger.info(
                    f"High-parallelism improvement over standard: {hp_improvement:.2f}x"
                )

        logger.info(f"\nResult verification:")
        for method, data in available_results.items():
            if data.get("result_shape"):
                logger.info(f"  {method}: result shape {data['result_shape']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
