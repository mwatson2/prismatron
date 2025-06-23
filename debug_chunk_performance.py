#!/usr/bin/env python3
"""
Test different chunk sizes to optimize SingleBlockMixedSparseTensor performance.
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


def create_realistic_tensor():
    """Create realistic tensor for performance testing."""
    logger.info(
        "Creating realistic tensor: 1000 LEDs, 3 channels, 480x800, 64x64 blocks"
    )

    batch_size = 1000
    channels = 3
    height = 480
    width = 800
    block_size = 64

    tensor = SingleBlockMixedSparseTensor(
        batch_size, channels, height, width, block_size
    )

    # Set all blocks with random realistic patterns
    np.random.seed(42)

    for led_id in range(batch_size):
        for channel in range(channels):
            # Random position
            top_row = np.random.randint(0, height - block_size)
            top_col = np.random.randint(0, width - block_size)

            # Gaussian-like pattern
            y, x = np.meshgrid(
                np.arange(block_size), np.arange(block_size), indexing="ij"
            )
            center_y, center_x = block_size // 2, block_size // 2

            sigma = np.random.uniform(8.0, 16.0)
            intensity = np.random.uniform(0.3, 1.0)

            pattern = intensity * np.exp(
                -((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * sigma**2)
            )
            pattern = pattern.astype(np.float32)

            tensor.set_block(led_id, channel, top_row, top_col, cp.asarray(pattern))

    logger.info(f"Created tensor with {cp.sum(tensor.blocks_set)} blocks")
    return tensor


def benchmark_chunk_sizes(tensor, chunk_sizes, num_runs=5):
    """Benchmark different chunk sizes."""
    target = cp.random.rand(tensor.height, tensor.width).astype(cp.float32)

    results = {}

    for chunk_size in chunk_sizes:
        logger.info(f"\nTesting chunk size: {chunk_size}")

        # Warm up
        _ = tensor.transpose_dot_product(target, chunk_size=chunk_size)
        cp.cuda.Device().synchronize()

        # Time multiple runs
        times = []
        for i in range(num_runs):
            cp.cuda.Device().synchronize()
            start_time = time.time()

            result = tensor.transpose_dot_product(target, chunk_size=chunk_size)

            cp.cuda.Device().synchronize()
            end_time = time.time()

            times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        num_chunks = (
            tensor.batch_size * tensor.channels + chunk_size - 1
        ) // chunk_size

        logger.info(
            f"  Chunk size {chunk_size}: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms"
        )
        logger.info(f"  Number of chunks: {num_chunks}")
        logger.info(f"  Time per chunk: {avg_time/num_chunks*1000:.3f}ms")

        results[chunk_size] = {
            "avg_time": avg_time,
            "std_time": std_time,
            "num_chunks": num_chunks,
            "result": result,
        }

    return results


def test_no_chunking_approach(tensor):
    """Test approach without chunking - process all blocks at once."""
    logger.info("\nTesting no-chunking approach...")

    target = cp.random.rand(tensor.height, tensor.width).astype(cp.float32)

    # Warm up
    _ = tensor.transpose_dot_product(
        target, chunk_size=tensor.batch_size * tensor.channels
    )
    cp.cuda.Device().synchronize()

    # Time multiple runs
    times = []
    for i in range(5):
        cp.cuda.Device().synchronize()
        start_time = time.time()

        # Process all at once
        result = tensor.transpose_dot_product(
            target, chunk_size=tensor.batch_size * tensor.channels
        )

        cp.cuda.Device().synchronize()
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    logger.info(f"  No chunking: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")

    return avg_time, result


def main():
    """Test chunk size optimization."""
    logger.info("Chunk Size Performance Optimization")
    logger.info("=" * 50)

    # Create test tensor
    tensor = create_realistic_tensor()

    # Test different chunk sizes
    chunk_sizes = [32, 64, 128, 256, 512, 1024, 2048]
    results = benchmark_chunk_sizes(tensor, chunk_sizes)

    # Test no chunking
    no_chunk_time, no_chunk_result = test_no_chunking_approach(tensor)

    # Find best chunk size
    best_chunk_size = min(results.keys(), key=lambda k: results[k]["avg_time"])
    best_time = results[best_chunk_size]["avg_time"]

    logger.info(f"\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)

    for chunk_size in sorted(results.keys()):
        data = results[chunk_size]
        speedup = best_time / data["avg_time"]
        logger.info(
            f"Chunk {chunk_size:4d}: {data['avg_time']*1000:6.2f}ms ({speedup:.2f}x)"
        )

    no_chunk_speedup = best_time / no_chunk_time
    logger.info(f"No chunk : {no_chunk_time*1000:6.2f}ms ({no_chunk_speedup:.2f}x)")

    logger.info(f"\nBest chunk size: {best_chunk_size}")
    logger.info(f"Best time: {best_time*1000:.2f}ms")

    if no_chunk_time < best_time:
        logger.info("✓ No chunking is fastest!")

    # Verify all results are consistent
    logger.info("\nVerifying result consistency...")
    base_result = results[chunk_sizes[0]]["result"]

    all_consistent = True
    for chunk_size in chunk_sizes[1:]:
        diff = cp.max(cp.abs(results[chunk_size]["result"] - base_result))
        if diff > 1e-5:
            logger.warning(f"Chunk {chunk_size} differs by {diff:.6f}")
            all_consistent = False

    no_chunk_diff = cp.max(cp.abs(no_chunk_result - base_result))
    if no_chunk_diff > 1e-5:
        logger.warning(f"No-chunk differs by {no_chunk_diff:.6f}")
        all_consistent = False

    if all_consistent:
        logger.info("✓ All chunk sizes produce consistent results")


if __name__ == "__main__":
    main()
