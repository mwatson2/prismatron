#!/usr/bin/env python3
"""
Test V3 kernel with production 3008 LED configuration.
"""

import logging

import cupy as cp
import numpy as np

from src.utils.kernels.compute_optimized_3d_batch_v3 import (
    cuda_transpose_dot_product_3d_batch_v3_frames,
)
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_production_v3():
    """Test V3 kernel with production configuration."""

    # Production configuration
    config = {
        "batch_size": 3008,
        "channels": 3,
        "height": 800,
        "width": 480,
        "block_size": 64,
    }

    batch_frames_list = [1, 2, 4, 8, 16]

    print("=" * 100)
    print("V3 PRODUCTION TEST: 3008 LEDs")
    print("=" * 100)
    print(f"Configuration: {config['batch_size']} LEDs, {config['height']}x{config['width']} display")
    print("-" * 100)
    print(f"{'Frames':<8} {'Single (ms)':<12} {'V3 (ms)':<12} {'Speedup':<10} {'Single FPS':<12} {'V3 FPS':<12}")
    print("-" * 100)

    # Create tensor
    logger.info("Creating production tensor...")
    tensor = SingleBlockMixedSparseTensor(
        batch_size=config["batch_size"],
        channels=config["channels"],
        height=config["height"],
        width=config["width"],
        block_size=config["block_size"],
        dtype=cp.float32,
    )

    # Set patterns (sample subset for speed)
    np.random.seed(42)
    for led_idx in range(0, config["batch_size"], 10):  # Every 10th LED for speed
        for channel_idx in range(config["channels"]):
            max_row = config["height"] - config["block_size"]
            max_col = config["width"] - config["block_size"]
            row = np.random.randint(0, max_row)
            col = np.random.randint(0, max_col)
            col = (col // 4) * 4

            block_data = cp.random.rand(config["block_size"], config["block_size"], dtype=cp.float32) * 0.5 + 0.1
            tensor.set_block(led_idx, channel_idx, row, col, block_data)

    # Fill remaining with default patterns
    default_block = cp.ones((config["block_size"], config["block_size"]), dtype=cp.float32) * 0.1
    for led_idx in range(config["batch_size"]):
        for channel_idx in range(config["channels"]):
            if led_idx % 10 != 0:  # Fill in the gaps
                row = (led_idx * 37) % (config["height"] - config["block_size"])
                col = ((led_idx * 73) % (config["width"] - config["block_size"]) // 4) * 4
                tensor.set_block(led_idx, channel_idx, row, col, default_block)

    logger.info("Starting benchmarks...")

    for batch_frames in batch_frames_list:
        # Create target batch
        target_batch = (
            cp.random.rand(batch_frames, config["channels"], config["height"], config["width"], dtype=cp.float32) * 0.8
            + 0.1
        )

        # Warmup
        for _ in range(3):
            _ = tensor.transpose_dot_product_3d(target_batch[0])

        # Benchmark single-frame operations
        single_times = []
        for _ in range(5):  # Fewer runs for large config
            start = cp.cuda.Event()
            end = cp.cuda.Event()

            start.record()
            for frame_idx in range(batch_frames):
                _ = tensor.transpose_dot_product_3d(target_batch[frame_idx])
            end.record()
            end.synchronize()

            single_times.append(cp.cuda.get_elapsed_time(start, end))
        single_time = np.median(single_times)

        # Test V3 frame-parallel kernel
        try:
            # Check if we can fit in block size limit
            threads_needed = batch_frames * 32
            if threads_needed > 1024:
                print(f"{batch_frames:<8} {'SKIP':<12} {'SKIP':<12} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
                continue

            # Warmup
            for _ in range(3):
                _ = cuda_transpose_dot_product_3d_batch_v3_frames(
                    tensor.sparse_values,
                    tensor.block_positions,
                    target_batch,
                    config["batch_size"],
                    config["channels"],
                    batch_frames,
                    config["block_size"],
                    interleaved=True,
                )

            # Benchmark V3
            v3_times = []
            for _ in range(5):
                start = cp.cuda.Event()
                end = cp.cuda.Event()

                start.record()
                _ = cuda_transpose_dot_product_3d_batch_v3_frames(
                    tensor.sparse_values,
                    tensor.block_positions,
                    target_batch,
                    config["batch_size"],
                    config["channels"],
                    batch_frames,
                    config["block_size"],
                    interleaved=True,
                )
                end.record()
                end.synchronize()

                v3_times.append(cp.cuda.get_elapsed_time(start, end))
            v3_time = np.median(v3_times)

            # Calculate metrics
            speedup = single_time / v3_time
            single_fps = batch_frames / (single_time / 1000)
            v3_fps = batch_frames / (v3_time / 1000)

            marker = " ✓" if speedup > 1.0 else ""
            print(
                f"{batch_frames:<8} {single_time:>10.2f} {v3_time:>10.2f} {speedup:>8.2f}x "
                f"{single_fps:>10.1f} {v3_fps:>10.1f}{marker}"
            )

        except Exception as e:
            print(f"{batch_frames:<8} {single_time:>10.2f} {'ERROR':<12} {'N/A':<10} {single_fps:>10.1f} {'N/A':<12}")
            logger.error(f"V3 kernel failed for {batch_frames} frames: {e}")

    print("\n" + "=" * 100)
    print("MEMORY ACCESS ANALYSIS")
    print("=" * 100)

    # Calculate theoretical vs actual memory bandwidth
    led_pattern_size = config["block_size"] * config["block_size"] * 4  # 16KB per pattern
    total_patterns = config["batch_size"] * config["channels"] * led_pattern_size  # ~140MB

    frame_size = config["height"] * config["width"] * config["channels"] * 4  # ~4.6MB per frame

    print(f"LED pattern data: {total_patterns / (1024*1024):.1f} MB")
    print(f"Per frame data: {frame_size / (1024*1024):.1f} MB")
    print("V3 kernel design:")
    print("  - Each block handles 1 LED × all frames")
    print("  - LED pattern loaded once to shared memory (16KB)")
    print(f"  - Frame data accessed {config['batch_frames']} times per pixel")
    print(f"  - Grid size: {config['batch_size'] * config['channels']:,} blocks")


if __name__ == "__main__":
    test_production_v3()
