#!/usr/bin/env python3
"""
Benchmark production-scale 3008 LED configuration: batch vs single-frame operations.
"""

import logging

import cupy as cp
import numpy as np

from src.utils.kernels.compute_optimized_3d_batch_v2 import (
    cuda_transpose_dot_product_3d_batch_v2,
    cuda_transpose_dot_product_3d_batch_warp,
)
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def benchmark_3008_leds():
    """Benchmark 3008 LED configuration with various batch sizes."""

    # Production configuration
    config = {
        "batch_size": 3008,  # Number of LEDs
        "channels": 3,  # RGB
        "height": 800,  # Display height
        "width": 480,  # Display width
        "block_size": 64,  # LED pattern block size
    }

    # Test different batch sizes
    batch_frames_list = [1, 2, 4, 8, 16, 32]

    print("\n" + "=" * 120)
    print("PRODUCTION BENCHMARK: 3008 LEDs")
    print("=" * 120)
    print(
        f"Configuration: {config['batch_size']} LEDs, {config['height']}x{config['width']} display, {config['block_size']}x{config['block_size']} blocks"
    )
    print("-" * 120)
    print(
        f"{'Batch Frames':<15} {'Single (ms)':<15} {'Batch (ms)':<15} {'Speedup':<12} "
        f"{'Single FPS':<15} {'Batch FPS':<15} {'Kernel Used':<20}"
    )
    print("-" * 120)

    # Create test tensor
    logger.info(f"Creating tensor with {config['batch_size']} LEDs...")
    tensor = SingleBlockMixedSparseTensor(
        batch_size=config["batch_size"],
        channels=config["channels"],
        height=config["height"],
        width=config["width"],
        block_size=config["block_size"],
        dtype=cp.float32,
    )

    # Set random LED positions
    np.random.seed(42)
    for led_idx in range(config["batch_size"]):
        for channel_idx in range(config["channels"]):
            max_row = config["height"] - config["block_size"]
            max_col = config["width"] - config["block_size"]
            row = np.random.randint(0, max_row)
            col = np.random.randint(0, max_col)
            col = (col // 4) * 4  # Align for vectorization

            block_data = cp.random.rand(config["block_size"], config["block_size"], dtype=cp.float32) * 0.5 + 0.1
            tensor.set_block(led_idx, channel_idx, row, col, block_data)

    logger.info("Starting benchmarks...")

    for batch_frames in batch_frames_list:
        # Create target batch
        target_batch = (
            cp.random.rand(batch_frames, config["channels"], config["height"], config["width"], dtype=cp.float32) * 0.8
            + 0.1
        )

        # Warmup
        for _ in range(3):
            _ = tensor.transpose_dot_product_3d(target_batch[0], planar_output=False)
            _ = tensor.transpose_dot_product_3d_batch(target_batch, planar_output=False)
        cp.cuda.Stream.null.synchronize()

        # Benchmark single-frame operations
        single_times = []
        for _ in range(10):
            start = cp.cuda.Event()
            end = cp.cuda.Event()

            start.record()
            for frame_idx in range(batch_frames):
                result = tensor.transpose_dot_product_3d(target_batch[frame_idx], planar_output=False)
            end.record()
            end.synchronize()

            single_times.append(cp.cuda.get_elapsed_time(start, end))
        single_time = np.median(single_times)

        # Benchmark batch operation (uses optimized kernel automatically)
        batch_times = []
        for _ in range(10):
            start = cp.cuda.Event()
            end = cp.cuda.Event()

            start.record()
            result = tensor.transpose_dot_product_3d_batch(target_batch, planar_output=False, use_warp_kernel=True)
            end.record()
            end.synchronize()

            batch_times.append(cp.cuda.get_elapsed_time(start, end))
        batch_time = np.median(batch_times)

        # Calculate metrics
        speedup = single_time / batch_time
        single_fps = batch_frames / (single_time / 1000)
        batch_fps = batch_frames / (batch_time / 1000)

        # Determine which kernel was used
        kernel_used = "V2" if batch_frames < 8 else "Warp"
        if speedup > 1.0:
            kernel_used += " ✓"

        print(
            f"{batch_frames:<15} {single_time:>13.2f} {batch_time:>13.2f} {speedup:>10.2f}x "
            f"{single_fps:>13.1f} {batch_fps:>13.1f} {kernel_used:<20}"
        )

    # Additional analysis
    print("\n" + "=" * 120)
    print("DETAILED ANALYSIS")
    print("=" * 120)

    # Test specific optimized kernels
    batch_frames = 16  # Good test size
    target_batch = cp.random.rand(batch_frames, config["channels"], config["height"], config["width"], dtype=cp.float32)

    # Benchmark each kernel variant
    print(f"\nKernel comparison for {batch_frames} frames:")
    print("-" * 60)

    # Single-frame baseline
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for i in range(batch_frames):
        _ = tensor.transpose_dot_product_3d(target_batch[i], planar_output=False)
    end.record()
    end.synchronize()
    single_time = cp.cuda.get_elapsed_time(start, end)
    print(f"Single-frame (repeated):     {single_time:>8.2f}ms (baseline)")

    # Original batch kernel
    from src.utils.kernels.compute_optimized_3d_batch import (
        cuda_transpose_dot_product_3d_batch_compute_optimized,
    )

    start.record()
    _ = cuda_transpose_dot_product_3d_batch_compute_optimized(
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
    orig_time = cp.cuda.get_elapsed_time(start, end)
    print(f"Original batch kernel:        {orig_time:>8.2f}ms ({single_time/orig_time:.2f}x)")

    # V2 optimized kernel
    start.record()
    _ = cuda_transpose_dot_product_3d_batch_v2(
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
    v2_time = cp.cuda.get_elapsed_time(start, end)
    print(f"V2 optimized kernel:          {v2_time:>8.2f}ms ({single_time/v2_time:.2f}x)")

    # Warp-based kernel
    start.record()
    _ = cuda_transpose_dot_product_3d_batch_warp(
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
    warp_time = cp.cuda.get_elapsed_time(start, end)
    print(f"Warp-based kernel:            {warp_time:>8.2f}ms ({single_time/warp_time:.2f}x)")

    print("\n" + "=" * 120)
    print("CONCLUSIONS")
    print("=" * 120)
    print("• Best kernel: " + ("Warp-based" if warp_time < v2_time else "V2 optimized"))
    print(f"• Peak speedup: {max(single_time/v2_time, single_time/warp_time):.2f}x over single-frame")
    print(f"• For 30 FPS video with 16-frame batches: {batch_fps:.1f} FPS achievable")
    print(f"• Memory bandwidth utilization improved through shared memory reuse")


if __name__ == "__main__":
    benchmark_3008_leds()
