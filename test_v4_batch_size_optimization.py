#!/usr/bin/env python3
"""
Find optimal batch size for V4 uint8 kernel with 3008 LEDs.
"""

import contextlib
import logging

import cupy as cp
import numpy as np

from src.utils.kernels.compute_optimized_3d_batch_v4_int8 import (
    cuda_transpose_dot_product_3d_batch_v4_int8,
)
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_batch_size():
    """Find optimal batch size for V4 uint8 kernel."""

    # Production configuration
    config = {
        "batch_size": 3008,
        "channels": 3,
        "height": 800,
        "width": 480,
        "block_size": 64,
    }

    print("=" * 100)
    print("V4 UINT8 BATCH SIZE OPTIMIZATION: 3008 LEDs")
    print("=" * 100)

    # Create uint8 tensor
    logger.info("Creating production uint8 tensor...")
    tensor = SingleBlockMixedSparseTensor(
        batch_size=config["batch_size"],
        channels=config["channels"],
        height=config["height"],
        width=config["width"],
        block_size=config["block_size"],
        dtype=cp.uint8,
        output_dtype=cp.float32,
    )

    # Set patterns efficiently
    np.random.seed(42)
    patterns_set = 0
    for led_idx in range(0, config["batch_size"], 8):  # Every 8th for speed
        for channel_idx in range(config["channels"]):
            max_row = config["height"] - config["block_size"]
            max_col = config["width"] - config["block_size"]
            row = np.random.randint(0, max_row)
            col = np.random.randint(0, max_col)
            col = (col // 4) * 4

            block_data = cp.random.randint(0, 128, (config["block_size"], config["block_size"]), dtype=cp.uint8)
            tensor.set_block(led_idx, channel_idx, row, col, block_data)
            patterns_set += 1

    print(f"Set {patterns_set:,} patterns for testing")

    # Test range of batch sizes
    batch_frames_list = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32]

    print(
        f"\n{'Frames':<8} {'Single (ms)':<12} {'V4 (ms)':<10} {'Speedup':<10} {'Throughput (FPS)':<18} {'Efficiency':<12}"
    )
    print("-" * 95)

    best_speedup = 0
    best_frames = 0

    for batch_frames in batch_frames_list:
        # Check thread limit (256 threads max per block)
        if batch_frames > 32:  # Safety margin
            continue

        # Create target batch
        target_batch = cp.random.randint(
            0, 256, (batch_frames, config["channels"], config["height"], config["width"]), dtype=cp.uint8
        )

        # Warmup
        cp.cuda.Stream.null.synchronize()
        for _ in range(2):
            _ = tensor.transpose_dot_product_3d(target_batch[0])
            with contextlib.suppress(Exception):
                _ = cuda_transpose_dot_product_3d_batch_v4_int8(
                    tensor.sparse_values,
                    tensor.block_positions,
                    target_batch,
                    config["batch_size"],
                    config["channels"],
                    batch_frames,
                    config["block_size"],
                    interleaved=True,
                )
        cp.cuda.Stream.null.synchronize()

        try:
            # Benchmark single-frame
            single_times = []
            for _ in range(5):
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record()
                for frame_idx in range(batch_frames):
                    _ = tensor.transpose_dot_product_3d(target_batch[frame_idx])
                end.record()
                end.synchronize()
                single_times.append(cp.cuda.get_elapsed_time(start, end))
            single_time = np.median(single_times)

            # Benchmark V4
            v4_times = []
            for _ in range(5):
                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record()
                _ = cuda_transpose_dot_product_3d_batch_v4_int8(
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
                v4_times.append(cp.cuda.get_elapsed_time(start, end))
            v4_time = np.median(v4_times)

            # Calculate metrics
            speedup = single_time / v4_time
            throughput_fps = batch_frames / (v4_time / 1000)

            # Theoretical memory advantage
            led_patterns_size = config["batch_size"] * config["channels"] * config["block_size"] ** 2
            frame_size = config["height"] * config["width"] * config["channels"]
            single_reads = batch_frames * (led_patterns_size + frame_size)
            batch_reads = led_patterns_size + batch_frames * frame_size
            theoretical_reduction = single_reads / batch_reads
            efficiency = speedup / theoretical_reduction * 100

            marker = " ✓" if speedup > best_speedup else ""
            if speedup > best_speedup:
                best_speedup = speedup
                best_frames = batch_frames

            print(
                f"{batch_frames:<8} {single_time:>10.2f} {v4_time:>8.2f} {speedup:>8.2f}x {throughput_fps:>14.1f} {efficiency:>10.1f}%{marker}"
            )

        except Exception as e:
            print(f"{batch_frames:<8} {'ERROR':<12} {'ERROR':<10} {'N/A':<10} {'N/A':<18} {'N/A':<12}")
            if batch_frames <= 8:  # Only log errors for reasonable batch sizes
                logger.error(f"V4 kernel failed for {batch_frames} frames: {e}")

    print("\n" + "=" * 100)
    print("OPTIMIZATION RESULTS")
    print("=" * 100)
    print("Best configuration:")
    print(f"  Optimal batch size: {best_frames} frames")
    print(f"  Best speedup: {best_speedup:.2f}x")
    print("  Theoretical max: 6.6x (memory bandwidth limited)")
    print(f"  Achieved efficiency: {best_speedup / 6.6 * 100:.1f}% of theoretical")

    # Test the optimal configuration with correctness check
    if best_frames > 0:
        print(f"\nVerifying correctness at optimal batch size ({best_frames} frames):")
        target_optimal = cp.random.randint(
            0, 256, (best_frames, config["channels"], config["height"], config["width"]), dtype=cp.uint8
        )

        # Reference
        ref_results = []
        for frame_idx in range(best_frames):
            result = tensor.transpose_dot_product_3d(target_optimal[frame_idx], planar_output=False)
            ref_results.append(result)
        reference = cp.stack(ref_results, axis=0)

        # V4
        v4_result = cuda_transpose_dot_product_3d_batch_v4_int8(
            tensor.sparse_values,
            tensor.block_positions,
            target_optimal,
            config["batch_size"],
            config["channels"],
            best_frames,
            config["block_size"],
            interleaved=True,
        )

        max_diff = cp.max(cp.abs(v4_result - reference)).get()
        rel_err = max_diff / (cp.mean(cp.abs(reference)).get() + 1e-7)

        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Relative error: {rel_err:.2e}")
        print(f"  {'✓ PASSED' if rel_err < 1e-3 else '✗ FAILED'}")


if __name__ == "__main__":
    optimize_batch_size()
