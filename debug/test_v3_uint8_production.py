#!/usr/bin/env python3
"""
Test V3 uint8 kernel with production 3008 LED configuration.
"""

import logging

import cupy as cp
import numpy as np

from src.utils.kernels.compute_optimized_3d_batch_v3_int8 import (
    cuda_transpose_dot_product_3d_batch_v3_int8,
)
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_production_uint8():
    """Test V3 uint8 kernel with production 3008 LED configuration."""

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
    print("V3 UINT8 PRODUCTION TEST: 3008 LEDs")
    print("=" * 100)
    print(f"Configuration: {config['batch_size']} LEDs, {config['height']}x{config['width']} display (uint8)")
    print("-" * 100)
    print(f"{'Frames':<8} {'Single (ms)':<12} {'V3 uint8 (ms)':<15} {'Speedup':<10} {'Single FPS':<12} {'V3 FPS':<12}")
    print("-" * 100)

    # Create uint8 tensor
    logger.info("Creating production uint8 tensor...")
    tensor = SingleBlockMixedSparseTensor(
        batch_size=config["batch_size"],
        channels=config["channels"],
        height=config["height"],
        width=config["width"],
        block_size=config["block_size"],
        dtype=cp.uint8,  # uint8 patterns
        output_dtype=cp.float32,
    )

    # Set patterns efficiently (sample for speed)
    np.random.seed(42)
    for led_idx in range(0, config["batch_size"], 5):  # Every 5th LED for speed
        for channel_idx in range(config["channels"]):
            max_row = config["height"] - config["block_size"]
            max_col = config["width"] - config["block_size"]
            row = np.random.randint(0, max_row)
            col = np.random.randint(0, max_col)
            col = (col // 4) * 4

            # Generate uint8 block patterns
            block_data = cp.random.randint(0, 256, (config["block_size"], config["block_size"]), dtype=cp.uint8)
            tensor.set_block(led_idx, channel_idx, row, col, block_data)

    # Fill remaining with sparse patterns (mostly low values)
    for led_idx in range(config["batch_size"]):
        for channel_idx in range(config["channels"]):
            if led_idx % 5 != 0:  # Fill in the gaps
                row = (led_idx * 37) % (config["height"] - config["block_size"])
                col = ((led_idx * 73) % (config["width"] - config["block_size"]) // 4) * 4
                # Sparse pattern with mostly low values
                block_data = cp.random.randint(0, 64, (config["block_size"], config["block_size"]), dtype=cp.uint8)
                tensor.set_block(led_idx, channel_idx, row, col, block_data)

    logger.info("Starting benchmarks...")

    for batch_frames in batch_frames_list:
        # Create uint8 target batch
        target_batch = cp.random.randint(
            0, 256, (batch_frames, config["channels"], config["height"], config["width"]), dtype=cp.uint8
        )

        # Check thread limit for V3 kernel
        threads_needed = batch_frames * 32
        if threads_needed > 1024:
            print(f"{batch_frames:<8} {'SKIP':<12} {'SKIP':<15} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
            continue

        # Warmup
        for _ in range(3):
            _ = tensor.transpose_dot_product_3d(target_batch[0])
            _ = cuda_transpose_dot_product_3d_batch_v3_int8(
                tensor.sparse_values,
                tensor.block_positions,
                target_batch,
                config["batch_size"],
                config["channels"],
                batch_frames,
                config["block_size"],
                interleaved=True,
                raw_output=False,
            )
        cp.cuda.Stream.null.synchronize()

        # Benchmark single-frame uint8 operations
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

        # Benchmark V3 uint8 batch kernel
        v3_times = []
        for _ in range(5):
            start = cp.cuda.Event()
            end = cp.cuda.Event()

            start.record()
            _ = cuda_transpose_dot_product_3d_batch_v3_int8(
                tensor.sparse_values,
                tensor.block_positions,
                target_batch,
                config["batch_size"],
                config["channels"],
                batch_frames,
                config["block_size"],
                interleaved=True,
                raw_output=False,
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
            f"{batch_frames:<8} {single_time:>10.2f} {v3_time:>13.2f} {speedup:>8.2f}x "
            f"{single_fps:>10.1f} {v3_fps:>10.1f}{marker}"
        )

    print("\n" + "=" * 100)
    print("UINT8 MEMORY EFFICIENCY ANALYSIS")
    print("=" * 100)

    # Calculate memory usage for uint8 vs float32
    led_patterns_uint8 = config["batch_size"] * config["channels"] * config["block_size"] ** 2 * 1  # 1 byte
    led_patterns_float32 = config["batch_size"] * config["channels"] * config["block_size"] ** 2 * 4  # 4 bytes

    frame_uint8 = config["height"] * config["width"] * config["channels"] * 1  # 1 byte
    frame_float32 = config["height"] * config["width"] * config["channels"] * 4  # 4 bytes

    print("Memory usage comparison (uint8 vs float32):")
    print(
        f"  LED patterns: {led_patterns_uint8/(1024*1024):.1f} MB vs {led_patterns_float32/(1024*1024):.1f} MB (4x reduction)"
    )
    print(f"  Single frame: {frame_uint8/(1024*1024):.2f} MB vs {frame_float32/(1024*1024):.2f} MB (4x reduction)")
    print(f"  8 frames: {8*frame_uint8/(1024*1024):.1f} MB vs {8*frame_float32/(1024*1024):.1f} MB")

    print("\nMemory bandwidth theoretical advantage:")
    single_reads_uint8 = 8 * (led_patterns_uint8 + frame_uint8)
    batch_reads_uint8 = led_patterns_uint8 + 8 * frame_uint8
    print(f"  Single approach: {single_reads_uint8/(1024*1024):.1f} MB")
    print(f"  Batch approach: {batch_reads_uint8/(1024*1024):.1f} MB")
    print(f"  Theoretical reduction: {single_reads_uint8/batch_reads_uint8:.1f}x")

    print("\nUint8 compute advantages:")
    print("  - 4x less memory bandwidth")
    print("  - uchar4 vectorization (4 uint8 ops per instruction)")
    print("  - uint32 accumulation avoids overflow for 64x64 blocks")
    print("  - Proper 255² scaling maintains precision")


if __name__ == "__main__":
    test_production_uint8()
