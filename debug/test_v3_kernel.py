#!/usr/bin/env python3
"""
Test the V3 memory-optimized batch kernel.
"""

import logging

import cupy as cp
import numpy as np

from src.utils.kernels.compute_optimized_3d_batch_v3 import (
    cuda_transpose_dot_product_3d_batch_v3,
    cuda_transpose_dot_product_3d_batch_v3_frames,
)
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_v3_correctness_and_performance():
    """Test V3 kernel correctness and performance."""

    # Test configuration
    config = {
        "batch_size": 256,
        "channels": 3,
        "height": 256,
        "width": 256,
        "block_size": 64,
        "batch_frames": 8,
    }

    print("=" * 80)
    print("V3 KERNEL CORRECTNESS AND PERFORMANCE TEST")
    print("=" * 80)

    # Create test tensor
    tensor = SingleBlockMixedSparseTensor(
        batch_size=config["batch_size"],
        channels=config["channels"],
        height=config["height"],
        width=config["width"],
        block_size=config["block_size"],
        dtype=cp.float32,
    )

    # Set random patterns
    np.random.seed(42)
    for led_idx in range(config["batch_size"]):
        for channel_idx in range(config["channels"]):
            max_row = config["height"] - config["block_size"]
            max_col = config["width"] - config["block_size"]
            row = np.random.randint(0, max_row)
            col = np.random.randint(0, max_col)
            col = (col // 4) * 4

            block_data = cp.random.rand(config["block_size"], config["block_size"], dtype=cp.float32) * 0.5 + 0.1
            tensor.set_block(led_idx, channel_idx, row, col, block_data)

    # Create target batch
    target_batch = (
        cp.random.rand(config["batch_frames"], config["channels"], config["height"], config["width"], dtype=cp.float32)
        * 0.8
        + 0.1
    )

    print(f"Testing with {config['batch_size']} LEDs, {config['batch_frames']} frames")

    # Get reference result from single-frame operations
    print("\nComputing reference result (single-frame)...")
    reference_results = []
    for frame_idx in range(config["batch_frames"]):
        result = tensor.transpose_dot_product_3d(target_batch[frame_idx], planar_output=False)
        reference_results.append(result)
    reference = cp.stack(reference_results, axis=0)

    # Test V3 spatial kernel
    print("Testing V3 spatial kernel...")
    try:
        v3_result = cuda_transpose_dot_product_3d_batch_v3(
            tensor.sparse_values,
            tensor.block_positions,
            target_batch,
            config["batch_size"],
            config["channels"],
            config["batch_frames"],
            config["block_size"],
            interleaved=True,
        )

        v3_diff = cp.max(cp.abs(v3_result - reference)).get()
        v3_rel_err = v3_diff / (cp.mean(cp.abs(reference)).get() + 1e-7)
        print(f"  Max diff: {v3_diff:.2e}, Relative error: {v3_rel_err:.2e}")

        if v3_rel_err < 1e-4:
            print("  ✓ V3 spatial kernel correctness PASSED")
        else:
            print("  ✗ V3 spatial kernel correctness FAILED")

    except Exception as e:
        print(f"  ✗ V3 spatial kernel FAILED: {e}")
        v3_result = None

    # Test V3 frame-parallel kernel
    print("Testing V3 frame-parallel kernel...")
    try:
        v3_frames_result = cuda_transpose_dot_product_3d_batch_v3_frames(
            tensor.sparse_values,
            tensor.block_positions,
            target_batch,
            config["batch_size"],
            config["channels"],
            config["batch_frames"],
            config["block_size"],
            interleaved=True,
        )

        v3f_diff = cp.max(cp.abs(v3_frames_result - reference)).get()
        v3f_rel_err = v3f_diff / (cp.mean(cp.abs(reference)).get() + 1e-7)
        print(f"  Max diff: {v3f_diff:.2e}, Relative error: {v3f_rel_err:.2e}")

        if v3f_rel_err < 1e-4:
            print("  ✓ V3 frame-parallel kernel correctness PASSED")
        else:
            print("  ✗ V3 frame-parallel kernel correctness FAILED")

    except Exception as e:
        print(f"  ✗ V3 frame-parallel kernel FAILED: {e}")
        v3_frames_result = None

    # Performance comparison
    if v3_result is not None or v3_frames_result is not None:
        print("\n" + "-" * 80)
        print("PERFORMANCE COMPARISON")
        print("-" * 80)

        # Warmup
        for _ in range(5):
            _ = tensor.transpose_dot_product_3d(target_batch[0])
            if v3_result is not None:
                _ = cuda_transpose_dot_product_3d_batch_v3(
                    tensor.sparse_values,
                    tensor.block_positions,
                    target_batch,
                    config["batch_size"],
                    config["channels"],
                    config["batch_frames"],
                    config["block_size"],
                    interleaved=True,
                )
        cp.cuda.Stream.null.synchronize()

        # Benchmark single-frame
        single_times = []
        for _ in range(10):
            start = cp.cuda.Event()
            end = cp.cuda.Event()

            start.record()
            for frame_idx in range(config["batch_frames"]):
                _ = tensor.transpose_dot_product_3d(target_batch[frame_idx])
            end.record()
            end.synchronize()

            single_times.append(cp.cuda.get_elapsed_time(start, end))
        single_time = np.median(single_times)

        print(f"Single-frame (repeated): {single_time:>8.2f}ms")

        # Benchmark V3 spatial if available
        if v3_result is not None:
            v3_times = []
            for _ in range(10):
                start = cp.cuda.Event()
                end = cp.cuda.Event()

                start.record()
                _ = cuda_transpose_dot_product_3d_batch_v3(
                    tensor.sparse_values,
                    tensor.block_positions,
                    target_batch,
                    config["batch_size"],
                    config["channels"],
                    config["batch_frames"],
                    config["block_size"],
                    interleaved=True,
                )
                end.record()
                end.synchronize()

                v3_times.append(cp.cuda.get_elapsed_time(start, end))
            v3_time = np.median(v3_times)

            speedup = single_time / v3_time
            print(f"V3 spatial kernel:       {v3_time:>8.2f}ms ({speedup:.2f}x)")

        # Benchmark V3 frame-parallel if available
        if v3_frames_result is not None:
            v3f_times = []
            for _ in range(10):
                start = cp.cuda.Event()
                end = cp.cuda.Event()

                start.record()
                _ = cuda_transpose_dot_product_3d_batch_v3_frames(
                    tensor.sparse_values,
                    tensor.block_positions,
                    target_batch,
                    config["batch_size"],
                    config["channels"],
                    config["batch_frames"],
                    config["block_size"],
                    interleaved=True,
                )
                end.record()
                end.synchronize()

                v3f_times.append(cp.cuda.get_elapsed_time(start, end))
            v3f_time = np.median(v3f_times)

            speedup_f = single_time / v3f_time
            print(f"V3 frame-parallel:       {v3f_time:>8.2f}ms ({speedup_f:.2f}x)")


if __name__ == "__main__":
    test_v3_correctness_and_performance()
