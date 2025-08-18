#!/usr/bin/env python3
"""
Benchmark script to compare different batch kernel optimizations.
"""

import logging
import time
from typing import Dict, List, Tuple

import cupy as cp
import numpy as np

from src.utils.kernels.compute_optimized_3d_batch import cuda_transpose_dot_product_3d_batch_compute_optimized
from src.utils.kernels.compute_optimized_3d_batch_v2 import (
    cuda_transpose_dot_product_3d_batch_v2,
    cuda_transpose_dot_product_3d_batch_warp,
)
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_test_data(config: Dict) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Create test data for benchmarking."""
    batch_size = config["batch_size"]
    channels = config["channels"]
    height = config["height"]
    width = config["width"]
    block_size = config["block_size"]
    batch_frames = config["batch_frames"]

    # Create sparse values (LED patterns)
    sparse_values = cp.random.rand(channels, batch_size, block_size, block_size, dtype=cp.float32)

    # Create block positions
    block_positions = cp.zeros((channels, batch_size, 2), dtype=cp.int32)
    for led in range(batch_size):
        for ch in range(channels):
            # Random positions ensuring blocks fit
            max_row = height - block_size
            max_col = width - block_size
            row = np.random.randint(0, max_row)
            col = np.random.randint(0, max_col)
            # Align for vectorization
            col = (col // 4) * 4
            block_positions[ch, led, 0] = row
            block_positions[ch, led, 1] = col

    # Create target batch
    target_batch = cp.random.rand(batch_frames, channels, height, width, dtype=cp.float32)

    return sparse_values, block_positions, target_batch


def benchmark_kernel(
    kernel_func,
    sparse_values: cp.ndarray,
    block_positions: cp.ndarray,
    target_batch: cp.ndarray,
    config: Dict,
    warmup_runs: int = 5,
    test_runs: int = 20,
) -> float:
    """Benchmark a single kernel implementation."""
    batch_size = config["batch_size"]
    channels = config["channels"]
    batch_frames = config["batch_frames"]
    block_size = config["block_size"]

    # Warmup
    for _ in range(warmup_runs):
        result = kernel_func(
            sparse_values,
            block_positions,
            target_batch,
            batch_size,
            channels,
            batch_frames,
            block_size,
            interleaved=True,
        )
        cp.cuda.Stream.null.synchronize()

    # Benchmark
    times = []
    for _ in range(test_runs):
        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        result = kernel_func(
            sparse_values,
            block_positions,
            target_batch,
            batch_size,
            channels,
            batch_frames,
            block_size,
            interleaved=True,
        )
        end.record()
        end.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start, end)
        times.append(elapsed_ms)

    return np.median(times)


def run_comparison():
    """Run comprehensive comparison of batch kernel implementations."""

    configs = [
        {
            "name": "Small_8LEDs_4frames",
            "batch_size": 8,
            "channels": 3,
            "height": 128,
            "width": 128,
            "block_size": 32,
            "batch_frames": 4,
        },
        {
            "name": "Medium_64LEDs_8frames",
            "batch_size": 64,
            "channels": 3,
            "height": 256,
            "width": 256,
            "block_size": 64,
            "batch_frames": 8,
        },
        {
            "name": "Large_256LEDs_16frames",
            "batch_size": 256,
            "channels": 3,
            "height": 256,
            "width": 256,
            "block_size": 64,
            "batch_frames": 16,
        },
        {
            "name": "Production_512LEDs_8frames",
            "batch_size": 512,
            "channels": 3,
            "height": 512,
            "width": 512,
            "block_size": 64,
            "batch_frames": 8,
        },
    ]

    kernels = {
        "Original": cuda_transpose_dot_product_3d_batch_compute_optimized,
        "V2_Optimized": cuda_transpose_dot_product_3d_batch_v2,
        "Warp_Based": cuda_transpose_dot_product_3d_batch_warp,
    }

    print("\n" + "=" * 100)
    print("BATCH KERNEL OPTIMIZATION COMPARISON")
    print("=" * 100)
    print(f"{'Config':<30} {'Kernel':<15} {'Time (ms)':<12} {'Speedup':<10} {'FPS':<10}")
    print("-" * 100)

    for config in configs:
        # Create test data
        sparse_values, block_positions, target_batch = create_test_data(config)

        results = {}
        for kernel_name, kernel_func in kernels.items():
            try:
                time_ms = benchmark_kernel(
                    kernel_func,
                    sparse_values,
                    block_positions,
                    target_batch,
                    config,
                )
                results[kernel_name] = time_ms
            except Exception as e:
                logger.error(f"Error benchmarking {kernel_name}: {e}")
                results[kernel_name] = float("inf")

        # Calculate speedups relative to original
        baseline_time = results["Original"]

        for kernel_name, time_ms in results.items():
            speedup = baseline_time / time_ms if time_ms > 0 else 0
            fps = config["batch_frames"] / (time_ms / 1000) if time_ms > 0 else 0

            print(f"{config['name']:<30} {kernel_name:<15} {time_ms:>10.2f}ms " f"{speedup:>8.2f}x {fps:>8.1f}")

        print()  # Blank line between configs

    # Also run a correctness check
    print("\n" + "=" * 100)
    print("CORRECTNESS VERIFICATION")
    print("=" * 100)

    config = configs[1]  # Use medium config for correctness test
    sparse_values, block_positions, target_batch = create_test_data(config)

    # Get results from all kernels
    original_result = cuda_transpose_dot_product_3d_batch_compute_optimized(
        sparse_values,
        block_positions,
        target_batch,
        config["batch_size"],
        config["channels"],
        config["batch_frames"],
        config["block_size"],
        interleaved=True,
    )

    v2_result = cuda_transpose_dot_product_3d_batch_v2(
        sparse_values,
        block_positions,
        target_batch,
        config["batch_size"],
        config["channels"],
        config["batch_frames"],
        config["block_size"],
        interleaved=True,
    )

    warp_result = cuda_transpose_dot_product_3d_batch_warp(
        sparse_values,
        block_positions,
        target_batch,
        config["batch_size"],
        config["channels"],
        config["batch_frames"],
        config["block_size"],
        interleaved=True,
    )

    # Compare results
    v2_diff = cp.max(cp.abs(v2_result - original_result)).get()
    warp_diff = cp.max(cp.abs(warp_result - original_result)).get()

    v2_rel_error = v2_diff / (cp.mean(cp.abs(original_result)).get() + 1e-7)
    warp_rel_error = warp_diff / (cp.mean(cp.abs(original_result)).get() + 1e-7)

    print(f"V2 vs Original:   Max diff = {v2_diff:.2e}, Relative error = {v2_rel_error:.2e}")
    print(f"Warp vs Original: Max diff = {warp_diff:.2e}, Relative error = {warp_rel_error:.2e}")

    if v2_rel_error < 1e-5 and warp_rel_error < 1e-5:
        print("\n✓ All kernels produce identical results (within floating point precision)")
    else:
        print("\n✗ WARNING: Kernels produce different results!")


if __name__ == "__main__":
    run_comparison()
