#!/usr/bin/env python3
"""
Test V3 uint8 batch kernel for correctness and performance.
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


def test_uint8_kernel():
    """Test uint8 batch kernel correctness and performance."""

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
    print("V3 UINT8 KERNEL CORRECTNESS AND PERFORMANCE TEST")
    print("=" * 80)

    # Create uint8 tensor
    tensor = SingleBlockMixedSparseTensor(
        batch_size=config["batch_size"],
        channels=config["channels"],
        height=config["height"],
        width=config["width"],
        block_size=config["block_size"],
        dtype=cp.uint8,  # uint8 patterns
        output_dtype=cp.float32,
    )

    # Set random uint8 patterns
    np.random.seed(42)
    for led_idx in range(config["batch_size"]):
        for channel_idx in range(config["channels"]):
            max_row = config["height"] - config["block_size"]
            max_col = config["width"] - config["block_size"]
            row = np.random.randint(0, max_row)
            col = np.random.randint(0, max_col)
            col = (col // 4) * 4

            # Generate uint8 block data [0, 255]
            block_data = cp.random.randint(0, 256, (config["block_size"], config["block_size"]), dtype=cp.uint8)
            tensor.set_block(led_idx, channel_idx, row, col, block_data)

    # Create uint8 target batch
    target_batch = cp.random.randint(
        0, 256, (config["batch_frames"], config["channels"], config["height"], config["width"]), dtype=cp.uint8
    )

    print(f"Testing with {config['batch_size']} LEDs, {config['batch_frames']} frames (uint8)")

    # Get reference result from single-frame uint8 operations
    print("\nComputing reference result (single-frame uint8)...")
    reference_results = []
    for frame_idx in range(config["batch_frames"]):
        result = tensor.transpose_dot_product_3d(target_batch[frame_idx], planar_output=False)
        reference_results.append(result)
    reference = cp.stack(reference_results, axis=0)

    print(f"Reference result shape: {reference.shape}, dtype: {reference.dtype}")
    print(f"Reference range: [{cp.min(reference):.6f}, {cp.max(reference):.6f}]")

    # Test V3 uint8 kernel with fp32 output (scaled)
    print("\nTesting V3 uint8 kernel (fp32 output with scaling)...")
    try:
        v3_result = cuda_transpose_dot_product_3d_batch_v3_int8(
            tensor.sparse_values,
            tensor.block_positions,
            target_batch,
            config["batch_size"],
            config["channels"],
            config["batch_frames"],
            config["block_size"],
            interleaved=True,
            raw_output=False,  # fp32 with 255² scaling
        )

        print(f"V3 result shape: {v3_result.shape}, dtype: {v3_result.dtype}")
        print(f"V3 result range: [{cp.min(v3_result):.6f}, {cp.max(v3_result):.6f}]")

        # Compare results
        v3_diff = cp.max(cp.abs(v3_result - reference)).get()
        v3_rel_err = v3_diff / (cp.mean(cp.abs(reference)).get() + 1e-7)

        print(f"Max diff: {v3_diff:.2e}, Relative error: {v3_rel_err:.2e}")

        if v3_rel_err < 1e-4:
            print("✓ V3 uint8 kernel correctness PASSED")
        else:
            print("✗ V3 uint8 kernel correctness FAILED")
            print("  Checking if scaling difference...")
            # Check if it's just a scaling factor difference
            if reference.size > 0 and v3_result.size > 0:
                ref_mean = cp.mean(reference).get()
                v3_mean = cp.mean(v3_result).get()
                if ref_mean > 0:
                    scale_ratio = v3_mean / ref_mean
                    print(f"  Scaling ratio: {scale_ratio:.2f} (expected ~1/65025 = {1/65025:.6f})")

        v3_valid = True

    except Exception as e:
        print(f"✗ V3 uint8 kernel FAILED: {e}")
        v3_result = None
        v3_valid = False

    # Test V3 uint8 kernel with raw uint32 output
    print("\nTesting V3 uint8 kernel (uint32 raw output)...")
    try:
        v3_raw_result = cuda_transpose_dot_product_3d_batch_v3_int8(
            tensor.sparse_values,
            tensor.block_positions,
            target_batch,
            config["batch_size"],
            config["channels"],
            config["batch_frames"],
            config["block_size"],
            interleaved=True,
            raw_output=True,  # uint32 without scaling
        )

        print(f"V3 raw shape: {v3_raw_result.shape}, dtype: {v3_raw_result.dtype}")
        print(f"V3 raw range: [{cp.min(v3_raw_result)}, {cp.max(v3_raw_result)}]")

        # Convert reference to expected uint32 range for comparison
        reference_scaled = reference * 65025.0  # Scale by 255²
        reference_uint32 = reference_scaled.astype(cp.uint32)

        raw_diff = cp.max(cp.abs(v3_raw_result.astype(cp.float32) - reference_uint32.astype(cp.float32))).get()
        raw_rel_err = raw_diff / (cp.mean(reference_uint32.astype(cp.float32)).get() + 1e-7)

        print(f"Max diff (vs scaled reference): {raw_diff:.2e}, Relative error: {raw_rel_err:.2e}")

        if raw_rel_err < 1e-4:
            print("✓ V3 uint8 raw kernel correctness PASSED")
        else:
            print("✗ V3 uint8 raw kernel correctness FAILED")

    except Exception as e:
        print(f"✗ V3 uint8 raw kernel FAILED: {e}")
        v3_raw_result = None

    # Performance comparison if kernels work
    if v3_valid:
        print("\n" + "-" * 80)
        print("PERFORMANCE COMPARISON")
        print("-" * 80)

        # Warmup
        for _ in range(5):
            _ = tensor.transpose_dot_product_3d(target_batch[0])
            _ = cuda_transpose_dot_product_3d_batch_v3_int8(
                tensor.sparse_values,
                tensor.block_positions,
                target_batch,
                config["batch_size"],
                config["channels"],
                config["batch_frames"],
                config["block_size"],
                interleaved=True,
                raw_output=False,
            )
        cp.cuda.Stream.null.synchronize()

        # Benchmark single-frame uint8
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

        # Benchmark V3 uint8 batch
        v3_times = []
        for _ in range(10):
            start = cp.cuda.Event()
            end = cp.cuda.Event()

            start.record()
            _ = cuda_transpose_dot_product_3d_batch_v3_int8(
                tensor.sparse_values,
                tensor.block_positions,
                target_batch,
                config["batch_size"],
                config["channels"],
                config["batch_frames"],
                config["block_size"],
                interleaved=True,
                raw_output=False,
            )
            end.record()
            end.synchronize()

            v3_times.append(cp.cuda.get_elapsed_time(start, end))
        v3_time = np.median(v3_times)

        speedup = single_time / v3_time
        single_fps = config["batch_frames"] / (single_time / 1000)
        v3_fps = config["batch_frames"] / (v3_time / 1000)

        print(f"Single-frame uint8:      {single_time:>8.2f}ms ({single_fps:>6.1f} FPS)")
        print(f"V3 batch uint8:          {v3_time:>8.2f}ms ({v3_fps:>6.1f} FPS)")
        print(f"Speedup:                 {speedup:>8.2f}x")

        print("\nMemory usage comparison:")
        print(f"  LED patterns: {tensor.sparse_values.nbytes / (1024*1024):.1f} MB (uint8)")
        print(f"  Target frames: {target_batch.nbytes / (1024*1024):.1f} MB (uint8)")
        print(f"  Single approach reads: ~{8 * tensor.sparse_values.nbytes / (1024*1024):.1f} MB")
        print(f"  Batch approach reads: ~{(tensor.sparse_values.nbytes + target_batch.nbytes) / (1024*1024):.1f} MB")
        reduction = (8 * tensor.sparse_values.nbytes + 8 * target_batch.nbytes) / (
            tensor.sparse_values.nbytes + target_batch.nbytes
        )
        print(f"  Theoretical memory reduction: {reduction:.1f}x")


if __name__ == "__main__":
    test_uint8_kernel()
