#!/usr/bin/env python3
"""
Test comparing batch frame optimizer with sequential single-frame optimizer.

This test validates that the new batch optimizer produces identical results
to running the single-frame optimizer 8 times sequentially.
"""

import logging
import time

import cupy as cp
import numpy as np

from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values
from src.utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from src.utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data(led_count=512, use_small_test=True):
    """Create test data for batch vs sequential comparison."""

    # Ensure LED count is multiple of 16 for tensor core operations
    if led_count % 16 != 0:
        raise ValueError(f"LED count must be multiple of 16 for batch operations, got {led_count}")

    if use_small_test:
        # Use smaller matrices for faster testing
        height, width = 400, 240
        block_size = 64  # Use standard block size
    else:
        # Full production size
        height, width = 800, 480
        block_size = 64

    channels = 3

    print(f"Creating test data: {led_count} LEDs, {height}x{width} images, {block_size}x{block_size} blocks")

    # Create mixed sparse tensor
    tensor = SingleBlockMixedSparseTensor(
        batch_size=led_count,
        channels=channels,
        height=height,
        width=width,
        block_size=block_size,
        dtype=cp.uint8,  # Use uint8 for production realism
    )

    # Set up random blocks with proper 4-pixel alignment for X coordinates
    np.random.seed(42)
    for led_idx in range(led_count):
        for channel in range(channels):
            top = np.random.randint(0, height - block_size + 1)
            left_raw = np.random.randint(0, width - block_size + 1)
            # CRITICAL: Align left coordinate to multiple of 4 for tensor cores
            left = (left_raw // 4) * 4
            values = cp.random.randint(0, 256, (block_size, block_size), dtype=cp.uint8)
            tensor.set_block(led_idx, channel, top, left, values)

    # Create dense ATA matrices for testing
    print("Computing ATA matrices...")
    dense_ata = tensor.compute_ata_dense()  # Shape: (led_count, led_count, channels)

    # Convert to symmetric diagonal format
    print("Creating symmetric diagonal ATA matrix...")
    symmetric_ata = SymmetricDiagonalATAMatrix.from_dense(
        dense_ata.transpose(2, 0, 1),  # Convert to (channels, led_count, led_count)
        led_count=led_count,
        significance_threshold=0.01,
        crop_size=block_size,
    )

    # Create ATA inverse using pseudoinverse
    print("Computing ATA inverse matrices...")
    ata_inverse = np.zeros((3, led_count, led_count), dtype=np.float32)

    for channel in range(3):
        ata_channel = dense_ata[:, :, channel]
        # Add small regularization for stability
        regularized = ata_channel + 1e-6 * np.eye(led_count)
        try:
            ata_inverse[channel] = np.linalg.pinv(regularized)
        except np.linalg.LinAlgError:
            # Fallback to identity if inversion fails
            print(f"Warning: Using identity for channel {channel} ATA inverse")
            ata_inverse[channel] = np.eye(led_count)

    # Create 8 random target frames
    print("Creating test target frames...")
    np.random.seed(123)  # Different seed for targets
    target_frames = np.random.randint(0, 256, (8, channels, height, width), dtype=np.uint8)
    target_frames_gpu = cp.asarray(target_frames)

    # Create batch symmetric ATA matrix for batch operations
    print("Creating batch symmetric ATA matrix...")
    batch_ata = BatchSymmetricDiagonalATAMatrix.from_symmetric_diagonal_matrix(
        symmetric_ata, batch_size=8  # For 8-frame batch processing
    )

    return tensor, symmetric_ata, batch_ata, ata_inverse, target_frames_gpu


def test_batch_vs_sequential(led_count=512, max_iterations=3, use_small_test=True):
    """Test batch optimizer against sequential single-frame optimizer."""

    # Ensure LED count is multiple of 16
    if led_count % 16 != 0:
        led_count = ((led_count + 15) // 16) * 16  # Round up to nearest multiple of 16
        print(f"Adjusted LED count to {led_count} (multiple of 16)")

    print("=" * 80)
    print("BATCH VS SEQUENTIAL OPTIMIZER COMPARISON")
    print("=" * 80)

    # Create test data
    tensor, symmetric_ata, batch_ata, ata_inverse, target_frames_gpu = create_test_data(
        led_count=led_count, use_small_test=use_small_test
    )

    print("\nTest configuration:")
    print(f"  LEDs: {led_count}")
    print("  Frames: 8")
    print(f"  Iterations: {max_iterations}")
    print(f"  Matrix type: Batch={type(batch_ata).__name__}, Sequential={type(symmetric_ata).__name__}")

    # Test 1: Run batch optimizer
    print("\n" + "=" * 60)
    print("RUNNING BATCH OPTIMIZER")
    print("=" * 60)

    start_time = time.time()
    batch_result = optimize_batch_frames_led_values(
        target_frames=target_frames_gpu,
        at_matrix=tensor,
        ata_matrix=batch_ata,  # Use batch ATA matrix
        ata_inverse=ata_inverse,
        max_iterations=max_iterations,
        debug=True,
        track_mse_per_iteration=True,
    )
    batch_time = time.time() - start_time

    print(f"Batch optimizer completed in {batch_time:.3f}s")
    print(f"Batch result shape: {batch_result.led_values.shape}")

    # Test 2: Run sequential optimizer (8 times)
    print("\n" + "=" * 60)
    print("RUNNING SEQUENTIAL OPTIMIZERS (8x)")
    print("=" * 60)

    sequential_results = []
    start_time = time.time()

    for frame_idx in range(8):
        print(f"  Processing frame {frame_idx + 1}/8...")

        frame_result = optimize_frame_led_values(
            target_frame=target_frames_gpu[frame_idx],  # (3, H, W)
            at_matrix=tensor,
            ata_matrix=symmetric_ata,
            ata_inverse=ata_inverse,
            max_iterations=max_iterations,
            debug=False,  # Reduce verbosity for sequential
            track_mse_per_iteration=True,
        )

        # frame_result.led_values is already (3, led_count)
        led_values = frame_result.led_values  # Keep as (3, led_count)
        sequential_results.append(led_values)

    sequential_time = time.time() - start_time

    # Stack sequential results: list of (3, led_count) -> (8, 3, led_count)
    sequential_led_values = cp.stack(sequential_results, axis=0)

    print(f"Sequential optimization completed in {sequential_time:.3f}s")
    print(f"Sequential result shape: {sequential_led_values.shape}")

    # Test 3: Compare results
    print("\n" + "=" * 60)
    print("COMPARING RESULTS")
    print("=" * 60)

    # Convert batch results to same format for comparison
    batch_led_values_float = batch_result.led_values.astype(cp.float32) / 255.0  # (8, 3, led_count)
    sequential_led_values_float = sequential_led_values.astype(cp.float32) / 255.0  # (8, 3, led_count)

    # Compute differences
    max_diff = cp.max(cp.abs(batch_led_values_float - sequential_led_values_float))
    mean_diff = cp.mean(cp.abs(batch_led_values_float - sequential_led_values_float))
    rms_diff = cp.sqrt(cp.mean((batch_led_values_float - sequential_led_values_float) ** 2))

    # Relative error
    batch_rms = cp.sqrt(cp.mean(batch_led_values_float**2))
    relative_error = rms_diff / batch_rms if batch_rms > 0 else float("inf")

    print("Numerical Comparison:")
    print(f"  Max absolute difference: {float(max_diff):.8f}")
    print(f"  Mean absolute difference: {float(mean_diff):.8f}")
    print(f"  RMS difference: {float(rms_diff):.8f}")
    print(f"  Relative error: {float(relative_error)*100:.6f}%")

    # Performance comparison
    speedup = sequential_time / batch_time
    print("\nPerformance Comparison:")
    print(f"  Batch time: {batch_time:.3f}s")
    print(f"  Sequential time: {sequential_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")

    # Per-frame comparison
    print("\nPer-frame LED value ranges:")
    for frame_idx in range(8):
        batch_frame = batch_led_values_float[frame_idx]
        sequential_frame = sequential_led_values_float[frame_idx]

        batch_range = f"[{float(cp.min(batch_frame)):.4f}, {float(cp.max(batch_frame)):.4f}]"
        sequential_range = f"[{float(cp.min(sequential_frame)):.4f}, {float(cp.max(sequential_frame)):.4f}]"
        frame_diff = float(cp.max(cp.abs(batch_frame - sequential_frame)))

        print(f"  Frame {frame_idx}: Batch {batch_range}, Sequential {sequential_range}, Max diff: {frame_diff:.6f}")

    # MSE comparison if available
    if batch_result.mse_per_iteration is not None:
        print("\nMSE Convergence Comparison:")
        batch_final_mse = batch_result.mse_per_iteration[-1]  # Last iteration MSE for all 8 frames
        print(f"  Batch final MSE per frame: {[f'{mse:.6f}' for mse in batch_final_mse]}")

    # Determine success
    success = relative_error < 0.01  # 1% relative error threshold

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if success:
        print("✓ SUCCESS: Batch and sequential optimizers produce equivalent results")
        print(f"  Relative error: {float(relative_error)*100:.6f}% (< 1.0%)")
        print(f"  Performance gain: {speedup:.2f}x faster")
    else:
        print("✗ FAILURE: Batch and sequential optimizers produce different results")
        print(f"  Relative error: {float(relative_error)*100:.6f}% (>= 1.0%)")

    return success, speedup, float(relative_error)


def main():
    """Run the batch vs sequential test."""

    try:
        # Test with smaller problem size first (multiple of 16)
        print("Testing with small problem size...")
        success_small, speedup_small, error_small = test_batch_vs_sequential(
            led_count=208, max_iterations=3, use_small_test=True  # 208 = 13 * 16
        )

        if success_small:
            print("\n" * 2)
            print("Small test passed! Running with larger problem size...")
            success_large, speedup_large, error_large = test_batch_vs_sequential(
                led_count=512, max_iterations=3, use_small_test=False  # 512 = 32 * 16
            )

            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(
                f"Small test (208 LEDs): {'PASS' if success_small else 'FAIL'} - {speedup_small:.2f}x speedup, {error_small*100:.6f}% error"
            )
            print(
                f"Large test (512 LEDs): {'PASS' if success_large else 'FAIL'} - {speedup_large:.2f}x speedup, {error_large*100:.6f}% error"
            )
        else:
            print("Small test failed - skipping large test")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return success_small


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
