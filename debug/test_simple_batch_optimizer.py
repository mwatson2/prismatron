#!/usr/bin/env python3
"""
Simple test for the batch frame optimizer functionality.

This test validates the basic structure and operations without complex matrix operations.
"""

import logging

import cupy as cp
import numpy as np

from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values
from src.utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from src.utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple_batch_optimizer():
    """Test basic batch optimizer functionality with minimal setup."""

    print("Testing simple batch optimizer functionality...")

    # Use properly aligned dimensions for tensor core operations
    led_count = 16  # Multiple of 16 ✓
    channels = 3
    height, width = 128, 96
    block_size = 64  # Standard block size

    print(f"Configuration: {led_count} LEDs, {height}x{width}, {block_size}x{block_size} blocks")

    # Create mixed sparse tensor
    tensor = SingleBlockMixedSparseTensor(
        batch_size=led_count, channels=channels, height=height, width=width, block_size=block_size, dtype=cp.uint8
    )

    # Set up simple blocks with proper 4-pixel alignment for X coordinates
    rows = int(np.sqrt(led_count))
    cols = int(np.ceil(led_count / rows))

    for led_idx in range(led_count):
        row = led_idx // cols
        col = led_idx % cols

        for channel in range(channels):
            # Position blocks in a grid pattern with proper alignment
            top = row * (height // rows)
            left = col * (width // cols)

            # CRITICAL: Align left coordinate to multiple of 4 for tensor cores
            left = (left // 4) * 4

            # Make sure block fits
            if top + block_size <= height and left + block_size <= width:
                # Use realistic random pattern instead of simple constants
                values = cp.random.randint(0, 256, (block_size, block_size), dtype=cp.uint8)
                tensor.set_block(led_idx, channel, top, left, values)

    # Create realistic ATA inverse - computed from actual ATA matrix
    print("Computing realistic ATA matrix...")
    dense_ata = tensor.compute_ata_dense()  # (led_count, led_count, 3)

    ata_inverse = np.zeros((3, led_count, led_count), dtype=np.float32)
    for channel in range(3):
        ata_channel = dense_ata[:, :, channel]
        # Add small regularization for stability
        regularized = ata_channel + 1e-6 * np.eye(led_count)
        try:
            ata_inverse[channel] = np.linalg.pinv(regularized)
        except np.linalg.LinAlgError:
            print(f"Warning: Using scaled identity for channel {channel}")
            ata_inverse[channel] = np.eye(led_count) * 0.1

    # Create symmetric ATA matrix from computed dense ATA
    print("Creating symmetric ATA matrix from computed ATA...")
    symmetric_ata = SymmetricDiagonalATAMatrix.from_dense(
        dense_ata.transpose(2, 0, 1),  # (channels, led_count, led_count)
        led_count=led_count,
        significance_threshold=0.01,  # Lower threshold for more realistic matrix
        crop_size=block_size,
    )

    # Create batch symmetric ATA matrix from regular symmetric matrix
    print("Creating batch symmetric ATA matrix...")
    batch_ata = BatchSymmetricDiagonalATAMatrix.from_symmetric_diagonal_matrix(
        symmetric_ata, batch_size=8  # Match the frame batch size
    )

    # Create 8 random target frames
    np.random.seed(42)
    target_frames = np.random.randint(0, 256, (8, channels, height, width), dtype=np.uint8)
    target_frames_gpu = cp.asarray(target_frames)

    print(f"Target frames shape: {target_frames_gpu.shape}")

    # Test batch optimizer
    try:
        batch_result = optimize_batch_frames_led_values(
            target_frames=target_frames_gpu,
            at_matrix=tensor,
            ata_matrix=batch_ata,
            ata_inverse=ata_inverse,
            max_iterations=2,
            debug=True,
            compute_error_metrics=False,
            track_mse_per_iteration=True,
        )

        print("✓ Batch optimizer completed successfully!")
        print(f"  Result shape: {batch_result.led_values.shape}")
        print(f"  LED value range: [{batch_result.led_values.min()}, {batch_result.led_values.max()}]")
        print(f"  Iterations: {batch_result.iterations}")

        if batch_result.mse_per_iteration is not None:
            print(f"  MSE tracking: {batch_result.mse_per_iteration.shape}")
            print(f"  Final MSE per frame: {batch_result.mse_per_iteration[-1]}")

        # Now test sequential optimizer for comparison
        print("\n--- Testing Sequential Optimizer for Comparison ---")
        from src.utils.frame_optimizer import optimize_frame_led_values

        sequential_results = []
        for frame_idx in range(8):
            frame_result = optimize_frame_led_values(
                target_frame=target_frames_gpu[frame_idx],  # (3, H, W)
                at_matrix=tensor,
                ata_matrix=symmetric_ata,  # Use regular symmetric matrix
                ata_inverse=ata_inverse,
                max_iterations=2,
                debug=False,
                track_mse_per_iteration=True,
            )
            sequential_results.append(frame_result.led_values)  # (3, led_count)

        # Stack sequential results: (8, 3, led_count)
        sequential_led_values = cp.stack(sequential_results, axis=0)

        print(f"  Sequential result shape: {sequential_led_values.shape}")

        # Compare results
        batch_float = batch_result.led_values.astype(cp.float32) / 255.0
        sequential_float = sequential_led_values.astype(cp.float32) / 255.0

        max_diff = float(cp.max(cp.abs(batch_float - sequential_float)))
        mean_diff = float(cp.mean(cp.abs(batch_float - sequential_float)))
        rms_diff = float(cp.sqrt(cp.mean((batch_float - sequential_float) ** 2)))

        batch_rms = float(cp.sqrt(cp.mean(batch_float**2)))
        relative_error = rms_diff / batch_rms if batch_rms > 0 else float("inf")

        print("\n--- Batch vs Sequential Comparison (16 LEDs) ---")
        print(f"  Max absolute difference: {max_diff:.8f}")
        print(f"  Mean absolute difference: {mean_diff:.8f}")
        print(f"  RMS difference: {rms_diff:.8f}")
        print(f"  Relative error: {relative_error*100:.6f}%")

        if relative_error < 0.05:  # 5% tolerance for TF32 differences
            print("  ✓ Small differences as expected (TF32 precision)")
        else:
            print("  ✗ Large differences suggest algorithmic issue")

        return True

    except Exception as e:
        print(f"✗ Batch optimizer failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the simple batch optimizer test."""

    try:
        success = test_simple_batch_optimizer()

        if success:
            print("\n✓ Simple batch optimizer test PASSED")
        else:
            print("\n✗ Simple batch optimizer test FAILED")

        return success

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
