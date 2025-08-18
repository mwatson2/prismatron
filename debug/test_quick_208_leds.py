#!/usr/bin/env python3
"""
Quick test of 208 LEDs to verify the fix works with multiple blocks.
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

logging.basicConfig(level=logging.WARNING)  # Reduce verbosity


def test_208_leds_quick():
    """Test 208 LEDs with minimal setup for speed."""

    print("Testing 208 LEDs (13x16, multiple blocks)...")

    led_count = 208  # 13 * 16
    channels = 3
    height, width = 200, 120  # Smaller for speed
    block_size = 64

    # Create sparse tensor
    tensor = SingleBlockMixedSparseTensor(
        batch_size=led_count, channels=channels, height=height, width=width, block_size=block_size, dtype=cp.uint8
    )

    # Set blocks quickly
    np.random.seed(42)
    for led_idx in range(0, led_count, 8):  # Every 8th LED for speed
        for channel in range(channels):
            top = np.random.randint(0, height - block_size + 1)
            left_raw = np.random.randint(0, width - block_size + 1)
            left = (left_raw // 4) * 4  # Align to 4

            if left + block_size <= width:
                values = cp.random.randint(0, 256, (block_size, block_size), dtype=cp.uint8)
                tensor.set_block(led_idx, channel, top, left, values)

    # Compute matrices
    print("Computing ATA matrices...")
    dense_ata = tensor.compute_ata_dense()

    # Create ATA inverse
    ata_inverse = np.zeros((3, led_count, led_count), dtype=np.float32)
    for channel in range(3):
        ata_channel = dense_ata[:, :, channel]
        regularized = ata_channel + 1e-6 * np.eye(led_count)
        try:
            ata_inverse[channel] = np.linalg.pinv(regularized)
        except np.linalg.LinAlgError:
            ata_inverse[channel] = np.eye(led_count) * 0.1

    # Create matrices
    regular_ata = SymmetricDiagonalATAMatrix.from_dense(
        dense_ata.transpose(2, 0, 1), led_count=led_count, significance_threshold=0.01, crop_size=block_size
    )

    batch_ata = BatchSymmetricDiagonalATAMatrix.from_symmetric_diagonal_matrix(regular_ata, batch_size=8)

    # Create target frames
    np.random.seed(123)
    target_frames = np.random.randint(0, 256, (8, channels, height, width), dtype=np.uint8)
    target_frames_gpu = cp.asarray(target_frames)

    # Test batch optimizer
    print("Running batch optimizer...")
    start_time = time.time()
    batch_result = optimize_batch_frames_led_values(
        target_frames=target_frames_gpu,
        at_matrix=tensor,
        ata_matrix=batch_ata,
        ata_inverse=ata_inverse,
        max_iterations=2,  # Fewer iterations for speed
        debug=False,
    )
    batch_time = time.time() - start_time

    # Test first frame only with sequential for comparison
    print("Running sequential optimizer (first frame only)...")
    start_time = time.time()
    sequential_result = optimize_frame_led_values(
        target_frame=target_frames_gpu[0],
        at_matrix=tensor,
        ata_matrix=regular_ata,
        ata_inverse=ata_inverse,
        max_iterations=2,
        debug=False,
    )
    sequential_time = time.time() - start_time

    # Compare first frame
    batch_first = batch_result.led_values[0].astype(cp.float32) / 255.0
    sequential_first = sequential_result.led_values.astype(cp.float32) / 255.0

    max_diff = float(cp.max(cp.abs(batch_first - sequential_first)))
    rms_diff = float(cp.sqrt(cp.mean((batch_first - sequential_first) ** 2)))
    rms_batch = float(cp.sqrt(cp.mean(batch_first**2)))
    relative_error = rms_diff / rms_batch if rms_batch > 0 else float("inf")

    print("\n=== Results ===")
    print(f"Batch optimizer: {batch_time:.2f}s")
    print(f"Sequential optimizer: {sequential_time:.2f}s")
    print(f"Max difference: {max_diff:.8f}")
    print(f"RMS difference: {rms_diff:.8f}")
    print(f"Relative error: {relative_error*100:.6f}%")

    success = relative_error < 0.01  # 1% tolerance

    if success:
        print("✓ SUCCESS: Batch and sequential produce equivalent results")
    else:
        print("✗ FAILURE: Results differ significantly")

    return success


if __name__ == "__main__":
    test_208_leds_quick()
