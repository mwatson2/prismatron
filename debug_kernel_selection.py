#!/usr/bin/env python3
"""
Debug kernel selection and verify normalization behavior.
"""

import sys
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def debug_kernel_behavior():
    """Debug which kernel is being used and its behavior."""

    # Load test data
    patterns_path = "diffusion_patterns/baseline_realistic.npz"
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Create simple test case
    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    float32_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    # Create int8 tensor with known values
    int8_tensor = SingleBlockMixedSparseTensor(
        batch_size=2,  # Just 2 LEDs for simple testing
        channels=1,  # Just 1 channel
        height=float32_tensor.height,
        width=float32_tensor.width,
        block_size=float32_tensor.block_size,
        device=float32_tensor.device,
        dtype=cp.uint8,
    )

    # Set known pattern values: 255 (max) and 128 (half)
    test_block_255 = cp.full(
        (float32_tensor.block_size, float32_tensor.block_size), 255, dtype=cp.uint8
    )
    test_block_128 = cp.full(
        (float32_tensor.block_size, float32_tensor.block_size), 128, dtype=cp.uint8
    )

    # Set blocks at known positions
    int8_tensor.set_block(0, 0, 100, 100, test_block_255)  # LED 0, channel 0
    int8_tensor.set_block(1, 0, 200, 200, test_block_128)  # LED 1, channel 0

    # Create simple target: all 255 (max)
    target_255 = cp.full(
        (1, float32_tensor.height, float32_tensor.width), 255, dtype=cp.uint8
    )

    print("=== Debug Kernel Selection and Behavior ===")
    print(f"Int8 tensor dtype: {int8_tensor.dtype}")
    print(f"Block 0 (255): {test_block_255[0, 0]} (should be 255)")
    print(f"Block 1 (128): {test_block_128[0, 0]} (should be 128)")
    print(
        f"Target dtype: {target_255.dtype}, value: {target_255[0, 0, 0]} (should be 255)"
    )

    # Test kernel call
    result = int8_tensor.transpose_dot_product_3d(target_255)
    print(f"\nKernel result shape: {result.shape}")
    print(f"Kernel result dtype: {result.dtype}")
    print(f"Result for LED 0 (255 block): {float(result[0, 0])}")
    print(f"Result for LED 1 (128 block): {float(result[1, 0])}")

    # Calculate expected values
    block_pixels = float32_tensor.block_size * float32_tensor.block_size

    # If normalization is applied: result = (block_value * target_value) / (255 * 255)
    expected_255_normalized = (255 * 255) / (255.0 * 255.0)  # Should be 1.0
    expected_128_normalized = (128 * 255) / (255.0 * 255.0)  # Should be 0.5019...

    # If no normalization: result = block_value * target_value
    expected_255_raw = 255 * 255  # Should be 65025
    expected_128_raw = 128 * 255  # Should be 32640

    print(f"\nExpected with normalization:")
    print(f"  LED 0: {expected_255_normalized * block_pixels:.2f} (block_pixels * 1.0)")
    print(
        f"  LED 1: {expected_128_normalized * block_pixels:.2f} (block_pixels * ~0.502)"
    )

    print(f"\nExpected without normalization:")
    print(f"  LED 0: {expected_255_raw * block_pixels:.0f} (block_pixels * 65025)")
    print(f"  LED 1: {expected_128_raw * block_pixels:.0f} (block_pixels * 32640)")

    # Check which expectation matches
    actual_0 = float(result[0, 0])
    actual_1 = float(result[1, 0])

    ratio_0 = actual_0 / (block_pixels * expected_255_normalized)
    ratio_1 = actual_1 / (block_pixels * expected_128_normalized)

    print(f"\nActual vs normalized expectation ratios:")
    print(f"  LED 0 ratio: {ratio_0:.6f} (should be ~1.0 if normalized)")
    print(f"  LED 1 ratio: {ratio_1:.6f} (should be ~1.0 if normalized)")

    if abs(ratio_0 - 1.0) < 0.01 and abs(ratio_1 - 1.0) < 0.01:
        print("✓ Kernel IS applying 255*255 normalization")
    else:
        print("✗ Kernel is NOT applying 255*255 normalization")
        print(f"Raw scale factor: {actual_0 / (block_pixels * 255 * 255):.10f}")


if __name__ == "__main__":
    debug_kernel_behavior()
