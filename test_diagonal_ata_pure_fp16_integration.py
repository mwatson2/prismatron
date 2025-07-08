#!/usr/bin/env python3
"""
Test integration of pure FP16 kernel into DiagonalATAMatrix class.

This test validates:
1. Pure FP16 kernel availability in DiagonalATAMatrix
2. Correct parameter validation
3. Basic functionality with small test case
"""

import sys
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix


def create_test_diffusion_matrix(led_count: int = 100, bandwidth: int = 10) -> sp.csc_matrix:
    """Create a simple test diffusion matrix."""
    pixels = 480 * 800  # Standard frame size
    total_cols = led_count * 3  # RGB channels

    # Create sparse test matrix with known structure
    matrix = sp.lil_matrix((pixels, total_cols), dtype=np.float32)

    # Add some non-zero patterns around each LED
    for led_idx in range(led_count):
        for channel in range(3):
            col_idx = led_idx * 3 + channel

            # Add some diagonal-like patterns
            for offset in range(-bandwidth, bandwidth + 1):
                pixel_idx = (led_idx * 10 + offset) % pixels  # Wrap around
                if 0 <= pixel_idx < pixels:
                    matrix[pixel_idx, col_idx] = np.random.random() * 0.1

    return matrix.tocsc()


def test_pure_fp16_integration():
    """Test pure FP16 kernel integration."""
    print("Testing DiagonalATAMatrix pure FP16 kernel integration...")

    # Create test data
    led_count = 50  # Small test case
    diffusion_matrix = create_test_diffusion_matrix(led_count)

    # Create DiagonalATAMatrix with FP16 output
    dia_matrix = DiagonalATAMatrix(led_count, output_dtype=cp.float16)

    print("Building ATA matrix from diffusion matrix...")
    dia_matrix.build_from_diffusion_matrix(diffusion_matrix)

    # Create test LED values in FP16
    led_values = cp.random.random((3, led_count), dtype=cp.float32).astype(cp.float16) * 255

    print("Testing pure FP16 kernel (basic)...")

    # Test with pure FP16 kernel (basic)
    try:
        result_pure_basic = dia_matrix.multiply_3d(
            led_values, use_custom_kernel=True, optimized_kernel=False, output_dtype=cp.float16, use_pure_fp16=True
        )
        print(f"✓ Pure FP16 basic kernel: shape {result_pure_basic.shape}, dtype {result_pure_basic.dtype}")
    except Exception as e:
        print(f"✗ Pure FP16 basic kernel failed: {e}")
        return False

    print("Testing pure FP16 kernel (optimized)...")

    # Test with pure FP16 kernel (optimized)
    try:
        result_pure_optimized = dia_matrix.multiply_3d(
            led_values, use_custom_kernel=True, optimized_kernel=True, output_dtype=cp.float16, use_pure_fp16=True
        )
        print(f"✓ Pure FP16 optimized kernel: shape {result_pure_optimized.shape}, dtype {result_pure_optimized.dtype}")
    except Exception as e:
        print(f"✗ Pure FP16 optimized kernel failed: {e}")
        return False

    # Test comparison with mixed precision
    print("Testing mixed precision comparison...")
    try:
        result_mixed = dia_matrix.multiply_3d(
            led_values,
            use_custom_kernel=True,
            optimized_kernel=True,
            output_dtype=cp.float16,
            use_pure_fp16=False,  # Mixed precision
        )
        print(f"✓ Mixed precision kernel: shape {result_mixed.shape}, dtype {result_mixed.dtype}")

        # Compare results
        diff = cp.abs(result_pure_optimized.astype(cp.float32) - result_mixed.astype(cp.float32))
        max_diff = cp.max(diff)
        print(f"✓ Max difference between pure FP16 and mixed precision: {max_diff:.6f}")

    except Exception as e:
        print(f"✗ Mixed precision comparison failed: {e}")
        return False

    # Test validation errors
    print("Testing validation...")

    # Test invalid pure FP16 with FP32 output
    try:
        dia_matrix.multiply_3d(
            led_values,
            use_custom_kernel=True,
            optimized_kernel=True,
            output_dtype=cp.float32,  # FP32 output
            use_pure_fp16=True,  # Should fail
        )
        print("✗ Validation error not caught")
        return False
    except ValueError as e:
        print(f"✓ Validation error correctly caught: {e}")

    print("✓ All tests passed!")
    return True


if __name__ == "__main__":
    if test_pure_fp16_integration():
        print("\n🎉 DiagonalATAMatrix pure FP16 kernel integration successful!")
        sys.exit(0)
    else:
        print("\n❌ DiagonalATAMatrix pure FP16 kernel integration failed!")
        sys.exit(1)
