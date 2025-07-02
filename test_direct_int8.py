#!/usr/bin/env python3
"""
Direct test of int8 operations bypassing frame optimizer normalization.

This tests the int8 kernels directly to verify proper scaling.
"""

import sys
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def test_direct_int8_operations():
    """Test int8 kernels directly with proper data types."""

    # Load test data
    patterns_path = "diffusion_patterns/baseline_realistic.npz"
    image_path = "flower_test.png"

    print(f"Loading patterns from: {patterns_path}")
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Convert mixed tensor to int8
    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    float32_mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    # Create int8 tensor
    int8_tensor = SingleBlockMixedSparseTensor(
        batch_size=float32_mixed_tensor.batch_size,
        channels=float32_mixed_tensor.channels,
        height=float32_mixed_tensor.height,
        width=float32_mixed_tensor.width,
        block_size=float32_mixed_tensor.block_size,
        device=float32_mixed_tensor.device,
        dtype=cp.uint8,
    )

    # Convert values: [0,1] -> [0,255]
    int8_values = (float32_mixed_tensor.sparse_values * 255.0).astype(cp.uint8)
    int8_tensor.sparse_values = int8_values
    int8_tensor.block_positions = float32_mixed_tensor.block_positions.copy()

    print(f"Int8 tensor: {int8_tensor.dtype}, values range [{int(cp.min(int8_values))}, {int(cp.max(int8_values))}]")

    # Load test image as uint8
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((800, 480))
    target_image_uint8 = np.array(image, dtype=np.uint8)
    target_planar_uint8 = target_image_uint8.transpose(2, 0, 1)  # (3, H, W)
    target_gpu_uint8 = cp.asarray(target_planar_uint8)

    print(f"Target image: {target_gpu_uint8.dtype}, shape {target_gpu_uint8.shape}")
    print(f"Target range: [{int(cp.min(target_gpu_uint8))}, {int(cp.max(target_gpu_uint8))}]")

    # Test A^T @ b with int8 kernel
    print("\n=== Testing Int8 A^T @ b Operation ===")
    ATb_int8 = int8_tensor.transpose_dot_product_3d(target_gpu_uint8)
    print(f"A^T @ b result shape: {ATb_int8.shape}")
    print(f"A^T @ b result dtype: {ATb_int8.dtype}")
    print(f"A^T @ b range: [{float(cp.min(ATb_int8)):.6f}, {float(cp.max(ATb_int8)):.6f}]")
    print(f"A^T @ b mean: {float(cp.mean(ATb_int8)):.6f}")

    # Compare with float32 kernel
    print("\n=== Comparing with Float32 Operation ===")
    target_float32 = target_gpu_uint8.astype(cp.float32) / 255.0
    ATb_float32 = float32_mixed_tensor.transpose_dot_product_3d(target_float32)
    print(f"Float32 A^T @ b range: [{float(cp.min(ATb_float32)):.6f}, {float(cp.max(ATb_float32)):.6f}]")
    print(f"Float32 A^T @ b mean: {float(cp.mean(ATb_float32)):.6f}")

    # Check scaling relationship
    # Int8 kernel should include / (255 * 255) normalization
    # So ATb_int8 should be roughly ATb_float32 * (1/255) due to target scaling difference
    scale_factor = float(cp.mean(ATb_int8) / cp.mean(ATb_float32))
    expected_scale = 1.0 / 255.0  # Because target is 255x larger in int8
    print("\nScaling analysis:")
    print(f"Actual scale factor (int8/float32): {scale_factor:.6f}")
    print(f"Expected scale factor (1/255): {expected_scale:.6f}")
    print(f"Scale ratio (actual/expected): {scale_factor / expected_scale:.6f}")

    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    led_values = cp.full((int8_tensor.batch_size, 3), 128, dtype=cp.float32)  # Mid-range values
    print(
        f"Test LED values: {led_values.dtype}, range [{float(cp.min(led_values)):.1f}, {float(cp.max(led_values)):.1f}]"
    )

    rendered_int8 = int8_tensor.forward_pass_3d(led_values)
    print(f"Rendered frame: {rendered_int8.dtype}, shape {rendered_int8.shape}")
    print(f"Rendered range: [{float(cp.min(rendered_int8)):.6f}, {float(cp.max(rendered_int8)):.6f}]")
    print(f"Rendered mean: {float(cp.mean(rendered_int8)):.6f}")

    # Also test float32 forward pass for comparison
    led_values_norm = led_values / 255.0  # Normalize for float32 tensor
    rendered_float32 = float32_mixed_tensor.forward_pass_3d(led_values_norm)
    print(f"Float32 rendered range: [{float(cp.min(rendered_float32)):.6f}, {float(cp.max(rendered_float32)):.6f}]")
    print(f"Float32 rendered mean: {float(cp.mean(rendered_float32)):.6f}")

    # Check forward pass scaling
    fwd_scale_factor = float(cp.mean(rendered_int8) / cp.mean(rendered_float32))
    print(f"Forward pass scale factor (int8/float32): {fwd_scale_factor:.6f}")


if __name__ == "__main__":
    test_direct_int8_operations()
