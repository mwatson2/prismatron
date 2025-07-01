#!/usr/bin/env python3
"""
Test optimization with int8 data and int8 kernels for proper scaling.

This converts the float32 patterns to int8 [0,255] and uses int8 kernels
that include the built-in 255*255 normalization.
"""

import sys
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def convert_to_int8_tensor(
    float32_tensor: SingleBlockMixedSparseTensor,
) -> SingleBlockMixedSparseTensor:
    """Convert float32 tensor [0,1] to int8 tensor [0,255]."""
    print("Converting float32 tensor to int8...")

    # Create new int8 tensor
    int8_tensor = SingleBlockMixedSparseTensor(
        batch_size=float32_tensor.batch_size,
        channels=float32_tensor.channels,
        height=float32_tensor.height,
        width=float32_tensor.width,
        block_size=float32_tensor.block_size,
        device=float32_tensor.device,
        dtype=cp.uint8,  # Use int8/uint8 dtype
    )

    # Convert values: [0,1] -> [0,255]
    float32_values = (
        float32_tensor.sparse_values
    )  # Shape: (channels, batch_size, block_size, block_size)
    int8_values = (float32_values * 255.0).astype(cp.uint8)

    # Copy converted data
    int8_tensor.sparse_values = int8_values
    int8_tensor.block_positions = float32_tensor.block_positions.copy()

    print(
        f"Converted tensor: {int8_tensor.dtype}, values range [{int(cp.min(int8_values))}, {int(cp.max(int8_values))}]"
    )
    return int8_tensor


def test_with_int8_scaling():
    """Test optimization with int8 data and int8 kernels."""

    # Load test data
    patterns_path = "diffusion_patterns/baseline_realistic.npz"
    image_path = "flower_test.png"

    print(f"Loading patterns from: {patterns_path}")
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Load and convert mixed tensor to int8
    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    float32_mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
    int8_mixed_tensor = convert_to_int8_tensor(float32_mixed_tensor)

    # Load CSC matrix and convert to int8 scale
    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_matrix = diffusion_csc.to_csc_matrix()

    # Convert CSC matrix: [0,1] -> [0,255] to match int8 scaling
    int8_csc_matrix = (csc_matrix * 255.0).astype(np.uint8)
    print(
        f"CSC matrix converted: {int8_csc_matrix.dtype}, range [{int8_csc_matrix.data.min()}, {int8_csc_matrix.data.max()}]"
    )

    # Build DIA matrix from int8 CSC
    dia_matrix = DiagonalATAMatrix(led_count=int8_mixed_tensor.batch_size)
    led_positions = patterns_data["led_positions"]

    # Suppress DIA build output
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    dia_matrix.build_from_diffusion_matrix(int8_csc_matrix, led_positions)
    sys.stdout = old_stdout

    # Load test image and convert to int8
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((800, 480))
    target_image_uint8 = np.array(image, dtype=np.uint8)  # Keep as uint8 [0,255]

    print(f"\n=== Testing with Int8 Data and Kernels ===")
    print(
        f"Target image dtype: {target_image_uint8.dtype}, range: [{target_image_uint8.min()}, {target_image_uint8.max()}]"
    )
    print(f"Mixed tensor dtype: {int8_mixed_tensor.dtype}")

    # Test A^T @ b scaling first
    target_planar = target_image_uint8.transpose(2, 0, 1)  # (3, H, W), keep uint8
    target_gpu = cp.asarray(target_planar)

    ATb = int8_mixed_tensor.transpose_dot_product_3d(target_gpu)
    print(f"Int8 A^T @ b range: [{float(cp.min(ATb)):.6f}, {float(cp.max(ATb)):.6f}]")
    print(f"Int8 A^T @ b mean: {float(cp.mean(ATb)):.6f}")

    # Test optimization with int8 data
    result = optimize_frame_led_values(
        target_frame=target_image_uint8,
        AT_matrix=int8_mixed_tensor,
        ATA_matrix=dia_matrix,
        max_iterations=10,
        convergence_threshold=1e-6,
        step_size_scaling=0.8,  # Normal step size scaling for int8
        compute_error_metrics=True,
        debug=True,
    )

    print(f"\n=== Results with Int8 Scaling ===")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(
        f"LED values range: [{result.led_values.min():.3f}, {result.led_values.max():.3f}]"
    )

    if result.error_metrics:
        print(f"MSE: {result.error_metrics.get('mse', 'N/A'):.6f}")
        print(f"MAE: {result.error_metrics.get('mae', 'N/A'):.6f}")
        print(f"PSNR: {result.error_metrics.get('psnr', 'N/A'):.2f}")

    if result.step_sizes is not None:
        print(f"Step sizes: {result.step_sizes}")
        print(
            f"Step size range: [{np.min(result.step_sizes):.6f}, {np.max(result.step_sizes):.6f}]"
        )
        print(f"Step size mean: {np.mean(result.step_sizes):.6f}")


if __name__ == "__main__":
    test_with_int8_scaling()
