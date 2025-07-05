#!/usr/bin/env python3
"""
Test optimization with proper int8 data paths and verify convergence.

This modifies the frame optimizer to handle int8 data properly without
converting to float32, allowing the int8 kernels to work as intended.
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


def convert_to_int8_tensor(
    float32_tensor: SingleBlockMixedSparseTensor,
) -> SingleBlockMixedSparseTensor:
    """Convert float32 tensor [0,1] to int8 tensor [0,255]."""
    print("Converting float32 tensor to int8...")

    int8_tensor = SingleBlockMixedSparseTensor(
        batch_size=float32_tensor.batch_size,
        channels=float32_tensor.channels,
        height=float32_tensor.height,
        width=float32_tensor.width,
        block_size=float32_tensor.block_size,
        device=float32_tensor.device,
        dtype=cp.uint8,
    )

    # Convert values: [0,1] -> [0,255]
    int8_values = (float32_tensor.sparse_values * 255.0).astype(cp.uint8)
    int8_tensor.sparse_values = int8_values
    int8_tensor.block_positions = float32_tensor.block_positions.copy()

    print(f"Converted tensor: values range [{int(cp.min(int8_values))}, {int(cp.max(int8_values))}]")
    return int8_tensor


def optimize_frame_int8(
    target_frame_uint8: np.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
    ata_matrix: DiagonalATAMatrix,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-6,
    step_size_scaling: float = 0.8,
    debug: bool = False,
):
    """
    Optimize LED values using int8 data path without float32 conversion.

    This bypasses the frame optimizer's float32 normalization to test
    the int8 kernels with proper scaling.
    """

    print("=== Int8 Optimization ===")
    print(f"Target frame shape: {target_frame_uint8.shape}, dtype: {target_frame_uint8.dtype}")
    print(f"AT matrix dtype: {at_matrix.dtype}")

    # Keep target as uint8 and convert to planar format
    if target_frame_uint8.shape == (480, 800, 3):
        target_planar_uint8 = target_frame_uint8.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    else:
        target_planar_uint8 = target_frame_uint8

    print(f"Target planar shape: {target_planar_uint8.shape}")
    print(f"Target range: [{target_planar_uint8.min()}, {target_planar_uint8.max()}]")

    # Step 1: Calculate A^T @ b using int8 kernel
    print("\n--- Step 1: A^T @ b calculation ---")
    target_gpu = cp.asarray(target_planar_uint8)
    ATb = at_matrix.transpose_dot_product_3d(target_gpu)  # Uses int8 kernel with normalization
    ATb = cp.asnumpy(ATb)  # Shape: (led_count, 3)

    print(f"A^T @ b shape: {ATb.shape}")
    print(f"A^T @ b range: [{ATb.min():.6f}, {ATb.max():.6f}]")
    print(f"A^T @ b mean: {ATb.mean():.6f}")

    # Convert to (3, led_count) format for optimization
    if ATb.shape[0] != 3:
        ATb = ATb.T
        print(f"Transposed A^T @ b to shape: {ATb.shape}")

    led_count = ATb.shape[1]

    # Step 2: Initialize LED values in range [0,1] (will be scaled to [0,255] for output)
    print("\n--- Step 2: Initialize LED values ---")
    led_values_normalized = np.full((3, led_count), 0.5, dtype=np.float32)
    print(f"Initial LED values shape: {led_values_normalized.shape}")
    print(f"Initial LED values range: [{led_values_normalized.min():.3f}, {led_values_normalized.max():.3f}]")

    # Step 3: Convert to RCM order for DIA matrix
    print("\n--- Step 3: RCM ordering ---")
    ATb_rcm = ata_matrix.reorder_led_values_to_rcm(ATb)
    led_values_rcm = ata_matrix.reorder_led_values_to_rcm(led_values_normalized)

    # Step 4: GPU transfer
    ATb_gpu = cp.asarray(ATb_rcm)
    led_values_gpu = cp.asarray(led_values_rcm)

    # Step 5: Optimization loop
    print("\n--- Step 4: Optimization loop ---")
    step_sizes = []

    for iteration in range(max_iterations):
        if debug:
            print(f"\nIteration {iteration + 1}")

        # Compute A^T A @ x
        ATA_x = ata_matrix.multiply_3d(led_values_gpu)
        if not isinstance(ATA_x, cp.ndarray):
            ATA_x = cp.asarray(ATA_x)

        # Compute gradient
        gradient = ATA_x - ATb_gpu

        # Compute step size
        g_dot_g = cp.sum(gradient * gradient)
        g_dot_ATA_g_per_channel = ata_matrix.g_ata_g_3d(gradient)
        if not isinstance(g_dot_ATA_g_per_channel, cp.ndarray):
            g_dot_ATA_g_per_channel = cp.asarray(g_dot_ATA_g_per_channel)
        g_dot_ATA_g = cp.sum(g_dot_ATA_g_per_channel)

        if g_dot_ATA_g > 0:
            step_size = float(step_size_scaling * g_dot_g / g_dot_ATA_g)
        else:
            step_size = 0.01
            print("WARNING: g^T @ A^T A @ g <= 0, using fallback step size")

        step_sizes.append(step_size)

        if debug:
            print(f"  Gradient range: [{float(cp.min(gradient)):.6f}, {float(cp.max(gradient)):.6f}]")
            print(f"  g^T @ g: {float(g_dot_g):.6f}")
            print(f"  g^T @ A^T A @ g: {float(g_dot_ATA_g):.6f}")
            print(f"  Step size: {step_size:.6f}")

        # Update LED values
        led_values_new = cp.clip(led_values_gpu - step_size * gradient, 0, 1)

        if debug:
            print(f"  LED values range: [{float(cp.min(led_values_new)):.3f}, {float(cp.max(led_values_new)):.3f}]")

        # Check convergence
        delta = cp.linalg.norm(led_values_new - led_values_gpu)
        if debug:
            print(f"  Delta: {float(delta):.6f} (threshold: {convergence_threshold:.6f})")

        if delta < convergence_threshold:
            print(f"*** CONVERGED after {iteration + 1} iterations ***")
            led_values_gpu = led_values_new
            converged = True
            break

        led_values_gpu = led_values_new
    else:
        print(f"*** DID NOT CONVERGE after {max_iterations} iterations ***")
        print(f"Final delta: {float(delta):.6f}")
        converged = False

    # Step 6: Convert back to spatial order and scale to [0,255]
    print("\n--- Step 5: Final conversion ---")
    led_values_spatial = ata_matrix.reorder_led_values_from_rcm(cp.asnumpy(led_values_gpu))
    led_values_final = (led_values_spatial * 255.0).astype(np.uint8)

    print(f"Final LED values spatial shape: {led_values_spatial.shape}")
    print(f"Final LED values range: [{led_values_final.min()}, {led_values_final.max()}]")

    return {
        "converged": converged,
        "iterations": iteration + 1,
        "led_values": led_values_final,
        "step_sizes": step_sizes,
        "final_delta": float(delta) if "delta" in locals() else None,
    }


def test_int8_convergence():
    """Test int8 optimization convergence."""

    # Load test data
    patterns_path = "diffusion_patterns/baseline_realistic.npz"
    image_path = "flower_test.png"

    print(f"Loading patterns from: {patterns_path}")
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Convert mixed tensor to int8
    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    float32_mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
    int8_mixed_tensor = convert_to_int8_tensor(float32_mixed_tensor)

    # Build DIA matrix from original float32 CSC (keep as float32)
    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_matrix = diffusion_csc.to_csc_matrix()

    dia_matrix = DiagonalATAMatrix(led_count=int8_mixed_tensor.batch_size)
    led_positions = patterns_data["led_positions"]

    # Suppress DIA build output
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    dia_matrix.build_from_diffusion_matrix(csc_matrix, led_positions)
    sys.stdout = old_stdout

    # Load test image as uint8
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((800, 480))
    target_image_uint8 = np.array(image, dtype=np.uint8)

    # Test optimization
    result = optimize_frame_int8(
        target_frame_uint8=target_image_uint8,
        at_matrix=int8_mixed_tensor,
        ata_matrix=dia_matrix,
        max_iterations=10,
        convergence_threshold=1e-6,
        step_size_scaling=0.8,
        debug=True,
    )

    print("\n=== RESULTS ===")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final delta: {result['final_delta']:.6f}")
    print(f"LED values range: [{result['led_values'].min()}, {result['led_values'].max()}]")
    print(f"Step sizes: {result['step_sizes']}")
    print(f"Step size range: [{np.min(result['step_sizes']):.6f}, {np.max(result['step_sizes']):.6f}]")
    print(f"Step size mean: {np.mean(result['step_sizes']):.6f}")


if __name__ == "__main__":
    test_int8_convergence()
