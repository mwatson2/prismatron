#!/usr/bin/env python3
"""
Debug A^T A scaling and g^T @ A^T A @ g calculation.

This script examines the A^T A matrix values and step size calculation
to identify scaling issues in the optimization.
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


def analyze_ata_scaling():
    """Analyze A^T A matrix scaling and step size calculation."""

    print("=== ANALYZING A^T A SCALING ===")

    # Load test data
    patterns_path = "diffusion_patterns/baseline_realistic.npz"
    image_path = "flower_test.png"

    print(f"Loading patterns from: {patterns_path}")
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Load CSC matrix to check A values
    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_matrix = diffusion_csc.to_csc_matrix()

    print(f"CSC matrix (A) shape: {csc_matrix.shape}")
    print(
        f"CSC matrix (A) values range: [{csc_matrix.data.min():.6f}, {csc_matrix.data.max():.6f}]"
    )
    print(f"CSC matrix (A) nnz: {csc_matrix.nnz}")

    # Build DIA matrix and examine A^T A values
    led_positions = patterns_data["led_positions"]
    led_count = len(led_positions)

    print(f"\nBuilding DIA matrix for {led_count} LEDs...")
    dia_matrix = DiagonalATAMatrix(led_count=led_count)
    dia_matrix.build_from_diffusion_matrix(csc_matrix, led_positions)

    print(f"DIA matrix shape: {dia_matrix.dia_data_cpu.shape}")
    print(
        f"DIA matrix values range: [{dia_matrix.dia_data_cpu.min():.6f}, {dia_matrix.dia_data_cpu.max():.6f}]"
    )
    print(f"DIA matrix nnz: {np.count_nonzero(dia_matrix.dia_data_cpu)}")

    # Check diagonal values (should be largest)
    main_diagonal_band = (
        dia_matrix.dia_data_cpu.shape[1] // 2
    )  # Middle band is main diagonal
    main_diagonal_values = dia_matrix.dia_data_cpu[:, main_diagonal_band, :]
    print(
        f"Main diagonal range: [{main_diagonal_values.min():.6f}, {main_diagonal_values.max():.6f}]"
    )
    print(f"Main diagonal mean: {main_diagonal_values.mean():.6f}")

    # Load mixed tensor and test image
    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((800, 480))
    target_image = np.array(image, dtype=np.uint8)
    target_planar = target_image.astype(np.float32) / 255.0
    target_planar = target_planar.transpose(2, 0, 1)  # (3, H, W)

    # Calculate A^T @ b
    target_gpu = cp.asarray(target_planar)
    ATb = mixed_tensor.transpose_dot_product_3d(target_gpu)  # Shape: (led_count, 3)
    ATb = ATb.T  # Convert to (3, led_count)
    print(f"\nA^T @ b shape: {ATb.shape}")
    print(f"A^T @ b range: [{float(cp.min(ATb)):.6f}, {float(cp.max(ATb)):.6f}]")
    print(f"A^T @ b mean: {float(cp.mean(ATb)):.6f}")

    # Test with different LED value scales
    test_scales = [0.1, 0.5, 1.0, 2.0, 10.0]

    for scale in test_scales:
        print(f"\n=== Testing with LED values scaled by {scale} ===")

        # Create test LED values
        led_values = cp.full((3, led_count), scale, dtype=cp.float32)

        # Convert to RCM order for DIA matrix
        led_values_rcm = dia_matrix.reorder_led_values_to_rcm(cp.asnumpy(led_values))
        led_values_rcm_gpu = cp.asarray(led_values_rcm)

        # Compute A^T A @ x
        ATA_x = dia_matrix.multiply_3d(led_values_rcm_gpu)
        if not isinstance(ATA_x, cp.ndarray):
            ATA_x = cp.asarray(ATA_x)

        print(f"A^T A @ x shape: {ATA_x.shape}")
        print(
            f"A^T A @ x range: [{float(cp.min(ATA_x)):.6f}, {float(cp.max(ATA_x)):.6f}]"
        )
        print(f"A^T A @ x mean: {float(cp.mean(ATA_x)):.6f}")

        # Test gradient calculation
        ATb_rcm = dia_matrix.reorder_led_values_to_rcm(cp.asnumpy(ATb))
        ATb_rcm_gpu = cp.asarray(ATb_rcm)

        gradient = ATA_x - ATb_rcm_gpu
        print(
            f"Gradient range: [{float(cp.min(gradient)):.6f}, {float(cp.max(gradient)):.6f}]"
        )

        # Test step size calculation
        g_dot_g = cp.sum(gradient * gradient)
        print(f"g^T @ g: {float(g_dot_g):.6f}")

        # Test g^T @ A^T A @ g calculation
        g_dot_ATA_g_per_channel = dia_matrix.g_ata_g_3d(gradient)
        if not isinstance(g_dot_ATA_g_per_channel, cp.ndarray):
            g_dot_ATA_g_per_channel = cp.asarray(g_dot_ATA_g_per_channel)
        g_dot_ATA_g = cp.sum(g_dot_ATA_g_per_channel)

        print(
            f"g^T @ A^T A @ g per channel: {[float(x) for x in g_dot_ATA_g_per_channel]}"
        )
        print(f"g^T @ A^T A @ g total: {float(g_dot_ATA_g):.6f}")

        # Calculate step size
        if g_dot_ATA_g > 0:
            step_size = float(0.8 * g_dot_g / g_dot_ATA_g)
        else:
            step_size = 0.01

        print(f"Step size: {step_size:.6f}")
        print(
            f"Step size ratio (g^T g / g^T A^T A g): {float(g_dot_g / g_dot_ATA_g):.6f}"
        )

        # Check if step size makes sense
        gradient_norm = float(cp.linalg.norm(gradient))
        update_norm = step_size * gradient_norm
        print(f"Gradient norm: {gradient_norm:.6f}")
        print(f"Update norm: {update_norm:.6f}")
        print(f"Update/Value ratio: {update_norm / (scale * 3 * led_count**0.5):.6f}")


def test_manual_ata_calculation():
    """Test manual A^T A calculation to verify DIA matrix."""

    print("\n=== MANUAL A^T A VERIFICATION ===")

    patterns_data = np.load(
        "diffusion_patterns/baseline_realistic.npz", allow_pickle=True
    )

    # Load CSC matrix
    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_matrix = diffusion_csc.to_csc_matrix()

    led_count = csc_matrix.shape[1] // 3

    # Manual calculation of A^T A for first few LEDs
    print(f"Manual A^T A calculation for first 3 LEDs...")

    for led_i in range(3):
        for led_j in range(3):
            ata_ij_total = 0.0

            for channel in range(3):
                col_i = led_i * 3 + channel
                col_j = led_j * 3 + channel

                # A^T A[i,j,c] = A[:,i,c]^T @ A[:,j,c]
                ata_ij_channel = csc_matrix[:, col_i].T @ csc_matrix[:, col_j]
                ata_ij_total += float(ata_ij_channel)

            print(f"A^T A[{led_i},{led_j}] = {ata_ij_total:.2f}")

    # Compare with DIA matrix calculation
    print(f"\nBuilding DIA matrix for comparison...")
    led_positions = patterns_data["led_positions"]
    dia_matrix = DiagonalATAMatrix(led_count=led_count)

    # Suppress build output
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    dia_matrix.build_from_diffusion_matrix(csc_matrix, led_positions)
    sys.stdout = old_stdout

    # Test DIA calculation with simple test vector
    test_vector = cp.zeros((3, led_count), dtype=cp.float32)
    test_vector[0, 0] = 1.0  # Set first LED, first channel to 1.0

    test_vector_rcm = dia_matrix.reorder_led_values_to_rcm(cp.asnumpy(test_vector))
    test_vector_rcm_gpu = cp.asarray(test_vector_rcm)

    result = dia_matrix.multiply_3d(test_vector_rcm_gpu)
    result_spatial = dia_matrix.reorder_led_values_from_rcm(cp.asnumpy(result))

    print(f"DIA matrix test - first column of A^T A:")
    for i in range(min(5, led_count)):
        print(f"  A^T A[{i},0] = {result_spatial[0, i]:.2f}")


if __name__ == "__main__":
    analyze_ata_scaling()
    test_manual_ata_calculation()
