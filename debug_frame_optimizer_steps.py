#!/usr/bin/env python3
"""
Debug frame optimizer step by step to find where DIA and dense diverge.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent / "src"))

import cupy as cp

from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.frame_optimizer import _calculate_atb
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def debug_frame_optimizer_steps():
    """Debug each step of frame optimizer to find where DIA and dense diverge."""

    # Load pattern file
    pattern_file = "diffusion_patterns/synthetic_2624_uint8_dia_10.0.npz"

    if not Path(pattern_file).exists():
        print(f"Pattern file not found: {pattern_file}")
        return

    print(f"Loading pattern file: {pattern_file}")
    data = np.load(pattern_file, allow_pickle=True)

    # Load components
    mixed_tensor_dict = data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    dia_dict = data["dia_matrix"].item()
    dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)

    ata_inverse_dense = data["ata_inverse"]
    ata_inverse_dia_dict = data["ata_inverse_dia"].item()
    ata_inverse_dia = DiagonalATAMatrix.from_dict(ata_inverse_dia_dict)

    print(f"Mixed tensor: {mixed_tensor.batch_size} LEDs, dtype={mixed_tensor.dtype}")
    print(f"DIA ATA inverse: bandwidth={ata_inverse_dia.bandwidth}, k={ata_inverse_dia.k}")

    # Create test frame (same as in frame optimizer test)
    target_frame = np.zeros((3, 480, 800), dtype=np.uint8)
    target_frame[0, 100:150, :] = 255  # Red stripe
    target_frame[1, 200:250, 200:600] = 128  # Green rectangle

    print(f"Target frame shape: {target_frame.shape}, dtype: {target_frame.dtype}")

    # Step 1: Calculate A^T @ b
    print("\n=== Step 1: A^T @ b Calculation ===")
    ATb_gpu = _calculate_atb(target_frame, mixed_tensor, debug=False)

    # Ensure ATb is in (3, led_count) format
    if ATb_gpu.shape[0] != 3:
        ATb_gpu = ATb_gpu.T

    print(f"A^T @ b shape: {ATb_gpu.shape}")
    print(f"A^T @ b range: [{float(cp.min(ATb_gpu)):.6f}, {float(cp.max(ATb_gpu)):.6f}]")

    # Step 2: Initialization using DIA ATA inverse
    print("\n=== Step 2a: DIA ATA Inverse Initialization ===")
    led_values_dia = ata_inverse_dia.multiply_3d(ATb_gpu)
    led_values_dia = led_values_dia.astype(cp.float32)
    led_values_dia = cp.clip(led_values_dia, 0.0, 1.0)

    print(f"DIA initialized values shape: {led_values_dia.shape}")
    print(f"DIA initialized values range: [{float(cp.min(led_values_dia)):.6f}, {float(cp.max(led_values_dia)):.6f}]")

    # Step 2b: Initialization using dense ATA inverse
    print("\n=== Step 2b: Dense ATA Inverse Initialization ===")
    ata_inverse_gpu = cp.asarray(ata_inverse_dense)
    led_values_dense = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_gpu)
    led_values_dense = led_values_dense.astype(cp.float32)
    led_values_dense = cp.clip(led_values_dense, 0.0, 1.0)

    print(f"Dense initialized values shape: {led_values_dense.shape}")
    print(
        f"Dense initialized values range: [{float(cp.min(led_values_dense)):.6f}, {float(cp.max(led_values_dense)):.6f}]"
    )

    # Compare initializations
    print("\n=== Initialization Comparison ===")
    init_diff = cp.abs(led_values_dia - led_values_dense)
    print(f"Max initialization difference: {float(cp.max(init_diff)):.6f}")
    print(f"Mean initialization difference: {float(cp.mean(init_diff)):.6f}")
    print(f"RMS initialization difference: {float(cp.sqrt(cp.mean(init_diff**2))):.6f}")

    # If initialization is different, that explains everything!
    if float(cp.max(init_diff)) > 1e-6:
        print("⚠️  INITIALIZATION DIFFERS SIGNIFICANTLY!")

        # Let's investigate why
        print("\n=== Investigating Initialization Difference ===")

        # Test the multiply operations directly
        test_result_dia = ata_inverse_dia.multiply_3d(ATb_gpu)
        test_result_dense = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_gpu)

        multiply_diff = cp.abs(test_result_dia - test_result_dense)
        print(f"Raw multiply max difference: {float(cp.max(multiply_diff)):.6f}")
        print(f"Raw multiply mean difference: {float(cp.mean(multiply_diff)):.6f}")

        if float(cp.max(multiply_diff)) > 1e-6:
            print("❌ DIA multiply operation differs from dense!")

            # Check if it's clipping that causes the issue
            print(f"DIA raw range: [{float(cp.min(test_result_dia)):.6f}, {float(cp.max(test_result_dia)):.6f}]")
            print(f"Dense raw range: [{float(cp.min(test_result_dense)):.6f}, {float(cp.max(test_result_dense)):.6f}]")

        else:
            print("✅ DIA multiply operation matches dense exactly")
            print("❌ Issue must be in clipping or subsequent processing")

            # Check clipping impact
            dia_clipped = cp.clip(test_result_dia, 0.0, 1.0)
            dense_clipped = cp.clip(test_result_dense, 0.0, 1.0)
            clipping_diff = cp.abs(dia_clipped - dense_clipped)
            print(f"Post-clipping max difference: {float(cp.max(clipping_diff)):.6f}")

    else:
        print("✅ Initializations match - issue must be in optimization loop")

        # Test one iteration of gradient descent
        print("\n=== Testing One Optimization Iteration ===")

        # Gradient: A^T A @ x - A^T @ b
        ATA_x_dia = dia_matrix.multiply_3d(led_values_dia)
        gradient_dia = ATA_x_dia - ATb_gpu

        ATA_x_dense = dia_matrix.multiply_3d(led_values_dense)  # Use same DIA matrix for fairness
        gradient_dense = ATA_x_dense - ATb_gpu

        grad_diff = cp.abs(gradient_dia - gradient_dense)
        print(f"Gradient max difference: {float(cp.max(grad_diff)):.6f}")
        print(f"Gradient mean difference: {float(cp.mean(grad_diff)):.6f}")


if __name__ == "__main__":
    debug_frame_optimizer_steps()
