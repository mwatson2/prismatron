#!/usr/bin/env python3
"""
Debug optimization steps to trace exactly what happens during optimization.

This script runs detailed optimization with step-by-step reporting to identify
convergence issues and verify mathematical correctness.
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


def detailed_optimization_trace(
    target_frame: np.ndarray,
    AT_matrix,
    ATA_matrix,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-6,
    step_size_scaling: float = 0.8,
):
    """
    Detailed trace of optimization steps with mathematical verification.
    """

    print("=== DETAILED OPTIMIZATION TRACE ===")
    print(f"Target frame shape: {target_frame.shape}")
    print(f"Target frame dtype: {target_frame.dtype}")
    print(f"Target frame range: [{target_frame.min()}, {target_frame.max()}]")

    # Convert target frame to planar format [0,1]
    if target_frame.shape == (480, 800, 3):
        target_planar = target_frame.astype(np.float32) / 255.0
        target_planar = target_planar.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    else:
        target_planar = target_frame.astype(np.float32) / 255.0

    print(f"Target planar shape: {target_planar.shape}")
    print(
        f"Target planar range: [{target_planar.min():.6f}, {target_planar.max():.6f}]"
    )

    # Step 1: Calculate A^T @ b
    print("\n=== STEP 1: Calculate A^T @ b ===")

    if isinstance(AT_matrix, SingleBlockMixedSparseTensor):
        print("Using SingleBlockMixedSparseTensor for A^T @ b")
        target_gpu = cp.asarray(target_planar)  # Shape: (3, height, width)
        print(f"Target GPU shape: {target_gpu.shape}")

        ATb = AT_matrix.transpose_dot_product_3d(target_gpu)  # Shape: (led_count, 3)
        ATb = cp.asnumpy(ATb)
        print(f"A^T @ b result shape: {ATb.shape}")
        print(f"A^T @ b range: [{ATb.min():.6f}, {ATb.max():.6f}]")

        # Convert to (3, led_count) format for consistency
        if ATb.shape[0] != 3:
            ATb = ATb.T
            print(f"Transposed A^T @ b to shape: {ATb.shape}")

    elif isinstance(AT_matrix, LEDDiffusionCSCMatrix):
        print("Using LEDDiffusionCSCMatrix for A^T @ b")
        csc_A = AT_matrix.to_csc_matrix()
        print(f"CSC matrix shape: {csc_A.shape}")
        print(f"CSC matrix nnz: {csc_A.nnz}")

        led_count = csc_A.shape[1] // 3
        ATb_result = np.zeros((led_count, 3), dtype=np.float32)

        for channel in range(3):
            target_channel = target_planar[channel].flatten()
            channel_cols = np.arange(channel, csc_A.shape[1], 3)
            A_channel = csc_A[:, channel_cols]
            ATb_channel = A_channel.T @ target_channel
            ATb_result[:, channel] = ATb_channel

            print(
                f"Channel {channel}: A_channel shape {A_channel.shape}, ATb range [{ATb_channel.min():.6f}, {ATb_channel.max():.6f}]"
            )

        ATb = ATb_result.T  # Convert to (3, led_count)
        print(f"Final A^T @ b shape: {ATb.shape}")
        print(f"Final A^T @ b range: [{ATb.min():.6f}, {ATb.max():.6f}]")

    else:
        print(f"Unknown AT_matrix type: {type(AT_matrix)}")
        return

    led_count = ATb.shape[1]
    print(f"LED count: {led_count}")

    # Step 2: Initialize LED values
    print("\n=== STEP 2: Initialize LED values ===")
    led_values_normalized = np.full((3, led_count), 0.5, dtype=np.float32)
    print(f"Initial LED values shape: {led_values_normalized.shape}")
    print(
        f"Initial LED values range: [{led_values_normalized.min():.6f}, {led_values_normalized.max():.6f}]"
    )

    # Handle ordering for DIA matrix
    if isinstance(ATA_matrix, DiagonalATAMatrix):
        print("Converting to RCM order for DIA matrix")
        ATb_opt_order = ATA_matrix.reorder_led_values_to_rcm(ATb)
        led_values_opt_order = ATA_matrix.reorder_led_values_to_rcm(
            led_values_normalized
        )
        print(f"DIA matrix shape: {ATA_matrix.dia_data_cpu.shape}")
    else:
        print("Using spatial order for dense matrix")
        ATb_opt_order = ATb
        led_values_opt_order = led_values_normalized
        print(f"Dense ATA matrix shape: {ATA_matrix.shape}")

    # Step 3: GPU transfer
    print("\n=== STEP 3: GPU Transfer ===")
    ATb_gpu = cp.asarray(ATb_opt_order)
    led_values_gpu = cp.asarray(led_values_opt_order)
    print(f"GPU ATb shape: {ATb_gpu.shape}")
    print(f"GPU LED values shape: {led_values_gpu.shape}")

    # Step 4: Optimization loop
    print("\n=== STEP 4: Optimization Loop ===")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration+1} ---")

        # Compute A^T A @ x
        if isinstance(ATA_matrix, DiagonalATAMatrix):
            print("Computing A^T A @ x using DIA matrix")
            ATA_x = ATA_matrix.multiply_3d(led_values_gpu)
            if not isinstance(ATA_x, cp.ndarray):
                ATA_x = cp.asarray(ATA_x)
        else:
            print("Computing A^T A @ x using dense matrix")
            ATA_x = cp.einsum("ijc,cj->ci", cp.asarray(ATA_matrix), led_values_gpu)

        print(f"A^T A @ x shape: {ATA_x.shape}")
        print(
            f"A^T A @ x range: [{float(cp.min(ATA_x)):.6f}, {float(cp.max(ATA_x)):.6f}]"
        )

        # Compute gradient
        gradient = ATA_x - ATb_gpu
        print(f"Gradient shape: {gradient.shape}")
        print(
            f"Gradient range: [{float(cp.min(gradient)):.6f}, {float(cp.max(gradient)):.6f}]"
        )

        # Compute step size
        g_dot_g = cp.sum(gradient * gradient)
        print(f"g^T @ g: {float(g_dot_g):.6f}")

        if isinstance(ATA_matrix, DiagonalATAMatrix):
            print("Computing g^T @ A^T A @ g using DIA matrix")
            g_dot_ATA_g_per_channel = ATA_matrix.g_ata_g_3d(gradient)
            if not isinstance(g_dot_ATA_g_per_channel, cp.ndarray):
                g_dot_ATA_g_per_channel = cp.asarray(g_dot_ATA_g_per_channel)
            g_dot_ATA_g = cp.sum(g_dot_ATA_g_per_channel)
            print(
                f"g^T @ A^T A @ g per channel: {[float(x) for x in g_dot_ATA_g_per_channel]}"
            )
        else:
            print("Computing g^T @ A^T A @ g using dense matrix")
            g_dot_ATA_g = cp.einsum(
                "ci,ijc,cj->", gradient, cp.asarray(ATA_matrix), gradient
            )

        print(f"g^T @ A^T A @ g: {float(g_dot_ATA_g):.6f}")

        if g_dot_ATA_g > 0:
            step_size = float(step_size_scaling * g_dot_g / g_dot_ATA_g)
        else:
            step_size = 0.01
            print("WARNING: g^T @ A^T A @ g <= 0, using fallback step size")

        print(f"Step size: {step_size:.6f}")

        # Update LED values
        led_values_new = cp.clip(led_values_gpu - step_size * gradient, 0, 1)
        print(
            f"LED values range after update: [{float(cp.min(led_values_new)):.6f}, {float(cp.max(led_values_new)):.6f}]"
        )

        # Check convergence
        delta = cp.linalg.norm(led_values_new - led_values_gpu)
        print(f"Delta (convergence check): {float(delta):.6f}")
        print(f"Convergence threshold: {convergence_threshold:.6f}")

        if delta < convergence_threshold:
            print(f"*** CONVERGED after {iteration+1} iterations ***")
            led_values_gpu = led_values_new
            break

        led_values_gpu = led_values_new

        # Check for nan/inf values
        if cp.any(cp.isnan(led_values_gpu)) or cp.any(cp.isinf(led_values_gpu)):
            print("ERROR: NaN or Inf values detected in LED values!")
            break

    else:
        print(f"*** DID NOT CONVERGE after {max_iterations} iterations ***")
        print(f"Final delta: {float(delta):.6f}")

    # Step 5: Error metrics computation
    print("\n=== STEP 5: Error Metrics Computation ===")

    # Convert back to spatial order
    if isinstance(ATA_matrix, DiagonalATAMatrix):
        led_values_spatial = ATA_matrix.reorder_led_values_from_rcm(
            cp.asnumpy(led_values_gpu)
        )
        print("Converted LED values from RCM back to spatial order")
    else:
        led_values_spatial = cp.asnumpy(led_values_gpu)
        print("LED values already in spatial order")

    print(f"Final LED values spatial shape: {led_values_spatial.shape}")
    print(
        f"Final LED values range: [{led_values_spatial.min():.6f}, {led_values_spatial.max():.6f}]"
    )

    # Compute forward pass for error metrics
    try:
        print("Computing forward pass for error metrics...")

        if isinstance(AT_matrix, SingleBlockMixedSparseTensor):
            print("Using mixed tensor forward pass")
            led_values_gpu = cp.asarray(
                led_values_spatial.T
            )  # Convert to (led_count, 3)
            print(f"LED values for forward pass shape: {led_values_gpu.shape}")

            rendered_gpu = AT_matrix.forward_pass_3d(led_values_gpu)
            rendered_planar = cp.asnumpy(rendered_gpu)
            print(f"Rendered frame shape: {rendered_planar.shape}")
            print(
                f"Rendered frame range: [{rendered_planar.min():.6f}, {rendered_planar.max():.6f}]"
            )

        elif isinstance(AT_matrix, LEDDiffusionCSCMatrix):
            print("Using CSC forward pass")
            csc_A = AT_matrix.to_csc_matrix()
            height, width = target_planar.shape[1], target_planar.shape[2]
            rendered_planar = np.zeros((3, height, width), dtype=np.float32)

            for channel in range(3):
                led_channel = led_values_spatial[channel]
                channel_cols = np.arange(channel, csc_A.shape[1], 3)
                A_channel = csc_A[:, channel_cols]
                rendered_channel = A_channel @ led_channel
                rendered_planar[channel] = rendered_channel.reshape(height, width)
                print(
                    f"Channel {channel} rendered range: [{rendered_channel.min():.6f}, {rendered_channel.max():.6f}]"
                )

        # Compute error metrics
        diff = rendered_planar - target_planar
        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))

        if mse > 0:
            psnr = float(20 * np.log10(1.0 / np.sqrt(mse)))
        else:
            psnr = float("inf")

        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"PSNR: {psnr:.2f}")
        print(f"Target mean: {float(np.mean(target_planar)):.6f}")
        print(f"Rendered mean: {float(np.mean(rendered_planar)):.6f}")

    except Exception as e:
        print(f"ERROR in forward pass computation: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run detailed optimization trace."""

    # Load test data
    patterns_path = "diffusion_patterns/baseline_realistic.npz"
    image_path = "flower_test.png"

    print(f"Loading patterns from: {patterns_path}")
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Load mixed tensor
    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
    print(f"Mixed tensor: {mixed_tensor.batch_size} LEDs")

    # Load DIA matrix
    dia_matrix = DiagonalATAMatrix(led_count=mixed_tensor.batch_size)
    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_full = diffusion_csc.to_csc_matrix()
    led_positions = patterns_data["led_positions"]

    # Suppress DIA build output
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    dia_matrix.build_from_diffusion_matrix(csc_full, led_positions)
    sys.stdout = old_stdout

    # Load test image
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((800, 480))
    target_image = np.array(image, dtype=np.uint8)

    print(f"\nTest image shape: {target_image.shape}")
    print(f"Test image range: [{target_image.min()}, {target_image.max()}]")

    # Test Mixed + DIA combination
    print("\n" + "=" * 80)
    print("TESTING: Mixed A^T + DIA A^T A")
    print("=" * 80)

    detailed_optimization_trace(
        target_frame=target_image,
        AT_matrix=mixed_tensor,
        ATA_matrix=dia_matrix,
        max_iterations=10,
        convergence_threshold=1e-6,
        step_size_scaling=0.8,
    )


if __name__ == "__main__":
    main()
