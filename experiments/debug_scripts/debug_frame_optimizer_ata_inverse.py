#!/usr/bin/env python3
"""
Debug script to compare dense vs DIA ATA inverse within frame optimizer.
Uses exact same setup as MSE convergence analysis with comparison logging.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_flower_image() -> np.ndarray:
    """Load and prepare the flower image for testing (same as MSE analysis)."""
    flower_path = Path("images/source/flower")

    # Try common image extensions
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        if (flower_path.parent / f"{flower_path.name}{ext}").exists():
            flower_path = flower_path.parent / f"{flower_path.name}{ext}"
            break

    if not flower_path.exists():
        # Fallback: create a synthetic flower-like test pattern (same as MSE analysis)
        print("Warning: Flower image not found, creating synthetic test pattern")
        frame = np.zeros((480, 800, 3), dtype=np.uint8)

        # Create flower-like pattern
        center_y, center_x = 240, 400
        for y in range(480):
            for x in range(800):
                dx, dy = x - center_x, y - center_y
                dist = np.sqrt(dx * dx + dy * dy)
                angle = np.arctan2(dy, dx)

                # Petals pattern
                petal_intensity = (np.sin(6 * angle) + 1) / 2
                if dist < 150:
                    frame[y, x, 0] = int(255 * petal_intensity * (1 - dist / 150))  # Red petals
                    frame[y, x, 1] = int(128 * (1 - dist / 200))  # Green center
                    frame[y, x, 2] = int(64 * petal_intensity)  # Blue accent

        return frame.transpose(2, 0, 1)  # Convert to (3, H, W)

    # Load real flower image
    image = Image.open(flower_path)
    image = image.convert("RGB")
    image = image.resize((800, 480), Image.Resampling.LANCZOS)

    # Convert to numpy array and transpose to planar format
    frame = np.array(image).transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    return frame.astype(np.uint8)


def debug_ata_inverse_in_frame_optimizer():
    """Debug ATA inverse comparison within frame optimizer."""

    print("Debug: Dense vs DIA ATA Inverse in Frame Optimizer")
    print("=" * 55)

    # Load flower image (exact same as MSE analysis)
    target_frame = load_flower_image()
    print(f"Target frame shape: {target_frame.shape}")

    # Load patterns with both dense and DIA ATA inverse (factor 10.0 for perfect comparison)
    pattern_file = "diffusion_patterns/synthetic_2624_uint8_dia_10.0.npz"

    if not Path(pattern_file).exists():
        print(f"Pattern file not found: {pattern_file}")
        return

    print(f"Loading patterns from: {pattern_file}")
    data = np.load(pattern_file, allow_pickle=True)

    # Load components (same as MSE analysis)
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())
    dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())
    ata_inverse_dense = data["ata_inverse"]
    ata_inverse_dia = DiagonalATAMatrix.from_dict(data["ata_inverse_dia"].item())

    print(f"Mixed tensor: {mixed_tensor.dtype}, batch_size: {mixed_tensor.batch_size}")
    print(f"Dense ATA inverse: {ata_inverse_dense.shape}, dtype: {ata_inverse_dense.dtype}")
    print(f"DIA ATA inverse: k={ata_inverse_dia.k}, bandwidth={ata_inverse_dia.bandwidth}")

    # Test 1: Use DIA as primary, dense as comparison
    print("\n" + "=" * 60)
    print("TEST 1: DIA Primary, Dense Comparison")
    print("=" * 60)

    result_dia = optimize_frame_led_values(
        target_frame=target_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dia_matrix,
        ata_inverse=ata_inverse_dia,  # DIA as primary
        compare_ata_inverse=ata_inverse_dense,  # Dense as comparison
        max_iterations=1,  # Just test initialization
        compute_error_metrics=True,
        track_mse_per_iteration=True,
    )

    print(f"DIA primary result: Initial MSE = {result_dia.mse_per_iteration[0]:.6f}")

    # Test 2: Use dense as primary, DIA as comparison
    print("\n" + "=" * 60)
    print("TEST 2: Dense Primary, DIA Comparison")
    print("=" * 60)

    result_dense = optimize_frame_led_values(
        target_frame=target_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dia_matrix,
        ata_inverse=ata_inverse_dense,  # Dense as primary
        compare_ata_inverse=ata_inverse_dia,  # DIA as comparison
        max_iterations=1,  # Just test initialization
        compute_error_metrics=True,
        track_mse_per_iteration=True,
    )

    print(f"Dense primary result: Initial MSE = {result_dense.mse_per_iteration[0]:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"DIA as primary:  Initial MSE = {result_dia.mse_per_iteration[0]:.6f}")
    print(f"Dense as primary: Initial MSE = {result_dense.mse_per_iteration[0]:.6f}")
    print(f"MSE ratio (DIA/Dense): {result_dia.mse_per_iteration[0]/result_dense.mse_per_iteration[0]:.2f}x")

    if abs(result_dia.mse_per_iteration[0] - result_dense.mse_per_iteration[0]) < 1e-5:
        print("✅ MSE values are essentially identical - ATA inverse calculation is working correctly")
    else:
        print("❌ MSE values differ significantly - there's an issue in the ATA inverse calculation")


if __name__ == "__main__":
    debug_ata_inverse_in_frame_optimizer()
