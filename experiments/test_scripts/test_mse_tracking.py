#!/usr/bin/env python3
"""
Simple test to verify MSE tracking functionality works.
"""

import numpy as np

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def test_mse_tracking():
    """Test that MSE tracking works with dense ATA inverse."""

    print("Testing MSE tracking functionality...")

    # Load pattern data
    data = np.load("diffusion_patterns/synthetic_2624_uint8.npz", allow_pickle=True)

    # Load components
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())
    dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())
    ata_inverse = data["ata_inverse"]

    print("Loaded mixed tensor successfully")
    print(f"ATA inverse shape: {ata_inverse.shape}")

    # Create test frame
    test_frame = np.random.randint(0, 256, (480, 800, 3), dtype=np.uint8)

    # Test MSE tracking
    print("\nTesting with MSE tracking enabled...")
    result = optimize_frame_led_values(
        target_frame=test_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dia_matrix,
        ata_inverse=ata_inverse,
        max_iterations=5,
        track_mse_per_iteration=True,
        debug=False,
    )

    print("Optimization completed:")
    print(f"  Iterations: {result.iterations}")
    print(f"  LED values shape: {result.led_values.shape}")

    if result.mse_per_iteration is not None:
        print(f"  MSE per iteration: {result.mse_per_iteration}")
        print(f"  MSE values count: {len(result.mse_per_iteration)}")
        print("✅ MSE tracking works!")
    else:
        print("❌ MSE tracking failed!")

    # Test without MSE tracking
    print("\nTesting without MSE tracking...")
    result_no_mse = optimize_frame_led_values(
        target_frame=test_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dia_matrix,
        ata_inverse=ata_inverse,
        max_iterations=5,
        track_mse_per_iteration=False,
        debug=False,
    )

    if result_no_mse.mse_per_iteration is None:
        print("✅ MSE tracking correctly disabled!")
    else:
        print("❌ MSE tracking should be None when disabled!")


if __name__ == "__main__":
    test_mse_tracking()
