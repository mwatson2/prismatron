#!/usr/bin/env python3
"""
Quick test script to verify DIA ATA inverse support in frame optimizer.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent / "src"))

from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.frame_optimizer import optimize_frame_led_values
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def test_dia_ata_inverse():
    """Test frame optimizer with DIA ATA inverse."""

    # Load pattern file with DIA ATA inverse
    pattern_file = "diffusion_patterns/synthetic_2624_uint8_dia_10.0.npz"

    if not Path(pattern_file).exists():
        print(f"Pattern file not found: {pattern_file}")
        return

    print(f"Loading pattern file: {pattern_file}")
    data = np.load(pattern_file, allow_pickle=True)

    # Load mixed tensor
    mixed_tensor_dict = data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    # Load DIA matrix
    dia_dict = data["dia_matrix"].item()
    dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)

    # Load DIA ATA inverse
    ata_inverse_dia_dict = data["ata_inverse_dia"].item()
    ata_inverse_dia = DiagonalATAMatrix.from_dict(ata_inverse_dia_dict)

    print(f"Mixed tensor: {mixed_tensor.batch_size} LEDs, dtype={mixed_tensor.dtype}")
    print(f"DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k}")
    print(f"DIA ATA inverse: bandwidth={ata_inverse_dia.bandwidth}, k={ata_inverse_dia.k}")

    # Create test frame
    target_frame = np.zeros((3, 480, 800), dtype=np.uint8)
    target_frame[0, 100:150, :] = 255  # Red stripe
    target_frame[1, 200:250, 200:600] = 128  # Green rectangle

    print("Running optimization with DIA ATA inverse...")

    # Test with DIA ATA inverse
    result = optimize_frame_led_values(
        target_frame=target_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dia_matrix,
        ata_inverse=ata_inverse_dia,  # DIA format
        max_iterations=5,
        compute_error_metrics=True,
        debug=True,
        track_mse_per_iteration=True,
    )

    print("✅ Optimization completed!")
    print(f"  LED values shape: {result.led_values.shape}")
    print(f"  LED values range: [{result.led_values.min()}, {result.led_values.max()}]")
    print(f"  Iterations: {result.iterations}")
    if result.error_metrics:
        print(f"  MSE: {result.error_metrics.get('mse', 'N/A'):.6f}")
        print(f"  PSNR: {result.error_metrics.get('psnr', 'N/A'):.2f} dB")
    if result.mse_per_iteration is not None:
        print(f"  Initial MSE: {result.mse_per_iteration[0]:.6f}")
        print(f"  Final MSE: {result.mse_per_iteration[-1]:.6f}")
        reduction = (result.mse_per_iteration[0] - result.mse_per_iteration[-1]) / result.mse_per_iteration[0] * 100
        print(f"  MSE reduction: {reduction:.1f}%")

    # Compare with dense ATA inverse for verification
    print("\nComparing with dense ATA inverse...")
    ata_inverse_dense = data["ata_inverse"]

    result_dense = optimize_frame_led_values(
        target_frame=target_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dia_matrix,
        ata_inverse=ata_inverse_dense,  # Dense format
        max_iterations=5,
        compute_error_metrics=True,
        debug=False,
        track_mse_per_iteration=True,
    )

    # Compare results
    max_diff = np.max(np.abs(result.led_values.astype(np.float32) - result_dense.led_values.astype(np.float32)))
    mean_diff = np.mean(np.abs(result.led_values.astype(np.float32) - result_dense.led_values.astype(np.float32)))

    print(f"  Dense result MSE: {result_dense.error_metrics.get('mse', 'N/A'):.6f}")
    print(f"  Difference: max={max_diff:.2f}, mean={mean_diff:.2f}")

    if max_diff < 10:  # Allow some approximation error
        print(f"  ✅ DIA and dense results are similar (max diff: {max_diff:.2f})")
    else:
        print(f"  ⚠️  Large difference between DIA and dense results (max diff: {max_diff:.2f})")

    return result, result_dense


if __name__ == "__main__":
    test_dia_ata_inverse()
