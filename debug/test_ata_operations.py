#!/usr/bin/env python3
"""
Test A^T A @ x operations between DIA and Dense matrices.
"""

import sys

import cupy as cp
import numpy as np

sys.path.append("src")
from utils.dense_ata_matrix import DenseATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix


def test_ata_operations():
    """Test if DIA and Dense matrices produce identical A^T A @ x results."""
    print("=== TESTING A^T A @ x OPERATIONS ===")

    # Load both matrices
    original_data = np.load("diffusion_patterns/synthetic_2624_uint8.npz", allow_pickle=True)
    dense_data = np.load("diffusion_patterns/synthetic-dense-test.npz", allow_pickle=True)

    dia_matrix_dict = original_data["dia_matrix"].item()
    dia_matrix = DiagonalATAMatrix.from_dict(dia_matrix_dict)

    dense_matrix_dict = dense_data["dense_ata_matrix_rebuilt"].item()
    dense_matrix = DenseATAMatrix.from_dict(dense_matrix_dict)

    print(f"DIA matrix: {dia_matrix.led_count} LEDs, k={dia_matrix.k} diagonals")
    print(f"Dense matrix: {dense_matrix.led_count} LEDs, {dense_matrix.memory_mb:.1f}MB")

    # Create test input vector
    np.random.seed(42)
    test_input = np.random.rand(3, dia_matrix.led_count).astype(np.float32)
    test_input_gpu = cp.asarray(test_input)

    print(f"Test input shape: {test_input.shape}")
    print(f"Test input range: [{test_input.min():.6f}, {test_input.max():.6f}]")

    # Test A^T A @ x operations
    print("\n=== A^T A @ x OPERATIONS ===")

    dia_result = dia_matrix.multiply_3d(test_input_gpu)
    dense_result = dense_matrix.multiply_vector(test_input_gpu)

    dia_cpu = cp.asnumpy(dia_result)
    dense_cpu = cp.asnumpy(dense_result)

    print(f"DIA result range: [{dia_cpu.min():.6f}, {dia_cpu.max():.6f}]")
    print(f"Dense result range: [{dense_cpu.min():.6f}, {dense_cpu.max():.6f}]")

    # Compare results
    max_diff = np.max(np.abs(dia_cpu - dense_cpu))
    mean_diff = np.mean(np.abs(dia_cpu - dense_cpu))
    rms_diff = np.sqrt(np.mean((dia_cpu - dense_cpu) ** 2))

    dia_magnitude = np.sqrt(np.mean(dia_cpu**2))
    relative_rms = rms_diff / dia_magnitude if dia_magnitude > 0 else float("inf")

    print(f"\nComparison results:")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"RMS difference: {rms_diff:.6f}")
    print(f"Relative RMS error: {relative_rms:.6f} ({relative_rms*100:.2f}%)")

    # Check tolerance
    tolerance = 1e-4  # Reasonable tolerance for floating point operations
    if max_diff <= tolerance:
        print(f"✅ SUCCESS: A^T A @ x operations match within tolerance ({tolerance})")
        return True
    else:
        print(f"❌ FAILURE: A^T A @ x operations differ by more than tolerance ({tolerance})")

        # Detailed analysis
        print(f"\nDetailed analysis:")
        print(f"DIA magnitude: {dia_magnitude:.6f}")
        dense_magnitude = np.sqrt(np.mean(dense_cpu**2))
        print(f"Dense magnitude: {dense_magnitude:.6f}")
        print(f"Magnitude ratio: {dense_magnitude/dia_magnitude:.6f}")

        # Show first few values
        print(f"\nFirst 5 values comparison:")
        for i in range(min(5, len(dia_cpu.flat))):
            print(
                f"  [{i}] DIA: {dia_cpu.flat[i]:.6f}, Dense: {dense_cpu.flat[i]:.6f}, Diff: {abs(dia_cpu.flat[i] - dense_cpu.flat[i]):.6f}"
            )

        return False


if __name__ == "__main__":
    success = test_ata_operations()
    if success:
        print("\n🎉 A^T A operations are equivalent!")
    else:
        print("\n❌ A^T A operations are NOT equivalent!")
        print("This explains why optimization produces different results.")
