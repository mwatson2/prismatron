#!/usr/bin/env python3
"""
Debug script to verify DIA ATA operations match dense operations.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent / "src"))

import cupy as cp

from utils.diagonal_ata_matrix import DiagonalATAMatrix


def test_dia_vs_dense_operations():
    """Test that DIA operations match dense operations exactly."""

    # Load a pattern file
    pattern_file = "diffusion_patterns/synthetic_2624_uint8_dia_10.0.npz"

    if not Path(pattern_file).exists():
        print(f"Pattern file not found: {pattern_file}")
        return

    print(f"Loading pattern file: {pattern_file}")
    data = np.load(pattern_file, allow_pickle=True)

    # Load ATA inverse (dense and DIA)
    ata_inverse_dense = data["ata_inverse"]  # (3, 2624, 2624)
    ata_inverse_dia_dict = data["ata_inverse_dia"].item()
    ata_inverse_dia = DiagonalATAMatrix.from_dict(ata_inverse_dia_dict)

    print(f"Dense ATA inverse: shape={ata_inverse_dense.shape}, dtype={ata_inverse_dense.dtype}")
    print(f"DIA ATA inverse: bandwidth={ata_inverse_dia.bandwidth}, k={ata_inverse_dia.k}")

    # Create a test vector (simulating A^T @ b)
    led_count = ata_inverse_dense.shape[1]
    test_vector = np.random.rand(3, led_count).astype(np.float32) * 0.1
    test_vector_gpu = cp.asarray(test_vector)

    print(f"Test vector shape: {test_vector.shape}")

    # Test dense operation: einsum("ijk,ik->ij", ata_inverse, test_vector)
    ata_inverse_gpu = cp.asarray(ata_inverse_dense)
    result_dense = cp.einsum("ijk,ik->ij", ata_inverse_gpu, test_vector_gpu)
    result_dense_cpu = cp.asnumpy(result_dense)

    # Test DIA operation: ata_inverse.multiply_3d(test_vector)
    result_dia = ata_inverse_dia.multiply_3d(test_vector_gpu)
    result_dia_cpu = cp.asnumpy(result_dia)

    print(f"Dense result shape: {result_dense_cpu.shape}")
    print(f"DIA result shape: {result_dia_cpu.shape}")

    # Compare results
    max_diff = np.max(np.abs(result_dense_cpu - result_dia_cpu))
    mean_diff = np.mean(np.abs(result_dense_cpu - result_dia_cpu))
    rms_diff = np.sqrt(np.mean((result_dense_cpu - result_dia_cpu) ** 2))

    print(f"\nDifferences:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  RMS difference: {rms_diff:.6f}")

    # Relative differences
    dense_rms = np.sqrt(np.mean(result_dense_cpu**2))
    dia_rms = np.sqrt(np.mean(result_dia_cpu**2))

    print(f"\nMagnitudes:")
    print(f"  Dense RMS: {dense_rms:.6f}")
    print(f"  DIA RMS: {dia_rms:.6f}")
    print(f"  Relative error: {rms_diff/dense_rms*100:.2f}%")

    # Test if DIA matrix reconstructs the original dense matrix approximately
    print(f"\n=== Testing DIA Matrix Reconstruction ===")

    # Convert DIA back to dense for comparison
    dia_data_cpu = cp.asnumpy(ata_inverse_dia.dia_data_gpu)  # (3, k, led_count)
    dia_offsets = ata_inverse_dia.dia_offsets

    print(f"DIA data shape: {dia_data_cpu.shape}")
    print(f"DIA offsets: {len(dia_offsets)} offsets, range [{np.min(dia_offsets)}, {np.max(dia_offsets)}]")

    # Reconstruct dense matrix from DIA format
    reconstructed_dense = np.zeros_like(ata_inverse_dense)

    for channel in range(3):
        for k_idx, offset in enumerate(dia_offsets):
            diagonal_data = dia_data_cpu[channel, k_idx, :]

            if offset >= 0:
                # Upper diagonal
                for i in range(led_count - offset):
                    reconstructed_dense[channel, i, i + offset] = diagonal_data[i]
            else:
                # Lower diagonal
                for i in range(led_count + offset):
                    reconstructed_dense[channel, i - offset, i] = diagonal_data[i - offset]

    # Compare original dense vs reconstructed
    reconstruction_max_diff = np.max(np.abs(ata_inverse_dense - reconstructed_dense))
    reconstruction_rms_diff = np.sqrt(np.mean((ata_inverse_dense - reconstructed_dense) ** 2))
    original_rms = np.sqrt(np.mean(ata_inverse_dense**2))

    print(f"Reconstruction vs original dense:")
    print(f"  Max difference: {reconstruction_max_diff:.6f}")
    print(f"  RMS difference: {reconstruction_rms_diff:.6f}")
    print(f"  Relative error: {reconstruction_rms_diff/original_rms*100:.2f}%")

    # Check some specific elements
    print(f"\n=== Sample Value Comparison ===")
    for i in range(min(5, led_count)):
        for j in range(min(5, led_count)):
            dense_val = ata_inverse_dense[0, i, j]
            reconstructed_val = reconstructed_dense[0, i, j]
            print(
                f"  [{i},{j}]: dense={dense_val:.6f}, reconstructed={reconstructed_val:.6f}, diff={abs(dense_val-reconstructed_val):.6f}"
            )

    return result_dense_cpu, result_dia_cpu, reconstruction_rms_diff / original_rms


if __name__ == "__main__":
    test_dia_vs_dense_operations()
