#!/usr/bin/env python3
"""
Debug the optimization differences between DIA and Dense matrices.
"""

import sys
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image

sys.path.append("src")
from utils.dense_ata_matrix import DenseATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def debug_matrix_initialization():
    """Debug the matrix initialization and A^T@b computation."""
    print("=== DEBUGGING MATRIX INITIALIZATION ===")

    # Load data
    original_data = np.load("diffusion_patterns/synthetic_2624_uint8.npz", allow_pickle=True)
    dense_data = np.load("diffusion_patterns/synthetic-dense-test.npz", allow_pickle=True)

    mixed_tensor_dict = original_data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    dia_matrix_dict = original_data["dia_matrix"].item()
    dia_matrix = DiagonalATAMatrix.from_dict(dia_matrix_dict)

    dense_matrix_dict = dense_data["dense_ata_matrix_rebuilt"].item()
    dense_matrix = DenseATAMatrix.from_dict(dense_matrix_dict)

    ata_inverse = original_data["ata_inverse"]

    print(f"Mixed tensor dtype: {mixed_tensor.dtype}")
    print(f"DIA matrix LED count: {dia_matrix.led_count}")
    print(f"Dense matrix LED count: {dense_matrix.led_count}")
    print(f"ATA inverse shape: {ata_inverse.shape}")

    # Load a simple test image
    with Image.open("images/source/flower.jpg") as img:
        img_resized = img.resize((800, 480), Image.LANCZOS).convert("RGB")
        img_array = np.array(img_resized, dtype=np.uint8).transpose(2, 0, 1)
        target_frame = cp.asarray(img_array)

    print(f"Target frame shape: {target_frame.shape}, dtype: {target_frame.dtype}")

    # Compute A^T @ b
    print("\n=== COMPUTING A^T @ b ===")
    ATb_gpu = mixed_tensor.transpose_dot_product_3d(target_frame, planar_output=True)
    ATb_cpu = cp.asnumpy(ATb_gpu)

    print(f"A^T @ b shape: {ATb_cpu.shape}")
    print(f"A^T @ b range: [{ATb_cpu.min():.6f}, {ATb_cpu.max():.6f}]")
    print(f"A^T @ b mean: {ATb_cpu.mean():.6f}")

    # Test ATA inverse initialization
    print("\n=== TESTING ATA INVERSE INITIALIZATION ===")

    # Dense ATA inverse (original)
    ata_inverse_gpu = cp.asarray(ata_inverse)
    led_values_dense_inv = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_gpu)
    led_values_dense_inv_cpu = cp.asnumpy(led_values_dense_inv)

    print("Dense ATA inverse initialization:")
    print(f"  Raw values range: [{led_values_dense_inv_cpu.min():.6f}, {led_values_dense_inv_cpu.max():.6f}]")
    print(f"  Raw values mean: {led_values_dense_inv_cpu.mean():.6f}")

    # DIA matrix as ATA inverse (this should be different!)
    led_values_dia_inv = dia_matrix.multiply_3d(ATb_gpu)
    led_values_dia_inv_cpu = cp.asnumpy(led_values_dia_inv)

    print("DIA matrix used as ATA inverse:")
    print(f"  Raw values range: [{led_values_dia_inv_cpu.min():.6f}, {led_values_dia_inv_cpu.max():.6f}]")
    print(f"  Raw values mean: {led_values_dia_inv_cpu.mean():.6f}")

    # Dense matrix as ATA inverse (this should also be different!)
    led_values_dense_ata_inv = dense_matrix.multiply_vector(ATb_gpu)
    led_values_dense_ata_inv_cpu = cp.asnumpy(led_values_dense_ata_inv)

    print("Dense matrix used as ATA inverse:")
    print(f"  Raw values range: [{led_values_dense_ata_inv_cpu.min():.6f}, {led_values_dense_ata_inv_cpu.max():.6f}]")
    print(f"  Raw values mean: {led_values_dense_ata_inv_cpu.mean():.6f}")

    # Compare initializations
    print("\n=== INITIALIZATION COMPARISON ===")
    diff_dia = np.abs(led_values_dense_inv_cpu - led_values_dia_inv_cpu)
    diff_dense_ata = np.abs(led_values_dense_inv_cpu - led_values_dense_ata_inv_cpu)

    print(f"Dense inv vs DIA matrix: max_diff={diff_dia.max():.6f}, mean_diff={diff_dia.mean():.6f}")
    print(f"Dense inv vs Dense ATA: max_diff={diff_dense_ata.max():.6f}, mean_diff={diff_dense_ata.mean():.6f}")

    # The issue might be that we're using the ATA matrix instead of ATA inverse!
    print("\n=== ISSUE ANALYSIS ===")
    print("🔍 The problem might be:")
    print("1. Both DIA and Dense optimization are using different ATA inverse matrices")
    print("2. We should be using the SAME ATA inverse for both (the original numpy array)")
    print("3. The DIA/Dense matrices should only be used for gradient computation (A^T A @ x)")
    print("4. NOT for initialization (that should use the true ATA inverse)")


if __name__ == "__main__":
    debug_matrix_initialization()
