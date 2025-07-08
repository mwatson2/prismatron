#!/usr/bin/env python3
"""
Check what data types are actually being used in batch optimization.
"""

import sys

import cupy as cp
import numpy as np

sys.path.insert(0, "src")

from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def main():
    print("Checking data types used in batch optimization...")

    # Load pattern data
    pattern_file = "diffusion_patterns/synthetic_2624_fp16_64x64.npz"
    data = np.load(pattern_file, allow_pickle=True)

    # Load mixed tensor
    mixed_tensor_dict = data["mixed_tensor"].item()
    at_matrix = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    print(f"\nMixed Tensor (AT matrix):")
    print(f"  dtype: {at_matrix.dtype}")
    print(f"  output_dtype: {at_matrix.output_dtype}")
    print(f"  sparse_values dtype: {at_matrix.sparse_values.dtype}")

    # Load DIA matrix
    dia_dict = data["dia_matrix"].item()
    ata_matrix = DiagonalATAMatrix.from_dict(dia_dict)

    print(f"\nDIA Matrix (ATA matrix):")
    print(f"  DIA data dtype: {ata_matrix.dia_data.dtype}")

    # Check ATA inverse
    ata_inverse = data["ata_inverse"]
    print(f"\nATA Inverse:")
    print(f"  dtype: {ata_inverse.dtype}")
    print(f"  shape: {ata_inverse.shape}")
    print(f"  memory: {ata_inverse.nbytes / 1024 / 1024:.1f} MB")

    # Test what happens during ATb calculation
    print(f"\nTesting ATb calculation data types...")
    test_frame = np.random.randint(0, 255, (3, 480, 800), dtype=np.uint8)

    if at_matrix.dtype == cp.uint8:
        target_gpu = cp.asarray(test_frame)
        print(f"  Using uint8 path: target_gpu dtype = {target_gpu.dtype}")
    else:
        target_float32 = test_frame.astype(np.float32) / 255.0
        target_gpu = cp.asarray(target_float32)
        print(f"  Using float32 path: target_gpu dtype = {target_gpu.dtype}")

    # Test the actual multiplication
    result = at_matrix.transpose_dot_product_3d(target_gpu)
    print(f"  ATb result dtype: {result.dtype}")
    print(f"  ATb result shape: {result.shape}")

    # Test ATA inverse operations
    print(f"\nTesting ATA inverse operations...")
    ATb_test = np.random.random((3, 2624)).astype(np.float32)

    # Test what happens when we use fp16 ATA inverse
    ata_inverse_gpu = cp.asarray(ata_inverse)  # This will be fp16
    ATb_test_gpu = cp.asarray(ATb_test)  # This will be float32

    print(f"  ATA inverse GPU dtype: {ata_inverse_gpu.dtype}")
    print(f"  ATb test GPU dtype: {ATb_test_gpu.dtype}")

    # Test einsum operation
    try:
        result_einsum = cp.einsum("ijk,bik->bij", ata_inverse_gpu, ATb_test_gpu.reshape(1, 3, 2624))
        print(f"  Einsum result dtype: {result_einsum.dtype}")
        print(f"  Mixed precision computation successful!")
    except Exception as e:
        print(f"  Einsum failed: {e}")

    # Check memory usage
    print(f"\nMemory comparison:")
    ata_fp32_size = 3 * 2624 * 2624 * 4 / 1024 / 1024  # float32
    ata_fp16_size = 3 * 2624 * 2624 * 2 / 1024 / 1024  # float16
    print(f"  ATA inverse fp32 would be: {ata_fp32_size:.1f} MB")
    print(f"  ATA inverse fp16 actual: {ata_fp16_size:.1f} MB")
    print(f"  Memory savings: {(ata_fp32_size - ata_fp16_size) / ata_fp32_size * 100:.1f}%")


if __name__ == "__main__":
    main()
