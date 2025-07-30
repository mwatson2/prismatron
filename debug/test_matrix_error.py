#!/usr/bin/env python3

import sys

import numpy as np

sys.path.append("src")
import cupy as cp

from utils.dense_ata_matrix import DenseATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix

print("=== INVESTIGATING 9.4% ERROR ===")
file = np.load("diffusion_patterns/capture-0728-01-clean-comparison.npz", allow_pickle=True)

dia_data = file["dia_matrix"].item()
dia_matrix = DiagonalATAMatrix.from_dict(dia_data)

dense_data = file["dense_ata_matrix"].item()
dense_matrix = DenseATAMatrix.from_dict(dense_data)

np.random.seed(42)
# Both matrices expect (3, led_count) shaped input
test_input = np.random.rand(3, dia_matrix.led_count).astype(np.float32)
test_vec_gpu = cp.asarray(test_input)

dia_result = dia_matrix.multiply_3d(test_vec_gpu)
dense_result = dense_matrix.multiply_vector(test_vec_gpu)

dia_cpu = cp.asnumpy(dia_result)
dense_cpu = cp.asnumpy(dense_result)

print(f"DIA result range: [{dia_cpu.min():.6f}, {dia_cpu.max():.6f}]")
print(f"Dense result range: [{dense_cpu.min():.6f}, {dense_cpu.max():.6f}]")

if not np.array_equal(dia_cpu, dense_cpu):
    diff = np.abs(dia_cpu - dense_cpu)
    rel_diff = diff / (np.abs(dia_cpu) + 1e-10)
    print(f"Max absolute difference: {diff.max():.2e}")
    print(f"Max relative difference: {rel_diff.max():.2e}")

    print(f"DIA mean: {dia_cpu.mean():.2e}")
    print(f"Dense mean: {dense_cpu.mean():.2e}")

    dense_mean = dense_cpu.mean()
    if dense_mean != 0:
        ratio = dia_cpu.mean() / dense_mean
        print(f"Mean value ratio (DIA/Dense): {ratio:.2f}")

    # Show first few values to understand the pattern
    print("First 5 values:")
    for i in range(5):
        print(f"  DIA[{i}]: {float(dia_cpu.flat[i]):.6f}, Dense[{i}]: {float(dense_cpu.flat[i]):.6f}")

    # Let's also check if both matrices are using the same global scaling
    print()
    print("=== CHECKING MATRIX CONSTRUCTION ===")
    print("DIA matrix attributes:")
    print(f"  led_count: {dia_matrix.led_count}")
    print(f"  channels: {dia_matrix.channels}")

    print("Dense matrix attributes:")
    print(f"  led_count: {dense_matrix.led_count}")
    print(f"  channels: {dense_matrix.channels}")

    # Check if global scaling is being applied consistently
    print()
    print("Global scaling factor from file:", file["global_scaling_factor"])
else:
    print("✅ Results are EXACTLY equal")
