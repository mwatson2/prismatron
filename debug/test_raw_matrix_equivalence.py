#!/usr/bin/env python3

import sys

import numpy as np

sys.path.append("src")
import cupy as cp

from utils.dense_ata_matrix import DenseATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

print("=== TESTING RAW MATRIX EQUIVALENCE ===")
print("Creating DIA and Dense matrices from same raw CSC data (no scaling)")

# Load synthetic data
file = np.load("diffusion_patterns/synthetic_2624_uint8.npz", allow_pickle=True)
mixed_tensor_data = file["mixed_tensor"].item()
mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_data)

print("Loaded mixed tensor successfully")

# Build raw CSC matrix (no scaling applied) using the same function as rebuild script
print("Building CSC matrix from mixed tensor...")
sys.path.append(".")  # Add current directory to path

# Import the function from rebuild script
import importlib.util

spec = importlib.util.spec_from_file_location("rebuild_module", "rebuild_ata_from_capture.py")
rebuild_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebuild_module)

csc_matrix = rebuild_module.build_csc_from_mixed_tensor(mixed_tensor)
print(f"CSC matrix: {csc_matrix.shape}, {csc_matrix.nnz} non-zeros")

led_count = csc_matrix.shape[1] // 3
print(f"Detected {led_count} LEDs")

# Create DIA matrix directly (same as original synthetic generation)
print("\\nBuilding DIA matrix (raw, no scaling)...")
dia_matrix = DiagonalATAMatrix(led_count)
dia_matrix.build_from_diffusion_matrix(csc_matrix)
print(f"DIA matrix: {dia_matrix.led_count} LEDs, k={dia_matrix.k} diagonals")

# Create Dense matrix directly (raw, no scaling)
print("\\nBuilding Dense matrix (raw, no scaling)...")
dense_matrix = DenseATAMatrix(led_count)
dense_matrix.build_from_diffusion_matrix(csc_matrix)
print(f"Dense matrix: {dense_matrix.led_count} LEDs")

# Test matrices with identical input
print("\\n=== TESTING MATRIX EQUIVALENCE ===")
np.random.seed(42)
test_input = np.random.rand(3, led_count).astype(np.float32)
test_input_gpu = cp.asarray(test_input)

dia_result = dia_matrix.multiply_3d(test_input_gpu)
dense_result = dense_matrix.multiply_vector(test_input_gpu)

dia_cpu = cp.asnumpy(dia_result)
dense_cpu = cp.asnumpy(dense_result)

print(f"DIA result range: [{dia_cpu.min():.6f}, {dia_cpu.max():.6f}]")
print(f"Dense result range: [{dense_cpu.min():.6f}, {dense_cpu.max():.6f}]")

if np.allclose(dia_cpu, dense_cpu, rtol=1e-5, atol=1e-5):
    print("✅ SUCCESS: Matrices are equivalent when built from same raw CSC data")
    max_diff = np.abs(dia_cpu - dense_cpu).max()
    print(f"Max difference: {max_diff:.2e}")
else:
    print("❌ FAILURE: Matrices differ even when built from same raw data")
    max_diff = np.abs(dia_cpu - dense_cpu).max()
    rel_diff = max_diff / (np.abs(dia_cpu).max() + 1e-10)
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Max relative difference: {rel_diff:.2e}")
