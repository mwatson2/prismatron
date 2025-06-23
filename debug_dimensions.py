#!/usr/bin/env python3
"""Debug dimension mismatch in CSC operation."""

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Load patterns
patterns_path = "diffusion_patterns/synthetic_1000.npz"
data = np.load(patterns_path)

matrix_data = data["matrix_data"]
matrix_indices = data["matrix_indices"]
matrix_indptr = data["matrix_indptr"]
matrix_shape = tuple(data["matrix_shape"])

matrix = sp.csc_matrix(
    (matrix_data, matrix_indices, matrix_indptr), shape=matrix_shape, dtype=np.float32
)

print(f"Matrix shape: {matrix.shape}")
print(f"Matrix transpose shape: {matrix.T.shape}")

# Create target vector
frame_height, frame_width = 480, 800
pixels = frame_height * frame_width

print(f"Pixels: {pixels}")
print(f"Expected target shape: {pixels}")

# Check if matrix expects different target format
print(f"Matrix expects input of size: {matrix.shape[0]}")
print(f"Matrix produces output of size: {matrix.shape[1]}")

# For A^T @ b, we need:
# A^T shape: (3000, 384000)
# b shape: (384000,)
# result: (3000,)

target_flat = np.random.rand(pixels * 3).astype(np.float32)
print(f"Target flat shape: {target_flat.shape}")

print(f"A^T shape should be: {matrix.T.shape}")
print(f"b shape should be: {target_flat.shape}")
print(f"Compatible? {matrix.T.shape[1] == target_flat.shape[0]}")
