"""
CUDA kernels for Prismatron LED optimization.

This module contains optimized CUDA kernels for various LED optimization operations.
"""

from .compute_optimized_3d import (
    cuda_transpose_dot_product_3d_compute_optimized,
    get_compute_optimized_3d_kernel,
)
from .compute_optimized_3d_int8 import (
    cuda_transpose_dot_product_3d_compute_optimized_int8,
    cuda_transpose_dot_product_3d_compute_optimized_int8_experimental,
    get_compute_optimized_3d_int8_experimental_kernel,
    get_compute_optimized_3d_int8_kernel,
)
from .dia_matvec import (
    CustomDIA3DMatVec,
    CustomDIAMatVec,
    benchmark_dia_kernels,
    create_test_dia_matrix,
    verify_kernel_correctness,
)

__all__ = [
    # FP32 kernels
    "get_compute_optimized_3d_kernel",
    "cuda_transpose_dot_product_3d_compute_optimized",
    "get_compute_optimized_3d_int8_kernel",
    "get_compute_optimized_3d_int8_experimental_kernel",
    "cuda_transpose_dot_product_3d_compute_optimized_int8",
    "cuda_transpose_dot_product_3d_compute_optimized_int8_experimental",
    "CustomDIAMatVec",
    "CustomDIA3DMatVec",
    "create_test_dia_matrix",
    "benchmark_dia_kernels",
    "verify_kernel_correctness",
]
