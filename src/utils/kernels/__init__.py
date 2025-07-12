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
from .dia_matvec_fp16 import (
    CustomDIA3DMatVecFP16,
    CustomDIAMatVecFP16,
)
from .pure_fp16_dia_kernel import PureFP16DIA3DKernel, get_kernel_info

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
    # FP16 DIA kernels
    "CustomDIAMatVecFP16",
    "CustomDIA3DMatVecFP16",
    # Pure FP16 kernels
    "PureFP16DIA3DKernel",
    "get_kernel_info",
]
