"""
Custom CUDA kernels for SingleBlockMixedSparseTensor operations.

This module provides compatibility imports for the reorganized CUDA kernels.
The actual kernel implementations are now located in the utils.kernels submodule.
"""

import logging
from typing import Tuple

import cupy as cp
import numpy as np

# Import all kernels from the reorganized structure
from .kernels import (
    cuda_transpose_dot_product_3d_compute_optimized,
    cuda_transpose_dot_product_3d_compute_optimized_int8,
    cuda_transpose_dot_product_3d_compute_optimized_int8_experimental,
    get_compute_optimized_3d_int8_experimental_kernel,
    get_compute_optimized_3d_int8_kernel,
    get_compute_optimized_3d_kernel,
)

logger = logging.getLogger(__name__)

# Legacy kernel definitions are now imported from kernels submodule
# All kernel code has been moved to separate files for better organization

# Backward compatibility - maintain existing API
__all__ = [
    "get_compute_optimized_3d_kernel",
    "cuda_transpose_dot_product_3d_compute_optimized",
    "get_compute_optimized_3d_int8_kernel",
    "get_compute_optimized_3d_int8_experimental_kernel",
    "cuda_transpose_dot_product_3d_compute_optimized_int8",
    "cuda_transpose_dot_product_3d_compute_optimized_int8_experimental",
]
