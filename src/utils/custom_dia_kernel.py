#!/usr/bin/env python3
"""
Custom CUDA kernel for optimized DIA (diagonal) matrix-vector multiplication.

This module provides compatibility imports for the reorganized DIA kernels.
The actual kernel implementations are now located in the utils.kernels submodule.
"""

import time
from typing import Optional, Tuple

import cupy
import cupyx.scipy.sparse as cusp
import numpy as np

# Import all DIA kernels from the reorganized structure
from .kernels import (
    CustomDIA3DMatVec,
    CustomDIA3DMatVecFP16,
    CustomDIAMatVec,
    CustomDIAMatVecFP16,
    benchmark_dia_kernels,
    create_test_dia_matrix,
    verify_kernel_correctness,
)

# Legacy kernel definitions are now imported from kernels submodule
# All kernel code has been moved to separate files for better organization

# Backward compatibility - maintain existing API
__all__ = [
    "CustomDIAMatVec",
    "CustomDIA3DMatVec",
    "CustomDIAMatVecFP16",
    "CustomDIA3DMatVecFP16",
    "create_test_dia_matrix",
    "benchmark_dia_kernels",
    "verify_kernel_correctness",
]


# Direct access for backward compatibility
def main():
    """Main function delegates to the kernels module."""
    from .kernels.dia_matvec import main as kernel_main

    return kernel_main()


if __name__ == "__main__":
    main()
