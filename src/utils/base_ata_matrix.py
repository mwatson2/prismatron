#!/usr/bin/env python3
"""
Abstract base class for A^T A matrix implementations.

Provides common interface for different A^T A matrix storage and computation strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional

try:
    import cupy
except ImportError:
    # Fallback for systems without CUDA
    import numpy as cupy

import numpy as np


class BaseATAMatrix(ABC):
    """
    Abstract base class for A^T A matrix implementations.

    Defines the core computational interface that all A^T A matrix
    implementations must provide.
    """

    @abstractmethod
    def multiply_3d(
        self,
        led_values: np.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
        output_dtype: Optional[cupy.dtype] = None,
        debug_logging: bool = False,
    ) -> np.ndarray:
        """
        Perform 3D DIA matrix-vector multiplication: (A^T)A @ led_values.

        Args:
            led_values: LED values array (3, leds)
            use_custom_kernel: Whether to use custom kernels
            optimized_kernel: Whether to use optimized kernel variant
            output_dtype: Desired output data type
            debug_logging: Enable debug logging

        Returns:
            Result array (3, leds)
        """

    @abstractmethod
    def g_ata_g_3d(
        self,
        gradient: np.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
        output_dtype: Optional[cupy.dtype] = None,
    ) -> np.ndarray:
        """
        Compute g^T (A^T A) g for step size calculation.

        Args:
            gradient: Gradient array (3, leds)
            use_custom_kernel: Whether to use custom kernels
            optimized_kernel: Whether to use optimized kernel variant
            output_dtype: Desired output data type

        Returns:
            Result array (3,) - one value per channel
        """
