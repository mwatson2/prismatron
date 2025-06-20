"""
Utility modules for the Prismatron LED Display System.

This package contains shared utilities used across different components
of the Prismatron software stack.
"""

from .optimization_utils import ImageComparison, OptimizationPipeline

__all__ = ["OptimizationPipeline", "ImageComparison"]
