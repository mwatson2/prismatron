#!/usr/bin/env python3
"""
Dense ATA Matrix Utility.

This module provides a dense matrix representation for A^T @ A operations,
complementing the diagonal (DIA) format for cases where the matrix is not
sufficiently sparse to benefit from DIA storage.

Dense format is preferred when:
- Bandwidth is close to matrix size (>80% of LEDs)
- Number of diagonals is large (>50% of theoretical maximum)
- Memory usage is acceptable for the LED count

Features:
- GPU-accelerated dense matrix operations using CuPy
- Support for both fp32 and fp16 precision
- Batch processing for RGB channels
- Memory-efficient storage and computation
- Compatible with existing optimization framework
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import cupy
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class DenseATAMatrix:
    """
    Dense ATA matrix representation for LED optimization.

    Stores A^T @ A matrices as dense 3D arrays on GPU for efficient computation.
    Suitable for cases where DIA format is not optimal due to high bandwidth.
    """

    def __init__(
        self,
        led_count: int,
        channels: int = 3,
        storage_dtype: cupy.dtype = cupy.float32,
        output_dtype: cupy.dtype = cupy.float32,
    ):
        """
        Initialize dense ATA matrix.

        Args:
            led_count: Number of LEDs
            channels: Number of color channels (default: 3 for RGB)
            storage_dtype: Data type for matrix storage (cupy.float32 or cupy.float16)
            output_dtype: Data type for computation output (cupy.float32 or cupy.float16)
        """
        self.led_count = led_count
        self.channels = channels
        self.storage_dtype = storage_dtype
        self.output_dtype = output_dtype

        # Validate dtypes
        if self.storage_dtype not in (cupy.float32, cupy.float16):
            raise ValueError(f"Unsupported storage dtype {self.storage_dtype}")
        if self.output_dtype not in (cupy.float32, cupy.float16):
            raise ValueError(f"Unsupported output dtype {self.output_dtype}")

        # Dense matrix storage: (channels, led_count, led_count)
        self.dense_matrices_gpu: Optional[cupy.ndarray] = None
        self.dense_matrices_cpu: Optional[np.ndarray] = None

        # Matrix properties
        self.is_built = False
        self.memory_mb = 0.0
        self.version = "1.0"

    def build_from_diffusion_matrix(self, diffusion_matrix: sp.csc_matrix) -> None:
        """
        Build dense ATA matrices from sparse diffusion matrix.

        Args:
            diffusion_matrix: Sparse diffusion matrix (pixels, leds*channels)
        """
        logger.info("Building dense ATA matrices from diffusion matrix...")

        if diffusion_matrix.shape[1] != self.led_count * self.channels:
            raise ValueError(
                f"Matrix column count {diffusion_matrix.shape[1]} doesn't match "
                f"expected {self.led_count * self.channels}"
            )

        # Convert to GPU sparse matrix
        diffusion_gpu = cupy.sparse.csc_matrix(diffusion_matrix)

        # Initialize dense matrices on GPU
        self.dense_matrices_gpu = cupy.zeros((self.channels, self.led_count, self.led_count), dtype=self.storage_dtype)

        # Compute A^T @ A for each channel - use same approach as DIA format
        for channel in range(self.channels):
            logger.info(f"Computing dense ATA for channel {channel + 1}/{self.channels}...")

            # Extract channel slice from diffusion matrix using interleaved pattern
            # This matches the CSC construction: col_idx = led_id * 3 + channel
            # Same approach as DIA matrix for consistency
            channel_cols = cupy.arange(channel, diffusion_gpu.shape[1], self.channels)
            A_channel = diffusion_gpu[:, channel_cols]

            # Compute A^T @ A using sparse operations (same as DIA approach)
            # Convert A_channel to CPU first to use scipy operations that work reliably
            logger.info("Converting channel to CPU for ATA computation...")
            A_channel_csr = A_channel.tocsr()  # Convert to CSR format on GPU
            A_channel_scipy = sp.csr_matrix(
                (
                    cupy.asnumpy(A_channel_csr.data),
                    cupy.asnumpy(A_channel_csr.indices),
                    cupy.asnumpy(A_channel_csr.indptr),
                ),
                shape=A_channel_csr.shape,
            )

            # Compute A^T @ A using scipy (same computation path as DIA)
            logger.info("Computing A^T @ A using scipy sparse operations...")
            ata_sparse_cpu = A_channel_scipy.T @ A_channel_scipy

            # Convert result to dense and move to GPU
            logger.info("Converting sparse result to dense and moving to GPU...")
            ata_dense_cpu = ata_sparse_cpu.toarray().astype(
                np.float32 if self.storage_dtype == cupy.float32 else np.float16
            )
            self.dense_matrices_gpu[channel] = cupy.asarray(ata_dense_cpu, dtype=self.storage_dtype)

            # Cleanup intermediate matrices
            del A_channel, A_channel_csr, A_channel_scipy, ata_sparse_cpu, ata_dense_cpu
            cupy.get_default_memory_pool().free_all_blocks()

        # Copy to CPU for serialization
        self.dense_matrices_cpu = cupy.asnumpy(self.dense_matrices_gpu)

        # Calculate memory usage
        total_elements = self.channels * self.led_count * self.led_count
        bytes_per_element = 4 if self.storage_dtype == cupy.float32 else 2
        self.memory_mb = (total_elements * bytes_per_element) / (1024 * 1024)

        self.is_built = True
        logger.info(f"Dense ATA matrices built: {self.led_count} LEDs, {self.memory_mb:.1f}MB")

    def multiply_vector(self, led_values: cupy.ndarray, output_dtype: Optional[cupy.dtype] = None) -> cupy.ndarray:
        """
        Multiply ATA matrices by LED values vector: result = (A^T @ A) @ led_values

        Args:
            led_values: LED values array, shape (3, led_count) in planar format
            output_dtype: Output data type (default: self.output_dtype)

        Returns:
            Result array, shape (3, led_count) in planar format
        """
        if not self.is_built:
            raise RuntimeError("Dense ATA matrices not built yet")

        if output_dtype is None:
            output_dtype = self.output_dtype

        # Validate input
        if led_values.shape != (self.channels, self.led_count):
            raise ValueError(f"Expected led_values shape {(self.channels, self.led_count)}, " f"got {led_values.shape}")

        # Ensure matrices are on GPU
        if self.dense_matrices_gpu is None:
            self.dense_matrices_gpu = cupy.asarray(self.dense_matrices_cpu, dtype=self.storage_dtype)

        # Prepare output
        result = cupy.zeros_like(led_values, dtype=output_dtype)

        # Matrix-vector multiplication for each channel
        for channel in range(self.channels):
            # Get ATA matrix for this channel
            ata_matrix = self.dense_matrices_gpu[channel]

            # Convert to computation dtype if needed
            if ata_matrix.dtype != output_dtype:
                ata_matrix = ata_matrix.astype(output_dtype)

            # Get LED values for this channel
            led_channel = led_values[channel].astype(output_dtype)

            # Debug: Check for issues
            if cupy.any(cupy.isnan(ata_matrix)) or cupy.any(cupy.isinf(ata_matrix)):
                logger.warning(f"Dense ATA matrix channel {channel} contains NaN/Inf values")
            if cupy.any(cupy.isnan(led_channel)) or cupy.any(cupy.isinf(led_channel)):
                logger.warning(f"LED values channel {channel} contain NaN/Inf values")

            # Matrix-vector multiplication: (led_count, led_count) @ (led_count,) -> (led_count,)
            result[channel] = ata_matrix @ led_channel

            # Debug: Check result
            if cupy.any(cupy.isnan(result[channel])) or cupy.any(cupy.isinf(result[channel])):
                logger.warning(f"Result channel {channel} contains NaN/Inf values")
            if cupy.max(cupy.abs(result[channel])) == 0:
                logger.warning(f"Result channel {channel} is all zeros")

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize dense ATA matrices to dictionary for saving.

        Returns:
            Dictionary containing matrix data and metadata
        """
        if not self.is_built:
            raise RuntimeError("Dense ATA matrices not built yet")

        return {
            "dense_matrices": self.dense_matrices_cpu,
            "led_count": self.led_count,
            "channels": self.channels,
            "storage_dtype": "float32" if self.storage_dtype == cupy.float32 else "float16",
            "output_dtype": "float32" if self.output_dtype == cupy.float32 else "float16",
            "memory_mb": self.memory_mb,
            "version": self.version,
            "format": "dense_ata",
            "matrix_shape": (self.channels, self.led_count, self.led_count),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DenseATAMatrix":
        """
        Load dense ATA matrices from dictionary.

        Args:
            data: Dictionary containing matrix data and metadata

        Returns:
            DenseATAMatrix instance
        """
        # Parse dtypes
        storage_dtype_str = data.get("storage_dtype", "float32")
        output_dtype_str = data.get("output_dtype", "float32")

        storage_dtype = cupy.float32 if storage_dtype_str == "float32" else cupy.float16
        output_dtype = cupy.float32 if output_dtype_str == "float32" else cupy.float16

        # Create instance
        instance = cls(
            led_count=data["led_count"],
            channels=data["channels"],
            storage_dtype=storage_dtype,
            output_dtype=output_dtype,
        )

        # Load matrices
        instance.dense_matrices_cpu = data["dense_matrices"]
        instance.memory_mb = data.get("memory_mb", 0.0)
        instance.version = data.get("version", "1.0")
        instance.is_built = True

        # Matrices will be loaded to GPU on first use
        instance.dense_matrices_gpu = None

        logger.info(f"Loaded dense ATA matrices: {instance.led_count} LEDs, {instance.memory_mb:.1f}MB")
        return instance

    def memory_info(self) -> Dict[str, float]:
        """
        Get memory usage information.

        Returns:
            Dictionary with memory usage details
        """
        return {
            "total_mb": self.memory_mb,
            "gpu_mb": self.memory_mb if self.dense_matrices_gpu is not None else 0.0,
            "cpu_mb": self.memory_mb if self.dense_matrices_cpu is not None else 0.0,
        }

    def get_info(self) -> Dict[str, Any]:
        """
        Get summary information about the dense matrices.

        Returns:
            Dictionary with matrix information
        """
        return {
            "format": "dense",
            "led_count": self.led_count,
            "channels": self.channels,
            "storage_dtype": "float32" if self.storage_dtype == cupy.float32 else "float16",
            "output_dtype": "float32" if self.output_dtype == cupy.float32 else "float16",
            "memory_mb": self.memory_mb,
            "matrix_shape": (self.channels, self.led_count, self.led_count),
            "is_built": self.is_built,
            "version": self.version,
        }

    def __str__(self) -> str:
        """String representation."""
        if self.is_built:
            return (
                f"DenseATAMatrix({self.led_count} LEDs, {self.channels} channels, "
                f"{self.memory_mb:.1f}MB, {self.storage_dtype})"
            )
        else:
            return f"DenseATAMatrix({self.led_count} LEDs, not built)"

    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()
