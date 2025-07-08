#!/usr/bin/env python3
"""
Diagonal A^T A Matrix utility class for LED optimization.

This class encapsulates the A^T A matrix in DIA format with RCM ordering
for efficient (A^T)Ax operations in LED optimization.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cupy
import numpy as np
import scipy.sparse as sp

# Note: RCM ordering is now handled by pattern generation tool directly

# Import custom DIA kernels
try:
    from .custom_dia_kernel import (
        CustomDIA3DMatVec,
        CustomDIA3DMatVecFP16,
        CustomDIAMatVec,
        CustomDIAMatVecFP16,
    )

    CUSTOM_KERNEL_AVAILABLE = True
    CUSTOM_3D_KERNEL_AVAILABLE = True
    CUSTOM_KERNEL_FP16_AVAILABLE = True
    CUSTOM_3D_KERNEL_FP16_AVAILABLE = True
except ImportError:
    CUSTOM_KERNEL_AVAILABLE = False
    CUSTOM_3D_KERNEL_AVAILABLE = False
    CUSTOM_KERNEL_FP16_AVAILABLE = False
    CUSTOM_3D_KERNEL_FP16_AVAILABLE = False
    print("Warning: Custom DIA kernels not available")


class DiagonalATAMatrix:
    """
    Utility class for diagonal A^T A matrices in LED optimization.

    Stores A^T A matrices for each RGB channel in true 3D DIA format with shared
    diagonal structure for efficient matrix-vector multiplication.

    Note: Expects diffusion matrix A to already be in optimal ordering (RCM)
    from pattern generation tool.

    3D DIA Format:
    - data: (3, k, leds) - diagonal band data for R, G, B channels
    - offsets: (k,) - shared diagonal offsets
    """

    def __init__(
        self,
        led_count: int,
        crop_size: int = 64,
        output_dtype: Optional[cupy.dtype] = None,
    ):
        """
        Initialize diagonal A^T A matrix container.

        Args:
            led_count: Number of LEDs
            crop_size: Crop size used for LED regions (affects adjacency)
            output_dtype: Data type for output tensors (cupy.float32 or cupy.float16).
                         If None, defaults to cupy.float32.
        """
        self.led_count = led_count
        self.crop_size = crop_size
        self.channels = 3  # RGB

        # Set output dtype with default
        self.output_dtype = output_dtype if output_dtype is not None else cupy.float32
        if self.output_dtype not in (cupy.float32, cupy.float16):
            raise ValueError(f"Unsupported output dtype {self.output_dtype}. Supported: cupy.float32, cupy.float16")

        # Storage for unified 3D DIA format - shape (channels, k, leds)
        self.dia_data_cpu = None  # Shape: (channels, k, leds) - unified diagonal band data
        self.dia_data_gpu = None  # CuPy version of above
        self.dia_offsets = None  # Shape: (k,) - unified diagonal offsets for non-empty diagonals only
        self.dia_offsets_gpu = None  # CuPy version of dia_offsets, cached for performance
        self.k = None  # Number of non-empty diagonal bands (max across all channels)

        # Note: RCM ordering handled by pattern generation, not stored here

        # Custom kernel instances (FP32)
        self.custom_kernel_basic = None
        self.custom_kernel_optimized = None

        # 3D kernel instances (FP32)
        self.custom_3d_kernel_basic = None
        self.custom_3d_kernel_optimized = None

        # FP16 kernel instances
        self.custom_kernel_basic_fp16 = None
        self.custom_kernel_optimized_fp16 = None

        # 3D FP16 kernel instances
        self.custom_3d_kernel_basic_fp16 = None
        self.custom_3d_kernel_optimized_fp16 = None

        # Metadata
        self.bandwidth = None
        self.sparsity = None  # Overall sparsity
        self.nnz = None  # Total non-zeros
        self.channel_nnz = [None, None, None]  # Per-channel nnz for info

        # Initialize custom kernels if available
        if CUSTOM_KERNEL_AVAILABLE:
            self.custom_kernel_basic = CustomDIAMatVec(use_optimized=False)
            self.custom_kernel_optimized = CustomDIAMatVec(use_optimized=True)

        # Initialize FP16 kernels if available
        if CUSTOM_KERNEL_FP16_AVAILABLE:
            self.custom_kernel_basic_fp16 = CustomDIAMatVecFP16(use_optimized=False)
            self.custom_kernel_optimized_fp16 = CustomDIAMatVecFP16(use_optimized=True)

    def _update_dia_offsets_cache(self):
        """Update GPU cache for dia_offsets to avoid repeated cupy.asarray calls."""
        if self.dia_offsets is not None:
            self.dia_offsets_gpu = cupy.asarray(self.dia_offsets, dtype=cupy.int32)
        else:
            self.dia_offsets_gpu = None

    def build_from_diffusion_matrix(self, diffusion_matrix: sp.spmatrix) -> None:
        """
        Build diagonal A^T A matrices from diffusion matrix A.

        Note: Expects diffusion matrix A to already be in optimal ordering
        (RCM) from pattern generation tool.

        Args:
            A: Diffusion matrix (pixels, leds*3) in optimal ordering
        """
        print("Building diagonal A^T A matrices...")
        print(f"  Input A: shape {diffusion_matrix.shape}, nnz {diffusion_matrix.nnz}")
        print(f"  LEDs: {self.led_count}, channels: {self.channels}")

        # Validate input
        expected_cols = self.led_count * self.channels
        if diffusion_matrix.shape[1] != expected_cols:
            raise ValueError(f"A matrix should have {expected_cols} columns, got {diffusion_matrix.shape[1]}")

        print("  Using diffusion matrix A in pre-optimized ordering (RCM from pattern generation)")
        A_ordered = diffusion_matrix

        # Build unified 3D DIA format - shape (channels, k, leds) where k = max non-empty diagonals
        print("  Building unified 3D DIA format...")

        # First pass: compute A^T A for each channel and collect all diagonal info
        channel_matrices = []
        channel_dia_matrices = []
        all_offsets = set()
        total_nnz = 0

        for channel in range(self.channels):
            print(f"  Processing channel {channel} ({['R', 'G', 'B'][channel]})...")

            # Extract channel columns: A_ordered shape (pixels, leds*3)
            channel_cols = np.arange(channel, A_ordered.shape[1], self.channels)
            A_channel = A_ordered[:, channel_cols]  # Shape: (pixels, leds)

            # Compute A^T A for this channel: (leds, pixels) @ (pixels, leds) -> (leds, leds)
            ATA_channel = A_channel.T @ A_channel

            # Convert to DIA format
            ATA_dia = sp.dia_matrix(ATA_channel)
            channel_matrices.append(ATA_channel)
            channel_dia_matrices.append(ATA_dia)

            # Collect all unique diagonal offsets that have non-zero elements
            for i, offset in enumerate(ATA_dia.offsets):
                diagonal_data = ATA_dia.data[i]  # Shape: (leds,)
                if np.any(np.abs(diagonal_data) > 1e-10):  # Non-zero threshold
                    all_offsets.add(offset)

            # Store per-channel metadata
            self.channel_nnz[channel] = ATA_dia.nnz
            total_nnz += ATA_dia.nnz

            print(f"    A^T A shape: {ATA_dia.shape}, nnz: {ATA_dia.nnz}")
            print(f"    Channel sparsity: {ATA_dia.nnz / (self.led_count * self.led_count) * 100:.3f}%")
            print(f"    Total DIA bands: {len(ATA_dia.offsets)}")

        # Create unified diagonal structure - only non-empty diagonals across ALL channels
        if all_offsets:
            self.dia_offsets = np.array(sorted(all_offsets), dtype=np.int32)  # Shape: (k,)
            self.k = len(self.dia_offsets)
        else:
            self.dia_offsets = np.array([], dtype=np.int32)
            self.k = 0

        # Cache GPU version of dia_offsets for performance
        self._update_dia_offsets_cache()

        print("  Unified diagonal structure:")
        print(f"    Non-empty diagonal bands (k): {self.k}")
        if self.k > 0:
            print(f"    Offset range: [{self.dia_offsets[0]}, {self.dia_offsets[-1]}]")

        # Create unified 3D DIA data structure: (channels, k, leds)
        if self.k > 0:
            self.dia_data_cpu = np.zeros((self.channels, self.k, self.led_count), dtype=np.float32)

            # Create mapping from offset to index in unified structure
            offset_to_idx = {offset: i for i, offset in enumerate(self.dia_offsets)}

            # Fill unified structure from each channel
            for channel in range(self.channels):
                ATA_dia = channel_dia_matrices[channel]

                for i, offset in enumerate(ATA_dia.offsets):
                    diagonal_data = ATA_dia.data[i]  # Shape: (leds,)

                    # Only store if this diagonal has non-zero elements
                    if np.any(np.abs(diagonal_data) > 1e-10):
                        unified_idx = offset_to_idx[offset]
                        self.dia_data_cpu[channel, unified_idx, :] = diagonal_data.astype(np.float32)

            # Create GPU version
            self.dia_data_gpu = cupy.asarray(self.dia_data_cpu)  # Shape: (channels, k, leds)

            # Calculate storage efficiency
            naive_elements = (
                len(set().union(*[dia.offsets for dia in channel_dia_matrices])) * self.channels * self.led_count
            )
            actual_elements = self.k * self.channels * self.led_count
            storage_efficiency = actual_elements / naive_elements * 100 if naive_elements > 0 else 100

            print(f"  Unified 3D DIA storage: shape {self.dia_data_cpu.shape}")
            total_unique_bands = len(set().union(*[dia.offsets for dia in channel_dia_matrices]))
            print(f"  Storage efficiency: {self.k} / {total_unique_bands} bands = {storage_efficiency:.1f}%")
            dense_elements = self.channels * self.led_count * self.led_count
            print(f"  Total stored elements: {actual_elements:,} vs dense {dense_elements:,}")

        else:
            self.dia_data_cpu = np.zeros((self.channels, 0, self.led_count), dtype=np.float32)
            self.dia_data_gpu = cupy.asarray(self.dia_data_cpu)

        # Store overall metadata
        self.nnz = total_nnz
        total_elements = self.channels * self.led_count * self.led_count
        self.sparsity = self.nnz / total_elements * 100

        # Compute 3D DIA sparsity
        dia_3d_nnz = np.count_nonzero(self.dia_data_cpu)
        dia_3d_elements = self.channels * self.k * self.led_count
        dia_3d_sparsity = dia_3d_nnz / dia_3d_elements * 100

        # Compute bandwidth from 3D DIA offsets
        self.bandwidth = int(np.max(np.abs(self.dia_offsets))) if self.k > 0 else 0

        print(f"  3D DIA structure: shape {self.dia_data_cpu.shape}")
        print(f"  3D DIA nnz: {dia_3d_nnz}, sparsity: {dia_3d_sparsity:.3f}%")
        print(f"  Total nnz: {self.nnz}, overall sparsity: {self.sparsity:.3f}%")
        print(f"  Estimated bandwidth: {self.bandwidth}")

        print("  True 3D DIA matrices built successfully!")

    def multiply_3d(
        self,
        led_values: np.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
        output_dtype: Optional[cupy.dtype] = None,
    ) -> np.ndarray:
        """
        Perform 3D DIA matrix-vector multiplication: (A^T)A @ led_values.

        This performs the 3D einsum operation: ijk,ik->ij where:
        - i: channel (R, G, B)
        - j,k: LED indices
        - Input: led_values (3, leds)
        - Output: result (3, leds)

        Args:
            led_values: LED values array (3, leds) in RCM order
            use_custom_kernel: Whether to use custom 3D DIA kernel
            optimized_kernel: Whether to use optimized custom kernel
            output_dtype: Desired output data type (cupy.float32 or cupy.float16).
                         If None, uses the instance's output_dtype setting.

        Returns:
            Result array (3, leds) in RCM order with specified output dtype
        """
        if led_values.shape != (self.channels, self.led_count):
            raise ValueError(f"LED values should be shape ({self.channels}, {self.led_count}), got {led_values.shape}")

        # Check if unified 3D matrix is built
        if self.dia_data_gpu is None or self.dia_data_gpu.shape != (
            self.channels,
            self.k,
            self.led_count,
        ):
            raise RuntimeError("Unified 3D DIA matrix not built. Call build_from_diffusion_matrix() first.")

        # Determine output dtype
        if output_dtype is None:
            output_dtype = self.output_dtype

        # Validate output dtype
        if output_dtype not in (cupy.float32, cupy.float16):
            raise ValueError(f"Unsupported output dtype {output_dtype}. Supported: cupy.float32, cupy.float16")

        # Determine expected input dtype based on output dtype and kernel requirements
        if output_dtype == cupy.float16:
            expected_input_dtype = cupy.float16  # FP16 kernels expect FP16 input
        else:
            expected_input_dtype = cupy.float32  # FP32 kernels expect FP32 input

        # Assert input is cupy array of correct dtype for optimal performance
        if not isinstance(led_values, cupy.ndarray):
            raise TypeError(
                f"led_values must be a cupy.ndarray of dtype {expected_input_dtype.__name__}, "
                f"got {type(led_values).__name__}. Convert before calling multiply_3d."
            )

        if led_values.dtype != expected_input_dtype:
            raise TypeError(
                f"led_values dtype must be {expected_input_dtype.__name__} for {output_dtype.__name__} output, "
                f"got {led_values.dtype}. Convert before calling multiply_3d to avoid unexpected copies."
            )

        led_values_gpu = led_values  # No conversion needed

        # Perform 3D DIA matrix-vector multiplication - Select kernel based on output dtype
        if use_custom_kernel:
            if output_dtype == cupy.float32:
                # Use FP32 kernels
                if not CUSTOM_3D_KERNEL_AVAILABLE:
                    raise RuntimeError("Custom 3D DIA FP32 kernel not available - required for performance measurement")

                if optimized_kernel:
                    if self.custom_3d_kernel_optimized is None:
                        self.custom_3d_kernel_optimized = CustomDIA3DMatVec(use_optimized=True)
                    result_gpu = self.custom_3d_kernel_optimized(
                        self.dia_data_gpu,  # Shape: (channels, k, leds)
                        self.dia_offsets_gpu,  # Shape: (k,) - cached GPU version
                        led_values_gpu,  # Shape: (channels, leds)
                    )
                else:
                    if self.custom_3d_kernel_basic is None:
                        self.custom_3d_kernel_basic = CustomDIA3DMatVec(use_optimized=False)
                    result_gpu = self.custom_3d_kernel_basic(
                        self.dia_data_gpu,  # Shape: (channels, k, leds)
                        self.dia_offsets_gpu,  # Shape: (k,) - cached GPU version
                        led_values_gpu,  # Shape: (channels, leds)
                    )

            elif output_dtype == cupy.float16:
                # Use FP16 kernels
                if not CUSTOM_3D_KERNEL_FP16_AVAILABLE:
                    raise RuntimeError("Custom 3D DIA FP16 kernel not available - required for FP16 output")

                if optimized_kernel:
                    if self.custom_3d_kernel_optimized_fp16 is None:
                        self.custom_3d_kernel_optimized_fp16 = CustomDIA3DMatVecFP16(use_optimized=True)
                    result_gpu = self.custom_3d_kernel_optimized_fp16(
                        self.dia_data_gpu,  # Shape: (channels, k, leds)
                        self.dia_offsets_gpu,  # Shape: (k,) - cached GPU version
                        led_values_gpu,  # Shape: (channels, leds)
                    )
                else:
                    if self.custom_3d_kernel_basic_fp16 is None:
                        self.custom_3d_kernel_basic_fp16 = CustomDIA3DMatVecFP16(use_optimized=False)
                    result_gpu = self.custom_3d_kernel_basic_fp16(
                        self.dia_data_gpu,  # Shape: (channels, k, leds)
                        self.dia_offsets_gpu,  # Shape: (k,) - cached GPU version
                        led_values_gpu,  # Shape: (channels, leds)
                    )
        else:
            # NO FALLBACK - custom kernel required for performance measurement
            raise RuntimeError("Custom 3D DIA kernel not available - required for performance measurement")

        # Convert back to numpy if input was numpy
        if isinstance(led_values, np.ndarray):
            return cupy.asnumpy(result_gpu)
        else:
            return result_gpu

    def _multiply_3d_fallback(self, led_values_gpu: cupy.ndarray) -> cupy.ndarray:
        """
        Unified 3D DIA multiplication using einsum-like operation on (channels, k, leds) tensor.

        Performs: result[c,i] = sum_k sum_j A[c,k,i] * delta(i+offset[k], j) * led_values[c,j]

        Args:
            led_values_gpu: LED values (channels, leds) on GPU

        Returns:
            Result (channels, leds) on GPU
        """
        result_gpu = cupy.zeros_like(led_values_gpu)  # Shape: (channels, leds)

        if self.k == 0:
            return result_gpu

        # Unified 3D DIA multiplication - process all channels in single operation
        # For each diagonal band k with offset dia_offsets[k]
        for band_idx in range(self.k):
            offset = int(self.dia_offsets[band_idx])  # Diagonal offset

            # Get diagonal data for all channels: shape (channels, leds)
            band_data_all_channels = self.dia_data_gpu[:, band_idx, :]  # Shape: (channels, leds)

            # DIA format: band_data[c,i] contains A[c,i,i+offset] for valid indices
            # Matrix multiplication: result[c,i] += A[c,i,i+offset] * led_values[c,i+offset]
            if offset >= 0:
                # Upper diagonal: A[c,i,i+offset] for i in [0, leds-offset)
                valid_i_range = self.led_count - offset
                if valid_i_range > 0:
                    # Vectorized across all channels: result[c,i] += band_data[c,i] * led_values[c,i+offset]
                    i_slice = slice(0, valid_i_range)
                    j_slice = slice(offset, offset + valid_i_range)
                    result_gpu[:, i_slice] += band_data_all_channels[:, i_slice] * led_values_gpu[:, j_slice]
            else:
                # Lower diagonal: A[c,i,i+offset] for i in [-offset, leds) where i+offset >= 0
                start_i = -offset
                valid_i_range = self.led_count - start_i
                if valid_i_range > 0:
                    # Vectorized across all channels: result[c,i] += band_data[c,i] * led_values[c,i+offset]
                    i_slice = slice(start_i, self.led_count)
                    j_slice = slice(0, valid_i_range)
                    result_gpu[:, i_slice] += band_data_all_channels[:, i_slice] * led_values_gpu[:, j_slice]

        return result_gpu

    def g_ata_g_3d(
        self,
        gradient: np.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
        output_dtype: Optional[cupy.dtype] = None,
    ) -> np.ndarray:
        """
        Compute g^T (A^T A) g for step size calculation using 3D DIA format.

        This performs the 3D einsum operation: ij,ijk,ik->i where:
        - i: channel (R, G, B)
        - j,k: LED indices
        - Input: gradient g (3, leds) in RCM order
        - Output: result (3,) - one value per channel

        Args:
            gradient: Gradient array (3, leds) in RCM order
            use_custom_kernel: Whether to use custom 3D DIA kernel
            optimized_kernel: Whether to use optimized custom kernel
            output_dtype: Desired output data type (cupy.float32 or cupy.float16).
                         If None, uses the instance's output_dtype setting.

        Returns:
            Result array (3,) - g^T (A^T A) g for each channel with specified output dtype
        """
        if gradient.shape != (self.channels, self.led_count):
            raise ValueError(f"Gradient should be shape ({self.channels}, {self.led_count}), got {gradient.shape}")

        if self.dia_data_gpu is None or self.dia_data_gpu.shape != (
            self.channels,
            self.k,
            self.led_count,
        ):
            raise RuntimeError("Unified 3D DIA matrix not built. Call build_from_diffusion_matrix() first.")

        # Determine output dtype
        if output_dtype is None:
            output_dtype = self.output_dtype

        # Validate output dtype
        if output_dtype not in (cupy.float32, cupy.float16):
            raise ValueError(f"Unsupported output dtype {output_dtype}. Supported: cupy.float32, cupy.float16")

        # Convert to GPU arrays - avoid unnecessary copies
        if not isinstance(gradient, cupy.ndarray):
            gradient_gpu = cupy.asarray(gradient, dtype=cupy.float32)
        else:
            # Only convert dtype if necessary, avoid copy if already float32
            if gradient.dtype != cupy.float32:
                gradient_gpu = gradient.astype(cupy.float32)
            else:
                gradient_gpu = gradient

        # Compute (A^T A) @ g using unified 3D DIA - Select kernel based on output dtype
        if use_custom_kernel:
            if output_dtype == cupy.float32:
                # Use FP32 kernels
                if not CUSTOM_3D_KERNEL_AVAILABLE:
                    raise RuntimeError("Custom 3D DIA FP32 kernel not available - required for performance measurement")

                if optimized_kernel:
                    if self.custom_3d_kernel_optimized is None:
                        self.custom_3d_kernel_optimized = CustomDIA3DMatVec(use_optimized=True)
                    ata_g_gpu = self.custom_3d_kernel_optimized(
                        self.dia_data_gpu,  # Shape: (channels, k, leds)
                        self.dia_offsets_gpu,  # Shape: (k,) - cached GPU version
                        gradient_gpu,  # Shape: (channels, leds)
                    )
                else:
                    if self.custom_3d_kernel_basic is None:
                        self.custom_3d_kernel_basic = CustomDIA3DMatVec(use_optimized=False)
                    ata_g_gpu = self.custom_3d_kernel_basic(
                        self.dia_data_gpu,  # Shape: (channels, k, leds)
                        self.dia_offsets_gpu,  # Shape: (k,) - cached GPU version
                        gradient_gpu,  # Shape: (channels, leds)
                    )

            elif output_dtype == cupy.float16:
                # Use FP16 kernels
                if not CUSTOM_3D_KERNEL_FP16_AVAILABLE:
                    raise RuntimeError("Custom 3D DIA FP16 kernel not available - required for FP16 output")

                if optimized_kernel:
                    if self.custom_3d_kernel_optimized_fp16 is None:
                        self.custom_3d_kernel_optimized_fp16 = CustomDIA3DMatVecFP16(use_optimized=True)
                    ata_g_gpu = self.custom_3d_kernel_optimized_fp16(
                        self.dia_data_gpu,  # Shape: (channels, k, leds)
                        self.dia_offsets_gpu,  # Shape: (k,) - cached GPU version
                        gradient_gpu,  # Shape: (channels, leds)
                    )
                else:
                    if self.custom_3d_kernel_basic_fp16 is None:
                        self.custom_3d_kernel_basic_fp16 = CustomDIA3DMatVecFP16(use_optimized=False)
                    ata_g_gpu = self.custom_3d_kernel_basic_fp16(
                        self.dia_data_gpu,  # Shape: (channels, k, leds)
                        self.dia_offsets_gpu,  # Shape: (k,) - cached GPU version
                        gradient_gpu,  # Shape: (channels, leds)
                    )
        else:
            # NO FALLBACK - custom kernel required for performance measurement
            raise RuntimeError("Custom 3D DIA kernel not available - required for performance measurement")

        # Compute g^T @ (A^T A @ g) for each channel using vectorized operation:
        # (channels,leds) * (channels,leds) -> (channels,)
        result_gpu = cupy.sum(gradient_gpu * ata_g_gpu, axis=1)  # Shape: (channels,)

        # Convert back to numpy if input was numpy
        if isinstance(gradient, np.ndarray):
            return cupy.asnumpy(result_gpu)
        else:
            return result_gpu

    # Note: RCM reordering methods removed - ordering handled by pattern generation

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary containing all necessary data
        """

        return {
            "led_count": self.led_count,
            "crop_size": self.crop_size,
            "channels": self.channels,
            # Unified 3D DIA format (current)
            "dia_data_3d": self.dia_data_cpu,
            "dia_offsets_3d": self.dia_offsets,
            "k": self.k,
            # Metadata
            "bandwidth": self.bandwidth,
            "sparsity": self.sparsity,
            "nnz": self.nnz,
            "channel_nnz": self.channel_nnz,
            "output_dtype": self.output_dtype.__name__,  # Store output dtype name
            "version": "8.0",  # Added FP16 support
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagonalATAMatrix":
        """
        Create instance from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            DiagonalATAMatrix instance
        """
        # Handle output_dtype - backward compatibility with files that don't have output_dtype
        output_dtype = None
        if "output_dtype" in data:
            output_dtype_str = str(data["output_dtype"])
            if output_dtype_str == "float32":
                output_dtype = cupy.float32
            elif output_dtype_str == "float16":
                output_dtype = cupy.float16

        # Create instance
        instance = cls(data["led_count"], data["crop_size"], output_dtype)

        # Restore metadata
        instance.channels = data["channels"]
        instance.bandwidth = data["bandwidth"]
        instance.sparsity = data["sparsity"]
        instance.nnz = data["nnz"]

        # Handle version compatibility
        version = data.get("version", "1.0")

        if version in ("8.0", "7.0"):
            # Current unified 3D DIA format (no RCM ordering stored)
            instance.dia_data_cpu = data["dia_data_3d"]
            instance.dia_offsets = data["dia_offsets_3d"]
            instance.k = data["k"]
            instance.channel_nnz = data["channel_nnz"]

            # Create GPU version
            if instance.dia_data_cpu is not None:
                instance.dia_data_gpu = cupy.asarray(instance.dia_data_cpu)
            else:
                instance.dia_data_gpu = None

            # Cache GPU version of dia_offsets
            instance._update_dia_offsets_cache()

        elif version == "6.0":
            # Legacy version with RCM ordering stored (still supported)
            instance.dia_data_cpu = data["dia_data_3d"]
            instance.dia_offsets = data["dia_offsets_3d"]
            instance.k = data["k"]
            instance.channel_nnz = data["channel_nnz"]

            # Create GPU version
            if instance.dia_data_cpu is not None:
                instance.dia_data_gpu = cupy.asarray(instance.dia_data_cpu)
            else:
                instance.dia_data_gpu = None

            # Cache GPU version of dia_offsets
            instance._update_dia_offsets_cache()

            print("Warning: Loading legacy DIA matrix with RCM ordering. Consider regenerating patterns.")

        else:
            # Legacy formats no longer supported
            raise ValueError(
                f"Legacy format version {version} not supported. Please rebuild matrices with current version."
            )

        return instance

    def save(self, filepath: str) -> None:
        """
        Save to .npz file.

        Args:
            filepath: Path to save file
        """
        data = self.to_dict()
        np.savez_compressed(filepath, **data)
        print(f"Diagonal A^T A matrices saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DiagonalATAMatrix":
        """
        Load from .npz file.

        Args:
            filepath: Path to load file

        Returns:
            DiagonalATAMatrix instance
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        data_npz = np.load(filepath, allow_pickle=True)
        data = {key: data_npz[key].item() if data_npz[key].ndim == 0 else data_npz[key] for key in data_npz}

        instance = cls.from_dict(data)
        print(f"Diagonal A^T A matrices loaded from {filepath}")
        return instance

    def get_info(self) -> Dict[str, Any]:
        """
        Get summary information about the matrices.

        Returns:
            Info dictionary
        """
        # Calculate unified storage stats
        unified_storage_built = self.dia_data_cpu is not None and self.k is not None

        return {
            "led_count": self.led_count,
            "crop_size": self.crop_size,
            "channels": self.channels,
            "bandwidth": self.bandwidth,
            "sparsity": self.sparsity,
            "nnz": self.nnz,
            "channel_nnz": self.channel_nnz,
            "ordering": "pre_optimized_from_pattern_generation",
            "custom_kernel_available": CUSTOM_KERNEL_AVAILABLE,
            "custom_kernel_fp16_available": CUSTOM_KERNEL_FP16_AVAILABLE,
            "output_dtype": str(self.output_dtype),
            "unified_storage_built": unified_storage_built,
            "unified_k": self.k,
            "unified_storage_shape": (self.dia_data_cpu.shape if self.dia_data_cpu is not None else None),
            "storage_format": "unified_3d_dia_v8_fp16",
        }

    def benchmark_3d(self, num_trials: int = 50, num_warmup: int = 10) -> Dict[str, float]:
        """
        Benchmark 3D DIA multiplication performance.

        Args:
            num_trials: Number of timing trials
            num_warmup: Number of warmup iterations

        Returns:
            Timing results dictionary
        """
        if self.dia_data_gpu is None:
            raise RuntimeError("3D DIA matrix not built")

        # Create test LED values
        test_values_cpu = [
            np.random.randn(self.channels, self.led_count).astype(np.float32) for _ in range(num_trials + num_warmup)
        ]
        test_values_gpu = [cupy.asarray(x) for x in test_values_cpu]

        results = {}

        # GPU timing (3D DIA multiply - fallback)
        for i in range(num_warmup):
            _ = self.multiply_3d(test_values_gpu[i], use_custom_kernel=False)
        cupy.cuda.Stream.null.synchronize()

        times = []
        for i in range(num_warmup, num_warmup + num_trials):
            start_event = cupy.cuda.Event()
            end_event = cupy.cuda.Event()

            start_event.record()
            _ = self.multiply_3d(test_values_gpu[i], use_custom_kernel=False)
            end_event.record()
            end_event.synchronize()

            times.append(cupy.cuda.get_elapsed_time(start_event, end_event))

        results["gpu_3d_fallback"] = np.mean(times)

        # GPU timing (3D DIA g_ata_g - fallback)
        for i in range(num_warmup):
            _ = self.g_ata_g_3d(test_values_gpu[i], use_custom_kernel=False)
        cupy.cuda.Stream.null.synchronize()

        times = []
        for i in range(num_warmup, num_warmup + num_trials):
            start_event = cupy.cuda.Event()
            end_event = cupy.cuda.Event()

            start_event.record()
            _ = self.g_ata_g_3d(test_values_gpu[i], use_custom_kernel=False)
            end_event.record()
            end_event.synchronize()

            times.append(cupy.cuda.get_elapsed_time(start_event, end_event))

        results["gpu_3d_g_ata_g_fallback"] = np.mean(times)

        # GPU timing (3D DIA custom kernel)
        if CUSTOM_KERNEL_AVAILABLE:
            for i in range(num_warmup):
                _ = self.multiply_3d(test_values_gpu[i], use_custom_kernel=True)
            cupy.cuda.Stream.null.synchronize()

            times = []
            for i in range(num_warmup, num_warmup + num_trials):
                start_event = cupy.cuda.Event()
                end_event = cupy.cuda.Event()

                start_event.record()
                _ = self.multiply_3d(test_values_gpu[i], use_custom_kernel=True)
                end_event.record()
                end_event.synchronize()

                times.append(cupy.cuda.get_elapsed_time(start_event, end_event))

            results["gpu_3d_custom"] = np.mean(times)

            # GPU timing (3D DIA custom kernel g_ata_g)
            for i in range(num_warmup):
                _ = self.g_ata_g_3d(test_values_gpu[i], use_custom_kernel=True)
            cupy.cuda.Stream.null.synchronize()

            times = []
            for i in range(num_warmup, num_warmup + num_trials):
                start_event = cupy.cuda.Event()
                end_event = cupy.cuda.Event()

                start_event.record()
                _ = self.g_ata_g_3d(test_values_gpu[i], use_custom_kernel=True)
                end_event.record()
                end_event.synchronize()

                times.append(cupy.cuda.get_elapsed_time(start_event, end_event))

            results["gpu_3d_custom_g_ata_g"] = np.mean(times)

        return results

    def get_channel_dia_matrix(self, channel: int) -> sp.dia_matrix:
        """
        Extract a single channel's A^T A matrix as scipy.sparse.dia_matrix.

        Args:
            channel: Channel index (0=Red, 1=Green, 2=Blue)

        Returns:
            scipy.sparse.dia_matrix for the specified channel
        """
        if self.dia_data_cpu is None or self.dia_offsets is None:
            raise RuntimeError("DIA matrix not built yet. Call build_from_diffusion_matrix() first.")

        if not 0 <= channel < self.channels:
            raise ValueError(f"Channel must be 0-{self.channels - 1}, got {channel}")

        # Extract DIA data for this channel: (k, led_count)
        dia_data_channel = self.dia_data_cpu[channel, :, :]  # Shape: (k, led_count)

        # Create scipy DIA matrix
        # scipy.sparse.dia_matrix expects data shape (num_diags, matrix_size)
        # and offsets shape (num_diags,)
        scipy_dia = sp.dia_matrix((dia_data_channel, self.dia_offsets), shape=(self.led_count, self.led_count))

        return scipy_dia
