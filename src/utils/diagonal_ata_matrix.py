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
import cupyx.scipy.sparse as cusp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.spatial import cKDTree

# Import custom DIA kernel
try:
    from custom_dia_kernel import CustomDIAMatVec

    CUSTOM_KERNEL_AVAILABLE = True
except ImportError:
    CUSTOM_KERNEL_AVAILABLE = False
    print("Warning: Custom DIA kernel not available")


class DiagonalATAMatrix:
    """
    Utility class for diagonal A^T A matrices in LED optimization.

    Stores A^T A matrices for each RGB channel in true 3D DIA format with shared
    diagonal structure and RCM ordering for efficient matrix-vector multiplication.

    3D DIA Format:
    - data: (3, k, leds) - diagonal band data for R, G, B channels
    - offsets: (k,) - shared diagonal offsets
    """

    def __init__(self, led_count: int, crop_size: int = 64):
        """
        Initialize diagonal A^T A matrix container.

        Args:
            led_count: Number of LEDs
            crop_size: Crop size used for LED regions (affects adjacency)
        """
        self.led_count = led_count
        self.crop_size = crop_size
        self.channels = 3  # RGB

        # Storage for unified 3D DIA format - shape (channels, k, leds)
        self.dia_data_cpu = (
            None  # Shape: (channels, k, leds) - unified diagonal band data
        )
        self.dia_data_gpu = None  # CuPy version of above
        self.dia_offsets = (
            None  # Shape: (k,) - unified diagonal offsets for non-empty diagonals only
        )
        self.k = None  # Number of non-empty diagonal bands (max across all channels)

        # Legacy storage for compatibility (will be deprecated)
        self.dia_matrix_cpu = None  # For backward compatibility
        self.dia_matrix_gpu = None  # For backward compatibility
        self.channel_offsets = None  # For backward compatibility

        # RCM ordering information
        self.led_order = None
        self.inverse_led_order = None

        # Custom kernel instances
        self.custom_kernel_basic = None
        self.custom_kernel_optimized = None

        # Metadata
        self.bandwidth = None
        self.sparsity = None  # Overall sparsity
        self.nnz = None  # Total non-zeros
        self.channel_nnz = [None, None, None]  # Per-channel nnz for info

        # Initialize custom kernels if available
        if CUSTOM_KERNEL_AVAILABLE:
            self.custom_kernel_basic = CustomDIAMatVec(use_optimized=False)
            self.custom_kernel_optimized = CustomDIAMatVec(use_optimized=True)

    def build_from_diffusion_matrix(
        self, A: sp.spmatrix, led_positions: np.ndarray, use_rcm: bool = True
    ) -> None:
        """
        Build diagonal A^T A matrices from diffusion matrix A.

        Args:
            A: Diffusion matrix (pixels, leds*3)
            led_positions: LED positions array (leds, 2)
            use_rcm: Whether to apply RCM ordering
        """
        print(f"Building diagonal A^T A matrices...")
        print(f"  Input A: shape {A.shape}, nnz {A.nnz}")
        print(f"  LEDs: {self.led_count}, channels: {self.channels}")

        # Validate input
        expected_cols = self.led_count * self.channels
        if A.shape[1] != expected_cols:
            raise ValueError(
                f"A matrix should have {expected_cols} columns, got {A.shape[1]}"
            )

        if len(led_positions) != self.led_count:
            raise ValueError(
                f"LED positions should have {self.led_count} entries, got {len(led_positions)}"
            )

        # Apply RCM ordering if requested
        if use_rcm:
            print(f"  Applying RCM ordering...")
            self.led_order = self._compute_rcm_ordering(led_positions)
            self.inverse_led_order = np.argsort(self.led_order)

            # Reorder A matrix columns according to RCM
            A_reordered = self._reorder_diffusion_matrix(A, self.led_order)
        else:
            print(f"  Using original LED ordering...")
            self.led_order = np.arange(self.led_count)
            self.inverse_led_order = np.arange(self.led_count)
            A_reordered = A

        # Build unified 3D DIA format - shape (channels, k, leds) where k = max non-empty diagonals
        print(f"  Building unified 3D DIA format...")

        # First pass: compute A^T A for each channel and collect all diagonal info
        channel_matrices = []
        channel_dia_matrices = []
        all_offsets = set()
        total_nnz = 0

        for channel in range(self.channels):
            print(f"  Processing channel {channel} ({['R', 'G', 'B'][channel]})...")

            # Extract channel columns: A_reordered shape (pixels, leds*3)
            channel_cols = np.arange(channel, A_reordered.shape[1], self.channels)
            A_channel = A_reordered[:, channel_cols]  # Shape: (pixels, leds)

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
            print(
                f"    Channel sparsity: {ATA_dia.nnz / (self.led_count * self.led_count) * 100:.3f}%"
            )
            print(f"    Total DIA bands: {len(ATA_dia.offsets)}")

        # Create unified diagonal structure - only non-empty diagonals across ALL channels
        if all_offsets:
            self.dia_offsets = np.array(
                sorted(all_offsets), dtype=np.int32
            )  # Shape: (k,)
            self.k = len(self.dia_offsets)
        else:
            self.dia_offsets = np.array([], dtype=np.int32)
            self.k = 0

        print(f"  Unified diagonal structure:")
        print(f"    Non-empty diagonal bands (k): {self.k}")
        if self.k > 0:
            print(f"    Offset range: [{self.dia_offsets[0]}, {self.dia_offsets[-1]}]")

        # Create unified 3D DIA data structure: (channels, k, leds)
        if self.k > 0:
            self.dia_data_cpu = np.zeros(
                (self.channels, self.k, self.led_count), dtype=np.float32
            )

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
                        self.dia_data_cpu[
                            channel, unified_idx, :
                        ] = diagonal_data.astype(np.float32)

            # Create GPU version
            self.dia_data_gpu = cupy.asarray(
                self.dia_data_cpu
            )  # Shape: (channels, k, leds)

            # Calculate storage efficiency
            naive_elements = (
                len(set().union(*[dia.offsets for dia in channel_dia_matrices]))
                * self.channels
                * self.led_count
            )
            actual_elements = self.k * self.channels * self.led_count
            storage_efficiency = (
                actual_elements / naive_elements * 100 if naive_elements > 0 else 100
            )

            print(f"  Unified 3D DIA storage: shape {self.dia_data_cpu.shape}")
            print(
                f"  Storage efficiency: {self.k} / {len(set().union(*[dia.offsets for dia in channel_dia_matrices]))} bands = {storage_efficiency:.1f}%"
            )
            print(
                f"  Total stored elements: {actual_elements:,} vs dense {self.channels * self.led_count * self.led_count:,}"
            )

        else:
            self.dia_data_cpu = np.zeros(
                (self.channels, 0, self.led_count), dtype=np.float32
            )
            self.dia_data_gpu = cupy.asarray(self.dia_data_cpu)

        # Create legacy block diagonal format for backward compatibility
        self.channel_offsets = [
            channel * self.led_count for channel in range(self.channels)
        ]
        self.dia_matrix_cpu = sp.block_diag(channel_matrices, format="dia")
        self.dia_matrix_gpu = cusp.dia_matrix(self.dia_matrix_cpu)

        # Store overall metadata
        self.nnz = total_nnz
        total_elements = self.channels * self.led_count * self.led_count
        self.sparsity = self.nnz / total_elements * 100

        # Compute 3D DIA sparsity
        dia_3d_nnz = np.count_nonzero(self.dia_data_cpu)
        dia_3d_elements = self.channels * self.k * self.led_count
        dia_3d_sparsity = dia_3d_nnz / dia_3d_elements * 100

        # Compute bandwidth
        self.bandwidth = self._estimate_bandwidth(self.dia_matrix_cpu)

        print(f"  3D DIA structure: shape {self.dia_data_cpu.shape}")
        print(f"  3D DIA nnz: {dia_3d_nnz}, sparsity: {dia_3d_sparsity:.3f}%")
        print(f"  Block diagonal shape: {self.dia_matrix_cpu.shape}")
        print(f"  Total nnz: {self.nnz}, overall sparsity: {self.sparsity:.3f}%")
        print(f"  Estimated bandwidth: {self.bandwidth}")

        print(f"  True 3D DIA matrices built successfully!")

    def _compute_rcm_ordering(self, led_positions: np.ndarray) -> np.ndarray:
        """Compute RCM ordering based on LED spatial adjacency."""
        # Build adjacency graph using KD-tree for efficiency
        tree = cKDTree(led_positions)
        max_distance = self.crop_size * np.sqrt(2)  # Conservative distance
        pairs = tree.query_pairs(max_distance)

        # Build sparse adjacency matrix
        adjacency = sp.lil_matrix((self.led_count, self.led_count), dtype=bool)

        overlaps = 0
        for i, j in pairs:
            led1_x, led1_y = led_positions[i]
            led2_x, led2_y = led_positions[j]

            # Check actual crop region overlap
            x1_min, x1_max = led1_x - self.crop_size // 2, led1_x + self.crop_size // 2
            y1_min, y1_max = led1_y - self.crop_size // 2, led1_y + self.crop_size // 2
            x2_min, x2_max = led2_x - self.crop_size // 2, led2_x + self.crop_size // 2
            y2_min, y2_max = led2_y - self.crop_size // 2, led2_y + self.crop_size // 2

            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

            if x_overlap > 0 and y_overlap > 0:
                adjacency[i, j] = True
                adjacency[j, i] = True
                overlaps += 1

        print(f"    Found {overlaps} LED region overlaps")

        # Apply RCM algorithm
        adjacency_csr = adjacency.tocsr()
        rcm_order = reverse_cuthill_mckee(adjacency_csr, symmetric_mode=True)

        return rcm_order

    def _reorder_diffusion_matrix(
        self, A: sp.spmatrix, led_order: np.ndarray
    ) -> sp.spmatrix:
        """Reorder diffusion matrix columns according to LED ordering."""
        # Create column permutation for RGB channels
        col_permutation = []
        for led_idx in led_order:
            # Add R, G, B columns for this LED
            col_permutation.extend([led_idx * 3, led_idx * 3 + 1, led_idx * 3 + 2])

        # Reorder columns
        return A[:, col_permutation]

    def _estimate_bandwidth(self, dia_matrix: sp.dia_matrix) -> int:
        """Estimate bandwidth of DIA matrix."""
        if dia_matrix is None or len(dia_matrix.offsets) == 0:
            return 0

        return int(np.max(np.abs(dia_matrix.offsets)))

    def multiply_3d(
        self,
        led_values: np.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
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

        Returns:
            Result array (3, leds) in RCM order
        """
        if led_values.shape != (self.channels, self.led_count):
            raise ValueError(
                f"LED values should be shape ({self.channels}, {self.led_count}), "
                f"got {led_values.shape}"
            )

        # Check if unified 3D matrix is built
        if self.dia_data_gpu is None or self.dia_data_gpu.shape != (
            self.channels,
            self.k,
            self.led_count,
        ):
            raise RuntimeError(
                "Unified 3D DIA matrix not built. Call build_from_diffusion_matrix() first."
            )

        # Convert to GPU arrays
        if not isinstance(led_values, cupy.ndarray):
            led_values_gpu = cupy.asarray(led_values, dtype=cupy.float32)
        else:
            led_values_gpu = led_values.astype(cupy.float32)

        # Perform 3D DIA matrix-vector multiplication
        if use_custom_kernel and self.custom_kernel_basic is not None:
            # Use custom 3D DIA kernel
            kernel = (
                self.custom_kernel_optimized
                if optimized_kernel
                else self.custom_kernel_basic
            )
            result_gpu = kernel.multiply_3d(
                self.dia_data_gpu, self.dia_offsets, led_values_gpu
            )
        else:
            # Fallback to CPU or legacy block diagonal
            result_gpu = self._multiply_3d_fallback(led_values_gpu)

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
            band_data_all_channels = self.dia_data_gpu[
                :, band_idx, :
            ]  # Shape: (channels, leds)

            # DIA format: band_data[c,i] contains A[c,i,i+offset] for valid indices
            # Matrix multiplication: result[c,i] += A[c,i,i+offset] * led_values[c,i+offset]
            if offset >= 0:
                # Upper diagonal: A[c,i,i+offset] for i in [0, leds-offset)
                valid_i_range = self.led_count - offset
                if valid_i_range > 0:
                    # Vectorized across all channels: result[c,i] += band_data[c,i] * led_values[c,i+offset]
                    i_slice = slice(0, valid_i_range)
                    j_slice = slice(offset, offset + valid_i_range)
                    result_gpu[:, i_slice] += (
                        band_data_all_channels[:, i_slice] * led_values_gpu[:, j_slice]
                    )
            else:
                # Lower diagonal: A[c,i,i+offset] for i in [-offset, leds) where i+offset >= 0
                start_i = -offset
                valid_i_range = self.led_count - start_i
                if valid_i_range > 0:
                    # Vectorized across all channels: result[c,i] += band_data[c,i] * led_values[c,i+offset]
                    i_slice = slice(start_i, self.led_count)
                    j_slice = slice(0, valid_i_range)
                    result_gpu[:, i_slice] += (
                        band_data_all_channels[:, i_slice] * led_values_gpu[:, j_slice]
                    )

        return result_gpu

    def multiply(
        self,
        led_values: np.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
    ) -> np.ndarray:
        """
        Perform matrix-vector multiplication: (A^T)A @ led_values.

        This performs the einsum operation: ijk,ik->ij where:
        - i: channel (R, G, B)
        - j,k: LED indices
        - Input: led_values (3, leds)
        - Output: result (3, leds)

        Args:
            led_values: LED values array (3, leds) in RCM order
            use_custom_kernel: Whether to use custom DIA kernel
            optimized_kernel: Whether to use optimized custom kernel

        Returns:
            Result array (3, leds) in RCM order
        """
        if led_values.shape != (self.channels, self.led_count):
            raise ValueError(
                f"LED values should be shape ({self.channels}, {self.led_count}), "
                f"got {led_values.shape}"
            )

        # Check if matrix is built
        if self.dia_matrix_gpu is None:
            raise RuntimeError(
                "Diagonal A^T A matrix not built. Call build_from_diffusion_matrix() first."
            )

        # Convert to GPU arrays
        if not isinstance(led_values, cupy.ndarray):
            led_values_gpu = cupy.asarray(led_values, dtype=cupy.float32)
        else:
            led_values_gpu = led_values.astype(cupy.float32)

        # Flatten (3, leds) -> (3*leds,) for block diagonal multiplication
        led_values_flat = led_values_gpu.flatten()  # Shape: (3*leds,)

        # Single matrix multiplication: (3*leds, 3*leds) @ (3*leds,) -> (3*leds,)
        if use_custom_kernel and self.custom_kernel_basic is not None:
            # Use custom DIA kernel
            kernel = (
                self.custom_kernel_optimized
                if optimized_kernel
                else self.custom_kernel_basic
            )
            result_flat = kernel(self.dia_matrix_gpu, led_values_flat)
        else:
            # Use CuPy built-in DIA
            result_flat = self.dia_matrix_gpu @ led_values_flat

        # Reshape back to (3, leds)
        result_gpu = result_flat.reshape(self.channels, self.led_count)

        # Convert back to numpy if input was numpy
        if isinstance(led_values, np.ndarray):
            return cupy.asnumpy(result_gpu)
        else:
            return result_gpu

    def multiply_cpu(self, led_values: np.ndarray) -> np.ndarray:
        """
        CPU version of matrix-vector multiplication.

        Args:
            led_values: LED values array (3, leds) in RCM order

        Returns:
            Result array (3, leds) in RCM order
        """
        if led_values.shape != (self.channels, self.led_count):
            raise ValueError(
                f"LED values should be shape ({self.channels}, {self.led_count}), "
                f"got {led_values.shape}"
            )

        if self.dia_matrix_cpu is None:
            raise RuntimeError("Diagonal A^T A matrix not built.")

        led_values = led_values.astype(np.float32)

        # Flatten (3, leds) -> (3*leds,) for block diagonal multiplication
        led_values_flat = led_values.flatten()  # Shape: (3*leds,)

        # Single matrix multiplication: (3*leds, 3*leds) @ (3*leds,) -> (3*leds,)
        result_flat = self.dia_matrix_cpu @ led_values_flat

        # Reshape back to (3, leds)
        result = result_flat.reshape(self.channels, self.led_count)

        return result

    def g_ata_g_3d(
        self,
        gradient: np.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
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

        Returns:
            Result array (3,) - g^T (A^T A) g for each channel
        """
        if gradient.shape != (self.channels, self.led_count):
            raise ValueError(
                f"Gradient should be shape ({self.channels}, {self.led_count}), "
                f"got {gradient.shape}"
            )

        if self.dia_data_gpu is None or self.dia_data_gpu.shape != (
            self.channels,
            self.k,
            self.led_count,
        ):
            raise RuntimeError(
                "Unified 3D DIA matrix not built. Call build_from_diffusion_matrix() first."
            )

        # Convert to GPU arrays
        if not isinstance(gradient, cupy.ndarray):
            gradient_gpu = cupy.asarray(gradient, dtype=cupy.float32)
        else:
            gradient_gpu = gradient.astype(cupy.float32)

        # Compute (A^T A) @ g using unified 3D DIA
        if use_custom_kernel and self.custom_kernel_basic is not None:
            kernel = (
                self.custom_kernel_optimized
                if optimized_kernel
                else self.custom_kernel_basic
            )
            ata_g_gpu = kernel.multiply_3d(
                self.dia_data_gpu, self.dia_offsets, gradient_gpu
            )
        else:
            ata_g_gpu = self._multiply_3d_fallback(gradient_gpu)

        # Compute g^T @ (A^T A @ g) for each channel using vectorized operation: (channels,leds) * (channels,leds) -> (channels,)
        result_gpu = cupy.sum(gradient_gpu * ata_g_gpu, axis=1)  # Shape: (channels,)

        # Convert back to numpy if input was numpy
        if isinstance(gradient, np.ndarray):
            return cupy.asnumpy(result_gpu)
        else:
            return result_gpu

    def g_ata_g(
        self,
        gradient: np.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
    ) -> np.ndarray:
        """
        Compute g^T (A^T A) g for step size calculation in optimization.

        This performs the einsum operation: ij,ijk,ik->i where:
        - i: channel (R, G, B)
        - j,k: LED indices
        - Input: gradient g (3, leds) in RCM order
        - Output: result (3,) - one value per channel

        Args:
            gradient: Gradient array (3, leds) in RCM order
            use_custom_kernel: Whether to use custom DIA kernel
            optimized_kernel: Whether to use optimized custom kernel

        Returns:
            Result array (3,) - g^T (A^T A) g for each channel
        """
        if gradient.shape != (self.channels, self.led_count):
            raise ValueError(
                f"Gradient should be shape ({self.channels}, {self.led_count}), "
                f"got {gradient.shape}"
            )

        if self.dia_matrix_gpu is None:
            raise RuntimeError(
                "Diagonal A^T A matrix not built. Call build_from_diffusion_matrix() first."
            )

        # Convert to GPU arrays
        if not isinstance(gradient, cupy.ndarray):
            gradient_gpu = cupy.asarray(gradient, dtype=cupy.float32)
        else:
            gradient_gpu = gradient.astype(cupy.float32)

        # Flatten (3, leds) -> (3*leds,) for block diagonal multiplication
        gradient_flat = gradient_gpu.flatten()  # Shape: (3*leds,)

        # Compute (A^T A) @ g: (3*leds, 3*leds) @ (3*leds,) -> (3*leds,)
        if use_custom_kernel and self.custom_kernel_basic is not None:
            kernel = (
                self.custom_kernel_optimized
                if optimized_kernel
                else self.custom_kernel_basic
            )
            ata_g_flat = kernel(self.dia_matrix_gpu, gradient_flat)
        else:
            ata_g_flat = self.dia_matrix_gpu @ gradient_flat

        # Reshape back to (3, leds) for channel-wise dot products
        ata_g = ata_g_flat.reshape(self.channels, self.led_count)

        # Compute g^T @ (A^T A @ g) for each channel: (leds,) @ (leds,) -> scalar
        result_gpu = cupy.zeros(self.channels, dtype=cupy.float32)
        for channel in range(self.channels):
            result_gpu[channel] = cupy.dot(gradient_gpu[channel], ata_g[channel])

        # Convert back to numpy if input was numpy
        if isinstance(gradient, np.ndarray):
            return cupy.asnumpy(result_gpu)
        else:
            return result_gpu

    def g_ata_g_cpu(self, gradient: np.ndarray) -> np.ndarray:
        """
        CPU version of g^T (A^T A) g computation.

        Args:
            gradient: Gradient array (3, leds) in RCM order

        Returns:
            Result array (3,) - g^T (A^T A) g for each channel
        """
        if gradient.shape != (self.channels, self.led_count):
            raise ValueError(
                f"Gradient should be shape ({self.channels}, {self.led_count}), "
                f"got {gradient.shape}"
            )

        if self.dia_matrix_cpu is None:
            raise RuntimeError("Diagonal A^T A matrix not built.")

        gradient = gradient.astype(np.float32)

        # Flatten (3, leds) -> (3*leds,) for block diagonal multiplication
        gradient_flat = gradient.flatten()  # Shape: (3*leds,)

        # Compute (A^T A) @ g: (3*leds, 3*leds) @ (3*leds,) -> (3*leds,)
        ata_g_flat = self.dia_matrix_cpu @ gradient_flat

        # Reshape back to (3, leds) for channel-wise dot products
        ata_g = ata_g_flat.reshape(self.channels, self.led_count)

        # Compute g^T @ (A^T A @ g) for each channel: (leds,) @ (leds,) -> scalar
        result = np.zeros(self.channels, dtype=np.float32)
        for channel in range(self.channels):
            result[channel] = np.dot(gradient[channel], ata_g[channel])

        return result

    def reorder_led_values_to_rcm(self, led_values_original: np.ndarray) -> np.ndarray:
        """
        Reorder LED values from original ordering to RCM ordering.

        Args:
            led_values_original: LED values in original order (3, leds)

        Returns:
            LED values in RCM order (3, leds)
        """
        if self.led_order is None:
            return led_values_original.copy()

        # Reorder each channel
        result = np.zeros_like(led_values_original)
        for channel in range(self.channels):
            result[channel] = led_values_original[channel, self.led_order]

        return result

    def reorder_led_values_from_rcm(self, led_values_rcm: np.ndarray) -> np.ndarray:
        """
        Reorder LED values from RCM ordering back to original ordering.

        Args:
            led_values_rcm: LED values in RCM order (3, leds)

        Returns:
            LED values in original order (3, leds)
        """
        if self.inverse_led_order is None:
            return led_values_rcm.copy()

        # Reorder each channel back
        result = np.zeros_like(led_values_rcm)
        for channel in range(self.channels):
            result[channel] = led_values_rcm[channel, self.inverse_led_order]

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary containing all necessary data
        """
        # Convert block diagonal DIA matrix to dict format (legacy)
        if self.dia_matrix_cpu is not None:
            dia_dict = {
                "data": self.dia_matrix_cpu.data,
                "offsets": self.dia_matrix_cpu.offsets,
                "shape": self.dia_matrix_cpu.shape,
            }
        else:
            dia_dict = None

        return {
            "led_count": self.led_count,
            "crop_size": self.crop_size,
            "channels": self.channels,
            # Unified 3D DIA format (current)
            "dia_data_3d": self.dia_data_cpu,
            "dia_offsets_3d": self.dia_offsets,
            "k": self.k,
            # Legacy block diagonal format
            "dia_matrix": dia_dict,
            "channel_offsets": self.channel_offsets,
            # Common metadata
            "led_order": self.led_order,
            "inverse_led_order": self.inverse_led_order,
            "bandwidth": self.bandwidth,
            "sparsity": self.sparsity,
            "nnz": self.nnz,
            "channel_nnz": self.channel_nnz,
            "version": "5.0",  # Updated for unified 3D DIA storage (channels, k, leds)
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
        # Create instance
        instance = cls(data["led_count"], data["crop_size"])

        # Restore metadata
        instance.channels = data["channels"]
        instance.led_order = data["led_order"]
        instance.inverse_led_order = data["inverse_led_order"]
        instance.bandwidth = data["bandwidth"]
        instance.sparsity = data["sparsity"]
        instance.nnz = data["nnz"]

        # Handle version compatibility
        version = data.get("version", "1.0")

        if version == "5.0":
            # Unified 3D DIA format (current)
            instance.dia_data_cpu = data["dia_data_3d"]
            instance.dia_offsets = data["dia_offsets_3d"]
            instance.k = data["k"]
            instance.channel_offsets = data["channel_offsets"]
            instance.channel_nnz = data["channel_nnz"]

            # Create GPU version
            if instance.dia_data_cpu is not None:
                instance.dia_data_gpu = cupy.asarray(instance.dia_data_cpu)
            else:
                instance.dia_data_gpu = None

            # Restore block diagonal DIA matrix
            dia_dict = data["dia_matrix"]
            if dia_dict is not None:
                dia_cpu = sp.dia_matrix(
                    (dia_dict["data"], dia_dict["offsets"]), shape=dia_dict["shape"]
                )
                instance.dia_matrix_cpu = dia_cpu
                instance.dia_matrix_gpu = cusp.dia_matrix(dia_cpu)
        elif version == "2.0" or version == "3.0" or version == "4.0":
            # Legacy formats - require rebuild
            instance.channel_offsets = data.get("channel_offsets")
            instance.channel_nnz = data.get("channel_nnz")

            # Restore block diagonal DIA matrix if available
            dia_dict = data.get("dia_matrix")
            if dia_dict is not None:
                dia_cpu = sp.dia_matrix(
                    (dia_dict["data"], dia_dict["offsets"]), shape=dia_dict["shape"]
                )
                instance.dia_matrix_cpu = dia_cpu
                instance.dia_matrix_gpu = cusp.dia_matrix(dia_cpu)

            # Initialize empty unified format (will need to be rebuilt)
            instance.dia_data_cpu = None
            instance.dia_data_gpu = None
            instance.dia_offsets = None
            instance.k = None
        else:
            # Legacy format with separate channel matrices - convert to new format
            raise ValueError(
                f"Legacy format version {version} not supported. Please rebuild matrices."
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
        data = {
            key: data_npz[key].item() if data_npz[key].ndim == 0 else data_npz[key]
            for key in data_npz.keys()
        }

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
            "rcm_applied": self.led_order is not None,
            "custom_kernel_available": CUSTOM_KERNEL_AVAILABLE,
            "matrix_built": self.dia_matrix_cpu is not None,
            "unified_storage_built": unified_storage_built,
            "unified_k": self.k,
            "unified_storage_shape": self.dia_data_cpu.shape
            if self.dia_data_cpu is not None
            else None,
            "storage_format": "unified_3d_dia_v5",
        }

    def benchmark(self, num_trials: int = 50, num_warmup: int = 10) -> Dict[str, float]:
        """
        Benchmark multiplication performance.

        Args:
            num_trials: Number of timing trials
            num_warmup: Number of warmup iterations

        Returns:
            Timing results dictionary
        """
        if self.dia_matrix_gpu is None:
            raise RuntimeError("Matrix not built")

        # Create test LED values
        test_values_cpu = [
            np.random.randn(self.channels, self.led_count).astype(np.float32)
            for _ in range(num_trials + num_warmup)
        ]
        test_values_gpu = [cupy.asarray(x) for x in test_values_cpu]

        results = {}

        # CPU timing (multiply)
        for i in range(num_warmup):
            _ = self.multiply_cpu(test_values_cpu[i])

        times = []
        for i in range(num_warmup, num_warmup + num_trials):
            start = time.perf_counter()
            _ = self.multiply_cpu(test_values_cpu[i])
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        results["cpu"] = np.mean(times)

        # CPU timing (g_ata_g)
        for i in range(num_warmup):
            _ = self.g_ata_g_cpu(test_values_cpu[i])

        times = []
        for i in range(num_warmup, num_warmup + num_trials):
            start = time.perf_counter()
            _ = self.g_ata_g_cpu(test_values_cpu[i])
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        results["cpu_g_ata_g"] = np.mean(times)

        # GPU timing (built-in multiply)
        for i in range(num_warmup):
            _ = self.multiply(test_values_gpu[i], use_custom_kernel=False)
        cupy.cuda.Stream.null.synchronize()

        times = []
        for i in range(num_warmup, num_warmup + num_trials):
            start_event = cupy.cuda.Event()
            end_event = cupy.cuda.Event()

            start_event.record()
            _ = self.multiply(test_values_gpu[i], use_custom_kernel=False)
            end_event.record()
            end_event.synchronize()

            times.append(cupy.cuda.get_elapsed_time(start_event, end_event))

        results["gpu_builtin"] = np.mean(times)

        # GPU timing (built-in g_ata_g)
        for i in range(num_warmup):
            _ = self.g_ata_g(test_values_gpu[i], use_custom_kernel=False)
        cupy.cuda.Stream.null.synchronize()

        times = []
        for i in range(num_warmup, num_warmup + num_trials):
            start_event = cupy.cuda.Event()
            end_event = cupy.cuda.Event()

            start_event.record()
            _ = self.g_ata_g(test_values_gpu[i], use_custom_kernel=False)
            end_event.record()
            end_event.synchronize()

            times.append(cupy.cuda.get_elapsed_time(start_event, end_event))

        results["gpu_builtin_g_ata_g"] = np.mean(times)

        # GPU timing (custom kernel)
        if CUSTOM_KERNEL_AVAILABLE:
            for i in range(num_warmup):
                _ = self.multiply(test_values_gpu[i], use_custom_kernel=True)
            cupy.cuda.Stream.null.synchronize()

            times = []
            for i in range(num_warmup, num_warmup + num_trials):
                start_event = cupy.cuda.Event()
                end_event = cupy.cuda.Event()

                start_event.record()
                _ = self.multiply(test_values_gpu[i], use_custom_kernel=True)
                end_event.record()
                end_event.synchronize()

                times.append(cupy.cuda.get_elapsed_time(start_event, end_event))

            results["gpu_custom"] = np.mean(times)

            # GPU timing (custom kernel g_ata_g)
            for i in range(num_warmup):
                _ = self.g_ata_g(test_values_gpu[i], use_custom_kernel=True)
            cupy.cuda.Stream.null.synchronize()

            times = []
            for i in range(num_warmup, num_warmup + num_trials):
                start_event = cupy.cuda.Event()
                end_event = cupy.cuda.Event()

                start_event.record()
                _ = self.g_ata_g(test_values_gpu[i], use_custom_kernel=True)
                end_event.record()
                end_event.synchronize()

                times.append(cupy.cuda.get_elapsed_time(start_event, end_event))

            results["gpu_custom_g_ata_g"] = np.mean(times)

        return results
