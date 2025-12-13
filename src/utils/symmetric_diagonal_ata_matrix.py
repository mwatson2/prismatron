#!/usr/bin/env python3
"""
Symmetric Diagonal A^T A Matrix utility class for LED optimization.

This class implements an optimized version of DiagonalATAMatrix that takes advantage
of the symmetric structure of A^T A matrices by storing only the main diagonal
and upper diagonals, reading lower diagonals from the symmetric positions.
"""

from typing import Optional

try:
    import cupy
except ImportError:
    # Fallback for systems without CUDA
    import numpy as cupy

import numpy as np

from .base_ata_matrix import BaseATAMatrix


def _extract_diagonal(matrices: np.ndarray, offset: int) -> np.ndarray:
    """
    Extract a diagonal from 3D matrices using straightforward indexing.

    For a matrix A, diagonal at offset k contains elements A[i, i+k] where i+k < matrix_size.
    For symmetric matrices, we only need upper diagonals (offset >= 0).

    Args:
        matrices: 3D array of shape (channels, led_count, led_count)
        offset: Diagonal offset (0 = main diagonal, 1 = first upper diagonal, etc.)

    Returns:
        Diagonal elements of shape (channels, diagonal_length)
    """
    channels, led_count, _ = matrices.shape
    diagonal_length = led_count - offset

    if diagonal_length <= 0:
        return np.zeros((channels, 0), dtype=matrices.dtype)

    # Extract diagonal elements for all channels using fancy indexing
    row_indices = np.arange(diagonal_length)
    col_indices = row_indices + offset

    diagonal = matrices[:, row_indices, col_indices]  # Shape: (channels, diagonal_length)

    return diagonal


# Import symmetric custom DIA kernels
try:
    from .kernels.symmetric_dia_kernel import SymmetricCustomDIA3DMatVec

    SYMMETRIC_KERNEL_AVAILABLE = True
except ImportError:
    SYMMETRIC_KERNEL_AVAILABLE = False
    print("Warning: Symmetric DIA kernels not available")


class SymmetricDiagonalATAMatrix(BaseATAMatrix):
    """
    Optimized diagonal A^T A matrix implementation using symmetric storage.

    Takes advantage of the fact that A^T A matrices are symmetric by storing
    only the main diagonal and upper diagonals, resulting in ~50% memory savings.
    Lower diagonals are accessed via symmetry: A[i,j] = A[j,i].

    Storage Format:
    - dia_data_gpu: (channels, k_upper, leds) - only main + upper diagonals
    - dia_offsets_upper: (k_upper,) - only non-negative offsets
    """

    def __init__(
        self,
        led_count: int,
        crop_size: int = 64,
        output_dtype: Optional[cupy.dtype] = None,
    ):
        """
        Initialize symmetric diagonal A^T A matrix container.

        Args:
            led_count: Number of LEDs
            crop_size: Crop size used for LED regions (affects adjacency)
            output_dtype: Data type for output tensors (cupy.float32 only for symmetric version)
        """
        self.led_count = led_count
        self.crop_size = crop_size
        self.channels = 3  # RGB

        # Symmetric version only supports FP32 for simplicity
        self.output_dtype = cupy.float32
        if output_dtype is not None and output_dtype != cupy.float32:
            raise ValueError("SymmetricDiagonalATAMatrix only supports cupy.float32 output_dtype")

        # Storage for symmetric 3D DIA format - shape (channels, k_upper, leds)
        # Only stores main diagonal (offset=0) and upper diagonals (offset>0)
        self.dia_data_gpu = None  # Shape: (channels, k_upper, leds) - symmetric storage
        self.dia_offsets_upper = None  # Shape: (k_upper,) - only non-negative offsets
        self.dia_offsets_upper_gpu = None  # CuPy version of offsets, cached for performance
        self.k_upper = None  # Number of upper diagonal bands (including main diagonal)

        # Symmetric kernel instances (FP32 only)
        self.symmetric_kernel_basic = None
        self.symmetric_kernel_optimized = None

        # Metadata
        self.bandwidth = None
        self.sparsity = None
        self.nnz = None
        self.original_k = None  # Original number of diagonals before symmetric optimization

        # Initialize symmetric kernels if available
        if SYMMETRIC_KERNEL_AVAILABLE:
            self.symmetric_kernel_basic = SymmetricCustomDIA3DMatVec(use_optimized=False)
            self.symmetric_kernel_optimized = SymmetricCustomDIA3DMatVec(use_optimized=True)

    def _validate_input_tensor_layout(self, tensor: cupy.ndarray, tensor_name: str) -> None:
        """
        Validate tensor memory layout for optimal kernel performance.

        Args:
            tensor: Input tensor to validate
            tensor_name: Name for error messages

        Raises:
            ValueError: If tensor layout is incompatible with kernels
        """
        # Check if tensor is a CuPy array
        if isinstance(tensor, cupy.ndarray):
            # Validate C-contiguous layout
            if not tensor.flags.c_contiguous:
                stride_info = f"strides={tensor.strides}, shape={tensor.shape}"
                layout = "F-contiguous" if tensor.flags.f_contiguous else "non-contiguous"

                raise ValueError(
                    f"Symmetric DIA kernel requires C-contiguous {tensor_name} tensor for optimal performance. "
                    f"Got {layout} tensor with {stride_info}. "
                    f"Use cp.ascontiguousarray({tensor_name}) to fix layout issues."
                )

            # Validate expected stride pattern for channels-first layout
            if len(tensor.shape) == 2:  # (channels, leds)
                expected_channel_stride = tensor.shape[1] * tensor.itemsize
                expected_led_stride = tensor.itemsize

                if tensor.strides != (expected_channel_stride, expected_led_stride):
                    raise ValueError(
                        f"Symmetric DIA kernel expects channels-first stride pattern for {tensor_name}. "
                        f"Expected strides ({expected_channel_stride}, {expected_led_stride}), "
                        f"got {tensor.strides}."
                    )
        else:
            raise TypeError(f"{tensor_name} must be cupy.ndarray, got {type(tensor)}")

    @staticmethod
    def from_dense(
        dense_ata_matrices: np.ndarray, led_count: int, significance_threshold: float = 0.01, crop_size: int = 64
    ) -> "SymmetricDiagonalATAMatrix":
        """
        Create SymmetricDiagonalATAMatrix directly from dense ATA matrices.

        This method extracts diagonals directly from dense matrices using a simple
        and straightforward approach, avoiding scipy sparse matrix conversion issues.

        Args:
            dense_ata_matrices: Dense ATA matrices of shape (channels, led_count, led_count)
            led_count: Number of LEDs
            significance_threshold: Threshold for filtering small values (relative to max value)
            crop_size: Crop size for metadata

        Returns:
            SymmetricDiagonalATAMatrix instance
        """
        print(f"Creating SymmetricDiagonalATAMatrix from dense matrices ({led_count} LEDs)...")

        if dense_ata_matrices.shape != (3, led_count, led_count):
            raise ValueError(
                f"Expected dense_ata_matrices shape (3, {led_count}, {led_count}), got {dense_ata_matrices.shape}"
            )

        # Apply significance threshold to all channels
        thresholded_matrices = np.zeros_like(dense_ata_matrices)
        for c in range(3):
            max_val = np.abs(dense_ata_matrices[c]).max()
            threshold = max_val * significance_threshold
            thresholded_matrices[c] = np.where(np.abs(dense_ata_matrices[c]) >= threshold, dense_ata_matrices[c], 0.0)
            print(f"  Channel {c}: max={max_val:.6f}, threshold={threshold:.6f}")

        # Find the maximum diagonal offset by checking from the top-right corner
        # Start from the maximum possible offset (led_count-1) and work down
        max_offset = -1

        for offset in range(led_count - 1, -1, -1):  # From top-right to main diagonal
            diagonal = _extract_diagonal(thresholded_matrices, offset)  # Shape: (3, diagonal_length)

            # Check if any element in any channel is non-zero
            if np.any(diagonal != 0):
                max_offset = offset
                break

        if max_offset == -1:
            # No non-zero diagonals found - create minimal matrix with just main diagonal
            max_offset = 0
            print("  Warning: No significant diagonals found, using only main diagonal")

        print(f"  Maximum significant diagonal offset: {max_offset}")

        # Create list of diagonal offsets from 0 to max_offset
        dia_offsets_upper = np.arange(0, max_offset + 1, dtype=np.int32)
        k_upper = len(dia_offsets_upper)

        print(f"  Storing {k_upper} upper diagonals (offsets 0 to {max_offset})")

        # Extract all diagonal data
        dia_data_cpu = np.zeros((3, k_upper, led_count), dtype=np.float32)

        total_nnz = 0
        for i, offset in enumerate(dia_offsets_upper):
            diagonal = _extract_diagonal(thresholded_matrices, offset)  # Shape: (3, diagonal_length)

            # Store in DIA format: dia_data[c, i, j] represents matrix element (j-offset, j)
            # For offset > 0, we need to shift the diagonal data to align with the column indices
            for c in range(3):
                diagonal_length = led_count - offset
                if diagonal_length > 0:
                    # In DIA format, diagonal data starts at column index = offset
                    dia_data_cpu[c, i, offset : offset + diagonal_length] = diagonal[c]
                    total_nnz += np.sum(diagonal[c] != 0)

        # Create symmetric instance
        symmetric = SymmetricDiagonalATAMatrix(led_count=led_count, crop_size=crop_size, output_dtype=cupy.float32)

        # Set all the data
        symmetric.k_upper = k_upper
        symmetric.dia_offsets_upper = dia_offsets_upper
        symmetric.bandwidth = max_offset
        symmetric.nnz = total_nnz
        symmetric.sparsity = (total_nnz / (3 * led_count * led_count)) * 100
        symmetric.original_k = k_upper * 2 - 1  # Estimate full matrix diagonal count

        # Convert to GPU
        symmetric.dia_data_gpu = cupy.asarray(dia_data_cpu, dtype=cupy.float32)
        symmetric.dia_offsets_upper_gpu = cupy.asarray(dia_offsets_upper, dtype=cupy.int32)

        print("  Created symmetric DIA matrix:")
        print(f"    Upper diagonals: {k_upper}")
        print(f"    Bandwidth: {symmetric.bandwidth}")
        print(f"    Total non-zeros: {total_nnz:,}")
        print(f"    Sparsity: {symmetric.sparsity:.2f}%")
        print(f"    GPU memory: {symmetric.dia_data_gpu.nbytes / (1024*1024):.1f} MB")

        return symmetric

    @staticmethod
    def from_diagonal_ata_matrix(regular_matrix) -> "SymmetricDiagonalATAMatrix":
        """
        Create symmetric version from regular DiagonalATAMatrix.

        Args:
            regular_matrix: Instance of DiagonalATAMatrix

        Returns:
            SymmetricDiagonalATAMatrix with symmetric storage
        """
        print("Converting DiagonalATAMatrix to symmetric storage...")

        # Create symmetric instance
        symmetric = SymmetricDiagonalATAMatrix(
            led_count=regular_matrix.led_count,
            crop_size=regular_matrix.crop_size,
            output_dtype=cupy.float32,  # Symmetric version is FP32 only
        )

        # Copy metadata
        symmetric.bandwidth = regular_matrix.bandwidth
        symmetric.sparsity = regular_matrix.sparsity
        symmetric.nnz = regular_matrix.nnz
        symmetric.original_k = regular_matrix.k

        # Extract upper diagonals (offset >= 0) from regular matrix
        if regular_matrix.dia_offsets is None or regular_matrix.dia_data_cpu is None:
            raise ValueError("Regular matrix not built yet. Call build_from_diffusion_matrix() first.")

        # Find upper diagonal indices (offset >= 0)
        upper_mask = regular_matrix.dia_offsets >= 0
        upper_indices = np.where(upper_mask)[0]

        if len(upper_indices) == 0:
            raise ValueError("No upper diagonals found in regular matrix")

        # Extract upper diagonals
        offsets_upper = regular_matrix.dia_offsets[upper_indices].copy()
        symmetric.dia_offsets_upper = offsets_upper
        symmetric.k_upper = len(offsets_upper)

        print(f"  Original diagonals: {regular_matrix.k}")
        print(f"  Upper diagonals (including main): {symmetric.k_upper}")
        print(f"  Memory reduction: {symmetric.k_upper / regular_matrix.k * 100:.1f}% of original")

        # Extract upper diagonal data: (channels, k_upper, leds)
        symmetric_data_cpu = regular_matrix.dia_data_cpu[:, upper_indices, :].copy()

        # Convert to GPU and store
        symmetric.dia_data_gpu = cupy.asarray(symmetric_data_cpu, dtype=cupy.float32)
        symmetric.dia_offsets_upper_gpu = cupy.asarray(symmetric.dia_offsets_upper, dtype=cupy.int32)

        print(f"  Symmetric storage shape: {symmetric.dia_data_gpu.shape}")
        print(f"  GPU memory usage: {symmetric.dia_data_gpu.nbytes / 1024**2:.1f} MB")
        print("  Symmetric conversion completed!")

        return symmetric

    def multiply_3d(
        self,
        led_values: cupy.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
        output_dtype: Optional[cupy.dtype] = None,
        debug_logging: bool = False,
    ) -> cupy.ndarray:
        """
        Perform symmetric 3D DIA matrix-vector multiplication: (A^T)A @ led_values.

        Uses symmetric storage and kernels for ~50% memory bandwidth reduction.

        Args:
            led_values: LED values array (3, leds)
            use_custom_kernel: Whether to use custom symmetric kernels
            optimized_kernel: Whether to use optimized kernel variant
            output_dtype: Desired output data type (must be cupy.float32)
            debug_logging: Enable detailed logging

        Returns:
            Result array (3, leds)
        """
        if led_values.shape != (self.channels, self.led_count):
            raise ValueError(f"LED values should be shape ({self.channels}, {self.led_count}), got {led_values.shape}")

        # Validate memory layout
        self._validate_input_tensor_layout(led_values, "led_values")

        # Check if symmetric matrix is built
        if self.dia_data_gpu is None or self.dia_offsets_upper_gpu is None:
            raise RuntimeError("Symmetric matrix not built. Call from_diagonal_ata_matrix() first.")

        # Validate output dtype (symmetric version is FP32 only)
        if output_dtype is not None and output_dtype != cupy.float32:
            raise ValueError("SymmetricDiagonalATAMatrix only supports cupy.float32 output")

        # Ensure input is cupy array of correct dtype
        if not isinstance(led_values, cupy.ndarray):
            raise TypeError("led_values must be a cupy.ndarray")

        if led_values.dtype != cupy.float32:
            raise TypeError("led_values must be cupy.float32 for symmetric implementation")

        # Perform symmetric 3D DIA matrix-vector multiplication
        if use_custom_kernel and SYMMETRIC_KERNEL_AVAILABLE:
            if optimized_kernel:
                if debug_logging:
                    print("Symmetric multiply_3d: Using OPTIMIZED SymmetricCustomDIA3DMatVec kernel")
                if self.symmetric_kernel_optimized is None:
                    self.symmetric_kernel_optimized = SymmetricCustomDIA3DMatVec(use_optimized=True)
                result_gpu = self.symmetric_kernel_optimized(
                    self.dia_data_gpu,  # Shape: (channels, k_upper, leds)
                    self.dia_offsets_upper_gpu,  # Shape: (k_upper,)
                    led_values,  # Shape: (channels, leds)
                )
            else:
                if debug_logging:
                    print("Symmetric multiply_3d: Using BASIC SymmetricCustomDIA3DMatVec kernel")
                if self.symmetric_kernel_basic is None:
                    self.symmetric_kernel_basic = SymmetricCustomDIA3DMatVec(use_optimized=False)
                result_gpu = self.symmetric_kernel_basic(
                    self.dia_data_gpu,  # Shape: (channels, k_upper, leds)
                    self.dia_offsets_upper_gpu,  # Shape: (k_upper,)
                    led_values,  # Shape: (channels, leds)
                )
        else:
            # Use fallback implementation
            if debug_logging:
                print("Symmetric multiply_3d: Using FALLBACK implementation")
            result_gpu = self._multiply_3d_symmetric_fallback(led_values)

        return result_gpu

    def g_ata_g_3d(
        self,
        gradient: cupy.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
        output_dtype: Optional[cupy.dtype] = None,
    ) -> cupy.ndarray:
        """
        Compute g^T (A^T A) g for step size calculation using symmetric storage.

        Args:
            gradient: Gradient array (3, leds)
            use_custom_kernel: Whether to use custom symmetric kernels
            optimized_kernel: Whether to use optimized kernel variant
            output_dtype: Desired output data type (must be cupy.float32)

        Returns:
            Result array (3,) - one value per channel
        """
        if gradient.shape != (self.channels, self.led_count):
            raise ValueError(f"Gradient should be shape ({self.channels}, {self.led_count}), got {gradient.shape}")

        # Validate memory layout
        self._validate_input_tensor_layout(gradient, "gradient")

        # Check if symmetric matrix is built
        if self.dia_data_gpu is None or self.dia_offsets_upper_gpu is None:
            raise RuntimeError("Symmetric matrix not built. Call from_diagonal_ata_matrix() first.")

        # Validate output dtype
        if output_dtype is not None and output_dtype != cupy.float32:
            raise ValueError("SymmetricDiagonalATAMatrix only supports cupy.float32 output")

        # Ensure input is cupy array of correct dtype
        if not isinstance(gradient, cupy.ndarray):
            gradient_gpu = cupy.asarray(gradient, dtype=cupy.float32)
        else:
            if gradient.dtype != cupy.float32:
                gradient_gpu = gradient.astype(cupy.float32)
            else:
                gradient_gpu = gradient

        # Compute (A^T A) @ g using symmetric kernels
        if use_custom_kernel and SYMMETRIC_KERNEL_AVAILABLE:
            if optimized_kernel:
                if self.symmetric_kernel_optimized is None:
                    self.symmetric_kernel_optimized = SymmetricCustomDIA3DMatVec(use_optimized=True)
                ata_g_gpu = self.symmetric_kernel_optimized(
                    self.dia_data_gpu,  # Shape: (channels, k_upper, leds)
                    self.dia_offsets_upper_gpu,  # Shape: (k_upper,)
                    gradient_gpu,  # Shape: (channels, leds)
                )
            else:
                if self.symmetric_kernel_basic is None:
                    self.symmetric_kernel_basic = SymmetricCustomDIA3DMatVec(use_optimized=False)
                ata_g_gpu = self.symmetric_kernel_basic(
                    self.dia_data_gpu,  # Shape: (channels, k_upper, leds)
                    self.dia_offsets_upper_gpu,  # Shape: (k_upper,)
                    gradient_gpu,  # Shape: (channels, leds)
                )
        else:
            # Use fallback implementation
            ata_g_gpu = self._multiply_3d_symmetric_fallback(gradient_gpu)

        # Compute g^T @ (A^T A @ g) for each channel using vectorized operation:
        # (channels,leds) * (channels,leds) -> (channels,)
        result_gpu = cupy.sum(gradient_gpu * ata_g_gpu, axis=1)  # Shape: (channels,)

        return result_gpu

    def _multiply_3d_symmetric_fallback(self, led_values_gpu: cupy.ndarray) -> cupy.ndarray:
        """
        Symmetric 3D DIA multiplication fallback using only upper diagonal storage.

        Args:
            led_values_gpu: LED values (channels, leds) on GPU

        Returns:
            Result (channels, leds) on GPU
        """
        result_gpu = cupy.zeros_like(led_values_gpu)  # Shape: (channels, leds)

        if self.k_upper == 0:
            return result_gpu

        # Symmetric 3D DIA multiplication - process all channels
        for band_idx in range(self.k_upper):
            offset = int(self.dia_offsets_upper[band_idx])  # Non-negative offset

            # Get diagonal data for all channels: shape (channels, leds)
            band_data_all_channels = self.dia_data_gpu[:, band_idx, :]  # Shape: (channels, leds)

            # DIA format: band_data[j] contains A[i,j] where i = j - offset

            # Process each stored diagonal element
            for j in range(self.led_count):
                if (
                    band_data_all_channels[0, j] != 0
                    or band_data_all_channels[1, j] != 0
                    or band_data_all_channels[2, j] != 0
                ):
                    i = j - offset  # Row index from DIA format

                    if 0 <= i < self.led_count:
                        # Direct contribution: A[i,j] * x[j] -> result[i]
                        result_gpu[:, i] += band_data_all_channels[:, j] * led_values_gpu[:, j]

                        # Symmetric contribution: A[j,i] * x[i] -> result[j] (since A[j,i] = A[i,j])
                        if offset > 0:  # Skip main diagonal to avoid double counting
                            result_gpu[:, j] += band_data_all_channels[:, j] * led_values_gpu[:, i]

        return result_gpu

    def multiply_batch_3d(
        self,
        led_values_batch: cupy.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
        output_dtype: Optional[cupy.dtype] = None,
        debug_logging: bool = False,
    ) -> cupy.ndarray:
        """
        Perform batched symmetric 3D DIA matrix-vector multiplication: (A^T)A @ led_values for multiple frames.

        Processes a batch of frames simultaneously, computing (A^T A) @ x for each frame.

        Args:
            led_values_batch: Batch of LED values (batch_size, channels, leds)
            use_custom_kernel: Whether to use custom symmetric kernels
            optimized_kernel: Whether to use optimized kernel variant
            output_dtype: Desired output data type (must be cupy.float32)
            debug_logging: Enable detailed logging

        Returns:
            Result array (batch_size, channels, leds)
        """
        batch_size = led_values_batch.shape[0]

        if led_values_batch.shape[1:] != (self.channels, self.led_count):
            raise ValueError(
                f"LED values batch should be shape (batch_size, {self.channels}, {self.led_count}), "
                f"got {led_values_batch.shape}"
            )

        # Process each frame through the matrix
        result_batch = cupy.zeros_like(led_values_batch)

        for frame_idx in range(batch_size):
            # Get LED values for this frame: (channels, leds)
            led_frame = led_values_batch[frame_idx]

            # Apply matrix multiplication using existing 3D method
            result_batch[frame_idx] = self.multiply_3d(
                led_frame,
                use_custom_kernel=use_custom_kernel,
                optimized_kernel=optimized_kernel,
                output_dtype=output_dtype,
                debug_logging=debug_logging,
            )

        return result_batch

    def g_ata_g_batch_3d(
        self,
        gradient_batch: cupy.ndarray,
        use_custom_kernel: bool = True,
        optimized_kernel: bool = False,
        output_dtype: Optional[cupy.dtype] = None,
    ) -> cupy.ndarray:
        """
        Compute g^T (A^T A) g for a batch of gradients using symmetric storage.

        Args:
            gradient_batch: Batch of gradient arrays (batch_size, channels, leds)
            use_custom_kernel: Whether to use custom symmetric kernels
            optimized_kernel: Whether to use optimized kernel variant
            output_dtype: Desired output data type (must be cupy.float32)

        Returns:
            Result array (batch_size, channels) - one value per channel per frame
        """
        batch_size = gradient_batch.shape[0]

        if gradient_batch.shape[1:] != (self.channels, self.led_count):
            raise ValueError(
                f"Gradient batch should be shape (batch_size, {self.channels}, {self.led_count}), "
                f"got {gradient_batch.shape}"
            )

        # Process each frame
        result_batch = cupy.zeros((batch_size, self.channels), dtype=cupy.float32)

        for frame_idx in range(batch_size):
            # Get gradient for this frame: (channels, leds)
            gradient_frame = gradient_batch[frame_idx]

            # Compute g^T @ (A^T A) @ g using existing method
            result_batch[frame_idx] = self.g_ata_g_3d(
                gradient_frame,
                use_custom_kernel=use_custom_kernel,
                optimized_kernel=optimized_kernel,
                output_dtype=output_dtype,
            )

        return result_batch

    def get_info(self):
        """Get summary information about the symmetric matrix."""
        return {
            "led_count": self.led_count,
            "crop_size": self.crop_size,
            "channels": self.channels,
            "bandwidth": self.bandwidth,
            "sparsity": self.sparsity,
            "nnz": self.nnz,
            "output_dtype": str(self.output_dtype),
            "symmetric_storage": True,
            "original_k": self.original_k,
            "k_upper": self.k_upper,
            "memory_reduction": f"{self.k_upper / self.original_k * 100:.1f}%" if self.original_k else None,
            "storage_shape": self.dia_data_gpu.shape if self.dia_data_gpu is not None else None,
            "kernel_available": SYMMETRIC_KERNEL_AVAILABLE,
        }

    def benchmark_3d(self, num_trials: int = 50, num_warmup: int = 10):
        """
        Benchmark symmetric 3D DIA multiplication performance.

        Args:
            num_trials: Number of timing trials
            num_warmup: Number of warmup iterations

        Returns:
            Timing results dictionary
        """
        if self.dia_data_gpu is None:
            raise RuntimeError("Symmetric matrix not built")

        # Create test LED values
        test_values_cpu = [
            np.random.randn(self.channels, self.led_count).astype(np.float32) for _ in range(num_trials + num_warmup)
        ]
        test_values_gpu = [cupy.asarray(x) for x in test_values_cpu]

        results = {}

        # GPU timing (symmetric fallback)
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

        results["symmetric_fallback"] = np.mean(times)

        # GPU timing (symmetric custom kernel)
        if SYMMETRIC_KERNEL_AVAILABLE:
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

            results["symmetric_custom"] = np.mean(times)

            # GPU timing (symmetric optimized kernel)
            for i in range(num_warmup):
                _ = self.multiply_3d(test_values_gpu[i], use_custom_kernel=True, optimized_kernel=True)
            cupy.cuda.Stream.null.synchronize()

            times = []
            for i in range(num_warmup, num_warmup + num_trials):
                start_event = cupy.cuda.Event()
                end_event = cupy.cuda.Event()

                start_event.record()
                _ = self.multiply_3d(test_values_gpu[i], use_custom_kernel=True, optimized_kernel=True)
                end_event.record()
                end_event.synchronize()

                times.append(cupy.cuda.get_elapsed_time(start_event, end_event))

            results["symmetric_optimized"] = np.mean(times)

        return results

    @staticmethod
    def from_dict(data_dict: dict) -> "SymmetricDiagonalATAMatrix":
        """
        Create SymmetricDiagonalATAMatrix from dictionary format.

        Args:
            data_dict: Dictionary containing symmetric DIA matrix data

        Returns:
            SymmetricDiagonalATAMatrix instance
        """
        # Create instance
        symmetric = SymmetricDiagonalATAMatrix(
            led_count=data_dict["led_count"],
            crop_size=data_dict.get("crop_size", 64),
            output_dtype=cupy.float32,  # Always FP32 for symmetric version
        )

        # Set data
        symmetric.dia_data_gpu = cupy.asarray(data_dict["dia_data_gpu"], dtype=cupy.float32)
        symmetric.dia_offsets_upper = data_dict["dia_offsets_upper"]
        symmetric.dia_offsets_upper_gpu = cupy.asarray(data_dict["dia_offsets_upper"], dtype=cupy.int32)
        symmetric.k_upper = data_dict["k_upper"]
        symmetric.bandwidth = data_dict["bandwidth"]
        symmetric.channels = data_dict.get("channels", 3)

        # Optional metadata
        if "original_k" in data_dict:
            symmetric.original_k = data_dict["original_k"]
        if "sparsity" in data_dict:
            symmetric.sparsity = data_dict["sparsity"]
        if "nnz" in data_dict:
            symmetric.nnz = data_dict["nnz"]

        return symmetric

    def to_dict(self) -> dict:
        """
        Convert SymmetricDiagonalATAMatrix to dictionary format for serialization.

        Returns:
            Dictionary containing all matrix data
        """
        return {
            "dia_data_gpu": cupy.asnumpy(self.dia_data_gpu),
            "dia_offsets_upper": self.dia_offsets_upper,
            "k_upper": self.k_upper,
            "bandwidth": self.bandwidth,
            "led_count": self.led_count,
            "channels": self.channels,
            "crop_size": self.crop_size,
            "output_dtype": str(self.output_dtype),
            "original_k": getattr(self, "original_k", None),
            "sparsity": getattr(self, "sparsity", None),
            "nnz": getattr(self, "nnz", None),
        }
