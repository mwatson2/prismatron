#!/usr/bin/env python3
"""
Batch Symmetric Diagonal A^T A Matrix utility class for LED optimization.

This class implements a batch-optimized version of SymmetricDiagonalATAMatrix that:
1. Uses 16x16 block storage format for WMMA tensor core optimization
2. Processes multiple vectors simultaneously (batch matmul instead of matvec)
3. Maintains symmetric storage benefits (~50% memory reduction)
4. Leverages tensor cores for maximum throughput on modern GPUs
5. REQUIRES GPU with compute capability 7.0+ for tensor core support

The matrix is stored as symmetric block diagonals where each block is 16x16,
optimized for WMMA operations on tensor cores. No fallback implementation.
"""

import math
from typing import Optional, Tuple

try:
    import cupy
except ImportError:
    # Fallback for systems without CUDA
    import numpy as cupy

import numpy as np

from .base_ata_matrix import BaseATAMatrix

# Import batch WMMA kernels - precompiled tensor core version only
try:
    from .kernels.precompiled_mma_kernel import PRECOMPILED_MMA_SUPPORTED, PrecompiledBatchSymmetricWMMAMatMul

    if not PRECOMPILED_MMA_SUPPORTED:
        raise RuntimeError(
            "Precompiled MMA kernels required but not available. Run 'make' in src/utils/kernels/ to compile."
        )

    BATCH_WMMA_KERNEL_AVAILABLE = True
    print("Using precompiled MMA tensor core kernels (ahead-of-time compilation)")

except ImportError:
    BATCH_WMMA_KERNEL_AVAILABLE = False
    print("Error: Precompiled MMA kernels not available. Run 'make' in src/utils/kernels/ to compile.")


class BatchSymmetricDiagonalATAMatrix(BaseATAMatrix):
    """
    Batch-optimized symmetric diagonal A^T A matrix implementation using 16x16 block storage.

    Key features:
    - 16x16 block storage format optimized for WMMA tensor cores
    - Symmetric storage: only upper triangular blocks stored
    - Batch processing: handles multiple input vectors simultaneously
    - FP32 precision for compute and accumulation
    - REQUIRES GPU with compute capability 7.0+ for tensor core support

    Storage Format:
    - block_data_gpu: (channels, block_diag_count, 16, 16) - 16x16 blocks
    - block_offsets_upper: (block_diag_count,) - block diagonal offsets
    - block_masks_gpu: (channels, block_diag_count, 16, 16) - sparsity masks
    """

    def __init__(
        self,
        led_count: int,
        crop_size: int = 64,
        batch_size: int = 16,
        block_size: int = 16,
        output_dtype: Optional[cupy.dtype] = None,
    ):
        """
        Initialize batch symmetric diagonal A^T A matrix container.

        Args:
            led_count: Number of LEDs
            crop_size: Crop size used for LED regions (affects adjacency)
            batch_size: Number of vectors to process simultaneously (8 or 16 recommended)
            block_size: Size of matrix blocks (fixed at 16 for WMMA)
            output_dtype: Data type for output tensors (cupy.float32 recommended)
        """
        self.led_count = led_count
        self.crop_size = crop_size
        self.channels = 3  # RGB
        self.batch_size = batch_size

        # Block configuration for WMMA tensor cores
        if block_size != 16:
            raise ValueError("Block size must be 16 for WMMA tensor core optimization")
        self.block_size = block_size

        # Calculate block matrix dimensions
        self.led_blocks = math.ceil(led_count / block_size)  # Number of blocks per dimension
        self.padded_led_count = self.led_blocks * block_size  # Padded to block boundary

        # Validate batch size for optimal tensor core usage
        if batch_size not in [8, 16]:
            print(f"Warning: batch_size {batch_size} not optimal for tensor cores. Recommended: 8 or 16")

        # Data type configuration
        self.output_dtype = output_dtype or cupy.float32
        self.compute_dtype = cupy.float32  # Use FP32 for accuracy

        # Block storage for symmetric format
        # Only stores main diagonal and upper diagonal blocks (leveraging symmetry)
        self.block_data_gpu = None  # Shape: (channels, block_diag_count, 16, 16)
        self.block_offsets_upper = None  # Shape: (block_diag_count,) - block offsets
        self.block_masks_gpu = None  # Shape: (channels, block_diag_count, 16, 16) - sparsity masks
        self.block_diag_count = None  # Number of upper diagonal block bands

        # WMMA kernel instances
        self.wmma_kernel_basic = None
        self.wmma_kernel_optimized = None

        # Metadata
        self.bandwidth = None
        self.sparsity = None
        self.nnz = None
        self.original_k = None  # Original number of element diagonals

        # Initialize MMA kernels - precompiled tensor core version only
        if not BATCH_WMMA_KERNEL_AVAILABLE:
            raise RuntimeError(
                "Precompiled MMA tensor cores required but not available. Run 'make' in src/utils/kernels/ to compile."
            )

        # Use precompiled tensor core kernels only
        self.wmma_kernel_basic = PrecompiledBatchSymmetricWMMAMatMul(use_optimized=False)
        self.wmma_kernel_optimized = PrecompiledBatchSymmetricWMMAMatMul(use_optimized=True)

    def _validate_batch_tensor_layout(self, tensor: cupy.ndarray, tensor_name: str) -> None:
        """
        Validate batch tensor memory layout for optimal WMMA kernel performance.

        Args:
            tensor: Input batch tensor to validate
            tensor_name: Name for error messages

        Raises:
            ValueError: If tensor layout is incompatible with WMMA kernels
        """
        if not isinstance(tensor, cupy.ndarray):
            raise TypeError(f"{tensor_name} must be cupy.ndarray, got {type(tensor)}")

        # Check batch tensor shape
        if len(tensor.shape) != 3:
            raise ValueError(
                f"{tensor_name} must be 3D batch tensor (batch_size, channels, leds), got shape {tensor.shape}"
            )

        batch_size, channels, leds = tensor.shape
        if channels != self.channels:
            raise ValueError(f"{tensor_name} must have {self.channels} channels, got {channels}")

        if leds != self.led_count:
            raise ValueError(f"{tensor_name} must have {self.led_count} LEDs, got {leds}")

        # Validate C-contiguous layout for optimal memory coalescing
        if not tensor.flags.c_contiguous:
            stride_info = f"strides={tensor.strides}, shape={tensor.shape}"
            layout = "F-contiguous" if tensor.flags.f_contiguous else "non-contiguous"

            raise ValueError(
                f"Batch WMMA kernel requires C-contiguous {tensor_name} tensor for optimal performance. "
                f"Got {layout} tensor with {stride_info}. "
                f"Use cp.ascontiguousarray({tensor_name}) to fix layout issues."
            )

    @staticmethod
    def from_symmetric_diagonal_matrix(symmetric_matrix) -> "BatchSymmetricDiagonalATAMatrix":
        """
        Create batch version from symmetric diagonal ATA matrix.

        Args:
            symmetric_matrix: Instance of SymmetricDiagonalATAMatrix

        Returns:
            BatchSymmetricDiagonalATAMatrix with 16x16 block storage
        """
        print("Converting SymmetricDiagonalATAMatrix to batch block storage...")

        # Create batch instance
        batch_matrix = BatchSymmetricDiagonalATAMatrix(
            led_count=symmetric_matrix.led_count,
            crop_size=symmetric_matrix.crop_size,
            batch_size=16,  # Default optimal batch size
            output_dtype=symmetric_matrix.output_dtype,
        )

        # Copy metadata
        batch_matrix.bandwidth = symmetric_matrix.bandwidth
        batch_matrix.sparsity = symmetric_matrix.sparsity
        batch_matrix.nnz = symmetric_matrix.nnz
        batch_matrix.original_k = symmetric_matrix.original_k

        # Convert element-wise diagonals to block diagonals
        if symmetric_matrix.dia_data_gpu is None or symmetric_matrix.dia_offsets_upper is None:
            raise ValueError("Symmetric matrix not built yet. Call from_diagonal_ata_matrix() first.")

        # Convert to block storage format
        batch_matrix._convert_diagonal_to_blocks(symmetric_matrix.dia_data_gpu, symmetric_matrix.dia_offsets_upper)

        print("  Conversion completed!")
        print(f"  Block storage shape: {batch_matrix.block_data_gpu.shape}")
        print(f"  Block diagonal count: {batch_matrix.block_diag_count}")
        print(f"  GPU memory usage: {batch_matrix.block_data_gpu.nbytes / 1024**2:.1f} MB")

        return batch_matrix

    @staticmethod
    def from_diagonal_ata_matrix(regular_matrix, batch_size: int = 16) -> "BatchSymmetricDiagonalATAMatrix":
        """
        Create batch version directly from regular DiagonalATAMatrix.

        Args:
            regular_matrix: Instance of DiagonalATAMatrix
            batch_size: Batch size for processing (8 or 16 recommended)

        Returns:
            BatchSymmetricDiagonalATAMatrix with 16x16 block storage
        """
        print("Converting DiagonalATAMatrix directly to batch block storage...")

        # Create batch instance
        batch_matrix = BatchSymmetricDiagonalATAMatrix(
            led_count=regular_matrix.led_count,
            crop_size=regular_matrix.crop_size,
            batch_size=batch_size,
            output_dtype=cupy.float32,  # Use FP32 output
        )

        # Copy metadata
        batch_matrix.bandwidth = regular_matrix.bandwidth
        batch_matrix.sparsity = regular_matrix.sparsity
        batch_matrix.nnz = regular_matrix.nnz
        batch_matrix.original_k = regular_matrix.k

        # Extract upper diagonals from regular matrix (for symmetric optimization)
        if regular_matrix.dia_offsets is None or regular_matrix.dia_data_cpu is None:
            raise ValueError("Regular matrix not built yet. Call build_from_diffusion_matrix() first.")

        # Find upper diagonal indices (offset >= 0)
        upper_mask = regular_matrix.dia_offsets >= 0
        upper_indices = np.where(upper_mask)[0]

        if len(upper_indices) == 0:
            raise ValueError("No upper diagonals found in regular matrix")

        # Extract upper diagonals
        upper_offsets = regular_matrix.dia_offsets[upper_indices].copy()
        upper_data_cpu = regular_matrix.dia_data_cpu[:, upper_indices, :].copy()

        print(f"  Original diagonals: {regular_matrix.k}")
        print(f"  Upper diagonals (including main): {len(upper_offsets)}")
        print(f"  Memory reduction: {len(upper_offsets) / regular_matrix.k * 100:.1f}% of original")

        # Convert to GPU for block conversion
        upper_data_gpu = cupy.asarray(upper_data_cpu, dtype=cupy.float32)

        # Convert to block storage format
        batch_matrix._convert_diagonal_to_blocks(upper_data_gpu, upper_offsets)

        print("  Direct conversion completed!")
        print(f"  Block storage shape: {batch_matrix.block_data_gpu.shape}")
        print(f"  Block diagonal count: {batch_matrix.block_diag_count}")
        print(f"  GPU memory usage: {batch_matrix.block_data_gpu.nbytes / 1024**2:.1f} MB")

        return batch_matrix

    def _convert_diagonal_to_blocks(self, dia_data_gpu: cupy.ndarray, dia_offsets_upper: np.ndarray) -> None:
        """
        Convert element-wise diagonal storage to 16x16 block storage.

        Simple approach: reconstruct dense matrix, divide into 16x16 blocks,
        store only upper triangular blocks in memory.

        Args:
            dia_data_gpu: Element diagonal data (channels, k_upper, leds)
            dia_offsets_upper: Element diagonal offsets (k_upper,)
        """
        print("  Converting element diagonals to 16x16 blocks...")

        # Step 1: Reconstruct the full symmetric matrix from diagonals
        print("    Reconstructing full symmetric matrix...")

        # Pad matrix size to multiple of 16 for clean block division
        padded_size = self.padded_led_count

        dense_matrices = []
        for channel in range(self.channels):
            # Create padded matrix (zeros in padding region)
            dense_matrix = np.zeros((padded_size, padded_size), dtype=np.float32)

            # Fill from element diagonals
            for diag_idx, offset in enumerate(dia_offsets_upper):
                diag_data = cupy.asnumpy(dia_data_gpu[channel, diag_idx, : self.led_count])

                # Fill upper diagonal
                for i in range(self.led_count):
                    j = i + offset
                    if j < self.led_count:
                        dense_matrix[i, j] = diag_data[i]

                        # Fill symmetric lower diagonal (if not main diagonal)
                        if offset > 0:
                            dense_matrix[j, i] = diag_data[i]

            dense_matrices.append(dense_matrix)

        # Step 2: Simple approach - store ALL upper triangular blocks
        print("    Dividing into 16x16 blocks...")

        # Calculate maximum number of block diagonals needed
        max_block_diag = self.led_blocks - 1  # From (0,0) to (0, led_blocks-1)

        # We'll store block diagonals 0, 1, 2, ..., max_block_diag
        block_offsets_cpu = np.arange(max_block_diag + 1, dtype=np.int32)
        self.block_offsets_upper = cupy.asarray(block_offsets_cpu, dtype=cupy.int32)
        self.block_diag_count = len(block_offsets_cpu)

        print(f"    Led blocks: {self.led_blocks}x{self.led_blocks}")
        print(f"    Block diagonals: {self.block_diag_count} (offsets 0 to {max_block_diag})")

        # Step 3: Initialize block storage
        block_shape = (self.channels, self.block_diag_count, self.block_size, self.block_size)
        self.block_data_gpu = cupy.zeros(block_shape, dtype=self.compute_dtype)

        # Step 4: Extract all 16x16 blocks from dense matrix
        print("    Extracting blocks...")

        for channel in range(self.channels):
            dense_matrix = dense_matrices[channel]

            for block_diag_idx, block_offset in enumerate(cupy.asnumpy(self.block_offsets_upper)):
                # Extract all blocks along this block diagonal
                for block_row in range(self.led_blocks):
                    block_col = block_row + block_offset

                    if block_col < self.led_blocks:
                        # Extract 16x16 block (padded matrix ensures this is always 16x16)
                        row_start = block_row * self.block_size
                        row_end = row_start + self.block_size
                        col_start = block_col * self.block_size
                        col_end = col_start + self.block_size

                        # Extract block directly (no padding needed since matrix is pre-padded)
                        block = dense_matrix[row_start:row_end, col_start:col_end].copy()

                        # Store block
                        self.block_data_gpu[channel, block_diag_idx, :, :] = cupy.asarray(
                            block, dtype=self.compute_dtype
                        )

        print("    Block conversion completed!")
        print(
            f"    Total blocks stored: {self.channels} channels × {self.block_diag_count} diagonals × {self.led_blocks} blocks = {self.channels * self.block_diag_count * self.led_blocks}"
        )
        print(f"    Memory usage: {self.block_data_gpu.nbytes / 1024**2:.1f} MB")

    def multiply_batch_3d(
        self,
        led_values_batch: cupy.ndarray,
        optimized_kernel: bool = True,
        debug_logging: bool = False,
    ) -> cupy.ndarray:
        """
        Perform batch symmetric 3D block diagonal matrix-vector multiplication.

        Uses async MMA tensor cores for optimal performance with 16x16 block operations.
        Requires GPU with compute capability 8.0+ for async tensor core support.

        Args:
            led_values_batch: Batch LED values (batch_size, 3, leds)
            optimized_kernel: Whether to use optimized kernel variant
            debug_logging: Enable detailed logging

        Returns:
            Result batch (batch_size, 3, leds)

        Raises:
            RuntimeError: If async MMA tensor cores are not available
        """
        batch_size, channels, leds = led_values_batch.shape

        if batch_size != self.batch_size:
            print(f"Warning: input batch_size {batch_size} != configured batch_size {self.batch_size}")

        # Validate batch tensor layout
        self._validate_batch_tensor_layout(led_values_batch, "led_values_batch")

        # Check if block matrix is built
        if self.block_data_gpu is None:
            raise RuntimeError("Block matrix not built. Call from_symmetric_diagonal_matrix() first.")

        # Ensure input is correct dtype
        if led_values_batch.dtype != self.compute_dtype:
            led_values_batch = led_values_batch.astype(self.compute_dtype)

        # Pad input to block boundary if needed
        if leds < self.padded_led_count:
            padded_input = cupy.zeros((batch_size, channels, self.padded_led_count), dtype=self.compute_dtype)
            padded_input[:, :, :leds] = led_values_batch
            led_values_batch = padded_input

        # Perform batch WMMA matrix-vector multiplication
        if optimized_kernel:
            if debug_logging:
                print("Batch multiply_3d: Using OPTIMIZED BatchSymmetricWMMAMatMul kernel")
            if self.wmma_kernel_optimized is None:
                self.wmma_kernel_optimized = PrecompiledBatchSymmetricWMMAMatMul(use_optimized=True)
            result_gpu = self.wmma_kernel_optimized(
                self.block_data_gpu,  # (channels, block_diag_count, 16, 16)
                self.block_offsets_upper,  # (block_diag_count,)
                led_values_batch,  # (batch_size, channels, leds)
                self.led_blocks,
                self.padded_led_count,
            )
        else:
            if debug_logging:
                print("Batch multiply_3d: Using BASIC BatchSymmetricWMMAMatMul kernel")
            if self.wmma_kernel_basic is None:
                self.wmma_kernel_basic = PrecompiledBatchSymmetricWMMAMatMul(use_optimized=False)
            result_gpu = self.wmma_kernel_basic(
                self.block_data_gpu,
                self.block_offsets_upper,
                led_values_batch,
                self.led_blocks,
                self.padded_led_count,
            )

        # Convert output to desired dtype and trim to original LED count
        if result_gpu.dtype != self.output_dtype:
            result_gpu = result_gpu.astype(self.output_dtype)

        # Trim padded dimensions back to original LED count
        result_gpu = result_gpu[:, :, : self.led_count]

        return result_gpu

    def g_ata_g_batch_3d(
        self,
        gradient_batch: cupy.ndarray,
        optimized_kernel: bool = True,
    ) -> cupy.ndarray:
        """
        Compute batch g^T (A^T A) g for step size calculation using WMMA kernels.

        Args:
            gradient_batch: Batch gradient arrays (batch_size, 3, leds)
            optimized_kernel: Whether to use optimized kernel variant

        Returns:
            Result batch (batch_size, 3) - one value per channel per batch item
        """
        # Validate batch tensor layout
        self._validate_batch_tensor_layout(gradient_batch, "gradient_batch")

        # Ensure input is correct dtype
        if gradient_batch.dtype != self.compute_dtype:
            gradient_batch = gradient_batch.astype(self.compute_dtype)

        # Compute (A^T A) @ g_batch using WMMA kernels
        ata_g_batch = self.multiply_batch_3d(gradient_batch, optimized_kernel=optimized_kernel, debug_logging=False)

        # Compute g^T @ (A^T A @ g) for each batch item and channel
        # (batch_size, channels, leds) * (batch_size, channels, leds) -> (batch_size, channels)
        result_batch = cupy.sum(gradient_batch * ata_g_batch, axis=2)

        # Convert to output dtype
        if result_batch.dtype != self.output_dtype:
            result_batch = result_batch.astype(self.output_dtype)

        return result_batch

    def multiply_3d(
        self,
        led_values: cupy.ndarray,
        optimized_kernel: bool = True,
        output_dtype: Optional[cupy.dtype] = None,
        debug_logging: bool = False,
    ) -> cupy.ndarray:
        """
        Single vector multiply (wrapper around batch version for BaseATAMatrix compatibility).

        Args:
            led_values: LED values array (3, leds)
            optimized_kernel: Whether to use optimized kernel variant
            output_dtype: Desired output data type
            debug_logging: Enable detailed logging

        Returns:
            Result array (3, leds)
        """
        # Validate input shape
        if len(led_values.shape) != 2 or led_values.shape[0] != 3:
            raise ValueError(f"Expected led_values shape (3, leds), got {led_values.shape}")

        # Create batch of size 1
        led_values_batch = led_values.reshape(1, led_values.shape[0], led_values.shape[1])

        # Call batch version
        result_batch = self.multiply_batch_3d(
            led_values_batch, optimized_kernel=optimized_kernel, debug_logging=debug_logging
        )

        # Extract single result
        result = result_batch[0]

        # Convert to output dtype if needed
        if output_dtype is not None and result.dtype != output_dtype:
            result = result.astype(output_dtype)

        return result

    def g_ata_g_3d(
        self,
        gradient: cupy.ndarray,
        optimized_kernel: bool = True,
        output_dtype: Optional[cupy.dtype] = None,
    ) -> cupy.ndarray:
        """
        Single vector g^T A^T A g (wrapper around batch version for BaseATAMatrix compatibility).

        Args:
            gradient: Gradient array (3, leds)
            optimized_kernel: Whether to use optimized kernel variant
            output_dtype: Desired output data type

        Returns:
            Result array (3,) - one value per channel
        """
        # Validate input shape
        if len(gradient.shape) != 2 or gradient.shape[0] != 3:
            raise ValueError(f"Expected gradient shape (3, leds), got {gradient.shape}")

        # Create batch of size 1
        gradient_batch = gradient.reshape(1, gradient.shape[0], gradient.shape[1])

        # Call batch version
        result_batch = self.g_ata_g_batch_3d(gradient_batch, optimized_kernel=optimized_kernel)

        # Extract single result
        result = result_batch[0]

        # Convert to output dtype if needed
        if output_dtype is not None and result.dtype != output_dtype:
            result = result.astype(output_dtype)

        return result

    def get_info(self):
        """Get summary information about the batch block matrix."""
        return {
            "led_count": self.led_count,
            "crop_size": self.crop_size,
            "channels": self.channels,
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "led_blocks": self.led_blocks,
            "padded_led_count": self.padded_led_count,
            "bandwidth": self.bandwidth,
            "sparsity": self.sparsity,
            "nnz": self.nnz,
            "output_dtype": str(self.output_dtype),
            "compute_dtype": str(self.compute_dtype),
            "block_storage": True,
            "symmetric_storage": True,
            "original_k": self.original_k,
            "block_diag_count": self.block_diag_count,
            "storage_shape": self.block_data_gpu.shape if self.block_data_gpu is not None else None,
            "wmma_kernel_available": BATCH_WMMA_KERNEL_AVAILABLE,
        }

    def benchmark_batch_3d(self, num_trials: int = 20, num_warmup: int = 5):
        """
        Benchmark batch 3D WMMA multiplication performance.

        Args:
            num_trials: Number of timing trials
            num_warmup: Number of warmup iterations

        Returns:
            Timing results dictionary
        """
        if self.block_data_gpu is None:
            raise RuntimeError("Block matrix not built")

        # Create test batch data
        test_batches_cpu = [
            np.random.randn(self.batch_size, self.channels, self.led_count).astype(np.float32)
            for _ in range(num_trials + num_warmup)
        ]
        test_batches_gpu = [cupy.asarray(x, dtype=self.compute_dtype) for x in test_batches_cpu]

        results = {}

        # Basic WMMA kernel
        for i in range(num_warmup):
            _ = self.multiply_batch_3d(test_batches_gpu[i], optimized_kernel=False)
        cupy.cuda.Stream.null.synchronize()

        times = []
        for i in range(num_warmup, num_warmup + num_trials):
            start_event = cupy.cuda.Event()
            end_event = cupy.cuda.Event()

            start_event.record()
            _ = self.multiply_batch_3d(test_batches_gpu[i], optimized_kernel=False)
            end_event.record()
            end_event.synchronize()

            times.append(cupy.cuda.get_elapsed_time(start_event, end_event))

        results["batch_wmma_basic"] = np.mean(times)

        # Optimized WMMA kernel
        for i in range(num_warmup):
            _ = self.multiply_batch_3d(test_batches_gpu[i], optimized_kernel=True)
        cupy.cuda.Stream.null.synchronize()

        times = []
        for i in range(num_warmup, num_warmup + num_trials):
            start_event = cupy.cuda.Event()
            end_event = cupy.cuda.Event()

            start_event.record()
            _ = self.multiply_batch_3d(test_batches_gpu[i], optimized_kernel=True)
            end_event.record()
            end_event.synchronize()

            times.append(cupy.cuda.get_elapsed_time(start_event, end_event))

        results["batch_wmma_optimized"] = np.mean(times)

        return results
