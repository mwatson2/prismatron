"""
Single Block Mixed Sparse Tensor - Custom tensor for LED diffusion patterns.

This module implements a specialized 4D tensor format optimized for LED diffusion
patterns where each LED has exactly one small dense region in the spatial dimensions.

Key features:
- 4D tensor (batch, channels, height, width) with mixed sparsity
- First two dimensions (batch, channels) are dense
- Last two dimensions (height, width) are sparse with exactly one square block per sub-tensor
- Optimized for GPU operations and repeated A^T @ b computations
- Memory efficient: stores only non-zero blocks + positions
"""

import logging
from typing import Dict, Optional, Tuple, Union

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)


class SingleBlockMixedSparseTensor:
    """
    Specialized 4D tensor for LED diffusion patterns with single dense blocks.

    Stores tensors of shape (batch_size, channels, height, width) where each
    (height, width) sub-tensor contains exactly one dense square block.

    Memory layout:
    - sparse_values: (batch_size, channels, block_size, block_size) - dense blocks
    - block_positions: (batch_size, channels, 2) - top-left coordinates

    Example:
        # Create tensor for 1000 LEDs, 3 channels, 800x480 images, 64x64 blocks
        tensor = SingleBlockMixedSparseTensor(1000, 3, 800, 480, 64)

        # Set a block for LED 0, red channel at position (100, 150)
        block_data = cp.random.rand(64, 64, dtype=cp.float32)
        tensor.set_block(0, 0, 100, 150, block_data)

        # Compute A^T @ b operation
        target_image = cp.random.rand(800, 480, dtype=cp.float32)
        result = tensor.transpose_dot_product(target_image)  # Returns (1000, 3)
    """

    def __init__(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        block_size: int = 96,
        device: str = "cuda",
    ):
        """
        Initialize the single block sparse tensor.

        Args:
            batch_size: Number of LEDs
            channels: Number of color channels (typically 3 for RGB)
            height: Spatial height of full images
            width: Spatial width of full images
            block_size: Size of square dense blocks (e.g., 64)
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.block_size = block_size
        self.device = device

        # Storage for dense blocks - only non-zero regions
        self.sparse_values = cp.zeros(
            (batch_size, channels, block_size, block_size), dtype=cp.float32
        )

        # Storage for block positions (top-left coordinates)
        self.block_positions = cp.zeros((batch_size, channels, 2), dtype=cp.int32)

        # Track which blocks have been set
        self.blocks_set = cp.zeros((batch_size, channels), dtype=cp.bool_)

        logger.debug(
            f"SingleBlockMixedSparseTensor created: "
            f"({batch_size}, {channels}, {height}, {width}) "
            f"with {block_size}x{block_size} blocks"
        )

    def set_block(
        self,
        batch_idx: int,
        channel_idx: int,
        top_left_row: int,
        top_left_col: int,
        values: cp.ndarray,
    ) -> None:
        """
        Set a dense block for a specific LED and channel.

        Args:
            batch_idx: LED index (0 to batch_size-1)
            channel_idx: Channel index (0 to channels-1)
            top_left_row: Row coordinate of block's top-left corner
            top_left_col: Column coordinate of block's top-left corner
            values: Dense block data, shape (block_size, block_size)
        """
        # Validate inputs
        if not (0 <= batch_idx < self.batch_size):
            raise ValueError(
                f"batch_idx {batch_idx} out of range [0, {self.batch_size})"
            )
        if not (0 <= channel_idx < self.channels):
            raise ValueError(
                f"channel_idx {channel_idx} out of range [0, {self.channels})"
            )
        if not (0 <= top_left_row <= self.height - self.block_size):
            raise ValueError(
                f"top_left_row {top_left_row} out of range "
                f"[0, {self.height - self.block_size}]"
            )
        if not (0 <= top_left_col <= self.width - self.block_size):
            raise ValueError(
                f"top_left_col {top_left_col} out of range "
                f"[0, {self.width - self.block_size}]"
            )
        if values.shape != (self.block_size, self.block_size):
            raise ValueError(
                f"values shape {values.shape} != expected "
                f"({self.block_size}, {self.block_size})"
            )

        # Store the block data and position
        self.sparse_values[batch_idx, channel_idx] = values
        self.block_positions[batch_idx, channel_idx, 0] = top_left_row
        self.block_positions[batch_idx, channel_idx, 1] = top_left_col
        self.blocks_set[batch_idx, channel_idx] = True

    def set_blocks_batch(self, positions: cp.ndarray, values: cp.ndarray) -> None:
        """
        Set multiple blocks efficiently in batch.

        Args:
            positions: Block positions, shape (batch_size, channels, 2)
                      positions[i, c, :] = [top_left_row, top_left_col]
            values: Block values, shape (batch_size, channels, block_size, block_size)
        """
        if positions.shape != (self.batch_size, self.channels, 2):
            raise ValueError(
                f"positions shape {positions.shape} != expected "
                f"({self.batch_size}, {self.channels}, 2)"
            )
        if values.shape != (
            self.batch_size,
            self.channels,
            self.block_size,
            self.block_size,
        ):
            raise ValueError(
                f"values shape {values.shape} != expected "
                f"({self.batch_size}, {self.channels}, {self.block_size}, {self.block_size})"
            )

        # Validate positions are within bounds
        rows = positions[:, :, 0]
        cols = positions[:, :, 1]
        if cp.any(rows < 0) or cp.any(rows > self.height - self.block_size):
            raise ValueError("Some row positions are out of bounds")
        if cp.any(cols < 0) or cp.any(cols > self.width - self.block_size):
            raise ValueError("Some column positions are out of bounds")

        # Set all blocks at once
        self.sparse_values[:] = values
        self.block_positions[:] = positions
        self.blocks_set[:] = True

    def transpose_dot_product(
        self, dense_matrix: cp.ndarray, chunk_size: int = 512
    ) -> cp.ndarray:
        """
        Compute A^T @ b operation efficiently using chunked block extraction.

        This is the core operation for LED optimization: multiply the transpose
        of the sparse tensor with a dense target image.

        Args:
            dense_matrix: Target image, shape (height, width)
            chunk_size: Number of blocks to process at once (memory vs speed tradeoff)

        Returns:
            Result of A^T @ b, shape (batch_size, channels)
        """
        if dense_matrix.shape != (self.height, self.width):
            raise ValueError(
                f"dense_matrix shape {dense_matrix.shape} != expected "
                f"({self.height}, {self.width})"
            )

        # Result tensor
        results = cp.zeros((self.batch_size, self.channels), dtype=cp.float32)

        # Process in chunks to manage memory usage
        total_elements = self.batch_size * self.channels

        for chunk_start in range(0, total_elements, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_elements)

            # Convert flat indices to (batch, channel) indices
            flat_indices = cp.arange(chunk_start, chunk_end)
            batch_indices = flat_indices // self.channels
            channel_indices = flat_indices % self.channels

            # Extract dense blocks from target image for this chunk
            dense_blocks = self._extract_dense_blocks_vectorized(
                dense_matrix, batch_indices, channel_indices
            )

            # Get corresponding sparse blocks
            sparse_blocks = self.sparse_values[batch_indices, channel_indices]

            # Compute element-wise multiplication and sum
            # Shape: (chunk_size, block_size, block_size) -> (chunk_size,)
            chunk_results = cp.sum(sparse_blocks * dense_blocks, axis=(1, 2))

            # Store results back to proper positions
            results[batch_indices, channel_indices] = chunk_results

        return results

    def transpose_dot_product_cuda(self, dense_matrix: cp.ndarray) -> cp.ndarray:
        """
        Compute A^T @ b operation using custom CUDA kernel (optimized).

        This method uses a custom CUDA kernel for maximum performance,
        eliminating chunking overhead and optimizing memory access patterns.

        Args:
            dense_matrix: Target image, shape (height, width)

        Returns:
            Result of A^T @ b, shape (batch_size, channels)
        """
        if dense_matrix.shape != (self.height, self.width):
            raise ValueError(
                f"dense_matrix shape {dense_matrix.shape} != expected "
                f"({self.height}, {self.width})"
            )

        try:
            from .cuda_kernels import cuda_transpose_dot_product

            # Use CUDA kernel implementation
            result = cuda_transpose_dot_product(
                self.sparse_values,
                self.block_positions,
                self.blocks_set,
                dense_matrix,
                self.batch_size,
                self.channels,
                self.block_size,
            )

            return result

        except ImportError as e:
            logger.warning(
                f"CUDA kernel not available: {e}. Falling back to chunked implementation."
            )
            # Fall back to chunked implementation
            return self.transpose_dot_product(dense_matrix)

    def transpose_dot_product_cuda_corrected(
        self, dense_matrix: cp.ndarray
    ) -> cp.ndarray:
        """
        Compute A^T @ b operation using corrected CUDA kernel with proper parallelism.

        This method uses a corrected CUDA kernel where 256 threads collaborate on each
        (LED, channel) dot product for maximum performance and correctness.

        Args:
            dense_matrix: Target image, shape (height, width)

        Returns:
            Result of A^T @ b, shape (batch_size, channels)
        """
        if dense_matrix.shape != (self.height, self.width):
            raise ValueError(
                f"dense_matrix shape {dense_matrix.shape} != expected "
                f"({self.height}, {self.width})"
            )

        try:
            from .cuda_kernels import cuda_transpose_dot_product_corrected

            # Use corrected CUDA kernel implementation
            result = cuda_transpose_dot_product_corrected(
                self.sparse_values,
                self.block_positions,
                self.blocks_set,
                dense_matrix,
                self.batch_size,
                self.channels,
                self.block_size,
            )

            return result

        except ImportError as e:
            logger.warning(
                f"Corrected CUDA kernel not available: {e}. Falling back to standard CUDA kernel."
            )
            # Fall back to standard CUDA kernel
            return self.transpose_dot_product_cuda(dense_matrix)

    def transpose_dot_product_cuda_high_performance(
        self, dense_matrix: cp.ndarray
    ) -> cp.ndarray:
        """
        Compute A^T @ b operation using high-performance CUDA kernel targeting 20+ GFLOPS.

        This method uses an optimized CUDA kernel designed for maximum throughput on
        high-end GPUs, targeting 2600+ LEDs with 10s of GFLOPS performance.

        Args:
            dense_matrix: Target image, shape (height, width)

        Returns:
            Result of A^T @ b, shape (batch_size, channels)
        """
        if dense_matrix.shape != (self.height, self.width):
            raise ValueError(
                f"dense_matrix shape {dense_matrix.shape} != expected "
                f"({self.height}, {self.width})"
            )

        try:
            from .cuda_kernels import cuda_transpose_dot_product_high_performance

            # Use high-performance CUDA kernel implementation
            result = cuda_transpose_dot_product_high_performance(
                self.sparse_values,
                self.block_positions,
                self.blocks_set,
                dense_matrix,
                self.batch_size,
                self.channels,
                self.block_size,
            )

            return result

        except ImportError as e:
            logger.warning(
                f"High-performance CUDA kernel not available: {e}. Falling back to corrected kernel."
            )
            # Fall back to corrected CUDA kernel
            return self.transpose_dot_product_cuda_corrected(dense_matrix)

    def transpose_dot_product_cuda_high_parallelism(
        self, dense_matrix: cp.ndarray
    ) -> cp.ndarray:
        """
        Compute A^T @ b operation using high-parallelism CUDA kernel (maximum performance).

        This method uses a high-parallelism CUDA kernel with 256 threads per LED computation
        for maximum GPU utilization and performance.

        Args:
            dense_matrix: Target image, shape (height, width)

        Returns:
            Result of A^T @ b, shape (batch_size, channels)
        """
        if dense_matrix.shape != (self.height, self.width):
            raise ValueError(
                f"dense_matrix shape {dense_matrix.shape} != expected "
                f"({self.height}, {self.width})"
            )

        try:
            from .cuda_kernels import cuda_transpose_dot_product_high_parallelism

            # Use high-parallelism CUDA kernel implementation
            result = cuda_transpose_dot_product_high_parallelism(
                self.sparse_values,
                self.block_positions,
                self.blocks_set,
                dense_matrix,
                self.batch_size,
                self.channels,
                self.block_size,
            )

            return result

        except ImportError as e:
            logger.warning(
                f"High-parallelism CUDA kernel not available: {e}. Falling back to standard CUDA kernel."
            )
            # Fall back to standard CUDA kernel
            return self.transpose_dot_product_cuda(dense_matrix)

    def transpose_dot_product_cuda_compute_optimized(
        self, dense_matrix: cp.ndarray
    ) -> cp.ndarray:
        """
        Compute A^T @ b operation using compute-optimized CUDA kernel (targeting ~14 GFLOPS).

        This method uses an architecture-matched CUDA kernel with 8-way parallelism
        targeting the SM architecture for maximum compute throughput and memory efficiency.

        Args:
            dense_matrix: Target image, shape (height, width)

        Returns:
            Result of A^T @ b, shape (batch_size, channels)
        """
        if dense_matrix.shape != (self.height, self.width):
            raise ValueError(
                f"dense_matrix shape {dense_matrix.shape} != expected "
                f"({self.height}, {self.width})"
            )

        try:
            from .cuda_kernels import cuda_transpose_dot_product_compute_optimized

            # Use compute-optimized CUDA kernel implementation
            result = cuda_transpose_dot_product_compute_optimized(
                self.sparse_values,
                self.block_positions,
                self.blocks_set,
                dense_matrix,
                self.batch_size,
                self.channels,
                self.block_size,
            )

            return result

        except ImportError as e:
            logger.warning(
                f"Compute-optimized CUDA kernel not available: {e}. Falling back to high-parallelism CUDA kernel."
            )
            # Fall back to high-parallelism CUDA kernel
            return self.transpose_dot_product_cuda_high_parallelism(dense_matrix)

    def _extract_dense_blocks_vectorized(
        self,
        dense_matrix: cp.ndarray,
        batch_indices: cp.ndarray,
        channel_indices: cp.ndarray,
    ) -> cp.ndarray:
        """
        Extract dense blocks from target image using vectorized indexing.

        Args:
            dense_matrix: Target image, shape (height, width)
            batch_indices: LED indices for this chunk
            channel_indices: Channel indices for this chunk

        Returns:
            Extracted blocks, shape (len(batch_indices), block_size, block_size)
        """
        # Get positions for this chunk
        top_left_rows = self.block_positions[batch_indices, channel_indices, 0]
        top_left_cols = self.block_positions[batch_indices, channel_indices, 1]

        # Create offset grids for block extraction
        row_offsets = cp.arange(self.block_size, dtype=cp.int32)
        col_offsets = cp.arange(self.block_size, dtype=cp.int32)

        # Broadcasting to get all pixel indices within blocks
        # Shape: (chunk_size, block_size, block_size)
        row_indices = top_left_rows[:, None, None] + row_offsets[None, :, None]
        col_indices = top_left_cols[:, None, None] + col_offsets[None, None, :]

        # Extract blocks using advanced indexing
        return dense_matrix[row_indices, col_indices]

    def to_dense(self, batch_idx: int, channel_idx: int) -> cp.ndarray:
        """
        Convert a specific sub-tensor to dense format for debugging/visualization.

        Args:
            batch_idx: LED index
            channel_idx: Channel index

        Returns:
            Dense tensor, shape (height, width)
        """
        if not self.blocks_set[batch_idx, channel_idx]:
            # Return zeros if block not set
            return cp.zeros((self.height, self.width), dtype=cp.float32)

        # Create dense tensor and place block
        dense = cp.zeros((self.height, self.width), dtype=cp.float32)

        top_row = int(self.block_positions[batch_idx, channel_idx, 0])
        top_col = int(self.block_positions[batch_idx, channel_idx, 1])

        dense[
            top_row : top_row + self.block_size, top_col : top_col + self.block_size
        ] = self.sparse_values[batch_idx, channel_idx]

        return dense

    def get_block_info(
        self, batch_idx: int, channel_idx: int
    ) -> Dict[str, Union[Tuple[int, int], cp.ndarray, bool]]:
        """
        Get information about a specific block.

        Args:
            batch_idx: LED index
            channel_idx: Channel index

        Returns:
            Dictionary with block information
        """
        is_set = bool(self.blocks_set[batch_idx, channel_idx])

        if not is_set:
            return {"is_set": False, "position": None, "values": None}

        position = (
            int(self.block_positions[batch_idx, channel_idx, 0]),
            int(self.block_positions[batch_idx, channel_idx, 1]),
        )

        return {
            "is_set": True,
            "position": position,
            "values": self.sparse_values[batch_idx, channel_idx].copy(),
        }

    def memory_info(self) -> Dict[str, Union[float, int]]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory information
        """
        # Calculate actual memory usage
        sparse_values_mb = self.sparse_values.nbytes / (1024 * 1024)
        positions_mb = self.block_positions.nbytes / (1024 * 1024)
        blocks_set_mb = self.blocks_set.nbytes / (1024 * 1024)
        total_mb = sparse_values_mb + positions_mb + blocks_set_mb

        # Calculate what equivalent dense storage would be
        dense_mb = (self.batch_size * self.channels * self.height * self.width * 4) / (
            1024 * 1024
        )

        compression_ratio = total_mb / dense_mb
        blocks_stored = int(cp.sum(self.blocks_set))

        return {
            "sparse_values_mb": sparse_values_mb,
            "positions_mb": positions_mb,
            "blocks_set_mb": blocks_set_mb,
            "total_mb": total_mb,
            "equivalent_dense_mb": dense_mb,
            "compression_ratio": compression_ratio,
            "blocks_stored": blocks_stored,
            "total_possible_blocks": self.batch_size * self.channels,
        }

    def validate_consistency(self) -> bool:
        """
        Validate internal consistency of the tensor.

        Returns:
            True if consistent, False otherwise
        """
        # Check that all set blocks have valid positions
        set_mask = self.blocks_set
        if cp.any(set_mask):
            set_positions = self.block_positions[set_mask]

            # Check row bounds
            if cp.any(set_positions[:, 0] < 0) or cp.any(
                set_positions[:, 0] > self.height - self.block_size
            ):
                logger.error("Some block positions have invalid row coordinates")
                return False

            # Check column bounds
            if cp.any(set_positions[:, 1] < 0) or cp.any(
                set_positions[:, 1] > self.width - self.block_size
            ):
                logger.error("Some block positions have invalid column coordinates")
                return False

        logger.debug("Tensor consistency validation passed")
        return True

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Export tensor data to a dictionary of numpy arrays for saving to npz files.

        Returns:
            Dictionary containing all tensor data as numpy arrays
        """
        # Convert CuPy arrays to NumPy for disk storage
        data_dict = {
            "sparse_values": cp.asnumpy(self.sparse_values),
            "block_positions": cp.asnumpy(self.block_positions),
            "blocks_set": cp.asnumpy(self.blocks_set),
            # Metadata
            "batch_size": np.array(self.batch_size, dtype=np.int32),
            "channels": np.array(self.channels, dtype=np.int32),
            "height": np.array(self.height, dtype=np.int32),
            "width": np.array(self.width, dtype=np.int32),
            "block_size": np.array(self.block_size, dtype=np.int32),
            "device": np.array(self.device, dtype="U10"),  # Unicode string
        }

        logger.debug(f"Exported tensor to dict with {len(data_dict)} arrays")
        return data_dict

    @classmethod
    def from_dict(
        cls, data_dict: Dict[str, np.ndarray], device: str = "cuda"
    ) -> "SingleBlockMixedSparseTensor":
        """
        Create tensor from a dictionary of numpy arrays (loaded from npz files).

        Args:
            data_dict: Dictionary containing tensor data as numpy arrays
            device: Device to create tensor on ('cuda' or 'cpu')

        Returns:
            New SingleBlockMixedSparseTensor instance
        """
        # Extract metadata
        batch_size = int(data_dict["batch_size"])
        channels = int(data_dict["channels"])
        height = int(data_dict["height"])
        width = int(data_dict["width"])
        block_size = int(data_dict["block_size"])

        # Create new tensor instance
        tensor = cls(batch_size, channels, height, width, block_size, device)

        # Load data arrays, converting to CuPy if needed
        tensor.sparse_values = cp.asarray(data_dict["sparse_values"])
        tensor.block_positions = cp.asarray(data_dict["block_positions"])
        tensor.blocks_set = cp.asarray(data_dict["blocks_set"])

        # Validate consistency
        if not tensor.validate_consistency():
            raise ValueError("Loaded tensor data is inconsistent")

        blocks_loaded = int(cp.sum(tensor.blocks_set))
        logger.debug(f"Loaded tensor from dict with {blocks_loaded} blocks")

        return tensor

    def __repr__(self) -> str:
        """String representation of the tensor."""
        blocks_stored = int(cp.sum(self.blocks_set))
        memory_mb = self.memory_info()["total_mb"]

        return (
            f"SingleBlockMixedSparseTensor("
            f"shape=({self.batch_size}, {self.channels}, {self.height}, {self.width}), "
            f"block_size={self.block_size}, "
            f"blocks_stored={blocks_stored}/{self.batch_size * self.channels}, "
            f"memory={memory_mb:.1f}MB)"
        )
