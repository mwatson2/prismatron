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
        # Shape: (channels, batch_size, block_size, block_size) for planar layout
        self.sparse_values = cp.zeros(
            (channels, batch_size, block_size, block_size), dtype=cp.float32
        )

        # Storage for block positions (top-left coordinates)
        # Shape: (channels, batch_size, 2) for planar layout
        self.block_positions = cp.zeros((channels, batch_size, 2), dtype=cp.int32)

        # Note: All blocks are assumed to be set - no tracking needed

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

        # Store the block data and position (channels-first indexing)
        self.sparse_values[channel_idx, batch_idx] = values
        self.block_positions[channel_idx, batch_idx, 0] = top_left_row
        self.block_positions[channel_idx, batch_idx, 1] = top_left_col

    def set_blocks_batch(self, positions: cp.ndarray, values: cp.ndarray) -> None:
        """
        Set multiple blocks efficiently in batch.

        Args:
            positions: Block positions, shape (channels, batch_size, 2)
                      positions[c, i, :] = [top_left_row, top_left_col]
            values: Block values, shape (channels, batch_size, block_size, block_size)
        """
        if positions.shape != (self.channels, self.batch_size, 2):
            raise ValueError(
                f"positions shape {positions.shape} != expected "
                f"({self.channels}, {self.batch_size}, 2)"
            )
        if values.shape != (
            self.channels,
            self.batch_size,
            self.block_size,
            self.block_size,
        ):
            raise ValueError(
                f"values shape {values.shape} != expected "
                f"({self.channels}, {self.batch_size}, {self.block_size}, {self.block_size})"
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

        # Result tensor - channels first for planar layout
        results = cp.zeros((self.channels, self.batch_size), dtype=cp.float32)

        # Process in chunks to manage memory usage
        total_elements = self.batch_size * self.channels

        for chunk_start in range(0, total_elements, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_elements)

            # Convert flat indices to (channel, batch) indices for new layout
            flat_indices = cp.arange(chunk_start, chunk_end)
            channel_indices = flat_indices // self.batch_size
            batch_indices = flat_indices % self.batch_size

            # Extract dense blocks from target image for this chunk
            dense_blocks = self._extract_dense_blocks_vectorized(
                dense_matrix, batch_indices, channel_indices
            )

            # Get corresponding sparse blocks (channels-first indexing)
            sparse_blocks = self.sparse_values[channel_indices, batch_indices]

            # Compute element-wise multiplication and sum
            # Shape: (chunk_size, block_size, block_size) -> (chunk_size,)
            chunk_results = cp.sum(sparse_blocks * dense_blocks, axis=(1, 2))

            # Store results back to proper positions (channels-first)
            results[channel_indices, batch_indices] = chunk_results

        # Transpose to maintain backward compatibility:
        # (channels, batch_size) -> (batch_size, channels)
        return results.T

    def transpose_dot_product_3d(self, target_3d: cp.ndarray) -> cp.ndarray:
        """
        Compute A^T @ b operation with 3D planar input (channels, height, width).

        This method processes all channels in one operation using the optimized 3D CUDA kernel.
        Implements einsum 'ijkl,jkl->ij' efficiently where:
        - Mixed tensor: (leds, channels, height, width) - shape 'ijkl'
        - Target: (channels, height, width) - shape 'jkl' (planar form)
        - Result: (leds, channels) - shape 'ij'

        Args:
            target_3d: Target image in planar form, shape (channels, height, width)

        Returns:
            Result of A^T @ b, shape (batch_size, channels)
        """
        if target_3d.shape != (self.channels, self.height, self.width):
            raise ValueError(
                f"target_3d shape {target_3d.shape} != expected "
                f"({self.channels}, {self.height}, {self.width})"
            )

        try:
            from .cuda_kernels import cuda_transpose_dot_product_3d_compute_optimized

            # Use 3D compute-optimized CUDA kernel - no blocks_set parameter needed
            # Storage format matches kernel expectations: (channels, batch, ...)
            result = cuda_transpose_dot_product_3d_compute_optimized(
                self.sparse_values,  # (channels, batch, H, W) - matches kernel expectation
                self.block_positions,  # (channels, batch, 2) - matches kernel expectation
                target_3d,  # (channels, height, width) - planar input
                self.batch_size,
                self.channels,
                self.block_size,
            )

            return result

        except ImportError as e:
            logger.warning(
                f"3D CUDA kernel not available: {e}. Falling back to chunked implementation."
            )
            # Fall back to per-channel processing with chunked implementation
            results = cp.zeros((self.batch_size, self.channels), dtype=cp.float32)
            for channel_idx in range(self.channels):
                channel_result = self.transpose_dot_product(target_3d[channel_idx])
                results[:, channel_idx] = channel_result[:, channel_idx]
            return results

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
        # Get positions for this chunk (channels-first indexing)
        top_left_rows = self.block_positions[channel_indices, batch_indices, 0]
        top_left_cols = self.block_positions[channel_indices, batch_indices, 1]

        # Create offset grids for block extraction
        row_offsets = cp.arange(self.block_size, dtype=cp.int32)
        col_offsets = cp.arange(self.block_size, dtype=cp.int32)

        # Broadcasting to get all pixel indices within blocks
        # Shape: (chunk_size, block_size, block_size)
        row_indices = top_left_rows[:, None, None] + row_offsets[None, :, None]
        col_indices = top_left_cols[:, None, None] + col_offsets[None, None, :]

        # Extract blocks using advanced indexing
        return dense_matrix[row_indices, col_indices]

    def to_array(self, batch_idx: int, channel_idx: int) -> cp.ndarray:
        """
        Convert a specific sub-tensor to dense array format for debugging/visualization.

        Args:
            batch_idx: LED index
            channel_idx: Channel index

        Returns:
            Dense array, shape (height, width)
        """
        # Create dense array and place block (all blocks assumed to be set)
        dense = cp.zeros((self.height, self.width), dtype=cp.float32)

        top_row = int(
            self.block_positions[channel_idx, batch_idx, 0]
        )  # Use channels-first indexing
        top_col = int(
            self.block_positions[channel_idx, batch_idx, 1]
        )  # Use channels-first indexing

        dense[
            top_row : top_row + self.block_size, top_col : top_col + self.block_size
        ] = self.sparse_values[
            channel_idx, batch_idx
        ]  # Use channels-first indexing

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
        # All blocks are assumed to be set
        position = (
            int(
                self.block_positions[channel_idx, batch_idx, 0]
            ),  # Use channels-first indexing
            int(
                self.block_positions[channel_idx, batch_idx, 1]
            ),  # Use channels-first indexing
        )

        return {
            "is_set": True,
            "position": position,
            "values": self.sparse_values[
                channel_idx, batch_idx
            ].copy(),  # Use channels-first indexing
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
        total_mb = sparse_values_mb + positions_mb

        # Calculate what equivalent dense storage would be
        dense_mb = (self.batch_size * self.channels * self.height * self.width * 4) / (
            1024 * 1024
        )

        compression_ratio = total_mb / dense_mb
        blocks_stored = self.batch_size * self.channels  # All blocks assumed to be set

        return {
            "sparse_values_mb": sparse_values_mb,
            "positions_mb": positions_mb,
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
        # Check that all block positions are valid (all blocks assumed to be set)
        # Check row bounds for all positions
        if cp.any(self.block_positions[:, :, 0] < 0) or cp.any(
            self.block_positions[:, :, 0] > self.height - self.block_size
        ):
            logger.error("Some block positions have invalid row coordinates")
            return False

        # Check column bounds for all positions
        if cp.any(self.block_positions[:, :, 1] < 0) or cp.any(
            self.block_positions[:, :, 1] > self.width - self.block_size
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

        # Handle backward compatibility: if blocks_set exists in data, ignore it
        # All blocks are now assumed to be set

        # Validate consistency
        if not tensor.validate_consistency():
            raise ValueError("Loaded tensor data is inconsistent")

        blocks_loaded = (
            tensor.batch_size * tensor.channels
        )  # All blocks assumed to be set
        logger.debug(f"Loaded tensor from dict with {blocks_loaded} blocks")

        return tensor

    def to_dense_patterns(self) -> np.ndarray:
        """
        Convert mixed tensor to dense patterns array for visualization.

        This method converts the sparse block representation back to dense
        4D arrays suitable for visualization and analysis. Each LED/channel
        pattern is materialized as a (height, width) array.

        Returns:
            Dense patterns array of shape (batch_size, height, width, channels)
            with dtype float32 and values in range [0, 1]
        """
        # Initialize dense patterns array (HWC format for compatibility)
        patterns = np.zeros(
            (self.batch_size, self.height, self.width, self.channels), dtype=np.float32
        )

        logger.debug(
            f"Converting {self.batch_size} LED patterns from mixed tensor to dense format..."
        )

        # Convert each block to dense format
        for batch_idx in range(self.batch_size):
            for channel_idx in range(self.channels):
                # Get block position and values (channels-first indexing)
                top_row = int(self.block_positions[channel_idx, batch_idx, 0])
                top_col = int(self.block_positions[channel_idx, batch_idx, 1])
                block_values = cp.asnumpy(self.sparse_values[channel_idx, batch_idx])

                # Place block in dense pattern
                patterns[
                    batch_idx,
                    top_row : top_row + self.block_size,
                    top_col : top_col + self.block_size,
                    channel_idx,
                ] = block_values

        logger.debug(f"Converted to dense patterns: {patterns.shape}")
        return patterns

    def get_block_summary(self) -> Dict[str, Union[float, int, np.ndarray]]:
        """
        Get comprehensive summary statistics for all blocks.

        Computes various statistics across all LEDs and channels including
        block positions, intensity distributions, and coverage statistics.

        Returns:
            Dictionary containing:
            - batch_size: Number of LEDs
            - channels: Number of color channels
            - block_size: Size of each block (square)
            - total_blocks: Total number of blocks
            - block_positions_stats: Statistics about block placement
            - intensity_stats: Statistics about block values
            - coverage_stats: Statistics about spatial coverage
            - memory_mb: Memory usage in megabytes
        """
        logger.debug("Computing block summary statistics...")

        # Convert to numpy for statistics computation
        positions = cp.asnumpy(self.block_positions)  # (channels, batch_size, 2)
        values = cp.asnumpy(
            self.sparse_values
        )  # (channels, batch_size, block_size, block_size)

        # Basic info
        summary = {
            "batch_size": self.batch_size,
            "channels": self.channels,
            "block_size": self.block_size,
            "total_blocks": self.batch_size * self.channels,
        }

        # Block position statistics
        all_positions = positions.reshape(-1, 2)  # Flatten to (total_blocks, 2)
        position_stats = {
            "min_row": int(np.min(all_positions[:, 0])),
            "max_row": int(np.max(all_positions[:, 0])),
            "min_col": int(np.min(all_positions[:, 1])),
            "max_col": int(np.max(all_positions[:, 1])),
            "mean_row": float(np.mean(all_positions[:, 0])),
            "mean_col": float(np.mean(all_positions[:, 1])),
            "std_row": float(np.std(all_positions[:, 0])),
            "std_col": float(np.std(all_positions[:, 1])),
        }

        # Block intensity statistics
        all_values = values.reshape(-1)  # Flatten all block values
        nonzero_values = all_values[all_values > 0]
        intensity_stats = {
            "total_values": len(all_values),
            "nonzero_values": len(nonzero_values),
            "sparsity_ratio": len(nonzero_values) / len(all_values),
            "min_intensity": float(np.min(all_values)),
            "max_intensity": float(np.max(all_values)),
            "mean_intensity": float(np.mean(all_values)),
            "std_intensity": float(np.std(all_values)),
        }

        # Per-LED statistics
        led_max_intensities = np.zeros(self.batch_size, dtype=np.float32)
        led_mean_intensities = np.zeros(self.batch_size, dtype=np.float32)

        for batch_idx in range(self.batch_size):
            led_values = values[:, batch_idx].flatten()  # All channels for this LED
            led_max_intensities[batch_idx] = np.max(led_values)
            led_mean_intensities[batch_idx] = (
                np.mean(led_values[led_values > 0]) if np.any(led_values > 0) else 0.0
            )

        # Coverage statistics (spatial distribution)
        coverage_stats = {
            "blocks_per_led": self.channels,
            "spatial_coverage_x": (
                position_stats["max_col"] - position_stats["min_col"] + self.block_size
            )
            / self.width,
            "spatial_coverage_y": (
                position_stats["max_row"] - position_stats["min_row"] + self.block_size
            )
            / self.height,
            "led_max_intensities": led_max_intensities,
            "led_mean_intensities": led_mean_intensities,
        }

        # Add computed statistics
        summary.update(
            {
                "block_positions_stats": position_stats,
                "intensity_stats": intensity_stats,
                "coverage_stats": coverage_stats,
                "memory_mb": self.memory_info()["total_mb"],
            }
        )

        logger.debug("Block summary computed")
        return summary

    def extract_pattern(self, led_idx: int, channel_idx: int) -> np.ndarray:
        """
        Extract dense pattern for specific LED and channel.

        This method extracts and returns a single LED/channel pattern
        as a dense array, suitable for visualization or analysis.

        Args:
            led_idx: LED index (0 to batch_size-1)
            channel_idx: Channel index (0 to channels-1)

        Returns:
            Dense pattern array of shape (height, width) with dtype float32
            containing the LED pattern for the specified channel
        """
        # Validate inputs
        if not (0 <= led_idx < self.batch_size):
            raise ValueError(f"led_idx {led_idx} out of range [0, {self.batch_size})")
        if not (0 <= channel_idx < self.channels):
            raise ValueError(
                f"channel_idx {channel_idx} out of range [0, {self.channels})"
            )

        # Use existing to_array method but convert to numpy
        dense_pattern = cp.asnumpy(self.to_array(led_idx, channel_idx))

        logger.debug(
            f"Extracted pattern for LED {led_idx}, channel {channel_idx}: "
            f"shape {dense_pattern.shape}"
        )
        return dense_pattern

    def __repr__(self) -> str:
        """String representation of the tensor."""
        blocks_stored = self.batch_size * self.channels  # All blocks assumed to be set
        memory_mb = self.memory_info()["total_mb"]

        return (
            f"SingleBlockMixedSparseTensor("
            f"shape=({self.batch_size}, {self.channels}, {self.height}, {self.width}), "
            f"block_size={self.block_size}, "
            f"blocks_stored={blocks_stored}/{self.batch_size * self.channels}, "
            f"memory={memory_mb:.1f}MB)"
        )
