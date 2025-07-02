"""
LED Diffusion CSC Matrix Wrapper - Encapsulates CSC matrix storage and operations.

This module implements a specialized wrapper for CSC sparse matrices used in LED
diffusion pattern storage. It provides convenient methods for working with LED
patterns while encapsulating the underlying storage format.

Key features:
- Encapsulates CSC matrix storage (data, indices, indptr)
- Dictionary serialization for saving/loading
- Dense materialization for individual LED/channel patterns
- Bounding box calculation and region extraction
- Image setting from dense arrays
- Horizontal stacking for matrix composition
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class LEDDiffusionCSCMatrix:
    """
    Wrapper for CSC sparse matrices used in LED diffusion patterns.

    This class encapsulates the storage and manipulation of sparse matrices
    representing LED diffusion patterns. Each column represents one LED/channel
    combination, with the matrix mapping from pixel positions to LED intensities.

    Matrix format:
    - Shape: (pixels, led_count * channels)
    - pixels = height * width (flattened spatial dimensions)
    - Columns organized as: [LED0_R, LED0_G, LED0_B, LED1_R, LED1_G, LED1_B, ...]

    Example:
        # Create from existing CSC matrix
        wrapper = LEDDiffusionCSCMatrix.from_csc_matrix(csc_matrix, height=480, width=800)

        # Get dense pattern for LED 5, red channel
        pattern = wrapper.materialize_dense(5, 0)  # Returns (480, 800) array

        # Get bounding box of non-zero region
        bbox = wrapper.get_bounding_box(5, 0)  # Returns (min_row, min_col, max_row, max_col)

        # Save to dictionary
        data_dict = wrapper.to_dict()
    """

    def __init__(
        self,
        csc_matrix: sp.csc_matrix,
        height: int,
        width: int,
        channels: int = 3,
    ):
        """
        Initialize LED diffusion CSC matrix wrapper.

        Args:
            csc_matrix: Scipy CSC sparse matrix
            height: Spatial height of images
            width: Spatial width of images
            channels: Number of color channels (typically 3 for RGB)
        """
        self.matrix = csc_matrix.copy().astype(np.float32)
        self.height = height
        self.width = width
        self.channels = channels

        # Validate inputs
        self._validate_inputs()

        # Calculate derived properties
        self.pixels = height * width
        self.led_count = self.matrix.shape[1] // channels

        logger.debug(
            f"LEDDiffusionCSCMatrix created: "
            f"shape={self.matrix.shape}, led_count={self.led_count}, "
            f"channels={channels}, nnz={self.matrix.nnz}"
        )

    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix shape (pixels, led_count * channels)."""
        return self.matrix.shape

    @property
    def data(self) -> np.ndarray:
        """CSC matrix data array (non-zero values)."""
        return self.matrix.data

    @property
    def indices(self) -> np.ndarray:
        """CSC matrix indices array (row indices of non-zero values)."""
        return self.matrix.indices

    @property
    def indptr(self) -> np.ndarray:
        """CSC matrix index pointer array (column boundaries)."""
        return self.matrix.indptr

    def _validate_inputs(self) -> None:
        """Validate constructor inputs for consistency."""
        if self.matrix.shape[0] != self.height * self.width:
            raise ValueError(f"Matrix rows {self.matrix.shape[0]} != height * width {self.height * self.width}")

        if self.matrix.shape[1] % self.channels != 0:
            raise ValueError(f"Matrix columns {self.matrix.shape[1]} not divisible by channels {self.channels}")

        if not sp.isspmatrix_csc(self.matrix):
            raise ValueError("Matrix must be in CSC format")

    @classmethod
    def from_csc_matrix(
        cls,
        csc_matrix: sp.csc_matrix,
        height: int,
        width: int,
        channels: int = 3,
    ) -> "LEDDiffusionCSCMatrix":
        """
        Create wrapper from existing scipy CSC matrix.

        Args:
            csc_matrix: Existing CSC matrix
            height: Spatial height of images
            width: Spatial width of images
            channels: Number of color channels

        Returns:
            New LEDDiffusionCSCMatrix instance
        """
        return cls(
            csc_matrix=csc_matrix,
            height=height,
            width=width,
            channels=channels,
        )

    @classmethod
    def from_arrays(
        cls,
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: Tuple[int, int],
        height: int,
        width: int,
        channels: int = 3,
    ) -> "LEDDiffusionCSCMatrix":
        """
        Create wrapper from CSC matrix component arrays (backward compatibility).

        Args:
            data: CSC matrix data array (non-zero values)
            indices: CSC matrix indices array (row indices of non-zero values)
            indptr: CSC matrix index pointer array (column boundaries)
            shape: Matrix shape (pixels, led_count * channels)
            height: Spatial height of images
            width: Spatial width of images
            channels: Number of color channels (typically 3 for RGB)

        Returns:
            New LEDDiffusionCSCMatrix instance
        """
        # Create CSC matrix from component arrays
        csc_matrix = sp.csc_matrix((data, indices, indptr), shape=shape, dtype=np.float32)

        return cls(
            csc_matrix=csc_matrix,
            height=height,
            width=width,
            channels=channels,
        )

    def to_csc_matrix(self) -> sp.csc_matrix:
        """
        Convert to scipy CSC matrix.

        Returns:
            scipy.sparse.csc_matrix with the stored data
        """
        return self.matrix.copy()

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Export matrix data to dictionary for saving to npz files.

        Returns:
            Dictionary containing all matrix data and metadata
        """
        return {
            "csc_data": self.matrix.data,
            "csc_indices": self.matrix.indices,
            "csc_indptr": self.matrix.indptr,
            "csc_shape": np.array(self.matrix.shape, dtype=np.int32),
            "csc_height": np.array(self.height, dtype=np.int32),
            "csc_width": np.array(self.width, dtype=np.int32),
            "csc_channels": np.array(self.channels, dtype=np.int32),
            "csc_led_count": np.array(self.led_count, dtype=np.int32),
            "csc_nnz": np.array(self.matrix.nnz, dtype=np.int32),
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, np.ndarray]) -> "LEDDiffusionCSCMatrix":
        """
        Create matrix from dictionary (loaded from npz files).

        Args:
            data_dict: Dictionary containing matrix data and metadata

        Returns:
            New LEDDiffusionCSCMatrix instance
        """
        # Reconstruct CSC matrix from component arrays
        csc_matrix = sp.csc_matrix(
            (data_dict["csc_data"], data_dict["csc_indices"], data_dict["csc_indptr"]),
            shape=tuple(data_dict["csc_shape"]),
            dtype=np.float32,
        )

        return cls(
            csc_matrix=csc_matrix,
            height=int(data_dict["csc_height"]),
            width=int(data_dict["csc_width"]),
            channels=int(data_dict["csc_channels"]),
        )

    def materialize_dense(self, led_idx: int, channel: int) -> np.ndarray:
        """
        Materialize dense form of a single LED/channel pattern.

        Args:
            led_idx: LED index (0 to led_count-1)
            channel: Channel index (0 to channels-1)

        Returns:
            Dense array of shape (height, width) with the LED pattern
        """
        if not (0 <= led_idx < self.led_count):
            raise ValueError(f"LED index {led_idx} out of range [0, {self.led_count})")
        if not (0 <= channel < self.channels):
            raise ValueError(f"Channel {channel} out of range [0, {self.channels})")

        # Calculate column index
        col_idx = led_idx * self.channels + channel

        # Extract column using csc_matrix slicing (returns column as sparse matrix)
        column = self.matrix[:, col_idx]

        # Convert to dense array and reshape to 2D spatial dimensions
        # column.toarray() returns shape (rows, 1), so we flatten it
        dense_column = column.toarray().flatten()
        return dense_column.reshape(self.height, self.width)

    def get_bounding_box(self, led_idx: int, channel: int) -> Tuple[int, int, int, int]:
        """
        Get bounding box of non-zero region for a LED/channel pattern.

        Args:
            led_idx: LED index (0 to led_count-1)
            channel: Channel index (0 to channels-1)

        Returns:
            Tuple of (min_row, min_col, max_row, max_col) or None if no non-zeros
        """
        if not (0 <= led_idx < self.led_count):
            raise ValueError(f"LED index {led_idx} out of range [0, {self.led_count})")
        if not (0 <= channel < self.channels):
            raise ValueError(f"Channel {channel} out of range [0, {self.channels})")

        # Calculate column index
        col_idx = led_idx * self.channels + channel

        # Extract column using csc_matrix slicing
        column = self.matrix[:, col_idx]

        # Get non-zero indices using the nonzero() method
        row_indices, _ = column.nonzero()

        if len(row_indices) == 0:
            # No non-zero values
            return (0, 0, 0, 0)

        # Convert linear indices to 2D coordinates
        # row_indices are the flattened pixel positions
        row_coords = row_indices // self.width
        col_coords = row_indices % self.width

        # Calculate bounding box
        min_row, max_row = row_coords.min(), row_coords.max()
        min_col, max_col = col_coords.min(), col_coords.max()

        return (min_row, min_col, max_row, max_col)

    def extract_region(
        self,
        led_idx: int,
        channel: int,
        min_row: int,
        min_col: int,
        max_row: int,
        max_col: int,
    ) -> np.ndarray:
        """
        Extract rectangular region of LED/channel pattern in dense form.

        Args:
            led_idx: LED index (0 to led_count-1)
            channel: Channel index (0 to channels-1)
            min_row: Minimum row (inclusive)
            min_col: Minimum column (inclusive)
            max_row: Maximum row (inclusive)
            max_col: Maximum column (inclusive)

        Returns:
            Dense array of shape (max_row - min_row + 1, max_col - min_col + 1)
        """
        # Validate region bounds
        if not (0 <= min_row <= max_row < self.height):
            raise ValueError(f"Invalid row range [{min_row}, {max_row}] for height {self.height}")
        if not (0 <= min_col <= max_col < self.width):
            raise ValueError(f"Invalid col range [{min_col}, {max_col}] for width {self.width}")

        # Get full dense pattern
        dense = self.materialize_dense(led_idx, channel)

        # Extract region
        return dense[min_row : max_row + 1, min_col : max_col + 1]

    def set_image(
        self,
        led_idx: int,
        channel: int,
        dense_image: np.ndarray,
        sparsity_threshold: float = 1e-6,
    ) -> None:
        """
        Set LED/channel pattern from dense image array.

        Args:
            led_idx: LED index (0 to led_count-1)
            channel: Channel index (0 to channels-1)
            dense_image: Dense array of shape (height, width)
            sparsity_threshold: Values below this are treated as zero
        """
        if not (0 <= led_idx < self.led_count):
            raise ValueError(f"LED index {led_idx} out of range [0, {self.led_count})")
        if not (0 <= channel < self.channels):
            raise ValueError(f"Channel {channel} out of range [0, {self.channels})")

        if dense_image.shape != (self.height, self.width):
            raise ValueError(f"Dense image shape {dense_image.shape} != expected ({self.height}, {self.width})")

        # Calculate column index
        col_idx = led_idx * self.channels + channel

        # Flatten and apply threshold
        flat_image = dense_image.reshape(-1).astype(np.float32)
        significant_mask = np.abs(flat_image) > sparsity_threshold

        # Create sparse column vector from thresholded data
        if np.any(significant_mask):
            significant_indices = np.where(significant_mask)[0]
            significant_values = flat_image[significant_mask]

            # Create new sparse column vector
            new_column = sp.csc_matrix(
                (
                    significant_values,
                    (
                        significant_indices,
                        np.zeros(len(significant_indices), dtype=int),
                    ),
                ),
                shape=(self.matrix.shape[0], 1),
                dtype=np.float32,
            )
        else:
            # Empty column
            new_column = sp.csc_matrix((self.matrix.shape[0], 1), dtype=np.float32)

        # Replace the column in the matrix
        # This creates a new matrix by horizontally stacking parts before and after the column
        if col_idx == 0:
            # Replacing first column
            if self.matrix.shape[1] > 1:
                self.matrix = sp.hstack([new_column, self.matrix[:, 1:]], format="csc")
            else:
                self.matrix = new_column
        elif col_idx == self.matrix.shape[1] - 1:
            # Replacing last column
            self.matrix = sp.hstack([self.matrix[:, :-1], new_column], format="csc")
        else:
            # Replacing middle column
            self.matrix = sp.hstack(
                [self.matrix[:, :col_idx], new_column, self.matrix[:, col_idx + 1 :]],
                format="csc",
            )

    @classmethod
    def hstack(cls, matrices: List["LEDDiffusionCSCMatrix"]) -> "LEDDiffusionCSCMatrix":
        """
        Horizontally stack multiple LED diffusion matrices.

        Args:
            matrices: List of LEDDiffusionCSCMatrix instances to stack

        Returns:
            New matrix with horizontally stacked columns
        """
        if not matrices:
            raise ValueError("Cannot stack empty list of matrices")

        # Validate all matrices have same spatial dimensions
        first = matrices[0]
        for i, matrix in enumerate(matrices[1:], 1):
            if matrix.height != first.height or matrix.width != first.width:
                raise ValueError(
                    f"Matrix {i} spatial dims ({matrix.height}, {matrix.width}) != "
                    f"first matrix ({first.height}, {first.width})"
                )
            if matrix.channels != first.channels:
                raise ValueError(f"Matrix {i} channels {matrix.channels} != first matrix {first.channels}")

        # Use scipy hstack directly on the stored matrices
        csc_matrices = [matrix.matrix for matrix in matrices]
        stacked_csc = sp.hstack(csc_matrices, format="csc")

        # Create new wrapper
        return cls(
            csc_matrix=stacked_csc,
            height=first.height,
            width=first.width,
            channels=first.channels,
        )

    def memory_info(self) -> Dict[str, Union[float, int]]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory information
        """
        data_mb = self.matrix.data.nbytes / (1024 * 1024)
        indices_mb = self.matrix.indices.nbytes / (1024 * 1024)
        indptr_mb = self.matrix.indptr.nbytes / (1024 * 1024)
        total_mb = data_mb + indices_mb + indptr_mb

        # Calculate equivalent dense storage
        dense_mb = (self.matrix.shape[0] * self.matrix.shape[1] * 4) / (1024 * 1024)  # float32

        sparsity_ratio = self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1])

        return {
            "data_mb": data_mb,
            "indices_mb": indices_mb,
            "indptr_mb": indptr_mb,
            "total_mb": total_mb,
            "equivalent_dense_mb": dense_mb,
            "compression_ratio": total_mb / dense_mb,
            "sparsity_ratio": sparsity_ratio,
            "nnz": self.matrix.nnz,
            "shape": self.matrix.shape,
        }

    def to_dense_patterns(self) -> np.ndarray:
        """
        Convert entire matrix to dense patterns array for visualization.

        This method converts the sparse CSC matrix representation back to a dense
        4D array suitable for visualization and analysis. Each LED pattern is
        materialized as a (height, width, channels) array.

        Returns:
            Dense patterns array of shape (led_count, height, width, channels)
            with dtype float32 and values in range [0, 1]
        """
        patterns = np.zeros((self.led_count, self.height, self.width, self.channels), dtype=np.float32)

        logger.debug(f"Converting {self.led_count} LED patterns to dense format...")

        for led_idx in range(self.led_count):
            for channel in range(self.channels):
                # Use existing materialize_dense method for each LED/channel
                patterns[led_idx, :, :, channel] = self.materialize_dense(led_idx, channel)

        logger.debug(f"Converted to dense patterns: {patterns.shape}")
        return patterns

    def get_pattern_summary(self) -> Dict[str, Union[float, int, List, np.ndarray]]:
        """
        Get comprehensive summary statistics for all LED patterns.

        Computes various statistics across all LEDs and channels including
        sparsity, intensity distributions, and spatial characteristics.

        Returns:
            Dictionary containing:
            - led_count: Number of LEDs
            - channels: Number of color channels
            - matrix_shape: Shape of underlying sparse matrix
            - nnz_total: Total non-zero entries
            - sparsity_ratio: Fraction of non-zero entries
            - max_intensities: Max intensity per LED (shape: led_count,)
            - mean_intensities: Mean intensity per LED (shape: led_count,)
            - pattern_extents: Bounding box sizes per LED (shape: led_count,)
            - channel_nnz: Non-zero counts per channel
            - memory_mb: Memory usage in megabytes
        """
        logger.debug("Computing pattern summary statistics...")

        # Basic matrix info
        summary = {
            "led_count": self.led_count,
            "channels": self.channels,
            "matrix_shape": list(self.matrix.shape),
            "nnz_total": self.matrix.nnz,
            "sparsity_ratio": self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1]),
        }

        # Per-LED statistics
        max_intensities = np.zeros(self.led_count, dtype=np.float32)
        mean_intensities = np.zeros(self.led_count, dtype=np.float32)
        pattern_extents = np.zeros(self.led_count, dtype=np.float32)  # Bounding box areas

        # Per-channel statistics
        channel_nnz = np.zeros(self.channels, dtype=int)

        for led_idx in range(self.led_count):
            led_max = 0.0
            led_total = 0.0
            led_pixels = 0

            # Process each channel for this LED
            for channel in range(self.channels):
                col_idx = led_idx * self.channels + channel

                # Get column data using CSC matrix slicing
                column = self.matrix[:, col_idx]
                if column.nnz > 0:
                    values = column.data
                    channel_nnz[channel] += column.nnz

                    # Update LED statistics
                    led_max = max(led_max, float(np.max(values)))
                    led_total += float(np.sum(values))
                    led_pixels += column.nnz

            max_intensities[led_idx] = led_max
            mean_intensities[led_idx] = led_total / max(led_pixels, 1)

            # Calculate bounding box area (use first channel as representative)
            bbox = self.get_bounding_box(led_idx, 0)
            if bbox != (0, 0, 0, 0):
                min_row, min_col, max_row, max_col = bbox
                area = (max_row - min_row + 1) * (max_col - min_col + 1)
                pattern_extents[led_idx] = area

        # Add computed statistics
        summary.update(
            {
                "max_intensities": max_intensities,
                "mean_intensities": mean_intensities,
                "pattern_extents": pattern_extents,
                "channel_nnz": channel_nnz,
                "memory_mb": self.memory_info()["total_mb"],
            }
        )

        logger.debug("Pattern summary computed")
        return summary

    def get_led_bounding_boxes(self) -> np.ndarray:
        """
        Get bounding boxes for all LEDs across all channels.

        Computes the bounding box for each LED by finding the union of bounding
        boxes across all color channels. This gives the overall spatial extent
        of each LED's diffusion pattern.

        Returns:
            Array of shape (led_count, 4) containing bounding boxes as
            [min_row, min_col, max_row, max_col] for each LED.
            Returns [0, 0, 0, 0] for LEDs with no non-zero patterns.
        """
        bboxes = np.zeros((self.led_count, 4), dtype=int)

        logger.debug(f"Computing bounding boxes for {self.led_count} LEDs...")

        for led_idx in range(self.led_count):
            # Find union of bounding boxes across all channels
            min_row, min_col = self.height, self.width
            max_row, max_col = -1, -1
            has_data = False

            for channel in range(self.channels):
                bbox = self.get_bounding_box(led_idx, channel)
                if bbox != (0, 0, 0, 0):  # Has non-zero data
                    ch_min_row, ch_min_col, ch_max_row, ch_max_col = bbox
                    min_row = min(min_row, ch_min_row)
                    min_col = min(min_col, ch_min_col)
                    max_row = max(max_row, ch_max_row)
                    max_col = max(max_col, ch_max_col)
                    has_data = True

            if has_data:
                bboxes[led_idx] = [min_row, min_col, max_row, max_col]
            else:
                bboxes[led_idx] = [0, 0, 0, 0]  # No data for this LED

        logger.debug(f"Computed bounding boxes for {self.led_count} LEDs")
        return bboxes

    def extract_rgb_channels(
        self,
    ) -> Tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix]:
        """
        Extract separate RGB channel matrices from the combined matrix.

        Returns:
            Tuple of (R_matrix, G_matrix, B_matrix) where each matrix has shape
            (pixels, led_count) containing only that channel's data.
        """
        logger.debug("Extracting RGB channel matrices...")

        # Extract each channel using column slicing
        R_matrix = self.matrix[:, 0 :: self.channels]  # Red channels
        G_matrix = self.matrix[:, 1 :: self.channels]  # Green channels
        B_matrix = self.matrix[:, 2 :: self.channels]  # Blue channels

        logger.debug(f"Extracted RGB matrices with shape: {R_matrix.shape}")
        return R_matrix, G_matrix, B_matrix

    def create_block_diagonal_matrix(self) -> sp.csc_matrix:
        """
        Create a block diagonal matrix for efficient combined RGB operations.

        The resulting matrix has shape (pixels * 3, led_count * 3) with structure:
        [R_matrix    0         0      ]
        [0         G_matrix    0      ]
        [0           0       B_matrix ]

        This enables vectorized A^T@b calculations across all RGB channels.

        Returns:
            Block diagonal CSC matrix combining all RGB channels
        """
        logger.debug("Creating block diagonal matrix for RGB channels...")

        # Extract RGB channel matrices
        R_matrix, G_matrix, B_matrix = self.extract_rgb_channels()

        # Create block diagonal matrix
        from scipy.sparse import block_diag

        block_diagonal = block_diag([R_matrix, G_matrix, B_matrix], format="csc")

        logger.debug(f"Created block diagonal matrix with shape: {block_diagonal.shape}")
        return block_diagonal

    def to_gpu_matrices(self):
        """
        Transfer RGB channel matrices to GPU using CuPy.

        Returns:
            Tuple of (R_gpu, G_gpu, B_gpu, combined_gpu) CuPy sparse matrices
        """
        try:
            import cupy as cp
            from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix
        except ImportError:
            raise ImportError("CuPy is required for GPU matrix operations")

        logger.debug("Transferring matrices to GPU...")

        # Extract RGB channels
        R_matrix, G_matrix, B_matrix = self.extract_rgb_channels()

        # Create block diagonal matrix
        combined_matrix = self.create_block_diagonal_matrix()

        # Transfer to GPU
        R_gpu = cupy_csc_matrix(R_matrix)
        G_gpu = cupy_csc_matrix(G_matrix)
        B_gpu = cupy_csc_matrix(B_matrix)
        combined_gpu = cupy_csc_matrix(combined_matrix)

        logger.debug("Successfully transferred matrices to GPU")
        return R_gpu, G_gpu, B_gpu, combined_gpu

    def __repr__(self) -> str:
        """String representation of the matrix."""
        memory_mb = self.memory_info()["total_mb"]
        sparsity = self.memory_info()["sparsity_ratio"] * 100

        return (
            f"LEDDiffusionCSCMatrix("
            f"shape={self.matrix.shape}, "
            f"led_count={self.led_count}, "
            f"channels={self.channels}, "
            f"nnz={self.matrix.nnz:,}, "
            f"sparsity={sparsity:.2f}%, "
            f"memory={memory_mb:.1f}MB)"
        )
