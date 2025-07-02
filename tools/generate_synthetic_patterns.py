#!/usr/bin/env python3
"""
Synthetic Diffusion Pattern Generator.

This tool generates synthetic diffusion patterns for LED optimization testing
and development. It's the centralized source for synthetic pattern generation
across the Prismatron system.

Usage:
    python generate_synthetic_patterns.py --output patterns.npz --led-count 3200
    python generate_synthetic_patterns.py --output test_patterns.npz --led-count 100 --seed 42
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
except ImportError:
    # Fallback for systems without CUDA
    import numpy as cp

# Add path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from src.utils.spatial_ordering import compute_rcm_ordering, reorder_matrix_columns

# Constants (can be overridden by command line)
DEFAULT_FRAME_WIDTH = 800
DEFAULT_FRAME_HEIGHT = 480
DEFAULT_LED_COUNT = 3200

logger = logging.getLogger(__name__)


class SyntheticPatternGenerator:
    """Generator for synthetic LED diffusion patterns with sparse matrix support."""

    def __init__(
        self,
        frame_width: int = DEFAULT_FRAME_WIDTH,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        seed: Optional[int] = None,
        sparsity_threshold: float = 0.01,
        block_size: int = 96,
    ):
        """
        Initialize pattern generator.

        Args:
            frame_width: Width of output frames
            frame_height: Height of output frames
            seed: Random seed for reproducible patterns
            sparsity_threshold: Threshold below which pixels are considered zero
            block_size: Size of diffusion blocks (e.g., 64, 96)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.seed = seed  # Store seed attribute
        self.sparsity_threshold = sparsity_threshold
        self.block_size = block_size

        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Using random seed: {seed}")

        # Generate LED positions (random for synthetic patterns)
        self.led_positions = None
        self.led_spatial_mapping = None
        self.reverse_spatial_mapping = None

    def generate_led_positions(self, led_count: int) -> np.ndarray:
        """
        Generate random LED positions (realistic hardware layout simulation).

        Args:
            led_count: Number of LEDs to position

        Returns:
            Array of LED positions (led_count, 2) with [x, y] coordinates
        """
        logger.info(f"Generating {led_count} random LED positions...")

        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate completely random positions within frame bounds
        # This simulates realistic hardware where LED positions are fixed
        margin = 20  # Small margin to avoid edge effects
        width_range = (margin, self.frame_width - margin)
        height_range = (margin, self.frame_height - margin)

        # Uniform random distribution across the frame
        x_positions = np.random.randint(width_range[0], width_range[1], led_count)
        y_positions = np.random.randint(height_range[0], height_range[1], led_count)

        positions = np.column_stack((x_positions, y_positions))
        self.led_positions = positions

        logger.info(f"Generated {led_count} random LED positions")

        # Don't create spatial mapping yet - will be done after RCM ordering
        self.led_spatial_mapping = None
        self.reverse_spatial_mapping = None

        return positions

    def morton_encode(self, x: float, y: float) -> int:
        """
        Convert 2D coordinates to Z-order (Morton) index for spatial locality.

        Args:
            x: X coordinate (normalized to frame width)
            y: Y coordinate (normalized to frame height)

        Returns:
            Morton-encoded integer for spatial ordering
        """
        # Normalize coordinates to [0, 1] and scale for precision
        x_norm = x / self.frame_width
        y_norm = y / self.frame_height
        x_int = int(x_norm * 65535)  # 16-bit precision
        y_int = int(y_norm * 65535)

        result = 0
        for i in range(16):  # 16-bit precision
            result |= (x_int & (1 << i)) << i | (y_int & (1 << i)) << (i + 1)
        return result

    def create_led_spatial_ordering(self, led_positions: np.ndarray) -> dict:
        """
        Create LED ordering based on spatial proximity using Z-order curve.

        Args:
            led_positions: Array of LED positions (led_count, 2)

        Returns:
            Dictionary mapping physical_led_id -> spatially_ordered_matrix_index
        """
        # Create list of (led_id, x, y, morton_code)
        led_list = []
        for led_id, (x, y) in enumerate(led_positions):
            morton_code = self.morton_encode(float(x), float(y))
            led_list.append((led_id, x, y, morton_code))

        # Sort by Morton code for spatial locality
        led_list.sort(key=lambda item: item[3])

        # Create mapping: physical_led_id -> spatially_ordered_matrix_index
        spatial_mapping = {
            led_id: matrix_idx for matrix_idx, (led_id, _, _, _) in enumerate(led_list)
        }

        logger.info(f"Created spatial ordering for {len(spatial_mapping)} LEDs")
        return spatial_mapping

    def generate_single_pattern(
        self,
        led_position: Tuple[int, int],
        block_position: Tuple[int, int],
        pattern_type: str = "gaussian_multi",
        base_intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Generate a single LED diffusion pattern with proper block position cropping.

        Args:
            led_position: LED position as (x, y) coordinates (center of LED)
            block_position: Block top-left position as (x, y) coordinates (aligned to multiple of 4)
            pattern_type: Type of pattern to generate
            base_intensity: Base intensity multiplier

        Returns:
            Pattern array (height, width, 3) for RGB channels
        """
        led_x, led_y = led_position
        block_x, block_y = block_position

        # Use block boundaries for cropping (consistent with adjacency calculation)
        crop_x_min = block_x
        crop_x_max = min(self.frame_width, block_x + self.block_size)
        crop_y_min = block_y
        crop_y_max = min(self.frame_height, block_y + self.block_size)

        # Create full-size pattern but only generate values in crop region
        pattern = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.float32)

        if pattern_type == "gaussian_multi":
            # Multiple Gaussian blobs for realistic diffusion
            for c in range(3):  # RGB channels
                channel_pattern = np.zeros(
                    (self.frame_height, self.frame_width), dtype=np.float32
                )

                # Create 2-4 Gaussian blobs per channel
                num_blobs = np.random.randint(2, 5)

                for _ in range(num_blobs):
                    # Random offset for sub-patterns (centered around LED position)
                    offset_x = np.random.normal(0, 15)
                    offset_y = np.random.normal(0, 15)
                    center_x = np.clip(led_x + offset_x, 0, self.frame_width - 1)
                    center_y = np.clip(led_y + offset_y, 0, self.frame_height - 1)

                    # Random sigma for different spread
                    sigma = np.random.uniform(8, 30)
                    intensity = np.random.uniform(0.3, 1.0) * base_intensity

                    # Create meshgrid only for crop region (block boundaries)
                    xx, yy = np.meshgrid(
                        np.arange(crop_x_min, crop_x_max) - center_x,
                        np.arange(crop_y_min, crop_y_max) - center_y,
                    )

                    # Gaussian pattern (cropped)
                    gaussian = intensity * np.exp(
                        -(xx**2 + yy**2) / (2 * sigma**2)
                    )
                    # Only update the crop region
                    channel_pattern[
                        crop_y_min:crop_y_max, crop_x_min:crop_x_max
                    ] += gaussian

                # Add some color variation between channels (only to cropped region)
                color_variation = np.random.uniform(0.7, 1.3)
                pattern[crop_y_min:crop_y_max, crop_x_min:crop_x_max, c] = (
                    channel_pattern[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                    * color_variation
                )

        elif pattern_type == "gaussian_simple":
            # Single Gaussian per channel (centered on LED, cropped to block)
            for c in range(3):
                sigma = np.random.uniform(15, 40)
                intensity = np.random.uniform(0.5, 1.0) * base_intensity

                # Create meshgrid only for crop region
                xx, yy = np.meshgrid(
                    np.arange(crop_x_min, crop_x_max) - led_x,
                    np.arange(crop_y_min, crop_y_max) - led_y,
                )

                # Gaussian pattern with color variation (cropped)
                color_variation = np.random.uniform(0.8, 1.2)
                gaussian = (
                    intensity
                    * color_variation
                    * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                )
                # Only update the crop region
                pattern[crop_y_min:crop_y_max, crop_x_min:crop_x_max, c] = gaussian

        elif pattern_type == "exponential":
            # Exponential falloff pattern (centered on LED, cropped to block)
            for c in range(3):
                decay_rate = np.random.uniform(0.05, 0.15)
                intensity = np.random.uniform(0.4, 1.0) * base_intensity

                # Create distance map only for crop region
                xx, yy = np.meshgrid(
                    np.arange(crop_x_min, crop_x_max) - led_x,
                    np.arange(crop_y_min, crop_y_max) - led_y,
                )
                distances = np.sqrt(xx**2 + yy**2)

                # Exponential decay with color variation (cropped)
                color_variation = np.random.uniform(0.7, 1.3)
                exponential_pattern = (
                    intensity * color_variation * np.exp(-decay_rate * distances)
                )
                # Only update the crop region
                pattern[
                    crop_y_min:crop_y_max, crop_x_min:crop_x_max, c
                ] = exponential_pattern

        # Normalize and clip to valid range
        if np.max(pattern) > 0:
            pattern = pattern / np.max(pattern)
        pattern = np.clip(pattern, 0, 1)

        return pattern

    def generate_sparse_patterns_chunked(
        self,
        led_count: int,
        pattern_type: str = "gaussian_multi",
        intensity_variation: bool = True,
        chunk_size: int = 50,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[LEDDiffusionCSCMatrix, dict]:
        """
        Generate sparse patterns in chunks to avoid memory issues.

        Args:
            led_count: Number of LEDs to generate patterns for
            pattern_type: Type of patterns to generate
            intensity_variation: Whether to vary intensity between LEDs
            chunk_size: Number of LEDs to process per chunk
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (LEDDiffusionCSCMatrix, led_spatial_mapping)
        """
        logger.info(
            f"Generating sparse patterns for {led_count} LEDs in chunks of {chunk_size}..."
        )
        logger.info(f"Pattern type: {pattern_type}")
        logger.info(f"Sparsity threshold: {self.sparsity_threshold}")

        start_time = time.time()

        # Generate LED positions
        if self.led_positions is None or len(self.led_positions) != led_count:
            self.generate_led_positions(led_count)

        # Import and use the unified block position calculation
        from tools.led_position_utils import calculate_block_positions

        # Calculate block positions with x-coordinates rounded to multiple of 4
        block_positions = calculate_block_positions(
            self.led_positions, self.block_size, self.frame_width, self.frame_height
        )

        # Compute RCM ordering directly from block positions
        logger.info("Computing RCM ordering for optimal bandwidth...")
        rcm_order, inverse_order, expected_ata_diagonals = compute_rcm_ordering(
            block_positions, self.block_size
        )
        logger.info(
            f"Expected A^T A diagonals (from adjacency): {expected_ata_diagonals}"
        )

        # Store expected diagonal count for later comparison
        self.expected_ata_diagonals = expected_ata_diagonals

        # Create mapping: physical_led_id -> rcm_ordered_matrix_index
        self.led_spatial_mapping = {
            original_id: rcm_pos for rcm_pos, original_id in enumerate(rcm_order)
        }
        # Create reverse mapping: rcm_ordered_matrix_index -> physical_led_id
        self.reverse_spatial_mapping = dict(enumerate(rcm_order))

        # Calculate chunk parameters
        num_chunks = (led_count + chunk_size - 1) // chunk_size
        pixels_per_channel = self.frame_height * self.frame_width

        # List to store sparse matrix chunks
        sparse_chunks = []

        logger.info(f"Processing {num_chunks} chunks in RCM order...")

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, led_count)
            chunk_led_count = chunk_end - chunk_start

            logger.info(
                f"Processing chunk {chunk_idx + 1}/{num_chunks}: "
                f"spatial indices {chunk_start}-{chunk_end-1}"
            )

            # Generate dense matrix for this chunk (384000, chunk_led_count * 3)
            chunk_matrix = np.zeros(
                (pixels_per_channel, chunk_led_count * 3), dtype=np.float32
            )

            for local_led_idx in range(chunk_led_count):
                spatial_idx = chunk_start + local_led_idx  # Spatial matrix index
                physical_led_id = self.reverse_spatial_mapping[
                    spatial_idx
                ]  # Get actual LED ID

                # Vary intensity if requested
                if intensity_variation:
                    base_intensity = np.random.uniform(0.6, 1.0)
                else:
                    base_intensity = 1.0

                # Generate pattern for this PHYSICAL LED at this SPATIAL position
                led_pos = tuple(self.led_positions[physical_led_id])
                block_pos = tuple(block_positions[physical_led_id])
                pattern = self.generate_single_pattern(
                    led_pos,
                    block_pos,
                    pattern_type=pattern_type,
                    base_intensity=base_intensity,
                )

                # Process all three color channels for this LED
                for channel in range(3):
                    # Extract the pattern for this channel (HWC format)
                    channel_pattern = pattern[:, :, channel]

                    # Flatten and apply threshold
                    pattern_flat = channel_pattern.reshape(-1)
                    significant_mask = pattern_flat > self.sparsity_threshold

                    # Store in dense chunk matrix at spatial position
                    # local_led_idx corresponds to spatial matrix column position
                    chunk_col_idx = local_led_idx * 3 + channel
                    chunk_matrix[significant_mask, chunk_col_idx] = pattern_flat[
                        significant_mask
                    ]

            # Convert chunk to sparse format
            chunk_sparse = sp.csc_matrix(chunk_matrix, dtype=np.float32)
            chunk_sparse.eliminate_zeros()
            sparse_chunks.append(chunk_sparse)

            # Progress reporting
            elapsed = time.time() - start_time
            eta = elapsed * (num_chunks / (chunk_idx + 1) - 1)
            chunk_sparsity = (
                chunk_sparse.nnz / (chunk_sparse.shape[0] * chunk_sparse.shape[1]) * 100
            )
            logger.info(
                f"Chunk {chunk_idx + 1}/{num_chunks} complete. "
                f"Sparsity: {chunk_sparsity:.2f}%, ETA: {eta:.1f}s"
            )

            if progress_callback:
                progress_callback(chunk_end, led_count)

        # Combine all chunks horizontally using hstack
        logger.info("Combining sparse chunks...")
        final_sparse_matrix = sp.hstack(sparse_chunks, format="csc")

        # Final cleanup
        final_sparse_matrix.eliminate_zeros()
        final_sparse_matrix = final_sparse_matrix.tocsc()

        generation_time = time.time() - start_time
        actual_sparsity = (
            final_sparse_matrix.nnz
            / (final_sparse_matrix.shape[0] * final_sparse_matrix.shape[1])
            * 100
        )
        memory_mb = final_sparse_matrix.data.nbytes / (1024 * 1024)

        logger.info(f"Generated chunked sparse matrix in {generation_time:.2f}s")
        logger.info(f"Matrix shape: {final_sparse_matrix.shape}")
        logger.info(f"Non-zero entries: {final_sparse_matrix.nnz:,}")
        logger.info(f"Actual sparsity: {actual_sparsity:.3f}%")
        logger.info(f"Memory usage: {memory_mb:.1f} MB")

        # Matrix is already in RCM order from generation - no need to reorder
        logger.info(
            "Matrix already generated in RCM order - skipping redundant reordering"
        )
        reordered_matrix = final_sparse_matrix

        # LED spatial mapping is already in RCM order from generation
        rcm_led_mapping = self.led_spatial_mapping

        # Wrap the reordered sparse matrix in our LEDDiffusionCSCMatrix wrapper
        diffusion_matrix = LEDDiffusionCSCMatrix(
            csc_matrix=reordered_matrix,
            height=self.frame_height,
            width=self.frame_width,
            channels=3,
        )

        logger.info(
            f"Created LEDDiffusionCSCMatrix wrapper with RCM ordering: {diffusion_matrix}"
        )

        return diffusion_matrix, rcm_led_mapping

    def _generate_mixed_tensor_format(
        self, sparse_matrix: sp.csc_matrix
    ) -> SingleBlockMixedSparseTensor:
        """
        Generate SingleBlockMixedSparseTensor from sparse matrix using the proper API.

        Args:
            sparse_matrix: Sparse CSC matrix to convert
            ( pixels, leds * channels ),
            pixels = flattened( height, width ),
            leds * channels = flattened(leds, channels)

        Returns:
            SingleBlockMixedSparseTensor object populated with block data
        """
        logger.info("Converting sparse matrix to mixed tensor format...")

        led_count = sparse_matrix.shape[1] // 3
        channels = 3
        height, width = self.frame_height, self.frame_width
        block_size = self.block_size  # Configurable block size

        # Create the SingleBlockMixedSparseTensor
        mixed_tensor = SingleBlockMixedSparseTensor(
            batch_size=led_count,
            channels=channels,
            height=height,
            width=width,
            block_size=block_size,
            device="cpu",  # Use CPU for pattern generation
        )

        blocks_stored = 0

        # Process each LED and channel to extract blocks from sparse matrix
        for led_id in range(led_count):
            for channel in range(channels):
                # Get the column for this LED/channel
                col_idx = led_id * channels + channel

                # Extract the sparse column
                column = sparse_matrix[:, col_idx]

                if column.nnz == 0:  # No non-zeros
                    # Set a zero block at position (0,0) - all blocks assumed to be set
                    block = np.zeros((block_size, block_size), dtype=np.float32)
                    block_cupy = cp.asarray(block)
                    mixed_tensor.set_block(led_id, channel, 0, 0, block_cupy)
                    continue

                # Get non-zero indices and values
                row_indices, values = column.nonzero()[0], column.data

                # Convert linear indices to 2D coordinates
                pixel_rows = row_indices // width
                pixel_cols = row_indices % width

                # Find bounding box of the pattern
                min_row, max_row = pixel_rows.min(), pixel_rows.max()
                min_col, max_col = pixel_cols.min(), pixel_cols.max()

                # Determine block position (try to center the pattern)
                pattern_height = max_row - min_row + 1
                pattern_width = max_col - min_col + 1

                # If pattern fits in block size, center it
                if pattern_height <= block_size and pattern_width <= block_size:
                    # Place block to contain the entire pattern
                    top_row = max(0, min(height - block_size, min_row))
                    top_col_candidate = max(0, min(width - block_size, min_col))
                    # ALIGNMENT: Round down x-coordinate to multiple of 4
                    top_col = (top_col_candidate // 4) * 4
                else:
                    # Pattern is larger than block, use LED position as center
                    if self.led_positions is not None and led_id < len(
                        self.led_positions
                    ):
                        # Use stored reverse mapping to get physical LED ID from spatial index
                        if led_id in self.reverse_spatial_mapping:
                            physical_led_id = self.reverse_spatial_mapping[led_id]
                            led_x, led_y = self.led_positions[physical_led_id]
                            top_row = max(
                                0, min(height - block_size, led_y - block_size // 2)
                            )
                            top_col_candidate = max(
                                0, min(width - block_size, led_x - block_size // 2)
                            )
                            # ALIGNMENT: Round down x-coordinate to multiple of 4
                            top_col = (top_col_candidate // 4) * 4
                        else:
                            # Fallback: use pattern center
                            center_row = (min_row + max_row) // 2
                            center_col = (min_col + max_col) // 2
                            top_row = max(
                                0,
                                min(height - block_size, center_row - block_size // 2),
                            )
                            top_col_candidate = max(
                                0, min(width - block_size, center_col - block_size // 2)
                            )
                            # ALIGNMENT: Round down x-coordinate to multiple of 4
                            top_col = (top_col_candidate // 4) * 4
                    else:
                        # Fallback: use pattern center
                        center_row = (min_row + max_row) // 2
                        center_col = (min_col + max_col) // 2
                        top_row = max(
                            0, min(height - block_size, center_row - block_size // 2)
                        )
                        top_col_candidate = max(
                            0, min(width - block_size, center_col - block_size // 2)
                        )
                        # ALIGNMENT: Round down x-coordinate to multiple of 4
                        top_col = (top_col_candidate // 4) * 4

                # Create the dense block
                block = np.zeros((block_size, block_size), dtype=np.float32)

                # Fill the block with pattern values
                for i, (pr, pc, val) in enumerate(zip(pixel_rows, pixel_cols, values)):
                    block_r = pr - top_row
                    block_c = pc - top_col

                    # Only include values that fit in the block
                    if 0 <= block_r < block_size and 0 <= block_c < block_size:
                        block[block_r, block_c] = val

                # Set the block in the mixed tensor using the proper API
                # Convert numpy array to cupy if needed
                block_cupy = cp.asarray(block)
                mixed_tensor.set_block(led_id, channel, top_row, top_col, block_cupy)
                blocks_stored += 1

        logger.info(f"Converted {blocks_stored} blocks to mixed tensor format")

        # Return the mixed tensor object
        return mixed_tensor

    def _precompute_dense_ata_matrices(self, sparse_matrix: sp.csc_matrix) -> dict:
        """
        Precompute dense A^T @ A matrices for each RGB channel using CSC matrix.

        Args:
            sparse_matrix: Sparse CSC matrix to compute A^T @ A from

        Returns:
            Dictionary with precomputed dense A^T @ A matrices
        """
        logger.info("Precomputing dense A^T @ A matrices from CSC sparse matrix...")

        led_count = sparse_matrix.shape[1] // 3
        channels = 3

        # Initialize dense A^T @ A tensor: (led_count, led_count, channels)
        ATA_dense = np.zeros((led_count, led_count, channels), dtype=np.float32)

        start_time = time.time()

        # Track bandwidth statistics for all channels
        ata_bandwidths = []

        # Compute A^T @ A for each channel separately
        for c in range(channels):
            logger.info(f"Computing A^T @ A for channel {c+1}/{channels}")

            # Extract channel matrix (pixels, leds) for this channel
            channel_cols = list(range(c, sparse_matrix.shape[1], channels))
            A_channel = sparse_matrix[:, channel_cols]

            # Compute A_c^T @ A_c (led_count, led_count) - this is dense
            ATA_channel = A_channel.T @ A_channel

            # Compute bandwidth before converting to dense
            coo = ATA_channel.tocoo()
            if coo.nnz > 0:
                diagonal_offsets = coo.col - coo.row
                unique_diagonals = np.unique(diagonal_offsets)
                bandwidth = len(unique_diagonals)
                ata_bandwidths.append(bandwidth)
                logger.info(f"Channel {c+1} A^T @ A bandwidth: {bandwidth} diagonals")
                logger.info(
                    f"Channel {c+1} A^T @ A diagonal range: "
                    f"[{diagonal_offsets.min()}, {diagonal_offsets.max()}]"
                )
                logger.info(f"Channel {c+1} A^T @ A nnz: {coo.nnz:,}")
                logger.info(
                    f"Channel {c+1} A^T @ A diagonals present: {sorted(unique_diagonals.tolist())}"
                )
                # Check if main diagonal (0) is present
                if 0 in unique_diagonals:
                    logger.info(
                        f"Channel {c+1} main diagonal (0) is present in A^T @ A"
                    )
                else:
                    logger.info(
                        f"Channel {c+1} main diagonal (0) is MISSING from A^T @ A"
                    )
            else:
                ata_bandwidths.append(0)
                logger.info(f"Channel {c+1} A^T @ A is empty")

            # Convert to dense and store
            ATA_dense[:, :, c] = ATA_channel.toarray().astype(np.float32)

            channel_time = time.time() - start_time
            logger.info(f"Channel {c+1} A^T @ A computed in {channel_time:.2f}s")

        total_time = time.time() - start_time
        memory_mb = ATA_dense.nbytes / (1024 * 1024)
        avg_ata_bandwidth = np.mean(ata_bandwidths) if ata_bandwidths else 0

        logger.info(f"Dense A^T @ A precomputation completed in {total_time:.2f}s")
        logger.info(f"Dense A^T @ A tensor shape: {ATA_dense.shape}")
        logger.info(f"Dense A^T @ A memory: {memory_mb:.1f}MB")
        logger.info(
            f"Average A^T @ A bandwidth across channels: {avg_ata_bandwidth:.1f} diagonals"
        )

        return {
            "dense_ata_matrices": ATA_dense,
            "dense_ata_led_count": led_count,
            "dense_ata_channels": channels,
            "dense_ata_computation_time": total_time,
            "ata_bandwidths": ata_bandwidths,
            "avg_ata_bandwidth": avg_ata_bandwidth,
        }

    def _generate_dia_matrix(
        self, sparse_matrix: sp.csc_matrix, led_positions: np.ndarray
    ) -> "DiagonalATAMatrix":
        """
        Generate DiagonalATAMatrix from sparse diffusion matrix.

        Args:
            sparse_matrix: Sparse CSC diffusion matrix (pixels, leds*3)
            led_positions: LED positions array (leds, 2)

        Returns:
            DiagonalATAMatrix object with 3D DIA format
        """
        logger.info("Building DiagonalATAMatrix from diffusion matrix...")

        led_count = sparse_matrix.shape[1] // 3

        # Create DiagonalATAMatrix instance
        dia_matrix = DiagonalATAMatrix(led_count, crop_size=self.block_size)

        # Build from diffusion matrix (already in optimal RCM ordering)
        dia_matrix.build_from_diffusion_matrix(sparse_matrix)

        logger.info(
            f"DiagonalATAMatrix built: {led_count} LEDs, "
            f"bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k} diagonals"
        )

        # Compare expected vs actual diagonal counts
        if hasattr(self, "expected_ata_diagonals"):
            expected = self.expected_ata_diagonals
            actual = dia_matrix.k
            ratio = actual / expected if expected > 0 else float("inf")
            if ratio > 2.0:
                logger.warning(
                    f"DIA matrix diagonal count mismatch: expected={expected}, "
                    f"actual={actual} ({ratio:.1f}x more than expected!)"
                )
                logger.warning(
                    "This indicates pattern generation may not be following "
                    "adjacency structure properly"
                )
            else:
                logger.info(
                    f"DIA matrix diagonal count: expected={expected}, "
                    f"actual={actual} ({ratio:.1f}x expected - good!)"
                )

        return dia_matrix

    def save_sparse_matrix(
        self,
        diffusion_matrix: LEDDiffusionCSCMatrix,
        led_spatial_mapping: dict,
        output_path: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Save LEDDiffusionCSCMatrix, spatial mapping and mixed tensor format for optimization.

        Args:
            diffusion_matrix: LEDDiffusionCSCMatrix to save
            led_spatial_mapping: LED spatial ordering mapping
            output_path: Output file path
            metadata: Optional metadata dictionary

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate mixed tensor format using SingleBlockMixedSparseTensor
            logger.info("Generating mixed tensor format...")
            mixed_tensor = self._generate_mixed_tensor_format(
                diffusion_matrix.to_csc_matrix()
            )

            # Prepare metadata
            save_metadata = {
                "generator": "SyntheticPatternGenerator",
                "format": "led_diffusion_csc_with_mixed_tensor",
                "led_count": diffusion_matrix.led_count,
                "frame_width": diffusion_matrix.width,
                "frame_height": diffusion_matrix.height,
                "channels": diffusion_matrix.channels,
                "matrix_shape": list(diffusion_matrix.shape),
                "nnz": diffusion_matrix.matrix.nnz,
                "sparsity_percent": diffusion_matrix.matrix.nnz
                / (diffusion_matrix.shape[0] * diffusion_matrix.shape[1])
                * 100,
                "sparsity_threshold": self.sparsity_threshold,
                "generation_timestamp": time.time(),
            }

            if metadata:
                save_metadata.update(metadata)

            # Generate DiagonalATAMatrix (DIA format)
            logger.info("Generating DiagonalATAMatrix (DIA format)...")
            dia_matrix = self._generate_dia_matrix(
                diffusion_matrix.to_csc_matrix(), self.led_positions
            )

            # Save everything in a single NPZ file
            save_dict = {
                # LED information
                "led_positions": self.led_positions,
                "led_spatial_mapping": led_spatial_mapping,
                # Metadata
                "metadata": save_metadata,
                # Mixed tensor stored as nested element using to_dict()
                "mixed_tensor": mixed_tensor.to_dict(),
                # DIA format A^T @ A matrix
                "dia_matrix": dia_matrix.to_dict(),
            }

            np.savez_compressed(output_path, **save_dict)

            # Log file info
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB

            logger.info(f"Saved mixed tensor and DIA matrix to {output_path}")
            logger.info(f"File size: {file_size:.1f} MB")
            logger.info(f"Mixed tensor format: SingleBlockMixedSparseTensor")
            logger.info(
                f"Mixed tensor: {mixed_tensor.batch_size} LEDs, "
                f"{mixed_tensor.height}x{mixed_tensor.width}, "
                f"{mixed_tensor.block_size}x{mixed_tensor.block_size} blocks"
            )
            logger.info(
                f"DIA matrix: {dia_matrix.led_count} LEDs, "
                f"bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k} diagonals"
            )
            storage_shape = (
                dia_matrix.dia_data_cpu.shape
                if dia_matrix.dia_data_cpu is not None
                else "None"
            )
            logger.info(f"DIA matrix storage shape: {storage_shape}")

            return True

        except Exception as e:
            logger.error(f"Failed to save LEDDiffusionCSCMatrix: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic LED diffusion patterns"
    )
    parser.add_argument("--output", "-o", required=True, help="Output NPZ file path")
    parser.add_argument(
        "--led-count",
        "-n",
        type=int,
        default=DEFAULT_LED_COUNT,
        help=f"Number of LEDs (default: {DEFAULT_LED_COUNT})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help=f"Frame width (default: {DEFAULT_FRAME_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help=f"Frame height (default: {DEFAULT_FRAME_HEIGHT})",
    )
    parser.add_argument(
        "--pattern-type",
        choices=["gaussian_multi", "gaussian_simple", "exponential"],
        default="gaussian_multi",
        help="Pattern type to generate",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible patterns"
    )
    parser.add_argument(
        "--no-intensity-variation",
        action="store_true",
        help="Disable intensity variation between LEDs",
    )
    parser.add_argument(
        "--sparsity-threshold",
        type=float,
        default=0.01,
        help="Threshold for sparse matrix (default: 0.01)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Number of LEDs to process per chunk for sparse generation (default: 50)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=96,
        help="Block size for LED diffusion patterns (default: 96)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # Create generator
        generator = SyntheticPatternGenerator(
            frame_width=args.width,
            frame_height=args.height,
            seed=args.seed,
            sparsity_threshold=args.sparsity_threshold,
            block_size=args.block_size,
        )

        # Prepare metadata
        metadata = {
            "pattern_type": args.pattern_type,
            "seed": args.seed,
            "intensity_variation": not args.no_intensity_variation,
            "block_size": args.block_size,
            "command_line": " ".join(sys.argv),
        }

        # Generate sparse patterns directly using chunked approach
        logger.info("Generating sparse patterns using chunked approach...")
        diffusion_matrix, led_mapping = generator.generate_sparse_patterns_chunked(
            led_count=args.led_count,
            pattern_type=args.pattern_type,
            intensity_variation=not args.no_intensity_variation,
            chunk_size=args.chunk_size,
        )

        # Add sparse-specific metadata
        metadata.update(
            {
                "format": "sparse_csc",
                "sparsity_threshold": args.sparsity_threshold,
                "generation_method": "chunked",
            }
        )

        # Save LEDDiffusionCSCMatrix
        if not generator.save_sparse_matrix(
            diffusion_matrix, led_mapping, args.output, metadata
        ):
            logger.error("Failed to save LEDDiffusionCSCMatrix")
            return 1

        logger.info("LEDDiffusionCSCMatrix generation completed successfully")

        return 0

    except Exception as e:
        logger.error(f"Pattern generation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
