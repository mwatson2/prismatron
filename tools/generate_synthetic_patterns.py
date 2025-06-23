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
    ):
        """
        Initialize pattern generator.

        Args:
            frame_width: Width of output frames
            frame_height: Height of output frames
            seed: Random seed for reproducible patterns
            sparsity_threshold: Threshold below which pixels are considered zero
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sparsity_threshold = sparsity_threshold

        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Using random seed: {seed}")

        # Generate LED positions (random for synthetic patterns)
        self.led_positions = None
        self.led_spatial_mapping = None

    def generate_led_positions(self, led_count: int) -> np.ndarray:
        """
        Generate random LED positions across the frame.

        Args:
            led_count: Number of LEDs to position

        Returns:
            Array of LED positions (led_count, 2) with [x, y] coordinates
        """
        positions = np.zeros((led_count, 2), dtype=int)
        positions[:, 0] = np.random.randint(0, self.frame_width, led_count)
        positions[:, 1] = np.random.randint(0, self.frame_height, led_count)

        self.led_positions = positions

        # Create spatial ordering for cache optimization
        self.led_spatial_mapping = self.create_led_spatial_ordering(positions)

        logger.info(f"Generated {led_count} LED positions with spatial ordering")
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
        pattern_type: str = "gaussian_multi",
        base_intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Generate a single LED diffusion pattern.

        Args:
            led_position: LED position as (x, y) coordinates
            pattern_type: Type of pattern to generate
            base_intensity: Base intensity multiplier

        Returns:
            Pattern array (height, width, 3) for RGB channels
        """
        x, y = led_position
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
                    # Random offset for sub-patterns
                    offset_x = np.random.normal(0, 15)
                    offset_y = np.random.normal(0, 15)
                    center_x = np.clip(x + offset_x, 0, self.frame_width - 1)
                    center_y = np.clip(y + offset_y, 0, self.frame_height - 1)

                    # Random sigma for different spread
                    sigma = np.random.uniform(8, 30)
                    intensity = np.random.uniform(0.3, 1.0) * base_intensity

                    # Create meshgrid
                    xx, yy = np.meshgrid(
                        np.arange(self.frame_width) - center_x,
                        np.arange(self.frame_height) - center_y,
                    )

                    # Gaussian pattern
                    gaussian = intensity * np.exp(
                        -(xx**2 + yy**2) / (2 * sigma**2)
                    )
                    channel_pattern += gaussian

                # Add some color variation between channels
                color_variation = np.random.uniform(0.7, 1.3)
                pattern[:, :, c] = channel_pattern * color_variation

        elif pattern_type == "gaussian_simple":
            # Single Gaussian per channel
            for c in range(3):
                sigma = np.random.uniform(15, 40)
                intensity = np.random.uniform(0.5, 1.0) * base_intensity

                # Create meshgrid
                xx, yy = np.meshgrid(
                    np.arange(self.frame_width) - x, np.arange(self.frame_height) - y
                )

                # Gaussian pattern with color variation
                color_variation = np.random.uniform(0.8, 1.2)
                gaussian = (
                    intensity
                    * color_variation
                    * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                )
                pattern[:, :, c] = gaussian

        elif pattern_type == "exponential":
            # Exponential falloff pattern
            for c in range(3):
                decay_rate = np.random.uniform(0.05, 0.15)
                intensity = np.random.uniform(0.4, 1.0) * base_intensity

                # Create distance map
                xx, yy = np.meshgrid(
                    np.arange(self.frame_width) - x, np.arange(self.frame_height) - y
                )
                distances = np.sqrt(xx**2 + yy**2)

                # Exponential decay with color variation
                color_variation = np.random.uniform(0.7, 1.3)
                pattern[:, :, c] = (
                    intensity * color_variation * np.exp(-decay_rate * distances)
                )

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
    ) -> Tuple[sp.csc_matrix, dict]:
        """
        Generate sparse patterns in chunks to avoid memory issues.

        Args:
            led_count: Number of LEDs to generate patterns for
            pattern_type: Type of patterns to generate
            intensity_variation: Whether to vary intensity between LEDs
            chunk_size: Number of LEDs to process per chunk
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (sparse_csc_matrix, led_spatial_mapping)
        """
        logger.info(
            f"Generating sparse patterns for {led_count} LEDs in chunks of {chunk_size}..."
        )
        logger.info(f"Pattern type: {pattern_type}")
        logger.info(f"Sparsity threshold: {self.sparsity_threshold}")

        start_time = time.time()

        # Generate LED positions and spatial mapping
        if self.led_positions is None or len(self.led_positions) != led_count:
            self.generate_led_positions(led_count)

        # Calculate chunk parameters
        num_chunks = (led_count + chunk_size - 1) // chunk_size
        pixels_per_channel = self.frame_height * self.frame_width

        # List to store sparse matrix chunks
        sparse_chunks = []

        logger.info(f"Processing {num_chunks} chunks...")

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, led_count)
            chunk_led_count = chunk_end - chunk_start

            logger.info(
                f"Processing chunk {chunk_idx + 1}/{num_chunks}: LEDs {chunk_start}-{chunk_end-1}"
            )

            # Generate dense matrix for this chunk (384000, chunk_led_count * 3)
            chunk_matrix = np.zeros(
                (pixels_per_channel, chunk_led_count * 3), dtype=np.float32
            )

            for local_led_idx in range(chunk_led_count):
                physical_led_id = chunk_start + local_led_idx
                matrix_led_idx = self.led_spatial_mapping[physical_led_id]

                # Vary intensity if requested
                if intensity_variation:
                    base_intensity = np.random.uniform(0.6, 1.0)
                else:
                    base_intensity = 1.0

                # Generate pattern for this LED
                led_pos = tuple(self.led_positions[physical_led_id])
                pattern = self.generate_single_pattern(
                    led_pos, pattern_type=pattern_type, base_intensity=base_intensity
                )

                # Process all three color channels for this LED
                for channel in range(3):
                    # Extract the pattern for this channel (HWC format)
                    channel_pattern = pattern[:, :, channel]

                    # Flatten and apply threshold
                    pattern_flat = channel_pattern.reshape(-1)
                    significant_mask = pattern_flat > self.sparsity_threshold

                    # Store in dense chunk matrix
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

        return final_sparse_matrix, self.led_spatial_mapping

    def _generate_mixed_tensor_format(self, sparse_matrix: sp.csc_matrix) -> dict:
        """
        Generate SingleBlockMixedSparseTensor format from sparse matrix.

        Args:
            sparse_matrix: Sparse CSC matrix to convert

        Returns:
            Dictionary with mixed tensor data for saving
        """
        logger.info("Converting sparse matrix to mixed tensor format...")

        led_count = sparse_matrix.shape[1] // 3
        channels = 3
        height, width = self.frame_height, self.frame_width
        block_size = 96  # Standard block size

        # Initialize mixed tensor data structures
        mixed_tensor_values = np.zeros(
            (led_count, channels, block_size, block_size), dtype=np.float32
        )
        mixed_tensor_positions = np.zeros((led_count, channels, 2), dtype=np.int32)
        mixed_tensor_blocks_set = np.zeros((led_count, channels), dtype=bool)

        blocks_stored = 0

        # Process each LED and channel
        for led_id in range(led_count):
            for channel in range(channels):
                # Get the column for this LED/channel
                col_idx = led_id * channels + channel

                # Extract the sparse column
                col_start = sparse_matrix.indptr[col_idx]
                col_end = sparse_matrix.indptr[col_idx + 1]

                if col_start == col_end:  # No non-zeros
                    continue

                # Get non-zero indices and values
                row_indices = sparse_matrix.indices[col_start:col_end]
                values = sparse_matrix.data[col_start:col_end]

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
                    top_col = max(0, min(width - block_size, min_col))
                else:
                    # Pattern is larger than block, use LED position as center
                    if self.led_positions is not None and led_id < len(
                        self.led_positions
                    ):
                        led_x, led_y = self.led_positions[led_id]
                        top_row = max(
                            0, min(height - block_size, led_y - block_size // 2)
                        )
                        top_col = max(
                            0, min(width - block_size, led_x - block_size // 2)
                        )
                    else:
                        # Fallback: use pattern center
                        center_row = (min_row + max_row) // 2
                        center_col = (min_col + max_col) // 2
                        top_row = max(
                            0, min(height - block_size, center_row - block_size // 2)
                        )
                        top_col = max(
                            0, min(width - block_size, center_col - block_size // 2)
                        )

                # Create the dense block
                block = np.zeros((block_size, block_size), dtype=np.float32)

                # Fill the block with pattern values
                for i, (pr, pc, val) in enumerate(zip(pixel_rows, pixel_cols, values)):
                    block_r = pr - top_row
                    block_c = pc - top_col

                    # Only include values that fit in the block
                    if 0 <= block_r < block_size and 0 <= block_c < block_size:
                        block[block_r, block_c] = val

                # Store the block data
                mixed_tensor_values[led_id, channel] = block
                mixed_tensor_positions[led_id, channel, 0] = top_row
                mixed_tensor_positions[led_id, channel, 1] = top_col
                mixed_tensor_blocks_set[led_id, channel] = True
                blocks_stored += 1

        logger.info(f"Converted {blocks_stored} blocks to mixed tensor format")

        return {
            "mixed_tensor_values": mixed_tensor_values,
            "mixed_tensor_positions": mixed_tensor_positions,
            "mixed_tensor_blocks_set": mixed_tensor_blocks_set,
            "mixed_tensor_block_size": block_size,
            "mixed_tensor_led_count": led_count,
            "mixed_tensor_channels": channels,
            "mixed_tensor_height": height,
            "mixed_tensor_width": width,
            "mixed_tensor_blocks_stored": blocks_stored,
        }

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

        # Compute A^T @ A for each channel separately
        for c in range(channels):
            logger.info(f"Computing A^T @ A for channel {c+1}/{channels}")

            # Extract channel matrix (pixels, leds) for this channel
            channel_cols = list(range(c, sparse_matrix.shape[1], channels))
            A_channel = sparse_matrix[:, channel_cols]

            # Compute A_c^T @ A_c (led_count, led_count) - this is dense
            ATA_channel = A_channel.T @ A_channel

            # Convert to dense and store
            ATA_dense[:, :, c] = ATA_channel.toarray().astype(np.float32)

            channel_time = time.time() - start_time
            logger.info(f"Channel {c+1} A^T @ A computed in {channel_time:.2f}s")

        total_time = time.time() - start_time
        memory_mb = ATA_dense.nbytes / (1024 * 1024)

        logger.info(f"Dense A^T @ A precomputation completed in {total_time:.2f}s")
        logger.info(f"Dense A^T @ A tensor shape: {ATA_dense.shape}")
        logger.info(f"Dense A^T @ A memory: {memory_mb:.1f}MB")

        return {
            "dense_ata_matrices": ATA_dense,
            "dense_ata_led_count": led_count,
            "dense_ata_channels": channels,
            "dense_ata_computation_time": total_time,
        }

    def save_sparse_matrix(
        self,
        sparse_matrix: sp.csc_matrix,
        led_spatial_mapping: dict,
        output_path: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Save sparse CSC matrix, spatial mapping and mixed tensor format for optimization.

        Args:
            sparse_matrix: Sparse CSC matrix to save
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

            # Prepare metadata
            save_metadata = {
                "generator": "SyntheticPatternGenerator",
                "format": "sparse_csc_with_mixed_tensor",
                "led_count": sparse_matrix.shape[1] // 3,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "matrix_shape": list(sparse_matrix.shape),
                "nnz": sparse_matrix.nnz,
                "sparsity_percent": sparse_matrix.nnz
                / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
                * 100,
                "sparsity_threshold": self.sparsity_threshold,
                "generation_timestamp": time.time(),
            }

            if metadata:
                save_metadata.update(metadata)

            # Generate mixed tensor format
            logger.info("Generating mixed tensor format...")
            mixed_tensor_data = self._generate_mixed_tensor_format(sparse_matrix)

            # Precompute dense A^T @ A matrices
            logger.info("Precomputing dense A^T @ A matrices...")
            dense_ata_data = self._precompute_dense_ata_matrices(sparse_matrix)

            # Save everything in a single NPZ file
            # Convert sparse matrix to component arrays for storage
            save_dict = {
                # Sparse matrix components (CSC format)
                "matrix_data": sparse_matrix.data,
                "matrix_indices": sparse_matrix.indices,
                "matrix_indptr": sparse_matrix.indptr,
                "matrix_shape": sparse_matrix.shape,
                # LED information
                "led_positions": self.led_positions,
                "led_spatial_mapping": led_spatial_mapping,
                # Metadata
                "metadata": save_metadata,
            }

            # Add mixed tensor data
            save_dict.update(mixed_tensor_data)

            # Add dense A^T @ A data
            save_dict.update(dense_ata_data)

            np.savez_compressed(output_path, **save_dict)

            # Log file info
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB

            logger.info(f"Saved sparse matrix and mixed tensor to {output_path}")
            logger.info(f"File size: {file_size:.1f} MB")
            logger.info(f"Matrix shape: {sparse_matrix.shape}")
            logger.info(f"Non-zero entries: {sparse_matrix.nnz:,}")
            logger.info(
                f"Mixed tensor blocks: {mixed_tensor_data['mixed_tensor_blocks_stored']}"
            )
            logger.info(
                f"Dense A^T @ A tensor shape: {dense_ata_data['dense_ata_matrices'].shape}"
            )
            logger.info(
                f"Dense A^T @ A memory: {dense_ata_data['dense_ata_matrices'].nbytes / (1024*1024):.1f}MB"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save sparse matrix: {e}")
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
        )

        # Prepare metadata
        metadata = {
            "pattern_type": args.pattern_type,
            "seed": args.seed,
            "intensity_variation": not args.no_intensity_variation,
            "command_line": " ".join(sys.argv),
        }

        # Generate sparse patterns directly using chunked approach
        logger.info("Generating sparse patterns using chunked approach...")
        sparse_matrix, led_mapping = generator.generate_sparse_patterns_chunked(
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

        # Save sparse matrix
        if not generator.save_sparse_matrix(
            sparse_matrix, led_mapping, args.output, metadata
        ):
            logger.error("Failed to save sparse matrix")
            return 1

        logger.info("Sparse matrix generation completed successfully")

        return 0

    except Exception as e:
        logger.error(f"Pattern generation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
