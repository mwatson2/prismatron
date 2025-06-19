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

    def generate_patterns(
        self,
        led_count: int,
        pattern_type: str = "gaussian_multi",
        intensity_variation: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> np.ndarray:
        """
        Generate synthetic diffusion patterns for all LEDs.

        Args:
            led_count: Number of LEDs to generate patterns for
            pattern_type: Type of patterns to generate
            intensity_variation: Whether to vary intensity between LEDs
            progress_callback: Optional callback for progress updates

        Returns:
            Patterns array (led_count, 3, height, width)
        """
        logger.info(f"Generating {led_count} synthetic diffusion patterns...")
        logger.info(f"Pattern type: {pattern_type}")
        logger.info(f"Frame size: {self.frame_width}x{self.frame_height}")

        start_time = time.time()

        # Generate LED positions if not already done
        if self.led_positions is None or len(self.led_positions) != led_count:
            self.generate_led_positions(led_count)

        # Initialize patterns array (led_count, height, width, 3) - HWC format for production
        patterns = np.zeros(
            (led_count, self.frame_height, self.frame_width, 3), dtype=np.uint8
        )

        for led_idx in range(led_count):
            # Vary intensity between LEDs if requested
            if intensity_variation:
                base_intensity = np.random.uniform(0.6, 1.0)
            else:
                base_intensity = 1.0

            # Generate pattern for this LED
            led_pos = tuple(self.led_positions[led_idx])
            pattern = self.generate_single_pattern(
                led_pos, pattern_type=pattern_type, base_intensity=base_intensity
            )

            # Convert to uint8 and store in HWC format (production format)
            pattern_uint8 = (pattern * 255).astype(np.uint8)
            patterns[led_idx] = pattern_uint8  # Already in HWC format

            # Progress reporting
            if progress_callback and (led_idx + 1) % 100 == 0:
                progress_callback(led_idx + 1, led_count)
            elif (led_idx + 1) % 500 == 0:
                elapsed = time.time() - start_time
                eta = elapsed * (led_count / (led_idx + 1) - 1)
                logger.info(
                    f"Generated {led_idx + 1}/{led_count} patterns... "
                    f"ETA: {eta:.1f}s"
                )

        generation_time = time.time() - start_time
        logger.info(f"Generated {led_count} patterns in {generation_time:.2f}s")

        return patterns

    def generate_sparse_csc_matrix(
        self,
        led_count: int,
        pattern_type: str = "gaussian_multi",
        intensity_variation: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[sp.csc_matrix, dict]:
        """
        Generate sparse CSC matrix directly from patterns for memory efficiency.

        Args:
            led_count: Number of LEDs to generate patterns for
            pattern_type: Type of patterns to generate
            intensity_variation: Whether to vary intensity between LEDs
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (sparse_csc_matrix, led_spatial_mapping)
        """
        logger.info(f"Generating sparse CSC matrix for {led_count} LEDs...")
        logger.info(f"Sparsity threshold: {self.sparsity_threshold}")

        start_time = time.time()

        # Generate LED positions if not already done
        if self.led_positions is None or len(self.led_positions) != led_count:
            self.generate_led_positions(led_count)

        # Prepare sparse matrix data structures
        rows = []
        cols = []
        values = []

        # Total number of pixels (RGB channels flattened)
        total_pixels = self.frame_height * self.frame_width * 3

        for physical_led_id in range(led_count):
            # Vary intensity between LEDs if requested
            if intensity_variation:
                base_intensity = np.random.uniform(0.6, 1.0)
            else:
                base_intensity = 1.0

            # Generate pattern for this LED
            led_pos = tuple(self.led_positions[physical_led_id])
            pattern = self.generate_single_pattern(
                led_pos, pattern_type=pattern_type, base_intensity=base_intensity
            )

            # Get spatially-ordered matrix column index
            matrix_led_idx = self.led_spatial_mapping[physical_led_id]

            # Extract significant pixels above threshold
            significant_pixels = self.extract_significant_pixels(
                pattern, self.sparsity_threshold
            )

            # Add to sparse matrix data
            for pixel_row, pixel_col, channel, intensity in significant_pixels:
                # Calculate flattened pixel index (channel-separate blocks format)
                # This matches the format expected by LED optimizer
                pixels_per_channel = self.frame_height * self.frame_width
                pixel_in_channel = pixel_row * self.frame_width + pixel_col
                pixel_idx = channel * pixels_per_channel + pixel_in_channel

                rows.append(pixel_idx)
                cols.append(matrix_led_idx)
                values.append(intensity)

            # Progress reporting
            if progress_callback and (physical_led_id + 1) % 100 == 0:
                progress_callback(physical_led_id + 1, led_count)
            elif (physical_led_id + 1) % 500 == 0:
                elapsed = time.time() - start_time
                eta = elapsed * (led_count / (physical_led_id + 1) - 1)
                sparsity = len(values) / ((physical_led_id + 1) * total_pixels) * 100
                logger.info(
                    f"Generated {physical_led_id + 1}/{led_count} sparse patterns... "
                    f"ETA: {eta:.1f}s, Sparsity: {sparsity:.2f}%"
                )

        # Create CSC matrix (optimal for A^T operations in LSQR)
        logger.info(f"Creating CSC matrix from {len(values)} non-zero entries...")
        A_sparse_csc = sp.csc_matrix(
            (values, (rows, cols)), shape=(total_pixels, led_count), dtype=np.float32
        )

        # Eliminate duplicate entries and compress
        A_sparse_csc.eliminate_zeros()
        A_sparse_csc = A_sparse_csc.tocsc()  # Ensure proper CSC format

        generation_time = time.time() - start_time
        actual_sparsity = (
            A_sparse_csc.nnz / (A_sparse_csc.shape[0] * A_sparse_csc.shape[1]) * 100
        )
        memory_mb = A_sparse_csc.data.nbytes / (1024 * 1024)

        logger.info(f"Generated sparse CSC matrix in {generation_time:.2f}s")
        logger.info(f"Matrix shape: {A_sparse_csc.shape}")
        logger.info(f"Non-zero entries: {A_sparse_csc.nnz:,}")
        logger.info(f"Actual sparsity: {actual_sparsity:.3f}%")
        logger.info(f"Memory usage: {memory_mb:.1f} MB")

        return A_sparse_csc, self.led_spatial_mapping

    def extract_significant_pixels(self, pattern: np.ndarray, threshold: float) -> list:
        """
        Extract pixels above threshold with their coordinates, channels, and intensities.
        Optimized version using NumPy vectorized operations.

        Args:
            pattern: Pattern array (height, width, 3)
            threshold: Intensity threshold (0-1)

        Returns:
            List of (row, col, channel, intensity) tuples for significant pixels
        """
        # Find all pixels above threshold using vectorized operations
        height, width, channels = pattern.shape

        # Create coordinate grids
        rows, cols, chans = np.meshgrid(
            np.arange(height), np.arange(width), np.arange(channels), indexing="ij"
        )

        # Create mask for significant pixels
        mask = pattern > threshold

        # Extract coordinates and values for significant pixels
        significant_rows = rows[mask]
        significant_cols = cols[mask]
        significant_chans = chans[mask]
        significant_intensities = pattern[mask]

        # Convert to list of tuples
        significant_pixels = [
            (int(row), int(col), int(chan), float(intensity))
            for row, col, chan, intensity in zip(
                significant_rows,
                significant_cols,
                significant_chans,
                significant_intensities,
            )
        ]

        return significant_pixels

    def save_patterns(
        self,
        patterns: np.ndarray,
        output_path: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Save patterns to compressed NPZ file.

        Args:
            patterns: Patterns array to save
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
                "led_count": patterns.shape[0],
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "pattern_shape": list(patterns.shape),
                "generation_timestamp": time.time(),
            }

            if metadata:
                save_metadata.update(metadata)

            logger.info(f"Compressing patterns shape: {patterns.shape}")

            # Save data
            np.savez_compressed(
                output_path,
                diffusion_patterns=patterns,
                led_positions=self.led_positions,
                metadata=save_metadata,
            )

            # Log file info
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Saved patterns to {output_path}")
            logger.info(f"File size: {file_size:.1f} MB")
            logger.info(f"Pattern shape: {patterns.shape}")

            return True

        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
            return False

    def save_sparse_matrix(
        self,
        sparse_matrix: sp.csc_matrix,
        led_spatial_mapping: dict,
        output_path: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Save sparse CSC matrix and spatial mapping for optimization.

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

            # Save sparse matrix
            matrix_path = output_path.replace(".npz", "_matrix.npz")
            sp.save_npz(matrix_path, sparse_matrix)

            # Prepare metadata for mapping file
            save_metadata = {
                "generator": "SyntheticPatternGenerator",
                "format": "sparse_csc",
                "led_count": sparse_matrix.shape[1],
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

            # Save LED spatial mapping and metadata
            mapping_path = output_path.replace(".npz", "_mapping.npz")
            np.savez_compressed(
                mapping_path,
                led_positions=self.led_positions,
                led_spatial_mapping=led_spatial_mapping,
                metadata=save_metadata,
            )

            # Log file info
            matrix_size = Path(matrix_path).stat().st_size / (1024 * 1024)  # MB
            mapping_size = Path(mapping_path).stat().st_size / (1024 * 1024)  # MB

            logger.info(f"Saved sparse matrix to {matrix_path}")
            logger.info(f"Saved spatial mapping to {mapping_path}")
            logger.info(f"Matrix file size: {matrix_size:.1f} MB")
            logger.info(f"Mapping file size: {mapping_size:.1f} MB")
            logger.info(f"Matrix shape: {sparse_matrix.shape}")
            logger.info(f"Non-zero entries: {sparse_matrix.nnz:,}")

            return True

        except Exception as e:
            logger.error(f"Failed to save sparse matrix: {e}")
            return False

    def get_pattern_info(self, patterns: np.ndarray) -> dict:
        """
        Get information about generated patterns (memory-efficient version).

        Args:
            patterns: Patterns array

        Returns:
            Dictionary with pattern statistics
        """
        # Calculate stats in chunks to avoid OOM
        chunk_size = min(50, patterns.shape[0])  # Process 50 LEDs at a time

        min_val = float("inf")
        max_val = float("-inf")
        sum_val = 0.0
        sum_sq = 0.0
        total_elements = 0

        for i in range(0, patterns.shape[0], chunk_size):
            end_idx = min(i + chunk_size, patterns.shape[0])
            chunk = patterns[i:end_idx]

            chunk_min = float(chunk.min())
            chunk_max = float(chunk.max())
            chunk_sum = float(chunk.sum())
            chunk_sum_sq = float((chunk.astype(np.float64) ** 2).sum())
            chunk_elements = chunk.size

            min_val = min(min_val, chunk_min)
            max_val = max(max_val, chunk_max)
            sum_val += chunk_sum
            sum_sq += chunk_sum_sq
            total_elements += chunk_elements

        mean_val = sum_val / total_elements
        std_val = np.sqrt((sum_sq / total_elements) - (mean_val**2))

        return {
            "shape": list(patterns.shape),
            "dtype": str(patterns.dtype),
            "led_count": patterns.shape[0],
            "channels": patterns.shape[1],
            "frame_height": patterns.shape[2],
            "frame_width": patterns.shape[3],
            "memory_size_mb": patterns.nbytes / (1024 * 1024),
            "intensity_range": [min_val, max_val],
            "mean_intensity": mean_val,
            "std_intensity": std_val,
        }


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
        "--sparse", "-s", action="store_true", help="Generate sparse CSC matrix format"
    )
    parser.add_argument(
        "--sparsity-threshold",
        type=float,
        default=0.01,
        help="Threshold for sparse matrix (default: 0.01)",
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

        if args.sparse:
            # Generate sparse CSC matrix
            logger.info("Generating sparse CSC matrix format...")
            sparse_matrix, led_mapping = generator.generate_sparse_csc_matrix(
                led_count=args.led_count,
                pattern_type=args.pattern_type,
                intensity_variation=not args.no_intensity_variation,
            )

            # Add sparse-specific metadata
            metadata.update(
                {
                    "format": "sparse_csc",
                    "sparsity_threshold": args.sparsity_threshold,
                }
            )

            # Save sparse matrix
            if not generator.save_sparse_matrix(
                sparse_matrix, led_mapping, args.output, metadata
            ):
                logger.error("Failed to save sparse matrix")
                return 1

            logger.info("Sparse matrix generation completed successfully")

        else:
            # Generate dense patterns (legacy format)
            logger.info("Generating dense pattern format...")
            patterns = generator.generate_patterns(
                led_count=args.led_count,
                pattern_type=args.pattern_type,
                intensity_variation=not args.no_intensity_variation,
            )

            # Get pattern info (memory-efficient)
            pattern_info = generator.get_pattern_info(patterns)
            logger.info("=== Pattern Generation Summary ===")
            for key, value in pattern_info.items():
                logger.info(f"{key}: {value}")

            # Save patterns
            if not generator.save_patterns(patterns, args.output, metadata):
                logger.error("Failed to save patterns")
                return 1

            logger.info("Dense pattern generation completed successfully")

        return 0

    except Exception as e:
        logger.error(f"Pattern generation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
