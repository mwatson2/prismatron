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

# Constants (can be overridden by command line)
DEFAULT_FRAME_WIDTH = 800
DEFAULT_FRAME_HEIGHT = 480
DEFAULT_LED_COUNT = 3200

logger = logging.getLogger(__name__)


class SyntheticPatternGenerator:
    """Generator for synthetic LED diffusion patterns."""

    def __init__(
        self,
        frame_width: int = DEFAULT_FRAME_WIDTH,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        seed: Optional[int] = None,
    ):
        """
        Initialize pattern generator.

        Args:
            frame_width: Width of output frames
            frame_height: Height of output frames
            seed: Random seed for reproducible patterns
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Using random seed: {seed}")

        # Generate LED positions (random for synthetic patterns)
        self.led_positions = None

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
        logger.info(f"Generated {led_count} LED positions")
        return positions

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

    def get_pattern_info(self, patterns: np.ndarray) -> dict:
        """
        Get information about generated patterns.

        Args:
            patterns: Patterns array

        Returns:
            Dictionary with pattern statistics
        """
        return {
            "shape": list(patterns.shape),
            "dtype": str(patterns.dtype),
            "led_count": patterns.shape[0],
            "channels": patterns.shape[1],
            "frame_height": patterns.shape[2],
            "frame_width": patterns.shape[3],
            "memory_size_mb": patterns.nbytes / (1024 * 1024),
            "intensity_range": [float(patterns.min()), float(patterns.max())],
            "mean_intensity": float(patterns.mean()),
            "std_intensity": float(patterns.std()),
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
        )

        # Generate patterns
        patterns = generator.generate_patterns(
            led_count=args.led_count,
            pattern_type=args.pattern_type,
            intensity_variation=not args.no_intensity_variation,
        )

        # Get pattern info
        pattern_info = generator.get_pattern_info(patterns)
        logger.info("=== Pattern Generation Summary ===")
        for key, value in pattern_info.items():
            logger.info(f"{key}: {value}")

        # Save patterns
        metadata = {
            "pattern_type": args.pattern_type,
            "seed": args.seed,
            "intensity_variation": not args.no_intensity_variation,
            "command_line": " ".join(sys.argv),
        }

        if not generator.save_patterns(patterns, args.output, metadata):
            logger.error("Failed to save patterns")
            return 1

        logger.info("Pattern generation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Pattern generation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
