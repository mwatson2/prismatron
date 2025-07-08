#!/usr/bin/env python3
"""
Batch Image Optimizer Tool

This tool processes all images in the images/source directory and creates optimized
LED reconstructions in the images/optimized directory using the frame_optimizer utility.

Usage:
    python tools/batch_image_optimizer.py --pattern-file diffusion_patterns/synthetic_2624_fp16.npz
    python tools/batch_image_optimizer.py --pattern-file patterns.npz --max-iterations 10 --verbose
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import (
    FrameOptimizationResult,
    load_ata_inverse_from_pattern,
    optimize_frame_led_values,
)
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


class BatchImageOptimizer:
    """Batch processor for optimizing source images to LED displays."""

    def __init__(
        self,
        pattern_file: str,
        source_dir: str = "images/source",
        output_dir: str = "images/optimized",
        target_width: int = 800,
        target_height: int = 480,
    ):
        """
        Initialize batch optimizer.

        Args:
            pattern_file: Path to diffusion patterns file (.npz)
            source_dir: Directory containing source images
            output_dir: Directory to save optimized images
            target_width: Target frame width for optimization
            target_height: Target frame height for optimization
        """
        self.pattern_file = pattern_file
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_width = target_width
        self.target_height = target_height

        # Pattern data (loaded once)
        self.at_matrix: Optional[SingleBlockMixedSparseTensor] = None
        self.ata_matrix: Optional[DiagonalATAMatrix] = None
        self.ata_inverse: Optional[np.ndarray] = None
        self.led_positions: Optional[np.ndarray] = None

        # Statistics
        self.total_images = 0
        self.processed_images = 0
        self.failed_images = 0
        self.total_optimization_time = 0.0

    def load_pattern_data(self) -> bool:
        """Load diffusion pattern data from file."""
        try:
            logger.info(f"Loading pattern data from {self.pattern_file}")

            if not Path(self.pattern_file).exists():
                logger.error(f"Pattern file not found: {self.pattern_file}")
                return False

            # Load pattern file
            data = np.load(self.pattern_file, allow_pickle=True)
            logger.info(f"Pattern file keys: {list(data.keys())}")

            # Load mixed tensor (A^T matrix)
            if "mixed_tensor" not in data:
                logger.error("No mixed_tensor found in pattern file")
                return False

            mixed_tensor_dict = data["mixed_tensor"].item()
            self.at_matrix = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
            logger.info(f"Loaded A^T matrix: {self.at_matrix.batch_size} LEDs, {self.at_matrix.channels} channels")

            # Load DIA matrix (A^T A matrix)
            if "dia_matrix" not in data:
                logger.error("No dia_matrix found in pattern file")
                return False

            dia_dict = data["dia_matrix"].item()
            self.ata_matrix = DiagonalATAMatrix.from_dict(dia_dict)
            logger.info(f"Loaded A^T A matrix: {self.ata_matrix.led_count} LEDs, bandwidth {self.ata_matrix.bandwidth}")

            # Load ATA inverse
            self.ata_inverse = load_ata_inverse_from_pattern(self.pattern_file)
            if self.ata_inverse is None:
                logger.error("No ATA inverse found in pattern file")
                return False

            logger.info(f"Loaded A^T A inverse: shape {self.ata_inverse.shape}")

            # Load LED positions if available
            self.led_positions = data.get("led_positions", None)
            if self.led_positions is not None:
                logger.info(f"Loaded LED positions: {len(self.led_positions)} LEDs")

            return True

        except Exception as e:
            logger.error(f"Failed to load pattern data: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def find_source_images(self) -> List[Path]:
        """Find all image files in the source directory."""
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

        image_files = []
        if not self.source_dir.exists():
            logger.warning(f"Source directory does not exist: {self.source_dir}")
            return image_files

        for file_path in self.source_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                image_files.append(file_path)

        # Sort for consistent processing order
        image_files.sort()
        logger.info(f"Found {len(image_files)} image files in {self.source_dir}")

        return image_files

    def load_and_resize_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load image and resize to target dimensions.

        Args:
            image_path: Path to source image

        Returns:
            Image array in (height, width, 3) format, or None if failed
        """
        try:
            # Load image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            # Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to target dimensions
            if image.shape[:2] != (self.target_height, self.target_width):
                image = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_LANCZOS4)
                logger.debug(f"Resized {image_path.name} to {self.target_width}x{self.target_height}")

            # Ensure uint8 format
            if image.dtype != np.uint8:
                image = np.clip(image * 255, 0, 255).astype(np.uint8)

            return image

        except Exception as e:
            logger.error(f"Failed to load/resize image {image_path}: {e}")
            return None

    def optimize_image(
        self,
        image: np.ndarray,
        max_iterations: int = 5,
        convergence_threshold: float = 0.3,
        step_size_scaling: float = 0.9,
        debug: bool = False,
    ) -> Optional[FrameOptimizationResult]:
        """
        Optimize image using frame optimizer.

        Args:
            image: Input image (height, width, 3) uint8
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold
            step_size_scaling: Step size scaling factor
            debug: Enable debug output

        Returns:
            Optimization result or None if failed
        """
        try:
            # Convert image to planar format for optimizer
            if image.shape[-1] == 3:
                # Convert (H, W, 3) to (3, H, W)
                target_frame = np.transpose(image, (2, 0, 1))
            else:
                target_frame = image

            # Ensure correct data type
            target_frame = target_frame.astype(np.uint8)

            # Run optimization
            start_time = time.time()
            result = optimize_frame_led_values(
                target_frame=target_frame,
                at_matrix=self.at_matrix,
                ata_matrix=self.ata_matrix,
                ata_inverse=self.ata_inverse,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                step_size_scaling=step_size_scaling,
                compute_error_metrics=True,
                debug=debug,
                enable_timing=False,
            )
            optimization_time = time.time() - start_time
            self.total_optimization_time += optimization_time

            if debug:
                logger.info(f"Optimization completed in {optimization_time:.2f}s:")
                logger.info(f"  Iterations: {result.iterations}")
                logger.info(f"  Converged: {result.converged}")
                logger.info(f"  Error metrics: {result.error_metrics}")

            return result

        except Exception as e:
            logger.error(f"Failed to optimize image: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def led_values_to_image(self, led_values: np.ndarray) -> np.ndarray:
        """
        Convert LED values back to reconstructed image.

        Args:
            led_values: LED values (3, led_count) in [0, 255]

        Returns:
            Reconstructed image (height, width, 3) uint8
        """
        try:
            # Use the A^T matrix to reconstruct the image
            # led_values shape: (3, led_count)

            # Convert LED values to proper shape and CuPy format
            # Input: (3, led_count) -> Need: (led_count, 3)
            import cupy as cp

            if led_values.shape == (3, self.at_matrix.batch_size):
                led_values_transposed = led_values.T  # Shape: (led_count, 3)
            else:
                led_values_transposed = led_values

            if isinstance(led_values_transposed, np.ndarray):
                led_values_gpu = cp.asarray(led_values_transposed)
            else:
                led_values_gpu = led_values_transposed

            # Reconstruct using the mixed tensor forward pass
            reconstructed = self.at_matrix.forward_pass_3d(led_values_gpu)  # Shape: (3, H, W)

            # Convert back to CPU if needed
            if hasattr(reconstructed, "get"):
                reconstructed = reconstructed.get()

            # Convert from planar (3, H, W) to standard (H, W, 3) format
            image = np.transpose(reconstructed, (1, 2, 0))

            # Ensure uint8 format and proper range
            image = np.clip(image, 0, 255).astype(np.uint8)

            return image

        except Exception as e:
            logger.error(f"Failed to convert LED values to image: {e}")
            import traceback

            logger.error(traceback.format_exc())

            # Return black image as fallback
            return np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

    def save_optimized_image(self, image: np.ndarray, output_path: Path) -> bool:
        """Save optimized image to file."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Save image
            success = cv2.imwrite(str(output_path), image_bgr)
            if not success:
                logger.error(f"Failed to save image: {output_path}")
                return False

            logger.debug(f"Saved optimized image: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save optimized image {output_path}: {e}")
            return False

    def process_image(
        self,
        image_path: Path,
        max_iterations: int = 5,
        convergence_threshold: float = 0.3,
        step_size_scaling: float = 0.9,
        debug: bool = False,
        overwrite: bool = False,
    ) -> bool:
        """
        Process a single image: load, optimize, reconstruct, and save.

        Args:
            image_path: Path to source image
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold
            step_size_scaling: Step size scaling factor
            debug: Enable debug output
            overwrite: Overwrite existing optimized images

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate output path
            output_filename = f"{image_path.stem}_optimized{image_path.suffix}"
            output_path = self.output_dir / output_filename

            # Check if output already exists
            if output_path.exists() and not overwrite:
                logger.info(f"Skipping {image_path.name} (output exists, use --overwrite to replace)")
                return True

            logger.info(f"Processing {image_path.name}...")

            # Load and resize image
            image = self.load_and_resize_image(image_path)
            if image is None:
                return False

            # Optimize image
            optimization_result = self.optimize_image(
                image,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                step_size_scaling=step_size_scaling,
                debug=debug,
            )
            if optimization_result is None:
                return False

            # Convert LED values back to image
            reconstructed_image = self.led_values_to_image(optimization_result.led_values)

            # Save optimized image
            if not self.save_optimized_image(reconstructed_image, output_path):
                return False

            logger.info(f"Successfully processed {image_path.name} -> {output_filename}")
            if debug:
                logger.info(
                    f"  Optimization: {optimization_result.iterations} iterations, "
                    f"converged: {optimization_result.converged}"
                )
                if optimization_result.error_metrics:
                    logger.info(f"  Error metrics: {optimization_result.error_metrics}")

            return True

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def process_all_images(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.3,
        step_size_scaling: float = 0.9,
        debug: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Process all images in the source directory."""
        # Find source images
        image_files = self.find_source_images()
        if not image_files:
            logger.warning("No images found to process")
            return

        self.total_images = len(image_files)
        logger.info(f"Processing {self.total_images} images...")

        # Process each image
        start_time = time.time()
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"[{i}/{self.total_images}] Processing {image_path.name}")

            success = self.process_image(
                image_path,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                step_size_scaling=step_size_scaling,
                debug=debug,
                overwrite=overwrite,
            )

            if success:
                self.processed_images += 1
            else:
                self.failed_images += 1
                logger.error(f"Failed to process {image_path.name}")

        # Print summary
        total_time = time.time() - start_time
        logger.info(f"\n=== Batch Processing Summary ===")
        logger.info(f"Total images: {self.total_images}")
        logger.info(f"Processed successfully: {self.processed_images}")
        logger.info(f"Failed: {self.failed_images}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Total optimization time: {self.total_optimization_time:.2f}s")
        if self.processed_images > 0:
            logger.info(f"Average time per image: {total_time / self.processed_images:.2f}s")
            logger.info(f"Average optimization time: {self.total_optimization_time / self.processed_images:.2f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch optimize images for LED display")
    parser.add_argument(
        "--pattern-file",
        type=str,
        required=True,
        help="Path to diffusion patterns file (.npz)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="images/source",
        help="Source images directory (default: images/source)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images/optimized",
        help="Output directory for optimized images (default: images/optimized)",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=800,
        help="Target frame width (default: 800)",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=480,
        help="Target frame height (default: 480)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum optimization iterations (default: 5)",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.3,
        help="Convergence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--step-size-scaling",
        type=float,
        default=0.9,
        help="Step size scaling factor (default: 0.9)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing optimized images",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Create batch optimizer
        optimizer = BatchImageOptimizer(
            pattern_file=args.pattern_file,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            target_width=args.target_width,
            target_height=args.target_height,
        )

        # Load pattern data
        if not optimizer.load_pattern_data():
            logger.error("Failed to load pattern data")
            return 1

        # Process all images
        optimizer.process_all_images(
            max_iterations=args.max_iterations,
            convergence_threshold=args.convergence_threshold,
            step_size_scaling=args.step_size_scaling,
            debug=args.debug,
            overwrite=args.overwrite,
        )

        logger.info("Batch processing completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
