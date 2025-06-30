#!/usr/bin/env python3
"""
Standalone LED Optimization Tool.

This tool performs LED optimization on input images using captured
or synthetic diffusion patterns and renders the result using the
production LEDOptimizer class.

Usage:
    python standalone_optimizer.py --input image.jpg --patterns captured.npz \
        --output result.png
    python standalone_optimizer.py --input image.jpg --patterns synthetic.npz \
        --output result.png --preview
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

# Import frame optimization function and supporting classes
from src.utils.frame_optimizer import (
    FrameOptimizationResult,
    optimize_frame_led_values,
    optimize_frame_with_dia_matrix,
    optimize_frame_with_mixed_tensor,
)
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


class StandaloneOptimizer:
    """Standalone LED optimization tool using the new frame_optimizer function."""

    def __init__(self, diffusion_patterns_path: str, optimizer_type: str = "sparse"):
        """Initialize optimizer with patterns file and optimizer type."""
        if not diffusion_patterns_path:
            raise ValueError("Diffusion patterns path is required")

        self.diffusion_patterns_path = diffusion_patterns_path
        self.optimizer_type = optimizer_type.lower()

        logger.info(f"Loading patterns from: {diffusion_patterns_path}")

        # Load the diffusion patterns
        patterns_data = np.load(diffusion_patterns_path, allow_pickle=True)

        if self.optimizer_type == "mixed":
            # Load mixed tensor format
            self.mixed_tensor = SingleBlockMixedSparseTensor.from_npz(
                diffusion_patterns_path
            )
            # For mixed tensor, we need dense A^T A matrix
            # This would normally be computed from the patterns
            logger.warning(
                "Mixed tensor mode requires dense A^T A matrix - using sparse mode instead"
            )
            self.optimizer_type = "sparse"

        if self.optimizer_type in ["sparse", "dense"]:
            # Load CSC sparse matrices and DIA A^T A matrix
            # Extract CSC data from nested dictionary format
            if (
                "diffusion_matrix" in patterns_data
                and "csc_data" in patterns_data["diffusion_matrix"].item()
            ):
                csc_data_dict = patterns_data["diffusion_matrix"].item()
                self.diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
            else:
                self.diffusion_csc = LEDDiffusionCSCMatrix.from_dict(patterns_data)

            # Try to load DIA matrix, fallback to creating one if needed
            try:
                if "dia_matrix" in patterns_data:
                    dia_data_dict = patterns_data["dia_matrix"].item()
                    self.dia_matrix = DiagonalATAMatrix.from_dict(dia_data_dict)
                else:
                    raise KeyError("DIA matrix not in patterns file")
            except Exception as e:
                logger.warning(f"Could not load DIA matrix: {e}")
                logger.info("Creating DIA matrix from CSC patterns...")
                self.dia_matrix = DiagonalATAMatrix(
                    led_count=self.diffusion_csc.led_count
                )
                csc_matrix = self.diffusion_csc.to_csc_matrix()
                led_positions = patterns_data["led_positions"]
                self.dia_matrix.build_from_diffusion_matrix(csc_matrix, led_positions)

        self.led_count = self.diffusion_csc.led_count
        logger.info(
            f"Initialized {self.optimizer_type} optimizer with {self.led_count} LEDs"
        )

    def show_preview(self, rendered_result: np.ndarray, target_image: np.ndarray):
        """Show side-by-side comparison."""
        if rendered_result is None:
            logger.warning("No rendered result to preview")
            return

        # Create side-by-side comparison
        target = target_image
        rendered = rendered_result

        # Ensure same height
        h = max(target.shape[0], rendered.shape[0])
        target_resized = cv2.resize(target, (target.shape[1], h))
        rendered_resized = cv2.resize(rendered, (rendered.shape[1], h))

        comparison = np.hstack([target_resized, rendered_resized])

        # Convert to BGR for OpenCV display
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)

        cv2.imshow("Optimization Result (Original | Rendered)", comparison_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        show_preview: bool = False,
    ):
        """Run optimization on input image using the new frame_optimizer function."""

        # Track timing for each phase
        total_start = time.time()

        # Phase 1: Load and prepare image
        load_start = time.time()
        original_image = self._load_image(input_path)
        load_time = time.time() - load_start

        # Phase 2: Run optimization using the new frame optimizer function
        optimize_start = time.time()

        logger.info(f"Running {self.optimizer_type} optimization...")

        if self.optimizer_type == "mixed":
            # Mixed tensor optimization (if we had the dense ATA matrix)
            result = optimize_frame_with_mixed_tensor(
                target_frame=original_image,
                mixed_tensor=self.mixed_tensor,
                ata_dense=None,  # Would need to compute this
                max_iterations=10,
                debug=True,
            )
        else:
            # Sparse optimization with DIA matrix
            result = optimize_frame_with_dia_matrix(
                target_frame=original_image,
                diffusion_csc=self.diffusion_csc,
                dia_matrix=self.dia_matrix,
                max_iterations=10,
                compute_error_metrics=True,
                debug=True,
            )

        optimize_time = time.time() - optimize_start

        # Phase 3: Render result
        render_start = time.time()
        rendered_result = self._render_result(result, original_image)
        render_time = time.time() - render_start

        # Phase 4: Save (if requested)
        save_time = 0.0
        if output_path:
            save_start = time.time()
            self._save_image(rendered_result, output_path)
            save_time = time.time() - save_start

        total_time = time.time() - total_start

        logger.info("=== Optimization Summary ===")
        logger.info(f"Optimization time: {optimize_time:.3f}s")
        logger.info(f"Converged: {result.converged} in {result.iterations} iterations")
        if result.error_metrics:
            logger.info(f"MSE: {result.error_metrics.get('mse', 'N/A'):.6f}")
            logger.info(f"PSNR: {result.error_metrics.get('psnr', 'N/A'):.2f} dB")

        # Store timing breakdown for compatibility
        result.timing_breakdown = {
            "load_time": load_time,
            "optimize_time": optimize_time,
            "render_time": render_time,
            "save_time": save_time,
            "total_time": total_time,
        }

        # DEBUG: Save LED values for comparison
        np.savez(
            f"debug_{self.optimizer_type}_led_values.npz", led_values=result.led_values
        )
        logger.info(
            f"LED values stats: min={result.led_values.min()}, "
            f"max={result.led_values.max()}, mean={result.led_values.mean():.3f}"
        )

        # Show preview if requested
        if show_preview:
            self.show_preview(rendered_result, original_image)

        # Add rendered result to result object for compatibility
        result.rendered_result = rendered_result

        return result, original_image

    def _load_image(self, input_path: str) -> np.ndarray:
        """Load and resize image for optimization."""
        if PIL_AVAILABLE:
            try:
                image = Image.open(input_path).convert("RGB")
                image = image.resize((FRAME_WIDTH, FRAME_HEIGHT))
                return np.array(image, dtype=np.uint8)
            except Exception as e:
                logger.warning(f"PIL failed to load {input_path}: {e}")

        # Fallback to OpenCV
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")

        # Convert BGR to RGB and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
        return image.astype(np.uint8)

    def _render_result(
        self, result: FrameOptimizationResult, target_image: np.ndarray
    ) -> np.ndarray:
        """Render optimization result using CSC matrices."""
        logger.info("Rendering result using CSC forward pass...")

        # Get the CSC matrix for forward rendering
        csc_A = self.diffusion_csc.to_csc_matrix()  # Shape: (pixels, led_count*3)
        led_count = result.led_values.shape[1]

        # Convert LED values from uint8 [0,255] to float32 [0,1]
        led_values_normalized = (
            result.led_values.astype(np.float32) / 255.0
        )  # Shape: (3, led_count)

        logger.info(f"LED values shape: {result.led_values.shape}")
        logger.info(f"CSC matrix shape: {csc_A.shape}")

        # Initialize rendered frame
        rendered_image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)

        # Process each channel separately
        for channel in range(3):
            # Get LED values for this channel
            led_channel = led_values_normalized[channel]  # Shape: (led_count,)

            # Extract A matrix columns for this channel
            # Channel 0 (R): columns 0, 3, 6, 9, ...
            # Channel 1 (G): columns 1, 4, 7, 10, ...
            # Channel 2 (B): columns 2, 5, 8, 11, ...
            channel_cols = np.arange(channel, csc_A.shape[1], 3)
            A_channel = csc_A[:, channel_cols]  # Shape: (pixels, led_count)

            # Forward pass: A @ led_values
            rendered_channel = A_channel @ led_channel  # Shape: (pixels,)

            # Reshape to spatial dimensions
            rendered_image[:, :, channel] = rendered_channel.reshape(
                FRAME_HEIGHT, FRAME_WIDTH
            )

        # Convert back to uint8 [0, 255] and clip
        rendered_image = np.clip(rendered_image * 255.0, 0, 255).astype(np.uint8)

        logger.info(f"Rendered image shape: {rendered_image.shape}")
        return rendered_image

    def _save_image(self, image: np.ndarray, output_path: str) -> None:
        """Save image."""
        if PIL_AVAILABLE:
            try:
                pil_image = Image.fromarray(image.astype(np.uint8))
                pil_image.save(output_path)
                return
            except Exception as e:
                logger.warning(f"PIL failed to save {output_path}: {e}")

        # Fallback to OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Standalone LED Optimization Tool")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--patterns", "-p", help="Diffusion patterns file (.npz)")
    parser.add_argument(
        "--synthetic",
        "-s",
        action="store_true",
        help="[DEPRECATED] Use pre-generated synthetic patterns from "
        "generate_synthetic_patterns.py instead",
    )
    parser.add_argument("--output", "-o", help="Output image path")
    parser.add_argument(
        "--preview", action="store_true", help="Show preview comparison"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--test", action="store_true", help="Use fewer LEDs for faster testing"
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Use sparse optimizer instead of dense tensor optimizer (default: dense)",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Use mixed tensor optimizer with custom CUDA kernels",
    )
    parser.add_argument(
        "--optimizer",
        choices=["dense", "sparse", "mixed"],
        default="dense",
        help="Optimizer type to use (default: dense)",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Validate inputs
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    if args.patterns:
        patterns_file = (
            f"{args.patterns}.npz"
            if not args.patterns.endswith(".npz")
            else args.patterns
        )
        if not Path(patterns_file).exists():
            logger.error(f"Patterns file not found: {patterns_file}")
            return 1

    if not args.patterns:
        logger.error("Must specify --patterns with path to diffusion patterns file")
        logger.error(
            "Generate synthetic patterns first with: python tools/generate_synthetic_patterns.py"
        )
        return 1

    try:
        # Test mode note: LED count is now determined by the patterns file
        if args.test:
            logger.info("Test mode: LED count determined by patterns file")

        # Determine optimizer type (default to sparse for new frame optimizer)
        if args.optimizer != "dense":
            optimizer_type = args.optimizer
        elif args.mixed:
            optimizer_type = "mixed"
        elif args.sparse:
            optimizer_type = "sparse"
        else:
            optimizer_type = "sparse"  # Default to sparse for frame optimizer

        logger.info(f"Using {optimizer_type} frame optimizer")
        optimizer = StandaloneOptimizer(
            diffusion_patterns_path=args.patterns, optimizer_type=optimizer_type
        )

        # Run optimization
        result, target_image = optimizer.run(
            input_path=args.input, output_path=args.output, show_preview=args.preview
        )

        # Print summary
        logger.info("=== Optimization Summary ===")
        logger.info(f"Input: {args.input}")
        logger.info(f"Target shape: {target_image.shape}")
        logger.info(f"LED count: {result.led_values.shape[0]}")

        # Print detailed timing breakdown
        if hasattr(result, "timing_breakdown"):
            timing = result.timing_breakdown
            logger.info("=== Timing Breakdown ===")
            load_pct = timing["load_time"] / timing["total_time"] * 100
            opt_pct = timing["optimize_time"] / timing["total_time"] * 100
            render_pct = timing["render_time"] / timing["total_time"] * 100

            logger.info(
                f"Image loading:     {timing['load_time']:.3f}s ({load_pct:.1f}%)"
            )
            logger.info(
                f"LED optimization:  {timing['optimize_time']:.3f}s ({opt_pct:.1f}%)"
            )
            logger.info(
                f"Result rendering:  {timing['render_time']:.3f}s ({render_pct:.1f}%)"
            )
            if timing["save_time"] > 0:
                save_pct = timing["save_time"] / timing["total_time"] * 100
                logger.info(
                    f"Image saving:      {timing['save_time']:.3f}s ({save_pct:.1f}%)"
                )
            logger.info(f"Total time:        {timing['total_time']:.3f}s")

            # Estimate FPS from optimization time
            fps = 1.0 / timing["optimize_time"] if timing["optimize_time"] > 0 else 0.0
            logger.info(f"Estimated FPS:     {fps:.1f}")

        # Print step size information if available
        if hasattr(result, "step_sizes") and result.step_sizes is not None:
            logger.info("=== Optimization Details ===")
            logger.info(
                f"Step sizes: min={result.step_sizes.min():.6f}, "
                f"max={result.step_sizes.max():.6f}, avg={result.step_sizes.mean():.6f}"
            )

        # Print optimization quality metrics
        if hasattr(result, "error_metrics") and result.error_metrics:
            logger.info("=== Quality Metrics ===")
            for metric, value in result.error_metrics.items():
                if isinstance(value, float):
                    if "psnr" in metric.lower():
                        logger.info(f"{metric.upper()}: {value:.2f} dB")
                    else:
                        logger.info(f"{metric.upper()}: {value:.6f}")
                else:
                    logger.info(f"{metric.upper()}: {value}")

        logger.info(
            f"LED values range: [{result.led_values.min()}, {result.led_values.max()}]"
        )

        if args.output:
            logger.info(f"Output saved: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
