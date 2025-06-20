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
from src.consumer.led_optimizer import LEDOptimizer, OptimizationResult

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


# Import shared utilities
from src.utils.optimization_utils import OptimizationPipeline


class StandaloneOptimizer:
    """Standalone LED optimization tool using shared OptimizationPipeline."""

    def __init__(self, diffusion_patterns_path: str):
        """Initialize optimizer with patterns file."""
        if not diffusion_patterns_path:
            raise ValueError("Diffusion patterns path is required")

        # Create shared optimization pipeline
        self.pipeline = OptimizationPipeline(
            diffusion_patterns_path=diffusion_patterns_path, use_gpu=True
        )

        # Initialize the pipeline
        if not self.pipeline.initialize():
            raise RuntimeError("Failed to initialize optimization pipeline")

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
        """Run optimization on input image."""
        # Use shared pipeline for complete workflow
        target_image, result, rendered_result = self.pipeline.run_full_pipeline(
            input_path, max_iterations=50
        )

        # Save result if output path provided
        if output_path:
            self.pipeline.save_image(rendered_result, output_path)

        # Show preview if requested
        if show_preview:
            self.show_preview(rendered_result, target_image)

        # Add rendered result to result object for compatibility
        result.rendered_result = rendered_result

        return result, target_image


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

    if args.patterns and not Path(args.patterns).exists():
        logger.error(f"Patterns file not found: {args.patterns}")
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

        # Create optimizer
        optimizer = StandaloneOptimizer(diffusion_patterns_path=args.patterns)

        # Run optimization
        result, target_image = optimizer.run(
            input_path=args.input, output_path=args.output, show_preview=args.preview
        )

        # Print summary
        logger.info("=== Optimization Summary ===")
        logger.info(f"Input: {args.input}")
        logger.info(f"Target shape: {target_image.shape}")
        logger.info(f"LED count: {result.led_values.shape[0]}")
        logger.info(f"Optimization time: {result.optimization_time:.3f}s")
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
