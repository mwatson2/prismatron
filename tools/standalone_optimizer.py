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


class ProductionLEDOptimizerWrapper:
    """Wrapper around production LEDOptimizer for standalone use."""

    def __init__(self, diffusion_patterns_path: str):
        """Initialize wrapper with production optimizer."""
        self.optimizer = LEDOptimizer(
            diffusion_patterns_path=diffusion_patterns_path,
            device=None,  # Auto-detect best device
        )
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize the production optimizer."""
        self.initialized = self.optimizer.initialize()
        if self.initialized:
            stats = self.optimizer.get_optimizer_stats()
            logger.info(f"Production optimizer initialized")
            logger.info(f"Device: {stats['device']}")
            logger.info(f"LED count: {stats['led_count']}")
            logger.info(f"Patterns shape: {stats['diffusion_patterns_shape']}")
        return self.initialized

    def optimize_image(
        self, target_image: np.ndarray, max_iterations: int = 50
    ) -> OptimizationResult:
        """Optimize LED values for target image using production optimizer."""
        if not self.initialized:
            raise RuntimeError("Optimizer not initialized. Call initialize() first.")

        logger.info(f"Optimizing image shape: {target_image.shape}")

        # Use production optimizer
        result = self.optimizer.optimize_frame(
            target_frame=target_image, max_iterations=max_iterations
        )

        logger.info(f"Optimization completed in {result.optimization_time:.3f}s")
        logger.info(f"Iterations: {result.iterations}")
        logger.info(f"Converged: {result.converged}")
        logger.info(f"MSE: {result.error_metrics.get('mse', 'N/A')}")
        logger.info(
            f"LED values range: [{result.led_values.min()}, {result.led_values.max()}]"
        )

        return result

    def render_result(self, result: OptimizationResult) -> np.ndarray:
        """Render optimization result to image."""
        if not self.initialized:
            raise RuntimeError("Optimizer not initialized")

        # Get diffusion patterns from optimizer
        stats = self.optimizer.get_optimizer_stats()
        patterns_shape = stats["diffusion_patterns_shape"]

        # Access the internal patterns (this is a bit of a hack but needed for rendering)
        patterns_tensor = self.optimizer._diffusion_patterns
        led_values_tensor = torch.from_numpy(result.led_values.astype(np.float32)).to(
            self.optimizer.device
        )

        # Render using the same approach as the production optimizer
        led_weights = led_values_tensor.unsqueeze(1).unsqueeze(
            1
        )  # (led_count, 1, 1, 3)
        reconstructed = torch.sum(
            patterns_tensor * led_weights, dim=0
        )  # (height, width, 3)

        # Convert to numpy and uint8
        result_np = reconstructed.detach().cpu().numpy()
        result_np = np.clip(result_np, 0, 255).astype(np.uint8)

        return result_np


class StandaloneOptimizer:
    """Standalone LED optimization tool using production optimizer."""

    def __init__(self, diffusion_patterns_path: str):
        """Initialize optimizer with patterns file."""
        if not diffusion_patterns_path or not Path(diffusion_patterns_path).exists():
            raise ValueError(
                f"Diffusion patterns file not found: {diffusion_patterns_path}"
            )

        self.optimizer_wrapper = ProductionLEDOptimizerWrapper(diffusion_patterns_path)

        # Initialize the production optimizer
        if not self.optimizer_wrapper.initialize():
            raise RuntimeError("Failed to initialize production LED optimizer")

    def load_image(self, image_path: str) -> np.ndarray:
        """Load and resize image to target dimensions."""
        logger.info(f"Loading image: {image_path}")

        # Try PIL first if available
        if PIL_AVAILABLE:
            try:
                img = Image.open(image_path)
                img = img.convert("RGB")
                img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.LANCZOS)
                image = np.array(img)
            except Exception as e:
                logger.warning(f"PIL failed: {e}, falling back to OpenCV")
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))

        logger.info(f"Loaded image shape: {image.shape}")
        return image

    def save_result(self, result: OptimizationResult, output_path: str):
        """Save optimization result."""
        if result.rendered_result is not None:
            if PIL_AVAILABLE:
                img = Image.fromarray(result.rendered_result)
                img.save(output_path)
            else:
                # Convert RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(result.rendered_result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, bgr_image)

            logger.info(f"Saved result to: {output_path}")
        else:
            logger.warning("No rendered result to save")

    def show_preview(self, result: OptimizationResult, target_image: np.ndarray):
        """Show side-by-side comparison."""
        if result.rendered_result is None:
            logger.warning("No rendered result to preview")
            return

        # Create side-by-side comparison
        target = target_image
        rendered = result.rendered_result

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
        # Load input image
        target_image = self.load_image(input_path)

        # Perform optimization using production optimizer
        result = self.optimizer_wrapper.optimize_image(target_image, max_iterations=50)

        # Render the result for saving/preview
        rendered_result = self.optimizer_wrapper.render_result(result)

        # Create a result object with rendered image
        result.rendered_result = rendered_result

        # Save result if output path provided
        if output_path:
            self.save_result(result, output_path)

        # Show preview if requested
        if show_preview:
            self.show_preview(result, target_image)

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
