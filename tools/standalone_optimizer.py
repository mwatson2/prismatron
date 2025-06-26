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
from src.consumer.led_optimizer_dense import DenseLEDOptimizer, DenseOptimizationResult

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
    """Standalone LED optimization tool with dense, sparse, and mixed optimizers."""

    def __init__(self, diffusion_patterns_path: str, optimizer_type: str = "dense"):
        """Initialize optimizer with patterns file and optimizer type."""
        if not diffusion_patterns_path:
            raise ValueError("Diffusion patterns path is required")

        self.diffusion_patterns_path = diffusion_patterns_path
        self.optimizer_type = optimizer_type.lower()

        if self.optimizer_type == "mixed":
            # Use unified optimizer with mixed tensor mode
            self.optimizer = DenseLEDOptimizer(
                diffusion_patterns_path=diffusion_patterns_path, use_mixed_tensor=True
            )
            if not self.optimizer.initialize():
                raise RuntimeError("Failed to initialize mixed tensor optimizer")
            self.pipeline = None
        else:
            # Use shared optimization pipeline for dense/sparse
            use_dense = self.optimizer_type == "dense"
            self.pipeline = OptimizationPipeline(
                diffusion_patterns_path=diffusion_patterns_path, use_dense=use_dense
            )
            if not self.pipeline.initialize():
                raise RuntimeError("Failed to initialize optimization pipeline")
            self.optimizer = None

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
        """Run optimization on input image with detailed timing."""

        # Track timing for each phase
        total_start = time.time()

        if self.optimizer_type == "mixed":
            # Use mixed tensor optimizer directly
            # Phase 1: Load image
            load_start = time.time()
            target_image = self._load_image_mixed(input_path)
            load_time = time.time() - load_start

            # Phase 2: Optimize
            optimize_start = time.time()
            result = self.optimizer.optimize_frame(target_image, debug=True)
            optimize_time = time.time() - optimize_start

            # Phase 3: Render result (placeholder for now)
            render_start = time.time()
            rendered_result = self._render_result_mixed(result, target_image)
            render_time = time.time() - render_start

            # Phase 4: Save (if requested)
            save_time = 0.0
            if output_path:
                save_start = time.time()
                self._save_image_mixed(rendered_result, output_path)
                save_time = time.time() - save_start
        else:
            # Use pipeline for dense/sparse
            # Phase 1: Load image
            load_start = time.time()
            target_image = self.pipeline.load_image(input_path)
            load_time = time.time() - load_start

            # Phase 2: Optimize
            optimize_start = time.time()
            result = self.pipeline.optimize_image(target_image, max_iterations=None)
            optimize_time = time.time() - optimize_start

            # Phase 3: Render result
            render_start = time.time()
            rendered_result = self.pipeline.render_result(result)
            render_time = time.time() - render_start

            # Phase 4: Save (if requested)
            save_time = 0.0
            if output_path:
                save_start = time.time()
                self.pipeline.save_image(rendered_result, output_path)
                save_time = time.time() - save_start

        total_time = time.time() - total_start

        # Store timing breakdown in result for reporting
        result.timing_breakdown = {
            "load_time": load_time,
            "optimize_time": optimize_time,
            "render_time": render_time,
            "save_time": save_time,
            "total_time": total_time,
        }
        
        # DEBUG: Save LED values for comparison
        if self.optimizer_type == "mixed":
            np.savez("debug_mixed_led_values.npz", led_values=result.led_values)
            logger.info(f"Mixed LED values stats: min={result.led_values.min()}, max={result.led_values.max()}, mean={result.led_values.mean():.3f}")
        else:
            np.savez("debug_csc_led_values.npz", led_values=result.led_values)
            logger.info(f"CSC LED values stats: min={result.led_values.min()}, max={result.led_values.max()}, mean={result.led_values.mean():.3f}")

        # Show preview if requested
        if show_preview:
            self.show_preview(rendered_result, target_image)

        # Add rendered result to result object for compatibility
        result.rendered_result = rendered_result

        return result, target_image

    def _load_image_mixed(self, input_path: str) -> np.ndarray:
        """Load and resize image for mixed tensor optimization."""
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

    def _render_result_mixed(
        self, result: DenseOptimizationResult, target_image: np.ndarray
    ) -> np.ndarray:
        """Render optimization result for mixed tensor using CSC matrices."""
        logger.info("Rendering result using CSC matrices...")

        # Use the CSC matrices that were loaded alongside the mixed tensor
        # These are available in the unified optimizer when use_mixed_tensor=True
        A_r = self.optimizer._A_r_csc_gpu.tocsr()
        A_g = self.optimizer._A_g_csc_gpu.tocsr()
        A_b = self.optimizer._A_b_csc_gpu.tocsr()

        # Convert to CPU for rendering
        try:
            import cupy as cp

            A_r_cpu = A_r.get().tocsr()
            A_g_cpu = A_g.get().tocsr()
            A_b_cpu = A_b.get().tocsr()
        except Exception:
            A_r_cpu = A_r
            A_g_cpu = A_g
            A_b_cpu = A_b

        # Convert LED values from uint8 [0,255] to float32 [0,1]
        led_values_normalized = result.led_values.astype(np.float32) / 255.0

        logger.info(f"LED values shape: {result.led_values.shape}")
        logger.info(f"Dense reconstruction - RGB matrix shapes: {A_r_cpu.shape}")

        # Render each channel separately: A @ x
        rendered_r = A_r_cpu @ led_values_normalized[:, 0]  # (pixels,)
        rendered_g = A_g_cpu @ led_values_normalized[:, 1]
        rendered_b = A_b_cpu @ led_values_normalized[:, 2]

        # Reshape each channel separately then combine (matches CSC approach)
        rendered_image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)
        rendered_image[:, :, 0] = rendered_r.reshape(FRAME_HEIGHT, FRAME_WIDTH)
        rendered_image[:, :, 1] = rendered_g.reshape(FRAME_HEIGHT, FRAME_WIDTH)
        rendered_image[:, :, 2] = rendered_b.reshape(FRAME_HEIGHT, FRAME_WIDTH)

        # Convert back to uint8 [0, 255] and clip
        rendered_image = np.clip(rendered_image * 255.0, 0, 255).astype(np.uint8)

        logger.info(f"Rendered image shape: {rendered_image.shape}")
        return rendered_image

    def _save_image_mixed(self, image: np.ndarray, output_path: str) -> None:
        """Save image for mixed tensor optimization."""
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

        # Determine optimizer type
        if args.optimizer != "dense":
            optimizer_type = args.optimizer
        elif args.mixed:
            optimizer_type = "mixed"
        elif args.sparse:
            optimizer_type = "sparse"
        else:
            optimizer_type = "dense"

        logger.info(f"Using {optimizer_type} tensor optimizer")
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

            # Print core per-frame timing
            if "core_per_frame_time" in timing:
                core_time = timing["core_per_frame_time"]
                core_pct = core_time / timing["total_time"] * 100
                logger.info(f"Core per-frame:    {core_time:.3f}s ({core_pct:.1f}%)")
                fps = 1.0 / core_time if core_time > 0 else 0.0
                logger.info(f"Estimated FPS:     {fps:.1f}")
        else:
            logger.info(f"Total time: {result.optimization_time:.3f}s")

        # Print core optimizer timing details if available
        if (
            hasattr(result, "flop_info")
            and result.flop_info
            and "detailed_timing" in result.flop_info
        ):
            detailed = result.flop_info["detailed_timing"]
            logger.info("=== Core Optimizer Timing ===")
            if "atb_time" in detailed:
                logger.info(f"A^T*b calculation: {detailed['atb_time']:.3f}s")
            if "loop_time" in detailed:
                logger.info(f"Optimization loop: {detailed['loop_time']:.3f}s")
            if "einsum_time" in detailed:
                logger.info(f"  - Dense einsum:  {detailed['einsum_time']:.3f}s")
            if "step_size_time" in detailed:
                logger.info(f"  - Step size:     {detailed['step_size_time']:.3f}s")
            if "iterations_completed" in detailed:
                logger.info(f"  - Iterations:    {detailed['iterations_completed']}")
            if "core_optimization_time" in detailed:
                core_opt = detailed["core_optimization_time"]
                fps = 1.0 / core_opt if core_opt > 0 else 0.0
                logger.info(f"Core optimization: {core_opt:.3f}s ({fps:.1f} FPS)")

        # Print optimization quality metrics
        if hasattr(result, "error_metrics") and result.error_metrics:
            logger.info("=== Quality Metrics ===")
            logger.info(f"MSE: {result.error_metrics.get('mse', 'N/A'):.6f}")
            logger.info(f"RMSE: {result.error_metrics.get('rmse', 'N/A'):.6f}")

        # Print FLOP performance if available
        if hasattr(result, "flop_info") and result.flop_info:
            flop_info = result.flop_info
            logger.info("=== Performance Metrics ===")
            logger.info(f"Total FLOPs: {flop_info.get('total_flops', 0):,}")
            logger.info(f"GFLOPS/second: {flop_info.get('gflops_per_second', 0):.1f}")

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
