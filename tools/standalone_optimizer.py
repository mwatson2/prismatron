#!/usr/bin/env python3
"""
Standalone LED Optimization Tool.

This tool performs LED optimization on input images using either captured
or synthetic diffusion patterns and renders the result.

Usage:
    python standalone_optimizer.py --input image.jpg --patterns captured.npz --output result.png
    python standalone_optimizer.py --input image.jpg --synthetic --output result.png --preview
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from consumer.led_optimizer import LEDOptimizer

logger = logging.getLogger(__name__)


class StandaloneOptimizer:
    """Standalone LED optimization tool."""

    def __init__(
        self, diffusion_patterns_path: Optional[str] = None, use_synthetic: bool = True
    ):
        """
        Initialize optimizer.

        Args:
            diffusion_patterns_path: Path to captured diffusion patterns (.npz)
            use_synthetic: Generate synthetic patterns if no file provided
        """
        self.diffusion_patterns_path = diffusion_patterns_path
        self.use_synthetic = use_synthetic

        # LED optimizer
        self.optimizer: Optional[LEDOptimizer] = None
        self.diffusion_patterns: Optional[np.ndarray] = None

    def initialize(self) -> bool:
        """Initialize the LED optimizer."""
        try:
            # Load or generate diffusion patterns
            if not self._load_diffusion_patterns():
                return False

            # Initialize LED optimizer with patterns
            self.optimizer = LEDOptimizer(
                diffusion_patterns_path=None,  # We'll set patterns directly
                led_count=LED_COUNT,
            )

            # Set the diffusion patterns directly
            if hasattr(self.optimizer, "_diffusion_patterns"):
                self.optimizer._diffusion_patterns = self.diffusion_patterns
                self.optimizer._patterns_loaded = True
                logger.info("Diffusion patterns loaded into optimizer")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            return False

    def _load_diffusion_patterns(self) -> bool:
        """Load diffusion patterns from file or generate synthetic ones."""
        try:
            if (
                self.diffusion_patterns_path
                and Path(self.diffusion_patterns_path).exists()
            ):
                logger.info(
                    f"Loading diffusion patterns from {self.diffusion_patterns_path}"
                )
                return self._load_captured_patterns()
            elif self.use_synthetic:
                logger.info("Generating synthetic diffusion patterns")
                return self._generate_synthetic_patterns()
            else:
                logger.error(
                    "No diffusion patterns provided and synthetic generation disabled"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to load diffusion patterns: {e}")
            return False

    def _load_captured_patterns(self) -> bool:
        """Load captured diffusion patterns from .npz file."""
        try:
            data = np.load(self.diffusion_patterns_path, allow_pickle=True)
            self.diffusion_patterns = data["diffusion_patterns"]

            # Validate shape
            expected_shape = (LED_COUNT, 3, FRAME_HEIGHT, FRAME_WIDTH)
            if self.diffusion_patterns.shape != expected_shape:
                logger.error(
                    f"Invalid pattern shape: {self.diffusion_patterns.shape}, expected: {expected_shape}"
                )
                return False

            logger.info(f"Loaded captured patterns: {self.diffusion_patterns.shape}")
            return True

        except Exception as e:
            logger.error(f"Failed to load captured patterns: {e}")
            return False

    def _generate_synthetic_patterns(self) -> bool:
        """Generate synthetic diffusion patterns."""
        try:
            logger.info("Generating synthetic diffusion patterns...")

            # Create synthetic patterns with realistic LED diffusion
            # Using uint8 to save memory: 3200×3×480×800×1 = ~3.5GB vs 14GB for float32
            self.diffusion_patterns = np.zeros(
                (LED_COUNT, 3, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8
            )

            # Generate random LED positions (reproducible)
            np.random.seed(42)
            led_positions = np.random.randint(
                20, min(FRAME_WIDTH, FRAME_HEIGHT) - 20, size=(LED_COUNT, 2)
            )

            for led_idx in range(LED_COUNT):
                x_center, y_center = led_positions[led_idx]

                # Ensure positions are within valid range
                x_center = max(20, min(x_center, FRAME_WIDTH - 20))
                y_center = max(20, min(y_center, FRAME_HEIGHT - 20))

                for channel in range(3):  # R, G, B
                    # Create realistic Gaussian diffusion pattern
                    pattern = self._create_led_diffusion(
                        x_center,
                        y_center,
                        sigma_x=np.random.uniform(25, 80),
                        sigma_y=np.random.uniform(25, 80),
                        intensity=np.random.uniform(100, 255),
                        asymmetry=np.random.uniform(0.7, 1.3),
                    )

                    self.diffusion_patterns[led_idx, channel] = pattern

            logger.info(
                f"Generated synthetic patterns: {self.diffusion_patterns.shape}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to generate synthetic patterns: {e}")
            return False

    def _create_led_diffusion(
        self,
        x_center: int,
        y_center: int,
        sigma_x: float,
        sigma_y: float,
        intensity: float,
        asymmetry: float = 1.0,
    ) -> np.ndarray:
        """
        Create realistic LED diffusion pattern with asymmetry and falloff.

        Args:
            x_center, y_center: LED position
            sigma_x, sigma_y: Gaussian spread parameters
            intensity: Peak intensity
            asymmetry: Asymmetry factor

        Returns:
            2D diffusion pattern
        """
        # Create coordinate grids
        x = np.arange(FRAME_WIDTH)
        y = np.arange(FRAME_HEIGHT)
        X, Y = np.meshgrid(x, y)

        # Apply asymmetry
        sigma_y_eff = sigma_y * asymmetry

        # Calculate Gaussian with some noise for realism
        pattern = intensity * np.exp(
            -(
                (X - x_center) ** 2 / (2 * sigma_x**2)
                + (Y - y_center) ** 2 / (2 * sigma_y_eff**2)
            )
        )

        # Add subtle noise for realism (scale to uint8 range)
        noise = np.random.normal(0, 5, pattern.shape)  # Scale noise for uint8
        pattern = np.clip(pattern + noise, 0, 255)

        # Add some subtle structured variation (simulating LED panel structure)
        grid_x = np.sin(X * 0.1) * 12  # Scale grid effects for uint8
        grid_y = np.sin(Y * 0.1) * 12
        pattern = np.clip(pattern + grid_x + grid_y, 0, 255)

        # Clip to valid uint8 range and convert
        return np.clip(pattern, 0, 255).astype(np.uint8)

    def load_input_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess input image.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image as (FRAME_HEIGHT, FRAME_WIDTH, 3) float array, or None if failed
        """
        try:
            # Load image
            if PIL_AVAILABLE:
                img = Image.open(image_path)
                img = img.convert("RGB")
                image = np.array(img)
            else:
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            logger.info(f"Loaded image: {image.shape} from {image_path}")

            # Resize to target dimensions while maintaining aspect ratio
            processed_image = self._resize_and_crop_image(image)

            # Convert to float and normalize
            processed_image = processed_image.astype(np.float32) / 255.0

            logger.info(f"Processed image: {processed_image.shape}")
            return processed_image

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

    def _resize_and_crop_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize and crop image to target dimensions (800x480) with 5:3 aspect ratio.

        Args:
            image: Input image array

        Returns:
            Resized and cropped image
        """
        src_h, src_w = image.shape[:2]
        target_aspect = FRAME_WIDTH / FRAME_HEIGHT  # 5:3
        src_aspect = src_w / src_h

        # Determine crop dimensions to maintain 5:3 aspect ratio
        if src_aspect > target_aspect:
            # Source is wider than target, crop width
            crop_h = src_h
            crop_w = int(src_h * target_aspect)
            crop_x = (src_w - crop_w) // 2
            crop_y = 0
        else:
            # Source is taller than target, crop height
            crop_w = src_w
            crop_h = int(src_w / target_aspect)
            crop_x = 0
            crop_y = (src_h - crop_h) // 2

        # Apply crop
        cropped = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

        # Resize to target dimensions
        resized = cv2.resize(
            cropped, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LANCZOS4
        )

        return resized

    def optimize_image(
        self,
        input_image: np.ndarray,
        optimization_method: str = "least_squares",
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Optimize LED values for input image.

        Args:
            input_image: Target image (FRAME_HEIGHT, FRAME_WIDTH, 3)
            optimization_method: Optimization method to use
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold

        Returns:
            Tuple of (led_values, optimization_stats) or None if failed
        """
        if not self.optimizer:
            logger.error("Optimizer not initialized")
            return None

        try:
            logger.info("Starting LED optimization...")
            start_time = time.time()

            # Convert image to the format expected by optimizer
            # Optimizer expects (height, width, channels)
            target_image = input_image.copy()

            # Run optimization
            result = self.optimizer.optimize_frame(
                target_frame=target_image, max_iterations=max_iterations
            )

            optimization_time = time.time() - start_time

            if not result.success:
                logger.error(f"Optimization failed: {result.error}")
                return None

            # Extract LED values and stats
            led_values = result.led_values  # Shape: (LED_COUNT, 3)

            stats = {
                "optimization_time": optimization_time,
                "iterations": result.iterations,
                "final_error": result.final_error,
                "convergence_achieved": result.convergence_achieved,
                "method": optimization_method,
                "led_count": LED_COUNT,
                "max_led_value": float(np.max(led_values)),
                "mean_led_value": float(np.mean(led_values)),
                "active_leds": int(np.sum(np.max(led_values, axis=1) > 0.01)),
            }

            logger.info(f"Optimization completed in {optimization_time:.2f}s")
            logger.info(f"Final error: {result.final_error:.6f}")
            logger.info(f"Iterations: {result.iterations}")
            logger.info(f"Active LEDs: {stats['active_leds']}/{LED_COUNT}")

            return led_values, stats

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None

    def render_result(self, led_values: np.ndarray) -> Optional[np.ndarray]:
        """
        Render the result by summing diffusion patterns weighted by LED values.

        Args:
            led_values: LED values array (LED_COUNT, 3)

        Returns:
            Rendered image (FRAME_HEIGHT, FRAME_WIDTH, 3) or None if failed
        """
        if self.diffusion_patterns is None:
            logger.error("No diffusion patterns available")
            return None

        try:
            logger.info("Rendering optimization result...")

            # Initialize result image
            result_image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)

            # Sum weighted diffusion patterns
            for led_idx in range(LED_COUNT):
                for channel in range(3):
                    led_intensity = led_values[led_idx, channel]
                    if led_intensity > 0:
                        # Convert uint8 pattern to float32 for accumulation
                        pattern = (
                            self.diffusion_patterns[led_idx, channel].astype(np.float32)
                            / 255.0
                        )
                        result_image[:, :, channel] += pattern * led_intensity

            # Clip to valid range (0-1 for float32 output)
            result_image = np.clip(result_image, 0, 1)

            logger.info("Result rendering completed")
            return result_image

        except Exception as e:
            logger.error(f"Result rendering failed: {e}")
            return None

    def save_result(
        self,
        result_image: np.ndarray,
        output_path: str,
        led_values: Optional[np.ndarray] = None,
        stats: Optional[dict] = None,
    ) -> bool:
        """
        Save optimization result.

        Args:
            result_image: Rendered result image
            output_path: Output file path
            led_values: Optional LED values to save
            stats: Optional optimization statistics

        Returns:
            True if save successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to 8-bit
            result_8bit = (np.clip(result_image, 0, 1) * 255).astype(np.uint8)

            # Save main result image
            if PIL_AVAILABLE:
                img = Image.fromarray(result_8bit)
                img.save(str(output_path))
            else:
                result_bgr = cv2.cvtColor(result_8bit, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), result_bgr)

            logger.info(f"Result saved to {output_path}")

            # Save additional data if requested
            if led_values is not None or stats is not None:
                data_path = output_path.with_suffix(".npz")
                save_data = {"result_image": result_image}

                if led_values is not None:
                    save_data["led_values"] = led_values

                if stats is not None:
                    save_data["optimization_stats"] = stats

                np.savez_compressed(str(data_path), **save_data)
                logger.info(f"Additional data saved to {data_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            return False

    def create_comparison_image(
        self, original: np.ndarray, result: np.ndarray
    ) -> np.ndarray:
        """
        Create side-by-side comparison image.

        Args:
            original: Original input image
            result: Optimization result

        Returns:
            Comparison image
        """
        try:
            # Ensure both images are the same size
            if original.shape != result.shape:
                original = cv2.resize(original, (result.shape[1], result.shape[0]))

            # Create side-by-side comparison
            comparison = np.hstack([original, result])

            # Add labels
            comparison = comparison.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (1.0, 1.0, 1.0)  # White
            thickness = 2

            # Original label
            cv2.putText(
                comparison, "Original", (10, 30), font, font_scale, color, thickness
            )

            # Result label
            cv2.putText(
                comparison,
                "Optimized",
                (original.shape[1] + 10, 30),
                font,
                font_scale,
                color,
                thickness,
            )

            return comparison

        except Exception as e:
            logger.warning(f"Failed to create comparison image: {e}")
            return result

    def show_preview(
        self,
        original: np.ndarray,
        result: np.ndarray,
        led_values: np.ndarray,
        stats: dict,
    ):
        """
        Show live preview of optimization result.

        Args:
            original: Original input image
            result: Optimization result
            led_values: LED values
            stats: Optimization statistics
        """
        try:
            # Create comparison image
            comparison = self.create_comparison_image(original, result)

            # Convert to 8-bit for display
            display_image = (np.clip(comparison, 0, 1) * 255).astype(np.uint8)
            display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

            # Add statistics overlay
            self._add_stats_overlay(display_image, led_values, stats)

            # Resize for display if too large
            max_display_width = 1200
            if display_image.shape[1] > max_display_width:
                scale = max_display_width / display_image.shape[1]
                new_width = max_display_width
                new_height = int(display_image.shape[0] * scale)
                display_image = cv2.resize(display_image, (new_width, new_height))

            # Show image
            cv2.imshow("LED Optimization Result", display_image)

            print("\nOptimization Results:")
            print(f"  Optimization Time: {stats['optimization_time']:.2f}s")
            print(f"  Final Error: {stats['final_error']:.6f}")
            print(f"  Iterations: {stats['iterations']}")
            print(f"  Active LEDs: {stats['active_leds']}/{LED_COUNT}")
            print(f"  Max LED Value: {stats['max_led_value']:.3f}")
            print("\nPress any key to continue...")

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            logger.warning(f"Preview display failed: {e}")

    def _add_stats_overlay(
        self, image: np.ndarray, led_values: np.ndarray, stats: dict
    ):
        """Add statistics overlay to image."""
        try:
            height, width = image.shape[:2]

            # Statistics text
            stat_lines = [
                f"Time: {stats['optimization_time']:.1f}s",
                f"Error: {stats['final_error']:.2e}",
                f"Iterations: {stats['iterations']}",
                f"Active LEDs: {stats['active_leds']}/{LED_COUNT}",
                f"Max Value: {stats['max_led_value']:.2f}",
            ]

            # Add semi-transparent background
            overlay = image.copy()
            cv2.rectangle(overlay, (width - 200, 50), (width - 10, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)
            thickness = 1

            for i, line in enumerate(stat_lines):
                y = 70 + i * 25
                cv2.putText(
                    image, line, (width - 195, y), font, font_scale, color, thickness
                )

        except Exception as e:
            logger.warning(f"Failed to add stats overlay: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Standalone LED optimization tool")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--patterns", help="Diffusion patterns file (.npz)")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic patterns if no patterns file",
    )
    parser.add_argument(
        "--method",
        default="least_squares",
        choices=["least_squares", "gradient_descent"],
        help="Optimization method",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum optimization iterations",
    )
    parser.add_argument(
        "--preview", action="store_true", help="Show preview before saving"
    )
    parser.add_argument(
        "--save-data", action="store_true", help="Save LED values and stats"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate inputs
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    if not args.patterns and not args.synthetic:
        logger.error("Must provide --patterns file or use --synthetic")
        return 1

    # Create optimizer
    optimizer = StandaloneOptimizer(
        diffusion_patterns_path=args.patterns, use_synthetic=args.synthetic
    )

    try:
        # Initialize
        if not optimizer.initialize():
            logger.error("Failed to initialize optimizer")
            return 1

        # Load input image
        input_image = optimizer.load_input_image(args.input)
        if input_image is None:
            logger.error("Failed to load input image")
            return 1

        # Optimize
        result = optimizer.optimize_image(
            input_image,
            optimization_method=args.method,
            max_iterations=args.max_iterations,
        )

        if result is None:
            logger.error("Optimization failed")
            return 1

        led_values, stats = result

        # Render result
        result_image = optimizer.render_result(led_values)
        if result_image is None:
            logger.error("Failed to render result")
            return 1

        # Show preview if requested
        if args.preview:
            optimizer.show_preview(input_image, result_image, led_values, stats)

        # Save result
        save_success = optimizer.save_result(
            result_image,
            args.output,
            led_values if args.save_data else None,
            stats if args.save_data else None,
        )

        if not save_success:
            logger.error("Failed to save result")
            return 1

        logger.info("Optimization completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
