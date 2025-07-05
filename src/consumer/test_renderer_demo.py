#!/usr/bin/env python3
"""
Test Renderer Demo Script.

This script demonstrates the test renderer functionality by loading a diffusion pattern,
creating test LED values, and displaying the rendered result in a window.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add the parent directory to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from consumer.test_renderer import (
    TestRenderer,
    TestRendererConfig,
    create_test_renderer_from_pattern,
    test_renderer_with_gradient,
    test_renderer_with_solid_color,
)

logger = logging.getLogger(__name__)


def create_test_patterns(led_count: int) -> list:
    """
    Create various test LED patterns.

    Args:
        led_count: Number of LEDs

    Returns:
        List of (name, led_values) tuples
    """
    patterns = []

    # 1. Solid colors
    solid_colors = [
        ("Red", [255, 0, 0]),
        ("Green", [0, 255, 0]),
        ("Blue", [0, 0, 255]),
        ("White", [255, 255, 255]),
        ("Cyan", [0, 255, 255]),
        ("Magenta", [255, 0, 255]),
        ("Yellow", [255, 255, 0]),
    ]

    for name, color in solid_colors:
        led_values = np.full((led_count, 3), color, dtype=np.uint8)
        patterns.append((f"Solid_{name}", led_values))

    # 2. Gradients
    # Red gradient
    red_gradient = np.zeros((led_count, 3), dtype=np.uint8)
    for i in range(led_count):
        intensity = int(255 * (i / (led_count - 1)) if led_count > 1 else 255)
        red_gradient[i] = [intensity, 0, 0]
    patterns.append(("Red_Gradient", red_gradient))

    # RGB gradient
    rgb_gradient = np.zeros((led_count, 3), dtype=np.uint8)
    for i in range(led_count):
        factor = i / (led_count - 1) if led_count > 1 else 0
        rgb_gradient[i] = [
            int(255 * factor),  # Red increases
            int(255 * (1 - factor)),  # Green decreases
            int(255 * abs(0.5 - factor) * 2),  # Blue triangle wave
        ]
    patterns.append(("RGB_Gradient", rgb_gradient))

    # 3. Rainbow pattern
    rainbow = np.zeros((led_count, 3), dtype=np.uint8)
    for i in range(led_count):
        hue = (i / led_count) * 360 if led_count > 1 else 0
        # Simple HSV to RGB conversion
        c = 255
        x = int(c * (1 - abs((hue / 60) % 2 - 1)))

        if 0 <= hue < 60:
            rainbow[i] = [c, x, 0]
        elif 60 <= hue < 120:
            rainbow[i] = [x, c, 0]
        elif 120 <= hue < 180:
            rainbow[i] = [0, c, x]
        elif 180 <= hue < 240:
            rainbow[i] = [0, x, c]
        elif 240 <= hue < 300:
            rainbow[i] = [x, 0, c]
        else:
            rainbow[i] = [c, 0, x]
    patterns.append(("Rainbow", rainbow))

    # 4. Alternating pattern
    alternating = np.zeros((led_count, 3), dtype=np.uint8)
    for i in range(led_count):
        if i % 3 == 0:
            alternating[i] = [255, 0, 0]  # Red
        elif i % 3 == 1:
            alternating[i] = [0, 255, 0]  # Green
        else:
            alternating[i] = [0, 0, 255]  # Blue
    patterns.append(("Alternating_RGB", alternating))

    # 5. Checkerboard pattern (if enough LEDs)
    if led_count >= 4:
        checkerboard = np.zeros((led_count, 3), dtype=np.uint8)
        for i in range(led_count):
            if (i // 2) % 2 == 0:
                checkerboard[i] = [255, 255, 255]  # White
            else:
                checkerboard[i] = [0, 0, 0]  # Black
        patterns.append(("Checkerboard", checkerboard))

    # 6. Random pattern
    random_pattern = np.random.randint(0, 256, (led_count, 3), dtype=np.uint8)
    patterns.append(("Random", random_pattern))

    return patterns


def run_demo(
    pattern_file: str,
    display_time: float = 3.0,
    scale: float = 1.0,
    no_stats: bool = False,
):
    """
    Run the test renderer demo.

    Args:
        pattern_file: Path to diffusion pattern file
        display_time: Time to display each pattern (seconds)
        scale: Display scale factor
        no_stats: Disable statistics overlay
    """
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger.info(f"Starting test renderer demo with pattern: {pattern_file}")

        # Check if pattern file exists
        if not os.path.exists(pattern_file):
            logger.error(f"Pattern file not found: {pattern_file}")
            return False

        # Create renderer configuration
        config = TestRendererConfig(
            display_scale=scale,
            show_stats=not no_stats,
            window_name="Prismatron Test Renderer Demo",
        )

        # Create renderer from pattern file
        renderer = create_test_renderer_from_pattern(pattern_file, config)
        if not renderer:
            logger.error("Failed to create test renderer")
            return False

        led_count = renderer.mixed_tensor.led_count
        logger.info(f"Test renderer created for {led_count} LEDs")

        # Start renderer
        if not renderer.start():
            logger.error("Failed to start test renderer")
            return False

        logger.info("Test renderer started successfully")
        logger.info(f"Display window: '{config.window_name}'")
        logger.info("Press 'q' or ESC in the window to exit early")
        logger.info(f"Each pattern will be displayed for {display_time} seconds")

        # Create test patterns
        patterns = create_test_patterns(led_count)
        logger.info(f"Created {len(patterns)} test patterns")

        # Display each pattern
        for i, (pattern_name, led_values) in enumerate(patterns):
            if not renderer.is_running:
                logger.info("Renderer stopped by user")
                break

            logger.info(f"Pattern {i + 1}/{len(patterns)}: {pattern_name}")

            # Render the pattern
            success = renderer.render_led_values(led_values)
            if not success:
                logger.warning(f"Failed to render pattern: {pattern_name}")
                continue

            # Wait for display time
            start_time = time.time()
            while time.time() - start_time < display_time:
                if not renderer.is_running:
                    break
                time.sleep(0.1)

        # Show final statistics
        stats = renderer.get_statistics()
        logger.info("=== Final Statistics ===")
        logger.info(f"Frames rendered: {stats['frames_rendered']}")
        logger.info(f"Average FPS: {stats['average_fps']:.1f}")
        logger.info(f"Average render time: {stats['average_render_time'] * 1000:.1f}ms")
        logger.info(f"Total render time: {stats['total_render_time']:.2f}s")

        # Wait a bit for user to see final stats
        if renderer.is_running:
            logger.info("Demo completed. Window will close in 5 seconds or press 'q' to exit now.")
            for i in range(50):  # 5 seconds in 0.1s increments
                if not renderer.is_running:
                    break
                time.sleep(0.1)

        # Stop renderer
        renderer.stop()
        logger.info("Test renderer demo completed successfully")
        return True

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        if "renderer" in locals():
            renderer.stop()
        return True
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        if "renderer" in locals():
            renderer.stop()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test LED renderer demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("pattern_file", help="Path to diffusion pattern file (.npz)")

    parser.add_argument(
        "--time",
        "-t",
        type=float,
        default=3.0,
        help="Time to display each pattern (seconds)",
    )

    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Display scale factor")

    parser.add_argument("--no-stats", action="store_true", help="Disable statistics overlay")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run demo
    success = run_demo(
        pattern_file=args.pattern_file,
        display_time=args.time,
        scale=args.scale,
        no_stats=args.no_stats,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
