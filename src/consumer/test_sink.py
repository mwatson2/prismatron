"""
Test Sink for LED Display System.

This module provides a test sink that uses the mixed sparse tensor to calculate
Ax (where x is the LED values) and displays the rendered result in a window for
testing and debugging the LED optimization system.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cupy as cp
import cv2
import numpy as np

from ..const import FRAME_HEIGHT, FRAME_WIDTH
from ..utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


@dataclass
class TestSinkConfig:
    """Configuration for the test sink."""

    window_name: str = "Prismatron Test Sink"
    display_scale: float = 1.0  # Scale factor for display window
    update_rate: float = 30.0  # Target update rate in Hz
    auto_brightness: bool = True  # Auto-adjust brightness for better visibility
    show_stats: bool = True  # Show rendering statistics on display
    max_brightness: float = 1.0  # Maximum brightness scaling
    gamma_correction: float = 2.2  # Gamma correction for display


class TestSink:
    """
    Test sink that uses mixed sparse tensor to render LED values to a display window.

    This sink takes LED values, uses the mixed sparse tensor to calculate Ax
    (forward pass through the diffusion matrix), and displays the result in a window
    for testing and debugging purposes.
    """

    def __init__(
        self,
        mixed_tensor: SingleBlockMixedSparseTensor,
        config: Optional[TestSinkConfig] = None,
    ):
        """
        Initialize the test sink.

        Args:
            mixed_tensor: Mixed sparse tensor for forward pass rendering
            config: Sink configuration (uses defaults if None)
        """
        self.mixed_tensor = mixed_tensor
        self.config = config or TestSinkConfig()

        # Display state
        self.is_running = False
        self.display_thread: Optional[threading.Thread] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # Statistics
        self.frames_rendered = 0
        self.total_render_time = 0.0
        self.last_render_time = 0.0
        self.current_fps = 0.0

        # OpenCV window setup
        self.window_created = False
        self.display_size = (
            int(FRAME_WIDTH * self.config.display_scale),
            int(FRAME_HEIGHT * self.config.display_scale),
        )

        logger.info(f"Test sink initialized with display size {self.display_size}")

    def start(self) -> bool:
        """
        Start the test sink display window.

        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.is_running:
                logger.warning("Test sink already running")
                return True

            # Create display window
            self._create_window()

            # Start display thread
            self.is_running = True
            self.display_thread = threading.Thread(target=self._display_loop, name="TestSink", daemon=True)
            self.display_thread.start()

            logger.info("Test sink started")
            return True

        except Exception as e:
            logger.error(f"Failed to start test sink: {e}")
            return False

    def stop(self) -> None:
        """Stop the test sink."""
        try:
            if not self.is_running:
                return

            logger.info("Stopping test sink...")
            self.is_running = False

            # Wait for display thread to finish
            if self.display_thread and self.display_thread.is_alive():
                self.display_thread.join(timeout=2.0)
                if self.display_thread.is_alive():
                    logger.warning("Display thread did not stop cleanly")

            # Close OpenCV window
            if self.window_created:
                cv2.destroyWindow(self.config.window_name)
                self.window_created = False

            logger.info("Test sink stopped")

        except Exception as e:
            logger.error(f"Error stopping test sink: {e}")

    def render_led_values(self, led_values: np.ndarray) -> bool:
        """
        Render LED values to the display using the mixed sparse tensor.

        Args:
            led_values: LED values array (led_count, 3) in range [0, 255]

        Returns:
            True if rendered successfully, False otherwise
        """
        try:
            start_time = time.time()

            # Validate input
            if led_values.ndim != 2 or led_values.shape[1] != 3:
                logger.error(f"Invalid LED values shape: {led_values.shape}, expected (led_count, 3)")
                return False

            # Convert to float32 [0, 1] range for tensor operations
            led_values_normalized = led_values.astype(np.float32) / 255.0

            # Forward pass through mixed sparse tensor: A @ x -> rendered_frame
            rendered_frame = self._forward_pass(led_values_normalized)

            # Apply display processing
            display_frame = self._process_for_display(rendered_frame)

            # Update current frame for display thread
            with self.frame_lock:
                self.current_frame = display_frame

            # Update statistics
            render_time = time.time() - start_time
            self.frames_rendered += 1
            self.total_render_time += render_time
            self.last_render_time = start_time

            # Calculate FPS
            if render_time > 0:
                self.current_fps = 1.0 / render_time

            logger.debug(f"Rendered frame in {render_time * 1000:.1f}ms")
            return True

        except Exception as e:
            logger.error(f"Error rendering LED values: {e}")
            return False

    def _forward_pass(self, led_values: np.ndarray) -> np.ndarray:
        """
        Perform forward pass through mixed sparse tensor.

        Args:
            led_values: Normalized LED values (led_count, 3) in range [0, 1]

        Returns:
            Rendered frame (3, height, width) in range [0, 1]
        """
        # Transfer to GPU
        led_values_gpu = cp.asarray(led_values)

        # Use mixed tensor forward pass (A @ x)
        # Mixed tensor expects (led_count, 3) input and returns (3, height, width)
        rendered_gpu = self.mixed_tensor.forward_pass_3d(led_values_gpu)

        # Transfer back to CPU
        rendered_frame = cp.asnumpy(rendered_gpu)

        return rendered_frame

    def _process_for_display(self, rendered_frame: np.ndarray) -> np.ndarray:
        """
        Process rendered frame for display.

        Args:
            rendered_frame: Rendered frame (3, height, width) in range [0, 1]

        Returns:
            Display-ready frame (height, width, 3) in range [0, 255]
        """
        # Convert from planar (3, H, W) to interleaved (H, W, 3)
        display_frame = rendered_frame.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)

        # Auto-brightness adjustment
        if self.config.auto_brightness:
            max_val = np.max(display_frame)
            if max_val > 0:
                brightness_scale = min(self.config.max_brightness / max_val, 1.0)
                display_frame = display_frame * brightness_scale

        # Gamma correction
        if self.config.gamma_correction != 1.0:
            display_frame = np.power(display_frame, 1.0 / self.config.gamma_correction)

        # Convert to uint8 [0, 255]
        display_frame = np.clip(display_frame * 255.0, 0, 255).astype(np.uint8)

        # Scale for display if needed
        if self.config.display_scale != 1.0:
            display_frame = cv2.resize(display_frame, self.display_size, interpolation=cv2.INTER_NEAREST)

        # Add statistics overlay if enabled
        if self.config.show_stats:
            display_frame = self._add_stats_overlay(display_frame)

        return display_frame

    def _add_stats_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Add statistics overlay to the display frame.

        Args:
            frame: Display frame (height, width, 3)

        Returns:
            Frame with statistics overlay
        """
        # Create a copy to avoid modifying the original
        overlay_frame = frame.copy()

        # Prepare statistics text
        avg_render_time = self.total_render_time / self.frames_rendered if self.frames_rendered > 0 else 0
        avg_fps = self.frames_rendered / self.total_render_time if self.total_render_time > 0 else 0

        stats_text = [
            f"FPS: {self.current_fps:.1f} (avg: {avg_fps:.1f})",
            f"Frames: {self.frames_rendered}",
            f"Render: {avg_render_time * 1000:.1f}ms",
            f"LEDs: {self.mixed_tensor.led_count}",
            f"Size: {FRAME_WIDTH}x{FRAME_HEIGHT}",
        ]

        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255)  # White text
        thickness = 1

        y_offset = 15
        for i, text in enumerate(stats_text):
            y_pos = y_offset + i * 15
            cv2.putText(overlay_frame, text, (10, y_pos), font, font_scale, color, thickness)

        return overlay_frame

    def _create_window(self) -> None:
        """Create OpenCV display window."""
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.config.window_name, self.display_size[0], self.display_size[1])
        self.window_created = True
        logger.info(f"Created display window: {self.config.window_name}")

    def _display_loop(self) -> None:
        """Main display loop running in separate thread."""
        logger.info("Test sink display loop started")

        frame_time = 1.0 / self.config.update_rate
        last_display_time = time.time()

        # Create a default frame for when no data is available
        default_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        if self.config.display_scale != 1.0:
            default_frame = cv2.resize(default_frame, self.display_size, interpolation=cv2.INTER_NEAREST)

        while self.is_running:
            try:
                current_time = time.time()

                # Get current frame
                with self.frame_lock:
                    display_frame = self.current_frame if self.current_frame is not None else default_frame

                # Update display
                if self.window_created:
                    cv2.imshow(self.config.window_name, display_frame)

                    # Handle window events
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # 'q' key or ESC
                        logger.info("User requested shutdown via window")
                        self.is_running = False
                        break

                # Frame rate limiting
                elapsed = current_time - last_display_time
                remaining = frame_time - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                last_display_time = time.time()

            except Exception as e:
                logger.error(f"Error in display loop: {e}")
                time.sleep(0.1)  # Brief pause before continuing

        logger.info("Test sink display loop ended")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get sink statistics.

        Returns:
            Dictionary with sink statistics
        """
        avg_render_time = self.total_render_time / self.frames_rendered if self.frames_rendered > 0 else 0
        avg_fps = self.frames_rendered / self.total_render_time if self.total_render_time > 0 else 0

        return {
            "is_running": self.is_running,
            "frames_rendered": self.frames_rendered,
            "total_render_time": self.total_render_time,
            "average_render_time": avg_render_time,
            "average_fps": avg_fps,
            "current_fps": self.current_fps,
            "display_size": self.display_size,
            "led_count": self.mixed_tensor.led_count,
            "frame_size": (FRAME_WIDTH, FRAME_HEIGHT),
            "config": {
                "window_name": self.config.window_name,
                "display_scale": self.config.display_scale,
                "update_rate": self.config.update_rate,
                "auto_brightness": self.config.auto_brightness,
                "show_stats": self.config.show_stats,
                "max_brightness": self.config.max_brightness,
                "gamma_correction": self.config.gamma_correction,
            },
        }

    def reset_statistics(self) -> None:
        """Reset sink statistics."""
        self.frames_rendered = 0
        self.total_render_time = 0.0
        self.current_fps = 0.0
        logger.info("Test sink statistics reset")

    def update_config(self, **kwargs) -> None:
        """
        Update sink configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

    def __enter__(self):
        """Context manager entry."""
        if self.start():
            return self
        else:
            raise RuntimeError("Failed to start test sink")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_test_sink_from_pattern(
    pattern_file_path: str, config: Optional[TestSinkConfig] = None
) -> Optional[TestSink]:
    """
    Create a test sink from a diffusion pattern file.

    Args:
        pattern_file_path: Path to the .npz pattern file
        config: Sink configuration (uses defaults if None)

    Returns:
        TestSink instance or None if failed to load
    """
    try:
        # Load pattern data
        data = np.load(pattern_file_path, allow_pickle=True)

        # Check for mixed tensor
        if "mixed_tensor" not in data:
            logger.error(f"No mixed tensor found in {pattern_file_path}")
            return None

        # Load mixed tensor
        mixed_dict = data["mixed_tensor"].item()
        mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_dict)

        # Create sink
        sink = TestSink(mixed_tensor, config)

        logger.info(f"Created test sink from pattern: {pattern_file_path}")
        return sink

    except Exception as e:
        logger.error(f"Failed to create test sink from pattern: {e}")
        return None


# Example usage and testing functions


def test_sink_with_solid_color(sink: TestSink, r: int, g: int, b: int) -> bool:
    """
    Test the sink with a solid color pattern.

    Args:
        sink: Test sink instance
        r, g, b: RGB color values (0-255)

    Returns:
        True if test successful, False otherwise
    """
    try:
        # Create solid color LED data
        led_count = sink.mixed_tensor.led_count
        led_values = np.full((led_count, 3), [r, g, b], dtype=np.uint8)

        # Render
        return sink.render_led_values(led_values)

    except Exception as e:
        logger.error(f"Solid color test failed: {e}")
        return False


def test_sink_with_gradient(sink: TestSink) -> bool:
    """
    Test the sink with a gradient pattern.

    Args:
        sink: Test sink instance

    Returns:
        True if test successful, False otherwise
    """
    try:
        # Create gradient LED data
        led_count = sink.mixed_tensor.led_count
        led_values = np.zeros((led_count, 3), dtype=np.uint8)

        # Create RGB gradient
        for i in range(led_count):
            factor = i / (led_count - 1) if led_count > 1 else 0
            led_values[i] = [
                int(255 * factor),  # Red gradient
                int(255 * (1 - factor)),  # Green inverse gradient
                128,  # Blue constant
            ]

        # Render
        return sink.render_led_values(led_values)

    except Exception as e:
        logger.error(f"Gradient test failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    import argparse
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Test LED sink")
    parser.add_argument("pattern_file", help="Path to diffusion pattern file")
    parser.add_argument("--scale", type=float, default=1.0, help="Display scale factor")
    parser.add_argument("--no-stats", action="store_true", help="Disable statistics overlay")

    args = parser.parse_args()

    # Create config
    config = TestSinkConfig(display_scale=args.scale, show_stats=not args.no_stats)

    # Create sink
    sink = create_test_sink_from_pattern(args.pattern_file, config)
    if not sink:
        logger.error("Failed to create sink")
        sys.exit(1)

    try:
        # Start sink
        if not sink.start():
            logger.error("Failed to start sink")
            sys.exit(1)

        logger.info("Test sink started. Press 'q' or ESC in window to exit.")

        # Test with different patterns
        test_patterns = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 255),  # White
            (128, 128, 128),  # Gray
        ]

        for r, g, b in test_patterns:
            logger.info(f"Testing with color ({r}, {g}, {b})")
            test_sink_with_solid_color(sink, r, g, b)
            time.sleep(2)  # Show each color for 2 seconds

            if not sink.is_running:
                break

        # Test gradient
        if sink.is_running:
            logger.info("Testing with gradient")
            test_sink_with_gradient(sink)
            time.sleep(3)

        # Keep running until user closes window
        while sink.is_running:
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        sink.stop()
        logger.info("Test sink demo completed")
# Test change
# Test formatting
