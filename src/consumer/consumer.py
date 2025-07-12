"""
Consumer Process Implementation.

This module implements the main consumer process that reads frames from
shared memory, optimizes LED values, and transmits to WLED controller.
"""

import logging
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from ..core import ControlState, FrameConsumer
from .led_optimizer_dense import LEDOptimizer
from .test_renderer import TestRenderer, TestRendererConfig
from .wled_client import WLEDClient, WLEDConfig

logger = logging.getLogger(__name__)


@dataclass
class ConsumerStats:
    """Consumer process statistics."""

    frames_processed: int = 0
    total_processing_time: float = 0.0
    total_optimization_time: float = 0.0
    total_transmission_time: float = 0.0
    optimization_errors: int = 0
    transmission_errors: int = 0
    last_frame_time: float = 0.0
    current_fps: float = 0.0

    def get_average_fps(self) -> float:
        """Get average FPS since start."""
        if self.total_processing_time > 0:
            return self.frames_processed / self.total_processing_time
        return 0.0

    def get_average_optimization_time(self) -> float:
        """Get average optimization time per frame."""
        if self.frames_processed > 0:
            return self.total_optimization_time / self.frames_processed
        return 0.0

    def get_average_transmission_time(self) -> float:
        """Get average transmission time per frame."""
        if self.frames_processed > 0:
            return self.total_transmission_time / self.frames_processed
        return 0.0


class ConsumerProcess:
    """
    Main consumer process for LED optimization and transmission.

    Reads frames from shared memory buffer, optimizes LED values using
    diffusion patterns, and transmits to WLED controller via UDP.
    """

    def __init__(
        self,
        buffer_name: str = "prismatron_buffer",
        control_name: str = "prismatron_control",
        wled_host: str = "192.168.1.100",
        wled_port: int = 4048,
        diffusion_patterns_path: Optional[str] = None,
        enable_test_renderer: bool = False,
        test_renderer_config: Optional[TestRendererConfig] = None,
    ):
        """
        Initialize consumer process.

        Args:
            buffer_name: Shared memory buffer name
            control_name: Control state name
            wled_host: WLED controller IP address
            wled_port: WLED controller port
            diffusion_patterns_path: Path to diffusion patterns
            enable_test_renderer: Enable test renderer for debugging
            test_renderer_config: Test renderer configuration
        """
        self.buffer_name = buffer_name
        self.control_name = control_name

        # Initialize components
        self._frame_consumer = FrameConsumer(buffer_name)
        self._control_state = ControlState(control_name)
        self._led_optimizer = LEDOptimizer(
            diffusion_patterns_path=diffusion_patterns_path,
        )
        # Configure WLED client
        wled_config = WLEDConfig(
            host=wled_host,
            port=wled_port,
            led_count=LED_COUNT,
        )
        self._wled_client = WLEDClient(wled_config)

        # Test renderer (optional)
        self.enable_test_renderer = enable_test_renderer
        self._test_renderer: Optional[TestRenderer] = None
        self._test_renderer_config = test_renderer_config or TestRendererConfig()

        # Process state
        self._running = False
        self._shutdown_requested = False
        self._stats = ConsumerStats()
        self._process_thread: Optional[threading.Thread] = None

        # Performance settings
        self.target_fps = 15.0
        self.max_frame_wait_timeout = 1.0
        self.use_optimization = True
        self.brightness_scale = 1.0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def initialize(self) -> bool:
        """
        Initialize all consumer components.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing consumer process...")

            # Initialize LED optimizer
            if not self._led_optimizer.initialize():
                logger.error("Failed to initialize LED optimizer")
                return False

            # Connect to WLED controller
            if not self._wled_client.connect():
                logger.error("Failed to connect to WLED controller")
                return False

            # Initialize test renderer if enabled
            if self.enable_test_renderer:
                self._initialize_test_renderer()

            logger.info("Consumer process initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Consumer initialization failed: {e}")
            return False

    def start(self) -> bool:
        """
        Start the consumer process.

        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self._running:
                logger.warning("Consumer process already running")
                return True

            if not self.initialize():
                return False

            # Start processing thread
            self._running = True
            self._shutdown_requested = False
            self._process_thread = threading.Thread(target=self._process_loop, name="ConsumerProcess")
            self._process_thread.start()

            logger.info("Consumer process started")
            return True

        except Exception as e:
            logger.error(f"Failed to start consumer process: {e}")
            return False

    def stop(self) -> None:
        """Stop the consumer process gracefully."""
        try:
            logger.info("Stopping consumer process...")

            # Signal shutdown
            self._shutdown_requested = True
            self._running = False

            # Wait for processing thread to finish
            if self._process_thread and self._process_thread.is_alive():
                self._process_thread.join(timeout=5.0)
                if self._process_thread.is_alive():
                    logger.warning("Process thread did not stop gracefully")

            # Cleanup components
            self._cleanup()

            logger.info("Consumer process stopped")

        except Exception as e:
            logger.error(f"Error stopping consumer process: {e}")

    def _process_loop(self) -> None:
        """Main processing loop."""
        logger.info("Consumer processing loop started")
        target_frame_time = 1.0 / self.target_fps

        while self._running and not self._shutdown_requested:
            try:
                loop_start_time = time.time()

                # Check for shutdown signal from control state
                if self._control_state.should_shutdown():
                    logger.info("Shutdown signal received from control state")
                    break

                # Wait for new frame
                buffer_info = self._frame_consumer.wait_for_ready_buffer(timeout=self.max_frame_wait_timeout)

                if buffer_info is None:
                    # Timeout waiting for frame
                    continue

                # Process the frame
                self._process_frame(buffer_info)

                # Update FPS tracking
                loop_time = time.time() - loop_start_time
                self._stats.current_fps = 1.0 / loop_time if loop_time > 0 else 0.0

                # Frame rate limiting
                remaining_time = target_frame_time - loop_time
                if remaining_time > 0:
                    time.sleep(remaining_time)

            except Exception as e:
                logger.error(f"Error in consumer processing loop: {e}")
                time.sleep(0.1)  # Brief pause before retrying

        logger.info("Consumer processing loop ended")

    def _process_frame(self, buffer_info) -> None:
        """
        Process a single frame from shared memory.

        Args:
            buffer_info: Buffer information from frame consumer
        """
        start_time = time.time()

        try:
            # Get frame array from buffer
            frame_array = buffer_info.get_array()

            # Validate frame shape
            if frame_array.shape != (FRAME_HEIGHT, FRAME_WIDTH, 4):
                logger.warning(f"Unexpected frame shape: {frame_array.shape}")
                return

            # Convert RGBA to RGB
            rgb_frame = frame_array[:, :, :3].astype(np.uint8)

            # Apply brightness scaling
            if self.brightness_scale != 1.0:
                rgb_frame = (rgb_frame * self.brightness_scale).clip(0, 255).astype(np.uint8)

            # Optimize LED values
            optimization_start = time.time()

            # Use diffusion pattern optimization
            max_iters = 50 if self.use_optimization else 5  # Fewer iterations for speed mode
            result = self._led_optimizer.optimize_frame(rgb_frame, max_iterations=max_iters)

            optimization_time = time.time() - optimization_start

            # Check optimization result
            if not result.converged:
                logger.warning(f"Optimization did not converge after {result.iterations} iterations")
                self._stats.optimization_errors += 1

            # Send to test renderer if enabled
            if self._test_renderer and self._test_renderer.is_running:
                try:
                    led_values_uint8 = result.led_values.astype(np.uint8)
                    self._test_renderer.render_led_values(led_values_uint8)
                except Exception as e:
                    logger.warning(f"Test renderer error: {e}")

            # Transmit to WLED
            transmission_start = time.time()
            led_values = result.led_values.astype(np.uint8)

            transmission_result = self._wled_client.send_led_data(led_values)
            transmission_time = time.time() - transmission_start

            if not transmission_result.success:
                logger.warning(f"WLED transmission failed: {transmission_result.errors}")
                self._stats.transmission_errors += 1

            # Update statistics
            total_time = time.time() - start_time
            self._stats.frames_processed += 1
            self._stats.total_processing_time += total_time
            self._stats.total_optimization_time += optimization_time
            self._stats.total_transmission_time += transmission_time
            self._stats.last_frame_time = start_time

            # Log performance periodically
            if self._stats.frames_processed % 100 == 0:
                avg_fps = self._stats.get_average_fps()
                avg_opt_time = self._stats.get_average_optimization_time()
                avg_tx_time = self._stats.get_average_transmission_time()

                logger.info(
                    f"Performance: {avg_fps:.1f} fps avg, "
                    f"opt: {avg_opt_time * 1000:.1f}ms, "
                    f"tx: {avg_tx_time * 1000:.1f}ms, "
                    f"errors: opt={self._stats.optimization_errors}, "
                    f"tx={self._stats.transmission_errors}"
                )

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            self._stats.optimization_errors += 1

    def _initialize_test_renderer(self) -> bool:
        """
        Initialize test renderer using mixed tensor from LED optimizer.

        Returns:
            True if initialized successfully, False otherwise
        """
        try:
            # Get mixed tensor from LED optimizer
            if not hasattr(self._led_optimizer, "_mixed_tensor") or self._led_optimizer._mixed_tensor is None:
                logger.error("LED optimizer does not have mixed tensor - test renderer cannot be initialized")
                return False

            mixed_tensor = self._led_optimizer._mixed_tensor

            # Create test renderer
            self._test_renderer = TestRenderer(mixed_tensor, self._test_renderer_config)

            # Start test renderer
            if self._test_renderer.start():
                logger.info("Test renderer initialized and started successfully")
                return True
            else:
                logger.error("Failed to start test renderer")
                self._test_renderer = None
                return False

        except Exception as e:
            logger.error(f"Failed to initialize test renderer: {e}")
            self._test_renderer = None
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get consumer process statistics.

        Returns:
            Dictionary with process statistics
        """
        return {
            "running": self._running,
            "frames_processed": self._stats.frames_processed,
            "current_fps": self._stats.current_fps,
            "average_fps": self._stats.get_average_fps(),
            "total_processing_time": self._stats.total_processing_time,
            "average_optimization_time": self._stats.get_average_optimization_time(),
            "average_transmission_time": self._stats.get_average_transmission_time(),
            "optimization_errors": self._stats.optimization_errors,
            "transmission_errors": self._stats.transmission_errors,
            "target_fps": self.target_fps,
            "use_optimization": self.use_optimization,
            "brightness_scale": self.brightness_scale,
            "led_count": LED_COUNT,
            "frame_dimensions": (FRAME_WIDTH, FRAME_HEIGHT),
            "wled_stats": self._wled_client.get_statistics(),
            "optimizer_stats": self._led_optimizer.get_optimizer_stats(),
            "test_renderer_enabled": self.enable_test_renderer,
            "test_renderer_stats": (self._test_renderer.get_statistics() if self._test_renderer else None),
        }

    def set_performance_settings(
        self,
        target_fps: Optional[float] = None,
        use_optimization: Optional[bool] = None,
        brightness_scale: Optional[float] = None,
    ) -> None:
        """
        Update performance settings.

        Args:
            target_fps: Target frames per second
            use_optimization: Whether to use full optimization or simple sampling
            brightness_scale: Brightness scaling factor (0.0 to 1.0)
        """
        if target_fps is not None:
            self.target_fps = max(1.0, min(60.0, target_fps))

        if use_optimization is not None:
            self.use_optimization = use_optimization

        if brightness_scale is not None:
            self.brightness_scale = max(0.0, min(1.0, brightness_scale))

        logger.info(
            f"Updated performance settings: fps={self.target_fps}, "
            f"optimization={self.use_optimization}, "
            f"brightness={self.brightness_scale}"
        )

    def set_test_renderer_enabled(self, enabled: bool) -> bool:
        """
        Enable or disable test renderer.

        Args:
            enabled: Whether to enable test renderer

        Returns:
            True if operation successful, False otherwise
        """
        try:
            if enabled == self.enable_test_renderer:
                return True  # Already in desired state

            if enabled:
                # Enable test renderer
                if not self._led_optimizer._matrix_loaded:
                    logger.error("Cannot enable test renderer: LED optimizer not loaded")
                    return False

                self.enable_test_renderer = True
                return self._initialize_test_renderer()
            else:
                # Disable test renderer
                self.enable_test_renderer = False
                if self._test_renderer:
                    self._test_renderer.stop()
                    self._test_renderer = None

                logger.info("Test renderer disabled")
                return True

        except Exception as e:
            logger.error(f"Error setting test renderer enabled state: {e}")
            return False

    def get_test_renderer(self) -> Optional[TestRenderer]:
        """
        Get the test renderer instance.

        Returns:
            TestRenderer instance or None if not enabled
        """
        return self._test_renderer

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()

    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Cleanup components in reverse order
            if hasattr(self, "_test_renderer") and self._test_renderer:
                self._test_renderer.stop()

            if hasattr(self, "_wled_client"):
                self._wled_client.disconnect()

            if hasattr(self, "_frame_consumer"):
                self._frame_consumer.cleanup()

            logger.debug("Consumer process cleanup completed")

        except Exception as e:
            logger.error(f"Error during consumer cleanup: {e}")


def main():
    """Main entry point for running consumer process standalone."""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Create and start consumer process
        consumer = ConsumerProcess()

        if not consumer.start():
            logger.error("Failed to start consumer process")
            sys.exit(1)

        # Run until interrupted
        logger.info("Consumer process running. Press Ctrl+C to stop.")

        try:
            # Keep main thread alive
            while consumer._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        consumer.stop()
        logger.info("Consumer process completed")

    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
