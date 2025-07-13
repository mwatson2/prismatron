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
from .frame_renderer import FrameRenderer
from .led_buffer import LEDBuffer
from .led_optimizer import LEDOptimizer
from .test_renderer import TestRenderer, TestRendererConfig
from .wled_client import WLEDClient, WLEDConfig

logger = logging.getLogger(__name__)


@dataclass
class ConsumerStats:
    """Consumer process statistics."""

    frames_processed: int = 0
    total_processing_time: float = 0.0
    total_optimization_time: float = 0.0
    optimization_errors: int = 0
    last_frame_time: float = 0.0
    current_optimization_fps: float = 0.0

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

        # New timestamp-based rendering components
        self._led_buffer = LEDBuffer(buffer_size=10)
        self._frame_renderer = FrameRenderer(first_frame_delay_ms=100.0, timing_tolerance_ms=5.0)

        # Test renderer (optional)
        self.enable_test_renderer = enable_test_renderer
        self._test_renderer: Optional[TestRenderer] = None
        self._test_renderer_config = test_renderer_config or TestRendererConfig()

        # Process state and threading
        self._running = False
        self._shutdown_requested = False
        self._initialized = False
        self._stats = ConsumerStats()
        self._optimization_thread: Optional[threading.Thread] = None
        self._renderer_thread: Optional[threading.Thread] = None

        # Performance settings
        self.max_frame_wait_timeout = 1.0
        self.use_optimization = True
        self.brightness_scale = 1.0
        self.max_optimization_iterations = 50  # No timing constraints

        # WLED connection management
        self.wled_reconnect_interval = 10.0  # Try to reconnect every 10 seconds
        self.last_wled_attempt = 0.0

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

            # Try to connect to WLED controller (not required for startup)
            if self._wled_client.connect():
                logger.info("Connected to WLED controller successfully")
            else:
                logger.warning("Failed to connect to WLED controller - will retry periodically")

            # Initialize test renderer if enabled
            if self.enable_test_renderer:
                self._initialize_test_renderer()

            # Configure frame renderer output targets
            self._frame_renderer.set_output_targets(wled_client=self._wled_client, test_renderer=self._test_renderer)

            self._initialized = True
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

            if not self._initialized and not self.initialize():
                return False

            # Start optimization and renderer threads
            self._running = True
            self._shutdown_requested = False

            self._optimization_thread = threading.Thread(target=self._optimization_loop, name="OptimizationThread")
            self._renderer_thread = threading.Thread(target=self._rendering_loop, name="RendererThread")

            self._optimization_thread.start()
            self._renderer_thread.start()

            logger.info("Consumer process started with dual threads")
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

            # Wait for both threads to finish
            if self._optimization_thread and self._optimization_thread.is_alive():
                self._optimization_thread.join(timeout=5.0)
                if self._optimization_thread.is_alive():
                    logger.warning("Optimization thread did not stop gracefully")

            if self._renderer_thread and self._renderer_thread.is_alive():
                self._renderer_thread.join(timeout=5.0)
                if self._renderer_thread.is_alive():
                    logger.warning("Renderer thread did not stop gracefully")

            # Cleanup components
            self._cleanup()

            logger.info("Consumer process stopped")

        except Exception as e:
            logger.error(f"Error stopping consumer process: {e}")

    def _optimization_loop(self) -> None:
        """Optimization thread - processes frames as fast as possible."""
        logger.info("Optimization thread started")

        while self._running and not self._shutdown_requested:
            try:
                # Check for shutdown signal from control state
                if self._control_state.should_shutdown():
                    logger.info("Shutdown signal received from control state")
                    break

                # Wait for new frame
                buffer_info = self._frame_consumer.wait_for_ready_buffer(timeout=self.max_frame_wait_timeout)

                if buffer_info is None:
                    # Timeout waiting for frame
                    continue

                # Process the frame for optimization only
                self._process_frame_optimization(buffer_info)

            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(0.1)  # Brief pause before retrying

        logger.info("Optimization thread ended")

    def _rendering_loop(self) -> None:
        """Rendering thread - handles timestamp-based frame output."""
        logger.info("Renderer thread started")

        while self._running and not self._shutdown_requested:
            try:
                # Check for WLED reconnection periodically
                current_time = time.time()
                if current_time - self.last_wled_attempt >= self.wled_reconnect_interval:
                    self._try_wled_reconnect()
                    self.last_wled_attempt = current_time

                # Get next LED values with timeout
                led_data = self._led_buffer.read_led_values(timeout=0.1)

                if led_data is None:
                    continue  # Timeout - no data available

                led_values, timestamp, metadata = led_data

                # Render with timestamp-based timing
                success = self._frame_renderer.render_frame_at_timestamp(led_values, timestamp, metadata)

                if not success:
                    logger.warning("Frame rendering failed")

            except Exception as e:
                logger.error(f"Error in rendering loop: {e}")
                time.sleep(0.01)

        logger.info("Renderer thread ended")

    def _try_wled_reconnect(self) -> None:
        """Try to reconnect to WLED controller if not connected."""
        try:
            if not self._wled_client.is_connected():
                logger.debug("Attempting WLED reconnection...")
                if self._wled_client.connect():
                    logger.info("WLED controller reconnected successfully")
                    # Update frame renderer with new connection
                    self._frame_renderer.set_output_targets(
                        wled_client=self._wled_client, test_renderer=self._test_renderer
                    )
                else:
                    logger.debug("WLED reconnection failed - will retry later")
        except Exception as e:
            logger.debug(f"Error during WLED reconnection attempt: {e}")

    def _process_frame_optimization(self, buffer_info) -> None:
        """
        Process frame for LED optimization only - no rendering.
        Rendering handled by separate renderer thread.

        Args:
            buffer_info: Buffer information from frame consumer
        """
        start_time = time.time()

        try:
            # Extract frame and timestamp
            frame_array = buffer_info.get_array()
            timestamp = getattr(buffer_info, "presentation_timestamp", None) or time.time()

            # Validate frame shape
            if frame_array.shape != (FRAME_HEIGHT, FRAME_WIDTH, 4):
                logger.warning(f"Unexpected frame shape: {frame_array.shape}")
                return

            # Convert RGBA to RGB
            rgb_frame = frame_array[:, :, :3].astype(np.uint8)

            # Apply brightness scaling
            if self.brightness_scale != 1.0:
                rgb_frame = (rgb_frame * self.brightness_scale).clip(0, 255).astype(np.uint8)

            # Optimize LED values (no timing constraints)
            optimization_start = time.time()
            result = self._led_optimizer.optimize_frame(rgb_frame, max_iterations=self.max_optimization_iterations)
            optimization_time = time.time() - optimization_start

            # Check optimization result
            if not result.converged:
                logger.debug(f"Optimization did not converge after {result.iterations} iterations")
                self._stats.optimization_errors += 1

            # Store in LED buffer with timestamp
            led_values_uint8 = result.led_values.astype(np.uint8)
            success = self._led_buffer.write_led_values(
                led_values_uint8,
                timestamp,
                {
                    "optimization_time": optimization_time,
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "error_metrics": result.error_metrics,
                },
            )

            if not success:
                logger.warning("Failed to write LED values to buffer")

            # Update statistics (optimization only)
            total_time = time.time() - start_time
            self._stats.frames_processed += 1
            self._stats.total_processing_time += total_time
            self._stats.total_optimization_time += optimization_time
            self._stats.last_frame_time = start_time

            # Update optimization FPS (independent of rendering)
            self._stats.current_optimization_fps = 1.0 / total_time if total_time > 0 else 0.0

            # Log performance periodically
            if self._stats.frames_processed % 100 == 0:
                avg_fps = self._stats.get_average_fps()
                avg_opt_time = self._stats.get_average_optimization_time()
                buffer_stats = self._led_buffer.get_buffer_stats()

                logger.info(
                    f"Optimization: {avg_fps:.1f} fps avg, "
                    f"opt: {avg_opt_time * 1000:.1f}ms, "
                    f"buffer: {buffer_stats['utilization']:.1%}, "
                    f"errors: {self._stats.optimization_errors}"
                )

        except Exception as e:
            logger.error(f"Error in optimization: {e}")
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
            "current_optimization_fps": self._stats.current_optimization_fps,
            "average_optimization_fps": self._stats.get_average_fps(),
            "total_processing_time": self._stats.total_processing_time,
            "average_optimization_time": self._stats.get_average_optimization_time(),
            "optimization_errors": self._stats.optimization_errors,
            "max_optimization_iterations": self.max_optimization_iterations,
            "use_optimization": self.use_optimization,
            "brightness_scale": self.brightness_scale,
            "led_count": LED_COUNT,
            "frame_dimensions": (FRAME_WIDTH, FRAME_HEIGHT),
            "wled_stats": self._wled_client.get_statistics(),
            "optimizer_stats": self._led_optimizer.get_optimizer_stats(),
            "test_renderer_enabled": self.enable_test_renderer,
            "test_renderer_stats": (self._test_renderer.get_statistics() if self._test_renderer else None),
            "led_buffer_stats": self._led_buffer.get_buffer_stats(),
            "renderer_stats": self._frame_renderer.get_renderer_stats(),
        }

    def set_performance_settings(
        self,
        max_optimization_iterations: Optional[int] = None,
        use_optimization: Optional[bool] = None,
        brightness_scale: Optional[float] = None,
    ) -> None:
        """
        Update performance settings.

        Args:
            max_optimization_iterations: Maximum optimization iterations (no timing constraints)
            use_optimization: Whether to use full optimization or simple sampling
            brightness_scale: Brightness scaling factor (0.0 to 1.0)
        """
        if max_optimization_iterations is not None:
            self.max_optimization_iterations = max(1, min(200, max_optimization_iterations))

        if use_optimization is not None:
            self.use_optimization = use_optimization

        if brightness_scale is not None:
            self.brightness_scale = max(0.0, min(1.0, brightness_scale))

        logger.info(
            f"Updated performance settings: max_iters={self.max_optimization_iterations}, "
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
                success = self._initialize_test_renderer()
                if success:
                    # Update frame renderer output targets
                    self._frame_renderer.set_test_renderer_enabled(True)
                return success
            else:
                # Disable test renderer
                self.enable_test_renderer = False
                self._frame_renderer.set_test_renderer_enabled(False)
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

    def set_timing_parameters(
        self, first_frame_delay_ms: Optional[float] = None, timing_tolerance_ms: Optional[float] = None
    ) -> None:
        """
        Update timing parameters for the frame renderer.

        Args:
            first_frame_delay_ms: New first frame delay
            timing_tolerance_ms: New timing tolerance
        """
        self._frame_renderer.set_timing_parameters(
            first_frame_delay_ms=first_frame_delay_ms, timing_tolerance_ms=timing_tolerance_ms
        )

    def set_led_buffer_size(self, buffer_size: int) -> bool:
        """
        Set LED buffer size (will clear existing data).

        Args:
            buffer_size: New buffer size

        Returns:
            True if successful, False otherwise
        """
        return self._led_buffer.set_buffer_size(buffer_size)

    def clear_led_buffer(self) -> None:
        """Clear all data from LED buffer."""
        self._led_buffer.clear()

    def reset_renderer_stats(self) -> None:
        """Reset renderer timing statistics."""
        self._frame_renderer.reset_stats()

    def set_wled_enabled(self, enabled: bool) -> None:
        """Enable or disable WLED output."""
        self._frame_renderer.set_wled_enabled(enabled)
        logger.info(f"WLED output {'enabled' if enabled else 'disabled'}")

    def is_renderer_initialized(self) -> bool:
        """Check if renderer is initialized with timing delta."""
        return self._frame_renderer.is_initialized()

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

            if hasattr(self, "_led_buffer"):
                self._led_buffer.clear()

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
