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
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..const import FRAME_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from ..core import ControlState, FrameConsumer
from ..utils.frame_timing import FrameTimingData, FrameTimingLogger
from .frame_renderer import FrameRenderer
from .led_buffer import LEDBuffer
from .led_optimizer import LEDOptimizer
from .preview_sink import PreviewSink, PreviewSinkConfig
from .test_sink import TestSink, TestSinkConfig
from .transition_processor import TransitionProcessor
from .wled_sink import WLEDSink, WLEDSinkConfig

logger = logging.getLogger(__name__)


@dataclass
class ConsumerStats:
    """Consumer process statistics."""

    frames_processed: int = 0
    frames_dropped_early: int = 0  # Dropped before optimization
    total_processing_time: float = 0.0
    total_optimization_time: float = 0.0
    optimization_errors: int = 0
    transmission_errors: int = 0
    last_frame_time: float = 0.0
    current_optimization_fps: float = 0.0

    # New FPS tracking
    consumer_input_fps: float = 0.0  # Last measured input FPS from producer
    renderer_output_fps_ewma: float = 0.0  # EWMA of renderer output FPS
    dropped_frames_percentage_ewma: float = 0.0  # EWMA of dropped frame percentage

    # Internal tracking for FPS calculations
    _last_frame_timestamp: float = 0.0
    _last_render_time: float = 0.0
    _render_count: int = 0

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

    def update_consumer_input_fps(self, frame_timestamp: float) -> None:
        """Update consumer input FPS based on frame timestamps."""
        if self._last_frame_timestamp > 0:
            time_diff = frame_timestamp - self._last_frame_timestamp
            if time_diff > 0:
                self.consumer_input_fps = 1.0 / time_diff
        self._last_frame_timestamp = frame_timestamp

    def update_renderer_output_fps(self, alpha: float = 0.1) -> None:
        """Update renderer output FPS using EWMA."""
        current_time = time.time()
        if self._last_render_time > 0:
            time_diff = current_time - self._last_render_time
            if time_diff > 0:
                current_fps = 1.0 / time_diff
                if self.renderer_output_fps_ewma == 0:
                    self.renderer_output_fps_ewma = current_fps
                else:
                    self.renderer_output_fps_ewma = (1 - alpha) * self.renderer_output_fps_ewma + alpha * current_fps
        self._last_render_time = current_time
        self._render_count += 1

    def update_dropped_frames_ewma(self, alpha: float = 0.1) -> None:
        """Update dropped frames percentage using EWMA."""
        total_frames = self.frames_processed + self.frames_dropped_early
        if total_frames > 0:
            current_drop_rate = self.frames_dropped_early / total_frames
            if self.dropped_frames_percentage_ewma == 0:
                self.dropped_frames_percentage_ewma = current_drop_rate
            else:
                self.dropped_frames_percentage_ewma = (
                    1 - alpha
                ) * self.dropped_frames_percentage_ewma + alpha * current_drop_rate


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
        test_renderer_config: Optional[TestSinkConfig] = None,
        timing_log_path: Optional[str] = None,
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
            timing_log_path: Path to CSV file for timing data logging (optional)
        """
        self.buffer_name = buffer_name
        self.control_name = control_name

        # Initialize components
        self._frame_consumer = FrameConsumer(buffer_name)
        self._control_state = ControlState(control_name)
        self._led_optimizer = LEDOptimizer(
            diffusion_patterns_path=diffusion_patterns_path,
        )

        # Note: Playlist sync handled by producer, consumer just tracks rendered items
        # Configure WLED client
        wled_config = WLEDSinkConfig(
            host=wled_host,
            port=wled_port,
            led_count=LED_COUNT,
        )
        self._wled_client = WLEDSink(wled_config)

        # New timestamp-based rendering components
        self._led_buffer = LEDBuffer(buffer_size=10)

        # Create frame renderer with LED ordering from pattern file if available
        from ..utils.pattern_loader import create_frame_renderer_with_pattern

        self._frame_renderer = create_frame_renderer_with_pattern(
            diffusion_patterns_path, first_frame_delay_ms=100.0, timing_tolerance_ms=5.0
        )
        logger.info(f"Frame renderer created with LED ordering from {diffusion_patterns_path}")

        # Test renderer (optional)
        self.enable_test_renderer = enable_test_renderer
        self._test_renderer: Optional[TestSink] = None
        self._test_renderer_config = test_renderer_config or TestSinkConfig()

        # Preview sink for web interface
        self._preview_sink: Optional[PreviewSink] = None
        self._preview_sink_config = PreviewSinkConfig()

        # Process state and threading
        self._running = False

        # Preview data is now handled by PreviewSink
        self._shutdown_requested = False
        self._initialized = False
        self._stats = ConsumerStats()

        # Debug frame writing (first 10 frames)
        self._debug_frame_count = 0
        self._debug_max_frames = 10
        self._debug_frame_dir = Path("/tmp/prismatron_debug_frames")
        self._debug_frame_dir.mkdir(exist_ok=True)

        # Periodic logging for pipeline debugging
        self._last_consumer_log_time = 0.0
        self._consumer_log_interval = 2.0  # Log every 2 seconds
        self._frames_with_content = 0  # Frames with non-zero LED values

        # Frame gap tracking for debugging
        self._last_frame_timestamp = 0.0  # Last frame timestamp received
        self._last_frame_receive_time = 0.0  # Last real-time when frame was received
        self._frame_timestamp_gap_threshold = 0.1  # 100ms threshold for frame timestamp gaps
        self._realtime_gap_threshold = 0.2  # 200ms threshold for real-time gaps

        # Frame sequence tracking for complete logging
        self._last_frame_index_seen = 0  # Track last frame index to detect gaps
        self._optimization_thread: Optional[threading.Thread] = None
        self._renderer_thread: Optional[threading.Thread] = None

        # Track last rendered playlist item index to detect transitions
        self._last_rendered_item_index = -1

        # Timing logger for performance analysis
        self._timing_logger: Optional[FrameTimingLogger] = None
        if timing_log_path:
            self._timing_logger = FrameTimingLogger(timing_log_path)

        # Performance settings
        self.max_frame_wait_timeout = 1.0
        self.brightness_scale = 1.0
        self.optimization_iterations = 5
        self.target_fps = 15.0  # Target processing FPS

        # WLED connection management
        self.wled_reconnect_interval = 10.0  # Try to reconnect every 10 seconds
        self._wled_reconnection_thread: Optional[threading.Thread] = None
        self._wled_reconnection_event = threading.Event()  # Signal for graceful shutdown

        # Transition processor for playlist item transitions
        self._transition_processor = TransitionProcessor()

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

            # Set LED optimizer logging to WARNING to reduce noise
            led_optimizer_logger = logging.getLogger("src.consumer.led_optimizer")
            led_optimizer_logger.setLevel(logging.WARNING)

            # Set sparse tensor logging to WARNING to reduce noise
            sparse_tensor_logger = logging.getLogger("src.utils.single_block_sparse_tensor")
            sparse_tensor_logger.setLevel(logging.WARNING)

            # Set compute kernel logging to WARNING to reduce noise
            compute_kernel_logger = logging.getLogger("src.utils.kernels.compute_optimized_3d_int8")
            compute_kernel_logger.setLevel(logging.WARNING)

            # Connect to shared memory buffer (required)
            if not self._frame_consumer.connect():
                logger.error("Failed to connect to shared memory buffer")
                return False

            # Connect to control state (required)
            if not self._control_state.connect():
                logger.error("Failed to connect to control state")
                return False

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

            # Initialize preview sink for web interface
            self._initialize_preview_sink()

            # Configure frame renderer output targets
            self._frame_renderer.set_output_targets(
                wled_sink=self._wled_client, test_sink=self._test_renderer, preview_sink=self._preview_sink
            )

            # Connect preview sink to frame renderer for statistics
            if self._preview_sink:
                self._preview_sink.set_frame_renderer(self._frame_renderer)

            # Note: Playlist sync handled by producer - consumer just logs renderer transitions

            # Start timing logger if configured
            if self._timing_logger:
                if not self._timing_logger.start_logging():
                    logger.warning("Failed to start timing logger, continuing without timing data")
                    self._timing_logger = None
                else:
                    logger.info("Started frame timing logger")

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
            self._wled_reconnection_thread = threading.Thread(
                target=self._wled_reconnection_loop, name="WLEDReconnectionThread"
            )

            # Backward compatibility alias
            self._process_thread = self._optimization_thread

            self._optimization_thread.start()
            self._renderer_thread.start()
            self._wled_reconnection_thread.start()

            logger.info("Consumer process started with optimization, renderer, and WLED reconnection threads")
            return True

        except Exception as e:
            logger.error(f"Failed to start consumer process: {e}")
            return False

    def stop(self) -> None:
        """Stop the consumer process gracefully."""
        if self._shutdown_requested:
            return  # Avoid duplicate stop calls

        try:
            self._shutdown_requested = True
            self._running = False

            # Signal WLED reconnection thread to stop
            self._wled_reconnection_event.set()

            logger.info("Stopping consumer process...")

            # Wait for all threads to finish
            if self._optimization_thread and self._optimization_thread.is_alive():
                self._optimization_thread.join(timeout=5.0)
                if self._optimization_thread.is_alive():
                    logger.warning("Optimization thread did not stop gracefully")

            if self._renderer_thread and self._renderer_thread.is_alive():
                self._renderer_thread.join(timeout=5.0)
                if self._renderer_thread.is_alive():
                    logger.warning("Renderer thread did not stop gracefully")

            if self._wled_reconnection_thread and self._wled_reconnection_thread.is_alive():
                self._wled_reconnection_thread.join(timeout=2.0)
                if self._wled_reconnection_thread.is_alive():
                    logger.warning("WLED reconnection thread did not stop gracefully")

            # Cleanup components
            self._cleanup()

            logger.info("Consumer process stopped")

        except Exception as e:
            logger.error(f"Error stopping consumer process: {e}")

    def _track_frame_gaps(
        self, frame_timestamp: float, receive_time: float, has_presentation_timestamp: bool = True
    ) -> None:
        """
        Track and log gaps in frame timestamps and real-time frame reception.

        Args:
            frame_timestamp: Presentation timestamp of the current frame (or receive time if no presentation timestamp)
            receive_time: Wall-clock time when frame was received
            has_presentation_timestamp: Whether frame_timestamp is a real presentation timestamp or fallback
        """
        # Check for frame timestamp gaps (content timing) - only if we have real presentation timestamps
        if has_presentation_timestamp and self._last_frame_timestamp > 0:
            timestamp_gap = frame_timestamp - self._last_frame_timestamp
            if timestamp_gap > self._frame_timestamp_gap_threshold:
                logger.warning(
                    f"Large frame timestamp gap: {timestamp_gap*1000:.1f}ms "
                    f"(previous: {self._last_frame_timestamp:.3f}, current: {frame_timestamp:.3f})"
                )

        # Check for real-time gaps (processing timing) - always track
        if self._last_frame_receive_time > 0:
            realtime_gap = receive_time - self._last_frame_receive_time
            if realtime_gap > self._realtime_gap_threshold:
                logger.warning(
                    f"Large real-time gap between frames: {realtime_gap*1000:.1f}ms "
                    f"(previous: {self._last_frame_receive_time:.3f}, current: {receive_time:.3f})"
                )

        # Update tracking variables
        self._last_frame_timestamp = frame_timestamp
        self._last_frame_receive_time = receive_time

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
                logger.error(f"Error in optimization loop: {e}", exc_info=True)
                time.sleep(0.1)  # Brief pause before retrying

        logger.info("Optimization thread ended")

    def _rendering_loop(self) -> None:
        """Rendering thread - handles timestamp-based frame output."""
        logger.info("Renderer thread started")

        while self._running and not self._shutdown_requested:
            try:
                # Get next LED values with timeout
                led_data = self._led_buffer.read_led_values(timeout=0.1)

                if led_data is None:
                    continue  # Timeout - no data available

                led_values, timestamp, metadata = led_data

                # Extract timing data and mark read-from-LED-buffer time
                timing_data = None
                if metadata and "timing_data" in metadata:
                    timing_data = metadata["timing_data"]
                    if timing_data:
                        timing_data.mark_read_from_led_buffer()

                # Check for playlist item transitions (but update rendering_index after successful render)
                is_first_frame_of_item = metadata and metadata.get("is_first_frame_of_item", False)
                playlist_item_index = metadata.get("playlist_item_index", -1) if metadata else -1
                should_update_rendering_index = (
                    is_first_frame_of_item
                    and playlist_item_index >= 0
                    and playlist_item_index != self._last_rendered_item_index
                )

                # Debug logging for index 0 issue
                if playlist_item_index == 0:
                    logger.info(
                        f"INDEX 0 DEBUG: is_first_frame={is_first_frame_of_item}, index={playlist_item_index}, last_rendered={self._last_rendered_item_index}, should_update={should_update_rendering_index}"
                    )

                if should_update_rendering_index:
                    logger.info(f"RENDERER: Starting to render playlist item {playlist_item_index}")

                # Debug logging for high FPS investigation
                frame_count = self._stats.frames_processed
                if frame_count % 100 == 0:  # Log every 100th frame
                    buffer_stats = self._led_buffer.get_buffer_stats()
                    current_time = time.time()
                    logger.debug(
                        f"Renderer pulling frame {frame_count}: timestamp={timestamp:.3f}, "
                        f"buffer_depth={buffer_stats['current_count']}, "
                        f"current_time={current_time:.3f}"
                    )

                # Render with timestamp-based timing
                success = self._frame_renderer.render_frame_at_timestamp(led_values, timestamp, metadata)

                # Update rendering_index AFTER successful render to reflect what's actually been rendered
                if success and should_update_rendering_index:
                    self._last_rendered_item_index = playlist_item_index
                    try:
                        self._control_state.update_status(rendering_index=playlist_item_index)
                        logger.info(f"Updated rendering_index to {playlist_item_index} after successful render")
                    except Exception as e:
                        logger.warning(f"Failed to update rendering_index in ControlState: {e}")

                    # Note: Don't send position updates to sync service as this creates a feedback loop
                    # The producer already sends next_item() commands when content finishes

                # Mark render time and log timing data based on success
                if timing_data:
                    if success:
                        # Successful render - mark render time and log complete data
                        timing_data.mark_render()
                    # Always log timing data (Case 3: failed renders have empty render_time)
                    if self._timing_logger:
                        self._timing_logger.log_frame(timing_data)
                        if not success:
                            logger.debug(f"Logged render-failed frame {timing_data.frame_index} (render failure)")

                # Update renderer output FPS tracking
                if success:
                    self._stats.update_renderer_output_fps()

                if not success:
                    logger.warning("Frame rendering failed")

            except Exception as e:
                logger.error(f"Error in rendering loop: {e}", exc_info=True)
                time.sleep(0.01)

        logger.info("Renderer thread ended")

    def _wled_reconnection_loop(self) -> None:
        """Dedicated thread for WLED reconnection attempts."""
        logger.info("WLED reconnection thread started")

        while self._running and not self._shutdown_requested:
            try:
                # Wait for reconnect interval or shutdown signal
                if self._wled_reconnection_event.wait(timeout=self.wled_reconnect_interval):
                    # Event was set - shutdown requested
                    break

                # Try reconnection if not connected
                if not self._wled_client.is_connected:
                    logger.debug("Attempting WLED reconnection...")
                    if self._wled_client.connect():
                        logger.info("WLED controller reconnected successfully")
                        # Update frame renderer with new connection
                        self._frame_renderer.set_output_targets(
                            wled_sink=self._wled_client, test_sink=self._test_renderer, preview_sink=self._preview_sink
                        )
                    else:
                        logger.debug("WLED reconnection failed - will retry later")

            except Exception as e:
                logger.error(f"Error in WLED reconnection loop: {e}", exc_info=True)
                time.sleep(1.0)  # Brief pause before retrying

        logger.info("WLED reconnection thread ended")

    def _try_wled_reconnect(self) -> None:
        """
        Legacy method - WLED reconnection now handled by dedicated thread.

        This method is kept for backward compatibility but does nothing
        since reconnection is now handled by _wled_reconnection_loop().
        """

    def _process_frame(self, buffer_info) -> None:
        """Process a frame (backward compatibility wrapper)."""
        # Process the frame through optimization
        self._process_frame_optimization(buffer_info)

        # For backward compatibility with tests, also handle direct transmission
        # Get the latest LED values from buffer and send to WLED
        try:
            led_data = self._led_buffer.read_latest_led_values()
            if led_data is not None and hasattr(self, "_wled_client"):
                # Send to WLED directly (for test compatibility)
                transmission_result = self._wled_client.send_led_data(led_data.led_values)
                if not transmission_result.success:
                    self._stats.transmission_errors += 1
        except Exception as e:
            logger.debug(f"Backward compatibility transmission failed: {e}")
            self._stats.transmission_errors += 1

    def _process_loop(self) -> None:
        """
        Main processing loop (backward compatibility method).

        This method is provided for test compatibility. The actual implementation
        uses separate optimization and rendering threads.
        """
        logger.info("Starting backward compatibility process loop")

        while self._running and not self._control_state.should_shutdown():
            try:
                # Wait for frame to be ready
                buffer_info = self._frame_consumer.wait_for_ready_buffer(timeout=self.max_frame_wait_timeout)

                if buffer_info is None:
                    # No frame available, continue
                    continue

                # Process the frame
                self._process_frame(buffer_info)

            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                self._stats.optimization_errors += 1

        logger.info("Process loop terminated")

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
            # TODO: Check get_array_interleaved to ensure this is on the GPU already
            frame_array = buffer_info.get_array_interleaved(FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS)

            # Separate presentation timestamp from receive time for proper gap tracking
            current_receive_time = time.time()
            if buffer_info.metadata and hasattr(buffer_info.metadata, "presentation_timestamp"):
                timestamp = buffer_info.metadata.presentation_timestamp
                self._track_frame_gaps(timestamp, current_receive_time, has_presentation_timestamp=True)
            else:
                # Use receive time as fallback, but track gaps separately
                logger.warning(
                    f"Frame missing presentation timestamp metadata, using receive time {current_receive_time} for processing"
                )
                timestamp = current_receive_time
                self._track_frame_gaps(timestamp, current_receive_time, has_presentation_timestamp=False)

            # Read playlist metadata for renderer synchronization
            playlist_item_index = -1
            is_first_frame_of_item = False
            timing_data = None
            if buffer_info.metadata:
                playlist_item_index = buffer_info.metadata.playlist_item_index
                is_first_frame_of_item = buffer_info.metadata.is_first_frame_of_item
                timing_data = buffer_info.metadata.timing_data

                # Mark read-from-buffer time in timing data
                if timing_data:
                    timing_data.mark_read_from_buffer()

                    # Detect and log missing frames (Case 1: Never received by consumer)
                    # Detect missing frames before processing this one
                    self._detect_and_log_missing_frames(timing_data.frame_index)

            # Update consumer input FPS tracking
            self._stats.update_consumer_input_fps(timestamp)

            # Check if frame is already late - drop if so, otherwise proceed with optimization
            if self._frame_renderer.is_frame_late(timestamp, late_threshold_ms=50.0):
                logger.debug(f"Frame already late (timestamp={timestamp:.3f}) - dropping before optimization")
                self._stats.frames_dropped_early += 1
                self._stats.update_dropped_frames_ewma()

                # Log dropped frame to timing data if configured (Case 2: Dropped before optimization)
                if timing_data and self._timing_logger:
                    # Case 2: Frame dropped before optimization - has frame info and buffer times, but no optimization/render times
                    self._timing_logger.log_frame(timing_data)
                    logger.debug(f"Logged early-dropped frame {timing_data.frame_index} (late before optimization)")

                return

            # Validate frame shape
            if frame_array.shape != (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS):
                logger.warning(f"Unexpected frame shape: {frame_array.shape}")
                return

            # Convert RGBA to RGB
            # TODO; Remove this, we only have RGB
            rgb_frame = frame_array[:, :, :3].astype(np.uint8)

            # Debug: Write first 10 frames to temporary files for analysis
            if self._debug_frame_count < self._debug_max_frames:
                try:
                    debug_file = self._debug_frame_dir / f"frame_{self._debug_frame_count:03d}.npy"
                    np.save(debug_file, rgb_frame)
                    logger.info(f"DEBUG: Wrote frame {self._debug_frame_count} to {debug_file}")
                    self._debug_frame_count += 1
                except Exception as e:
                    logger.warning(f"DEBUG: Failed to write frame {self._debug_frame_count}: {e}")

            # Apply brightness scaling
            if self.brightness_scale != 1.0:
                # TODO: Do this on the GPU (i.e. don't move data back and forth)
                rgb_frame = (rgb_frame * self.brightness_scale).clip(0, 255).astype(np.uint8)

            # Apply playlist item transitions before LED optimization
            transition_start = time.time()
            metadata_dict = self._extract_metadata_dict(buffer_info.metadata)
            rgb_frame = self._transition_processor.process_frame(rgb_frame, metadata_dict)
            transition_time = time.time() - transition_start

            # Optimize LED values (no timing constraints)
            optimization_start = time.time()

            # Set iterations
            iterations = self.optimization_iterations
            result = self._led_optimizer.optimize_frame(rgb_frame, max_iterations=iterations)
            optimization_time = time.time() - optimization_start

            # Check optimization result
            if not result.converged:
                self._stats.optimization_errors += 1

            # Store in LED buffer with timestamp
            led_values_uint8 = result.led_values.astype(np.uint8)

            # Check if LED values have non-zero content for logging
            if led_values_uint8.max() > 0:
                self._frames_with_content += 1

            # Mark write-to-LED-buffer time before writing
            if timing_data:
                timing_data.mark_write_to_led_buffer()

            success = self._led_buffer.write_led_values(
                led_values_uint8,
                timestamp,
                {
                    "optimization_time": optimization_time,
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "error_metrics": result.error_metrics,
                    "playlist_item_index": playlist_item_index,
                    "is_first_frame_of_item": is_first_frame_of_item,
                    "timing_data": timing_data,  # Pass timing data through to renderer
                },
                block=True,  # Use backpressure - wait for renderer to free up space
                timeout=0.1,  # Short timeout - just wait for one slot to become available
            )

            # Preview data is now handled by PreviewSink in frame renderer

            if not success:
                logger.warning("Failed to write LED values to buffer")

                # Log optimization-completed but buffer-write-failed frame to timing data
                if timing_data and self._timing_logger:
                    # This frame was optimized but couldn't be written to LED buffer
                    optimization_failed_timing = FrameTimingData(
                        frame_index=timing_data.frame_index,
                        plugin_timestamp=timing_data.plugin_timestamp,
                        producer_timestamp=timing_data.producer_timestamp,
                        item_duration=timing_data.item_duration,
                        write_to_buffer_time=timing_data.write_to_buffer_time,
                        read_from_buffer_time=timing_data.read_from_buffer_time,
                        write_to_led_buffer_time=timing_data.write_to_led_buffer_time,
                        # Leave render times None to indicate LED buffer write failure
                    )
                    self._timing_logger.log_frame(optimization_failed_timing)

            # Update statistics (optimization only)
            total_time = time.time() - start_time
            self._stats.frames_processed += 1
            self._stats.total_processing_time += total_time
            self._stats.total_optimization_time += optimization_time
            self._stats.last_frame_time = start_time

            # Update optimization FPS (independent of rendering)
            self._stats.current_optimization_fps = 1.0 / total_time if total_time > 0 else 0.0

            # Update dropped frames EWMA (includes both early drops and successful processing)
            self._stats.update_dropped_frames_ewma()

            # Periodic logging for pipeline debugging (every 2 seconds)
            current_time = time.time()
            if current_time - self._last_consumer_log_time >= self._consumer_log_interval:
                content_ratio = (self._frames_with_content / max(1, self._stats.frames_processed)) * 100
                avg_fps = self._stats.get_average_fps()
                avg_opt_time = self._stats.get_average_optimization_time()

                # Get LED buffer stats
                buffer_stats = self._led_buffer.get_buffer_stats()
                buffer_depth = buffer_stats["current_count"]

                logger.info(
                    f"CONSUMER PIPELINE: {self._stats.frames_processed} frames optimized, "
                    f"{self._stats.frames_dropped_early} dropped early, "
                    f"{self._frames_with_content} with LED content ({content_ratio:.1f}%), "
                    f"input FPS: {self._stats.consumer_input_fps:.1f}, "
                    f"output FPS: {self._stats.renderer_output_fps_ewma:.1f}, "
                    f"drop rate: {self._stats.dropped_frames_percentage_ewma * 100:.1f}%, "
                    f"opt time: {avg_opt_time * 1000:.1f}ms, "
                    f"LED buffer depth: {buffer_depth}"
                )
                self._last_consumer_log_time = current_time

                # Update consumer statistics in ControlState for IPC with web server
                self._update_consumer_statistics_in_control_state()

            # Log performance periodically (every 100 frames)
            if self._stats.frames_processed % 100 == 0:
                avg_fps = self._stats.get_average_fps()
                avg_opt_time = self._stats.get_average_optimization_time()
                buffer_stats = self._led_buffer.get_buffer_stats()

                logger.debug(
                    f"Optimization: {avg_fps:.1f} fps avg, "
                    f"opt: {avg_opt_time * 1000:.1f}ms, "
                    f"buffer: {buffer_stats['utilization']:.1%}, "
                    f"errors: {self._stats.optimization_errors}"
                )

        except Exception as e:
            logger.error(f"Error in optimization: {e}", exc_info=True)
            self._stats.optimization_errors += 1

    def _extract_metadata_dict(self, metadata) -> Dict[str, Any]:
        """
        Extract frame metadata into dictionary format for transition processor.

        Args:
            metadata: Frame metadata object from shared buffer

        Returns:
            Dictionary containing metadata fields needed by transition processor
        """
        try:
            if metadata is None:
                return {}

            # Extract fields that the transition processor needs
            metadata_dict = {}

            # Add transition fields if they exist
            for field in [
                "transition_in_type",
                "transition_in_duration",
                "transition_out_type",
                "transition_out_duration",
                "item_timestamp",
                "item_duration",
            ]:
                if hasattr(metadata, field):
                    metadata_dict[field] = getattr(metadata, field)
                else:
                    # Set default values for missing fields
                    if field.endswith("_type"):
                        metadata_dict[field] = "none"
                    elif field.endswith(("_duration", "_timestamp")):
                        metadata_dict[field] = 0.0

            return metadata_dict

        except Exception as e:
            logger.warning(f"Error extracting metadata for transitions: {e}")
            return {}

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
            self._test_renderer = TestSink(mixed_tensor, self._test_renderer_config)

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

    def _initialize_preview_sink(self) -> bool:
        """
        Initialize the preview sink for web interface.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if not self._led_optimizer:
                logger.error("LED optimizer not available for preview sink")
                return False

            # Create preview sink (spatial mapping handled by frame renderer now)
            self._preview_sink = PreviewSink(self._preview_sink_config)

            # Start preview sink
            if self._preview_sink.start():
                logger.info("Preview sink initialized and started successfully")
                return True
            else:
                logger.error("Failed to start preview sink")
                self._preview_sink = None
                return False

        except Exception as e:
            logger.error(f"Error initializing preview sink: {e}")
            self._preview_sink = None
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
            "frames_dropped_early": self._stats.frames_dropped_early,
            "current_optimization_fps": self._stats.current_optimization_fps,
            "average_optimization_fps": self._stats.get_average_fps(),
            "consumer_input_fps": self._stats.consumer_input_fps,
            "renderer_output_fps": self._stats.renderer_output_fps_ewma,
            "dropped_frames_percentage": self._stats.dropped_frames_percentage_ewma * 100,  # Convert to percentage
            "total_processing_time": self._stats.total_processing_time,
            "average_optimization_time": self._stats.get_average_optimization_time(),
            "optimization_errors": self._stats.optimization_errors,
            "transmission_errors": self._stats.transmission_errors,
            "optimization_iterations": self.optimization_iterations,
            "brightness_scale": self.brightness_scale,
            "target_fps": self.target_fps,
            "led_count": LED_COUNT,
            "frame_dimensions": (FRAME_WIDTH, FRAME_HEIGHT),
            "wled_stats": self._wled_client.get_statistics(),
            "optimizer_stats": self._led_optimizer.get_optimizer_stats(),
            "test_renderer_enabled": self.enable_test_renderer,
            "test_renderer_stats": (self._test_renderer.get_statistics() if self._test_renderer else None),
            "led_buffer_stats": self._led_buffer.get_buffer_stats(),
            "renderer_stats": self._frame_renderer.get_renderer_stats(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get consumer process statistics (backward compatibility method).

        Returns:
            Dictionary with process statistics
        """
        return self.get_stats()

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
                    self._frame_renderer.set_test_sink_enabled(True)
                return success
            else:
                # Disable test renderer
                self.enable_test_renderer = False
                self._frame_renderer.set_test_sink_enabled(False)
                if self._test_renderer:
                    self._test_renderer.stop()
                    self._test_renderer = None

                logger.info("Test renderer disabled")
                return True

        except Exception as e:
            logger.error(f"Error setting test renderer enabled state: {e}")
            return False

    def get_test_renderer(self) -> Optional[TestSink]:
        """
        Get the test renderer instance.

        Returns:
            TestSink instance or None if not enabled
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

    def _detect_and_log_missing_frames(self, current_frame_index: int) -> None:
        """
        Detect missing frames and log them as Case 1: Never received by consumer.

        Since the circular buffer now blocks writes until consumed, there should be no
        missing frames in normal operation. This function will detect any anomalies.

        Args:
            current_frame_index: Frame index of current frame being processed
        """
        try:
            logger.debug(f"Gap detection called: frame={current_frame_index}, last_seen={self._last_frame_index_seen}")

            if self._last_frame_index_seen == 0:
                # Very first frame processed
                logger.debug(f"First frame: {current_frame_index}")
                if current_frame_index > 1:
                    # Missing frames before first frame (shouldn't happen with fixed buffer)
                    logger.warning(
                        f"First frame is {current_frame_index} - missing frames 1 to {current_frame_index-1}"
                    )
                    for missing_frame_index in range(1, current_frame_index):
                        if self._timing_logger:
                            missing_frame_timing = FrameTimingData(
                                frame_index=missing_frame_index,
                                plugin_timestamp=0.0,
                                producer_timestamp=0.0,
                                item_duration=0.0,
                            )
                            self._timing_logger.log_frame(missing_frame_timing)
                            logger.debug(f"Logged missing frame {missing_frame_index} (never received by consumer)")

                old_value = self._last_frame_index_seen
                self._last_frame_index_seen = current_frame_index
                logger.debug(f"Updated _last_frame_index_seen: {old_value} -> {current_frame_index}")
                return

            # Check for gaps in sequence
            expected_next = self._last_frame_index_seen + 1
            if current_frame_index == expected_next:
                # Normal sequence
                logger.debug(f"Normal sequence: {current_frame_index} follows {self._last_frame_index_seen}")
            elif current_frame_index > expected_next:
                # Gap detected - shouldn't happen with fixed buffer
                logger.warning(f"Frame gap detected: expected {expected_next}, got {current_frame_index}")
                for missing_frame_index in range(expected_next, current_frame_index):
                    if self._timing_logger:
                        missing_frame_timing = FrameTimingData(
                            frame_index=missing_frame_index,
                            plugin_timestamp=0.0,
                            producer_timestamp=0.0,
                            item_duration=0.0,
                        )
                        self._timing_logger.log_frame(missing_frame_timing)
                        logger.debug(f"Logged missing frame {missing_frame_index} (never received by consumer)")
            else:
                # Frame went backwards - major error
                logger.error(f"Frame sequence went backwards: expected {expected_next}, got {current_frame_index}")
                return  # Don't update last_seen if sequence is broken

            self._last_frame_index_seen = current_frame_index

        except Exception as e:
            logger.warning(f"Error detecting missing frames: {e}")

    def is_renderer_initialized(self) -> bool:
        """Check if renderer is initialized with timing delta."""
        return self._frame_renderer.is_initialized()

    def set_performance_settings(
        self,
        target_fps: Optional[float] = None,
        brightness_scale: Optional[float] = None,
    ) -> None:
        """
        Update performance settings.

        Args:
            target_fps: Target processing FPS (clamped to 1.0-60.0)
            use_optimization: Whether to use optimization
            brightness_scale: Brightness scaling factor (0.0-1.0)
        """
        if target_fps is not None:
            # Clamp target_fps to reasonable bounds
            self.target_fps = max(1.0, min(60.0, target_fps))
            logger.info(f"Target FPS set to {self.target_fps}")

        if brightness_scale is not None:
            # Clamp brightness to 0.0-1.0
            self.brightness_scale = max(0.0, min(1.0, brightness_scale))
            logger.info(f"Brightness scale set to {self.brightness_scale}")

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        # Use print instead of logger to avoid reentrant calls during signal handling
        try:
            print(f"Consumer: Received signal {signum}, initiating shutdown...")
            self.stop()
        except Exception as e:
            print(f"Consumer: Error during signal handling: {e}")
            # Force exit if graceful shutdown fails
            import os

            os._exit(1)

    # Preview data methods removed - now handled by PreviewSink

    def _update_consumer_statistics_in_control_state(self) -> None:
        """Update consumer statistics in ControlState for multi-process IPC."""
        try:
            # Get renderer statistics for late frame percentage
            renderer_stats = self._frame_renderer.get_renderer_stats()
            late_frame_percentage = renderer_stats.get("late_frame_percentage", 0.0)

            # Update the control state with consumer statistics
            status_updates = {
                "consumer_input_fps": self._stats.consumer_input_fps,
                "renderer_output_fps": self._stats.renderer_output_fps_ewma,
                "dropped_frames_percentage": self._stats.dropped_frames_percentage_ewma * 100,  # Convert to percentage
                "late_frame_percentage": late_frame_percentage,
            }

            # Update ControlState with new statistics
            self._control_state.update_status(**status_updates)

        except Exception as e:
            logger.warning(f"Failed to update consumer statistics in ControlState: {e}")

    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Cleanup components in reverse order
            if hasattr(self, "_preview_sink") and self._preview_sink:
                self._preview_sink.stop()

            if hasattr(self, "_test_renderer") and self._test_renderer:
                self._test_renderer.stop()

            # Note: No playlist sync client to cleanup in consumer

            # Stop timing logger
            if hasattr(self, "_timing_logger") and self._timing_logger:
                self._timing_logger.stop_logging()

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
    from src.utils.logging_utils import create_app_time_formatter

    formatter = create_app_time_formatter()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
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
