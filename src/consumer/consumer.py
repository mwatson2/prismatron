"""
Consumer Process Implementation.

This module implements the main consumer process that reads frames from
shared memory, optimizes LED values, and transmits to WLED controller.
"""

import contextlib
import logging
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cupy as cp
import numpy as np

from ..const import FRAME_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH
from ..core import ControlState, FrameConsumer
from ..core.control_state import ProducerState, RendererState
from ..utils.frame_drop_rate_ewma import FrameDropRateEwma
from ..utils.frame_timing import FrameTimingData, FrameTimingLogger
from .adaptive_frame_dropper import AdaptiveFrameDropper
from .audio_beat_analyzer import AudioBeatAnalyzer, BeatEvent
from .frame_renderer import FrameRenderer
from .led_buffer import LEDBuffer
from .led_optimizer import LEDOptimizer
from .led_transition_processor import LEDTransitionProcessor
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
    total_led_transition_time: float = 0.0
    optimization_errors: int = 0
    transmission_errors: int = 0
    last_frame_time: float = 0.0
    current_optimization_fps: float = 0.0

    # New FPS tracking
    consumer_input_fps: float = 0.0  # Last measured input FPS from producer
    renderer_output_fps_ewma: float = 0.0  # EWMA of renderer output FPS

    # Internal tracking for FPS calculations
    _last_frame_timestamp: float = 0.0
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

    def get_average_led_transition_time(self) -> float:
        """Get average LED transition time per frame."""
        if self.frames_processed > 0:
            return self.total_led_transition_time / self.frames_processed
        return 0.0

    def update_consumer_input_fps(self, frame_timestamp: float) -> None:
        """Update consumer input FPS based on frame timestamps."""
        old_fps = self.consumer_input_fps
        if self._last_frame_timestamp > 0:
            time_diff = frame_timestamp - self._last_frame_timestamp
            if time_diff > 0:
                self.consumer_input_fps = 1.0 / time_diff
        self._last_frame_timestamp = frame_timestamp

        # Debug logging every 30 updates
        if not hasattr(self, "_input_fps_debug_counter"):
            self._input_fps_debug_counter = 0
        self._input_fps_debug_counter += 1
        if self._input_fps_debug_counter % 30 == 0:
            logger.debug(f"CONSUMER INPUT FPS DEBUG: Updated from {old_fps:.2f} to {self.consumer_input_fps:.2f}")


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
        wled_hosts: Union[str, List[str]] = "192.168.7.140",
        wled_port: int = 4048,
        diffusion_patterns_path: Optional[str] = None,
        enable_test_renderer: bool = False,
        test_renderer_config: Optional[TestSinkConfig] = None,
        timing_log_path: Optional[str] = None,
        enable_adaptive_frame_dropping: bool = True,
        enable_audio_reactive: bool = False,
        audio_device: str = "auto",
        enable_batch_mode: bool = False,
        enable_position_shifting: bool = False,
        max_shift_distance: int = 3,
        shift_direction: str = "alternating",
    ):
        """
        Initialize consumer process.

        Args:
            buffer_name: Shared memory buffer name
            control_name: Control state name
            wled_hosts: WLED controller IP address(es) or hostname(s) - can be a single string or list
            wled_port: WLED controller port
            diffusion_patterns_path: Path to diffusion patterns
            enable_test_renderer: Enable test renderer for debugging
            test_renderer_config: Test renderer configuration
            timing_log_path: Path to CSV file for timing data logging (optional)
            enable_adaptive_frame_dropping: Enable adaptive frame dropping for LED buffer management
            enable_audio_reactive: Enable audio-reactive effects with beat detection
            audio_device: Audio device selection ('auto', 'cuda', 'cpu')
            enable_batch_mode: Enable batch processing (8 frames at once) for improved performance
            enable_position_shifting: Enable audio-reactive LED position shifting effects
            max_shift_distance: Maximum LED positions to shift on beats (3-4 typical)
            shift_direction: Position shift direction ("left", "right", "alternating")
        """
        self.buffer_name = buffer_name
        self.control_name = control_name

        # Initialize components
        self._frame_consumer = FrameConsumer(buffer_name)
        self._control_state = ControlState(control_name)
        self._led_optimizer = LEDOptimizer(
            diffusion_patterns_path=diffusion_patterns_path,
            enable_batch_mode=enable_batch_mode,
        )

        # Audio beat analyzer (always try to initialize for dynamic enable/disable)
        self._audio_beat_analyzer: Optional[AudioBeatAnalyzer] = None
        self._enable_audio_reactive = enable_audio_reactive
        self._audio_analysis_running = False
        self._audio_device = audio_device

        # Always try to initialize audio analyzer for potential runtime enabling
        try:
            self._audio_beat_analyzer = AudioBeatAnalyzer(beat_callback=self._on_beat_detected, device=audio_device)
            logger.info("Audio beat analyzer initialized (ready for dynamic enable/disable)")
        except Exception as e:
            logger.warning(f"Audio beat analyzer unavailable: {e}")
            self._audio_beat_analyzer = None

        # Note: Playlist sync handled by producer, consumer just tracks rendered items
        # WLED and LED buffer will be created after LED optimizer loads pattern file
        # Convert single host to list for uniform handling
        if isinstance(wled_hosts, str):
            self.wled_hosts = [wled_hosts]
        else:
            self.wled_hosts = wled_hosts
        self.wled_port = wled_port
        self._wled_client = None  # Will be created after LED count is known
        self._led_buffer = None  # Will be created after LED count is known

        # Adaptive frame dropping for LED buffer management
        self.enable_adaptive_frame_dropping = enable_adaptive_frame_dropping
        if self.enable_adaptive_frame_dropping:
            self._adaptive_frame_dropper = AdaptiveFrameDropper(
                led_buffer_capacity=10,  # Expected LED buffer capacity
                led_buffer_ewma_alpha=0.03,  # Balanced EWMA - responsive but still smoothing
                max_drop_rate=0.66,  # Maximum drop rate (supports up to 2x input rate)
                use_pid_controller=True,  # Use PID controller for better buffer management
                kp=3.0,  # Proportional gain
                ki=0.5,  # Integral gain for steady-state error elimination
                kd=1.0,  # Derivative gain for oscillation damping
                target_buffer_level=10,  # Maintain buffer at capacity for smooth playback
            )
            logger.info("Adaptive frame dropping enabled with PID controller")
        else:
            self._adaptive_frame_dropper = None
            logger.info("Adaptive frame dropping disabled")

        # Position shifting configuration - set before frame renderer creation
        self.enable_position_shifting = enable_position_shifting
        self.max_shift_distance = max_shift_distance
        self.shift_direction = shift_direction

        # Create frame renderer with LED ordering from pattern file if available
        from ..utils.pattern_loader import create_frame_renderer_with_pattern

        self._frame_renderer = create_frame_renderer_with_pattern(
            diffusion_patterns_path,
            first_frame_delay_ms=0.0,
            timing_tolerance_ms=5.0,
            control_state=self._control_state,
            audio_beat_analyzer=self._audio_beat_analyzer,
            enable_position_shifting=self.enable_position_shifting,
            max_shift_distance=self.max_shift_distance,
            shift_direction=self.shift_direction,
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

        # Track current playlist item for early drop decisions (changes before renderer sees it)
        self._current_optimization_item_index = -1

        # Track late frame dropping suspension for new items
        self._suspend_late_frame_drops = False

        # Track renderer state for pause/resume handling
        self._last_renderer_state = None

        # Track expected frame indices for overall drop rate calculation
        self._last_rendered_frame_index = 0
        self._expected_next_frame_index = 1

        # Timing logger for performance analysis
        self._timing_logger: Optional[FrameTimingLogger] = None
        if timing_log_path:
            self._timing_logger = FrameTimingLogger(timing_log_path)

        # Performance settings
        self.max_frame_wait_timeout = 1.0
        self.brightness_scale = 1.0
        self.optimization_iterations = 5
        self.target_fps = 15.0  # Target processing FPS

        # Frame drop rate tracking with EWMA
        self.pre_optimization_drop_rate_ewma = FrameDropRateEwma(alpha=0.1, name="PreOptimizationDrops")
        self.overall_drop_rate_ewma = FrameDropRateEwma(alpha=0.1, name="OverallDrops")

        # WLED connection management
        self.wled_reconnect_interval = 10.0  # Try to reconnect every 10 seconds
        self._wled_reconnection_thread: Optional[threading.Thread] = None
        self._wled_reconnection_event = threading.Event()  # Signal for graceful shutdown

        # Transition processor for playlist item transitions
        self._transition_processor = TransitionProcessor()

        # LED transition processor for LED-based transitions
        self._led_transition_processor = LEDTransitionProcessor()

        # Thread health monitoring
        self._renderer_thread_heartbeat = 0.0
        self._optimization_thread_heartbeat = 0.0
        self._thread_monitor_interval = 5.0  # Check thread health every 5 seconds
        self._thread_monitor_thread: Optional[threading.Thread] = None
        self._last_thread_check = 0.0

        # Batch processing configuration
        self.enable_batch_mode = enable_batch_mode
        self._frame_batch = []  # Accumulate frames for batch processing
        self._batch_metadata = []  # Metadata for each frame in batch
        self._batch_size = 8  # Target batch size
        self._batch_timeout = 0.5  # Max time to wait for batch completion (seconds)
        self._last_batch_start_time = 0.0  # When current batch started accumulating

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _update_audio_reactive_state(self):
        """Check control state and start/stop audio analysis as needed."""
        if not self._audio_beat_analyzer:
            return  # Audio analyzer not available

        try:
            # Get current control state
            status = self._control_state.get_status()
            should_be_running = status and status.audio_reactive_enabled

            # Start audio analysis if needed
            if should_be_running and not self._audio_analysis_running:
                try:
                    self._audio_beat_analyzer.start_analysis()
                    self._audio_analysis_running = True
                    # Update control state to indicate audio is enabled
                    self._control_state.update_status(audio_enabled=True)
                    logger.info("Audio reactive system started")
                except Exception as e:
                    logger.error(f"Failed to start audio analysis: {e}")

            # Stop audio analysis if needed
            elif not should_be_running and self._audio_analysis_running:
                try:
                    self._audio_beat_analyzer.stop_analysis()
                    self._audio_analysis_running = False
                    # Update control state to indicate audio is disabled
                    self._control_state.update_status(audio_enabled=False)
                    logger.info("Audio reactive system stopped")
                except Exception as e:
                    logger.error(f"Failed to stop audio analysis: {e}")

        except Exception as e:
            logger.warning(f"Error updating audio reactive state: {e}")

    def _on_beat_detected(self, beat_event: BeatEvent):
        """Handle detected beat events and update control state"""
        try:
            # Update control state with beat information
            self._control_state.update_status(
                audio_enabled=self._audio_analysis_running,
                current_bpm=beat_event.bpm,
                beat_count=beat_event.beat_count,
                last_beat_time=beat_event.timestamp,
                beat_confidence=beat_event.confidence,
                audio_intensity=beat_event.intensity,
            )

            # Update downbeat time if this is a downbeat
            if beat_event.is_downbeat:
                self._control_state.update_status(last_downbeat_time=beat_event.timestamp)

            beat_type = "DOWNBEAT" if beat_event.is_downbeat else "BEAT"
            logger.info(
                f"🎵 {beat_type} #{beat_event.beat_count}: BPM={beat_event.bpm:.1f}, "
                f"Intensity={beat_event.intensity:.2f}, "
                f"Confidence={beat_event.confidence:.2f}"
            )

        except Exception as e:
            logger.error(f"Error handling beat event: {e}")

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

            # Set initial renderer state
            self._control_state.set_renderer_state(RendererState.STOPPED)

            # Initialize LED optimizer
            if not self._led_optimizer.initialize():
                logger.error("Failed to initialize LED optimizer")
                return False

            # Now that LED optimizer is loaded, get the actual LED count
            actual_led_count = self._led_optimizer._actual_led_count
            logger.info(f"Using LED count from pattern: {actual_led_count}")

            # Create LED buffer with actual LED count - increased to 20 for video startup latency
            self._led_buffer = LEDBuffer(led_count=actual_led_count, buffer_size=20)

            # Configure WLED client with actual LED count and list of hosts
            wled_config = WLEDSinkConfig(
                led_count=actual_led_count,
                hosts=self.wled_hosts,  # Pass list of hosts
                port=self.wled_port,
            )
            self._wled_client = WLEDSink(wled_config)

            # Try to connect to WLED controller (not required for startup)
            if self._wled_client.connect():
                logger.info("Connected to WLED controller successfully")
            else:
                # WLED sink already logs the specific connection failure with IP address
                pass

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
            self._thread_monitor_thread = threading.Thread(target=self._thread_monitor_loop, name="ThreadMonitorThread")

            self._optimization_thread.start()
            self._renderer_thread.start()
            self._wled_reconnection_thread.start()
            self._thread_monitor_thread.start()

            # Check initial audio reactive state from control state
            self._update_audio_reactive_state()

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

            # Set renderer state to stopped
            self._control_state.set_renderer_state(RendererState.STOPPED)

            # Signal WLED reconnection thread to stop
            self._wled_reconnection_event.set()

            # Stop audio beat analyzer if running
            if self._audio_beat_analyzer and self._audio_analysis_running:
                try:
                    self._audio_beat_analyzer.stop_analysis()
                    self._audio_analysis_running = False
                    logger.info("Audio beat analysis stopped")
                except Exception as e:
                    logger.error(f"Error stopping audio beat analysis: {e}")

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

            if self._thread_monitor_thread and self._thread_monitor_thread.is_alive():
                self._thread_monitor_thread.join(timeout=2.0)
                if self._thread_monitor_thread.is_alive():
                    logger.warning("Thread monitor thread did not stop gracefully")

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
        # In batch mode, expect larger gaps between batch starts
        if self._last_frame_receive_time > 0:
            realtime_gap = receive_time - self._last_frame_receive_time
            # In batch mode, gaps are expected to be larger (up to batch_size * frame_interval)
            expected_batch_gap = self._realtime_gap_threshold * (self._batch_size if self.enable_batch_mode else 1)
            if realtime_gap > expected_batch_gap:
                logger.warning(
                    f"Large real-time gap between frames: {realtime_gap*1000:.1f}ms "
                    f"(previous: {self._last_frame_receive_time:.3f}, current: {receive_time:.3f})"
                    f"{' [batch mode: expected up to ' + str(int(expected_batch_gap*1000)) + 'ms]' if self.enable_batch_mode else ''}"
                )

        # Update tracking variables
        self._last_frame_timestamp = frame_timestamp
        self._last_frame_receive_time = receive_time

    def _optimization_loop(self) -> None:
        """Optimization thread - processes frames as fast as possible."""
        logger.info("Optimization thread started")

        # Track last audio state check to avoid checking too frequently
        last_audio_check = 0.0
        audio_check_interval = 1.0  # Check every 1 second
        last_heartbeat_log = 0.0

        while self._running and not self._shutdown_requested:
            try:
                # Update heartbeat
                self._optimization_thread_heartbeat = time.time()

                # Log heartbeat periodically
                if time.time() - last_heartbeat_log > 30.0:
                    logger.debug("Optimization thread heartbeat")
                    last_heartbeat_log = time.time()
                # Check for shutdown signal from control state
                if self._control_state.should_shutdown():
                    logger.info("Shutdown signal received from control state")
                    break

                # Periodically check for audio reactive state changes
                current_time = time.time()
                if current_time - last_audio_check >= audio_check_interval:
                    self._update_audio_reactive_state()
                    last_audio_check = current_time

                # Wait for new frame
                logger.debug(
                    f"Optimization: About to wait for ready buffer (heartbeat={self._optimization_thread_heartbeat:.1f})"
                )
                buffer_info = self._frame_consumer.wait_for_ready_buffer(timeout=self.max_frame_wait_timeout)
                logger.debug(f"Optimization: wait_for_ready_buffer returned, got buffer_info={buffer_info is not None}")

                if buffer_info is None:
                    # Timeout waiting for frame
                    logger.debug("Optimization: Timeout waiting for frame")
                    continue

                # Process the frame for optimization only
                logger.debug("Optimization: About to process frame optimization")
                self._process_frame_optimization(buffer_info)
                logger.debug("Optimization: process_frame_optimization completed")

            except Exception as e:
                logger.error(f"Error in optimization loop: {e}", exc_info=True)
                time.sleep(0.1)  # Brief pause before retrying

        logger.info("Optimization thread ended")
        # Clear heartbeat to signal thread death
        self._optimization_thread_heartbeat = 0.0

    def _handle_renderer_state_transitions(self, control_status) -> None:
        """
        Handle renderer state transitions based on buffer status and user commands.

        Args:
            control_status: Current system status from control state
        """
        try:
            # Don't trust control_status if it might be from a corrupted read
            # Check if the status looks like a default/error status (all zeros/defaults)
            if (
                control_status.consumer_input_fps == 0.0
                and control_status.renderer_output_fps == 0.0
                and control_status.uptime == 0.0
            ):
                # This looks like a default status from a failed read - don't use it for state transitions
                logger.debug("Skipping state transition check - control status appears to be default/error status")
                return

            current_renderer_state = control_status.renderer_state
            buffer_stats = self._led_buffer.get_buffer_stats()
            buffer_frames = buffer_stats.get("current_count", 0)
            buffer_capacity = buffer_stats.get("buffer_size", 10)

            # State change detection (only log when state actually changes)
            if not hasattr(self, "_last_logged_state"):
                self._last_logged_state = (current_renderer_state, control_status.producer_state, buffer_frames)

            current_state = (current_renderer_state, control_status.producer_state, buffer_frames)
            if current_state[0] != self._last_logged_state[0] or current_state[1] != self._last_logged_state[1]:
                logger.info(
                    f"🔄 Renderer state change: {self._last_logged_state[0]} -> {current_renderer_state}, producer={control_status.producer_state}, buffer={buffer_frames}/{buffer_capacity}"
                )
                self._last_logged_state = current_state

            # Update buffer monitoring in control state
            self._control_state.update_buffer_status(buffer_frames, buffer_capacity)

            # State transition logic
            if current_renderer_state == RendererState.STOPPED:
                # STOPPED state - should only transition via API commands
                pass

            elif current_renderer_state == RendererState.WAITING:
                # WAITING for frames - transition to PLAYING when buffer is full
                if buffer_frames >= buffer_capacity:
                    logger.info(f"🎬 RENDERER WAITING → PLAYING: LED buffer full ({buffer_frames}/{buffer_capacity})")
                    success = self._control_state.set_renderer_state(RendererState.PLAYING)
                    if success:
                        logger.info("✅ Renderer state successfully set to PLAYING")
                    else:
                        logger.error("❌ Failed to set renderer state to PLAYING")

            elif current_renderer_state == RendererState.PLAYING:
                # Check if buffer is empty and producer is stopped
                if buffer_frames == 0 and control_status.producer_state == ProducerState.STOPPED:
                    logger.info("🛑 RENDERER STOPPING: buffer empty and producer stopped")
                    self._control_state.set_renderer_state(RendererState.STOPPED)

            elif current_renderer_state == RendererState.PAUSED:
                # Check if buffer is empty and producer is stopped
                if buffer_frames == 0 and control_status.producer_state == ProducerState.STOPPED:
                    logger.info("🛑 RENDERER STOPPING FROM PAUSE: buffer empty and producer stopped")
                    self._control_state.set_renderer_state(RendererState.STOPPED)

            elif (
                current_renderer_state == RendererState.WAITING
                and control_status.producer_state == ProducerState.STOPPED
            ):
                # If producer stops while we're waiting, go back to stopped
                logger.info("🛑 RENDERER STOPPING FROM WAITING: producer stopped")
                self._control_state.set_renderer_state(RendererState.STOPPED)

        except Exception as e:
            logger.error(f"Error handling renderer state transitions: {e}", exc_info=True)

    def _rendering_loop(self) -> None:
        """Rendering thread - handles timestamp-based frame output."""
        logger.info("Renderer thread started")

        consecutive_errors = 0
        max_consecutive_errors = 10
        last_heartbeat_log = 0.0
        last_frame_render_time = 0.0

        # Log initial renderer state
        initial_status = self._control_state.get_status()
        if initial_status:
            logger.info(f"Initial renderer state: {initial_status.renderer_state}")

        while self._running and not self._shutdown_requested:
            try:
                # Update heartbeat
                self._renderer_thread_heartbeat = time.time()

                # Log heartbeat periodically
                if time.time() - last_heartbeat_log > 30.0:
                    logger.debug(
                        f"Renderer thread heartbeat (last frame {time.time() - last_frame_render_time:.1f}s ago)"
                    )
                    last_heartbeat_log = time.time()
                # Check renderer state and handle transitions
                control_status = self._control_state.get_status()
                if control_status:
                    self._handle_renderer_state_transitions(control_status)

                    # Handle pause/resume detection for frame renderer timing compensation
                    if self._last_renderer_state != control_status.renderer_state:
                        if (
                            control_status.renderer_state == RendererState.PAUSED
                            and self._last_renderer_state != RendererState.PAUSED
                        ):
                            # Transitioning to PAUSED state
                            self._frame_renderer.pause_renderer()
                        elif (
                            self._last_renderer_state == RendererState.PAUSED
                            and control_status.renderer_state != RendererState.PAUSED
                        ):
                            # Transitioning from PAUSED to any other state (resume)
                            self._frame_renderer.resume_renderer()

                        self._last_renderer_state = control_status.renderer_state
                else:
                    logger.warning("Could not get control status for renderer state transitions")

                # Handle renderer state
                if control_status and control_status.renderer_state == RendererState.PLAYING:
                    # Get next LED values with timeout
                    try:
                        logger.debug(
                            f"Renderer: About to read from LED buffer (heartbeat={self._renderer_thread_heartbeat:.1f})"
                        )
                        led_data = self._led_buffer.read_led_values(timeout=0.1)
                        logger.debug(f"Renderer: Read from LED buffer complete, got data={led_data is not None}")
                    except Exception as buffer_error:
                        logger.error(f"Failed to read from LED buffer: {buffer_error}", exc_info=True)
                        consecutive_errors += 1
                        continue

                    if led_data is None:
                        continue  # Timeout - no data available

                elif control_status and control_status.renderer_state == RendererState.PAUSED:
                    # Paused - keep displaying current frame, don't advance buffer
                    time.sleep(0.1)
                    continue

                elif control_status and control_status.renderer_state == RendererState.WAITING:
                    # Waiting for frames - just wait, state transition will handle the switch to PLAYING
                    time.sleep(0.1)
                    continue

                else:
                    # Stopped or invalid state - just wait
                    # Log if we're unexpectedly in STOPPED state with data in buffer
                    if control_status and control_status.renderer_state == RendererState.STOPPED:
                        buffer_stats = self._led_buffer.get_buffer_stats()
                        if buffer_stats.get("current_count", 0) > 0:
                            logger.warning(
                                f"Renderer in STOPPED state but LED buffer has {buffer_stats.get('current_count')} frames - "
                                "possible control state corruption"
                            )
                    time.sleep(0.1)
                    continue

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

                # Track overall frame drops based on missing frame indices
                current_frame_index = None
                if timing_data and hasattr(timing_data, "frame_index"):
                    current_frame_index = timing_data.frame_index

                    # Check for missing frames between last rendered and current
                    if current_frame_index > self._expected_next_frame_index:
                        # Missing frames detected - mark them as dropped in overall tracking
                        missing_count = current_frame_index - self._expected_next_frame_index
                        logger.debug(f"Detected {missing_count} missing frames before frame {current_frame_index}")

                        # Update overall drop rate EWMA for each missing frame
                        for _ in range(missing_count):
                            self.overall_drop_rate_ewma.update(dropped=True)

                    # Update expected next frame index
                    self._expected_next_frame_index = current_frame_index + 1

                should_update_rendering_index = (
                    is_first_frame_of_item
                    and playlist_item_index >= 0
                    and playlist_item_index != self._last_rendered_item_index
                )

                if should_update_rendering_index:
                    logger.info(f"RENDERER: Starting to render playlist item {playlist_item_index}")

                # Debug logging for high FPS investigation
                frame_count = self._stats.frames_processed
                if frame_count % 100 == 0:  # Log every 100th frame
                    buffer_stats = self._led_buffer.get_buffer_stats()
                    current_time = time.time()
                    # Debug logging for high FPS investigation

                # Render with timestamp-based timing
                try:
                    logger.debug(
                        f"Renderer: About to call render_frame_at_timestamp for frame index {current_frame_index}"
                    )
                    success = self._frame_renderer.render_frame_at_timestamp(led_values, timestamp, metadata)
                    logger.debug(f"Renderer: render_frame_at_timestamp returned success={success}")
                    last_frame_render_time = time.time()
                    consecutive_errors = 0  # Reset error counter on success
                except Exception as render_error:
                    logger.error(
                        f"CRITICAL: Frame renderer crashed during render_frame_at_timestamp: {render_error}",
                        exc_info=True,
                    )
                    success = False
                    consecutive_errors += 1

                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical(
                            f"Renderer thread has failed {consecutive_errors} times consecutively - stopping thread"
                        )
                        break

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
                            pass  # Log render-failed frame to timing data

                # Update renderer output FPS tracking and overall drop rate
                if success:
                    # Get output FPS from frame renderer (measured at sink output)
                    self._stats.renderer_output_fps_ewma = self._frame_renderer.get_output_fps()
                    # Mark successful render in overall drop rate tracking
                    if current_frame_index is not None:
                        self.overall_drop_rate_ewma.update(dropped=False)
                        self._last_rendered_frame_index = current_frame_index
                else:
                    # Mark render failure in overall drop rate tracking
                    if current_frame_index is not None:
                        self.overall_drop_rate_ewma.update(dropped=True)

                if not success:
                    logger.warning("Frame rendering failed")

            except Exception as e:
                logger.error(f"Error in rendering loop: {e}", exc_info=True)
                consecutive_errors += 1

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        f"Renderer thread has encountered {consecutive_errors} consecutive errors - exiting thread"
                    )
                    # Set renderer state to STOPPED to signal issue
                    with contextlib.suppress(Exception):
                        self._control_state.set_renderer_state(RendererState.STOPPED)
                    break

                time.sleep(0.01)

        # Thread is exiting - log reason
        if consecutive_errors >= max_consecutive_errors:
            logger.critical("RENDERER THREAD CRASHED: Exiting due to excessive consecutive errors")
        elif self._shutdown_requested:
            logger.info("Renderer thread ended: Shutdown requested")
        elif not self._running:
            logger.info("Renderer thread ended: Consumer stopped")
        else:
            logger.warning("Renderer thread ended: Unknown reason")

        # Clear heartbeat to signal thread death
        self._renderer_thread_heartbeat = 0.0

    def _thread_monitor_loop(self) -> None:
        """Monitor thread health and log warnings if threads appear stuck or dead."""
        logger.info("Thread monitor started")

        while self._running and not self._shutdown_requested:
            try:
                current_time = time.time()

                # Check optimization thread (reduced to 10 seconds for faster detection)
                if self._optimization_thread_heartbeat > 0:
                    opt_age = current_time - self._optimization_thread_heartbeat
                    if opt_age > 10.0:  # No heartbeat for 10 seconds
                        logger.critical(f"OPTIMIZATION THREAD APPEARS STUCK: No heartbeat for {opt_age:.1f} seconds")
                        # Log thread state
                        if self._optimization_thread and self._optimization_thread.is_alive():
                            logger.critical("Optimization thread is_alive=True but not responding")
                        else:
                            logger.critical("Optimization thread is_alive=False - thread has crashed!")

                # Check renderer thread (reduced to 10 seconds for faster detection)
                if self._renderer_thread_heartbeat > 0:
                    render_age = current_time - self._renderer_thread_heartbeat
                    if render_age > 10.0:  # No heartbeat for 10 seconds
                        logger.critical(f"RENDERER THREAD APPEARS STUCK: No heartbeat for {render_age:.1f} seconds")
                        # Log thread state and buffer status
                        if self._renderer_thread and self._renderer_thread.is_alive():
                            logger.critical("Renderer thread is_alive=True but not responding")
                            # Check LED buffer status
                            try:
                                buffer_stats = self._led_buffer.get_buffer_stats()
                                logger.critical(f"LED buffer status: {buffer_stats}")
                            except Exception:
                                pass
                        else:
                            logger.critical("Renderer thread is_alive=False - thread has crashed!")
                            # Try to get more info
                            try:
                                control_status = self._control_state.get_status()
                                if control_status:
                                    logger.critical(
                                        f"System state at crash: renderer_state={control_status.renderer_state}, producer_state={control_status.producer_state}"
                                    )
                            except Exception:
                                pass
                elif self._renderer_thread_heartbeat == 0.0 and current_time - self._last_thread_check > 30.0:
                    # Heartbeat was cleared (thread died)
                    logger.critical("RENDERER THREAD HEARTBEAT CLEARED - Thread has terminated!")
                    if self._renderer_thread and not self._renderer_thread.is_alive():
                        logger.critical("Confirmed: Renderer thread is not alive")

                self._last_thread_check = current_time

                # Sleep for monitor interval
                time.sleep(self._thread_monitor_interval)

            except Exception as e:
                logger.error(f"Error in thread monitor loop: {e}", exc_info=True)
                time.sleep(1.0)

        logger.info("Thread monitor ended")

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

    def _process_frame_optimization(self, buffer_info) -> bool:
        """
        Process frame for LED optimization only - no rendering.
        Rendering handled by separate renderer thread.

        Args:
            buffer_info: Buffer information from frame consumer

        Returns:
            True if frame was dropped, False if frame was processed successfully
        """
        start_time = time.time()

        try:
            # Extract frame and timestamp - move to GPU immediately
            frame_array_cpu = buffer_info.get_array_interleaved(FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS)

            # Move frame to GPU immediately for GPU-native pipeline
            frame_array = cp.asarray(frame_array_cpu, dtype=cp.uint8)

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

            # Adaptive frame dropping based on LED buffer occupancy - only when renderer is PLAYING
            if self.enable_adaptive_frame_dropping and self._adaptive_frame_dropper:
                # Get current renderer state for adaptive dropping
                try:
                    control_status = self._control_state.get_status()
                    renderer_state = control_status.renderer_state.value if control_status else "STOPPED"
                    renderer_is_playing = control_status and control_status.renderer_state == RendererState.PLAYING
                except Exception as e:
                    logger.warning(f"Could not get renderer state for adaptive dropping: {e}")
                    renderer_state = "STOPPED"
                    renderer_is_playing = False

                # Always call adaptive frame dropper, but only apply dropping when renderer is playing
                led_buffer_size = len(self._led_buffer)
                should_drop_adaptive = self._adaptive_frame_dropper.should_drop_frame(
                    timestamp, led_buffer_size, renderer_state
                )

                if renderer_is_playing and should_drop_adaptive:
                    # Drop frame for LED buffer management
                    self._stats.frames_dropped_early += 1
                    self.pre_optimization_drop_rate_ewma.update(dropped=True)

                    # Log dropped frame to timing data if configured
                    if timing_data and self._timing_logger:
                        # Adaptive drop - has frame info and buffer times, but no optimization/render times
                        self._timing_logger.log_frame(timing_data)

                    logger.info(
                        f"📦 FRAME DROPPED (adaptive): Frame {timing_data.frame_index if timing_data else 'unknown'} - LED buffer size: {led_buffer_size}, drop_rate: {self.pre_optimization_drop_rate_ewma.get_rate_percentage():.1f}%"
                    )
                    return True

            # Track current playlist item for early drop decisions (changes before renderer sees it)
            if playlist_item_index >= 0 and playlist_item_index != self._current_optimization_item_index:
                self._current_optimization_item_index = playlist_item_index
                self._suspend_late_frame_drops = False  # Reset suspension for new item
                logger.info(f"CONSUMER: Processing playlist item {playlist_item_index} in optimization")

            # Check if we should drop this frame due to lateness
            frame_is_late = self._frame_renderer.is_frame_late(timestamp, late_threshold_ms=50.0)

            if frame_is_late and not self._suspend_late_frame_drops:
                if is_first_frame_of_item:
                    # First frame of new item is late - suspend late drops until we catch up
                    self._suspend_late_frame_drops = True
                    logger.warning(
                        f"🎬 First frame of new playlist item {playlist_item_index} is late - suspending late frame drops until caught up"
                    )
                else:
                    # Frame already late - drop it
                    self._stats.frames_dropped_early += 1
                    self.pre_optimization_drop_rate_ewma.update(dropped=True)

                    # Log dropped frame to timing data if configured (Case 2: Dropped before optimization)
                    if timing_data and self._timing_logger:
                        # Case 2: Frame dropped before optimization - has frame info and buffer times, but no optimization/render times
                        self._timing_logger.log_frame(timing_data)
                        # Log early-dropped frame to timing data

                    # Enhanced logging for late frame drops
                    current_time = time.time()
                    if self._frame_renderer.is_initialized():
                        target_wallclock = timestamp + self._frame_renderer.get_adjusted_wallclock_delta()
                        lateness_ms = (current_time - target_wallclock) * 1000
                        mode_info = f"batch_size={len(self._frame_batch)}" if self.enable_batch_mode else "single"
                        logger.info(
                            f"⏰ FRAME DROPPED (late): Frame {timing_data.frame_index if timing_data else 'unknown'} - {lateness_ms:.1f}ms late, {mode_info}, drop_rate: {self.pre_optimization_drop_rate_ewma.get_rate_percentage():.1f}%"
                        )
                    else:
                        logger.info(
                            f"⏰ FRAME DROPPED (uninitialized): Frame {timing_data.frame_index if timing_data else 'unknown'} - renderer not ready, drop_rate: {self.pre_optimization_drop_rate_ewma.get_rate_percentage():.1f}%"
                        )

                    return True
            elif not frame_is_late and self._suspend_late_frame_drops:
                # Frame is not late and we were suspending drops - we've caught up
                self._suspend_late_frame_drops = False
                logger.warning("⏰ Caught up with timeline - resuming normal late frame dropping")

            # Validate frame shape
            if frame_array.shape != (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS):
                logger.warning(f"Unexpected frame shape: {frame_array.shape}")
                return True  # Invalid shape counts as dropped

            # Convert RGBA to RGB (on GPU)
            # TODO; Remove this, we only have RGB
            rgb_frame = frame_array[:, :, :3].astype(cp.uint8)

            # Debug: Write first 10 frames to temporary files for analysis (convert to CPU for saving)
            if self._debug_frame_count < self._debug_max_frames:
                try:
                    debug_file = self._debug_frame_dir / f"frame_{self._debug_frame_count:03d}.npy"
                    rgb_frame_cpu = cp.asnumpy(rgb_frame)  # Convert to CPU for saving
                    np.save(debug_file, rgb_frame_cpu)
                    # Write debug frame to file
                    self._debug_frame_count += 1
                except Exception as e:
                    logger.warning(f"DEBUG: Failed to write frame {self._debug_frame_count}: {e}")

            # Apply brightness scaling (on GPU)
            if self.brightness_scale != 1.0:
                rgb_frame = (rgb_frame * self.brightness_scale).clip(0, 255).astype(cp.uint8)

            # Apply playlist item transitions before LED optimization
            transition_start = time.time()
            metadata_dict = self._extract_metadata_dict(buffer_info.metadata)

            # Apply transitions to frame if metadata is available
            rgb_frame = self._transition_processor.process_frame(rgb_frame, metadata_dict)
            transition_time = time.time() - transition_start

            # Handle batch vs single frame processing
            if self.enable_batch_mode and self._led_optimizer.supports_batch_optimization():
                logger.debug("Optimization: Processing frame for batch")
                return self._process_frame_for_batch(rgb_frame, buffer_info, metadata_dict, transition_time, start_time)
            else:
                # Single frame processing (existing logic)
                logger.debug("Optimization: Processing single frame")
                return self._process_single_frame(rgb_frame, buffer_info, metadata_dict, transition_time, start_time)

        except Exception as e:
            logger.error(f"Error in optimization: {e}", exc_info=True)
            self._stats.optimization_errors += 1
            return True  # Processing error counts as dropped

    def _process_frame_for_batch(
        self, rgb_frame: cp.ndarray, buffer_info, metadata_dict: Dict, transition_time: float, start_time: float
    ) -> bool:
        """
        Process frame for batch optimization by accumulating it.

        Returns:
            True if frame was dropped, False if frame was processed successfully
        """
        try:
            # Keep frame on GPU for batch accumulation
            # No conversion needed - rgb_frame is already a cupy array

            # Initialize batch timing if this is the first frame
            if len(self._frame_batch) == 0:
                self._last_batch_start_time = time.time()

            # Extract metadata for this frame
            timestamp = buffer_info.metadata.presentation_timestamp if buffer_info.metadata else start_time
            playlist_item_index = buffer_info.metadata.playlist_item_index if buffer_info.metadata else -1
            is_first_frame_of_item = buffer_info.metadata.is_first_frame_of_item if buffer_info.metadata else False
            timing_data = buffer_info.metadata.timing_data if buffer_info.metadata else None

            # Include all metadata including transition fields
            frame_metadata = {
                "timestamp": timestamp,
                "playlist_item_index": playlist_item_index,
                "is_first_frame_of_item": is_first_frame_of_item,
                "timing_data": timing_data,
                "transition_time": transition_time,
                "optimization_time": 0.0,  # Will be filled in batch processing
                **metadata_dict,  # Include all transition metadata fields
            }

            # Add frame to batch - keep on GPU
            self._frame_batch.append(rgb_frame)  # Keep as cupy array
            self._batch_metadata.append(frame_metadata)

            # Debug log to confirm transition metadata is in batch
            if (
                frame_metadata.get("transition_in_type") != "none"
                or frame_metadata.get("transition_out_type") != "none"
            ):
                logger.debug(
                    f"Batch frame {len(self._frame_batch)}: transition metadata included - "
                    f"in_type={frame_metadata.get('transition_in_type')}, "
                    f"out_type={frame_metadata.get('transition_out_type')}"
                )

            # Check if batch should be processed
            if self._should_process_batch():
                logger.debug(f"Optimization: Processing batch of {len(self._frame_batch)} frames")
                batch_success = self._process_frame_batch()
                logger.debug(f"Optimization: Batch processing completed, success={batch_success}")
                if not batch_success:
                    logger.warning("Batch processing failed")

                # Process successful - reset batch accumulation
                # Statistics are updated in _process_frame_batch
                return False  # Frame was processed (not dropped)

            # Frame accumulated for batch - not processed yet
            logger.debug(f"Frame accumulated for batch processing ({len(self._frame_batch)}/{self._batch_size})")
            return False  # Frame not dropped, just batched

        except Exception as e:
            logger.error(f"Error in batch frame processing: {e}", exc_info=True)
            return True  # Error counts as dropped

    def _process_single_frame(
        self, rgb_frame: cp.ndarray, buffer_info, metadata_dict: Dict, transition_time: float, start_time: float
    ) -> bool:
        """
        Process single frame using traditional single-frame optimizer.

        Returns:
            True if frame was dropped, False if frame was processed successfully
        """
        try:
            # Optimize LED values (no timing constraints)
            optimization_start = time.time()

            # Get current optimization iterations from control state, fallback to instance variable
            iterations = self.optimization_iterations
            try:
                if self._control_state:
                    control_status = self._control_state.get_status()
                    if control_status and hasattr(control_status, "optimization_iterations"):
                        iterations = control_status.optimization_iterations
                        # Update instance variable for consistency
                        if iterations != self.optimization_iterations:
                            self.optimization_iterations = iterations
                            # Update optimization iterations from ControlState
            except Exception as e:
                pass  # Failed to read optimization iterations from ControlState

            # Move to CPU for single frame optimizer if needed
            if isinstance(rgb_frame, cp.ndarray):
                rgb_frame_cpu = cp.asnumpy(rgb_frame)
            else:
                rgb_frame_cpu = rgb_frame

            result = self._led_optimizer.optimize_frame(rgb_frame_cpu, max_iterations=iterations)
            optimization_time = time.time() - optimization_start

            # Extract metadata for this frame
            timestamp = buffer_info.metadata.presentation_timestamp if buffer_info.metadata else start_time
            playlist_item_index = buffer_info.metadata.playlist_item_index if buffer_info.metadata else -1
            is_first_frame_of_item = buffer_info.metadata.is_first_frame_of_item if buffer_info.metadata else False
            timing_data = buffer_info.metadata.timing_data if buffer_info.metadata else None

            # Check optimization result
            if not result.converged:
                self._stats.optimization_errors += 1

            # Apply LED transitions to optimized LED values (on GPU)
            led_transition_start = time.time()
            if isinstance(result.led_values, cp.ndarray):
                # LED values are on GPU, apply LED transitions
                led_values_gpu = self._led_transition_processor.process_led_values(result.led_values, metadata_dict)
                # Convert to CPU for LED buffer
                led_values_uint8 = cp.asnumpy(led_values_gpu).astype(np.uint8)
            else:
                # LED values already on CPU, convert to GPU for LED transitions then back
                led_values_gpu = cp.asarray(result.led_values)
                led_values_with_transitions = self._led_transition_processor.process_led_values(
                    led_values_gpu, metadata_dict
                )
                led_values_uint8 = cp.asnumpy(led_values_with_transitions).astype(np.uint8)
            led_transition_time = time.time() - led_transition_start

            # Mark LED transition completion time for timing data
            if timing_data:
                timing_data.mark_led_transition_complete()

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
            self._stats.total_led_transition_time += led_transition_time
            self._stats.last_frame_time = start_time

            # Update optimization FPS (independent of rendering)
            self._stats.current_optimization_fps = 1.0 / total_time if total_time > 0 else 0.0

            # Update pre-optimization drop rate EWMA for successful processing
            self.pre_optimization_drop_rate_ewma.update(dropped=False)

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
                    f"pre-opt drop rate: {self.pre_optimization_drop_rate_ewma.get_rate_percentage():.1f}%, "
                    f"opt time: {avg_opt_time * 1000:.1f}ms, "
                    f"LED buffer depth: {buffer_depth}"
                )
                self._last_consumer_log_time = current_time

                # Update consumer statistics in ControlState for IPC with web server
                self._update_consumer_statistics_in_control_state()

            # Frame was processed successfully
            return False

        except Exception as e:
            logger.error(f"Error in single frame processing: {e}", exc_info=True)
            self._stats.optimization_errors += 1
            return True  # Processing error counts as dropped

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
            missing_fields = []

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
                    missing_fields.append(field)

            # Log missing fields only in debug mode
            if missing_fields and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using default values for missing transition fields: {', '.join(missing_fields)}")

            # Log transition metadata received from producer
            if any(
                metadata_dict.get(f) != "none" and metadata_dict.get(f) != 0.0
                for f in ["transition_in_type", "transition_out_type"]
            ):
                logger.info(
                    f"CONSUMER: Received transition metadata - "
                    f"item_timestamp={metadata_dict.get('item_timestamp', 0.0):.3f}s, "
                    f"item_duration={metadata_dict.get('item_duration', 0.0):.3f}s, "
                    f"in_type='{metadata_dict.get('transition_in_type', 'none')}' "
                    f"(duration={metadata_dict.get('transition_in_duration', 0.0):.3f}s), "
                    f"out_type='{metadata_dict.get('transition_out_type', 'none')}' "
                    f"(duration={metadata_dict.get('transition_out_duration', 0.0):.3f}s)"
                )

            return metadata_dict

        except Exception as e:
            logger.warning(f"Error extracting metadata for transitions: {e}")
            return {}

    def _should_process_batch(self) -> bool:
        """
        Check if we should process the current batch.

        Returns:
            True if batch is ready for processing
        """
        if not self.enable_batch_mode:
            return False

        # Batch is full
        if len(self._frame_batch) >= self._batch_size:
            return True

        # Batch has frames and timeout reached
        if len(self._frame_batch) > 0:
            current_time = time.time()
            if current_time - self._last_batch_start_time >= self._batch_timeout:
                return True

        return False

    def _clear_batch(self) -> None:
        """Clear the current frame batch."""
        self._frame_batch.clear()
        self._batch_metadata.clear()
        self._last_batch_start_time = 0.0

    def _process_frame_batch(self) -> bool:
        """
        Process accumulated frame batch using batch optimizer.

        Returns:
            True if batch was processed successfully
        """
        if len(self._frame_batch) == 0:
            return True

        try:
            # Check if LED optimizer supports batch mode
            if not self._led_optimizer.supports_batch_optimization():
                logger.warning(
                    "LED optimizer does not support batch optimization - falling back to single frame processing"
                )
                return self._process_batch_frames_individually()

            # Pad batch to target size if needed
            while len(self._frame_batch) < self._batch_size:
                # Duplicate last frame to fill batch (copy on GPU)
                if self._frame_batch:
                    self._frame_batch.append(self._frame_batch[-1].copy())  # cupy array copy
                    self._batch_metadata.append(self._batch_metadata[-1])
                else:
                    break

            logger.debug(f"Processing batch of {len(self._frame_batch)} frames")

            # Convert frames to batch format (batch_size, 3, H, W) on GPU
            # Frames are already cupy arrays - just stack them
            batch_frames = cp.stack(self._frame_batch[: self._batch_size], axis=0)  # (8, H, W, 3)
            batch_frames_gpu = batch_frames.transpose(0, 3, 1, 2)  # (8, 3, H, W)

            # Get current optimization iterations
            iterations = self.optimization_iterations
            try:
                control_status = self._control_state.get_status()
                if control_status and hasattr(control_status, "optimization_iterations"):
                    iterations = control_status.optimization_iterations
            except Exception:
                pass

            # Process batch
            batch_result = self._led_optimizer.optimize_batch_frames(
                batch_frames_gpu, max_iterations=iterations, debug=False
            )

            # Track LED transition time for batch
            batch_led_transition_time = 0.0

            # Write each result to LED buffer
            success_count = 0
            for frame_idx in range(len(self._frame_batch)):
                if frame_idx >= batch_result.led_values.shape[0]:
                    break

                # Extract LED values for this frame
                frame_led_values = batch_result.led_values[frame_idx]  # (3, led_count)

                # Apply LED transitions to this frame's LED values (GPU processing)
                metadata = self._batch_metadata[frame_idx]
                frame_led_values_transposed = (
                    frame_led_values.T
                )  # Convert to (led_count, 3) for LED transition processor

                led_transition_start = time.time()
                frame_led_values_with_transitions = self._led_transition_processor.process_led_values(
                    frame_led_values_transposed, metadata
                )
                batch_led_transition_time += time.time() - led_transition_start

                # Convert to CPU for LED buffer
                frame_led_values_cpu = cp.asnumpy(frame_led_values_with_transitions).astype(np.uint8)  # (led_count, 3)

                timestamp = metadata.get("timestamp", time.time())

                # Write to LED buffer
                success = self._led_buffer.write_led_values(
                    frame_led_values_cpu,
                    timestamp,
                    {
                        "batch_optimization": True,
                        "batch_index": frame_idx,
                        "batch_size": len(self._frame_batch),
                        "optimization_time": metadata.get("optimization_time", 0.0),
                        "converged": batch_result.converged,
                        "iterations": batch_result.iterations,
                        "error_metrics": (
                            batch_result.error_metrics[frame_idx] if frame_idx < len(batch_result.error_metrics) else {}
                        ),
                        **metadata,  # Include original metadata
                    },
                    block=True,
                    timeout=0.1,
                )

                if success:
                    success_count += 1

            # Update statistics
            self._stats.frames_processed += success_count

            # Note: Renderer output FPS is updated in the renderer thread when frames are actually rendered,
            # not here in the optimization thread. Batch mode doesn't change this.

            # Periodic logging for pipeline debugging (every 2 seconds) - same as single frame mode
            current_time = time.time()
            if current_time - self._last_consumer_log_time >= self._consumer_log_interval:
                content_ratio = (self._frames_with_content / max(1, self._stats.frames_processed)) * 100
                avg_fps = self._stats.get_average_fps()
                avg_opt_time = self._stats.get_average_optimization_time()

                # Get LED buffer stats
                buffer_stats = self._led_buffer.get_buffer_stats()
                buffer_depth = buffer_stats["current_count"]

                logger.info(
                    f"CONSUMER PIPELINE (BATCH): {self._stats.frames_processed} frames optimized, "
                    f"{self._stats.frames_dropped_early} dropped early, "
                    f"{self._frames_with_content} with LED content ({content_ratio:.1f}%), "
                    f"input FPS: {self._stats.consumer_input_fps:.1f}, "
                    f"output FPS: {self._stats.renderer_output_fps_ewma:.1f}, "
                    f"pre-opt drop rate: {self.pre_optimization_drop_rate_ewma.get_rate_percentage():.1f}%, "
                    f"LED buffer depth: {buffer_depth}, batch_size: {len(self._frame_batch)}"
                )
                self._last_consumer_log_time = current_time

                # Update consumer statistics in ControlState for IPC with web server
                self._update_consumer_statistics_in_control_state()

            # Update LED transition statistics
            self._stats.total_led_transition_time += batch_led_transition_time

            logger.debug(
                f"Batch processing completed: {success_count}/{len(self._frame_batch)} frames successful, LED transitions took {batch_led_transition_time:.3f}s"
            )
            return success_count > 0

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            return self._process_batch_frames_individually()
        finally:
            self._clear_batch()

    def _process_batch_frames_individually(self) -> bool:
        """
        Fallback: Process batch frames individually using single frame optimizer.

        Returns:
            True if at least one frame was processed successfully
        """
        success_count = 0

        for frame_idx, (frame, metadata) in enumerate(zip(self._frame_batch, self._batch_metadata)):
            try:
                # Convert GPU frame to CPU for single frame optimizer
                if isinstance(frame, cp.ndarray):
                    frame_cpu = cp.asnumpy(frame)
                else:
                    frame_cpu = frame

                # Process single frame using existing logic
                result = self._led_optimizer.optimize_frame(frame_cpu, max_iterations=self.optimization_iterations)

                # Convert to uint8 if needed
                if isinstance(result.led_values, cp.ndarray):
                    led_values_cpu = cp.asnumpy(result.led_values).astype(np.uint8)
                else:
                    led_values_cpu = result.led_values.astype(np.uint8)

                timestamp = metadata.get("timestamp", time.time())

                # Write to LED buffer
                success = self._led_buffer.write_led_values(
                    led_values_cpu,
                    timestamp,
                    {
                        "batch_fallback": True,
                        "optimization_time": metadata.get("optimization_time", 0.0),
                        "converged": result.converged,
                        "iterations": result.iterations,
                        "error_metrics": result.error_metrics,
                        **metadata,
                    },
                    block=True,
                    timeout=0.1,
                )

                if success:
                    success_count += 1

            except Exception as e:
                logger.error(f"Error processing individual frame {frame_idx} from batch: {e}")

        # Update statistics
        self._stats.frames_processed += success_count

        logger.debug(f"Batch fallback processing: {success_count}/{len(self._frame_batch)} frames successful")
        self._clear_batch()
        return success_count > 0

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
            actual_led_count = self._led_optimizer._actual_led_count
            self._preview_sink = PreviewSink(led_count=actual_led_count, config=self._preview_sink_config)

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
            "dropped_frames_percentage": self.overall_drop_rate_ewma.get_rate_percentage(),  # Overall rate for compatibility
            "pre_optimization_dropped_frames_percentage": self.pre_optimization_drop_rate_ewma.get_rate_percentage(),
            "overall_dropped_frames_percentage": self.overall_drop_rate_ewma.get_rate_percentage(),
            "total_processing_time": self._stats.total_processing_time,
            "average_optimization_time": self._stats.get_average_optimization_time(),
            "average_led_transition_time": self._stats.get_average_led_transition_time(),
            "optimization_errors": self._stats.optimization_errors,
            "transmission_errors": self._stats.transmission_errors,
            "optimization_iterations": self.optimization_iterations,
            "brightness_scale": self.brightness_scale,
            "target_fps": self.target_fps,
            "led_count": self._led_optimizer._actual_led_count,
            "frame_dimensions": (FRAME_WIDTH, FRAME_HEIGHT),
            "wled_stats": self._wled_client.get_statistics(),
            "optimizer_stats": self._led_optimizer.get_optimizer_stats(),
            "test_renderer_enabled": self.enable_test_renderer,
            "test_renderer_stats": (self._test_renderer.get_statistics() if self._test_renderer else None),
            "led_buffer_stats": self._led_buffer.get_buffer_stats(),
            "led_transition_stats": self._led_transition_processor.get_statistics(),
            "renderer_stats": self._frame_renderer.get_renderer_stats(),
            "adaptive_frame_dropper_enabled": self.enable_adaptive_frame_dropping,
            "adaptive_frame_dropper_stats": (
                self._adaptive_frame_dropper.get_stats() if self._adaptive_frame_dropper else None
            ),
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
            # Gap detection for missing frames

            if self._last_frame_index_seen == 0:
                # Very first frame processed
                # First frame processing
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
                            # Log missing frame to timing data

                old_value = self._last_frame_index_seen
                self._last_frame_index_seen = current_frame_index
                # Update last seen frame index
                return

            # Check for gaps in sequence
            expected_next = self._last_frame_index_seen + 1
            if current_frame_index == expected_next:
                # Normal sequence - no gap
                pass
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
                        # Log missing frame to timing data
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
            # Get renderer statistics for renderer-specific late frames
            renderer_stats = self._frame_renderer.get_renderer_stats()
            renderer_late_frame_percentage = renderer_stats.get("late_frame_percentage", 0.0)

            # Calculate comprehensive late frame percentage including:
            # 1. Pre-optimization drops (frames dropped before optimization due to lateness/buffer)
            # 2. Renderer late frames (frames that reached renderer but were late)
            pre_opt_drop_percentage = self.pre_optimization_drop_rate_ewma.get_rate_percentage()
            comprehensive_late_frame_percentage = pre_opt_drop_percentage + renderer_late_frame_percentage

            # Update the control state with consumer statistics
            # Note: ControlState only has one dropped_frames_percentage field, so we use overall rate
            status_updates = {
                "consumer_input_fps": self._stats.consumer_input_fps,
                "renderer_output_fps": self._stats.renderer_output_fps_ewma,
                "dropped_frames_percentage": self.overall_drop_rate_ewma.get_rate_percentage(),
                "late_frame_percentage": comprehensive_late_frame_percentage,
            }

            # Log FPS values being written to control state
            logger.debug(
                f"CONSUMER FPS DEBUG: Writing to ControlState - input_fps={self._stats.consumer_input_fps:.2f}, output_fps={self._stats.renderer_output_fps_ewma:.2f}"
            )

            # Update ControlState with new statistics
            self._control_state.update_status(**status_updates)

            # Verify the values were written correctly
            verification = self._control_state.get_status_dict()
            logger.debug(
                f"CONSUMER FPS DEBUG: Verification read from ControlState - input_fps={verification.get('consumer_input_fps', 'MISSING'):.2f}, output_fps={verification.get('renderer_output_fps', 'MISSING'):.2f}"
            )

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
