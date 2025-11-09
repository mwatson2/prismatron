"""
Timestamp-Based Frame Renderer.

This module implements precise timestamp-based rendering for LED displays.
Handles wallclock timing establishment, late/early frame logic, and output
to multiple targets (WLED, test renderer).
"""

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .audio_reactive_position_shifter import AudioReactivePositionShifter
from .led_effect import LedEffectManager
from .test_sink import TestSink
from .wled_sink import WLEDSink

logger = logging.getLogger(__name__)


class FrameRenderer:
    """
    Timestamp-based frame renderer that handles precise timing for LED display.

    Features:
    - Establishes wallclock delta from first frame timestamp
    - Renders frames at their designated timestamps
    - Handles late/early frame timing
    - Supports multiple output targets (WLED, test renderer)
    - Converts LED values from spatial to physical order before output
    """

    def __init__(
        self,
        led_ordering: np.ndarray,
        first_frame_delay_ms: float = 100.0,
        timing_tolerance_ms: float = 5.0,
        late_frame_log_threshold_ms: float = 50.0,
        control_state=None,
        audio_beat_analyzer=None,
        enable_position_shifting: bool = False,
        max_shift_distance: int = 3,
        shift_direction: str = "alternating",
    ):
        """
        Initialize frame renderer.

        Args:
            first_frame_delay_ms: Default delay for first frame buffering
            timing_tolerance_ms: Acceptable timing deviation
            late_frame_log_threshold_ms: Log late frames above this threshold
            led_ordering: Array mapping spatial indices to physical LED IDs
            control_state: ControlState instance for audio reactive settings
            audio_beat_analyzer: AudioBeatAnalyzer instance for beat state access
            enable_position_shifting: Enable audio-reactive position shifting effects
            max_shift_distance: Maximum LED positions to shift (3-4 typical)
            shift_direction: Shift direction ("left", "right", "alternating")
        """
        self.first_frame_delay = first_frame_delay_ms / 1000.0
        self.timing_tolerance = timing_tolerance_ms / 1000.0
        self.late_frame_log_threshold = late_frame_log_threshold_ms / 1000.0

        # LED ordering for spatial to physical conversion
        self.led_ordering = led_ordering

        # Audio reactive components
        self._control_state = control_state
        self._audio_beat_analyzer = audio_beat_analyzer

        # Position shifter for audio-reactive effects
        self._position_shifter = AudioReactivePositionShifter(
            max_shift_distance=max_shift_distance,
            shift_direction=shift_direction,
            control_state=control_state,
            audio_beat_analyzer=audio_beat_analyzer,
            enabled=enable_position_shifting,
        )

        # Timing state
        self.wallclock_delta = None  # Established from first frame
        self.first_frame_received = False
        self.first_frame_timestamp = None

        # Pause time tracking
        self.is_paused = False
        self.pause_start_time = None
        self.total_pause_time = 0.0

        # Output sinks (multiple sink support)
        self.sinks = []  # List of registered sinks
        self.sink_names = {}  # Map sink instances to names for logging

        # Legacy compatibility - maintain individual references
        self.wled_sink: Optional[WLEDSink] = None
        self.test_sink: Optional[TestSink] = None
        self.enable_wled = True
        self.enable_test_sink = False

        # Statistics
        self.frames_rendered = 0
        self.late_frames = 0
        self.early_frames = 0
        self.on_time_frames = 0
        self.dropped_frames = 0  # For future frame dropping policy
        self.total_wait_time = 0.0
        self.total_late_time = 0.0
        self.start_time = time.time()

        # Beat effect statistics for periodic reporting
        self.beat_boost_count = 0
        self.beat_boost_sum = 0.0
        self.beat_shift_count = 0
        self.beat_shift_sum = 0.0
        self.last_beat_stats_log = time.time()
        self.beat_stats_log_interval = 30.0  # Log every 30 seconds

        # EWMA statistics for recent performance tracking
        self.ewma_alpha = 0.1  # EWMA smoothing factor
        self.ewma_fps = 0.0
        self.ewma_late_fraction = 0.0
        self.ewma_dropped_fraction = 0.0
        self.last_ewma_update = 0.0
        self.last_frame_timestamp = 0.0  # Last frame timestamp for interval calculation
        self.large_timestamp_gap_threshold = 2.0  # Log gaps larger than 2 seconds

        # Output FPS tracking (frames sent to sinks per second)
        self.output_fps_ewma = 0.0
        self._last_sink_call_time = 0.0  # Track sink call completion time for FPS calculation

        # Timing distribution tracking
        self.timing_errors = []  # Track last 100 timing errors for analysis
        self.max_timing_history = 100

        # Debug LED value writing (first 10 different frames)
        self._debug_led_count = 0
        self._debug_max_leds = 0
        self._debug_led_dir = Path("/tmp/prismatron_debug_leds")
        self._debug_led_dir.mkdir(exist_ok=True)
        self._debug_previous_led_values = None  # Track previous frame for uniqueness

        # Track error message timing to silence after 1 minute
        self._error_message_start_time = time.time()
        self._silent_after_minutes = 1.0

        # Track first frame timestamp for current playlist item (for playback position)
        self.current_item_first_frame_timestamp = None
        self.current_rendering_index = -1

        # Track if we need to adjust timeline for a new playlist item
        self._pending_timeline_adjustment = False

        # Beat boost logging state
        self._beat_boost_logged = False

        # LED effects manager
        self.effect_manager = LedEffectManager()

        logger.info(
            f"FrameRenderer initialized: delay={first_frame_delay_ms}ms, " f"tolerance=±{timing_tolerance_ms}ms"
        )

    def _calculate_beat_brightness_boost(self, current_time: float) -> float:
        """
        Calculate brightness boost based on beat timing for audio-reactive effects.

        NOTE: This is the legacy inline implementation. Consider migrating to the new
        BeatBrightnessEffect class which creates effect instances per beat:

            from .led_effect import BeatBrightnessEffect

            # When beat detected, create effect using beat timestamp from audio timeline
            # (converted to frame timeline)
            effect = BeatBrightnessEffect(
                start_time=beat_frame_timestamp,  # Beat time on frame timeline
                bpm=beat_state.current_bpm,
                beat_intensity=beat_state.beat_intensity,
                boost_intensity=4.0,
                duration_fraction=0.4
            )
            renderer.add_led_effect(effect)

        Implements a configurable sine wave brightness boost during a portion of each beat interval.
        Formula: 1.0 + intensity * sin(t * pi / (duration * d)) where:
        - t = time since beat start
        - d = inter-beat duration (60.0 / BPM)
        - intensity = configurable boost intensity (0.0 to 1.0)
        - duration = configurable fraction of beat duration (0.1 to 1.0)

        Args:
            current_time: Current system time in seconds

        Returns:
            Brightness multiplier (1.0 = no boost, up to 2.0 = 100% boost)
        """
        # Check if audio reactive effects are enabled
        if not self._control_state or not self._audio_beat_analyzer:
            logger.debug("Beat brightness boost: No control state or audio analyzer")
            return 1.0

        try:
            # Get current control state to check if audio reactive is enabled
            status = self._control_state.get_status()
            if not status:
                logger.debug("Beat brightness boost: No status available")
                return 1.0

            if not status.audio_reactive_enabled:
                logger.debug("Beat brightness boost: Audio reactive not enabled")
                return 1.0

            if not status.audio_enabled:
                logger.debug("Beat brightness boost: Audio not enabled")
                return 1.0

            # Check if beat brightness boost is specifically enabled
            if not status.beat_brightness_enabled:
                logger.debug("Beat brightness boost: Beat brightness not enabled")
                return 1.0

            # Get configurable parameters with fallbacks
            boost_intensity = getattr(status, "beat_brightness_intensity", 4.0)  # Strong brightness boost (max 5.0)
            boost_duration_fraction = getattr(status, "beat_brightness_duration", 0.4)  # 400ms at 60 BPM
            confidence_threshold = getattr(status, "beat_confidence_threshold", 0.5)  # Ignore weak beats

            # Clamp parameters to safe ranges (intensity can now go up to 5x)
            boost_intensity = max(0.0, min(5.0, boost_intensity))
            boost_duration_fraction = max(0.1, min(1.0, boost_duration_fraction))
            confidence_threshold = max(0.0, min(1.0, confidence_threshold))

            # Log beat boost configuration once
            if not self._beat_boost_logged:
                logger.info(
                    f"🎵 Beat brightness boost enabled: intensity={boost_intensity:.2f}, "
                    f"duration={boost_duration_fraction:.2f}, confidence_threshold={confidence_threshold:.2f}"
                )
                self._beat_boost_logged = True

            # Get audio beat state
            beat_state = self._audio_beat_analyzer.get_current_state()
            if not beat_state or not beat_state.is_active:
                return 1.0

            # Calculate inter-beat duration from current BPM
            if beat_state.current_bpm <= 0:
                return 1.0
            beat_duration = 60.0 / beat_state.current_bpm

            # Calculate audio timeline time
            audio_time = current_time - self._audio_beat_analyzer.start_time

            # Get the most recent beat time from beat_state
            last_beat_time = beat_state.last_beat_time

            # Calculate time since the most recent beat
            t = audio_time - last_beat_time

            # Log beat state age and phase (DEBUG level)
            beat_state_age_ms = (current_time - self._audio_beat_analyzer.start_time - last_beat_time) * 1000
            beat_phase = (t / beat_duration) if beat_duration > 0 else 0
            logger.debug(
                f"Beat state read: BPM={beat_state.current_bpm:.1f}, "
                f"state_age={beat_state_age_ms:.1f}ms, "
                f"beat_phase={beat_phase:.2f}, "
                f"t={t:.3f}s"
            )

            # If we're past one full beat interval, we need to find which beat we're in
            if t > beat_duration:
                # Calculate how many beats have passed since last detected beat
                beats_passed = int(t / beat_duration)
                # Find the start of the current beat interval
                current_beat_start = last_beat_time + (beats_passed * beat_duration)
                # Recalculate t as time since current beat start
                t = audio_time - current_beat_start

            # Apply sine wave boost for configured duration of beat interval
            boost_duration = boost_duration_fraction * beat_duration
            if 0 <= t <= boost_duration:
                # Get beat intensity and confidence for dynamic boost
                # Use the beat_intensity and confidence from the beat state
                beat_intensity_value = getattr(beat_state, "beat_intensity", 1.0)
                beat_confidence = getattr(beat_state, "confidence", 1.0)

                # Apply confidence threshold - ignore weak beats
                if beat_confidence < confidence_threshold:
                    logger.debug(
                        f"Beat ignored: confidence {beat_confidence:.2f} < threshold {confidence_threshold:.2f}"
                    )
                    return 1.0

                # Calculate dynamic boost with improved intensity scaling
                # Use sqrt to expand the intensity range (0.1 -> 0.32, 0.4 -> 0.63)
                intensity_scaled = math.sqrt(beat_intensity_value)
                # Boost is: base * scaled_intensity * sine_wave (confidence already filtered)
                dynamic_boost_factor = boost_intensity * intensity_scaled
                boost = dynamic_boost_factor * math.sin(t * math.pi / boost_duration)
                multiplier = 1.0 + boost

                # Log boost calculation at DEBUG level
                logger.debug(
                    f"BRIGHTNESS_BOOST: multiplier={multiplier:.3f}, t={t:.3f}s, "
                    f"boost_intensity={boost_intensity:.2f}, intensity_raw={beat_intensity_value:.2f}, "
                    f"intensity_scaled={intensity_scaled:.3f}, confidence={beat_confidence:.2f}, "
                    f"dynamic_factor={dynamic_boost_factor:.3f}, boost={boost:.3f}"
                )
                return multiplier
            else:
                if np.random.random() < 0.01:  # Log 1% of non-boost events to reduce spam
                    logger.debug(
                        f"🎵 Beat brightness boost: No boost (t={t:.3f}s > duration={boost_duration:.3f}s), BPM={beat_state.current_bpm:.1f}"
                    )
                return 1.0  # No boost outside beat window

        except Exception as e:
            logger.warning(f"Error calculating beat brightness boost: {e}")
            return 1.0

    def register_sink(self, sink, name: str, enabled: bool = True) -> None:
        """
        Register a new output sink.

        Args:
            sink: Sink instance that must have a method to receive LED data
            name: Human-readable name for the sink
            enabled: Whether the sink is initially enabled
        """
        if hasattr(sink, "send_led_data") or hasattr(sink, "render_led_values"):
            self.sinks.append({"sink": sink, "name": name, "enabled": enabled, "failing": False})
            self.sink_names[sink] = name
            logger.info(f"Registered sink: {name} (enabled={enabled})")
        else:
            raise ValueError(f"Sink {name} must have 'send_led_data' or 'render_led_values' method")

    def unregister_sink(self, sink) -> None:
        """
        Unregister an output sink.

        Args:
            sink: Sink instance to remove
        """
        name = self.sink_names.get(sink, "Unknown")
        self.sinks = [s for s in self.sinks if s["sink"] != sink]
        if sink in self.sink_names:
            del self.sink_names[sink]
        logger.info(f"Unregistered sink: {name}")

    def set_sink_enabled(self, sink, enabled: bool) -> None:
        """
        Enable or disable a specific sink.

        Args:
            sink: Sink instance
            enabled: Whether to enable the sink
        """
        for s in self.sinks:
            if s["sink"] == sink:
                s["enabled"] = enabled
                name = self.sink_names.get(sink, "Unknown")
                logger.info(f"Set sink {name} enabled={enabled}")
                break

    def set_output_targets(
        self, wled_sink: Optional[WLEDSink] = None, test_sink: Optional[TestSink] = None, preview_sink: Optional = None
    ) -> None:
        """
        Set output targets for rendering (legacy compatibility method).

        Args:
            wled_sink: WLED sink for LED output
            test_sink: Test sink for debugging
            preview_sink: Preview sink for web interface
        """
        # Clear existing sinks
        self.sinks.clear()
        self.sink_names.clear()

        # Register provided sinks
        if wled_sink is not None:
            self.register_sink(wled_sink, "WLED", enabled=True)
        if test_sink is not None:
            self.register_sink(test_sink, "TestSink", enabled=False)
        if preview_sink is not None:
            self.register_sink(preview_sink, "PreviewSink", enabled=True)

        # Maintain legacy references
        self.wled_sink = wled_sink
        self.test_sink = test_sink
        self.enable_wled = wled_sink is not None
        self.enable_test_sink = test_sink is not None

        logger.info(
            f"Output targets: WLED={self.enable_wled}, TestSink={self.enable_test_sink}, Preview={preview_sink is not None}"
        )

    def set_wled_enabled(self, enabled: bool) -> None:
        """Enable or disable WLED output (legacy compatibility)."""
        self.enable_wled = enabled and (self.wled_sink is not None)
        if self.wled_sink is not None:
            self.set_sink_enabled(self.wled_sink, enabled)

    def set_test_sink_enabled(self, enabled: bool) -> None:
        """Enable or disable test sink output (legacy compatibility)."""
        self.enable_test_sink = enabled and (self.test_sink is not None)
        if self.test_sink is not None:
            self.set_sink_enabled(self.test_sink, enabled)

    def establish_wallclock_delta(self, first_timestamp: float) -> None:
        """
        Establish fixed delta between frame timestamps and wallclock time.

        Args:
            first_timestamp: Presentation timestamp of first frame
        """
        if self.first_frame_received:
            logger.warning("Wallclock delta already established")
            return

        current_wallclock = time.time()

        # Add default delay for buffering
        target_wallclock = current_wallclock + self.first_frame_delay

        # Calculate delta: wallclock_time = frame_timestamp + delta
        self.wallclock_delta = target_wallclock - first_timestamp
        self.first_frame_timestamp = first_timestamp
        self.first_frame_received = True

        logger.info(
            f"Established wallclock delta: {self.wallclock_delta:.3f}s "
            f"(first frame delay: {self.first_frame_delay:.3f}s)"
        )

    def render_frame_at_timestamp(
        self, led_values: np.ndarray, frame_timestamp: float, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Render frame at its designated timestamp with timing logic.

        Args:
            led_values: Optimized LED values to display, shape (led_count, 3)
            frame_timestamp: Original presentation timestamp from producer
            metadata: Optional frame metadata

        Returns:
            True if rendered successfully, False otherwise
        """
        # Use the metadata from producer to check if this is the first frame of a new item
        is_first_frame_of_new_item = metadata.get("is_first_frame_of_item", False) if metadata else False
        playlist_item_index = metadata.get("playlist_item_index", -1) if metadata else -1

        if is_first_frame_of_new_item and playlist_item_index >= 0:
            logger.info(f"RENDERER: First frame of new playlist item {playlist_item_index} detected")

        if not self.first_frame_received:
            self.establish_wallclock_delta(frame_timestamp)
        elif is_first_frame_of_new_item:
            # First frame of a new item - check if we need to adjust timeline
            current_wallclock = time.time()
            test_target = frame_timestamp + self.get_adjusted_wallclock_delta()
            lateness = current_wallclock - test_target

            if lateness > 0.05:  # If more than 50ms late
                # Adjust the wallclock delta to make this frame on-time
                logger.warning(f"🎬 Adjusting timeline for new playlist item - frame was {lateness*1000:.1f}ms late")
                self.wallclock_delta = current_wallclock - frame_timestamp
                # Reset pause time since we're resetting the timeline
                self.total_pause_time = 0.0
                logger.info(f"New wallclock delta: {self.wallclock_delta:.3f}s")

        # Calculate target wallclock time with pause compensation
        target_wallclock = frame_timestamp + self.get_adjusted_wallclock_delta()
        current_wallclock = time.time()

        # Time difference (negative = early, positive = late)
        time_diff = current_wallclock - target_wallclock

        # Track timing error for statistics
        self._track_timing_error(time_diff)

        # Debug logging for high FPS investigation
        if self.frames_rendered % 1 == 0:  # Log every frame
            waiting = f", waiting {-time_diff*1000:.1f}ms" if time_diff < -self.timing_tolerance else ""
            late = f", late {time_diff*1000:.1f}ms" if time_diff > self.timing_tolerance else ""
            logger.debug(
                f"Frame {self.frames_rendered}: timestamp={frame_timestamp:.3f}, "
                f"target_wall={target_wallclock:.3f}, current_wall={current_wallclock:.3f}, "
                f"time_diff={time_diff*1000:.1f}ms, ewma_fps={self.ewma_fps:.1f}"
                f"{waiting}{late}"
            )

        try:
            # Store current frame timestamp for use in _send_to_outputs
            self._current_frame_timestamp = frame_timestamp

            # Capture first frame timestamp for current item if needed
            if metadata and "playlist_item_index" in metadata and self.current_item_first_frame_timestamp is None:
                self.current_item_first_frame_timestamp = frame_timestamp
                logger.debug(
                    f"Captured first frame timestamp {frame_timestamp:.3f} for item {metadata['playlist_item_index']}"
                )

            if time_diff > self.timing_tolerance:
                # Late frame - render immediately
                self.late_frames += 1
                self.total_late_time += time_diff

                self._send_to_outputs(led_values, metadata)

            elif time_diff < -self.timing_tolerance:
                # Early frame - wait until target time
                wait_time = -time_diff
                self.early_frames += 1
                self.total_wait_time += wait_time

                time.sleep(wait_time)
                self._send_to_outputs(led_values, metadata)

            else:
                # On time - render immediately
                self.on_time_frames += 1
                self._send_to_outputs(led_values, metadata)

            self.frames_rendered += 1
            self._update_ewma_statistics(frame_timestamp)
            self._log_beat_statistics_periodic()
            return True

        except Exception as e:
            logger.error(f"Error rendering frame: {e}")
            return False

    def _send_to_outputs(self, led_values: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Send LED values to all enabled output sinks.

        Args:
            led_values: LED RGB values in spatial order, shape (led_count, 3)
            metadata: Optional frame metadata
        """
        # Convert from spatial to physical order before sending to sinks
        physical_led_values = self._convert_spatial_to_physical(led_values)

        # Apply LED effects (templates, animations, etc.)
        # Convert current wall-clock time to frame timeline position
        # This accounts for late frames - we use the current position on frame timeline,
        # not the frame's target timestamp
        current_wall_clock = time.time()
        frame_timeline_time = current_wall_clock - self.get_adjusted_wallclock_delta()
        self.effect_manager.apply_effects(physical_led_values, frame_timeline_time)

        # Apply audio-reactive brightness boost if enabled (legacy inline implementation)
        # Note: Uses wall-clock time, not frame timestamp
        brightness_multiplier = self._calculate_beat_brightness_boost(time.time())

        # Structured logging for timeline reconstruction (log every frame's brightness)
        logger.debug(f"BRIGHTNESS_BOOST: wall_time={time.time():.6f}, multiplier={brightness_multiplier:.4f}")

        if brightness_multiplier > 1.01:  # Only apply if boost is meaningful (> 1%)
            # Apply brightness boost to all LED values
            boosted_values = physical_led_values.astype(np.float32) * brightness_multiplier
            # Ensure values stay within valid range [0, 255]
            physical_led_values = np.clip(boosted_values, 0, 255).astype(np.uint8)

            # Track boost statistics
            self.beat_boost_count += 1
            self.beat_boost_sum += brightness_multiplier

            # Log brightness boost application
            if np.random.random() < 0.2:  # Log 20% of boost applications
                logger.info(f"🎵 Beat brightness boost applied: {brightness_multiplier:.3f}x")

        # Calculate position offset for audio-reactive position shifting
        position_offset = self._position_shifter.calculate_position_offset(time.time(), physical_led_values.shape[0])

        # Track position shift statistics
        if position_offset != 0:
            self.beat_shift_count += 1
            self.beat_shift_sum += abs(position_offset)

            # Log position shift calculation (DEBUG level)
            logger.debug(
                f"Position offset calculated: {position_offset} LEDs "
                f"(max_shift={self._position_shifter.max_shift_distance})"
            )

        # Add rendering_index to metadata for PreviewSink
        enhanced_metadata = metadata.copy() if metadata else {}
        if metadata and "playlist_item_index" in metadata:
            enhanced_metadata["rendering_index"] = metadata["playlist_item_index"]

            # Track first frame timestamp when rendering index changes (for playback position calculation)
            new_rendering_index = metadata["playlist_item_index"]
            if new_rendering_index != self.current_rendering_index:
                self.current_rendering_index = new_rendering_index
                # Reset first frame timestamp for the new item
                self.current_item_first_frame_timestamp = None
                logger.debug(f"Rendering index changed to {new_rendering_index}, resetting first frame timestamp")

            # Add playback position if we have the first frame timestamp
            if self.current_item_first_frame_timestamp is not None and hasattr(self, "_current_frame_timestamp"):
                # Calculate playback position as difference from first frame timestamp for this item
                # This gets the frame_timestamp that was passed to render_frame_at_timestamp
                # We need to access it from the current frame being rendered
                playback_position = self._current_frame_timestamp - self.current_item_first_frame_timestamp
                enhanced_metadata["playback_position"] = max(0.0, playback_position)  # Ensure non-negative
                # Removed spammy playback position log

        # Debug: Write first 10 different LED value sets to temporary files for analysis
        if self._debug_led_count < self._debug_max_leds:
            try:
                # Check if this frame is different from the previous one
                is_different = True
                if self._debug_previous_led_values is not None:
                    # Compare with previous frame (use spatial values for comparison)
                    diff = np.abs(led_values.astype(np.float32) - self._debug_previous_led_values.astype(np.float32))
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)

                    # Consider frames different if max difference > 1 or mean difference > 0.1
                    is_different = max_diff > 1.0 or mean_diff > 0.1

                if is_different:
                    # Save both spatial and physical LED values for comparison
                    debug_spatial_file = self._debug_led_dir / f"led_spatial_{self._debug_led_count:03d}.npy"
                    debug_physical_file = self._debug_led_dir / f"led_physical_{self._debug_led_count:03d}.npy"
                    np.save(debug_spatial_file, led_values)
                    np.save(debug_physical_file, physical_led_values)
                    logger.info(
                        f"DEBUG: Wrote unique LED values {self._debug_led_count} to {debug_spatial_file} and {debug_physical_file}"
                    )

                    # Update previous frame and increment counter
                    self._debug_previous_led_values = led_values.copy()
                    self._debug_led_count += 1

            except Exception as e:
                logger.warning(f"DEBUG: Failed to write LED values {self._debug_led_count}: {e}")

        # Send to all registered sinks
        for sink_info in self.sinks:
            if not sink_info["enabled"]:
                continue

            sink = sink_info["sink"]
            name = sink_info["name"]

            try:
                # Try different sink interfaces
                if hasattr(sink, "send_led_data"):
                    # WLED-style sink
                    result = sink.send_led_data(physical_led_values, position_offset)
                    if hasattr(result, "success") and not result.success:
                        if not sink_info["failing"]:
                            logger.warning(f"{name} transmission failed: {result.errors}")
                            sink_info["failing"] = True
                    else:
                        sink_info["failing"] = False
                elif hasattr(sink, "render_led_values"):
                    # Renderer-style sink
                    if hasattr(sink, "is_running") and not sink.is_running:
                        continue  # Skip if sink is not running
                    # Try to call with metadata (for preview sink), fall back to older signatures if it fails
                    try:
                        # First try with metadata (preview sink needs this for playback position)
                        sink.render_led_values(physical_led_values.astype(np.uint8), enhanced_metadata)
                        if name == "PreviewSink" and "playback_position" in enhanced_metadata:
                            logger.debug(
                                f"Called PreviewSink with metadata containing playback_position={enhanced_metadata['playback_position']:.3f}"
                            )
                    except TypeError as e:
                        logger.debug(f"Sink {name} doesn't accept metadata parameter, trying older signatures: {e}")
                        try:
                            # Fall back to signature with position_offset
                            sink.render_led_values(physical_led_values.astype(np.uint8), position_offset)
                        except TypeError:
                            # Fall back to old signature without position_offset (for compatibility)
                            sink.render_led_values(physical_led_values.astype(np.uint8))
                else:
                    logger.warning(f"Sink {name} has no compatible interface")

            except Exception as e:
                logger.error(f"{name} sink error: {e}")

        # Measure time immediately after sending to all sinks for FPS calculation
        current_time = time.time()
        self._update_output_fps(current_time)

    def _track_timing_error(self, time_diff: float) -> None:
        """
        Track timing errors for statistical analysis.

        Args:
            time_diff: Timing difference in seconds (positive = late, negative = early)
        """
        self.timing_errors.append(time_diff)

        # Keep only recent history
        if len(self.timing_errors) > self.max_timing_history:
            self.timing_errors = self.timing_errors[-self.max_timing_history :]

    def _update_ewma_statistics(self, frame_timestamp: float) -> None:
        """
        Update EWMA-based statistics for recent performance tracking.

        Args:
            frame_timestamp: Current frame's presentation timestamp
        """
        current_time = time.time()

        # Log large timestamp gaps that might indicate transitions
        if self.last_frame_timestamp > 0:
            frame_interval = frame_timestamp - self.last_frame_timestamp
            if frame_interval > self.large_timestamp_gap_threshold:
                logger.warning(
                    f"Large frame timestamp gap detected: {frame_interval:.3f}s "
                    f"(previous: {self.last_frame_timestamp:.3f}, current: {frame_timestamp:.3f})"
                )

        # Calculate instantaneous values based on wall-clock render timing
        if self.last_ewma_update > 0:
            wall_clock_interval = current_time - self.last_ewma_update
            instant_fps = 1.0 / wall_clock_interval if wall_clock_interval > 0 else 0.0

            # Update EWMA FPS based on actual render timing
            if self.ewma_fps == 0.0:
                self.ewma_fps = instant_fps
            else:
                self.ewma_fps = (1 - self.ewma_alpha) * self.ewma_fps + self.ewma_alpha * instant_fps

        # Update EWMA fractions
        late_fraction = self.late_frames / max(1, self.frames_rendered)
        dropped_fraction = self.dropped_frames / max(1, self.frames_rendered)

        if self.frames_rendered == 1:
            # First frame, initialize EWMA
            self.ewma_late_fraction = late_fraction
            self.ewma_dropped_fraction = dropped_fraction
        else:
            # Update EWMA
            self.ewma_late_fraction = (1 - self.ewma_alpha) * self.ewma_late_fraction + self.ewma_alpha * late_fraction
            self.ewma_dropped_fraction = (
                1 - self.ewma_alpha
            ) * self.ewma_dropped_fraction + self.ewma_alpha * dropped_fraction

        self.last_ewma_update = current_time
        self.last_frame_timestamp = frame_timestamp

    def _log_beat_statistics_periodic(self) -> None:
        """
        Log beat effect statistics periodically (every 30 seconds).
        Provides visibility into beat boost and position shift application rates.
        """
        current_time = time.time()
        time_since_last_log = current_time - self.last_beat_stats_log

        if time_since_last_log >= self.beat_stats_log_interval:
            # Calculate statistics
            boost_percentage = (self.beat_boost_count / max(1, self.frames_rendered)) * 100
            avg_boost = self.beat_boost_sum / max(1, self.beat_boost_count)
            shift_percentage = (self.beat_shift_count / max(1, self.frames_rendered)) * 100
            avg_shift = self.beat_shift_sum / max(1, self.beat_shift_count)

            logger.info(
                f"🎵 Beat effects summary ({self.beat_stats_log_interval:.0f}s): "
                f"{self.beat_boost_count}/{self.frames_rendered} frames boosted ({boost_percentage:.1f}%), "
                f"avg_boost={avg_boost:.3f}x, "
                f"{self.beat_shift_count} shifts ({shift_percentage:.1f}%), "
                f"avg_shift={avg_shift:.1f} LEDs"
            )

            # Reset counters for next period
            self.beat_boost_count = 0
            self.beat_boost_sum = 0.0
            self.beat_shift_count = 0
            self.beat_shift_sum = 0.0
            self.last_beat_stats_log = current_time

    def _update_output_fps(self, current_time: float, alpha: float = 0.1) -> None:
        """
        Update output FPS tracking based on wall-clock time between sink calls.
        This measures the actual rate at which frames are rendered to sinks.

        Args:
            current_time: Time when sink calls completed (from time.time())
            alpha: EWMA smoothing factor
        """
        # Calculate FPS based on the time since the PREVIOUS sink call completed
        if hasattr(self, "_last_sink_call_time") and self._last_sink_call_time > 0:
            time_diff = current_time - self._last_sink_call_time
            if time_diff > 0:
                current_fps = 1.0 / time_diff
                if not hasattr(self, "output_fps_ewma"):
                    self.output_fps_ewma = current_fps
                else:
                    self.output_fps_ewma = (1 - alpha) * self.output_fps_ewma + alpha * current_fps

        # Store this time as the start of the next interval
        self._last_sink_call_time = current_time

        # Initialize if needed
        if not hasattr(self, "output_fps_ewma"):
            self.output_fps_ewma = 0.0

    def get_output_fps(self) -> float:
        """Get the current output FPS (frames sent to sinks per second)."""
        return getattr(self, "output_fps_ewma", 0.0)

    def _convert_spatial_to_physical(self, led_values: np.ndarray) -> np.ndarray:
        """
        Convert LED values from spatial order to physical order using explicit element copying.

        Args:
            led_values: LED values in spatial order, shape (led_count, 3)

        Returns:
            LED values in physical order, shape (led_count, 3)
        """
        # LED ordering should have been validated at load time
        # led_ordering maps spatial_index -> physical_led_id
        # We want to place spatial_led_values[spatial_idx] at physical_led_values[physical_led_id]
        physical_led_values = np.zeros_like(led_values)

        # Use explicit element-by-element copy to ensure proper memory reordering
        # self.led_ordering[i] gives the physical LED ID for spatial index i
        for spatial_idx in range(len(led_values)):
            physical_led_id = self.led_ordering[spatial_idx]
            physical_led_values[physical_led_id] = led_values[spatial_idx].copy()

        return physical_led_values

    def get_timing_stats(self) -> Dict[str, Any]:
        """
        Get detailed timing statistics.

        Returns:
            Dictionary with timing statistics
        """
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frames_rendered / elapsed_time if elapsed_time > 0 else 0.0

        # Calculate timing error statistics
        timing_error_stats = {}
        if self.timing_errors:
            errors_ms = [err * 1000 for err in self.timing_errors]
            timing_error_stats = {
                "mean_error_ms": np.mean(errors_ms),
                "std_error_ms": np.std(errors_ms),
                "min_error_ms": np.min(errors_ms),
                "max_error_ms": np.max(errors_ms),
                "p95_error_ms": np.percentile(errors_ms, 95),
                "p99_error_ms": np.percentile(errors_ms, 99),
            }

        return {
            # Basic counts
            "frames_rendered": self.frames_rendered,
            "late_frames": self.late_frames,
            "early_frames": self.early_frames,
            "on_time_frames": self.on_time_frames,
            "dropped_frames": self.dropped_frames,
            # Timing statistics
            "avg_render_fps": avg_fps,
            "late_frame_percentage": (self.late_frames / max(1, self.frames_rendered)) * 100,
            "early_frame_percentage": (self.early_frames / max(1, self.frames_rendered)) * 100,
            "on_time_percentage": (self.on_time_frames / max(1, self.frames_rendered)) * 100,
            "dropped_frame_percentage": (self.dropped_frames / max(1, self.frames_rendered)) * 100,
            # EWMA statistics (recent performance)
            "ewma_fps": self.ewma_fps,
            "ewma_late_fraction": self.ewma_late_fraction,
            "ewma_dropped_fraction": self.ewma_dropped_fraction,
            "ewma_late_percentage": self.ewma_late_fraction * 100,
            "ewma_dropped_percentage": self.ewma_dropped_fraction * 100,
            "ewma_alpha": self.ewma_alpha,
            # Timing details
            "avg_wait_time_ms": (self.total_wait_time / max(1, self.early_frames)) * 1000,
            "avg_late_time_ms": (self.total_late_time / max(1, self.late_frames)) * 1000,
            "total_wait_time_s": self.total_wait_time,
            "total_late_time_s": self.total_late_time,
            # Configuration
            "first_frame_delay_ms": self.first_frame_delay * 1000,
            "timing_tolerance_ms": self.timing_tolerance * 1000,
            "wallclock_delta_s": self.wallclock_delta,
            "adjusted_wallclock_delta_s": self.get_adjusted_wallclock_delta(),
            "total_pause_time_s": self.total_pause_time,
            "is_paused": self.is_paused,
            "first_frame_received": self.first_frame_received,
            # Output sinks
            "registered_sinks": len(self.sinks),
            "enabled_sinks": sum(1 for s in self.sinks if s["enabled"]),
            "sink_names": [s["name"] for s in self.sinks if s["enabled"]],
            # Legacy compatibility
            "wled_enabled": self.enable_wled,
            "test_sink_enabled": self.enable_test_sink,
            # LED effects
            "led_effects": self.effect_manager.get_stats(),
            # Timing distribution
            **timing_error_stats,
        }

    def get_renderer_stats(self) -> Dict[str, Any]:
        """Get renderer statistics (alias for get_timing_stats)."""
        return self.get_timing_stats()

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.frames_rendered = 0
        self.late_frames = 0
        self.early_frames = 0
        self.on_time_frames = 0
        self.dropped_frames = 0
        self.total_wait_time = 0.0
        self.total_late_time = 0.0
        self.timing_errors.clear()
        self.start_time = time.time()

        # Reset EWMA statistics
        self.ewma_fps = 0.0
        self.ewma_late_fraction = 0.0
        self.ewma_dropped_fraction = 0.0
        self.last_ewma_update = 0.0
        self.last_frame_timestamp = 0.0

        # Reset output FPS tracking
        self.output_fps_ewma = 0.0
        self._last_sink_call_time = 0.0

        # Reset pause tracking
        self.is_paused = False
        self.pause_start_time = None
        self.total_pause_time = 0.0

        logger.debug("Renderer statistics reset")

    def set_timing_parameters(
        self,
        first_frame_delay_ms: Optional[float] = None,
        timing_tolerance_ms: Optional[float] = None,
        late_frame_log_threshold_ms: Optional[float] = None,
    ) -> None:
        """
        Update timing parameters.

        Args:
            first_frame_delay_ms: New first frame delay
            timing_tolerance_ms: New timing tolerance
            late_frame_log_threshold_ms: New late frame log threshold
        """
        if first_frame_delay_ms is not None:
            self.first_frame_delay = first_frame_delay_ms / 1000.0

        if timing_tolerance_ms is not None:
            self.timing_tolerance = timing_tolerance_ms / 1000.0

        if late_frame_log_threshold_ms is not None:
            self.late_frame_log_threshold = late_frame_log_threshold_ms / 1000.0

        logger.info(
            f"Timing parameters updated: delay={self.first_frame_delay*1000:.1f}ms, "
            f"tolerance=±{self.timing_tolerance*1000:.1f}ms, "
            f"log_threshold={self.late_frame_log_threshold*1000:.1f}ms"
        )

    def set_ewma_alpha(self, alpha: float) -> None:
        """
        Set the EWMA alpha parameter for recent statistics tracking.

        Args:
            alpha: EWMA alpha parameter (0 < alpha <= 1, smaller = more smoothing)
        """
        if not (0 < alpha <= 1):
            raise ValueError("EWMA alpha must be between 0 and 1")

        self.ewma_alpha = alpha
        logger.info(f"EWMA alpha set to {alpha:.3f}")

    def mark_frame_dropped(self) -> None:
        """
        Mark a frame as dropped (for future frame dropping policies).

        This method should be called when a frame is intentionally dropped
        due to timing constraints or buffer overruns.
        """
        self.dropped_frames += 1
        self._update_ewma_statistics()
        logger.debug(f"Frame marked as dropped (total: {self.dropped_frames})")

    def get_recent_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent performance using EWMA statistics.

        Returns:
            Dictionary with recent performance metrics
        """
        return {
            "recent_fps": self.ewma_fps,
            "recent_late_percentage": self.ewma_late_fraction * 100,
            "recent_dropped_percentage": self.ewma_dropped_fraction * 100,
            "ewma_alpha": self.ewma_alpha,
            "frames_rendered": self.frames_rendered,
            "is_performing_well": (
                self.ewma_fps > 25  # At least 25 FPS
                and self.ewma_late_fraction < 0.1  # Less than 10% late frames
                and self.ewma_dropped_fraction < 0.05  # Less than 5% dropped frames
            ),
        }

    def is_initialized(self) -> bool:
        """Check if renderer is initialized with timing delta."""
        return self.first_frame_received and self.wallclock_delta is not None

    def is_frame_late(
        self, frame_timestamp: float, late_threshold_ms: float = 50.0, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a frame is already late for rendering.

        Args:
            frame_timestamp: Presentation timestamp of the frame
            late_threshold_ms: Threshold in milliseconds for considering a frame late
            metadata: Optional frame metadata to check for first frame of new item

        Returns:
            True if frame is late and should be dropped, False if frame should be processed
        """
        if not self.is_initialized():
            # If renderer not initialized, we can't determine timing - process the frame
            return False

        # Use the metadata from producer to check if this is the first frame of a new item
        if metadata and metadata.get("is_first_frame_of_item", False):
            # First frame of new item - never consider it late (will adjust timeline instead)
            return False

        # Calculate target wallclock time with pause compensation
        target_wallclock = frame_timestamp + self.get_adjusted_wallclock_delta()
        current_wallclock = time.time()

        # Time difference (positive = late)
        time_diff = current_wallclock - target_wallclock
        late_threshold = late_threshold_ms / 1000.0

        return time_diff > late_threshold

    def pause_renderer(self) -> None:
        """
        Mark the renderer as paused and start tracking pause time.
        """
        if not self.is_paused:
            self.is_paused = True
            self.pause_start_time = time.time()
            logger.debug("Renderer paused, started tracking pause time")

    def resume_renderer(self) -> None:
        """
        Mark the renderer as resumed and add accumulated pause time to offset.
        """
        if self.is_paused and self.pause_start_time is not None:
            pause_duration = time.time() - self.pause_start_time
            self.total_pause_time += pause_duration
            self.is_paused = False
            self.pause_start_time = None
            logger.info(
                f"Renderer resumed, added {pause_duration:.3f}s pause time (total: {self.total_pause_time:.3f}s)"
            )

    def get_adjusted_wallclock_delta(self) -> float:
        """
        Get wallclock delta adjusted for pause time.

        Returns:
            Adjusted wallclock delta that accounts for time spent in pause
        """
        if self.wallclock_delta is None:
            return 0.0

        # Add total pause time to the delta to compensate for paused periods
        adjusted_delta = self.wallclock_delta + self.total_pause_time

        # If currently paused, also add the current pause duration
        if self.is_paused and self.pause_start_time is not None:
            current_pause_duration = time.time() - self.pause_start_time
            adjusted_delta += current_pause_duration

        return adjusted_delta

    def add_led_effect(self, effect) -> None:
        """
        Add a new LED effect to the active effects list.

        Args:
            effect: LedEffect instance to add

        Example:
            from .led_effect import TemplateEffect
            template = np.load('template_leds.npy')

            # Use frame timestamp (from current frame being rendered)
            # NOT wall-clock time!
            effect = TemplateEffect(
                start_time=current_frame_timestamp,  # From frame timeline
                template=template,
                duration=2.0,  # Effect plays over 2 seconds
                blend_mode='alpha',
                intensity=0.8
            )
            renderer.add_led_effect(effect)
        """
        self.effect_manager.add_effect(effect)

    def clear_led_effects(self) -> None:
        """Remove all active LED effects."""
        self.effect_manager.clear_effects()

    def get_active_effects_count(self) -> int:
        """Get the number of active LED effects."""
        return self.effect_manager.get_active_count()

    def get_led_effects_stats(self) -> Dict[str, Any]:
        """Get LED effects statistics."""
        return self.effect_manager.get_stats()
