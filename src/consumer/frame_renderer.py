"""
Timestamp-Based Frame Renderer.

This module implements precise timestamp-based rendering for LED displays.
Handles wallclock timing establishment, late/early frame logic, and output
to multiple targets (WLED, test renderer).
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

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
    """

    def __init__(
        self,
        first_frame_delay_ms: float = 100.0,
        timing_tolerance_ms: float = 5.0,
        late_frame_log_threshold_ms: float = 50.0,
    ):
        """
        Initialize frame renderer.

        Args:
            first_frame_delay_ms: Default delay for first frame buffering
            timing_tolerance_ms: Acceptable timing deviation
            late_frame_log_threshold_ms: Log late frames above this threshold
        """
        self.first_frame_delay = first_frame_delay_ms / 1000.0
        self.timing_tolerance = timing_tolerance_ms / 1000.0
        self.late_frame_log_threshold = late_frame_log_threshold_ms / 1000.0

        # Timing state
        self.wallclock_delta = None  # Established from first frame
        self.first_frame_received = False
        self.first_frame_timestamp = None

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

        # EWMA statistics for recent performance tracking
        self.ewma_alpha = 0.1  # EWMA smoothing factor
        self.ewma_fps = 0.0
        self.ewma_late_fraction = 0.0
        self.ewma_dropped_fraction = 0.0
        self.last_ewma_update = 0.0

        # Timing distribution tracking
        self.timing_errors = []  # Track last 100 timing errors for analysis
        self.max_timing_history = 100

        logger.info(
            f"FrameRenderer initialized: delay={first_frame_delay_ms}ms, " f"tolerance=±{timing_tolerance_ms}ms"
        )

    def register_sink(self, sink, name: str, enabled: bool = True) -> None:
        """
        Register a new output sink.

        Args:
            sink: Sink instance that must have a method to receive LED data
            name: Human-readable name for the sink
            enabled: Whether the sink is initially enabled
        """
        if hasattr(sink, "send_led_data") or hasattr(sink, "render_led_values"):
            self.sinks.append({"sink": sink, "name": name, "enabled": enabled})
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
        if not self.first_frame_received:
            self.establish_wallclock_delta(frame_timestamp)

        # Calculate target wallclock time
        target_wallclock = frame_timestamp + self.wallclock_delta
        current_wallclock = time.time()

        # Time difference (negative = early, positive = late)
        time_diff = current_wallclock - target_wallclock

        # Track timing error for statistics
        self._track_timing_error(time_diff)

        try:
            if time_diff > self.timing_tolerance:
                # Late frame - render immediately
                self.late_frames += 1
                self.total_late_time += time_diff

                if time_diff > self.late_frame_log_threshold:
                    logger.debug(f"Late frame: {time_diff*1000:.1f}ms")

                self._send_to_outputs(led_values, metadata)

            elif time_diff < -self.timing_tolerance:
                # Early frame - wait until target time
                wait_time = -time_diff
                self.early_frames += 1
                self.total_wait_time += wait_time

                logger.debug(f"Early frame: waiting {wait_time*1000:.1f}ms")
                time.sleep(wait_time)
                self._send_to_outputs(led_values, metadata)

            else:
                # On time - render immediately
                self.on_time_frames += 1
                self._send_to_outputs(led_values, metadata)

            self.frames_rendered += 1
            self._update_ewma_statistics()
            return True

        except Exception as e:
            logger.error(f"Error rendering frame: {e}")
            return False

    def _send_to_outputs(self, led_values: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Send LED values to all enabled output sinks.

        Args:
            led_values: LED RGB values, shape (led_count, 3)
            metadata: Optional frame metadata
        """
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
                    result = sink.send_led_data(led_values)
                    if hasattr(result, "success") and not result.success:
                        logger.warning(f"{name} transmission failed: {result.errors}")
                elif hasattr(sink, "render_led_values"):
                    # Renderer-style sink
                    if hasattr(sink, "is_running") and not sink.is_running:
                        continue  # Skip if sink is not running
                    sink.render_led_values(led_values.astype(np.uint8), metadata)
                else:
                    logger.warning(f"Sink {name} has no compatible interface")

            except Exception as e:
                logger.error(f"{name} sink error: {e}")

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

    def _update_ewma_statistics(self) -> None:
        """
        Update EWMA-based statistics for recent performance tracking.
        """
        current_time = time.time()

        # Calculate instantaneous values
        if self.last_ewma_update > 0:
            frame_interval = current_time - self.last_ewma_update
            instant_fps = 1.0 / frame_interval if frame_interval > 0 else 0.0

            # Update EWMA FPS
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
            "first_frame_received": self.first_frame_received,
            # Output sinks
            "registered_sinks": len(self.sinks),
            "enabled_sinks": sum(1 for s in self.sinks if s["enabled"]),
            "sink_names": [s["name"] for s in self.sinks if s["enabled"]],
            # Legacy compatibility
            "wled_enabled": self.enable_wled,
            "test_sink_enabled": self.enable_test_sink,
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
