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

from .test_renderer import TestRenderer
from .wled_client import WLEDClient

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

        # Output targets
        self.wled_client: Optional[WLEDClient] = None
        self.test_renderer: Optional[TestRenderer] = None
        self.enable_wled = True
        self.enable_test_renderer = False

        # Statistics
        self.frames_rendered = 0
        self.late_frames = 0
        self.early_frames = 0
        self.on_time_frames = 0
        self.total_wait_time = 0.0
        self.total_late_time = 0.0
        self.start_time = time.time()

        # Timing distribution tracking
        self.timing_errors = []  # Track last 100 timing errors for analysis
        self.max_timing_history = 100

        logger.info(
            f"FrameRenderer initialized: delay={first_frame_delay_ms}ms, " f"tolerance=±{timing_tolerance_ms}ms"
        )

    def set_output_targets(
        self, wled_client: Optional[WLEDClient] = None, test_renderer: Optional[TestRenderer] = None
    ) -> None:
        """
        Set output targets for rendering.

        Args:
            wled_client: WLED client for LED output
            test_renderer: Test renderer for debugging
        """
        self.wled_client = wled_client
        self.test_renderer = test_renderer
        self.enable_wled = wled_client is not None
        self.enable_test_renderer = test_renderer is not None

        logger.info(f"Output targets: WLED={self.enable_wled}, " f"TestRenderer={self.enable_test_renderer}")

    def set_wled_enabled(self, enabled: bool) -> None:
        """Enable or disable WLED output."""
        self.enable_wled = enabled and (self.wled_client is not None)

    def set_test_renderer_enabled(self, enabled: bool) -> None:
        """Enable or disable test renderer output."""
        self.enable_test_renderer = enabled and (self.test_renderer is not None)

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
            return True

        except Exception as e:
            logger.error(f"Error rendering frame: {e}")
            return False

    def _send_to_outputs(self, led_values: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Send LED values to all enabled output targets.

        Args:
            led_values: LED RGB values, shape (led_count, 3)
            metadata: Optional frame metadata
        """
        # Send to WLED
        if self.enable_wled and self.wled_client:
            try:
                result = self.wled_client.send_led_data(led_values)
                if not result.success:
                    logger.warning(f"WLED transmission failed: {result.errors}")
            except Exception as e:
                logger.error(f"WLED output error: {e}")

        # Send to test renderer
        if self.enable_test_renderer and self.test_renderer:
            try:
                if self.test_renderer.is_running:
                    self.test_renderer.render_led_values(led_values.astype(np.uint8))
            except Exception as e:
                logger.error(f"Test renderer output error: {e}")

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
            # Timing statistics
            "avg_render_fps": avg_fps,
            "late_frame_percentage": (self.late_frames / max(1, self.frames_rendered)) * 100,
            "early_frame_percentage": (self.early_frames / max(1, self.frames_rendered)) * 100,
            "on_time_percentage": (self.on_time_frames / max(1, self.frames_rendered)) * 100,
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
            # Output targets
            "wled_enabled": self.enable_wled,
            "test_renderer_enabled": self.enable_test_renderer,
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
        self.total_wait_time = 0.0
        self.total_late_time = 0.0
        self.timing_errors.clear()
        self.start_time = time.time()

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

    def is_initialized(self) -> bool:
        """Check if renderer is initialized with timing delta."""
        return self.first_frame_received and self.wallclock_delta is not None
