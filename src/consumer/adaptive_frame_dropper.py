"""
Adaptive Frame Dropper for LED Buffer Management.

This module implements a feedback control system for frame dropping that maintains
healthy LED buffer levels by adaptively adjusting frame drop rates based on
buffer occupancy using EWMA tracking.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class AdaptiveFrameDropper:
    """
    Adaptive frame dropper that maintains LED buffer health through feedback control.

    The system works by:
    1. Tracking actual frame drop rate with EWMA
    2. Tracking LED buffer occupancy with EWMA
    3. Adjusting target drop rate based on buffer level
    4. Dropping frames deterministically when actual < target
    """

    def __init__(
        self,
        led_buffer_low_threshold: float = 3.0,
        led_buffer_high_threshold: float = 8.0,
        drop_rate_step_size: float = 0.05,
        led_buffer_ewma_alpha: float = 0.1,
        frame_drop_ewma_alpha: float = 0.1,
        initial_target_drop_rate: float = 0.0,
    ):
        """
        Initialize adaptive frame dropper.

        Args:
            led_buffer_low_threshold: LED buffer level below which to increase drop rate
            led_buffer_high_threshold: LED buffer level above which to decrease drop rate
            drop_rate_step_size: Step size for adjusting target drop rate (e.g., 0.05 = 5%)
            led_buffer_ewma_alpha: EWMA alpha for LED buffer level tracking
            frame_drop_ewma_alpha: EWMA alpha for actual drop rate tracking
            initial_target_drop_rate: Initial target drop rate (0.0 = 0%, 1.0 = 100%)
        """
        # Configuration parameters
        self.led_buffer_low_threshold = led_buffer_low_threshold
        self.led_buffer_high_threshold = led_buffer_high_threshold
        self.drop_rate_step_size = drop_rate_step_size
        self.led_buffer_ewma_alpha = led_buffer_ewma_alpha
        self.frame_drop_ewma_alpha = frame_drop_ewma_alpha

        # State variables
        self.target_drop_rate = initial_target_drop_rate
        self.actual_drop_rate_ewma = 0.0
        self.led_buffer_level_ewma = 0.0

        # Frame tracking
        self.frames_processed = 0
        self.frames_dropped = 0
        self.last_update_time = time.time()

        # Statistics
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        self.target_adjustments_up = 0
        self.target_adjustments_down = 0

        logger.info(
            f"AdaptiveFrameDropper initialized: "
            f"low_threshold={led_buffer_low_threshold}, "
            f"high_threshold={led_buffer_high_threshold}, "
            f"step_size={drop_rate_step_size}, "
            f"buffer_alpha={led_buffer_ewma_alpha}, "
            f"drop_alpha={frame_drop_ewma_alpha}"
        )

    def should_drop_frame(self, frame_timestamp: float, led_buffer_size: int) -> bool:
        """
        Determine if a frame should be dropped based on adaptive strategy.

        Args:
            frame_timestamp: Timestamp of the frame being considered
            led_buffer_size: Current LED buffer occupancy

        Returns:
            True if frame should be dropped, False otherwise
        """
        # Update LED buffer level EWMA
        self._update_led_buffer_ewma(led_buffer_size)

        # Update target drop rate based on buffer level
        self._update_target_drop_rate()

        # Determine if we should drop this frame
        should_drop = self._should_drop_deterministic()

        # Update actual drop rate EWMA
        self._update_actual_drop_rate_ewma(should_drop)

        # Update statistics
        self.frames_processed += 1
        self.total_frames_processed += 1
        if should_drop:
            self.frames_dropped += 1
            self.total_frames_dropped += 1

        # Log periodic statistics
        self._log_periodic_stats()

        return should_drop

    def _update_led_buffer_ewma(self, current_buffer_size: int) -> None:
        """Update LED buffer level EWMA."""
        if self.led_buffer_level_ewma == 0.0:
            # Initialize on first sample
            self.led_buffer_level_ewma = float(current_buffer_size)
        else:
            self.led_buffer_level_ewma = (
                1 - self.led_buffer_ewma_alpha
            ) * self.led_buffer_level_ewma + self.led_buffer_ewma_alpha * current_buffer_size

    def _update_target_drop_rate(self) -> None:
        """Update target drop rate based on LED buffer level EWMA."""
        old_target = self.target_drop_rate

        if self.led_buffer_level_ewma < self.led_buffer_low_threshold:
            # Buffer too low - increase drop rate to reduce load
            self.target_drop_rate = min(1.0, self.target_drop_rate + self.drop_rate_step_size)
            if self.target_drop_rate > old_target:
                self.target_adjustments_up += 1

        elif self.led_buffer_level_ewma > self.led_buffer_high_threshold:
            # Buffer too high - decrease drop rate to process more frames
            self.target_drop_rate = max(0.0, self.target_drop_rate - self.drop_rate_step_size)
            if self.target_drop_rate < old_target:
                self.target_adjustments_down += 1

        # Log target adjustments
        if self.target_drop_rate != old_target:
            logger.debug(
                f"Target drop rate adjusted: {old_target:.3f} -> {self.target_drop_rate:.3f} "
                f"(buffer_ewma={self.led_buffer_level_ewma:.1f})"
            )

    def _should_drop_deterministic(self) -> bool:
        """
        Deterministically decide if frame should be dropped.

        Returns True when actual drop rate falls below target.
        """
        # If actual drop rate is below target, we should drop more frames
        return self.actual_drop_rate_ewma < self.target_drop_rate

    def _update_actual_drop_rate_ewma(self, frame_dropped: bool) -> None:
        """Update actual drop rate EWMA."""
        drop_sample = 1.0 if frame_dropped else 0.0

        if self.frames_processed == 1:
            # Initialize on first sample
            self.actual_drop_rate_ewma = drop_sample
        else:
            self.actual_drop_rate_ewma = (
                1 - self.frame_drop_ewma_alpha
            ) * self.actual_drop_rate_ewma + self.frame_drop_ewma_alpha * drop_sample

    def _log_periodic_stats(self) -> None:
        """Log statistics periodically."""
        current_time = time.time()

        # Log every 100 frames or every 5 seconds
        if self.frames_processed % 100 == 0 or (current_time - self.last_update_time) > 5.0:
            recent_drop_rate = self.frames_dropped / max(1, self.frames_processed) * 100
            total_drop_rate = self.total_frames_dropped / max(1, self.total_frames_processed) * 100

            logger.info(
                f"Frame Drop Stats: "
                f"target={self.target_drop_rate*100:.1f}%, "
                f"actual_ewma={self.actual_drop_rate_ewma*100:.1f}%, "
                f"recent={recent_drop_rate:.1f}%, "
                f"total={total_drop_rate:.1f}%, "
                f"buffer_ewma={self.led_buffer_level_ewma:.1f}, "
                f"adjustments_up={self.target_adjustments_up}, "
                f"adjustments_down={self.target_adjustments_down}"
            )

            # Reset periodic counters
            self.frames_processed = 0
            self.frames_dropped = 0
            self.last_update_time = current_time

    def get_stats(self) -> dict:
        """
        Get current statistics.

        Returns:
            Dictionary with current state and statistics
        """
        total_drop_rate = self.total_frames_dropped / max(1, self.total_frames_processed)

        return {
            "target_drop_rate": self.target_drop_rate,
            "actual_drop_rate_ewma": self.actual_drop_rate_ewma,
            "led_buffer_level_ewma": self.led_buffer_level_ewma,
            "total_frames_processed": self.total_frames_processed,
            "total_frames_dropped": self.total_frames_dropped,
            "total_drop_rate": total_drop_rate,
            "target_adjustments_up": self.target_adjustments_up,
            "target_adjustments_down": self.target_adjustments_down,
            "config": {
                "led_buffer_low_threshold": self.led_buffer_low_threshold,
                "led_buffer_high_threshold": self.led_buffer_high_threshold,
                "drop_rate_step_size": self.drop_rate_step_size,
                "led_buffer_ewma_alpha": self.led_buffer_ewma_alpha,
                "frame_drop_ewma_alpha": self.frame_drop_ewma_alpha,
            },
        }

    def reset_stats(self) -> None:
        """Reset all statistics and counters."""
        self.frames_processed = 0
        self.frames_dropped = 0
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        self.target_adjustments_up = 0
        self.target_adjustments_down = 0
        self.last_update_time = time.time()

        # Reset EWMA values to start fresh
        self.actual_drop_rate_ewma = 0.0
        self.led_buffer_level_ewma = 0.0
        self.target_drop_rate = 0.0

        logger.info("AdaptiveFrameDropper statistics reset")

    def update_config(
        self,
        led_buffer_low_threshold: Optional[float] = None,
        led_buffer_high_threshold: Optional[float] = None,
        drop_rate_step_size: Optional[float] = None,
        led_buffer_ewma_alpha: Optional[float] = None,
        frame_drop_ewma_alpha: Optional[float] = None,
    ) -> None:
        """
        Update configuration parameters.

        Args:
            led_buffer_low_threshold: New low threshold (optional)
            led_buffer_high_threshold: New high threshold (optional)
            drop_rate_step_size: New step size (optional)
            led_buffer_ewma_alpha: New LED buffer EWMA alpha (optional)
            frame_drop_ewma_alpha: New frame drop EWMA alpha (optional)
        """
        if led_buffer_low_threshold is not None:
            self.led_buffer_low_threshold = led_buffer_low_threshold
        if led_buffer_high_threshold is not None:
            self.led_buffer_high_threshold = led_buffer_high_threshold
        if drop_rate_step_size is not None:
            self.drop_rate_step_size = drop_rate_step_size
        if led_buffer_ewma_alpha is not None:
            self.led_buffer_ewma_alpha = led_buffer_ewma_alpha
        if frame_drop_ewma_alpha is not None:
            self.frame_drop_ewma_alpha = frame_drop_ewma_alpha

        logger.info(f"AdaptiveFrameDropper configuration updated: {self.get_stats()['config']}")
