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
        led_buffer_capacity: int = 10,
        led_buffer_ewma_alpha: float = 0.005,
        max_drop_rate: float = 0.66,
    ):
        """
        Initialize adaptive frame dropper with direct scaling approach.

        Args:
            led_buffer_capacity: Expected LED buffer capacity for normalization
            led_buffer_ewma_alpha: EWMA alpha for LED buffer level tracking
            max_drop_rate: Maximum allowed drop rate (0.66 = support up to 2x input rate)
        """
        # Configuration parameters
        self.led_buffer_capacity = led_buffer_capacity
        self.led_buffer_ewma_alpha = led_buffer_ewma_alpha
        self.max_drop_rate = max_drop_rate

        # State variables
        self.target_drop_rate = 0.0
        self.led_buffer_level_ewma = 0.0

        # Frame tracking
        self.frames_processed = 0
        self.frames_dropped = 0
        self.last_update_time = time.time()

        # Statistics
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        self.rate_calculations = 0

        logger.info(
            f"AdaptiveFrameDropper initialized: "
            f"buffer_capacity={led_buffer_capacity}, "
            f"ewma_alpha={led_buffer_ewma_alpha}, "
            f"max_drop_rate={max_drop_rate:.2f}"
        )

    def should_drop_frame(self, frame_timestamp: float, led_buffer_size: int, renderer_state: str) -> bool:
        """
        Determine if a frame should be dropped based on direct scaling strategy.

        Args:
            frame_timestamp: Timestamp of the frame being considered
            led_buffer_size: Current LED buffer occupancy
            renderer_state: Current renderer state (only update EWMA when PLAYING)

        Returns:
            True if frame should be dropped, False otherwise
        """
        # Log every call to diagnose issues
        logger.debug(
            f"Adaptive frame dropper called: renderer_state={renderer_state}, led_buffer_size={led_buffer_size}, ewma={self.led_buffer_level_ewma:.1f}"
        )

        # Only update EWMA and calculate drop rate when renderer is PLAYING (case insensitive)
        if renderer_state.upper() == "PLAYING":
            logger.debug("Renderer is PLAYING - updating EWMA and calculating drop rate")
            self._update_led_buffer_ewma(led_buffer_size, renderer_state)
            # Calculate drop rate directly from buffer occupancy
            self._calculate_drop_rate()
            # Determine if we should drop this frame
            should_drop = self._should_drop_probabilistic()
            logger.debug(f"Drop decision: should_drop={should_drop}, target_rate={self.target_drop_rate:.3f}")
        else:
            # When not playing, never drop frames
            should_drop = False
            logger.debug(f"Renderer not PLAYING ({renderer_state}) - not dropping frame")

        # Update statistics
        self.frames_processed += 1
        self.total_frames_processed += 1
        if should_drop:
            self.frames_dropped += 1
            self.total_frames_dropped += 1

        # Log periodic statistics
        self._log_periodic_stats()

        return should_drop

    def _update_led_buffer_ewma(self, current_buffer_size: int, renderer_state: str) -> None:
        """Update LED buffer level EWMA with proper initialization for PLAYING state."""
        old_ewma = self.led_buffer_level_ewma

        if self.led_buffer_level_ewma == 0.0:
            # Initialize EWMA to current buffer size
            # Note: We theoretically transition to PLAYING when buffer is full, but by the time
            # adaptive dropping is called, the buffer may have already started draining
            self.led_buffer_level_ewma = float(current_buffer_size)
            logger.info(f"Initialized LED buffer EWMA to current size: {current_buffer_size} (state={renderer_state})")
        else:
            self.led_buffer_level_ewma = (
                1 - self.led_buffer_ewma_alpha
            ) * self.led_buffer_level_ewma + self.led_buffer_ewma_alpha * current_buffer_size

        logger.debug(
            f"EWMA updated: {old_ewma:.2f} -> {self.led_buffer_level_ewma:.2f} (buffer_size={current_buffer_size})"
        )

    def _calculate_drop_rate(self) -> None:
        """Calculate drop rate directly from buffer occupancy using scaling approach."""
        old_target = self.target_drop_rate

        # Normalize buffer occupancy to 0-1 range based on capacity
        normalized_occupancy = min(1.0, self.led_buffer_level_ewma / self.led_buffer_capacity)

        # Drop rate = min(max_drop_rate, 1 - normalized_occupancy)
        # When buffer is full (1.0), drop rate = 0.0
        # When buffer is empty (0.0), drop rate = max_drop_rate (0.66)
        # When buffer is half full (0.5), drop rate = 0.5
        self.target_drop_rate = min(self.max_drop_rate, 1.0 - normalized_occupancy)

        # Update statistics
        self.rate_calculations += 1

        # Log all drop rate calculations for debugging
        logger.debug(
            f"Drop rate calculated: {old_target:.3f} -> {self.target_drop_rate:.3f} "
            f"(buffer_ewma={self.led_buffer_level_ewma:.1f}, "
            f"normalized_occupancy={normalized_occupancy:.3f})"
        )

    def _should_drop_probabilistic(self) -> bool:
        """
        Probabilistically decide if frame should be dropped based on target drop rate.

        Uses a pattern-based approach to achieve the target drop rate over time.
        For example, drop rate 0.66 means drop 2 out of every 3 frames.
        """
        if self.target_drop_rate == 0.0:
            logger.debug("Drop rate is 0.0 - not dropping")
            return False
        if self.target_drop_rate >= 1.0:
            logger.debug(f"Drop rate >= 1.0 ({self.target_drop_rate:.3f}) - dropping")
            return True

        # Use pattern-based approach for fractional drop rates
        # Convert drop rate to a pattern over N frames
        if self.target_drop_rate > 0.0:
            # Use a repeating pattern approach
            # For 0.66 drop rate: drop 2 out of 3 frames (pattern length 3)
            # For 0.5 drop rate: drop 1 out of 2 frames (pattern length 2)
            # For 0.33 drop rate: drop 1 out of 3 frames (pattern length 3)

            # Find a reasonable pattern length (up to 10 frames)
            pattern_length = None
            for length in range(2, 11):
                expected_drops = round(self.target_drop_rate * length)
                actual_rate = expected_drops / length
                # Accept if within 5% of target rate
                if abs(actual_rate - self.target_drop_rate) < 0.05:
                    pattern_length = length
                    break

            if pattern_length is None:
                # Fallback: use length 10 for fine-grained control
                pattern_length = 10

            drops_in_pattern = round(self.target_drop_rate * pattern_length)
            frame_position = self.total_frames_processed % pattern_length

            # Drop the first N frames in each pattern cycle
            should_drop = frame_position < drops_in_pattern

            logger.debug(
                f"Pattern length: {pattern_length}, drops: {drops_in_pattern}, position: {frame_position}, frame: {self.total_frames_processed}, should_drop: {should_drop}"
            )
            return should_drop

        return False

    def _log_periodic_stats(self) -> None:
        """Log statistics periodically."""
        current_time = time.time()

        # Log every 100 frames or every 5 seconds
        if self.frames_processed % 100 == 0 or (current_time - self.last_update_time) > 5.0:
            recent_drop_rate = self.frames_dropped / max(1, self.frames_processed) * 100
            total_drop_rate = self.total_frames_dropped / max(1, self.total_frames_processed) * 100
            normalized_occupancy = min(1.0, self.led_buffer_level_ewma / self.led_buffer_capacity)

            logger.info(
                f"Frame Drop Stats: "
                f"target={self.target_drop_rate*100:.1f}%, "
                f"recent={recent_drop_rate:.1f}%, "
                f"total={total_drop_rate:.1f}%, "
                f"buffer_ewma={self.led_buffer_level_ewma:.1f}, "
                f"buffer_occupancy={normalized_occupancy:.3f}, "
                f"rate_calculations={self.rate_calculations}"
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

        normalized_occupancy = min(1.0, self.led_buffer_level_ewma / self.led_buffer_capacity)

        return {
            "target_drop_rate": self.target_drop_rate,
            "led_buffer_level_ewma": self.led_buffer_level_ewma,
            "normalized_buffer_occupancy": normalized_occupancy,
            "total_frames_processed": self.total_frames_processed,
            "total_frames_dropped": self.total_frames_dropped,
            "total_drop_rate": total_drop_rate,
            "rate_calculations": self.rate_calculations,
            "config": {
                "led_buffer_capacity": self.led_buffer_capacity,
                "led_buffer_ewma_alpha": self.led_buffer_ewma_alpha,
                "max_drop_rate": self.max_drop_rate,
            },
        }

    def reset_stats(self) -> None:
        """Reset all statistics and counters."""
        self.frames_processed = 0
        self.frames_dropped = 0
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        self.rate_calculations = 0
        self.last_update_time = time.time()

        # Reset EWMA values to start fresh
        self.led_buffer_level_ewma = 0.0
        self.target_drop_rate = 0.0

        logger.info("AdaptiveFrameDropper statistics reset")

    def update_config(
        self,
        led_buffer_capacity: Optional[int] = None,
        led_buffer_ewma_alpha: Optional[float] = None,
        max_drop_rate: Optional[float] = None,
    ) -> None:
        """
        Update configuration parameters.

        Args:
            led_buffer_capacity: New buffer capacity for normalization (optional)
            led_buffer_ewma_alpha: New LED buffer EWMA alpha (optional)
            max_drop_rate: New maximum drop rate (optional)
        """
        if led_buffer_capacity is not None:
            self.led_buffer_capacity = led_buffer_capacity
        if led_buffer_ewma_alpha is not None:
            self.led_buffer_ewma_alpha = led_buffer_ewma_alpha
        if max_drop_rate is not None:
            self.max_drop_rate = max_drop_rate

        logger.info(f"AdaptiveFrameDropper configuration updated: {self.get_stats()['config']}")
