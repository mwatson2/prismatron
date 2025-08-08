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
        use_pid_controller: bool = True,
        kp: float = 3.0,
        ki: float = 0.5,
        kd: float = 1.0,
        utilization_penalty_coefficient: float = 1.0,
        target_buffer_level: int = None,
    ):
        """
        Initialize adaptive frame dropper with PID control.

        Args:
            led_buffer_capacity: Expected LED buffer capacity for normalization
            led_buffer_ewma_alpha: EWMA alpha for LED buffer level tracking
            max_drop_rate: Maximum allowed drop rate (0.66 = support up to 2x input rate)
            use_pid_controller: Whether to use PID controller (True) or legacy proportional-only (False)
            kp: Proportional gain for PID controller
            ki: Integral gain for PID controller
            kd: Derivative gain for PID controller
            utilization_penalty_coefficient: Coefficient for wait time penalty in PID error calculation
            target_buffer_level: Target buffer level for PID controller (defaults to capacity if None)
        """
        # Configuration parameters
        self.led_buffer_capacity = led_buffer_capacity
        self.led_buffer_ewma_alpha = led_buffer_ewma_alpha
        self.max_drop_rate = max_drop_rate
        self.use_pid_controller = use_pid_controller

        # PID controller parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.utilization_penalty_coefficient = utilization_penalty_coefficient

        # State variables
        self.target_drop_rate = 0.0
        self.led_buffer_level_ewma = 0.0

        # Timer-based EWMA tracking
        self.ewma_update_interval = 0.05  # Update every 50ms for true average tracking
        self.last_ewma_update_time = 0.0
        self.current_buffer_size = 0  # Track current buffer size for timer updates
        
        # Adaptive EWMA state (starts "empty" and builds up weight)
        self.ewma_weighted_sum = 0.0  # Numerator: sum of α*(1-α)^i * x_i for i>=0
        self.ewma_total_weight = 0.0  # Denominator: sum of α*(1-α)^i for i>=0

        # PID controller state
        self.target_buffer_level = float(
            target_buffer_level if target_buffer_level is not None else led_buffer_capacity
        )
        self.error_integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

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
            f"max_drop_rate={max_drop_rate:.2f}, "
            f"use_pid={'PID' if use_pid_controller else 'Proportional'}, "
            f"gains=({kp:.2f}, {ki:.2f}, {kd:.2f})"
        )

    def should_drop_frame(
        self, frame_timestamp: float, led_buffer_size: int, renderer_state: str, wait_time_penalty: float = 0.0
    ) -> bool:
        """
        Determine if a frame should be dropped based on direct scaling strategy.

        Args:
            frame_timestamp: Timestamp of the frame being considered
            led_buffer_size: Current LED buffer occupancy
            renderer_state: Current renderer state (only update EWMA when PLAYING)
            wait_time_penalty: Wait time EWMA for LED buffer space (utilization penalty)

        Returns:
            True if frame should be dropped, False otherwise
        """
        # Log every call to diagnose issues
        logger.debug(
            f"Adaptive frame dropper called: renderer_state={renderer_state}, led_buffer_size={led_buffer_size}, ewma={self.led_buffer_level_ewma:.1f}"
        )

        # Always update current buffer size for timer-based EWMA
        self.current_buffer_size = led_buffer_size

        # Update EWMA and calculate drop rate
        # Both controllers only active when renderer is PLAYING
        if renderer_state.upper() == "PLAYING":
            logger.debug("Updating EWMA and calculating drop rate")
            self._update_led_buffer_ewma_timer_based(frame_timestamp, renderer_state)

            # Calculate drop rate using either PID or legacy proportional control
            if self.use_pid_controller:
                self._calculate_drop_rate_pid(frame_timestamp, wait_time_penalty)
            else:
                self._calculate_drop_rate_legacy()
            # Determine if we should drop this frame
            should_drop = self._should_drop_probabilistic()
            logger.debug(f"Drop decision: should_drop={should_drop}, target_rate={self.target_drop_rate:.3f}")
        else:
            # Legacy controller when not playing: never drop frames
            should_drop = False
            logger.debug(f"Legacy controller - Renderer not PLAYING ({renderer_state}) - not dropping frame")

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

    def _update_led_buffer_ewma_timer_based(self, current_time: float, renderer_state: str) -> None:
        """Update LED buffer level EWMA on a timer basis with adaptive initialization."""
        # Check if it's time to update EWMA (every 50ms)
        if current_time - self.last_ewma_update_time >= self.ewma_update_interval:
            old_ewma = self.led_buffer_level_ewma

            # Adaptive EWMA: Build up weighted sum and total weight incrementally
            # This avoids assuming infinite past data and responds faster at startup
            
            # Add current observation with weight α
            self.ewma_weighted_sum = (
                (1 - self.led_buffer_ewma_alpha) * self.ewma_weighted_sum + 
                self.led_buffer_ewma_alpha * self.current_buffer_size
            )
            self.ewma_total_weight = (
                (1 - self.led_buffer_ewma_alpha) * self.ewma_total_weight + 
                self.led_buffer_ewma_alpha
            )
            
            # Calculate adaptive EWMA = weighted_sum / total_weight
            if self.ewma_total_weight > 0:
                self.led_buffer_level_ewma = self.ewma_weighted_sum / self.ewma_total_weight
            else:
                self.led_buffer_level_ewma = float(self.current_buffer_size)

            self.last_ewma_update_time = current_time

            logger.debug(
                f"Adaptive EWMA updated: {old_ewma:.2f} -> {self.led_buffer_level_ewma:.2f} "
                f"(buffer_size={self.current_buffer_size}, weight={self.ewma_total_weight:.3f})"
            )

    def _calculate_drop_rate_legacy(self) -> None:
        """Calculate drop rate directly from buffer occupancy using legacy scaling approach."""
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
            f"Legacy drop rate: {old_target:.3f} -> {self.target_drop_rate:.3f} "
            f"(buffer_ewma={self.led_buffer_level_ewma:.1f}, "
            f"normalized_occupancy={normalized_occupancy:.3f})"
        )

    def _calculate_drop_rate_pid(self, current_time: float, wait_time_penalty: float = 0.0) -> None:
        """Calculate drop rate using PID controller to maintain target buffer level."""
        old_target = self.target_drop_rate

        # Calculate error: actual_level - target_level
        # Positive error means buffer is above target (need MORE dropping to drain buffer)
        # Negative error means buffer is below target (need LESS dropping to let buffer fill)
        buffer_error = self.led_buffer_level_ewma - self.target_buffer_level

        # Add utilization penalty: High wait times indicate underutilization (wasted optimization capacity)
        # This creates positive error that drives drop rate down, letting more frames through
        utilization_penalty = wait_time_penalty * self.utilization_penalty_coefficient

        # Combined error: buffer level error + utilization penalty
        error = buffer_error + utilization_penalty

        # Initialize time tracking for derivative term
        if self.previous_time is None:
            self.previous_time = current_time
            self.previous_error = error
            # Don't update drop rate on first call, need time differential
            logger.debug(f"PID controller initialized at t={current_time:.3f}, error={error:.2f}")
            return

        # Calculate time differential
        dt = current_time - self.previous_time
        if dt <= 0:
            logger.debug(f"Invalid time differential dt={dt:.6f}, skipping PID update")
            return

        # Proportional term: directly proportional to current error
        p_term = self.kp * error

        # Integral term: accumulate error over time (with anti-windup)
        # Only accumulate if Ki > 0 to avoid confusion in logs
        if self.ki > 0:
            self.error_integral += error * dt
            # Anti-windup: clamp integral based on buffer capacity, not Ki value
            # Allow integral to accumulate error equivalent to ~5x buffer capacity
            max_integral = 5.0 * self.led_buffer_capacity  # Allow substantial integral buildup
            self.error_integral = max(-max_integral, min(max_integral, self.error_integral))
            i_term = self.ki * self.error_integral
        else:
            # Ki = 0, so no integral accumulation
            self.error_integral = 0.0
            i_term = 0.0

        # Derivative term: rate of change of error
        error_derivative = (error - self.previous_error) / dt
        d_term = self.kd * error_derivative

        # Combine PID terms to get control output
        # More positive = need less dropping, more negative = need more dropping
        pid_output = p_term + i_term + d_term

        # Convert PID output to drop rate
        # Scale PID output to drop rate range [0, max_drop_rate]
        # When error > 0 (buffer above target), we want FEWER drops to slow production and drain buffer
        # When error < 0 (buffer below target), we want MORE drops to speed production and fill buffer

        # Normalize PID output by buffer capacity to get proportional response
        normalized_pid = pid_output / self.led_buffer_capacity

        # Map to drop rate: center at zero for asymmetric control
        # When buffer below target: positive drop rate (can only increase dropping)
        # When buffer above target: negative drop rate → clamped to 0% (natural balance)
        # Positive PID output (buffer above target) → FEWER drops (MINUS sign)
        # Negative PID output (buffer below target) → MORE drops (MINUS sign)
        center_drop_rate = 0.0  # Zero center point for asymmetric control
        scale_factor = self.max_drop_rate / 2  # 0.33 for max_drop_rate=0.66 - allows full range

        self.target_drop_rate = center_drop_rate - (normalized_pid * scale_factor)  # Note: MINUS sign (correct)

        # Clamp to valid range [0, max_drop_rate]
        self.target_drop_rate = max(0.0, min(self.max_drop_rate, self.target_drop_rate))

        # Update state for next iteration
        self.previous_error = error
        self.previous_time = current_time
        self.rate_calculations += 1

        # Comprehensive PID logging
        logger.info(
            f"PID drop rate: {old_target:.3f} -> {self.target_drop_rate:.3f} "
            f"(buffer_error={buffer_error:.2f}, util_penalty={utilization_penalty:.3f}, total_error={error:.2f}, "
            f"P={p_term:.3f}, I={i_term:.3f}, D={d_term:.3f}, "
            f"pid_output={pid_output:.3f}, dt={dt:.3f}s, buffer_ewma={self.led_buffer_level_ewma:.1f})"
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
            "pid_state": {
                "error_integral": self.error_integral,
                "previous_error": self.previous_error,
                "target_buffer_level": self.target_buffer_level,
            },
            "config": {
                "led_buffer_capacity": self.led_buffer_capacity,
                "led_buffer_ewma_alpha": self.led_buffer_ewma_alpha,
                "max_drop_rate": self.max_drop_rate,
                "use_pid_controller": self.use_pid_controller,
                "kp": self.kp,
                "ki": self.ki,
                "kd": self.kd,
                "utilization_penalty_coefficient": self.utilization_penalty_coefficient,
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

        # Reset timer-based EWMA state
        self.last_ewma_update_time = 0.0
        self.current_buffer_size = 0
        
        # Reset adaptive EWMA state
        self.ewma_weighted_sum = 0.0
        self.ewma_total_weight = 0.0

        # Reset PID controller state
        self.error_integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

        logger.info("AdaptiveFrameDropper statistics and PID state reset")

    def update_config(
        self,
        led_buffer_capacity: Optional[int] = None,
        led_buffer_ewma_alpha: Optional[float] = None,
        max_drop_rate: Optional[float] = None,
        use_pid_controller: Optional[bool] = None,
        kp: Optional[float] = None,
        ki: Optional[float] = None,
        kd: Optional[float] = None,
        utilization_penalty_coefficient: Optional[float] = None,
    ) -> None:
        """
        Update configuration parameters.

        Args:
            led_buffer_capacity: New buffer capacity for normalization (optional)
            led_buffer_ewma_alpha: New LED buffer EWMA alpha (optional)
            max_drop_rate: New maximum drop rate (optional)
            use_pid_controller: Whether to use PID controller (optional)
            kp: Proportional gain (optional)
            ki: Integral gain (optional)
            kd: Derivative gain (optional)
            utilization_penalty_coefficient: Coefficient for wait time penalty (optional)
        """
        if led_buffer_capacity is not None:
            self.led_buffer_capacity = led_buffer_capacity
            self.target_buffer_level = float(led_buffer_capacity)  # Update PID target
        if led_buffer_ewma_alpha is not None:
            self.led_buffer_ewma_alpha = led_buffer_ewma_alpha
        if max_drop_rate is not None:
            self.max_drop_rate = max_drop_rate
        if use_pid_controller is not None:
            if self.use_pid_controller != use_pid_controller:
                # Reset PID state when switching controller type
                self.reset_stats()
            self.use_pid_controller = use_pid_controller
        if kp is not None:
            self.kp = kp
        if ki is not None:
            old_ki = self.ki
            self.ki = ki
            if ki == 0.0:
                # Always reset integral term when using proportional-only control
                self.error_integral = 0.0
                self.previous_error = 0.0
                self.previous_time = None
                logger.info(f"Reset PID state for proportional-only control (Ki changed {old_ki} -> {ki})")
        if kd is not None:
            self.kd = kd
        if utilization_penalty_coefficient is not None:
            self.utilization_penalty_coefficient = utilization_penalty_coefficient

        logger.info(f"AdaptiveFrameDropper configuration updated: {self.get_stats()['config']}")
