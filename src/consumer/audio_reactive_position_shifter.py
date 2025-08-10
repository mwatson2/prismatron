#!/usr/bin/env python3
"""
Audio-Reactive LED Position Shifter for Prismatron LED Display System.

This module provides position shifting effects that respond to audio beats,
creating a "shudder" or "shake" effect that shifts LED positions left or right
in sync with music beats. Works alongside the existing brightness pulse effects.

The position shift follows the same sine wave pattern as brightness boosting,
occurring during the first quarter of each beat interval.
"""

import logging
import math
import time
from typing import Optional

logger = logging.getLogger(__name__)


class AudioReactivePositionShifter:
    """
    Audio-reactive LED position shifter that creates shudder effects on beats.

    Features:
    - Sine wave position shifting synchronized with detected beats
    - Configurable maximum shift distance (3-4 LEDs typical)
    - Can shift left, right, or alternating based on configuration
    - Works alongside brightness pulse effects
    - Follows same timing pattern as brightness boost (first quarter of beat interval)
    """

    def __init__(
        self,
        max_shift_distance: int = 3,
        shift_direction: str = "alternating",  # "left", "right", "alternating"
        control_state=None,
        audio_beat_analyzer=None,
        enabled: bool = True,
    ):
        """
        Initialize position shifter.

        Args:
            max_shift_distance: Maximum number of LEDs to shift (3-4 typical)
            shift_direction: Shift direction ("left", "right", "alternating")
            control_state: ControlState instance for audio reactive settings
            audio_beat_analyzer: AudioBeatAnalyzer instance for beat state access
            enabled: Whether position shifting is enabled
        """
        self.max_shift_distance = max_shift_distance
        self.shift_direction = shift_direction
        self.enabled = enabled

        # Audio reactive components
        self._control_state = control_state
        self._audio_beat_analyzer = audio_beat_analyzer

        # Internal state for alternating direction
        self._last_shift_direction = 1  # 1 for right, -1 for left
        self._last_beat_count = 0

        logger.info(
            f"AudioReactivePositionShifter initialized: max_shift={max_shift_distance}, "
            f"direction={shift_direction}, enabled={enabled}"
        )

    def calculate_position_offset(self, current_time: float, led_count: int) -> int:
        """
        Calculate LED position offset based on beat timing for audio-reactive effects.

        Implements a sine wave position shift during the first quarter of each beat interval.
        The offset follows the same timing pattern as brightness boost but affects position.

        Args:
            current_time: Current system time in seconds
            led_count: Total number of LEDs (for wraparound bounds checking)

        Returns:
            Position offset in LED positions (negative = left shift, positive = right shift)
        """
        if not self.enabled:
            return 0

        # Check if audio reactive effects are enabled
        if not self._control_state or not self._audio_beat_analyzer:
            return 0

        try:
            # Get current control state to check if audio reactive is enabled
            status = self._control_state.get_status()
            if not status or not status.audio_reactive_enabled or not status.audio_enabled:
                return 0

            # Get audio beat state
            beat_state = self._audio_beat_analyzer.get_current_state()
            if not beat_state or not beat_state.is_active:
                return 0

            # Calculate inter-beat duration from current BPM
            if beat_state.current_bpm <= 0:
                return 0
            beat_duration = 60.0 / beat_state.current_bpm

            # Determine reference beat time (use prediction if beat has started, otherwise last detected beat)
            audio_time = current_time - self._audio_beat_analyzer.start_time  # Convert to audio timeline
            predicted_next_beat = self._audio_beat_analyzer.predict_next_beat(audio_time)

            if predicted_next_beat < audio_time:
                # Beat has started, use predicted time
                reference_beat_time = predicted_next_beat
            else:
                # Use last detected beat time
                reference_beat_time = beat_state.last_beat_time

            # Calculate time since beat start
            t = audio_time - reference_beat_time

            # Apply sine wave shift for first quarter of beat interval
            boost_duration = 0.25 * beat_duration
            if 0 <= t <= boost_duration:
                # Update shift direction if we're on a new beat (for alternating mode)
                if beat_state.beat_count != self._last_beat_count:
                    self._last_beat_count = beat_state.beat_count
                    if self.shift_direction == "alternating":
                        self._last_shift_direction *= -1  # Flip direction

                # Determine shift direction multiplier
                direction_multiplier = self._get_direction_multiplier()

                # Sine wave shift (0 to max_shift_distance)
                # Uses same sine wave pattern as brightness but for position
                shift_magnitude = self.max_shift_distance * math.sin(t * math.pi / boost_duration)
                offset = int(shift_magnitude * direction_multiplier)

                # Ensure offset is within bounds
                if led_count > 0:
                    offset = offset % led_count

                return offset
            else:
                return 0  # No shift outside beat window

        except Exception as e:
            logger.warning(f"Error calculating position offset: {e}")
            return 0

    def _get_direction_multiplier(self) -> int:
        """
        Get direction multiplier based on current shift direction setting.

        Returns:
            -1 for left shift, 1 for right shift
        """
        if self.shift_direction == "left":
            return -1
        elif self.shift_direction == "right":
            return 1
        elif self.shift_direction == "alternating":
            return self._last_shift_direction
        else:
            logger.warning(f"Unknown shift direction: {self.shift_direction}, defaulting to right")
            return 1

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable position shifting."""
        self.enabled = enabled
        logger.info(f"Position shifter {'enabled' if enabled else 'disabled'}")

    def set_max_shift_distance(self, distance: int) -> None:
        """Update maximum shift distance."""
        if distance < 0:
            distance = 0
        elif distance > 10:  # Reasonable upper limit
            distance = 10
            logger.warning(f"Clamping max shift distance to {distance}")

        self.max_shift_distance = distance
        logger.info(f"Max shift distance set to {distance}")

    def set_shift_direction(self, direction: str) -> None:
        """Update shift direction."""
        valid_directions = ["left", "right", "alternating"]
        if direction not in valid_directions:
            logger.warning(f"Invalid direction '{direction}', must be one of {valid_directions}")
            return

        self.shift_direction = direction
        logger.info(f"Shift direction set to {direction}")

    def get_current_offset(self, led_count: int) -> int:
        """
        Get current position offset for immediate use.

        Args:
            led_count: Total number of LEDs

        Returns:
            Current position offset
        """
        return self.calculate_position_offset(time.time(), led_count)

    def get_status(self) -> dict:
        """
        Get current status for debugging/monitoring.

        Returns:
            Dictionary with current shifter status
        """
        return {
            "enabled": self.enabled,
            "max_shift_distance": self.max_shift_distance,
            "shift_direction": self.shift_direction,
            "last_shift_direction": self._last_shift_direction,
            "last_beat_count": self._last_beat_count,
        }


# Example usage and testing
if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Create position shifter (without audio components for basic testing)
    shifter = AudioReactivePositionShifter(max_shift_distance=3, shift_direction="alternating", enabled=True)

    # Test basic functionality
    led_count = 100

    print("Testing position shifter (without audio input)...")
    for i in range(10):
        offset = shifter.get_current_offset(led_count)
        status = shifter.get_status()
        print(f"Frame {i}: offset={offset}, status={status}")
        time.sleep(0.1)

    print("Position shifter test completed")
