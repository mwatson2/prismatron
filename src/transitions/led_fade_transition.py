"""
LED fade transition implementation for playlist items.

This module implements LED fade-in and fade-out transitions that directly
adjust the brightness of LED values after optimization, providing a more
efficient alternative to image-based fade transitions.
"""

import logging
import math
from typing import Any, Dict, Tuple, Union

import cupy as cp
import numpy as np

from .base_led_transition import BaseLEDTransition

logger = logging.getLogger(__name__)


class LEDFadeTransition(BaseLEDTransition):
    """
    LED fade transition implementation with GPU-native support.

    Provides smooth fade-in and fade-out effects by directly adjusting LED
    brightness values over a specified duration. Since this operates on the
    final optimized LED values rather than the source image, it's computationally
    more efficient and provides equivalent visual results for fade effects.

    Supports different interpolation curves for natural-looking transitions.
    GPU-native processing maintains LED value format throughout the pipeline.
    """

    def apply_led_transition(
        self,
        led_values: cp.ndarray,
        timestamp: float,
        item_duration: float,
        transition_config: Dict[str, Any],
        direction: str,
    ) -> cp.ndarray:
        """
        Apply LED fade transition to LED values.

        Args:
            led_values: LED RGB values as cupy GPU array (led_count, 3) with values 0-255
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration with parameters
            direction: "in" for fade-in, "out" for fade-out

        Returns:
            LED values with fade transition applied (GPU array)

        Raises:
            ValueError: If parameters are invalid or led_values is not on GPU
            RuntimeError: If transition processing fails
        """
        try:
            # Validate inputs - led_values must be GPU array
            if not isinstance(led_values, cp.ndarray):
                logger.error(f"Expected GPU cupy array, got {type(led_values)}")
                raise ValueError(f"LED values must be cupy GPU array, got {type(led_values)}")

            if led_values.ndim != 2 or led_values.shape[1] != 3:
                raise ValueError(f"Expected LED values with shape (led_count, 3), got {led_values.shape}")

            if not self.validate_parameters(transition_config.get("parameters", {})):
                raise ValueError("Invalid LED fade transition parameters")

            # Check if we're in the transition region
            if not self.is_in_transition_region(timestamp, item_duration, transition_config, direction):
                return led_values  # No transition needed

            # Get transition parameters
            parameters = transition_config.get("parameters", {})
            curve_type = parameters.get("curve", "linear")
            min_brightness = parameters.get("min_brightness", 0.0)

            # Calculate fade progress (0.0 to 1.0)
            progress = self.get_transition_progress(timestamp, item_duration, transition_config, direction)

            # Apply curve transformation
            curve_progress = self._apply_curve(progress, curve_type)

            # Calculate fade factor based on direction
            if direction == "in":
                # Fade in: start at min_brightness, end at full brightness
                fade_factor = min_brightness + (1.0 - min_brightness) * curve_progress
            elif direction == "out":
                # Fade out: start at full brightness, end at min_brightness
                fade_factor = 1.0 - (1.0 - min_brightness) * curve_progress
            else:
                raise ValueError(f"Invalid direction '{direction}', must be 'in' or 'out'")

            # Apply fade to LED values - GPU-only processing
            # Convert to float32 for calculation, then back to original dtype
            original_dtype = led_values.dtype
            led_float = led_values.astype(cp.float32)
            faded_led = led_float * fade_factor

            # Clamp values and convert back to original dtype on GPU
            if original_dtype == cp.uint8:
                faded_led = cp.clip(faded_led, 0, 255).astype(cp.uint8)
            else:
                # Assume float32 format in 0-1 range
                faded_led = cp.clip(faded_led, 0.0, 1.0).astype(cp.float32)

            return faded_led

        except Exception as e:
            logger.error(f"Error applying LED fade transition: {e}")
            # Return original LED values on error to avoid breaking the pipeline
            return led_values

    def get_transition_region(
        self, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> Tuple[float, float]:
        """
        Calculate the time region where the LED fade transition is active.

        Args:
            item_duration: Total duration of the playlist item (seconds)
            transition_config: Transition configuration including parameters
            direction: "in" for fade-in, "out" for fade-out

        Returns:
            Tuple of (start_time, end_time) in seconds from item start

        Raises:
            ValueError: If parameters are invalid
        """
        parameters = transition_config.get("parameters", {})
        duration = parameters.get("duration", 1.0)

        # Clamp duration to item duration
        duration = min(duration, item_duration)

        if direction == "in":
            # Fade in at the beginning of the item
            return (0.0, duration)
        elif direction == "out":
            # Fade out at the end of the item
            return (item_duration - duration, item_duration)
        else:
            raise ValueError(f"Invalid direction '{direction}', must be 'in' or 'out'")

    def is_in_transition_region(
        self, timestamp: float, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> bool:
        """
        Check if a timestamp falls within the LED fade transition region.

        Args:
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration including parameters
            direction: "in" for fade-in, "out" for fade-out

        Returns:
            True if timestamp is within the transition region, False otherwise
        """
        try:
            start_time, end_time = self.get_transition_region(item_duration, transition_config, direction)
            return start_time <= timestamp <= end_time
        except Exception as e:
            logger.warning(f"Error checking LED transition region: {e}")
            return False

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate LED fade transition parameters.

        Args:
            parameters: Dictionary of LED fade transition parameters

        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # Check duration
            duration = parameters.get("duration", 1.0)
            if not isinstance(duration, (int, float)):
                logger.error(f"Duration must be a number, got {type(duration)}")
                return False
            if duration <= 0:
                logger.error(f"Duration must be positive, got {duration}")
                return False
            if duration > 60.0:  # Reasonable upper limit
                logger.error(f"Duration too large: {duration} seconds (max 60)")
                return False

            # Check curve type
            curve = parameters.get("curve", "linear")
            if curve not in ["linear", "ease-in", "ease-out", "ease-in-out"]:
                logger.error(f"Invalid curve type '{curve}'")
                return False

            # Check min_brightness
            min_brightness = parameters.get("min_brightness", 0.0)
            if not isinstance(min_brightness, (int, float)):
                logger.error(f"min_brightness must be a number, got {type(min_brightness)}")
                return False
            if min_brightness < 0.0 or min_brightness > 1.0:
                logger.error(f"min_brightness must be between 0.0 and 1.0, got {min_brightness}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating LED fade parameters: {e}")
            return False

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for LED fade transition parameters.

        Returns:
            JSON schema dictionary for LED fade transition parameters
        """
        return {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 60.0,
                    "default": 1.0,
                    "description": "LED fade duration in seconds",
                },
                "curve": {
                    "type": "string",
                    "enum": ["linear", "ease-in", "ease-out", "ease-in-out"],
                    "default": "linear",
                    "description": "Interpolation curve for LED fade transition",
                },
                "min_brightness": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                    "description": "Minimum brightness level (0.0 = full fade to black)",
                },
            },
            "required": ["duration"],
            "additionalProperties": False,
        }

    def _apply_curve(self, progress: float, curve_type: str) -> float:
        """
        Apply interpolation curve to linear progress.

        Args:
            progress: Linear progress value between 0.0 and 1.0
            curve_type: Type of curve to apply

        Returns:
            Transformed progress value with curve applied
        """
        progress = max(0.0, min(1.0, progress))  # Clamp to valid range

        if curve_type == "linear":
            return progress
        elif curve_type == "ease-in":
            # Quadratic ease-in (slow start, fast end)
            return progress * progress
        elif curve_type == "ease-out":
            # Quadratic ease-out (fast start, slow end)
            return 1.0 - (1.0 - progress) * (1.0 - progress)
        elif curve_type == "ease-in-out":
            # Cubic ease-in-out (slow start and end, fast middle)
            if progress < 0.5:
                return 2.0 * progress * progress
            else:
                return 1.0 - 2.0 * (1.0 - progress) * (1.0 - progress)
        else:
            # Fallback to linear
            logger.warning(f"Unknown curve type '{curve_type}', using linear")
            return progress
