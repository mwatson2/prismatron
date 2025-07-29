"""
LED random transition implementation for playlist items.

This module implements LED random-in and random-out transitions that light up
or blank out LEDs in a random permutation order, creating dynamic sparkle-like
effects during transitions.
"""

import logging
import math
from typing import Any, Dict, Tuple

import cupy as cp
import numpy as np

from .base_led_transition import BaseLEDTransition

logger = logging.getLogger(__name__)


class LEDRandomTransition(BaseLEDTransition):
    """
    LED random transition implementation with GPU-native random processing.

    Creates random-in and random-out effects by lighting up or blanking out LEDs
    in a random permutation order. The transition remembers the LED ordering
    throughout the transition duration and processes multiple LEDs per frame
    for smooth animation.

    GPU-native processing maintains LED value format throughout the pipeline.
    """

    def __init__(self):
        """Initialize the LED random transition with state tracking."""
        super().__init__()
        # Cache for random permutations per item (to maintain consistency within an item)
        self._permutation_cache = {}
        self._last_processed_frame = {}

    def apply_led_transition(
        self,
        led_values: cp.ndarray,
        timestamp: float,
        item_duration: float,
        transition_config: Dict[str, Any],
        direction: str,
    ) -> cp.ndarray:
        """
        Apply LED random transition to LED values.

        Args:
            led_values: LED RGB values as cupy GPU array (led_count, 3) with values 0-255
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration with parameters
            direction: "in" for random-in, "out" for random-out

        Returns:
            LED values with random transition applied (GPU array)

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
                raise ValueError("Invalid LED random transition parameters")

            # Check if we're in the transition region
            if not self.is_in_transition_region(timestamp, item_duration, transition_config, direction):
                return led_values  # No transition needed

            # Get transition parameters
            parameters = transition_config.get("parameters", {})
            leds_per_frame = parameters.get("leds_per_frame", 10)
            fade_tail = parameters.get("fade_tail", True)
            seed = parameters.get("seed", 42)

            # Calculate transition progress (0.0 to 1.0)
            progress = self.get_transition_progress(timestamp, item_duration, transition_config, direction)

            # Generate or retrieve cached permutation for this transition
            cache_key = f"{direction}_{seed}_{len(led_values)}"
            if cache_key not in self._permutation_cache:
                self._permutation_cache[cache_key] = self._generate_random_permutation(len(led_values), seed)

            permutation = self._permutation_cache[cache_key]

            # Apply random transition effect
            result_led = self._apply_random_effect(
                led_values, progress, direction, permutation, leds_per_frame, fade_tail
            )

            return result_led

        except Exception as e:
            logger.error(f"Error applying LED random transition: {e}")
            # Return original LED values on error to avoid breaking the pipeline
            return led_values

    def _generate_random_permutation(self, led_count: int, seed: int) -> cp.ndarray:
        """
        Generate a random permutation of LED indices on GPU.

        Args:
            led_count: Number of LEDs
            seed: Random seed for reproducible permutations

        Returns:
            Random permutation as cupy GPU array
        """
        # Generate permutation on CPU first, then move to GPU
        np.random.seed(seed)
        permutation_cpu = np.random.permutation(led_count)
        return cp.asarray(permutation_cpu)

    def _apply_random_effect(
        self,
        led_values: cp.ndarray,
        progress: float,
        direction: str,
        permutation: cp.ndarray,
        leds_per_frame: int,
        fade_tail: bool,
    ) -> cp.ndarray:
        """
        Apply random lighting/blanking effect to LED values.

        Args:
            led_values: LED RGB values as cupy GPU array (led_count, 3)
            progress: Transition progress (0.0 to 1.0)
            direction: "in" for random-in, "out" for random-out
            permutation: Random permutation of LED indices
            leds_per_frame: Number of LEDs to process per frame
            fade_tail: Whether to apply fade tail effect

        Returns:
            LED values with random effect applied
        """
        try:
            led_count = len(led_values)
            original_dtype = led_values.dtype
            result_led = led_values.astype(cp.float32)

            # Calculate how many LEDs should be affected based on progress
            total_leds_affected = int(progress * led_count)

            if direction == "in":
                # Random in: LEDs start black and randomly light up to their target values
                # Create mask for LEDs that should be lit
                lit_mask = cp.zeros(led_count, dtype=cp.bool_)
                if total_leds_affected > 0:
                    lit_indices = permutation[:total_leds_affected]
                    lit_mask[lit_indices] = True

                # Apply fade tail effect if enabled
                if fade_tail and total_leds_affected > leds_per_frame:
                    # Recent LEDs are full brightness, older LEDs fade
                    recent_start = max(0, total_leds_affected - leds_per_frame * 3)
                    for i in range(recent_start, total_leds_affected):
                        led_idx = permutation[i]
                        # Calculate fade factor based on how recently this LED was lit
                        recency = (total_leds_affected - i) / float(leds_per_frame * 3)
                        fade_factor = max(0.3, min(1.0, recency))  # Fade from 1.0 to 0.3
                        result_led[led_idx] = result_led[led_idx] * fade_factor

                # Set unlit LEDs to black
                result_led[~lit_mask] = 0.0

            elif direction == "out":
                # Random out: LEDs start at their target values and randomly blank out
                # Create mask for LEDs that should be blanked
                blanked_mask = cp.zeros(led_count, dtype=cp.bool_)
                if total_leds_affected > 0:
                    blanked_indices = permutation[:total_leds_affected]
                    blanked_mask[blanked_indices] = True

                # Apply fade tail effect if enabled
                if fade_tail and total_leds_affected > leds_per_frame:
                    # Recent blanked LEDs fade gradually, older ones are completely black
                    recent_start = max(0, total_leds_affected - leds_per_frame * 3)
                    for i in range(recent_start, total_leds_affected):
                        led_idx = permutation[i]
                        # Calculate fade factor based on how recently this LED was blanked
                        recency = (total_leds_affected - i) / float(leds_per_frame * 3)
                        fade_factor = max(0.0, min(0.7, 1.0 - recency))  # Fade from 0.7 to 0.0
                        result_led[led_idx] = result_led[led_idx] * fade_factor

                # Set fully blanked LEDs to black
                old_blanked = total_leds_affected
                if not fade_tail:
                    result_led[blanked_mask] = 0.0
                else:
                    # Only blank the older LEDs completely
                    old_end = max(0, total_leds_affected - leds_per_frame * 3)
                    if old_end > 0:
                        old_indices = permutation[:old_end]
                        result_led[old_indices] = 0.0

            else:
                raise ValueError(f"Invalid direction '{direction}', must be 'in' or 'out'")

            # Clamp values and convert back to original dtype on GPU
            if original_dtype == cp.uint8:
                result_led = cp.clip(result_led, 0, 255).astype(cp.uint8)
            else:
                # Assume float32 format in 0-1 range
                result_led = cp.clip(result_led, 0.0, 1.0).astype(cp.float32)

            return result_led

        except Exception as e:
            logger.error(f"Error applying random effect: {e}")
            return led_values

    def get_transition_region(
        self, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> Tuple[float, float]:
        """
        Calculate the time region where the LED random transition is active.

        Args:
            item_duration: Total duration of the playlist item (seconds)
            transition_config: Transition configuration including parameters
            direction: "in" for random-in, "out" for random-out

        Returns:
            Tuple of (start_time, end_time) in seconds from item start

        Raises:
            ValueError: If parameters are invalid
        """
        parameters = transition_config.get("parameters", {})
        duration = parameters.get("duration", 2.0)

        # Clamp duration to item duration
        duration = min(duration, item_duration)

        if direction == "in":
            # Random in at the beginning of the item
            return (0.0, duration)
        elif direction == "out":
            # Random out at the end of the item
            return (item_duration - duration, item_duration)
        else:
            raise ValueError(f"Invalid direction '{direction}', must be 'in' or 'out'")

    def is_in_transition_region(
        self, timestamp: float, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> bool:
        """
        Check if a timestamp falls within the LED random transition region.

        Args:
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration including parameters
            direction: "in" for random-in, "out" for random-out

        Returns:
            True if timestamp is within the transition region, False otherwise
        """
        try:
            start_time, end_time = self.get_transition_region(item_duration, transition_config, direction)
            return start_time <= timestamp <= end_time
        except Exception as e:
            logger.warning(f"Error checking LED random transition region: {e}")
            return False

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate LED random transition parameters.

        Args:
            parameters: Dictionary of LED random transition parameters

        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # Check duration
            duration = parameters.get("duration", 2.0)
            if not isinstance(duration, (int, float)):
                logger.error(f"Duration must be a number, got {type(duration)}")
                return False
            if duration <= 0:
                logger.error(f"Duration must be positive, got {duration}")
                return False
            if duration > 60.0:  # Reasonable upper limit
                logger.error(f"Duration too large: {duration} seconds (max 60)")
                return False

            # Check LEDs per frame
            leds_per_frame = parameters.get("leds_per_frame", 10)
            if not isinstance(leds_per_frame, int):
                logger.error(f"LEDs per frame must be an integer, got {type(leds_per_frame)}")
                return False
            if leds_per_frame < 1 or leds_per_frame > 100:
                logger.error(f"LEDs per frame must be between 1 and 100, got {leds_per_frame}")
                return False

            # Check fade tail
            fade_tail = parameters.get("fade_tail", True)
            if not isinstance(fade_tail, bool):
                logger.error(f"Fade tail must be a boolean, got {type(fade_tail)}")
                return False

            # Check seed
            seed = parameters.get("seed", 42)
            if not isinstance(seed, int):
                logger.error(f"Seed must be an integer, got {type(seed)}")
                return False
            if seed < 0 or seed > 999999:
                logger.error(f"Seed must be between 0 and 999999, got {seed}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating LED random parameters: {e}")
            return False

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for LED random transition parameters.

        Returns:
            JSON schema dictionary for LED random transition parameters
        """
        return {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 60.0,
                    "default": 2.0,
                    "description": "LED random transition duration in seconds",
                },
                "leds_per_frame": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "Number of LEDs to light/blank per frame update",
                },
                "fade_tail": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to apply fade tail effect to recently changed LEDs",
                },
                "seed": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 999999,
                    "default": 42,
                    "description": "Random seed for reproducible permutation patterns",
                },
            },
            "required": ["duration"],
            "additionalProperties": False,
        }
