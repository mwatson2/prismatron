"""
LED Transition Effect Wrappers.

This module provides LedEffect wrapper classes for LED transitions (fade, random)
that integrate with the unified LED effect framework. These wrappers delegate to
the existing LED transition implementations while providing the LedEffect interface.
"""

import logging
from typing import Any, Dict, Optional

import cupy as cp
import numpy as np

from ..transitions.led_fade_transition import LEDFadeTransition
from ..transitions.led_random_transition import LEDRandomTransition
from .led_effect import LedEffect

logger = logging.getLogger(__name__)


class FadeInEffect(LedEffect):
    """
    Fade-in transition effect using LED fade transition.

    Fades LED brightness from min_brightness to full brightness over the
    specified duration. Uses existing LEDFadeTransition implementation.
    """

    def __init__(
        self, start_time: float, duration: float, curve: str = "linear", min_brightness: float = 0.0, **kwargs
    ):
        """
        Initialize fade-in effect.

        Args:
            start_time: Frame timestamp when effect starts (from frame timeline)
            duration: Fade duration in seconds
            curve: Interpolation curve ("linear", "ease-in", "ease-out", "ease-in-out")
            min_brightness: Starting brightness level [0.0, 1.0]
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(start_time=start_time, duration=duration, **kwargs)

        # Create LED fade transition instance
        self._transition = LEDFadeTransition()

        # Build transition configuration
        self._transition_config = {
            "type": "led_fade",
            "parameters": {
                "duration": duration,
                "curve": curve,
                "min_brightness": min_brightness,
            },
        }

        # Store parameters for info
        self.curve = curve
        self.min_brightness = min_brightness

        logger.debug(
            f"Created FadeInEffect: start={start_time:.3f}s, duration={duration:.2f}s, "
            f"curve={curve}, min_brightness={min_brightness:.2f}"
        )

    def apply(self, led_values: np.ndarray, frame_timestamp: float) -> bool:
        """
        Apply fade-in effect to LED values.

        Args:
            led_values: LED values to modify (led_count, 3) in range [0, 255]
            frame_timestamp: Current frame timestamp (from frame timeline)

        Returns:
            True if effect is complete and should be removed
        """
        self.frame_count += 1

        # Convert frame timeline timestamp to item-relative timestamp
        # item_timestamp is relative to effect start
        item_timestamp = frame_timestamp - self.start_time

        # Skip if not yet started
        if item_timestamp < 0:
            return False

        # Clamp to duration to ensure we apply final state
        duration = self.duration or 0.0
        if item_timestamp > duration:
            item_timestamp = duration

        # Convert to cupy if needed for GPU processing
        if isinstance(led_values, np.ndarray):
            led_values_gpu = cp.asarray(led_values)
        else:
            led_values_gpu = led_values

        # Apply fade transition
        result = self._transition.apply_led_transition(
            led_values=led_values_gpu,
            timestamp=item_timestamp,
            item_duration=duration,
            transition_config=self._transition_config,
            direction="in",
        )

        # Convert back to numpy if input was numpy
        if isinstance(led_values, np.ndarray):
            led_values[:] = cp.asnumpy(result)
        else:
            led_values[:] = result

        # Check if effect is complete after applying
        return self.is_complete(frame_timestamp)

    def get_info(self) -> Dict[str, Any]:
        """Get fade-in effect information."""
        info = super().get_info()
        info.update(
            {
                "curve": self.curve,
                "min_brightness": self.min_brightness,
            }
        )
        return info


class FadeOutEffect(LedEffect):
    """
    Fade-out transition effect using LED fade transition.

    Fades LED brightness from full brightness to min_brightness over the
    specified duration. Uses existing LEDFadeTransition implementation.
    """

    def __init__(
        self, start_time: float, duration: float, curve: str = "linear", min_brightness: float = 0.0, **kwargs
    ):
        """
        Initialize fade-out effect.

        Args:
            start_time: Frame timestamp when effect starts (from frame timeline)
            duration: Fade duration in seconds
            curve: Interpolation curve ("linear", "ease-in", "ease-out", "ease-in-out")
            min_brightness: Ending brightness level [0.0, 1.0]
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(start_time=start_time, duration=duration, **kwargs)

        # Create LED fade transition instance
        self._transition = LEDFadeTransition()

        # Build transition configuration
        self._transition_config = {
            "type": "led_fade",
            "parameters": {
                "duration": duration,
                "curve": curve,
                "min_brightness": min_brightness,
            },
        }

        # Store parameters for info
        self.curve = curve
        self.min_brightness = min_brightness

        logger.debug(
            f"Created FadeOutEffect: start={start_time:.3f}s, duration={duration:.2f}s, "
            f"curve={curve}, min_brightness={min_brightness:.2f}"
        )

    def apply(self, led_values: np.ndarray, frame_timestamp: float) -> bool:
        """
        Apply fade-out effect to LED values.

        Args:
            led_values: LED values to modify (led_count, 3) in range [0, 255]
            frame_timestamp: Current frame timestamp (from frame timeline)

        Returns:
            True if effect is complete and should be removed
        """
        self.frame_count += 1

        # Convert frame timeline timestamp to item-relative timestamp
        item_timestamp = frame_timestamp - self.start_time

        # Skip if not yet started
        if item_timestamp < 0:
            return False

        # Clamp to duration to ensure we apply final state
        duration = self.duration or 0.0
        if item_timestamp > duration:
            item_timestamp = duration

        # Convert to cupy if needed for GPU processing
        if isinstance(led_values, np.ndarray):
            led_values_gpu = cp.asarray(led_values)
        else:
            led_values_gpu = led_values

        # Apply fade transition
        result = self._transition.apply_led_transition(
            led_values=led_values_gpu,
            timestamp=item_timestamp,
            item_duration=duration,
            transition_config=self._transition_config,
            direction="out",
        )

        # Convert back to numpy if input was numpy
        if isinstance(led_values, np.ndarray):
            led_values[:] = cp.asnumpy(result)
        else:
            led_values[:] = result

        # Check if effect is complete after applying
        return self.is_complete(frame_timestamp)

    def get_info(self) -> Dict[str, Any]:
        """Get fade-out effect information."""
        info = super().get_info()
        info.update(
            {
                "curve": self.curve,
                "min_brightness": self.min_brightness,
            }
        )
        return info


class RandomInEffect(LedEffect):
    """
    Random-in transition effect using LED random transition.

    Lights up LEDs in random order from black to full brightness over the
    specified duration. Uses existing LEDRandomTransition implementation.
    """

    def __init__(
        self,
        start_time: float,
        duration: float,
        leds_per_frame: int = 10,
        fade_tail: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize random-in effect.

        Args:
            start_time: Frame timestamp when effect starts (from frame timeline)
            duration: Random transition duration in seconds
            leds_per_frame: Number of LEDs to light per frame update
            fade_tail: Whether to apply fade tail effect to recently lit LEDs
            seed: Random seed for reproducible permutation patterns
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(start_time=start_time, duration=duration, **kwargs)

        # Create LED random transition instance
        self._transition = LEDRandomTransition()

        # Build transition configuration
        self._transition_config = {
            "type": "led_random",
            "parameters": {
                "duration": duration,
                "leds_per_frame": leds_per_frame,
                "fade_tail": fade_tail,
                "seed": seed,
            },
        }

        # Store parameters for info
        self.leds_per_frame = leds_per_frame
        self.fade_tail = fade_tail
        self.seed = seed

        logger.debug(
            f"Created RandomInEffect: start={start_time:.3f}s, duration={duration:.2f}s, "
            f"leds_per_frame={leds_per_frame}, fade_tail={fade_tail}, seed={seed}"
        )

    def apply(self, led_values: np.ndarray, frame_timestamp: float) -> bool:
        """
        Apply random-in effect to LED values.

        Args:
            led_values: LED values to modify (led_count, 3) in range [0, 255]
            frame_timestamp: Current frame timestamp (from frame timeline)

        Returns:
            True if effect is complete and should be removed
        """
        self.frame_count += 1

        # Convert frame timeline timestamp to item-relative timestamp
        item_timestamp = frame_timestamp - self.start_time

        # Skip if not yet started
        if item_timestamp < 0:
            return False

        # Clamp to duration to ensure we apply final state
        duration = self.duration or 0.0
        if item_timestamp > duration:
            item_timestamp = duration

        # Convert to cupy if needed for GPU processing
        if isinstance(led_values, np.ndarray):
            led_values_gpu = cp.asarray(led_values)
        else:
            led_values_gpu = led_values

        # Apply random transition
        result = self._transition.apply_led_transition(
            led_values=led_values_gpu,
            timestamp=item_timestamp,
            item_duration=duration,
            transition_config=self._transition_config,
            direction="in",
        )

        # Convert back to numpy if input was numpy
        if isinstance(led_values, np.ndarray):
            led_values[:] = cp.asnumpy(result)
        else:
            led_values[:] = result

        # Check if effect is complete after applying
        return self.is_complete(frame_timestamp)

    def get_info(self) -> Dict[str, Any]:
        """Get random-in effect information."""
        info = super().get_info()
        info.update(
            {
                "leds_per_frame": self.leds_per_frame,
                "fade_tail": self.fade_tail,
                "seed": self.seed,
            }
        )
        return info


class RandomOutEffect(LedEffect):
    """
    Random-out transition effect using LED random transition.

    Blanks out LEDs in random order from full brightness to black over the
    specified duration. Uses existing LEDRandomTransition implementation.
    """

    def __init__(
        self,
        start_time: float,
        duration: float,
        leds_per_frame: int = 10,
        fade_tail: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize random-out effect.

        Args:
            start_time: Frame timestamp when effect starts (from frame timeline)
            duration: Random transition duration in seconds
            leds_per_frame: Number of LEDs to blank per frame update
            fade_tail: Whether to apply fade tail effect to recently blanked LEDs
            seed: Random seed for reproducible permutation patterns
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(start_time=start_time, duration=duration, **kwargs)

        # Create LED random transition instance
        self._transition = LEDRandomTransition()

        # Build transition configuration
        self._transition_config = {
            "type": "led_random",
            "parameters": {
                "duration": duration,
                "leds_per_frame": leds_per_frame,
                "fade_tail": fade_tail,
                "seed": seed,
            },
        }

        # Store parameters for info
        self.leds_per_frame = leds_per_frame
        self.fade_tail = fade_tail
        self.seed = seed

        logger.debug(
            f"Created RandomOutEffect: start={start_time:.3f}s, duration={duration:.2f}s, "
            f"leds_per_frame={leds_per_frame}, fade_tail={fade_tail}, seed={seed}"
        )

    def apply(self, led_values: np.ndarray, frame_timestamp: float) -> bool:
        """
        Apply random-out effect to LED values.

        Args:
            led_values: LED values to modify (led_count, 3) in range [0, 255]
            frame_timestamp: Current frame timestamp (from frame timeline)

        Returns:
            True if effect is complete and should be removed
        """
        self.frame_count += 1

        # Convert frame timeline timestamp to item-relative timestamp
        item_timestamp = frame_timestamp - self.start_time

        # Skip if not yet started
        if item_timestamp < 0:
            return False

        # Clamp to duration to ensure we apply final state
        duration = self.duration or 0.0
        if item_timestamp > duration:
            item_timestamp = duration

        # Convert to cupy if needed for GPU processing
        if isinstance(led_values, np.ndarray):
            led_values_gpu = cp.asarray(led_values)
        else:
            led_values_gpu = led_values

        # Apply random transition
        result = self._transition.apply_led_transition(
            led_values=led_values_gpu,
            timestamp=item_timestamp,
            item_duration=duration,
            transition_config=self._transition_config,
            direction="out",
        )

        # Convert back to numpy if input was numpy
        if isinstance(led_values, np.ndarray):
            led_values[:] = cp.asnumpy(result)
        else:
            led_values[:] = result

        # Check if effect is complete after applying
        return self.is_complete(frame_timestamp)

    def get_info(self) -> Dict[str, Any]:
        """Get random-out effect information."""
        info = super().get_info()
        info.update(
            {
                "leds_per_frame": self.leds_per_frame,
                "fade_tail": self.fade_tail,
                "seed": self.seed,
            }
        )
        return info
