"""
LED blur transition implementation for playlist items.

This module implements LED blur-in and blur-out transitions that apply
a 1D spatial blur to the LED array, creating an effect where content
emerges from complete blur into focus or fades into complete blur.
"""

import logging
import math
from typing import Any, Dict, Tuple

import cupy as cp
import numpy as np

from .base_led_transition import BaseLEDTransition

logger = logging.getLogger(__name__)


class LEDBlurTransition(BaseLEDTransition):
    """
    LED blur transition implementation with GPU-native spatial processing.

    Applies a 1D blur along the spatial ordering of LEDs to create smooth
    blur-in and blur-out effects. The blur operates on the LED values in
    their spatial order, creating a realistic blur effect where content
    gradually comes into focus or fades into blur.

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
        Apply LED blur transition to LED values.

        Args:
            led_values: LED RGB values as cupy GPU array (led_count, 3) with values 0-255
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration with parameters
            direction: "in" for blur-in (blur to focus), "out" for blur-out (focus to blur)

        Returns:
            LED values with blur transition applied (GPU array)

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
                raise ValueError("Invalid LED blur transition parameters")

            # Check if we're in the transition region
            if not self.is_in_transition_region(timestamp, item_duration, transition_config, direction):
                return led_values  # No transition needed

            # Get transition parameters
            parameters = transition_config.get("parameters", {})
            blur_intensity = parameters.get("blur_intensity", 1.0)
            kernel_size = parameters.get("kernel_size", 5)

            # Calculate blur progress (0.0 to 1.0)
            progress = self.get_transition_progress(timestamp, item_duration, transition_config, direction)

            # Calculate blur strength based on direction
            if direction == "in":
                # Blur in: start with maximum blur, end with no blur
                blur_strength = (1.0 - progress) * blur_intensity
            elif direction == "out":
                # Blur out: start with no blur, end with maximum blur
                blur_strength = progress * blur_intensity
            else:
                raise ValueError(f"Invalid direction '{direction}', must be 'in' or 'out'")

            # Apply blur if strength is significant
            if blur_strength > 0.01:  # Only apply blur if strength is meaningful
                blurred_led = self._apply_spatial_blur(led_values, blur_strength, kernel_size)
                return blurred_led
            else:
                return led_values  # No blur needed

        except Exception as e:
            logger.error(f"Error applying LED blur transition: {e}")
            # Return original LED values on error to avoid breaking the pipeline
            return led_values

    def _apply_spatial_blur(self, led_values: cp.ndarray, blur_strength: float, kernel_size: int) -> cp.ndarray:
        """
        Apply 1D spatial blur to LED values along the spatial ordering using optimized CuPy operations.

        Args:
            led_values: LED RGB values as cupy GPU array (led_count, 3)
            blur_strength: Blur strength factor (0.0 = no blur, 1.0 = full blur)
            kernel_size: Size of the blur kernel (odd integer)

        Returns:
            Blurred LED values as cupy GPU array
        """
        try:
            # Ensure kernel size is odd and at least 3
            kernel_size = max(3, kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Create 1D Gaussian blur kernel on GPU
            sigma = blur_strength * (kernel_size / 3.0)  # Scale sigma with blur strength
            kernel = self._create_gaussian_kernel(kernel_size, sigma)

            # Convert to appropriate dtype for processing
            original_dtype = led_values.dtype
            led_float = led_values.astype(cp.float32)

            # Use CuPy's optimized 1D convolution with padding for circular effect
            # This is much faster than our manual convolution loop
            from cupyx.scipy import ndimage

            # Apply 1D convolution to each color channel using scipy.ndimage.convolve1d
            # which is GPU-optimized in CuPy
            blurred_led = cp.zeros_like(led_float)

            for channel in range(3):
                # Use mode='wrap' for circular boundary conditions (LEDs wrap around)
                blurred_led[:, channel] = ndimage.convolve1d(led_float[:, channel], kernel, axis=0, mode="wrap")

            # Mix blurred and original based on blur strength
            mixed_led = (1.0 - blur_strength) * led_float + blur_strength * blurred_led

            # Clamp values and convert back to original dtype on GPU
            if original_dtype == cp.uint8:
                mixed_led = cp.clip(mixed_led, 0, 255).astype(cp.uint8)
            else:
                # Assume float32 format in 0-1 range
                mixed_led = cp.clip(mixed_led, 0.0, 1.0).astype(cp.float32)

            return mixed_led

        except Exception as e:
            logger.error(f"Error applying spatial blur: {e}")
            return led_values

    def _create_gaussian_kernel(self, size: int, sigma: float) -> cp.ndarray:
        """
        Create a 1D Gaussian blur kernel on GPU.

        Args:
            size: Kernel size (odd integer)
            sigma: Gaussian standard deviation

        Returns:
            Normalized Gaussian kernel as cupy GPU array
        """
        # Create kernel indices centered at 0
        x = cp.arange(size, dtype=cp.float32) - (size - 1) / 2.0

        # Compute Gaussian values
        if sigma > 0:
            kernel = cp.exp(-0.5 * (x / sigma) ** 2)
        else:
            # Delta function for sigma=0 (no blur)
            kernel = cp.zeros(size, dtype=cp.float32)
            kernel[size // 2] = 1.0

        # Normalize kernel
        kernel = kernel / cp.sum(kernel)

        return kernel

    def get_transition_region(
        self, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> Tuple[float, float]:
        """
        Calculate the time region where the LED blur transition is active.

        Args:
            item_duration: Total duration of the playlist item (seconds)
            transition_config: Transition configuration including parameters
            direction: "in" for blur-in, "out" for blur-out

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
            # Blur in at the beginning of the item
            return (0.0, duration)
        elif direction == "out":
            # Blur out at the end of the item
            return (item_duration - duration, item_duration)
        else:
            raise ValueError(f"Invalid direction '{direction}', must be 'in' or 'out'")

    def is_in_transition_region(
        self, timestamp: float, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> bool:
        """
        Check if a timestamp falls within the LED blur transition region.

        Args:
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration including parameters
            direction: "in" for blur-in, "out" for blur-out

        Returns:
            True if timestamp is within the transition region, False otherwise
        """
        try:
            start_time, end_time = self.get_transition_region(item_duration, transition_config, direction)
            return start_time <= timestamp <= end_time
        except Exception as e:
            logger.warning(f"Error checking LED blur transition region: {e}")
            return False

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate LED blur transition parameters.

        Args:
            parameters: Dictionary of LED blur transition parameters

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

            # Check blur intensity
            blur_intensity = parameters.get("blur_intensity", 1.0)
            if not isinstance(blur_intensity, (int, float)):
                logger.error(f"Blur intensity must be a number, got {type(blur_intensity)}")
                return False
            if blur_intensity < 0.0 or blur_intensity > 2.0:
                logger.error(f"Blur intensity must be between 0.0 and 2.0, got {blur_intensity}")
                return False

            # Check kernel size
            kernel_size = parameters.get("kernel_size", 5)
            if not isinstance(kernel_size, int):
                logger.error(f"Kernel size must be an integer, got {type(kernel_size)}")
                return False
            if kernel_size < 3 or kernel_size > 21:
                logger.error(f"Kernel size must be between 3 and 21, got {kernel_size}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating LED blur parameters: {e}")
            return False

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for LED blur transition parameters.

        Returns:
            JSON schema dictionary for LED blur transition parameters
        """
        return {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 60.0,
                    "default": 1.0,
                    "description": "LED blur transition duration in seconds",
                },
                "blur_intensity": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 1.0,
                    "description": "Blur intensity factor (0.0 = no blur, 1.0 = normal blur, 2.0 = heavy blur)",
                },
                "kernel_size": {
                    "type": "integer",
                    "minimum": 3,
                    "maximum": 21,
                    "default": 5,
                    "description": "Blur kernel size (odd integer, larger = more blur)",
                },
            },
            "required": ["duration"],
            "additionalProperties": False,
        }
