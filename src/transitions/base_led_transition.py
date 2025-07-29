"""
Base LED transition interface for LED value transitions.

This module defines the abstract base class that all LED transition implementations
must inherit from, providing a consistent interface for applying transitions
directly to LED values after optimization but before transmission to WLED.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import cupy as cp
import numpy as np


class BaseLEDTransition(ABC):
    """
    Abstract base class for all LED transition implementations.

    LED transitions are applied to optimized LED values (in spatial/matrix order)
    on the GPU, providing effects that are more computationally efficient than
    image-based transitions and equally effective for fade/brightness effects.
    """

    @abstractmethod
    def apply_led_transition(
        self,
        led_values: cp.ndarray,
        timestamp: float,
        item_duration: float,
        transition_config: Dict[str, Any],
        direction: str,
    ) -> cp.ndarray:
        """
        Apply transition effect to LED values.

        Args:
            led_values: LED RGB values as cupy GPU array (led_count, 3) with values 0-255
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration including type and parameters
            direction: "in" for transition_in, "out" for transition_out

        Returns:
            Modified LED values with transition applied (GPU array)

        Raises:
            ValueError: If parameters are invalid or led_values is not on GPU
            RuntimeError: If transition processing fails
        """

    @abstractmethod
    def get_transition_region(
        self, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> Tuple[float, float]:
        """
        Calculate the time region where this transition is active.

        Args:
            item_duration: Total duration of the playlist item (seconds)
            transition_config: Transition configuration including parameters
            direction: "in" for transition_in, "out" for transition_out

        Returns:
            Tuple of (start_time, end_time) in seconds from item start
            For transition_in: typically (0.0, duration)
            For transition_out: typically (item_duration - duration, item_duration)

        Raises:
            ValueError: If parameters are invalid
        """

    @abstractmethod
    def is_in_transition_region(
        self, timestamp: float, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> bool:
        """
        Check if a timestamp falls within the transition region.

        Args:
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration including parameters
            direction: "in" for transition_in, "out" for transition_out

        Returns:
            True if timestamp is within the transition region, False otherwise
        """

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate transition parameters.

        Args:
            parameters: Dictionary of transition-specific parameters

        Returns:
            True if parameters are valid, False otherwise

        Note:
            Should log specific validation errors for debugging
        """

    @abstractmethod
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for this transition's parameters.

        Returns:
            JSON schema dictionary describing valid parameters,
            their types, ranges, and default values

        Example:
            {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 10.0,
                        "default": 1.0
                    }
                },
                "required": ["duration"]
            }
        """

    def get_transition_name(self) -> str:
        """
        Get the human-readable name of this transition type.

        Returns:
            Human-readable transition name for UI display
        """
        return self.__class__.__name__.replace("LEDTransition", "").lower()

    def get_transition_progress(
        self, timestamp: float, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> float:
        """
        Calculate transition progress as a value between 0.0 and 1.0.

        Args:
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration including parameters
            direction: "in" for transition_in, "out" for transition_out

        Returns:
            Progress value between 0.0 (transition start) and 1.0 (transition end)
            Returns 0.0 if before transition region, 1.0 if after transition region

        Note:
            This is a helper method that derived classes can use for progress calculations
        """
        if not self.is_in_transition_region(timestamp, item_duration, transition_config, direction):
            # Outside transition region
            start_time, end_time = self.get_transition_region(item_duration, transition_config, direction)
            if timestamp < start_time:
                return 0.0
            else:
                return 1.0

        start_time, end_time = self.get_transition_region(item_duration, transition_config, direction)
        transition_duration = end_time - start_time

        if transition_duration <= 0:
            return 1.0

        progress = (timestamp - start_time) / transition_duration
        return max(0.0, min(1.0, progress))
