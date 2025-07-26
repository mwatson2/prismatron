"""
Base transition interface for playlist item transitions.

This module defines the abstract base class that all transition implementations
must inherit from, providing a consistent interface for applying visual
transitions between playlist items.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class BaseTransition(ABC):
    """
    Abstract base class for all transition implementations.

    Transitions are applied to individual frames during playback to create
    visual effects at the beginning (transition_in) or end (transition_out)
    of playlist items.
    """

    @abstractmethod
    def apply_transition(
        self,
        frame: np.ndarray,
        timestamp: float,
        item_duration: float,
        transition_config: Dict[str, Any],
        direction: str,
    ) -> np.ndarray:
        """
        Apply transition effect to a frame.

        Args:
            frame: RGB frame data as numpy array (H, W, 3) with values 0-255
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration including type and parameters
            direction: "in" for transition_in, "out" for transition_out

        Returns:
            Modified frame data with transition applied

        Raises:
            ValueError: If parameters are invalid
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
        return self.__class__.__name__.replace("Transition", "").lower()

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
