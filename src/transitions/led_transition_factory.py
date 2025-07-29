"""
LED transition factory and registry for managing LED transition types.

This module provides a centralized registry for all available LED transition types
and factory methods for creating LED transition instances based on configuration.
"""

import logging
from typing import Any, Dict, List, Optional, Type

import cupy as cp

from .base_led_transition import BaseLEDTransition
from .led_blur_transition import LEDBlurTransition
from .led_fade_transition import LEDFadeTransition
from .led_random_transition import LEDRandomTransition

logger = logging.getLogger(__name__)


class LEDTransitionFactory:
    """
    Factory and registry for LED transition implementations.

    Manages the available LED transition types, creates instances based on
    configuration, and provides validation and introspection capabilities.
    """

    def __init__(self):
        """Initialize the LED transition factory with built-in transitions."""
        # Registry of LED transition type name -> transition class
        self._led_transitions: Dict[str, Type[BaseLEDTransition]] = {}

        # Cache of LED transition instances (one per type for reuse)
        self._led_transition_instances: Dict[str, BaseLEDTransition] = {}

        # Register built-in LED transitions
        self.register_led_transition("none", NoneLEDTransition)
        self.register_led_transition("ledfade", LEDFadeTransition)
        self.register_led_transition("ledblur", LEDBlurTransition)
        self.register_led_transition("ledrandom", LEDRandomTransition)

    def register_led_transition(self, name: str, transition_class: Type[BaseLEDTransition]) -> None:
        """
        Register an LED transition implementation.

        Args:
            name: Unique name for the LED transition type
            transition_class: Class implementing BaseLEDTransition

        Raises:
            ValueError: If name is invalid or class doesn't implement BaseLEDTransition
        """
        if not name or not isinstance(name, str):
            raise ValueError("LED transition name must be a non-empty string")

        if not issubclass(transition_class, BaseLEDTransition):
            raise ValueError("LED transition class must inherit from BaseLEDTransition")

        if name in self._led_transitions:
            logger.warning(f"Overriding existing LED transition '{name}'")

        self._led_transitions[name] = transition_class
        logger.debug(f"Registered LED transition type '{name}': {transition_class.__name__}")

    def get_available_led_transitions(self) -> List[str]:
        """
        Get list of available LED transition type names.

        Returns:
            List of registered LED transition type names
        """
        return list(self._led_transitions.keys())

    def create_led_transition(self, transition_type: str) -> Optional[BaseLEDTransition]:
        """
        Get a cached LED transition instance by type name (creates once, reuses thereafter).

        Args:
            transition_type: Name of the LED transition type to get

        Returns:
            Cached LED transition instance, or None if type not found

        Note:
            Logs error if LED transition type is not registered
        """
        if transition_type not in self._led_transitions:
            logger.error(
                f"Unknown LED transition type '{transition_type}'. Available types: {self.get_available_led_transitions()}"
            )
            return None

        # Return cached instance if available
        if transition_type in self._led_transition_instances:
            return self._led_transition_instances[transition_type]

        # Create and cache new instance
        try:
            transition_class = self._led_transitions[transition_type]
            instance = transition_class()
            self._led_transition_instances[transition_type] = instance
            logger.debug(f"Created and cached LED transition instance for '{transition_type}'")
            return instance
        except Exception as e:
            logger.error(f"Error creating LED transition '{transition_type}': {e}")
            return None

    def validate_led_transition_config(self, transition_config: Dict[str, Any]) -> bool:
        """
        Validate a complete LED transition configuration.

        Args:
            transition_config: Configuration dict with 'type' and 'parameters' keys

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required structure
            if not isinstance(transition_config, dict):
                logger.error("LED transition config must be a dictionary")
                return False

            transition_type = transition_config.get("type")
            if not transition_type:
                logger.error("LED transition config missing 'type' field")
                return False

            # Check if LED transition type exists
            if transition_type not in self._led_transitions:
                logger.error(f"Unknown LED transition type '{transition_type}'")
                return False

            # Validate parameters with LED transition implementation
            parameters = transition_config.get("parameters", {})
            transition = self.create_led_transition(transition_type)
            if transition is None:
                return False

            return transition.validate_parameters(parameters)

        except Exception as e:
            logger.error(f"Error validating LED transition config: {e}")
            return False

    def get_led_transition_schema(self, transition_type: str) -> Optional[Dict[str, Any]]:
        """
        Get parameter schema for a specific LED transition type.

        Args:
            transition_type: Name of the LED transition type

        Returns:
            JSON schema dictionary, or None if type not found
        """
        transition = self.create_led_transition(transition_type)
        if transition is None:
            return None

        try:
            return transition.get_parameter_schema()
        except Exception as e:
            logger.error(f"Error getting schema for LED transition '{transition_type}': {e}")
            return None

    def get_all_led_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter schemas for all registered LED transition types.

        Returns:
            Dictionary mapping LED transition type names to their schemas
        """
        schemas = {}
        for transition_type in self.get_available_led_transitions():
            schema = self.get_led_transition_schema(transition_type)
            if schema is not None:
                schemas[transition_type] = schema
        return schemas

    def apply_led_transition_to_values(
        self,
        led_values: cp.ndarray,
        timestamp: float,
        item_duration: float,
        transition_config: Dict[str, Any],
        direction: str,
    ) -> cp.ndarray:
        """
        Apply an LED transition to LED values using the factory.

        Args:
            led_values: LED RGB values as cupy GPU array (led_count, 3) with values 0-255
            timestamp: Time within the current playlist item (seconds)
            item_duration: Total duration of the playlist item (seconds)
            transition_config: LED transition configuration with type and parameters
            direction: "in" for transition_in, "out" for transition_out

        Returns:
            LED values with transition applied, or original values if error/no transition

        Note:
            This is a convenience method that handles LED transition creation
            and error handling automatically.
        """
        try:
            # Validate config
            if not self.validate_led_transition_config(transition_config):
                logger.warning("Invalid LED transition config, skipping LED transition")
                return led_values

            # Create and apply LED transition
            transition_type = transition_config["type"]
            transition = self.create_led_transition(transition_type)
            if transition is None:
                return led_values

            return transition.apply_led_transition(led_values, timestamp, item_duration, transition_config, direction)

        except Exception as e:
            logger.error(f"Error applying LED transition: {e}")
            return led_values


class NoneLEDTransition(BaseLEDTransition):
    """
    Pass-through LED transition that applies no effect.

    This is the default LED transition type that leaves LED values unmodified.
    """

    def apply_led_transition(
        self,
        led_values: cp.ndarray,
        timestamp: float,
        item_duration: float,
        transition_config: Dict[str, Any],
        direction: str,
    ) -> cp.ndarray:
        """Return LED values unmodified."""
        # Validate LED values is GPU array for consistency
        if not isinstance(led_values, cp.ndarray):
            logger.error(f"Expected GPU cupy array, got {type(led_values)}")
            raise ValueError(f"LED values must be cupy GPU array, got {type(led_values)}")
        return led_values

    def get_transition_region(self, item_duration: float, transition_config: Dict[str, Any], direction: str):
        """Return empty transition region."""
        return (0.0, 0.0)

    def is_in_transition_region(
        self, timestamp: float, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> bool:
        """Always return False since no transition is applied."""
        return False

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """No parameters needed for none LED transition."""
        return True

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Return empty schema since no parameters are needed."""
        return {"type": "object", "properties": {}, "additionalProperties": False}


# Global LED factory instance
_led_transition_factory: Optional[LEDTransitionFactory] = None


def get_led_transition_factory() -> LEDTransitionFactory:
    """
    Get the global LED transition factory instance.

    Returns:
        Singleton LEDTransitionFactory instance
    """
    global _led_transition_factory
    if _led_transition_factory is None:
        _led_transition_factory = LEDTransitionFactory()
    return _led_transition_factory


def register_custom_led_transition(name: str, transition_class: Type[BaseLEDTransition]) -> None:
    """
    Register a custom LED transition type globally.

    Args:
        name: Unique name for the LED transition type
        transition_class: Class implementing BaseLEDTransition
    """
    factory = get_led_transition_factory()
    factory.register_led_transition(name, transition_class)
