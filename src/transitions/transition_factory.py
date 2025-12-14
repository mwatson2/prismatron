"""
Transition factory and registry for managing transition types.

This module provides a centralized registry for all available transition types
and factory methods for creating transition instances based on configuration.
"""

import logging
from typing import Any, Dict, List, Optional, Type

import cupy as cp

from .base_transition import BaseTransition
from .blur_transition import BlurTransition
from .fade_transition import FadeTransition
from .led_transition_factory import get_led_transition_factory

logger = logging.getLogger(__name__)


class TransitionFactory:
    """
    Factory and registry for transition implementations.

    Manages the available transition types, creates instances based on
    configuration, and provides validation and introspection capabilities.
    """

    def __init__(self):
        """Initialize the transition factory with built-in transitions."""
        # Registry of transition type name -> transition class
        self._transitions: Dict[str, Type[BaseTransition]] = {}

        # Cache of transition instances (one per type for reuse)
        self._transition_instances: Dict[str, BaseTransition] = {}

        # Register built-in transitions
        self.register_transition("none", NoneTransition)
        self.register_transition("fade", FadeTransition)
        self.register_transition("blur", BlurTransition)

    def register_transition(self, name: str, transition_class: Type[BaseTransition]) -> None:
        """
        Register a transition implementation.

        Args:
            name: Unique name for the transition type
            transition_class: Class implementing BaseTransition

        Raises:
            ValueError: If name is invalid or class doesn't implement BaseTransition
        """
        if not name or not isinstance(name, str):
            raise ValueError("Transition name must be a non-empty string")

        if not issubclass(transition_class, BaseTransition):
            raise ValueError("Transition class must inherit from BaseTransition")

        if name in self._transitions:
            logger.warning(f"Overriding existing transition '{name}'")

        self._transitions[name] = transition_class
        logger.debug(f"Registered transition type '{name}': {transition_class.__name__}")

    def get_available_transitions(self) -> List[str]:
        """
        Get list of available transition type names.

        Returns:
            List of registered transition type names
        """
        return list(self._transitions.keys())

    def create_transition(self, transition_type: str) -> Optional[BaseTransition]:
        """
        Get a cached transition instance by type name (creates once, reuses thereafter).

        Args:
            transition_type: Name of the transition type to get

        Returns:
            Cached transition instance, or None if type not found

        Note:
            Logs error if transition type is not registered
        """
        if transition_type not in self._transitions:
            logger.error(
                f"Unknown transition type '{transition_type}'. Available types: {self.get_available_transitions()}"
            )
            return None

        # Return cached instance if available
        if transition_type in self._transition_instances:
            return self._transition_instances[transition_type]

        # Create and cache new instance
        try:
            transition_class = self._transitions[transition_type]
            instance = transition_class()
            self._transition_instances[transition_type] = instance
            logger.debug(f"Created and cached transition instance for '{transition_type}'")
            return instance
        except Exception as e:
            logger.error(f"Error creating transition '{transition_type}': {e}")
            return None

    def validate_transition_config(self, transition_config: Dict[str, Any]) -> bool:
        """
        Validate a complete transition configuration.

        Args:
            transition_config: Configuration dict with 'type' and 'parameters' keys

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required structure (type annotation guarantees dict)
            transition_type = transition_config.get("type")
            if not transition_type:
                logger.error("Transition config missing 'type' field")
                return False

            # Check if transition type exists
            if transition_type not in self._transitions:
                logger.error(f"Unknown transition type '{transition_type}'")
                return False

            # Validate parameters with transition implementation
            parameters = transition_config.get("parameters", {})
            transition = self.create_transition(transition_type)
            if transition is None:
                return False

            return transition.validate_parameters(parameters)

        except Exception as e:
            logger.error(f"Error validating transition config: {e}")
            return False

    def get_transition_schema(self, transition_type: str) -> Optional[Dict[str, Any]]:
        """
        Get parameter schema for a specific transition type.

        Args:
            transition_type: Name of the transition type

        Returns:
            JSON schema dictionary, or None if type not found
        """
        transition = self.create_transition(transition_type)
        if transition is None:
            return None

        try:
            return transition.get_parameter_schema()
        except Exception as e:
            logger.error(f"Error getting schema for '{transition_type}': {e}")
            return None

    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter schemas for all registered transition types.

        Returns:
            Dictionary mapping transition type names to their schemas
        """
        schemas = {}
        for transition_type in self.get_available_transitions():
            schema = self.get_transition_schema(transition_type)
            if schema is not None:
                schemas[transition_type] = schema
        return schemas

    def get_all_schemas_with_led(self) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter schemas for all transition types including LED transitions.

        Returns:
            Dictionary mapping transition type names to their schemas
        """
        schemas = {}

        # Add regular image transitions
        for transition_type in self.get_available_transitions():
            schema = self.get_transition_schema(transition_type)
            if schema is not None:
                schemas[transition_type] = schema

        # Add LED transitions
        led_factory = get_led_transition_factory()
        led_schemas = led_factory.get_all_led_schemas()
        schemas.update(led_schemas)

        return schemas

    def get_available_transitions_with_led(self) -> List[str]:
        """
        Get list of available transition type names including LED transitions.

        Returns:
            List of all registered transition type names (image + LED)
        """
        transitions = self.get_available_transitions()

        # Add LED transitions (excluding duplicates like "none")
        led_factory = get_led_transition_factory()
        led_transitions = led_factory.get_available_led_transitions()

        # Add only LED transitions that aren't already in the list
        for led_transition in led_transitions:
            if led_transition not in transitions:
                transitions.append(led_transition)

        return transitions

    def apply_transition_to_frame(
        self,
        frame,
        timestamp: float,
        item_duration: float,
        transition_config: Dict[str, Any],
        direction: str,
    ):
        """
        Apply a transition to a frame using the factory.

        Args:
            frame: RGB frame data as numpy array
            timestamp: Time within the current playlist item (seconds)
            item_duration: Total duration of the playlist item (seconds)
            transition_config: Transition configuration with type and parameters
            direction: "in" for transition_in, "out" for transition_out

        Returns:
            Frame with transition applied, or original frame if error/no transition

        Note:
            This is a convenience method that handles transition creation
            and error handling automatically.
        """
        try:
            # Validate config
            if not self.validate_transition_config(transition_config):
                logger.warning("Invalid transition config, skipping transition")
                return frame

            # Create and apply transition
            transition_type = transition_config["type"]
            transition = self.create_transition(transition_type)
            if transition is None:
                return frame

            return transition.apply_transition(frame, timestamp, item_duration, transition_config, direction)

        except Exception as e:
            logger.error(f"Error applying transition: {e}")
            return frame


class NoneTransition(BaseTransition):
    """
    Pass-through transition that applies no effect.

    This is the default transition type that leaves frames unmodified.
    """

    def apply_transition(
        self, frame, timestamp: float, item_duration: float, transition_config: Dict[str, Any], direction: str
    ):
        """Return frame unmodified."""
        # Validate frame is GPU array for consistency
        if not isinstance(frame, cp.ndarray):
            logger.error(f"Expected GPU cupy array, got {type(frame)}")
            raise ValueError(f"Frame must be cupy GPU array, got {type(frame)}")
        return frame

    def get_transition_region(self, item_duration: float, transition_config: Dict[str, Any], direction: str):
        """Return empty transition region."""
        return (0.0, 0.0)

    def is_in_transition_region(
        self, timestamp: float, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> bool:
        """Always return False since no transition is applied."""
        return False

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """No parameters needed for none transition."""
        return True

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Return empty schema since no parameters are needed."""
        return {"type": "object", "properties": {}, "additionalProperties": False}


# Global factory instance
_transition_factory: Optional[TransitionFactory] = None


def get_transition_factory() -> TransitionFactory:
    """
    Get the global transition factory instance.

    Returns:
        Singleton TransitionFactory instance
    """
    global _transition_factory
    if _transition_factory is None:
        _transition_factory = TransitionFactory()
    return _transition_factory


def register_custom_transition(name: str, transition_class: Type[BaseTransition]) -> None:
    """
    Register a custom transition type globally.

    Args:
        name: Unique name for the transition type
        transition_class: Class implementing BaseTransition
    """
    factory = get_transition_factory()
    factory.register_transition(name, transition_class)
