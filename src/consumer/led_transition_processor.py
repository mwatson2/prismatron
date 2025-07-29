"""
LED transition processor component for the consumer pipeline.

This module handles the application of playlist item transitions to LED values
after optimization but before transmission to WLED. It provides a more efficient
alternative to image-based transitions for effects like fade transitions.
"""

import logging
from typing import Any, Dict, Optional

import cupy as cp
import numpy as np

from ..transitions.led_transition_factory import get_led_transition_factory

logger = logging.getLogger(__name__)


class LEDTransitionProcessor:
    """
    Handles LED transition processing in the consumer pipeline.

    This component extracts transition metadata and applies LED-based transition
    effects directly to optimized LED values. This is more efficient than image-based
    transitions for effects like fade transitions since it operates on the final
    LED data rather than the source image.
    """

    def __init__(self):
        """Initialize the LED transition processor."""
        self._factory = get_led_transition_factory()
        self._frame_count = 0

        logger.info("LED transition processor initialized")

    def process_led_values(self, led_values: cp.ndarray, metadata: Dict[str, Any]) -> cp.ndarray:
        """
        Process LED values with LED transitions if applicable.

        Args:
            led_values: LED RGB values as cupy GPU array (led_count, 3) with values 0-255
            metadata: Frame metadata from shared buffer containing transition info

        Returns:
            LED values with LED transitions applied, or original values if no LED transitions
        """
        try:
            self._frame_count += 1

            # Extract transition metadata from frame metadata
            transition_context = self._extract_transition_context(metadata)
            if not transition_context:
                # No transition metadata available
                return led_values

            # Check if we need to apply any LED transitions
            led_values_with_transitions = led_values.copy()
            transitions_applied = []

            # Apply LED transition_in if frame is in the transition region
            if self._should_apply_led_transition_in(transition_context):
                led_values_with_transitions = self._apply_led_transition(
                    led_values_with_transitions, transition_context, "in"
                )
                transitions_applied.append("led_in")

            # Apply LED transition_out if frame is in the transition region
            if self._should_apply_led_transition_out(transition_context):
                led_values_with_transitions = self._apply_led_transition(
                    led_values_with_transitions, transition_context, "out"
                )
                transitions_applied.append("led_out")

            if transitions_applied:
                logger.debug(f"Applied LED transitions to frame {self._frame_count}: {', '.join(transitions_applied)}")

            return led_values_with_transitions

        except Exception as e:
            logger.error(f"Error processing LED transitions: {e}")
            # Return original LED values on error to avoid breaking the pipeline
            return led_values

    def _extract_transition_context(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract transition context from frame metadata.

        Args:
            metadata: Frame metadata dictionary

        Returns:
            Dictionary containing transition context, or None if not available
        """
        try:
            # Check if required transition fields are present (same format as regular transition processor)
            required_fields = [
                "transition_in_type",
                "transition_in_duration",
                "transition_out_type",
                "transition_out_duration",
                "item_timestamp",
                "item_duration",
            ]

            missing_fields = [field for field in required_fields if field not in metadata]
            if missing_fields:
                return None

            # Extract transition configuration (same format as regular transition processor)
            return {
                "timestamp": metadata["item_timestamp"],
                "item_duration": metadata["item_duration"],
                "transition_in": {
                    "type": metadata["transition_in_type"],
                    "parameters": metadata.get("transition_in_parameters", {}),
                },
                "transition_out": {
                    "type": metadata["transition_out_type"],
                    "parameters": metadata.get("transition_out_parameters", {}),
                },
            }

        except Exception as e:
            logger.warning(f"Error extracting LED transition context: {e}")
            return None

    def _should_apply_led_transition_in(self, context: Dict[str, Any]) -> bool:
        """
        Check if LED transition_in should be applied to this frame.

        Args:
            context: Transition context dictionary

        Returns:
            True if LED transition_in should be applied
        """
        try:
            transition_config = context["transition_in"]

            # Check if this is an LED transition
            if not transition_config.get("type", "").startswith("led"):
                return False

            transition = self._factory.create_led_transition(transition_config["type"])
            if transition is None:
                return False

            return transition.is_in_transition_region(
                context["timestamp"], context["item_duration"], transition_config, "in"
            )

        except Exception as e:
            logger.warning(f"Error checking LED transition_in: {e}")
            return False

    def _should_apply_led_transition_out(self, context: Dict[str, Any]) -> bool:
        """
        Check if LED transition_out should be applied to this frame.

        Args:
            context: Transition context dictionary

        Returns:
            True if LED transition_out should be applied
        """
        try:
            transition_config = context["transition_out"]

            # Check if this is an LED transition
            if not transition_config.get("type", "").startswith("led"):
                return False

            transition = self._factory.create_led_transition(transition_config["type"])
            if transition is None:
                return False

            return transition.is_in_transition_region(
                context["timestamp"], context["item_duration"], transition_config, "out"
            )

        except Exception as e:
            logger.warning(f"Error checking LED transition_out: {e}")
            return False

    def _apply_led_transition(self, led_values: cp.ndarray, context: Dict[str, Any], direction: str) -> cp.ndarray:
        """
        Apply LED transition to LED values.

        Args:
            led_values: LED values to apply transition to
            context: Transition context dictionary
            direction: "in" or "out"

        Returns:
            LED values with transition applied
        """
        try:
            # Get the transition config for this direction
            transition_key = f"transition_{direction}"
            transition_config = context[transition_key]

            # Apply the LED transition using the factory
            return self._factory.apply_led_transition_to_values(
                led_values,
                context["timestamp"],
                context["item_duration"],
                transition_config,
                direction,
            )

        except Exception as e:
            logger.error(f"Error applying LED transition_{direction}: {e}")
            return led_values

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get LED transition processor statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "frames_processed": self._frame_count,
            "available_led_transitions": self._factory.get_available_led_transitions(),
        }
