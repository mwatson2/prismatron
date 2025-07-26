"""
Transition processor component for the consumer pipeline.

This module handles the application of playlist item transitions to frames
before LED optimization. It extracts transition metadata from frame data
and applies the appropriate transition effects using the factory system.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..transitions.transition_factory import get_transition_factory

logger = logging.getLogger(__name__)


class TransitionProcessor:
    """
    Handles transition processing in the consumer pipeline.

    This component extracts transition metadata from frame data and applies
    the appropriate transition effects using the centralized factory system.
    It processes both transition_in and transition_out effects as needed.
    """

    def __init__(self):
        """Initialize the transition processor."""
        self._factory = get_transition_factory()
        self._frame_count = 0

        logger.info("Transition processor initialized")

    def process_frame(self, frame: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Process a frame with transitions if applicable.

        Args:
            frame: RGB frame data as numpy array (H, W, 3) with values 0-255
            metadata: Frame metadata from shared buffer containing transition info

        Returns:
            Frame with transitions applied, or original frame if no transitions
        """
        try:
            self._frame_count += 1

            # Extract transition metadata from frame metadata
            transition_context = self._extract_transition_context(metadata)
            if not transition_context:
                # No transition metadata available
                return frame

            # Check if we need to apply any transitions
            frame_with_transitions = frame.copy()

            # Apply transition_in if frame is in the transition region
            if self._should_apply_transition_in(transition_context):
                frame_with_transitions = self._apply_transition(frame_with_transitions, transition_context, "in")

            # Apply transition_out if frame is in the transition region
            if self._should_apply_transition_out(transition_context):
                frame_with_transitions = self._apply_transition(frame_with_transitions, transition_context, "out")

            return frame_with_transitions

        except Exception as e:
            logger.error(f"Error processing frame transitions: {e}")
            # Return original frame on error to avoid breaking the pipeline
            return frame

    def _extract_transition_context(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract transition context from frame metadata.

        Args:
            metadata: Frame metadata from shared buffer

        Returns:
            Transition context dictionary, or None if no transition data
        """
        try:
            # Check if required transition fields are present
            required_fields = [
                "transition_in_type",
                "transition_in_duration",
                "transition_out_type",
                "transition_out_duration",
                "item_timestamp",
                "item_duration",
            ]

            if not all(field in metadata for field in required_fields):
                logger.debug("Frame metadata missing transition fields")
                return None

            # Extract transition configuration
            transition_context = {
                "transition_in": {
                    "type": metadata["transition_in_type"],
                    "parameters": {"duration": metadata["transition_in_duration"]},
                },
                "transition_out": {
                    "type": metadata["transition_out_type"],
                    "parameters": {"duration": metadata["transition_out_duration"]},
                },
                "item_timestamp": metadata["item_timestamp"],
                "item_duration": metadata["item_duration"],
            }

            # Skip processing if both transitions are "none"
            if (
                transition_context["transition_in"]["type"] == "none"
                and transition_context["transition_out"]["type"] == "none"
            ):
                return None

            return transition_context

        except Exception as e:
            logger.warning(f"Error extracting transition context: {e}")
            return None

    def _should_apply_transition_in(self, transition_context: Dict[str, Any]) -> bool:
        """
        Check if transition_in should be applied to this frame.

        Args:
            transition_context: Transition context from frame metadata

        Returns:
            True if transition_in should be applied, False otherwise
        """
        try:
            transition_config = transition_context["transition_in"]
            if transition_config["type"] == "none":
                return False

            # Create transition instance and check if frame is in transition region
            transition = self._factory.create_transition(transition_config["type"])
            if transition is None:
                return False

            return transition.is_in_transition_region(
                timestamp=transition_context["item_timestamp"],
                item_duration=transition_context["item_duration"],
                transition_config=transition_config,
                direction="in",
            )

        except Exception as e:
            logger.warning(f"Error checking transition_in applicability: {e}")
            return False

    def _should_apply_transition_out(self, transition_context: Dict[str, Any]) -> bool:
        """
        Check if transition_out should be applied to this frame.

        Args:
            transition_context: Transition context from frame metadata

        Returns:
            True if transition_out should be applied, False otherwise
        """
        try:
            transition_config = transition_context["transition_out"]
            if transition_config["type"] == "none":
                return False

            # Create transition instance and check if frame is in transition region
            transition = self._factory.create_transition(transition_config["type"])
            if transition is None:
                return False

            return transition.is_in_transition_region(
                timestamp=transition_context["item_timestamp"],
                item_duration=transition_context["item_duration"],
                transition_config=transition_config,
                direction="out",
            )

        except Exception as e:
            logger.warning(f"Error checking transition_out applicability: {e}")
            return False

    def _apply_transition(self, frame: np.ndarray, transition_context: Dict[str, Any], direction: str) -> np.ndarray:
        """
        Apply a specific transition to a frame.

        Args:
            frame: RGB frame data to process
            transition_context: Transition context from frame metadata
            direction: "in" for transition_in, "out" for transition_out

        Returns:
            Frame with transition applied
        """
        try:
            # Get the appropriate transition configuration
            transition_config = transition_context[f"transition_{direction}"]

            # Use the factory to apply the transition
            processed_frame = self._factory.apply_transition_to_frame(
                frame=frame,
                timestamp=transition_context["item_timestamp"],
                item_duration=transition_context["item_duration"],
                transition_config=transition_config,
                direction=direction,
            )

            # Log successful transition application (debug level to avoid spam)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Applied {direction} transition '{transition_config['type']}' "
                    f"at timestamp {transition_context['item_timestamp']:.3f}s"
                )

            return processed_frame

        except Exception as e:
            logger.error(f"Error applying {direction} transition: {e}")
            # Return original frame on error
            return frame

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get transition processor statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            "frames_processed": self._frame_count,
            "available_transitions": self._factory.get_available_transitions(),
        }

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._frame_count = 0
