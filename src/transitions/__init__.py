"""
Transitions module for playlist item transitions.

This module provides the infrastructure for applying visual transitions
between playlist items, including image-based transitions (fade, blur)
and LED-based transitions (LED fade) for efficient brightness effects.
"""

from .base_led_transition import BaseLEDTransition
from .base_transition import BaseTransition
from .fade_transition import FadeTransition
from .led_fade_transition import LEDFadeTransition
from .led_transition_factory import LEDTransitionFactory
from .transition_factory import TransitionFactory

__all__ = [
    "BaseTransition",
    "FadeTransition",
    "TransitionFactory",
    "BaseLEDTransition",
    "LEDFadeTransition",
    "LEDTransitionFactory",
]
