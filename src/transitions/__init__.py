"""
Transitions module for playlist item transitions.

This module provides the infrastructure for applying visual transitions
between playlist items, including fade effects and extensible support
for additional transition types.
"""

from .base_transition import BaseTransition
from .fade_transition import FadeTransition
from .transition_factory import TransitionFactory

__all__ = [
    "BaseTransition",
    "FadeTransition", 
    "TransitionFactory",
]