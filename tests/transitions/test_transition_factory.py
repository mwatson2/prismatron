"""
Unit tests for Transition Factory.

Tests TransitionFactory and LEDTransitionFactory classes that manage
transition type registration and instance creation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# TransitionFactory Tests
# =============================================================================


class TestTransitionFactory:
    """Test TransitionFactory class."""

    def test_initialization(self):
        """Test TransitionFactory initializes with built-in transitions."""
        from src.transitions.transition_factory import TransitionFactory

        factory = TransitionFactory()

        available = factory.get_available_transitions()
        assert "none" in available
        assert "fade" in available
        assert "blur" in available

    def test_get_available_transitions(self):
        """Test getting list of available transitions."""
        from src.transitions.transition_factory import TransitionFactory

        factory = TransitionFactory()

        transitions = factory.get_available_transitions()

        assert isinstance(transitions, list)
        assert len(transitions) >= 3  # At least none, fade, blur

    def test_register_transition(self):
        """Test registering a new transition type."""
        from src.transitions.base_transition import BaseTransition
        from src.transitions.transition_factory import TransitionFactory

        factory = TransitionFactory()

        # Create a custom transition class
        class CustomTransition(BaseTransition):
            def apply_frame_transition(self, *args, **kwargs):
                pass

            def is_in_transition_region(self, *args, **kwargs):
                return False

            def get_default_parameters(self):
                return {}

        factory.register_transition("custom", CustomTransition)

        assert "custom" in factory.get_available_transitions()

    def test_register_transition_invalid_name(self):
        """Test registering with invalid name raises error."""
        from src.transitions.base_transition import BaseTransition
        from src.transitions.transition_factory import TransitionFactory

        factory = TransitionFactory()

        class CustomTransition(BaseTransition):
            def apply_frame_transition(self, *args, **kwargs):
                pass

            def is_in_transition_region(self, *args, **kwargs):
                return False

            def get_default_parameters(self):
                return {}

        with pytest.raises(ValueError, match="non-empty string"):
            factory.register_transition("", CustomTransition)

        with pytest.raises(ValueError, match="non-empty string"):
            factory.register_transition(None, CustomTransition)

    def test_register_transition_invalid_class(self):
        """Test registering with non-BaseTransition class raises error."""
        from src.transitions.transition_factory import TransitionFactory

        factory = TransitionFactory()

        class NotATransition:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseTransition"):
            factory.register_transition("invalid", NotATransition)

    def test_create_transition_valid_type(self):
        """Test creating a valid transition type."""
        from src.transitions.transition_factory import TransitionFactory

        factory = TransitionFactory()

        transition = factory.create_transition("fade")

        assert transition is not None

    def test_create_transition_caches_instance(self):
        """Test that create_transition caches and reuses instances."""
        from src.transitions.transition_factory import TransitionFactory

        factory = TransitionFactory()

        transition1 = factory.create_transition("fade")
        transition2 = factory.create_transition("fade")

        # Should return same cached instance
        assert transition1 is transition2

    def test_create_transition_invalid_type(self):
        """Test creating invalid transition type returns None."""
        from src.transitions.transition_factory import TransitionFactory

        factory = TransitionFactory()

        transition = factory.create_transition("nonexistent")

        assert transition is None

    def test_create_transition_none_type(self):
        """Test creating 'none' transition."""
        from src.transitions.transition_factory import TransitionFactory

        factory = TransitionFactory()

        transition = factory.create_transition("none")

        assert transition is not None


# =============================================================================
# LEDTransitionFactory Tests
# =============================================================================


class TestLEDTransitionFactory:
    """Test LEDTransitionFactory class."""

    def test_initialization(self):
        """Test LEDTransitionFactory initializes with built-in LED transitions."""
        from src.transitions.led_transition_factory import LEDTransitionFactory

        factory = LEDTransitionFactory()

        available = factory.get_available_led_transitions()
        assert "none" in available
        assert "ledfade" in available
        assert "ledblur" in available
        assert "ledrandom" in available

    def test_get_available_led_transitions(self):
        """Test getting list of available LED transitions."""
        from src.transitions.led_transition_factory import LEDTransitionFactory

        factory = LEDTransitionFactory()

        transitions = factory.get_available_led_transitions()

        assert isinstance(transitions, list)
        assert len(transitions) >= 4  # At least none, ledfade, ledblur, ledrandom

    def test_register_led_transition(self):
        """Test registering a new LED transition type."""
        from src.transitions.base_led_transition import BaseLEDTransition
        from src.transitions.led_transition_factory import LEDTransitionFactory

        factory = LEDTransitionFactory()

        # Create a custom LED transition class
        class CustomLEDTransition(BaseLEDTransition):
            def apply_led_transition(self, *args, **kwargs):
                pass

            def is_in_transition_region(self, *args, **kwargs):
                return False

            def get_default_parameters(self):
                return {}

        factory.register_led_transition("customled", CustomLEDTransition)

        assert "customled" in factory.get_available_led_transitions()

    def test_register_led_transition_invalid_name(self):
        """Test registering with invalid name raises error."""
        from src.transitions.base_led_transition import BaseLEDTransition
        from src.transitions.led_transition_factory import LEDTransitionFactory

        factory = LEDTransitionFactory()

        class CustomLEDTransition(BaseLEDTransition):
            def apply_led_transition(self, *args, **kwargs):
                pass

            def is_in_transition_region(self, *args, **kwargs):
                return False

            def get_default_parameters(self):
                return {}

        with pytest.raises(ValueError, match="non-empty string"):
            factory.register_led_transition("", CustomLEDTransition)

    def test_register_led_transition_invalid_class(self):
        """Test registering with non-BaseLEDTransition class raises error."""
        from src.transitions.led_transition_factory import LEDTransitionFactory

        factory = LEDTransitionFactory()

        class NotATransition:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseLEDTransition"):
            factory.register_led_transition("invalid", NotATransition)

    def test_create_led_transition_valid_type(self):
        """Test creating a valid LED transition type."""
        from src.transitions.led_transition_factory import LEDTransitionFactory

        factory = LEDTransitionFactory()

        transition = factory.create_led_transition("ledfade")

        assert transition is not None

    def test_create_led_transition_caches_instance(self):
        """Test that create_led_transition caches and reuses instances."""
        from src.transitions.led_transition_factory import LEDTransitionFactory

        factory = LEDTransitionFactory()

        transition1 = factory.create_led_transition("ledfade")
        transition2 = factory.create_led_transition("ledfade")

        # Should return same cached instance
        assert transition1 is transition2

    def test_create_led_transition_invalid_type(self):
        """Test creating invalid LED transition type returns None."""
        from src.transitions.led_transition_factory import LEDTransitionFactory

        factory = LEDTransitionFactory()

        transition = factory.create_led_transition("nonexistent")

        assert transition is None

    def test_create_led_transition_none_type(self):
        """Test creating 'none' LED transition."""
        from src.transitions.led_transition_factory import LEDTransitionFactory

        factory = LEDTransitionFactory()

        transition = factory.create_led_transition("none")

        assert transition is not None


# =============================================================================
# get_transition_factory and get_led_transition_factory Tests
# =============================================================================


class TestFactorySingletons:
    """Test singleton factory getter functions."""

    def test_get_transition_factory_returns_same_instance(self):
        """Test that get_transition_factory returns singleton."""
        from src.transitions.transition_factory import get_transition_factory

        factory1 = get_transition_factory()
        factory2 = get_transition_factory()

        assert factory1 is factory2

    def test_get_led_transition_factory_returns_same_instance(self):
        """Test that get_led_transition_factory returns singleton."""
        from src.transitions.led_transition_factory import get_led_transition_factory

        factory1 = get_led_transition_factory()
        factory2 = get_led_transition_factory()

        assert factory1 is factory2


# =============================================================================
# NoneTransition Tests
# =============================================================================


class TestNoneTransition:
    """Test NoneTransition (no-op transition)."""

    def test_none_transition_is_in_region(self):
        """Test NoneTransition is never in transition region."""
        from src.transitions.transition_factory import TransitionFactory

        factory = TransitionFactory()
        transition = factory.create_transition("none")

        config = {"type": "none", "parameters": {"duration": 1.0}}

        # Should always return False
        assert transition.is_in_transition_region(0.0, 10.0, config, "in") is False
        assert transition.is_in_transition_region(0.5, 10.0, config, "in") is False
        assert transition.is_in_transition_region(9.5, 10.0, config, "out") is False


# =============================================================================
# NoneLEDTransition Tests
# =============================================================================


class TestNoneLEDTransition:
    """Test NoneLEDTransition (no-op LED transition)."""

    def test_none_led_transition_is_in_region(self):
        """Test NoneLEDTransition is never in transition region."""
        from src.transitions.led_transition_factory import LEDTransitionFactory

        factory = LEDTransitionFactory()
        transition = factory.create_led_transition("none")

        config = {"type": "none", "parameters": {"duration": 1.0}}

        # Should always return False
        assert transition.is_in_transition_region(0.0, 10.0, config, "in") is False
        assert transition.is_in_transition_region(0.5, 10.0, config, "in") is False
        assert transition.is_in_transition_region(9.5, 10.0, config, "out") is False
