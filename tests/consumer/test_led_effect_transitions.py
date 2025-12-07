"""
Unit tests for LED Effect Transition wrappers.

Tests FadeInEffect, FadeOutEffect, RandomInEffect, RandomOutEffect classes
that wrap LED transitions for use in the unified LED effect framework.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_led_values():
    """Create sample LED values array."""
    # 100 LEDs with random RGB values
    return np.random.randint(0, 256, (100, 3), dtype=np.uint8)


@pytest.fixture
def mock_led_fade_transition():
    """Create mock LEDFadeTransition."""
    mock = MagicMock()
    mock.apply_led_transition.return_value = np.zeros((100, 3), dtype=np.uint8)
    return mock


@pytest.fixture
def mock_led_random_transition():
    """Create mock LEDRandomTransition."""
    mock = MagicMock()
    mock.apply_led_transition.return_value = np.zeros((100, 3), dtype=np.uint8)
    return mock


# =============================================================================
# FadeInEffect Tests
# =============================================================================


class TestFadeInEffect:
    """Test FadeInEffect class."""

    def test_initialization(self):
        """Test FadeInEffect initialization."""
        with patch("src.consumer.led_effect_transitions.LEDFadeTransition"):
            from src.consumer.led_effect_transitions import FadeInEffect

            effect = FadeInEffect(start_time=0.0, duration=1.0)

            assert effect.start_time == 0.0
            assert effect.duration == 1.0
            assert effect.curve == "linear"
            assert effect.min_brightness == 0.0

    def test_initialization_with_custom_params(self):
        """Test FadeInEffect with custom parameters."""
        with patch("src.consumer.led_effect_transitions.LEDFadeTransition"):
            from src.consumer.led_effect_transitions import FadeInEffect

            effect = FadeInEffect(
                start_time=1.0,
                duration=2.0,
                curve="ease-in",
                min_brightness=0.2,
            )

            assert effect.start_time == 1.0
            assert effect.duration == 2.0
            assert effect.curve == "ease-in"
            assert effect.min_brightness == 0.2

    def test_apply_before_start(self, sample_led_values):
        """Test apply returns False when before start time."""
        with patch("src.consumer.led_effect_transitions.LEDFadeTransition"):
            from src.consumer.led_effect_transitions import FadeInEffect

            effect = FadeInEffect(start_time=1.0, duration=1.0)

            # Frame timestamp before start
            result = effect.apply(sample_led_values, frame_timestamp=0.5)

            assert result is False  # Not complete
            assert effect.frame_count == 1

    def test_apply_during_effect(self, sample_led_values):
        """Test apply during effect duration."""
        with patch("src.consumer.led_effect_transitions.LEDFadeTransition") as MockTransition:
            mock_transition = MagicMock()
            # Return cupy array mock
            mock_result = MagicMock()
            mock_transition.apply_led_transition.return_value = mock_result
            MockTransition.return_value = mock_transition

            # Also mock cupy
            with patch("src.consumer.led_effect_transitions.cp") as mock_cp:
                mock_cp.asarray.return_value = sample_led_values
                mock_cp.asnumpy.return_value = sample_led_values

                from src.consumer.led_effect_transitions import FadeInEffect

                effect = FadeInEffect(start_time=0.0, duration=1.0)

                # Frame timestamp during effect
                result = effect.apply(sample_led_values, frame_timestamp=0.5)

                # Verify transition was called
                mock_transition.apply_led_transition.assert_called_once()
                call_kwargs = mock_transition.apply_led_transition.call_args
                assert call_kwargs[1]["direction"] == "in"

    def test_apply_after_duration(self, sample_led_values):
        """Test apply returns True when effect is complete."""
        with patch("src.consumer.led_effect_transitions.LEDFadeTransition") as MockTransition:
            mock_transition = MagicMock()
            mock_transition.apply_led_transition.return_value = sample_led_values
            MockTransition.return_value = mock_transition

            with patch("src.consumer.led_effect_transitions.cp") as mock_cp:
                mock_cp.asarray.return_value = sample_led_values
                mock_cp.asnumpy.return_value = sample_led_values

                from src.consumer.led_effect_transitions import FadeInEffect

                effect = FadeInEffect(start_time=0.0, duration=1.0)

                # Frame timestamp after duration
                result = effect.apply(sample_led_values, frame_timestamp=1.5)

                # Should be complete
                assert result is True

    def test_get_info(self):
        """Test get_info returns correct information."""
        with patch("src.consumer.led_effect_transitions.LEDFadeTransition"):
            from src.consumer.led_effect_transitions import FadeInEffect

            effect = FadeInEffect(
                start_time=0.0,
                duration=1.0,
                curve="ease-out",
                min_brightness=0.3,
            )

            info = effect.get_info()

            assert info["curve"] == "ease-out"
            assert info["min_brightness"] == 0.3
            assert "start_time" in info
            assert "duration" in info


# =============================================================================
# FadeOutEffect Tests
# =============================================================================


class TestFadeOutEffect:
    """Test FadeOutEffect class."""

    def test_initialization(self):
        """Test FadeOutEffect initialization."""
        with patch("src.consumer.led_effect_transitions.LEDFadeTransition"):
            from src.consumer.led_effect_transitions import FadeOutEffect

            effect = FadeOutEffect(start_time=0.0, duration=1.0)

            assert effect.start_time == 0.0
            assert effect.duration == 1.0
            assert effect.curve == "linear"
            assert effect.min_brightness == 0.0

    def test_initialization_with_custom_params(self):
        """Test FadeOutEffect with custom parameters."""
        with patch("src.consumer.led_effect_transitions.LEDFadeTransition"):
            from src.consumer.led_effect_transitions import FadeOutEffect

            effect = FadeOutEffect(
                start_time=2.0,
                duration=0.5,
                curve="ease-in-out",
                min_brightness=0.1,
            )

            assert effect.start_time == 2.0
            assert effect.duration == 0.5
            assert effect.curve == "ease-in-out"
            assert effect.min_brightness == 0.1

    def test_apply_direction_is_out(self, sample_led_values):
        """Test that apply uses 'out' direction."""
        with patch("src.consumer.led_effect_transitions.LEDFadeTransition") as MockTransition:
            mock_transition = MagicMock()
            mock_transition.apply_led_transition.return_value = sample_led_values
            MockTransition.return_value = mock_transition

            with patch("src.consumer.led_effect_transitions.cp") as mock_cp:
                mock_cp.asarray.return_value = sample_led_values
                mock_cp.asnumpy.return_value = sample_led_values

                from src.consumer.led_effect_transitions import FadeOutEffect

                effect = FadeOutEffect(start_time=0.0, duration=1.0)
                effect.apply(sample_led_values, frame_timestamp=0.5)

                # Verify direction is "out"
                call_kwargs = mock_transition.apply_led_transition.call_args
                assert call_kwargs[1]["direction"] == "out"

    def test_get_info(self):
        """Test get_info returns correct information."""
        with patch("src.consumer.led_effect_transitions.LEDFadeTransition"):
            from src.consumer.led_effect_transitions import FadeOutEffect

            effect = FadeOutEffect(
                start_time=0.0,
                duration=2.0,
                curve="linear",
                min_brightness=0.0,
            )

            info = effect.get_info()

            assert info["curve"] == "linear"
            assert info["min_brightness"] == 0.0


# =============================================================================
# RandomInEffect Tests
# =============================================================================


class TestRandomInEffect:
    """Test RandomInEffect class."""

    def test_initialization(self):
        """Test RandomInEffect initialization."""
        with patch("src.consumer.led_effect_transitions.LEDRandomTransition"):
            from src.consumer.led_effect_transitions import RandomInEffect

            effect = RandomInEffect(start_time=0.0, duration=1.0)

            assert effect.start_time == 0.0
            assert effect.duration == 1.0
            assert effect.leds_per_frame == 10
            assert effect.fade_tail is True
            assert effect.seed == 42

    def test_initialization_with_custom_params(self):
        """Test RandomInEffect with custom parameters."""
        with patch("src.consumer.led_effect_transitions.LEDRandomTransition"):
            from src.consumer.led_effect_transitions import RandomInEffect

            effect = RandomInEffect(
                start_time=1.0,
                duration=3.0,
                leds_per_frame=20,
                fade_tail=False,
                seed=123,
            )

            assert effect.start_time == 1.0
            assert effect.duration == 3.0
            assert effect.leds_per_frame == 20
            assert effect.fade_tail is False
            assert effect.seed == 123

    def test_apply_before_start(self, sample_led_values):
        """Test apply returns False when before start time."""
        with patch("src.consumer.led_effect_transitions.LEDRandomTransition"):
            from src.consumer.led_effect_transitions import RandomInEffect

            effect = RandomInEffect(start_time=1.0, duration=1.0)

            result = effect.apply(sample_led_values, frame_timestamp=0.5)

            assert result is False

    def test_apply_direction_is_in(self, sample_led_values):
        """Test that apply uses 'in' direction."""
        with patch("src.consumer.led_effect_transitions.LEDRandomTransition") as MockTransition:
            mock_transition = MagicMock()
            mock_transition.apply_led_transition.return_value = sample_led_values
            MockTransition.return_value = mock_transition

            with patch("src.consumer.led_effect_transitions.cp") as mock_cp:
                mock_cp.asarray.return_value = sample_led_values
                mock_cp.asnumpy.return_value = sample_led_values

                from src.consumer.led_effect_transitions import RandomInEffect

                effect = RandomInEffect(start_time=0.0, duration=1.0)
                effect.apply(sample_led_values, frame_timestamp=0.5)

                call_kwargs = mock_transition.apply_led_transition.call_args
                assert call_kwargs[1]["direction"] == "in"

    def test_get_info(self):
        """Test get_info returns correct information."""
        with patch("src.consumer.led_effect_transitions.LEDRandomTransition"):
            from src.consumer.led_effect_transitions import RandomInEffect

            effect = RandomInEffect(
                start_time=0.0,
                duration=1.0,
                leds_per_frame=15,
                fade_tail=True,
                seed=99,
            )

            info = effect.get_info()

            assert info["leds_per_frame"] == 15
            assert info["fade_tail"] is True
            assert info["seed"] == 99


# =============================================================================
# RandomOutEffect Tests
# =============================================================================


class TestRandomOutEffect:
    """Test RandomOutEffect class."""

    def test_initialization(self):
        """Test RandomOutEffect initialization."""
        with patch("src.consumer.led_effect_transitions.LEDRandomTransition"):
            from src.consumer.led_effect_transitions import RandomOutEffect

            effect = RandomOutEffect(start_time=0.0, duration=1.0)

            assert effect.start_time == 0.0
            assert effect.duration == 1.0
            assert effect.leds_per_frame == 10
            assert effect.fade_tail is True
            assert effect.seed == 42

    def test_initialization_with_custom_params(self):
        """Test RandomOutEffect with custom parameters."""
        with patch("src.consumer.led_effect_transitions.LEDRandomTransition"):
            from src.consumer.led_effect_transitions import RandomOutEffect

            effect = RandomOutEffect(
                start_time=2.0,
                duration=0.5,
                leds_per_frame=5,
                fade_tail=False,
                seed=456,
            )

            assert effect.start_time == 2.0
            assert effect.duration == 0.5
            assert effect.leds_per_frame == 5
            assert effect.fade_tail is False
            assert effect.seed == 456

    def test_apply_direction_is_out(self, sample_led_values):
        """Test that apply uses 'out' direction."""
        with patch("src.consumer.led_effect_transitions.LEDRandomTransition") as MockTransition:
            mock_transition = MagicMock()
            mock_transition.apply_led_transition.return_value = sample_led_values
            MockTransition.return_value = mock_transition

            with patch("src.consumer.led_effect_transitions.cp") as mock_cp:
                mock_cp.asarray.return_value = sample_led_values
                mock_cp.asnumpy.return_value = sample_led_values

                from src.consumer.led_effect_transitions import RandomOutEffect

                effect = RandomOutEffect(start_time=0.0, duration=1.0)
                effect.apply(sample_led_values, frame_timestamp=0.5)

                call_kwargs = mock_transition.apply_led_transition.call_args
                assert call_kwargs[1]["direction"] == "out"

    def test_get_info(self):
        """Test get_info returns correct information."""
        with patch("src.consumer.led_effect_transitions.LEDRandomTransition"):
            from src.consumer.led_effect_transitions import RandomOutEffect

            effect = RandomOutEffect(
                start_time=0.0,
                duration=1.0,
                leds_per_frame=25,
                fade_tail=False,
                seed=789,
            )

            info = effect.get_info()

            assert info["leds_per_frame"] == 25
            assert info["fade_tail"] is False
            assert info["seed"] == 789
