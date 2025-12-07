"""
Unit tests for LED transition implementations.

Tests LEDFadeTransition, LEDBlurTransition, LEDRandomTransition classes
that apply effects directly to LED values on the GPU.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_cupy():
    """Create mock cupy module for GPU operations."""
    mock_cp = MagicMock()

    # Create a mock array class that behaves like cupy array
    class MockCupyArray:
        def __init__(self, data):
            self._data = np.array(data) if not isinstance(data, np.ndarray) else data
            self.shape = self._data.shape
            self.ndim = self._data.ndim
            self.dtype = self._data.dtype

        def __mul__(self, other):
            return MockCupyArray(self._data * other)

        def __rmul__(self, other):
            return MockCupyArray(other * self._data)

        def copy(self):
            return MockCupyArray(self._data.copy())

        def get(self):
            return self._data

    mock_cp.ndarray = MockCupyArray
    mock_cp.asarray = lambda x: MockCupyArray(x)
    mock_cp.asnumpy = lambda x: x._data if hasattr(x, "_data") else np.array(x)
    mock_cp.zeros_like = lambda x: MockCupyArray(np.zeros_like(x._data))
    mock_cp.ones = lambda shape, dtype=np.float32: MockCupyArray(np.ones(shape, dtype=dtype))
    mock_cp.clip = lambda x, a, b: MockCupyArray(np.clip(x._data, a, b))
    mock_cp.uint8 = np.uint8
    mock_cp.float32 = np.float32

    return mock_cp


@pytest.fixture
def sample_led_values():
    """Create sample LED values array (100 LEDs, RGB)."""
    return np.random.randint(0, 256, (100, 3), dtype=np.uint8)


@pytest.fixture
def fade_config():
    """Create fade transition configuration."""
    return {
        "type": "ledfade",
        "parameters": {
            "duration": 1.0,
            "curve": "linear",
            "min_brightness": 0.0,
        },
    }


@pytest.fixture
def blur_config():
    """Create blur transition configuration."""
    return {
        "type": "ledblur",
        "parameters": {
            "duration": 1.0,
            "blur_intensity": 1.0,
            "kernel_size": 5,
        },
    }


@pytest.fixture
def random_config():
    """Create random transition configuration."""
    return {
        "type": "ledrandom",
        "parameters": {
            "duration": 1.0,
            "leds_per_frame": 10,
            "seed": 42,
        },
    }


# =============================================================================
# LEDFadeTransition Tests
# =============================================================================


class TestLEDFadeTransition:
    """Test LEDFadeTransition class."""

    def test_initialization(self):
        """Test LEDFadeTransition can be instantiated."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()
        assert transition is not None

    def test_get_transition_name(self):
        """Test transition name."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()
        name = transition.get_transition_name()
        assert "fade" in name.lower()

    def test_validate_parameters_valid(self):
        """Test parameter validation with valid parameters."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()

        valid_params = {"duration": 1.0, "curve": "linear", "min_brightness": 0.0}
        assert transition.validate_parameters(valid_params) is True

    def test_validate_parameters_empty(self):
        """Test parameter validation with empty parameters (uses defaults)."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()
        # Empty params should use defaults
        assert transition.validate_parameters({}) is True

    def test_get_transition_region_in(self, fade_config):
        """Test getting transition region for fade-in."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()

        start, end = transition.get_transition_region(10.0, fade_config, "in")

        assert start == 0.0
        assert end == 1.0  # Duration from config

    def test_get_transition_region_out(self, fade_config):
        """Test getting transition region for fade-out."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()

        start, end = transition.get_transition_region(10.0, fade_config, "out")

        assert start == 9.0  # item_duration - duration
        assert end == 10.0

    def test_is_in_transition_region_in_start(self, fade_config):
        """Test timestamp at start of fade-in region."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()

        # At start of item (should be in fade-in region)
        assert transition.is_in_transition_region(0.0, 10.0, fade_config, "in") is True
        assert transition.is_in_transition_region(0.5, 10.0, fade_config, "in") is True

    def test_is_in_transition_region_in_outside(self, fade_config):
        """Test timestamp outside fade-in region."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()

        # Middle of item (outside fade-in region)
        assert transition.is_in_transition_region(5.0, 10.0, fade_config, "in") is False

    def test_is_in_transition_region_out(self, fade_config):
        """Test timestamp in fade-out region."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()

        # Near end of item (should be in fade-out region)
        assert transition.is_in_transition_region(9.5, 10.0, fade_config, "out") is True
        assert transition.is_in_transition_region(5.0, 10.0, fade_config, "out") is False

    def test_get_transition_progress(self, fade_config):
        """Test transition progress calculation."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()

        # Start of fade-in
        progress = transition.get_transition_progress(0.0, 10.0, fade_config, "in")
        assert progress == 0.0

        # Middle of fade-in
        progress = transition.get_transition_progress(0.5, 10.0, fade_config, "in")
        assert progress == 0.5

        # End of fade-in
        progress = transition.get_transition_progress(1.0, 10.0, fade_config, "in")
        assert progress == 1.0

    def test_get_parameter_schema(self):
        """Test parameter schema includes required fields."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()
        schema = transition.get_parameter_schema()

        assert "properties" in schema
        assert "duration" in schema["properties"]


# =============================================================================
# LEDBlurTransition Tests
# =============================================================================


class TestLEDBlurTransition:
    """Test LEDBlurTransition class."""

    def test_initialization(self):
        """Test LEDBlurTransition can be instantiated."""
        from src.transitions.led_blur_transition import LEDBlurTransition

        transition = LEDBlurTransition()
        assert transition is not None

    def test_get_transition_name(self):
        """Test transition name."""
        from src.transitions.led_blur_transition import LEDBlurTransition

        transition = LEDBlurTransition()
        name = transition.get_transition_name()
        assert "blur" in name.lower()

    def test_validate_parameters_valid(self):
        """Test parameter validation with valid parameters."""
        from src.transitions.led_blur_transition import LEDBlurTransition

        transition = LEDBlurTransition()

        valid_params = {"duration": 1.0, "blur_intensity": 1.0, "kernel_size": 5}
        assert transition.validate_parameters(valid_params) is True

    def test_validate_parameters_empty(self):
        """Test parameter validation with empty parameters."""
        from src.transitions.led_blur_transition import LEDBlurTransition

        transition = LEDBlurTransition()
        assert transition.validate_parameters({}) is True

    def test_get_transition_region_in(self, blur_config):
        """Test getting transition region for blur-in."""
        from src.transitions.led_blur_transition import LEDBlurTransition

        transition = LEDBlurTransition()

        start, end = transition.get_transition_region(10.0, blur_config, "in")

        assert start == 0.0
        assert end == 1.0

    def test_get_transition_region_out(self, blur_config):
        """Test getting transition region for blur-out."""
        from src.transitions.led_blur_transition import LEDBlurTransition

        transition = LEDBlurTransition()

        start, end = transition.get_transition_region(10.0, blur_config, "out")

        assert start == 9.0
        assert end == 10.0

    def test_is_in_transition_region(self, blur_config):
        """Test is_in_transition_region."""
        from src.transitions.led_blur_transition import LEDBlurTransition

        transition = LEDBlurTransition()

        assert transition.is_in_transition_region(0.5, 10.0, blur_config, "in") is True
        assert transition.is_in_transition_region(5.0, 10.0, blur_config, "in") is False
        assert transition.is_in_transition_region(9.5, 10.0, blur_config, "out") is True

    def test_get_parameter_schema(self):
        """Test parameter schema includes blur-specific fields."""
        from src.transitions.led_blur_transition import LEDBlurTransition

        transition = LEDBlurTransition()
        schema = transition.get_parameter_schema()

        assert "properties" in schema
        assert "duration" in schema["properties"]


# =============================================================================
# LEDRandomTransition Tests
# =============================================================================


class TestLEDRandomTransition:
    """Test LEDRandomTransition class."""

    def test_initialization(self):
        """Test LEDRandomTransition can be instantiated."""
        from src.transitions.led_random_transition import LEDRandomTransition

        transition = LEDRandomTransition()
        assert transition is not None

    def test_initialization_has_cache(self):
        """Test LEDRandomTransition initializes with permutation cache."""
        from src.transitions.led_random_transition import LEDRandomTransition

        transition = LEDRandomTransition()
        assert hasattr(transition, "_permutation_cache")
        assert isinstance(transition._permutation_cache, dict)

    def test_get_transition_name(self):
        """Test transition name."""
        from src.transitions.led_random_transition import LEDRandomTransition

        transition = LEDRandomTransition()
        name = transition.get_transition_name()
        assert "random" in name.lower()

    def test_validate_parameters_valid(self):
        """Test parameter validation with valid parameters."""
        from src.transitions.led_random_transition import LEDRandomTransition

        transition = LEDRandomTransition()

        valid_params = {"duration": 1.0, "leds_per_frame": 10, "seed": 42}
        assert transition.validate_parameters(valid_params) is True

    def test_validate_parameters_empty(self):
        """Test parameter validation with empty parameters."""
        from src.transitions.led_random_transition import LEDRandomTransition

        transition = LEDRandomTransition()
        assert transition.validate_parameters({}) is True

    def test_get_transition_region_in(self, random_config):
        """Test getting transition region for random-in."""
        from src.transitions.led_random_transition import LEDRandomTransition

        transition = LEDRandomTransition()

        start, end = transition.get_transition_region(10.0, random_config, "in")

        assert start == 0.0
        assert end == 1.0

    def test_get_transition_region_out(self, random_config):
        """Test getting transition region for random-out."""
        from src.transitions.led_random_transition import LEDRandomTransition

        transition = LEDRandomTransition()

        start, end = transition.get_transition_region(10.0, random_config, "out")

        assert start == 9.0
        assert end == 10.0

    def test_is_in_transition_region(self, random_config):
        """Test is_in_transition_region."""
        from src.transitions.led_random_transition import LEDRandomTransition

        transition = LEDRandomTransition()

        assert transition.is_in_transition_region(0.5, 10.0, random_config, "in") is True
        assert transition.is_in_transition_region(5.0, 10.0, random_config, "in") is False
        assert transition.is_in_transition_region(9.5, 10.0, random_config, "out") is True

    def test_get_parameter_schema(self):
        """Test parameter schema includes random-specific fields."""
        from src.transitions.led_random_transition import LEDRandomTransition

        transition = LEDRandomTransition()
        schema = transition.get_parameter_schema()

        assert "properties" in schema
        assert "duration" in schema["properties"]


# =============================================================================
# Base LED Transition Tests
# =============================================================================


class TestBaseLEDTransition:
    """Test BaseLEDTransition base class methods."""

    def test_get_transition_progress_before_region(self):
        """Test progress is 0.0 before transition region."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()
        config = {"type": "ledfade", "parameters": {"duration": 1.0}}

        # For fade-out at start of item (before transition region)
        progress = transition.get_transition_progress(-0.5, 10.0, config, "in")
        assert progress == 0.0

    def test_get_transition_progress_after_region(self):
        """Test progress is 1.0 after transition region."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()
        config = {"type": "ledfade", "parameters": {"duration": 1.0}}

        # For fade-in after transition region
        progress = transition.get_transition_progress(5.0, 10.0, config, "in")
        assert progress == 1.0

    def test_get_transition_progress_clamped(self):
        """Test progress is clamped between 0.0 and 1.0."""
        from src.transitions.led_fade_transition import LEDFadeTransition

        transition = LEDFadeTransition()
        config = {"type": "ledfade", "parameters": {"duration": 1.0}}

        # At various points in fade-in
        progress = transition.get_transition_progress(0.0, 10.0, config, "in")
        assert 0.0 <= progress <= 1.0

        progress = transition.get_transition_progress(0.5, 10.0, config, "in")
        assert 0.0 <= progress <= 1.0

        progress = transition.get_transition_progress(1.0, 10.0, config, "in")
        assert 0.0 <= progress <= 1.0
