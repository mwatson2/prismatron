"""
Unit tests for Pattern Loader utilities.

Tests functions for loading diffusion patterns and LED ordering.
"""

import sys
import tempfile
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
def temp_pattern_file(tmp_path):
    """Create a temporary pattern file with valid LED ordering."""
    pattern_file = tmp_path / "test_pattern.npz"

    # Create valid LED ordering (permutation of 0 to 99)
    led_ordering = np.arange(100, dtype=np.int32)
    np.random.shuffle(led_ordering)

    # Create metadata
    metadata = {
        "led_count": 100,
        "frame_width": 64,
        "frame_height": 64,
        "block_size": 64,
        "pattern_type": "diffusion",
        "sparsity_threshold": 0.001,
    }

    # Save pattern
    np.savez(
        pattern_file,
        led_ordering=led_ordering,
        metadata=metadata,
        mixed_tensor=np.zeros((100, 100)),
    )

    return str(pattern_file)


@pytest.fixture
def temp_pattern_file_no_ordering(tmp_path):
    """Create a temporary pattern file without LED ordering."""
    pattern_file = tmp_path / "test_pattern_no_ordering.npz"

    # Save pattern without LED ordering
    np.savez(
        pattern_file,
        mixed_tensor=np.zeros((100, 100)),
    )

    return str(pattern_file)


@pytest.fixture
def valid_led_ordering():
    """Create a valid LED ordering (permutation)."""
    ordering = np.arange(100, dtype=np.int32)
    np.random.shuffle(ordering)
    return ordering


@pytest.fixture
def invalid_led_ordering_duplicates():
    """Create invalid LED ordering with duplicates."""
    return np.array([0, 1, 1, 3, 4], dtype=np.int32)  # Duplicate 1


@pytest.fixture
def invalid_led_ordering_2d():
    """Create invalid 2D LED ordering."""
    return np.arange(100).reshape(10, 10)


# =============================================================================
# validate_led_ordering Tests
# =============================================================================


class TestValidateLedOrdering:
    """Test validate_led_ordering function."""

    def test_valid_ordering(self, valid_led_ordering):
        """Test validation of a valid LED ordering."""
        from src.utils.pattern_loader import validate_led_ordering

        is_valid, message = validate_led_ordering(valid_led_ordering)

        assert is_valid is True
        assert "Valid" in message

    def test_invalid_ordering_duplicates(self, invalid_led_ordering_duplicates):
        """Test validation fails for ordering with duplicates."""
        from src.utils.pattern_loader import validate_led_ordering

        is_valid, message = validate_led_ordering(invalid_led_ordering_duplicates)

        assert is_valid is False
        assert "duplicates" in message.lower() or "permutation" in message.lower()

    def test_invalid_ordering_2d(self, invalid_led_ordering_2d):
        """Test validation fails for 2D array."""
        from src.utils.pattern_loader import validate_led_ordering

        is_valid, message = validate_led_ordering(invalid_led_ordering_2d)

        assert is_valid is False
        assert "1D" in message

    def test_invalid_ordering_not_permutation(self):
        """Test validation fails when not a proper permutation."""
        from src.utils.pattern_loader import validate_led_ordering

        # Missing values (not a permutation of 0 to n-1)
        invalid = np.array([0, 2, 3, 5, 7], dtype=np.int32)  # Missing 1, 4, 6

        is_valid, message = validate_led_ordering(invalid)

        assert is_valid is False
        assert "permutation" in message.lower()


# =============================================================================
# load_led_ordering_from_pattern Tests
# =============================================================================


class TestLoadLedOrderingFromPattern:
    """Test load_led_ordering_from_pattern function."""

    def test_load_valid_pattern(self, temp_pattern_file):
        """Test loading LED ordering from valid pattern file."""
        from src.utils.pattern_loader import load_led_ordering_from_pattern

        ordering = load_led_ordering_from_pattern(temp_pattern_file)

        assert ordering is not None
        assert len(ordering) == 100
        assert ordering.ndim == 1

    def test_load_pattern_no_ordering(self, temp_pattern_file_no_ordering):
        """Test loading returns None when no LED ordering in file."""
        from src.utils.pattern_loader import load_led_ordering_from_pattern

        ordering = load_led_ordering_from_pattern(temp_pattern_file_no_ordering)

        assert ordering is None

    def test_load_nonexistent_file(self):
        """Test loading returns None for nonexistent file."""
        from src.utils.pattern_loader import load_led_ordering_from_pattern

        ordering = load_led_ordering_from_pattern("/nonexistent/path.npz")

        assert ordering is None


# =============================================================================
# load_pattern_info Tests
# =============================================================================


class TestLoadPatternInfo:
    """Test load_pattern_info function."""

    def test_load_info_valid_file(self, temp_pattern_file):
        """Test loading pattern info from valid file."""
        from src.utils.pattern_loader import load_pattern_info

        info = load_pattern_info(temp_pattern_file)

        assert info["file_exists"] is True
        assert info["has_led_ordering"] is True
        assert info["has_mixed_tensor"] is True
        assert "keys" in info
        assert isinstance(info["keys"], list)

    def test_load_info_with_metadata(self, temp_pattern_file):
        """Test pattern info includes metadata fields."""
        from src.utils.pattern_loader import load_pattern_info

        info = load_pattern_info(temp_pattern_file)

        assert info.get("led_count") == 100
        assert info.get("frame_width") == 64
        assert info.get("frame_height") == 64
        assert info.get("block_size") == 64

    def test_load_info_includes_led_ordering_shape(self, temp_pattern_file):
        """Test pattern info includes LED ordering shape."""
        from src.utils.pattern_loader import load_pattern_info

        info = load_pattern_info(temp_pattern_file)

        assert "led_ordering_shape" in info
        assert info["led_ordering_shape"] == (100,)

    def test_load_info_nonexistent_file(self):
        """Test loading info for nonexistent file."""
        from src.utils.pattern_loader import load_pattern_info

        info = load_pattern_info("/nonexistent/path.npz")

        assert info["file_exists"] is False
        assert "error" in info

    def test_load_info_no_led_ordering(self, temp_pattern_file_no_ordering):
        """Test loading info when no LED ordering present."""
        from src.utils.pattern_loader import load_pattern_info

        info = load_pattern_info(temp_pattern_file_no_ordering)

        assert info["file_exists"] is True
        assert info["has_led_ordering"] is False


# =============================================================================
# create_frame_renderer_with_pattern Tests
# =============================================================================


class TestCreateFrameRendererWithPattern:
    """Test create_frame_renderer_with_pattern function."""

    def test_create_renderer_with_pattern(self, temp_pattern_file):
        """Test creating frame renderer with pattern."""
        # FrameRenderer is imported inside the function from src.consumer.frame_renderer
        with patch("src.consumer.frame_renderer.FrameRenderer") as MockRenderer:
            mock_renderer = MagicMock()
            MockRenderer.return_value = mock_renderer

            from src.utils.pattern_loader import create_frame_renderer_with_pattern

            renderer = create_frame_renderer_with_pattern(temp_pattern_file)

            # Verify FrameRenderer was called with led_ordering
            MockRenderer.assert_called_once()
            call_kwargs = MockRenderer.call_args[1]
            assert call_kwargs["led_ordering"] is not None
            assert renderer is mock_renderer

    def test_create_renderer_without_ordering(self, temp_pattern_file_no_ordering):
        """Test creating frame renderer when no LED ordering."""
        with patch("src.consumer.frame_renderer.FrameRenderer") as MockRenderer:
            mock_renderer = MagicMock()
            MockRenderer.return_value = mock_renderer

            from src.utils.pattern_loader import create_frame_renderer_with_pattern

            renderer = create_frame_renderer_with_pattern(temp_pattern_file_no_ordering)

            # Verify FrameRenderer was called with led_ordering=None
            MockRenderer.assert_called_once()
            call_kwargs = MockRenderer.call_args[1]
            assert call_kwargs["led_ordering"] is None

    def test_create_renderer_with_custom_params(self, temp_pattern_file):
        """Test creating frame renderer with custom parameters."""
        with patch("src.consumer.frame_renderer.FrameRenderer") as MockRenderer:
            mock_renderer = MagicMock()
            MockRenderer.return_value = mock_renderer

            from src.utils.pattern_loader import create_frame_renderer_with_pattern

            renderer = create_frame_renderer_with_pattern(
                temp_pattern_file,
                first_frame_delay_ms=200.0,
                timing_tolerance_ms=10.0,
                late_frame_log_threshold_ms=100.0,
            )

            call_kwargs = MockRenderer.call_args[1]
            assert call_kwargs["first_frame_delay_ms"] == 200.0
            assert call_kwargs["timing_tolerance_ms"] == 10.0
            assert call_kwargs["late_frame_log_threshold_ms"] == 100.0
