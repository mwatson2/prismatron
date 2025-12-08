"""
Unit tests for TextContentSource.

Tests the text content source for rendering text on the LED display
with configurable fonts, colors, and animations.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Check if PIL is available
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Skip all tests if PIL not available
pytestmark = pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def basic_config():
    """Create a basic text configuration."""
    return json.dumps(
        {
            "text": "Hello",
            "font_size": 24,
            "fg_color": "#FFFFFF",
            "bg_color": "#000000",
            "animation": "static",
            "duration": 1.0,
            "fps": 10,
        }
    )


@pytest.fixture
def scroll_config():
    """Create a scrolling text configuration."""
    return json.dumps(
        {
            "text": "Scrolling",
            "font_size": 20,
            "animation": "scroll",
            "duration": 2.0,
            "fps": 10,
        }
    )


@pytest.fixture
def fade_config():
    """Create a fade animation configuration."""
    return json.dumps(
        {
            "text": "Fade",
            "font_size": 20,
            "animation": "fade",
            "duration": 1.0,
            "fps": 10,
        }
    )


@pytest.fixture
def nested_config():
    """Create a nested configuration (from web API)."""
    return json.dumps(
        {
            "config": {
                "text": "Nested",
                "font_size": 24,
                "animation": "static",
                "duration": 1.0,
            }
        }
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestTextContentSourceInit:
    """Test TextContentSource initialization."""

    def test_basic_initialization(self, basic_config):
        """Test basic text source initialization."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)

        assert source.text == "Hello"
        assert source.font_size == 24
        assert source.animation == "static"

    def test_default_values(self):
        """Test default values are applied."""
        from src.producer.content_sources.text_source import TextContentSource

        config = json.dumps({"text": "Test"})
        source = TextContentSource(config)

        assert source.text == "Test"
        assert source.font_family == "arial"
        assert source.font_style == "normal"
        assert source.alignment == "center"
        assert source.vertical_alignment == "center"

    def test_nested_config(self, nested_config):
        """Test nested configuration from web API."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(nested_config)

        assert source.text == "Nested"

    def test_invalid_json_raises(self):
        """Test invalid JSON raises ValueError."""
        from src.producer.content_sources.text_source import TextContentSource

        with pytest.raises(ValueError, match="Invalid text configuration JSON"):
            TextContentSource("not valid json")

    def test_content_info_populated(self, basic_config):
        """Test content info is populated correctly."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)

        assert source.content_info.width == 800  # FRAME_WIDTH
        assert source.content_info.height == 480  # FRAME_HEIGHT
        assert source.content_info.duration == 1.0
        assert source.content_info.fps == 10


# =============================================================================
# Color Parsing Tests
# =============================================================================


class TestColorParsing:
    """Test color parsing functionality."""

    def test_parse_hex_color(self, basic_config):
        """Test parsing hex color."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)

        assert source.fg_color == (255, 255, 255)
        assert source.bg_color == (0, 0, 0)

    def test_parse_named_colors(self):
        """Test parsing named colors."""
        from src.producer.content_sources.text_source import TextContentSource

        config = json.dumps({"text": "Test", "fg_color": "red", "bg_color": "blue"})
        source = TextContentSource(config)

        assert source.fg_color == (255, 0, 0)
        assert source.bg_color == (0, 0, 255)

    def test_parse_tuple_color(self):
        """Test parsing tuple/list colors."""
        from src.producer.content_sources.text_source import TextContentSource

        config = json.dumps({"text": "Test", "fg_color": [128, 64, 32]})
        source = TextContentSource(config)

        assert source.fg_color == (128, 64, 32)


# =============================================================================
# Setup Tests
# =============================================================================


class TestTextContentSourceSetup:
    """Test TextContentSource setup."""

    def test_setup_static(self, basic_config):
        """Test setup with static animation."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)
        result = source.setup()

        assert result is True
        assert source._frame_generated is True
        assert len(source._frames) == source.frame_count

    def test_setup_scroll(self, scroll_config):
        """Test setup with scroll animation."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(scroll_config)
        result = source.setup()

        assert result is True
        assert len(source._frames) == source.frame_count

    def test_setup_fade(self, fade_config):
        """Test setup with fade animation."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(fade_config)
        result = source.setup()

        assert result is True
        assert len(source._frames) == source.frame_count


# =============================================================================
# Frame Retrieval Tests
# =============================================================================


class TestFrameRetrieval:
    """Test frame retrieval functionality."""

    def test_get_next_frame(self, basic_config):
        """Test getting next frame."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)
        source.setup()

        frame_data = source.get_next_frame()

        assert frame_data is not None
        assert frame_data.array.shape == (3, 480, 800)  # Planar format
        assert frame_data.width == 800
        assert frame_data.height == 480

    def test_get_all_frames(self, basic_config):
        """Test getting all frames until end."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)
        source.setup()

        frames = []
        while True:
            frame = source.get_next_frame()
            if frame is None:
                break
            frames.append(frame)

        assert len(frames) == source.frame_count

    def test_get_frame_before_setup(self, basic_config):
        """Test getting frame before setup returns None."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)

        frame = source.get_next_frame()

        assert frame is None

    def test_frame_timestamps_sequential(self, basic_config):
        """Test frame timestamps are sequential."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)
        source.setup()

        timestamps = []
        while True:
            frame = source.get_next_frame()
            if frame is None:
                break
            timestamps.append(frame.presentation_timestamp)

        # Timestamps should be increasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]


# =============================================================================
# Duration and Seek Tests
# =============================================================================


class TestDurationAndSeek:
    """Test duration and seek functionality."""

    def test_get_duration(self, basic_config):
        """Test getting duration."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)

        assert source.get_duration() == 1.0

    def test_seek_valid_timestamp(self, basic_config):
        """Test seeking to valid timestamp."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)
        source.setup()

        result = source.seek(0.5)

        assert result is True

    def test_seek_invalid_timestamp(self, basic_config):
        """Test seeking to invalid timestamp."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)
        source.setup()

        result = source.seek(-1.0)
        assert result is False

        result = source.seek(10.0)  # Beyond duration
        assert result is False

    def test_seek_before_setup(self, basic_config):
        """Test seeking before setup returns False."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)

        result = source.seek(0.5)

        assert result is False


# =============================================================================
# Reset and Cleanup Tests
# =============================================================================


class TestResetAndCleanup:
    """Test reset and cleanup functionality."""

    def test_reset(self, basic_config):
        """Test resetting the source."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)
        source.setup()

        # Read some frames
        source.get_next_frame()
        source.get_next_frame()

        result = source.reset()

        assert result is True
        assert source._next_frame_index == 0
        assert source.current_frame == 0

    def test_reset_before_setup(self, basic_config):
        """Test reset before setup returns False."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)

        result = source.reset()

        assert result is False

    def test_cleanup(self, basic_config):
        """Test cleanup clears frames."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)
        source.setup()

        source.cleanup()

        assert len(source._frames) == 0
        assert source._frame_generated is False


# =============================================================================
# Font Handling Tests
# =============================================================================


class TestFontHandling:
    """Test font handling functionality."""

    def test_auto_font_size_static(self):
        """Test auto font size for static text."""
        from src.producer.content_sources.text_source import TextContentSource

        config = json.dumps({"text": "Auto", "font_size": "auto", "animation": "static"})
        source = TextContentSource(config)

        assert isinstance(source.font_size, int)
        assert source.font_size > 0

    def test_auto_font_size_scroll(self):
        """Test auto font size for scrolling text."""
        from src.producer.content_sources.text_source import TextContentSource

        config = json.dumps({"text": "Auto Scroll", "font_size": "auto", "animation": "scroll"})
        source = TextContentSource(config)

        assert isinstance(source.font_size, int)
        assert source.font_size > 0

    def test_font_style_bold(self):
        """Test bold font style."""
        from src.producer.content_sources.text_source import TextContentSource

        config = json.dumps({"text": "Bold", "font_style": "bold"})
        source = TextContentSource(config)

        assert source.font_style == "bold"

    def test_font_style_italic(self):
        """Test italic font style."""
        from src.producer.content_sources.text_source import TextContentSource

        config = json.dumps({"text": "Italic", "font_style": "italic"})
        source = TextContentSource(config)

        assert source.font_style == "italic"


# =============================================================================
# Animation Tests
# =============================================================================


class TestAnimations:
    """Test different animation types."""

    def test_static_frames_identical(self, basic_config):
        """Test static animation produces identical frames."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(basic_config)
        source.setup()

        # Get first two frames
        frame1 = source.get_next_frame()
        frame2 = source.get_next_frame()

        # Static frames should be identical
        np.testing.assert_array_equal(frame1.array, frame2.array)

    def test_scroll_frames_different(self, scroll_config):
        """Test scroll animation produces different frames."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(scroll_config)
        source.setup()

        # Get first and last frame
        frames = []
        while True:
            frame = source.get_next_frame()
            if frame is None:
                break
            frames.append(frame)

        # First and last frames should be different (unless text renders as blank on this system)
        # Some systems may not have fonts available, so frames might all be black
        if np.any(frames[0].array != 0) or np.any(frames[-1].array != 0):
            # Only check if at least one frame has non-zero content
            assert not np.array_equal(frames[0].array, frames[-1].array)
        else:
            # Text couldn't be rendered - skip the comparison
            pytest.skip("Text rendering not available on this system")

    def test_fade_frames_different(self, fade_config):
        """Test fade animation produces different frames."""
        from src.producer.content_sources.text_source import TextContentSource

        source = TextContentSource(fade_config)
        source.setup()

        frames = []
        while True:
            frame = source.get_next_frame()
            if frame is None:
                break
            frames.append(frame)

        # First and middle frames should be different (fade in)
        middle = len(frames) // 2
        assert not np.array_equal(frames[0].array, frames[middle].array)


# =============================================================================
# Alignment Tests
# =============================================================================


class TestAlignment:
    """Test text alignment options."""

    def test_left_alignment(self):
        """Test left alignment."""
        from src.producer.content_sources.text_source import TextContentSource

        config = json.dumps({"text": "Left", "alignment": "left"})
        source = TextContentSource(config)

        assert source.alignment == "left"

    def test_right_alignment(self):
        """Test right alignment."""
        from src.producer.content_sources.text_source import TextContentSource

        config = json.dumps({"text": "Right", "alignment": "right"})
        source = TextContentSource(config)

        assert source.alignment == "right"

    def test_vertical_top(self):
        """Test top vertical alignment."""
        from src.producer.content_sources.text_source import TextContentSource

        config = json.dumps({"text": "Top", "vertical_alignment": "top"})
        source = TextContentSource(config)

        assert source.vertical_alignment == "top"

    def test_vertical_bottom(self):
        """Test bottom vertical alignment."""
        from src.producer.content_sources.text_source import TextContentSource

        config = json.dumps({"text": "Bottom", "vertical_alignment": "bottom"})
        source = TextContentSource(config)

        assert source.vertical_alignment == "bottom"
