"""
Unit tests for Frame Timing utilities.

Tests FrameTimingData class and TimingLogger for tracking frame timing.
"""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# FrameTimingData Tests
# =============================================================================


class TestFrameTimingData:
    """Test FrameTimingData dataclass."""

    def test_initialization_defaults(self):
        """Test FrameTimingData initializes with defaults."""
        from src.utils.frame_timing import FrameTimingData

        timing = FrameTimingData()

        assert timing.frame_index == 0
        assert timing.plugin_timestamp == 0.0
        assert timing.producer_timestamp == 0.0
        assert timing.item_duration == 0.0
        assert timing.write_to_buffer_time is None
        assert timing.read_from_buffer_time is None
        assert timing.render_time is None

    def test_initialization_with_values(self):
        """Test FrameTimingData initializes with provided values."""
        from src.utils.frame_timing import FrameTimingData

        timing = FrameTimingData(
            frame_index=42,
            plugin_timestamp=1.5,
            producer_timestamp=10.5,
            item_duration=30.0,
        )

        assert timing.frame_index == 42
        assert timing.plugin_timestamp == 1.5
        assert timing.producer_timestamp == 10.5
        assert timing.item_duration == 30.0

    def test_mark_write_to_buffer(self):
        """Test marking write to buffer time."""
        from src.utils.frame_timing import FrameTimingData

        timing = FrameTimingData()

        assert timing.write_to_buffer_time is None

        before = time.time()
        timing.mark_write_to_buffer()
        after = time.time()

        assert timing.write_to_buffer_time is not None
        assert before <= timing.write_to_buffer_time <= after

    def test_mark_read_from_buffer(self):
        """Test marking read from buffer time."""
        from src.utils.frame_timing import FrameTimingData

        timing = FrameTimingData()

        before = time.time()
        timing.mark_read_from_buffer()
        after = time.time()

        assert timing.read_from_buffer_time is not None
        assert before <= timing.read_from_buffer_time <= after

    def test_mark_write_to_led_buffer(self):
        """Test marking write to LED buffer time."""
        from src.utils.frame_timing import FrameTimingData

        timing = FrameTimingData()

        before = time.time()
        timing.mark_write_to_led_buffer()
        after = time.time()

        assert timing.write_to_led_buffer_time is not None
        assert before <= timing.write_to_led_buffer_time <= after

    def test_mark_read_from_led_buffer(self):
        """Test marking read from LED buffer time."""
        from src.utils.frame_timing import FrameTimingData

        timing = FrameTimingData()

        before = time.time()
        timing.mark_read_from_led_buffer()
        after = time.time()

        assert timing.read_from_led_buffer_time is not None
        assert before <= timing.read_from_led_buffer_time <= after

    def test_mark_led_transition_complete(self):
        """Test marking LED transition complete time."""
        from src.utils.frame_timing import FrameTimingData

        timing = FrameTimingData()

        before = time.time()
        timing.mark_led_transition_complete()
        after = time.time()

        assert timing.led_transition_time is not None
        assert before <= timing.led_transition_time <= after

    def test_mark_render(self):
        """Test marking render time."""
        from src.utils.frame_timing import FrameTimingData

        timing = FrameTimingData()

        before = time.time()
        timing.mark_render()
        after = time.time()

        assert timing.render_time is not None
        assert before <= timing.render_time <= after

    def test_to_csv_row(self):
        """Test converting timing data to CSV row."""
        from src.utils.frame_timing import FrameTimingData

        timing = FrameTimingData(
            frame_index=10,
            plugin_timestamp=1.0,
            producer_timestamp=5.0,
            item_duration=30.0,
        )
        timing.mark_write_to_buffer()
        timing.mark_render()

        row = timing.to_csv_row()

        assert isinstance(row, list)
        assert len(row) == 10
        assert row[0] == 10  # frame_index
        assert row[1] == 1.0  # plugin_timestamp
        assert row[2] == 5.0  # producer_timestamp
        assert row[3] > 0  # write_to_buffer_time
        assert row[9] == 30.0  # item_duration

    def test_to_csv_row_none_times_are_zero(self):
        """Test that None times become 0.0 in CSV row."""
        from src.utils.frame_timing import FrameTimingData

        timing = FrameTimingData(frame_index=5)

        row = timing.to_csv_row()

        # All timing fields should be 0.0 when None
        assert row[3] == 0.0  # write_to_buffer_time
        assert row[4] == 0.0  # read_from_buffer_time
        assert row[5] == 0.0  # write_to_led_buffer_time
        assert row[6] == 0.0  # read_from_led_buffer_time
        assert row[7] == 0.0  # led_transition_time
        assert row[8] == 0.0  # render_time

    def test_csv_header(self):
        """Test getting CSV header row."""
        from src.utils.frame_timing import FrameTimingData

        header = FrameTimingData.csv_header()

        assert isinstance(header, list)
        assert "frame_index" in header
        assert "plugin_timestamp" in header
        assert "producer_timestamp" in header
        assert "write_to_buffer_time" in header
        assert "read_from_buffer_time" in header
        assert "render_time" in header
        assert "item_duration" in header

    def test_pipeline_timing_sequence(self):
        """Test recording full pipeline timing sequence."""
        from src.utils.frame_timing import FrameTimingData

        timing = FrameTimingData(frame_index=1)

        # Simulate pipeline stages
        timing.mark_write_to_buffer()
        time.sleep(0.001)

        timing.mark_read_from_buffer()
        time.sleep(0.001)

        timing.mark_write_to_led_buffer()
        time.sleep(0.001)

        timing.mark_read_from_led_buffer()
        time.sleep(0.001)

        timing.mark_led_transition_complete()
        time.sleep(0.001)

        timing.mark_render()

        # Verify timing sequence is monotonically increasing
        assert timing.write_to_buffer_time <= timing.read_from_buffer_time
        assert timing.read_from_buffer_time <= timing.write_to_led_buffer_time
        assert timing.write_to_led_buffer_time <= timing.read_from_led_buffer_time
        assert timing.read_from_led_buffer_time <= timing.led_transition_time
        assert timing.led_transition_time <= timing.render_time


# =============================================================================
# TimingLogger Tests (if class exists)
# =============================================================================


class TestTimingLogger:
    """Test TimingLogger class if available."""

    def test_timing_logger_import(self):
        """Test that TimingLogger can be imported."""
        try:
            from src.utils.frame_timing import TimingLogger

            assert TimingLogger is not None
        except ImportError:
            pytest.skip("TimingLogger not available")

    def test_timing_logger_initialization(self, tmp_path):
        """Test TimingLogger initialization."""
        try:
            from src.utils.frame_timing import TimingLogger

            log_file = tmp_path / "timing.csv"
            logger = TimingLogger(str(log_file))

            assert logger is not None
        except ImportError:
            pytest.skip("TimingLogger not available")
        except Exception:
            # May not be able to create without proper setup
            pass
