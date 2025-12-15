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


class TestFrameTimingLogger:
    """Test FrameTimingLogger class."""

    def test_timing_logger_import(self):
        """Test that FrameTimingLogger can be imported."""
        from src.utils.frame_timing import FrameTimingLogger

        assert FrameTimingLogger is not None

    def test_timing_logger_initialization(self, tmp_path):
        """Test FrameTimingLogger initialization."""
        from src.utils.frame_timing import FrameTimingLogger

        log_file = tmp_path / "timing.csv"
        logger = FrameTimingLogger(str(log_file))

        assert logger is not None
        assert logger.log_file_path == log_file

    def test_start_logging_creates_file(self, tmp_path):
        """Test that start_logging creates the CSV file with header."""
        from src.utils.frame_timing import FrameTimingData, FrameTimingLogger

        log_file = tmp_path / "timing.csv"
        logger = FrameTimingLogger(str(log_file))

        result = logger.start_logging()

        assert result is True
        assert log_file.exists()

        # Verify header was written
        with open(log_file) as f:
            header = f.readline().strip()
            expected_header = ",".join(FrameTimingData.csv_header())
            assert header == expected_header

        logger.stop_logging()

    def test_log_frame_writes_data(self, tmp_path):
        """Test that log_frame writes timing data to CSV."""
        from src.utils.frame_timing import FrameTimingData, FrameTimingLogger

        log_file = tmp_path / "timing.csv"

        with FrameTimingLogger(str(log_file)) as logger:
            timing = FrameTimingData(frame_index=42, plugin_timestamp=1.5)
            timing.mark_write_to_buffer()
            logger.log_frame(timing)

        # Verify data was written
        with open(log_file) as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 data row
            data_row = lines[1].strip().split(",")
            assert data_row[0] == "42"  # frame_index

    def test_context_manager(self, tmp_path):
        """Test FrameTimingLogger as context manager."""
        from src.utils.frame_timing import FrameTimingLogger

        log_file = tmp_path / "timing.csv"

        with FrameTimingLogger(str(log_file)) as logger:
            assert logger._file_handle is not None

        # After exiting context, file should be closed
        assert logger._file_handle is None

    def test_stop_logging_closes_file(self, tmp_path):
        """Test that stop_logging properly closes the file."""
        from src.utils.frame_timing import FrameTimingLogger

        log_file = tmp_path / "timing.csv"
        logger = FrameTimingLogger(str(log_file))
        logger.start_logging()

        assert logger._file_handle is not None

        logger.stop_logging()

        assert logger._file_handle is None
        assert logger._csv_writer is None
