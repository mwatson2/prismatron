"""
Unit tests for logging utilities with custom formatters.

Tests the AppTimeFormatter and related utilities for time tracking in logs.
"""

import logging
import sys
import time
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def reset_app_start_time():
    """Reset app start time before and after test."""
    from src.utils import logging_utils

    original = logging_utils._app_start_time
    logging_utils._app_start_time = None
    yield
    logging_utils._app_start_time = original


# =============================================================================
# AppTimeFormatter Tests
# =============================================================================


class TestAppTimeFormatter:
    """Test AppTimeFormatter class."""

    def test_basic_initialization(self):
        """Test basic formatter initialization."""
        from src.utils.logging_utils import AppTimeFormatter

        formatter = AppTimeFormatter()

        assert formatter.app_start_time is not None
        assert formatter.app_start_time <= time.time()

    def test_initialization_with_custom_start_time(self):
        """Test formatter with custom start time."""
        from src.utils.logging_utils import AppTimeFormatter

        custom_start = time.time() - 100  # 100 seconds ago
        formatter = AppTimeFormatter(app_start_time=custom_start)

        assert formatter.app_start_time == custom_start

    def test_initialization_with_format_string(self):
        """Test formatter with custom format string."""
        from src.utils.logging_utils import AppTimeFormatter

        fmt = "%(levelname)s - %(app_time)s - %(message)s"
        formatter = AppTimeFormatter(fmt=fmt)

        assert formatter._fmt == fmt

    def test_format_adds_app_time(self):
        """Test that format adds app_time to record."""
        from src.utils.logging_utils import AppTimeFormatter

        formatter = AppTimeFormatter(
            fmt="%(app_time)s - %(message)s",
            app_start_time=time.time() - 65,  # 1 minute 5 seconds ago
        )

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        # Should have app_time in mm:ss.xxx format
        assert "01:" in formatted  # At least 1 minute
        assert "Test message" in formatted

    def test_format_time_calculation(self):
        """Test time calculation in format."""
        from src.utils.logging_utils import AppTimeFormatter

        # Set start time to exactly 125.5 seconds ago (2:05.500)
        start_time = time.time() - 125.5
        formatter = AppTimeFormatter(
            fmt="%(app_time)s",
            app_start_time=start_time,
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        # Should be approximately 02:05.xxx
        assert formatted.startswith("02:05")

    def test_format_zero_time(self):
        """Test formatting at app start time."""
        from src.utils.logging_utils import AppTimeFormatter

        now = time.time()
        formatter = AppTimeFormatter(fmt="%(app_time)s", app_start_time=now)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        # Set record.created to match formatter's start time
        record.created = now

        formatted = formatter.format(record)

        # Should be 00:00.xxx (close to zero)
        assert formatted.startswith("00:00")


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunctions:
    """Test module-level utility functions."""

    def test_get_app_start_time_initializes(self, reset_app_start_time):
        """Test get_app_start_time initializes on first call."""
        from src.utils.logging_utils import get_app_start_time

        start_time = get_app_start_time()

        assert start_time is not None
        assert start_time <= time.time()

    def test_get_app_start_time_returns_same(self, reset_app_start_time):
        """Test get_app_start_time returns same value."""
        from src.utils.logging_utils import get_app_start_time

        time1 = get_app_start_time()
        time.sleep(0.01)  # Small delay
        time2 = get_app_start_time()

        assert time1 == time2

    def test_set_app_start_time(self, reset_app_start_time):
        """Test set_app_start_time updates the value."""
        from src.utils.logging_utils import get_app_start_time, set_app_start_time

        custom_time = 1234567890.0
        set_app_start_time(custom_time)

        assert get_app_start_time() == custom_time

    def test_create_app_time_formatter_default(self, reset_app_start_time):
        """Test create_app_time_formatter with defaults."""
        from src.utils.logging_utils import create_app_time_formatter

        formatter = create_app_time_formatter()

        assert formatter is not None
        assert "%(app_time)s" in formatter._fmt

    def test_create_app_time_formatter_custom_format(self, reset_app_start_time):
        """Test create_app_time_formatter with custom format."""
        from src.utils.logging_utils import create_app_time_formatter

        custom_fmt = "%(levelname)s: %(message)s"
        formatter = create_app_time_formatter(fmt=custom_fmt)

        assert formatter._fmt == custom_fmt

    def test_create_app_time_formatter_uses_global_start_time(self, reset_app_start_time):
        """Test that formatter uses global start time."""
        from src.utils.logging_utils import (
            create_app_time_formatter,
            get_app_start_time,
            set_app_start_time,
        )

        custom_time = time.time() - 500
        set_app_start_time(custom_time)

        formatter = create_app_time_formatter()

        assert formatter.app_start_time == custom_time


# =============================================================================
# Integration Tests
# =============================================================================


class TestLoggingIntegration:
    """Integration tests with Python logging system."""

    def test_formatter_with_logger(self, reset_app_start_time):
        """Test formatter works with Python logger."""
        from src.utils.logging_utils import AppTimeFormatter

        # Create a logger with our formatter
        logger = logging.getLogger("test_integration")
        logger.setLevel(logging.DEBUG)

        # Create handler with our formatter
        handler = logging.StreamHandler()
        formatter = AppTimeFormatter(
            fmt="%(app_time)s - %(name)s - %(message)s",
            app_start_time=time.time(),
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        try:
            # This should not raise any exceptions
            logger.info("Test log message")
        finally:
            logger.removeHandler(handler)
