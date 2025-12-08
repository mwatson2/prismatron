"""
Unit tests for LogRotator class and log rotation utilities.

Tests the log rotation functionality used to prevent log files from
filling up disk space.
"""

import logging
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import the log_rotation module
try:
    from src.utils.log_rotation import LogRotator

    LOG_ROTATION_AVAILABLE = True
except ImportError:
    LOG_ROTATION_AVAILABLE = False
    LogRotator = None

# Skip all tests if module not available
pytestmark = pytest.mark.skipif(not LOG_ROTATION_AVAILABLE, reason="log_rotation module not importable")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary directory for log files."""
    return tmp_path


@pytest.fixture
def temp_log_file(temp_log_dir):
    """Create a temporary log file."""
    log_file = temp_log_dir / "prismatron.log"
    log_file.touch()
    return log_file


@pytest.fixture
def log_rotator(temp_log_file):
    """Create a LogRotator instance with temp log file."""
    rotator = LogRotator(
        log_file_path=str(temp_log_file),
        check_interval=1,  # Short interval for testing
    )
    yield rotator
    # Cleanup
    if rotator.is_running():
        rotator.stop(timeout=2.0)


# =============================================================================
# LogRotator Initialization Tests
# =============================================================================


class TestLogRotatorInit:
    """Test LogRotator initialization."""

    def test_basic_initialization(self, temp_log_file):
        """Test basic initialization with defaults."""
        rotator = LogRotator(log_file_path=str(temp_log_file))

        assert rotator.log_file_path == temp_log_file
        assert rotator.check_interval == 300  # Default
        assert rotator.is_running() is False

    def test_initialization_with_custom_interval(self, temp_log_file):
        """Test initialization with custom check interval."""
        rotator = LogRotator(log_file_path=str(temp_log_file), check_interval=60)

        assert rotator.check_interval == 60

    def test_backup_path_creation(self, temp_log_file):
        """Test that backup path is created correctly."""
        rotator = LogRotator(log_file_path=str(temp_log_file))

        expected_backup = temp_log_file.with_name("prismatron.1.log")
        assert rotator.backup_log_path == expected_backup


# =============================================================================
# File Size Tests
# =============================================================================


class TestLogRotatorFileSize:
    """Test file size calculation methods."""

    def test_get_file_size_mb_empty_file(self, log_rotator, temp_log_file):
        """Test getting size of empty file."""
        size = log_rotator.get_file_size_mb(temp_log_file)
        assert size == 0.0

    def test_get_file_size_mb_with_content(self, log_rotator, temp_log_file):
        """Test getting size of file with content."""
        # Write 1KB of data
        temp_log_file.write_text("x" * 1024)

        size = log_rotator.get_file_size_mb(temp_log_file)
        assert 0.0009 < size < 0.0011  # ~0.001 MB

    def test_get_file_size_mb_nonexistent(self, log_rotator, temp_log_dir):
        """Test getting size of nonexistent file."""
        nonexistent = temp_log_dir / "nonexistent.log"
        size = log_rotator.get_file_size_mb(nonexistent)
        assert size == 0.0


# =============================================================================
# Rotation Tests
# =============================================================================


class TestLogRotation:
    """Test log rotation functionality."""

    def test_rotation_not_needed_small_file(self, log_rotator, temp_log_file):
        """Test that small files are not rotated."""
        temp_log_file.write_text("small content")

        result = log_rotator.rotate_log_file()
        assert result is False

    def test_rotate_log_file_nonexistent(self, log_rotator, temp_log_file):
        """Test rotation when log file doesn't exist."""
        temp_log_file.unlink()

        result = log_rotator.rotate_log_file()
        assert result is False


# =============================================================================
# Start/Stop Tests
# =============================================================================


class TestLogRotatorStartStop:
    """Test starting and stopping the log rotator."""

    def test_start_rotator(self, log_rotator):
        """Test starting the rotator."""
        result = log_rotator.start()

        assert result is True
        assert log_rotator.is_running() is True

    def test_start_already_running(self, log_rotator):
        """Test starting when already running."""
        log_rotator.start()

        result = log_rotator.start()
        assert result is False  # Already running

    def test_stop_rotator(self, log_rotator):
        """Test stopping the rotator."""
        log_rotator.start()
        assert log_rotator.is_running() is True

        result = log_rotator.stop(timeout=2.0)
        assert result is True
        assert log_rotator.is_running() is False

    def test_stop_not_running(self, log_rotator):
        """Test stopping when not running."""
        result = log_rotator.stop()
        assert result is True


# =============================================================================
# Log Stats Tests
# =============================================================================


class TestLogStats:
    """Test log statistics functionality."""

    def test_get_log_stats(self, log_rotator, temp_log_file):
        """Test getting log statistics."""
        temp_log_file.write_text("test content")

        stats = log_rotator.get_log_stats()

        assert "main_log_path" in stats
        assert "main_log_size_mb" in stats
        assert "backup_log_path" in stats
        assert "backup_log_size_mb" in stats
        assert "total_size_mb" in stats
        assert "max_size_mb" in stats
        assert "rotation_needed" in stats
        assert "is_running" in stats
        assert "check_interval_seconds" in stats

    def test_get_log_stats_values(self, log_rotator, temp_log_file):
        """Test log statistics values."""
        temp_log_file.write_text("x" * 1024)

        stats = log_rotator.get_log_stats()

        assert stats["main_log_path"] == str(temp_log_file)
        assert stats["main_log_size_mb"] > 0
        assert stats["backup_log_size_mb"] == 0  # No backup yet
        assert stats["is_running"] is False


# =============================================================================
# Module-level Functions Tests
# =============================================================================


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_get_log_rotator_singleton(self, temp_log_file):
        """Test that get_log_rotator returns singleton."""
        from src.utils import log_rotation

        # Reset global state
        log_rotation._global_rotator = None

        rotator1 = log_rotation.get_log_rotator(str(temp_log_file))
        rotator2 = log_rotation.get_log_rotator(str(temp_log_file))

        assert rotator1 is rotator2

        # Cleanup
        log_rotation._global_rotator = None

    def test_get_log_stats_no_rotator(self, temp_log_file):
        """Test get_log_stats when no rotator initialized."""
        from src.utils import log_rotation

        # Reset global state
        original = log_rotation._global_rotator
        log_rotation._global_rotator = None

        stats = log_rotation.get_log_stats()

        assert "error" in stats

        # Restore
        log_rotation._global_rotator = original

    def test_stop_log_rotation_no_rotator(self, temp_log_file):
        """Test stop_log_rotation when no rotator."""
        from src.utils import log_rotation

        original = log_rotation._global_rotator
        log_rotation._global_rotator = None

        result = log_rotation.stop_log_rotation()
        assert result is True

        log_rotation._global_rotator = original
