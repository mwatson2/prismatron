"""
Unit tests for LEDBuffer ring buffer class.

Tests the LED values ring buffer used for optimized LED value storage
with timestamp metadata.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

pytest.importorskip("cupy")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def led_buffer():
    """Create a LEDBuffer instance."""
    from src.consumer.led_buffer import LEDBuffer

    return LEDBuffer(led_count=100, buffer_size=10)


@pytest.fixture
def sample_led_values():
    """Create sample LED values."""
    return np.random.randint(0, 256, (100, 3), dtype=np.uint8)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestLEDBufferInit:
    """Test LEDBuffer initialization."""

    def test_basic_initialization(self):
        """Test basic buffer initialization."""
        from src.consumer.led_buffer import LEDBuffer

        buffer = LEDBuffer(led_count=100, buffer_size=10)

        assert buffer.led_count == 100
        assert buffer.buffer_size == 10
        assert buffer.count == 0
        assert len(buffer) == 0

    def test_initialization_default_buffer_size(self):
        """Test initialization with default buffer size."""
        from src.consumer.led_buffer import LEDBuffer

        buffer = LEDBuffer(led_count=50)

        assert buffer.led_count == 50
        assert buffer.buffer_size == 100  # Default

    def test_array_shapes(self, led_buffer):
        """Test that internal arrays have correct shapes."""
        assert led_buffer.led_arrays.shape == (10, 100, 3)
        assert led_buffer.timestamps.shape == (10,)

    def test_memory_usage(self, led_buffer):
        """Test memory usage calculation."""
        memory_mb = led_buffer._get_memory_usage_mb()

        # Expected: (10 * 100 * 3) + (10 * 8) bytes
        # = 3000 + 80 = 3080 bytes ≈ 0.003 MB
        assert 0.002 < memory_mb < 0.004


# =============================================================================
# Write Tests
# =============================================================================


class TestLEDBufferWrite:
    """Test writing to LED buffer."""

    def test_write_single_frame(self, led_buffer, sample_led_values):
        """Test writing a single frame."""
        result = led_buffer.write_led_values(
            led_values=sample_led_values,
            timestamp=1.0,
        )

        assert result is True
        assert led_buffer.count == 1
        assert led_buffer.frames_written == 1

    def test_write_multiple_frames(self, led_buffer, sample_led_values):
        """Test writing multiple frames."""
        for i in range(5):
            result = led_buffer.write_led_values(
                led_values=sample_led_values,
                timestamp=float(i),
            )
            assert result is True

        assert led_buffer.count == 5
        assert led_buffer.frames_written == 5

    def test_write_with_metadata(self, led_buffer, sample_led_values):
        """Test writing with metadata."""
        metadata = {"source": "test", "frame_id": 42}
        result = led_buffer.write_led_values(
            led_values=sample_led_values,
            timestamp=1.0,
            metadata=metadata,
        )

        assert result is True

    def test_write_transposed_format(self, led_buffer):
        """Test writing LED values in (3, led_count) format."""
        # Create values in (3, led_count) format
        led_values = np.random.randint(0, 256, (3, 100), dtype=np.uint8)

        result = led_buffer.write_led_values(
            led_values=led_values,
            timestamp=1.0,
        )

        assert result is True

    def test_write_invalid_shape(self, led_buffer):
        """Test writing with invalid shape."""
        invalid_values = np.random.randint(0, 256, (50, 3), dtype=np.uint8)

        result = led_buffer.write_led_values(
            led_values=invalid_values,
            timestamp=1.0,
        )

        assert result is False

    def test_write_overflow_nonblocking(self, led_buffer, sample_led_values):
        """Test buffer overflow in non-blocking mode."""
        # Fill buffer completely
        for i in range(10):
            led_buffer.write_led_values(sample_led_values, float(i))

        assert led_buffer.count == 10

        # Write one more - should drop oldest
        led_buffer.write_led_values(sample_led_values, 10.0, block=False)

        assert led_buffer.count == 10
        assert led_buffer.overflow_count == 1


# =============================================================================
# Read Tests
# =============================================================================


class TestLEDBufferRead:
    """Test reading from LED buffer."""

    def test_read_single_frame(self, led_buffer, sample_led_values):
        """Test reading a single frame."""
        led_buffer.write_led_values(sample_led_values, 1.0)

        result = led_buffer.read_led_values(timeout=1.0)

        assert result is not None
        led_values, timestamp, metadata = result
        assert led_values.shape == (100, 3)
        assert timestamp == 1.0
        assert led_buffer.count == 0

    def test_read_preserves_order(self, led_buffer, sample_led_values):
        """Test that read preserves FIFO order."""
        for i in range(5):
            led_buffer.write_led_values(sample_led_values, float(i))

        for i in range(5):
            result = led_buffer.read_led_values(timeout=1.0)
            assert result is not None
            _, timestamp, _ = result
            assert timestamp == float(i)

    def test_read_empty_buffer_timeout(self, led_buffer):
        """Test reading from empty buffer with timeout."""
        result = led_buffer.read_led_values(timeout=0.1)

        assert result is None
        assert led_buffer.underflow_count == 1

    def test_read_latest(self, led_buffer, sample_led_values):
        """Test reading latest frame without consuming."""
        for i in range(5):
            led_buffer.write_led_values(sample_led_values, float(i))

        result = led_buffer.read_latest_led_values()

        assert result is not None
        _, timestamp, _ = result
        assert timestamp == 4.0  # Latest
        assert led_buffer.count == 5  # Not consumed

    def test_read_latest_empty(self, led_buffer):
        """Test reading latest from empty buffer."""
        result = led_buffer.read_latest_led_values()

        assert result is None


# =============================================================================
# Peek Tests
# =============================================================================


class TestLEDBufferPeek:
    """Test peeking at buffer."""

    def test_peek_next_timestamp(self, led_buffer, sample_led_values):
        """Test peeking at next timestamp."""
        led_buffer.write_led_values(sample_led_values, 1.5)

        timestamp = led_buffer.peek_next_timestamp()

        assert timestamp == 1.5
        assert led_buffer.count == 1  # Not consumed

    def test_peek_empty_buffer(self, led_buffer):
        """Test peeking at empty buffer."""
        timestamp = led_buffer.peek_next_timestamp()

        assert timestamp is None


# =============================================================================
# Stats Tests
# =============================================================================


class TestLEDBufferStats:
    """Test buffer statistics."""

    def test_get_buffer_stats(self, led_buffer, sample_led_values):
        """Test getting buffer statistics."""
        for i in range(5):
            led_buffer.write_led_values(sample_led_values, float(i))

        led_buffer.read_led_values(timeout=1.0)
        led_buffer.read_led_values(timeout=1.0)

        stats = led_buffer.get_buffer_stats()

        assert stats["buffer_size"] == 10
        assert stats["current_count"] == 3
        assert stats["frames_written"] == 5
        assert stats["frames_read"] == 2
        assert 0.29 < stats["utilization"] < 0.31  # 3/10 = 0.3
        assert stats["is_full"] is False
        assert stats["is_empty"] is False

    def test_stats_empty_buffer(self, led_buffer):
        """Test stats for empty buffer."""
        stats = led_buffer.get_buffer_stats()

        assert stats["current_count"] == 0
        assert stats["utilization"] == 0.0
        assert stats["is_empty"] is True

    def test_stats_full_buffer(self, led_buffer, sample_led_values):
        """Test stats for full buffer."""
        for i in range(10):
            led_buffer.write_led_values(sample_led_values, float(i))

        stats = led_buffer.get_buffer_stats()

        assert stats["current_count"] == 10
        assert stats["utilization"] == 1.0
        assert stats["is_full"] is True


# =============================================================================
# Clear and Resize Tests
# =============================================================================


class TestLEDBufferClearResize:
    """Test clearing and resizing buffer."""

    def test_clear_buffer(self, led_buffer, sample_led_values):
        """Test clearing the buffer."""
        for i in range(5):
            led_buffer.write_led_values(sample_led_values, float(i))

        led_buffer.clear()

        assert led_buffer.count == 0
        assert led_buffer.write_index == 0
        assert led_buffer.read_index == 0

    def test_resize_buffer(self, led_buffer, sample_led_values):
        """Test resizing the buffer."""
        for i in range(5):
            led_buffer.write_led_values(sample_led_values, float(i))

        result = led_buffer.set_buffer_size(20)

        assert result is True
        assert led_buffer.buffer_size == 20
        assert led_buffer.count == 0  # Data cleared on resize

    def test_resize_invalid_size(self, led_buffer):
        """Test resizing with invalid size."""
        result = led_buffer.set_buffer_size(0)

        assert result is False
        assert led_buffer.buffer_size == 10  # Unchanged


# =============================================================================
# Dunder Methods Tests
# =============================================================================


class TestLEDBufferDunderMethods:
    """Test dunder methods."""

    def test_len(self, led_buffer, sample_led_values):
        """Test __len__ method."""
        assert len(led_buffer) == 0

        led_buffer.write_led_values(sample_led_values, 1.0)
        assert len(led_buffer) == 1

    def test_bool_empty(self, led_buffer):
        """Test __bool__ for empty buffer."""
        assert bool(led_buffer) is False

    def test_bool_not_empty(self, led_buffer, sample_led_values):
        """Test __bool__ for non-empty buffer."""
        led_buffer.write_led_values(sample_led_values, 1.0)

        assert bool(led_buffer) is True


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestLEDBufferThreading:
    """Test thread safety of buffer operations."""

    def test_concurrent_write_read(self, led_buffer):
        """Test concurrent write and read operations."""
        results = []
        write_count = 50
        read_count = 50

        def writer():
            for i in range(write_count):
                led_values = np.random.randint(0, 256, (100, 3), dtype=np.uint8)
                led_buffer.write_led_values(led_values, float(i))
                time.sleep(0.001)

        def reader():
            for _ in range(read_count):
                result = led_buffer.read_led_values(timeout=1.0)
                if result is not None:
                    results.append(result)
                time.sleep(0.001)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        # Some reads should have succeeded
        assert len(results) > 0
        assert led_buffer.frames_written == write_count
