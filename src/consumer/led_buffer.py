"""
LED Values Ring Buffer.

This module implements a ring buffer for optimized LED values with timestamp metadata.
Stores small LED arrays instead of full frames, allowing deep buffering to absorb
jitter in the optimization process.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# LED count is now passed as parameter - no import needed

logger = logging.getLogger(__name__)


class LEDBuffer:
    """
    Ring buffer for optimized LED values with timestamp metadata.

    Stores small LED arrays (3 * 2624 bytes) instead of full frames,
    allowing deep buffering to absorb jitter in optimization process.

    Memory usage: buffer_size * led_count * 3 bytes for LED values
    plus buffer_size * 8 bytes for timestamps.
    """

    def __init__(self, led_count: int, buffer_size: int = 100):
        """
        Initialize LED buffer.

        Args:
            led_count: Number of LEDs (must be provided from pattern file)
            buffer_size: Number of LED value frames to buffer
        """
        self.buffer_size = buffer_size
        self.led_count = led_count

        # LED value storage (led_count, 3) format for each frame
        self.led_arrays = np.zeros((buffer_size, led_count, 3), dtype=np.uint8)
        self.timestamps = np.zeros(buffer_size, dtype=np.float64)
        self.metadata = [None] * buffer_size

        # Ring buffer state
        self.write_index = 0
        self.read_index = 0
        self.count = 0
        self.lock = threading.RLock()

        # Event signaling for space availability
        self.space_available = threading.Condition(self.lock)
        self.data_available = threading.Condition(self.lock)

        # Statistics
        self.frames_written = 0
        self.frames_read = 0
        self.overflow_count = 0
        self.underflow_count = 0

        logger.info(f"Initialized LED buffer with {buffer_size} frames " f"({self._get_memory_usage_mb():.1f}MB)")

    def _get_memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        led_bytes = self.led_arrays.nbytes
        timestamp_bytes = self.timestamps.nbytes
        return (led_bytes + timestamp_bytes) / (1024 * 1024)

    def write_led_values(
        self,
        led_values: np.ndarray,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None,
        block: bool = False,
        timeout: float = 1.0,
    ) -> bool:
        """
        Write LED values to buffer with timestamp.

        Args:
            led_values: LED RGB values, shape (led_count, 3) or (3, led_count)
            timestamp: Frame timestamp
            metadata: Optional metadata dictionary
            block: If True, wait for space instead of dropping frames
            timeout: Maximum time to wait for space when blocking

        Returns:
            True if written successfully, False if buffer overflow or timeout
        """
        if block:
            # Blocking mode: wait for space using condition variable
            with self.space_available:
                while self.count >= self.buffer_size:
                    if not self.space_available.wait(timeout=timeout):
                        logger.warning(f"LED buffer write timeout after {timeout:.1f}s")
                        return False

        with self.lock:
            # Handle buffer overflow (non-blocking mode only)
            if self.count >= self.buffer_size:
                if not block:  # Only drop in non-blocking mode
                    logger.warning("LED buffer overflow - dropping oldest frame")
                    self.overflow_count += 1
                    # Advance read index to make space
                    self.read_index = (self.read_index + 1) % self.buffer_size
                    self.count -= 1
                else:
                    # This shouldn't happen in blocking mode, but safety check
                    logger.error("LED buffer still full after blocking wait")
                    return False

            # Validate and convert LED values format
            if led_values.shape == (3, self.led_count):
                # Convert (3, led_count) to (led_count, 3)
                led_values = led_values.T
            elif led_values.shape != (self.led_count, 3):
                logger.error(
                    f"Invalid LED values shape: {led_values.shape}, "
                    f"expected ({self.led_count}, 3) or (3, {self.led_count})"
                )
                return False

            # Store LED values, timestamp, and metadata
            self.led_arrays[self.write_index] = led_values.astype(np.uint8)
            self.timestamps[self.write_index] = timestamp
            self.metadata[self.write_index] = metadata or {}

            # Advance write index
            self.write_index = (self.write_index + 1) % self.buffer_size
            self.count += 1
            self.frames_written += 1

            # Signal that data is now available
            self.data_available.notify()

            return True

    def read_led_values(self, timeout: Optional[float] = None) -> Optional[Tuple[np.ndarray, float, Dict[str, Any]]]:
        """
        Read LED values from buffer with timestamp.

        Args:
            timeout: Maximum time to wait for data (None = no timeout)

        Returns:
            Tuple of (led_values, timestamp, metadata) or None if timeout/empty
        """
        start_time = time.time() if timeout is not None else None

        with self.data_available:
            # Wait for data using condition variable
            while self.count == 0:
                if timeout is None:
                    # No timeout, wait indefinitely
                    self.data_available.wait()
                else:
                    # Wait with timeout
                    if not self.data_available.wait(timeout=timeout):
                        # Timeout reached
                        with self.lock:
                            self.underflow_count += 1
                        return None

            # Data available, read it
            led_values = self.led_arrays[self.read_index].copy()
            timestamp = self.timestamps[self.read_index]
            metadata = self.metadata[self.read_index] or {}

            # Advance read index
            self.read_index = (self.read_index + 1) % self.buffer_size
            self.count -= 1
            self.frames_read += 1

            # Signal that space is now available
            self.space_available.notify()

            return led_values, timestamp, metadata

    def read_latest_led_values(self) -> Optional[Tuple[np.ndarray, float, Dict[str, Any]]]:
        """
        Read the latest LED values from buffer without blocking.

        Returns:
            Tuple of (led_values, timestamp, metadata) or None if buffer empty
        """
        with self.lock:
            if self.count == 0:
                return None

            # Get the most recent frame (latest written)
            latest_index = (self.write_index - 1) % self.buffer_size
            led_values = self.led_arrays[latest_index].copy()
            timestamp = self.timestamps[latest_index]
            metadata = self.metadata[latest_index] or {}

            return led_values, timestamp, metadata

    def peek_next_timestamp(self) -> Optional[float]:
        """
        Peek at the timestamp of the next frame without consuming it.

        Returns:
            Next frame timestamp or None if buffer empty
        """
        with self.lock:
            if self.count > 0:
                return self.timestamps[self.read_index]
            return None

    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        with self.lock:
            utilization = self.count / self.buffer_size if self.buffer_size > 0 else 0.0

            return {
                "buffer_size": self.buffer_size,
                "current_count": self.count,
                "utilization": utilization,
                "write_index": self.write_index,
                "read_index": self.read_index,
                "frames_written": self.frames_written,
                "frames_read": self.frames_read,
                "overflow_count": self.overflow_count,
                "underflow_count": self.underflow_count,
                "memory_usage_mb": self._get_memory_usage_mb(),
                "is_full": self.count >= self.buffer_size,
                "is_empty": self.count == 0,
            }

    def clear(self) -> None:
        """Clear all data from buffer."""
        with self.lock:
            self.write_index = 0
            self.read_index = 0
            self.count = 0
            self.metadata = [None] * self.buffer_size
            logger.debug("LED buffer cleared")

    def set_buffer_size(self, new_size: int) -> bool:
        """
        Resize buffer (data will be lost).

        Args:
            new_size: New buffer size

        Returns:
            True if resize successful, False otherwise
        """
        if new_size <= 0:
            logger.error(f"Invalid buffer size: {new_size}")
            return False

        with self.lock:
            try:
                # Allocate new arrays
                new_led_arrays = np.zeros((new_size, self.led_count, 3), dtype=np.uint8)
                new_timestamps = np.zeros(new_size, dtype=np.float64)
                new_metadata = [None] * new_size

                # Replace arrays
                self.led_arrays = new_led_arrays
                self.timestamps = new_timestamps
                self.metadata = new_metadata
                self.buffer_size = new_size

                # Reset indices
                self.write_index = 0
                self.read_index = 0
                self.count = 0

                logger.info(f"LED buffer resized to {new_size} frames " f"({self._get_memory_usage_mb():.1f}MB)")
                return True

            except Exception as e:
                logger.error(f"Failed to resize LED buffer: {e}")
                return False

    def __len__(self) -> int:
        """Get current number of frames in buffer."""
        with self.lock:
            return self.count

    def __bool__(self) -> bool:
        """Check if buffer has data."""
        with self.lock:
            return self.count > 0
