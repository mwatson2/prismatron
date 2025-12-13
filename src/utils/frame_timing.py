"""
Frame timing data tracking and logging utilities.

This module provides classes and utilities for tracking detailed timing information
through the frame processing pipeline, from producer plugin to final render.
"""

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TextIO

logger = logging.getLogger(__name__)


@dataclass
class FrameTimingData:
    """
    Tracks detailed timing information for a single frame through the processing pipeline.

    All wallclock times are in seconds since epoch (time.time()).
    Plugin timestamp is 0-based from content start.
    Producer timestamp is global presentation timestamp.
    """

    # Frame identification
    frame_index: int = 0  # Global frame counter
    plugin_timestamp: float = 0.0  # 0-based timestamp from content plugin
    producer_timestamp: float = 0.0  # Global presentation timestamp
    item_duration: float = 0.0  # Duration of current playlist item

    # Pipeline timing (wallclock times)
    write_to_buffer_time: Optional[float] = None  # When frame written to shared buffer
    read_from_buffer_time: Optional[float] = None  # When frame read from shared buffer
    write_to_led_buffer_time: Optional[float] = None  # When LED data written to LED buffer
    read_from_led_buffer_time: Optional[float] = None  # When LED data read from LED buffer
    led_transition_time: Optional[float] = None  # When LED transitions completed
    render_time: Optional[float] = None  # When frame actually rendered/sent to hardware

    def mark_write_to_buffer(self) -> None:
        """Mark when frame is written to shared buffer."""
        self.write_to_buffer_time = time.time()

    def mark_read_from_buffer(self) -> None:
        """Mark when frame is read from shared buffer."""
        self.read_from_buffer_time = time.time()

    def mark_write_to_led_buffer(self) -> None:
        """Mark when LED data is written to LED buffer."""
        self.write_to_led_buffer_time = time.time()

    def mark_read_from_led_buffer(self) -> None:
        """Mark when LED data is read from LED buffer."""
        self.read_from_led_buffer_time = time.time()

    def mark_led_transition_complete(self) -> None:
        """Mark when LED transitions are completed."""
        self.led_transition_time = time.time()

    def mark_render(self) -> None:
        """Mark when frame is actually rendered/sent to hardware."""
        self.render_time = time.time()

    def to_csv_row(self) -> list:
        """
        Convert timing data to CSV row format.

        Returns:
            List of values: [frame_index, plugin_timestamp, producer_timestamp,
                           write_to_buffer_time, read_from_buffer_time,
                           write_to_led_buffer_time, read_from_led_buffer_time,
                           led_transition_time, render_time, item_duration]
        """
        return [
            self.frame_index,
            self.plugin_timestamp,
            self.producer_timestamp,
            self.write_to_buffer_time or 0.0,
            self.read_from_buffer_time or 0.0,
            self.write_to_led_buffer_time or 0.0,
            self.read_from_led_buffer_time or 0.0,
            self.led_transition_time or 0.0,
            self.render_time or 0.0,
            self.item_duration,
        ]

    @classmethod
    def csv_header(cls) -> list:
        """
        Get CSV header row for timing data.

        Returns:
            List of column names
        """
        return [
            "frame_index",
            "plugin_timestamp",
            "producer_timestamp",
            "write_to_buffer_time",
            "read_from_buffer_time",
            "write_to_led_buffer_time",
            "read_from_led_buffer_time",
            "led_transition_time",
            "render_time",
            "item_duration",
        ]


class FrameTimingLogger:
    """
    Logs frame timing data to CSV file for analysis and visualization.
    """

    def __init__(self, log_file_path: str):
        """
        Initialize timing logger.

        Args:
            log_file_path: Path to CSV file for logging timing data
        """
        self.log_file_path = Path(log_file_path)
        self._file_handle: Optional[TextIO] = None
        self._csv_writer: Optional[Any] = None  # csv.writer object
        self._header_written = False

        # Ensure parent directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def start_logging(self) -> bool:
        """
        Start logging to file.

        Returns:
            True if logging started successfully, False otherwise
        """
        try:
            # Initialize file with header
            with open(self.log_file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(FrameTimingData.csv_header())
                f.flush()

            # Open for appending data
            self._file_handle = open(self.log_file_path, "a", newline="")  # noqa: SIM115
            self._csv_writer = csv.writer(self._file_handle)
            self._header_written = True

            logger.info(f"Started frame timing logging to {self.log_file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to start timing logger: {e}")
            self.stop_logging()
            return False

    def log_frame(self, timing_data: FrameTimingData) -> None:
        """
        Log a single frame's timing data.

        Args:
            timing_data: Frame timing data to log
        """
        if not self._csv_writer or not self._header_written:
            logger.warning("Timing logger not properly initialized, skipping frame")
            return

        try:
            self._csv_writer.writerow(timing_data.to_csv_row())
            if self._file_handle:
                self._file_handle.flush()  # Ensure data is written immediately

        except Exception as e:
            logger.error(f"Failed to log frame timing: {e}")

    def stop_logging(self) -> None:
        """Stop logging and close file."""
        if self._file_handle:
            try:
                self._file_handle.close()
                logger.info(f"Stopped frame timing logging to {self.log_file_path}")
            except Exception as e:
                logger.error(f"Error closing timing log file: {e}")
            finally:
                self._file_handle = None
                self._csv_writer = None
                self._header_written = False

    def __enter__(self):
        """Context manager entry."""
        self.start_logging()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_logging()


# Global frame counter for consistent frame indexing across processes
_global_frame_counter = 0


def get_next_frame_index() -> int:
    """
    Get the next global frame index.

    Returns:
        Next frame index (thread-safe)
    """
    global _global_frame_counter
    _global_frame_counter += 1
    return _global_frame_counter


def reset_frame_counter() -> None:
    """Reset global frame counter to 0."""
    global _global_frame_counter
    _global_frame_counter = 0
