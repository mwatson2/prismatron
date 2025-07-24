"""
Logging utilities with custom formatters for time tracking.

This module provides utilities for logging with time-from-app-start tracking.
"""

import logging
import time
from typing import Optional


class AppTimeFormatter(logging.Formatter):
    """Custom formatter that includes time from application start."""

    def __init__(
        self, fmt: Optional[str] = None, datefmt: Optional[str] = None, app_start_time: Optional[float] = None
    ):
        """
        Initialize the formatter.

        Args:
            fmt: Log format string
            datefmt: Date format string
            app_start_time: Application start time (time.time()). If None, uses current time.
        """
        super().__init__(fmt, datefmt)
        self.app_start_time = app_start_time or time.time()

    def format(self, record):
        """Format the log record with app start time."""
        # Calculate elapsed time from app start
        elapsed_seconds = record.created - self.app_start_time

        # Convert to mm:ss.xxx format
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds % 60
        app_time_str = f"{minutes:02d}:{seconds:06.3f}"

        # Add app_time to the record
        record.app_time = app_time_str

        return super().format(record)


# Global app start time - set when logging is first configured
_app_start_time: Optional[float] = None


def get_app_start_time() -> float:
    """Get the application start time."""
    global _app_start_time
    if _app_start_time is None:
        _app_start_time = time.time()
    return _app_start_time


def set_app_start_time(start_time: float) -> None:
    """Set the application start time."""
    global _app_start_time
    _app_start_time = start_time


def create_app_time_formatter(fmt: Optional[str] = None, datefmt: Optional[str] = None) -> AppTimeFormatter:
    """
    Create a formatter with app start time tracking.

    Args:
        fmt: Log format string. If None, uses default with app_time field.
        datefmt: Date format string

    Returns:
        AppTimeFormatter instance
    """
    if fmt is None:
        fmt = "%(asctime)s - %(app_time)s - %(name)s - %(levelname)s - %(message)s"

    return AppTimeFormatter(fmt=fmt, datefmt=datefmt, app_start_time=get_app_start_time())
