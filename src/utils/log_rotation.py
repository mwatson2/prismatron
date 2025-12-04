#!/usr/bin/env python3
"""
Log Rotation Utility for Prismatron LED Display System.

This module provides simple log file rotation to prevent log files from
filling up disk space. It implements a simple rotation scheme:
- When prismatron.log > LOG_MAX_SIZE_MB, rotate it to prismatron.1.log
- Delete any existing prismatron.1.log before rotation
- Start with a fresh empty prismatron.log

This keeps a maximum of 2 * LOG_MAX_SIZE_MB of log data.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from const import LOG_MAX_SIZE_MB

logger = logging.getLogger(__name__)


class LogRotator:
    """
    Simple log file rotator that runs in a background thread.

    Monitors the main log file and rotates it when it exceeds the size limit.
    Keeps one backup copy (prismatron.1.log) for a total of ~200MB max log storage.
    """

    def __init__(self, log_file_path: str = "logs/prismatron.log", check_interval: int = 300):
        """
        Initialize log rotator.

        Args:
            log_file_path: Path to the main log file to monitor
            check_interval: How often to check file size (seconds, default: 5 minutes)
        """
        self.log_file_path = Path(log_file_path)
        self.backup_log_path = self.log_file_path.with_name(f"{self.log_file_path.stem}.1{self.log_file_path.suffix}")
        self.check_interval = check_interval
        self.max_size_bytes = LOG_MAX_SIZE_MB * 1024 * 1024  # Convert MB to bytes

        self._stop_event = threading.Event()
        self._rotation_thread: Optional[threading.Thread] = None
        self._is_running = False

    def get_file_size_mb(self, file_path: Path) -> float:
        """
        Get file size in MB.

        Args:
            file_path: Path to file

        Returns:
            File size in MB, or 0 if file doesn't exist
        """
        try:
            if file_path.exists():
                size_bytes = file_path.stat().st_size
                return size_bytes / (1024 * 1024)
            return 0.0
        except OSError as e:
            logger.warning(f"Failed to get size of {file_path}: {e}")
            return 0.0

    def rotate_log_file(self) -> bool:
        """
        Rotate the log file.

        1. Check if main log file exceeds size limit
        2. If yes, remove existing backup (if any)
        3. Rename main log to backup
        4. Let logging system create new main log file

        Returns:
            True if rotation was performed, False otherwise
        """
        try:
            # Check if rotation is needed
            current_size_mb = self.get_file_size_mb(self.log_file_path)

            if current_size_mb < LOG_MAX_SIZE_MB:
                return False

            logger.info(f"Log file {self.log_file_path} is {current_size_mb:.1f}MB, rotating...")

            # Remove existing backup if it exists
            if self.backup_log_path.exists():
                try:
                    self.backup_log_path.unlink()
                    logger.debug(f"Removed existing backup log: {self.backup_log_path}")
                except OSError as e:
                    logger.warning(f"Failed to remove backup log {self.backup_log_path}: {e}")
                    # Continue anyway - rename might still work

            # Rename current log to backup
            if self.log_file_path.exists():
                try:
                    # First, we need to close and remove the existing FileHandler
                    # to release the file handle before renaming
                    root_logger = logging.getLogger()

                    # Find and remove the FileHandler that's writing to prismatron.log
                    file_handlers_to_remove = []
                    for handler in root_logger.handlers:
                        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(
                            self.log_file_path.absolute()
                        ):
                            file_handlers_to_remove.append(handler)

                    # Close and remove the old handlers
                    for handler in file_handlers_to_remove:
                        handler.close()
                        root_logger.removeHandler(handler)

                    # Now rename the file
                    self.log_file_path.rename(self.backup_log_path)
                    logger.info(f"Rotated {self.log_file_path} -> {self.backup_log_path}")

                    # Create a new FileHandler for the fresh log file
                    if file_handlers_to_remove:
                        # Use the same formatter from the old handler
                        old_handler = file_handlers_to_remove[0]
                        new_handler = logging.FileHandler(str(self.log_file_path), mode="a")
                        new_handler.setFormatter(old_handler.formatter)
                        new_handler.setLevel(old_handler.level)
                        root_logger.addHandler(new_handler)
                        logger.info(f"Created new FileHandler for {self.log_file_path}")

                    # Log the new state
                    backup_size_mb = self.get_file_size_mb(self.backup_log_path)
                    logger.info(f"Log rotation complete. Backup size: {backup_size_mb:.1f}MB, starting fresh log file")

                    return True

                except OSError as e:
                    logger.error(f"Failed to rotate log file {self.log_file_path}: {e}")
                    return False
            else:
                logger.debug(f"Log file {self.log_file_path} does not exist, no rotation needed")
                return False

        except Exception as e:
            logger.error(f"Unexpected error during log rotation: {e}")
            return False

    def _rotation_worker(self):
        """Background thread worker that periodically checks and rotates logs."""
        logger.info(
            f"Log rotation worker started, checking every {self.check_interval}s for files > {LOG_MAX_SIZE_MB}MB"
        )

        while not self._stop_event.is_set():
            try:
                # Perform rotation check
                rotated = self.rotate_log_file()
                if rotated:
                    # Log some stats about current log storage
                    main_size = self.get_file_size_mb(self.log_file_path)
                    backup_size = self.get_file_size_mb(self.backup_log_path)
                    total_size = main_size + backup_size
                    logger.info(
                        f"Log storage: main={main_size:.1f}MB, backup={backup_size:.1f}MB, total={total_size:.1f}MB"
                    )

            except Exception as e:
                logger.error(f"Error in log rotation worker: {e}")

            # Wait for next check or stop signal
            self._stop_event.wait(self.check_interval)

    def start(self) -> bool:
        """
        Start the background log rotation worker thread.

        Returns:
            True if started successfully, False if already running
        """
        if self._is_running:
            logger.warning("Log rotator is already running")
            return False

        try:
            self._stop_event.clear()
            self._rotation_thread = threading.Thread(
                target=self._rotation_worker, name="LogRotator", daemon=True  # Don't prevent process exit
            )
            self._rotation_thread.start()
            self._is_running = True

            logger.info(f"Log rotator started: monitoring {self.log_file_path} (max {LOG_MAX_SIZE_MB}MB)")
            return True

        except Exception as e:
            logger.error(f"Failed to start log rotator: {e}")
            return False

    def stop(self, timeout: float = 5.0) -> bool:
        """
        Stop the background log rotation worker thread.

        Args:
            timeout: Maximum time to wait for thread to stop (seconds)

        Returns:
            True if stopped successfully, False on timeout
        """
        if not self._is_running:
            return True

        try:
            self._stop_event.set()

            if self._rotation_thread and self._rotation_thread.is_alive():
                self._rotation_thread.join(timeout)

                if self._rotation_thread.is_alive():
                    logger.warning(f"Log rotator thread did not stop within {timeout}s")
                    return False

            self._is_running = False
            logger.info("Log rotator stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping log rotator: {e}")
            return False

    def is_running(self) -> bool:
        """Check if the log rotator is currently running."""
        return self._is_running

    def force_rotation(self) -> bool:
        """
        Force an immediate log rotation regardless of file size.

        Returns:
            True if rotation was performed successfully
        """
        logger.info("Forcing log rotation...")

        # Temporarily bypass size check
        original_max_size = LOG_MAX_SIZE_MB

        # Set a very small threshold to force rotation
        import const

        const.LOG_MAX_SIZE_MB = 0

        try:
            result = self.rotate_log_file()
        finally:
            # Restore original threshold
            const.LOG_MAX_SIZE_MB = original_max_size

        return result

    def get_log_stats(self) -> dict:
        """
        Get current log file statistics.

        Returns:
            Dictionary with log file information
        """
        main_size_mb = self.get_file_size_mb(self.log_file_path)
        backup_size_mb = self.get_file_size_mb(self.backup_log_path)

        return {
            "main_log_path": str(self.log_file_path),
            "main_log_size_mb": main_size_mb,
            "backup_log_path": str(self.backup_log_path),
            "backup_log_size_mb": backup_size_mb,
            "total_size_mb": main_size_mb + backup_size_mb,
            "max_size_mb": LOG_MAX_SIZE_MB,
            "rotation_needed": main_size_mb >= LOG_MAX_SIZE_MB,
            "is_running": self._is_running,
            "check_interval_seconds": self.check_interval,
        }


# Global log rotator instance
_global_rotator: Optional[LogRotator] = None


def get_log_rotator(log_file_path: str = "logs/prismatron.log", check_interval: int = 300) -> LogRotator:
    """
    Get the global log rotator instance (singleton pattern).

    Args:
        log_file_path: Path to the main log file (only used on first call)
        check_interval: Check interval in seconds (only used on first call)

    Returns:
        LogRotator instance
    """
    global _global_rotator

    if _global_rotator is None:
        _global_rotator = LogRotator(log_file_path, check_interval)

    return _global_rotator


def start_log_rotation(log_file_path: str = "logs/prismatron.log", check_interval: int = 300) -> bool:
    """
    Start global log rotation.

    Args:
        log_file_path: Path to the main log file to monitor
        check_interval: How often to check file size (seconds, default: 5 minutes)

    Returns:
        True if started successfully
    """
    rotator = get_log_rotator(log_file_path, check_interval)
    return rotator.start()


def stop_log_rotation(timeout: float = 5.0) -> bool:
    """
    Stop global log rotation.

    Args:
        timeout: Maximum time to wait for stop (seconds)

    Returns:
        True if stopped successfully
    """
    global _global_rotator

    if _global_rotator is not None:
        return _global_rotator.stop(timeout)

    return True


def get_log_stats() -> dict:
    """
    Get current log statistics from global rotator.

    Returns:
        Dictionary with log file information
    """
    global _global_rotator

    if _global_rotator is not None:
        return _global_rotator.get_log_stats()

    return {"error": "Log rotator not initialized"}
