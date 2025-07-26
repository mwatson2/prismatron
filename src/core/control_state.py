"""
Control State Manager for lightweight IPC between processes.

This module provides a shared state system for coordinating between
producer, consumer, and web server processes. Handles play/pause control,
configuration settings, and system status monitoring.
"""

import json
import logging
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

# Check for shared_memory availability (Python 3.8+)
try:
    from multiprocessing import shared_memory

    SHARED_MEMORY_AVAILABLE = True
except ImportError:
    # Fallback for older Python versions
    SHARED_MEMORY_AVAILABLE = False
    shared_memory = None  # type: ignore


logger = logging.getLogger(__name__)


class PlayState(Enum):
    """Playback state enumeration."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    ERROR = "error"


class SystemState(Enum):
    """System state enumeration."""

    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    SHUTTING_DOWN = "shutting_down"
    RESTARTING = "restarting"
    REBOOTING = "rebooting"
    ERROR = "error"


@dataclass
class SystemStatus:
    """System status information."""

    play_state: PlayState = PlayState.STOPPED
    system_state: SystemState = SystemState.INITIALIZING
    current_file: str = ""
    brightness: float = 1.0
    frame_rate: float = 0.0
    producer_fps: float = 0.0
    consumer_fps: float = 0.0
    error_message: str = ""
    uptime: float = 0.0
    last_update: float = 0.0

    # New consumer statistics for multi-process IPC
    consumer_input_fps: float = 0.0
    renderer_output_fps: float = 0.0
    dropped_frames_percentage: float = 0.0
    late_frame_percentage: float = 0.0

    # Rendering index to track which item is actually being rendered
    rendering_index: int = -1


class ControlState:
    """
    Lightweight IPC system for process coordination and configuration.

    Provides shared state management for system control, configuration,
    and status monitoring across multiple processes.
    """

    def __init__(self, name: str = "prismatron_control"):
        """
        Initialize the control state manager.

        Args:
            name: Unique name for the shared memory segment
        """
        self.name = name
        self._shared_memory: Optional[Any] = None
        self._control_dict = None
        self._lock = mp.Lock()
        self._initialized = False
        self._start_time = time.time()
        self._is_creator = False  # Track if this instance created the shared memory

        # Events for coordination
        self._shutdown_event = mp.Event()
        self._restart_event = mp.Event()
        self._reboot_event = mp.Event()
        self._config_updated_event = mp.Event()
        self._status_updated_event = mp.Event()

        # Default status
        self._default_status = SystemStatus()

        # Calculate buffer size based on typical status object size
        self._buffer_size = self._calculate_buffer_size()

    def _calculate_buffer_size(self) -> int:
        """
        Calculate appropriate buffer size for status objects.

        We estimate the size by serializing a test status object with reasonable
        content lengths, then add a safety margin.

        Returns:
            Buffer size in bytes
        """
        try:
            # Create test status with reasonable content sizes
            test_status = SystemStatus(
                current_file="/very/long/path/to/some/content/file/with/long/name.mp4",
                error_message="This is a reasonably long error message that might occur during operation",
                uptime=999999.99,
                last_update=time.time(),
            )

            # Serialize to estimate size
            status_dict = asdict(test_status)
            status_dict["play_state"] = test_status.play_state.value
            status_dict["system_state"] = test_status.system_state.value

            json_data = json.dumps(status_dict, separators=(",", ":"))
            estimated_size = len(json_data.encode("utf-8"))

            # Add 50% safety margin and align to 1KB boundary
            safe_size = int(estimated_size * 1.5)
            aligned_size = ((safe_size + 1023) // 1024) * 1024  # Round up to 1KB

            return max(aligned_size, 2048)  # Minimum 2KB

        except Exception as e:
            logger.warning(f"Failed to calculate buffer size: {e}")
            return 4096  # Fallback to 4KB

    def initialize(self) -> bool:
        """
        Initialize shared memory for control state.

        Returns:
            True if initialization successful, False otherwise
        """
        if not SHARED_MEMORY_AVAILABLE:
            logger.error("Shared memory not available (requires Python 3.8+)")
            return False

        try:
            # Create shared memory for control data
            self._shared_memory = shared_memory.SharedMemory(name=self.name, create=True, size=self._buffer_size)

            # Initialize with default status
            self._write_status(self._default_status)

            self._initialized = True
            self._is_creator = True  # Mark this instance as the creator
            logger.info(f"Initialized control state '{self.name}' with {self._buffer_size} bytes")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize control state: {e}")
            self.cleanup()
            return False

    def connect(self) -> bool:
        """
        Connect to existing control state (for other processes).

        Returns:
            True if connection successful, False otherwise
        """
        if not SHARED_MEMORY_AVAILABLE:
            logger.error("Shared memory not available (requires Python 3.8+)")
            return False

        try:
            # Connect to existing shared memory
            self._shared_memory = shared_memory.SharedMemory(name=self.name)
            self._initialized = True

            logger.info(f"Connected to control state '{self.name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to control state: {e}")
            return False

    def _write_status(self, status: SystemStatus) -> bool:
        """
        Write status to shared memory using JSON serialization.

        The status is serialized to JSON and stored in the shared memory buffer.
        We include metadata timestamps and convert enums to strings for JSON compatibility.

        Args:
            status: SystemStatus object to write

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False

        try:
            with self._lock:
                # Update timestamps before serialization
                status.last_update = time.time()
                status.uptime = time.time() - self._start_time

                # Convert to dictionary for JSON serialization
                status_dict = asdict(status)
                # Convert enums to strings (JSON doesn't support enum objects)
                status_dict["play_state"] = status.play_state.value
                status_dict["system_state"] = status.system_state.value

                # Serialize to JSON with compact formatting
                json_data = json.dumps(status_dict, separators=(",", ":"))
                json_bytes = json_data.encode("utf-8")

                # Check if data fits in buffer
                if self._shared_memory is not None and len(json_bytes) >= self._shared_memory.size:
                    buffer_size = self._shared_memory.size if self._shared_memory else 0
                    logger.error(f"Status data too large: {len(json_bytes)} bytes, buffer size: {buffer_size} bytes")
                    return False

                # Clear entire buffer first (prevents old data remnants)
                if self._shared_memory is not None:
                    self._shared_memory.buf[:] = b"\x00" * self._shared_memory.size

                    # Write new data and null terminator
                    self._shared_memory.buf[: len(json_bytes)] = json_bytes
                    self._shared_memory.buf[len(json_bytes)] = 0  # Null terminator

                return True

        except Exception as e:
            logger.error(f"Failed to write status: {e}")
            return False

    def _read_status(self) -> Optional[SystemStatus]:
        """
        Read status from shared memory and deserialize from JSON.

        Finds the null-terminated JSON string in the buffer, deserializes it,
        and converts string enums back to enum objects.

        Returns:
            SystemStatus object or default status if error/empty
        """
        if not self._initialized:
            return None

        try:
            with self._lock:
                # Read buffer and find the null terminator
                if self._shared_memory is None:
                    return self._default_status
                buffer_data = bytes(self._shared_memory.buf)
                null_pos = buffer_data.find(b"\x00")

                if null_pos == -1:
                    # No null terminator found, might be corrupted
                    logger.warning("No null terminator found in status buffer")
                    return self._default_status

                if null_pos == 0:
                    # Empty data (just null terminator at start)
                    return self._default_status

                # Extract JSON data up to null terminator
                json_bytes = buffer_data[:null_pos]

                # Decode and parse JSON
                json_data = json_bytes.decode("utf-8")
                status_dict = json.loads(json_data)

                # Convert string enums back to enum objects
                # This is needed because JSON serialization converts enums to strings
                status_dict["play_state"] = PlayState(status_dict["play_state"])
                status_dict["system_state"] = SystemState(status_dict["system_state"])

                return SystemStatus(**status_dict)

        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            logger.warning(f"Failed to decode status JSON: {e}")
            return self._default_status
        except Exception as e:
            logger.error(f"Failed to read status: {e}")
            return self._default_status

    def set_play_state(self, state: PlayState) -> bool:
        """
        Set the current playback state.

        Args:
            state: New playback state

        Returns:
            True if successful, False otherwise
        """
        try:
            current_status = self._read_status()
            if current_status:
                current_status.play_state = state
                result = self._write_status(current_status)
                if result:
                    self._status_updated_event.set()
                return result
            return False

        except Exception as e:
            logger.error(f"Failed to set play state: {e}")
            return False

    def set_current_file(self, filepath: str) -> bool:
        """
        Set the current content file path.

        Args:
            filepath: Path to current content file

        Returns:
            True if successful, False otherwise
        """
        try:
            current_status = self._read_status()
            if current_status:
                current_status.current_file = filepath
                result = self._write_status(current_status)
                if result:
                    self._config_updated_event.set()
                return result
            return False

        except Exception as e:
            logger.error(f"Failed to set current file: {e}")
            return False

    def set_brightness(self, value: float) -> bool:
        """
        Set the system brightness level.

        Args:
            value: Brightness value (0.0 to 1.0)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clamp brightness to valid range
            value = max(0.0, min(1.0, value))

            current_status = self._read_status()
            if current_status:
                current_status.brightness = value
                result = self._write_status(current_status)
                if result:
                    self._config_updated_event.set()
                return result
            return False

        except Exception as e:
            logger.error(f"Failed to set brightness: {e}")
            return False

    def set_frame_rates(self, producer_fps: float, consumer_fps: float) -> bool:
        """
        Update frame rate monitoring information.

        Args:
            producer_fps: Producer process frame rate
            consumer_fps: Consumer process frame rate

        Returns:
            True if successful, False otherwise
        """
        try:
            current_status = self._read_status()
            if current_status:
                current_status.producer_fps = producer_fps
                current_status.consumer_fps = consumer_fps
                current_status.frame_rate = min(producer_fps, consumer_fps)
                result = self._write_status(current_status)
                if result:
                    self._status_updated_event.set()
                return result
            return False

        except Exception as e:
            logger.error(f"Failed to set frame rates: {e}")
            return False

    def set_error(self, error_message: str) -> bool:
        """
        Set system error state and message.

        Args:
            error_message: Error description

        Returns:
            True if successful, False otherwise
        """
        try:
            current_status = self._read_status()
            if current_status:
                current_status.system_state = SystemState.ERROR
                current_status.error_message = error_message
                result = self._write_status(current_status)
                if result:
                    self._status_updated_event.set()
                return result
            return False

        except Exception as e:
            logger.error(f"Failed to set error: {e}")
            return False

    def clear_error(self) -> bool:
        """
        Clear error state and return to running.

        Returns:
            True if successful, False otherwise
        """
        try:
            current_status = self._read_status()
            if current_status:
                current_status.system_state = SystemState.RUNNING
                current_status.error_message = ""
                result = self._write_status(current_status)
                if result:
                    self._status_updated_event.set()
                return result
            return False

        except Exception as e:
            logger.error(f"Failed to clear error: {e}")
            return False

    def signal_shutdown(self) -> None:
        """Signal all processes to shutdown gracefully."""
        try:
            current_status = self._read_status()
            if current_status:
                current_status.system_state = SystemState.SHUTTING_DOWN
                self._write_status(current_status)

            self._shutdown_event.set()
            logger.info("Shutdown signal sent")

        except Exception as e:
            logger.error(f"Failed to signal shutdown: {e}")

    def signal_restart(self) -> None:
        """Signal system restart."""
        try:
            current_status = self._read_status()
            if current_status:
                current_status.system_state = SystemState.RESTARTING
                self._write_status(current_status)

            self._restart_event.set()
            logger.info("Restart signal sent")

        except Exception as e:
            logger.error(f"Failed to signal restart: {e}")

    def signal_reboot(self) -> None:
        """Signal device reboot."""
        try:
            current_status = self._read_status()
            if current_status:
                current_status.system_state = SystemState.REBOOTING
                self._write_status(current_status)

            self._reboot_event.set()
            logger.info("Reboot signal sent")

        except Exception as e:
            logger.error(f"Failed to signal reboot: {e}")

    def is_shutdown_requested(self) -> bool:
        """
        Check if shutdown has been requested.

        Returns:
            True if shutdown requested, False otherwise
        """
        return self._shutdown_event.is_set()

    def is_restart_requested(self) -> bool:
        """
        Check if restart has been requested.

        Returns:
            True if restart requested, False otherwise
        """
        return self._restart_event.is_set()

    def is_reboot_requested(self) -> bool:
        """
        Check if reboot has been requested.

        Returns:
            True if reboot requested, False otherwise
        """
        return self._reboot_event.is_set()

    def should_shutdown(self) -> bool:
        """
        Check if any shutdown condition exists (shutdown, restart, or reboot).

        Returns:
            True if any shutdown condition is active, False otherwise
        """
        return self._shutdown_event.is_set() or self._restart_event.is_set() or self._reboot_event.is_set()

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for shutdown signal.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if shutdown signaled, False if timeout
        """
        return self._shutdown_event.wait(timeout)

    def wait_for_config_update(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for configuration update.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if config updated, False if timeout
        """
        result = self._config_updated_event.wait(timeout)
        if result:
            self._config_updated_event.clear()
        return result

    def wait_for_status_update(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for status update.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if status updated, False if timeout
        """
        result = self._status_updated_event.wait(timeout)
        if result:
            self._status_updated_event.clear()
        return result

    def get_status(self) -> Optional[SystemStatus]:
        """
        Get current system status.

        Returns:
            SystemStatus object or None if error
        """
        return self._read_status()

    def get_status_dict(self) -> Dict[str, Any]:
        """
        Get current system status as dictionary.

        Returns:
            Dictionary with status information
        """
        status = self._read_status()
        if status:
            status_dict = asdict(status)
            status_dict["play_state"] = status.play_state.value
            status_dict["system_state"] = status.system_state.value
            return status_dict
        else:
            return {"error": "Failed to read status", "initialized": self._initialized}

    def update_system_state(self, state: SystemState) -> bool:
        """
        Update the system state.

        Args:
            state: New system state

        Returns:
            True if successful, False otherwise
        """
        try:
            current_status = self._read_status()
            if current_status:
                current_status.system_state = state
                result = self._write_status(current_status)
                if result:
                    self._status_updated_event.set()
                return result
            return False

        except Exception as e:
            logger.error(f"Failed to update system state: {e}")
            return False

    def update_status(self, **updates) -> bool:
        """
        Update specific fields in the system status.

        This method allows updating individual status fields without overwriting
        the entire status object. It uses the same read-modify-write pattern as
        other methods, so it's subject to the same race condition limitations.

        Args:
            **updates: Keyword arguments for status fields to update
                      Valid fields: consumer_input_fps, renderer_output_fps,
                      dropped_frames_percentage, late_frame_percentage, etc.

        Returns:
            True if successful, False otherwise
        """
        try:
            current_status = self._read_status()
            if not current_status:
                return False

            # Update only the provided fields
            for field_name, value in updates.items():
                if hasattr(current_status, field_name):
                    setattr(current_status, field_name, value)
                else:
                    logger.warning(f"Unknown status field: {field_name}")

            result = self._write_status(current_status)
            if result:
                self._status_updated_event.set()
            return result

        except Exception as e:
            logger.error(f"Failed to update status fields: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up shared memory resources."""
        try:
            if self._shared_memory:
                try:
                    self._shared_memory.close()
                    # Only unlink if this instance created the shared memory
                    if self._is_creator:
                        self._shared_memory.unlink()
                        logger.info(f"Unlinked shared memory '{self.name}' (creator cleanup)")
                    else:
                        logger.debug(f"Closed connection to shared memory '{self.name}' (non-creator cleanup)")
                except Exception as e:
                    # Only log if it's not a "file not found" error (normal during shutdown)
                    if hasattr(e, "errno") and e.errno != 2:  # ENOENT - No such file or directory
                        logger.warning(f"Error cleaning up control state memory: {e}")

            self._initialized = False
            logger.info(f"Cleaned up control state '{self.name}'")

        except Exception as e:
            logger.error(f"Error during control state cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, "_initialized") and self._initialized:
            self.cleanup()
