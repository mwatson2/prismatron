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
    """Legacy playback state enumeration for compatibility."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    ERROR = "error"


class ProducerState(Enum):
    """Producer state enumeration for content generation."""

    STOPPED = "stopped"
    PLAYING = "playing"
    ERROR = "error"


class RendererState(Enum):
    """Renderer state enumeration for LED playback."""

    STOPPED = "stopped"
    WAITING = "waiting"
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

    # Legacy field for compatibility
    play_state: PlayState = PlayState.STOPPED

    # New decoupled state fields
    producer_state: ProducerState = ProducerState.STOPPED
    renderer_state: RendererState = RendererState.STOPPED

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

    # Audio beat detection state
    audio_enabled: bool = False
    current_bpm: float = 120.0
    beat_count: int = 0
    last_beat_time: float = 0.0
    last_downbeat_time: float = 0.0
    beat_confidence: float = 0.0
    audio_intensity: float = 0.0

    # Audio level and AGC state
    audio_level: float = 0.0  # Raw RMS audio level (0-1)
    agc_gain_db: float = 0.0  # Current AGC gain in dB

    # Build-up/drop detection state
    buildup_state: str = "NORMAL"  # NORMAL or BUILDUP
    buildup_intensity: float = 0.0  # Continuous build-up progression (can exceed 1.0)
    bass_energy: float = 0.0  # Current bass energy level
    high_energy: float = 0.0  # Current high-frequency energy level
    last_cut_time: float = 0.0  # Wall-clock time of last cut event
    last_drop_time: float = 0.0  # Wall-clock time of last drop event

    # Audio reactive effects control
    audio_reactive_enabled: bool = False
    use_audio_test_file: bool = True  # True = test file, False = live microphone

    # Audio reactive trigger configuration (new framework)
    audio_reactive_trigger_config: dict = None  # Dictionary with test_interval and rules list

    # Beat brightness boost settings for audio reactive effects (legacy)
    beat_brightness_enabled: bool = True
    beat_brightness_intensity: float = 2.5  # 0.0 to 5.0 (multiplier for beat boost)
    beat_brightness_duration: float = 0.25  # 0.1 to 1.0 (fraction of beat duration)
    beat_confidence_threshold: float = 0.5  # 0.0 to 1.0 (minimum confidence to apply boost)

    # Optimization settings
    optimization_iterations: int = 10

    # Buffer monitoring for state transitions
    led_buffer_frames: int = 0
    led_buffer_capacity: int = 0

    # Timing synchronization for producer/consumer coordination
    # wallclock_delta: wallclock_time = frame_timestamp + wallclock_delta
    wallclock_delta: Optional[float] = None

    # Playlist transition timing option
    # When enabled, content items end on the last downbeat to sync with music
    transition_on_downbeat_enabled: bool = False

    # Audio recording control (for capturing processed microphone audio)
    audio_recording_requested: bool = False  # Set to True to trigger recording
    audio_recording_duration: float = 60.0  # Duration in seconds
    audio_recording_output_path: str = ""  # Output file path
    audio_recording_in_progress: bool = False  # True while recording
    audio_recording_progress: float = 0.0  # Recording progress 0.0-1.0
    audio_recording_status: str = ""  # Status message (e.g., "recording", "complete", "error")


class ControlState:
    """
    Lightweight IPC system for process coordination and configuration.

    Provides shared state management for system control, configuration,
    and status monitoring across multiple processes.
    """

    # Class-level shared lock that will be set by the main process
    _shared_lock: Optional[Any] = None  # multiprocessing.Lock

    # Class-level shared events that will be set by the main process
    _shared_shutdown_event: Optional[Any] = None  # multiprocessing.Event
    _shared_restart_event: Optional[Any] = None  # multiprocessing.Event
    _shared_reboot_event: Optional[Any] = None  # multiprocessing.Event
    _shared_config_updated_event: Optional[Any] = None  # multiprocessing.Event
    _shared_status_updated_event: Optional[Any] = None  # multiprocessing.Event

    @classmethod
    def set_shared_lock(cls, lock: Any) -> None:  # mp.Lock
        """
        Set the shared lock to be used by all ControlState instances.
        This must be called by the main process before creating subprocesses.

        Args:
            lock: A multiprocessing.Lock created by the main process
        """
        cls._shared_lock = lock
        logger.info("Shared lock set for inter-process synchronization")

    @classmethod
    def set_shared_events(
        cls,
        shutdown_event: Any,  # mp.Event
        restart_event: Any,  # mp.Event
        reboot_event: Any,  # mp.Event
        config_updated_event: Any,  # mp.Event
        status_updated_event: Any,  # mp.Event
    ) -> None:
        """
        Set the shared events to be used by all ControlState instances.
        This must be called by the main process before creating subprocesses.

        Args:
            shutdown_event: Shared shutdown event
            restart_event: Shared restart event
            reboot_event: Shared reboot event
            config_updated_event: Shared config update event
            status_updated_event: Shared status update event
        """
        cls._shared_shutdown_event = shutdown_event
        cls._shared_restart_event = restart_event
        cls._shared_reboot_event = reboot_event
        cls._shared_config_updated_event = config_updated_event
        cls._shared_status_updated_event = status_updated_event
        logger.info("Shared events set for inter-process coordination")

    def __init__(self, name: str = "prismatron_control"):
        """
        Initialize the control state manager.

        Args:
            name: Unique name for the shared memory segment
        """
        self.name = name
        self._shared_memory: Optional[Any] = None
        self._control_dict = None
        # Use the shared lock if available, otherwise create a local one
        # The shared lock should be set by the main process before creating subprocesses
        if ControlState._shared_lock is not None:
            self._lock = ControlState._shared_lock
        else:
            # Fallback to local lock (will only work within same process)
            self._lock = mp.Lock()
            logger.warning("Using process-local lock - inter-process synchronization may not work correctly")
        self._initialized = False
        self._start_time = time.time()
        self._is_creator = False  # Track if this instance created the shared memory

        # Use shared events if available, otherwise create local ones
        if ControlState._shared_shutdown_event is not None:
            self._shutdown_event = ControlState._shared_shutdown_event
            self._restart_event = ControlState._shared_restart_event
            self._reboot_event = ControlState._shared_reboot_event
            self._config_updated_event = ControlState._shared_config_updated_event
            self._status_updated_event = ControlState._shared_status_updated_event
        else:
            # Fallback to local events (will only work within same process)
            self._shutdown_event = mp.Event()
            self._restart_event = mp.Event()
            self._reboot_event = mp.Event()
            self._config_updated_event = mp.Event()
            self._status_updated_event = mp.Event()
            logger.warning("Using process-local events - inter-process event coordination may not work correctly")

        # Default status
        self._default_status = SystemStatus()

        # Cache the last successfully read status to use when JSON is corrupted
        self._last_good_status: Optional[SystemStatus] = None

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
            status_dict["producer_state"] = test_status.producer_state.value
            status_dict["renderer_state"] = test_status.renderer_state.value

            json_data = json.dumps(status_dict, separators=(",", ":"))
            estimated_size = len(json_data.encode("utf-8"))

            # Add 50% safety margin and align to 1KB boundary
            safe_size = int(estimated_size * 1.5)
            aligned_size = ((safe_size + 1023) // 1024) * 1024  # Round up to 1KB

            return max(aligned_size, 4096)  # Minimum 4KB (increased from 2KB for build-drop fields)

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
                status_dict["producer_state"] = status.producer_state.value
                status_dict["renderer_state"] = status.renderer_state.value

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

                # Handle new state fields with backward compatibility
                if "producer_state" in status_dict:
                    status_dict["producer_state"] = ProducerState(status_dict["producer_state"])
                if "renderer_state" in status_dict:
                    status_dict["renderer_state"] = RendererState(status_dict["renderer_state"])

                # Successfully parsed - cache this as last good status
                new_status = SystemStatus(**status_dict)
                self._last_good_status = new_status
                return new_status

        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            logger.warning(f"Failed to decode status JSON: {e}")
            # Return last known good status if available, otherwise default
            if self._last_good_status is not None:
                logger.debug("Using cached last good status due to JSON decode error")
                return self._last_good_status
            return self._default_status
        except Exception as e:
            logger.error(f"Failed to read status: {e}")
            # Return last known good status if available, otherwise default
            if self._last_good_status is not None:
                logger.debug("Using cached last good status due to read error")
                return self._last_good_status
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
                if result and self._status_updated_event:
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
                if result and self._config_updated_event:
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
                if result and self._config_updated_event:
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
                if result and self._status_updated_event:
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
                if result and self._status_updated_event:
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
                if result and self._status_updated_event:
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

            if self._shutdown_event:
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

            if self._restart_event:
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

            if self._reboot_event:
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
        return self._shutdown_event.is_set() if self._shutdown_event else False

    def is_restart_requested(self) -> bool:
        """
        Check if restart has been requested.

        Returns:
            True if restart requested, False otherwise
        """
        return self._restart_event.is_set() if self._restart_event else False

    def is_reboot_requested(self) -> bool:
        """
        Check if reboot has been requested.

        Returns:
            True if reboot requested, False otherwise
        """
        return self._reboot_event.is_set() if self._reboot_event else False

    def should_shutdown(self) -> bool:
        """
        Check if any shutdown condition exists (shutdown, restart, or reboot).

        Returns:
            True if any shutdown condition is active, False otherwise
        """
        shutdown = self._shutdown_event.is_set() if self._shutdown_event else False
        restart = self._restart_event.is_set() if self._restart_event else False
        reboot = self._reboot_event.is_set() if self._reboot_event else False
        return shutdown or restart or reboot

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for shutdown signal.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if shutdown signaled, False if timeout
        """
        if not self._shutdown_event:
            return False
        return self._shutdown_event.wait(timeout)

    def wait_for_config_update(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for configuration update.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if config updated, False if timeout
        """
        if not self._config_updated_event:
            return False
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
        if not self._status_updated_event:
            return False
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
            status_dict["producer_state"] = status.producer_state.value
            status_dict["renderer_state"] = status.renderer_state.value
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
                if result and self._status_updated_event:
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
            builddrop_updated = False
            for field_name, value in updates.items():
                if hasattr(current_status, field_name):
                    setattr(current_status, field_name, value)
                    # Track if build-drop fields are being updated
                    if field_name in (
                        "buildup_state",
                        "buildup_intensity",
                        "bass_energy",
                        "high_energy",
                        "last_cut_time",
                        "last_drop_time",
                    ):
                        builddrop_updated = True
                else:
                    logger.warning(f"Unknown status field: {field_name}")

            # Log build-drop updates (with sampling to avoid spam)
            if builddrop_updated:
                if not hasattr(self, "_builddrop_update_counter"):
                    self._builddrop_update_counter = 0
                self._builddrop_update_counter += 1
                if self._builddrop_update_counter % 20 == 0:  # Log every 20th update
                    logger.info(
                        f"ControlState: build-drop update #{self._builddrop_update_counter}: "
                        f"state={getattr(current_status, 'buildup_state', 'N/A')}, "
                        f"intensity={getattr(current_status, 'buildup_intensity', 0.0):.2f}, "
                        f"bass={getattr(current_status, 'bass_energy', 0.0):.6f}, "
                        f"high={getattr(current_status, 'high_energy', 0.0):.6f}"
                    )

            result = self._write_status(current_status)
            if result and self._status_updated_event:
                self._status_updated_event.set()
            return result

        except Exception as e:
            logger.error(f"Failed to update status fields: {e}")
            return False

    def set_producer_state(self, state: ProducerState) -> bool:
        """
        Set the producer state independently.

        Args:
            state: New producer state

        Returns:
            True if successful, False otherwise
        """
        try:
            current_status = self._read_status()
            if current_status:
                current_status.producer_state = state
                # Update legacy play_state for compatibility
                current_status.play_state = self._compute_legacy_play_state(state, current_status.renderer_state)
                result = self._write_status(current_status)
                if result and self._status_updated_event:
                    self._status_updated_event.set()
                return result
            return False

        except Exception as e:
            logger.error(f"Failed to set producer state: {e}")
            return False

    def set_renderer_state(self, state: RendererState) -> bool:
        """
        Set the renderer state independently.

        Args:
            state: New renderer state

        Returns:
            True if successful, False otherwise
        """
        try:
            current_status = self._read_status()
            if current_status:
                current_status.renderer_state = state
                # Update legacy play_state for compatibility
                current_status.play_state = self._compute_legacy_play_state(current_status.producer_state, state)
                result = self._write_status(current_status)
                if result and self._status_updated_event:
                    self._status_updated_event.set()
                return result
            return False

        except Exception as e:
            logger.error(f"Failed to set renderer state: {e}")
            return False

    def _compute_legacy_play_state(self, producer_state: ProducerState, renderer_state: RendererState) -> PlayState:
        """
        Compute legacy PlayState from decoupled producer/renderer states.

        This maintains backward compatibility for existing code that relies on PlayState.

        Args:
            producer_state: Current producer state
            renderer_state: Current renderer state

        Returns:
            Effective legacy PlayState
        """
        # Error states take priority
        if producer_state == ProducerState.ERROR or renderer_state == RendererState.ERROR:
            return PlayState.ERROR

        # If renderer is paused, system is paused regardless of producer
        if renderer_state == RendererState.PAUSED:
            return PlayState.PAUSED

        # If both are playing, system is playing
        if producer_state == ProducerState.PLAYING and renderer_state == RendererState.PLAYING:
            return PlayState.PLAYING

        # If producer is playing but renderer is waiting, system is playing (starting up)
        if producer_state == ProducerState.PLAYING and renderer_state == RendererState.WAITING:
            return PlayState.PLAYING

        # Otherwise system is stopped
        return PlayState.STOPPED

    def update_buffer_status(self, frames: int, capacity: int) -> bool:
        """
        Update LED buffer monitoring information.

        Args:
            frames: Current number of frames in buffer
            capacity: Total buffer capacity

        Returns:
            True if successful, False otherwise
        """
        try:
            current_status = self._read_status()
            if current_status:
                current_status.led_buffer_frames = frames
                current_status.led_buffer_capacity = capacity
                result = self._write_status(current_status)
                if result and self._status_updated_event:
                    self._status_updated_event.set()
                return result
            return False

        except Exception as e:
            logger.error(f"Failed to update buffer status: {e}")
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
