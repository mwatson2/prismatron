"""
Producer Process Core.

This module implements the main producer process that loads content,
manages playlists, and renders frames to shared memory buffers for
the consumer process.
"""

import contextlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..const import FRAME_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH
from ..core.control_state import ControlState, PlayState, ProducerState, SystemState
from ..core.playlist_sync import PlaylistState as SyncPlaylistState
from ..core.playlist_sync import PlaylistSyncClient, TransitionConfig
from ..core.shared_buffer import FrameProducer
from ..utils.frame_timing import FrameTimingData
from .content_preparer import ContentPreparer
from .content_sources import (
    ContentSource,
    ContentSourceRegistry,
    ContentStatus,
    ContentType,
    FrameData,
)

logger = logging.getLogger(__name__)


class PlaylistItem:
    """Represents an item in the content playlist."""

    def __init__(
        self,
        filepath: str,
        duration: Optional[float] = None,
        repeat: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        transition_in: Optional[TransitionConfig] = None,
        transition_out: Optional[TransitionConfig] = None,
    ):
        """
        Initialize playlist item.

        Args:
            filepath: Path to content file
            duration: Override duration (None = use file duration)
            repeat: Number of times to repeat (1 = play once)
            metadata: Additional metadata
            transition_in: Transition configuration for item start
            transition_out: Transition configuration for item end
        """
        self.filepath = filepath
        self.duration_override = duration
        self.repeat_count = repeat
        self.metadata = metadata or {}
        self.current_repeat = 0

        # Transition configurations
        self.transition_in = transition_in or TransitionConfig()
        self.transition_out = transition_out or TransitionConfig()

        # Content source will be created when needed
        self._content_source: Optional[ContentSource] = None
        self._detected_type: Optional[ContentType] = None

    def get_content_source(self) -> Optional[ContentSource]:
        """
        Get or create content source for this item.

        Returns:
            ContentSource instance or None if creation failed
        """
        if self._content_source is None:
            # Detect content type
            self._detected_type = ContentSourceRegistry.detect_content_type(self.filepath)

            # Create content source
            self._content_source = ContentSourceRegistry.create_source(self.filepath, self._detected_type)

        return self._content_source

    def cleanup(self) -> None:
        """Clean up content source resources."""
        if self._content_source:
            self._content_source.cleanup()
            self._content_source = None

    def reset_repeats(self) -> None:
        """Reset repeat counter."""
        self.current_repeat = 0

    def should_repeat(self) -> bool:
        """Check if item should repeat."""
        return self.current_repeat < self.repeat_count

    def get_effective_duration(self) -> float:
        """Get effective duration (override or source duration)."""
        if self.duration_override is not None:
            return self.duration_override

        source = self.get_content_source()
        if source:
            return source.get_duration()

        return 0.0

    def __str__(self) -> str:
        """String representation."""
        return f"PlaylistItem('{self.filepath}', repeat={self.repeat_count})"


class ContentPlaylist:
    """Manages a playlist of content items."""

    def __init__(self):
        """Initialize empty playlist."""
        self._items: List[PlaylistItem] = []
        self._current_index = 0
        self._loop_playlist = True
        self._lock = threading.RLock()  # Reentrant lock for nested calls

    def add_item(
        self,
        filepath: str,
        duration: Optional[float] = None,
        repeat: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        transition_in: Optional[TransitionConfig] = None,
        transition_out: Optional[TransitionConfig] = None,
    ) -> bool:
        """
        Add item to playlist.

        Args:
            filepath: Path to content file or JSON config for text content
            duration: Override duration
            repeat: Repeat count
            metadata: Additional metadata
            transition_in: Transition configuration for item start
            transition_out: Transition configuration for item end

        Returns:
            True if added successfully, False otherwise
        """
        with self._lock:
            try:
                # Detect content type to handle text content vs file content
                content_type = ContentSourceRegistry.detect_content_type(filepath)

                # For JSON content (text or effects), filepath contains JSON config, not a file path
                if content_type in (ContentType.TEXT, ContentType.EFFECT):
                    # Validate JSON configuration
                    import json

                    try:
                        config = json.loads(filepath)
                        if content_type == ContentType.TEXT:
                            if not ("text" in config and isinstance(config["text"], str)):
                                logger.error(f"Invalid text configuration: {filepath}")
                                return False
                            logger.debug(f"Adding text content: {config.get('text', '')[:50]}...")
                        elif content_type == ContentType.EFFECT:
                            if "effect_id" not in config:
                                logger.error(f"Invalid effect configuration: {filepath}")
                                return False
                            logger.debug(f"Adding effect content: {config.get('effect_id')}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON configuration: {e}")
                        return False
                else:
                    # For file-based content, validate file exists
                    if not os.path.exists(filepath):
                        logger.error(f"Content file not found: {filepath}")
                        return False

                item = PlaylistItem(filepath, duration, repeat, metadata, transition_in, transition_out)
                self._items.append(item)

                if content_type == ContentType.TEXT:
                    logger.info("Added text content to playlist")
                elif content_type == ContentType.EFFECT:
                    logger.info(f"Added effect content to playlist: {json.loads(filepath).get('effect_id', 'unknown')}")
                else:
                    logger.info(f"Added to playlist: {filepath}")
                return True

            except Exception as e:
                logger.error(f"Failed to add playlist item: {e}")
                return False

    def remove_item(self, index: int) -> bool:
        """
        Remove item from playlist.

        Args:
            index: Index of item to remove

        Returns:
            True if removed successfully, False otherwise
        """
        with self._lock:
            try:
                if 0 <= index < len(self._items):
                    item = self._items.pop(index)
                    item.cleanup()

                    # Adjust current index if needed
                    if self._current_index >= len(self._items):
                        self._current_index = 0
                    elif self._current_index > index:
                        self._current_index -= 1

                    logger.info(f"Removed from playlist: {item.filepath}")
                    return True
                else:
                    logger.error(f"Invalid playlist index: {index}")
                    return False

            except Exception as e:
                logger.error(f"Failed to remove playlist item: {e}")
                return False

    def remove_item_by_filepath(self, filepath: str) -> bool:
        """
        Remove item from playlist by filepath.

        Args:
            filepath: Path of item to remove

        Returns:
            True if removed successfully, False otherwise
        """
        with self._lock:
            try:
                for i, item in enumerate(self._items):
                    if item.filepath == filepath:
                        return self.remove_item(i)
                return False

            except Exception as e:
                logger.error(f"Error removing playlist item by filepath: {e}")
                return False

    def clear(self) -> None:
        """Clear all items from playlist."""
        with self._lock:
            for item in self._items:
                item.cleanup()
            self._items.clear()
            self._current_index = 0
            logger.info("Playlist cleared")

    def get_current_item(self) -> Optional[PlaylistItem]:
        """
        Get current playlist item.

        Returns:
            Current PlaylistItem or None if playlist is empty
        """
        with self._lock:
            if 0 <= self._current_index < len(self._items):
                return self._items[self._current_index]
            return None

    def get_current_index(self) -> Optional[int]:
        """
        Get current playlist index.

        Returns:
            Current index or None if playlist is empty
        """
        with self._lock:
            if 0 <= self._current_index < len(self._items):
                return self._current_index
            return None

    def get_item_at_index(self, index: int) -> Optional[PlaylistItem]:
        """
        Get playlist item at specific index.

        Args:
            index: Index of item to get

        Returns:
            PlaylistItem at index or None if invalid index
        """
        with self._lock:
            if 0 <= index < len(self._items):
                return self._items[index]
            return None

    def advance_to_next(self) -> bool:
        """
        Advance to next item in playlist.

        Returns:
            True if advanced to next item, False if end of playlist
        """
        with self._lock:
            current_item = self.get_current_item()
            if current_item:
                # Check if current item should repeat
                current_item.current_repeat += 1
                if current_item.should_repeat():
                    # Reset content source for repeat
                    if current_item._content_source:
                        current_item._content_source.reset()
                    return True

                # Reset repeat counter for next time
                current_item.reset_repeats()

            # Move to next item
            self._current_index += 1

            # Check if we've reached the end
            if self._current_index >= len(self._items):
                if self._loop_playlist and self._items:
                    self._current_index = 0
                    logger.debug("Playlist looped to beginning")
                    return True
                else:
                    logger.debug("End of playlist reached")
                    return False

            logger.debug(f"Advanced to playlist item {self._current_index}")
            return True

    def set_loop(self, loop: bool) -> None:
        """Set playlist loop mode."""
        with self._lock:
            self._loop_playlist = loop

    def get_playlist_info(self) -> Dict[str, Any]:
        """
        Get playlist information.

        Returns:
            Dictionary with playlist details
        """
        with self._lock:
            total_duration = sum(item.get_effective_duration() for item in self._items)

            return {
                "total_items": len(self._items),
                "current_index": self._current_index,
                "total_duration": total_duration,
                "loop_enabled": self._loop_playlist,
                "items": [
                    {
                        "filepath": item.filepath,
                        "duration": item.get_effective_duration(),
                        "repeat": item.repeat_count,
                        "type": (item._detected_type.value if item._detected_type else "unknown"),
                    }
                    for item in self._items
                ],
            }

    def __len__(self) -> int:
        """Get number of items in playlist."""
        with self._lock:
            return len(self._items)

    @contextlib.contextmanager
    def batch_update(self):
        """Context manager for performing multiple playlist operations atomically."""
        with self._lock:
            yield self

    def set_current_index(self, index: int) -> None:
        """Set current playlist index (used internally, can be called within batch_update)."""
        with self._lock:
            if 0 <= index < len(self._items):
                self._current_index = index
            else:
                self._current_index = 0

    def get_current_item_and_index(self) -> Tuple[Optional["PlaylistItem"], Optional[int]]:
        """Atomically get current item and index together to avoid race conditions."""
        with self._lock:
            if 0 <= self._current_index < len(self._items):
                return self._items[self._current_index], self._current_index
            return None, None


class ProducerProcess:
    """
    Main producer process that manages content loading and frame rendering.

    Handles playlist management, content source coordination, and frame
    output to shared memory buffers.
    """

    def __init__(
        self,
        buffer_name: str = "prismatron_buffer",
        control_name: str = "prismatron_control",
    ):
        """
        Initialize producer process.

        Args:
            buffer_name: Name for shared memory buffer
            control_name: Name for control state
        """
        self.buffer_name = buffer_name
        self.control_name = control_name

        # Core components
        self._frame_buffer = FrameProducer(buffer_name)
        self._control_state = ControlState(control_name)
        self._playlist = ContentPlaylist()
        self._playlist_sync_client: Optional[PlaylistSyncClient] = None

        # Current content and state
        self._current_source: Optional[ContentSource] = None
        self._current_item: Optional[PlaylistItem] = None
        self._current_item_index: int = -1  # Track the index of the currently loaded content
        self._is_first_frame_of_current_item = True  # Track first frame of each playlist item
        self._content_finished_processed = False  # Prevent multiple next_item() calls for same content
        self._waiting_for_sync_response = False  # Prevent loading content while waiting for sync service

        # Threading and timing
        self._producer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._sync_lock = threading.RLock()  # Protect producer state from sync callbacks

        # Frame timing
        self._target_fps = 30.0  # Default target FPS
        self._frame_interval = 1.0 / self._target_fps
        self._last_frame_time = 0.0

        # Statistics
        self._frames_produced = 0
        self._start_time = 0.0

        # Periodic logging for pipeline debugging
        self._last_log_time = 0.0
        self._log_interval = 2.0  # Log every 2 seconds
        self._frames_with_content = 0  # Frames with non-zero content
        self._last_logged_state = None  # Track last logged state to avoid repetitive STOPPED logs
        self._last_logged_frames_produced = 0  # Track last logged frame count to avoid spam when paused

        # Global timestamp mapping state
        self._playlist_start_time = 0.0  # When the playlist started playing
        self._item_start_times: List[float] = []  # Cumulative start times for each item
        self._current_item_global_offset = 0.0  # Global offset for current item
        self._accumulated_duration = 0.0  # Accumulated actual durations from completed items
        self._last_frame_duration: Optional[float] = None  # Duration from last frame for accumulation

        # Content pre-preparer for seamless transitions
        self._content_preparer: Optional[ContentPreparer] = None

    def initialize(self) -> bool:
        """
        Initialize producer process components.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize shared memory buffer
            if not self._frame_buffer.initialize():
                logger.error("Failed to initialize frame buffer")
                return False

            # Connect to existing control state (created by main process)
            if not self._control_state.connect():
                logger.error("Failed to connect to control state")
                return False

            # Set initial system state
            self._control_state.update_system_state(SystemState.RUNNING)
            self._control_state.set_producer_state(ProducerState.STOPPED)

            # Connect to playlist synchronization service
            self._playlist_sync_client = PlaylistSyncClient(client_name="producer")
            self._playlist_sync_client.on_playlist_update = self._on_playlist_sync_update
            if self._playlist_sync_client.connect():
                logger.info("Producer connected to playlist synchronization service")
            else:
                logger.warning("Producer failed to connect to playlist synchronization service - using fallback")

            # Create content preparer for seamless transitions (timer-based, no polling thread)
            # 5s lookahead gives enough buffer for downbeat transition truncation
            self._content_preparer = ContentPreparer(self._playlist, lookahead_seconds=5.0)
            logger.info("Content preparer initialized")

            logger.info("Producer process initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Producer initialization failed: {e}")
            return False

    def add_content(self, filepath: str, duration: Optional[float] = None, repeat: int = 1) -> bool:
        """
        Add content to playlist.

        Args:
            filepath: Path to content file
            duration: Override duration
            repeat: Repeat count

        Returns:
            True if added successfully, False otherwise
        """
        # If connected to sync service, add through sync service
        if self._playlist_sync_client and self._playlist_sync_client.connected:
            import uuid

            from ..core.playlist_sync import PlaylistItem as SyncPlaylistItem

            # Determine content type
            file_path = Path(filepath)
            file_ext = file_path.suffix.lower()
            if file_ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}:
                content_type = "image"
            elif file_ext in {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm"}:
                content_type = "video"
            else:
                content_type = "image"  # Default

            sync_item = SyncPlaylistItem(
                id=str(uuid.uuid4()),
                name=file_path.name,
                type=content_type,
                file_path=filepath,
                duration=duration,
                created_at=time.time(),
                order=len(self._playlist._items),  # Current playlist length as order
            )

            return self._playlist_sync_client.add_item(sync_item)
        else:
            logger.error("Playlist sync client not connected - cannot add content")
            return False

    def load_playlist_from_directory(self, directory: str) -> int:
        """
        Load all supported content from directory into playlist.

        Args:
            directory: Directory path to scan

        Returns:
            Number of files added to playlist
        """
        added_count = 0

        try:
            directory_path = Path(directory)
            if not directory_path.exists() or not directory_path.is_dir():
                logger.error(f"Directory not found: {directory}")
                return 0

            # Supported extensions
            supported_extensions = {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".webp",  # Images
                ".mp4",
                ".avi",
                ".mkv",
                ".mov",
                ".wmv",
                ".webm",  # Videos
                ".gif",  # Animations
            }

            # Scan directory and populate sync service
            import uuid

            from ..core.playlist_sync import PlaylistItem as SyncPlaylistItem

            for file_path in directory_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    # Determine content type
                    file_ext = file_path.suffix.lower()
                    if file_ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}:
                        content_type = "image"
                    elif file_ext in {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm"}:
                        content_type = "video"
                    else:
                        content_type = "image"  # Default

                    # Create sync playlist item
                    sync_item = SyncPlaylistItem(
                        id=str(uuid.uuid4()),
                        name=file_path.name,
                        type=content_type,
                        file_path=str(file_path),
                        duration=None,
                        created_at=time.time(),
                        order=added_count,
                    )

                    # Add to sync service if available
                    if self._playlist_sync_client and self._playlist_sync_client.connected:
                        if self._playlist_sync_client.add_item(sync_item):
                            added_count += 1
                        else:
                            logger.warning(f"Failed to add {file_path} to sync service")
                    else:
                        logger.warning(f"Playlist sync client not connected - skipping {file_path}")

            logger.info(f"Loaded {added_count} files from {directory}")
            return added_count

        except Exception as e:
            logger.error(f"Failed to load playlist from directory: {e}")
            return added_count

    def start(self) -> bool:
        """
        Start producer process.

        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            logger.warning("Producer process already running")
            return True

        try:
            # Start producer thread
            self._producer_thread = threading.Thread(target=self._producer_main_loop, daemon=True)

            self._running = True
            self._start_time = time.time()
            self._stop_event.clear()

            self._producer_thread.start()

            logger.info("Producer process started")
            return True

        except Exception as e:
            logger.error(f"Failed to start producer process: {e}")
            self._running = False
            return False

    def stop(self) -> None:
        """Stop producer process."""
        if not self._running:
            return

        logger.info("Stopping producer process...")

        self._running = False
        self._stop_event.set()

        # Wait for producer thread to finish
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=5.0)

        # Clean up current content
        if self._current_source:
            self._current_source.cleanup()
            self._current_source = None
            self._current_item = None
            self._current_item_index = -1
            self._content_finished_processed = False  # Reset flag when producer stops

        # Update control state
        self._control_state.set_producer_state(ProducerState.STOPPED)

        logger.info("Producer process stopped")

    def _producer_main_loop(self) -> None:
        """Main producer loop running in separate thread."""
        logger.debug("Producer main loop started")

        try:
            while not self._stop_event.is_set():
                # Check control state
                status = self._control_state.get_status()
                if not status:
                    time.sleep(0.1)
                    continue

                # Playlist commands now handled through playlist sync service only

                # Log producer state changes for debugging
                if not hasattr(self, "_last_producer_state") or self._last_producer_state != status.producer_state:
                    logger.debug(f"Producer state changed to {status.producer_state}")
                    self._last_producer_state = status.producer_state

                # Handle producer state
                if status.producer_state == ProducerState.PLAYING:
                    self._handle_playing_state()
                else:
                    self._handle_stopped_state()

                # Update frame rate statistics
                self._update_statistics()

                # Periodic logging for pipeline debugging (only when playing or if there's activity)
                current_time = time.time()
                if current_time - self._last_log_time >= self._log_interval:
                    current_state = status.producer_state if status else None

                    # Check if frame count has changed (indicates actual progress)
                    frames_changed = self._frames_produced != self._last_logged_frames_produced
                    state_changed = current_state != self._last_logged_state

                    # Only log if frames changed OR state changed (avoid spam when paused)
                    should_log = frames_changed or state_changed

                    # For STOPPED state, only log once when entering the state
                    if current_state == ProducerState.STOPPED and self._last_logged_state == ProducerState.STOPPED:
                        should_log = False

                    if should_log:
                        content_ratio = (self._frames_with_content / max(1, self._frames_produced)) * 100
                        producer_state_str = f"[{status.producer_state}]" if status else "[UNKNOWN]"
                        logger.info(
                            f"PRODUCER PIPELINE {producer_state_str}: {self._frames_produced} frames produced, "
                            f"{self._frames_with_content} with content ({content_ratio:.1f}%), "
                            f"current: {self._current_item.filepath if self._current_item else 'None'}"
                        )
                        self._last_log_time = current_time
                        self._last_logged_state = current_state
                        self._last_logged_frames_produced = self._frames_produced

                # Brief sleep to prevent busy waiting
                time.sleep(0.001)

        except Exception as e:
            logger.error(f"Producer main loop error: {e}")
            self._control_state.set_error(f"Producer error: {e}")

        finally:
            logger.debug("Producer main loop ended")

    def _handle_playing_state(self) -> None:
        """Handle playing state - produce frames."""
        try:
            # Ensure we have current content
            if not self._ensure_current_content():
                # No content available, wait for playlist sync to provide next content
                # Don't switch to stopped - we're waiting for the next item
                time.sleep(0.1)  # Prevent busy loop while waiting for sync service
                return

            # Producer should output frames as fast as possible
            # Frame rate is determined by content timestamps, not throttling
            current_time = time.time()

            # Get current source (guaranteed by _ensure_current_content)
            current_source = self._current_source
            current_item = self._current_item
            if current_source is None or current_item is None:
                return

            # Get next frame from content source
            frame_data = current_source.get_next_frame()

            if frame_data is not None:
                # Reset no-frame tracking since we got a frame
                if hasattr(self, "_no_frame_start_time"):
                    delattr(self, "_no_frame_start_time")
                if hasattr(self, "_last_no_frame_warning"):
                    delattr(self, "_last_no_frame_warning")

                # Log first frame for debugging
                if self._frames_produced == 0:
                    logger.debug(f"Got first frame from {current_item.filepath}")

                # Check if frame should be dropped for downbeat transition timing
                local_timestamp = frame_data.presentation_timestamp or 0.0
                if self._should_drop_frame_for_downbeat_transition(local_timestamp):
                    # Frame is after the last downbeat - end content item to transition on downbeat
                    current_item_name = os.path.basename(current_item.filepath)
                    logger.info(f"🎵 Ending content early for downbeat transition: {current_item_name}")

                    # Calculate truncated duration: content played up to this point
                    # local_timestamp is the timestamp of the frame we're dropping, which equals
                    # the end time of the last rendered frame (previous frame ends when this one starts)
                    truncated_duration = local_timestamp
                    self._accumulated_duration += truncated_duration
                    logger.debug(
                        f"Downbeat transition: truncated duration {truncated_duration:.3f}s, "
                        f"accumulated total now {self._accumulated_duration:.3f}s"
                    )
                    self._last_frame_duration = None  # Reset for next item

                    # Mark as finished and advance to next
                    self._content_finished_processed = True
                    self._content_finished_time = time.time()
                    self._waiting_for_sync_response = True
                    self._advance_to_next_content()

                    # Clean up current source
                    if self._current_source:
                        self._current_source.cleanup()
                        self._current_source = None
                        self._current_item = None
                        self._logged_downbeat_drop = False  # Reset for next item
                        logger.debug("Cleaned up content source for downbeat transition")
                    return

                # Render frame to shared memory
                if self._render_frame_to_buffer(frame_data):
                    self._frames_produced += 1
                    self._last_frame_time = current_time

            elif current_source.is_finished():
                # Prevent duplicate next_item() calls for the same content
                if not self._content_finished_processed:
                    # Content finished, accumulate actual duration and advance to next
                    if self._last_frame_duration is not None:
                        self._accumulated_duration += self._last_frame_duration
                        logger.debug(
                            f"Item completed - accumulated {self._last_frame_duration:.3f}s, total now {self._accumulated_duration:.3f}s"
                        )
                        self._last_frame_duration = None  # Reset for next item

                    current_item_name = os.path.basename(current_item.filepath)
                    content_type = current_item._detected_type.value if current_item._detected_type else "unknown"
                    logger.info(f"Content finished: {current_item_name} (type: {content_type})")

                    self._content_finished_processed = True
                    self._content_finished_time = time.time()  # Track when we first detected finish
                    logger.debug(f"Advancing to next content after {current_item_name} finished")

                    # Set flag to prevent race condition with sync callback
                    self._waiting_for_sync_response = True
                    self._advance_to_next_content()

                    # Clean up current source immediately to prevent busy loop
                    # The sync service will trigger reload of next content via _on_playlist_sync_update
                    if self._current_source:
                        self._current_source.cleanup()
                        self._current_source = None
                        self._current_item = None
                        logger.debug("Cleaned up finished content source, waiting for sync service response")
            else:
                # No frame available yet (but not finished) - track how long we've been waiting
                if not hasattr(self, "_no_frame_start_time"):
                    self._no_frame_start_time = time.time()
                    logger.debug(f"No frame available from {current_item.filepath} (waiting...)")
                else:
                    # Log at INFO level if we've been waiting too long
                    wait_time = time.time() - self._no_frame_start_time
                    if wait_time >= 2.0 and (
                        not hasattr(self, "_last_no_frame_warning") or time.time() - self._last_no_frame_warning >= 5.0
                    ):
                        # Gather diagnostic info from video source if available
                        diagnostics = ""
                        if hasattr(current_source, "_ffmpeg_process") and current_source._ffmpeg_process:
                            poll_result = current_source._ffmpeg_process.poll()
                            ffmpeg_status = "running" if poll_result is None else f"exited({poll_result})"
                            reader_status = (
                                "alive"
                                if (
                                    hasattr(current_source, "_frame_reader_thread")
                                    and current_source._frame_reader_thread
                                    and current_source._frame_reader_thread.is_alive()
                                )
                                else "dead"
                            )
                            diagnostics = f", FFmpeg: {ffmpeg_status}, reader thread: {reader_status}"

                        logger.info(
                            f"No frames from {os.path.basename(current_item.filepath)} for {wait_time:.1f}s - "
                            f"content source may be stalled (status: {current_source.status}{diagnostics})"
                        )
                        self._last_no_frame_warning = time.time()

        except Exception as e:
            logger.error(f"Error in playing state: {e}", exc_info=True)
            self._control_state.set_error(f"Playback error: {e}")

    def _handle_stopped_state(self) -> None:
        """Handle stopped state."""
        # Clean up current content if any
        if self._current_source:
            self._current_source.cleanup()
            self._current_source = None
            self._current_item = None
            self._current_item_index = -1
            self._content_finished_processed = False  # Reset flag when content is cleaned up

        time.sleep(0.1)

    def _ensure_current_content(self) -> bool:
        """
        Ensure we have current content loaded.

        Returns:
            True if content is available, False otherwise
        """
        # Acquire lock to prevent sync callback from clearing content during setup
        with self._sync_lock:
            if self._current_source and self._current_source.status == ContentStatus.PLAYING:
                return True

            # Don't try to load content if we're waiting for sync service to respond
            # This prevents race condition where we load stale content before sync update arrives
            if self._waiting_for_sync_response:
                # Check for timeout - if we've been waiting too long, clear the flag and proceed
                # This prevents permanent lockup if sync service doesn't respond
                if hasattr(self, "_content_finished_time"):
                    wait_time = time.time() - self._content_finished_time
                    if wait_time > 5.0:
                        logger.warning(
                            f"Sync service response timeout after {wait_time:.1f}s - clearing wait flag and proceeding"
                        )
                        self._waiting_for_sync_response = False
                        # Fall through to load content from current playlist state
                    else:
                        logger.debug("Waiting for sync service response before loading content")
                        return False
                else:
                    logger.debug("Waiting for sync service response before loading content")
                    return False

            # Atomically get current playlist item and index to avoid race conditions
            current_item, current_playlist_index = self._playlist.get_current_item_and_index()
            if not current_item or current_playlist_index is None:
                logger.debug("No current playlist item available")
                return False

            logger.debug(
                f"_ensure_current_content called - playlist_index={current_playlist_index}, loaded_index={self._current_item_index}"
            )

            # Load content source if needed
            if self._current_item != current_item:
                # Clean up previous content
                if self._current_source:
                    self._current_source.cleanup()

                # Try to get pre-prepared source first (fast path)
                pending_source = None
                if self._content_preparer:
                    pending_source = self._content_preparer.get_pending_source(
                        expected_index=current_playlist_index,
                        expected_filepath=current_item.filepath,
                    )

                if pending_source:
                    # Use pre-prepared source (seamless transition)
                    logger.info(f"Using pre-prepared content source for index {current_playlist_index}")
                    self._current_source = pending_source
                    self._current_item = current_item
                    self._current_item_index = current_playlist_index
                    self._is_first_frame_of_current_item = True
                    self._content_finished_processed = False

                    # Calculate global timestamp offset for this item
                    self._update_global_timestamp_offset()

                    # Update control state with current file
                    self._control_state.set_current_file(current_item.filepath)

                    # Adjust target FPS based on content
                    if hasattr(self._current_source, "content_info"):
                        content_fps = self._current_source.content_info.fps
                        if content_fps > 0:
                            self._target_fps = min(content_fps, 60.0)
                            self._frame_interval = 1.0 / self._target_fps

                    # Schedule pre-preparation of NEXT item
                    item_duration = self._current_source.get_duration()
                    if item_duration > 0 and self._content_preparer is not None:
                        self._content_preparer.schedule_preparation(
                            current_index=current_playlist_index,
                            item_duration=item_duration,
                        )

                    logger.info(f"Pre-prepared content ready: {current_item.filepath} (fps: {self._target_fps})")
                else:
                    # Fall back to inline setup (no pre-prepared source available)
                    # Check if this is an effect (JSON config)
                    content_type = ContentSourceRegistry.detect_content_type(current_item.filepath)
                    if content_type == ContentType.EFFECT:
                        logger.info("Loading effect content from JSON config")
                        import json

                        try:
                            config = json.loads(current_item.filepath)
                            logger.info(f"Effect config: {config}")
                        except Exception as e:
                            logger.error(f"Failed to parse effect config: {e}")
                    else:
                        logger.info(f"Loading content (inline): {os.path.basename(current_item.filepath)}")

                    self._current_source = current_item.get_content_source()
                    self._current_item = current_item
                    self._current_item_index = current_playlist_index
                    self._is_first_frame_of_current_item = True
                    self._content_finished_processed = False

                    # Log the index update for debugging
                    logger.debug(
                        f"Loaded content at index {self._current_item_index}: {os.path.basename(current_item.filepath)}"
                    )

                    # Calculate global timestamp offset for this item
                    self._update_global_timestamp_offset()

                    if not self._current_source:
                        logger.error(
                            f"Failed to create content source for type {content_type}: {current_item.filepath[:100]}"
                        )
                        return False
                    else:
                        logger.info(f"Created content source: {type(self._current_source).__name__}")

                    # Setup content source
                    logger.info(f"Setting up content source {type(self._current_source).__name__}...")
                    logger.debug(
                        f"About to call setup() on {type(self._current_source).__name__} for {current_item.filepath}"
                    )

                    try:
                        setup_success = self._current_source.setup()
                        logger.debug(f"setup() returned: {setup_success}")

                        if not setup_success:
                            logger.error(f"Failed to setup content: {current_item.filepath[:100]}")
                            return False

                        logger.info(f"Content source setup successful, status={self._current_source.status}")
                    except Exception as setup_error:
                        logger.error(f"Exception during content source setup: {setup_error}", exc_info=True)
                        return False

                    # Update control state with current file
                    self._control_state.set_current_file(current_item.filepath)

                    # Adjust target FPS based on content
                    if hasattr(self._current_source, "content_info"):
                        content_fps = self._current_source.content_info.fps
                        if content_fps > 0:
                            self._target_fps = min(content_fps, 60.0)
                            self._frame_interval = 1.0 / self._target_fps

                    # Schedule pre-preparation of next item
                    if self._content_preparer:
                        item_duration = self._current_source.get_duration()
                        if item_duration > 0:
                            self._content_preparer.schedule_preparation(
                                current_index=current_playlist_index,
                                item_duration=item_duration,
                            )

                    if content_type == ContentType.EFFECT:
                        logger.info(f"Loaded effect content (fps: {self._target_fps})")
                    else:
                        logger.info(f"Loaded content: {current_item.filepath} (fps: {self._target_fps})")
            return True

    def _advance_to_next_content(self) -> None:
        """Advance to next content in playlist."""
        try:
            # If connected to sync service, send next command instead of advancing locally
            if self._playlist_sync_client and self._playlist_sync_client.connected:
                current_index = self._playlist.get_current_index()
                logger.info(f"Requesting next item from sync service (current index: {current_index})")
                success = self._playlist_sync_client.next_item()
                if not success:
                    logger.error("Failed to send next command to sync service - message send failed")
                    # Note: Don't advance locally - let sync service handle it
                else:
                    logger.debug(f"Successfully sent next_item request to sync service from index {current_index}")
            else:
                logger.error("Playlist sync client not connected - cannot advance to next content")
                self._control_state.set_producer_state(ProducerState.STOPPED)

        except Exception as e:
            logger.error(f"Failed to advance to next content: {e}")

    def _update_global_timestamp_offset(self) -> None:
        """
        Update the global timestamp offset for the current item.

        Uses accumulated actual durations from completed items plus elapsed time
        of the currently playing item to handle playlist updates during playback.
        """
        try:
            # Start with accumulated duration from completed items
            offset = self._accumulated_duration

            # If we have a current source that's playing, add its elapsed time
            # This handles the case where items are added to playlist during playback
            if self._current_source and hasattr(self._current_source, "get_current_time"):
                current_item_elapsed = self._current_source.get_current_time()
                offset += current_item_elapsed
                logger.debug(
                    f"Current item elapsed: {current_item_elapsed:.3f}s, "
                    f"adding to accumulated duration: {self._accumulated_duration:.3f}s"
                )

            self._current_item_global_offset = offset

            current_index = self._playlist.get_current_index()
            logger.debug(f"Global timestamp offset for item {current_index}: {self._current_item_global_offset:.3f}s")

        except Exception as e:
            logger.warning(f"Failed to update global timestamp offset: {e}")
            self._current_item_global_offset = 0.0

    def _update_timing_data_in_shared_memory(self, buffer_info, timing_data: FrameTimingData) -> None:
        """
        Update timing data fields in shared memory metadata.

        Args:
            buffer_info: Buffer info object from frame buffer
            timing_data: Timing data to store
        """
        try:
            # Get the frame buffer's metadata array to update timing fields
            if hasattr(self._frame_buffer, "_metadata_array") and self._frame_buffer._metadata_array is not None:
                # Use the actual buffer index that was allocated
                buffer_idx = buffer_info.buffer_index

                # Update timing fields in shared memory
                metadata_record = self._frame_buffer._metadata_array[buffer_idx]
                metadata_record["frame_index"] = timing_data.frame_index
                metadata_record["plugin_timestamp"] = timing_data.plugin_timestamp
                metadata_record["producer_timestamp"] = timing_data.producer_timestamp
                metadata_record["item_duration"] = timing_data.item_duration
                metadata_record["write_to_buffer_time"] = timing_data.write_to_buffer_time or 0.0

        except Exception as e:
            logger.warning(f"Producer: Failed to update timing data in shared memory: {e}")

    def _update_transition_metadata_in_shared_memory(
        self, buffer_info, item_timestamp: float, frame_index: int
    ) -> None:
        """
        Update transition metadata fields in shared memory.

        Args:
            buffer_info: Buffer info object from frame buffer
            item_timestamp: Time within the current playlist item (seconds from item start)
            frame_index: Current frame index for logging
        """
        try:
            # Get the frame buffer's metadata array to update transition fields
            if hasattr(self._frame_buffer, "_metadata_array") and self._frame_buffer._metadata_array is not None:
                # Use the actual buffer index that was allocated
                buffer_idx = buffer_info.buffer_index

                # Get current item and its transition configuration
                current_item = self._playlist.get_current_item()
                if current_item:
                    # Update transition fields in shared memory
                    metadata_record = self._frame_buffer._metadata_array[buffer_idx]

                    # Set transition_in configuration
                    in_type = current_item.transition_in.type if current_item.transition_in else "none"
                    in_params = current_item.transition_in.parameters if current_item.transition_in else None
                    in_duration = in_params.get("duration", 0.0) if in_params else 0.0
                    metadata_record["transition_in_type"] = in_type
                    metadata_record["transition_in_duration"] = in_duration

                    # Set transition_out configuration
                    out_type = current_item.transition_out.type if current_item.transition_out else "none"
                    out_params = current_item.transition_out.parameters if current_item.transition_out else None
                    out_duration = out_params.get("duration", 0.0) if out_params else 0.0
                    metadata_record["transition_out_type"] = out_type
                    metadata_record["transition_out_duration"] = out_duration

                    # Set item timestamp for transition calculations
                    # Note: item_duration is already set by _update_timing_data_in_shared_memory
                    # and may have been clamped based on frame_data.duration
                    metadata_record["item_timestamp"] = item_timestamp
                    # Don't overwrite item_duration - it's already correctly set
                    item_duration = metadata_record["item_duration"]

                    # Log transition metadata being written to frame
                    logger.debug(
                        f"PRODUCER: Frame {frame_index} transition metadata - "
                        f"item_timestamp={item_timestamp:.3f}s, "
                        f"item_duration={item_duration:.3f}s, "
                        f"in_type='{in_type}' (duration={in_duration:.3f}s), "
                        f"out_type='{out_type}' (duration={out_duration:.3f}s)"
                    )
                else:
                    logger.debug(f"No current item available for frame {frame_index} - transition metadata not set")
            else:
                logger.debug("Frame buffer has no metadata array - transition metadata not set")

        except Exception as e:
            logger.warning(f"Producer: Failed to update transition metadata in shared memory: {e}")

    def _render_frame_to_buffer(self, frame_data: FrameData) -> bool:
        """
        Render frame data to shared memory buffer.

        Args:
            frame_data: Frame data to render

        Returns:
            True if rendered successfully, False otherwise
        """
        try:
            # Calculate global presentation timestamp
            local_timestamp = frame_data.presentation_timestamp or 0.0
            global_timestamp = self._current_item_global_offset + local_timestamp

            # Get write buffer with global timestamp and playlist information FIRST
            # Use longer timeout since waiting for buffer availability is normal flow control
            buffer_info = self._frame_buffer.get_write_buffer(
                timeout=2.0,  # Reasonable timeout for normal flow control
                presentation_timestamp=global_timestamp,  # Use global timestamp
                source_width=frame_data.width,
                source_height=frame_data.height,
                playlist_item_index=self._current_item_index,
                is_first_frame_of_item=self._is_first_frame_of_current_item,
            )

            if not buffer_info:
                # Check if renderer is paused - if so, timeout is expected (back-pressure working)
                try:
                    from ..core.control_state import RendererState

                    status = self._control_state.get_status()
                    is_paused = status and status.renderer_state == RendererState.PAUSED
                except Exception:
                    is_paused = False

                if is_paused:
                    # During pause, write buffer timeout is expected - back-pressure is working correctly
                    logger.debug("Write buffer timeout during PAUSED state (back-pressure working correctly)")
                else:
                    # This indicates a more serious issue if we timeout after 2 seconds when not paused
                    logger.error("Unable to get write buffer after 2s timeout - consumer may be blocked")
                return False

            # Use the circular buffer's frame ID to ensure perfect synchronization
            timing_data = FrameTimingData(
                frame_index=buffer_info.frame_id,  # Matches the actual frame position in circular buffer
                plugin_timestamp=local_timestamp,  # 0-based timestamp from content plugin
                producer_timestamp=global_timestamp,  # Global presentation timestamp
                item_duration=self._current_item.get_effective_duration() if self._current_item else 0.0,
            )

            # Apply duration clamping for robustness
            if frame_data.duration is not None:
                # Ensure duration is at least one frame interval longer than the last frame timestamp
                min_duration = local_timestamp + self._frame_interval
                if frame_data.duration < min_duration:
                    logger.debug(f"Clamping item duration from {frame_data.duration:.3f}s to {min_duration:.3f}s")
                    # Update the current item's effective duration for future items
                    if self._current_item:
                        self._current_item.duration_override = min_duration

                # Track the last frame duration for accumulation when item completes
                self._last_frame_duration = max(frame_data.duration, min_duration)

                # Update timing data with clamped duration
                timing_data.item_duration = max(frame_data.duration, min_duration)

            # Attach timing data to buffer metadata
            if buffer_info.metadata:
                buffer_info.metadata.timing_data = timing_data

            # Get buffer array in planar format (3, H, W)
            buffer_array = buffer_info.get_array(FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS)

            # Scale and copy frame data to buffer (both in planar format)
            self._copy_frame_to_buffer(frame_data, buffer_array)

            # Mark write-to-buffer time after data is copied
            timing_data.mark_write_to_buffer()

            # Update timing data in shared memory metadata
            buffer_idx = timing_data.frame_index % self._frame_buffer.buffer_count
            self._update_timing_data_in_shared_memory(buffer_info, timing_data)

            # Update transition metadata in shared memory
            self._update_transition_metadata_in_shared_memory(buffer_info, local_timestamp, timing_data.frame_index)

            # Check if frame has non-zero content for logging
            if buffer_array.max() > 0:
                self._frames_with_content += 1

            # Advance write buffer
            if not self._frame_buffer.advance_write():
                logger.warning("Failed to advance write buffer")
                return False

            # Mark that we've written the first frame of this item
            if self._is_first_frame_of_current_item:
                self._is_first_frame_of_current_item = False

            return True

        except Exception as e:
            logger.error(f"Failed to render frame to buffer: {e}")
            return False

    def _copy_frame_to_buffer(self, frame_data: FrameData, buffer_array) -> None:
        """
        Copy and scale frame data to buffer array.

        Args:
            frame_data: Source frame data
            buffer_array: Target buffer array
        """
        try:
            # Handle different frame sizes
            src_h, src_w = frame_data.height, frame_data.width
            dst_h, dst_w = FRAME_HEIGHT, FRAME_WIDTH

            if src_h == dst_h and src_w == dst_w:
                # Direct copy
                if frame_data.channels == FRAME_CHANNELS:
                    buffer_array[:] = frame_data.array
                else:
                    # Convert channels
                    if frame_data.channels == 3 and FRAME_CHANNELS == 4:
                        # RGB to RGBA
                        buffer_array[:, :, :3] = frame_data.array
                        buffer_array[:, :, 3] = 255  # Alpha
                    elif frame_data.channels == 4 and FRAME_CHANNELS == 3:
                        # RGBA to RGB
                        buffer_array[:] = frame_data.array[:, :, :3]
            else:
                # Scale frame using planar-aware scaling
                self._scale_frame_to_buffer_planar(frame_data, buffer_array)

        except Exception as e:
            logger.error(f"Failed to copy frame to buffer: {e}")

    def _scale_frame_to_buffer(self, frame_data: FrameData, buffer_array) -> None:
        """
        Scale frame data to fit buffer with 5:3 aspect ratio cropping.

        Crops the source image to 5:3 aspect ratio (center crop) and scales
        to fill the buffer completely with no padding.

        Args:
            frame_data: Source frame data
            buffer_array: Target buffer array
        """
        try:
            import cv2

            src_h, src_w = frame_data.height, frame_data.width
            target_aspect = FRAME_WIDTH / FRAME_HEIGHT  # 5:3 = 1.667
            source_aspect = src_w / src_h

            # Calculate crop dimensions to achieve 5:3 aspect ratio
            if source_aspect > target_aspect:
                # Source is wider than 5:3, crop width
                crop_h = src_h
                crop_w = int(src_h * target_aspect)
                crop_x = (src_w - crop_w) // 2
                crop_y = 0
            else:
                # Source is taller than 5:3, crop height
                crop_w = src_w
                crop_h = int(src_w / target_aspect)
                crop_x = 0
                crop_y = (src_h - crop_h) // 2

            # Extract cropped region
            cropped = frame_data.array[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

            # Scale to target buffer dimensions
            resized = cv2.resize(
                cropped,
                (FRAME_WIDTH, FRAME_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )

            # Copy to buffer with channel conversion if needed
            if frame_data.channels == FRAME_CHANNELS:
                buffer_array[:] = resized
            elif frame_data.channels == 3 and FRAME_CHANNELS == 4:
                buffer_array[:, :, :3] = resized
                buffer_array[:, :, 3] = 255
            elif frame_data.channels == 4 and FRAME_CHANNELS == 3:
                buffer_array[:] = resized[:, :, :3]

        except ImportError:
            # Fallback: simple center crop without scaling
            logger.warning("OpenCV not available, using simple crop")
            self._simple_crop_frame(frame_data, buffer_array)
        except Exception as e:
            logger.error(f"Failed to scale frame: {e}")

    def _simple_crop_frame(self, frame_data: FrameData, buffer_array) -> None:
        """
        Simple center crop to 5:3 aspect ratio without scaling.

        This fallback method crops to 5:3 aspect ratio but doesn't scale,
        so the result may not fill the entire buffer if dimensions don't match.

        Args:
            frame_data: Source frame data
            buffer_array: Target buffer array
        """
        try:
            src_h, src_w = frame_data.height, frame_data.width
            dst_h, dst_w = FRAME_HEIGHT, FRAME_WIDTH
            target_aspect = dst_w / dst_h  # 5:3
            source_aspect = src_w / src_h

            # Calculate crop dimensions to achieve 5:3 aspect ratio
            if source_aspect > target_aspect:
                # Source is wider than 5:3, crop width
                crop_h = src_h
                crop_w = int(src_h * target_aspect)
                crop_x = (src_w - crop_w) // 2
                crop_y = 0
            else:
                # Source is taller than 5:3, crop height
                crop_w = src_w
                crop_h = int(src_w / target_aspect)
                crop_x = 0
                crop_y = (src_h - crop_h) // 2

            # Calculate how much of the cropped region fits in the buffer
            copy_h = min(crop_h, dst_h)
            copy_w = min(crop_w, dst_w)

            # Center the copied region in the buffer
            dst_y = (dst_h - copy_h) // 2
            dst_x = (dst_w - copy_w) // 2

            # Clear buffer first
            buffer_array.fill(0)

            # Copy the cropped region
            if frame_data.channels == FRAME_CHANNELS:
                buffer_array[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = frame_data.array[
                    crop_y : crop_y + copy_h, crop_x : crop_x + copy_w
                ]
            elif frame_data.channels == 3 and FRAME_CHANNELS == 4:
                buffer_array[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w, :3] = frame_data.array[
                    crop_y : crop_y + copy_h, crop_x : crop_x + copy_w
                ]
                buffer_array[:, :, 3] = 255
            elif frame_data.channels == 4 and FRAME_CHANNELS == 3:
                buffer_array[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = frame_data.array[
                    crop_y : crop_y + copy_h, crop_x : crop_x + copy_w, :3
                ]

        except Exception as e:
            logger.error(f"Failed to crop frame: {e}")

    def _scale_frame_to_buffer_planar(self, frame_data: FrameData, buffer_array) -> None:
        """
        Scale frame data to fit buffer with 5:3 aspect ratio cropping (planar format).

        Crops the source image to 5:3 aspect ratio (center crop) and scales
        to fill the buffer completely with no padding.

        Args:
            frame_data: Source frame data in planar format (3, H, W)
            buffer_array: Target buffer array in planar format (3, H, W)
        """
        try:
            import cv2

            src_h, src_w = frame_data.height, frame_data.width
            target_aspect = FRAME_WIDTH / FRAME_HEIGHT  # 5:3 = 1.667
            source_aspect = src_w / src_h

            # Calculate crop dimensions to achieve 5:3 aspect ratio
            if source_aspect > target_aspect:
                # Source is wider than 5:3, crop width
                crop_h = src_h
                crop_w = int(src_h * target_aspect)
                crop_x = (src_w - crop_w) // 2
                crop_y = 0
            else:
                # Source is taller than 5:3, crop height
                crop_w = src_w
                crop_h = int(src_w / target_aspect)
                crop_x = 0
                crop_y = (src_h - crop_h) // 2

            # Process each channel separately in planar format
            for c in range(frame_data.channels):
                # Extract cropped region for this channel
                cropped_channel = frame_data.array[c, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

                # Scale to target buffer dimensions
                resized_channel = cv2.resize(
                    cropped_channel,
                    (FRAME_WIDTH, FRAME_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )

                # Copy to buffer
                if c < FRAME_CHANNELS:
                    buffer_array[c] = resized_channel

            # Handle channel conversion if needed
            if frame_data.channels == 3 and FRAME_CHANNELS == 4:
                buffer_array[3] = 255  # Alpha channel

        except ImportError:
            # Fallback: simple center crop without scaling
            logger.warning("OpenCV not available, using simple crop")
            self._simple_crop_frame_planar(frame_data, buffer_array)
        except Exception as e:
            logger.error(f"Failed to scale frame: {e}")

    def _simple_crop_frame_planar(self, frame_data: FrameData, buffer_array) -> None:
        """
        Simple center crop to 5:3 aspect ratio without scaling (planar format).

        This fallback method crops to 5:3 aspect ratio but doesn't scale,
        so the result may not fill the entire buffer if dimensions don't match.

        Args:
            frame_data: Source frame data in planar format (3, H, W)
            buffer_array: Target buffer array in planar format (3, H, W)
        """
        try:
            src_h, src_w = frame_data.height, frame_data.width
            dst_h, dst_w = FRAME_HEIGHT, FRAME_WIDTH
            target_aspect = dst_w / dst_h  # 5:3
            source_aspect = src_w / src_h

            # Calculate crop dimensions to achieve 5:3 aspect ratio
            if source_aspect > target_aspect:
                # Source is wider than 5:3, crop width
                crop_h = src_h
                crop_w = int(src_h * target_aspect)
                crop_x = (src_w - crop_w) // 2
                crop_y = 0
            else:
                # Source is taller than 5:3, crop height
                crop_w = src_w
                crop_h = int(src_w / target_aspect)
                crop_x = 0
                crop_y = (src_h - crop_h) // 2

            # Calculate how much of the cropped region fits in the buffer
            copy_h = min(crop_h, dst_h)
            copy_w = min(crop_w, dst_w)

            # Center the copied region in the buffer
            dst_y = (dst_h - copy_h) // 2
            dst_x = (dst_w - copy_w) // 2

            # Clear buffer first
            buffer_array.fill(0)

            # Copy the cropped region for each channel in planar format
            for c in range(min(frame_data.channels, FRAME_CHANNELS)):
                buffer_array[c, dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = frame_data.array[
                    c, crop_y : crop_y + copy_h, crop_x : crop_x + copy_w
                ]

            # Handle channel conversion if needed
            if frame_data.channels == 3 and FRAME_CHANNELS == 4:
                buffer_array[3] = 255  # Alpha channel

        except Exception as e:
            logger.error(f"Failed to crop frame: {e}")

    def _update_statistics(self) -> None:
        """Update producer statistics."""
        try:
            current_time = time.time()

            # Rate limit statistics updates to reduce control state contention
            # Only update every 0.5 seconds (2 Hz) instead of every frame
            if not hasattr(self, "_last_stats_update_time"):
                self._last_stats_update_time = 0.0

            if current_time - self._last_stats_update_time < 0.5:
                return  # Skip this update

            self._last_stats_update_time = current_time
            elapsed = current_time - self._start_time

            if elapsed > 0:
                producer_fps = self._frames_produced / elapsed

                # Update control state with frame rate
                self._control_state.set_frame_rates(producer_fps, 0.0)  # Consumer FPS unknown

        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")

    def _should_drop_frame_for_downbeat_transition(self, local_timestamp: float) -> bool:
        """
        Check if frame should be dropped to time playlist transition to downbeat.

        When enabled, this feature drops frames near the end of a content item
        that would render after the last downbeat, ensuring the next item starts
        on a downbeat for musical synchronization.

        The algorithm:
        1. Get timing data from control state (last_downbeat_time, BPM, wallclock_delta)
        2. Calculate when this frame will be rendered in wall-clock time
        3. Calculate the last downbeat that will occur during this content item
        4. If frame renders after that last downbeat, drop it

        Args:
            local_timestamp: Frame's presentation timestamp within the current item

        Returns:
            True if frame should be dropped (past last downbeat), False otherwise
        """
        try:
            # Check if feature is enabled via control state
            status = self._control_state.get_status()
            if not status:
                return False

            if not status.transition_on_downbeat_enabled:
                return False

            # Get timing data from control state
            wallclock_delta = status.wallclock_delta
            current_bpm = status.current_bpm
            last_downbeat_time = status.last_downbeat_time

            # Validate timing data is available
            if wallclock_delta is None:
                # wallclock_delta not yet established by renderer
                return False

            if current_bpm <= 0:
                # No valid BPM detected
                return False

            if last_downbeat_time <= 0:
                # No downbeat detected yet
                return False

            # Get current item duration
            if not self._current_item:
                return False

            item_duration = self._current_item.get_effective_duration()
            if item_duration <= 0:
                return False

            # Calculate when this frame will be rendered in wall-clock time
            global_timestamp = self._current_item_global_offset + local_timestamp
            frame_render_time = global_timestamp + wallclock_delta

            # Calculate item start and end in wall-clock time
            item_start_realtime = self._current_item_global_offset + wallclock_delta
            item_end_realtime = item_start_realtime + item_duration

            # Calculate bar interval (4 beats per bar, assuming 4/4 time)
            bar_interval = (60.0 / current_bpm) * 4

            # Project forward from last_downbeat_time to find the last downbeat
            # that occurs BEFORE item_end_realtime
            # Start from last known downbeat and step forward
            current_downbeat = last_downbeat_time

            # First, if current_downbeat is before item start, step forward to first downbeat in item
            while current_downbeat < item_start_realtime:
                current_downbeat += bar_interval

            # Now step forward to find the last downbeat before item end
            last_downbeat_in_item = None
            while current_downbeat < item_end_realtime:
                last_downbeat_in_item = current_downbeat
                current_downbeat += bar_interval

            # If no downbeat found in item (item shorter than one bar), don't drop
            if last_downbeat_in_item is None:
                return False

            # Check if this frame renders after the last downbeat
            if frame_render_time > last_downbeat_in_item:
                # Calculate how much time is left in the item
                time_remaining = item_end_realtime - frame_render_time
                time_after_downbeat = frame_render_time - last_downbeat_in_item

                # Log the decision (first time per item)
                if not hasattr(self, "_logged_downbeat_drop") or not self._logged_downbeat_drop:
                    logger.info(
                        f"🎵 Dropping frame for downbeat transition: "
                        f"frame_time={frame_render_time:.3f}s, "
                        f"last_downbeat={last_downbeat_in_item:.3f}s, "
                        f"time_after_downbeat={time_after_downbeat*1000:.0f}ms, "
                        f"time_remaining={time_remaining*1000:.0f}ms, "
                        f"BPM={current_bpm:.1f}"
                    )
                    self._logged_downbeat_drop = True

                return True

            return False

        except Exception as e:
            logger.error(f"Error checking downbeat transition: {e}")
            return False

    def get_producer_stats(self) -> Dict[str, Any]:
        """
        Get producer statistics.

        Returns:
            Dictionary with producer statistics
        """
        current_time = time.time()
        elapsed = current_time - self._start_time if self._start_time > 0 else 0

        stats = {
            "running": self._running,
            "frames_produced": self._frames_produced,
            "elapsed_time": elapsed,
            "producer_fps": self._frames_produced / elapsed if elapsed > 0 else 0.0,
            "target_fps": self._target_fps,
            "playlist_info": self._playlist.get_playlist_info(),
            "current_content": (self._current_item.filepath if self._current_item else None),
        }

        # Add buffer stats
        buffer_stats = self._frame_buffer.get_producer_stats()
        stats.update(buffer_stats)

        return stats

    # External playlist manipulation methods
    def remove_playlist_item(self, index: int) -> bool:
        """
        Remove item from playlist by index.

        Args:
            index: Index of item to remove

        Returns:
            True if removed successfully, False otherwise
        """
        return self._playlist.remove_item(index)

    def clear_playlist(self) -> None:
        """Clear all items from playlist."""
        self._playlist.clear()

    def set_playlist_loop(self, loop: bool) -> None:
        """
        Set playlist loop mode.

        Args:
            loop: True to enable looping, False to disable
        """
        self._playlist.set_loop(loop)

    def get_playlist_info(self) -> Dict[str, Any]:
        """
        Get current playlist information.

        Returns:
            Dictionary with playlist details
        """
        return self._playlist.get_playlist_info()

    def get_playlist_length(self) -> int:
        """
        Get number of items in playlist.

        Returns:
            Number of playlist items
        """
        return len(self._playlist)

    def _on_playlist_sync_update(self, sync_state: SyncPlaylistState) -> None:
        """Handle playlist updates from synchronization service."""
        try:
            old_index = self._playlist.get_current_index() if hasattr(self, "_playlist") else None
            playlist_len = len(self._playlist) if hasattr(self, "_playlist") else 0

            # Only invalidate pre-prepared content if this is NOT a natural advance to next item.
            # Natural advance: old_index + 1 == new_index (or wrap from last to 0)
            # Skip/reorder/etc: anything else should invalidate
            if self._content_preparer and old_index is not None:
                is_natural_advance = sync_state.current_index == old_index + 1 or (
                    old_index == playlist_len - 1 and sync_state.current_index == 0
                )
                if not is_natural_advance:
                    logger.debug(
                        f"Non-sequential playlist change ({old_index} -> {sync_state.current_index}), "
                        "invalidating pre-prepared content"
                    )
                    self._content_preparer.invalidate()
            logger.info(
                f"Received playlist sync update: {len(sync_state.items)} items, "
                f"index change: {old_index} -> {sync_state.current_index}, "
                f"playing={sync_state.is_playing}, loaded_index={self._current_item_index}"
            )

            # Use batch update to perform all playlist changes atomically
            with self._playlist.batch_update():
                # Clear current playlist
                self._playlist.clear()

                # Convert sync playlist items to producer playlist items
                for sync_item in sync_state.items:
                    if sync_item.type in ["image", "video"] and sync_item.file_path:
                        # Add regular media files with transition configurations
                        self._playlist.add_item(
                            filepath=sync_item.file_path,
                            duration=sync_item.duration,
                            repeat=1,
                            transition_in=sync_item.transition_in,
                            transition_out=sync_item.transition_out,
                        )
                    elif sync_item.type == "text" and sync_item.file_path:
                        # Handle text content (file_path contains JSON config) with transitions
                        self._playlist.add_item(
                            filepath=sync_item.file_path,
                            duration=sync_item.duration,
                            repeat=1,
                            transition_in=sync_item.transition_in,
                            transition_out=sync_item.transition_out,
                        )
                    elif sync_item.type == "effect" and sync_item.effect_config:
                        # Handle effect content - convert effect_config to JSON string for playlist
                        import json

                        # Add item-level duration to effect config
                        effect_config_with_duration = sync_item.effect_config.copy()
                        effect_config_with_duration["duration"] = sync_item.duration

                        effect_json = json.dumps(effect_config_with_duration)
                        self._playlist.add_item(
                            filepath=effect_json,
                            duration=sync_item.duration,
                            repeat=1,
                            transition_in=sync_item.transition_in,
                            transition_out=sync_item.transition_out,
                        )

                # Update current index (still within batch_update context)
                self._playlist.set_current_index(sync_state.current_index)

            # All playlist operations completed atomically

            # Check if the item at current index has changed (even if index number is same)
            # Use lock to prevent race condition with _ensure_current_content
            with self._sync_lock:
                # Clear the waiting flag now that we've received the sync response
                if self._waiting_for_sync_response:
                    logger.debug("Sync response received, clearing wait flag")
                    self._waiting_for_sync_response = False

                needs_reload = False
                if self._current_item_index != sync_state.current_index:
                    # Index changed
                    logger.info(
                        f"PLAYLIST INDEX SYNC: Producer index {self._current_item_index} -> playlist index {sync_state.current_index}"
                    )
                    needs_reload = True
                elif self._current_item_index >= 0 and self._current_item:
                    # Index is same, but check if the item itself changed (e.g., rename, replace)
                    try:
                        current_playlist_item = self._playlist.get_item(sync_state.current_index)
                        if current_playlist_item and current_playlist_item.filepath != self._current_item.filepath:
                            logger.info(
                                f"PLAYLIST ITEM CHANGED at index {sync_state.current_index}: "
                                f"{self._current_item.filepath} -> {current_playlist_item.filepath}"
                            )
                            needs_reload = True
                    except Exception as e:
                        logger.debug(f"Could not check if item changed: {e}")

                if needs_reload:
                    # Clear stuck state tracking since we're changing content
                    if hasattr(self, "_content_finished_time"):
                        delattr(self, "_content_finished_time")

                    # Mark that content needs to be reloaded
                    # Also cleanup current source to cancel any in-progress setup
                    if self._current_source:
                        logger.info("Cleaning up current content source due to playlist change")
                        try:
                            self._current_source.cleanup()
                        except Exception as e:
                            logger.warning(f"Error cleaning up content source: {e}")
                        self._current_source = None

                    self._current_item = None  # Force content reload in _ensure_current_content
                    logger.debug("Set _current_item = None to force content reload (sync callback thread)")
                else:
                    # Staying on the same item, update the global timestamp offset
                    # to account for new items added after the current position
                    self._update_global_timestamp_offset()
                    logger.debug("Updated global timestamp offset after playlist sync (same item)")

            # Update producer state based on sync state
            if sync_state.is_playing:
                self._control_state.set_producer_state(ProducerState.PLAYING)
            else:
                self._control_state.set_producer_state(ProducerState.STOPPED)

            # Playlist state now managed through sync service only

            logger.info(
                f"Producer playlist synchronized: {len(self._playlist._items)} items, current_index={self._playlist._current_index}, playing={sync_state.is_playing}"
            )

        except Exception as e:
            logger.error(f"Error handling playlist sync update: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def cleanup(self) -> None:
        """Clean up producer resources."""
        try:
            # Stop if running
            self.stop()

            # Clean up content preparer
            if self._content_preparer:
                self._content_preparer.cleanup()
                self._content_preparer = None

            # Clean up playlist
            self._playlist.clear()

            # Disconnect from playlist sync service
            if self._playlist_sync_client:
                self._playlist_sync_client.disconnect()
                logger.info("Producer disconnected from playlist synchronization service")

            # Clean up shared resources
            self._frame_buffer.cleanup()
            self._control_state.cleanup()

            logger.info("Producer process cleaned up")

        except Exception as e:
            logger.error(f"Error during producer cleanup: {e}")


# Control interface functions
def play() -> bool:
    """Start producer."""
    try:
        control = ControlState()
        if control.connect():
            return control.set_producer_state(ProducerState.PLAYING)
        return False
    except Exception:
        return False


def stop() -> bool:
    """Stop producer."""
    try:
        control = ControlState()
        if control.connect():
            return control.set_producer_state(ProducerState.STOPPED)
        return False
    except Exception:
        return False
