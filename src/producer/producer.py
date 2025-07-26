"""
Producer Process Core.

This module implements the main producer process that loads content,
manages playlists, and renders frames to shared memory buffers for
the consumer process.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..const import FRAME_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH
from ..core.control_state import ControlState, PlayState, SystemState
from ..core.playlist_sync import PlaylistState as SyncPlaylistState
from ..core.playlist_sync import PlaylistSyncClient, TransitionConfig
from ..core.shared_buffer import FrameProducer
from ..utils.frame_timing import FrameTimingData
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

                # For text content, filepath contains JSON config, not a file path
                if content_type == ContentType.TEXT:
                    # Validate JSON text configuration
                    import json

                    try:
                        config = json.loads(filepath)
                        if not ("text" in config and isinstance(config["text"], str)):
                            logger.error(f"Invalid text configuration: {filepath}")
                            return False
                        logger.debug(f"Adding text content: {config.get('text', '')[:50]}...")
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON text configuration: {e}")
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

        # Threading and timing
        self._producer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

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

        # Global timestamp mapping state
        self._playlist_start_time = 0.0  # When the playlist started playing
        self._item_start_times: List[float] = []  # Cumulative start times for each item
        self._current_item_global_offset = 0.0  # Global offset for current item
        self._accumulated_duration = 0.0  # Accumulated actual durations from completed items
        self._last_frame_duration: Optional[float] = None  # Duration from last frame for accumulation

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
            self._control_state.set_play_state(PlayState.STOPPED)

            # Connect to playlist synchronization service
            self._playlist_sync_client = PlaylistSyncClient(client_name="producer")
            self._playlist_sync_client.on_playlist_update = self._on_playlist_sync_update
            if self._playlist_sync_client.connect():
                logger.info("Producer connected to playlist synchronization service")
            else:
                logger.warning("Producer failed to connect to playlist synchronization service - using fallback")

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
        self._control_state.set_play_state(PlayState.STOPPED)

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

                # Log play state changes for debugging
                if not hasattr(self, "_last_play_state") or self._last_play_state != status.play_state:
                    logger.debug(f"Play state changed to {status.play_state}")
                    self._last_play_state = status.play_state

                # Handle play state
                if status.play_state == PlayState.PLAYING:
                    self._handle_playing_state()
                elif status.play_state == PlayState.PAUSED:
                    self._handle_paused_state()
                else:
                    self._handle_stopped_state()

                # Update frame rate statistics
                self._update_statistics()

                # Periodic logging for pipeline debugging (only when playing or if there's activity)
                current_time = time.time()
                if current_time - self._last_log_time >= self._log_interval:
                    # Only log if we're playing, or if we have frames produced, or every 10 seconds when idle
                    should_log = (
                        (status and status.play_state == PlayState.PLAYING)
                        or self._frames_produced > 0
                        or (current_time - self._last_log_time) >= 10.0
                    )

                    if should_log:
                        content_ratio = (self._frames_with_content / max(1, self._frames_produced)) * 100
                        play_state_str = f"[{status.play_state}]" if status else "[UNKNOWN]"
                        logger.info(
                            f"PRODUCER PIPELINE {play_state_str}: {self._frames_produced} frames produced, "
                            f"{self._frames_with_content} with content ({content_ratio:.1f}%), "
                            f"current: {self._current_item.filepath if self._current_item else 'None'}"
                        )
                        self._last_log_time = current_time

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
                # No content available, switch to stopped
                logger.warning("PRODUCER PLAYING: No current content available, switching to STOPPED")
                self._control_state.set_play_state(PlayState.STOPPED)
                return

            # Producer should output frames as fast as possible
            # Frame rate is determined by content timestamps, not throttling
            current_time = time.time()

            # Get next frame from content source
            frame_data = self._current_source.get_next_frame()

            if frame_data is not None:
                # Log first frame for debugging
                if self._frames_produced == 0:
                    logger.debug(f"Got first frame from {self._current_item.filepath}")

                # Render frame to shared memory
                if self._render_frame_to_buffer(frame_data):
                    self._frames_produced += 1
                    self._last_frame_time = current_time

            elif self._current_source.is_finished():
                # Content finished, accumulate actual duration and advance to next
                if self._last_frame_duration is not None:
                    self._accumulated_duration += self._last_frame_duration
                    logger.debug(
                        f"Item completed - accumulated {self._last_frame_duration:.3f}s, total now {self._accumulated_duration:.3f}s"
                    )
                    self._last_frame_duration = None  # Reset for next item

                current_item_name = os.path.basename(self._current_item.filepath) if self._current_item else "unknown"
                logger.info(f"Content finished: {current_item_name}")

                # Prevent duplicate next_item() calls for the same content
                if not self._content_finished_processed:
                    self._content_finished_processed = True
                    self._advance_to_next_content()
                else:
                    logger.debug(
                        f"Content '{current_item_name}' already processed for advancement, skipping duplicate call"
                    )
            else:
                # No frame available yet (but not finished)
                if not hasattr(self, "_no_frame_logged") or not self._no_frame_logged:
                    logger.debug(f"No frame available from {self._current_item.filepath} (waiting...)")
                    self._no_frame_logged = True

        except Exception as e:
            logger.error(f"Error in playing state: {e}")
            self._control_state.set_error(f"Playback error: {e}")

    def _handle_paused_state(self) -> None:
        """Handle paused state."""
        # Just wait, don't produce frames
        time.sleep(0.1)

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
        if self._current_source and self._current_source.status == ContentStatus.PLAYING:
            return True

        # Try to load current playlist item
        current_item = self._playlist.get_current_item()
        if not current_item:
            logger.debug("No current playlist item available")
            return False

        current_playlist_index = self._playlist.get_current_index()
        logger.debug(
            f"_ensure_current_content called - playlist_index={current_playlist_index}, loaded_index={self._current_item_index}"
        )

        # Load content source if needed
        if self._current_item != current_item:
            # Clean up previous content
            if self._current_source:
                self._current_source.cleanup()

            # Load new content
            logger.info(f"Loading content: {os.path.basename(current_item.filepath)}")
            self._current_source = current_item.get_content_source()
            self._current_item = current_item
            self._current_item_index = self._playlist.get_current_index()  # Store the index of the loaded content
            self._is_first_frame_of_current_item = True  # Reset flag for new item
            self._content_finished_processed = False  # Reset flag for new content

            # Log the index update for debugging
            logger.debug(
                f"Loaded content at index {self._current_item_index}: {os.path.basename(current_item.filepath)}"
            )

            # Calculate global timestamp offset for this item
            self._update_global_timestamp_offset()

            if not self._current_source:
                logger.error(f"Failed to create content source: {current_item.filepath}")
                return False

            # Setup content source
            if not self._current_source.setup():
                logger.error(f"Failed to setup content: {current_item.filepath}")
                return False

            # Update control state with current file
            self._control_state.set_current_file(current_item.filepath)

            # Adjust target FPS based on content
            if hasattr(self._current_source, "content_info"):
                content_fps = self._current_source.content_info.fps
                if content_fps > 0:
                    self._target_fps = min(content_fps, 60.0)  # Cap at 60 FPS
                    self._frame_interval = 1.0 / self._target_fps

            logger.info(f"Loaded content: {current_item.filepath} (fps: {self._target_fps})")

        return True

    def _advance_to_next_content(self) -> None:
        """Advance to next content in playlist."""
        try:
            # If connected to sync service, send next command instead of advancing locally
            if self._playlist_sync_client and self._playlist_sync_client.connected:
                logger.debug("Requesting next item from sync service")
                success = self._playlist_sync_client.next_item()
                if not success:
                    logger.warning("Failed to send next command to sync service")
                    # Note: Don't advance locally - let sync service handle it
            else:
                logger.warning("Playlist sync client not connected - cannot advance to next content")
                self._control_state.set_play_state(PlayState.STOPPED)

        except Exception as e:
            logger.error(f"Failed to advance to next content: {e}")

    def _update_global_timestamp_offset(self) -> None:
        """
        Update the global timestamp offset for the current item.

        Uses accumulated actual durations from completed items rather than
        playlist metadata to handle duration variations.
        """
        try:
            # Simply use the accumulated duration from completed items
            self._current_item_global_offset = self._accumulated_duration

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
                logger.debug(
                    f"Producer writing to buffer {buffer_idx}: frame_index={timing_data.frame_index} -> shared memory"
                )
                metadata_record["frame_index"] = timing_data.frame_index
                metadata_record["plugin_timestamp"] = timing_data.plugin_timestamp
                metadata_record["producer_timestamp"] = timing_data.producer_timestamp
                metadata_record["item_duration"] = timing_data.item_duration
                metadata_record["write_to_buffer_time"] = timing_data.write_to_buffer_time or 0.0
                logger.debug(
                    f"Producer wrote to shared memory buffer {buffer_idx}: frame_index={metadata_record['frame_index']}"
                )

        except Exception as e:
            logger.warning(f"Producer: Failed to update timing data in shared memory: {e}")

    def _update_transition_metadata_in_shared_memory(self, buffer_info, item_timestamp: float) -> None:
        """
        Update transition metadata fields in shared memory.

        Args:
            buffer_info: Buffer info object from frame buffer
            item_timestamp: Time within the current playlist item (seconds from item start)
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
                    metadata_record["transition_in_type"] = current_item.transition_in.type
                    metadata_record["transition_in_duration"] = current_item.transition_in.parameters.get(
                        "duration", 0.0
                    )

                    # Set transition_out configuration
                    metadata_record["transition_out_type"] = current_item.transition_out.type
                    metadata_record["transition_out_duration"] = current_item.transition_out.parameters.get(
                        "duration", 0.0
                    )

                    # Set item timestamp for transition calculations
                    metadata_record["item_timestamp"] = item_timestamp

                    logger.debug(
                        f"Producer wrote transition metadata to buffer {buffer_idx}: "
                        f"in={current_item.transition_in.type}({metadata_record['transition_in_duration']}s), "
                        f"out={current_item.transition_out.type}({metadata_record['transition_out_duration']}s), "
                        f"item_timestamp={item_timestamp:.3f}s"
                    )

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
                # This indicates a more serious issue if we timeout after 2 seconds
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
            logger.debug(
                f"Writing timing data to shared memory: frame_index={timing_data.frame_index}, calculated_buffer_idx={buffer_idx}"
            )
            self._update_timing_data_in_shared_memory(buffer_info, timing_data)

            # Update transition metadata in shared memory
            self._update_transition_metadata_in_shared_memory(buffer_info, local_timestamp)

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
            elapsed = current_time - self._start_time

            if elapsed > 0:
                producer_fps = self._frames_produced / elapsed

                # Update control state with frame rate
                self._control_state.set_frame_rates(producer_fps, 0.0)  # Consumer FPS unknown

        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")

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
            logger.debug(
                f"Received playlist update: {len(sync_state.items)} items, current_index={sync_state.current_index}"
            )

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
                    # Handle effect content (for future implementation)
                    logger.info(f"Effect content not yet supported: {sync_item.effect_config}")

            # Update current index and playback state
            self._playlist._current_index = sync_state.current_index

            # If the current content needs to change (different index), reset loaded content tracking
            if self._current_item_index != sync_state.current_index:
                logger.info(
                    f"PLAYLIST INDEX SYNC: Producer index {self._current_item_index} -> playlist index {sync_state.current_index}"
                )
                # Mark that content needs to be reloaded to sync with new index
                self._current_item = None  # Force content reload in _ensure_current_content

            # Update play state based on sync state
            if sync_state.is_playing:
                self._control_state.set_play_state(PlayState.PLAYING)
            else:
                self._control_state.set_play_state(PlayState.PAUSED)

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
    """Start playback."""
    try:
        control = ControlState()
        if control.connect():
            return control.set_play_state(PlayState.PLAYING)
        return False
    except Exception:
        return False


def pause() -> bool:
    """Pause playback."""
    try:
        control = ControlState()
        if control.connect():
            return control.set_play_state(PlayState.PAUSED)
        return False
    except Exception:
        return False


def stop() -> bool:
    """Stop playback."""
    try:
        control = ControlState()
        if control.connect():
            return control.set_play_state(PlayState.STOPPED)
        return False
    except Exception:
        return False
