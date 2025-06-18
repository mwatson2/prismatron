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
from ..core.shared_buffer import FrameProducer
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
    ):
        """
        Initialize playlist item.

        Args:
            filepath: Path to content file
            duration: Override duration (None = use file duration)
            repeat: Number of times to repeat (1 = play once)
            metadata: Additional metadata
        """
        self.filepath = filepath
        self.duration_override = duration
        self.repeat_count = repeat
        self.metadata = metadata or {}
        self.current_repeat = 0

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
            self._detected_type = ContentSourceRegistry.detect_content_type(
                self.filepath
            )

            # Create content source
            self._content_source = ContentSourceRegistry.create_source(
                self.filepath, self._detected_type
            )

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

    def add_item(
        self,
        filepath: str,
        duration: Optional[float] = None,
        repeat: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add item to playlist.

        Args:
            filepath: Path to content file
            duration: Override duration
            repeat: Repeat count
            metadata: Additional metadata

        Returns:
            True if added successfully, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Content file not found: {filepath}")
                return False

            item = PlaylistItem(filepath, duration, repeat, metadata)
            self._items.append(item)
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

    def clear(self) -> None:
        """Clear all items from playlist."""
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
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index]
        return None

    def advance_to_next(self) -> bool:
        """
        Advance to next item in playlist.

        Returns:
            True if advanced to next item, False if end of playlist
        """
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
        self._loop_playlist = loop

    def get_playlist_info(self) -> Dict[str, Any]:
        """
        Get playlist information.

        Returns:
            Dictionary with playlist details
        """
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
                    "type": item._detected_type.value
                    if item._detected_type
                    else "unknown",
                }
                for item in self._items
            ],
        }

    def __len__(self) -> int:
        """Get number of items in playlist."""
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

        # Current content and state
        self._current_source: Optional[ContentSource] = None
        self._current_item: Optional[PlaylistItem] = None

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

            # Initialize control state
            if not self._control_state.initialize():
                logger.error("Failed to initialize control state")
                return False

            # Set initial system state
            self._control_state.update_system_state(SystemState.RUNNING)
            self._control_state.set_play_state(PlayState.STOPPED)

            logger.info("Producer process initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Producer initialization failed: {e}")
            return False

    def add_content(
        self, filepath: str, duration: Optional[float] = None, repeat: int = 1
    ) -> bool:
        """
        Add content to playlist.

        Args:
            filepath: Path to content file
            duration: Override duration
            repeat: Repeat count

        Returns:
            True if added successfully, False otherwise
        """
        return self._playlist.add_item(filepath, duration, repeat)

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

            # Scan directory
            for file_path in directory_path.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in supported_extensions
                ):
                    if self._playlist.add_item(str(file_path)):
                        added_count += 1

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
            self._producer_thread = threading.Thread(
                target=self._producer_main_loop, daemon=True
            )

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

                # Handle play state
                if status.play_state == PlayState.PLAYING:
                    self._handle_playing_state()
                elif status.play_state == PlayState.PAUSED:
                    self._handle_paused_state()
                else:
                    self._handle_stopped_state()

                # Update frame rate statistics
                self._update_statistics()

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
                self._control_state.set_play_state(PlayState.STOPPED)
                return

            # Check frame timing
            current_time = time.time()
            if current_time - self._last_frame_time < self._frame_interval:
                return  # Not time for next frame yet

            # Get next frame from content source
            frame_data = self._current_source.get_next_frame()

            if frame_data is not None:
                # Render frame to shared memory
                if self._render_frame_to_buffer(frame_data):
                    self._frames_produced += 1
                    self._last_frame_time = current_time

            elif self._current_source.is_finished():
                # Content finished, advance to next
                self._advance_to_next_content()

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

        time.sleep(0.1)

    def _ensure_current_content(self) -> bool:
        """
        Ensure we have current content loaded.

        Returns:
            True if content is available, False otherwise
        """
        if (
            self._current_source
            and self._current_source.status == ContentStatus.PLAYING
        ):
            return True

        # Try to load current playlist item
        current_item = self._playlist.get_current_item()
        if not current_item:
            logger.debug("No current playlist item")
            return False

        # Load content source if needed
        if self._current_item != current_item:
            # Clean up previous content
            if self._current_source:
                self._current_source.cleanup()

            # Load new content
            self._current_source = current_item.get_content_source()
            self._current_item = current_item

            if not self._current_source:
                logger.error(
                    f"Failed to create content source: {current_item.filepath}"
                )
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

            logger.info(
                f"Loaded content: {current_item.filepath} (fps: {self._target_fps})"
            )

        return True

    def _advance_to_next_content(self) -> None:
        """Advance to next content in playlist."""
        try:
            if self._playlist.advance_to_next():
                # Next content available
                logger.debug("Advanced to next content")
            else:
                # End of playlist
                logger.info("End of playlist reached")
                self._control_state.set_play_state(PlayState.STOPPED)

        except Exception as e:
            logger.error(f"Failed to advance to next content: {e}")

    def _render_frame_to_buffer(self, frame_data: FrameData) -> bool:
        """
        Render frame data to shared memory buffer.

        Args:
            frame_data: Frame data to render

        Returns:
            True if rendered successfully, False otherwise
        """
        try:
            # Get write buffer
            buffer_info = self._frame_buffer.get_write_buffer(
                timeout=0.1,  # Short timeout to avoid blocking
                presentation_timestamp=frame_data.presentation_timestamp,
                source_width=frame_data.width,
                source_height=frame_data.height,
            )

            if not buffer_info:
                logger.warning("Failed to get write buffer")
                return False

            # Get buffer array
            buffer_array = buffer_info.get_array(
                FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS
            )

            # Scale and copy frame data to buffer
            self._copy_frame_to_buffer(frame_data, buffer_array)

            # Advance write buffer
            if not self._frame_buffer.advance_write():
                logger.warning("Failed to advance write buffer")
                return False

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
                # Scale frame (simplified - center crop/pad)
                self._scale_frame_to_buffer(frame_data, buffer_array)

        except Exception as e:
            logger.error(f"Failed to copy frame to buffer: {e}")

    def _scale_frame_to_buffer(self, frame_data: FrameData, buffer_array) -> None:
        """
        Scale frame data to fit buffer with center crop/pad.

        Args:
            frame_data: Source frame data
            buffer_array: Target buffer array
        """
        try:
            import cv2

            # Resize frame to buffer dimensions
            if frame_data.channels == 3:
                resized = cv2.resize(
                    frame_data.array,
                    (FRAME_WIDTH, FRAME_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                resized = cv2.resize(
                    frame_data.array,
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
            # Fallback: simple center crop/pad without scaling
            logger.warning("OpenCV not available, using simple crop/pad")
            self._simple_crop_pad_frame(frame_data, buffer_array)
        except Exception as e:
            logger.error(f"Failed to scale frame: {e}")

    def _simple_crop_pad_frame(self, frame_data: FrameData, buffer_array) -> None:
        """
        Simple center crop/pad without scaling.

        Args:
            frame_data: Source frame data
            buffer_array: Target buffer array
        """
        try:
            src_h, src_w = frame_data.height, frame_data.width
            dst_h, dst_w = FRAME_HEIGHT, FRAME_WIDTH

            # Calculate copy region
            copy_h = min(src_h, dst_h)
            copy_w = min(src_w, dst_w)

            # Calculate offsets for centering
            src_y = (src_h - copy_h) // 2
            src_x = (src_w - copy_w) // 2
            dst_y = (dst_h - copy_h) // 2
            dst_x = (dst_w - copy_w) // 2

            # Clear buffer first
            buffer_array.fill(0)

            # Copy overlapping region
            if frame_data.channels == FRAME_CHANNELS:
                buffer_array[
                    dst_y : dst_y + copy_h, dst_x : dst_x + copy_w
                ] = frame_data.array[src_y : src_y + copy_h, src_x : src_x + copy_w]
            elif frame_data.channels == 3 and FRAME_CHANNELS == 4:
                buffer_array[
                    dst_y : dst_y + copy_h, dst_x : dst_x + copy_w, :3
                ] = frame_data.array[src_y : src_y + copy_h, src_x : src_x + copy_w]
                buffer_array[:, :, 3] = 255
            elif frame_data.channels == 4 and FRAME_CHANNELS == 3:
                buffer_array[
                    dst_y : dst_y + copy_h, dst_x : dst_x + copy_w
                ] = frame_data.array[src_y : src_y + copy_h, src_x : src_x + copy_w, :3]

        except Exception as e:
            logger.error(f"Failed to crop/pad frame: {e}")

    def _update_statistics(self) -> None:
        """Update producer statistics."""
        try:
            current_time = time.time()
            elapsed = current_time - self._start_time

            if elapsed > 0:
                producer_fps = self._frames_produced / elapsed

                # Update control state with frame rate
                self._control_state.set_frame_rates(
                    producer_fps, 0.0
                )  # Consumer FPS unknown

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
            "current_content": self._current_item.filepath
            if self._current_item
            else None,
        }

        # Add buffer stats
        buffer_stats = self._frame_buffer.get_producer_stats()
        stats.update(buffer_stats)

        return stats

    def cleanup(self) -> None:
        """Clean up producer resources."""
        try:
            # Stop if running
            self.stop()

            # Clean up playlist
            self._playlist.clear()

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
