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
        with self._lock:
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

        # Periodic logging for pipeline debugging
        self._last_log_time = 0.0
        self._log_interval = 2.0  # Log every 2 seconds
        self._frames_with_content = 0  # Frames with non-zero content

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
        result = self._playlist.add_item(filepath, duration, repeat)
        if result:
            self._sync_playlist_to_control_state()
        return result

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
                    and self._playlist.add_item(str(file_path))
                ):
                    added_count += 1

            logger.info(f"Loaded {added_count} files from {directory}")
            if added_count > 0:
                self._sync_playlist_to_control_state()
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

                # Process playlist commands from web server
                self._process_playlist_commands()

                # Log play state changes for debugging
                if not hasattr(self, "_last_play_state") or self._last_play_state != status.play_state:
                    logger.info(f"PRODUCER CONTROL STATE: Play state changed to {status.play_state}")
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

            # Check frame timing
            current_time = time.time()
            if current_time - self._last_frame_time < self._frame_interval:
                return  # Not time for next frame yet

            # Get next frame from content source
            frame_data = self._current_source.get_next_frame()

            if frame_data is not None:
                # Log first frame for debugging
                if self._frames_produced == 0:
                    logger.info(f"PRODUCER FRAMES: Got first frame from {self._current_item.filepath}")

                # Render frame to shared memory
                if self._render_frame_to_buffer(frame_data):
                    self._frames_produced += 1
                    self._last_frame_time = current_time

            elif self._current_source.is_finished():
                # Content finished, advance to next
                logger.info(f"PRODUCER FRAMES: Content finished for {self._current_item.filepath}")
                self._advance_to_next_content()
            else:
                # No frame available yet (but not finished)
                if not hasattr(self, "_no_frame_logged") or not self._no_frame_logged:
                    logger.info(f"PRODUCER FRAMES: No frame available from {self._current_item.filepath} (waiting...)")
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
            logger.info("PRODUCER CONTENT: No current playlist item available")
            return False

        # Load content source if needed
        if self._current_item != current_item:
            # Clean up previous content
            if self._current_source:
                self._current_source.cleanup()

            # Load new content
            logger.info(f"PRODUCER CONTENT: Loading content source for {current_item.filepath}")
            self._current_source = current_item.get_content_source()
            self._current_item = current_item

            if not self._current_source:
                logger.error(f"Failed to create content source: {current_item.filepath}")
                return False

            # Setup content source
            logger.info(f"PRODUCER CONTENT: Setting up content source for {current_item.filepath}")
            if not self._current_source.setup():
                logger.error(f"Failed to setup content: {current_item.filepath}")
                return False

            logger.info(f"PRODUCER CONTENT: Successfully loaded {current_item.filepath}")

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
            if self._playlist.advance_to_next():
                # Next content available
                logger.debug("Advanced to next content")
                self._sync_playlist_to_control_state()
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

            # Get buffer array in planar format (3, H, W)
            buffer_array = buffer_info.get_array(FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS)

            # Scale and copy frame data to buffer (both in planar format)
            self._copy_frame_to_buffer(frame_data, buffer_array)

            # Check if frame has non-zero content for logging
            if buffer_array.max() > 0:
                self._frames_with_content += 1

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

    def _sync_playlist_to_control_state(self) -> None:
        """Synchronize playlist information to shared control state."""
        try:
            playlist_info = self.get_playlist_info()
            self._control_state.update_playlist_info(playlist_info)
        except Exception as e:
            logger.warning(f"Failed to sync playlist to control state: {e}")

    def _process_playlist_commands(self) -> None:
        """Process playlist commands from web server via control state."""
        try:
            command = self._control_state.get_playlist_command()
            if command:
                action = command.get("action")
                data = command.get("data", {})

                if action == "add_item":
                    filepath = data.get("filepath")
                    duration = data.get("duration")
                    if filepath:
                        success = self._playlist.add_item(filepath, duration)
                        if success:
                            logger.info(f"PRODUCER PLAYLIST: Added item from web server: {filepath}")
                            self._sync_playlist_to_control_state()
                        else:
                            logger.error(f"PRODUCER PLAYLIST: Failed to add item: {filepath}")

                elif action == "remove_item":
                    filepath = data.get("filepath")
                    if filepath:
                        success = self._playlist.remove_item_by_filepath(filepath)
                        if success:
                            logger.info(f"PRODUCER PLAYLIST: Removed item from web server: {filepath}")
                            self._sync_playlist_to_control_state()
                        else:
                            logger.error(f"PRODUCER PLAYLIST: Failed to remove item: {filepath}")

                elif action == "clear":
                    self._playlist.clear()
                    logger.info("PRODUCER PLAYLIST: Cleared playlist from web server")
                    self._sync_playlist_to_control_state()

                else:
                    logger.warning(f"PRODUCER PLAYLIST: Unknown command action: {action}")

        except Exception as e:
            logger.error(f"Error processing playlist commands: {e}")

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
