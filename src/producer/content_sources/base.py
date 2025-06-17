"""
Base Content Source Plugin Architecture.

This module defines the abstract base class and common interfaces for
content source plugins that provide frame data to the producer process.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Data for a single frame from a content source."""

    array: np.ndarray  # Frame data as numpy array (H, W, C)
    width: int  # Actual frame width
    height: int  # Actual frame height
    channels: int  # Number of channels (typically 3 for RGB)
    presentation_timestamp: Optional[
        float
    ] = None  # When this frame should be displayed


class ContentType(Enum):
    """Content type enumeration."""

    IMAGE = "image"
    VIDEO = "video"
    ANIMATION = "animation"
    LIVE = "live"
    UNKNOWN = "unknown"


class ContentStatus(Enum):
    """Content source status enumeration."""

    UNINITIALIZED = "uninitialized"
    READY = "ready"
    PLAYING = "playing"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"


class ContentInfo:
    """Information about content source."""

    def __init__(self) -> None:
        self.content_type: ContentType = ContentType.UNKNOWN
        self.filepath: str = ""
        self.width: int = 0
        self.height: int = 0
        self.duration: float = 0.0
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.has_audio: bool = False
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content_type": self.content_type.value,
            "filepath": self.filepath,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "has_audio": self.has_audio,
            "metadata": self.metadata.copy(),
        }


class ContentSource(ABC):
    """
    Abstract base class for content source plugins.

    All content sources must implement this interface to provide
    frame data to the producer process.
    """

    def __init__(self, filepath: str):
        """
        Initialize content source.

        Args:
            filepath: Path to content file
        """
        self.filepath = filepath
        self.content_info = ContentInfo()
        self.content_info.filepath = filepath
        self.status = ContentStatus.UNINITIALIZED
        self.current_frame = 0
        self.current_time = 0.0
        self._error_message = ""

    @abstractmethod
    def setup(self) -> bool:
        """
        Initialize the content source and prepare for playback.

        Returns:
            True if setup successful, False otherwise
        """
        pass

    @abstractmethod
    def get_next_frame(self) -> Optional[FrameData]:
        """
        Get the next frame from the content source.

        Returns:
            FrameData object with frame information, or None if end/error
        """
        pass

    @abstractmethod
    def get_duration(self) -> float:
        """
        Get total duration of content in seconds.

        Returns:
            Duration in seconds, or 0.0 if unknown/infinite
        """
        pass

    @abstractmethod
    def seek(self, timestamp: float) -> bool:
        """
        Seek to specific timestamp in content.

        Args:
            timestamp: Target timestamp in seconds

        Returns:
            True if seek successful, False otherwise
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and close content source."""
        pass

    def get_content_info(self) -> ContentInfo:
        """
        Get information about the content.

        Returns:
            ContentInfo object with content details
        """
        return self.content_info

    def get_frame_at_time(self, timestamp: float) -> Optional[FrameData]:
        """
        Get frame at specific timestamp.

        Args:
            timestamp: Target timestamp in seconds

        Returns:
            FrameData object with frame information, or None if not available
        """
        if self.seek(timestamp):
            return self.get_next_frame()
        return None

    def get_status(self) -> ContentStatus:
        """
        Get current status of content source.

        Returns:
            Current ContentStatus
        """
        return self.status

    def get_current_time(self) -> float:
        """
        Get current playback time in seconds.

        Returns:
            Current time in seconds
        """
        return self.current_time

    def get_current_frame(self) -> int:
        """
        Get current frame number.

        Returns:
            Current frame index
        """
        return self.current_frame

    def get_progress(self) -> float:
        """
        Get playback progress as percentage.

        Returns:
            Progress from 0.0 to 1.0, or 0.0 if duration unknown
        """
        if self.content_info.duration > 0:
            return min(1.0, self.current_time / self.content_info.duration)
        return 0.0

    def is_finished(self) -> bool:
        """
        Check if content has finished playing.

        Returns:
            True if finished, False otherwise
        """
        return self.status == ContentStatus.ENDED

    def has_error(self) -> bool:
        """
        Check if content source has an error.

        Returns:
            True if error state, False otherwise
        """
        return self.status == ContentStatus.ERROR

    def get_error_message(self) -> str:
        """
        Get error message if in error state.

        Returns:
            Error message string, empty if no error
        """
        return self._error_message

    def set_error(self, message: str) -> None:
        """
        Set error state with message.

        Args:
            message: Error description
        """
        self.status = ContentStatus.ERROR
        self._error_message = message
        logger.error(f"Content source error: {message}")

    def clear_error(self) -> None:
        """Clear error state."""
        if self.status == ContentStatus.ERROR:
            self.status = ContentStatus.READY
            self._error_message = ""

    def pause(self) -> bool:
        """
        Pause content playback.

        Returns:
            True if successful, False otherwise
        """
        if self.status == ContentStatus.PLAYING:
            self.status = ContentStatus.PAUSED
            return True
        return False

    def resume(self) -> bool:
        """
        Resume content playback.

        Returns:
            True if successful, False otherwise
        """
        if self.status == ContentStatus.PAUSED:
            self.status = ContentStatus.PLAYING
            return True
        return False

    def reset(self) -> bool:
        """
        Reset content to beginning.

        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.seek(0.0)
            if result:
                self.current_frame = 0
                self.current_time = 0.0
                if self.status != ContentStatus.ERROR:
                    self.status = ContentStatus.READY
            return result
        except Exception as e:
            self.set_error(f"Reset failed: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        if self.setup():
            return self
        else:
            raise RuntimeError(f"Failed to setup content source: {self.filepath}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}('{self.filepath}', {self.status.value})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"filepath='{self.filepath}', "
            f"status={self.status.value}, "
            f"type={self.content_info.content_type.value}, "
            f"duration={self.content_info.duration:.2f}s, "
            f"current_time={self.current_time:.2f}s)"
        )


class ContentSourceRegistry:
    """Registry for content source plugins."""

    _sources: Dict[ContentType, type] = {}

    @classmethod
    def register(cls, content_type: ContentType, source_class: type) -> None:
        """
        Register a content source plugin.

        Args:
            content_type: Type of content this source handles
            source_class: ContentSource subclass
        """
        cls._sources[content_type] = source_class
        logger.info(
            f"Registered content source: {content_type.value} -> {source_class.__name__}"
        )

    @classmethod
    def get_source_class(cls, content_type: ContentType) -> Optional[type]:
        """
        Get content source class for given type.

        Args:
            content_type: Type of content

        Returns:
            ContentSource subclass or None if not found
        """
        return cls._sources.get(content_type)

    @classmethod
    def create_source(
        cls, filepath: str, content_type: Optional[ContentType] = None
    ) -> Optional[ContentSource]:
        """
        Create appropriate content source for file.

        Args:
            filepath: Path to content file
            content_type: Optional explicit content type

        Returns:
            ContentSource instance or None if no suitable source found
        """
        if content_type is None:
            content_type = cls.detect_content_type(filepath)

        source_class = cls.get_source_class(content_type)
        if source_class:
            try:
                return source_class(filepath)
            except Exception as e:
                logger.error(f"Failed to create content source: {e}")
                return None

        logger.warning(f"No content source registered for type: {content_type}")
        return None

    @classmethod
    def detect_content_type(cls, filepath: str) -> ContentType:
        """
        Detect content type from file path.

        Args:
            filepath: Path to content file

        Returns:
            Detected ContentType
        """
        filepath_lower = filepath.lower()

        # Image formats
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tga", ".webp"}
        if any(filepath_lower.endswith(ext) for ext in image_exts):
            return ContentType.IMAGE

        # Video formats
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"}
        if any(filepath_lower.endswith(ext) for ext in video_exts):
            return ContentType.VIDEO

        # Animation formats
        animation_exts = {".gif", ".apng"}
        if any(filepath_lower.endswith(ext) for ext in animation_exts):
            return ContentType.ANIMATION

        return ContentType.UNKNOWN

    @classmethod
    def get_registered_types(cls) -> list:
        """
        Get list of registered content types.

        Returns:
            List of registered ContentType values
        """
        return list(cls._sources.keys())

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered content sources."""
        cls._sources.clear()
