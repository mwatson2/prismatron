"""
Content Sources Package.

This package provides content source plugins for the Prismatron LED display.
Each plugin handles a different type of content (images, videos, animations, etc.)
and provides a consistent interface for frame data access.
"""

from .base import (
    ContentInfo,
    ContentSource,
    ContentSourceRegistry,
    ContentStatus,
    ContentType,
    FrameData,
)
from .image_source import ImageSource
from .video_source import VideoSource

# Auto-register available content sources
__all__ = [
    "ContentSource",
    "ContentType",
    "ContentStatus",
    "ContentInfo",
    "ContentSourceRegistry",
    "FrameData",
    "ImageSource",
    "VideoSource",
]
