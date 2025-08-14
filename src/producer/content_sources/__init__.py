"""
Content Sources Package.

This package provides content source plugins for the Prismatron LED display.
Each plugin handles a different type of content (images, videos, animations, etc.)
and provides a consistent interface for frame data access.
"""

import logging

# Initialize hardware acceleration cache proactively during module import
# This avoids delays when first video source is created
import threading

from .base import (
    ContentInfo,
    ContentSource,
    ContentSourceRegistry,
    ContentStatus,
    ContentType,
    FrameData,
)
from .hardware_acceleration_cache import initialize_hardware_acceleration
from .image_source import ImageSource
from .text_source import TextContentSource
from .video_source import VideoSource

logger = logging.getLogger(__name__)


def _init_hardware_acceleration_async():
    """Initialize hardware acceleration in background thread to avoid blocking import."""
    try:
        initialize_hardware_acceleration()
    except Exception as e:
        logger.warning(f"Failed to initialize hardware acceleration cache: {e}")


# Start hardware acceleration detection in background thread
_hw_init_thread = threading.Thread(target=_init_hardware_acceleration_async, daemon=True)
_hw_init_thread.start()
logger.info("Started hardware acceleration detection in background")

# Auto-register available content sources
__all__ = [
    "ContentSource",
    "ContentType",
    "ContentStatus",
    "ContentInfo",
    "ContentSourceRegistry",
    "FrameData",
    "ImageSource",
    "TextContentSource",
    "VideoSource",
]
