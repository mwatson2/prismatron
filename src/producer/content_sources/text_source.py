"""
Text Content Source.

This module provides a content source for rendering text on the LED display
with configurable fonts, colors, and animations.
"""

import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from ...const import FRAME_HEIGHT, FRAME_WIDTH
from .base import ContentInfo, ContentSource, ContentStatus, ContentType, FrameData

logger = logging.getLogger(__name__)


class TextContentSource(ContentSource):
    """
    Content source for rendering text with customizable appearance.

    Supports:
    - Custom text content
    - Font family and size selection
    - Foreground and background colors
    - Text positioning and alignment
    - Animation effects (scrolling, fade, etc.)
    """

    def __init__(self, config_str: str):
        """
        Initialize text content source.

        Args:
            config_str: JSON string containing text configuration
        """
        # For text effects, we use the config string as the "filepath"
        super().__init__(config_str)

        if not PIL_AVAILABLE:
            raise RuntimeError("PIL (Pillow) is required for text rendering")

        # Parse configuration
        try:
            self.config = json.loads(config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid text configuration JSON: {e}") from e

        # Text content and appearance
        self.text = self.config.get("text", "Hello World")
        self.font_family = self.config.get("font_family", "arial")
        self.font_size = self.config.get("font_size", 24)
        self.fg_color = self._parse_color(self.config.get("fg_color", "#FFFFFF"))
        self.bg_color = self._parse_color(self.config.get("bg_color", "#000000"))
        
        # Handle auto font sizing
        if self.font_size == "auto":
            self.font_size = self._calculate_auto_font_size()
        elif isinstance(self.font_size, str):
            try:
                self.font_size = int(self.font_size)
            except ValueError:
                self.font_size = 24

        # Animation and timing
        self.animation = self.config.get("animation", "static")  # static, scroll, fade
        self.duration = self.config.get("duration", 10.0)  # seconds
        self.fps = self.config.get("fps", 30.0)

        # Text positioning
        self.alignment = self.config.get("alignment", "center")  # left, center, right
        self.vertical_alignment = self.config.get("vertical_alignment", "center")  # top, center, bottom

        # Animation state
        self.start_time = 0.0
        self.frame_count = int(self.duration * self.fps)
        self.frame_interval = 1.0 / self.fps

        # Rendered frames cache
        self._frames = []
        self._frame_generated = False

        # Setup content info
        self.content_info.content_type = ContentType.TEXT
        self.content_info.filepath = f"text:{self.text[:20]}..."
        self.content_info.width = FRAME_WIDTH
        self.content_info.height = FRAME_HEIGHT
        self.content_info.duration = self.duration
        self.content_info.fps = self.fps
        self.content_info.frame_count = self.frame_count
        self.content_info.metadata = {
            "text": self.text,
            "font_family": self.font_family,
            "font_size": self.font_size,
            "animation": self.animation,
            "fg_color": self.fg_color,
            "bg_color": self.bg_color,
        }

    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """
        Parse color string to RGB tuple.

        Args:
            color_str: Color string in hex format (#RRGGBB) or named color

        Returns:
            RGB tuple (r, g, b)
        """
        if isinstance(color_str, (list, tuple)) and len(color_str) == 3:
            return tuple(int(c) for c in color_str)

        if color_str.startswith("#"):
            # Hex color
            hex_color = color_str[1:]
            if len(hex_color) == 6:
                return (
                    int(hex_color[0:2], 16),
                    int(hex_color[2:4], 16),
                    int(hex_color[4:6], 16),
                )

        # Named colors fallback
        named_colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
        }

        return named_colors.get(color_str.lower(), (255, 255, 255))

    def _calculate_auto_font_size(self) -> int:
        """
        Calculate font size to fill width with 10% margins.
        
        Returns:
            Optimal font size in pixels
        """
        if not self.text.strip():
            return 24  # Default size for empty text
            
        # Target width is 80% of frame width (10% margins on each side)
        target_width = int(FRAME_WIDTH * 0.8)
        
        # Binary search for optimal font size
        min_size = 8
        max_size = 72
        best_size = 24
        
        try:
            while min_size <= max_size:
                test_size = (min_size + max_size) // 2
                
                # Test this font size
                test_font = self._get_font_for_size(test_size)
                temp_img = Image.new("RGB", (1, 1))
                temp_draw = ImageDraw.Draw(temp_img)
                bbox = temp_draw.textbbox((0, 0), self.text, font=test_font)
                text_width = bbox[2] - bbox[0]
                
                if text_width <= target_width:
                    best_size = test_size
                    min_size = test_size + 1
                else:
                    max_size = test_size - 1
                    
        except Exception as e:
            logger.warning(f"Error calculating auto font size: {e}, using default")
            return 24
            
        return max(8, min(72, best_size))  # Clamp to reasonable range

    def _get_font_for_size(self, size: int) -> ImageFont.ImageFont:
        """
        Get PIL font object for a specific size.
        
        Args:
            size: Font size in pixels
            
        Returns:
            PIL font object
        """
        try:
            # Try to load system font
            font_paths = {
                "arial": ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"],
                "helvetica": ["helvetica.ttf", "Helvetica.ttf", "DejaVuSans.ttf"],
                "times": ["times.ttf", "Times.ttf", "DejaVuSerif.ttf"],
                "courier": ["courier.ttf", "Courier.ttf", "DejaVuSansMono.ttf"],
                "roboto": ["Roboto-Regular.ttf", "DejaVuSans.ttf"],
            }

            font_candidates = font_paths.get(self.font_family.lower(), ["DejaVuSans.ttf"])

            for font_name in font_candidates:
                try:
                    return ImageFont.truetype(font_name, size)
                except OSError:
                    continue

            # Fallback to default font
            return ImageFont.load_default()

        except Exception as e:
            logger.warning(f"Failed to load font {self.font_family} size {size}: {e}, using default")
            return ImageFont.load_default()

    def _get_font(self) -> ImageFont.ImageFont:
        """
        Get PIL font object.

        Returns:
            PIL font object
        """
        return self._get_font_for_size(self.font_size)

    def _render_static_frame(self) -> np.ndarray:
        """
        Render static text frame.

        Returns:
            RGB frame array in planar format (3, H, W)
        """
        # Create PIL image
        img = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), self.bg_color)
        draw = ImageDraw.Draw(img)
        font = self._get_font()

        # Get text dimensions
        bbox = draw.textbbox((0, 0), self.text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate position based on alignment
        if self.alignment == "left":
            x = 5
        elif self.alignment == "right":
            x = FRAME_WIDTH - text_width - 5
        else:  # center
            x = (FRAME_WIDTH - text_width) // 2

        if self.vertical_alignment == "top":
            y = 5
        elif self.vertical_alignment == "bottom":
            y = FRAME_HEIGHT - text_height - 5
        else:  # center
            y = (FRAME_HEIGHT - text_height) // 2

        # Draw text
        draw.text((x, y), self.text, font=font, fill=self.fg_color)

        # Convert to numpy array and planar format
        frame_array = np.array(img, dtype=np.uint8)  # (H, W, 3)
        return FrameData.convert_interleaved_to_planar(frame_array)  # (3, H, W)

    def _render_scrolling_frames(self) -> list:
        """
        Render scrolling text animation frames.

        Returns:
            List of RGB frame arrays in planar format
        """
        frames = []
        font = self._get_font()

        # Create a temporary image to measure text
        temp_img = Image.new("RGB", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), self.text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate scroll range (text moves from right to left)
        start_x = FRAME_WIDTH
        end_x = -text_width
        total_distance = start_x - end_x

        # Calculate vertical position
        if self.vertical_alignment == "top":
            y = 5
        elif self.vertical_alignment == "bottom":
            y = FRAME_HEIGHT - text_height - 5
        else:  # center
            y = (FRAME_HEIGHT - text_height) // 2

        # Generate frames
        for frame_idx in range(self.frame_count):
            progress = frame_idx / max(1, self.frame_count - 1)
            x = int(start_x - (total_distance * progress))

            # Create frame
            img = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), self.bg_color)
            draw = ImageDraw.Draw(img)
            draw.text((x, y), self.text, font=font, fill=self.fg_color)

            # Convert to planar format
            frame_array = np.array(img, dtype=np.uint8)
            planar_frame = FrameData.convert_interleaved_to_planar(frame_array)
            frames.append(planar_frame)

        return frames

    def _render_fade_frames(self) -> list:
        """
        Render fade in/out animation frames.

        Returns:
            List of RGB frame arrays in planar format
        """
        frames = []
        static_frame = self._render_static_frame()
        bg_frame = np.zeros_like(static_frame)
        bg_frame[:] = np.array(self.bg_color)[:, np.newaxis, np.newaxis]

        # Fade in for first half, fade out for second half
        for frame_idx in range(self.frame_count):
            progress = frame_idx / max(1, self.frame_count - 1)

            if progress <= 0.5:
                # Fade in
                alpha = progress * 2  # 0 to 1
            else:
                # Fade out
                alpha = (1.0 - progress) * 2  # 1 to 0

            # Blend frames
            alpha = max(0.0, min(1.0, alpha))
            blended = (alpha * static_frame + (1 - alpha) * bg_frame).astype(np.uint8)
            frames.append(blended)

        return frames

    def setup(self) -> bool:
        """
        Initialize the text content source.

        Returns:
            True if setup successful, False otherwise
        """
        try:
            if not PIL_AVAILABLE:
                self.set_error("PIL (Pillow) not available for text rendering")
                return False

            # Generate frames based on animation type
            if self.animation == "scroll":
                self._frames = self._render_scrolling_frames()
            elif self.animation == "fade":
                self._frames = self._render_fade_frames()
            else:  # static
                # For static text, create a single frame and duplicate it
                static_frame = self._render_static_frame()
                self._frames = [static_frame] * self.frame_count

            self._frame_generated = True
            self.status = ContentStatus.READY
            self.start_time = time.time()

            logger.info(f"Text content ready: '{self.text}' ({len(self._frames)} frames)")
            return True

        except Exception as e:
            self.set_error(f"Text setup failed: {e}")
            return False

    def get_next_frame(self) -> Optional[FrameData]:
        """
        Get the next frame from the text content.

        Returns:
            FrameData object with frame information, or None if end/error
        """
        if not self._frame_generated or self.status == ContentStatus.ERROR:
            return None

        if self.status == ContentStatus.ENDED:
            return None

        # Calculate current frame based on elapsed time
        if self.status != ContentStatus.PLAYING:
            self.status = ContentStatus.PLAYING

        elapsed_time = time.time() - self.start_time
        frame_index = int(elapsed_time / self.frame_interval)

        if frame_index >= len(self._frames):
            self.status = ContentStatus.ENDED
            return None

        # Update current time and frame
        self.current_time = elapsed_time
        self.current_frame = frame_index

        # Create FrameData object
        frame_data = FrameData(
            array=self._frames[frame_index],
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            channels=3,
            presentation_timestamp=self.start_time + (frame_index * self.frame_interval),
        )

        return frame_data

    def get_duration(self) -> float:
        """
        Get total duration of text content in seconds.

        Returns:
            Duration in seconds
        """
        return self.duration

    def seek(self, timestamp: float) -> bool:
        """
        Seek to specific timestamp in text content.

        Args:
            timestamp: Target timestamp in seconds

        Returns:
            True if seek successful, False otherwise
        """
        if not self._frame_generated:
            return False

        if timestamp < 0 or timestamp > self.duration:
            return False

        # Update timing
        self.start_time = time.time() - timestamp
        self.current_time = timestamp
        self.current_frame = int(timestamp / self.frame_interval)

        if self.status == ContentStatus.ENDED and timestamp < self.duration:
            self.status = ContentStatus.READY

        return True

    def cleanup(self) -> None:
        """Clean up text content resources."""
        self._frames.clear()
        self._frame_generated = False
        self.status = ContentStatus.UNINITIALIZED
        logger.debug(f"Text content cleaned up: {self.text[:20]}...")

    def reset(self) -> bool:
        """
        Reset text content to beginning.

        Returns:
            True if successful, False otherwise
        """
        if self._frame_generated:
            self.start_time = time.time()
            self.current_time = 0.0
            self.current_frame = 0
            self.status = ContentStatus.READY
            return True
        return False


# Auto-register the text content source
def register_text_source():
    """Register text content source with the registry."""
    from .base import ContentSourceRegistry

    ContentSourceRegistry.register(ContentType.TEXT, TextContentSource)


# Register when module is imported
register_text_source()
