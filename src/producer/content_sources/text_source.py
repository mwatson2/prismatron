"""
Text Content Source.

This module provides a content source for rendering text on the LED display
with configurable fonts, colors, and animations.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from ...const import DEFAULT_CONTENT_FPS, FRAME_HEIGHT, FRAME_WIDTH
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
            parsed_config = json.loads(config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid text configuration JSON: {e}") from e

        # Check if config is nested in a "config" key (from web API)
        if "config" in parsed_config and isinstance(parsed_config["config"], dict):
            logger.debug("Using nested config from web API")
            self.config = parsed_config["config"]
        else:
            logger.debug("Using direct config format")
            self.config = parsed_config

        logger.info(f"Text source config: {self.config}")

        # Text content and appearance
        self.text = self.config.get("text", "Hello World")
        self.font_family = self.config.get("font_family", "arial")
        self.font_style = self.config.get("font_style", "normal")  # normal, bold, italic, bold-italic
        self.fg_color = self._parse_color(self.config.get("fg_color", "#FFFFFF"))
        self.bg_color = self._parse_color(self.config.get("bg_color", "#000000"))

        # Animation and timing (set before font size calculation)
        self.animation = self.config.get("animation", "static")  # static, scroll, fade

        # Handle font sizing (after animation is set)
        self.font_size = self.config.get("font_size", 24)
        if self.font_size == "auto":
            self.font_size = self._calculate_auto_font_size()
        elif isinstance(self.font_size, str):
            try:
                self.font_size = int(self.font_size)
            except ValueError:
                self.font_size = 24
        self.duration = self.config.get("duration", 10.0)  # seconds
        self.fps = self.config.get("fps", DEFAULT_CONTENT_FPS)

        # Text positioning
        self.alignment = self.config.get("alignment", "center")  # left, center, right
        self.vertical_alignment = self.config.get("vertical_alignment", "center")  # top, center, bottom

        # Animation state
        self.frame_count = int(self.duration * self.fps)
        self.frame_interval = 1.0 / self.fps
        self._next_frame_index = 0  # Track which frame to return next

        # Rendered frames cache
        self._frames: List[np.ndarray] = []
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
            "font_style": self.font_style,
            "font_size": self.font_size,
            "animation": self.animation,
            "fg_color": self.fg_color,
            "bg_color": self.bg_color,
        }

    def _parse_color(self, color_value: Union[str, list, tuple]) -> Tuple[int, int, int]:
        """
        Parse color value to RGB tuple.

        Args:
            color_value: Color as hex string (#RRGGBB), named color, or RGB list/tuple

        Returns:
            RGB tuple (r, g, b)
        """
        # Handle list/tuple from JSON config
        if isinstance(color_value, (list, tuple)) and len(color_value) == 3:
            return (int(color_value[0]), int(color_value[1]), int(color_value[2]))

        # Handle hex format
        if isinstance(color_value, str) and color_value.startswith("#"):
            # Hex color
            hex_color = color_value[1:]
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

        # Named colors fallback (only for string values)
        if isinstance(color_value, str):
            return named_colors.get(color_value.lower(), (255, 255, 255))
        return (255, 255, 255)

    def _get_font_paths_with_style(self) -> Dict[str, list]:
        """
        Get font path candidates based on font family and style.

        Returns:
            Dictionary mapping font family to list of candidate paths
        """
        # Base font paths by font name (converted to lowercase with underscores)
        font_name_lower = self.font_family.lower()
        base_paths = {
            "arial": ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
            "helvetica": [
                "helvetica.ttf",
                "Helvetica.ttf",
                "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ],
            "times": ["times.ttf", "Times.ttf", "DejaVuSerif.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
            "courier": [
                "courier.ttf",
                "Courier.ttf",
                "DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            ],
            "roboto": ["Roboto-Regular.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
            "dejavu_sans": ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
            "ubuntu": [
                "Ubuntu-Regular.ttf",
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-Regular.ttf",
                "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ],
            "liberation_serif": [
                "LiberationSerif-Regular.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                "DejaVuSerif.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            ],
        }

        # Apply style variations - use font_name_lower as key
        styled_candidates = []
        paths = base_paths.get(font_name_lower, base_paths.get("dejavu_sans", []))

        # Add style-specific variants first
        if self.font_style == "bold":
            styled_candidates.extend([path.replace(".ttf", "-Bold.ttf") for path in paths])
            styled_candidates.extend([path.replace(".ttf", "b.ttf") for path in paths])  # Arial bold = arialb.ttf
        elif self.font_style == "italic":
            styled_candidates.extend([path.replace(".ttf", "-Italic.ttf") for path in paths])
            styled_candidates.extend([path.replace(".ttf", "i.ttf") for path in paths])  # Arial italic = ariali.ttf
        elif self.font_style == "bold-italic":
            styled_candidates.extend([path.replace(".ttf", "-BoldItalic.ttf") for path in paths])
            styled_candidates.extend(
                [path.replace(".ttf", "bi.ttf") for path in paths]  # Arial bold-italic = arialbi.ttf
            )

        # Add regular variants as fallback
        styled_candidates.extend(paths)

        return {font_name_lower: styled_candidates}

    def _calculate_auto_font_size(self) -> int:
        """
        Calculate font size based on animation type.
        For scroll animation: fills 90% of frame height for horizontal scrolling.
        For other animations: fills the entire frame without borders.

        Returns:
            Optimal font size in pixels
        """
        if not self.text.strip():
            return 24  # Default size for empty text

        if self.animation == "scroll":
            # For scroll animation: target 90% height for horizontal scrolling
            target_width = None  # No width constraint for scrolling
            target_height = int(FRAME_HEIGHT * 0.9)
            logger.info(
                f"Auto font sizing for horizontal scroll text '{self.text}': target_height={target_height} (90% of {FRAME_HEIGHT})"
            )
        else:
            # For static/fade animations: fill the frame without borders
            target_width = FRAME_WIDTH
            target_height = FRAME_HEIGHT
            logger.info(
                f"Auto font sizing for text '{self.text}': target_width={target_width}, target_height={target_height}"
            )

        # Binary search for optimal font size
        min_size = 8
        max_size = target_height
        best_size = 24

        try:
            while min_size <= max_size:
                test_size = (min_size + max_size) // 2

                # Test this font size
                test_font = self._get_font_for_size(test_size)
                temp_img = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT))
                temp_draw = ImageDraw.Draw(temp_img)
                bbox = temp_draw.textbbox((0, 0), self.text, font=test_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                logger.info(f"Testing font size {test_size}: text_width={text_width}, text_height={text_height}")

                # Check constraints based on animation type
                if self.animation == "scroll":
                    # For scroll: only check height constraint
                    if text_height <= target_height:
                        best_size = test_size
                        min_size = test_size + 1
                    else:
                        max_size = test_size - 1
                else:
                    # For static/fade: check both width and height constraints
                    if target_width is not None and text_width <= target_width and text_height <= target_height:
                        best_size = test_size
                        min_size = test_size + 1
                    else:
                        max_size = test_size - 1

        except Exception as e:
            logger.warning(f"Error calculating auto font size: {e}, using default")
            return 24

        final_size = best_size
        logger.info(f"Auto font size calculated: {final_size}px for text '{self.text}'")
        return final_size

    def _get_font_for_size(self, size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
        """
        Get PIL font object for a specific size.

        Args:
            size: Font size in pixels

        Returns:
            PIL font object
        """
        try:
            # First, try to find system fonts using matplotlib if available
            try:
                import matplotlib.font_manager as fm

                # Map font style to matplotlib properties
                weight = "normal"
                style: str = "normal"

                if self.font_style == "bold":
                    weight = "bold"
                elif self.font_style == "italic":
                    style = "italic"
                elif self.font_style == "bold-italic":
                    weight = "bold"
                    style = "italic"

                # Try to find font using matplotlib font manager with font name
                # First try by name (since font_family now contains the actual font name)
                font_prop = fm.FontProperties(fname=None)
                font_prop.set_name(self.font_family)
                font_prop.set_weight(weight)
                font_prop.set_style(style)  # type: ignore[arg-type]
                font_file = fm.findfont(font_prop)

                if font_file:
                    logger.debug(f"Found font via matplotlib: {font_file}")
                    return ImageFont.truetype(font_file, size)

            except ImportError:
                logger.debug("Matplotlib not available for font detection, using fallback")
            except Exception as e:
                logger.debug(f"Matplotlib font detection failed: {e}, using fallback")

            # Fallback: Try to load system font with style variants
            font_paths = self._get_font_paths_with_style()
            font_name_lower = self.font_family.lower()

            font_candidates = font_paths.get(
                font_name_lower,
                [
                    "DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/System/Library/Fonts/Arial.ttf",  # macOS
                    "/Windows/Fonts/arial.ttf",  # Windows
                ],
            )

            for font_name in font_candidates:
                try:
                    font = ImageFont.truetype(font_name, size)
                    logger.debug(f"Successfully loaded font {font_name} at size {size}")
                    return font
                except OSError:
                    continue

            # Try to get a default font with the specified size
            try:
                # PIL's default font is very small, try to get a bigger default
                default_font = ImageFont.load_default()
                logger.warning(f"Using PIL default font for size {size} (font family {self.font_family} not found)")
                return default_font
            except Exception:
                logger.error(f"Failed to load any font for size {size}")
                return ImageFont.load_default()

        except Exception as e:
            logger.warning(f"Failed to load font {self.font_family} size {size}: {e}, using default")
            return ImageFont.load_default()

    def _get_font(self) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
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

        # Get the bbox offset (this accounts for ascenders/descenders)
        bbox_offset_y = bbox[1]  # Top offset from baseline

        # Calculate position based on alignment (no borders)
        # Use int() to ensure integer coordinates for PIL text drawing
        if self.alignment == "left":
            x = 0
        elif self.alignment == "right":
            x = int(FRAME_WIDTH - text_width)
        else:  # center
            x = int((FRAME_WIDTH - text_width) // 2)

        if self.vertical_alignment == "top":
            y = int(-bbox_offset_y)  # Compensate for bbox offset
        elif self.vertical_alignment == "bottom":
            y = int(FRAME_HEIGHT - text_height - bbox_offset_y)  # Compensate for bbox offset
        else:  # center
            y = int((FRAME_HEIGHT - text_height) // 2 - bbox_offset_y)  # Compensate for bbox offset

        # Draw text
        logger.info(f"Rendering static text frame: '{self.text}' at ({x}, {y}) with size {self.font_size}")
        draw.text((x, y), self.text, font=font, fill=self.fg_color)

        # Convert to numpy array and planar format
        frame_array = np.array(img, dtype=np.uint8)  # (H, W, 3)
        return FrameData.convert_interleaved_to_planar(frame_array)  # (3, H, W)

    def _render_scrolling_frames(self) -> list:
        """
        Render horizontal scrolling text animation frames.
        Text starts at 10% from left edge and scrolls until right edge is 10% from right edge.

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

        # Get the bbox offset (this accounts for ascenders/descenders)
        bbox_offset_y = bbox[1]  # Top offset from baseline

        # Calculate scroll range based on font sizing mode
        if self.font_size == "auto" or (isinstance(self.font_size, str) and self.font_size == "auto"):
            # For auto-sized scroll: use 10% margins and center vertically
            margin = int(FRAME_WIDTH * 0.1)
            start_x = margin  # Start at 10% from left edge
            end_x = FRAME_WIDTH - margin - text_width  # End when right edge is 10% from right edge
            total_distance = start_x - end_x
            # Center vertically for auto-sized text
            y = (FRAME_HEIGHT - text_height) // 2 - bbox_offset_y
        else:
            # For manual font size: use original scroll behavior (right to left across full width)
            start_x = FRAME_WIDTH
            end_x = -text_width
            total_distance = start_x - end_x
            # Use original vertical alignment
            if self.vertical_alignment == "top":
                y = -bbox_offset_y
            elif self.vertical_alignment == "bottom":
                y = FRAME_HEIGHT - text_height - bbox_offset_y
            else:  # center
                y = (FRAME_HEIGHT - text_height) // 2 - bbox_offset_y

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

        # Use sequential frame counter instead of real time
        if self.status != ContentStatus.PLAYING:
            self.status = ContentStatus.PLAYING

        frame_index = self._next_frame_index

        if frame_index >= len(self._frames):
            self.status = ContentStatus.ENDED
            return None

        # Update current time and frame based on frame index
        self.current_time = frame_index * self.frame_interval
        self.current_frame = frame_index

        # Create FrameData object with local timestamp (starting from zero)
        frame_data = FrameData(
            array=self._frames[frame_index],
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            channels=3,
            presentation_timestamp=frame_index * self.frame_interval,  # Local timestamp from zero
            duration=self.duration,  # Total duration of this text item
        )

        # Increment for next call
        self._next_frame_index += 1

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

        # Update frame index based on timestamp
        self._next_frame_index = int(timestamp / self.frame_interval)
        self.current_time = timestamp
        self.current_frame = self._next_frame_index

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
            self._next_frame_index = 0
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
