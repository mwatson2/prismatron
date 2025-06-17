"""
Image Content Source Plugin.

This module implements a content source for static images with support
for various image formats, scaling, and duration control.
"""

import numpy as np
import logging
from typing import Optional
import os

from .base import ContentSource, ContentType, ContentStatus, ContentSourceRegistry, FrameData

# Try to import PIL/Pillow for image loading
try:
    from PIL import Image, ImageOps
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    Image = None
    ImageOps = None

# Try to import OpenCV as fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None


logger = logging.getLogger(__name__)


class ImageSource(ContentSource):
    """
    Content source for static images.
    
    Supports various image formats and provides configurable display duration.
    Uses PIL/Pillow for image loading with OpenCV as fallback.
    """
    
    def __init__(self, filepath: str, duration: float = 5.0):
        """
        Initialize image source.
        
        Args:
            filepath: Path to image file
            duration: How long to display image in seconds (default: 5.0)
        """
        super().__init__(filepath)
        self.duration = duration
        self.content_info.content_type = ContentType.IMAGE
        self.content_info.duration = duration
        self.content_info.fps = 1.0  # Static image, minimal fps
        
        self._image_data = None
        self._display_start_time = 0.0
        self._has_been_displayed = False
    
    def setup(self) -> bool:
        """
        Load and prepare the image for display.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(self.filepath):
                self.set_error(f"Image file not found: {self.filepath}")
                return False
            
            # Load image using available library
            if PILLOW_AVAILABLE:
                success = self._load_with_pillow()
            elif OPENCV_AVAILABLE:
                success = self._load_with_opencv()
            else:
                self.set_error("No image loading library available (PIL/OpenCV)")
                return False
            
            if success:
                self.status = ContentStatus.READY
                self.current_time = 0.0
                self.current_frame = 0
                logger.info(f"Loaded image: {self.filepath} ({self.content_info.width}x{self.content_info.height})")
                return True
            else:
                return False
            
        except Exception as e:
            self.set_error(f"Failed to setup image source: {e}")
            return False
    
    def _load_with_pillow(self) -> bool:
        """
        Load image using PIL/Pillow.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open and process image
            with Image.open(self.filepath) as img:
                # Convert to RGBA if needed
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Auto-orient based on EXIF data
                img = ImageOps.exif_transpose(img)
                
                # Store image info
                self.content_info.width = img.width
                self.content_info.height = img.height
                
                # Convert to numpy array
                self._image_data = np.array(img, dtype=np.uint8)
                
                # Get metadata if available
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    if exif:
                        self.content_info.metadata['exif'] = exif
                
                return True
                
        except Exception as e:
            self.set_error(f"PIL image loading failed: {e}")
            return False
    
    def _load_with_opencv(self) -> bool:
        """
        Load image using OpenCV.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image (BGR format)
            img_bgr = cv2.imread(self.filepath, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                self.set_error(f"OpenCV could not load image: {self.filepath}")
                return False
            
            # Handle different channel counts
            if len(img_bgr.shape) == 2:
                # Grayscale - convert to RGB
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
            elif img_bgr.shape[2] == 3:
                # BGR - convert to RGB
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            elif img_bgr.shape[2] == 4:
                # BGRA - convert to RGB (drop alpha)
                img_bgr_no_alpha = img_bgr[:, :, :3]
                img_rgb = cv2.cvtColor(img_bgr_no_alpha, cv2.COLOR_BGR2RGB)
            else:
                self.set_error(f"Unsupported image format: {img_bgr.shape[2]} channels")
                return False
            
            self._image_data = img_rgb
            self.content_info.width = img_rgb.shape[1]
            self.content_info.height = img_rgb.shape[0]
            
            return True
            
        except Exception as e:
            self.set_error(f"OpenCV image loading failed: {e}")
            return False
    
    def get_next_frame(self) -> Optional[FrameData]:
        """
        Get the next frame from the image source.
        
        Returns:
            FrameData object with frame information, or None if finished/error
        """
        try:
            if self.status == ContentStatus.ERROR:
                return None
            
            if self._image_data is None:
                self.set_error("No image data loaded")
                return None
            
            # Check if we've exceeded display duration
            if self._has_been_displayed and self.current_time >= self.duration:
                self.status = ContentStatus.ENDED
                return None
            
            # Mark as playing and note start time if first frame
            if not self._has_been_displayed:
                self.status = ContentStatus.PLAYING
                self._display_start_time = 0.0
                self._has_been_displayed = True
            
            # Create frame data with the loaded image
            frame_data = FrameData(
                array=self._image_data.copy(),  # Copy to avoid modification
                width=self.content_info.width,
                height=self.content_info.height,
                channels=3,  # RGB
                presentation_timestamp=self.current_time
            )
            
            # Update timing (simulate minimal frame rate)
            self.current_time += 1.0 / self.content_info.fps
            self.current_frame += 1
            
            return frame_data
            
        except Exception as e:
            self.set_error(f"Failed to get next frame: {e}")
            return None
    
    
    def get_duration(self) -> float:
        """
        Get display duration of image.
        
        Returns:
            Duration in seconds
        """
        return self.duration
    
    def seek(self, timestamp: float) -> bool:
        """
        Seek to specific time in image display.
        
        Args:
            timestamp: Target timestamp in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if timestamp < 0:
                timestamp = 0.0
            elif timestamp > self.duration:
                timestamp = self.duration
            
            self.current_time = timestamp
            
            # Reset display state if seeking to beginning
            if timestamp == 0.0:
                self._has_been_displayed = False
                self.status = ContentStatus.READY
            
            return True
            
        except Exception as e:
            self.set_error(f"Seek failed: {e}")
            return False
    
    def set_duration(self, duration: float) -> None:
        """
        Set display duration for image.
        
        Args:
            duration: New duration in seconds
        """
        if duration > 0:
            self.duration = duration
            self.content_info.duration = duration
    
    def cleanup(self) -> None:
        """Clean up image resources."""
        try:
            self._image_data = None
            self._has_been_displayed = False
            self.status = ContentStatus.UNINITIALIZED
            logger.debug(f"Cleaned up image source: {self.filepath}")
            
        except Exception as e:
            logger.error(f"Error during image source cleanup: {e}")
    
    
    def get_thumbnail(self, max_size: tuple = (128, 128)) -> Optional[np.ndarray]:
        """
        Get thumbnail version of image.
        
        Args:
            max_size: Maximum dimensions for thumbnail (width, height)
            
        Returns:
            Thumbnail as numpy array or None if error
        """
        if self._image_data is None:
            return None
        
        try:
            if PILLOW_AVAILABLE:
                pil_img = Image.fromarray(self._image_data)
                pil_img.thumbnail(max_size, Image.LANCZOS)
                return np.array(pil_img)
            elif OPENCV_AVAILABLE:
                height, width = self._image_data.shape[:2]
                max_w, max_h = max_size
                
                # Calculate scaling factor
                scale = min(max_w / width, max_h / height)
                if scale < 1.0:
                    new_w = int(width * scale)
                    new_h = int(height * scale)
                    return cv2.resize(self._image_data, (new_w, new_h))
                else:
                    return self._image_data.copy()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return None


# Register the image source plugin
ContentSourceRegistry.register(ContentType.IMAGE, ImageSource)