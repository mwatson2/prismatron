"""
PrismatronImage - Unified image handling for the Prismatron LED Display System.

This module provides a comprehensive image class that handles format conversions,
file I/O, quality metrics, and image analysis for the Prismatron system.

Canonical format: (3, width, height) uint8 - 'planar' format
Alternative format: (width, height, 3) uint8 - 'interleaved' format
"""

import base64
import io
import logging
import statistics
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Optional imports with fallbacks
try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ndimage = None

logger = logging.getLogger(__name__)

# Constants for Prismatron display
PRISMATRON_WIDTH = 800
PRISMATRON_HEIGHT = 480


class ImageBackend(ABC):
    """Abstract base class for image I/O backends."""
    
    @abstractmethod
    def load(self, filepath: str) -> np.ndarray:
        """Load image as (height, width, 3) uint8 interleaved format."""
        
    @abstractmethod
    def save(self, array: np.ndarray, filepath: str, **kwargs) -> None:
        """Save (height, width, 3) uint8 array to file."""
        
    @abstractmethod
    def load_bytes(self, data: bytes) -> np.ndarray:
        """Load image from bytes as (height, width, 3) uint8."""
        
    @abstractmethod
    def to_bytes(self, array: np.ndarray, format: str = "PNG", **kwargs) -> bytes:
        """Convert (height, width, 3) uint8 array to bytes."""


class PILBackend(ImageBackend):
    """PIL/Pillow backend for image I/O."""
    
    def load(self, filepath: str) -> np.ndarray:
        """Load image using PIL."""
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available")
            
        with Image.open(filepath) as img:
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array (height, width, 3) - standard format
            array = np.array(img, dtype=np.uint8)
            return array
    
    def save(self, array: np.ndarray, filepath: str, quality: int = 95, **kwargs) -> None:
        """Save image using PIL."""
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available")
            
        # Array is already in (height, width, 3) format
        img = Image.fromarray(array, 'RGB')
        
        # Set quality for JPEG
        save_kwargs = {}
        if Path(filepath).suffix.lower() in ['.jpg', '.jpeg']:
            save_kwargs['quality'] = quality
            
        img.save(filepath, **save_kwargs)
    
    def load_bytes(self, data: bytes) -> np.ndarray:
        """Load image from bytes using PIL."""
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available")
            
        with Image.open(io.BytesIO(data)) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            array = np.array(img, dtype=np.uint8)
            return array
    
    def to_bytes(self, array: np.ndarray, format: str = "PNG", quality: int = 95, **kwargs) -> bytes:
        """Convert array to bytes using PIL."""
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available")
            
        # Array is already in (height, width, 3) format
        img = Image.fromarray(array, 'RGB')
        
        buffer = io.BytesIO()
        save_kwargs = {}
        if format.upper() in ['JPEG', 'JPG']:
            save_kwargs['quality'] = quality
            
        img.save(buffer, format=format.upper(), **save_kwargs)
        return buffer.getvalue()


class OpenCVBackend(ImageBackend):
    """OpenCV backend for image I/O."""
    
    def load(self, filepath: str) -> np.ndarray:
        """Load image using OpenCV."""
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available")
            
        # OpenCV loads as BGR, need to convert to RGB
        array = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if array is None:
            raise ValueError(f"Could not load image: {filepath}")
            
        # Convert BGR to RGB - keep standard (height, width, 3) format
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return array
    
    def save(self, array: np.ndarray, filepath: str, quality: int = 95, **kwargs) -> None:
        """Save image using OpenCV."""
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available")
            
        # Convert RGB to BGR - array is already (height, width, 3)
        bgr_array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        
        # Set quality for JPEG
        if Path(filepath).suffix.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(filepath, bgr_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(filepath, bgr_array)
    
    def load_bytes(self, data: bytes) -> np.ndarray:
        """Load image from bytes using OpenCV."""
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available")
            
        # Decode bytes to array
        nparr = np.frombuffer(data, np.uint8)
        array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if array is None:
            raise ValueError("Could not decode image data")
            
        # Convert BGR to RGB - keep standard (height, width, 3) format
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return array
    
    def to_bytes(self, array: np.ndarray, format: str = "PNG", quality: int = 95, **kwargs) -> bytes:
        """Convert array to bytes using OpenCV."""
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available")
            
        # Convert RGB to BGR - array is already (height, width, 3)
        bgr_array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        
        # Encode to bytes
        if format.upper() in ['JPEG', 'JPG']:
            success, buffer = cv2.imencode('.jpg', bgr_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            success, buffer = cv2.imencode('.png', bgr_array)
            
        if not success:
            raise ValueError(f"Could not encode image to {format}")
            
        return buffer.tobytes()


class BasicBackend(ImageBackend):
    """Basic numpy backend for simple operations."""
    
    def load(self, filepath: str) -> np.ndarray:
        """Load numpy array from file."""
        if Path(filepath).suffix.lower() == '.npy':
            array = np.load(filepath)
            if array.dtype != np.uint8:
                array = (array * 255).astype(np.uint8)
            return array
        else:
            raise ValueError(f"BasicBackend only supports .npy files, got: {filepath}")
    
    def save(self, array: np.ndarray, filepath: str, **kwargs) -> None:
        """Save array to numpy file."""
        if Path(filepath).suffix.lower() == '.npy':
            np.save(filepath, array)
        else:
            raise ValueError(f"BasicBackend only supports .npy files, got: {filepath}")
    
    def load_bytes(self, data: bytes) -> np.ndarray:
        """Load numpy array from bytes."""
        raise NotImplementedError("BasicBackend does not support loading from bytes")
    
    def to_bytes(self, array: np.ndarray, format: str = "NPY", **kwargs) -> bytes:
        """Convert array to bytes."""
        if format.upper() == "NPY":
            buffer = io.BytesIO()
            np.save(buffer, array)
            return buffer.getvalue()
        else:
            raise ValueError(f"BasicBackend only supports NPY format, got: {format}")


class CameraBackend:
    """Camera capture backend using OpenCV."""
    
    def __init__(self, camera_id: int = 0):
        """Initialize camera backend."""
        self.camera_id = camera_id
        self._cap = None
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def open(self) -> None:
        """Open camera connection."""
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available for camera capture")
            
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
    
    def close(self) -> None:
        """Close camera connection."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def capture(self) -> np.ndarray:
        """Capture single frame as (height, width, 3) uint8."""
        if self._cap is None:
            raise RuntimeError("Camera not opened")
            
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
            
        # Convert BGR to RGB - keep standard (height, width, 3) format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


class PrismatronImage:
    """
    Immutable wrapper for Prismatron RGB images with format conversion and analysis.
    
    Canonical internal format: (3, height, width) uint8 numpy array ('planar')
    Alternative format: (height, width, 3) uint8 numpy array ('interleaved')
    """
    
    def __init__(self, data: np.ndarray, format_hint: str = "planar"):
        """
        Initialize from numpy array with automatic format detection.
        
        Args:
            data: Image data in various supported formats
            format_hint: Format hint for ambiguous shapes
        """
        self._data = self._normalize_to_planar(data, format_hint)
        self._validate()
    
    def _normalize_to_planar(self, data: np.ndarray, format_hint: str) -> np.ndarray:
        """Convert input data to canonical (3, height, width) planar format."""
        data = np.asarray(data, dtype=np.uint8)
        total_size = data.size
        
        if data.ndim == 3:
            if data.shape[0] == 3:
                # Already in planar format (3, height, width)
                return data.copy()
            elif data.shape[2] == 3:
                # Interleaved format (height, width, 3) -> planar (3, height, width)
                return np.transpose(data, (2, 0, 1))
            else:
                raise ValueError(f"Invalid 3D shape: {data.shape}. Expected (3, H, W) or (H, W, 3)")
        
        elif data.ndim == 2:
            # Flattened formats - need format hint or shape analysis
            if data.shape[1] == 3:
                # flat_spatial format (height*width, 3)
                pixels = data.shape[0]
                width, height = self._guess_dimensions(pixels)
                return data.T.reshape(3, height, width)
            
            elif data.shape[0] == 3:
                # flat_planar format (3, height*width)
                pixels_per_channel = data.shape[1]
                width, height = self._guess_dimensions(pixels_per_channel)
                return data.reshape(3, height, width)
            
            else:
                raise ValueError(f"Ambiguous 2D shape: {data.shape}")
        
        elif data.ndim == 1:
            # flat_interleaved format (width*height*3,)
            if total_size % 3 != 0:
                raise ValueError(f"Flat array size {total_size} not divisible by 3")
            
            pixels = total_size // 3
            width, height = self._guess_dimensions(pixels)
            
            # Reshape to (height, width, 3) then convert to planar
            interleaved = data.reshape(height, width, 3)
            return np.transpose(interleaved, (2, 0, 1))
        
        else:
            raise ValueError(f"Unsupported array dimensions: {data.ndim}")
    
    def _guess_dimensions(self, pixel_count: int) -> Tuple[int, int]:
        """Guess width and height from pixel count, preferring Prismatron dimensions."""
        # Check if it matches Prismatron dimensions
        if pixel_count == PRISMATRON_WIDTH * PRISMATRON_HEIGHT:
            return PRISMATRON_WIDTH, PRISMATRON_HEIGHT
        
        # Try to find factors that make a reasonable aspect ratio
        import math
        sqrt_pixels = int(math.sqrt(pixel_count))
        
        for width in range(sqrt_pixels, 0, -1):
            if pixel_count % width == 0:
                height = pixel_count // width
                aspect_ratio = width / height
                # Prefer aspect ratios between 1:2 and 2:1
                if 0.5 <= aspect_ratio <= 2.0:
                    return width, height
        
        # Fallback: use sqrt dimensions
        width = height = sqrt_pixels
        if width * height != pixel_count:
            raise ValueError(f"Cannot determine dimensions for {pixel_count} pixels")
        
        return width, height
    
    def _validate(self) -> None:
        """Validate internal data integrity."""
        if self._data.ndim != 3:
            raise ValueError(f"Internal data must be 3D, got {self._data.ndim}D")
        
        if self._data.shape[0] != 3:
            raise ValueError(f"First dimension must be 3 (RGB), got {self._data.shape[0]}")
        
        if self._data.dtype != np.uint8:
            raise ValueError(f"Data type must be uint8, got {self._data.dtype}")
        
        if self._data.shape[1] < 1 or self._data.shape[2] < 1:
            raise ValueError(f"Invalid dimensions: {self._data.shape}")
    
    # ===== Factory Methods =====
    
    @classmethod
    def from_file(cls, filepath: str, backend: str = "auto") -> "PrismatronImage":
        """Load image from file with backend fallback."""
        filepath = str(filepath)
        
        # Determine backend order
        if backend == "auto":
            backends = []
            if PIL_AVAILABLE:
                backends.append(PILBackend())
            if OPENCV_AVAILABLE:
                backends.append(OpenCVBackend())
            backends.append(BasicBackend())
        elif backend == "pil":
            backends = [PILBackend()]
        elif backend == "opencv":
            backends = [OpenCVBackend()]
        elif backend == "basic":
            backends = [BasicBackend()]
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Try backends in order
        last_error = None
        for backend_impl in backends:
            try:
                array = backend_impl.load(filepath)
                return cls.from_array(array, "interleaved")
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(f"All backends failed to load {filepath}. Last error: {last_error}")
    
    @classmethod
    def from_array(cls, array: np.ndarray, format: str) -> "PrismatronImage":
        """Create from numpy array with explicit format specification."""
        return cls(array, format_hint=format)
    
    @classmethod
    def from_bytes(cls, data: bytes, backend: str = "auto") -> "PrismatronImage":
        """Create from raw bytes."""
        # Determine backend order
        backends = []
        if backend == "auto":
            if PIL_AVAILABLE:
                backends.append(PILBackend())
            if OPENCV_AVAILABLE:
                backends.append(OpenCVBackend())
        elif backend == "pil":
            backends = [PILBackend()]
        elif backend == "opencv":
            backends = [OpenCVBackend()]
        else:
            raise ValueError(f"Backend {backend} not supported for bytes loading")
        
        # Try backends in order
        last_error = None
        for backend_impl in backends:
            try:
                array = backend_impl.load_bytes(data)
                return cls.from_array(array, "interleaved")
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(f"All backends failed to load from bytes. Last error: {last_error}")
    
    @classmethod
    def from_camera(cls, camera_id: int = 0, target_size: Optional[Tuple[int, int]] = None) -> "PrismatronImage":
        """Capture image from camera."""
        with CameraBackend(camera_id) as camera:
            array = camera.capture()
            img = cls.from_array(array, "interleaved")
            
            if target_size is not None:
                img = img.resize(target_size[0], target_size[1])
            
            return img
    
    @classmethod
    def zeros(cls, width: int, height: int) -> "PrismatronImage":
        """Create black image of specified size."""
        data = np.zeros((3, height, width), dtype=np.uint8)
        return cls(data, "planar")
    
    @classmethod
    def ones(cls, width: int, height: int) -> "PrismatronImage":
        """Create white image of specified size."""
        data = np.full((3, height, width), 255, dtype=np.uint8)
        return cls(data, "planar")
    
    @classmethod
    def solid_color(cls, width: int, height: int, color: Tuple[int, int, int]) -> "PrismatronImage":
        """Create solid color image."""
        data = np.full((3, height, width), 0, dtype=np.uint8)
        for i, c in enumerate(color):
            data[i, :, :] = c
        return cls(data, "planar")
    
    # ===== Properties =====
    
    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._data.shape[1]
    
    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._data.shape[2]
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Canonical shape (3, height, width)."""
        return self._data.shape
    
    @property
    def size(self) -> int:
        """Total pixel count (width * height)."""
        return self.width * self.height
    
    @property
    def dtype(self) -> np.dtype:
        """Data type (always uint8)."""
        return self._data.dtype
    
    # ===== Format Conversion Methods =====
    
    def as_planar(self) -> np.ndarray:
        """Return as (3, height, width) - canonical format."""
        return self._data.copy()
    
    def as_interleaved(self) -> np.ndarray:
        """Return as (height, width, 3) - standard format."""
        return np.transpose(self._data, (1, 2, 0))
    
    def as_flat_interleaved(self) -> np.ndarray:
        """Return as (height*width*3,) - completely flattened RGBRGB..."""
        return self.as_interleaved().ravel()
    
    def as_flat_spatial(self) -> np.ndarray:
        """Return as (height*width, 3) - flattened spatial, channel-last."""
        return self.as_interleaved().reshape(-1, 3)
    
    def as_flat_planar(self) -> np.ndarray:
        """Return as (3, height*width) - flattened per-channel planes."""
        return self._data.reshape(3, -1)
    
    def as_normalized_float(self) -> np.ndarray:
        """Return as (height, width, 3) float32 in range [0, 1]."""
        return self.as_interleaved().astype(np.float32) / 255.0
    
    def as_normalized_planar_float(self) -> np.ndarray:
        """Return as (3, height, width) float32 in range [0, 1]."""
        return self._data.astype(np.float32) / 255.0
    
    # ===== File I/O Methods =====
    
    def save(self, filepath: str, quality: int = 95, backend: str = "auto") -> None:
        """Save to file with format detection from extension."""
        filepath = str(filepath)
        
        # Determine backend
        if backend == "auto":
            if PIL_AVAILABLE:
                backend_impl = PILBackend()
            elif OPENCV_AVAILABLE:
                backend_impl = OpenCVBackend()
            else:
                backend_impl = BasicBackend()
        elif backend == "pil":
            backend_impl = PILBackend()
        elif backend == "opencv":
            backend_impl = OpenCVBackend()
        elif backend == "basic":
            backend_impl = BasicBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Save as interleaved format
        interleaved = self.as_interleaved()
        backend_impl.save(interleaved, filepath, quality=quality)
    
    def to_bytes(self, format: str = "PNG", quality: int = 95, backend: str = "auto") -> bytes:
        """Convert to bytes in specified format."""
        # Determine backend
        if backend == "auto":
            if PIL_AVAILABLE:
                backend_impl = PILBackend()
            elif OPENCV_AVAILABLE:
                backend_impl = OpenCVBackend()
            else:
                backend_impl = BasicBackend()
        elif backend == "pil":
            backend_impl = PILBackend()
        elif backend == "opencv":
            backend_impl = OpenCVBackend()
        elif backend == "basic":
            backend_impl = BasicBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Convert as interleaved format
        interleaved = self.as_interleaved()
        return backend_impl.to_bytes(interleaved, format=format, quality=quality)
    
    def to_base64(self, format: str = "PNG", quality: int = 95, backend: str = "auto") -> str:
        """Convert to base64 string."""
        bytes_data = self.to_bytes(format=format, quality=quality, backend=backend)
        return base64.b64encode(bytes_data).decode('ascii')
    
    # ===== Quality Metrics =====
    
    def psnr(self, other: "PrismatronImage") -> float:
        """Calculate Peak Signal-to-Noise Ratio with another image."""
        if self.shape != other.shape:
            raise ValueError(f"Image shapes don't match: {self.shape} vs {other.shape}")
        
        mse = np.mean((self._data.astype(float) - other._data.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def ssim(self, other: "PrismatronImage", window_size: int = 11) -> float:
        """Calculate Structural Similarity Index with another image."""
        if self.shape != other.shape:
            raise ValueError(f"Image shapes don't match: {self.shape} vs {other.shape}")
        
        # Simple SSIM implementation for each channel, then average
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim_values = []
        for channel in range(3):
            img1 = self._data[channel].astype(float)
            img2 = other._data[channel].astype(float)
            
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            
            sigma1_sq = np.var(img1)
            sigma2_sq = np.var(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
            denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
            
            ssim_values.append(numerator / denominator)
        
        return np.mean(ssim_values)
    
    def mse(self, other: "PrismatronImage") -> float:
        """Calculate Mean Squared Error with another image."""
        if self.shape != other.shape:
            raise ValueError(f"Image shapes don't match: {self.shape} vs {other.shape}")
        
        return float(np.mean((self._data.astype(float) - other._data.astype(float)) ** 2))
    
    def mae(self, other: "PrismatronImage") -> float:
        """Calculate Mean Absolute Error with another image."""
        if self.shape != other.shape:
            raise ValueError(f"Image shapes don't match: {self.shape} vs {other.shape}")
        
        return float(np.mean(np.abs(self._data.astype(float) - other._data.astype(float))))
    
    def compare(self, other: "PrismatronImage") -> Dict[str, float]:
        """Calculate all quality metrics at once."""
        return {
            "psnr": self.psnr(other),
            "ssim": self.ssim(other),
            "mse": self.mse(other),
            "mae": self.mae(other),
        }
    
    # ===== Image Operations =====
    
    def resize(self, width: int, height: int, method: str = "bilinear") -> "PrismatronImage":
        """Return resized copy using specified interpolation method."""
        if method == "nearest":
            # Simple nearest neighbor using numpy
            x_ratio = self.width / width
            y_ratio = self.height / height
            
            new_data = np.zeros((3, height, width), dtype=np.uint8)  # (3, height, width)
            for i in range(width):
                for j in range(height):
                    # Map new coordinates to source coordinates
                    src_x = min(int(i * x_ratio), self.width - 1)
                    src_y = min(int(j * y_ratio), self.height - 1)
                    # Copy from source (3, height, width) to destination
                    new_data[:, j, i] = self._data[:, src_y, src_x]
            
            return PrismatronImage(new_data, "planar")
        
        else:
            # Use PIL or OpenCV for better interpolation
            interleaved = self.as_interleaved()
            
            if PIL_AVAILABLE:
                # interleaved is already (height, width, 3) - perfect for PIL
                img = Image.fromarray(interleaved, 'RGB')
                
                if method == "bilinear":
                    resampling = Image.BILINEAR
                elif method == "bicubic":
                    resampling = Image.BICUBIC
                elif method == "lanczos":
                    resampling = Image.LANCZOS
                else:
                    resampling = Image.BILINEAR
                
                resized_img = img.resize((width, height), resampling)
                resized_hwc = np.array(resized_img, dtype=np.uint8)  # (height, width, 3)
                
                return PrismatronImage.from_array(resized_hwc, "interleaved")
            
            elif OPENCV_AVAILABLE:
                # interleaved is already (height, width, 3) - perfect for OpenCV
                if method == "bilinear":
                    interpolation = cv2.INTER_LINEAR
                elif method == "bicubic":
                    interpolation = cv2.INTER_CUBIC
                elif method == "lanczos":
                    interpolation = cv2.INTER_LANCZOS4
                else:
                    interpolation = cv2.INTER_LINEAR
                
                resized_hwc = cv2.resize(interleaved, (width, height), interpolation=interpolation)
                
                return PrismatronImage.from_array(resized_hwc, "interleaved")
            
            else:
                # Fallback to nearest neighbor
                return self.resize(width, height, "nearest")
    
    def crop(self, x: int, y: int, width: int, height: int) -> "PrismatronImage":
        """Return cropped copy of specified region."""
        if x < 0 or y < 0 or x + width > self.width or y + height > self.height:
            raise ValueError(f"Crop region ({x}, {y}, {width}, {height}) exceeds image bounds ({self.width}, {self.height})")
        
        cropped_data = self._data[:, y:y+height, x:x+width].copy()  # (3, height, width) format
        return PrismatronImage(cropped_data, "planar")
    
    def thumbnail(self, max_size: int = 256) -> "PrismatronImage":
        """Return thumbnail with max dimension preserving aspect ratio."""
        if max(self.width, self.height) <= max_size:
            return self.copy()
        
        if self.width > self.height:
            new_width = max_size
            new_height = int((max_size * self.height) / self.width)
        else:
            new_height = max_size
            new_width = int((max_size * self.width) / self.height)
        
        return self.resize(new_width, new_height)
    
    def center_crop(self, width: int, height: int) -> "PrismatronImage":
        """Return center-cropped copy to specified size."""
        if width > self.width or height > self.height:
            raise ValueError(f"Crop size ({width}, {height}) larger than image ({self.width}, {self.height})")
        
        x = (self.width - width) // 2
        y = (self.height - height) // 2
        
        return self.crop(x, y, width, height)
    
    def bounding_box(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Return the smallest bounding box around non-zero pixels.
        
        Returns:
            (x, y, width, height) or None if image is all zeros
        """
        # Find non-zero pixels across all channels
        non_zero = np.any(self._data > 0, axis=0)  # (height, width)
        
        if not np.any(non_zero):
            return None  # All pixels are zero
        
        # Find bounds - non_zero is (height, width)
        rows = np.any(non_zero, axis=1)  # Any non-zero in each row (height)
        cols = np.any(non_zero, axis=0)  # Any non-zero in each col (width)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
    
    def crop_to_content(self) -> "PrismatronImage":
        """Return image cropped to its content bounding box."""
        bbox = self.bounding_box()
        if bbox is None:
            # Return a 1x1 black image if no content
            return PrismatronImage.zeros(1, 1)
        
        x, y, width, height = bbox
        return self.crop(x, y, width, height)
    
    # ===== Analysis Methods =====
    
    def histogram(self, bins: int = 256) -> Dict[str, np.ndarray]:
        """Calculate per-channel histograms."""
        histograms = {}
        channel_names = ['red', 'green', 'blue']
        
        for i, name in enumerate(channel_names):
            hist, _ = np.histogram(self._data[i], bins=bins, range=(0, 255))
            histograms[name] = hist
        
        return histograms
    
    def center_of_mass(self) -> Tuple[float, float]:
        """Calculate center of mass of brightness."""
        # Calculate brightness as average of RGB channels
        brightness = np.mean(self._data, axis=0)  # (height, width)
        
        # Create coordinate grids - brightness is (height, width)
        y_coords, x_coords = np.meshgrid(range(self.height), range(self.width), indexing='ij')
        
        total_brightness = np.sum(brightness)
        if total_brightness == 0:
            return (self.width / 2, self.height / 2)
        
        center_x = np.sum(x_coords * brightness) / total_brightness
        center_y = np.sum(y_coords * brightness) / total_brightness
        
        return (float(center_x), float(center_y))
    
    def brightness_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate brightness statistics per channel."""
        stats = {}
        channel_names = ['red', 'green', 'blue', 'average']
        
        # Per-channel stats
        for i, name in enumerate(channel_names[:3]):
            channel_data = self._data[i].astype(float)
            stats[name] = {
                "mean": float(np.mean(channel_data)),
                "std": float(np.std(channel_data)),
                "min": float(np.min(channel_data)),
                "max": float(np.max(channel_data)),
            }
        
        # Average across channels
        avg_data = np.mean(self._data, axis=0).astype(float)
        stats["average"] = {
            "mean": float(np.mean(avg_data)),
            "std": float(np.std(avg_data)),
            "min": float(np.min(avg_data)),
            "max": float(np.max(avg_data)),
        }
        
        return stats
    
    def color_distribution(self) -> Dict[str, float]:
        """Analyze color distribution properties."""
        # Convert to HSV for better color analysis
        rgb_float = self.as_normalized_float()
        
        # Simple RGB to HSV conversion
        max_val = np.max(rgb_float, axis=2)
        min_val = np.min(rgb_float, axis=2)
        diff = max_val - min_val
        
        # Saturation
        saturation = np.where(max_val == 0, 0, np.divide(diff, max_val, out=np.zeros_like(diff), where=max_val!=0))
        avg_saturation = float(np.mean(saturation))
        
        # Value (brightness)
        avg_value = float(np.mean(max_val))
        
        # Color variance
        color_variance = float(np.var(self._data.astype(float)))
        
        return {
            "avg_saturation": avg_saturation,
            "avg_value": avg_value,
            "color_variance": color_variance,
        }
    
    # ===== Utility Methods =====
    
    def copy(self) -> "PrismatronImage":
        """Return deep copy."""
        return PrismatronImage(self._data.copy(), "planar")
    
    def validate(self) -> bool:
        """Validate internal data integrity."""
        try:
            self._validate()
            return True
        except Exception:
            return False
    
    def __eq__(self, other) -> bool:
        """Equality comparison (pixel-perfect)."""
        if not isinstance(other, PrismatronImage):
            return False
        return np.array_equal(self._data, other._data)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"PrismatronImage(shape={self.shape}, dtype={self.dtype})"
    
    def __array__(self) -> np.ndarray:
        """NumPy array interface - returns canonical format."""
        return self._data.copy()