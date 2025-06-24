# PrismatronImage Class Proposal

## Overview

A unified image class that wraps the standard Prismatron image representation (RGB, uint8) and provides comprehensive format conversion, I/O, and quality analysis capabilities.

## Design Goals

- **Canonical Format**: Internal storage as `(height, width, 3)` uint8 numpy array
- **Format Flexibility**: Seamless conversion between all internal representation formats
- **File I/O**: Support for multiple image formats with backend fallback
- **Quality Metrics**: Built-in PSNR, SSIM, and comparison utilities
- **No GPU Dependencies**: Pure CPU implementation using NumPy/PIL/OpenCV
- **Immutable by Default**: Defensive copying to prevent accidental modification

## Class Interface

```python
class PrismatronImage:
    """
    Immutable wrapper for Prismatron RGB images with format conversion and analysis.
    
    Canonical internal format: (height, width, 3) uint8 numpy array
    Supports all common internal formats used throughout the Prismatron codebase.
    """
    
    def __init__(self, data: np.ndarray, format_hint: str = "hwc"):
        """
        Initialize from numpy array with automatic format detection.
        
        Args:
            data: Image data in various supported formats
            format_hint: Format hint for ambiguous shapes ("hwc", "chw", "flat", etc.)
        """
    
    # ===== Factory Methods =====
    
    @classmethod
    def from_file(cls, filepath: str, backend: str = "auto") -> "PrismatronImage":
        """Load image from file with backend fallback (PIL → OpenCV → fallback)."""
    
    @classmethod
    def from_array(cls, array: np.ndarray, format: str) -> "PrismatronImage":
        """Create from numpy array with explicit format specification."""
        
    @classmethod
    def from_bytes(cls, data: bytes, format: str = None) -> "PrismatronImage":
        """Create from raw bytes (useful for web APIs)."""
        
    @classmethod
    def zeros(cls, height: int, width: int) -> "PrismatronImage":
        """Create black image of specified size."""
        
    @classmethod
    def ones(cls, height: int, width: int) -> "PrismatronImage":
        """Create white image of specified size."""
    
    # ===== Properties =====
    
    @property
    def height(self) -> int:
        """Image height in pixels."""
        
    @property
    def width(self) -> int:
        """Image width in pixels."""
        
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Canonical shape (height, width, 3)."""
        
    @property
    def size(self) -> int:
        """Total pixel count (height * width)."""
        
    @property
    def dtype(self) -> np.dtype:
        """Data type (always uint8)."""
    
    # ===== Format Conversion Methods =====
    
    def as_hwc(self) -> np.ndarray:
        """Return as (height, width, 3) - canonical format."""
        
    def as_chw(self) -> np.ndarray:
        """Return as (3, height, width) - channel-first format."""
        
    def as_flat_interleaved(self) -> np.ndarray:
        """Return as (height*width*3,) - completely flattened RGBRGB..."""
        
    def as_flat_spatial(self) -> np.ndarray:
        """Return as (height*width, 3) - flattened spatial, channel-last."""
        
    def as_flat_planar(self) -> np.ndarray:
        """Return as (3, height*width) - flattened per-channel planes."""
        
    def as_normalized_float(self) -> np.ndarray:
        """Return as (height, width, 3) float32 in range [0, 1]."""
        
    def as_normalized_chw_float(self) -> np.ndarray:
        """Return as (3, height, width) float32 in range [0, 1]."""
    
    # ===== File I/O Methods =====
    
    def save(self, filepath: str, quality: int = 95, backend: str = "auto") -> None:
        """Save to file with format detection from extension."""
        
    def to_bytes(self, format: str = "PNG", quality: int = 95) -> bytes:
        """Convert to bytes in specified format (useful for web APIs)."""
        
    def to_base64(self, format: str = "PNG", quality: int = 95) -> str:
        """Convert to base64 string (useful for web interfaces)."""
    
    # ===== Quality Metrics =====
    
    def psnr(self, other: "PrismatronImage") -> float:
        """Calculate Peak Signal-to-Noise Ratio with another image."""
        
    def ssim(self, other: "PrismatronImage") -> float:
        """Calculate Structural Similarity Index with another image."""
        
    def mse(self, other: "PrismatronImage") -> float:
        """Calculate Mean Squared Error with another image."""
        
    def mae(self, other: "PrismatronImage") -> float:
        """Calculate Mean Absolute Error with another image."""
        
    def compare(self, other: "PrismatronImage") -> Dict[str, float]:
        """Calculate all quality metrics at once."""
    
    # ===== Image Operations =====
    
    def resize(self, height: int, width: int, method: str = "bilinear") -> "PrismatronImage":
        """Return resized copy using specified interpolation method."""
        
    def crop(self, x: int, y: int, width: int, height: int) -> "PrismatronImage":
        """Return cropped copy of specified region."""
        
    def thumbnail(self, max_size: int = 256) -> "PrismatronImage":
        """Return thumbnail with max dimension preserved aspect ratio."""
        
    def center_crop(self, size: Tuple[int, int]) -> "PrismatronImage":
        """Return center-cropped copy to specified size."""
        
    def pad_to_size(self, height: int, width: int, fill_color: Tuple[int, int, int] = (0, 0, 0)) -> "PrismatronImage":
        """Return padded copy to specified size."""
    
    # ===== Analysis Methods =====
    
    def histogram(self) -> Dict[str, np.ndarray]:
        """Calculate per-channel histograms."""
        
    def center_of_mass(self) -> Tuple[float, float]:
        """Calculate center of mass of brightness."""
        
    def brightness_stats(self) -> Dict[str, float]:
        """Calculate brightness statistics (mean, std, min, max per channel)."""
        
    def color_distribution(self) -> Dict[str, float]:
        """Analyze color distribution properties."""
    
    # ===== Utility Methods =====
    
    def copy(self) -> "PrismatronImage":
        """Return deep copy."""
        
    def validate(self) -> bool:
        """Validate internal data integrity."""
        
    def __eq__(self, other) -> bool:
        """Equality comparison (pixel-perfect)."""
        
    def __repr__(self) -> str:
        """String representation for debugging."""
        
    def __array__(self) -> np.ndarray:
        """NumPy array interface - returns canonical format."""
```

## Format Specifications

### Supported Internal Formats

| Format Name | Shape | Description | Use Case |
|------------|-------|-------------|----------|
| `hwc` | `(height, width, 3)` | Standard format (canonical) | General use, display |
| `chw` | `(3, height, width)` | Channel-first | Deep learning, some optimizers |
| `flat_interleaved` | `(height*width*3,)` | RGBRGB... flat | WLED transmission, raw data |
| `flat_spatial` | `(height*width, 3)` | Spatial flat, RGB together | Matrix operations |
| `flat_planar` | `(3, height*width)` | RRR...GGG...BBB... | Channel-separated processing |

### Automatic Format Detection

```python
def _detect_format(shape: Tuple[int, ...], hint: str = None) -> str:
    """
    Automatic format detection based on array shape.
    
    Rules:
    - (H, W, 3): hwc
    - (3, H, W): chw  
    - (H*W*3,): flat_interleaved
    - (H*W, 3): flat_spatial
    - (3, H*W): flat_planar
    - Ambiguous cases use hint parameter
    """
```

## Implementation Details

### File I/O Backend Strategy

```python
class ImageBackend:
    """Abstract base for image I/O backends."""
    
    @abstractmethod
    def load(self, filepath: str) -> np.ndarray:
        """Load image as (H, W, 3) uint8."""
        
    @abstractmethod  
    def save(self, array: np.ndarray, filepath: str, **kwargs) -> None:
        """Save (H, W, 3) uint8 array."""

class PILBackend(ImageBackend):
    """PIL/Pillow backend (primary)."""
    
class OpenCVBackend(ImageBackend):
    """OpenCV backend (fallback)."""
    
class BasicBackend(ImageBackend):
    """Basic numpy backend for simple formats."""
```

### Quality Metrics Implementation

```python
def _calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """CPU-based PSNR calculation."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def _calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """CPU-based SSIM calculation using sliding window."""
    # Implementation using scipy.ndimage or pure numpy
```

## Usage Examples

### Basic Usage
```python
# Load image
img = PrismatronImage.from_file("test.jpg")
print(f"Image size: {img.width}x{img.height}")

# Convert to different formats
hwc_array = img.as_hwc()                    # (800, 480, 3)
chw_array = img.as_chw()                    # (3, 800, 480)  
flat_array = img.as_flat_interleaved()     # (1152000,)
spatial_flat = img.as_flat_spatial()       # (384000, 3)
planar_flat = img.as_flat_planar()         # (3, 384000)

# Save in different format
img.save("output.png")
```

### Quality Analysis
```python
# Compare two images
img1 = PrismatronImage.from_file("original.jpg")
img2 = PrismatronImage.from_file("optimized.jpg")

metrics = img1.compare(img2)
print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"MSE: {metrics['mse']:.2f}")
```

### Format Conversion
```python
# Create from various formats
data_hwc = np.random.randint(0, 256, (480, 800, 3), dtype=np.uint8)
img1 = PrismatronImage.from_array(data_hwc, "hwc")

data_chw = np.random.randint(0, 256, (3, 480, 800), dtype=np.uint8)  
img2 = PrismatronImage.from_array(data_chw, "chw")

data_flat = np.random.randint(0, 256, (1152000,), dtype=np.uint8)
img3 = PrismatronImage.from_array(data_flat, "flat_interleaved")

assert img1.shape == img2.shape == img3.shape == (480, 800, 3)
```

### Web API Support
```python
# Convert to base64 for web transmission
base64_data = img.to_base64("PNG")

# Create thumbnail for preview
thumb = img.thumbnail(256)
thumb_b64 = thumb.to_base64("JPEG", quality=85)
```

## Dependencies

- **Required**: `numpy`
- **Optional**: `pillow` (PIL backend)
- **Optional**: `opencv-python` (OpenCV backend)  
- **Optional**: `scipy` (for advanced SSIM calculation)

## Testing Strategy

- Unit tests for each format conversion
- Round-trip conversion tests
- Quality metric validation against reference implementations
- File I/O tests with various formats
- Performance benchmarks for conversion operations
- Edge case testing (empty images, single pixels, etc.)

## Benefits

1. **Unified Interface**: Single class for all image operations
2. **Format Safety**: Automatic validation and conversion
3. **No GPU Dependencies**: Works without CuPy/CUDA
4. **Comprehensive**: Covers all current use cases in codebase
5. **Extensible**: Easy to add new formats or operations
6. **Testable**: Isolated functionality with clear interfaces
7. **Performance**: Efficient numpy-based operations

This design provides a robust foundation for image handling throughout the Prismatron system while maintaining flexibility and avoiding GPU dependencies.