#!/usr/bin/env python3
"""
PrismatronImage Usage Examples

This script demonstrates how to use the PrismatronImage class for unified
image handling in the Prismatron LED Display System.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.prismatron_image import PrismatronImage


def basic_usage_examples():
    """Demonstrate basic PrismatronImage usage."""
    print("=== Basic Usage Examples ===\n")
    
    # Create images using factory methods
    print("1. Creating images:")
    
    # Black image
    black_img = PrismatronImage.zeros(100, 80)
    print(f"   Black image: {black_img.width}x{black_img.height}")
    
    # White image
    white_img = PrismatronImage.ones(100, 80)
    print(f"   White image: {white_img.width}x{white_img.height}")
    
    # Solid color image (red)
    red_img = PrismatronImage.solid_color(100, 80, (255, 0, 0))
    print(f"   Red image: {red_img.width}x{red_img.height}")
    
    # From numpy array
    random_data = np.random.randint(0, 256, (3, 100, 80), dtype=np.uint8)
    random_img = PrismatronImage(random_data, "planar")
    print(f"   Random image: {random_img.width}x{random_img.height}")
    print()


def format_conversion_examples():
    """Demonstrate format conversion capabilities."""
    print("=== Format Conversion Examples ===\n")
    
    # Create test image
    img = PrismatronImage.solid_color(50, 30, (128, 64, 192))
    print(f"Original image: {img.width}x{img.height}, shape={img.shape}")
    
    # Convert to different formats
    print("\n2. Format conversions:")
    
    # Canonical planar format (3, width, height)
    planar = img.as_planar()
    print(f"   Planar: {planar.shape} - canonical format")
    
    # Alternative interleaved format (width, height, 3)
    interleaved = img.as_interleaved()
    print(f"   Interleaved: {interleaved.shape} - alternative format")
    
    # Flattened formats
    flat_interleaved = img.as_flat_interleaved()
    print(f"   Flat interleaved: {flat_interleaved.shape} - RGBRGB...")
    
    flat_spatial = img.as_flat_spatial()
    print(f"   Flat spatial: {flat_spatial.shape} - (pixels, RGB)")
    
    flat_planar = img.as_flat_planar()
    print(f"   Flat planar: {flat_planar.shape} - (3, pixels)")
    
    # Normalized float formats
    norm_float = img.as_normalized_float()
    print(f"   Normalized float: {norm_float.shape}, range [{norm_float.min():.3f}, {norm_float.max():.3f}]")
    
    norm_planar = img.as_normalized_planar_float()
    print(f"   Normalized planar: {norm_planar.shape}, dtype={norm_planar.dtype}")
    print()


def image_operations_examples():
    """Demonstrate image manipulation operations."""
    print("=== Image Operations Examples ===\n")
    
    # Create test pattern
    img = PrismatronImage.zeros(100, 80)
    data = img.as_planar()
    # Add colored rectangles
    data[0, 20:80, 20:60] = 255  # Red rectangle
    data[1, 10:90, 10:70] = 128  # Green background
    data[2, 75:95, 40:60] = 200  # Blue corner
    img = PrismatronImage(data, "planar")
    
    print(f"Original: {img.width}x{img.height}")
    
    print("\n3. Image operations:")
    
    # Resize operations
    resized = img.resize(50, 40, "nearest")
    print(f"   Resized (nearest): {resized.width}x{resized.height}")
    
    # Cropping
    cropped = img.crop(10, 5, 30, 20)
    print(f"   Cropped: {cropped.width}x{cropped.height}")
    
    center_crop = img.center_crop(60, 50)
    print(f"   Center crop: {center_crop.width}x{center_crop.height}")
    
    # Thumbnail with aspect ratio preservation
    thumbnail = img.thumbnail(40)
    print(f"   Thumbnail: {thumbnail.width}x{thumbnail.height}")
    
    # Content-based cropping
    content_crop = img.crop_to_content()
    print(f"   Content crop: {content_crop.width}x{content_crop.height}")
    
    # Bounding box detection
    bbox = img.bounding_box()
    if bbox:
        x, y, w, h = bbox
        print(f"   Bounding box: ({x}, {y}) size {w}x{h}")
    print()


def quality_analysis_examples():
    """Demonstrate quality metrics and analysis."""
    print("=== Quality Analysis Examples ===\n")
    
    # Create similar images for comparison
    img1 = PrismatronImage.solid_color(50, 30, (100, 100, 100))
    img2 = PrismatronImage.solid_color(50, 30, (110, 110, 110))  # Slightly different
    img3 = img1.copy()  # Identical
    
    print("4. Quality metrics:")
    
    # Compare identical images
    psnr_identical = img1.psnr(img3)
    mse_identical = img1.mse(img3)
    print(f"   Identical images - PSNR: {psnr_identical}, MSE: {mse_identical}")
    
    # Compare similar images
    metrics = img1.compare(img2)
    print(f"   Similar images - PSNR: {metrics['psnr']:.2f}dB, SSIM: {metrics['ssim']:.4f}")
    print(f"                    MSE: {metrics['mse']:.2f}, MAE: {metrics['mae']:.2f}")
    
    print("\n5. Image analysis:")
    
    # Create colorful test image
    test_img = PrismatronImage.zeros(60, 40)
    data = test_img.as_planar()
    data[0, :30, :] = 200   # Red half
    data[1, 30:, :] = 150   # Green half  
    data[2, :, 20:] = 100   # Blue overlay
    test_img = PrismatronImage(data, "planar")
    
    # Brightness statistics
    brightness_stats = test_img.brightness_stats()
    print(f"   Red channel mean: {brightness_stats['red']['mean']:.1f}")
    print(f"   Green channel mean: {brightness_stats['green']['mean']:.1f}")
    print(f"   Overall brightness: {brightness_stats['average']['mean']:.1f}")
    
    # Color distribution
    color_dist = test_img.color_distribution()
    print(f"   Average saturation: {color_dist['avg_saturation']:.3f}")
    print(f"   Average value: {color_dist['avg_value']:.3f}")
    
    # Center of mass
    com_x, com_y = test_img.center_of_mass()
    print(f"   Center of mass: ({com_x:.1f}, {com_y:.1f})")
    
    # Histogram
    histogram = test_img.histogram()
    print(f"   Histogram channels: {list(histogram.keys())}")
    print(f"   Red channel non-zero bins: {np.count_nonzero(histogram['red'])}")
    print()


def file_io_examples():
    """Demonstrate file I/O capabilities."""
    print("=== File I/O Examples ===\n")
    
    # Create test image
    img = PrismatronImage.solid_color(30, 20, (255, 128, 64))
    
    print("6. File operations:")
    
    # Convert to bytes
    npy_bytes = img.to_bytes("NPY", backend="basic")
    print(f"   NPY bytes: {len(npy_bytes)} bytes")
    
    # Test if PIL is available for more formats
    try:
        png_bytes = img.to_bytes("PNG")
        print(f"   PNG bytes: {len(png_bytes)} bytes")
        
        # Base64 encoding
        b64_string = img.to_base64("PNG")
        print(f"   Base64 length: {len(b64_string)} characters")
        
        # Create from bytes
        img_from_bytes = PrismatronImage.from_bytes(png_bytes)
        print(f"   Loaded from bytes: {img_from_bytes.width}x{img_from_bytes.height}")
        
    except Exception as e:
        print(f"   PNG/JPEG operations require PIL (not available): {e}")
    
    print()


def advanced_usage_examples():
    """Demonstrate advanced usage patterns."""
    print("=== Advanced Usage Examples ===\n")
    
    print("7. Format auto-detection:")
    
    # Create data in various formats and let PrismatronImage detect
    
    # Flat interleaved (common for LED arrays)
    led_data = np.random.randint(0, 256, 800*480*3, dtype=np.uint8)  # Prismatron size
    img_from_led = PrismatronImage(led_data, "flat_interleaved")
    print(f"   From LED data: {img_from_led.width}x{img_from_led.height}")
    
    # Flat spatial (matrix operations)
    spatial_data = np.random.randint(0, 256, (100*80, 3), dtype=np.uint8)
    img_from_spatial = PrismatronImage(spatial_data, "flat_spatial")
    print(f"   From spatial data: {img_from_spatial.width}x{img_from_spatial.height}")
    
    # Interleaved (common image format)
    hwc_data = np.random.randint(0, 256, (60, 40, 3), dtype=np.uint8)
    img_from_hwc = PrismatronImage(hwc_data, "interleaved")
    print(f"   From HWC data: {img_from_hwc.width}x{img_from_hwc.height}")
    
    print("\n8. Round-trip validation:")
    
    # Test that conversions preserve data
    original = PrismatronImage.solid_color(40, 30, (123, 234, 56))
    
    # Convert through different formats and back
    interleaved = original.as_interleaved()
    roundtrip1 = PrismatronImage.from_array(interleaved, "interleaved")
    
    flat_spatial = original.as_flat_spatial()
    roundtrip2 = PrismatronImage(flat_spatial, "flat_spatial")
    
    print(f"   Original == Interleaved roundtrip: {original == roundtrip1}")
    print(f"   Original dimensions == Spatial roundtrip: {original.size == roundtrip2.size}")
    
    print("\n9. Utility features:")
    
    # Copy and equality
    img_copy = original.copy()
    print(f"   Copy equals original: {original == img_copy}")
    print(f"   Copy is independent: {original._data is not img_copy._data}")
    
    # Validation
    print(f"   Data validation: {original.validate()}")
    
    # String representation
    print(f"   String repr: {repr(original)}")
    
    # NumPy array interface
    as_array = np.array(original)
    print(f"   NumPy interface: {as_array.shape}, dtype={as_array.dtype}")
    
    print()


def main():
    """Run all examples."""
    print("PrismatronImage Class Demonstration")
    print("=" * 50)
    print()
    
    try:
        basic_usage_examples()
        format_conversion_examples()
        image_operations_examples()
        quality_analysis_examples()
        file_io_examples()
        advanced_usage_examples()
        
        print("🎉 All examples completed successfully!")
        print("\nThe PrismatronImage class provides:")
        print("• Unified image representation with format conversions")
        print("• Quality metrics (PSNR, SSIM, MSE, MAE)")
        print("• Image operations (resize, crop, thumbnail)")
        print("• Content analysis (histograms, brightness stats)")
        print("• File I/O with multiple backend support")
        print("• Camera capture integration")
        print("• Robust error handling and validation")
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()