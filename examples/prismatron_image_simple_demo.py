#!/usr/bin/env python3
"""
Simple PrismatronImage Demo

Direct usage demonstration without package import issues.
"""

import sys
import numpy as np
from pathlib import Path

# Direct import to avoid package issues
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "utils"))
import prismatron_image

def main():
    """Simple demonstration of PrismatronImage functionality."""
    print("PrismatronImage Simple Demo")
    print("=" * 40)
    print()
    
    # Basic creation
    print("1. Creating images:")
    black = prismatron_image.PrismatronImage.zeros(100, 80)
    white = prismatron_image.PrismatronImage.ones(100, 80)
    color = prismatron_image.PrismatronImage.solid_color(100, 80, (255, 128, 64))
    print(f"   Black: {black.width}x{black.height}")
    print(f"   White: {white.width}x{white.height}")
    print(f"   Color: {color.width}x{color.height}")
    
    # Format conversions
    print("\n2. Format conversions:")
    planar = color.as_planar()
    interleaved = color.as_interleaved()
    flat = color.as_flat_interleaved()
    print(f"   Planar (canonical): {planar.shape}")
    print(f"   Interleaved: {interleaved.shape}")
    print(f"   Flat array: {flat.shape}")
    
    # Operations
    print("\n3. Image operations:")
    resized = color.resize(50, 40)
    cropped = color.crop(25, 20, 50, 40)
    thumb = color.thumbnail(60)
    print(f"   Resized: {resized.width}x{resized.height}")
    print(f"   Cropped: {cropped.width}x{cropped.height}")
    print(f"   Thumbnail: {thumb.width}x{thumb.height}")
    
    # Quality metrics
    print("\n4. Quality comparison:")
    similar = prismatron_image.PrismatronImage.solid_color(100, 80, (250, 125, 60))
    metrics = color.compare(similar)
    print(f"   PSNR: {metrics['psnr']:.2f} dB")
    print(f"   SSIM: {metrics['ssim']:.4f}")
    print(f"   MSE: {metrics['mse']:.2f}")
    
    # Analysis
    print("\n5. Image analysis:")
    stats = color.brightness_stats()
    print(f"   Red mean: {stats['red']['mean']:.1f}")
    print(f"   Green mean: {stats['green']['mean']:.1f}")
    print(f"   Blue mean: {stats['blue']['mean']:.1f}")
    
    com = color.center_of_mass()
    print(f"   Center of mass: ({com[0]:.1f}, {com[1]:.1f})")
    
    # Test Prismatron dimensions
    print("\n6. Prismatron format recognition:")
    pixels = prismatron_image.PRISMATRON_WIDTH * prismatron_image.PRISMATRON_HEIGHT
    led_data = np.random.randint(0, 256, pixels * 3, dtype=np.uint8)
    prismatron_img = prismatron_image.PrismatronImage(led_data, "flat_interleaved")
    print(f"   Prismatron size: {prismatron_img.width}x{prismatron_img.height}")
    
    print("\n✓ Demo completed successfully!")
    print("\nPrismatronImage supports:")
    print("• Multiple format conversions (planar, interleaved, flat)")
    print("• Image operations (resize, crop, thumbnail)")
    print("• Quality metrics (PSNR, SSIM, MSE, MAE)")
    print("• Image analysis (histograms, statistics)")
    print("• File I/O with backend fallback")
    print("• Camera capture integration")

if __name__ == "__main__":
    main()