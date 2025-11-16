#!/usr/bin/env python3
"""
Generate a growing heart-shaped template for LED effects.

Creates a numpy array with shape (frames, width, height) containing a heart
that grows from the center to the edge of the screen. Each pixel's intensity
is calculated based on its distance from the heart outline using a sine falloff.

Heart parametric equation (point down):
x = 16 * sin(t)^3
y = -(13 * cos(t) - 5 * cos(2t) - 2 * cos(3t) - cos(4t))

Output is saved as a .npy file with fp16 precision.
"""

import argparse
from pathlib import Path

import numpy as np


def heart_distance_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate signed distance to heart shape boundary.

    Uses parametric heart equation and samples it to compute distance.
    Heart is normalized to unit size, centered at origin.

    Args:
        x: X coordinates (normalized, centered at 0)
        y: Y coordinates (normalized, centered at 0)

    Returns:
        Distance to heart boundary (positive = outside, negative = inside)
    """
    # Sample heart parametric curve
    t = np.linspace(0, 2 * np.pi, 1000)
    heart_x = 16 * np.sin(t) ** 3
    heart_y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))

    # Normalize to unit size (heart height is about 26 units)
    heart_scale = 26.0
    heart_x = heart_x / heart_scale
    heart_y = heart_y / heart_scale

    # Calculate distance from each pixel to all heart boundary points
    # Process in chunks to avoid memory issues
    x_flat = x.flatten()
    y_flat = y.flatten()
    distances = np.zeros(len(x_flat), dtype=np.float32)

    chunk_size = 10000  # Process 10k pixels at a time
    for i in range(0, len(x_flat), chunk_size):
        end_idx = min(i + chunk_size, len(x_flat))
        x_chunk = x_flat[i:end_idx, np.newaxis]  # Shape: (chunk, 1)
        y_chunk = y_flat[i:end_idx, np.newaxis]  # Shape: (chunk, 1)

        # Broadcast heart points
        dx = heart_x[np.newaxis, :] - x_chunk  # Shape: (chunk, 1000)
        dy = heart_y[np.newaxis, :] - y_chunk  # Shape: (chunk, 1000)
        dist_to_boundary = np.sqrt(dx**2 + dy**2)  # Shape: (chunk, 1000)

        # Find minimum distance for each pixel in chunk
        distances[i:end_idx] = np.min(dist_to_boundary, axis=1)

    return distances.reshape(x.shape)


def generate_heart_frame(width: int, height: int, radius: float, outline_width: float) -> np.ndarray:
    """
    Generate a single frame of a heart outline at the specified scale.

    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        radius: Scale factor for heart size (larger = bigger heart)
        outline_width: Width of the outline in pixels

    Returns:
        Array of shape (height, width) with intensity values [0, 1]
    """
    # Create coordinate grids centered at image center
    center_x = width / 2.0
    center_y = height / 2.0

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Convert to normalized coordinates relative to center
    # Normalize by radius to scale the heart
    x_norm = (x - center_x) / max(1.0, radius)
    y_norm = (y - center_y) / max(1.0, radius)

    # Calculate distance to heart boundary
    dist_to_heart = heart_distance_field(x_norm, y_norm)

    # Convert distance back to pixels
    dist_pixels = dist_to_heart * radius

    # Initialize intensity array
    intensity = np.zeros((height, width), dtype=np.float32)

    # Create outline using sine falloff
    # Pixels within outline_width of the boundary get intensity based on sine
    mask = dist_pixels <= outline_width

    # Calculate intensity using sine falloff
    # intensity = sin((outline_width - dist) * pi / (2 * outline_width))
    # This gives 1.0 at the boundary and fades to 0 at outline_width away
    intensity[mask] = np.sin((outline_width - dist_pixels[mask]) * np.pi / (2 * outline_width))

    return intensity


def generate_heart_animation(
    num_frames: int,
    width: int,
    height: int,
    outline_width: float,
    start_radius: float = 10.0,
    end_radius: float = None,
) -> np.ndarray:
    """
    Generate a growing heart animation.

    Args:
        num_frames: Number of frames in the animation
        width: Frame width in pixels
        height: Frame height in pixels
        outline_width: Width of the outline in pixels
        start_radius: Starting scale factor (default: 10.0)
        end_radius: Ending scale factor (default: diagonal + heart size + outline to go fully off screen)

    Returns:
        Array of shape (num_frames, height, width) with fp16 precision
    """
    # Calculate default end radius if not specified
    # Need to go beyond corner by heart size (1.0 normalized units) plus outline
    if end_radius is None:
        center_x = width / 2.0
        center_y = height / 2.0
        diagonal_to_corner = np.sqrt(center_x**2 + center_y**2)
        # Heart has normalized size of 1.0, so add this to diagonal
        # Also add outline width to ensure outline is completely off screen
        end_radius = diagonal_to_corner + diagonal_to_corner * 1.0 + outline_width

    # Generate array to hold all frames
    animation = np.zeros((num_frames, height, width), dtype=np.float16)

    # Generate each frame with linearly increasing scale
    for frame_idx in range(num_frames):
        # Linear interpolation from start to end radius
        t = frame_idx / max(1, num_frames - 1)
        radius = start_radius + t * (end_radius - start_radius)

        # Generate the heart at this scale
        frame = generate_heart_frame(width, height, radius, outline_width)
        animation[frame_idx] = frame.astype(np.float16)

        print(f"Generated frame {frame_idx+1}/{num_frames} (scale={radius:.2f})")

    return animation


def main():
    parser = argparse.ArgumentParser(
        description="Generate a growing heart-shaped template for LED effects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of frames in the animation",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Frame width in pixels",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height in pixels",
    )

    parser.add_argument(
        "--outline-width",
        type=float,
        default=40.0,
        help="Width of the outline in pixels",
    )

    parser.add_argument(
        "--start-radius",
        type=float,
        default=10.0,
        help="Starting scale factor",
    )

    parser.add_argument(
        "--end-radius",
        type=float,
        default=None,
        help="Ending scale factor (default: distance to corner)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="heart_template.npy",
        help="Output filename (.npy)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="templates",
        help="Output directory",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the animation
    print(f"Generating {args.frames} frames of {args.width}x{args.height} heart animation...")
    print(f"Outline width: {args.outline_width} pixels")
    print(f"Scale range: {args.start_radius} to {args.end_radius or 'corner'}")

    animation = generate_heart_animation(
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        outline_width=args.outline_width,
        start_radius=args.start_radius,
        end_radius=args.end_radius,
    )

    # Save to file
    output_path = output_dir / args.output
    np.save(output_path, animation)

    print(f"\nSaved animation to: {output_path}")
    print(f"Shape: {animation.shape}")
    print(f"Dtype: {animation.dtype}")
    print(f"Value range: [{animation.min():.4f}, {animation.max():.4f}]")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
