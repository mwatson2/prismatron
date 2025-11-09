#!/usr/bin/env python3
"""
Generate a growing ring template for LED effects.

Creates a numpy array with shape (frames, width, height) containing a ring
that grows from the center to the edge of the screen. Each pixel's intensity
is calculated based on its distance from the ring using a sine falloff.

Output is saved as a .npy file with fp16 precision.
"""

import argparse
from pathlib import Path

import numpy as np


def generate_ring_frame(width: int, height: int, radius: float, ring_width: float) -> np.ndarray:
    """
    Generate a single frame of a ring at the specified radius.

    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        radius: Ring radius (R)
        ring_width: Width of the ring (w)

    Returns:
        Array of shape (height, width) with intensity values [0, 1]
    """
    # Create coordinate grids centered at image center
    center_x = width / 2.0
    center_y = height / 2.0

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Convert to polar coordinates - calculate distance from center
    dx = x - center_x
    dy = y - center_y
    r = np.sqrt(dx**2 + dy**2)

    # Initialize intensity array
    intensity = np.zeros((height, width), dtype=np.float32)

    # Calculate ring bounds
    inner_radius = radius - ring_width
    outer_radius = radius + ring_width

    # For pixels within the ring (R-w <= r <= R+w)
    # intensity = sin((r-(R-w))*pi/(2*w))
    mask = (r >= inner_radius) & (r <= outer_radius)

    # Calculate intensity using sine falloff
    # Formula: sin((r-(R-w))*pi/(2*w))
    intensity[mask] = np.sin((r[mask] - inner_radius) * np.pi / (2 * ring_width))

    return intensity


def generate_ring_animation(
    num_frames: int,
    width: int,
    height: int,
    ring_width: float,
    start_radius: float = 0.0,
    end_radius: float = None,
) -> np.ndarray:
    """
    Generate a growing ring animation.

    Args:
        num_frames: Number of frames in the animation
        width: Frame width in pixels
        height: Frame height in pixels
        ring_width: Width of the ring in pixels
        start_radius: Starting radius (default: 0.0 for center)
        end_radius: Ending radius (default: diagonal distance to corner)

    Returns:
        Array of shape (num_frames, height, width) with fp16 precision
    """
    # Calculate default end radius if not specified (distance to corner)
    if end_radius is None:
        center_x = width / 2.0
        center_y = height / 2.0
        end_radius = np.sqrt(center_x**2 + center_y**2)

    # Generate array to hold all frames
    animation = np.zeros((num_frames, height, width), dtype=np.float16)

    # Generate each frame with linearly increasing radius
    for frame_idx in range(num_frames):
        # Linear interpolation from start to end radius
        t = frame_idx / max(1, num_frames - 1)
        radius = start_radius + t * (end_radius - start_radius)

        # Generate the ring at this radius
        frame = generate_ring_frame(width, height, radius, ring_width)
        animation[frame_idx] = frame.astype(np.float16)

        print(f"Generated frame {frame_idx+1}/{num_frames} (radius={radius:.2f})")

    return animation


def main():
    parser = argparse.ArgumentParser(
        description="Generate a growing ring template for LED effects",
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
        "--ring-width",
        type=float,
        default=40.0,
        help="Width of the ring in pixels",
    )

    parser.add_argument(
        "--start-radius",
        type=float,
        default=0.0,
        help="Starting radius (0 = center)",
    )

    parser.add_argument(
        "--end-radius",
        type=float,
        default=None,
        help="Ending radius (default: distance to corner)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="ring_template.npy",
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
    print(f"Generating {args.frames} frames of {args.width}x{args.height} ring animation...")
    print(f"Ring width: {args.ring_width} pixels")
    print(f"Radius range: {args.start_radius} to {args.end_radius or 'corner'}")

    animation = generate_ring_animation(
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        ring_width=args.ring_width,
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
