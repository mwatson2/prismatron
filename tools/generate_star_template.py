#!/usr/bin/env python3
"""
Generate a growing 7-pointed star template for LED effects.

Creates a numpy array with shape (frames, width, height) containing a 7-pointed star
that grows from the center until completely off-screen. Each pixel's intensity
is calculated based on its distance from the star outline using a sine falloff.

The star has:
- 7 points evenly distributed around a circle
- Outer corners at 2x the distance of inner corners from center
- Animation continues until the entire star is off-screen

Output is saved as a .npy file with fp16 precision.
"""

import argparse
from pathlib import Path

import numpy as np


def generate_star_points(num_points: int, outer_radius: float, inner_radius: float) -> tuple:
    """
    Generate coordinates for a star's vertices.

    Args:
        num_points: Number of star points
        outer_radius: Radius to outer (pointed) vertices
        inner_radius: Radius to inner (valley) vertices

    Returns:
        Tuple of (x_coords, y_coords) for all vertices
    """
    vertices_x = []
    vertices_y = []

    # Generate vertices alternating between outer and inner radii
    for i in range(num_points * 2):
        angle = i * np.pi / num_points - np.pi / 2  # Start at top, rotate clockwise

        if i % 2 == 0:
            # Outer point
            r = outer_radius
        else:
            # Inner point
            r = inner_radius

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        vertices_x.append(x)
        vertices_y.append(y)

    return np.array(vertices_x), np.array(vertices_y)


def point_to_line_segment_distance_vectorized(
    px: np.ndarray, py: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> np.ndarray:
    """
    Calculate distance from points (px, py) to line segment (x1,y1)-(x2,y2).
    Vectorized version for performance.

    Args:
        px, py: Point coordinate arrays
        x1, y1: Start of line segment
        x2, y2: End of line segment

    Returns:
        Array of distances from points to line segment
    """
    # Vector from line start to points
    dx = px - x1
    dy = py - y1

    # Vector of line segment
    lx = x2 - x1
    ly = y2 - y1

    # Length squared of line segment
    len_sq = lx * lx + ly * ly

    if len_sq < 1e-10:
        # Line segment is essentially a point
        return np.sqrt(dx * dx + dy * dy)

    # Project points onto line segment (clamped to [0, 1])
    t = np.clip((dx * lx + dy * ly) / len_sq, 0, 1)

    # Closest points on line segment
    closest_x = x1 + t * lx
    closest_y = y1 + t * ly

    # Distance to closest points
    return np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def star_distance_field(x: np.ndarray, y: np.ndarray, num_points: int, outer_radius: float) -> np.ndarray:
    """
    Calculate distance to star shape boundary.

    Args:
        x: X coordinates (centered at 0)
        y: Y coordinates (centered at 0)
        num_points: Number of star points
        outer_radius: Radius to outer points (inner = outer/2)

    Returns:
        Distance to star boundary
    """
    inner_radius = outer_radius / 2.0

    # Generate star vertices
    star_x, star_y = generate_star_points(num_points, outer_radius, inner_radius)

    # Calculate distance from each pixel to star outline
    distances = np.full_like(x, fill_value=np.inf, dtype=np.float32)

    # Check distance to each edge of the star (vectorized)
    num_vertices = len(star_x)
    for i in range(num_vertices):
        x1, y1 = star_x[i], star_y[i]
        x2, y2 = star_x[(i + 1) % num_vertices], star_y[(i + 1) % num_vertices]

        # Calculate distance from all points to this line segment (vectorized)
        dist = point_to_line_segment_distance_vectorized(x, y, x1, y1, x2, y2)
        distances = np.minimum(distances, dist)

    return distances


def generate_star_frame(
    width: int, height: int, radius: float, outline_width: float, num_points: int = 7
) -> np.ndarray:
    """
    Generate a single frame of a star outline at the specified radius.

    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        radius: Outer radius of star (inner radius = radius/2)
        outline_width: Width of the outline in pixels
        num_points: Number of star points (default: 7)

    Returns:
        Array of shape (height, width) with intensity values [0, 1]
    """
    # Create coordinate grids centered at image center
    center_x = width / 2.0
    center_y = height / 2.0

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Convert to coordinates relative to center
    x_rel = x - center_x
    y_rel = y - center_y

    # Calculate distance to star boundary
    dist_to_star = star_distance_field(x_rel, y_rel, num_points, radius)

    # Initialize intensity array
    intensity = np.zeros((height, width), dtype=np.float32)

    # Create outline using sine falloff
    # Pixels within outline_width of the boundary get intensity based on sine
    mask = dist_to_star <= outline_width

    # Calculate intensity using sine falloff
    # intensity = sin((outline_width - dist) * pi / (2 * outline_width))
    intensity[mask] = np.sin((outline_width - dist_to_star[mask]) * np.pi / (2 * outline_width))

    return intensity


def generate_star_animation(
    num_frames: int,
    width: int,
    height: int,
    outline_width: float,
    num_points: int = 7,
    start_radius: float = 0.0,
    end_radius: float = None,
) -> np.ndarray:
    """
    Generate a growing star animation.

    Args:
        num_frames: Number of frames in the animation
        width: Frame width in pixels
        height: Frame height in pixels
        outline_width: Width of the outline in pixels
        num_points: Number of star points (default: 7)
        start_radius: Starting outer radius (default: 0.0 for center)
        end_radius: Ending outer radius (default: enough to go off-screen)

    Returns:
        Array of shape (num_frames, height, width) with fp16 precision
    """
    # Calculate default end radius if not specified
    # Need to go far enough that outer points are completely off screen
    if end_radius is None:
        center_x = width / 2.0
        center_y = height / 2.0
        diagonal = np.sqrt(center_x**2 + center_y**2)
        # Add extra to ensure star is fully off-screen
        end_radius = diagonal + outline_width * 2

    # Generate array to hold all frames
    animation = np.zeros((num_frames, height, width), dtype=np.float16)

    # Generate each frame with linearly increasing radius
    for frame_idx in range(num_frames):
        # Linear interpolation from start to end radius
        t = frame_idx / max(1, num_frames - 1)
        radius = start_radius + t * (end_radius - start_radius)

        # Generate the star at this radius
        frame = generate_star_frame(width, height, radius, outline_width, num_points)
        animation[frame_idx] = frame.astype(np.float16)

        print(f"Generated frame {frame_idx+1}/{num_frames} (radius={radius:.2f})")

    return animation


def main():
    parser = argparse.ArgumentParser(
        description="Generate a growing 7-pointed star template for LED effects",
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
        "--num-points",
        type=int,
        default=7,
        help="Number of star points",
    )

    parser.add_argument(
        "--start-radius",
        type=float,
        default=0.0,
        help="Starting outer radius (0 = center)",
    )

    parser.add_argument(
        "--end-radius",
        type=float,
        default=None,
        help="Ending outer radius (default: fully off-screen)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="star_template.npy",
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
    print(f"Generating {args.frames} frames of {args.width}x{args.height} {args.num_points}-pointed star animation...")
    print(f"Outline width: {args.outline_width} pixels")
    print(f"Radius range: {args.start_radius} to {args.end_radius or 'off-screen'}")

    animation = generate_star_animation(
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        outline_width=args.outline_width,
        num_points=args.num_points,
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
