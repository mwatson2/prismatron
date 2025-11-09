#!/usr/bin/env python3
"""
Visualize LED effect templates by displaying all frames in a grid.

Loads a .npy template file and creates a grid of thumbnail images showing
each frame of the animation.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def visualize_template(
    template_path: Path,
    output_path: Path = None,
    cols: int = None,
    figsize_per_frame: float = 2.0,
    cmap: str = "viridis",
    show_frame_numbers: bool = True,
    show_stats: bool = True,
):
    """
    Visualize a template file as a grid of frames.

    Args:
        template_path: Path to the .npy template file
        output_path: Path to save the visualization (default: same name as template with .png)
        cols: Number of columns in the grid (default: auto-calculate for roughly square grid)
        figsize_per_frame: Size of each frame thumbnail in inches
        cmap: Matplotlib colormap to use
        show_frame_numbers: Whether to show frame numbers on each thumbnail
        show_stats: Whether to show statistics in the title
    """
    # Load the template
    print(f"Loading template from: {template_path}")
    template = np.load(template_path)

    if template.ndim != 3:
        raise ValueError(f"Expected 3D array (frames, height, width), got shape {template.shape}")

    num_frames, height, width = template.shape
    print(f"Template shape: {template.shape}")
    print(f"Dtype: {template.dtype}")
    print(f"Value range: [{template.min():.6f}, {template.max():.6f}]")

    # Calculate grid layout
    if cols is None:
        # Try to make a roughly square grid
        cols = math.ceil(math.sqrt(num_frames))
    rows = math.ceil(num_frames / cols)

    print(f"Creating {rows}x{cols} grid for {num_frames} frames")

    # Create figure
    fig_width = cols * figsize_per_frame
    fig_height = rows * figsize_per_frame
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Handle single row/column cases
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each frame
    for idx in range(num_frames):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Display frame
        im = ax.imshow(template[idx], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")

        # Add frame number
        if show_frame_numbers:
            # Calculate statistics for this frame
            frame_max = template[idx].max()
            frame_mean = template[idx].mean()
            frame_nonzero = np.count_nonzero(template[idx] > 0.01)

            ax.text(
                0.05,
                0.95,
                f"Frame {idx}",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

            # Add stats text at bottom
            if show_stats:
                stats_text = f"max={frame_max:.2f}\nmean={frame_mean:.3f}\npx>{0.01}={frame_nonzero}"
                ax.text(
                    0.05,
                    0.05,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=6,
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )

    # Hide unused subplots
    for idx in range(num_frames, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation="horizontal", pad=0.02, aspect=40, label="Intensity")

    # Add overall title
    title = f"Template: {template_path.name}\n"
    title += f"Shape: {template.shape}, Dtype: {template.dtype}, "
    title += f"Range: [{template.min():.3f}, {template.max():.3f}]"
    fig.suptitle(title, fontsize=12, y=0.98)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if output_path:
        print(f"Saving visualization to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved successfully (size: {output_path.stat().st_size / (1024*1024):.2f} MB)")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LED effect templates as a grid of frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "template",
        type=str,
        help="Path to the .npy template file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output image path (default: same as template with .png extension)",
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of columns in grid (default: auto)",
    )

    parser.add_argument(
        "--frame-size",
        type=float,
        default=2.0,
        help="Size of each frame thumbnail in inches",
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap (viridis, hot, gray, etc.)",
    )

    parser.add_argument(
        "--no-frame-numbers",
        action="store_true",
        help="Don't show frame numbers on thumbnails",
    )

    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Don't show statistics on thumbnails",
    )

    args = parser.parse_args()

    # Parse paths
    template_path = Path(args.template)
    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = template_path.with_suffix(".png")

    # Create visualization
    try:
        visualize_template(
            template_path=template_path,
            output_path=output_path,
            cols=args.cols,
            figsize_per_frame=args.frame_size,
            cmap=args.cmap,
            show_frame_numbers=not args.no_frame_numbers,
            show_stats=not args.no_stats,
        )
        return 0
    except Exception as e:
        print(f"Error visualizing template: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
