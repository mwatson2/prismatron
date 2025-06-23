#!/usr/bin/env python3
"""
Analyze existing diffusion patterns to determine optimal block size.

This script examines the synthetic_1000.npz patterns to find:
1. The actual extent (bounding box) of each LED's diffusion pattern
2. The optimal block size needed to capture all non-zero elements
3. The density distribution of the patterns
"""

import logging
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_diffusion_patterns(patterns_path):
    """Load diffusion patterns from npz file."""
    logger.info(f"Loading patterns from {patterns_path}")

    data = np.load(patterns_path)
    logger.info(f"Available arrays: {list(data.keys())}")

    # Load the sparse matrix
    if "matrix_data" in data:
        # Sparse CSC format
        matrix_data = data["matrix_data"]
        matrix_indices = data["matrix_indices"]
        matrix_indptr = data["matrix_indptr"]
        matrix_shape = tuple(data["matrix_shape"])

        # Reconstruct sparse matrix
        matrix = sp.csc_matrix(
            (matrix_data, matrix_indices, matrix_indptr),
            shape=matrix_shape,
            dtype=np.float32,
        )

        logger.info(f"Loaded sparse matrix: {matrix.shape}")
        logger.info(f"Non-zeros: {matrix.nnz:,}")
        logger.info(
            f"Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.3f}%"
        )

        return matrix, data
    else:
        raise ValueError("Could not find sparse matrix data in file")


def analyze_led_extents(matrix, frame_height=480, frame_width=800):
    """Analyze the spatial extent of each LED's diffusion pattern."""
    logger.info("Analyzing LED pattern extents...")

    # Matrix shape: (pixels, leds*3) where pixels = height * width
    pixels, led_channels = matrix.shape
    led_count = led_channels // 3

    logger.info(f"Matrix shape: {matrix.shape}")
    logger.info(f"Detected LED count: {led_count}")
    logger.info(f"Expected pixels: {frame_height * frame_width}")

    extents = []
    pattern_sizes = []
    densities = []

    for led_id in range(led_count):
        for channel in range(3):
            col_idx = channel * led_count + led_id

            # Get column for this LED/channel
            col_data = matrix[:, col_idx]
            nonzero_rows = col_data.nonzero()[0]

            if len(nonzero_rows) == 0:
                continue

            # Convert pixel indices to (row, col) coordinates
            pixel_rows = nonzero_rows // frame_width
            pixel_cols = nonzero_rows % frame_width

            # Find bounding box
            min_row, max_row = pixel_rows.min(), pixel_rows.max()
            min_col, max_col = pixel_cols.min(), pixel_cols.max()

            height_extent = max_row - min_row + 1
            width_extent = max_col - min_col + 1
            max_extent = max(height_extent, width_extent)

            pattern_area = len(nonzero_rows)
            bounding_box_area = height_extent * width_extent
            pattern_density = (
                pattern_area / bounding_box_area if bounding_box_area > 0 else 0
            )

            extents.append(
                {
                    "led_id": led_id,
                    "channel": channel,
                    "min_row": min_row,
                    "max_row": max_row,
                    "min_col": min_col,
                    "max_col": max_col,
                    "height_extent": height_extent,
                    "width_extent": width_extent,
                    "max_extent": max_extent,
                    "pattern_area": pattern_area,
                    "bounding_box_area": bounding_box_area,
                    "pattern_density": pattern_density,
                }
            )

            pattern_sizes.append(pattern_area)
            densities.append(pattern_density)

    return extents, pattern_sizes, densities


def analyze_block_size_requirements(extents):
    """Determine optimal block size to capture all patterns."""
    logger.info("Analyzing block size requirements...")

    max_extents = [e["max_extent"] for e in extents]
    height_extents = [e["height_extent"] for e in extents]
    width_extents = [e["width_extent"] for e in extents]

    logger.info(f"Max extent statistics:")
    logger.info(f"  Min: {min(max_extents)}")
    logger.info(f"  Max: {max(max_extents)}")
    logger.info(f"  Mean: {np.mean(max_extents):.1f}")
    logger.info(f"  Median: {np.median(max_extents):.1f}")
    logger.info(f"  95th percentile: {np.percentile(max_extents, 95):.1f}")
    logger.info(f"  99th percentile: {np.percentile(max_extents, 99):.1f}")

    logger.info(f"Height extent statistics:")
    logger.info(f"  Min: {min(height_extents)}")
    logger.info(f"  Max: {max(height_extents)}")
    logger.info(f"  Mean: {np.mean(height_extents):.1f}")
    logger.info(f"  99th percentile: {np.percentile(height_extents, 99):.1f}")

    logger.info(f"Width extent statistics:")
    logger.info(f"  Min: {min(width_extents)}")
    logger.info(f"  Max: {max(width_extents)}")
    logger.info(f"  Mean: {np.mean(width_extents):.1f}")
    logger.info(f"  99th percentile: {np.percentile(width_extents, 99):.1f}")

    # Suggested block sizes
    block_sizes = [32, 64, 96, 128, 160, 192, 224, 256]

    logger.info("Block size coverage analysis:")
    for block_size in block_sizes:
        covered = sum(1 for extent in max_extents if extent <= block_size)
        coverage_pct = covered / len(max_extents) * 100
        logger.info(
            f"  Block size {block_size:3d}: covers {covered:4d}/{len(max_extents)} patterns ({coverage_pct:5.1f}%)"
        )

    return max_extents, height_extents, width_extents


def analyze_pattern_density(extents, pattern_sizes):
    """Analyze the density distribution of patterns."""
    logger.info("Analyzing pattern density...")

    densities = [e["pattern_density"] for e in extents]

    logger.info(f"Pattern density statistics:")
    logger.info(f"  Min: {min(densities):.3f}")
    logger.info(f"  Max: {max(densities):.3f}")
    logger.info(f"  Mean: {np.mean(densities):.3f}")
    logger.info(f"  Median: {np.median(densities):.3f}")

    logger.info(f"Pattern size statistics:")
    logger.info(f"  Min: {min(pattern_sizes)}")
    logger.info(f"  Max: {max(pattern_sizes)}")
    logger.info(f"  Mean: {np.mean(pattern_sizes):.1f}")
    logger.info(f"  Median: {np.median(pattern_sizes):.1f}")

    return densities


def recommend_block_size(max_extents):
    """Recommend optimal block size based on analysis."""
    logger.info("Block size recommendations:")

    # Find smallest block size that covers 95% and 99% of patterns
    percentiles = [90, 95, 99, 99.5]

    for pct in percentiles:
        required_size = np.percentile(max_extents, pct)

        # Round up to next power of 2 or common block size
        common_sizes = [32, 64, 96, 128, 160, 192, 224, 256]
        recommended = next(
            (size for size in common_sizes if size >= required_size), 256
        )

        logger.info(
            f"  {pct:4.1f}% coverage: needs {required_size:.1f}, recommend {recommended}"
        )

    # Final recommendation
    p99_size = np.percentile(max_extents, 99)
    if p99_size <= 64:
        recommendation = 64
    elif p99_size <= 96:
        recommendation = 96
    elif p99_size <= 128:
        recommendation = 128
    else:
        recommendation = 160

    coverage = sum(1 for extent in max_extents if extent <= recommendation)
    coverage_pct = coverage / len(max_extents) * 100

    logger.info(f"\nFINAL RECOMMENDATION: Block size {recommendation}")
    logger.info(
        f"  Covers {coverage}/{len(max_extents)} patterns ({coverage_pct:.1f}%)"
    )

    return recommendation


def main():
    """Run pattern extent analysis."""
    logger.info("Diffusion Pattern Extent Analysis")
    logger.info("=" * 50)

    patterns_path = "diffusion_patterns/synthetic_1000.npz"

    if not Path(patterns_path).exists():
        logger.error(f"Patterns file not found: {patterns_path}")
        return 1

    # Load patterns
    matrix, data = load_diffusion_patterns(patterns_path)

    # Analyze extents
    extents, pattern_sizes, densities = analyze_led_extents(matrix)

    logger.info(f"\nAnalyzed {len(extents)} LED/channel patterns")

    # Block size analysis
    max_extents, height_extents, width_extents = analyze_block_size_requirements(
        extents
    )

    # Density analysis
    pattern_densities = analyze_pattern_density(extents, pattern_sizes)

    # Recommendation
    recommended_block_size = recommend_block_size(max_extents)

    # Summary for next steps
    logger.info(f"\n" + "=" * 50)
    logger.info("SUMMARY FOR PERFORMANCE TESTING")
    logger.info("=" * 50)
    logger.info(
        f"Matrix density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.3f}%"
    )
    logger.info(f"Recommended block size: {recommended_block_size}")
    logger.info(f"Average pattern size: {np.mean(pattern_sizes):.1f} pixels")
    logger.info(f"Average pattern density within blocks: {np.mean(densities):.3f}")

    # Calculate expected block density
    avg_pattern_size = np.mean(pattern_sizes)
    block_area = recommended_block_size * recommended_block_size
    expected_block_density = avg_pattern_size / block_area

    logger.info(f"Expected block density: {expected_block_density:.3f}")
    logger.info(f"Expected non-zeros per block: {avg_pattern_size:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
