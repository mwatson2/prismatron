#!/usr/bin/env python3
"""
LED position and block position utilities.

This module extracts the LED positioning logic for analysis and testing.
"""

from typing import Tuple

import numpy as np


def generate_random_led_positions(
    led_count: int, frame_width: int = 800, frame_height: int = 640, seed: int = None
) -> np.ndarray:
    """
    Generate random LED positions (realistic hardware layout simulation).

    Args:
        led_count: Number of LEDs to position
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        seed: Random seed for reproducibility

    Returns:
        Array of LED positions (led_count, 2) with [x, y] coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate completely random positions within frame bounds
    # This simulates realistic hardware where LED positions are fixed
    margin = 20  # Small margin to avoid edge effects
    width_range = (margin, frame_width - margin)
    height_range = (margin, frame_height - margin)

    # Uniform random distribution across the frame
    x_positions = np.random.randint(width_range[0], width_range[1], led_count)
    y_positions = np.random.randint(height_range[0], height_range[1], led_count)

    positions = np.column_stack((x_positions, y_positions))
    return positions


def calculate_block_positions(
    led_positions: np.ndarray,
    block_size: int,
    frame_width: int = 800,
    frame_height: int = 640,
) -> np.ndarray:
    """
    Calculate block positions from LED positions with CUDA alignment constraints.

    Args:
        led_positions: Array of LED positions (led_count, 2) with [x, y] coordinates
        block_size: Size of square blocks (e.g., 64 for 64x64 blocks)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Array of block positions (led_count, 2) with [x, y] top-left corner coordinates
    """
    block_positions = []

    for led_id, (x, y) in enumerate(led_positions):
        # Calculate block top-left corner centered on LED
        block_x_candidate = max(0, min(frame_width - block_size, x - block_size // 2))
        block_y = max(0, min(frame_height - block_size, y - block_size // 2))

        # CRITICAL: Round x-coordinate down to multiple of 4 for CUDA kernel alignment
        block_x = (block_x_candidate // 4) * 4

        block_positions.append([block_x, block_y])

    return np.array(block_positions)


def generate_adjacency_matrix(
    block_positions: np.ndarray, block_size: int
) -> np.ndarray:
    """
    Generate adjacency matrix showing which LEDs have overlapping blocks.

    This creates the same adjacency pattern as A^T A would have, where
    A^T A[i,j] != 0 if LED i and LED j have overlapping diffusion patterns.

    Args:
        block_positions: Array of block positions (led_count, 2)
        block_size: Size of square blocks

    Returns:
        Dense binary adjacency matrix (led_count, led_count) where 1 = overlap
    """
    n_leds = len(block_positions)
    adjacency = np.zeros((n_leds, n_leds), dtype=np.int32)

    # Check all pairs for block overlap
    for i in range(n_leds):
        for j in range(i, n_leds):  # Only check upper triangle + diagonal
            if i == j:
                adjacency[i, j] = 1  # LED always overlaps with itself
                continue

            pos1_x, pos1_y = block_positions[i]
            pos2_x, pos2_y = block_positions[j]

            # Block extents: [top_left, top_left + block_size]
            x1_min, x1_max = pos1_x, pos1_x + block_size
            y1_min, y1_max = pos1_y, pos1_y + block_size
            x2_min, x2_max = pos2_x, pos2_x + block_size
            y2_min, y2_max = pos2_y, pos2_y + block_size

            # Compute overlap dimensions
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

            # Blocks are adjacent if they have any overlap
            if x_overlap > 0 and y_overlap > 0:
                adjacency[i, j] = 1
                adjacency[j, i] = 1  # Symmetric

    return adjacency


def analyze_matrix_bandwidth(matrix: np.ndarray) -> dict:
    """
    Analyze the bandwidth properties of a matrix.

    Args:
        matrix: Square matrix to analyze

    Returns:
        Dictionary with bandwidth statistics
    """
    n = matrix.shape[0]
    if matrix.shape[1] != n:
        raise ValueError("Matrix must be square")

    # Find all non-zero positions
    rows, cols = np.nonzero(matrix)

    if len(rows) == 0:
        return {
            "nnz": 0,
            "sparsity": 0.0,
            "bandwidth": 0,
            "max_diagonal_offset": 0,
            "num_diagonals": 0,
            "diagonal_offsets": [],
        }

    # Calculate diagonal offsets: offset = col - row
    diagonal_offsets = cols - rows
    unique_offsets = np.unique(diagonal_offsets)

    # Bandwidth is the maximum distance from main diagonal
    bandwidth = max(abs(diagonal_offsets.min()), abs(diagonal_offsets.max()))

    # Count non-zeros
    nnz = len(rows)
    sparsity = nnz / (n * n) * 100

    return {
        "nnz": nnz,
        "sparsity": sparsity,
        "bandwidth": bandwidth,
        "max_diagonal_offset": int(
            max(abs(diagonal_offsets.min()), abs(diagonal_offsets.max()))
        ),
        "num_diagonals": len(unique_offsets),
        "diagonal_offsets": sorted(unique_offsets.tolist()),
        "matrix_size": n,
    }
