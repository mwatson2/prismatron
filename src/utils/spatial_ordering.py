#!/usr/bin/env python3
"""
Spatial ordering utilities for LED optimization.

This module provides utilities for computing spatial orderings of blocks/LEDs
to optimize matrix bandwidth and locality in optimization algorithms.
"""

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.spatial import cKDTree


def compute_rcm_ordering(block_positions: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute Reverse Cuthill-McKee (RCM) ordering for spatial blocks.

    This function computes an RCM ordering based on spatial adjacency of blocks
    (e.g., LED crop regions) to minimize matrix bandwidth and improve cache locality.

    Args:
        block_positions: Array of block center positions, shape (n_blocks, 2)
        block_size: Size of each block (assumed square, e.g., crop size for LEDs)

    Returns:
        Tuple of:
        - rcm_order: Array of block indices in RCM order, shape (n_blocks,)
        - inverse_order: Inverse permutation to map from RCM back to original order, shape (n_blocks,)
        - adjacency_diagonals: Number of diagonals in the RCM-ordered adjacency matrix
    """
    n_blocks = len(block_positions)

    if n_blocks == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0

    if n_blocks == 1:
        return np.array([0], dtype=np.int32), np.array([0], dtype=np.int32), 1

    print(f"Computing RCM ordering for {n_blocks} blocks with size {block_size}...")

    # Build adjacency graph using KD-tree for efficiency
    tree = cKDTree(block_positions)

    # Conservative distance threshold - blocks that could potentially overlap
    max_distance = block_size * np.sqrt(2)  # Diagonal distance
    pairs = tree.query_pairs(max_distance)

    print(f"  Found {len(pairs)} potential block pairs within distance {max_distance:.1f}")

    # Build sparse adjacency matrix
    adjacency = sp.lil_matrix((n_blocks, n_blocks), dtype=bool)

    overlaps = 0
    for i, j in pairs:
        pos1_x, pos1_y = block_positions[i]
        pos2_x, pos2_y = block_positions[j]

        # Check actual block region overlap
        # Block extents: [center - size/2, center + size/2]
        half_size = block_size // 2

        x1_min, x1_max = pos1_x - half_size, pos1_x + half_size
        y1_min, y1_max = pos1_y - half_size, pos1_y + half_size
        x2_min, x2_max = pos2_x - half_size, pos2_x + half_size
        y2_min, y2_max = pos2_y - half_size, pos2_y + half_size

        # Compute overlap dimensions
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

        # Blocks are adjacent if they have any overlap
        if x_overlap > 0 and y_overlap > 0:
            adjacency[i, j] = True
            adjacency[j, i] = True  # Symmetric
            overlaps += 1

    print(f"  Found {overlaps} actual block overlaps")

    # Apply RCM algorithm
    adjacency_csr = adjacency.tocsr()

    if adjacency_csr.nnz == 0:
        # No adjacencies - return original order
        print("  No adjacencies found, using original order")
        rcm_order = np.arange(n_blocks, dtype=np.int32)
        adjacency_diagonals = 1  # Only main diagonal
    else:
        print("  Applying RCM algorithm...")
        rcm_order = reverse_cuthill_mckee(adjacency_csr, symmetric_mode=True).astype(np.int32)

        # Compute diagonal count after RCM reordering
        adjacency_rcm = adjacency[rcm_order][:, rcm_order]
        rows, cols = adjacency_rcm.nonzero()
        if len(rows) > 0:
            diagonal_offsets = cols - rows
            unique_diagonals = np.unique(diagonal_offsets)
            adjacency_diagonals = len(unique_diagonals)
            print(f"  Adjacency matrix diagonal range: [{diagonal_offsets.min()}, {diagonal_offsets.max()}]")
            print(f"  Adjacency matrix nnz: {len(rows):,}")
            print(f"  Adjacency matrix diagonals present: {sorted(unique_diagonals.tolist())}")
            # Check if main diagonal (0) is present
            if 0 in unique_diagonals:
                print("  Main diagonal (0) is present in adjacency matrix")
            else:
                print("  Main diagonal (0) is MISSING from adjacency matrix")
        else:
            adjacency_diagonals = 0

    # Compute inverse permutation: inverse_order[rcm_order[i]] = i
    inverse_order = np.argsort(rcm_order).astype(np.int32)

    print("  RCM ordering computed successfully")
    print(f"  Original order range: [0, {n_blocks - 1}]")
    print(f"  RCM order range: [{rcm_order.min()}, {rcm_order.max()}]")
    print(f"  RCM-ordered adjacency diagonals: {adjacency_diagonals}")

    return rcm_order, inverse_order, adjacency_diagonals


def reorder_matrix_columns(matrix: sp.spmatrix, block_order: np.ndarray, channels_per_block: int = 3) -> sp.spmatrix:
    """
    Reorder matrix columns according to block ordering.

    This function reorders the columns of a matrix (e.g., diffusion matrix A)
    according to a spatial block ordering. Each block has multiple channels
    (e.g., RGB channels for LEDs).

    Args:
        matrix: Sparse matrix to reorder, shape (rows, blocks * channels_per_block)
        block_order: Block ordering array, shape (blocks,)
        channels_per_block: Number of channels per block (default: 3 for RGB)

    Returns:
        Reordered matrix with columns permuted according to block_order
    """
    n_blocks = len(block_order)
    expected_cols = n_blocks * channels_per_block

    if matrix.shape[1] != expected_cols:
        raise ValueError(
            f"Matrix should have {expected_cols} columns ({n_blocks} blocks × {channels_per_block} channels), "
            f"got {matrix.shape[1]}"
        )

    # Create column permutation: for each block in the new order,
    # include all its channels (e.g., R, G, B for LEDs)
    col_permutation_list = []
    for block_idx in block_order:
        for channel in range(channels_per_block):
            col_permutation_list.append(block_idx * channels_per_block + channel)

    col_permutation = np.array(col_permutation_list, dtype=np.int32)

    # Convert to CSC format for efficient column slicing if needed
    if not hasattr(matrix, "__getitem__") or matrix.format == "coo":
        matrix = matrix.tocsc()

    # Reorder matrix columns
    return matrix[:, col_permutation]


def reorder_block_values(values: np.ndarray, block_order: np.ndarray, from_ordered: bool = False) -> np.ndarray:
    """
    Reorder block values according to spatial ordering.

    Args:
        values: Block values array, shape (channels, blocks) or (blocks, channels)
        block_order: Block ordering array, shape (blocks,)
        from_ordered: If True, reorder from spatial order back to original order

    Returns:
        Reordered values array with same shape as input
    """
    if values.ndim == 2:
        channels, blocks = values.shape
        if blocks != len(block_order):
            # Try swapped dimensions
            if channels == len(block_order):
                values = values.T
                channels, blocks = values.shape
                swapped = True
            else:
                raise ValueError(
                    f"Values array dimension mismatch: shape {values.shape}, "
                    f"expected one dimension to be {len(block_order)}"
                )
        else:
            swapped = False
    else:
        raise ValueError(f"Values array should be 2D, got shape {values.shape}")

    # Choose ordering direction
    if from_ordered:
        # From spatial order back to original: use inverse ordering
        inverse_order = np.argsort(block_order)
        reorder_indices = inverse_order
    else:
        # From original to spatial order: use direct ordering
        reorder_indices = block_order

    # Reorder each channel
    result = np.zeros_like(values)
    for channel in range(channels):
        result[channel] = values[channel, reorder_indices]

    # Restore original dimension order if we swapped
    if swapped:
        result = result.T

    return result
