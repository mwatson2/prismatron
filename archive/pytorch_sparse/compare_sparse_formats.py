#!/usr/bin/env python3
"""
Compare memory usage of different sparse matrix formats.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_sparse_formats():
    """Compare COO vs CSC/CSR storage efficiency."""

    # Matrix dimensions from logs
    pixels = 480 * 800  # 384,000
    leds = 1000
    channels = 3
    nnz = 41_945_614

    logger.info("=== Sparse Format Storage Comparison ===")
    logger.info(f"Matrix: {pixels:,} pixels × {leds*channels:,} LED channels")
    logger.info(f"Non-zero values: {nnz:,}")
    logger.info(f"Density: {(nnz / (pixels * leds * channels)) * 100:.3f}%")

    logger.info("\n=== CSC Format (Current CuPy/SciPy approach) ===")
    # CSC: data + row_indices + column_pointers
    csc_data = nnz * 4  # float32 values
    csc_indices = nnz * 4  # int32 row indices
    csc_indptr = (leds * channels + 1) * 4  # int32 column pointers
    csc_total = csc_data + csc_indices + csc_indptr

    logger.info(f"Data array:      {csc_data / 1024**2:.1f} MB ({nnz:,} × 4 bytes)")
    logger.info(f"Row indices:     {csc_indices / 1024**2:.1f} MB ({nnz:,} × 4 bytes)")
    logger.info(
        f"Column pointers: {csc_indptr / 1024**2:.3f} MB ({leds*channels+1:,} × 4 bytes)"
    )
    logger.info(f"Total CSC:       {csc_total / 1024**2:.1f} MB")

    logger.info("\n=== COO Format (PyTorch approach) ===")
    # COO: data + row_coords + col_coords (2D matrix)
    coo_2d_data = nnz * 4  # float32 values
    coo_2d_row = nnz * 4  # int32 row coordinates
    coo_2d_col = nnz * 4  # int32 column coordinates
    coo_2d_total = coo_2d_data + coo_2d_row + coo_2d_col

    logger.info(f"2D COO tensor:")
    logger.info(f"  Data array:    {coo_2d_data / 1024**2:.1f} MB")
    logger.info(f"  Row coords:    {coo_2d_row / 1024**2:.1f} MB")
    logger.info(f"  Col coords:    {coo_2d_col / 1024**2:.1f} MB")
    logger.info(f"  Total 2D COO:  {coo_2d_total / 1024**2:.1f} MB")

    # 4D COO: data + 4 coordinate arrays (height, width, led, channel)
    coo_4d_data = nnz * 4  # float32 values
    coo_4d_coords = nnz * 8 * 4  # int64 coordinates × 4 dimensions
    coo_4d_total = coo_4d_data + coo_4d_coords

    logger.info(f"\n4D COO tensor (PyTorch implementation):")
    logger.info(f"  Data array:    {coo_4d_data / 1024**2:.1f} MB")
    logger.info(
        f"  Coordinates:   {coo_4d_coords / 1024**2:.1f} MB ({nnz:,} × 4 dims × 8 bytes)"
    )
    logger.info(f"  Total 4D COO:  {coo_4d_total / 1024**2:.1f} MB")

    logger.info("\n=== Memory Efficiency Analysis ===")
    ratio_2d = coo_2d_total / csc_total
    ratio_4d = coo_4d_total / csc_total

    logger.info(f"2D COO vs CSC: {ratio_2d:.1f}x larger")
    logger.info(f"4D COO vs CSC: {ratio_4d:.1f}x larger")
    logger.info(f"4D COO vs 2D COO: {coo_4d_total / coo_2d_total:.1f}x larger")

    logger.info("\n=== Why 4D COO is So Large ===")
    logger.info("1. CSC uses compressed column storage - only stores column starts")
    logger.info("2. 2D COO stores 2 coordinates per non-zero (row, col)")
    logger.info("3. 4D COO stores 4 coordinates per non-zero (h, w, led, channel)")
    logger.info("4. PyTorch uses int64 (8 bytes) vs int32 (4 bytes) for coordinates")

    coord_overhead = coo_4d_coords / coo_4d_data
    logger.info(
        f"5. Coordinate overhead in 4D COO: {coord_overhead:.1f}x the data size!"
    )

    logger.info("\n=== Recommendations ===")
    logger.info("1. The 4D COO approach is fundamentally memory-inefficient")
    logger.info("2. Stick with CuPy/SciPy CSC matrices for production")
    logger.info("3. If using PyTorch, convert to 2D COO or CSR format")
    logger.info("4. Consider PyTorch sparse CSR tensors (torch.sparse_csr_tensor)")


if __name__ == "__main__":
    compare_sparse_formats()
