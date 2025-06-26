#!/usr/bin/env python3
"""
Quick test to validate clipping equivalence concept with a few LEDs.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.const import FRAME_HEIGHT, FRAME_WIDTH

# Load pattern data
data = np.load('diffusion_patterns/synthetic_1000_with_ata.npz')

# Get mixed tensor metadata
positions = data['mixed_tensor_positions']  # (1000, 3, 2)
blocks_set = data['mixed_tensor_blocks_set']  # (1000, 3)
block_size = int(data['mixed_tensor_block_size'])

print(f"Block size: {block_size}")
print(f"Frame size: {FRAME_HEIGHT} x {FRAME_WIDTH}")

# Reconstruct CSC matrix
matrix_shape = tuple(data['matrix_shape'])
matrix_data = data['matrix_data']
matrix_indices = data['matrix_indices'] 
matrix_indptr = data['matrix_indptr']

sparse_matrix_csc = sp.csc_matrix((matrix_data, matrix_indices, matrix_indptr), shape=matrix_shape)
print(f"CSC matrix shape: {sparse_matrix_csc.shape}")

# Test clipping concept for first LED, red channel
led_id = 0
channel = 0  # Red
col_idx = led_id * 3 + channel

print(f"\nTesting LED {led_id}, channel {channel} (column {col_idx})")

if blocks_set[led_id, channel]:
    top_row, top_col = positions[led_id, channel]
    print(f"Mixed tensor block position: ({top_row}, {top_col})")
    
    # Get the CSC column for this LED/channel
    col_start = sparse_matrix_csc.indptr[col_idx]
    col_end = sparse_matrix_csc.indptr[col_idx + 1]
    row_indices = sparse_matrix_csc.indices[col_start:col_end]
    values = sparse_matrix_csc.data[col_start:col_end]
    
    print(f"CSC column has {len(row_indices)} non-zero entries")
    
    if len(row_indices) > 0:
        # Convert to 2D coordinates
        row_coords = row_indices // FRAME_WIDTH
        col_coords = row_indices % FRAME_WIDTH
        
        print(f"CSC pattern bounds: rows [{row_coords.min()}, {row_coords.max()}], cols [{col_coords.min()}, {col_coords.max()}]")
        
        # Define clipping region
        bottom_row = top_row + block_size
        right_col = top_col + block_size
        print(f"Clipping region: rows [{top_row}, {bottom_row}), cols [{top_col}, {right_col})")
        
        # Check how many pixels would be clipped
        inside_mask = ((top_row <= row_coords) & (row_coords < bottom_row) & 
                       (top_col <= col_coords) & (col_coords < right_col))
        
        pixels_inside = np.sum(inside_mask)
        pixels_outside = len(row_indices) - pixels_inside
        
        print(f"Pixels inside clipping region: {pixels_inside}")
        print(f"Pixels outside clipping region: {pixels_outside}")
        print(f"Clipping would remove {pixels_outside/len(row_indices)*100:.1f}% of pixels")
        
        # Check if mixed tensor stores similar number of non-zeros
        mixed_values = data['mixed_tensor_values'][led_id, channel]
        mixed_nonzeros = np.count_nonzero(mixed_values)
        print(f"Mixed tensor non-zeros: {mixed_nonzeros}")
        print(f"Ratio (mixed/csc_inside): {mixed_nonzeros/pixels_inside:.2f}")

else:
    print("No block set for this LED/channel")

# Quick summary of the equivalence concept
print(f"\n=== Clipping Equivalence Concept ===")
print(f"1. Mixed tensor stores 96x96 blocks, ignoring pixels outside")
print(f"2. CSC stores full patterns with potentially more pixels")
print(f"3. To make them equivalent, we need to zero CSC pixels outside the 96x96 blocks")
print(f"4. This should make A^T@b calculations identical between the two approaches")
print(f"5. The performance difference comes from A^T@b calculation method, not the data content")