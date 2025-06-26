#!/usr/bin/env python3
"""
Create a small test mixed tensor file for testing pattern visualization.
"""

import numpy as np
from pathlib import Path

def create_test_mixed_tensor():
    """Create a small test mixed tensor pattern file."""
    
    # Parameters for small test
    batch_size = 10  # 10 LEDs
    channels = 3     # RGB
    height = 160     # Frame height
    width = 120      # Frame width  
    block_size = 16  # 16x16 blocks

    print(f"Creating test mixed tensor: {batch_size} LEDs, {channels} channels, {height}x{width} images, {block_size}x{block_size} blocks")

    # Create sparse values: (channels, batch_size, block_size, block_size)
    sparse_values = np.random.rand(channels, batch_size, block_size, block_size).astype(np.float32)
    
    # Create block positions: (channels, batch_size, 2) 
    block_positions = np.zeros((channels, batch_size, 2), dtype=np.int32)
    
    # Generate random positions for each LED and channel
    np.random.seed(42)  # For reproducibility
    for c in range(channels):
        for led in range(batch_size):
            # Random position ensuring block fits in frame
            max_row = height - block_size
            max_col = width - block_size
            block_positions[c, led, 0] = np.random.randint(0, max_row + 1)  # row
            block_positions[c, led, 1] = np.random.randint(0, max_col + 1)  # col

    # All blocks are set (for simplicity)
    blocks_set = np.ones((channels, batch_size), dtype=bool)

    # Create LED positions 
    led_positions = np.random.rand(batch_size, 2) * np.array([width, height])
    led_positions = led_positions.astype(np.int32)

    # Create metadata
    metadata = {
        "format": "mixed_tensor",
        "created_by": "test_script",
        "description": "Test mixed tensor patterns for visualization testing",
        "led_count": batch_size,
        "total_blocks": batch_size * channels
    }

    # Save to NPZ file
    output_file = "test_mixed_tensor_patterns.npz"
    np.savez_compressed(
        output_file,
        sparse_values=sparse_values,
        block_positions=block_positions,
        blocks_set=blocks_set,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        block_size=block_size,
        led_positions=led_positions,
        metadata=metadata
    )

    print(f"Created test mixed tensor file: {output_file}")
    
    # Verify file
    data = np.load(output_file, allow_pickle=True)
    print("File contents:")
    for key in data.keys():
        if hasattr(data[key], 'shape'):
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            print(f"  {key}: {data[key]}")

if __name__ == "__main__":
    create_test_mixed_tensor()