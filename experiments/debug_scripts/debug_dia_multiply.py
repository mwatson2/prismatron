#!/usr/bin/env python3
"""
Debug script to check DIA ATA inverse multiplication.
"""

import cupy as cp
import numpy as np

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix


def test_dia_multiplication():
    """Test DIA matrix multiplication with debug output."""

    # Load a DIA pattern
    data = np.load("diffusion_patterns/synthetic_2624_uint8_dia_1.2.npz", allow_pickle=True)
    ata_inverse_dia = data["ata_inverse_dia"].item()

    print(f"ATA inverse DIA keys: {list(ata_inverse_dia.keys())}")

    # Inspect channel 0 data structure
    channel_0_data = ata_inverse_dia["channel_0"]
    print(f"Channel 0 data type: {type(channel_0_data)}")
    print(f"Channel 0 keys: {list(channel_0_data.keys()) if isinstance(channel_0_data, dict) else 'Not a dict'}")

    # Check what type of DIA matrix this is
    if "dia_data_cpu" in channel_0_data:
        dia_data = channel_0_data["dia_data_cpu"]
        print(f"DIA data shape: {dia_data.shape}")
        print(f"DIA data type: {dia_data.dtype}")

    if "dia_offsets" in channel_0_data:
        offsets = channel_0_data["dia_offsets"]
        print(f"DIA offsets shape: {offsets.shape}")
        print(f"DIA offsets: {offsets[:10]}...")  # First 10 offsets

    # Check if this is actually a single-channel DIA matrix
    if "led_count" in channel_0_data:
        print(f"LED count: {channel_0_data['led_count']}")

    if "channels" in channel_0_data:
        print(f"Channels: {channel_0_data['channels']}")

    # Try to understand the intended multiplication
    led_count = 2624
    atb_test = cp.random.rand(led_count).astype(cp.float32)  # Single channel test

    print(f"Single channel A^T @ b shape: {atb_test.shape}")

    try:
        # Create DIA matrix
        dia_matrix = DiagonalATAMatrix.from_dict(channel_0_data)
        print(f"DIA matrix created successfully")
        print(f"DIA matrix led_count: {dia_matrix.led_count}")
        print(f"DIA matrix channels: {dia_matrix.channels}")

        # The real question: what does this DIA matrix expect as input?
        # If it's a single-channel ATA inverse, it should work with single channel input

    except Exception as e:
        print(f"Error creating DIA matrix: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_dia_multiplication()
