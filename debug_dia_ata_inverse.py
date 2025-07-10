#!/usr/bin/env python3
"""
Debug script to understand the DIA ATA inverse loading issue.
"""

import numpy as np

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix


def debug_dia_loading():
    """Debug both regular DIA ATA and DIA ATA inverse loading."""

    # Load a pattern with both regular DIA ATA and DIA ATA inverse
    data = np.load("diffusion_patterns/synthetic_2624_uint8_dia_1.2.npz", allow_pickle=True)

    print("Available keys:", list(data.keys()))

    # 1. Test regular DIA ATA matrix (this should work)
    print("\n=== REGULAR DIA ATA MATRIX ===")
    dia_ata_data = data["dia_matrix"].item()
    print("DIA ATA data keys:", list(dia_ata_data.keys()))
    print("DIA ATA version:", dia_ata_data.get("version", "unknown"))

    try:
        dia_ata_matrix = DiagonalATAMatrix.from_dict(dia_ata_data)
        print("✅ Regular DIA ATA matrix loaded successfully")
        print(f"  LED count: {dia_ata_matrix.led_count}")
        print(f"  Channels: {dia_ata_matrix.channels}")
        print(f"  GPU data built: {dia_ata_matrix.dia_data_gpu is not None}")
        if dia_ata_matrix.dia_data_gpu is not None:
            print(f"  GPU data shape: {dia_ata_matrix.dia_data_gpu.shape}")

        # Try a simple operation
        import cupy as cp

        test_input = cp.random.rand(3, 2624).astype(cp.float32)
        result = dia_ata_matrix.multiply_3d(test_input)
        print(f"  ✅ multiply_3d works: output shape {result.shape}")

    except Exception as e:
        print(f"❌ Regular DIA ATA matrix failed: {e}")
        import traceback

        traceback.print_exc()

    # 2. Test DIA ATA inverse (this is failing)
    print("\n=== DIA ATA INVERSE MATRIX ===")
    ata_inverse_dia_data = data["ata_inverse_dia"].item()
    print("DIA ATA inverse data keys:", list(ata_inverse_dia_data.keys()))

    # Check channel_0 structure
    channel_0_data = ata_inverse_dia_data["channel_0"]
    print("Channel 0 keys:", list(channel_0_data.keys()))
    print("Channel 0 version:", channel_0_data.get("version", "unknown"))

    try:
        dia_ata_inverse = DiagonalATAMatrix.from_dict(channel_0_data)
        print("✅ DIA ATA inverse loaded successfully")
        print(f"  LED count: {dia_ata_inverse.led_count}")
        print(f"  Channels: {dia_ata_inverse.channels}")
        print(f"  GPU data built: {dia_ata_inverse.dia_data_gpu is not None}")
        if dia_ata_inverse.dia_data_gpu is not None:
            print(f"  GPU data shape: {dia_ata_inverse.dia_data_gpu.shape}")

        # Try a simple operation
        import cupy as cp

        test_input = cp.random.rand(3, 2624).astype(cp.float32)
        result = dia_ata_inverse.multiply_3d(test_input)
        print(f"  ✅ multiply_3d works: output shape {result.shape}")

    except Exception as e:
        print(f"❌ DIA ATA inverse failed: {e}")
        import traceback

        traceback.print_exc()

    # 3. Compare the data structures
    print("\n=== COMPARISON ===")
    print("Regular DIA ATA structure:")
    for key in ["dia_data_3d", "dia_offsets_3d", "k", "led_count", "channels"]:
        if key in dia_ata_data:
            value = dia_ata_data[key]
            if hasattr(value, "shape"):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")

    print("\nDIA ATA inverse structure:")
    for key in ["dia_data_3d", "dia_offsets_3d", "k", "led_count", "channels"]:
        if key in channel_0_data:
            value = channel_0_data[key]
            if hasattr(value, "shape"):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    debug_dia_loading()
