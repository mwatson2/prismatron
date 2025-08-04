#!/usr/bin/env python3
"""
Debug script to check what ATA inverse formats are being loaded.
"""

from pathlib import Path

import numpy as np


def check_ata_formats():
    """Check what ATA inverse formats are available in pattern files."""

    pattern_dir = Path("diffusion_patterns")

    patterns = [
        "synthetic_2624_uint8.npz",
        "synthetic_2624_uint8_dia_1.2.npz",
        "synthetic_2624_uint8_dia_1.6.npz",
        "synthetic_2624_uint8_dia_2.0.npz",
    ]

    for pattern_name in patterns:
        pattern_path = pattern_dir / pattern_name
        if not pattern_path.exists():
            print(f"❌ {pattern_name}: File not found")
            continue

        print(f"\n📁 {pattern_name}:")

        try:
            data = np.load(pattern_path, allow_pickle=True)
            print(f"  Keys: {list(data.keys())}")

            # Check for ATA inverse
            if "ata_inverse" in data:
                ata_inv = data["ata_inverse"]
                print(f"  ✓ ata_inverse: shape={ata_inv.shape}, dtype={ata_inv.dtype}")
                print(f"    Type: {type(ata_inv)}")
            else:
                print("  ❌ No 'ata_inverse' key")

            # Check for DIA ATA inverse
            if "ata_inverse_dia" in data:
                ata_inv_dia = data["ata_inverse_dia"].item()
                print(f"  ✓ ata_inverse_dia: type={type(ata_inv_dia)}")
                if isinstance(ata_inv_dia, dict):
                    print(f"    Dict keys: {list(ata_inv_dia.keys())}")
            else:
                print("  ❌ No 'ata_inverse_dia' key")

        except Exception as e:
            print(f"  ❌ Error loading: {e}")


if __name__ == "__main__":
    check_ata_formats()
