#!/usr/bin/env python3
"""
Convert float32 mixed tensor patterns to int8 format.

This script loads the existing float32 patterns and creates int8 versions
according to the specification: A stored as int8 [0,255].
"""

import sys
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def convert_patterns_to_int8(input_path: str, output_path: str):
    """Convert patterns from float32 to int8 format."""

    print(f"Loading patterns from: {input_path}")
    patterns_data = np.load(input_path, allow_pickle=True)

    # Load existing float32 mixed tensor
    if "mixed_tensor" not in patterns_data:
        print("ERROR: No mixed tensor found in patterns file")
        return

    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    float32_mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    print(
        f"Original mixed tensor: {float32_mixed_tensor.batch_size} LEDs, dtype={float32_mixed_tensor.dtype}"
    )
    min_val = float(cp.min(float32_mixed_tensor.sparse_values))
    max_val = float(cp.max(float32_mixed_tensor.sparse_values))
    print(f"Original values range: [{min_val:.6f}, {max_val:.6f}]")

    # Create int8 version
    print("Converting to int8...")
    int8_mixed_tensor = SingleBlockMixedSparseTensor(
        batch_size=float32_mixed_tensor.batch_size,
        channels=float32_mixed_tensor.channels,
        height=float32_mixed_tensor.height,
        width=float32_mixed_tensor.width,
        block_size=float32_mixed_tensor.block_size,
        device=float32_mixed_tensor.device,
        dtype=cp.uint8,  # Use uint8 (equivalent to int8 for our purposes)
    )

    # Convert values: [0,1] -> [0,255]
    int8_values = (float32_mixed_tensor.sparse_values * 255.0).astype(cp.uint8)
    int8_mixed_tensor.sparse_values = int8_values
    int8_mixed_tensor.block_positions = float32_mixed_tensor.block_positions.copy()

    print(f"Converted mixed tensor: dtype={int8_mixed_tensor.dtype}")
    print(
        f"Converted values range: [{int(cp.min(int8_values))}, {int(cp.max(int8_values))}]"
    )

    # Verify conversion
    sample_float = float32_mixed_tensor.sparse_values[0, 0, 0, 0]
    sample_int8 = int8_mixed_tensor.sparse_values[0, 0, 0, 0]
    expected_int8 = int(sample_float * 255)
    print(
        f"Sample conversion: {float(sample_float):.6f} -> {int(sample_int8)} (expected: {expected_int8})"
    )

    # Export to new dict format
    int8_mixed_dict = int8_mixed_tensor.to_dict()

    # Copy all other data from original patterns
    output_data = {}
    for key, value in patterns_data.items():
        if key == "mixed_tensor":
            # Replace with int8 version
            output_data[key] = int8_mixed_dict
            print("Replaced mixed_tensor with int8 version")
        else:
            # Copy existing data
            output_data[key] = value
            print(f"Copied existing data: {key}")

    # Save to new file
    print(f"Saving int8 patterns to: {output_path}")
    np.savez_compressed(output_path, **output_data)

    # Verify the saved file
    print("Verifying saved file...")
    verify_data = np.load(output_path, allow_pickle=True)
    verify_mixed_dict = verify_data["mixed_tensor"].item()
    verify_mixed_tensor = SingleBlockMixedSparseTensor.from_dict(verify_mixed_dict)

    print(
        f"Verified mixed tensor: {verify_mixed_tensor.batch_size} LEDs, dtype={verify_mixed_tensor.dtype}"
    )
    min_verify = int(cp.min(verify_mixed_tensor.sparse_values))
    max_verify = int(cp.max(verify_mixed_tensor.sparse_values))
    print(f"Verified values range: [{min_verify}, {max_verify}]")

    print("Conversion completed successfully!")


if __name__ == "__main__":
    input_patterns = "diffusion_patterns/baseline_realistic.npz"
    output_patterns = "diffusion_patterns/baseline_realistic_int8.npz"

    convert_patterns_to_int8(input_patterns, output_patterns)
