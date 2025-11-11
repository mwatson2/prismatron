#!/usr/bin/env python3
"""
Convert diffusion pattern files from sRGB to linear light space.

This tool converts LED diffusion patterns from gamma-corrected sRGB space to
linear light space, which is required for correct LED optimization calculations.
The mathematical assumption that LED contributions can be added linearly only
holds in linear light space.

Usage:
    python convert_patterns_to_linear.py --input patterns_srgb.npz --output patterns_linear.npz

The tool:
1. Loads the Mixed Sparse Tensor from the input file
2. Converts pattern data from sRGB to linear (uint8 → fp32 → linear → fp32 → uint8)
3. Adds color_space='linear' flag to metadata
4. Preserves LED positions, spatial mapping, and ordering
5. Does NOT copy ATA matrices (use tools/compute_matrices.py to regenerate)

After conversion, use tools/compute_matrices.py to regenerate the ATA matrices
for the linear pattern file.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def srgb_to_linear(srgb_values: np.ndarray) -> np.ndarray:
    """
    Convert sRGB values to linear light space.

    Uses the standard sRGB transfer function:
    - For values <= 0.04045: linear = srgb / 12.92
    - For values > 0.04045: linear = ((srgb + 0.055) / 1.055) ^ 2.4

    Args:
        srgb_values: Array of sRGB values in range [0, 1]

    Returns:
        Array of linear light values in range [0, 1]
    """
    # Apply sRGB inverse gamma correction
    linear = np.where(
        srgb_values <= 0.04045,
        srgb_values / 12.92,
        np.power((srgb_values + 0.055) / 1.055, 2.4),
    )
    return linear


def linear_to_srgb(linear_values: np.ndarray) -> np.ndarray:
    """
    Convert linear light values to sRGB space.

    Uses the standard sRGB transfer function:
    - For values <= 0.0031308: srgb = linear * 12.92
    - For values > 0.0031308: srgb = 1.055 * linear^(1/2.4) - 0.055

    Args:
        linear_values: Array of linear light values in range [0, 1]

    Returns:
        Array of sRGB values in range [0, 1]
    """
    # Apply sRGB gamma correction
    srgb = np.where(
        linear_values <= 0.0031308,
        linear_values * 12.92,
        1.055 * np.power(linear_values, 1.0 / 2.4) - 0.055,
    )
    return srgb


def convert_mixed_tensor_to_linear(mixed_tensor: SingleBlockMixedSparseTensor) -> SingleBlockMixedSparseTensor:
    """
    Convert a Mixed Sparse Tensor from sRGB to linear light space.

    Args:
        mixed_tensor: Input tensor in sRGB space

    Returns:
        New tensor with pattern data converted to linear light space
    """
    logger.info("Converting Mixed Sparse Tensor from sRGB to linear...")
    logger.info(f"  Input tensor: {mixed_tensor}")

    # Convert sparse values from sRGB to linear
    # Shape: (channels, batch_size, block_size, block_size)
    import cupy as cp

    sparse_values = mixed_tensor.sparse_values  # CuPy array

    # Convert uint8 → fp32 (normalize to [0, 1])
    if mixed_tensor.dtype == cp.uint8:
        logger.info("  Converting uint8 → fp32...")
        srgb_fp32 = sparse_values.astype(cp.float32) / 255.0
    else:
        # Already float32, assume already normalized
        srgb_fp32 = sparse_values

    # Convert to NumPy for sRGB → linear conversion
    srgb_np = cp.asnumpy(srgb_fp32)

    # Convert sRGB → linear
    logger.info("  Converting sRGB → linear...")
    linear_np = srgb_to_linear(srgb_np)

    # Convert back to CuPy
    linear_fp32 = cp.asarray(linear_np)

    # Convert fp32 → uint8 (scale back to [0, 255])
    if mixed_tensor.dtype == cp.uint8:
        logger.info("  Converting fp32 → uint8...")
        # Clamp to [0, 1] and scale to [0, 255]
        linear_uint8 = cp.clip(linear_fp32 * 255.0, 0, 255).astype(cp.uint8)
        converted_values = linear_uint8
        output_dtype = cp.uint8
    else:
        # Keep as float32
        converted_values = linear_fp32
        output_dtype = cp.float32

    # Create new tensor with converted values
    new_tensor = SingleBlockMixedSparseTensor(
        batch_size=mixed_tensor.batch_size,
        channels=mixed_tensor.channels,
        height=mixed_tensor.height,
        width=mixed_tensor.width,
        block_size=mixed_tensor.block_size,
        device=mixed_tensor.device,
        dtype=output_dtype,
        output_dtype=mixed_tensor.output_dtype,
    )

    # Copy converted values and original positions
    new_tensor.sparse_values = converted_values
    new_tensor.block_positions = mixed_tensor.block_positions.copy()

    logger.info(f"  Output tensor: {new_tensor}")

    # Log statistics about the conversion
    orig_max = float(cp.max(mixed_tensor.sparse_values))
    new_max = float(cp.max(new_tensor.sparse_values))
    orig_mean = float(cp.mean(mixed_tensor.sparse_values.astype(cp.float32)))
    new_mean = float(cp.mean(new_tensor.sparse_values.astype(cp.float32)))

    logger.info(f"  Value ranges - Original: max={orig_max:.1f}, mean={orig_mean:.1f}")
    logger.info(f"  Value ranges - Converted: max={new_max:.1f}, mean={new_mean:.1f}")

    return new_tensor


def convert_pattern_file(input_path: Path, output_path: Path) -> None:
    """
    Convert a pattern file from sRGB to linear light space.

    Args:
        input_path: Path to input pattern file (sRGB)
        output_path: Path to output pattern file (linear)
    """
    logger.info(f"Loading pattern file: {input_path}")

    # Load the pattern file
    try:
        data = np.load(input_path, allow_pickle=True)
    except Exception as e:
        logger.error(f"Failed to load pattern file: {e}")
        raise

    # Check required keys
    required_keys = ["mixed_tensor", "metadata", "led_positions"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"Pattern file missing required keys: {missing_keys}")

    logger.info("  Found keys: " + ", ".join(data.keys()))

    # Load metadata
    metadata = data["metadata"].item() if data["metadata"].ndim == 0 else dict(data["metadata"])
    logger.info(f"  Original metadata: {metadata}")

    # Check if already converted
    if metadata.get("color_space") == "linear":
        logger.warning("  WARNING: Pattern file already marked as linear color space!")
        logger.warning("  Converting again will result in incorrect double conversion!")
        response = input("  Continue anyway? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            logger.info("  Conversion cancelled.")
            return

    # Load the mixed tensor
    logger.info("Loading Mixed Sparse Tensor...")
    mixed_tensor_dict = data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict, device="cuda")

    # Convert to linear
    linear_tensor = convert_mixed_tensor_to_linear(mixed_tensor)

    # Update metadata
    metadata["color_space"] = "linear"
    metadata["converted_from"] = str(input_path)

    logger.info(f"  Updated metadata: {metadata}")

    # Prepare output dictionary
    save_dict = {
        "mixed_tensor": linear_tensor.to_dict(),
        "metadata": metadata,
        "led_positions": data["led_positions"],
    }

    # Copy optional keys if present
    optional_keys = ["led_spatial_mapping", "led_ordering"]
    for key in optional_keys:
        if key in data:
            save_dict[key] = data[key]
            logger.info(f"  Preserved {key}")

    # Skip ATA matrices (they need to be regenerated)
    ata_keys = ["dense_ata_matrix", "symmetric_dia_matrix", "dia_matrix", "ata_inverse"]
    skipped_keys = [key for key in ata_keys if key in data]
    if skipped_keys:
        logger.info(f"  Skipped ATA matrices: {', '.join(skipped_keys)}")
        logger.info(f"  Run tools/compute_matrices.py to regenerate them")

    # Save converted pattern file
    logger.info(f"Saving converted pattern file: {output_path}")
    np.savez_compressed(output_path, **save_dict)

    # Verify the saved file
    logger.info("Verifying saved file...")
    verify_data = np.load(output_path, allow_pickle=True)
    verify_metadata = (
        verify_data["metadata"].item() if verify_data["metadata"].ndim == 0 else dict(verify_data["metadata"])
    )

    if verify_metadata.get("color_space") != "linear":
        raise ValueError("Saved file does not have color_space='linear' in metadata!")

    logger.info("✓ Conversion complete!")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Regenerate ATA matrices: python tools/compute_matrices.py {output_path}")
    logger.info(f"  2. Test with visualizer: python tools/visualize_diffusion_patterns.py {output_path}")
    logger.info(f"  3. Update consumer to use linear patterns")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert LED diffusion patterns from sRGB to linear light space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a pattern file
  python convert_patterns_to_linear.py --input patterns_srgb.npz --output patterns_linear.npz

  # After conversion, regenerate matrices
  python tools/compute_matrices.py patterns_linear.npz

Notes:
  - The tool converts uint8 [0, 255] values through the sRGB → linear transformation
  - Pattern values are temporarily converted to float32 for accurate conversion
  - ATA matrices are NOT copied and must be regenerated with compute_matrices.py
  - The visualizer and LED optimizer will be updated to handle linear patterns
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input pattern file (sRGB color space)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output pattern file (linear color space)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force conversion even if output file exists",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        logger.error(f"Input file does not exist: {args.input}")
        sys.exit(1)

    # Check output file
    if args.output.exists() and not args.force:
        logger.error(f"Output file already exists: {args.output}")
        logger.error("Use --force to overwrite")
        sys.exit(1)

    # Perform conversion
    try:
        convert_pattern_file(args.input, args.output)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
