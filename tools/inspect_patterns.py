#!/usr/bin/env python3
"""
Diffusion Pattern Inspector Tool.

This tool provides a quick way to inspect the contents of diffusion pattern files
without needing to run the full visualizer. It shows metadata, dimensions, matrix
formats (DIA/Dense), ATA inverse status, and other useful information about the
pattern file.

Usage:
    python inspect_patterns.py pattern_file.npz
    python inspect_patterns.py pattern_file.npz --verbose
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def format_bytes(bytes_val: int) -> str:
    """Format bytes into human readable units."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def format_timestamp(timestamp: float) -> str:
    """Format timestamp into human readable string."""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    except (ValueError, OSError):
        return f"Invalid timestamp: {timestamp}"


def inspect_metadata(data: Dict[str, Any], verbose: bool = False) -> None:
    """Inspect and display metadata."""
    if "metadata" not in data:
        print("❌ No metadata found")
        return

    metadata = data["metadata"].item() if hasattr(data["metadata"], "item") else data["metadata"]
    print("📋 Metadata:")

    # Key information first
    important_keys = ["led_count", "frame_width", "frame_height", "channels", "block_size"]
    for key in important_keys:
        if key in metadata:
            print(f"  {key}: {metadata[key]}")

    # Generation info
    generation_keys = ["generator", "pattern_type", "seed", "generation_method", "sparsity_threshold"]
    print("\n  Generation Info:")
    for key in generation_keys:
        if key in metadata:
            print(f"    {key}: {metadata[key]}")

    if "generation_timestamp" in metadata:
        timestamp_str = format_timestamp(metadata["generation_timestamp"])
        print(f"    generation_time: {timestamp_str}")

    # Matrix info
    matrix_keys = ["matrix_shape", "nnz", "sparsity_percent"]
    print("\n  Matrix Info:")
    for key in matrix_keys:
        if key in metadata:
            value = metadata[key]
            if key == "sparsity_percent":
                print(f"    {key}: {value:.3f}%")
            elif key == "nnz":
                print(f"    {key}: {value:,}")
            else:
                print(f"    {key}: {value}")

    # Options
    option_keys = ["use_fp16", "use_uint8", "intensity_variation", "led_size_scaling"]
    options = [f"{key}={metadata[key]}" for key in option_keys if key in metadata]
    if options:
        print(f"\n  Options: {', '.join(options)}")

    if verbose:
        print("\n  All Metadata Keys:")
        for key, value in sorted(metadata.items()):
            if key not in important_keys + generation_keys + matrix_keys + option_keys + ["generation_timestamp"]:
                print(f"    {key}: {value}")


def inspect_led_data(data: Dict[str, Any], verbose: bool = False) -> None:
    """Inspect LED position and ordering data."""
    print("\n🎯 LED Data:")

    # LED positions
    if "led_positions" in data:
        positions = data["led_positions"]
        print(f"  led_positions: shape={positions.shape}, dtype={positions.dtype}")
        if verbose and len(positions) <= 10:
            print(f"    positions: {positions.tolist()}")
        elif verbose:
            print(f"    first 5: {positions[:5].tolist()}")
            print(f"    last 5: {positions[-5:].tolist()}")

    # LED ordering
    if "led_ordering" in data:
        ordering = data["led_ordering"]
        print(f"  led_ordering: shape={ordering.shape}, dtype={ordering.dtype}")
        if verbose and len(ordering) <= 20:
            print(f"    ordering: {ordering.tolist()}")
        elif verbose:
            print(f"    first 10: {ordering[:10].tolist()}")
            print(f"    last 10: {ordering[-10:].tolist()}")

        # Check if ordering looks valid
        unique_vals = np.unique(ordering)
        expected_range = np.arange(len(ordering))
        if np.array_equal(np.sort(unique_vals), expected_range):
            print("    ✅ Ordering appears valid (permutation of 0 to led_count-1)")
        else:
            print("    ❌ Ordering may be invalid (not a proper permutation)")
    else:
        print("  led_ordering: ❌ Not found")

    # Spatial mapping
    if "led_spatial_mapping" in data:
        mapping = (
            data["led_spatial_mapping"].item()
            if hasattr(data["led_spatial_mapping"], "item")
            else data["led_spatial_mapping"]
        )
        print(f"  led_spatial_mapping: {len(mapping)} entries")
        if verbose:
            sample_items = list(mapping.items())[:5]
            print(f"    sample: {sample_items}")


def inspect_format_summary(data: Dict[str, Any]) -> None:
    """Inspect and summarize matrix formats available."""
    print("\n📊 Matrix Format Summary:")

    available_formats = []

    # Check for different matrix formats
    if "dia_matrix" in data:
        dia_dict = data["dia_matrix"].item() if hasattr(data["dia_matrix"], "item") else data["dia_matrix"]
        led_count = dia_dict.get("led_count", "unknown")
        k_diagonals = dia_dict.get("k", "unknown")
        available_formats.append(f"DIA ({led_count} LEDs, {k_diagonals} diagonals)")

    if "dense_ata_matrix" in data:
        dense_dict = (
            data["dense_ata_matrix"].item() if hasattr(data["dense_ata_matrix"], "item") else data["dense_ata_matrix"]
        )
        led_count = dense_dict.get("led_count", "unknown")
        memory_mb = dense_dict.get("memory_mb", "unknown")
        available_formats.append(f"Dense ATA ({led_count} LEDs, {memory_mb}MB)")

    if "mixed_tensor" in data:
        tensor_dict = data["mixed_tensor"].item() if hasattr(data["mixed_tensor"], "item") else data["mixed_tensor"]
        batch_size = tensor_dict.get("batch_size", "unknown")
        dtype = tensor_dict.get("dtype", "unknown")
        available_formats.append(f"Mixed Tensor ({batch_size} LEDs, {dtype})")

    # Check for ATA inverse
    has_ata_inverse = "ata_inverse" in data
    has_ata_inverse_dia = "ata_inverse_dia" in data

    if available_formats:
        print(f"  Available formats: {', '.join(available_formats)}")
    else:
        print("  ❌ No matrix formats found")

    # ATA inverse status
    inverse_status = []
    if has_ata_inverse:
        ata_inv = data["ata_inverse"]
        inverse_status.append(f"Dense inverse ({ata_inv.shape}, {ata_inv.dtype})")
    if has_ata_inverse_dia:
        inverse_status.append("DIA inverse")

    if inverse_status:
        print(f"  ATA inverse: {', '.join(inverse_status)}")
    else:
        print("  ATA inverse: ❌ Not available")

    # Optimization readiness
    can_optimize_dia = "dia_matrix" in data and "mixed_tensor" in data
    can_optimize_dense = "dense_ata_matrix" in data and "mixed_tensor" in data

    if can_optimize_dia or can_optimize_dense:
        print("  ✅ Ready for optimization")
        if can_optimize_dia:
            print("    - Can use DIA format optimization")
        if can_optimize_dense:
            print("    - Can use Dense format optimization")
    else:
        print("  ❌ Missing components for optimization")
        if "mixed_tensor" not in data:
            print("    - Missing mixed_tensor (A^T matrix)")
        if "dia_matrix" not in data and "dense_ata_matrix" not in data:
            print("    - Missing ATA matrix (dia_matrix or dense_ata_matrix)")


def inspect_matrices(data: Dict[str, Any], verbose: bool = False) -> None:
    """Inspect matrix data."""
    print("\n🔢 Matrix Data:")

    # Mixed tensor
    if "mixed_tensor" in data:
        tensor_dict = data["mixed_tensor"].item() if hasattr(data["mixed_tensor"], "item") else data["mixed_tensor"]
        print(f"  mixed_tensor: {len(tensor_dict)} fields")
        if "batch_size" in tensor_dict and "block_size" in tensor_dict:
            print(f"    batch_size: {tensor_dict['batch_size']}, block_size: {tensor_dict['block_size']}")
        if "dtype" in tensor_dict:
            print(f"    dtype: {tensor_dict['dtype']}")
        if verbose:
            print(f"    fields: {list(tensor_dict.keys())}")

    # DIA matrix
    if "dia_matrix" in data:
        dia_dict = data["dia_matrix"].item() if hasattr(data["dia_matrix"], "item") else data["dia_matrix"]
        print(f"  dia_matrix: {len(dia_dict)} fields")
        if "k" in dia_dict and "bandwidth" in dia_dict:
            print(f"    diagonals: {dia_dict['k']}, bandwidth: {dia_dict['bandwidth']}")
        if "led_count" in dia_dict:
            print(f"    led_count: {dia_dict['led_count']}")
        if "storage_dtype" in dia_dict and "output_dtype" in dia_dict:
            print(f"    storage_dtype: {dia_dict['storage_dtype']}, output_dtype: {dia_dict['output_dtype']}")
        if "sparsity" in dia_dict:
            print(f"    sparsity: {dia_dict['sparsity']:.3f}%")
        if verbose:
            print(f"    fields: {list(dia_dict.keys())}")

    # Dense ATA matrix
    if "dense_ata_matrix" in data:
        dense_dict = (
            data["dense_ata_matrix"].item() if hasattr(data["dense_ata_matrix"], "item") else data["dense_ata_matrix"]
        )
        print(f"  dense_ata_matrix: {len(dense_dict)} fields")
        if "led_count" in dense_dict:
            print(f"    led_count: {dense_dict['led_count']}")
        if "channels" in dense_dict:
            print(f"    channels: {dense_dict['channels']}")
        if "storage_dtype" in dense_dict and "output_dtype" in dense_dict:
            print(f"    storage_dtype: {dense_dict['storage_dtype']}, output_dtype: {dense_dict['output_dtype']}")
        if "memory_mb" in dense_dict:
            print(f"    memory: {dense_dict['memory_mb']:.1f} MB")
        if "matrix_shape" in dense_dict:
            print(f"    matrix_shape: {dense_dict['matrix_shape']}")
        if verbose:
            print(f"    fields: {list(dense_dict.keys())}")

    # ATA inverse
    if "ata_inverse" in data:
        ata_inv = data["ata_inverse"]
        print(f"  ata_inverse: shape={ata_inv.shape}, dtype={ata_inv.dtype}")
        memory_mb = ata_inv.nbytes / (1024 * 1024)
        print(f"    memory: {memory_mb:.1f} MB")

        # Check for ATA inverse metadata
        if "ata_inverse_metadata" in data:
            meta = (
                data["ata_inverse_metadata"].item()
                if hasattr(data["ata_inverse_metadata"], "item")
                else data["ata_inverse_metadata"]
            )
            if "computation_time" in meta:
                print(f"    computation_time: {meta['computation_time']:.2f}s")
            if "successful_inversions" in meta:
                print(f"    successful_inversions: {meta['successful_inversions']}/3")
            if "avg_condition_number" in meta:
                print(f"    avg_condition_number: {meta['avg_condition_number']:.2e}")

    # ATA inverse DIA
    if "ata_inverse_dia" in data:
        ata_inv_dia = (
            data["ata_inverse_dia"].item() if hasattr(data["ata_inverse_dia"], "item") else data["ata_inverse_dia"]
        )
        print(f"  ata_inverse_dia: {len(ata_inv_dia)} fields")
        if verbose:
            print(f"    fields: {list(ata_inv_dia.keys())}")


def inspect_file_info(file_path: Path) -> None:
    """Inspect file-level information."""
    file_size = file_path.stat().st_size
    print(f"📁 File Info:")
    print(f"  path: {file_path}")
    print(f"  size: {format_bytes(file_size)}")
    print(f"  modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_path.stat().st_mtime))}")


def inspect_pattern_file(file_path: Path, verbose: bool = False) -> None:
    """Main inspection function."""
    try:
        print(f"🔍 Inspecting Diffusion Pattern: {file_path.name}")
        print("=" * 60)

        # File info
        inspect_file_info(file_path)

        # Load data
        data = np.load(file_path, allow_pickle=True)

        print(f"\n🗂️  Top-level Keys: {list(data.keys())}")

        # Inspect each section
        inspect_metadata(data, verbose)
        inspect_led_data(data, verbose)
        inspect_format_summary(data)
        inspect_matrices(data, verbose)

        print("\n" + "=" * 60)
        print("✅ Inspection complete")

    except Exception as e:
        print(f"❌ Error inspecting file: {e}")
        if verbose:
            import traceback

            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect diffusion pattern files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection
  python inspect_patterns.py patterns.npz

  # Verbose inspection with more details
  python inspect_patterns.py patterns.npz --verbose

  # Inspect multiple files
  python inspect_patterns.py pattern1.npz pattern2.npz
""",
    )

    parser.add_argument("files", nargs="+", type=Path, help="Pattern files to inspect")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")

    args = parser.parse_args()

    for file_path in args.files:
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue

        if not file_path.suffix.lower() == ".npz":
            print(f"⚠️  Warning: {file_path} doesn't have .npz extension")

        inspect_pattern_file(file_path, args.verbose)

        # Add separator if inspecting multiple files
        if len(args.files) > 1 and file_path != args.files[-1]:
            print("\n" + "🔄" * 20 + "\n")


if __name__ == "__main__":
    main()
