#!/usr/bin/env python3
"""
Repair DIA Matrix Tool.

This tool fixes existing pattern files that have inefficient DIA matrices with too many
empty diagonals. It converts from the dense ATA matrix back to an optimized DIA format
that only stores significant diagonals.

Usage:
    python repair_dia_matrix.py pattern_file.npz
    python repair_dia_matrix.py pattern_file.npz --output repaired_file.npz
    python repair_dia_matrix.py pattern_file.npz --dry-run  # Show analysis only
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
except ImportError:
    # Fallback for systems without CUDA
    import numpy as cp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.dense_ata_matrix import DenseATAMatrix
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix


def analyze_current_dia_matrix(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the current DIA matrix efficiency."""
    if "dia_matrix" not in data:
        return {"status": "no_dia_matrix", "error": "No DIA matrix found in file"}

    dia_matrix_dict = data["dia_matrix"].item() if hasattr(data["dia_matrix"], "item") else data["dia_matrix"]

    # Extract key metrics
    k_diagonals = dia_matrix_dict.get("k", 0)
    bandwidth = dia_matrix_dict.get("bandwidth", 0)
    led_count = dia_matrix_dict.get("led_count", 0)
    sparsity = dia_matrix_dict.get("sparsity", 0.0)

    if "dia_data_3d" in dia_matrix_dict:
        dia_data_3d = dia_matrix_dict["dia_data_3d"]
        storage_size_mb = dia_data_3d.nbytes / (1024 * 1024)

        # Analyze diagonal usage
        channel_0_data = dia_data_3d[0]  # Shape: (k, led_count)
        diagonal_nnz = []
        for i in range(k_diagonals):
            diagonal = channel_0_data[i, :]
            nnz = np.count_nonzero(diagonal)
            diagonal_nnz.append(nnz)

        diagonal_nnz = np.array(diagonal_nnz)
        non_empty_diagonals = np.sum(diagonal_nnz > 0)
        empty_diagonals = k_diagonals - non_empty_diagonals

        # Calculate efficiency based on bandwidth vs diagonal count
        # The actual bandwidth analysis from our dense matrix showed only ~895 diagonals needed
        # The issue is not empty diagonals but storing the full bandwidth unnecessarily

        # A reasonable estimate: bandwidth should be proportional to sqrt(led_count) for well-structured matrices
        reasonable_bandwidth = min(bandwidth, int(2 * np.sqrt(led_count)))
        reasonable_diagonals = 2 * reasonable_bandwidth + 1  # -bandwidth to +bandwidth

        # If we're storing much more than a reasonable bandwidth suggests, we need repair
        bandwidth_efficiency_ratio = k_diagonals / reasonable_diagonals if reasonable_diagonals > 0 else 1

        return {
            "status": "found",
            "k_diagonals": k_diagonals,
            "bandwidth": bandwidth,
            "led_count": led_count,
            "sparsity": sparsity,
            "storage_size_mb": storage_size_mb,
            "non_empty_diagonals": non_empty_diagonals,
            "empty_diagonals": empty_diagonals,
            "empty_percentage": empty_diagonals / k_diagonals * 100 if k_diagonals > 0 else 0,
            "reasonable_bandwidth": reasonable_bandwidth,
            "reasonable_diagonals": reasonable_diagonals,
            "bandwidth_efficiency_ratio": bandwidth_efficiency_ratio,
            "needs_repair": bandwidth_efficiency_ratio > 3.0,  # More than 3x reasonable diagonals
        }
    else:
        return {"status": "invalid", "error": "DIA matrix missing dia_data_3d"}


def create_optimized_dia_from_dense(dense_ata_dict: Dict[str, Any], led_count: int) -> DiagonalATAMatrix:
    """Create optimized DIA matrix from dense ATA matrix."""
    print("Creating optimized DIA matrix from dense ATA matrix...")

    # Reconstruct dense ATA matrix
    dense_ata = DenseATAMatrix.from_dict(dense_ata_dict)
    dense_matrices = dense_ata_dict["dense_matrices"]  # Shape: (3, led_count, led_count)

    print(f"  Dense ATA shape: {dense_matrices.shape}")
    print(f"  Dense ATA memory: {dense_ata_dict['memory_mb']:.1f} MB")

    # Create new DIA matrix instance
    dia_matrix = DiagonalATAMatrix(led_count=led_count, crop_size=64)

    # Convert dense matrices to optimized DIA format
    print("  Converting to optimized DIA format...")

    # Manually build the DIA matrix with proper sparsity filtering
    all_offsets = set()
    channel_dia_matrices = []
    total_nnz = 0

    for channel in range(3):
        print(f"  Processing channel {channel} ({['R', 'G', 'B'][channel]})...")

        # Get dense matrix for this channel
        dense_channel = dense_matrices[channel]  # Shape: (led_count, led_count)

        # Convert to sparse for analysis
        sparse_channel = sp.csc_matrix(dense_channel)

        # Convert to DIA format
        dia_channel = sp.dia_matrix(sparse_channel)
        channel_dia_matrices.append(dia_channel)

        # Apply the same filtering logic as the fixed DiagonalATAMatrix
        matrix_max = dense_channel.max()
        if matrix_max > 0:
            significance_threshold = matrix_max * 0.001
            min_nnz_per_diagonal = max(1, int(led_count * 0.01))
        else:
            significance_threshold = 1e-10
            min_nnz_per_diagonal = 1

        print(f"    Matrix max: {matrix_max:.3e}, threshold: {significance_threshold:.3e}")

        significant_diagonals = 0
        for i, offset in enumerate(dia_channel.offsets):
            diagonal_data = dia_channel.data[i]
            significant_values = np.sum(np.abs(diagonal_data) > significance_threshold)

            if significant_values >= min_nnz_per_diagonal:
                all_offsets.add(offset)
                significant_diagonals += 1

        print(f"    Original diagonals: {len(dia_channel.offsets)}, significant: {significant_diagonals}")
        total_nnz += dia_channel.nnz

    # Create unified diagonal structure
    if all_offsets:
        dia_matrix.dia_offsets = np.array(sorted(all_offsets), dtype=np.int32)
        dia_matrix.k = len(dia_matrix.dia_offsets)
    else:
        dia_matrix.dia_offsets = np.array([], dtype=np.int32)
        dia_matrix.k = 0

    print(f"  Unified diagonal structure: {dia_matrix.k} diagonals")
    if dia_matrix.k > 0:
        print(f"    Offset range: [{dia_matrix.dia_offsets[0]}, {dia_matrix.dia_offsets[-1]}]")

    # Create unified 3D DIA data structure
    if dia_matrix.k > 0:
        dia_matrix.dia_data_cpu = np.zeros((3, dia_matrix.k, led_count), dtype=np.float32)

        # Create mapping from offset to index
        offset_to_idx = {offset: i for i, offset in enumerate(dia_matrix.dia_offsets)}

        # Fill unified structure
        for channel in range(3):
            dia_channel = channel_dia_matrices[channel]
            dense_channel = dense_matrices[channel]

            # Recalculate thresholds
            matrix_max = dense_channel.max()
            if matrix_max > 0:
                significance_threshold = matrix_max * 0.001
                min_nnz_per_diagonal = max(1, int(led_count * 0.01))
            else:
                significance_threshold = 1e-10
                min_nnz_per_diagonal = 1

            for i, offset in enumerate(dia_channel.offsets):
                diagonal_data = dia_channel.data[i]
                significant_values = np.sum(np.abs(diagonal_data) > significance_threshold)

                if significant_values >= min_nnz_per_diagonal and offset in offset_to_idx:
                    unified_idx = offset_to_idx[offset]
                    dia_matrix.dia_data_cpu[channel, unified_idx, :] = diagonal_data.astype(np.float32)
    else:
        dia_matrix.dia_data_cpu = np.zeros((3, 0, led_count), dtype=np.float32)

    # Update GPU caches and metadata
    dia_matrix._update_dia_offsets_cache()
    dia_matrix._update_dia_data_cache()

    # Set metadata
    dia_matrix.nnz = total_nnz
    dia_matrix.bandwidth = int(np.max(np.abs(dia_matrix.dia_offsets))) if dia_matrix.k > 0 else 0
    total_elements = 3 * led_count * led_count
    dia_matrix.sparsity = total_nnz / total_elements * 100

    # Set per-channel metadata
    for channel in range(3):
        dia_matrix.channel_nnz[channel] = channel_dia_matrices[channel].nnz

    print(f"  Optimized DIA matrix created: {dia_matrix.k} diagonals, bandwidth={dia_matrix.bandwidth}")
    return dia_matrix


def repair_pattern_file(input_path: Path, output_path: Path = None, dry_run: bool = False) -> bool:
    """Repair a pattern file with inefficient DIA matrix."""
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return False

    print(f"🔍 Analyzing pattern file: {input_path.name}")
    print("=" * 60)

    # Load pattern file
    try:
        data = np.load(str(input_path), allow_pickle=True)
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return False

    # Check file format
    if not all(key in data for key in ["dense_ata_matrix", "dia_matrix", "metadata"]):
        print("❌ File is missing required components (dense_ata_matrix, dia_matrix, metadata)")
        return False

    # Analyze current DIA matrix
    current_analysis = analyze_current_dia_matrix(data)

    print("📊 Current DIA Matrix Analysis:")
    if current_analysis["status"] == "found":
        print(f"  Stored diagonals: {current_analysis['k_diagonals']:,}")
        print(f"  Non-empty diagonals: {current_analysis['non_empty_diagonals']:,}")
        print(
            f"  Empty diagonals: {current_analysis['empty_diagonals']:,} ({current_analysis['empty_percentage']:.1f}%)"
        )
        print(f"  Bandwidth: {current_analysis['bandwidth']}")
        print(f"  Storage size: {current_analysis['storage_size_mb']:.1f} MB")
        print(f"  LED count: {current_analysis['led_count']:,}")
        print(f"  Sparsity: {current_analysis['sparsity']:.2f}%")
        print(f"  Reasonable bandwidth: {current_analysis['reasonable_bandwidth']}")
        print(f"  Reasonable diagonals: {current_analysis['reasonable_diagonals']:,}")
        print(f"  Bandwidth efficiency: {current_analysis['bandwidth_efficiency_ratio']:.1f}x reasonable")

        if current_analysis["needs_repair"]:
            print(
                f"  ⚠️  NEEDS REPAIR: storing {current_analysis['bandwidth_efficiency_ratio']:.1f}x more diagonals than reasonable"
            )
        else:
            print("  ✅ Already efficient")
            if not dry_run:
                print("File doesn't need repair - skipping")
                return True
    else:
        print(f"  ❌ {current_analysis.get('error', 'Unknown error')}")
        return False

    if dry_run:
        print("\n🔬 Dry run complete - no changes made")
        return True

    if not current_analysis["needs_repair"]:
        print("File is already efficient - no repair needed")
        return True

    print("\n🔧 Repairing DIA matrix...")

    # Create optimized DIA matrix from dense version
    try:
        dense_ata_dict = data["dense_ata_matrix"].item()
        led_count = current_analysis["led_count"]

        optimized_dia = create_optimized_dia_from_dense(dense_ata_dict, led_count)

        # Create output data with repaired DIA matrix
        output_data = {}
        for key in data.keys():
            if key == "dia_matrix":
                # Replace with optimized version
                output_data[key] = optimized_dia.to_dict()
            else:
                # Copy original data
                output_data[key] = data[key]

        # Set output path if not specified
        if output_path is None:
            stem = input_path.stem
            suffix = input_path.suffix
            output_path = input_path.parent / f"{stem}_repaired{suffix}"

        # Save repaired file
        print(f"💾 Saving repaired file: {output_path.name}")
        np.savez_compressed(str(output_path), **output_data)

        # Show repair results
        repaired_analysis = analyze_current_dia_matrix({"dia_matrix": optimized_dia.to_dict()})

        print("\n✅ Repair completed!")
        print("📊 Repair Results:")
        print(f"  Original diagonals: {current_analysis['k_diagonals']:,}")
        print(f"  Repaired diagonals: {repaired_analysis['k_diagonals']:,}")
        print(
            f"  Reduction: {current_analysis['k_diagonals'] - repaired_analysis['k_diagonals']:,} diagonals ({(current_analysis['k_diagonals'] - repaired_analysis['k_diagonals']) / current_analysis['k_diagonals'] * 100:.1f}%)"
        )
        print(f"  Original storage: {current_analysis['storage_size_mb']:.1f} MB")
        print(f"  Repaired storage: {repaired_analysis['storage_size_mb']:.1f} MB")
        print(
            f"  Storage reduction: {current_analysis['storage_size_mb'] - repaired_analysis['storage_size_mb']:.1f} MB ({(current_analysis['storage_size_mb'] - repaired_analysis['storage_size_mb']) / current_analysis['storage_size_mb'] * 100:.1f}%)"
        )

        # Verify file size
        original_size = input_path.stat().st_size / (1024 * 1024)
        repaired_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {original_size:.1f} MB → {repaired_size:.1f} MB ({repaired_size - original_size:+.1f} MB)")

        return True

    except Exception as e:
        print(f"❌ Repair failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Repair DIA matrices in pattern files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a file (dry run)
  python repair_dia_matrix.py patterns.npz --dry-run

  # Repair a file (creates patterns_repaired.npz)
  python repair_dia_matrix.py patterns.npz

  # Repair with custom output name
  python repair_dia_matrix.py patterns.npz --output fixed_patterns.npz

  # Repair multiple files
  python repair_dia_matrix.py *.npz
""",
    )

    parser.add_argument("files", nargs="+", type=Path, help="Pattern files to repair")
    parser.add_argument("--output", "-o", type=Path, help="Output file path (only for single file)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't repair")

    args = parser.parse_args()

    if len(args.files) > 1 and args.output:
        print("❌ Cannot specify --output when processing multiple files")
        return 1

    success_count = 0
    total_files = len(args.files)

    for file_path in args.files:
        if total_files > 1:
            print(f"\n{'='*20} {file_path.name} {'='*20}")

        if repair_pattern_file(file_path, args.output, args.dry_run):
            success_count += 1
        else:
            print(f"❌ Failed to repair {file_path.name}")

    if total_files > 1:
        print(f"\n{'='*60}")
        print(f"📈 Summary: {success_count}/{total_files} files processed successfully")

        if success_count == total_files:
            print("✅ All files processed successfully!")
        elif success_count > 0:
            print("⚠️  Some files had issues")
        else:
            print("❌ No files were processed successfully")

    return 0 if success_count == total_files else 1


if __name__ == "__main__":
    sys.exit(main())
