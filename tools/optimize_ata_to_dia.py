#!/usr/bin/env python3
"""
Optimize ATA Matrix to Efficient DIA Format.

This tool takes pattern files with either dense ATA matrices or inefficient DIA matrices
and creates optimized DIA matrices with only the significant diagonals.

Usage:
    python optimize_ata_to_dia.py pattern_file.npz
    python optimize_ata_to_dia.py pattern_file.npz --output optimized_file.npz
    python optimize_ata_to_dia.py pattern_file.npz --dry-run  # Show analysis only
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

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


def analyze_matrix_structure(dense_matrices: np.ndarray, led_count: int) -> Dict[str, Any]:
    """
    Analyze the structure of dense ATA matrices to determine optimal diagonal storage.

    Args:
        dense_matrices: Shape (3, led_count, led_count) dense ATA matrices
        led_count: Number of LEDs

    Returns:
        Analysis dictionary with diagonal statistics
    """
    print("Analyzing matrix structure...")

    # Convert each channel to DIA format and analyze
    all_offsets = set()
    channel_analyses = []

    for channel in range(3):
        channel_name = ["R", "G", "B"][channel]
        print(f"  Channel {channel} ({channel_name}):")

        dense_channel = dense_matrices[channel]

        # Convert to sparse then DIA to find all non-zero diagonals
        sparse_channel = sp.csr_matrix(dense_channel)
        dia_channel = sp.dia_matrix(sparse_channel)

        # Find matrix statistics
        matrix_max = np.max(np.abs(dense_channel))
        matrix_nnz = sparse_channel.nnz

        # Calculate significance threshold
        # For LED patterns, we expect a sparse block-diagonal structure
        # Use smart filtering to avoid storing more diagonals than dense would use
        if matrix_max > 0:
            # Use 5% of max as threshold - captures main structure
            significance_threshold = matrix_max * 0.05
            # Require minimal density - even sparse diagonals can be important for LED patterns
            min_nnz_per_diagonal = max(5, int(led_count * 0.001))  # 0.1% density minimum
        else:
            significance_threshold = 1e-10
            min_nnz_per_diagonal = 1

        print(f"    Matrix max: {matrix_max:.3e}")
        print(f"    Matrix nnz: {matrix_nnz:,}")
        print(f"    Total diagonals with any non-zero: {len(dia_channel.offsets)}")

        # Analyze each diagonal
        significant_offsets = []
        insignificant_offsets = []
        empty_offsets = []

        for i, offset in enumerate(dia_channel.offsets):
            diagonal_data = dia_channel.data[i]
            nnz_count = np.count_nonzero(diagonal_data)
            max_val = np.max(np.abs(diagonal_data))
            significant_values = np.sum(np.abs(diagonal_data) > significance_threshold)

            if nnz_count == 0:
                empty_offsets.append(offset)
            elif significant_values < min_nnz_per_diagonal:
                insignificant_offsets.append(offset)
            else:
                significant_offsets.append(offset)
                all_offsets.add(offset)

        channel_analyses.append(
            {
                "channel": channel,
                "matrix_max": matrix_max,
                "matrix_nnz": matrix_nnz,
                "total_diagonals": len(dia_channel.offsets),
                "significant_diagonals": len(significant_offsets),
                "insignificant_diagonals": len(insignificant_offsets),
                "empty_diagonals": len(empty_offsets),
                "significance_threshold": significance_threshold,
                "min_nnz_per_diagonal": min_nnz_per_diagonal,
                "significant_offsets": significant_offsets,
            }
        )

        print(f"    Significant diagonals: {len(significant_offsets)}")
        print(f"    Insignificant diagonals: {len(insignificant_offsets)}")
        print(f"    Empty diagonals: {len(empty_offsets)}")

    # Calculate unified statistics
    sorted_offsets = sorted(all_offsets)
    bandwidth = max(abs(o) for o in sorted_offsets) if sorted_offsets else 0

    print(f"\n  Unified diagonal structure:")
    print(f"    Total significant diagonals (union): {len(sorted_offsets)}")
    print(f"    Bandwidth: {bandwidth}")
    print(
        f"    Offset range: [{min(sorted_offsets) if sorted_offsets else 0}, {max(sorted_offsets) if sorted_offsets else 0}]"
    )

    # Check if diagonals are contiguous or sparse
    if sorted_offsets:
        gaps = []
        for i in range(1, len(sorted_offsets)):
            gap = sorted_offsets[i] - sorted_offsets[i - 1] - 1
            if gap > 0:
                gaps.append(gap)

        if gaps:
            print(f"    Diagonal gaps: {len(gaps)} gaps, max gap: {max(gaps)}")
        else:
            print(f"    Diagonals are contiguous")

    return {
        "led_count": led_count,
        "channel_analyses": channel_analyses,
        "all_significant_offsets": sorted_offsets,
        "bandwidth": bandwidth,
        "total_significant_diagonals": len(sorted_offsets),
        "memory_dia_mb": (3 * len(sorted_offsets) * led_count * 4) / (1024 * 1024),
        "memory_dense_mb": (3 * led_count * led_count * 4) / (1024 * 1024),
    }


def create_optimized_dia_from_dense(dense_ata_dict: Dict[str, Any]) -> DiagonalATAMatrix:
    """
    Create optimized DIA matrix from dense ATA matrix.

    Args:
        dense_ata_dict: Dictionary from dense_ata_matrix in pattern file

    Returns:
        Optimized DiagonalATAMatrix with only significant diagonals
    """
    print("\nCreating optimized DIA matrix from dense ATA matrix...")

    # Load dense matrix
    dense_ata = DenseATAMatrix.from_dict(dense_ata_dict)
    dense_matrices = dense_ata_dict["dense_matrices"]  # Shape: (3, led_count, led_count)
    led_count = dense_ata.led_count

    print(f"  Dense ATA shape: {dense_matrices.shape}")
    print(f"  Dense ATA memory: {dense_ata_dict['memory_mb']:.1f} MB")

    # Analyze structure
    analysis = analyze_matrix_structure(dense_matrices, led_count)

    # Create new DIA matrix instance
    dia_matrix = DiagonalATAMatrix(led_count=led_count, crop_size=64)

    # Set diagonal offsets
    dia_matrix.dia_offsets = np.array(analysis["all_significant_offsets"], dtype=np.int32)
    dia_matrix.k = len(dia_matrix.dia_offsets)

    print(f"\n  Building optimized DIA structure with {dia_matrix.k} diagonals...")

    if dia_matrix.k > 0:
        # Create unified 3D DIA data structure
        dia_matrix.dia_data_cpu = np.zeros((3, dia_matrix.k, led_count), dtype=np.float32)

        # Create mapping from offset to index
        offset_to_idx = {offset: i for i, offset in enumerate(dia_matrix.dia_offsets)}

        # Fill data for each channel
        for channel in range(3):
            channel_name = ["R", "G", "B"][channel]
            print(f"    Processing channel {channel} ({channel_name})...")

            dense_channel = dense_matrices[channel]

            # Convert to DIA format
            sparse_channel = sp.csr_matrix(dense_channel)
            dia_channel = sp.dia_matrix(sparse_channel)

            # Get significance threshold for this channel
            channel_analysis = analysis["channel_analyses"][channel]
            significance_threshold = channel_analysis["significance_threshold"]
            min_nnz_per_diagonal = channel_analysis["min_nnz_per_diagonal"]

            # Copy significant diagonals
            copied = 0
            for i, offset in enumerate(dia_channel.offsets):
                if offset in offset_to_idx:
                    diagonal_data = dia_channel.data[i]
                    significant_values = np.sum(np.abs(diagonal_data) > significance_threshold)

                    if significant_values >= min_nnz_per_diagonal:
                        unified_idx = offset_to_idx[offset]
                        dia_matrix.dia_data_cpu[channel, unified_idx, :] = diagonal_data.astype(np.float32)
                        copied += 1

            print(f"      Copied {copied} significant diagonals")
    else:
        dia_matrix.dia_data_cpu = np.zeros((3, 0, led_count), dtype=np.float32)

    # Update GPU caches and metadata
    dia_matrix._update_dia_offsets_cache()
    dia_matrix._update_dia_data_cache()

    # Calculate metadata
    total_nnz = 0
    for channel in range(3):
        channel_nnz = np.count_nonzero(dia_matrix.dia_data_cpu[channel])
        dia_matrix.channel_nnz[channel] = channel_nnz
        total_nnz += channel_nnz

    dia_matrix.nnz = total_nnz
    dia_matrix.bandwidth = int(np.max(np.abs(dia_matrix.dia_offsets))) if dia_matrix.k > 0 else 0
    total_elements = 3 * led_count * led_count
    dia_matrix.sparsity = (1 - total_nnz / total_elements) * 100 if total_elements > 0 else 100

    print(f"\n  Optimized DIA matrix created:")
    print(f"    Diagonals: {dia_matrix.k}")
    print(f"    Bandwidth: {dia_matrix.bandwidth}")
    print(f"    Memory: {dia_matrix.dia_data_cpu.nbytes / (1024*1024):.1f} MB")
    print(f"    Sparsity: {dia_matrix.sparsity:.3f}%")
    print(f"    Non-zeros: {dia_matrix.nnz:,}")

    return dia_matrix


def optimize_pattern_file(input_path: Path, output_path: Path = None, dry_run: bool = False) -> bool:
    """
    Optimize a pattern file by creating efficient DIA matrix from dense or inefficient DIA.

    Args:
        input_path: Input pattern file path
        output_path: Output path (optional, will append _optimized if not specified)
        dry_run: If True, only analyze without saving

    Returns:
        True if successful
    """
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

    # Check what matrix formats are available
    has_dense = "dense_ata_matrix" in data
    has_dia = "dia_matrix" in data

    if not has_dense and not has_dia:
        print("❌ File contains neither dense_ata_matrix nor dia_matrix")
        return False

    print(f"📊 Available formats:")
    if has_dense:
        dense_dict = data["dense_ata_matrix"].item()
        print(f"  ✓ Dense ATA matrix: {dense_dict['led_count']} LEDs, {dense_dict['memory_mb']:.1f} MB")
    if has_dia:
        dia_dict = data["dia_matrix"].item()
        print(f"  ✓ DIA matrix: {dia_dict['led_count']} LEDs, {dia_dict['k']} diagonals")

    # Prefer dense if available (more accurate source)
    if has_dense:
        print("\n🔧 Using dense ATA matrix as source...")
        dense_dict = data["dense_ata_matrix"].item()
        optimized_dia = create_optimized_dia_from_dense(dense_dict)
    else:
        print("\n⚠️  No dense matrix available, cannot optimize")
        print("   Use convert_dia_to_dense.py first to create dense matrix")
        return False

    if dry_run:
        print("\n🔬 Dry run complete - no changes made")

        # Compare with existing DIA if present
        if has_dia:
            original_dia = data["dia_matrix"].item()
            print("\n📊 Comparison with existing DIA:")
            print(f"  Original: {original_dia['k']} diagonals, bandwidth={original_dia['bandwidth']}")
            print(f"  Optimized: {optimized_dia.k} diagonals, bandwidth={optimized_dia.bandwidth}")
            print(
                f"  Reduction: {original_dia['k'] - optimized_dia.k} diagonals ({(original_dia['k'] - optimized_dia.k) / original_dia['k'] * 100:.1f}%)"
            )

            original_size_mb = (3 * original_dia["k"] * original_dia["led_count"] * 4) / (1024 * 1024)
            optimized_size_mb = optimized_dia.dia_data_cpu.nbytes / (1024 * 1024)
            print(
                f"  Storage: {original_size_mb:.1f} MB → {optimized_size_mb:.1f} MB ({(original_size_mb - optimized_size_mb) / original_size_mb * 100:.1f}% reduction)"
            )

        return True

    # Create output data with optimized DIA matrix
    print("\n💾 Preparing output file...")
    output_data = {}

    # Copy all original data
    for key in data.keys():
        if key == "dia_matrix":
            # Replace with optimized version
            output_data[key] = optimized_dia.to_dict()
        else:
            # Copy original
            output_data[key] = data[key]

    # If there was no DIA matrix before, add it
    if not has_dia:
        output_data["dia_matrix"] = optimized_dia.to_dict()

    # Update metadata
    if "metadata" in output_data:
        metadata = (
            output_data["metadata"].item() if hasattr(output_data["metadata"], "item") else output_data["metadata"]
        )
        metadata["optimization_tool"] = "optimize_ata_to_dia.py"
        metadata["optimization_timestamp"] = time.time()
        metadata["optimized_diagonals"] = optimized_dia.k
        metadata["optimized_bandwidth"] = optimized_dia.bandwidth
        output_data["metadata"] = metadata

    # Set output path if not specified
    if output_path is None:
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_optimized{suffix}"

    # Save optimized file
    print(f"💾 Saving optimized file: {output_path.name}")
    np.savez_compressed(str(output_path), **output_data)

    # Show results
    print("\n✅ Optimization completed!")

    if has_dia:
        original_dia = data["dia_matrix"].item()
        print("\n📊 Optimization Results:")
        print(f"  Original diagonals: {original_dia['k']:,}")
        print(f"  Optimized diagonals: {optimized_dia.k:,}")
        print(
            f"  Reduction: {original_dia['k'] - optimized_dia.k:,} diagonals ({(original_dia['k'] - optimized_dia.k) / original_dia['k'] * 100:.1f}%)"
        )

        original_size_mb = (3 * original_dia["k"] * original_dia["led_count"] * 4) / (1024 * 1024)
        optimized_size_mb = optimized_dia.dia_data_cpu.nbytes / (1024 * 1024)
        print(
            f"  DIA storage: {original_size_mb:.1f} MB → {optimized_size_mb:.1f} MB ({(original_size_mb - optimized_size_mb) / original_size_mb * 100:.1f}% reduction)"
        )
    else:
        print("\n📊 Created optimized DIA matrix:")
        print(f"  Diagonals: {optimized_dia.k:,}")
        print(f"  Bandwidth: {optimized_dia.bandwidth}")
        print(f"  Storage: {optimized_dia.dia_data_cpu.nbytes / (1024 * 1024):.1f} MB")

    # File size comparison
    original_size = input_path.stat().st_size / (1024 * 1024)
    optimized_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {original_size:.1f} MB → {optimized_size:.1f} MB ({optimized_size - original_size:+.1f} MB)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Optimize ATA matrices to efficient DIA format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a file (dry run)
  python optimize_ata_to_dia.py patterns.npz --dry-run

  # Optimize a file (creates patterns_optimized.npz)
  python optimize_ata_to_dia.py patterns.npz

  # Optimize with custom output name
  python optimize_ata_to_dia.py patterns.npz --output fixed_patterns.npz

  # Optimize multiple files
  python optimize_ata_to_dia.py *.npz
""",
    )

    parser.add_argument("files", nargs="+", type=Path, help="Pattern files to optimize")
    parser.add_argument("--output", "-o", type=Path, help="Output file path (only for single file)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't optimize")

    args = parser.parse_args()

    if len(args.files) > 1 and args.output:
        print("❌ Cannot specify --output when processing multiple files")
        return 1

    success_count = 0
    total_files = len(args.files)

    for file_path in args.files:
        if total_files > 1:
            print(f"\n{'='*20} {file_path.name} {'='*20}")

        if optimize_pattern_file(file_path, args.output, args.dry_run):
            success_count += 1
        else:
            print(f"❌ Failed to optimize {file_path.name}")

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
