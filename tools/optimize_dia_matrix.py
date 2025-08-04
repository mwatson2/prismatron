#!/usr/bin/env python3
"""
Optimize DIA Matrix Tool.

This tool reconstructs DIA matrices from dense format, removing empty diagonals
and optimizing storage efficiency. It converts dense ATA matrices to optimized
DIA format using scipy sparse support and replaces the original DIA matrix in the file.

Usage:
    python optimize_dia_matrix.py --input pattern_file.npz --output optimized_file.npz
    python optimize_dia_matrix.py --input pattern_file.npz --inplace  # Modify in place
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def analyze_dense_matrix_sparsity(dense_matrix: np.ndarray, name: str) -> Dict[str, Any]:
    """Analyze sparsity structure of dense matrix."""
    print(f"\n=== Analyzing {name} ===")

    total_elements = dense_matrix.size
    nonzero_elements = np.count_nonzero(dense_matrix)
    sparsity_percent = (1 - nonzero_elements / total_elements) * 100

    print(f"Total elements: {total_elements:,}")
    print(f"Non-zero elements: {nonzero_elements:,}")
    print(f"Sparsity: {sparsity_percent:.2f}%")

    # Find actual diagonal structure
    rows, cols = np.nonzero(dense_matrix)
    if len(rows) == 0:
        return {"empty": True}

    diagonal_offsets = cols - rows
    unique_offsets = np.unique(diagonal_offsets)
    min_offset = np.min(diagonal_offsets)
    max_offset = np.max(diagonal_offsets)

    # Current DIA storage would need
    current_dia_storage = max_offset - min_offset + 1
    actual_nonzero_diagonals = len(unique_offsets)

    print(f"Diagonal offset range: [{min_offset}, {max_offset}]")
    print(f"Current DIA storage: {current_dia_storage} diagonals")
    print(f"Actual non-zero diagonals: {actual_nonzero_diagonals}")
    print(f"Storage efficiency: {actual_nonzero_diagonals/current_dia_storage*100:.1f}%")

    return {
        "empty": False,
        "total_elements": total_elements,
        "nonzero_elements": nonzero_elements,
        "sparsity_percent": sparsity_percent,
        "min_offset": min_offset,
        "max_offset": max_offset,
        "current_dia_storage": current_dia_storage,
        "actual_nonzero_diagonals": actual_nonzero_diagonals,
        "unique_offsets": unique_offsets,
        "storage_efficiency": actual_nonzero_diagonals / current_dia_storage,
    }


def create_optimized_dia_from_dense(dense_matrices: np.ndarray, led_count: int) -> Dict[str, Any]:
    """Create optimized DIA matrix from dense matrices using scipy sparse conversion."""
    print("\n=== Creating Optimized DIA Matrix ===")

    # Analyze each channel to find optimal storage
    all_offsets = set()
    channel_analyses = []

    for channel in range(3):
        analysis = analyze_dense_matrix_sparsity(dense_matrices[channel], f"Channel {channel}")
        channel_analyses.append(analysis)

        if not analysis["empty"]:
            all_offsets.update(analysis["unique_offsets"])

    if not all_offsets:
        raise ValueError("All matrices are empty!")

    # Create optimized DIA structure
    optimized_offsets = sorted(all_offsets)
    num_diagonals = len(optimized_offsets)

    print("Optimized DIA structure:")
    print(f"  Diagonals needed: {num_diagonals}")
    print(f"  Offset range: [{min(optimized_offsets)}, {max(optimized_offsets)}]")

    # Create DIA data structure
    dia_data_3d = np.zeros((3, num_diagonals, led_count), dtype=np.float32)

    # Fill DIA data for each channel
    for channel in range(3):
        if channel_analyses[channel]["empty"]:
            continue

        dense_matrix = dense_matrices[channel]

        # Convert to scipy sparse DIA format first
        sparse_matrix = sp.csr_matrix(dense_matrix)

        # Extract each diagonal
        for diag_idx, offset in enumerate(optimized_offsets):
            diagonal_values = sparse_matrix.diagonal(k=offset)

            # Store in DIA format
            if offset >= 0:
                # Upper diagonal: store starting from position 0
                dia_data_3d[channel, diag_idx, : len(diagonal_values)] = diagonal_values
            else:
                # Lower diagonal: store starting from position abs(offset)
                start_pos = abs(offset)
                end_pos = start_pos + len(diagonal_values)
                if end_pos <= led_count:
                    dia_data_3d[channel, diag_idx, start_pos:end_pos] = diagonal_values

    # Calculate bandwidth and other properties
    bandwidth = max(abs(min(optimized_offsets)), abs(max(optimized_offsets)))

    # Calculate actual sparsity
    total_elements = 3 * led_count * led_count
    total_nonzero = sum(analysis["nonzero_elements"] for analysis in channel_analyses if not analysis["empty"])
    actual_sparsity = (1 - total_nonzero / total_elements) * 100

    print("Final optimized DIA matrix:")
    print(f"  Bandwidth: {bandwidth}")
    print(f"  Diagonals: {num_diagonals}")
    print(f"  Sparsity: {actual_sparsity:.2f}%")
    print(f"  Memory: {dia_data_3d.nbytes / (1024*1024):.1f}MB")

    # Create DiagonalATAMatrix dict without requiring CuPy
    dia_dict = {
        "led_count": led_count,
        "channels": 3,
        "k": num_diagonals,
        "bandwidth": bandwidth,
        "sparsity": actual_sparsity,
        "storage_dtype": "float32",
        "output_dtype": "float32",
        "dia_offsets_3d": np.array(optimized_offsets, dtype=np.int32),
        "dia_data_3d": dia_data_3d,
        "version": "1.0",
        "format": "diagonal_ata",
        "is_built": True,
    }

    return dia_dict


def optimize_pattern_file(input_path: str, output_path: Optional[str] = None, inplace: bool = False) -> bool:
    """Optimize DIA matrix in pattern file."""
    try:
        input_file = Path(input_path)

        if not input_file.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return False

        print(f"Loading pattern file: {input_path}")
        data = np.load(input_path, allow_pickle=True)

        # Check if we have both dense and DIA matrices
        if "dense_ata_matrix" not in data:
            logger.error("No dense_ata_matrix found - cannot optimize DIA matrix")
            return False

        if "dia_matrix" not in data:
            logger.error("No dia_matrix found - nothing to optimize")
            return False

        # Load existing DIA and dense matrices
        dense_dict = data["dense_ata_matrix"].item()
        dense_matrices = dense_dict["dense_matrices"]
        led_count = dense_dict["led_count"]

        old_dia_dict = data["dia_matrix"].item()
        old_dia_k = old_dia_dict["k"]
        old_dia_bandwidth = old_dia_dict["bandwidth"]
        old_dia_sparsity = old_dia_dict["sparsity"]

        print("Current DIA matrix:")
        print(f"  Diagonals: {old_dia_k}")
        print(f"  Bandwidth: {old_dia_bandwidth}")
        print(f"  Sparsity: {old_dia_sparsity:.2f}%")

        # Create optimized DIA matrix
        start_time = time.time()
        optimized_dia = create_optimized_dia_from_dense(dense_matrices, led_count)
        optimization_time = time.time() - start_time

        print(f"\nOptimization completed in {optimization_time:.2f}s")

        # Calculate improvements
        diagonal_reduction = (old_dia_k - optimized_dia["k"]) / old_dia_k * 100
        old_memory = old_dia_k * led_count * 3 * 4 / (1024 * 1024)  # Rough estimate
        new_memory = optimized_dia["k"] * led_count * 3 * 4 / (1024 * 1024)
        memory_reduction = (old_memory - new_memory) / old_memory * 100

        print("\nOptimization results:")
        print(f"  Diagonal reduction: {diagonal_reduction:.1f}% ({old_dia_k} → {optimized_dia['k']})")
        print(f"  Memory reduction: {memory_reduction:.1f}% ({old_memory:.1f}MB → {new_memory:.1f}MB)")
        print(f"  Storage efficiency: {optimized_dia['k']}/{old_dia_k} = {optimized_dia['k']/old_dia_k:.1%}")

        # Prepare output data
        output_data = {}

        # Copy all existing data
        for key in data.keys():
            if key != "dia_matrix":
                output_data[key] = data[key]

        # Add optimized DIA matrix
        output_data["dia_matrix"] = optimized_dia

        # Update metadata
        if "metadata" in output_data:
            metadata = (
                output_data["metadata"].item() if hasattr(output_data["metadata"], "item") else output_data["metadata"]
            )
            metadata["dia_optimization_tool"] = "optimize_dia_matrix.py"
            metadata["dia_optimization_timestamp"] = time.time()
            metadata["dia_optimization_diagonal_reduction"] = diagonal_reduction
            metadata["dia_optimization_memory_reduction"] = memory_reduction
            output_data["metadata"] = metadata

        # Determine output path
        if inplace:
            output_file = input_file
        elif output_path:
            output_file = Path(output_path)
        else:
            stem = input_file.stem
            suffix = input_file.suffix
            output_file = input_file.parent / f"{stem}_optimized{suffix}"

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save optimized file
        print(f"Saving optimized file: {output_file}")
        np.savez_compressed(output_file, **output_data)

        # Report file size changes
        if not inplace:
            input_size = input_file.stat().st_size / (1024 * 1024)
            output_size = output_file.stat().st_size / (1024 * 1024)
            size_change = (output_size - input_size) / input_size * 100

            print("\nFile size comparison:")
            print(f"  Input: {input_size:.1f}MB")
            print(f"  Output: {output_size:.1f}MB")
            print(f"  Change: {size_change:+.1f}%")

        print("\n✅ DIA matrix optimization completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to optimize DIA matrix: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Optimize DIA matrix by removing empty diagonals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create optimized copy
  python optimize_dia_matrix.py --input patterns.npz --output patterns_optimized.npz

  # Modify in place
  python optimize_dia_matrix.py --input patterns.npz --inplace

  # Auto-generate output name
  python optimize_dia_matrix.py --input patterns.npz
""",
    )

    parser.add_argument("--input", "-i", required=True, help="Input NPZ file with dense_ata_matrix")
    parser.add_argument("--output", "-o", help="Output NPZ file (optional)")
    parser.add_argument("--inplace", action="store_true", help="Modify input file in place")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Validate arguments
    if args.output and args.inplace:
        logger.error("Cannot specify both --output and --inplace")
        return 1

    try:
        success = optimize_pattern_file(args.input, args.output, args.inplace)
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
