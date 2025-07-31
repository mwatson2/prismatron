#!/usr/bin/env python3
"""
Debug script to analyze the actual diagonal structure of the dense and DIA matrices
from the captured patterns to understand why there's a discrepancy in reported sparsity.
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from utils.dense_ata_matrix import DenseATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix


def analyze_matrix_diagonals(matrix: np.ndarray, name: str) -> dict:
    """Analyze diagonal structure of a matrix."""
    print(f"\n=== {name} Analysis ===")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix dtype: {matrix.dtype}")

    # Basic statistics
    total_elements = matrix.size
    nonzero_elements = np.count_nonzero(matrix)
    sparsity = (1 - nonzero_elements / total_elements) * 100

    print(f"Total elements: {total_elements:,}")
    print(f"Non-zero elements: {nonzero_elements:,}")
    print(f"Sparsity: {sparsity:.2f}%")

    # Find which diagonals have non-zero elements
    rows, cols = np.nonzero(matrix)
    if len(rows) == 0:
        print("Matrix is completely zero!")
        return {"diagonals": 0, "bandwidth": 0, "max_distance": 0}

    # Calculate diagonal offsets for each non-zero element
    diagonal_offsets = cols - rows  # Positive = upper diagonal, negative = lower diagonal

    # Get unique diagonals that have at least one non-zero element
    unique_diagonals = np.unique(diagonal_offsets)
    num_nonzero_diagonals = len(unique_diagonals)

    # Calculate bandwidth (max distance from main diagonal)
    max_distance = np.max(np.abs(diagonal_offsets))
    bandwidth = max_distance

    # Calculate theoretical DIA storage requirements
    min_offset = np.min(diagonal_offsets)
    max_offset = np.max(diagonal_offsets)
    dia_storage_diagonals = max_offset - min_offset + 1

    print(f"Actual non-zero diagonals: {num_nonzero_diagonals}")
    print(f"Diagonal offset range: [{min_offset}, {max_offset}]")
    print(f"DIA storage would need: {dia_storage_diagonals} diagonals")
    print(f"Bandwidth (max distance from main diagonal): {bandwidth}")

    # Check diagonal distribution
    print(f"Main diagonal (offset=0) elements: {np.sum(diagonal_offsets == 0)}")
    print(f"Elements within ±10 of main diagonal: {np.sum(np.abs(diagonal_offsets) <= 10)}")
    print(f"Elements within ±50 of main diagonal: {np.sum(np.abs(diagonal_offsets) <= 50)}")
    print(f"Elements within ±100 of main diagonal: {np.sum(np.abs(diagonal_offsets) <= 100)}")

    # Check for block structure (64x64 blocks as mentioned by user)
    block_distances = np.abs(diagonal_offsets) // 64
    unique_blocks = np.unique(block_distances)
    print(f"Non-zero elements span {len(unique_blocks)} 64-element blocks from diagonal")
    print(f"Block distances: {unique_blocks[:10]}{'...' if len(unique_blocks) > 10 else ''}")

    return {
        "total_elements": total_elements,
        "nonzero_elements": nonzero_elements,
        "sparsity": sparsity,
        "actual_diagonals": num_nonzero_diagonals,
        "dia_storage_diagonals": dia_storage_diagonals,
        "bandwidth": bandwidth,
        "min_offset": min_offset,
        "max_offset": max_offset,
        "diagonal_offsets": diagonal_offsets,
    }


def main():
    # Load the capture file
    file_path = Path("diffusion_patterns/capture-0728-01-comparison-fixed.npz")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"Loading capture file: {file_path}")
    data = np.load(file_path, allow_pickle=True)

    print(f"Available keys: {list(data.keys())}")

    # Load DIA matrix
    dia_dict = data["dia_matrix"].item()
    dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)

    print(f"\nDIA Matrix Info:")
    print(f"  LED count: {dia_matrix.led_count}")
    print(f"  Reported k diagonals: {dia_matrix.k}")
    print(f"  Reported bandwidth: {dia_matrix.bandwidth}")
    print(f"  Reported sparsity: {dia_matrix.sparsity:.2f}%")

    # Load Dense matrix
    dense_dict = data["dense_ata_matrix"].item()
    dense_matrix = DenseATAMatrix.from_dict(dense_dict)

    print(f"\nDense Matrix Info:")
    print(f"  LED count: {dense_matrix.led_count}")
    print(f"  Memory: {dense_matrix.memory_mb:.1f}MB")
    print(f"  Matrix shape: {dense_matrix.dense_matrices_cpu.shape}")

    # Analyze each channel of the dense matrix
    for channel in range(3):
        channel_matrix = dense_matrix.dense_matrices_cpu[channel]
        stats = analyze_matrix_diagonals(channel_matrix, f"Dense Channel {channel}")

        # Compare with DIA matrix for this channel
        print(f"\n--- DIA vs Dense Comparison for Channel {channel} ---")
        print(f"DIA reported diagonals: {dia_matrix.k}")
        print(f"Dense actual non-zero diagonals: {stats['actual_diagonals']}")
        print(f"Dense DIA storage needed: {stats['dia_storage_diagonals']}")

        if stats["dia_storage_diagonals"] != dia_matrix.k:
            print(f"❌ MISMATCH: DIA says {dia_matrix.k} diagonals, Dense needs {stats['dia_storage_diagonals']}")
        else:
            print(f"✅ Match: Both indicate {dia_matrix.k} diagonals needed")

        # Check if DIA is actually storing unnecessary diagonals
        if stats["actual_diagonals"] < stats["dia_storage_diagonals"]:
            efficiency = stats["actual_diagonals"] / stats["dia_storage_diagonals"] * 100
            print(
                f"📊 DIA efficiency: {efficiency:.1f}% ({stats['actual_diagonals']}/{stats['dia_storage_diagonals']} diagonals have data)"
            )

        # Check the distribution - is it really sparse or dense?
        if stats["sparsity"] < 50:  # Less than 50% sparse
            print(f"⚠️  WARNING: Matrix is quite dense ({100-stats['sparsity']:.1f}% non-zero)")
            print("   DIA format may not be optimal for this matrix")

        if channel == 0:  # Only analyze first channel in detail to avoid too much output
            # Show sample of where non-zeros are located
            rows, cols = np.nonzero(channel_matrix)
            if len(rows) > 0:
                print(f"\nSample non-zero locations (first 10):")
                for i in range(min(10, len(rows))):
                    r, c = rows[i], cols[i]
                    offset = c - r
                    print(f"  [{r:4d}, {c:4d}] = {channel_matrix[r,c]:.6f} (diagonal offset: {offset:+4d})")

    # Final analysis
    print(f"\n{'='*60}")
    print("SUMMARY AND CONCLUSIONS")
    print(f"{'='*60}")

    # Check if this looks like it should be sparse
    channel_0_stats = analyze_matrix_diagonals(dense_matrix.dense_matrices_cpu[0], "Channel 0 Summary")

    if channel_0_stats["sparsity"] > 95:
        print("✅ Matrix is very sparse - DIA format is appropriate")
    elif channel_0_stats["sparsity"] > 90:
        print("📊 Matrix is moderately sparse - DIA format may be appropriate")
    else:
        print("❌ Matrix is quite dense - DIA format may be inefficient")

    # Check if the pattern matches expected 64x64 block overlap
    expected_pattern = channel_0_stats["bandwidth"] < 200  # Rough heuristic
    if expected_pattern:
        print("✅ Bandwidth suggests local 64x64 block pattern")
    else:
        print("❌ Large bandwidth suggests whole-image capture (not 64x64 blocks)")
        print("   This could explain why DIA matrix is less sparse than expected")


if __name__ == "__main__":
    main()
