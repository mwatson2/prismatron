#!/usr/bin/env python3
"""
Analyze A^T A matrix structure to understand diagonal count issues.

This script loads synthetic patterns and analyzes the A^T A matrix structure
to understand why we're getting too many diagonals for efficient DIA format.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from tools.led_position_utils import analyze_matrix_bandwidth


def analyze_captured_ata_matrices(data):
    """Analyze ATA matrices from captured pattern format."""
    print("=== Analyzing Captured ATA Matrices ===")

    # Load both matrix formats
    dense_ata_dict = data["dense_ata_matrix"].item()
    dia_matrix_dict = data["dia_matrix"].item()

    print("1. Dense ATA Matrix Info:")
    dense_matrices = dense_ata_dict["dense_matrices"]  # Shape: (3, led_count, led_count)
    led_count = dense_ata_dict["led_count"]
    print(f"   Shape: {dense_matrices.shape}")
    print(f"   LED count: {led_count}")
    print(f"   Memory: {dense_ata_dict['memory_mb']:.1f} MB")
    print(f"   Storage dtype: {dense_ata_dict['storage_dtype']}")
    print()

    print("2. DIA Matrix Info:")
    dia_data_3d = dia_matrix_dict["dia_data_3d"]  # Shape: (3, k, led_count)
    dia_offsets_3d = dia_matrix_dict["dia_offsets_3d"]  # Shape: (k,)
    k_diagonals = dia_matrix_dict["k"]
    bandwidth = dia_matrix_dict["bandwidth"]
    sparsity = dia_matrix_dict["sparsity"]

    print(f"   DIA data shape: {dia_data_3d.shape}")
    print(f"   Number of diagonals: {k_diagonals}")
    print(f"   Bandwidth: {bandwidth}")
    print(f"   Sparsity: {sparsity:.2f}%")
    print(f"   Storage dtype: {dia_matrix_dict['storage_dtype']}")
    print()

    # Analyze diagonal usage for first channel
    print("3. Diagonal Usage Analysis (Channel 0):")
    dia_channel_0 = dia_data_3d[0]  # Shape: (k, led_count)

    # Count how many diagonals actually have non-zero values
    diagonal_nnz = []
    for i in range(k_diagonals):
        diagonal = dia_channel_0[i, :]
        nnz = np.count_nonzero(diagonal)
        diagonal_nnz.append(nnz)

    diagonal_nnz = np.array(diagonal_nnz)
    non_empty_diagonals = np.sum(diagonal_nnz > 0)

    print(f"   Total stored diagonals: {k_diagonals}")
    print(f"   Non-empty diagonals: {non_empty_diagonals}")
    print(f"   Empty diagonals: {k_diagonals - non_empty_diagonals}")
    print(f"   Percentage empty: {(k_diagonals - non_empty_diagonals) / k_diagonals * 100:.1f}%")
    print()

    # Show diagonal offset distribution
    print("4. Diagonal Offset Distribution:")
    print(f"   Offset range: {dia_offsets_3d.min()} to {dia_offsets_3d.max()}")

    # Group by offset ranges to see distribution
    offset_ranges = [
        ("Main diagonal", lambda x: x == 0),
        ("Near diagonal (±1-10)", lambda x: (np.abs(x) >= 1) & (np.abs(x) <= 10)),
        ("Medium distance (±11-100)", lambda x: (np.abs(x) >= 11) & (np.abs(x) <= 100)),
        ("Far distance (±101-500)", lambda x: (np.abs(x) >= 101) & (np.abs(x) <= 500)),
        ("Very far (±501+)", lambda x: np.abs(x) >= 501),
    ]

    for range_name, range_func in offset_ranges:
        mask = range_func(dia_offsets_3d)
        count = np.sum(mask)
        if count > 0:
            # Check how many of these diagonals are actually used
            used_in_range = 0
            for i in np.where(mask)[0]:
                if diagonal_nnz[i] > 0:
                    used_in_range += 1
            print(f"   {range_name}: {count} stored, {used_in_range} used ({used_in_range/count*100:.1f}% usage)")
    print()

    # Compare to dense matrix structure
    print("5. Comparing DIA vs Dense Matrix Structure:")
    dense_channel_0 = dense_matrices[0]  # Shape: (led_count, led_count)

    # Count actual non-zeros in dense matrix
    dense_nnz = np.count_nonzero(dense_channel_0)
    dense_sparsity = (1 - dense_nnz / (led_count * led_count)) * 100

    print(f"   Dense matrix non-zeros: {dense_nnz:,}")
    print(f"   Dense matrix sparsity: {dense_sparsity:.2f}%")

    # Analyze bandwidth of dense matrix
    bandwidth_stats = analyze_matrix_bandwidth(dense_channel_0)
    print(f"   Actual bandwidth from dense: {bandwidth_stats['bandwidth']}")
    print(f"   Actual diagonals needed: {bandwidth_stats['num_diagonals']}")
    print()

    # Efficiency comparison
    print("6. Storage Efficiency Analysis:")
    dia_storage_size = dia_data_3d.nbytes / (1024 * 1024)  # MB
    dense_storage_size = dense_matrices.nbytes / (1024 * 1024)  # MB

    print(f"   DIA storage: {dia_storage_size:.1f} MB")
    print(f"   Dense storage: {dense_storage_size:.1f} MB")
    print(f"   DIA efficiency: {dense_storage_size / dia_storage_size:.1f}x smaller than dense")

    # But account for empty diagonals
    actual_diagonals_needed = bandwidth_stats["num_diagonals"]
    optimal_dia_size = actual_diagonals_needed * led_count * 4 / (1024 * 1024)  # float32
    efficiency_loss = dia_storage_size / optimal_dia_size

    print(f"   Optimal DIA storage: {optimal_dia_size:.1f} MB")
    print(f"   Current DIA efficiency loss: {efficiency_loss:.1f}x (due to empty diagonals)")

    if efficiency_loss > 2:
        print(f"   ❌ DIA matrix is storing {efficiency_loss:.1f}x more data than needed!")
        print(f"   Recommendation: Optimize DIA matrix to store only {actual_diagonals_needed} diagonals")
    else:
        print("   ✅ DIA storage is reasonably efficient")


def analyze_ata_structure(pattern_file="synthetic_1000_64x64_fixed2.npz"):
    """Analyze A^T A matrix structure in detail."""
    print("=== Analyzing A^T A Matrix Structure ===")
    print(f"Pattern file: {pattern_file}")
    print()

    # Load patterns - support both old synthetic format and new captured format
    pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / pattern_file
    if not pattern_path.exists():
        print(f"❌ Pattern file not found: {pattern_path}")
        return

    try:
        data = np.load(str(pattern_path), allow_pickle=True)

        # Check for new mixed tensor format (captured data)
        if "dense_ata_matrix" in data and "dia_matrix" in data:
            print("Found captured pattern format with both DIA and dense ATA matrices")
            analyze_captured_ata_matrices(data)
            return

        # Old synthetic format
        if "diffusion_matrix" not in data:
            print(f"❌ No diffusion_matrix found in {pattern_file}")
            return

        diffusion_dict = data["diffusion_matrix"].item()

        # Reconstruct diffusion matrix A: (pixels, LEDs*3)
        A = sp.csc_matrix(
            (
                diffusion_dict["csc_data"],
                diffusion_dict["csc_indices"],
                diffusion_dict["csc_indptr"],
            ),
            shape=diffusion_dict["csc_shape"],
        )

        print("1. Diffusion matrix A analysis:")
        print(f"   Shape: {A.shape}")
        print(f"   NNZ: {A.nnz:,}")
        print(f"   Sparsity: {A.nnz / (A.shape[0] * A.shape[1]) * 100:.2f}%")
        print(f"   Memory: {A.data.nbytes / 1e6:.1f} MB")
        print()

        # Compute A^T A
        print("2. Computing A^T A...")
        ata_matrix = A.T @ A
        print(f"   A^T A shape: {ata_matrix.shape}")
        print(f"   A^T A NNZ: {ata_matrix.nnz:,}")
        print(f"   A^T A sparsity: {ata_matrix.nnz / (ata_matrix.shape[0] * ata_matrix.shape[1]) * 100:.2f}%")
        print()

        # Analyze A^T A bandwidth
        print("3. A^T A bandwidth analysis (before reordering)...")
        # Convert to dense for analysis (use subset if too large)
        subset_size = min(1000, ata_matrix.shape[0])
        ata_subset = ata_matrix[:subset_size, :subset_size]
        ata_dense = ata_subset.toarray()

        bandwidth_stats = analyze_matrix_bandwidth(ata_dense)
        print(f"   Matrix size: {bandwidth_stats['matrix_size']}x{bandwidth_stats['matrix_size']}")
        print(f"   Non-zeros: {bandwidth_stats['nnz']:,}")
        print(f"   Sparsity: {bandwidth_stats['sparsity']:.2f}%")
        print(f"   Bandwidth: {bandwidth_stats['bandwidth']}")
        print(f"   Number of diagonals: {bandwidth_stats['num_diagonals']}")
        print(f"   Max diagonal offset: {bandwidth_stats['max_diagonal_offset']}")
        print()

        # Check if A^T A has expected structure
        print("4. Checking A^T A structure...")

        # A^T A should be block diagonal with 3x3 blocks for each LED pair
        # Let's analyze the pattern for first few LEDs
        led_count = A.shape[1] // 3
        print(f"   LED count: {led_count}")
        print(f"   Expected block structure: {led_count} LEDs × 3 channels = {led_count * 3} rows/cols")

        # Check if there are unexpected long-range connections
        rows, cols = ata_matrix.nonzero()

        # Convert to LED indices (divide by 3)
        led_rows = rows // 3
        led_cols = cols // 3
        led_distances = np.abs(led_rows - led_cols)

        print("   LED distance statistics:")
        print(f"     Min LED distance: {led_distances.min()}")
        print(f"     Max LED distance: {led_distances.max()}")
        print(f"     Mean LED distance: {led_distances.mean():.1f}")
        print(f"     90th percentile: {np.percentile(led_distances, 90):.1f}")
        print(f"     95th percentile: {np.percentile(led_distances, 95):.1f}")
        print(f"     99th percentile: {np.percentile(led_distances, 99):.1f}")

        # Count connections by distance
        distance_counts = np.bincount(led_distances)
        print("   Connection counts by LED distance:")
        for dist in range(min(20, len(distance_counts))):
            if distance_counts[dist] > 0:
                print(f"     Distance {dist}: {distance_counts[dist]:,} connections")

        if len(distance_counts) > 20:
            far_connections = np.sum(distance_counts[20:])
            print(f"     Distance 20+: {far_connections:,} connections")
        print()

        # Check if RCM reordering is being applied to A^T A
        print("5. Testing RCM reordering on A^T A...")
        try:
            # Convert to adjacency (just structure, not values)
            ata_adj = (ata_matrix != 0).astype(int)
            rcm_order = reverse_cuthill_mckee(ata_adj, symmetric_mode=True)

            # Reorder A^T A
            ata_rcm = ata_matrix[rcm_order][:, rcm_order]

            # Analyze bandwidth after RCM
            subset_rcm = ata_rcm[:subset_size, :subset_size]
            ata_rcm_dense = subset_rcm.toarray()
            bandwidth_stats_rcm = analyze_matrix_bandwidth(ata_rcm_dense)

            print("   After RCM reordering:")
            print(f"     Bandwidth: {bandwidth_stats_rcm['bandwidth']} (was {bandwidth_stats['bandwidth']})")
            print(f"     Diagonals: {bandwidth_stats_rcm['num_diagonals']} (was {bandwidth_stats['num_diagonals']})")

            bandwidth_reduction = (
                (bandwidth_stats["bandwidth"] - bandwidth_stats_rcm["bandwidth"]) / bandwidth_stats["bandwidth"] * 100
            )
            diagonal_reduction = (
                (bandwidth_stats["num_diagonals"] - bandwidth_stats_rcm["num_diagonals"])
                / bandwidth_stats["num_diagonals"]
                * 100
            )

            print(f"     Bandwidth reduction: {bandwidth_reduction:.1f}%")
            print(f"     Diagonal reduction: {diagonal_reduction:.1f}%")

        except Exception as e:
            print(f"   RCM reordering failed: {e}")
        print()

        # Summary and diagnosis
        print("6. Diagnosis:")
        expected_diagonals = int(1.5 * led_count)
        actual_diagonals = bandwidth_stats.get("num_diagonals", 0)

        if actual_diagonals > expected_diagonals * 2:
            print(f"   ❌ Too many diagonals: {actual_diagonals} >> {expected_diagonals} (expected)")
            print("   Issue likely in diffusion pattern generation:")

            # Check if patterns are too spread out
            max_distance = led_distances.max()
            if max_distance > led_count * 0.5:
                print(f"     - Patterns create connections across {max_distance} LEDs (too wide)")

            # Check if most connections are local
            local_connections = np.sum(distance_counts[:10]) if len(distance_counts) > 10 else np.sum(distance_counts)
            total_connections = np.sum(distance_counts)
            local_ratio = local_connections / total_connections * 100

            print(f"     - Local connections (≤9 LEDs): {local_ratio:.1f}%")
            if local_ratio < 80:
                print("     - Too many long-range connections! Should be >80% local")
        else:
            print(f"   ✅ Diagonal count reasonable: {actual_diagonals} ≈ {expected_diagonals} (expected)")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()


def compare_adjacency_vs_ata():
    """Compare adjacency matrix (expected) vs actual A^T A structure."""
    print(f"\n{'=' * 80}")
    print("COMPARING ADJACENCY MATRIX vs A^T A MATRIX")
    print(f"{'=' * 80}")

    # Import adjacency generation
    from src.utils.spatial_ordering import compute_rcm_ordering
    from tools.led_position_utils import (
        calculate_block_positions,
        generate_adjacency_matrix,
        generate_random_led_positions,
    )

    # Generate the same LED positions as the pattern generator
    led_positions = generate_random_led_positions(1000, seed=42)
    block_positions = calculate_block_positions(led_positions, 64)

    # Generate adjacency matrix
    adjacency = generate_adjacency_matrix(block_positions, 64)

    # Apply RCM to adjacency
    rcm_order, _, expected_adjacency_diagonals = compute_rcm_ordering(block_positions, 64)
    adjacency_rcm = adjacency[rcm_order][:, rcm_order]

    # Analyze adjacency bandwidth
    adjacency_stats = analyze_matrix_bandwidth(adjacency_rcm)

    print("Expected (Adjacency Matrix):")
    print(f"  Sparsity: {adjacency_stats['sparsity']:.2f}%")
    print(f"  Bandwidth: {adjacency_stats['bandwidth']}")
    print(f"  Diagonals: {adjacency_stats['num_diagonals']}")

    # Analyze actual A^T A
    try:
        pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_1000_64x64_fixed2.npz"
        data = np.load(str(pattern_path), allow_pickle=True)
        diffusion_dict = data["diffusion_matrix"].item()

        A = sp.csc_matrix(
            (
                diffusion_dict["csc_data"],
                diffusion_dict["csc_indices"],
                diffusion_dict["csc_indptr"],
            ),
            shape=diffusion_dict["csc_shape"],
        )

        ata_matrix = A.T @ A

        # Convert LED-level A^T A to binary adjacency for comparison
        led_count = 1000
        ata_led_adjacency = np.zeros((led_count, led_count), dtype=int)

        for i in range(led_count):
            for j in range(led_count):
                # Check if any of the 3x3 blocks between LED i and LED j are non-zero
                block = ata_matrix[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]
                if block.nnz > 0:
                    ata_led_adjacency[i, j] = 1

        ata_stats = analyze_matrix_bandwidth(ata_led_adjacency)

        print("Actual (A^T A derived):")
        print(f"  Sparsity: {ata_stats['sparsity']:.2f}%")
        print(f"  Bandwidth: {ata_stats['bandwidth']}")
        print(f"  Diagonals: {ata_stats['num_diagonals']}")

        print("\nComparison:")
        print(f"  Sparsity ratio: {ata_stats['sparsity'] / adjacency_stats['sparsity']:.1f}x")
        print(f"  Bandwidth ratio: {ata_stats['bandwidth'] / adjacency_stats['bandwidth']:.1f}x")
        print(f"  Diagonal ratio: {ata_stats['num_diagonals'] / adjacency_stats['num_diagonals']:.1f}x")

        if ata_stats["num_diagonals"] > adjacency_stats["num_diagonals"] * 2:
            ratio = ata_stats["num_diagonals"] / adjacency_stats["num_diagonals"]
            print(f"  ❌ A^T A has {ratio:.1f}x more diagonals than expected!")
            print("     This suggests diffusion patterns are creating unexpected long-range connections")
        else:
            print("  ✅ A^T A structure matches adjacency expectations")

    except Exception as e:
        print(f"Failed to analyze actual A^T A: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pattern_file = sys.argv[1]
    else:
        pattern_file = "capture-0728-01-final-comparison.npz"

    # Analyze the patterns
    analyze_ata_structure(pattern_file)

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
