#!/usr/bin/env python3
"""
Debug diagonal counting discrepancies between adjacency and A^T A matrices.

This script carefully checks:
1. Exact LED position matching between adjacency and A^T A calculations
2. Block position rounding to multiple of 4
3. Diagonal counting methodology
4. Per-plane diagonal counts for A^T A
5. Overlap detection sensitivity (1-pixel overlaps)
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from tools.led_position_utils import (
    analyze_matrix_bandwidth,
    calculate_block_positions,
    generate_adjacency_matrix,
    generate_random_led_positions,
)

# Set up logging
logging.basicConfig(level=logging.INFO)


def debug_led_positions_and_rounding():
    """Debug LED position generation and block position rounding."""
    print("=== Debugging LED Position Generation and Rounding ===")
    print()

    # Generate LED positions using same parameters as pattern generation
    led_positions = generate_random_led_positions(
        1000, frame_width=800, frame_height=640, seed=42
    )

    print("1. Raw LED positions (first 10):")
    for i in range(10):
        print(f"   LED {i}: ({led_positions[i][0]}, {led_positions[i][1]})")
    print()

    # Calculate block positions with rounding
    block_positions = calculate_block_positions(
        led_positions, block_size=64, frame_width=800, frame_height=640
    )

    print("2. Block positions after rounding (first 10):")
    for i in range(10):
        led_x, led_y = led_positions[i]
        block_x, block_y = block_positions[i]

        # Manual calculation to verify
        block_x_candidate = max(0, min(800 - 64, led_x - 64 // 2))
        block_y_expected = max(0, min(640 - 64, led_y - 64 // 2))
        block_x_expected = (block_x_candidate // 4) * 4  # Round to multiple of 4

        x_match = block_x == block_x_expected
        y_match = block_y == block_y_expected

        print(
            f"   LED {i}: LED({led_x:3}, {led_y:3}) -> Block({block_x:3}, {block_y:3}) "
            f"[Expected: ({block_x_expected:3}, {block_y_expected:3})] "
            f"{'✅' if x_match and y_match else '❌'}"
        )

    # Check x-coordinate alignment
    x_coords = block_positions[:, 0]
    unaligned_count = np.sum(x_coords % 4 != 0)
    print("\n3. X-coordinate alignment check:")
    print(f"   Total LEDs: {len(block_positions)}")
    print(f"   Unaligned x-coordinates: {unaligned_count}")
    print(f"   All aligned to multiple of 4: {'✅' if unaligned_count == 0 else '❌'}")

    return led_positions, block_positions


def debug_adjacency_calculation(block_positions, block_size=64):
    """Debug adjacency matrix calculation in detail."""
    print("\n=== Debugging Adjacency Matrix Calculation ===")
    print(f"Block size: {block_size}x{block_size}")
    print()

    n_leds = len(block_positions)
    adjacency = np.zeros((n_leds, n_leds), dtype=np.int32)

    overlap_details = []

    # Check all pairs for block overlap (same logic as generate_adjacency_matrix)
    for i in range(n_leds):
        for j in range(i, n_leds):  # Only check upper triangle + diagonal
            if i == j:
                adjacency[i, j] = 1  # LED always overlaps with itself
                continue

            pos1_x, pos1_y = block_positions[i]
            pos2_x, pos2_y = block_positions[j]

            # Block extents: [top_left, top_left + block_size]
            x1_min, x1_max = pos1_x, pos1_x + block_size
            y1_min, y1_max = pos1_y, pos1_y + block_size
            x2_min, x2_max = pos2_x, pos2_x + block_size
            y2_min, y2_max = pos2_y, pos2_y + block_size

            # Compute overlap dimensions
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

            # Blocks are adjacent if they have any overlap
            if x_overlap > 0 and y_overlap > 0:
                adjacency[i, j] = 1
                adjacency[j, i] = 1  # Symmetric

                # Store overlap details for small overlaps
                overlap_area = x_overlap * y_overlap
                overlap_details.append(
                    {
                        "led_i": i,
                        "led_j": j,
                        "x_overlap": x_overlap,
                        "y_overlap": y_overlap,
                        "area": overlap_area,
                    }
                )

    # Analyze overlap details
    print("1. Overlap analysis:")
    print(f"   Total LED pairs: {n_leds * (n_leds - 1) // 2}")
    print(f"   Overlapping pairs: {len(overlap_details)}")

    if overlap_details:
        areas = [detail["area"] for detail in overlap_details]
        x_overlaps = [detail["x_overlap"] for detail in overlap_details]
        y_overlaps = [detail["y_overlap"] for detail in overlap_details]

        print("   Overlap area stats:")
        print(f"     Min: {np.min(areas):.1f} pixels")
        print(f"     Max: {np.max(areas):.1f} pixels")
        print(f"     Mean: {np.mean(areas):.1f} pixels")

        print(f"   X-overlap stats: Min={np.min(x_overlaps)}, Max={np.max(x_overlaps)}")
        print(f"   Y-overlap stats: Min={np.min(y_overlaps)}, Max={np.max(y_overlaps)}")

        # Show smallest overlaps
        smallest_overlaps = sorted(overlap_details, key=lambda x: x["area"])[:5]
        print("   Smallest 5 overlaps:")
        for detail in smallest_overlaps:
            print(
                f"     LEDs {detail['led_i']}-{detail['led_j']}: "
                f"{detail['x_overlap']}x{detail['y_overlap']} = {detail['area']} pixels"
            )

    # Analyze adjacency matrix structure
    bandwidth_stats = analyze_matrix_bandwidth(adjacency)

    print("\n2. Adjacency matrix structure:")
    print(f"   Size: {adjacency.shape}")
    print(f"   Non-zeros: {bandwidth_stats['nnz']:,}")
    print(f"   Sparsity: {bandwidth_stats['sparsity']:.2f}%")
    print(f"   Bandwidth: {bandwidth_stats['bandwidth']}")
    print(f"   Number of diagonals: {bandwidth_stats['num_diagonals']}")
    min_offset = min(bandwidth_stats["diagonal_offsets"])
    max_offset = max(bandwidth_stats["diagonal_offsets"])
    print(f"   Diagonal offsets range: [{min_offset}, {max_offset}]")

    return adjacency, bandwidth_stats


def debug_ata_per_plane(mixed_tensor):
    """Debug A^T A calculation per RGB plane."""
    print("\n=== Debugging A^T A Per RGB Plane ===")
    print()

    # Compute A^T A
    ata_tensor = mixed_tensor.compute_ata_dense()  # Shape: (leds, leds, channels)

    print(f"A^T A tensor shape: {ata_tensor.shape}")

    # Analyze each plane separately
    plane_stats = []
    for channel in range(3):
        plane_name = ["Red", "Green", "Blue"][channel]
        ata_plane = ata_tensor[:, :, channel]

        # Convert to binary adjacency
        ata_binary = (ata_plane != 0).astype(int)

        # Analyze bandwidth
        bandwidth_stats = analyze_matrix_bandwidth(ata_binary)
        plane_stats.append(bandwidth_stats)

        print(f"{channel + 1}. {plane_name} plane:")
        print(f"   Non-zeros: {bandwidth_stats['nnz']:,}")
        print(f"   Sparsity: {bandwidth_stats['sparsity']:.2f}%")
        print(f"   Bandwidth: {bandwidth_stats['bandwidth']}")
        print(f"   Number of diagonals: {bandwidth_stats['num_diagonals']}")
        min_offset = min(bandwidth_stats["diagonal_offsets"])
        max_offset = max(bandwidth_stats["diagonal_offsets"])
        print(f"   Diagonal range: [{min_offset}, {max_offset}]")

        # Check for non-zero values where adjacency is zero (this shouldn't happen!)
        print(f"   Value range: [{np.min(ata_plane):.6f}, {np.max(ata_plane):.6f}]")

    return ata_tensor, plane_stats


def compare_adjacency_vs_ata_exact(adjacency, ata_tensor):
    """Compare adjacency vs A^T A with exact position matching."""
    print("\n=== Exact Comparison: Adjacency vs A^T A ===")
    print()

    # Compare each plane
    for channel in range(3):
        plane_name = ["Red", "Green", "Blue"][channel]
        ata_plane = ata_tensor[:, :, channel]
        ata_binary = (ata_plane != 0).astype(int)

        print(f"{channel + 1}. {plane_name} plane comparison:")

        # Check if A^T A binary pattern is subset of adjacency
        ata_not_in_adj = ata_binary & (
            ~adjacency
        )  # Elements in A^T A but not in adjacency
        adj_not_in_ata = adjacency & (
            ~ata_binary
        )  # Elements in adjacency but not in A^T A

        violations = np.sum(ata_not_in_adj)
        missing = np.sum(adj_not_in_ata)

        print(f"   A^T A non-zeros NOT in adjacency: {violations}")
        print(f"   Adjacency non-zeros NOT in A^T A: {missing}")

        if violations > 0:
            print("   ❌ VIOLATION: A^T A has connections where blocks don't overlap!")
            # Show first few violations
            violation_positions = np.where(ata_not_in_adj)
            for i in range(min(5, len(violation_positions[0]))):
                row, col = violation_positions[0][i], violation_positions[1][i]
                value = ata_plane[row, col]
                print(
                    f"     Position ({row}, {col}): A^T A = {value:.6f}, Adjacency = 0"
                )

        if missing > 0:
            print(
                "   ⚠️  Missing: Some overlapping blocks have zero A^T A (possible with pattern generation)"
            )

        if violations == 0 and missing == 0:
            print("   ✅ PERFECT: A^T A pattern exactly matches adjacency")
        elif violations == 0:
            print("   ✅ VALID: A^T A is proper subset of adjacency")

    return violations == 0


def test_exact_led_positions():
    """Test that we're using exactly the same LED positions for both calculations."""
    print("\n=== Testing Exact LED Position Matching ===")

    # Get LED positions used in adjacency calculation
    led_positions_adj, block_positions_adj = debug_led_positions_and_rounding()

    # Load mixed tensor and check if it uses the same positions
    try:
        pattern_path = (
            Path(__file__).parent.parent
            / "diffusion_patterns"
            / "synthetic_1000_64x64_fixed2.npz"
        )
        data = np.load(str(pattern_path), allow_pickle=True)
        mixed_tensor_dict = data["mixed_tensor"].item()

        mixed_tensor = SingleBlockMixedSparseTensor.from_dict(
            mixed_tensor_dict, device="cpu"
        )

        # Extract block positions from mixed tensor
        if hasattr(mixed_tensor.block_positions, "cpu"):
            block_positions_tensor = (
                mixed_tensor.block_positions.cpu().numpy()
            )  # Shape: (channels, leds, 2)
        else:
            block_positions_tensor = mixed_tensor.block_positions  # Already numpy array

        # Check if positions match across channels and with adjacency calculation
        print(f"Mixed tensor block positions shape: {block_positions_tensor.shape}")

        positions_match = True
        for channel in range(3):
            # Ensure we have numpy arrays for comparison
            try:
                if hasattr(block_positions_tensor[channel], "get"):
                    channel_positions = block_positions_tensor[
                        channel
                    ].get()  # CuPy array
                elif hasattr(block_positions_tensor[channel], "numpy"):
                    channel_positions = block_positions_tensor[
                        channel
                    ].numpy()  # PyTorch tensor
                else:
                    channel_positions = np.asarray(
                        block_positions_tensor[channel]
                    )  # Numpy array
            except Exception as e:
                print(f"   ❌ Failed to convert positions to numpy: {e}")
                positions_match = False
                continue

            block_positions_adj_np = np.asarray(block_positions_adj)

            # Compare with adjacency positions
            position_diff = np.max(np.abs(channel_positions - block_positions_adj_np))

            if position_diff > 0:
                print(f"   Channel {channel}: Position difference = {position_diff}")
                positions_match = False
            else:
                print(f"   Channel {channel}: Positions match exactly ✅")

        if positions_match:
            print("   ✅ All channels use identical block positions")
            print("   ✅ Mixed tensor positions match adjacency calculation")
        else:
            print("   ❌ Position mismatch detected!")

        return mixed_tensor, positions_match

    except Exception as e:
        print(f"   ❌ Failed to load mixed tensor: {e}")
        return None, False


def main():
    """Run complete diagonal counting debug analysis."""
    print("🔍 DEBUGGING DIAGONAL COUNT DISCREPANCIES")
    print("=" * 80)

    # Step 1: Debug LED position generation and rounding
    led_positions, block_positions = debug_led_positions_and_rounding()

    # Step 2: Debug adjacency matrix calculation
    adjacency, adj_stats = debug_adjacency_calculation(block_positions, block_size=64)

    # Step 3: Test exact LED position matching
    mixed_tensor, positions_match = test_exact_led_positions()

    if mixed_tensor is None:
        print("❌ Cannot proceed without mixed tensor")
        return

    # Step 4: Debug A^T A per plane
    ata_tensor, ata_stats = debug_ata_per_plane(mixed_tensor)

    # Step 5: Compare adjacency vs A^T A exactly
    is_valid = compare_adjacency_vs_ata_exact(adjacency, ata_tensor)

    # Step 6: Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Position matching: {'✅' if positions_match else '❌'}")
    print(f"A^T A validation: {'✅' if is_valid else '❌'}")
    print()

    print("Diagonal counts:")
    print(f"  Adjacency matrix:     {adj_stats['num_diagonals']:4d} diagonals")
    for channel in range(3):
        plane_name = ["Red", "Green", "Blue"][channel]
        print(
            f"  A^T A {plane_name:5s} plane:    {ata_stats[channel]['num_diagonals']:4d} diagonals"
        )

    # Check if diagonal counts are reasonable
    max_ata_diagonals = max(stats["num_diagonals"] for stats in ata_stats)
    diagonal_ratio = max_ata_diagonals / adj_stats["num_diagonals"]

    print("\nDiagonal count analysis:")
    print(f"  Expected (adjacency): {adj_stats['num_diagonals']}")
    print(f"  Actual (max A^T A):   {max_ata_diagonals}")
    print(f"  Ratio:                {diagonal_ratio:.2f}x")

    if diagonal_ratio <= 1.0:
        print("  ✅ PERFECT: A^T A has ≤ adjacency diagonals (as expected)")
    elif diagonal_ratio <= 1.2:
        print("  ✅ GOOD: A^T A has reasonable diagonal count")
    else:
        print("  ❌ ISSUE: A^T A has significantly more diagonals than expected")

    return is_valid and positions_match and diagonal_ratio <= 1.2


if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎉 DEBUGGING COMPLETE - ALL CHECKS PASSED!")
    else:
        print("\n❌ DEBUGGING REVEALED ISSUES - FURTHER INVESTIGATION NEEDED!")
