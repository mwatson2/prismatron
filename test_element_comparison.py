#!/usr/bin/env python3
"""
Element-by-element comparison test between DiagonalATAMatrix and BatchSymmetricDiagonalATAMatrix.

This test verifies that the conversion from diagonal storage to 5D block storage
preserves all matrix elements correctly.
"""

import os
import sys

import cupy
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix


def create_test_diagonal_matrix(led_count=64, bandwidth=32):
    """
    Create a test diagonal matrix with known pattern.

    Args:
        led_count: Number of LEDs
        bandwidth: Maximum diagonal offset

    Returns:
        DiagonalATAMatrix instance
    """
    print(f"Creating test diagonal matrix: {led_count} LEDs, bandwidth {bandwidth}")

    # Create diagonal pattern with decreasing values
    # Main diagonal = 10, first upper diagonal = 9, second = 8, etc.
    max_diagonals = min(bandwidth, led_count)

    # Create diagonal offsets (including main diagonal at 0)
    offsets = np.arange(max_diagonals, dtype=np.int32)

    # Create diagonal data for 3 channels
    channels = 3
    dia_data = np.zeros((channels, max_diagonals, led_count), dtype=np.float32)

    # Fill with test pattern
    for ch in range(channels):
        for diag_idx, offset in enumerate(offsets):
            # Use different values per channel for better testing
            base_value = 10 - diag_idx
            channel_multiplier = 1.0 + ch * 0.1  # 1.0, 1.1, 1.2 for R,G,B
            value = base_value * channel_multiplier

            # Fill the diagonal (accounting for shorter diagonals)
            diag_length = led_count - offset
            dia_data[ch, diag_idx, :diag_length] = value

    # Create DiagonalATAMatrix
    regular_matrix = DiagonalATAMatrix(led_count, crop_size=1)
    regular_matrix.bandwidth = bandwidth
    regular_matrix.dia_data_upper_gpu = cupy.asarray(dia_data)
    regular_matrix.dia_offsets_upper_gpu = cupy.asarray(offsets)

    # Set attributes expected by from_diagonal_ata_matrix
    regular_matrix.k = max_diagonals  # Total number of diagonals
    regular_matrix.sparsity = 0.95  # Dummy value
    regular_matrix.nnz = led_count * max_diagonals  # Approximate

    # Create full diagonal data/offsets for compatibility
    regular_matrix.dia_offsets = offsets
    regular_matrix.dia_data_cpu = cupy.asnumpy(dia_data)

    print(f"  Diagonal matrix created with {max_diagonals} diagonals")
    print("  Channel multipliers: R=1.0, G=1.1, B=1.2")
    print(
        f"  Diagonal values: main={10*1.0:.1f}/{10*1.1:.1f}/{10*1.2:.1f}, first_upper={9*1.0:.1f}/{9*1.1:.1f}/{9*1.2:.1f}, etc."
    )

    return regular_matrix


def compare_element_by_element(diagonal_matrix, batch_matrix, test_positions=None):
    """
    Compare matrices element by element.

    Args:
        diagonal_matrix: DiagonalATAMatrix instance
        batch_matrix: BatchSymmetricDiagonalATAMatrix instance
        test_positions: List of (row, col) positions to test, or None for automatic selection

    Returns:
        bool: True if all elements match
    """
    print("\n=== Element-by-Element Comparison ===")

    led_count = diagonal_matrix.led_count
    channels = 3
    errors = 0
    tests_performed = 0

    # First reconstruct the full symmetric matrix from diagonal storage for reference
    print("Reconstructing reference symmetric matrices from diagonal storage...")
    reference_matrices = []

    for channel in range(channels):
        # Create dense matrix for this channel
        dense_matrix = np.zeros((led_count, led_count), dtype=np.float32)

        # Fill upper triangle from diagonal storage
        dia_data_cpu = cupy.asnumpy(diagonal_matrix.dia_data_upper_gpu[channel])
        dia_offsets = cupy.asnumpy(diagonal_matrix.dia_offsets_upper_gpu)

        for diag_idx, offset in enumerate(dia_offsets):
            diag_data = dia_data_cpu[diag_idx, :led_count]

            # Fill upper diagonal
            for i in range(led_count - offset):
                row, col = i, i + offset
                dense_matrix[row, col] = diag_data[i]

                # Fill symmetric lower diagonal (if not main diagonal)
                if offset > 0:
                    dense_matrix[col, row] = diag_data[i]

        reference_matrices.append(dense_matrix)

    # Determine test positions
    if test_positions is None:
        # Create comprehensive test positions
        test_positions = []

        # Test main and near diagonals comprehensively
        for offset in range(min(10, led_count)):
            for i in range(min(20, led_count - offset)):  # Test first 20 positions of each diagonal
                test_positions.append((i, i + offset))
                if offset > 0:  # Also test symmetric positions
                    test_positions.append((i + offset, i))

        # Add some random positions
        np.random.seed(42)  # For reproducible tests
        for _ in range(50):
            row = np.random.randint(0, led_count)
            col = np.random.randint(0, led_count)
            test_positions.append((row, col))

        # Remove duplicates
        test_positions = list(set(test_positions))

    print(f"Testing {len(test_positions)} element positions across {channels} channels...")

    # Compare each position
    for row, col in test_positions:
        if row >= led_count or col >= led_count:
            continue

        for channel in range(channels):
            # Get reference value
            reference_value = reference_matrices[channel][row, col]

            # Get batch matrix value using get_element method
            try:
                batch_value = batch_matrix.get_element(channel, row, col)
            except Exception as e:
                print(f"ERROR: Failed to get element[{channel},{row},{col}]: {e}")
                errors += 1
                continue

            # Compare values
            abs_diff = abs(reference_value - batch_value)
            rel_diff = abs_diff / (abs(reference_value) + 1e-10)  # Avoid division by zero

            if abs_diff > 1e-6 and rel_diff > 1e-6:
                print(
                    f"ERROR: Element[{channel},{row},{col}] mismatch: reference={reference_value:.6f}, batch={batch_value:.6f}, diff={abs_diff:.6f}"
                )
                errors += 1

            tests_performed += 1

    print(f"Comparison completed: {tests_performed} elements tested")
    print(f"Errors found: {errors}")
    print(
        f"Accuracy: {((tests_performed - errors) / tests_performed * 100):.2f}%"
        if tests_performed > 0
        else "No tests performed"
    )

    return errors == 0


def test_specific_patterns():
    """
    Test specific known patterns to verify correctness.
    """
    print("\n=== Testing Specific Patterns ===")

    # Test 1: Small matrix with known values
    print("\nTest 1: 32x32 matrix with bandwidth 16")
    regular_matrix = create_test_diagonal_matrix(led_count=32, bandwidth=16)
    batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=16)

    # Test specific positions we know the values for
    test_positions = [
        (0, 0),  # Main diagonal, should be 10.0/11.0/12.0
        (0, 1),  # First upper diagonal, should be 9.0/9.9/10.8
        (1, 0),  # Should be same as (0,1) due to symmetry
        (5, 7),  # Second upper diagonal, should be 8.0/8.8/9.6
        (7, 5),  # Should be same as (5,7) due to symmetry
        (10, 10),  # Main diagonal again
        (15, 20),  # Should be within bandwidth
        (20, 31),  # Should be beyond bandwidth, should be 0
    ]

    success = compare_element_by_element(regular_matrix, batch_matrix, test_positions)
    if not success:
        return False

    # Test 2: Larger matrix
    print("\nTest 2: 64x64 matrix with bandwidth 32")
    regular_matrix = create_test_diagonal_matrix(led_count=64, bandwidth=32)
    batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=16)

    success = compare_element_by_element(regular_matrix, batch_matrix)
    if not success:
        return False

    # Test 3: Edge case - very small matrix
    print("\nTest 3: 16x16 matrix with bandwidth 8")
    regular_matrix = create_test_diagonal_matrix(led_count=16, bandwidth=8)
    batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=16)

    success = compare_element_by_element(regular_matrix, batch_matrix)
    return success


def main():
    """Main test function."""
    print("=" * 80)
    print("Element-by-Element Comparison Test")
    print("DiagonalATAMatrix vs BatchSymmetricDiagonalATAMatrix")
    print("=" * 80)

    try:
        # Run specific pattern tests
        success = test_specific_patterns()

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Element-by-element comparison: {'PASS' if success else 'FAIL'}")

        if success:
            print("\n✓ All matrix elements match between diagonal and batch storage formats")
            print("✓ The 5D block storage conversion preserves matrix values correctly")
            print("✓ The get_element() method works correctly with symmetric storage")
        else:
            print("\n✗ Element mismatches found!")
            print("✗ The conversion or get_element() method has issues")

        return success

    except Exception as e:
        print(f"\nTEST FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
