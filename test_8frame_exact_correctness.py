#!/usr/bin/env python3
"""
Exact correctness tests for 8-frame WMMA kernels using small integers.

Uses small integer values that can be exactly represented in FP16/TF32
to eliminate precision uncertainty and focus on mathematical correctness.
"""

import os
import sys

import cupy
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix


def create_exact_identity_matrix(led_count):
    """Create an exact identity matrix using small integers."""
    print(f"Creating {led_count}x{led_count} exact identity matrix")

    channels = 3
    offsets = np.array([0], dtype=np.int32)  # Only main diagonal
    dia_data = np.ones((channels, 1, led_count), dtype=np.float32)  # Exactly 1.0

    # Create DiagonalATAMatrix
    matrix = DiagonalATAMatrix(led_count, crop_size=1)
    matrix.bandwidth = 1
    matrix.dia_data_upper_gpu = cupy.asarray(dia_data)
    matrix.dia_offsets_upper_gpu = cupy.asarray(offsets)
    matrix.k = 1
    matrix.sparsity = 1.0 - (1.0 / led_count)
    matrix.nnz = led_count
    matrix.dia_offsets = offsets
    matrix.dia_data_cpu = dia_data

    return matrix


def create_exact_simple_matrix(led_count, pattern_type="tridiagonal"):
    """Create a matrix with exact small integer values."""
    print(f"Creating {led_count}x{led_count} exact {pattern_type} matrix")

    channels = 3

    if pattern_type == "tridiagonal":
        # Simple tridiagonal: main=2, off=1 (exactly representable)
        offsets = np.array([0, 1], dtype=np.int32)
        dia_data = np.zeros((channels, 2, led_count), dtype=np.float32)

        for ch in range(channels):
            dia_data[ch, 0, :] = 2.0  # Main diagonal = 2
            dia_data[ch, 1, : led_count - 1] = 1.0  # Off diagonal = 1

    elif pattern_type == "simple_banded":
        # Simple banded: main=3, first_off=2, second_off=1
        offsets = np.array([0, 1, 2], dtype=np.int32)
        dia_data = np.zeros((channels, 3, led_count), dtype=np.float32)

        for ch in range(channels):
            dia_data[ch, 0, :] = 3.0  # Main diagonal = 3
            dia_data[ch, 1, : led_count - 1] = 2.0  # First off = 2
            dia_data[ch, 2, : led_count - 2] = 1.0  # Second off = 1

    # Create DiagonalATAMatrix
    matrix = DiagonalATAMatrix(led_count, crop_size=1)
    matrix.bandwidth = len(offsets)
    matrix.dia_data_upper_gpu = cupy.asarray(dia_data)
    matrix.dia_offsets_upper_gpu = cupy.asarray(offsets)
    matrix.k = len(offsets)
    matrix.sparsity = 1.0 - (len(offsets) / led_count)
    matrix.nnz = sum(led_count - offset for offset in offsets)
    matrix.dia_offsets = offsets
    matrix.dia_data_cpu = dia_data

    return matrix


def create_exact_test_vectors(batch_size, channels, led_count, pattern="small_integers"):
    """Create test vectors with exact small integer values."""
    if pattern == "small_integers":
        # Use small integers 1, 2, 3, etc. (exactly representable)
        test_vectors = np.zeros((batch_size, channels, led_count), dtype=np.float32)
        for b in range(batch_size):
            for c in range(channels):
                for i in range(led_count):
                    # Create a simple pattern: 1, 2, 3, 1, 2, 3, ... (mod 3 + 1)
                    test_vectors[b, c, i] = float((i % 3) + 1)
    elif pattern == "ones":
        # All ones - perfect for identity matrix testing
        test_vectors = np.ones((batch_size, channels, led_count), dtype=np.float32)
    elif pattern == "sequential":
        # Sequential integers: 1, 2, 3, 4, ...
        test_vectors = np.zeros((batch_size, channels, led_count), dtype=np.float32)
        for b in range(batch_size):
            for c in range(channels):
                for i in range(led_count):
                    test_vectors[b, c, i] = float(i + 1)

    return test_vectors


def compute_exact_reference(matrix_type, matrix_values, input_vectors):
    """Compute exact reference result for comparison."""
    batch_size, channels, led_count = input_vectors.shape

    if matrix_type == "identity":
        # For identity matrix: result = input (exactly)
        return input_vectors.copy()

    elif matrix_type == "tridiagonal":
        # For tridiagonal [2,1; 1,2,1; 1,2,1; ...; 1,2]
        result = np.zeros_like(input_vectors)
        for b in range(batch_size):
            for c in range(channels):
                input_vec = input_vectors[b, c, :]

                # First element: 2*input[0] + 1*input[1]
                result[b, c, 0] = 2.0 * input_vec[0] + 1.0 * input_vec[1]

                # Interior elements: 1*input[i-1] + 2*input[i] + 1*input[i+1]
                for i in range(1, led_count - 1):
                    result[b, c, i] = 1.0 * input_vec[i - 1] + 2.0 * input_vec[i] + 1.0 * input_vec[i + 1]

                # Last element: 1*input[n-2] + 2*input[n-1]
                result[b, c, led_count - 1] = 1.0 * input_vec[led_count - 2] + 2.0 * input_vec[led_count - 1]

        return result

    elif matrix_type == "simple_banded":
        # For simple banded [3,2,1; 2,3,2,1; 1,2,3,2,1; ...]
        result = np.zeros_like(input_vectors)
        for b in range(batch_size):
            for c in range(channels):
                input_vec = input_vectors[b, c, :]

                for i in range(led_count):
                    value = 3.0 * input_vec[i]  # Main diagonal

                    # First off-diagonal contributions
                    if i > 0:
                        value += 2.0 * input_vec[i - 1]
                    if i < led_count - 1:
                        value += 2.0 * input_vec[i + 1]

                    # Second off-diagonal contributions
                    if i > 1:
                        value += 1.0 * input_vec[i - 2]
                    if i < led_count - 2:
                        value += 1.0 * input_vec[i + 2]

                    result[b, c, i] = value

        return result

    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")


def test_exact_correctness(diagonal_matrix, matrix_type, test_name, input_vectors, expected_result):
    """Test exact correctness with zero tolerance for small integer operations."""
    print(f"\n--- {test_name} ---")

    # Create 8-frame batch matrix
    batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(diagonal_matrix, batch_size=8)

    # Convert to GPU
    input_gpu = cupy.ascontiguousarray(cupy.asarray(input_vectors, dtype=cupy.float32))

    print(f"Testing with exact {input_vectors.shape} input")
    print(f"Input sample [0,0,:5]: {input_vectors[0, 0, :5]}")
    if matrix_type != "identity":
        print(f"Expected sample [0,0,:5]: {expected_result[0, 0, :5]}")

    # Compute using 8-frame WMMA kernel
    wmma_result = batch_matrix.multiply_batch_3d(input_gpu)
    wmma_result_cpu = cupy.asnumpy(wmma_result)

    print(f"WMMA result sample [0,0,:5]: {wmma_result_cpu[0, 0, :5]}")

    # Compare with exact expected result
    abs_diff = np.abs(expected_result - wmma_result_cpu)
    max_abs_diff = np.max(abs_diff)

    # For exact integer operations, we expect perfect or near-perfect accuracy
    tolerance = 1e-6  # Very strict tolerance for exact operations

    print("Error Analysis:")
    print(f"  Max absolute error: {max_abs_diff:.2e}")
    print(f"  Expected: Exact match (< {tolerance:.0e})")

    if max_abs_diff < tolerance:
        print("  ✓ PASS: Exact correctness verified")
        success = True
    else:
        print("  ✗ FAIL: Error exceeds exact tolerance")

        # Show worst error details
        worst_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        batch_idx, ch, led_idx = worst_idx
        print(f"  Worst error at [{batch_idx}][{ch}][{led_idx}]:")
        print(f"    Expected: {expected_result[batch_idx, ch, led_idx]:.6f}")
        print(f"    WMMA:     {wmma_result_cpu[batch_idx, ch, led_idx]:.6f}")
        print(f"    Diff:     {abs_diff[batch_idx, ch, led_idx]:.6f}")
        success = False

    return success


def test_32x32_identity():
    """Test 32x32 identity matrix with 8-frame kernel."""
    print("=" * 60)
    print("TEST 1: 32x32 Identity Matrix (8-frame)")
    print("=" * 60)

    # Create exact identity matrix
    diagonal_matrix = create_exact_identity_matrix(32)

    # Test with ones (should get ones back exactly)
    print("\nSubtest 1.1: All-ones input")
    input_vectors = create_exact_test_vectors(8, 3, 32, pattern="ones")
    expected_result = input_vectors.copy()  # Identity: output = input

    success1 = test_exact_correctness(
        diagonal_matrix, "identity", "32x32 Identity with Ones", input_vectors, expected_result
    )

    # Test with small integers
    print("\nSubtest 1.2: Small integer pattern input")
    input_vectors = create_exact_test_vectors(8, 3, 32, pattern="small_integers")
    expected_result = input_vectors.copy()  # Identity: output = input

    success2 = test_exact_correctness(
        diagonal_matrix, "identity", "32x32 Identity with Small Integers", input_vectors, expected_result
    )

    return success1 and success2


def test_64x64_identity():
    """Test 64x64 identity matrix with 8-frame kernel."""
    print("\n" + "=" * 60)
    print("TEST 2: 64x64 Identity Matrix (8-frame)")
    print("=" * 60)

    # Create exact identity matrix
    diagonal_matrix = create_exact_identity_matrix(64)

    # Test with ones
    print("\nSubtest 2.1: All-ones input")
    input_vectors = create_exact_test_vectors(8, 3, 64, pattern="ones")
    expected_result = input_vectors.copy()  # Identity: output = input

    success1 = test_exact_correctness(
        diagonal_matrix, "identity", "64x64 Identity with Ones", input_vectors, expected_result
    )

    # Test with sequential integers
    print("\nSubtest 2.2: Sequential integer input")
    input_vectors = create_exact_test_vectors(8, 3, 64, pattern="sequential")
    expected_result = input_vectors.copy()  # Identity: output = input

    success2 = test_exact_correctness(
        diagonal_matrix, "identity", "64x64 Identity with Sequential", input_vectors, expected_result
    )

    return success1 and success2


def test_32x32_tridiagonal():
    """Test 32x32 tridiagonal matrix with 8-frame kernel."""
    print("\n" + "=" * 60)
    print("TEST 3: 32x32 Tridiagonal Matrix (8-frame)")
    print("=" * 60)

    # Create exact tridiagonal matrix
    diagonal_matrix = create_exact_simple_matrix(32, pattern_type="tridiagonal")

    # Test with ones (easy to verify manually)
    print("\nSubtest 3.1: All-ones input")
    input_vectors = create_exact_test_vectors(8, 3, 32, pattern="ones")
    expected_result = compute_exact_reference("tridiagonal", None, input_vectors)

    print("Manual verification for all-ones tridiagonal:")
    print("  Expected[0] = 2*1 + 1*1 = 3.0")
    print("  Expected[1] = 1*1 + 2*1 + 1*1 = 4.0")
    print("  Expected[31] = 1*1 + 2*1 = 3.0")

    success1 = test_exact_correctness(
        diagonal_matrix, "tridiagonal", "32x32 Tridiagonal with Ones", input_vectors, expected_result
    )

    # Test with small integer pattern
    print("\nSubtest 3.2: Small integer pattern input")
    input_vectors = create_exact_test_vectors(8, 3, 32, pattern="small_integers")
    expected_result = compute_exact_reference("tridiagonal", None, input_vectors)

    success2 = test_exact_correctness(
        diagonal_matrix, "tridiagonal", "32x32 Tridiagonal with Pattern", input_vectors, expected_result
    )

    return success1 and success2


def test_32x32_simple_banded():
    """Test 32x32 simple banded matrix with 8-frame kernel."""
    print("\n" + "=" * 60)
    print("TEST 4: 32x32 Simple Banded Matrix (8-frame)")
    print("=" * 60)

    # Create exact simple banded matrix
    diagonal_matrix = create_exact_simple_matrix(32, pattern_type="simple_banded")

    # Test with ones
    print("\nSubtest 4.1: All-ones input")
    input_vectors = create_exact_test_vectors(8, 3, 32, pattern="ones")
    expected_result = compute_exact_reference("simple_banded", None, input_vectors)

    print("Manual verification for all-ones simple banded:")
    print("  Expected[0] = 3*1 + 2*1 + 1*1 = 6.0 (main + 1st_off + 2nd_off)")
    print("  Expected[1] = 2*1 + 3*1 + 2*1 + 1*1 = 8.0")
    print("  Expected[2] = 1*1 + 2*1 + 3*1 + 2*1 + 1*1 = 9.0 (full pattern)")

    success1 = test_exact_correctness(
        diagonal_matrix, "simple_banded", "32x32 Simple Banded with Ones", input_vectors, expected_result
    )

    return success1


def main():
    """Main test function."""
    print("=" * 80)
    print("8-Frame WMMA Kernel Exact Correctness Tests")
    print("Using Small Integer Values for Exact Precision")
    print("=" * 80)

    all_tests_passed = True

    # Run progressive tests
    if not test_32x32_identity():
        all_tests_passed = False

    if not test_64x64_identity():
        all_tests_passed = False

    if not test_32x32_tridiagonal():
        all_tests_passed = False

    if not test_32x32_simple_banded():
        all_tests_passed = False

    # Final summary
    print("\n" + "=" * 80)
    print("EXACT CORRECTNESS SUMMARY")
    print("=" * 80)

    if all_tests_passed:
        print("✓ ALL EXACT TESTS PASSED")
        print("✓ 8-frame WMMA kernels are mathematically exact for integer operations")
        print("✓ Identity matrices: Perfect exactness verified")
        print("✓ Tridiagonal matrices: Exact computation verified")
        print("✓ Banded matrices: Exact computation verified")
        print("✓ No precision issues - all operations exact within FP16/TF32 precision")
    else:
        print("✗ SOME EXACT TESTS FAILED")
        print("✗ 8-frame WMMA kernels have mathematical errors")
        print("✗ Check detailed error analysis above")

    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
