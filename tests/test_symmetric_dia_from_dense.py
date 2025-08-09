#!/usr/bin/env python3
"""
Unit tests for SymmetricDiagonalATAMatrix.from_dense() method.

Tests the direct creation of symmetric diagonal matrices from dense matrices
with various synthetic test cases and validation of matrix-vector operations.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import cupy

    CUPY_AVAILABLE = True
except ImportError:
    import numpy as cupy

    CUPY_AVAILABLE = False

from src.utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix


class TestSymmetricDiagonalATAFromDense(unittest.TestCase):
    """Test cases for SymmetricDiagonalATAMatrix.from_dense() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.led_counts = [8, 16, 32]
        self.channels = 3

    def create_test_matrix(self, led_count: int, pattern_type: str) -> np.ndarray:
        """
        Create synthetic test matrices with known patterns.

        Args:
            led_count: Size of the matrix
            pattern_type: Type of pattern ('diagonal', 'tridiagonal', 'block', 'random')

        Returns:
            Dense ATA matrix of shape (3, led_count, led_count)
        """
        matrices = np.zeros((3, led_count, led_count), dtype=np.float32)

        for c in range(3):
            if pattern_type == "diagonal":
                # Only main diagonal
                np.fill_diagonal(matrices[c], 1.0 + c * 0.1)

            elif pattern_type == "tridiagonal":
                # Main diagonal + first upper/lower diagonal
                np.fill_diagonal(matrices[c], 2.0 + c * 0.1)
                if led_count > 1:
                    np.fill_diagonal(matrices[c][:-1, 1:], 0.5 + c * 0.05)
                    np.fill_diagonal(matrices[c][1:, :-1], 0.5 + c * 0.05)  # Symmetric

            elif pattern_type == "block":
                # Block diagonal pattern (simulates LED adjacency)
                block_size = min(4, led_count // 2)
                for i in range(0, led_count, block_size):
                    end_i = min(i + block_size, led_count)
                    for j in range(i, min(i + block_size * 2, led_count)):
                        end_j = min(j + 1, led_count)
                        if i < led_count and j < led_count:
                            matrices[c, i:end_i, j:end_j] = 0.3 + c * 0.1
                            matrices[c, j:end_j, i:end_i] = 0.3 + c * 0.1  # Symmetric

            elif pattern_type == "random":
                # Random sparse symmetric matrix
                np.random.seed(42 + c)  # Reproducible
                temp = np.random.randn(led_count, led_count).astype(np.float32)
                temp = (temp + temp.T) / 2  # Make symmetric
                # Sparsify
                mask = np.random.rand(led_count, led_count) < 0.3
                mask = mask | mask.T  # Keep symmetric
                np.fill_diagonal(mask, True)  # Keep diagonal
                matrices[c] = temp * mask

        return matrices

    def test_diagonal_matrix(self):
        """Test with pure diagonal matrix."""
        for led_count in self.led_counts:
            with self.subTest(led_count=led_count):
                # Create diagonal matrix
                dense_matrices = self.create_test_matrix(led_count, "diagonal")

                # Create symmetric DIA
                symmetric_dia = SymmetricDiagonalATAMatrix.from_dense(
                    dense_matrices, led_count, significance_threshold=0.01
                )

                # Should only have main diagonal (offset 0)
                self.assertEqual(symmetric_dia.k_upper, 1)
                self.assertEqual(symmetric_dia.bandwidth, 0)
                self.assertTrue(np.array_equal(symmetric_dia.dia_offsets_upper, [0]))

                # Test matrix-vector multiplication
                self._test_matvec_consistency(dense_matrices, symmetric_dia, led_count)

    def test_tridiagonal_matrix(self):
        """Test with tridiagonal matrix."""
        for led_count in [8, 16]:  # Skip 32 for faster testing
            with self.subTest(led_count=led_count):
                # Create tridiagonal matrix
                dense_matrices = self.create_test_matrix(led_count, "tridiagonal")

                # Create symmetric DIA
                symmetric_dia = SymmetricDiagonalATAMatrix.from_dense(
                    dense_matrices, led_count, significance_threshold=0.01
                )

                # Should have main diagonal and first upper diagonal
                self.assertEqual(symmetric_dia.k_upper, 2)
                self.assertEqual(symmetric_dia.bandwidth, 1)
                self.assertTrue(np.array_equal(symmetric_dia.dia_offsets_upper, [0, 1]))

                # Test matrix-vector multiplication
                self._test_matvec_consistency(dense_matrices, symmetric_dia, led_count)

    def test_block_matrix(self):
        """Test with block-structured matrix."""
        for led_count in [16, 32]:
            with self.subTest(led_count=led_count):
                # Create block matrix
                dense_matrices = self.create_test_matrix(led_count, "block")

                # Create symmetric DIA with higher threshold for this test
                symmetric_dia = SymmetricDiagonalATAMatrix.from_dense(
                    dense_matrices, led_count, significance_threshold=0.1
                )

                # Should have reasonable number of diagonals based on block structure
                self.assertGreater(symmetric_dia.k_upper, 1)
                self.assertLessEqual(symmetric_dia.k_upper, led_count // 2)

                # Test matrix-vector multiplication
                self._test_matvec_consistency(dense_matrices, symmetric_dia, led_count)

    def test_random_sparse_matrix(self):
        """Test with random sparse symmetric matrix."""
        for led_count in [8, 16]:
            with self.subTest(led_count=led_count):
                # Create random sparse matrix
                dense_matrices = self.create_test_matrix(led_count, "random")

                # Create symmetric DIA
                symmetric_dia = SymmetricDiagonalATAMatrix.from_dense(
                    dense_matrices, led_count, significance_threshold=0.05
                )

                # Should have reasonable sparsity
                self.assertGreater(symmetric_dia.k_upper, 0)
                self.assertLessEqual(symmetric_dia.k_upper, led_count)

                # Test matrix-vector multiplication
                self._test_matvec_consistency(dense_matrices, symmetric_dia, led_count)

    def test_significance_threshold(self):
        """Test different significance thresholds."""
        led_count = 16
        dense_matrices = self.create_test_matrix(led_count, "random")

        # Test with different thresholds
        thresholds = [0.001, 0.01, 0.1, 0.5]
        prev_k_upper = float("inf")

        for threshold in thresholds:
            with self.subTest(threshold=threshold):
                symmetric_dia = SymmetricDiagonalATAMatrix.from_dense(
                    dense_matrices, led_count, significance_threshold=threshold
                )

                # Higher threshold should give fewer diagonals
                self.assertLessEqual(symmetric_dia.k_upper, prev_k_upper)
                prev_k_upper = symmetric_dia.k_upper

                # Should always have at least main diagonal
                self.assertGreaterEqual(symmetric_dia.k_upper, 1)

    def test_edge_cases(self):
        """Test edge cases like very small matrices."""
        # Test 1x1 matrix
        dense_matrices = np.ones((3, 1, 1), dtype=np.float32)
        symmetric_dia = SymmetricDiagonalATAMatrix.from_dense(dense_matrices, 1, significance_threshold=0.01)

        self.assertEqual(symmetric_dia.k_upper, 1)
        self.assertEqual(symmetric_dia.bandwidth, 0)

        # Test 2x2 matrix
        dense_matrices = np.array(
            [[[2.0, 1.0], [1.0, 2.0]], [[3.0, 1.5], [1.5, 3.0]], [[4.0, 2.0], [2.0, 4.0]]], dtype=np.float32
        )

        symmetric_dia = SymmetricDiagonalATAMatrix.from_dense(dense_matrices, 2, significance_threshold=0.01)

        self.assertEqual(symmetric_dia.k_upper, 2)  # Main + first upper
        self.assertEqual(symmetric_dia.bandwidth, 1)

    def _test_matvec_consistency(
        self, dense_matrices: np.ndarray, symmetric_dia: SymmetricDiagonalATAMatrix, led_count: int
    ):
        """
        Test that matrix-vector multiplication gives consistent results between
        dense matrix multiplication and symmetric DIA multiplication.
        """
        # Create test vector
        np.random.seed(123)
        test_vector = np.random.randn(3, led_count).astype(np.float32)

        # Compute reference result using dense matrix
        reference_result = np.zeros((3, led_count), dtype=np.float32)
        for c in range(3):
            reference_result[c] = dense_matrices[c] @ test_vector[c]

        if CUPY_AVAILABLE:
            # Test with CuPy arrays
            test_vector_gpu = cupy.asarray(test_vector)

            # Compute result using symmetric DIA (fallback implementation)
            symmetric_result_gpu = symmetric_dia.multiply_3d(test_vector_gpu, use_custom_kernel=False)
            symmetric_result = cupy.asnumpy(symmetric_result_gpu)

            # Compare results (allow for small numerical differences)
            np.testing.assert_allclose(
                reference_result,
                symmetric_result,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Matrix-vector multiplication mismatch for {led_count} LEDs",
            )

    def test_element_by_element_accuracy(self):
        """Test element-by-element accuracy of stored diagonals."""
        led_count = 8
        dense_matrices = self.create_test_matrix(led_count, "tridiagonal")

        symmetric_dia = SymmetricDiagonalATAMatrix.from_dense(dense_matrices, led_count, significance_threshold=0.01)

        # Convert back to CPU for detailed checking
        dia_data_cpu = cupy.asnumpy(symmetric_dia.dia_data_gpu)

        # Check each stored diagonal
        for i, offset in enumerate(symmetric_dia.dia_offsets_upper):
            for c in range(3):
                # Extract diagonal from original matrix
                diagonal_length = led_count - offset
                if diagonal_length > 0:
                    row_indices = np.arange(diagonal_length)
                    col_indices = row_indices + offset
                    expected_diagonal = dense_matrices[c, row_indices, col_indices]

                    # Compare with stored diagonal (accounting for DIA format offset)
                    stored_diagonal = dia_data_cpu[c, i, offset : offset + diagonal_length]

                    np.testing.assert_allclose(
                        expected_diagonal,
                        stored_diagonal,
                        rtol=1e-6,
                        atol=1e-7,
                        err_msg=f"Diagonal mismatch at offset {offset}, channel {c}",
                    )

    def test_matrix_properties(self):
        """Test that matrix properties are correctly computed."""
        led_count = 16
        dense_matrices = self.create_test_matrix(led_count, "block")

        symmetric_dia = SymmetricDiagonalATAMatrix.from_dense(dense_matrices, led_count, significance_threshold=0.05)

        # Test basic properties
        self.assertEqual(symmetric_dia.led_count, led_count)
        self.assertEqual(symmetric_dia.channels, 3)
        self.assertGreater(symmetric_dia.nnz, 0)
        self.assertGreater(symmetric_dia.sparsity, 0)
        self.assertLessEqual(symmetric_dia.sparsity, 100)
        self.assertGreaterEqual(symmetric_dia.bandwidth, 0)
        self.assertLessEqual(symmetric_dia.bandwidth, led_count - 1)

        # Test that GPU arrays are created
        self.assertIsNotNone(symmetric_dia.dia_data_gpu)
        self.assertIsNotNone(symmetric_dia.dia_offsets_upper_gpu)
        self.assertEqual(symmetric_dia.dia_data_gpu.shape, (3, symmetric_dia.k_upper, led_count))
        self.assertEqual(symmetric_dia.dia_offsets_upper_gpu.shape, (symmetric_dia.k_upper,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
