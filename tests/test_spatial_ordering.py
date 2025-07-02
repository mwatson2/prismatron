#!/usr/bin/env python3
"""
Tests for spatial ordering utilities.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from src.utils.spatial_ordering import (
    compute_rcm_ordering,
    reorder_block_values,
    reorder_matrix_columns,
)


class TestComputeRCMOrdering:
    """Test RCM ordering computation."""

    def test_empty_positions(self):
        """Test with empty positions array."""
        positions = np.array([]).reshape(0, 2)
        rcm_order, inverse_order = compute_rcm_ordering(positions, block_size=64)

        assert len(rcm_order) == 0
        assert len(inverse_order) == 0

    def test_single_position(self):
        """Test with single position."""
        positions = np.array([[100, 100]])
        rcm_order, inverse_order = compute_rcm_ordering(positions, block_size=64)

        assert len(rcm_order) == 1
        assert len(inverse_order) == 1
        assert rcm_order[0] == 0
        assert inverse_order[0] == 0

    def test_two_overlapping_positions(self):
        """Test with two overlapping positions."""
        # Two positions that should overlap with block_size=64
        positions = np.array([[100, 100], [120, 120]])  # Distance ~28, both blocks size 64
        rcm_order, inverse_order = compute_rcm_ordering(positions, block_size=64)

        assert len(rcm_order) == 2
        assert len(inverse_order) == 2
        assert set(rcm_order) == {0, 1}  # Should contain both indices

        # Test inverse relationship
        for i in range(2):
            assert inverse_order[rcm_order[i]] == i

    def test_two_non_overlapping_positions(self):
        """Test with two non-overlapping positions."""
        # Two positions that should NOT overlap with block_size=64
        positions = np.array([[100, 100], [300, 300]])  # Distance ~283, blocks size 64
        rcm_order, inverse_order = compute_rcm_ordering(positions, block_size=64)

        assert len(rcm_order) == 2
        assert len(inverse_order) == 2
        assert set(rcm_order) == {0, 1}

        # Test inverse relationship
        for i in range(2):
            assert inverse_order[rcm_order[i]] == i

    def test_grid_positions(self):
        """Test with grid of positions."""
        # 2x2 grid of positions that should form adjacencies
        positions = np.array(
            [
                [0, 0],  # Bottom-left
                [50, 0],  # Bottom-right
                [0, 50],  # Top-left
                [50, 50],  # Top-right
            ]
        )

        rcm_order, inverse_order = compute_rcm_ordering(positions, block_size=64)

        assert len(rcm_order) == 4
        assert len(inverse_order) == 4
        assert set(rcm_order) == {0, 1, 2, 3}

        # Test inverse relationship
        for i in range(4):
            assert inverse_order[rcm_order[i]] == i

    def test_ordering_consistency(self):
        """Test that ordering is consistent and reversible."""
        # Random positions
        np.random.seed(42)
        n_blocks = 10
        positions = np.random.uniform(0, 500, (n_blocks, 2))

        rcm_order, inverse_order = compute_rcm_ordering(positions, block_size=64)

        assert len(rcm_order) == n_blocks
        assert len(inverse_order) == n_blocks
        assert set(rcm_order) == set(range(n_blocks))

        # Test inverse relationship: inverse_order[rcm_order[i]] == i
        for i in range(n_blocks):
            assert inverse_order[rcm_order[i]] == i

        # Test forward relationship: rcm_order[inverse_order[i]] == i
        for i in range(n_blocks):
            assert rcm_order[inverse_order[i]] == i


class TestReorderMatrixColumns:
    """Test matrix column reordering."""

    def test_reorder_simple_matrix(self):
        """Test reordering a simple matrix."""
        # 3x6 matrix (3 rows, 2 blocks × 3 channels)
        matrix = sp.csr_matrix(
            [
                [1, 2, 3, 4, 5, 6],  # Row affects blocks [0,0,0,1,1,1]
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
            ]
        )

        # Reverse block order: [1, 0]
        block_order = np.array([1, 0])

        reordered = reorder_matrix_columns(matrix, block_order, channels_per_block=3)

        # Expected: columns [3,4,5,0,1,2] (block 1 channels, then block 0 channels)
        expected = sp.csr_matrix([[4, 5, 6, 1, 2, 3], [10, 11, 12, 7, 8, 9], [16, 17, 18, 13, 14, 15]])

        np.testing.assert_array_equal(reordered.toarray(), expected.toarray())

    def test_reorder_identity_order(self):
        """Test reordering with identity order (no change)."""
        matrix = sp.random(5, 9, density=0.5, random_state=42)  # 5x9 matrix (3 blocks × 3 channels)
        block_order = np.array([0, 1, 2])  # Identity order

        reordered = reorder_matrix_columns(matrix, block_order, channels_per_block=3)

        np.testing.assert_array_equal(reordered.toarray(), matrix.toarray())

    def test_reorder_dimension_mismatch(self):
        """Test error handling for dimension mismatch."""
        matrix = sp.csr_matrix([[1, 2, 3, 4]])  # 1x4 matrix
        block_order = np.array([0, 1])  # 2 blocks × 3 channels = 6 columns expected

        with pytest.raises(ValueError, match="Matrix should have 6 columns"):
            reorder_matrix_columns(matrix, block_order, channels_per_block=3)


class TestReorderBlockValues:
    """Test block values reordering."""

    def test_reorder_channels_blocks_shape(self):
        """Test reordering with (channels, blocks) shape."""
        # 3 channels, 2 blocks
        values = np.array(
            [
                [10, 20],  # Channel 0: block values [10, 20]
                [30, 40],  # Channel 1: block values [30, 40]
                [50, 60],  # Channel 2: block values [50, 60]
            ]
        )

        # Reverse block order: [1, 0]
        block_order = np.array([1, 0])

        reordered = reorder_block_values(values, block_order, from_ordered=False)

        expected = np.array(
            [
                [20, 10],  # Channel 0: reordered to [20, 10]
                [40, 30],  # Channel 1: reordered to [40, 30]
                [60, 50],  # Channel 2: reordered to [60, 50]
            ]
        )

        np.testing.assert_array_equal(reordered, expected)

    def test_reorder_blocks_channels_shape(self):
        """Test reordering with (blocks, channels) shape."""
        # 2 blocks, 3 channels
        values = np.array(
            [
                [10, 30, 50],  # Block 0: channel values [10, 30, 50]
                [20, 40, 60],  # Block 1: channel values [20, 40, 60]
            ]
        )

        # Reverse block order: [1, 0]
        block_order = np.array([1, 0])

        reordered = reorder_block_values(values, block_order, from_ordered=False)

        expected = np.array(
            [
                [20, 40, 60],  # Block 1 moved to position 0
                [10, 30, 50],  # Block 0 moved to position 1
            ]
        )

        np.testing.assert_array_equal(reordered, expected)

    def test_reorder_from_ordered(self):
        """Test reordering from spatial order back to original."""
        # Original values
        original = np.array(
            [
                [10, 20, 30],  # Channel 0
                [40, 50, 60],  # Channel 1
            ]
        )

        block_order = np.array([2, 0, 1])  # Spatial ordering

        # First reorder to spatial order
        spatial_ordered = reorder_block_values(original, block_order, from_ordered=False)

        # Then reorder back to original
        back_to_original = reorder_block_values(spatial_ordered, block_order, from_ordered=True)

        np.testing.assert_array_equal(back_to_original, original)

    def test_reorder_identity_order(self):
        """Test reordering with identity order (no change)."""
        values = np.array(
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
            ]
        )

        block_order = np.array([0, 1, 2])  # Identity order

        reordered = reorder_block_values(values, block_order, from_ordered=False)

        np.testing.assert_array_equal(reordered, values)

    def test_dimension_mismatch_error(self):
        """Test error handling for dimension mismatch."""
        values = np.array([[1, 2], [3, 4]])  # 2x2 array
        block_order = np.array([0, 1, 2])  # 3 blocks

        with pytest.raises(ValueError, match="Values array dimension mismatch"):
            reorder_block_values(values, block_order, from_ordered=False)

    def test_invalid_dimensions_error(self):
        """Test error handling for invalid array dimensions."""
        values = np.array([1, 2, 3])  # 1D array
        block_order = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="Values array should be 2D"):
            reorder_block_values(values, block_order, from_ordered=False)


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_rcm_matrix_reorder_roundtrip(self):
        """Test full workflow: RCM ordering + matrix reordering."""
        # Create test positions
        positions = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])  # 2x2 grid

        # Create test matrix (6 rows, 4 blocks × 3 channels = 12 columns)
        np.random.seed(42)
        matrix = sp.random(6, 12, density=0.3)

        # Compute RCM ordering
        rcm_order, inverse_order = compute_rcm_ordering(positions, block_size=64)

        # Reorder matrix
        reordered_matrix = reorder_matrix_columns(matrix, rcm_order, channels_per_block=3)

        # Should have same shape and density
        assert reordered_matrix.shape == matrix.shape
        assert abs(reordered_matrix.nnz - matrix.nnz) <= 1  # Allow small differences due to sparsity

    def test_values_reorder_roundtrip(self):
        """Test values reordering roundtrip."""
        # Test data
        original_values = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],  # Channel 0
                [5.0, 6.0, 7.0, 8.0],  # Channel 1
                [9.0, 10.0, 11.0, 12.0],  # Channel 2
            ]
        )

        positions = np.array([[0, 0], [50, 0], [0, 50], [50, 50]])
        rcm_order, _ = compute_rcm_ordering(positions, block_size=64)

        # Reorder to spatial order
        spatial_values = reorder_block_values(original_values, rcm_order, from_ordered=False)

        # Reorder back to original
        restored_values = reorder_block_values(spatial_values, rcm_order, from_ordered=True)

        np.testing.assert_array_equal(restored_values, original_values)
