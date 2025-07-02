#!/usr/bin/env python3
"""
Unit tests for LEDDiffusionCSCMatrix wrapper class.

This module tests all functionality of the LED diffusion CSC matrix wrapper,
including storage format encapsulation, dense materialization, bounding box
calculation, region extraction, image setting, and matrix stacking.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLEDDiffusionCSCMatrix:
    """Test suite for LEDDiffusionCSCMatrix class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a small test matrix with known patterns
        self.height = 10
        self.width = 8
        self.channels = 3
        self.led_count = 4
        self.pixels = self.height * self.width

        # Create test data with distinct patterns for each LED/channel
        self.test_matrix = self._create_test_matrix()
        self.wrapper = LEDDiffusionCSCMatrix.from_csc_matrix(self.test_matrix, self.height, self.width, self.channels)

    def _create_test_matrix(self) -> sp.csc_matrix:
        """Create a test CSC matrix with known patterns."""
        total_cols = self.led_count * self.channels
        matrix = sp.lil_matrix((self.pixels, total_cols), dtype=np.float32)

        # LED 0, Channel 0 (Red): 3x3 block at (2,2) with value 1.0
        for i in range(2, 5):
            for j in range(2, 5):
                linear_idx = i * self.width + j
                matrix[linear_idx, 0] = 1.0

        # LED 0, Channel 1 (Green): 2x2 block at (1,1) with value 0.5
        for i in range(1, 3):
            for j in range(1, 3):
                linear_idx = i * self.width + j
                matrix[linear_idx, 1] = 0.5

        # LED 1, Channel 0 (Red): Single pixel at (5,5) with value 2.0
        linear_idx = 5 * self.width + 5
        matrix[linear_idx, 3] = 2.0  # LED 1, channel 0 is column 3

        # LED 2, Channel 2 (Blue): Diagonal line with decreasing values
        for i in range(3):
            linear_idx = i * self.width + i
            matrix[linear_idx, 8] = 1.0 - i * 0.2  # LED 2, channel 2 is column 8

        return matrix.tocsc()

    def test_initialization(self):
        """Test matrix wrapper initialization."""
        assert self.wrapper.height == self.height
        assert self.wrapper.width == self.width
        assert self.wrapper.channels == self.channels
        assert self.wrapper.led_count == self.led_count
        assert self.wrapper.pixels == self.pixels
        assert self.wrapper.shape == (self.pixels, self.led_count * self.channels)

    def test_initialization_validation(self):
        """Test input validation during initialization."""
        # Test mismatched spatial dimensions
        bad_matrix = sp.csc_matrix(
            (np.array([1.0]), np.array([0]), np.array([0, 1, 1, 1])),
            shape=(50, 3),
            dtype=np.float32,  # 50 != 10*8
        )
        with pytest.raises(ValueError, match="Matrix rows.*!= height"):
            LEDDiffusionCSCMatrix(csc_matrix=bad_matrix, height=10, width=8, channels=3)

        # Test non-divisible columns
        bad_matrix2 = sp.csc_matrix(
            (
                np.array([1.0]),
                np.array([0]),
                np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            ),
            shape=(80, 10),
            dtype=np.float32,  # 10 not divisible by 3
        )
        with pytest.raises(ValueError, match="not divisible by channels"):
            LEDDiffusionCSCMatrix(csc_matrix=bad_matrix2, height=10, width=8, channels=3)

    def test_to_csc_matrix(self):
        """Test conversion back to scipy CSC matrix."""
        recovered_matrix = self.wrapper.to_csc_matrix()

        # Check that we get back the original matrix
        assert recovered_matrix.shape == self.test_matrix.shape
        assert recovered_matrix.nnz == self.test_matrix.nnz
        assert recovered_matrix.format == "csc"

        # Check data equality
        np.testing.assert_array_equal(recovered_matrix.data, self.test_matrix.data)
        np.testing.assert_array_equal(recovered_matrix.indices, self.test_matrix.indices)
        np.testing.assert_array_equal(recovered_matrix.indptr, self.test_matrix.indptr)

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        # Export to dictionary
        data_dict = self.wrapper.to_dict()

        # Check expected keys are present
        expected_keys = {
            "csc_data",
            "csc_indices",
            "csc_indptr",
            "csc_shape",
            "csc_height",
            "csc_width",
            "csc_channels",
            "csc_led_count",
            "csc_nnz",
        }
        assert set(data_dict.keys()) == expected_keys

        # Check metadata values
        assert data_dict["csc_height"] == self.height
        assert data_dict["csc_width"] == self.width
        assert data_dict["csc_channels"] == self.channels
        assert data_dict["csc_led_count"] == self.led_count

        # Load from dictionary
        loaded_wrapper = LEDDiffusionCSCMatrix.from_dict(data_dict)

        # Verify loaded wrapper matches original
        assert loaded_wrapper.height == self.wrapper.height
        assert loaded_wrapper.width == self.wrapper.width
        assert loaded_wrapper.channels == self.wrapper.channels
        assert loaded_wrapper.led_count == self.wrapper.led_count

        # Check matrix data matches
        np.testing.assert_array_equal(loaded_wrapper.data, self.wrapper.data)
        np.testing.assert_array_equal(loaded_wrapper.indices, self.wrapper.indices)
        np.testing.assert_array_equal(loaded_wrapper.indptr, self.wrapper.indptr)

    def test_materialize_dense(self):
        """Test dense materialization of LED patterns."""
        # Test LED 0, Channel 0 (should have 3x3 block)
        dense = self.wrapper.materialize_dense(0, 0)
        assert dense.shape == (self.height, self.width)

        # Check the 3x3 block at (2,2)
        expected_region = np.ones((3, 3)) * 1.0
        np.testing.assert_array_equal(dense[2:5, 2:5], expected_region)

        # Check rest is zeros
        dense_copy = dense.copy()
        dense_copy[2:5, 2:5] = 0
        np.testing.assert_array_equal(dense_copy, np.zeros_like(dense_copy))

        # Test LED 0, Channel 1 (should have 2x2 block)
        dense = self.wrapper.materialize_dense(0, 1)
        expected_region = np.ones((2, 2)) * 0.5
        np.testing.assert_array_equal(dense[1:3, 1:3], expected_region)

        # Test LED 1, Channel 0 (should have single pixel)
        dense = self.wrapper.materialize_dense(1, 0)
        assert dense[5, 5] == 2.0
        dense[5, 5] = 0  # Clear the single pixel
        np.testing.assert_array_equal(dense, np.zeros_like(dense))

    def test_materialize_dense_validation(self):
        """Test validation in materialize_dense method."""
        # Test invalid LED index
        with pytest.raises(ValueError, match="LED index.*out of range"):
            self.wrapper.materialize_dense(10, 0)

        # Test invalid channel index
        with pytest.raises(ValueError, match="Channel.*out of range"):
            self.wrapper.materialize_dense(0, 5)

    def test_get_bounding_box(self):
        """Test bounding box calculation."""
        # Test LED 0, Channel 0 (3x3 block at (2,2))
        bbox = self.wrapper.get_bounding_box(0, 0)
        assert bbox == (2, 2, 4, 4)  # min_row, min_col, max_row, max_col

        # Test LED 0, Channel 1 (2x2 block at (1,1))
        bbox = self.wrapper.get_bounding_box(0, 1)
        assert bbox == (1, 1, 2, 2)

        # Test LED 1, Channel 0 (single pixel at (5,5))
        bbox = self.wrapper.get_bounding_box(1, 0)
        assert bbox == (5, 5, 5, 5)

        # Test empty pattern (LED 3, Channel 0 should be empty)
        bbox = self.wrapper.get_bounding_box(3, 0)
        assert bbox == (0, 0, 0, 0)

    def test_extract_region(self):
        """Test region extraction."""
        # Extract the 3x3 block from LED 0, Channel 0
        region = self.wrapper.extract_region(0, 0, 2, 2, 4, 4)
        expected = np.ones((3, 3)) * 1.0
        np.testing.assert_array_equal(region, expected)

        # Extract larger region that includes zeros
        region = self.wrapper.extract_region(0, 0, 1, 1, 5, 5)
        assert region.shape == (5, 5)
        # Check the inner 3x3 block is correct
        np.testing.assert_array_equal(region[1:4, 1:4], np.ones((3, 3)))

        # Extract single pixel
        region = self.wrapper.extract_region(1, 0, 5, 5, 5, 5)
        assert region.shape == (1, 1)
        assert region[0, 0] == 2.0

    def test_extract_region_validation(self):
        """Test validation in extract_region method."""
        # Test invalid row range
        with pytest.raises(ValueError, match="Invalid row range"):
            self.wrapper.extract_region(0, 0, -1, 0, 5, 5)

        with pytest.raises(ValueError, match="Invalid row range"):
            self.wrapper.extract_region(0, 0, 5, 0, 15, 5)  # max_row >= height

        # Test invalid column range
        with pytest.raises(ValueError, match="Invalid col range"):
            self.wrapper.extract_region(0, 0, 0, -1, 5, 5)

        with pytest.raises(ValueError, match="Invalid col range"):
            self.wrapper.extract_region(0, 0, 0, 5, 5, 15)  # max_col >= width

    def test_set_image(self):
        """Test setting image from dense array."""
        # Create a new pattern (2x2 block with value 3.0)
        new_pattern = np.zeros((self.height, self.width))
        new_pattern[6:8, 3:5] = 3.0

        # Set it to LED 3, Channel 1 (which should be empty initially)
        original_nnz = len(self.wrapper.data)
        self.wrapper.set_image(3, 1, new_pattern)

        # Verify the pattern was set correctly
        dense = self.wrapper.materialize_dense(3, 1)
        np.testing.assert_array_equal(dense, new_pattern)

        # Verify bounding box is correct
        bbox = self.wrapper.get_bounding_box(3, 1)
        assert bbox == (6, 3, 7, 4)

        # Verify matrix size increased (we added 4 non-zeros)
        assert len(self.wrapper.data) == original_nnz + 4

    def test_set_image_replace(self):
        """Test replacing existing image with set_image."""
        # Replace LED 0, Channel 0 with a different pattern
        new_pattern = np.zeros((self.height, self.width))
        new_pattern[0, 0] = 5.0  # Single pixel

        original_nnz = len(self.wrapper.data)
        self.wrapper.set_image(0, 0, new_pattern)

        # Verify the new pattern
        dense = self.wrapper.materialize_dense(0, 0)
        np.testing.assert_array_equal(dense, new_pattern)

        # The original pattern had 9 non-zeros, new has 1, so total should decrease by 8
        assert len(self.wrapper.data) == original_nnz - 8

    def test_set_image_validation(self):
        """Test validation in set_image method."""
        wrong_shape = np.zeros((5, 5))

        # Test wrong shape
        with pytest.raises(ValueError, match="Dense image shape.*!= expected"):
            self.wrapper.set_image(0, 0, wrong_shape)

        # Test invalid indices
        correct_shape = np.zeros((self.height, self.width))
        with pytest.raises(ValueError, match="LED index.*out of range"):
            self.wrapper.set_image(10, 0, correct_shape)

    def test_hstack(self):
        """Test horizontal stacking of matrices."""
        # Create a second matrix with different LED count
        matrix2_led_count = 2
        matrix2_cols = matrix2_led_count * self.channels
        matrix2 = sp.lil_matrix((self.pixels, matrix2_cols), dtype=np.float32)

        # Add some data to second matrix
        matrix2[10, 0] = 7.0  # LED 0, Channel 0
        matrix2[20, 3] = 8.0  # LED 1, Channel 0

        wrapper2 = LEDDiffusionCSCMatrix.from_csc_matrix(matrix2.tocsc(), self.height, self.width, self.channels)

        # Stack the matrices
        stacked = LEDDiffusionCSCMatrix.hstack([self.wrapper, wrapper2])

        # Verify properties
        assert stacked.height == self.height
        assert stacked.width == self.width
        assert stacked.channels == self.channels
        assert stacked.led_count == self.led_count + matrix2_led_count
        assert stacked.shape == (
            self.pixels,
            (self.led_count + matrix2_led_count) * self.channels,
        )

        # Verify data from first matrix is preserved
        dense_original = self.wrapper.materialize_dense(0, 0)
        dense_stacked = stacked.materialize_dense(0, 0)
        np.testing.assert_array_equal(dense_original, dense_stacked)

        # Verify data from second matrix is in correct position
        dense_second = stacked.materialize_dense(self.led_count, 0)  # First LED from second matrix
        expected = np.zeros((self.height, self.width))
        expected.flat[10] = 7.0
        np.testing.assert_array_equal(dense_second, expected)

    def test_hstack_validation(self):
        """Test validation in hstack method."""
        # Test empty list
        with pytest.raises(ValueError, match="Cannot stack empty list"):
            LEDDiffusionCSCMatrix.hstack([])

        # Test mismatched dimensions
        different_height = LEDDiffusionCSCMatrix.from_csc_matrix(
            sp.csc_matrix((50, 3)),
            height=5,
            width=10,
            channels=3,  # Different height
        )

        with pytest.raises(ValueError, match="spatial dims.*!= first matrix"):
            LEDDiffusionCSCMatrix.hstack([self.wrapper, different_height])

    def test_memory_info(self):
        """Test memory usage reporting."""
        info = self.wrapper.memory_info()

        # Check required keys
        required_keys = {
            "data_mb",
            "indices_mb",
            "indptr_mb",
            "total_mb",
            "equivalent_dense_mb",
            "compression_ratio",
            "sparsity_ratio",
            "nnz",
            "shape",
        }
        assert set(info.keys()) == required_keys

        # Check basic sanity
        assert info["total_mb"] > 0
        assert info["compression_ratio"] < 1.0  # Should be compressed
        assert info["sparsity_ratio"] < 1.0  # Should be sparse
        assert info["nnz"] == len(self.wrapper.data)
        assert info["shape"] == self.wrapper.shape

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.wrapper)

        # Check that key information is included
        assert "LEDDiffusionCSCMatrix" in repr_str
        assert f"shape={self.wrapper.shape}" in repr_str
        assert f"led_count={self.led_count}" in repr_str
        assert f"channels={self.channels}" in repr_str
        assert "nnz=" in repr_str
        assert "sparsity=" in repr_str
        assert "memory=" in repr_str

    def test_round_trip_consistency(self):
        """Test full round-trip: create -> to_dict -> save -> load -> from_dict."""
        # Export original wrapper
        original_dict = self.wrapper.to_dict()

        # Simulate save/load by creating from dict
        loaded_wrapper = LEDDiffusionCSCMatrix.from_dict(original_dict)

        # Test that all LED patterns are identical
        for led_idx in range(self.led_count):
            for channel in range(self.channels):
                original_dense = self.wrapper.materialize_dense(led_idx, channel)
                loaded_dense = loaded_wrapper.materialize_dense(led_idx, channel)
                np.testing.assert_array_equal(
                    original_dense,
                    loaded_dense,
                    err_msg=f"Mismatch in LED {led_idx}, channel {channel}",
                )

        # Test that matrix properties match
        assert loaded_wrapper.shape == self.wrapper.shape
        assert loaded_wrapper.height == self.wrapper.height
        assert loaded_wrapper.width == self.wrapper.width
        assert loaded_wrapper.channels == self.wrapper.channels
        assert loaded_wrapper.led_count == self.wrapper.led_count

    def test_to_dense_patterns(self):
        """Test conversion to dense patterns array."""
        dense_patterns = self.wrapper.to_dense_patterns()

        # Check shape
        expected_shape = (self.led_count, self.height, self.width, self.channels)
        assert dense_patterns.shape == expected_shape
        assert dense_patterns.dtype == np.float32

        # Check that patterns match individual materialize_dense calls
        for led_idx in range(self.led_count):
            for channel in range(self.channels):
                individual_pattern = self.wrapper.materialize_dense(led_idx, channel)
                batch_pattern = dense_patterns[led_idx, :, :, channel]
                np.testing.assert_array_equal(
                    individual_pattern,
                    batch_pattern,
                    err_msg=f"Mismatch in LED {led_idx}, channel {channel}",
                )

        # Check that we have some non-zero values (based on test data)
        assert np.any(dense_patterns > 0), "Dense patterns should have some non-zero values"

    def test_get_pattern_summary(self):
        """Test pattern summary statistics."""
        summary = self.wrapper.get_pattern_summary()

        # Check required keys
        required_keys = {
            "led_count",
            "channels",
            "matrix_shape",
            "nnz_total",
            "sparsity_ratio",
            "max_intensities",
            "mean_intensities",
            "pattern_extents",
            "channel_nnz",
            "memory_mb",
        }
        assert set(summary.keys()) == required_keys

        # Check basic properties
        assert summary["led_count"] == self.led_count
        assert summary["channels"] == self.channels
        assert summary["matrix_shape"] == list(self.wrapper.shape)
        assert summary["nnz_total"] == self.wrapper.matrix.nnz
        assert 0 <= summary["sparsity_ratio"] <= 1

        # Check array shapes
        assert summary["max_intensities"].shape == (self.led_count,)
        assert summary["mean_intensities"].shape == (self.led_count,)
        assert summary["pattern_extents"].shape == (self.led_count,)
        assert summary["channel_nnz"].shape == (self.channels,)

        # Check that max intensities are non-negative
        assert np.all(summary["max_intensities"] >= 0)
        assert np.all(summary["mean_intensities"] >= 0)
        assert np.all(summary["pattern_extents"] >= 0)

        # Check that channel non-zero counts sum to total nnz
        assert np.sum(summary["channel_nnz"]) == summary["nnz_total"]

        # Check memory is positive
        assert summary["memory_mb"] > 0

    def test_get_led_bounding_boxes(self):
        """Test LED bounding box computation."""
        bboxes = self.wrapper.get_led_bounding_boxes()

        # Check shape
        assert bboxes.shape == (self.led_count, 4)
        assert bboxes.dtype == int

        # Check each LED's bounding box
        for led_idx in range(self.led_count):
            bbox = bboxes[led_idx]
            min_row, min_col, max_row, max_col = bbox

            # Check basic validity
            assert min_row >= 0 and min_row < self.height
            assert min_col >= 0 and min_col < self.width
            assert max_row >= 0 and max_row < self.height
            assert max_col >= 0 and max_col < self.width

            # Check ordering (if not all zeros)
            if not np.array_equal(bbox, [0, 0, 0, 0]):
                assert min_row <= max_row
                assert min_col <= max_col

                # Check that it matches union of individual channel bounding boxes
                expected_min_row, expected_min_col = self.height, self.width
                expected_max_row, expected_max_col = -1, -1
                has_data = False

                for channel in range(self.channels):
                    ch_bbox = self.wrapper.get_bounding_box(led_idx, channel)
                    if ch_bbox != (0, 0, 0, 0):
                        ch_min_row, ch_min_col, ch_max_row, ch_max_col = ch_bbox
                        expected_min_row = min(expected_min_row, ch_min_row)
                        expected_min_col = min(expected_min_col, ch_min_col)
                        expected_max_row = max(expected_max_row, ch_max_row)
                        expected_max_col = max(expected_max_col, ch_max_col)
                        has_data = True

                if has_data:
                    expected_bbox = [
                        expected_min_row,
                        expected_min_col,
                        expected_max_row,
                        expected_max_col,
                    ]
                    np.testing.assert_array_equal(
                        bbox,
                        expected_bbox,
                        err_msg=f"LED {led_idx} bounding box mismatch",
                    )

    def test_enhancement_methods_empty_matrix(self):
        """Test enhancement methods with empty matrix."""
        empty_matrix = sp.csc_matrix((self.pixels, self.led_count * self.channels), dtype=np.float32)
        empty_wrapper = LEDDiffusionCSCMatrix.from_csc_matrix(empty_matrix, self.height, self.width, self.channels)

        # Test to_dense_patterns
        dense_patterns = empty_wrapper.to_dense_patterns()
        assert dense_patterns.shape == (
            self.led_count,
            self.height,
            self.width,
            self.channels,
        )
        np.testing.assert_array_equal(dense_patterns, np.zeros_like(dense_patterns))

        # Test get_pattern_summary
        summary = empty_wrapper.get_pattern_summary()
        assert summary["nnz_total"] == 0
        assert summary["sparsity_ratio"] == 0.0
        assert np.all(summary["max_intensities"] == 0)
        assert np.all(summary["mean_intensities"] == 0)
        assert np.all(summary["channel_nnz"] == 0)

        # Test get_led_bounding_boxes
        bboxes = empty_wrapper.get_led_bounding_boxes()
        assert bboxes.shape == (self.led_count, 4)
        np.testing.assert_array_equal(bboxes, np.zeros_like(bboxes))


def test_empty_matrix():
    """Test handling of empty matrix."""
    empty_matrix = sp.csc_matrix((80, 12), dtype=np.float32)  # 4 LEDs, 3 channels
    wrapper = LEDDiffusionCSCMatrix.from_csc_matrix(empty_matrix, height=10, width=8, channels=3)

    # All patterns should be zero
    for led_idx in range(4):
        for channel in range(3):
            dense = wrapper.materialize_dense(led_idx, channel)
            np.testing.assert_array_equal(dense, np.zeros((10, 8)))

            bbox = wrapper.get_bounding_box(led_idx, channel)
            assert bbox == (0, 0, 0, 0)


def test_single_pixel_patterns():
    """Test handling of single-pixel patterns."""
    matrix = sp.lil_matrix((80, 3), dtype=np.float32)  # 1 LED, 3 channels
    matrix[25, 0] = 1.0  # Single pixel in red channel

    wrapper = LEDDiffusionCSCMatrix.from_csc_matrix(matrix.tocsc(), height=10, width=8, channels=3)

    # Check red channel has single pixel
    dense = wrapper.materialize_dense(0, 0)
    expected = np.zeros((10, 8))
    expected[3, 1] = 1.0  # Linear index 25 = row 3, col 1
    np.testing.assert_array_equal(dense, expected)

    # Check bounding box
    bbox = wrapper.get_bounding_box(0, 0)
    assert bbox == (3, 1, 3, 1)

    # Check region extraction
    region = wrapper.extract_region(0, 0, 3, 1, 3, 1)
    assert region.shape == (1, 1)
    assert region[0, 0] == 1.0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
