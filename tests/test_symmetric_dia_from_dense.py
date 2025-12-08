#!/usr/bin/env python3
"""
Unit tests for SymmetricDiagonalATAMatrix.from_dense() method.

Tests the conversion from dense ATA matrices to symmetric DIA format,
which is the key conversion used in tools/compute_matrices.py during
pattern file generation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix


@pytest.fixture(autouse=True)
def cuda_cleanup():
    """Ensure clean CUDA state before and after each test."""
    if cp.cuda.is_available():
        try:
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    yield

    if cp.cuda.is_available():
        try:
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def create_banded_symmetric_ata(led_count: int, bandwidth: int) -> np.ndarray:
    """
    Create a synthetic banded symmetric ATA matrix for testing.

    Creates a positive-definite matrix with known structure:
    - Main diagonal has values 1.0
    - Off-diagonals decay as 0.5^offset

    Args:
        led_count: Number of LEDs (matrix size)
        bandwidth: Maximum diagonal offset

    Returns:
        Dense ATA matrices of shape (3, led_count, led_count)
    """
    dense_ata = np.zeros((3, led_count, led_count), dtype=np.float32)

    for c in range(3):
        # Scale factor per channel (different for each to test channel independence)
        scale = 1.0 + 0.1 * c

        for i in range(led_count):
            for j in range(led_count):
                offset = abs(i - j)
                if offset <= bandwidth:
                    # Decay with distance, symmetric
                    value = scale * (0.5**offset)
                    dense_ata[c, i, j] = value

    return dense_ata


class TestSymmetricDiaFromDense:
    """Tests for SymmetricDiagonalATAMatrix.from_dense() method."""

    def test_from_dense_basic_creation(self):
        """Test basic creation from dense matrices."""
        led_count = 100
        bandwidth = 10

        # Create dense ATA matrix
        dense_ata = create_banded_symmetric_ata(led_count, bandwidth)

        # Convert to DIA format
        dia_matrix = SymmetricDiagonalATAMatrix.from_dense(
            dense_ata, led_count, significance_threshold=0.01, crop_size=64
        )

        # Verify basic properties
        assert dia_matrix.led_count == led_count
        assert dia_matrix.channels == 3
        assert dia_matrix.k_upper > 0
        assert dia_matrix.bandwidth <= bandwidth + 1  # May capture slightly more
        assert dia_matrix.dia_data_gpu is not None
        assert dia_matrix.dia_offsets_upper_gpu is not None

    def test_from_dense_shape_validation(self):
        """Test that from_dense validates input shape."""
        led_count = 50

        # Wrong number of channels
        wrong_channels = np.zeros((2, led_count, led_count), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected dense_ata_matrices shape"):
            SymmetricDiagonalATAMatrix.from_dense(wrong_channels, led_count)

        # Wrong matrix dimensions
        wrong_dims = np.zeros((3, led_count, led_count + 10), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected dense_ata_matrices shape"):
            SymmetricDiagonalATAMatrix.from_dense(wrong_dims, led_count)

    def test_from_dense_diagonal_extraction(self):
        """Test that diagonals are correctly extracted from dense matrix."""
        led_count = 20
        bandwidth = 5

        # Create dense ATA with known pattern
        dense_ata = create_banded_symmetric_ata(led_count, bandwidth)

        # Convert to DIA
        dia_matrix = SymmetricDiagonalATAMatrix.from_dense(
            dense_ata, led_count, significance_threshold=0.001, crop_size=64
        )

        # Verify main diagonal (offset=0) is correct
        dia_data = cp.asnumpy(dia_matrix.dia_data_gpu)
        dia_offsets = dia_matrix.dia_offsets_upper

        # Find main diagonal index
        main_diag_idx = np.where(dia_offsets == 0)[0]
        assert len(main_diag_idx) == 1, "Main diagonal should exist"
        main_diag_idx = main_diag_idx[0]

        # Main diagonal values should match dense matrix diagonal
        for c in range(3):
            expected_main = np.diag(dense_ata[c])
            actual_main = dia_data[c, main_diag_idx, :]
            np.testing.assert_allclose(actual_main, expected_main, rtol=1e-5)

    def test_from_dense_multiply_matches_dense(self):
        """Test that DIA multiply produces same result as dense multiply."""
        led_count = 50
        bandwidth = 8

        # Create dense ATA matrix
        dense_ata = create_banded_symmetric_ata(led_count, bandwidth)

        # Convert to DIA
        dia_matrix = SymmetricDiagonalATAMatrix.from_dense(
            dense_ata, led_count, significance_threshold=0.001, crop_size=64
        )

        # Create test input vector
        np.random.seed(42)
        test_input = np.random.rand(3, led_count).astype(np.float32)
        test_input_gpu = cp.asarray(test_input)

        # Compute expected result with dense multiplication
        expected = np.zeros((3, led_count), dtype=np.float32)
        for c in range(3):
            expected[c] = dense_ata[c] @ test_input[c]

        # Compute result with DIA multiplication
        actual_gpu = dia_matrix.multiply_3d(test_input_gpu, use_custom_kernel=False)
        actual = cp.asnumpy(actual_gpu)

        # Compare results
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)

    def test_from_dense_significance_threshold(self):
        """Test that significance threshold filters small values."""
        led_count = 30
        bandwidth = 10

        # Create dense ATA
        dense_ata = create_banded_symmetric_ata(led_count, bandwidth)

        # Use high threshold to filter more
        dia_high_thresh = SymmetricDiagonalATAMatrix.from_dense(
            dense_ata, led_count, significance_threshold=0.1, crop_size=64
        )

        # Use low threshold to keep more
        dia_low_thresh = SymmetricDiagonalATAMatrix.from_dense(
            dense_ata, led_count, significance_threshold=0.001, crop_size=64
        )

        # High threshold should have fewer diagonals or sparser data
        assert dia_high_thresh.bandwidth <= dia_low_thresh.bandwidth

    def test_from_dense_symmetric_storage(self):
        """Test that only upper diagonals are stored (symmetric optimization)."""
        led_count = 40
        bandwidth = 6

        dense_ata = create_banded_symmetric_ata(led_count, bandwidth)

        dia_matrix = SymmetricDiagonalATAMatrix.from_dense(
            dense_ata, led_count, significance_threshold=0.001, crop_size=64
        )

        # All stored offsets should be non-negative (upper triangular)
        assert all(offset >= 0 for offset in dia_matrix.dia_offsets_upper)

        # Should not store more than led_count diagonals
        assert dia_matrix.k_upper <= led_count

    def test_from_dense_preserves_symmetry(self):
        """Test that multiplication result is consistent with symmetric matrix."""
        led_count = 40
        bandwidth = 5

        dense_ata = create_banded_symmetric_ata(led_count, bandwidth)

        dia_matrix = SymmetricDiagonalATAMatrix.from_dense(
            dense_ata, led_count, significance_threshold=0.001, crop_size=64
        )

        # Create two different test vectors
        np.random.seed(123)
        x = np.random.rand(3, led_count).astype(np.float32)
        y = np.random.rand(3, led_count).astype(np.float32)

        x_gpu = cp.asarray(x)
        y_gpu = cp.asarray(y)

        # Compute Ax and Ay
        Ax = cp.asnumpy(dia_matrix.multiply_3d(x_gpu, use_custom_kernel=False))
        Ay = cp.asnumpy(dia_matrix.multiply_3d(y_gpu, use_custom_kernel=False))

        # For symmetric A: <Ax, y> = <x, Ay>
        # This property must hold if the matrix is truly symmetric
        lhs = np.sum(Ax * y)  # <Ax, y>
        rhs = np.sum(x * Ay)  # <x, Ay>

        np.testing.assert_allclose(lhs, rhs, rtol=1e-4)

    def test_from_dense_to_dict_roundtrip(self):
        """Test that DIA matrix can be serialized and deserialized."""
        led_count = 30
        bandwidth = 5

        dense_ata = create_banded_symmetric_ata(led_count, bandwidth)

        # Create DIA matrix
        dia_matrix = SymmetricDiagonalATAMatrix.from_dense(
            dense_ata, led_count, significance_threshold=0.01, crop_size=64
        )

        # Serialize to dict
        dia_dict = dia_matrix.to_dict()

        # Verify dict has expected keys
        assert "led_count" in dia_dict
        assert "dia_data_gpu" in dia_dict
        assert "dia_offsets_upper" in dia_dict
        assert "bandwidth" in dia_dict
        assert "k_upper" in dia_dict
        assert dia_dict["led_count"] == led_count

        # Deserialize from dict
        restored = SymmetricDiagonalATAMatrix.from_dict(dia_dict)

        # Verify properties match
        assert restored.led_count == dia_matrix.led_count
        assert restored.bandwidth == dia_matrix.bandwidth
        assert restored.k_upper == dia_matrix.k_upper

        # Verify multiplication produces same result
        np.random.seed(456)
        test_input = np.random.rand(3, led_count).astype(np.float32)
        test_input_gpu = cp.asarray(test_input)

        original_result = cp.asnumpy(dia_matrix.multiply_3d(test_input_gpu, use_custom_kernel=False))
        restored_result = cp.asnumpy(restored.multiply_3d(test_input_gpu, use_custom_kernel=False))

        np.testing.assert_allclose(original_result, restored_result, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
