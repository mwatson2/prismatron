"""
Unit tests for DiffusionPatternManager.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from src.utils.diffusion_pattern_manager import DiffusionPatternManager


class TestDiffusionPatternManager:
    """Test cases for DiffusionPatternManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.led_count = 10
        self.frame_height = 8
        self.frame_width = 12
        self.manager = DiffusionPatternManager(
            led_count=self.led_count,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
        )

    def test_initialization(self):
        """Test manager initialization."""
        assert self.manager.led_count == self.led_count
        assert self.manager.frame_height == self.frame_height
        assert self.manager.frame_width == self.frame_width
        assert self.manager.pixels_per_frame == self.frame_height * self.frame_width
        assert not self.manager.is_loaded
        assert not self.manager.is_generation_complete

    def test_frame_shapes(self):
        """Test frame shape methods."""
        expected_frame_shape = (3, self.frame_height, self.frame_width)
        expected_output_shape = (3, self.led_count)

        assert self.manager.get_frame_shape() == expected_frame_shape
        assert self.manager.get_led_output_shape() == expected_output_shape

    def test_pattern_generation_workflow(self):
        """Test complete pattern generation workflow."""
        # Start generation
        led_positions = np.random.rand(self.led_count, 2) * [
            self.frame_width,
            self.frame_height,
        ]
        self.manager.start_pattern_generation(led_positions)

        # Add patterns for each LED
        for led_id in range(self.led_count):
            # Create synthetic diffusion pattern in planar format
            pattern = self._create_synthetic_pattern(led_id)
            self.manager.add_led_pattern(led_id, pattern)

        # Finalize generation
        stats = self.manager.finalize_pattern_generation()

        # Verify generation completed
        assert self.manager.is_generation_complete
        assert stats["led_count"] == self.led_count
        assert stats["patterns_processed"] == self.led_count
        assert "generation_time" in stats
        assert "sparse_stats" in stats
        assert "ata_stats" in stats

        # Verify shapes
        assert stats["frame_shape"] == (3, self.frame_height, self.frame_width)
        assert stats["led_output_shape"] == (3, self.led_count)

    def test_add_led_pattern_validation(self):
        """Test validation in add_led_pattern."""
        self.manager.start_pattern_generation()

        # Test invalid LED ID
        pattern = self._create_synthetic_pattern(0)
        with pytest.raises(AssertionError):
            self.manager.add_led_pattern(-1, pattern)

        with pytest.raises(AssertionError):
            self.manager.add_led_pattern(self.led_count, pattern)

        # Test invalid pattern shape
        wrong_shape_pattern = np.random.rand(self.frame_height, self.frame_width, 3)  # Wrong format
        with pytest.raises(AssertionError):
            self.manager.add_led_pattern(0, wrong_shape_pattern)

    def test_sparse_matrix_creation(self):
        """Test sparse matrix creation."""
        self._generate_complete_patterns()

        sparse_matrices = self.manager.get_sparse_matrices()

        # Check matrix names
        expected_names = ["A_r", "A_g", "A_b", "A_combined"]
        assert set(sparse_matrices.keys()) == set(expected_names)

        # Check individual channel matrices
        pixels = self.frame_height * self.frame_width
        for channel in ["A_r", "A_g", "A_b"]:
            matrix = sparse_matrices[channel]
            assert matrix.shape == (pixels, self.led_count)
            assert sp.issparse(matrix)

        # Check combined matrix (block diagonal)
        combined = sparse_matrices["A_combined"]
        assert combined.shape == (pixels * 3, self.led_count * 3)
        assert sp.issparse(combined)

    def test_dense_ata_computation(self):
        """Test dense A^T @ A computation."""
        self._generate_complete_patterns()

        dense_ata = self.manager.get_dense_ata()

        # Check shape
        assert dense_ata.shape == (self.led_count, self.led_count, 3)
        assert dense_ata.dtype == np.float32

        # Check symmetry (A^T @ A should be symmetric)
        for channel in range(3):
            ata_channel = dense_ata[:, :, channel]
            np.testing.assert_allclose(ata_channel, ata_channel.T, rtol=1e-5)

        # Check positive semi-definite (diagonal should be non-negative)
        for channel in range(3):
            ata_channel = dense_ata[:, :, channel]
            assert np.all(np.diag(ata_channel) >= 0)

    def test_save_and_load_patterns(self):
        """Test saving and loading patterns."""
        # Generate patterns
        self._generate_complete_patterns()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Save patterns
            self.manager.save_patterns(tmp_path, include_diffusion_tensor=True)
            assert tmp_path.exists()

            # Create new manager and load
            new_manager = DiffusionPatternManager(
                led_count=self.led_count,
                frame_height=self.frame_height,
                frame_width=self.frame_width,
            )
            load_stats = new_manager.load_patterns(tmp_path)

            # Verify loading
            assert new_manager.is_loaded
            assert load_stats["format"] == "planar_2.0"
            assert load_stats["led_count"] == self.led_count
            assert load_stats["has_dense_ata"]
            assert load_stats["has_diffusion_tensor"]

            # Compare matrices
            original_sparse = self.manager.get_sparse_matrices()
            loaded_sparse = new_manager.get_sparse_matrices()

            for name in original_sparse:
                original_matrix = original_sparse[name]
                loaded_matrix = loaded_sparse[name]
                assert original_matrix.shape == loaded_matrix.shape
                np.testing.assert_allclose(original_matrix.toarray(), loaded_matrix.toarray(), rtol=1e-6)

            # Compare dense A^T @ A
            original_ata = self.manager.get_dense_ata()
            loaded_ata = new_manager.get_dense_ata()
            np.testing.assert_allclose(original_ata, loaded_ata, rtol=1e-6)

        finally:
            # Clean up
            if tmp_path.exists():
                tmp_path.unlink()

    def test_error_conditions(self):
        """Test error conditions."""
        # Test accessing data before generation/loading
        with pytest.raises(ValueError):
            self.manager.get_sparse_matrices()

        with pytest.raises(ValueError):
            self.manager.get_dense_ata()

        with pytest.raises(ValueError):
            self.manager.get_led_positions()

        # Test finalizing without patterns
        self.manager.start_pattern_generation()
        with pytest.raises(ValueError):
            self.manager.finalize_pattern_generation()

        # Test saving without data
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp_file, pytest.raises(ValueError):
            self.manager.save_patterns(tmp_file.name)

        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            self.manager.load_patterns("non_existent_file.npz")

    def test_different_sparse_formats(self):
        """Test different sparse matrix formats."""
        self._generate_patterns_for_format_test()

        for sparse_format in ["csc", "csr", "coo"]:
            stats = self.manager.finalize_pattern_generation(sparse_format=sparse_format)
            assert stats["sparse_stats"]["format"] == sparse_format

            sparse_matrices = self.manager.get_sparse_matrices()
            for matrix in sparse_matrices.values():
                if sparse_format == "csc":
                    assert sp.isspmatrix_csc(matrix)
                elif sparse_format == "csr":
                    assert sp.isspmatrix_csr(matrix)
                elif sparse_format == "coo":
                    assert sp.isspmatrix_coo(matrix)

            # Reset for next test
            self.manager = DiffusionPatternManager(
                led_count=self.led_count,
                frame_height=self.frame_height,
                frame_width=self.frame_width,
            )
            self._generate_patterns_for_format_test()

    def test_led_positions(self):
        """Test LED position handling."""
        # Test with custom positions
        custom_positions = np.random.rand(self.led_count, 2) * [
            self.frame_width,
            self.frame_height,
        ]
        self.manager.start_pattern_generation(custom_positions)

        positions = self.manager.get_led_positions()
        np.testing.assert_allclose(positions, custom_positions)

        # Test with default positions
        new_manager = DiffusionPatternManager(
            led_count=self.led_count,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
        )
        new_manager.start_pattern_generation()  # No positions provided

        default_positions = new_manager.get_led_positions()
        assert default_positions.shape == (self.led_count, 2)
        assert np.all(default_positions[:, 0] >= 0) and np.all(default_positions[:, 0] <= self.frame_width)
        assert np.all(default_positions[:, 1] >= 0) and np.all(default_positions[:, 1] <= self.frame_height)

    def _create_synthetic_pattern(self, led_id: int) -> np.ndarray:
        """Create synthetic diffusion pattern for testing."""
        # Create pattern in planar format (3, height, width)
        pattern = np.zeros((3, self.frame_height, self.frame_width), dtype=np.float32)

        # Create a simple Gaussian-like pattern centered at different positions for each LED
        center_y = (led_id % 3 + 1) * self.frame_height // 4
        center_x = (led_id % 4 + 1) * self.frame_width // 5

        for c in range(3):
            for y in range(self.frame_height):
                for x in range(self.frame_width):
                    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
                    pattern[c, y, x] = np.exp(-dist_sq / (2 * 3.0**2))  # Gaussian-like

        # Add some channel-specific variation
        pattern[0] *= 1.2  # Red slightly brighter
        pattern[1] *= 1.0  # Green normal
        pattern[2] *= 0.8  # Blue slightly dimmer

        # Normalize to [0, 1]
        pattern = np.clip(pattern, 0, 1)

        return pattern

    def _generate_complete_patterns(self):
        """Generate complete set of patterns for testing."""
        self.manager.start_pattern_generation()

        for led_id in range(self.led_count):
            pattern = self._create_synthetic_pattern(led_id)
            self.manager.add_led_pattern(led_id, pattern)

        self.manager.finalize_pattern_generation()

    def _generate_patterns_for_format_test(self):
        """Generate patterns for format testing (without finalizing)."""
        self.manager.start_pattern_generation()

        for led_id in range(self.led_count):
            pattern = self._create_synthetic_pattern(led_id)
            self.manager.add_led_pattern(led_id, pattern)


class TestDiffusionPatternManagerIntegration:
    """Integration tests for DiffusionPatternManager."""

    def test_realistic_led_count(self):
        """Test with realistic LED count."""
        led_count = 100
        frame_height = 64
        frame_width = 80

        manager = DiffusionPatternManager(led_count=led_count, frame_height=frame_height, frame_width=frame_width)

        manager.start_pattern_generation()

        # Add patterns for subset of LEDs (realistic scenario)
        test_leds = [0, 10, 25, 50, 75, 99]
        for led_id in test_leds:
            # Create more realistic pattern
            pattern = np.random.exponential(0.1, (3, frame_height, frame_width)).astype(np.float32)
            pattern = np.clip(pattern, 0, 1)  # Ensure valid range
            manager.add_led_pattern(led_id, pattern)

        stats = manager.finalize_pattern_generation()

        # Verify generation worked with partial patterns
        assert stats["patterns_processed"] == len(test_leds)
        assert manager.is_generation_complete

        # Check that sparse matrices have expected sparsity
        sparse_matrices = manager.get_sparse_matrices()
        combined = sparse_matrices["A_combined"]
        density = combined.nnz / (combined.shape[0] * combined.shape[1])
        assert density < 0.1  # Should be quite sparse

        # Check dense A^T @ A is reasonable size
        dense_ata = manager.get_dense_ata()
        ata_memory_mb = dense_ata.nbytes / (1024**2)
        assert ata_memory_mb < 10  # Should be reasonable for 100 LEDs

    def test_memory_efficiency(self):
        """Test memory efficiency of the implementation."""
        # Use moderate size for memory test
        led_count = 50
        frame_height = 32
        frame_width = 40

        manager = DiffusionPatternManager(led_count, frame_height, frame_width)
        manager.start_pattern_generation()

        # Add patterns
        for led_id in range(led_count):
            # Create sparse-ish pattern (more sparse than before)
            pattern = np.random.exponential(0.01, (3, frame_height, frame_width)).astype(np.float32)
            # Apply threshold to make it very sparse
            pattern = np.where(pattern > 0.1, pattern, 0)
            pattern = np.clip(pattern, 0, 1)
            manager.add_led_pattern(led_id, pattern)

        stats = manager.finalize_pattern_generation()

        # Check memory usage is reasonable
        sparse_stats = stats["sparse_stats"]
        ata_stats = stats["ata_stats"]

        assert sparse_stats["total_size_mb"] < 50  # Sparse should be efficient
        assert ata_stats["ata_memory_mb"] < 10  # Dense A^T @ A should be manageable

        # Check sparsity
        assert sparse_stats["combined_density"] < 0.2  # Should be sparse
