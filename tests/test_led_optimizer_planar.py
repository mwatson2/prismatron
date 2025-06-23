"""
Unit tests for PlanarLEDOptimizer.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.consumer.led_optimizer_planar import (
    PlanarLEDOptimizer,
    PlanarOptimizationResult,
)
from src.utils.diffusion_pattern_manager import DiffusionPatternManager


class TestPlanarLEDOptimizer:
    """Test cases for PlanarLEDOptimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.led_count = 20
        self.frame_height = 16
        self.frame_width = 24

        # Create pattern manager with test data
        self.pattern_manager = DiffusionPatternManager(
            led_count=self.led_count,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
        )

        # Generate synthetic patterns
        self.pattern_manager.start_pattern_generation()

        for led_id in range(self.led_count):
            pattern = self._create_synthetic_pattern(led_id)
            self.pattern_manager.add_led_pattern(led_id, pattern)

        self.pattern_manager.finalize_pattern_generation()

        # Create optimizer
        self.optimizer = PlanarLEDOptimizer(pattern_manager=self.pattern_manager)

    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.led_count == self.led_count
        assert self.optimizer.frame_shape == (3, self.frame_height, self.frame_width)
        assert self.optimizer.led_output_shape == (3, self.led_count)

        # Initialize optimizer
        success = self.optimizer.initialize()
        assert success

        # Check internal state
        assert self.optimizer._is_initialized()

    def test_frame_optimization(self):
        """Test frame optimization with planar format."""
        self.optimizer.initialize()

        # Create test frame in planar format
        target_frame = self._create_test_frame()

        # Optimize frame
        result = self.optimizer.optimize_frame(target_frame, debug=True)

        # Verify result
        assert isinstance(result, PlanarOptimizationResult)
        assert result.led_values.shape == (3, self.led_count)
        assert result.led_values.dtype == np.uint8
        assert np.all(result.led_values >= 0) and np.all(result.led_values <= 255)

        # Check that result contains reasonable values
        assert np.any(result.led_values > 0)  # Should have some non-zero values

        # Check timing information
        assert result.optimization_time > 0
        assert result.iterations > 0
        assert "atb_calculation_time" in result.timing_breakdown
        assert "optimization_loop_time" in result.timing_breakdown

    def test_input_validation(self):
        """Test input validation."""
        self.optimizer.initialize()

        # Test wrong frame shape
        wrong_shape_frame = np.random.rand(
            self.frame_height, self.frame_width, 3
        )  # Wrong format
        with pytest.raises(ValueError):
            self.optimizer.optimize_frame(wrong_shape_frame)

        # Test wrong dimensions
        wrong_size_frame = np.random.rand(3, 10, 10)  # Wrong size
        with pytest.raises(ValueError):
            self.optimizer.optimize_frame(wrong_size_frame)

        # Test optimization without initialization
        uninit_optimizer = PlanarLEDOptimizer(pattern_manager=self.pattern_manager)
        target_frame = self._create_test_frame()
        with pytest.raises(RuntimeError):
            uninit_optimizer.optimize_frame(target_frame)

    def test_multiple_optimizations(self):
        """Test multiple frame optimizations."""
        self.optimizer.initialize()

        results = []
        for i in range(3):
            target_frame = self._create_test_frame(variation=i)
            result = self.optimizer.optimize_frame(target_frame)
            results.append(result)

        # Check that all optimizations completed successfully
        for result in results:
            assert result.converged
            assert result.optimization_time > 0

        # Check statistics
        stats = self.optimizer.get_statistics()
        assert stats["optimization_count"] == 3
        assert stats["average_optimization_time"] > 0
        assert stats["approach"] == "planar_dense_ata"

    def test_different_iteration_counts(self):
        """Test with different iteration counts."""
        self.optimizer.initialize()
        target_frame = self._create_test_frame()

        for max_iter in [1, 5, 15]:
            result = self.optimizer.optimize_frame(
                target_frame, max_iterations=max_iter
            )
            assert result.iterations == max_iter

    def test_debug_mode(self):
        """Test debug mode functionality."""
        self.optimizer.initialize()
        target_frame = self._create_test_frame()

        # Test with debug enabled
        result_debug = self.optimizer.optimize_frame(target_frame, debug=True)
        assert result_debug.target_frame is not None
        assert result_debug.pattern_info is not None
        assert "debug_time" in result_debug.timing_breakdown
        assert len(result_debug.error_metrics) > 0

        # Test with debug disabled
        result_no_debug = self.optimizer.optimize_frame(target_frame, debug=False)
        assert result_no_debug.target_frame is None
        assert result_no_debug.pattern_info is None
        assert "debug_time" not in result_no_debug.timing_breakdown
        assert len(result_no_debug.error_metrics) == 0

    def test_output_format_consistency(self):
        """Test that output format is consistent with expectations."""
        self.optimizer.initialize()
        target_frame = self._create_test_frame()

        result = self.optimizer.optimize_frame(target_frame)

        # Check output format
        led_values = result.led_values
        assert led_values.shape == (3, self.led_count)  # Planar format

        # Check value ranges
        assert np.all(led_values >= 0)
        assert np.all(led_values <= 255)

        # Check data type
        assert led_values.dtype == np.uint8

    def test_pattern_manager_integration(self):
        """Test integration with DiffusionPatternManager."""
        # Create optimizer from pattern file path (once we have the conversion working)
        # For now, test that the optimizer correctly uses pattern manager data

        # Check that pattern manager data is correctly loaded
        assert self.optimizer.pattern_manager.is_generation_complete

        sparse_matrices = self.optimizer.pattern_manager.get_sparse_matrices()
        assert len(sparse_matrices) > 0

        dense_ata = self.optimizer.pattern_manager.get_dense_ata()
        assert dense_ata.shape == (self.led_count, self.led_count, 3)

        # Check that optimizer uses the data correctly
        self.optimizer.initialize()
        assert self.optimizer._ATA_gpu is not None
        assert len(self.optimizer._sparse_matrices_gpu) > 0

    def _create_synthetic_pattern(self, led_id: int) -> np.ndarray:
        """Create synthetic diffusion pattern for testing."""
        # Create pattern in planar format (3, height, width)
        pattern = np.zeros((3, self.frame_height, self.frame_width), dtype=np.float32)

        # Create a simple pattern centered at different positions for each LED
        center_y = (led_id % 4 + 1) * self.frame_height // 5
        center_x = (led_id % 6 + 1) * self.frame_width // 7

        for c in range(3):
            for y in range(self.frame_height):
                for x in range(self.frame_width):
                    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
                    pattern[c, y, x] = np.exp(
                        -dist_sq / (2 * 2.0**2)
                    )  # Gaussian-like

        # Add some channel variation
        pattern[0] *= 1.1  # Red slightly brighter
        pattern[1] *= 1.0  # Green normal
        pattern[2] *= 0.9  # Blue slightly dimmer

        # Normalize and make sparse
        pattern = np.clip(pattern, 0, 1)
        pattern = np.where(pattern > 0.1, pattern, 0)  # Apply threshold for sparsity

        return pattern

    def _create_test_frame(self, variation: int = 0) -> np.ndarray:
        """Create test frame in planar format."""
        # Create frame in planar format (3, height, width)
        frame = np.random.rand(3, self.frame_height, self.frame_width).astype(
            np.float32
        )

        # Add some structured content
        center_y = self.frame_height // 2 + variation * 2
        center_x = self.frame_width // 2 + variation * 3

        for c in range(3):
            for y in range(self.frame_height):
                for x in range(self.frame_width):
                    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
                    frame[c, y, x] += 0.5 * np.exp(-dist_sq / (2 * 5.0**2))

        # Normalize to [0, 1]
        frame = np.clip(frame, 0, 1)

        return frame


class TestPlanarLEDOptimizerIntegration:
    """Integration tests for PlanarLEDOptimizer."""

    def test_save_load_patterns_and_optimize(self):
        """Test saving patterns, loading them, and optimizing."""
        led_count = 30
        frame_height = 20
        frame_width = 32

        # Create and save patterns
        pattern_manager = DiffusionPatternManager(led_count, frame_height, frame_width)
        pattern_manager.start_pattern_generation()

        for led_id in range(led_count):
            # Create realistic sparse pattern
            pattern = np.random.exponential(
                0.02, (3, frame_height, frame_width)
            ).astype(np.float32)
            pattern = np.where(pattern > 0.1, pattern, 0)
            pattern = np.clip(pattern, 0, 1)
            pattern_manager.add_led_pattern(led_id, pattern)

        pattern_manager.finalize_pattern_generation()

        # Save patterns
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            pattern_manager.save_patterns(tmp_path)

            # Load patterns with new manager
            new_pattern_manager = DiffusionPatternManager(
                led_count, frame_height, frame_width
            )
            new_pattern_manager.load_patterns(tmp_path)

            # Create optimizer with loaded patterns
            optimizer = PlanarLEDOptimizer(pattern_manager=new_pattern_manager)
            optimizer.initialize()

            # Test optimization
            target_frame = np.random.rand(3, frame_height, frame_width).astype(
                np.float32
            )
            result = optimizer.optimize_frame(target_frame)

            # Verify optimization completed successfully
            assert result.converged
            assert result.led_values.shape == (3, led_count)
            assert np.all(result.led_values >= 0) and np.all(result.led_values <= 255)

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_realistic_performance(self):
        """Test with realistic size and measure performance."""
        led_count = 100
        frame_height = 32
        frame_width = 40

        # Create pattern manager
        pattern_manager = DiffusionPatternManager(led_count, frame_height, frame_width)
        pattern_manager.start_pattern_generation()

        # Add patterns for subset of LEDs (realistic scenario)
        test_leds = list(range(0, led_count, 5))  # Every 5th LED
        for led_id in test_leds:
            pattern = np.random.exponential(
                0.01, (3, frame_height, frame_width)
            ).astype(np.float32)
            pattern = np.where(pattern > 0.15, pattern, 0)
            pattern = np.clip(pattern, 0, 1)
            pattern_manager.add_led_pattern(led_id, pattern)

        pattern_manager.finalize_pattern_generation()

        # Create and initialize optimizer
        optimizer = PlanarLEDOptimizer(pattern_manager=pattern_manager)
        optimizer.initialize()

        # Test multiple optimizations
        frame_times = []
        for i in range(5):
            target_frame = np.random.rand(3, frame_height, frame_width).astype(
                np.float32
            )
            result = optimizer.optimize_frame(target_frame)
            frame_times.append(result.optimization_time)

            assert result.converged
            assert result.led_values.shape == (3, led_count)

        # Check performance is reasonable
        avg_time = np.mean(frame_times)
        assert avg_time < 1.0  # Should complete in less than 1 second

        # Check statistics
        stats = optimizer.get_statistics()
        assert stats["optimization_count"] == 5
        assert stats["led_count"] == led_count
