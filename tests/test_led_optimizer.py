#!/usr/bin/env python3
"""
Unit tests for LEDOptimizer class.

Tests the LEDOptimizer wrapper around the standardized frame optimizer,
including initialization, API compatibility, and integration with
mixed tensor and DIA matrix formats.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from src.consumer.led_optimizer import LEDOptimizer, OptimizationResult


class TestLEDOptimizer:
    """Test suite for LEDOptimizer class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_patterns_path = str(Path(self.temp_dir) / "test_patterns.npz")

        # Create mock pattern data
        mock_mixed_tensor_dict = {"batch_size": 100, "channels": 3, "height": 480, "width": 800}
        mock_dia_matrix_dict = {"led_count": 100, "k": 50, "bandwidth": 25}

        self.mock_pattern_data = {
            "mixed_tensor": Mock(),
            "dia_matrix": Mock(),
            "ata_inverse": np.random.rand(3, 100, 100).astype(np.float32),
            "metadata": {"led_count": 100},
        }
        self.mock_pattern_data["mixed_tensor"].item.return_value = mock_mixed_tensor_dict
        self.mock_pattern_data["dia_matrix"].item.return_value = mock_dia_matrix_dict

    def tearDown(self):
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_default_parameters(self):
        """Test LEDOptimizer initialization with default parameters."""
        optimizer = LEDOptimizer()

        assert optimizer.diffusion_patterns_path == "diffusion_patterns/synthetic_1000"
        assert optimizer.max_iterations == 10
        assert optimizer.convergence_threshold == 1e-3
        assert optimizer.step_size_scaling == 0.9
        assert optimizer.timing is not None  # Performance timing enabled by default

    def test_initialization_custom_parameters(self):
        """Test LEDOptimizer initialization with custom parameters."""
        custom_path = "custom/patterns"
        optimizer = LEDOptimizer(
            diffusion_patterns_path=custom_path,
            use_mixed_tensor=True,  # Should be ignored (deprecated)
            enable_performance_timing=False,
        )

        assert optimizer.diffusion_patterns_path == custom_path
        assert optimizer.timing is None  # Performance timing disabled

    @patch("src.consumer.led_optimizer.Path.exists")
    @patch("numpy.load")
    def test_initialization_failure_missing_patterns(self, mock_load, mock_exists):
        """Test initialization failure when pattern files are missing."""
        mock_exists.return_value = False

        optimizer = LEDOptimizer()
        result = optimizer.initialize()

        assert not result
        mock_load.assert_not_called()

    @patch("src.consumer.led_optimizer.Path.exists")
    @patch("numpy.load")
    @patch("src.consumer.led_optimizer.SingleBlockMixedSparseTensor.from_dict")
    @patch("src.consumer.led_optimizer.DiagonalATAMatrix.from_dict")
    def test_initialization_success(self, mock_dia_from_dict, mock_tensor_from_dict, mock_load, mock_exists):
        """Test successful initialization with mock pattern data."""
        mock_exists.return_value = True
        mock_load.return_value = self.mock_pattern_data

        # Mock the tensor and matrix objects
        mock_tensor = Mock()
        mock_tensor.batch_size = 100
        mock_tensor_from_dict.return_value = mock_tensor

        mock_dia_matrix = Mock()
        mock_dia_matrix.led_count = 100
        mock_dia_matrix.bandwidth = 25
        mock_dia_matrix.k = 50
        mock_dia_matrix.dia_data_cpu = np.random.rand(3, 50, 100).astype(np.float32)
        mock_dia_from_dict.return_value = mock_dia_matrix

        optimizer = LEDOptimizer()
        result = optimizer.initialize()

        assert result
        assert optimizer._matrix_loaded
        assert optimizer._actual_led_count == 100

    def test_optimize_frame_api_integration(self):
        """Test that optimize_frame integrates correctly with mocked frame optimizer."""
        # Create optimizer
        optimizer = LEDOptimizer()

        # Mock the entire optimize_frame method to test API integration
        with patch.object(optimizer, "optimize_frame") as mock_optimize:
            mock_result = OptimizationResult(
                led_values=np.random.randint(0, 255, (100, 3), dtype=np.uint8),
                error_metrics={"mse": 0.01},
                iterations=5,
                converged=True,
            )
            mock_optimize.return_value = mock_result

            # Test frame
            target_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

            # Call optimize_frame
            result = optimizer.optimize_frame(target_frame, debug=True)

            # Verify method was called
            mock_optimize.assert_called_once()

            # Verify result
            assert isinstance(result, OptimizationResult)
            assert result.iterations == 5
            assert result.converged

    def test_optimize_frame_invalid_input_shape(self):
        """Test optimize_frame with invalid input frame shape."""
        optimizer = LEDOptimizer()
        optimizer._matrix_loaded = True

        # Wrong frame shape
        wrong_frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

        result = optimizer.optimize_frame(wrong_frame)

        # Should return error result
        assert isinstance(result, OptimizationResult)
        assert result.iterations == 0
        assert not result.converged
        # Error result may have empty error_metrics or may have inf values
        assert len(result.error_metrics) == 0 or result.error_metrics.get("mse") == float("inf")

    def test_optimize_frame_matrix_not_loaded(self):
        """Test optimize_frame when matrices are not loaded."""
        optimizer = LEDOptimizer()
        optimizer._matrix_loaded = False

        target_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        result = optimizer.optimize_frame(target_frame)

        # Should return error result
        assert isinstance(result, OptimizationResult)
        assert result.iterations == 0
        assert not result.converged

    def test_get_optimizer_stats(self):
        """Test getting optimizer statistics."""
        optimizer = LEDOptimizer()
        optimizer._matrix_loaded = True
        optimizer._actual_led_count = 100
        optimizer._optimization_count = 5
        optimizer._diagonal_ata_matrix = Mock()
        optimizer._diagonal_ata_matrix.bandwidth = 25
        optimizer._diagonal_ata_matrix.k = 50
        optimizer._diagonal_ata_matrix.dia_data_cpu = Mock()
        optimizer._diagonal_ata_matrix.dia_data_cpu.nbytes = 1024 * 1024  # 1MB

        stats = optimizer.get_optimizer_stats()

        assert stats["optimizer_type"] == "standardized_frame_optimizer"
        assert stats["led_count"] == 100
        assert stats["optimization_count"] == 5
        assert stats["matrix_loaded"]
        assert stats["ata_format"] == "DIA_sparse"
        assert stats["ata_bandwidth"] == 25
        assert stats["ata_k_bands"] == 50
        assert stats["approach_description"] == "Standardized frame optimizer with mixed tensor and DIA matrix"

    def test_deprecated_methods_warning(self):
        """Test that deprecated methods issue warnings."""
        optimizer = LEDOptimizer()

        # Mock the mixed tensor method to avoid AttributeError
        with patch.object(optimizer, "_calculate_atb_mixed_tensor", return_value=Mock()), patch(
            "src.consumer.led_optimizer.logger"
        ) as mock_logger:
            target_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

            # Call deprecated method
            optimizer._calculate_atb(target_frame)

            # Should have logged a warning
            mock_logger.warning.assert_called_with(
                "_calculate_atb is deprecated - use standardized frame optimizer API"
            )

    def test_initial_values_format_conversion(self):
        """Test that initial values are correctly converted between formats."""
        optimizer = LEDOptimizer()
        optimizer._matrix_loaded = True
        optimizer._actual_led_count = 100
        optimizer._mixed_tensor = Mock()
        optimizer._diagonal_ata_matrix = Mock()
        optimizer._ATA_inverse_cpu = np.random.rand(3, 100, 100).astype(np.float32)
        optimizer._has_ata_inverse = True

        with patch("src.consumer.led_optimizer.optimize_frame_led_values") as mock_optimize:
            mock_result = Mock()
            mock_result.led_values = np.random.randint(0, 255, (3, 100), dtype=np.uint8)
            mock_result.error_metrics = {}
            mock_result.iterations = 5
            mock_result.converged = True
            mock_result.timing_data = None
            mock_optimize.return_value = mock_result

            target_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

            # Test with (led_count, 3) format
            initial_values_led_3 = np.random.rand(100, 3).astype(np.float32)
            optimizer.optimize_frame(target_frame, initial_values=initial_values_led_3)

            # Should convert to (3, led_count) format for frame optimizer
            call_kwargs = mock_optimize.call_args[1]
            assert call_kwargs["initial_values"].shape == (3, 100)

            # Test with (3, led_count) format
            initial_values_3_led = np.random.rand(3, 100).astype(np.float32)
            optimizer.optimize_frame(target_frame, initial_values=initial_values_3_led)

            # Should keep (3, led_count) format
            call_kwargs = mock_optimize.call_args[1]
            assert call_kwargs["initial_values"].shape == (3, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
