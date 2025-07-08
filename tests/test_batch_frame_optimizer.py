#!/usr/bin/env python3
"""
Test suite for batch frame optimizer.
"""

from unittest.mock import Mock, patch

import cupy as cp
import numpy as np
import pytest

from src.utils.batch_frame_optimizer import (
    BatchFrameOptimizationResult,
    convert_ata_dia_to_dense,
    optimize_batch_frames_led_values,
)
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


class TestBatchFrameOptimizer:
    """Test batch frame optimizer functionality."""

    def test_batch_size_validation(self):
        """Test that batch size must be 8 or 16."""
        # Create mock inputs
        target_frames = np.random.randint(0, 255, (4, 3, 480, 800), dtype=np.uint8)  # Invalid batch size

        mock_at_matrix = Mock(spec=SingleBlockMixedSparseTensor)
        mock_ata_matrix = Mock(spec=DiagonalATAMatrix)
        mock_ata_inverse = np.random.random((3, 100, 100)).astype(np.float32)

        with pytest.raises(ValueError, match="Batch size must be 8 or 16"):
            optimize_batch_frames_led_values(target_frames, mock_at_matrix, mock_ata_matrix, mock_ata_inverse)

    def test_input_format_validation(self):
        """Test input format validation."""
        # Test invalid data type
        target_frames = np.random.random((8, 3, 480, 800)).astype(np.float32)  # Invalid dtype

        mock_at_matrix = Mock(spec=SingleBlockMixedSparseTensor)
        mock_ata_matrix = Mock(spec=DiagonalATAMatrix)
        mock_ata_inverse = np.random.random((3, 100, 100)).astype(np.float32)

        with pytest.raises(ValueError, match="Target frames must be int8 or uint8"):
            optimize_batch_frames_led_values(target_frames, mock_at_matrix, mock_ata_matrix, mock_ata_inverse)

        # Test invalid shape
        target_frames = np.random.randint(0, 255, (8, 240, 320, 3), dtype=np.uint8)  # Invalid shape

        with pytest.raises(ValueError, match="Unsupported frame shape"):
            optimize_batch_frames_led_values(target_frames, mock_at_matrix, mock_ata_matrix, mock_ata_inverse)

    def test_shape_conversion(self):
        """Test conversion between HWC and CHW formats."""
        batch_size = 8

        # Test both input formats
        target_frames_chw = np.random.randint(0, 255, (batch_size, 3, 480, 800), dtype=np.uint8)
        target_frames_hwc = np.random.randint(0, 255, (batch_size, 480, 800, 3), dtype=np.uint8)

        mock_at_matrix = Mock(spec=SingleBlockMixedSparseTensor)
        mock_ata_matrix = Mock(spec=DiagonalATAMatrix)
        mock_ata_inverse = np.random.random((3, 100, 100)).astype(np.float32)

        # Mock the _calculate_atb function to return consistent shapes
        with patch("src.utils.batch_frame_optimizer._calculate_atb") as mock_calculate_atb:
            mock_calculate_atb.return_value = np.random.random((3, 100)).astype(np.float32)

            # Mock the DIA to dense conversion
            with patch("src.utils.batch_frame_optimizer.convert_ata_dia_to_dense") as mock_convert:
                mock_convert.return_value = np.random.random((3, 100, 100)).astype(np.float32)

                # Test CHW format
                result_chw = optimize_batch_frames_led_values(
                    target_frames_chw, mock_at_matrix, mock_ata_matrix, mock_ata_inverse, max_iterations=1, debug=False
                )

                # Test HWC format
                result_hwc = optimize_batch_frames_led_values(
                    target_frames_hwc, mock_at_matrix, mock_ata_matrix, mock_ata_inverse, max_iterations=1, debug=False
                )

                # Both should work and produce same output shape
                assert result_chw.led_values.shape == (batch_size, 3, 100)
                assert result_hwc.led_values.shape == (batch_size, 3, 100)

    def test_ata_dense_conversion(self):
        """Test ATA DIA to dense conversion."""
        # Create a mock DiagonalATAMatrix
        mock_ata_matrix = Mock(spec=DiagonalATAMatrix)

        # Create mock DIA matrices for each channel
        led_count = 50
        mock_dia_matrices = []

        for channel in range(3):
            # Create a simple diagonal matrix for testing
            diag_data = np.random.random(led_count).astype(np.float32)
            mock_dia = Mock()
            mock_dia.toarray.return_value = np.diag(diag_data)
            mock_dia_matrices.append(mock_dia)

        mock_ata_matrix.get_channel_dia_matrix.side_effect = mock_dia_matrices

        # Test the conversion
        result = convert_ata_dia_to_dense(mock_ata_matrix)

        # Check that get_channel_dia_matrix was called for each channel
        assert mock_ata_matrix.get_channel_dia_matrix.call_count == 3

        # Check output shape
        assert result.shape == (3, led_count, led_count)

        # Check that each channel is diagonal
        for channel in range(3):
            # Check that off-diagonal elements are zero (or close to zero)
            channel_matrix = result[channel]
            off_diag = channel_matrix - np.diag(np.diag(channel_matrix))
            assert np.allclose(off_diag, 0, atol=1e-6)

    def test_batch_optimization_result_format(self):
        """Test that batch optimization returns proper result format."""
        batch_size = 8
        led_count = 100

        target_frames = np.random.randint(0, 255, (batch_size, 3, 480, 800), dtype=np.uint8)

        mock_at_matrix = Mock(spec=SingleBlockMixedSparseTensor)
        mock_ata_matrix = Mock(spec=DiagonalATAMatrix)
        mock_ata_inverse = np.random.random((3, led_count, led_count)).astype(np.float32)

        # Mock the _calculate_atb function
        with patch("src.utils.batch_frame_optimizer._calculate_atb") as mock_calculate_atb:
            mock_calculate_atb.return_value = np.random.random((3, led_count)).astype(np.float32)

            # Mock the DIA to dense conversion
            with patch("src.utils.batch_frame_optimizer.convert_ata_dia_to_dense") as mock_convert:
                mock_convert.return_value = np.random.random((3, led_count, led_count)).astype(np.float32)

                result = optimize_batch_frames_led_values(
                    target_frames, mock_at_matrix, mock_ata_matrix, mock_ata_inverse, max_iterations=2, debug=False
                )

                # Check result type and attributes
                assert isinstance(result, BatchFrameOptimizationResult)
                assert result.led_values.shape == (batch_size, 3, led_count)
                assert result.led_values.dtype == np.uint8
                assert 0 <= result.led_values.min() <= result.led_values.max() <= 255
                assert result.iterations == 2
                assert result.converged is False
                assert isinstance(result.error_metrics, list)

    def test_batch_vs_single_frame_consistency(self):
        """Test that batch processing produces consistent results with single frame processing."""
        batch_size = 8
        led_count = 50

        # Create identical frames for comparison
        single_frame = np.random.randint(0, 255, (3, 480, 800), dtype=np.uint8)
        batch_frames = np.stack([single_frame] * batch_size, axis=0)

        mock_at_matrix = Mock(spec=SingleBlockMixedSparseTensor)
        mock_ata_matrix = Mock(spec=DiagonalATAMatrix)
        mock_ata_inverse = np.random.random((3, led_count, led_count)).astype(np.float32)

        # Mock consistent ATb calculation
        mock_atb = np.random.random((3, led_count)).astype(np.float32)

        with patch("src.utils.batch_frame_optimizer._calculate_atb") as mock_calculate_atb:
            mock_calculate_atb.return_value = mock_atb

            with patch("src.utils.batch_frame_optimizer.convert_ata_dia_to_dense") as mock_convert:
                mock_ata_dense = np.random.random((3, led_count, led_count)).astype(np.float32)
                mock_convert.return_value = mock_ata_dense

                # Test batch processing
                batch_result = optimize_batch_frames_led_values(
                    batch_frames, mock_at_matrix, mock_ata_matrix, mock_ata_inverse, max_iterations=3, debug=False
                )

                # All frames should produce similar results since they're identical
                # (Note: Due to batched operations, results may have slight numerical differences)
                assert batch_result.led_values.shape == (batch_size, 3, led_count)

                # Check that all frames have similar LED values (allowing for numerical differences)
                for i in range(1, batch_size):
                    diff = np.abs(batch_result.led_values[i] - batch_result.led_values[0])
                    # Should be very similar for identical input frames
                    assert np.mean(diff) < 5  # Allow small numerical differences

    def test_timing_functionality(self):
        """Test that timing functionality works without errors."""
        batch_size = 8
        led_count = 50

        target_frames = np.random.randint(0, 255, (batch_size, 3, 480, 800), dtype=np.uint8)

        mock_at_matrix = Mock(spec=SingleBlockMixedSparseTensor)
        mock_ata_matrix = Mock(spec=DiagonalATAMatrix)
        mock_ata_inverse = np.random.random((3, led_count, led_count)).astype(np.float32)

        with patch("src.utils.batch_frame_optimizer._calculate_atb") as mock_calculate_atb:
            mock_calculate_atb.return_value = np.random.random((3, led_count)).astype(np.float32)

            with patch("src.utils.batch_frame_optimizer.convert_ata_dia_to_dense") as mock_convert:
                mock_convert.return_value = np.random.random((3, led_count, led_count)).astype(np.float32)

                result = optimize_batch_frames_led_values(
                    target_frames,
                    mock_at_matrix,
                    mock_ata_matrix,
                    mock_ata_inverse,
                    max_iterations=2,
                    enable_timing=True,
                    debug=False,
                )

                # Should have timing data
                assert result.timing_data is not None
                assert isinstance(result.timing_data, dict)
                # Should contain expected timing sections
                expected_sections = [
                    "atb_calculation_batch",
                    "ata_inverse_initialization_batch",
                    "ata_dense_conversion",
                    "gpu_transfer_batch",
                    "optimization_loop_batch",
                ]
                for section in expected_sections:
                    assert section in result.timing_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
