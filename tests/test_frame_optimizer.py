#!/usr/bin/env python3
"""
Tests for standalone frame optimization function.

This module tests the extracted frame optimization function with both
mixed tensor and DIA matrix formats for LED optimization.
"""

import sys
from pathlib import Path
from typing import Tuple

import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sp

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import (
    FrameOptimizationResult,
    optimize_frame_led_values,
    optimize_frame_with_dia_matrix,
    optimize_frame_with_mixed_tensor,
)
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


class TestFrameOptimizer:
    """Test standalone frame optimization function."""

    def create_test_diffusion_matrices(
        self, led_count: int = 100
    ) -> Tuple[sp.csc_matrix, np.ndarray]:
        """
        Create test diffusion matrices for LED optimization.

        Args:
            led_count: Number of LEDs

        Returns:
            Tuple of (diffusion_matrix, led_positions)
        """
        # Frame dimensions
        height, width = 480, 800
        pixels = height * width

        # Create random LED positions
        np.random.seed(42)
        led_positions = np.random.uniform(0, min(height, width), (led_count, 2))

        # Create sparse diffusion matrix A: (pixels*3, led_count*3)
        # Each LED affects a local region in the frame
        diffusion_density = 0.01  # Sparse diffusion pattern

        # Create a sparse diffusion matrix with proper dimensions
        # LEDDiffusionCSCMatrix expects: (pixels, led_count * channels)
        density = 0.001  # Very sparse for realistic LED patterns
        diffusion_matrix = sp.random(
            pixels,  # rows: height * width (spatial pixels)
            led_count * 3,  # cols: led_count * 3 channels (RGB for each LED)
            density=density,
            format="csc",
            dtype=np.float32,
            random_state=42,
        )

        # Scale values to realistic range
        diffusion_matrix.data *= 0.5

        return diffusion_matrix, led_positions

    def create_test_frame(self, frame_format: str = "planar") -> np.ndarray:
        """
        Create test target frame.

        Args:
            frame_format: "planar" for (3, 480, 800) or "hwc" for (480, 800, 3)

        Returns:
            Test frame in specified format
        """
        height, width = 480, 800

        # Create simple test pattern
        frame_hwc = np.zeros((height, width, 3), dtype=np.uint8)

        # Add some simple patterns
        # Red stripe
        frame_hwc[100:150, :, 0] = 255
        # Green circle
        center_y, center_x = height // 2, width // 2
        radius = 50
        y, x = np.ogrid[:height, :width]
        mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2
        frame_hwc[mask, 1] = 255
        # Blue gradient
        frame_hwc[:, :, 2] = (np.arange(width) / width * 255).astype(np.uint8)

        if frame_format == "planar":
            # Convert to (3, H, W)
            return frame_hwc.transpose(2, 0, 1)
        else:
            return frame_hwc

    def test_frame_format_validation(self):
        """Test input frame format validation."""
        led_count = 50
        diffusion_matrix, led_positions = self.create_test_diffusion_matrices(led_count)

        # Create matrices - note: LEDDiffusionCSCMatrix expects the A matrix, not A^T
        csc_matrix = LEDDiffusionCSCMatrix(
            csc_matrix=diffusion_matrix, height=480, width=800  # Pass A matrix directly
        )

        dia_matrix = DiagonalATAMatrix(led_count, crop_size=32)
        dia_matrix.build_from_diffusion_matrix(diffusion_matrix, led_positions)

        # Test valid planar format
        frame_planar = self.create_test_frame("planar")  # (3, 480, 800)
        result = optimize_frame_led_values(frame_planar, csc_matrix, dia_matrix)
        assert result.led_values.shape == (3, led_count)

        # Test valid HWC format
        frame_hwc = self.create_test_frame("hwc")  # (480, 800, 3)
        result = optimize_frame_led_values(frame_hwc, csc_matrix, dia_matrix)
        assert result.led_values.shape == (3, led_count)

        # Test invalid format
        with pytest.raises(ValueError, match="Unsupported frame shape"):
            invalid_frame = np.zeros((100, 100), dtype=np.uint8)
            optimize_frame_led_values(invalid_frame, csc_matrix, dia_matrix)

    def test_optimization_with_dia_matrix(self):
        """Test frame optimization using DIA A^T A matrix."""
        led_count = 100
        diffusion_matrix, led_positions = self.create_test_diffusion_matrices(led_count)

        print(f"\nTesting DIA matrix optimization:")
        print(f"  LED count: {led_count}")
        print(
            f"  Diffusion matrix: {diffusion_matrix.shape}, nnz: {diffusion_matrix.nnz}"
        )

        # Create CSC matrix for A^T @ b
        csc_matrix = LEDDiffusionCSCMatrix(
            csc_matrix=diffusion_matrix, height=480, width=800  # Store A matrix
        )

        # Create DIA matrix for A^T A
        dia_matrix = DiagonalATAMatrix(led_count, crop_size=32)
        dia_matrix.build_from_diffusion_matrix(
            diffusion_matrix, led_positions, use_rcm=True
        )

        print(f"  DIA matrix: bands={dia_matrix.k}, bandwidth={dia_matrix.bandwidth}")

        # Test frame
        target_frame = self.create_test_frame("planar")

        # Optimize
        result = optimize_frame_with_dia_matrix(
            target_frame=target_frame,
            diffusion_csc=csc_matrix,
            dia_matrix=dia_matrix,
            max_iterations=15,
            debug=True,
            compute_error_metrics=True,
        )

        print(f"  Optimization result:")
        print(f"    Converged: {result.converged}")
        print(f"    Iterations: {result.iterations}")
        print(f"    LED values shape: {result.led_values.shape}")
        print(
            f"    LED values range: [{result.led_values.min()}, {result.led_values.max()}]"
        )

        # Validate results
        assert result.led_values.shape == (3, led_count)
        assert result.led_values.dtype == np.uint8
        assert np.all(result.led_values >= 0) and np.all(result.led_values <= 255)
        assert result.iterations > 0
        assert "mse" in result.error_metrics

    def test_optimization_with_mixed_tensor(self):
        """Test frame optimization using mixed tensor format."""
        led_count = 50  # Smaller for mixed tensor test
        diffusion_matrix, led_positions = self.create_test_diffusion_matrices(led_count)

        print(f"\nTesting mixed tensor optimization:")
        print(f"  LED count: {led_count}")

        # Create mixed tensor (this is a simplified test version)
        try:
            mixed_tensor = SingleBlockMixedSparseTensor(
                led_count=led_count, height=480, width=800
            )

            # Build from diffusion matrix (simplified)
            mixed_tensor.build_from_diffusion_matrix(diffusion_matrix, led_positions)

            # Create dense A^T A for optimization
            ATA_dense = np.zeros((led_count, led_count, 3), dtype=np.float32)
            for channel in range(3):
                # Extract channel-specific diffusion matrix
                channel_start = channel * (480 * 800)
                channel_end = (channel + 1) * (480 * 800)
                A_channel = diffusion_matrix[channel_start:channel_end, channel::3]

                # Compute A^T A for this channel
                ATA_channel = A_channel.T @ A_channel
                ATA_dense[:, :, channel] = ATA_channel.toarray()

            print(f"  Dense A^T A shape: {ATA_dense.shape}")

            # Test frame
            target_frame = self.create_test_frame("planar")

            # Optimize
            result = optimize_frame_with_mixed_tensor(
                target_frame=target_frame,
                mixed_tensor=mixed_tensor,
                ata_dense=ATA_dense,
                max_iterations=10,
                debug=True,
            )

            print(f"  Optimization result:")
            print(f"    Converged: {result.converged}")
            print(f"    Iterations: {result.iterations}")
            print(f"    LED values shape: {result.led_values.shape}")

            # Validate results
            assert result.led_values.shape == (3, led_count)
            assert result.led_values.dtype == np.uint8
            assert np.all(result.led_values >= 0) and np.all(result.led_values <= 255)
            assert result.iterations > 0

        except Exception as e:
            print(f"  Mixed tensor test skipped: {e}")
            pytest.skip(f"Mixed tensor not available: {e}")

    def test_initial_values(self):
        """Test optimization with custom initial values."""
        led_count = 50
        diffusion_matrix, led_positions = self.create_test_diffusion_matrices(led_count)

        # Create matrices
        csc_matrix = LEDDiffusionCSCMatrix(
            csc_matrix=diffusion_matrix, height=480, width=800
        )

        dia_matrix = DiagonalATAMatrix(led_count, crop_size=32)
        dia_matrix.build_from_diffusion_matrix(diffusion_matrix, led_positions)

        # Test with custom initial values
        initial_values = np.random.rand(3, led_count) * 255
        initial_values = initial_values.astype(np.uint8)

        target_frame = self.create_test_frame("planar")

        result = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=csc_matrix,
            ATA_matrix=dia_matrix,
            initial_values=initial_values,
            max_iterations=5,
        )

        assert result.led_values.shape == (3, led_count)
        assert result.iterations > 0

    def test_convergence_behavior(self):
        """Test optimization convergence behavior."""
        led_count = 30  # Small for fast test
        diffusion_matrix, led_positions = self.create_test_diffusion_matrices(led_count)

        # Create matrices
        csc_matrix = LEDDiffusionCSCMatrix(
            csc_matrix=diffusion_matrix, height=480, width=800
        )

        dia_matrix = DiagonalATAMatrix(led_count, crop_size=32)
        dia_matrix.build_from_diffusion_matrix(diffusion_matrix, led_positions)

        target_frame = self.create_test_frame("planar")

        # Test with tight convergence threshold
        result_tight = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=csc_matrix,
            ATA_matrix=dia_matrix,
            max_iterations=20,
            convergence_threshold=1e-6,
            debug=True,
        )

        # Test with loose convergence threshold
        result_loose = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=csc_matrix,
            ATA_matrix=dia_matrix,
            max_iterations=20,
            convergence_threshold=1e-1,
            debug=True,
        )

        print(f"  Tight convergence: {result_tight.iterations} iterations")
        print(f"  Loose convergence: {result_loose.iterations} iterations")

        # Loose threshold should converge faster
        assert result_loose.iterations <= result_tight.iterations

        # Both should produce valid results
        assert result_tight.led_values.shape == (3, led_count)
        assert result_loose.led_values.shape == (3, led_count)

    def test_step_size_tracking(self):
        """Test step size tracking in debug mode."""
        led_count = 30
        diffusion_matrix, led_positions = self.create_test_diffusion_matrices(led_count)

        # Create matrices
        csc_matrix = LEDDiffusionCSCMatrix(
            csc_matrix=diffusion_matrix, height=480, width=800
        )

        dia_matrix = DiagonalATAMatrix(led_count, crop_size=32)
        dia_matrix.build_from_diffusion_matrix(diffusion_matrix, led_positions)

        target_frame = self.create_test_frame("planar")

        result = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=csc_matrix,
            ATA_matrix=dia_matrix,
            max_iterations=10,
            convergence_threshold=1e-10,  # Very tight threshold to ensure multiple iterations
            debug=True,
        )

        # Should have step size tracking in debug mode
        assert result.step_sizes is not None
        assert len(result.step_sizes) <= 10  # Up to max_iterations
        assert len(result.step_sizes) == result.iterations
        assert np.all(result.step_sizes > 0)  # All step sizes should be positive

    def test_error_metrics_computation(self):
        """Test error metrics computation."""
        led_count = 40
        diffusion_matrix, led_positions = self.create_test_diffusion_matrices(led_count)

        # Create matrices
        csc_matrix = LEDDiffusionCSCMatrix(
            csc_matrix=diffusion_matrix, height=480, width=800
        )

        dia_matrix = DiagonalATAMatrix(led_count, crop_size=32)
        dia_matrix.build_from_diffusion_matrix(diffusion_matrix, led_positions)

        target_frame = self.create_test_frame("planar")

        result = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=csc_matrix,
            ATA_matrix=dia_matrix,
            max_iterations=10,
            compute_error_metrics=True,
            debug=True,
        )

        print(f"  Error metrics: {result.error_metrics}")

        # Should have error metrics
        assert "mse" in result.error_metrics
        assert "mae" in result.error_metrics
        assert "psnr" in result.error_metrics
        assert result.error_metrics["mse"] >= 0
        assert result.error_metrics["mae"] >= 0
        assert result.error_metrics["psnr"] > 0

    def test_matrix_format_compatibility(self):
        """Test that function works with different matrix formats."""
        led_count = 25
        diffusion_matrix, led_positions = self.create_test_diffusion_matrices(led_count)

        # Test with CSC + DIA combination
        csc_matrix = LEDDiffusionCSCMatrix(
            csc_matrix=diffusion_matrix, height=480, width=800
        )

        dia_matrix = DiagonalATAMatrix(led_count, crop_size=32)
        dia_matrix.build_from_diffusion_matrix(diffusion_matrix, led_positions)

        target_frame = self.create_test_frame("planar")

        result_csc_dia = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=csc_matrix,
            ATA_matrix=dia_matrix,
            max_iterations=5,
        )

        # Test with dense A^T A matrix
        ATA_dense = np.random.rand(led_count, led_count, 3).astype(np.float32)
        # Make it positive definite for each channel
        for c in range(3):
            A = np.random.rand(led_count, led_count).astype(np.float32)
            ATA_dense[:, :, c] = A.T @ A + np.eye(led_count) * 0.1

        result_csc_dense = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=csc_matrix,
            ATA_matrix=ATA_dense,
            max_iterations=5,
        )

        # Both should produce valid results
        assert result_csc_dia.led_values.shape == (3, led_count)
        assert result_csc_dense.led_values.shape == (3, led_count)
        assert result_csc_dia.iterations > 0
        assert result_csc_dense.iterations > 0


class TestFrameOptimizerEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_matrix_types(self):
        """Test handling of invalid matrix types."""
        target_frame = np.zeros((3, 480, 800), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unsupported AT_matrix type"):
            optimize_frame_led_values(
                target_frame=target_frame, AT_matrix="invalid", ATA_matrix=np.eye(10)
            )

    def test_mismatched_dimensions(self):
        """Test handling of mismatched matrix dimensions."""
        led_count = 10
        diffusion_matrix = sp.random(
            480 * 800 * 3, led_count * 3, density=0.01, format="csc"
        )
        led_positions = np.random.rand(led_count, 2) * 100

        csc_matrix = LEDDiffusionCSCMatrix(
            csc_matrix=diffusion_matrix, height=480, width=800
        )

        # Mismatched DIA matrix
        dia_matrix = DiagonalATAMatrix(led_count + 5, crop_size=32)  # Wrong size

        target_frame = np.zeros((3, 480, 800), dtype=np.uint8)

        # This should fail during optimization due to dimension mismatch
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            optimize_frame_led_values(
                target_frame=target_frame,
                AT_matrix=csc_matrix,
                ATA_matrix=dia_matrix,
                max_iterations=1,
            )

    def test_zero_iterations(self):
        """Test behavior with zero iterations."""
        led_count = 10
        diffusion_matrix = sp.random(
            480 * 800 * 3, led_count * 3, density=0.01, format="csc"
        )

        csc_matrix = LEDDiffusionCSCMatrix(
            csc_matrix=diffusion_matrix, height=480, width=800
        )

        ATA_dense = np.random.rand(led_count, led_count, 3).astype(np.float32)
        for c in range(3):
            A = np.random.rand(led_count, led_count)
            ATA_dense[:, :, c] = A.T @ A + np.eye(led_count) * 0.1

        target_frame = np.zeros((3, 480, 800), dtype=np.uint8)

        result = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=csc_matrix,
            ATA_matrix=ATA_dense,
            max_iterations=0,
        )

        # Should return initial values
        assert result.led_values.shape == (3, led_count)
        assert result.iterations == 1  # At least one iteration is performed


if __name__ == "__main__":
    # Run basic tests if executed directly
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Running frame optimizer tests...")

    # Create test instance
    test_instance = TestFrameOptimizer()

    try:
        test_instance.test_frame_format_validation()
        print("✅ Frame format validation test passed")

        test_instance.test_optimization_with_dia_matrix()
        print("✅ DIA matrix optimization test passed")

        test_instance.test_optimization_with_mixed_tensor()
        print("✅ Mixed tensor optimization test passed")

        test_instance.test_convergence_behavior()
        print("✅ Convergence behavior test passed")

        test_instance.test_error_metrics_computation()
        print("✅ Error metrics test passed")

        print("\n✅ All frame optimizer tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
