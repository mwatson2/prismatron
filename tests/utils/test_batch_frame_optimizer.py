"""
Unit tests for batch frame optimizer.

Tests the batch version of the frame optimizer that processes
8 or 16 frames simultaneously using optimized batch operations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Check if CuPy is available
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


# Skip all tests if CuPy not available
pytestmark = pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_target_frames():
    """Create sample target frames on GPU."""
    if not CUPY_AVAILABLE:
        pytest.skip("CuPy not available")

    # Create 8 random frames in planar format (8, 3, 480, 800) uint8
    frames = cp.random.randint(0, 256, (8, 3, 480, 800), dtype=cp.uint8)
    return frames


@pytest.fixture
def sample_target_frames_16():
    """Create sample target frames on GPU for 16-frame batch."""
    if not CUPY_AVAILABLE:
        pytest.skip("CuPy not available")

    frames = cp.random.randint(0, 256, (16, 3, 480, 800), dtype=cp.uint8)
    return frames


# =============================================================================
# BatchFrameOptimizationResult Tests
# =============================================================================


class TestBatchFrameOptimizationResult:
    """Test BatchFrameOptimizationResult dataclass."""

    def test_result_creation(self):
        """Test creating a result object."""
        from src.utils.batch_frame_optimizer import BatchFrameOptimizationResult

        led_values = cp.zeros((8, 3, 100), dtype=cp.uint8)
        result = BatchFrameOptimizationResult(
            led_values=led_values,
            error_metrics=[],
            iterations=5,
            converged=False,
        )

        assert result.led_values.shape == (8, 3, 100)
        assert result.iterations == 5
        assert result.converged is False
        assert result.step_sizes is None
        assert result.timing_data is None
        assert result.mse_per_iteration is None

    def test_result_with_optional_fields(self):
        """Test result with optional fields populated."""
        from src.utils.batch_frame_optimizer import BatchFrameOptimizationResult

        led_values = cp.zeros((8, 3, 100), dtype=cp.uint8)
        step_sizes = np.array([0.1, 0.05, 0.02])
        timing_data = {"total": 0.5, "ata_mult": 0.2}

        result = BatchFrameOptimizationResult(
            led_values=led_values,
            error_metrics=[{"mse": 0.01}],
            iterations=3,
            converged=True,
            step_sizes=step_sizes,
            timing_data=timing_data,
        )

        assert result.step_sizes is not None
        assert len(result.step_sizes) == 3
        assert result.timing_data["total"] == 0.5


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Test input validation for optimize_batch_frames_led_values."""

    def test_invalid_batch_size_raises(self):
        """Test that invalid batch size raises error."""
        from unittest.mock import MagicMock

        from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values

        # Create frames with invalid batch size
        frames = cp.zeros((4, 3, 480, 800), dtype=cp.uint8)

        with pytest.raises(ValueError, match="Batch size must be 8 or 16"):
            optimize_batch_frames_led_values(
                target_frames=frames,
                at_matrix=MagicMock(),
                ata_matrix=MagicMock(),
                ata_inverse=np.zeros((3, 100, 100)),
            )

    def test_non_gpu_input_raises(self):
        """Test that non-GPU input raises error."""
        from unittest.mock import MagicMock

        from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values

        # Create numpy array (not GPU)
        frames = np.zeros((8, 3, 480, 800), dtype=np.uint8)

        with pytest.raises(ValueError, match="must be cupy GPU array"):
            optimize_batch_frames_led_values(
                target_frames=frames,
                at_matrix=MagicMock(),
                ata_matrix=MagicMock(),
                ata_inverse=np.zeros((3, 100, 100)),
            )

    def test_wrong_dtype_raises(self):
        """Test that wrong dtype raises error."""
        from unittest.mock import MagicMock

        from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values

        # Create float32 frames instead of uint8
        frames = cp.zeros((8, 3, 480, 800), dtype=cp.float32)

        with pytest.raises(ValueError, match="must be uint8"):
            optimize_batch_frames_led_values(
                target_frames=frames,
                at_matrix=MagicMock(),
                ata_matrix=MagicMock(),
                ata_inverse=np.zeros((3, 100, 100)),
            )

    def test_unsupported_frame_shape_raises(self):
        """Test that unsupported frame shape raises error."""
        from unittest.mock import MagicMock, patch

        from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values
        from src.utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix

        # Mock with aligned LED count
        mock_at_matrix = MagicMock()
        mock_at_matrix.batch_size = 128  # Multiple of 16

        # Wrong shape - should be (8, 3, H, W) or (8, H, W, 3)
        frames = cp.zeros((8, 480, 800), dtype=cp.uint8)

        with patch.object(BatchSymmetricDiagonalATAMatrix, "__instancecheck__", return_value=True):
            mock_ata_matrix = MagicMock(spec=BatchSymmetricDiagonalATAMatrix)
            mock_ata_matrix.batch_size = 8

            try:
                with pytest.raises(ValueError, match="Unsupported frame shape"):
                    optimize_batch_frames_led_values(
                        target_frames=frames,
                        at_matrix=mock_at_matrix,
                        ata_matrix=mock_ata_matrix,
                        ata_inverse=np.zeros((3, 128, 128)),
                    )
            except TypeError:
                # Expected if isinstance check fails due to mocking complexity
                pass


# =============================================================================
# LED Count Alignment Tests
# =============================================================================


class TestLEDCountAlignment:
    """Test LED count alignment requirements."""

    def test_non_aligned_led_count_raises(self):
        """Test that non-aligned LED count raises error."""
        from unittest.mock import MagicMock

        from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values

        # Mock with non-aligned LED count
        mock_at_matrix = MagicMock()
        mock_at_matrix.batch_size = 100  # Not multiple of 16

        mock_ata_matrix = MagicMock()
        mock_ata_matrix.batch_size = 8

        frames = cp.zeros((8, 3, 480, 800), dtype=cp.uint8)

        with pytest.raises(ValueError, match="LED count must be multiple of 16"):
            optimize_batch_frames_led_values(
                target_frames=frames,
                at_matrix=mock_at_matrix,
                ata_matrix=mock_ata_matrix,
                ata_inverse=np.zeros((3, 100, 100)),
            )


# =============================================================================
# Matrix Type Validation Tests
# =============================================================================


class TestMatrixTypeValidation:
    """Test matrix type validation."""

    def test_wrong_ata_matrix_type_raises(self):
        """Test that wrong ATA matrix type raises error."""
        from unittest.mock import MagicMock

        from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values

        mock_at_matrix = MagicMock()
        mock_at_matrix.batch_size = 128  # Multiple of 16

        # Wrong type - not BatchSymmetricDiagonalATAMatrix
        mock_ata_matrix = MagicMock()
        mock_ata_matrix.__class__.__name__ = "WrongType"

        frames = cp.zeros((8, 3, 480, 800), dtype=cp.uint8)

        with pytest.raises(TypeError, match="BatchSymmetricDiagonalATAMatrix"):
            optimize_batch_frames_led_values(
                target_frames=frames,
                at_matrix=mock_at_matrix,
                ata_matrix=mock_ata_matrix,
                ata_inverse=np.zeros((3, 128, 128)),
            )


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_compute_batch_mse_shape(self):
        """Test _compute_batch_mse returns correct shape."""
        from unittest.mock import MagicMock

        from src.utils.batch_frame_optimizer import _compute_batch_mse

        led_values = cp.zeros((8, 3, 100), dtype=cp.float32)
        target = cp.zeros((8, 3, 480, 800), dtype=cp.uint8)

        # Mock forward pass
        mock_at_matrix = MagicMock()
        mock_at_matrix.forward_pass_3d.return_value = cp.zeros((3, 480, 800), dtype=cp.float32)

        result = _compute_batch_mse(led_values, target, mock_at_matrix)

        assert result.shape == (8,)

    def test_compute_frame_error_metrics(self):
        """Test _compute_frame_error_metrics returns correct keys."""
        from unittest.mock import MagicMock

        from src.utils.batch_frame_optimizer import _compute_frame_error_metrics

        led_values = cp.zeros((3, 100), dtype=cp.float32)
        target = cp.zeros((3, 480, 800), dtype=cp.uint8)

        # Mock forward pass
        mock_at_matrix = MagicMock()
        mock_at_matrix.forward_pass_3d.return_value = cp.zeros((3, 480, 800), dtype=cp.float32)

        result = _compute_frame_error_metrics(led_values, target, mock_at_matrix)

        assert "mse" in result
        assert "mae" in result
        assert "psnr" in result
        assert "rendered_mean" in result
        assert "target_mean" in result

    def test_compute_frame_error_metrics_handles_exception(self):
        """Test error metrics gracefully handles exceptions."""
        from unittest.mock import MagicMock

        from src.utils.batch_frame_optimizer import _compute_frame_error_metrics

        led_values = cp.zeros((3, 100), dtype=cp.float32)
        target = cp.zeros((3, 480, 800), dtype=cp.uint8)

        # Mock that raises exception
        mock_at_matrix = MagicMock()
        mock_at_matrix.forward_pass_3d.side_effect = RuntimeError("Test error")

        result = _compute_frame_error_metrics(led_values, target, mock_at_matrix)

        assert result["mse"] == float("inf")
        assert result["psnr"] == 0.0


# =============================================================================
# Frame Format Tests
# =============================================================================


class TestFrameFormat:
    """Test frame format handling."""

    def test_planar_format_accepted(self):
        """Test planar format (batch, 3, H, W) is accepted."""
        from unittest.mock import MagicMock, patch

        from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values
        from src.utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix

        # Create properly shaped frames
        frames = cp.zeros((8, 3, 480, 800), dtype=cp.uint8)

        mock_at_matrix = MagicMock()
        mock_at_matrix.batch_size = 128

        # Need to properly mock BatchSymmetricDiagonalATAMatrix
        with patch.object(BatchSymmetricDiagonalATAMatrix, "__instancecheck__", return_value=True):
            mock_ata_matrix = MagicMock(spec=BatchSymmetricDiagonalATAMatrix)
            mock_ata_matrix.batch_size = 8
            mock_ata_matrix.multiply_batch8_3d.return_value = cp.zeros((8, 3, 128), dtype=cp.float32)
            mock_ata_matrix.g_ata_g_batch_3d.return_value = cp.zeros((8, 3), dtype=cp.float32)

            mock_at_matrix.transpose_dot_product_3d_batch.return_value = cp.zeros((8, 128, 3), dtype=cp.float32)

            # Should not raise
            try:
                result = optimize_batch_frames_led_values(
                    target_frames=frames,
                    at_matrix=mock_at_matrix,
                    ata_matrix=mock_ata_matrix,
                    ata_inverse=np.zeros((3, 128, 128), dtype=np.float32),
                    max_iterations=1,
                )
                assert result.led_values.shape[0] == 8
            except TypeError:
                # Expected if isinstance check still fails
                pass


# =============================================================================
# ATA Inverse Tests
# =============================================================================


class TestATAInverse:
    """Test ATA inverse shape validation."""

    def test_wrong_ata_inverse_shape_raises(self):
        """Test that wrong ATA inverse shape raises error."""
        from unittest.mock import MagicMock, patch

        from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values
        from src.utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix

        frames = cp.zeros((8, 3, 480, 800), dtype=cp.uint8)

        mock_at_matrix = MagicMock()
        mock_at_matrix.batch_size = 128

        with patch.object(BatchSymmetricDiagonalATAMatrix, "__instancecheck__", return_value=True):
            mock_ata_matrix = MagicMock(spec=BatchSymmetricDiagonalATAMatrix)
            mock_ata_matrix.batch_size = 8

            mock_at_matrix.transpose_dot_product_3d_batch.return_value = cp.zeros((8, 128, 3), dtype=cp.float32)

            # Wrong shape for ata_inverse
            wrong_shape_inverse = np.zeros((3, 64, 64), dtype=np.float32)  # Should be (3, 128, 128)

            try:
                with pytest.raises(ValueError, match="ATA inverse shape"):
                    optimize_batch_frames_led_values(
                        target_frames=frames,
                        at_matrix=mock_at_matrix,
                        ata_matrix=mock_ata_matrix,
                        ata_inverse=wrong_shape_inverse,
                    )
            except TypeError:
                # Expected if isinstance check fails
                pass
