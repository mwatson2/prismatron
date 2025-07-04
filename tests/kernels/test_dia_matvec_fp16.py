"""
Unit tests for dia_matvec_fp16 kernels.

Tests the FP16 output DIA matrix-vector multiplication CUDA kernels for A^T A operations
with comprehensive validation against reference implementations and precision analysis.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

# CuPy imports - conditionally available
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    import scipy.sparse as sp

    from src.utils.kernels.dia_matvec import (
        CustomDIA3DMatVec,
        CustomDIAMatVec,
    )
    from src.utils.kernels.dia_matvec_fp16 import (
        CustomDIA3DMatVecFP16,
        CustomDIAMatVecFP16,
    )

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestDIAMatVecFP16Kernel:
    """Test suite for FP16 output DIA matrix-vector multiplication kernels."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tolerance_fp16 = 5e-3  # FP16 has lower precision
        self.tolerance_comparison = 1e-2  # For comparing FP16 vs FP32
        self.n = 1000  # Matrix dimension
        self.channels = 3

    def create_simple_dia_matrix(self, n: int = 5) -> Tuple[cusp.dia_matrix, np.ndarray]:
        """Create simple DIA matrix for basic testing."""
        # Create dense matrix with known pattern
        dense = np.array(
            [
                [1.0, 2.0, 0.0, 0.0, 0.0],
                [3.0, 4.0, 5.0, 0.0, 0.0],
                [0.0, 6.0, 7.0, 8.0, 0.0],
                [0.0, 0.0, 9.0, 10.0, 11.0],
                [0.0, 0.0, 0.0, 12.0, 13.0],
            ],
            dtype=np.float32,
        )

        # Expand if needed
        if n > 5:
            new_dense = np.zeros((n, n), dtype=np.float32)
            new_dense[:5, :5] = dense
            # Add some diagonal elements
            for i in range(5, n):
                new_dense[i, i] = float(i + 1)
                if i > 0:
                    new_dense[i, i - 1] = float(i)
                if i < n - 1:
                    new_dense[i, i + 1] = float(i + 2)
            dense = new_dense

        # Convert to DIA format
        dia_scipy = sp.dia_matrix(dense)
        dia_cupy = cusp.dia_matrix(dia_scipy)

        return dia_cupy, dense

    def create_realistic_ata_matrix(self, n: int = 500, bandwidth: int = 50) -> Tuple[cusp.dia_matrix, np.ndarray]:
        """Create realistic A^T A matrix structure."""
        # Create band offsets
        max_offset = bandwidth // 2
        offsets = list(range(-max_offset, max_offset + 1))
        num_bands = len(offsets)

        # Create data array: shape (num_bands, n)
        data = np.zeros((num_bands, n), dtype=np.float32)

        np.random.seed(42)

        for i, offset in enumerate(offsets):
            # Band density decreases with distance from main diagonal
            density = max(0.3, 1.0 - abs(offset) / max_offset * 0.7)

            # Fill band with random values
            for j in range(n):
                if np.random.rand() < density:
                    # Higher values near main diagonal
                    intensity = np.exp(-abs(offset) / 10) * (np.random.rand() * 100 + 10)
                    data[i, j] = intensity

        # Create DIA matrix
        dia_matrix = cusp.dia_matrix((data, offsets), shape=(n, n))

        # Create dense equivalent for verification
        dense_matrix = dia_matrix.toarray()

        return dia_matrix, dense_matrix

    def test_kernel_compilation_fp16(self):
        """Test that FP16 kernels compile successfully."""
        # Test basic FP16 kernel
        custom_fp16_basic = CustomDIAMatVecFP16(use_optimized=False)
        assert custom_fp16_basic is not None

        # Test optimized FP16 kernel
        custom_fp16_opt = CustomDIAMatVecFP16(use_optimized=True)
        assert custom_fp16_opt is not None

    def test_output_dtype_fp16(self):
        """Test that kernels output FP16 data type."""
        dia_matrix, dense_matrix = self.create_simple_dia_matrix(5)
        x = cp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=cp.float32)

        # Test FP16 kernel
        custom_fp16 = CustomDIAMatVecFP16(use_optimized=False)
        y_fp16 = custom_fp16(dia_matrix, x)

        # Verify dtype is FP16
        assert y_fp16.dtype == cp.float16, f"Expected FP16, got {y_fp16.dtype}"
        assert y_fp16.shape == (5,)

    def test_fp16_vs_fp32_comparison(self):
        """Test FP16 kernel against FP32 kernel for precision analysis."""
        dia_matrix, dense_matrix = self.create_realistic_ata_matrix(200, 20)

        # Create test vector
        np.random.seed(42)
        x = cp.random.randn(200, dtype=cp.float32)

        # Compute FP32 result
        custom_fp32 = CustomDIAMatVec(use_optimized=False)
        y_fp32 = custom_fp32(dia_matrix, x)

        # Compute FP16 result
        custom_fp16 = CustomDIAMatVecFP16(use_optimized=False)
        y_fp16 = custom_fp16(dia_matrix, x)

        # Convert FP16 to FP32 for comparison
        y_fp16_as_fp32 = y_fp16.astype(cp.float32)

        # Compare results
        max_error = cp.max(cp.abs(y_fp32 - y_fp16_as_fp32))
        relative_error = cp.max(cp.abs((y_fp32 - y_fp16_as_fp32) / (y_fp32 + 1e-8)))

        max_error_cpu = float(cp.asnumpy(max_error))
        relative_error_cpu = float(cp.asnumpy(relative_error))

        assert max_error_cpu < self.tolerance_comparison, f"Max error {max_error_cpu} exceeds tolerance"
        assert relative_error_cpu < 0.05, f"Relative error {relative_error_cpu} too high"  # 5% relative error

        logger.info(f"FP16 vs FP32 - Max error: {max_error_cpu:.6f}, Relative error: {relative_error_cpu:.6f}")

    def test_simple_matrix_correctness_fp16(self):
        """Test FP16 kernels with simple known matrix."""
        dia_matrix, dense_matrix = self.create_simple_dia_matrix(5)

        # Create test vector
        x = cp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=cp.float32)

        # Reference result
        y_ref = cp.asarray(dense_matrix, dtype=cp.float32) @ x

        # Test FP16 kernels
        custom_fp16_basic = CustomDIAMatVecFP16(use_optimized=False)
        custom_fp16_opt = CustomDIAMatVecFP16(use_optimized=True)

        y_fp16_basic = custom_fp16_basic(dia_matrix, x)
        y_fp16_opt = custom_fp16_opt(dia_matrix, x)

        # Convert to FP32 for comparison
        y_fp16_basic_as_fp32 = y_fp16_basic.astype(cp.float32)
        y_fp16_opt_as_fp32 = y_fp16_opt.astype(cp.float32)

        # Compare results
        error_basic = cp.max(cp.abs(y_fp16_basic_as_fp32 - y_ref))
        error_opt = cp.max(cp.abs(y_fp16_opt_as_fp32 - y_ref))

        assert float(error_basic) < self.tolerance_comparison, f"Basic FP16 kernel error: {float(error_basic)}"
        assert float(error_opt) < self.tolerance_comparison, f"Optimized FP16 kernel error: {float(error_opt)}"

    def test_memory_efficiency(self):
        """Test that FP16 kernels use less memory than FP32."""
        dia_matrix, dense_matrix = self.create_realistic_ata_matrix(500, 50)
        x = cp.random.randn(500, dtype=cp.float32)

        # Compute results
        custom_fp32 = CustomDIAMatVec(use_optimized=False)
        custom_fp16 = CustomDIAMatVecFP16(use_optimized=False)

        y_fp32 = custom_fp32(dia_matrix, x)
        y_fp16 = custom_fp16(dia_matrix, x)

        # Check memory usage
        fp16_bytes = y_fp16.nbytes
        fp32_bytes = y_fp32.nbytes

        assert fp16_bytes == fp32_bytes // 2, f"FP16 should use half memory: {fp16_bytes} vs {fp32_bytes}"
        assert y_fp16.dtype == cp.float16
        assert y_fp32.dtype == cp.float32

    def test_3d_dia_fp16_functionality(self):
        """Test 3D DIA FP16 kernel functionality."""
        # Test basic 3D FP16 kernel
        custom_3d_fp16_basic = CustomDIA3DMatVecFP16(use_optimized=False)
        assert custom_3d_fp16_basic is not None

        # Test optimized 3D FP16 kernel
        custom_3d_fp16_opt = CustomDIA3DMatVecFP16(use_optimized=True)
        assert custom_3d_fp16_opt is not None

    def test_3d_dia_fp16_correctness(self):
        """Test 3D DIA FP16 kernel correctness."""
        n = 100
        channels = 3
        bandwidth = 20

        # Create 3D DIA data
        max_offset = bandwidth // 2
        offsets = list(range(-max_offset, max_offset + 1))
        num_bands = len(offsets)

        # Create random 3D DIA data
        np.random.seed(42)
        dia_data_3d = np.random.rand(channels, num_bands, n).astype(np.float32) * 100
        dia_offsets = np.array(offsets, dtype=np.int32)

        # Create input vectors
        x = np.random.rand(channels, n).astype(np.float32)

        # Convert to GPU
        dia_data_gpu = cp.asarray(dia_data_3d)
        dia_offsets_gpu = cp.asarray(dia_offsets)
        x_gpu = cp.asarray(x)

        # Test 3D FP16 kernels
        custom_3d_fp16_basic = CustomDIA3DMatVecFP16(use_optimized=False)
        custom_3d_fp16_opt = CustomDIA3DMatVecFP16(use_optimized=True)

        y_fp16_basic = custom_3d_fp16_basic(dia_data_gpu, dia_offsets_gpu, x_gpu)
        y_fp16_opt = custom_3d_fp16_opt(dia_data_gpu, dia_offsets_gpu, x_gpu)

        # Verify output dtype
        assert y_fp16_basic.dtype == cp.float16
        assert y_fp16_opt.dtype == cp.float16

        # Compute reference result channel by channel
        y_ref = np.zeros((channels, n), dtype=np.float32)
        for c in range(channels):
            # Create DIA matrix for this channel
            dia_matrix_c = cusp.dia_matrix((dia_data_3d[c], offsets), shape=(n, n))
            y_ref_c = dia_matrix_c @ cp.asarray(x[c])
            y_ref[c] = cp.asnumpy(y_ref_c)

        y_ref_gpu = cp.asarray(y_ref)

        # Convert FP16 results to FP32 for comparison
        y_fp16_basic_as_fp32 = y_fp16_basic.astype(cp.float32)
        y_fp16_opt_as_fp32 = y_fp16_opt.astype(cp.float32)

        # Compare results
        error_basic = cp.max(cp.abs(y_fp16_basic_as_fp32 - y_ref_gpu))
        error_opt = cp.max(cp.abs(y_fp16_opt_as_fp32 - y_ref_gpu))

        assert float(error_basic) < self.tolerance_comparison, f"3D Basic FP16 kernel error: {float(error_basic)}"
        assert float(error_opt) < self.tolerance_comparison, f"3D Optimized FP16 kernel error: {float(error_opt)}"

    def test_3d_fp16_vs_fp32_comparison(self):
        """Test 3D FP16 kernel against 3D FP32 kernel."""
        n = 50
        channels = 3
        bandwidth = 10

        # Create 3D DIA data
        max_offset = bandwidth // 2
        offsets = list(range(-max_offset, max_offset + 1))
        num_bands = len(offsets)

        np.random.seed(42)
        dia_data_3d = np.random.rand(channels, num_bands, n).astype(np.float32) * 50
        dia_offsets = np.array(offsets, dtype=np.int32)
        x = np.random.rand(channels, n).astype(np.float32)

        # Convert to GPU
        dia_data_gpu = cp.asarray(dia_data_3d)
        dia_offsets_gpu = cp.asarray(dia_offsets)
        x_gpu = cp.asarray(x)

        # Compute FP32 result
        custom_3d_fp32 = CustomDIA3DMatVec(use_optimized=False)
        y_fp32 = custom_3d_fp32(dia_data_gpu, dia_offsets_gpu, x_gpu)

        # Compute FP16 result
        custom_3d_fp16 = CustomDIA3DMatVecFP16(use_optimized=False)
        y_fp16 = custom_3d_fp16(dia_data_gpu, dia_offsets_gpu, x_gpu)

        # Convert FP16 to FP32 for comparison
        y_fp16_as_fp32 = y_fp16.astype(cp.float32)

        # Compare results
        max_error = cp.max(cp.abs(y_fp32 - y_fp16_as_fp32))
        relative_error = cp.max(cp.abs((y_fp32 - y_fp16_as_fp32) / (y_fp32 + 1e-8)))

        max_error_cpu = float(cp.asnumpy(max_error))
        relative_error_cpu = float(cp.asnumpy(relative_error))

        assert max_error_cpu < self.tolerance_comparison, f"3D Max error {max_error_cpu} exceeds tolerance"
        assert relative_error_cpu < 0.1, f"3D Relative error {relative_error_cpu} too high"

    def test_precision_edge_cases_fp16(self):
        """Test FP16 precision with edge cases."""
        dia_matrix, dense_matrix = self.create_simple_dia_matrix(10)

        # Test with very small values
        x_small = cp.ones(10, dtype=cp.float32) * 1e-3

        custom_fp16 = CustomDIAMatVecFP16(use_optimized=False)
        y_fp16 = custom_fp16(dia_matrix, x_small)

        # Should not contain NaN or infinity
        y_cpu = cp.asnumpy(y_fp16.astype(cp.float32))
        assert not np.any(np.isnan(y_cpu)), "Result contains NaN values"
        assert not np.any(np.isinf(y_cpu)), "Result contains infinity values"

    def test_zero_input_cases_fp16(self):
        """Test FP16 kernels with zero inputs."""
        dia_matrix, dense_matrix = self.create_simple_dia_matrix(10)

        # Test zero input vector
        x_zero = cp.zeros(10, dtype=cp.float32)

        custom_fp16 = CustomDIAMatVecFP16(use_optimized=False)
        y_fp16 = custom_fp16(dia_matrix, x_zero)

        y_cpu = cp.asnumpy(y_fp16.astype(cp.float32))
        assert np.allclose(y_cpu, 0.0, atol=1e-6)

    def test_performance_comparison_fp16(self):
        """Compare FP16 vs FP32 performance."""
        dia_matrix, dense_matrix = self.create_realistic_ata_matrix(1000, 50)
        x = cp.random.randn(1000, dtype=cp.float32)

        # Test kernels
        custom_fp32 = CustomDIAMatVec(use_optimized=False)
        custom_fp16 = CustomDIAMatVecFP16(use_optimized=False)

        # Warm-up
        for _ in range(3):
            _ = custom_fp32(dia_matrix, x)
            _ = custom_fp16(dia_matrix, x)
        cp.cuda.Stream.null.synchronize()

        # Time FP32 kernel
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        y_fp32 = custom_fp32(dia_matrix, x)
        end_event.record()
        end_event.synchronize()

        time_fp32 = cp.cuda.get_elapsed_time(start_event, end_event)

        # Time FP16 kernel
        start_event.record()
        y_fp16 = custom_fp16(dia_matrix, x)
        end_event.record()
        end_event.synchronize()

        time_fp16 = cp.cuda.get_elapsed_time(start_event, end_event)

        # Verify results are correct
        assert y_fp32.shape == (1000,)
        assert y_fp16.shape == (1000,)
        assert y_fp32.dtype == cp.float32
        assert y_fp16.dtype == cp.float16

        # Performance should be reasonable
        assert time_fp32 < 50.0, f"FP32 kernel took {time_fp32:.2f}ms, too slow"
        assert time_fp16 < 50.0, f"FP16 kernel took {time_fp16:.2f}ms, too slow"

        logger.info(f"Performance - FP32: {time_fp32:.2f}ms, FP16: {time_fp16:.2f}ms")
        logger.info(f"Memory - FP32: {y_fp32.nbytes} bytes, FP16: {y_fp16.nbytes} bytes")
        logger.info(f"Output memory reduction: {y_fp32.nbytes / y_fp16.nbytes:.1f}x")

    def test_different_matrix_sizes_fp16(self):
        """Test FP16 kernels with different matrix sizes."""
        sizes = [10, 50, 100, 200]

        for n in sizes:
            dia_matrix, dense_matrix = self.create_realistic_ata_matrix(n, min(20, n // 5))

            # Create test vector
            np.random.seed(42)
            x = cp.random.randn(n, dtype=cp.float32)

            # Reference result
            y_ref = cp.asarray(dense_matrix, dtype=cp.float32) @ x

            # Test FP16 kernels
            custom_fp16_basic = CustomDIAMatVecFP16(use_optimized=False)
            custom_fp16_opt = CustomDIAMatVecFP16(use_optimized=True)

            y_fp16_basic = custom_fp16_basic(dia_matrix, x)
            y_fp16_opt = custom_fp16_opt(dia_matrix, x)

            # Convert to FP32 for comparison
            y_fp16_basic_as_fp32 = y_fp16_basic.astype(cp.float32)
            y_fp16_opt_as_fp32 = y_fp16_opt.astype(cp.float32)

            # Compare results
            error_basic = cp.max(cp.abs(y_fp16_basic_as_fp32 - y_ref))
            error_opt = cp.max(cp.abs(y_fp16_opt_as_fp32 - y_ref))

            assert float(error_basic) < self.tolerance_comparison, f"Size {n} FP16 basic error: {float(error_basic)}"
            assert float(error_opt) < self.tolerance_comparison, f"Size {n} FP16 optimized error: {float(error_opt)}"

    def test_input_validation_fp16(self):
        """Test input validation for FP16 kernels."""
        dia_matrix, dense_matrix = self.create_simple_dia_matrix(5)

        custom_fp16 = CustomDIAMatVecFP16(use_optimized=False)

        # Test wrong input size
        x_wrong = cp.ones(10, dtype=cp.float32)  # Wrong size

        with pytest.raises((ValueError, RuntimeError)):
            custom_fp16(dia_matrix, x_wrong)

    def test_3d_input_validation_fp16(self):
        """Test input validation for 3D FP16 kernels."""
        n = 50
        channels = 3
        num_bands = 5

        # Create valid data
        dia_data_3d = np.random.rand(channels, num_bands, n).astype(np.float32)
        dia_offsets = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
        x = np.random.rand(channels, n).astype(np.float32)

        dia_data_gpu = cp.asarray(dia_data_3d)
        dia_offsets_gpu = cp.asarray(dia_offsets)
        x_gpu = cp.asarray(x)

        custom_3d_fp16 = CustomDIA3DMatVecFP16(use_optimized=False)

        # Test wrong offset shape
        wrong_offsets = cp.array([-1, 0, 1], dtype=cp.int32)  # Wrong length

        with pytest.raises(AssertionError):
            custom_3d_fp16(dia_data_gpu, wrong_offsets, x_gpu)

        # Test wrong x shape
        wrong_x = cp.random.rand(2, n, dtype=cp.float32)  # Wrong channels

        with pytest.raises(AssertionError):
            custom_3d_fp16(dia_data_gpu, dia_offsets_gpu, wrong_x)

    def test_range_preservation_fp16(self):
        """Test that FP16 conversion preserves reasonable value ranges."""
        # Test different scales to analyze FP16 behavior
        test_scales = [0.1, 1.0, 10.0, 100.0]

        for scale in test_scales:
            dia_matrix, dense_matrix = self.create_realistic_ata_matrix(50, 10)

            # Scale the matrix data
            dia_matrix.data = dia_matrix.data * scale

            # Create test vector
            x = cp.ones(50, dtype=cp.float32)

            custom_fp16 = CustomDIAMatVecFP16(use_optimized=False)
            y_fp16 = custom_fp16(dia_matrix, x)

            y_cpu = cp.asnumpy(y_fp16.astype(cp.float32))

            # Check for reasonable values (no overflow/underflow to zero)
            if scale >= 1.0:
                assert np.any(y_cpu > 0), f"All results zero for scale {scale}"

            # No NaN or infinity
            assert not np.any(np.isnan(y_cpu)), f"NaN values for scale {scale}"
            assert not np.any(np.isinf(y_cpu)), f"Infinity values for scale {scale}"
