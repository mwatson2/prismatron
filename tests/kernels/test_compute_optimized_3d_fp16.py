"""
Unit tests for compute_optimized_3d_fp16 kernels.

Tests the FP16 output compute-optimized 3D CUDA kernels for A^T @ b operations
with comprehensive validation against reference implementations and precision analysis.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

# CuPy imports - conditionally available
try:
    import cupy as cp

    from src.utils.kernels.compute_optimized_3d import (
        cuda_transpose_dot_product_3d_compute_optimized,
    )
    from src.utils.kernels.compute_optimized_3d_fp16 import (
        cuda_transpose_dot_product_3d_compute_optimized_fp16,
        get_compute_optimized_3d_fp16_kernel,
    )

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestComputeOptimized3DFP16Kernel:
    """Test suite for FP16 output compute-optimized 3D kernels."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tolerance_fp16 = 5e-3  # FP16 has lower precision
        self.tolerance_comparison = 1e-2  # For comparing FP16 vs FP32
        self.channels = 3
        self.height = 400
        self.width = 600
        self.block_size = 64

    def generate_test_data(self, batch_size: int = 50):
        """Generate realistic test data for kernel validation."""
        # Generate sparse values (channels, batch_size, block_size, block_size)
        np.random.seed(42)
        sparse_values = np.random.rand(self.channels, batch_size, self.block_size, self.block_size).astype(np.float32)

        # Generate block positions (channels, batch_size, 2)
        max_row = self.height - self.block_size
        max_col = self.width - self.block_size
        block_positions = np.zeros((self.channels, batch_size, 2), dtype=np.int32)

        for c in range(self.channels):
            for b in range(batch_size):
                block_positions[c, b, 0] = np.random.randint(0, max_row)  # top_row
                block_positions[c, b, 1] = np.random.randint(0, max_col)  # top_col

        # Generate target image (channels, height, width)
        target_3d = np.random.rand(self.channels, self.height, self.width).astype(np.float32)

        return sparse_values, block_positions, target_3d

    def compute_reference_result(self, sparse_values, block_positions, target_3d):
        """Compute reference result using CPU implementation."""
        channels, batch_size, block_size, _ = sparse_values.shape
        result = np.zeros((batch_size, channels), dtype=np.float32)

        for led_id in range(batch_size):
            for channel_id in range(channels):
                # Get block position
                top_row = block_positions[channel_id, led_id, 0]
                top_col = block_positions[channel_id, led_id, 1]

                # Extract sparse block and target block
                sparse_block = sparse_values[channel_id, led_id]
                target_block = target_3d[
                    channel_id,
                    top_row : top_row + block_size,
                    top_col : top_col + block_size,
                ]

                # Compute dot product
                result[led_id, channel_id] = np.sum(sparse_block * target_block)

        return result

    def test_kernel_compilation(self):
        """Test that the FP16 kernel compiles successfully."""
        kernel = get_compute_optimized_3d_fp16_kernel()
        assert kernel is not None

    def test_output_dtype_fp16(self):
        """Test that kernel outputs FP16 data type."""
        batch_size = 8
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        # Convert to CuPy arrays
        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Compute GPU result
        result_gpu = cuda_transpose_dot_product_3d_compute_optimized_fp16(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Verify dtype is FP16
        assert result_gpu.dtype == cp.float16, f"Expected FP16, got {result_gpu.dtype}"
        assert result_gpu.shape == (batch_size, self.channels)

    def test_fp16_vs_fp32_comparison(self):
        """Test FP16 kernel against FP32 kernel for precision analysis."""
        batch_size = 32
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        # Convert to CuPy arrays
        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Compute FP32 result
        result_fp32 = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Compute FP16 result
        result_fp16 = cuda_transpose_dot_product_3d_compute_optimized_fp16(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Convert FP16 to FP32 for comparison
        result_fp16_as_fp32 = result_fp16.astype(cp.float32)

        # Compare results
        max_error = cp.max(cp.abs(result_fp32 - result_fp16_as_fp32))
        relative_error = cp.max(cp.abs((result_fp32 - result_fp16_as_fp32) / (result_fp32 + 1e-8)))

        max_error_cpu = float(cp.asnumpy(max_error))
        relative_error_cpu = float(cp.asnumpy(relative_error))

        assert max_error_cpu < self.tolerance_comparison, f"Max error {max_error_cpu} exceeds tolerance"
        assert relative_error_cpu < 0.05, f"Relative error {relative_error_cpu} too high"  # 5% relative error

        logger.info(f"FP16 vs FP32 - Max error: {max_error_cpu:.6f}, Relative error: {relative_error_cpu:.6f}")

    def test_small_batch_correctness(self):
        """Test FP16 kernel correctness with small batch size."""
        batch_size = 8
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        # Convert to CuPy arrays
        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Compute GPU result
        result_gpu = cuda_transpose_dot_product_3d_compute_optimized_fp16(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Compute reference result
        result_ref = self.compute_reference_result(sparse_values, block_positions, target_3d)

        # Compare results (convert FP16 to FP32 for comparison)
        result_cpu = cp.asnumpy(result_gpu.astype(cp.float32))
        max_error = np.max(np.abs(result_cpu - result_ref))

        assert (
            max_error < self.tolerance_comparison
        ), f"Max error {max_error} exceeds tolerance {self.tolerance_comparison}"

    def test_medium_batch_correctness(self):
        """Test FP16 kernel correctness with medium batch size."""
        batch_size = 128
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        result_gpu = cuda_transpose_dot_product_3d_compute_optimized_fp16(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        result_ref = self.compute_reference_result(sparse_values, block_positions, target_3d)
        result_cpu = cp.asnumpy(result_gpu.astype(cp.float32))

        max_error = np.max(np.abs(result_cpu - result_ref))
        assert max_error < self.tolerance_comparison

    def test_memory_efficiency(self):
        """Test that FP16 kernel uses less memory than FP32."""
        batch_size = 256
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Compute results
        result_fp16 = cuda_transpose_dot_product_3d_compute_optimized_fp16(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        result_fp32 = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Check memory usage
        fp16_bytes = result_fp16.nbytes
        fp32_bytes = result_fp32.nbytes

        assert fp16_bytes == fp32_bytes // 2, f"FP16 should use half memory: {fp16_bytes} vs {fp32_bytes}"
        assert result_fp16.dtype == cp.float16
        assert result_fp32.dtype == cp.float32

    def test_precision_edge_cases(self):
        """Test FP16 precision with edge cases."""
        batch_size = 16
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        # Test with very small values
        sparse_values = sparse_values * 1e-3
        target_3d = target_3d * 1e-3

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        result_gpu = cuda_transpose_dot_product_3d_compute_optimized_fp16(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Should not contain NaN or infinity
        result_cpu = cp.asnumpy(result_gpu)
        assert not np.any(np.isnan(result_cpu)), "Result contains NaN values"
        assert not np.any(np.isinf(result_cpu)), "Result contains infinity values"

    def test_zero_input_cases(self):
        """Test FP16 kernel with zero inputs."""
        batch_size = 16
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        # Test zero sparse values
        sparse_values[:] = 0.0

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        result_gpu = cuda_transpose_dot_product_3d_compute_optimized_fp16(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        result_cpu = cp.asnumpy(result_gpu.astype(cp.float32))
        assert np.allclose(result_cpu, 0.0, atol=1e-6)

    def test_input_validation(self):
        """Test FP16 kernel input validation."""
        batch_size = 8
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Test mismatched channels
        with pytest.raises(ValueError, match="Input channels .* != expected"):
            wrong_target = cp.zeros((2, self.height, self.width), dtype=cp.float32)
            cuda_transpose_dot_product_3d_compute_optimized_fp16(
                sparse_values_gpu,
                block_positions_gpu,
                wrong_target,
                batch_size,
                self.channels,
                self.block_size,
            )

    def test_performance_comparison(self):
        """Compare FP16 vs FP32 performance."""
        batch_size = 500
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Warm-up
        for _ in range(3):
            _ = cuda_transpose_dot_product_3d_compute_optimized_fp16(
                sparse_values_gpu,
                block_positions_gpu,
                target_3d_gpu,
                batch_size,
                self.channels,
                self.block_size,
            )
        cp.cuda.Stream.null.synchronize()

        # Time FP16 kernel
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        result_fp16 = cuda_transpose_dot_product_3d_compute_optimized_fp16(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )
        end_event.record()
        end_event.synchronize()

        elapsed_fp16 = cp.cuda.get_elapsed_time(start_event, end_event)

        # Time FP32 kernel for comparison
        start_event.record()
        result_fp32 = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )
        end_event.record()
        end_event.synchronize()

        elapsed_fp32 = cp.cuda.get_elapsed_time(start_event, end_event)

        # Verify results
        assert result_fp16.shape == (batch_size, self.channels)
        assert result_fp16.dtype == cp.float16
        assert result_fp32.dtype == cp.float32

        # Performance assertions
        assert elapsed_fp16 < 100.0, f"FP16 kernel took {elapsed_fp16:.2f}ms, too slow"

        logger.info(f"Performance - FP16: {elapsed_fp16:.2f}ms, FP32: {elapsed_fp32:.2f}ms")
        logger.info(f"Memory - FP16: {result_fp16.nbytes} bytes, FP32: {result_fp32.nbytes} bytes")

    def test_range_preservation(self):
        """Test that FP16 conversion preserves reasonable value ranges."""
        batch_size = 32
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        # Scale to test different value ranges
        test_scales = [0.01, 1.0, 100.0]

        for scale in test_scales:
            scaled_sparse = sparse_values * scale
            scaled_target = target_3d * scale

            sparse_values_gpu = cp.asarray(scaled_sparse)
            block_positions_gpu = cp.asarray(block_positions)
            target_3d_gpu = cp.asarray(scaled_target)

            result_gpu = cuda_transpose_dot_product_3d_compute_optimized_fp16(
                sparse_values_gpu,
                block_positions_gpu,
                target_3d_gpu,
                batch_size,
                self.channels,
                self.block_size,
            )

            result_cpu = cp.asnumpy(result_gpu.astype(cp.float32))

            # Check for reasonable values (no overflow/underflow to zero)
            if scale >= 1.0:
                assert np.any(result_cpu > 0), f"All results zero for scale {scale}"

            # No NaN or infinity
            assert not np.any(np.isnan(result_cpu)), f"NaN values for scale {scale}"
            assert not np.any(np.isinf(result_cpu)), f"Infinity values for scale {scale}"
