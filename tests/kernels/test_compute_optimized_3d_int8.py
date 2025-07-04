"""
Unit tests for compute_optimized_3d_int8 kernels.

Tests the INT8 input compute-optimized 3D CUDA kernels for A^T @ b operations
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
    from src.utils.kernels.compute_optimized_3d_int8 import (
        cuda_transpose_dot_product_3d_compute_optimized_int8,
        cuda_transpose_dot_product_3d_compute_optimized_int8_experimental,
        get_compute_optimized_3d_int8_experimental_kernel,
        get_compute_optimized_3d_int8_kernel,
    )

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestComputeOptimized3DInt8Kernel:
    """Test suite for INT8 input compute-optimized 3D kernels."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tolerance = 1e-3  # Tolerance for normalized INT8 results
        self.channels = 3
        self.height = 400
        self.width = 600
        self.block_size = 64

    def generate_test_data_int8(self, batch_size: int = 50):
        """Generate realistic INT8 test data for kernel validation."""
        # Generate sparse values (channels, batch_size, block_size, block_size) as UINT8
        np.random.seed(42)
        sparse_values = np.random.randint(
            0,
            256,
            size=(self.channels, batch_size, self.block_size, self.block_size),
            dtype=np.uint8,
        )

        # Generate block positions (channels, batch_size, 2)
        max_row = self.height - self.block_size
        max_col = self.width - self.block_size
        block_positions = np.zeros((self.channels, batch_size, 2), dtype=np.int32)

        for c in range(self.channels):
            for b in range(batch_size):
                block_positions[c, b, 0] = np.random.randint(0, max_row)  # top_row
                block_positions[c, b, 1] = np.random.randint(0, max_col)  # top_col

        # Generate target image (channels, height, width) as UINT8
        target_3d = np.random.randint(0, 256, size=(self.channels, self.height, self.width), dtype=np.uint8)

        return sparse_values, block_positions, target_3d

    def generate_test_data_fp32(self, batch_size: int = 50):
        """Generate FP32 test data for comparison."""
        # Generate sparse values as FP32 normalized [0,1]
        np.random.seed(42)
        sparse_values = np.random.rand(self.channels, batch_size, self.block_size, self.block_size).astype(np.float32)

        # Generate block positions (same as INT8)
        max_row = self.height - self.block_size
        max_col = self.width - self.block_size
        block_positions = np.zeros((self.channels, batch_size, 2), dtype=np.int32)

        for c in range(self.channels):
            for b in range(batch_size):
                block_positions[c, b, 0] = np.random.randint(0, max_row)
                block_positions[c, b, 1] = np.random.randint(0, max_col)

        # Generate target image as FP32 normalized [0,1]
        target_3d = np.random.rand(self.channels, self.height, self.width).astype(np.float32)

        return sparse_values, block_positions, target_3d

    def compute_reference_result_int8(self, sparse_values, block_positions, target_3d):
        """Compute reference result using CPU implementation with INT8 normalization."""
        channels, batch_size, block_size, _ = sparse_values.shape
        result = np.zeros((batch_size, channels), dtype=np.float32)

        for led_id in range(batch_size):
            for channel_id in range(channels):
                # Get block position
                top_row = block_positions[channel_id, led_id, 0]
                top_col = block_positions[channel_id, led_id, 1]

                # Extract sparse block and target block
                sparse_block = sparse_values[channel_id, led_id].astype(np.int32)
                target_block = target_3d[
                    channel_id,
                    top_row : top_row + block_size,
                    top_col : top_col + block_size,
                ].astype(np.int32)

                # Compute dot product in integer arithmetic
                dot_product = np.sum(sparse_block * target_block)

                # Normalize by (255 * 255) to match kernel behavior
                result[led_id, channel_id] = float(dot_product) / (255.0 * 255.0)

        return result

    def test_kernel_compilation(self):
        """Test that both INT8 kernels compile successfully."""
        kernel_standard = get_compute_optimized_3d_int8_kernel()
        kernel_experimental = get_compute_optimized_3d_int8_experimental_kernel()

        assert kernel_standard is not None
        assert kernel_experimental is not None

    def test_int8_data_types(self):
        """Test that kernels handle UINT8 input correctly."""
        batch_size = 8
        sparse_values, block_positions, target_3d = self.generate_test_data_int8(batch_size)

        # Verify input data types
        assert sparse_values.dtype == np.uint8
        assert target_3d.dtype == np.uint8
        assert block_positions.dtype == np.int32

        # Convert to CuPy arrays
        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Test standard kernel
        result_gpu = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Verify output type and shape
        assert result_gpu.dtype == cp.float32, f"Expected FP32 output, got {result_gpu.dtype}"
        assert result_gpu.shape == (batch_size, self.channels)

    def test_int8_vs_reference_correctness(self):
        """Test INT8 kernel against CPU reference implementation."""
        batch_size = 16
        sparse_values, block_positions, target_3d = self.generate_test_data_int8(batch_size)

        # Convert to CuPy arrays
        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Compute GPU result
        result_gpu = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Compute reference result
        result_ref = self.compute_reference_result_int8(sparse_values, block_positions, target_3d)

        # Compare results
        result_cpu = cp.asnumpy(result_gpu)
        max_error = np.max(np.abs(result_cpu - result_ref))

        assert max_error < self.tolerance, f"Max error {max_error} exceeds tolerance {self.tolerance}"

    def test_int8_experimental_vs_standard(self):
        """Test experimental INT8 kernel against standard INT8 kernel."""
        batch_size = 32
        sparse_values, block_positions, target_3d = self.generate_test_data_int8(batch_size)

        # Ensure x-positions are aligned to 4 for experimental kernel
        for c in range(self.channels):
            for b in range(batch_size):
                block_positions[c, b, 1] = (block_positions[c, b, 1] // 4) * 4  # Align to 4

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Compute results with both kernels
        result_standard = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        result_experimental = cuda_transpose_dot_product_3d_compute_optimized_int8_experimental(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Compare results
        max_error = cp.max(cp.abs(result_standard - result_experimental))
        max_error_cpu = float(cp.asnumpy(max_error))

        assert max_error_cpu < 1e-6, f"Standard vs experimental error {max_error_cpu} too high"

    def test_int8_vs_fp32_equivalence(self):
        """Test that INT8 kernel produces equivalent results to FP32 kernel when normalized."""
        batch_size = 24

        # Generate FP32 data
        sparse_fp32, block_positions, target_fp32 = self.generate_test_data_fp32(batch_size)

        # Convert to INT8 by scaling [0,1] -> [0,255]
        sparse_int8 = (sparse_fp32 * 255.0).astype(np.uint8)
        target_int8 = (target_fp32 * 255.0).astype(np.uint8)

        # Convert to CuPy arrays
        sparse_fp32_gpu = cp.asarray(sparse_fp32)
        sparse_int8_gpu = cp.asarray(sparse_int8)
        block_positions_gpu = cp.asarray(block_positions)
        target_fp32_gpu = cp.asarray(target_fp32)
        target_int8_gpu = cp.asarray(target_int8)

        # Compute FP32 result
        result_fp32 = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_fp32_gpu,
            block_positions_gpu,
            target_fp32_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Compute INT8 result (already normalized)
        result_int8 = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_int8_gpu,
            block_positions_gpu,
            target_int8_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Compare results
        max_error = cp.max(cp.abs(result_fp32 - result_int8))
        relative_error = cp.max(cp.abs((result_fp32 - result_int8) / (result_fp32 + 1e-8)))

        max_error_cpu = float(cp.asnumpy(max_error))
        relative_error_cpu = float(cp.asnumpy(relative_error))

        # Allow for quantization error
        assert max_error_cpu < 0.01, f"Max error {max_error_cpu} too high for INT8 vs FP32"
        assert relative_error_cpu < 0.1, f"Relative error {relative_error_cpu} too high"

        logger.info(f"INT8 vs FP32 - Max error: {max_error_cpu:.6f}, Relative error: {relative_error_cpu:.6f}")

    def test_block_size_validation(self):
        """Test that kernels enforce block size constraints."""
        batch_size = 8
        sparse_values, block_positions, target_3d = self.generate_test_data_int8(batch_size)

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Test invalid block size (not multiple of 4)
        with pytest.raises(ValueError, match="block_size .* must be multiple of 4"):
            cuda_transpose_dot_product_3d_compute_optimized_int8(
                sparse_values_gpu,
                block_positions_gpu,
                target_3d_gpu,
                batch_size,
                self.channels,
                63,  # Invalid: 63 % 4 != 0
            )

    def test_input_dtype_validation(self):
        """Test that kernels validate input data types."""
        batch_size = 8
        sparse_values, block_positions, target_3d = self.generate_test_data_int8(batch_size)

        # Test with wrong sparse dtype
        sparse_wrong_dtype = sparse_values.astype(np.float32)
        sparse_wrong_gpu = cp.asarray(sparse_wrong_dtype)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        with pytest.raises(ValueError, match="sparse_values must be uint8"):
            cuda_transpose_dot_product_3d_compute_optimized_int8(
                sparse_wrong_gpu,
                block_positions_gpu,
                target_3d_gpu,
                batch_size,
                self.channels,
                self.block_size,
            )

        # Test with wrong target dtype
        target_wrong_dtype = target_3d.astype(np.float32)
        sparse_values_gpu = cp.asarray(sparse_values)
        target_wrong_gpu = cp.asarray(target_wrong_dtype)

        with pytest.raises(ValueError, match="target_3d must be uint8"):
            cuda_transpose_dot_product_3d_compute_optimized_int8(
                sparse_values_gpu,
                block_positions_gpu,
                target_wrong_gpu,
                batch_size,
                self.channels,
                self.block_size,
            )

    def test_zero_input_cases(self):
        """Test INT8 kernels with zero inputs."""
        batch_size = 16
        sparse_values, block_positions, target_3d = self.generate_test_data_int8(batch_size)

        # Test zero sparse values
        sparse_values[:] = 0

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        result_gpu = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        result_cpu = cp.asnumpy(result_gpu)
        assert np.allclose(result_cpu, 0.0, atol=1e-8)

    def test_max_value_inputs(self):
        """Test INT8 kernels with maximum value inputs."""
        batch_size = 16
        sparse_values, block_positions, target_3d = self.generate_test_data_int8(batch_size)

        # Set to maximum values
        sparse_values[:] = 255
        target_3d[:] = 255

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        result_gpu = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        result_cpu = cp.asnumpy(result_gpu)

        # Expected value: 255 * 255 * block_size^2 / (255 * 255) = block_size^2
        expected_value = self.block_size * self.block_size

        assert np.allclose(result_cpu, expected_value, rtol=1e-6)

    def test_performance_comparison(self):
        """Compare INT8 vs FP32 performance."""
        batch_size = 500

        # Generate INT8 data
        sparse_int8, block_positions, target_int8 = self.generate_test_data_int8(batch_size)

        # Generate equivalent FP32 data
        sparse_fp32 = sparse_int8.astype(np.float32) / 255.0
        target_fp32 = target_int8.astype(np.float32) / 255.0

        # Convert to GPU
        sparse_int8_gpu = cp.asarray(sparse_int8)
        sparse_fp32_gpu = cp.asarray(sparse_fp32)
        block_positions_gpu = cp.asarray(block_positions)
        target_int8_gpu = cp.asarray(target_int8)
        target_fp32_gpu = cp.asarray(target_fp32)

        # Warm-up
        for _ in range(3):
            _ = cuda_transpose_dot_product_3d_compute_optimized_int8(
                sparse_int8_gpu,
                block_positions_gpu,
                target_int8_gpu,
                batch_size,
                self.channels,
                self.block_size,
            )
        cp.cuda.Stream.null.synchronize()

        # Time INT8 kernel
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        result_int8 = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_int8_gpu,
            block_positions_gpu,
            target_int8_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )
        end_event.record()
        end_event.synchronize()

        elapsed_int8 = cp.cuda.get_elapsed_time(start_event, end_event)

        # Time FP32 kernel for comparison
        start_event.record()
        result_fp32 = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_fp32_gpu,
            block_positions_gpu,
            target_fp32_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )
        end_event.record()
        end_event.synchronize()

        elapsed_fp32 = cp.cuda.get_elapsed_time(start_event, end_event)

        # Verify results are reasonable
        assert result_int8.shape == (batch_size, self.channels)
        assert result_int8.dtype == cp.float32

        # Performance assertions
        assert elapsed_int8 < 100.0, f"INT8 kernel took {elapsed_int8:.2f}ms, too slow"

        # Compare memory usage
        int8_input_bytes = sparse_int8_gpu.nbytes + target_int8_gpu.nbytes
        fp32_input_bytes = sparse_fp32_gpu.nbytes + target_fp32_gpu.nbytes

        logger.info(f"Performance - INT8: {elapsed_int8:.2f}ms, FP32: {elapsed_fp32:.2f}ms")
        logger.info(f"Input memory - INT8: {int8_input_bytes} bytes, FP32: {fp32_input_bytes} bytes")
        logger.info(f"Memory reduction: {fp32_input_bytes / int8_input_bytes:.1f}x")

    def test_different_block_sizes(self):
        """Test INT8 kernels with different valid block sizes."""
        for block_size in [32, 64, 128]:
            if block_size % 4 != 0:
                continue

            batch_size = 16
            old_block_size = self.block_size
            self.block_size = block_size

            try:
                sparse_values, block_positions, target_3d = self.generate_test_data_int8(batch_size)

                sparse_values_gpu = cp.asarray(sparse_values)
                block_positions_gpu = cp.asarray(block_positions)
                target_3d_gpu = cp.asarray(target_3d)

                result_gpu = cuda_transpose_dot_product_3d_compute_optimized_int8(
                    sparse_values_gpu,
                    block_positions_gpu,
                    target_3d_gpu,
                    batch_size,
                    self.channels,
                    block_size,
                )

                result_ref = self.compute_reference_result_int8(sparse_values, block_positions, target_3d)
                result_cpu = cp.asnumpy(result_gpu)

                max_error = np.max(np.abs(result_cpu - result_ref))
                assert max_error < self.tolerance, f"Block size {block_size}: Max error {max_error}"

            finally:
                self.block_size = old_block_size
