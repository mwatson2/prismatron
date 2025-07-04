"""
Unit tests for compute_optimized_3d kernels (FP32 version).

Tests the FP32 compute-optimized 3D CUDA kernels for A^T @ b operations
with comprehensive validation against reference implementations.
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
        get_compute_optimized_3d_kernel,
    )

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestComputeOptimized3DKernel:
    """Test suite for FP32 compute-optimized 3D kernels."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tolerance = 1e-4  # FP32 tolerance
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
        """Test that the kernel compiles successfully."""
        kernel = get_compute_optimized_3d_kernel()
        assert kernel is not None

    def test_small_batch_correctness(self):
        """Test kernel correctness with small batch size."""
        batch_size = 8
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        # Convert to CuPy arrays
        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Compute GPU result
        result_gpu = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Compute reference result
        result_ref = self.compute_reference_result(sparse_values, block_positions, target_3d)

        # Compare results
        result_cpu = cp.asnumpy(result_gpu)
        max_error = np.max(np.abs(result_cpu - result_ref))

        assert max_error < self.tolerance, f"Max error {max_error} exceeds tolerance {self.tolerance}"
        assert result_cpu.shape == result_ref.shape, f"Shape mismatch: {result_cpu.shape} vs {result_ref.shape}"
        assert result_cpu.dtype == np.float32, f"Unexpected dtype: {result_cpu.dtype}"

    def test_medium_batch_correctness(self):
        """Test kernel correctness with medium batch size."""
        batch_size = 128
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        # Convert to CuPy arrays
        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Compute GPU result
        result_gpu = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        # Compute reference result
        result_ref = self.compute_reference_result(sparse_values, block_positions, target_3d)

        # Compare results
        result_cpu = cp.asnumpy(result_gpu)
        max_error = np.max(np.abs(result_cpu - result_ref))

        assert max_error < self.tolerance, f"Max error {max_error} exceeds tolerance {self.tolerance}"

    def test_edge_cases(self):
        """Test kernel with edge cases."""
        # Test single LED
        batch_size = 1
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        result_gpu = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        result_ref = self.compute_reference_result(sparse_values, block_positions, target_3d)
        result_cpu = cp.asnumpy(result_gpu)

        max_error = np.max(np.abs(result_cpu - result_ref))
        assert max_error < self.tolerance

    def test_zero_sparse_values(self):
        """Test kernel with zero sparse values."""
        batch_size = 16
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        # Set sparse values to zero
        sparse_values[:] = 0.0

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        result_gpu = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        result_cpu = cp.asnumpy(result_gpu)

        # Result should be all zeros
        assert np.allclose(result_cpu, 0.0, atol=self.tolerance)

    def test_zero_target_values(self):
        """Test kernel with zero target values."""
        batch_size = 16
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        # Set target values to zero
        target_3d[:] = 0.0

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        result_gpu = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )

        result_cpu = cp.asnumpy(result_gpu)

        # Result should be all zeros
        assert np.allclose(result_cpu, 0.0, atol=self.tolerance)

    def test_input_validation(self):
        """Test kernel input validation."""
        batch_size = 8
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Test mismatched channels
        with pytest.raises(ValueError, match="Input channels .* != expected"):
            wrong_target = cp.zeros((2, self.height, self.width), dtype=cp.float32)  # Wrong channel count
            cuda_transpose_dot_product_3d_compute_optimized(
                sparse_values_gpu,
                block_positions_gpu,
                wrong_target,
                batch_size,
                self.channels,
                self.block_size,
            )

    def test_performance_benchmarking(self):
        """Basic performance test to ensure kernel runs efficiently."""
        batch_size = 500  # Realistic batch size
        sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

        sparse_values_gpu = cp.asarray(sparse_values)
        block_positions_gpu = cp.asarray(block_positions)
        target_3d_gpu = cp.asarray(target_3d)

        # Warm-up
        for _ in range(3):
            _ = cuda_transpose_dot_product_3d_compute_optimized(
                sparse_values_gpu,
                block_positions_gpu,
                target_3d_gpu,
                batch_size,
                self.channels,
                self.block_size,
            )
        cp.cuda.Stream.null.synchronize()

        # Timing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        result_gpu = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_values_gpu,
            block_positions_gpu,
            target_3d_gpu,
            batch_size,
            self.channels,
            self.block_size,
        )
        end_event.record()
        end_event.synchronize()

        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        # Verify result is correct
        assert result_gpu.shape == (batch_size, self.channels)
        assert result_gpu.dtype == cp.float32

        # Performance assertion - should complete within reasonable time
        assert elapsed_ms < 100.0, f"Kernel took {elapsed_ms:.2f}ms, too slow"

        logger.info(f"Kernel performance: {elapsed_ms:.2f}ms for {batch_size} LEDs")

    def test_different_block_sizes(self):
        """Test kernel with different block sizes."""
        for block_size in [32, 64, 128]:
            if block_size % 32 != 0:
                continue  # Skip invalid block sizes

            self.block_size = block_size
            batch_size = 16

            sparse_values, block_positions, target_3d = self.generate_test_data(batch_size)

            sparse_values_gpu = cp.asarray(sparse_values)
            block_positions_gpu = cp.asarray(block_positions)
            target_3d_gpu = cp.asarray(target_3d)

            result_gpu = cuda_transpose_dot_product_3d_compute_optimized(
                sparse_values_gpu,
                block_positions_gpu,
                target_3d_gpu,
                batch_size,
                self.channels,
                block_size,
            )

            result_ref = self.compute_reference_result(sparse_values, block_positions, target_3d)
            result_cpu = cp.asnumpy(result_gpu)

            max_error = np.max(np.abs(result_cpu - result_ref))
            assert max_error < self.tolerance, f"Block size {block_size}: Max error {max_error}"
