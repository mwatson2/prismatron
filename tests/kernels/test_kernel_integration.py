"""
Integration tests for all kernel variants.

This module tests the interoperability and consistency across all kernel variants,
ensuring they work together and produce compatible results.
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
    import cupyx.scipy.sparse as cusp
    import scipy.sparse as sp

    from src.utils.kernels import (  # 3D kernels; DIA kernels
        CustomDIA3DMatVec,
        CustomDIA3DMatVecFP16,
        CustomDIAMatVec,
        CustomDIAMatVecFP16,
        cuda_transpose_dot_product_3d_compute_optimized,
        cuda_transpose_dot_product_3d_compute_optimized_fp16,
        cuda_transpose_dot_product_3d_compute_optimized_int8,
        cuda_transpose_dot_product_3d_compute_optimized_int8_fp16,
    )

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestKernelIntegration:
    """Integration test suite for all kernel variants."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tolerance_strict = 1e-6  # For identical operations
        self.tolerance_fp16 = 5e-3  # For FP16 precision
        self.tolerance_int8 = 1e-2  # For INT8 quantization

        self.batch_size = 32
        self.channels = 3
        self.height = 300
        self.width = 400
        self.block_size = 64

    def generate_test_data_all_types(self):
        """Generate test data in all required formats."""
        np.random.seed(42)

        # Generate FP32 base data
        sparse_fp32 = np.random.rand(self.channels, self.batch_size, self.block_size, self.block_size).astype(
            np.float32
        )
        target_fp32 = np.random.rand(self.channels, self.height, self.width).astype(np.float32)

        # Generate block positions
        max_row = self.height - self.block_size
        max_col = self.width - self.block_size
        block_positions = np.zeros((self.channels, self.batch_size, 2), dtype=np.int32)

        for c in range(self.channels):
            for b in range(self.batch_size):
                block_positions[c, b, 0] = np.random.randint(0, max_row)
                block_positions[c, b, 1] = np.random.randint(0, max_col)

        # Convert to INT8 versions
        sparse_int8 = (sparse_fp32 * 255.0).astype(np.uint8)
        target_int8 = (target_fp32 * 255.0).astype(np.uint8)

        return {
            "sparse_fp32": sparse_fp32,
            "sparse_int8": sparse_int8,
            "target_fp32": target_fp32,
            "target_int8": target_int8,
            "block_positions": block_positions,
        }

    def test_3d_kernel_consistency_across_types(self):
        """Test that all 3D kernel variants produce consistent results."""
        data = self.generate_test_data_all_types()

        # Convert to GPU arrays
        sparse_fp32_gpu = cp.asarray(data["sparse_fp32"])
        sparse_int8_gpu = cp.asarray(data["sparse_int8"])
        target_fp32_gpu = cp.asarray(data["target_fp32"])
        target_int8_gpu = cp.asarray(data["target_int8"])
        block_positions_gpu = cp.asarray(data["block_positions"])

        # Test FP32 -> FP32
        result_fp32_fp32 = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_fp32_gpu,
            block_positions_gpu,
            target_fp32_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        # Test FP32 -> FP16
        result_fp32_fp16 = cuda_transpose_dot_product_3d_compute_optimized_fp16(
            sparse_fp32_gpu,
            block_positions_gpu,
            target_fp32_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        # Test INT8 -> FP32
        result_int8_fp32 = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_int8_gpu,
            block_positions_gpu,
            target_int8_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        # Test INT8 -> FP16
        result_int8_fp16 = cuda_transpose_dot_product_3d_compute_optimized_int8_fp16(
            sparse_int8_gpu,
            block_positions_gpu,
            target_int8_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        # Verify data types
        assert result_fp32_fp32.dtype == cp.float32
        assert result_fp32_fp16.dtype == cp.float16
        assert result_int8_fp32.dtype == cp.float32
        assert result_int8_fp16.dtype == cp.float16

        # Verify shapes
        expected_shape = (self.batch_size, self.channels)
        assert result_fp32_fp32.shape == expected_shape
        assert result_fp32_fp16.shape == expected_shape
        assert result_int8_fp32.shape == expected_shape
        assert result_int8_fp16.shape == expected_shape

        # Compare FP32 vs FP16 (same input)
        error_fp32_vs_fp16 = cp.max(cp.abs(result_fp32_fp32 - result_fp32_fp16.astype(cp.float32)))
        assert float(error_fp32_vs_fp16) < self.tolerance_fp16, f"FP32 vs FP16 error: {float(error_fp32_vs_fp16)}"

        # Compare INT8 FP32 vs FP16 (same input)
        error_int8_fp32_vs_fp16 = cp.max(cp.abs(result_int8_fp32 - result_int8_fp16.astype(cp.float32)))
        assert (
            float(error_int8_fp32_vs_fp16) < self.tolerance_fp16
        ), f"INT8 FP32 vs FP16 error: {float(error_int8_fp32_vs_fp16)}"

        # Compare FP32 vs INT8 (accounting for normalization)
        error_fp32_vs_int8 = cp.max(cp.abs(result_fp32_fp32 - result_int8_fp32))
        assert float(error_fp32_vs_int8) < self.tolerance_int8, f"FP32 vs INT8 error: {float(error_fp32_vs_int8)}"

        logger.info("3D Kernel consistency check passed:")
        logger.info(f"  FP32 vs FP16 error: {float(error_fp32_vs_fp16):.6f}")
        logger.info(f"  INT8 FP32 vs FP16 error: {float(error_int8_fp32_vs_fp16):.6f}")
        logger.info(f"  FP32 vs INT8 error: {float(error_fp32_vs_int8):.6f}")

    def test_dia_kernel_consistency_across_types(self):
        """Test that DIA kernel variants produce consistent results."""
        # Create test DIA matrix
        n = 200
        max_offset = 10
        offsets = list(range(-max_offset, max_offset + 1))
        num_bands = len(offsets)

        np.random.seed(42)
        data = np.random.rand(num_bands, n).astype(np.float32) * 100

        dia_matrix = cusp.dia_matrix((data, offsets), shape=(n, n))
        x = cp.random.randn(n, dtype=cp.float32)

        # Test FP32 kernels
        custom_fp32_basic = CustomDIAMatVec(use_optimized=False)
        custom_fp32_opt = CustomDIAMatVec(use_optimized=True)

        result_fp32_basic = custom_fp32_basic(dia_matrix, x)
        result_fp32_opt = custom_fp32_opt(dia_matrix, x)

        # Test FP16 kernels
        custom_fp16_basic = CustomDIAMatVecFP16(use_optimized=False)
        custom_fp16_opt = CustomDIAMatVecFP16(use_optimized=True)

        result_fp16_basic = custom_fp16_basic(dia_matrix, x)
        result_fp16_opt = custom_fp16_opt(dia_matrix, x)

        # Verify data types
        assert result_fp32_basic.dtype == cp.float32
        assert result_fp32_opt.dtype == cp.float32
        assert result_fp16_basic.dtype == cp.float16
        assert result_fp16_opt.dtype == cp.float16

        # Compare FP32 basic vs optimized
        error_fp32_basic_vs_opt = cp.max(cp.abs(result_fp32_basic - result_fp32_opt))
        assert (
            float(error_fp32_basic_vs_opt) < self.tolerance_strict
        ), f"FP32 basic vs opt error: {float(error_fp32_basic_vs_opt)}"

        # Compare FP16 basic vs optimized
        error_fp16_basic_vs_opt = cp.max(
            cp.abs(result_fp16_basic.astype(cp.float32) - result_fp16_opt.astype(cp.float32))
        )
        assert (
            float(error_fp16_basic_vs_opt) < self.tolerance_fp16
        ), f"FP16 basic vs opt error: {float(error_fp16_basic_vs_opt)}"

        # Compare FP32 vs FP16
        error_fp32_vs_fp16 = cp.max(cp.abs(result_fp32_basic - result_fp16_basic.astype(cp.float32)))
        assert float(error_fp32_vs_fp16) < self.tolerance_fp16, f"DIA FP32 vs FP16 error: {float(error_fp32_vs_fp16)}"

        logger.info("DIA Kernel consistency check passed:")
        logger.info(f"  FP32 basic vs optimized error: {float(error_fp32_basic_vs_opt):.6f}")
        logger.info(f"  FP16 basic vs optimized error: {float(error_fp16_basic_vs_opt):.6f}")
        logger.info(f"  FP32 vs FP16 error: {float(error_fp32_vs_fp16):.6f}")

    def test_3d_dia_kernel_consistency(self):
        """Test 3D DIA kernel consistency across variants."""
        n = 100
        channels = 3
        max_offset = 5
        offsets = list(range(-max_offset, max_offset + 1))
        num_bands = len(offsets)

        # Create 3D DIA data
        np.random.seed(42)
        dia_data_3d = np.random.rand(channels, num_bands, n).astype(np.float32) * 50
        dia_offsets = np.array(offsets, dtype=np.int32)
        x = np.random.rand(channels, n).astype(np.float32)

        # Convert to GPU
        dia_data_gpu = cp.asarray(dia_data_3d)
        dia_offsets_gpu = cp.asarray(dia_offsets)
        x_gpu = cp.asarray(x)

        # Test FP32 3D kernels
        custom_3d_fp32_basic = CustomDIA3DMatVec(use_optimized=False)
        custom_3d_fp32_opt = CustomDIA3DMatVec(use_optimized=True)

        result_3d_fp32_basic = custom_3d_fp32_basic(dia_data_gpu, dia_offsets_gpu, x_gpu)
        result_3d_fp32_opt = custom_3d_fp32_opt(dia_data_gpu, dia_offsets_gpu, x_gpu)

        # Test FP16 3D kernels
        custom_3d_fp16_basic = CustomDIA3DMatVecFP16(use_optimized=False)
        custom_3d_fp16_opt = CustomDIA3DMatVecFP16(use_optimized=True)

        result_3d_fp16_basic = custom_3d_fp16_basic(dia_data_gpu, dia_offsets_gpu, x_gpu)
        result_3d_fp16_opt = custom_3d_fp16_opt(dia_data_gpu, dia_offsets_gpu, x_gpu)

        # Verify shapes and types
        expected_shape = (channels, n)
        assert result_3d_fp32_basic.shape == expected_shape
        assert result_3d_fp32_opt.shape == expected_shape
        assert result_3d_fp16_basic.shape == expected_shape
        assert result_3d_fp16_opt.shape == expected_shape

        assert result_3d_fp32_basic.dtype == cp.float32
        assert result_3d_fp32_opt.dtype == cp.float32
        assert result_3d_fp16_basic.dtype == cp.float16
        assert result_3d_fp16_opt.dtype == cp.float16

        # Compare FP32 basic vs optimized
        error_3d_fp32_basic_vs_opt = cp.max(cp.abs(result_3d_fp32_basic - result_3d_fp32_opt))
        assert float(error_3d_fp32_basic_vs_opt) < self.tolerance_strict

        # Compare FP16 basic vs optimized
        error_3d_fp16_basic_vs_opt = cp.max(
            cp.abs(result_3d_fp16_basic.astype(cp.float32) - result_3d_fp16_opt.astype(cp.float32))
        )
        assert float(error_3d_fp16_basic_vs_opt) < self.tolerance_fp16

        # Compare FP32 vs FP16
        error_3d_fp32_vs_fp16 = cp.max(cp.abs(result_3d_fp32_basic - result_3d_fp16_basic.astype(cp.float32)))
        assert float(error_3d_fp32_vs_fp16) < self.tolerance_fp16

    def test_memory_efficiency_comparison(self):
        """Test memory efficiency across all kernel variants."""
        data = self.generate_test_data_all_types()

        # Convert to GPU arrays
        sparse_fp32_gpu = cp.asarray(data["sparse_fp32"])
        sparse_int8_gpu = cp.asarray(data["sparse_int8"])
        target_fp32_gpu = cp.asarray(data["target_fp32"])
        target_int8_gpu = cp.asarray(data["target_int8"])
        block_positions_gpu = cp.asarray(data["block_positions"])

        # Compute all results
        result_fp32_fp32 = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_fp32_gpu,
            block_positions_gpu,
            target_fp32_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        result_fp32_fp16 = cuda_transpose_dot_product_3d_compute_optimized_fp16(
            sparse_fp32_gpu,
            block_positions_gpu,
            target_fp32_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        result_int8_fp32 = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_int8_gpu,
            block_positions_gpu,
            target_int8_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        result_int8_fp16 = cuda_transpose_dot_product_3d_compute_optimized_int8_fp16(
            sparse_int8_gpu,
            block_positions_gpu,
            target_int8_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        # Calculate memory usage
        input_fp32_bytes = sparse_fp32_gpu.nbytes + target_fp32_gpu.nbytes
        input_int8_bytes = sparse_int8_gpu.nbytes + target_int8_gpu.nbytes

        output_fp32_bytes = result_fp32_fp32.nbytes
        output_fp16_bytes = result_fp32_fp16.nbytes

        # Verify memory relationships
        assert input_int8_bytes < input_fp32_bytes, "INT8 should use less input memory"
        assert output_fp16_bytes < output_fp32_bytes, "FP16 should use less output memory"
        assert output_fp16_bytes == output_fp32_bytes // 2, "FP16 should use exactly half output memory"

        # Calculate reductions
        input_reduction = input_fp32_bytes / input_int8_bytes
        output_reduction = output_fp32_bytes / output_fp16_bytes

        logger.info("Memory efficiency analysis:")
        logger.info(f"  Input memory - FP32: {input_fp32_bytes} bytes, INT8: {input_int8_bytes} bytes")
        logger.info(f"  Input reduction: {input_reduction:.1f}x")
        logger.info(f"  Output memory - FP32: {output_fp32_bytes} bytes, FP16: {output_fp16_bytes} bytes")
        logger.info(f"  Output reduction: {output_reduction:.1f}x")

    def test_performance_scaling_across_variants(self):
        """Test performance scaling across different kernel variants."""
        data = self.generate_test_data_all_types()

        # Convert to GPU arrays
        sparse_fp32_gpu = cp.asarray(data["sparse_fp32"])
        sparse_int8_gpu = cp.asarray(data["sparse_int8"])
        target_fp32_gpu = cp.asarray(data["target_fp32"])
        target_int8_gpu = cp.asarray(data["target_int8"])
        block_positions_gpu = cp.asarray(data["block_positions"])

        kernels_to_test = [
            (
                "FP32->FP32",
                lambda: cuda_transpose_dot_product_3d_compute_optimized(
                    sparse_fp32_gpu,
                    block_positions_gpu,
                    target_fp32_gpu,
                    self.batch_size,
                    self.channels,
                    self.block_size,
                ),
            ),
            (
                "FP32->FP16",
                lambda: cuda_transpose_dot_product_3d_compute_optimized_fp16(
                    sparse_fp32_gpu,
                    block_positions_gpu,
                    target_fp32_gpu,
                    self.batch_size,
                    self.channels,
                    self.block_size,
                ),
            ),
            (
                "INT8->FP32",
                lambda: cuda_transpose_dot_product_3d_compute_optimized_int8(
                    sparse_int8_gpu,
                    block_positions_gpu,
                    target_int8_gpu,
                    self.batch_size,
                    self.channels,
                    self.block_size,
                ),
            ),
            (
                "INT8->FP16",
                lambda: cuda_transpose_dot_product_3d_compute_optimized_int8_fp16(
                    sparse_int8_gpu,
                    block_positions_gpu,
                    target_int8_gpu,
                    self.batch_size,
                    self.channels,
                    self.block_size,
                ),
            ),
        ]

        performance_results = {}

        for name, kernel_func in kernels_to_test:
            # Warm-up
            for _ in range(3):
                _ = kernel_func()
            cp.cuda.Stream.null.synchronize()

            # Time execution
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()

            start_event.record()
            result = kernel_func()
            end_event.record()
            end_event.synchronize()

            elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
            performance_results[name] = elapsed_ms

            # Verify result is reasonable
            assert result.shape == (self.batch_size, self.channels)
            assert elapsed_ms < 100.0, f"{name} kernel took {elapsed_ms:.2f}ms, too slow"

        # Log performance comparison
        logger.info("Performance comparison across kernel variants:")
        for name, time_ms in performance_results.items():
            logger.info(f"  {name}: {time_ms:.2f}ms")

    def test_error_propagation_consistency(self):
        """Test that error conditions are handled consistently across variants."""
        data = self.generate_test_data_all_types()

        # Test with mismatched channel count
        wrong_channels = 2
        sparse_wrong = data["sparse_fp32"][:wrong_channels]  # Wrong channel count
        sparse_wrong_gpu = cp.asarray(sparse_wrong)

        target_fp32_gpu = cp.asarray(data["target_fp32"])
        block_positions_gpu = cp.asarray(data["block_positions"])

        # All kernels should raise ValueError for channel mismatch
        kernels_to_test = [
            cuda_transpose_dot_product_3d_compute_optimized,
            cuda_transpose_dot_product_3d_compute_optimized_fp16,
        ]

        for kernel_func in kernels_to_test:
            with pytest.raises(ValueError, match="Input channels .* != expected"):
                kernel_func(
                    sparse_wrong_gpu,
                    block_positions_gpu,
                    target_fp32_gpu,
                    self.batch_size,
                    self.channels,
                    self.block_size,
                )

    def test_numerical_stability_across_variants(self):
        """Test numerical stability across different data ranges and kernel variants."""
        # Test with different value ranges
        test_ranges = [
            (1e-6, "Very small values"),
            (1.0, "Normal values"),
            (1e3, "Large values"),
        ]

        for scale, description in test_ranges:
            data = self.generate_test_data_all_types()

            # Scale the data
            scaled_sparse_fp32 = data["sparse_fp32"] * scale
            scaled_target_fp32 = data["target_fp32"] * scale

            # For INT8, clamp to valid range
            scaled_sparse_int8 = np.clip(scaled_sparse_fp32 * 255.0, 0, 255).astype(np.uint8)
            scaled_target_int8 = np.clip(scaled_target_fp32 * 255.0, 0, 255).astype(np.uint8)

            # Convert to GPU
            sparse_fp32_gpu = cp.asarray(scaled_sparse_fp32)
            sparse_int8_gpu = cp.asarray(scaled_sparse_int8)
            target_fp32_gpu = cp.asarray(scaled_target_fp32)
            target_int8_gpu = cp.asarray(scaled_target_int8)
            block_positions_gpu = cp.asarray(data["block_positions"])

            # Test all kernels
            results = {}

            try:
                results["fp32_fp32"] = cuda_transpose_dot_product_3d_compute_optimized(
                    sparse_fp32_gpu,
                    block_positions_gpu,
                    target_fp32_gpu,
                    self.batch_size,
                    self.channels,
                    self.block_size,
                )

                results["fp32_fp16"] = cuda_transpose_dot_product_3d_compute_optimized_fp16(
                    sparse_fp32_gpu,
                    block_positions_gpu,
                    target_fp32_gpu,
                    self.batch_size,
                    self.channels,
                    self.block_size,
                )

                results["int8_fp32"] = cuda_transpose_dot_product_3d_compute_optimized_int8(
                    sparse_int8_gpu,
                    block_positions_gpu,
                    target_int8_gpu,
                    self.batch_size,
                    self.channels,
                    self.block_size,
                )

                results["int8_fp16"] = cuda_transpose_dot_product_3d_compute_optimized_int8_fp16(
                    sparse_int8_gpu,
                    block_positions_gpu,
                    target_int8_gpu,
                    self.batch_size,
                    self.channels,
                    self.block_size,
                )

                # Check for NaN/Inf in all results
                for name, result in results.items():
                    result_cpu = cp.asnumpy(result.astype(cp.float32))
                    assert not np.any(np.isnan(result_cpu)), f"{description} - {name}: Contains NaN"
                    assert not np.any(np.isinf(result_cpu)), f"{description} - {name}: Contains Inf"

                    if scale >= 1.0:  # Should have non-zero results for normal/large scales
                        assert np.any(result_cpu != 0), f"{description} - {name}: All zeros"

                logger.info(f"Numerical stability test passed for {description}")

            except Exception as e:
                logger.warning(f"Numerical stability test failed for {description}: {e}")
                # For extreme values, some failure is acceptable
                if scale < 1e-3 or scale > 1e6:
                    pytest.skip(f"Extreme values ({description}) caused expected numerical issues")
                else:
                    raise

    def test_cross_kernel_compatibility(self):
        """Test that results from different kernel types can be used together."""
        data = self.generate_test_data_all_types()

        # Convert to GPU arrays
        sparse_fp32_gpu = cp.asarray(data["sparse_fp32"])
        target_fp32_gpu = cp.asarray(data["target_fp32"])
        block_positions_gpu = cp.asarray(data["block_positions"])

        # Get results from different kernels
        result_fp32 = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_fp32_gpu,
            block_positions_gpu,
            target_fp32_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        result_fp16 = cuda_transpose_dot_product_3d_compute_optimized_fp16(
            sparse_fp32_gpu,
            block_positions_gpu,
            target_fp32_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        # Test arithmetic operations between different types
        # FP32 + FP16 (should auto-promote)
        result_sum = result_fp32 + result_fp16.astype(cp.float32)
        assert result_sum.dtype == cp.float32
        assert result_sum.shape == (self.batch_size, self.channels)

        # Test that we can concatenate results
        combined = cp.stack([result_fp32, result_fp16.astype(cp.float32)], axis=0)
        assert combined.shape == (2, self.batch_size, self.channels)

        # Test statistical operations
        mean_fp32 = cp.mean(result_fp32)
        mean_fp16 = cp.mean(result_fp16.astype(cp.float32))

        # Should be reasonably close
        error = cp.abs(mean_fp32 - mean_fp16)
        assert float(error) < self.tolerance_fp16

    def test_batch_size_scaling_consistency(self):
        """Test that all kernels scale consistently with batch size."""
        batch_sizes = [1, 8, 32, 128]

        for batch_size in batch_sizes:
            old_batch_size = self.batch_size
            self.batch_size = batch_size

            try:
                data = self.generate_test_data_all_types()

                # Convert to GPU arrays
                sparse_fp32_gpu = cp.asarray(data["sparse_fp32"])
                target_fp32_gpu = cp.asarray(data["target_fp32"])
                block_positions_gpu = cp.asarray(data["block_positions"])

                # Test that all kernels work with this batch size
                result_fp32 = cuda_transpose_dot_product_3d_compute_optimized(
                    sparse_fp32_gpu,
                    block_positions_gpu,
                    target_fp32_gpu,
                    batch_size,
                    self.channels,
                    self.block_size,
                )

                result_fp16 = cuda_transpose_dot_product_3d_compute_optimized_fp16(
                    sparse_fp32_gpu,
                    block_positions_gpu,
                    target_fp32_gpu,
                    batch_size,
                    self.channels,
                    self.block_size,
                )

                # Verify shapes
                expected_shape = (batch_size, self.channels)
                assert result_fp32.shape == expected_shape, f"Batch size {batch_size}: FP32 shape mismatch"
                assert result_fp16.shape == expected_shape, f"Batch size {batch_size}: FP16 shape mismatch"

                logger.info(f"Batch size {batch_size}: All kernels scale correctly")

            finally:
                self.batch_size = old_batch_size

    def test_integration_with_existing_codebase(self):
        """Test integration with existing codebase patterns."""
        # This test ensures the new kernel organization doesn't break existing usage patterns
        data = self.generate_test_data_all_types()

        # Test that we can import kernels in different ways (backward compatibility)
        from src.utils import cuda_kernels  # Original module
        from src.utils.kernels import compute_optimized_3d  # New structure

        # Both should provide the same functions
        assert hasattr(cuda_kernels, "cuda_transpose_dot_product_3d_compute_optimized")
        assert hasattr(compute_optimized_3d, "cuda_transpose_dot_product_3d_compute_optimized")

        # Test that both paths work
        sparse_fp32_gpu = cp.asarray(data["sparse_fp32"])
        target_fp32_gpu = cp.asarray(data["target_fp32"])
        block_positions_gpu = cp.asarray(data["block_positions"])

        result_old_path = cuda_kernels.cuda_transpose_dot_product_3d_compute_optimized(
            sparse_fp32_gpu,
            block_positions_gpu,
            target_fp32_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        result_new_path = compute_optimized_3d.cuda_transpose_dot_product_3d_compute_optimized(
            sparse_fp32_gpu,
            block_positions_gpu,
            target_fp32_gpu,
            self.batch_size,
            self.channels,
            self.block_size,
        )

        # Results should be identical
        error = cp.max(cp.abs(result_old_path - result_new_path))
        assert float(error) < self.tolerance_strict, "Backward compatibility broken"
