#!/usr/bin/env python3
"""
Unit tests for DiagonalATAMatrix class.

Tests the DIA matrix implementation including:
- Basic initialization and matrix building
- FP16 storage with FP32 computation (mixed precision)
- Matrix-vector multiplication operations
- Serialization/deserialization
- Performance characteristics
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse as sp

cp = pytest.importorskip("cupy")

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix


@pytest.fixture(autouse=True)
def cuda_cleanup():
    """Ensure clean CUDA state before and after each test."""
    # Clear CUDA memory and reset state before test
    if cp.cuda.is_available():
        try:
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            # Clear any cached CUDA modules if available
            if hasattr(cp._core, "_kernel") and hasattr(cp._core._kernel, "clear_memo"):
                cp._core._kernel.clear_memo()
        except Exception:
            # If cleanup fails, continue with test
            pass

    yield  # Run the test

    # Clean up after test
    if cp.cuda.is_available():
        try:
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            # Clear any cached CUDA modules if available
            if hasattr(cp._core, "_kernel") and hasattr(cp._core._kernel, "clear_memo"):
                cp._core._kernel.clear_memo()
        except Exception:
            # If cleanup fails, don't fail the test
            pass


class TestDiagonalATAMatrix:
    """Test suite for DiagonalATAMatrix class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.led_count = 50
        self.crop_size = 64
        self.channels = 3
        self.pixels = 100

        # Create test diffusion matrix
        np.random.seed(42)  # For reproducible tests
        self.A_matrix = sp.random(self.pixels, self.led_count * self.channels, density=0.1, format="csc")

    def test_initialization_defaults(self):
        """Test default initialization."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size)

        assert matrix.led_count == self.led_count
        assert matrix.crop_size == self.crop_size
        assert matrix.channels == self.channels
        assert matrix.output_dtype == cp.float32
        assert matrix.storage_dtype == cp.float32
        assert matrix.dia_data_cpu is None
        assert matrix.dia_data_gpu is None
        assert matrix.k is None

    def test_initialization_with_custom_dtypes(self):
        """Test initialization with custom data types."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size, output_dtype=cp.float16, storage_dtype=cp.float16)

        assert matrix.output_dtype == cp.float16
        assert matrix.storage_dtype == cp.float16

    def test_initialization_invalid_dtypes(self):
        """Test initialization with invalid data types."""
        with pytest.raises(ValueError, match="Unsupported output dtype"):
            DiagonalATAMatrix(self.led_count, output_dtype=cp.int32)

        with pytest.raises(ValueError, match="Unsupported storage dtype"):
            DiagonalATAMatrix(self.led_count, storage_dtype=cp.int32)

    def test_build_from_diffusion_matrix_fp32(self):
        """Test building DIA matrix from diffusion matrix in FP32."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Check basic properties
        assert matrix.dia_data_cpu is not None
        assert matrix.dia_data_gpu is not None
        assert matrix.dia_offsets is not None
        assert matrix.k > 0

        # Check dimensions
        assert matrix.dia_data_cpu.shape == (self.channels, matrix.k, self.led_count)
        assert matrix.dia_data_gpu.shape == (self.channels, matrix.k, self.led_count)
        assert matrix.dia_offsets.shape == (matrix.k,)

        # Check data types
        assert matrix.dia_data_cpu.dtype == np.float32
        assert matrix.dia_data_gpu.dtype == cp.float32

        # Check metadata
        assert matrix.nnz > 0
        assert matrix.sparsity > 0
        assert matrix.bandwidth > 0

    def test_build_from_diffusion_matrix_fp16_storage(self):
        """Test building DIA matrix with FP16 storage."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size, output_dtype=cp.float32, storage_dtype=cp.float16)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Check basic properties
        assert matrix.dia_data_cpu is not None
        assert matrix.dia_data_gpu is not None
        assert matrix.k > 0

        # Check data types - storage should be FP16
        assert matrix.dia_data_cpu.dtype == np.float16
        assert matrix.dia_data_gpu.dtype == cp.float16

        # Check that FP16 cached version exists
        assert matrix.dia_data_gpu_fp16 is not None
        assert matrix.dia_data_gpu_fp16.dtype == cp.float16

    def test_multiply_3d_fp32(self):
        """Test 3D matrix-vector multiplication in FP32."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Create test LED values
        led_values = cp.random.rand(self.channels, self.led_count).astype(cp.float32)

        # Test multiplication
        result = matrix.multiply_3d(led_values)

        # Check result properties
        assert result.shape == (self.channels, self.led_count)
        assert result.dtype == cp.float32

        # Test with fallback implementation
        result_fallback = matrix.multiply_3d(led_values, use_custom_kernel=False)

        # Check that fallback also has correct properties
        assert result_fallback.shape == (self.channels, self.led_count)
        assert result_fallback.dtype == cp.float32

        # Test basic mathematical property: result should be finite
        assert np.all(np.isfinite(cp.asnumpy(result)))
        assert np.all(np.isfinite(cp.asnumpy(result_fallback)))

    def test_multiply_3d_mixed_precision(self):
        """Test 3D matrix-vector multiplication with mixed precision."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size, output_dtype=cp.float32, storage_dtype=cp.float16)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Create test LED values in FP32
        led_values = cp.random.rand(self.channels, self.led_count).astype(cp.float32)

        # Test multiplication - should work with FP16 storage but FP32 computation
        result = matrix.multiply_3d(led_values)

        # Check result properties
        assert result.shape == (self.channels, self.led_count)
        assert result.dtype == cp.float32

        # Test with fallback implementation
        result_fallback = matrix.multiply_3d(led_values, use_custom_kernel=False)

        # Check that fallback also has correct properties
        assert result_fallback.shape == (self.channels, self.led_count)
        assert result_fallback.dtype == cp.float32

        # Test basic mathematical property: result should be finite
        assert np.all(np.isfinite(cp.asnumpy(result)))
        assert np.all(np.isfinite(cp.asnumpy(result_fallback)))

    def test_kernel_vs_fallback_basic_properties(self):
        """Test that both custom kernel and fallback produce valid results."""
        # Use a simpler test matrix that should work consistently
        simple_led_count = 10
        simple_matrix = DiagonalATAMatrix(simple_led_count, self.crop_size)

        # Create a simple diffusion matrix
        np.random.seed(42)  # For reproducibility
        simple_A = sp.random(20, simple_led_count * 3, density=0.2, format="csc")
        simple_matrix.build_from_diffusion_matrix(simple_A)

        # Create test LED values
        led_values = cp.ones((self.channels, simple_led_count), dtype=cp.float32) * 0.5

        # Test with custom kernel if available
        try:
            result_custom = simple_matrix.multiply_3d(led_values, use_custom_kernel=True)

            # Check basic properties
            assert result_custom.shape == (self.channels, simple_led_count)
            assert result_custom.dtype == cp.float32
            assert np.all(np.isfinite(cp.asnumpy(result_custom)))

        except RuntimeError as e:
            if "Custom" in str(e):
                pytest.skip("Custom kernel not available for this test")
            else:
                raise

        # Test with fallback
        result_fallback = simple_matrix.multiply_3d(led_values, use_custom_kernel=False)

        # Check basic properties
        assert result_fallback.shape == (self.channels, simple_led_count)
        assert result_fallback.dtype == cp.float32
        assert np.all(np.isfinite(cp.asnumpy(result_fallback)))

    def test_multiply_3d_input_validation(self):
        """Test input validation for multiply_3d."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Test wrong shape
        with pytest.raises(ValueError, match="LED values should be shape"):
            wrong_shape = cp.random.rand(2, self.led_count).astype(cp.float32)
            matrix.multiply_3d(wrong_shape)

        # Test wrong input type
        with pytest.raises(TypeError, match="led_values must be a cupy.ndarray"):
            wrong_type = np.random.rand(self.channels, self.led_count).astype(np.float32)
            matrix.multiply_3d(wrong_type)

        # Test wrong dtype
        with pytest.raises(TypeError, match="led_values dtype must be"):
            wrong_dtype = cp.random.rand(self.channels, self.led_count).astype(cp.float16)
            matrix.multiply_3d(wrong_dtype)

    def test_g_ata_g_3d_fp32(self):
        """Test g^T (A^T A) g computation in FP32."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Create test gradient
        gradient = cp.random.rand(self.channels, self.led_count).astype(cp.float32)

        # Test computation
        result = matrix.g_ata_g_3d(gradient)

        # Check result properties
        assert result.shape == (self.channels,)
        assert result.dtype == cp.float32
        assert np.all(result >= 0)  # Should be positive (quadratic form)

    def test_g_ata_g_3d_mixed_precision(self):
        """Test g^T (A^T A) g computation with mixed precision."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size, output_dtype=cp.float32, storage_dtype=cp.float16)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Create test gradient
        gradient = cp.random.rand(self.channels, self.led_count).astype(cp.float32)

        # Test computation with mixed precision
        result = matrix.g_ata_g_3d(gradient)

        # Check result properties
        assert result.shape == (self.channels,)
        assert result.dtype == cp.float32
        assert np.all(result >= 0)  # Should be positive (quadratic form)

    def test_serialization_fp32(self):
        """Test serialization and deserialization in FP32."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Test dictionary conversion
        data_dict = matrix.to_dict()

        # Check keys
        expected_keys = {
            "led_count",
            "crop_size",
            "channels",
            "dia_data_3d",
            "dia_offsets_3d",
            "k",
            "bandwidth",
            "sparsity",
            "nnz",
            "channel_nnz",
            "output_dtype",
            "storage_dtype",
            "version",
        }
        assert set(data_dict.keys()) == expected_keys

        # Test reconstruction
        matrix_restored = DiagonalATAMatrix.from_dict(data_dict)

        # Check properties match
        assert matrix_restored.led_count == matrix.led_count
        assert matrix_restored.crop_size == matrix.crop_size
        assert matrix_restored.k == matrix.k
        assert matrix_restored.output_dtype == matrix.output_dtype
        assert matrix_restored.storage_dtype == matrix.storage_dtype

        # Check data matches
        assert np.allclose(matrix_restored.dia_data_cpu, matrix.dia_data_cpu)
        assert np.array_equal(matrix_restored.dia_offsets, matrix.dia_offsets)

    def test_serialization_fp16_storage(self):
        """Test serialization and deserialization with FP16 storage."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size, output_dtype=cp.float32, storage_dtype=cp.float16)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Test dictionary conversion
        data_dict = matrix.to_dict()

        # Check version updated for FP16 support
        assert data_dict["version"] == "8.1"
        assert data_dict["storage_dtype"] == "float16"
        assert data_dict["output_dtype"] == "float32"

        # Test reconstruction
        matrix_restored = DiagonalATAMatrix.from_dict(data_dict)

        # Check properties match
        assert matrix_restored.storage_dtype == cp.float16
        assert matrix_restored.output_dtype == cp.float32

        # Check data matches and has correct dtype
        assert np.allclose(matrix_restored.dia_data_cpu, matrix.dia_data_cpu)
        assert matrix_restored.dia_data_cpu.dtype == np.float16

    def test_save_load_fp32(self):
        """Test saving and loading to/from file in FP32."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Test saving
            matrix.save(tmp_path)

            # Test loading
            matrix_loaded = DiagonalATAMatrix.load(tmp_path)

            # Check properties match
            assert matrix_loaded.led_count == matrix.led_count
            assert matrix_loaded.k == matrix.k
            assert matrix_loaded.output_dtype == matrix.output_dtype
            assert matrix_loaded.storage_dtype == matrix.storage_dtype

            # Check data matches
            assert np.allclose(matrix_loaded.dia_data_cpu, matrix.dia_data_cpu)
            assert np.array_equal(matrix_loaded.dia_offsets, matrix.dia_offsets)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_load_fp16_storage(self):
        """Test saving and loading to/from file with FP16 storage."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size, output_dtype=cp.float32, storage_dtype=cp.float16)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Test saving
            matrix.save(tmp_path)

            # Test loading
            matrix_loaded = DiagonalATAMatrix.load(tmp_path)

            # Check properties match
            assert matrix_loaded.storage_dtype == cp.float16
            assert matrix_loaded.output_dtype == cp.float32
            assert matrix_loaded.dia_data_cpu.dtype == np.float16

            # Check data matches
            assert np.allclose(matrix_loaded.dia_data_cpu, matrix.dia_data_cpu)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_get_info(self):
        """Test getting matrix information."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size, output_dtype=cp.float32, storage_dtype=cp.float16)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        info = matrix.get_info()

        # Check info dictionary
        expected_keys = {
            "led_count",
            "crop_size",
            "channels",
            "bandwidth",
            "sparsity",
            "nnz",
            "channel_nnz",
            "ordering",
            "custom_kernel_available",
            "custom_kernel_fp16_available",
            "output_dtype",
            "storage_dtype",
            "unified_storage_built",
            "unified_k",
            "unified_storage_shape",
            "storage_format",
        }
        assert set(info.keys()) == expected_keys

        # Check values
        assert info["led_count"] == self.led_count
        assert info["output_dtype"] == str(cp.float32)
        assert info["storage_dtype"] == str(cp.float16)
        assert info["unified_storage_built"]
        assert info["storage_format"] == "unified_3d_dia_v8.1_mixed_precision"

    def test_get_channel_dia_matrix(self):
        """Test extracting single channel DIA matrix."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Test each channel
        for channel in range(self.channels):
            channel_matrix = matrix.get_channel_dia_matrix(channel)

            # Check properties
            assert isinstance(channel_matrix, sp.dia_matrix)
            assert channel_matrix.shape == (self.led_count, self.led_count)

        # Test invalid channel
        with pytest.raises(ValueError, match="Channel must be"):
            matrix.get_channel_dia_matrix(3)

    def test_backward_compatibility(self):
        """Test backward compatibility with older versions."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Create dictionary simulating older version
        data_dict = matrix.to_dict()
        data_dict["version"] = "7.0"

        # Remove new keys to simulate older version
        if "storage_dtype" in data_dict:
            del data_dict["storage_dtype"]

        # Should still load successfully
        matrix_loaded = DiagonalATAMatrix.from_dict(data_dict)
        assert matrix_loaded.led_count == matrix.led_count

    def test_memory_efficiency(self):
        """Test memory efficiency of FP16 vs FP32 storage."""
        # Create FP32 matrix
        matrix_fp32 = DiagonalATAMatrix(self.led_count, self.crop_size, storage_dtype=cp.float32)
        matrix_fp32.build_from_diffusion_matrix(self.A_matrix)

        # Create FP16 matrix
        matrix_fp16 = DiagonalATAMatrix(self.led_count, self.crop_size, storage_dtype=cp.float16)
        matrix_fp16.build_from_diffusion_matrix(self.A_matrix)

        # Compare memory usage
        fp32_memory = matrix_fp32.dia_data_cpu.nbytes
        fp16_memory = matrix_fp16.dia_data_cpu.nbytes

        # FP16 should use approximately half the memory
        assert fp16_memory < fp32_memory
        assert fp16_memory >= fp32_memory * 0.4  # Allow some overhead
        assert fp16_memory <= fp32_memory * 0.6  # Should be close to 50%

    def test_numerical_precision(self):
        """Test numerical precision of FP16 vs FP32."""
        # Create both matrices
        matrix_fp32 = DiagonalATAMatrix(self.led_count, self.crop_size, storage_dtype=cp.float32)
        matrix_fp32.build_from_diffusion_matrix(self.A_matrix)

        matrix_fp16 = DiagonalATAMatrix(
            self.led_count, self.crop_size, output_dtype=cp.float32, storage_dtype=cp.float16
        )
        matrix_fp16.build_from_diffusion_matrix(self.A_matrix)

        # Test with same input
        led_values = cp.random.rand(self.channels, self.led_count).astype(cp.float32)

        result_fp32 = matrix_fp32.multiply_3d(led_values, use_custom_kernel=False)
        result_fp16 = matrix_fp16.multiply_3d(led_values, use_custom_kernel=False)

        # Results should be close but not identical due to FP16 precision
        assert np.allclose(cp.asnumpy(result_fp32), cp.asnumpy(result_fp16), rtol=1e-3)

        # The difference should be small but measurable
        max_diff = np.max(np.abs(cp.asnumpy(result_fp32) - cp.asnumpy(result_fp16)))
        assert max_diff > 1e-6  # Should have some difference due to precision
        assert max_diff < 1e-2  # But not too large

    def test_gpu_memory_management(self):
        """Test GPU memory management."""
        matrix = DiagonalATAMatrix(self.led_count, self.crop_size, storage_dtype=cp.float16)
        matrix.build_from_diffusion_matrix(self.A_matrix)

        # Check GPU arrays exist
        assert matrix.dia_data_gpu is not None
        assert matrix.dia_data_gpu_fp16 is not None
        assert matrix.dia_offsets_gpu is not None

        # Check GPU memory types
        assert matrix.dia_data_gpu.dtype == cp.float16
        assert matrix.dia_data_gpu_fp16.dtype == cp.float16
        assert matrix.dia_offsets_gpu.dtype == cp.int32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
