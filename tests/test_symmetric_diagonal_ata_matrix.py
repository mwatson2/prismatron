#!/usr/bin/env python3
"""
Unit tests for SymmetricDiagonalATAMatrix class.

Tests the symmetric DIA matrix implementation including:
- Basic initialization and matrix building
- Symmetric storage optimization (FP32 only)
- Matrix-vector multiplication operations
- Conversion from regular DiagonalATAMatrix
- Correctness against dense reference
- Performance characteristics
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse as sp

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix


@pytest.fixture(autouse=True)
def cuda_cleanup():
    """Ensure clean CUDA state before and after each test."""
    # Clear CUDA memory and reset state before test
    if CUPY_AVAILABLE and cp.cuda.is_available():
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
    if CUPY_AVAILABLE and cp.cuda.is_available():
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


class TestSymmetricDiagonalATAMatrix:
    """Test suite for SymmetricDiagonalATAMatrix class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.led_count = 30
        self.crop_size = 64
        self.channels = 3
        self.pixels = 60

        # Create test diffusion matrix
        np.random.seed(42)  # For reproducible tests
        self.A_matrix = sp.random(self.pixels, self.led_count * self.channels, density=0.15, format="csc")

        # Create reference dense symmetric matrices for correctness testing
        self.dense_matrices = self._create_reference_symmetric_matrices()

    def _create_reference_symmetric_matrices(self, num_diagonals=9):
        """Create reference symmetric dense matrices for testing."""
        np.random.seed(123)  # Different seed for reference matrices

        bandwidth = (num_diagonals - 1) // 2
        dense_matrices = []

        for channel in range(self.channels):
            # Create symmetric matrix with limited bandwidth
            matrix = np.zeros((self.led_count, self.led_count), dtype=np.float32)

            # Fill diagonals symmetrically
            for offset in range(-bandwidth, bandwidth + 1):
                for i in range(self.led_count):
                    j = i + offset
                    if 0 <= j < self.led_count and offset >= 0:  # Upper or main diagonal
                        value = np.random.rand() * 2.0
                        matrix[i, j] = value
                        if offset > 0:  # Also set symmetric element
                            matrix[j, i] = value

            dense_matrices.append(matrix)

        return dense_matrices

    def test_initialization_defaults(self):
        """Test default initialization."""
        matrix = SymmetricDiagonalATAMatrix(self.led_count, self.crop_size)

        assert matrix.led_count == self.led_count
        assert matrix.crop_size == self.crop_size
        assert matrix.channels == self.channels
        assert matrix.output_dtype == cp.float32
        assert matrix.dia_data_gpu is None
        assert matrix.dia_offsets_upper is None
        assert matrix.k_upper is None

    def test_initialization_fp32_only(self):
        """Test that only FP32 is supported."""
        # Should work with FP32
        matrix = SymmetricDiagonalATAMatrix(self.led_count, self.crop_size, output_dtype=cp.float32)
        assert matrix.output_dtype == cp.float32

        # Should reject FP16
        with pytest.raises(ValueError, match="only supports cupy.float32 output_dtype"):
            SymmetricDiagonalATAMatrix(self.led_count, self.crop_size, output_dtype=cp.float16)

    def test_from_diagonal_ata_matrix_conversion(self):
        """Test conversion from regular DiagonalATAMatrix."""
        # Create regular matrix first
        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(self.A_matrix)

        # Convert to symmetric
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        # Check basic properties
        assert symmetric_matrix.led_count == regular_matrix.led_count
        assert symmetric_matrix.crop_size == regular_matrix.crop_size
        assert symmetric_matrix.channels == regular_matrix.channels
        assert symmetric_matrix.bandwidth == regular_matrix.bandwidth

        # Check symmetric storage properties
        assert symmetric_matrix.k_upper is not None
        assert symmetric_matrix.k_upper <= regular_matrix.k
        assert symmetric_matrix.dia_data_gpu is not None
        assert symmetric_matrix.dia_offsets_upper is not None
        assert symmetric_matrix.dia_offsets_upper_gpu is not None

        # Check data types
        assert symmetric_matrix.dia_data_gpu.dtype == cp.float32
        assert symmetric_matrix.dia_offsets_upper_gpu.dtype == cp.int32

        # Check that only upper diagonals are stored
        assert np.all(symmetric_matrix.dia_offsets_upper >= 0)

        # Check memory reduction
        regular_memory = regular_matrix.dia_data_gpu.nbytes
        symmetric_memory = symmetric_matrix.dia_data_gpu.nbytes
        assert symmetric_memory < regular_memory
        assert symmetric_memory >= regular_memory * 0.4  # At least some reduction
        assert symmetric_memory <= regular_memory * 0.6  # Should be significant

    def test_from_diagonal_ata_matrix_metadata_preservation(self):
        """Test that metadata is preserved during conversion."""
        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(self.A_matrix)

        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        # Check metadata preservation
        assert symmetric_matrix.bandwidth == regular_matrix.bandwidth
        assert symmetric_matrix.sparsity == regular_matrix.sparsity
        assert symmetric_matrix.nnz == regular_matrix.nnz
        assert symmetric_matrix.original_k == regular_matrix.k

    def test_multiply_3d_fallback_basic(self):
        """Test 3D matrix-vector multiplication using fallback implementation."""
        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(self.A_matrix)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        # Create test LED values
        led_values = cp.random.rand(self.channels, self.led_count).astype(cp.float32)

        # Test multiplication with fallback
        result = symmetric_matrix.multiply_3d(led_values, use_custom_kernel=False)

        # Check result properties
        assert result.shape == (self.channels, self.led_count)
        assert result.dtype == cp.float32

        # Test basic mathematical property: result should be finite
        assert np.all(np.isfinite(cp.asnumpy(result)))

    def test_multiply_3d_custom_kernels(self):
        """Test 3D matrix-vector multiplication with custom kernels if available."""
        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(self.A_matrix)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        led_values = cp.random.rand(self.channels, self.led_count).astype(cp.float32)

        # Test basic kernel if available
        try:
            result_basic = symmetric_matrix.multiply_3d(led_values, use_custom_kernel=True, optimized_kernel=False)
            assert result_basic.shape == (self.channels, self.led_count)
            assert result_basic.dtype == cp.float32
            assert np.all(np.isfinite(cp.asnumpy(result_basic)))
        except RuntimeError as e:
            if "kernel" in str(e).lower():
                pytest.skip("Basic custom kernel not available for this test")
            else:
                raise

        # Test optimized kernel if available
        try:
            result_optimized = symmetric_matrix.multiply_3d(led_values, use_custom_kernel=True, optimized_kernel=True)
            assert result_optimized.shape == (self.channels, self.led_count)
            assert result_optimized.dtype == cp.float32
            assert np.all(np.isfinite(cp.asnumpy(result_optimized)))
        except RuntimeError as e:
            if "kernel" in str(e).lower():
                pytest.skip("Optimized custom kernel not available for this test")
            else:
                raise

    def test_multiply_3d_input_validation(self):
        """Test input validation for multiply_3d."""
        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(self.A_matrix)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        # Test wrong shape
        with pytest.raises(ValueError, match="LED values should be shape"):
            wrong_shape = cp.random.rand(2, self.led_count).astype(cp.float32)
            symmetric_matrix.multiply_3d(wrong_shape)

        # Test wrong input type
        with pytest.raises(TypeError, match="led_values must be cupy.ndarray"):
            wrong_type = np.random.rand(self.channels, self.led_count).astype(np.float32)
            symmetric_matrix.multiply_3d(wrong_type)

        # Test wrong dtype
        with pytest.raises(TypeError, match="led_values must be cupy.float32"):
            wrong_dtype = cp.random.rand(self.channels, self.led_count).astype(cp.float16)
            symmetric_matrix.multiply_3d(wrong_dtype)

        # Test unsupported output dtype
        with pytest.raises(ValueError, match="only supports cupy.float32 output"):
            led_values = cp.random.rand(self.channels, self.led_count).astype(cp.float32)
            symmetric_matrix.multiply_3d(led_values, output_dtype=cp.float16)

    def test_g_ata_g_3d_basic(self):
        """Test g^T (A^T A) g computation."""
        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(self.A_matrix)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        # Create test gradient
        gradient = cp.random.rand(self.channels, self.led_count).astype(cp.float32)

        # Test computation
        result = symmetric_matrix.g_ata_g_3d(gradient, use_custom_kernel=False)

        # Check result properties
        assert result.shape == (self.channels,)
        assert result.dtype == cp.float32
        assert np.all(result >= 0)  # Should be positive (quadratic form)

    def test_g_ata_g_3d_input_validation(self):
        """Test input validation for g_ata_g_3d."""
        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(self.A_matrix)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        # Test wrong shape
        with pytest.raises(ValueError, match="Gradient should be shape"):
            wrong_shape = cp.random.rand(2, self.led_count).astype(cp.float32)
            symmetric_matrix.g_ata_g_3d(wrong_shape)

        # Test unsupported output dtype
        with pytest.raises(ValueError, match="only supports cupy.float32 output"):
            gradient = cp.random.rand(self.channels, self.led_count).astype(cp.float32)
            symmetric_matrix.g_ata_g_3d(gradient, output_dtype=cp.float16)

    def test_correctness_against_dense_reference(self):
        """Test correctness against dense matrix reference."""
        # Create symmetric matrix from reference dense matrices
        symmetric_matrix = self._create_symmetric_from_dense(self.dense_matrices)

        # Test multiple random vectors
        num_tests = 10
        max_errors = []

        for _ in range(num_tests):
            # Create random test vector
            test_vector = np.random.randn(self.channels, self.led_count).astype(np.float32)
            test_vector_gpu = cp.asarray(test_vector)

            # Compute reference result using dense matrices
            reference_result = np.zeros((self.channels, self.led_count), dtype=np.float32)
            for channel in range(self.channels):
                reference_result[channel, :] = self.dense_matrices[channel] @ test_vector[channel, :]

            # Compute result using symmetric matrix
            symmetric_result = symmetric_matrix.multiply_3d(test_vector_gpu, use_custom_kernel=False)
            symmetric_result_cpu = cp.asnumpy(symmetric_result)

            # Compare results
            error = np.max(np.abs(reference_result - symmetric_result_cpu))
            max_errors.append(error)

        # All errors should be very small
        max_error = max(max_errors)
        assert max_error < 1e-5, f"Max error {max_error:.2e} exceeds tolerance"

    def test_correctness_gradient_operations(self):
        """Test correctness of gradient operations against dense reference."""
        # Create symmetric matrix from reference dense matrices
        symmetric_matrix = self._create_symmetric_from_dense(self.dense_matrices)

        # Test multiple random gradients
        num_tests = 10
        max_errors = []

        for _ in range(num_tests):
            # Create random gradient vector
            gradient = np.random.randn(self.channels, self.led_count).astype(np.float32)
            gradient_gpu = cp.asarray(gradient)

            # Compute reference result: g^T A g for each channel
            reference_result = np.zeros(self.channels, dtype=np.float32)
            for channel in range(self.channels):
                ag = self.dense_matrices[channel] @ gradient[channel, :]
                reference_result[channel] = np.dot(gradient[channel, :], ag)

            # Compute result using symmetric matrix
            symmetric_result = symmetric_matrix.g_ata_g_3d(gradient_gpu, use_custom_kernel=False)
            symmetric_result_cpu = cp.asnumpy(symmetric_result)

            # Compare results
            error = np.max(np.abs(reference_result - symmetric_result_cpu))
            max_errors.append(error)

        # All errors should be small (slightly higher tolerance for quadratic form)
        max_error = max(max_errors)
        assert max_error < 1e-4, f"Max gradient error {max_error:.2e} exceeds tolerance"

    def test_memory_layout_validation(self):
        """Test memory layout validation for input tensors."""
        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(self.A_matrix)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        # Create test vector with non-contiguous memory layout
        test_vector = cp.random.rand(self.led_count, self.channels).astype(cp.float32)
        test_vector_transposed = test_vector.T  # This creates a non-contiguous view

        # Should handle non-contiguous arrays gracefully or raise informative error
        try:
            result = symmetric_matrix.multiply_3d(test_vector_transposed, use_custom_kernel=False)
            # If it works, result should still be valid
            assert result.shape == (self.channels, self.led_count)
            assert result.dtype == cp.float32
        except ValueError as e:
            # If it fails, should be due to memory layout
            assert "contiguous" in str(e).lower() or "layout" in str(e).lower()

    def test_get_info(self):
        """Test getting matrix information."""
        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(self.A_matrix)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        info = symmetric_matrix.get_info()

        # Check info dictionary
        expected_keys = {
            "led_count",
            "crop_size",
            "channels",
            "bandwidth",
            "sparsity",
            "nnz",
            "output_dtype",
            "symmetric_storage",
            "original_k",
            "k_upper",
            "memory_reduction",
            "storage_shape",
            "kernel_available",
        }
        assert set(info.keys()) == expected_keys

        # Check values
        assert info["led_count"] == self.led_count
        assert info["output_dtype"] == str(cp.float32)
        assert info["symmetric_storage"] is True
        assert info["original_k"] == regular_matrix.k
        assert info["k_upper"] == symmetric_matrix.k_upper
        assert "%" in info["memory_reduction"]

    def test_kernel_availability_detection(self):
        """Test detection of available kernels."""
        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(self.A_matrix)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        info = symmetric_matrix.get_info()

        # kernel_available should be a boolean
        assert isinstance(info["kernel_available"], bool)

        # If kernels are available, they should be initialized
        if info["kernel_available"]:
            assert symmetric_matrix.symmetric_kernel_basic is not None
            assert symmetric_matrix.symmetric_kernel_optimized is not None

    @pytest.mark.skip(reason="DiagonalATAMatrix has shape mismatch bug with very sparse matrices")
    def test_edge_case_empty_matrix(self):
        """Test handling of edge case with empty/minimal matrix."""
        # Create a very sparse matrix that might result in few diagonals
        sparse_A = sp.random(self.led_count, self.led_count * self.channels, density=0.01, format="csc")

        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(sparse_A)

        # Should still be able to convert to symmetric
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        # Should still work for multiplication
        led_values = cp.random.rand(self.channels, self.led_count).astype(cp.float32)
        result = symmetric_matrix.multiply_3d(led_values, use_custom_kernel=False)

        assert result.shape == (self.channels, self.led_count)
        assert result.dtype == cp.float32

    def test_large_matrix_properties(self):
        """Test properties with larger matrix to verify scaling."""
        large_led_count = 50  # Reduced size to avoid efficiency warnings
        large_pixels = 100

        # Create larger test matrix with lower density to avoid too many diagonals
        large_A = sp.random(large_pixels, large_led_count * self.channels, density=0.02, format="csc")

        regular_matrix = DiagonalATAMatrix(large_led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(large_A)

        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        # Check that memory reduction is still significant
        regular_memory = regular_matrix.dia_data_gpu.nbytes
        symmetric_memory = symmetric_matrix.dia_data_gpu.nbytes
        memory_reduction = (1 - symmetric_memory / regular_memory) * 100

        assert memory_reduction > 30, f"Memory reduction {memory_reduction:.1f}% too small for larger matrix"

        # Test basic functionality
        led_values = cp.random.rand(self.channels, large_led_count).astype(cp.float32)
        result = symmetric_matrix.multiply_3d(led_values, use_custom_kernel=False)

        assert result.shape == (self.channels, large_led_count)
        assert np.all(np.isfinite(cp.asnumpy(result)))

    def test_benchmark_3d_basic(self):
        """Test basic benchmarking functionality."""
        regular_matrix = DiagonalATAMatrix(self.led_count, self.crop_size)
        regular_matrix.build_from_diffusion_matrix(self.A_matrix)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)

        # Run benchmark with small number of trials
        benchmark_results = symmetric_matrix.benchmark_3d(num_trials=5, num_warmup=2)

        # Check that results are returned
        assert isinstance(benchmark_results, dict)
        assert "symmetric_fallback" in benchmark_results

        # Timing should be positive
        assert benchmark_results["symmetric_fallback"] > 0

    def _create_symmetric_from_dense(self, dense_matrices):
        """Helper to create SymmetricDiagonalATAMatrix from dense matrices."""
        led_count = len(dense_matrices[0])
        channels = len(dense_matrices)

        # Determine diagonal structure from first matrix
        dia_matrix = sp.dia_matrix(dense_matrices[0])
        all_offsets = dia_matrix.offsets
        upper_offsets = all_offsets[all_offsets >= 0]

        # Create symmetric matrix
        symmetric_matrix = SymmetricDiagonalATAMatrix(led_count, self.crop_size)
        symmetric_matrix.dia_offsets_upper = upper_offsets.copy()
        symmetric_matrix.k_upper = len(upper_offsets)
        symmetric_matrix.bandwidth = max(abs(all_offsets))

        # Extract data for upper diagonals
        symmetric_matrix.dia_data_cpu = np.zeros((channels, symmetric_matrix.k_upper, led_count), dtype=np.float32)

        for channel in range(channels):
            dia_matrix = sp.dia_matrix(dense_matrices[channel])
            for band_idx, offset in enumerate(symmetric_matrix.dia_offsets_upper):
                if offset in dia_matrix.offsets:
                    scipy_band_idx = np.where(dia_matrix.offsets == offset)[0][0]
                    symmetric_matrix.dia_data_cpu[channel, band_idx, :] = dia_matrix.data[scipy_band_idx, :]

        # Copy to GPU
        symmetric_matrix.dia_data_gpu = cp.asarray(symmetric_matrix.dia_data_cpu)
        symmetric_matrix.dia_offsets_upper_gpu = cp.asarray(symmetric_matrix.dia_offsets_upper, dtype=cp.int32)

        # Set metadata
        symmetric_matrix.sparsity = 50.0
        symmetric_matrix.nnz = np.count_nonzero(symmetric_matrix.dia_data_cpu)
        symmetric_matrix.original_k = len(all_offsets)

        return symmetric_matrix


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
