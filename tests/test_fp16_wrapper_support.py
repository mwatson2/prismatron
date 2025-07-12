"""
Unit tests for FP16 support in wrapper classes.

This module tests the FP16 output functionality added to the
SingleBlockMixedSparseTensor and DiagonalATAMatrix wrapper classes.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pytest

# GPU availability check
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


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


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestSingleBlockMixedSparseTensorFP16(unittest.TestCase):
    """Test FP16 support in SingleBlockMixedSparseTensor."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid issues if CuPy is not available
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

        self.batch_size = 10
        self.channels = 3
        self.height = 100
        self.width = 100
        self.block_size = 32

    def test_fp16_initialization(self):
        """Test initialization with FP16 output dtype."""
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

        # Test FP16 output specification at construction
        tensor_fp16 = SingleBlockMixedSparseTensor(
            batch_size=self.batch_size,
            channels=self.channels,
            height=self.height,
            width=self.width,
            block_size=self.block_size,
            output_dtype=cp.float16,
        )

        self.assertEqual(tensor_fp16.output_dtype, cp.float16)
        self.assertEqual(tensor_fp16.dtype, cp.float32)  # Input still FP32

    def test_int8_fp16_initialization(self):
        """Test initialization with INT8 input and FP16 output."""
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

        # Test INT8 -> FP16 pipeline
        tensor_int8_fp16 = SingleBlockMixedSparseTensor(
            batch_size=self.batch_size,
            channels=self.channels,
            height=self.height,
            width=self.width,
            block_size=self.block_size,
            dtype=cp.uint8,
            output_dtype=cp.float16,
        )

        self.assertEqual(tensor_int8_fp16.output_dtype, cp.float16)
        self.assertEqual(tensor_int8_fp16.dtype, cp.uint8)

    def test_default_output_dtype_logic(self):
        """Test intelligent default output dtype selection."""
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

        # For FP32 input, default should be FP32 output
        tensor_fp32 = SingleBlockMixedSparseTensor(
            batch_size=self.batch_size,
            channels=self.channels,
            height=self.height,
            width=self.width,
            block_size=self.block_size,
            dtype=cp.float32,
        )
        self.assertEqual(tensor_fp32.output_dtype, cp.float32)

        # For INT8 input, default should be FP32 output (most common)
        tensor_int8 = SingleBlockMixedSparseTensor(
            batch_size=self.batch_size,
            channels=self.channels,
            height=self.height,
            width=self.width,
            block_size=self.block_size,
            dtype=cp.uint8,
        )
        self.assertEqual(tensor_int8.output_dtype, cp.float32)

    @patch("src.utils.cuda_kernels.cuda_transpose_dot_product_3d_compute_optimized_int8_fp16")
    def test_fp16_kernel_selection(self, mock_fp16_kernel):
        """Test that correct kernels are selected for FP16 output."""
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

        # Create tensor with FP16 output
        tensor = SingleBlockMixedSparseTensor(
            batch_size=self.batch_size,
            channels=self.channels,
            height=self.height,
            width=self.width,
            block_size=self.block_size,
            dtype=cp.uint8,
            output_dtype=cp.float16,
        )

        # Mock the FP16 kernel function
        mock_fp16_kernel.return_value = cp.zeros((self.batch_size, self.channels), dtype=cp.float16)

        # Create test data
        target_3d = cp.random.randint(0, 255, (self.channels, self.height, self.width), dtype=cp.uint8)

        # Call transpose_dot_product_3d with FP16 output
        result = tensor.transpose_dot_product_3d(target_3d)

        # Verify FP16 kernel was called
        mock_fp16_kernel.assert_called_once()
        self.assertEqual(result.dtype, cp.float16)

    def test_repr_includes_output_dtype(self):
        """Test that __repr__ includes output dtype information."""
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

        tensor = SingleBlockMixedSparseTensor(
            batch_size=self.batch_size,
            channels=self.channels,
            height=self.height,
            width=self.width,
            block_size=self.block_size,
            output_dtype=cp.float16,
        )

        repr_str = str(tensor)
        self.assertIn("output_dtype=<class 'numpy.float16'>", repr_str)

    def test_to_dict_from_dict_fp16(self):
        """Test serialization/deserialization with FP16 output dtype."""
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

        # Create tensor with FP16 output
        original_tensor = SingleBlockMixedSparseTensor(
            batch_size=self.batch_size,
            channels=self.channels,
            height=self.height,
            width=self.width,
            block_size=self.block_size,
            output_dtype=cp.float16,
        )

        # Export to dict
        tensor_dict = original_tensor.to_dict()
        self.assertIn("output_dtype", tensor_dict)
        self.assertEqual(str(tensor_dict["output_dtype"]), "float16")

        # Import from dict
        restored_tensor = SingleBlockMixedSparseTensor.from_dict(tensor_dict)
        self.assertEqual(restored_tensor.output_dtype, cp.float16)
        self.assertEqual(restored_tensor.dtype, original_tensor.dtype)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestDiagonalATAMatrixFP16(unittest.TestCase):
    """Test FP16 support in DiagonalATAMatrix."""

    def setUp(self):
        """Set up test fixtures."""
        self.led_count = 100
        self.crop_size = 64

    def test_fp16_initialization(self):
        """Test initialization with FP16 output dtype."""
        from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

        # Test FP16 output specification at construction
        matrix_fp16 = DiagonalATAMatrix(led_count=self.led_count, crop_size=self.crop_size, output_dtype=cp.float16)

        self.assertEqual(matrix_fp16.output_dtype, cp.float16)

    def test_default_output_dtype(self):
        """Test default output dtype is FP32."""
        from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

        matrix = DiagonalATAMatrix(led_count=self.led_count, crop_size=self.crop_size)
        self.assertEqual(matrix.output_dtype, cp.float32)

    def test_invalid_output_dtype(self):
        """Test that invalid output dtypes raise errors."""
        from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

        with self.assertRaises(ValueError):
            DiagonalATAMatrix(
                led_count=self.led_count,
                crop_size=self.crop_size,
                output_dtype=cp.int32,
            )

    @patch("src.utils.diagonal_ata_matrix.CUSTOM_3D_KERNEL_FP16_AVAILABLE", True)
    @patch("src.utils.diagonal_ata_matrix.CustomDIA3DMatVecFP16")
    def test_fp16_kernel_selection_multiply_3d(self, mock_fp16_kernel_class):
        """Test that FP16 kernels are selected for multiply_3d."""
        from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

        # Create matrix with FP16 output
        matrix = DiagonalATAMatrix(led_count=self.led_count, crop_size=self.crop_size, output_dtype=cp.float16)

        # Mock the kernel instance and its call
        mock_kernel_instance = mock_fp16_kernel_class.return_value
        mock_kernel_instance.return_value = cp.zeros((3, self.led_count), dtype=cp.float16)

        # Set up minimal matrix data for testing
        matrix.dia_data_gpu = cp.zeros((3, 10, self.led_count), dtype=cp.float32)
        matrix.dia_offsets = cp.zeros(10, dtype=cp.int32)
        matrix.k = 10

        # Create test data
        led_values = np.random.rand(3, self.led_count).astype(np.float32)

        # Call multiply_3d with FP16 output
        result = matrix.multiply_3d(led_values, output_dtype=cp.float16)

        # Verify FP16 kernel class was instantiated
        mock_fp16_kernel_class.assert_called()
        # Verify the kernel was called
        mock_kernel_instance.assert_called()

    def test_to_dict_from_dict_fp16(self):
        """Test serialization/deserialization with FP16 output dtype."""
        from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

        # Create matrix with FP16 output
        original_matrix = DiagonalATAMatrix(led_count=self.led_count, crop_size=self.crop_size, output_dtype=cp.float16)

        # Export to dict
        matrix_dict = original_matrix.to_dict()
        self.assertIn("output_dtype", matrix_dict)
        self.assertEqual(matrix_dict["output_dtype"], "float16")
        self.assertEqual(matrix_dict["version"], "8.0")

        # Import from dict
        restored_matrix = DiagonalATAMatrix.from_dict(matrix_dict)
        self.assertEqual(restored_matrix.output_dtype, cp.float16)

    def test_get_info_includes_fp16_info(self):
        """Test that get_info includes FP16 kernel availability."""
        from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

        matrix = DiagonalATAMatrix(led_count=self.led_count, crop_size=self.crop_size, output_dtype=cp.float16)

        info = matrix.get_info()
        self.assertIn("custom_kernel_fp16_available", info)
        self.assertIn("output_dtype", info)
        self.assertEqual(info["storage_format"], "unified_3d_dia_v8_fp16")


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility for classes without FP16 support."""

    def test_single_block_tensor_old_dict_format(self):
        """Test loading old SingleBlockMixedSparseTensor dict without output_dtype."""
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

        # Simulate old dict format without output_dtype
        old_dict = {
            "sparse_values": np.random.rand(3, 10, 32, 32).astype(np.float32),
            "block_positions": np.random.randint(0, 68, (3, 10, 2)).astype(np.int32),
            "batch_size": np.array(10, dtype=np.int32),
            "channels": np.array(3, dtype=np.int32),
            "height": np.array(100, dtype=np.int32),
            "width": np.array(100, dtype=np.int32),
            "block_size": np.array(32, dtype=np.int32),
            "device": np.array("cuda", dtype="U10"),
            "dtype": np.array("float32", dtype="U10"),
            # Note: no output_dtype key
        }

        # Should load successfully with default output_dtype
        tensor = SingleBlockMixedSparseTensor.from_dict(old_dict)
        self.assertEqual(tensor.output_dtype, cp.float32)  # Default for FP32 input

    def test_diagonal_ata_matrix_old_dict_format(self):
        """Test loading old DiagonalATAMatrix dict without output_dtype."""
        from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

        # Simulate old dict format without output_dtype
        old_dict = {
            "led_count": 100,
            "crop_size": 64,
            "channels": 3,
            "dia_data_3d": np.zeros((3, 10, 100), dtype=np.float32),
            "dia_offsets_3d": np.arange(10, dtype=np.int32),
            "k": 10,
            "bandwidth": 50,
            "sparsity": 0.1,
            "nnz": 1000,
            "channel_nnz": [300, 350, 350],
            "version": "7.0",
            # Note: no output_dtype key
        }

        # Should load successfully with default output_dtype
        matrix = DiagonalATAMatrix.from_dict(old_dict)
        self.assertEqual(matrix.output_dtype, cp.float32)  # Default


if __name__ == "__main__":
    unittest.main()
