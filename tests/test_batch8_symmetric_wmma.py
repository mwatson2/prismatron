#!/usr/bin/env python3
"""
Comprehensive test suite for 8-frame batch symmetric WMMA matrix multiplication.

Tests the BatchSymmetricDiagonalATAMatrix class with batch_size=8, validating:
1. Correctness against reference implementations
2. Performance characteristics
3. Edge cases and boundary conditions
4. Integration with existing systems
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import cupy

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cupy = np  # Fallback

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix


# Fixtures for test matrices
@pytest.fixture
def small_matrix_8frame():
    """Create small 8-frame test matrix (64 LEDs, 4x4 blocks)."""
    led_count = 64  # 4x4 blocks of 16x16
    crop_size = 64
    batch_size = 8

    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count, crop_size=crop_size, batch_size=batch_size, output_dtype=cupy.float32
    )

    return matrix


@pytest.fixture
def medium_matrix_8frame():
    """Create medium 8-frame test matrix (320 LEDs, 20x20 blocks)."""
    led_count = 320  # 20x20 blocks of 16x16
    crop_size = 64
    batch_size = 8

    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count, crop_size=crop_size, batch_size=batch_size, output_dtype=cupy.float32
    )

    return matrix


class TestBatch8SymmetricWMMA:
    """Test suite for 8-frame batch symmetric WMMA operations."""

    def create_small_matrix_8frame(self):
        """Create small 8-frame test matrix (64 LEDs, 4x4 blocks)."""
        led_count = 64  # 4x4 blocks of 16x16
        crop_size = 64
        batch_size = 8

        matrix = BatchSymmetricDiagonalATAMatrix(
            led_count=led_count, crop_size=crop_size, batch_size=batch_size, output_dtype=cupy.float32
        )

        return matrix

    def create_medium_matrix_8frame(self):
        """Create medium 8-frame test matrix (320 LEDs, 20x20 blocks)."""
        led_count = 320  # 20x20 blocks of 16x16
        crop_size = 64
        batch_size = 8

        matrix = BatchSymmetricDiagonalATAMatrix(
            led_count=led_count, crop_size=crop_size, batch_size=batch_size, output_dtype=cupy.float32
        )

        return matrix

    def create_test_input_batch(self, led_count: int, batch_size: int = 8) -> cupy.ndarray:
        """Create test input batch with known patterns."""
        # Create test data with recognizable patterns
        input_batch = np.zeros((batch_size, 3, led_count), dtype=np.float32)

        for batch_idx in range(batch_size):
            for channel in range(3):
                # Each batch item has a different pattern for testing
                base_value = (batch_idx + 1) * 0.1 + channel * 0.01

                # Create pattern based on LED position
                for led_idx in range(led_count):
                    input_batch[batch_idx, channel, led_idx] = base_value + led_idx * 0.001

        return cupy.asarray(input_batch, dtype=cupy.float32)

    def create_simple_diagonal_matrix(self, matrix: BatchSymmetricDiagonalATAMatrix):
        """Create simple diagonal test matrix for validation."""
        # Create simple test data - identity-like matrix with some off-diagonals
        channels, led_count = 3, matrix.led_count

        # Simple diagonal pattern
        dia_offsets = np.array([0, 1, 2], dtype=np.int32)  # Main + 2 upper diagonals
        dia_data = np.zeros((channels, len(dia_offsets), led_count), dtype=np.float32)

        # Main diagonal: identity values
        dia_data[:, 0, :] = 1.0

        # First upper diagonal: 0.5
        dia_data[:, 1, :-1] = 0.5

        # Second upper diagonal: 0.25
        dia_data[:, 2, :-2] = 0.25

        # Convert to GPU and build matrix
        dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
        matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

        return matrix

    def test_8frame_basic_functionality(self, small_matrix_8frame):
        """Test basic 8-frame batch multiplication functionality."""
        matrix = small_matrix_8frame

        # Create simple test matrix
        matrix = self.create_simple_diagonal_matrix(matrix)

        # Create test input
        input_batch = self.create_test_input_batch(matrix.led_count, batch_size=8)

        # Perform 8-frame batch multiplication
        result = matrix.multiply_batch8_3d(input_batch, debug_logging=True)

        # Validate output shape
        assert result.shape == (8, 3, matrix.led_count)

        # Validate result is not all zeros
        assert cupy.sum(cupy.abs(result)) > 0

        print(f"8-frame test passed: output shape {result.shape}, non-zero sum: {cupy.sum(cupy.abs(result)):.6f}")

    def test_8frame_vs_cpu_reference(self, small_matrix_8frame):
        """Test 8-frame batch vs CPU reference correctness."""
        matrix = small_matrix_8frame

        # Create simple test matrix - identity matrix
        dia_offsets = np.array([0], dtype=np.int32)  # Only main diagonal
        dia_data = np.ones((3, 1, matrix.led_count), dtype=np.float32)  # Identity
        dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
        matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

        # Create test input
        input_batch = self.create_test_input_batch(matrix.led_count, batch_size=8)

        # Process with 8-frame batch
        result_8frame = matrix.multiply_batch8_3d(input_batch, debug_logging=False)

        # CPU reference: for identity matrix, output should equal input (A*x = x)
        expected_output = cupy.asnumpy(input_batch)

        # Compare results
        max_diff = cupy.max(cupy.abs(result_8frame - cupy.asarray(expected_output)))
        relative_error = max_diff / (cupy.max(cupy.abs(result_8frame)) + 1e-10)

        print(f"8-frame vs CPU reference: max_diff={max_diff:.6f}, relative_error={relative_error:.6f}")

        # Should be very close (within numerical precision for FP16->FP32 conversion)
        assert relative_error < 0.01, f"8-frame result differs too much from CPU reference: {relative_error}"

    def test_8frame_automatic_routing(self, small_matrix_8frame):
        """Test automatic routing to 8-frame kernel via multiply_batch_3d."""
        matrix = small_matrix_8frame

        # Create simple test matrix
        matrix = self.create_simple_diagonal_matrix(matrix)

        # Create test input
        input_batch = self.create_test_input_batch(matrix.led_count, batch_size=8)

        # Use general multiply_batch_3d method (should route to 8-frame automatically)
        result_auto = matrix.multiply_batch_3d(input_batch, debug_logging=True)

        # Use explicit 8-frame method
        result_explicit = matrix.multiply_batch8_3d(input_batch, debug_logging=False)

        # Results should be identical
        max_diff = cupy.max(cupy.abs(result_auto - result_explicit))
        assert max_diff < 1e-6, f"Automatic routing differs from explicit: {max_diff}"

        print(f"Automatic routing test passed: max_diff={max_diff:.9f}")

    def test_8frame_input_validation(self, small_matrix_8frame):
        """Test input validation for 8-frame processing."""
        matrix = small_matrix_8frame

        # Test wrong batch size
        wrong_batch = cupy.zeros((16, 3, matrix.led_count), dtype=cupy.float32)
        try:
            matrix.multiply_batch8_3d(wrong_batch)
            raise AssertionError("Should have raised ValueError for wrong batch size")
        except ValueError as e:
            assert "requires batch_size=8" in str(e)

        # Test wrong shape
        wrong_shape = cupy.zeros((8, 3), dtype=cupy.float32)  # Missing LED dimension
        try:
            matrix.multiply_batch8_3d(wrong_shape)
            raise AssertionError("Should have raised ValueError for wrong shape")
        except ValueError as e:
            assert "not enough values to unpack" in str(e) or "3D batch tensor" in str(e)

        # Test wrong channels
        wrong_channels = cupy.zeros((8, 4, matrix.led_count), dtype=cupy.float32)
        try:
            matrix.multiply_batch8_3d(wrong_channels)
            raise AssertionError("Should have raised ValueError for wrong channels")
        except ValueError as e:
            assert "must have 3 channels" in str(e)

        print("Input validation tests passed")

    def test_8frame_memory_layout(self, small_matrix_8frame):
        """Test memory layout requirements for 8-frame processing."""
        matrix = small_matrix_8frame

        # Create test input with non-contiguous layout
        input_data = cupy.zeros((8, 3, matrix.led_count * 2), dtype=cupy.float32)
        input_non_contiguous = input_data[:, :, ::2]  # Non-contiguous view

        # Should raise error for non-contiguous input
        try:
            matrix.multiply_batch8_3d(input_non_contiguous)
            raise AssertionError("Should have raised ValueError for non-contiguous tensor")
        except ValueError as e:
            assert "C-contiguous" in str(e)

        # Fix layout and test again
        input_fixed = cupy.ascontiguousarray(input_non_contiguous)

        # Create simple test matrix
        matrix = self.create_simple_diagonal_matrix(matrix)

        # Should work now
        result = matrix.multiply_batch8_3d(input_fixed, debug_logging=False)
        assert result.shape == (8, 3, matrix.led_count)

        print("Memory layout validation tests passed")

    def test_8frame_medium_matrix(self, medium_matrix_8frame):
        """Test 8-frame processing on medium-sized matrix."""
        matrix = medium_matrix_8frame

        # Create simple test matrix
        matrix = self.create_simple_diagonal_matrix(matrix)

        # Create test input
        input_batch = self.create_test_input_batch(matrix.led_count, batch_size=8)

        # Perform 8-frame batch multiplication
        start_time = time.time()
        result = matrix.multiply_batch8_3d(input_batch, debug_logging=True)
        elapsed_time = time.time() - start_time

        # Validate output
        assert result.shape == (8, 3, matrix.led_count)
        assert cupy.sum(cupy.abs(result)) > 0

        print(f"Medium matrix (320 LEDs) test passed in {elapsed_time*1000:.2f}ms")
        print(f"Result shape: {result.shape}, non-zero sum: {cupy.sum(cupy.abs(result)):.6f}")

    def test_8frame_kernel_availability(self):
        """Test kernel availability detection."""
        from utils.batch_symmetric_diagonal_ata_matrix import BATCH8_WMMA_KERNEL_AVAILABLE

        print(f"8-frame WMMA kernels available: {BATCH8_WMMA_KERNEL_AVAILABLE}")

        # Should be True if kernels compiled successfully
        if BATCH8_WMMA_KERNEL_AVAILABLE:
            print("✓ 8-frame kernels are available and loaded")
        else:
            print("⚠ 8-frame kernels not available - using sequential fallback")

    def test_8frame_class_info(self, small_matrix_8frame):
        """Test 8-frame matrix info reporting."""
        matrix = small_matrix_8frame

        info = matrix.get_info()

        assert info["batch_size"] == 8
        assert info["led_count"] == 64
        assert info["channels"] == 3
        assert info["block_size"] == 16
        assert info["block_storage"]
        assert info["symmetric_storage"]

        print("Matrix info:", info)


def run_8frame_tests():
    """Run all 8-frame tests directly."""
    print("Running 8-frame batch WMMA tests...")

    if not CUDA_AVAILABLE:
        print("CUDA not available - skipping tests")
        return

    test_suite = TestBatch8SymmetricWMMA()

    try:
        # Create matrices
        small_matrix = test_suite.create_small_matrix_8frame()
        medium_matrix = test_suite.create_medium_matrix_8frame()

        # Run tests
        print("\n1. Testing basic functionality...")
        test_suite.test_8frame_basic_functionality(small_matrix)

        print("\n2. Testing vs CPU reference correctness...")
        test_suite.test_8frame_vs_cpu_reference(small_matrix)

        print("\n3. Testing automatic routing...")
        test_suite.test_8frame_automatic_routing(small_matrix)

        print("\n4. Testing input validation...")
        test_suite.test_8frame_input_validation(small_matrix)

        print("\n5. Testing memory layout...")
        test_suite.test_8frame_memory_layout(small_matrix)

        print("\n6. Testing medium matrix...")
        test_suite.test_8frame_medium_matrix(medium_matrix)

        print("\n7. Testing kernel availability...")
        test_suite.test_8frame_kernel_availability()

        print("\n8. Testing class info...")
        test_suite.test_8frame_class_info(small_matrix)

        print("\n✅ All 8-frame batch WMMA tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_8frame_tests()
    sys.exit(0 if success else 1)
