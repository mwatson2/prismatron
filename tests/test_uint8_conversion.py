#!/usr/bin/env python3
"""
Unit tests for float32 to uint8 conversion in SingleBlockMixedSparseTensor.

Tests the conversion of mixed sparse tensors from float32 [0,1] range
to uint8 [0,254] range for memory efficiency.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

cp = pytest.importorskip("cupy")

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


class TestUint8Conversion:
    """Test suite for float32 to uint8 tensor conversion."""

    def test_simple_conversion(self):
        """Test basic conversion with simple values."""
        # Create float32 tensor with known values
        tensor_fp32 = SingleBlockMixedSparseTensor(
            batch_size=2,
            channels=3,
            height=100,
            width=100,
            block_size=10,
            dtype=cp.float32,
        )

        # Set blocks with specific values
        # LED 0, Red channel: values around 0.5
        block_data = cp.ones((10, 10), dtype=cp.float32) * 0.5
        tensor_fp32.set_block(0, 0, 10, 20, block_data)

        # LED 0, Green channel: values at 1.0 (max)
        block_data = cp.ones((10, 10), dtype=cp.float32) * 1.0
        tensor_fp32.set_block(0, 1, 30, 40, block_data)

        # LED 1, Blue channel: values at 0.0 (min)
        block_data = cp.zeros((10, 10), dtype=cp.float32)
        tensor_fp32.set_block(1, 2, 50, 60, block_data)

        # Convert to uint8
        tensor_uint8 = tensor_fp32.to_uint8()

        # Verify dtype changed
        assert tensor_uint8.dtype == cp.uint8
        assert tensor_fp32.dtype == cp.float32  # Original unchanged

        # Verify dimensions preserved
        assert tensor_uint8.batch_size == tensor_fp32.batch_size
        assert tensor_uint8.channels == tensor_fp32.channels
        assert tensor_uint8.height == tensor_fp32.height
        assert tensor_uint8.width == tensor_fp32.width
        assert tensor_uint8.block_size == tensor_fp32.block_size

        # Check converted values
        # 0.5 * 254 = 127
        red_block = tensor_uint8.sparse_values[0, 0]
        assert cp.allclose(red_block, 127, atol=1)

        # 1.0 * 254 = 254
        green_block = tensor_uint8.sparse_values[1, 0]
        assert cp.allclose(green_block, 254, atol=1)

        # 0.0 * 254 = 0
        blue_block = tensor_uint8.sparse_values[2, 1]
        assert cp.allclose(blue_block, 0, atol=1)

    def test_value_clamping(self):
        """Test that values outside [0,1] are clamped before conversion."""
        tensor_fp32 = SingleBlockMixedSparseTensor(
            batch_size=1,
            channels=1,
            height=100,
            width=100,
            block_size=10,
            dtype=cp.float32,
        )

        # Create block with values outside [0,1] range
        block_data = cp.array(
            [
                [-0.5, 0.0, 0.25, 0.5],
                [0.75, 1.0, 1.5, 2.0],
                [-1.0, 0.1, 0.9, 3.0],
                [0.3, 0.7, 1.1, -0.2],
            ],
            dtype=cp.float32,
        )
        # Pad to 10x10
        block_data_full = cp.zeros((10, 10), dtype=cp.float32)
        block_data_full[:4, :4] = block_data

        tensor_fp32.set_block(0, 0, 10, 20, block_data_full)

        # Convert to uint8
        tensor_uint8 = tensor_fp32.to_uint8()

        # Check that values were clamped to [0, 254]
        converted_block = tensor_uint8.sparse_values[0, 0]
        converted_values = converted_block[:4, :4]

        # Expected values after clamping and scaling
        expected = cp.array(
            [
                [0, 0, 63, 127],  # -0.5->0, 0->0, 0.25->63, 0.5->127
                [190, 254, 254, 254],  # 0.75->190, 1->254, 1.5->254, 2->254
                [0, 25, 228, 254],  # -1->0, 0.1->25, 0.9->228, 3->254
                [76, 177, 254, 0],  # 0.3->76, 0.7->177, 1.1->254, -0.2->0
            ],
            dtype=cp.uint8,
        )

        assert cp.allclose(converted_values, expected, atol=2)

    def test_memory_reduction(self):
        """Test that uint8 conversion reduces memory usage."""
        # Create larger tensor to see memory difference
        tensor_fp32 = SingleBlockMixedSparseTensor(
            batch_size=100,
            channels=3,
            height=800,
            width=600,
            block_size=64,
            dtype=cp.float32,
        )

        # Set random blocks
        for led in range(100):
            for channel in range(3):
                block_data = cp.random.rand(64, 64).astype(cp.float32)
                # Position must be multiple of 4 for tensor core alignment
                row = (led % 10) * 70
                col = ((led // 10) * 60) % (600 - 64)
                col = (col // 4) * 4  # Ensure multiple of 4
                tensor_fp32.set_block(led, channel, row, col, block_data)

        # Get memory before conversion
        memory_fp32 = tensor_fp32.memory_info()["total_mb"]

        # Convert to uint8
        tensor_uint8 = tensor_fp32.to_uint8()
        memory_uint8 = tensor_uint8.memory_info()["total_mb"]

        # Verify memory reduction (should be roughly 4x for sparse values)
        # Note: block_positions stay as int32, so not full 4x reduction
        assert memory_uint8 < memory_fp32
        # Sparse values should be 4x smaller
        assert tensor_uint8.sparse_values.nbytes == tensor_fp32.sparse_values.nbytes // 4

    def test_error_on_uint8_input(self):
        """Test that converting uint8 tensor raises error."""
        tensor_uint8 = SingleBlockMixedSparseTensor(
            batch_size=1,
            channels=1,
            height=100,
            width=100,
            block_size=10,
            dtype=cp.uint8,
        )

        with pytest.raises(ValueError, match="already in uint8 format"):
            tensor_uint8.to_uint8()

    def test_positions_preserved(self):
        """Test that block positions are preserved during conversion."""
        tensor_fp32 = SingleBlockMixedSparseTensor(
            batch_size=3,
            channels=2,
            height=200,
            width=200,
            block_size=20,
            dtype=cp.float32,
        )

        # Set blocks at specific positions
        positions = [
            (0, 0, 10, 20),  # LED 0, channel 0
            (0, 1, 30, 40),  # LED 0, channel 1
            (1, 0, 50, 60),  # LED 1, channel 0
            (1, 1, 70, 80),  # LED 1, channel 1
            (2, 0, 90, 100),  # LED 2, channel 0
            (2, 1, 110, 120),  # LED 2, channel 1
        ]

        for led, channel, row, col in positions:
            block_data = cp.random.rand(20, 20).astype(cp.float32)
            tensor_fp32.set_block(led, channel, row, col, block_data)

        # Convert to uint8
        tensor_uint8 = tensor_fp32.to_uint8()

        # Verify all positions are identical
        assert cp.array_equal(tensor_uint8.block_positions, tensor_fp32.block_positions)

    def test_output_dtype_preserved(self):
        """Test that output_dtype remains float32 after conversion."""
        tensor_fp32 = SingleBlockMixedSparseTensor(
            batch_size=1,
            channels=1,
            height=100,
            width=100,
            block_size=10,
            dtype=cp.float32,
            output_dtype=cp.float32,
        )

        # Convert to uint8
        tensor_uint8 = tensor_fp32.to_uint8()

        # Output dtype should still be float32 for compatibility
        assert tensor_uint8.output_dtype == cp.float32

    def test_fractional_values(self):
        """Test conversion of various fractional values."""
        tensor_fp32 = SingleBlockMixedSparseTensor(
            batch_size=1,
            channels=1,
            height=100,
            width=100,
            block_size=10,
            dtype=cp.float32,
        )

        # Create block with various fractional values
        test_values = cp.array(
            [
                0.0,
                0.1,
                0.2,
                0.25,
                0.333,
                0.5,
                0.666,
                0.75,
                0.9,
                1.0,
            ],
            dtype=cp.float32,
        )
        block_data = cp.zeros((10, 10), dtype=cp.float32)
        block_data[0, :10] = test_values

        tensor_fp32.set_block(0, 0, 10, 20, block_data)

        # Convert to uint8
        tensor_uint8 = tensor_fp32.to_uint8()

        # Check converted values (scaled by 254)
        converted_block = tensor_uint8.sparse_values[0, 0]
        converted_values = converted_block[0, :10]

        expected = cp.array(
            [
                0,  # 0.0 * 254 = 0
                25,  # 0.1 * 254 = 25.4
                51,  # 0.2 * 254 = 50.8
                64,  # 0.25 * 254 = 63.5
                84,  # 0.333 * 254 = 84.582
                127,  # 0.5 * 254 = 127
                169,  # 0.666 * 254 = 169.164
                191,  # 0.75 * 254 = 190.5
                229,  # 0.9 * 254 = 228.6
                254,  # 1.0 * 254 = 254
            ],
            dtype=cp.uint8,
        )

        assert cp.allclose(converted_values, expected, atol=1)

    def test_round_trip_accuracy(self):
        """Test accuracy of values after conversion and scaling back."""
        tensor_fp32 = SingleBlockMixedSparseTensor(
            batch_size=1,
            channels=1,
            height=100,
            width=100,
            block_size=10,
            dtype=cp.float32,
        )

        # Create block with random values in [0, 1]
        original_values = cp.random.rand(10, 10).astype(cp.float32)
        tensor_fp32.set_block(0, 0, 10, 20, original_values)

        # Convert to uint8
        tensor_uint8 = tensor_fp32.to_uint8()

        # Scale back to [0, 1] range
        uint8_values = tensor_uint8.sparse_values[0, 0]
        recovered_values = uint8_values.astype(cp.float32) / 254.0

        # Check that values are close (within quantization error)
        # Maximum error should be 1/254 ≈ 0.0039
        max_error = cp.max(cp.abs(original_values - recovered_values))
        assert max_error < 0.005  # Allow for rounding errors


if __name__ == "__main__":
    # Run tests
    test = TestUint8Conversion()

    print("Running test_simple_conversion...")
    test.test_simple_conversion()
    print("✓ Passed")

    print("Running test_value_clamping...")
    test.test_value_clamping()
    print("✓ Passed")

    print("Running test_memory_reduction...")
    test.test_memory_reduction()
    print("✓ Passed")

    print("Running test_error_on_uint8_input...")
    test.test_error_on_uint8_input()
    print("✓ Passed")

    print("Running test_positions_preserved...")
    test.test_positions_preserved()
    print("✓ Passed")

    print("Running test_output_dtype_preserved...")
    test.test_output_dtype_preserved()
    print("✓ Passed")

    print("Running test_fractional_values...")
    test.test_fractional_values()
    print("✓ Passed")

    print("Running test_round_trip_accuracy...")
    test.test_round_trip_accuracy()
    print("✓ Passed")

    print("\n✅ All tests passed!")
