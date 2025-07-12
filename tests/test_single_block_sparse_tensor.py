#!/usr/bin/env python3
"""
Test script for SingleBlockMixedSparseTensor implementation.

This script validates the custom sparse tensor implementation with comprehensive tests
including correctness, performance, and memory efficiency.
"""

import logging
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def test_basic_functionality():
    """Test basic tensor creation and block setting."""
    logger.info("=== Testing Basic Functionality ===")

    # Create a small tensor for testing
    tensor = SingleBlockMixedSparseTensor(batch_size=5, channels=3, height=100, width=80, block_size=8)

    logger.info(f"Created tensor: {tensor}")

    # Test setting individual blocks
    for batch_idx in range(5):
        for channel_idx in range(3):
            # Create a simple test pattern
            values = cp.ones((8, 8), dtype=cp.float32) * (batch_idx * 3 + channel_idx + 1)

            # Position blocks at different locations
            row = batch_idx * 15
            col = channel_idx * 20

            tensor.set_block(batch_idx, channel_idx, row, col, values)

    logger.info("Set all test blocks successfully")

    # Verify block information
    for batch_idx in range(2):  # Check first 2 LEDs
        for channel_idx in range(3):
            info = tensor.get_block_info(batch_idx, channel_idx)
            expected_value = batch_idx * 3 + channel_idx + 1

            assert info["is_set"], f"Block ({batch_idx}, {channel_idx}) should be set"
            assert info["position"] == (batch_idx * 15, channel_idx * 20)
            assert cp.allclose(info["values"], expected_value)

    logger.info("✓ Basic functionality test passed")


def test_batch_operations():
    """Test batch setting operations."""
    logger.info("=== Testing Batch Operations ===")

    batch_size, channels = 10, 3
    height, width = 64, 64
    block_size = 16

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size)

    # Create batch data (note: positions should be channels-first for new layout)
    positions = cp.random.randint(0, height - block_size, (channels, batch_size, 2))
    # Align x-coordinates to multiples of 4 for uint8 vectorization
    positions[:, :, 1] = (positions[:, :, 1] // 4) * 4
    values = cp.random.rand(channels, batch_size, block_size, block_size).astype(cp.float32)

    # Set blocks in batch
    start_time = time.time()
    tensor.set_blocks_batch(positions, values)
    batch_set_time = time.time() - start_time

    logger.info(f"Batch block setting took {batch_set_time:.4f}s")

    # All blocks are assumed to be set - no explicit verification needed

    # Test 3D target computation
    target_3d = cp.random.rand(channels, height, width).astype(cp.float32)
    result = tensor.transpose_dot_product_3d(target_3d)

    logger.info(f"3D target result shape: {result.shape}")
    assert result.shape == (batch_size, channels)

    logger.info("✓ Batch operations test passed")


def test_memory_efficiency():
    """Test memory usage and compression."""
    logger.info("=== Testing Memory Efficiency ===")

    # Create a realistically sized tensor
    batch_size = 1000  # 1000 LEDs
    channels = 3  # RGB
    height = 480  # Frame height
    width = 800  # Frame width
    block_size = 64  # 64x64 blocks

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size)

    # Get initial memory info
    initial_memory = tensor.memory_info()
    logger.info(f"Initial memory usage: {initial_memory['total_mb']:.1f}MB")
    logger.info(f"Equivalent dense would be: {initial_memory['equivalent_dense_mb']:.1f}MB")
    logger.info(f"Compression ratio: {initial_memory['compression_ratio']:.1%}")

    # Set some blocks with random data
    num_blocks_to_set = 500  # Set blocks for ~17% of LEDs
    for i in range(num_blocks_to_set):
        batch_idx = i % batch_size
        channel_idx = i % channels

        row = cp.random.randint(0, height - block_size)
        col = cp.random.randint(0, width - block_size)
        values = cp.random.rand(block_size, block_size).astype(cp.float32)

        tensor.set_block(batch_idx, channel_idx, row, col, values)

    # Get final memory info
    final_memory = tensor.memory_info()
    logger.info(f"Final memory usage: {final_memory['total_mb']:.1f}MB")
    logger.info(f"Blocks stored: {final_memory['blocks_stored']}")

    # Memory should still be much smaller than dense
    assert (
        final_memory["compression_ratio"] < 0.1
    ), f"Compression ratio {final_memory['compression_ratio']:.1%} should be < 10%"

    logger.info("✓ Memory efficiency test passed")


def test_performance_comparison():
    """Compare performance against naive dense implementation."""
    logger.info("=== Testing Performance Comparison ===")

    # Medium-sized test case
    batch_size = 100
    channels = 3
    height = 200
    width = 200
    block_size = 32

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size)

    # Set all blocks with random data (note: positions should be channels-first for new layout)
    positions = cp.random.randint(0, height - block_size, (channels, batch_size, 2))
    # Align x-coordinates to multiples of 4 for uint8 vectorization
    positions[:, :, 1] = (positions[:, :, 1] // 4) * 4
    values = cp.random.rand(channels, batch_size, block_size, block_size).astype(cp.float32)
    tensor.set_blocks_batch(positions, values)

    # Create multiple target matrices (3D format)
    num_targets = 10
    targets_3d = cp.random.rand(num_targets, channels, height, width).astype(cp.float32)

    # Time sparse implementation
    cp.cuda.Device().synchronize()
    start_time = time.time()

    sparse_results = []
    for i in range(num_targets):
        result = tensor.transpose_dot_product_3d(targets_3d[i])
        sparse_results.append(result)

    cp.cuda.Device().synchronize()
    sparse_time = time.time() - start_time

    logger.info(f"Sparse implementation: {sparse_time:.4f}s for {num_targets} targets")
    logger.info(f"Sparse per target: {sparse_time / num_targets * 1000:.2f}ms")

    # For comparison, time a single dense conversion (too expensive to do all)
    dense_conversion_start = time.time()
    dense_sample = tensor.to_array(0, 0)
    dense_conversion_time = time.time() - dense_conversion_start

    logger.info(f"Dense conversion for one sub-tensor: {dense_conversion_time * 1000:.2f}ms")

    # Estimate dense approach would take much longer
    estimated_dense_time = dense_conversion_time * batch_size * channels * num_targets
    logger.info(f"Estimated dense approach time: {estimated_dense_time:.2f}s")

    if estimated_dense_time > sparse_time:
        speedup = estimated_dense_time / sparse_time
        logger.info(f"Estimated speedup: {speedup:.1f}x")

    logger.info("✓ Performance comparison completed")


def test_save_load():
    """Test save/load functionality for npz integration."""
    logger.info("=== Testing Save/Load Functionality ===")

    # Create original tensor with some data
    original = SingleBlockMixedSparseTensor(8, 2, 40, 50, 8)

    # Set some blocks with known values
    test_data = [
        (0, 0, 5, 10, cp.ones((8, 8), dtype=cp.float32) * 1.5),
        (1, 1, 15, 20, cp.ones((8, 8), dtype=cp.float32) * 2.5),
        (3, 0, 25, 5, cp.ones((8, 8), dtype=cp.float32) * 3.5),
        (7, 1, 10, 35, cp.ones((8, 8), dtype=cp.float32) * 4.5),
    ]

    for batch_idx, channel_idx, row, col, values in test_data:
        original.set_block(batch_idx, channel_idx, row, col, values)

    # Export to dictionary
    data_dict = original.to_dict()
    logger.info(f"Exported {len(data_dict)} arrays to dictionary")

    # Verify dictionary contains expected keys (blocks_set removed)
    expected_keys = {
        "sparse_values",
        "block_positions",
        "batch_size",
        "channels",
        "height",
        "width",
        "block_size",
        "device",
        "dtype",
        "output_dtype",
    }
    assert set(data_dict.keys()) == expected_keys, f"Missing keys: {expected_keys - set(data_dict.keys())}"

    # Verify data types are numpy arrays
    for key, value in data_dict.items():
        assert isinstance(value, np.ndarray), f"Key {key} is not a numpy array"

    # Load from dictionary
    loaded = SingleBlockMixedSparseTensor.from_dict(data_dict)
    logger.info(f"Loaded tensor: {loaded}")

    # Verify loaded tensor matches original
    assert loaded.batch_size == original.batch_size
    assert loaded.channels == original.channels
    assert loaded.height == original.height
    assert loaded.width == original.width
    assert loaded.block_size == original.block_size

    # Verify block data matches (blocks_set no longer exists)
    assert cp.allclose(loaded.sparse_values, original.sparse_values)
    assert cp.array_equal(loaded.block_positions, original.block_positions)

    # Verify same computational results
    target_3d = cp.random.rand(2, 40, 50).astype(cp.float32)
    original_result = original.transpose_dot_product_3d(target_3d)
    loaded_result = loaded.transpose_dot_product_3d(target_3d)

    assert cp.allclose(original_result, loaded_result), "Loaded tensor produces different results"

    # Test edge case: empty tensor
    empty_tensor = SingleBlockMixedSparseTensor(3, 2, 20, 30, 4)
    empty_dict = empty_tensor.to_dict()
    empty_loaded = SingleBlockMixedSparseTensor.from_dict(empty_dict)

    assert empty_loaded.batch_size == 3
    assert empty_loaded.channels == 2
    # All blocks are assumed to be set - no explicit check needed

    logger.info("✓ Save/load functionality test passed")


def test_error_handling():
    """Test error handling and edge cases."""
    logger.info("=== Testing Error Handling ===")

    tensor = SingleBlockMixedSparseTensor(5, 2, 50, 60, 8)

    # Test out-of-bounds indices
    try:
        tensor.set_block(10, 0, 0, 0, cp.ones((8, 8)))
        raise AssertionError("Should have raised ValueError for batch_idx out of bounds")
    except ValueError:
        pass

    try:
        tensor.set_block(0, 5, 0, 0, cp.ones((8, 8)))
        raise AssertionError("Should have raised ValueError for channel_idx out of bounds")
    except ValueError:
        pass

    # Test out-of-bounds positions
    try:
        tensor.set_block(0, 0, 50, 0, cp.ones((8, 8)))  # row too large
        raise AssertionError("Should have raised ValueError for row out of bounds")
    except ValueError:
        pass

    try:
        tensor.set_block(0, 0, 0, 60, cp.ones((8, 8)))  # col too large
        raise AssertionError("Should have raised ValueError for col out of bounds")
    except ValueError:
        pass

    # Test wrong value shape
    try:
        tensor.set_block(0, 0, 0, 0, cp.ones((10, 10)))  # wrong block size
        raise AssertionError("Should have raised ValueError for wrong value shape")
    except ValueError:
        pass

    logger.info("✓ Error handling test passed")


def test_to_dense_patterns():
    """Test conversion to dense patterns array."""
    logger.info("=== Testing to_dense_patterns ===")

    # Create a small tensor with known patterns
    batch_size, channels = 3, 2
    height, width = 50, 60
    block_size = 8

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size)

    # Set blocks with known values and positions
    test_cases = [
        (0, 0, 10, 15, cp.ones((8, 8), dtype=cp.float32) * 2.0),  # LED 0, Red
        (0, 1, 20, 25, cp.ones((8, 8), dtype=cp.float32) * 3.0),  # LED 0, Green
        (1, 0, 5, 30, cp.ones((8, 8), dtype=cp.float32) * 4.0),  # LED 1, Red
        (1, 1, 35, 10, cp.ones((8, 8), dtype=cp.float32) * 5.0),  # LED 1, Green
        (2, 0, 25, 45, cp.ones((8, 8), dtype=cp.float32) * 6.0),  # LED 2, Red
        (2, 1, 15, 5, cp.ones((8, 8), dtype=cp.float32) * 7.0),  # LED 2, Green
    ]

    for batch_idx, channel_idx, row, col, values in test_cases:
        tensor.set_block(batch_idx, channel_idx, row, col, values)

    # Convert to dense patterns
    dense_patterns = tensor.to_dense_patterns()

    # Check shape
    assert dense_patterns.shape == (batch_size, height, width, channels)
    assert dense_patterns.dtype == np.float32

    # Verify specific patterns
    for batch_idx, channel_idx, row, col, expected_value in test_cases:
        # Check that the block region has the expected value
        block_region = dense_patterns[batch_idx, row : row + block_size, col : col + block_size, channel_idx]
        assert np.allclose(block_region, expected_value), f"LED {batch_idx}, channel {channel_idx} block mismatch"

        # Check that individual extraction matches
        individual_pattern = tensor.extract_pattern(batch_idx, channel_idx)
        batch_pattern = dense_patterns[batch_idx, :, :, channel_idx]
        np.testing.assert_array_equal(
            individual_pattern,
            batch_pattern,
            err_msg=f"Mismatch between extract_pattern and to_dense_patterns for LED "
            f"{batch_idx}, channel {channel_idx}",
        )

    logger.info("✓ to_dense_patterns test passed")


def test_get_block_summary():
    """Test block summary statistics."""
    logger.info("=== Testing get_block_summary ===")

    # Create tensor with known patterns
    batch_size, channels = 4, 3
    height, width = 64, 64
    block_size = 8

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size)

    # Set blocks with known values at different positions
    test_data = [
        (0, 0, 10, 20, 1.0),  # LED 0, Red
        (0, 1, 15, 25, 2.0),  # LED 0, Green
        (1, 2, 30, 40, 3.0),  # LED 1, Blue
        (2, 0, 5, 5, 4.0),  # LED 2, Red
        (3, 1, 50, 50, 5.0),  # LED 3, Green
    ]

    for batch_idx, channel_idx, row, col, value in test_data:
        values = cp.ones((block_size, block_size), dtype=cp.float32) * value
        tensor.set_block(batch_idx, channel_idx, row, col, values)

    # Get summary statistics
    summary = tensor.get_block_summary()

    # Check required keys
    required_keys = {
        "batch_size",
        "channels",
        "block_size",
        "total_blocks",
        "block_positions_stats",
        "intensity_stats",
        "coverage_stats",
        "memory_mb",
    }
    assert set(summary.keys()) == required_keys

    # Check basic properties
    assert summary["batch_size"] == batch_size
    assert summary["channels"] == channels
    assert summary["block_size"] == block_size
    assert summary["total_blocks"] == batch_size * channels

    # Check position statistics
    pos_stats = summary["block_positions_stats"]
    assert pos_stats["min_row"] >= 0
    assert pos_stats["max_row"] < height
    assert pos_stats["min_col"] >= 0
    assert pos_stats["max_col"] < width
    assert pos_stats["min_row"] <= pos_stats["max_row"]
    assert pos_stats["min_col"] <= pos_stats["max_col"]

    # Check intensity statistics
    intensity_stats = summary["intensity_stats"]
    assert intensity_stats["min_intensity"] >= 0
    assert intensity_stats["max_intensity"] >= intensity_stats["min_intensity"]
    assert 0 <= intensity_stats["sparsity_ratio"] <= 1
    assert intensity_stats["nonzero_values"] > 0  # We set some blocks

    # Check coverage statistics
    coverage_stats = summary["coverage_stats"]
    assert coverage_stats["blocks_per_led"] == channels
    assert 0 <= coverage_stats["spatial_coverage_x"] <= 1
    assert 0 <= coverage_stats["spatial_coverage_y"] <= 1
    assert coverage_stats["led_max_intensities"].shape == (batch_size,)
    assert coverage_stats["led_mean_intensities"].shape == (batch_size,)

    # Check memory info
    assert summary["memory_mb"] > 0

    logger.info("✓ get_block_summary test passed")


def test_extract_pattern():
    """Test individual pattern extraction."""
    logger.info("=== Testing extract_pattern ===")

    # Create tensor with known patterns
    batch_size, channels = 3, 2
    height, width = 40, 50
    block_size = 8

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size)

    # Set specific patterns
    test_cases = [
        (0, 0, 5, 10, 2.5),  # LED 0, Red
        (1, 1, 20, 30, 7.8),  # LED 1, Green
        (2, 0, 15, 5, 1.2),  # LED 2, Red
    ]

    for batch_idx, channel_idx, row, col, value in test_cases:
        values = cp.ones((block_size, block_size), dtype=cp.float32) * value
        tensor.set_block(batch_idx, channel_idx, row, col, values)

    # Test pattern extraction
    for batch_idx, channel_idx, row, col, expected_value in test_cases:
        pattern = tensor.extract_pattern(batch_idx, channel_idx)

        # Check shape and type
        assert pattern.shape == (height, width)
        assert pattern.dtype == np.float32

        # Check that block region has expected value
        block_region = pattern[row : row + block_size, col : col + block_size]
        assert np.allclose(block_region, expected_value), f"LED {batch_idx}, channel {channel_idx} value mismatch"

        # Check that areas outside block are zero
        outside_mask = np.ones((height, width), dtype=bool)
        outside_mask[row : row + block_size, col : col + block_size] = False
        assert np.allclose(
            pattern[outside_mask], 0
        ), f"LED {batch_idx}, channel {channel_idx} has non-zero values outside block"

    # Test error handling
    try:
        tensor.extract_pattern(batch_size, 0)  # Invalid LED index
        raise AssertionError("Should have raised ValueError for invalid LED index")
    except ValueError:
        pass

    try:
        tensor.extract_pattern(0, channels)  # Invalid channel index
        raise AssertionError("Should have raised ValueError for invalid channel index")
    except ValueError:
        pass

    logger.info("✓ extract_pattern test passed")


def test_enhancement_methods_consistency():
    """Test consistency between enhancement methods."""
    logger.info("=== Testing Enhancement Methods Consistency ===")

    # Create tensor with diverse patterns
    batch_size, channels = 5, 3
    height, width = 80, 100
    block_size = 16

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size)

    # Set various blocks with different intensities
    np.random.seed(42)  # For reproducible test
    for batch_idx in range(batch_size):
        for channel_idx in range(channels):
            if np.random.random() > 0.3:  # Set ~70% of blocks
                row = np.random.randint(0, height - block_size)
                col = np.random.randint(0, width - block_size)
                intensity = np.random.uniform(0.1, 1.0)
                values = cp.ones((block_size, block_size), dtype=cp.float32) * intensity
                tensor.set_block(batch_idx, channel_idx, row, col, values)

    # Get data from all methods
    dense_patterns = tensor.to_dense_patterns()
    summary = tensor.get_block_summary()

    # Check consistency between methods
    # 1. Dense patterns shape matches summary info
    assert dense_patterns.shape[0] == summary["batch_size"]
    assert dense_patterns.shape[3] == summary["channels"]

    # 2. Individual extraction matches dense patterns
    for batch_idx in range(min(3, batch_size)):  # Test first 3 LEDs
        for channel_idx in range(channels):
            individual = tensor.extract_pattern(batch_idx, channel_idx)
            from_dense = dense_patterns[batch_idx, :, :, channel_idx]
            np.testing.assert_array_equal(
                individual,
                from_dense,
                err_msg=f"Inconsistency for LED {batch_idx}, channel {channel_idx}",
            )

    # 3. Summary statistics are consistent with actual data
    max_intensities = summary["coverage_stats"]["led_max_intensities"]
    for batch_idx in range(batch_size):
        led_max_actual = np.max(dense_patterns[batch_idx])
        assert np.isclose(max_intensities[batch_idx], led_max_actual), f"Max intensity mismatch for LED {batch_idx}"

    logger.info("✓ Enhancement methods consistency test passed")


@pytest.mark.parametrize("dtype", [cp.float32, cp.uint8])
def test_dtype_support(dtype):
    """Test tensor creation and basic operations with different dtypes."""
    logger.info(f"=== Testing dtype support: {dtype} ===")

    # Create tensor with specified dtype
    tensor = SingleBlockMixedSparseTensor(batch_size=3, channels=2, height=40, width=50, block_size=8, dtype=dtype)

    # Verify dtype is set correctly
    assert tensor.dtype == dtype
    assert tensor.sparse_values.dtype == dtype

    # Create test data of appropriate dtype
    if dtype == cp.float32:
        test_value = 0.5
        values = cp.ones((8, 8), dtype=dtype) * test_value
    else:  # uint8
        test_value = 128
        values = cp.ones((8, 8), dtype=dtype) * test_value

    # Test setting blocks with correct dtype
    tensor.set_block(0, 0, 5, 10, values)

    # Verify block was set correctly
    block_info = tensor.get_block_info(0, 0)
    assert cp.allclose(block_info["values"], test_value)

    logger.info(f"✓ dtype support test passed for {dtype}")


def test_dtype_validation():
    """Test dtype validation in constructor and methods."""
    logger.info("=== Testing dtype validation ===")

    # Test unsupported dtype in constructor
    try:
        SingleBlockMixedSparseTensor(5, 2, 40, 50, 8, dtype=cp.int32)
        raise AssertionError("Should have raised ValueError for unsupported dtype")
    except ValueError:
        pass

    # Test dtype mismatch in set_block
    tensor = SingleBlockMixedSparseTensor(5, 2, 40, 50, 8, dtype=cp.float32)

    try:
        wrong_dtype_values = cp.ones((8, 8), dtype=cp.uint8)
        tensor.set_block(0, 0, 5, 10, wrong_dtype_values)
        raise AssertionError("Should have raised ValueError for dtype mismatch")
    except ValueError:
        pass

    # Test dtype mismatch in set_blocks_batch
    try:
        positions = cp.zeros((2, 5, 2), dtype=cp.int32)
        wrong_dtype_values = cp.ones((2, 5, 8, 8), dtype=cp.uint8)
        tensor.set_blocks_batch(positions, wrong_dtype_values)
        raise AssertionError("Should have raised ValueError for dtype mismatch")
    except ValueError:
        pass

    logger.info("✓ dtype validation test passed")


def test_file_io_with_dtype():
    """Test save/load functionality preserves dtype."""
    logger.info("=== Testing file I/O with dtype ===")

    for dtype in [cp.float32, cp.uint8]:
        logger.info(f"Testing file I/O with {dtype}")

        # Create tensor with specific dtype
        original = SingleBlockMixedSparseTensor(4, 2, 30, 40, 8, dtype=dtype)

        # Set some test data
        if dtype == cp.float32:
            test_values = [0.1, 0.5, 0.9, 1.0]
        else:  # uint8
            test_values = [25, 128, 200, 255]

        for i, test_value in enumerate(test_values):
            values = cp.ones((8, 8), dtype=dtype) * test_value
            original.set_block(i, 0, 5 + i * 5, 10, values)

        # Save to dict
        data_dict = original.to_dict()
        assert "dtype" in data_dict
        assert str(data_dict["dtype"]) == dtype.__name__

        # Load from dict
        loaded = SingleBlockMixedSparseTensor.from_dict(data_dict)

        # Verify dtype is preserved
        assert loaded.dtype == dtype
        assert loaded.sparse_values.dtype == dtype

        # Verify data is preserved
        assert cp.allclose(loaded.sparse_values, original.sparse_values)

    logger.info("✓ file I/O dtype test passed")


def test_int8_fp32_equivalence():
    """
    Test that int8 and fp32 kernels produce equivalent results.

    Strategy:
    1. Create int8 tensor and target data
    2. Convert int8 data to fp32 (unscaled: 0-255 range)
    3. Run fp32 kernel and divide result by (255*255)
    4. Run int8 kernel directly
    5. Compare results - should be identical
    """
    logger.info("=== Testing int8/fp32 equivalence ===")

    # Test parameters
    batch_size, channels = 5, 3
    height, width = 64, 80
    block_size = 32  # Multiple of 4 for vectorization

    # Create test data in int8 range [0, 255]
    np.random.seed(42)  # For reproducibility
    int8_sparse_data = np.random.randint(0, 256, (channels, batch_size, block_size, block_size), dtype=np.uint8)
    int8_target_data = np.random.randint(0, 256, (channels, height, width), dtype=np.uint8)
    positions = np.random.randint(0, min(height, width) - block_size, (channels, batch_size, 2), dtype=np.int32)
    # Align x-coordinates to multiples of 4 for uint8 vectorization
    positions[:, :, 1] = (positions[:, :, 1] // 4) * 4

    # Convert to CuPy arrays
    int8_sparse_cupy = cp.asarray(int8_sparse_data)
    int8_target_cupy = cp.asarray(int8_target_data)
    positions_cupy = cp.asarray(positions)

    # 1. Create int8 tensor and compute result
    int8_tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size, dtype=cp.uint8)
    int8_tensor.set_blocks_batch(positions_cupy, int8_sparse_cupy)

    int8_result = int8_tensor.transpose_dot_product_3d(int8_target_cupy)

    # 2. Create fp32 tensor with unscaled data (0-255 range) and compute result
    fp32_sparse_data = int8_sparse_data.astype(np.float32)  # Unscaled conversion
    fp32_target_data = int8_target_data.astype(np.float32)  # Unscaled conversion

    fp32_sparse_cupy = cp.asarray(fp32_sparse_data)
    fp32_target_cupy = cp.asarray(fp32_target_data)

    fp32_tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size, dtype=cp.float32)
    fp32_tensor.set_blocks_batch(positions_cupy, fp32_sparse_cupy)

    fp32_result = fp32_tensor.transpose_dot_product_3d(fp32_target_cupy)

    # 3. Scale fp32 result by (255*255) to match int8 kernel normalization
    fp32_result_scaled = fp32_result / (255.0 * 255.0)

    # 4. Compare results - should be identical (or very close due to floating point precision)
    logger.info(f"int8 result sample: {int8_result[0, :3]}")
    logger.info(f"fp32 scaled result sample: {fp32_result_scaled[0, :3]}")
    logger.info(f"Max absolute difference: {cp.max(cp.abs(int8_result - fp32_result_scaled))}")

    # Allow small tolerance for floating point precision differences
    cp.testing.assert_allclose(
        int8_result,
        fp32_result_scaled,
        rtol=1e-6,
        atol=1e-8,
        err_msg="int8 and fp32 (scaled) results should be equivalent",
    )

    logger.info("✓ int8/fp32 equivalence test passed - results are mathematically equivalent!")


def test_memory_efficiency_comparison():
    """Test memory efficiency between int8 and fp32 tensors."""
    logger.info("=== Testing memory efficiency comparison ===")

    # Create tensors with same dimensions but different dtypes
    batch_size, channels = 100, 3
    height, width = 200, 300
    block_size = 64

    fp32_tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size, dtype=cp.float32)

    int8_tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size, dtype=cp.uint8)

    # Get memory info
    fp32_memory = fp32_tensor.memory_info()
    int8_memory = int8_tensor.memory_info()

    logger.info(f"fp32 memory usage: {fp32_memory['total_mb']:.2f}MB")
    logger.info(f"int8 memory usage: {int8_memory['total_mb']:.2f}MB")

    # int8 should use ~4x less memory for sparse_values
    memory_ratio = fp32_memory["total_mb"] / int8_memory["total_mb"]
    logger.info(f"Memory reduction factor: {memory_ratio:.2f}x")

    # Should be close to 4x reduction (exact ratio depends on position storage overhead)
    assert memory_ratio > 3.0, f"Expected >3x memory reduction, got {memory_ratio:.2f}x"

    logger.info("✓ memory efficiency comparison test passed")


@pytest.mark.parametrize("dtype", [cp.float32, cp.uint8])
def test_transpose_dot_product_3d_dtypes(dtype):
    """Test transpose_dot_product_3d with different dtypes."""
    logger.info(f"=== Testing transpose_dot_product_3d with {dtype} ===")

    batch_size, channels = 3, 2
    height, width = 48, 64
    block_size = 16

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size, dtype=dtype)

    # Create test data of appropriate dtype and range
    if dtype == cp.float32:
        # Use [0, 1] range for fp32
        sparse_data = cp.random.rand(channels, batch_size, block_size, block_size).astype(dtype)
        target_data = cp.random.rand(channels, height, width).astype(dtype)
    else:  # uint8
        # Use [0, 255] range for int8
        sparse_data = cp.random.randint(0, 256, (channels, batch_size, block_size, block_size), dtype=dtype)
        target_data = cp.random.randint(0, 256, (channels, height, width), dtype=dtype)

    positions = cp.random.randint(0, min(height, width) - block_size, (channels, batch_size, 2))
    # Align x-coordinates to multiples of 4 for uint8 vectorization
    positions[:, :, 1] = (positions[:, :, 1] // 4) * 4

    # Set data
    tensor.set_blocks_batch(positions, sparse_data)

    # Test dtype mismatch error
    if dtype == cp.float32:
        wrong_target = cp.random.randint(0, 256, (channels, height, width), dtype=cp.uint8)
    else:
        wrong_target = cp.random.rand(channels, height, width).astype(cp.float32)

    try:
        tensor.transpose_dot_product_3d(wrong_target)
        raise AssertionError("Should have raised ValueError for dtype mismatch")
    except ValueError:
        pass

    # Test correct computation
    result = tensor.transpose_dot_product_3d(target_data)

    # Verify output shape and dtype
    assert result.shape == (batch_size, channels)
    assert result.dtype == cp.float32  # Output is always fp32

    logger.info(f"✓ transpose_dot_product_3d test passed for {dtype}")


def test_planar_output_basic_functionality():
    """Test basic functionality of the planar_output parameter."""
    logger.info("=== Testing planar_output basic functionality ===")

    batch_size, channels = 5, 3
    height, width = 64, 80
    block_size = 16

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size, dtype=cp.float32)

    # Set some test data
    sparse_data = cp.random.rand(channels, batch_size, block_size, block_size).astype(cp.float32)
    positions = cp.random.randint(0, min(height, width) - block_size, (channels, batch_size, 2))
    # Align x-coordinates to multiples of 4 for uint8 vectorization
    positions[:, :, 1] = (positions[:, :, 1] // 4) * 4
    target_data = cp.random.rand(channels, height, width).astype(cp.float32)

    tensor.set_blocks_batch(positions, sparse_data)

    # Test default behavior (planar_output=False)
    result_interleaved = tensor.transpose_dot_product_3d(target_data, planar_output=False)
    assert result_interleaved.shape == (batch_size, channels)
    assert result_interleaved.flags.c_contiguous

    # Test planar output (planar_output=True)
    result_planar = tensor.transpose_dot_product_3d(target_data, planar_output=True)
    assert result_planar.shape == (channels, batch_size)
    assert result_planar.flags.c_contiguous

    logger.info("✓ planar_output basic functionality test passed")


@pytest.mark.parametrize("dtype", [cp.float32, cp.uint8])
def test_planar_output_equivalence(dtype):
    """Test that planar_output=True/False produce equivalent results."""
    logger.info(f"=== Testing planar_output equivalence with {dtype} ===")

    batch_size, channels = 4, 3
    height, width = 48, 64
    block_size = 16

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size, dtype=dtype)

    # Create test data of appropriate dtype
    if dtype == cp.float32:
        sparse_data = cp.random.rand(channels, batch_size, block_size, block_size).astype(dtype)
        target_data = cp.random.rand(channels, height, width).astype(dtype)
    else:  # uint8
        sparse_data = cp.random.randint(0, 256, (channels, batch_size, block_size, block_size), dtype=dtype)
        target_data = cp.random.randint(0, 256, (channels, height, width), dtype=dtype)

    positions = cp.random.randint(0, min(height, width) - block_size, (channels, batch_size, 2))
    # Align x-coordinates for uint8 vectorization
    positions[:, :, 1] = (positions[:, :, 1] // 4) * 4

    tensor.set_blocks_batch(positions, sparse_data)

    # Get results in both formats
    result_interleaved = tensor.transpose_dot_product_3d(target_data, planar_output=False)  # (batch_size, channels)
    result_planar = tensor.transpose_dot_product_3d(target_data, planar_output=True)  # (channels, batch_size)

    # Transpose one to match the other for comparison
    result_interleaved_transposed = result_interleaved.T  # (channels, batch_size)

    # Results should be numerically identical
    cp.testing.assert_allclose(
        result_planar,
        result_interleaved_transposed,
        rtol=1e-6,
        atol=1e-8,
        err_msg=f"planar_output results should be equivalent for {dtype}",
    )

    logger.info(f"✓ planar_output equivalence test passed for {dtype}")


def test_planar_output_memory_layout():
    """Test memory layout properties of planar_output results."""
    logger.info("=== Testing planar_output memory layout ===")

    batch_size, channels = 6, 3
    height, width = 64, 80
    block_size = 16

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size, dtype=cp.float32)

    # Set test data
    sparse_data = cp.random.rand(channels, batch_size, block_size, block_size).astype(cp.float32)
    positions = cp.random.randint(0, min(height, width) - block_size, (channels, batch_size, 2))
    # Align x-coordinates to multiples of 4 for uint8 vectorization
    positions[:, :, 1] = (positions[:, :, 1] // 4) * 4
    target_data = cp.random.rand(channels, height, width).astype(cp.float32)

    tensor.set_blocks_batch(positions, sparse_data)

    # Test interleaved output memory layout
    result_interleaved = tensor.transpose_dot_product_3d(target_data, planar_output=False)
    assert result_interleaved.shape == (batch_size, channels)
    assert result_interleaved.flags.c_contiguous
    assert not result_interleaved.flags.f_contiguous
    expected_strides_interleaved = (channels * result_interleaved.itemsize, result_interleaved.itemsize)
    assert result_interleaved.strides == expected_strides_interleaved

    # Test planar output memory layout
    result_planar = tensor.transpose_dot_product_3d(target_data, planar_output=True)
    assert result_planar.shape == (channels, batch_size)
    assert result_planar.flags.c_contiguous
    assert not result_planar.flags.f_contiguous
    expected_strides_planar = (batch_size * result_planar.itemsize, result_planar.itemsize)
    assert result_planar.strides == expected_strides_planar

    # Demonstrate the old problem: transposing interleaved creates F-contiguous
    result_interleaved_transposed = result_interleaved.T
    assert result_interleaved_transposed.shape == (channels, batch_size)
    assert not result_interleaved_transposed.flags.c_contiguous
    assert result_interleaved_transposed.flags.f_contiguous
    # F-contiguous has interleaved strides
    f_contiguous_strides = (result_interleaved.itemsize, channels * result_interleaved.itemsize)
    assert result_interleaved_transposed.strides == f_contiguous_strides

    logger.info("✓ planar_output memory layout test passed")


def test_planar_output_performance_benefit():
    """Test that planar_output eliminates transpose operations."""
    logger.info("=== Testing planar_output performance benefit ===")

    batch_size, channels = 10, 3
    height, width = 64, 80
    block_size = 16

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size, dtype=cp.float32)

    # Set test data
    sparse_data = cp.random.rand(channels, batch_size, block_size, block_size).astype(cp.float32)
    positions = cp.random.randint(0, min(height, width) - block_size, (channels, batch_size, 2))
    # Align x-coordinates to multiples of 4 for uint8 vectorization
    positions[:, :, 1] = (positions[:, :, 1] // 4) * 4
    target_data = cp.random.rand(channels, height, width).astype(cp.float32)

    tensor.set_blocks_batch(positions, sparse_data)

    # Method 1: Old approach (interleaved + transpose + fix)
    result_old = tensor.transpose_dot_product_3d(target_data, planar_output=False)  # (batch_size, channels)
    result_old_transposed = result_old.T  # Creates F-contiguous view
    result_old_fixed = cp.ascontiguousarray(result_old_transposed)  # Fix memory layout

    # Method 2: New approach (direct planar output)
    result_new = tensor.transpose_dot_product_3d(target_data, planar_output=True)  # (channels, batch_size)

    # Verify results are identical
    cp.testing.assert_allclose(result_old_fixed, result_new, rtol=1e-6, atol=1e-8)

    # Verify memory layout properties
    assert result_old_fixed.flags.c_contiguous  # Required cp.ascontiguousarray()
    assert result_new.flags.c_contiguous  # Direct C-contiguous output

    # Verify no additional operations needed for new approach
    assert result_new.flags.owndata  # Direct allocation, not a view
    assert not result_old_transposed.flags.owndata  # Transpose is a view
    assert result_old_fixed.flags.owndata  # ascontiguousarray creates new allocation

    logger.info("✓ planar_output performance benefit test passed")


@pytest.mark.parametrize("output_dtype", [cp.float32, cp.float16])
def test_planar_output_with_output_dtype(output_dtype):
    """Test planar_output works correctly with different output dtypes."""
    logger.info(f"=== Testing planar_output with output_dtype={output_dtype} ===")

    batch_size, channels = 4, 2
    height, width = 32, 48
    block_size = 8

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size, dtype=cp.float32)

    # Set test data
    sparse_data = cp.random.rand(channels, batch_size, block_size, block_size).astype(cp.float32)
    positions = cp.random.randint(0, min(height, width) - block_size, (channels, batch_size, 2))
    # Align x-coordinates to multiples of 4 for uint8 vectorization
    positions[:, :, 1] = (positions[:, :, 1] // 4) * 4
    target_data = cp.random.rand(channels, height, width).astype(cp.float32)

    tensor.set_blocks_batch(positions, sparse_data)

    # Test both planar_output modes with specified output dtype
    result_interleaved = tensor.transpose_dot_product_3d(target_data, output_dtype=output_dtype, planar_output=False)
    result_planar = tensor.transpose_dot_product_3d(target_data, output_dtype=output_dtype, planar_output=True)

    # Verify shapes and dtypes
    assert result_interleaved.shape == (batch_size, channels)
    assert result_planar.shape == (channels, batch_size)
    assert result_interleaved.dtype == output_dtype
    assert result_planar.dtype == output_dtype

    # Verify results are equivalent (accounting for dtype precision)
    if output_dtype == cp.float16:
        # Lower precision for FP16
        rtol, atol = 1e-3, 1e-4
    else:
        # Higher precision for FP32
        rtol, atol = 1e-6, 1e-8

    cp.testing.assert_allclose(result_planar, result_interleaved.T, rtol=rtol, atol=atol)

    logger.info(f"✓ planar_output with output_dtype={output_dtype} test passed")


def test_planar_output_backward_compatibility():
    """Test that the default behavior maintains backward compatibility."""
    logger.info("=== Testing planar_output backward compatibility ===")

    batch_size, channels = 3, 2
    height, width = 32, 48
    block_size = 8

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size, dtype=cp.float32)

    # Set test data
    sparse_data = cp.random.rand(channels, batch_size, block_size, block_size).astype(cp.float32)
    positions = cp.random.randint(0, min(height, width) - block_size, (channels, batch_size, 2))
    # Align x-coordinates to multiples of 4 for uint8 vectorization
    positions[:, :, 1] = (positions[:, :, 1] // 4) * 4
    target_data = cp.random.rand(channels, height, width).astype(cp.float32)

    tensor.set_blocks_batch(positions, sparse_data)

    # Test that default behavior (no planar_output specified) matches planar_output=False
    result_default = tensor.transpose_dot_product_3d(target_data)
    result_explicit_false = tensor.transpose_dot_product_3d(target_data, planar_output=False)

    # Results should be identical
    cp.testing.assert_array_equal(result_default, result_explicit_false)

    # Both should have (batch_size, channels) shape
    assert result_default.shape == (batch_size, channels)
    assert result_explicit_false.shape == (batch_size, channels)

    logger.info("✓ planar_output backward compatibility test passed")


def main():
    """Run all tests."""
    logger.info("Starting SingleBlockMixedSparseTensor tests...")

    try:
        test_basic_functionality()
        test_batch_operations()
        test_memory_efficiency()
        test_performance_comparison()
        test_save_load()
        test_error_handling()

        # New enhancement method tests
        test_to_dense_patterns()
        test_get_block_summary()
        test_extract_pattern()
        test_enhancement_methods_consistency()

        # New dtype support tests
        test_dtype_support(cp.float32)
        test_dtype_support(cp.uint8)
        test_dtype_validation()
        test_file_io_with_dtype()
        test_int8_fp32_equivalence()
        test_memory_efficiency_comparison()
        test_transpose_dot_product_3d_dtypes(cp.float32)
        test_transpose_dot_product_3d_dtypes(cp.uint8)

        # Planar output optimization tests
        test_planar_output_basic_functionality()
        test_planar_output_equivalence(cp.float32)
        test_planar_output_equivalence(cp.uint8)
        test_planar_output_memory_layout()
        test_planar_output_performance_benefit()
        test_planar_output_with_output_dtype(cp.float32)
        test_planar_output_with_output_dtype(cp.float16)
        test_planar_output_backward_compatibility()

        logger.info("🎉 All tests passed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
