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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_basic_functionality():
    """Test basic tensor creation and block setting."""
    logger.info("=== Testing Basic Functionality ===")

    # Create a small tensor for testing
    tensor = SingleBlockMixedSparseTensor(
        batch_size=5, channels=3, height=100, width=80, block_size=8
    )

    logger.info(f"Created tensor: {tensor}")

    # Test setting individual blocks
    for batch_idx in range(5):
        for channel_idx in range(3):
            # Create a simple test pattern
            values = cp.ones((8, 8), dtype=cp.float32) * (
                batch_idx * 3 + channel_idx + 1
            )

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


def test_transpose_dot_product():
    """Test the core A^T @ b operation."""
    logger.info("=== Testing Transpose Dot Product ===")

    # Create a small tensor with known patterns
    batch_size, channels = 3, 2
    height, width = 50, 60
    block_size = 8

    tensor = SingleBlockMixedSparseTensor(
        batch_size, channels, height, width, block_size
    )

    # Set blocks with known values and positions
    test_cases = [
        (0, 0, 10, 15, cp.ones((8, 8)) * 2.0),  # LED 0, Red: all 2.0s at (10,15)
        (0, 1, 20, 25, cp.ones((8, 8)) * 3.0),  # LED 0, Green: all 3.0s at (20,25)
        (1, 0, 5, 30, cp.ones((8, 8)) * 4.0),  # LED 1, Red: all 4.0s at (5,30)
        (1, 1, 35, 10, cp.ones((8, 8)) * 5.0),  # LED 1, Green: all 5.0s at (35,10)
        (2, 0, 25, 45, cp.ones((8, 8)) * 6.0),  # LED 2, Red: all 6.0s at (25,45)
        (2, 1, 15, 5, cp.ones((8, 8)) * 7.0),  # LED 2, Green: all 7.0s at (15,5)
    ]

    for batch_idx, channel_idx, row, col, values in test_cases:
        tensor.set_block(batch_idx, channel_idx, row, col, values)

    # Create a target image with known values
    target = cp.zeros((height, width), dtype=cp.float32)
    target[10:18, 15:23] = 1.0  # Overlaps with LED 0, Red (2.0 * 1.0 * 64 = 128)
    target[20:28, 25:33] = 0.5  # Overlaps with LED 0, Green (3.0 * 0.5 * 64 = 96)
    target[5:13, 30:38] = 2.0  # Overlaps with LED 1, Red (4.0 * 2.0 * 64 = 512)

    # Compute A^T @ b
    result = tensor.transpose_dot_product(target)

    logger.info(f"Transpose dot product result shape: {result.shape}")
    logger.info(f"Result values:\n{result}")

    # Verify expected results
    expected = cp.array(
        [
            [128.0, 96.0],  # LED 0: Red=2*1*64, Green=3*0.5*64
            [512.0, 0.0],  # LED 1: Red=4*2*64, Green=0 (no overlap)
            [0.0, 0.0],  # LED 2: No overlaps
        ]
    )

    assert cp.allclose(
        result, expected, rtol=1e-5
    ), f"Expected {expected}, got {result}"
    logger.info("✓ Transpose dot product test passed")


def test_batch_operations():
    """Test batch setting operations."""
    logger.info("=== Testing Batch Operations ===")

    batch_size, channels = 10, 3
    height, width = 64, 64
    block_size = 16

    tensor = SingleBlockMixedSparseTensor(
        batch_size, channels, height, width, block_size
    )

    # Create batch data
    positions = cp.random.randint(0, height - block_size, (batch_size, channels, 2))
    values = cp.random.rand(batch_size, channels, block_size, block_size).astype(
        cp.float32
    )

    # Set blocks in batch
    start_time = time.time()
    tensor.set_blocks_batch(positions, values)
    batch_set_time = time.time() - start_time

    logger.info(f"Batch block setting took {batch_set_time:.4f}s")

    # Verify all blocks were set
    assert cp.all(tensor.blocks_set), "Not all blocks were set in batch operation"

    # Test single target computation
    target = cp.random.rand(height, width).astype(cp.float32)
    result = tensor.transpose_dot_product(target)

    logger.info(f"Single target result shape: {result.shape}")
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

    tensor = SingleBlockMixedSparseTensor(
        batch_size, channels, height, width, block_size
    )

    # Get initial memory info
    initial_memory = tensor.memory_info()
    logger.info(f"Initial memory usage: {initial_memory['total_mb']:.1f}MB")
    logger.info(
        f"Equivalent dense would be: {initial_memory['equivalent_dense_mb']:.1f}MB"
    )
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

    tensor = SingleBlockMixedSparseTensor(
        batch_size, channels, height, width, block_size
    )

    # Set all blocks with random data
    positions = cp.random.randint(0, height - block_size, (batch_size, channels, 2))
    values = cp.random.rand(batch_size, channels, block_size, block_size).astype(
        cp.float32
    )
    tensor.set_blocks_batch(positions, values)

    # Create multiple target matrices
    num_targets = 10
    targets = cp.random.rand(num_targets, height, width).astype(cp.float32)

    # Time sparse implementation
    cp.cuda.Device().synchronize()
    start_time = time.time()

    sparse_results = []
    for i in range(num_targets):
        result = tensor.transpose_dot_product(targets[i])
        sparse_results.append(result)

    cp.cuda.Device().synchronize()
    sparse_time = time.time() - start_time

    logger.info(f"Sparse implementation: {sparse_time:.4f}s for {num_targets} targets")
    logger.info(f"Sparse per target: {sparse_time / num_targets * 1000:.2f}ms")

    # For comparison, time a single dense conversion (too expensive to do all)
    dense_conversion_start = time.time()
    dense_sample = tensor.to_dense(0, 0)
    dense_conversion_time = time.time() - dense_conversion_start

    logger.info(
        f"Dense conversion for one sub-tensor: {dense_conversion_time * 1000:.2f}ms"
    )

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
        (0, 0, 5, 10, cp.ones((8, 8)) * 1.5),
        (1, 1, 15, 20, cp.ones((8, 8)) * 2.5),
        (3, 0, 25, 5, cp.ones((8, 8)) * 3.5),
        (7, 1, 10, 35, cp.ones((8, 8)) * 4.5),
    ]

    for batch_idx, channel_idx, row, col, values in test_data:
        original.set_block(batch_idx, channel_idx, row, col, values)

    # Export to dictionary
    data_dict = original.to_dict()
    logger.info(f"Exported {len(data_dict)} arrays to dictionary")

    # Verify dictionary contains expected keys
    expected_keys = {
        "sparse_values",
        "block_positions",
        "blocks_set",
        "batch_size",
        "channels",
        "height",
        "width",
        "block_size",
        "device",
    }
    assert (
        set(data_dict.keys()) == expected_keys
    ), f"Missing keys: {expected_keys - set(data_dict.keys())}"

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

    # Verify block data matches
    assert cp.allclose(loaded.sparse_values, original.sparse_values)
    assert cp.array_equal(loaded.block_positions, original.block_positions)
    assert cp.array_equal(loaded.blocks_set, original.blocks_set)

    # Verify same computational results
    target = cp.random.rand(40, 50).astype(cp.float32)
    original_result = original.transpose_dot_product(target)
    loaded_result = loaded.transpose_dot_product(target)

    assert cp.allclose(
        original_result, loaded_result
    ), "Loaded tensor produces different results"

    # Test edge case: empty tensor
    empty_tensor = SingleBlockMixedSparseTensor(3, 2, 20, 30, 4)
    empty_dict = empty_tensor.to_dict()
    empty_loaded = SingleBlockMixedSparseTensor.from_dict(empty_dict)

    assert empty_loaded.batch_size == 3
    assert empty_loaded.channels == 2
    assert cp.sum(empty_loaded.blocks_set) == 0  # No blocks set

    logger.info("✓ Save/load functionality test passed")


def test_error_handling():
    """Test error handling and edge cases."""
    logger.info("=== Testing Error Handling ===")

    tensor = SingleBlockMixedSparseTensor(5, 2, 50, 60, 8)

    # Test out-of-bounds indices
    try:
        tensor.set_block(10, 0, 0, 0, cp.ones((8, 8)))
        assert False, "Should have raised ValueError for batch_idx out of bounds"
    except ValueError:
        pass

    try:
        tensor.set_block(0, 5, 0, 0, cp.ones((8, 8)))
        assert False, "Should have raised ValueError for channel_idx out of bounds"
    except ValueError:
        pass

    # Test out-of-bounds positions
    try:
        tensor.set_block(0, 0, 50, 0, cp.ones((8, 8)))  # row too large
        assert False, "Should have raised ValueError for row out of bounds"
    except ValueError:
        pass

    try:
        tensor.set_block(0, 0, 0, 60, cp.ones((8, 8)))  # col too large
        assert False, "Should have raised ValueError for col out of bounds"
    except ValueError:
        pass

    # Test wrong value shape
    try:
        tensor.set_block(0, 0, 0, 0, cp.ones((10, 10)))  # wrong block size
        assert False, "Should have raised ValueError for wrong value shape"
    except ValueError:
        pass

    logger.info("✓ Error handling test passed")


def main():
    """Run all tests."""
    logger.info("Starting SingleBlockMixedSparseTensor tests...")

    try:
        test_basic_functionality()
        test_transpose_dot_product()
        test_batch_operations()
        test_memory_efficiency()
        test_performance_comparison()
        test_save_load()
        test_error_handling()

        logger.info("🎉 All tests passed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
