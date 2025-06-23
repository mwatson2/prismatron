#!/usr/bin/env python3
"""
Example demonstrating SingleBlockMixedSparseTensor save/load with npz files.

This shows how to integrate the custom sparse tensor with the existing npz
pattern storage format used by the Prismatron system.
"""

import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_example_tensor():
    """Create an example sparse tensor with some LED patterns."""
    logger.info("Creating example sparse tensor...")

    # Create tensor for LED display: 100 LEDs, RGB channels, 160x120 images, 16x16 blocks
    tensor = SingleBlockMixedSparseTensor(
        batch_size=100,  # 100 LEDs
        channels=3,  # RGB
        height=160,  # Frame height
        width=120,  # Frame width
        block_size=16,  # 16x16 blocks
    )

    # Generate some example LED diffusion patterns
    for led_id in range(100):
        for channel in range(3):
            # Create a Gaussian-like pattern for each LED
            center_row = np.random.randint(0, 160 - 16)
            center_col = np.random.randint(0, 120 - 16)

            # Create a pattern that decreases from center
            y, x = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
            center_y, center_x = 8, 8

            # Gaussian-like decay
            pattern = np.exp(
                -((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * 3**2)
            )
            pattern = pattern.astype(np.float32)

            # Make it channel-specific
            if channel == 0:  # Red
                pattern *= 1.2
            elif channel == 1:  # Green
                pattern *= 1.0
            else:  # Blue
                pattern *= 0.8

            # Set the block
            tensor.set_block(
                led_id, channel, center_row, center_col, cp.asarray(pattern)
            )

    logger.info(f"Created tensor with {cp.sum(tensor.blocks_set)} blocks")
    return tensor


def save_to_npz_with_metadata(tensor, filename):
    """Save tensor to npz file along with additional metadata."""
    logger.info(f"Saving tensor to {filename}...")

    # Get tensor data as dictionary
    tensor_data = tensor.to_dict()

    # Add some additional metadata that might be useful
    additional_data = {
        "metadata": np.array(
            {
                "created_by": "SingleBlockMixedSparseTensor",
                "version": "1.0",
                "description": "LED diffusion patterns in single block sparse format",
                "led_count": tensor.batch_size,
                "total_blocks": int(cp.sum(tensor.blocks_set)),
            }
        ),
        "led_positions": np.random.rand(tensor.batch_size, 2).astype(
            np.float32
        ),  # Example LED positions
        "system_info": np.array(
            {"frame_rate": 30.0, "optimization_target": "real_time"}
        ),
    }

    # Combine tensor data with additional metadata
    save_data = {**tensor_data, **additional_data}

    # Save to npz file
    np.savez_compressed(filename, **save_data)

    logger.info(f"Saved {len(save_data)} arrays to {filename}")

    # Show what was saved
    file_size_mb = Path(filename).stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f}MB")


def load_from_npz_and_test(filename):
    """Load tensor from npz file and test functionality."""
    logger.info(f"Loading tensor from {filename}...")

    # Load npz file
    data = np.load(filename, allow_pickle=True)

    logger.info(f"Loaded npz with {len(data.files)} arrays")
    logger.info(f"Arrays: {data.files}")

    # Extract just the tensor data (filter out non-tensor keys)
    tensor_keys = {
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

    tensor_data = {key: data[key] for key in tensor_keys if key in data.files}

    # Load tensor from the data
    loaded_tensor = SingleBlockMixedSparseTensor.from_dict(tensor_data)

    logger.info(f"Loaded tensor: {loaded_tensor}")

    # Test the loaded tensor with a computation
    target_image = cp.random.rand(160, 120).astype(cp.float32)
    result = loaded_tensor.transpose_dot_product(target_image)

    logger.info(f"Test computation result shape: {result.shape}")
    logger.info(f"Result range: [{result.min():.3f}, {result.max():.3f}]")

    # Access additional metadata
    if "metadata" in data.files:
        metadata = data["metadata"].item()
        logger.info(f"Metadata: {metadata}")

    if "led_positions" in data.files:
        led_positions = data["led_positions"]
        logger.info(f"LED positions shape: {led_positions.shape}")

    data.close()
    return loaded_tensor


def compare_memory_usage():
    """Compare memory usage of different storage approaches."""
    logger.info("Comparing memory usage...")

    # Create a realistic sized tensor
    tensor = SingleBlockMixedSparseTensor(1000, 3, 480, 800, 64)

    # Set 50% of blocks
    num_blocks = 1500  # 50% of 3000 total blocks
    for i in range(num_blocks):
        led_id = i % 1000
        channel = i % 3
        row = np.random.randint(0, 480 - 64)
        col = np.random.randint(0, 800 - 64)
        pattern = cp.random.rand(64, 64).astype(cp.float32)
        tensor.set_block(led_id, channel, row, col, pattern)

    # Get memory info
    memory_info = tensor.memory_info()

    logger.info("Memory usage comparison:")
    logger.info(f"  SingleBlockMixedSparseTensor: {memory_info['total_mb']:.1f}MB")
    logger.info(
        f"  Equivalent dense storage: {memory_info['equivalent_dense_mb']:.1f}MB"
    )
    logger.info(f"  Compression ratio: {memory_info['compression_ratio']:.1%}")
    logger.info(f"  Blocks stored: {memory_info['blocks_stored']}")

    # Save and check file size
    tensor_data = tensor.to_dict()
    np.savez_compressed("temp_tensor.npz", **tensor_data)

    file_size_mb = Path("temp_tensor.npz").stat().st_size / (1024 * 1024)
    logger.info(f"  Compressed npz file: {file_size_mb:.1f}MB")

    # Clean up
    Path("temp_tensor.npz").unlink()


def main():
    """Run the example."""
    logger.info("SingleBlockMixedSparseTensor npz integration example")

    # Create and save tensor
    tensor = create_example_tensor()
    save_to_npz_with_metadata(tensor, "example_led_patterns.npz")

    # Load and test
    loaded_tensor = load_from_npz_and_test("example_led_patterns.npz")

    # Verify they produce same results
    target = cp.random.rand(160, 120).astype(cp.float32)
    original_result = tensor.transpose_dot_product(target)
    loaded_result = loaded_tensor.transpose_dot_product(target)

    if cp.allclose(original_result, loaded_result):
        logger.info("✓ Original and loaded tensors produce identical results")
    else:
        logger.error("✗ Results differ between original and loaded tensors")

    # Memory comparison
    compare_memory_usage()

    # Clean up
    Path("example_led_patterns.npz").unlink()

    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
