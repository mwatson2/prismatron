#!/usr/bin/env python3
"""
Performance comparison test: SingleBlockMixedSparseTensor vs CSC matrices.

This test provides a like-for-like performance comparison between the custom
SingleBlockMixedSparseTensor and traditional CSC sparse matrices for LED
optimization A^T @ b operations.

Test configuration:
- Shape: (1000, 3, 800, 480) - 1000 LEDs, RGB channels, 800x480 frames
- Block size: 64x64 dense blocks per LED/channel
- Multiple runs with cache warming exclusion
- GFLOPS calculations for both approaches
"""

import logging
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CSCMatrixWrapper:
    """Wrapper for CSC matrix approach to match SingleBlockMixedSparseTensor API."""

    def __init__(self, batch_size: int, channels: int, height: int, width: int):
        """Initialize CSC matrix wrapper."""
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.pixels = height * width

        # Combined matrix: [A_r 0 0; 0 A_g 0; 0 0 A_b] layout
        # Shape: (pixels * channels, leds * channels)
        self.matrix_shape = (self.pixels * channels, batch_size * channels)
        self.A_combined = None
        self.A_combined_gpu = None

        logger.info(f"CSC matrix shape: {self.matrix_shape}")

    def set_blocks_from_tensor(self, single_block_tensor: SingleBlockMixedSparseTensor):
        """Create CSC matrix from SingleBlockMixedSparseTensor for fair comparison."""
        logger.info("Converting SingleBlockMixedSparseTensor to CSC format...")

        # Create the combined sparse matrix
        rows = []
        cols = []
        data = []

        # Convert each block to CSC entries
        for batch_idx in range(self.batch_size):
            for channel_idx in range(self.channels):
                if not single_block_tensor.blocks_set[batch_idx, channel_idx]:
                    continue

                # Get block data and position
                block_values = cp.asnumpy(single_block_tensor.sparse_values[batch_idx, channel_idx])
                top_row = int(single_block_tensor.block_positions[batch_idx, channel_idx, 0])
                top_col = int(single_block_tensor.block_positions[batch_idx, channel_idx, 1])

                # Convert block to sparse entries
                block_size = single_block_tensor.block_size
                for i in range(block_size):
                    for j in range(block_size):
                        value = block_values[i, j]
                        if abs(value) > 1e-10:  # Only store non-zero values
                            # Global pixel position
                            pixel_row = top_row + i
                            pixel_col = top_col + j
                            pixel_idx = pixel_row * self.width + pixel_col

                            # CSC matrix position (channel-specific offset)
                            matrix_row = channel_idx * self.pixels + pixel_idx
                            matrix_col = channel_idx * self.batch_size + batch_idx

                            rows.append(matrix_row)
                            cols.append(matrix_col)
                            data.append(value)

        # Create CSC matrix
        self.A_combined = sp.csc_matrix((data, (rows, cols)), shape=self.matrix_shape, dtype=np.float32)

        logger.info(f"Created CSC matrix with {self.A_combined.nnz:,} non-zero elements")
        logger.info(f"Matrix density: {self.A_combined.nnz / (self.matrix_shape[0] * self.matrix_shape[1]) * 100:.3f}%")

        # Transfer to GPU
        from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix

        self.A_combined_gpu = cupy_csc_matrix(self.A_combined)

        memory_mb = self.A_combined_gpu.data.nbytes / (1024 * 1024)
        logger.info(f"CSC GPU memory: {memory_mb:.1f}MB")

    def transpose_dot_product(self, target_image: cp.ndarray) -> cp.ndarray:
        """Compute A^T @ b using CSC matrix."""
        if self.A_combined_gpu is None:
            raise RuntimeError("CSC matrix not initialized")

        # Convert target image to combined vector format [R_pixels; G_pixels; B_pixels]
        target_combined = cp.zeros(self.pixels * self.channels, dtype=cp.float32)
        target_flat = target_image.ravel()

        # CRITICAL: Each channel should get the same target image, not different slices
        for c in range(self.channels):
            start_idx = c * self.pixels
            end_idx = (c + 1) * self.pixels
            target_combined[start_idx:end_idx] = target_flat  # Same target for all channels

        # Compute A^T @ b
        result_combined = self.A_combined_gpu.T @ target_combined

        # Convert back to (batch_size, channels) format
        result = cp.zeros((self.batch_size, self.channels), dtype=cp.float32)
        for c in range(self.channels):
            start_idx = c * self.batch_size
            end_idx = (c + 1) * self.batch_size
            result[:, c] = result_combined[start_idx:end_idx]

        return result


def create_test_tensor():
    """Create test tensor with realistic LED diffusion patterns."""
    logger.info("Creating test tensor with 1000 LEDs, 3 channels, 800x480 frames...")

    batch_size = 1000
    channels = 3
    height = 480
    width = 800
    block_size = 64

    tensor = SingleBlockMixedSparseTensor(batch_size, channels, height, width, block_size)

    # Set blocks for all LEDs with realistic patterns
    logger.info("Generating realistic LED diffusion patterns...")
    np.random.seed(42)  # For reproducible results

    for led_id in range(batch_size):
        for channel in range(channels):
            # Random position for LED block
            top_row = np.random.randint(0, height - block_size)
            top_col = np.random.randint(0, width - block_size)

            # Create Gaussian-like diffusion pattern
            y, x = np.meshgrid(np.arange(block_size), np.arange(block_size), indexing="ij")
            center_y, center_x = block_size // 2, block_size // 2

            # Gaussian decay with some randomness
            sigma = np.random.uniform(8.0, 16.0)  # Variable diffusion width
            intensity = np.random.uniform(0.3, 1.0)  # Variable LED intensity

            pattern = intensity * np.exp(-((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * sigma**2))

            # Add some noise for realism
            noise = np.random.normal(0, 0.02, pattern.shape)
            pattern = np.clip(pattern + noise, 0, 1).astype(np.float32)

            tensor.set_block(led_id, channel, top_row, top_col, cp.asarray(pattern))

    blocks_set = int(cp.sum(tensor.blocks_set))
    logger.info(f"Created tensor with {blocks_set} blocks")

    return tensor


def calculate_gflops(operations: int, time_seconds: float) -> float:
    """Calculate GFLOPS (Giga Floating Point Operations Per Second)."""
    return operations / (time_seconds * 1e9)


def estimate_operations(tensor_shape, nnz_per_led_channel, num_targets=1):
    """
    Estimate number of floating point operations for A^T @ b.

    Args:
        tensor_shape: (batch_size, channels, height, width)
        nnz_per_led_channel: Average non-zero elements per LED/channel
        num_targets: Number of target images

    Returns:
        Estimated number of FLOPs
    """
    batch_size, channels, height, width = tensor_shape

    # For sparse matrix multiply: 2 * nnz (one multiply + one add per non-zero)
    total_nnz = batch_size * channels * nnz_per_led_channel
    operations_per_target = 2 * total_nnz

    return operations_per_target * num_targets


def benchmark_single_block_tensor(tensor: SingleBlockMixedSparseTensor, num_runs: int = 10):
    """Benchmark SingleBlockMixedSparseTensor performance."""
    logger.info("=== Benchmarking SingleBlockMixedSparseTensor ===")

    # Create test target images
    targets = [cp.random.rand(tensor.height, tensor.width).astype(cp.float32) for _ in range(num_runs)]

    # Warm up (exclude from timing)
    logger.info("Warming up GPU cache...")
    _ = tensor.transpose_dot_product(targets[0])
    cp.cuda.Device().synchronize()

    # Benchmark runs
    logger.info(f"Running {num_runs - 1} timed iterations...")
    times = []

    for i in range(1, num_runs):  # Skip first run (warm-up)
        cp.cuda.Device().synchronize()
        start_time = time.time()

        result = tensor.transpose_dot_product(targets[i])

        cp.cuda.Device().synchronize()
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    logger.info("SingleBlockMixedSparseTensor results:")
    logger.info(f"  Average time: {avg_time * 1000:.2f}ms ± {std_time * 1000:.2f}ms")
    logger.info(f"  Min time: {min(times) * 1000:.2f}ms")
    logger.info(f"  Max time: {max(times) * 1000:.2f}ms")

    return avg_time, result


def benchmark_csc_matrix(csc_wrapper: CSCMatrixWrapper, num_runs: int = 10):
    """Benchmark CSC matrix performance."""
    logger.info("=== Benchmarking CSC Matrix ===")

    # Create test target images
    targets = [cp.random.rand(csc_wrapper.height, csc_wrapper.width).astype(cp.float32) for _ in range(num_runs)]

    # Warm up (exclude from timing)
    logger.info("Warming up GPU cache...")
    _ = csc_wrapper.transpose_dot_product(targets[0])
    cp.cuda.Device().synchronize()

    # Benchmark runs
    logger.info(f"Running {num_runs - 1} timed iterations...")
    times = []

    for i in range(1, num_runs):  # Skip first run (warm-up)
        cp.cuda.Device().synchronize()
        start_time = time.time()

        result = csc_wrapper.transpose_dot_product(targets[i])

        cp.cuda.Device().synchronize()
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    logger.info("CSC Matrix results:")
    logger.info(f"  Average time: {avg_time * 1000:.2f}ms ± {std_time * 1000:.2f}ms")
    logger.info(f"  Min time: {min(times) * 1000:.2f}ms")
    logger.info(f"  Max time: {max(times) * 1000:.2f}ms")

    return avg_time, result


def verify_correctness(single_block_result: cp.ndarray, csc_result: cp.ndarray, tolerance: float = 1e-4):
    """Verify that both approaches produce the same results."""
    logger.info("=== Verifying Correctness ===")

    max_diff = cp.max(cp.abs(single_block_result - csc_result))
    mean_diff = cp.mean(cp.abs(single_block_result - csc_result))
    relative_diff = max_diff / cp.max(cp.abs(single_block_result))

    logger.info("Result comparison:")
    logger.info(f"  Max absolute difference: {max_diff:.6f}")
    logger.info(f"  Mean absolute difference: {mean_diff:.6f}")
    logger.info(f"  Max relative difference: {relative_diff:.6f}")

    # Debug first few values to see the pattern
    logger.info(f"First 5 SingleBlock results: {single_block_result.ravel()[:5]}")
    logger.info(f"First 5 CSC results: {csc_result.ravel()[:5]}")

    # Check if results have different shapes or ordering
    logger.info(f"SingleBlock result shape: {single_block_result.shape}")
    logger.info(f"CSC result shape: {csc_result.shape}")

    if max_diff < tolerance:
        logger.info("✓ Results match within tolerance")
        return True
    else:
        logger.warning(f"⚠ Results differ by more than tolerance {tolerance}")
        # Let's also check if it's just a reordering issue
        single_sorted = cp.sort(single_block_result.ravel())
        csc_sorted = cp.sort(csc_result.ravel())
        sorted_diff = cp.max(cp.abs(single_sorted - csc_sorted))
        logger.info(f"  Max difference after sorting: {sorted_diff:.6f}")
        if sorted_diff < tolerance:
            logger.warning("  → Results have same values but different order!")
        return False


def main():
    """Run comprehensive performance comparison."""
    logger.info("SingleBlockMixedSparseTensor vs CSC Performance Comparison")
    logger.info("=" * 60)

    # Create test data
    tensor = create_test_tensor()

    # Setup CSC matrix for comparison
    csc_wrapper = CSCMatrixWrapper(tensor.batch_size, tensor.channels, tensor.height, tensor.width)
    csc_wrapper.set_blocks_from_tensor(tensor)

    # Get memory usage information
    tensor_memory = tensor.memory_info()
    csc_memory_mb = csc_wrapper.A_combined_gpu.data.nbytes / (1024 * 1024)

    logger.info("\nMemory Usage:")
    logger.info(f"  SingleBlockMixedSparseTensor: {tensor_memory['total_mb']:.1f}MB")
    logger.info(f"  CSC Matrix: {csc_memory_mb:.1f}MB")
    logger.info(f"  Memory ratio (SingleBlock/CSC): {tensor_memory['total_mb'] / csc_memory_mb:.2f}x")

    # Estimate operations for GFLOPS calculation
    # Average block is 64x64 = 4096 elements, but some may be zero due to Gaussian pattern
    # Estimate ~70% of block elements are non-zero on average
    avg_nnz_per_block = int(0.7 * 64 * 64)
    total_operations = estimate_operations(
        (tensor.batch_size, tensor.channels, tensor.height, tensor.width),
        avg_nnz_per_block,
    )

    logger.info(f"\nEstimated operations per A^T @ b: {total_operations:,} FLOPs")

    # Run benchmarks
    num_runs = 11  # 10 timed runs + 1 warmup

    single_block_time, single_block_result = benchmark_single_block_tensor(tensor, num_runs)
    csc_time, csc_result = benchmark_csc_matrix(csc_wrapper, num_runs)

    # Verify correctness
    results_match = verify_correctness(single_block_result, csc_result)

    # Calculate GFLOPS
    single_block_gflops = calculate_gflops(total_operations, single_block_time)
    csc_gflops = calculate_gflops(total_operations, csc_time)

    # Performance summary
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)

    speedup = csc_time / single_block_time

    logger.info("SingleBlockMixedSparseTensor:")
    logger.info(f"  Time: {single_block_time * 1000:.2f}ms")
    logger.info(f"  GFLOPS: {single_block_gflops:.2f}")
    logger.info(f"  Memory: {tensor_memory['total_mb']:.1f}MB")

    logger.info("\nCSC Matrix:")
    logger.info(f"  Time: {csc_time * 1000:.2f}ms")
    logger.info(f"  GFLOPS: {csc_gflops:.2f}")
    logger.info(f"  Memory: {csc_memory_mb:.1f}MB")

    logger.info("\nPerformance Comparison:")
    logger.info(f"  Speedup: {speedup:.2f}x")
    logger.info(f"  GFLOPS improvement: {single_block_gflops / csc_gflops:.2f}x")
    logger.info(f"  Memory efficiency: {csc_memory_mb / tensor_memory['total_mb']:.2f}x less memory")
    logger.info(f"  Results correctness: {'✓ PASS' if results_match else '✗ FAIL'}")

    # Final recommendations
    logger.info("\nRecommendations:")
    if speedup > 1.5:
        logger.info("✓ SingleBlockMixedSparseTensor shows significant performance advantage")
    elif speedup > 1.1:
        logger.info("✓ SingleBlockMixedSparseTensor shows moderate performance advantage")
    else:
        logger.info("⚠ Performance difference is marginal")

    if tensor_memory["total_mb"] < csc_memory_mb * 0.8:
        logger.info("✓ SingleBlockMixedSparseTensor uses significantly less memory")

    if results_match:
        logger.info("✓ Both approaches produce identical results")
    else:
        logger.warning("⚠ Results differ - investigate numerical precision")


if __name__ == "__main__":
    main()
