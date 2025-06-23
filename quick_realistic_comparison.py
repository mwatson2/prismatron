#!/usr/bin/env python3
"""
Quick realistic performance comparison using the actual CSC matrix approach
vs a practical SingleBlockMixedSparseTensor with smaller but realistic blocks.
"""

import logging
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_real_csc_matrix():
    """Load the actual CSC matrix used in production."""
    logger.info("Loading real CSC matrix...")

    patterns_path = "diffusion_patterns/synthetic_1000.npz"
    data = np.load(patterns_path)

    # Load sparse matrix
    matrix_data = data["matrix_data"]
    matrix_indices = data["matrix_indices"]
    matrix_indptr = data["matrix_indptr"]
    matrix_shape = tuple(data["matrix_shape"])

    # Reconstruct sparse matrix
    matrix = sp.csc_matrix(
        (matrix_data, matrix_indices, matrix_indptr),
        shape=matrix_shape,
        dtype=np.float32,
    )

    logger.info(f"Real CSC matrix: {matrix.shape}")
    logger.info(f"Non-zeros: {matrix.nnz:,}")
    logger.info(
        f"Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.3f}%"
    )

    # Transfer to GPU
    from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix

    matrix_gpu = cupy_csc_matrix(matrix)

    return matrix_gpu


def create_realistic_tensor_patterns():
    """Create SingleBlockMixedSparseTensor with realistic but smaller patterns."""
    logger.info("Creating realistic tensor patterns with 96x96 blocks...")

    # Use 96x96 blocks (more practical for 480x800 frames)
    batch_size = 1000
    channels = 3
    height = 480
    width = 800
    block_size = 96

    tensor = SingleBlockMixedSparseTensor(
        batch_size, channels, height, width, block_size
    )

    # Generate patterns with realistic density to match 3.6% overall density
    # Each block is 96x96 = 9216 pixels
    # Frame is 480x800 = 384000 pixels
    # If we have 1000 LEDs * 3 channels = 3000 blocks total
    # And want 3.6% overall density: 0.036 * 384000 = 13824 total non-zeros per channel
    # So per block we want: 13824 / 1000 ≈ 14 non-zeros per block
    # But to match the pattern analysis (13981 pixels per pattern), we need more

    # Target: ~3600 non-zeros per block to match real pattern sizes
    # This gives density of 3600/9216 ≈ 39% within each block

    np.random.seed(42)

    for led_id in range(batch_size):
        for channel in range(channels):
            # Random position that fits in frame
            max_row = height - block_size
            max_col = width - block_size

            top_row = np.random.randint(0, max_row + 1)
            top_col = np.random.randint(0, max_col + 1)

            # Create realistic Gaussian-like pattern
            y, x = np.meshgrid(
                np.arange(block_size), np.arange(block_size), indexing="ij"
            )
            center_y, center_x = block_size // 2, block_size // 2

            # Multiple Gaussian centers for realistic diffusion
            pattern = np.zeros((block_size, block_size), dtype=np.float32)

            # Primary center
            sigma1 = np.random.uniform(15.0, 25.0)
            intensity1 = np.random.uniform(0.8, 1.0)
            pattern += intensity1 * np.exp(
                -((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * sigma1**2)
            )

            # Secondary centers for realistic spread
            for _ in range(2):
                offset_y = np.random.randint(-block_size // 3, block_size // 3)
                offset_x = np.random.randint(-block_size // 3, block_size // 3)
                sec_center_y = center_y + offset_y
                sec_center_x = center_x + offset_x

                sigma2 = np.random.uniform(8.0, 15.0)
                intensity2 = np.random.uniform(0.3, 0.6)
                pattern += intensity2 * np.exp(
                    -((y - sec_center_y) ** 2 + (x - sec_center_x) ** 2)
                    / (2 * sigma2**2)
                )

            # Add some noise and threshold to get realistic sparsity
            noise = np.random.normal(0, 0.05, pattern.shape)
            pattern = np.clip(pattern + noise, 0, 1)

            # Threshold to get desired density
            threshold = np.percentile(pattern, 60)  # Keep top 40% of values
            pattern[pattern < threshold] = 0

            tensor.set_block(led_id, channel, top_row, top_col, cp.asarray(pattern))

    # Check actual density achieved
    memory_info = tensor.memory_info()
    logger.info(f"Created tensor with {memory_info['blocks_stored']} blocks")
    logger.info(f"Tensor memory: {memory_info['total_mb']:.1f}MB")

    return tensor


def benchmark_realistic_comparison():
    """Run realistic per-frame performance comparison."""
    logger.info("=" * 60)
    logger.info("REALISTIC PER-FRAME BENCHMARK")
    logger.info("Measuring CSC A^T@b vs SingleBlock transpose_dot_product")
    logger.info("Including CPU->GPU transfer costs")
    logger.info("=" * 60)

    # Load production CSC matrix
    csc_matrix = load_real_csc_matrix()

    # Create realistic tensor
    tensor = create_realistic_tensor_patterns()

    # Test parameters
    num_runs = 11
    frame_height = 480
    frame_width = 800

    # Create test images (realistic per-frame scenario)
    logger.info("Creating test target images...")
    targets_uint8 = []
    for i in range(num_runs):
        target = np.random.randint(
            0, 256, (frame_height, frame_width, 3), dtype=np.uint8
        )
        targets_uint8.append(target)

    # Benchmark CSC approach (production path)
    logger.info("\n=== CSC Matrix Approach (Production) ===")

    def csc_per_frame(target_image):
        """CSC per-frame operation matching production code."""
        # Phase 1: CPU preprocessing and GPU transfer
        transfer_start = time.time()
        target_normalized = target_image.astype(np.float32) / 255.0

        # For fair comparison, just use single channel like tensor approach
        # Production code would process all 3 channels, but for this comparison
        # we'll measure the core A^T @ b operation cost
        target_single_channel = target_normalized[:, :, 0].ravel()  # Shape: (384000,)

        # Transfer to GPU
        target_gpu = cp.asarray(target_single_channel)
        transfer_time = time.time() - transfer_start

        # Phase 2: A^T @ b operation
        cp.cuda.Device().synchronize()
        operation_start = time.time()

        # Use first 1000 columns of matrix (single channel equivalent)
        matrix_single = csc_matrix[:, :1000]  # Shape: (384000, 1000)
        result = matrix_single.T @ target_gpu  # Shape: (1000,)

        cp.cuda.Device().synchronize()
        operation_time = time.time() - operation_start

        total_time = transfer_time + operation_time
        return result, total_time, operation_time

    # Warm up CSC
    _, _, _ = csc_per_frame(targets_uint8[0])

    # Time CSC runs
    csc_total_times = []
    csc_operation_times = []

    for i in range(1, num_runs):
        result, total_time, op_time = csc_per_frame(targets_uint8[i])
        csc_total_times.append(total_time)
        csc_operation_times.append(op_time)

    csc_avg_total = np.mean(csc_total_times)
    csc_avg_operation = np.mean(csc_operation_times)

    logger.info(
        f"CSC per-frame: {csc_avg_total*1000:.2f}ms (operation: {csc_avg_operation*1000:.2f}ms)"
    )

    # Benchmark Tensor approach
    logger.info("\n=== SingleBlockMixedSparseTensor Approach ===")

    def tensor_per_frame(target_image):
        """Tensor per-frame operation."""
        # Phase 1: CPU->GPU transfer
        transfer_start = time.time()
        target_gpu = cp.asarray(target_image.astype(np.float32) / 255.0)
        # Use single channel for simplicity in this test
        target_single = target_gpu[:, :, 0]
        transfer_time = time.time() - transfer_start

        # Phase 2: Block extraction and computation
        cp.cuda.Device().synchronize()
        operation_start = time.time()

        result = tensor.transpose_dot_product(target_single)

        cp.cuda.Device().synchronize()
        operation_time = time.time() - operation_start

        total_time = transfer_time + operation_time
        return result, total_time, operation_time

    # Warm up tensor
    _, _, _ = tensor_per_frame(targets_uint8[0])

    # Time tensor runs
    tensor_total_times = []
    tensor_operation_times = []

    for i in range(1, num_runs):
        result, total_time, op_time = tensor_per_frame(targets_uint8[i])
        tensor_total_times.append(total_time)
        tensor_operation_times.append(op_time)

    tensor_avg_total = np.mean(tensor_total_times)
    tensor_avg_operation = np.mean(tensor_operation_times)

    logger.info(
        f"Tensor per-frame: {tensor_avg_total*1000:.2f}ms (operation: {tensor_avg_operation*1000:.2f}ms)"
    )

    # Performance analysis
    logger.info("\n=== PERFORMANCE COMPARISON ===")

    total_speedup = csc_avg_total / tensor_avg_total
    operation_speedup = csc_avg_operation / tensor_avg_operation

    logger.info(f"Total per-frame speedup: {total_speedup:.2f}x")
    logger.info(f"Core operation speedup: {operation_speedup:.2f}x")

    if total_speedup > 1.1:
        logger.info("✓ Tensor approach is significantly faster")
    elif total_speedup < 0.9:
        logger.info("✓ CSC approach maintains advantage")
    else:
        logger.info("≈ Performance is comparable")

    # Memory comparison
    csc_memory = csc_matrix.data.nbytes / (1024 * 1024)
    tensor_memory = tensor.memory_info()["total_mb"]

    logger.info(f"\nMemory usage:")
    logger.info(f"  CSC matrix: {csc_memory:.1f}MB")
    logger.info(f"  Tensor:     {tensor_memory:.1f}MB")
    logger.info(f"  Ratio:      {tensor_memory/csc_memory:.2f}x")

    # Throughput analysis
    fps_csc = 1.0 / csc_avg_total
    fps_tensor = 1.0 / tensor_avg_total

    logger.info(f"\nMax throughput:")
    logger.info(f"  CSC:    {fps_csc:.1f} FPS")
    logger.info(f"  Tensor: {fps_tensor:.1f} FPS")


def main():
    """Run realistic comparison."""
    logger.info("Realistic Performance Comparison")
    logger.info("CSC vs SingleBlockMixedSparseTensor")
    logger.info("Per-frame costs at production scale")

    benchmark_realistic_comparison()

    return 0


if __name__ == "__main__":
    main()
