#!/usr/bin/env python3
"""
Performance comparison: CSC vs Chunked vs CUDA kernel approaches.

This test compares three approaches for LED optimization A^T @ b operations:
1. CSC sparse matrices (production approach)
2. SingleBlockMixedSparseTensor with chunking (research baseline)
3. SingleBlockMixedSparseTensor with custom CUDA kernel (optimized research)

Uses 96x96 blocks as a practical compromise for real deployment.
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


def load_production_csc_matrix():
    """Load the production CSC matrix for comparison."""
    logger.info("Loading production CSC matrix...")

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

    logger.info(f"CSC matrix: {matrix.shape}")
    logger.info(f"Non-zeros: {matrix.nnz:,}")
    logger.info(
        f"Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.3f}%"
    )

    # Transfer to GPU
    from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix

    matrix_gpu = cupy_csc_matrix(matrix)
    memory_mb = matrix_gpu.data.nbytes / (1024 * 1024)
    logger.info(f"CSC GPU memory: {memory_mb:.1f}MB")

    return matrix_gpu


def create_realistic_96x96_tensor():
    """Create SingleBlockMixedSparseTensor with realistic 96x96 patterns."""
    logger.info("Creating realistic tensor with 96x96 blocks...")

    # Use 96x96 blocks (practical for real deployment)
    batch_size = 1000
    channels = 3
    height = 480
    width = 800
    block_size = 96

    tensor = SingleBlockMixedSparseTensor(
        batch_size, channels, height, width, block_size
    )

    # Generate realistic diffusion patterns
    np.random.seed(42)  # For reproducibility

    for led_id in range(batch_size):
        for channel in range(channels):
            # Random position that fits in frame
            max_row = height - block_size
            max_col = width - block_size

            top_row = np.random.randint(0, max_row + 1)
            top_col = np.random.randint(0, max_col + 1)

            # Create realistic Gaussian-like diffusion pattern
            y, x = np.meshgrid(
                np.arange(block_size), np.arange(block_size), indexing="ij"
            )
            center_y, center_x = block_size // 2, block_size // 2

            # Primary Gaussian
            sigma1 = np.random.uniform(15.0, 25.0)
            intensity1 = np.random.uniform(0.8, 1.0)
            pattern = intensity1 * np.exp(
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

            # Add noise and threshold for realistic sparsity
            noise = np.random.normal(0, 0.05, pattern.shape)
            pattern = np.clip(pattern + noise, 0, 1)

            # Keep top 40% of values for realistic density
            threshold = np.percentile(pattern, 60)
            pattern[pattern < threshold] = 0

            tensor.set_block(led_id, channel, top_row, top_col, cp.asarray(pattern))

    # Report statistics
    memory_info = tensor.memory_info()
    logger.info(f"Created tensor with {memory_info['blocks_stored']} blocks")
    logger.info(f"Tensor memory: {memory_info['total_mb']:.1f}MB")

    return tensor


def benchmark_three_approaches(csc_matrix, tensor, num_runs=11):
    """Benchmark all three approaches with per-frame measurements."""
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE PERFORMANCE BENCHMARK")
    logger.info("CSC vs Chunked vs CUDA Kernel approaches")
    logger.info("=" * 70)

    frame_height = 480
    frame_width = 800

    # Create test images (realistic per-frame scenario)
    logger.info(f"Creating {num_runs} test target images...")
    targets_uint8 = []
    for i in range(num_runs):
        target = np.random.randint(
            0, 256, (frame_height, frame_width, 3), dtype=np.uint8
        )
        targets_uint8.append(target)

    # Approach 1: CSC Matrix (Production)
    logger.info("\n=== Approach 1: CSC Matrix (Production) ===")

    def csc_per_frame(target_image):
        """CSC per-frame operation."""
        # CPU preprocessing and GPU transfer
        transfer_start = time.time()
        target_normalized = target_image.astype(np.float32) / 255.0
        target_single_channel = target_normalized[:, :, 0].ravel()
        target_gpu = cp.asarray(target_single_channel)
        transfer_time = time.time() - transfer_start

        # A^T @ b operation
        cp.cuda.Device().synchronize()
        operation_start = time.time()

        # Use first 1000 columns (single channel equivalent)
        matrix_single = csc_matrix[:, :1000]
        result = matrix_single.T @ target_gpu

        cp.cuda.Device().synchronize()
        operation_time = time.time() - operation_start

        return result, transfer_time + operation_time, operation_time

    # Warm up and benchmark CSC
    _, _, _ = csc_per_frame(targets_uint8[0])

    csc_total_times = []
    csc_operation_times = []

    for i in range(1, num_runs):
        result, total_time, op_time = csc_per_frame(targets_uint8[i])
        csc_total_times.append(total_time)
        csc_operation_times.append(op_time)

    csc_avg_total = np.mean(csc_total_times)
    csc_avg_operation = np.mean(csc_operation_times)
    csc_std = np.std(csc_total_times)

    logger.info(f"CSC per-frame: {csc_avg_total*1000:.2f}ms ± {csc_std*1000:.2f}ms")
    logger.info(f"  Operation only: {csc_avg_operation*1000:.2f}ms")

    # Approach 2: Chunked Tensor (Research Baseline)
    logger.info("\n=== Approach 2: Chunked Tensor (Research Baseline) ===")

    def chunked_per_frame(target_image):
        """Chunked tensor per-frame operation."""
        # CPU->GPU transfer
        transfer_start = time.time()
        target_gpu = cp.asarray(target_image.astype(np.float32) / 255.0)
        target_single = target_gpu[:, :, 0]
        transfer_time = time.time() - transfer_start

        # Block extraction and computation (chunked)
        cp.cuda.Device().synchronize()
        operation_start = time.time()

        result = tensor.transpose_dot_product(target_single)

        cp.cuda.Device().synchronize()
        operation_time = time.time() - operation_start

        return result, transfer_time + operation_time, operation_time

    # Warm up and benchmark chunked
    _, _, _ = chunked_per_frame(targets_uint8[0])

    chunked_total_times = []
    chunked_operation_times = []

    for i in range(1, num_runs):
        result, total_time, op_time = chunked_per_frame(targets_uint8[i])
        chunked_total_times.append(total_time)
        chunked_operation_times.append(op_time)

    chunked_avg_total = np.mean(chunked_total_times)
    chunked_avg_operation = np.mean(chunked_operation_times)
    chunked_std = np.std(chunked_total_times)

    logger.info(
        f"Chunked per-frame: {chunked_avg_total*1000:.2f}ms ± {chunked_std*1000:.2f}ms"
    )
    logger.info(f"  Operation only: {chunked_avg_operation*1000:.2f}ms")

    # Approach 3: CUDA Kernel (Optimized Research)
    logger.info("\n=== Approach 3: CUDA Kernel (Optimized Research) ===")

    def cuda_per_frame(target_image):
        """CUDA kernel per-frame operation."""
        # CPU->GPU transfer
        transfer_start = time.time()
        target_gpu = cp.asarray(target_image.astype(np.float32) / 255.0)
        target_single = target_gpu[:, :, 0]
        transfer_time = time.time() - transfer_start

        # CUDA kernel computation
        cp.cuda.Device().synchronize()
        operation_start = time.time()

        result = tensor.transpose_dot_product_cuda(target_single)

        cp.cuda.Device().synchronize()
        operation_time = time.time() - operation_start

        return result, transfer_time + operation_time, operation_time

    # Warm up and benchmark CUDA kernel
    try:
        _, _, _ = cuda_per_frame(targets_uint8[0])

        cuda_total_times = []
        cuda_operation_times = []

        for i in range(1, num_runs):
            result, total_time, op_time = cuda_per_frame(targets_uint8[i])
            cuda_total_times.append(total_time)
            cuda_operation_times.append(op_time)

        cuda_avg_total = np.mean(cuda_total_times)
        cuda_avg_operation = np.mean(cuda_operation_times)
        cuda_std = np.std(cuda_total_times)

        logger.info(
            f"CUDA per-frame: {cuda_avg_total*1000:.2f}ms ± {cuda_std*1000:.2f}ms"
        )
        logger.info(f"  Operation only: {cuda_avg_operation*1000:.2f}ms")

        cuda_available = True

    except Exception as e:
        logger.warning(f"CUDA kernel failed: {e}")
        cuda_avg_total = float("inf")
        cuda_avg_operation = float("inf")
        cuda_std = 0.0
        cuda_available = False

    # Performance Analysis
    logger.info("\n" + "=" * 70)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 70)

    # Calculate speedups relative to CSC (baseline)
    chunked_speedup = csc_avg_total / chunked_avg_total
    if cuda_available:
        cuda_speedup = csc_avg_total / cuda_avg_total
        operation_speedup = chunked_avg_operation / cuda_avg_operation

    logger.info(f"CSC Matrix (baseline):     {csc_avg_total*1000:.2f}ms")
    logger.info(
        f"Chunked Tensor:            {chunked_avg_total*1000:.2f}ms ({chunked_speedup:.2f}x)"
    )

    if cuda_available:
        logger.info(
            f"CUDA Kernel:               {cuda_avg_total*1000:.2f}ms ({cuda_speedup:.2f}x)"
        )
        logger.info(f"CUDA vs Chunked speedup:   {operation_speedup:.2f}x")

    # Memory comparison
    csc_memory = csc_matrix.data.nbytes / (1024 * 1024)
    tensor_memory = tensor.memory_info()["total_mb"]

    logger.info(f"\nMemory Usage:")
    logger.info(f"  CSC matrix: {csc_memory:.1f}MB")
    logger.info(
        f"  Tensor:     {tensor_memory:.1f}MB ({tensor_memory/csc_memory:.2f}x)"
    )

    # Throughput analysis
    fps_csc = 1.0 / csc_avg_total
    fps_chunked = 1.0 / chunked_avg_total

    logger.info(f"\nMax Throughput:")
    logger.info(f"  CSC:     {fps_csc:.1f} FPS")
    logger.info(f"  Chunked: {fps_chunked:.1f} FPS")

    if cuda_available:
        fps_cuda = 1.0 / cuda_avg_total
        logger.info(f"  CUDA:    {fps_cuda:.1f} FPS")

    # Final recommendations
    logger.info(f"\n" + "=" * 70)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 70)

    if cuda_available and cuda_speedup > 1.5:
        logger.info("✓ CUDA kernel shows significant performance advantage!")
        logger.info("  Recommend using CUDA kernel for production deployment.")
    elif cuda_available and cuda_speedup > 1.1:
        logger.info("✓ CUDA kernel shows moderate performance improvement.")
    elif chunked_speedup > 1.1:
        logger.info("✓ Chunked tensor approach is competitive with CSC.")
    else:
        logger.info("✓ CSC matrices remain the performance leader.")
        logger.info("  Custom tensor approach valuable for memory efficiency.")


def main():
    """Run the comprehensive three-way performance comparison."""
    logger.info("CUDA Kernel Performance Test")
    logger.info("Comparing CSC vs Chunked vs CUDA kernel approaches")
    logger.info("Using 96x96 blocks for practical deployment")

    # Load production CSC matrix
    csc_matrix = load_production_csc_matrix()

    # Create realistic tensor with 96x96 blocks
    tensor = create_realistic_96x96_tensor()

    # Run comprehensive benchmark
    benchmark_three_approaches(csc_matrix, tensor, num_runs=11)

    return 0


if __name__ == "__main__":
    main()
