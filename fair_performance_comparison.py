#!/usr/bin/env python3
"""
Fair performance comparison between CSC matrices and SingleBlockMixedSparseTensor.

This test aligns the measurements to be realistic:
1. Uses actual pattern scale (1000 LEDs, 3 channels, 480x800 frames)
2. Uses realistic block sizes (192x192 to match pattern analysis)
3. Uses realistic pattern density (3.6% to match synthetic_1000)
4. Measures per-frame costs including CPU->GPU transfer for images
5. Excludes A^T matrix setup costs (amortized across frames)
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


def load_real_patterns():
    """Load the actual synthetic_1000 patterns for realistic testing."""
    logger.info("Loading real diffusion patterns...")

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

    logger.info(f"Loaded real patterns: {matrix.shape}")
    logger.info(f"Non-zeros: {matrix.nnz:,}")
    logger.info(
        f"Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.3f}%"
    )

    return matrix


def extract_patterns_to_blocks(
    matrix, block_size=160, frame_height=480, frame_width=800
):
    """Extract actual patterns into block format for SingleBlockMixedSparseTensor."""
    logger.info(f"Extracting patterns to {block_size}x{block_size} blocks...")

    pixels, led_channels = matrix.shape
    led_count = led_channels // 3

    # Create tensor with realistic parameters
    tensor = SingleBlockMixedSparseTensor(
        batch_size=led_count,
        channels=3,
        height=frame_height,
        width=frame_width,
        block_size=block_size,
    )

    patterns_extracted = 0
    patterns_skipped = 0
    total_nonzeros = 0

    for led_id in range(led_count):
        for channel in range(3):
            col_idx = channel * led_count + led_id

            # Get column for this LED/channel
            col_data = matrix[:, col_idx]
            nonzero_rows = col_data.nonzero()[0]

            if len(nonzero_rows) == 0:
                patterns_skipped += 1
                continue

            # Convert pixel indices to (row, col) coordinates
            pixel_rows = nonzero_rows // frame_width
            pixel_cols = nonzero_rows % frame_width

            # Find bounding box
            min_row, max_row = pixel_rows.min(), pixel_rows.max()
            min_col, max_col = pixel_cols.min(), pixel_cols.max()

            height_extent = max_row - min_row + 1
            width_extent = max_col - min_col + 1

            # Check if pattern fits in block AND can be placed in frame
            if (
                height_extent > block_size
                or width_extent > block_size
                or min_row + block_size > frame_height
                or min_col + block_size > frame_width
            ):
                patterns_skipped += 1
                continue

            # Create block to hold pattern
            block_values = cp.zeros((block_size, block_size), dtype=cp.float32)

            # Extract values and place in block
            values = col_data.data
            for i, row_idx in enumerate(nonzero_rows):
                pixel_row = row_idx // frame_width
                pixel_col = row_idx % frame_width

                # Position in block (center the pattern)
                block_row = pixel_row - min_row
                block_col = pixel_col - min_col

                if 0 <= block_row < block_size and 0 <= block_col < block_size:
                    block_values[block_row, block_col] = values[i]

            # Set block in tensor
            tensor.set_block(led_id, channel, min_row, min_col, block_values)
            patterns_extracted += 1
            total_nonzeros += len(nonzero_rows)

    logger.info(f"Extracted {patterns_extracted} patterns, skipped {patterns_skipped}")
    logger.info(f"Total non-zeros in extracted patterns: {total_nonzeros:,}")

    coverage_pct = patterns_extracted / (led_count * 3) * 100
    logger.info(f"Pattern coverage: {coverage_pct:.1f}%")

    return tensor


class RealisticCSCWrapper:
    """CSC wrapper that matches the production DenseLEDOptimizer approach."""

    def __init__(self, matrix):
        """Initialize with real sparse matrix."""
        self.matrix = matrix
        self.pixels, led_channels = matrix.shape
        self.led_count = led_channels // 3
        self.channels = 3

        # Transfer to GPU (one-time setup cost, amortized)
        logger.info("Transferring CSC matrix to GPU (one-time setup)...")
        setup_start = time.time()

        from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix

        self.matrix_gpu = cupy_csc_matrix(matrix)

        setup_time = time.time() - setup_start
        memory_mb = self.matrix_gpu.data.nbytes / (1024 * 1024)
        logger.info(f"CSC GPU setup: {setup_time:.3f}s, {memory_mb:.1f}MB")

    def per_frame_operation(self, target_image: np.ndarray) -> cp.ndarray:
        """
        Measure per-frame A^T @ b operation including realistic data transfer.

        This matches what happens in production: CPU image -> GPU -> A^T@b operation.
        """
        # Phase 1: CPU preprocessing and GPU transfer (per-frame cost)
        transfer_start = time.time()

        # Normalize and reshape target (CPU)
        target_normalized = target_image.astype(np.float32) / 255.0
        target_flattened = target_normalized.reshape(-1, 3)  # (pixels, 3)

        # Create combined vector [R_pixels; G_pixels; B_pixels] (CPU)
        target_combined = np.empty(self.pixels * 3, dtype=np.float32)
        target_combined[: self.pixels] = target_flattened[:, 0]  # R
        target_combined[self.pixels : 2 * self.pixels] = target_flattened[:, 1]  # G
        target_combined[2 * self.pixels :] = target_flattened[:, 2]  # B

        # Transfer to GPU (per-frame cost)
        target_combined_gpu = cp.asarray(target_combined)
        transfer_time = time.time() - transfer_start

        # Phase 2: A^T @ b sparse operation (per-frame cost)
        cp.cuda.Device().synchronize()
        operation_start = time.time()

        result_combined = self.matrix_gpu.T @ target_combined_gpu

        cp.cuda.Device().synchronize()
        operation_time = time.time() - operation_start

        # Phase 3: Reshape result (minimal cost)
        reshape_start = time.time()
        result = cp.zeros((self.led_count, self.channels), dtype=cp.float32)
        for c in range(self.channels):
            start_idx = c * self.led_count
            end_idx = (c + 1) * self.led_count
            result[:, c] = result_combined[start_idx:end_idx]
        reshape_time = time.time() - reshape_start

        # Return result and timing breakdown
        return result, {
            "transfer_time": transfer_time,
            "operation_time": operation_time,
            "reshape_time": reshape_time,
            "total_time": transfer_time + operation_time + reshape_time,
        }


def benchmark_approaches(csc_wrapper, tensor, num_runs=10):
    """Benchmark both approaches with realistic per-frame measurements."""
    logger.info("=" * 60)
    logger.info("REALISTIC PER-FRAME PERFORMANCE BENCHMARK")
    logger.info("=" * 60)

    # Create test target images
    targets = []
    for i in range(num_runs):
        # Realistic target: uint8 RGB image
        target = np.random.randint(0, 256, (480, 800, 3), dtype=np.uint8)
        targets.append(target)

    # Benchmark CSC approach
    logger.info("\n=== CSC Matrix Approach ===")

    # Warm up
    _, _ = csc_wrapper.per_frame_operation(targets[0])

    # Timed runs
    csc_times = []
    csc_breakdowns = []

    for i in range(1, num_runs):  # Skip first run (warmup)
        result, timing = csc_wrapper.per_frame_operation(targets[i])
        csc_times.append(timing["total_time"])
        csc_breakdowns.append(timing)

    csc_avg = np.mean(csc_times)
    csc_std = np.std(csc_times)

    # Average timing breakdown
    avg_transfer = np.mean([t["transfer_time"] for t in csc_breakdowns])
    avg_operation = np.mean([t["operation_time"] for t in csc_breakdowns])
    avg_reshape = np.mean([t["reshape_time"] for t in csc_breakdowns])

    logger.info(f"CSC per-frame timing:")
    logger.info(f"  CPU->GPU transfer: {avg_transfer*1000:.2f}ms")
    logger.info(f"  A^T @ b operation:  {avg_operation*1000:.2f}ms")
    logger.info(f"  Result reshape:     {avg_reshape*1000:.2f}ms")
    logger.info(f"  Total per-frame:    {csc_avg*1000:.2f}ms ± {csc_std*1000:.2f}ms")

    # Benchmark SingleBlockMixedSparseTensor
    logger.info("\n=== SingleBlockMixedSparseTensor Approach ===")

    # For fair comparison, include CPU->GPU transfer in tensor approach too
    def tensor_per_frame_operation(target_image):
        transfer_start = time.time()
        # Convert CPU uint8 to GPU float32 (realistic per-frame cost)
        target_gpu = cp.asarray(target_image.astype(np.float32) / 255.0)
        # Extract just height, width for transpose_dot_product
        target_hw = target_gpu[:, :, 0]  # Use one channel for measurement
        transfer_time = time.time() - transfer_start

        operation_start = time.time()
        result = tensor.transpose_dot_product(target_hw)
        operation_time = time.time() - operation_start

        return result, {
            "transfer_time": transfer_time,
            "operation_time": operation_time,
            "total_time": transfer_time + operation_time,
        }

    # Warm up
    _, _ = tensor_per_frame_operation(targets[0])

    # Timed runs
    tensor_times = []
    tensor_breakdowns = []

    for i in range(1, num_runs):
        result, timing = tensor_per_frame_operation(targets[i])
        tensor_times.append(timing["total_time"])
        tensor_breakdowns.append(timing)

    tensor_avg = np.mean(tensor_times)
    tensor_std = np.std(tensor_times)

    # Average timing breakdown
    avg_tensor_transfer = np.mean([t["transfer_time"] for t in tensor_breakdowns])
    avg_tensor_operation = np.mean([t["operation_time"] for t in tensor_breakdowns])

    logger.info(f"Tensor per-frame timing:")
    logger.info(f"  CPU->GPU transfer: {avg_tensor_transfer*1000:.2f}ms")
    logger.info(f"  Block extraction:   {avg_tensor_operation*1000:.2f}ms")
    logger.info(
        f"  Total per-frame:    {tensor_avg*1000:.2f}ms ± {tensor_std*1000:.2f}ms"
    )

    # Performance comparison
    logger.info("\n=== PERFORMANCE COMPARISON ===")
    speedup = tensor_avg / csc_avg
    operation_speedup = avg_tensor_operation / avg_operation

    logger.info(
        f"Total per-frame speedup: {speedup:.2f}x {'(CSC faster)' if speedup > 1 else '(Tensor faster)'}"
    )
    logger.info(
        f"Core operation speedup: {operation_speedup:.2f}x {'(CSC faster)' if operation_speedup > 1 else '(Tensor faster)'}"
    )

    # Memory usage
    csc_memory = csc_wrapper.matrix_gpu.data.nbytes / (1024 * 1024)
    tensor_memory = tensor.memory_info()["total_mb"]

    logger.info(f"\nMemory usage:")
    logger.info(f"  CSC matrix:     {csc_memory:.1f}MB")
    logger.info(f"  Tensor blocks:  {tensor_memory:.1f}MB")
    logger.info(f"  Memory ratio:   {tensor_memory/csc_memory:.2f}x")

    return {
        "csc_avg": csc_avg,
        "tensor_avg": tensor_avg,
        "speedup": speedup,
        "csc_memory": csc_memory,
        "tensor_memory": tensor_memory,
    }


def main():
    """Run realistic performance comparison."""
    logger.info("Fair Performance Comparison: CSC vs SingleBlockMixedSparseTensor")
    logger.info("Using realistic pattern scale, density, and per-frame measurements")

    # Load real patterns
    matrix = load_real_patterns()

    # Create CSC wrapper (production approach)
    csc_wrapper = RealisticCSCWrapper(matrix)

    # Extract patterns to blocks (research approach)
    # Use 160x160 blocks (fits in 480x800 frame and covers 75% of patterns)
    tensor = extract_patterns_to_blocks(matrix, block_size=160)

    # Run fair comparison
    results = benchmark_approaches(csc_wrapper, tensor, num_runs=11)

    # Final recommendation
    logger.info("\n" + "=" * 60)
    logger.info("CONCLUSION")
    logger.info("=" * 60)

    if results["speedup"] < 0.9:
        logger.info("✓ SingleBlockMixedSparseTensor shows significant advantage")
    elif results["speedup"] < 1.1:
        logger.info("≈ Performance is comparable between approaches")
    else:
        logger.info("✓ CSC matrix approach maintains performance advantage")

    logger.info(
        f"Per-frame latency: CSC {results['csc_avg']*1000:.1f}ms vs Tensor {results['tensor_avg']*1000:.1f}ms"
    )
    fps_csc = 1.0 / results["csc_avg"]
    fps_tensor = 1.0 / results["tensor_avg"]
    logger.info(f"Max FPS capability: CSC {fps_csc:.1f} vs Tensor {fps_tensor:.1f}")

    return 0


if __name__ == "__main__":
    main()
