#!/usr/bin/env python3
"""
GFLOPS Scaling Analysis for LED Optimization.

This script analyzes the GFLOPS performance and scaling behavior of the A^T @ b
operation across different LED counts, comparing CSC matrices, chunked tensors,
and CUDA kernels.

The script measures actual performance rather than predicting it, using real
synthetic diffusion patterns to understand GPU utilization and scaling.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_theoretical_gflops(nnz: int, time_seconds: float) -> float:
    """
    Calculate GFLOPS based on sparse matrix operations.

    For sparse A^T @ b: each non-zero element requires 1 multiply + 1 add = 2 FLOPs

    Args:
        nnz: Number of non-zero elements in the matrix
        time_seconds: Time to complete the operation

    Returns:
        GFLOPS (Giga Floating Point Operations Per Second)
    """
    total_flops = 2 * nnz  # 1 multiply + 1 add per non-zero
    gflops = total_flops / (time_seconds * 1e9)
    return gflops


def load_csc_patterns(led_count: int) -> Tuple[sp.csc_matrix, Dict]:
    """Load CSC patterns for given LED count."""
    patterns_path = f"diffusion_patterns/synthetic_{led_count}.npz"

    if not Path(patterns_path).exists():
        raise FileNotFoundError(f"Pattern file not found: {patterns_path}")

    logger.info(f"Loading CSC patterns for {led_count} LEDs...")
    data = np.load(patterns_path, allow_pickle=True)

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

    # Load LED spatial mapping (if available)
    led_spatial_mapping = None
    if "led_spatial_mapping" in data:
        try:
            led_spatial_mapping = data["led_spatial_mapping"].item()
        except Exception:
            pass

    stats = {
        "shape": matrix.shape,
        "nnz": matrix.nnz,
        "density": matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100,
        "memory_mb": matrix.data.nbytes / (1024 * 1024),
    }

    logger.info(f"  Shape: {stats['shape']}")
    logger.info(f"  Non-zeros: {stats['nnz']:,}")
    logger.info(f"  Density: {stats['density']:.3f}%")
    logger.info(f"  Memory: {stats['memory_mb']:.1f}MB")

    return matrix, stats


def create_tensor_from_csc(
    matrix: sp.csc_matrix,
    led_count: int,
    height: int = 480,
    width: int = 800,
    block_size: int = 96,
) -> SingleBlockMixedSparseTensor:
    """Create SingleBlockMixedSparseTensor from CSC matrix for comparison."""
    logger.info(f"Converting CSC to tensor format for {led_count} LEDs...")

    channels = 3
    tensor = SingleBlockMixedSparseTensor(
        led_count, channels, height, width, block_size
    )

    # Convert CSC matrix back to block format
    # This is approximate - we extract dominant patterns for each LED/channel
    pixels = height * width

    for led_id in range(led_count):
        for channel in range(channels):
            # Get column corresponding to this LED/channel
            col_idx = channel * led_count + led_id
            if col_idx >= matrix.shape[1]:
                continue

            # Extract non-zero pattern for this LED/channel
            col_start = matrix.indptr[col_idx]
            col_end = matrix.indptr[col_idx + 1]

            if col_start == col_end:  # No non-zeros
                continue

            # Get non-zero indices and values
            row_indices = matrix.indices[col_start:col_end]
            values = matrix.data[col_start:col_end]

            # Convert to 2D coordinates (channel-adjusted)
            pixel_indices = row_indices % pixels
            pixel_rows = pixel_indices // width
            pixel_cols = pixel_indices % width

            # Find bounding box
            min_row, max_row = pixel_rows.min(), pixel_rows.max()
            min_col, max_col = pixel_cols.min(), pixel_cols.max()

            # Create block that encompasses the pattern
            block_height = min(block_size, max_row - min_row + 1)
            block_width = min(block_size, max_col - min_col + 1)

            # Center the block around the pattern
            top_row = max(0, min(height - block_size, min_row))
            top_col = max(0, min(width - block_size, min_col))

            # Create dense block
            block = np.zeros((block_size, block_size), dtype=np.float32)

            # Fill block with pattern values
            for i, (pr, pc, val) in enumerate(zip(pixel_rows, pixel_cols, values)):
                block_r = pr - top_row
                block_c = pc - top_col
                if 0 <= block_r < block_size and 0 <= block_c < block_size:
                    block[block_r, block_c] = val

            tensor.set_block(led_id, channel, top_row, top_col, cp.asarray(block))

    memory_info = tensor.memory_info()
    logger.info(f"  Tensor blocks stored: {memory_info['blocks_stored']}")
    logger.info(f"  Tensor memory: {memory_info['total_mb']:.1f}MB")

    return tensor


def benchmark_csc_approach(
    matrix: sp.csc_matrix, led_count: int, num_runs: int = 10
) -> Tuple[float, float]:
    """Benchmark CSC matrix approach."""
    logger.info(f"Benchmarking CSC approach for {led_count} LEDs...")

    height, width = 480, 800

    # Transfer to GPU
    from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix

    matrix_gpu = cupy_csc_matrix(matrix)

    # Create test targets
    targets = []
    for _ in range(num_runs):
        target = cp.random.rand(height * width * 3).astype(cp.float32)
        targets.append(target)

    # Warm up
    _ = matrix_gpu.T @ targets[0]
    cp.cuda.Device().synchronize()

    # Benchmark
    times = []
    for i in range(1, num_runs):
        cp.cuda.Device().synchronize()
        start_time = time.time()

        result = matrix_gpu.T @ targets[i]

        cp.cuda.Device().synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    gflops = calculate_theoretical_gflops(matrix.nnz, avg_time)

    logger.info(f"  CSC time: {avg_time*1000:.2f}ms")
    logger.info(f"  CSC GFLOPS: {gflops:.2f}")

    return avg_time, gflops


def benchmark_cuda_tensor(
    tensor: SingleBlockMixedSparseTensor, nnz_estimate: int, num_runs: int = 10
) -> Tuple[float, float]:
    """Benchmark CUDA tensor approach."""
    led_count = tensor.batch_size
    logger.info(f"Benchmarking CUDA tensor for {led_count} LEDs...")

    # Create test targets
    targets = []
    for _ in range(num_runs):
        target = cp.random.rand(tensor.height, tensor.width).astype(cp.float32)
        targets.append(target)

    # Warm up
    try:
        _ = tensor.transpose_dot_product_cuda(targets[0])
        cuda_available = True
    except Exception as e:
        logger.warning(f"CUDA kernel not available: {e}")
        cuda_available = False
        return float("inf"), 0.0

    cp.cuda.Device().synchronize()

    # Benchmark
    times = []
    for i in range(1, num_runs):
        cp.cuda.Device().synchronize()
        start_time = time.time()

        result = tensor.transpose_dot_product_cuda(targets[i])

        cp.cuda.Device().synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    gflops = calculate_theoretical_gflops(nnz_estimate, avg_time)

    logger.info(f"  CUDA time: {avg_time*1000:.2f}ms")
    logger.info(f"  CUDA GFLOPS: {gflops:.2f}")

    return avg_time, gflops


def benchmark_chunked_tensor(
    tensor: SingleBlockMixedSparseTensor, nnz_estimate: int, num_runs: int = 10
) -> Tuple[float, float]:
    """Benchmark chunked tensor approach."""
    led_count = tensor.batch_size
    logger.info(f"Benchmarking chunked tensor for {led_count} LEDs...")

    # Create test targets
    targets = []
    for _ in range(num_runs):
        target = cp.random.rand(tensor.height, tensor.width).astype(cp.float32)
        targets.append(target)

    # Warm up
    _ = tensor.transpose_dot_product(targets[0])
    cp.cuda.Device().synchronize()

    # Benchmark
    times = []
    for i in range(1, num_runs):
        cp.cuda.Device().synchronize()
        start_time = time.time()

        result = tensor.transpose_dot_product(targets[i])

        cp.cuda.Device().synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    gflops = calculate_theoretical_gflops(nnz_estimate, avg_time)

    logger.info(f"  Chunked time: {avg_time*1000:.2f}ms")
    logger.info(f"  Chunked GFLOPS: {gflops:.2f}")

    return avg_time, gflops


def analyze_single_led_count(led_count: int) -> Dict:
    """Analyze performance for a single LED count."""
    logger.info(f"\n{'='*70}")
    logger.info(f"ANALYZING {led_count} LEDs")
    logger.info(f"{'='*70}")

    try:
        # Load CSC patterns
        matrix, csc_stats = load_csc_patterns(led_count)

        # Create tensor from CSC for comparison
        tensor = create_tensor_from_csc(matrix, led_count)

        # Benchmark all approaches
        num_runs = 10

        csc_time, csc_gflops = benchmark_csc_approach(matrix, led_count, num_runs)
        cuda_time, cuda_gflops = benchmark_cuda_tensor(tensor, matrix.nnz, num_runs)
        chunked_time, chunked_gflops = benchmark_chunked_tensor(
            tensor, matrix.nnz, num_runs
        )

        # Calculate speedups
        csc_cuda_speedup = csc_time / cuda_time if cuda_time > 0 else 0
        chunked_cuda_speedup = chunked_time / cuda_time if cuda_time > 0 else 0

        results = {
            "led_count": led_count,
            "nnz": matrix.nnz,
            "density": csc_stats["density"],
            "csc_time_ms": csc_time * 1000,
            "csc_gflops": csc_gflops,
            "cuda_time_ms": cuda_time * 1000,
            "cuda_gflops": cuda_gflops,
            "chunked_time_ms": chunked_time * 1000,
            "chunked_gflops": chunked_gflops,
            "csc_cuda_speedup": csc_cuda_speedup,
            "chunked_cuda_speedup": chunked_cuda_speedup,
            "fps_csc": 1.0 / csc_time,
            "fps_cuda": 1.0 / cuda_time if cuda_time > 0 else 0,
            "fps_chunked": 1.0 / chunked_time,
        }

        logger.info(f"\nRESULTS FOR {led_count} LEDs:")
        logger.info(f"  Non-zeros: {results['nnz']:,}")
        logger.info(f"  Density: {results['density']:.3f}%")
        logger.info(
            f"  CSC:     {results['csc_time_ms']:.1f}ms, {results['csc_gflops']:.1f} GFLOPS, {results['fps_csc']:.1f} FPS"
        )
        if cuda_time < float("inf"):
            logger.info(
                f"  CUDA:    {results['cuda_time_ms']:.1f}ms, {results['cuda_gflops']:.1f} GFLOPS, {results['fps_cuda']:.1f} FPS"
            )
            logger.info(f"  Speedup: {results['csc_cuda_speedup']:.2f}x")
        logger.info(
            f"  Chunked: {results['chunked_time_ms']:.1f}ms, {results['chunked_gflops']:.1f} GFLOPS, {results['fps_chunked']:.1f} FPS"
        )

        return results

    except FileNotFoundError as e:
        logger.warning(f"Skipping {led_count} LEDs: {e}")
        return None
    except Exception as e:
        logger.error(f"Error analyzing {led_count} LEDs: {e}")
        return None


def find_available_led_counts() -> List[int]:
    """Find available LED pattern files."""
    available = []
    patterns_dir = Path("diffusion_patterns")

    for pattern_file in patterns_dir.glob("synthetic_*.npz"):
        try:
            led_count_str = pattern_file.stem.split("_")[1]
            led_count = int(led_count_str)
            available.append(led_count)
        except (IndexError, ValueError):
            continue

    available.sort()
    logger.info(f"Found patterns for LED counts: {available}")
    return available


def analyze_scaling_trends(results: List[Dict]) -> None:
    """Analyze scaling trends across LED counts."""
    logger.info(f"\n{'='*70}")
    logger.info("SCALING ANALYSIS")
    logger.info(f"{'='*70}")

    # Sort by LED count
    results = sorted(
        [r for r in results if r is not None], key=lambda x: x["led_count"]
    )

    if len(results) < 2:
        logger.warning("Need at least 2 data points for scaling analysis")
        return

    # Print summary table
    logger.info("\nPERFORMANCE SCALING TABLE:")
    logger.info(
        "LED Count | Non-zeros  | CSC GFLOPS | CUDA GFLOPS | Speedup | Memory Efficiency"
    )
    logger.info("-" * 80)

    for r in results:
        speedup_str = f"{r['csc_cuda_speedup']:.2f}x" if r["cuda_gflops"] > 0 else "N/A"
        logger.info(
            f"{r['led_count']:8d} | {r['nnz']:9,d} | {r['csc_gflops']:9.1f} | {r['cuda_gflops']:10.1f} | {speedup_str:7s} | {r['density']:.2f}%"
        )

    # Analyze trends
    led_counts = [r["led_count"] for r in results]
    csc_gflops = [r["csc_gflops"] for r in results]
    cuda_gflops = [r["cuda_gflops"] for r in results if r["cuda_gflops"] > 0]

    if len(cuda_gflops) > 1:
        logger.info(f"\nTREND ANALYSIS:")
        logger.info(f"  LED count range: {min(led_counts)} - {max(led_counts)}")
        logger.info(
            f"  CSC GFLOPS range: {min(csc_gflops):.1f} - {max(csc_gflops):.1f}"
        )
        logger.info(
            f"  CUDA GFLOPS range: {min(cuda_gflops):.1f} - {max(cuda_gflops):.1f}"
        )

        # Calculate per-LED performance
        avg_gflops_per_led = [
            g / c for g, c in zip(cuda_gflops, led_counts[: len(cuda_gflops)])
        ]
        logger.info(
            f"  GFLOPS per LED: {min(avg_gflops_per_led):.4f} - {max(avg_gflops_per_led):.4f}"
        )

        # Predict peak performance
        if max(cuda_gflops) > 0:
            peak_led_estimate = 3200 * max(cuda_gflops) / max(csc_gflops)
            logger.info(
                f"  Estimated peak throughput at 3200 LEDs: {peak_led_estimate:.1f} GFLOPS"
            )


def main():
    """Run comprehensive GFLOPS scaling analysis."""
    logger.info("GFLOPS Scaling Analysis for LED Optimization")
    logger.info("Testing A^T @ b performance across different LED counts")
    logger.info("=" * 70)

    # Find available LED counts
    available_led_counts = find_available_led_counts()

    if not available_led_counts:
        logger.error("No synthetic pattern files found!")
        logger.info("Generate patterns using:")
        logger.info(
            "  python tools/generate_synthetic_patterns.py --output diffusion_patterns/synthetic_1000.npz --led-count 1000"
        )
        return 1

    # Analyze each LED count
    all_results = []
    for led_count in available_led_counts:
        result = analyze_single_led_count(led_count)
        if result:
            all_results.append(result)

    # Analyze scaling trends
    if all_results:
        analyze_scaling_trends(all_results)

    logger.info(f"\nAnalysis complete. Tested {len(all_results)} LED configurations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
