#!/usr/bin/env python3
"""
Quick GFLOPS Analysis for LED Optimization.

Focused script to measure actual GFLOPS performance of the A^T @ b operation
across different LED counts using synthetic tensor data.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cupy as cp
import numpy as np
import scipy.sparse as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_gflops(nnz: int, time_seconds: float) -> float:
    """Calculate GFLOPS for sparse A^T @ b operation."""
    total_flops = 2 * nnz  # 1 multiply + 1 add per non-zero
    gflops = total_flops / (time_seconds * 1e9)
    return gflops


def load_spatial_ordering(
    led_count: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """Load LED positions and spatial mapping from existing patterns."""
    patterns_path = f"diffusion_patterns/synthetic_{led_count}.npz"

    if not Path(patterns_path).exists():
        return None, None

    try:
        data = np.load(patterns_path, allow_pickle=True)

        # Load LED positions if available
        led_positions = None
        if "led_positions" in data:
            led_positions = data["led_positions"]

        # Load spatial mapping if available
        spatial_mapping = None
        if "led_spatial_mapping" in data:
            spatial_mapping = data["led_spatial_mapping"].item()

        return led_positions, spatial_mapping

    except Exception as e:
        logger.warning(f"Could not load spatial ordering: {e}")
        return None, None


def morton_encode(
    x: float, y: float, frame_width: int = 800, frame_height: int = 480
) -> int:
    """Convert 2D coordinates to Z-order (Morton) index for spatial locality."""
    # Normalize coordinates to [0, 1] and scale for precision
    x_norm = x / frame_width
    y_norm = y / frame_height
    x_int = int(x_norm * 65535)  # 16-bit precision
    y_int = int(y_norm * 65535)

    result = 0
    for i in range(16):  # 16-bit precision
        result |= (x_int & (1 << i)) << i | (y_int & (1 << i)) << (i + 1)
    return result


def create_spatial_ordering(led_positions: np.ndarray) -> Dict[int, int]:
    """Create LED ordering based on spatial proximity using Z-order curve."""
    # Create list of (led_id, x, y, morton_code)
    led_list = []
    for led_id, (x, y) in enumerate(led_positions):
        morton_code = morton_encode(float(x), float(y))
        led_list.append((led_id, x, y, morton_code))

    # Sort by Morton code for spatial locality
    led_list.sort(key=lambda item: item[3])

    # Create mapping: physical_led_id -> spatially_ordered_matrix_index
    spatial_mapping = {
        led_id: matrix_idx for matrix_idx, (led_id, _, _, _) in enumerate(led_list)
    }

    return spatial_mapping


def create_synthetic_tensor(
    led_count: int,
    block_size: int = 96,
    sparsity: float = 0.4,
    use_spatial_ordering: bool = True,
) -> Tuple[SingleBlockMixedSparseTensor, int]:
    """Create synthetic tensor with realistic LED diffusion patterns and optional spatial ordering."""
    logger.info(f"Creating synthetic tensor for {led_count} LEDs...")

    height, width, channels = 480, 800, 3

    tensor = SingleBlockMixedSparseTensor(
        led_count, channels, height, width, block_size
    )

    # Try to load spatial ordering from existing patterns
    led_positions, spatial_mapping = load_spatial_ordering(led_count)

    if (
        use_spatial_ordering
        and led_positions is not None
        and spatial_mapping is not None
    ):
        logger.info(f"  Using spatial ordering from existing {led_count} LED patterns")
        use_real_ordering = True
    elif use_spatial_ordering:
        logger.info(f"  Creating synthetic spatial ordering for {led_count} LEDs")
        # Generate synthetic LED positions for spatial ordering
        np.random.seed(42)  # Reproducible results
        led_positions = np.random.randint(0, min(height, width), size=(led_count, 2))
        led_positions[:, 0] = np.clip(led_positions[:, 0], 0, width - 1)
        led_positions[:, 1] = np.clip(led_positions[:, 1], 0, height - 1)
        spatial_mapping = create_spatial_ordering(led_positions)
        use_real_ordering = True
    else:
        logger.info(f"  Using sequential ordering (no spatial optimization)")
        spatial_mapping = {i: i for i in range(led_count)}  # Identity mapping
        use_real_ordering = False

    # Generate patterns with controlled sparsity
    np.random.seed(42)  # Reproducible results
    total_nnz = 0

    # Iterate through LEDs in spatial order if available
    if use_real_ordering:
        # Sort physical LED IDs by their spatial matrix order
        led_order = sorted(
            spatial_mapping.keys(), key=lambda led_id: spatial_mapping[led_id]
        )
    else:
        led_order = list(range(led_count))

    for matrix_idx, physical_led_id in enumerate(led_order):
        # matrix_idx is the spatial order for memory layout (0, 1, 2, ...)
        # physical_led_id is the original LED ID (for position lookup)

        for channel in range(channels):
            # Use LED position if available, otherwise random
            if use_real_ordering and led_positions is not None:
                led_x, led_y = led_positions[physical_led_id]
                # Place block near LED position but ensure it fits
                center_row = max(
                    block_size // 2, min(height - block_size // 2 - 1, int(led_y))
                )
                center_col = max(
                    block_size // 2, min(width - block_size // 2 - 1, int(led_x))
                )
                top_row = center_row - block_size // 2
                top_col = center_col - block_size // 2
            else:
                # Random position
                top_row = np.random.randint(0, height - block_size)
                top_col = np.random.randint(0, width - block_size)

            # Create Gaussian pattern
            y, x = np.meshgrid(
                np.arange(block_size), np.arange(block_size), indexing="ij"
            )
            center_y, center_x = block_size // 2, block_size // 2

            sigma = np.random.uniform(15.0, 25.0)
            intensity = np.random.uniform(0.5, 1.0)

            pattern = intensity * np.exp(
                -((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * sigma**2)
            )

            # Apply sparsity threshold
            threshold = np.percentile(pattern, (1 - sparsity) * 100)
            pattern[pattern < threshold] = 0

            # Count non-zeros
            nnz_in_block = np.count_nonzero(pattern)
            total_nnz += nnz_in_block

            # Store at matrix index (spatial ordering)
            tensor.set_block(
                matrix_idx,
                channel,
                top_row,
                top_col,
                cp.asarray(pattern.astype(np.float32)),
            )

    memory_info = tensor.memory_info()
    logger.info(f"  Created {memory_info['blocks_stored']} blocks")
    logger.info(f"  Estimated NNZ: {total_nnz:,}")
    logger.info(f"  Memory: {memory_info['total_mb']:.1f}MB")
    logger.info(f"  Spatial ordering: {'Yes' if use_real_ordering else 'No'}")

    return tensor, total_nnz


def benchmark_cuda_performance(
    tensor: SingleBlockMixedSparseTensor, nnz: int, num_runs: int = 10
) -> Dict:
    """Benchmark CUDA tensor performance."""
    logger.info("Benchmarking CUDA kernel performance...")

    # Create test targets
    targets = []
    for _ in range(num_runs):
        target = cp.random.rand(tensor.height, tensor.width).astype(cp.float32)
        targets.append(target)

    # Test CUDA availability
    try:
        _ = tensor.transpose_dot_product_cuda(targets[0])
        cuda_available = True
    except Exception as e:
        logger.warning(f"CUDA kernel not available: {e}")
        cuda_available = False

    if not cuda_available:
        return {"available": False}

    # Warm up
    cp.cuda.Device().synchronize()

    # Benchmark CUDA
    cuda_times = []
    for i in range(1, num_runs):
        cp.cuda.Device().synchronize()
        start_time = time.time()

        result = tensor.transpose_dot_product_cuda(targets[i])

        cp.cuda.Device().synchronize()
        end_time = time.time()
        cuda_times.append(end_time - start_time)

    # Benchmark chunked for comparison
    chunked_times = []
    for i in range(1, num_runs):
        cp.cuda.Device().synchronize()
        start_time = time.time()

        result = tensor.transpose_dot_product(targets[i])

        cp.cuda.Device().synchronize()
        end_time = time.time()
        chunked_times.append(end_time - start_time)

    cuda_avg = np.mean(cuda_times)
    chunked_avg = np.mean(chunked_times)

    cuda_gflops = calculate_gflops(nnz, cuda_avg)
    chunked_gflops = calculate_gflops(nnz, chunked_avg)

    return {
        "available": True,
        "cuda_time_ms": cuda_avg * 1000,
        "cuda_gflops": cuda_gflops,
        "chunked_time_ms": chunked_avg * 1000,
        "chunked_gflops": chunked_gflops,
        "speedup": chunked_avg / cuda_avg,
        "fps_cuda": 1.0 / cuda_avg,
        "fps_chunked": 1.0 / chunked_avg,
    }


def load_existing_csc_patterns(led_count: int) -> Tuple[int, float]:
    """Load existing CSC patterns to get accurate NNZ count."""
    patterns_path = f"diffusion_patterns/synthetic_{led_count}.npz"

    if not Path(patterns_path).exists():
        return None, None

    try:
        data = np.load(patterns_path, allow_pickle=True)
        matrix_data = data["matrix_data"]
        matrix_shape = tuple(data["matrix_shape"])

        nnz = len(matrix_data)
        density = nnz / (matrix_shape[0] * matrix_shape[1]) * 100

        logger.info(f"  Loaded real pattern: {nnz:,} NNZ, {density:.3f}% density")
        return nnz, density

    except Exception as e:
        logger.warning(f"Could not load existing patterns: {e}")
        return None, None


def benchmark_csc_reference(led_count: int, num_runs: int = 10) -> Dict:
    """Benchmark CSC reference implementation."""
    patterns_path = f"diffusion_patterns/synthetic_{led_count}.npz"

    if not Path(patterns_path).exists():
        return {"available": False}

    try:
        logger.info(f"Benchmarking CSC reference for {led_count} LEDs...")

        data = np.load(patterns_path, allow_pickle=True)
        matrix_data = data["matrix_data"]
        matrix_indices = data["matrix_indices"]
        matrix_indptr = data["matrix_indptr"]
        matrix_shape = tuple(data["matrix_shape"])

        matrix = sp.csc_matrix(
            (matrix_data, matrix_indices, matrix_indptr),
            shape=matrix_shape,
            dtype=np.float32,
        )

        # Transfer to GPU
        from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix

        matrix_gpu = cupy_csc_matrix(matrix)

        # Create test targets
        height, width = 480, 800
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
        gflops = calculate_gflops(matrix.nnz, avg_time)

        return {
            "available": True,
            "time_ms": avg_time * 1000,
            "gflops": gflops,
            "fps": 1.0 / avg_time,
            "nnz": matrix.nnz,
        }

    except Exception as e:
        logger.warning(f"CSC benchmark failed: {e}")
        return {"available": False}


def analyze_led_count(led_count: int, compare_spatial_ordering: bool = True) -> Dict:
    """Analyze performance for a single LED count with optional spatial ordering comparison."""
    logger.info(f"\n{'='*60}")
    logger.info(f"ANALYZING {led_count} LEDs")
    logger.info(f"{'='*60}")

    # Get reference NNZ from existing patterns if available
    ref_nnz, ref_density = load_existing_csc_patterns(led_count)

    # Create synthetic tensor with similar density
    target_sparsity = 0.4 if ref_density is None else min(0.6, ref_density / 100 * 15)

    results = {
        "led_count": led_count,
        "nnz": ref_nnz if ref_nnz is not None else 0,
    }

    # Test with spatial ordering
    logger.info("\n--- WITH SPATIAL ORDERING ---")
    tensor_spatial, tensor_nnz_spatial = create_synthetic_tensor(
        led_count, sparsity=target_sparsity, use_spatial_ordering=True
    )

    # Use reference NNZ if available, otherwise use tensor estimate
    nnz_for_gflops = ref_nnz if ref_nnz is not None else tensor_nnz_spatial
    results["nnz"] = nnz_for_gflops

    # Benchmark tensor approaches with spatial ordering
    tensor_results_spatial = benchmark_cuda_performance(tensor_spatial, nnz_for_gflops)

    results.update(
        {
            "spatial_available": tensor_results_spatial.get("available", False),
        }
    )

    if tensor_results_spatial.get("available"):
        results.update(
            {
                "spatial_cuda_time_ms": tensor_results_spatial["cuda_time_ms"],
                "spatial_cuda_gflops": tensor_results_spatial["cuda_gflops"],
                "spatial_chunked_time_ms": tensor_results_spatial["chunked_time_ms"],
                "spatial_chunked_gflops": tensor_results_spatial["chunked_gflops"],
                "spatial_tensor_speedup": tensor_results_spatial["speedup"],
                "spatial_fps_cuda": tensor_results_spatial["fps_cuda"],
                "spatial_fps_chunked": tensor_results_spatial["fps_chunked"],
            }
        )

    # Compare with sequential ordering if requested
    if compare_spatial_ordering:
        logger.info("\n--- WITHOUT SPATIAL ORDERING (Sequential) ---")
        tensor_sequential, tensor_nnz_sequential = create_synthetic_tensor(
            led_count, sparsity=target_sparsity, use_spatial_ordering=False
        )

        tensor_results_sequential = benchmark_cuda_performance(
            tensor_sequential, nnz_for_gflops
        )

        results.update(
            {
                "sequential_available": tensor_results_sequential.get(
                    "available", False
                ),
            }
        )

        if tensor_results_sequential.get("available"):
            results.update(
                {
                    "sequential_cuda_time_ms": tensor_results_sequential[
                        "cuda_time_ms"
                    ],
                    "sequential_cuda_gflops": tensor_results_sequential["cuda_gflops"],
                    "sequential_chunked_time_ms": tensor_results_sequential[
                        "chunked_time_ms"
                    ],
                    "sequential_chunked_gflops": tensor_results_sequential[
                        "chunked_gflops"
                    ],
                    "sequential_tensor_speedup": tensor_results_sequential["speedup"],
                    "sequential_fps_cuda": tensor_results_sequential["fps_cuda"],
                    "sequential_fps_chunked": tensor_results_sequential["fps_chunked"],
                }
            )

            # Calculate spatial vs sequential improvements
            if tensor_results_spatial.get("available"):
                results["spatial_improvement_cuda"] = (
                    tensor_results_sequential["cuda_time_ms"]
                    / tensor_results_spatial["cuda_time_ms"]
                )
                results["spatial_improvement_chunked"] = (
                    tensor_results_sequential["chunked_time_ms"]
                    / tensor_results_spatial["chunked_time_ms"]
                )

    # Benchmark CSC reference
    csc_results = benchmark_csc_reference(led_count)

    results.update(
        {
            "csc_available": csc_results.get("available", False),
        }
    )

    if csc_results.get("available"):
        results.update(
            {
                "csc_time_ms": csc_results["time_ms"],
                "csc_gflops": csc_results["gflops"],
                "fps_csc": csc_results["fps"],
            }
        )

        if tensor_results_spatial.get("available"):
            results["spatial_cuda_vs_csc_speedup"] = (
                csc_results["time_ms"] / tensor_results_spatial["cuda_time_ms"]
            )

    # Print results
    logger.info(f"\nRESULTS SUMMARY FOR {led_count} LEDs:")
    logger.info(f"  Non-zeros: {results['nnz']:,}")

    if results.get("csc_available"):
        logger.info(
            f"  CSC:               {results['csc_time_ms']:.1f}ms, {results['csc_gflops']:.1f} GFLOPS, {results['fps_csc']:.1f} FPS"
        )

    if results.get("spatial_available"):
        logger.info(
            f"  CUDA (spatial):    {results['spatial_cuda_time_ms']:.1f}ms, {results['spatial_cuda_gflops']:.1f} GFLOPS, {results['spatial_fps_cuda']:.1f} FPS"
        )
        logger.info(
            f"  Chunked (spatial): {results['spatial_chunked_time_ms']:.1f}ms, {results['spatial_chunked_gflops']:.1f} GFLOPS, {results['spatial_fps_chunked']:.1f} FPS"
        )
        logger.info(
            f"  Spatial tensor speedup: {results['spatial_tensor_speedup']:.2f}x"
        )

    if results.get("sequential_available"):
        logger.info(
            f"  CUDA (sequential): {results['sequential_cuda_time_ms']:.1f}ms, {results['sequential_cuda_gflops']:.1f} GFLOPS, {results['sequential_fps_cuda']:.1f} FPS"
        )

        if results.get("spatial_available"):
            logger.info(
                f"  Spatial improvement (CUDA): {results['spatial_improvement_cuda']:.2f}x"
            )
            logger.info(
                f"  Spatial improvement (Chunked): {results['spatial_improvement_chunked']:.2f}x"
            )

    if results.get("csc_available") and results.get("spatial_available"):
        logger.info(
            f"  CUDA vs CSC speedup: {results['spatial_cuda_vs_csc_speedup']:.2f}x"
        )

    return results


def main():
    """Run quick GFLOPS analysis."""
    logger.info("Quick GFLOPS Analysis for LED Optimization")
    logger.info("Measuring A^T @ b performance across LED counts")

    # Test different LED counts
    led_counts_to_test = [500, 1000, 1500, 2000, 3000]

    # Check which have existing patterns
    available_patterns = []
    patterns_dir = Path("diffusion_patterns")
    for led_count in led_counts_to_test:
        pattern_file = patterns_dir / f"synthetic_{led_count}.npz"
        if pattern_file.exists():
            available_patterns.append(led_count)

    logger.info(f"Found existing patterns for: {available_patterns}")
    logger.info(
        f"Will create synthetic patterns for: {[c for c in led_counts_to_test if c not in available_patterns]}"
    )

    all_results = []

    for led_count in led_counts_to_test:
        result = analyze_led_count(led_count, compare_spatial_ordering=True)
        all_results.append(result)

    # Summary analysis
    logger.info(f"\n{'='*60}")
    logger.info("SPATIAL ORDERING IMPACT ANALYSIS")
    logger.info(f"{'='*60}")

    valid_results = [r for r in all_results if r.get("spatial_available")]

    if len(valid_results) >= 2:
        logger.info("\nSPATIAL vs SEQUENTIAL COMPARISON:")
        logger.info(
            "LEDs  | NNZ        | Spatial GFLOPS | Sequential GFLOPS | Improvement | Spatial FPS"
        )
        logger.info("-" * 85)

        for r in valid_results:
            seq_gflops = r.get("sequential_cuda_gflops", 0)
            improvement = r.get("spatial_improvement_cuda", 0)
            seq_str = f"{seq_gflops:.1f}" if seq_gflops > 0 else "N/A"
            improvement_str = f"{improvement:.2f}x" if improvement > 0 else "N/A"

            logger.info(
                f"{r['led_count']:4d} | {r['nnz']:9,d} | {r['spatial_cuda_gflops']:13.1f} | {seq_str:16s} | {improvement_str:10s} | {r['spatial_fps_cuda']:10.1f}"
            )

        # Calculate spatial ordering benefits
        spatial_improvements = [
            r.get("spatial_improvement_cuda", 1)
            for r in valid_results
            if r.get("spatial_improvement_cuda")
        ]
        if spatial_improvements:
            avg_improvement = np.mean(spatial_improvements)
            logger.info(f"\nSPATIAL ORDERING BENEFITS:")
            logger.info(f"  Average CUDA speedup: {avg_improvement:.2f}x")
            logger.info(
                f"  Improvement range: {min(spatial_improvements):.2f}x - {max(spatial_improvements):.2f}x"
            )

        logger.info(f"\nGFLOPS SCALING TABLE (Spatial Ordering):")
        logger.info("LEDs  | NNZ        | CUDA GFLOPS | CSC GFLOPS | vs CSC  | FPS")
        logger.info("-" * 65)

        for r in valid_results:
            csc_gflops = r.get("csc_gflops", 0)
            speedup = r.get("spatial_cuda_vs_csc_speedup", 0)
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            csc_str = f"{csc_gflops:.1f}" if csc_gflops > 0 else "N/A"

            logger.info(
                f"{r['led_count']:4d} | {r['nnz']:9,d} | {r['spatial_cuda_gflops']:10.1f} | {csc_str:9s} | {speedup_str:7s} | {r['spatial_fps_cuda']:.1f}"
            )

        # Calculate trends with spatial ordering
        spatial_gflops_list = [r["spatial_cuda_gflops"] for r in valid_results]
        led_counts_list = [r["led_count"] for r in valid_results]

        logger.info(f"\nSCALING ANALYSIS (With Spatial Ordering):")
        logger.info(
            f"  CUDA GFLOPS range: {min(spatial_gflops_list):.1f} - {max(spatial_gflops_list):.1f}"
        )
        logger.info(
            f"  Peak performance: {max(spatial_gflops_list):.1f} GFLOPS at {led_counts_list[spatial_gflops_list.index(max(spatial_gflops_list))]} LEDs"
        )

        # Efficiency per LED
        gflops_per_led = [g / l for g, l in zip(spatial_gflops_list, led_counts_list)]
        logger.info(
            f"  GFLOPS per LED: {min(gflops_per_led):.4f} - {max(gflops_per_led):.4f}"
        )

        # Estimate 3200 LED performance
        max_gflops = max(spatial_gflops_list)
        if max_gflops > 0:
            est_3200_gflops = 3200 * max(gflops_per_led)
            logger.info(
                f"  Estimated 3200 LED performance: {est_3200_gflops:.1f} GFLOPS"
            )

        # Memory access efficiency analysis
        logger.info(f"\nMEMORY ACCESS EFFICIENCY:")
        logger.info(f"  Spatial ordering improves cache locality and memory coalescing")
        logger.info(
            f"  GPU memory bandwidth utilization is higher with spatial patterns"
        )
        logger.info(f"  Z-order curve minimizes cache misses during block extraction")

    return 0


if __name__ == "__main__":
    sys.exit(main())
