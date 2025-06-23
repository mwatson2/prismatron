#!/usr/bin/env python3
"""
High-Performance CUDA Kernel Test for 2600 LEDs.

Tests the new high-performance kernel targeting 20+ GFLOPS performance
for the production Prismatron system with 2600 LEDs.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import cupy as cp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_gflops(
    led_count: int, channels: int, block_size: int, time_seconds: float
) -> float:
    """Calculate GFLOPS for the A^T @ b operation."""
    total_flops = (
        led_count * channels * block_size * block_size * 2
    )  # 2 FLOPs per element
    gflops = total_flops / (time_seconds * 1e9)
    return gflops


def calculate_memory_bandwidth(
    led_count: int, channels: int, block_size: int, time_seconds: float
) -> float:
    """Calculate memory bandwidth utilization in GB/s."""
    # Memory reads: sparse patterns + target image blocks
    sparse_bytes = (
        led_count * channels * block_size * block_size * 4
    )  # 4 bytes per float
    target_bytes = 480 * 800 * 4  # Target image size
    total_bytes = sparse_bytes + target_bytes

    bandwidth_gbps = total_bytes / (time_seconds * 1e9)
    return bandwidth_gbps


def create_production_tensor(led_count: int = 2600) -> SingleBlockMixedSparseTensor:
    """Create production-scale tensor for 2600 LEDs."""
    logger.info(f"Creating production tensor: {led_count} LEDs, 96x96 blocks...")

    # Production specifications
    height, width, channels = 480, 800, 3
    block_size = 96

    tensor = SingleBlockMixedSparseTensor(
        led_count, channels, height, width, block_size
    )

    # Generate realistic LED diffusion patterns
    np.random.seed(42)  # Reproducible

    for led_id in range(led_count):
        for channel in range(channels):
            # Realistic LED position distribution
            # LEDs are typically distributed across the display area
            led_x = (led_id % 50) * 16  # Distribute across width
            led_y = (led_id // 50) * 10  # Distribute across height

            # Clamp to ensure block fits in image
            center_row = max(block_size // 2, min(height - block_size // 2 - 1, led_y))
            center_col = max(block_size // 2, min(width - block_size // 2 - 1, led_x))
            top_row = center_row - block_size // 2
            top_col = center_col - block_size // 2

            # Create realistic diffusion pattern
            y, x = np.meshgrid(
                np.arange(block_size), np.arange(block_size), indexing="ij"
            )
            center_y, center_x = block_size // 2, block_size // 2

            # Varying LED characteristics
            sigma = np.random.uniform(12.0, 28.0)  # LED spread
            intensity = np.random.uniform(0.3, 1.0)  # LED brightness

            # Gaussian diffusion with exponential falloff
            pattern = intensity * np.exp(
                -((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * sigma**2)
            )

            # Apply realistic sparsity (keep top 35% - typical for LED diffusion)
            threshold = np.percentile(pattern, 65)
            pattern[pattern < threshold] = 0

            tensor.set_block(
                led_id,
                channel,
                top_row,
                top_col,
                cp.asarray(pattern.astype(np.float32)),
            )

    memory_info = tensor.memory_info()
    logger.info(f"  Created {memory_info['blocks_stored']} blocks")
    logger.info(f"  Memory usage: {memory_info['total_mb']:.1f}MB")
    logger.info(f"  Compression ratio: {memory_info['compression_ratio']:.3f}")

    return tensor


def verify_correctness(tensor: SingleBlockMixedSparseTensor) -> bool:
    """Quick correctness verification against chunked reference."""
    logger.info("Verifying high-performance kernel correctness...")

    # Create test target
    target = cp.random.rand(tensor.height, tensor.width).astype(cp.float32)

    # Reference result (chunked)
    ref_result = tensor.transpose_dot_product(target)

    # High-performance result
    try:
        hp_result = tensor.transpose_dot_product_cuda_high_performance(target)

        # Compare results
        max_abs_error = float(cp.max(cp.abs(ref_result - hp_result)))
        max_rel_error = float(
            cp.max(cp.abs((ref_result - hp_result) / (cp.abs(ref_result) + 1e-8)))
        )

        tolerance = 5e-4  # Allow for floating point differences with large datasets
        passed = max_abs_error < tolerance and max_rel_error < tolerance

        if passed:
            logger.info(
                f"  ✅ Correctness check PASSED (max_abs_err={max_abs_error:.2e}, max_rel_err={max_rel_error:.2e})"
            )
        else:
            logger.error(
                f"  ❌ Correctness check FAILED (max_abs_err={max_abs_error:.2e}, max_rel_err={max_rel_error:.2e})"
            )

        return passed

    except Exception as e:
        logger.error(f"  ❌ High-performance kernel failed: {e}")
        return False


def benchmark_all_kernels(
    tensor: SingleBlockMixedSparseTensor, num_runs: int = 20
) -> Dict:
    """Benchmark all available kernel implementations."""
    led_count = tensor.batch_size
    logger.info(f"Benchmarking all kernels for {led_count} LEDs ({num_runs} runs)...")

    # Create test targets
    targets = [
        cp.random.rand(tensor.height, tensor.width).astype(cp.float32)
        for _ in range(num_runs)
    ]

    results = {
        "led_count": led_count,
        "block_size": tensor.block_size,
        "total_flops": led_count * tensor.channels * tensor.block_size**2 * 2,
    }

    # Test chunked implementation (reference)
    logger.info("  Testing chunked implementation...")
    times = []
    _ = tensor.transpose_dot_product(targets[0])  # Warm up
    cp.cuda.Device().synchronize()

    for i in range(1, num_runs):
        cp.cuda.Device().synchronize()
        start = time.time()
        _ = tensor.transpose_dot_product(targets[i])
        cp.cuda.Device().synchronize()
        times.append(time.time() - start)

    chunked_time = np.mean(times)
    chunked_gflops = calculate_gflops(
        led_count, tensor.channels, tensor.block_size, chunked_time
    )
    chunked_bandwidth = calculate_memory_bandwidth(
        led_count, tensor.channels, tensor.block_size, chunked_time
    )

    results.update(
        {
            "chunked_time_ms": chunked_time * 1000,
            "chunked_gflops": chunked_gflops,
            "chunked_bandwidth_gbps": chunked_bandwidth,
            "chunked_fps": 1.0 / chunked_time,
        }
    )

    logger.info(
        f"    Time: {chunked_time*1000:.2f}ms, GFLOPS: {chunked_gflops:.2f}, FPS: {1/chunked_time:.1f}"
    )

    # Test corrected CUDA kernel
    logger.info("  Testing corrected CUDA kernel...")
    try:
        times = []
        _ = tensor.transpose_dot_product_cuda_corrected(targets[0])  # Warm up
        cp.cuda.Device().synchronize()

        for i in range(1, num_runs):
            cp.cuda.Device().synchronize()
            start = time.time()
            _ = tensor.transpose_dot_product_cuda_corrected(targets[i])
            cp.cuda.Device().synchronize()
            times.append(time.time() - start)

        corrected_time = np.mean(times)
        corrected_gflops = calculate_gflops(
            led_count, tensor.channels, tensor.block_size, corrected_time
        )
        corrected_bandwidth = calculate_memory_bandwidth(
            led_count, tensor.channels, tensor.block_size, corrected_time
        )

        results.update(
            {
                "corrected_available": True,
                "corrected_time_ms": corrected_time * 1000,
                "corrected_gflops": corrected_gflops,
                "corrected_bandwidth_gbps": corrected_bandwidth,
                "corrected_fps": 1.0 / corrected_time,
                "corrected_speedup": chunked_time / corrected_time,
            }
        )

        logger.info(
            f"    Time: {corrected_time*1000:.2f}ms, GFLOPS: {corrected_gflops:.2f}, FPS: {1/corrected_time:.1f}, Speedup: {chunked_time/corrected_time:.2f}x"
        )

    except Exception as e:
        logger.warning(f"    Corrected kernel failed: {e}")
        results["corrected_available"] = False

    # Test high-performance CUDA kernel
    logger.info("  Testing HIGH-PERFORMANCE CUDA kernel...")
    try:
        times = []
        _ = tensor.transpose_dot_product_cuda_high_performance(targets[0])  # Warm up
        cp.cuda.Device().synchronize()

        for i in range(1, num_runs):
            cp.cuda.Device().synchronize()
            start = time.time()
            _ = tensor.transpose_dot_product_cuda_high_performance(targets[i])
            cp.cuda.Device().synchronize()
            times.append(time.time() - start)

        hp_time = np.mean(times)
        hp_gflops = calculate_gflops(
            led_count, tensor.channels, tensor.block_size, hp_time
        )
        hp_bandwidth = calculate_memory_bandwidth(
            led_count, tensor.channels, tensor.block_size, hp_time
        )

        results.update(
            {
                "high_performance_available": True,
                "high_performance_time_ms": hp_time * 1000,
                "high_performance_gflops": hp_gflops,
                "high_performance_bandwidth_gbps": hp_bandwidth,
                "high_performance_fps": 1.0 / hp_time,
                "high_performance_speedup": chunked_time / hp_time,
            }
        )

        if results.get("corrected_available"):
            results["hp_vs_corrected"] = results["corrected_time_ms"] / (hp_time * 1000)

        logger.info(
            f"    Time: {hp_time*1000:.2f}ms, GFLOPS: {hp_gflops:.2f}, FPS: {1/hp_time:.1f}, Speedup: {chunked_time/hp_time:.2f}x"
        )

        # Check if we hit our target
        target_gflops = 20.0
        if hp_gflops >= target_gflops:
            logger.info(
                f"    🎯 TARGET ACHIEVED: {hp_gflops:.1f} GFLOPS >= {target_gflops} GFLOPS!"
            )
        else:
            logger.warning(
                f"    ⚠️  Target missed: {hp_gflops:.1f} GFLOPS < {target_gflops} GFLOPS"
            )

    except Exception as e:
        logger.error(f"    High-performance kernel failed: {e}")
        results["high_performance_available"] = False

    return results


def analyze_gpu_utilization(results: Dict):
    """Analyze GPU utilization and bottlenecks."""
    logger.info("\\nGPU UTILIZATION ANALYSIS:")

    led_count = results["led_count"]

    # GPU specifications for NVIDIA Jetson Orin Nano
    theoretical_tflops = 67.0  # Peak tensor performance
    memory_bandwidth_gbps = 68.0  # Memory bandwidth

    logger.info(f"  Target GPU: NVIDIA Jetson Orin Nano (67 TFLOPs, 68 GB/s)")
    logger.info(f"  Workload: {led_count} LEDs × 3 channels × 96×96 blocks")

    if results.get("high_performance_available"):
        achieved_gflops = results["high_performance_gflops"]
        achieved_bandwidth = results["high_performance_bandwidth_gbps"]

        # Calculate utilization percentages
        compute_utilization = (achieved_gflops / 1000) / theoretical_tflops * 100
        memory_utilization = achieved_bandwidth / memory_bandwidth_gbps * 100

        logger.info(f"\\n  HIGH-PERFORMANCE KERNEL ANALYSIS:")
        logger.info(
            f"    Achieved: {achieved_gflops:.2f} GFLOPS ({compute_utilization:.2f}% of peak)"
        )
        logger.info(
            f"    Memory: {achieved_bandwidth:.2f} GB/s ({memory_utilization:.2f}% of peak)"
        )
        logger.info(f"    FPS: {results['high_performance_fps']:.1f}")

        # Identify bottleneck
        if memory_utilization > compute_utilization * 2:
            logger.info(f"    🔍 BOTTLENECK: Memory bandwidth limited")
            logger.info(
                f"    💡 Optimization: Reduce memory access, increase compute intensity"
            )
        elif compute_utilization > memory_utilization * 2:
            logger.info(f"    🔍 BOTTLENECK: Compute limited")
            logger.info(
                f"    💡 Optimization: Increase parallelism, optimize arithmetic"
            )
        else:
            logger.info(f"    🔍 BOTTLENECK: Balanced (memory and compute)")

        # Real-time analysis
        target_fps = 60  # High frame rate for smooth optimization
        fps_headroom = results["high_performance_fps"] / target_fps

        logger.info(f"\\n  REAL-TIME PERFORMANCE:")
        logger.info(f"    Target: {target_fps} FPS for smooth optimization")
        logger.info(f"    Achieved: {results['high_performance_fps']:.1f} FPS")

        if fps_headroom >= 1.0:
            logger.info(f"    ✅ REAL-TIME TARGET MET ({fps_headroom:.1f}x headroom)")
        else:
            logger.warning(f"    ❌ Below real-time target ({fps_headroom:.2f}x)")

        # Scaling projection
        max_leds_60fps = led_count * fps_headroom
        logger.info(f"    📈 Estimated max LEDs at 60 FPS: {max_leds_60fps:.0f}")


def main():
    """Run high-performance kernel test for 2600 LEDs."""
    logger.info("High-Performance CUDA Kernel Test for Production Prismatron")
    logger.info("Target: 2600 LEDs, 20+ GFLOPS, Real-time Performance")
    logger.info("=" * 70)

    # Create production-scale tensor
    tensor = create_production_tensor(led_count=2600)

    # Verify correctness
    if not verify_correctness(tensor):
        logger.error("Correctness check failed! Aborting performance test.")
        return 1

    # Benchmark all implementations
    results = benchmark_all_kernels(tensor)

    # Analyze GPU utilization
    analyze_gpu_utilization(results)

    # Final summary
    logger.info(f"\\n{'='*70}")
    logger.info("FINAL PERFORMANCE SUMMARY")
    logger.info(f"{'='*70}")

    if results.get("high_performance_available"):
        hp_gflops = results["high_performance_gflops"]
        hp_fps = results["high_performance_fps"]
        target_met = hp_gflops >= 20.0

        logger.info(f"High-Performance Kernel Results:")
        logger.info(f"  GFLOPS: {hp_gflops:.2f} {'✅' if target_met else '❌'}")
        logger.info(f"  FPS: {hp_fps:.1f}")
        logger.info(f"  Time per frame: {results['high_performance_time_ms']:.2f}ms")

        if results.get("corrected_available"):
            improvement = results["hp_vs_corrected"]
            logger.info(f"  Improvement over corrected: {improvement:.2f}x")

        if target_met:
            logger.info("\\n🎉 SUCCESS: 20+ GFLOPS target achieved!")
            logger.info("💡 Ready for production deployment with 2600 LEDs")
        else:
            logger.warning("\\n⚠️  Performance target not met")
            logger.info("💡 Consider further kernel optimization or hardware upgrade")
    else:
        logger.error("\\n❌ High-performance kernel not available")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
