#!/usr/bin/env python3
"""
Comprehensive benchmarking script for batch frame optimizer.

This script runs proper benchmarks with warmup runs and excludes one-off
initialization costs like ATA conversion from timing measurements.
"""

import gc

# Add src to path
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cupy as cp
import numpy as np

sys.path.append("src")

# Import the batch optimizer
from utils.batch_frame_optimizer import convert_ata_dia_to_dense, optimize_batch_frames_led_values
from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.frame_optimizer import load_ata_inverse_from_pattern, optimize_frame_led_values
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_2624_patterns(pattern_path: str) -> Tuple:
    """Load 2624 LED pattern matrices."""
    print(f"Loading 2624 LED patterns from {pattern_path}...")

    # Load the pattern data
    data = np.load(pattern_path, allow_pickle=True)

    led_count = int(data["led_count"])
    frame_height = int(data["frame_height"])
    frame_width = int(data["frame_width"])

    print(f"Pattern info: {led_count} LEDs, {frame_height}x{frame_width} frame")

    # Create SingleBlockMixedSparseTensor for AT matrix
    at_matrix = SingleBlockMixedSparseTensor(led_count=led_count, frame_height=frame_height, frame_width=frame_width)

    # Load diffusion matrix
    if "diffusion_matrix" in data:
        diffusion_matrix = data["diffusion_matrix"]
        print(f"Diffusion matrix shape: {diffusion_matrix.shape}")
        at_matrix.build_from_csc_matrix(diffusion_matrix)
    else:
        raise ValueError("No diffusion matrix found in pattern file")

    # Create DiagonalATAMatrix
    ata_matrix = DiagonalATAMatrix(led_count=led_count, frame_height=frame_height, frame_width=frame_width)

    # Build ATA matrix from the same diffusion matrix
    print("Building ATA matrix from diffusion matrix...")
    ata_matrix.build_from_diffusion_matrix(diffusion_matrix)

    # Load ATA inverse
    ata_inverse = load_ata_inverse_from_pattern(pattern_path)
    if ata_inverse is None:
        raise ValueError("No ATA inverse found in pattern file")

    print(f"ATA inverse shape: {ata_inverse.shape}")

    return at_matrix, ata_matrix, ata_inverse, led_count, frame_height, frame_width


def create_realistic_test_frames(batch_size: int, frame_height: int, frame_width: int) -> np.ndarray:
    """Create realistic test frames for benchmarking."""
    print(f"Creating {batch_size} realistic test frames ({frame_height}x{frame_width})...")

    frames = []
    for i in range(batch_size):
        # Create realistic frame patterns
        frame = np.zeros((3, frame_height, frame_width), dtype=np.uint8)

        # Add realistic content patterns
        if i % 5 == 0:
            # Smooth gradient
            x = np.linspace(0, 1, frame_width)
            y = np.linspace(0, 1, frame_height)
            X, Y = np.meshgrid(x, y)
            frame[0] = (X * 255).astype(np.uint8)
            frame[1] = (Y * 255).astype(np.uint8)
            frame[2] = ((X + Y) * 127).astype(np.uint8)
        elif i % 5 == 1:
            # Sinusoidal pattern
            x = np.linspace(0, 4 * np.pi, frame_width)
            y = np.linspace(0, 4 * np.pi, frame_height)
            X, Y = np.meshgrid(x, y)
            frame[0] = ((np.sin(X) * np.cos(Y) + 1) * 127).astype(np.uint8)
            frame[1] = ((np.cos(X) * np.sin(Y) + 1) * 127).astype(np.uint8)
            frame[2] = ((np.sin(X + Y) + 1) * 127).astype(np.uint8)
        elif i % 5 == 2:
            # Circular patterns
            center_y, center_x = frame_height // 2, frame_width // 2
            y, x = np.ogrid[:frame_height, :frame_width]
            for c in range(3):
                radius = (c + 1) * min(frame_height, frame_width) // 6
                mask = ((y - center_y) ** 2 + (x - center_x) ** 2) < radius**2
                frame[c, mask] = 255
        elif i % 5 == 3:
            # Noise with structure
            base = np.random.randint(0, 128, (3, frame_height, frame_width), dtype=np.uint8)
            # Add some structure
            structure = np.random.randint(0, 128, (3, frame_height // 4, frame_width // 4), dtype=np.uint8)
            structure = np.repeat(np.repeat(structure, 4, axis=1), 4, axis=2)
            frame = np.clip(base + structure[:, :frame_height, :frame_width], 0, 255).astype(np.uint8)
        else:
            # Random but with some correlation
            frame = np.random.randint(0, 255, (3, frame_height, frame_width), dtype=np.uint8)

        frames.append(frame)

    return np.stack(frames, axis=0)


def warmup_gpu():
    """Warmup GPU to ensure stable timing."""
    print("Warming up GPU...")

    # Perform some GPU operations to warm up
    for _ in range(10):
        a = cp.random.random((1000, 1000))
        b = cp.random.random((1000, 1000))
        c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()

    # Clear GPU memory
    del a, b, c
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


def benchmark_single_frame_processing(
    at_matrix, ata_matrix, ata_inverse, test_frames, max_iterations: int = 5, num_trials: int = 10, num_warmup: int = 3
) -> Dict[str, float]:
    """Benchmark single frame processing with proper warmup."""
    batch_size = test_frames.shape[0]

    print(f"Benchmarking single frame processing ({num_trials} trials, {num_warmup} warmup)...")

    # Warmup runs
    for i in range(num_warmup):
        frame = test_frames[i % batch_size]
        result = optimize_frame_led_values(
            frame, at_matrix, ata_matrix, ata_inverse, max_iterations=max_iterations, debug=False
        )
        del result
        gc.collect()

    # Timed runs
    times = []
    for trial in range(num_trials):
        trial_start = time.time()

        for i in range(batch_size):
            frame = test_frames[i]
            result = optimize_frame_led_values(
                frame, at_matrix, ata_matrix, ata_inverse, max_iterations=max_iterations, debug=False
            )
            del result

        trial_time = time.time() - trial_start
        times.append(trial_time)

        # Clear memory between trials
        gc.collect()

    # Calculate statistics
    times = np.array(times)
    stats = {
        "total_time_mean": np.mean(times),
        "total_time_std": np.std(times),
        "total_time_min": np.min(times),
        "per_frame_mean": np.mean(times) / batch_size,
        "per_frame_std": np.std(times) / batch_size,
        "per_frame_min": np.min(times) / batch_size,
        "frames_per_second": batch_size / np.mean(times),
    }

    return stats


def benchmark_batch_processing(
    at_matrix,
    ata_matrix,
    ata_inverse,
    ata_dense,
    test_frames,
    max_iterations: int = 5,
    num_trials: int = 10,
    num_warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark batch processing with proper warmup, excluding ATA conversion."""
    batch_size = test_frames.shape[0]

    print(f"Benchmarking batch processing ({num_trials} trials, {num_warmup} warmup)...")

    # Warmup runs
    for i in range(num_warmup):
        result = optimize_batch_frames_led_values(
            test_frames,
            at_matrix,
            ata_matrix,
            ata_inverse,
            max_iterations=max_iterations,
            debug=False,
            enable_timing=False,
        )
        del result
        gc.collect()

    # Timed runs
    times = []
    detailed_timings = []

    for trial in range(num_trials):
        trial_start = time.time()

        result = optimize_batch_frames_led_values(
            test_frames,
            at_matrix,
            ata_matrix,
            ata_inverse,
            max_iterations=max_iterations,
            debug=False,
            enable_timing=True,
        )

        trial_time = time.time() - trial_start
        times.append(trial_time)

        # Extract detailed timing (excluding ATA conversion)
        if result.timing_data:
            # Remove ATA conversion time from the total
            timing_without_conversion = result.timing_data.copy()
            conversion_time = timing_without_conversion.pop("ata_dense_conversion", 0)
            detailed_timings.append(
                {
                    "total_with_conversion": sum(result.timing_data.values()),
                    "total_without_conversion": sum(timing_without_conversion.values()),
                    "conversion_time": conversion_time,
                    "breakdown": timing_without_conversion,
                }
            )

        del result
        gc.collect()

    # Calculate statistics
    times = np.array(times)

    # Calculate timing without ATA conversion
    times_without_conversion = np.array([t["total_without_conversion"] for t in detailed_timings])

    stats = {
        "total_time_mean": np.mean(times),
        "total_time_std": np.std(times),
        "total_time_min": np.min(times),
        "optimization_time_mean": np.mean(times_without_conversion),
        "optimization_time_std": np.std(times_without_conversion),
        "optimization_time_min": np.min(times_without_conversion),
        "per_frame_mean": np.mean(times_without_conversion) / batch_size,
        "per_frame_std": np.std(times_without_conversion) / batch_size,
        "per_frame_min": np.min(times_without_conversion) / batch_size,
        "frames_per_second": batch_size / np.mean(times_without_conversion),
        "conversion_time_mean": np.mean([t["conversion_time"] for t in detailed_timings]),
        "detailed_timings": detailed_timings,
    }

    return stats


def run_comprehensive_benchmark():
    """Run comprehensive benchmark with 2624 LED patterns."""
    print("Comprehensive Batch Frame Optimizer Benchmark")
    print("=" * 60)

    # Find pattern file
    pattern_candidates = [
        "patterns_2624_fp16.npz",
        "diffusion_patterns/synthetic_2624_fp16_64x64.npz",
        "diffusion_patterns/synthetic_2624_fp16.npz",
    ]

    pattern_path = None
    for candidate in pattern_candidates:
        if Path(candidate).exists():
            pattern_path = candidate
            break

    if not pattern_path:
        print("No 2624 LED pattern file found. Looking for any pattern file...")
        pattern_files = list(Path(".").glob("*patterns*.npz"))
        if pattern_files:
            pattern_path = str(pattern_files[0])
        else:
            print("No pattern files found. Exiting.")
            return

    print(f"Using pattern file: {pattern_path}")

    # Load matrices
    at_matrix, ata_matrix, ata_inverse, led_count, frame_height, frame_width = load_2624_patterns(pattern_path)

    # One-time ATA conversion (excluded from timing)
    print("\nPerforming one-time ATA DIA to dense conversion...")
    conversion_start = time.time()
    ata_dense = convert_ata_dia_to_dense(ata_matrix)
    conversion_time = time.time() - conversion_start
    print(f"ATA conversion time: {conversion_time:.3f}s")
    print(f"Dense ATA memory: {ata_dense.nbytes / 1024 / 1024:.1f} MB")

    # Warmup GPU
    warmup_gpu()

    # Test parameters
    test_params = {"max_iterations": 5, "num_trials": 10, "num_warmup": 3}

    # Test both batch sizes
    results = {}

    for batch_size in [8, 16]:
        print("\n" + "=" * 60)
        print(f"BATCH SIZE: {batch_size} frames")
        print("=" * 60)

        # Create test frames
        test_frames = create_realistic_test_frames(batch_size, frame_height, frame_width)

        # Benchmark single frame processing
        single_stats = benchmark_single_frame_processing(at_matrix, ata_matrix, ata_inverse, test_frames, **test_params)

        # Benchmark batch processing
        batch_stats = benchmark_batch_processing(
            at_matrix, ata_matrix, ata_inverse, ata_dense, test_frames, **test_params
        )

        # Store results
        results[batch_size] = {"single": single_stats, "batch": batch_stats}

        # Print results
        print("\nSingle Frame Processing Results:")
        print(f"  Total time: {single_stats['total_time_mean']:.3f} ± {single_stats['total_time_std']:.3f}s")
        print(f"  Per frame: {single_stats['per_frame_mean']:.3f} ± {single_stats['per_frame_std']:.3f}s")
        print(f"  FPS: {single_stats['frames_per_second']:.1f}")

        print("\nBatch Processing Results (excluding ATA conversion):")
        print(
            f"  Total time: {batch_stats['optimization_time_mean']:.3f} ± {batch_stats['optimization_time_std']:.3f}s"
        )
        print(f"  Per frame: {batch_stats['per_frame_mean']:.3f} ± {batch_stats['per_frame_std']:.3f}s")
        print(f"  FPS: {batch_stats['frames_per_second']:.1f}")
        print(f"  ATA conversion: {batch_stats['conversion_time_mean']:.3f}s (one-time cost)")

        # Calculate speedup
        speedup = single_stats["per_frame_mean"] / batch_stats["per_frame_mean"]
        print(f"\nSpeedup: {speedup:.2f}x")

        # Print detailed timing breakdown
        if batch_stats["detailed_timings"]:
            avg_timing = {}
            for section in batch_stats["detailed_timings"][0]["breakdown"]:
                section_times = [t["breakdown"][section] for t in batch_stats["detailed_timings"]]
                avg_timing[section] = np.mean(section_times)

            print("\nDetailed Timing Breakdown (average):")
            total_optimization = sum(avg_timing.values())
            for section, time_val in sorted(avg_timing.items(), key=lambda x: x[1], reverse=True):
                percentage = (time_val / total_optimization) * 100
                per_frame = time_val / batch_size
                print(f"  {section}: {time_val:.3f}s ({percentage:.1f}%, {per_frame:.3f}s per frame)")

        # Memory analysis
        print("\nMemory Analysis:")
        print(f"  LED count: {led_count}")
        print(f"  Frame size: {frame_height}x{frame_width}")
        print(f"  Batch frames memory: {test_frames.nbytes / 1024 / 1024:.1f} MB")
        print(f"  ATA dense memory: {ata_dense.nbytes / 1024 / 1024:.1f} MB")

        # Performance per LED
        print("\nPer-LED Performance:")
        print(f"  Single: {single_stats['per_frame_mean'] * 1000 / led_count:.3f} ms per LED")
        print(f"  Batch: {batch_stats['per_frame_mean'] * 1000 / led_count:.3f} ms per LED")

        # Cleanup
        del test_frames
        gc.collect()

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)

    print(f"Pattern: {led_count} LEDs, {frame_height}x{frame_width} frames")
    print(f"Iterations: {test_params['max_iterations']}")
    print(f"Trials: {test_params['num_trials']}")

    print("\nPer-Frame Performance (seconds):")
    print(f"{'Method':<20} {'8-frame':<12} {'16-frame':<12} {'Speedup 8x':<12} {'Speedup 16x':<12}")
    print("-" * 68)

    single_8 = results[8]["single"]["per_frame_mean"]
    single_16 = results[16]["single"]["per_frame_mean"]
    batch_8 = results[8]["batch"]["per_frame_mean"]
    batch_16 = results[16]["batch"]["per_frame_mean"]

    print(f"{'Single Frame':<20} {single_8:.4f}      {single_16:.4f}      {'-':<12} {'-':<12}")
    print(
        f"{'Batch Processing':<20} {batch_8:.4f}      {batch_16:.4f}      {single_8/batch_8:.2f}x        {single_16/batch_16:.2f}x"
    )

    print("\nFrames Per Second:")
    print(f"{'Method':<20} {'8-frame':<12} {'16-frame':<12}")
    print("-" * 44)
    print(
        f"{'Single Frame':<20} {results[8]['single']['frames_per_second']:.1f}         {results[16]['single']['frames_per_second']:.1f}"
    )
    print(
        f"{'Batch Processing':<20} {results[8]['batch']['frames_per_second']:.1f}         {results[16]['batch']['frames_per_second']:.1f}"
    )

    print("\nKey Insights:")
    print(f"  - ATA conversion is one-time cost: {conversion_time:.3f}s")
    print(f"  - Batch processing achieves {single_8/batch_8:.1f}x speedup on 8 frames")
    print(f"  - Batch processing achieves {single_16/batch_16:.1f}x speedup on 16 frames")
    print(f"  - Memory overhead: {ata_dense.nbytes / 1024 / 1024:.1f} MB for dense ATA")


if __name__ == "__main__":
    run_comprehensive_benchmark()
