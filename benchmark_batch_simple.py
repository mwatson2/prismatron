#!/usr/bin/env python3
"""
Simplified benchmarking script for batch frame optimizer.
Avoids complex imports by running directly from project root.
"""

import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cupy as cp
import numpy as np

# Direct imports to avoid circular dependencies
sys.path.insert(0, "src")

import logging

logging.basicConfig(level=logging.WARNING)  # Reduce noise

# Import modules directly to avoid __init__.py issues
import importlib.util

# Load batch_frame_optimizer
spec = importlib.util.spec_from_file_location("batch_frame_optimizer", "src/utils/batch_frame_optimizer.py")
batch_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch_optimizer)

# Load diagonal_ata_matrix
spec = importlib.util.spec_from_file_location("diagonal_ata_matrix", "src/utils/diagonal_ata_matrix.py")
dia_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dia_module)

# Load single_block_sparse_tensor
spec = importlib.util.spec_from_file_location("single_block_sparse_tensor", "src/utils/single_block_sparse_tensor.py")
tensor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tensor_module)

# Extract what we need
optimize_batch_frames_led_values = batch_optimizer.optimize_batch_frames_led_values
convert_ata_dia_to_dense = batch_optimizer.convert_ata_dia_to_dense
DiagonalATAMatrix = dia_module.DiagonalATAMatrix
SingleBlockMixedSparseTensor = tensor_module.SingleBlockMixedSparseTensor


def load_ata_inverse_simple(pattern_file_path: str):
    """Simple ATA inverse loader."""
    try:
        data = np.load(pattern_file_path, allow_pickle=True)
        if "ata_inverse" in data:
            return data["ata_inverse"]
        else:
            print(f"Warning: No ATA inverse found in {pattern_file_path}")
            return None
    except Exception as e:
        print(f"Warning: Could not load ATA inverse from {pattern_file_path}: {e}")
        return None


def optimize_frame_simple(target_frame, at_matrix, ata_matrix, ata_inverse, max_iterations=5):
    """Simplified single frame optimizer to avoid import issues."""
    # Validate input frame format and convert to planar int8 if needed
    if target_frame.dtype != np.int8 and target_frame.dtype != np.uint8:
        raise ValueError(f"Target frame must be int8 or uint8, got {target_frame.dtype}")

    # Handle both planar (3, H, W) and standard (H, W, 3) formats
    if target_frame.shape == (3, 480, 800) or target_frame.shape == (3, 640, 800):
        target_planar_uint8 = target_frame.astype(np.uint8)
    else:
        target_planar_uint8 = target_frame.astype(np.uint8).transpose(2, 0, 1)

    # Calculate A^T @ b
    ATb = at_matrix.multiply_planar_uint8(target_planar_uint8)

    # Ensure ATb is in (3, led_count) format
    if ATb.shape[0] != 3:
        ATb = ATb.T

    led_count = ATb.shape[1]

    # Initialize using ATA inverse
    ata_inverse_gpu = cp.asarray(ata_inverse)
    ATb_gpu = cp.asarray(ATb)
    led_values_gpu = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_gpu)
    led_values_gpu = cp.clip(led_values_gpu, 0.0, 1.0)

    ATb_gpu = cp.asarray(ATb)

    # Gradient descent optimization
    for iteration in range(max_iterations):
        ATA_x = ata_matrix.multiply_3d(led_values_gpu)
        gradient = ATA_x - ATb_gpu

        g_dot_g = cp.sum(gradient * gradient)
        g_dot_ATA_g_per_channel = ata_matrix.g_ata_g_3d(gradient)
        g_dot_ATA_g = cp.sum(g_dot_ATA_g_per_channel)

        if g_dot_ATA_g > 0:
            step_size = float(0.9 * g_dot_g / g_dot_ATA_g)
        else:
            step_size = 0.01

        led_values_gpu = cp.clip(led_values_gpu - step_size * gradient, 0, 1)

    # Convert back to CPU and scale
    led_values_output = (cp.asnumpy(led_values_gpu) * 255.0).astype(np.uint8)

    return led_values_output


def load_2624_patterns(pattern_path: str) -> Tuple:
    """Load 2624 LED pattern matrices."""
    print(f"Loading 2624 LED patterns from {pattern_path}...")

    data = np.load(pattern_path, allow_pickle=True)

    led_count = int(data["led_count"])
    frame_height = int(data["frame_height"])
    frame_width = int(data["frame_width"])

    print(f"Pattern info: {led_count} LEDs, {frame_height}x{frame_width} frame")

    # Create AT matrix
    at_matrix = SingleBlockMixedSparseTensor(led_count=led_count, frame_height=frame_height, frame_width=frame_width)

    if "diffusion_matrix" in data:
        diffusion_matrix = data["diffusion_matrix"]
        print(f"Diffusion matrix shape: {diffusion_matrix.shape}")
        at_matrix.build_from_csc_matrix(diffusion_matrix)
    else:
        raise ValueError("No diffusion matrix found in pattern file")

    # Create ATA matrix
    ata_matrix = DiagonalATAMatrix(led_count=led_count, frame_height=frame_height, frame_width=frame_width)

    print("Building ATA matrix...")
    ata_matrix.build_from_diffusion_matrix(diffusion_matrix)

    # Load ATA inverse
    ata_inverse = load_ata_inverse_simple(pattern_path)
    if ata_inverse is None:
        raise ValueError("No ATA inverse found in pattern file")

    print(f"ATA inverse shape: {ata_inverse.shape}")

    return at_matrix, ata_matrix, ata_inverse, led_count, frame_height, frame_width


def create_test_frames(batch_size: int, frame_height: int, frame_width: int) -> np.ndarray:
    """Create test frames."""
    print(f"Creating {batch_size} test frames ({frame_height}x{frame_width})...")

    frames = []
    for i in range(batch_size):
        frame = np.zeros((3, frame_height, frame_width), dtype=np.uint8)

        if i % 4 == 0:
            # Gradient
            x = np.linspace(0, 255, frame_width)
            frame[0] = x
            frame[1] = x[::-1]
            frame[2] = 128
        elif i % 4 == 1:
            # Sinusoidal
            x = np.linspace(0, 4 * np.pi, frame_width)
            y = np.linspace(0, 4 * np.pi, frame_height)
            X, Y = np.meshgrid(x, y)
            frame[0] = ((np.sin(X) + 1) * 127).astype(np.uint8)
            frame[1] = ((np.cos(Y) + 1) * 127).astype(np.uint8)
            frame[2] = ((np.sin(X + Y) + 1) * 127).astype(np.uint8)
        elif i % 4 == 2:
            # Circle
            center_y, center_x = frame_height // 2, frame_width // 2
            y, x = np.ogrid[:frame_height, :frame_width]
            mask = ((y - center_y) ** 2 + (x - center_x) ** 2) < (min(frame_height, frame_width) // 4) ** 2
            frame[0, mask] = 255
            frame[1, mask] = 128
            frame[2, mask] = 64
        else:
            # Random
            frame = np.random.randint(0, 255, (3, frame_height, frame_width), dtype=np.uint8)

        frames.append(frame)

    return np.stack(frames, axis=0)


def warmup_gpu():
    """Warmup GPU."""
    print("Warming up GPU...")
    for _ in range(5):
        a = cp.random.random((1000, 1000))
        b = cp.random.random((1000, 1000))
        c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()

    del a, b, c
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


def benchmark_methods(at_matrix, ata_matrix, ata_inverse, ata_dense, test_frames, max_iterations=5):
    """Benchmark both methods with proper timing."""
    batch_size = test_frames.shape[0]
    num_trials = 5
    num_warmup = 2

    print(f"Benchmarking {batch_size} frames, {max_iterations} iterations...")

    # Warmup runs
    print("Warming up...")
    for i in range(num_warmup):
        # Single frame warmup
        frame = test_frames[i % batch_size]
        result = optimize_frame_simple(frame, at_matrix, ata_matrix, ata_inverse, max_iterations)
        del result

        # Batch warmup
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

    # Benchmark single frame processing
    print("Benchmarking single frame processing...")
    single_times = []

    for trial in range(num_trials):
        start_time = time.time()

        for i in range(batch_size):
            frame = test_frames[i]
            result = optimize_frame_simple(frame, at_matrix, ata_matrix, ata_inverse, max_iterations)
            del result

        end_time = time.time()
        single_times.append(end_time - start_time)
        gc.collect()

    # Benchmark batch processing
    print("Benchmarking batch processing...")
    batch_times = []
    batch_optimization_times = []

    for trial in range(num_trials):
        start_time = time.time()

        result = optimize_batch_frames_led_values(
            test_frames,
            at_matrix,
            ata_matrix,
            ata_inverse,
            max_iterations=max_iterations,
            debug=False,
            enable_timing=True,
        )

        end_time = time.time()
        batch_times.append(end_time - start_time)

        # Extract optimization time (excluding ATA conversion)
        if result.timing_data and "ata_dense_conversion" in result.timing_data:
            optimization_time = sum(result.timing_data.values()) - result.timing_data["ata_dense_conversion"]
            batch_optimization_times.append(optimization_time)
        else:
            batch_optimization_times.append(end_time - start_time)

        del result
        gc.collect()

    # Calculate statistics
    single_mean = np.mean(single_times)
    single_std = np.std(single_times)
    batch_mean = np.mean(batch_times)
    batch_std = np.std(batch_times)
    batch_opt_mean = np.mean(batch_optimization_times)
    batch_opt_std = np.std(batch_optimization_times)

    return {
        "single_total": single_mean,
        "single_per_frame": single_mean / batch_size,
        "single_std": single_std,
        "batch_total": batch_mean,
        "batch_per_frame": batch_mean / batch_size,
        "batch_std": batch_std,
        "batch_opt_total": batch_opt_mean,
        "batch_opt_per_frame": batch_opt_mean / batch_size,
        "batch_opt_std": batch_opt_std,
        "speedup": single_mean / batch_opt_mean,
        "fps_single": batch_size / single_mean,
        "fps_batch": batch_size / batch_opt_mean,
    }


def main():
    """Run the benchmark."""
    print("Simplified Batch Frame Optimizer Benchmark")
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
        print("No 2624 LED pattern file found.")
        return

    print(f"Using pattern file: {pattern_path}")

    # Load matrices
    try:
        at_matrix, ata_matrix, ata_inverse, led_count, frame_height, frame_width = load_2624_patterns(pattern_path)
    except Exception as e:
        print(f"Error loading patterns: {e}")
        return

    # One-time ATA conversion
    print(f"\nPerforming one-time ATA DIA to dense conversion...")
    conversion_start = time.time()
    ata_dense = convert_ata_dia_to_dense(ata_matrix)
    conversion_time = time.time() - conversion_start
    print(f"ATA conversion time: {conversion_time:.3f}s")
    print(f"Dense ATA memory: {ata_dense.nbytes / 1024 / 1024:.1f} MB")

    # Warmup GPU
    warmup_gpu()

    # Test both batch sizes
    for batch_size in [8, 16]:
        print(f"\n" + "=" * 50)
        print(f"BATCH SIZE: {batch_size} frames")
        print("=" * 50)

        # Create test frames
        test_frames = create_test_frames(batch_size, frame_height, frame_width)

        # Run benchmark
        results = benchmark_methods(at_matrix, ata_matrix, ata_inverse, ata_dense, test_frames, max_iterations=5)

        # Print results
        print(f"\nResults:")
        print(f"Single Frame Processing:")
        print(f"  Total time: {results['single_total']:.3f} ± {results['single_std']:.3f}s")
        print(f"  Per frame: {results['single_per_frame']:.3f}s")
        print(f"  FPS: {results['fps_single']:.1f}")

        print(f"\nBatch Processing (excluding ATA conversion):")
        print(f"  Total time: {results['batch_opt_total']:.3f} ± {results['batch_opt_std']:.3f}s")
        print(f"  Per frame: {results['batch_opt_per_frame']:.3f}s")
        print(f"  FPS: {results['fps_batch']:.1f}")
        print(f"  Speedup: {results['speedup']:.2f}x")

        print(f"\nPer-LED Performance:")
        print(f"  Single: {results['single_per_frame'] * 1000 / led_count:.3f} ms per LED")
        print(f"  Batch: {results['batch_opt_per_frame'] * 1000 / led_count:.3f} ms per LED")

        del test_frames
        gc.collect()

    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pattern: {led_count} LEDs, {frame_height}x{frame_width} frames")
    print(f"ATA conversion: {conversion_time:.3f}s (one-time cost)")
    print(f"Dense ATA memory: {ata_dense.nbytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
