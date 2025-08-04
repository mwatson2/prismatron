#!/usr/bin/env python3
"""
Demo script for batch frame optimizer.

This script demonstrates how to use the new batch frame optimizer
to process 8 or 16 frames simultaneously.
"""

# Import the batch optimizer
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from utils.batch_frame_optimizer import convert_ata_dia_to_dense, optimize_batch_frames_led_values
from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.frame_optimizer import load_ata_inverse_from_pattern, optimize_frame_led_values
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_test_matrices(pattern_path: str):
    """Load test matrices from pattern file."""
    print(f"Loading matrices from {pattern_path}...")

    # Load the pattern data
    data = np.load(pattern_path, allow_pickle=True)

    # Create SingleBlockMixedSparseTensor for AT matrix
    at_matrix = SingleBlockMixedSparseTensor(
        led_count=data["led_count"], frame_height=data["frame_height"], frame_width=data["frame_width"]
    )

    # Load diffusion matrix if available
    if "diffusion_matrix" in data:
        # Build from existing diffusion matrix
        diffusion_matrix = data["diffusion_matrix"]
        at_matrix.build_from_csc_matrix(diffusion_matrix)
    else:
        print("Warning: No diffusion matrix found in pattern file")
        return None, None, None

    # Create DiagonalATAMatrix
    ata_matrix = DiagonalATAMatrix(
        led_count=data["led_count"], frame_height=data["frame_height"], frame_width=data["frame_width"]
    )

    # Build ATA matrix from the same diffusion matrix
    ata_matrix.build_from_diffusion_matrix(diffusion_matrix)

    # Load ATA inverse
    ata_inverse = load_ata_inverse_from_pattern(pattern_path)
    if ata_inverse is None:
        print("Warning: No ATA inverse found in pattern file")
        return None, None, None

    return at_matrix, ata_matrix, ata_inverse


def create_test_frames(batch_size: int, frame_shape: tuple = (3, 480, 800)) -> np.ndarray:
    """Create test frames for batch processing."""
    print(f"Creating {batch_size} test frames with shape {frame_shape}...")

    # Create diverse test frames
    frames = []
    for i in range(batch_size):
        # Create a frame with different patterns
        frame = np.zeros(frame_shape, dtype=np.uint8)

        # Add some simple patterns
        if i % 4 == 0:
            # Red gradient
            frame[0] = np.linspace(0, 255, frame_shape[1] * frame_shape[2]).reshape(frame_shape[1], frame_shape[2])
        elif i % 4 == 1:
            # Green checkerboard
            frame[1] = ((np.arange(frame_shape[1])[:, None] + np.arange(frame_shape[2])) % 2) * 255
        elif i % 4 == 2:
            # Blue circle
            center_y, center_x = frame_shape[1] // 2, frame_shape[2] // 2
            y, x = np.ogrid[: frame_shape[1], : frame_shape[2]]
            mask = ((y - center_y) ** 2 + (x - center_x) ** 2) < (min(frame_shape[1], frame_shape[2]) // 4) ** 2
            frame[2, mask] = 255
        else:
            # Random noise
            frame = np.random.randint(0, 255, frame_shape, dtype=np.uint8)

        frames.append(frame)

    return np.stack(frames, axis=0)


def benchmark_batch_vs_single(at_matrix, ata_matrix, ata_inverse, batch_frames, max_iterations=5):
    """Benchmark batch optimizer vs single frame processing."""
    batch_size = batch_frames.shape[0]

    print(f"\nBenchmarking {batch_size} frames with {max_iterations} iterations...")

    # Benchmark single frame processing
    print("Running single frame optimization...")
    single_start = time.time()
    single_results = []

    for i in range(batch_size):
        frame = batch_frames[i]
        result = optimize_frame_led_values(
            frame, at_matrix, ata_matrix, ata_inverse, max_iterations=max_iterations, debug=False
        )
        single_results.append(result)

    single_time = time.time() - single_start

    # Benchmark batch processing
    print("Running batch optimization...")
    batch_start = time.time()

    batch_result = optimize_batch_frames_led_values(
        batch_frames, at_matrix, ata_matrix, ata_inverse, max_iterations=max_iterations, debug=False, enable_timing=True
    )

    batch_time = time.time() - batch_start

    # Print results
    print("\nBenchmark Results:")
    print(f"Single frame processing: {single_time:.3f}s ({single_time/batch_size:.3f}s per frame)")
    print(f"Batch processing: {batch_time:.3f}s ({batch_time/batch_size:.3f}s per frame)")
    print(f"Speedup: {single_time/batch_time:.2f}x")

    # Print timing breakdown for batch processing
    if batch_result.timing_data:
        print("\nBatch processing timing breakdown:")
        total_time = sum(batch_result.timing_data.values())
        for section, time_val in batch_result.timing_data.items():
            percentage = (time_val / total_time) * 100
            print(f"  {section}: {time_val:.3f}s ({percentage:.1f}%)")

    # Verify results are similar
    print("\nResult verification:")
    max_diff = 0
    for i in range(batch_size):
        single_led = single_results[i].led_values
        batch_led = batch_result.led_values[i]
        diff = np.abs(single_led.astype(np.float32) - batch_led.astype(np.float32))
        frame_max_diff = np.max(diff)
        max_diff = max(max_diff, frame_max_diff)

    print(f"  Maximum LED value difference: {max_diff:.2f} (out of 255)")
    if max_diff < 5:
        print("  ✓ Results are consistent between single and batch processing")
    else:
        print("  ⚠ Results have significant differences - check implementation")

    return batch_result


def main():
    """Main demo function."""
    print("Batch Frame Optimizer Demo")
    print("=" * 50)

    # Look for pattern files
    pattern_files = list(Path(".").glob("*patterns*.npz"))
    if not pattern_files:
        print("No pattern files found. Please ensure you have diffusion pattern files (.npz) in the current directory.")
        print("Pattern files should contain: diffusion_matrix, ata_inverse, led_count, frame_height, frame_width")
        return

    pattern_path = str(pattern_files[0])
    print(f"Using pattern file: {pattern_path}")

    # Load matrices
    at_matrix, ata_matrix, ata_inverse = load_test_matrices(pattern_path)
    if at_matrix is None:
        print("Failed to load matrices. Exiting.")
        return

    led_count = ata_inverse.shape[1]
    print(f"Loaded matrices for {led_count} LEDs")

    # Test both batch sizes
    for batch_size in [8, 16]:
        print("\n" + "=" * 50)
        print(f"Testing batch size: {batch_size}")
        print("=" * 50)

        # Create test frames
        test_frames = create_test_frames(batch_size)

        # Run benchmark
        batch_result = benchmark_batch_vs_single(at_matrix, ata_matrix, ata_inverse, test_frames, max_iterations=5)

        # Test DIA to dense conversion
        print("\nTesting ATA DIA to dense conversion...")
        dense_start = time.time()
        ata_dense = convert_ata_dia_to_dense(ata_matrix)
        dense_time = time.time() - dense_start

        print(f"ATA dense conversion: {dense_time:.3f}s")
        print(f"Dense ATA shape: {ata_dense.shape}")
        print(f"Dense ATA memory: {ata_dense.nbytes / 1024 / 1024:.1f} MB")

        # Verify the conversion is correct by comparing a few matrix-vector products
        print("Verifying DIA to dense conversion...")
        test_vector = np.random.random((3, led_count)).astype(np.float32)

        # Use DIA matrix
        import cupy as cp

        test_vector_gpu = cp.asarray(test_vector)
        dia_result = ata_matrix.multiply_3d(test_vector_gpu)
        dia_result = cp.asnumpy(dia_result)

        # Use dense matrix
        ata_dense_gpu = cp.asarray(ata_dense)
        test_vector_gpu = cp.asarray(test_vector)
        dense_result = cp.einsum("ijk,ik->ij", ata_dense_gpu, test_vector_gpu)
        dense_result = cp.asnumpy(dense_result)

        max_diff = np.max(np.abs(dia_result - dense_result))
        print(f"Maximum difference between DIA and dense: {max_diff:.2e}")

        if max_diff < 1e-5:
            print("✓ DIA to dense conversion is correct")
        else:
            print("⚠ DIA to dense conversion has significant errors")


if __name__ == "__main__":
    main()
