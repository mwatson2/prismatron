#!/usr/bin/env python3
"""
Test script to measure LED optimization performance improvements.
"""

import time

import numpy as np

from src.const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from src.consumer.led_optimizer import LEDOptimizer


def create_test_patterns(led_count=100, sparse=True):
    """Create small test patterns for performance testing."""
    patterns_path = f"test_patterns_{led_count}.npz"

    if sparse:
        # Create sparse test matrix (384000 × led_count*3)
        from scipy import sparse

        # Create random sparse matrix with ~1% density
        pixels = FRAME_HEIGHT * FRAME_WIDTH
        cols = led_count * 3
        density = 0.01
        nnz = int(pixels * cols * density)

        # Random coordinates
        row_coords = np.random.randint(0, pixels, nnz)
        col_coords = np.random.randint(0, cols, nnz)
        data = np.random.random(nnz).astype(np.float32)

        # Create sparse matrix
        matrix = sparse.csc_matrix(
            (data, (row_coords, col_coords)), shape=(pixels, cols)
        )

        # Save in the expected format
        np.savez_compressed(
            patterns_path,
            matrix_data=matrix.data,
            matrix_indices=matrix.indices,
            matrix_indptr=matrix.indptr,
            matrix_shape=matrix.shape,
            led_spatial_mapping={i: i for i in range(led_count)},
            led_positions=np.random.random((led_count, 2)),
        )

        print(f"Created sparse test patterns: {patterns_path}")
        print(f"Matrix shape: {matrix.shape}, NNZ: {matrix.nnz:,}")
        return patterns_path

    return None


def test_optimization_performance():
    """Test the optimization performance with different LED counts."""

    # Test with small LED count first
    for led_count in [100, 500, 1000]:
        print(f"\n=== Testing with {led_count} LEDs ===")

        # Create test patterns
        patterns_path = create_test_patterns(led_count)

        # Initialize optimizer
        optimizer = LEDOptimizer(
            diffusion_patterns_path=patterns_path.replace(".npz", ""), use_gpu=True
        )

        if not optimizer.initialize():
            print(f"Failed to initialize optimizer for {led_count} LEDs")
            continue

        # Create test frame
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )

        # Warm up
        optimizer.optimize_frame(test_frame, max_iterations=5)

        # Time multiple runs
        times = []
        for i in range(3):
            start_time = time.time()
            result = optimizer.optimize_frame(test_frame, max_iterations=50)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        print(f"Average optimization time: {avg_time:.3f}s")
        print(f"Estimated FPS: {1.0/avg_time:.1f}")
        print(f"LED count: {result.led_values.shape[0]}")
        print(f"MSE: {result.error_metrics.get('mse', 'N/A'):.6f}")

        # Clean up
        import os

        if os.path.exists(patterns_path):
            os.remove(patterns_path)


if __name__ == "__main__":
    test_optimization_performance()
