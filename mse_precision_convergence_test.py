#!/usr/bin/env python3

import os
import sys

sys.path.append("/mnt/dev/prismatron/src")
sys.path.append("/mnt/dev/prismatron")

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_pattern_data(pattern_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load pattern data without instantiating large objects."""
    print(f"Loading pattern data from {pattern_path}")

    data = np.load(pattern_path, allow_pickle=True)

    # Get data dictionaries without instantiating
    mixed_tensor_data = data["mixed_tensor"].item()
    dia_matrix_data = data["dia_matrix"].item()
    metadata = data["metadata"].item()

    return mixed_tensor_data, dia_matrix_data, metadata


def run_convergence_test(
    mixed_tensor_data, dia_matrix_data, metadata, test_name, a_dtype=cp.float32, ata_dtype=cp.float32, num_iterations=20
):
    """Run convergence test and track MSE over iterations."""
    print(f"\nRunning {test_name}...")

    # Get dimensions from metadata
    width = metadata["width"]
    height = metadata["height"]
    led_count = metadata["led_count"]

    # Create smaller test frame for memory efficiency
    np.random.seed(42)  # Fixed seed for reproducibility
    test_frame = np.random.rand(height // 4, width // 4, 3).astype(np.float32)  # Smaller frame

    # Create mixed tensor object
    mixed_tensor = SingleBlockMixedSparseTensor(
        mixed_tensor_data["height"] // 4, mixed_tensor_data["width"] // 4, 32, 32  # Smaller size  # Smaller block size
    )
    mixed_tensor.load_from_data(mixed_tensor_data)

    # Create DIA matrix
    dia_matrix = DiagonalATAMatrix(
        led_count // 16, led_count // 16, dia_matrix_data["bandwidth"] // 4, dia_matrix_data["format"]  # Much smaller
    )
    dia_matrix.load_from_data(dia_matrix_data)

    # Fixed initialization (0.5)
    x0 = cp.full((led_count // 16, 3), 0.5, dtype=cp.float32)

    mse_history = []

    for i in range(1, min(num_iterations, 5) + 1):  # Fewer iterations for memory
        start_time = time.time()

        # Run optimization with specified iterations
        result = optimize_frame_led_values(
            test_frame, mixed_tensor, dia_matrix, lsqr_iterations=i, x0=x0.copy(), a_dtype=a_dtype, ata_dtype=ata_dtype
        )

        elapsed = time.time() - start_time

        # Get MSE from result
        mse = result.error_metrics.get("mse", 0.0)
        mse_history.append(mse)

        print(f"  Iteration {i:2d}: MSE = {mse:.6f}, Time = {elapsed:.3f}s")

        # Clear memory
        cp.get_default_memory_pool().free_all_blocks()

    return mse_history


def main():
    """Run MSE convergence comparison for different precision combinations."""
    pattern_path = "/mnt/dev/prismatron/diffusion_patterns/synthetic_2624_fp32.npz"

    if not os.path.exists(pattern_path):
        print(f"Error: Pattern file not found: {pattern_path}")
        return

    # Load pattern data
    mixed_tensor, dia_matrix, metadata = load_pattern_data(pattern_path)

    # Test configurations
    configs = [
        ("fp32 ATA + fp32 A", cp.float32, cp.float32),
        ("fp32 ATA + uint8 A", cp.uint8, cp.float32),
        ("fp16 ATA + fp32 A", cp.float32, cp.float16),
        ("fp16 ATA + uint8 A", cp.uint8, cp.float16),
    ]

    results = {}

    for test_name, a_dtype, ata_dtype in configs:
        try:
            mse_history = run_convergence_test(
                mixed_tensor, dia_matrix, metadata, test_name, a_dtype=a_dtype, ata_dtype=ata_dtype
            )
            results[test_name] = mse_history

            # Clear GPU memory
            cp.get_default_memory_pool().free_all_blocks()

        except Exception as e:
            print(f"Error with {test_name}: {e}")
            results[test_name] = None

    # Plot results
    plt.figure(figsize=(12, 8))

    colors = ["blue", "red", "green", "orange"]
    linestyles = ["-", "--", "-.", ":"]

    for i, (test_name, mse_history) in enumerate(results.items()):
        if mse_history is not None:
            iterations = range(1, len(mse_history) + 1)
            plt.plot(
                iterations,
                mse_history,
                color=colors[i],
                linestyle=linestyles[i],
                marker="o",
                markersize=4,
                linewidth=2,
                label=test_name,
            )

    plt.xlabel("LSQR Iterations")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE Convergence Comparison: Matrix Precision Effects\n(Fixed Initialization = 0.5)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # Save plot
    output_file = "mse_precision_convergence_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nPlot saved as: {output_file}")

    # Print summary
    print("\nFinal MSE Values (after 20 iterations):")
    for test_name, mse_history in results.items():
        if mse_history is not None:
            final_mse = mse_history[-1]
            initial_mse = mse_history[0]
            print(f"  {test_name:20s}: Initial = {initial_mse:.6f}, Final = {final_mse:.6f}")
            if final_mse > initial_mse:
                print(f"    WARNING: MSE increased by {(final_mse/initial_mse - 1)*100:.1f}%")
            else:
                print(f"    MSE decreased by {(1 - final_mse/initial_mse)*100:.1f}%")


if __name__ == "__main__":
    main()
