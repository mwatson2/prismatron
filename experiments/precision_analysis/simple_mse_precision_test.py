#!/usr/bin/env python3

import sys

sys.path.append("/mnt/dev/prismatron/src")

import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import lsqr


def create_synthetic_problem(n_pixels=1000, n_leds=200, sparsity=0.1):
    """Create a synthetic sparse LED optimization problem."""
    np.random.seed(42)

    # Create sparse A matrix (n_pixels x n_leds)
    A = np.random.rand(n_pixels, n_leds).astype(np.float32)
    A[np.random.rand(n_pixels, n_leds) > sparsity] = 0  # Make sparse

    # Create target vector b
    b = np.random.rand(n_pixels).astype(np.float32)

    # Compute ATA matrix
    ATA = A.T @ A
    ATb = A.T @ b

    return A, ATA, ATb, b


def test_lsqr_convergence(A, ATA, ATb, b, test_name, a_dtype=np.float32, ata_dtype=np.float32, max_iter=20):
    """Test LSQR convergence with different precisions."""
    print(f"\nTesting {test_name}")
    print(f"A dtype: {a_dtype}, ATA dtype: {ata_dtype}")

    # Convert matrices to specified precisions
    A_test = A.astype(a_dtype)
    ATA_test = ATA.astype(ata_dtype)
    ATb_test = ATb.astype(np.float32)  # Keep ATb as fp32

    # Fixed initialization
    x0 = np.full(A.shape[1], 0.5, dtype=np.float32)

    mse_history = []

    for i in range(1, max_iter + 1):
        start_time = time.time()

        # Run LSQR with current iteration count
        x, istop, itn, r1norm = lsqr(A_test, b, x0=x0.copy(), iter_lim=i)[:4]

        elapsed = time.time() - start_time

        # Calculate MSE
        residual = A_test @ x - b
        mse = np.mean(residual**2)
        mse_history.append(mse)

        print(f"  Iter {i:2d}: MSE = {mse:.6f}, Converged = {istop==1}, Time = {elapsed:.3f}s")

    return mse_history


def main():
    """Run MSE convergence comparison for different precision combinations."""

    # Create synthetic problem
    A, ATA, ATb, b = create_synthetic_problem(n_pixels=1000, n_leds=200)

    # Test configurations
    configs = [
        ("fp32 A + fp32 ATA", np.float32, np.float32),
        ("uint8 A + fp32 ATA", np.uint8, np.float32),
        ("fp32 A + fp16 ATA", np.float32, np.float16),
        ("uint8 A + fp16 ATA", np.uint8, np.float16),
    ]

    results = {}

    for test_name, a_dtype, ata_dtype in configs:
        try:
            mse_history = test_lsqr_convergence(A, ATA, ATb, b, test_name, a_dtype, ata_dtype)
            results[test_name] = mse_history
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
    plt.title("MSE Convergence: Matrix Precision Effects (Fixed Init = 0.5)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # Save plot
    output_file = "simple_mse_precision_convergence.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nPlot saved as: {output_file}")

    # Print summary
    print("\nFinal MSE Values (after 20 iterations):")
    for test_name, mse_history in results.items():
        if mse_history is not None and len(mse_history) > 0:
            final_mse = mse_history[-1]
            initial_mse = mse_history[0]
            print(f"  {test_name:20s}: Initial = {initial_mse:.6f}, Final = {final_mse:.6f}")
            if final_mse > initial_mse:
                print(f"    WARNING: MSE increased by {(final_mse/initial_mse - 1)*100:.1f}%")
            else:
                print(f"    MSE decreased by {(1 - final_mse/initial_mse)*100:.1f}%")


if __name__ == "__main__":
    main()
