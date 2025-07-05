#!/usr/bin/env python3
"""
Test CSC vs DIA matrix performance for A^T A operations.

Since the DIA matrix has 979 diagonals (almost dense), CSC might be faster.
"""

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix


def test_csc_vs_dia_performance():
    """Compare CSC vs DIA matrix performance for A^T A operations."""

    print("=== CSC vs DIA A^T A Performance Test ===")

    # Load patterns
    patterns_path = "diffusion_patterns/baseline_realistic_int8.npz"
    patterns_data = np.load(patterns_path, allow_pickle=True)

    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    A_csc = diffusion_csc.to_csc_matrix()  # A matrix (pixels, led_count*3)
    led_positions = patterns_data["led_positions"]

    led_count = 500
    print(f"A matrix shape: {A_csc.shape}, nnz: {A_csc.nnz}")

    # === Method 1: Current DIA matrix approach ===
    print("\n--- Method 1: DIA Matrix ---")

    dia_matrix = DiagonalATAMatrix(led_count=led_count)

    build_start = time.time()
    # Suppress output
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    dia_matrix.build_from_diffusion_matrix(A_csc, led_positions)
    sys.stdout = old_stdout

    build_time = time.time() - build_start
    print(f"DIA build time: {build_time:.3f}s")
    print(f"DIA shape: {dia_matrix.dia_data_cpu.shape}")
    print(f"DIA diagonals: {dia_matrix.dia_data_cpu.shape[1]}")

    # === Method 2: Direct CSC A^T A computation ===
    print("\n--- Method 2: CSC A^T A Matrix ---")

    # Compute A^T A directly using CSC operations
    ata_start = time.time()

    # We need to handle the 3-channel structure properly
    ATA_per_channel = []
    for channel in range(3):
        # Extract A matrix for this channel: A[:, channel::3]
        channel_cols = np.arange(channel, A_csc.shape[1], 3)
        A_channel = A_csc[:, channel_cols]  # Shape: (pixels, led_count)

        # Compute A^T A for this channel
        ATA_channel = A_channel.T @ A_channel  # Shape: (led_count, led_count)
        ATA_per_channel.append(ATA_channel)

    ata_time = time.time() - ata_start
    print(f"CSC A^T A build time: {ata_time:.3f}s")
    print(f"A^T A per channel shape: {ATA_per_channel[0].shape}")
    print(f"A^T A per channel nnz: {[mat.nnz for mat in ATA_per_channel]}")
    print(f"A^T A sparsity: {[mat.nnz / (led_count**2) * 100 for mat in ATA_per_channel]}%")

    # === Performance Testing ===
    print("\n--- Performance Comparison ---")

    # Test vectors
    test_led_values = np.full((3, led_count), 0.5, dtype=np.float32)
    test_led_values_gpu = cp.asarray(test_led_values)

    # Test DIA performance
    print("\nDIA Matrix Performance:")
    dia_times = []
    for _ in range(10):
        start = time.time()
        dia_result = dia_matrix.multiply_3d(test_led_values_gpu)
        if not isinstance(dia_result, cp.ndarray):
            dia_result = cp.asarray(dia_result)
        cp.cuda.Device().synchronize()
        dia_times.append(time.time() - start)

    dia_avg = np.mean(dia_times) * 1000
    print(f"  DIA multiply_3d: {dia_avg:.3f}ms")

    # Test CSC performance
    print("\nCSC Matrix Performance:")
    csc_times = []

    # Convert CSC matrices to GPU
    ATA_gpu = [cp.sparse.csr_matrix(mat.tocsr()) for mat in ATA_per_channel]

    for _ in range(10):
        start = time.time()

        # Manual CSC A^T A @ x computation
        csc_result = cp.zeros((3, led_count), dtype=cp.float32)
        for channel in range(3):
            csc_result[channel] = ATA_gpu[channel] @ test_led_values_gpu[channel]

        cp.cuda.Device().synchronize()
        csc_times.append(time.time() - start)

    csc_avg = np.mean(csc_times) * 1000
    print(f"  CSC A^T A @ x: {csc_avg:.3f}ms")

    # === Test g^T @ A^T A @ g operation ===
    print("\nTesting g^T @ A^T A @ g:")

    test_gradient = dia_result  # Use DIA result as test gradient

    # DIA g^T @ A^T A @ g
    dia_gatag_times = []
    for _ in range(10):
        start = time.time()
        dia_gatag = dia_matrix.g_ata_g_3d(test_gradient)
        if not isinstance(dia_gatag, cp.ndarray):
            dia_gatag = cp.asarray(dia_gatag)
        cp.cuda.Device().synchronize()
        dia_gatag_times.append(time.time() - start)

    dia_gatag_avg = np.mean(dia_gatag_times) * 1000
    print(f"  DIA g^T @ A^T A @ g: {dia_gatag_avg:.3f}ms")

    # CSC g^T @ A^T A @ g (manual computation)
    csc_gatag_times = []
    for _ in range(10):
        start = time.time()

        csc_gatag = cp.zeros(3, dtype=cp.float32)
        for channel in range(3):
            # g^T @ A^T A @ g = g^T @ (A^T A @ g)
            ata_g = ATA_gpu[channel] @ test_gradient[channel]
            csc_gatag[channel] = cp.dot(test_gradient[channel], ata_g)

        cp.cuda.Device().synchronize()
        csc_gatag_times.append(time.time() - start)

    csc_gatag_avg = np.mean(csc_gatag_times) * 1000
    print(f"  CSC g^T @ A^T A @ g: {csc_gatag_avg:.3f}ms")

    # === Summary ===
    print("\n--- Summary ---")
    diag_count = dia_matrix.dia_data_cpu.shape[1]
    efficiency = diag_count / 999 * 100
    print(f"DIA matrix diagonals: {diag_count} (efficiency: {efficiency:.1f}% of dense)")
    print(f"CSC matrix density: {np.mean([mat.nnz / (led_count**2) * 100 for mat in ATA_per_channel]):.1f}%")
    print("")
    print("Performance (target: <1ms each):")
    print(f"  DIA A^T A @ x:       {dia_avg:.3f}ms")
    print(f"  CSC A^T A @ x:       {csc_avg:.3f}ms")
    print(f"  DIA g^T @ A^T A @ g:  {dia_gatag_avg:.3f}ms")
    print(f"  CSC g^T @ A^T A @ g:  {csc_gatag_avg:.3f}ms")
    print("")
    if csc_avg < dia_avg:
        print(f"🏆 CSC is {dia_avg / csc_avg:.1f}x faster for A^T A @ x")
    else:
        print(f"🏆 DIA is {csc_avg / dia_avg:.1f}x faster for A^T A @ x")

    if csc_gatag_avg < dia_gatag_avg:
        print(f"🏆 CSC is {dia_gatag_avg / csc_gatag_avg:.1f}x faster for g^T @ A^T A @ g")
    else:
        print(f"🏆 DIA is {csc_gatag_avg / dia_gatag_avg:.1f}x faster for g^T @ A^T A @ g")


if __name__ == "__main__":
    test_csc_vs_dia_performance()
