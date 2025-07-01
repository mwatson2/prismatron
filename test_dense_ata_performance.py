#!/usr/bin/env python3
"""
Test dense A^T A matrix performance.

Since the A^T A matrix is 98% dense (979/999 diagonals), treating it as
dense with optimized BLAS operations might be faster.
"""

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix


def test_dense_ata_performance():
    """Test dense A^T A matrix performance vs sparse approaches."""

    print("=== Dense A^T A Performance Test ===")

    # Load patterns
    patterns_path = "diffusion_patterns/baseline_realistic_int8.npz"
    patterns_data = np.load(patterns_path, allow_pickle=True)

    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    A_csc = diffusion_csc.to_csc_matrix()
    led_positions = patterns_data["led_positions"]

    led_count = 500
    print(f"A matrix shape: {A_csc.shape}, nnz: {A_csc.nnz}")

    # === Method 1: Current DIA (for comparison) ===
    print(f"\n--- Method 1: DIA Matrix (Current) ---")

    dia_matrix = DiagonalATAMatrix(led_count=led_count)

    # Suppress DIA build output
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    dia_matrix.build_from_diffusion_matrix(A_csc, led_positions)
    sys.stdout = old_stdout

    print(
        f"DIA diagonals: {dia_matrix.dia_data_cpu.shape[1]} ({dia_matrix.dia_data_cpu.shape[1]/999*100:.1f}% dense)"
    )

    # === Method 2: Dense A^T A matrices ===
    print(f"\n--- Method 2: Dense A^T A Matrices ---")

    dense_build_start = time.time()

    # Build dense A^T A matrices per channel
    ATA_dense_channels = []
    for channel in range(3):
        # Extract A matrix for this channel
        channel_cols = np.arange(channel, A_csc.shape[1], 3)
        A_channel = A_csc[:, channel_cols]  # Shape: (pixels, led_count)

        # Compute dense A^T A
        ATA_channel_sparse = A_channel.T @ A_channel
        ATA_channel_dense = ATA_channel_sparse.toarray().astype(np.float32)

        ATA_dense_channels.append(ATA_channel_dense)

    # Transfer to GPU
    ATA_dense_gpu = [cp.asarray(mat) for mat in ATA_dense_channels]

    dense_build_time = time.time() - dense_build_start
    print(f"Dense A^T A build time: {dense_build_time:.3f}s")
    print(f"Dense A^T A shape per channel: {ATA_dense_channels[0].shape}")
    print(
        f"Dense A^T A memory per channel: {ATA_dense_channels[0].nbytes / 1024 / 1024:.1f} MB"
    )
    print(
        f"Total dense A^T A memory: {3 * ATA_dense_channels[0].nbytes / 1024 / 1024:.1f} MB"
    )

    # === Performance Testing ===
    print(f"\n--- Performance Comparison ---")

    # Test vectors
    test_led_values = np.full((3, led_count), 0.5, dtype=np.float32)
    test_led_values_gpu = cp.asarray(test_led_values)

    # === Test A^T A @ x operations ===
    print(f"\nA^T A @ x Performance:")

    # DIA performance
    dia_times = []
    for _ in range(20):
        start = time.time()
        dia_result = dia_matrix.multiply_3d(test_led_values_gpu)
        if not isinstance(dia_result, cp.ndarray):
            dia_result = cp.asarray(dia_result)
        cp.cuda.Device().synchronize()
        dia_times.append(time.time() - start)

    dia_avg = np.mean(dia_times) * 1000
    dia_std = np.std(dia_times) * 1000
    print(f"  DIA A^T A @ x:  {dia_avg:.3f} ± {dia_std:.3f} ms")

    # Dense performance
    dense_times = []
    for _ in range(20):
        start = time.time()

        dense_result = cp.zeros((3, led_count), dtype=cp.float32)
        for channel in range(3):
            dense_result[channel] = (
                ATA_dense_gpu[channel] @ test_led_values_gpu[channel]
            )

        cp.cuda.Device().synchronize()
        dense_times.append(time.time() - start)

    dense_avg = np.mean(dense_times) * 1000
    dense_std = np.std(dense_times) * 1000
    print(f"  Dense A^T A @ x: {dense_avg:.3f} ± {dense_std:.3f} ms")

    # === Test g^T @ A^T A @ g operations ===
    print(f"\ng^T @ A^T A @ g Performance:")

    test_gradient = dia_result  # Use result as gradient

    # DIA g^T @ A^T A @ g
    dia_gatag_times = []
    for _ in range(20):
        start = time.time()
        dia_gatag = dia_matrix.g_ata_g_3d(test_gradient)
        if not isinstance(dia_gatag, cp.ndarray):
            dia_gatag = cp.asarray(dia_gatag)
        cp.cuda.Device().synchronize()
        dia_gatag_times.append(time.time() - start)

    dia_gatag_avg = np.mean(dia_gatag_times) * 1000
    dia_gatag_std = np.std(dia_gatag_times) * 1000
    print(f"  DIA g^T @ A^T A @ g:  {dia_gatag_avg:.3f} ± {dia_gatag_std:.3f} ms")

    # Dense g^T @ A^T A @ g
    dense_gatag_times = []
    for _ in range(20):
        start = time.time()

        dense_gatag = cp.zeros(3, dtype=cp.float32)
        for channel in range(3):
            # Two approaches: compute A^T A @ g first, or use dot(g^T, A^T A @ g)
            ata_g = ATA_dense_gpu[channel] @ test_gradient[channel]
            dense_gatag[channel] = cp.dot(test_gradient[channel], ata_g)

        cp.cuda.Device().synchronize()
        dense_gatag_times.append(time.time() - start)

    dense_gatag_avg = np.mean(dense_gatag_times) * 1000
    dense_gatag_std = np.std(dense_gatag_times) * 1000
    print(f"  Dense g^T @ A^T A @ g: {dense_gatag_avg:.3f} ± {dense_gatag_std:.3f} ms")

    # === Test optimized dense operations ===
    print(f"\nOptimized Dense Operations:")

    # Use cuBLAS directly for matrix-vector multiplication
    cublas_times = []
    for _ in range(20):
        start = time.time()

        # Use cuBLAS GEMV (General Matrix-Vector multiplication)
        cublas_result = cp.zeros((3, led_count), dtype=cp.float32)
        for channel in range(3):
            # GEMV: y = alpha * A @ x + beta * y
            cublas_result[channel] = cp.dot(
                ATA_dense_gpu[channel], test_led_values_gpu[channel]
            )

        cp.cuda.Device().synchronize()
        cublas_times.append(time.time() - start)

    cublas_avg = np.mean(cublas_times) * 1000
    cublas_std = np.std(cublas_times) * 1000
    print(f"  cuBLAS A^T A @ x: {cublas_avg:.3f} ± {cublas_std:.3f} ms")

    # === Summary ===
    print(f"\n--- Performance Summary (target: <1ms each) ---")

    operations = [
        ("DIA A^T A @ x", dia_avg, dia_std),
        ("Dense A^T A @ x", dense_avg, dense_std),
        ("cuBLAS A^T A @ x", cublas_avg, cublas_std),
        ("DIA g^T @ A^T A @ g", dia_gatag_avg, dia_gatag_std),
        ("Dense g^T @ A^T A @ g", dense_gatag_avg, dense_gatag_std),
    ]

    print(f"")
    for name, avg_time, std_time in operations:
        status = "✅" if avg_time < 1.0 else f"⚠️ {avg_time:.1f}x slower"
        print(f"{name:20}: {avg_time:6.3f} ± {std_time:5.3f} ms  {status}")

    # Best performer
    ata_performers = [("DIA", dia_avg), ("Dense", dense_avg), ("cuBLAS", cublas_avg)]
    best_ata = min(ata_performers, key=lambda x: x[1])

    gatag_performers = [("DIA", dia_gatag_avg), ("Dense", dense_gatag_avg)]
    best_gatag = min(gatag_performers, key=lambda x: x[1])

    print(f"\n🏆 Best A^T A @ x: {best_ata[0]} ({best_ata[1]:.3f}ms)")
    print(f"🏆 Best g^T @ A^T A @ g: {best_gatag[0]} ({best_gatag[1]:.3f}ms)")

    # Memory usage comparison
    dia_memory = dia_matrix.dia_data_cpu.nbytes / 1024 / 1024
    dense_memory = sum(mat.nbytes for mat in ATA_dense_channels) / 1024 / 1024

    print(f"\n📊 Memory Usage:")
    print(f"  DIA matrix: {dia_memory:.1f} MB")
    print(f"  Dense matrices: {dense_memory:.1f} MB")
    print(f"  Memory ratio: {dense_memory/dia_memory:.1f}x")


if __name__ == "__main__":
    test_dense_ata_performance()
