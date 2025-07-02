#!/usr/bin/env python3
"""
Debug DIA matrix operations performance.

Focus on the multiply_3d and g_ata_g_3d operations that should be <1ms.
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


def debug_dia_matrix_performance():
    """Debug DIA matrix operations performance."""

    print("=== DEBUGGING DIA MATRIX PERFORMANCE ===")

    # Load patterns and build DIA matrix
    patterns_path = "diffusion_patterns/baseline_realistic_int8.npz"
    patterns_data = np.load(patterns_path, allow_pickle=True)

    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_matrix = diffusion_csc.to_csc_matrix()
    led_positions = patterns_data["led_positions"]

    led_count = 500
    dia_matrix = DiagonalATAMatrix(led_count=led_count)

    print("Building DIA matrix...")
    build_start = time.time()
    dia_matrix.build_from_diffusion_matrix(csc_matrix, led_positions)
    build_time = time.time() - build_start
    print(f"DIA matrix build time: {build_time:.3f}s")
    print(f"DIA matrix shape: {dia_matrix.dia_data_cpu.shape}")
    print(f"DIA matrix nnz: {np.count_nonzero(dia_matrix.dia_data_cpu)}")

    # Create test vectors
    test_vectors = {
        "small": np.random.randn(3, led_count).astype(np.float32) * 0.001,
        "medium": np.random.randn(3, led_count).astype(np.float32) * 1.0,
        "large": np.random.randn(3, led_count).astype(np.float32) * 100.0,
        "half_values": np.full((3, led_count), 0.5, dtype=np.float32),
    }

    for test_name, led_values_cpu in test_vectors.items():
        print(f"\n--- Testing with {test_name} values ---")
        print(f"LED values range: [{led_values_cpu.min():.6f}, {led_values_cpu.max():.6f}]")

        # Convert to GPU
        transfer_start = time.time()
        led_values_gpu = cp.asarray(led_values_cpu)
        transfer_time = time.time() - transfer_start
        print(f"CPU->GPU transfer: {transfer_time * 1000:.3f}ms")

        # === Test multiply_3d (A^T A @ x) ===
        print("\n  Testing multiply_3d (A^T A @ x):")

        # Warmup
        for _ in range(3):
            result = dia_matrix.multiply_3d(led_values_gpu)
            if not isinstance(result, cp.ndarray):
                result = cp.asarray(result)
            cp.cuda.Device().synchronize()

        # Timing
        times = []
        for _ in range(10):
            start = time.time()
            result = dia_matrix.multiply_3d(led_values_gpu)
            if not isinstance(result, cp.ndarray):
                result = cp.asarray(result)
            cp.cuda.Device().synchronize()
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        print(f"    multiply_3d: {avg_time:7.3f} ± {std_time:5.3f} ms")
        print(f"    Result shape: {result.shape}")
        print(f"    Result range: [{float(cp.min(result)):.6f}, {float(cp.max(result)):.6f}]")

        # === Test g_ata_g_3d (g^T @ A^T A @ g) ===
        print("\n  Testing g_ata_g_3d (g^T @ A^T A @ g):")

        # Use result as gradient for this test
        gradient_gpu = result

        # Warmup
        for _ in range(3):
            g_ata_g_result = dia_matrix.g_ata_g_3d(gradient_gpu)
            if not isinstance(g_ata_g_result, cp.ndarray):
                g_ata_g_result = cp.asarray(g_ata_g_result)
            cp.cuda.Device().synchronize()

        # Timing
        times = []
        for _ in range(10):
            start = time.time()
            g_ata_g_result = dia_matrix.g_ata_g_3d(gradient_gpu)
            if not isinstance(g_ata_g_result, cp.ndarray):
                g_ata_g_result = cp.asarray(g_ata_g_result)
            cp.cuda.Device().synchronize()
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        print(f"    g_ata_g_3d: {avg_time:7.3f} ± {std_time:5.3f} ms")
        print(f"    Result shape: {g_ata_g_result.shape}")
        print(f"    Result values: {[float(x) for x in g_ata_g_result]}")

    # === Test implementation details ===
    print("\n--- Implementation Analysis ---")

    # Check if DIA operations are using CPU or GPU
    print(f"DIA data location: {'GPU' if hasattr(dia_matrix, 'dia_data_gpu') else 'CPU'}")
    print(f"DIA data type: {dia_matrix.dia_data_cpu.dtype}")

    # Test CPU vs GPU performance if both are available
    test_led_values = cp.asarray(np.full((3, led_count), 0.5, dtype=np.float32))

    print("\nMemory transfer analysis:")

    # Test multiple operations to see cumulative overhead
    total_start = time.time()
    for i in range(5):
        result1 = dia_matrix.multiply_3d(test_led_values)
        if not isinstance(result1, cp.ndarray):
            result1 = cp.asarray(result1)

        result2 = dia_matrix.g_ata_g_3d(result1)
        if not isinstance(result2, cp.ndarray):
            result2 = cp.asarray(result2)

        cp.cuda.Device().synchronize()

    total_time = time.time() - total_start
    print(f"5 multiply_3d + g_ata_g_3d operations: {total_time * 1000:.3f}ms")
    print(f"Average per operation pair: {total_time * 1000 / 5:.3f}ms")

    # Check for any scipy warnings or inefficiencies
    import scipy.sparse

    print("\nSciPy sparse format analysis:")
    print("  DIA format efficiency warning: Check console for scipy warnings")

    # Test manual conversion impact
    dia_sparse = scipy.sparse.diags_array(
        dia_matrix.dia_data_cpu,
        offsets=dia_matrix.offsets,
        shape=(led_count, led_count),
    )
    print(f"  DIA matrix density: {dia_sparse.nnz / (led_count * led_count) * 100:.2f}%")
    print(f"  Number of diagonals: {len(dia_matrix.offsets)}")


if __name__ == "__main__":
    debug_dia_matrix_performance()
