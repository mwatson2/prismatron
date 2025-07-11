#!/usr/bin/env python3
"""
Test different diagonal factors to find the sweet spot between accuracy and memory efficiency.
"""

import sys
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.append(str(Path(__file__).parent / "src"))

from utils.diagonal_ata_matrix import DiagonalATAMatrix


def test_diagonal_factor_sweep():
    """Test different diagonal factors available in pattern files."""

    # Available pattern files with different diagonal factors
    factors = [1.0, 1.2, 2.0, 10.0]  # Based on what's likely available

    results = []

    # Load dense reference
    reference_file = "diffusion_patterns/synthetic_2624_uint8_dia_10.0.npz"
    if not Path(reference_file).exists():
        print(f"Reference file not found: {reference_file}")
        return

    data_ref = np.load(reference_file, allow_pickle=True)
    ata_inverse_dense = data_ref["ata_inverse"]
    ata_inverse_gpu = cp.asarray(ata_inverse_dense)

    # Create test vector
    np.random.seed(42)
    test_vector = np.random.rand(3, 2624).astype(np.float32) * 0.1
    test_vector_gpu = cp.asarray(test_vector)

    # Dense reference result
    result_dense = cp.einsum("ijk,ik->ij", ata_inverse_gpu, test_vector_gpu)
    result_dense_cpu = cp.asnumpy(result_dense)
    dense_rms = np.sqrt(np.mean(result_dense_cpu**2))

    print("Testing different diagonal factors:")
    print("=" * 60)

    for factor in factors:
        pattern_file = f"diffusion_patterns/synthetic_2624_uint8_dia_{factor}.npz"

        if not Path(pattern_file).exists():
            print(f"❌ Factor {factor}: File not found - {pattern_file}")
            continue

        # Load DIA matrix
        data = np.load(pattern_file, allow_pickle=True)
        ata_inv_dia = DiagonalATAMatrix.from_dict(data["ata_inverse_dia"].item())

        # Test operation
        result_dia = ata_inv_dia.multiply_3d(test_vector_gpu)
        result_dia_cpu = cp.asnumpy(result_dia)

        # Compare with dense
        max_diff = np.max(np.abs(result_dense_cpu - result_dia_cpu))
        rms_diff = np.sqrt(np.mean((result_dense_cpu - result_dia_cpu) ** 2))
        rel_error = (rms_diff / dense_rms * 100) if dense_rms > 1e-10 else 0

        # Memory usage estimate
        dia_elements = ata_inv_dia.k * ata_inv_dia.led_count * ata_inv_dia.channels
        dense_elements = ata_inv_dia.led_count * ata_inv_dia.led_count * ata_inv_dia.channels
        memory_ratio = dia_elements / dense_elements

        results.append(
            {
                "factor": factor,
                "k": ata_inv_dia.k,
                "bandwidth": ata_inv_dia.bandwidth,
                "max_diff": max_diff,
                "rms_diff": rms_diff,
                "rel_error": rel_error,
                "memory_ratio": memory_ratio,
            }
        )

        print(
            f"✅ Factor {factor:4.1f}: k={ata_inv_dia.k:4d}, bandwidth={ata_inv_dia.bandwidth:4d}, "
            f"rel_error={rel_error:6.2f}%, memory={memory_ratio:.3f}x"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Factor':>6} {'K':>6} {'BWid':>6} {'RelErr%':>8} {'Memory':>8} {'Quality':>12}")
    print("-" * 60)

    for r in results:
        quality = (
            "Perfect"
            if r["rel_error"] < 0.01
            else "Excellent" if r["rel_error"] < 1.0 else "Good" if r["rel_error"] < 5.0 else "Fair"
        )
        print(
            f"{r['factor']:6.1f} {r['k']:6d} {r['bandwidth']:6d} {r['rel_error']:8.2f} {r['memory_ratio']:8.3f} {quality:>12}"
        )

    # Find sweet spot
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    # Perfect accuracy
    perfect = [r for r in results if r["rel_error"] < 0.01]
    if perfect:
        best_perfect = min(perfect, key=lambda x: x["memory_ratio"])
        print(
            f"Best perfect accuracy: Factor {best_perfect['factor']:.1f} "
            f"(k={best_perfect['k']}, memory={best_perfect['memory_ratio']:.3f}x)"
        )

    # Best balance (< 1% error, minimum memory)
    good_balance = [r for r in results if r["rel_error"] < 1.0]
    if good_balance:
        best_balance = min(good_balance, key=lambda x: x["memory_ratio"])
        print(
            f"Best balance (<1% error): Factor {best_balance['factor']:.1f} "
            f"(k={best_balance['k']}, {best_balance['rel_error']:.2f}% error, "
            f"memory={best_balance['memory_ratio']:.3f}x)"
        )

    # Most memory efficient
    if results:
        most_efficient = min(results, key=lambda x: x["memory_ratio"])
        print(
            f"Most memory efficient: Factor {most_efficient['factor']:.1f} "
            f"(k={most_efficient['k']}, {most_efficient['rel_error']:.2f}% error, "
            f"memory={most_efficient['memory_ratio']:.3f}x)"
        )


if __name__ == "__main__":
    test_diagonal_factor_sweep()
