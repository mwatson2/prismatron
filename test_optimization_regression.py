#!/usr/bin/env python3
"""
Quick regression test to diagnose optimization convergence issues.

This script tests basic optimization convergence with simple test cases
to identify where the regression might be occurring.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_2600_patterns():
    """Load 2600 LED patterns."""
    pattern_path = Path(__file__).parent / "diffusion_patterns" / "synthetic_2600_64x64_v7.npz"
    data = np.load(str(pattern_path), allow_pickle=True)

    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())
    dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())

    return mixed_tensor, dia_matrix


def test_simple_convergence():
    """Test convergence with very simple target frames."""
    print("=" * 60)
    print("OPTIMIZATION REGRESSION TEST")
    print("=" * 60)

    mixed_tensor, dia_matrix = load_2600_patterns()
    led_count = mixed_tensor.batch_size

    print(f"Loaded patterns: {led_count} LEDs")
    print(f"DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k}")

    # Test cases with increasing complexity
    test_cases = [
        ("Black frame", np.zeros((3, 480, 800), dtype=np.uint8)),
        ("White frame", np.full((3, 480, 800), 255, dtype=np.uint8)),
        ("Gray frame", np.full((3, 480, 800), 128, dtype=np.uint8)),
        (
            "Red frame",
            np.array(
                [
                    np.full((480, 800), 255, dtype=np.uint8),  # Red channel
                    np.zeros((480, 800), dtype=np.uint8),  # Green channel
                    np.zeros((480, 800), dtype=np.uint8),  # Blue channel
                ]
            ),
        ),
        (
            "Simple gradient",
            np.array(
                [
                    np.linspace(0, 255, 800, dtype=np.uint8)[None, :].repeat(480, axis=0),  # Red gradient
                    np.zeros((480, 800), dtype=np.uint8),  # Green
                    np.zeros((480, 800), dtype=np.uint8),  # Blue
                ]
            ),
        ),
    ]

    results = {}

    for test_name, target_frame in test_cases:
        print(f"\n--- {test_name} ---")
        print(f"Target shape: {target_frame.shape}")
        print(f"Target range: [{target_frame.min()}, {target_frame.max()}]")

        try:
            start_time = time.perf_counter()
            result = optimize_frame_led_values(
                target_frame=target_frame,
                AT_matrix=mixed_tensor,
                ATA_matrix=dia_matrix,
                max_iterations=50,
                convergence_threshold=0.3,
                debug=True,
                enable_timing=False,
            )
            total_time = time.perf_counter() - start_time

            print(f"Result: {result.iterations} iterations, converged: {result.converged}")
            print(f"Time: {total_time:.3f}s ({total_time / result.iterations * 1000:.1f}ms per iter)")
            print(f"LED values range: [{result.led_values.min()}, {result.led_values.max()}]")

            results[test_name] = {
                "iterations": result.iterations,
                "converged": result.converged,
                "total_time": total_time,
                "led_range": (result.led_values.min(), result.led_values.max()),
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()
            results[test_name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    converged_tests = []
    non_converged_tests = []

    for test_name, result in results.items():
        if "error" in result:
            print(f"❌ {test_name}: ERROR - {result['error']}")
        elif result["converged"]:
            converged_tests.append(test_name)
            print(f"✅ {test_name}: {result['iterations']} iterations")
        else:
            non_converged_tests.append(test_name)
            print(f"⚠️  {test_name}: {result['iterations']} iterations (no convergence)")

    print(f"\nConverged: {len(converged_tests)}/{len(test_cases)}")

    if len(converged_tests) == 0:
        print("\n🚨 REGRESSION DETECTED: No test cases converged!")
        print("This suggests a fundamental issue with the optimization loop.")
    elif len(converged_tests) < len(test_cases):
        print("\n⚠️  PARTIAL REGRESSION: Only simple cases converged")
        print("This suggests issues with complex optimization landscapes.")
    else:
        print("\n✅ All test cases converged - optimization appears healthy")

    return results


def test_basic_operations():
    """Test basic matrix operations to ensure they're working correctly."""
    print("\n" + "=" * 60)
    print("BASIC OPERATIONS TEST")
    print("=" * 60)

    mixed_tensor, dia_matrix = load_2600_patterns()
    led_count = mixed_tensor.batch_size

    # Test 1: DIA matrix multiply
    print("Testing DIA matrix operations...")
    test_values = np.random.rand(3, led_count).astype(np.float32)

    try:
        result = dia_matrix.multiply_3d(test_values)
        print(f"✅ DIA multiply: input {test_values.shape} -> output {result.shape}")
        print(f"   Input range: [{test_values.min():.3f}, {test_values.max():.3f}]")
        print(f"   Output range: [{result.min():.3f}, {result.max():.3f}]")
    except Exception as e:
        print(f"❌ DIA multiply failed: {e}")
        return False

    # Test 2: Mixed tensor transpose dot product
    print("\nTesting mixed tensor operations...")
    test_frame = np.random.randint(0, 256, (3, 480, 800), dtype=np.uint8)

    try:
        from src.utils.frame_optimizer import _calculate_ATb

        ATb = _calculate_ATb(test_frame, mixed_tensor, debug=False)
        print(f"✅ Mixed tensor A^T @ b: frame {test_frame.shape} -> ATb {ATb.shape}")
        print(f"   Frame range: [{test_frame.min()}, {test_frame.max()}]")
        print(f"   ATb range: [{ATb.min():.3f}, {ATb.max():.3f}]")
    except Exception as e:
        print(f"❌ Mixed tensor operation failed: {e}")
        return False

    # Test 3: Step size calculation
    print("\nTesting step size calculation...")
    gradient = np.random.randn(3, led_count).astype(np.float32) * 0.1

    try:
        g_dot_g = np.sum(gradient * gradient)
        g_dot_ATA_g_per_channel = dia_matrix.g_ata_g_3d(gradient)
        g_dot_ATA_g = np.sum(g_dot_ATA_g_per_channel)

        if g_dot_ATA_g > 0:
            step_size = 0.9 * g_dot_g / g_dot_ATA_g
            print(f"✅ Step size calculation: {step_size:.6f}")
            print(f"   g^T g: {g_dot_g:.6f}")
            print(f"   g^T A^T A g: {g_dot_ATA_g:.6f}")
        else:
            print(f"❌ Invalid step size: g^T A^T A g = {g_dot_ATA_g}")
            return False

    except Exception as e:
        print(f"❌ Step size calculation failed: {e}")
        return False

    print("\n✅ All basic operations working correctly")
    return True


if __name__ == "__main__":
    # Test basic operations first
    operations_ok = test_basic_operations()

    if operations_ok:
        # Test convergence
        test_simple_convergence()
    else:
        print("\n🚨 Basic operations failed - skipping convergence tests")
