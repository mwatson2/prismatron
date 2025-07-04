#!/usr/bin/env python3
"""
Test 1000 LEDs specifically to confirm no regression and measure iteration savings.
"""

import time
from pathlib import Path

import numpy as np

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def test_1000_leds():
    """Test with 1000 LEDs to confirm no regression."""
    # Check available pattern files
    patterns_dir = Path("diffusion_patterns")
    pattern_files = list(patterns_dir.glob("*.npz"))

    print("Available pattern files:")
    for f in pattern_files:
        print(f"  {f.name}")

    # Look for 1000 LED pattern
    pattern_1000_path = None
    for f in pattern_files:
        if "1000" in f.name:
            pattern_1000_path = f
            break

    if not pattern_1000_path:
        print("\n❌ No 1000 LED pattern file found")
        print("Available files suggest using 2600 LED patterns for testing")
        return None

    print(f"\n✅ Found 1000 LED pattern: {pattern_1000_path.name}")

    # Load patterns
    data = np.load(str(pattern_1000_path), allow_pickle=True)
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())
    dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())

    print(f"Loaded: {mixed_tensor.batch_size} LEDs")

    # Test several frames
    test_frames = [
        ("Black", np.zeros((3, 480, 800), dtype=np.uint8)),
        ("White", np.full((3, 480, 800), 255, dtype=np.uint8)),
        ("Gray", np.full((3, 480, 800), 128, dtype=np.uint8)),
    ]

    print("\n1000 LED Pattern - Regression Test:")
    print(f"{'Frame':12} {'Iterations':>10} {'Converged':>10} {'Time(ms)':>10}")
    print("-" * 50)

    for name, frame in test_frames:
        start_time = time.perf_counter()
        result = optimize_frame_led_values(
            target_frame=frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            max_iterations=50,
            debug=False,
        )
        time_ms = (time.perf_counter() - start_time) * 1000

        print(f"{name:12} {result.iterations:>10} {str(result.converged):>10} {time_ms:>10.1f}")

    print("\n✅ 1000 LED optimization working correctly")
    return mixed_tensor, dia_matrix


def test_iteration_savings_2600():
    """Test iteration savings with 2600 LEDs and proper convergence thresholds."""
    print("\n" + "=" * 70)
    print("ITERATION SAVINGS ANALYSIS - 2600 LEDs")
    print("=" * 70)

    # Load 2600 LED patterns with ATA inverse
    pattern_path = Path("diffusion_patterns/synthetic_2600_64x64_v7.npz")
    data = np.load(str(pattern_path), allow_pickle=True)

    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())
    dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())
    ata_inverse = data.get("ata_inverse", None)

    if ata_inverse is None:
        print("❌ No ATA inverse found")
        return

    print(f"Loaded: {mixed_tensor.batch_size} LEDs with ATA inverse")

    # Create a test frame
    target_frame = np.zeros((3, 480, 800), dtype=np.uint8)
    target_frame[0, 100:200, 200:400] = 255  # Red square
    target_frame[1, 200:300, 300:500] = 255  # Green square
    target_frame[2, 150:250, 250:450] = 255  # Blue square

    print("Test frame: Complex multi-color pattern")

    # Test parameters for convergence
    test_params = [
        {"threshold": 1.0, "max_iters": 50},
        {"threshold": 0.5, "max_iters": 50},
        {"threshold": 0.3, "max_iters": 50},
    ]

    for params in test_params:
        print(f"\n--- Convergence threshold: {params['threshold']} ---")

        # Test WITHOUT ATA inverse
        result_default = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            max_iterations=params["max_iters"],
            convergence_threshold=params["threshold"],
            debug=False,
        )

        # Test WITH ATA inverse
        result_ata_inv = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            ATA_inverse=ata_inverse,
            max_iterations=params["max_iters"],
            convergence_threshold=params["threshold"],
            debug=False,
        )

        # Calculate savings
        iteration_savings = result_default.iterations - result_ata_inv.iterations

        print(f"  DEFAULT:    {result_default.iterations:2d} iterations, converged: {result_default.converged}")
        print(f"  ATA INVERSE: {result_ata_inv.iterations:2d} iterations, converged: {result_ata_inv.converged}")
        print(f"  SAVINGS:     {iteration_savings:2d} iterations")

        if result_default.converged and result_ata_inv.converged:
            print(f"  ✅ Both converged - iteration savings: {iteration_savings}")
            break

    return mixed_tensor, dia_matrix, ata_inverse


def test_timing_section_investigation():
    """Investigate why timing sections appear faster with ATA inverse."""
    print("\n" + "=" * 70)
    print("TIMING SECTION INVESTIGATION")
    print("=" * 70)

    # Load patterns
    pattern_path = Path("diffusion_patterns/synthetic_2600_64x64_v7.npz")
    data = np.load(str(pattern_path), allow_pickle=True)

    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())
    dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())
    ata_inverse = data.get("ata_inverse", None)

    if ata_inverse is None:
        print("❌ No ATA inverse found")
        return

    # Simple test frame
    target_frame = np.full((3, 480, 800), 128, dtype=np.uint8)

    print("Testing with same number of iterations to isolate timing effects...")

    # Run with exact same number of iterations
    fixed_iterations = 10

    print(f"\n--- Fixed {fixed_iterations} iterations test ---")

    # Test WITHOUT ATA inverse
    result_default = optimize_frame_led_values(
        target_frame=target_frame,
        AT_matrix=mixed_tensor,
        ATA_matrix=dia_matrix,
        max_iterations=fixed_iterations,
        convergence_threshold=1e-10,  # Very strict so it won't converge early
        enable_timing=True,
        debug=False,
    )

    # Test WITH ATA inverse
    result_ata_inv = optimize_frame_led_values(
        target_frame=target_frame,
        AT_matrix=mixed_tensor,
        ATA_matrix=dia_matrix,
        ATA_inverse=ata_inverse,
        max_iterations=fixed_iterations,
        convergence_threshold=1e-10,  # Very strict so it won't converge early
        enable_timing=True,
        debug=False,
    )

    print(f"DEFAULT:     {result_default.iterations} iterations")
    print(f"ATA INVERSE: {result_ata_inv.iterations} iterations")

    if result_default.timing_data and result_ata_inv.timing_data:
        print("\nCore optimization timing comparison:")
        core_sections = ["ata_multiply", "gradient_calculation", "gradient_step"]

        for section in core_sections:
            if section in result_default.timing_data and section in result_ata_inv.timing_data:
                default_time = result_default.timing_data[section]
                ata_inv_time = result_ata_inv.timing_data[section]
                diff_pct = (ata_inv_time - default_time) / default_time * 100

                print(f"  {section:20s}: {default_time:.6f}s vs {ata_inv_time:.6f}s ({diff_pct:+.1f}%)")

        # Show initialization time
        if "ata_inverse_initialization" in result_ata_inv.timing_data:
            init_time = result_ata_inv.timing_data["ata_inverse_initialization"]
            print(f"  {'initialization':20s}: {init_time:.6f}s (ATA inverse only)")

    print("\n✅ Investigation completed")


if __name__ == "__main__":
    # Test 1: 1000 LED regression check
    test_1000_leds()

    # Test 2: Iteration savings analysis
    test_iteration_savings_2600()

    # Test 3: Timing section investigation
    test_timing_section_investigation()
