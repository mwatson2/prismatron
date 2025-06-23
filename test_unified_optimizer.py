#!/usr/bin/env python3
"""
Test unified optimizer with both CSC and mixed tensor modes.
"""

import os
import sys
import time

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

# Direct import to avoid package issues
sys.path.append("src")
from consumer.led_optimizer_dense import DenseLEDOptimizer


def create_test_image():
    """Create a test image for optimization."""
    width, height = 800, 480
    test_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create a simple pattern
    for y in range(height):
        for x in range(width):
            test_image[y, x, 0] = int(255 * x / width)  # Red gradient
            test_image[y, x, 1] = int(255 * y / height)  # Green gradient
            test_image[y, x, 2] = 128  # Fixed blue

    return test_image


def test_optimizer_mode(use_mixed_tensor: bool):
    """Test optimizer in CSC or mixed tensor mode."""
    mode_name = "Mixed Tensor" if use_mixed_tensor else "CSC Format"
    print(f"\n=== Testing {mode_name} Mode ===")

    try:
        # Initialize optimizer
        optimizer = DenseLEDOptimizer(
            diffusion_patterns_path="diffusion_patterns/synthetic_1000_with_ata",
            use_mixed_tensor=use_mixed_tensor,
        )

        # Initialize
        init_start = time.time()
        success = optimizer.initialize()
        init_time = time.time() - init_start

        print(
            f"Initialization: {'SUCCESS' if success else 'FAILED'} ({init_time:.3f}s)"
        )

        if not success:
            return None

        # Test optimization
        test_image = create_test_image()

        # Run optimization (exclude frame conversion time)
        conversion_start = time.time()
        if use_mixed_tensor:
            target_converted = optimizer._convert_frame_to_planar_format(test_image)
        else:
            target_converted = optimizer._convert_frame_to_flat_format(test_image)
        conversion_time = time.time() - conversion_start

        # Run actual optimization
        opt_start = time.time()
        result = optimizer.optimize_frame(test_image, debug=False)
        opt_time = time.time() - opt_start

        print(f"Frame conversion: {conversion_time:.4f}s")
        print(f"Optimization: {opt_time:.3f}s")
        print(
            f"LED values range: [{result.led_values.min()}, {result.led_values.max()}]"
        )

        return {
            "mode": mode_name,
            "init_time": init_time,
            "conversion_time": conversion_time,
            "optimization_time": opt_time,
            "total_time": opt_time + conversion_time,
            "success": True,
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return {"mode": mode_name, "success": False, "error": str(e)}


def main():
    """Main test function."""
    print("Testing Unified LED Optimizer with CSC vs Mixed Tensor modes")
    print("=" * 60)

    # Test CSC mode
    csc_result = test_optimizer_mode(use_mixed_tensor=False)

    # Test Mixed Tensor mode
    mixed_result = test_optimizer_mode(use_mixed_tensor=True)

    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    if (
        csc_result
        and csc_result.get("success")
        and mixed_result
        and mixed_result.get("success")
    ):
        print(f"{'Metric':<25} {'CSC Format':<15} {'Mixed Tensor':<15} {'Ratio':<10}")
        print("-" * 65)

        csc_opt = csc_result["optimization_time"]
        mixed_opt = mixed_result["optimization_time"]
        opt_ratio = csc_opt / mixed_opt

        print(
            f"{'Optimization Time (s)':<25} {csc_opt:.3f}{'':>10} {mixed_opt:.3f}{'':>10} {opt_ratio:.2f}x{'':>5}"
        )

        csc_conv = csc_result["conversion_time"]
        mixed_conv = mixed_result["conversion_time"]
        conv_ratio = csc_conv / mixed_conv

        print(
            f"{'Frame Conversion (s)':<25} {csc_conv:.4f}{'':>9} {mixed_conv:.4f}{'':>9} {conv_ratio:.2f}x{'':>5}"
        )

        print(f"\nNote: Frame conversion time excluded from performance comparison")
        print(f"      as this would be optimized in production.")

    else:
        print("Cannot compare - one or both modes failed")
        if csc_result and not csc_result.get("success"):
            print(f"CSC Error: {csc_result.get('error')}")
        if mixed_result and not mixed_result.get("success"):
            print(f"Mixed Tensor Error: {mixed_result.get('error')}")


if __name__ == "__main__":
    main()
