#!/usr/bin/env python3
"""
Analyze LED Brightness and Clipping.

Analyze LED brightness distribution and clipping in captured patterns
to understand saturation issues in optimization results.
"""

import logging

# Set up path
import sys
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.dense_ata_matrix import DenseATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_led_brightness_distribution():
    """Analyze LED brightness distribution in optimization results."""

    pattern_file = "diffusion_patterns/capture-0728-01-dense_fixed.npz"

    if not Path(pattern_file).exists():
        logger.error(f"Pattern file not found: {pattern_file}")
        return False

    logger.info(f"Loading pattern file: {pattern_file}")
    data = np.load(pattern_file, allow_pickle=True)

    # Load components
    mixed_tensor_data = data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_data)

    dense_ata_data = data["dense_ata_matrix"].item()
    dense_ata = DenseATAMatrix.from_dict(dense_ata_data)

    logger.info(f"Mixed tensor: {mixed_tensor}")
    logger.info(f"Dense ATA matrix: {dense_ata}")

    # Test with different brightness levels to see clipping behavior
    test_cases = [("Dark (64)", 64), ("Mid-gray (128)", 128), ("Bright (192)", 192), ("Very bright (255)", 255)]

    results = {}

    for name, brightness in test_cases:
        logger.info(f"\n=== Testing {name} ===")

        # Create test frame
        test_frame = cp.ones((3, 480, 800), dtype=cp.uint8) * brightness

        # Run optimization
        simple_init = np.ones((3, dense_ata.led_count), dtype=np.float32) * 0.1

        try:
            result = optimize_frame_led_values(
                target_frame=test_frame,
                at_matrix=mixed_tensor,
                ata_matrix=dense_ata,
                ata_inverse=data["ata_inverse"],
                initial_values=simple_init,
                max_iterations=10,
                compute_error_metrics=True,
                debug=False,
            )

            # Analyze LED values
            led_values_cpu = cp.asnumpy(result.led_values)

            # Count clipping
            clipped_low = np.sum(led_values_cpu == 0)
            clipped_high = np.sum(led_values_cpu == 255)
            total_values = led_values_cpu.size

            # Statistics
            stats = {
                "brightness": brightness,
                "led_min": led_values_cpu.min(),
                "led_max": led_values_cpu.max(),
                "led_mean": led_values_cpu.mean(),
                "led_std": led_values_cpu.std(),
                "led_median": np.median(led_values_cpu),
                "clipped_low": clipped_low,
                "clipped_high": clipped_high,
                "clipped_total": clipped_low + clipped_high,
                "clipped_percent": (clipped_low + clipped_high) / total_values * 100,
                "error_metrics": result.error_metrics,
            }

            results[name] = stats

            logger.info(f"Target brightness: {brightness}")
            logger.info(
                f"LED values: [{stats['led_min']:.1f}, {stats['led_max']:.1f}], mean={stats['led_mean']:.1f}, std={stats['led_std']:.1f}"
            )
            logger.info(
                f"Clipping: {stats['clipped_low']} low + {stats['clipped_high']} high = {stats['clipped_total']} total ({stats['clipped_percent']:.1f}%)"
            )
            logger.info(f"MSE: {stats['error_metrics']['mse']:.1f}, MAE: {stats['error_metrics']['mae']:.1f}")

        except Exception as e:
            logger.error(f"Failed optimization for {name}: {e}")
            continue

    # Print summary
    logger.info("\n=== BRIGHTNESS ANALYSIS SUMMARY ===")
    for name, stats in results.items():
        logger.info(
            f"{name}: Target={stats['brightness']}, LEDs=[{stats['led_min']:.0f},{stats['led_max']:.0f}], Mean={stats['led_mean']:.1f}, Clipped={stats['clipped_percent']:.1f}%, MSE={stats['error_metrics']['mse']:.1f}"
        )

    # Check for concerning patterns
    logger.info("\n=== POTENTIAL ISSUES ===")

    for name, stats in results.items():
        if stats["clipped_percent"] > 10:
            logger.warning(f"❌ {name}: High clipping ({stats['clipped_percent']:.1f}%) - indicates saturation issues")

        if stats["led_max"] == 255 and stats["brightness"] < 255:
            logger.warning(f"❌ {name}: LEDs hitting max brightness for target < 255 - patterns may be too bright")

        if stats["led_mean"] > stats["brightness"] * 1.5:
            logger.warning(
                f"❌ {name}: LED mean ({stats['led_mean']:.1f}) much higher than target ({stats['brightness']}) - scaling issue"
            )

    return results


def analyze_pattern_brightness():
    """Analyze the brightness of captured diffusion patterns themselves."""

    pattern_file = "diffusion_patterns/capture-0728-01-dense_fixed.npz"
    data = np.load(pattern_file, allow_pickle=True)

    logger.info("\n=== ANALYZING CAPTURED PATTERN BRIGHTNESS ===")

    # Load mixed tensor to examine pattern values
    mixed_tensor_data = data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_data)

    # Sample some blocks to analyze pattern brightness
    sample_blocks = []
    max_samples = 50  # Sample first 50 LEDs

    for led_idx in range(min(max_samples, mixed_tensor.batch_size)):
        for channel in range(3):
            block = mixed_tensor.sparse_values[channel, led_idx]
            if hasattr(block, "get"):  # CuPy array
                block_np = cp.asnumpy(block)
            else:
                block_np = block

            # Only analyze non-zero blocks
            if np.max(block_np) > 0:
                sample_blocks.append(block_np)

    if sample_blocks:
        # Combine all sampled blocks
        all_values = np.concatenate([block.flatten() for block in sample_blocks])
        non_zero_values = all_values[all_values > 0]

        logger.info(f"Sampled {len(sample_blocks)} non-zero pattern blocks")
        logger.info(f"Pattern value range: [{non_zero_values.min():.1f}, {non_zero_values.max():.1f}]")
        logger.info(f"Pattern value mean: {non_zero_values.mean():.1f}")
        logger.info(f"Pattern value std: {non_zero_values.std():.1f}")
        logger.info(f"Pattern value median: {np.median(non_zero_values):.1f}")

        # Check if patterns are uint8 scaled
        if mixed_tensor.dtype == np.uint8:
            logger.info("Pattern dtype: uint8 [0,255]")
            max_theoretical = 255
        else:
            logger.info("Pattern dtype: float32 [0,1]")
            max_theoretical = 1.0

        brightness_ratio = non_zero_values.max() / max_theoretical
        logger.info(f"Brightness ratio: {brightness_ratio:.3f} (1.0 = full scale)")

        if brightness_ratio > 0.8:
            logger.warning("❌ Patterns are very bright - may cause saturation when multiple LEDs overlap")
        elif brightness_ratio < 0.1:
            logger.warning("❌ Patterns are very dim - may cause poor optimization convergence")
        else:
            logger.info("✅ Pattern brightness appears reasonable")

    else:
        logger.error("No non-zero pattern blocks found for analysis")


def suggest_brightness_scaling():
    """Suggest brightness scaling based on analysis."""

    logger.info("\n=== BRIGHTNESS SCALING RECOMMENDATIONS ===")

    # The key insight: if LEDs are additive and we have overlapping patterns,
    # the individual LED patterns should be scaled down so that when multiple
    # LEDs contribute to the same pixel, the sum doesn't exceed [0,255]

    logger.info("Theoretical considerations:")
    logger.info("1. LEDs are additive - multiple LEDs contribute to same pixels")
    logger.info("2. Each LED has a ~64x64 diffusion pattern")
    logger.info("3. In dense LED arrangements, 4-6 LEDs may overlap at same pixel")
    logger.info("4. If each LED pattern has max value 255, overlap of 4 LEDs = 1020 (clipped to 255)")
    logger.info("5. For no clipping with N overlapping LEDs, max pattern value should be 255/N")

    # Estimate typical overlap from LED density
    pattern_file = "diffusion_patterns/capture-0728-01-dense_fixed.npz"
    data = np.load(pattern_file, allow_pickle=True)

    led_count = len(data["led_positions"])
    frame_area = 480 * 800  # pixels
    led_density = led_count / frame_area
    pattern_area = 64 * 64  # pixels per LED pattern

    # Rough estimate of overlap
    coverage_per_led = pattern_area / frame_area
    expected_overlap = led_count * coverage_per_led

    logger.info("\nEstimated overlap analysis:")
    logger.info(f"LED count: {led_count}")
    logger.info(f"Frame area: {frame_area} pixels")
    logger.info(f"Pattern area per LED: {pattern_area} pixels")
    logger.info(f"Expected average overlap: {expected_overlap:.1f}x")

    if expected_overlap > 2:
        suggested_scale = 255 / expected_overlap
        logger.info(
            f"\n💡 RECOMMENDATION: Scale LED patterns by {suggested_scale:.1f} ({1/suggested_scale:.2f}x reduction)"
        )
        logger.info(f"   This would allow up to {expected_overlap:.1f} LEDs to overlap without clipping")
    else:
        logger.info("\n✅ Current scaling may be acceptable (overlap < 2x)")


if __name__ == "__main__":
    logger.info("=== LED BRIGHTNESS AND CLIPPING ANALYSIS ===")

    # Run analysis
    brightness_results = analyze_led_brightness_distribution()
    analyze_pattern_brightness()
    suggest_brightness_scaling()

    logger.info("\n=== ANALYSIS COMPLETE ===")
    logger.info("Use this information to adjust LED pattern brightness scaling if needed")
