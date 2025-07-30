#!/usr/bin/env python3
"""
Compare Pattern Brightness: Captured vs Synthetic.

Compare brightness levels between captured and synthetic patterns to identify
scaling differences between DIA and dense implementations.
"""

import logging

# Set up path
import sys
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.dense_ata_matrix import DenseATAMatrix
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_pattern_file(pattern_file: str, name: str):
    """Analyze a pattern file for brightness characteristics."""

    if not Path(pattern_file).exists():
        logger.error(f"Pattern file not found: {pattern_file}")
        return None

    logger.info(f"\n=== ANALYZING {name} ===")
    logger.info(f"File: {pattern_file}")

    data = np.load(pattern_file, allow_pickle=True)
    logger.info(f"Available keys: {list(data.keys())}")

    # Load mixed tensor
    mixed_tensor_data = data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_data)
    logger.info(f"Mixed tensor: {mixed_tensor}")

    # Analyze pattern brightness
    sample_blocks = []
    max_samples = 50

    for led_idx in range(min(max_samples, mixed_tensor.batch_size)):
        for channel in range(3):
            block = mixed_tensor.sparse_values[channel, led_idx]
            if hasattr(block, "get"):  # CuPy array
                block_np = cp.asnumpy(block)
            else:
                block_np = block

            if np.max(block_np) > 0:
                sample_blocks.append(block_np)

    if sample_blocks:
        all_values = np.concatenate([block.flatten() for block in sample_blocks])
        non_zero_values = all_values[all_values > 0]

        pattern_stats = {
            "file": pattern_file,
            "name": name,
            "led_count": mixed_tensor.batch_size,
            "dtype": mixed_tensor.dtype,
            "blocks_sampled": len(sample_blocks),
            "min_value": float(non_zero_values.min()),
            "max_value": float(non_zero_values.max()),
            "mean_value": float(non_zero_values.mean()),
            "median_value": float(np.median(non_zero_values)),
            "std_value": float(non_zero_values.std()),
            "brightness_ratio": float(non_zero_values.max()) / (255 if mixed_tensor.dtype == np.uint8 else 1.0),
        }

        logger.info(f"Pattern brightness analysis:")
        logger.info(f"  LED count: {pattern_stats['led_count']}")
        logger.info(f"  Data type: {pattern_stats['dtype']}")
        logger.info(f"  Value range: [{pattern_stats['min_value']:.1f}, {pattern_stats['max_value']:.1f}]")
        logger.info(f"  Mean: {pattern_stats['mean_value']:.1f}, Median: {pattern_stats['median_value']:.1f}")
        logger.info(f"  Std: {pattern_stats['std_value']:.1f}")
        logger.info(f"  Brightness ratio: {pattern_stats['brightness_ratio']:.3f}")

        return pattern_stats
    else:
        logger.error("No non-zero blocks found")
        return None


def test_optimization_with_both_formats(pattern_file: str, name: str):
    """Test optimization with both DIA and dense formats if available."""

    data = np.load(pattern_file, allow_pickle=True)

    # Load mixed tensor
    mixed_tensor_data = data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_data)

    # Test frame
    test_frame = cp.ones((3, 480, 800), dtype=cp.uint8) * 128  # Mid-gray
    simple_init = np.ones((3, mixed_tensor.batch_size), dtype=np.float32) * 0.1

    logger.info(f"\n=== OPTIMIZATION TEST: {name} ===")

    results = {}

    # Test with DIA matrix if available
    if "dia_matrix" in data:
        logger.info("Testing with DIA matrix...")
        dia_data = data["dia_matrix"].item()
        dia_matrix = DiagonalATAMatrix.from_dict(dia_data)

        try:
            result = optimize_frame_led_values(
                target_frame=test_frame,
                at_matrix=mixed_tensor,
                ata_matrix=dia_matrix,
                ata_inverse=data.get("ata_inverse", None),
                initial_values=simple_init,
                max_iterations=5,
                compute_error_metrics=False,
                debug=False,
            )

            led_values_cpu = cp.asnumpy(result.led_values)
            clipped_high = np.sum(led_values_cpu == 255)
            clipped_low = np.sum(led_values_cpu == 0)
            total_clipped = clipped_high + clipped_low
            clipped_percent = total_clipped / led_values_cpu.size * 100

            results["dia"] = {
                "led_min": led_values_cpu.min(),
                "led_max": led_values_cpu.max(),
                "led_mean": led_values_cpu.mean(),
                "led_std": led_values_cpu.std(),
                "clipped_high": clipped_high,
                "clipped_low": clipped_low,
                "clipped_percent": clipped_percent,
            }

            logger.info(
                f"DIA results: range=[{results['dia']['led_min']:.1f}, {results['dia']['led_max']:.1f}], mean={results['dia']['led_mean']:.1f}, clipped={clipped_percent:.1f}%"
            )

        except Exception as e:
            logger.error(f"DIA optimization failed: {e}")

    # Test with dense matrix if available
    if "dense_ata_matrix" in data:
        logger.info("Testing with dense matrix...")
        dense_ata_data = data["dense_ata_matrix"].item()
        dense_ata = DenseATAMatrix.from_dict(dense_ata_data)

        try:
            result = optimize_frame_led_values(
                target_frame=test_frame,
                at_matrix=mixed_tensor,
                ata_matrix=dense_ata,
                ata_inverse=data.get("ata_inverse", None),
                initial_values=simple_init,
                max_iterations=5,
                compute_error_metrics=False,
                debug=False,
            )

            led_values_cpu = cp.asnumpy(result.led_values)
            clipped_high = np.sum(led_values_cpu == 255)
            clipped_low = np.sum(led_values_cpu == 0)
            total_clipped = clipped_high + clipped_low
            clipped_percent = total_clipped / led_values_cpu.size * 100

            results["dense"] = {
                "led_min": led_values_cpu.min(),
                "led_max": led_values_cpu.max(),
                "led_mean": led_values_cpu.mean(),
                "led_std": led_values_cpu.std(),
                "clipped_high": clipped_high,
                "clipped_low": clipped_low,
                "clipped_percent": clipped_percent,
            }

            logger.info(
                f"Dense results: range=[{results['dense']['led_min']:.1f}, {results['dense']['led_max']:.1f}], mean={results['dense']['led_mean']:.1f}, clipped={clipped_percent:.1f}%"
            )

        except Exception as e:
            logger.error(f"Dense optimization failed: {e}")

    return results


def compare_files():
    """Compare captured vs synthetic pattern files."""

    files_to_compare = [
        ("diffusion_patterns/synthetic_2624_uint8.npz", "Synthetic 2624"),
        ("diffusion_patterns/capture-0728-01-dense_fixed.npz", "Captured (Fixed)"),
    ]

    # If original captured file exists, include it too
    if Path("diffusion_patterns/capture-0728-01.npz").exists():
        files_to_compare.append(("diffusion_patterns/capture-0728-01.npz", "Captured (Original)"))

    pattern_results = {}
    optimization_results = {}

    for file_path, name in files_to_compare:
        # Analyze pattern brightness
        pattern_stats = analyze_pattern_file(file_path, name)
        if pattern_stats:
            pattern_results[name] = pattern_stats

        # Test optimization behavior
        opt_results = test_optimization_with_both_formats(file_path, name)
        if opt_results:
            optimization_results[name] = opt_results

    # Compare results
    logger.info(f"\n=== BRIGHTNESS COMPARISON SUMMARY ===")

    for name, stats in pattern_results.items():
        logger.info(
            f"{name}: max={stats['max_value']:.1f}, mean={stats['mean_value']:.1f}, ratio={stats['brightness_ratio']:.3f}"
        )

    logger.info(f"\n=== OPTIMIZATION COMPARISON SUMMARY ===")

    for name, results in optimization_results.items():
        logger.info(f"\n{name}:")
        for format_type, stats in results.items():
            logger.info(
                f"  {format_type.upper()}: range=[{stats['led_min']:.1f},{stats['led_max']:.1f}], mean={stats['led_mean']:.1f}, clipped={stats['clipped_percent']:.1f}%"
            )

    # Check for scaling differences
    logger.info(f"\n=== SCALING ANALYSIS ===")

    # Compare pattern brightness
    if "Synthetic 2624" in pattern_results and "Captured (Fixed)" in pattern_results:
        synthetic_max = pattern_results["Synthetic 2624"]["max_value"]
        captured_max = pattern_results["Captured (Fixed)"]["max_value"]
        brightness_ratio = captured_max / synthetic_max

        logger.info(f"Pattern brightness ratio (Captured/Synthetic): {brightness_ratio:.2f}x")

        if brightness_ratio > 2:
            logger.warning(
                f"❌ Captured patterns are {brightness_ratio:.1f}x brighter than synthetic - this could cause clipping"
            )
        elif abs(brightness_ratio - 1.0) < 0.1:
            logger.info(f"✅ Pattern brightness is similar - scaling issue may be elsewhere")

    # Compare optimization behavior between DIA and dense
    for name in optimization_results:
        results = optimization_results[name]
        if "dia" in results and "dense" in results:
            dia_clipped = results["dia"]["clipped_percent"]
            dense_clipped = results["dense"]["clipped_percent"]
            clipping_ratio = dense_clipped / dia_clipped if dia_clipped > 0 else float("inf")

            logger.info(
                f"{name}: DIA clipping={dia_clipped:.1f}%, Dense clipping={dense_clipped:.1f}% (ratio={clipping_ratio:.2f}x)"
            )

            if clipping_ratio > 1.5:
                logger.warning(
                    f"❌ Dense format clips {clipping_ratio:.1f}x more than DIA - scaling issue in dense implementation"
                )
            elif abs(clipping_ratio - 1.0) < 0.2:
                logger.info(f"✅ Similar clipping behavior between DIA and dense formats")


if __name__ == "__main__":
    logger.info("=== COMPARING PATTERN BRIGHTNESS: CAPTURED vs SYNTHETIC ===")
    compare_files()
    logger.info("\n=== COMPARISON COMPLETE ===")
