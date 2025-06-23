#!/usr/bin/env python3
"""
Debug script to compare planar vs dense LED optimizer results.
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.consumer.led_optimizer_dense import DenseLEDOptimizer
from src.consumer.led_optimizer_planar import PlanarLEDOptimizer
from src.utils.diffusion_pattern_manager import DiffusionPatternManager

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_image():
    """Load a test image."""
    image_path = "env/lib/python3.10/site-packages/sklearn/datasets/images/flower.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Test image not found: {image_path}")

    # Convert BGR to RGB and resize to expected dimensions
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (800, 640))  # W, H

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


def test_dense_optimizer():
    """Test the dense LED optimizer."""
    logger.info("=== Testing Dense LED Optimizer ===")

    # Load test image in interleaved format (H, W, C)
    test_image = load_test_image()  # (640, 800, 3)
    logger.info(f"Test image shape (interleaved): {test_image.shape}")

    # Initialize dense optimizer
    optimizer = DenseLEDOptimizer(
        diffusion_patterns_path="diffusion_patterns/synthetic_1000"
    )

    if not optimizer.initialize():
        raise RuntimeError("Failed to initialize dense optimizer")

    # Optimize
    result = optimizer.optimize_frame(test_image, debug=True, max_iterations=10)

    logger.info(f"Dense optimizer results:")
    logger.info(f"  LED values shape: {result.led_values.shape}")
    logger.info(
        f"  LED values range: [{result.led_values.min()}, {result.led_values.max()}]"
    )
    logger.info(f"  Optimization time: {result.optimization_time:.3f}s")
    logger.info(f"  Iterations: {result.iterations}")
    logger.info(f"  MSE: {result.error_metrics.get('mse', 'N/A')}")

    return result


def test_planar_optimizer():
    """Test the planar LED optimizer."""
    logger.info("=== Testing Planar LED Optimizer ===")

    # Load test image and convert to planar format (C, H, W)
    test_image_interleaved = load_test_image()  # (640, 800, 3)
    test_image_planar = np.transpose(test_image_interleaved, (2, 0, 1))  # (3, 640, 800)
    logger.info(f"Test image shape (planar): {test_image_planar.shape}")

    # Create pattern manager and generate patterns
    pattern_manager = DiffusionPatternManager(
        led_count=1000, frame_height=640, frame_width=800
    )

    # Load patterns from existing file (if available)
    try:
        pattern_manager.load_patterns("diffusion_patterns/synthetic_1000.npz")
        logger.info("Loaded existing patterns")
    except:
        logger.info("Generating synthetic patterns...")
        # Generate synthetic patterns
        pattern_manager.start_pattern_generation()

        for led_id in range(1000):
            # Create simple synthetic pattern
            pattern = np.random.exponential(0.1, (3, 640, 800)).astype(np.float32)
            pattern = np.where(pattern > 0.2, pattern, 0)
            pattern = np.clip(pattern, 0, 1)
            pattern_manager.add_led_pattern(led_id, pattern)

        pattern_manager.finalize_pattern_generation()

    # Initialize planar optimizer
    optimizer = PlanarLEDOptimizer(pattern_manager=pattern_manager)

    if not optimizer.initialize():
        raise RuntimeError("Failed to initialize planar optimizer")

    # Optimize
    result = optimizer.optimize_frame(test_image_planar, debug=True, max_iterations=10)

    logger.info(f"Planar optimizer results:")
    logger.info(f"  LED values shape: {result.led_values.shape}")
    logger.info(
        f"  LED values range: [{result.led_values.min()}, {result.led_values.max()}]"
    )
    logger.info(f"  Optimization time: {result.optimization_time:.3f}s")
    logger.info(f"  Iterations: {result.iterations}")
    logger.info(f"  MSE: {result.error_metrics.get('mse', 'N/A')}")

    return result


def compare_results(dense_result, planar_result):
    """Compare results from both optimizers."""
    logger.info("=== Comparing Results ===")

    # Convert planar result to dense format for comparison
    # Planar: (3, led_count) -> Dense: (led_count, 3)
    planar_led_values_dense_format = np.transpose(planar_result.led_values, (1, 0))

    # Calculate differences
    led_diff = np.abs(dense_result.led_values - planar_led_values_dense_format)
    max_diff = np.max(led_diff)
    mean_diff = np.mean(led_diff)

    logger.info(f"LED value differences:")
    logger.info(f"  Max difference: {max_diff:.3f} (out of 255)")
    logger.info(f"  Mean difference: {mean_diff:.3f}")
    logger.info(f"  Relative max diff: {max_diff/255.0*100:.2f}%")

    # Check if results are similar
    tolerance = 5.0  # Allow 5 units difference out of 255
    if max_diff <= tolerance:
        logger.info("✓ Results are very similar (within tolerance)")
    else:
        logger.warning("⚠ Results differ significantly")

        # Show some example differences
        diff_indices = np.unravel_index(np.argmax(led_diff), led_diff.shape)
        logger.info(
            f"Largest difference at LED {diff_indices[0]}, channel {diff_indices[1]}:"
        )
        logger.info(f"  Dense: {dense_result.led_values[diff_indices]}")
        logger.info(f"  Planar: {planar_led_values_dense_format[diff_indices]}")

    return max_diff <= tolerance


def main():
    logger.info("Starting planar vs dense optimizer comparison...")

    try:
        # Test both optimizers
        dense_result = test_dense_optimizer()
        planar_result = test_planar_optimizer()

        # Compare results
        results_similar = compare_results(dense_result, planar_result)

        if results_similar:
            logger.info("✓ Both optimizers produce similar results")
        else:
            logger.error("✗ Optimizers produce different results - investigate!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
