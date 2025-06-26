#!/usr/bin/env python3
"""
Compare LED values between mixed tensor and CSC approaches.
"""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.consumer.led_optimizer_dense import DenseLEDOptimizer

logging.basicConfig(level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)


def compare_led_values():
    """Compare LED values from mixed tensor vs CSC approaches."""
    logger.warning("=== Comparing LED Values Between Mixed Tensor and CSC ===")
    
    # Load test image
    img = Image.open('env/lib/python3.10/site-packages/sklearn/datasets/images/flower.jpg')
    img = img.resize((800, 480))
    target_image = np.array(img).astype(np.float32) / 255.0
    
    # Test CSC approach
    logger.warning("Testing CSC approach...")
    optimizer_csc = DenseLEDOptimizer(
        diffusion_patterns_path='diffusion_patterns/synthetic_1000_with_ata_clipped.npz',
        use_mixed_tensor=False
    )
    result_csc = optimizer_csc.optimize_frame(target_image)
    led_values_csc = result_csc.led_values
    
    # Test mixed tensor approach
    logger.warning("Testing mixed tensor approach...")
    optimizer_mixed = DenseLEDOptimizer(
        diffusion_patterns_path='diffusion_patterns/synthetic_1000_with_ata_clipped.npz',
        use_mixed_tensor=True
    )
    result_mixed = optimizer_mixed.optimize_frame(target_image)
    led_values_mixed = result_mixed.led_values
    
    # Compare LED values
    logger.warning("=== LED Values Comparison ===")
    logger.warning(f"CSC LED values shape: {led_values_csc.shape}")
    logger.warning(f"Mixed LED values shape: {led_values_mixed.shape}")
    
    logger.warning(f"CSC LED values range: [{led_values_csc.min()}, {led_values_csc.max()}]")
    logger.warning(f"Mixed LED values range: [{led_values_mixed.min()}, {led_values_mixed.max()}]")
    
    logger.warning(f"CSC LED values mean: {led_values_csc.mean():.3f}")
    logger.warning(f"Mixed LED values mean: {led_values_mixed.mean():.3f}")
    
    # Calculate differences
    diff = np.abs(led_values_csc - led_values_mixed)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    logger.warning(f"Max LED value difference: {max_diff}")
    logger.warning(f"Mean LED value difference: {mean_diff}")
    logger.warning(f"Relative difference: {max_diff / led_values_csc.max():.6f}")
    
    # Show sample differences
    logger.warning("Sample LED values (first 5 LEDs):")
    for i in range(5):
        logger.warning(f"LED {i}: CSC={led_values_csc[i]}, Mixed={led_values_mixed[i]}, Diff={diff[i]}")
    
    # Check optimization metrics
    logger.warning("=== Optimization Metrics ===")
    logger.warning(f"CSC - Converged: {result_csc.converged}, Iterations: {result_csc.iterations}, MSE: {result_csc.error_metrics.get('mse', 0):.6f}")
    logger.warning(f"Mixed - Converged: {result_mixed.converged}, Iterations: {result_mixed.iterations}, MSE: {result_mixed.error_metrics.get('mse', 0):.6f}")
    
    # Return success if differences are small
    success = max_diff < 5.0  # LED values are in range [0, 255], so 5.0 is about 2%
    return success


if __name__ == "__main__":
    success = compare_led_values()
    if success:
        print("✅ LED values are similar")
    else:
        print("❌ LED values differ significantly")
    sys.exit(0 if success else 1)