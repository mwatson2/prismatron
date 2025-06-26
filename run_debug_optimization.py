#!/usr/bin/env python3
"""
Run both optimizations and compare internal A^T@b values.
"""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.consumer.led_optimizer_dense import DenseLEDOptimizer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def run_debug_optimization():
    """Run both optimizations and compare results."""
    logger.warning("=== Debugging Optimization Differences ===")
    
    # Load test image
    img = Image.open('env/lib/python3.10/site-packages/sklearn/datasets/images/flower.jpg')
    img = img.resize((800, 480))
    target_image = np.array(img).astype(np.float32) / 255.0
    
    # CSC optimization with debug
    logger.warning("Running CSC optimization...")
    optimizer_csc = DenseLEDOptimizer(
        diffusion_patterns_path='diffusion_patterns/synthetic_1000_with_ata_clipped.npz',
        use_mixed_tensor=False
    )
    result_csc = optimizer_csc.optimize_frame(target_image, debug=True)
    
    # Mixed tensor optimization with debug  
    logger.warning("Running Mixed tensor optimization...")
    optimizer_mixed = DenseLEDOptimizer(
        diffusion_patterns_path='diffusion_patterns/synthetic_1000_with_ata_clipped.npz',
        use_mixed_tensor=True
    )
    result_mixed = optimizer_mixed.optimize_frame(target_image, debug=True)
    
    # Compare results
    logger.warning("=== Results Comparison ===")
    logger.warning(f"CSC MSE: {result_csc.error_metrics.get('mse', 0):.6f}")
    logger.warning(f"Mixed MSE: {result_mixed.error_metrics.get('mse', 0):.6f}")
    logger.warning(f"CSC iterations: {result_csc.iterations}")
    logger.warning(f"Mixed iterations: {result_mixed.iterations}")
    
    # Compare LED values
    led_diff = np.abs(result_csc.led_values - result_mixed.led_values)
    logger.warning(f"LED values max diff: {led_diff.max()}")
    logger.warning(f"LED values mean diff: {led_diff.mean():.3f}")
    
    # Compare A^T@b after optimization (they should be stored in the internal buffers)
    if hasattr(optimizer_csc, '_ATb_gpu') and hasattr(optimizer_mixed, '_ATb_gpu'):
        atb_csc = optimizer_csc._ATb_gpu.get()
        atb_mixed = optimizer_mixed._ATb_gpu.get()
        
        atb_diff = np.abs(atb_csc - atb_mixed)
        logger.warning(f"A^T@b max diff: {atb_diff.max():.6f}")
        logger.warning(f"A^T@b mean diff: {atb_diff.mean():.6f}")
        
        # Show sample A^T@b values
        logger.warning("Sample A^T@b values (first 3 LEDs):")
        for i in range(3):
            logger.warning(f"LED {i}: CSC={atb_csc[i]}, Mixed={atb_mixed[i]}, Diff={atb_diff[i]}")
    
    return True


if __name__ == "__main__":
    success = run_debug_optimization()
    sys.exit(0 if success else 1)