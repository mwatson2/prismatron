#!/usr/bin/env python3
"""
Compare A^T@b values between CSC and mixed tensor approaches.
"""

import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse as sp
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.consumer.led_optimizer_dense import DenseLEDOptimizer

logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)


def compare_atb_values():
    """Compare A^T@b values between approaches."""
    logger.warning("=== Comparing A^T@b Values ===")
    
    # Load test image
    img = Image.open('env/lib/python3.10/site-packages/sklearn/datasets/images/flower.jpg')
    img = img.resize((800, 480))
    target_image = np.array(img).astype(np.float32) / 255.0
    
    # Create optimizers but don't run full optimization
    optimizer_csc = DenseLEDOptimizer(
        diffusion_patterns_path='diffusion_patterns/synthetic_1000_with_ata_clipped.npz',
        use_mixed_tensor=False
    )
    optimizer_mixed = DenseLEDOptimizer(
        diffusion_patterns_path='diffusion_patterns/synthetic_1000_with_ata_clipped.npz',
        use_mixed_tensor=True
    )
    
    # Manually call the A^T@b calculation to get the values
    logger.warning("Computing CSC A^T@b...")
    atb_csc = optimizer_csc._calculate_ATb(target_image).get()
    
    logger.warning("Computing Mixed tensor A^T@b...")
    atb_mixed = optimizer_mixed._calculate_ATb(target_image).get()
    
    # Compare the A^T@b values
    logger.warning("=== A^T@b Comparison ===")
    logger.warning(f"CSC A^T@b shape: {atb_csc.shape}")
    logger.warning(f"Mixed A^T@b shape: {atb_mixed.shape}")
    
    logger.warning(f"CSC A^T@b range: [{atb_csc.min():.3f}, {atb_csc.max():.3f}]")
    logger.warning(f"Mixed A^T@b range: [{atb_mixed.min():.3f}, {atb_mixed.max():.3f}]")
    
    logger.warning(f"CSC A^T@b mean: {atb_csc.mean():.3f}")
    logger.warning(f"Mixed A^T@b mean: {atb_mixed.mean():.3f}")
    
    # Calculate differences
    diff = np.abs(atb_csc - atb_mixed)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    logger.warning(f"Max A^T@b difference: {max_diff:.6f}")
    logger.warning(f"Mean A^T@b difference: {mean_diff:.6f}")
    logger.warning(f"Relative difference: {max_diff / atb_csc.max():.6f}")
    
    # Show sample differences
    logger.warning("Sample A^T@b values (first 5 LEDs):")
    for i in range(5):
        logger.warning(f"LED {i}: CSC={atb_csc[i]}, Mixed={atb_mixed[i]}, Diff={diff[i]}")
    
    # Find largest differences
    max_diff_indices = np.unravel_index(np.argmax(diff), diff.shape)
    led_id, channel = max_diff_indices
    logger.warning(f"Largest difference at LED {led_id}, channel {channel}:")
    logger.warning(f"  CSC: {atb_csc[led_id, channel]:.6f}")
    logger.warning(f"  Mixed: {atb_mixed[led_id, channel]:.6f}")
    logger.warning(f"  Difference: {diff[led_id, channel]:.6f}")
    
    # Check if differences are significant
    significant_diff = max_diff > 1.0  # A^T@b values are typically in hundreds
    return not significant_diff


if __name__ == "__main__":
    success = compare_atb_values()
    if success:
        print("✅ A^T@b values are similar")
    else:
        print("❌ A^T@b values differ significantly")
    sys.exit(0 if success else 1)