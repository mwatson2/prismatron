#!/usr/bin/env python3
"""
Test that mixed tensor and CSC modes produce similar optimization results.
"""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_images(image1_path: str, image2_path: str) -> dict:
    """Compare two images and return similarity metrics."""
    
    # Load images
    img1 = np.array(Image.open(image1_path))
    img2 = np.array(Image.open(image2_path))
    
    # Ensure same shape
    if img1.shape != img2.shape:
        logger.error(f"Image shapes don't match: {img1.shape} vs {img2.shape}")
        return {"error": "Shape mismatch"}
    
    # Convert to float for calculations
    img1_f = img1.astype(np.float32) / 255.0
    img2_f = img2.astype(np.float32) / 255.0
    
    # Calculate metrics
    mse = np.mean((img1_f - img2_f) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(img1_f - img2_f))
    max_diff = np.max(np.abs(img1_f - img2_f))
    
    # Calculate PSNR
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Calculate structural similarity (simple version)
    mean1 = np.mean(img1_f)
    mean2 = np.mean(img2_f)
    var1 = np.var(img1_f)
    var2 = np.var(img2_f)
    cov = np.mean((img1_f - mean1) * (img2_f - mean2))
    
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    
    ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
           ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_diff": max_diff,
        "psnr": psnr,
        "ssim": ssim
    }


def test_mixed_vs_csc_final():
    """Test that mixed tensor and CSC produce similar results."""
    logger.info("=== Testing Mixed Tensor vs CSC Final Results ===")
    
    # Check if output files exist
    mixed_path = "test_images/flower_test_mixed_fixed.png"
    csc_path = "test_images/flower_test_csc_fixed.png"
    
    if not Path(mixed_path).exists():
        logger.error(f"Mixed tensor output not found: {mixed_path}")
        return False
        
    if not Path(csc_path).exists():
        logger.error(f"CSC output not found: {csc_path}")
        return False
    
    # Compare the images
    metrics = compare_images(mixed_path, csc_path)
    
    if "error" in metrics:
        logger.error(f"Image comparison failed: {metrics['error']}")
        return False
    
    # Log metrics
    logger.info("=== Image Similarity Metrics ===")
    logger.info(f"MSE: {metrics['mse']:.6f}")
    logger.info(f"RMSE: {metrics['rmse']:.6f}")
    logger.info(f"MAE: {metrics['mae']:.6f}")
    logger.info(f"Max difference: {metrics['max_diff']:.6f}")
    logger.info(f"PSNR: {metrics['psnr']:.2f} dB")
    logger.info(f"SSIM: {metrics['ssim']:.4f}")
    
    # Define success criteria
    success_criteria = {
        "rmse": 0.1,      # Root mean square error < 10%
        "psnr": 20.0,     # Peak SNR > 20 dB
        "ssim": 0.8       # Structural similarity > 0.8
    }
    
    # Check success
    success = (
        metrics['rmse'] < success_criteria['rmse'] and
        metrics['psnr'] > success_criteria['psnr'] and
        metrics['ssim'] > success_criteria['ssim']
    )
    
    if success:
        logger.info("✅ SUCCESS: Mixed tensor and CSC produce very similar results!")
        logger.info("The mixed tensor optimization is working correctly with clipping equivalence.")
        return True
    else:
        logger.error("❌ FAILURE: Mixed tensor and CSC produce different results")
        logger.error("There may still be an issue with the mixed tensor implementation.")
        
        # Show which criteria failed
        if metrics['rmse'] >= success_criteria['rmse']:
            logger.error(f"RMSE too high: {metrics['rmse']:.6f} >= {success_criteria['rmse']}")
        if metrics['psnr'] <= success_criteria['psnr']:
            logger.error(f"PSNR too low: {metrics['psnr']:.2f} <= {success_criteria['psnr']}")
        if metrics['ssim'] <= success_criteria['ssim']:
            logger.error(f"SSIM too low: {metrics['ssim']:.4f} <= {success_criteria['ssim']}")
        
        return False


if __name__ == "__main__":
    success = test_mixed_vs_csc_final()
    sys.exit(0 if success else 1)