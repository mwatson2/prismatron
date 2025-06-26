#!/usr/bin/env python3
"""
Debug A^T@b comparison by modifying the standalone optimizer.
"""

import logging
import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from consumer.led_optimizer_dense import DenseLEDOptimizer
from const import FRAME_HEIGHT, FRAME_WIDTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Debug A^T@b comparison."""
    
    # Load test image
    image_path = "env/lib/python3.10/site-packages/sklearn/datasets/images/flower.jpg"
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return 1
    
    # Convert and resize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
    
    logger.info(f"Loaded test image shape: {image.shape}")
    
    # Initialize optimizer with mixed tensor support
    patterns_path = "diffusion_patterns/synthetic_1000_with_ata_clipped"
    optimizer = DenseLEDOptimizer(diffusion_patterns_path=patterns_path, use_mixed_tensor=True)
    
    if not optimizer.initialize():
        logger.error("Failed to initialize optimizer")
        return 1
    
    logger.info("Optimizer initialized successfully")
    
    # Use the comparison method
    try:
        comparison = optimizer.compare_atb_methods(image)
        logger.info("Comparison completed successfully")
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())