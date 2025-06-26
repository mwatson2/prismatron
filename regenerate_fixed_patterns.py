#!/usr/bin/env python3
"""
Regenerate synthetic patterns with fixed spatial ordering.
"""

import logging
import sys
from pathlib import Path

# Add tools to path for imports
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from generate_synthetic_patterns import SyntheticPatternGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Generate 1000 LED patterns with fixed spatial ordering."""
    logger.info("=== Regenerating 1000 LED patterns with fixed spatial ordering ===")
    
    # Create generator with same parameters as original
    generator = SyntheticPatternGenerator(
        frame_width=800,
        frame_height=480,
        seed=42,  # Same seed for reproducibility
        sparsity_threshold=0.01,
    )
    
    # Prepare metadata
    metadata = {
        "pattern_type": "gaussian_multi",
        "seed": 42,
        "intensity_variation": True,
        "spatial_ordering_fixed": True,
        "description": "Fixed spatial ordering - LEDs processed in spatial order, CSC and mixed tensor now consistent"
    }
    
    # Generate sparse patterns using chunked approach with spatial ordering fix
    logger.info("Generating sparse patterns with fixed spatial ordering...")
    sparse_matrix, led_mapping = generator.generate_sparse_patterns_chunked(
        led_count=1000,
        pattern_type="gaussian_multi",
        intensity_variation=True,
        chunk_size=50,
    )
    
    # Save with mixed tensor and A^T@A data
    output_path = "diffusion_patterns/synthetic_1000_with_ata_fixed.npz"
    logger.info(f"Saving complete pattern data to {output_path}...")
    
    success = generator.save_sparse_matrix(
        sparse_matrix=sparse_matrix,
        led_spatial_mapping=led_mapping,
        output_path=output_path,
        metadata=metadata
    )
    
    if success:
        logger.info("✓ Successfully regenerated patterns with fixed spatial ordering")
        logger.info(f"✓ Output saved to: {output_path}")
        
        # Verify the file was created
        file_path = Path(output_path)
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ File size: {file_size:.1f} MB")
        
        return 0
    else:
        logger.error("✗ Failed to regenerate patterns")
        return 1

if __name__ == "__main__":
    sys.exit(main())