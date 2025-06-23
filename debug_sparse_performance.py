#!/usr/bin/env python3
"""
Debug script to analyze sparse matrix performance bottlenecks.
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.consumer.led_optimizer_dense import DenseLEDOptimizer

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting sparse matrix performance analysis...")

    # Initialize optimizer
    optimizer = DenseLEDOptimizer(
        diffusion_patterns_path="diffusion_patterns/synthetic_1000"
    )

    if not optimizer.initialize():
        logger.error("Failed to initialize optimizer")
        return

    # Create test frame with correct dimensions for the loaded matrix
    # The matrix expects 384,000 pixels = 640×600 (not 640×800)
    test_frame = np.random.randint(0, 255, (640, 600, 3), dtype=np.uint8)

    # Run detailed performance analysis
    results = optimizer.debug_sparse_performance(test_frame)

    logger.info("=== Summary ===")
    fastest = min(results.values())
    slowest = max(v for v in results.values() if v > 0)

    logger.info(f"Fastest method: {fastest:.4f}s")
    logger.info(f"Slowest method: {slowest:.4f}s")
    logger.info(f"Performance ratio: {slowest/fastest:.1f}x")

    # Recommendations
    if (
        results.get("individual_channels", 0) > 0
        and results.get("combined_block_diagonal", 0) > 0
    ):
        if results["individual_channels"] < results["combined_block_diagonal"] * 0.8:
            logger.info(
                "RECOMMENDATION: Consider switching to individual channel operations"
            )
        else:
            logger.info("Current combined approach is optimal")

    if (
        results.get("csr_format", 0) > 0
        and results.get("combined_block_diagonal", 0) > 0
    ):
        if results["csr_format"] < results["combined_block_diagonal"] * 0.8:
            logger.info("RECOMMENDATION: Consider using CSR format instead of CSC")

    # BSR real format analysis
    if results.get("bsr_real_method", 0) > 0:
        logger.info(f"BSR real method timing: {results['bsr_real_method']:.4f}s")
        logger.info(
            "Note: BSR uses small sample (96 LEDs vs 1000 CSC LEDs) and CPU operation"
        )

    if (
        results.get("bsr_real_method", 0) > 0
        and results.get("combined_block_diagonal", 0) > 0
    ):
        # Note: This comparison is not directly meaningful due to different matrix sizes
        ratio = results["bsr_real_method"] / results["combined_block_diagonal"]
        logger.info(
            f"BSR sample vs full CSC ratio: {ratio:.2f}x (not directly comparable due to size difference)"
        )
        if ratio < 0.5:
            logger.info(
                "BSR sample operation is faster (expected due to much smaller size)"
            )
        else:
            logger.info(
                "BSR sample shows framework overhead (expected for small operations)"
            )

    if results.get("bsr_cusparse_repeated", 0) > 0:
        logger.info(f"BSR repeated operations: {results['bsr_cusparse_repeated']:.4f}s")

    # Cache effects analysis
    if (
        results.get("repeated_operations", 0) > 0
        and results.get("combined_block_diagonal", 0) > 0
    ):
        speedup = results["combined_block_diagonal"] / results["repeated_operations"]
        logger.info(f"Cache warming speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
