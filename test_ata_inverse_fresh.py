#!/usr/bin/env python3
"""
Test ATA inverse computation with fresh pattern file
"""

import sys
from pathlib import Path

import numpy as np

from debug_command_logger import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix


def test_fresh_pattern():
    """Test the fresh pattern file"""
    pattern_file = "/mnt/dev/prismatron/diffusion_patterns/synthetic_1000_fresh.npz"

    logger.log_command("test_fresh_pattern", pattern_file)

    try:
        # Load the pattern file
        logger.log_command("load_pattern", "loading npz file")
        data = np.load(pattern_file, allow_pickle=True)
        logger.log_success("load_pattern", f"Keys: {list(data.files)}")

        # Check if DIA matrix exists
        if "dia_matrix" in data:
            logger.log_command("load_dia_matrix", "extracting DIA matrix")
            dia_dict = data["dia_matrix"].item()
            dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)
            logger.log_success(
                "load_dia_matrix",
                f"LED count: {dia_matrix.led_count}, bandwidth: {dia_matrix.bandwidth}",
            )

            # Check if ATA inverse already exists
            if "ata_inverse" in data:
                logger.log_command("verify_ata_inverse", "checking existing inverse")
                ata_inverse = data["ata_inverse"]
                ata_metadata = data["ata_inverse_metadata"].item()
                logger.log_success(
                    "verify_ata_inverse",
                    f"Shape: {ata_inverse.shape}, metadata: {ata_metadata}",
                )
            else:
                logger.log_error("no_ata_inverse", "ATA inverse not found in pattern file")
                return False

            return True

        else:
            logger.log_error("no_dia_matrix", "DIA matrix not found in pattern file")
            return False

    except Exception as e:
        logger.log_error("test_fresh_pattern", e)
        return False


if __name__ == "__main__":
    success = test_fresh_pattern()
    if success:
        print("✅ Test passed")
    else:
        print("❌ Test failed")
