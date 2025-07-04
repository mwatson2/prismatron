#!/usr/bin/env python3
"""
Safe test script for ATA inverse computation with command logging
"""

import os
import sys
import traceback
from pathlib import Path

import numpy as np

from debug_command_logger import logger, safe_execute

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def test_pattern_file_loading(pattern_file_path):
    """Test loading a pattern file"""
    logger.log_command("load_pattern_file", pattern_file_path)

    try:
        if not Path(pattern_file_path).exists():
            logger.log_error("file_not_found", f"Pattern file not found: {pattern_file_path}")
            return None

        data = np.load(pattern_file_path, allow_pickle=True)
        logger.log_success("load_pattern_file", f"Loaded {len(data.files)} keys")

        # Log available keys
        logger.log_entry(f"Pattern file keys: {list(data.files)}")

        return data

    except Exception as e:
        logger.log_error("load_pattern_file", e)
        return None


def test_dia_matrix_extraction(data):
    """Test extracting DIA matrix from pattern data"""
    logger.log_command("extract_dia_matrix", "checking for dia_matrix key")

    try:
        if "dia_matrix" not in data:
            logger.log_error("dia_matrix_missing", "No dia_matrix key found in pattern file")
            return None

        dia_dict = data["dia_matrix"].item()
        logger.log_success("extract_dia_matrix", f"DIA dict keys: {list(dia_dict.keys())}")

        # Import DiagonalATAMatrix
        from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

        dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)
        logger.log_success(
            "create_dia_matrix",
            f"LED count: {dia_matrix.led_count}, bandwidth: {dia_matrix.bandwidth}",
        )

        return dia_matrix

    except Exception as e:
        logger.log_error("extract_dia_matrix", e)
        return None


def test_ata_inverse_computation(dia_matrix):
    """Test ATA inverse computation"""
    logger.log_command("compute_ata_inverse", f"LED count: {dia_matrix.led_count}")

    try:
        # Use small regularization for testing
        regularization = 1e-6
        max_condition_number = 1e12

        logger.log_entry(f"Using regularization={regularization}, max_condition={max_condition_number}")

        # Call the ATA inverse computation
        ata_inverse, successful_inversions, condition_numbers, avg_condition_number = dia_matrix.compute_ata_inverse(
            regularization=regularization, max_condition_number=max_condition_number
        )

        logger.log_success(
            "compute_ata_inverse",
            f"Shape: {ata_inverse.shape}, successful: {successful_inversions}/3",
        )
        logger.log_entry(f"Condition numbers: {condition_numbers}")
        logger.log_entry(f"Average condition number: {avg_condition_number}")

        return (
            ata_inverse,
            successful_inversions,
            condition_numbers,
            avg_condition_number,
        )

    except Exception as e:
        logger.log_error("compute_ata_inverse", e)
        return None, 0, [], float("inf")


def test_specific_pattern_file(pattern_file_path):
    """Test a specific pattern file end-to-end"""
    logger.log_entry(f"=== TESTING PATTERN FILE: {pattern_file_path} ===")

    # Test loading
    data = test_pattern_file_loading(pattern_file_path)
    if data is None:
        logger.log_entry("❌ Failed to load pattern file")
        return False

    # Test DIA matrix extraction
    dia_matrix = test_dia_matrix_extraction(data)
    if dia_matrix is None:
        logger.log_entry("❌ Failed to extract DIA matrix")
        return False

    # Test ATA inverse computation
    ata_inverse, successful_inversions, condition_numbers, avg_condition_number = test_ata_inverse_computation(
        dia_matrix
    )
    if ata_inverse is None:
        logger.log_entry("❌ Failed to compute ATA inverse")
        return False

    logger.log_entry("✅ All tests passed for this pattern file")
    return True


def main():
    """Main test function"""
    logger.log_entry("=== STARTING SAFE ATA INVERSE TESTING ===")

    # Test pattern files that have DIA matrices
    test_files = [
        "/mnt/dev/prismatron/diffusion_patterns/synthetic_1000.npz",
        "/mnt/dev/prismatron/diffusion_patterns/synthetic_1000_aligned.npz",
        "/mnt/dev/prismatron/diffusion_patterns/synthetic_1000_64x64_fixed.npz",
        "/mnt/dev/prismatron/diffusion_patterns/synthetic_100.npz",
    ]

    for pattern_file in test_files:
        try:
            logger.log_entry(f"Testing pattern file: {pattern_file}")
            if Path(pattern_file).exists():
                success = test_specific_pattern_file(pattern_file)
                if not success:
                    logger.log_entry(f"❌ Test failed for {pattern_file}")
                else:
                    logger.log_entry(f"✅ Test passed for {pattern_file}")
            else:
                logger.log_entry(f"⚠️ Pattern file not found: {pattern_file}")
        except Exception as e:
            logger.log_crash(f"Testing {pattern_file}")
            logger.log_error("main_loop", e)

    logger.log_entry("=== TESTING COMPLETE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.log_crash("main execution")
        print(f"Critical error: {e}")
        traceback.print_exc()

    print("\nTest completed. Check debug_commands.log for detailed execution trace.")
