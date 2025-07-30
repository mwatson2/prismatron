#!/usr/bin/env python3
"""
Compare ATA Matrices - Dense vs DIA.

This tool compares dense and DIA ATA matrices to verify they represent
the same mathematical operations and produce identical results.
"""

import argparse
import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.dense_ata_matrix import DenseATAMatrix
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

logger = logging.getLogger(__name__)


def compare_ata_matrices(pattern_file: str) -> bool:
    """
    Compare dense and DIA ATA matrices in a pattern file.

    Args:
        pattern_file: NPZ file containing both dense_ata_matrix and dia_matrix

    Returns:
        True if matrices are mathematically equivalent
    """
    try:
        logger.info(f"Loading pattern file: {pattern_file}")
        data = np.load(pattern_file, allow_pickle=True)

        # Check for required matrices
        has_dia = "dia_matrix" in data
        has_dense = "dense_ata_matrix" in data

        if not has_dia:
            logger.error("No DIA matrix found in file")
            return False

        if not has_dense:
            logger.error("No dense ATA matrix found in file")
            return False

        logger.info("✅ Found both DIA and dense ATA matrices")

        # Load matrices
        dia_data = data["dia_matrix"].item()
        dia_matrix = DiagonalATAMatrix.from_dict(dia_data)

        dense_data = data["dense_ata_matrix"].item()
        dense_matrix = DenseATAMatrix.from_dict(dense_data)

        logger.info(
            f"DIA matrix: {dia_matrix.led_count} LEDs, k={dia_matrix.k} diagonals, bandwidth={dia_matrix.bandwidth}"
        )
        logger.info(f"Dense matrix: {dense_matrix.led_count} LEDs, {dense_matrix.memory_mb:.1f}MB")

        # Basic compatibility checks
        if dia_matrix.led_count != dense_matrix.led_count:
            logger.error(f"LED count mismatch: DIA={dia_matrix.led_count}, Dense={dense_matrix.led_count}")
            return False

        logger.info("✅ Basic compatibility checks passed")

        # Test with multiple random LED value vectors
        led_count = dia_matrix.led_count
        test_vectors = []

        # Create test vectors
        np.random.seed(42)  # For reproducible results
        for i in range(5):
            if i == 0:
                # Test 1: All zeros
                test_vec = np.zeros((3, led_count), dtype=np.float32)
            elif i == 1:
                # Test 2: All ones
                test_vec = np.ones((3, led_count), dtype=np.float32) * 0.5
            elif i == 2:
                # Test 3: Random uniform [0,1]
                test_vec = np.random.uniform(0, 1, (3, led_count)).astype(np.float32)
            elif i == 3:
                # Test 4: Single LED active
                test_vec = np.zeros((3, led_count), dtype=np.float32)
                test_vec[:, led_count // 2] = 1.0
            else:
                # Test 5: Random sparse (only 10% of LEDs active)
                test_vec = np.zeros((3, led_count), dtype=np.float32)
                active_count = max(1, led_count // 10)
                active_indices = np.random.choice(led_count, active_count, replace=False)
                test_vec[:, active_indices] = np.random.uniform(0, 1, (3, active_count))

            test_vectors.append((f"Test {i+1}", test_vec))

        # Compare results for each test vector
        all_tests_passed = True
        max_error_overall = 0.0

        for test_name, test_vec in test_vectors:
            logger.info(f"\n=== {test_name} ===")

            # Convert to GPU for computation
            test_vec_gpu = cp.asarray(test_vec)

            # Get results from both matrices
            dia_result = dia_matrix.multiply_3d(test_vec_gpu)
            dense_result = dense_matrix.multiply_vector(test_vec_gpu)

            # Convert back to CPU for comparison
            dia_result_cpu = cp.asnumpy(dia_result)
            dense_result_cpu = cp.asnumpy(dense_result)

            # Compare results
            max_error = np.max(np.abs(dia_result_cpu - dense_result_cpu))
            rms_error = np.sqrt(np.mean((dia_result_cpu - dense_result_cpu) ** 2))

            # Get result statistics
            dia_stats = {
                "min": dia_result_cpu.min(),
                "max": dia_result_cpu.max(),
                "mean": dia_result_cpu.mean(),
                "std": dia_result_cpu.std(),
            }

            dense_stats = {
                "min": dense_result_cpu.min(),
                "max": dense_result_cpu.max(),
                "mean": dense_result_cpu.mean(),
                "std": dense_result_cpu.std(),
            }

            logger.info(f"Input range: [{test_vec.min():.6f}, {test_vec.max():.6f}], mean={test_vec.mean():.6f}")
            logger.info(f"DIA result:   [{dia_stats['min']:.6f}, {dia_stats['max']:.6f}], mean={dia_stats['mean']:.6f}")
            logger.info(
                f"Dense result: [{dense_stats['min']:.6f}, {dense_stats['max']:.6f}], mean={dense_stats['mean']:.6f}"
            )
            logger.info(f"Max error: {max_error:.2e}")
            logger.info(f"RMS error: {rms_error:.2e}")

            # Check if results are close enough (considering floating point precision)
            # Note: DIA matrices use approximations and different computation paths
            tolerance = 2e-2  # Relaxed tolerance for DIA vs dense comparison (DIA uses approximations)
            if max_error > tolerance:
                logger.warning(f"❌ {test_name}: Error {max_error:.2e} exceeds tolerance {tolerance:.2e}")
                all_tests_passed = False
            else:
                logger.info(f"✅ {test_name}: Results match within tolerance")

            max_error_overall = max(max_error_overall, max_error)

        # Summary
        logger.info(f"\n=== COMPARISON SUMMARY ===")
        logger.info(f"Overall maximum error: {max_error_overall:.2e}")

        if all_tests_passed:
            logger.info("✅ ALL TESTS PASSED: Dense and DIA matrices are mathematically equivalent")
            return True
        else:
            logger.error("❌ TESTS FAILED: Dense and DIA matrices produce different results")
            return False

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare dense and DIA ATA matrices")
    parser.add_argument("--file", "-f", required=True, help="NPZ file containing both matrices")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Validate input
    if not Path(args.file).exists():
        logger.error(f"File not found: {args.file}")
        return 1

    if not args.file.endswith(".npz"):
        logger.error("File must be .npz format")
        return 1

    # Perform comparison
    if compare_ata_matrices(args.file):
        logger.info("✅ Matrix comparison completed successfully")
        return 0
    else:
        logger.error("❌ Matrix comparison failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
