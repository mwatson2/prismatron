#!/usr/bin/env python3
"""
Standalone tool to compute ATA inverse matrices for diffusion pattern files.

This tool loads a diffusion pattern file, extracts the DIA matrix, computes
the ATA inverse matrices, and saves them back to the file.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

logger = logging.getLogger(__name__)


def compute_ata_inverse_from_dia(
    dia_matrix: DiagonalATAMatrix,
    regularization: float = 1e-6,
    max_condition_number: float = 1e12,
) -> tuple:
    """
    Compute ATA inverse matrices from DIA matrix.

    Args:
        dia_matrix: DiagonalATAMatrix instance
        regularization: Regularization parameter for numerical stability
        max_condition_number: Maximum condition number to accept

    Returns:
        Tuple of (ata_inverse, successful_inversions, condition_numbers, avg_condition_number)
    """
    # TODO: Implement efficient ATA inverse computation
    # This is where you can experiment with different inversion algorithms
    raise NotImplementedError("ATA inverse computation needs to be implemented in standalone utility")


def main():
    parser = argparse.ArgumentParser(description="Compute ATA inverse matrices for diffusion patterns")
    parser.add_argument("pattern_file", help="Path to diffusion pattern file (.npz)")
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help="Regularization parameter (default: 1e-6)",
    )
    parser.add_argument(
        "--max-condition",
        type=float,
        default=1e12,
        help="Maximum condition number before using pseudo-inverse (default: 1e12)",
    )
    parser.add_argument("--backup", action="store_true", help="Create backup of original file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if ATA inverse already exists",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    pattern_file = Path(args.pattern_file)
    if not pattern_file.exists():
        print(f"❌ Pattern file not found: {pattern_file}")
        return 1

    print(f"Processing: {pattern_file}")

    try:
        # Load pattern file
        print("Loading diffusion pattern file...")
        data = np.load(str(pattern_file), allow_pickle=True)

        # Check if ATA inverse already exists
        has_ata_inverse = False
        if "ata_inverse" in data:
            has_ata_inverse = True
            print("✅ ATA inverse already exists in file")
        elif "dense_ata" in data:
            dense_ata_dict = data["dense_ata"].item()
            if "ata_inverse" in dense_ata_dict or "dense_ata_inverse_matrices" in dense_ata_dict:
                has_ata_inverse = True
                print("✅ ATA inverse already exists in dense_ata dictionary")

        if has_ata_inverse and not args.force:
            # Skip prompt in non-interactive mode
            try:
                response = input("ATA inverse already exists. Recompute? (y/N): ")
                if response.lower() != "y":
                    print("Skipping computation.")
                    return 0
            except EOFError:
                # Non-interactive mode - assume yes to recompute
                print("Non-interactive mode detected. Recomputing ATA inverse.")
        elif has_ata_inverse and args.force:
            print("Force flag specified. Recomputing ATA inverse.")

        # Load DIA matrix or dense ATA matrices
        if "dia_matrix" in data:
            print("Loading DIA matrix...")
            dia_dict = data["dia_matrix"].item()
            dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)

            print(f"DIA matrix loaded: {dia_matrix.led_count} LEDs, bandwidth={dia_matrix.bandwidth}")

            # Create backup if requested
            if args.backup:
                backup_path = pattern_file.with_suffix(".bak.npz")
                print(f"Creating backup: {backup_path}")
                import shutil

                shutil.copy2(str(pattern_file), str(backup_path))

            # Compute ATA inverse from DIA matrix
            start_time = time.perf_counter()
            (
                ata_inverse,
                successful_inversions,
                condition_numbers,
                avg_condition_number,
            ) = compute_ata_inverse_from_dia(
                dia_matrix,
                regularization=args.regularization,
                max_condition_number=args.max_condition,
            )
            computation_time = time.perf_counter() - start_time

        elif "dense_ata" in data:
            dense_ata_dict = data["dense_ata"].item()

            if dense_ata_dict and "dense_ata_matrices" in dense_ata_dict:
                print("❌ Dense ATA matrices are deprecated. Please regenerate patterns with DIA matrix support.")
                return 1
            else:
                print("❌ No dense ATA matrices found. Pattern file needs DIA matrix for inverse computation.")
                return 1

        else:
            print("❌ No DIA matrix or dense ATA matrices found in pattern file")
            return 1

        print(f"ATA inverse computation completed in {computation_time:.2f}s")

        # Validate results
        memory_mb = ata_inverse.nbytes / (1024 * 1024)
        print(f"ATA inverse memory: {memory_mb:.1f}MB")
        print(f"ATA inverse shape: {ata_inverse.shape}")
        print(f"ATA inverse dtype: {ata_inverse.dtype}")

        # Save updated file
        print("Saving updated pattern file...")

        # Convert to dictionary for saving
        save_dict = {}
        for key in data:
            save_dict[key] = data[key]

        # Add ATA inverse data
        save_dict["ata_inverse"] = ata_inverse

        # Add metadata about the inverse computation
        ata_inverse_metadata = {
            "computation_time": computation_time,
            "successful_inversions": successful_inversions,
            "condition_numbers": condition_numbers,
            "avg_condition_number": avg_condition_number,
            "regularization": args.regularization,
            "max_condition_number": args.max_condition,
            "timestamp": time.time(),
        }
        save_dict["ata_inverse_metadata"] = ata_inverse_metadata

        # Save file
        np.savez_compressed(str(pattern_file), **save_dict)

        print(f"✅ Successfully saved ATA inverse to {pattern_file}")
        print(f"   Memory usage: {memory_mb:.1f}MB")
        print(f"   Computation time: {computation_time:.2f}s")
        print(f"   Successful inversions: {successful_inversions}/3")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
