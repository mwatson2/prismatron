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
    output_fp16: bool = False,
) -> tuple:
    """
    Compute ATA inverse matrices from DIA matrix.

    Args:
        dia_matrix: DiagonalATAMatrix instance
        regularization: Regularization parameter for numerical stability
        max_condition_number: Maximum condition number to accept
        output_fp16: Whether to output FP16 format (if input ATA matrix is FP16)

    Returns:
        Tuple of (ata_inverse, successful_inversions, condition_numbers, avg_condition_number)
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    print(f"Computing ATA inverse from DIA format: {dia_matrix.led_count} LEDs, {dia_matrix.k} diagonals")

    # Detect input format and decide output format
    # Check storage dtype for FP16 storage detection
    input_storage_fp16 = hasattr(dia_matrix, "storage_dtype") and dia_matrix.storage_dtype == np.float16
    input_output_fp16 = dia_matrix.output_dtype == np.float16

    # Use FP16 output if storage is FP16, output is FP16, or explicitly requested
    use_fp16_output = input_storage_fp16 or input_output_fp16 or output_fp16
    output_dtype = np.float16 if use_fp16_output else np.float32

    if input_storage_fp16:
        print("  Detected FP16 storage ATA matrix - will output FP16 inverse")
    elif input_output_fp16:
        print("  Detected FP16 output ATA matrix - will output FP16 inverse")
    elif output_fp16:
        print("  Using FP16 output format as requested")
    else:
        print("  Using FP32 output format")

    led_count = dia_matrix.led_count
    ata_inverse = np.zeros((3, led_count, led_count), dtype=output_dtype)
    condition_numbers = []
    successful_inversions = 0

    for c in range(3):
        channel_name = ["Red", "Green", "Blue"][c]
        print(f"  Channel {c} ({channel_name})...")

        try:
            # Extract DIA matrix for this channel
            ata_dia = dia_matrix.get_channel_dia_matrix(c)

            # Convert to FP32 if needed (scipy doesn't support FP16)
            if ata_dia.dtype == np.float16:
                ata_dia = ata_dia.astype(np.float32)

            # Convert to CSC for better solving performance (spsolve prefers CSC)
            ata_csc = ata_dia.tocsc()

            # Add regularization for numerical stability
            ata_regularized = ata_csc + regularization * sp.eye(led_count, format="csc", dtype=np.float32)

            # Compute condition number for stability assessment
            # Convert a small sample to dense for condition number estimation
            if led_count <= 500:
                # For small matrices, compute exact condition number
                ata_dense_sample = ata_regularized.toarray()
                cond_num = np.linalg.cond(ata_dense_sample)
            else:
                # For large matrices, estimate condition number
                # Use the ratio of largest to smallest diagonal elements as approximation
                diag = ata_regularized.diagonal()
                cond_num = np.max(diag) / np.max([np.min(diag[diag > 0]), regularization])

            condition_numbers.append(cond_num)
            print(f"    Condition number: {cond_num:.2e}")

            if cond_num > max_condition_number:
                print(f"    ⚠️  High condition number ({cond_num:.2e}), using pseudo-inverse")
                # Use pseudo-inverse for ill-conditioned matrices
                ata_dense = ata_regularized.toarray()
                ata_inverse[c, :, :] = np.linalg.pinv(ata_dense).astype(output_dtype)
            else:
                print("    ✅ Computing sparse inverse using spsolve")
                # Use sparse solve: solve ATA * inv = I for inv
                # Create identity matrix in CSC format for spsolve efficiency
                identity = sp.eye(led_count, format="csc", dtype=np.float32)

                # Solve ATA * X = I to get ATA^-1
                # spsolve can handle multiple right-hand sides efficiently
                inverse_sparse = spla.spsolve(ata_regularized, identity)

                # Convert result to dense format with appropriate dtype
                if sp.issparse(inverse_sparse):
                    ata_inverse[c, :, :] = inverse_sparse.toarray().astype(output_dtype)
                else:
                    ata_inverse[c, :, :] = inverse_sparse.astype(output_dtype)

                successful_inversions += 1

        except Exception as e:
            print(f"    ❌ Error computing inverse: {e}")
            print("    Using pseudo-inverse as fallback")
            condition_numbers.append(float("inf"))

            # Fallback: convert to dense and use pseudo-inverse
            try:
                ata_dia = dia_matrix.get_channel_dia_matrix(c)
                # Convert to FP32 if needed (numpy pseudo-inverse works better with FP32)
                if ata_dia.dtype == np.float16:
                    ata_dia = ata_dia.astype(np.float32)
                ata_dense = ata_dia.toarray()
                ata_regularized_dense = ata_dense + regularization * np.eye(led_count, dtype=np.float32)
                ata_inverse[c, :, :] = np.linalg.pinv(ata_regularized_dense).astype(output_dtype)
            except Exception as e2:
                print(f"    ❌ Pseudo-inverse also failed: {e2}")
                # Last resort: identity matrix scaled by regularization
                ata_inverse[c, :, :] = np.eye(led_count, dtype=output_dtype) / regularization

    # Calculate average condition number (excluding infinite values)
    finite_cond_nums = [cn for cn in condition_numbers if cn != float("inf")]
    avg_condition_number = np.mean(finite_cond_nums) if finite_cond_nums else float("inf")

    print("ATA inverse computation summary:")
    print(f"  Successful inversions: {successful_inversions}/3")
    print(f"  Average condition number: {avg_condition_number:.2e}")
    print(f"  Output shape: {ata_inverse.shape}")
    print(f"  Memory usage: {ata_inverse.nbytes / 1024 / 1024:.1f} MB")

    return ata_inverse, successful_inversions, condition_numbers, avg_condition_number


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
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force FP16 output format (automatically detected from ATA matrix if not specified)",
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

        # Initialize variables
        ata_inverse = None
        successful_inversions = 0
        condition_numbers = []
        avg_condition_number = 0
        computation_time = 0
        input_was_fp16 = False

        # Load DIA matrix or dense ATA matrices
        if "dia_matrix" in data:
            print("Loading DIA matrix...")
            dia_dict = data["dia_matrix"].item()
            dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)

            # Check if input is FP16 (check storage dtype for mixed precision)
            input_was_fp16 = (
                hasattr(dia_matrix, "storage_dtype") and dia_matrix.storage_dtype == np.float16
            ) or dia_matrix.output_dtype == np.float16

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
                output_fp16=args.fp16,
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
            "output_dtype": str(ata_inverse.dtype),
            "input_was_fp16": input_was_fp16,
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
