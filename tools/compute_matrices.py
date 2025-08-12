#!/usr/bin/env python3
"""
Compute ATA matrices from Mixed Sparse Tensor diffusion patterns.

This tool loads a diffusion pattern file containing a Mixed Sparse Tensor,
computes the Dense ATA matrix using the tensor's compute_ata_dense() method,
creates optimized Symmetric Diagonal ATA matrix, computes ATA inverse,
and saves all matrices back to the file.

This is part of the matrix computation pipeline refactor where pattern
generation/capture tools only create Mixed Sparse Tensors, and matrix
computation is handled separately.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cupy
except ImportError:
    import numpy as cupy

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from src.utils.dense_ata_matrix import DenseATAMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from src.utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix

logger = logging.getLogger(__name__)


def compute_matrices_from_tensor(pattern_file: Path, args) -> int:
    """
    Compute all ATA matrices from Mixed Sparse Tensor in pattern file.

    Args:
        pattern_file: Path to pattern file with Mixed Sparse Tensor
        args: Command line arguments

    Returns:
        0 on success, 1 on error
    """
    try:
        # Load pattern file
        print(f"Loading pattern file: {pattern_file}")
        data = np.load(str(pattern_file), allow_pickle=True)

        # Check for Mixed Sparse Tensor
        if "mixed_tensor" not in data:
            print("❌ No 'mixed_tensor' found in pattern file")
            print(f"   Available keys: {list(data.keys())}")
            return 1

        # Load Mixed Sparse Tensor
        print("Loading Mixed Sparse Tensor...")
        mixed_tensor_dict = data["mixed_tensor"].item()
        tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

        print(f"Mixed Sparse Tensor loaded:")
        print(f"  LEDs: {tensor.batch_size}")
        print(f"  Channels: {tensor.channels}")
        print(f"  Block size: {tensor.block_size}x{tensor.block_size}")
        print(f"  Data type: {tensor.dtype}")
        print(f"  Memory: {tensor.memory_info()['total_mb']:.1f}MB")

        # Convert float32 tensor to uint8 if needed
        tensor_converted = False
        if tensor.dtype == cupy.float32 and not args.no_convert:
            print("\n🔄 Converting float32 tensor to uint8 for memory efficiency...")
            tensor_uint8 = tensor.to_uint8()

            # Replace the tensor with the uint8 version
            original_memory = tensor.memory_info()["total_mb"]
            tensor = tensor_uint8
            tensor_converted = True

            print(f"✅ Conversion complete:")
            print(f"  New data type: {tensor.dtype}")
            print(f"  Memory reduction: {original_memory:.1f}MB -> {tensor.memory_info()['total_mb']:.1f}MB")
            print(f"  Reduction factor: {original_memory / tensor.memory_info()['total_mb']:.1f}x")

        # Check if matrices already exist
        has_dense_ata = "dense_ata_matrix" in data
        has_symmetric_dia = "symmetric_dia_matrix" in data
        has_batch_symmetric_dia = "batch_symmetric_dia_matrix" in data
        has_ata_inverse = "ata_inverse" in data

        # Determine which matrices need computation
        missing_matrices = []
        if not has_dense_ata and not has_symmetric_dia:
            missing_matrices.append("dense_ata or symmetric_dia")
        if not has_ata_inverse:
            missing_matrices.append("ata_inverse")
        if not has_batch_symmetric_dia and (has_symmetric_dia or "dia_matrix" in data or "diagonal_ata_matrix" in data):
            missing_matrices.append("batch_symmetric_dia")

        if args.force:
            print("Force flag specified. Recomputing all matrices.")
            compute_all = True
        elif not missing_matrices:
            print("✅ All matrices already exist in file")
            try:
                response = input("Recompute all matrices? (y/N): ")
                if response.lower() != "y":
                    print("Skipping computation.")
                    # If tensor was converted to uint8, save it even if not recomputing matrices
                    if tensor_converted:
                        print("💾 Saving converted uint8 tensor...")
                        save_dict = {}
                        for key in data:
                            save_dict[key] = data[key]
                        save_dict["mixed_tensor"] = tensor.to_dict()
                        np.savez_compressed(str(pattern_file), **save_dict)
                        print(f"✅ Pattern file updated with uint8 tensor: {pattern_file}")
                    return 0
                else:
                    compute_all = True
            except EOFError:
                print("Non-interactive mode detected. Adding missing matrices only.")
                compute_all = False
        else:
            print(f"Missing matrices: {', '.join(missing_matrices)}")
            compute_all = False

        # Create backup if requested
        if args.backup:
            backup_path = pattern_file.with_suffix(".bak.npz")
            print(f"Creating backup: {backup_path}")
            import shutil

            shutil.copy2(str(pattern_file), str(backup_path))

        # Step 1: Compute or load Dense ATA matrix
        dense_ata_matrix = None
        dense_computation_time = 0.0

        if compute_all or not has_dense_ata:
            print("\n🔄 Step 1: Computing Dense ATA matrix...")
            start_time = time.perf_counter()

            dense_ata_array = tensor.compute_ata_dense()  # Shape: (led_count, led_count, channels)

            # Transpose to match DenseATAMatrix expected format: (channels, led_count, led_count)
            dense_ata_transposed = np.transpose(dense_ata_array, (2, 0, 1))

            dense_computation_time = time.perf_counter() - start_time
            print(f"Dense ATA computed in {dense_computation_time:.2f}s")
            print(f"  Shape: {dense_ata_transposed.shape}")
            print(f"  Memory: {dense_ata_transposed.nbytes / (1024*1024):.1f}MB")

            # Create DenseATAMatrix object
            dense_ata_matrix = DenseATAMatrix(
                led_count=tensor.batch_size, storage_dtype=np.float32, output_dtype=np.float32
            )
            dense_ata_matrix.dense_matrices_cpu = dense_ata_transposed
            dense_ata_matrix.memory_mb = dense_ata_transposed.nbytes / (1024 * 1024)
            dense_ata_matrix.is_built = True
        else:
            print("\n✅ Step 1: Loading existing Dense ATA matrix...")
            dense_ata_dict = data["dense_ata_matrix"].item()
            dense_ata_matrix = DenseATAMatrix.from_dict(dense_ata_dict)
            print(f"  Shape: {dense_ata_matrix.dense_matrices_cpu.shape}")
            print(f"  Memory: {dense_ata_matrix.memory_mb:.1f}MB")

        # Step 2: Create or load Symmetric Diagonal ATA Matrix
        symmetric_dia_matrix = None
        sparsity_info = []
        dia_computation_time = 0.0

        if compute_all or not has_symmetric_dia:
            print("\n🔄 Step 2: Creating Symmetric Diagonal ATA matrix...")
            start_time = time.perf_counter()

            try:
                symmetric_dia_matrix = SymmetricDiagonalATAMatrix.from_dense(
                    dense_ata_matrix.dense_matrices_cpu,
                    tensor.batch_size,
                    args.significance_threshold,
                    tensor.block_size,
                )

                # Collect sparsity info for reporting
                for c in range(3):
                    ata_channel = dense_ata_matrix.dense_matrices_cpu[c]
                    max_val = np.abs(ata_channel).max()
                    threshold = max_val * args.significance_threshold
                    significant_mask = np.abs(ata_channel) >= threshold
                    nnz = np.sum(significant_mask)
                    sparsity = (nnz / (tensor.batch_size * tensor.batch_size)) * 100
                    sparsity_info.append(
                        {"channel": c, "nnz": nnz, "sparsity": sparsity, "max_val": max_val, "threshold": threshold}
                    )
                    print(
                        f"  Channel {c}: {nnz:,} non-zeros ({sparsity:.2f}% dense), max={max_val:.6f}, threshold={threshold:.6f}"
                    )

                print(f"Symmetric DIA matrix created:")
                print(f"  Upper diagonals: {symmetric_dia_matrix.k_upper}")
                print(f"  Bandwidth: {symmetric_dia_matrix.bandwidth}")
                print(f"  Memory: {symmetric_dia_matrix.dia_data_gpu.nbytes / (1024*1024):.1f}MB")

            except Exception as e:
                print(f"  ⚠️  Could not create symmetric DIA matrix: {e}")
                print("  Continuing with dense ATA only...")
                symmetric_dia_matrix = None

                # Fallback sparsity analysis
                for c in range(3):
                    ata_channel = dense_ata_matrix.dense_matrices_cpu[c]
                    max_val = np.abs(ata_channel).max()
                    threshold = max_val * args.significance_threshold
                    significant_mask = np.abs(ata_channel) >= threshold
                    nnz = np.sum(significant_mask)
                    sparsity = (nnz / (tensor.batch_size * tensor.batch_size)) * 100
                    sparsity_info.append(
                        {"channel": c, "nnz": nnz, "sparsity": sparsity, "max_val": max_val, "threshold": threshold}
                    )
                    print(
                        f"  Channel {c}: {nnz:,} non-zeros ({sparsity:.2f}% dense), max={max_val:.6f}, threshold={threshold:.6f}"
                    )

            dia_computation_time = time.perf_counter() - start_time
            print(f"Symmetric DIA computation completed in {dia_computation_time:.2f}s")
        else:
            print("\n✅ Step 2: Loading existing Symmetric Diagonal ATA matrix...")
            symmetric_dia_dict = data["symmetric_dia_matrix"].item()
            symmetric_dia_matrix = SymmetricDiagonalATAMatrix.from_dict(symmetric_dia_dict)
            print(f"  Upper diagonals: {symmetric_dia_matrix.k_upper}")
            print(f"  Bandwidth: {symmetric_dia_matrix.bandwidth}")
            print(f"  Memory: {symmetric_dia_matrix.dia_data_gpu.nbytes / (1024*1024):.1f}MB")

            # Create sparsity info for compatibility
            for c in range(3):
                sparsity_info.append(
                    {
                        "channel": c,
                        "nnz": symmetric_dia_matrix.nnz or 0,
                        "sparsity": symmetric_dia_matrix.sparsity or 0,
                        "max_val": 1.0,
                        "threshold": args.significance_threshold,
                    }
                )

        # Step 3: Compute or load ATA Inverse
        inverse_computation_time = 0.0

        if compute_all or not has_ata_inverse:
            print("\n🔄 Step 3: Computing ATA inverse...")
            start_time = time.perf_counter()

            # Import the compute_ata_inverse functions
            from tools.compute_ata_inverse import compute_ata_inverse_from_dense

            ata_inverse, successful_inversions, condition_numbers, avg_condition_number = (
                compute_ata_inverse_from_dense(
                    dense_ata_matrix,
                    regularization=args.regularization,
                    max_condition_number=args.max_condition,
                    output_fp16=args.fp16,
                )
            )

            inverse_computation_time = time.perf_counter() - start_time
            print(f"ATA inverse computed in {inverse_computation_time:.2f}s")
        else:
            print("\n✅ Step 3: Loading existing ATA inverse...")
            ata_inverse = data["ata_inverse"]
            successful_inversions = 3  # Assume all channels successful for existing
            condition_numbers = [0.0, 0.0, 0.0]  # Placeholder values
            avg_condition_number = 0.0
            print(f"  Shape: {ata_inverse.shape}")
            print(f"  Memory: {ata_inverse.nbytes / (1024*1024):.1f}MB")

        # Step 4: Compute or load Batch Symmetric ATA Matrix
        batch_symmetric_ata_matrix = None
        batch_computation_time = 0.0

        if compute_all or not has_batch_symmetric_dia:
            # Only compute batch version if we have a symmetric matrix and LED count is compatible
            if symmetric_dia_matrix is not None and tensor.batch_size % 16 == 0:
                print("\n🔄 Step 4: Creating Batch Symmetric Diagonal ATA matrix...")
                start_time = time.perf_counter()

                try:
                    # Create batch version for 8-frame operations (default used in system)
                    batch_symmetric_ata_matrix = BatchSymmetricDiagonalATAMatrix.from_symmetric_diagonal_matrix(
                        symmetric_dia_matrix, batch_size=8
                    )

                    batch_computation_time = time.perf_counter() - start_time
                    print(f"Batch symmetric ATA matrix created in {batch_computation_time:.2f}s")
                    print(f"  Batch size: 8 frames")
                    print(f"  Block storage shape: {batch_symmetric_ata_matrix.block_data_gpu.shape}")
                    print(f"  Block diagonal count: {batch_symmetric_ata_matrix.block_diag_count}")
                    print(f"  GPU memory usage: {batch_symmetric_ata_matrix.block_data_gpu.nbytes / (1024*1024):.1f}MB")

                except Exception as e:
                    print(f"  ⚠️  Could not create batch symmetric ATA matrix: {e}")
                    print("  This is normal for LED counts not divisible by 16 or when batch kernels are not available")
                    batch_symmetric_ata_matrix = None
            else:
                if symmetric_dia_matrix is None:
                    print("\n⚠️  Step 4: Skipping batch computation (no symmetric matrix available)")
                else:
                    print(
                        f"\n⚠️  Step 4: Skipping batch computation (LED count {tensor.batch_size} not divisible by 16)"
                    )
        elif has_batch_symmetric_dia:
            print("\n✅ Step 4: Batch symmetric ATA matrix already exists")
            # Note: We don't load it here as it's only needed for display - the system loads it on demand
        else:
            print("\n⚠️  Step 4: Skipping batch computation (not requested)")

        # Step 5: Save all matrices to file
        print("\n🔄 Step 5: Saving matrices to file...")

        # Convert to dictionary for saving
        save_dict = {}
        for key in data:
            save_dict[key] = data[key]

        # Update the mixed tensor if it was converted to uint8
        if tensor_converted:
            save_dict["mixed_tensor"] = tensor.to_dict()
            print(f"  Saving converted uint8 tensor to file...")

        # Add computed or existing matrices
        if compute_all or not has_dense_ata:
            save_dict["dense_ata_matrix"] = dense_ata_matrix.to_dict()
        if compute_all or not has_ata_inverse:
            save_dict["ata_inverse"] = ata_inverse
        if compute_all or missing_matrices:
            save_dict["sparsity_analysis"] = sparsity_info

        # Add symmetric DIA matrix if created or recomputed
        if symmetric_dia_matrix is not None and (compute_all or not has_symmetric_dia):
            save_dict["symmetric_dia_matrix"] = {
                "dia_data_gpu": cupy.asnumpy(symmetric_dia_matrix.dia_data_gpu),
                "dia_offsets_upper": symmetric_dia_matrix.dia_offsets_upper,
                "k_upper": symmetric_dia_matrix.k_upper,
                "bandwidth": symmetric_dia_matrix.bandwidth,
                "led_count": symmetric_dia_matrix.led_count,
                "channels": symmetric_dia_matrix.channels,
                "crop_size": symmetric_dia_matrix.crop_size,
                "output_dtype": str(symmetric_dia_matrix.output_dtype),
                "original_k": symmetric_dia_matrix.original_k,
                "sparsity": symmetric_dia_matrix.sparsity,
                "nnz": symmetric_dia_matrix.nnz,
            }

        # Add batch symmetric DIA matrix if created
        if batch_symmetric_ata_matrix is not None:
            save_dict["batch_symmetric_dia_matrix"] = {
                "block_data_gpu": cupy.asnumpy(batch_symmetric_ata_matrix.block_data_gpu),
                "max_block_diag": batch_symmetric_ata_matrix.max_block_diag,
                "block_diag_count": batch_symmetric_ata_matrix.block_diag_count,
                "led_count": batch_symmetric_ata_matrix.led_count,
                "crop_size": batch_symmetric_ata_matrix.crop_size,
                "channels": batch_symmetric_ata_matrix.channels,
                "batch_size": batch_symmetric_ata_matrix.batch_size,
                "block_size": batch_symmetric_ata_matrix.block_size,
                "led_blocks": batch_symmetric_ata_matrix.led_blocks,
                "padded_led_count": batch_symmetric_ata_matrix.padded_led_count,
                "bandwidth": batch_symmetric_ata_matrix.bandwidth,
                "sparsity": batch_symmetric_ata_matrix.sparsity,
                "nnz": batch_symmetric_ata_matrix.nnz,
                "original_k": batch_symmetric_ata_matrix.original_k,
                "output_dtype": batch_symmetric_ata_matrix.output_dtype.__name__,
                "compute_dtype": batch_symmetric_ata_matrix.compute_dtype.__name__,
            }

        # Add computation metadata
        total_time = dense_computation_time + dia_computation_time + inverse_computation_time + batch_computation_time
        computation_metadata = {
            "dense_ata_computation_time": dense_computation_time,
            "dia_computation_time": dia_computation_time,
            "inverse_computation_time": inverse_computation_time,
            "batch_computation_time": batch_computation_time,
            "total_computation_time": total_time,
            "successful_inversions": successful_inversions,
            "condition_numbers": condition_numbers,
            "avg_condition_number": avg_condition_number,
            "regularization": args.regularization,
            "max_condition_number": args.max_condition,
            "significance_threshold": args.significance_threshold,
            "output_dtype": str(ata_inverse.dtype),
            "timestamp": time.time(),
        }
        save_dict["ata_computation_metadata"] = computation_metadata

        # Save file
        np.savez_compressed(str(pattern_file), **save_dict)

        # Summary
        symmetric_dia_memory = symmetric_dia_matrix.dia_data_gpu.nbytes if symmetric_dia_matrix is not None else 0
        batch_symmetric_memory = (
            batch_symmetric_ata_matrix.block_data_gpu.nbytes if batch_symmetric_ata_matrix is not None else 0
        )
        total_memory = (
            dense_ata_matrix.dense_matrices_cpu.nbytes
            + symmetric_dia_memory
            + batch_symmetric_memory
            + ata_inverse.nbytes
        ) / (1024 * 1024)

        print(f"\n✅ Matrix computation completed successfully!")
        print(f"   Total computation time: {total_time:.2f}s")
        print(f"   Total matrix memory: {total_memory:.1f}MB")
        print(f"   Dense ATA: {dense_ata_matrix.dense_matrices_cpu.nbytes / (1024*1024):.1f}MB")
        if symmetric_dia_matrix is not None:
            print(
                f"   Symmetric DIA: {symmetric_dia_matrix.dia_data_gpu.nbytes / (1024*1024):.1f}MB ({symmetric_dia_matrix.k_upper} upper diagonals)"
            )
        if batch_symmetric_ata_matrix is not None:
            print(
                f"   Batch Symmetric DIA: {batch_symmetric_ata_matrix.block_data_gpu.nbytes / (1024*1024):.1f}MB (8-frame batches, {batch_symmetric_ata_matrix.block_diag_count} block diagonals)"
            )
        print(f"   ATA inverse: {ata_inverse.nbytes / (1024*1024):.1f}MB")
        if sparsity_info:
            print(f"   Sparsity: ~{sparsity_info[0]['sparsity']:.2f}% dense")
        print(f"   Successful inversions: {successful_inversions}/3 channels")
        print(f"   Pattern file updated: {pattern_file}")

        return 0

    except Exception as e:
        print(f"❌ Error computing matrices: {e}")
        import traceback

        if args.verbose:
            traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description="Compute ATA matrices from Mixed Sparse Tensor diffusion patterns")
    parser.add_argument("pattern_file", help="Path to diffusion pattern file (.npz)")
    parser.add_argument(
        "--significance-threshold",
        type=float,
        default=0.05,
        help="Significance threshold for diagonal filtering (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help="Regularization parameter for matrix inversion (default: 1e-6)",
    )
    parser.add_argument(
        "--max-condition",
        type=float,
        default=1e12,
        help="Maximum condition number before using pseudo-inverse (default: 1e12)",
    )
    parser.add_argument("--backup", action="store_true", help="Create backup of original file")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if matrices exist")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 output format for matrices")
    parser.add_argument("--no-convert", action="store_true", help="Don't convert float32 tensors to uint8")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    pattern_file = Path(args.pattern_file)
    if not pattern_file.exists():
        print(f"❌ Pattern file not found: {pattern_file}")
        return 1

    return compute_matrices_from_tensor(pattern_file, args)


if __name__ == "__main__":
    sys.exit(main())
