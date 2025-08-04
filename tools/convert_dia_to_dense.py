#!/usr/bin/env python3
"""
Convert DIA Diffusion Patterns to Dense Format.

This tool converts existing diffusion pattern files that use DiagonalATAMatrix (DIA)
format to DenseATAMatrix format for better performance with high-bandwidth matrices.

Usage:
    python convert_dia_to_dense.py --input patterns_dia.npz --output patterns_dense.npz
    python convert_dia_to_dense.py --input-dir ./patterns/ --output-dir ./patterns_dense/
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Add path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.dense_ata_matrix import DenseATAMatrix
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

logger = logging.getLogger(__name__)


def convert_pattern_file(input_path: str, output_path: str, force_overwrite: bool = False) -> bool:
    """
    Convert a single pattern file from DIA to dense format.

    Args:
        input_path: Path to input NPZ file with DIA format
        output_path: Path to output NPZ file with dense format
        force_overwrite: Whether to overwrite existing output files

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        input_file = Path(input_path)
        output_file = Path(output_path)

        # Check if input exists
        if not input_file.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return False

        # Check if output exists and not forcing overwrite
        if output_file.exists() and not force_overwrite:
            logger.error(f"Output file exists (use --force to overwrite): {output_path}")
            return False

        # Load the pattern file
        logger.info(f"Loading pattern file: {input_path}")
        data = np.load(input_path, allow_pickle=True)

        # Check if it has DIA matrix
        if "dia_matrix" not in data:
            logger.error("Input file doesn't contain 'dia_matrix' - not a DIA format file")
            return False

        # Check if it already has dense matrix
        if "dense_ata_matrix" in data:
            logger.warning("Input file already contains 'dense_ata_matrix' - already converted?")

        # Load the DIA matrix
        dia_data = data["dia_matrix"].item()  # Load dict from numpy scalar
        dia_matrix = DiagonalATAMatrix.from_dict(dia_data)

        logger.info(
            f"Loaded DIA matrix: {dia_matrix.led_count} LEDs, {dia_matrix.k} diagonals, bandwidth={dia_matrix.bandwidth}"
        )

        # Analyze whether dense conversion makes sense
        led_count = dia_matrix.led_count
        diagonal_ratio = dia_matrix.k / (2 * led_count - 1) if led_count > 1 else 1.0
        bandwidth_ratio = dia_matrix.bandwidth / led_count if led_count > 0 else 1.0

        logger.info("Matrix analysis:")
        logger.info(f"  Diagonal ratio: {diagonal_ratio:.3f} ({dia_matrix.k}/{2*led_count-1} diagonals)")
        logger.info(f"  Bandwidth ratio: {bandwidth_ratio:.3f} ({dia_matrix.bandwidth}/{led_count})")

        # Estimate memory usage
        dia_memory_mb = (3 * dia_matrix.k * led_count * 4) / (1024 * 1024)  # FP32
        dense_memory_mb = (3 * led_count * led_count * 4) / (1024 * 1024)  # FP32

        logger.info(f"  Current DIA memory: {dia_memory_mb:.1f}MB")
        logger.info(f"  Dense memory will be: {dense_memory_mb:.1f}MB ({dense_memory_mb/dia_memory_mb:.1f}x)")

        if diagonal_ratio < 0.5 and dense_memory_mb > dia_memory_mb * 2:
            logger.warning(f"Converting sparse matrix ({diagonal_ratio:.1%}) to dense may not be optimal")
            logger.warning(f"Dense format will use {dense_memory_mb/dia_memory_mb:.1f}x more memory")

        # Get diffusion matrix from mixed tensor or reconstruct from DIA
        if "mixed_tensor" in data:
            # Use mixed tensor to get diffusion matrix
            mixed_tensor_data = data["mixed_tensor"].item()

            # We need to reconstruct the sparse diffusion matrix
            # This is complex, so for now we'll convert from DIA matrix
            logger.info("Converting DIA matrix to dense format (direct conversion)")

            # Create dense matrix with same precision settings
            dense_matrix = DenseATAMatrix(
                led_count=dia_matrix.led_count,
                channels=3,
                storage_dtype=dia_matrix.storage_dtype,
                output_dtype=dia_matrix.output_dtype,
            )

            # Convert DIA to dense by reconstructing the full matrices
            logger.info("Reconstructing dense ATA matrices from DIA format...")
            start_time = time.time()

            # Get DIA data on GPU
            if dia_matrix.dia_data_gpu is None:
                dia_matrix._load_dia_to_gpu()

            # Reconstruct each channel's ATA matrix
            import cupy as cp

            dense_matrices_gpu = cp.zeros((3, led_count, led_count), dtype=dense_matrix.storage_dtype)

            for channel in range(3):
                # Use the existing DiagonalATAMatrix method to get scipy DIA matrix
                scipy_dia = dia_matrix.get_channel_dia_matrix(channel)

                # Convert scipy DIA to dense
                dense_channel_cpu = scipy_dia.toarray().astype(
                    np.float32 if dense_matrix.storage_dtype == cp.float32 else np.float16
                )

                # Move to GPU
                dense_matrices_gpu[channel] = cp.asarray(dense_channel_cpu, dtype=dense_matrix.storage_dtype)

            # Store in dense matrix object
            dense_matrix.dense_matrices_gpu = dense_matrices_gpu
            dense_matrix.dense_matrices_cpu = cp.asnumpy(dense_matrices_gpu)
            dense_matrix.is_built = True

            # Calculate memory usage
            total_elements = 3 * led_count * led_count
            bytes_per_element = 4 if dense_matrix.storage_dtype == cp.float32 else 2
            dense_matrix.memory_mb = (total_elements * bytes_per_element) / (1024 * 1024)

            conversion_time = time.time() - start_time
            logger.info(f"Dense matrix conversion completed in {conversion_time:.2f}s")
            logger.info(f"Dense matrix: {dense_matrix.led_count} LEDs, {dense_matrix.memory_mb:.1f}MB")

        else:
            logger.error("No mixed_tensor found in input file - cannot reconstruct diffusion matrix")
            return False

        # Create output data dictionary
        output_data = {}

        # Copy all original data except DIA matrix
        for key in data.keys():
            if key != "dia_matrix":
                output_data[key] = data[key]

        # Add dense ATA matrix
        output_data["dense_ata_matrix"] = dense_matrix.to_dict()

        # Update metadata
        if "metadata" in output_data:
            metadata = output_data["metadata"].item()
            metadata["conversion_tool"] = "convert_dia_to_dense.py"
            metadata["conversion_timestamp"] = time.time()
            metadata["original_format"] = "dia"
            metadata["converted_format"] = "dense"
            output_data["metadata"] = metadata

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save converted file
        logger.info(f"Saving converted file: {output_path}")
        np.savez_compressed(output_path, **output_data)

        # Log results
        input_size = input_file.stat().st_size / (1024 * 1024)
        output_size = output_file.stat().st_size / (1024 * 1024)

        logger.info("Conversion completed successfully")
        logger.info(f"Input file size: {input_size:.1f}MB")
        logger.info(f"Output file size: {output_size:.1f}MB ({output_size/input_size:.1f}x)")
        logger.info(f"Removed DIA matrix ({dia_memory_mb:.1f}MB)")
        logger.info(f"Added dense matrix ({dense_matrix.memory_mb:.1f}MB)")

        return True

    except Exception as e:
        logger.error(f"Failed to convert {input_path}: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback

            traceback.print_exc()
        return False


def convert_directory(input_dir: str, output_dir: str, pattern: str = "*.npz", force_overwrite: bool = False) -> int:
    """
    Convert all pattern files in a directory.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        pattern: File pattern to match (default: "*.npz")
        force_overwrite: Whether to overwrite existing files

    Returns:
        Number of files successfully converted
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 0

    # Find matching files
    pattern_files = list(input_path.glob(pattern))
    if not pattern_files:
        logger.warning(f"No files matching pattern '{pattern}' found in {input_dir}")
        return 0

    logger.info(f"Found {len(pattern_files)} files to convert")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert files
    successful = 0
    for input_file in pattern_files:
        output_file = output_path / input_file.name
        logger.info(f"Converting {input_file.name}...")

        if convert_pattern_file(str(input_file), str(output_file), force_overwrite):
            successful += 1
        else:
            logger.error(f"Failed to convert {input_file.name}")

    logger.info(f"Successfully converted {successful}/{len(pattern_files)} files")
    return successful


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert DIA diffusion patterns to dense format")

    # Input/output options
    parser.add_argument("--input", "-i", help="Input NPZ file path")
    parser.add_argument("--output", "-o", help="Output NPZ file path")
    parser.add_argument("--input-dir", help="Input directory for batch conversion")
    parser.add_argument("--output-dir", help="Output directory for batch conversion")
    parser.add_argument("--pattern", default="*.npz", help="File pattern for batch conversion (default: *.npz)")

    # Options
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Validate arguments
    if args.input and args.input_dir:
        logger.error("Cannot specify both --input and --input-dir")
        return 1

    if not args.input and not args.input_dir:
        logger.error("Must specify either --input or --input-dir")
        return 1

    if args.input and not args.output:
        logger.error("Must specify --output when using --input")
        return 1

    if args.input_dir and not args.output_dir:
        logger.error("Must specify --output-dir when using --input-dir")
        return 1

    try:
        if args.input:
            # Single file conversion
            logger.info(f"Converting single file: {args.input} -> {args.output}")
            success = convert_pattern_file(args.input, args.output, args.force)
            return 0 if success else 1
        else:
            # Directory conversion
            logger.info(f"Converting directory: {args.input_dir} -> {args.output_dir}")
            successful = convert_directory(args.input_dir, args.output_dir, args.pattern, args.force)
            return 0 if successful > 0 else 1

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
