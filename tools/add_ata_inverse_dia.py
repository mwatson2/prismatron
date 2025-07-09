#!/usr/bin/env python3
"""
Tool to add DIA format ATA inverse matrices to diffusion pattern files.

This tool creates a sparse approximation of the dense ATA inverse matrices
by converting them to diagonal (DIA) storage format and zeroing out
far-off-diagonal elements. This provides a memory-efficient alternative
to the dense ATA inverse while potentially maintaining good optimization
performance for most practical cases.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def dense_to_dia_approximation(
    dense_matrix: np.ndarray, reference_bandwidth: int, diagonal_factor: float
) -> Tuple[DiagonalATAMatrix, Dict[str, float]]:
    """
    Convert a dense matrix to DIA format approximation.

    Args:
        dense_matrix: Dense matrix to approximate (led_count, led_count)
        reference_bandwidth: Bandwidth of the reference ATA matrix
        diagonal_factor: Multiplier for number of diagonals to keep

    Returns:
        Tuple of (DiagonalATAMatrix object, metadata dict)
    """
    led_count = dense_matrix.shape[0]
    assert dense_matrix.shape == (led_count, led_count), f"Expected square matrix, got {dense_matrix.shape}"

    # Calculate target number of diagonals
    reference_k = 2 * reference_bandwidth + 1  # Total diagonals in reference ATA matrix
    target_k = int(reference_k * diagonal_factor)
    target_bandwidth = target_k // 2

    logger.info(f"  Reference ATA: bandwidth={reference_bandwidth}, k={reference_k}")
    logger.info(f"  Target DIA: bandwidth={target_bandwidth}, k={target_k}, factor={diagonal_factor}")

    # Create band mask to zero out far-off-diagonal elements
    row_idx, col_idx = np.indices(dense_matrix.shape)
    distance = np.abs(row_idx - col_idx)
    band_mask = distance <= target_bandwidth

    # Apply band mask to create sparse approximation
    sparse_matrix = dense_matrix.copy()
    sparse_matrix[~band_mask] = 0.0

    # Convert to DIA format using scipy
    sparse_csr = sp.csr_matrix(sparse_matrix)
    sparse_dia = sparse_csr.todia()

    # Create DiagonalATAMatrix object
    dia_matrix = DiagonalATAMatrix(led_count)

    # Set the DIA data directly
    dia_matrix.k = sparse_dia.nnz_diagonals if hasattr(sparse_dia, "nnz_diagonals") else len(sparse_dia.offsets)
    dia_matrix.bandwidth = target_bandwidth
    dia_matrix.dia_offsets_cpu = sparse_dia.offsets.copy()
    dia_matrix.dia_data_cpu = sparse_dia.data.copy()

    # Calculate compression and error metrics
    original_nnz = np.count_nonzero(dense_matrix)
    dia_nnz = np.count_nonzero(sparse_matrix)
    compression_ratio = (sparse_matrix.nbytes) / dense_matrix.nbytes

    # Calculate approximation error (RMS)
    error_matrix = dense_matrix - sparse_matrix
    rms_error = np.sqrt(np.mean(error_matrix**2))

    metadata = {
        "diagonal_factor": diagonal_factor,
        "original_nnz": int(original_nnz),
        "dia_nnz": int(dia_nnz),
        "compression_ratio": float(compression_ratio),
        "approximation_error": float(rms_error),
        "target_bandwidth": int(target_bandwidth),
        "target_k": int(target_k),
        "generation_timestamp": time.time(),
    }

    logger.info(f"  Compression: {compression_ratio:.3f} ({dia_nnz:,} / {original_nnz:,} non-zeros)")
    logger.info(f"  RMS error: {rms_error:.6f}")

    return dia_matrix, metadata


def create_ata_inverse_dia(
    ata_inverse: np.ndarray, reference_dia_matrix: DiagonalATAMatrix, diagonal_factor: float
) -> Tuple[Dict[str, any], Dict[str, any]]:
    """
    Create DIA format approximation of ATA inverse matrices.

    Args:
        ata_inverse: Dense ATA inverse matrices (3, led_count, led_count)
        reference_dia_matrix: Reference ATA matrix for bandwidth info
        diagonal_factor: Multiplier for number of diagonals to keep

    Returns:
        Tuple of (ata_inverse_dia dict, metadata dict)
    """
    channels, led_count, _ = ata_inverse.shape
    assert channels == 3, f"Expected 3 channels, got {channels}"
    assert ata_inverse.shape == (3, led_count, led_count), f"Unexpected ATA inverse shape: {ata_inverse.shape}"

    logger.info(f"Creating DIA format ATA inverse approximation:")
    logger.info(f"  Input shape: {ata_inverse.shape}")
    logger.info(f"  Diagonal factor: {diagonal_factor}")
    logger.info(f"  Reference bandwidth: {reference_dia_matrix.bandwidth}")

    # Convert each channel separately
    channel_matrices = {}
    channel_metadata = {}

    total_original_nnz = 0
    total_dia_nnz = 0
    total_rms_error = 0

    for channel in range(channels):
        logger.info(f"  Processing channel {channel}...")

        channel_matrix = ata_inverse[channel]
        dia_matrix, metadata = dense_to_dia_approximation(
            channel_matrix, reference_dia_matrix.bandwidth, diagonal_factor
        )

        # Store as serializable dict
        channel_matrices[f"channel_{channel}"] = dia_matrix.to_dict()
        channel_metadata[f"channel_{channel}"] = metadata

        total_original_nnz += metadata["original_nnz"]
        total_dia_nnz += metadata["dia_nnz"]
        total_rms_error += metadata["approximation_error"]

    # Create overall metadata
    overall_metadata = {
        "diagonal_factor": diagonal_factor,
        "channels": channels,
        "led_count": led_count,
        "reference_bandwidth": reference_dia_matrix.bandwidth,
        "total_original_nnz": total_original_nnz,
        "total_dia_nnz": total_dia_nnz,
        "overall_compression_ratio": total_dia_nnz / total_original_nnz if total_original_nnz > 0 else 0,
        "average_rms_error": total_rms_error / channels,
        "generation_timestamp": time.time(),
        "channel_metadata": channel_metadata,
    }

    logger.info(f"  Overall compression: {overall_metadata['overall_compression_ratio']:.3f}")
    logger.info(f"  Average RMS error: {overall_metadata['average_rms_error']:.6f}")

    return channel_matrices, overall_metadata


def add_ata_inverse_dia_to_file(input_file: Path, diagonal_factor: float, output_file: Path = None):
    """
    Add DIA format ATA inverse to a pattern file.

    Args:
        input_file: Input pattern file path
        diagonal_factor: Multiplier for number of diagonals to keep
        output_file: Output file path (if None, updates input file in place)
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_file is None:
        output_file = input_file

    logger.info(f"Processing pattern file: {input_file}")
    logger.info(f"Diagonal factor: {diagonal_factor}")
    logger.info(f"Output file: {output_file}")

    # Load existing data
    data = np.load(input_file, allow_pickle=True)

    # Check required keys
    required_keys = ["ata_inverse", "dia_matrix"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Required key '{key}' not found in pattern file")

    # Load ATA inverse and reference DIA matrix
    ata_inverse = data["ata_inverse"]
    dia_matrix_dict = data["dia_matrix"].item()
    dia_matrix = DiagonalATAMatrix.from_dict(dia_matrix_dict)

    logger.info(f"Loaded ATA inverse: shape={ata_inverse.shape}, dtype={ata_inverse.dtype}")
    logger.info(f"Loaded reference DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k}")

    # Create DIA format approximation
    ata_inverse_dia, ata_inverse_dia_metadata = create_ata_inverse_dia(ata_inverse, dia_matrix, diagonal_factor)

    # Prepare output data (copy all existing data)
    output_data = {key: data[key] for key in data.keys()}

    # Add new DIA format data
    output_data["ata_inverse_dia"] = ata_inverse_dia
    output_data["ata_inverse_dia_metadata"] = ata_inverse_dia_metadata

    # Save updated file
    logger.info(f"Saving updated pattern file to: {output_file}")
    np.savez_compressed(output_file, **output_data)

    # Report file size change if updating in place
    if output_file == input_file:
        new_size = output_file.stat().st_size / 1024 / 1024
        logger.info(f"Updated file size: {new_size:.1f} MB")
    else:
        input_size = input_file.stat().st_size / 1024 / 1024
        output_size = output_file.stat().st_size / 1024 / 1024
        logger.info(f"File size: {input_size:.1f} MB -> {output_size:.1f} MB ({output_size - input_size:+.1f} MB)")

    logger.info("Successfully added DIA format ATA inverse")


def main():
    parser = argparse.ArgumentParser(
        description="Add DIA format ATA inverse matrices to diffusion pattern files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add DIA format ATA inverse with same number of diagonals as ATA matrix
  python add_ata_inverse_dia.py patterns.npz --diagonal-factor 1.0

  # Add DIA format with 1.5x the diagonals for better approximation
  python add_ata_inverse_dia.py patterns.npz --diagonal-factor 1.5 --output new_patterns.npz

  # Process multiple factors
  for factor in 1.0 1.2 1.4 1.6 1.8 2.0; do
      python add_ata_inverse_dia.py base_patterns.npz --diagonal-factor $factor --output patterns_dia_$factor.npz
  done
""",
    )

    parser.add_argument("input_file", type=Path, help="Input pattern file (.npz) containing ata_inverse and dia_matrix")
    parser.add_argument(
        "--diagonal-factor",
        type=float,
        required=True,
        help="Multiplier for number of diagonals (1.0 = same as ATA matrix, 2.0 = twice as many)",
    )
    parser.add_argument("--output", type=Path, help="Output file path (if not specified, updates input file in place)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.diagonal_factor <= 0:
        parser.error("Diagonal factor must be positive")

    try:
        add_ata_inverse_dia_to_file(args.input_file, args.diagonal_factor, args.output)
        print(f"✅ Successfully added DIA format ATA inverse with factor {args.diagonal_factor}")

    except Exception as e:
        logger.error(f"Failed to process file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
