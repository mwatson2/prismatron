#!/usr/bin/env python3
"""
Debug version of MSE convergence test to investigate issues.
"""

import logging
from pathlib import Path

import numpy as np

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pattern_data(pattern_path: str, use_dia_ata_inverse: bool = False):
    """Load and debug pattern data."""
    logger.info(f"Loading pattern data from {pattern_path} (use_dia_ata_inverse={use_dia_ata_inverse})")

    data = np.load(pattern_path, allow_pickle=True)

    # Load mixed tensor (A^T matrix)
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())

    # Load DIA matrix (A^T A matrix)
    dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())

    # Load ATA inverse
    ata_inverse = None
    if use_dia_ata_inverse and "ata_inverse_dia" in data:
        ata_inverse_dia_dict = data["ata_inverse_dia"].item()
        ata_inverse_dia_matrix = DiagonalATAMatrix.from_dict(ata_inverse_dia_dict)
        ata_inverse = ata_inverse_dia_matrix

        # Debug: Check DIA matrix properties
        logger.info(f"DIA ATA inverse - k: {ata_inverse.k}, bandwidth: {ata_inverse.bandwidth}")
        logger.info(f"DIA data shape: {ata_inverse.dia_data_cpu.shape}")
        logger.info(f"DIA offsets range: [{ata_inverse.dia_offsets.min()}, {ata_inverse.dia_offsets.max()}]")

        # Check if matrices are actually different
        data_hash = hash(ata_inverse.dia_data_cpu.tobytes())
        logger.info(f"DIA data hash: {data_hash}")

        logger.info("Using DIA format ATA inverse")
    elif "ata_inverse" in data:
        ata_inverse = data["ata_inverse"]
        logger.info(f"Dense ATA inverse shape: {ata_inverse.shape}")
        dense_hash = hash(ata_inverse.tobytes())
        logger.info(f"Dense data hash: {dense_hash}")
        logger.info("Using dense format ATA inverse")
    else:
        raise ValueError("No suitable ATA inverse found")

    return mixed_tensor, dia_matrix, ata_inverse


def create_simple_test_frame(height: int = 480, width: int = 800) -> np.ndarray:
    """Create a simple test frame for debugging."""
    # Simple red gradient - easier to debug
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)  # Red gradient
    return frame


def debug_initial_values(pattern_name: str, ata_inverse, mixed_tensor):
    """Debug the initial LED values calculation."""
    logger.info(f"\n=== Debugging {pattern_name} ===")

    # Create simple target
    target_frame = create_simple_test_frame()

    # Calculate A^T @ b manually
    if target_frame.shape == (480, 800, 3):
        target_planar_uint8 = target_frame.astype(np.uint8).transpose(2, 0, 1)
    else:
        target_planar_uint8 = target_frame.astype(np.uint8)

    import cupy as cp

    # Compute A^T @ b
    if mixed_tensor.dtype == cp.uint8:
        target_gpu = cp.asarray(target_planar_uint8)
    else:
        target_float32 = target_planar_uint8.astype(np.float32) / 255.0
        target_gpu = cp.asarray(target_float32)

    ATb_gpu = mixed_tensor.transpose_dot_product_3d(target_gpu)
    if ATb_gpu.shape[0] != 3:
        ATb_gpu = ATb_gpu.T

    logger.info(f"A^T @ b shape: {ATb_gpu.shape}")
    logger.info(f"A^T @ b range: [{float(ATb_gpu.min()):.6f}, {float(ATb_gpu.max()):.6f}]")
    logger.info(f"A^T @ b mean: {float(ATb_gpu.mean()):.6f}")

    # Compute initial values
    if isinstance(ata_inverse, DiagonalATAMatrix):
        logger.info("Computing DIA format initial values")
        led_values_gpu = ata_inverse.multiply_3d(ATb_gpu)
    else:
        logger.info("Computing dense format initial values")
        ata_inverse_gpu = cp.asarray(ata_inverse)
        led_values_gpu = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_gpu)

    logger.info(f"Pre-clip LED values range: [{float(led_values_gpu.min()):.6f}, {float(led_values_gpu.max()):.6f}]")
    logger.info(f"Pre-clip LED values mean: {float(led_values_gpu.mean()):.6f}")

    # Apply clipping
    led_values_clipped = cp.clip(led_values_gpu, 0.0, 1.0)

    logger.info(
        f"Post-clip LED values range: [{float(led_values_clipped.min()):.6f}, {float(led_values_clipped.max()):.6f}]"
    )
    logger.info(f"Post-clip LED values mean: {float(led_values_clipped.mean()):.6f}")

    # Check how many values were clipped
    clipped_low = float(cp.sum(led_values_gpu < 0.0))
    clipped_high = float(cp.sum(led_values_gpu > 1.0))
    total_values = led_values_gpu.size

    logger.info(f"Values clipped to 0: {clipped_low}/{total_values} ({clipped_low/total_values*100:.2f}%)")
    logger.info(f"Values clipped to 1: {clipped_high}/{total_values} ({clipped_high/total_values*100:.2f}%)")

    return led_values_clipped


def main():
    """Debug main function."""
    logger.info("Starting MSE convergence debug analysis")

    pattern_dir = Path("diffusion_patterns")

    # Test a few key patterns
    patterns_to_debug = [
        ("Dense", pattern_dir / "synthetic_2624_uint8.npz", False),
        ("DIA 1.0", pattern_dir / "synthetic_2624_uint8_dia_1.0.npz", True),
        ("DIA 1.2", pattern_dir / "synthetic_2624_uint8_dia_1.2.npz", True),
        ("DIA 2.0", pattern_dir / "synthetic_2624_uint8_dia_2.0.npz", True),
    ]

    for pattern_name, pattern_path, use_dia in patterns_to_debug:
        if not pattern_path.exists():
            logger.warning(f"Pattern file not found: {pattern_path}")
            continue

        try:
            mixed_tensor, dia_matrix, ata_inverse = load_pattern_data(str(pattern_path), use_dia)
            initial_values = debug_initial_values(pattern_name, ata_inverse, mixed_tensor)

            # Run just one optimization step
            target_frame = create_simple_test_frame()
            result = optimize_frame_led_values(
                target_frame=target_frame,
                at_matrix=mixed_tensor,
                ata_matrix=dia_matrix,
                ata_inverse=ata_inverse,
                max_iterations=2,
                track_mse_per_iteration=True,
                debug=True,
            )

            logger.info(f"{pattern_name} MSE progression: {result.mse_per_iteration}")

        except Exception as e:
            logger.error(f"Error debugging {pattern_name}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
