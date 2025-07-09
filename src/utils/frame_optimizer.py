#!/usr/bin/env python3
"""
Standalone frame optimization function for LED displays.

This module provides a clean, standalone function for optimizing LED values
from target frames using either mixed tensor or DIA matrix formats.
Extracted from the DenseLEDOptimizer class for modular usage.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import cupy as cp
import numpy as np

from .diagonal_ata_matrix import DiagonalATAMatrix
from .performance_timing import PerformanceTiming
from .single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


@dataclass
class FrameOptimizationResult:
    """Results from frame optimization process."""

    led_values: np.ndarray  # RGB values for each LED (3, led_count) in spatial order [0,255]
    error_metrics: Dict[str, float]  # Error metrics (mse, mae, etc.)
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged
    step_sizes: Optional[np.ndarray] = None  # Step sizes per iteration (for debugging)
    timing_data: Optional[Dict[str, float]] = None  # Performance timing breakdown


def load_ata_inverse_from_pattern(pattern_file_path: str) -> Optional[np.ndarray]:
    """
    Load ATA inverse matrices from a diffusion pattern file.

    Args:
        pattern_file_path: Path to the .npz pattern file

    Returns:
        ATA inverse matrices (3, led_count, led_count) or None if not found
    """
    try:
        import numpy as np

        data = np.load(pattern_file_path, allow_pickle=True)

        if "ata_inverse" in data:
            return data["ata_inverse"]
        else:
            print(f"Warning: No ATA inverse found in {pattern_file_path}")
            return None
    except Exception as e:
        print(f"Warning: Could not load ATA inverse from {pattern_file_path}: {e}")
        return None


def optimize_frame_led_values(
    target_frame: np.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
    ata_matrix: DiagonalATAMatrix,
    ata_inverse: np.ndarray,
    initial_values: Optional[np.ndarray] = None,
    max_iterations: int = 5,
    convergence_threshold: float = 0.3,
    step_size_scaling: float = 0.9,
    compute_error_metrics: bool = False,
    debug: bool = False,
    enable_timing: bool = False,
) -> FrameOptimizationResult:
    """
    Optimize LED values for a target frame using gradient descent with optimal ATA inverse initialization.

    This is a standalone function that uses modern tensor formats and ATA inverse initialization
    for optimal convergence: SingleBlockMixedSparseTensor for A^T @ b and DiagonalATAMatrix
    for efficient A^T A operations.

    Args:
        target_frame: Target image in planar int8 format (3, 480, 800) or standard (480, 800, 3)
        at_matrix: A^T matrix for computing A^T @ b (SingleBlockMixedSparseTensor with 3D CUDA kernels)
        ata_matrix: A^T A matrix for gradient computation (DiagonalATAMatrix in DIA format with RCM ordering)
        ata_inverse: A^T A inverse matrices for optimal initialization (3, led_count, led_count) [REQUIRED]
        initial_values: Override for initial LED values (3, led_count), if None uses ATA inverse initialization
        max_iterations: Maximum optimization iterations
        convergence_threshold: Convergence threshold for delta norm
        step_size_scaling: Step size scaling factor (0.9 typical)
        compute_error_metrics: Whether to compute error metrics (slower)
        debug: Enable debug output and tracking
        enable_timing: Enable detailed performance timing breakdown

    Returns:
        FrameOptimizationResult with LED values in spatial order (3, led_count)
    """

    # Initialize performance timing if requested
    timing = PerformanceTiming("frame_optimizer", enable_gpu_timing=True) if enable_timing else None
    # Validate input frame format and convert to planar int8 if needed
    if target_frame.dtype != np.int8 and target_frame.dtype != np.uint8:
        raise ValueError(f"Target frame must be int8 or uint8, got {target_frame.dtype}")

    # Handle both planar (3, H, W) and standard (H, W, 3) formats
    # Keep target as uint8 for int8 mixed tensors, convert to float32 for CSC matrices
    if target_frame.shape == (3, 480, 800):
        # Already in planar format - keep as uint8
        target_planar_uint8 = target_frame.astype(np.uint8)
    elif target_frame.shape == (480, 800, 3):
        # Convert from HWC to CHW planar format - keep as uint8
        target_planar_uint8 = target_frame.astype(np.uint8).transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    else:
        raise ValueError(f"Unsupported frame shape {target_frame.shape}, expected (3, 480, 800) or (480, 800, 3)")

    debug and logger.info(f"Target frame shape: {target_planar_uint8.shape}")

    # Step 1: Calculate A^T @ b using the appropriate format
    debug and logger.info("Computing A^T @ b...")

    timing and timing.start("atb_calculation", use_gpu_events=True)

    ATb = _calculate_atb(target_planar_uint8, at_matrix, debug=debug)  # Shape: (3, led_count) or (led_count, 3)

    timing and timing.stop("atb_calculation")

    # Ensure ATb is in (3, led_count) format for consistency
    if ATb.shape[0] != 3:
        ATb = ATb.T  # Convert (led_count, 3) -> (3, led_count)

    debug and logger.info(f"A^T @ b shape: {ATb.shape}")

    # Step 2: Initialize LED values in optimization order - ATA inverse is now required
    led_count = ATb.shape[1]

    # Validate ATA inverse shape (required parameter)
    if ata_inverse.shape != (3, led_count, led_count):
        raise ValueError(f"ATA inverse shape {ata_inverse.shape} != (3, {led_count}, {led_count})")

    if initial_values is not None:
        # Use provided initial values (override ATA inverse)
        if initial_values.shape != (3, led_count):
            raise ValueError(f"Initial values shape {initial_values.shape} != (3, {led_count})")
        # Normalize to [0,1] if needed
        led_values_normalized = (initial_values / 255.0 if initial_values.max() > 1.0 else initial_values).astype(
            np.float32
        )
        debug and logger.info("Using provided initial values (overriding ATA inverse)")
    else:
        # Use ATA inverse for optimal initialization: x_init = (A^T A)^-1 * A^T b
        debug and logger.info("Using ATA inverse for optimal initialization")
        timing and timing.start("ata_inverse_initialization")

        # Compute optimal initial guess for all channels using efficient einsum

        timing and timing.start("ata_inverse_matvec_total", use_gpu_events=True)
        # Transfer to GPU for efficient computation
        ata_inverse_gpu = cp.asarray(ata_inverse)  # Shape: (3, led_count, led_count)
        ATb_gpu = cp.asarray(ATb)  # Shape: (3, led_count)

        # Efficient einsum: (3, led_count, led_count) @ (3, led_count) -> (3, led_count)
        led_values_gpu = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_gpu)

        # Transfer back to CPU
        led_values_normalized = cp.asnumpy(led_values_gpu).astype(np.float32)

        timing and timing.stop("ata_inverse_matvec_total")

        # Clamp to valid range [0, 1]
        led_values_normalized = np.clip(led_values_normalized, 0.0, 1.0)

        timing and timing.stop("ata_inverse_initialization")
        debug and logger.info("Initialization completed using ATA inverse")

    debug and logger.info(f"Initial LED values shape: {led_values_normalized.shape}")

    # Use values in their provided order (pattern generation handles optimal ordering)
    # DIA matrix expects values in same order as pattern generation (pre-optimized)
    ATb_opt_order = ATb
    led_values_opt_order = led_values_normalized
    debug and logger.info("Using DIA matrix with pre-optimized ordering from pattern generation")

    # Step 3: Transfer to GPU for optimization
    timing and timing.start("gpu_transfer", use_gpu_events=True)

    ATb_gpu = cp.asarray(ATb_opt_order)
    led_values_gpu = cp.asarray(led_values_opt_order)

    timing and timing.stop("gpu_transfer")

    # Step 4: Gradient descent optimization loop
    debug and logger.info(f"Starting optimization: max_iterations={max_iterations}")
    step_sizes = [] if debug else None

    timing and timing.start("optimization_loop", use_gpu_events=True)
    for iteration in range(max_iterations):
        # Compute gradient: A^T A @ x - A^T @ b using DIA matrix operations
        ATA_x = ata_matrix.multiply_3d(led_values_gpu)
        gradient = ATA_x - ATb_gpu  # Shape: (3, led_count)

        # Compute step size: (g^T @ g) / (g^T @ A^T A @ g)
        g_dot_g = cp.sum(gradient * gradient)  # Shape: scalar, stays on GPU
        g_dot_ATA_g_per_channel = ata_matrix.g_ata_g_3d(gradient)
        g_dot_ATA_g = cp.sum(g_dot_ATA_g_per_channel)  # Shape: scalar, stays on GPU

        if g_dot_ATA_g > 0:
            step_size = float(step_size_scaling * g_dot_g / g_dot_ATA_g)
        else:
            step_size = 0.01  # Fallback

        # Record step size for debugging
        if debug and step_sizes is not None:
            step_sizes.append(step_size)

        # Gradient descent step with projection to [0, 1]
        led_values_gpu = cp.clip(led_values_gpu - step_size * gradient, 0, 1)

    timing and timing.stop("optimization_loop")

    # Step 5: Convert back to spatial order and CPU
    led_values_final_gpu = led_values_gpu

    if timing:
        with timing.section("cpu_transfer", use_gpu_events=True):
            # Values are already in correct order (pattern generation handles ordering)
            led_values_spatial = cp.asnumpy(led_values_final_gpu)

            # Convert to uint8 [0, 255] range
            led_values_output = (led_values_spatial * 255.0).astype(np.uint8)
    else:
        # Values are already in correct order (pattern generation handles ordering)
        led_values_spatial = cp.asnumpy(led_values_final_gpu)

        # Convert to uint8 [0, 255] range
        led_values_output = (led_values_spatial * 255.0).astype(np.uint8)

    # Step 6: Compute error metrics if requested
    error_metrics = {}
    if compute_error_metrics:
        if timing:
            with timing.section("error_metrics", use_gpu_events=True):
                error_metrics = _compute_error_metrics(led_values_spatial, target_planar_uint8, at_matrix, debug=debug)
        else:
            error_metrics = _compute_error_metrics(led_values_spatial, target_planar_uint8, at_matrix, debug=debug)

    # Extract timing data if available
    timing_data = None
    if timing:
        timing_stats = timing.get_timing_data()
        timing_data = {section: data["duration"] for section, data in timing_stats["sections"].items()}

    # Create result
    result = FrameOptimizationResult(
        led_values=led_values_output,
        error_metrics=error_metrics,
        iterations=iteration + 1,
        converged=False,  # Fixed iterations, no convergence checking
        step_sizes=np.array(step_sizes) if step_sizes else None,
        timing_data=timing_data,
    )

    debug and logger.info(f"Optimization completed in {result.iterations} iterations")

    return result


def _calculate_atb(
    target_planar: np.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
    debug: bool = False,
) -> np.ndarray:
    """
    Calculate A^T @ b using mixed tensor format.

    Args:
        target_planar: Target frame in planar format (3, height, width)
        at_matrix: A^T matrix in mixed tensor format with 3D CUDA kernels
        debug: Enable debug output

    Returns:
        A^T @ b result (led_count, 3)
    """
    # Mixed tensor format: use 3D CUDA kernel with uint8 data
    debug and logger.info("Using mixed tensor format for A^T @ b")

    # For int8 mixed tensors, pass uint8 target directly
    if at_matrix.dtype == cp.uint8:
        target_gpu = cp.asarray(target_planar)  # Keep as uint8: (3, height, width)
    else:
        # For float32 mixed tensors, convert target to float32 [0,1]
        target_float32 = target_planar.astype(np.float32) / 255.0
        target_gpu = cp.asarray(target_float32)

    result = at_matrix.transpose_dot_product_3d(target_gpu)  # Shape: (led_count, 3), dtype: float32
    return cp.asnumpy(result)


def _compute_error_metrics(
    led_values: np.ndarray,
    target_planar: np.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Compute error metrics by forward pass through diffusion matrix.

    Args:
        led_values: LED values (3, led_count) in spatial order [0,1]
        target_planar: Target frame (3, height, width) [0,1]
        at_matrix: Mixed tensor matrix for forward computation
        debug: Enable debug output

    Returns:
        Dictionary of error metrics
    """
    try:
        # Forward pass: A @ led_values -> rendered_frame using mixed tensor
        led_values_gpu = cp.asarray(led_values.T)  # Convert to (led_count, 3)
        rendered_gpu = at_matrix.forward_pass_3d(led_values_gpu)  # Shape: (3, height, width)
        rendered_planar = cp.asnumpy(rendered_gpu)

        # Convert target to float32 [0,1] for error computation
        target_float32 = target_planar.astype(np.float32) / 255.0

        # Compute error metrics (both in [0,1] range)
        diff = rendered_planar - target_float32
        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))

        # Peak signal-to-noise ratio
        if mse > 0:
            psnr = float(20 * np.log10(1.0 / np.sqrt(mse)))
        else:
            psnr = float("inf")

        return {
            "mse": mse,
            "mae": mae,
            "psnr": psnr,
            "rendered_mean": float(np.mean(rendered_planar)),
            "target_mean": float(np.mean(target_planar)),
        }

    except Exception as e:
        debug and logger.error(f"Error computing metrics: {e}")
        return {"mse": float("inf"), "mae": float("inf")}


# Convenience functions for common use cases


def optimize_frame_with_tensors(
    target_frame: np.ndarray,
    mixed_tensor: SingleBlockMixedSparseTensor,
    dia_matrix: DiagonalATAMatrix,
    ata_inverse: Optional[np.ndarray] = None,
    **kwargs,
) -> FrameOptimizationResult:
    """
    Optimize frame using modern tensor formats: mixed tensor A^T and DIA A^T A matrices.

    Args:
        target_frame: Target frame (3, 480, 800) or (480, 800, 3)
        mixed_tensor: Mixed tensor for A^T @ b computation with 3D CUDA kernels
        dia_matrix: DIA format A^T A matrix for optimization
        ata_inverse: Optional ATA inverse matrices for optimal initialization (3, led_count, led_count)
        **kwargs: Additional arguments for optimize_frame_led_values

    Returns:
        FrameOptimizationResult with LED values in spatial order
    """
    return optimize_frame_led_values(
        target_frame=target_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dia_matrix,
        ata_inverse=ata_inverse,
        **kwargs,
    )
