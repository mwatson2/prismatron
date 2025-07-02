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
from .led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from .performance_timing import PerformanceTiming
from .single_block_sparse_tensor import SingleBlockMixedSparseTensor
from .spatial_ordering import reorder_block_values

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


def optimize_frame_led_values(
    target_frame: np.ndarray,
    AT_matrix: Union[LEDDiffusionCSCMatrix, SingleBlockMixedSparseTensor],
    ATA_matrix: Union[DiagonalATAMatrix, np.ndarray],
    initial_values: Optional[np.ndarray] = None,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-3,
    step_size_scaling: float = 0.9,
    compute_error_metrics: bool = False,
    debug: bool = False,
    enable_timing: bool = False,
) -> FrameOptimizationResult:
    """
    Optimize LED values for a target frame using gradient descent.

    This is a standalone function extracted from DenseLEDOptimizer that supports
    both mixed tensor and DIA matrix formats for flexible optimization.

    Args:
        target_frame: Target image in planar int8 format (3, 480, 800) or standard (480, 800, 3)
        AT_matrix: A^T matrix for computing A^T @ b
            - LEDDiffusionCSCMatrix: CSC sparse format
            - SingleBlockMixedSparseTensor: Mixed tensor format with 3D CUDA kernels
        ATA_matrix: A^T A matrix for gradient computation
            - DiagonalATAMatrix: DIA format with RCM ordering
            - np.ndarray: Dense format (led_count, led_count, 3)
        initial_values: Initial LED values (3, led_count) in spatial order, if None uses 0.5
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
    timing = (
        PerformanceTiming("frame_optimizer", enable_gpu_timing=True)
        if enable_timing
        else None
    )
    # Validate input frame format and convert to planar int8 if needed
    if target_frame.dtype != np.int8 and target_frame.dtype != np.uint8:
        raise ValueError(
            f"Target frame must be int8 or uint8, got {target_frame.dtype}"
        )

    # Handle both planar (3, H, W) and standard (H, W, 3) formats
    # Keep target as uint8 for int8 mixed tensors, convert to float32 for CSC matrices
    if target_frame.shape == (3, 480, 800):
        # Already in planar format - keep as uint8
        target_planar_uint8 = target_frame.astype(np.uint8)
    elif target_frame.shape == (480, 800, 3):
        # Convert from HWC to CHW planar format - keep as uint8
        target_planar_uint8 = target_frame.astype(np.uint8).transpose(
            2, 0, 1
        )  # (H, W, 3) -> (3, H, W)
    else:
        raise ValueError(
            f"Unsupported frame shape {target_frame.shape}, expected (3, 480, 800) or (480, 800, 3)"
        )

    debug and logger.info(f"Target frame shape: {target_planar_uint8.shape}")

    # Step 1: Calculate A^T @ b using the appropriate format
    debug and logger.info("Computing A^T @ b...")

    if timing:
        with timing.section("atb_calculation", use_gpu_events=True):
            ATb = _calculate_ATb(
                target_planar_uint8, AT_matrix, debug=debug
            )  # Shape: (3, led_count) or (led_count, 3)
    else:
        ATb = _calculate_ATb(
            target_planar_uint8, AT_matrix, debug=debug
        )  # Shape: (3, led_count) or (led_count, 3)

    # Ensure ATb is in (3, led_count) format for consistency
    if ATb.shape[0] != 3:
        ATb = ATb.T  # Convert (led_count, 3) -> (3, led_count)

    debug and logger.info(f"A^T @ b shape: {ATb.shape}")

    # Step 2: Initialize LED values in optimization order
    led_count = ATb.shape[1]
    if initial_values is not None:
        if initial_values.shape != (3, led_count):
            raise ValueError(
                f"Initial values shape {initial_values.shape} != (3, {led_count})"
            )
        # Normalize to [0,1] if needed
        led_values_normalized = (
            initial_values / 255.0 if initial_values.max() > 1.0 else initial_values
        ).astype(np.float32)
    else:
        # Default initialization
        led_values_normalized = np.full((3, led_count), 0.5, dtype=np.float32)

    debug and logger.info(f"Initial LED values shape: {led_values_normalized.shape}")

    # Convert matrices to optimization order if needed (RCM for DIA)
    if isinstance(ATA_matrix, DiagonalATAMatrix):
        # Convert ATb and initial values to RCM order for DIA matrix
        ATb_opt_order = ATA_matrix.reorder_led_values_to_rcm(ATb)
        led_values_opt_order = ATA_matrix.reorder_led_values_to_rcm(
            led_values_normalized
        )
        debug and logger.info("Using DIA matrix with RCM ordering")
    else:
        # Dense ATA matrix uses spatial order
        ATb_opt_order = ATb
        led_values_opt_order = led_values_normalized
        debug and logger.info("Using dense ATA matrix with spatial ordering")

    # Step 3: Transfer to GPU for optimization
    if timing:
        with timing.section("gpu_transfer", use_gpu_events=True):
            ATb_gpu = cp.asarray(ATb_opt_order)
            led_values_gpu = cp.asarray(led_values_opt_order)
    else:
        ATb_gpu = cp.asarray(ATb_opt_order)
        led_values_gpu = cp.asarray(led_values_opt_order)

    # Step 4: Gradient descent optimization loop
    debug and logger.info(f"Starting optimization: max_iterations={max_iterations}")
    step_sizes = [] if debug else None

    if timing:
        timing.start("optimization_loop")

    for iteration in range(max_iterations):
        # Compute gradient: A^T A @ x - A^T @ b
        if timing:
            with timing.section("ata_multiply", use_gpu_events=True):
                if isinstance(ATA_matrix, DiagonalATAMatrix):
                    # Use 3D DIA matrix operations
                    ATA_x = ATA_matrix.multiply_3d(led_values_gpu)
                    # multiply_3d returns cupy array, no conversion needed
                else:
                    # Dense matrix: ATA shape (led_count, led_count, 3), x shape (3, led_count)
                    # For each channel c: ATA[:,:,c] @ x[c,:] - use optimized matrix multiply
                    ATA_matrix_gpu = cp.asarray(ATA_matrix)
                    ATA_x = cp.zeros_like(led_values_gpu)
                    for c in range(3):
                        ATA_x[c] = ATA_matrix_gpu[:, :, c] @ led_values_gpu[c]
        else:
            if isinstance(ATA_matrix, DiagonalATAMatrix):
                # Use 3D DIA matrix operations
                ATA_x = ATA_matrix.multiply_3d(led_values_gpu)
                # multiply_3d returns cupy array, no conversion needed
            else:
                # Dense matrix: ATA shape (led_count, led_count, 3), x shape (3, led_count)
                # For each channel c: ATA[:,:,c] @ x[c,:] - use optimized matrix multiply
                ATA_matrix_gpu = cp.asarray(ATA_matrix)
                ATA_x = cp.zeros_like(led_values_gpu)
                for c in range(3):
                    ATA_x[c] = ATA_matrix_gpu[:, :, c] @ led_values_gpu[c]

        gradient = ATA_x - ATb_gpu  # Shape: (3, led_count)

        # Compute step size: (g^T @ g) / (g^T @ A^T A @ g)
        if timing:
            with timing.section("step_size_calculation", use_gpu_events=True):
                g_dot_g = cp.sum(gradient * gradient)

                if isinstance(ATA_matrix, DiagonalATAMatrix):
                    # Use DIA matrix for g^T @ A^T A @ g
                    g_dot_ATA_g_per_channel = ATA_matrix.g_ata_g_3d(gradient)
                    # g_ata_g_3d returns cupy array, no conversion needed
                    g_dot_ATA_g = cp.sum(g_dot_ATA_g_per_channel)
                else:
                    # Dense matrix computation - use optimized matrix multiply per channel
                    # gradient: (3, led_count), ATA: (led_count, led_count, 3)
                    ATA_matrix_gpu = cp.asarray(ATA_matrix)
                    g_dot_ATA_g = 0.0
                    for c in range(3):
                        ata_g = ATA_matrix_gpu[:, :, c] @ gradient[c]
                        g_dot_ATA_g += cp.dot(gradient[c], ata_g)

                if g_dot_ATA_g > 0:
                    step_size = float(step_size_scaling * g_dot_g / g_dot_ATA_g)
                else:
                    step_size = 0.01  # Fallback
        else:
            g_dot_g = cp.sum(gradient * gradient)

            if isinstance(ATA_matrix, DiagonalATAMatrix):
                # Use DIA matrix for g^T @ A^T A @ g
                g_dot_ATA_g_per_channel = ATA_matrix.g_ata_g_3d(gradient)
                # g_ata_g_3d returns cupy array, no conversion needed
                g_dot_ATA_g = cp.sum(g_dot_ATA_g_per_channel)
            else:
                # Dense matrix computation - use optimized matrix multiply per channel
                # gradient: (3, led_count), ATA: (led_count, led_count, 3)
                ATA_matrix_gpu = cp.asarray(ATA_matrix)
                g_dot_ATA_g = 0.0
                for c in range(3):
                    ata_g = ATA_matrix_gpu[:, :, c] @ gradient[c]
                    g_dot_ATA_g += cp.dot(gradient[c], ata_g)

            if g_dot_ATA_g > 0:
                step_size = float(step_size_scaling * g_dot_g / g_dot_ATA_g)
            else:
                step_size = 0.01  # Fallback

        if debug and step_sizes is not None:
            step_sizes.append(step_size)

        # Gradient descent step with projection to [0, 1]
        if timing:
            with timing.section("gradient_step", use_gpu_events=True):
                led_values_new = cp.clip(led_values_gpu - step_size * gradient, 0, 1)
        else:
            led_values_new = cp.clip(led_values_gpu - step_size * gradient, 0, 1)

        # Check convergence
        if timing:
            with timing.section("convergence_check", use_gpu_events=True):
                delta = cp.linalg.norm(led_values_new - led_values_gpu)
        else:
            delta = cp.linalg.norm(led_values_new - led_values_gpu)

        if delta < convergence_threshold:
            debug and logger.info(
                f"Converged after {iteration+1} iterations, delta: {delta:.6f}"
            )
            led_values_gpu = led_values_new
            break

        led_values_gpu = led_values_new

        if debug and (iteration + 1) % 5 == 0:
            logger.info(
                f"Iteration {iteration+1}: delta={delta:.6f}, step_size={step_size:.6f}"
            )

    if timing:
        timing.stop("optimization_loop")

    # Step 5: Convert back to spatial order and CPU
    led_values_final_gpu = led_values_gpu

    if timing:
        with timing.section("cpu_transfer_and_reorder", use_gpu_events=True):
            if isinstance(ATA_matrix, DiagonalATAMatrix):
                # Convert from RCM order back to spatial order
                led_values_spatial = ATA_matrix.reorder_led_values_from_rcm(
                    cp.asnumpy(led_values_final_gpu)
                )
            else:
                # Already in spatial order
                led_values_spatial = cp.asnumpy(led_values_final_gpu)

            # Convert to uint8 [0, 255] range
            led_values_output = (led_values_spatial * 255.0).astype(np.uint8)
    else:
        if isinstance(ATA_matrix, DiagonalATAMatrix):
            # Convert from RCM order back to spatial order
            led_values_spatial = ATA_matrix.reorder_led_values_from_rcm(
                cp.asnumpy(led_values_final_gpu)
            )
        else:
            # Already in spatial order
            led_values_spatial = cp.asnumpy(led_values_final_gpu)

        # Convert to uint8 [0, 255] range
        led_values_output = (led_values_spatial * 255.0).astype(np.uint8)

    # Step 6: Compute error metrics if requested
    error_metrics = {}
    if compute_error_metrics:
        if timing:
            with timing.section("error_metrics", use_gpu_events=True):
                error_metrics = _compute_error_metrics(
                    led_values_spatial, target_planar_uint8, AT_matrix, debug=debug
                )
        else:
            error_metrics = _compute_error_metrics(
                led_values_spatial, target_planar_uint8, AT_matrix, debug=debug
            )

    # Extract timing data if available
    timing_data = None
    if timing:
        timing_stats = timing.get_timing_data()
        timing_data = {
            section: data["duration"]
            for section, data in timing_stats["sections"].items()
        }

    # Create result
    result = FrameOptimizationResult(
        led_values=led_values_output,
        error_metrics=error_metrics,
        iterations=iteration + 1,
        converged=(delta < convergence_threshold if "delta" in locals() else False),
        step_sizes=np.array(step_sizes) if step_sizes else None,
        timing_data=timing_data,
    )

    debug and logger.info(f"Optimization completed in {result.iterations} iterations")

    return result


def _calculate_ATb(
    target_planar: np.ndarray,
    AT_matrix: Union[LEDDiffusionCSCMatrix, SingleBlockMixedSparseTensor],
    debug: bool = False,
) -> np.ndarray:
    """
    Calculate A^T @ b using the appropriate matrix format.

    Args:
        target_planar: Target frame in planar format (3, height, width)
        AT_matrix: A^T matrix in CSC or mixed tensor format
        debug: Enable debug output

    Returns:
        A^T @ b result (3, led_count) or (led_count, 3) depending on matrix format
    """
    if isinstance(AT_matrix, SingleBlockMixedSparseTensor):
        # Mixed tensor format: use 3D CUDA kernel with uint8 data
        debug and logger.info("Using mixed tensor format for A^T @ b")

        # For int8 mixed tensors, pass uint8 target directly
        if AT_matrix.dtype == cp.uint8:
            target_gpu = cp.asarray(target_planar)  # Keep as uint8: (3, height, width)
        else:
            # For float32 mixed tensors, convert target to float32 [0,1]
            target_float32 = target_planar.astype(np.float32) / 255.0
            target_gpu = cp.asarray(target_float32)

        result = AT_matrix.transpose_dot_product_3d(
            target_gpu
        )  # Shape: (led_count, 3), dtype: float32
        return cp.asnumpy(result)

    elif isinstance(AT_matrix, LEDDiffusionCSCMatrix):
        # CSC sparse format: use sparse matrix operations with float32 data
        debug and logger.info("Using CSC sparse format for A^T @ b")

        # Convert target to float32 [0,1] for CSC matrix operations
        target_float32 = target_planar.astype(np.float32) / 255.0

        # CSC matrix format: (pixels, led_count*3), columns: [LED0_R, LED0_G, LED0_B, ...]
        # We need to compute A^T @ target for each channel separately then combine

        csc_A = AT_matrix.to_csc_matrix()  # Get A matrix: (pixels, led_count*3)
        pixels = target_float32.shape[1] * target_float32.shape[2]
        led_count = csc_A.shape[1] // 3

        # Initialize result: (led_count, 3)
        ATb_result = np.zeros((led_count, 3), dtype=np.float32)

        # Process each channel separately
        for channel in range(3):
            target_channel = target_float32[channel].flatten()  # Shape: (pixels,)

            # Extract columns for this channel from A matrix
            # Channel 0 (R): columns 0, 3, 6, 9, ... (every 3rd starting from 0)
            # Channel 1 (G): columns 1, 4, 7, 10, ... (every 3rd starting from 1)
            # Channel 2 (B): columns 2, 5, 8, 11, ... (every 3rd starting from 2)
            channel_cols = np.arange(channel, csc_A.shape[1], 3)
            A_channel = csc_A[:, channel_cols]  # Shape: (pixels, led_count)

            # Compute A^T @ target for this channel
            ATb_channel = A_channel.T @ target_channel  # Shape: (led_count,)
            ATb_result[:, channel] = ATb_channel

        return ATb_result

    else:
        raise ValueError(f"Unsupported AT_matrix type: {type(AT_matrix)}")


def _compute_error_metrics(
    led_values: np.ndarray,
    target_planar: np.ndarray,
    AT_matrix: Union[LEDDiffusionCSCMatrix, SingleBlockMixedSparseTensor],
    debug: bool = False,
) -> Dict[str, float]:
    """
    Compute error metrics by forward pass through diffusion matrix.

    Args:
        led_values: LED values (3, led_count) in spatial order [0,1]
        target_planar: Target frame (3, height, width) [0,1]
        AT_matrix: Matrix for forward computation
        debug: Enable debug output

    Returns:
        Dictionary of error metrics
    """
    try:
        # Forward pass: A @ led_values -> rendered_frame
        if isinstance(AT_matrix, SingleBlockMixedSparseTensor):
            # Use mixed tensor forward pass
            led_values_gpu = cp.asarray(led_values.T)  # Convert to (led_count, 3)
            rendered_gpu = AT_matrix.forward_pass_3d(
                led_values_gpu
            )  # Shape: (3, height, width)
            rendered_planar = cp.asnumpy(rendered_gpu)

        elif isinstance(AT_matrix, LEDDiffusionCSCMatrix):
            # Use CSC forward pass with (pixels, led_count*3) format
            csc_A = AT_matrix.to_csc_matrix()  # Shape: (pixels, led_count*3)
            led_count = led_values.shape[1]
            height, width = target_planar.shape[1], target_planar.shape[2]

            # Initialize rendered frame
            rendered_planar = np.zeros((3, height, width), dtype=np.float32)

            # Process each channel separately
            for channel in range(3):
                # Get LED values for this channel
                led_channel = led_values[channel]  # Shape: (led_count,)

                # Extract A matrix columns for this channel
                channel_cols = np.arange(channel, csc_A.shape[1], 3)
                A_channel = csc_A[:, channel_cols]  # Shape: (pixels, led_count)

                # Forward pass: A @ led_values
                rendered_channel = A_channel @ led_channel  # Shape: (pixels,)

                # Reshape to spatial dimensions
                rendered_planar[channel] = rendered_channel.reshape(height, width)

        else:
            debug and logger.warning("Error metrics not supported for this matrix type")
            return {"mse": 0.0, "mae": 0.0}

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


def optimize_frame_with_dia_matrix(
    target_frame: np.ndarray,
    diffusion_csc: LEDDiffusionCSCMatrix,
    dia_matrix: DiagonalATAMatrix,
    **kwargs,
) -> FrameOptimizationResult:
    """
    Optimize frame using CSC sparse A^T and DIA A^T A matrices.

    Args:
        target_frame: Target frame (3, 480, 800) or (480, 800, 3)
        diffusion_csc: CSC sparse diffusion matrix for A^T @ b
        dia_matrix: DIA format A^T A matrix for optimization
        **kwargs: Additional arguments for optimize_frame_led_values

    Returns:
        FrameOptimizationResult with LED values in spatial order
    """
    return optimize_frame_led_values(
        target_frame=target_frame,
        AT_matrix=diffusion_csc,
        ATA_matrix=dia_matrix,
        **kwargs,
    )


def optimize_frame_with_mixed_tensor(
    target_frame: np.ndarray,
    mixed_tensor: SingleBlockMixedSparseTensor,
    ata_dense: np.ndarray,
    **kwargs,
) -> FrameOptimizationResult:
    """
    Optimize frame using mixed tensor A^T and dense A^T A matrices.

    Args:
        target_frame: Target frame (3, 480, 800) or (480, 800, 3)
        mixed_tensor: Mixed tensor for 3D CUDA operations
        ata_dense: Dense A^T A matrix (led_count, led_count, 3)
        **kwargs: Additional arguments for optimize_frame_led_values

    Returns:
        FrameOptimizationResult with LED values in spatial order
    """
    return optimize_frame_led_values(
        target_frame=target_frame,
        AT_matrix=mixed_tensor,
        ATA_matrix=ata_dense,
        **kwargs,
    )
