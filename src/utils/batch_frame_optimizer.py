#!/usr/bin/env python3
"""
Batch frame optimization function for LED displays.

This module provides a batch version of the frame optimizer that processes
8 frames simultaneously using optimized batch operations for all three channels.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import cupy as cp
import numpy as np

from .base_ata_matrix import BaseATAMatrix
from .batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from .single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


@dataclass
class BatchFrameOptimizationResult:
    """Results from batch frame optimization process."""

    led_values: cp.ndarray  # RGB values for each LED (batch_size, 3, led_count) [0,255] GPU array
    error_metrics: List[Dict[str, float]]  # Error metrics per frame (mse, mae, etc.)
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged
    step_sizes: Optional[np.ndarray] = None  # Step sizes per iteration (for debugging)
    timing_data: Optional[Dict[str, float]] = None  # Performance timing breakdown
    mse_per_iteration: Optional[np.ndarray] = None  # MSE values per iteration (batch_size, iterations+1)


def optimize_batch_frames_led_values(
    target_frames: cp.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
    ata_matrix: BatchSymmetricDiagonalATAMatrix,
    ata_inverse: np.ndarray,
    initial_values: Optional[cp.ndarray] = None,
    max_iterations: int = 5,
    convergence_threshold: float = 0.3,
    step_size_scaling: float = 0.9,
    compute_error_metrics: bool = False,
    debug: bool = False,
    track_mse_per_iteration: bool = False,
) -> BatchFrameOptimizationResult:
    """
    Optimize LED values for a batch of 8 or 16 target frames using batch tensor core operations.

    This function processes frames simultaneously using:
    - Batch transpose_dot_product_3d_batch() for A^T @ b computation
    - Cupy einsum for (A^T A)^-1 multiplication
    - Batch operations on SymmetricDiagonalATAMatrix for gradient computation

    Args:
        target_frames: Target images (batch_size, 3, H, W) or (batch_size, H, W, 3) - GPU cupy array, uint8
        at_matrix: A^T matrix for computing A^T @ b (SingleBlockMixedSparseTensor)
        ata_matrix: A^T A matrix for batch gradient computation (BatchSymmetricDiagonalATAMatrix)
        ata_inverse: A^T A inverse matrices (3, led_count, led_count) [REQUIRED]
        initial_values: Override for initial LED values (batch_size, 3, led_count) GPU array
        max_iterations: Maximum optimization iterations
        convergence_threshold: Convergence threshold for delta norm (not used currently)
        step_size_scaling: Step size scaling factor (0.9 typical)
        compute_error_metrics: Whether to compute error metrics (slower)
        debug: Enable debug output and tracking
        track_mse_per_iteration: Track MSE at each iteration for convergence analysis

    Returns:
        BatchFrameOptimizationResult with LED values (batch_size, 3, led_count) on GPU
    """
    # Validate batch size
    batch_size = target_frames.shape[0]
    if batch_size not in [8, 16]:
        raise ValueError(f"Batch size must be 8 or 16 for tensor core operations, got {batch_size}")

    # Validate GPU input
    if not isinstance(target_frames, cp.ndarray):
        raise ValueError(f"Target frames must be cupy GPU array, got {type(target_frames)}")

    if target_frames.dtype != cp.uint8:
        raise ValueError(f"Target frames must be uint8, got {target_frames.dtype}")

    # CRITICAL ALIGNMENT CHECKS for tensor core operations
    led_count = at_matrix.batch_size

    # 1. LED count must be multiple of 16 for BatchSymmetricDiagonalATAMatrix
    if led_count % 16 != 0:
        raise ValueError(
            f"LED count must be multiple of 16 for batch tensor core operations. "
            f"Got {led_count}. Use regular SymmetricDiagonalATAMatrix for non-aligned counts."
        )

    # 2. Validate that ata_matrix is the correct batch type
    if not isinstance(ata_matrix, BatchSymmetricDiagonalATAMatrix):
        raise TypeError(
            f"ata_matrix must be BatchSymmetricDiagonalATAMatrix for batch operations, "
            f"got {type(ata_matrix)}. Use regular optimize_frame_led_values for single frames."
        )

    # 3. Validate batch size matches matrix batch size
    if ata_matrix.batch_size != batch_size:
        raise ValueError(f"ATA matrix batch size ({ata_matrix.batch_size}) must match frame batch size ({batch_size})")

    # Handle both planar (batch, 3, H, W) and standard (batch, H, W, 3) formats
    if len(target_frames.shape) == 4 and target_frames.shape[1] == 3:
        # Already in planar format (8, 3, H, W)
        target_batch_planar = cp.ascontiguousarray(target_frames)
        height, width = target_frames.shape[2], target_frames.shape[3]
    elif len(target_frames.shape) == 4 and target_frames.shape[3] == 3:
        # Convert from HWC to CHW planar format (8, H, W, 3) -> (8, 3, H, W)
        target_batch_planar = cp.ascontiguousarray(target_frames.transpose(0, 3, 1, 2))
        height, width = target_frames.shape[1], target_frames.shape[2]
    else:
        raise ValueError(f"Unsupported frame shape {target_frames.shape}, expected (8, 3, H, W) or (8, H, W, 3)")

    if debug:
        logger.info(f"Target batch shape: {target_batch_planar.shape}")

    # Step 1: Calculate A^T @ b for batch using batch operation
    if debug:
        logger.info("Computing A^T @ b for batch using batch kernel...")

    # Use batch operation for all 8 frames at once
    # Returns shape (8, led_count, 3) with interleaved=False
    ATb_batch = at_matrix.transpose_dot_product_3d_batch(
        target_batch_planar,
        planar_output=False,  # Get (batch, leds, channels)
        use_warp_kernel=True,  # Use optimized V4 kernel for uint8
    )

    # Transpose to (batch, channels, leds) for consistency
    ATb_batch = ATb_batch.transpose(0, 2, 1)  # (8, leds, 3) -> (8, 3, leds)
    led_count = ATb_batch.shape[2]

    if debug:
        logger.info(f"A^T @ b batch shape: {ATb_batch.shape}")

    # Step 2: Initialize LED values using ATA inverse
    if ata_inverse.shape != (3, led_count, led_count):
        raise ValueError(f"ATA inverse shape {ata_inverse.shape} != (3, {led_count}, {led_count})")

    if initial_values is not None:
        # Use provided initial values
        if not isinstance(initial_values, cp.ndarray):
            raise ValueError("Initial values must be GPU cupy array")
        if initial_values.shape != (8, 3, led_count):
            raise ValueError(f"Initial values shape {initial_values.shape} != (8, 3, {led_count})")
        # Normalize to [0,1] if needed
        if initial_values.max() > 1.0:
            led_values_batch = (initial_values / 255.0).astype(cp.float32)
        else:
            led_values_batch = initial_values.astype(cp.float32)
        if debug:
            logger.info("Using provided initial values")
    else:
        # Use ATA inverse for optimal initialization: x_init = (A^T A)^-1 * A^T b
        if debug:
            logger.info("Using ATA inverse for optimal batch initialization")

        # Transfer to GPU for efficient computation
        ata_inverse_gpu = cp.asarray(ata_inverse)  # Shape: (3, led_count, led_count)

        # Compute optimal initial guess for all frames and channels using efficient einsum
        # ata_inverse_gpu: (3, led_count, led_count) = (channels, i, j)
        # ATb_batch: (batch_size, 3, led_count) = (batch, channels, j)
        # Result: (batch_size, 3, led_count) = (batch, channels, i)
        led_values_batch = cp.einsum("cij,bcj->bci", ata_inverse_gpu, ATb_batch)

        # Clamp to valid range [0, 1]
        led_values_batch = cp.clip(led_values_batch, 0.0, 1.0)

        if debug:
            logger.info("Batch initialization completed using ATA inverse")

    if debug:
        logger.info(f"Initial LED values batch shape: {led_values_batch.shape}")

    # Step 3: Track MSE if requested
    mse_values: Optional[List[np.ndarray]] = [] if track_mse_per_iteration else None

    if track_mse_per_iteration and mse_values is not None:
        # Compute initial MSE for all frames
        initial_mse = _compute_batch_mse(led_values_batch, target_batch_planar, at_matrix)
        mse_values.append(initial_mse)  # Shape: (8,)

    # Step 4: Batch gradient descent optimization loop
    if debug:
        logger.info(f"Starting batch optimization: max_iterations={max_iterations}")
    step_sizes: Optional[List[float]] = [] if debug else None

    for iteration in range(max_iterations):
        # Process all 8 frames together using batch operations
        # led_values_batch: (8, 3, led_count)

        # Use batch-specific methods based on batch size
        if batch_size == 8:
            # Compute ATA @ x for all frames and channels at once using 8-frame batch method
            ATA_x_batch = ata_matrix.multiply_batch8_3d(led_values_batch)  # (8, 3, led_count)

            # Compute gradient: ATA @ x - ATb
            gradient_batch = ATA_x_batch - ATb_batch  # (8, 3, led_count)

            # Compute step sizes for each frame
            # g^T @ g for each frame
            g_dot_g_batch = cp.sum(gradient_batch * gradient_batch, axis=(1, 2))  # (8,)

            # Compute g^T @ ATA @ g for all frames at once using batch method
            g_ata_g_channels_batch = ata_matrix.g_ata_g_batch_3d(gradient_batch)  # (8, 3)
            # Sum across channels for each frame
            g_dot_ata_g_batch = cp.sum(g_ata_g_channels_batch, axis=1)  # (8,)

        elif batch_size == 16:
            # Compute ATA @ x for all frames and channels at once using 16-frame batch method
            ATA_x_batch = ata_matrix.multiply_batch_3d(led_values_batch)  # (16, 3, led_count)

            # Compute gradient: ATA @ x - ATb
            gradient_batch = ATA_x_batch - ATb_batch  # (16, 3, led_count)

            # Compute step sizes for each frame
            # g^T @ g for each frame
            g_dot_g_batch = cp.sum(gradient_batch * gradient_batch, axis=(1, 2))  # (16,)

            # Compute g^T @ ATA @ g for all frames at once using 16-frame batch method
            g_ata_g_channels_batch = ata_matrix.g_ata_g_batch_3d(gradient_batch)  # (16, 3)
            # Sum across channels for each frame
            g_dot_ata_g_batch = cp.sum(g_ata_g_channels_batch, axis=1)  # (16,)

        else:
            raise ValueError(f"Unsupported batch size {batch_size}. Must be 8 or 16.")

        # Compute step sizes with safety check
        step_sizes_batch = cp.where(
            g_dot_ata_g_batch > 0, step_size_scaling * g_dot_g_batch / g_dot_ata_g_batch, 0.01  # Fallback
        )

        # Record average step size for debugging
        if debug and step_sizes is not None:
            step_sizes.append(float(cp.mean(step_sizes_batch)))

        # Batch gradient descent step with projection to [0, 1]
        # Reshape step_sizes for broadcasting: (batch_size, 1, 1)
        step_sizes_reshaped = step_sizes_batch[:, cp.newaxis, cp.newaxis]
        led_values_batch = cp.clip(led_values_batch - step_sizes_reshaped * gradient_batch, 0, 1)

        # Track MSE after this iteration if requested
        if track_mse_per_iteration and mse_values is not None:
            current_mse = _compute_batch_mse(led_values_batch, target_batch_planar, at_matrix)
            mse_values.append(current_mse)  # Shape: (8,)

    # Step 5: Convert to output format [0, 255] uint8
    led_values_output = (led_values_batch * 255.0).astype(cp.uint8)

    # Step 6: Compute error metrics if requested
    error_metrics = []
    if compute_error_metrics:
        for frame_idx in range(batch_size):
            frame_metrics = _compute_frame_error_metrics(
                led_values_batch[frame_idx],
                target_batch_planar[frame_idx],
                at_matrix,
                debug=debug,
            )
            error_metrics.append(frame_metrics)

    # Create result
    result = BatchFrameOptimizationResult(
        led_values=led_values_output,
        error_metrics=error_metrics,
        iterations=max_iterations,
        converged=False,  # Fixed iterations, no convergence checking
        step_sizes=np.array(step_sizes) if step_sizes else None,
        timing_data=None,  # Timing removed for simplicity
        mse_per_iteration=cp.asnumpy(cp.stack(mse_values)) if mse_values else None,  # (iterations+1, 8)
    )

    if debug:
        logger.info(f"Batch optimization completed in {result.iterations} iterations")

    return result


def _compute_batch_mse(
    led_values_batch: cp.ndarray,
    target_batch_planar: cp.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
) -> cp.ndarray:
    """
    Compute MSE for batch of frames.

    Args:
        led_values_batch: LED values (batch_size, 3, led_count) in [0,1] range
        target_batch_planar: Target frames (batch_size, 3, 480, 800) uint8
        at_matrix: Mixed tensor for forward computation

    Returns:
        MSE values for each frame (batch_size,)
    """
    batch_size = led_values_batch.shape[0]
    mse_values = cp.zeros(batch_size, dtype=cp.float32)

    for frame_idx in range(batch_size):
        try:
            # Get LED values for this frame and transpose
            led_frame = led_values_batch[frame_idx].T  # (3, led_count) -> (led_count, 3)

            # Forward pass
            rendered_gpu = at_matrix.forward_pass_3d(led_frame)  # (3, H, W)

            # Convert target to [0,1] for comparison
            target_gpu = target_batch_planar[frame_idx].astype(cp.float32) / 255.0

            # Compute MSE
            diff = rendered_gpu - target_gpu
            mse_values[frame_idx] = cp.mean(diff**2)

        except Exception:
            mse_values[frame_idx] = float("inf")

    return mse_values


def _compute_frame_error_metrics(
    led_values: cp.ndarray,
    target_planar: cp.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Compute error metrics for a single frame.

    Args:
        led_values: LED values (3, led_count) in [0,1] range - GPU array
        target_planar: Target frame (3, 480, 800) uint8 - GPU array
        at_matrix: Mixed tensor for forward computation
        debug: Enable debug output

    Returns:
        Dictionary of error metrics
    """
    try:
        # Forward pass: A @ led_values -> rendered_frame
        led_values_transposed = led_values.T  # (3, led_count) -> (led_count, 3)
        rendered_gpu = at_matrix.forward_pass_3d(led_values_transposed)  # (3, H, W)

        # Convert target to [0,1] for comparison
        target_float32 = target_planar.astype(cp.float32) / 255.0

        # Compute error metrics
        diff = rendered_gpu - target_float32
        mse = float(cp.mean(diff**2))
        mae = float(cp.mean(cp.abs(diff)))

        # Peak signal-to-noise ratio
        if mse > 0:
            psnr = float(20 * np.log10(1.0 / np.sqrt(mse)))
        else:
            psnr = float("inf")

        return {
            "mse": mse,
            "mae": mae,
            "psnr": psnr,
            "rendered_mean": float(cp.mean(rendered_gpu)),
            "target_mean": float(cp.mean(target_float32)),
        }

    except Exception as e:
        if debug:
            logger.error(f"Error computing metrics: {e}")
        return {"mse": float("inf"), "mae": float("inf"), "psnr": 0.0}
