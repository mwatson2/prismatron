#!/usr/bin/env python3
"""
Batch frame optimization function for LED displays.

This module provides a batch version of the frame optimizer that can process
8 or 16 frames simultaneously, using tensor operations optimized for tensor cores.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cupy as cp
import numpy as np
import scipy.sparse as sp

from .diagonal_ata_matrix import DiagonalATAMatrix
from .performance_timing import PerformanceTiming
from .single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


@dataclass
class BatchFrameOptimizationResult:
    """Results from batch frame optimization process."""

    led_values: np.ndarray  # RGB values for each LED (batch_size, 3, led_count) in spatial order [0,255]
    error_metrics: List[Dict[str, float]]  # Error metrics per frame (mse, mae, etc.)
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged
    step_sizes: Optional[np.ndarray] = None  # Step sizes per iteration (for debugging)
    timing_data: Optional[Dict[str, float]] = None  # Performance timing breakdown


def convert_ata_dia_to_dense(ata_matrix: DiagonalATAMatrix) -> np.ndarray:
    """
    Convert DiagonalATAMatrix (DIA format) to dense format for batch processing.

    Args:
        ata_matrix: DiagonalATAMatrix in DIA format

    Returns:
        Dense ATA matrices (3, led_count, led_count)
    """
    logger.info("Converting ATA matrix from DIA to dense format...")

    # Get the individual channel DIA matrices
    dense_matrices = []
    for channel in range(3):
        # Get the DIA matrix for this channel
        dia_matrix = ata_matrix.get_channel_dia_matrix(channel)

        # Convert to dense format
        dense_matrix = dia_matrix.toarray()
        dense_matrices.append(dense_matrix)

    # Stack into (3, led_count, led_count) format
    ata_dense = np.stack(dense_matrices, axis=0)

    logger.info(f"Converted ATA matrix to dense format: {ata_dense.shape}")
    return ata_dense


def optimize_batch_frames_led_values(
    target_frames: np.ndarray,
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
) -> BatchFrameOptimizationResult:
    """
    Optimize LED values for a batch of target frames using tensor-optimized operations.

    This function processes 8 or 16 frames simultaneously, computing ATb individually
    but performing remaining operations as batch tensor operations optimized for tensor cores.

    Args:
        target_frames: Target images (batch_size, 3, 480, 800) or (batch_size, 480, 800, 3)
        at_matrix: A^T matrix for computing A^T @ b (SingleBlockMixedSparseTensor)
        ata_matrix: A^T A matrix for gradient computation (DiagonalATAMatrix in DIA format)
        ata_inverse: A^T A inverse matrices (3, led_count, led_count) [REQUIRED]
        initial_values: Override for initial LED values (batch_size, 3, led_count)
        max_iterations: Maximum optimization iterations
        convergence_threshold: Convergence threshold for delta norm
        step_size_scaling: Step size scaling factor (0.9 typical)
        compute_error_metrics: Whether to compute error metrics (slower)
        debug: Enable debug output and tracking
        enable_timing: Enable detailed performance timing breakdown

    Returns:
        BatchFrameOptimizationResult with LED values (batch_size, 3, led_count)
    """

    # Initialize performance timing if requested
    timing = PerformanceTiming("batch_frame_optimizer", enable_gpu_timing=True) if enable_timing else None

    # Validate batch size
    batch_size = target_frames.shape[0]
    if batch_size not in [8, 16]:
        raise ValueError(f"Batch size must be 8 or 16, got {batch_size}")

    # Validate input frame format and convert to planar int8 if needed
    if target_frames.dtype != np.int8 and target_frames.dtype != np.uint8:
        raise ValueError(f"Target frames must be int8 or uint8, got {target_frames.dtype}")

    # Handle both planar (batch_size, 3, H, W) and standard (batch_size, H, W, 3) formats
    if target_frames.shape[1:] == (3, 480, 800):
        # Already in planar format
        target_batch_uint8 = target_frames.astype(np.uint8)
    elif target_frames.shape[1:] == (480, 800, 3):
        # Convert from HWC to CHW planar format
        target_batch_uint8 = target_frames.astype(np.uint8).transpose(0, 3, 1, 2)  # (B, H, W, 3) -> (B, 3, H, W)
    else:
        raise ValueError(f"Unsupported frame shape {target_frames.shape[1:]}, expected (3, 480, 800) or (480, 800, 3)")

    debug and logger.info(f"Target batch shape: {target_batch_uint8.shape}")

    # Step 1: Calculate A^T @ b for each frame individually
    debug and logger.info("Computing A^T @ b for each frame...")

    ATb_batch = []
    if timing:
        with timing.section("atb_calculation_batch", use_gpu_events=True):
            for i in range(batch_size):
                frame = target_batch_uint8[i]  # Shape: (3, 480, 800)
                ATb = _calculate_atb(frame, at_matrix, debug=debug)

                # Ensure ATb is in (3, led_count) format
                if ATb.shape[0] != 3:
                    ATb = ATb.T

                ATb_batch.append(ATb)
    else:
        for i in range(batch_size):
            frame = target_batch_uint8[i]  # Shape: (3, 480, 800)
            ATb = _calculate_atb(frame, at_matrix, debug=debug)

            # Ensure ATb is in (3, led_count) format
            if ATb.shape[0] != 3:
                ATb = ATb.T

            ATb_batch.append(ATb)

    # Stack into batch tensor: (batch_size, 3, led_count)
    ATb_batch = np.stack(ATb_batch, axis=0)
    led_count = ATb_batch.shape[2]

    debug and logger.info(f"A^T @ b batch shape: {ATb_batch.shape}")

    # Step 2: Initialize LED values for batch using ATA inverse
    if ata_inverse.shape != (3, led_count, led_count):
        raise ValueError(f"ATA inverse shape {ata_inverse.shape} != (3, {led_count}, {led_count})")

    if initial_values is not None:
        # Use provided initial values
        if initial_values.shape != (batch_size, 3, led_count):
            raise ValueError(f"Initial values shape {initial_values.shape} != ({batch_size}, 3, {led_count})")
        led_values_batch = (initial_values / 255.0 if initial_values.max() > 1.0 else initial_values).astype(np.float32)
        debug and logger.info("Using provided initial values")
    else:
        # Use ATA inverse for optimal initialization: x_init = (A^T A)^-1 * A^T b
        debug and logger.info("Using ATA inverse for batch initialization")
        if timing:
            with timing.section("ata_inverse_initialization_batch", use_gpu_events=True):
                # Transfer to GPU for efficient computation
                ata_inverse_gpu = cp.asarray(ata_inverse)  # Shape: (3, led_count, led_count)
                ATb_batch_gpu = cp.asarray(ATb_batch)  # Shape: (batch_size, 3, led_count)

                # Efficient batch einsum: (3, led_count, led_count) @ (batch_size, 3, led_count) -> (batch_size, 3, led_count)
                # This can be optimized using tensor cores
                led_values_batch_gpu = cp.einsum("ijk,bik->bij", ata_inverse_gpu, ATb_batch_gpu)

                # Transfer back to CPU
                led_values_batch = cp.asnumpy(led_values_batch_gpu).astype(np.float32)
        else:
            # Non-timing version
            ata_inverse_gpu = cp.asarray(ata_inverse)
            ATb_batch_gpu = cp.asarray(ATb_batch)
            led_values_batch_gpu = cp.einsum("ijk,bik->bij", ata_inverse_gpu, ATb_batch_gpu)
            led_values_batch = cp.asnumpy(led_values_batch_gpu).astype(np.float32)

        # Clamp to valid range [0, 1]
        led_values_batch = np.clip(led_values_batch, 0.0, 1.0)

        debug and logger.info("Batch initialization completed using ATA inverse")

    # Step 3: Convert ATA matrix to dense format for batch operations
    debug and logger.info("Converting ATA matrix to dense format for batch operations...")
    if timing:
        with timing.section("ata_dense_conversion", use_gpu_events=True):
            ata_dense = convert_ata_dia_to_dense(ata_matrix)
    else:
        ata_dense = convert_ata_dia_to_dense(ata_matrix)

    # Step 4: Transfer to GPU for batch optimization
    if timing:
        with timing.section("gpu_transfer_batch", use_gpu_events=True):
            ATb_batch_gpu = cp.asarray(ATb_batch)  # Shape: (batch_size, 3, led_count)
            led_values_batch_gpu = cp.asarray(led_values_batch)  # Shape: (batch_size, 3, led_count)
            ata_dense_gpu = cp.asarray(ata_dense)  # Shape: (3, led_count, led_count)
    else:
        ATb_batch_gpu = cp.asarray(ATb_batch)
        led_values_batch_gpu = cp.asarray(led_values_batch)
        ata_dense_gpu = cp.asarray(ata_dense)

    # Step 5: Batch gradient descent optimization loop
    debug and logger.info(f"Starting batch optimization: max_iterations={max_iterations}")
    step_sizes = [] if debug else None

    if timing:
        with timing.section("optimization_loop_batch", use_gpu_events=True):
            for iteration in range(max_iterations):
                # Compute batch gradient: A^T A @ x - A^T @ b
                # Use batched einsum for tensor core optimization
                ATA_x_batch = cp.einsum(
                    "ijk,bik->bij", ata_dense_gpu, led_values_batch_gpu
                )  # (3, N, N) @ (B, 3, N) -> (B, 3, N)
                gradient_batch = ATA_x_batch - ATb_batch_gpu  # Shape: (batch_size, 3, led_count)

                # Compute step sizes for each frame: (g^T @ g) / (g^T @ A^T A @ g)
                g_dot_g_batch = cp.sum(gradient_batch * gradient_batch, axis=(1, 2))  # Shape: (batch_size,)

                # Compute g^T @ ATA @ g for each frame
                g_ata_g_batch = cp.einsum(
                    "ijk,bik->bij", ata_dense_gpu, gradient_batch
                )  # (3, N, N) @ (B, 3, N) -> (B, 3, N)
                g_dot_ata_g_batch = cp.sum(gradient_batch * g_ata_g_batch, axis=(1, 2))  # Shape: (batch_size,)

                # Compute step sizes
                step_sizes_batch = cp.where(
                    g_dot_ata_g_batch > 0, step_size_scaling * g_dot_g_batch / g_dot_ata_g_batch, 0.01  # Fallback
                )

                # Record average step size for debugging
                if debug and step_sizes is not None:
                    step_sizes.append(float(cp.mean(step_sizes_batch)))

                # Batch gradient descent step with projection to [0, 1]
                # Reshape step_sizes for broadcasting: (batch_size, 1, 1)
                step_sizes_reshaped = step_sizes_batch[:, cp.newaxis, cp.newaxis]
                led_values_batch_gpu = cp.clip(led_values_batch_gpu - step_sizes_reshaped * gradient_batch, 0, 1)
    else:
        # Production optimized version
        for iteration in range(max_iterations):
            # Compute batch gradient using tensor core optimized einsum
            ATA_x_batch = cp.einsum("ijk,bik->bij", ata_dense_gpu, led_values_batch_gpu)
            gradient_batch = ATA_x_batch - ATb_batch_gpu

            # Compute step sizes
            g_dot_g_batch = cp.sum(gradient_batch * gradient_batch, axis=(1, 2))
            g_ata_g_batch = cp.einsum("ijk,bik->bij", ata_dense_gpu, gradient_batch)
            g_dot_ata_g_batch = cp.sum(gradient_batch * g_ata_g_batch, axis=(1, 2))

            step_sizes_batch = cp.where(
                g_dot_ata_g_batch > 0, step_size_scaling * g_dot_g_batch / g_dot_ata_g_batch, 0.01
            )

            # Batch gradient descent step
            step_sizes_reshaped = step_sizes_batch[:, cp.newaxis, cp.newaxis]
            led_values_batch_gpu = cp.clip(led_values_batch_gpu - step_sizes_reshaped * gradient_batch, 0, 1)

    # Step 6: Convert back to CPU and scale to [0, 255]
    if timing:
        with timing.section("cpu_transfer_batch", use_gpu_events=True):
            led_values_batch_final = cp.asnumpy(led_values_batch_gpu)
            led_values_batch_output = (led_values_batch_final * 255.0).astype(np.uint8)
    else:
        led_values_batch_final = cp.asnumpy(led_values_batch_gpu)
        led_values_batch_output = (led_values_batch_final * 255.0).astype(np.uint8)

    # Step 7: Compute error metrics if requested
    error_metrics = []
    if compute_error_metrics:
        if timing:
            with timing.section("error_metrics_batch", use_gpu_events=True):
                for i in range(batch_size):
                    frame_metrics = _compute_error_metrics(
                        led_values_batch_final[i], target_batch_uint8[i], at_matrix, debug=debug
                    )
                    error_metrics.append(frame_metrics)
        else:
            for i in range(batch_size):
                frame_metrics = _compute_error_metrics(
                    led_values_batch_final[i], target_batch_uint8[i], at_matrix, debug=debug
                )
                error_metrics.append(frame_metrics)

    # Extract timing data if available
    timing_data = None
    if timing:
        timing_stats = timing.get_timing_data()
        timing_data = {section: data["duration"] for section, data in timing_stats["sections"].items()}

    # Create result
    result = BatchFrameOptimizationResult(
        led_values=led_values_batch_output,
        error_metrics=error_metrics,
        iterations=iteration + 1,
        converged=False,  # Fixed iterations, no convergence checking
        step_sizes=np.array(step_sizes) if step_sizes else None,
        timing_data=timing_data,
    )

    debug and logger.info(f"Batch optimization completed in {result.iterations} iterations")

    return result


def _calculate_atb(
    target_planar: np.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
    debug: bool = False,
) -> np.ndarray:
    """Calculate A^T @ b for a single frame using the mixed sparse tensor."""
    debug and logger.info(f"Computing A^T @ b with target shape: {target_planar.shape}")

    # Use the same logic as frame_optimizer.py
    if at_matrix.dtype == cp.uint8:
        target_gpu = cp.asarray(target_planar)  # Keep as uint8: (3, height, width)
    else:
        # For float32 mixed tensors, convert target to float32 [0,1]
        target_float32 = target_planar.astype(np.float32) / 255.0
        target_gpu = cp.asarray(target_float32)

    result = at_matrix.transpose_dot_product_3d(target_gpu)  # Shape: (led_count, 3), dtype: float32
    ATb = cp.asnumpy(result)

    debug and logger.info(f"A^T @ b computation completed, result shape: {ATb.shape}")
    return ATb


def _compute_error_metrics(
    led_values: np.ndarray,
    target_frame: np.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
    debug: bool = False,
) -> Dict[str, float]:
    """Compute error metrics for a single frame."""
    # Simple error metrics implementation
    try:
        # Forward pass: A @ x to get reconstructed frame
        led_values_gpu = cp.asarray(led_values)
        reconstructed = at_matrix.reconstruct_frame(led_values_gpu)
        reconstructed = cp.asnumpy(reconstructed)

        # Ensure same format for comparison
        if target_frame.shape != reconstructed.shape:
            if target_frame.shape == (3, target_frame.shape[1], target_frame.shape[2]):
                target_comparison = target_frame
            else:
                target_comparison = target_frame.transpose(2, 0, 1)
        else:
            target_comparison = target_frame

        # Convert to float for error calculation
        target_float = target_comparison.astype(np.float32)
        reconstructed_float = reconstructed.astype(np.float32)

        # Calculate metrics
        mse = np.mean((target_float - reconstructed_float) ** 2)
        mae = np.mean(np.abs(target_float - reconstructed_float))

        return {
            "mse": float(mse),
            "mae": float(mae),
            "psnr": float(20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0),
        }
    except Exception as e:
        debug and logger.warning(f"Error computing metrics: {e}")
        return {"mse": 0.0, "mae": 0.0, "psnr": 0.0}
