#!/usr/bin/env python3
"""
Standalone frame optimization function for LED displays.

This module provides a clean, standalone function for optimizing LED values
from target frames using either mixed tensor or DIA matrix formats.
Extracted from the LEDOptimizer class for modular usage.
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

    led_values: cp.ndarray  # RGB values for each LED (3, led_count) in spatial order [0,255] - GPU array only
    error_metrics: Dict[str, float]  # Error metrics (mse, mae, etc.)
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged
    step_sizes: Optional[np.ndarray] = None  # Step sizes per iteration (for debugging)
    timing_data: Optional[Dict[str, float]] = None  # Performance timing breakdown
    mse_per_iteration: Optional[np.ndarray] = None  # MSE value at each iteration (for convergence analysis)


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
    target_frame: cp.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
    ata_matrix: DiagonalATAMatrix,
    ata_inverse: Union[np.ndarray, DiagonalATAMatrix, Dict[str, any]],
    initial_values: Optional[np.ndarray] = None,
    max_iterations: int = 5,
    convergence_threshold: float = 0.3,
    step_size_scaling: float = 0.9,
    compute_error_metrics: bool = False,
    debug: bool = False,
    enable_timing: bool = False,
    track_mse_per_iteration: bool = False,
    compare_ata_inverse: Optional[Union[np.ndarray, DiagonalATAMatrix]] = None,
) -> FrameOptimizationResult:
    """
    Optimize LED values for a target frame using gradient descent with optimal ATA inverse initialization.

    This is a standalone function that uses modern tensor formats and ATA inverse initialization
    for optimal convergence: SingleBlockMixedSparseTensor for A^T @ b and DiagonalATAMatrix
    for efficient A^T A operations. Supports both fp32 and uint8 mixed tensor formats.

    Args:
        target_frame: Target image in uint8 format (3, 480, 800) or standard (480, 800, 3) - GPU cupy array
        at_matrix: A^T matrix for computing A^T @ b (SingleBlockMixedSparseTensor with 3D CUDA kernels)
                   - If dtype=uint8: uses uint8 x uint8 -> fp32 kernels with proper scaling
                   - If dtype=fp32: uses fp32 x fp32 -> fp32 kernels
        ata_matrix: A^T A matrix for gradient computation (DiagonalATAMatrix in DIA format with RCM ordering)
                    - Maintained in fp32 or fp16 format regardless of A matrix dtype
        ata_inverse: A^T A inverse matrices for optimal initialization [REQUIRED]
                     - Dense format: (3, led_count, led_count) numpy array
                     - DIA format: DiagonalATAMatrix object with 3D data
                     - Legacy DIA format: Dict with serialized DiagonalATAMatrix
        initial_values: Override for initial LED values (3, led_count), if None uses ATA inverse initialization
        max_iterations: Maximum optimization iterations
        convergence_threshold: Convergence threshold for delta norm
        step_size_scaling: Step size scaling factor (0.9 typical)
        compute_error_metrics: Whether to compute error metrics (slower)
        debug: Enable debug output and tracking
        enable_timing: Enable detailed performance timing breakdown
        track_mse_per_iteration: Whether to track MSE at each iteration (for convergence analysis)
        compare_ata_inverse: Optional second ATA inverse to compare initialization with (for debugging)

    Returns:
        FrameOptimizationResult with LED values in spatial order (3, led_count)
    """

    # Initialize performance timing if requested
    timing = PerformanceTiming("frame_optimizer", enable=False, enable_gpu_timing=False)  # Disable timing for now

    # Validate GPU input frame and format
    if not isinstance(target_frame, cp.ndarray):
        logger.error(f"Expected GPU cupy array, got {type(target_frame)}")
        raise ValueError(f"Target frame must be cupy GPU array, got {type(target_frame)}")

    if target_frame.dtype != cp.int8 and target_frame.dtype != cp.uint8:
        raise ValueError(f"Target frame must be int8 or uint8, got {target_frame.dtype}")

    # Handle both planar (3, H, W) and standard (H, W, 3) formats on GPU
    if target_frame.shape == (3, 480, 800):
        # Already in planar format - ensure uint8 and C-contiguous
        target_planar_uint8 = cp.ascontiguousarray(target_frame.astype(cp.uint8))
    elif target_frame.shape == (480, 800, 3):
        # Convert from HWC to CHW planar format on GPU
        target_planar_uint8 = cp.ascontiguousarray(
            target_frame.astype(cp.uint8).transpose(2, 0, 1)
        )  # (H, W, 3) -> (3, H, W)
    else:
        raise ValueError(f"Unsupported frame shape {target_frame.shape}, expected (3, 480, 800) or (480, 800, 3)")

    debug and logger.info(f"Target frame shape: {target_planar_uint8.shape}, dtype: {target_planar_uint8.dtype}")
    debug and logger.info(f"Mixed tensor dtype: {at_matrix.dtype}")

    # Step 1: Calculate A^T @ b using the appropriate format
    debug and logger.info("Computing A^T @ b...")

    ATb_gpu = _calculate_atb(target_planar_uint8, at_matrix, debug=debug)  # Shape: (3, led_count), dtype: fp32

    # ATb is already in correct (3, led_count) format and C-contiguous layout
    # The planar_output=True parameter eliminates transpose operations and memory layout issues

    debug and logger.info(f"A^T @ b shape: {ATb_gpu.shape}")

    # Step 2: Initialize LED values in optimization order - ATA inverse is now required
    led_count = ATb_gpu.shape[1]

    # Detect ATA inverse format and validate
    is_dia_format = isinstance(ata_inverse, DiagonalATAMatrix)
    is_legacy_dia_format = isinstance(ata_inverse, dict)

    if initial_values is not None:
        # Use provided initial values (override ATA inverse)
        if initial_values.shape != (3, led_count):
            raise ValueError(f"Initial values shape {initial_values.shape} != (3, {led_count})")
        # Normalize to [0,1] if needed
        led_values_gpu_raw = cp.asarray(
            (initial_values / 255.0 if initial_values.max() > 1.0 else initial_values).astype(np.float32)
        )
        led_values_gpu = led_values_gpu_raw  # No clipping needed for provided values
        debug and logger.info("Using provided initial values (overriding ATA inverse)")
    else:
        # Use ATA inverse for optimal initialization: x_init = (A^T A)^-1 * A^T b
        if is_dia_format:
            debug and logger.info("Using DIA format ATA inverse for optimal initialization")
            # Use DIA format approximation directly - same operation as ATA multiply
            led_values_gpu_raw = ata_inverse.multiply_3d(ATb_gpu)
        elif is_legacy_dia_format:
            debug and logger.info("Using legacy DIA format ATA inverse for optimal initialization")
            # Load the DIA ATA inverse matrix - unified format
            ata_inverse_dia = DiagonalATAMatrix.from_dict(ata_inverse)
            # Use DIA format approximation
            led_values_gpu_raw = ata_inverse_dia.multiply_3d(ATb_gpu)
        else:
            debug and logger.info("Using dense ATA inverse for optimal initialization")
            # Validate dense ATA inverse shape
            if ata_inverse.shape != (3, led_count, led_count):
                raise ValueError(f"ATA inverse shape {ata_inverse.shape} != (3, {led_count}, {led_count})")

            # Transfer to GPU for efficient computation
            ata_inverse_gpu = cp.asarray(ata_inverse)  # Shape: (3, led_count, led_count)

            # Compute optimal initial guess for all channels using efficient einsum
            # Efficient einsum: (3, led_count, led_count) @ (3, led_count) -> (3, led_count)
            led_values_gpu_raw = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_gpu)

        # Transfer back to CPU if needed and clamp to valid range
        led_values_gpu = led_values_gpu_raw.astype(cp.float32)
        led_values_gpu = cp.clip(led_values_gpu, 0.0, 1.0)

        debug and logger.info("Initialization completed using ATA inverse")

    debug and logger.info(f"Initial LED values shape: {led_values_gpu.shape}")

    # DEBUG: Compare with alternative ATA inverse if provided
    if compare_ata_inverse is not None:
        print("\n=== COMPARING ATA INVERSE INITIALIZATION ===")
        print(f"A^T @ b range: [{cp.asnumpy(ATb_gpu).min():.2f}, {cp.asnumpy(ATb_gpu).max():.2f}]")
        print(f"A^T @ b RMS: {float(cp.sqrt(cp.mean(ATb_gpu**2))):.2f}")

        # Get current initialization result (RAW, before clipping)
        led_current_raw_cpu = cp.asnumpy(led_values_gpu_raw)
        led_current_clipped_cpu = cp.asnumpy(led_values_gpu)
        print("\nPrimary ATA inverse initialization:")
        print(
            f"  LED values (raw): range=[{led_current_raw_cpu.min():.6f}, {led_current_raw_cpu.max():.6f}], RMS={np.sqrt(np.mean(led_current_raw_cpu**2)):.6f}"
        )
        print(
            f"  LED values (clipped): range=[{led_current_clipped_cpu.min():.6f}, {led_current_clipped_cpu.max():.6f}], RMS={np.sqrt(np.mean(led_current_clipped_cpu**2)):.6f}"
        )

        # Test comparison initialization
        if isinstance(compare_ata_inverse, DiagonalATAMatrix):
            print(f"  Type: DIA format (k={compare_ata_inverse.k}, bandwidth={compare_ata_inverse.bandwidth})")
            led_compare_gpu_raw = compare_ata_inverse.multiply_3d(ATb_gpu)
        else:
            print(f"  Type: Dense format {compare_ata_inverse.shape}")
            compare_gpu = cp.asarray(compare_ata_inverse)
            led_compare_gpu_raw = cp.einsum("ijk,ik->ij", compare_gpu, ATb_gpu)

        led_compare_gpu = cp.clip(led_compare_gpu_raw, 0.0, 1.0)
        led_compare_raw_cpu = cp.asnumpy(led_compare_gpu_raw)
        led_compare_clipped_cpu = cp.asnumpy(led_compare_gpu)

        print("\nComparison ATA inverse initialization:")
        print(
            f"  LED values (raw): range=[{led_compare_raw_cpu.min():.6f}, {led_compare_raw_cpu.max():.6f}], RMS={np.sqrt(np.mean(led_compare_raw_cpu**2)):.6f}"
        )
        print(
            f"  LED values (clipped): range=[{led_compare_clipped_cpu.min():.6f}, {led_compare_clipped_cpu.max():.6f}], RMS={np.sqrt(np.mean(led_compare_clipped_cpu**2)):.6f}"
        )

        # Compare the RAW results (before clipping)
        max_diff_raw = np.max(np.abs(led_current_raw_cpu - led_compare_raw_cpu))
        rms_diff_raw = np.sqrt(np.mean((led_current_raw_cpu - led_compare_raw_cpu) ** 2))
        current_rms_raw = np.sqrt(np.mean(led_current_raw_cpu**2))

        # Compare the CLIPPED results (after clipping)
        max_diff_clipped = np.max(np.abs(led_current_clipped_cpu - led_compare_clipped_cpu))
        rms_diff_clipped = np.sqrt(np.mean((led_current_clipped_cpu - led_compare_clipped_cpu) ** 2))
        current_rms_clipped = np.sqrt(np.mean(led_current_clipped_cpu**2))

        print("\nRAW initialization comparison (before clipping):")
        print(f"  Max difference: {max_diff_raw:.6f}")
        print(f"  RMS difference: {rms_diff_raw:.6f}")
        if current_rms_raw > 1e-10:
            print(f"  Relative error: {rms_diff_raw/current_rms_raw*100:.2f}%")

        print("\nCLIPPED initialization comparison (after clipping):")
        print(f"  Max difference: {max_diff_clipped:.6f}")
        print(f"  RMS difference: {rms_diff_clipped:.6f}")
        if current_rms_clipped > 1e-10:
            print(f"  Relative error: {rms_diff_clipped/current_rms_clipped*100:.2f}%")

        # Compute initial MSE for both
        initial_mse_current = _compute_mse_only(led_values_gpu, target_planar_uint8, at_matrix)
        initial_mse_compare = _compute_mse_only(led_compare_gpu, target_planar_uint8, at_matrix)

        print("\nInitial MSE comparison:")
        print(f"  Primary MSE: {initial_mse_current:.6f}")
        print(f"  Comparison MSE: {initial_mse_compare:.6f}")
        print(f"  MSE ratio: {initial_mse_compare/initial_mse_current:.2f}x")
        print("=== END COMPARISON ===\n")

    # Step 4: Gradient descent optimization loop
    debug and logger.info(f"Starting optimization: max_iterations={max_iterations}")
    step_sizes = [] if debug else None
    mse_values = [] if track_mse_per_iteration else None

    # Track initial MSE before any optimization steps (after clipping)
    if track_mse_per_iteration:
        initial_mse = _compute_mse_only(led_values_gpu, target_planar_uint8, at_matrix)
        mse_values.append(float(initial_mse))

    iteration = 0
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

        # Ensure LED values remain C-contiguous for next iteration's DIA operations
        if not led_values_gpu.flags.c_contiguous:
            led_values_gpu = cp.ascontiguousarray(led_values_gpu)

        # Track MSE after this iteration if requested
        if track_mse_per_iteration:
            current_mse = _compute_mse_only(led_values_gpu, target_planar_uint8, at_matrix)
            mse_values.append(float(current_mse))

    # Step 5: Convert back to spatial order - GPU output only
    # Values are already in correct order (pattern generation handles ordering)
    led_values_spatial_gpu = led_values_gpu
    # Convert to uint8 [0, 255] range on GPU
    led_values_output = (led_values_spatial_gpu * 255.0).astype(cp.uint8)

    # Step 6: Compute error metrics if requested
    error_metrics = {}
    if compute_error_metrics:
        # Convert to CPU for error metrics computation (temporary)
        led_values_for_metrics = cp.asnumpy(led_values_gpu)
        error_metrics = _compute_error_metrics(led_values_for_metrics, target_planar_uint8, at_matrix, debug=debug)

    # Extract timing data if available
    timing_data = None
    if timing.enabled:
        timing_stats = timing.get_timing_data()
        timing_data = {section: data["duration"] for section, data in timing_stats["sections"].items()}

    # Create result
    result = FrameOptimizationResult(
        led_values=led_values_output,
        error_metrics=error_metrics,
        iterations=max_iterations,  # Use actual max_iterations instead of iteration + 1
        converged=False,  # Fixed iterations, no convergence checking
        step_sizes=np.array(step_sizes) if step_sizes else None,
        timing_data=timing_data,
        mse_per_iteration=np.array(mse_values) if mse_values else None,
    )

    debug and logger.info(f"Optimization completed in {result.iterations} iterations")

    return result


def _calculate_atb(
    target_planar: np.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
    debug: bool = False,
) -> cp.ndarray:
    """
    Calculate A^T @ b using mixed tensor format with proper dtype handling.

    Args:
        target_planar: Target frame in planar format (3, height, width) uint8 [0,255]
        at_matrix: A^T matrix in mixed tensor format with 3D CUDA kernels
        debug: Enable debug output

    Returns:
        A^T @ b result (3, led_count) float32 - always normalized to [0,1] equivalent range
    """
    debug and logger.info("Using mixed tensor format for A^T @ b")

    # Handle different matrix dtypes appropriately
    if at_matrix.dtype == cp.uint8:
        # For uint8 mixed tensors: use uint8 x uint8 -> fp32 kernel with built-in scaling
        # The kernel applies / (255 * 255) scaling to produce [0,1] equivalent results
        target_gpu = cp.asarray(target_planar)  # Keep as uint8: (3, height, width)
        debug and logger.info("Using uint8 x uint8 -> fp32 kernel with automatic scaling")
    else:
        # For float32 mixed tensors: convert target to float32 [0,1] for fp32 x fp32 -> fp32 kernel
        target_float32 = target_planar.astype(np.float32) / 255.0
        target_gpu = cp.asarray(target_float32)
        debug and logger.info("Using fp32 x fp32 -> fp32 kernel")

    # Use planar_output=True to get result directly in (3, led_count) format
    # This eliminates the need for transpose operations and prevents F-contiguous memory layout issues
    result = at_matrix.transpose_dot_product_3d(target_gpu, planar_output=True)  # Shape: (3, led_count), dtype: float32
    return result


def _compute_mse_only(
    led_values_gpu: cp.ndarray,
    target_planar: np.ndarray,
    at_matrix: SingleBlockMixedSparseTensor,
) -> float:
    """
    Compute MSE only for fast convergence tracking.

    Args:
        led_values_gpu: LED values on GPU (3, led_count) in [0,1] range
        target_planar: Target frame (3, height, width) uint8 [0,255]
        at_matrix: Mixed tensor matrix for forward computation

    Returns:
        MSE value as float
    """
    try:
        # Forward pass: A @ led_values -> rendered_frame using mixed tensor
        led_values_transposed = led_values_gpu.T  # Convert to (led_count, 3)
        rendered_gpu = at_matrix.forward_pass_3d(led_values_transposed)  # Shape: (3, height, width), dtype: float32

        # Handle different matrix types for error computation
        if at_matrix.dtype == cp.uint8:
            # For uint8 A matrix: rendered_gpu is already in [0,1] range from forward_pass_3d
            # Convert target to [0,1] for comparison
            target_gpu = cp.asarray(target_planar, dtype=cp.float32) / 255.0
        else:
            # For float32 A matrix: both are in [0,1] range
            target_gpu = cp.asarray(target_planar, dtype=cp.float32) / 255.0

        # Compute MSE efficiently on GPU
        diff = rendered_gpu - target_gpu
        mse = float(cp.mean(diff**2))

        return mse

    except Exception:
        return float("inf")


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
        target_planar: Target frame (3, height, width) uint8 [0,255]
        at_matrix: Mixed tensor matrix for forward computation
        debug: Enable debug output

    Returns:
        Dictionary of error metrics
    """
    try:
        # Forward pass: A @ led_values -> rendered_frame using mixed tensor
        led_values_gpu = cp.asarray(led_values.T)  # Convert to (led_count, 3)
        rendered_gpu = at_matrix.forward_pass_3d(led_values_gpu)  # Shape: (3, height, width), dtype: float32
        rendered_planar = cp.asnumpy(rendered_gpu)

        # Handle different matrix types for error computation
        if at_matrix.dtype == cp.uint8:
            # For uint8 A matrix: rendered_planar is already in [0,1] range from forward_pass_3d
            # Convert target to [0,1] for comparison
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
        else:
            # For float32 A matrix: both are in [0,1] range
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
            "target_mean": float(np.mean(target_float32)),
        }

    except Exception as e:
        debug and logger.error(f"Error computing metrics: {e}")
        return {"mse": float("inf"), "mae": float("inf")}


# Convenience functions for common use cases


def optimize_frame_with_tensors(
    target_frame: cp.ndarray,
    mixed_tensor: SingleBlockMixedSparseTensor,
    dia_matrix: DiagonalATAMatrix,
    ata_inverse: Optional[np.ndarray] = None,
    **kwargs,
) -> FrameOptimizationResult:
    """
    Optimize frame using modern tensor formats: mixed tensor A^T and DIA A^T A matrices.

    Supports both uint8 and fp32 mixed tensors with automatic dtype handling.

    Args:
        target_frame: Target frame uint8 (3, 480, 800) or (480, 800, 3) - GPU cupy array
        mixed_tensor: Mixed tensor for A^T @ b computation with 3D CUDA kernels
                     - uint8 dtype: uses optimized uint8 x uint8 -> fp32 kernels
                     - fp32 dtype: uses fp32 x fp32 -> fp32 kernels
        dia_matrix: DIA format A^T A matrix for optimization (fp32 or fp16)
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
