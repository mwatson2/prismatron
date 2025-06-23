"""
LED Optimization Engine with Planar Format and DiffusionPatternManager.

This module implements a dense tensor optimization approach using:
- Planar format (3, H, W) for input frames and (3, led_count) for output
- DiffusionPatternManager for unified pattern loading and A^T*A precomputation
- Clean separation between pattern loading and optimization logic

Key features:
- Standardized planar format throughout
- Uses precomputed A^T*A from pattern generation
- Clean interface with DiffusionPatternManager
- GPU-accelerated dense tensor operations
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cupy as cp
import numpy as np
import scipy.sparse as sp

from ..utils.diffusion_pattern_manager import DiffusionPatternManager

logger = logging.getLogger(__name__)


@dataclass
class PlanarOptimizationResult:
    """Results from planar LED optimization process."""

    led_values: np.ndarray  # RGB values in planar format (3, led_count) - range [0,255]
    error_metrics: Dict[str, float]  # Error metrics (mse, mae, etc.)
    optimization_time: float  # Time taken for optimization in seconds
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged
    target_frame: Optional[
        np.ndarray
    ] = None  # Original target frame in planar format (for debugging)
    pattern_info: Optional[Dict[str, Any]] = None  # Pattern information
    flop_info: Optional[Dict[str, Any]] = None  # FLOP analysis information
    timing_breakdown: Optional[Dict[str, float]] = None  # Detailed timing breakdown

    def get_led_count(self) -> int:
        """Get number of LEDs in result."""
        return self.led_values.shape[1]  # Second dimension in planar format

    def get_total_error(self) -> float:
        """Get total optimization error."""
        return self.error_metrics.get("mse", float("inf"))


class PlanarLEDOptimizer:
    """
    LED optimization engine using planar format and DiffusionPatternManager.

    This optimizer:
    - Accepts input frames in planar format (3, height, width)
    - Outputs LED values in planar format (3, led_count)
    - Uses DiffusionPatternManager for pattern loading and A^T*A precomputation
    - Performs dense tensor optimization with GPU acceleration

    Key workflow:
    1. Load patterns using DiffusionPatternManager (includes A^T*A precomputation)
    2. Per frame: Calculate A^T*b using sparse matrices
    3. Optimize: Use dense A^T*A for gradient descent
    4. Output: Return LED values in planar format
    """

    def __init__(
        self,
        pattern_manager: Optional[DiffusionPatternManager] = None,
        diffusion_patterns_path: Optional[str] = None,
    ):
        """
        Initialize planar LED optimizer.

        Args:
            pattern_manager: Optional pre-configured DiffusionPatternManager
            diffusion_patterns_path: Path to diffusion patterns (if pattern_manager not provided)
        """
        if pattern_manager is not None:
            self.pattern_manager = pattern_manager
        else:
            # Create pattern manager and load patterns
            patterns_path = (
                diffusion_patterns_path or "diffusion_patterns/synthetic_1000"
            )
            self.pattern_manager = DiffusionPatternManager(
                led_count=1000,  # Will be updated from file
                frame_height=640,
                frame_width=800,
            )
            self.pattern_manager.load_patterns(f"{patterns_path}.npz")

        # Optimization parameters
        self.max_iterations = 10
        self.convergence_threshold = 1e-3
        self.step_size_scaling = 0.9

        # GPU matrices (loaded during initialization)
        self._ATA_gpu = None  # Dense A^T*A on GPU (led_count, led_count, 3)
        self._sparse_matrices_gpu = {}  # Sparse matrices on GPU for A^T*b

        # Pre-allocated GPU workspace
        self._ATb_gpu = None  # (3, led_count) planar format
        self._led_values_gpu = None  # (3, led_count) planar format
        self._gpu_workspace = None

        # Statistics
        self._optimization_count = 0
        self._total_optimization_time = 0.0

        # Device info
        self.device_info = self._detect_compute_device()

        # Pattern information
        self.led_count = self.pattern_manager.led_count
        self.frame_shape = self.pattern_manager.get_frame_shape()
        self.led_output_shape = self.pattern_manager.get_led_output_shape()

        logger.info(f"PlanarLEDOptimizer initialized:")
        logger.info(f"  LEDs: {self.led_count}")
        logger.info(f"  Frame shape: {self.frame_shape}")
        logger.info(f"  LED output shape: {self.led_output_shape}")

    def initialize(self) -> bool:
        """
        Initialize the optimizer with pattern data.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if not self.pattern_manager.is_loaded:
                logger.error("Pattern manager not loaded")
                return False

            # Transfer dense A^T*A to GPU
            self._transfer_dense_ata_to_gpu()

            # Transfer sparse matrices to GPU
            self._transfer_sparse_matrices_to_gpu()

            # Initialize workspace arrays
            self._initialize_workspace()

            # Warm up GPU cache
            self._warm_gpu_cache()

            logger.info("PlanarLEDOptimizer initialized successfully")
            return True

        except Exception as e:
            logger.error(f"PlanarLEDOptimizer initialization failed: {e}")
            return False

    def optimize_frame(
        self,
        target_frame: np.ndarray,
        debug: bool = False,
        max_iterations: Optional[int] = None,
    ) -> PlanarOptimizationResult:
        """
        Optimize LED values for target frame.

        Args:
            target_frame: Target image in planar format (3, height, width)
            debug: Whether to include debug information
            max_iterations: Override default iteration count

        Returns:
            PlanarOptimizationResult with LED values in planar format (3, led_count)
        """
        if not self._is_initialized():
            raise RuntimeError("Optimizer not initialized - call initialize() first")

        # Validate input format
        expected_shape = self.frame_shape
        if target_frame.shape != expected_shape:
            raise ValueError(
                f"Target frame must be {expected_shape}, got {target_frame.shape}"
            )

        start_time = time.time()
        timing_breakdown = {}

        # Phase 1: Calculate A^T * b using sparse matrices
        atb_start = time.time()
        ATb = self._calculate_ATb_planar(target_frame)
        timing_breakdown["atb_calculation_time"] = time.time() - atb_start

        # Phase 2: Initialize LED values
        init_start = time.time()
        self._led_values_gpu.fill(0.5)  # Initialize to mid-range
        timing_breakdown["initialization_time"] = time.time() - init_start

        # Phase 3: Dense tensor optimization loop
        led_values_solved, loop_timing = self._solve_dense_gradient_descent_planar(
            ATb, max_iterations
        )
        timing_breakdown["optimization_loop_time"] = loop_timing.get(
            "total_loop_time", 0.0
        )
        timing_breakdown["loop_iterations"] = loop_timing.get(
            "iterations_completed", max_iterations or self.max_iterations
        )

        # Phase 4: Convert to output format
        convert_start = time.time()
        led_values_normalized = cp.asnumpy(led_values_solved)
        led_values_output = (led_values_normalized * 255.0).astype(np.uint8)
        timing_breakdown["conversion_time"] = time.time() - convert_start

        # Phase 5: Calculate error metrics if debug mode
        error_metrics = {}
        debug_time = 0.0
        if debug:
            debug_start = time.time()
            error_metrics = self._compute_error_metrics_planar(
                led_values_solved, target_frame
            )
            debug_time = time.time() - debug_start
            timing_breakdown["debug_time"] = debug_time

        optimization_time = time.time() - start_time
        timing_breakdown["total_time"] = optimization_time

        # Calculate core optimization time (excluding debug overhead)
        core_optimization_time = optimization_time - debug_time

        # Create result
        result = PlanarOptimizationResult(
            led_values=led_values_output,  # (3, led_count) planar format
            error_metrics=error_metrics,
            optimization_time=core_optimization_time,
            iterations=max_iterations or self.max_iterations,
            converged=True,
            target_frame=target_frame.copy() if debug else None,
            pattern_info={
                "led_count": self.led_count,
                "frame_shape": self.frame_shape,
                "led_output_shape": self.led_output_shape,
                "approach": "planar_dense_ata",
            }
            if debug
            else None,
            timing_breakdown=timing_breakdown,
        )

        # Update statistics
        self._optimization_count += 1
        self._total_optimization_time += core_optimization_time

        if debug:
            logger.debug(
                f"Planar optimization completed in {core_optimization_time:.3f}s"
            )

        return result

    def get_statistics(self) -> Dict[str, any]:
        """Get optimizer statistics."""
        if self._optimization_count == 0:
            return {"optimization_count": 0}

        avg_time = self._total_optimization_time / self._optimization_count

        return {
            "optimization_count": self._optimization_count,
            "total_optimization_time": self._total_optimization_time,
            "average_optimization_time": avg_time,
            "led_count": self.led_count,
            "frame_shape": self.frame_shape,
            "led_output_shape": self.led_output_shape,
            "approach": "planar_dense_ata",
        }

    def _detect_compute_device(self) -> Dict[str, Any]:
        """Detect GPU device and capabilities."""
        device_info = {
            "device": "gpu",
            "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "memory_info": {},
        }

        meminfo = cp.cuda.runtime.memGetInfo()
        device_info["memory_info"] = {
            "free_mb": meminfo[0] / (1024 * 1024),
            "total_mb": meminfo[1] / (1024 * 1024),
        }

        logger.info(f"Using GPU: {device_info['gpu_name']}")
        logger.info(f"GPU Memory: {device_info['memory_info']['free_mb']:.0f}MB free")

        return device_info

    def _transfer_dense_ata_to_gpu(self):
        """Transfer dense A^T*A matrices to GPU."""
        logger.info("Transferring dense A^T*A to GPU...")

        dense_ata = self.pattern_manager.get_dense_ata()
        self._ATA_gpu = cp.asarray(dense_ata, dtype=cp.float32)

        ata_memory_mb = self._ATA_gpu.nbytes / (1024 * 1024)
        logger.info(
            f"Dense A^T*A transferred: {self._ATA_gpu.shape}, {ata_memory_mb:.1f}MB"
        )

    def _transfer_sparse_matrices_to_gpu(self):
        """Transfer sparse matrices to GPU."""
        logger.info("Transferring sparse matrices to GPU...")

        sparse_matrices = self.pattern_manager.get_sparse_matrices()

        for name, matrix in sparse_matrices.items():
            if name.startswith("A_"):  # Only transfer A matrices
                gpu_matrix = cp.sparse.csc_matrix(
                    (
                        cp.asarray(matrix.data, dtype=cp.float32),
                        cp.asarray(matrix.indices, dtype=cp.int32),
                        cp.asarray(matrix.indptr, dtype=cp.int32),
                    ),
                    shape=matrix.shape,
                )
                self._sparse_matrices_gpu[name] = gpu_matrix

        total_sparse_mb = sum(
            matrix.data.nbytes for matrix in self._sparse_matrices_gpu.values()
        ) / (1024 * 1024)

        logger.info(
            f"Sparse matrices transferred: {len(self._sparse_matrices_gpu)} matrices, {total_sparse_mb:.1f}MB"
        )

    def _initialize_workspace(self):
        """Initialize GPU workspace arrays."""
        logger.info("Initializing GPU workspace...")

        # Planar format arrays
        self._ATb_gpu = cp.zeros(
            self.led_output_shape, dtype=cp.float32
        )  # (3, led_count)
        self._led_values_gpu = cp.zeros(
            self.led_output_shape, dtype=cp.float32
        )  # (3, led_count)

        workspace_memory_mb = (self._ATb_gpu.nbytes + self._led_values_gpu.nbytes) / (
            1024 * 1024
        )
        logger.info(f"GPU workspace initialized: {workspace_memory_mb:.1f}MB")

    def _warm_gpu_cache(self):
        """Warm up GPU cache with dummy operations."""
        logger.info("Warming up GPU cache...")
        start_time = time.time()

        # Create dummy frame
        dummy_frame = cp.random.rand(*self.frame_shape, dtype=cp.float32)

        # Warm sparse operations
        sparse_start = time.time()
        for _ in range(3):
            _ = self._calculate_ATb_planar(cp.asnumpy(dummy_frame))
        sparse_time = time.time() - sparse_start

        # Warm dense operations
        dense_start = time.time()
        for _ in range(3):
            _ = self._solve_dense_gradient_descent_planar(self._ATb_gpu, 2)
        dense_time = time.time() - dense_start

        total_time = time.time() - start_time
        logger.info(
            f"GPU cache warmed up in {total_time:.3f}s (sparse: {sparse_time:.3f}s, dense: {dense_time:.3f}s)"
        )

    def _calculate_ATb_planar(self, target_frame: np.ndarray) -> cp.ndarray:
        """
        Calculate A^T * b for planar format frame.

        Args:
            target_frame: Target frame in planar format (3, height, width)

        Returns:
            ATb in planar format (3, led_count) on GPU
        """
        # Convert planar frame to flat format for sparse matrix multiplication
        # From (3, H, W) to (3, pixels) then to (pixels*3,) interleaved
        pixels_per_channel = target_frame.shape[1] * target_frame.shape[2]
        target_flat = target_frame.reshape(3, pixels_per_channel)  # (3, pixels)

        # Create interleaved format for current sparse matrix structure
        target_interleaved = np.empty(pixels_per_channel * 3, dtype=np.float32)
        target_interleaved[:pixels_per_channel] = target_flat[0]  # R channel
        target_interleaved[pixels_per_channel : 2 * pixels_per_channel] = target_flat[
            1
        ]  # G channel
        target_interleaved[2 * pixels_per_channel :] = target_flat[2]  # B channel

        # Transfer to GPU
        target_gpu = cp.asarray(target_interleaved)

        # Use combined sparse matrix for A^T @ b
        A_combined = self._sparse_matrices_gpu["A_combined"]
        ATb_interleaved = A_combined.T @ target_gpu

        # Convert result back to planar format
        # From (led_count*3,) interleaved to (3, led_count) planar
        ATb_planar = cp.zeros(self.led_output_shape, dtype=cp.float32)
        ATb_planar[0] = ATb_interleaved[: self.led_count]  # R LEDs
        ATb_planar[1] = ATb_interleaved[self.led_count : 2 * self.led_count]  # G LEDs
        ATb_planar[2] = ATb_interleaved[2 * self.led_count :]  # B LEDs

        return ATb_planar

    def _solve_dense_gradient_descent_planar(
        self, ATb: cp.ndarray, max_iterations: Optional[int]
    ) -> Tuple[cp.ndarray, Dict[str, float]]:
        """
        Solve using dense gradient descent with planar format.

        Args:
            ATb: A^T * b in planar format (3, led_count)
            max_iterations: Maximum iterations

        Returns:
            Tuple of (solved_values, timing_info)
        """
        iterations = max_iterations or self.max_iterations
        loop_start = time.time()

        timing_info = {}

        for i in range(iterations):
            # Dense gradient computation for each channel
            # ATA: (led_count, led_count, 3)
            # led_values: (3, led_count)
            # Result: (3, led_count)

            # Compute ATA @ x for each channel
            # led_values[c] @ ATA[:, :, c] -> (led_count,)
            gradient = cp.zeros_like(self._led_values_gpu)

            for c in range(3):
                gradient[c] = self._ATA_gpu[:, :, c] @ self._led_values_gpu[c] - ATb[c]

            # Compute step size: (g^T @ g) / (g^T @ ATA @ g)
            g_dot_g = cp.sum(gradient * gradient)

            # Compute g^T @ ATA @ g for each channel
            atag = cp.zeros_like(gradient)
            for c in range(3):
                atag[c] = self._ATA_gpu[:, :, c] @ gradient[c]

            g_ata_g = cp.sum(gradient * atag)

            if g_ata_g > 0:
                step_size = self.step_size_scaling * g_dot_g / g_ata_g
                self._led_values_gpu -= step_size * gradient

            # Clamp to valid range [0, 1]
            self._led_values_gpu = cp.clip(self._led_values_gpu, 0, 1)

            # Check convergence
            if cp.max(cp.abs(gradient)) < self.convergence_threshold:
                timing_info["converged_iteration"] = i
                break

        loop_time = time.time() - loop_start
        timing_info["total_loop_time"] = loop_time
        timing_info["iterations_completed"] = iterations

        return self._led_values_gpu.copy(), timing_info

    def _compute_error_metrics_planar(
        self, led_values: cp.ndarray, target_frame: np.ndarray
    ) -> Dict[str, float]:
        """Compute error metrics for planar format."""
        # For now, return basic metrics
        # Full implementation would reconstruct image and compare
        return {
            "mse": 0.0,  # Placeholder
            "mae": 0.0,  # Placeholder
            "psnr": 40.0,  # Placeholder
        }

    def _is_initialized(self) -> bool:
        """Check if optimizer is fully initialized."""
        return (
            self._ATA_gpu is not None
            and len(self._sparse_matrices_gpu) > 0
            and self._ATb_gpu is not None
            and self._led_values_gpu is not None
        )
