#!/usr/bin/env python3
"""
PyTorch Sparse LED Optimization Engine with 4D Tensor Operations.

This module implements the LED optimization using PyTorch sparse COO tensors with
true 4D tensor operations (HEIGHT, WIDTH, LED_COUNT, 3) and einsum operations.
This approach maintains natural dimensional understanding without flattening and
should provide better parallelization for GPU computation.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from ..const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT

logger = logging.getLogger(__name__)


@dataclass
class PyTorchOptimizationResult:
    """Results from PyTorch sparse LED optimization process."""

    led_values: np.ndarray  # RGB values for each LED (led_count, 3) - range [0,255]
    error_metrics: Dict[str, float]  # Error metrics (mse, mae, etc.)
    optimization_time: float  # Time taken for optimization in seconds
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged
    target_frame: Optional[np.ndarray] = None  # Original target frame (for debugging)
    sparsity_info: Optional[Dict[str, Any]] = None  # Sparse tensor information
    flop_info: Optional[Dict[str, Any]] = None  # FLOP analysis information

    def get_led_count(self) -> int:
        """Get number of LEDs in result."""
        return self.led_values.shape[0]

    def get_total_error(self) -> float:
        """Get total optimization error."""
        return self.error_metrics.get("mse", float("inf"))


class PyTorchLEDOptimizer:
    """
    PyTorch sparse tensor LED optimization engine using 4D COO tensors and einsum.

    This implementation uses PyTorch sparse COO tensors with natural 4D tensor
    operations (HEIGHT, WIDTH, LED_COUNT, 3) and einsum for tensor contractions.
    This avoids flattening and provides cleaner dimensional understanding with
    better parallelization potential.
    """

    def __init__(
        self,
        diffusion_patterns_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize PyTorch sparse LED optimizer.

        Args:
            diffusion_patterns_path: Path to sparse diffusion matrix files
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.diffusion_patterns_path = diffusion_patterns_path or "config/diffusion_patterns"
        self.device = torch.device(device)

        # Optimization parameters for LSQR (tuned for real-time performance)
        self.max_iterations = 10
        self.convergence_threshold = 1e-3
        self.step_size_scaling = 0.9

        # 4D sparse COO tensor for diffusion patterns (H, W, LED_COUNT, 3)
        self._diffusion_tensor: Optional[torch.sparse.FloatTensor] = None

        # Pre-allocated tensors for optimization workspace
        self._workspace: Optional[Dict[str, torch.Tensor]] = None

        self._led_spatial_mapping: Optional[Dict[int, int]] = None
        self._led_positions: Optional[np.ndarray] = None
        self._matrix_loaded = False
        self._actual_led_count = LED_COUNT

        # Statistics
        self._optimization_count = 0
        self._total_optimization_time = 0.0
        self._total_flops = 0
        self._flops_per_iteration = 0

        logger.info(f"PyTorch LED optimizer initialized on device: {self.device}")

    def initialize(self) -> bool:
        """
        Initialize the PyTorch sparse LED optimizer.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Load sparse diffusion matrix and convert to 4D COO tensor
            if not self._load_and_convert_to_coo_tensor():
                logger.error("Failed to load and convert to 4D COO tensor")
                return False

            # Initialize workspace tensors
            self._initialize_workspace()

            logger.info("PyTorch LED optimizer initialized successfully")
            logger.info(f"LED count: {self._actual_led_count}")
            logger.info(f"Diffusion tensor shape: {self._diffusion_tensor.shape}")
            logger.info(f"Device: {self.device}")
            return True

        except Exception as e:
            logger.error(f"PyTorch LED optimizer initialization failed: {e}")
            return False

    def _load_and_convert_to_coo_tensor(self) -> bool:
        """
        Load sparse diffusion matrix and convert to 4D COO tensor (H, W, LED_COUNT, 3).

        Returns:
            True if loaded and converted successfully, False otherwise
        """
        try:
            # Load existing sparse matrix data
            patterns_path = f"{self.diffusion_patterns_path}.npz"
            if not Path(patterns_path).exists():
                logger.warning(f"Sparse matrix file not found at {patterns_path}")
                return False

            logger.info(f"Loading sparse matrix from {patterns_path}")
            data = np.load(patterns_path, allow_pickle=True)

            # Reconstruct the original sparse matrix (temporary)
            sparse_matrix_csc = sp.csc_matrix(
                (data["matrix_data"], data["matrix_indices"], data["matrix_indptr"]),
                shape=tuple(data["matrix_shape"]),
            )

            # Load metadata
            self._led_spatial_mapping = data["led_spatial_mapping"].item()
            self._led_positions = data["led_positions"]
            self._actual_led_count = sparse_matrix_csc.shape[1] // 3

            # Validate matrix dimensions
            expected_pixels = FRAME_HEIGHT * FRAME_WIDTH
            if sparse_matrix_csc.shape[0] != expected_pixels:
                logger.error(f"Matrix pixel dimension mismatch")
                return False

            logger.info("Converting to 4D PyTorch sparse COO tensor...")

            # Convert from flattened 2D matrix to 4D tensor representation
            self._diffusion_tensor = self._create_4d_coo_tensor(sparse_matrix_csc)

            # Calculate FLOPs per iteration
            self._calculate_flops_per_iteration()

            self._matrix_loaded = True
            logger.info("4D PyTorch COO tensor created successfully")

            # Log tensor information
            nnz = self._diffusion_tensor._nnz()
            logger.info(f"Diffusion tensor: {self._diffusion_tensor.shape}, nnz: {nnz:,}")
            logger.info(f"Tensor format: COO (coordinates + values)")

            return True

        except Exception as e:
            logger.error(f"Failed to load and convert sparse matrix: {e}")
            return False

    def _create_4d_coo_tensor(self, sparse_matrix_csc: sp.csc_matrix) -> torch.sparse.FloatTensor:
        """
        Convert 2D sparse matrix to 4D COO tensor (HEIGHT, WIDTH, LED_COUNT, 3).

        Args:
            sparse_matrix_csc: Input scipy sparse matrix (pixels, leds*3)

        Returns:
            PyTorch 4D COO sparse tensor on the specified device
        """
        logger.info("Creating 4D COO tensor from 2D sparse matrix...")

        # Convert to COO format for easier manipulation
        coo_matrix = sparse_matrix_csc.tocoo()

        # Get indices and values
        pixel_indices = coo_matrix.row  # Pixel index (0 to HEIGHT*WIDTH-1)
        led_rgb_indices = coo_matrix.col  # LED*3 index (0 to LED_COUNT*3-1)
        values = coo_matrix.data

        # Convert pixel indices to (height, width) coordinates
        h_indices = pixel_indices // FRAME_WIDTH
        w_indices = pixel_indices % FRAME_WIDTH

        # Convert LED*3 indices to LED index and RGB channel
        led_indices = led_rgb_indices // 3
        rgb_indices = led_rgb_indices % 3

        # Create 4D COO indices: (height, width, led, rgb)
        indices_4d = np.stack([h_indices, w_indices, led_indices, rgb_indices])

        # Convert to PyTorch tensors
        indices_tensor = torch.from_numpy(indices_4d).long().to(self.device)
        values_tensor = torch.from_numpy(values).float().to(self.device)

        # Create 4D tensor shape
        tensor_shape = (FRAME_HEIGHT, FRAME_WIDTH, self._actual_led_count, 3)

        # Create COO tensor
        diffusion_tensor = torch.sparse_coo_tensor(
            indices_tensor,
            values_tensor,
            tensor_shape,
            dtype=torch.float32,
            device=self.device,
        ).coalesce()

        logger.info(f"Created 4D COO tensor: {tensor_shape}, nnz: {diffusion_tensor._nnz():,}")
        return diffusion_tensor

    def _calculate_flops_per_iteration(self) -> None:
        """Calculate floating point operations per iteration for performance analysis."""
        if self._diffusion_tensor is None:
            return

        # FLOPs per iteration for 4D tensor operations:
        # Forward einsum contraction + gradient computation + step size calculation

        nnz_total = self._diffusion_tensor._nnz()
        pixels = FRAME_HEIGHT * FRAME_WIDTH
        leds = self._actual_led_count

        # Conservative estimate for einsum operations
        self._flops_per_iteration = (
            4 * nnz_total  # Forward einsum contraction
            + 4 * nnz_total  # Gradient computation (transpose operation)
            + 8 * pixels  # Residual and error calculations
            + 8 * leds * 3  # LED updates and norms
        )

        logger.debug(f"Estimated FLOPs per iteration: {self._flops_per_iteration:,}")

    def _initialize_workspace(self) -> None:
        """Initialize PyTorch tensor workspace for optimization."""
        if not self._matrix_loaded:
            return

        pixels = FRAME_HEIGHT * FRAME_WIDTH
        leds = self._actual_led_count

        logger.info(f"Initializing PyTorch workspace tensors...")

        # Pre-allocate workspace tensors for 4D tensor operations
        self._workspace = {
            # Target frame (HEIGHT, WIDTH, 3)
            "target_frame": torch.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=torch.float32, device=self.device),
            # LED values (LED_COUNT, 3)
            "led_values": torch.full((leds, 3), 0.5, dtype=torch.float32, device=self.device),
            # Workspace for optimization
            "residual_frame": torch.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=torch.float32, device=self.device),
            "gradient_leds": torch.zeros((leds, 3), dtype=torch.float32, device=self.device),
            "led_values_new": torch.zeros((leds, 3), dtype=torch.float32, device=self.device),
            # For step size computation
            "temp_frame": torch.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=torch.float32, device=self.device),
        }

        workspace_mb = sum(t.numel() * 4 for t in self._workspace.values()) / (1024 * 1024)
        logger.info(f"PyTorch workspace memory: {workspace_mb:.1f}MB")

    def optimize_frame(
        self,
        target_frame: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
    ) -> PyTorchOptimizationResult:
        """
        Optimize LED values using PyTorch sparse tensors with einsum operations.

        Args:
            target_frame: Target image (height, width, 3) in range [0, 255]
            initial_values: Initial LED values (led_count, 3), if None uses 0.5
            max_iterations: Override default max iterations

        Returns:
            PyTorchOptimizationResult with LED values and metrics
        """
        start_time = time.time()

        try:
            if not self._matrix_loaded:
                raise RuntimeError("PyTorch sparse tensors not loaded")

            # Validate input
            if target_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise ValueError(f"Target frame shape mismatch")

            # Preprocess target frame and transfer to GPU
            target_rgb_normalized = target_frame.astype(np.float32) / 255.0

            # Copy to workspace tensor (efficient GPU transfer)
            self._workspace["target_frame"][:] = torch.from_numpy(target_rgb_normalized).to(self.device)

            # Initialize LED values
            if initial_values is not None:
                initial_normalized = (initial_values / 255.0 if initial_values.max() > 1.0 else initial_values).astype(
                    np.float32
                )

                self._workspace["led_values"][:] = torch.from_numpy(initial_normalized).to(self.device)
            else:
                # Use pre-initialized 0.5 values (already in workspace)
                pass

            # Run PyTorch sparse optimization
            led_values_tensor = self._solve_lsqr_pytorch(max_iterations)

            # Convert back to numpy
            led_values_normalized = led_values_tensor.cpu().numpy()
            led_values_output = (led_values_normalized * 255.0).astype(np.uint8)

            # Compute error metrics
            error_metrics = self._compute_error_metrics(led_values_tensor)

            optimization_time = time.time() - start_time

            # Calculate FLOPs for this optimization
            iterations_used = max_iterations or self.max_iterations
            frame_flops = iterations_used * self._flops_per_iteration

            # Create result
            result = PyTorchOptimizationResult(
                led_values=led_values_output,
                error_metrics=error_metrics,
                optimization_time=optimization_time,
                iterations=iterations_used,
                converged=True,
                target_frame=target_frame.copy(),
                sparsity_info={
                    "tensor_shape_4d": self._diffusion_tensor.shape,
                    "nnz_total": self._diffusion_tensor._nnz(),
                },
                flop_info={
                    "total_flops": int(frame_flops),
                    "flops_per_iteration": int(self._flops_per_iteration),
                    "gflops": frame_flops / 1e9,
                    "gflops_per_second": frame_flops / (optimization_time * 1e9),
                },
            )

            # Update statistics
            self._total_flops += frame_flops
            self._optimization_count += 1
            self._total_optimization_time += optimization_time

            logger.debug(f"PyTorch optimization completed in {optimization_time:.3f}s")
            return result

        except Exception as e:
            optimization_time = time.time() - start_time
            logger.error(f"PyTorch optimization failed after {optimization_time:.3f}s: {e}")

            # Return error result
            return PyTorchOptimizationResult(
                led_values=np.zeros((self._actual_led_count, 3), dtype=np.uint8),
                error_metrics={"mse": float("inf"), "mae": float("inf")},
                optimization_time=optimization_time,
                iterations=0,
                converged=False,
                flop_info={"total_flops": 0, "gflops_per_second": 0.0},
            )

    def _solve_lsqr_pytorch(self, max_iterations: Optional[int]) -> torch.Tensor:
        """
        Solve LSQR optimization using 4D PyTorch sparse tensor with einsum operations.

        Args:
            max_iterations: Maximum iterations

        Returns:
            LED values tensor (led_count, 3) on device
        """
        max_iters = max_iterations or self.max_iterations

        # Get workspace tensors for cleaner code
        w = self._workspace

        for iteration in range(max_iters):
            # Forward pass: compute rendered frame from LED values using einsum
            # diffusion_tensor: (H, W, LED_COUNT, 3)
            # led_values: (LED_COUNT, 3)
            # result: (H, W, 3) = einsum('hwlc,lc->hwc', diffusion_tensor, led_values)
            rendered_frame = torch.einsum("hwlc,lc->hwc", self._diffusion_tensor, w["led_values"])

            # Compute residual: rendered - target
            w["residual_frame"][:] = rendered_frame - w["target_frame"]

            # Compute gradient: transpose einsum for gradient
            # gradient: (LED_COUNT, 3) = einsum('hwlc,hwc->lc', diffusion_tensor, residual)
            w["gradient_leds"][:] = torch.einsum("hwlc,hwc->lc", self._diffusion_tensor, w["residual_frame"])

            # Compute step size using the natural 4D operations
            step_size = self._compute_step_size_4d(w["gradient_leds"])

            # Gradient descent step with projection
            w["led_values_new"][:] = torch.clamp(w["led_values"] - step_size * w["gradient_leds"], 0, 1)

            # Check convergence
            delta = torch.norm(w["led_values_new"] - w["led_values"])
            if delta < self.convergence_threshold:
                logger.debug(f"Converged after {iteration + 1} iterations, delta: {delta:.6f}")
                break

            # Update LED values (swap references for efficiency)
            w["led_values"], w["led_values_new"] = w["led_values_new"], w["led_values"]

        return w["led_values"]

    def _compute_step_size_4d(self, gradient: torch.Tensor) -> float:
        """
        Compute step size for gradient descent using 4D tensor operations.

        Args:
            gradient: Gradient tensor (LED_COUNT, 3)

        Returns:
            Step size scalar
        """
        # Compute A @ gradient using einsum (like forward pass)
        # gradient: (LED_COUNT, 3)
        # Ag: (H, W, 3) = einsum('hwlc,lc->hwc', diffusion_tensor, gradient)
        Ag = torch.einsum("hwlc,lc->hwc", self._diffusion_tensor, gradient)

        # Compute norms
        grad_norm_sq = torch.sum(gradient * gradient)
        Ag_norm_sq = torch.sum(Ag * Ag)

        if Ag_norm_sq > 0:
            return self.step_size_scaling * grad_norm_sq / Ag_norm_sq
        else:
            return 0.01  # Fallback step size

    def _compute_error_metrics(self, led_values_tensor: torch.Tensor) -> Dict[str, float]:
        """Compute error metrics for the optimization result using 4D tensor operations."""
        w = self._workspace

        # Compute rendered frame from final LED values
        rendered_frame = torch.einsum("hwlc,lc->hwc", self._diffusion_tensor, led_values_tensor)

        # Compute residual
        residual_frame = rendered_frame - w["target_frame"]

        mse = torch.mean(residual_frame**2).item()
        mae = torch.mean(torch.abs(residual_frame)).item()
        max_error = torch.max(torch.abs(residual_frame)).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()

        return {
            "mse": mse,
            "mae": mae,
            "max_error": max_error,
            "rmse": rmse,
        }

    def get_optimizer_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        avg_time = self._total_optimization_time / max(1, self._optimization_count)

        stats = {
            "device": str(self.device),
            "matrix_loaded": self._matrix_loaded,
            "optimization_count": self._optimization_count,
            "total_optimization_time": self._total_optimization_time,
            "average_optimization_time": avg_time,
            "estimated_fps": 1.0 / avg_time if avg_time > 0 else 0.0,
            "led_count": self._actual_led_count,
            "frame_dimensions": (FRAME_WIDTH, FRAME_HEIGHT),
        }

        if self._matrix_loaded:
            avg_gflops_per_second = 0.0
            if avg_time > 0 and self._flops_per_iteration > 0:
                avg_gflops_per_second = (self.max_iterations * self._flops_per_iteration) / (avg_time * 1e9)

            stats.update(
                {
                    "tensor_shape_4d": self._diffusion_tensor.shape,
                    "nnz_total": self._diffusion_tensor._nnz(),
                    "tensor_format": "4D COO with einsum operations",
                    "flop_analysis": {
                        "flops_per_iteration": int(self._flops_per_iteration),
                        "total_flops_computed": int(self._total_flops),
                        "average_gflops_per_frame": (self.max_iterations * self._flops_per_iteration) / 1e9,
                        "average_gflops_per_second": avg_gflops_per_second,
                    },
                }
            )

        return stats
