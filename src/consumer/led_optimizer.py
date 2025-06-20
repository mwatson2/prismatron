"""
LED Optimization Engine with Sparse Matrix LSQR.

This module implements the core optimization algorithm that finds LED brightness
values to best approximate a target image using sparse diffusion matrices. The optimization
solves: minimize ||A*x - b||² where A is sparse diffusion matrix, x is LED brightness, b is target.

Uses sparse CSC matrices with LSQR gradient descent and projection for
real-time performance targeting 15fps with memory efficiency.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix
    from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from ..const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from sparse LED optimization process."""

    led_values: np.ndarray  # RGB values for each LED (led_count, 3) - range [0,255]
    error_metrics: Dict[str, float]  # Error metrics (mse, mae, etc.)
    optimization_time: float  # Time taken for optimization in seconds
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged
    target_frame: Optional[np.ndarray] = None  # Original target frame (for debugging)
    sparsity_info: Optional[Dict[str, Any]] = None  # Sparse matrix information

    def get_led_count(self) -> int:
        """Get number of LEDs in result."""
        return self.led_values.shape[0]

    def get_total_error(self) -> float:
        """Get total optimization error."""
        return self.error_metrics.get("mse", float("inf"))


class LEDOptimizer:
    """
    Sparse LED optimization engine using LSQR gradient descent.

    Implements memory-efficient optimization to solve the inverse lighting
    problem: minimize ||A*x - b||² where A is sparse diffusion matrix,
    x is LED brightness vector, and b is target image.

    Uses sparse CSC matrices for optimal A^T operations in LSQR iterations,
    with optional GPU acceleration via CuPy for real-time performance.
    """

    def __init__(
        self,
        diffusion_patterns_path: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize sparse LED optimizer.

        Args:
            diffusion_patterns_path: Path to sparse diffusion matrix files
            use_gpu: Whether to use GPU acceleration (requires CuPy)
        """
        self.diffusion_patterns_path = (
            diffusion_patterns_path or "config/diffusion_patterns"
        )
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        # Optimization parameters for LSQR
        self.max_iterations = 50  # Fewer iterations for real-time performance
        self.convergence_threshold = 1e-6
        self.step_size_scaling = 0.8  # Conservative step scaling for stability

        # Sparse matrix state
        self._sparse_matrix_csc: Optional[
            sp.csc_matrix
        ] = None  # CSC for A^T operations
        self._sparse_matrix_csr: Optional[sp.csr_matrix] = None  # CSR for A operations
        self._gpu_matrix_csc = None  # GPU CSC matrix
        self._gpu_matrix_csr = None  # GPU CSR matrix
        self._led_spatial_mapping: Optional[Dict[int, int]] = None
        self._led_positions: Optional[np.ndarray] = None
        self._matrix_loaded = False
        self._actual_led_count = LED_COUNT

        # Statistics
        self._optimization_count = 0
        self._total_optimization_time = 0.0

        # Detect compute capability
        self.device_info = self._detect_compute_device()

    def _detect_compute_device(self) -> Dict[str, Any]:
        """
        Detect best available compute device and capabilities.

        Returns:
            Dictionary with device information
        """
        device_info = {
            "cupy_available": CUPY_AVAILABLE,
            "use_gpu": self.use_gpu,
            "device": "cpu",
            "memory_info": {},
        }

        if CUPY_AVAILABLE and self.use_gpu:
            try:
                device_info["device"] = "gpu"
                device_info["gpu_name"] = cp.cuda.runtime.getDeviceProperties(0)[
                    "name"
                ].decode()

                # Get memory info
                meminfo = cp.cuda.runtime.memGetInfo()
                device_info["memory_info"] = {
                    "free_mb": meminfo[0] / (1024 * 1024),
                    "total_mb": meminfo[1] / (1024 * 1024),
                }
                logger.info(f"Using GPU: {device_info['gpu_name']}")
                logger.info(
                    f"GPU Memory: {device_info['memory_info']['free_mb']:.0f}MB free"
                )
            except Exception as e:
                logger.warning(f"GPU detection failed, falling back to CPU: {e}")
                self.use_gpu = False
                device_info["use_gpu"] = False
                device_info["device"] = "cpu"

        if not self.use_gpu:
            logger.info("Using CPU for sparse matrix operations")

        return device_info

    def initialize(self) -> bool:
        """
        Initialize the sparse LED optimizer.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Load sparse diffusion matrix
            if not self._load_sparse_matrix():
                logger.error("Sparse diffusion matrix not found")
                logger.error(
                    "Generate patterns first with: python tools/generate_synthetic_patterns.py --sparse"
                )
                return False

            logger.info(
                f"LED optimizer initialized with device: {self.device_info['device']}"
            )
            logger.info(f"Sparse matrix shape: {self._sparse_matrix_csc.shape}")
            logger.info(f"Matrix sparsity: {self._get_sparsity_percentage():.3f}%")
            return True

        except Exception as e:
            logger.error(f"LED optimizer initialization failed: {e}")
            return False

    def _load_sparse_matrix(self) -> bool:
        """
        Load sparse diffusion matrix and spatial mapping.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Try to load single sparse matrix NPZ file
            patterns_path = f"{self.diffusion_patterns_path}.npz"

            if not Path(patterns_path).exists():
                logger.warning(f"Sparse matrix file not found at {patterns_path}")
                return False

            # Load all data from single NPZ file
            data = np.load(patterns_path, allow_pickle=True)

            # Reconstruct sparse matrix from components
            self._sparse_matrix_csc = sp.csc_matrix(
                (data["matrix_data"], data["matrix_indices"], data["matrix_indptr"]),
                shape=tuple(data["matrix_shape"]),
            )

            # Convert to CSR for forward operations (A*x)
            self._sparse_matrix_csr = self._sparse_matrix_csc.tocsr()

            # Load spatial mapping and metadata
            self._led_spatial_mapping = data["led_spatial_mapping"].item()
            self._led_positions = data["led_positions"]

            # Get actual LED count from matrix (shape is pixels × led_count*3)
            self._actual_led_count = self._sparse_matrix_csc.shape[1] // 3

            # Validate matrix dimensions (single channel format)
            expected_pixels = FRAME_HEIGHT * FRAME_WIDTH
            if self._sparse_matrix_csc.shape[0] != expected_pixels:
                logger.error(
                    f"Matrix pixel dimension mismatch: "
                    f"{self._sparse_matrix_csc.shape[0]} != {expected_pixels}"
                )
                return False

            # Transfer to GPU if requested
            if self.use_gpu:
                try:
                    self._gpu_matrix_csc = cupy_csc_matrix(self._sparse_matrix_csc)
                    self._gpu_matrix_csr = cupy_csr_matrix(self._sparse_matrix_csr)
                    logger.info("Sparse matrices transferred to GPU")
                except Exception as e:
                    logger.warning(f"GPU transfer failed, using CPU: {e}")
                    self.use_gpu = False

            self._matrix_loaded = True
            logger.info(f"Loaded sparse matrix from {patterns_path}")
            logger.info(f"Matrix shape: {self._sparse_matrix_csc.shape}")
            logger.info(f"Non-zero entries: {self._sparse_matrix_csc.nnz:,}")

            return True

        except Exception as e:
            logger.error(f"Failed to load sparse matrix: {e}")
            return False

    def _get_sparsity_percentage(self) -> float:
        """Get sparsity percentage of the matrix."""
        if self._sparse_matrix_csc is None:
            return 0.0

        total_elements = (
            self._sparse_matrix_csc.shape[0] * self._sparse_matrix_csc.shape[1]
        )
        return (self._sparse_matrix_csc.nnz / total_elements) * 100

    def optimize_frame(
        self,
        target_frame: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Optimize LED values using sparse LSQR with projection.

        Args:
            target_frame: Target image (height, width, 3) in range [0, 255]
            initial_values: Initial LED values (led_count, 3), if None uses 0.5
            max_iterations: Override default max iterations

        Returns:
            OptimizationResult with LED values and metrics
        """
        start_time = time.time()

        try:
            if not self._matrix_loaded:
                raise RuntimeError("Sparse matrix not loaded")

            # Validate input
            if target_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise ValueError(
                    f"Target frame shape {target_frame.shape} != {(FRAME_HEIGHT, FRAME_WIDTH, 3)}"
                )

            # Convert target to grayscale and flatten to 1D, normalize to [0,1]
            # Use luminance formula: 0.299*R + 0.587*G + 0.114*B
            target_gray = (
                0.299 * target_frame[:, :, 0]
                + 0.587 * target_frame[:, :, 1]
                + 0.114 * target_frame[:, :, 2]
            )
            target_flat = target_gray.reshape(-1).astype(np.float32) / 255.0

            # Initialize LED values in range [0,1]
            if initial_values is not None:
                # Ensure 2D shape (led_count, 3) and normalize if needed
                if initial_values.ndim == 1:
                    led_values = np.full(
                        (self._actual_led_count, 3),
                        initial_values[0] / 255.0,
                        dtype=np.float32,
                    )
                else:
                    led_values = (
                        initial_values / 255.0
                        if initial_values.max() > 1.0
                        else initial_values
                    ).astype(np.float32)
            else:
                # Start with 50% brightness for better convergence
                led_values = np.full((self._actual_led_count, 3), 0.5, dtype=np.float32)

            # Matrix format is (384,000, led_count*3) - treat as 300 independent monochrome LEDs
            # LED vector should be (300,) = [LED0_R, LED0_G, LED0_B, LED1_R, LED1_G, LED1_B, ...]
            led_vector = led_values.flatten()  # Shape: (led_count * 3,)

            # Use the sparse matrix directly
            A_csc = self._sparse_matrix_csc
            A_csr = self._sparse_matrix_csr
            target_vector = target_flat

            # Transfer to GPU if available
            if self.use_gpu:
                gpu_A_csc = cupy_csc_matrix(A_csc)
                gpu_A_csr = cupy_csr_matrix(A_csr)
                target_gpu = cp.asarray(target_vector, dtype=cp.float32)
                x_gpu = cp.asarray(led_vector, dtype=cp.float32)

                # Run LSQR on GPU
                result_vector = self._solve_lsqr_gpu(
                    gpu_A_csc, gpu_A_csr, target_gpu, x_gpu, max_iterations
                )

                # Transfer back to CPU
                led_vector = cp.asnumpy(result_vector)
            else:
                # Run LSQR on CPU
                led_vector = self._solve_lsqr_cpu(
                    A_csc, A_csr, target_vector, led_vector, max_iterations
                )

            # Reshape back to (led_count, 3)
            led_values = led_vector.reshape((self._actual_led_count, 3))

            # Compute final error metrics
            final_reconstruction = A_csr @ led_vector
            mse = np.mean((final_reconstruction - target_vector) ** 2)
            mae = np.mean(np.abs(final_reconstruction - target_vector))
            max_error = np.max(np.abs(final_reconstruction - target_vector))

            # Scale LED values from [0,1] to [0,255] for output
            led_values_output = (led_values * 255.0).astype(np.uint8)

            optimization_time = time.time() - start_time

            # Create result
            result = OptimizationResult(
                led_values=led_values_output,
                error_metrics={
                    "mse": float(mse),
                    "mae": float(mae),
                    "max_error": float(max_error),
                    "rmse": float(np.sqrt(mse)),
                },
                optimization_time=optimization_time,
                iterations=max_iterations or self.max_iterations,
                converged=True,  # LSQR always converges
                target_frame=target_frame.copy(),
                sparsity_info={
                    "matrix_shape": self._sparse_matrix_csc.shape,
                    "nnz": self._sparse_matrix_csc.nnz,
                    "sparsity_percent": self._get_sparsity_percentage(),
                },
            )

            # Update statistics
            self._optimization_count += 1
            self._total_optimization_time += optimization_time

            logger.debug(
                f"Sparse optimization completed in {optimization_time:.3f}s, MSE: {mse:.6f}"
            )
            return result

        except Exception as e:
            optimization_time = time.time() - start_time
            logger.error(
                f"Sparse optimization failed after {optimization_time:.3f}s: {e}"
            )

            # Return error result
            return OptimizationResult(
                led_values=np.zeros((self._actual_led_count, 3), dtype=np.uint8),
                error_metrics={
                    "mse": float("inf"),
                    "mae": float("inf"),
                    "max_error": float("inf"),
                    "rmse": float("inf"),
                },
                optimization_time=optimization_time,
                iterations=0,
                converged=False,
            )

    def _expand_matrix_for_rgb(self) -> sp.csr_matrix:
        """
        Expand sparse matrix to handle RGB channels properly.

        The original matrix has shape (pixels*3, led_count) where pixels are RGB interleaved.
        We need to expand it to (pixels*3, led_count*3) to handle RGB LED values.

        Returns:
            Expanded sparse matrix
        """
        # For now, we'll duplicate the matrix for each RGB channel
        # This assumes each LED affects all RGB channels equally
        led_count = self._sparse_matrix_csc.shape[1]
        pixel_count = self._sparse_matrix_csc.shape[0]

        # Create block diagonal matrix: each LED channel affects corresponding image channel
        blocks = []
        for c in range(3):  # R, G, B
            # Extract the channel-specific part of the matrix
            channel_start = c * (pixel_count // 3)
            channel_end = (c + 1) * (pixel_count // 3)
            channel_matrix = self._sparse_matrix_csc[channel_start:channel_end, :]
            blocks.append(channel_matrix)

        # Stack blocks to form expanded matrix
        A_expanded = sp.block_diag(blocks, format="csr")

        return A_expanded

    def _solve_lsqr_cpu(
        self,
        A_csc: sp.csc_matrix,
        A_csr: sp.csr_matrix,
        b: np.ndarray,
        x0: np.ndarray,
        max_iterations: Optional[int],
    ) -> np.ndarray:
        """
        Solve LSQR with projection on CPU using projected gradient descent.

        Args:
            A_csc: Sparse matrix in CSC format (for A^T operations)
            A_csr: Sparse matrix in CSR format (for A operations)
            b: Target vector
            x0: Initial solution vector
            max_iterations: Maximum iterations

        Returns:
            Solution vector
        """
        x = x0.copy()
        max_iters = max_iterations or self.max_iterations

        for iteration in range(max_iters):
            # Compute residual: r = A*x - b
            residual = A_csr @ x - b

            # Compute gradient: g = A^T * r
            gradient = A_csc.T @ residual

            # Estimate step size using simple line search
            Ag = A_csr @ gradient
            if np.dot(Ag, Ag) > 0:
                step_size = (
                    self.step_size_scaling * np.dot(gradient, gradient) / np.dot(Ag, Ag)
                )
            else:
                step_size = 0.01  # Fallback step size

            # Gradient descent step
            x_new = x - step_size * gradient

            # Project onto feasible region [0, 1]
            x_new = np.clip(x_new, 0, 1)

            # Check convergence
            if np.linalg.norm(x_new - x) < self.convergence_threshold:
                break

            x = x_new

        return x

    def _solve_lsqr_gpu(self, A_csc, A_csr, b, x0, max_iterations: Optional[int]):
        """
        Solve LSQR with projection on GPU using CuPy.

        Args:
            A_csc: Sparse matrix in CSC format (CuPy)
            A_csr: Sparse matrix in CSR format (CuPy)
            b: Target vector (CuPy)
            x0: Initial solution vector (CuPy)
            max_iterations: Maximum iterations

        Returns:
            Solution vector (CuPy)
        """
        x = x0.copy()
        max_iters = max_iterations or self.max_iterations

        for iteration in range(max_iters):
            # Compute residual: r = A*x - b
            residual = A_csr @ x - b

            # Compute gradient: g = A^T * r
            gradient = A_csc.T @ residual

            # Estimate step size
            Ag = A_csr @ gradient
            grad_norm_sq = cp.dot(gradient, gradient)
            Ag_norm_sq = cp.dot(Ag, Ag)

            if Ag_norm_sq > 0:
                step_size = self.step_size_scaling * grad_norm_sq / Ag_norm_sq
            else:
                step_size = 0.01

            # Gradient descent step
            x_new = x - step_size * gradient

            # Project onto feasible region [0, 1]
            x_new = cp.clip(x_new, 0, 1)

            # Check convergence
            if cp.linalg.norm(x_new - x) < self.convergence_threshold:
                break

            x = x_new

        return x

    def get_optimizer_stats(self) -> Dict[str, Any]:
        """
        Get optimizer statistics.

        Returns:
            Dictionary with optimizer statistics
        """
        avg_time = self._total_optimization_time / max(1, self._optimization_count)

        stats = {
            "device_info": self.device_info,
            "matrix_loaded": self._matrix_loaded,
            "optimization_count": self._optimization_count,
            "total_optimization_time": self._total_optimization_time,
            "average_optimization_time": avg_time,
            "estimated_fps": 1.0 / avg_time if avg_time > 0 else 0.0,
            "parameters": {
                "max_iterations": self.max_iterations,
                "convergence_threshold": self.convergence_threshold,
                "step_size_scaling": self.step_size_scaling,
            },
            "led_count": self._actual_led_count,
            "frame_dimensions": (FRAME_WIDTH, FRAME_HEIGHT),
        }

        if self._matrix_loaded:
            stats.update(
                {
                    "matrix_shape": self._sparse_matrix_csc.shape,
                    "matrix_nnz": self._sparse_matrix_csc.nnz,
                    "sparsity_percent": self._get_sparsity_percentage(),
                    "memory_usage_mb": self._sparse_matrix_csc.data.nbytes
                    / (1024 * 1024),
                }
            )

        return stats

    def set_optimization_parameters(
        self,
        max_iterations: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        step_size_scaling: Optional[float] = None,
    ) -> None:
        """
        Update optimization parameters.

        Args:
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold
            step_size_scaling: Step size scaling factor
        """
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if convergence_threshold is not None:
            self.convergence_threshold = convergence_threshold
        if step_size_scaling is not None:
            self.step_size_scaling = step_size_scaling

        logger.info(
            f"Updated optimization parameters: max_iter={self.max_iterations}, "
            f"threshold={self.convergence_threshold}, step_scaling={self.step_size_scaling}"
        )
