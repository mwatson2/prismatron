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

        # Sparse matrix state (pre-computed for performance)
        self._sparse_matrix_csc: Optional[
            sp.csc_matrix
        ] = None  # CSC for A^T operations (optimized for transpose)
        self._sparse_matrix_csr: Optional[
            sp.csr_matrix
        ] = None  # CSR for A operations (optimized for forward)
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

            # Note: With unified memory, no explicit GPU transfers needed
            # CuPy will work directly with the sparse matrices as needed

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

            # Process RGB channels separately - matrix has columns [LED0_R, LED0_G, LED0_B, LED1_R, LED1_G, LED1_B, ...]
            # For proper RGB optimization, we solve 3 separate problems:
            # A_red @ led_red_values = target_red_channel
            # A_green @ led_green_values = target_green_channel
            # A_blue @ led_blue_values = target_blue_channel
            target_rgb_normalized = target_frame.astype(np.float32) / 255.0

            # Extract RGB channels as flat arrays (384000 pixels each)
            target_r = target_rgb_normalized[:, :, 0].reshape(-1)  # Red channel
            target_g = target_rgb_normalized[:, :, 1].reshape(-1)  # Green channel
            target_b = target_rgb_normalized[:, :, 2].reshape(-1)  # Blue channel

            # Extract RGB matrix columns (every 3rd column starting from 0, 1, 2)
            A_csc = self._sparse_matrix_csc
            A_csr = self._sparse_matrix_csr

            # Split matrix into R, G, B components
            A_r_csc = A_csc[:, 0::3]  # Columns 0, 3, 6, ... (Red LEDs)
            A_g_csc = A_csc[:, 1::3]  # Columns 1, 4, 7, ... (Green LEDs)
            A_b_csc = A_csc[:, 2::3]  # Columns 2, 5, 8, ... (Blue LEDs)

            A_r_csr = A_csr[:, 0::3]
            A_g_csr = A_csr[:, 1::3]
            A_b_csr = A_csr[:, 2::3]

            # Initialize LED values for each channel
            if initial_values is not None:
                if initial_values.ndim == 1:
                    led_r_init = np.full(
                        self._actual_led_count,
                        initial_values[0] / 255.0,
                        dtype=np.float32,
                    )
                    led_g_init = np.full(
                        self._actual_led_count,
                        initial_values[0] / 255.0,
                        dtype=np.float32,
                    )
                    led_b_init = np.full(
                        self._actual_led_count,
                        initial_values[0] / 255.0,
                        dtype=np.float32,
                    )
                else:
                    initial_normalized = (
                        initial_values / 255.0
                        if initial_values.max() > 1.0
                        else initial_values
                    ).astype(np.float32)
                    led_r_init = initial_normalized[:, 0]
                    led_g_init = initial_normalized[:, 1]
                    led_b_init = initial_normalized[:, 2]
            else:
                # Start with 50% brightness for better convergence
                led_r_init = np.full(self._actual_led_count, 0.5, dtype=np.float32)
                led_g_init = np.full(self._actual_led_count, 0.5, dtype=np.float32)
                led_b_init = np.full(self._actual_led_count, 0.5, dtype=np.float32)

            # Solve 3 separate optimization problems
            if self.use_gpu and CUPY_AVAILABLE:
                try:
                    # GPU solve for all channels
                    target_r_gpu = cp.asarray(target_r, dtype=cp.float32)
                    target_g_gpu = cp.asarray(target_g, dtype=cp.float32)
                    target_b_gpu = cp.asarray(target_b, dtype=cp.float32)

                    led_r_gpu = cp.asarray(led_r_init, dtype=cp.float32)
                    led_g_gpu = cp.asarray(led_g_init, dtype=cp.float32)
                    led_b_gpu = cp.asarray(led_b_init, dtype=cp.float32)

                    # Convert matrices to CuPy
                    A_r_csc_gpu = cupy_csc_matrix(A_r_csc)
                    A_g_csc_gpu = cupy_csc_matrix(A_g_csc)
                    A_b_csc_gpu = cupy_csc_matrix(A_b_csc)

                    A_r_csr_gpu = cupy_csr_matrix(A_r_csr)
                    A_g_csr_gpu = cupy_csr_matrix(A_g_csr)
                    A_b_csr_gpu = cupy_csr_matrix(A_b_csr)

                    # Solve each channel
                    led_r_solved = self._solve_lsqr_gpu(
                        A_r_csc_gpu,
                        A_r_csr_gpu,
                        target_r_gpu,
                        led_r_gpu,
                        max_iterations,
                    )
                    led_g_solved = self._solve_lsqr_gpu(
                        A_g_csc_gpu,
                        A_g_csr_gpu,
                        target_g_gpu,
                        led_g_gpu,
                        max_iterations,
                    )
                    led_b_solved = self._solve_lsqr_gpu(
                        A_b_csc_gpu,
                        A_b_csr_gpu,
                        target_b_gpu,
                        led_b_gpu,
                        max_iterations,
                    )

                    # Convert back to numpy
                    led_r_values = cp.asnumpy(led_r_solved)
                    led_g_values = cp.asnumpy(led_g_solved)
                    led_b_values = cp.asnumpy(led_b_solved)

                except Exception as e:
                    logger.warning(f"GPU optimization failed, falling back to CPU: {e}")
                    # Fall back to CPU
                    led_r_values = self._solve_lsqr_cpu(
                        A_r_csc, A_r_csr, target_r, led_r_init, max_iterations
                    )
                    led_g_values = self._solve_lsqr_cpu(
                        A_g_csc, A_g_csr, target_g, led_g_init, max_iterations
                    )
                    led_b_values = self._solve_lsqr_cpu(
                        A_b_csc, A_b_csr, target_b, led_b_init, max_iterations
                    )
            else:
                # CPU solve for all channels
                led_r_values = self._solve_lsqr_cpu(
                    A_r_csc, A_r_csr, target_r, led_r_init, max_iterations
                )
                led_g_values = self._solve_lsqr_cpu(
                    A_g_csc, A_g_csr, target_g, led_g_init, max_iterations
                )
                led_b_values = self._solve_lsqr_cpu(
                    A_b_csc, A_b_csr, target_b, led_b_init, max_iterations
                )

            # Combine RGB channels into final LED values
            led_values = np.stack(
                [led_r_values, led_g_values, led_b_values], axis=1
            )  # Shape: (led_count, 3)

            # Compute error metrics by reconstructing each channel and combining
            r_residual = A_r_csr @ led_r_values - target_r
            g_residual = A_g_csr @ led_g_values - target_g
            b_residual = A_b_csr @ led_b_values - target_b

            # Combine residuals for overall error metrics
            total_residual = np.concatenate([r_residual, g_residual, b_residual])

            mse = np.mean(total_residual**2)
            mae = np.mean(np.abs(total_residual))
            max_error = np.max(np.abs(total_residual))
            rmse = np.sqrt(mse)

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
                    "rmse": float(rmse),
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
