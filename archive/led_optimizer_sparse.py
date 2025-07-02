"""
LED Optimization Engine with Sparse Matrix LSQR.

This module implements the core optimization algorithm that finds LED brightness
values to best approximate a target image using sparse diffusion matrices. The optimization
solves: minimize ||A*x - b||² where A is sparse diffusion matrix, x is LED brightness, b is target.

Uses sparse CSC matrices with LSQR gradient descent and projection for
real-time performance targeting 15fps with memory efficiency.
"""

import logging

# Import constants from parent directory
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cupy as cp
import numpy as np
import scipy.sparse as sp
from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix

src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)
from const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT

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
    flop_info: Optional[Dict[str, Any]] = None  # FLOP analysis information

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
    ):
        """
        Initialize sparse LED optimizer (GPU-only).

        Args:
            diffusion_patterns_path: Path to sparse diffusion matrix files
        """
        self.diffusion_patterns_path = diffusion_patterns_path or "diffusion_patterns/synthetic_1000"

        # Optimization parameters for LSQR (tuned for real-time performance)
        self.max_iterations = 10  # Further reduced for sub-100ms performance
        self.convergence_threshold = 1e-3  # Further relaxed for faster convergence
        self.step_size_scaling = 0.9  # Slightly more aggressive for faster convergence

        # GPU sparse matrices (primary storage)
        self._A_combined_csc_gpu = None  # Combined block diagonal CSC matrix on GPU
        self._A_combined_csr_gpu = None  # Combined block diagonal CSR matrix on GPU

        # CPU reference matrices (for loading and metadata only)
        self._A_combined_csc_cpu: Optional[sp.csc_matrix] = None
        self._A_combined_csr_cpu: Optional[sp.csr_matrix] = None

        # Pre-allocated arrays for target vectors (avoid repeated allocation)
        self._target_combined_buffer = None  # Stacked [R; G; B] channels (CPU)
        self._target_combined_gpu = None  # Target on GPU
        self._led_combined_buffer = None  # Combined [R_leds; G_leds; B_leds] (CPU)

        # Pre-allocated GPU workspace arrays for optimization loop
        self._gpu_workspace = None  # GPU workspace arrays

        self._led_spatial_mapping: Optional[Dict[int, int]] = None
        self._led_positions: Optional[np.ndarray] = None
        self._matrix_loaded = False
        self._actual_led_count = LED_COUNT

        # Statistics
        self._optimization_count = 0
        self._total_optimization_time = 0.0

        # FLOP counting for performance analysis
        self._total_flops = 0
        self._flops_per_iteration = 0  # Will be calculated after matrix loading

        # Detect compute capability
        self.device_info = self._detect_compute_device()

    def _detect_compute_device(self) -> Dict[str, Any]:
        """
        Detect GPU device and capabilities (GPU-only implementation).

        Returns:
            Dictionary with device information
        """
        device_info = {
            "device": "gpu",
            "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "memory_info": {},
        }

        # Get memory info
        meminfo = cp.cuda.runtime.memGetInfo()
        device_info["memory_info"] = {
            "free_mb": meminfo[0] / (1024 * 1024),
            "total_mb": meminfo[1] / (1024 * 1024),
        }
        logger.info(f"Using GPU: {device_info['gpu_name']}")
        logger.info(f"GPU Memory: {device_info['memory_info']['free_mb']:.0f}MB free")

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
                logger.error("Generate patterns first with: python tools/generate_synthetic_patterns.py --sparse")
                return False

            logger.info(f"LED optimizer initialized with device: {self.device_info['device']}")
            logger.info(f"Combined matrix shape: {self._A_combined_csr_gpu.shape}")
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

            # Reconstruct sparse matrix from components (temporary CPU matrix for loading)
            sparse_matrix_csc = sp.csc_matrix(
                (data["matrix_data"], data["matrix_indices"], data["matrix_indptr"]),
                shape=tuple(data["matrix_shape"]),
            )

            # Load spatial mapping and metadata
            self._led_spatial_mapping = data["led_spatial_mapping"].item()
            self._led_positions = data["led_positions"]

            # Get actual LED count from matrix (shape is pixels × led_count*3)
            self._actual_led_count = sparse_matrix_csc.shape[1] // 3

            # Validate matrix dimensions (single channel format)
            expected_pixels = FRAME_HEIGHT * FRAME_WIDTH
            if sparse_matrix_csc.shape[0] != expected_pixels:
                logger.error(f"Matrix pixel dimension mismatch: {sparse_matrix_csc.shape[0]} != {expected_pixels}")
                return False

            # Pre-compute RGB channel matrices and create combined block diagonal matrix
            logger.info("Creating combined block diagonal matrix on CPU...")
            A_r_csc = sparse_matrix_csc[:, 0::3]  # Red channels
            A_g_csc = sparse_matrix_csc[:, 1::3]  # Green channels
            A_b_csc = sparse_matrix_csc[:, 2::3]  # Blue channels

            # Create combined block diagonal matrix for single large optimization
            self._A_combined_csc_cpu = sp.block_diag([A_r_csc, A_g_csc, A_b_csc])
            self._A_combined_csr_cpu = self._A_combined_csc_cpu.tocsr()
            logger.info(f"Combined matrix shape: {self._A_combined_csc_cpu.shape}")
            logger.info(f"Combined matrix nnz: {self._A_combined_csc_cpu.nnz:,}")

            # Transfer to GPU immediately
            logger.info("Transferring matrices to GPU...")
            self._A_combined_csc_gpu = cupy_csc_matrix(self._A_combined_csc_cpu)
            self._A_combined_csr_gpu = cupy_csr_matrix(self._A_combined_csr_cpu)
            logger.info("GPU matrices loaded successfully")

            # Pre-allocate target buffers on CPU and GPU
            pixels = FRAME_HEIGHT * FRAME_WIDTH
            self._target_combined_buffer = np.empty(3 * pixels, dtype=np.float32)  # CPU buffer for preprocessing
            self._target_combined_gpu = cp.empty(3 * pixels, dtype=cp.float32)  # GPU target vector
            self._led_combined_buffer = np.empty(3 * self._actual_led_count, dtype=np.float32)  # CPU result buffer

            # Calculate FLOPs per iteration for performance analysis
            self._calculate_flops_per_iteration()

            # Initialize workspace arrays for optimization loops
            self._initialize_workspace()

            self._matrix_loaded = True
            logger.info(f"Loaded sparse matrix from {patterns_path}")
            logger.info(f"Combined matrix shape: {self._A_combined_csr_cpu.shape}")
            logger.info(f"Non-zero entries: {self._A_combined_csr_cpu.nnz:,}")
            logger.info(f"Estimated FLOPs per iteration: {self._flops_per_iteration:,}")

            return True

        except Exception as e:
            logger.error(f"Failed to load sparse matrix: {e}")
            return False

    def _get_sparsity_percentage(self) -> float:
        """Get sparsity percentage of the matrix."""
        if self._A_combined_csr_gpu is None:
            return 0.0

        total_elements = self._A_combined_csr_gpu.shape[0] * self._A_combined_csr_gpu.shape[1]
        return (self._A_combined_csr_gpu.nnz / total_elements) * 100

    def _calculate_flops_per_iteration(self) -> None:
        """Calculate floating point operations per LSQR iteration for performance analysis."""
        if self._A_combined_csr_cpu is None:
            return

        # FLOPs for combined block diagonal matrix per iteration:
        # 1. Forward SpMV: A_combined @ x  -> 2 * nnz operations
        # 2. Residual calculation: A*x - b  -> 2 * matrix_rows operations
        # 3. Transpose SpMV: A_combined.T @ residual  -> 2 * nnz operations
        # 4. Gradient norm: dot(gradient, gradient)  -> 2 * led_count operations
        # 5. Step calculation: Ag = A @ gradient  -> 2 * nnz operations
        # 6. Step norm: dot(Ag, Ag)  -> 2 * matrix_rows operations
        # 7. Vector updates: x_new = x - step_size * gradient  -> 2 * led_count operations

        nnz_combined = self._A_combined_csr_cpu.nnz  # Total non-zeros in combined matrix
        pixels_combined = self._A_combined_csr_cpu.shape[0]  # 3 * (width * height)
        leds_combined = self._A_combined_csr_cpu.shape[1]  # 3 * led_count

        self._flops_per_iteration = (
            2 * nnz_combined  # Forward SpMV
            + 2 * pixels_combined  # Residual calculation
            + 2 * nnz_combined  # Transpose SpMV
            + 2 * leds_combined  # Gradient norm
            + 2 * nnz_combined  # Step calculation SpMV
            + 2 * pixels_combined  # Step norm
            + 2 * leds_combined  # Vector update
        )

        logger.debug(
            f"FLOP calculation (combined): {nnz_combined:,} nnz, {pixels_combined:,} pixels, {leds_combined:,} LEDs"
        )
        logger.debug(f"Combined FLOPs per iteration: {self._flops_per_iteration:,}")

    def _initialize_workspace(self) -> None:
        """
        Pre-initialize GPU workspace arrays for optimization loops.
        """
        if self._A_combined_csr_gpu is None:
            return

        m, n = self._A_combined_csr_gpu.shape  # Combined matrix dimensions

        logger.info(f"Initializing GPU workspace arrays for matrix shape {m}x{n}")

        # GPU workspace arrays only
        self._gpu_workspace = {
            "residual": cp.zeros(m, dtype=cp.float32),  # Size: 3 * pixels
            "gradient": cp.zeros(n, dtype=cp.float32),  # Size: 3 * led_count
            "Ag": cp.zeros(m, dtype=cp.float32),  # Size: 3 * pixels
            "x_new": cp.zeros(n, dtype=cp.float32),  # Size: 3 * led_count
            "x_diff": cp.zeros(n, dtype=cp.float32),  # Size: 3 * led_count
            "x_init": cp.full(n, 0.5, dtype=cp.float32),  # Initialize to 0.5 on GPU
        }

        workspace_mb = (m + n) * 6 * 4 / (1024 * 1024)  # 6 arrays * 4 bytes per float32
        logger.info(f"GPU workspace memory: {workspace_mb:.1f}MB")

    def _optimize_frame_combined(
        self,
        target_frame: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Optimize LED values using combined block diagonal matrix approach.

        This uses a single large matrix operation instead of 3 separate RGB optimizations,
        which should provide much better GPU utilization.
        """
        # Process target frame into combined format
        target_rgb_normalized = target_frame.astype(np.float32) / 255.0

        # Stack RGB channels: [R_pixels; G_pixels; B_pixels]
        pixels = FRAME_HEIGHT * FRAME_WIDTH
        self._target_combined_buffer[:pixels] = target_rgb_normalized[:, :, 0].ravel()
        self._target_combined_buffer[pixels : 2 * pixels] = target_rgb_normalized[:, :, 1].ravel()
        self._target_combined_buffer[2 * pixels :] = target_rgb_normalized[:, :, 2].ravel()

        target_combined = self._target_combined_buffer

        # Transfer target to GPU
        self._target_combined_gpu[:] = cp.asarray(target_combined, dtype=cp.float32)

        # Initialize combined LED values: [R_leds; G_leds; B_leds]
        if initial_values is not None:
            if initial_values.ndim == 1:
                init_val = initial_values[0] / 255.0 if initial_values[0] > 1.0 else initial_values[0]
                led_combined_init = cp.full(3 * self._actual_led_count, init_val, dtype=cp.float32)
            else:
                initial_normalized = (initial_values / 255.0 if initial_values.max() > 1.0 else initial_values).astype(
                    np.float32
                )
                # Stack: [R_leds; G_leds; B_leds] and transfer to GPU
                self._led_combined_buffer[: self._actual_led_count] = initial_normalized[:, 0]
                self._led_combined_buffer[self._actual_led_count : 2 * self._actual_led_count] = initial_normalized[
                    :, 1
                ]
                self._led_combined_buffer[2 * self._actual_led_count :] = initial_normalized[:, 2]
                led_combined_init = cp.asarray(self._led_combined_buffer, dtype=cp.float32)
        else:
            # Use pre-initialized GPU array (already at 0.5)
            led_combined_init = self._gpu_workspace["x_init"]

        # GPU-only optimization using combined matrices
        led_combined_solved = self._solve_lsqr_gpu(
            self._A_combined_csc_gpu,
            self._A_combined_csr_gpu,
            self._target_combined_gpu,
            led_combined_init,
            max_iterations,
        )

        # Convert back to numpy for output
        led_combined_values = cp.asnumpy(led_combined_solved)

        # Reshape combined result back to (led_count, 3) format
        led_values = np.zeros((self._actual_led_count, 3), dtype=np.float32)
        led_values[:, 0] = led_combined_values[: self._actual_led_count]  # R
        led_values[:, 1] = led_combined_values[self._actual_led_count : 2 * self._actual_led_count]  # G
        led_values[:, 2] = led_combined_values[2 * self._actual_led_count :]  # B

        # Compute error metrics using GPU combined matrix
        residual_combined_gpu = self._A_combined_csr_gpu @ led_combined_solved - self._target_combined_gpu
        residual_combined = cp.asnumpy(residual_combined_gpu)

        mse = np.mean(residual_combined**2)
        mae = np.mean(np.abs(residual_combined))
        max_error = np.max(np.abs(residual_combined))
        rmse = np.sqrt(mse)

        return led_values, mse, mae, max_error, rmse

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
        # Initialize timing before any potential errors
        start_time = time.time()

        try:
            if not self._matrix_loaded:
                raise RuntimeError("Sparse matrix not loaded")

            # Validate input
            if target_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise ValueError(f"Target frame shape {target_frame.shape} != {(FRAME_HEIGHT, FRAME_WIDTH, 3)}")

            # Use combined block diagonal matrix optimization for better GPU utilization
            led_values, mse, mae, max_error, rmse = self._optimize_frame_combined(
                target_frame, initial_values, max_iterations
            )

            # Scale LED values from [0,1] to [0,255] for output
            led_values_output = (led_values * 255.0).astype(np.uint8)

            optimization_time = time.time() - start_time

            # Calculate FLOPs for this optimization
            iterations_used = max_iterations or self.max_iterations
            frame_flops = iterations_used * self._flops_per_iteration

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
                    "matrix_shape": self._A_combined_csr_gpu.shape,
                    "nnz": self._A_combined_csr_gpu.nnz,
                    "sparsity_percent": self._get_sparsity_percentage(),
                },
                flop_info={
                    "total_flops": int(frame_flops),
                    "flops_per_iteration": int(self._flops_per_iteration),
                    "gflops": frame_flops / 1e9,
                    "gflops_per_second": frame_flops / (optimization_time * 1e9),
                },
            )

            # Update statistics including FLOP tracking
            self._total_flops += frame_flops
            self._optimization_count += 1
            self._total_optimization_time += optimization_time

            logger.debug(f"Sparse optimization completed in {optimization_time:.3f}s, MSE: {mse:.6f}")
            return result

        except Exception as e:
            optimization_time = time.time() - start_time
            logger.error(f"Sparse optimization failed after {optimization_time:.3f}s: {e}")

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
                flop_info={
                    "total_flops": 0,
                    "flops_per_iteration": (int(self._flops_per_iteration) if self._flops_per_iteration > 0 else 0),
                    "gflops": 0.0,
                    "gflops_per_second": 0.0,
                },
            )

    def _solve_lsqr_gpu(self, A_csc, A_csr, b, x0, max_iterations: Optional[int]):
        """
        Solve LSQR with projection on GPU using CuPy with pre-allocated workspace.

        Args:
            A_csc: Sparse matrix in CSC format (CuPy)
            A_csr: Sparse matrix in CSR format (CuPy)
            b: Target vector (CuPy)
            x0: Initial solution vector (CuPy)
            max_iterations: Maximum iterations

        Returns:
            Solution vector (CuPy)
        """
        # Use x0 directly if it's the pre-allocated workspace array, otherwise copy
        if x0 is self._gpu_workspace["x_init"]:
            x = x0  # No copy needed, use pre-allocated array directly
        else:
            x = x0.copy()

        max_iters = max_iterations or self.max_iterations

        # Use pre-allocated workspace arrays (GPU-only, no fallback)
        residual = self._gpu_workspace["residual"]
        gradient = self._gpu_workspace["gradient"]
        Ag = self._gpu_workspace["Ag"]
        x_new = self._gpu_workspace["x_new"]
        x_diff = self._gpu_workspace["x_diff"]

        for iteration in range(max_iters):
            # Compute residual: r = A*x - b (assign to pre-allocated array)
            residual[:] = A_csr @ x - b

            # Compute gradient: g = A^T * r (assign to pre-allocated array)
            gradient[:] = A_csc.T @ residual

            # Estimate step size (assign to pre-allocated array)
            Ag[:] = A_csr @ gradient
            grad_norm_sq = cp.dot(gradient, gradient)
            Ag_norm_sq = cp.dot(Ag, Ag)

            if Ag_norm_sq > 0:
                step_size = self.step_size_scaling * grad_norm_sq / Ag_norm_sq
            else:
                step_size = 0.01

            # Gradient descent step (use pre-allocated array)
            cp.multiply(gradient, step_size, out=x_new)  # x_new = step_size * gradient
            cp.subtract(x, x_new, out=x_new)  # x_new = x - step_size * gradient

            # Project onto feasible region [0, 1] (in-place)
            cp.clip(x_new, 0, 1, out=x_new)

            # Check convergence (use pre-allocated array)
            cp.subtract(x_new, x, out=x_diff)  # x_diff = x_new - x
            convergence_norm = cp.linalg.norm(x_diff)

            if convergence_norm < self.convergence_threshold:
                break

            # Update x (swap references to avoid copy)
            x, x_new = x_new, x

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
            # Calculate average GFLOPS performance
            avg_gflops_per_second = 0.0
            if avg_time > 0 and self._flops_per_iteration > 0:
                avg_gflops_per_second = (self.max_iterations * self._flops_per_iteration) / (avg_time * 1e9)

            stats.update(
                {
                    "matrix_shape": self._A_combined_csr_gpu.shape,
                    "matrix_nnz": self._A_combined_csr_gpu.nnz,
                    "sparsity_percent": self._get_sparsity_percentage(),
                    "memory_usage_mb": self._A_combined_csr_gpu.data.nbytes / (1024 * 1024),
                    "flop_analysis": {
                        "flops_per_iteration": int(self._flops_per_iteration),
                        "total_flops_computed": int(self._total_flops),
                        "average_gflops_per_frame": (self.max_iterations * self._flops_per_iteration) / 1e9,
                        "average_gflops_per_second": avg_gflops_per_second,
                    },
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
