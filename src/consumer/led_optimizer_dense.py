"""
LED Optimization Engine with Dense Tensor Precomputation.

This module implements a dense tensor optimization approach using precomputed A^T*A
matrices. Instead of sparse matrix operations in the optimization loop, we precompute
dense (leds x leds) matrices for each RGB channel and use dense tensor operations
for much better GPU utilization.

The key insight: precompute A^T*A (dense) and A^T*b (per frame) to convert the sparse
optimization into dense tensor operations with clean channel separation.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cupy as cp
import numpy as np
import scipy.sparse as sp

from ..const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from ..utils.performance_timing import PerformanceTiming

logger = logging.getLogger(__name__)


@dataclass
class DenseOptimizationResult:
    """Results from dense tensor LED optimization process."""

    led_values: np.ndarray  # RGB values for each LED (led_count, 3) - range [0,255]
    error_metrics: Dict[str, float]  # Error metrics (mse, mae, etc.)
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged
    target_frame: Optional[np.ndarray] = None  # Original target frame (for debugging)
    precomputation_info: Optional[Dict[str, Any]] = None  # Dense matrix information

    def get_led_count(self) -> int:
        """Get number of LEDs in result."""
        return self.led_values.shape[0]

    def get_total_error(self) -> float:
        """Get total optimization error."""
        return self.error_metrics.get("mse", float("inf"))


class DenseLEDOptimizer:
    """
    Dense tensor LED optimization engine using precomputed A^T*A matrices.

    This approach precomputes the dense (led_count x led_count) matrices A^T*A
    for each RGB channel, then uses dense tensor operations in the optimization
    loop for much better GPU utilization compared to sparse operations.

    Key steps:
    1. Precompute: ATA_rgb = A_rgb^T @ A_rgb (dense matrices)
    2. Per frame: ATb = A^T @ b (dense vector)
    3. Loop: gradient = ATA @ x - ATb (dense operations)
    4. Step size: (g^T @ g) / (g^T @ ATA @ g) (dense operations)
    """

    def __init__(
        self,
        diffusion_patterns_path: Optional[str] = None,
    ):
        """
        Initialize dense tensor LED optimizer.

        Args:
            diffusion_patterns_path: Path to sparse diffusion matrix files
        """
        self.diffusion_patterns_path = (
            diffusion_patterns_path or "diffusion_patterns/synthetic_1000"
        )

        # Optimization parameters for gradient descent
        self.max_iterations = 10
        self.convergence_threshold = 1e-3
        self.step_size_scaling = 0.9

        # Precomputed dense matrices (key insight: A^T*A is dense, small)
        self._ATA_gpu = None  # Shape: (led_count, led_count, 3) - dense on GPU
        self._ATA_cpu = None  # Shape: (led_count, led_count, 3) - dense on CPU

        # Sparse matrices for A^T*b calculation (kept sparse for efficiency)
        self._A_r_csc_gpu = None  # Red channel sparse matrix (pixels, leds)
        self._A_g_csc_gpu = None  # Green channel sparse matrix
        self._A_b_csc_gpu = None  # Blue channel sparse matrix
        self._A_combined_csc_gpu = (
            None  # Combined block diagonal matrix for faster A^T*b
        )

        # Pre-allocated arrays for optimization
        self._target_rgb_buffer = None  # (pixels, 3) CPU buffer
        self._ATb_gpu = None  # (led_count, 3) dense vector on GPU
        self._led_values_gpu = None  # (led_count, 3) LED values on GPU

        # GPU workspace for optimization loop (all dense)
        self._gpu_workspace = None

        self._led_spatial_mapping: Optional[Dict[int, int]] = None
        self._led_positions: Optional[np.ndarray] = None
        self._matrix_loaded = False
        self._actual_led_count = LED_COUNT

        # Statistics
        self._optimization_count = 0
        self._total_optimization_time = 0.0

        # FLOP counting for performance analysis
        self._total_flops = 0
        self._flops_per_iteration = 0

        # Performance timing framework
        self._timing = PerformanceTiming("DenseLEDOptimizer", enable_gpu_timing=True)

        # Detect compute capability
        self.device_info = self._detect_compute_device()

    def _detect_compute_device(self) -> Dict[str, Any]:
        """
        Detect GPU device and capabilities.

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
        Initialize the dense tensor LED optimizer with A^T*A precomputation.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Load sparse matrices and precompute dense A^T*A
            if not self._load_and_precompute_dense_matrices():
                logger.error("Failed to load and precompute dense matrices")
                return False

            # Initialize workspace arrays
            self._initialize_workspace()

            logger.info("Dense LED optimizer initialized successfully")
            logger.info(f"LED count: {self._actual_led_count}")
            logger.info(f"Dense ATA tensor shape: {self._ATA_gpu.shape}")
            logger.info(f"Device: {self.device_info['device']}")
            return True

        except Exception as e:
            logger.error(f"Dense LED optimizer initialization failed: {e}")
            return False

    def _load_and_precompute_dense_matrices(self) -> bool:
        """
        Load sparse diffusion matrices and precompute dense A^T*A tensors.

        This is the key innovation: convert A^T*A from sparse to dense for
        much better GPU utilization in the optimization loop.

        Returns:
            True if loaded and precomputed successfully, False otherwise
        """
        try:
            patterns_path = f"{self.diffusion_patterns_path}.npz"
            if not Path(patterns_path).exists():
                logger.warning(f"Patterns file not found at {patterns_path}")
                return False

            logger.info(f"Loading sparse matrix from {patterns_path}")
            data = np.load(patterns_path, allow_pickle=True)

            # Reconstruct sparse matrix from components (temporary for loading)
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
                logger.error(
                    f"Matrix pixel dimension mismatch: "
                    f"{sparse_matrix_csc.shape[0]} != {expected_pixels}"
                )
                return False

            logger.info("Extracting RGB channel matrices...")
            A_r_csc = sparse_matrix_csc[:, 0::3]  # Red channels
            A_g_csc = sparse_matrix_csc[:, 1::3]  # Green channels
            A_b_csc = sparse_matrix_csc[:, 2::3]  # Blue channels

            logger.info(f"RGB matrix shapes: {A_r_csc.shape}")

            # KEY STEP 1: Precompute A^T*A for each channel (dense matrices)
            logger.info("Precomputing dense A^T*A matrices...")

            # Compute A^T*A on CPU first (sparse @ sparse -> dense)
            ATA_r = (A_r_csc.T @ A_r_csc).toarray()  # (leds, leds) - should be dense
            ATA_g = (A_g_csc.T @ A_g_csc).toarray()
            ATA_b = (A_b_csc.T @ A_b_csc).toarray()

            logger.info(f"Dense ATA matrix shapes: {ATA_r.shape}")
            logger.info(
                f"ATA_r density: {np.count_nonzero(ATA_r) / ATA_r.size * 100:.1f}%"
            )

            # Stack into 3D tensor: (led_count, led_count, 3)
            self._ATA_cpu = np.stack([ATA_r, ATA_g, ATA_b], axis=2).astype(np.float32)

            # Transfer to GPU
            logger.info("Transferring dense ATA tensor to GPU...")
            self._ATA_gpu = cp.asarray(self._ATA_cpu)

            # Create combined block diagonal matrix for faster A^T*b calculation
            logger.info("Creating combined block diagonal matrix for A^T*b...")
            from scipy.sparse import block_diag

            A_combined_csc = block_diag([A_r_csc, A_g_csc, A_b_csc], format="csc")
            logger.info(f"Combined matrix shape: {A_combined_csc.shape}")

            # Transfer matrices to GPU
            logger.info("Transferring matrices to GPU...")
            from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix

            self._A_r_csc_gpu = cupy_csc_matrix(A_r_csc)
            self._A_g_csc_gpu = cupy_csc_matrix(A_g_csc)
            self._A_b_csc_gpu = cupy_csc_matrix(A_b_csc)
            self._A_combined_csc_gpu = cupy_csc_matrix(A_combined_csc)

            # Calculate memory usage
            ata_memory_mb = self._ATA_gpu.nbytes / (1024 * 1024)
            logger.info(f"Dense ATA tensor memory: {ata_memory_mb:.1f}MB")

            # Calculate FLOPs per iteration
            self._calculate_flops_per_iteration()

            self._matrix_loaded = True
            logger.info("Dense matrix precomputation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load and precompute dense matrices: {e}")
            return False

    def _calculate_flops_per_iteration(self) -> None:
        """Calculate floating point operations per iteration for dense operations."""
        if self._ATA_gpu is None:
            return

        led_count = self._actual_led_count

        # Calculate FLOPs for dense optimization loop (per iteration):
        # 1. ATA @ x: 3 channels × (led_count × led_count) matrix-vector = led_count^2 * 3 * 2
        # 2. gradient = ATA @ x - ATb: led_count * 3 * 2 ops
        # 3. g^T @ g: led_count * 3 * 2 ops
        # 4. g^T @ ATA @ g: 3 channels × (led_count × led_count) = led_count^2 * 3 * 2
        # 5. Vector updates: led_count * 3 * 2 ops

        dense_flops_per_iteration = (
            led_count**2 * 3 * 2  # ATA @ x (dense matrix-vector per channel)
            + led_count * 3 * 2  # Gradient computation
            + led_count * 3 * 2  # g^T @ g
            + led_count**2 * 3 * 2  # g^T @ ATA @ g (dense matrix-vector per channel)
            + led_count * 3 * 2  # Vector update
        )

        # Calculate FLOPs for A^T*b computation (once per frame, not per iteration):
        # This now uses the combined block diagonal matrix: A_combined^T @ target_combined
        total_nnz = 0
        if (
            hasattr(self, "_A_combined_csc_gpu")
            and self._A_combined_csc_gpu is not None
        ):
            # Get non-zero count from combined block diagonal matrix
            total_nnz = self._A_combined_csc_gpu.nnz

        atb_flops_per_frame = (
            total_nnz * 2
        )  # sparse matrix-vector multiply: 2 ops per non-zero

        # Store both values for accurate reporting
        self._dense_flops_per_iteration = dense_flops_per_iteration
        self._atb_flops_per_frame = atb_flops_per_frame

        # For compatibility, store total FLOPs per iteration (including amortized A^T*b)
        # Note: A^T*b is computed once per frame, so amortize over iterations
        self._flops_per_iteration = dense_flops_per_iteration + (
            atb_flops_per_frame / self.max_iterations
        )

        logger.debug(
            f"Dense optimization FLOPs per iteration: {dense_flops_per_iteration:,}"
        )
        logger.debug(f"A^T*b FLOPs per frame: {atb_flops_per_frame:,}")
        logger.debug(
            f"Total FLOPs per iteration (amortized): {self._flops_per_iteration:,}"
        )
        logger.debug(f"Sparse matrix non-zeros (total): {total_nnz:,}")

    def _initialize_workspace(self) -> None:
        """Initialize GPU workspace arrays for dense optimization."""
        if not self._matrix_loaded:
            return

        pixels = FRAME_HEIGHT * FRAME_WIDTH
        led_count = self._actual_led_count

        logger.info(f"Initializing dense workspace arrays...")

        # CPU buffers
        self._target_rgb_buffer = np.empty((pixels, 3), dtype=np.float32)

        # GPU arrays for optimization (all dense)
        self._ATb_gpu = cp.zeros((led_count, 3), dtype=cp.float32)  # A^T @ b
        self._led_values_gpu = cp.full((led_count, 3), 0.5, dtype=cp.float32)  # x

        # Workspace for optimization loop
        self._gpu_workspace = {
            "gradient": cp.zeros((led_count, 3), dtype=cp.float32),  # ATA @ x - ATb
            "ATA_x": cp.zeros((led_count, 3), dtype=cp.float32),  # ATA @ x
            "x_new": cp.zeros((led_count, 3), dtype=cp.float32),  # Updated x
            "ATA_g": cp.zeros((led_count, 3), dtype=cp.float32),  # ATA @ gradient
        }

        workspace_mb = sum(arr.nbytes for arr in self._gpu_workspace.values()) / (
            1024 * 1024
        )
        logger.info(f"Dense workspace memory: {workspace_mb:.1f}MB")

    def optimize_frame(
        self,
        target_frame: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
    ) -> DenseOptimizationResult:
        """
        Optimize LED values using dense tensor operations (production method).

        This is the production optimization method that focuses purely on performance.
        For debugging with error metrics, use optimize_frame_with_debug() instead.

        Args:
            target_frame: Target image (height, width, 3) in range [0, 255]
            initial_values: Initial LED values (led_count, 3), if None uses 0.5
            max_iterations: Override default max iterations

        Returns:
            DenseOptimizationResult with LED values (no error metrics computed)
        """
        with self._timing.section("optimize_frame_production") as frame_timing:
            try:
                if not self._matrix_loaded:
                    raise RuntimeError("Dense matrices not loaded")

                # Validate input
                with self._timing.section("validation"):
                    if target_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                        raise ValueError(
                            f"Target frame shape {target_frame.shape} != {(FRAME_HEIGHT, FRAME_WIDTH, 3)}"
                        )

                # KEY STEP 2: Calculate A^T*b for current frame
                with self._timing.section("atb_calculation", flops=getattr(self, "_atb_flops_per_frame", 0)):
                    ATb = self._calculate_ATb(target_frame)

                # Initialize LED values
                with self._timing.section("initialization"):
                    if initial_values is not None:
                        initial_normalized = (
                            initial_values / 255.0
                            if initial_values.max() > 1.0
                            else initial_values
                        ).astype(np.float32)
                        self._led_values_gpu[:] = cp.asarray(initial_normalized)
                    else:
                        # Use pre-initialized 0.5 values
                        self._led_values_gpu.fill(0.5)

                # Dense tensor optimization loop with detailed timing
                iterations_used = max_iterations or self.max_iterations
                dense_loop_flops = iterations_used * getattr(self, "_dense_flops_per_iteration", 0)
                with self._timing.section("optimization_loop", flops=dense_loop_flops):
                    led_values_solved, loop_timing = self._solve_dense_gradient_descent(
                        ATb, max_iterations
                    )

                # Convert back to numpy
                with self._timing.section("conversion"):
                    led_values_normalized = cp.asnumpy(led_values_solved)
                    led_values_output = (led_values_normalized * 255.0).astype(np.uint8)

                # Get timing data from PerformanceTiming for statistics
                timing_data = self._timing.get_timing_data()
                optimization_time = timing_data["sections"]["optimize_frame_production"]["duration"]
                
                # Calculate FLOPs for statistics tracking
                iterations_used = max_iterations or self.max_iterations
                atb_flops = getattr(self, "_atb_flops_per_frame", 0)
                dense_loop_flops = iterations_used * getattr(self, "_dense_flops_per_iteration", 0)
                frame_flops = atb_flops + dense_loop_flops

                # Create streamlined result (no error metrics for production)
                result = DenseOptimizationResult(
                    led_values=led_values_output,
                    error_metrics={},  # Empty for production
                    iterations=iterations_used,
                    converged=True,
                )

                # Update statistics
                self._total_flops += frame_flops
                self._optimization_count += 1
                self._total_optimization_time += optimization_time

                logger.debug(f"Dense optimization completed in {optimization_time:.3f}s")
                return result

            except Exception as e:
                # Get timing data even on failure
                timing_data = self._timing.get_timing_data()
                optimization_time = timing_data.get("total_elapsed_time", 0.0)
                logger.error(
                    f"Dense optimization failed after {optimization_time:.3f}s: {e}"
                )

            # Return error result
            return DenseOptimizationResult(
                led_values=np.zeros((self._actual_led_count, 3), dtype=np.uint8),
                error_metrics={},
                iterations=0,
                converged=False,
            )

    def optimize_frame_with_debug(
        self,
        target_frame: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
    ) -> DenseOptimizationResult:
        """
        Optimize LED values with full debugging metrics (for development/testing).

        This method includes error metrics calculation and detailed timing information.
        Use optimize_frame() for production to avoid the overhead.

        Args:
            target_frame: Target image (height, width, 3) in range [0, 255]
            initial_values: Initial LED values (led_count, 3), if None uses 0.5
            max_iterations: Override default max iterations

        Returns:
            DenseOptimizationResult with LED values and comprehensive debug metrics
        """
        with self._timing.section("optimize_frame_debug") as frame_timing:
            try:
                if not self._matrix_loaded:
                    raise RuntimeError("Dense matrices not loaded")

                # Validate input
                with self._timing.section("validation"):
                    if target_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                        raise ValueError(
                            f"Target frame shape {target_frame.shape} != {(FRAME_HEIGHT, FRAME_WIDTH, 3)}"
                        )

                # Calculate A^T*b for current frame
                with self._timing.section("atb_calculation", flops=getattr(self, "_atb_flops_per_frame", 0)):
                    ATb = self._calculate_ATb(target_frame)

                # Initialize LED values
                with self._timing.section("initialization"):
                    if initial_values is not None:
                        initial_normalized = (
                            initial_values / 255.0
                            if initial_values.max() > 1.0
                            else initial_values
                        ).astype(np.float32)
                        self._led_values_gpu[:] = cp.asarray(initial_normalized)
                    else:
                        self._led_values_gpu.fill(0.5)

                # Dense tensor optimization loop with detailed timing
                iterations_used = max_iterations or self.max_iterations
                dense_loop_flops = iterations_used * getattr(self, "_dense_flops_per_iteration", 0)
                with self._timing.section("optimization_loop", flops=dense_loop_flops):
                    led_values_solved, loop_timing = self._solve_dense_gradient_descent(
                        ATb, max_iterations
                    )

                # Convert back to numpy
                with self._timing.section("conversion"):
                    led_values_normalized = cp.asnumpy(led_values_solved)
                    led_values_output = (led_values_normalized * 255.0).astype(np.uint8)

                # DEBUG: Compute error metrics using sparse matrices for accuracy
                with self._timing.section("debug_error_metrics"):
                    error_metrics = self._compute_error_metrics(led_values_solved, target_frame)

                # Get timing data for statistics tracking
                timing_data = self._timing.get_timing_data()
                optimization_time = timing_data["sections"]["optimize_frame_debug"]["duration"]
                
                # Calculate FLOPs for statistics tracking
                atb_flops = getattr(self, "_atb_flops_per_frame", 0)
                dense_loop_flops = iterations_used * getattr(self, "_dense_flops_per_iteration", 0)
                frame_flops = atb_flops + dense_loop_flops

                # Create comprehensive debug result
                result = DenseOptimizationResult(
                    led_values=led_values_output,
                    error_metrics=error_metrics,
                    iterations=iterations_used,
                    converged=True,
                    target_frame=target_frame.copy(),
                    precomputation_info={
                        "ata_shape": self._ATA_gpu.shape,
                        "ata_memory_mb": self._ATA_gpu.nbytes / (1024 * 1024),
                        "approach": "dense_precomputed_ata",
                    },
                )

                # Update statistics
                self._total_flops += frame_flops
                self._optimization_count += 1
                self._total_optimization_time += optimization_time

                logger.debug(f"Dense optimization completed in {optimization_time:.3f}s")
                return result

            except Exception as e:
                # Get timing data even on failure
                timing_data = self._timing.get_timing_data()
                optimization_time = timing_data.get("total_elapsed_time", 0.0)
                logger.error(
                    f"Dense optimization failed after {optimization_time:.3f}s: {e}"
                )

                # Return error result
                return DenseOptimizationResult(
                    led_values=np.zeros((self._actual_led_count, 3), dtype=np.uint8),
                    error_metrics={"mse": float("inf"), "mae": float("inf")},
                    iterations=0,
                    converged=False,
                )

    def _calculate_ATb(self, target_frame: np.ndarray) -> cp.ndarray:
        """
        Calculate A^T * b for the current target frame using block diagonal matrix.

        Args:
            target_frame: Target image (height, width, 3)

        Returns:
            ATb vector (led_count, 3) on GPU
        """
        # Preprocess target frame to flattened combined format [R; G; B]
        target_rgb_normalized = target_frame.astype(np.float32) / 255.0
        target_flattened = target_rgb_normalized.reshape(-1, 3)  # (pixels, 3)

        # Create combined target vector [R_pixels; G_pixels; B_pixels]
        pixels = target_flattened.shape[0]
        target_combined = np.empty(pixels * 3, dtype=np.float32)
        target_combined[:pixels] = target_flattened[:, 0]  # R channel
        target_combined[pixels : 2 * pixels] = target_flattened[:, 1]  # G channel
        target_combined[2 * pixels :] = target_flattened[:, 2]  # B channel

        # Transfer to GPU
        target_combined_gpu = cp.asarray(target_combined)

        # Single matrix multiplication: A_combined^T @ target_combined
        ATb_combined = self._A_combined_csc_gpu.T @ target_combined_gpu

        # Reshape result back to (led_count, 3)
        led_count = self._actual_led_count
        self._ATb_gpu[:, 0] = ATb_combined[:led_count]  # R LEDs
        self._ATb_gpu[:, 1] = ATb_combined[led_count : 2 * led_count]  # G LEDs
        self._ATb_gpu[:, 2] = ATb_combined[2 * led_count :]  # B LEDs

        return self._ATb_gpu

    def _solve_dense_gradient_descent(
        self, ATb: cp.ndarray, max_iterations: Optional[int]
    ) -> Tuple[cp.ndarray, Dict[str, float]]:
        """
        Solve using dense gradient descent with precomputed A^T*A using optimized einsum.

        This is the core innovation: all operations in the loop are dense tensors
        using einsum for optimal GPU utilization.

        Args:
            ATb: A^T * b vector (led_count, 3)
            max_iterations: Maximum iterations

        Returns:
            Tuple of (LED values (led_count, 3) on GPU, timing_info dict)
        """
        max_iters = max_iterations or self.max_iterations

        # Verify tensor shapes for einsum optimization
        assert self._ATA_gpu.shape == (
            self._actual_led_count,
            self._actual_led_count,
            3,
        ), (
            f"ATA shape {self._ATA_gpu.shape} != expected "
            f"({self._actual_led_count}, {self._actual_led_count}, 3)"
        )
        assert ATb.shape == (
            self._actual_led_count,
            3,
        ), f"ATb shape {ATb.shape} != expected ({self._actual_led_count}, 3)"

        # Get workspace arrays
        w = self._gpu_workspace
        x = self._led_values_gpu  # (led_count, 3)

        # Verify x shape
        assert x.shape == (
            self._actual_led_count,
            3,
        ), f"x shape {x.shape} != expected ({self._actual_led_count}, 3)"

        with self._timing.section("gradient_descent_loop") as loop_timing:
            for iteration in range(max_iters):
                # KEY OPTIMIZATION: Use einsum for parallel ATA @ x computation
                # ATA: (led_count, led_count, 3), x: (led_count, 3) -> (led_count, 3)
                # einsum 'ijk,jk->ik' computes all 3 channels in parallel
                with self._timing.section("einsum_operation", use_gpu_events=True):
                    w["ATA_x"][:] = cp.einsum("ijk,jk->ik", self._ATA_gpu, x)
                    w["gradient"][:] = w["ATA_x"] - ATb

                # KEY STEP 4: Compute step size using optimized dense operations
                with self._timing.section("step_size_calculation", use_gpu_events=True):
                    step_size, step_einsum_duration = self._compute_dense_step_size(
                        w["gradient"]
                    )

                # Gradient descent step with projection to [0, 1]
                w["x_new"][:] = cp.clip(x - step_size * w["gradient"], 0, 1)

                # Check convergence
                delta = cp.linalg.norm(w["x_new"] - x)
                if delta < self.convergence_threshold:
                    logger.debug(
                        f"Converged after {iteration+1} iterations, delta: {delta:.6f}"
                    )
                    break

                # Update (swap references for efficiency)
                x, w["x_new"] = w["x_new"], x

            # Ensure we return the current x
            self._led_values_gpu[:] = x

            # Get timing data from PerformanceTiming framework
            timing_data = self._timing.get_timing_data()
            
            # Create legacy timing info for compatibility
            timing_info = {
                "total_time": timing_data["sections"]["gradient_descent_loop"]["duration"],
                "iterations": iteration + 1,
                "total_loop_time": timing_data["sections"]["gradient_descent_loop"]["duration"],
                "einsum_time": timing_data["sections"].get("einsum_operation", {}).get("duration", 0.0),
                "step_size_time": timing_data["sections"].get("step_size_calculation", {}).get("duration", 0.0),
                "step_einsum_time": 0.0,  # Will be calculated in _compute_dense_step_size
                "iterations_completed": iteration + 1,
            }

            return x, timing_info

    def _compute_dense_step_size(self, gradient: cp.ndarray) -> Tuple[float, float]:
        """
        Compute step size using single einsum for g^T @ ATA @ g: (g^T @ g) / (g^T @ A^T*A @ g).

        Args:
            gradient: Gradient vector (led_count, 3)

        Returns:
            Tuple of (step_size, einsum_time)
        """
        # g^T @ g (sum over all elements)
        g_dot_g = cp.sum(gradient * gradient)

        # ULTRA-OPTIMIZED: Compute g^T @ ATA @ g for all channels in single einsum
        # ATA: (led_count, led_count, 3), gradient: (led_count, 3)
        # Computes sum over k of: g[:,k]^T @ ATA[:,:,k] @ g[:,k]

        # Use PerformanceTiming framework for GPU timing

        with self._timing.section("step_einsum_operation", use_gpu_events=True):
            g_dot_ATA_g = cp.einsum("ik,ijk,jk->", gradient, self._ATA_gpu, gradient)

        # Get timing data for this operation
        timing_data = self._timing.get_timing_data()
        einsum_duration = timing_data["sections"].get("step_einsum_operation", {}).get("duration", 0.0)

        if g_dot_ATA_g > 0:
            return (
                float(self.step_size_scaling * g_dot_g / g_dot_ATA_g),
                einsum_duration,
            )
        else:
            return 0.01, einsum_duration  # Fallback step size

    def _compute_error_metrics(
        self, led_values: cp.ndarray, target_frame: np.ndarray
    ) -> Dict[str, float]:
        """Compute error metrics using sparse matrices for accuracy."""
        # Convert target frame
        target_rgb = target_frame.astype(np.float32) / 255.0
        target_r = cp.asarray(target_rgb[:, :, 0].ravel())
        target_g = cp.asarray(target_rgb[:, :, 1].ravel())
        target_b = cp.asarray(target_rgb[:, :, 2].ravel())

        # Compute A @ x for each channel
        rendered_r = self._A_r_csc_gpu @ led_values[:, 0]
        rendered_g = self._A_g_csc_gpu @ led_values[:, 1]
        rendered_b = self._A_b_csc_gpu @ led_values[:, 2]

        # Compute residuals
        residual_r = rendered_r - target_r
        residual_g = rendered_g - target_g
        residual_b = rendered_b - target_b

        # Combine residuals for overall metrics
        residual_combined = cp.concatenate([residual_r, residual_g, residual_b])

        mse = float(cp.mean(residual_combined**2))
        mae = float(cp.mean(cp.abs(residual_combined)))
        max_error = float(cp.max(cp.abs(residual_combined)))
        rmse = float(cp.sqrt(mse))

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
            "optimizer_type": "dense_precomputed_ata",
            "device": str(self.device_info["device"]),
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
                avg_gflops_per_second = (
                    self.max_iterations * self._flops_per_iteration
                ) / (avg_time * 1e9)

            stats.update(
                {
                    "ata_tensor_shape": self._ATA_gpu.shape,
                    "ata_memory_mb": self._ATA_gpu.nbytes / (1024 * 1024),
                    "approach_description": "Precomputed A^T*A dense tensors with einsum",
                    "flop_analysis": {
                        "flops_per_iteration": int(self._flops_per_iteration),
                        "total_flops_computed": int(self._total_flops),
                        "average_gflops_per_frame": (
                            self.max_iterations * self._flops_per_iteration
                        )
                        / 1e9,
                        "average_gflops_per_second": avg_gflops_per_second,
                    },
                }
            )

        # Add PerformanceTiming insights
        timing_stats = self._timing.get_stats()
        stats["performance_timing"] = {
            "framework_active": True,
            "section_count": timing_stats["section_count"],
            "total_timing_overhead": timing_stats["total_duration"],
            "error_count": timing_stats["error_count"],
            "gpu_timing_enabled": timing_stats["gpu_timing_enabled"],
        }

        return stats
    

    def log_performance_insights(self, logger: logging.Logger, include_percentages: bool = True):
        """Log detailed performance insights using PerformanceTiming framework."""
        self._timing and self._timing.log(
            logger, 
            include_percentages=include_percentages, 
            sort_by="time"
        )
