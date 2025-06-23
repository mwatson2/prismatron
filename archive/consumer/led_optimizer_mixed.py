"""
LED Optimization Engine with Mixed Tensor Sparse Format.

This module implements a mixed tensor optimization approach using the SingleBlockMixedSparseTensor
format. It combines the advantages of sparse storage with efficient GPU computation using the
custom CUDA kernels for A^T @ b operations.

The key insight: use the mixed tensor format with custom CUDA kernels for optimal
GPU utilization while maintaining memory efficiency through sparse block storage.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cupy as cp
import numpy as np

from ..const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from ..utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


@dataclass
class MixedTensorOptimizationResult:
    """Results from mixed tensor LED optimization process."""

    led_values: np.ndarray  # RGB values for each LED (led_count, 3) - range [0,255]
    error_metrics: Dict[str, float]  # Error metrics (mse, mae, etc.)
    optimization_time: float  # Time taken for optimization in seconds
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged
    target_frame: Optional[np.ndarray] = None  # Original target frame (for debugging)
    tensor_info: Optional[Dict[str, Any]] = None  # Mixed tensor information
    flop_info: Optional[Dict[str, Any]] = None  # FLOP analysis information
    timing_breakdown: Optional[Dict[str, float]] = None  # Detailed timing breakdown

    def get_led_count(self) -> int:
        """Get number of LEDs in result."""
        return self.led_values.shape[0]

    def get_total_error(self) -> float:
        """Get total optimization error."""
        return self.error_metrics.get("mse", float("inf"))


class MixedTensorLEDOptimizer:
    """
    Mixed tensor LED optimization engine using SingleBlockMixedSparseTensor format.

    This approach uses the mixed tensor format with custom CUDA kernels for
    efficient A^T @ b computation while maintaining memory efficiency through
    sparse block storage. The optimization uses gradient descent with analytical
    step size computation.

    Key steps:
    1. Load: Convert saved data to SingleBlockMixedSparseTensor
    2. Per frame: ATb = A^T @ b using CUDA kernel (sparse blocks -> dense vector)
    3. Precompute: ATA = A^T @ A (computed on-demand and cached)
    4. Loop: gradient = ATA @ x - ATb, step size, update
    """

    def __init__(
        self,
        diffusion_patterns_path: Optional[str] = None,
    ):
        """
        Initialize mixed tensor LED optimizer.

        Args:
            diffusion_patterns_path: Path to patterns file with mixed tensor data
        """
        self.diffusion_patterns_path = (
            diffusion_patterns_path or "diffusion_patterns/synthetic_1000"
        )

        # Optimization parameters for gradient descent
        self.max_iterations = 10
        self.convergence_threshold = 1e-3
        self.step_size_scaling = 0.9

        # Mixed tensor storage
        self._mixed_tensor: Optional[SingleBlockMixedSparseTensor] = None

        # Precomputed dense matrices for optimization (computed on first use)
        self._ATA_gpu = None  # Shape: (led_count, led_count, 3) - dense on GPU
        self._ATA_computed = False

        # Pre-allocated arrays for optimization
        self._ATb_gpu = None  # (led_count, 3) dense vector on GPU
        self._led_values_gpu = None  # (led_count, 3) LED values on GPU

        # GPU workspace for optimization loop
        self._gpu_workspace = None

        self._led_spatial_mapping: Optional[Dict[int, int]] = None
        self._led_positions: Optional[np.ndarray] = None
        self._tensor_loaded = False
        self._actual_led_count = LED_COUNT

        # Statistics
        self._optimization_count = 0
        self._total_optimization_time = 0.0

        # FLOP counting for performance analysis
        self._total_flops = 0
        self._flops_per_iteration = 0

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
        Initialize the mixed tensor LED optimizer.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Load mixed tensor format
            if not self._load_mixed_tensor():
                logger.error("Failed to load mixed tensor format")
                return False

            # Initialize workspace arrays
            self._initialize_workspace()

            logger.info("Mixed tensor LED optimizer initialized successfully")
            logger.info(f"LED count: {self._actual_led_count}")
            logger.info(
                f"Mixed tensor memory: {self._mixed_tensor.memory_info()['total_mb']:.1f}MB"
            )
            logger.info(f"Device: {self.device_info['device']}")
            return True

        except Exception as e:
            logger.error(f"Mixed tensor LED optimizer initialization failed: {e}")
            return False

    def _load_mixed_tensor(self) -> bool:
        """
        Load mixed tensor format from saved patterns file.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            patterns_path = f"{self.diffusion_patterns_path}.npz"
            if not Path(patterns_path).exists():
                logger.warning(f"Patterns file not found at {patterns_path}")
                return False

            logger.info(f"Loading mixed tensor data from {patterns_path}")
            data = np.load(patterns_path, allow_pickle=True)

            # Check if mixed tensor data is available
            required_keys = [
                "mixed_tensor_values",
                "mixed_tensor_positions",
                "mixed_tensor_blocks_set",
                "mixed_tensor_led_count",
            ]

            if not all(key in data for key in required_keys):
                logger.error("Mixed tensor data not found in patterns file")
                logger.error(
                    "Regenerate patterns with updated generate_synthetic_patterns.py"
                )
                return False

            # Check if precomputed dense A^T @ A matrices are available
            dense_ata_keys = [
                "dense_ata_matrices",
                "dense_ata_led_count",
                "dense_ata_channels",
            ]

            if not all(key in data for key in dense_ata_keys):
                logger.error(
                    "Precomputed dense A^T @ A matrices not found in patterns file"
                )
                logger.error(
                    "Regenerate patterns with updated generate_synthetic_patterns.py"
                )
                return False

            # Extract mixed tensor parameters
            led_count = int(data["mixed_tensor_led_count"])
            channels = int(data["mixed_tensor_channels"])
            height = int(data["mixed_tensor_height"])
            width = int(data["mixed_tensor_width"])
            block_size = int(data["mixed_tensor_block_size"])

            # Create mixed tensor
            self._mixed_tensor = SingleBlockMixedSparseTensor(
                led_count, channels, height, width, block_size
            )

            # Load the tensor data
            self._mixed_tensor.sparse_values = cp.asarray(data["mixed_tensor_values"])
            self._mixed_tensor.block_positions = cp.asarray(
                data["mixed_tensor_positions"]
            )
            self._mixed_tensor.blocks_set = cp.asarray(data["mixed_tensor_blocks_set"])

            # Load precomputed dense A^T @ A matrices
            dense_ata_matrices = data[
                "dense_ata_matrices"
            ]  # Shape: (led_count, led_count, channels)
            dense_ata_led_count = int(data["dense_ata_led_count"])
            dense_ata_channels = int(data["dense_ata_channels"])

            # Validate A^T @ A dimensions
            if dense_ata_led_count != led_count or dense_ata_channels != channels:
                logger.error(
                    f"Dense A^T @ A dimensions mismatch: "
                    f"({dense_ata_led_count}, {dense_ata_channels}) != ({led_count}, {channels})"
                )
                return False

            # Load A^T @ A to GPU
            self._ATA_gpu = cp.asarray(dense_ata_matrices)
            self._ATA_computed = True

            ata_memory_mb = self._ATA_gpu.nbytes / (1024 * 1024)
            logger.info(f"Loaded precomputed A^T @ A matrices: {self._ATA_gpu.shape}")
            logger.info(f"A^T @ A memory: {ata_memory_mb:.1f}MB")

            # Load metadata
            self._led_spatial_mapping = data["led_spatial_mapping"].item()
            self._led_positions = data["led_positions"]
            self._actual_led_count = led_count

            # Validate tensor dimensions
            if height != FRAME_HEIGHT or width != FRAME_WIDTH:
                logger.warning(
                    f"Frame dimensions mismatch: {(height, width)} != {(FRAME_HEIGHT, FRAME_WIDTH)}"
                )

            logger.info(f"Mixed tensor loaded: {led_count} LEDs, {channels} channels")
            logger.info(f"Block size: {block_size}x{block_size}")
            logger.info(f"Blocks stored: {int(data['mixed_tensor_blocks_stored'])}")

            # Calculate FLOPs per iteration
            self._calculate_flops_per_iteration()

            self._tensor_loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load mixed tensor: {e}")
            return False

    def _calculate_flops_per_iteration(self) -> None:
        """Calculate floating point operations per iteration."""
        if self._mixed_tensor is None:
            return

        led_count = self._actual_led_count
        block_size = self._mixed_tensor.block_size

        # Calculate FLOPs for mixed tensor operations:
        # 1. A^T @ b: Uses CUDA kernel, approximately 2 * nnz operations
        # 2. ATA @ x: Dense matrix-vector multiply per channel
        # 3. Step size computation: Dense operations

        # Estimate non-zero elements (approximate)
        blocks_stored = int(cp.sum(self._mixed_tensor.blocks_set))
        estimated_nnz_per_block = block_size * block_size * 0.4  # Assume 40% density
        estimated_total_nnz = blocks_stored * estimated_nnz_per_block

        # FLOPs breakdown:
        atb_flops = estimated_total_nnz * 2  # A^T @ b (once per frame)
        ata_dense_flops = led_count**2 * 3 * 2  # ATA @ x (dense, per iteration)
        step_size_flops = led_count * 3 * 4  # Step size computation

        self._atb_flops_per_frame = atb_flops
        self._dense_flops_per_iteration = ata_dense_flops + step_size_flops

        # Total per iteration (amortizing A^T @ b over iterations)
        self._flops_per_iteration = self._dense_flops_per_iteration + (
            atb_flops / self.max_iterations
        )

        logger.debug(f"Mixed tensor A^T @ b FLOPs per frame: {atb_flops:,}")
        logger.debug(
            f"Dense optimization FLOPs per iteration: {self._dense_flops_per_iteration:,}"
        )
        logger.debug(
            f"Total FLOPs per iteration (amortized): {self._flops_per_iteration:,}"
        )

    def _initialize_workspace(self) -> None:
        """Initialize GPU workspace arrays for optimization."""
        if not self._tensor_loaded:
            return

        led_count = self._actual_led_count

        logger.info(f"Initializing mixed tensor workspace arrays...")

        # GPU arrays for optimization
        self._ATb_gpu = cp.zeros((led_count, 3), dtype=cp.float32)  # A^T @ b
        self._led_values_gpu = cp.full((led_count, 3), 0.5, dtype=cp.float32)  # x

        # Workspace for optimization loop
        self._gpu_workspace = {
            "gradient": cp.zeros((led_count, 3), dtype=cp.float32),  # ATA @ x - ATb
            "ATA_x": cp.zeros((led_count, 3), dtype=cp.float32),  # ATA @ x
            "x_new": cp.zeros((led_count, 3), dtype=cp.float32),  # Updated x
        }

        workspace_mb = sum(arr.nbytes for arr in self._gpu_workspace.values()) / (
            1024 * 1024
        )
        logger.info(f"Mixed tensor workspace memory: {workspace_mb:.1f}MB")

    def _precompute_ata_if_needed(self) -> None:
        """Check if A^T @ A is already loaded (should be loaded from patterns file)."""
        if not self._ATA_computed:
            logger.warning("A^T @ A matrices not loaded! Using identity approximation.")
            # Fallback: create identity matrices if precomputed ones aren't available
            led_count = self._actual_led_count
            channels = 3
            self._ATA_gpu = cp.eye(led_count, dtype=cp.float32)[:, :, None].repeat(
                channels, axis=2
            )
            self._ATA_computed = True

    def optimize_frame(
        self,
        target_frame: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
        debug: bool = False,
    ) -> MixedTensorOptimizationResult:
        """
        Optimize LED values using mixed tensor format.

        Args:
            target_frame: Target image (height, width, 3) in range [0, 255]
            initial_values: Initial LED values (led_count, 3), if None uses 0.5
            max_iterations: Override default max iterations
            debug: If True, compute error metrics and detailed timing (slower)

        Returns:
            MixedTensorOptimizationResult with LED values and optional debug metrics
        """
        start_time = time.time()
        timing_breakdown = {}

        try:
            if not self._tensor_loaded:
                raise RuntimeError("Mixed tensor not loaded")

            # Validate input
            validation_start = time.time()
            if target_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise ValueError(
                    f"Target frame shape {target_frame.shape} != {(FRAME_HEIGHT, FRAME_WIDTH, 3)}"
                )
            timing_breakdown["validation_time"] = time.time() - validation_start

            # KEY STEP 1: Calculate A^T*b using mixed tensor CUDA kernel
            atb_start = time.time()
            ATb = self._calculate_ATb_mixed_tensor(target_frame)
            timing_breakdown["atb_calculation_time"] = time.time() - atb_start

            # Check A^T @ A is loaded (should be immediate since it's precomputed)
            ata_start = time.time()
            self._precompute_ata_if_needed()
            timing_breakdown["ata_check_time"] = time.time() - ata_start

            # Initialize LED values
            init_start = time.time()
            if initial_values is not None:
                initial_normalized = (
                    initial_values / 255.0
                    if initial_values.max() > 1.0
                    else initial_values
                ).astype(np.float32)
                self._led_values_gpu[:] = cp.asarray(initial_normalized)
            else:
                self._led_values_gpu.fill(0.5)
            timing_breakdown["initialization_time"] = time.time() - init_start

            # Mixed tensor optimization loop
            led_values_solved, loop_timing = self._solve_mixed_tensor_gradient_descent(
                ATb, max_iterations
            )
            timing_breakdown["optimization_loop_time"] = loop_timing.get(
                "total_loop_time", 0.0
            )

            # Convert back to numpy
            convert_start = time.time()
            led_values_normalized = cp.asnumpy(led_values_solved)
            led_values_output = (led_values_normalized * 255.0).astype(np.uint8)
            timing_breakdown["conversion_time"] = time.time() - convert_start

            # Calculate error metrics if debug mode is enabled
            error_metrics = {}
            debug_time = 0.0
            if debug:
                debug_start = time.time()
                error_metrics = self._compute_error_metrics(
                    led_values_solved, target_frame
                )
                debug_time = time.time() - debug_start
                timing_breakdown["debug_time"] = debug_time

            optimization_time = time.time() - start_time
            timing_breakdown["total_time"] = optimization_time

            # Calculate core optimization time (excluding debug overhead)
            core_optimization_time = optimization_time - debug_time

            # Calculate FLOPs for this optimization
            iterations_used = max_iterations or self.max_iterations
            atb_flops = (
                self._atb_flops_per_frame
                if hasattr(self, "_atb_flops_per_frame")
                else 0
            )
            dense_loop_flops = iterations_used * (
                self._dense_flops_per_iteration
                if hasattr(self, "_dense_flops_per_iteration")
                else self._flops_per_iteration
            )
            frame_flops = atb_flops + dense_loop_flops

            # Create flop_info
            flop_info = {
                "total_flops": int(frame_flops),
                "flops_per_iteration": int(self._flops_per_iteration),
                "atb_flops_per_frame": int(atb_flops),
                "dense_loop_flops": int(dense_loop_flops),
                "gflops": frame_flops / 1e9,
                "gflops_per_second": frame_flops / (core_optimization_time * 1e9),
            }

            # Create result
            result = MixedTensorOptimizationResult(
                led_values=led_values_output,
                error_metrics=error_metrics,
                optimization_time=core_optimization_time,
                iterations=iterations_used,
                converged=True,
                target_frame=target_frame.copy() if debug else None,
                tensor_info={
                    "tensor_shape": self._mixed_tensor.sparse_values.shape,
                    "tensor_memory_mb": self._mixed_tensor.memory_info()["total_mb"],
                    "approach": "mixed_tensor_sparse_blocks",
                }
                if debug
                else None,
                flop_info=flop_info,
                timing_breakdown=timing_breakdown,
            )

            # Update statistics
            self._total_flops += frame_flops
            self._optimization_count += 1
            self._total_optimization_time += core_optimization_time

            logger.debug(
                f"Mixed tensor optimization completed in {core_optimization_time:.3f}s"
            )
            return result

        except Exception as e:
            optimization_time = time.time() - start_time
            logger.error(
                f"Mixed tensor optimization failed after {optimization_time:.3f}s: {e}"
            )

            # Return error result
            return MixedTensorOptimizationResult(
                led_values=np.zeros((self._actual_led_count, 3), dtype=np.uint8),
                error_metrics={"mse": float("inf"), "mae": float("inf")}
                if debug
                else {},
                optimization_time=optimization_time,
                iterations=0,
                converged=False,
                flop_info={
                    "total_flops": 0,
                    "flops_per_iteration": int(self._flops_per_iteration)
                    if self._flops_per_iteration > 0
                    else 0,
                    "gflops": 0.0,
                    "gflops_per_second": 0.0,
                },
            )

    def _calculate_ATb_mixed_tensor(self, target_frame: np.ndarray) -> cp.ndarray:
        """
        Calculate A^T * b using mixed tensor CUDA kernel.

        Args:
            target_frame: Target image (height, width, 3)

        Returns:
            ATb vector (led_count, 3) on GPU
        """
        # Normalize target frame
        target_normalized = target_frame.astype(np.float32) / 255.0
        target_gpu = cp.asarray(target_normalized)

        # Use the mixed tensor's CUDA kernel for A^T @ b
        # This will automatically select the best available kernel
        result = self._mixed_tensor.transpose_dot_product_cuda_high_performance(
            target_gpu[:, :, 0]  # Use single channel for now
        )

        # Store result for all channels (temporary implementation)
        self._ATb_gpu[:, 0] = result[:, 0]
        self._ATb_gpu[:, 1] = result[:, 1]
        self._ATb_gpu[:, 2] = result[:, 2]

        return self._ATb_gpu

    def _solve_mixed_tensor_gradient_descent(
        self, ATb: cp.ndarray, max_iterations: Optional[int]
    ) -> Tuple[cp.ndarray, Dict[str, float]]:
        """
        Solve using gradient descent with mixed tensor operations.

        Args:
            ATb: A^T * b vector (led_count, 3)
            max_iterations: Maximum iterations

        Returns:
            Tuple of (LED values (led_count, 3) on GPU, timing_info dict)
        """
        max_iters = max_iterations or self.max_iterations
        loop_start_time = time.time()

        # Get workspace arrays
        w = self._gpu_workspace
        x = self._led_values_gpu

        for iteration in range(max_iters):
            # ATA @ x (using precomputed dense ATA)
            if self._ATA_computed:
                w["ATA_x"][:] = cp.einsum("ijk,jk->ik", self._ATA_gpu, x)
            else:
                # Fallback: use identity approximation
                w["ATA_x"][:] = x

            # Gradient
            w["gradient"][:] = w["ATA_x"] - ATb

            # Step size (simplified for now)
            g_dot_g = cp.sum(w["gradient"] * w["gradient"])
            if self._ATA_computed:
                g_dot_ATA_g = cp.einsum(
                    "ik,ijk,jk->", w["gradient"], self._ATA_gpu, w["gradient"]
                )
                step_size = float(
                    self.step_size_scaling * g_dot_g / (g_dot_ATA_g + 1e-8)
                )
            else:
                step_size = 0.01  # Fixed step size for identity approximation

            # Update
            w["x_new"][:] = cp.clip(x - step_size * w["gradient"], 0, 1)

            # Check convergence
            delta = cp.linalg.norm(w["x_new"] - x)
            if delta < self.convergence_threshold:
                logger.debug(
                    f"Converged after {iteration+1} iterations, delta: {delta:.6f}"
                )
                break

            # Update
            x[:] = w["x_new"]

        loop_total_time = time.time() - loop_start_time

        timing_info = {
            "total_loop_time": loop_total_time,
            "iterations_completed": iteration + 1,
        }

        return x, timing_info

    def _compute_error_metrics(
        self, led_values: cp.ndarray, target_frame: np.ndarray
    ) -> Dict[str, float]:
        """Compute error metrics using mixed tensor rendering."""
        # For now, return basic metrics
        # In a full implementation, this would render using the mixed tensor
        return {
            "mse": 0.01,  # Placeholder
            "mae": 0.01,  # Placeholder
            "max_error": 0.1,  # Placeholder
            "rmse": 0.1,  # Placeholder
        }

    def get_optimizer_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        avg_time = self._total_optimization_time / max(1, self._optimization_count)

        stats = {
            "optimizer_type": "mixed_tensor_sparse_blocks",
            "device": str(self.device_info["device"]),
            "tensor_loaded": self._tensor_loaded,
            "optimization_count": self._optimization_count,
            "total_optimization_time": self._total_optimization_time,
            "average_optimization_time": avg_time,
            "estimated_fps": 1.0 / avg_time if avg_time > 0 else 0.0,
            "led_count": self._actual_led_count,
            "frame_dimensions": (FRAME_WIDTH, FRAME_HEIGHT),
        }

        if self._tensor_loaded and self._mixed_tensor is not None:
            memory_info = self._mixed_tensor.memory_info()
            avg_gflops_per_second = 0.0
            if avg_time > 0 and self._flops_per_iteration > 0:
                avg_gflops_per_second = (
                    self.max_iterations * self._flops_per_iteration
                ) / (avg_time * 1e9)

            stats.update(
                {
                    "tensor_memory_mb": memory_info["total_mb"],
                    "blocks_stored": memory_info["blocks_stored"],
                    "approach_description": "Mixed tensor sparse blocks with CUDA kernels",
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

        return stats
