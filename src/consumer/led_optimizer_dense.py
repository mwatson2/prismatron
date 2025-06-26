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

# import time  # Removed for clean utility class implementation
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cupy as cp
import numpy as np
import scipy.sparse as sp

from ..const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from ..utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from ..utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

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
    optimization_time: float = 0.0  # Timing removed, kept for API compatibility

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
        use_mixed_tensor: bool = False,
    ):
        """
        Initialize dense tensor LED optimizer.

        Args:
            diffusion_patterns_path: Path to sparse diffusion matrix files
            use_mixed_tensor: If True, use mixed tensor format for A^T@b calculation.
                             If False, use CSC sparse format for A^T@b calculation.
                             Both modes use the same dense A^T@A matrices for optimization.
        """
        self.diffusion_patterns_path = (
            diffusion_patterns_path or "diffusion_patterns/synthetic_1000"
        )
        self.use_mixed_tensor = use_mixed_tensor

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

        # Utility class instances for encapsulated matrix operations
        self._diffusion_matrix: Optional[LEDDiffusionCSCMatrix] = None
        self._mixed_tensor: Optional[SingleBlockMixedSparseTensor] = None

        self._led_spatial_mapping: Optional[Dict[int, int]] = None
        self._led_positions: Optional[np.ndarray] = None
        self._matrix_loaded = False
        self._actual_led_count = LED_COUNT

        # Statistics
        self._optimization_count = 0

        # Timing and FLOP fields removed but kept for API compatibility
        self._total_optimization_time = 0.0
        self._total_flops = 0
        self._flops_per_iteration = 0
        self._dense_flops_per_iteration = 0

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

            # Warm up GPU cache with dummy operations
            self._warm_gpu_cache()

            logger.info("Dense LED optimizer initialized successfully")
            logger.info(f"LED count: {self._actual_led_count}")
            logger.info(f"Dense ATA tensor shape: {self._ATA_gpu.shape}")
            logger.info(f"Device: {self.device_info['device']}")
            return True

        except Exception as e:
            logger.error(f"Dense LED optimizer initialization failed: {e}")
            return False

    def _load_precomputed_ata_matrices(self, data: np.lib.npyio.NpzFile) -> bool:
        """
        Load precomputed A^T@A matrices from the pattern file if available.

        Args:
            data: Loaded NPZ file data

        Returns:
            True if precomputed matrices were loaded successfully, False otherwise
        """
        try:
            # Check for required dense A^T@A keys
            required_keys = [
                "dense_ata_matrices",
                "dense_ata_led_count",
                "dense_ata_channels",
            ]

            if not all(key in data for key in required_keys):
                logger.debug("Precomputed A^T@A matrices not found in pattern file")
                return False

            # Load A^T@A data
            dense_ata_matrices = data[
                "dense_ata_matrices"
            ]  # Shape: (led_count, led_count, channels)
            dense_ata_led_count = int(data["dense_ata_led_count"])
            dense_ata_channels = int(data["dense_ata_channels"])

            # Validate dimensions
            if dense_ata_led_count != self._actual_led_count or dense_ata_channels != 3:
                logger.warning(
                    f"Precomputed A^T@A dimensions mismatch: "
                    f"({dense_ata_led_count}, {dense_ata_channels}) != "
                    f"({self._actual_led_count}, 3)"
                )
                return False

            # Store the precomputed matrices
            self._ATA_cpu = dense_ata_matrices.astype(np.float32)

            ata_memory_mb = self._ATA_cpu.nbytes / (1024 * 1024)
            computation_time = data.get("dense_ata_computation_time", 0.0)

            logger.info(f"Loaded precomputed A^T@A matrices: {self._ATA_cpu.shape}")
            logger.info(f"A^T@A memory: {ata_memory_mb:.1f}MB")
            logger.info(f"Original computation time: {computation_time:.2f}s")

            return True

        except Exception as e:
            logger.warning(f"Failed to load precomputed A^T@A matrices: {e}")
            return False

    def _load_csc_format(self, A_r_csc, A_g_csc, A_b_csc) -> None:
        """Load CSC sparse format for A^T@b calculation."""
        # Create combined block diagonal matrix for faster A^T*b calculation
        logger.info("Creating combined block diagonal matrix for A^T*b...")
        from scipy.sparse import block_diag

        A_combined_csc = block_diag([A_r_csc, A_g_csc, A_b_csc], format="csc")
        logger.info(f"Combined matrix shape: {A_combined_csc.shape}")

        # Transfer matrices to GPU
        logger.info("Transferring CSC matrices to GPU...")
        from cupyx.scipy.sparse import csc_matrix as cupy_csc_matrix

        self._A_r_csc_gpu = cupy_csc_matrix(A_r_csc)
        self._A_g_csc_gpu = cupy_csc_matrix(A_g_csc)
        self._A_b_csc_gpu = cupy_csc_matrix(A_b_csc)
        self._A_combined_csc_gpu = cupy_csc_matrix(A_combined_csc)

    def _load_mixed_tensor_format(self, data: np.lib.npyio.NpzFile) -> None:
        """Load mixed tensor format for A^T@b calculation."""
        from ..utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

        # Check if mixed tensor data is available
        required_keys = [
            "mixed_tensor_values",
            "mixed_tensor_positions",
            "mixed_tensor_blocks_set",
            "mixed_tensor_led_count",
        ]

        if not all(key in data for key in required_keys):
            raise ValueError(
                "Mixed tensor data not found in patterns file. "
                "Regenerate patterns with mixed tensor support."
            )

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

        # Load the tensor data and transpose to channels-first layout
        # Old format: (LEDs, Channels, H, W) -> New format: (Channels, LEDs, H, W)
        old_values = data["mixed_tensor_values"]  # (1000, 3, 96, 96)
        old_positions = data["mixed_tensor_positions"]  # (1000, 3, 2)
        old_blocks_set = data["mixed_tensor_blocks_set"]  # (1000, 3)

        # Transpose to channels-first layout
        self._mixed_tensor.sparse_values = cp.asarray(
            old_values.transpose(1, 0, 2, 3)
        )  # (3, 1000, 96, 96)
        self._mixed_tensor.block_positions = cp.asarray(
            old_positions.transpose(1, 0, 2)
        )  # (3, 1000, 2)
        self._mixed_tensor.blocks_set = cp.asarray(
            old_blocks_set.transpose(1, 0)
        )  # (3, 1000)

        blocks_stored = int(data["mixed_tensor_blocks_stored"])
        logger.info(f"Mixed tensor loaded: {led_count} LEDs, {channels} channels")
        logger.info(f"Block size: {block_size}x{block_size}")
        logger.info(f"Blocks stored: {blocks_stored}")

    def _load_and_precompute_dense_matrices(self) -> bool:
        """
        Load diffusion matrices using utility classes and precomputed A^T*A tensors.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            patterns_path = (
                self.diffusion_patterns_path
                if self.diffusion_patterns_path.endswith(".npz")
                else f"{self.diffusion_patterns_path}.npz"
            )
            if not Path(patterns_path).exists():
                logger.warning(f"Patterns file not found at {patterns_path}")
                return False

            logger.info(f"Loading patterns from {patterns_path}")
            data = np.load(patterns_path, allow_pickle=True)

            # Check for new nested format first
            if "diffusion_matrix" in data and "mixed_tensor" in data:
                return self._load_new_format(data)
            else:
                return self._load_legacy_format(data)

        except Exception as e:
            logger.error(f"Failed to load matrices: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def _load_new_format(self, data: np.lib.npyio.NpzFile) -> bool:
        """Load matrices from new nested format using utility classes."""
        logger.info("Detected new nested format with utility classes")

        # Load LEDDiffusionCSCMatrix
        logger.info("Loading LEDDiffusionCSCMatrix...")
        diffusion_dict = data["diffusion_matrix"].item()
        self._diffusion_matrix = LEDDiffusionCSCMatrix.from_dict(diffusion_dict)

        # Load metadata from diffusion matrix
        self._led_spatial_mapping = data.get("led_spatial_mapping", {})
        if hasattr(self._led_spatial_mapping, "item"):
            self._led_spatial_mapping = self._led_spatial_mapping.item()
        self._led_positions = data.get("led_positions", None)
        self._actual_led_count = self._diffusion_matrix.led_count

        # Validate matrix dimensions
        expected_pixels = FRAME_HEIGHT * FRAME_WIDTH
        if self._diffusion_matrix.pixels != expected_pixels:
            logger.error(
                f"Matrix pixel dimension mismatch: "
                f"{self._diffusion_matrix.pixels} != {expected_pixels}"
            )
            return False

        # Load precomputed A^T*A matrices
        if not self._load_precomputed_ata_matrices(data):
            logger.error("Precomputed A^T*A matrices required but not found")
            return False

        # Transfer to GPU
        logger.info("Transferring dense ATA tensor to GPU...")
        self._ATA_gpu = cp.asarray(self._ATA_cpu)

        # Load mixed tensor if needed
        if self.use_mixed_tensor:
            logger.info("Loading SingleBlockMixedSparseTensor...")
            mixed_dict = data["mixed_tensor"].item()
            self._mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_dict)
            logger.info("Loaded mixed tensor format for A^T@b calculation")

        # Load CSC format for A^T@b calculation (always needed)
        logger.info("Setting up CSC matrices for A^T@b calculation...")
        self._setup_csc_matrices()

        # Calculate memory usage
        ata_memory_mb = self._ATA_gpu.nbytes / (1024 * 1024)
        logger.info(f"Dense ATA tensor memory: {ata_memory_mb:.1f}MB")

        self._matrix_loaded = True
        logger.info("Matrix loading completed successfully")
        return True

    def _setup_csc_matrices(self) -> None:
        """Setup CSC matrices for GPU operations using utility class methods."""
        logger.info("Setting up CSC matrices for GPU operations...")

        # Use utility class methods to get GPU matrices
        (
            self._A_r_csc_gpu,
            self._A_g_csc_gpu,
            self._A_b_csc_gpu,
            self._A_combined_csc_gpu,
        ) = self._diffusion_matrix.to_gpu_matrices()

        logger.info(f"CSC matrices transferred to GPU successfully")

    def _load_legacy_format(self, data: np.lib.npyio.NpzFile) -> bool:
        """Load matrices from legacy format (fallback for old files)."""
        logger.warning("Using legacy format loading - consider regenerating patterns")

        # Reconstruct sparse matrix from components
        sparse_matrix_csc = sp.csc_matrix(
            (data["matrix_data"], data["matrix_indices"], data["matrix_indptr"]),
            shape=tuple(data["matrix_shape"]),
        )

        # Create LEDDiffusionCSCMatrix from legacy data
        self._diffusion_matrix = LEDDiffusionCSCMatrix.from_csc_matrix(
            sparse_matrix_csc, FRAME_HEIGHT, FRAME_WIDTH, channels=3
        )

        # Load metadata
        self._led_spatial_mapping = data["led_spatial_mapping"].item()
        self._led_positions = data["led_positions"]
        self._actual_led_count = self._diffusion_matrix.led_count

        # Load precomputed A^T*A matrices or compute fallback
        if not self._load_precomputed_ata_matrices(data):
            logger.warning("Computing A^T*A matrices from scratch (slow)")
            self._compute_ata_fallback()

        # Transfer to GPU
        self._ATA_gpu = cp.asarray(self._ATA_cpu)

        # Setup CSC matrices
        self._setup_csc_matrices()

        self._matrix_loaded = True
        return True

    def _compute_ata_fallback(self) -> None:
        """Compute A^T*A matrices as fallback for legacy files."""
        logger.info("Computing A^T*A matrices from sparse matrices...")

        # Extract RGB channels using utility class
        A_r_csc, A_g_csc, A_b_csc = self._diffusion_matrix.extract_rgb_channels()

        # Compute A^T*A on CPU
        ATA_r = (A_r_csc.T @ A_r_csc).toarray()
        ATA_g = (A_g_csc.T @ A_g_csc).toarray()
        ATA_b = (A_b_csc.T @ A_b_csc).toarray()

        # Stack into 3D tensor: (led_count, led_count, 3)
        self._ATA_cpu = np.stack([ATA_r, ATA_g, ATA_b], axis=2).astype(np.float32)

    def _calculate_flops_per_iteration(self) -> None:
        """FLOP calculation removed - kept as stub for API compatibility."""
        # All FLOP calculations removed for clean implementation
        # These fields are maintained as zeros for API compatibility
        self._dense_flops_per_iteration = 0
        self._flops_per_iteration = 0

        logger.debug("FLOP calculations removed for clean implementation")

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

    def _reset_workspace_references(self) -> None:
        """Reset workspace array references to ensure deterministic behavior."""
        if not hasattr(self, "_gpu_workspace") or self._gpu_workspace is None:
            return

        # Clear all workspace arrays to zero
        for key, arr in self._gpu_workspace.items():
            arr.fill(0.0)

        # Ensure LED values are properly initialized
        if hasattr(self, "_led_values_gpu") and self._led_values_gpu is not None:
            self._led_values_gpu.fill(0.5)

    def _warm_gpu_cache(self) -> None:
        """
        Warm up GPU cache by running dummy sparse and dense operations.

        This eliminates cold-start penalties observed in both A^T*b and dense operations.
        """
        logger.info("Warming up GPU cache for sparse and dense operations...")

        # Part 1: Warm sparse/mixed tensor operations
        if self.use_mixed_tensor:
            # Warm mixed tensor CUDA kernels (single channel)
            dummy_channel = cp.ones((FRAME_HEIGHT, FRAME_WIDTH), dtype=cp.float32)
            for i in range(3):
                _ = self._mixed_tensor.transpose_dot_product_cuda_high_performance(
                    dummy_channel
                )
        else:
            # Warm CSC sparse operations
            dummy_size = self._A_combined_csc_gpu.shape[0]
            dummy_target = cp.ones(dummy_size, dtype=cp.float32)
            for i in range(3):
                _ = self._A_combined_csc_gpu.T @ dummy_target
        cp.cuda.runtime.deviceSynchronize()

        # Part 2: Warm dense operations (einsum and step size calculation)
        dummy_x = cp.ones((self._actual_led_count, 3), dtype=cp.float32)
        dummy_gradient = cp.ones((self._actual_led_count, 3), dtype=cp.float32)

        for i in range(3):
            # Warm the main einsum operation
            _ = cp.einsum("ijk,jk->ik", self._ATA_gpu, dummy_x)
            # Warm the step size calculation (full path including GPU events)
            _ = self._compute_dense_step_size(dummy_gradient)
        cp.cuda.runtime.deviceSynchronize()

        logger.info("GPU cache warmed up")

    def optimize_frame(
        self,
        target_frame: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
        debug: bool = False,
    ) -> DenseOptimizationResult:
        """
        Optimize LED values using dense tensor operations.

        Args:
            target_frame: Target image (height, width, 3) in range [0, 255]
            initial_values: Initial LED values (led_count, 3), if None uses 0.5
            max_iterations: Override default max iterations
            debug: If True, compute error metrics and detailed timing (slower)

        Returns:
            DenseOptimizationResult with LED values and optional debug metrics
        """

        try:
            if not self._matrix_loaded:
                raise RuntimeError("Dense matrices not loaded")

            # Validate input
            if target_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise ValueError(
                    f"Target frame shape {target_frame.shape} != {(FRAME_HEIGHT, FRAME_WIDTH, 3)}"
                )

            # KEY STEP 2: Calculate A^T*b for current frame
            ATb = self._calculate_ATb(target_frame)

            # Initialize LED values
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

            # Reset workspace references to ensure deterministic behavior
            self._reset_workspace_references()

            # Dense tensor optimization loop
            (
                led_values_solved,
                iterations_completed,
            ) = self._solve_dense_gradient_descent(ATb, max_iterations)

            # Convert back to numpy
            led_values_normalized = cp.asnumpy(led_values_solved)
            led_values_output = (led_values_normalized * 255.0).astype(np.uint8)

            # Calculate error metrics if debug mode is enabled
            error_metrics = {}
            if debug:
                error_metrics = self._compute_error_metrics(
                    led_values_solved, target_frame
                )

            # Create result with optional debug information
            result = DenseOptimizationResult(
                led_values=led_values_output,
                error_metrics=error_metrics,
                iterations=iterations_completed,
                converged=True,
                target_frame=target_frame.copy() if debug else None,
                precomputation_info={
                    "ata_shape": self._ATA_gpu.shape,
                    "ata_memory_mb": self._ATA_gpu.nbytes / (1024 * 1024),
                    "approach": "dense_precomputed_ata",
                }
                if debug
                else None,
            )

            # Update statistics
            self._optimization_count += 1

            logger.debug("Dense optimization completed")
            return result

        except Exception as e:
            logger.error(f"Dense optimization failed: {e}")

            # Return error result
            return DenseOptimizationResult(
                led_values=np.zeros((self._actual_led_count, 3), dtype=np.uint8),
                error_metrics={"mse": float("inf"), "mae": float("inf")}
                if debug
                else {},
                iterations=0,
                converged=False,
            )

    def _calculate_ATb(self, target_frame: np.ndarray) -> cp.ndarray:
        """
        Calculate A^T * b for the current target frame.

        Uses either CSC sparse format or mixed tensor format based on use_mixed_tensor flag.

        Args:
            target_frame: Target image (height, width, 3)

        Returns:
            ATb vector (led_count, 3) on GPU
        """
        if self.use_mixed_tensor:
            return self._calculate_ATb_mixed_tensor(target_frame)
        else:
            return self._calculate_ATb_csc_format(target_frame)

    def _convert_frame_to_flat_format(self, target_frame: np.ndarray) -> cp.ndarray:
        """Convert frame to flat format for CSC sparse operations."""
        target_rgb_normalized = target_frame.astype(np.float32) / 255.0
        target_flattened = target_rgb_normalized.reshape(-1, 3)  # (pixels, 3)

        # Create combined target vector [R_pixels; G_pixels; B_pixels]
        pixels = target_flattened.shape[0]
        target_combined = np.empty(pixels * 3, dtype=np.float32)
        target_combined[:pixels] = target_flattened[:, 0]  # R channel
        target_combined[pixels : 2 * pixels] = target_flattened[:, 1]  # G channel
        target_combined[2 * pixels :] = target_flattened[:, 2]  # B channel

        # Transfer to GPU
        return cp.asarray(target_combined)

    def _convert_frame_to_planar_format(self, target_frame: np.ndarray) -> cp.ndarray:
        """Convert frame to planar format for mixed tensor operations."""
        # Normalize and convert to float32
        target_normalized = target_frame.astype(np.float32) / 255.0

        # Convert from HWC to CHW format for mixed tensor
        target_planar = np.transpose(target_normalized, (2, 0, 1))  # (3, height, width)

        # Transfer to GPU
        return cp.asarray(target_planar)

    def _calculate_ATb_csc_format(self, target_frame: np.ndarray) -> cp.ndarray:
        """Calculate A^T@b using CSC sparse format."""
        # Convert frame to flat format
        target_combined_gpu = self._convert_frame_to_flat_format(target_frame)

        # Actual sparse matrix operation
        ATb_combined = self._A_combined_csc_gpu.T @ target_combined_gpu

        # Convert result back to tensor form
        led_count = self._actual_led_count
        self._ATb_gpu[:, 0] = ATb_combined[:led_count]  # R LEDs
        self._ATb_gpu[:, 1] = ATb_combined[led_count : 2 * led_count]  # G LEDs
        self._ATb_gpu[:, 2] = ATb_combined[2 * led_count :]  # B LEDs

        return self._ATb_gpu

    def _calculate_ATb_mixed_tensor(self, target_frame: np.ndarray) -> cp.ndarray:
        """
        Calculate A^T@b using mixed tensor format with SINGLE TENSOR OPERATION.

        Target: Implement einsum 'ijkm,jkm->ij' where:
        - Mixed tensor diffusion patterns: (1000, 3, 480, 800) - shape 'ijkm'
        - Target frame: (480, 800, 3) -> transpose to (3, 480, 800) - shape 'jkm'
        - Result: (1000, 3) - shape 'ij'

        However, since CUDA kernel only accepts (height, width) input, we need to
        construct the full tensor operation differently or use the existing kernel properly.
        """
        # Step 1: Normalize and transfer target to GPU
        # Target shape: (480, 800, 3) -> (height, width, channels)
        target_normalized = (
            target_frame.astype(np.float32) / 255.0
        )  # Shape: (480, 800, 3)
        target_gpu = cp.asarray(target_normalized)  # Shape: (480, 800, 3)

        logger.debug(f"Target shape after GPU transfer: {target_gpu.shape}")

        # Step 2: Prepare target for einsum operation
        # Need to convert target from (height, width, channels) to (channels, height, width)
        target_chw = target_gpu.transpose(2, 0, 1)  # Shape: (3, 480, 800)
        logger.debug(f"Target shape after transpose (CHW): {target_chw.shape}")

        # Step 3: TRUE SINGLE TENSOR OPERATION using einsum
        # Convert sparse mixed tensor to dense format for proper einsum
        # Mixed tensor storage: (3, 1000, 96, 96) -> logical (1000, 3, 480, 800)

        # Build dense diffusion tensor: (1000, 3, 480, 800)
        logger.debug("Converting mixed tensor to dense format for einsum...")
        diffusion_dense = cp.zeros(
            (
                self._actual_led_count,
                3,
                self._mixed_tensor.height,
                self._mixed_tensor.width,
            ),
            dtype=cp.float32,
        )  # Shape: (1000, 3, 480, 800)

        # Fill dense tensor from mixed tensor sparse blocks
        for led_idx in range(self._actual_led_count):
            for channel_idx in range(3):
                if self._mixed_tensor.blocks_set[
                    channel_idx, led_idx
                ]:  # Storage: (channels, batch)
                    # Get block position and data
                    top_row = int(
                        self._mixed_tensor.block_positions[channel_idx, led_idx, 0]
                    )
                    top_col = int(
                        self._mixed_tensor.block_positions[channel_idx, led_idx, 1]
                    )
                    block_data = self._mixed_tensor.sparse_values[
                        channel_idx, led_idx
                    ]  # Shape: (96, 96)

                    # Place block in dense tensor
                    diffusion_dense[
                        led_idx,
                        channel_idx,
                        top_row : top_row + self._mixed_tensor.block_size,
                        top_col : top_col + self._mixed_tensor.block_size,
                    ] = block_data

        logger.debug(f"Dense diffusion tensor shape: {diffusion_dense.shape}")
        logger.debug(f"Target CHW shape: {target_chw.shape}")

        # Step 4: SINGLE EINSUM OPERATION
        # einsum('ijkm,jkm->ij', diffusion_dense, target_chw)
        # Where: i=LEDs (1000), j=channels (3), k=height (480), m=width (800)
        result = cp.einsum(
            "ijkm,jkm->ij", diffusion_dense, target_chw
        )  # Shape: (1000, 3)

        logger.debug(f"Einsum result shape: {result.shape}, sample: {result[:3].get()}")

        # Step 4: Convert to output format
        self._ATb_gpu[:] = result  # Shape: (1000, 3)

        return self._ATb_gpu

    def compare_atb_methods(self, target_frame: np.ndarray) -> Dict[str, any]:
        """
        Compare A^T@b calculation between CSC and mixed tensor methods element by element.

        Returns detailed comparison for debugging divergences.
        """
        logger.info("=== Comparing A^T@b methods ===")

        # Calculate using CSC method
        logger.info("Computing A^T@b with CSC method...")
        atb_csc = self._calculate_ATb_csc_format(target_frame)

        # Calculate using mixed tensor method
        logger.info("Computing A^T@b with mixed tensor method...")
        atb_mixed = self._calculate_ATb_mixed_tensor(target_frame)

        # Element-wise comparison
        diff = cp.abs(atb_csc - atb_mixed)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        # Find elements with largest differences
        diff_cpu = diff.get()
        atb_csc_cpu = atb_csc.get()
        atb_mixed_cpu = atb_mixed.get()

        # Get top 10 differences
        flat_diff = diff_cpu.flatten()
        flat_indices = np.argsort(flat_diff)[-10:][::-1]

        top_differences = []
        for flat_idx in flat_indices:
            led_idx = flat_idx // 3
            channel_idx = flat_idx % 3
            csc_val = atb_csc_cpu[led_idx, channel_idx]
            mixed_val = atb_mixed_cpu[led_idx, channel_idx]
            diff_val = diff_cpu[led_idx, channel_idx]

            top_differences.append(
                {
                    "led": led_idx,
                    "channel": channel_idx,
                    "csc": csc_val,
                    "mixed": mixed_val,
                    "diff": diff_val,
                    "rel_diff": diff_val / max(abs(csc_val), abs(mixed_val), 1e-8),
                }
            )

        logger.info(f"Max absolute difference: {max_diff:.6f}")
        logger.info(f"Mean absolute difference: {mean_diff:.6f}")
        logger.info(f"Close (atol=1e-3): {cp.allclose(atb_csc, atb_mixed, atol=1e-3)}")
        logger.info(f"Close (atol=1e-2): {cp.allclose(atb_csc, atb_mixed, atol=1e-2)}")

        logger.info("Top 5 differences:")
        for i, diff_info in enumerate(top_differences[:5]):
            logger.info(
                f"  {i+1}. LED {diff_info['led']}, channel {diff_info['channel']}: "
                f"CSC={diff_info['csc']:.6f}, Mixed={diff_info['mixed']:.6f}, "
                f"diff={diff_info['diff']:.6f} ({diff_info['rel_diff']*100:.1f}%)"
            )

        return {
            "atb_csc": atb_csc,
            "atb_mixed": atb_mixed,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "top_differences": top_differences,
            "close_1e3": cp.allclose(atb_csc, atb_mixed, atol=1e-3),
            "close_1e2": cp.allclose(atb_csc, atb_mixed, atol=1e-2),
        }

    def _solve_dense_gradient_descent(
        self, ATb: cp.ndarray, max_iterations: Optional[int]
    ) -> Tuple[cp.ndarray, int]:
        """
        Solve using dense gradient descent with precomputed A^T*A using optimized einsum.

        This is the core innovation: all operations in the loop are dense tensors
        using einsum for optimal GPU utilization.

        Args:
            ATb: A^T * b vector (led_count, 3)
            max_iterations: Maximum iterations

        Returns:
            Tuple of (LED values (led_count, 3) on GPU, iterations_completed)
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

        for iteration in range(max_iters):
            # KEY OPTIMIZATION: Use einsum for parallel ATA @ x computation
            # ATA: (led_count, led_count, 3), x: (led_count, 3) -> (led_count, 3)
            # einsum 'ijk,jk->ik' computes all 3 channels in parallel
            w["ATA_x"][:] = cp.einsum("ijk,jk->ik", self._ATA_gpu, x)
            w["gradient"][:] = w["ATA_x"] - ATb

            # KEY STEP 4: Compute step size using optimized dense operations
            step_size = self._compute_dense_step_size(w["gradient"])

            # Gradient descent step with projection to [0, 1]
            w["x_new"][:] = cp.clip(x - step_size * w["gradient"], 0, 1)

            # Check convergence
            delta = cp.linalg.norm(w["x_new"] - x)
            if delta < self.convergence_threshold:
                logger.debug(
                    f"Converged after {iteration+1} iterations, delta: {delta:.6f}"
                )
                break

            # Update (copy values to avoid reference swapping issues)
            x[:] = w["x_new"]

        # Ensure we return the current x
        self._led_values_gpu[:] = x

        return x, iteration + 1

    def _compute_dense_step_size(self, gradient: cp.ndarray) -> float:
        """
        Compute step size using single einsum for g^T @ ATA @ g: (g^T @ g) / (g^T @ A^T*A @ g).

        Args:
            gradient: Gradient vector (led_count, 3)

        Returns:
            Step size (float)
        """
        # g^T @ g (sum over all elements)
        g_dot_g = cp.sum(gradient * gradient)

        # ULTRA-OPTIMIZED: Compute g^T @ ATA @ g for all channels in single einsum
        # ATA: (led_count, led_count, 3), gradient: (led_count, 3)
        # Computes sum over k of: g[:,k]^T @ ATA[:,:,k] @ g[:,k]
        g_dot_ATA_g = cp.einsum("ik,ijk,jk->", gradient, self._ATA_gpu, gradient)

        if g_dot_ATA_g > 0:
            return float(self.step_size_scaling * g_dot_g / g_dot_ATA_g)
        else:
            return 0.01  # Fallback step size

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
        stats = {
            "optimizer_type": "dense_precomputed_ata",
            "device": str(self.device_info["device"]),
            "matrix_loaded": self._matrix_loaded,
            "optimization_count": self._optimization_count,
            "total_optimization_time": 0.0,  # Timing removed
            "average_optimization_time": 0.0,  # Timing removed
            "estimated_fps": 0.0,  # Timing removed
            "led_count": self._actual_led_count,
            "frame_dimensions": (FRAME_WIDTH, FRAME_HEIGHT),
        }

        if self._matrix_loaded:
            stats.update(
                {
                    "ata_tensor_shape": self._ATA_gpu.shape,
                    "ata_memory_mb": self._ATA_gpu.nbytes / (1024 * 1024),
                    "approach_description": "Precomputed A^T*A dense tensors with einsum",
                    "flop_analysis": {
                        "flops_per_iteration": 0,  # FLOP counting removed
                        "total_flops_computed": 0,  # FLOP counting removed
                        "average_gflops_per_frame": 0.0,  # FLOP counting removed
                        "average_gflops_per_second": 0.0,  # FLOP counting removed
                    },
                }
            )

        return stats
