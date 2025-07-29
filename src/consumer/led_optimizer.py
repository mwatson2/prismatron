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
from ..utils.diagonal_ata_matrix import DiagonalATAMatrix
from ..utils.frame_optimizer import (
    load_ata_inverse_from_pattern,
    optimize_frame_led_values,
)
from ..utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from ..utils.performance_timing import PerformanceTiming
from ..utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
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


class LEDOptimizer:
    """
    LED optimization engine using standardized frame optimizer with modern tensor formats.

    This class serves as a wrapper around the standardized frame_optimizer module,
    providing compatibility with the existing API while using the latest optimization
    techniques including mixed tensor A^T, DIA matrix A^T*A, and dense A^T*A inverse.

    Key components:
    1. SingleBlockMixedSparseTensor for A^T @ b calculation (uint8 or fp32)
    2. DiagonalATAMatrix for A^T*A operations (DIA format with FP16 storage)
    3. Dense A^T*A inverse for optimal initialization
    4. Standardized optimize_frame_led_values function for optimization
    """

    def __init__(
        self,
        diffusion_patterns_path: Optional[str] = None,
        use_mixed_tensor: bool = False,
        enable_performance_timing: bool = True,
    ):
        """
        Initialize LED optimizer using standardized frame optimizer.

        Args:
            diffusion_patterns_path: Path to diffusion pattern files with mixed tensor and DIA matrix
            use_mixed_tensor: Deprecated parameter - always uses mixed tensor format
            enable_performance_timing: If True, enable detailed performance timing
        """
        if diffusion_patterns_path is None:
            raise ValueError("diffusion_patterns_path must be provided - no default fallback")
        self.diffusion_patterns_path = diffusion_patterns_path
        self.use_mixed_tensor = use_mixed_tensor

        # Performance timing
        self.timing = PerformanceTiming("LEDOptimizer", enable_gpu_timing=True) if enable_performance_timing else None

        # Optimization parameters for gradient descent
        self.max_iterations = 10
        self.convergence_threshold = 1e-3
        self.step_size_scaling = 0.9

        # ATA matrices: DIA format for efficiency, dense inverse for optimal initialization
        self._diagonal_ata_matrix = None  # DiagonalATAMatrix instance for sparse ATA operations
        self._ATA_inverse_gpu = None  # Shape: (3, led_count, led_count) - inverse on GPU
        self._ATA_inverse_cpu = None  # Shape: (3, led_count, led_count) - inverse on CPU
        self._has_ata_inverse = False  # Whether ATA inverse matrices are available

        # Sparse matrices for A^T*b calculation (kept sparse for efficiency)
        self._A_r_csc_gpu = None  # Red channel sparse matrix (pixels, leds)
        self._A_g_csc_gpu = None  # Green channel sparse matrix
        self._A_b_csc_gpu = None  # Blue channel sparse matrix
        self._A_combined_csc_gpu = None  # Combined block diagonal matrix for faster A^T*b

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

        # Periodic logging for pipeline debugging
        self._last_log_time = 0.0
        self._log_interval = 2.0  # Log every 2 seconds
        self._frames_with_content = 0  # Input frames with non-zero content
        self._optimizations_with_result = 0  # Optimizations that produced non-zero LED values

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
            if not self._load():
                logger.error("Failed to load and precompute dense matrices")
                return False

            # Initialize workspace arrays
            self._initialize_workspace()

            logger.info("LED optimizer initialized successfully")
            logger.info(f"LED count: {self._actual_led_count}")
            logger.info(f"ATA format: {'DIA sparse' if self._diagonal_ata_matrix else 'None'}")
            logger.info(f"Device: {self.device_info['device']}")
            return True

        except Exception as e:
            logger.error(f"LED optimizer initialization failed: {e}")
            return False

    def _load_precomputed_ata_matrices(self, data: np.lib.npyio.NpzFile) -> bool:
        """
        Load precomputed A^T@A matrices from the pattern file if available.
        Uses DiagonalATAMatrix for sparse ATA storage and loads dense inverse if available.

        Args:
            data: Loaded NPZ file data

        Returns:
            True if precomputed matrices were loaded successfully, False otherwise
        """
        try:
            # Check for diagonal ATA matrix first (preferred format)
            if "diagonal_ata_matrix" in data:
                logger.debug("Loading A^T@A matrices from diagonal_ata_matrix key")
                diagonal_ata_dict = data["diagonal_ata_matrix"].item()
                # Create with FP32 storage and computation for stability with real hardware patterns
                self._diagonal_ata_matrix = DiagonalATAMatrix.from_dict(diagonal_ata_dict)
                logger.info(f"Loaded diagonal A^T@A matrix: {self._diagonal_ata_matrix.led_count} LEDs")
                logger.info(
                    f"DIA format - bandwidth: {self._diagonal_ata_matrix.bandwidth}, k: {self._diagonal_ata_matrix.k}"
                )

                # Calculate DIA memory usage
                if self._diagonal_ata_matrix.dia_data_cpu is not None:
                    dia_memory_mb = self._diagonal_ata_matrix.dia_data_cpu.nbytes / (1024 * 1024)
                    logger.info(f"DIA A^T@A memory: {dia_memory_mb:.1f}MB")

            # Check for new dia_matrix format (from pattern generation)
            elif "dia_matrix" in data:
                logger.debug("Loading A^T@A matrices from dia_matrix key")
                dia_dict = data["dia_matrix"].item()
                # Create with FP32 storage and computation for stability with real hardware patterns
                self._diagonal_ata_matrix = DiagonalATAMatrix.from_dict(dia_dict)
                logger.info(f"Loaded DIA A^T@A matrix: {self._diagonal_ata_matrix.led_count} LEDs")
                logger.info(
                    f"DIA format - bandwidth: {self._diagonal_ata_matrix.bandwidth}, k: {self._diagonal_ata_matrix.k}"
                )

                # Calculate DIA memory usage
                if self._diagonal_ata_matrix.dia_data_cpu is not None:
                    dia_memory_mb = self._diagonal_ata_matrix.dia_data_cpu.nbytes / (1024 * 1024)
                    logger.info(f"DIA A^T@A memory: {dia_memory_mb:.1f}MB")

            # Check for dense ATA format (fallback for compatibility)
            elif "dense_ata" in data:
                logger.warning("Loading A^T@A matrices from legacy dense format - consider regenerating patterns")
                return False  # Force regeneration with DIA format
            else:
                logger.debug("No precomputed A^T@A matrices found in pattern file")
                return False

            # Load ATA inverse matrices if available
            inverse_loaded = False

            # Check for new standalone ATA inverse format (preferred)
            if "ata_inverse" in data:
                logger.debug("Loading ATA inverse from standalone format")
                ata_inverse = data["ata_inverse"]

                if ata_inverse.shape == (
                    3,
                    self._actual_led_count,
                    self._actual_led_count,
                ):
                    self._ATA_inverse_cpu = ata_inverse.astype(np.float32)
                    self._has_ata_inverse = True
                    inverse_loaded = True

                    ata_inv_memory_mb = self._ATA_inverse_cpu.nbytes / (1024 * 1024)
                    logger.info(f"Loaded A^T@A inverse matrices: {self._ATA_inverse_cpu.shape}")
                    logger.info(f"A^T@A inverse memory: {ata_inv_memory_mb:.1f}MB")
                else:
                    expected_shape = (3, self._actual_led_count, self._actual_led_count)
                    logger.warning(f"ATA inverse shape {ata_inverse.shape} != {expected_shape}")

            # Fallback to legacy dense_ata format
            elif "dense_ata" in data:
                logger.debug("Loading ATA inverse from legacy dense_ata format")
                dense_ata_dict = data["dense_ata"].item()
                dense_ata_inverse_matrices = dense_ata_dict.get("dense_ata_inverse_matrices", None)
                successful_inversions = dense_ata_dict.get("successful_inversions", 0)
                avg_condition_number = dense_ata_dict.get("avg_condition_number", float("inf"))

                if dense_ata_inverse_matrices is not None:
                    # Validate dimensions
                    if (
                        dense_ata_inverse_matrices.shape[0] != self._actual_led_count
                        or dense_ata_inverse_matrices.shape[1] != self._actual_led_count
                    ):
                        logger.warning(f"ATA inverse dimensions mismatch: {dense_ata_inverse_matrices.shape}")
                    else:
                        dense_ata_inverse_matrices = dense_ata_inverse_matrices.astype(np.float32)

                        # Convert to channel-first format (3, led_count, led_count) if needed
                        if dense_ata_inverse_matrices.shape == (
                            self._actual_led_count,
                            self._actual_led_count,
                            3,
                        ):
                            logger.debug(
                                "Converting ATA inverse from (led_count, led_count, 3) to (3, led_count, led_count)"
                            )
                            self._ATA_inverse_cpu = np.transpose(dense_ata_inverse_matrices, (2, 0, 1))
                        elif dense_ata_inverse_matrices.shape == (
                            3,
                            self._actual_led_count,
                            self._actual_led_count,
                        ):
                            logger.debug("ATA inverse already in channel-first format")
                            self._ATA_inverse_cpu = dense_ata_inverse_matrices
                        else:
                            logger.warning(f"Unexpected ATA inverse shape: {dense_ata_inverse_matrices.shape}")
                            self._ATA_inverse_cpu = None

                        if self._ATA_inverse_cpu is not None:
                            self._has_ata_inverse = True
                            inverse_loaded = True

                            ata_inv_memory_mb = self._ATA_inverse_cpu.nbytes / (1024 * 1024)
                            logger.info(f"Loaded A^T@A inverse matrices: {self._ATA_inverse_cpu.shape}")
                            logger.info(f"A^T@A inverse memory: {ata_inv_memory_mb:.1f}MB")
                            logger.info(f"Successful inversions: {successful_inversions}/3")
                            if successful_inversions > 0:
                                logger.info(f"Average condition number: {avg_condition_number:.2e}")

            if not inverse_loaded:
                self._ATA_inverse_cpu = None
                self._has_ata_inverse = False
                logger.info("No A^T@A inverse matrices available - will use default initialization")

            return True

        except Exception as e:
            logger.warning(f"Failed to load precomputed A^T@A matrices: {e}")
            return False

    def _load(self) -> bool:
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
            if "mixed_tensor" in data and ("diffusion_matrix" in data or "dia_matrix" in data):
                return self._load_matricies_from_file(data)
            else:
                logger.error(f"{patterns_path} is in unsupported legacy format")
                return False

        except Exception as e:
            logger.error(f"Failed to load matrices: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def _load_matricies_from_file(self, data: np.lib.npyio.NpzFile) -> bool:
        """Load matrices from new nested format using utility classes."""
        logger.info("Detected new nested format with utility classes")

        # Load LEDDiffusionCSCMatrix - try diffusion_matrix first, then dia_matrix
        if "diffusion_matrix" in data:
            logger.info("Loading LEDDiffusionCSCMatrix from diffusion_matrix...")
            diffusion_dict = data["diffusion_matrix"].item()
            self._diffusion_matrix = LEDDiffusionCSCMatrix.from_dict(diffusion_dict)
        elif "dia_matrix" in data:
            logger.info("Loading from DIA matrix format - using direct frame optimizer approach...")
            # Since we're using the standardized frame optimizer, we don't need the CSC wrapper
            # We'll set _diffusion_matrix to None and skip CSC setup
            self._diffusion_matrix = None
            logger.info("DIA matrix format detected - will use frame optimizer directly")
        else:
            logger.error("No diffusion matrix found in data")
            return False

        # Load metadata from diffusion matrix or pattern data
        self._led_spatial_mapping = data.get("led_spatial_mapping", {})
        if hasattr(self._led_spatial_mapping, "item"):
            self._led_spatial_mapping = self._led_spatial_mapping.item()
        self._led_positions = data.get("led_positions", None)

        if self._diffusion_matrix is not None:
            self._actual_led_count = self._diffusion_matrix.led_count
            # Validate matrix dimensions
            expected_pixels = FRAME_HEIGHT * FRAME_WIDTH
            if self._diffusion_matrix.pixels != expected_pixels:
                logger.error(f"Matrix pixel dimension mismatch: {self._diffusion_matrix.pixels} != {expected_pixels}")
                return False
        else:
            # Get LED count from metadata when using DIA format
            metadata = data.get("metadata", {})
            if hasattr(metadata, "item"):
                metadata = metadata.item()
            self._actual_led_count = metadata.get("led_count", LED_COUNT)
            logger.info(f"Using LED count from metadata: {self._actual_led_count}")

        # Load precomputed A^T*A matrices (DIA format)
        if not self._load_precomputed_ata_matrices(data):
            logger.error("Precomputed A^T*A matrices required but not found")
            return False

        # Transfer ATA inverse to GPU if available
        if self._has_ata_inverse:
            logger.info("Transferring dense ATA inverse tensor to GPU...")
            self._ATA_inverse_gpu = cp.asarray(self._ATA_inverse_cpu)
        else:
            self._ATA_inverse_gpu = None

        # Load mixed tensor
        logger.info("Loading SingleBlockMixedSparseTensor...")
        mixed_dict = data["mixed_tensor"].item()
        self._mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_dict)
        logger.info("Loaded mixed tensor format for A^T@b calculation")

        # Load CSC format for A^T@b calculation (only if using legacy diffusion matrix)
        if self._diffusion_matrix is not None:
            logger.info("Setting up CSC matrices for A^T@b calculation...")
            self._setup_csc_matrices()
        else:
            logger.info("Skipping CSC matrix setup - using frame optimizer directly")

        # Log memory usage summary
        if self._diagonal_ata_matrix and self._diagonal_ata_matrix.dia_data_cpu is not None:
            dia_memory_mb = self._diagonal_ata_matrix.dia_data_cpu.nbytes / (1024 * 1024)
            logger.info(f"Total A^T@A memory (DIA format): {dia_memory_mb:.1f}MB")

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

        logger.info("CSC matrices transferred to GPU successfully")

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

        logger.info("Initializing workspace arrays...")

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

        workspace_mb = sum(arr.nbytes for arr in self._gpu_workspace.values()) / (1024 * 1024)
        logger.info(f"Workspace memory: {workspace_mb:.1f}MB")

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

    def optimize_frame(
        self,
        target_frame: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
        debug: bool = False,
    ) -> OptimizationResult:
        """
        Optimize LED values using the standardized frame optimizer API.

        Args:
            target_frame: Target image (height, width, 3) in range [0, 255]
            initial_values: Initial LED values (led_count, 3), if None uses ATA inverse initialization
            max_iterations: Override default max iterations
            debug: If True, compute error metrics and detailed timing (slower)

        Returns:
            OptimizationResult with LED values and optional debug metrics
        """

        try:
            if not self._matrix_loaded:
                raise RuntimeError("Dense matrices not loaded")

            # Track input for logging
            self._optimization_count += 1
            if target_frame.max() > 0:
                self._frames_with_content += 1

            # Validate input
            if target_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise ValueError(f"Target frame shape {target_frame.shape} != {(FRAME_HEIGHT, FRAME_WIDTH, 3)}")

            # Load ATA inverse if not already available
            if not self._has_ata_inverse:
                logger.warning("ATA inverse not loaded - attempting to load from pattern file")
                ata_inverse = load_ata_inverse_from_pattern(
                    self.diffusion_patterns_path
                    if self.diffusion_patterns_path.endswith(".npz")
                    else f"{self.diffusion_patterns_path}.npz"
                )
                if ata_inverse is not None:
                    self._ATA_inverse_cpu = ata_inverse
                    self._ATA_inverse_gpu = cp.asarray(ata_inverse)
                    self._has_ata_inverse = True
                    logger.info("Successfully loaded ATA inverse from pattern file")
                else:
                    raise RuntimeError("ATA inverse matrices required but not available")

            # Prepare initial values in correct format for frame optimizer
            # Frame optimizer expects (3, led_count) format, different from our (led_count, 3)
            if initial_values is not None:
                if initial_values.shape == (self._actual_led_count, 3):
                    # Convert from (led_count, 3) to (3, led_count)
                    initial_values_frame_opt = initial_values.T
                elif initial_values.shape == (3, self._actual_led_count):
                    # Already in correct format
                    initial_values_frame_opt = initial_values
                else:
                    raise ValueError(f"Initial values shape {initial_values.shape} not supported")
            else:
                initial_values_frame_opt = None

            # Use standardized frame optimizer with ATA inverse
            from ..utils.frame_optimizer import FrameOptimizationResult

            result_frame_opt = optimize_frame_led_values(
                target_frame=target_frame,
                at_matrix=self._mixed_tensor,  # Updated parameter name
                ata_matrix=self._diagonal_ata_matrix,  # Updated parameter name
                ata_inverse=self._ATA_inverse_cpu,  # Required parameter - use CPU version
                initial_values=initial_values_frame_opt,
                max_iterations=max_iterations if max_iterations is not None else self.max_iterations,
                convergence_threshold=self.convergence_threshold,
                step_size_scaling=self.step_size_scaling,
                compute_error_metrics=debug,
                debug=debug,
                enable_timing=self.timing is not None,
            )

            # Convert result from frame optimizer format to our format
            # Frame optimizer returns (3, led_count), we need (led_count, 3)
            led_values_output = result_frame_opt.led_values.T  # (3, led_count) -> (led_count, 3)

            # Create result compatible with our OptimizationResult format
            result = OptimizationResult(
                led_values=led_values_output,
                error_metrics=result_frame_opt.error_metrics,
                iterations=result_frame_opt.iterations,
                converged=result_frame_opt.converged,
                target_frame=target_frame.copy() if debug else None,
                precomputation_info=(
                    {
                        "ata_format": ("DIA_sparse" if self._diagonal_ata_matrix else "None"),
                        "ata_memory_mb": (
                            self._diagonal_ata_matrix.dia_data_cpu.nbytes / (1024 * 1024)
                            if self._diagonal_ata_matrix and self._diagonal_ata_matrix.dia_data_cpu is not None
                            else 0
                        ),
                        "approach": "standardized_frame_optimizer",
                        "frame_optimizer_timing": result_frame_opt.timing_data,
                    }
                    if debug
                    else None
                ),
                optimization_time=0.0,  # Timing handled by frame optimizer
            )

            # Track successful optimization with non-zero output for logging
            if led_values_output.max() > 0:
                self._optimizations_with_result += 1

            # Periodic logging for pipeline debugging
            import time

            current_time = time.time()
            if current_time - self._last_log_time >= self._log_interval:
                content_ratio = (self._frames_with_content / max(1, self._optimization_count)) * 100
                result_ratio = (self._optimizations_with_result / max(1, self._optimization_count)) * 100

                logger.info(
                    f"LED OPTIMIZER PIPELINE: {self._optimization_count} optimizations, "
                    f"{self._frames_with_content} with input content ({content_ratio:.1f}%), "
                    f"{self._optimizations_with_result} with LED output ({result_ratio:.1f}%)"
                )
                self._last_log_time = current_time

            logger.debug("Optimization completed using standardized frame optimizer API")
            return result

        except Exception as e:
            logger.error(f"Frame optimization failed: {e}")

            # Return error result
            return OptimizationResult(
                led_values=np.zeros((self._actual_led_count, 3), dtype=np.uint8),
                error_metrics=({"mse": float("inf"), "mae": float("inf")} if debug else {}),
                iterations=0,
                converged=False,
            )

    def _calculate_atb(self, target_frame: np.ndarray) -> cp.ndarray:
        """
        DEPRECATED: Calculate A^T * b - now handled by frame optimizer.

        This method is kept for compatibility but should not be used.
        The standardized frame optimizer handles A^T @ b calculation internally.
        """
        logger.warning("_calculate_atb is deprecated - use standardized frame optimizer API")
        return self._calculate_atb_mixed_tensor(target_frame)

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

    def _calculate_atb_csc_format(self, target_frame: np.ndarray) -> cp.ndarray:
        """Calculate A^T@b using CSC sparse format."""
        self.timing and self.timing.start("ATb_data_preparation")

        # Convert frame to flat format
        target_combined_gpu = self._convert_frame_to_flat_format(target_frame)

        self.timing and self.timing.stop("ATb_data_preparation")

        # Actual sparse matrix operation
        self.timing and self.timing.start("ATb_csc_sparse_matmul", use_gpu_events=True)

        ATb_combined = self._A_combined_csc_gpu.T @ target_combined_gpu

        self.timing and self.timing.stop("ATb_csc_sparse_matmul")

        # Convert result back to tensor form
        led_count = self._actual_led_count
        self._ATb_gpu[:, 0] = ATb_combined[:led_count]  # R LEDs
        self._ATb_gpu[:, 1] = ATb_combined[led_count : 2 * led_count]  # G LEDs
        self._ATb_gpu[:, 2] = ATb_combined[2 * led_count :]  # B LEDs

        return self._ATb_gpu

    def _calculate_atb_mixed_tensor(self, target_frame: np.ndarray) -> cp.ndarray:
        """
        Calculate A^T@b using mixed tensor format with 3D CUDA kernel.

        Uses the new transpose_dot_product_3d method to process all channels in one
        optimized CUDA operation. Implements einsum 'ijkl,jkl->ij' efficiently:
        - Mixed tensor: (leds, channels, height, width) - shape 'ijkl'
        - Target frame: (channels, height, width) - shape 'jkl' (planar form)
        - Result: (leds, channels) - shape 'ij'

        Args:
            target_frame: Target image (height, width, 3) in range [0, 255]

        Returns:
            ATb vector (led_count, 3) on GPU
        """
        self.timing and self.timing.start("ATb_data_preparation")

        # Step 1: Normalize target frame and convert to planar form
        target_normalized = target_frame.astype(np.float32) / 255.0  # Shape: (height, width, 3)

        # Convert from HWC to CHW (planar form): (height, width, 3) -> (3, height, width)
        target_planar = target_normalized.transpose(2, 0, 1)  # Shape: (3, height, width)
        target_gpu = cp.asarray(target_planar)  # Shape: (3, height, width)

        logger.debug(f"Target planar shape: {target_gpu.shape}")

        self.timing and self.timing.stop("ATb_data_preparation")

        # Step 2: Use the new 3D transpose_dot_product method
        # This processes all channels in one CUDA kernel operation
        self.timing and self.timing.start("ATb_mixed_tensor_3d_kernel", use_gpu_events=True)

        result = self._mixed_tensor.transpose_dot_product_3d(target_gpu)  # Shape: (batch_size, channels)

        self.timing and self.timing.stop("ATb_mixed_tensor_3d_kernel")

        # Store in the ATb buffer
        self._ATb_gpu[:] = result

        logger.debug(f"ATb result shape: {self._ATb_gpu.shape}, sample: {self._ATb_gpu[:3].get()}")

        return self._ATb_gpu

    def _solve_dense_gradient_descent(self, atb: cp.ndarray, max_iterations: Optional[int]) -> Tuple[cp.ndarray, int]:
        """
        DEPRECATED: Dense gradient descent - now handled by frame optimizer.

        This method is kept for compatibility but should not be used.
        The standardized frame optimizer handles optimization internally.
        """
        logger.warning("_solve_dense_gradient_descent is deprecated - use standardized frame optimizer API")
        max_iters = max_iterations or self.max_iterations

        # Verify input shapes and DIA matrix availability
        if self._diagonal_ata_matrix is None:
            raise RuntimeError("Diagonal ATA matrix not loaded")
        assert atb.shape == (
            self._actual_led_count,
            3,
        ), f"ATb shape {atb.shape} != expected ({self._actual_led_count}, 3)"

        # Get workspace arrays
        w = self._gpu_workspace
        x = self._led_values_gpu  # (led_count, 3)

        # Verify x shape
        assert x.shape == (
            self._actual_led_count,
            3,
        ), f"x shape {x.shape} != expected ({self._actual_led_count}, 3)"

        for iteration in range(max_iters):
            # KEY OPTIMIZATION: Use DIA format for sparse ATA @ x computation
            # Convert x from (led_count, 3) to (3, led_count) for DIA multiplication
            self.timing and self.timing.start("gradient_calculation", use_gpu_events=True)

            x_transposed = x.T  # Shape: (3, led_count)
            ata_x_transposed = self._diagonal_ata_matrix.multiply_3d(cp.asnumpy(x_transposed))  # Shape: (3, led_count)
            w["ATA_x"][:] = cp.asarray(ata_x_transposed).T  # Convert back to (led_count, 3)
            w["gradient"][:] = w["ATA_x"] - atb

            self.timing and self.timing.stop("gradient_calculation")

            # KEY STEP 4: Compute step size using optimized dense operations
            self.timing and self.timing.start("step_size_calculation", use_gpu_events=True)

            step_size = self._compute_dense_step_size(w["gradient"])

            self.timing and self.timing.stop("step_size_calculation")

            # Gradient descent step with projection to [0, 1]
            self.timing and self.timing.start("gradient_step", use_gpu_events=True)

            w["x_new"][:] = cp.clip(x - step_size * w["gradient"], 0, 1)

            self.timing and self.timing.stop("gradient_step")

            # Check convergence
            delta = cp.linalg.norm(w["x_new"] - x)
            if delta < self.convergence_threshold:
                logger.debug(f"Converged after {iteration + 1} iterations, delta: {delta:.6f}")
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

        # Compute g^T @ ATA @ g using DIA format
        # Convert gradient from (led_count, 3) to (3, led_count)
        gradient_transposed = gradient.T  # Shape: (3, led_count)
        g_ata_g_per_channel = self._diagonal_ata_matrix.g_ata_g_3d(cp.asnumpy(gradient_transposed))  # Shape: (3,)
        g_dot_ATA_g = cp.sum(cp.asarray(g_ata_g_per_channel))

        if g_dot_ATA_g > 0:
            return float(self.step_size_scaling * g_dot_g / g_dot_ATA_g)
        else:
            return 0.01  # Fallback step size

    def _compute_error_metrics(self, led_values: cp.ndarray, target_frame: np.ndarray) -> Dict[str, float]:
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
            "optimizer_type": "standardized_frame_optimizer",
            "device": str(self.device_info["device"]),
            "matrix_loaded": self._matrix_loaded,
            "optimization_count": self._optimization_count,
            "total_optimization_time": 0.0,  # Timing removed
            "average_optimization_time": 0.0,  # Timing removed
            "estimated_fps": 0.0,  # Timing removed
            "led_count": self._actual_led_count,
            "frame_dimensions": (FRAME_WIDTH, FRAME_HEIGHT),
            "performance_timing_enabled": self.timing is not None,
        }

        if self._matrix_loaded:
            ata_info = {}
            if self._diagonal_ata_matrix:
                ata_info = {
                    "ata_format": "DIA_sparse",
                    "ata_bandwidth": self._diagonal_ata_matrix.bandwidth,
                    "ata_k_bands": self._diagonal_ata_matrix.k,
                    "ata_memory_mb": (
                        self._diagonal_ata_matrix.dia_data_cpu.nbytes / (1024 * 1024)
                        if self._diagonal_ata_matrix.dia_data_cpu is not None
                        else 0
                    ),
                    "approach_description": "Standardized frame optimizer with mixed tensor and DIA matrix",
                }
            else:
                ata_info = {
                    "ata_format": "None",
                    "approach_description": "No matrices available",
                }

            stats.update(ata_info)
            stats.update(
                {
                    "flop_analysis": {
                        "flops_per_iteration": 0,  # FLOP counting removed
                        "total_flops_computed": 0,  # FLOP counting removed
                        "average_gflops_per_frame": 0.0,  # FLOP counting removed
                        "average_gflops_per_second": 0.0,  # FLOP counting removed
                    },
                }
            )

        # Add performance timing stats if available
        if self.timing:
            timing_stats = self.timing.get_stats()
            stats["performance_timing"] = timing_stats

        return stats

    def log_performance_timing(self, logger_instance: logging.Logger = None) -> None:
        """
        Log performance timing results.

        Args:
            logger_instance: Logger to use, defaults to module logger
        """
        if self.timing:
            log_target = logger_instance or logger
            self.timing.log(log_target, include_percentages=True, sort_by="time")
        else:
            logger.info("Performance timing not enabled")

    def get_timing_data(self) -> Dict[str, Any]:
        """
        Get structured timing data for analysis.

        Returns:
            Dictionary containing timing data, empty if timing disabled
        """
        if self.timing:
            return self.timing.get_timing_data()
        return {}

    def reset_timing(self) -> None:
        """Reset performance timing data."""
        if self.timing:
            self.timing.reset()
