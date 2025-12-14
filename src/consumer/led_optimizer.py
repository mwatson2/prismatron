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

from ..const import FRAME_HEIGHT, FRAME_WIDTH
from ..utils.batch_frame_optimizer import BatchFrameOptimizationResult, optimize_batch_frames_led_values
from ..utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from ..utils.dense_ata_matrix import DenseATAMatrix
from ..utils.diagonal_ata_matrix import DiagonalATAMatrix
from ..utils.frame_optimizer import (
    load_ata_inverse_from_pattern,
    optimize_frame_led_values,
)
from ..utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from ..utils.performance_timing import PerformanceTiming
from ..utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from ..utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix

logger = logging.getLogger(__name__)

USE_DENSE_ATA = False  # Use DIA A^T*A matrices by default


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

    # Class-level type annotations for Optional attributes
    _dense_ata_matrix: Optional[Any]
    _symmetric_ata_matrix: Optional[Any]
    _batch_symmetric_ata_matrix: Optional[Any]
    _diagonal_ata_matrix: Optional[Any]
    _ATA_inverse_gpu: Optional[Any]
    _ATA_inverse_cpu: Optional[np.ndarray]
    _A_r_csc_gpu: Optional[Any]
    _A_g_csc_gpu: Optional[Any]
    _A_b_csc_gpu: Optional[Any]
    _A_combined_csc_gpu: Optional[Any]
    _target_rgb_buffer: Optional[np.ndarray]
    _ATb_gpu: Optional[Any]
    _led_values_gpu: Optional[Any]
    _gpu_workspace: Optional[Any]

    def __init__(
        self,
        diffusion_patterns_path: Optional[str] = None,
        use_mixed_tensor: bool = False,
        enable_performance_timing: bool = True,
        enable_batch_mode: bool = False,
        optimization_iterations: int = 10,
    ):
        """
        Initialize LED optimizer using standardized frame optimizer.

        Args:
            diffusion_patterns_path: Path to diffusion pattern files with mixed tensor and DIA matrix
            use_mixed_tensor: Deprecated parameter - always uses mixed tensor format
            enable_performance_timing: If True, enable detailed performance timing
            enable_batch_mode: Whether to enable batch processing mode
            optimization_iterations: Number of optimization iterations for LED calculations (0-20, 0 = pseudo inverse only)
        """
        if diffusion_patterns_path is None:
            raise ValueError("diffusion_patterns_path must be provided - no default fallback")
        self.diffusion_patterns_path = diffusion_patterns_path
        self.enable_batch_mode = enable_batch_mode
        self.use_mixed_tensor = use_mixed_tensor

        # Performance timing
        self.timing = PerformanceTiming("LEDOptimizer", enable_gpu_timing=True) if enable_performance_timing else None

        # Optimization parameters for gradient descent
        self.max_iterations = optimization_iterations
        self.convergence_threshold = 1e-3
        self.step_size_scaling = 0.9

        # ATA matrices: Dense format preferred, Symmetric DIA format for efficiency, DIA format for fallback, dense inverse for optimal initialization
        self._dense_ata_matrix = None  # DenseATAMatrix instance for dense ATA operations (preferred)
        self._symmetric_ata_matrix = (
            None  # SymmetricDiagonalATAMatrix instance for efficient ATA operations (preferred over regular DIA)
        )
        self._batch_symmetric_ata_matrix = None  # BatchSymmetricDiagonalATAMatrix for batch operations
        self._diagonal_ata_matrix = None  # DiagonalATAMatrix instance for sparse ATA operations (fallback)
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
        self._actual_led_count = 0  # Will be set from pattern file
        self._pattern_color_space = "srgb"  # Color space of diffusion patterns (srgb or linear)

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

    def _srgb_to_linear(self, srgb_frame: np.ndarray) -> np.ndarray:
        """
        Convert sRGB frame to linear light space.

        Uses the standard sRGB transfer function:
        - For values <= 0.04045: linear = srgb / 12.92
        - For values > 0.04045: linear = ((srgb + 0.055) / 1.055) ^ 2.4

        Args:
            srgb_frame: Frame in sRGB space, shape (H, W, 3), range [0, 255]

        Returns:
            Frame in linear light space, shape (H, W, 3), range [0, 255]
        """
        # Normalize to [0, 1]
        srgb_normalized = srgb_frame.astype(np.float32) / 255.0

        # Apply sRGB inverse gamma correction
        linear_normalized = np.where(
            srgb_normalized <= 0.04045,
            srgb_normalized / 12.92,
            np.power((srgb_normalized + 0.055) / 1.055, 2.4),
        )

        # Scale back to [0, 255] and convert to uint8
        linear_frame = (linear_normalized * 255.0).astype(np.uint8)

        return linear_frame

    def _srgb_to_linear_batch(self, srgb_frames: cp.ndarray) -> cp.ndarray:
        """
        Convert batch of sRGB frames to linear light space (GPU version).

        Args:
            srgb_frames: Frames in sRGB space on GPU, shape (batch, 3, H, W) or (batch, H, W, 3), range [0, 255]

        Returns:
            Frames in linear light space on GPU, shape same as input, range [0, 255]
        """
        # Normalize to [0, 1]
        srgb_normalized = srgb_frames.astype(cp.float32) / 255.0

        # Apply sRGB inverse gamma correction
        linear_normalized = cp.where(
            srgb_normalized <= 0.04045,
            srgb_normalized / 12.92,
            cp.power((srgb_normalized + 0.055) / 1.055, 2.4),
        )

        # Scale back to [0, 255]
        linear_frames = (linear_normalized * 255.0).astype(cp.uint8)

        return linear_frames

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
            if self._dense_ata_matrix is not None:
                logger.info("ATA format: Dense (preferred)")
            elif self._symmetric_ata_matrix is not None:
                logger.info("ATA format: Symmetric DIA (efficient)")
            elif self._diagonal_ata_matrix is not None:
                logger.info("ATA format: DIA sparse (fallback)")
            else:
                logger.info("ATA format: None")
            logger.info(f"Device: {self.device_info['device']}")
            return True

        except Exception as e:
            logger.error(f"LED optimizer initialization failed: {e}")
            return False

    def _load_precomputed_ata_matrices(self, data: np.lib.npyio.NpzFile) -> bool:
        """
        Load precomputed A^T@A matrices from the pattern file if available.
        Prioritizes DenseATAMatrix over DiagonalATAMatrix when both formats are present.

        Args:
            data: Loaded NPZ file data

        Returns:
            True if precomputed matrices were loaded successfully, False otherwise
        """
        try:
            # Check for dense ATA matrix first (preferred format)
            if USE_DENSE_ATA and "dense_ata_matrix" in data:
                logger.debug("Loading A^T@A matrices from dense_ata_matrix key (preferred format)")
                dense_ata_dict = data["dense_ata_matrix"].item()
                dense_ata_matrix = DenseATAMatrix.from_dict(dense_ata_dict)
                self._dense_ata_matrix = dense_ata_matrix
                logger.info(f"Loaded dense A^T@A matrix: {dense_ata_matrix.led_count} LEDs")
                if dense_ata_matrix.dense_matrices_cpu is not None:
                    logger.info(f"Dense format - shape: {dense_ata_matrix.dense_matrices_cpu.shape}")

                # Calculate dense memory usage
                dense_memory_mb = self._dense_ata_matrix.memory_mb
                logger.info(f"Dense A^T@A memory: {dense_memory_mb:.1f}MB")

            # Check for diagonal ATA matrix (fallback format)
            elif "diagonal_ata_matrix" in data:
                logger.debug("Loading A^T@A matrices from diagonal_ata_matrix key (fallback format)")
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

                # Create symmetric version from the regular DiagonalATAMatrix for better performance
                try:
                    logger.info("Creating SymmetricDiagonalATAMatrix from regular DiagonalATAMatrix...")
                    self._symmetric_ata_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(
                        self._diagonal_ata_matrix
                    )
                    logger.info("Successfully created symmetric ATA matrix")

                    # Load or create batch version for batch processing (only if batch mode enabled)
                    if self.enable_batch_mode:
                        self._load_or_create_batch_symmetric_ata_matrix(data)
                except Exception as e:
                    logger.warning(f"Failed to create symmetric ATA matrix: {e}. Will use regular DIA matrix.")
                    self._symmetric_ata_matrix = None

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

                # Create symmetric version from the regular DiagonalATAMatrix for better performance
                try:
                    logger.info("Creating SymmetricDiagonalATAMatrix from regular DiagonalATAMatrix...")
                    self._symmetric_ata_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(
                        self._diagonal_ata_matrix
                    )
                    logger.info("Successfully created symmetric ATA matrix")

                    # Load or create batch version for batch processing (only if batch mode enabled)
                    if self.enable_batch_mode:
                        self._load_or_create_batch_symmetric_ata_matrix(data)
                except Exception as e:
                    logger.warning(f"Failed to create symmetric ATA matrix: {e}. Will use regular DIA matrix.")
                    self._symmetric_ata_matrix = None

            # Check for symmetric_dia_matrix format (direct symmetric format)
            elif "symmetric_dia_matrix" in data:
                logger.debug("Loading A^T@A matrices from symmetric_dia_matrix key")
                symmetric_dia_dict = data["symmetric_dia_matrix"].item()
                self._symmetric_ata_matrix = SymmetricDiagonalATAMatrix.from_dict(symmetric_dia_dict)
                logger.info(f"Loaded symmetric DIA A^T@A matrix: {self._symmetric_ata_matrix.led_count} LEDs")
                logger.info(
                    f"Symmetric DIA format - upper diagonals: {self._symmetric_ata_matrix.k_upper}, bandwidth: {self._symmetric_ata_matrix.bandwidth}"
                )

                # Calculate memory usage
                if self._symmetric_ata_matrix.dia_data_gpu is not None:
                    symmetric_memory_mb = self._symmetric_ata_matrix.dia_data_gpu.nbytes / (1024 * 1024)
                    logger.info(f"Symmetric DIA A^T@A memory: {symmetric_memory_mb:.1f}MB")

                # Load or create batch version for batch processing (only if batch mode enabled)
                if self.enable_batch_mode:
                    self._load_or_create_batch_symmetric_ata_matrix(data)

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
            if "mixed_tensor" in data and (
                "diffusion_matrix" in data or "dia_matrix" in data or "symmetric_dia_matrix" in data
            ):
                return self._load_matricies_from_file(data)
            else:
                logger.error(f"{patterns_path} is in unsupported legacy format")
                return False

        except Exception as e:
            logger.error(f"Failed to load matrices: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def _load_or_create_batch_symmetric_ata_matrix(self, data: np.lib.npyio.NpzFile) -> None:
        """Load batch symmetric ATA matrix from file or create from regular symmetric ATA matrix for batch operations."""
        if self._symmetric_ata_matrix is None:
            logger.warning("Cannot create batch symmetric ATA matrix: regular symmetric ATA matrix not available")
            return

        try:
            # Check LED count alignment for tensor core operations
            led_count = self._actual_led_count
            if led_count % 16 != 0:
                msg = f"LED count {led_count} not aligned to 16 - batch optimization not available"
                if self.enable_batch_mode:
                    logger.error(f"BATCH MODE INCOMPATIBLE: {msg}")
                else:
                    logger.info(msg)
                return

            # Try to load from file first
            if "batch_symmetric_dia_matrix" in data:
                logger.info("Loading BatchSymmetricDiagonalATAMatrix from file...")
                batch_dict = data["batch_symmetric_dia_matrix"].item()

                # Parse dtypes from saved strings (supports both old and new format)
                def parse_dtype(dtype_str):
                    """Parse dtype from string representation (old: "<class 'numpy.float32'>", new: "float32")"""
                    if "float32" in dtype_str:
                        return cp.float32
                    elif "float16" in dtype_str:
                        return cp.float16
                    elif "float64" in dtype_str:
                        return cp.float64
                    else:
                        logger.warning(f"Unknown dtype string: {dtype_str}, defaulting to float32")
                        return cp.float32

                # Create batch matrix instance and populate from saved data
                self._batch_symmetric_ata_matrix = BatchSymmetricDiagonalATAMatrix(
                    led_count=batch_dict["led_count"],
                    crop_size=batch_dict["crop_size"],
                    batch_size=batch_dict["batch_size"],
                    block_size=batch_dict["block_size"],
                    output_dtype=parse_dtype(batch_dict["output_dtype"]),
                )

                # Load the block data from saved format
                self._batch_symmetric_ata_matrix.block_data_gpu = cp.asarray(
                    batch_dict["block_data_gpu"], dtype=parse_dtype(batch_dict["compute_dtype"])
                )
                self._batch_symmetric_ata_matrix.max_block_diag = batch_dict["max_block_diag"]
                self._batch_symmetric_ata_matrix.block_diag_count = batch_dict["block_diag_count"]
                self._batch_symmetric_ata_matrix.led_blocks = batch_dict["led_blocks"]
                self._batch_symmetric_ata_matrix.padded_led_count = batch_dict["padded_led_count"]
                self._batch_symmetric_ata_matrix.bandwidth = batch_dict["bandwidth"]
                self._batch_symmetric_ata_matrix.sparsity = batch_dict["sparsity"]
                self._batch_symmetric_ata_matrix.nnz = batch_dict["nnz"]
                self._batch_symmetric_ata_matrix.original_k = batch_dict["original_k"]

                logger.info("✅ Successfully loaded batch symmetric ATA matrix from file")
                logger.info(f"  Batch size: {batch_dict['batch_size']} frames")
                logger.info(f"  Block storage shape: {self._batch_symmetric_ata_matrix.block_data_gpu.shape}")
                logger.info(f"  Block diagonal count: {self._batch_symmetric_ata_matrix.block_diag_count}")
                logger.info(
                    f"  GPU memory usage: {self._batch_symmetric_ata_matrix.block_data_gpu.nbytes / (1024*1024):.1f}MB"
                )

            else:
                # Fallback: create from symmetric matrix
                msg = (
                    "Batch matrix not found in file, creating BatchSymmetricDiagonalATAMatrix from symmetric matrix..."
                )
                if self.enable_batch_mode:
                    logger.warning(f"BATCH MODE FALLBACK: {msg}")
                    logger.warning(
                        "Consider running tools/compute_matrices.py to pre-compute the batch matrix for faster startup"
                    )
                else:
                    logger.info(msg)
                self._batch_symmetric_ata_matrix = BatchSymmetricDiagonalATAMatrix.from_symmetric_diagonal_matrix(
                    self._symmetric_ata_matrix, batch_size=8
                )
                logger.info("Successfully created batch symmetric ATA matrix for 8-frame operations")

        except Exception as e:
            logger.error(f"Failed to load/create batch symmetric ATA matrix: {e}")
            if self.enable_batch_mode:
                logger.error("CRITICAL: Batch mode was requested but batch matrix setup failed!")
            self._batch_symmetric_ata_matrix = None

    def _cleanup_unused_matrices(self) -> None:
        """Clean up unused matrices to free GPU/CPU memory based on current mode."""
        try:
            memory_freed = 0

            # Always clean up dense ATA matrix as we use sparse formats
            if self._dense_ata_matrix is not None:
                if (
                    hasattr(self._dense_ata_matrix, "dense_matrices_cpu")
                    and self._dense_ata_matrix.dense_matrices_cpu is not None
                ):
                    memory_freed += self._dense_ata_matrix.dense_matrices_cpu.nbytes / (1024 * 1024)
                    self._dense_ata_matrix.dense_matrices_cpu = None
                if (
                    hasattr(self._dense_ata_matrix, "dense_matrices_gpu")
                    and self._dense_ata_matrix.dense_matrices_gpu is not None
                ):
                    memory_freed += self._dense_ata_matrix.dense_matrices_gpu.nbytes / (1024 * 1024)
                    self._dense_ata_matrix.dense_matrices_gpu = None
                self._dense_ata_matrix = None
                logger.debug("Cleaned up dense ATA matrix")

            # In batch mode, clean up the regular symmetric matrix (keep batch version)
            if self.enable_batch_mode and self._batch_symmetric_ata_matrix is not None:
                if self._symmetric_ata_matrix is not None:
                    if (
                        hasattr(self._symmetric_ata_matrix, "dia_data_gpu")
                        and self._symmetric_ata_matrix.dia_data_gpu is not None
                    ):
                        memory_freed += self._symmetric_ata_matrix.dia_data_gpu.nbytes / (1024 * 1024)
                        self._symmetric_ata_matrix.dia_data_gpu = None
                    if (
                        hasattr(self._symmetric_ata_matrix, "dia_data_cpu")
                        and self._symmetric_ata_matrix.dia_data_cpu is not None
                    ):
                        memory_freed += self._symmetric_ata_matrix.dia_data_cpu.nbytes / (1024 * 1024)
                        self._symmetric_ata_matrix.dia_data_cpu = None
                    self._symmetric_ata_matrix = None
                    logger.debug("Cleaned up regular symmetric ATA matrix (using batch version)")

            # In non-batch mode, clean up the batch matrix (keep regular symmetric version)
            elif (
                not self.enable_batch_mode
                and self._symmetric_ata_matrix is not None
                and self._batch_symmetric_ata_matrix is not None
            ):
                if (
                    hasattr(self._batch_symmetric_ata_matrix, "block_data_gpu")
                    and self._batch_symmetric_ata_matrix.block_data_gpu is not None
                ):
                    memory_freed += self._batch_symmetric_ata_matrix.block_data_gpu.nbytes / (1024 * 1024)
                    self._batch_symmetric_ata_matrix.block_data_gpu = None
                self._batch_symmetric_ata_matrix = None
                logger.debug("Cleaned up batch symmetric ATA matrix (using regular version)")

            # Clean up diagonal ATA matrix if we have symmetric versions
            if (
                self._symmetric_ata_matrix is not None or self._batch_symmetric_ata_matrix is not None
            ) and self._diagonal_ata_matrix is not None:
                if (
                    hasattr(self._diagonal_ata_matrix, "dia_data_cpu")
                    and self._diagonal_ata_matrix.dia_data_cpu is not None
                ):
                    memory_freed += self._diagonal_ata_matrix.dia_data_cpu.nbytes / (1024 * 1024)
                    self._diagonal_ata_matrix.dia_data_cpu = None
                if (
                    hasattr(self._diagonal_ata_matrix, "dia_data_gpu")
                    and self._diagonal_ata_matrix.dia_data_gpu is not None
                ):
                    memory_freed += self._diagonal_ata_matrix.dia_data_gpu.nbytes / (1024 * 1024)
                    self._diagonal_ata_matrix.dia_data_gpu = None
                self._diagonal_ata_matrix = None
                logger.debug("Cleaned up regular diagonal ATA matrix (using symmetric version)")

            if memory_freed > 0:
                logger.info(f"Freed {memory_freed:.1f}MB by cleaning up unused matrices")

        except Exception as e:
            logger.warning(f"Error during matrix cleanup: {e}")

    def _load_matricies_from_file(self, data: np.lib.npyio.NpzFile) -> bool:
        """Load matrices from new nested format using utility classes."""
        logger.info("Detected new nested format with utility classes")

        # Load LEDDiffusionCSCMatrix - try diffusion_matrix first, then dia_matrix, then symmetric_dia_matrix
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
        elif "symmetric_dia_matrix" in data:
            logger.info("Loading from symmetric DIA matrix format - using direct frame optimizer approach...")
            # Since we're using the standardized frame optimizer, we don't need the CSC wrapper
            # We'll set _diffusion_matrix to None and skip CSC setup
            self._diffusion_matrix = None
            logger.info("Symmetric DIA matrix format detected - will use frame optimizer directly")
        else:
            logger.error("No diffusion matrix found in data")
            return False

        # Load metadata from diffusion matrix or pattern data
        led_spatial_mapping = data.get("led_spatial_mapping", {})
        if hasattr(led_spatial_mapping, "item"):
            led_spatial_mapping = led_spatial_mapping.item()
        self._led_spatial_mapping = led_spatial_mapping
        self._led_positions = data.get("led_positions", None)

        # Load color space information from metadata
        metadata: Dict[str, Any] = data.get("metadata", {})
        if hasattr(metadata, "item"):
            metadata = metadata.item()
        self._pattern_color_space = metadata.get("color_space", "srgb")
        logger.info(f"Pattern color space: {self._pattern_color_space}")
        if self._pattern_color_space == "linear":
            logger.info("  Input frames will be converted from sRGB to linear for optimization")

        if self._diffusion_matrix is not None:
            self._actual_led_count = self._diffusion_matrix.led_count
            # Validate matrix dimensions
            expected_pixels = FRAME_HEIGHT * FRAME_WIDTH
            if self._diffusion_matrix.pixels != expected_pixels:
                logger.error(f"Matrix pixel dimension mismatch: {self._diffusion_matrix.pixels} != {expected_pixels}")
                return False
        else:
            # Get LED count from metadata when using DIA format (metadata already loaded above)
            if "led_count" not in metadata:
                logger.error("Pattern file is missing required 'led_count' metadata")
                return False
            self._actual_led_count = metadata["led_count"]
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

        # Validate batch mode requirements
        if self.enable_batch_mode:
            if self._batch_symmetric_ata_matrix is None:
                error_msg = (
                    "BATCH MODE REQUESTED BUT NOT AVAILABLE: "
                    "--batch-mode was specified but no batch symmetric ATA matrix was loaded. "
                    f"Either the pattern file ({self.diffusion_patterns_path}) is missing the batch matrix, "
                    "or the LED count is not compatible with batch operations (must be divisible by 16). "
                    "Run tools/compute_matrices.py to add the batch matrix to your pattern file."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                logger.info(
                    f"✅ BATCH MODE ACTIVE: Using {self._batch_symmetric_ata_matrix.batch_size}-frame batch optimization with {self._batch_symmetric_ata_matrix.block_diag_count} block diagonals"
                )

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

        # Clean up unused matrices to free memory
        self._cleanup_unused_matrices()

        self._matrix_loaded = True
        logger.info("Matrix loading completed successfully")
        return True

    def _setup_csc_matrices(self) -> None:
        """Setup CSC matrices for GPU operations using utility class methods."""
        logger.info("Setting up CSC matrices for GPU operations...")

        if self._diffusion_matrix is None:
            raise RuntimeError("Diffusion matrix not loaded - call load_diffusion_matrices first")

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

            # Convert target frame from sRGB to linear if patterns are in linear space
            if self._pattern_color_space == "linear":
                target_frame = self._srgb_to_linear(target_frame)

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

            # Determine which ATA matrix to use - prefer dense, then symmetric, then regular DIA
            if self._dense_ata_matrix is not None:
                ata_matrix = self._dense_ata_matrix
            elif self._symmetric_ata_matrix is not None:
                ata_matrix = self._symmetric_ata_matrix
            elif self._diagonal_ata_matrix is not None:
                ata_matrix = self._diagonal_ata_matrix
            else:
                raise RuntimeError("No ATA matrix available (dense, symmetric, or DIA format)")

            # Convert target frame to GPU (cupy) for frame optimizer
            target_frame_gpu = cp.asarray(target_frame)

            result_frame_opt = optimize_frame_led_values(
                target_frame=target_frame_gpu,
                at_matrix=self._mixed_tensor,  # Updated parameter name
                ata_matrix=ata_matrix,  # Use dense matrix if available, otherwise DIA
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
                        "ata_format": (
                            "Dense"
                            if self._dense_ata_matrix is not None
                            else (
                                "Symmetric_DIA"
                                if self._symmetric_ata_matrix is not None
                                else "DIA_sparse" if self._diagonal_ata_matrix is not None else "None"
                            )
                        ),
                        "ata_memory_mb": (
                            self._dense_ata_matrix.memory_mb
                            if self._dense_ata_matrix is not None
                            else (
                                self._symmetric_ata_matrix.dia_data_gpu.nbytes / (1024 * 1024)
                                if self._symmetric_ata_matrix and self._symmetric_ata_matrix.dia_data_gpu is not None
                                else (
                                    self._diagonal_ata_matrix.dia_data_cpu.nbytes / (1024 * 1024)
                                    if self._diagonal_ata_matrix and self._diagonal_ata_matrix.dia_data_cpu is not None
                                    else 0
                                )
                            )
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

    def optimize_batch_frames(
        self,
        target_frames: cp.ndarray,
        max_iterations: Optional[int] = None,
        debug: bool = False,
    ) -> BatchFrameOptimizationResult:
        """
        Optimize LED values for a batch of 8 target frames using batch tensor core operations.

        Args:
            target_frames: Target images (8, 3, H, W) or (8, H, W, 3) - GPU cupy array, uint8
            max_iterations: Override default max iterations
            debug: If True, compute error metrics and detailed timing (slower)

        Returns:
            BatchFrameOptimizationResult with LED values (8, 3, led_count) on GPU
        """
        try:
            if not self._matrix_loaded:
                raise RuntimeError("Matrices not loaded")

            if self._batch_symmetric_ata_matrix is None:
                raise RuntimeError("Batch symmetric ATA matrix not available - LED count must be multiple of 16")

            # Convert target frames from sRGB to linear if patterns are in linear space
            if self._pattern_color_space == "linear":
                target_frames = self._srgb_to_linear_batch(target_frames)

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

            # Use batch frame optimizer
            result = optimize_batch_frames_led_values(
                target_frames=target_frames,
                at_matrix=self._mixed_tensor,
                ata_matrix=self._batch_symmetric_ata_matrix,
                ata_inverse=self._ATA_inverse_cpu,
                max_iterations=max_iterations if max_iterations is not None else self.max_iterations,
                convergence_threshold=self.convergence_threshold,
                step_size_scaling=self.step_size_scaling,
                compute_error_metrics=debug,
                debug=debug,
                track_mse_per_iteration=debug,
            )

            logger.debug(f"Batch optimization completed for {target_frames.shape[0]} frames")
            return result

        except Exception as e:
            logger.error(f"Batch frame optimization failed: {e}")

            # Return error result with proper shape
            batch_size = target_frames.shape[0] if len(target_frames.shape) >= 1 else 8
            error_result = BatchFrameOptimizationResult(
                led_values=cp.zeros((batch_size, 3, self._actual_led_count), dtype=cp.uint8),
                error_metrics=[{"mse": float("inf"), "mae": float("inf")}] * batch_size if debug else [],
                iterations=0,
                converged=False,
            )
            return error_result

    def supports_batch_optimization(self) -> bool:
        """
        Check if batch optimization is supported.

        Returns:
            True if batch optimization is available, False otherwise
        """
        return self._matrix_loaded and self._batch_symmetric_ata_matrix is not None and self._actual_led_count % 16 == 0

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

        if self._A_combined_csc_gpu is None:
            raise RuntimeError("CSC matrices not initialized - call _setup_csc_matrices first")
        ATb_combined = self._A_combined_csc_gpu.T @ target_combined_gpu

        self.timing and self.timing.stop("ATb_csc_sparse_matmul")

        # Convert result back to tensor form
        led_count = self._actual_led_count
        ATb_gpu = self._ATb_gpu
        if ATb_gpu is None:
            raise RuntimeError("GPU buffers not initialized - call _initialize_gpu_buffers first")
        ATb_gpu[:, 0] = ATb_combined[:led_count]  # R LEDs
        ATb_gpu[:, 1] = ATb_combined[led_count : 2 * led_count]  # G LEDs
        ATb_gpu[:, 2] = ATb_combined[2 * led_count :]  # B LEDs

        return ATb_gpu

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

        if self._mixed_tensor is None:
            raise RuntimeError("Mixed tensor not initialized - call load_diffusion_matrices first")
        result = self._mixed_tensor.transpose_dot_product_3d(target_gpu)  # Shape: (batch_size, channels)

        self.timing and self.timing.stop("ATb_mixed_tensor_3d_kernel")

        # Store in the ATb buffer
        ATb_gpu = self._ATb_gpu
        if ATb_gpu is None:
            raise RuntimeError("GPU buffers not initialized - call _initialize_gpu_buffers first")
        ATb_gpu[:] = result

        logger.debug(f"ATb result shape: {ATb_gpu.shape}, sample: {ATb_gpu[:3].get()}")

        return ATb_gpu

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

        if x is None:
            raise RuntimeError("LED values GPU buffer not initialized")
        if self._diagonal_ata_matrix is None:
            raise RuntimeError("Diagonal ATA matrix not loaded")

        # Verify x shape
        assert x.shape == (
            self._actual_led_count,
            3,
        ), f"x shape {x.shape} != expected ({self._actual_led_count}, 3)"

        diagonal_ata = self._diagonal_ata_matrix  # Local ref for type narrowing

        for iteration in range(max_iters):
            # KEY OPTIMIZATION: Use DIA format for sparse ATA @ x computation
            # Convert x from (led_count, 3) to (3, led_count) for DIA multiplication
            self.timing and self.timing.start("gradient_calculation", use_gpu_events=True)

            x_transposed = x.T  # Shape: (3, led_count)
            ata_x_transposed = diagonal_ata.multiply_3d(cp.asnumpy(x_transposed))  # Shape: (3, led_count)
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
        if self._diagonal_ata_matrix is None:
            raise RuntimeError("Diagonal ATA matrix not loaded")

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
            if self._dense_ata_matrix is not None:
                ata_info = {
                    "ata_format": "Dense",
                    "ata_shape": self._dense_ata_matrix.dense_matrices_cpu.shape,
                    "ata_memory_mb": self._dense_ata_matrix.memory_mb,
                    "approach_description": "Standardized frame optimizer with mixed tensor and dense ATA matrix",
                }
            elif self._symmetric_ata_matrix is not None:
                symmetric_info = self._symmetric_ata_matrix.get_info()
                ata_info = {
                    "ata_format": "Symmetric_DIA",
                    "ata_bandwidth": symmetric_info.get("bandwidth", 0),
                    "ata_k_upper": symmetric_info.get("k_upper", 0),
                    "ata_original_k": symmetric_info.get("original_k", 0),
                    "ata_memory_reduction": symmetric_info.get("memory_reduction", "Unknown"),
                    "ata_memory_mb": (
                        self._symmetric_ata_matrix.dia_data_gpu.nbytes / (1024 * 1024)
                        if self._symmetric_ata_matrix.dia_data_gpu is not None
                        else 0
                    ),
                    "approach_description": "Standardized frame optimizer with mixed tensor and symmetric DIA matrix",
                }
            elif self._diagonal_ata_matrix is not None:
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
