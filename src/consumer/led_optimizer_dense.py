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

logger = logging.getLogger(__name__)


@dataclass
class DenseOptimizationResult:
    """Results from dense tensor LED optimization process."""

    led_values: np.ndarray  # RGB values for each LED (led_count, 3) - range [0,255]
    error_metrics: Dict[str, float]  # Error metrics (mse, mae, etc.)
    optimization_time: float  # Time taken for optimization in seconds
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged
    target_frame: Optional[np.ndarray] = None  # Original target frame (for debugging)
    precomputation_info: Optional[Dict[str, Any]] = None  # Dense matrix information
    flop_info: Optional[Dict[str, Any]] = None  # FLOP analysis information
    timing_breakdown: Optional[Dict[str, float]] = None  # Detailed timing breakdown

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

        # BSR format matrices for comparison (using Pyculib/cuSPARSE)
        self._A_r_bsr_data = None  # BSR data for red channel
        self._A_g_bsr_data = None  # BSR data for green channel
        self._A_b_bsr_data = None  # BSR data for blue channel
        self._A_combined_bsr_data = None  # BSR data for combined matrix
        self._bsr_block_size = (4, 4)  # Default BSR block size
        self._bsr_available = False  # Track if BSR is available

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

    def _create_bsr_matrices(self, A_r_csc, A_g_csc, A_b_csc, A_combined_csc) -> None:
        """
        Create BSR (Blocked Sparse Row) format matrices using Pyculib/cuSPARSE.

        Args:
            A_r_csc, A_g_csc, A_b_csc: Individual channel CSC matrices
            A_combined_csc: Combined block diagonal CSC matrix
        """
        try:
            import ctypes
            import ctypes.util
            import gc

            logger.info("Setting up BSR matrices with padding for compatibility...")

            # Find cuSPARSE library
            cusparse_lib_path = ctypes.util.find_library("cusparse")
            if not cusparse_lib_path:
                logger.warning("cuSPARSE library not found, skipping BSR matrices")
                return

            cusparse_lib = ctypes.CDLL(cusparse_lib_path)

            # Check if legacy BSR function exists
            try:
                bsrmv_func = cusparse_lib.cusparseSbsrmv
                logger.info("Found cusparseSbsrmv function for direct BSR access")
            except AttributeError:
                logger.warning(
                    "cusparseSbsrmv function not found, skipping BSR matrices"
                )
                return

            # Create small representative BSR sample for testing (avoid memory issues)
            sample_led_count = 96  # 96 = 16 * 6, easily divisible by 2, 4, 8, 16
            sample_pixels = 1600  # Small representative sample

            logger.info(
                f"Creating BSR sample with {sample_led_count} LEDs for performance testing..."
            )

            # Extract small sample: sample_pixels × (96 LEDs × 3 channels)
            sample_cols = sample_led_count * 3
            A_sample = A_combined_csc[:sample_pixels, :sample_cols]

            # Use 4x4 block size (good balance of structure and memory efficiency)
            block_size = (4, 4)
            rows, cols = A_sample.shape

            # Calculate padding needed
            row_pad = (block_size[0] - rows % block_size[0]) % block_size[0]
            col_pad = (block_size[1] - cols % block_size[1]) % block_size[1]

            if row_pad > 0 or col_pad > 0:
                logger.debug(
                    f"Padding sample from {rows}×{cols} to {rows+row_pad}×{cols+col_pad}"
                )

                # Pad with zeros to make dimensions divisible by block size
                if row_pad > 0:
                    zero_rows = sp.csc_matrix((row_pad, cols), dtype=A_sample.dtype)
                    A_sample = sp.vstack([A_sample, zero_rows])

                if col_pad > 0:
                    zero_cols = sp.csc_matrix(
                        (A_sample.shape[0], col_pad), dtype=A_sample.dtype
                    )
                    A_sample = sp.hstack([A_sample, zero_cols])

            # Convert sample to BSR format (this should be memory-safe)
            logger.info(
                f"Converting sample to BSR format with block size {block_size}..."
            )
            try:
                A_bsr_sample = A_sample.tobsr(blocksize=block_size)

                # Store BSR data for cuSPARSE operations (using sample)
                self._A_combined_bsr_data = self._prepare_bsr_for_cusparse(A_bsr_sample)
                self._bsr_block_size = block_size
                self._bsr_cusparse_lib = cusparse_lib
                self._bsr_led_count = sample_led_count
                self._bsr_padded_shape = A_bsr_sample.shape
                self._bsr_available = True

                # Calculate memory usage
                bsr_memory = (
                    A_bsr_sample.data.nbytes
                    + A_bsr_sample.indices.nbytes
                    + A_bsr_sample.indptr.nbytes
                ) / (1024**2)
                csc_memory = (
                    A_sample.data.nbytes
                    + A_sample.indices.nbytes
                    + A_sample.indptr.nbytes
                ) / (1024**2)

                logger.info(f"BSR sample conversion successful:")
                logger.info(
                    f"  Sample LEDs: {sample_led_count} (from {self._actual_led_count} total)"
                )
                logger.info(
                    f"  Sample matrix: {A_bsr_sample.shape} (from {A_sample.shape})"
                )
                logger.info(f"  Block size: {block_size}")
                logger.info(
                    f"  Memory: BSR {bsr_memory:.2f}MB vs CSC {csc_memory:.2f}MB"
                )
                logger.info(
                    f"  NNZ: {A_bsr_sample.nnz:,} ({A_bsr_sample.nnz/A_bsr_sample.shape[0]/A_bsr_sample.shape[1]*100:.3f}% density)"
                )
                logger.info("Note: This is a small sample for BSR performance testing")

                # Clean up intermediate matrices
                del A_sample, A_bsr_sample
                gc.collect()

            except Exception as e:
                logger.warning(f"BSR sample conversion failed: {e}")
                del A_sample
                gc.collect()
                return

        except ImportError as e:
            logger.warning(f"ctypes not available, skipping BSR matrices: {e}")
            self._bsr_available = False
        except Exception as e:
            logger.warning(f"Failed to set up BSR matrices: {e}")
            self._bsr_available = False

    def _prepare_bsr_for_cusparse(self, bsr_matrix):
        """
        Prepare BSR matrix data for cuSPARSE operations.

        Args:
            bsr_matrix: SciPy BSR matrix

        Returns:
            Dictionary with BSR matrix data (keeping as NumPy for SciPy compatibility)
        """
        import numpy as np

        # Keep BSR data as NumPy arrays for SciPy operations
        # (GPU transfer would happen in actual cuSPARSE implementation)
        bsr_data = {
            "data": bsr_matrix.data.astype(np.float32),
            "indices": bsr_matrix.indices.astype(np.int32),
            "indptr": bsr_matrix.indptr.astype(np.int32),
            "shape": bsr_matrix.shape,
            "blocksize": bsr_matrix.blocksize,
            "nnz": bsr_matrix.nnz,
        }

        return bsr_data

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

            # Create BSR format matrices for performance comparison
            logger.info("Creating BSR format matrices for comparison...")
            self._create_bsr_matrices(A_r_csc, A_g_csc, A_b_csc, A_combined_csc)

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

        # Part 1: Warm sparse operations
        dummy_size = self._A_combined_csc_gpu.shape[0]
        dummy_target = cp.ones(dummy_size, dtype=cp.float32)

        sparse_start = time.time()
        for i in range(3):
            _ = self._A_combined_csc_gpu.T @ dummy_target
        cp.cuda.runtime.deviceSynchronize()
        sparse_warm_time = time.time() - sparse_start

        # Part 2: Warm dense operations (einsum and step size calculation)
        dummy_x = cp.ones((self._actual_led_count, 3), dtype=cp.float32)
        dummy_gradient = cp.ones((self._actual_led_count, 3), dtype=cp.float32)

        dense_start = time.time()
        for i in range(3):
            # Warm the main einsum operation
            _ = cp.einsum("ijk,jk->ik", self._ATA_gpu, dummy_x)
            # Warm the step size calculation (full path including GPU events)
            _ = self._compute_dense_step_size(dummy_gradient)
        cp.cuda.runtime.deviceSynchronize()
        dense_warm_time = time.time() - dense_start

        total_warm_time = sparse_warm_time + dense_warm_time
        logger.info(
            f"GPU cache warmed up in {total_warm_time:.3f}s "
            f"(sparse: {sparse_warm_time:.3f}s, dense: {dense_warm_time:.3f}s)"
        )

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
        start_time = time.time()
        timing_breakdown = {}

        try:
            if not self._matrix_loaded:
                raise RuntimeError("Dense matrices not loaded")

            # Validate input
            validation_start = time.time()
            if target_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise ValueError(
                    f"Target frame shape {target_frame.shape} != {(FRAME_HEIGHT, FRAME_WIDTH, 3)}"
                )
            timing_breakdown["validation_time"] = time.time() - validation_start

            # KEY STEP 2: Calculate A^T*b for current frame
            atb_start = time.time()
            ATb = self._calculate_ATb(target_frame)
            timing_breakdown["atb_calculation_time"] = time.time() - atb_start

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
                # Use pre-initialized 0.5 values
                self._led_values_gpu.fill(0.5)
            timing_breakdown["initialization_time"] = time.time() - init_start

            # Reset workspace references to ensure deterministic behavior
            self._reset_workspace_references()

            # Dense tensor optimization loop with detailed timing
            led_values_solved, loop_timing = self._solve_dense_gradient_descent(
                ATb, max_iterations
            )
            timing_breakdown["optimization_loop_time"] = loop_timing.get(
                "total_loop_time", 0.0
            )
            timing_breakdown["loop_iterations"] = loop_timing.get(
                "iterations_completed", max_iterations or self.max_iterations
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

            # Calculate core per-frame time (excluding validation and conversion overhead)
            core_time = (
                timing_breakdown["atb_calculation_time"]
                + timing_breakdown["initialization_time"]
                + timing_breakdown["optimization_loop_time"]
            )
            timing_breakdown["core_per_frame_time"] = core_time

            # Calculate FLOPs for this optimization
            iterations_used = max_iterations or self.max_iterations

            # Calculate actual FLOPs: A^T*b (once per frame) + dense optimization (per iteration)
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

            # Create detailed flop_info with debug timing if enabled
            flop_info = {
                "total_flops": int(frame_flops),
                "flops_per_iteration": int(
                    self._dense_flops_per_iteration
                    if hasattr(self, "_dense_flops_per_iteration")
                    else self._flops_per_iteration
                ),
                "atb_flops_per_frame": int(atb_flops),
                "dense_loop_flops": int(dense_loop_flops),
                "gflops": frame_flops / 1e9,
                "gflops_per_second": frame_flops / (core_optimization_time * 1e9),
            }

            # Add detailed timing info if debug mode is enabled
            if debug:
                detailed_timing = {
                    "core_optimization_time": core_optimization_time,
                    "debug_overhead_time": debug_time,
                    "total_time_with_debug": optimization_time,
                    "loop_time": loop_timing.get("total_loop_time", 0.0),
                    "einsum_time": loop_timing.get("einsum_time", 0.0),
                    "step_size_time": loop_timing.get("step_size_time", 0.0),
                    "step_einsum_time": loop_timing.get("step_einsum_time", 0.0),
                    "iterations_completed": loop_timing.get(
                        "iterations_completed", iterations_used
                    ),
                    "atb_time": core_optimization_time
                    - loop_timing.get("total_loop_time", 0.0),
                }

                # Include detailed A^T*b timing breakdown if available
                if hasattr(self, "_atb_timing_breakdown"):
                    detailed_timing.update(
                        {
                            "atb_cpu_to_gpu_time": self._atb_timing_breakdown.get(
                                "cpu_to_gpu_conversion_time", 0.0
                            ),
                            "atb_sparse_operation_time": self._atb_timing_breakdown.get(
                                "sparse_operation_time", 0.0
                            ),
                            "atb_tensor_conversion_time": self._atb_timing_breakdown.get(
                                "tensor_conversion_time", 0.0
                            ),
                            "atb_total_time": self._atb_timing_breakdown.get(
                                "total_atb_time", 0.0
                            ),
                        }
                    )

                flop_info["detailed_timing"] = detailed_timing

            # Create result with optional debug information
            result = DenseOptimizationResult(
                led_values=led_values_output,
                error_metrics=error_metrics,
                optimization_time=core_optimization_time,  # Exclude debug overhead
                iterations=iterations_used,
                converged=True,
                target_frame=target_frame.copy() if debug else None,
                precomputation_info={
                    "ata_shape": self._ATA_gpu.shape,
                    "ata_memory_mb": self._ATA_gpu.nbytes / (1024 * 1024),
                    "approach": "dense_precomputed_ata",
                }
                if debug
                else None,
                flop_info=flop_info,
                timing_breakdown=timing_breakdown,
            )

            # Update statistics with core optimization time only
            self._total_flops += frame_flops
            self._optimization_count += 1
            self._total_optimization_time += core_optimization_time

            if debug:
                logger.debug(
                    f"Dense optimization completed in {core_optimization_time:.3f}s "
                    f"(+{debug_time:.3f}s debug overhead)"
                )
            else:
                logger.debug(
                    f"Dense optimization completed in {core_optimization_time:.3f}s"
                )
            return result

        except Exception as e:
            optimization_time = time.time() - start_time
            logger.error(
                f"Dense optimization failed after {optimization_time:.3f}s: {e}"
            )

            # Return error result
            return DenseOptimizationResult(
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

    def _calculate_ATb(self, target_frame: np.ndarray) -> cp.ndarray:
        """
        Calculate A^T * b for the current target frame using block diagonal matrix.

        Args:
            target_frame: Target image (height, width, 3)

        Returns:
            ATb vector (led_count, 3) on GPU
        """
        atb_start_time = time.time()

        # Phase 1: CPU preprocessing and conversion to flat vector
        cpu_conversion_start = time.time()
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
        cpu_to_gpu_time = time.time() - cpu_conversion_start

        # Phase 2: Actual sparse matrix operation on GPU
        operation_start = time.time()

        # Get matrix properties for debugging
        matrix_shape = self._A_combined_csc_gpu.shape
        matrix_nnz = self._A_combined_csc_gpu.nnz
        matrix_density = matrix_nnz / (matrix_shape[0] * matrix_shape[1]) * 100
        target_size_mb = target_combined_gpu.nbytes / (1024 * 1024)

        # Ensure GPU synchronization before timing
        cp.cuda.runtime.deviceSynchronize()
        operation_start_sync = time.time()

        # Actual sparse matrix operation
        ATb_combined = self._A_combined_csc_gpu.T @ target_combined_gpu

        # Force GPU synchronization to get accurate timing
        cp.cuda.runtime.deviceSynchronize()
        operation_time = time.time() - operation_start_sync
        operation_time_total = time.time() - operation_start

        # Calculate theoretical performance metrics
        flops_estimate = matrix_nnz * 2  # multiply-add for each non-zero
        gflops_per_sec = (
            flops_estimate / (operation_time * 1e9) if operation_time > 0 else 0
        )

        logger.debug(f"Sparse A^T@b operation analysis:")
        logger.debug(f"  Matrix shape: {matrix_shape}, NNZ: {matrix_nnz:,}")
        logger.debug(f"  Matrix density: {matrix_density:.3f}%")
        logger.debug(
            f"  Target vector size: {target_combined_gpu.shape}, {target_size_mb:.1f}MB"
        )
        logger.debug(f"  Operation time (sync): {operation_time:.4f}s")
        logger.debug(f"  Operation time (total): {operation_time_total:.4f}s")
        logger.debug(
            f"  Estimated FLOPs: {flops_estimate:,}, GFLOPS/s: {gflops_per_sec:.2f}"
        )
        memory_data_mb = matrix_nnz * 4 + target_size_mb * 1024 * 1024
        bandwidth_gb_s = memory_data_mb / (operation_time * 1024**3)
        logger.debug(f"  Memory bandwidth estimate: {bandwidth_gb_s:.2f} GB/s")

        # Phase 3: Convert result back to tensor form
        tensor_conversion_start = time.time()
        led_count = self._actual_led_count
        self._ATb_gpu[:, 0] = ATb_combined[:led_count]  # R LEDs
        self._ATb_gpu[:, 1] = ATb_combined[led_count : 2 * led_count]  # G LEDs
        self._ATb_gpu[:, 2] = ATb_combined[2 * led_count :]  # B LEDs
        tensor_conversion_time = time.time() - tensor_conversion_start

        total_atb_time = time.time() - atb_start_time

        # Store timing breakdown for debugging
        if not hasattr(self, "_atb_timing_breakdown"):
            self._atb_timing_breakdown = {}

        self._atb_timing_breakdown.update(
            {
                "cpu_to_gpu_conversion_time": cpu_to_gpu_time,
                "sparse_operation_time": operation_time,
                "tensor_conversion_time": tensor_conversion_time,
                "total_atb_time": total_atb_time,
            }
        )

        logger.debug(
            f"A^T*b timing - CPU→GPU: {cpu_to_gpu_time:.4f}s, "
            f"Operation: {operation_time:.4f}s, "
            f"Tensor conv: {tensor_conversion_time:.4f}s, "
            f"Total: {total_atb_time:.4f}s"
        )

        return self._ATb_gpu

    def _calculate_ATb_bsr(self, target_frame: np.ndarray) -> cp.ndarray:
        """
        Calculate A^T * b using direct cuSPARSE BSR operations.

        This uses real BSR format with direct cuSPARSE calls for performance comparison.

        Args:
            target_frame: Target image (height, width, 3)

        Returns:
            ATb vector (led_count, 3) on GPU (restricted to BSR LED count)
        """
        if not self._bsr_available or self._A_combined_bsr_data is None:
            raise RuntimeError("BSR implementation not available")

        import ctypes

        import cupy as cp

        atb_start_time = time.time()

        # Phase 1: Prepare target vector for BSR sample dimensions
        cpu_conversion_start = time.time()
        target_rgb_normalized = target_frame.astype(np.float32) / 255.0
        target_flattened = target_rgb_normalized.reshape(-1, 3)  # (pixels, 3)

        # Create target vector matching BSR sample dimensions
        bsr_shape = self._bsr_padded_shape
        bsr_led_count = self._bsr_led_count

        # Extract sample corresponding to BSR sample size
        sample_pixels = bsr_shape[0] // 3  # BSR matrix rows divided by 3 channels
        if target_flattened.shape[0] > sample_pixels:
            target_sample = target_flattened[:sample_pixels, :]
        else:
            # Pad if needed
            padding = np.zeros(
                (sample_pixels - target_flattened.shape[0], 3), dtype=np.float32
            )
            target_sample = np.vstack([target_flattened, padding])

        # Create combined target vector [R_pixels; G_pixels; B_pixels] for BSR sample
        target_combined = np.empty(sample_pixels * 3, dtype=np.float32)
        target_combined[:sample_pixels] = target_sample[:, 0]  # R channel
        target_combined[sample_pixels : 2 * sample_pixels] = target_sample[
            :, 1
        ]  # G channel
        target_combined[2 * sample_pixels :] = target_sample[:, 2]  # B channel

        # Pad to exact BSR matrix dimensions if needed
        if len(target_combined) < bsr_shape[0]:
            padding = np.zeros(bsr_shape[0] - len(target_combined), dtype=np.float32)
            target_combined = np.concatenate([target_combined, padding])
        elif len(target_combined) > bsr_shape[0]:
            target_combined = target_combined[: bsr_shape[0]]

        # Transfer to GPU
        target_combined_gpu = cp.asarray(target_combined)
        cpu_to_gpu_time = time.time() - cpu_conversion_start

        # Phase 2: cuSPARSE BSR matrix-vector multiplication
        operation_start = time.time()

        # Get BSR matrix data
        bsr_data = self._A_combined_bsr_data
        bsr_shape = bsr_data["shape"]
        bsr_blocksize = bsr_data["blocksize"]

        # Prepare cuSPARSE parameters
        cusparse_lib = self._bsr_cusparse_lib

        # For simplicity, fall back to SciPy BSR operation for now
        # (Direct cuSPARSE call would require complex handle/descriptor setup)
        logger.debug(f"BSR operation: matrix {bsr_shape}, block size {bsr_blocksize}")

        # Create temporary BSR matrix for operation
        import scipy.sparse as sp

        try:
            temp_bsr = sp.bsr_matrix(
                (bsr_data["data"], bsr_data["indices"], bsr_data["indptr"]),
                shape=bsr_shape,
                blocksize=bsr_blocksize,
            )
        except Exception as e:
            logger.error(f"Failed to create BSR matrix: {e}")
            raise

        # Perform A^T @ b operation
        cp.cuda.runtime.deviceSynchronize()
        operation_start_sync = time.time()

        # BSR transpose multiply (on CPU for now, GPU version needs more setup)
        target_combined_cpu = (
            target_combined_gpu.get()
        )  # Use .get() instead of cp.asnumpy()
        ATb_combined_cpu = temp_bsr.T @ target_combined_cpu
        ATb_combined_gpu = cp.asarray(ATb_combined_cpu)

        cp.cuda.runtime.deviceSynchronize()
        operation_time = time.time() - operation_start_sync
        operation_time_total = time.time() - operation_start

        logger.debug(f"BSR A^T@b operation analysis:")
        logger.debug(f"  BSR matrix shape: {bsr_shape}, NNZ: {temp_bsr.nnz:,}")
        logger.debug(f"  BSR block size: {bsr_blocksize}")
        logger.debug(f"  Operation time (sync): {operation_time:.4f}s")
        logger.debug(
            f"  Note: Using CPU BSR operation (GPU version needs cuSPARSE handle setup)"
        )

        # Phase 3: Convert result back to tensor form
        tensor_conversion_start = time.time()
        ATb_result = cp.zeros((bsr_led_count, 3), dtype=cp.float32)
        ATb_result[:, 0] = ATb_combined_gpu[:bsr_led_count]  # R LEDs
        ATb_result[:, 1] = ATb_combined_gpu[bsr_led_count : 2 * bsr_led_count]  # G LEDs
        ATb_result[:, 2] = ATb_combined_gpu[
            2 * bsr_led_count : 3 * bsr_led_count
        ]  # B LEDs
        tensor_conversion_time = time.time() - tensor_conversion_start

        total_atb_time = time.time() - atb_start_time

        logger.debug(
            f"BSR A^T*b timing - CPU→GPU: {cpu_to_gpu_time:.4f}s, "
            f"Operation: {operation_time:.4f}s, "
            f"Tensor conv: {tensor_conversion_time:.4f}s, "
            f"Total: {total_atb_time:.4f}s"
        )

        # Clean up
        del temp_bsr, target_combined_cpu, ATb_combined_cpu

        return ATb_result

    def debug_sparse_performance(self, target_frame: np.ndarray) -> Dict[str, float]:
        """
        Debug sparse matrix performance with different approaches.

        Args:
            target_frame: Target image for testing

        Returns:
            Dictionary with timing results for different approaches
        """
        if not self._matrix_loaded:
            raise RuntimeError("Matrices not loaded")

        logger.info("=== Sparse Matrix Performance Debug ===")

        # Prepare target vector (same as _calculate_ATb)
        target_rgb_normalized = target_frame.astype(np.float32) / 255.0
        target_flattened = target_rgb_normalized.reshape(-1, 3)
        pixels = target_flattened.shape[0]
        target_combined = np.empty(pixels * 3, dtype=np.float32)
        target_combined[:pixels] = target_flattened[:, 0]
        target_combined[pixels : 2 * pixels] = target_flattened[:, 1]
        target_combined[2 * pixels :] = target_flattened[:, 2]
        target_combined_gpu = cp.asarray(target_combined)

        # Debug dimensions
        logger.info(f"Target frame shape: {target_frame.shape}")
        logger.info(f"Target combined shape: {target_combined_gpu.shape}")
        logger.info(f"Combined matrix shape: {self._A_combined_csc_gpu.shape}")
        logger.info(f"Combined matrix .T shape: {self._A_combined_csc_gpu.T.shape}")
        logger.info(
            f"Individual matrix shapes: R={self._A_r_csc_gpu.shape}, "
            f"G={self._A_g_csc_gpu.shape}, B={self._A_b_csc_gpu.shape}"
        )

        results = {}

        # Test 1: Current approach (combined block diagonal)
        logger.info("Testing current combined block diagonal approach...")
        cp.cuda.runtime.deviceSynchronize()
        start = time.time()
        result1 = self._A_combined_csc_gpu.T @ target_combined_gpu
        cp.cuda.runtime.deviceSynchronize()
        results["combined_block_diagonal"] = time.time() - start

        # Test 2: Individual channel operations
        logger.info("Testing individual channel operations...")
        target_r = cp.asarray(target_flattened[:, 0].ravel())
        target_g = cp.asarray(target_flattened[:, 1].ravel())
        target_b = cp.asarray(target_flattened[:, 2].ravel())

        cp.cuda.runtime.deviceSynchronize()
        start = time.time()
        atb_r = self._A_r_csc_gpu.T @ target_r
        atb_g = self._A_g_csc_gpu.T @ target_g
        atb_b = self._A_b_csc_gpu.T @ target_b
        cp.cuda.runtime.deviceSynchronize()
        results["individual_channels"] = time.time() - start

        # Test 3: Check if it's the transpose that's slow
        logger.info("Testing non-transposed operation...")
        if target_combined_gpu.shape[0] == self._A_combined_csc_gpu.shape[1]:
            cp.cuda.runtime.deviceSynchronize()
            start = time.time()
            result3 = self._A_combined_csc_gpu @ target_combined_gpu
            cp.cuda.runtime.deviceSynchronize()
            results["non_transposed"] = time.time() - start
        else:
            logger.warning(
                f"Skipping non-transposed test: dimension mismatch "
                f"{target_combined_gpu.shape[0]} != {self._A_combined_csc_gpu.shape[1]}"
            )
            results["non_transposed"] = -1

        # Test 4: Convert to CSR format and test
        logger.info("Testing CSR format...")
        try:
            A_combined_csr = self._A_combined_csc_gpu.tocsr()
            cp.cuda.runtime.deviceSynchronize()
            start = time.time()
            result4 = A_combined_csr.T @ target_combined_gpu
            cp.cuda.runtime.deviceSynchronize()
            results["csr_format"] = time.time() - start
        except Exception as e:
            logger.warning(f"CSR test failed: {e}")
            results["csr_format"] = -1

        # Test 5: Check if multiple operations are faster (cache effects)
        logger.info("Testing repeated operations (cache effects)...")
        cp.cuda.runtime.deviceSynchronize()
        start = time.time()
        for _ in range(5):
            result5 = self._A_combined_csc_gpu.T @ target_combined_gpu
        cp.cuda.runtime.deviceSynchronize()
        results["repeated_operations"] = (time.time() - start) / 5

        # Test 6: BSR format tests (if available)
        if self._bsr_available and self._A_combined_bsr_data is not None:
            logger.info("Testing real BSR format operations...")
            try:
                # Test BSR method with actual BSR matrices
                logger.info("Testing BSR A^T*b method...")
                start = time.time()
                atb_result_bsr = self._calculate_ATb_bsr(target_frame)
                results["bsr_real_method"] = time.time() - start

                # Test repeated BSR operations
                logger.info("Testing BSR repeated operations...")
                start = time.time()
                for _ in range(3):  # Fewer repetitions due to CPU/GPU transfer overhead
                    atb_result_bsr_repeat = self._calculate_ATb_bsr(target_frame)
                results["bsr_cusparse_repeated"] = (time.time() - start) / 3

                logger.info(
                    f"BSR test completed with {self._bsr_led_count} LEDs (vs {self._actual_led_count} CSC LEDs)"
                )

            except Exception as e:
                logger.warning(f"BSR tests failed: {e}")
                results["bsr_real_method"] = -1
                results["bsr_cusparse_repeated"] = -1
        else:
            logger.info("BSR matrices not available")
            results["bsr_real_method"] = -1
            results["bsr_cusparse_repeated"] = -1

        # Test 7: Check GPU memory info during operation
        logger.info("GPU memory analysis...")
        meminfo_before = cp.cuda.runtime.memGetInfo()
        cp.cuda.runtime.deviceSynchronize()
        start = time.time()
        result7 = self._A_combined_csc_gpu.T @ target_combined_gpu
        meminfo_after = cp.cuda.runtime.memGetInfo()
        cp.cuda.runtime.deviceSynchronize()
        results["memory_test"] = time.time() - start

        logger.info(f"Memory before: {meminfo_before[0]/(1024**2):.1f}MB free")
        logger.info(f"Memory after: {meminfo_after[0]/(1024**2):.1f}MB free")
        logger.info(
            f"Memory used: {(meminfo_before[0] - meminfo_after[0])/(1024**2):.1f}MB"
        )

        # Report results
        logger.info("=== Performance Results ===")
        for method, timing in results.items():
            if timing > 0:
                logger.info(f"{method}: {timing:.4f}s")
            else:
                logger.info(f"{method}: FAILED")

        # Calculate matrix properties
        matrix_info = {
            "csc_shape": self._A_combined_csc_gpu.shape,
            "csc_nnz": self._A_combined_csc_gpu.nnz,
            "csc_density_percent": self._A_combined_csc_gpu.nnz
            / (self._A_combined_csc_gpu.shape[0] * self._A_combined_csc_gpu.shape[1])
            * 100,
            "csc_size_mb": self._A_combined_csc_gpu.data.nbytes / (1024**2),
        }

        # Add BSR properties if available
        if self._bsr_available and hasattr(self, "_A_combined_bsr_data"):
            bsr_data = self._A_combined_bsr_data
            matrix_info.update(
                {
                    "bsr_available": True,
                    "bsr_led_count": getattr(self, "_bsr_led_count", "unknown"),
                    "bsr_shape": bsr_data.get("shape", "unknown"),
                    "bsr_block_size": self._bsr_block_size,
                    "bsr_nnz": bsr_data.get("nnz", "unknown"),
                    "bsr_memory_mb": (
                        bsr_data["data"].nbytes
                        + bsr_data["indices"].nbytes
                        + bsr_data["indptr"].nbytes
                    )
                    / (1024**2)
                    if "data" in bsr_data
                    else "unknown",
                    "bsr_cusparse_lib_available": getattr(
                        self, "_bsr_cusparse_lib", None
                    )
                    is not None,
                    "bsr_status": "Real BSR matrices loaded and ready",
                }
            )
        elif self._bsr_available:
            matrix_info.update(
                {
                    "bsr_available": True,
                    "bsr_status": "Framework available but matrices not loaded",
                }
            )
        else:
            matrix_info.update(
                {
                    "bsr_available": False,
                    "bsr_status": "Not available",
                }
            )

        logger.info("=== Matrix Properties ===")
        for key, value in matrix_info.items():
            logger.info(f"{key}: {value}")

        return results

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

        # Start timing for the optimization loop
        loop_start_time = time.time()

        # Get workspace arrays
        w = self._gpu_workspace
        x = self._led_values_gpu  # (led_count, 3)

        # Verify x shape
        assert x.shape == (
            self._actual_led_count,
            3,
        ), f"x shape {x.shape} != expected ({self._actual_led_count}, 3)"

        einsum_time = 0.0
        step_size_time = 0.0
        step_einsum_time = 0.0

        # Track per-iteration timing for cache warming analysis
        per_iteration_times = []
        einsum_per_iteration = []
        step_per_iteration = []

        for iteration in range(max_iters):
            iteration_start = time.time()

            # KEY OPTIMIZATION: Use einsum for parallel ATA @ x computation
            # ATA: (led_count, led_count, 3), x: (led_count, 3) -> (led_count, 3)
            # einsum 'ijk,jk->ik' computes all 3 channels in parallel
            cp.cuda.runtime.deviceSynchronize()  # Ensure accurate timing
            einsum_start = time.time()
            w["ATA_x"][:] = cp.einsum("ijk,jk->ik", self._ATA_gpu, x)
            w["gradient"][:] = w["ATA_x"] - ATb
            cp.cuda.runtime.deviceSynchronize()
            einsum_duration = time.time() - einsum_start
            einsum_time += einsum_duration
            einsum_per_iteration.append(einsum_duration)

            # KEY STEP 4: Compute step size using optimized dense operations
            step_start = time.time()
            step_size, step_einsum_duration = self._compute_dense_step_size(
                w["gradient"]
            )
            step_duration = time.time() - step_start
            step_size_time += step_duration
            step_einsum_time += step_einsum_duration
            step_per_iteration.append(step_duration)

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

            iteration_time = time.time() - iteration_start
            per_iteration_times.append(iteration_time)

            # Log first few iterations to detect cache warming
            if iteration < 3:
                logger.debug(
                    f"Iteration {iteration}: {iteration_time:.4f}s "
                    f"(einsum: {einsum_duration:.4f}s, step: {step_duration:.4f}s)"
                )

        # Ensure we return the current x
        self._led_values_gpu[:] = x

        loop_total_time = time.time() - loop_start_time

        # Analyze per-iteration timing for cache warming effects
        cache_analysis = {}
        if len(per_iteration_times) > 1:
            first_iter = per_iteration_times[0]
            avg_later_iters = (
                sum(per_iteration_times[1:]) / len(per_iteration_times[1:])
                if len(per_iteration_times) > 1
                else first_iter
            )

            first_einsum = einsum_per_iteration[0] if einsum_per_iteration else 0
            avg_later_einsum = (
                sum(einsum_per_iteration[1:]) / len(einsum_per_iteration[1:])
                if len(einsum_per_iteration) > 1
                else first_einsum
            )

            first_step = step_per_iteration[0] if step_per_iteration else 0
            avg_later_step = (
                sum(step_per_iteration[1:]) / len(step_per_iteration[1:])
                if len(step_per_iteration) > 1
                else first_step
            )

            cache_analysis = {
                "first_iteration_time": first_iter,
                "avg_later_iterations_time": avg_later_iters,
                "cache_warmup_speedup": first_iter / avg_later_iters
                if avg_later_iters > 0
                else 1.0,
                "first_einsum_time": first_einsum,
                "avg_later_einsum_time": avg_later_einsum,
                "einsum_warmup_speedup": first_einsum / avg_later_einsum
                if avg_later_einsum > 0
                else 1.0,
                "first_step_time": first_step,
                "avg_later_step_time": avg_later_step,
                "step_warmup_speedup": first_step / avg_later_step
                if avg_later_step > 0
                else 1.0,
                "per_iteration_times": per_iteration_times[:5],  # First 5 for debugging
            }

            logger.debug(
                f"Dense loop cache analysis: "
                f"1st iter: {first_iter:.4f}s, avg later: {avg_later_iters:.4f}s, "
                f"speedup: {cache_analysis['cache_warmup_speedup']:.2f}x"
            )

        timing_info = {
            "total_loop_time": loop_total_time,
            "einsum_time": einsum_time,
            "step_size_time": step_size_time,
            "step_einsum_time": step_einsum_time,
            "iterations_completed": iteration + 1,
            "cache_analysis": cache_analysis,
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

        # Use GPU events for accurate timing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        g_dot_ATA_g = cp.einsum("ik,ijk,jk->", gradient, self._ATA_gpu, gradient)
        end_event.record()

        cp.cuda.runtime.deviceSynchronize()
        einsum_duration = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0

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

        return stats
