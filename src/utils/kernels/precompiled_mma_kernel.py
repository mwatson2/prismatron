#!/usr/bin/env python3
"""
Precompiled MMA Kernel loader for symmetric block diagonal matrix multiplication.

This module loads ahead-of-time compiled CUDA MMA kernels from shared libraries,
avoiding runtime compilation issues with MMA headers.

Key features:
- 16x16x16 async MMA operations for tensor core utilization
- Batch processing for multiple input vectors
- Symmetric storage optimization (upper triangular blocks only)
- FP32 precision for compute and accumulation
- No runtime compilation dependencies
"""

import ctypes
import os
from typing import Optional

try:
    import cupy
    from cupy.cuda import runtime
except ImportError:
    # Fallback for systems without CUDA
    import numpy as cupy

    runtime = None

import numpy as np


class PrecompiledMMAKernel:
    """
    Loader for precompiled MMA kernels with ctypes interface.

    Uses ahead-of-time compiled CUDA shared library to avoid runtime
    compilation issues with MMA headers.
    """

    def __init__(self):
        """
        Initialize precompiled MMA kernel loader.
        """
        self.lib = None
        self.kernel_basic = None

        # Load precompiled shared library
        self._load_library()

    def _load_library(self):
        """Load the precompiled CUDA shared library."""
        # Find the shared library relative to this file
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(kernel_dir, "libbatch_mma_kernel.so")

        if not os.path.exists(lib_path):
            raise RuntimeError(
                f"Precompiled MMA kernel library not found at {lib_path}. "
                f"Run 'make' in {kernel_dir} to compile the kernels."
            )

        try:
            # Load shared library with ctypes
            self.lib = ctypes.CDLL(lib_path)

            # Get function pointers
            self.kernel_basic = self.lib.batch_symmetric_block_dia_multiply_wmma

            # Set argument types for better error checking
            for kernel in [self.kernel_basic]:
                kernel.argtypes = [
                    ctypes.c_void_p,  # block_data
                    ctypes.c_void_p,  # block_offsets
                    ctypes.c_void_p,  # input_batch
                    ctypes.c_void_p,  # output_batch
                    ctypes.c_int,  # batch_size
                    ctypes.c_int,  # channels
                    ctypes.c_int,  # led_blocks
                    ctypes.c_int,  # block_diag_count
                    ctypes.c_int,  # padded_leds
                ]
                kernel.restype = None

        except Exception as e:
            raise RuntimeError(f"Failed to load precompiled MMA kernels: {e}") from e

    def __call__(
        self,
        block_data_gpu: cupy.ndarray,  # (channels, block_diag_count, 16, 16)
        block_offsets: cupy.ndarray,  # (block_diag_count,)
        input_batch: cupy.ndarray,  # (batch_size, channels, padded_leds)
        led_blocks: int,
        padded_led_count: int,
    ) -> cupy.ndarray:
        """
        Perform batch MMA matrix multiplication using precompiled kernels.

        Args:
            block_data_gpu: Block diagonal data
            block_offsets: Block offset array
            input_batch: Batch input vectors
            led_blocks: Number of 16x16 blocks per dimension
            padded_led_count: Padded LED count (multiple of 16)

        Returns:
            Output batch (batch_size, channels, padded_leds)
        """
        if self.lib is None:
            raise RuntimeError("Precompiled MMA kernels not loaded")

        batch_size, channels, padded_leds = input_batch.shape

        # Validate input shapes
        if block_data_gpu.shape[2:] != (16, 16):
            raise ValueError("Block data must have 16x16 blocks")
        if padded_leds != padded_led_count:
            raise ValueError("Input LED count must match padded count")

        # Initialize output
        output_batch = cupy.zeros_like(input_batch)

        # Ensure FP32 data type
        if block_data_gpu.dtype != cupy.float32:
            block_data_gpu = block_data_gpu.astype(cupy.float32)
        if input_batch.dtype != cupy.float32:
            input_batch = input_batch.astype(cupy.float32)
        if block_offsets.dtype != cupy.int32:
            block_offsets = block_offsets.astype(cupy.int32)

        # Kernel launch configuration
        # 1D grid for basic kernel
        grid = (batch_size * channels * ((led_blocks + 7) // 8),)  # 8 warps per block
        block = (256,)  # 8 warps * 32 threads
        kernel = self.kernel_basic

        # Launch kernel using CuPy's raw kernel interface
        # We need to use CuPy's kernel launch mechanism, not ctypes directly
        self._launch_kernel_cupy(
            kernel,
            grid,
            block,
            block_data_gpu,
            block_offsets,
            input_batch,
            output_batch,
            batch_size,
            channels,
            led_blocks,
            len(block_offsets),
            padded_led_count,
        )

        return output_batch

    def _launch_kernel_cupy(
        self,
        kernel_func,
        grid,
        block,
        block_data_gpu,
        block_offsets,
        input_batch,
        output_batch,
        batch_size,
        channels,
        led_blocks,
        block_diag_count,
        padded_led_count,
    ):
        """
        Launch precompiled kernel using CuPy's RawKernel interface.

        This is a workaround since we can't directly call ctypes functions
        as CUDA kernels from CuPy.
        """
        # For now, we'll need to use CuPy's RawKernel with the compiled code
        # This requires a different approach - we need to create a CuPy module
        # from the precompiled object code

        # Alternative approach: Use CuPy's direct kernel launch
        # This requires CUDA driver API access through CuPy

        try:
            # Get CUDA function from the loaded library
            # This is a more complex integration that would require
            # CuPy's internal CUDA module system

            # For now, raise an error indicating this needs CuPy integration
            raise NotImplementedError(
                "Direct kernel launch from precompiled library requires "
                "additional CuPy integration. This will be implemented "
                "in the next iteration."
            )

        except Exception as e:
            raise RuntimeError(f"Failed to launch precompiled kernel: {e}") from e


class PrecompiledBatch8CorrectedSymmetricWMMAMatMul:
    """
    8-frame batch symmetric WMMA matrix multiplication using corrected kernel.

    This uses the fixed implementation with proper accumulation logic and FP32 input/output.
    """

    def __init__(self):
        """
        Initialize precompiled corrected 8-frame WMMA kernel.
        """
        # Load precompiled corrected kernel using CuPy's RawModule
        self._kernel = None
        self._load_precompiled_corrected_kernel()

    def _load_precompiled_corrected_kernel(self):
        """Load precompiled corrected 8-frame kernel using CuPy's RawModule with PTX or CUBIN."""
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        ptx_path = os.path.join(kernel_dir, "batch8_corrected_kernel.ptx")
        cubin_path = os.path.join(kernel_dir, "batch8_corrected_kernel.cubin")

        # Try PTX first, then fall back to CUBIN
        if os.path.exists(ptx_path):
            kernel_path = ptx_path
            use_ptx = True
        elif os.path.exists(cubin_path):
            kernel_path = cubin_path
            use_ptx = False
        else:
            raise RuntimeError(
                f"Precompiled corrected 8-frame MMA kernel module not found. "
                f"Expected {ptx_path} or {cubin_path}. "
                f"Run 'make' in {kernel_dir} to compile the kernels."
            )

        try:
            # Create a RawModule from the kernel file
            self._module = cupy.RawModule(path=kernel_path)

            # Get the corrected kernel function
            kernel_name_basic = "batch8_symmetric_block_pair_multiply_wmma"

            self._kernel_basic_func = self._module.get_function(kernel_name_basic)

            kernel_type = "PTX" if use_ptx else "CUBIN"
            print(
                f"Precompiled corrected 8-frame MMA tensor core kernels loaded successfully "
                f"from {kernel_type}: {kernel_path}"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load precompiled corrected 8-frame MMA kernels: {e}") from e

    def __call__(
        self,
        block_data_gpu: cupy.ndarray,
        block_offsets: cupy.ndarray,
        input_batch: cupy.ndarray,
        led_blocks: int,
        led_count: int,  # Changed from padded_led_count to led_count
    ) -> cupy.ndarray:
        """
        Perform 8-frame batch WMMA matrix multiplication using corrected implementation.

        Args:
            block_data_gpu: Block diagonal data (channels, block_diag_count, 16, 16)
            block_offsets: Block offset array (block_diag_count,)
            input_batch: Batch input vectors (8, channels, leds)
            led_blocks: Number of 16x16 blocks per dimension
            led_count: LED count (multiple of 32, no padding needed)

        Returns:
            Output batch (8, channels, leds)
        """
        if not hasattr(self, "_module"):
            raise RuntimeError("Precompiled corrected 8-frame kernels not loaded")

        batch_size, channels, leds = input_batch.shape

        if batch_size != 8:
            raise ValueError(f"Corrected 8-frame kernel requires batch_size=8, got {batch_size}")

        # Validate input shapes
        if block_data_gpu.shape[2:] != (16, 16):
            raise ValueError("Block data must have 16x16 blocks")
        if leds != led_count:
            raise ValueError("Input LED count must match expected count")

        # Initialize output
        output_batch = cupy.zeros_like(input_batch)

        # Ensure correct data types
        if block_data_gpu.dtype != cupy.float32:
            block_data_gpu = block_data_gpu.astype(cupy.float32)
        if input_batch.dtype != cupy.float32:
            input_batch = input_batch.astype(cupy.float32)
        if block_offsets.dtype != cupy.int32:
            block_offsets = block_offsets.astype(cupy.int32)

        # Select kernel and launch configuration
        kernel_func = self._kernel_basic_func
        block_x, block_y, block_z = 32, 1, 1  # 1 warp per block

        # 2D grid for corrected 8-frame kernel: (channels, led_blocks/2)
        # Each kernel processes one vertical pair (32x16 block) for 8-frame batch
        grid_x, grid_y, grid_z = channels, (led_blocks + 1) // 2, 1

        # Launch the corrected kernel with proper signature
        kernel_func(
            (grid_x, grid_y, grid_z),  # grid
            (block_x, block_y, block_z),  # block
            (
                block_data_gpu,
                block_offsets,
                input_batch,
                output_batch,
                cupy.int32(batch_size),  # Always 8
                cupy.int32(channels),
                cupy.int32(led_blocks),
                cupy.int32(len(block_offsets)),
                cupy.int32(led_count),  # Use led_count instead of padded_led_count
            ),
        )

        return output_batch


class PrecompiledBatch8SymmetricWMMAMatMul:
    """
    8-frame batch symmetric WMMA matrix multiplication using precompiled kernels.

    Uses vertical block pair processing with 32x8x16 WMMA operations for optimal 8-frame
    batch processing. Processes vertically adjacent 16x16 blocks in pairs to form 32x16 matrices.
    """

    def __init__(self):
        """
        Initialize precompiled 8-frame WMMA kernel.
        """
        # Load precompiled 8-frame kernel using CuPy's RawModule
        self._kernel = None
        self._load_precompiled_8frame_kernel()

    def _load_precompiled_8frame_kernel(self):
        """Load precompiled 8-frame vertical pair kernel using CuPy's RawModule with PTX or CUBIN."""
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        ptx_path = os.path.join(kernel_dir, "batch8_vertical_pair_kernel.ptx")
        cubin_path = os.path.join(kernel_dir, "batch8_vertical_pair_kernel.cubin")

        # Try PTX first, then fall back to CUBIN
        if os.path.exists(ptx_path):
            kernel_path = ptx_path
            use_ptx = True
        elif os.path.exists(cubin_path):
            kernel_path = cubin_path
            use_ptx = False
        else:
            raise RuntimeError(
                f"Precompiled 8-frame MMA kernel module not found. "
                f"Expected {ptx_path} or {cubin_path}. "
                f"Run 'make' in {kernel_dir} to compile the kernels."
            )

        try:
            # Create a RawModule from the kernel file
            self._module = cupy.RawModule(path=kernel_path)

            # Get the 8-frame vertical pair kernel function
            kernel_name_basic = "batch8_vertical_pair_multiply_wmma"

            self._kernel_basic_func = self._module.get_function(kernel_name_basic)

            kernel_type = "PTX" if use_ptx else "CUBIN"
            print(f"Precompiled 8-frame MMA tensor core kernels loaded successfully from {kernel_type}: {kernel_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load precompiled 8-frame MMA kernels: {e}") from e

    def __call__(
        self,
        block_data_gpu: cupy.ndarray,
        block_offsets: cupy.ndarray,
        input_batch: cupy.ndarray,
        led_blocks: int,
        padded_led_count: int,
    ) -> cupy.ndarray:
        """
        Perform 8-frame batch WMMA matrix multiplication using vertical block pair processing.

        Args:
            block_data_gpu: Block diagonal data (channels, block_diag_count, 16, 16)
            block_offsets: Block offset array (block_diag_count,)
            input_batch: Batch input vectors (8, channels, padded_leds)
            led_blocks: Number of 16x16 blocks per dimension
            padded_led_count: Padded LED count (multiple of 16)

        Returns:
            Output batch (8, channels, padded_leds)
        """
        if not hasattr(self, "_module"):
            raise RuntimeError("Precompiled 8-frame kernels not loaded")

        batch_size, channels, padded_leds = input_batch.shape

        if batch_size != 8:
            raise ValueError(f"8-frame kernel requires batch_size=8, got {batch_size}")

        # Validate input shapes
        if block_data_gpu.shape[2:] != (16, 16):
            raise ValueError("Block data must have 16x16 blocks")
        if padded_leds != padded_led_count:
            raise ValueError("Input LED count must match padded count")

        # Initialize output
        output_batch = cupy.zeros_like(input_batch)

        # Ensure correct data types
        if block_data_gpu.dtype != cupy.float32:
            block_data_gpu = block_data_gpu.astype(cupy.float32)
        if input_batch.dtype != cupy.float32:
            input_batch = input_batch.astype(cupy.float32)
        if block_offsets.dtype != cupy.int32:
            block_offsets = block_offsets.astype(cupy.int32)

        # Select kernel and launch configuration
        kernel_func = self._kernel_basic_func
        # Basic kernel uses 1 warp (32 threads)
        block_x, block_y, block_z = 32, 1, 1  # 1 warp per block

        # 2D grid for 8-frame vertical pair kernel: (channels, led_blocks/2)
        # Each kernel processes one vertical pair (32x16 block) for 8-frame batch
        grid_x, grid_y, grid_z = channels, (led_blocks + 1) // 2, 1

        # Launch the precompiled 8-frame kernel
        kernel_func(
            (grid_x, grid_y, grid_z),  # grid
            (block_x, block_y, block_z),  # block
            (
                block_data_gpu,
                block_offsets,
                input_batch,
                output_batch,
                cupy.int32(batch_size),  # Always 8
                cupy.int32(channels),
                cupy.int32(led_blocks),
                cupy.int32(len(block_offsets)),
                cupy.int32(padded_led_count),
            ),
        )

        return output_batch


class PrecompiledBatchSymmetricWMMAMatMul:
    """
    Batch symmetric WMMA matrix multiplication using precompiled kernels.

    This class provides the same interface as BatchSymmetricWMMAMatMul
    but uses ahead-of-time compiled kernels instead of runtime compilation.
    """

    def __init__(self):
        """
        Initialize precompiled WMMA kernel.
        """
        # For now, fallback to a simpler approach using CuPy's LoadLibrary
        self._kernel = None
        self._load_precompiled_kernel()

    def _load_precompiled_kernel(self):
        """Load precompiled kernel using CuPy's RawModule with PTX or CUBIN."""
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        ptx_path = os.path.join(kernel_dir, "batch_mma_kernel.ptx")
        cubin_path = os.path.join(kernel_dir, "batch_mma_kernel.cubin")

        # Try PTX first, then fall back to CUBIN
        if os.path.exists(ptx_path):
            kernel_path = ptx_path
            with open(ptx_path) as f:
                kernel_source = f.read()
            use_ptx = True
        elif os.path.exists(cubin_path):
            kernel_path = cubin_path
            with open(cubin_path, "rb") as f:
                kernel_source = f.read()
            use_ptx = False
        else:
            raise RuntimeError(
                f"Precompiled MMA kernel module not found. "
                f"Expected {ptx_path} or {cubin_path}. "
                f"Run 'make' in {kernel_dir} to compile the kernels."
            )

        try:
            # Create a RawModule from the kernel file
            # Both PTX and CUBIN can be loaded using path parameter
            self._module = cupy.RawModule(path=kernel_path)

            # Get the kernel function
            kernel_name_basic = "batch_symmetric_block_dia_multiply_wmma"

            self._kernel_basic_func = self._module.get_function(kernel_name_basic)

            kernel_type = "PTX" if use_ptx else "CUBIN"
            print(f"Precompiled MMA tensor core kernels loaded successfully from {kernel_type}: {kernel_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load precompiled MMA kernels: {e}") from e

    def __call__(
        self,
        block_data_gpu: cupy.ndarray,
        block_offsets: cupy.ndarray,
        input_batch: cupy.ndarray,
        led_blocks: int,
        padded_led_count: int,
    ) -> cupy.ndarray:
        """
        Perform batch WMMA matrix multiplication.

        Args:
            block_data_gpu: Block diagonal data
            block_offsets: Block offset array
            input_batch: Batch input vectors
            led_blocks: Number of 16x16 blocks per dimension
            padded_led_count: Padded LED count (multiple of 16)

        Returns:
            Output batch (batch_size, channels, padded_leds)
        """
        if not hasattr(self, "_module"):
            raise RuntimeError("Precompiled kernels not loaded")

        batch_size, channels, padded_leds = input_batch.shape

        # Validate input shapes
        if block_data_gpu.shape[2:] != (16, 16):
            raise ValueError("Block data must have 16x16 blocks")
        if padded_leds != padded_led_count:
            raise ValueError("Input LED count must match padded count")

        # Initialize output
        output_batch = cupy.zeros_like(input_batch)

        # Ensure correct data types
        if block_data_gpu.dtype != cupy.float32:
            block_data_gpu = block_data_gpu.astype(cupy.float32)
        if input_batch.dtype != cupy.float32:
            input_batch = input_batch.astype(cupy.float32)
        if block_offsets.dtype != cupy.int32:
            block_offsets = block_offsets.astype(cupy.int32)

        # Select kernel and launch configuration
        kernel_func = self._kernel_basic_func
        # 2D grid for basic kernel: (channels, led_blocks)
        # Each kernel processes entire batch (16 vectors) simultaneously
        grid_x, grid_y, grid_z = channels, led_blocks, 1
        block_x, block_y, block_z = 32, 1, 1  # 1 warp per block

        # Launch the precompiled kernel directly using CuPy's RawKernel interface
        # kernel_func is already a CuPy function from RawModule.get_function()
        kernel_func(
            (grid_x, grid_y, grid_z),  # grid
            (block_x, block_y, block_z),  # block
            (
                block_data_gpu,
                block_offsets,
                input_batch,
                output_batch,
                cupy.int32(batch_size),
                cupy.int32(channels),
                cupy.int32(led_blocks),
                cupy.int32(len(block_offsets)),
                cupy.int32(padded_led_count),
            ),
        )

        return output_batch


# Kernel availability check
def check_precompiled_mma_support() -> bool:
    """Check if precompiled MMA kernels are available."""
    try:
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        ptx_path = os.path.join(kernel_dir, "batch_mma_kernel.ptx")
        cubin_path = os.path.join(kernel_dir, "batch_mma_kernel.cubin")

        # Check if PTX or CUBIN file exists
        return os.path.exists(ptx_path) or os.path.exists(cubin_path)
    except Exception:
        return False


def check_precompiled_8frame_mma_support() -> bool:
    """Check if precompiled 8-frame vertical pair MMA kernels are available."""
    try:
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        ptx_path = os.path.join(kernel_dir, "batch8_vertical_pair_kernel.ptx")
        cubin_path = os.path.join(kernel_dir, "batch8_vertical_pair_kernel.cubin")

        # Check if PTX or CUBIN file exists
        return os.path.exists(ptx_path) or os.path.exists(cubin_path)
    except Exception:
        return False


def check_precompiled_8frame_corrected_mma_support() -> bool:
    """Check if precompiled corrected 8-frame MMA kernels are available."""
    try:
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        ptx_path = os.path.join(kernel_dir, "batch8_corrected_kernel.ptx")
        cubin_path = os.path.join(kernel_dir, "batch8_corrected_kernel.cubin")

        # Check if PTX or CUBIN file exists
        return os.path.exists(ptx_path) or os.path.exists(cubin_path)
    except Exception:
        return False


# Module-level availability flags
PRECOMPILED_MMA_SUPPORTED = check_precompiled_mma_support()
PRECOMPILED_8FRAME_MMA_SUPPORTED = check_precompiled_8frame_mma_support()
PRECOMPILED_8FRAME_CORRECTED_MMA_SUPPORTED = check_precompiled_8frame_corrected_mma_support()

if __name__ == "__main__":
    # Test precompiled kernel loading
    print("Testing precompiled MMA kernel loading...")

    if PRECOMPILED_MMA_SUPPORTED:
        print("✓ Precompiled MMA kernels are available")

        try:
            kernel = PrecompiledBatchSymmetricWMMAMatMul()
            print("✓ Precompiled kernel loader initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize precompiled kernel loader: {e}")
    else:
        print("✗ Precompiled MMA kernels are not available")
        print("Run 'make' in the kernels directory to compile them")
