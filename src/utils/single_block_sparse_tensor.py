"""
Single Block Mixed Sparse Tensor - Custom tensor for LED diffusion patterns.

This module implements a specialized 4D tensor format optimized for LED diffusion
patterns where each LED has exactly one small dense region in the spatial dimensions.

Key features:
- 4D tensor (batch, channels, height, width) with mixed sparsity
- First two dimensions (batch, channels) are dense
- Last two dimensions (height, width) are sparse with exactly one square block per sub-tensor
- Optimized for GPU operations and repeated A^T @ b computations
- Memory efficient: stores only non-zero blocks + positions
"""

import logging
from typing import Dict, Optional, Tuple, Union

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)


class SingleBlockMixedSparseTensor:
    """
    Specialized 4D tensor for LED diffusion patterns with single dense blocks.

    Stores tensors of shape (batch_size, channels, height, width) where each
    (height, width) sub-tensor contains exactly one dense square block.

    Memory layout:
    - sparse_values: (batch_size, channels, block_size, block_size) - dense blocks
    - block_positions: (batch_size, channels, 2) - top-left coordinates

    Example:
        # Create tensor for 1000 LEDs, 3 channels, 800x480 images, 64x64 blocks
        tensor = SingleBlockMixedSparseTensor(1000, 3, 800, 480, 64)

        # Set a block for LED 0, red channel at position (100, 150)
        block_data = cp.random.rand(64, 64, dtype=cp.float32)
        tensor.set_block(0, 0, 100, 150, block_data)

        # Compute A^T @ b operation
        target_image = cp.random.rand(800, 480, dtype=cp.float32)
        result = tensor.transpose_dot_product(target_image)  # Returns (1000, 3)
    """

    def __init__(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        block_size: int = 96,
        device: str = "cuda",
        dtype: cp.dtype = cp.float32,
        output_dtype: Optional[cp.dtype] = None,
    ):
        """
        Initialize the single block sparse tensor.

        Args:
            batch_size: Number of LEDs
            channels: Number of color channels (typically 3 for RGB)
            height: Spatial height of full images
            width: Spatial width of full images
            block_size: Size of square dense blocks (e.g., 64)
            device: Device to store tensors on ('cuda' or 'cpu')
            dtype: Data type for sparse values (cp.float32 or cp.uint8)
            output_dtype: Data type for output tensors (cp.float32 or cp.float16).
                         If None, defaults to cp.float32 for cp.uint8 input, or same as dtype for others.
        """
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.block_size = block_size
        self.device = device
        self.dtype = self._validate_dtype(dtype)

        # Set output dtype with intelligent defaults
        if output_dtype is None:
            # Default to FP32 for INT8 input (most common case), same dtype for FP32 input
            self.output_dtype = cp.float32 if self.dtype == cp.uint8 else self.dtype
        else:
            self.output_dtype = self._validate_output_dtype(output_dtype)

        # Storage for dense blocks - only non-zero regions
        # Shape: (channels, batch_size, block_size, block_size) for planar layout
        self.sparse_values = cp.zeros((channels, batch_size, block_size, block_size), dtype=self.dtype)

        # For int8, alignment is less critical, but still check for fp32
        if self.dtype == cp.float32:
            assert self.sparse_values.data.ptr % 16 == 0, "Sparse values array must be 16-byte aligned for fp32"

        # Storage for block positions (top-left coordinates)
        # Shape: (channels, batch_size, 2) for planar layout
        self.block_positions = cp.zeros((channels, batch_size, 2), dtype=cp.int32)

        # Note: All blocks are assumed to be set - no tracking needed

        logger.debug(
            f"SingleBlockMixedSparseTensor created: "
            f"({batch_size}, {channels}, {height}, {width}) "
            f"with {block_size}x{block_size} blocks, dtype={self.dtype}, output_dtype={self.output_dtype}"
        )

    def _validate_tensor_memory_layout(self, tensor: cp.ndarray, tensor_name: str) -> None:
        """
        Validate tensor memory layout for optimal CUDA kernel performance.

        All mixed sparse tensor CUDA kernels expect C-contiguous tensors with specific
        stride patterns for optimal memory access and vectorization.

        Args:
            tensor: Input tensor to validate (CuPy array)
            tensor_name: Name for error messages

        Raises:
            ValueError: If tensor layout is incompatible with kernels
            TypeError: If tensor is not a CuPy array
        """
        # Only validate CuPy arrays (kernels only work with GPU tensors)
        if not isinstance(tensor, cp.ndarray):
            raise TypeError(f"{tensor_name} must be a cupy.ndarray for GPU kernels, got {type(tensor).__name__}")

        # Validate C-contiguous layout
        if not tensor.flags.c_contiguous:
            # Provide detailed diagnostic information
            stride_info = f"strides={tensor.strides}, shape={tensor.shape}"
            layout = "F-contiguous" if tensor.flags.f_contiguous else "non-contiguous"

            raise ValueError(
                f"Mixed sparse tensor kernel requires C-contiguous {tensor_name} tensor for optimal performance. "
                f"Got {layout} tensor with {stride_info}. "
                f"Use cp.ascontiguousarray({tensor_name}) to fix layout issues. "
                f"Non-contiguous tensors can cause significant performance degradation or incorrect results in "
                f"uint8 and fp32 CUDA kernels."
            )

        # Validate expected stride patterns for known tensor types
        if tensor_name == "target_3d":
            # target_3d should be (channels, height, width) with C-contiguous strides
            if len(tensor.shape) == 3:
                expected_strides = (
                    tensor.shape[1] * tensor.shape[2] * tensor.itemsize,  # channel stride
                    tensor.shape[2] * tensor.itemsize,  # height stride
                    tensor.itemsize,  # width stride
                )
                if tensor.strides != expected_strides:
                    raise ValueError(
                        f"target_3d has unexpected stride pattern for (channels, height, width) layout. "
                        f"Expected strides {expected_strides}, got {tensor.strides}. "
                        f"This indicates incorrect memory layout (not planar C-contiguous)."
                    )

        elif tensor_name == "sparse_values":
            # sparse_values should be (channels, batch_size, block_size, block_size) with C-contiguous strides
            if len(tensor.shape) == 4:
                expected_strides = (
                    tensor.shape[1] * tensor.shape[2] * tensor.shape[3] * tensor.itemsize,  # channel stride
                    tensor.shape[2] * tensor.shape[3] * tensor.itemsize,  # batch stride
                    tensor.shape[3] * tensor.itemsize,  # block_height stride
                    tensor.itemsize,  # block_width stride
                )
                if tensor.strides != expected_strides:
                    raise ValueError(
                        f"sparse_values has unexpected stride pattern for (channels, batch, block_h, block_w) layout. "
                        f"Expected strides {expected_strides}, got {tensor.strides}. "
                        f"This indicates incorrect memory layout (not channels-first C-contiguous)."
                    )

        elif tensor_name == "block_positions":
            # block_positions should be (channels, batch_size, 2) with C-contiguous strides
            if len(tensor.shape) == 3 and tensor.shape[2] == 2:
                expected_strides = (
                    tensor.shape[1] * tensor.shape[2] * tensor.itemsize,  # channel stride
                    tensor.shape[2] * tensor.itemsize,  # batch stride
                    tensor.itemsize,  # coordinate stride
                )
                if tensor.strides != expected_strides:
                    raise ValueError(
                        f"block_positions has unexpected stride pattern for (channels, batch, 2) layout. "
                        f"Expected strides {expected_strides}, got {tensor.strides}. "
                        f"This indicates incorrect memory layout (not channels-first C-contiguous)."
                    )

        # Check data ownership for debugging memory issues
        if not tensor.flags.owndata:
            # This is a warning, not an error, as views can work if properly contiguous
            import warnings

            warnings.warn(
                f"{tensor_name} is a tensor view (flags.owndata=False). "
                f"This may indicate upstream memory layout issues. "
                f"Consider using cp.ascontiguousarray() if experiencing problems.",
                UserWarning,
            )

    def _validate_dtype(self, dtype: cp.dtype) -> cp.dtype:
        """
        Validate and normalize the data type.

        Args:
            dtype: Data type to validate (accepts both numpy and cupy dtypes)

        Returns:
            Validated dtype (normalized to cupy dtype)

        Raises:
            ValueError: If dtype is not supported
        """
        # Normalize dtype to cupy version - handle both numpy and cupy inputs
        dtype_str = str(dtype)
        if dtype_str in ("float32", "<class 'numpy.float32'>", "numpy.float32"):
            return cp.float32
        elif dtype_str in ("uint8", "<class 'numpy.uint8'>", "numpy.uint8"):
            return cp.uint8
        elif dtype == cp.float32:
            return cp.float32
        elif dtype == cp.uint8:
            return cp.uint8
        else:
            raise ValueError(f"Unsupported dtype {dtype}. Supported types: float32, uint8 (numpy or cupy)")

    def _validate_output_dtype(self, output_dtype: cp.dtype) -> cp.dtype:
        """
        Validate and normalize the output data type.

        Args:
            output_dtype: Output data type to validate

        Returns:
            Validated output dtype (normalized to cupy dtype)

        Raises:
            ValueError: If output dtype is not supported
        """
        # Normalize dtype to cupy version
        dtype_str = str(output_dtype)
        if dtype_str in ("float32", "<class 'numpy.float32'>", "numpy.float32"):
            return cp.float32
        elif dtype_str in ("float16", "<class 'numpy.float16'>", "numpy.float16"):
            return cp.float16
        elif output_dtype == cp.float32:
            return cp.float32
        elif output_dtype == cp.float16:
            return cp.float16
        else:
            raise ValueError(
                f"Unsupported output dtype {output_dtype}. Supported types: float32, float16 (numpy or cupy)"
            )

    def set_block(
        self,
        batch_idx: int,
        channel_idx: int,
        top_left_row: int,
        top_left_col: int,
        values: cp.ndarray,
    ) -> None:
        """
        Set a dense block for a specific LED and channel.

        Args:
            batch_idx: LED index (0 to batch_size-1)
            channel_idx: Channel index (0 to channels-1)
            top_left_row: Row coordinate of block's top-left corner
            top_left_col: Column coordinate of block's top-left corner
            values: Dense block data, shape (block_size, block_size)
        """
        # Validate inputs
        if not (0 <= batch_idx < self.batch_size):
            raise ValueError(f"batch_idx {batch_idx} out of range [0, {self.batch_size})")
        if not (0 <= channel_idx < self.channels):
            raise ValueError(f"channel_idx {channel_idx} out of range [0, {self.channels})")
        if not (0 <= top_left_row <= self.height - self.block_size):
            raise ValueError(f"top_left_row {top_left_row} out of range [0, {self.height - self.block_size}]")
        if not (0 <= top_left_col <= self.width - self.block_size):
            raise ValueError(f"top_left_col {top_left_col} out of range [0, {self.width - self.block_size}]")
        if values.shape != (self.block_size, self.block_size):
            raise ValueError(f"values shape {values.shape} != expected ({self.block_size}, {self.block_size})")
        if values.dtype != self.dtype:
            raise ValueError(f"values dtype {values.dtype} != expected {self.dtype}")

        # Store the block data and position (channels-first indexing)
        self.sparse_values[channel_idx, batch_idx] = values
        self.block_positions[channel_idx, batch_idx, 0] = top_left_row
        self.block_positions[channel_idx, batch_idx, 1] = top_left_col

    def set_blocks_batch(self, positions: cp.ndarray, values: cp.ndarray) -> None:
        """
        Set multiple blocks efficiently in batch.

        Args:
            positions: Block positions, shape (channels, batch_size, 2)
                      positions[c, i, :] = [top_left_row, top_left_col]
            values: Block values, shape (channels, batch_size, block_size, block_size)
        """
        if positions.shape != (self.channels, self.batch_size, 2):
            raise ValueError(f"positions shape {positions.shape} != expected ({self.channels}, {self.batch_size}, 2)")
        if values.shape != (
            self.channels,
            self.batch_size,
            self.block_size,
            self.block_size,
        ):
            raise ValueError(
                f"values shape {values.shape} != expected "
                f"({self.channels}, {self.batch_size}, {self.block_size}, {self.block_size})"
            )
        if values.dtype != self.dtype:
            raise ValueError(f"values dtype {values.dtype} != expected {self.dtype}")

        # Validate positions are within bounds
        rows = positions[:, :, 0]
        cols = positions[:, :, 1]
        if cp.any(rows < 0) or cp.any(rows > self.height - self.block_size):
            raise ValueError("Some row positions are out of bounds")
        if cp.any(cols < 0) or cp.any(cols > self.width - self.block_size):
            raise ValueError("Some column positions are out of bounds")

        # Set all blocks at once
        self.sparse_values[:] = values
        self.block_positions[:] = positions

    def transpose_dot_product_3d(self, target_3d: cp.ndarray, output_dtype: Optional[cp.dtype] = None) -> cp.ndarray:
        """
        Compute A^T @ b operation with 3D planar input (channels, height, width).

        This method processes all channels in one operation using the optimized 3D CUDA kernel.
        Implements einsum 'ijkl,jkl->ij' efficiently where:
        - Mixed tensor: (leds, channels, height, width) - shape 'ijkl'
        - Target: (channels, height, width) - shape 'jkl' (planar form)
        - Result: (leds, channels) - shape 'ij'

        Args:
            target_3d: Target image in planar form, shape (channels, height, width)
            output_dtype: Desired output data type (cp.float32 or cp.float16).
                         If None, uses the instance's output_dtype setting.

        Returns:
            Result of A^T @ b, shape (batch_size, channels) with specified output dtype
        """
        if target_3d.shape != (self.channels, self.height, self.width):
            raise ValueError(
                f"target_3d shape {target_3d.shape} != expected ({self.channels}, {self.height}, {self.width})"
            )

        # MEMORY LAYOUT VALIDATION: Critical for kernel correctness and performance
        self._validate_tensor_memory_layout(target_3d, "target_3d")
        self._validate_tensor_memory_layout(self.sparse_values, "sparse_values")
        self._validate_tensor_memory_layout(self.block_positions, "block_positions")

        # Determine output dtype
        if output_dtype is None:
            output_dtype = self.output_dtype
        else:
            output_dtype = self._validate_output_dtype(output_dtype)

        try:
            # Route to appropriate kernel based on input dtype and desired output dtype
            if self.dtype == cp.float32:
                # Validate target dtype matches tensor dtype
                if target_3d.dtype != cp.float32:
                    raise ValueError(f"target_3d dtype {target_3d.dtype} must match tensor dtype {self.dtype}")

                if output_dtype == cp.float32:
                    from .cuda_kernels import (
                        cuda_transpose_dot_product_3d_compute_optimized,
                    )

                    # Use fp32 -> fp32 compute-optimized CUDA kernel
                    result = cuda_transpose_dot_product_3d_compute_optimized(
                        self.sparse_values,  # (channels, batch, H, W) - fp32
                        self.block_positions,  # (channels, batch, 2) - int32
                        target_3d,  # (channels, height, width) - fp32 planar input
                        self.batch_size,
                        self.channels,
                        self.block_size,
                    )
                elif output_dtype == cp.float16:
                    from .cuda_kernels import (
                        cuda_transpose_dot_product_3d_compute_optimized_fp16,
                    )

                    # Use fp32 -> fp16 compute-optimized CUDA kernel
                    result = cuda_transpose_dot_product_3d_compute_optimized_fp16(
                        self.sparse_values,  # (channels, batch, H, W) - fp32
                        self.block_positions,  # (channels, batch, 2) - int32
                        target_3d,  # (channels, height, width) - fp32 planar input
                        self.batch_size,
                        self.channels,
                        self.block_size,
                    )
                else:
                    raise ValueError(f"Unsupported output dtype {output_dtype} for FP32 input")

            elif self.dtype == cp.uint8:
                # Validate target dtype matches tensor dtype
                if target_3d.dtype != cp.uint8:
                    raise ValueError(f"target_3d dtype {target_3d.dtype} must match tensor dtype {self.dtype}")

                if output_dtype == cp.float32:
                    from .cuda_kernels import (
                        cuda_transpose_dot_product_3d_compute_optimized_int8,
                    )

                    # Use int8 -> fp32 compute-optimized CUDA kernel
                    result = cuda_transpose_dot_product_3d_compute_optimized_int8(
                        self.sparse_values,  # (channels, batch, H, W) - uint8
                        self.block_positions,  # (channels, batch, 2) - int32
                        target_3d,  # (channels, height, width) - uint8 planar input
                        self.batch_size,
                        self.channels,
                        self.block_size,
                    )
                elif output_dtype == cp.float16:
                    from .cuda_kernels import (
                        cuda_transpose_dot_product_3d_compute_optimized_int8_fp16,
                    )

                    # Use int8 -> fp16 compute-optimized CUDA kernel (main use case)
                    result = cuda_transpose_dot_product_3d_compute_optimized_int8_fp16(
                        self.sparse_values,  # (channels, batch, H, W) - uint8
                        self.block_positions,  # (channels, batch, 2) - int32
                        target_3d,  # (channels, height, width) - uint8 planar input
                        self.batch_size,
                        self.channels,
                        self.block_size,
                    )
                else:
                    raise ValueError(f"Unsupported output dtype {output_dtype} for INT8 input")

            else:
                raise ValueError(f"Unsupported input dtype {self.dtype} for CUDA kernel")

            return result

        except ImportError as e:
            logger.warning(f"3D CUDA kernel not available: {e}. No fallback available.")
            raise ImportError(
                "CUDA kernels are required for transpose_dot_product_3d operation. "
                "The previous transpose_dot_product fallback has been removed."
            ) from e

    def forward_pass_3d(self, led_values: cp.ndarray) -> cp.ndarray:
        """
        Compute A @ x operation (forward pass) with LED values input.

        This method implements the forward pass: A @ led_values -> rendered_frame
        where A is the diffusion matrix and led_values are the LED brightness values.

        Args:
            led_values: LED brightness values, shape (batch_size, channels)

        Returns:
            Rendered frame in planar form, shape (channels, height, width)
        """
        if led_values.shape != (self.batch_size, self.channels):
            raise ValueError(f"led_values shape {led_values.shape} != expected ({self.batch_size}, {self.channels})")

        # Initialize output frame
        output_frame = cp.zeros((self.channels, self.height, self.width), dtype=cp.float32)

        # Process each LED and channel
        for led_idx in range(self.batch_size):
            for channel in range(self.channels):
                # Get LED brightness for this channel
                led_brightness = led_values[led_idx, channel]

                # Skip if brightness is zero (optimization)
                if led_brightness == 0.0:
                    continue

                # Get block position for this LED and channel
                top_row = int(self.block_positions[channel, led_idx, 0])
                top_col = int(self.block_positions[channel, led_idx, 1])

                # Get the diffusion pattern block
                pattern_block = self.sparse_values[channel, led_idx]  # Shape: (block_size, block_size)

                # Add weighted pattern to output frame
                bottom_row = min(top_row + self.block_size, self.height)
                right_col = min(top_col + self.block_size, self.width)

                # Handle boundary clipping
                pattern_height = bottom_row - top_row
                pattern_width = right_col - top_col

                if pattern_height > 0 and pattern_width > 0:
                    # Apply scaling to match the kernel normalization
                    if self.dtype == cp.uint8:
                        # For uint8 A matrix: scale by 255 (only A is uint8, led_values are float32 [0,1])
                        scale_factor = led_brightness / 255.0
                    else:
                        # For float32 data, apply direct scaling
                        scale_factor = led_brightness

                    output_frame[channel, top_row:bottom_row, top_col:right_col] += (
                        pattern_block[:pattern_height, :pattern_width] * scale_factor
                    )

        return output_frame

    def to_array(self, batch_idx: int, channel_idx: int) -> cp.ndarray:
        """
        Convert a specific sub-tensor to dense array format for debugging/visualization.

        Args:
            batch_idx: LED index
            channel_idx: Channel index

        Returns:
            Dense array, shape (height, width)
        """
        # Create dense array and place block (all blocks assumed to be set)
        # Use the same dtype as the stored data
        dense = cp.zeros((self.height, self.width), dtype=self.dtype)

        top_row = int(self.block_positions[channel_idx, batch_idx, 0])  # Use channels-first indexing
        top_col = int(self.block_positions[channel_idx, batch_idx, 1])  # Use channels-first indexing

        dense[top_row : top_row + self.block_size, top_col : top_col + self.block_size] = self.sparse_values[
            channel_idx, batch_idx
        ]  # Use channels-first indexing

        return dense

    def get_block_info(self, batch_idx: int, channel_idx: int) -> Dict[str, Union[Tuple[int, int], cp.ndarray, bool]]:
        """
        Get information about a specific block.

        Args:
            batch_idx: LED index
            channel_idx: Channel index

        Returns:
            Dictionary with block information
        """
        # All blocks are assumed to be set
        position = (
            int(self.block_positions[channel_idx, batch_idx, 0]),  # Use channels-first indexing
            int(self.block_positions[channel_idx, batch_idx, 1]),  # Use channels-first indexing
        )

        return {
            "is_set": True,
            "position": position,
            "values": self.sparse_values[channel_idx, batch_idx].copy(),  # Use channels-first indexing
        }

    def memory_info(self) -> Dict[str, Union[float, int]]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory information
        """
        # Calculate actual memory usage
        sparse_values_mb = self.sparse_values.nbytes / (1024 * 1024)
        positions_mb = self.block_positions.nbytes / (1024 * 1024)
        total_mb = sparse_values_mb + positions_mb

        # Calculate what equivalent dense storage would be
        dense_mb = (self.batch_size * self.channels * self.height * self.width * 4) / (1024 * 1024)

        compression_ratio = total_mb / dense_mb
        blocks_stored = self.batch_size * self.channels  # All blocks assumed to be set

        return {
            "sparse_values_mb": sparse_values_mb,
            "positions_mb": positions_mb,
            "total_mb": total_mb,
            "equivalent_dense_mb": dense_mb,
            "compression_ratio": compression_ratio,
            "blocks_stored": blocks_stored,
            "total_possible_blocks": self.batch_size * self.channels,
        }

    def validate_consistency(self) -> bool:
        """
        Validate internal consistency of the tensor.

        Returns:
            True if consistent, False otherwise
        """
        # Check that all block positions are valid (all blocks assumed to be set)
        # Check row bounds for all positions
        if cp.any(self.block_positions[:, :, 0] < 0) or cp.any(
            self.block_positions[:, :, 0] > self.height - self.block_size
        ):
            logger.error("Some block positions have invalid row coordinates")
            return False

        # Check column bounds for all positions
        if cp.any(self.block_positions[:, :, 1] < 0) or cp.any(
            self.block_positions[:, :, 1] > self.width - self.block_size
        ):
            logger.error("Some block positions have invalid column coordinates")
            return False

        logger.debug("Tensor consistency validation passed")
        return True

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Export tensor data to a dictionary of numpy arrays for saving to npz files.

        Returns:
            Dictionary containing all tensor data as numpy arrays
        """
        # Convert CuPy arrays to NumPy for disk storage
        data_dict = {
            "sparse_values": cp.asnumpy(self.sparse_values),
            "block_positions": cp.asnumpy(self.block_positions),
            # Metadata
            "batch_size": np.array(self.batch_size, dtype=np.int32),
            "channels": np.array(self.channels, dtype=np.int32),
            "height": np.array(self.height, dtype=np.int32),
            "width": np.array(self.width, dtype=np.int32),
            "block_size": np.array(self.block_size, dtype=np.int32),
            "device": np.array(self.device, dtype="U10"),  # Unicode string
            "dtype": np.array(self.dtype.__name__, dtype="U10"),  # Store dtype name as string
            "output_dtype": np.array(self.output_dtype.__name__, dtype="U10"),  # Store output dtype name
        }

        logger.debug(f"Exported tensor to dict with {len(data_dict)} arrays")
        return data_dict

    @classmethod
    def from_dict(cls, data_dict: Dict[str, np.ndarray], device: str = "cuda") -> "SingleBlockMixedSparseTensor":
        """
        Create tensor from a dictionary of numpy arrays (loaded from npz files).

        Args:
            data_dict: Dictionary containing tensor data as numpy arrays
            device: Device to create tensor on ('cuda' or 'cpu')

        Returns:
            New SingleBlockMixedSparseTensor instance
        """
        # Extract metadata
        batch_size = int(data_dict["batch_size"])
        channels = int(data_dict["channels"])
        height = int(data_dict["height"])
        width = int(data_dict["width"])
        block_size = int(data_dict["block_size"])

        # Handle dtype - backward compatibility with files that don't have dtype
        if "dtype" in data_dict:
            dtype_str = str(data_dict["dtype"])
            if dtype_str == "float32":
                dtype = cp.float32
            elif dtype_str == "uint8":
                dtype = cp.uint8
            elif dtype_str.startswith("<class"):  # Truncated old format - infer from data
                # Fallback to infer from data for truncated strings
                dtype = cp.dtype(data_dict["sparse_values"].dtype)
            else:
                # Fallback to infer from data
                dtype = cp.dtype(data_dict["sparse_values"].dtype)
        else:
            # Backward compatibility: infer from sparse_values dtype
            dtype = cp.dtype(data_dict["sparse_values"].dtype)

        # Handle output_dtype - backward compatibility with files that don't have output_dtype
        output_dtype = None
        if "output_dtype" in data_dict:
            output_dtype_str = str(data_dict["output_dtype"])
            if output_dtype_str == "float32":
                output_dtype = cp.float32
            elif output_dtype_str == "float16":
                output_dtype = cp.float16
            elif output_dtype_str.startswith("<class"):
                # Fallback to None for old format
                output_dtype = None

        # Create new tensor instance
        tensor = cls(batch_size, channels, height, width, block_size, device, dtype, output_dtype)

        # Load data arrays, converting to CuPy if needed
        tensor.sparse_values = cp.asarray(data_dict["sparse_values"])
        tensor.block_positions = cp.asarray(data_dict["block_positions"])

        # Handle backward compatibility: if blocks_set exists in data, ignore it
        # All blocks are now assumed to be set

        # Validate consistency
        if not tensor.validate_consistency():
            raise ValueError("Loaded tensor data is inconsistent")

        blocks_loaded = tensor.batch_size * tensor.channels  # All blocks assumed to be set
        logger.debug(f"Loaded tensor from dict with {blocks_loaded} blocks")

        return tensor

    def to_dense_patterns(self) -> np.ndarray:
        """
        Convert mixed tensor to dense patterns array for visualization.

        This method converts the sparse block representation back to dense
        4D arrays suitable for visualization and analysis. Each LED/channel
        pattern is materialized as a (height, width) array.

        Returns:
            Dense patterns array of shape (batch_size, height, width, channels)
            with dtype float32 and values in range [0, 1]
        """
        # Initialize dense patterns array (HWC format for compatibility)
        patterns = np.zeros((self.batch_size, self.height, self.width, self.channels), dtype=np.float32)

        logger.debug(f"Converting {self.batch_size} LED patterns from mixed tensor to dense format...")

        # Convert each block to dense format
        for batch_idx in range(self.batch_size):
            for channel_idx in range(self.channels):
                # Get block position and values (channels-first indexing)
                top_row = int(self.block_positions[channel_idx, batch_idx, 0])
                top_col = int(self.block_positions[channel_idx, batch_idx, 1])
                block_values = cp.asnumpy(self.sparse_values[channel_idx, batch_idx])

                # Place block in dense pattern
                patterns[
                    batch_idx,
                    top_row : top_row + self.block_size,
                    top_col : top_col + self.block_size,
                    channel_idx,
                ] = block_values

        logger.debug(f"Converted to dense patterns: {patterns.shape}")
        return patterns

    def get_block_summary(self) -> Dict[str, Union[float, int, np.ndarray]]:
        """
        Get comprehensive summary statistics for all blocks.

        Computes various statistics across all LEDs and channels including
        block positions, intensity distributions, and coverage statistics.

        Returns:
            Dictionary containing:
            - batch_size: Number of LEDs
            - channels: Number of color channels
            - block_size: Size of each block (square)
            - total_blocks: Total number of blocks
            - block_positions_stats: Statistics about block placement
            - intensity_stats: Statistics about block values
            - coverage_stats: Statistics about spatial coverage
            - memory_mb: Memory usage in megabytes
        """
        logger.debug("Computing block summary statistics...")

        # Convert to numpy for statistics computation
        positions = cp.asnumpy(self.block_positions)  # (channels, batch_size, 2)
        values = cp.asnumpy(self.sparse_values)  # (channels, batch_size, block_size, block_size)

        # Basic info
        summary = {
            "batch_size": self.batch_size,
            "channels": self.channels,
            "block_size": self.block_size,
            "total_blocks": self.batch_size * self.channels,
        }

        # Block position statistics
        all_positions = positions.reshape(-1, 2)  # Flatten to (total_blocks, 2)
        position_stats = {
            "min_row": int(np.min(all_positions[:, 0])),
            "max_row": int(np.max(all_positions[:, 0])),
            "min_col": int(np.min(all_positions[:, 1])),
            "max_col": int(np.max(all_positions[:, 1])),
            "mean_row": float(np.mean(all_positions[:, 0])),
            "mean_col": float(np.mean(all_positions[:, 1])),
            "std_row": float(np.std(all_positions[:, 0])),
            "std_col": float(np.std(all_positions[:, 1])),
        }

        # Block intensity statistics
        all_values = values.reshape(-1)  # Flatten all block values
        nonzero_values = all_values[all_values > 0]
        intensity_stats = {
            "total_values": len(all_values),
            "nonzero_values": len(nonzero_values),
            "sparsity_ratio": len(nonzero_values) / len(all_values),
            "min_intensity": float(np.min(all_values)),
            "max_intensity": float(np.max(all_values)),
            "mean_intensity": float(np.mean(all_values)),
            "std_intensity": float(np.std(all_values)),
        }

        # Per-LED statistics
        led_max_intensities = np.zeros(self.batch_size, dtype=np.float32)
        led_mean_intensities = np.zeros(self.batch_size, dtype=np.float32)

        for batch_idx in range(self.batch_size):
            led_values = values[:, batch_idx].flatten()  # All channels for this LED
            led_max_intensities[batch_idx] = np.max(led_values)
            led_mean_intensities[batch_idx] = np.mean(led_values[led_values > 0]) if np.any(led_values > 0) else 0.0

        # Coverage statistics (spatial distribution)
        coverage_stats = {
            "blocks_per_led": self.channels,
            "spatial_coverage_x": (position_stats["max_col"] - position_stats["min_col"] + self.block_size)
            / self.width,
            "spatial_coverage_y": (position_stats["max_row"] - position_stats["min_row"] + self.block_size)
            / self.height,
            "led_max_intensities": led_max_intensities,
            "led_mean_intensities": led_mean_intensities,
        }

        # Add computed statistics
        summary.update(
            {
                "block_positions_stats": position_stats,
                "intensity_stats": intensity_stats,
                "coverage_stats": coverage_stats,
                "memory_mb": self.memory_info()["total_mb"],
            }
        )

        logger.debug("Block summary computed")
        return summary

    def extract_pattern(self, led_idx: int, channel_idx: int) -> np.ndarray:
        """
        Extract dense pattern for specific LED and channel.

        This method extracts and returns a single LED/channel pattern
        as a dense array, suitable for visualization or analysis.

        Args:
            led_idx: LED index (0 to batch_size-1)
            channel_idx: Channel index (0 to channels-1)

        Returns:
            Dense pattern array of shape (height, width) with dtype float32
            containing the LED pattern for the specified channel
        """
        # Validate inputs
        if not (0 <= led_idx < self.batch_size):
            raise ValueError(f"led_idx {led_idx} out of range [0, {self.batch_size})")
        if not (0 <= channel_idx < self.channels):
            raise ValueError(f"channel_idx {channel_idx} out of range [0, {self.channels})")

        # Use existing to_array method but convert to numpy
        dense_pattern = cp.asnumpy(self.to_array(led_idx, channel_idx))

        logger.debug(f"Extracted pattern for LED {led_idx}, channel {channel_idx}: shape {dense_pattern.shape}")
        return dense_pattern

    def compute_ata_dense(self) -> np.ndarray:
        """
        Compute A^T A matrix directly from block data avoiding CSC conversion issues.

        This method computes the (led_count, led_count, channels) A^T A matrix
        by directly computing dot products between overlapping block regions.
        This gives us the exact sparsity pattern we expect based on block adjacency.

        Returns:
            Dense A^T A array of shape (led_count, led_count, channels) where
            ata[i, j, c] = dot(pattern_i_c, pattern_j_c) for overlapping regions
        """
        logger.info("Computing A^T A directly from block data...")

        # Initialize A^T A matrix: (led_count, led_count, channels)
        ata_matrix = np.zeros((self.batch_size, self.batch_size, self.channels), dtype=np.float32)

        # Convert block data to numpy for computation
        block_values = cp.asnumpy(self.sparse_values)  # (channels, batch_size, block_size, block_size)
        block_positions = cp.asnumpy(self.block_positions)  # (channels, batch_size, 2)

        total_pairs = self.batch_size * (self.batch_size + 1) // 2
        computed_pairs = 0

        # Compute A^T A for each pair of LEDs
        for led_i in range(self.batch_size):
            for led_j in range(led_i, self.batch_size):  # Only upper triangle + diagonal
                # For each RGB channel
                for channel in range(self.channels):
                    # Get block positions and values for both LEDs
                    pos_i = block_positions[channel, led_i]  # (2,) [row, col]
                    pos_j = block_positions[channel, led_j]  # (2,) [row, col]

                    values_i = block_values[channel, led_i]  # (block_size, block_size)
                    values_j = block_values[channel, led_j]  # (block_size, block_size)

                    # Calculate block extents
                    r1_start, c1_start = int(pos_i[0]), int(pos_i[1])
                    r1_end = r1_start + self.block_size
                    c1_end = c1_start + self.block_size

                    r2_start, c2_start = int(pos_j[0]), int(pos_j[1])
                    r2_end = r2_start + self.block_size
                    c2_end = c2_start + self.block_size

                    # Find overlapping region
                    overlap_r_start = max(r1_start, r2_start)
                    overlap_r_end = min(r1_end, r2_end)
                    overlap_c_start = max(c1_start, c2_start)
                    overlap_c_end = min(c1_end, c2_end)

                    # Check if blocks overlap
                    if overlap_r_start < overlap_r_end and overlap_c_start < overlap_c_end:
                        # Extract overlapping regions from both blocks
                        # Convert global coordinates to local block coordinates
                        local_r1_start = overlap_r_start - r1_start
                        local_r1_end = overlap_r_end - r1_start
                        local_c1_start = overlap_c_start - c1_start
                        local_c1_end = overlap_c_end - c1_start

                        local_r2_start = overlap_r_start - r2_start
                        local_r2_end = overlap_r_end - r2_start
                        local_c2_start = overlap_c_start - c2_start
                        local_c2_end = overlap_c_end - c2_start

                        # Extract overlapping regions
                        overlap_i = values_i[local_r1_start:local_r1_end, local_c1_start:local_c1_end]
                        overlap_j = values_j[local_r2_start:local_r2_end, local_c2_start:local_c2_end]

                        # Compute dot product: sum of element-wise multiplication
                        dot_product = np.sum(overlap_i * overlap_j)

                        # Store in A^T A matrix
                        ata_matrix[led_i, led_j, channel] = dot_product
                        if led_i != led_j:  # Symmetric matrix
                            ata_matrix[led_j, led_i, channel] = dot_product

                computed_pairs += 1
                if computed_pairs % 10000 == 0:
                    progress = computed_pairs / total_pairs * 100
                    logger.info(f"  Computed {computed_pairs}/{total_pairs} LED pairs ({progress:.1f}%)")

        # Count non-zeros per channel for validation
        for channel in range(self.channels):
            nnz = np.count_nonzero(ata_matrix[:, :, channel])
            sparsity = nnz / (self.batch_size**2) * 100
            logger.info(f"  Channel {channel + 1}: {nnz} non-zeros ({sparsity:.2f}% dense)")

        logger.info(f"A^T A computation complete: shape {ata_matrix.shape}")
        return ata_matrix

    def __repr__(self) -> str:
        """String representation of the tensor."""
        blocks_stored = self.batch_size * self.channels  # All blocks assumed to be set
        memory_mb = self.memory_info()["total_mb"]

        return (
            f"SingleBlockMixedSparseTensor("
            f"shape=({self.batch_size}, {self.channels}, {self.height}, {self.width}), "
            f"block_size={self.block_size}, "
            f"dtype={self.dtype}, "
            f"output_dtype={self.output_dtype}, "
            f"blocks_stored={blocks_stored}/{self.batch_size * self.channels}, "
            f"memory={memory_mb:.1f}MB)"
        )
