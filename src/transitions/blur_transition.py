"""
Blur transition implementation for playlist items.

This module implements blur-in and blur-out transitions that apply
Gaussian blur effects to frames at the beginning or end of playlist items.
All operations are performed on GPU for optimal performance.
"""

import contextlib
import logging
import math
import time
from typing import Any, Dict, Tuple, Union

import cupy as cp
import cv2
import numpy as np

from .base_transition import BaseTransition

logger = logging.getLogger(__name__)


class BlurTransition(BaseTransition):
    """
    Blur transition implementation with GPU acceleration.

    Provides smooth blur-in and blur-out effects by applying Gaussian blur
    with varying intensity over a specified duration. Transition in starts
    with strong blur and reduces to clear image. Transition out does the reverse.

    All operations are performed on GPU using CuPy for optimal performance.
    Optimized for real-time use with <5ms target performance.
    """

    # Class-level type annotations for dynamically created attributes
    _last_log_time: float

    def __init__(self):
        """Initialize blur transition with pre-compiled kernels."""
        super().__init__()
        self._kernel_cache = {}
        self._precompile_kernels()

    def _precompile_kernels(self):
        """
        Pre-compile common blur kernels to avoid first-frame overhead.
        """
        try:
            # Import and test CuPy availability
            import cupyx.scipy.ndimage

            # Pre-create test frame and common kernels
            test_frame = cp.ones((64, 64, 3), dtype=cp.float32)

            # Pre-compile common blur sizes
            common_sigmas = [1.0, 2.0, 3.0, 5.0, 7.0]
            for sigma in common_sigmas:
                with contextlib.suppress(Exception):
                    # This will compile the kernels
                    cupyx.scipy.ndimage.gaussian_filter(test_frame, sigma=sigma, mode="nearest")

            # Pre-cache common 1D Gaussian kernels
            common_radii = [3, 5, 10, 15, 20, 25]
            for radius in common_radii:
                kernel_size = radius * 2 + 1
                sigma = radius / 3.0

                x = cp.arange(kernel_size, dtype=cp.float32) - kernel_size // 2
                kernel_1d = cp.exp(-0.5 * (x / sigma) ** 2)
                kernel_1d = kernel_1d / cp.sum(kernel_1d)

                cache_key = f"{kernel_size}_{sigma:.2f}"
                self._kernel_cache[cache_key] = kernel_1d

            logger.info(f"Pre-compiled {len(self._kernel_cache)} blur kernels for optimal performance")

        except Exception as e:
            logger.warning(f"Kernel pre-compilation failed: {e}")
            self._kernel_cache = {}

    def apply_transition(
        self,
        frame: cp.ndarray,
        timestamp: float,
        item_duration: float,
        transition_config: Dict[str, Any],
        direction: str,
    ) -> cp.ndarray:
        """
        Apply blur transition to a frame.

        Args:
            frame: RGB frame data as cupy GPU array (H, W, 3) with values 0-255
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration with parameters
            direction: "in" for blur-in, "out" for blur-out

        Returns:
            Frame with blur transition applied (GPU array)

        Raises:
            ValueError: If parameters are invalid or frame is not on GPU
            RuntimeError: If transition processing fails
        """
        try:
            # Validate inputs - frame must be GPU array
            if not isinstance(frame, cp.ndarray):
                logger.error(f"Expected GPU cupy array, got {type(frame)}")
                raise ValueError(f"Frame must be cupy GPU array, got {type(frame)}")

            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}")

            if not self.validate_parameters(transition_config.get("parameters", {})):
                raise ValueError("Invalid blur transition parameters")

            # Check if we're in the transition region
            if not self.is_in_transition_region(timestamp, item_duration, transition_config, direction):
                return frame  # No transition needed

            # Get transition parameters
            parameters = transition_config.get("parameters", {})
            max_blur_radius = parameters.get("max_blur_radius", 20.0)
            curve_type = parameters.get("curve", "linear")

            # Calculate blur progress (0.0 to 1.0)
            progress = self.get_transition_progress(timestamp, item_duration, transition_config, direction)

            # Apply curve transformation
            curve_progress = self._apply_curve(progress, curve_type)

            # Calculate blur radius based on direction
            if direction == "in":
                # Blur in: start at max blur, end at no blur
                blur_radius = max_blur_radius * (1.0 - curve_progress)
            elif direction == "out":
                # Blur out: start at no blur, end at max blur
                blur_radius = max_blur_radius * curve_progress
            else:
                raise ValueError(f"Invalid direction '{direction}', must be 'in' or 'out'")

            # Skip blur if radius is very small (optimization)
            if blur_radius < 1.0:
                return frame

            # Apply GPU-only Gaussian blur with timing
            start_time = time.perf_counter()
            blurred_frame = self._gpu_only_blur(frame, blur_radius)
            blur_time = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds

            # Log performance metrics (throttled to avoid spam)
            if not hasattr(self, "_last_log_time") or (time.time() - self._last_log_time) > 5.0:
                logger.info(
                    f"Blur transition: radius={blur_radius:.1f}px, time={blur_time:.2f}ms, "
                    f"frame_shape={frame.shape}, direction={direction}"
                )
                self._last_log_time = time.time()

            return blurred_frame

        except Exception as e:
            logger.error(f"Error applying blur transition: {e}")
            # Return original frame on error to avoid breaking the pipeline
            return frame

    def _gpu_only_blur(self, frame: cp.ndarray, blur_radius: float) -> cp.ndarray:
        """
        Apply Gaussian blur using GPU-only processing - strict GPU input/output.

        Args:
            frame: GPU cupy array (H, W, 3)
            blur_radius: Blur radius in pixels

        Returns:
            Blurred frame as GPU cupy array

        Raises:
            ValueError: If frame is not a GPU cupy array
        """
        # Validate input is GPU array
        if not isinstance(frame, cp.ndarray):
            logger.error(f"Expected GPU cupy array, got {type(frame)}")
            raise ValueError(f"Frame must be cupy GPU array, got {type(frame)}")

        # Pre-calculate kernel parameters
        kernel_size = int(blur_radius * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, min(kernel_size, 101))
        sigma = blur_radius / 3.0

        # Choose fastest GPU algorithm based on kernel size
        if kernel_size <= 11:
            # Small kernels: use CuPy's optimized ndimage
            result = self._fast_ndimage_blur(frame, sigma)
        else:
            # Larger kernels: use optimized separable convolution
            result = self._optimized_separable_blur(frame, kernel_size, sigma)

        return result

    def _fast_ndimage_blur(self, gpu_frame: cp.ndarray, sigma: float) -> cp.ndarray:
        """
        Ultra-fast blur for small kernels using CuPy's optimized ndimage.
        Pre-compiled for zero overhead after first use.
        """
        try:
            from cupyx.scipy import ndimage

            # Apply Gaussian filter efficiently
            # Process all channels at once for better performance
            result = ndimage.gaussian_filter(gpu_frame.astype(cp.float32), sigma=sigma, mode="nearest")

            return cp.clip(result, 0, 255).astype(cp.uint8)

        except ImportError as err:
            logger.error("cupyx.scipy not available - required for GPU blur processing")
            raise RuntimeError("cupyx.scipy required for GPU-native blur transition") from err

    def _optimized_separable_blur(self, gpu_frame: cp.ndarray, kernel_size: int, sigma: float) -> cp.ndarray:
        """
        Highly optimized separable Gaussian blur using pre-compiled CuPy operations.
        """
        # Create 1D Gaussian kernel (cached for performance)
        kernel_cache_key = f"{kernel_size}_{sigma:.2f}"
        if not hasattr(self, "_kernel_cache"):
            self._kernel_cache = {}

        if kernel_cache_key not in self._kernel_cache:
            x = cp.arange(kernel_size, dtype=cp.float32) - kernel_size // 2
            kernel_1d = cp.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d = kernel_1d / cp.sum(kernel_1d)
            self._kernel_cache[kernel_cache_key] = kernel_1d
        else:
            kernel_1d = self._kernel_cache[kernel_cache_key]

        # Use CuPy's optimized convolution operations
        h, w, c = gpu_frame.shape

        # Convert to float32 for processing
        frame_float = gpu_frame.astype(cp.float32)

        # Apply separable convolution using scipy.ndimage (much faster)
        try:
            from cupyx.scipy import ndimage

            # Horizontal blur
            h_kernel = kernel_1d.reshape(1, -1)
            h_blurred = ndimage.convolve(frame_float, h_kernel[:, :, cp.newaxis], mode="nearest")

            # Vertical blur
            v_kernel = kernel_1d.reshape(-1, 1)
            result = ndimage.convolve(h_blurred, v_kernel[:, :, cp.newaxis], mode="nearest")

        except ImportError as err:
            logger.error("cupyx.scipy not available - required for GPU blur processing")
            raise RuntimeError("cupyx.scipy required for GPU-native blur transition") from err

        return cp.clip(result, 0, 255).astype(cp.uint8)

    def _manual_separable_blur(self, frame_float: cp.ndarray, kernel_1d: cp.ndarray) -> cp.ndarray:
        """
        Manual separable convolution as fallback if scipy not available.
        """
        h, w, c = frame_float.shape
        kernel_size = len(kernel_1d)
        pad_size = kernel_size // 2

        # Horizontal convolution
        h_result = cp.zeros_like(frame_float)
        for i in range(kernel_size):
            shift = i - pad_size
            if shift < 0:
                src_slice = slice(0, w + shift)
                dst_slice = slice(-shift, w)
            elif shift > 0:
                src_slice = slice(shift, w)
                dst_slice = slice(0, w - shift)
            else:
                src_slice = slice(0, w)
                dst_slice = slice(0, w)

            h_result[:, dst_slice, :] += frame_float[:, src_slice, :] * kernel_1d[i]

        # Vertical convolution
        v_result = cp.zeros_like(frame_float)
        for i in range(kernel_size):
            shift = i - pad_size
            if shift < 0:
                src_slice = slice(0, h + shift)
                dst_slice = slice(-shift, h)
            elif shift > 0:
                src_slice = slice(shift, h)
                dst_slice = slice(0, h - shift)
            else:
                src_slice = slice(0, h)
                dst_slice = slice(0, h)

            v_result[dst_slice, :, :] += h_result[src_slice, :, :] * kernel_1d[i]

        return v_result

    def _optimized_gaussian_blur_gpu(self, gpu_frame: cp.ndarray, kernel_size: int, sigma: float) -> cp.ndarray:
        """
        Apply highly optimized Gaussian blur on GPU using separable convolution with CuPy.

        Args:
            gpu_frame: Frame data on GPU
            kernel_size: Size of the Gaussian kernel (odd number)
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Blurred frame on GPU
        """
        # Choose the best algorithm based on blur radius and performance testing
        if kernel_size <= 7:  # Very small blur: use CuPy's optimized ndimage
            return self._fast_small_blur_gpu(gpu_frame, kernel_size, sigma)
        elif kernel_size <= 25:  # Small-medium blur: use separable convolution
            return self._separable_blur_gpu(gpu_frame, kernel_size, sigma)
        else:  # Very large blur: fall back to CPU only for extreme cases
            # For typical transition blurs (20-50px), let's try GPU first
            if kernel_size <= 51:  # Up to ~25px radius
                logger.debug(f"Large blur (kernel_size={kernel_size}), trying GPU separable convolution")
                return self._separable_blur_gpu(gpu_frame, kernel_size, sigma)
            else:  # Extremely large blur
                logger.debug(f"Extremely large blur (kernel_size={kernel_size}), delegating to CPU")
                raise Exception("Extremely large blur - use CPU fallback")

    def _fast_small_blur_gpu(self, gpu_frame: cp.ndarray, kernel_size: int, sigma: float) -> cp.ndarray:
        """
        Fast GPU blur for small kernel sizes using CuPy's optimized filtering.
        """
        try:
            # Use CuPy's ndimage for small kernels (much faster)
            from cupyx.scipy import ndimage

            # Apply Gaussian filter to each channel separately for better performance
            h, w, c = gpu_frame.shape
            result = cp.empty_like(gpu_frame)

            for ch in range(c):
                result[:, :, ch] = ndimage.gaussian_filter(
                    gpu_frame[:, :, ch].astype(cp.float32), sigma=sigma, mode="nearest"
                )

            return cp.clip(result, 0, 255).astype(gpu_frame.dtype)

        except ImportError:
            # Fallback if cupyx.scipy not available
            logger.warning("cupyx.scipy not available, using basic convolution")
            return self._separable_blur_gpu(gpu_frame, kernel_size, sigma)

    def _separable_blur_gpu(self, gpu_frame: cp.ndarray, kernel_size: int, sigma: float) -> cp.ndarray:
        """
        Optimized separable Gaussian blur using vectorized operations.
        """
        # Create 1D Gaussian kernel
        x = cp.arange(kernel_size, dtype=cp.float32) - kernel_size // 2
        kernel_1d = cp.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / cp.sum(kernel_1d)

        # Convert to float32 for better GPU performance
        frame_float = gpu_frame.astype(cp.float32)

        # Pad the frame for convolution
        pad_size = kernel_size // 2
        padded = cp.pad(frame_float, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="edge")

        # Apply horizontal blur using optimized convolution
        h_blurred = self._convolve_horizontal_gpu(padded, kernel_1d)

        # Apply vertical blur using optimized convolution
        v_blurred = self._convolve_vertical_gpu(h_blurred, kernel_1d)

        # Extract valid region (remove padding)
        result = v_blurred[pad_size:-pad_size, pad_size:-pad_size, :]

        return cp.clip(result, 0, 255).astype(gpu_frame.dtype)

    def _convolve_horizontal_gpu(self, padded_frame: cp.ndarray, kernel: cp.ndarray) -> cp.ndarray:
        """
        Fast horizontal convolution using vectorized operations.
        """
        h, w, c = padded_frame.shape
        kernel_size = len(kernel)
        pad_size = kernel_size // 2

        # Create output array
        result = cp.zeros((h, w - 2 * pad_size, c), dtype=cp.float32)

        # Vectorized convolution across all rows and channels simultaneously
        for i in range(kernel_size):
            start_col = i
            end_col = w - kernel_size + i + 1
            result += padded_frame[:, start_col:end_col, :] * kernel[i]

        return result

    def _convolve_vertical_gpu(self, frame: cp.ndarray, kernel: cp.ndarray) -> cp.ndarray:
        """
        Fast vertical convolution using vectorized operations.
        """
        h, w, c = frame.shape
        kernel_size = len(kernel)
        pad_size = kernel_size // 2

        # Create output array
        result = cp.zeros((h - 2 * pad_size, w, c), dtype=cp.float32)

        # Vectorized convolution across all columns and channels simultaneously
        for i in range(kernel_size):
            start_row = i
            end_row = h - kernel_size + i + 1
            result += frame[start_row:end_row, :, :] * kernel[i]

        return result

    def _create_cuda_blur_kernel(self) -> bool:
        """
        Create a custom CUDA kernel for ultra-fast box blur approximation.

        Box blur can approximate Gaussian blur when applied multiple times
        and is much faster for large kernel sizes.
        """
        if hasattr(self, "_cuda_kernel_h"):
            return True

        cuda_code = """
        extern "C" __global__
        void box_blur_horizontal(unsigned char* input, unsigned char* output,
                               int width, int height, int channels, int radius) {
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int c = blockIdx.z;

            if (x >= width || y >= height || c >= channels) return;

            int idx = (y * width + x) * channels + c;
            int sum = 0;
            int count = 0;

            // Box blur: average pixels within radius
            for (int i = max(0, x - radius); i <= min(width - 1, x + radius); i++) {
                sum += input[(y * width + i) * channels + c];
                count++;
            }

            output[idx] = sum / count;
        }

        extern "C" __global__
        void box_blur_vertical(unsigned char* input, unsigned char* output,
                             int width, int height, int channels, int radius) {
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int c = blockIdx.z;

            if (x >= width || y >= height || c >= channels) return;

            int idx = (y * width + x) * channels + c;
            int sum = 0;
            int count = 0;

            // Box blur: average pixels within radius
            for (int i = max(0, y - radius); i <= min(height - 1, y + radius); i++) {
                sum += input[(i * width + x) * channels + c];
                count++;
            }

            output[idx] = sum / count;
        }
        """

        try:
            self._cuda_kernel_h = cp.RawKernel(cuda_code, "box_blur_horizontal")
            self._cuda_kernel_v = cp.RawKernel(cuda_code, "box_blur_vertical")
            logger.info("CUDA blur kernels compiled successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to compile CUDA blur kernels: {e}")
            return False

    def _ultra_fast_blur_gpu(self, gpu_frame: cp.ndarray, blur_radius: float) -> cp.ndarray:
        """
        Ultra-fast blur using custom CUDA kernels for very large blur radii.

        Uses box blur approximation which is much faster than Gaussian for large kernels.
        """
        if not self._create_cuda_blur_kernel():
            # Fallback to separable blur
            kernel_size = int(blur_radius * 2) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            return self._separable_blur_gpu(gpu_frame, kernel_size, blur_radius / 3.0)

        h, w, c = gpu_frame.shape
        radius = int(blur_radius)

        # Create temporary buffer
        temp_buffer = cp.empty_like(gpu_frame)

        # Configure CUDA launch parameters
        threads_per_block = (16, 16)
        blocks_x = (w + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (h + threads_per_block[1] - 1) // threads_per_block[1]
        grid_size = (blocks_x, blocks_y, c)

        # Apply horizontal box blur
        self._cuda_kernel_h(grid_size, threads_per_block, (gpu_frame, temp_buffer, w, h, c, radius))

        # Apply vertical box blur
        self._cuda_kernel_v(grid_size, threads_per_block, (temp_buffer, gpu_frame, w, h, c, radius))

        # Apply 2-3 iterations for better Gaussian approximation
        iterations = 2 if blur_radius < 20 else 3
        for _ in range(iterations - 1):
            self._cuda_kernel_h(grid_size, threads_per_block, (gpu_frame, temp_buffer, w, h, c, radius))
            self._cuda_kernel_v(grid_size, threads_per_block, (temp_buffer, gpu_frame, w, h, c, radius))

        return gpu_frame

    def get_transition_region(
        self, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> Tuple[float, float]:
        """
        Calculate the time region where the blur transition is active.

        Args:
            item_duration: Total duration of the playlist item (seconds)
            transition_config: Transition configuration including parameters
            direction: "in" for blur-in, "out" for blur-out

        Returns:
            Tuple of (start_time, end_time) in seconds from item start

        Raises:
            ValueError: If parameters are invalid
        """
        parameters = transition_config.get("parameters", {})
        duration = parameters.get("duration", 1.0)

        # Clamp duration to item duration
        duration = min(duration, item_duration)

        if direction == "in":
            # Blur in at the beginning of the item
            return (0.0, duration)
        elif direction == "out":
            # Blur out at the end of the item
            return (item_duration - duration, item_duration)
        else:
            raise ValueError(f"Invalid direction '{direction}', must be 'in' or 'out'")

    def is_in_transition_region(
        self, timestamp: float, item_duration: float, transition_config: Dict[str, Any], direction: str
    ) -> bool:
        """
        Check if a timestamp falls within the blur transition region.

        Args:
            timestamp: Time within the current playlist item (seconds from item start)
            item_duration: Total duration of the current playlist item (seconds)
            transition_config: Transition configuration including parameters
            direction: "in" for blur-in, "out" for blur-out

        Returns:
            True if timestamp is within the transition region, False otherwise
        """
        try:
            start_time, end_time = self.get_transition_region(item_duration, transition_config, direction)
            return start_time <= timestamp <= end_time
        except Exception as e:
            logger.warning(f"Error checking transition region: {e}")
            return False

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate blur transition parameters.

        Args:
            parameters: Dictionary of blur transition parameters

        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # Check duration
            duration = parameters.get("duration", 1.0)
            if not isinstance(duration, (int, float)):
                logger.error(f"Duration must be a number, got {type(duration)}")
                return False
            if duration <= 0:
                logger.error(f"Duration must be positive, got {duration}")
                return False
            if duration > 60.0:  # Reasonable upper limit
                logger.error(f"Duration too large: {duration} seconds (max 60)")
                return False

            # Check max_blur_radius
            max_blur_radius = parameters.get("max_blur_radius", 20.0)
            if not isinstance(max_blur_radius, (int, float)):
                logger.error(f"max_blur_radius must be a number, got {type(max_blur_radius)}")
                return False
            if max_blur_radius <= 0:
                logger.error(f"max_blur_radius must be positive, got {max_blur_radius}")
                return False
            if max_blur_radius > 100.0:  # Reasonable upper limit
                logger.error(f"max_blur_radius too large: {max_blur_radius} pixels (max 100)")
                return False

            # Check curve type
            curve = parameters.get("curve", "linear")
            if curve not in ["linear", "ease-in", "ease-out", "ease-in-out"]:
                logger.error(f"Invalid curve type '{curve}'")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating blur parameters: {e}")
            return False

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for blur transition parameters.

        Returns:
            JSON schema dictionary for blur transition parameters
        """
        return {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 60.0,
                    "default": 1.0,
                    "description": "Blur transition duration in seconds",
                },
                "max_blur_radius": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 100.0,
                    "default": 20.0,
                    "description": "Maximum blur radius in pixels",
                },
                "curve": {
                    "type": "string",
                    "enum": ["linear", "ease-in", "ease-out", "ease-in-out"],
                    "default": "linear",
                    "description": "Interpolation curve for blur transition",
                },
            },
            "required": ["duration"],
            "additionalProperties": False,
        }

    def _apply_curve(self, progress: float, curve_type: str) -> float:
        """
        Apply interpolation curve to linear progress.

        Args:
            progress: Linear progress value between 0.0 and 1.0
            curve_type: Type of curve to apply

        Returns:
            Transformed progress value with curve applied
        """
        progress = max(0.0, min(1.0, progress))  # Clamp to valid range

        if curve_type == "linear":
            return progress
        elif curve_type == "ease-in":
            # Quadratic ease-in (slow start, fast end)
            return progress * progress
        elif curve_type == "ease-out":
            # Quadratic ease-out (fast start, slow end)
            return 1.0 - (1.0 - progress) * (1.0 - progress)
        elif curve_type == "ease-in-out":
            # Cubic ease-in-out (slow start and end, fast middle)
            if progress < 0.5:
                return 2.0 * progress * progress
            else:
                return 1.0 - 2.0 * (1.0 - progress) * (1.0 - progress)
        else:
            # Fallback to linear
            logger.warning(f"Unknown curve type '{curve_type}', using linear")
            return progress

    def benchmark_blur_performance(self, frame_shape=(480, 800, 3), test_radii=None) -> Dict[str, Dict[str, float]]:
        """
        Benchmark blur performance across different algorithms and radii.

        Args:
            frame_shape: Shape of test frame (H, W, C)
            test_radii: List of blur radii to test

        Returns:
            Dictionary with performance results
        """
        if test_radii is None:
            test_radii = [5, 10, 20, 30, 50]

        logger.info(f"Benchmarking blur performance with frame shape {frame_shape}")

        # Create test frame
        test_frame = np.random.randint(0, 255, frame_shape, dtype=np.uint8)

        results = {}

        for radius in test_radii:
            logger.info(f"Testing blur radius {radius}px...")

            # Test GPU implementation
            try:
                gpu_frame = cp.asarray(test_frame)
                start_time = time.perf_counter()
                gpu_result = self._gpu_only_blur(gpu_frame, radius)
                cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
                gpu_time = (time.perf_counter() - start_time) * 1000
                logger.info(f"  GPU blur: {gpu_time:.2f}ms")
            except Exception as e:
                gpu_time = float("inf")
                logger.warning(f"  GPU blur failed: {e}")

            # Test CPU implementation (using OpenCV)
            try:
                start_time = time.perf_counter()
                kernel_size = int(radius * 2) | 1  # Ensure odd
                sigma = radius / 3.0
                cpu_result = cv2.GaussianBlur(test_frame, (kernel_size, kernel_size), sigma)
                cpu_time = (time.perf_counter() - start_time) * 1000
                logger.info(f"  CPU blur: {cpu_time:.2f}ms")
            except Exception as e:
                cpu_time = float("inf")
                logger.warning(f"  CPU blur failed: {e}")

            speedup = cpu_time / gpu_time if gpu_time > 0 and gpu_time != float("inf") else 0

            results[f"radius_{radius}"] = {
                "gpu_time_ms": gpu_time,
                "cpu_time_ms": cpu_time,
                "speedup": speedup,
                "gpu_faster": gpu_time < cpu_time,
            }

            logger.info(f"  Speedup: {speedup:.2f}x ({'GPU' if gpu_time < cpu_time else 'CPU'} faster)")

        return results
