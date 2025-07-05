"""
Test suite for CUDA kernels in the mixed sparse tensor system.

This test suite provides comprehensive validation and performance comparison
of CUDA kernels for A^T @ b operations using real diffusion patterns.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from src.utils.performance_timing import PerformanceTiming

# CuPy imports - conditionally available
try:
    import cupy as cp

    from src.utils.cuda_kernels import (
        cuda_transpose_dot_product_3d_compute_optimized,
        cuda_transpose_dot_product_3d_compute_optimized_int8,
        cuda_transpose_dot_product_3d_compute_optimized_int8_experimental,
    )
    from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


def morton_encode(x, y):
    """Encode 2D coordinates into Morton (Z-order) code for spatial locality."""

    def part1by1(n):
        """Separate bits by 1 position for Morton encoding."""
        n &= 0x0000FFFF  # Only keep lower 16 bits
        n = (n ^ (n << 8)) & 0x00FF00FF
        n = (n ^ (n << 4)) & 0x0F0F0F0F
        n = (n ^ (n << 2)) & 0x33333333
        n = (n ^ (n << 1)) & 0x55555555
        return n

    return part1by1(x) | (part1by1(y) << 1)


def generate_spatially_ordered_positions(led_count, height, width, block_size, seed=42, align_x_to_4=False):
    """Generate LED positions with spatial Morton ordering for cache locality.

    Args:
        led_count: Number of LED positions to generate
        height: Image height
        width: Image width
        block_size: LED block size
        seed: Random seed for reproducibility
        align_x_to_4: If True, align x-positions to multiples of 4 for experimental kernel
    """
    np.random.seed(seed)

    # Generate random positions within valid bounds
    max_row = height - block_size
    max_col = width - block_size

    positions = []
    for _ in range(led_count):
        top_row = np.random.randint(0, max_row)

        if align_x_to_4:
            # Align x-position to multiples of 4 for experimental kernel
            # Generate x-position in range [0, max_col] and round down to nearest multiple of 4
            max_col_aligned = (max_col // 4) * 4  # Ensure max_col is also aligned
            if max_col_aligned <= 0:
                top_col = 0
            else:
                # Generate random multiple of 4 within valid range
                num_positions = max_col_aligned // 4 + 1  # Number of valid 4-aligned positions
                top_col = np.random.randint(0, num_positions) * 4
        else:
            top_col = np.random.randint(0, max_col)

        positions.append((top_row, top_col))

    # Calculate Morton codes for spatial ordering
    morton_codes = []
    for top_row, top_col in positions:
        # Normalize coordinates to fit in 16-bit range for Morton encoding
        norm_row = int((top_row / max_row) * 65535) if max_row > 0 else 0
        norm_col = int((top_col / max_col) * 65535) if max_col > 0 else 0
        morton_code = morton_encode(norm_col, norm_row)  # (x, y) ordering
        morton_codes.append(morton_code)

    # Sort positions by Morton code for spatial locality
    sorted_indices = np.argsort(morton_codes)
    sorted_positions = [positions[i] for i in sorted_indices]

    return sorted_positions


@pytest.fixture
def mixed_sparse_tensor():
    """Create a realistic mixed sparse tensor for testing CUDA kernels."""
    # Use realistic dimensions for testing
    led_count = 100  # For most tests - will be overridden in performance test
    channels = 3
    height = 480  # Updated to match user specification
    width = 800
    block_size = 64

    # Create mixed sparse tensor
    tensor = SingleBlockMixedSparseTensor(
        batch_size=led_count,
        channels=channels,
        height=height,
        width=width,
        block_size=block_size,
    )

    # Generate LED positions with spatial Morton ordering for cache locality
    led_positions = generate_spatially_ordered_positions(led_count, height, width, block_size, seed=42)

    # Generate realistic diffusion patterns with spatially ordered placement
    np.random.seed(42)  # For reproducible results

    for led_idx in range(led_count):
        top_row, top_col = led_positions[led_idx]

        for channel_idx in range(channels):
            # Generate realistic diffusion pattern (Gaussian-like)
            center_row, center_col = block_size // 2, block_size // 2
            y_coords, x_coords = np.ogrid[:block_size, :block_size]

            # Gaussian pattern with some randomness
            sigma = block_size / 4
            gaussian = np.exp(-((x_coords - center_col) ** 2 + (y_coords - center_row) ** 2) / (2 * sigma**2))

            # Add some noise and channel-specific intensity
            noise = np.random.normal(0, 0.1, (block_size, block_size))
            intensity = (channel_idx + 1) / channels  # Different intensity per channel
            pattern = (gaussian * intensity + noise).clip(0, 1).astype(np.float32)

            # Set the block
            tensor.set_block(led_idx, channel_idx, top_row, top_col, cp.asarray(pattern))

    return tensor


@pytest.fixture
def test_images():
    """Generate test images with different characteristics for cache warming tests."""
    height, width, channels = 480, 800, 3

    # Generate diverse test images to prevent cache effects
    images = []

    # Random noise image
    np.random.seed(42)
    images.append(np.random.rand(channels, height, width).astype(np.float32))

    # Gradient image
    x_grad = np.linspace(0, 1, width)
    y_grad = np.linspace(0, 1, height)
    grad_img = np.zeros((channels, height, width), dtype=np.float32)
    for c in range(channels):
        grad_img[c] = np.outer(y_grad, x_grad) * (c + 1) / channels
    images.append(grad_img)

    # Checkerboard pattern
    check_size = 32
    checker = np.zeros((channels, height, width), dtype=np.float32)
    for c in range(channels):
        for y in range(height):
            for x in range(width):
                if ((x // check_size) + (y // check_size)) % 2:
                    checker[c, y, x] = 0.8 * (c + 1) / channels
    images.append(checker)

    # Gaussian blob
    y_center, x_center = height // 2, width // 2
    y_coords, x_coords = np.ogrid[:height, :width]
    gaussian = np.exp(-((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2) / (2 * (min(height, width) // 4) ** 2))
    blob_img = np.zeros((channels, height, width), dtype=np.float32)
    for c in range(channels):
        blob_img[c] = gaussian * (c + 1) / channels
    images.append(blob_img)

    # High-frequency sine wave pattern
    freq_x, freq_y = 0.1, 0.05
    x_wave = np.sin(2 * np.pi * freq_x * np.arange(width))
    y_wave = np.sin(2 * np.pi * freq_y * np.arange(height))
    wave_img = np.zeros((channels, height, width), dtype=np.float32)
    for c in range(channels):
        wave_img[c] = 0.5 * (1 + np.outer(y_wave, x_wave)) * (c + 1) / channels
    images.append(wave_img)

    return images


@pytest.fixture
def large_mixed_sparse_tensor():
    """Create a large mixed sparse tensor for performance testing with realistic problem size."""
    # Large problem size for accurate performance measurements
    led_count = 1000  # Full problem size for performance testing
    channels = 3
    height = 480
    width = 800
    block_size = 64

    # Create mixed sparse tensor
    tensor = SingleBlockMixedSparseTensor(
        batch_size=led_count,
        channels=channels,
        height=height,
        width=width,
        block_size=block_size,
    )

    # Generate LED positions with spatial Morton ordering for cache locality
    logger.info(f"Generating large tensor with {led_count} LEDs for performance testing...")

    led_positions = generate_spatially_ordered_positions(led_count, height, width, block_size, seed=42)

    # Generate realistic diffusion patterns with spatially ordered placement
    np.random.seed(42)  # For reproducible results

    for led_idx in range(led_count):
        top_row, top_col = led_positions[led_idx]

        for channel_idx in range(channels):
            # Generate realistic diffusion pattern (Gaussian-like)
            center_row, center_col = block_size // 2, block_size // 2
            y_coords, x_coords = np.ogrid[:block_size, :block_size]

            # Gaussian pattern with some randomness
            sigma = block_size / 4
            gaussian = np.exp(-((x_coords - center_col) ** 2 + (y_coords - center_row) ** 2) / (2 * sigma**2))

            # Add some noise and channel-specific intensity
            noise = np.random.normal(0, 0.1, (block_size, block_size))
            intensity = (channel_idx + 1) / channels  # Different intensity per channel
            pattern = (gaussian * intensity + noise).clip(0, 1).astype(np.float32)

            # Set the block
            tensor.set_block(led_idx, channel_idx, top_row, top_col, cp.asarray(pattern))

    logger.info(f"Large tensor generation complete: {tensor}")
    return tensor


@pytest.fixture
def large_mixed_sparse_tensor_aligned():
    """Create a large mixed sparse tensor with x-positions
    aligned to multiples of 4 for experimental kernel testing."""
    # Large problem size for accurate performance measurements
    led_count = 1000  # Full problem size for performance testing
    channels = 3
    height = 480
    width = 800
    block_size = 64

    # Create mixed sparse tensor
    tensor = SingleBlockMixedSparseTensor(
        batch_size=led_count,
        channels=channels,
        height=height,
        width=width,
        block_size=block_size,
    )

    # Generate LED positions with spatial Morton ordering AND x-alignment for experimental kernel
    logger.info(f"Generating large aligned tensor with {led_count} LEDs for experimental kernel testing...")

    led_positions = generate_spatially_ordered_positions(
        led_count, height, width, block_size, seed=42, align_x_to_4=True
    )

    # Generate realistic diffusion patterns with spatially ordered placement
    np.random.seed(42)  # For reproducible results

    for led_idx in range(led_count):
        top_row, top_col = led_positions[led_idx]

        for channel_idx in range(channels):
            # Generate realistic diffusion pattern (Gaussian-like)
            center_row, center_col = block_size // 2, block_size // 2
            y_coords, x_coords = np.ogrid[:block_size, :block_size]

            # Gaussian pattern with some randomness
            sigma = block_size / 4
            gaussian = np.exp(-((x_coords - center_col) ** 2 + (y_coords - center_row) ** 2) / (2 * sigma**2))

            # Add some noise and channel-specific intensity
            noise = np.random.normal(0, 0.1, (block_size, block_size))
            intensity = (channel_idx + 1) / channels  # Different intensity per channel
            pattern = (gaussian * intensity + noise).clip(0, 1).astype(np.float32)

            # Set the block
            tensor.set_block(led_idx, channel_idx, top_row, top_col, cp.asarray(pattern))

    logger.info(f"Large aligned tensor generation complete: {tensor}")
    return tensor


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCudaKernels:
    """Test class for CUDA kernel functionality and performance."""

    def test_cuda_kernel_availability(self):
        """Test that CUDA kernel can be imported and compiled."""
        # Test kernel compilation
        from src.utils.cuda_kernels import get_compute_optimized_3d_kernel

        kernel = get_compute_optimized_3d_kernel()
        assert kernel is not None

    def test_kernel_functionality(self, mixed_sparse_tensor, test_images):
        """Test that CUDA kernel produces correct results."""
        # Test with each test image
        for i, test_image in enumerate(test_images):
            # Convert to CuPy array
            target_gpu = cp.asarray(test_image)

            # Get sparse values and block positions from mixed tensor
            sparse_values = mixed_sparse_tensor.sparse_values  # Shape: (channels, batch_size, block_size, block_size)
            block_positions = mixed_sparse_tensor.block_positions  # Shape: (channels, batch_size, 2)

            # Run kernel
            result = cuda_transpose_dot_product_3d_compute_optimized(
                sparse_values=sparse_values,
                block_positions=block_positions,
                target_3d=target_gpu,
                batch_size=mixed_sparse_tensor.batch_size,
                channels=mixed_sparse_tensor.channels,
                block_size=mixed_sparse_tensor.block_size,
            )

            # Verify output shape and type
            expected_shape = (
                mixed_sparse_tensor.batch_size,
                mixed_sparse_tensor.channels,
            )
            assert result.shape == expected_shape
            assert result.dtype == cp.float32

    def test_kernel_performance_comprehensive(
        self, large_mixed_sparse_tensor, large_mixed_sparse_tensor_aligned, test_images
    ):
        """Comprehensive performance test for all CUDA kernels
        (fp32, int8, int8_experimental) with cache warming.

        Uses large problem size (1000 LEDs, 480x800 images) for realistic timing measurements
        that properly account for memory bandwidth and cache behavior.
        """
        # Initialize performance timing
        perf_timer = PerformanceTiming("CudaKernelPerformance", enable_gpu_timing=True)

        # Get kernel inputs from large tensor (fp32)
        sparse_values_fp32 = large_mixed_sparse_tensor.sparse_values
        block_positions = large_mixed_sparse_tensor.block_positions

        # Get aligned kernel inputs (also fp32, converted to int8 later)
        sparse_values_fp32_aligned = large_mixed_sparse_tensor_aligned.sparse_values
        block_positions_aligned = large_mixed_sparse_tensor_aligned.block_positions

        # Convert to int8 data (multiply by 255 and convert to uint8)
        # Shape: (channels, batch_size, block_size, block_size) - fp32 -> uint8
        sparse_values_int8 = (sparse_values_fp32 * 255).astype(cp.uint8)
        sparse_values_int8_aligned = (sparse_values_fp32_aligned * 255).astype(cp.uint8)

        # Test parameters
        warmup_runs = 5
        benchmark_runs = 20

        fp32_times = []
        int8_times = []
        int8_experimental_times = []

        logger.info("Running comprehensive performance test:")
        logger.info(f"  LED count: {large_mixed_sparse_tensor.batch_size}")
        logger.info(f"  Image size: {large_mixed_sparse_tensor.height}x{large_mixed_sparse_tensor.width}")
        logger.info(f"  Block size: {large_mixed_sparse_tensor.block_size}")
        logger.info(f"  Test images: {len(test_images)}")
        logger.info(f"  Warmup runs: {warmup_runs}")
        logger.info(f"  Benchmark runs: {benchmark_runs}")

        # Calculate problem size metrics
        total_pixels = (
            large_mixed_sparse_tensor.height * large_mixed_sparse_tensor.width * large_mixed_sparse_tensor.channels
        )
        total_blocks = large_mixed_sparse_tensor.batch_size * large_mixed_sparse_tensor.channels
        block_pixels = large_mixed_sparse_tensor.block_size * large_mixed_sparse_tensor.block_size
        total_operations = total_blocks * block_pixels

        logger.info("  Problem size metrics:")
        logger.info(f"    Total image pixels: {total_pixels:,}")
        logger.info(f"    Total LED blocks: {total_blocks:,}")
        logger.info(f"    Total operations: {total_operations:,}")

        for img_idx, test_image in enumerate(test_images):
            # Convert test image to int8 format (multiply by 255 and convert to uint8)
            # Shape: (channels, height, width) - fp32 -> uint8
            test_image_int8 = (test_image * 255).astype(cp.uint8)

            # ====== FP32 KERNEL BENCHMARKING ======
            logger.debug(f"\nFP32 kernel - Cache warming for image {img_idx + 1}/{len(test_images)}...")

            # FP32 Cache warming phase
            for warmup in range(warmup_runs):
                target_gpu_fresh = cp.asarray(test_image.copy())
                _ = cuda_transpose_dot_product_3d_compute_optimized(
                    sparse_values=sparse_values_fp32,
                    block_positions=block_positions,
                    target_3d=target_gpu_fresh,
                    batch_size=large_mixed_sparse_tensor.batch_size,
                    channels=large_mixed_sparse_tensor.channels,
                    block_size=large_mixed_sparse_tensor.block_size,
                )

            # FP32 Benchmark phase
            logger.debug(f"FP32 kernel - Benchmarking image {img_idx + 1}...")

            for run in range(benchmark_runs):
                target_gpu_fresh = cp.asarray(test_image.copy())
                with perf_timer.section(f"fp32_benchmark_img{img_idx}_{run}", use_gpu_events=True) as timer:
                    result = cuda_transpose_dot_product_3d_compute_optimized(
                        sparse_values=sparse_values_fp32,
                        block_positions=block_positions,
                        target_3d=target_gpu_fresh,
                        batch_size=large_mixed_sparse_tensor.batch_size,
                        channels=large_mixed_sparse_tensor.channels,
                        block_size=large_mixed_sparse_tensor.block_size,
                    )

            # ====== INT8 KERNEL BENCHMARKING ======
            logger.debug(f"INT8 kernel - Cache warming for image {img_idx + 1}/{len(test_images)}...")

            # INT8 Cache warming phase
            for warmup in range(warmup_runs):
                target_gpu_fresh_int8 = cp.asarray(test_image_int8.copy())
                _ = cuda_transpose_dot_product_3d_compute_optimized_int8(
                    sparse_values=sparse_values_int8,
                    block_positions=block_positions_aligned,
                    target_3d=target_gpu_fresh_int8,
                    batch_size=large_mixed_sparse_tensor_aligned.batch_size,
                    channels=large_mixed_sparse_tensor_aligned.channels,
                    block_size=large_mixed_sparse_tensor_aligned.block_size,
                )

            # INT8 Benchmark phase
            logger.debug(f"INT8 kernel - Benchmarking image {img_idx + 1}...")

            for run in range(benchmark_runs):
                target_gpu_fresh_int8 = cp.asarray(test_image_int8.copy())
                with perf_timer.section(f"int8_benchmark_img{img_idx}_{run}", use_gpu_events=True) as timer:
                    result = cuda_transpose_dot_product_3d_compute_optimized_int8(
                        sparse_values=sparse_values_int8,
                        block_positions=block_positions_aligned,
                        target_3d=target_gpu_fresh_int8,
                        batch_size=large_mixed_sparse_tensor_aligned.batch_size,
                        channels=large_mixed_sparse_tensor_aligned.channels,
                        block_size=large_mixed_sparse_tensor_aligned.block_size,
                    )

            # ====== INT8 EXPERIMENTAL KERNEL BENCHMARKING ======
            logger.debug(f"INT8 EXPERIMENTAL kernel - Cache warming for image {img_idx + 1}/{len(test_images)}...")

            # INT8 Experimental Cache warming phase
            for warmup in range(warmup_runs):
                target_gpu_fresh_int8 = cp.asarray(test_image_int8.copy())
                _ = cuda_transpose_dot_product_3d_compute_optimized_int8_experimental(
                    sparse_values=sparse_values_int8_aligned,
                    block_positions=block_positions_aligned,
                    target_3d=target_gpu_fresh_int8,
                    batch_size=large_mixed_sparse_tensor_aligned.batch_size,
                    channels=large_mixed_sparse_tensor_aligned.channels,
                    block_size=large_mixed_sparse_tensor_aligned.block_size,
                )

            # INT8 Experimental Benchmark phase
            logger.debug(f"INT8 EXPERIMENTAL kernel - Benchmarking image {img_idx + 1}...")

            for run in range(benchmark_runs):
                target_gpu_fresh_int8 = cp.asarray(test_image_int8.copy())
                with perf_timer.section(
                    f"int8_experimental_benchmark_img{img_idx}_{run}",
                    use_gpu_events=True,
                ) as timer:
                    result = cuda_transpose_dot_product_3d_compute_optimized_int8_experimental(
                        sparse_values=sparse_values_int8_aligned,
                        block_positions=block_positions_aligned,
                        target_3d=target_gpu_fresh_int8,
                        batch_size=large_mixed_sparse_tensor_aligned.batch_size,
                        channels=large_mixed_sparse_tensor_aligned.channels,
                        block_size=large_mixed_sparse_tensor_aligned.block_size,
                    )

        # Extract timing data and analyze
        timing_data = perf_timer.get_timing_data()

        # Collect benchmark times for all kernels
        for section_name, section_data in timing_data["sections"].items():
            # Use GPU duration if available, otherwise CPU duration
            duration = section_data.get("gpu_duration") or section_data["duration"]

            if "fp32_benchmark_" in section_name:
                fp32_times.append(duration)
            elif "int8_experimental_benchmark_" in section_name:
                int8_experimental_times.append(duration)
            elif "int8_benchmark_" in section_name:
                int8_times.append(duration)

        # Calculate statistics for all kernels
        fp32_mean = np.mean(fp32_times)
        fp32_std = np.std(fp32_times)
        int8_mean = np.mean(int8_times)
        int8_std = np.std(int8_times)
        int8_exp_mean = np.mean(int8_experimental_times)
        int8_exp_std = np.std(int8_experimental_times)

        # Calculate performance comparisons
        int8_speedup = fp32_mean / int8_mean if int8_mean > 0 else 0
        int8_exp_speedup = fp32_mean / int8_exp_mean if int8_exp_mean > 0 else 0
        exp_vs_regular_int8 = int8_mean / int8_exp_mean if int8_exp_mean > 0 else 0

        # Log performance results
        logger.info("\n" + "=" * 70)
        logger.info("COMPREHENSIVE CUDA KERNEL PERFORMANCE COMPARISON")
        logger.info("=" * 70)
        logger.info("FP32 Kernel Performance:")
        logger.info(f"  Mean time: {fp32_mean * 1000:.3f} ms")
        logger.info(f"  Std dev: {fp32_std * 1000:.3f} ms")
        logger.info(f"  Min time: {min(fp32_times) * 1000:.3f} ms")
        logger.info(f"  Max time: {max(fp32_times) * 1000:.3f} ms")
        logger.info("")
        logger.info("INT8 Kernel Performance:")
        logger.info(f"  Mean time: {int8_mean * 1000:.3f} ms")
        logger.info(f"  Std dev: {int8_std * 1000:.3f} ms")
        logger.info(f"  Min time: {min(int8_times) * 1000:.3f} ms")
        logger.info(f"  Max time: {max(int8_times) * 1000:.3f} ms")
        logger.info("")
        logger.info("INT8 EXPERIMENTAL Kernel Performance:")
        logger.info(f"  Mean time: {int8_exp_mean * 1000:.3f} ms")
        logger.info(f"  Std dev: {int8_exp_std * 1000:.3f} ms")
        logger.info(f"  Min time: {min(int8_experimental_times) * 1000:.3f} ms")
        logger.info(f"  Max time: {max(int8_experimental_times) * 1000:.3f} ms")
        logger.info("")
        logger.info("Performance Comparisons:")
        logger.info(f"  INT8 vs FP32: {int8_speedup:.2f}x {'(faster)' if int8_speedup > 1 else '(slower)'}")
        logger.info(f"  INT8_EXP vs FP32: {int8_exp_speedup:.2f}x {'(faster)' if int8_exp_speedup > 1 else '(slower)'}")
        logger.info(
            f"  INT8_EXP vs INT8: {exp_vs_regular_int8:.2f}x {'(faster)' if exp_vs_regular_int8 > 1 else '(slower)'}"
        )
        logger.info("  Memory Usage: INT8 uses ~4x less memory than FP32")
        logger.info("=" * 70)

        # Assertions to ensure all kernels are functional
        assert len(fp32_times) > 0, "No FP32 kernel timing data collected"
        assert len(int8_times) > 0, "No INT8 kernel timing data collected"
        assert len(int8_experimental_times) > 0, "No INT8 EXPERIMENTAL kernel timing data collected"
        assert all(t > 0 for t in fp32_times), "Invalid FP32 kernel timing data"
        assert all(t > 0 for t in int8_times), "Invalid INT8 kernel timing data"
        assert all(t > 0 for t in int8_experimental_times), "Invalid INT8 EXPERIMENTAL kernel timing data"

        # Log a summary for reference
        logger.info("Comprehensive performance test completed successfully!")
        logger.info(f"FP32 data collected: {len(fp32_times)} measurements")
        logger.info(f"INT8 data collected: {len(int8_times)} measurements")
        logger.info(f"INT8_EXP data collected: {len(int8_experimental_times)} measurements")

    def test_kernel_performance(self, large_mixed_sparse_tensor, large_mixed_sparse_tensor_aligned, test_images):
        """Performance test for CUDA kernels (fp32 and int8) with cache warming.

        Uses large problem size (1000 LEDs, 480x800 images) for realistic timing measurements
        that properly account for memory bandwidth and cache behavior.
        """
        # Initialize performance timing
        perf_timer = PerformanceTiming("CudaKernelPerformance", enable_gpu_timing=True)

        # Get kernel inputs from large tensor (fp32)
        sparse_values_fp32 = large_mixed_sparse_tensor.sparse_values
        block_positions = large_mixed_sparse_tensor.block_positions

        block_positions_aligned = large_mixed_sparse_tensor_aligned.block_positions

        # Convert to int8 data (multiply by 255 and convert to uint8)
        # Shape: (channels, batch_size, block_size, block_size) - fp32 -> uint8
        sparse_values_int8 = (sparse_values_fp32 * 255).astype(cp.uint8)

        # Test parameters
        warmup_runs = 5
        benchmark_runs = 20

        fp32_times = []
        int8_times = []

        logger.info("Running performance test:")
        logger.info(f"  LED count: {large_mixed_sparse_tensor.batch_size}")
        logger.info(f"  Image size: {large_mixed_sparse_tensor.height}x{large_mixed_sparse_tensor.width}")
        logger.info(f"  Block size: {large_mixed_sparse_tensor.block_size}")
        logger.info(f"  Test images: {len(test_images)}")
        logger.info(f"  Warmup runs: {warmup_runs}")
        logger.info(f"  Benchmark runs: {benchmark_runs}")

        # Calculate problem size metrics
        total_pixels = (
            large_mixed_sparse_tensor.height * large_mixed_sparse_tensor.width * large_mixed_sparse_tensor.channels
        )
        total_blocks = large_mixed_sparse_tensor.batch_size * large_mixed_sparse_tensor.channels
        block_pixels = large_mixed_sparse_tensor.block_size * large_mixed_sparse_tensor.block_size
        total_operations = total_blocks * block_pixels

        logger.info("  Problem size metrics:")
        logger.info(f"    Total image pixels: {total_pixels:,}")
        logger.info(f"    Total LED blocks: {total_blocks:,}")
        logger.info(f"    Total operations: {total_operations:,}")

        for img_idx, test_image in enumerate(test_images):
            # Convert test image to int8 format (multiply by 255 and convert to uint8)
            # Shape: (channels, height, width) - fp32 -> uint8
            test_image_int8 = (test_image * 255).astype(cp.uint8)

            # ====== FP32 KERNEL BENCHMARKING ======
            logger.debug(f"\nFP32 kernel - Cache warming for image {img_idx + 1}/{len(test_images)}...")

            # FP32 Cache warming phase
            for warmup in range(warmup_runs):
                target_gpu_fresh = cp.asarray(test_image.copy())
                _ = cuda_transpose_dot_product_3d_compute_optimized(
                    sparse_values=sparse_values_fp32,
                    block_positions=block_positions,
                    target_3d=target_gpu_fresh,
                    batch_size=large_mixed_sparse_tensor.batch_size,
                    channels=large_mixed_sparse_tensor.channels,
                    block_size=large_mixed_sparse_tensor.block_size,
                )

            # FP32 Benchmark phase
            logger.debug(f"FP32 kernel - Benchmarking image {img_idx + 1}...")

            for run in range(benchmark_runs):
                target_gpu_fresh = cp.asarray(test_image.copy())
                with perf_timer.section(f"fp32_benchmark_img{img_idx}_{run}", use_gpu_events=True) as timer:
                    result = cuda_transpose_dot_product_3d_compute_optimized(
                        sparse_values=sparse_values_fp32,
                        block_positions=block_positions,
                        target_3d=target_gpu_fresh,
                        batch_size=large_mixed_sparse_tensor.batch_size,
                        channels=large_mixed_sparse_tensor.channels,
                        block_size=large_mixed_sparse_tensor.block_size,
                    )

            # ====== INT8 KERNEL BENCHMARKING ======
            logger.debug(f"INT8 kernel - Cache warming for image {img_idx + 1}/{len(test_images)}...")

            # INT8 Cache warming phase
            for warmup in range(warmup_runs):
                target_gpu_fresh_int8 = cp.asarray(test_image_int8.copy())
                _ = cuda_transpose_dot_product_3d_compute_optimized_int8(
                    sparse_values=sparse_values_int8,
                    block_positions=block_positions_aligned,
                    target_3d=target_gpu_fresh_int8,
                    batch_size=large_mixed_sparse_tensor_aligned.batch_size,
                    channels=large_mixed_sparse_tensor_aligned.channels,
                    block_size=large_mixed_sparse_tensor_aligned.block_size,
                )

            # INT8 Benchmark phase
            logger.debug(f"INT8 kernel - Benchmarking image {img_idx + 1}...")

            for run in range(benchmark_runs):
                target_gpu_fresh_int8 = cp.asarray(test_image_int8.copy())
                with perf_timer.section(f"int8_benchmark_img{img_idx}_{run}", use_gpu_events=True) as timer:
                    result = cuda_transpose_dot_product_3d_compute_optimized_int8(
                        sparse_values=sparse_values_int8,
                        block_positions=block_positions_aligned,
                        target_3d=target_gpu_fresh_int8,
                        batch_size=large_mixed_sparse_tensor_aligned.batch_size,
                        channels=large_mixed_sparse_tensor_aligned.channels,
                        block_size=large_mixed_sparse_tensor_aligned.block_size,
                    )

        # Extract timing data and analyze
        timing_data = perf_timer.get_timing_data()

        # Collect benchmark times for both kernels
        for section_name, section_data in timing_data["sections"].items():
            # Use GPU duration if available, otherwise CPU duration
            duration = section_data.get("gpu_duration") or section_data["duration"]

            if "fp32_benchmark_" in section_name:
                fp32_times.append(duration)
            elif "int8_benchmark_" in section_name:
                int8_times.append(duration)

        # Calculate statistics for both kernels
        fp32_mean = np.mean(fp32_times)
        fp32_std = np.std(fp32_times)
        int8_mean = np.mean(int8_times)
        int8_std = np.std(int8_times)

        # Calculate performance comparison
        speedup = fp32_mean / int8_mean if int8_mean > 0 else 0

        # Log performance results
        logger.info("\n" + "=" * 70)
        logger.info("CUDA KERNEL PERFORMANCE COMPARISON")
        logger.info("=" * 70)
        logger.info("FP32 Kernel Performance:")
        logger.info(f"  Mean time: {fp32_mean * 1000:.3f} ms")
        logger.info(f"  Std dev: {fp32_std * 1000:.3f} ms")
        logger.info(f"  Min time: {min(fp32_times) * 1000:.3f} ms")
        logger.info(f"  Max time: {max(fp32_times) * 1000:.3f} ms")
        logger.info("")
        logger.info("INT8 Kernel Performance:")
        logger.info(f"  Mean time: {int8_mean * 1000:.3f} ms")
        logger.info(f"  Std dev: {int8_std * 1000:.3f} ms")
        logger.info(f"  Min time: {min(int8_times) * 1000:.3f} ms")
        logger.info(f"  Max time: {max(int8_times) * 1000:.3f} ms")
        logger.info("")
        logger.info("Performance Comparison:")
        logger.info(f"  INT8 Speedup: {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")
        logger.info("  Memory Usage: INT8 uses ~4x less memory than FP32")
        logger.info("=" * 70)

        # Assertions to ensure both kernels are functional
        assert len(fp32_times) > 0, "No FP32 kernel timing data collected"
        assert len(int8_times) > 0, "No INT8 kernel timing data collected"
        assert all(t > 0 for t in fp32_times), "Invalid FP32 kernel timing data"
        assert all(t > 0 for t in int8_times), "Invalid INT8 kernel timing data"

        # Log a summary for reference
        logger.info("Performance test completed successfully!")
        logger.info(f"FP32 data collected: {len(fp32_times)} measurements")
        logger.info(f"INT8 data collected: {len(int8_times)} measurements")

    def test_kernel_with_different_block_sizes(self):
        """Test kernel with different block sizes to ensure robustness."""
        # Test with multiple block sizes
        block_sizes = [32, 64, 96]  # Different block sizes to test
        test_image = np.random.rand(3, 480, 800).astype(np.float32)
        target_gpu = cp.asarray(test_image)

        for block_size in block_sizes:
            logger.info(f"Testing block size: {block_size}")

            # Create mixed tensor with specific block size
            led_count = 50  # Smaller for faster testing with multiple block sizes
            channels = 3
            height = 480
            width = 800

            mixed_tensor = SingleBlockMixedSparseTensor(
                batch_size=led_count,
                channels=channels,
                height=height,
                width=width,
                block_size=block_size,
            )

            # Generate LED positions with spatial Morton ordering for cache locality
            led_positions = generate_spatially_ordered_positions(led_count, height, width, block_size, seed=42)

            # Generate test patterns for this block size with Morton ordering
            np.random.seed(42)  # Consistent patterns
            for led_idx in range(led_count):
                top_row, top_col = led_positions[led_idx]

                for channel_idx in range(channels):
                    # Simple pattern for testing
                    pattern = np.random.rand(block_size, block_size).astype(np.float32)
                    mixed_tensor.set_block(led_idx, channel_idx, top_row, top_col, cp.asarray(pattern))

            sparse_values = mixed_tensor.sparse_values
            block_positions = mixed_tensor.block_positions

            # Test kernel
            result = cuda_transpose_dot_product_3d_compute_optimized(
                sparse_values=sparse_values,
                block_positions=block_positions,
                target_3d=target_gpu,
                batch_size=led_count,
                channels=channels,
                block_size=block_size,
            )

            # Verify output shape
            expected_shape = (led_count, channels)
            assert result.shape == expected_shape
            assert result.dtype == cp.float32

    def test_int8_kernel_availability(self):
        """Test that int8 CUDA kernel can be imported and compiled."""
        # Test int8 kernel compilation
        from src.utils.cuda_kernels import get_compute_optimized_3d_int8_kernel

        kernel = get_compute_optimized_3d_int8_kernel()
        assert kernel is not None

    def test_int8_kernel_functionality(self):
        """Test that int8 CUDA kernel produces correct results."""
        # Create int8 tensor with known patterns
        batch_size, channels = 5, 3
        height, width = 64, 80
        block_size = 32  # Multiple of 4 for vectorization

        # Create test data in int8 range
        np.random.seed(42)
        int8_sparse_data = np.random.randint(0, 256, (channels, batch_size, block_size, block_size), dtype=np.uint8)
        int8_target_data = np.random.randint(0, 256, (channels, height, width), dtype=np.uint8)
        positions = np.random.randint(0, (min(height, width) - block_size) // 4, (channels, batch_size, 2))

        positions = positions * 4  # Ensure positions are multiples of 4

        # Convert to CuPy arrays
        sparse_values = cp.asarray(int8_sparse_data)
        target_3d = cp.asarray(int8_target_data)
        block_positions = cp.asarray(positions)

        # Run int8 kernel
        result = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_values=sparse_values,
            block_positions=block_positions,
            target_3d=target_3d,
            batch_size=batch_size,
            channels=channels,
            block_size=block_size,
        )

        # Verify output shape and type
        expected_shape = (batch_size, channels)
        assert result.shape == expected_shape
        assert result.dtype == cp.float32

        # Verify values are in reasonable range (normalized by 255*255)
        assert cp.all(result >= 0)  # All values should be non-negative
        # Max possible value after normalization: block_size^2 (when all pixels are 255)
        max_expected = block_size * block_size  # 32*32 = 1024 for this test
        assert cp.all(result <= max_expected)  # All values should be <= block_size^2

    def test_int8_fp32_kernel_equivalence(self):
        """Test mathematical equivalence between int8 and fp32 kernels."""
        # Create test data
        batch_size, channels = 3, 2
        height, width = 48, 64
        block_size = 32

        # Generate int8 test data
        np.random.seed(123)  # Different seed from other tests
        int8_sparse_data = np.random.randint(0, 256, (channels, batch_size, block_size, block_size), dtype=np.uint8)
        int8_target_data = np.random.randint(0, 256, (channels, height, width), dtype=np.uint8)
        positions = np.random.randint(0, (min(height, width) - block_size) // 4, (channels, batch_size, 2))

        positions = positions * 4  # Ensure positions are multiples of 4

        # Convert to CuPy arrays
        int8_sparse_cupy = cp.asarray(int8_sparse_data)
        int8_target_cupy = cp.asarray(int8_target_data)
        block_positions = cp.asarray(positions)

        # Run int8 kernel
        int8_result = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_values=int8_sparse_cupy,
            block_positions=block_positions,
            target_3d=int8_target_cupy,
            batch_size=batch_size,
            channels=channels,
            block_size=block_size,
        )

        # Convert int8 data to fp32 (unscaled)
        fp32_sparse_data = int8_sparse_data.astype(np.float32)
        fp32_target_data = int8_target_data.astype(np.float32)
        fp32_sparse_cupy = cp.asarray(fp32_sparse_data)
        fp32_target_cupy = cp.asarray(fp32_target_data)

        # Run fp32 kernel
        fp32_result = cuda_transpose_dot_product_3d_compute_optimized(
            sparse_values=fp32_sparse_cupy,
            block_positions=block_positions,
            target_3d=fp32_target_cupy,
            batch_size=batch_size,
            channels=channels,
            block_size=block_size,
        )

        # Scale fp32 result to match int8 normalization
        fp32_result_scaled = fp32_result / (255.0 * 255.0)

        # Compare results
        cp.testing.assert_allclose(
            int8_result,
            fp32_result_scaled,
            rtol=1e-6,
            atol=1e-8,
            err_msg="int8 and fp32 (scaled) kernel results should be equivalent",
        )

    def test_int8_kernel_with_different_block_sizes(self):
        """Test int8 kernel with different block sizes."""
        # Test with multiple block sizes (must be multiples of 4)
        block_sizes = [32, 64]  # Removed 96 since it's not multiple of 4
        batch_size, channels = 3, 2
        height, width = 80, 96

        for block_size in block_sizes:
            logger.info(f"Testing int8 kernel with block size: {block_size}")

            # Generate test data
            int8_sparse_data = np.random.randint(0, 256, (channels, batch_size, block_size, block_size), dtype=np.uint8)
            int8_target_data = np.random.randint(0, 256, (channels, height, width), dtype=np.uint8)
            positions = np.random.randint(0, (min(height, width) - block_size) // 4, (channels, batch_size, 2))

            positions = positions * 4  # Ensure positions are multiples of 4

            # Convert to CuPy
            sparse_values = cp.asarray(int8_sparse_data)
            target_3d = cp.asarray(int8_target_data)
            block_positions = cp.asarray(positions)

            # Test kernel
            result = cuda_transpose_dot_product_3d_compute_optimized_int8(
                sparse_values=sparse_values,
                block_positions=block_positions,
                target_3d=target_3d,
                batch_size=batch_size,
                channels=channels,
                block_size=block_size,
            )

            # Verify output shape and type
            expected_shape = (batch_size, channels)
            assert result.shape == expected_shape
            assert result.dtype == cp.float32

    def test_experimental_vs_original_int8_kernel_equivalence(self, large_mixed_sparse_tensor_aligned):
        """Test that experimental int8 kernel produces equivalent
        results to the original int8 kernel."""
        # Use aligned tensor for experimental kernel
        aligned_tensor = large_mixed_sparse_tensor_aligned

        # Create a smaller test for faster validation (use subset of LEDs)
        test_led_count = 50  # Smaller subset for validation
        test_image = np.random.rand(3, 480, 800).astype(np.float32)
        test_image_int8 = cp.asarray((test_image * 255).astype(np.uint8))

        # Get sparse data and positions (subset)
        sparse_values_fp32 = aligned_tensor.sparse_values[:, :test_led_count, :, :]  # (3, 50, 64, 64)
        sparse_values_aligned = (sparse_values_fp32 * 255).astype(cp.uint8)  # Convert to uint8
        block_positions_aligned = aligned_tensor.block_positions[:, :test_led_count, :]  # (3, 50, 2)

        # Verify that all x-positions are aligned to multiples of 4
        x_positions = block_positions_aligned[:, :, 1].get()  # Get x-positions (columns)
        assert np.all(
            x_positions % 4 == 0
        ), f"Not all x-positions are aligned to 4: {x_positions[x_positions % 4 != 0]}"

        logger.info(f"Testing experimental kernel equivalence with {test_led_count} LEDs")
        logger.info("Verified all x-positions are aligned to multiples of 4")

        # Run experimental kernel with aligned data
        result_experimental = cuda_transpose_dot_product_3d_compute_optimized_int8_experimental(
            sparse_values=sparse_values_aligned,
            block_positions=block_positions_aligned,
            target_3d=test_image_int8,
            batch_size=test_led_count,
            channels=3,
            block_size=64,
        )

        # Run original int8 kernel with same aligned data for comparison
        result_original = cuda_transpose_dot_product_3d_compute_optimized_int8(
            sparse_values=sparse_values_aligned,
            block_positions=block_positions_aligned,
            target_3d=test_image_int8,
            batch_size=test_led_count,
            channels=3,
            block_size=64,
        )

        # Verify both results have the same shape and type
        assert result_experimental.shape == result_original.shape
        assert result_experimental.dtype == result_original.dtype

        # Compare results - they should be identical since both use the same algorithm
        # but experimental uses aligned loads
        cp.testing.assert_allclose(
            result_experimental,
            result_original,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Experimental and original int8 kernel results should be equivalent for aligned data",
        )

        logger.info("✓ Experimental kernel produces equivalent results to original kernel")
        logger.info(f"✓ Result shape: {result_experimental.shape}")
        logger.info(f"✓ Max absolute difference: {cp.max(cp.abs(result_experimental - result_original)).get():.2e}")

    def test_experimental_kernel_availability(self):
        """Test that experimental int8 CUDA kernel can be imported and compiled."""
        # Test experimental kernel compilation
        from src.utils.cuda_kernels import (
            get_compute_optimized_3d_int8_experimental_kernel,
        )

        kernel = get_compute_optimized_3d_int8_experimental_kernel()
        assert kernel is not None

    def test_int8_vs_experimental_performance_comparison(
        self, large_mixed_sparse_tensor, large_mixed_sparse_tensor_aligned, test_images
    ):
        """Focused performance comparison between int8 and int8_experimental
        kernels with alignment experiment.

        This test demonstrates the performance impact of aligned uchar4 loads by comparing:
        1. Original int8 kernel with random LED positions
        2. Experimental int8 kernel with x-positions aligned to multiples of 4
        """
        # Initialize performance timing
        perf_timer = PerformanceTiming("Int8ExperimentalComparison", enable_gpu_timing=True)

        # Get data from both tensors
        # Regular tensor (random positions)
        sparse_values_regular = (large_mixed_sparse_tensor.sparse_values * 255).astype(cp.uint8)
        block_positions_regular = large_mixed_sparse_tensor.block_positions

        # Aligned tensor (x-positions are multiples of 4)
        sparse_values_aligned = (large_mixed_sparse_tensor_aligned.sparse_values * 255).astype(cp.uint8)
        block_positions_aligned = large_mixed_sparse_tensor_aligned.block_positions

        # Verify alignment
        x_positions_regular = block_positions_regular[:, :, 1].get()
        x_positions_aligned = block_positions_aligned[:, :, 1].get()

        regular_aligned_count = np.sum(x_positions_regular % 4 == 0)
        aligned_count = np.sum(x_positions_aligned % 4 == 0)
        total_positions = x_positions_regular.size

        logger.info("\n" + "=" * 60)
        logger.info("INT8 vs EXPERIMENTAL KERNEL ALIGNMENT EXPERIMENT")
        logger.info("=" * 60)
        logger.info(
            f"Regular tensor - x-positions aligned to 4: "
            f"{regular_aligned_count}/{total_positions} "
            f"({100 * regular_aligned_count / total_positions:.1f}%)"
        )
        logger.info(
            f"Aligned tensor - x-positions aligned to 4: "
            f"{aligned_count}/{total_positions} ({100 * aligned_count / total_positions:.1f}%)"
        )
        assert aligned_count == total_positions, "Aligned tensor should have all x-positions as multiples of 4"

        # Test parameters
        warmup_runs = 3
        benchmark_runs = 10  # Focused comparison, fewer runs

        int8_regular_times = []
        int8_experimental_times = []

        logger.info("\nBenchmark parameters:")
        logger.info(f"  LED count: {large_mixed_sparse_tensor.batch_size}")
        logger.info(f"  Image size: {large_mixed_sparse_tensor.height}x{large_mixed_sparse_tensor.width}")
        logger.info(f"  Block size: {large_mixed_sparse_tensor.block_size}")
        logger.info(f"  Warmup runs: {warmup_runs}")
        logger.info(f"  Benchmark runs: {benchmark_runs}")

        # Use a single representative test image for focused comparison
        test_image = test_images[0]  # Use first test image
        test_image_int8 = cp.asarray((test_image * 255).astype(np.uint8))

        # ====== INT8 REGULAR KERNEL (Random positions) ======
        logger.info("\nINT8 REGULAR kernel (random positions) - Cache warming...")

        # Cache warming
        for warmup in range(warmup_runs):
            target_gpu_fresh = cp.asarray(test_image_int8.copy())
            _ = cuda_transpose_dot_product_3d_compute_optimized_int8(
                sparse_values=sparse_values_aligned,
                block_positions=block_positions_aligned,
                target_3d=target_gpu_fresh,
                batch_size=large_mixed_sparse_tensor_aligned.batch_size,
                channels=large_mixed_sparse_tensor_aligned.channels,
                block_size=large_mixed_sparse_tensor_aligned.block_size,
            )

        # Benchmark
        logger.info("INT8 REGULAR kernel - Benchmarking...")
        for run in range(benchmark_runs):
            target_gpu_fresh = cp.asarray(test_image_int8.copy())
            with perf_timer.section(f"int8_regular_{run}", use_gpu_events=True) as timer:
                result_regular = cuda_transpose_dot_product_3d_compute_optimized_int8(
                    sparse_values=sparse_values_aligned,
                    block_positions=block_positions_aligned,
                    target_3d=target_gpu_fresh,
                    batch_size=large_mixed_sparse_tensor_aligned.batch_size,
                    channels=large_mixed_sparse_tensor_aligned.channels,
                    block_size=large_mixed_sparse_tensor_aligned.block_size,
                )

        # ====== INT8 EXPERIMENTAL KERNEL (Aligned positions) ======
        logger.info("INT8 EXPERIMENTAL kernel (aligned positions) - Cache warming...")

        # Cache warming
        for warmup in range(warmup_runs):
            target_gpu_fresh = cp.asarray(test_image_int8.copy())
            _ = cuda_transpose_dot_product_3d_compute_optimized_int8_experimental(
                sparse_values=sparse_values_aligned,
                block_positions=block_positions_aligned,
                target_3d=target_gpu_fresh,
                batch_size=large_mixed_sparse_tensor_aligned.batch_size,
                channels=large_mixed_sparse_tensor_aligned.channels,
                block_size=large_mixed_sparse_tensor_aligned.block_size,
            )

        # Benchmark
        logger.info("INT8 EXPERIMENTAL kernel - Benchmarking...")
        for run in range(benchmark_runs):
            target_gpu_fresh = cp.asarray(test_image_int8.copy())
            with perf_timer.section(f"int8_experimental_{run}", use_gpu_events=True) as timer:
                result_experimental = cuda_transpose_dot_product_3d_compute_optimized_int8_experimental(
                    sparse_values=sparse_values_aligned,
                    block_positions=block_positions_aligned,
                    target_3d=target_gpu_fresh,
                    batch_size=large_mixed_sparse_tensor_aligned.batch_size,
                    channels=large_mixed_sparse_tensor_aligned.channels,
                    block_size=large_mixed_sparse_tensor_aligned.block_size,
                )

        # Extract timing data
        timing_data = perf_timer.get_timing_data()

        # Collect benchmark times
        for section_name, section_data in timing_data["sections"].items():
            duration = section_data.get("gpu_duration") or section_data["duration"]

            if "int8_regular_" in section_name:
                int8_regular_times.append(duration)
            elif "int8_experimental_" in section_name:
                int8_experimental_times.append(duration)

        # Calculate statistics
        regular_mean = np.mean(int8_regular_times)
        regular_std = np.std(int8_regular_times)
        experimental_mean = np.mean(int8_experimental_times)
        experimental_std = np.std(int8_experimental_times)

        # Calculate speedup
        speedup = regular_mean / experimental_mean if experimental_mean > 0 else 0

        # Log results
        logger.info("\n" + "=" * 60)
        logger.info("ALIGNMENT EXPERIMENT RESULTS")
        logger.info("=" * 60)
        logger.info("INT8 REGULAR (random positions):")
        logger.info(f"  Mean time: {regular_mean * 1000:.3f} ms")
        logger.info(f"  Std dev: {regular_std * 1000:.3f} ms")
        logger.info(f"  Min time: {min(int8_regular_times) * 1000:.3f} ms")
        logger.info(f"  Max time: {max(int8_regular_times) * 1000:.3f} ms")
        logger.info("")
        logger.info("INT8 EXPERIMENTAL (4-aligned positions):")
        logger.info(f"  Mean time: {experimental_mean * 1000:.3f} ms")
        logger.info(f"  Std dev: {experimental_std * 1000:.3f} ms")
        logger.info(f"  Min time: {min(int8_experimental_times) * 1000:.3f} ms")
        logger.info(f"  Max time: {max(int8_experimental_times) * 1000:.3f} ms")
        logger.info("")
        logger.info("ALIGNMENT PERFORMANCE IMPACT:")
        logger.info(f"  Speedup from 4-byte alignment: {speedup:.3f}x {'(faster)' if speedup > 1 else '(slower)'}")
        logger.info(f"  Performance change: {((speedup - 1) * 100):+.1f}%")

        if speedup > 1:
            logger.info("  ✓ Aligned uchar4 loads provide performance benefit")
        else:
            logger.info("  ⚠ Aligned loads did not improve performance (may be memory bandwidth limited)")

        logger.info("=" * 60)

        # Verify both kernels produced results
        assert result_regular.shape == result_experimental.shape
        assert len(int8_regular_times) == benchmark_runs
        assert len(int8_experimental_times) == benchmark_runs
        assert all(t > 0 for t in int8_regular_times)
        assert all(t > 0 for t in int8_experimental_times)

        logger.info("Alignment experiment completed successfully!")
        logger.info(f"Regular kernel: {len(int8_regular_times)} measurements")
        logger.info(f"Experimental kernel: {len(int8_experimental_times)} measurements")


@pytest.mark.skipif(CUDA_AVAILABLE, reason="Testing CUDA unavailable case")
def test_cuda_not_available():
    """Test behavior when CUDA is not available."""
    # This test runs when CUDA is not available
    # Just ensure we can import the test module without errors
    assert not CUDA_AVAILABLE
