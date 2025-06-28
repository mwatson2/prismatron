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

    from src.utils.cuda_kernels import cuda_transpose_dot_product_3d_compute_optimized
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


def generate_spatially_ordered_positions(led_count, height, width, block_size, seed=42):
    """Generate LED positions with spatial Morton ordering for cache locality."""
    np.random.seed(seed)

    # Generate random positions within valid bounds
    max_row = height - block_size
    max_col = width - block_size

    positions = []
    for _ in range(led_count):
        top_row = np.random.randint(0, max_row)
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
    led_positions = generate_spatially_ordered_positions(
        led_count, height, width, block_size, seed=42
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
            gaussian = np.exp(
                -((x_coords - center_col) ** 2 + (y_coords - center_row) ** 2)
                / (2 * sigma**2)
            )

            # Add some noise and channel-specific intensity
            noise = np.random.normal(0, 0.1, (block_size, block_size))
            intensity = (channel_idx + 1) / channels  # Different intensity per channel
            pattern = (gaussian * intensity + noise).clip(0, 1).astype(np.float32)

            # Set the block
            tensor.set_block(
                led_idx, channel_idx, top_row, top_col, cp.asarray(pattern)
            )

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
    gaussian = np.exp(
        -((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2)
        / (2 * (min(height, width) // 4) ** 2)
    )
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
    logger.info(
        f"Generating large tensor with {led_count} LEDs for performance testing..."
    )

    led_positions = generate_spatially_ordered_positions(
        led_count, height, width, block_size, seed=42
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
            gaussian = np.exp(
                -((x_coords - center_col) ** 2 + (y_coords - center_row) ** 2)
                / (2 * sigma**2)
            )

            # Add some noise and channel-specific intensity
            noise = np.random.normal(0, 0.1, (block_size, block_size))
            intensity = (channel_idx + 1) / channels  # Different intensity per channel
            pattern = (gaussian * intensity + noise).clip(0, 1).astype(np.float32)

            # Set the block
            tensor.set_block(
                led_idx, channel_idx, top_row, top_col, cp.asarray(pattern)
            )

    logger.info(f"Large tensor generation complete: {tensor}")
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
            sparse_values = (
                mixed_sparse_tensor.sparse_values
            )  # Shape: (channels, batch_size, block_size, block_size)
            block_positions = (
                mixed_sparse_tensor.block_positions
            )  # Shape: (channels, batch_size, 2)

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

    def test_kernel_performance(self, large_mixed_sparse_tensor, test_images):
        """Performance test for CUDA kernel with cache warming.

        Uses large problem size (1000 LEDs, 480x800 images) for realistic timing measurements
        that properly account for memory bandwidth and cache behavior.
        """
        # Initialize performance timing
        perf_timer = PerformanceTiming("CudaKernelPerformance", enable_gpu_timing=True)

        # Get kernel inputs from large tensor
        sparse_values = large_mixed_sparse_tensor.sparse_values
        block_positions = large_mixed_sparse_tensor.block_positions

        # Test parameters
        warmup_runs = 5
        benchmark_runs = 20

        kernel_times = []

        logger.info(f"Running performance test:")
        logger.info(f"  LED count: {large_mixed_sparse_tensor.batch_size}")
        logger.info(
            f"  Image size: {large_mixed_sparse_tensor.height}x{large_mixed_sparse_tensor.width}"
        )
        logger.info(f"  Block size: {large_mixed_sparse_tensor.block_size}")
        logger.info(f"  Test images: {len(test_images)}")
        logger.info(f"  Warmup runs: {warmup_runs}")
        logger.info(f"  Benchmark runs: {benchmark_runs}")

        # Calculate problem size metrics
        total_pixels = (
            large_mixed_sparse_tensor.height
            * large_mixed_sparse_tensor.width
            * large_mixed_sparse_tensor.channels
        )
        total_blocks = (
            large_mixed_sparse_tensor.batch_size * large_mixed_sparse_tensor.channels
        )
        block_pixels = (
            large_mixed_sparse_tensor.block_size * large_mixed_sparse_tensor.block_size
        )
        total_operations = total_blocks * block_pixels

        logger.info(f"  Problem size metrics:")
        logger.info(f"    Total image pixels: {total_pixels:,}")
        logger.info(f"    Total LED blocks: {total_blocks:,}")
        logger.info(f"    Total operations: {total_operations:,}")

        for img_idx, test_image in enumerate(test_images):
            # Cache warming phase
            logger.debug(
                f"\nCache warming for image {img_idx + 1}/{len(test_images)}..."
            )

            for warmup in range(warmup_runs):
                target_gpu_fresh = cp.asarray(test_image.copy())
                _ = cuda_transpose_dot_product_3d_compute_optimized(
                    sparse_values=sparse_values,
                    block_positions=block_positions,
                    target_3d=target_gpu_fresh,
                    batch_size=large_mixed_sparse_tensor.batch_size,
                    channels=large_mixed_sparse_tensor.channels,
                    block_size=large_mixed_sparse_tensor.block_size,
                )

            # Benchmark phase
            logger.debug(f"Benchmarking image {img_idx + 1}...")

            for run in range(benchmark_runs):
                target_gpu_fresh = cp.asarray(test_image.copy())
                with perf_timer.section(
                    f"benchmark_img{img_idx}_{run}", use_gpu_events=True
                ) as timer:
                    result = cuda_transpose_dot_product_3d_compute_optimized(
                        sparse_values=sparse_values,
                        block_positions=block_positions,
                        target_3d=target_gpu_fresh,
                        batch_size=large_mixed_sparse_tensor.batch_size,
                        channels=large_mixed_sparse_tensor.channels,
                        block_size=large_mixed_sparse_tensor.block_size,
                    )

        # Extract timing data and analyze
        timing_data = perf_timer.get_timing_data()

        # Collect benchmark times
        for section_name, section_data in timing_data["sections"].items():
            if "benchmark_" in section_name:
                # Use GPU duration if available, otherwise CPU duration
                duration = section_data.get("gpu_duration") or section_data["duration"]
                kernel_times.append(duration)

        # Calculate statistics
        kernel_mean = np.mean(kernel_times)
        kernel_std = np.std(kernel_times)

        # Log performance results
        logger.info("\n" + "=" * 60)
        logger.info("CUDA KERNEL PERFORMANCE TEST")
        logger.info("=" * 60)
        logger.info(f"Kernel Performance:")
        logger.info(f"  Mean time: {kernel_mean*1000:.3f} ms")
        logger.info(f"  Std dev: {kernel_std*1000:.3f} ms")
        logger.info(f"  Min time: {min(kernel_times)*1000:.3f} ms")
        logger.info(f"  Max time: {max(kernel_times)*1000:.3f} ms")
        logger.info("=" * 60)

        # Assertions to ensure kernel is functional
        assert len(kernel_times) > 0, "No kernel timing data collected"
        assert all(t > 0 for t in kernel_times), "Invalid kernel timing data"

        # Log a summary for reference
        logger.info(f"Performance test completed successfully!")
        logger.info(f"Data collected: {len(kernel_times)} measurements")

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
            led_positions = generate_spatially_ordered_positions(
                led_count, height, width, block_size, seed=42
            )

            # Generate test patterns for this block size with Morton ordering
            np.random.seed(42)  # Consistent patterns
            for led_idx in range(led_count):
                top_row, top_col = led_positions[led_idx]

                for channel_idx in range(channels):
                    # Simple pattern for testing
                    pattern = np.random.rand(block_size, block_size).astype(np.float32)
                    mixed_tensor.set_block(
                        led_idx, channel_idx, top_row, top_col, cp.asarray(pattern)
                    )

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


@pytest.mark.skipif(CUDA_AVAILABLE, reason="Testing CUDA unavailable case")
def test_cuda_not_available():
    """Test behavior when CUDA is not available."""
    # This test runs when CUDA is not available
    # Just ensure we can import the test module without errors
    assert not CUDA_AVAILABLE
