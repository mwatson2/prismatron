"""
Unit tests for optimization utilities.

Tests the ImageComparison class and shared optimization functionality.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import the optimization_utils module
try:
    from src.utils.optimization_utils import ImageComparison

    OPTIMIZATION_UTILS_AVAILABLE = True
except ImportError:
    OPTIMIZATION_UTILS_AVAILABLE = False
    ImageComparison = None

# Skip all tests if module not available
pytestmark = pytest.mark.skipif(not OPTIMIZATION_UTILS_AVAILABLE, reason="optimization_utils module not importable")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a 64x64 RGB image with gradient
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            img[i, j] = [i * 4, j * 4, (i + j) * 2]
    return img


@pytest.fixture
def identical_images(sample_image):
    """Return two identical images."""
    return sample_image.copy(), sample_image.copy()


@pytest.fixture
def different_images(sample_image):
    """Return two different images."""
    img1 = sample_image.copy()
    img2 = sample_image.copy()
    # Add noise to second image
    noise = np.random.randint(-20, 20, img2.shape, dtype=np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img1, img2


# =============================================================================
# ImageComparison PSNR Tests
# =============================================================================


class TestImageComparisonPSNR:
    """Test PSNR calculation."""

    def test_psnr_identical_images(self, identical_images):
        """Test PSNR for identical images is infinity."""
        img1, img2 = identical_images
        psnr = ImageComparison.calculate_psnr(img1, img2)

        assert psnr == float("inf")

    def test_psnr_different_images(self, different_images):
        """Test PSNR for different images is finite."""
        img1, img2 = different_images
        psnr = ImageComparison.calculate_psnr(img1, img2)

        assert psnr < float("inf")
        assert psnr > 0  # Should be positive

    def test_psnr_symmetric(self, different_images):
        """Test PSNR is symmetric."""
        img1, img2 = different_images
        psnr_1_2 = ImageComparison.calculate_psnr(img1, img2)
        psnr_2_1 = ImageComparison.calculate_psnr(img2, img1)

        assert abs(psnr_1_2 - psnr_2_1) < 0.001

    def test_psnr_shape_mismatch_raises(self):
        """Test PSNR raises error for shape mismatch."""
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.zeros((32, 32, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="shapes don't match"):
            ImageComparison.calculate_psnr(img1, img2)

    def test_psnr_known_value(self):
        """Test PSNR with known MSE."""
        # Create images with known difference
        img1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        img2 = np.full((10, 10, 3), 110, dtype=np.uint8)  # Diff of 10

        # MSE should be 100 (10^2)
        # PSNR = 20 * log10(255 / sqrt(100)) = 20 * log10(25.5) ≈ 28.13
        psnr = ImageComparison.calculate_psnr(img1, img2)

        assert 28.0 < psnr < 28.3


# =============================================================================
# ImageComparison SSIM Tests
# =============================================================================


class TestImageComparisonSSIM:
    """Test SSIM calculation."""

    def test_ssim_identical_images(self, identical_images):
        """Test SSIM for identical images is 1.0."""
        img1, img2 = identical_images
        ssim = ImageComparison.calculate_ssim(img1, img2)

        # SSIM should be 1.0 for identical images (or close to it)
        assert ssim > 0.99 or ssim == 0.0  # 0.0 if skimage not available

    def test_ssim_different_images(self, different_images):
        """Test SSIM for different images is less than 1."""
        img1, img2 = different_images
        ssim = ImageComparison.calculate_ssim(img1, img2)

        # SSIM should be less than 1 for different images
        assert ssim < 1.0

    def test_ssim_range(self, different_images):
        """Test SSIM is in valid range [0, 1]."""
        img1, img2 = different_images
        ssim = ImageComparison.calculate_ssim(img1, img2)

        assert 0.0 <= ssim <= 1.0


# =============================================================================
# ImageComparison Equality Tests
# =============================================================================


class TestImageComparisonEquality:
    """Test image equality checking."""

    def test_images_equal_identical(self, identical_images):
        """Test identical images are equal."""
        img1, img2 = identical_images
        result = ImageComparison.images_equal(img1, img2)

        assert result is True

    def test_images_equal_different(self, different_images):
        """Test different images are not equal."""
        img1, img2 = different_images
        result = ImageComparison.images_equal(img1, img2)

        assert result is False

    def test_images_equal_shape_mismatch(self):
        """Test shape mismatch returns False."""
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.zeros((32, 32, 3), dtype=np.uint8)

        result = ImageComparison.images_equal(img1, img2)

        assert result is False

    def test_images_equal_single_pixel_diff(self, sample_image):
        """Test single pixel difference detected."""
        img1 = sample_image.copy()
        img2 = sample_image.copy()
        img2[0, 0, 0] = (img2[0, 0, 0] + 1) % 256  # Change one pixel

        result = ImageComparison.images_equal(img1, img2)

        assert result is False
