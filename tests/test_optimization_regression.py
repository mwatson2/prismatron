"""
Regression tests for LED optimization pipeline.

This module tests the complete optimization workflow from input image to rendered output,
ensuring that optimization results remain consistent across code changes.

The tests use the same command workflow as:
    ./tools/standalone_optimizer.py --input env/lib/python3.10/site-packages/sklearn/datasets/images/flower.jpg
                                  --patterns tests/fixtures/test_clean.npz
                                  --output test_images/flower_test.png --verbose

Key features:
- Pixel-perfect regression detection with stored fixtures
- PSNR fallback for quality comparison when pixels differ
- Automatic fixture creation on first run
- Shared utilities with standalone_optimizer.py (no code duplication)
- Tests reproducibility, quality metrics, and basic functionality
- Uses small test patterns (100 LEDs) stored in fixtures/ for git tracking
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

from src.utils.optimization_utils import ImageComparison, OptimizationPipeline

logger = logging.getLogger(__name__)


class TestOptimizationRegression:
    """Regression tests for the complete LED optimization pipeline."""

    # Test configuration
    TEST_IMAGE_PATH = (
        "env/lib/python3.10/site-packages/sklearn/datasets/images/flower.jpg"
    )
    PATTERNS_PATH = "tests/fixtures/test_clean"  # Small pattern for faster tests
    MAX_ITERATIONS = 15  # Reduced for test speed
    PSNR_THRESHOLD = 25.0  # Minimum PSNR for acceptable quality

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def pipeline(self):
        """Create and initialize optimization pipeline."""
        pipeline = OptimizationPipeline(
            diffusion_patterns_path=self.PATTERNS_PATH,
            use_dense=True,  # Use dense tensor optimizer (default)
        )

        success = pipeline.initialize()
        if not success:
            pytest.skip("Cannot initialize optimization pipeline - patterns missing")

        return pipeline

    @pytest.fixture
    def test_image_path(self):
        """Verify test image exists."""
        if not Path(self.TEST_IMAGE_PATH).exists():
            pytest.skip(f"Test image not found: {self.TEST_IMAGE_PATH}")
        return self.TEST_IMAGE_PATH

    @pytest.fixture
    def reference_output_path(self, temp_dir):
        """Path for storing reference output images."""
        return Path(temp_dir) / "reference_output.png"

    def get_expected_output_path(self) -> Path:
        """Get path to expected output fixture."""
        # Store expected outputs in tests/fixtures/
        fixture_dir = Path(__file__).parent / "fixtures"
        fixture_dir.mkdir(exist_ok=True)
        return fixture_dir / "flower_optimization_expected.png"

    def test_optimization_pipeline_functionality(
        self, pipeline, test_image_path, temp_dir
    ):
        """Test that the optimization pipeline runs without errors."""
        # Run the complete pipeline
        original, result, rendered = pipeline.run_full_pipeline(
            test_image_path, max_iterations=self.MAX_ITERATIONS
        )

        # Basic sanity checks
        assert original.shape == (
            480,
            800,
            3,
        ), f"Unexpected original shape: {original.shape}"
        assert rendered.shape == (
            480,
            800,
            3,
        ), f"Unexpected rendered shape: {rendered.shape}"
        assert (
            result.led_values.shape[1] == 3
        ), "LED values should have 3 channels (RGB)"
        assert result.converged, "Optimization should converge"
        assert result.optimization_time > 0, "Optimization should take measurable time"

        # Check that rendered image is reasonable
        assert (
            rendered.min() >= 0 and rendered.max() <= 255
        ), "Rendered values out of range"
        assert not np.all(rendered == 0), "Rendered image should not be all black"
        assert not np.all(rendered == 255), "Rendered image should not be all white"

        logger.info(f"Pipeline test passed - MSE: {result.error_metrics['mse']:.6f}")

    def test_optimization_regression_with_pixel_accuracy(
        self, pipeline, test_image_path, temp_dir
    ):
        """
        Test for pixel-accurate regression detection.

        This test checks if the optimization output exactly matches a stored reference.
        If not pixel-perfect, it falls back to PSNR comparison to detect if the
        optimization has improved or regressed.
        """
        expected_output_path = self.get_expected_output_path()

        # Run optimization
        original, result, rendered = pipeline.run_full_pipeline(
            test_image_path, max_iterations=self.MAX_ITERATIONS
        )

        # Save current output for comparison
        current_output_path = Path(temp_dir) / "current_output.png"
        pipeline.save_image(rendered, str(current_output_path))

        if expected_output_path.exists():
            # Load expected output
            expected_image = cv2.imread(str(expected_output_path))
            expected_image = cv2.cvtColor(expected_image, cv2.COLOR_BGR2RGB)

            # Check for pixel-perfect match
            if ImageComparison.images_equal(rendered, expected_image):
                logger.info("✓ Pixel-perfect match with expected output")
                return

            # Fall back to PSNR comparison
            psnr = ImageComparison.calculate_psnr(expected_image, rendered)
            logger.info(f"PSNR vs expected: {psnr:.2f} dB")

            if psnr >= self.PSNR_THRESHOLD:
                logger.warning(
                    f"Output differs from expected but PSNR {psnr:.2f} >= {self.PSNR_THRESHOLD} dB. "
                    f"Optimization may have improved. Consider updating fixture."
                )
                # Save comparison for manual inspection
                self._save_comparison_image(
                    original, expected_image, rendered, temp_dir
                )
            else:
                # Save comparison for debugging
                self._save_comparison_image(
                    original, expected_image, rendered, temp_dir
                )
                pytest.fail(
                    f"Optimization output differs significantly from expected. "
                    f"PSNR: {psnr:.2f} dB < {self.PSNR_THRESHOLD} dB threshold. "
                    f"This indicates a regression in optimization quality. "
                    f"Check comparison images in {temp_dir}"
                )
        else:
            # No expected output exists - this is the first run
            logger.warning(f"No expected output found at {expected_output_path}")
            logger.info("Creating initial expected output fixture...")

            # Create the fixture directory
            expected_output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save current output as expected
            pipeline.save_image(rendered, str(expected_output_path))

            # Also save metadata about this test run
            metadata_path = expected_output_path.with_suffix(".txt")
            with open(metadata_path, "w") as f:
                f.write(f"Test fixture created for optimization regression test\n")
                f.write(f"Input image: {test_image_path}\n")
                f.write(f"Patterns: {self.PATTERNS_PATH}\n")
                f.write(f"Max iterations: {self.MAX_ITERATIONS}\n")
                f.write(f"LED count: {result.led_values.shape[0]}\n")
                f.write(f"MSE: {result.error_metrics['mse']:.6f}\n")
                f.write(f"Optimization time: {result.optimization_time:.3f}s\n")

            logger.info(f"Expected output saved to: {expected_output_path}")
            logger.info("Re-run test to perform actual regression checking")

    def test_optimization_quality_metrics(self, pipeline, test_image_path):
        """Test that optimization produces reasonable quality metrics."""
        original, result, rendered = pipeline.run_full_pipeline(
            test_image_path, max_iterations=self.MAX_ITERATIONS
        )

        # Calculate quality metrics
        psnr_vs_original = ImageComparison.calculate_psnr(original, rendered)

        # Quality thresholds (tuned for test pattern with 100 LEDs)
        assert (
            result.error_metrics["mse"] < 1.0
        ), f"MSE too high: {result.error_metrics['mse']}"
        assert (
            psnr_vs_original > 12.0
        ), f"PSNR too low vs original: {psnr_vs_original:.2f} dB"

        logger.info(f"Quality metrics passed - PSNR: {psnr_vs_original:.2f} dB")

    def test_reproducibility(self, pipeline, test_image_path):
        """Test that optimization produces reproducible results."""
        # Run optimization twice
        _, result1, rendered1 = pipeline.run_full_pipeline(
            test_image_path, max_iterations=self.MAX_ITERATIONS
        )

        _, result2, rendered2 = pipeline.run_full_pipeline(
            test_image_path, max_iterations=self.MAX_ITERATIONS
        )

        # Results should be identical (optimization is deterministic)
        assert np.array_equal(
            result1.led_values, result2.led_values
        ), "LED values should be reproducible"
        assert np.array_equal(
            rendered1, rendered2
        ), "Rendered images should be reproducible"

        # MSE should be identical
        mse_diff = abs(result1.error_metrics["mse"] - result2.error_metrics["mse"])
        assert mse_diff < 1e-6, f"MSE should be reproducible, diff: {mse_diff}"

        logger.info("Reproducibility test passed")

    def _save_comparison_image(
        self,
        original: np.ndarray,
        expected: np.ndarray,
        current: np.ndarray,
        temp_dir: str,
    ):
        """Save side-by-side comparison of original, expected, and current output."""
        # Ensure all images are the same height
        h = max(original.shape[0], expected.shape[0], current.shape[0])
        original_resized = cv2.resize(original, (original.shape[1], h))
        expected_resized = cv2.resize(expected, (expected.shape[1], h))
        current_resized = cv2.resize(current, (current.shape[1], h))

        # Create side-by-side comparison
        comparison = np.hstack([original_resized, expected_resized, current_resized])

        # Save comparison
        comparison_path = Path(temp_dir) / "comparison.png"
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(comparison_path), comparison_bgr)

        logger.info(f"Comparison saved: {comparison_path}")
        logger.info("Layout: Original | Expected | Current")


class TestImageComparison:
    """Tests for image comparison utilities."""

    def test_psnr_calculation(self):
        """Test PSNR calculation with known values."""
        # Create test images
        img1 = np.full((100, 100, 3), 128, dtype=np.uint8)  # Gray image
        img2 = np.full((100, 100, 3), 138, dtype=np.uint8)  # Slightly different gray

        psnr = ImageComparison.calculate_psnr(img1, img2)

        # PSNR should be finite and reasonable
        assert psnr > 0, "PSNR should be positive"
        assert psnr < 100, "PSNR should be realistic"

        # Perfect match should give infinite PSNR
        psnr_perfect = ImageComparison.calculate_psnr(img1, img1)
        assert psnr_perfect == float(
            "inf"
        ), "Identical images should have infinite PSNR"

    def test_images_equal(self):
        """Test pixel-perfect image comparison."""
        img1 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        img2 = img1.copy()
        img3 = img1.copy()
        img3[0, 0, 0] = (img3[0, 0, 0] + 1) % 256  # Change one pixel

        assert ImageComparison.images_equal(
            img1, img2
        ), "Identical images should be equal"
        assert not ImageComparison.images_equal(
            img1, img3
        ), "Different images should not be equal"


# Pytest configuration for this module
def pytest_configure(config):
    """Configure pytest for regression tests."""
    # Ensure we have necessary dependencies
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        pytest.skip(f"Required dependency missing: {e}", allow_module_level=True)
