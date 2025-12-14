"""
Shared optimization utilities for LED optimization testing and production.

This module contains shared functionality extracted from standalone_optimizer.py
for use in both production code and regression tests.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import sys

from ..const import FRAME_HEIGHT, FRAME_WIDTH

# Import sparse optimizer from archive
archive_path = str(Path(__file__).parent.parent.parent / "archive")
sys.path.insert(0, archive_path)
from led_optimizer_sparse import LEDOptimizer as SparseLEDOptimizer
from led_optimizer_sparse import OptimizationResult

# Import moved to functions to avoid circular dependency

logger = logging.getLogger(__name__)


class ImageComparison:
    """Utility class for comparing images with various metrics."""

    @staticmethod
    def calculate_psnr(original: np.ndarray, comparison: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio between two images.

        Args:
            original: Original image (H, W, 3) in range [0, 255]
            comparison: Comparison image (H, W, 3) in range [0, 255]

        Returns:
            PSNR value in dB (higher is better)
        """
        if original.shape != comparison.shape:
            raise ValueError(f"Image shapes don't match: {original.shape} vs {comparison.shape}")

        # Convert to float32 for precision
        img1 = original.astype(np.float32)
        img2 = comparison.astype(np.float32)

        # Calculate MSE
        mse = np.mean((img1 - img2) ** 2)

        if mse == 0:
            return float("inf")  # Perfect match

        # Calculate PSNR
        max_pixel_value = 255.0
        psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
        return psnr

    @staticmethod
    def calculate_ssim(original: np.ndarray, comparison: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index between two images.

        Args:
            original: Original image (H, W, 3) in range [0, 255]
            comparison: Comparison image (H, W, 3) in range [0, 255]

        Returns:
            SSIM value in range [0, 1] (higher is better)
        """
        try:
            from skimage.metrics import structural_similarity as ssim

            # Convert to grayscale for SSIM calculation
            gray1 = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(comparison, cv2.COLOR_RGB2GRAY)

            return ssim(gray1, gray2, data_range=255)
        except ImportError:
            logger.warning("scikit-image not available, SSIM calculation skipped")
            return 0.0

    @staticmethod
    def images_equal(img1: np.ndarray, img2: np.ndarray) -> bool:
        """
        Check if two images are pixel-perfect matches.

        Args:
            img1: First image
            img2: Second image

        Returns:
            True if images are identical
        """
        if img1.shape != img2.shape:
            return False
        return np.array_equal(img1, img2)


class OptimizationPipeline:
    """
    Complete optimization pipeline for LED image processing.

    This class encapsulates the full workflow from input image to rendered output,
    shared between standalone_optimizer.py and regression tests.
    """

    def __init__(self, diffusion_patterns_path: str, use_dense: bool = True):
        """
        Initialize optimization pipeline.

        Args:
            diffusion_patterns_path: Path to diffusion patterns file
            use_dense: If True, use dense tensor optimizer; if False, use sparse optimizer
                (default: True)
        """
        self.diffusion_patterns_path = self._normalize_patterns_path(diffusion_patterns_path)
        self.use_dense = use_dense

        # Initialize the appropriate optimizer
        if use_dense:
            from ..consumer.led_optimizer import LEDOptimizer

            self.optimizer = LEDOptimizer(
                diffusion_patterns_path=self.diffusion_patterns_path,
                enable_performance_timing=True,
            )
        else:
            self.optimizer = SparseLEDOptimizer(diffusion_patterns_path=self.diffusion_patterns_path)

        self.initialized = False

    def _normalize_patterns_path(self, patterns_path: str) -> str:
        """Normalize patterns path to remove .npz extension."""
        if patterns_path.endswith("_matrix.npz"):
            return patterns_path.replace("_matrix.npz", "")
        elif patterns_path.endswith(".npz"):
            return patterns_path.replace(".npz", "")
        return patterns_path

    def initialize(self) -> bool:
        """
        Initialize the optimization pipeline.

        Returns:
            True if initialization successful
        """
        self.initialized = self.optimizer.initialize()

        if self.initialized:
            stats = self.optimizer.get_optimizer_stats()
            optimizer_type = "Dense" if self.use_dense else "Sparse"
            logger.info(f"{optimizer_type} optimization pipeline initialized")

            # Handle different stat formats between optimizers
            if hasattr(stats, "get"):
                device = stats.get("device", "unknown")
                led_count = stats.get("led_count", "unknown")
                logger.info(f"Device: {device}")
                logger.info(f"LED count: {led_count}")

                if self.use_dense:
                    if "ata_tensor_shape" in stats:
                        logger.info(f"ATA tensor shape: {stats['ata_tensor_shape']}")
                        logger.info(f"ATA memory: {stats.get('ata_memory_mb', 'N/A'):.1f}MB")
                else:
                    if "matrix_shape" in stats:
                        logger.info(f"Matrix shape: {stats['matrix_shape']}")
                        if "sparsity_percent" in stats:
                            logger.info(f"Sparsity: {stats['sparsity_percent']:.3f}%")

        return self.initialized

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and resize image to target frame dimensions.

        Args:
            image_path: Path to input image

        Returns:
            RGB image array (FRAME_HEIGHT, FRAME_WIDTH, 3) in range [0, 255]
        """
        logger.info(f"Loading image: {image_path}")

        # Try PIL first if available for better quality
        if PIL_AVAILABLE:
            try:
                img: Image.Image = Image.open(image_path)
                img = img.convert("RGB")
                img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.LANCZOS)
                image = np.array(img, dtype=np.uint8)
            except Exception as e:
                logger.warning(f"PIL failed: {e}, falling back to OpenCV")
                image = self._load_with_opencv(image_path)
        else:
            image = self._load_with_opencv(image_path)

        logger.info(f"Loaded image shape: {image.shape}")
        return image

    def _load_with_opencv(self, image_path: str) -> np.ndarray:
        """Load image using OpenCV."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
        return image.astype(np.uint8)

    def optimize_image(self, target_image: np.ndarray, max_iterations: Optional[int] = None) -> OptimizationResult:
        """
        Optimize LED values for target image.

        Args:
            target_image: RGB target image (FRAME_HEIGHT, FRAME_WIDTH, 3)
            max_iterations: Maximum optimization iterations (None uses optimizer default)

        Returns:
            OptimizationResult with LED values and metrics
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        logger.info(f"Optimizing image shape: {target_image.shape}")

        # Use debug version for testing/analysis tools to get error metrics
        if self.use_dense:
            result = self.optimizer.optimize_frame(target_frame=target_image, debug=True, max_iterations=max_iterations)
        else:
            result = self.optimizer.optimize_frame(target_frame=target_image, max_iterations=max_iterations)

        logger.info(f"Optimization completed in {result.optimization_time:.3f}s")
        if result.error_metrics:
            logger.info(f"MSE: {result.error_metrics.get('mse', 'N/A'):.6f}")
        else:
            logger.info("No error metrics computed (production mode)")

        return result

    def render_result(self, result: OptimizationResult) -> np.ndarray:
        """
        Render optimization result back to image using sparse matrix reconstruction.

        Args:
            result: OptimizationResult from optimize_image()

        Returns:
            Rendered RGB image (FRAME_HEIGHT, FRAME_WIDTH, 3) in range [0, 255]
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")

        logger.info("Rendering result using sparse matrix reconstruction...")

        # Handle different optimizer types
        if self.use_dense:
            # Dense optimizer: use sparse matrices kept for A^T*b calculation
            if (
                self.optimizer._A_r_csc_gpu is None
                or self.optimizer._A_g_csc_gpu is None
                or self.optimizer._A_b_csc_gpu is None
            ):
                raise RuntimeError("Sparse matrices not loaded in optimizer")
            A_r = self.optimizer._A_r_csc_gpu.tocsr()
            A_g = self.optimizer._A_g_csc_gpu.tocsr()
            A_b = self.optimizer._A_b_csc_gpu.tocsr()

            # Convert to CPU for rendering
            try:
                import cupy as cp

                A_r_cpu = A_r.get().tocsr()
                A_g_cpu = A_g.get().tocsr()
                A_b_cpu = A_b.get().tocsr()
            except Exception:
                A_r_cpu = A_r
                A_g_cpu = A_g
                A_b_cpu = A_b
        else:
            # Sparse optimizer: use combined matrix
            A_csr = getattr(self.optimizer, "_A_combined_csr_cpu", None)
            if A_csr is None:
                raise RuntimeError("Sparse matrix _A_combined_csr_cpu not available in optimizer")

        # Convert LED values from uint8 [0,255] to float32 [0,1]
        led_values_normalized = result.led_values.astype(np.float32) / 255.0

        # Reconstruct RGB channels
        reconstructed_rgb = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)

        logger.info(f"LED values shape: {result.led_values.shape}")

        if self.use_dense:
            # Dense optimizer: reconstruct each channel separately
            logger.info(f"Dense reconstruction - RGB matrix shapes: {A_r_cpu.shape}")

            reconstructed_r = A_r_cpu @ led_values_normalized[:, 0]
            reconstructed_g = A_g_cpu @ led_values_normalized[:, 1]
            reconstructed_b = A_b_cpu @ led_values_normalized[:, 2]

            reconstructed_rgb[:, :, 0] = reconstructed_r.reshape((FRAME_HEIGHT, FRAME_WIDTH))
            reconstructed_rgb[:, :, 1] = reconstructed_g.reshape((FRAME_HEIGHT, FRAME_WIDTH))
            reconstructed_rgb[:, :, 2] = reconstructed_b.reshape((FRAME_HEIGHT, FRAME_WIDTH))
        else:
            # Sparse optimizer: use combined block diagonal matrix
            assert A_csr is not None  # Verified in sparse branch above
            logger.info(f"Sparse reconstruction - Matrix shape: {A_csr.shape}")

            # For combined block diagonal matrix, we need to reconstruct differently
            # The combined matrix has structure: [[A_r, 0, 0], [0, A_g, 0], [0, 0, A_b]]
            # LED values are stacked: [R_leds; G_leds; B_leds]
            led_count = result.led_values.shape[0]
            led_combined = np.zeros(3 * led_count, dtype=np.float32)
            led_combined[:led_count] = led_values_normalized[:, 0]  # R
            led_combined[led_count : 2 * led_count] = led_values_normalized[:, 1]  # G
            led_combined[2 * led_count :] = led_values_normalized[:, 2]  # B

            # Reconstruct all channels at once with combined matrix
            reconstructed_flat = A_csr @ led_combined  # Shape: (3 * pixels,)

            # Reshape back to (height, width, 3)
            pixels = FRAME_HEIGHT * FRAME_WIDTH
            reconstructed_rgb[:, :, 0] = reconstructed_flat[:pixels].reshape((FRAME_HEIGHT, FRAME_WIDTH))
            reconstructed_rgb[:, :, 1] = reconstructed_flat[pixels : 2 * pixels].reshape((FRAME_HEIGHT, FRAME_WIDTH))
            reconstructed_rgb[:, :, 2] = reconstructed_flat[2 * pixels :].reshape((FRAME_HEIGHT, FRAME_WIDTH))

        # Convert to uint8 and clamp to valid range
        result_image = np.clip(reconstructed_rgb * 255, 0, 255).astype(np.uint8)

        logger.info(f"Rendered image shape: {result_image.shape}")
        return result_image

    def run_full_pipeline(
        self, input_path: str, max_iterations: Optional[int] = None
    ) -> Tuple[np.ndarray, OptimizationResult, np.ndarray]:
        """
        Run the complete optimization pipeline.

        Args:
            input_path: Path to input image
            max_iterations: Maximum optimization iterations (None uses optimizer default)

        Returns:
            Tuple of (original_image, optimization_result, rendered_image)
        """
        # Load input image
        original_image = self.load_image(input_path)

        # Optimize
        result = self.optimize_image(original_image, max_iterations)

        # Render result
        rendered_image = self.render_result(result)

        return original_image, result, rendered_image

    def save_image(self, image: np.ndarray, output_path: str):
        """
        Save image to file.

        Args:
            image: RGB image array to save
            output_path: Output file path
        """
        if PIL_AVAILABLE:
            img = Image.fromarray(image)
            img.save(output_path)
        else:
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, bgr_image)

        logger.info(f"Saved image to: {output_path}")
