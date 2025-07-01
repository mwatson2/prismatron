#!/usr/bin/env python3
"""
Test optimization with corrected scaling (adding missing 255*255 normalization).

This adds the missing normalization factor that should be applied to float32 data
to match the int8 kernel behavior.
"""

import sys
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


class NormalizedMixedTensor(SingleBlockMixedSparseTensor):
    """Wrapper that adds missing 255*255 normalization to float32 mixed tensor."""

    def __init__(self, base_tensor):
        # Initialize as SingleBlockMixedSparseTensor so isinstance() works
        super().__init__(
            base_tensor.batch_size,
            base_tensor.channels,
            base_tensor.height,
            base_tensor.width,
            base_tensor.block_size,
            base_tensor.device,
            base_tensor.dtype,
        )

        # Copy data from base tensor
        self.sparse_values = base_tensor.sparse_values
        self.block_positions = base_tensor.block_positions
        self.base_tensor = base_tensor

    def transpose_dot_product_3d(self, target_3d):
        """A^T @ b with missing normalization factor."""
        result = self.base_tensor.transpose_dot_product_3d(target_3d)
        # Apply missing normalization: divide by 255*255 to match int8 kernel
        return result / (255.0 * 255.0)

    def forward_pass_3d(self, led_values):
        """A @ x with missing normalization factor."""
        # Scale led_values up by 255*255 to compensate for pattern scaling
        scaled_led_values = led_values * (255.0 * 255.0)
        return self.base_tensor.forward_pass_3d(scaled_led_values)


def test_with_corrected_scaling():
    """Test optimization with corrected scaling."""

    # Load test data
    patterns_path = "diffusion_patterns/baseline_realistic.npz"
    image_path = "flower_test.png"

    print(f"Loading patterns from: {patterns_path}")
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Load mixed tensor and wrap with normalization
    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    base_mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
    normalized_mixed_tensor = NormalizedMixedTensor(base_mixed_tensor)

    # Load DIA matrix (this should be correct as-is)
    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_matrix = diffusion_csc.to_csc_matrix()

    # Apply same normalization to CSC matrix for A^T A calculation
    normalized_csc = csc_matrix / (255.0 * 255.0)

    dia_matrix = DiagonalATAMatrix(led_count=base_mixed_tensor.batch_size)
    led_positions = patterns_data["led_positions"]

    # Suppress DIA build output
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    dia_matrix.build_from_diffusion_matrix(normalized_csc, led_positions)
    sys.stdout = old_stdout

    # Load test image
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((800, 480))
    target_image = np.array(image, dtype=np.uint8)

    print(f"\n=== Testing with Corrected Scaling ===")

    # Test A^T @ b scaling first
    target_planar = target_image.astype(np.float32) / 255.0
    target_planar = target_planar.transpose(2, 0, 1)  # (3, H, W)
    target_gpu = cp.asarray(target_planar)

    ATb = normalized_mixed_tensor.transpose_dot_product_3d(target_gpu)
    print(
        f"Corrected A^T @ b range: [{float(cp.min(ATb)):.6f}, {float(cp.max(ATb)):.6f}]"
    )
    print(f"Corrected A^T @ b mean: {float(cp.mean(ATb)):.6f}")

    # Test optimization with much smaller step size scaling
    result = optimize_frame_led_values(
        target_frame=target_image,
        AT_matrix=normalized_mixed_tensor,
        ATA_matrix=dia_matrix,
        max_iterations=10,
        convergence_threshold=1e-6,
        step_size_scaling=0.000001,  # Even smaller to compensate for scaling
        compute_error_metrics=True,
        debug=True,
    )

    print(f"\n=== Results with Corrected Scaling ===")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"LED values range: [{result.led_values.min()}, {result.led_values.max()}]")

    if result.error_metrics:
        print(f"MSE: {result.error_metrics.get('mse', 'N/A'):.6f}")
        print(f"MAE: {result.error_metrics.get('mae', 'N/A'):.6f}")
        print(f"PSNR: {result.error_metrics.get('psnr', 'N/A'):.2f}")

    if result.step_sizes is not None:
        print(f"Step sizes: {result.step_sizes}")
        print(
            f"Step size range: [{np.min(result.step_sizes):.6f}, {np.max(result.step_sizes):.6f}]"
        )
        print(f"Step size mean: {np.mean(result.step_sizes):.6f}")


if __name__ == "__main__":
    test_with_corrected_scaling()
