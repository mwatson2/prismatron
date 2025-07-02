#!/usr/bin/env python3
"""
Test optimization with normalized diffusion patterns.

This script loads the existing patterns, normalizes each LED's diffusion pattern
to sum to 1.0, and tests if convergence improves.
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


def normalize_mixed_tensor(
    mixed_tensor: SingleBlockMixedSparseTensor,
) -> SingleBlockMixedSparseTensor:
    """
    Normalize the mixed tensor so each LED's diffusion pattern sums to 1.0.
    """
    print("Normalizing mixed tensor patterns...")

    # Create a copy of the tensor
    normalized_values = mixed_tensor.sparse_values.copy()

    # Normalize each LED's pattern across all channels
    for led_idx in range(mixed_tensor.batch_size):
        for channel in range(mixed_tensor.channels):
            pattern = normalized_values[channel, led_idx]  # Shape: (block_size, block_size)
            pattern_sum = float(cp.sum(pattern))

            if pattern_sum > 0:
                # Normalize so the pattern sums to 1.0
                normalized_values[channel, led_idx] = pattern / pattern_sum

    # Create new tensor with normalized values
    normalized_tensor = SingleBlockMixedSparseTensor(
        batch_size=mixed_tensor.batch_size,
        channels=mixed_tensor.channels,
        height=mixed_tensor.height,
        width=mixed_tensor.width,
        block_size=mixed_tensor.block_size,
        device=mixed_tensor.device,
        dtype=mixed_tensor.dtype,
    )

    normalized_tensor.sparse_values = normalized_values
    normalized_tensor.block_positions = mixed_tensor.block_positions.copy()

    print("Normalized tensor created")

    # Check normalization
    sample_sums = []
    for led_idx in range(min(10, mixed_tensor.batch_size)):
        for channel in range(mixed_tensor.channels):
            pattern_sum = float(cp.sum(normalized_tensor.sparse_values[channel, led_idx]))
            sample_sums.append(pattern_sum)

    print(f"Sample pattern sums after normalization: {sample_sums[:5]} (should be ~1.0)")

    return normalized_tensor


def normalize_csc_matrix(csc_matrix, led_count: int):
    """
    Normalize CSC matrix so each LED's diffusion pattern sums to 1.0.
    """
    print("Normalizing CSC matrix...")

    normalized_matrix = csc_matrix.copy()

    # Process each LED (3 columns per LED for RGB)
    for led_idx in range(led_count):
        led_sum = 0.0

        # Calculate total sum across all channels for this LED
        for channel in range(3):
            col_idx = led_idx * 3 + channel
            col_sum = normalized_matrix[:, col_idx].sum()
            led_sum += col_sum

        # Normalize all channels for this LED
        if led_sum > 0:
            for channel in range(3):
                col_idx = led_idx * 3 + channel
                normalized_matrix[:, col_idx] = normalized_matrix[:, col_idx] / led_sum

    print("CSC matrix normalized")
    return normalized_matrix


def test_with_normalization():
    """Test optimization with normalized patterns."""

    # Load test data
    patterns_path = "diffusion_patterns/baseline_realistic.npz"
    image_path = "flower_test.png"

    print(f"Loading patterns from: {patterns_path}")
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Load and normalize mixed tensor
    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
    normalized_mixed_tensor = normalize_mixed_tensor(mixed_tensor)

    # Load and normalize CSC matrix
    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_matrix = diffusion_csc.to_csc_matrix()
    normalized_csc = normalize_csc_matrix(csc_matrix, mixed_tensor.batch_size)

    # Build DIA matrix from normalized CSC
    dia_matrix = DiagonalATAMatrix(led_count=mixed_tensor.batch_size)
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

    print("\n=== Testing Optimization with Normalized Patterns ===")

    # Test with normalized patterns
    result = optimize_frame_led_values(
        target_frame=target_image,
        AT_matrix=normalized_mixed_tensor,
        ATA_matrix=dia_matrix,
        max_iterations=10,
        convergence_threshold=1e-6,
        step_size_scaling=0.8,
        compute_error_metrics=True,
        debug=True,
    )

    print("\n=== Results ===")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"LED values range: [{result.led_values.min()}, {result.led_values.max()}]")

    if result.error_metrics:
        print(f"MSE: {result.error_metrics.get('mse', 'N/A'):.6f}")
        print(f"MAE: {result.error_metrics.get('mae', 'N/A'):.6f}")
        print(f"PSNR: {result.error_metrics.get('psnr', 'N/A'):.2f}")

    if result.step_sizes is not None:
        print(f"Step sizes: {result.step_sizes}")
        print(f"Step size range: [{np.min(result.step_sizes):.6f}, {np.max(result.step_sizes):.6f}]")


if __name__ == "__main__":
    test_with_normalization()
