#!/usr/bin/env python3
"""
Simple test script to run batch frame optimizer for tensor core profiling.
"""

from unittest.mock import Mock

import cupy as cp
import numpy as np

from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def main():
    print("Setting up test data...")

    # Create test data
    batch_size = 8
    led_count = 100
    target_frames = np.random.randint(0, 255, (batch_size, 3, 480, 800), dtype=np.uint8)

    # Create mock matrices
    mock_at_matrix = Mock(spec=SingleBlockMixedSparseTensor)
    mock_ata_matrix = Mock(spec=DiagonalATAMatrix)
    mock_ata_inverse = np.random.random((3, led_count, led_count)).astype(np.float32)

    # Mock DIA matrices for ATA conversion
    mock_dia_matrices = []
    for channel in range(3):
        diag_data = np.random.random(led_count).astype(np.float32)
        mock_dia = Mock()
        mock_dia.toarray.return_value = np.diag(diag_data)
        mock_dia_matrices.append(mock_dia)

    mock_ata_matrix.get_channel_dia_matrix.side_effect = mock_dia_matrices

    # Mock AT calculation to return realistic ATb values
    def mock_calculate_atb(frames, at_matrix, debug=False):
        return np.random.random((3, led_count)).astype(np.float32)

    # Patch the calculate_atb function
    import src.utils.batch_frame_optimizer

    original_calculate_atb = src.utils.batch_frame_optimizer._calculate_atb
    src.utils.batch_frame_optimizer._calculate_atb = mock_calculate_atb

    print("Running batch frame optimization...")
    try:
        result = optimize_batch_frames_led_values(
            target_frames,
            mock_at_matrix,
            mock_ata_matrix,
            mock_ata_inverse,
            max_iterations=5,
            debug=True,
            enable_timing=True,
        )

        print("Optimization completed:")
        print(f"  - Shape: {result.led_values.shape}")
        print(f"  - Iterations: {result.iterations}")
        print(f"  - Converged: {result.converged}")
        print(f"  - Timing data available: {result.timing_data is not None}")

        if result.timing_data:
            print("  - Timing breakdown:")
            for section, time_ms in result.timing_data.items():
                print(f"    {section}: {time_ms:.2f}ms")

    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Restore original function
        src.utils.batch_frame_optimizer._calculate_atb = original_calculate_atb


if __name__ == "__main__":
    main()
