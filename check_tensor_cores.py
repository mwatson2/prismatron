#!/usr/bin/env python3
"""
Check tensor core usage with CuPy profiling.
"""

from unittest.mock import Mock

import cupy as cp
import numpy as np
from cupyx.profiler import benchmark

from src.utils.batch_frame_optimizer import optimize_batch_frames_led_values
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def check_gpu_info():
    """Check GPU capabilities."""
    print("GPU Information:")
    print(f"  Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(
        f"  Compute capability: {cp.cuda.runtime.getDeviceProperties(0)['major']}.{cp.cuda.runtime.getDeviceProperties(0)['minor']}"
    )
    print(f"  CuPy version: {cp.__version__}")

    # Check if tensor cores are supported
    props = cp.cuda.runtime.getDeviceProperties(0)
    major = props["major"]
    minor = props["minor"]

    # Tensor cores are available on Volta (7.0+), Turing (7.5+), Ampere (8.0+), etc.
    if major >= 7:
        print("  Tensor cores: Likely supported")
    else:
        print("  Tensor cores: Not supported")

    return major, minor


def test_matrix_operations():
    """Test matrix operations that might use tensor cores."""
    print("\nTesting matrix operations...")

    # Create test matrices with shapes that favor tensor cores
    # Tensor cores work best with specific shapes and data types
    batch_size = 8
    m, n, k = 128, 128, 128  # Common tensor core dimensions

    # Test different data types
    for dtype in [cp.float32]:  # CuPy random only supports float32/float64
        print(f"\nTesting {dtype} matrices...")

        # Create matrices on GPU
        A = cp.random.random((batch_size, m, k), dtype=dtype)
        B = cp.random.random((batch_size, k, n), dtype=dtype)

        # Also test float16 by converting
        if dtype == cp.float32:
            A_fp16 = A.astype(cp.float16)
            B_fp16 = B.astype(cp.float16)

            print(f"  Testing float16 matrices (converted from float32)...")
            with cp.cuda.profiler.profile():
                result_fp16 = benchmark(lambda: cp.matmul(A_fp16, B_fp16), n_repeat=10, n_warmup=2)
            print(
                f"  Matrix multiply float16 ({m}x{k} @ {k}x{n}) batch={batch_size}: {result_fp16.cpu_times.mean():.4f}s"
            )

        # Benchmark matrix multiplication
        with cp.cuda.profiler.profile():
            result = benchmark(lambda: cp.matmul(A, B), n_repeat=10, n_warmup=2)

        print(f"  Matrix multiply ({m}x{k} @ {k}x{n}) batch={batch_size}: {result.cpu_times.mean():.4f}s")


def test_batch_optimizer():
    """Test the batch frame optimizer."""
    print("\nTesting batch frame optimizer...")

    # Create test data
    batch_size = 8
    led_count = 256  # Use power of 2 for better tensor core utilization
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

    # Mock AT calculation
    def mock_calculate_atb(frames, at_matrix, debug=False):
        return np.random.random((3, led_count)).astype(np.float32)

    # Patch the calculate_atb function
    import src.utils.batch_frame_optimizer

    original_calculate_atb = src.utils.batch_frame_optimizer._calculate_atb
    src.utils.batch_frame_optimizer._calculate_atb = mock_calculate_atb

    try:
        # Enable profiling
        with cp.cuda.profiler.profile():
            result = optimize_batch_frames_led_values(
                target_frames,
                mock_at_matrix,
                mock_ata_matrix,
                mock_ata_inverse,
                max_iterations=3,
                debug=False,
                enable_timing=True,
            )

        print(f"Batch optimization completed:")
        print(f"  - Shape: {result.led_values.shape}")
        print(f"  - Iterations: {result.iterations}")

        if result.timing_data:
            print("  - GPU timing breakdown:")
            for section, time_ms in result.timing_data.items():
                if "batch" in section:
                    print(f"    {section}: {time_ms:.2f}ms")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Restore original function
        src.utils.batch_frame_optimizer._calculate_atb = original_calculate_atb


def main():
    # Check GPU info
    major, minor = check_gpu_info()

    # Test matrix operations
    test_matrix_operations()

    # Test batch optimizer
    test_batch_optimizer()

    print("\nNote: To definitively check tensor core usage, use:")
    print("  - nsight compute (requires root privileges)")
    print("  - nsight systems for timeline profiling")
    print("  - Check if operations use cuBLAS/cuDNN with tensor core support")


if __name__ == "__main__":
    main()
