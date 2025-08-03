#!/usr/bin/env python3
"""
Pytest suite for BatchSymmetricDiagonalATAMatrix WMMA tensor core implementation.

Tests correctness and basic performance of the batch WMMA kernel across
different matrix sizes and complexity levels.
"""

import os
import sys

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix


def create_test_matrix(led_count, num_diagonals):
    """Create a test matrix with specified size and diagonal count."""
    ata_matrix = DiagonalATAMatrix(led_count=led_count, crop_size=64)

    # Create symmetric diagonal pattern
    max_offset = (num_diagonals - 1) // 2
    offsets = list(range(-max_offset, max_offset + 1))
    k = len(offsets)

    # Create simple diagonal data with known pattern
    dia_data_cpu = np.zeros((3, k, led_count), dtype=np.float32)

    for channel in range(3):
        for diag_idx, offset in enumerate(offsets):
            abs_offset = abs(offset)
            if abs_offset == 0:
                # Main diagonal: strong coupling
                dia_data_cpu[channel, diag_idx, :] = 5.0
            else:
                # Off-diagonals: decay with distance
                decay = np.exp(-abs_offset / 10.0)
                dia_data_cpu[channel, diag_idx, :] = decay

    # Set matrix data
    ata_matrix.dia_data_cpu = dia_data_cpu
    ata_matrix.dia_offsets = np.array(offsets, dtype=np.int32)
    ata_matrix.k = k
    ata_matrix.bandwidth = max_offset
    ata_matrix.sparsity = 0.0
    ata_matrix.nnz = np.count_nonzero(dia_data_cpu)

    # Convert to GPU
    if CUPY_AVAILABLE:
        ata_matrix.dia_data_gpu = cp.asarray(dia_data_cpu)
        ata_matrix.dia_offsets_gpu = cp.asarray(offsets, dtype=cp.int32)

    return ata_matrix


def verify_correctness(symmetric_matrix, batch_matrix, test_input, tolerance=1e-2):
    """Verify that sequential and batch methods produce the same results."""
    batch_size = test_input.shape[0]

    # Sequential processing
    sequential_results = []
    for frame_idx in range(batch_size):
        frame_input = test_input[frame_idx]  # Shape: (3, led_count)
        result = symmetric_matrix.multiply_3d(frame_input, use_custom_kernel=True, optimized_kernel=True)
        sequential_results.append(result)

    sequential_batch = cp.stack(sequential_results, axis=0)

    # Batch processing
    batch_result = batch_matrix.multiply_batch_3d(test_input, optimized_kernel=False, debug_logging=False)

    # Compare results
    sequential_cpu = cp.asnumpy(sequential_batch)
    batch_cpu = cp.asnumpy(batch_result)

    diff = np.abs(sequential_cpu - batch_cpu)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = max_diff / np.max(np.abs(sequential_cpu)) if np.max(np.abs(sequential_cpu)) > 0 else 0

    return {"max_diff": max_diff, "mean_diff": mean_diff, "rel_diff": rel_diff, "success": max_diff < tolerance}


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestBatchSymmetricWMMA:
    """Test suite for BatchSymmetricDiagonalATAMatrix WMMA implementation."""

    def test_16x16_single_block(self):
        """Test 16x16 matrix with single block (1 block diagonal)."""
        led_count = 16
        num_diagonals = 11
        batch_size = 16

        # Create matrices
        regular_matrix = create_test_matrix(led_count, num_diagonals)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)
        batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=batch_size)

        # Create test input
        np.random.seed(42)
        test_input = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.1
        test_input_gpu = cp.asarray(test_input)

        # Verify correctness
        result = verify_correctness(symmetric_matrix, batch_matrix, test_input_gpu)

        assert result["success"], (
            f"16x16 single block test failed: max_diff={result['max_diff']:.6f}, " f"rel_diff={result['rel_diff']:.3f}%"
        )
        assert result["rel_diff"] < 0.01, f"Relative error too large: {result['rel_diff']:.3f}%"

    def test_32x32_main_diagonal(self):
        """Test 32x32 matrix with main diagonal only (2 blocks)."""
        led_count = 32
        num_diagonals = 21
        batch_size = 16

        # Create matrices
        regular_matrix = create_test_matrix(led_count, num_diagonals)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)
        batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=batch_size)

        # Create test input
        np.random.seed(42)
        test_input = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.1
        test_input_gpu = cp.asarray(test_input)

        # Verify correctness
        result = verify_correctness(symmetric_matrix, batch_matrix, test_input_gpu)

        assert result["success"], (
            f"32x32 main diagonal test failed: max_diff={result['max_diff']:.6f}, "
            f"rel_diff={result['rel_diff']:.3f}%"
        )
        assert result["rel_diff"] < 0.01, f"Relative error too large: {result['rel_diff']:.3f}%"

    def test_48x48_off_diagonal(self):
        """Test 48x48 matrix with off-diagonal blocks (5 blocks)."""
        led_count = 48
        num_diagonals = 31
        batch_size = 16

        # Create matrices
        regular_matrix = create_test_matrix(led_count, num_diagonals)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)
        batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=batch_size)

        # Create test input
        np.random.seed(42)
        test_input = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.1
        test_input_gpu = cp.asarray(test_input)

        # Verify correctness
        result = verify_correctness(symmetric_matrix, batch_matrix, test_input_gpu)

        assert result["success"], (
            f"48x48 off-diagonal test failed: max_diff={result['max_diff']:.6f}, " f"rel_diff={result['rel_diff']:.3f}%"
        )
        assert result["rel_diff"] < 0.01, f"Relative error too large: {result['rel_diff']:.3f}%"

    def test_64x64_complex(self):
        """Test 64x64 matrix with complex diagonal pattern (4x4 blocks)."""
        led_count = 64
        num_diagonals = 41
        batch_size = 16

        # Create matrices
        regular_matrix = create_test_matrix(led_count, num_diagonals)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)
        batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=batch_size)

        # Create test input
        np.random.seed(42)
        test_input = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.1
        test_input_gpu = cp.asarray(test_input)

        # Verify correctness
        result = verify_correctness(symmetric_matrix, batch_matrix, test_input_gpu)

        assert result["success"], (
            f"64x64 complex test failed: max_diff={result['max_diff']:.6f}, " f"rel_diff={result['rel_diff']:.3f}%"
        )
        assert result["rel_diff"] < 0.01, f"Relative error too large: {result['rel_diff']:.3f}%"

    def test_256x256_medium_scale(self):
        """Test 256x256 matrix for medium-scale performance."""
        led_count = 256
        num_diagonals = 81
        batch_size = 16

        # Create matrices
        regular_matrix = create_test_matrix(led_count, num_diagonals)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)
        batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=batch_size)

        # Create test input
        np.random.seed(42)
        test_input = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.1
        test_input_gpu = cp.asarray(test_input)

        # Verify correctness
        result = verify_correctness(symmetric_matrix, batch_matrix, test_input_gpu)

        assert result["success"], (
            f"256x256 medium scale test failed: max_diff={result['max_diff']:.6f}, "
            f"rel_diff={result['rel_diff']:.3f}%"
        )
        assert result["rel_diff"] < 0.01, f"Relative error too large: {result['rel_diff']:.3f}%"

    @pytest.mark.slow
    def test_2624x2624_production_scale(self):
        """Test 2624x2624 matrix at production scale (slow test)."""
        led_count = 2624
        num_diagonals = 841
        batch_size = 16

        # Create matrices
        regular_matrix = create_test_matrix(led_count, num_diagonals)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)
        batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=batch_size)

        # Create test input
        np.random.seed(42)
        test_input = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.1
        test_input_gpu = cp.asarray(test_input)

        # Verify correctness (more lenient tolerance for large matrices)
        result = verify_correctness(symmetric_matrix, batch_matrix, test_input_gpu, tolerance=5e-3)

        assert result["success"], (
            f"2624x2624 production scale test failed: max_diff={result['max_diff']:.6f}, "
            f"rel_diff={result['rel_diff']:.3f}%"
        )
        assert result["rel_diff"] < 0.05, f"Relative error too large: {result['rel_diff']:.3f}%"

    def test_batch_size_compatibility(self):
        """Test different batch sizes work correctly."""
        led_count = 64
        num_diagonals = 21

        # Test different batch sizes
        for batch_size in [4, 8, 16]:
            regular_matrix = create_test_matrix(led_count, num_diagonals)
            symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)
            batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(
                regular_matrix, batch_size=batch_size
            )

            # Create test input
            np.random.seed(42)
            test_input = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.1
            test_input_gpu = cp.asarray(test_input)

            # Verify correctness
            result = verify_correctness(symmetric_matrix, batch_matrix, test_input_gpu)

            assert result["success"], (
                f"Batch size {batch_size} test failed: max_diff={result['max_diff']:.6f}, "
                f"rel_diff={result['rel_diff']:.3f}%"
            )

    def test_matrix_info_consistency(self):
        """Test that matrix info is consistent between implementations."""
        led_count = 128
        num_diagonals = 41
        batch_size = 16

        # Create matrices
        regular_matrix = create_test_matrix(led_count, num_diagonals)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)
        batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=batch_size)

        # Get matrix info
        symmetric_info = symmetric_matrix.get_info()
        batch_info = batch_matrix.get_info()

        # Check consistency
        assert symmetric_info["led_count"] == batch_info["led_count"]
        assert symmetric_info["channels"] == batch_info["channels"]
        assert symmetric_info["bandwidth"] == batch_info["bandwidth"]
        assert symmetric_info["nnz"] == batch_info["nnz"]
        assert symmetric_info["original_k"] == batch_info["original_k"]

        # Batch-specific checks
        assert batch_info["batch_size"] == batch_size
        assert batch_info["block_storage"]
        assert batch_info["wmma_kernel_available"]
        assert "block_diag_count" in batch_info
        assert "led_blocks" in batch_info

    def test_kernel_availability(self):
        """Test that WMMA kernels are available and working."""
        # Import the kernel checker
        from utils.kernels.precompiled_mma_kernel import check_precompiled_mma_support

        # Check availability
        assert check_precompiled_mma_support(), "Precompiled MMA kernels not available"

        # Test kernel can be loaded
        from utils.kernels.precompiled_mma_kernel import PrecompiledBatchSymmetricWMMAMatMul

        try:
            kernel = PrecompiledBatchSymmetricWMMAMatMul(use_optimized=False)
            assert kernel is not None, "Failed to create WMMA kernel instance"
        except Exception as e:
            pytest.fail(f"WMMA kernel initialization failed: {e}")


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestBatchWMMAPerformance:
    """Performance-focused tests for batch WMMA implementation."""

    def test_basic_performance_smoke(self):
        """Basic performance smoke test to ensure no major regressions."""
        led_count = 256
        num_diagonals = 81
        batch_size = 16

        # Create matrices
        regular_matrix = create_test_matrix(led_count, num_diagonals)
        symmetric_matrix = SymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix)
        batch_matrix = BatchSymmetricDiagonalATAMatrix.from_diagonal_ata_matrix(regular_matrix, batch_size=batch_size)

        # Create test input
        np.random.seed(42)
        test_input = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.1
        test_input_gpu = cp.asarray(test_input)

        # Time both methods (simple timing)
        import time

        # Sequential timing
        cp.cuda.Stream.null.synchronize()
        start_time = time.time()

        for frame_idx in range(batch_size):
            frame_input = test_input_gpu[frame_idx]
            _ = symmetric_matrix.multiply_3d(frame_input, use_custom_kernel=True, optimized_kernel=True)

        cp.cuda.Stream.null.synchronize()
        sequential_time = time.time() - start_time

        # Batch timing
        cp.cuda.Stream.null.synchronize()
        start_time = time.time()

        _ = batch_matrix.multiply_batch_3d(test_input_gpu, optimized_kernel=False, debug_logging=False)

        cp.cuda.Stream.null.synchronize()
        batch_time = time.time() - start_time

        # Check that batch is faster (at least 20% improvement)
        speedup = sequential_time / batch_time
        assert speedup > 1.2, f"Insufficient speedup: {speedup:.2f}x (expected > 1.2x)"

        print(f"Performance smoke test: {speedup:.2f}x speedup")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
