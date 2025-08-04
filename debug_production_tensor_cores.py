#!/usr/bin/env python3
"""
Debug tensor core usage at production scale (2624 LEDs).
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    import cupy
    CUDA_AVAILABLE = True
    print(f"✓ CUDA available: {cupy.cuda.runtime.runtimeGetVersion()}")
except ImportError:
    print("✗ CUDA not available")
    sys.exit(1)

from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix


def debug_production_tensor_cores():
    """Debug tensor core usage at production scale."""
    print("\n" + "="*60)
    print("DEBUGGING PRODUCTION SCALE TENSOR CORE USAGE")
    print("="*60)

    led_count = 2624
    batch_size = 8

    print(f"Creating production matrix: {led_count} LEDs, batch_size={batch_size}")

    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count,
        crop_size=64,
        batch_size=batch_size,
        output_dtype=cupy.float32
    )

    # Create identity matrix for reproducible results
    dia_offsets = np.array([0], dtype=np.int32)
    dia_data = np.ones((3, 1, led_count), dtype=np.float32)
    dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
    matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

    print(f"Matrix created: {matrix.get_info()}")

    # Create smaller test input for precise numerical analysis
    np.random.seed(42)
    input_batch = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.01  # Small values
    input_batch_gpu = cupy.asarray(input_batch)

    print(f"Input shape: {input_batch_gpu.shape}")
    print(f"Input range: [{cupy.min(input_batch_gpu):.6f}, {cupy.max(input_batch_gpu):.6f}]")

    # Test with full debug logging
    print("\n=== Production Scale 8-Frame Test ===")
    result = matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=False, debug_logging=True)

    print(f"Result shape: {result.shape}")
    print(f"Result range: [{cupy.min(result):.9f}, {cupy.max(result):.9f}]")
    print(f"Result sum: {cupy.sum(cupy.abs(result)):.9f}")

    # CPU reference
    expected = input_batch.copy()  # Identity matrix
    expected_gpu = cupy.asarray(expected)

    # Detailed numerical analysis
    max_diff = cupy.max(cupy.abs(result - expected_gpu))
    mean_diff = cupy.mean(cupy.abs(result - expected_gpu))
    relative_error = max_diff / (cupy.max(cupy.abs(result)) + 1e-10)

    print("\nNumerical Analysis:")
    print(f"Max absolute difference: {max_diff:.12f}")
    print(f"Mean absolute difference: {mean_diff:.12f}")
    print(f"Relative error: {relative_error:.12f}")

    # Check specific patterns that indicate tensor core usage
    if max_diff == 0.0:
        print("🚫 CRITICAL: Perfect precision - NOT using tensor cores!")
        print("   This suggests fallback to regular CUDA or CPU computation")
        return False
    elif relative_error < 1e-10:
        print("🚫 SUSPICIOUS: Near-perfect precision - likely NOT using FP16 tensor cores")
        print("   FP16→FP32 conversion should introduce ~1e-4 to 1e-6 errors")
        return False
    elif 1e-6 <= relative_error <= 1e-3:
        print("✅ GOOD: Error range consistent with FP16 tensor core precision")
        return True
    else:
        print("❌ ERROR: Unexpectedly large errors - potential implementation bug")
        return False


def check_kernel_execution_path():
    """Check which execution path is actually taken."""
    print("\n" + "="*60)
    print("CHECKING KERNEL EXECUTION PATH")
    print("="*60)

    # Test different configurations to isolate the issue
    test_configs = [
        (32, 8, "Small matrix"),
        (160, 8, "Medium matrix"),
        (2624, 8, "Production matrix")
    ]

    for led_count, batch_size, description in test_configs:
        print(f"\n--- {description}: {led_count} LEDs ---")

        matrix = BatchSymmetricDiagonalATAMatrix(
            led_count=led_count,
            crop_size=64,
            batch_size=batch_size,
            output_dtype=cupy.float32
        )

        # Identity matrix
        dia_offsets = np.array([0], dtype=np.int32)
        dia_data = np.ones((3, 1, led_count), dtype=np.float32)
        dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
        matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

        # Small test input
        np.random.seed(42)
        input_batch = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.01
        input_batch_gpu = cupy.asarray(input_batch)

        # Execute with debug logging
        result = matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=False, debug_logging=True)

        # Check precision
        expected = cupy.asarray(input_batch)
        max_diff = cupy.max(cupy.abs(result - expected))
        relative_error = max_diff / (cupy.max(cupy.abs(result)) + 1e-10)

        print(f"Max diff: {max_diff:.9f}, Relative error: {relative_error:.9f}")

        if max_diff == 0.0:
            print("🚫 PERFECT precision - NOT using tensor cores")
        elif relative_error < 1e-8:
            print("🚫 NEAR-PERFECT precision - likely NOT using tensor cores")
        else:
            print("✅ Imperfect precision - likely using tensor cores")


def main():
    """Main debug function."""
    print("Production Scale Tensor Core Debug")
    print("==================================")

    # Check execution paths
    check_kernel_execution_path()

    # Focus on production scale
    tensor_cores_used = debug_production_tensor_cores()

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if not tensor_cores_used:
        print("❌ PROBLEM: Not using tensor cores at production scale")
        print("\nPossible causes:")
        print("1. Kernel falls back to non-tensor operations for large matrices")
        print("2. Implementation uses regular CUDA instead of WMMA instructions")
        print("3. Matrix size exceeds tensor core limitations")
        print("4. Memory layout issues prevent tensor core usage")
        print("\nNeed to investigate the CUDA kernel implementation.")
    else:
        print("✅ SUCCESS: Using tensor cores correctly at production scale")

    return tensor_cores_used


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
