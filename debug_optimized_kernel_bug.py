#!/usr/bin/env python3
"""
Debug the optimized kernel correctness bug with minimal test case.
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


def debug_small_matrix():
    """Debug with small matrix to isolate the issue."""
    print("\n" + "="*60)
    print("DEBUGGING OPTIMIZED KERNEL WITH SMALL MATRIX")
    print("="*60)

    led_count = 32  # Small matrix for detailed analysis
    batch_size = 8

    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count,
        crop_size=64,
        batch_size=batch_size,
        output_dtype=cupy.float32
    )

    # Identity matrix for predictable results
    dia_offsets = np.array([0], dtype=np.int32)
    dia_data = np.ones((3, 1, led_count), dtype=np.float32)
    dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
    matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

    # Simple test input - constant values for easy verification
    input_batch = np.ones((batch_size, 3, led_count), dtype=np.float32)
    for batch_idx in range(batch_size):
        for channel in range(3):
            input_batch[batch_idx, channel, :] = (batch_idx + 1) + channel * 0.1

    input_batch_gpu = cupy.asarray(input_batch)

    print(f"Matrix: {led_count} LEDs, {matrix.led_blocks}x{matrix.led_blocks} blocks")
    print(f"Input shape: {input_batch_gpu.shape}")
    print(f"Expected result (identity): input sum = {cupy.sum(cupy.abs(input_batch_gpu)):.6f}")

    # Test basic kernel
    print("\n=== Basic Kernel ===")
    result_basic = matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=False, debug_logging=True)
    basic_sum = cupy.sum(cupy.abs(result_basic))
    print(f"Basic result sum: {basic_sum:.6f}")

    # Test optimized kernel
    print("\n=== Optimized Kernel ===")
    result_optimized = matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=True, debug_logging=True)
    optimized_sum = cupy.sum(cupy.abs(result_optimized))
    print(f"Optimized result sum: {optimized_sum:.6f}")

    # Compare element by element
    print("\n=== Element-by-Element Comparison ===")
    max_diff = cupy.max(cupy.abs(result_basic - result_optimized))
    relative_error = max_diff / (cupy.max(cupy.abs(result_basic)) + 1e-10)

    print(f"Max difference: {max_diff:.9f}")
    print(f"Relative error: {relative_error:.9f}")
    print(f"Sum ratio (opt/basic): {optimized_sum/basic_sum:.6f}")

    # Detailed analysis of first batch item
    print("\n=== Detailed Analysis (Batch 0, Channel 0) ===")
    basic_vals = cupy.asnumpy(result_basic[0, 0, :8])  # First 8 LEDs
    opt_vals = cupy.asnumpy(result_optimized[0, 0, :8])
    expected_vals = cupy.asnumpy(input_batch_gpu[0, 0, :8])  # Should equal input for identity

    print("LED | Expected | Basic    | Optimized| Diff     | Ratio")
    print("----|----------|----------|----------|----------|----------")
    for i in range(8):
        diff = opt_vals[i] - basic_vals[i]
        ratio = opt_vals[i] / basic_vals[i] if abs(basic_vals[i]) > 1e-10 else float('inf')
        print(f"{i:3d} | {expected_vals[i]:8.6f} | {basic_vals[i]:8.6f} | {opt_vals[i]:8.6f} | {diff:8.6f} | {ratio:8.3f}")

    # Pattern analysis
    if abs(optimized_sum / basic_sum - 0.5) < 0.1:
        print("\n🔍 PATTERN: Optimized result is ~50% of basic - suggests missing half the computation")
        print("   Possible causes:")
        print("   1. Processing only half the diagonals")
        print("   2. Missing symmetric contributions")
        print("   3. Loading only half the matrix data")
        print("   4. Accumulation bug in diagonal loop")

    return {
        'basic_sum': float(basic_sum),
        'optimized_sum': float(optimized_sum),
        'relative_error': float(relative_error)
    }


def main():
    """Main debug function."""
    print("Optimized Kernel Bug Debug Tool")
    print("===============================")

    results = debug_small_matrix()

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if results['relative_error'] < 0.01:
        print("✅ Optimized kernel is working correctly")
    else:
        print("❌ Optimized kernel has a correctness bug")
        print(f"   Relative error: {results['relative_error']:.3f}")
        print(f"   Sum ratio: {results['optimized_sum']/results['basic_sum']:.3f}")

        if abs(results['optimized_sum'] / results['basic_sum'] - 0.5) < 0.1:
            print("\n💡 DIAGNOSIS: Optimized kernel produces ~50% of expected result")
            print("   This is a systematic error, not random numerical differences")
            print("   Check the CUDA kernel implementation for:")
            print("   1. Diagonal processing loop completeness")
            print("   2. Symmetric contribution handling")
            print("   3. Memory loading patterns")
            print("   4. Accumulation logic")

    return results['relative_error'] < 0.01


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
