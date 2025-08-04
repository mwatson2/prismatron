#!/usr/bin/env python3
"""
Debug script to verify tensor core usage in 8-frame kernel.

This script will add detailed logging to determine if we're actually
using the WMMA tensor core kernel or falling back to something else.
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

from utils.batch_symmetric_diagonal_ata_matrix import BATCH8_WMMA_KERNEL_AVAILABLE, BatchSymmetricDiagonalATAMatrix


def debug_kernel_usage():
    """Debug what kernel is actually being used."""
    print("\n" + "="*60)
    print("DEBUGGING TENSOR CORE KERNEL USAGE")
    print("="*60)

    # Check kernel availability
    print(f"BATCH8_WMMA_KERNEL_AVAILABLE: {BATCH8_WMMA_KERNEL_AVAILABLE}")

    if not BATCH8_WMMA_KERNEL_AVAILABLE:
        print("❌ 8-frame kernels not available - will use fallback")
        return False

    # Create small test case
    led_count = 32
    batch_size = 8

    print(f"Creating test matrix: {led_count} LEDs, batch_size={batch_size}")

    matrix = BatchSymmetricDiagonalATAMatrix(
        led_count=led_count,
        crop_size=64,
        batch_size=batch_size,
        output_dtype=cupy.float32
    )

    # Create identity matrix
    dia_offsets = np.array([0], dtype=np.int32)
    dia_data = np.ones((3, 1, led_count), dtype=np.float32)
    dia_data_gpu = cupy.asarray(dia_data, dtype=cupy.float32)
    matrix._convert_diagonal_to_blocks(dia_data_gpu, dia_offsets)

    # Create test input
    input_batch = np.random.randn(batch_size, 3, led_count).astype(np.float32) * 0.1
    input_batch_gpu = cupy.asarray(input_batch)

    print(f"Test input shape: {input_batch_gpu.shape}")
    print(f"Matrix info: {matrix.get_info()}")

    # Test with maximum debugging
    print("\n=== Testing 8-frame kernel with full debug logging ===")
    result = matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=False, debug_logging=True)

    print(f"Result shape: {result.shape}")
    print(f"Result sum: {cupy.sum(cupy.abs(result)):.6f}")
    print(f"Result sample (first element): {result[0, 0, 0]:.6f}")

    # CPU reference for comparison
    expected = input_batch.copy()  # Identity matrix
    max_diff = cupy.max(cupy.abs(result - cupy.asarray(expected)))
    relative_error = max_diff / (cupy.max(cupy.abs(result)) + 1e-10)

    print(f"Max difference vs CPU: {max_diff:.9f}")
    print(f"Relative error: {relative_error:.9f}")

    # Check if we're really using tensor cores
    if relative_error < 1e-8:
        print("🚫 SUSPICIOUS: Perfect precision suggests NOT using FP16 tensor cores")
        print("   Tensor cores should introduce small FP16→FP32 conversion errors")
    elif relative_error < 1e-3:
        print("✅ GOOD: Small errors consistent with FP16 tensor core precision")
    else:
        print("❌ ERROR: Large errors suggest implementation bug")

    return True


def check_kernel_implementation():
    """Check the actual kernel implementation."""
    print("\n" + "="*60)
    print("CHECKING KERNEL IMPLEMENTATION")
    print("="*60)

    # Import the kernel directly
    try:
        from utils.kernels.precompiled_mma_kernel import (
            PRECOMPILED_8FRAME_MMA_SUPPORTED,
            PrecompiledBatch8SymmetricWMMAMatMul,
        )

        print(f"PRECOMPILED_8FRAME_MMA_SUPPORTED: {PRECOMPILED_8FRAME_MMA_SUPPORTED}")

        if not PRECOMPILED_8FRAME_MMA_SUPPORTED:
            print("❌ 8-frame kernels not compiled")
            return False

        # Create kernel instance
        kernel = PrecompiledBatch8SymmetricWMMAMatMul(use_optimized=False)
        print(f"Kernel created: {kernel}")
        print(f"Kernel type: {type(kernel)}")

        # Check if kernel has the expected methods
        if hasattr(kernel, '_module'):
            print("✅ Kernel has _module attribute")
            if hasattr(kernel, '_kernel_basic_func'):
                print("✅ Kernel has _kernel_basic_func")
            else:
                print("❌ Kernel missing _kernel_basic_func")
        else:
            print("❌ Kernel missing _module attribute")

        return True

    except ImportError as e:
        print(f"❌ Failed to import 8-frame kernel: {e}")
        return False


def main():
    """Main debug function."""
    print("Tensor Core Usage Debug Tool")
    print("============================")

    # Check kernel implementation
    kernel_ok = check_kernel_implementation()

    if not kernel_ok:
        print("\n❌ Kernel implementation issues found")
        return False

    # Debug actual kernel usage
    usage_ok = debug_kernel_usage()

    if not usage_ok:
        print("\n❌ Kernel usage issues found")
        return False

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("Based on the analysis above:")
    print("1. If relative error is < 1e-8, we're likely NOT using tensor cores")
    print("2. If relative error is ~1e-4 to 1e-6, we're probably using tensor cores correctly")
    print("3. Check the debug output to see which kernel path is actually taken")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
