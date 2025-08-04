#!/usr/bin/env python3
"""
Test the 8-frame vertical pair WMMA integration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np

try:
    import cupy
    print(f"CUDA available: {cupy.cuda.runtime.runtimeGetVersion()}")
    GPU_AVAILABLE = True
except ImportError:
    print("CUDA not available")
    GPU_AVAILABLE = False
    sys.exit(1)

def test_8frame_batch_matrix():
    """Test creating and loading 8-frame batch matrix."""
    print("\n=== Testing 8-Frame Batch Matrix Creation ===")

    try:
        from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix

        # Create a small test matrix with 8-frame batch
        led_count = 64  # Small for testing
        batch_matrix = BatchSymmetricDiagonalATAMatrix(
            led_count=led_count,
            batch_size=8  # This should trigger 8-frame kernel loading
        )

        print("✓ Successfully created 8-frame batch matrix:")
        print(f"  LED count: {batch_matrix.led_count}")
        print(f"  Batch size: {batch_matrix.batch_size}")
        print(f"  LED blocks: {batch_matrix.led_blocks}")
        print(f"  Padded LED count: {batch_matrix.padded_led_count}")

        # Check if 8-frame kernels were loaded
        has_8frame_basic = batch_matrix.wmma_kernel_8frame_basic is not None
        has_8frame_optimized = batch_matrix.wmma_kernel_8frame_optimized is not None

        print(f"  8-frame basic kernel loaded: {has_8frame_basic}")
        print(f"  8-frame optimized kernel loaded: {has_8frame_optimized}")

        if has_8frame_basic:
            print("✓ 8-frame vertical pair WMMA integration successful!")
            return True
        else:
            print("✗ 8-frame kernels not loaded properly")
            return False

    except Exception as e:
        print(f"✗ Error creating 8-frame batch matrix: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kernel_loading():
    """Test loading the 8-frame kernel directly."""
    print("\n=== Testing Direct 8-Frame Kernel Loading ===")

    try:
        from utils.kernels.precompiled_mma_kernel import PrecompiledBatch8SymmetricWMMAMatMul

        # Try to load the kernel
        kernel = PrecompiledBatch8SymmetricWMMAMatMul(use_optimized=False)
        print("✓ Successfully loaded 8-frame basic kernel")

        kernel_opt = PrecompiledBatch8SymmetricWMMAMatMul(use_optimized=True)
        print("✓ Successfully loaded 8-frame optimized kernel")

        return True

    except Exception as e:
        print(f"✗ Error loading 8-frame kernel: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("GPU required for testing")
        sys.exit(1)

    success1 = test_kernel_loading()
    success2 = test_8frame_batch_matrix()

    if success1 and success2:
        print("\n🎉 All 8-frame integration tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some 8-frame integration tests failed")
        sys.exit(1)
