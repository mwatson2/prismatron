#!/usr/bin/env python3
"""
Debug the 8-frame vertical pair kernel with a 32x32 identity matrix.
This is the simplest case: identity matrix should return input unchanged.
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

def test_32x32_identity():
    """Test 32x32 identity matrix case."""
    print("\n=== Testing 32x32 Identity Matrix ===")
    
    try:
        from utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix
        
        # Create 32x32 identity case
        led_count = 32
        channels = 3  # RGB channels (required by batch matrix)
        batch_size = 8
        
        batch_matrix = BatchSymmetricDiagonalATAMatrix(
            led_count=led_count,
            batch_size=batch_size
        )
        
        print(f"Matrix size: {led_count}x{led_count}")
        print(f"Block structure: {batch_matrix.led_blocks}x{batch_matrix.led_blocks}")
        print(f"Padded LEDs: {batch_matrix.padded_led_count}")
        
        # Create identity matrix in block diagonal format
        # For 32x32 identity, we have 2x2 blocks, only main diagonal (offset=0) exists
        block_diag_count = 1
        block_offsets = np.array([0], dtype=np.int32)  # Only main diagonal
        
        # Create block data: identity blocks for all channels
        block_data = np.zeros((channels, block_diag_count, 16, 16), dtype=np.float32)
        for c in range(channels):
            block_data[c, 0] = np.eye(16, dtype=np.float32)  # Identity block for each channel
        
        print(f"Block offsets: {block_offsets}")
        print(f"Block data shape: {block_data.shape}")
        print(f"Block content (first block): identity matrix")
        
        # Load into batch matrix
        batch_matrix.block_data_gpu = cupy.array(block_data)
        batch_matrix.block_offsets_upper = cupy.array(block_offsets, dtype=cupy.int32)
        batch_matrix.block_diag_count = block_diag_count
        
        # Create test input: random batch for all channels (8, 3, 32)
        np.random.seed(42)
        input_batch = np.random.normal(0, 1, (8, 3, 32)).astype(np.float32)
        input_batch_gpu = cupy.array(input_batch)
        
        print(f"Input batch shape: {input_batch.shape}")
        print(f"Input sample (first few elements, channel 0): {input_batch[0, 0, :5]}")
        
        # Expected result: identity @ input = input (unchanged)
        expected_result = input_batch.copy()  # Identity matrix preserves input
        
        # Compute with WMMA kernel
        print("\nComputing with 8-frame vertical pair WMMA...")
        result_wmma = batch_matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=False)
        result_wmma_cpu = cupy.asnumpy(result_wmma)
        
        print(f"WMMA result shape: {result_wmma_cpu.shape}")
        print(f"WMMA result sample (first few elements, channel 0): {result_wmma_cpu[0, 0, :5]}")
        print(f"Expected result sample (first few elements, channel 0): {expected_result[0, 0, :5]}")
        
        # Compare results
        abs_diff = np.abs(result_wmma_cpu - expected_result)
        max_abs_error = np.max(abs_diff)
        mean_abs_error = np.mean(abs_diff)
        
        print(f"\n=== Results ===")
        print(f"Maximum absolute error: {max_abs_error:.6f}")
        print(f"Mean absolute error: {mean_abs_error:.6f}")
        
        # For identity matrix, result should be nearly identical to input
        TOLERANCE = 1e-5  # Very strict since it's identity
        
        if max_abs_error < TOLERANCE:
            print(f"✅ IDENTITY TEST PASSED!")
            print(f"   Kernel correctly implements identity matrix multiplication")
            return True
        else:
            print(f"❌ IDENTITY TEST FAILED!")
            print(f"   Error {max_abs_error:.6f} > tolerance {TOLERANCE}")
            
            # Show detailed comparison for debugging
            print(f"\nDetailed comparison (first 8 elements, channel 0):")
            for i in range(min(8, len(result_wmma_cpu[0, 0]))):
                wmma_val = result_wmma_cpu[0, 0, i]
                expected_val = expected_result[0, 0, i]
                diff = abs(wmma_val - expected_val)
                print(f"  [{i}] WMMA: {wmma_val:.6f}, Expected: {expected_val:.6f}, Error: {diff:.6f}")
            
            return False
        
    except Exception as e:
        print(f"❌ Error during identity test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("GPU required for testing")
        sys.exit(1)
    
    success = test_32x32_identity()
    
    if success:
        print("\n🎉 Identity matrix test PASSED - kernel logic is correct!")
    else:
        print("\n💥 Identity matrix test FAILED - kernel needs debugging!")
        print("\nKernel issues to investigate:")
        print("1. Block diagonal iteration logic")
        print("2. Block pair indexing")
        print("3. Input tensor slicing")
        print("4. WMMA operation parameters")
    
    sys.exit(0 if success else 1)