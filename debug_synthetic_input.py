#!/usr/bin/env python3
"""
Debug the 8-frame corrected kernel with synthetic input data.

Uses 32x32 identity matrix and synthetic input where element [i,j] = i + j*32.
This makes it easy to track which elements are computed correctly.
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

def create_synthetic_input(batch_size=8, channels=3, leds=32):
    """Create synthetic input where element [batch, channel, led] = led + batch*32 + channel*256."""
    input_batch = np.zeros((batch_size, channels, leds), dtype=np.float32)
    
    for b in range(batch_size):
        for c in range(channels):
            for i in range(leds):
                # Make each element unique and traceable
                input_batch[b, c, i] = i + b*32 + c*256
    
    return input_batch

def test_32x32_identity_synthetic():
    """Test 32x32 identity matrix with synthetic input data."""
    print("\n=== Testing 32x32 Identity Matrix with Synthetic Input ===")
    
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
        print(f"LED count: {batch_matrix.led_count}")
        
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
        print(f"Block content: identity blocks")
        
        # Load into batch matrix
        batch_matrix.block_data_gpu = cupy.array(block_data)
        batch_matrix.block_offsets_upper = cupy.array(block_offsets, dtype=cupy.int32)
        batch_matrix.block_diag_count = block_diag_count
        
        # Create synthetic input
        input_batch = create_synthetic_input(batch_size, channels, led_count)
        input_batch_gpu = cupy.array(input_batch)
        
        print(f"\nInput batch shape: {input_batch.shape}")
        print(f"Input values structure: element [batch, channel, led] = led + batch*32 + channel*256")
        print(f"Input sample values:")
        print(f"  [0,0,0] = {input_batch[0,0,0]} (expected: 0)")
        print(f"  [0,0,1] = {input_batch[0,0,1]} (expected: 1)")  
        print(f"  [0,0,31] = {input_batch[0,0,31]} (expected: 31)")
        print(f"  [1,0,0] = {input_batch[1,0,0]} (expected: 32)")
        print(f"  [0,1,0] = {input_batch[0,1,0]} (expected: 256)")
        
        # Expected result: identity @ input = input (unchanged for identity matrix)
        expected_result = input_batch.copy()
        
        # Compute with corrected WMMA kernel
        print("\nComputing with corrected 8-frame WMMA kernel...")
        result_wmma = batch_matrix.multiply_batch8_3d(input_batch_gpu, optimized_kernel=False, debug_logging=True)
        result_wmma_cpu = cupy.asnumpy(result_wmma)
        
        print(f"\nWMMA result shape: {result_wmma_cpu.shape}")
        print(f"WMMA result sample values:")
        print(f"  [0,0,0] = {result_wmma_cpu[0,0,0]} (expected: 0)")
        print(f"  [0,0,1] = {result_wmma_cpu[0,0,1]} (expected: 1)")
        print(f"  [0,0,31] = {result_wmma_cpu[0,0,31]} (expected: 31)")
        print(f"  [1,0,0] = {result_wmma_cpu[1,0,0]} (expected: 32)")
        print(f"  [0,1,0] = {result_wmma_cpu[0,1,0]} (expected: 256)")
        
        # Analyze errors systematically
        print(f"\n=== Detailed Error Analysis ===")
        
        abs_diff = np.abs(result_wmma_cpu - expected_result)
        max_abs_error = np.max(abs_diff)
        mean_abs_error = np.mean(abs_diff)
        
        print(f"Maximum absolute error: {max_abs_error:.6f}")
        print(f"Mean absolute error: {mean_abs_error:.6f}")
        
        # Check each element systematically
        print(f"\nFirst 8 elements of each batch/channel:")
        for b in range(min(2, batch_size)):  # First 2 batches
            for c in range(min(2, channels)):  # First 2 channels
                print(f"\nBatch {b}, Channel {c}:")
                for i in range(min(8, led_count)):
                    wmma_val = result_wmma_cpu[b, c, i]
                    expected_val = expected_result[b, c, i]
                    error = abs(wmma_val - expected_val)
                    print(f"  LED {i:2d}: WMMA={wmma_val:7.1f}, Expected={expected_val:7.1f}, Error={error:7.3f}")
        
        # Check for patterns in errors
        print(f"\n=== Error Pattern Analysis ===")
        
        # Check if certain positions always have errors
        error_by_led = np.mean(abs_diff, axis=(0, 1))  # Average error across batches and channels
        print(f"Average error by LED position (first 16):")
        for i in range(min(16, led_count)):
            print(f"  LED {i:2d}: {error_by_led[i]:7.3f}")
        
        # Check if certain batches/channels have more errors
        error_by_batch = np.mean(abs_diff, axis=(1, 2))  # Average error across channels and LEDs
        error_by_channel = np.mean(abs_diff, axis=(0, 2))  # Average error across batches and LEDs
        
        print(f"\nAverage error by batch:")
        for b in range(batch_size):
            print(f"  Batch {b}: {error_by_batch[b]:7.3f}")
            
        print(f"\nAverage error by channel:")
        for c in range(channels):
            print(f"  Channel {c}: {error_by_channel[c]:7.3f}")
        
        # Success/failure assessment
        TOLERANCE = 1e-3  # Generous tolerance for debugging
        
        if max_abs_error < TOLERANCE:
            print(f"\n✅ SYNTHETIC INPUT TEST PASSED!")
            print(f"   Max error {max_abs_error:.6f} < tolerance {TOLERANCE}")
            return True
        else:
            print(f"\n❌ SYNTHETIC INPUT TEST FAILED!")
            print(f"   Max error {max_abs_error:.6f} >= tolerance {TOLERANCE}")
            
            # Identify the worst error for focused debugging
            max_error_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            b_max, c_max, i_max = max_error_idx
            print(f"\nWorst error at [batch={b_max}, channel={c_max}, led={i_max}]:")
            print(f"  WMMA: {result_wmma_cpu[b_max, c_max, i_max]}")
            print(f"  Expected: {expected_result[b_max, c_max, i_max]}")
            print(f"  Error: {abs_diff[b_max, c_max, i_max]}")
            
            return False
        
    except Exception as e:
        print(f"❌ Error during synthetic input test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if not GPU_AVAILABLE:
        print("GPU required for testing")
        sys.exit(1)
    
    success = test_32x32_identity_synthetic()
    
    if success:
        print("\n🎉 Synthetic input test PASSED - kernel logic is working correctly!")
    else:
        print("\n💥 Synthetic input test FAILED - kernel needs systematic debugging!")
        print("\nNext debugging steps:")
        print("1. Check WMMA fragment loading and storage")
        print("2. Verify block pair indexing logic") 
        print("3. Validate input tensor coordinate mapping")
        print("4. Check accumulation across block columns")
    
    sys.exit(0 if success else 1)