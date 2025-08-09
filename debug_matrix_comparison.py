#!/usr/bin/env python3
"""
Debug script to compare BatchSymmetricDiagonalATAMatrix vs SymmetricDiagonalATAMatrix
on identical input data to isolate the source of differences.
"""

import logging
import numpy as np
import cupy as cp

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from src.utils.symmetric_diagonal_ata_matrix import SymmetricDiagonalATAMatrix
from src.utils.batch_symmetric_diagonal_ata_matrix import BatchSymmetricDiagonalATAMatrix

logging.basicConfig(level=logging.WARNING)  # Reduce verbosity

def compare_matrix_operations():
    """Compare matrix operations between regular and batch ATA matrices."""
    
    # Use small aligned problem
    led_count = 16  # Multiple of 16 for tensor cores
    channels = 3
    height, width = 128, 96
    block_size = 64
    
    print("=== Matrix Operation Comparison ===")
    print(f"LEDs: {led_count}, Image: {height}x{width}")
    
    # Create identical sparse tensor
    tensor = SingleBlockMixedSparseTensor(
        batch_size=led_count,
        channels=channels,
        height=height,
        width=width,
        block_size=block_size,
        dtype=cp.uint8
    )
    
    # Set identical blocks
    np.random.seed(42)  # Fixed seed
    rows = int(np.sqrt(led_count))
    cols = int(np.ceil(led_count / rows))
    
    for led_idx in range(led_count):
        row = led_idx // cols
        col = led_idx % cols
        
        for channel in range(channels):
            top = row * (height // rows)
            left = col * (width // cols)
            left = (left // 4) * 4  # Align to 4
            
            if top + block_size <= height and left + block_size <= width:
                values = cp.random.randint(0, 256, (block_size, block_size), dtype=cp.uint8)
                tensor.set_block(led_idx, channel, top, left, values)
    
    # Compute ATA matrix
    dense_ata = tensor.compute_ata_dense()  # (led_count, led_count, 3)
    
    # Create both matrix types from identical dense data
    regular_ata = SymmetricDiagonalATAMatrix.from_dense(
        dense_ata.transpose(2, 0, 1),
        led_count=led_count,
        significance_threshold=0.01,
        crop_size=block_size
    )
    
    batch_ata = BatchSymmetricDiagonalATAMatrix.from_symmetric_diagonal_matrix(
        regular_ata,
        batch_size=8
    )
    
    print(f"Regular ATA: {regular_ata.get_info().get('num_diagonals', 'unknown')} diagonals")
    print(f"Batch ATA: {batch_ata.block_diag_count} block diagonals")
    
    # Test identical input vectors
    np.random.seed(123)  # Different seed for test vectors
    test_input = np.random.randn(3, led_count).astype(np.float32)
    test_input_gpu = cp.asarray(test_input)
    
    print(f"\nTest input range: [{test_input.min():.6f}, {test_input.max():.6f}]")
    
    # Test 1: Single multiply operation
    print("\n--- Single Multiply Test ---")
    regular_result = regular_ata.multiply_3d(test_input_gpu)
    
    # For batch, we need to create batch input
    batch_input = cp.tile(test_input_gpu[None, :, :], (8, 1, 1))  # (8, 3, led_count)
    batch_result = batch_ata.multiply_batch8_3d(batch_input)
    
    # Compare first frame of batch result
    regular_cpu = cp.asnumpy(regular_result)
    batch_cpu = cp.asnumpy(batch_result[0])  # First frame
    
    max_diff = np.max(np.abs(regular_cpu - batch_cpu))
    rms_diff = np.sqrt(np.mean((regular_cpu - batch_cpu) ** 2))
    rms_regular = np.sqrt(np.mean(regular_cpu ** 2))
    
    print(f"Regular result range: [{regular_cpu.min():.6f}, {regular_cpu.max():.6f}]")
    print(f"Batch result range: [{batch_cpu.min():.6f}, {batch_cpu.max():.6f}]")
    print(f"Max difference: {max_diff:.8f}")
    print(f"RMS difference: {rms_diff:.8f}")
    if rms_regular > 1e-10:
        print(f"Relative error: {rms_diff/rms_regular*100:.6f}%")
    
    # Test 2: g^T A^T A g operation
    print("\n--- g^T A^T A g Test ---")
    regular_gag = regular_ata.g_ata_g_3d(test_input_gpu)
    batch_gag = batch_ata.g_ata_g_batch_3d(batch_input)
    
    regular_gag_cpu = cp.asnumpy(regular_gag)
    batch_gag_cpu = cp.asnumpy(batch_gag[0])  # First frame
    
    max_diff_gag = np.max(np.abs(regular_gag_cpu - batch_gag_cpu))
    rms_diff_gag = np.sqrt(np.mean((regular_gag_cpu - batch_gag_cpu) ** 2))
    rms_regular_gag = np.sqrt(np.mean(regular_gag_cpu ** 2))
    
    print(f"Regular g^TAG result: {regular_gag_cpu}")
    print(f"Batch g^TAG result: {batch_gag_cpu}")
    print(f"Max difference: {max_diff_gag:.8f}")
    print(f"RMS difference: {rms_diff_gag:.8f}")
    if rms_regular_gag > 1e-10:
        print(f"Relative error: {rms_diff_gag/rms_regular_gag*100:.6f}%")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    multiply_ok = max_diff < 1e-4
    gag_ok = max_diff_gag < 1e-4
    
    print(f"Multiply operation: {'✓ PASS' if multiply_ok else '✗ FAIL'} ({max_diff:.2e} max diff)")
    print(f"g^TAG operation: {'✓ PASS' if gag_ok else '✗ FAIL'} ({max_diff_gag:.2e} max diff)")
    
    if multiply_ok and gag_ok:
        print("Matrix operations are equivalent - issue is elsewhere")
    else:
        print("Matrix operations differ - found the bug!")
    
    return multiply_ok and gag_ok

if __name__ == "__main__":
    compare_matrix_operations()