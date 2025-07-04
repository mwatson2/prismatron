#!/usr/bin/env python3
"""
Simple performance test for frame optimizer functions.
"""

import time
import numpy as np
import cupy as cp
from pathlib import Path

from src.utils.frame_optimizer import optimize_frame_led_values, load_ata_inverse_from_pattern
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

def test_frame_optimization_performance():
    """Test frame optimization performance with timing breakdown."""
    
    print("=== Frame Optimizer Performance Test ===\n")
    
    # Use the fresh 1000 LED pattern that has ATA inverse
    pattern_file = "/mnt/dev/prismatron/diffusion_patterns/synthetic_1000_fresh.npz"
    
    if not Path(pattern_file).exists():
        print(f"Pattern file {pattern_file} not found")
        return
    
    print(f"Testing with pattern: {Path(pattern_file).name}")
    
    try:
        # Load pattern data
        print("Loading pattern data...")
        data = np.load(pattern_file, allow_pickle=True)
        print(f"Pattern keys: {list(data.files)}")
        
        # Load DIA matrix (ATA matrix)
        dia_dict = data['dia_matrix'].item()
        ata_matrix = DiagonalATAMatrix.from_dict(dia_dict)
        print(f"ATA matrix: {ata_matrix.led_count} LEDs, bandwidth {ata_matrix.bandwidth}")
        
        # Load mixed tensor (AT matrix)
        mixed_tensor_dict = data['mixed_tensor'].item()
        at_matrix = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
        print(f"AT matrix: {at_matrix.batch_size} LEDs, {at_matrix.channels} channels")
        
        # Load ATA inverse
        ata_inverse = load_ata_inverse_from_pattern(pattern_file)
        if ata_inverse is None:
            print("No ATA inverse found")
            return
        print(f"ATA inverse shape: {ata_inverse.shape}")
        
        # Create test frame
        height, width = 480, 800
        test_frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        print(f"Test frame shape: {test_frame.shape}")
        
        # Warm up GPU
        print("\nWarming up...")
        for _ in range(3):
            _ = optimize_frame_led_values(
                target_frame=test_frame,
                AT_matrix=at_matrix,
                ATA_matrix=ata_matrix,
                ATA_inverse=ata_inverse,
                max_iterations=50,
                convergence_threshold=0.3,
                enable_timing=False
            )
        
        # Performance test with timing
        print("\nRunning performance test with timing breakdown...")
        
        num_trials = 5
        results = []
        
        for trial in range(num_trials):
            print(f"\nTrial {trial + 1}/{num_trials}:")
            
            start_time = time.time()
            result = optimize_frame_led_values(
                target_frame=test_frame,
                AT_matrix=at_matrix,
                ATA_matrix=ata_matrix,
                ATA_inverse=ata_inverse,
                max_iterations=100,
                convergence_threshold=0.3,
                enable_timing=True
            )
            total_time = time.time() - start_time
            
            print(f"  Total time: {total_time*1000:.1f}ms")
            print(f"  Iterations: {result.iterations}")
            print(f"  Converged: {result.converged}")
            print(f"  LED values shape: {result.led_values.shape}")
            
            if result.timing_data:
                print("  Timing breakdown:")
                for operation, timing in result.timing_data.items():
                    print(f"    {operation}: {timing:.2f}ms")
            
            if result.error_metrics:
                print("  Error metrics:")
                for metric, value in result.error_metrics.items():
                    print(f"    {metric}: {value:.6f}")
            
            results.append({
                'total_time': total_time,
                'iterations': result.iterations,
                'converged': result.converged,
                'timing_data': result.timing_data,
                'error_metrics': result.error_metrics
            })
        
        # Calculate statistics
        total_times = [r['total_time'] for r in results]
        iterations = [r['iterations'] for r in results]
        
        print(f"\n=== Performance Summary ({num_trials} trials) ===")
        print(f"Average total time: {np.mean(total_times)*1000:.1f}ms")
        print(f"Min total time: {np.min(total_times)*1000:.1f}ms")
        print(f"Max total time: {np.max(total_times)*1000:.1f}ms")
        print(f"Average iterations: {np.mean(iterations):.1f}")
        print(f"Theoretical FPS: {1.0/np.mean(total_times):.1f}")
        
        # Average timing breakdown
        if results[0]['timing_data']:
            print(f"\nAverage timing breakdown:")
            timing_keys = results[0]['timing_data'].keys()
            for key in timing_keys:
                avg_timing = np.mean([r['timing_data'][key] for r in results])
                print(f"  {key}: {avg_timing:.2f}ms")
        
        print(f"\nKey findings:")
        print(f"- LED count: {ata_matrix.led_count}")
        print(f"- Matrix bandwidth: {ata_matrix.bandwidth}")
        print(f"- Sparsity: {ata_matrix.sparsity:.1%}")
        print(f"- Memory efficient DIA format in use")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_frame_optimization_performance()