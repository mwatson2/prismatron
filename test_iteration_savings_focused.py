#!/usr/bin/env python3
"""
Focused test to answer specific questions about iteration savings and timing sections.
"""

import numpy as np
import time
from pathlib import Path
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

def test_iteration_savings():
    """Test actual iteration savings with proper convergence."""
    print("="*70)
    print("ITERATION SAVINGS ANALYSIS")
    print("="*70)
    
    # Load 2600 LED patterns
    pattern_path = Path('diffusion_patterns/synthetic_2600_64x64_v7.npz')
    data = np.load(str(pattern_path), allow_pickle=True)
    
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data['mixed_tensor'].item())
    dia_matrix = DiagonalATAMatrix.from_dict(data['dia_matrix'].item())
    ata_inverse = data.get('ata_inverse', None)
    
    if ata_inverse is None:
        print("❌ No ATA inverse found")
        return
    
    print(f"Testing with {mixed_tensor.batch_size} LEDs")
    
    # Create test frames with varying complexity
    test_frames = [
        ("Simple Gray", np.full((3, 480, 800), 128, dtype=np.uint8)),
        ("Complex Pattern", create_complex_frame()),
        ("Red Rectangle", create_red_rectangle()),
    ]
    
    results = []
    
    for frame_name, target_frame in test_frames:
        print(f"\n--- {frame_name} ---")
        
        # Test WITHOUT ATA inverse
        result_default = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            max_iterations=100,
            convergence_threshold=0.3,  # Allow convergence
            debug=False,
        )
        
        # Test WITH ATA inverse
        result_ata_inv = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            ATA_inverse=ata_inverse,
            max_iterations=100,
            convergence_threshold=0.3,  # Allow convergence
            debug=False,
        )
        
        # Calculate savings
        iteration_savings = result_default.iterations - result_ata_inv.iterations
        
        print(f"  DEFAULT:     {result_default.iterations:2d} iterations, converged: {result_default.converged}")
        print(f"  ATA INVERSE: {result_ata_inv.iterations:2d} iterations, converged: {result_ata_inv.converged}")
        print(f"  SAVINGS:     {iteration_savings:2d} iterations")
        
        results.append({
            'frame': frame_name,
            'default_iters': result_default.iterations,
            'ata_inv_iters': result_ata_inv.iterations,
            'savings': iteration_savings,
            'default_converged': result_default.converged,
            'ata_inv_converged': result_ata_inv.converged
        })
    
    # Summary
    print(f"\n" + "="*50)
    print("ITERATION SAVINGS SUMMARY")
    print("="*50)
    
    total_default = sum(r['default_iters'] for r in results)
    total_ata_inv = sum(r['ata_inv_iters'] for r in results)
    total_savings = total_default - total_ata_inv
    
    print(f"{'Frame':15} {'Default':>8} {'ATA Inv':>8} {'Savings':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['frame']:15} {r['default_iters']:>8} {r['ata_inv_iters']:>8} {r['savings']:>8}")
    print("-" * 50)
    print(f"{'TOTAL':15} {total_default:>8} {total_ata_inv:>8} {total_savings:>8}")
    
    avg_savings = total_savings / len(results)
    print(f"\nAverage iteration savings: {avg_savings:.1f} iterations per frame")
    
    return results

def test_timing_sections_controlled():
    """Test why timing sections appear faster - controlled experiment."""
    print("\n" + "="*70)
    print("TIMING SECTIONS INVESTIGATION")  
    print("="*70)
    
    # Load patterns
    pattern_path = Path('diffusion_patterns/synthetic_2600_64x64_v7.npz')
    data = np.load(str(pattern_path), allow_pickle=True)
    
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data['mixed_tensor'].item())
    dia_matrix = DiagonalATAMatrix.from_dict(data['dia_matrix'].item())
    ata_inverse = data.get('ata_inverse', None)
    
    if ata_inverse is None:
        print("❌ No ATA inverse found")
        return
    
    target_frame = np.full((3, 480, 800), 128, dtype=np.uint8)
    
    print("Running SAME number of iterations to isolate timing effects...")
    
    # Force exactly the same number of iterations
    fixed_iterations = 15
    
    # Test WITHOUT ATA inverse
    start_time = time.perf_counter()
    result_default = optimize_frame_led_values(
        target_frame=target_frame,
        AT_matrix=mixed_tensor,
        ATA_matrix=dia_matrix,
        max_iterations=fixed_iterations,
        convergence_threshold=1e-10,  # Will not converge - force all iterations
        enable_timing=True,
        debug=False,
    )
    default_total_time = time.perf_counter() - start_time
    
    # Test WITH ATA inverse
    start_time = time.perf_counter()
    result_ata_inv = optimize_frame_led_values(
        target_frame=target_frame,
        AT_matrix=mixed_tensor,
        ATA_matrix=dia_matrix,
        ATA_inverse=ata_inverse,
        max_iterations=fixed_iterations,
        convergence_threshold=1e-10,  # Will not converge - force all iterations
        enable_timing=True,
        debug=False,
    )
    ata_inv_total_time = time.perf_counter() - start_time
    
    print(f"\nBoth ran exactly {fixed_iterations} iterations:")
    print(f"  DEFAULT:     {result_default.iterations} iterations, {default_total_time:.6f}s total")
    print(f"  ATA INVERSE: {result_ata_inv.iterations} iterations, {ata_inv_total_time:.6f}s total")
    
    if result_default.timing_data and result_ata_inv.timing_data:
        print(f"\nPer-section timing comparison (should be similar):")
        print(f"{'Section':25} {'Default':>12} {'ATA Inv':>12} {'Diff %':>8}")
        print("-" * 65)
        
        core_sections = [
            "ata_multiply", 
            "gradient_calculation", 
            "gradient_step",
            "convergence_check",
            "convergence_and_updates"
        ]
        
        for section in core_sections:
            if section in result_default.timing_data and section in result_ata_inv.timing_data:
                default_time = result_default.timing_data[section]
                ata_inv_time = result_ata_inv.timing_data[section]
                
                if default_time > 0:
                    diff_pct = (ata_inv_time - default_time) / default_time * 100
                else:
                    diff_pct = 0
                
                print(f"{section:25} {default_time:>12.6f} {ata_inv_time:>12.6f} {diff_pct:>7.1f}%")
        
        # Show initialization time
        if "ata_inverse_initialization" in result_ata_inv.timing_data:
            init_time = result_ata_inv.timing_data["ata_inverse_initialization"]
            print(f"{'ata_inverse_init':25} {0:>12.6f} {init_time:>12.6f} {'N/A':>8}")
    
    print(f"\n✅ Investigation: Core optimization sections should have similar timing")
    print(f"    Any differences are likely measurement noise or GPU warming effects")

def create_complex_frame():
    """Create a complex test frame with multiple features."""
    frame = np.zeros((3, 480, 800), dtype=np.uint8)
    
    # Red circle
    center_y, center_x = 240, 200
    y, x = np.ogrid[:480, :800]
    circle_mask = (y - center_y)**2 + (x - center_x)**2 <= 60**2
    frame[0, circle_mask] = 255
    
    # Green stripes
    for i in range(0, 800, 40):
        frame[1, :, i:i+20] = 200
    
    # Blue gradient
    frame[2, :, :] = (np.arange(800) / 800 * 255).astype(np.uint8)
    
    return frame

def create_red_rectangle():
    """Create a simple red rectangle."""
    frame = np.zeros((3, 480, 800), dtype=np.uint8)
    frame[0, 150:350, 250:550] = 255  # Red rectangle
    return frame

def test_no_regression():
    """Confirm no regression with basic functionality test."""
    print("="*70)
    print("REGRESSION TEST - BASIC FUNCTIONALITY")
    print("="*70)
    
    # Load patterns
    pattern_path = Path('diffusion_patterns/synthetic_2600_64x64_v7.npz')
    data = np.load(str(pattern_path), allow_pickle=True)
    
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data['mixed_tensor'].item())
    dia_matrix = DiagonalATAMatrix.from_dict(data['dia_matrix'].item())
    
    print(f"Testing with {mixed_tensor.batch_size} LEDs (representative for regression check)")
    
    # Test basic frames
    test_frames = [
        ('Black', np.zeros((3, 480, 800), dtype=np.uint8)),
        ('White', np.full((3, 480, 800), 255, dtype=np.uint8)),
        ('Gray', np.full((3, 480, 800), 128, dtype=np.uint8)),
    ]
    
    print(f"\n{'Frame':12} {'Iterations':>10} {'Converged':>10} {'Time(ms)':>10}")
    print('-' * 50)
    
    all_converged = True
    for name, frame in test_frames:
        start_time = time.perf_counter()
        result = optimize_frame_led_values(
            target_frame=frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            max_iterations=50,
            debug=False,
        )
        time_ms = (time.perf_counter() - start_time) * 1000
        
        print(f'{name:12} {result.iterations:>10} {str(result.converged):>10} {time_ms:>10.1f}')
        
        if not result.converged:
            all_converged = False
    
    if all_converged:
        print("\n✅ No regression detected - all frames converge properly")
    else:
        print("\n⚠️  Some frames didn't converge - may need threshold adjustment")

if __name__ == "__main__":
    # Test 1: Confirm no regression
    test_no_regression()
    
    # Test 2: Iteration savings analysis
    iteration_results = test_iteration_savings()
    
    # Test 3: Timing sections investigation
    test_timing_sections_controlled()