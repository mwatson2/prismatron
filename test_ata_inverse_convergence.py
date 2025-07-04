#!/usr/bin/env python3
"""
Rigorous ATA inverse convergence analysis.

This script focuses on convergence behavior and iteration savings,
with proper warmup for each initialization method and MSE tracking.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import optimize_frame_led_values
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_2600_patterns():
    """Load 2600 LED patterns."""
    pattern_path = Path(__file__).parent / "diffusion_patterns" / "synthetic_2600_64x64_v7.npz"
    
    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_path}")
    
    data = np.load(str(pattern_path), allow_pickle=True)
    
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())
    dia_matrix = DiagonalATAMatrix.from_dict(data["dia_matrix"].item())
    ata_inverse = data["ata_inverse"] if "ata_inverse" in data else None
    
    return mixed_tensor, dia_matrix, ata_inverse


def create_realistic_test_frame():
    """Create a realistic test frame with varied content."""
    height, width = 480, 800
    frame = np.zeros((3, height, width), dtype=np.uint8)
    
    # Complex pattern with multiple features
    # Red circular gradient
    center_y, center_x = height // 3, width // 4
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    max_dist = np.sqrt(center_y**2 + center_x**2)
    red_pattern = np.clip(255 * (1 - dist_from_center / max_dist), 0, 255)
    frame[0] = red_pattern.astype(np.uint8)
    
    # Green stripes
    stripe_width = 30
    for i in range(0, width, stripe_width * 2):
        frame[1, :, i:i+stripe_width] = 200
    
    # Blue diagonal gradient
    for i in range(height):
        for j in range(width):
            frame[2, i, j] = int(255 * ((i + j) / (height + width)))
    
    # Add some structured noise
    noise_pattern = np.sin(np.arange(height)[:, None] * 0.1) * np.cos(np.arange(width) * 0.1)
    noise_pattern = ((noise_pattern + 1) * 25).astype(np.uint8)
    frame = np.clip(frame.astype(np.int16) + noise_pattern, 0, 255).astype(np.uint8)
    
    return frame


def run_mse_tracking_optimization(target_frame, mixed_tensor, dia_matrix, 
                                ata_inverse=None, max_iterations=50, 
                                convergence_threshold=1e-4):
    """
    Run optimization with detailed MSE tracking per iteration.
    """
    # Custom optimization loop to track MSE at each iteration
    print(f"Running optimization {'WITH' if ata_inverse is not None else 'WITHOUT'} ATA inverse...")
    
    # Step 1: Calculate A^T @ b
    if mixed_tensor.dtype == np.uint8:
        target_gpu = np.asarray(target_frame)
    else:
        target_float32 = target_frame.astype(np.float32) / 255.0
        target_gpu = np.asarray(target_float32)
    
    # Use the frame optimizer's ATb calculation
    from src.utils.frame_optimizer import _calculate_ATb
    ATb = _calculate_ATb(target_frame, mixed_tensor, debug=False)
    
    # Ensure ATb is in (3, led_count) format
    if ATb.shape[0] != 3:
        ATb = ATb.T
    
    led_count = ATb.shape[1]
    
    # Step 2: Initialize LED values
    if ata_inverse is not None:
        print("  Using ATA inverse initialization...")
        start_init = time.perf_counter()
        
        # Compute optimal initial guess: x_c = (A^T A)^-1 * ATb_c
        led_values = np.zeros((3, led_count), dtype=np.float32)
        for c in range(3):
            ATb_channel = ATb[c, :]
            ATA_inv_channel = ata_inverse[c, :, :]
            x_channel = ATA_inv_channel @ ATb_channel
            led_values[c, :] = x_channel
        
        # Clamp to valid range
        led_values = np.clip(led_values, 0.0, 1.0)
        init_time = time.perf_counter() - start_init
        print(f"  Initialization time: {init_time:.6f}s")
    else:
        print("  Using default initialization (0.5)...")
        led_values = np.full((3, led_count), 0.5, dtype=np.float32)
        init_time = 0.0
    
    # Step 3: Optimization loop with MSE tracking
    mse_history = []
    iteration_times = []
    
    print(f"  Starting optimization loop...")
    print(f"  {'Iter':>4} {'MSE':>12} {'Delta':>12} {'Time(ms)':>10}")
    
    for iteration in range(max_iterations):
        iter_start = time.perf_counter()
        
        # Compute A^T A @ x using DIA matrix
        ATA_x = dia_matrix.multiply_3d(led_values)
        
        # Compute gradient: A^T A @ x - A^T @ b
        gradient = ATA_x - ATb
        
        # Compute MSE (gradient magnitude)
        mse = float(np.mean(gradient**2))
        mse_history.append(mse)
        
        # Compute step size
        g_dot_g = np.sum(gradient * gradient)
        g_dot_ATA_g_per_channel = dia_matrix.g_ata_g_3d(gradient)
        g_dot_ATA_g = np.sum(g_dot_ATA_g_per_channel)
        
        if g_dot_ATA_g > 0:
            step_size = float(0.9 * g_dot_g / g_dot_ATA_g)
        else:
            step_size = 0.01
        
        # Gradient descent step
        led_values_new = np.clip(led_values - step_size * gradient, 0, 1)
        
        # Check convergence
        delta = np.linalg.norm(led_values_new - led_values)
        led_values = led_values_new
        
        iter_time = time.perf_counter() - iter_start
        iteration_times.append(iter_time)
        
        print(f"  {iteration+1:>4} {mse:>12.6e} {delta:>12.6e} {iter_time*1000:>10.3f}")
        
        if delta < convergence_threshold:
            print(f"  ✅ Converged after {iteration + 1} iterations")
            break
    else:
        print(f"  ⚠️  Did not converge after {max_iterations} iterations")
    
    return {
        'iterations': iteration + 1,
        'converged': delta < convergence_threshold,
        'mse_history': mse_history,
        'final_mse': mse_history[-1],
        'init_time': init_time,
        'total_optimization_time': sum(iteration_times),
        'avg_iteration_time': np.mean(iteration_times),
        'final_led_values': led_values,
    }


def run_warmup_phase(mixed_tensor, dia_matrix, ata_inverse, target_frame, num_warmup=5):
    """
    Dedicated warmup phase for each initialization method.
    """
    print(f"=== WARMUP PHASE ===")
    
    # Warmup without ATA inverse
    print(f"Warming up DEFAULT initialization ({num_warmup} runs)...")
    for i in range(num_warmup):
        result = optimize_frame_led_values(
            target_frame=target_frame,
            AT_matrix=mixed_tensor,
            ATA_matrix=dia_matrix,
            max_iterations=5,
            debug=False,
        )
        print(f"  Warmup {i+1}: {result.iterations} iters")
    
    if ata_inverse is not None:
        # Warmup with ATA inverse
        print(f"Warming up ATA INVERSE initialization ({num_warmup} runs)...")
        for i in range(num_warmup):
            result = optimize_frame_led_values(
                target_frame=target_frame,
                AT_matrix=mixed_tensor,
                ATA_matrix=dia_matrix,
                ATA_inverse=ata_inverse,
                max_iterations=5,
                debug=False,
            )
            print(f"  Warmup {i+1}: {result.iterations} iters")
    
    print("Warmup completed!\n")


def main():
    print("="*80)
    print("ATA INVERSE CONVERGENCE ANALYSIS - 2600 LEDs")
    print("="*80)
    
    # Load patterns
    try:
        mixed_tensor, dia_matrix, ata_inverse = load_2600_patterns()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    if ata_inverse is None:
        print("❌ ATA inverse not available - run compute_ata_inverse.py first")
        return
    
    led_count = mixed_tensor.batch_size
    print(f"Loaded patterns: {led_count} LEDs")
    print(f"DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k}")
    print(f"ATA inverse: shape={ata_inverse.shape}, memory={ata_inverse.nbytes/(1024*1024):.1f}MB")
    
    # Create test frame
    target_frame = create_realistic_test_frame()
    print(f"Test frame: shape={target_frame.shape}, complexity=realistic")
    
    # Warmup phase with separate warmup for each method
    run_warmup_phase(mixed_tensor, dia_matrix, ata_inverse, target_frame, num_warmup=5)
    
    # Test parameters
    max_iterations = 100
    convergence_threshold = 1e-5
    
    print(f"Test parameters:")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Convergence threshold: {convergence_threshold}")
    
    # Run convergence analysis
    print(f"\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    
    # Test WITHOUT ATA inverse
    print(f"\n[1] DEFAULT INITIALIZATION")
    print("-" * 40)
    result_default = run_mse_tracking_optimization(
        target_frame, mixed_tensor, dia_matrix, 
        ata_inverse=None,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold
    )
    
    print(f"\n[2] ATA INVERSE INITIALIZATION") 
    print("-" * 40)
    result_ata_inv = run_mse_tracking_optimization(
        target_frame, mixed_tensor, dia_matrix,
        ata_inverse=ata_inverse,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold
    )
    
    # Analysis
    print(f"\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"\nConvergence Results:")
    print(f"  DEFAULT:")
    print(f"    Iterations: {result_default['iterations']}")
    print(f"    Converged: {result_default['converged']}")
    print(f"    Final MSE: {result_default['final_mse']:.6e}")
    print(f"    Total optimization time: {result_default['total_optimization_time']:.6f}s")
    print(f"    Avg time per iteration: {result_default['avg_iteration_time']:.6f}s")
    
    print(f"  ATA INVERSE:")
    print(f"    Iterations: {result_ata_inv['iterations']}")
    print(f"    Converged: {result_ata_inv['converged']}")
    print(f"    Final MSE: {result_ata_inv['final_mse']:.6e}")
    print(f"    Initialization time: {result_ata_inv['init_time']:.6f}s")
    print(f"    Total optimization time: {result_ata_inv['total_optimization_time']:.6f}s")
    print(f"    Avg time per iteration: {result_ata_inv['avg_iteration_time']:.6f}s")
    
    # Key metrics
    iteration_savings = result_default['iterations'] - result_ata_inv['iterations']
    time_savings = result_default['total_optimization_time'] - result_ata_inv['total_optimization_time']
    total_time_with_init = result_ata_inv['total_optimization_time'] + result_ata_inv['init_time']
    net_time_savings = result_default['total_optimization_time'] - total_time_with_init
    
    print(f"\nKey Metrics:")
    print(f"  Iteration savings: {iteration_savings} iterations")
    print(f"  Optimization time savings: {time_savings:.6f}s")
    print(f"  Initialization cost: {result_ata_inv['init_time']:.6f}s")
    print(f"  Net time savings: {net_time_savings:.6f}s")
    
    if net_time_savings > 0:
        print(f"  ✅ ATA inverse provides net speedup of {net_time_savings*1000:.2f}ms")
    else:
        print(f"  ❌ ATA inverse adds {abs(net_time_savings)*1000:.2f}ms overhead")
    
    # Per-iteration time comparison
    if abs(result_default['avg_iteration_time'] - result_ata_inv['avg_iteration_time']) > 1e-6:
        time_diff_pct = (result_ata_inv['avg_iteration_time'] / result_default['avg_iteration_time'] - 1) * 100
        print(f"  ⚠️  Per-iteration time difference: {time_diff_pct:+.2f}% (should be ~0%)")
    else:
        print(f"  ✅ Per-iteration times are consistent (good!)")
    
    print(f"\n✅ Analysis completed!")


if __name__ == "__main__":
    main()