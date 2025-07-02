#!/usr/bin/env python3
"""
Debug optimization loop timing to identify bottlenecks.

This script instruments each operation within the gradient descent loop
to identify unnecessary copies, conversions, or suboptimal kernels.
"""

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def detailed_optimization_timing():
    """Detailed timing breakdown of optimization loop operations."""

    print("=== DETAILED OPTIMIZATION LOOP TIMING ===")

    # Load int8 patterns
    patterns_path = "diffusion_patterns/baseline_realistic_int8.npz"
    patterns_data = np.load(patterns_path, allow_pickle=True)

    # Load int8 mixed tensor
    mixed_tensor_dict = patterns_data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
    print(f"Mixed tensor: {mixed_tensor.batch_size} LEDs, dtype={mixed_tensor.dtype}")

    # Load DIA matrix
    csc_data_dict = patterns_data["diffusion_matrix"].item()
    diffusion_csc = LEDDiffusionCSCMatrix.from_dict(csc_data_dict)
    csc_matrix = diffusion_csc.to_csc_matrix()
    led_positions = patterns_data["led_positions"]

    dia_matrix = DiagonalATAMatrix(led_count=mixed_tensor.batch_size)

    # Suppress DIA build output
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    dia_matrix.build_from_diffusion_matrix(csc_matrix, led_positions)
    sys.stdout = old_stdout

    print(f"DIA matrix built: {dia_matrix.dia_data_cpu.shape}")

    # Load test image as uint8
    from PIL import Image

    image = Image.open("flower_test.png").convert("RGB").resize((800, 480))
    target_image_uint8 = np.array(image, dtype=np.uint8)
    target_planar_uint8 = target_image_uint8.transpose(2, 0, 1)  # (3, H, W)

    print(f"Target: {target_planar_uint8.shape}, dtype={target_planar_uint8.dtype}")

    # === SETUP PHASE ===
    print("\n--- Setup Phase ---")

    setup_start = time.time()

    # Calculate A^T @ b (one-time setup)
    atb_start = time.time()
    target_gpu = cp.asarray(target_planar_uint8)
    ATb = mixed_tensor.transpose_dot_product_3d(target_gpu)  # Shape: (led_count, 3)
    ATb = cp.asnumpy(ATb)
    if ATb.shape[0] != 3:
        ATb = ATb.T  # Convert to (3, led_count)
    atb_time = time.time() - atb_start

    print(f"A^T @ b calculation: {atb_time * 1000:.2f}ms")
    print(f"A^T @ b shape: {ATb.shape}, range: [{ATb.min():.6f}, {ATb.max():.6f}]")

    # Initialize LED values
    led_count = ATb.shape[1]
    led_values_normalized = np.full((3, led_count), 0.5, dtype=np.float32)

    # Convert to RCM order
    rcm_start = time.time()
    ATb_rcm = dia_matrix.reorder_led_values_to_rcm(ATb)
    led_values_rcm = dia_matrix.reorder_led_values_to_rcm(led_values_normalized)
    rcm_time = time.time() - rcm_start

    print(f"RCM reordering: {rcm_time * 1000:.2f}ms")

    # GPU transfer
    gpu_start = time.time()
    ATb_gpu = cp.asarray(ATb_rcm)
    led_values_gpu = cp.asarray(led_values_rcm)
    gpu_time = time.time() - gpu_start

    print(f"GPU transfer: {gpu_time * 1000:.2f}ms")

    setup_time = time.time() - setup_start
    print(f"Total setup: {setup_time * 1000:.2f}ms")

    # === OPTIMIZATION LOOP ===
    print("\n--- Optimization Loop Detailed Timing ---")

    max_iterations = 5  # Test with fewer iterations for detailed analysis
    step_size_scaling = 0.8
    convergence_threshold = 1e-6

    iteration_times = []
    operation_times = {
        "ATAx": [],
        "gradient_calc": [],
        "g_dot_g": [],
        "g_ATA_g": [],
        "step_calc": [],
        "led_update": [],
        "convergence_check": [],
    }

    for iteration in range(max_iterations):
        iter_start = time.time()

        print(f"\n  Iteration {iteration + 1}:")

        # 1. Compute A^T A @ x
        atax_start = time.time()
        ATA_x = dia_matrix.multiply_3d(led_values_gpu)
        if not isinstance(ATA_x, cp.ndarray):
            ATA_x = cp.asarray(ATA_x)
        cp.cuda.Device().synchronize()  # Ensure GPU computation completes
        atax_time = time.time() - atax_start
        operation_times["ATAx"].append(atax_time)
        print(f"    A^T A @ x: {atax_time * 1000:.3f}ms")

        # 2. Compute gradient
        grad_start = time.time()
        gradient = ATA_x - ATb_gpu
        grad_time = time.time() - grad_start
        operation_times["gradient_calc"].append(grad_time)
        print(f"    Gradient calc: {grad_time * 1000:.3f}ms")

        # 3. Compute g^T @ g
        gtg_start = time.time()
        g_dot_g = cp.sum(gradient * gradient)
        cp.cuda.Device().synchronize()
        gtg_time = time.time() - gtg_start
        operation_times["g_dot_g"].append(gtg_time)
        print(f"    g^T @ g: {gtg_time * 1000:.3f}ms")

        # 4. Compute g^T @ A^T A @ g
        gatag_start = time.time()
        g_dot_ATA_g_per_channel = dia_matrix.g_ata_g_3d(gradient)
        if not isinstance(g_dot_ATA_g_per_channel, cp.ndarray):
            g_dot_ATA_g_per_channel = cp.asarray(g_dot_ATA_g_per_channel)
        g_dot_ATA_g = cp.sum(g_dot_ATA_g_per_channel)
        cp.cuda.Device().synchronize()
        gatag_time = time.time() - gatag_start
        operation_times["g_ATA_g"].append(gatag_time)
        print(f"    g^T @ A^T A @ g: {gatag_time * 1000:.3f}ms")

        # 5. Calculate step size
        step_start = time.time()
        if g_dot_ATA_g > 0:
            step_size = float(step_size_scaling * g_dot_g / g_dot_ATA_g)
        else:
            step_size = 0.01
        step_time = time.time() - step_start
        operation_times["step_calc"].append(step_time)
        print(f"    Step size calc: {step_time * 1000:.3f}ms")

        # 6. Update LED values
        update_start = time.time()
        led_values_new = cp.clip(led_values_gpu - step_size * gradient, 0, 1)
        update_time = time.time() - update_start
        operation_times["led_update"].append(update_time)
        print(f"    LED update: {update_time * 1000:.3f}ms")

        # 7. Check convergence
        conv_start = time.time()
        delta = cp.linalg.norm(led_values_new - led_values_gpu)
        cp.cuda.Device().synchronize()
        conv_time = time.time() - conv_start
        operation_times["convergence_check"].append(conv_time)
        print(f"    Convergence check: {conv_time * 1000:.3f}ms")

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        print(f"    Total iteration: {iter_time * 1000:.2f}ms")
        print(f"    Step size: {step_size:.6f}, Delta: {float(delta):.6f}")

        led_values_gpu = led_values_new

        if delta < convergence_threshold:
            print(f"    Converged after {iteration + 1} iterations")
            break

    # === SUMMARY ===
    print("\n--- Timing Summary ---")

    avg_iter_time = np.mean(iteration_times) * 1000
    print(f"Average iteration time: {avg_iter_time:.2f}ms")

    print("\nOperation breakdown (average ± std):")
    for op, times in operation_times.items():
        if times:
            avg_time = np.mean(times) * 1000
            std_time = np.std(times) * 1000
            percentage = (np.mean(times) / np.mean(iteration_times)) * 100
            print(f"  {op:15}: {avg_time:7.3f} ± {std_time:5.3f} ms ({percentage:5.1f}%)")

    # Check for performance issues
    print("\n--- Performance Analysis ---")
    atax_avg = np.mean(operation_times["ATAx"]) * 1000
    gatag_avg = np.mean(operation_times["g_ATA_g"]) * 1000

    print(f"A^T A @ x average: {atax_avg:.3f}ms (target: <1ms)")
    print(f"g^T @ A^T A @ g average: {gatag_avg:.3f}ms (target: <1ms)")

    if atax_avg > 1.0:
        print(f"⚠️  A^T A @ x is {atax_avg:.1f}x slower than target")
    else:
        print("✅ A^T A @ x meets performance target")

    if gatag_avg > 1.0:
        print(f"⚠️  g^T @ A^T A @ g is {gatag_avg:.1f}x slower than target")
    else:
        print("✅ g^T @ A^T A @ g meets performance target")

    total_critical = atax_avg + gatag_avg
    print(f"Combined critical operations: {total_critical:.3f}ms per iteration")

    if avg_iter_time > 10.0:
        print(f"⚠️  Total iteration time is {avg_iter_time / 10:.1f}x slower than reasonable (>10ms)")
        overhead = avg_iter_time - total_critical
        print(f"Non-critical overhead: {overhead:.3f}ms ({overhead / avg_iter_time * 100:.1f}%)")


if __name__ == "__main__":
    detailed_optimization_timing()
