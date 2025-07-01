#!/usr/bin/env python3
"""
Debug the g^T @ g calculation bottleneck.

This should be a trivial operation but is taking 4-20ms.
"""

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def debug_gtg_calculation():
    """Debug g^T @ g calculation performance."""

    print("=== DEBUGGING g^T @ g CALCULATION ===")

    # Create test gradient similar to optimization
    led_count = 500
    channels = 3
    gradient_shape = (channels, led_count)

    print(f"Gradient shape: {gradient_shape}")

    # Test with different gradient magnitudes
    test_cases = [
        ("Small values", np.random.randn(*gradient_shape).astype(np.float32) * 0.001),
        ("Medium values", np.random.randn(*gradient_shape).astype(np.float32) * 1.0),
        ("Large values", np.random.randn(*gradient_shape).astype(np.float32) * 1000.0),
    ]

    for case_name, gradient_cpu in test_cases:
        print(f"\n--- {case_name} ---")
        print(
            f"CPU gradient range: [{gradient_cpu.min():.6f}, {gradient_cpu.max():.6f}]"
        )

        # Transfer to GPU
        transfer_start = time.time()
        gradient_gpu = cp.asarray(gradient_cpu)
        transfer_time = time.time() - transfer_start
        print(f"CPU->GPU transfer: {transfer_time*1000:.3f}ms")

        # Test different g^T @ g calculation methods
        methods = [
            ("cp.sum(g * g)", lambda g: cp.sum(g * g)),
            ("cp.sum(g**2)", lambda g: cp.sum(g**2)),
            ("cp.linalg.norm(g)**2", lambda g: cp.linalg.norm(g) ** 2),
            (
                "cp.dot(g.flatten(), g.flatten())",
                lambda g: cp.dot(g.flatten(), g.flatten()),
            ),
        ]

        for method_name, method_func in methods:
            # Warmup
            for _ in range(3):
                result = method_func(gradient_gpu)
                cp.cuda.Device().synchronize()

            # Timing
            times = []
            for _ in range(10):
                start = time.time()
                result = method_func(gradient_gpu)
                cp.cuda.Device().synchronize()
                times.append(time.time() - start)

            avg_time = np.mean(times) * 1000
            std_time = np.std(times) * 1000
            print(
                f"  {method_name:30}: {avg_time:7.3f} ± {std_time:5.3f} ms, result: {float(result):.3e}"
            )

    # Test memory access patterns
    print(f"\n--- Memory Access Pattern Analysis ---")

    gradient_gpu = cp.asarray(np.random.randn(*gradient_shape).astype(np.float32))

    # Test contiguous vs non-contiguous
    gradient_contiguous = cp.ascontiguousarray(gradient_gpu)
    gradient_transposed = gradient_gpu.T  # Non-contiguous

    print(f"Original contiguous: {gradient_gpu.flags.c_contiguous}")
    print(f"Transposed contiguous: {gradient_transposed.flags.c_contiguous}")

    for name, grad in [
        ("Contiguous", gradient_contiguous),
        ("Non-contiguous", gradient_transposed),
    ]:
        times = []
        for _ in range(10):
            start = time.time()
            result = cp.sum(grad * grad)
            cp.cuda.Device().synchronize()
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        print(f"  {name:15}: {avg_time:7.3f}ms")

    # Test GPU memory allocation overhead
    print(f"\n--- GPU Memory Allocation Test ---")

    times_with_alloc = []
    times_without_alloc = []

    # Pre-allocate
    temp_gpu = cp.zeros_like(gradient_gpu)

    for i in range(10):
        # With allocation
        start = time.time()
        temp = gradient_gpu * gradient_gpu  # Creates new array
        result = cp.sum(temp)
        cp.cuda.Device().synchronize()
        times_with_alloc.append(time.time() - start)

        # Without allocation (reuse)
        start = time.time()
        cp.multiply(gradient_gpu, gradient_gpu, out=temp_gpu)
        result = cp.sum(temp_gpu)
        cp.cuda.Device().synchronize()
        times_without_alloc.append(time.time() - start)

    print(f"  With allocation:    {np.mean(times_with_alloc)*1000:.3f}ms")
    print(f"  Without allocation: {np.mean(times_without_alloc)*1000:.3f}ms")

    # Test reduction algorithm scaling
    print(f"\n--- Scaling Analysis ---")

    sizes = [100, 500, 1000, 2000, 5000]

    for size in sizes:
        test_grad = cp.random.randn(3, size, dtype=cp.float32)

        times = []
        for _ in range(5):
            start = time.time()
            result = cp.sum(test_grad * test_grad)
            cp.cuda.Device().synchronize()
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        print(f"  Size {size:4d}: {avg_time:7.3f}ms")


if __name__ == "__main__":
    debug_gtg_calculation()
