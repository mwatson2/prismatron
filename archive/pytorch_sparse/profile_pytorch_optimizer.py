#!/usr/bin/env python3
"""
Profile PyTorch sparse LED optimization to identify performance bottlenecks.

This tool measures the time of individual operations in the PyTorch optimizer
to understand why it's 38x slower than the CuPy version.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.const import FRAME_HEIGHT, FRAME_WIDTH
from src.consumer.led_optimizer_pytorch import PyTorchLEDOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProfiledPyTorchOptimizer(PyTorchLEDOptimizer):
    """PyTorch optimizer with detailed profiling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_data = {}

    def _solve_lsqr_pytorch_profiled(self, max_iterations):
        """Profiled version of LSQR solver."""
        max_iters = max_iterations or self.max_iterations
        w = self._workspace

        # Profile data storage
        times = {
            "forward_spmv": [],
            "residual_calc": [],
            "transpose_spmv": [],
            "step_size_calc": [],
            "gradient_step": [],
            "convergence_check": [],
            "tensor_updates": [],
        }

        logger.info(f"Starting profiled optimization with {max_iters} iterations")

        for iteration in range(max_iters):
            # Profile forward sparse matrix vector multiplication
            torch.cuda.synchronize()  # Ensure accurate timing
            start_time = time.time()

            residual_r = torch.sparse.mm(self._A_r_sparse, w["x_r"].unsqueeze(1)).squeeze(1)
            residual_g = torch.sparse.mm(self._A_g_sparse, w["x_g"].unsqueeze(1)).squeeze(1)
            residual_b = torch.sparse.mm(self._A_b_sparse, w["x_b"].unsqueeze(1)).squeeze(1)

            torch.cuda.synchronize()
            times["forward_spmv"].append(time.time() - start_time)

            # Profile residual calculation
            start_time = time.time()
            w["residual_r"][:] = residual_r - w["target_r"]
            w["residual_g"][:] = residual_g - w["target_g"]
            w["residual_b"][:] = residual_b - w["target_b"]
            torch.cuda.synchronize()
            times["residual_calc"].append(time.time() - start_time)

            # Profile transpose sparse matrix vector multiplication
            start_time = time.time()
            w["gradient_r"][:] = torch.sparse.mm(self._A_r_sparse.t(), w["residual_r"].unsqueeze(1)).squeeze(1)
            w["gradient_g"][:] = torch.sparse.mm(self._A_g_sparse.t(), w["residual_g"].unsqueeze(1)).squeeze(1)
            w["gradient_b"][:] = torch.sparse.mm(self._A_b_sparse.t(), w["residual_b"].unsqueeze(1)).squeeze(1)
            torch.cuda.synchronize()
            times["transpose_spmv"].append(time.time() - start_time)

            # Profile step size computation
            start_time = time.time()
            step_size_r = self._compute_step_size(self._A_r_sparse, w["gradient_r"])
            step_size_g = self._compute_step_size(self._A_g_sparse, w["gradient_g"])
            step_size_b = self._compute_step_size(self._A_b_sparse, w["gradient_b"])
            torch.cuda.synchronize()
            times["step_size_calc"].append(time.time() - start_time)

            # Profile gradient descent step
            start_time = time.time()
            w["x_new_r"] = torch.clamp(w["x_r"] - step_size_r * w["gradient_r"], 0, 1)
            w["x_new_g"] = torch.clamp(w["x_g"] - step_size_g * w["gradient_g"], 0, 1)
            w["x_new_b"] = torch.clamp(w["x_b"] - step_size_b * w["gradient_b"], 0, 1)
            torch.cuda.synchronize()
            times["gradient_step"].append(time.time() - start_time)

            # Profile convergence check
            start_time = time.time()
            delta_r = torch.norm(w["x_new_r"] - w["x_r"])
            delta_g = torch.norm(w["x_new_g"] - w["x_g"])
            delta_b = torch.norm(w["x_new_b"] - w["x_b"])
            total_delta = torch.sqrt(delta_r**2 + delta_g**2 + delta_b**2)
            torch.cuda.synchronize()
            times["convergence_check"].append(time.time() - start_time)

            if total_delta < self.convergence_threshold:
                logger.info(f"Converged after {iteration + 1} iterations")
                break

            # Profile tensor updates (reference swapping)
            start_time = time.time()
            w["x_r"], w["x_new_r"] = w["x_new_r"], w["x_r"]
            w["x_g"], w["x_new_g"] = w["x_new_g"], w["x_g"]
            w["x_b"], w["x_new_b"] = w["x_new_b"], w["x_b"]
            torch.cuda.synchronize()
            times["tensor_updates"].append(time.time() - start_time)

        # Store profiling data
        self.profile_data = times

        # Return result
        led_values = torch.stack([w["x_r"], w["x_g"], w["x_b"]], dim=1)
        return led_values

    def optimize_frame_profiled(self, target_frame: np.ndarray, max_iterations=None):
        """Profiled version of optimize_frame."""
        start_time = time.time()

        # Profile preprocessing
        preprocess_start = time.time()

        if not self._matrix_loaded:
            raise RuntimeError("PyTorch sparse tensors not loaded")

        if target_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise ValueError(f"Target frame shape mismatch")

        # Preprocess target frame and transfer to GPU
        target_rgb_normalized = target_frame.astype(np.float32) / 255.0

        # Copy to workspace tensors
        self._workspace["target_r"][:] = torch.from_numpy(target_rgb_normalized[:, :, 0].ravel()).to(self.device)
        self._workspace["target_g"][:] = torch.from_numpy(target_rgb_normalized[:, :, 1].ravel()).to(self.device)
        self._workspace["target_b"][:] = torch.from_numpy(target_rgb_normalized[:, :, 2].ravel()).to(self.device)

        torch.cuda.synchronize()
        preprocess_time = time.time() - preprocess_start

        # Profile optimization
        optimization_start = time.time()
        led_values_tensor = self._solve_lsqr_pytorch_profiled(max_iterations)
        torch.cuda.synchronize()
        optimization_time = time.time() - optimization_start

        # Profile postprocessing
        postprocess_start = time.time()
        led_values_normalized = led_values_tensor.cpu().numpy()
        led_values_output = (led_values_normalized * 255.0).astype(np.uint8)
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - start_time

        # Store timing breakdown
        self.timing_breakdown = {
            "preprocess_time": preprocess_time,
            "optimization_time": optimization_time,
            "postprocess_time": postprocess_time,
            "total_time": total_time,
        }

        return led_values_output


def analyze_operation_times(times_dict: Dict[str, List[float]], total_optimization_time: float):
    """Analyze and report timing breakdown of operations."""

    logger.info("=== PyTorch Operation Timing Analysis ===")
    logger.info(f"Total optimization time: {total_optimization_time:.3f}s")

    total_accounted = 0.0

    for operation, times in times_dict.items():
        if not times:
            continue

        total_time = sum(times)
        avg_time = np.mean(times)
        percentage = (total_time / total_optimization_time) * 100

        total_accounted += total_time

        logger.info(f"{operation:20s}: {total_time:.3f}s ({percentage:5.1f}%) - avg: {avg_time:.4f}s/iter")

    unaccounted = total_optimization_time - total_accounted
    unaccounted_pct = (unaccounted / total_optimization_time) * 100

    logger.info(f"{'Unaccounted time':20s}: {unaccounted:.3f}s ({unaccounted_pct:5.1f}%)")
    logger.info("")

    # Identify bottlenecks
    bottlenecks = sorted(times_dict.items(), key=lambda x: sum(x[1]), reverse=True)

    logger.info("=== Top Bottlenecks ===")
    for i, (operation, times) in enumerate(bottlenecks[:3]):
        if times:
            total_time = sum(times)
            percentage = (total_time / total_optimization_time) * 100
            logger.info(f"{i + 1}. {operation}: {total_time:.3f}s ({percentage:.1f}%)")


def compare_tensor_formats():
    """Compare different PyTorch tensor formats for sparse operations."""
    logger.info("=== Comparing Tensor Formats ===")

    # Create a small test matrix
    size = 1000
    density = 0.05
    nnz = int(size * size * density)

    # Generate random sparse matrix
    np.random.seed(42)
    rows = np.random.randint(0, size, nnz)
    cols = np.random.randint(0, size, nnz)
    values = np.random.random(nnz).astype(np.float32)

    # Test vector
    x = torch.randn(size, 1, device="cuda")

    # COO format (current)
    indices_coo = torch.from_numpy(np.vstack([rows, cols])).long().cuda()
    values_tensor = torch.from_numpy(values).cuda()
    sparse_coo = torch.sparse_coo_tensor(indices_coo, values_tensor, (size, size)).coalesce()

    # CSR format
    from scipy.sparse import coo_matrix

    scipy_coo = coo_matrix((values, (rows, cols)), shape=(size, size))
    scipy_csr = scipy_coo.tocsr()

    sparse_csr = torch.sparse_csr_tensor(
        torch.from_numpy(scipy_csr.indptr).cuda(),
        torch.from_numpy(scipy_csr.indices).cuda(),
        torch.from_numpy(scipy_csr.data).cuda(),
        (size, size),
    )

    # Time operations
    num_iterations = 100

    # COO timing
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        result_coo = torch.sparse.mm(sparse_coo, x)
    torch.cuda.synchronize()
    coo_time = time.time() - start_time

    # CSR timing
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        result_csr = torch.sparse.mm(sparse_csr, x)
    torch.cuda.synchronize()
    csr_time = time.time() - start_time

    logger.info(f"COO SpMV time: {coo_time:.3f}s ({coo_time / num_iterations * 1000:.2f}ms per op)")
    logger.info(f"CSR SpMV time: {csr_time:.3f}s ({csr_time / num_iterations * 1000:.2f}ms per op)")
    logger.info(f"CSR speedup: {coo_time / csr_time:.2f}x")


def main():
    """Main profiling function."""
    patterns_path = "diffusion_patterns/synthetic_1000"

    if not Path(f"{patterns_path}.npz").exists():
        logger.error(f"Patterns file not found: {patterns_path}.npz")
        return 1

    # Compare tensor formats first
    compare_tensor_formats()

    # Create test image
    np.random.seed(42)
    test_image = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    # Create profiled optimizer
    logger.info("Initializing profiled PyTorch optimizer...")
    optimizer = ProfiledPyTorchOptimizer(diffusion_patterns_path=patterns_path, device="cuda")

    if not optimizer.initialize():
        logger.error("Failed to initialize optimizer")
        return 1

    # Run profiled optimization
    logger.info("Running profiled optimization...")
    led_values = optimizer.optimize_frame_profiled(test_image, max_iterations=5)

    # Analyze results
    analyze_operation_times(optimizer.profile_data, optimizer.timing_breakdown["optimization_time"])

    logger.info("=== Overall Timing Breakdown ===")
    timing = optimizer.timing_breakdown
    total = timing["total_time"]

    logger.info(f"Preprocessing:  {timing['preprocess_time']:.3f}s ({timing['preprocess_time'] / total * 100:.1f}%)")
    logger.info(
        f"Optimization:   {timing['optimization_time']:.3f}s ({timing['optimization_time'] / total * 100:.1f}%)"
    )
    logger.info(f"Postprocessing: {timing['postprocess_time']:.3f}s ({timing['postprocess_time'] / total * 100:.1f}%)")
    logger.info(f"Total:          {total:.3f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
