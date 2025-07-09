#!/usr/bin/env python3
"""
Tests for standalone frame optimization function.

This module tests the extracted frame optimization function with both
mixed tensor and DIA matrix formats for LED optimization.
"""

import sys
import time
from pathlib import Path
from typing import Tuple

import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sp

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
from src.utils.frame_optimizer import (
    FrameOptimizationResult,
    optimize_frame_led_values,
    optimize_frame_with_tensors,
)
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


class TestFrameOptimizer:
    """Test standalone frame optimization function."""

    def load_real_diffusion_patterns(
        self, precision: str = "fp16"
    ) -> Tuple[SingleBlockMixedSparseTensor, DiagonalATAMatrix]:
        """
        Load real diffusion patterns from stored patterns with DIA matrix.

        Args:
            precision: Either "fp16" or "fp32" to specify which patterns to load

        Returns:
            Tuple of (mixed_tensor, dia_matrix)
        """
        from pathlib import Path

        # Load the patterns based on precision
        if precision == "fp16":
            pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_fp16.npz"
        elif precision == "fp32":
            pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_fp32.npz"
        else:
            raise ValueError(f"Invalid precision: {precision}. Must be 'fp16' or 'fp32'")

        if not pattern_path.exists():
            raise FileNotFoundError(f"Pattern file not found: {pattern_path}")

        print(f"Loading real diffusion patterns from: {pattern_path}")
        data = np.load(str(pattern_path), allow_pickle=True)

        # Load mixed tensor using from_dict()
        mixed_tensor_dict = data["mixed_tensor"].item()
        mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

        # Load DIA matrix
        dia_dict = data["dia_matrix"].item()
        dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)
        print(
            f"  DIA matrix: {dia_matrix.led_count} LEDs, bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k} diagonals"
        )

        print(
            f"  Mixed tensor: {mixed_tensor.batch_size} LEDs, "
            f"{mixed_tensor.height}x{mixed_tensor.width}, "
            f"{mixed_tensor.block_size}x{mixed_tensor.block_size} blocks"
        )

        return mixed_tensor, dia_matrix

    def load_ata_inverse(self, precision: str = "fp16") -> np.ndarray:
        """
        Load real ATA inverse matrices from stored patterns.

        Args:
            precision: Either "fp16" or "fp32" to specify which patterns to load

        Returns:
            ATA inverse matrices in shape (3, led_count, led_count)
        """
        from pathlib import Path

        # Load the patterns based on precision
        if precision == "fp16":
            pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_fp16.npz"
        elif precision == "fp32":
            pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_fp32.npz"
        else:
            raise ValueError(f"Invalid precision: {precision}. Must be 'fp16' or 'fp32'")

        if not pattern_path.exists():
            raise FileNotFoundError(f"Pattern file not found: {pattern_path}")

        data = np.load(str(pattern_path), allow_pickle=True)

        if "ata_inverse" in data:
            ata_inverse = data["ata_inverse"]
            print(f"Loaded real ATA inverse: shape={ata_inverse.shape}, dtype={ata_inverse.dtype}")
            return ata_inverse
        else:
            raise KeyError("No ATA inverse found in pattern file")

    def create_test_frame(self, frame_format: str = "planar") -> np.ndarray:
        """
        Create test target frame.

        Args:
            frame_format: "planar" for (3, 480, 800) or "hwc" for (480, 800, 3)

        Returns:
            Test frame in specified format
        """
        height, width = 480, 800

        # Create simple test pattern
        frame_hwc = np.zeros((height, width, 3), dtype=np.uint8)

        # Add some simple patterns
        # Red stripe
        frame_hwc[100:150, :, 0] = 255
        # Green circle
        center_y, center_x = height // 2, width // 2
        radius = 50
        y, x = np.ogrid[:height, :width]
        mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius**2
        frame_hwc[mask, 1] = 255
        # Blue gradient
        frame_hwc[:, :, 2] = (np.arange(width) / width * 255).astype(np.uint8)

        if frame_format == "planar":
            # Convert to (3, H, W)
            return frame_hwc.transpose(2, 0, 1)
        else:
            return frame_hwc

    def test_optimization_with_real_patterns(self):
        """Test optimization using real patterns with detailed profiling."""
        # Load real diffusion patterns
        mixed_tensor, dia_matrix = self.load_real_diffusion_patterns()
        led_count = mixed_tensor.batch_size

        print("\nTesting optimization with real patterns and DIA matrix:")
        print(f"  LED count: {led_count}")
        print(f"  Mixed tensor format: {mixed_tensor.dtype}")
        print(f"  DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k} diagonals")

        # Test frame
        target_frame = self.create_test_frame("planar")

        # Import timing functions
        import time

        from src.utils.frame_optimizer import _calculate_atb

        print("\n  === WARMUP PHASE ====")

        # Load ATA inverse for optimization
        ata_inverse = self.load_ata_inverse()

        # Warmup runs to eliminate initialization costs (3 runs)
        for i in range(3):
            _ = optimize_frame_led_values(
                target_frame=target_frame,
                at_matrix=mixed_tensor,  # Use mixed tensor for A^T b
                ata_matrix=dia_matrix,  # Use DIA matrix for A^T A operations
                ata_inverse=ata_inverse,  # Required parameter
                max_iterations=5,
                compute_error_metrics=False,
                debug=False,
                enable_timing=False,  # Disable timing for warmup
            )

        print("  Warmup complete (3 iterations)")

        print("\n  === DETAILED PERFORMANCE PROFILING ====")

        # Step 1: Time A^T b calculation separately
        print("\n  [Step 1] A^T b calculation:")
        target_frame_uint8 = target_frame.astype(np.uint8)

        # Time multiple A^T b calculations
        atb_times = []
        for i in range(5):
            start_time = time.perf_counter()
            ATb = _calculate_atb(target_frame_uint8, mixed_tensor, debug=False)
            atb_times.append(time.perf_counter() - start_time)

        atb_avg = np.mean(atb_times)
        atb_std = np.std(atb_times)
        print(f"    A^T b time: {atb_avg:.4f}±{atb_std:.4f}s (5 runs)")
        print(f"    A^T b shape: {ATb.shape}")

        # Step 2: Run optimization with detailed timing breakdown
        print("\n  [Step 2] Full optimization with timing breakdown:")

        result = optimize_frame_led_values(
            target_frame=target_frame,
            at_matrix=mixed_tensor,  # Use mixed tensor for A^T b
            ata_matrix=dia_matrix,  # Use DIA matrix for A^T A operations
            ata_inverse=ata_inverse,  # Required parameter
            max_iterations=5,  # Use new default with ATA inverse initialization
            compute_error_metrics=False,  # Exclude MSE calculation
            debug=True,
            enable_timing=False,  # Disable timing for now due to issues
        )

        print("    Optimization completed:")
        print(f"      Converged: {result.converged}")
        print(f"      Iterations: {result.iterations}")
        print(f"      LED values shape: {result.led_values.shape}")
        print(f"      LED values range: [{result.led_values.min()}, {result.led_values.max()}]")

        if result.timing_data:
            print("\n    Timing breakdown:")
            for step, duration in result.timing_data.items():
                print(f"      {step}: {duration:.4f}s")

        if hasattr(result, "step_sizes") and result.step_sizes is not None:
            print(f"    Step sizes: {[f'{s:.6f}' for s in result.step_sizes[:5]]} (first 5)")

        # Step 3: Run multiple trials for statistical accuracy
        print("\n  [Step 3] Multiple trials for performance statistics (10 runs):")

        trial_times = []
        trial_iterations = []
        trial_timings = []

        for trial in range(10):
            start_time = time.perf_counter()
            trial_result = optimize_frame_led_values(
                target_frame=target_frame,
                at_matrix=mixed_tensor,  # Mixed tensor for A^T b
                ata_matrix=dia_matrix,  # DIA matrix for A^T A operations
                ata_inverse=ata_inverse,  # Required parameter
                max_iterations=5,  # Use new default with ATA inverse initialization
                compute_error_metrics=False,  # Exclude MSE calculation time
                debug=False,
                enable_timing=False,  # Disable timing for now
            )
            trial_time = time.perf_counter() - start_time

            trial_times.append(trial_time)
            trial_iterations.append(trial_result.iterations)
            if trial_result.timing_data:
                trial_timings.append(trial_result.timing_data)

        # Performance statistics
        avg_time = np.mean(trial_times)
        std_time = np.std(trial_times)
        avg_iterations = np.mean(trial_iterations)
        avg_time_per_iter = avg_time / avg_iterations

        print(f"    Total time: {avg_time:.4f}±{std_time:.4f}s")
        print(f"    Average iterations: {avg_iterations:.1f}")
        print(f"    Time per iteration: {avg_time_per_iter:.4f}s")
        print(f"    Potential FPS: {1.0 / avg_time:.1f} fps")  # FPS based on total time, not per iteration

        # Step 4: Average timing breakdown across trials with detailed step size analysis
        if trial_timings:
            print("\n  [Step 4] Average timing breakdown across trials:")

            # Calculate average for each timing section
            avg_timings = {}
            for section in trial_timings[0]:
                section_times = [t[section] for t in trial_timings if section in t]
                if section_times:
                    avg_timings[section] = np.mean(section_times)

            total_tracked = sum(avg_timings.values())

            # Group and display timing sections
            print("    === Core Optimization Sections ===")
            core_sections = [
                "ata_multiply",
                "gradient_calculation",
                "gradient_step",
                "convergence_check",
                "convergence_and_updates",
            ]
            for section in core_sections:
                if section in avg_timings:
                    avg_duration = avg_timings[section]
                    percentage = (avg_duration / total_tracked * 100) if total_tracked > 0 else 0
                    print(f"    {section}: {avg_duration:.4f}s ({percentage:.1f}%)")

            print("    === Step Size Calculation Breakdown ===")
            step_size_sections = [
                "step_size_g_dot_g",
                "step_size_g_ata_g",
                "step_size_division",
            ]
            step_size_total = sum(avg_timings.get(s, 0) for s in step_size_sections)
            for section in step_size_sections:
                if section in avg_timings:
                    avg_duration = avg_timings[section]
                    percentage = (avg_duration / total_tracked * 100) if total_tracked > 0 else 0
                    step_pct = (avg_duration / step_size_total * 100) if step_size_total > 0 else 0
                    print(f"    {section}: {avg_duration:.4f}s ({percentage:.1f}% total, {step_pct:.1f}% of step size)")

            step_size_pct = (step_size_total / total_tracked * 100) if total_tracked > 0 else 0
            print(f"    Total step size calculation: {step_size_total:.4f}s ({step_size_pct:.1f}%)")

            print("    === Other Sections ===")
            other_sections = [s for s in avg_timings if s not in core_sections + step_size_sections]
            for section in sorted(other_sections):
                avg_duration = avg_timings[section]
                percentage = (avg_duration / total_tracked * 100) if total_tracked > 0 else 0
                print(f"    {section}: {avg_duration:.4f}s ({percentage:.1f}%)")

            # Check optimization loop timing coverage
            optimization_loop_time = avg_timings.get("optimization_loop", 0)
            if optimization_loop_time > 0:
                print("    \n    === Optimization Loop Analysis ===")
                print(f"    Optimization loop total: {optimization_loop_time:.4f}s")
                print(f"    Average per iteration: {optimization_loop_time / 5:.4f}s")
                print("    ✅ Complete timing coverage - all operations captured")

        # Final validation
        print("\n  === VALIDATION ===")
        assert result.led_values.shape == (3, led_count)
        assert result.led_values.dtype == np.uint8
        assert np.all(result.led_values >= 0) and np.all(result.led_values <= 255)
        assert result.iterations > 0
        print("    ✅ All validations passed")

        # Summary
        print("\n  === PERFORMANCE SUMMARY ====")
        print("    Configuration: Mixed tensor (A^T b) + DIA matrix (A^T A)")
        print(f"    LED count: {led_count}")
        print("    Frame size: 480x800")
        print(f"    DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k} diagonals")
        print(f"    Average optimization time: {avg_time:.4f}s")
        print(f"    Time per iteration: {avg_time_per_iter:.4f}s")
        print("    Target performance: <5ms per iteration")

    def get_expected_led_values_fixture(self, precision: str) -> np.ndarray:
        """
        Get expected LED values for regression testing.

        This fixture contains the expected LED output values for the standard test frame
        when optimized with the specified precision patterns. It serves as a regression test to detect
        changes in optimization behavior.

        Args:
            precision: Either "fp16" or "fp32" to specify which fixture to load

        Returns:
            Expected LED values in shape (3, 2624) with dtype uint8
        """
        # Load the expected LED values fixture from file
        # Test frame: standard test pattern with red stripe, green circle, blue gradient
        # Parameters: max_iterations=5, specified precision
        from pathlib import Path

        fixture_path = Path(__file__).parent / "fixtures" / f"led_values_fixture_{precision}.npy"
        if fixture_path.exists():
            expected_values = np.load(fixture_path)
            return expected_values
        else:
            # Fixture not found - test must fail
            return None

    def _test_optimization_patterns_regression(self, precision: str):
        """
        Test optimization with specified precision patterns and validate against fixture for regression detection.

        Args:
            precision: Either "fp16" or "fp32" to specify which patterns to use
        """
        # Load patterns with specified precision
        mixed_tensor, dia_matrix = self.load_real_diffusion_patterns(precision=precision)
        ata_inverse = self.load_ata_inverse(precision=precision)
        led_count = mixed_tensor.batch_size

        print(f"\nTesting optimization with {precision} patterns:")
        print(f"  LED count: {led_count}")
        print(f"  Mixed tensor format: {mixed_tensor.dtype}")
        print(f"  DIA matrix storage dtype: {dia_matrix.storage_dtype}")
        print(f"  ATA inverse dtype: {ata_inverse.dtype}")

        # Create the same test frame as used in other tests
        target_frame = self.create_test_frame("planar")

        # Run optimization with standard parameters
        result = optimize_frame_led_values(
            target_frame=target_frame,
            at_matrix=mixed_tensor,
            ata_matrix=dia_matrix,
            ata_inverse=ata_inverse,
            max_iterations=5,
            compute_error_metrics=True,
            debug=False,
            enable_timing=False,
        )

        # Validate basic properties
        assert result.led_values.shape == (3, led_count)
        assert result.led_values.dtype == np.uint8
        assert np.all(result.led_values >= 0) and np.all(result.led_values <= 255)
        assert result.iterations > 0

        print(f"  LED values shape: {result.led_values.shape}")
        print(f"  LED values range: [{result.led_values.min()}, {result.led_values.max()}]")
        print(f"  Iterations: {result.iterations}")

        if result.error_metrics:
            print(
                f"  Error metrics: MSE={result.error_metrics.get('mse', 'N/A'):.6f}, "
                f"PSNR={result.error_metrics.get('psnr', 'N/A'):.2f}"
            )

        # Get expected values from fixture
        expected_values = self.get_expected_led_values_fixture(precision)

        if expected_values is None:
            # No fixture found - generate it and fail the test
            print("\n  === FIXTURE GENERATION ===")
            print(f"  No fixture found for {precision} patterns - generating fixture")
            print("  " + "=" * 60)

            # Save to fixture file
            from pathlib import Path

            fixture_file = Path(__file__).parent / "fixtures" / f"led_values_fixture_{precision}.npy"
            fixture_file.parent.mkdir(exist_ok=True)
            np.save(fixture_file, result.led_values)

            # Print info about the generated fixture
            print(f"  # Expected LED values for {precision} patterns (generated {time.strftime('%Y-%m-%d')})")
            print("  # Test frame: standard test pattern with red stripe, green circle, blue gradient")
            print(f"  # Parameters: max_iterations=5, {precision} precision")
            print(f"  # LED count: {led_count}")
            print(f"  # Shape: {result.led_values.shape}")
            print(f"  # Range: [{result.led_values.min()}, {result.led_values.max()}]")
            print(f"  # Dtype: {result.led_values.dtype}")
            print(f"  # Saved to: {fixture_file}")

            led_mean = np.mean(result.led_values)
            led_std = np.std(result.led_values)
            print(f"  # Statistics: mean={led_mean:.2f}, std={led_std:.2f}")
            print("  " + "=" * 60)

            # Fail the test - fixture generation run
            pytest.fail(
                f"Generated fixture for {precision} patterns. Please run test again to validate against fixture."
            )

        else:
            # Regression test - compare against expected values
            print("\n  === REGRESSION VALIDATION ===")
            print("  Comparing against expected fixture values...")

            # Compare shapes
            assert (
                result.led_values.shape == expected_values.shape
            ), f"Shape mismatch: got {result.led_values.shape}, expected {expected_values.shape}"

            # Compare dtypes
            assert (
                result.led_values.dtype == expected_values.dtype
            ), f"Dtype mismatch: got {result.led_values.dtype}, expected {expected_values.dtype}"

            # Compare values with tolerance for numerical precision differences
            max_diff = np.max(np.abs(result.led_values.astype(np.float32) - expected_values.astype(np.float32)))
            mean_diff = np.mean(np.abs(result.led_values.astype(np.float32) - expected_values.astype(np.float32)))

            print(f"  Difference statistics: max={max_diff:.2f}, mean={mean_diff:.2f}")

            # Set tolerance to 1.0 as requested
            tolerance = 1.0
            assert (
                max_diff <= tolerance
            ), f"LED values differ by more than {tolerance}. Max difference: {max_diff:.2f}. This indicates a regression in optimization behavior."

            print(f"  ✅ Values match within tolerance ({tolerance})")

        print(f"  ✅ {precision} patterns test completed")

    def test_optimization_fp16_patterns_regression(self):
        """Test optimization with fp16 patterns and validate against fixture for regression detection."""
        self._test_optimization_patterns_regression("fp16")

    def test_optimization_fp32_patterns_regression(self):
        """Test optimization with fp32 patterns and validate against fixture for regression detection."""
        self._test_optimization_patterns_regression("fp32")
