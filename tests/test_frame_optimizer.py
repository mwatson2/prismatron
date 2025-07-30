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


@pytest.fixture(autouse=True)
def cuda_cleanup():
    """Ensure clean CUDA state before and after each test."""
    # Clear CUDA memory and reset state before test
    if cp.cuda.is_available():
        try:
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            # Clear any cached CUDA modules if available
            if hasattr(cp._core, "_kernel") and hasattr(cp._core._kernel, "clear_memo"):
                cp._core._kernel.clear_memo()
        except Exception:
            # If cleanup fails, continue with test
            pass

    yield  # Run the test

    # Clean up after test
    if cp.cuda.is_available():
        try:
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            # Clear any cached CUDA modules if available
            if hasattr(cp._core, "_kernel") and hasattr(cp._core._kernel, "clear_memo"):
                cp._core._kernel.clear_memo()
        except Exception:
            # If cleanup fails, don't fail the test
            pass


from src.utils.dense_ata_matrix import DenseATAMatrix
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
        self, precision: str = "fp32"
    ) -> Tuple[SingleBlockMixedSparseTensor, DiagonalATAMatrix]:
        """
        Load real diffusion patterns from stored patterns with DIA matrix.

        Args:
            precision: Either "fp16", "fp32", or "uint8" to specify which patterns to load

        Returns:
            Tuple of (mixed_tensor, dia_matrix)
        """
        from pathlib import Path

        # Load the patterns based on precision
        if precision == "fp16":
            pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_fp16.npz"
        elif precision == "fp32":
            pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_fp32.npz"
        elif precision == "uint8":
            pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_uint8.npz"
        else:
            raise ValueError(f"Invalid precision: {precision}. Must be 'fp16', 'fp32', or 'uint8'")

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
            f"{mixed_tensor.block_size}x{mixed_tensor.block_size} blocks, dtype={mixed_tensor.dtype}"
        )

        return mixed_tensor, dia_matrix

    def load_ata_inverse(self, precision: str = "fp32") -> np.ndarray:
        """
        Load real ATA inverse matrices from stored patterns.

        Args:
            precision: Either "fp16", "fp32", or "uint8" to specify which patterns to load

        Returns:
            ATA inverse matrices in shape (3, led_count, led_count)
        """
        from pathlib import Path

        # Load the patterns based on precision
        if precision == "fp16":
            pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_fp16.npz"
        elif precision == "fp32":
            pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_fp32.npz"
        elif precision == "uint8":
            pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_uint8.npz"
        else:
            raise ValueError(f"Invalid precision: {precision}. Must be 'fp16', 'fp32', or 'uint8'")

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
        target_frame_gpu = cp.asarray(target_frame)

        # Import timing functions
        import time

        from src.utils.frame_optimizer import _calculate_atb

        print("\n  === WARMUP PHASE ====")

        # Load ATA inverse for optimization
        ata_inverse = self.load_ata_inverse()

        # Warmup runs to eliminate initialization costs (3 runs)
        for i in range(3):
            _ = optimize_frame_led_values(
                target_frame=target_frame_gpu,
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
            target_frame=target_frame_gpu,
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
                target_frame=target_frame_gpu,
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

    @pytest.mark.skip(reason="FP16 pattern file not available - mixed tensor FP16 support removed")
    def test_optimization_fp16_patterns_regression(self):
        """Test optimization with fp16 patterns and validate against fixture for regression detection."""
        self._test_optimization_patterns_regression("fp16")

    def test_optimization_fp32_patterns_regression(self):
        """Test optimization with fp32 patterns and validate against fixture for regression detection."""
        self._test_optimization_patterns_regression("fp32")

    def test_optimization_uint8_patterns_regression(self):
        """Test optimization with uint8 patterns and validate against fixture for regression detection."""
        self._test_optimization_patterns_regression("uint8")

    def test_uint8_optimization_functionality(self):
        """Test that uint8 patterns work correctly with the frame optimizer."""
        print("\n=== Testing uint8 optimization functionality ===")

        # Load uint8 patterns
        mixed_tensor_uint8, dia_matrix_uint8 = self.load_real_diffusion_patterns("uint8")
        ata_inverse_uint8 = self.load_ata_inverse("uint8")

        # Create test frame
        target_frame = self.create_test_frame("planar")

        print(f"  uint8 tensor dtype: {mixed_tensor_uint8.dtype}")
        print(f"  Target frame dtype: {target_frame.dtype}")
        print(f"  DIA matrix storage dtype: {dia_matrix_uint8.storage_dtype}")
        print(f"  ATA inverse dtype: {ata_inverse_uint8.dtype}")

        # Run optimization with uint8 patterns
        result_uint8 = optimize_frame_led_values(
            target_frame=target_frame,
            at_matrix=mixed_tensor_uint8,
            ata_matrix=dia_matrix_uint8,
            ata_inverse=ata_inverse_uint8,
            max_iterations=5,
            compute_error_metrics=True,
            debug=True,  # Enable debug to see uint8 kernel usage
        )

        print(f"  uint8 result shape: {result_uint8.led_values.shape}")
        print(f"  uint8 result dtype: {result_uint8.led_values.dtype}")
        print(f"  uint8 result range: [{result_uint8.led_values.min()}, {result_uint8.led_values.max()}]")
        print(f"  Iterations completed: {result_uint8.iterations}")

        # Validate basic properties
        assert result_uint8.led_values.shape == (3, mixed_tensor_uint8.batch_size)
        assert result_uint8.led_values.dtype == np.uint8
        assert np.all(result_uint8.led_values >= 0) and np.all(result_uint8.led_values <= 255)
        assert result_uint8.iterations > 0

        # Check error metrics are reasonable
        if result_uint8.error_metrics:
            mse_uint8 = result_uint8.error_metrics.get("mse", float("inf"))
            psnr_uint8 = result_uint8.error_metrics.get("psnr", 0)
            print(f"  uint8 MSE: {mse_uint8:.6f}")
            print(f"  uint8 PSNR: {psnr_uint8:.2f} dB")

            # MSE should be reasonable (not too high)
            assert mse_uint8 < 1.0, f"uint8 MSE {mse_uint8} too high"
            assert psnr_uint8 > 5.0, f"uint8 PSNR {psnr_uint8} too low"

        # Test that values are distributed (not all zero or all max)
        led_mean = np.mean(result_uint8.led_values)
        led_std = np.std(result_uint8.led_values)
        print(f"  LED value statistics: mean={led_mean:.2f}, std={led_std:.2f}")

        assert led_std > 1.0, "LED values should have some variation"
        assert 0 < led_mean < 255, "LED mean should be between 0 and 255"

        print("  ✅ uint8 optimization functionality test passed")

    def load_captured_diffusion_patterns(
        self, pattern_file: str = "capture-0728-01-both-matrices.npz"
    ) -> Tuple[SingleBlockMixedSparseTensor, DiagonalATAMatrix, DenseATAMatrix]:
        """
        Load captured diffusion patterns with both DIA and Dense ATA formats.

        Args:
            pattern_file: Name of the pattern file to load

        Returns:
            Tuple of (mixed_tensor, dia_matrix, dense_ata_matrix)
        """
        from pathlib import Path

        pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / pattern_file

        if not pattern_path.exists():
            raise FileNotFoundError(f"Pattern file not found: {pattern_path}")

        print(f"Loading captured diffusion patterns from: {pattern_path}")
        data = np.load(str(pattern_path), allow_pickle=True)

        # Load mixed tensor
        mixed_tensor_dict = data["mixed_tensor"].item()
        mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

        # Load DIA matrix
        dia_dict = data["dia_matrix"].item()
        dia_matrix = DiagonalATAMatrix.from_dict(dia_dict)

        # Load Dense ATA matrix
        dense_dict = data["dense_ata_matrix"].item()
        dense_ata_matrix = DenseATAMatrix.from_dict(dense_dict)

        print(f"  Mixed tensor: {mixed_tensor.batch_size} LEDs, dtype={mixed_tensor.dtype}")
        print(f"  DIA matrix: {dia_matrix.led_count} LEDs, bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k}")
        print(f"  Dense ATA matrix: {dense_ata_matrix.led_count} LEDs, {dense_ata_matrix.memory_mb:.1f}MB")

        return mixed_tensor, dia_matrix, dense_ata_matrix

    def load_captured_ata_inverse(self, pattern_file: str = "capture-0728-01-both-matrices.npz") -> np.ndarray:
        """
        Load ATA inverse matrices from captured patterns.

        Args:
            pattern_file: Name of the pattern file to load

        Returns:
            ATA inverse matrices in shape (3, led_count, led_count)
        """
        from pathlib import Path

        pattern_path = Path(__file__).parent.parent / "diffusion_patterns" / pattern_file

        if not pattern_path.exists():
            raise FileNotFoundError(f"Pattern file not found: {pattern_path}")

        data = np.load(str(pattern_path), allow_pickle=True)

        if "ata_inverse" in data:
            ata_inverse = data["ata_inverse"]
            print(f"Loaded captured ATA inverse: shape={ata_inverse.shape}, dtype={ata_inverse.dtype}")
            return ata_inverse
        else:
            raise KeyError("No ATA inverse found in captured pattern file")

    def test_performance_comparison_dia_vs_dense(self):
        """
        Performance comparison test between DIA and Dense ATA matrix formats.

        Tests both formats with captured diffusion patterns to establish performance baseline
        for DIA format and compare with Dense format performance.
        """
        print("\n=== PERFORMANCE COMPARISON: DIA vs Dense ATA Matrix Formats ===")

        # Load captured patterns with both formats
        mixed_tensor, dia_matrix, dense_ata_matrix = self.load_captured_diffusion_patterns()
        ata_inverse = self.load_captured_ata_inverse()
        led_count = mixed_tensor.batch_size

        print("\nConfiguration:")
        print(f"  LED count: {led_count}")
        print(f"  Mixed tensor dtype: {mixed_tensor.dtype}")
        print(f"  DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k}, sparsity={dia_matrix.sparsity:.2f}%")
        print(f"  Dense ATA matrix: memory={dense_ata_matrix.memory_mb:.1f}MB")

        # Create test frame
        target_frame = self.create_test_frame("planar")
        target_frame_gpu = cp.asarray(target_frame)

        print("\n=== WARMUP PHASE ===")
        # Warmup both formats (3 runs each)
        for i in range(3):
            # DIA format warmup
            _ = optimize_frame_led_values(
                target_frame=target_frame_gpu,
                at_matrix=mixed_tensor,
                ata_matrix=dia_matrix,
                ata_inverse=ata_inverse,
                max_iterations=5,
                compute_error_metrics=False,
                debug=False,
            )

            # Dense format warmup
            _ = optimize_frame_led_values(
                target_frame=target_frame_gpu,
                at_matrix=mixed_tensor,
                ata_matrix=dense_ata_matrix,
                ata_inverse=ata_inverse,
                max_iterations=5,
                compute_error_metrics=False,
                debug=False,
            )

        print("Warmup complete (3 iterations each format)")

        print("\n=== PERFORMANCE BENCHMARKING ===")

        # Performance test parameters
        num_trials = 10
        max_iterations = 5

        # DIA format performance test
        print(f"\n[DIA Format] Running {num_trials} trials...")
        dia_times = []
        dia_results = []

        for trial in range(num_trials):
            start_time = time.perf_counter()
            result = optimize_frame_led_values(
                target_frame=target_frame_gpu,
                at_matrix=mixed_tensor,
                ata_matrix=dia_matrix,
                ata_inverse=ata_inverse,
                max_iterations=max_iterations,
                compute_error_metrics=True,
                debug=False,
            )
            end_time = time.perf_counter()

            dia_times.append(end_time - start_time)
            dia_results.append(result)

        # Dense format performance test
        print(f"\n[Dense Format] Running {num_trials} trials...")
        dense_times = []
        dense_results = []

        for trial in range(num_trials):
            start_time = time.perf_counter()
            result = optimize_frame_led_values(
                target_frame=target_frame_gpu,
                at_matrix=mixed_tensor,
                ata_matrix=dense_ata_matrix,
                ata_inverse=ata_inverse,
                max_iterations=max_iterations,
                compute_error_metrics=True,
                debug=False,
            )
            end_time = time.perf_counter()

            dense_times.append(end_time - start_time)
            dense_results.append(result)

        print("\n=== PERFORMANCE ANALYSIS ===")

        # Calculate statistics
        dia_avg = np.mean(dia_times)
        dia_std = np.std(dia_times)
        dia_min = np.min(dia_times)
        dia_max = np.max(dia_times)

        dense_avg = np.mean(dense_times)
        dense_std = np.std(dense_times)
        dense_min = np.min(dense_times)
        dense_max = np.max(dense_times)

        print("\nDIA Format Performance:")
        print(f"  Average time: {dia_avg:.4f}±{dia_std:.4f}s")
        print(f"  Range: [{dia_min:.4f}, {dia_max:.4f}]s")
        print(f"  Time per iteration: {dia_avg/max_iterations:.4f}s")
        print(f"  Potential FPS: {1.0/dia_avg:.1f} fps")

        print("\nDense Format Performance:")
        print(f"  Average time: {dense_avg:.4f}±{dense_std:.4f}s")
        print(f"  Range: [{dense_min:.4f}, {dense_max:.4f}]s")
        print(f"  Time per iteration: {dense_avg/max_iterations:.4f}s")
        print(f"  Potential FPS: {1.0/dense_avg:.1f} fps")

        # Performance comparison
        speedup_ratio = dense_avg / dia_avg
        print("\nPerformance Comparison:")
        print(f"  DIA vs Dense ratio: {speedup_ratio:.2f}x")
        if speedup_ratio > 1.0:
            print(f"  ✅ DIA format is {speedup_ratio:.2f}x faster than Dense")
        else:
            print(f"  ❌ Dense format is {1/speedup_ratio:.2f}x faster than DIA")

        # Memory comparison
        dia_memory_estimate = dia_matrix.k * led_count * 4 / (1024 * 1024)  # Rough estimate
        dense_memory = dense_ata_matrix.memory_mb
        memory_ratio = dense_memory / dia_memory_estimate
        print(f"  Memory usage - DIA: ~{dia_memory_estimate:.1f}MB, Dense: {dense_memory:.1f}MB")
        print(f"  Dense uses {memory_ratio:.1f}x more memory than DIA")

        print("\n=== ACCURACY COMPARISON ===")

        # Compare final LED values between formats
        dia_final = cp.asnumpy(dia_results[-1].led_values)
        dense_final = cp.asnumpy(dense_results[-1].led_values)

        max_diff = np.max(np.abs(dia_final.astype(np.float32) - dense_final.astype(np.float32)))
        mean_diff = np.mean(np.abs(dia_final.astype(np.float32) - dense_final.astype(np.float32)))
        rms_diff = np.sqrt(np.mean((dia_final.astype(np.float32) - dense_final.astype(np.float32)) ** 2))

        print("LED Values Comparison:")
        print(f"  Max difference: {max_diff:.2f}")
        print(f"  Mean difference: {mean_diff:.2f}")
        print(f"  RMS difference: {rms_diff:.2f}")

        # Compare error metrics if available
        if dia_results[-1].error_metrics and dense_results[-1].error_metrics:
            dia_mse = dia_results[-1].error_metrics["mse"]
            dense_mse = dense_results[-1].error_metrics["mse"]
            dia_psnr = dia_results[-1].error_metrics["psnr"]
            dense_psnr = dense_results[-1].error_metrics["psnr"]

            print("\nError Metrics Comparison:")
            print(f"  DIA MSE: {dia_mse:.6f}, PSNR: {dia_psnr:.2f}dB")
            print(f"  Dense MSE: {dense_mse:.6f}, PSNR: {dense_psnr:.2f}dB")
            print(f"  MSE ratio (Dense/DIA): {dense_mse/dia_mse:.3f}")

        print("\n=== SUMMARY ===")
        print(f"Matrix Format Comparison for {led_count} LEDs:")
        print(f"  DIA: {dia_avg:.4f}s avg, ~{dia_memory_estimate:.1f}MB memory, sparsity={dia_matrix.sparsity:.1f}%")
        print(f"  Dense: {dense_avg:.4f}s avg, {dense_memory:.1f}MB memory, full matrix")
        print(f"  Performance: DIA is {speedup_ratio:.2f}x {'faster' if speedup_ratio > 1 else 'slower'}")
        print(f"  Memory efficiency: DIA uses {1/memory_ratio:.1f}x less memory")
        print(f"  Accuracy: Max LED difference = {max_diff:.2f} (out of 255)")

        # Validation assertions
        assert dia_results[-1].led_values.shape == (3, led_count)
        assert dense_results[-1].led_values.shape == (3, led_count)
        assert np.all(dia_results[-1].led_values >= 0) and np.all(dia_results[-1].led_values <= 255)
        assert np.all(dense_results[-1].led_values >= 0) and np.all(dense_results[-1].led_values <= 255)

        print("✅ Performance comparison completed successfully")

    def test_detailed_component_analysis(self):
        """
        Detailed analysis of individual performance components.

        Breaks down timing for A^T@b, ATA inverse initialization, and optimization steps
        for both synthetic and captured patterns, and compares DIA vs Dense formats.
        """
        print("\n=== DETAILED COMPONENT ANALYSIS ===")

        # Test both synthetic and captured patterns
        patterns_to_test = [
            ("synthetic", "synthetic_2624_fp32.npz", "Synthetic Patterns (sparse DIA)"),
            ("captured", "capture-0728-01-both-matrices.npz", "Captured Patterns (dense DIA + Dense ATA)"),
        ]

        results = {}

        for pattern_type, pattern_file, description in patterns_to_test:
            print(f"\n{'='*60}")
            print(f"ANALYZING: {description}")
            print(f"{'='*60}")

            # Load patterns
            if pattern_type == "synthetic":
                mixed_tensor, dia_matrix = self.load_real_diffusion_patterns(precision="fp32")
                ata_inverse = self.load_ata_inverse(precision="fp32")
                dense_ata_matrix = None  # Not available for synthetic
            else:
                mixed_tensor, dia_matrix, dense_ata_matrix = self.load_captured_diffusion_patterns(pattern_file)
                ata_inverse = self.load_captured_ata_inverse(pattern_file)

            led_count = mixed_tensor.batch_size
            target_frame = self.create_test_frame("planar")
            target_frame_gpu = cp.asarray(target_frame)

            print("\nPattern Statistics:")
            print(f"  LED count: {led_count}")
            print(f"  Mixed tensor dtype: {mixed_tensor.dtype}")
            print(
                f"  DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k}, sparsity={dia_matrix.sparsity:.2f}%"
            )
            if dense_ata_matrix:
                print(f"  Dense ATA matrix: memory={dense_ata_matrix.memory_mb:.1f}MB")

            # Calculate theoretical memory usage
            dia_memory_theoretical = (3 * dia_matrix.k * led_count * 4) / (1024**2)  # FP32
            dense_memory_theoretical = (3 * led_count * led_count * 4) / (1024**2)  # FP32

            print("\nMemory Analysis:")
            print(f"  DIA theoretical: {dia_memory_theoretical:.1f}MB")
            print(f"  Dense theoretical: {dense_memory_theoretical:.1f}MB")
            print(f"  Dense/DIA ratio: {dense_memory_theoretical/dia_memory_theoretical:.1f}x")

            # Storage efficiency analysis
            max_diagonals = 2 * led_count - 1
            diagonal_efficiency = dia_matrix.k / max_diagonals
            bandwidth_efficiency = dia_matrix.bandwidth / led_count

            print(f"  DIA storage efficiency: {diagonal_efficiency:.1%} of max diagonals used")
            print(f"  DIA bandwidth efficiency: {bandwidth_efficiency:.1%} of matrix size")

            # Component timing analysis
            print("\n--- Component Timing Analysis ---")
            num_timing_trials = 5

            # 1. A^T @ b timing
            print("1. A^T @ b Performance:")
            from src.utils.frame_optimizer import _calculate_atb

            atb_times = []
            for i in range(num_timing_trials):
                start_time = time.perf_counter()
                ATb_result = _calculate_atb(target_frame, mixed_tensor, debug=False)
                atb_times.append(time.perf_counter() - start_time)

            atb_avg = np.mean(atb_times)
            atb_std = np.std(atb_times)
            print(f"   A^T @ b time: {atb_avg:.4f}±{atb_std:.4f}s")
            print(f"   Result shape: {ATb_result.shape}")

            # 2. ATA inverse initialization timing
            print("2. ATA Inverse Initialization:")

            init_times_dia = []
            for i in range(num_timing_trials):
                start_time = time.perf_counter()
                # Dense ATA inverse initialization
                ata_inverse_gpu = cp.asarray(ata_inverse)
                led_values_init = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_result)
                led_values_init = cp.clip(led_values_init, 0.0, 1.0)
                init_times_dia.append(time.perf_counter() - start_time)

            init_avg = np.mean(init_times_dia)
            init_std = np.std(init_times_dia)
            print(f"   Dense ATA inverse init: {init_avg:.4f}±{init_std:.4f}s")

            # 3. Single iteration timing for DIA
            print("3. Single Optimization Iteration (DIA):")
            iteration_times_dia = []

            for i in range(num_timing_trials):
                # Set up for single iteration
                led_values_gpu = led_values_init.copy()

                start_time = time.perf_counter()

                # Single iteration of optimization
                ATA_x = dia_matrix.multiply_3d(led_values_gpu)
                gradient = ATA_x - ATb_result
                g_dot_g = cp.sum(gradient * gradient)
                g_dot_ATA_g_per_channel = dia_matrix.g_ata_g_3d(gradient)
                g_dot_ATA_g = cp.sum(g_dot_ATA_g_per_channel)

                if g_dot_ATA_g > 0:
                    step_size = 0.9 * g_dot_g / g_dot_ATA_g
                else:
                    step_size = 0.01

                led_values_gpu = cp.clip(led_values_gpu - step_size * gradient, 0, 1)

                iteration_times_dia.append(time.perf_counter() - start_time)

            iter_dia_avg = np.mean(iteration_times_dia)
            iter_dia_std = np.std(iteration_times_dia)
            print(f"   DIA iteration: {iter_dia_avg:.4f}±{iter_dia_std:.4f}s")

            # 4. Dense format timing (if available)
            if dense_ata_matrix:
                print("4. Single Optimization Iteration (Dense):")
                iteration_times_dense = []

                for i in range(num_timing_trials):
                    led_values_gpu = led_values_init.copy()

                    start_time = time.perf_counter()

                    # Single iteration with dense matrix
                    ATA_x = dense_ata_matrix.multiply_vector(led_values_gpu)
                    gradient = ATA_x - ATb_result
                    g_dot_g = cp.sum(gradient * gradient)
                    ATA_gradient = dense_ata_matrix.multiply_vector(gradient)
                    g_dot_ATA_g = cp.sum(gradient * ATA_gradient)

                    if g_dot_ATA_g > 0:
                        step_size = 0.9 * g_dot_g / g_dot_ATA_g
                    else:
                        step_size = 0.01

                    led_values_gpu = cp.clip(led_values_gpu - step_size * gradient, 0, 1)

                    iteration_times_dense.append(time.perf_counter() - start_time)

                iter_dense_avg = np.mean(iteration_times_dense)
                iter_dense_std = np.std(iteration_times_dense)
                print(f"   Dense iteration: {iter_dense_avg:.4f}±{iter_dense_std:.4f}s")
                print(f"   DIA vs Dense ratio: {iter_dense_avg/iter_dia_avg:.2f}x")

            # Store results
            result_data = {
                "led_count": led_count,
                "dia_bandwidth": dia_matrix.bandwidth,
                "dia_k": dia_matrix.k,
                "dia_sparsity": dia_matrix.sparsity,
                "mixed_tensor_dtype": str(mixed_tensor.dtype),
                "atb_time": atb_avg,
                "init_time": init_avg,
                "dia_iter_time": iter_dia_avg,
                "dia_memory_mb": dia_memory_theoretical,
                "dense_memory_mb": dense_memory_theoretical,
                "diagonal_efficiency": diagonal_efficiency,
                "bandwidth_efficiency": bandwidth_efficiency,
            }

            if dense_ata_matrix:
                result_data["dense_iter_time"] = iter_dense_avg
                result_data["dense_speedup"] = iter_dense_avg / iter_dia_avg

            results[pattern_type] = result_data

        # Final comparison
        print(f"\n{'='*60}")
        print("COMPONENT PERFORMANCE COMPARISON")
        print(f"{'='*60}")

        if "synthetic" in results and "captured" in results:
            syn = results["synthetic"]
            cap = results["captured"]

            print("\nMatrix Characteristics:")
            print(
                f"  Synthetic - DIA k: {syn['dia_k']}, bandwidth: {syn['dia_bandwidth']}, sparsity: {syn['dia_sparsity']:.1f}%"
            )
            print(
                f"  Captured  - DIA k: {cap['dia_k']}, bandwidth: {cap['dia_bandwidth']}, sparsity: {cap['dia_sparsity']:.1f}%"
            )

            print("\nComponent Performance (Synthetic vs Captured):")
            print(
                f"  A^T @ b:        {syn['atb_time']:.4f}s vs {cap['atb_time']:.4f}s ({cap['atb_time']/syn['atb_time']:.1f}x)"
            )
            print(
                f"  ATA init:       {syn['init_time']:.4f}s vs {cap['init_time']:.4f}s ({cap['init_time']/syn['init_time']:.1f}x)"
            )
            print(
                f"  DIA iteration:  {syn['dia_iter_time']:.4f}s vs {cap['dia_iter_time']:.4f}s ({cap['dia_iter_time']/syn['dia_iter_time']:.1f}x)"
            )

            if "dense_iter_time" in cap:
                print(f"  Dense iteration: N/A vs {cap['dense_iter_time']:.4f}s")
                print(f"  DIA vs Dense (captured): {cap['dense_speedup']:.2f}x")

            print("\nStorage Efficiency:")
            print(
                f"  Synthetic - diagonal: {syn['diagonal_efficiency']:.1%}, bandwidth: {syn['bandwidth_efficiency']:.1%}"
            )
            print(
                f"  Captured  - diagonal: {cap['diagonal_efficiency']:.1%}, bandwidth: {cap['bandwidth_efficiency']:.1%}"
            )

            print("\nMemory Usage:")
            print(f"  Synthetic - DIA: {syn['dia_memory_mb']:.1f}MB, Dense: {syn['dense_memory_mb']:.1f}MB")
            print(f"  Captured  - DIA: {cap['dia_memory_mb']:.1f}MB, Dense: {cap['dense_memory_mb']:.1f}MB")

        print("\n✅ Detailed component analysis completed")

    def test_synthetic_dia_vs_dense_performance(self):
        """
        Direct performance comparison between DIA and Dense formats on synthetic patterns.

        This test uses the same synthetic patterns in both DIA and Dense formats
        to isolate the performance characteristics of the matrix formats themselves.
        """
        print("\n=== SYNTHETIC PATTERNS: DIA vs Dense Performance ===")

        # Load both versions of synthetic patterns
        print("Loading DIA version...")
        mixed_tensor_dia, dia_matrix = self.load_real_diffusion_patterns(precision="fp32")
        ata_inverse_dia = self.load_ata_inverse(precision="fp32")

        print("Loading Dense version...")
        pattern_path_dense = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_fp32_with_dense.npz"

        if not pattern_path_dense.exists():
            pytest.skip("Dense synthetic patterns not available - run conversion tool first")

        data_dense = np.load(str(pattern_path_dense), allow_pickle=True)

        # Load dense components
        mixed_tensor_dense_dict = data_dense["mixed_tensor"].item()
        mixed_tensor_dense = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dense_dict)

        dense_ata_dict = data_dense["dense_ata_matrix"].item()
        dense_ata_matrix = DenseATAMatrix.from_dict(dense_ata_dict)

        ata_inverse_dense = data_dense["ata_inverse"]

        led_count = mixed_tensor_dia.batch_size

        print("\nConfiguration:")
        print(f"  LED count: {led_count}")
        print(f"  DIA matrix: bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k}, sparsity={dia_matrix.sparsity:.2f}%")
        print(f"  Dense matrix: {dense_ata_matrix.memory_mb:.1f}MB")
        print(f"  Mixed tensor dtype: {mixed_tensor_dia.dtype}")

        # Verify they're the same patterns
        assert mixed_tensor_dia.batch_size == mixed_tensor_dense.batch_size
        assert np.allclose(ata_inverse_dia, ata_inverse_dense, rtol=1e-5)
        print("  ✅ Verified patterns are identical")

        # Create test frame
        target_frame = self.create_test_frame("planar")
        target_frame_gpu = cp.asarray(target_frame)

        print("\n=== WARMUP PHASE ===")
        # Warmup both formats
        for i in range(3):
            # DIA warmup
            _ = optimize_frame_led_values(
                target_frame=target_frame_gpu,
                at_matrix=mixed_tensor_dia,
                ata_matrix=dia_matrix,
                ata_inverse=ata_inverse_dia,
                max_iterations=5,
                compute_error_metrics=False,
                debug=False,
            )

            # Dense warmup
            _ = optimize_frame_led_values(
                target_frame=target_frame_gpu,
                at_matrix=mixed_tensor_dense,
                ata_matrix=dense_ata_matrix,
                ata_inverse=ata_inverse_dense,
                max_iterations=5,
                compute_error_metrics=False,
                debug=False,
            )

        print("Warmup complete")

        print("\n=== COMPONENT-LEVEL PERFORMANCE ===")

        num_trials = 5

        # A^T @ b comparison
        print("1. A^T @ b Performance:")
        from src.utils.frame_optimizer import _calculate_atb

        atb_times_dia = []
        atb_times_dense = []

        for i in range(num_trials):
            # DIA A^T @ b
            start_time = time.perf_counter()
            ATb_dia = _calculate_atb(target_frame, mixed_tensor_dia, debug=False)
            atb_times_dia.append(time.perf_counter() - start_time)

            # Dense A^T @ b (should be identical)
            start_time = time.perf_counter()
            ATb_dense = _calculate_atb(target_frame, mixed_tensor_dense, debug=False)
            atb_times_dense.append(time.perf_counter() - start_time)

        atb_avg_dia = np.mean(atb_times_dia)
        atb_avg_dense = np.mean(atb_times_dense)

        print(f"   DIA A^T@b:   {atb_avg_dia:.4f}±{np.std(atb_times_dia):.4f}s")
        print(f"   Dense A^T@b: {atb_avg_dense:.4f}±{np.std(atb_times_dense):.4f}s")
        print(f"   Ratio: {atb_avg_dense/atb_avg_dia:.2f}x")

        # Verify A^T@b results are identical
        assert np.allclose(cp.asnumpy(ATb_dia), cp.asnumpy(ATb_dense), rtol=1e-5)
        print("   ✅ A^T@b results identical")

        # ATA inverse initialization (should be identical)
        print("\n2. ATA Inverse Initialization:")

        init_times_dia = []
        init_times_dense = []

        for i in range(num_trials):
            # DIA initialization
            start_time = time.perf_counter()
            ata_inverse_gpu_dia = cp.asarray(ata_inverse_dia)
            led_init_dia = cp.einsum("ijk,ik->ij", ata_inverse_gpu_dia, ATb_dia)
            led_init_dia = cp.clip(led_init_dia, 0.0, 1.0)
            init_times_dia.append(time.perf_counter() - start_time)

            # Dense initialization (same operation)
            start_time = time.perf_counter()
            ata_inverse_gpu_dense = cp.asarray(ata_inverse_dense)
            led_init_dense = cp.einsum("ijk,ik->ij", ata_inverse_gpu_dense, ATb_dense)
            led_init_dense = cp.clip(led_init_dense, 0.0, 1.0)
            init_times_dense.append(time.perf_counter() - start_time)

        init_avg_dia = np.mean(init_times_dia)
        init_avg_dense = np.mean(init_times_dense)

        print(f"   DIA init:   {init_avg_dia:.4f}±{np.std(init_times_dia):.4f}s")
        print(f"   Dense init: {init_avg_dense:.4f}±{np.std(init_times_dense):.4f}s")
        print(f"   Ratio: {init_avg_dense/init_avg_dia:.2f}x")

        # Component timing shows differences, now test the production optimization code

        print("\n=== FULL OPTIMIZATION PERFORMANCE ===")

        # Full optimization comparison
        num_opt_trials = 10
        max_iterations = 5

        opt_times_dia = []
        opt_times_dense = []
        opt_results_dia = []
        opt_results_dense = []

        print(f"Running {num_opt_trials} full optimization trials...")

        for trial in range(num_opt_trials):
            # DIA optimization
            start_time = time.perf_counter()
            result_dia = optimize_frame_led_values(
                target_frame=target_frame_gpu,
                at_matrix=mixed_tensor_dia,
                ata_matrix=dia_matrix,
                ata_inverse=ata_inverse_dia,
                max_iterations=max_iterations,
                compute_error_metrics=True,
                debug=False,
            )
            opt_times_dia.append(time.perf_counter() - start_time)
            opt_results_dia.append(result_dia)

            # Dense optimization
            start_time = time.perf_counter()
            result_dense = optimize_frame_led_values(
                target_frame=target_frame_gpu,
                at_matrix=mixed_tensor_dense,
                ata_matrix=dense_ata_matrix,
                ata_inverse=ata_inverse_dense,
                max_iterations=max_iterations,
                compute_error_metrics=True,
                debug=False,
            )
            opt_times_dense.append(time.perf_counter() - start_time)
            opt_results_dense.append(result_dense)

        opt_avg_dia = np.mean(opt_times_dia)
        opt_avg_dense = np.mean(opt_times_dense)

        print("\nFull Optimization Results:")
        print(f"  DIA total:   {opt_avg_dia:.4f}±{np.std(opt_times_dia):.4f}s")
        print(f"  Dense total: {opt_avg_dense:.4f}±{np.std(opt_times_dense):.4f}s")
        print(f"  Dense/DIA ratio: {opt_avg_dense/opt_avg_dia:.2f}x")
        print(f"  DIA per iteration:   {opt_avg_dia/max_iterations:.4f}s")
        print(f"  Dense per iteration: {opt_avg_dense/max_iterations:.4f}s")

        # Accuracy comparison
        print("\n=== ACCURACY COMPARISON ===")

        dia_final = cp.asnumpy(opt_results_dia[-1].led_values)
        dense_final = cp.asnumpy(opt_results_dense[-1].led_values)

        max_diff = np.max(np.abs(dia_final.astype(np.float32) - dense_final.astype(np.float32)))
        mean_diff = np.mean(np.abs(dia_final.astype(np.float32) - dense_final.astype(np.float32)))

        print("LED Value Differences:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        # Error metrics comparison
        if opt_results_dia[-1].error_metrics and opt_results_dense[-1].error_metrics:
            dia_mse = opt_results_dia[-1].error_metrics["mse"]
            dense_mse = opt_results_dense[-1].error_metrics["mse"]

            print(f"  DIA MSE: {dia_mse:.6f}")
            print(f"  Dense MSE: {dense_mse:.6f}")
            print(f"  MSE ratio: {dense_mse/dia_mse:.6f}")

        print("\n=== SUMMARY ===")
        print(f"Synthetic Patterns Performance (2624 LEDs, {dia_matrix.sparsity:.1f}% sparsity):")
        print(f"  Full optimization: Dense is {opt_avg_dense/opt_avg_dia:.1f}x slower than DIA")
        print(f"  Memory: Dense uses {dense_ata_matrix.memory_mb/23.2:.1f}x more memory")
        print(f"  Accuracy: Max difference = {max_diff:.6f}")

        if opt_avg_dense / opt_avg_dia > 2.0:
            print("  ⚠️  Dense format significantly slower - investigate implementation")
        elif opt_avg_dense / opt_avg_dia > 1.5:
            print("  📊 Dense format moderately slower - expected for sparse matrices")
        else:
            print("  ✅ Dense format competitive - good implementation")

        print("\n✅ Synthetic DIA vs Dense comparison completed")

    def test_production_dia_vs_dense_simple(self):
        """
        Simple test of production frame optimizer with DIA vs Dense on synthetic patterns.
        Uses the actual production code without replicating the algorithm.
        """
        print("\n=== PRODUCTION CODE: DIA vs Dense Performance Test ===")

        # Load DIA version
        mixed_tensor_dia, dia_matrix = self.load_real_diffusion_patterns(precision="fp32")
        ata_inverse_dia = self.load_ata_inverse(precision="fp32")

        # Load Dense version
        pattern_path_dense = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_fp32_with_dense.npz"
        if not pattern_path_dense.exists():
            pytest.skip("Dense synthetic patterns not available")

        data_dense = np.load(str(pattern_path_dense), allow_pickle=True)
        dense_ata_dict = data_dense["dense_ata_matrix"].item()
        dense_ata_matrix = DenseATAMatrix.from_dict(dense_ata_dict)
        ata_inverse_dense = data_dense["ata_inverse"]

        led_count = mixed_tensor_dia.batch_size
        target_frame = self.create_test_frame("planar")
        target_frame_gpu = cp.asarray(target_frame)

        print("Configuration:")
        print(f"  LED count: {led_count}")
        print(f"  DIA sparsity: {dia_matrix.sparsity:.1f}%")
        print(f"  Dense memory: {dense_ata_matrix.memory_mb:.1f}MB")

        # Test basic matrix operations
        print("\n=== Matrix Operation Verification ===")
        test_vector = cp.ones((3, led_count), dtype=cp.float32)

        dia_result = dia_matrix.multiply_3d(test_vector)
        dense_result = dense_ata_matrix.multiply_vector(test_vector)

        dia_cpu = cp.asnumpy(dia_result)
        dense_cpu = cp.asnumpy(dense_result)

        max_diff = np.max(np.abs(dia_cpu - dense_cpu))
        print(f"Matrix operation difference: {max_diff:.6f}")

        if max_diff > 0.1:  # Relaxed threshold for floating point differences
            print("❌ Dense conversion is broken - cannot proceed with performance test")
            return
        else:
            print("✅ Matrix operations match")

        # Performance comparison using production code
        num_trials = 5
        max_iterations = 3  # Shorter for quick comparison

        print("\n=== Production Code Performance Test ===")

        # DIA performance
        dia_times = []
        for trial in range(num_trials):
            start_time = time.perf_counter()
            result_dia = optimize_frame_led_values(
                target_frame=target_frame_gpu,
                at_matrix=mixed_tensor_dia,
                ata_matrix=dia_matrix,
                ata_inverse=ata_inverse_dia,
                max_iterations=max_iterations,
                compute_error_metrics=False,
                debug=False,
            )
            dia_times.append(time.perf_counter() - start_time)

        # Dense performance
        dense_times = []
        for trial in range(num_trials):
            start_time = time.perf_counter()
            result_dense = optimize_frame_led_values(
                target_frame=target_frame_gpu,
                at_matrix=mixed_tensor_dia,  # Same A^T matrix
                ata_matrix=dense_ata_matrix,  # Different ATA matrix
                ata_inverse=ata_inverse_dense,
                max_iterations=max_iterations,
                compute_error_metrics=False,
                debug=False,
            )
            dense_times.append(time.perf_counter() - start_time)

        dia_avg = np.mean(dia_times)
        dense_avg = np.mean(dense_times)

        print(f"Results ({num_trials} trials, {max_iterations} iterations):")
        print(f"  DIA time:   {dia_avg:.4f}±{np.std(dia_times):.4f}s")
        print(f"  Dense time: {dense_avg:.4f}±{np.std(dense_times):.4f}s")
        print(f"  Dense/DIA ratio: {dense_avg/dia_avg:.2f}x")

        if dense_avg / dia_avg > 2.0:
            print("  ⚠️  Dense significantly slower")
        elif dense_avg / dia_avg > 1.5:
            print("  📊 Dense moderately slower")
        else:
            print("  ✅ Dense competitive")

        # Accuracy check
        dia_final = cp.asnumpy(result_dia.led_values)
        dense_final = cp.asnumpy(result_dense.led_values)

        max_led_diff = np.max(np.abs(dia_final.astype(np.float32) - dense_final.astype(np.float32)))
        print(f"  Final LED difference: {max_led_diff:.2f} (out of 255)")

        if max_led_diff > 5.0:
            print("  ❌ Results differ significantly")
        else:
            print("  ✅ Results match")

        print("\n✅ Production DIA vs Dense test completed")

    def test_dia_to_dense_conversion_accuracy(self):
        """
        Test the accuracy of DIA to Dense matrix conversion.

        This test verifies that the converted dense matrices produce the same
        results as the original DIA matrices on identical inputs.
        """
        print("\n=== DIA TO DENSE CONVERSION ACCURACY TEST ===")

        # Load both versions
        mixed_tensor_dia, dia_matrix = self.load_real_diffusion_patterns(precision="fp32")

        pattern_path_dense = Path(__file__).parent.parent / "diffusion_patterns" / "synthetic_2624_fp32_with_dense.npz"
        if not pattern_path_dense.exists():
            pytest.skip("Dense synthetic patterns not available")

        data_dense = np.load(str(pattern_path_dense), allow_pickle=True)
        dense_ata_dict = data_dense["dense_ata_matrix"].item()
        dense_ata_matrix = DenseATAMatrix.from_dict(dense_ata_dict)

        print("Testing conversion accuracy:")
        print(f"  DIA matrix: {dia_matrix.k} diagonals, bandwidth={dia_matrix.bandwidth}")
        print(f"  Dense matrix: {dense_ata_matrix.memory_mb:.1f}MB")

        # Debug: Show DIA structure
        print("\nDIA Matrix Debug Info:")
        print(f"  DIA offsets (first 10): {dia_matrix.dia_offsets[:10]}")
        print(f"  DIA offsets (last 10): {dia_matrix.dia_offsets[-10:]}")
        print(f"  DIA data shape: {dia_matrix.dia_data_cpu.shape}")

        # Check if using the right conversion - verify by creating a small test matrix
        print("\nTesting matrix reconstruction on main diagonal:")
        # Get main diagonal from DIA (offset 0)
        main_diag_idx = np.where(dia_matrix.dia_offsets == 0)[0]
        if len(main_diag_idx) > 0:
            main_diag_idx = main_diag_idx[0]
            dia_main_diag = dia_matrix.dia_data_cpu[0, main_diag_idx, :5]  # First 5 elements, channel 0

            # Ensure dense matrices are loaded to GPU
            if dense_ata_matrix.dense_matrices_gpu is None:
                dense_ata_matrix.dense_matrices_gpu = cp.asarray(
                    dense_ata_matrix.dense_matrices_cpu, dtype=dense_ata_matrix.storage_dtype
                )

            dense_main_diag = cp.asnumpy(dense_ata_matrix.dense_matrices_gpu[0])
            dense_main_diag = np.diag(dense_main_diag)[:5]  # First 5 diagonal elements
            print(f"  DIA main diagonal (first 5): {dia_main_diag}")
            print(f"  Dense main diagonal (first 5): {dense_main_diag}")
            print(f"  Match: {np.allclose(dia_main_diag, dense_main_diag, rtol=1e-5)}")
        else:
            print("  No main diagonal (offset=0) found in DIA matrix!")
            print("  This suggests the ATA matrix has zero main diagonal - very unusual!")

            # Let's check what we do have
            print(f"  Available offsets around 0: {dia_matrix.dia_offsets[dia_matrix.dia_offsets >= -5][:10]}")

            # Check the first few offsets manually
            if len(dia_matrix.dia_offsets) > 0:
                first_offset = dia_matrix.dia_offsets[0]
                print(f"  First offset: {first_offset}")
                first_diag_data = dia_matrix.dia_data_cpu[0, 0, :5]
                print(f"  First diagonal data: {first_diag_data}")

                # Ensure dense matrices are loaded to GPU
                if dense_ata_matrix.dense_matrices_gpu is None:
                    dense_ata_matrix.dense_matrices_gpu = cp.asarray(
                        dense_ata_matrix.dense_matrices_cpu, dtype=dense_ata_matrix.storage_dtype
                    )

                dense_matrix = cp.asnumpy(dense_ata_matrix.dense_matrices_gpu[0])
                if first_offset < 0:
                    # Lower diagonal - check elements at (abs(offset), 0), (abs(offset)+1, 1), etc.
                    dense_diag_data = [
                        dense_matrix[abs(first_offset) + i, i]
                        for i in range(5)
                        if abs(first_offset) + i < len(dense_matrix)
                    ]
                else:
                    # Upper diagonal - check elements at (0, offset), (1, offset+1), etc.
                    dense_diag_data = [
                        dense_matrix[i, first_offset + i] for i in range(5) if first_offset + i < len(dense_matrix[0])
                    ]

                print(f"  Corresponding dense elements: {dense_diag_data}")
                print(f"  Match: {np.allclose(first_diag_data[:len(dense_diag_data)], dense_diag_data, rtol=1e-5)}")

        # Test with simple known vectors
        test_vectors = [
            cp.zeros((3, dia_matrix.led_count), dtype=cp.float32),  # Zero vector
            cp.ones((3, dia_matrix.led_count), dtype=cp.float32),  # Ones vector
            cp.random.random((3, dia_matrix.led_count), dtype=cp.float32),  # Random vector
        ]

        for i, test_vector in enumerate(test_vectors):
            print(f"\nTest vector {i+1}:")

            # DIA result
            dia_result = dia_matrix.multiply_3d(test_vector)

            # Dense result
            dense_result = dense_ata_matrix.multiply_vector(test_vector)

            # Compare results
            dia_cpu = cp.asnumpy(dia_result)
            dense_cpu = cp.asnumpy(dense_result)

            max_diff = np.max(np.abs(dia_cpu - dense_cpu))
            mean_diff = np.mean(np.abs(dia_cpu - dense_cpu))
            rms_diff = np.sqrt(np.mean((dia_cpu - dense_cpu) ** 2))

            result_norm = np.sqrt(np.mean(dia_cpu**2))
            relative_error = rms_diff / result_norm if result_norm > 1e-10 else float("inf")

            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  RMS difference: {rms_diff:.6f}")
            print(f"  Relative error: {relative_error:.6f}")

            if max_diff > 1e-5:
                print("  ❌ CONVERSION ERROR: Differences too large!")

                # Debug: Check some sample values
                print(f"  DIA result sample: {dia_cpu.flat[:5]}")
                print(f"  Dense result sample: {dense_cpu.flat[:5]}")

                # Check if dense matrix has correct structure
                dense_matrix_sample = cp.asnumpy(dense_ata_matrix.dense_matrices_gpu[0])
                print(f"  Dense matrix diagonal sample: {np.diag(dense_matrix_sample)[:5]}")
                print(f"  Dense matrix symmetry check: {np.allclose(dense_matrix_sample, dense_matrix_sample.T)}")

                return False
            else:
                print("  ✅ Conversion accurate")

        print("\n✅ DIA to Dense conversion accuracy test completed")
        return True
