"""
Unit tests for the LED Optimization Engine.

Tests optimization algorithms, diffusion pattern handling, GPU acceleration,
and performance metrics without LED position dependencies.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT

# Add archive directory to path for sparse optimizer
archive_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "archive",
)
sys.path.insert(0, archive_path)
from led_optimizer_sparse import CUPY_AVAILABLE, LEDOptimizer, OptimizationResult


class TestOptimizationResult(unittest.TestCase):
    """Test cases for OptimizationResult class."""

    def test_initialization(self):
        """Test optimization result initialization."""
        led_values = np.random.random((LED_COUNT, 3)).astype(np.float32) * 255
        error_metrics = {"mse": 10.5, "mae": 5.2}

        result = OptimizationResult(
            led_values=led_values,
            error_metrics=error_metrics,
            optimization_time=0.123,
            iterations=50,
            converged=True,
        )

        self.assertEqual(result.get_led_count(), LED_COUNT)
        self.assertEqual(result.get_total_error(), 10.5)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 50)

    def test_error_handling(self):
        """Test error result handling."""
        result = OptimizationResult(
            led_values=np.zeros((LED_COUNT, 3)),
            error_metrics={"mse": float("inf")},
            optimization_time=0.0,
            iterations=0,
            converged=False,
        )

        self.assertEqual(result.get_total_error(), float("inf"))
        self.assertFalse(result.converged)


class TestLEDOptimizer(unittest.TestCase):
    """Test cases for LEDOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test patterns file for optimizer to use
        self.patterns_path = os.path.join(self.temp_dir, "test_patterns.npz")
        self._create_test_patterns()

        # Create optimizer with test patterns
        self.optimizer = LEDOptimizer(
            diffusion_patterns_path=self.patterns_path.replace(".npz", ""),
            use_gpu=False,  # Force CPU for testing
        )

    def _create_test_patterns(self):
        """Create minimal test diffusion patterns for testing."""
        # Create sparse test patterns that match the new format
        from scipy import sparse

        test_led_count = 50  # Use small number for fast tests
        pixels = FRAME_HEIGHT * FRAME_WIDTH
        cols = test_led_count * 3  # R, G, B for each LED

        # Create random sparse matrix with ~1% density
        density = 0.01
        nnz = int(pixels * cols * density)

        # Random coordinates
        row_coords = np.random.randint(0, pixels, nnz)
        col_coords = np.random.randint(0, cols, nnz)
        data = np.random.random(nnz).astype(np.float32)

        # Create sparse matrix
        matrix = sparse.csc_matrix(
            (data, (row_coords, col_coords)), shape=(pixels, cols)
        )

        # Save in the expected sparse format
        np.savez_compressed(
            self.patterns_path,
            matrix_data=matrix.data,
            matrix_indices=matrix.indices,
            matrix_indptr=matrix.indptr,
            matrix_shape=matrix.shape,
            led_spatial_mapping={i: i for i in range(test_led_count)},
            led_positions=np.random.random((test_led_count, 2)),
        )

    def tearDown(self):
        """Clean up after tests."""
        # Clean up temp files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_gpu_detection(self):
        """Test GPU usage detection logic."""
        # Test explicit GPU setting
        optimizer_cpu = LEDOptimizer(use_gpu=False)
        self.assertFalse(optimizer_cpu.use_gpu)

        # Test automatic detection (will depend on system)
        optimizer_auto = LEDOptimizer(use_gpu=True)
        # Should fall back to CPU if no GPU available, which is fine

    def test_initialization_success(self):
        """Test successful optimizer initialization."""
        result = self.optimizer.initialize()

        self.assertTrue(result)
        self.assertTrue(self.optimizer._matrix_loaded)

    def test_initialization_no_patterns(self):
        """Test initialization failure without patterns."""
        optimizer = LEDOptimizer(diffusion_patterns_path="nonexistent_path")
        result = optimizer.initialize()

        self.assertFalse(result)

    def test_diffusion_pattern_loading_failure(self):
        """Test diffusion pattern loading failure when no patterns exist."""
        # Create optimizer with nonexistent patterns path
        optimizer = LEDOptimizer(diffusion_patterns_path="nonexistent_path")
        result = optimizer.initialize()
        self.assertFalse(result)
        self.assertFalse(optimizer._matrix_loaded)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_diffusion_pattern_optimization(self):
        """Test diffusion pattern optimization."""
        self.optimizer.initialize()

        # Create test frame
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )

        # Run optimization
        result = self.optimizer.optimize_frame(test_frame, max_iterations=5)

        # LED count will be from test patterns (50), not full LED_COUNT (3200)
        self.assertEqual(result.led_values.shape[1], 3)  # RGB channels
        self.assertGreater(result.led_values.shape[0], 0)  # Some LEDs
        self.assertGreater(result.iterations, 0)
        self.assertLessEqual(result.iterations, 5)
        self.assertIn("mse", result.error_metrics)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_full_optimization(self):
        """Test full optimization with diffusion patterns."""
        self.optimizer.initialize()

        # Create simple test frame (solid color)
        test_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 128, dtype=np.uint8)

        # Run optimization
        result = self.optimizer.optimize_frame(test_frame, max_iterations=10)

        self.assertEqual(result.led_values.shape[1], 3)  # RGB channels
        self.assertGreater(result.led_values.shape[0], 0)  # Some LEDs
        self.assertGreater(result.iterations, 0)
        self.assertLess(result.optimization_time, 10.0)  # Should be fast
        self.assertIn("mse", result.error_metrics)

    def test_optimization_with_invalid_frame(self):
        """Test optimization with invalid frame dimensions."""
        if not CUPY_AVAILABLE:
            self.skipTest("PyTorch not available")

        self.optimizer.initialize()

        # Create frame with wrong dimensions
        wrong_frame = np.zeros((100, 200, 3), dtype=np.uint8)

        result = self.optimizer.optimize_frame(wrong_frame)

        # Should return error result
        self.assertEqual(result.get_total_error(), float("inf"))
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_optimization_with_initial_values(self):
        """Test optimization with initial LED values."""
        self.optimizer.initialize()

        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )
        # Get actual LED count from test patterns
        test_led_count = 50  # matches our test pattern size
        initial_values = np.random.random((test_led_count, 3)).astype(np.float32) * 255

        result = self.optimizer.optimize_frame(
            test_frame, initial_values=initial_values, max_iterations=5
        )

        self.assertEqual(result.led_values.shape[1], 3)  # RGB channels
        self.assertGreater(result.led_values.shape[0], 0)  # Some LEDs
        self.assertLessEqual(result.iterations, 5)

    def test_parameter_updates(self):
        """Test optimization parameter updates."""
        self.optimizer.set_optimization_parameters(
            max_iterations=200,
            convergence_threshold=1e-8,
            step_size_scaling=0.5,
        )

        self.assertEqual(self.optimizer.max_iterations, 200)
        self.assertEqual(self.optimizer.convergence_threshold, 1e-8)
        self.assertEqual(self.optimizer.step_size_scaling, 0.5)

    def test_optimizer_statistics(self):
        """Test optimizer statistics collection."""
        self.optimizer.initialize()

        # Run a few optimizations
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )
        self.optimizer.optimize_frame(test_frame, max_iterations=3)
        self.optimizer.optimize_frame(test_frame, max_iterations=3)

        stats = self.optimizer.get_optimizer_stats()

        self.assertEqual(stats["device_info"]["device"], "cpu")
        self.assertTrue(stats["matrix_loaded"])
        self.assertEqual(stats["optimization_count"], 2)
        self.assertGreater(stats["total_optimization_time"], 0)
        self.assertGreater(stats["estimated_fps"], 0)
        self.assertIn("parameters", stats)
        # LED count in stats should match our test patterns (50), not full LED_COUNT (3200)
        self.assertEqual(stats["led_count"], 50)  # Should match test pattern size

    def test_sparse_diffusion_pattern_loading(self):
        """Test loading sparse diffusion patterns from fixture."""
        # Test loading from our regression test fixture
        fixture_path = "tests/fixtures/test_clean"

        # Create optimizer with fixture patterns
        optimizer = LEDOptimizer(diffusion_patterns_path=fixture_path, use_gpu=False)

        # Should successfully initialize with fixture patterns
        result = optimizer.initialize()
        self.assertTrue(result, "Should successfully load sparse patterns from fixture")
        self.assertTrue(optimizer._matrix_loaded, "Matrix should be loaded")

        # Get stats to verify pattern details
        stats = optimizer.get_optimizer_stats()
        self.assertEqual(stats["led_count"], 100, "Should have 100 LEDs from fixture")
        self.assertIn("matrix_shape", stats, "Should have matrix shape info")
        self.assertIn("sparsity_percent", stats, "Should have sparsity info")
        self.assertGreater(
            stats["sparsity_percent"], 0, "Should be sparse (>0% sparsity)"
        )
        self.assertLess(stats["sparsity_percent"], 100, "Should not be 100% sparse")

    def test_load_nonexistent_diffusion_patterns(self):
        """Test loading non-existent diffusion patterns."""
        nonexistent_path = "/tmp/nonexistent_patterns"
        optimizer = LEDOptimizer(diffusion_patterns_path=nonexistent_path)

        # Should fail without patterns file
        result = optimizer.initialize()
        self.assertFalse(result)
        self.assertFalse(optimizer._matrix_loaded)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_led_value_clamping(self):
        """Test that LED values are clamped to valid range."""
        self.optimizer.initialize()

        # Create frame that might produce out-of-range values
        bright_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 255, dtype=np.uint8)

        result = self.optimizer.optimize_frame(bright_frame, max_iterations=5)

        # All LED values should be in [0, 255] range
        self.assertTrue(np.all(result.led_values >= 0))
        self.assertTrue(np.all(result.led_values <= 255))

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_convergence_detection(self):
        """Test optimization convergence detection."""
        self.optimizer.convergence_threshold = 1e-3  # Relaxed threshold
        self.optimizer.initialize()

        # Create simple uniform frame
        uniform_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 100, dtype=np.uint8)

        result = self.optimizer.optimize_frame(uniform_frame, max_iterations=100)

        # Should always converge (LSQR always converges)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.iterations, 100)

    def test_error_handling_in_optimization(self):
        """Test error handling during optimization."""
        # Test with uninitialized optimizer
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )

        result = self.optimizer.optimize_frame(test_frame)

        # Should return error result
        self.assertEqual(result.get_total_error(), float("inf"))
        self.assertFalse(result.converged)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_performance_measurement(self):
        """Test that optimization time is measured correctly."""
        self.optimizer.initialize()

        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )

        result = self.optimizer.optimize_frame(test_frame, max_iterations=3)

        self.assertGreater(result.optimization_time, 0)
        self.assertLess(result.optimization_time, 5.0)  # Should be reasonably fast

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_different_iteration_counts(self):
        """Test optimization with different iteration counts."""
        self.optimizer.initialize()

        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )

        # Test few iterations
        result1 = self.optimizer.optimize_frame(test_frame, max_iterations=1)

        # Test more iterations
        result2 = self.optimizer.optimize_frame(test_frame, max_iterations=10)

        # Both should produce valid results
        self.assertEqual(result1.led_values.shape[1], 3)  # RGB channels
        self.assertEqual(result2.led_values.shape[1], 3)  # RGB channels
        self.assertGreater(result1.led_values.shape[0], 0)  # Some LEDs
        self.assertGreater(result2.led_values.shape[0], 0)  # Some LEDs

        # More iterations should generally produce better results
        self.assertLessEqual(result1.iterations, 1)
        self.assertLessEqual(result2.iterations, 10)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_pattern_weighted_sum(self):
        """Test that optimization produces reasonable weighted sum."""
        self.optimizer.initialize()

        # Create a simple test: black frame should result in near-zero LED values
        black_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        result = self.optimizer.optimize_frame(black_frame, max_iterations=20)

        # LED values should be close to zero for black target
        avg_led_value = np.mean(result.led_values)
        self.assertLess(avg_led_value, 50)  # Should be much less than mid-range


class TestOptimizerIntegration(unittest.TestCase):
    """Integration tests for LED optimizer."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test patterns file for optimizer to use
        self.patterns_path = os.path.join(self.temp_dir, "test_patterns.npz")
        self._create_test_patterns()

        # Create optimizer with test patterns
        self.optimizer = LEDOptimizer(
            diffusion_patterns_path=self.patterns_path.replace(".npz", ""),
            use_gpu=False,
        )

    def _create_test_patterns(self):
        """Create minimal test diffusion patterns for testing."""
        # Create sparse test patterns that match the new format
        from scipy import sparse

        test_led_count = 50  # Use small number for fast tests
        pixels = FRAME_HEIGHT * FRAME_WIDTH
        cols = test_led_count * 3  # R, G, B for each LED

        # Create random sparse matrix with ~1% density
        density = 0.01
        nnz = int(pixels * cols * density)

        # Random coordinates
        row_coords = np.random.randint(0, pixels, nnz)
        col_coords = np.random.randint(0, cols, nnz)
        data = np.random.random(nnz).astype(np.float32)

        # Create sparse matrix
        matrix = sparse.csc_matrix(
            (data, (row_coords, col_coords)), shape=(pixels, cols)
        )

        # Save in the expected sparse format
        np.savez_compressed(
            self.patterns_path,
            matrix_data=matrix.data,
            matrix_indices=matrix.indices,
            matrix_indptr=matrix.indptr,
            matrix_shape=matrix.shape,
            led_spatial_mapping={i: i for i in range(test_led_count)},
            led_positions=np.random.random((test_led_count, 2)),
        )

    def tearDown(self):
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization pipeline."""
        # Initialize optimizer
        self.assertTrue(self.optimizer.initialize())

        # Create test content
        test_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # Add some patterns to the frame
        test_frame[100:200, 200:300] = [255, 0, 0]  # Red square
        test_frame[300:400, 400:500] = [0, 255, 0]  # Green square
        test_frame[200:300, 600:700] = [0, 0, 255]  # Blue square

        # Run optimization
        result = self.optimizer.optimize_frame(test_frame, max_iterations=20)

        # Verify results
        self.assertEqual(result.led_values.shape[1], 3)  # RGB channels
        self.assertGreater(result.led_values.shape[0], 0)  # Some LEDs
        self.assertGreater(result.iterations, 0)
        self.assertTrue(result.optimization_time > 0)
        self.assertIsNotNone(result.error_metrics)

        # LED values should be reasonable
        self.assertTrue(np.all(result.led_values >= 0))
        self.assertTrue(np.all(result.led_values <= 255))

    def test_consistent_results(self):
        """Test that optimization produces consistent results."""
        if not CUPY_AVAILABLE:
            self.skipTest("PyTorch not available")

        self.optimizer.initialize()

        # Create deterministic test frame
        test_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 128, dtype=np.uint8)

        # Set deterministic parameters
        self.optimizer.set_optimization_parameters(max_iterations=5)

        # Run optimization multiple times with same initial conditions
        # Get actual LED count from test patterns
        test_led_count = 50  # matches our test pattern size
        initial_values = np.zeros((test_led_count, 3), dtype=np.float32)

        result1 = self.optimizer.optimize_frame(
            test_frame, initial_values=initial_values.copy()
        )
        result2 = self.optimizer.optimize_frame(
            test_frame, initial_values=initial_values.copy()
        )

        # Results should be very similar (allowing for minor numerical differences)
        diff = np.abs(result1.led_values - result2.led_values)
        max_diff = np.max(diff)
        self.assertLess(max_diff, 1.0)  # Should be very close


if __name__ == "__main__":
    unittest.main()
