"""
Unit tests for the LED Optimization Engine.

Tests optimization algorithms, diffusion pattern handling, GPU acceleration,
and performance metrics.
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
from src.consumer.led_mapper import LEDMapper
from src.consumer.led_optimizer import TORCH_AVAILABLE, LEDOptimizer, OptimizationResult


class TestOptimizationResult(unittest.TestCase):
    """Test cases for OptimizationResult class."""

    def test_initialization(self):
        """Test optimization result initialization."""
        led_values = np.random.randint(0, 255, (LED_COUNT, 3)).astype(np.float32)
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

        # Create LED mapper
        self.led_mapper = LEDMapper()
        self.led_mapper.generate_random_positions(seed=42)

        # Create optimizer
        self.optimizer = LEDOptimizer(
            led_mapper=self.led_mapper, device="cpu"  # Force CPU for testing
        )

    def tearDown(self):
        """Clean up after tests."""
        # Clean up temp files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_device_detection(self):
        """Test device detection logic."""
        # Test explicit device setting
        optimizer_cpu = LEDOptimizer(self.led_mapper, device="cpu")
        self.assertEqual(optimizer_cpu.device, "cpu")

        # Test automatic detection (will depend on system)
        optimizer_auto = LEDOptimizer(self.led_mapper)
        self.assertIn(optimizer_auto.device, ["cpu", "cuda"])

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_initialization_success(self):
        """Test successful optimizer initialization."""
        result = self.optimizer.initialize()

        self.assertTrue(result)
        self.assertIsNotNone(self.optimizer._led_positions_tensor)
        self.assertIsNotNone(self.optimizer._diffusion_matrix)
        self.assertTrue(self.optimizer._diffusion_patterns_loaded)

    def test_initialization_no_torch(self):
        """Test initialization failure without PyTorch."""
        with patch("src.consumer.led_optimizer.TORCH_AVAILABLE", False):
            optimizer = LEDOptimizer(self.led_mapper)
            result = optimizer.initialize()

            self.assertFalse(result)

    def test_initialization_no_led_positions(self):
        """Test initialization failure without LED positions."""
        empty_mapper = LEDMapper()
        optimizer = LEDOptimizer(empty_mapper)

        result = optimizer.initialize()
        self.assertFalse(result)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_mock_diffusion_pattern_generation(self):
        """Test mock diffusion pattern generation."""
        self.optimizer.initialize()

        self.assertTrue(self.optimizer._diffusion_patterns_loaded)
        self.assertIsNotNone(self.optimizer._diffusion_matrix)

        # Check dimensions
        num_pixels = FRAME_HEIGHT * FRAME_WIDTH
        expected_shape = (num_pixels, LED_COUNT, 3)
        self.assertEqual(self.optimizer._diffusion_matrix.shape, expected_shape)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_frame_sampling_optimization(self):
        """Test simple frame sampling optimization."""
        self.optimizer.initialize()

        # Create test frame
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )

        # Run sampling optimization
        result = self.optimizer.sample_and_optimize(test_frame)

        self.assertEqual(result.led_values.shape, (LED_COUNT, 3))
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(
            result.error_metrics["mse"], 0.0
        )  # No error for direct sampling

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_full_optimization(self):
        """Test full optimization with diffusion patterns."""
        self.optimizer.initialize()

        # Create simple test frame (solid color)
        test_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 128, dtype=np.uint8)

        # Run optimization
        result = self.optimizer.optimize_frame(test_frame, max_iterations=10)

        self.assertEqual(result.led_values.shape, (LED_COUNT, 3))
        self.assertGreater(result.iterations, 0)
        self.assertLess(result.optimization_time, 10.0)  # Should be fast
        self.assertIn("mse", result.error_metrics)

    def test_optimization_with_invalid_frame(self):
        """Test optimization with invalid frame dimensions."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        self.optimizer.initialize()

        # Create frame with wrong dimensions
        wrong_frame = np.zeros((100, 200, 3), dtype=np.uint8)

        result = self.optimizer.optimize_frame(wrong_frame)

        # Should return error result
        self.assertEqual(result.get_total_error(), float("inf"))
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_optimization_with_initial_values(self):
        """Test optimization with initial LED values."""
        self.optimizer.initialize()

        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )
        initial_values = np.random.randint(0, 255, (LED_COUNT, 3)).astype(np.float32)

        result = self.optimizer.optimize_frame(
            test_frame, initial_values=initial_values, max_iterations=5
        )

        self.assertEqual(result.led_values.shape, (LED_COUNT, 3))
        self.assertLessEqual(result.iterations, 5)

    def test_parameter_updates(self):
        """Test optimization parameter updates."""
        self.optimizer.set_optimization_parameters(
            max_iterations=200,
            learning_rate=0.02,
            convergence_threshold=1e-8,
            regularization_weight=0.002,
        )

        self.assertEqual(self.optimizer.max_iterations, 200)
        self.assertEqual(self.optimizer.learning_rate, 0.02)
        self.assertEqual(self.optimizer.convergence_threshold, 1e-8)
        self.assertEqual(self.optimizer.regularization_weight, 0.002)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_optimizer_statistics(self):
        """Test optimizer statistics collection."""
        self.optimizer.initialize()

        # Run a few optimizations
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )
        self.optimizer.sample_and_optimize(test_frame)
        self.optimizer.sample_and_optimize(test_frame)

        stats = self.optimizer.get_optimizer_stats()

        self.assertEqual(stats["device"], "cpu")
        self.assertTrue(stats["torch_available"])
        self.assertTrue(stats["diffusion_patterns_loaded"])
        self.assertEqual(stats["optimization_count"], 2)
        self.assertGreater(stats["total_optimization_time"], 0)
        self.assertGreater(stats["estimated_fps"], 0)
        self.assertIn("parameters", stats)
        self.assertEqual(stats["led_count"], LED_COUNT)

    def test_diffusion_pattern_saving(self):
        """Test saving diffusion patterns."""
        save_path = os.path.join(self.temp_dir, "test_patterns.npz")
        optimizer = LEDOptimizer(self.led_mapper, diffusion_patterns_path=save_path)

        # Create mock diffusion patterns (use smaller size for testing)
        test_led_count = 10  # Use smaller count for memory efficiency
        patterns = np.random.random(
            (test_led_count, FRAME_HEIGHT, FRAME_WIDTH, 3)
        ).astype(np.float32)
        metadata = {"version": "test", "description": "Test patterns"}

        result = optimizer.save_diffusion_patterns(patterns, metadata)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(save_path))

        # Verify saved data
        loaded_data = np.load(save_path)
        self.assertIn("diffusion_patterns", loaded_data)
        self.assertEqual(loaded_data["diffusion_patterns"].shape, patterns.shape)
        self.assertEqual(
            loaded_data["led_count"], LED_COUNT
        )  # This is still LED_COUNT from const

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_diffusion_pattern_loading(self):
        """Test loading diffusion patterns from file."""
        save_path = os.path.join(self.temp_dir, "test_patterns.npz")

        # Create and save test patterns
        patterns = np.random.random((LED_COUNT, FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.float32
        )
        save_data = {
            "diffusion_patterns": patterns,
            "metadata": {},
            "led_count": LED_COUNT,
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
        }
        np.savez_compressed(save_path, **save_data)

        # Create optimizer and load patterns
        optimizer = LEDOptimizer(self.led_mapper, diffusion_patterns_path=save_path)
        result = optimizer.initialize()

        self.assertTrue(result)
        self.assertTrue(optimizer._diffusion_patterns_loaded)

    def test_load_nonexistent_diffusion_patterns(self):
        """Test loading non-existent diffusion patterns."""
        nonexistent_path = "/tmp/nonexistent_patterns.npz"
        optimizer = LEDOptimizer(
            self.led_mapper, diffusion_patterns_path=nonexistent_path
        )

        if TORCH_AVAILABLE:
            # Should fall back to mock patterns
            result = optimizer.initialize()
            self.assertTrue(result)
            self.assertTrue(optimizer._diffusion_patterns_loaded)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_led_value_clamping(self):
        """Test that LED values are clamped to valid range."""
        self.optimizer.initialize()

        # Create frame that might produce out-of-range values
        bright_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 255, dtype=np.uint8)

        result = self.optimizer.optimize_frame(bright_frame, max_iterations=5)

        # All LED values should be in [0, 255] range
        self.assertTrue(np.all(result.led_values >= 0))
        self.assertTrue(np.all(result.led_values <= 255))

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_convergence_detection(self):
        """Test optimization convergence detection."""
        self.optimizer.convergence_threshold = 1e-3  # Relaxed threshold
        self.optimizer.initialize()

        # Create simple uniform frame that should converge quickly
        uniform_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 100, dtype=np.uint8)

        result = self.optimizer.optimize_frame(uniform_frame, max_iterations=100)

        # Should converge before max iterations for simple case
        self.assertLess(result.iterations, 100)

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

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_performance_measurement(self):
        """Test that optimization time is measured correctly."""
        self.optimizer.initialize()

        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )

        result = self.optimizer.sample_and_optimize(test_frame)

        self.assertGreater(result.optimization_time, 0)
        self.assertLess(result.optimization_time, 1.0)  # Should be fast for sampling

    def test_sampling_error_handling(self):
        """Test error handling in frame sampling."""
        # Test with wrong frame shape
        wrong_frame = np.zeros((50, 100, 3), dtype=np.uint8)

        result = self.optimizer.sample_and_optimize(wrong_frame)

        # Should return error result
        self.assertEqual(result.get_total_error(), float("inf"))
        self.assertFalse(result.converged)

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_different_optimization_methods(self):
        """Test different optimization approaches."""
        self.optimizer.initialize()

        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(
            np.uint8
        )

        # Test sampling method
        result1 = self.optimizer.sample_and_optimize(test_frame)

        # Test full optimization
        result2 = self.optimizer.optimize_frame(test_frame, max_iterations=5)

        # Both should produce valid results
        self.assertEqual(result1.led_values.shape, (LED_COUNT, 3))
        self.assertEqual(result2.led_values.shape, (LED_COUNT, 3))

        # Sampling should be faster and have no error
        self.assertLess(result1.optimization_time, result2.optimization_time)
        self.assertEqual(result1.error_metrics["mse"], 0.0)


class TestOptimizerIntegration(unittest.TestCase):
    """Integration tests for LED optimizer with mapper."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.led_mapper = LEDMapper()
        self.led_mapper.generate_random_positions(seed=123)

        self.optimizer = LEDOptimizer(self.led_mapper, device="cpu")

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization pipeline."""
        # Initialize components
        self.assertTrue(self.led_mapper.validate_positions())
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
        self.assertEqual(result.led_values.shape, (LED_COUNT, 3))
        self.assertGreater(result.iterations, 0)
        self.assertTrue(result.optimization_time > 0)
        self.assertIsNotNone(result.error_metrics)

        # LED values should be reasonable
        self.assertTrue(np.all(result.led_values >= 0))
        self.assertTrue(np.all(result.led_values <= 255))

    def test_consistent_results(self):
        """Test that optimization produces consistent results."""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        self.optimizer.initialize()

        # Create deterministic test frame
        test_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 128, dtype=np.uint8)

        # Run optimization multiple times
        result1 = self.optimizer.sample_and_optimize(test_frame)
        result2 = self.optimizer.sample_and_optimize(test_frame)

        # Results should be identical for sampling
        np.testing.assert_array_equal(result1.led_values, result2.led_values)


if __name__ == "__main__":
    unittest.main()
