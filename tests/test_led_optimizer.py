"""
Unit tests for the LED Optimization Engine.

Tests optimization algorithms, diffusion pattern handling, GPU acceleration,
and performance metrics without LED position dependencies.
"""

import os
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT

# Import current dense optimizer
from src.consumer.led_optimizer_dense import DenseLEDOptimizer, DenseOptimizationResult

# GPU availability check
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Create aliases for compatibility
LEDOptimizer = DenseLEDOptimizer
OptimizationResult = DenseOptimizationResult


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
        self.optimizer = LEDOptimizer(diffusion_patterns_path=self.patterns_path.replace(".npz", ""))

    def _create_test_patterns(self):
        """Create minimal test diffusion patterns in new nested format for testing."""
        from scipy import sparse

        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

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
        matrix = sparse.csc_matrix((data, (row_coords, col_coords)), shape=(pixels, cols))

        # Create LEDDiffusionCSCMatrix utility class
        diffusion_matrix = LEDDiffusionCSCMatrix(csc_matrix=matrix, height=FRAME_HEIGHT, width=FRAME_WIDTH, channels=3)

        # Create SingleBlockMixedSparseTensor utility class
        mixed_tensor = SingleBlockMixedSparseTensor(
            batch_size=test_led_count,
            channels=3,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            block_size=96,
        )

        # Create simple precomputed A^T@A matrices (identity-like for testing)
        ata_matrices = np.zeros((test_led_count, test_led_count, 3), dtype=np.float32)
        for i in range(test_led_count):
            ata_matrices[i, i, :] = 1.0  # Diagonal matrices for simple testing

        # Create dense_ata dictionary
        dense_ata_dict = {
            "dense_ata_matrices": ata_matrices,
            "dense_ata_led_count": test_led_count,
            "dense_ata_channels": 3,
            "dense_ata_computation_time": 0.1,
        }

        # Save in new nested format
        np.savez_compressed(
            self.patterns_path,
            # Top-level metadata
            led_spatial_mapping={i: i for i in range(test_led_count)},
            led_positions=np.random.random((test_led_count, 2)),
            metadata={
                "generator": "TestPatternGenerator",
                "format": "sparse_csc",
                "led_count": test_led_count,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "channels": 3,
                "matrix_shape": [pixels, cols],
                "nnz": nnz,
                "sparsity_percent": (1.0 - nnz / (pixels * cols)) * 100,
                "generation_timestamp": 0.0,
                "pattern_type": "test",
                "seed": None,
                "intensity_variation": False,
            },
            # Nested utility class data
            diffusion_matrix=diffusion_matrix.to_dict(),
            mixed_tensor=mixed_tensor.to_dict(),
            dense_ata=dense_ata_dict,
        )

    def tearDown(self):
        """Clean up after tests."""
        # Clean up temp files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_gpu_detection(self):
        """Test GPU detection and device info."""
        # Test default initialization (automatic GPU detection)
        optimizer = LEDOptimizer()
        self.assertIsNotNone(optimizer.device_info)
        self.assertIn("device", optimizer.device_info)

        # Device should be either 'gpu' or 'cpu'
        device_type = optimizer.device_info.get("device")
        self.assertIn(device_type, ["gpu", "cpu"])

    def test_initialization_success(self):
        """Test successful optimizer initialization with new nested format."""
        result = self.optimizer.initialize()

        self.assertTrue(result)
        self.assertTrue(self.optimizer._matrix_loaded)

    def test_initialization_no_patterns(self):
        """Test initialization failure without patterns."""
        optimizer = LEDOptimizer(diffusion_patterns_path="nonexistent_path")
        result = optimizer.initialize()

        self.assertFalse(result)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_diffusion_pattern_optimization(self):
        """Test diffusion pattern optimization with new nested format."""
        self.optimizer.initialize()

        # Create test frame
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(np.uint8)

        # Run optimization with debug mode to get error metrics
        result = self.optimizer.optimize_frame(test_frame, debug=True, max_iterations=5)

        # LED count will be from test patterns (50), not full LED_COUNT (3200)
        self.assertEqual(result.led_values.shape[1], 3)  # RGB channels
        self.assertGreater(result.led_values.shape[0], 0)  # Some LEDs
        self.assertGreater(result.iterations, 0)
        self.assertLessEqual(result.iterations, 5)
        self.assertIn("mse", result.error_metrics)

    def test_diffusion_pattern_loading_failure(self):
        """Test diffusion pattern loading failure when no patterns exist."""
        # Create optimizer with nonexistent patterns path
        optimizer = LEDOptimizer(diffusion_patterns_path="nonexistent_path")
        result = optimizer.initialize()
        self.assertFalse(result)
        self.assertFalse(optimizer._matrix_loaded)

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

        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(np.uint8)
        # Get actual LED count from test patterns
        test_led_count = 50  # matches our test pattern size
        initial_values = np.random.random((test_led_count, 3)).astype(np.float32) * 255

        result = self.optimizer.optimize_frame(test_frame, initial_values=initial_values, max_iterations=5)

        self.assertEqual(result.led_values.shape[1], 3)  # RGB channels
        self.assertGreater(result.led_values.shape[0], 0)  # Some LEDs
        self.assertLessEqual(result.iterations, 5)

    def test_optimizer_statistics(self):
        """Test optimizer statistics collection with new nested format."""
        self.optimizer.initialize()

        # Run a few optimizations
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(np.uint8)
        self.optimizer.optimize_frame(test_frame, max_iterations=3)
        self.optimizer.optimize_frame(test_frame, max_iterations=3)

        stats = self.optimizer.get_optimizer_stats()

        self.assertIn(stats["device"], ["gpu", "cpu"])  # Device type from stats
        self.assertTrue(stats["matrix_loaded"])
        self.assertEqual(stats["optimization_count"], 2)
        # Timing removed - verify fields exist but are 0
        self.assertEqual(stats["total_optimization_time"], 0.0)
        self.assertEqual(stats["estimated_fps"], 0.0)
        # LED count in stats should match our test patterns (50), not full LED_COUNT (3200)
        self.assertEqual(stats["led_count"], 50)  # Should match test pattern size

    def test_parameter_updates(self):
        """Test optimization parameter updates."""
        # Set parameters directly (DenseLEDOptimizer doesn't have set_optimization_parameters)
        self.optimizer.max_iterations = 200
        self.optimizer.convergence_threshold = 1e-8
        self.optimizer.step_size_scaling = 0.5

        self.assertEqual(self.optimizer.max_iterations, 200)
        self.assertEqual(self.optimizer.convergence_threshold, 1e-8)
        self.assertEqual(self.optimizer.step_size_scaling, 0.5)

    def test_sparse_diffusion_pattern_loading(self):
        """Test loading sparse diffusion patterns from new format fixture."""
        # Test loading from new format test fixture
        fixture_path = "diffusion_patterns/test_new_format"

        # Create optimizer with fixture patterns
        optimizer = LEDOptimizer(diffusion_patterns_path=fixture_path)

        # Should successfully initialize with new format fixture patterns
        result = optimizer.initialize()
        self.assertTrue(result, "Should successfully load sparse patterns from new format fixture")
        self.assertTrue(optimizer._matrix_loaded, "Matrix should be loaded")

        # Get stats to verify pattern details
        stats = optimizer.get_optimizer_stats()
        self.assertEqual(stats["led_count"], 100, "Should have 100 LEDs from fixture")
        self.assertIn("ata_tensor_shape", stats, "Should have ATA tensor shape info")
        # Dense optimizer provides ATA memory info instead of sparsity
        self.assertIn("ata_memory_mb", stats, "Should have ATA memory info")

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

    def test_error_handling_in_optimization(self):
        """Test error handling during optimization."""
        # Test with uninitialized optimizer
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(np.uint8)

        result = self.optimizer.optimize_frame(test_frame)

        # Should return error result
        self.assertEqual(result.get_total_error(), float("inf"))
        self.assertFalse(result.converged)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_performance_measurement(self):
        """Test that optimization time is measured correctly."""
        self.optimizer.initialize()

        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(np.uint8)

        result = self.optimizer.optimize_frame(test_frame, max_iterations=3)

        # Timing removed - verify field exists but is 0
        self.assertEqual(result.optimization_time, 0.0)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_different_iteration_counts(self):
        """Test optimization with different iteration counts."""
        self.optimizer.initialize()

        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3)).astype(np.uint8)

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
        self.optimizer = LEDOptimizer(diffusion_patterns_path=self.patterns_path.replace(".npz", ""))

    def _create_test_patterns(self):
        """Create minimal test diffusion patterns in new nested format for testing."""
        from scipy import sparse

        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

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
        matrix = sparse.csc_matrix((data, (row_coords, col_coords)), shape=(pixels, cols))

        # Create LEDDiffusionCSCMatrix utility class
        diffusion_matrix = LEDDiffusionCSCMatrix(csc_matrix=matrix, height=FRAME_HEIGHT, width=FRAME_WIDTH, channels=3)

        # Create SingleBlockMixedSparseTensor utility class
        mixed_tensor = SingleBlockMixedSparseTensor(
            batch_size=test_led_count,
            channels=3,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            block_size=96,
        )

        # Create simple precomputed A^T@A matrices (identity-like for testing)
        ata_matrices = np.zeros((test_led_count, test_led_count, 3), dtype=np.float32)
        for i in range(test_led_count):
            ata_matrices[i, i, :] = 1.0  # Diagonal matrices for simple testing

        # Create dense_ata dictionary
        dense_ata_dict = {
            "dense_ata_matrices": ata_matrices,
            "dense_ata_led_count": test_led_count,
            "dense_ata_channels": 3,
            "dense_ata_computation_time": 0.1,
        }

        # Save in new nested format
        np.savez_compressed(
            self.patterns_path,
            # Top-level metadata
            led_spatial_mapping={i: i for i in range(test_led_count)},
            led_positions=np.random.random((test_led_count, 2)),
            metadata={
                "generator": "TestPatternGenerator",
                "format": "sparse_csc",
                "led_count": test_led_count,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "channels": 3,
                "matrix_shape": [pixels, cols],
                "nnz": nnz,
                "sparsity_percent": (1.0 - nnz / (pixels * cols)) * 100,
                "generation_timestamp": 0.0,
                "pattern_type": "test",
                "seed": None,
                "intensity_variation": False,
            },
            # Nested utility class data
            diffusion_matrix=diffusion_matrix.to_dict(),
            mixed_tensor=mixed_tensor.to_dict(),
            dense_ata=dense_ata_dict,
        )

    def tearDown(self):
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization pipeline with new nested format."""
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
        # Timing removed - verify field exists but is 0
        self.assertEqual(result.optimization_time, 0.0)
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
        self.optimizer.max_iterations = 5

        # Run optimization multiple times with same initial conditions
        # Get actual LED count from test patterns
        test_led_count = 50  # matches our test pattern size
        initial_values = np.zeros((test_led_count, 3), dtype=np.float32)

        result1 = self.optimizer.optimize_frame(test_frame, initial_values=initial_values.copy())
        result2 = self.optimizer.optimize_frame(test_frame, initial_values=initial_values.copy())

        # Results should be very similar (allowing for minor numerical differences)
        diff = np.abs(result1.led_values - result2.led_values)
        max_diff = np.max(diff)
        self.assertLess(max_diff, 1.0)  # Should be very close


class TestATAInverseOptimization(unittest.TestCase):
    """Test cases for ATA inverse optimization performance comparison."""

    def setUp(self):
        """Set up test fixtures with and without ATA inverse."""
        self.temp_dir = tempfile.mkdtemp()

        # Create patterns WITH ATA inverse
        self.patterns_with_inverse_path = os.path.join(self.temp_dir, "patterns_with_inverse.npz")
        self._create_test_patterns_with_inverse()

        # Create patterns WITHOUT ATA inverse
        self.patterns_without_inverse_path = os.path.join(self.temp_dir, "patterns_without_inverse.npz")
        self._create_test_patterns_without_inverse()

    def _create_test_patterns_with_inverse(self):
        """Create test patterns with ATA inverse matrices."""
        from scipy import sparse

        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

        test_led_count = 2600  # Use realistic production LED count
        pixels = FRAME_HEIGHT * FRAME_WIDTH
        cols = test_led_count * 3  # R, G, B for each LED

        # Create random sparse matrix with realistic structure
        density = 0.01
        nnz = int(pixels * cols * density)

        # Create more realistic pattern structure
        np.random.seed(42)  # For reproducible tests
        row_coords = np.random.randint(0, pixels, nnz)
        col_coords = np.random.randint(0, cols, nnz)
        data = np.random.random(nnz).astype(np.float32) * 0.5 + 0.1  # Positive values

        matrix = sparse.csc_matrix((data, (row_coords, col_coords)), shape=(pixels, cols))
        diffusion_matrix = LEDDiffusionCSCMatrix(csc_matrix=matrix, height=FRAME_HEIGHT, width=FRAME_WIDTH, channels=3)

        mixed_tensor = SingleBlockMixedSparseTensor(
            batch_size=test_led_count,
            channels=3,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            block_size=96,
        )

        # Create realistic ATA matrices with some off-diagonal structure
        ata_matrices = np.zeros((3, test_led_count, test_led_count), dtype=np.float32)
        for c in range(3):
            # Create positive definite matrix
            A_channel = matrix[:, c::3]  # Extract channel matrix
            ata_channel = A_channel.T @ A_channel  # Compute actual ATA
            ata_matrices[c, :, :] = ata_channel.toarray().astype(np.float32)

            # Add small regularization for numerical stability
            ata_matrices[c, :, :] += 1e-4 * np.eye(test_led_count)

        # Compute ATA inverse matrices
        ata_inverse_matrices = np.zeros((3, test_led_count, test_led_count), dtype=np.float32)
        condition_numbers = []
        successful_inversions = 0

        for c in range(3):
            try:
                # Add small regularization for inversion
                regularization = 1e-6
                ata_regularized = ata_matrices[c, :, :] + regularization * np.eye(test_led_count)
                cond_num = np.linalg.cond(ata_regularized)
                condition_numbers.append(cond_num)

                ata_inverse_matrices[c, :, :] = np.linalg.inv(ata_regularized).astype(np.float32)
                successful_inversions += 1
            except np.linalg.LinAlgError:
                ata_inverse_matrices[c, :, :] = np.eye(test_led_count, dtype=np.float32)
                condition_numbers.append(float("inf"))

        # Create dense_ata dictionary with inverse
        dense_ata_dict = {
            "dense_ata_matrices": ata_matrices,
            "dense_ata_inverse_matrices": ata_inverse_matrices,
            "dense_ata_led_count": test_led_count,
            "dense_ata_channels": 3,
            "dense_ata_computation_time": 0.1,
            "condition_numbers": condition_numbers,
            "avg_condition_number": np.mean([cn for cn in condition_numbers if cn != float("inf")]),
            "successful_inversions": successful_inversions,
            "inversion_successful": [True] * successful_inversions + [False] * (3 - successful_inversions),
        }

        # Save patterns with ATA inverse
        np.savez_compressed(
            self.patterns_with_inverse_path,
            led_spatial_mapping={i: i for i in range(test_led_count)},
            led_positions=np.random.random((test_led_count, 2)),
            metadata={
                "generator": "TestPatternGenerator",
                "format": "sparse_csc_with_ata_inverse",
                "led_count": test_led_count,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "channels": 3,
            },
            diffusion_matrix=diffusion_matrix.to_dict(),
            mixed_tensor=mixed_tensor.to_dict(),
            dense_ata=dense_ata_dict,
        )

    def _create_test_patterns_without_inverse(self):
        """Create test patterns without ATA inverse matrices (same base patterns)."""
        from scipy import sparse

        from src.utils.led_diffusion_csc_matrix import LEDDiffusionCSCMatrix
        from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

        test_led_count = 2600
        pixels = FRAME_HEIGHT * FRAME_WIDTH
        cols = test_led_count * 3

        # Use same seed for identical base patterns
        np.random.seed(42)
        density = 0.01
        nnz = int(pixels * cols * density)
        row_coords = np.random.randint(0, pixels, nnz)
        col_coords = np.random.randint(0, cols, nnz)
        data = np.random.random(nnz).astype(np.float32) * 0.5 + 0.1

        matrix = sparse.csc_matrix((data, (row_coords, col_coords)), shape=(pixels, cols))
        diffusion_matrix = LEDDiffusionCSCMatrix(csc_matrix=matrix, height=FRAME_HEIGHT, width=FRAME_WIDTH, channels=3)

        mixed_tensor = SingleBlockMixedSparseTensor(
            batch_size=test_led_count,
            channels=3,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            block_size=96,
        )

        # Create ATA matrices but NO inverse
        ata_matrices = np.zeros((3, test_led_count, test_led_count), dtype=np.float32)
        for c in range(3):
            A_channel = matrix[:, c::3]
            ata_channel = A_channel.T @ A_channel
            ata_matrices[c, :, :] = ata_channel.toarray().astype(np.float32)
            ata_matrices[c, :, :] += 1e-4 * np.eye(test_led_count)

        # Create dense_ata dictionary WITHOUT inverse
        dense_ata_dict = {
            "dense_ata_matrices": ata_matrices,
            "dense_ata_led_count": test_led_count,
            "dense_ata_channels": 3,
            "dense_ata_computation_time": 0.1,
            # No inverse matrices included
        }

        # Save patterns without ATA inverse
        np.savez_compressed(
            self.patterns_without_inverse_path,
            led_spatial_mapping={i: i for i in range(test_led_count)},
            led_positions=np.random.random((test_led_count, 2)),
            metadata={
                "generator": "TestPatternGenerator",
                "format": "sparse_csc_without_ata_inverse",
                "led_count": test_led_count,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "channels": 3,
            },
            diffusion_matrix=diffusion_matrix.to_dict(),
            mixed_tensor=mixed_tensor.to_dict(),
            dense_ata=dense_ata_dict,
        )

    def tearDown(self):
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_ata_inverse_performance_comparison(self):
        """Compare optimization performance with and without ATA inverse initialization."""
        print("\n" + "=" * 100)
        print("ATA INVERSE PERFORMANCE COMPARISON TEST (2600 LEDs)")
        print("=" * 100)

        # Create test frame with realistic content
        test_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        test_frame[100:200, 200:300] = [255, 100, 50]  # Orange-ish region
        test_frame[300:400, 400:500] = [50, 255, 100]  # Green-ish region
        test_frame[200:300, 600:700] = [100, 50, 255]  # Purple-ish region

        # Test parameters
        max_iterations = 50
        num_warmup_runs = 2  # Warm-up runs not included in timing
        num_test_runs = 5  # Actual measurement runs

        # Initialize optimizers
        print("Initializing optimizers...")
        optimizer_with_inverse = LEDOptimizer(
            diffusion_patterns_path=self.patterns_with_inverse_path.replace(".npz", "")
        )
        optimizer_without_inverse = LEDOptimizer(
            diffusion_patterns_path=self.patterns_without_inverse_path.replace(".npz", "")
        )

        self.assertTrue(optimizer_with_inverse.initialize())
        self.assertTrue(optimizer_without_inverse.initialize())
        self.assertTrue(optimizer_with_inverse._has_ata_inverse, "Should have ATA inverse available")
        self.assertFalse(
            optimizer_without_inverse._has_ata_inverse,
            "Should NOT have ATA inverse available",
        )

        print(
            f"✓ Optimizer WITH inverse: {optimizer_with_inverse._actual_led_count} LEDs, "
            f"has_inverse={optimizer_with_inverse._has_ata_inverse}"
        )
        print(
            f"✓ Optimizer WITHOUT inverse: {optimizer_without_inverse._actual_led_count} LEDs, "
            f"has_inverse={optimizer_without_inverse._has_ata_inverse}"
        )

        # WARM-UP RUNS (not included in measurements)
        print(f"\nPerforming {num_warmup_runs} warm-up runs (not included in timing)...")
        for i in range(num_warmup_runs):
            optimizer_with_inverse.optimize_frame(test_frame, max_iterations=5, debug=False)
            optimizer_without_inverse.optimize_frame(test_frame, max_iterations=5, debug=False)
            print(f"  Warm-up {i + 1}/{num_warmup_runs} completed")

        # Test WITH ATA inverse
        print(f"\nTesting optimization WITH ATA inverse ({num_test_runs} measurement runs)...")
        with_inverse_results = []
        with_inverse_timings = []

        for run in range(num_test_runs):
            # Enable detailed timing for this optimizer
            if optimizer_with_inverse.timing:
                optimizer_with_inverse.timing.reset()

            start_time = time.perf_counter()
            result_with = optimizer_with_inverse.optimize_frame(test_frame, max_iterations=max_iterations, debug=True)
            end_time = time.perf_counter()

            total_time = (end_time - start_time) * 1000  # Convert to ms

            with_inverse_results.append(result_with)
            with_inverse_timings.append(total_time)

            print(
                f"  Run {run + 1}: {result_with.iterations} iterations, "
                f"MSE: {result_with.error_metrics['mse']:.6f}, Time: {total_time:.2f}ms"
            )

        # Test WITHOUT ATA inverse
        print(f"\nTesting optimization WITHOUT ATA inverse ({num_test_runs} measurement runs)...")
        without_inverse_results = []
        without_inverse_timings = []

        for run in range(num_test_runs):
            # Enable detailed timing for this optimizer
            if optimizer_without_inverse.timing:
                optimizer_without_inverse.timing.reset()

            start_time = time.perf_counter()
            result_without = optimizer_without_inverse.optimize_frame(
                test_frame, max_iterations=max_iterations, debug=True
            )
            end_time = time.perf_counter()

            total_time = (end_time - start_time) * 1000  # Convert to ms

            without_inverse_results.append(result_without)
            without_inverse_timings.append(total_time)

            print(
                f"  Run {run + 1}: {result_without.iterations} iterations, "
                f"MSE: {result_without.error_metrics['mse']:.6f}, Time: {total_time:.2f}ms"
            )

        # Calculate statistics
        avg_iterations_with = np.mean([r.iterations for r in with_inverse_results])
        avg_iterations_without = np.mean([r.iterations for r in without_inverse_results])
        std_iterations_with = np.std([r.iterations for r in with_inverse_results])
        std_iterations_without = np.std([r.iterations for r in without_inverse_results])

        avg_mse_with = np.mean([r.error_metrics["mse"] for r in with_inverse_results])
        avg_mse_without = np.mean([r.error_metrics["mse"] for r in without_inverse_results])
        std_mse_with = np.std([r.error_metrics["mse"] for r in with_inverse_results])
        std_mse_without = np.std([r.error_metrics["mse"] for r in without_inverse_results])

        avg_time_with = np.mean(with_inverse_timings)
        avg_time_without = np.mean(without_inverse_timings)
        std_time_with = np.std(with_inverse_timings)
        std_time_without = np.std(without_inverse_timings)

        # Calculate improvements
        iteration_reduction = avg_iterations_without - avg_iterations_with
        iteration_reduction_percent = (
            (iteration_reduction / avg_iterations_without) * 100 if avg_iterations_without > 0 else 0
        )
        mse_improvement = avg_mse_without - avg_mse_with
        mse_improvement_percent = (mse_improvement / avg_mse_without) * 100 if avg_mse_without > 0 else 0
        time_difference = avg_time_with - avg_time_without

        # Detailed timing breakdown (if available)
        timing_breakdown_with = None
        timing_breakdown_without = None

        if optimizer_with_inverse.timing and hasattr(optimizer_with_inverse.timing, "get_summary"):
            timing_breakdown_with = optimizer_with_inverse.timing.get_summary()
        if optimizer_without_inverse.timing and hasattr(optimizer_without_inverse.timing, "get_summary"):
            timing_breakdown_without = optimizer_without_inverse.timing.get_summary()

        # Print comprehensive results
        print("\n" + "=" * 100)
        print("COMPREHENSIVE PERFORMANCE COMPARISON RESULTS")
        print("=" * 100)
        print(f"LED Count: {optimizer_with_inverse._actual_led_count}")
        print(f"Frame Size: {FRAME_HEIGHT}x{FRAME_WIDTH}")
        print(f"Test Runs: {num_test_runs} (after {num_warmup_runs} warm-up runs)")
        print()

        print("ITERATION COMPARISON:")
        print(f"  WITH ATA inverse:    {avg_iterations_with:.1f} ± {std_iterations_with:.1f} iterations")
        print(f"  WITHOUT ATA inverse: {avg_iterations_without:.1f} ± {std_iterations_without:.1f} iterations")
        print(f"  Reduction:           {iteration_reduction:.1f} iterations ({iteration_reduction_percent:+.1f}%)")
        print()

        print("MSE COMPARISON:")
        print(f"  WITH ATA inverse:    {avg_mse_with:.6f} ± {std_mse_with:.6f}")
        print(f"  WITHOUT ATA inverse: {avg_mse_without:.6f} ± {std_mse_without:.6f}")
        print(f"  Improvement:         {mse_improvement:+.6f} ({mse_improvement_percent:+.1f}%)")
        print()

        print("TIMING COMPARISON:")
        print(f"  WITH ATA inverse:    {avg_time_with:.2f} ± {std_time_with:.2f} ms")
        print(f"  WITHOUT ATA inverse: {avg_time_without:.2f} ± {std_time_without:.2f} ms")
        print(f"  Difference:          {time_difference:+.2f} ms")
        print()

        # Detailed timing breakdown
        if timing_breakdown_with or timing_breakdown_without:
            print("DETAILED TIMING BREAKDOWN:")
            if timing_breakdown_with:
                print("  WITH ATA inverse:")
                for operation, time_ms in timing_breakdown_with.items():
                    print(f"    {operation:25}: {time_ms:.3f} ms")
            if timing_breakdown_without:
                print("  WITHOUT ATA inverse:")
                for operation, time_ms in timing_breakdown_without.items():
                    print(f"    {operation:25}: {time_ms:.3f} ms")
            print()

        # Performance metrics
        if avg_iterations_without > 0:
            convergence_efficiency = (avg_iterations_without - avg_iterations_with) / avg_iterations_without * 100
            print(f"CONVERGENCE EFFICIENCY GAIN: {convergence_efficiency:+.1f}%")

        if avg_time_without > 0:
            time_efficiency = (avg_time_without - avg_time_with) / avg_time_without * 100
            print(f"TIME EFFICIENCY CHANGE: {time_efficiency:+.1f}%")

        print("=" * 100)

        # Assertions
        self.assertGreater(avg_iterations_with, 0, "Should have non-zero iterations with inverse")
        self.assertGreater(avg_iterations_without, 0, "Should have non-zero iterations without inverse")

        # ATA inverse should not significantly increase iterations
        self.assertLessEqual(
            avg_iterations_with,
            avg_iterations_without + 3,
            f"ATA inverse should not significantly increase iterations: "
            f"{avg_iterations_with:.1f} vs {avg_iterations_without:.1f}",
        )

        # Results should be valid
        for result in with_inverse_results + without_inverse_results:
            self.assertGreater(result.iterations, 0)
            self.assertTrue(np.all(result.led_values >= 0))
            self.assertTrue(np.all(result.led_values <= 255))
            self.assertIn("mse", result.error_metrics)

        # Report key findings
        if iteration_reduction >= 1:
            print(
                f"🎯 KEY RESULT: ATA inverse reduced iterations by {iteration_reduction:.1f} "
                f"({iteration_reduction_percent:.1f}%)"
            )
        else:
            print(f"📊 KEY RESULT: Similar convergence rate (difference: {iteration_reduction:.1f} iterations)")

        if mse_improvement > 0:
            print(f"✨ QUALITY GAIN: {mse_improvement_percent:.1f}% better final MSE with ATA inverse")

        print("=" * 100)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available")
    def test_ata_inverse_initialization_timing(self):
        """Test and measure the timing of ATA inverse initialization with 2600 LEDs."""
        print("\n" + "=" * 80)
        print("ATA INVERSE INITIALIZATION TIMING TEST (2600 LEDs)")
        print("=" * 80)

        # Create optimizer with ATA inverse
        optimizer = LEDOptimizer(diffusion_patterns_path=self.patterns_with_inverse_path.replace(".npz", ""))
        self.assertTrue(optimizer.initialize())
        self.assertTrue(optimizer._has_ata_inverse)

        print(f"✓ Optimizer initialized with {optimizer._actual_led_count} LEDs")
        print(f"✓ ATA inverse available: {optimizer._has_ata_inverse}")

        # Create test frame with realistic content
        test_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        test_frame[100:200, 200:300] = [255, 100, 50]  # Orange-ish region
        test_frame[300:400, 400:500] = [50, 255, 100]  # Green-ish region
        test_frame[200:300, 600:700] = [100, 50, 255]  # Purple-ish region

        # Test parameters
        num_warmup_runs = 2  # Warm-up runs not included in timing
        num_measurement_runs = 10  # Actual measurement runs
        max_iterations = 5  # Limited iterations to focus on initialization

        # WARM-UP RUNS (not included in measurements)
        print(f"\nPerforming {num_warmup_runs} warm-up runs (not included in timing)...")
        for i in range(num_warmup_runs):
            optimizer.optimize_frame(test_frame, max_iterations=max_iterations, debug=False)
            print(f"  Warm-up {i + 1}/{num_warmup_runs} completed")

        # MEASUREMENT RUNS
        print(f"\nMeasuring ATA inverse initialization and optimization over {num_measurement_runs} runs...")
        init_times = []
        iteration_counts = []
        mse_values = []

        for i in range(num_measurement_runs):
            # Enable detailed timing for this run
            if optimizer.timing:
                optimizer.timing.reset()

            start_time = time.perf_counter()
            result = optimizer.optimize_frame(test_frame, max_iterations=max_iterations, debug=True)
            end_time = time.perf_counter()

            total_time = (end_time - start_time) * 1000  # Convert to ms
            init_times.append(total_time)
            iteration_counts.append(result.iterations)
            mse_values.append(result.error_metrics["mse"])

            print(
                f"  Run {i + 1}: {result.iterations} iterations, "
                f"MSE: {result.error_metrics['mse']:.6f}, Time: {total_time:.2f}ms"
            )

        # Calculate comprehensive statistics
        avg_time = np.mean(init_times)
        std_time = np.std(init_times)
        min_time = np.min(init_times)
        max_time = np.max(init_times)

        avg_iterations = np.mean(iteration_counts)
        std_iterations = np.std(iteration_counts)
        avg_mse = np.mean(mse_values)
        std_mse = np.std(mse_values)

        # Detailed timing breakdown (if available)
        timing_breakdown = None
        if optimizer.timing and hasattr(optimizer.timing, "get_summary"):
            timing_breakdown = optimizer.timing.get_summary()

        # Print comprehensive results
        print("\n" + "=" * 80)
        print("DETAILED TIMING RESULTS")
        print("=" * 80)
        print(f"LED Count: {optimizer._actual_led_count}")
        print(f"Frame Size: {FRAME_HEIGHT}x{FRAME_WIDTH}")
        print(f"Max Iterations per Run: {max_iterations}")
        print()
        print("PERFORMANCE STATISTICS:")
        print(f"  Total Time:     {avg_time:.2f} ± {std_time:.2f} ms (range: {min_time:.2f} - {max_time:.2f})")
        print(f"  Iterations:     {avg_iterations:.1f} ± {std_iterations:.1f}")
        print(f"  Final MSE:      {avg_mse:.6f} ± {std_mse:.6f}")
        print(f"  Time per LED:   {(avg_time / optimizer._actual_led_count):.4f} ms/LED")

        if timing_breakdown:
            print("\nDETAILED TIMING BREAKDOWN:")
            for component, timing_data in timing_breakdown.items():
                if isinstance(timing_data, dict) and "total_time" in timing_data:
                    print(f"  {component:20s}: {timing_data['total_time']:.2f} ms")

        # Verify the results are valid
        self.assertGreater(avg_iterations, 0)
        self.assertTrue(
            all(
                np.all(result.led_values >= 0)
                for result in [optimizer.optimize_frame(test_frame, max_iterations=1, debug=False)]
            )
        )
        self.assertTrue(
            all(
                np.all(result.led_values <= 255)
                for result in [optimizer.optimize_frame(test_frame, max_iterations=1, debug=False)]
            )
        )

        # Performance assertions for 2600 LEDs
        # The initialization should be reasonably fast even for 2600 LEDs
        self.assertLess(
            avg_time,
            1000.0,
            f"ATA inverse optimization should complete in reasonable time for 2600 LEDs, got {avg_time:.2f} ms",
        )

        print("\n✓ ATA inverse initialization and optimization completed successfully")
        print(f"✓ Average performance: {avg_time:.2f} ms for {optimizer._actual_led_count} LEDs")
        print("=" * 80)


if __name__ == "__main__":
    unittest.main()
