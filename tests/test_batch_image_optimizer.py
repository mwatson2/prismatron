#!/usr/bin/env python3
"""
Unit tests for BatchImageOptimizer tool.

Tests the batch image processing functionality including:
- Pattern data loading and validation
- Image loading, resizing, and format conversion
- LED optimization pipeline
- Image reconstruction from LED values
- Batch processing workflows
- Error handling and edge cases
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Add tools to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from batch_image_optimizer import BatchImageOptimizer


class TestBatchImageOptimizer:
    """Test suite for BatchImageOptimizer class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test directories
        self.source_dir = self.temp_path / "source"
        self.output_dir = self.temp_path / "output"
        self.source_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        # Create mock pattern file
        self.pattern_file = self.temp_path / "test_patterns.npz"

        # Test parameters
        self.led_count = 100
        self.target_width = 800
        self.target_height = 480
        self.channels = 3

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_pattern_file(self):
        """Create a mock pattern file for testing."""
        # Create mock mixed tensor
        mixed_tensor_dict = {
            "batch_size": self.led_count,
            "channels": self.channels,
            "frame_width": self.target_width,
            "frame_height": self.target_height,
            "indices_cpu": np.random.randint(0, self.led_count, size=(self.led_count,)),
            "data_cpu": np.random.randn(self.led_count, self.channels, self.target_height, self.target_width).astype(
                np.float32
            ),
        }

        # Create mock DIA matrix
        dia_dict = {
            "led_count": self.led_count,
            "crop_size": 64,
            "channels": self.channels,
            "bandwidth": 10,
            "sparsity": 0.1,
            "nnz": 1000,
            "k": 21,
            "dia_data_cpu": np.random.randn(self.channels, 21, self.led_count).astype(np.float32),
            "dia_offsets": np.arange(-10, 11),
        }

        # Create mock ATA inverse
        ata_inverse = np.random.randn(self.channels, self.led_count, self.led_count).astype(np.float32)

        # Create mock LED positions
        led_positions = np.random.randn(self.led_count, 3).astype(np.float32)

        # Save pattern file
        np.savez(
            self.pattern_file,
            mixed_tensor=mixed_tensor_dict,
            dia_matrix=dia_dict,
            ata_inverse=ata_inverse,
            led_positions=led_positions,
        )

    def create_test_image(self, filename: str, width: int = 100, height: int = 100):
        """Create a test image file."""
        # Create a simple test image (gradient)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                image[i, j] = [i % 256, j % 256, (i + j) % 256]

        image_path = self.source_dir / filename
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return image_path, image

    def test_initialization(self):
        """Test BatchImageOptimizer initialization."""
        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file),
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir),
            target_width=self.target_width,
            target_height=self.target_height,
        )

        assert optimizer.pattern_file == str(self.pattern_file)
        assert optimizer.source_dir == self.source_dir
        assert optimizer.output_dir == self.output_dir
        assert optimizer.target_width == self.target_width
        assert optimizer.target_height == self.target_height

        # Check initial state
        assert optimizer.at_matrix is None
        assert optimizer.ata_matrix is None
        assert optimizer.ata_inverse is None
        assert optimizer.led_positions is None
        assert optimizer.total_images == 0
        assert optimizer.processed_images == 0
        assert optimizer.failed_images == 0

    @patch("batch_image_optimizer.SingleBlockMixedSparseTensor")
    @patch("batch_image_optimizer.DiagonalATAMatrix")
    @patch("batch_image_optimizer.load_ata_inverse_from_pattern")
    def test_load_pattern_data_success(self, mock_load_inverse, mock_dia_matrix, mock_mixed_tensor):
        """Test successful pattern data loading."""
        self.create_mock_pattern_file()

        # Setup mocks
        mock_mixed_tensor.from_dict.return_value = MagicMock()
        mock_mixed_tensor.from_dict.return_value.batch_size = self.led_count
        mock_mixed_tensor.from_dict.return_value.channels = self.channels

        mock_dia_matrix.from_dict.return_value = MagicMock()
        mock_dia_matrix.from_dict.return_value.led_count = self.led_count
        mock_dia_matrix.from_dict.return_value.bandwidth = 10

        mock_load_inverse.return_value = np.random.randn(self.channels, self.led_count, self.led_count).astype(
            np.float32
        )

        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )

        # Test loading
        success = optimizer.load_pattern_data()
        assert success is True

        # Check that data was loaded
        assert optimizer.at_matrix is not None
        assert optimizer.ata_matrix is not None
        assert optimizer.ata_inverse is not None
        assert optimizer.led_positions is not None

    def test_load_pattern_data_missing_file(self):
        """Test pattern data loading with missing file."""
        optimizer = BatchImageOptimizer(
            pattern_file="/nonexistent/file.npz", source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )

        success = optimizer.load_pattern_data()
        assert success is False

    def test_find_source_images(self):
        """Test finding source images in directory."""
        # Create test images with various extensions
        test_files = [
            "test1.jpg",
            "test2.png",
            "test3.JPEG",
            "test4.bmp",
            "not_image.txt",  # Should be ignored
            "test5.tiff",
        ]

        for filename in test_files:
            if filename.endswith(".txt"):
                # Create text file
                (self.source_dir / filename).write_text("not an image")
            else:
                # Create dummy image
                self.create_test_image(filename)

        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )

        image_files = optimizer.find_source_images()

        # Should find 5 image files (excluding .txt)
        assert len(image_files) == 5

        # Should be sorted
        filenames = [f.name for f in image_files]
        assert filenames == sorted(filenames)

        # Should not include text file
        assert "not_image.txt" not in filenames

    def test_find_source_images_empty_directory(self):
        """Test finding images in empty directory."""
        empty_dir = self.temp_path / "empty"
        empty_dir.mkdir()

        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(empty_dir), output_dir=str(self.output_dir)
        )

        image_files = optimizer.find_source_images()
        assert len(image_files) == 0

    def test_load_and_resize_image(self):
        """Test image loading and resizing."""
        # Create test image
        original_width, original_height = 50, 30
        image_path, original_image = self.create_test_image("test.jpg", original_width, original_height)

        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file),
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir),
            target_width=100,
            target_height=80,
        )

        loaded_image = optimizer.load_and_resize_image(image_path)

        # Check that image was loaded and resized
        assert loaded_image is not None
        assert loaded_image.shape == (80, 100, 3)  # (height, width, channels)
        assert loaded_image.dtype == np.uint8

    def test_load_and_resize_image_no_resize_needed(self):
        """Test loading image that doesn't need resizing."""
        # Create image with target dimensions
        image_path, original_image = self.create_test_image("test.jpg", self.target_width, self.target_height)

        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file),
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir),
            target_width=self.target_width,
            target_height=self.target_height,
        )

        loaded_image = optimizer.load_and_resize_image(image_path)

        assert loaded_image is not None
        assert loaded_image.shape == (self.target_height, self.target_width, 3)

    def test_load_and_resize_image_invalid_file(self):
        """Test loading invalid image file."""
        # Create non-image file
        invalid_file = self.source_dir / "invalid.jpg"
        invalid_file.write_text("not an image")

        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )

        loaded_image = optimizer.load_and_resize_image(invalid_file)
        assert loaded_image is None

    @patch("batch_image_optimizer.optimize_frame_led_values")
    def test_optimize_image_success(self, mock_optimize):
        """Test successful image optimization."""
        # Setup mock optimization result
        mock_result = MagicMock()
        mock_result.led_values = np.random.randn(3, self.led_count).astype(np.float32)
        mock_result.iterations = 5
        mock_result.converged = True
        mock_result.error_metrics = {"mse": 0.1, "psnr": 25.0}
        mock_optimize.return_value = mock_result

        # Create test image
        test_image = np.random.randint(0, 256, (self.target_height, self.target_width, 3), dtype=np.uint8)

        # Create optimizer with mock data
        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )
        optimizer.at_matrix = MagicMock()
        optimizer.ata_matrix = MagicMock()
        optimizer.ata_inverse = np.random.randn(3, self.led_count, self.led_count).astype(np.float32)

        result = optimizer.optimize_image(test_image, max_iterations=5, debug=True)

        assert result is not None
        assert result.led_values.shape == (3, self.led_count)
        assert result.iterations == 5
        assert result.converged is True

        # Check that optimize_frame_led_values was called with correct parameters
        mock_optimize.assert_called_once()

    @patch("batch_image_optimizer.optimize_frame_led_values")
    def test_optimize_image_failure(self, mock_optimize):
        """Test image optimization failure."""
        # Make optimization fail
        mock_optimize.side_effect = Exception("Optimization failed")

        test_image = np.random.randint(0, 256, (self.target_height, self.target_width, 3), dtype=np.uint8)

        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )
        optimizer.at_matrix = MagicMock()
        optimizer.ata_matrix = MagicMock()
        optimizer.ata_inverse = np.random.randn(3, self.led_count, self.led_count).astype(np.float32)

        result = optimizer.optimize_image(test_image)
        assert result is None

    @patch("cupy.asarray")
    def test_led_values_to_image(self, mock_cupy_asarray):
        """Test converting LED values back to image."""
        if not CUPY_AVAILABLE:
            pytest.skip("CuPy not available")

        # Create mock AT matrix
        mock_at_matrix = MagicMock()
        mock_reconstructed = np.random.randint(0, 256, (3, self.target_height, self.target_width)).astype(np.float32)
        mock_at_matrix.forward_pass_3d.return_value.get.return_value = mock_reconstructed

        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file),
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir),
            target_width=self.target_width,
            target_height=self.target_height,
        )
        optimizer.at_matrix = mock_at_matrix
        optimizer.at_matrix.batch_size = self.led_count

        # Test LED values
        led_values = np.random.randn(3, self.led_count).astype(np.float32)

        result_image = optimizer.led_values_to_image(led_values)

        assert result_image.shape == (self.target_height, self.target_width, 3)
        assert result_image.dtype == np.uint8

    def test_led_values_to_image_failure(self):
        """Test LED values to image conversion failure."""
        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file),
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir),
            target_width=self.target_width,
            target_height=self.target_height,
        )
        optimizer.at_matrix = None  # This will cause failure

        led_values = np.random.randn(3, self.led_count).astype(np.float32)

        result_image = optimizer.led_values_to_image(led_values)

        # Should return black image as fallback
        expected_shape = (self.target_height, self.target_width, 3)
        assert result_image.shape == expected_shape
        assert result_image.dtype == np.uint8
        assert np.all(result_image == 0)  # Should be black

    def test_save_optimized_image(self):
        """Test saving optimized image."""
        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )

        # Create test image
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        output_path = self.output_dir / "test_output.jpg"

        success = optimizer.save_optimized_image(test_image, output_path)

        assert success is True
        assert output_path.exists()

        # Verify saved image can be loaded
        loaded_image = cv2.imread(str(output_path))
        assert loaded_image is not None
        assert loaded_image.shape == (100, 100, 3)

    @patch.object(BatchImageOptimizer, "load_and_resize_image")
    @patch.object(BatchImageOptimizer, "optimize_image")
    @patch.object(BatchImageOptimizer, "led_values_to_image")
    @patch.object(BatchImageOptimizer, "save_optimized_image")
    def test_process_image_success(self, mock_save, mock_led_to_image, mock_optimize, mock_load):
        """Test successful single image processing."""
        # Setup mocks
        mock_load.return_value = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        mock_result = MagicMock()
        mock_result.led_values = np.random.randn(3, self.led_count).astype(np.float32)
        mock_result.iterations = 3
        mock_result.converged = True
        mock_result.error_metrics = {"mse": 0.1}
        mock_optimize.return_value = mock_result

        mock_led_to_image.return_value = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mock_save.return_value = True

        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )

        # Create test image file
        test_image_path = self.source_dir / "test.jpg"
        test_image_path.touch()

        success = optimizer.process_image(test_image_path, debug=True)

        assert success is True
        mock_load.assert_called_once_with(test_image_path)
        mock_optimize.assert_called_once()
        mock_led_to_image.assert_called_once()
        mock_save.assert_called_once()

    @patch.object(BatchImageOptimizer, "load_and_resize_image")
    def test_process_image_load_failure(self, mock_load):
        """Test image processing with load failure."""
        mock_load.return_value = None  # Simulate load failure

        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )

        test_image_path = self.source_dir / "test.jpg"
        test_image_path.touch()

        success = optimizer.process_image(test_image_path)
        assert success is False

    def test_process_image_skip_existing(self):
        """Test skipping existing optimized images."""
        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )

        # Create source and output files
        source_file = self.source_dir / "test.jpg"
        source_file.touch()
        output_file = self.output_dir / "test_optimized.jpg"
        output_file.touch()

        # Should skip without overwrite
        success = optimizer.process_image(source_file, overwrite=False)
        assert success is True  # Skipping is considered success

    @patch.object(BatchImageOptimizer, "process_image")
    def test_process_all_images(self, mock_process):
        """Test processing all images in directory."""
        # Setup mock to simulate mixed success/failure
        mock_process.side_effect = [True, False, True]  # 2 success, 1 failure

        # Create test images
        for i in range(3):
            self.create_test_image(f"test{i}.jpg")

        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )

        optimizer.process_all_images(debug=True)

        # Check statistics
        assert optimizer.total_images == 3
        assert optimizer.processed_images == 2
        assert optimizer.failed_images == 1

        # Should have called process_image for each file
        assert mock_process.call_count == 3

    def test_process_all_images_no_images(self):
        """Test processing with no images found."""
        optimizer = BatchImageOptimizer(
            pattern_file=str(self.pattern_file), source_dir=str(self.source_dir), output_dir=str(self.output_dir)
        )

        # Should complete without errors
        optimizer.process_all_images()

        assert optimizer.total_images == 0
        assert optimizer.processed_images == 0
        assert optimizer.failed_images == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
