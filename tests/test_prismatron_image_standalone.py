"""
Standalone unit tests for PrismatronImage class.

This test file directly imports PrismatronImage to avoid CuPy import issues
from other utils modules.
"""

import base64
import io
import tempfile
import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Direct import to avoid utils/__init__.py CuPy issues
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "utils"))
import prismatron_image

# Import what we need for testing
PrismatronImage = prismatron_image.PrismatronImage
PIL_AVAILABLE = prismatron_image.PIL_AVAILABLE
OPENCV_AVAILABLE = prismatron_image.OPENCV_AVAILABLE
PILBackend = prismatron_image.PILBackend
OpenCVBackend = prismatron_image.OpenCVBackend
BasicBackend = prismatron_image.BasicBackend
CameraBackend = prismatron_image.CameraBackend
PRISMATRON_WIDTH = prismatron_image.PRISMATRON_WIDTH
PRISMATRON_HEIGHT = prismatron_image.PRISMATRON_HEIGHT


class TestPrismatronImageBasic:
    """Test basic PrismatronImage functionality."""
    
    def test_initialization_planar(self):
        """Test initialization with planar format."""
        data = np.random.randint(0, 256, (3, 100, 80), dtype=np.uint8)
        img = PrismatronImage(data, "planar")
        
        assert img.width == 100
        assert img.height == 80
        assert img.shape == (3, 100, 80)
        assert img.dtype == np.uint8
        assert img.size == 8000
    
    def test_initialization_interleaved(self):
        """Test initialization with interleaved format."""
        data = np.random.randint(0, 256, (100, 80, 3), dtype=np.uint8)
        img = PrismatronImage(data, "interleaved")
        
        assert img.width == 100
        assert img.height == 80
        assert img.shape == (3, 100, 80)
    
    def test_initialization_flat_interleaved(self):
        """Test initialization with flat interleaved format."""
        # Use Prismatron dimensions for predictable reconstruction
        pixels = PRISMATRON_WIDTH * PRISMATRON_HEIGHT
        data = np.random.randint(0, 256, pixels * 3, dtype=np.uint8)
        img = PrismatronImage(data, "flat_interleaved")
        
        assert img.width == PRISMATRON_WIDTH
        assert img.height == PRISMATRON_HEIGHT
        assert img.shape == (3, PRISMATRON_WIDTH, PRISMATRON_HEIGHT)
    
    def test_initialization_flat_spatial(self):
        """Test initialization with flat spatial format."""
        pixels = 100 * 80
        data = np.random.randint(0, 256, (pixels, 3), dtype=np.uint8)
        img = PrismatronImage(data, "flat_spatial")
        
        # Should guess reasonable dimensions
        assert img.width * img.height == pixels
        assert img.shape[0] == 3
    
    def test_initialization_flat_planar(self):
        """Test initialization with flat planar format."""
        pixels = 100 * 80
        data = np.random.randint(0, 256, (3, pixels), dtype=np.uint8)
        img = PrismatronImage(data, "flat_planar")
        
        assert img.width * img.height == pixels
        assert img.shape[0] == 3
    
    def test_validation_errors(self):
        """Test validation catches invalid data."""
        # Wrong number of channels
        with pytest.raises(ValueError, match="Invalid 3D shape"):
            PrismatronImage(np.zeros((100, 80, 4), dtype=np.uint8))
        
        # Wrong data type gets converted
        data = np.random.rand(3, 100, 80)  # float64
        img = PrismatronImage(data, "planar")
        assert img.dtype == np.uint8
        
        # Invalid dimensions
        with pytest.raises(ValueError, match="Invalid dimensions"):
            PrismatronImage(np.zeros((3, 0, 10), dtype=np.uint8))


class TestFormatConversions:
    """Test format conversion methods."""
    
    def setup_method(self):
        """Setup test image."""
        self.test_data = np.random.randint(0, 256, (3, 50, 40), dtype=np.uint8)
        self.img = PrismatronImage(self.test_data, "planar")
    
    def test_as_planar(self):
        """Test planar format output."""
        planar = self.img.as_planar()
        assert planar.shape == (3, 50, 40)
        assert np.array_equal(planar, self.test_data)
        # Should be a copy
        assert planar is not self.img._data
    
    def test_as_interleaved(self):
        """Test interleaved format output."""
        interleaved = self.img.as_interleaved()
        assert interleaved.shape == (50, 40, 3)
        
        # Check conversion correctness
        for i in range(3):
            assert np.array_equal(interleaved[:, :, i], self.test_data[i, :, :])
    
    def test_as_flat_interleaved(self):
        """Test flat interleaved format."""
        flat = self.img.as_flat_interleaved()
        assert flat.shape == (50 * 40 * 3,)
        assert flat.dtype == np.uint8
    
    def test_as_flat_spatial(self):
        """Test flat spatial format."""
        flat_spatial = self.img.as_flat_spatial()
        assert flat_spatial.shape == (50 * 40, 3)
        assert flat_spatial.dtype == np.uint8
    
    def test_as_flat_planar(self):
        """Test flat planar format."""
        flat_planar = self.img.as_flat_planar()
        assert flat_planar.shape == (3, 50 * 40)
        assert flat_planar.dtype == np.uint8
    
    def test_as_normalized_float(self):
        """Test normalized float format."""
        norm = self.img.as_normalized_float()
        assert norm.shape == (50, 40, 3)
        assert norm.dtype == np.float32
        assert np.all(norm >= 0.0) and np.all(norm <= 1.0)
    
    def test_as_normalized_planar_float(self):
        """Test normalized planar float format."""
        norm = self.img.as_normalized_planar_float()
        assert norm.shape == (3, 50, 40)
        assert norm.dtype == np.float32
        assert np.all(norm >= 0.0) and np.all(norm <= 1.0)
    
    def test_round_trip_conversions(self):
        """Test round-trip format conversions preserve data."""
        # Planar -> Interleaved -> Planar
        interleaved = self.img.as_interleaved()
        img2 = PrismatronImage.from_array(interleaved, "interleaved")
        assert np.array_equal(self.img.as_planar(), img2.as_planar())
        
        # Test with flat formats
        flat_spatial = self.img.as_flat_spatial()
        img3 = PrismatronImage(flat_spatial, "flat_spatial")
        assert img3.width * img3.height == self.img.size


class TestFactoryMethods:
    """Test factory methods."""
    
    def test_from_array(self):
        """Test from_array factory method."""
        data = np.random.randint(0, 256, (100, 80, 3), dtype=np.uint8)
        img = PrismatronImage.from_array(data, "interleaved")
        
        assert img.width == 100
        assert img.height == 80
        assert img.shape == (3, 100, 80)
    
    def test_zeros(self):
        """Test zeros factory method."""
        img = PrismatronImage.zeros(50, 30)
        
        assert img.width == 50
        assert img.height == 30
        assert np.all(img.as_planar() == 0)
    
    def test_ones(self):
        """Test ones factory method."""
        img = PrismatronImage.ones(50, 30)
        
        assert img.width == 50
        assert img.height == 30
        assert np.all(img.as_planar() == 255)
    
    def test_solid_color(self):
        """Test solid color factory method."""
        img = PrismatronImage.solid_color(50, 30, (128, 64, 192))
        
        assert img.width == 50
        assert img.height == 30
        
        planar = img.as_planar()
        assert np.all(planar[0] == 128)  # R
        assert np.all(planar[1] == 64)   # G
        assert np.all(planar[2] == 192)  # B


class TestImageOperations:
    """Test image manipulation operations."""
    
    def setup_method(self):
        """Setup test image."""
        # Create a simple test pattern
        data = np.zeros((3, 100, 80), dtype=np.uint8)
        data[0, 25:75, 20:60] = 255  # Red rectangle
        data[1, 10:90, 10:70] = 128  # Green background
        self.img = PrismatronImage(data, "planar")
    
    def test_resize_nearest(self):
        """Test resize with nearest neighbor."""
        resized = self.img.resize(50, 40, "nearest")
        
        assert resized.width == 50
        assert resized.height == 40
        assert isinstance(resized, PrismatronImage)
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_resize_bilinear(self):
        """Test resize with bilinear interpolation."""
        resized = self.img.resize(200, 160, "bilinear")
        
        assert resized.width == 200
        assert resized.height == 160
    
    def test_crop(self):
        """Test cropping operation."""
        cropped = self.img.crop(10, 5, 30, 20)
        
        assert cropped.width == 30
        assert cropped.height == 20
        
        # Check that crop bounds are validated
        with pytest.raises(ValueError, match="exceeds image bounds"):
            self.img.crop(90, 70, 20, 20)  # Would go beyond image
    
    def test_center_crop(self):
        """Test center cropping."""
        center_cropped = self.img.center_crop(50, 40)
        
        assert center_cropped.width == 50
        assert center_cropped.height == 40
        
        # Should fail if crop size larger than image
        with pytest.raises(ValueError, match="larger than image"):
            self.img.center_crop(200, 200)
    
    def test_thumbnail(self):
        """Test thumbnail generation."""
        thumb = self.img.thumbnail(50)
        
        # Should preserve aspect ratio
        assert max(thumb.width, thumb.height) <= 50
        aspect_ratio = thumb.width / thumb.height
        original_ratio = self.img.width / self.img.height
        assert abs(aspect_ratio - original_ratio) < 0.01
    
    def test_bounding_box(self):
        """Test bounding box detection."""
        bbox = self.img.bounding_box()
        
        assert bbox is not None
        x, y, w, h = bbox
        assert isinstance(x, int) and isinstance(y, int)
        assert isinstance(w, int) and isinstance(h, int)
        assert x >= 0 and y >= 0
        assert x + w <= self.img.width
        assert y + h <= self.img.height
    
    def test_bounding_box_all_zeros(self):
        """Test bounding box with all-zero image."""
        zero_img = PrismatronImage.zeros(50, 30)
        bbox = zero_img.bounding_box()
        assert bbox is None
    
    def test_crop_to_content(self):
        """Test cropping to content bounding box."""
        content_cropped = self.img.crop_to_content()
        
        # Should be smaller than original (since there are zero regions)
        assert content_cropped.size <= self.img.size
        
        # Test with all-zero image
        zero_img = PrismatronImage.zeros(50, 30)
        zero_cropped = zero_img.crop_to_content()
        assert zero_cropped.width == 1 and zero_cropped.height == 1


class TestQualityMetrics:
    """Test quality comparison metrics."""
    
    def setup_method(self):
        """Setup test images."""
        # Create identical images for perfect match tests
        self.img1 = PrismatronImage.zeros(50, 40)
        self.img2 = PrismatronImage.zeros(50, 40)
        
        # Create slightly different image
        data = np.zeros((3, 50, 40), dtype=np.uint8)
        data[0, :, :] = 10  # Small difference
        self.img3 = PrismatronImage(data, "planar")
    
    def test_psnr_identical(self):
        """Test PSNR with identical images."""
        psnr = self.img1.psnr(self.img2)
        assert psnr == float('inf')
    
    def test_psnr_different(self):
        """Test PSNR with different images."""
        psnr = self.img1.psnr(self.img3)
        assert isinstance(psnr, float)
        assert psnr > 0  # Should be positive for small differences
    
    def test_ssim_identical(self):
        """Test SSIM with identical images."""
        ssim = self.img1.ssim(self.img2)
        assert ssim == 1.0
    
    def test_ssim_different(self):
        """Test SSIM with different images."""
        ssim = self.img1.ssim(self.img3)
        assert isinstance(ssim, float)
        assert 0 <= ssim <= 1
    
    def test_mse_identical(self):
        """Test MSE with identical images."""
        mse = self.img1.mse(self.img2)
        assert mse == 0.0
    
    def test_mae_identical(self):
        """Test MAE with identical images."""
        mae = self.img1.mae(self.img2)
        assert mae == 0.0
    
    def test_compare_all_metrics(self):
        """Test compare method returns all metrics."""
        metrics = self.img1.compare(self.img3)
        
        expected_keys = {'psnr', 'ssim', 'mse', 'mae'}
        assert set(metrics.keys()) == expected_keys
        
        for key, value in metrics.items():
            assert isinstance(value, float)
    
    def test_shape_mismatch_error(self):
        """Test that shape mismatches raise errors."""
        other_img = PrismatronImage.zeros(30, 20)  # Different size
        
        with pytest.raises(ValueError, match="shapes don't match"):
            self.img1.psnr(other_img)


class TestAnalysisMethods:
    """Test image analysis methods."""
    
    def setup_method(self):
        """Setup test image with known properties."""
        # Create image with specific pattern
        data = np.zeros((3, 100, 80), dtype=np.uint8)
        data[0, 25:75, 20:60] = 255  # Red rectangle
        data[1, :50, :] = 100        # Green half
        data[2, 75:, 40:] = 200      # Blue corner
        self.img = PrismatronImage(data, "planar")
    
    def test_histogram(self):
        """Test histogram calculation."""
        hist = self.img.histogram()
        
        assert isinstance(hist, dict)
        assert set(hist.keys()) == {'red', 'green', 'blue'}
        
        for channel, values in hist.items():
            assert isinstance(values, np.ndarray)
            assert len(values) == 256  # Default bins
    
    def test_center_of_mass(self):
        """Test center of mass calculation."""
        com_x, com_y = self.img.center_of_mass()
        
        assert isinstance(com_x, float)
        assert isinstance(com_y, float)
        assert 0 <= com_x <= self.img.width
        assert 0 <= com_y <= self.img.height
    
    def test_brightness_stats(self):
        """Test brightness statistics."""
        stats = self.img.brightness_stats()
        
        expected_channels = {'red', 'green', 'blue', 'average'}
        assert set(stats.keys()) == expected_channels
        
        for channel, channel_stats in stats.items():
            assert set(channel_stats.keys()) == {'mean', 'std', 'min', 'max'}
            for stat_name, value in channel_stats.items():
                assert isinstance(value, float)
                assert value >= 0
    
    def test_color_distribution(self):
        """Test color distribution analysis."""
        dist = self.img.color_distribution()
        
        expected_keys = {'avg_saturation', 'avg_value', 'color_variance'}
        assert set(dist.keys()) == expected_keys
        
        for key, value in dist.items():
            assert isinstance(value, float)
            assert value >= 0


class TestFileIO:
    """Test file I/O operations."""
    
    def setup_method(self):
        """Setup test image and temporary directory."""
        self.img = PrismatronImage.solid_color(50, 30, (128, 64, 192))
        self.temp_dir = tempfile.mkdtemp()
    
    def test_to_bytes(self):
        """Test conversion to bytes."""
        # Test with basic backend first (no dependencies)
        npy_bytes = self.img.to_bytes("NPY", backend="basic")
        assert isinstance(npy_bytes, bytes)
        assert len(npy_bytes) > 0
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_to_bytes_pil(self):
        """Test conversion to bytes with PIL."""
        png_bytes = self.img.to_bytes("PNG", backend="pil")
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        
        # Test with JPEG
        jpg_bytes = self.img.to_bytes("JPEG", quality=85, backend="pil")
        assert isinstance(jpg_bytes, bytes)
        assert len(jpg_bytes) > 0
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_to_base64(self):
        """Test conversion to base64."""
        b64_str = self.img.to_base64("PNG")
        assert isinstance(b64_str, str)
        assert len(b64_str) > 0
        
        # Should be valid base64
        decoded = base64.b64decode(b64_str)
        assert isinstance(decoded, bytes)
    
    def test_save_load_roundtrip(self):
        """Test save and load round trip."""
        # Test with basic backend (numpy)
        npy_path = Path(self.temp_dir) / "test.npy"
        self.img.save(str(npy_path), backend="basic")
        assert npy_path.exists()
        
        # Load back
        loaded_img = PrismatronImage.from_file(str(npy_path), backend="basic")
        assert self.img == loaded_img
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_save_load_pil(self):
        """Test save and load with PIL backend."""
        png_path = Path(self.temp_dir) / "test.png"
        self.img.save(str(png_path), backend="pil")
        assert png_path.exists()
        
        # Load back (will have some compression artifacts)
        loaded_img = PrismatronImage.from_file(str(png_path), backend="pil")
        assert loaded_img.shape == self.img.shape
    
    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_from_bytes(self):
        """Test creation from bytes."""
        # First convert to bytes
        png_bytes = self.img.to_bytes("PNG")
        
        # Then create from bytes
        img_from_bytes = PrismatronImage.from_bytes(png_bytes)
        assert img_from_bytes.shape == self.img.shape


class TestCameraBackend:
    """Test camera capture functionality."""
    
    @patch('prismatron_image.cv2')
    def test_camera_context_manager(self, mock_cv2):
        """Test camera backend context manager."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        with CameraBackend(0) as camera:
            assert camera._cap == mock_cap
        
        mock_cap.release.assert_called_once()
    
    @patch('prismatron_image.cv2')
    def test_camera_capture(self, mock_cv2):
        """Test camera frame capture."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Mock successful frame capture
        test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_cv2.cvtColor.return_value = test_frame
        
        camera = CameraBackend(0)
        camera.open()
        
        frame = camera.capture()
        assert frame.shape == (640, 480, 3)  # Should be transposed
        
        camera.close()
    
    @patch('prismatron_image.OPENCV_AVAILABLE', False)
    def test_camera_no_opencv(self):
        """Test camera backend without OpenCV."""
        with pytest.raises(RuntimeError, match="OpenCV not available"):
            with CameraBackend(0) as camera:
                pass
    
    @patch('prismatron_image.cv2')
    def test_from_camera_factory(self, mock_cv2):
        """Test from_camera factory method."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Mock frame capture
        test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_cv2.cvtColor.return_value = test_frame
        
        img = PrismatronImage.from_camera(0, target_size=(100, 80))
        
        # Should be resized to target size
        assert img.width == 100
        assert img.height == 80


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_copy(self):
        """Test copy method."""
        img1 = PrismatronImage.solid_color(50, 30, (128, 64, 192))
        img2 = img1.copy()
        
        assert img1 == img2
        assert img1._data is not img2._data  # Should be different objects
    
    def test_validate(self):
        """Test validate method."""
        img = PrismatronImage.solid_color(50, 30, (128, 64, 192))
        assert img.validate() is True
    
    def test_equality(self):
        """Test equality comparison."""
        img1 = PrismatronImage.solid_color(50, 30, (128, 64, 192))
        img2 = PrismatronImage.solid_color(50, 30, (128, 64, 192))
        img3 = PrismatronImage.solid_color(50, 30, (255, 255, 255))
        
        assert img1 == img2
        assert img1 != img3
        assert img1 != "not an image"
    
    def test_repr(self):
        """Test string representation."""
        img = PrismatronImage.solid_color(50, 30, (128, 64, 192))
        repr_str = repr(img)
        
        assert "PrismatronImage" in repr_str
        assert "shape=(3, 50, 30)" in repr_str
        assert "uint8" in repr_str
    
    def test_array_interface(self):
        """Test numpy array interface."""
        img = PrismatronImage.solid_color(50, 30, (128, 64, 192))
        arr = np.array(img)
        
        assert arr.shape == (3, 50, 30)
        assert arr.dtype == np.uint8
        assert np.array_equal(arr, img.as_planar())


class TestDimensionGuessing:
    """Test dimension guessing for flat arrays."""
    
    def test_prismatron_dimensions(self):
        """Test that Prismatron dimensions are recognized."""
        pixels = PRISMATRON_WIDTH * PRISMATRON_HEIGHT
        data = np.random.randint(0, 256, pixels * 3, dtype=np.uint8)
        img = PrismatronImage(data, "flat_interleaved")
        
        assert img.width == PRISMATRON_WIDTH
        assert img.height == PRISMATRON_HEIGHT
    
    def test_square_dimensions(self):
        """Test square dimension guessing."""
        # 100x100 image
        pixels = 100 * 100
        data = np.random.randint(0, 256, pixels * 3, dtype=np.uint8)
        img = PrismatronImage(data, "flat_interleaved")
        
        assert img.width == 100
        assert img.height == 100
    
    def test_aspect_ratio_preference(self):
        """Test that reasonable aspect ratios are preferred."""
        # 80x60 = 4800 pixels (4:3 aspect ratio)
        pixels = 80 * 60
        data = np.random.randint(0, 256, pixels * 3, dtype=np.uint8)
        img = PrismatronImage(data, "flat_interleaved")
        
        # Should pick reasonable aspect ratio
        aspect_ratio = img.width / img.height
        assert 0.5 <= aspect_ratio <= 2.0
    
    def test_invalid_pixel_count(self):
        """Test invalid pixel counts."""
        # Prime number that can't be factored reasonably
        with pytest.raises(ValueError, match="Cannot determine dimensions"):
            pixels = 97  # Prime number
            data = np.random.randint(0, 256, pixels * 3, dtype=np.uint8)
            PrismatronImage(data, "flat_interleaved")


class TestBackendFallback:
    """Test backend fallback behavior."""
    
    def test_backend_auto_selection(self):
        """Test automatic backend selection."""
        img = PrismatronImage.solid_color(10, 10, (255, 0, 0))
        
        # Should work with auto backend - test with format that works with available backends
        if PIL_AVAILABLE:
            bytes_data = img.to_bytes("PNG", backend="auto")
        else:
            bytes_data = img.to_bytes("NPY", backend="basic")
        assert isinstance(bytes_data, bytes)
        assert len(bytes_data) > 0
    
    def test_backend_specific_selection(self):
        """Test specific backend selection."""
        img = PrismatronImage.solid_color(10, 10, (255, 0, 0))
        
        # Test basic backend
        bytes_data = img.to_bytes("NPY", backend="basic")
        assert isinstance(bytes_data, bytes)
    
    @patch('prismatron_image.PIL_AVAILABLE', False)
    @patch('prismatron_image.OPENCV_AVAILABLE', False)
    def test_fallback_to_basic(self):
        """Test fallback to basic backend when others unavailable."""
        img = PrismatronImage.solid_color(10, 10, (255, 0, 0))
        
        # Should fall back to basic backend
        bytes_data = img.to_bytes("NPY", backend="auto")
        assert isinstance(bytes_data, bytes)
    
    def test_unknown_backend_error(self):
        """Test error with unknown backend."""
        img = PrismatronImage.solid_color(10, 10, (255, 0, 0))
        
        with pytest.raises(ValueError, match="Unknown backend"):
            img.to_bytes("PNG", backend="unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])