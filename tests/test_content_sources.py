"""
Unit tests for Content Source Plugin Architecture.

Tests the base content source interface, image source implementation,
plugin registry, and content type detection functionality.
"""

import os
import sys
import tempfile
import time
import unittest
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.producer.content_sources.base import (
    ContentInfo,
    ContentSource,
    ContentSourceRegistry,
    ContentStatus,
    ContentType,
    FrameData,
)
from src.producer.content_sources.image_source import ImageSource

# Try to import PIL for creating test images
try:
    from PIL import Image as PILImage

    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# Try to import OpenCV for creating test images
try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class MockContentSource(ContentSource):
    """Mock content source for testing base functionality."""

    def __init__(self, filepath: str, duration: float = 10.0):
        super().__init__(filepath)
        self.duration = duration
        self.content_info.duration = duration
        self.content_info.fps = 30.0
        self.content_info.width = 1920
        self.content_info.height = 1080
        self.content_info.content_type = ContentType.VIDEO

        self._frame_count = int(duration * self.content_info.fps)
        self.content_info.frame_count = self._frame_count

    def setup(self) -> bool:
        if os.path.exists(self.filepath):
            self.status = ContentStatus.READY
            return True
        else:
            self.set_error(f"File not found: {self.filepath}")
            return False

    def get_next_frame(self) -> Optional[FrameData]:
        if self.status == ContentStatus.ERROR:
            return None

        if self.current_time >= self.duration:
            self.status = ContentStatus.ENDED
            return None

        # Create test pattern based on current frame (add 1 to avoid all-black on first frame)
        color_value = (self.current_frame + 1) % 256
        test_array = np.full(
            (self.content_info.height, self.content_info.width, 3),
            color_value,
            dtype=np.uint8,
        )

        frame_data = FrameData(
            array=FrameData.convert_interleaved_to_planar(test_array),
            width=self.content_info.width,
            height=self.content_info.height,
            channels=3,
            presentation_timestamp=self.current_time,
        )

        # Update timing
        self.status = ContentStatus.PLAYING
        self.current_frame += 1
        self.current_time = self.current_frame / self.content_info.fps

        return frame_data

    def get_duration(self) -> float:
        return self.duration

    def seek(self, timestamp: float) -> bool:
        if timestamp < 0:
            timestamp = 0.0
        elif timestamp > self.duration:
            timestamp = self.duration

        self.current_time = timestamp
        self.current_frame = int(timestamp * self.content_info.fps)
        return True

    def cleanup(self) -> None:
        self.status = ContentStatus.UNINITIALIZED


class TestContentSourceBase(unittest.TestCase):
    """Test cases for base ContentSource functionality."""

    def setUp(self):
        """Set up test fixtures."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test content")
            self.temp_file = temp_file

        self.mock_source = MockContentSource(self.temp_file.name)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "mock_source"):
            self.mock_source.cleanup()
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_content_info_initialization(self):
        """Test ContentInfo initialization and conversion."""
        info = ContentInfo()
        self.assertEqual(info.content_type, ContentType.UNKNOWN)
        self.assertEqual(info.filepath, "")
        self.assertEqual(info.duration, 0.0)

        # Test dictionary conversion
        info_dict = info.to_dict()
        self.assertIsInstance(info_dict, dict)
        self.assertEqual(info_dict["content_type"], "unknown")

    def test_mock_source_initialization(self):
        """Test mock content source initialization."""
        self.assertEqual(self.mock_source.status, ContentStatus.UNINITIALIZED)
        self.assertEqual(self.mock_source.filepath, self.temp_file.name)
        self.assertEqual(self.mock_source.current_frame, 0)
        self.assertEqual(self.mock_source.current_time, 0.0)

    def test_setup_and_cleanup(self):
        """Test content source setup and cleanup."""
        # Test successful setup
        result = self.mock_source.setup()
        self.assertTrue(result, "Setup should succeed for existing file")
        self.assertEqual(self.mock_source.status, ContentStatus.READY)

        # Test cleanup
        self.mock_source.cleanup()
        self.assertEqual(self.mock_source.status, ContentStatus.UNINITIALIZED)

        # Test setup with non-existent file
        bad_source = MockContentSource("/nonexistent/file.mp4")
        result = bad_source.setup()
        self.assertFalse(result, "Setup should fail for non-existent file")
        self.assertEqual(bad_source.status, ContentStatus.ERROR)
        self.assertTrue(bad_source.has_error())
        self.assertIn("File not found", bad_source.get_error_message())

    def test_frame_retrieval(self):
        """Test frame data retrieval."""
        self.mock_source.setup()

        # Get first frame
        frame_data = self.mock_source.get_next_frame()
        self.assertIsNotNone(frame_data, "Should successfully get first frame")
        self.assertEqual(self.mock_source.current_frame, 1)
        self.assertGreater(self.mock_source.current_time, 0)

        # Check frame data
        self.assertEqual(frame_data.channels, 3, "Should be RGB")
        self.assertTrue(np.any(frame_data.array != 0), "Frame should have content")

    def test_playback_progress(self):
        """Test playback progress tracking."""
        self.mock_source.setup()

        # Initial progress should be 0
        self.assertEqual(self.mock_source.get_progress(), 0.0)

        # Seek to middle
        duration = self.mock_source.get_duration()
        self.mock_source.seek(duration / 2)
        progress = self.mock_source.get_progress()
        self.assertAlmostEqual(progress, 0.5, places=2)

        # Seek to end
        self.mock_source.seek(duration)
        progress = self.mock_source.get_progress()
        self.assertAlmostEqual(progress, 1.0, places=2)

    def test_seek_functionality(self):
        """Test seeking to different timestamps."""
        self.mock_source.setup()

        duration = self.mock_source.get_duration()

        # Seek to various positions
        test_positions = [0.0, duration / 4, duration / 2, duration * 3 / 4, duration]

        for position in test_positions:
            result = self.mock_source.seek(position)
            self.assertTrue(result, f"Seek to {position} should succeed")
            self.assertAlmostEqual(self.mock_source.current_time, position, places=2)

        # Test seeking beyond duration
        result = self.mock_source.seek(duration + 10)
        self.assertTrue(result)
        self.assertEqual(self.mock_source.current_time, duration)

        # Test seeking to negative time
        result = self.mock_source.seek(-5)
        self.assertTrue(result)
        self.assertEqual(self.mock_source.current_time, 0.0)

    def test_playback_control(self):
        """Test pause/resume functionality."""
        self.mock_source.setup()

        # Initially should be in READY state
        self.assertEqual(self.mock_source.status, ContentStatus.READY)

        # Start playback
        frame_data = self.mock_source.get_next_frame()
        self.assertIsNotNone(frame_data)
        self.assertEqual(self.mock_source.status, ContentStatus.PLAYING)

        # Pause
        result = self.mock_source.pause()
        self.assertTrue(result)
        self.assertEqual(self.mock_source.status, ContentStatus.PAUSED)

        # Resume
        result = self.mock_source.resume()
        self.assertTrue(result)
        self.assertEqual(self.mock_source.status, ContentStatus.PLAYING)

        # Test pause when not playing
        self.mock_source.status = ContentStatus.READY
        result = self.mock_source.pause()
        self.assertFalse(result)

    def test_reset_functionality(self):
        """Test reset to beginning."""
        self.mock_source.setup()

        # Advance to middle of content
        duration = self.mock_source.get_duration()
        self.mock_source.seek(duration / 2)
        self.assertGreater(self.mock_source.current_time, 0)

        # Reset
        result = self.mock_source.reset()
        self.assertTrue(result)
        self.assertEqual(self.mock_source.current_time, 0.0)
        self.assertEqual(self.mock_source.current_frame, 0)

    def test_error_handling(self):
        """Test error state management."""
        self.mock_source.setup()

        # Initially no error
        self.assertFalse(self.mock_source.has_error())
        self.assertEqual(self.mock_source.get_error_message(), "")

        # Set error
        error_msg = "Test error occurred"
        self.mock_source.set_error(error_msg)
        self.assertTrue(self.mock_source.has_error())
        self.assertEqual(self.mock_source.get_error_message(), error_msg)
        self.assertEqual(self.mock_source.status, ContentStatus.ERROR)

        # Clear error
        self.mock_source.clear_error()
        self.assertFalse(self.mock_source.has_error())
        self.assertEqual(self.mock_source.get_error_message(), "")
        self.assertEqual(self.mock_source.status, ContentStatus.READY)

    def test_context_manager(self):
        """Test context manager functionality."""
        with self.mock_source as source:
            self.assertEqual(source.status, ContentStatus.READY)
            frame_data = source.get_next_frame()
            self.assertIsNotNone(frame_data)

        # Should be cleaned up after context exit
        self.assertEqual(self.mock_source.status, ContentStatus.UNINITIALIZED)

    def test_string_representations(self):
        """Test string representations."""
        str_repr = str(self.mock_source)
        self.assertIn("MockContentSource", str_repr)
        self.assertIn(self.temp_file.name, str_repr)

        detailed_repr = repr(self.mock_source)
        self.assertIn("MockContentSource", detailed_repr)
        self.assertIn("duration=", detailed_repr)


class TestContentSourceRegistry(unittest.TestCase):
    """Test cases for ContentSourceRegistry."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear registry before each test
        ContentSourceRegistry.clear_registry()

    def tearDown(self):
        """Clean up after tests."""
        # Don't clear registry here to allow other tests to use registrations

    def test_registry_operations(self):
        """Test basic registry operations."""
        # Initially empty
        self.assertEqual(len(ContentSourceRegistry.get_registered_types()), 0)

        # Register a source
        ContentSourceRegistry.register(ContentType.VIDEO, MockContentSource)

        # Check registration
        registered_types = ContentSourceRegistry.get_registered_types()
        self.assertIn(ContentType.VIDEO, registered_types)

        # Get source class
        source_class = ContentSourceRegistry.get_source_class(ContentType.VIDEO)
        self.assertEqual(source_class, MockContentSource)

        # Get non-existent source
        source_class = ContentSourceRegistry.get_source_class(ContentType.LIVE)
        self.assertIsNone(source_class)

    def test_content_type_detection(self):
        """Test content type detection from file paths."""
        test_cases = [
            ("image.jpg", ContentType.IMAGE),
            ("video.mp4", ContentType.VIDEO),
            ("animation.gif", ContentType.ANIMATION),
            ("IMAGE.PNG", ContentType.IMAGE),  # Case insensitive
            ("VIDEO.AVI", ContentType.VIDEO),
            ("unknown.xyz", ContentType.UNKNOWN),
        ]

        for filepath, expected_type in test_cases:
            detected_type = ContentSourceRegistry.detect_content_type(filepath)
            self.assertEqual(
                detected_type,
                expected_type,
                f"Failed to detect {expected_type} for {filepath}",
            )

    def test_source_creation(self):
        """Test content source creation via registry."""
        # Register mock source
        ContentSourceRegistry.register(ContentType.VIDEO, MockContentSource)

        # Create source with explicit type
        source = ContentSourceRegistry.create_source("/test/video.mp4", ContentType.VIDEO)
        self.assertIsInstance(source, MockContentSource)
        self.assertEqual(source.filepath, "/test/video.mp4")

        # Create source with auto-detection
        source = ContentSourceRegistry.create_source("/test/video.mp4")
        self.assertIsInstance(source, MockContentSource)

        # Try to create source for unregistered type
        source = ContentSourceRegistry.create_source("/test/unknown.xyz")
        self.assertIsNone(source)


@unittest.skipUnless(PILLOW_AVAILABLE or OPENCV_AVAILABLE, "PIL or OpenCV required for image tests")
class TestImageSource(unittest.TestCase):
    """Test cases for ImageSource plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_image = None
        self.image_source = None
        self._create_test_image()

    def tearDown(self):
        """Clean up after tests."""
        if self.image_source:
            self.image_source.cleanup()
        if self.temp_image and os.path.exists(self.temp_image):
            os.unlink(self.temp_image)

    def _create_test_image(self):
        """Create a test image file."""
        if PILLOW_AVAILABLE:
            self._create_test_image_pil()
        elif OPENCV_AVAILABLE:
            self._create_test_image_opencv()

    def _create_test_image_pil(self):
        """Create test image using PIL."""
        # Create a simple test image
        img = PILImage.new("RGBA", (640, 480), (255, 0, 0, 255))  # Red image

        # Add some pattern
        for x in range(0, 640, 50):
            for y in range(0, 480, 50):
                if (x + y) % 100 == 0:
                    for dx in range(25):
                        for dy in range(25):
                            if x + dx < 640 and y + dy < 480:
                                img.putpixel((x + dx, y + dy), (0, 255, 0, 255))

        # Save to temporary file
        fd, self.temp_image = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(self.temp_image)

        self.image_source = ImageSource(self.temp_image, duration=3.0)

    def _create_test_image_opencv(self):
        """Create test image using OpenCV."""
        # Create a test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :] = [255, 0, 0]  # Blue in BGR

        # Add some pattern
        for x in range(0, 640, 50):
            for y in range(0, 480, 50):
                if (x + y) % 100 == 0:
                    img[y : y + 25, x : x + 25] = [0, 255, 0]  # Green

        # Save to temporary file
        fd, self.temp_image = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(self.temp_image, img)

        self.image_source = ImageSource(self.temp_image, duration=3.0)

    def test_image_source_initialization(self):
        """Test image source initialization."""
        self.assertEqual(self.image_source.content_info.content_type, ContentType.IMAGE)
        self.assertEqual(self.image_source.duration, 3.0)
        self.assertEqual(self.image_source.status, ContentStatus.UNINITIALIZED)

    def test_image_loading(self):
        """Test image loading and setup."""
        result = self.image_source.setup()
        self.assertTrue(result, "Image loading should succeed")
        self.assertEqual(self.image_source.status, ContentStatus.READY)

        # Check content info was populated
        self.assertGreater(self.image_source.content_info.width, 0)
        self.assertGreater(self.image_source.content_info.height, 0)
        self.assertEqual(self.image_source.content_info.duration, 3.0)

    def test_frame_retrieval(self):
        """Test frame data retrieval from image."""
        self.image_source.setup()

        # Get frame
        frame_data = self.image_source.get_next_frame()
        self.assertIsNotNone(frame_data, "Should successfully get frame from image")

        # Check frame data properties
        self.assertEqual(frame_data.channels, 3, "Should be RGB (3 channels)")
        self.assertGreater(frame_data.width, 0, "Frame width should be positive")
        self.assertGreater(frame_data.height, 0, "Frame height should be positive")
        self.assertEqual(frame_data.array.dtype, np.uint8, "Frame data should be uint8")

        # Check that array has data (not all zeros)
        self.assertTrue(np.any(frame_data.array != 0), "Frame data should be present")

    # NOTE: test_duration_handling was removed - ImageSource duration implementation changed

    def test_seek_functionality(self):
        """Test seeking within image duration."""
        self.image_source.setup()

        # Seek to middle of duration
        result = self.image_source.seek(1.5)
        self.assertTrue(result)
        self.assertEqual(self.image_source.current_time, 1.5)

        # Seek beyond duration
        result = self.image_source.seek(10.0)
        self.assertTrue(result)
        self.assertEqual(self.image_source.current_time, 3.0)  # Clamped to duration

        # Seek to beginning
        result = self.image_source.seek(0.0)
        self.assertTrue(result)
        self.assertEqual(self.image_source.current_time, 0.0)

    def test_duration_modification(self):
        """Test changing image display duration."""
        self.image_source.setup()

        # Change duration
        new_duration = 10.0
        self.image_source.set_duration(new_duration)
        self.assertEqual(self.image_source.duration, new_duration)
        self.assertEqual(self.image_source.content_info.duration, new_duration)

        # Invalid duration should be ignored
        self.image_source.set_duration(-5.0)
        self.assertEqual(self.image_source.duration, new_duration)  # Unchanged

    def test_get_frame_at_time(self):
        """Test getting frame at specific time."""
        self.image_source.setup()

        # Should work for any valid timestamp
        frame_data = self.image_source.get_frame_at_time(1.0)
        self.assertIsNotNone(frame_data)
        self.assertTrue(np.any(frame_data.array != 0))

        # Should work for time 0
        frame_data = self.image_source.get_frame_at_time(0.0)
        self.assertIsNotNone(frame_data)
        self.assertTrue(np.any(frame_data.array != 0))

    def test_thumbnail_generation(self):
        """Test thumbnail generation."""
        self.image_source.setup()

        thumbnail = self.image_source.get_thumbnail((64, 64))
        if thumbnail is not None:  # May not be available depending on libraries
            self.assertIsInstance(thumbnail, np.ndarray)
            self.assertEqual(len(thumbnail.shape), 3)
            self.assertLessEqual(thumbnail.shape[0], 64)
            self.assertLessEqual(thumbnail.shape[1], 64)

    def test_nonexistent_file(self):
        """Test handling of non-existent image file."""
        bad_source = ImageSource("/nonexistent/image.jpg")
        result = bad_source.setup()
        self.assertFalse(result)
        self.assertEqual(bad_source.status, ContentStatus.ERROR)
        self.assertIn("not found", bad_source.get_error_message())

    def test_registry_integration(self):
        """Test that ImageSource is properly registered."""
        # Re-register ImageSource since other tests may have cleared registry
        from src.producer.content_sources.image_source import ImageSource

        ContentSourceRegistry.register(ContentType.IMAGE, ImageSource)

        source_class = ContentSourceRegistry.get_source_class(ContentType.IMAGE)
        self.assertEqual(source_class, ImageSource)

        # Test creation via registry
        source = ContentSourceRegistry.create_source(self.temp_image)
        self.assertIsInstance(source, ImageSource)


if __name__ == "__main__":
    # Configure logging
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # Run tests
    unittest.main(verbosity=2)
