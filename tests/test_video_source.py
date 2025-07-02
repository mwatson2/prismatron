"""
Unit tests for the Video Content Source Plugin.

Tests video file loading, metadata extraction, frame decoding,
hardware acceleration detection, and resource cleanup.
"""

import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.producer.content_sources.base import ContentStatus, ContentType
from src.producer.content_sources.video_source import FFMPEG_AVAILABLE, VideoSource


class TestVideoSource(unittest.TestCase):
    """Test cases for VideoSource class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_video_path = "/tmp/test_video.mp4"  # Mock path

        # Mock video metadata
        self.mock_video_metadata = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "duration": "10.0",
                    "codec_name": "h264",
                    "codec_long_name": "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                    "bit_rate": "5000000",
                }
            ],
            "format": {
                "duration": "10.0",
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "size": "6250000",
            },
        }

    def tearDown(self):
        """Clean up after tests."""

    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", True)
    def test_initialization(self):
        """Test video source initialization."""
        video_source = VideoSource(self.test_video_path)

        self.assertEqual(video_source.filepath, self.test_video_path)
        self.assertEqual(video_source.content_info.content_type, ContentType.VIDEO)
        self.assertEqual(video_source.status, ContentStatus.UNINITIALIZED)
        self.assertEqual(video_source.current_frame, 0)
        self.assertEqual(video_source.current_time, 0.0)

    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", False)
    def test_initialization_without_ffmpeg(self):
        """Test video source initialization when FFmpeg is not available."""
        video_source = VideoSource(self.test_video_path)

        result = video_source.setup()

        self.assertFalse(result)
        self.assertEqual(video_source.status, ContentStatus.ERROR)
        self.assertIn("FFmpeg not available", video_source.get_error_message())

    @patch("src.producer.content_sources.video_source.ffmpeg")
    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", True)
    @patch("os.path.exists")
    def test_setup_nonexistent_file(self, mock_exists, mock_ffmpeg):
        """Test setup with non-existent video file."""
        mock_exists.return_value = False

        video_source = VideoSource(self.test_video_path)
        result = video_source.setup()

        self.assertFalse(result)
        self.assertEqual(video_source.status, ContentStatus.ERROR)
        self.assertIn("Video file not found", video_source.get_error_message())

    @patch("src.producer.content_sources.video_source.ffmpeg")
    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", True)
    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_successful_setup(self, mock_subprocess, mock_exists, mock_ffmpeg):
        """Test successful video source setup."""
        # Setup mocks
        mock_exists.return_value = True
        mock_ffmpeg.probe.return_value = self.mock_video_metadata

        # Mock hardware acceleration detection
        mock_subprocess.return_value = Mock(returncode=0, stdout="Hardware acceleration methods:\ncuda\nnvdec\n")

        # Mock FFmpeg process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.stdout = Mock()
        mock_ffmpeg.run_async.return_value = mock_process

        video_source = VideoSource(self.test_video_path)

        with patch("queue.Queue"), patch("threading.Thread"):
            result = video_source.setup()

        self.assertTrue(result)
        self.assertEqual(video_source.status, ContentStatus.READY)
        self.assertEqual(video_source.content_info.width, 1920)
        self.assertEqual(video_source.content_info.height, 1080)
        self.assertEqual(video_source.content_info.fps, 30.0)
        self.assertEqual(video_source.content_info.duration, 10.0)
        self.assertEqual(video_source.content_info.frame_count, 300)

    @patch("src.producer.content_sources.video_source.ffmpeg")
    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", True)
    @patch("os.path.exists")
    def test_metadata_extraction(self, mock_exists, mock_ffmpeg):
        """Test video metadata extraction."""
        mock_exists.return_value = True
        mock_ffmpeg.probe.return_value = self.mock_video_metadata

        video_source = VideoSource(self.test_video_path)

        result = video_source._probe_video_metadata()

        self.assertTrue(result)
        self.assertEqual(video_source._frame_width, 1920)
        self.assertEqual(video_source._frame_height, 1080)
        self.assertEqual(video_source._frame_rate, 30.0)
        self.assertEqual(video_source.content_info.duration, 10.0)
        self.assertEqual(video_source.content_info.metadata["codec_name"], "h264")

    @patch("src.producer.content_sources.video_source.ffmpeg")
    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", True)
    @patch("os.path.exists")
    def test_metadata_extraction_no_video_stream(self, mock_exists, mock_ffmpeg):
        """Test metadata extraction with no video stream."""
        mock_exists.return_value = True
        mock_ffmpeg.probe.return_value = {
            "streams": [{"codec_type": "audio"}],  # No video stream
            "format": {"duration": "10.0"},
        }

        video_source = VideoSource(self.test_video_path)

        result = video_source._probe_video_metadata()

        self.assertFalse(result)
        self.assertEqual(video_source.status, ContentStatus.ERROR)
        self.assertIn("No video stream found", video_source.get_error_message())

    @patch("subprocess.run")
    def test_hardware_acceleration_detection_cuda(self, mock_subprocess):
        """Test CUDA hardware acceleration detection."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="Hardware acceleration methods:\ncuda\n")

        video_source = VideoSource(self.test_video_path)
        video_source._detect_hardware_acceleration()

        self.assertEqual(video_source._hardware_acceleration, "cuda")

    @patch("subprocess.run")
    def test_hardware_acceleration_detection_nvdec(self, mock_subprocess):
        """Test NVDEC hardware acceleration detection."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="Hardware acceleration methods:\nnvdec\n")

        video_source = VideoSource(self.test_video_path)
        video_source._detect_hardware_acceleration()

        self.assertEqual(video_source._hardware_acceleration, "nvdec")

    @patch("subprocess.run")
    def test_hardware_acceleration_detection_none(self, mock_subprocess):
        """Test no hardware acceleration available."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="Hardware acceleration methods:\nsoftware\n")

        video_source = VideoSource(self.test_video_path)
        video_source._detect_hardware_acceleration()

        self.assertIsNone(video_source._hardware_acceleration)

    @patch("subprocess.run")
    def test_hardware_acceleration_detection_error(self, mock_subprocess):
        """Test hardware acceleration detection error."""
        mock_subprocess.side_effect = Exception("Command failed")

        video_source = VideoSource(self.test_video_path)
        video_source._detect_hardware_acceleration()

        self.assertIsNone(video_source._hardware_acceleration)

    @patch("src.producer.content_sources.video_source.ffmpeg")
    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", True)
    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_frame_reading_simulation(self, mock_subprocess, mock_exists, mock_ffmpeg):
        """Test frame reading simulation."""
        # Setup mocks
        mock_exists.return_value = True
        mock_ffmpeg.probe.return_value = self.mock_video_metadata
        mock_subprocess.return_value = Mock(returncode=0, stdout="software\n")

        # Mock FFmpeg process with frame data
        mock_process = Mock()
        mock_process.pid = 12345

        # Create mock frame data (RGB24 format)
        frame_width, frame_height = 1920, 1080
        frame_size = frame_width * frame_height * 3
        mock_frame_data = np.random.randint(0, 255, frame_size, dtype=np.uint8).tobytes()

        mock_process.stdout.read.return_value = mock_frame_data
        mock_ffmpeg.run_async.return_value = mock_process

        video_source = VideoSource(self.test_video_path)

        with patch("queue.Queue") as mock_queue_class, patch("threading.Thread") as mock_thread:
            # Setup queue mock
            mock_queue = Mock()
            mock_queue_class.return_value = mock_queue
            mock_queue.get_nowait.return_value = Mock(
                array=np.zeros((1080, 1920, 3), dtype=np.uint8),
                width=1920,
                height=1080,
                channels=3,
                presentation_timestamp=0.0,
            )

            # Setup and get frame
            video_source.setup()
            video_source.status = ContentStatus.READY  # Force status

            frame_data = video_source.get_next_frame()

            self.assertIsNotNone(frame_data)
            self.assertEqual(video_source.status, ContentStatus.PLAYING)

    @patch("src.producer.content_sources.video_source.ffmpeg")
    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", True)
    @patch("os.path.exists")
    def test_get_next_frame_error_status(self, mock_exists, mock_ffmpeg):
        """Test get_next_frame when in error status."""
        video_source = VideoSource(self.test_video_path)
        video_source.status = ContentStatus.ERROR

        frame_data = video_source.get_next_frame()

        self.assertIsNone(frame_data)

    @patch("src.producer.content_sources.video_source.ffmpeg")
    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", True)
    @patch("os.path.exists")
    def test_get_next_frame_no_queue(self, mock_exists, mock_ffmpeg):
        """Test get_next_frame when frame queue is not initialized."""
        video_source = VideoSource(self.test_video_path)
        video_source.status = ContentStatus.READY
        video_source._frame_queue = None

        frame_data = video_source.get_next_frame()

        self.assertIsNone(frame_data)
        self.assertEqual(video_source.status, ContentStatus.ERROR)

    def test_get_duration(self):
        """Test get_duration method."""
        video_source = VideoSource(self.test_video_path)
        video_source.content_info.duration = 15.5

        duration = video_source.get_duration()

        self.assertEqual(duration, 15.5)

    @patch("src.producer.content_sources.video_source.ffmpeg")
    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", True)
    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_seek_functionality(self, mock_subprocess, mock_exists, mock_ffmpeg):
        """Test seek functionality."""
        # Setup mocks
        mock_exists.return_value = True
        mock_ffmpeg.probe.return_value = self.mock_video_metadata
        mock_subprocess.return_value = Mock(returncode=0, stdout="software\n")

        mock_process = Mock()
        mock_process.pid = 12345
        mock_ffmpeg.run_async.return_value = mock_process

        video_source = VideoSource(self.test_video_path)

        with patch("queue.Queue"), patch("threading.Thread"):
            video_source.setup()

            # Mock the stop and restart methods
            video_source._stop_current_playback = Mock()
            video_source._start_ffmpeg_with_seek = Mock(return_value=True)

            result = video_source.seek(5.0)

            self.assertTrue(result)
            self.assertEqual(video_source.current_time, 5.0)
            video_source._stop_current_playback.assert_called_once()
            video_source._start_ffmpeg_with_seek.assert_called_once_with(5.0)

    def test_seek_invalid_timestamp(self):
        """Test seek with invalid timestamp."""
        video_source = VideoSource(self.test_video_path)
        video_source.content_info.duration = 10.0

        # Test negative timestamp
        result = video_source.seek(-1.0)
        self.assertFalse(result)

        # Test timestamp beyond duration
        result = video_source.seek(15.0)
        self.assertFalse(result)

    @patch("src.producer.content_sources.video_source.ffmpeg")
    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", True)
    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_cleanup(self, mock_subprocess, mock_exists, mock_ffmpeg):
        """Test cleanup functionality."""
        # Setup mocks
        mock_exists.return_value = True
        mock_ffmpeg.probe.return_value = self.mock_video_metadata
        mock_subprocess.return_value = Mock(returncode=0, stdout="software\n")

        mock_process = Mock()
        mock_process.pid = 12345
        mock_ffmpeg.run_async.return_value = mock_process

        video_source = VideoSource(self.test_video_path)

        with patch("queue.Queue"), patch("threading.Thread"):
            video_source.setup()

            # Mock cleanup methods
            video_source._stop_current_playback = Mock()

            video_source.cleanup()

            self.assertEqual(video_source.status, ContentStatus.UNINITIALIZED)
            self.assertEqual(video_source.current_time, 0.0)
            self.assertEqual(video_source._current_frame_number, 0)
            video_source._stop_current_playback.assert_called_once()

    @patch("src.producer.content_sources.video_source.ffmpeg")
    @patch("src.producer.content_sources.video_source.FFMPEG_AVAILABLE", True)
    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_get_video_info(self, mock_subprocess, mock_exists, mock_ffmpeg):
        """Test get_video_info method."""
        # Setup mocks
        mock_exists.return_value = True
        mock_ffmpeg.probe.return_value = self.mock_video_metadata
        mock_subprocess.return_value = Mock(returncode=0, stdout="Hardware acceleration methods:\ncuda\n")

        video_source = VideoSource(self.test_video_path)

        with patch("queue.Queue"), patch("threading.Thread"):
            video_source.setup()

            video_info = video_source.get_video_info()

            expected_keys = [
                "width",
                "height",
                "fps",
                "duration",
                "total_frames",
                "current_frame",
                "hardware_acceleration",
                "codec",
                "format",
                "bitrate",
            ]

            for key in expected_keys:
                self.assertIn(key, video_info)

            self.assertEqual(video_info["width"], 1920)
            self.assertEqual(video_info["height"], 1080)
            self.assertEqual(video_info["fps"], 30.0)
            self.assertEqual(video_info["hardware_acceleration"], "cuda")

    def test_context_manager(self):
        """Test video source as context manager."""
        video_source = VideoSource(self.test_video_path)

        # Mock setup to succeed
        video_source.setup = Mock(return_value=True)
        video_source.cleanup = Mock()

        with video_source as vs:
            self.assertEqual(vs, video_source)
            video_source.setup.assert_called_once()

        video_source.cleanup.assert_called_once()

    def test_context_manager_setup_failure(self):
        """Test context manager when setup fails."""
        video_source = VideoSource(self.test_video_path)

        # Mock setup to fail
        video_source.setup = Mock(return_value=False)

        with self.assertRaises(RuntimeError), video_source:
            pass

    def test_string_representation(self):
        """Test string representation methods."""
        video_source = VideoSource(self.test_video_path)
        video_source.status = ContentStatus.READY
        video_source.content_info.duration = 10.0

        str_repr = str(video_source)
        self.assertIn("VideoSource", str_repr)
        self.assertIn(self.test_video_path, str_repr)
        self.assertIn("ready", str_repr)

        repr_str = repr(video_source)
        self.assertIn("VideoSource", repr_str)
        self.assertIn("duration=10.00s", repr_str)


class TestVideoSourceRegistry(unittest.TestCase):
    """Test cases for VideoSource registry integration."""

    def setUp(self):
        """Set up test fixtures."""
        from src.producer.content_sources.base import ContentSourceRegistry, ContentType

        # Clear registry and re-register VideoSource to ensure clean state
        ContentSourceRegistry.clear_registry()
        ContentSourceRegistry.register(ContentType.VIDEO, VideoSource)

    def test_video_source_registration(self):
        """Test that VideoSource is properly registered."""
        from src.producer.content_sources.base import ContentSourceRegistry, ContentType

        # Test that VIDEO type is registered
        video_class = ContentSourceRegistry.get_source_class(ContentType.VIDEO)
        self.assertEqual(video_class, VideoSource)

    def test_video_content_detection(self):
        """Test video content type detection."""
        from src.producer.content_sources.base import ContentSourceRegistry

        test_files = ["test.mp4", "movie.avi", "video.mkv", "clip.mov", "stream.webm"]

        for filename in test_files:
            content_type = ContentSourceRegistry.detect_content_type(filename)
            self.assertEqual(content_type, ContentType.VIDEO)

    def test_create_video_source(self):
        """Test creating video source through registry."""
        from src.producer.content_sources.base import ContentSourceRegistry, ContentType

        video_source = ContentSourceRegistry.create_source("/tmp/test.mp4", ContentType.VIDEO)

        self.assertIsInstance(video_source, VideoSource)
        self.assertEqual(video_source.filepath, "/tmp/test.mp4")


if __name__ == "__main__":
    unittest.main()
