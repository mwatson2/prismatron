"""
Unit tests for the Producer Process Core.

Tests content playlist management, frame timing, buffer management,
and producer process coordination.
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

from src.core.control_state import PlayState
from src.producer.content_sources.base import ContentStatus, ContentType, FrameData
from src.producer.producer import (
    ContentPlaylist,
    PlaylistItem,
    ProducerProcess,
    pause,
    play,
    stop,
)


class TestPlaylistItem(unittest.TestCase):
    """Test cases for PlaylistItem class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_filepath = "/tmp/test_content.mp4"

    def test_initialization(self):
        """Test playlist item initialization."""
        item = PlaylistItem(self.test_filepath, duration=10.0, repeat=3)

        self.assertEqual(item.filepath, self.test_filepath)
        self.assertEqual(item.duration_override, 10.0)
        self.assertEqual(item.repeat_count, 3)
        self.assertEqual(item.current_repeat, 0)
        self.assertIsNone(item._content_source)
        self.assertIsNone(item._detected_type)

    def test_repeat_functionality(self):
        """Test repeat functionality."""
        item = PlaylistItem(self.test_filepath, repeat=3)

        # Should repeat 3 times
        self.assertTrue(item.should_repeat())  # 0 < 3

        item.current_repeat = 1
        self.assertTrue(item.should_repeat())  # 1 < 3

        item.current_repeat = 2
        self.assertTrue(item.should_repeat())  # 2 < 3

        item.current_repeat = 3
        self.assertFalse(item.should_repeat())  # 3 = 3

        # Reset repeats
        item.reset_repeats()
        self.assertEqual(item.current_repeat, 0)
        self.assertTrue(item.should_repeat())

    @patch(
        "src.producer.content_sources.base.ContentSourceRegistry.detect_content_type"
    )
    @patch("src.producer.content_sources.base.ContentSourceRegistry.create_source")
    def test_get_content_source(self, mock_create_source, mock_detect_type):
        """Test content source creation."""
        # Setup mocks
        mock_detect_type.return_value = ContentType.VIDEO
        mock_source = Mock()
        mock_create_source.return_value = mock_source

        item = PlaylistItem(self.test_filepath)

        # First call should create source
        source = item.get_content_source()

        self.assertEqual(source, mock_source)
        self.assertEqual(item._detected_type, ContentType.VIDEO)
        mock_detect_type.assert_called_once_with(self.test_filepath)
        mock_create_source.assert_called_once_with(
            self.test_filepath, ContentType.VIDEO
        )

        # Second call should return cached source
        source2 = item.get_content_source()
        self.assertEqual(source2, mock_source)
        self.assertEqual(mock_create_source.call_count, 1)  # Not called again

    def test_get_effective_duration(self):
        """Test effective duration calculation."""
        # Test with override
        item = PlaylistItem(self.test_filepath, duration=15.0)
        self.assertEqual(item.get_effective_duration(), 15.0)

        # Test without override (mock source)
        item2 = PlaylistItem(self.test_filepath)
        mock_source = Mock()
        mock_source.get_duration.return_value = 20.0
        item2._content_source = mock_source

        self.assertEqual(item2.get_effective_duration(), 20.0)

        # Test with no source (mock to return None)
        item3 = PlaylistItem(self.test_filepath)
        item3.get_content_source = Mock(return_value=None)
        self.assertEqual(item3.get_effective_duration(), 0.0)

    def test_cleanup(self):
        """Test cleanup functionality."""
        item = PlaylistItem(self.test_filepath)
        mock_source = Mock()
        item._content_source = mock_source

        item.cleanup()

        mock_source.cleanup.assert_called_once()
        self.assertIsNone(item._content_source)

    def test_string_representation(self):
        """Test string representation."""
        item = PlaylistItem(self.test_filepath, repeat=2)
        str_repr = str(item)

        self.assertIn("PlaylistItem", str_repr)
        self.assertIn(self.test_filepath, str_repr)
        self.assertIn("repeat=2", str_repr)


class TestContentPlaylist(unittest.TestCase):
    """Test cases for ContentPlaylist class."""

    def setUp(self):
        """Set up test fixtures."""
        self.playlist = ContentPlaylist()
        self.test_files = ["/tmp/test1.mp4", "/tmp/test2.jpg", "/tmp/test3.mp4"]

    def tearDown(self):
        """Clean up after tests."""
        self.playlist.clear()

    @patch("os.path.exists")
    def test_add_item_success(self, mock_exists):
        """Test successful item addition."""
        mock_exists.return_value = True

        result = self.playlist.add_item(self.test_files[0])

        self.assertTrue(result)
        self.assertEqual(len(self.playlist), 1)

    @patch("os.path.exists")
    def test_add_item_nonexistent_file(self, mock_exists):
        """Test adding non-existent file."""
        mock_exists.return_value = False

        result = self.playlist.add_item(self.test_files[0])

        self.assertFalse(result)
        self.assertEqual(len(self.playlist), 0)

    @patch("os.path.exists")
    def test_multiple_items(self, mock_exists):
        """Test multiple item management."""
        mock_exists.return_value = True

        # Add multiple items
        for filepath in self.test_files:
            self.assertTrue(self.playlist.add_item(filepath))

        self.assertEqual(len(self.playlist), 3)

        # Test current item
        current = self.playlist.get_current_item()
        self.assertIsNotNone(current)
        self.assertEqual(current.filepath, self.test_files[0])

    @patch("os.path.exists")
    def test_remove_item(self, mock_exists):
        """Test item removal."""
        mock_exists.return_value = True

        # Add items
        for filepath in self.test_files:
            self.playlist.add_item(filepath)

        # Remove middle item
        result = self.playlist.remove_item(1)

        self.assertTrue(result)
        self.assertEqual(len(self.playlist), 2)

        # Test invalid index
        result = self.playlist.remove_item(10)
        self.assertFalse(result)

    @patch("os.path.exists")
    def test_playlist_advancement(self, mock_exists):
        """Test playlist advancement."""
        mock_exists.return_value = True

        # Add items with repeats
        self.playlist.add_item(self.test_files[0], repeat=2)
        self.playlist.add_item(self.test_files[1], repeat=1)

        # First item should repeat
        current = self.playlist.get_current_item()
        self.assertEqual(current.filepath, self.test_files[0])

        # Advance - should repeat first item
        result = self.playlist.advance_to_next()
        self.assertTrue(result)
        current = self.playlist.get_current_item()
        self.assertEqual(current.filepath, self.test_files[0])
        self.assertEqual(current.current_repeat, 1)

        # Advance again - should move to second item
        result = self.playlist.advance_to_next()
        self.assertTrue(result)
        current = self.playlist.get_current_item()
        self.assertEqual(current.filepath, self.test_files[1])

        # Advance again - should loop back to first item
        result = self.playlist.advance_to_next()
        self.assertTrue(result)
        current = self.playlist.get_current_item()
        self.assertEqual(current.filepath, self.test_files[0])

    @patch("os.path.exists")
    def test_playlist_no_loop(self, mock_exists):
        """Test playlist without looping."""
        mock_exists.return_value = True

        self.playlist.set_loop(False)
        self.playlist.add_item(self.test_files[0])

        # Advance past end should return False
        result = self.playlist.advance_to_next()
        self.assertFalse(result)

    def test_clear_playlist(self):
        """Test playlist clearing."""
        # Mock some items
        mock_item1 = Mock()
        mock_item2 = Mock()
        self.playlist._items = [mock_item1, mock_item2]

        self.playlist.clear()

        self.assertEqual(len(self.playlist), 0)
        mock_item1.cleanup.assert_called_once()
        mock_item2.cleanup.assert_called_once()

    @patch("os.path.exists")
    def test_get_playlist_info(self, mock_exists):
        """Test playlist info generation."""
        mock_exists.return_value = True

        self.playlist.add_item(self.test_files[0], duration=10.0)
        self.playlist.add_item(self.test_files[1], duration=5.0)

        info = self.playlist.get_playlist_info()

        self.assertEqual(info["total_items"], 2)
        self.assertEqual(info["current_index"], 0)
        self.assertEqual(info["total_duration"], 15.0)
        self.assertTrue(info["loop_enabled"])
        self.assertEqual(len(info["items"]), 2)


class TestProducerProcess(unittest.TestCase):
    """Test cases for ProducerProcess class."""

    def setUp(self):
        """Set up test fixtures."""
        self.buffer_name = f"test_buffer_{int(time.time() * 1000000)}"
        self.control_name = f"test_control_{int(time.time() * 1000000)}"

        # Create producer with mocked components
        with patch("src.producer.producer.FrameProducer"), patch(
            "src.producer.producer.ControlState"
        ):
            self.producer = ProducerProcess(self.buffer_name, self.control_name)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.producer, "_running") and self.producer._running:
            self.producer.stop()

    @patch("src.producer.producer.FrameProducer")
    @patch("src.producer.producer.ControlState")
    def test_initialization_success(self, mock_control_state, mock_frame_buffer):
        """Test successful producer initialization."""
        # Setup mocks
        mock_buffer = Mock()
        mock_buffer.initialize.return_value = True
        mock_frame_buffer.return_value = mock_buffer

        mock_control = Mock()
        mock_control.initialize.return_value = True
        mock_control_state.return_value = mock_control

        producer = ProducerProcess(self.buffer_name, self.control_name)
        result = producer.initialize()

        self.assertTrue(result)
        mock_buffer.initialize.assert_called_once()
        mock_control.initialize.assert_called_once()

    @patch("src.producer.producer.FrameProducer")
    @patch("src.producer.producer.ControlState")
    def test_initialization_failure(self, mock_control_state, mock_frame_buffer):
        """Test producer initialization failure."""
        # Setup mocks to fail
        mock_buffer = Mock()
        mock_buffer.initialize.return_value = False
        mock_frame_buffer.return_value = mock_buffer

        producer = ProducerProcess(self.buffer_name, self.control_name)
        result = producer.initialize()

        self.assertFalse(result)

    @patch("src.producer.producer.Path")
    def test_load_playlist_from_directory(self, mock_path_class):
        """Test loading playlist from directory."""
        # Setup mock directory instance
        mock_dir = Mock()
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = True

        # Mock files
        mock_files = [
            Mock(is_file=Mock(return_value=True), suffix=".mp4"),
            Mock(is_file=Mock(return_value=True), suffix=".jpg"),
            Mock(is_file=Mock(return_value=True), suffix=".txt"),  # Unsupported
            Mock(is_file=Mock(return_value=False), suffix=".mp4"),  # Not a file
        ]

        for i, f in enumerate(mock_files):
            f.__str__ = Mock(return_value=f"/tmp/file{i}{f.suffix}")

        mock_dir.rglob.return_value = mock_files
        mock_path_class.return_value = mock_dir

        # Mock playlist.add_item
        self.producer._playlist.add_item = Mock(return_value=True)

        count = self.producer.load_playlist_from_directory("/tmp/test")

        # Should add 2 files (mp4 and jpg, not txt or non-file)
        self.assertEqual(count, 2)
        self.assertEqual(self.producer._playlist.add_item.call_count, 2)

    def test_start_stop_producer(self):
        """Test starting and stopping producer."""
        # Mock threading
        with patch("threading.Thread") as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            # Start producer
            result = self.producer.start()

            self.assertTrue(result)
            self.assertTrue(self.producer._running)
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()

            # Stop producer
            self.producer.stop()

            self.assertFalse(self.producer._running)

    def test_frame_rendering_simulation(self):
        """Test frame rendering to buffer."""
        # Create mock frame data
        frame_data = FrameData(
            array=np.zeros((480, 640, 3), dtype=np.uint8),
            width=640,
            height=480,
            channels=3,
            presentation_timestamp=0.0,
        )

        # Mock buffer operations
        mock_buffer_info = Mock()
        mock_buffer_array = np.zeros((1080, 1920, 4), dtype=np.uint8)
        mock_buffer_info.get_array.return_value = mock_buffer_array

        self.producer._frame_buffer.get_write_buffer = Mock(
            return_value=mock_buffer_info
        )
        self.producer._frame_buffer.advance_write = Mock(return_value=True)

        with patch.object(self.producer, "_copy_frame_to_buffer") as mock_copy:
            result = self.producer._render_frame_to_buffer(frame_data)

            self.assertTrue(result)
            self.producer._frame_buffer.get_write_buffer.assert_called_once()
            mock_copy.assert_called_once_with(frame_data, mock_buffer_array)
            self.producer._frame_buffer.advance_write.assert_called_once()

    def test_get_producer_stats(self):
        """Test producer statistics."""
        # Setup some state
        self.producer._running = True
        self.producer._frames_produced = 100
        self.producer._start_time = time.time() - 10.0  # 10 seconds ago
        self.producer._target_fps = 30.0

        # Mock playlist and buffer stats
        self.producer._playlist.get_playlist_info = Mock(
            return_value={"total_items": 5}
        )
        self.producer._frame_buffer.get_producer_stats = Mock(
            return_value={"buffer_stat": "value"}
        )

        stats = self.producer.get_producer_stats()

        self.assertTrue(stats["running"])
        self.assertEqual(stats["frames_produced"], 100)
        self.assertGreater(stats["producer_fps"], 0)
        self.assertEqual(stats["target_fps"], 30.0)
        self.assertIn("playlist_info", stats)
        self.assertIn("buffer_stat", stats)

    def test_cleanup(self):
        """Test producer cleanup."""
        # Mock components
        self.producer._playlist.clear = Mock()
        self.producer._frame_buffer.cleanup = Mock()
        self.producer._control_state.cleanup = Mock()
        self.producer.stop = Mock()

        self.producer.cleanup()

        self.producer.stop.assert_called_once()
        self.producer._playlist.clear.assert_called_once()
        self.producer._frame_buffer.cleanup.assert_called_once()
        self.producer._control_state.cleanup.assert_called_once()


class TestProducerControlFunctions(unittest.TestCase):
    """Test cases for producer control functions."""

    @patch("src.producer.producer.ControlState")
    def test_play_function(self, mock_control_state):
        """Test play control function."""
        mock_control = Mock()
        mock_control.connect.return_value = True
        mock_control.set_play_state.return_value = True
        mock_control_state.return_value = mock_control

        result = play()

        self.assertTrue(result)
        mock_control.connect.assert_called_once()
        mock_control.set_play_state.assert_called_once_with(PlayState.PLAYING)

    @patch("src.producer.producer.ControlState")
    def test_pause_function(self, mock_control_state):
        """Test pause control function."""
        mock_control = Mock()
        mock_control.connect.return_value = True
        mock_control.set_play_state.return_value = True
        mock_control_state.return_value = mock_control

        result = pause()

        self.assertTrue(result)
        mock_control.set_play_state.assert_called_once_with(PlayState.PAUSED)

    @patch("src.producer.producer.ControlState")
    def test_stop_function(self, mock_control_state):
        """Test stop control function."""
        mock_control = Mock()
        mock_control.connect.return_value = True
        mock_control.set_play_state.return_value = True
        mock_control_state.return_value = mock_control

        result = stop()

        self.assertTrue(result)
        mock_control.set_play_state.assert_called_once_with(PlayState.STOPPED)

    @patch("src.producer.producer.ControlState")
    def test_control_function_failure(self, mock_control_state):
        """Test control function failure handling."""
        mock_control = Mock()
        mock_control.connect.return_value = False
        mock_control_state.return_value = mock_control

        result = play()

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
