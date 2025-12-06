"""
Unit tests for the Consumer Process.

Tests consumer process initialization, frame processing, LED optimization
integration, WLED communication, and performance monitoring.
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.const import FRAME_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH

# LED_COUNT is now dynamic from pattern files; use test constant
LED_COUNT = 2624
from src.consumer.consumer import ConsumerProcess, ConsumerStats


class TestConsumerStats(unittest.TestCase):
    """Test cases for ConsumerStats class."""

    def test_initialization(self):
        """Test consumer stats initialization."""
        stats = ConsumerStats()

        self.assertEqual(stats.frames_processed, 0)
        self.assertEqual(stats.total_processing_time, 0.0)
        self.assertEqual(stats.optimization_errors, 0)
        self.assertEqual(stats.transmission_errors, 0)

    def test_fps_calculations(self):
        """Test FPS calculation methods."""
        stats = ConsumerStats(
            frames_processed=100,
            total_processing_time=10.0,
            total_optimization_time=2.0,
        )

        self.assertEqual(stats.get_average_fps(), 10.0)
        self.assertEqual(stats.get_average_optimization_time(), 0.02)

        # Test edge cases
        empty_stats = ConsumerStats()
        self.assertEqual(empty_stats.get_average_fps(), 0.0)
        self.assertEqual(empty_stats.get_average_optimization_time(), 0.0)


class TestConsumerProcess(unittest.TestCase):
    """Test cases for ConsumerProcess class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create consumer with mocked dependencies
        with patch("src.consumer.consumer.FrameConsumer") as mock_frame_consumer, patch(
            "src.consumer.consumer.ControlState"
        ) as mock_control_state, patch("src.consumer.consumer.LEDOptimizer") as mock_led_optimizer, patch(
            "src.consumer.consumer.WLEDSink"
        ) as mock_wled_sink, patch(
            "src.consumer.consumer.LEDBuffer"
        ) as mock_led_buffer, patch(
            "src.consumer.consumer.FrameRenderer"
        ) as mock_frame_renderer:

            # Configure mocks to return success for initialization
            mock_frame_consumer.return_value.connect.return_value = True
            mock_led_optimizer.return_value.initialize.return_value = True
            mock_wled_sink.return_value.connect.return_value = True
            mock_wled_sink.return_value.get_statistics.return_value = {"packets_sent": 0}
            mock_led_optimizer.return_value.get_optimizer_stats.return_value = {"led_count": 100}
            mock_led_buffer.return_value.get_buffer_stats.return_value = {"size": 10}
            mock_frame_renderer.return_value.get_renderer_stats.return_value = {"frames_rendered": 0}
            mock_frame_renderer.return_value.is_initialized.return_value = True

            self.consumer = ConsumerProcess(
                buffer_name="test_buffer",
                control_name="test_control",
                wled_hosts="192.168.7.140",
                wled_port=4048,
            )

            # Store mock references for use in tests
            self.mock_frame_consumer = mock_frame_consumer.return_value
            self.mock_led_optimizer = mock_led_optimizer.return_value
            self.mock_wled_sink = mock_wled_sink.return_value
            self.mock_led_buffer = mock_led_buffer.return_value
            self.mock_frame_renderer = mock_frame_renderer.return_value

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.consumer, "_running") and self.consumer._running:
            self.consumer.stop()

        # Clean up temp files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test consumer process initialization."""
        self.assertEqual(self.consumer.buffer_name, "test_buffer")
        self.assertEqual(self.consumer.control_name, "test_control")
        self.assertFalse(self.consumer._running)
        self.assertFalse(self.consumer._shutdown_requested)
        self.assertEqual(self.consumer.target_fps, 15.0)

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.WLEDSink")
    @patch("src.consumer.consumer.LEDBuffer")
    @patch("src.consumer.consumer.FrameRenderer")
    def test_component_initialization_success(
        self, mock_frame_renderer, mock_led_buffer, mock_wled, mock_optimizer, mock_control_state, mock_frame_consumer
    ):
        """Test successful component initialization."""
        # Setup mocks to return success
        mock_frame_consumer.return_value.connect.return_value = True
        mock_optimizer.return_value.initialize.return_value = True
        mock_wled.return_value.connect.return_value = True
        mock_wled.return_value.get_statistics.return_value = {"packets_sent": 0}
        mock_optimizer.return_value.get_optimizer_stats.return_value = {"led_count": 100}
        mock_led_buffer.return_value.get_buffer_stats.return_value = {"size": 10}
        mock_frame_renderer.return_value.get_renderer_stats.return_value = {"frames_rendered": 0}
        mock_frame_renderer.return_value.is_initialized.return_value = True

        consumer = ConsumerProcess()
        result = consumer.initialize()

        self.assertTrue(result)
        mock_frame_consumer.return_value.connect.assert_called_once()
        mock_optimizer.return_value.initialize.assert_called_once()
        mock_wled.return_value.connect.assert_called_once()

    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.WLEDSink")
    def test_component_initialization_failure(self, mock_wled, mock_optimizer):
        """Test component initialization failure."""
        # Setup optimizer to fail
        mock_optimizer.return_value.initialize.return_value = False

        consumer = ConsumerProcess()
        result = consumer.initialize()

        self.assertFalse(result)

    def test_performance_settings_update(self):
        """Test updating performance settings."""
        self.consumer.set_performance_settings(target_fps=30.0, brightness_scale=0.8)

        self.assertEqual(self.consumer.target_fps, 30.0)
        self.assertEqual(self.consumer.brightness_scale, 0.8)

    def test_performance_settings_bounds(self):
        """Test performance settings bounds checking."""
        # Test FPS bounds
        self.consumer.set_performance_settings(target_fps=100.0)
        self.assertEqual(self.consumer.target_fps, 60.0)

        self.consumer.set_performance_settings(target_fps=0.5)
        self.assertEqual(self.consumer.target_fps, 1.0)

        # Test brightness bounds
        self.consumer.set_performance_settings(brightness_scale=1.5)
        self.assertEqual(self.consumer.brightness_scale, 1.0)

        self.consumer.set_performance_settings(brightness_scale=-0.1)
        self.assertEqual(self.consumer.brightness_scale, 0.0)

    def test_get_stats(self):
        """Test getting consumer statistics."""
        # Update some stats
        self.consumer._stats.frames_processed = 50
        self.consumer._stats.total_processing_time = 5.0
        self.consumer._stats.optimization_errors = 2
        self.consumer._stats.transmission_errors = 1

        # Setup mock component stats
        self.mock_wled_sink.get_statistics.return_value = {"packets_sent": 100}
        self.mock_led_optimizer.get_optimizer_stats.return_value = {"optimization_count": 50}

        stats = self.consumer.get_stats()

        # Verify key statistics
        self.assertEqual(stats["frames_processed"], 50)
        self.assertEqual(stats["average_optimization_fps"], 10.0)
        self.assertEqual(stats["optimization_errors"], 2)
        self.assertEqual(stats["transmission_errors"], 1)
        self.assertEqual(stats["target_fps"], 15.0)
        self.assertEqual(stats["led_count"], LED_COUNT)
        self.assertIn("wled_stats", stats)
        self.assertIn("optimizer_stats", stats)

    def test_frame_rate_limiting(self):
        """Test frame rate limiting behavior in performance settings."""
        # Test frame rate limiting configuration
        self.consumer.set_performance_settings(target_fps=30.0)
        self.assertEqual(self.consumer.target_fps, 30.0)

        # Test frame rate bounds
        self.consumer.set_performance_settings(target_fps=100.0)
        self.assertEqual(self.consumer.target_fps, 60.0)  # Should be clamped to max

        self.consumer.set_performance_settings(target_fps=0.5)
        self.assertEqual(self.consumer.target_fps, 1.0)  # Should be clamped to min

        # Verify target frame time calculation would work
        target_frame_time = 1.0 / self.consumer.target_fps
        self.assertAlmostEqual(target_frame_time, 1.0, places=1)

    def test_cleanup(self):
        """Test resource cleanup."""
        # Setup cleanup calls
        self.mock_wled_sink.disconnect = Mock()
        self.consumer._frame_consumer.cleanup = Mock()

        # Call cleanup
        self.consumer._cleanup()

        # Verify cleanup was called
        self.mock_wled_sink.disconnect.assert_called_once()
        self.consumer._frame_consumer.cleanup.assert_called_once()

    def test_start_stop_integration(self):
        """Test starting and stopping the consumer process."""
        # Mock initialization to succeed
        with patch.object(self.consumer, "initialize", return_value=True):
            # Start the consumer
            result = self.consumer.start()
            self.assertTrue(result)
            self.assertTrue(self.consumer._running)
            self.assertIsNotNone(self.consumer._optimization_thread)

            # Stop the consumer
            self.consumer.stop()
            self.assertFalse(self.consumer._running)
            self.assertTrue(self.consumer._shutdown_requested)

    def test_start_initialization_failure(self):
        """Test start failure when initialization fails."""
        with patch.object(self.consumer, "initialize", return_value=False):
            result = self.consumer.start()
            self.assertFalse(result)
            self.assertFalse(self.consumer._running)

    def test_start_already_running(self):
        """Test starting when already running."""
        self.consumer._running = True

        result = self.consumer.start()
        self.assertTrue(result)  # Should return True but not start again

    @patch("src.consumer.consumer.signal.signal")
    def test_signal_handler_setup(self, mock_signal):
        """Test signal handler setup."""
        import signal

        # Create new consumer to trigger signal setup
        with patch("src.consumer.consumer.FrameConsumer"), patch("src.consumer.consumer.ControlState"), patch(
            "src.consumer.consumer.LEDOptimizer"
        ), patch("src.consumer.consumer.WLEDSink"):
            consumer = ConsumerProcess()

        # Verify signal handlers were registered
        signal_calls = mock_signal.call_args_list
        self.assertTrue(any(call[0][0] == signal.SIGINT for call in signal_calls))
        self.assertTrue(any(call[0][0] == signal.SIGTERM for call in signal_calls))


class TestConsumerProcessIntegration(unittest.TestCase):
    """Integration tests for consumer process."""

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.WLEDSink")
    @patch("src.consumer.consumer.LEDBuffer")
    @patch("src.consumer.consumer.FrameRenderer")
    def test_end_to_end_processing(
        self, mock_frame_renderer, mock_led_buffer, mock_wled, mock_optimizer, mock_control, mock_frame_consumer
    ):
        """Test complete end-to-end frame processing."""
        # Setup all components to succeed
        mock_frame_consumer.return_value.connect.return_value = True
        mock_optimizer.return_value.initialize.return_value = True
        mock_wled.return_value.connect.return_value = True
        mock_wled.return_value.get_statistics.return_value = {"packets_sent": 0}
        mock_optimizer.return_value.get_optimizer_stats.return_value = {"led_count": 100}
        mock_led_buffer.return_value.get_buffer_stats.return_value = {"size": 10}
        mock_led_buffer.return_value.write_led_values.return_value = True
        mock_led_buffer.return_value.read_latest_led_values.return_value = Mock(
            led_values=np.zeros((LED_COUNT, 3), dtype=np.uint8)
        )
        mock_frame_renderer.return_value.get_renderer_stats.return_value = {"frames_rendered": 0}
        mock_frame_renderer.return_value.is_initialized.return_value = True

        # Setup frame data
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8)
        mock_buffer_info = Mock()
        mock_buffer_info.get_array_interleaved.return_value = test_frame
        mock_buffer_info.presentation_timestamp = time.time()

        mock_frame_consumer.return_value.wait_for_ready_buffer.return_value = mock_buffer_info
        mock_control.return_value.should_shutdown.return_value = False

        # Setup optimization result
        optimization_result = Mock()
        optimization_result.converged = True
        optimization_result.iterations = 25
        optimization_result.led_values = (np.random.random((LED_COUNT, 3)) * 255).astype(np.float32)
        optimization_result.error_metrics = {"mse": 0.1}
        mock_optimizer.return_value.optimize_frame.return_value = optimization_result

        # Setup transmission result
        transmission_result = Mock()
        transmission_result.success = True
        mock_wled.return_value.send_led_data.return_value = transmission_result

        # Create and initialize consumer
        consumer = ConsumerProcess()
        self.assertTrue(consumer.initialize())

        # Process one frame
        frame_dropped = consumer._process_frame_optimization(mock_buffer_info)

        # Verify the pipeline executed
        mock_buffer_info.get_array_interleaved.assert_called_once_with(FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS)
        mock_optimizer.return_value.optimize_frame.assert_called_once()

        # Check stats
        self.assertEqual(consumer._stats.frames_processed, 1)
        self.assertGreater(consumer._stats.total_processing_time, 0)


if __name__ == "__main__":
    unittest.main()
