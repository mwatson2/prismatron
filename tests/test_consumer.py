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

from src.const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
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
        with patch("src.consumer.consumer.FrameConsumer"), patch("src.consumer.consumer.ControlState"), patch(
            "src.consumer.consumer.LEDOptimizer"
        ), patch("src.consumer.consumer.WLEDClient"):
            self.consumer = ConsumerProcess(
                buffer_name="test_buffer",
                control_name="test_control",
                wled_host="192.168.1.100",
                wled_port=4048,
            )

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
        self.assertTrue(self.consumer.use_optimization)

    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.WLEDClient")
    def test_component_initialization_success(self, mock_wled, mock_optimizer):
        """Test successful component initialization."""
        # Setup mocks to return success
        mock_optimizer.return_value.initialize.return_value = True
        mock_wled.return_value.connect.return_value = True

        consumer = ConsumerProcess()
        result = consumer.initialize()

        self.assertTrue(result)
        mock_optimizer.return_value.initialize.assert_called_once()
        mock_wled.return_value.connect.assert_called_once()

    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.WLEDClient")
    def test_component_initialization_failure(self, mock_wled, mock_optimizer):
        """Test component initialization failure."""
        # Setup optimizer to fail
        mock_optimizer.return_value.initialize.return_value = False

        consumer = ConsumerProcess()
        result = consumer.initialize()

        self.assertFalse(result)

    def test_performance_settings_update(self):
        """Test updating performance settings."""
        self.consumer.set_performance_settings(target_fps=30.0, use_optimization=False, brightness_scale=0.8)

        self.assertEqual(self.consumer.target_fps, 30.0)
        self.assertFalse(self.consumer.use_optimization)
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

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    def test_process_frame_valid_input(self, mock_control, mock_frame_consumer):
        """Test processing a valid frame."""
        # Setup mock frame buffer
        mock_buffer_info = Mock()
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.uint8)
        mock_buffer_info.get_array.return_value = test_frame

        # Setup mock optimizer result
        mock_optimizer_result = Mock()
        mock_optimizer_result.converged = True
        mock_optimizer_result.iterations = 25
        mock_optimizer_result.led_values = (np.random.random((LED_COUNT, 3)) * 255).astype(np.float32)

        self.consumer._led_optimizer.optimize_frame.return_value = mock_optimizer_result

        # Setup mock transmission result
        mock_transmission_result = Mock()
        mock_transmission_result.success = True
        mock_transmission_result.errors = []

        self.consumer._wled_client.send_led_data.return_value = mock_transmission_result

        # Process the frame
        self.consumer._process_frame(mock_buffer_info)

        # Verify calls were made
        mock_buffer_info.get_array.assert_called_once()
        self.consumer._led_optimizer.optimize_frame.assert_called_once()
        self.consumer._wled_client.send_led_data.assert_called_once()

        # Check stats were updated
        self.assertEqual(self.consumer._stats.frames_processed, 1)
        self.assertGreater(self.consumer._stats.total_processing_time, 0)

    def test_process_frame_invalid_shape(self):
        """Test processing frame with invalid shape."""
        mock_buffer_info = Mock()
        # Wrong frame shape
        wrong_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        mock_buffer_info.get_array.return_value = wrong_frame

        # Process should handle gracefully
        self.consumer._process_frame(mock_buffer_info)

        # No optimization should have been called
        self.consumer._led_optimizer.optimize_frame.assert_not_called()

    def test_process_frame_optimization_failure(self):
        """Test handling optimization failures."""
        mock_buffer_info = Mock()
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.uint8)
        mock_buffer_info.get_array.return_value = test_frame

        # Setup optimizer to not converge
        mock_optimizer_result = Mock()
        mock_optimizer_result.converged = False
        mock_optimizer_result.iterations = 100
        mock_optimizer_result.led_values = np.zeros((LED_COUNT, 3), dtype=np.float32)

        self.consumer._led_optimizer.optimize_frame.return_value = mock_optimizer_result

        # Setup successful transmission
        mock_transmission_result = Mock()
        mock_transmission_result.success = True
        self.consumer._wled_client.send_led_data.return_value = mock_transmission_result

        # Process the frame
        self.consumer._process_frame(mock_buffer_info)

        # Should still process but increment error count
        self.assertEqual(self.consumer._stats.optimization_errors, 1)
        self.assertEqual(self.consumer._stats.frames_processed, 1)

    def test_process_frame_transmission_failure(self):
        """Test handling transmission failures."""
        mock_buffer_info = Mock()
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.uint8)
        mock_buffer_info.get_array.return_value = test_frame

        # Setup successful optimization
        mock_optimizer_result = Mock()
        mock_optimizer_result.converged = True
        mock_optimizer_result.led_values = (np.random.random((LED_COUNT, 3)) * 255).astype(np.float32)
        self.consumer._led_optimizer.optimize_frame.return_value = mock_optimizer_result

        # Setup transmission failure
        mock_transmission_result = Mock()
        mock_transmission_result.success = False
        mock_transmission_result.errors = ["Network timeout"]
        self.consumer._wled_client.send_led_data.return_value = mock_transmission_result

        # Process the frame
        self.consumer._process_frame(mock_buffer_info)

        # Should increment transmission error count
        self.assertEqual(self.consumer._stats.transmission_errors, 1)
        self.assertEqual(self.consumer._stats.frames_processed, 1)

    def test_process_frame_with_brightness_scaling(self):
        """Test frame processing with brightness scaling."""
        mock_buffer_info = Mock()
        test_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 4), 255, dtype=np.uint8)  # Bright frame
        mock_buffer_info.get_array.return_value = test_frame

        # Setup optimizer
        mock_optimizer_result = Mock()
        mock_optimizer_result.converged = True
        mock_optimizer_result.led_values = (np.random.random((LED_COUNT, 3)) * 255).astype(np.float32)
        self.consumer._led_optimizer.optimize_frame.return_value = mock_optimizer_result

        # Setup transmission
        mock_transmission_result = Mock()
        mock_transmission_result.success = True
        self.consumer._wled_client.send_led_data.return_value = mock_transmission_result

        # Set brightness scaling
        self.consumer.brightness_scale = 0.5

        # Process the frame
        self.consumer._process_frame(mock_buffer_info)

        # Verify optimization was called (brightness scaling applied internally)
        self.consumer._led_optimizer.optimize_frame.assert_called_once()
        call_args = self.consumer._led_optimizer.optimize_frame.call_args[0]
        processed_frame = call_args[0]

        # Frame should be scaled down (255 * 0.5 = 127.5 -> 127)
        self.assertTrue(np.all(processed_frame <= 127))

    def test_optimization_vs_speed_mode(self):
        """Test switching between full optimization and speed modes."""
        mock_buffer_info = Mock()
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.uint8)
        mock_buffer_info.get_array.return_value = test_frame

        # Setup mock results
        mock_optimizer_result = Mock()
        mock_optimizer_result.converged = True
        mock_optimizer_result.led_values = (np.random.random((LED_COUNT, 3)) * 255).astype(np.float32)

        self.consumer._led_optimizer.optimize_frame.return_value = mock_optimizer_result

        mock_transmission_result = Mock()
        mock_transmission_result.success = True
        self.consumer._wled_client.send_led_data.return_value = mock_transmission_result

        # Test full optimization mode (50 iterations)
        self.consumer.use_optimization = True
        self.consumer._process_frame(mock_buffer_info)

        # Check that optimize_frame was called with higher max_iterations
        call_args = self.consumer._led_optimizer.optimize_frame.call_args
        self.assertEqual(call_args[1]["max_iterations"], 50)

        # Reset mocks
        self.consumer._led_optimizer.reset_mock()

        # Test speed mode (5 iterations)
        self.consumer.use_optimization = False
        self.consumer._process_frame(mock_buffer_info)

        # Check that optimize_frame was called with lower max_iterations
        call_args = self.consumer._led_optimizer.optimize_frame.call_args
        self.assertEqual(call_args[1]["max_iterations"], 5)

    def test_get_stats(self):
        """Test getting consumer statistics."""
        # Update some stats
        self.consumer._stats.frames_processed = 50
        self.consumer._stats.total_processing_time = 5.0
        self.consumer._stats.optimization_errors = 2
        self.consumer._stats.transmission_errors = 1

        # Setup mock component stats
        self.consumer._wled_client.get_statistics.return_value = {"packets_sent": 100}
        self.consumer._led_optimizer.get_optimizer_stats.return_value = {"optimization_count": 50}

        stats = self.consumer.get_stats()

        # Verify key statistics
        self.assertEqual(stats["frames_processed"], 50)
        self.assertEqual(stats["average_fps"], 10.0)
        self.assertEqual(stats["optimization_errors"], 2)
        self.assertEqual(stats["transmission_errors"], 1)
        self.assertEqual(stats["target_fps"], 15.0)
        self.assertEqual(stats["led_count"], LED_COUNT)
        self.assertIn("wled_stats", stats)
        self.assertIn("optimizer_stats", stats)

    @patch("src.consumer.consumer.time.sleep")
    def test_process_loop_graceful_shutdown(self, mock_sleep):
        """Test graceful shutdown of processing loop."""
        # Setup mocks
        self.consumer._frame_consumer.wait_for_ready_buffer.side_effect = [None, None]
        self.consumer._control_state.should_shutdown.return_value = False

        # Start the process loop in a thread
        self.consumer._running = True
        loop_thread = threading.Thread(target=self.consumer._process_loop)
        loop_thread.start()

        # Give it a moment to start
        time.sleep(0.1)

        # Signal shutdown
        self.consumer._running = False

        # Wait for thread to finish
        loop_thread.join(timeout=1.0)
        self.assertFalse(loop_thread.is_alive())

    def test_process_loop_control_state_shutdown(self):
        """Test shutdown via control state signal."""
        # Setup control state to signal shutdown
        self.consumer._control_state.should_shutdown.return_value = True
        self.consumer._frame_consumer.wait_for_ready_buffer.return_value = None

        # Run one iteration of the loop
        self.consumer._running = True
        self.consumer._process_loop()

        # Loop should have detected shutdown signal
        self.consumer._control_state.should_shutdown.assert_called()

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
        self.consumer._wled_client.disconnect = Mock()
        self.consumer._frame_consumer.cleanup = Mock()

        # Call cleanup
        self.consumer._cleanup()

        # Verify cleanup was called
        self.consumer._wled_client.disconnect.assert_called_once()
        self.consumer._frame_consumer.cleanup.assert_called_once()

    def test_start_stop_integration(self):
        """Test starting and stopping the consumer process."""
        # Mock initialization to succeed
        with patch.object(self.consumer, "initialize", return_value=True):
            # Start the consumer
            result = self.consumer.start()
            self.assertTrue(result)
            self.assertTrue(self.consumer._running)
            self.assertIsNotNone(self.consumer._process_thread)

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
        ), patch("src.consumer.consumer.WLEDClient"):
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
    @patch("src.consumer.consumer.WLEDClient")
    def test_end_to_end_processing(self, mock_wled, mock_optimizer, mock_control, mock_frame_consumer):
        """Test complete end-to-end frame processing."""
        # Setup all components to succeed
        mock_optimizer.return_value.initialize.return_value = True
        mock_wled.return_value.connect.return_value = True

        # Setup frame data
        test_frame = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.uint8)
        mock_buffer_info = Mock()
        mock_buffer_info.get_array.return_value = test_frame

        mock_frame_consumer.return_value.wait_for_ready_buffer.return_value = mock_buffer_info
        mock_control.return_value.should_shutdown.return_value = False

        # Setup optimization result
        optimization_result = Mock()
        optimization_result.converged = True
        optimization_result.led_values = (np.random.random((LED_COUNT, 3)) * 255).astype(np.float32)
        mock_optimizer.return_value.optimize_frame.return_value = optimization_result

        # Setup transmission result
        transmission_result = Mock()
        transmission_result.success = True
        mock_wled.return_value.send_led_data.return_value = transmission_result

        # Create and initialize consumer
        consumer = ConsumerProcess()
        self.assertTrue(consumer.initialize())

        # Process one frame
        consumer._process_frame(mock_buffer_info)

        # Verify the pipeline executed
        mock_buffer_info.get_array.assert_called_once()
        mock_optimizer.return_value.optimize_frame.assert_called_once()
        mock_wled.return_value.send_led_data.assert_called_once()

        # Check stats
        self.assertEqual(consumer._stats.frames_processed, 1)
        self.assertGreater(consumer._stats.total_processing_time, 0)


if __name__ == "__main__":
    unittest.main()
