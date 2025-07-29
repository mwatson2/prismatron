"""
Unit tests for the WLED Client Module.

Tests DDP protocol implementation, UDP communication, error handling,
and performance monitoring for the consolidated WLED client.
"""

import os
import socket
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.const import DDP_HEADER_SIZE, LED_COUNT
from src.consumer.wled_sink import TransmissionResult, WLEDSink, WLEDSinkConfig


class TestTransmissionResult(unittest.TestCase):
    """Test cases for TransmissionResult class."""

    def test_initialization(self):
        """Test transmission result initialization."""
        result = TransmissionResult(
            success=True,
            packets_sent=5,
            bytes_sent=1000,
            transmission_time=0.1,
            errors=[],
        )

        self.assertTrue(result.success)
        self.assertEqual(result.packets_sent, 5)
        self.assertEqual(result.bytes_sent, 1000)
        self.assertEqual(result.transmission_time, 0.1)

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        result = TransmissionResult(
            success=True,
            packets_sent=1,
            bytes_sent=1000,  # 1000 bytes
            transmission_time=0.1,  # 0.1 seconds
            errors=[],
        )

        # 1000 bytes * 8 bits / 0.1 seconds / 1_000_000 = 0.08 Mbps
        expected_mbps = (1000 * 8) / (0.1 * 1_000_000)
        self.assertAlmostEqual(result.get_throughput_mbps(), expected_mbps, places=6)

    def test_zero_time_throughput(self):
        """Test throughput with zero transmission time."""
        result = TransmissionResult(
            success=True,
            packets_sent=1,
            bytes_sent=1000,
            transmission_time=0.0,
            errors=[],
        )

        self.assertEqual(result.get_throughput_mbps(), 0.0)


class TestWLEDSinkConfig(unittest.TestCase):
    """Test cases for WLEDSinkConfig class."""

    def test_default_initialization(self):
        """Test config with default values."""
        config = WLEDSinkConfig()

        self.assertEqual(config.led_count, LED_COUNT)
        self.assertEqual(config.max_fps, 60.0)
        self.assertFalse(config.persistent_retry)

    def test_custom_initialization(self):
        """Test config with custom values."""
        config = WLEDSinkConfig(
            host="192.168.7.140",
            port=4048,
            led_count=1000,
            max_fps=30.0,
            persistent_retry=True,
        )

        self.assertEqual(config.host, "192.168.7.140")
        self.assertEqual(config.port, 4048)
        self.assertEqual(config.led_count, 1000)
        self.assertEqual(config.max_fps, 30.0)
        self.assertTrue(config.persistent_retry)


class TestWLEDSink(unittest.TestCase):
    """Test cases for WLEDSink class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = WLEDSinkConfig(host="192.168.7.140", port=4048, timeout=1.0)
        self.client = WLEDSink(self.config)

    def tearDown(self):
        """Clean up after tests."""
        if self.client.is_connected:
            self.client.disconnect()

    def test_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.config.host, "192.168.7.140")
        self.assertEqual(self.client.config.port, 4048)
        self.assertFalse(self.client.is_connected)
        self.assertEqual(self.client.sequence_number, 0)

    @patch("socket.socket")
    def test_connection_success(self, mock_socket_class):
        """Test successful connection."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Mock successful query response
        with patch.object(self.client, "_send_query", return_value=(True, {"name": "WLED"})):
            result = self.client.connect()

        self.assertTrue(result)
        self.assertTrue(self.client.is_connected)
        mock_socket_class.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)
        mock_socket.settimeout.assert_called_once_with(1.0)

    @patch("socket.socket")
    def test_connection_failure(self, mock_socket_class):
        """Test connection failure."""
        mock_socket_class.side_effect = Exception("Connection failed")

        result = self.client.connect()

        self.assertFalse(result)
        self.assertFalse(self.client.is_connected)

    @patch("socket.socket")
    def test_disconnect(self, mock_socket_class):
        """Test disconnection."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        with patch.object(self.client, "_send_query", return_value=(True, {})):
            self.client.connect()
        self.client.disconnect()

        self.assertFalse(self.client.is_connected)
        mock_socket.close.assert_called_once()

    @patch("socket.socket")
    def test_led_data_validation_bytes(self, mock_socket_class):
        """Test LED data validation with bytes input."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        with patch.object(self.client, "_send_query", return_value=(True, {})):
            self.client.connect()

        # Test with wrong size
        wrong_data = b"RGB" * 100  # Wrong LED count
        result = self.client.send_led_data(wrong_data)

        self.assertFalse(result.success)
        self.assertIn("Invalid LED data size", result.errors[0])

    @patch("socket.socket")
    def test_led_data_validation_numpy(self, mock_socket_class):
        """Test LED data validation with numpy input."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        with patch.object(self.client, "_send_query", return_value=(True, {})):
            self.client.connect()

        # Test with wrong shape
        wrong_data = np.zeros((100, 3))  # Wrong LED count
        result = self.client.send_led_data(wrong_data)

        self.assertFalse(result.success)
        self.assertIn("Invalid LED data shape", result.errors[0])

    @patch("socket.socket")
    def test_successful_data_transmission_bytes(self, mock_socket_class):
        """Test successful LED data transmission with bytes."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        with patch.object(self.client, "_send_query", return_value=(True, {})):
            self.client.connect()

        # Create valid LED data
        led_data = b"RGB" * self.config.led_count

        result = self.client.send_led_data(led_data)

        self.assertTrue(result.success)
        self.assertGreater(result.packets_sent, 0)
        self.assertGreater(result.bytes_sent, 0)

    @patch("socket.socket")
    def test_successful_data_transmission_numpy(self, mock_socket_class):
        """Test successful LED data transmission with numpy array."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        with patch.object(self.client, "_send_query", return_value=(True, {})):
            self.client.connect()

        # Create valid LED data
        led_data = np.random.randint(0, 255, (self.config.led_count, 3)).astype(np.uint8)

        result = self.client.send_led_data(led_data)

        self.assertTrue(result.success)
        self.assertGreater(result.packets_sent, 0)
        self.assertGreater(result.bytes_sent, 0)

    def test_led_data_without_connection(self):
        """Test sending LED data without connection."""
        led_data = np.random.randint(0, 255, (self.config.led_count, 3)).astype(np.uint8)

        result = self.client.send_led_data(led_data)

        self.assertFalse(result.success)
        self.assertIn("Not connected", result.errors[0])

    @patch("socket.socket")
    def test_solid_color(self, mock_socket_class):
        """Test setting solid color."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        with patch.object(self.client, "_send_query", return_value=(True, {})):
            self.client.connect()

        result = self.client.set_solid_color(255, 128, 0)

        self.assertTrue(result.success)
        self.assertGreater(result.packets_sent, 0)

    def test_solid_color_invalid_values(self):
        """Test solid color with invalid RGB values."""
        result = self.client.set_solid_color(-10, 300, 128)

        self.assertFalse(result.success)
        self.assertIn("Invalid RGB values", result.errors[0])

    @patch("socket.socket")
    def test_test_patterns(self, mock_socket_class):
        """Test test pattern generation and transmission."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        with patch.object(self.client, "_send_query", return_value=(True, {})):
            self.client.connect()

        # Test different patterns
        patterns = ["rainbow", "solid", "off"]

        for pattern in patterns:
            result = self.client.send_test_pattern(pattern)
            self.assertTrue(result.success, f"Pattern '{pattern}' failed")

    def test_statistics_collection(self):
        """Test statistics collection."""
        stats = self.client.get_statistics()

        # Check that all expected keys are present
        expected_keys = [
            "frames_sent",
            "packets_sent",
            "transmission_errors",
            "is_connected",
            "host",
            "port",
            "led_count",
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

    def test_statistics_reset(self):
        """Test statistics reset."""
        # Manually set some statistics
        self.client.frames_sent = 10
        self.client.packets_sent = 100

        self.client.reset_statistics()

        self.assertEqual(self.client.frames_sent, 0)
        self.assertEqual(self.client.packets_sent, 0)

    @patch("socket.socket")
    def test_context_manager(self, mock_socket_class):
        """Test context manager functionality."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        with patch.object(self.client, "_send_query", return_value=(True, {})), self.client as client:
            self.assertTrue(client.is_connected)

        self.assertFalse(self.client.is_connected)

    def test_context_manager_connection_failure(self):
        """Test context manager with connection failure."""
        with patch.object(self.client, "connect", return_value=False), self.assertRaises(ConnectionError), self.client:
            pass

    def test_fragmentation_calculation(self):
        """Test LED data fragmentation calculation."""
        # The client should automatically calculate fragmentation
        self.assertGreater(self.client.packets_per_frame, 0)
        self.assertGreater(self.client.data_per_packet, 0)
        self.assertGreater(self.client.last_packet_size, 0)


class TestWLEDSinkIntegration(unittest.TestCase):
    """Integration tests for WLED client."""

    @patch("socket.socket")
    def test_end_to_end_transmission(self, mock_socket_class):
        """Test complete end-to-end LED data transmission."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        config = WLEDSinkConfig(host="192.168.7.140", port=4048)
        client = WLEDSink(config)

        # Connect and send test data
        with patch.object(client, "_send_query", return_value=(True, {"name": "WLED"})):
            self.assertTrue(client.connect())

        # Create realistic LED data
        led_data = np.zeros((config.led_count, 3), dtype=np.uint8)
        led_data[:100] = [255, 0, 0]  # First 100 LEDs red
        led_data[100:200] = [0, 255, 0]  # Next 100 LEDs green
        led_data[200:300] = [0, 0, 255]  # Next 100 LEDs blue

        # Send data
        result = client.send_led_data(led_data)

        # Verify transmission
        self.assertTrue(result.success)
        self.assertGreater(result.packets_sent, 0)
        self.assertGreater(result.bytes_sent, 0)

        # Check that socket.sendto was called
        self.assertGreater(mock_socket.sendto.call_count, 0)

        client.disconnect()


if __name__ == "__main__":
    unittest.main()
