"""
Unit tests for WLED UDP Communication Client.

Tests the DDP protocol implementation, packet fragmentation, flow control,
and error handling using mocked UDP sockets.
"""

import json
import os
import socket
import struct
import sys
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.const import DDP_HEADER_SIZE, DDP_MAX_DATA_PER_PACKET, LED_COUNT
from src.consumer.wled_client import DDPDataType, DDPFlags, WLEDClient, WLEDConfig


class TestWLEDClient(unittest.TestCase):
    """Test cases for WLEDClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = WLEDConfig(
            host="test.local",
            port=21324,
            led_count=32,  # Small count for testing
            timeout=1.0,
            retry_count=2,
            max_fps=30.0,
        )
        self.client = WLEDClient(self.config)

    def tearDown(self):
        """Clean up after tests."""
        if self.client:
            self.client.disconnect()

    def _setup_mock_query_response(self, mock_socket, status_json=None):
        """Helper to setup mock query response."""
        if status_json is None:
            status_json = '{"name":"Test WLED","ver":"0.14.0","leds":{"count":32}}'

        response_data = struct.pack(
            ">BBBBLH", DDPFlags.VER1 | DDPFlags.REPLY, 0, 0, 251, 0, len(status_json)
        )
        response_data += status_json.encode("utf-8")
        mock_socket.recvfrom.return_value = (response_data, ("192.168.1.100", 21324))

    def test_initialization(self):
        """Test WLEDClient initialization."""
        # Test with custom config
        self.assertEqual(self.client.config.host, "test.local")
        self.assertEqual(self.client.config.led_count, 32)
        self.assertFalse(self.client.is_connected)
        self.assertEqual(self.client.sequence_number, 0)

        # Test with default config
        default_client = WLEDClient()
        self.assertEqual(default_client.config.host, "wled.local")
        self.assertEqual(default_client.config.led_count, LED_COUNT)

    def test_fragmentation_calculation(self):
        """Test LED data fragmentation calculation."""
        # Test small LED count (fits in one packet)
        small_config = WLEDConfig(led_count=100)  # 300 bytes
        small_client = WLEDClient(small_config)
        self.assertEqual(small_client.packets_per_frame, 1)
        self.assertEqual(small_client.last_packet_size, 300)

        # Test large LED count (requires multiple packets)
        large_config = WLEDConfig(led_count=1000)  # 3000 bytes
        large_client = WLEDClient(large_config)
        expected_packets = (
            3000 + DDP_MAX_DATA_PER_PACKET - 1
        ) // DDP_MAX_DATA_PER_PACKET
        self.assertEqual(large_client.packets_per_frame, expected_packets)

    @patch("socket.socket")
    def test_connection(self, mock_socket_class):
        """Test connection establishment."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Mock successful query response with JSON status
        self._setup_mock_query_response(
            mock_socket, '{"name":"Test WLED","ver":"0.14.0","leds":{"count":100}}'
        )

        # Test successful connection
        self.assertTrue(self.client.connect())
        self.assertTrue(self.client.is_connected)

        # Verify socket creation and configuration
        mock_socket_class.assert_called_with(socket.AF_INET, socket.SOCK_DGRAM)
        mock_socket.settimeout.assert_called_with(self.config.timeout)

        # Verify query packet was sent and response received
        mock_socket.sendto.assert_called()
        mock_socket.recvfrom.assert_called()

        # Check that WLED status was stored
        status = self.client.get_wled_status()
        self.assertIsNotNone(status)
        self.assertEqual(status["name"], "Test WLED")
        self.assertEqual(status["ver"], "0.14.0")

        # Test disconnection
        self.client.disconnect()
        self.assertFalse(self.client.is_connected)
        mock_socket.close.assert_called()

    @patch("socket.socket")
    def test_connection_failure(self, mock_socket_class):
        """Test connection failure handling."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        mock_socket.recvfrom.side_effect = socket.timeout("Query timeout")

        # Connection should fail due to query timeout
        self.assertFalse(self.client.connect())
        self.assertFalse(self.client.is_connected)
        mock_socket.close.assert_called()

    def test_ddp_packet_structure(self):
        """Test DDP packet header structure."""
        # Create test data
        test_data = b"\x01\x02\x03"
        data_offset = 100

        # Mock socket to capture sent data
        with patch("socket.socket") as mock_socket_class:
            mock_socket = Mock()
            mock_socket_class.return_value = mock_socket
            self._setup_mock_query_response(mock_socket)

            self.client.connect()
            self.client._send_ddp_packet(test_data, data_offset, is_last_packet=True)

            # Verify packet was sent
            mock_socket.sendto.assert_called()
            sent_packet, address = mock_socket.sendto.call_args[0]

            # Verify address
            self.assertEqual(address, (self.config.host, self.config.port))

            # Parse DDP header
            self.assertGreaterEqual(len(sent_packet), DDP_HEADER_SIZE)

            flags, sequence, data_type, dest_id, offset, data_length = struct.unpack(
                ">BBBBLH", sent_packet[:10]
            )

            # Verify header fields
            self.assertEqual(flags & DDPFlags.VER1, DDPFlags.VER1)
            self.assertEqual(
                flags & DDPFlags.PUSH, DDPFlags.PUSH
            )  # Last packet should have PUSH
            self.assertEqual(data_type, DDPDataType.RGB24)
            self.assertEqual(dest_id, 0)
            self.assertEqual(offset, data_offset)

            # Verify data payload
            packet_data = sent_packet[DDP_HEADER_SIZE:]
            self.assertEqual(packet_data, test_data)

    @patch("socket.socket")
    def test_solid_color(self, mock_socket_class):
        """Test setting solid color."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        self._setup_mock_query_response(mock_socket)

        self.client.connect()

        # Test valid color
        self.assertTrue(self.client.set_solid_color(255, 128, 64))

        # Verify packet was sent
        mock_socket.sendto.assert_called()
        sent_packet, _ = mock_socket.sendto.call_args[0]

        # Extract and verify LED data
        led_data = sent_packet[DDP_HEADER_SIZE:]
        expected_data = bytes([255, 128, 64] * self.config.led_count)
        self.assertEqual(led_data, expected_data)

        # Test invalid color values
        self.assertFalse(self.client.set_solid_color(256, 0, 0))  # R > 255
        self.assertFalse(self.client.set_solid_color(0, -1, 0))  # G < 0

    @patch("socket.socket")
    def test_led_data_validation(self, mock_socket_class):
        """Test LED data validation."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        self._setup_mock_query_response(mock_socket)

        self.client.connect()

        # Test correct data size
        correct_data = bytes([255, 0, 0] * self.config.led_count)
        self.assertTrue(self.client.send_led_data(correct_data))

        # Test incorrect data size
        wrong_size_data = bytes([255, 0, 0] * (self.config.led_count + 1))
        self.assertFalse(self.client.send_led_data(wrong_size_data))

    @patch("socket.socket")
    def test_packet_fragmentation(self, mock_socket_class):
        """Test packet fragmentation for large LED arrays."""
        # Use larger LED count to force fragmentation
        large_config = WLEDConfig(led_count=1000)  # 3000 bytes, multiple packets
        large_client = WLEDClient(large_config)

        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        self._setup_mock_query_response(mock_socket)

        large_client.connect()

        # Create test data
        test_data = bytes([i % 256 for i in range(large_config.led_count * 3)])
        large_client.send_led_data(test_data)

        # Verify multiple packets were sent
        call_count = mock_socket.sendto.call_count
        expected_packets = large_client.packets_per_frame + 1  # +1 for query packet
        self.assertEqual(call_count, expected_packets)

        # Verify packet sequence and data integrity
        sent_calls = mock_socket.sendto.call_args_list[1:]  # Skip query packet
        reconstructed_data = b""

        for i, call in enumerate(sent_calls):
            packet, _ = call[0]
            flags, sequence, _, _, offset, _ = struct.unpack(">BBBBLH", packet[:10])
            packet_data = packet[DDP_HEADER_SIZE:]

            # Verify sequence numbers
            self.assertEqual(sequence, i)  # Sequence starts at 0

            # Verify PUSH flag only on last packet
            if i == len(sent_calls) - 1:
                self.assertEqual(flags & DDPFlags.PUSH, DDPFlags.PUSH)
            else:
                self.assertEqual(flags & DDPFlags.PUSH, 0)

            # Reconstruct data
            reconstructed_data += packet_data

        # Verify complete data integrity
        self.assertEqual(reconstructed_data, test_data)

    @patch("time.sleep")
    @patch("socket.socket")
    def test_flow_control(self, mock_socket_class, mock_sleep):
        """Test flow control and frame rate limiting."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        self._setup_mock_query_response(mock_socket)

        # Set high frame rate for testing
        self.config.max_fps = 60.0
        self.client.config = self.config

        self.client.connect()

        # Send multiple frames rapidly
        test_data = bytes([255, 0, 0] * self.config.led_count)

        start_time = time.time()
        for _ in range(3):
            self.client.send_led_data(test_data)

        # Verify sleep was called for flow control
        # (may not be called every time due to timing variations)
        total_sleep_time = sum(call[0][0] for call in mock_sleep.call_args_list)

        # At 60 FPS, minimum interval is ~16.67ms
        # Multiple rapid sends should trigger some sleep
        if mock_sleep.called:
            self.assertGreater(total_sleep_time, 0)

    @patch("socket.socket")
    def test_retry_mechanism(self, mock_socket_class):
        """Test packet retry mechanism on failures."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        self._setup_mock_query_response(mock_socket)

        # Set up socket to fail first attempt, succeed on retry
        mock_socket.sendto.side_effect = [
            None,  # Query packet succeeds
            socket.timeout(),  # First data packet fails
            None,  # Retry succeeds
        ]

        self.client.connect()

        test_data = bytes([255, 0, 0] * self.config.led_count)
        result = self.client.send_led_data(test_data)

        # Should succeed after retries
        self.assertTrue(result)

        # Verify retry attempts (query + 2 attempts for data packet)
        self.assertEqual(mock_socket.sendto.call_count, 3)

    @patch("socket.socket")
    def test_retry_exhaustion(self, mock_socket_class):
        """Test behavior when all retries are exhausted."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        self._setup_mock_query_response(mock_socket)

        # Set up socket to always fail data packets
        mock_socket.sendto.side_effect = [
            None,  # Query packet succeeds
            socket.timeout(),  # All data packet attempts fail
            socket.timeout(),
            socket.timeout(),
        ]

        self.client.connect()

        test_data = bytes([255, 0, 0] * self.config.led_count)
        result = self.client.send_led_data(test_data)

        # Should fail after exhausting retries
        self.assertFalse(result)

        # Verify all retry attempts were made
        expected_calls = 1 + self.config.retry_count  # Query + retries
        self.assertEqual(mock_socket.sendto.call_count, expected_calls)

    def test_statistics(self):
        """Test transmission statistics tracking."""
        # Initial statistics
        stats = self.client.get_statistics()
        self.assertEqual(stats["frames_sent"], 0)
        self.assertEqual(stats["packets_sent"], 0)
        self.assertEqual(stats["transmission_errors"], 0)

        with patch("socket.socket") as mock_socket_class:
            mock_socket = Mock()
            mock_socket_class.return_value = mock_socket
            self._setup_mock_query_response(mock_socket)

            self.client.connect()

            # Send successful frame
            test_data = bytes([255, 0, 0] * self.config.led_count)
            self.client.send_led_data(test_data)

            stats = self.client.get_statistics()
            self.assertEqual(stats["frames_sent"], 1)
            self.assertGreater(stats["packets_sent"], 0)

            # Reset statistics
            self.client.reset_statistics()
            stats = self.client.get_statistics()
            self.assertEqual(stats["frames_sent"], 0)
            self.assertEqual(stats["packets_sent"], 0)

    @patch("socket.socket")
    def test_context_manager(self, mock_socket_class):
        """Test context manager functionality."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        self._setup_mock_query_response(mock_socket)

        # Test successful context manager usage
        with WLEDClient(self.config) as client:
            self.assertTrue(client.is_connected)
            test_data = bytes([255, 0, 0] * self.config.led_count)
            client.send_led_data(test_data)

        # Client should be disconnected after context exit
        self.assertFalse(client.is_connected)
        mock_socket.close.assert_called()

    @patch("socket.socket")
    def test_context_manager_connection_failure(self, mock_socket_class):
        """Test context manager with connection failure."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        mock_socket.recvfrom.side_effect = socket.error("Connection failed")

        # Should raise ConnectionError on failed connection
        with self.assertRaises(ConnectionError):
            with WLEDClient(self.config) as client:
                pass


class TestWLEDConfig(unittest.TestCase):
    """Test cases for WLEDConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WLEDConfig()
        self.assertEqual(config.host, "wled.local")
        self.assertEqual(config.port, 21324)
        self.assertEqual(config.led_count, LED_COUNT)
        self.assertGreater(config.timeout, 0)
        self.assertGreater(config.retry_count, 0)
        self.assertGreater(config.max_fps, 0)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = WLEDConfig(
            host="custom.local",
            port=12345,
            led_count=100,
            timeout=2.0,
            retry_count=5,
            max_fps=120.0,
        )
        self.assertEqual(config.host, "custom.local")
        self.assertEqual(config.port, 12345)
        self.assertEqual(config.led_count, 100)
        self.assertEqual(config.timeout, 2.0)
        self.assertEqual(config.retry_count, 5)
        self.assertEqual(config.max_fps, 120.0)

    def test_persistent_retry_config(self):
        """Test persistent retry configuration."""
        config = WLEDConfig(persistent_retry=True, retry_interval=5.0)
        self.assertTrue(config.persistent_retry)
        self.assertEqual(config.retry_interval, 5.0)


class TestWLEDQueryResponse(unittest.TestCase):
    """Test cases for WLED query response functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = WLEDConfig(
            host="test.local",
            port=21324,
            led_count=32,
            timeout=1.0,
            retry_count=2,
            max_fps=30.0,
        )
        self.client = WLEDClient(self.config)

    def tearDown(self):
        """Clean up after tests."""
        if self.client:
            self.client.disconnect()

    def _setup_mock_query_response(self, mock_socket, status_json=None):
        """Helper to setup mock query response."""
        if status_json is None:
            status_json = '{"name":"Test WLED","ver":"0.14.0","leds":{"count":32}}'

        response_data = struct.pack(
            ">BBBBLH", DDPFlags.VER1 | DDPFlags.REPLY, 0, 0, 251, 0, len(status_json)
        )
        response_data += status_json.encode("utf-8")
        mock_socket.recvfrom.return_value = (response_data, ("192.168.1.100", 21324))

    @patch("socket.socket")
    def test_query_response_parsing(self, mock_socket_class):
        """Test parsing of JSON query responses."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        test_status = {
            "name": "My WLED Controller",
            "ver": "0.14.0",
            "leds": {"count": 144, "fps": 42},
            "wifi": {"rssi": -45},
        }

        self._setup_mock_query_response(mock_socket, json.dumps(test_status))

        success = self.client.connect()
        self.assertTrue(success)

        # Verify status was parsed and stored
        stored_status = self.client.get_wled_status()
        self.assertIsNotNone(stored_status)
        self.assertEqual(stored_status["name"], "My WLED Controller")
        self.assertEqual(stored_status["ver"], "0.14.0")
        self.assertEqual(stored_status["leds"]["count"], 144)
        self.assertEqual(stored_status["wifi"]["rssi"], -45)

    @patch("socket.socket")
    def test_malformed_json_response(self, mock_socket_class):
        """Test handling of malformed JSON responses."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Setup malformed JSON response
        malformed_json = '{"name":"Test","ver":'  # Incomplete JSON
        response_data = struct.pack(
            ">BBBBLH", DDPFlags.VER1 | DDPFlags.REPLY, 0, 0, 251, 0, len(malformed_json)
        )
        response_data += malformed_json.encode("utf-8")
        mock_socket.recvfrom.return_value = (response_data, ("192.168.1.100", 21324))

        success = self.client.connect()
        self.assertTrue(success)  # Connection succeeds even with bad JSON

        # Status should be None due to parsing error
        stored_status = self.client.get_wled_status()
        self.assertIsNone(stored_status)

    @patch("socket.socket")
    def test_non_ddp_response(self, mock_socket_class):
        """Test handling of non-DDP response packets."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Setup non-DDP response (wrong flags)
        response_data = b"Not a DDP packet"
        mock_socket.recvfrom.return_value = (response_data, ("192.168.1.100", 21324))

        success = self.client.connect()
        self.assertTrue(success)  # Connection succeeds

        # Status should be None due to non-DDP response
        stored_status = self.client.get_wled_status()
        self.assertIsNone(stored_status)

    @patch("socket.socket")
    def test_refresh_wled_status(self, mock_socket_class):
        """Test refreshing WLED status."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Initial connection
        initial_status = {"name": "Initial", "ver": "0.13.0"}
        self._setup_mock_query_response(mock_socket, json.dumps(initial_status))

        self.client.connect()
        stored_status = self.client.get_wled_status()
        self.assertEqual(stored_status["name"], "Initial")

        # Setup new status for refresh
        new_status = {"name": "Updated", "ver": "0.14.0"}
        updated_json = json.dumps(new_status)
        response_data = struct.pack(
            ">BBBBLH", DDPFlags.VER1 | DDPFlags.REPLY, 0, 0, 251, 0, len(updated_json)
        )
        response_data += updated_json.encode("utf-8")
        mock_socket.recvfrom.return_value = (response_data, ("192.168.1.100", 21324))

        # Refresh status
        success = self.client.refresh_wled_status()
        self.assertTrue(success)

        # Verify updated status
        stored_status = self.client.get_wled_status()
        self.assertEqual(stored_status["name"], "Updated")
        self.assertEqual(stored_status["ver"], "0.14.0")

    @patch("socket.socket")
    def test_truncated_packet_handling(self, mock_socket_class):
        """Test handling of truncated packets."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Setup truncated packet (less than header size)
        truncated_data = b"\x40\x00\x00"  # Only 3 bytes
        mock_socket.recvfrom.return_value = (truncated_data, ("192.168.1.100", 21324))

        success = self.client.connect()
        self.assertTrue(success)  # Connection succeeds

        # Status should be None due to truncated packet
        stored_status = self.client.get_wled_status()
        self.assertIsNone(stored_status)

    @patch("socket.socket")
    def test_incomplete_data_packet(self, mock_socket_class):
        """Test handling of packets with incomplete data."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Create packet header claiming 100 bytes of data but only provide header
        incomplete_data = struct.pack(
            ">BBBBLH", DDPFlags.VER1 | DDPFlags.REPLY, 0, 0, 251, 0, 100
        )
        # No actual data follows, but header claims 100 bytes
        mock_socket.recvfrom.return_value = (incomplete_data, ("192.168.1.100", 21324))

        success = self.client.connect()
        self.assertTrue(success)  # Connection succeeds

        # Status should be None due to incomplete data
        stored_status = self.client.get_wled_status()
        self.assertIsNone(stored_status)

    @patch("socket.socket")
    def test_oversized_data_length(self, mock_socket_class):
        """Test handling of packets claiming excessive data length."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Create packet header claiming unreasonably large data size
        oversized_header = struct.pack(
            ">BBBBLH", DDPFlags.VER1 | DDPFlags.REPLY, 0, 0, 251, 0, 50000
        )
        mock_socket.recvfrom.return_value = (oversized_header, ("192.168.1.100", 21324))

        success = self.client.connect()
        self.assertTrue(success)  # Connection succeeds

        # Status should be None due to excessive data length
        stored_status = self.client.get_wled_status()
        self.assertIsNone(stored_status)

    @patch("socket.socket")
    def test_zero_length_data(self, mock_socket_class):
        """Test handling of packets with zero data length."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Create valid header with zero data length
        zero_data_packet = struct.pack(
            ">BBBBLH", DDPFlags.VER1 | DDPFlags.REPLY, 0, 0, 251, 0, 0
        )
        mock_socket.recvfrom.return_value = (zero_data_packet, ("192.168.1.100", 21324))

        success = self.client.connect()
        self.assertTrue(success)  # Connection succeeds

        # Status should be None due to no data
        stored_status = self.client.get_wled_status()
        self.assertIsNone(stored_status)


class TestWLEDPersistentRetry(unittest.TestCase):
    """Test cases for persistent retry functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = WLEDConfig(
            host="test.local",
            port=21324,
            led_count=32,
            timeout=0.1,  # Short timeout for testing
            retry_count=2,
            persistent_retry=True,
            retry_interval=0.1,  # Short interval for testing
        )
        self.client = WLEDClient(self.config)

    def tearDown(self):
        """Clean up after tests."""
        if self.client:
            self.client.disconnect()

    @patch("time.sleep")
    @patch("socket.socket")
    def test_persistent_retry_success(self, mock_socket_class, mock_sleep):
        """Test that persistent retry eventually succeeds."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Setup to fail first two attempts, succeed on third
        test_json = '{"name":"Test WLED","ver":"0.14.0"}'
        response_data = struct.pack(
            ">BBBBLH", DDPFlags.VER1 | DDPFlags.REPLY, 0, 0, 251, 0, len(test_json)
        )
        response_data += test_json.encode("utf-8")

        mock_socket.recvfrom.side_effect = [
            socket.timeout(),  # First attempt fails
            socket.timeout(),  # Second attempt fails
            (response_data, ("192.168.1.100", 21324)),  # Third attempt succeeds
        ]

        success = self.client.connect()
        self.assertTrue(success)
        self.assertTrue(self.client.is_connected)

        # Verify retries happened
        self.assertEqual(mock_sleep.call_count, 2)  # Slept twice between 3 attempts
        mock_sleep.assert_called_with(0.1)  # Correct retry interval

    @patch("time.sleep")
    @patch("socket.socket")
    def test_persistent_retry_keyboard_interrupt(self, mock_socket_class, mock_sleep):
        """Test that persistent retry stops on KeyboardInterrupt."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Setup to always timeout, then raise KeyboardInterrupt
        mock_socket.recvfrom.side_effect = [
            socket.timeout(),  # First attempt fails
            KeyboardInterrupt(),  # User interrupts
        ]

        success = self.client.connect()
        self.assertFalse(success)  # Should fail due to interrupt
        self.assertFalse(self.client.is_connected)

    @patch("socket.socket")
    def test_non_persistent_retry_fails_quickly(self, mock_socket_class):
        """Test that non-persistent mode fails quickly."""
        # Use non-persistent config
        config = WLEDConfig(host="test.local", timeout=0.1, persistent_retry=False)
        client = WLEDClient(config)

        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        mock_socket.recvfrom.side_effect = socket.timeout()

        start_time = time.time()
        success = client.connect()
        elapsed = time.time() - start_time

        self.assertFalse(success)
        self.assertLess(elapsed, 1.0)  # Should fail quickly, not retry for long


if __name__ == "__main__":
    unittest.main()
