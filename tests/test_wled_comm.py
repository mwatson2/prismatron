"""
Unit tests for the WLED UDP Communication Module.

Tests DDP protocol implementation, UDP communication, error handling,
and performance monitoring.
"""

import os
import socket
import struct
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.const import DDP_HEADER_SIZE, DDP_MAX_DATA_PER_PACKET, LED_COUNT
from src.consumer.wled_comm import (
    DDPProtocol,
    TransmissionResult,
    WLEDCommunicator,
    WLEDPacket,
    WLEDProtocol,
)


class TestWLEDPacket(unittest.TestCase):
    """Test cases for WLEDPacket class."""

    def test_initialization(self):
        """Test packet initialization."""
        data = b"test_data"
        packet = WLEDPacket(protocol=WLEDProtocol.DDP, data=data, sequence=42)

        self.assertEqual(packet.protocol, WLEDProtocol.DDP)
        self.assertEqual(packet.data, data)
        self.assertEqual(packet.sequence, 42)
        self.assertEqual(packet.packet_size, len(data))

    def test_packet_size_calculation(self):
        """Test automatic packet size calculation."""
        data = b"x" * 100
        packet = WLEDPacket(WLEDProtocol.DDP, data)

        self.assertEqual(packet.packet_size, 100)


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


class TestDDPProtocol(unittest.TestCase):
    """Test cases for DDPProtocol class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ddp = DDPProtocol()

    def test_initialization(self):
        """Test DDP protocol initialization."""
        self.assertEqual(self.ddp.sequence_number, 0)

    def test_packet_creation(self):
        """Test DDP packet creation."""
        test_data = b"RGB" * 10  # 30 bytes of test data
        packet = self.ddp.create_packet(test_data, offset=0, is_last=True)

        self.assertGreater(len(packet), DDP_HEADER_SIZE)
        self.assertEqual(len(packet), DDP_HEADER_SIZE + len(test_data))

        # Verify header structure
        header = packet[:DDP_HEADER_SIZE]
        flags, sequence, data_type, source, offset, length = struct.unpack(
            ">BBBBIH", header
        )

        self.assertEqual(flags & 0x40, 0x40)  # Version 1
        self.assertEqual(flags & 0x01, 0x01)  # Push flag for last packet
        self.assertEqual(sequence, 0)  # First packet
        self.assertEqual(data_type, 1)  # RGB data
        self.assertEqual(offset, 0)
        self.assertEqual(length, len(test_data))

    def test_sequence_increment(self):
        """Test sequence number incrementing."""
        initial_seq = self.ddp.sequence_number

        self.ddp.create_packet(b"test1")
        self.assertEqual(self.ddp.sequence_number, (initial_seq + 1) % 256)

        self.ddp.create_packet(b"test2")
        self.assertEqual(self.ddp.sequence_number, (initial_seq + 2) % 256)

    def test_sequence_wraparound(self):
        """Test sequence number wraparound at 256."""
        self.ddp.sequence_number = 255

        self.ddp.create_packet(b"test")
        self.assertEqual(self.ddp.sequence_number, 0)

    def test_data_fragmentation(self):
        """Test LED data fragmentation."""
        # Create large test data that needs fragmentation
        large_data = b"RGB" * 1000  # 3000 bytes
        packets = self.ddp.fragment_data(large_data)

        self.assertGreater(len(packets), 1)  # Should create multiple packets

        # Verify total data size
        total_data = b""
        for packet in packets:
            data_part = packet[DDP_HEADER_SIZE:]  # Skip header
            total_data += data_part

        self.assertEqual(len(total_data), len(large_data))

    def test_fragmentation_packet_flags(self):
        """Test that fragmentation sets correct flags."""
        large_data = b"x" * (DDP_MAX_DATA_PER_PACKET * 2)  # Force multiple packets
        packets = self.ddp.fragment_data(large_data)

        self.assertGreater(len(packets), 1)

        # Check that only the last packet has push flag
        for i, packet in enumerate(packets):
            flags = packet[0]  # First byte is flags
            is_last = i == len(packets) - 1
            has_push = bool(flags & 0x01)

            self.assertEqual(has_push, is_last)

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        packet = self.ddp.create_packet(b"")

        self.assertEqual(len(packet), DDP_HEADER_SIZE)

        # Verify header for empty data
        header = packet[:DDP_HEADER_SIZE]
        _, _, _, _, _, length = struct.unpack(">BBBBIH", header)
        self.assertEqual(length, 0)

    def test_fragmentation_with_small_data(self):
        """Test fragmentation with data smaller than max packet size."""
        small_data = b"RGB" * 10  # 30 bytes
        packets = self.ddp.fragment_data(small_data)

        self.assertEqual(len(packets), 1)  # Should create only one packet

        # Verify packet content
        packet = packets[0]
        data_part = packet[DDP_HEADER_SIZE:]
        self.assertEqual(data_part, small_data)


class TestWLEDCommunicator(unittest.TestCase):
    """Test cases for WLEDCommunicator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.communicator = WLEDCommunicator(
            host="192.168.1.100", port=4048, timeout=1.0
        )

    def tearDown(self):
        """Clean up after tests."""
        if self.communicator.is_connected():
            self.communicator.disconnect()

    def test_initialization(self):
        """Test communicator initialization."""
        self.assertEqual(self.communicator.host, "192.168.1.100")
        self.assertEqual(self.communicator.port, 4048)
        self.assertEqual(self.communicator.protocol, WLEDProtocol.DDP)
        self.assertFalse(self.communicator.is_connected())

    @patch("socket.socket")
    def test_connection_success(self, mock_socket_class):
        """Test successful connection."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        result = self.communicator.connect()

        self.assertTrue(result)
        self.assertTrue(self.communicator.is_connected())
        mock_socket_class.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)
        mock_socket.settimeout.assert_called_once_with(1.0)

    @patch("socket.socket")
    def test_connection_failure(self, mock_socket_class):
        """Test connection failure."""
        mock_socket_class.side_effect = Exception("Connection failed")

        result = self.communicator.connect()

        self.assertFalse(result)
        self.assertFalse(self.communicator.is_connected())

    @patch("socket.socket")
    def test_disconnect(self, mock_socket_class):
        """Test disconnection."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        self.communicator.connect()
        self.communicator.disconnect()

        self.assertFalse(self.communicator.is_connected())
        mock_socket.close.assert_called_once()

    @patch("socket.socket")
    def test_led_data_validation(self, mock_socket_class):
        """Test LED data validation."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        self.communicator.connect()

        # Test with wrong shape
        wrong_data = np.zeros((100, 3))  # Wrong LED count
        result = self.communicator.send_led_data(wrong_data)

        self.assertFalse(result.success)
        self.assertIn("LED data shape", result.errors[0])

    @patch("socket.socket")
    def test_successful_data_transmission(self, mock_socket_class):
        """Test successful LED data transmission."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        self.communicator.connect()

        # Create valid LED data
        led_data = np.random.randint(0, 255, (LED_COUNT, 3)).astype(np.float32)

        result = self.communicator.send_led_data(led_data, force_send=True)

        self.assertTrue(result.success)
        self.assertGreater(result.packets_sent, 0)
        self.assertGreater(result.bytes_sent, 0)
        self.assertGreater(result.transmission_time, 0)

    def test_led_data_without_connection(self):
        """Test sending LED data without connection."""
        led_data = np.random.randint(0, 255, (LED_COUNT, 3)).astype(np.float32)

        result = self.communicator.send_led_data(led_data)

        self.assertFalse(result.success)
        self.assertIn("Not connected", result.errors[0])

    def test_frame_rate_limiting(self):
        """Test frame rate limiting."""
        self.communicator.set_frame_rate_limit(30.0)  # 30 FPS

        # Check that minimum frame interval is set correctly
        expected_interval = 1.0 / 30.0
        self.assertAlmostEqual(
            self.communicator._min_frame_interval, expected_interval, places=6
        )

    @patch("socket.socket")
    @patch("time.time")
    def test_frame_rate_limiting_behavior(self, mock_time, mock_socket_class):
        """Test that frame rate limiting actually works."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Mock time to control frame timing
        mock_time.side_effect = [
            0.0,
            0.0,
            0.01,
            0.01,
            0.02,
            0.02,
            0.03,
        ]  # More time values for all calls

        self.communicator.connect()
        self.communicator.set_frame_rate_limit(30.0)  # 30 FPS = ~0.033s interval

        led_data = np.random.randint(0, 255, (LED_COUNT, 3)).astype(np.float32)

        # First frame should succeed
        result1 = self.communicator.send_led_data(led_data, force_send=True)
        self.assertTrue(result1.success)

        # Second frame should be rate limited (0.01s < 0.033s interval)
        result2 = self.communicator.send_led_data(led_data, force_send=False)
        self.assertFalse(result2.success)
        self.assertIn("Frame rate limited", result2.errors[0])

    def test_led_data_preparation(self):
        """Test LED data preparation and clamping."""
        # Test data with values outside [0, 255] range
        led_data = np.array(
            [
                [-10, 128, 300],  # Out of range values
                [0, 255, 128],  # Valid values
            ]
        ).astype(np.float32)

        prepared_data = self.communicator._prepare_led_data(led_data)

        # Should be flattened and clamped
        expected_length = led_data.size
        self.assertEqual(len(prepared_data), expected_length)

        # Convert back to check clamping
        prepared_array = np.frombuffer(prepared_data, dtype=np.uint8)
        self.assertTrue(np.all(prepared_array >= 0))
        self.assertTrue(np.all(prepared_array <= 255))

    @patch("socket.socket")
    def test_test_pattern_generation(self, mock_socket_class):
        """Test test pattern generation and transmission."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        self.communicator.connect()

        # Test different patterns
        patterns = ["rainbow", "solid", "off"]

        for pattern in patterns:
            result = self.communicator.send_test_pattern(pattern)
            self.assertTrue(result.success, f"Pattern '{pattern}' failed")

    def test_statistics_collection(self):
        """Test communication statistics collection."""
        stats = self.communicator.get_communication_stats()

        # Check that all expected keys are present
        expected_keys = [
            "host",
            "port",
            "protocol",
            "connected",
            "frames_sent",
            "total_bytes_sent",
            "total_transmission_time",
            "average_transmission_time",
            "estimated_fps",
            "frame_rate_limit",
            "error_count",
            "recent_errors",
            "throughput_mbps",
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

    def test_statistics_reset(self):
        """Test statistics reset."""
        # Manually set some statistics
        self.communicator._frames_sent = 10
        self.communicator._total_bytes_sent = 1000
        self.communicator._errors.append("Test error")

        self.communicator.reset_statistics()

        self.assertEqual(self.communicator._frames_sent, 0)
        self.assertEqual(self.communicator._total_bytes_sent, 0)
        self.assertEqual(len(self.communicator._errors), 0)

    @patch("socket.socket")
    def test_context_manager(self, mock_socket_class):
        """Test context manager functionality."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        with self.communicator as comm:
            self.assertTrue(comm.is_connected())

        self.assertFalse(self.communicator.is_connected())

    def test_context_manager_connection_failure(self):
        """Test context manager with connection failure."""
        with patch.object(self.communicator, "connect", return_value=False):
            with self.assertRaises(RuntimeError):
                with self.communicator:
                    pass

    @patch("socket.socket")
    def test_ddp_packet_retry_logic(self, mock_socket_class):
        """Test DDP packet retry logic on send failure."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # First call succeeds (connection test), then some packets fail and retry
        # We need 7 packets for full LED array, let's make first packet fail 2 times then succeed
        mock_socket.sendto.side_effect = [
            None,  # Connection test succeeds
            Exception("Network error"),  # First packet attempt 1 fails
            Exception("Network error"),  # First packet attempt 2 fails
            None,  # First packet attempt 3 succeeds
            None,
            None,
            None,
            None,
            None,
            None,  # Remaining 6 packets succeed
        ]

        self.communicator.connect()
        led_data = np.random.randint(0, 255, (LED_COUNT, 3)).astype(np.float32)

        result = self.communicator.send_led_data(led_data, force_send=True)

        # Should eventually succeed after retries
        self.assertTrue(result.success)
        self.assertEqual(
            mock_socket.sendto.call_count, 10
        )  # 1 for connection + 9 for LED data (2 retries + 7 success)

    @patch("socket.socket")
    def test_ddp_packet_retry_exhaustion(self, mock_socket_class):
        """Test DDP packet retry exhaustion."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        # Connection test succeeds, but all LED data sends fail
        side_effects = [None] + [Exception("Network error")] * 10  # Enough failures
        mock_socket.sendto.side_effect = side_effects

        self.communicator.connect()
        led_data = np.random.randint(0, 255, (LED_COUNT, 3)).astype(np.float32)

        result = self.communicator.send_led_data(led_data, force_send=True)

        # Should fail after exhausting retries
        self.assertFalse(result.success)
        self.assertGreater(len(result.errors), 0)

    def test_protocol_enum(self):
        """Test WLED protocol enum."""
        self.assertEqual(WLEDProtocol.DDP.value, "ddp")
        self.assertEqual(WLEDProtocol.UDP_RAW.value, "udp_raw")
        self.assertEqual(WLEDProtocol.E131.value, "e131")


class TestWLEDIntegration(unittest.TestCase):
    """Integration tests for WLED communication."""

    @patch("socket.socket")
    def test_end_to_end_transmission(self, mock_socket_class):
        """Test complete end-to-end LED data transmission."""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket

        communicator = WLEDCommunicator()

        # Connect and send test data
        self.assertTrue(communicator.connect())

        # Create realistic LED data
        led_data = np.zeros((LED_COUNT, 3), dtype=np.float32)
        led_data[:100] = [255, 0, 0]  # First 100 LEDs red
        led_data[100:200] = [0, 255, 0]  # Next 100 LEDs green
        led_data[200:300] = [0, 0, 255]  # Next 100 LEDs blue

        # Send data
        result = communicator.send_led_data(led_data, force_send=True)

        # Verify transmission
        self.assertTrue(result.success)
        self.assertGreater(result.packets_sent, 0)
        # Note: The last packet might be smaller than max size, so just verify bytes were sent
        self.assertGreater(result.bytes_sent, 0)

        # Check that socket.sendto was called
        self.assertGreater(mock_socket.sendto.call_count, 0)

        communicator.disconnect()

    def test_large_led_array_fragmentation(self):
        """Test handling of large LED arrays requiring fragmentation."""
        ddp = DDPProtocol()

        # Create LED data for full array
        led_data_bytes = b"RGB" * LED_COUNT  # 3 * 3200 = 9600 bytes

        packets = ddp.fragment_data(led_data_bytes)

        # Verify fragmentation
        self.assertGreater(len(packets), 1)

        # Verify total data reconstruction
        reconstructed_data = b""
        for packet in packets:
            data_part = packet[DDP_HEADER_SIZE:]
            reconstructed_data += data_part

        self.assertEqual(len(reconstructed_data), len(led_data_bytes))


if __name__ == "__main__":
    unittest.main()
