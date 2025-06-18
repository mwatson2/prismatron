"""
WLED UDP Communication Module.

This module implements UDP communication protocols for WLED controllers,
including DDP (Distributed Display Protocol) for real-time LED data
transmission with error handling and performance optimization.
"""

import logging
import socket
import struct
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..const import (
    DDP_HEADER_SIZE,
    DDP_MAX_DATA_PER_PACKET,
    DDP_MAX_PACKET_SIZE,
    LED_COUNT,
    LED_DATA_SIZE,
    WLED_DDP_PORT,
    WLED_DEFAULT_HOST,
    WLED_DEFAULT_PORT,
    WLED_MAX_FPS,
    WLED_MIN_FRAME_INTERVAL,
    WLED_RETRY_COUNT,
    WLED_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class WLEDProtocol(Enum):
    """Supported WLED communication protocols."""

    DDP = "ddp"
    UDP_RAW = "udp_raw"
    E131 = "e131"


@dataclass
class WLEDPacket:
    """Represents a WLED data packet."""

    protocol: WLEDProtocol
    data: bytes
    sequence: int = 0
    packet_size: int = 0

    def __post_init__(self):
        """Calculate packet size."""
        self.packet_size = len(self.data)


@dataclass
class TransmissionResult:
    """Result of WLED transmission."""

    success: bool
    packets_sent: int
    bytes_sent: int
    transmission_time: float
    errors: List[str]
    frame_rate: float = 0.0

    def get_throughput_mbps(self) -> float:
        """Calculate throughput in Mbps."""
        if self.transmission_time > 0:
            bits_per_second = (self.bytes_sent * 8) / self.transmission_time
            return bits_per_second / 1_000_000
        return 0.0


class DDPProtocol:
    """
    DDP (Distributed Display Protocol) implementation for WLED.

    DDP is the preferred protocol for real-time LED data transmission
    with automatic packet fragmentation and sequencing.
    """

    # DDP packet format constants
    DDP_FLAGS_VER = 0x40  # Version 1, no response required (bit 6)
    DDP_FLAGS_PUSH = 0x01  # Push flag for last packet (bit 0)
    DDP_FLAGS_QUERY = 0x02  # Query flag
    DDP_FLAGS_REPLY = 0x04  # Reply flag
    DDP_FLAGS_STORAGE = 0x08  # Storage flag
    DDP_FLAGS_TIME = 0x10  # Time flag

    def __init__(self):
        """Initialize DDP protocol handler."""
        self.sequence_number = 0

    def create_packet(
        self, led_data: bytes, offset: int = 0, is_last: bool = False
    ) -> bytes:
        """
        Create DDP packet.

        Args:
            led_data: RGB LED data
            offset: Offset in the LED strip
            is_last: Whether this is the last packet in sequence

        Returns:
            DDP packet bytes
        """
        try:
            # Calculate flags
            flags = self.DDP_FLAGS_VER
            if is_last:
                flags |= self.DDP_FLAGS_PUSH

            # Create DDP header (10 bytes)
            # Format: flags, sequence, type, source, offset (4 bytes), length (2 bytes)
            header = struct.pack(
                ">BBBBIH",
                flags,  # Flags (1 byte)
                self.sequence_number,  # Sequence number (1 byte)
                1,  # Data type: RGB (1 byte)
                1,  # Source ID (1 byte)
                offset,  # Offset (4 bytes)
                len(led_data),  # Data length (2 bytes)
            )

            # Increment sequence number
            self.sequence_number = (self.sequence_number + 1) % 256

            return header + led_data

        except Exception as e:
            logger.error(f"Failed to create DDP packet: {e}")
            return b""

    def fragment_data(self, led_data: bytes) -> List[bytes]:
        """
        Fragment LED data into DDP packets.

        Args:
            led_data: Complete LED data array

        Returns:
            List of DDP packet bytes
        """
        try:
            packets = []
            data_offset = 0

            while data_offset < len(led_data):
                # Calculate chunk size for this packet
                remaining = len(led_data) - data_offset
                chunk_size = min(remaining, DDP_MAX_DATA_PER_PACKET)

                # Extract data chunk
                chunk = led_data[data_offset : data_offset + chunk_size]

                # Determine if this is the last packet
                is_last = (data_offset + chunk_size) >= len(led_data)

                # Create packet
                packet = self.create_packet(chunk, data_offset, is_last)
                if packet:
                    packets.append(packet)

                data_offset += chunk_size

            logger.debug(
                f"Fragmented {len(led_data)} bytes into {len(packets)} DDP packets"
            )
            return packets

        except Exception as e:
            logger.error(f"Failed to fragment DDP data: {e}")
            return []


class WLEDCommunicator:
    """
    WLED UDP communication handler.

    Manages UDP connections, packet transmission, error handling,
    and performance monitoring for WLED controllers.
    """

    def __init__(
        self,
        host: str = WLED_DEFAULT_HOST,
        port: int = WLED_DDP_PORT,
        protocol: WLEDProtocol = WLEDProtocol.DDP,
        timeout: float = WLED_TIMEOUT_SECONDS,
    ):
        """
        Initialize WLED communicator.

        Args:
            host: WLED controller hostname/IP
            port: UDP port number
            protocol: Communication protocol to use
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.protocol = protocol
        self.timeout = timeout

        # Socket management
        self._socket: Optional[socket.socket] = None
        self._connected = False

        # Protocol handlers
        self._ddp = DDPProtocol()

        # Statistics
        self._frames_sent = 0
        self._total_bytes_sent = 0
        self._total_transmission_time = 0.0
        self._last_frame_time = 0.0
        self._errors: List[str] = []

        # Rate limiting
        self._min_frame_interval = WLED_MIN_FRAME_INTERVAL
        self._last_send_time = 0.0

    def connect(self) -> bool:
        """
        Establish UDP connection to WLED controller.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create UDP socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.settimeout(self.timeout)

            # Test connection with a small packet
            test_result = self._test_connection()

            if test_result:
                self._connected = True
                logger.info(
                    f"Connected to WLED at {self.host}:{self.port} using {self.protocol.value}"
                )
                return True
            else:
                self.disconnect()
                return False

        except Exception as e:
            logger.error(f"Failed to connect to WLED: {e}")
            self._errors.append(f"Connection failed: {e}")
            self.disconnect()
            return False

    def _test_connection(self) -> bool:
        """
        Test connection with a small test packet.

        Returns:
            True if test successful, False otherwise
        """
        try:
            # Send a small test packet (single black LED)
            test_data = bytes([0, 0, 0])  # Black RGB
            test_packet = self._ddp.create_packet(test_data, is_last=True)

            self._socket.sendto(test_packet, (self.host, self.port))
            logger.debug(f"Test packet sent to {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            return False

    def disconnect(self) -> None:
        """Close UDP connection."""
        try:
            if self._socket:
                self._socket.close()
                self._socket = None

            self._connected = False
            logger.debug(f"Disconnected from WLED at {self.host}:{self.port}")

        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")

    def send_led_data(
        self, led_values: np.ndarray, force_send: bool = False
    ) -> TransmissionResult:
        """
        Send LED data to WLED controller.

        Args:
            led_values: LED RGB values (led_count, 3) in range [0, 255]
            force_send: Override frame rate limiting

        Returns:
            TransmissionResult with transmission metrics
        """
        start_time = time.time()

        try:
            # Check connection
            if not self._connected or not self._socket:
                raise RuntimeError("Not connected to WLED controller")

            # Validate LED data
            if led_values.shape != (LED_COUNT, 3):
                raise ValueError(
                    f"LED data shape {led_values.shape} != expected {(LED_COUNT, 3)}"
                )

            # Check frame rate limiting
            current_time = time.time()
            if not force_send:
                time_since_last = current_time - self._last_send_time
                if time_since_last < self._min_frame_interval:
                    # Skip frame to maintain rate limit
                    return TransmissionResult(
                        success=False,
                        packets_sent=0,
                        bytes_sent=0,
                        transmission_time=time.time() - start_time,
                        errors=["Frame rate limited"],
                        frame_rate=0.0,
                    )

            # Convert LED values to bytes
            led_data = self._prepare_led_data(led_values)

            # Send data using selected protocol
            if self.protocol == WLEDProtocol.DDP:
                result = self._send_ddp_data(led_data)
            else:
                raise NotImplementedError(f"Protocol {self.protocol} not implemented")

            # Update statistics
            if result.success:
                self._frames_sent += 1
                self._total_bytes_sent += result.bytes_sent
                self._total_transmission_time += result.transmission_time
                self._last_send_time = current_time

                # Calculate frame rate
                if self._last_frame_time > 0:
                    frame_interval = current_time - self._last_frame_time
                    result.frame_rate = (
                        1.0 / frame_interval if frame_interval > 0 else 0.0
                    )

                self._last_frame_time = current_time

            return result

        except Exception as e:
            error_msg = f"LED data transmission failed: {e}"
            logger.error(error_msg)
            self._errors.append(error_msg)

            return TransmissionResult(
                success=False,
                packets_sent=0,
                bytes_sent=0,
                transmission_time=time.time() - start_time,
                errors=[error_msg],
            )

    def _prepare_led_data(self, led_values: np.ndarray) -> bytes:
        """
        Prepare LED data for transmission.

        Args:
            led_values: LED RGB values (led_count, 3)

        Returns:
            Packed LED data bytes
        """
        try:
            # Ensure values are in valid range [0, 255]
            led_data = np.clip(led_values, 0, 255).astype(np.uint8)

            # Convert to bytes (flatten and pack RGB values)
            return led_data.flatten().tobytes()

        except Exception as e:
            logger.error(f"Failed to prepare LED data: {e}")
            return b""

    def _send_ddp_data(self, led_data: bytes) -> TransmissionResult:
        """
        Send LED data using DDP protocol.

        Args:
            led_data: Packed LED data bytes

        Returns:
            TransmissionResult with transmission metrics
        """
        start_time = time.time()
        packets_sent = 0
        bytes_sent = 0
        errors = []

        try:
            # Fragment data into DDP packets
            packets = self._ddp.fragment_data(led_data)

            if not packets:
                raise RuntimeError("Failed to create DDP packets")

            # Send each packet with retry logic
            for packet in packets:
                success = False

                for attempt in range(WLED_RETRY_COUNT):
                    try:
                        self._socket.sendto(packet, (self.host, self.port))
                        packets_sent += 1
                        bytes_sent += len(packet)
                        success = True
                        break

                    except Exception as e:
                        error_msg = f"Packet send attempt {attempt + 1} failed: {e}"
                        logger.warning(error_msg)
                        if attempt == WLED_RETRY_COUNT - 1:  # Last attempt
                            errors.append(error_msg)

                if not success:
                    errors.append(
                        f"Failed to send packet after {WLED_RETRY_COUNT} attempts"
                    )

            transmission_time = time.time() - start_time
            success = len(errors) == 0

            if success:
                logger.debug(
                    f"Sent {packets_sent} DDP packets ({bytes_sent} bytes) "
                    f"in {transmission_time:.3f}s"
                )
            else:
                logger.warning(f"DDP transmission completed with {len(errors)} errors")

            return TransmissionResult(
                success=success,
                packets_sent=packets_sent,
                bytes_sent=bytes_sent,
                transmission_time=transmission_time,
                errors=errors,
            )

        except Exception as e:
            error_msg = f"DDP transmission failed: {e}"
            logger.error(error_msg)

            return TransmissionResult(
                success=False,
                packets_sent=packets_sent,
                bytes_sent=bytes_sent,
                transmission_time=time.time() - start_time,
                errors=[error_msg],
            )

    def set_frame_rate_limit(self, max_fps: float) -> None:
        """
        Set maximum frame rate for transmission.

        Args:
            max_fps: Maximum frames per second
        """
        max_fps = min(max_fps, WLED_MAX_FPS)  # Enforce system limit
        self._min_frame_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        logger.info(f"Frame rate limit set to {max_fps} FPS")

    def send_test_pattern(self, pattern: str = "rainbow") -> TransmissionResult:
        """
        Send a test pattern to verify connection.

        Args:
            pattern: Test pattern type ("rainbow", "solid", "off")

        Returns:
            TransmissionResult with transmission metrics
        """
        try:
            # Generate test pattern
            led_data = self._generate_test_pattern(pattern)

            # Send test data
            result = self.send_led_data(led_data, force_send=True)

            if result.success:
                logger.info(f"Test pattern '{pattern}' sent successfully")
            else:
                logger.warning(f"Test pattern '{pattern}' failed: {result.errors}")

            return result

        except Exception as e:
            error_msg = f"Test pattern failed: {e}"
            logger.error(error_msg)
            return TransmissionResult(
                success=False,
                packets_sent=0,
                bytes_sent=0,
                transmission_time=0.0,
                errors=[error_msg],
            )

    def _generate_test_pattern(self, pattern: str) -> np.ndarray:
        """
        Generate test LED pattern.

        Args:
            pattern: Pattern type

        Returns:
            LED values array (led_count, 3)
        """
        led_data = np.zeros((LED_COUNT, 3), dtype=np.uint8)

        if pattern == "rainbow":
            # Rainbow pattern
            for i in range(LED_COUNT):
                hue = (i / LED_COUNT) * 360
                # Simple HSV to RGB conversion for rainbow
                c = 255  # Assume max saturation and value
                x = int(c * (1 - abs((hue / 60) % 2 - 1)))

                if 0 <= hue < 60:
                    led_data[i] = [c, x, 0]
                elif 60 <= hue < 120:
                    led_data[i] = [x, c, 0]
                elif 120 <= hue < 180:
                    led_data[i] = [0, c, x]
                elif 180 <= hue < 240:
                    led_data[i] = [0, x, c]
                elif 240 <= hue < 300:
                    led_data[i] = [x, 0, c]
                else:
                    led_data[i] = [c, 0, x]

        elif pattern == "solid":
            # Solid white
            led_data.fill(128)

        elif pattern == "off":
            # All LEDs off
            led_data.fill(0)

        return led_data.astype(np.float32)

    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get communication statistics.

        Returns:
            Dictionary with communication statistics
        """
        avg_transmission_time = self._total_transmission_time / max(
            1, self._frames_sent
        )

        estimated_fps = (
            1.0 / avg_transmission_time if avg_transmission_time > 0 else 0.0
        )

        return {
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol.value,
            "connected": self._connected,
            "frames_sent": self._frames_sent,
            "total_bytes_sent": self._total_bytes_sent,
            "total_transmission_time": self._total_transmission_time,
            "average_transmission_time": avg_transmission_time,
            "estimated_fps": estimated_fps,
            "frame_rate_limit": 1.0 / self._min_frame_interval
            if self._min_frame_interval > 0
            else 0.0,
            "error_count": len(self._errors),
            "recent_errors": self._errors[-5:],  # Last 5 errors
            "throughput_mbps": (
                (self._total_bytes_sent * 8)
                / (self._total_transmission_time * 1_000_000)
                if self._total_transmission_time > 0
                else 0.0
            ),
        }

    def reset_statistics(self) -> None:
        """Reset communication statistics."""
        self._frames_sent = 0
        self._total_bytes_sent = 0
        self._total_transmission_time = 0.0
        self._last_frame_time = 0.0
        self._errors.clear()
        logger.debug("Communication statistics reset")

    def is_connected(self) -> bool:
        """Check if connected to WLED controller."""
        return self._connected and self._socket is not None

    def __enter__(self):
        """Context manager entry."""
        if not self.connect():
            raise RuntimeError("Failed to connect to WLED controller")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
