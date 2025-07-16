"""
WLED UDP Communication Client.

This module implements a client for communicating with WLED controllers over UDP
using the DDP (Distributed Display Protocol) for efficient LED data transmission.
Includes flow control, packet fragmentation, and error handling.
"""

import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests

from ..const import (
    DDP_HEADER_SIZE,
    DDP_MAX_DATA_PER_PACKET,
    LED_COUNT,
    LED_DATA_SIZE,
    WLED_DEFAULT_HOST,
    WLED_DEFAULT_PORT,
    WLED_MIN_FRAME_INTERVAL,
    WLED_RETRY_COUNT,
    WLED_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class DDPFlags(IntEnum):
    """DDP Protocol flags."""

    VER1 = 0x40  # Version 1
    PUSH = 0x01  # Push flag - display data immediately
    QUERY = 0x02  # Query flag
    REPLY = 0x04  # Reply flag
    STORAGE = 0x08  # Storage flag
    TIME = 0x10  # Time flag


class DDPDataType(IntEnum):
    """DDP data types."""

    RGB24 = 0x00  # 24-bit RGB data
    HSV24 = 0x01  # 24-bit HSV data
    RGBW32 = 0x02  # 32-bit RGBW data


@dataclass
class TransmissionResult:
    """Result of LED data transmission."""

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


@dataclass
class WLEDConfig:
    """Configuration for WLED controller communication."""

    host: str = WLED_DEFAULT_HOST
    port: int = WLED_DEFAULT_PORT
    led_count: int = LED_COUNT
    timeout: float = WLED_TIMEOUT_SECONDS
    retry_count: int = WLED_RETRY_COUNT
    max_fps: float = 60.0  # Maximum frame rate
    persistent_retry: bool = False  # Retry connection indefinitely
    retry_interval: float = 10.0  # Seconds between connection retries
    keepalive_interval: float = 1.0  # Seconds between keepalive transmissions
    enable_keepalive: bool = True  # Whether to enable keepalive functionality


class WLEDClient:
    """
    Client for communicating with WLED controllers via DDP protocol.

    Provides efficient transmission of LED data with flow control,
    packet fragmentation for large arrays, and error handling.
    """

    def __init__(self, config: Optional[WLEDConfig] = None):
        """
        Initialize WLED client.

        Args:
            config: WLED configuration (uses defaults if None)
        """
        self.config = config or WLEDConfig()
        self.socket: Optional[socket.socket] = None
        self.sequence_number = 0
        self.last_frame_time = 0.0
        self.is_connected = False

        # Thread safety for sequence numbers and timing
        self._lock = threading.Lock()

        # Statistics
        self.frames_sent = 0
        self.packets_sent = 0
        self.transmission_errors = 0

        # WLED status from last query
        self.wled_status: Optional[Dict[str, Any]] = None

        # Keepalive functionality to prevent WLED from reverting to local patterns
        self._last_led_data: Optional[bytes] = None
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop_event = threading.Event()
        self._keepalive_interval = self.config.keepalive_interval
        self._enable_keepalive = self.config.enable_keepalive
        self._last_data_time = 0.0

        # Calculate packet fragmentation info
        self._calculate_fragmentation()

    def _calculate_fragmentation(self) -> None:
        """Calculate how to fragment LED data into DDP packets."""
        data_size = self.config.led_count * 3  # RGB data

        self.packets_per_frame = (data_size + DDP_MAX_DATA_PER_PACKET - 1) // DDP_MAX_DATA_PER_PACKET
        self.data_per_packet = DDP_MAX_DATA_PER_PACKET
        self.last_packet_size = data_size % DDP_MAX_DATA_PER_PACKET
        if self.last_packet_size == 0:
            self.last_packet_size = DDP_MAX_DATA_PER_PACKET

        logger.info(
            f"LED array fragmentation: {self.packets_per_frame} packets, "
            f"{self.data_per_packet} bytes per packet, "
            f"last packet {self.last_packet_size} bytes"
        )

    def connect(self) -> bool:
        """
        Establish connection to WLED controller.

        Returns:
            True if connection successful, False otherwise
        """
        attempt = 0
        start_time = time.time()

        while True:
            attempt += 1

            try:
                if self.socket:
                    self.disconnect()

                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.settimeout(self.config.timeout)

                # Test connectivity with a query packet
                success, status = self._send_query()
                if success:
                    self.is_connected = True

                    # Store and log connection details
                    self.wled_status = status
                    if status:
                        logger.info(
                            f"Connected to WLED '{status.get('name', 'Unknown')}' "
                            f"v{status.get('ver', 'Unknown')} at "
                            f"{self.config.host}:{self.config.port}"
                        )

                        # Validate LED count if available
                        wled_led_count = (
                            status.get("leds", {}).get("count", 0) if isinstance(status.get("leds"), dict) else 0
                        )
                        if wled_led_count > 0 and wled_led_count != self.config.led_count:
                            logger.warning(
                                f"LED count mismatch: configured {self.config.led_count}, WLED reports {wled_led_count}"
                            )
                    else:
                        logger.info(
                            f"Connected to WLED controller at "
                            f"{self.config.host}:{self.config.port} "
                            f"(no status response)"
                        )

                    # Start keepalive thread if enabled
                    if self._enable_keepalive:
                        self._start_keepalive_thread()
                    return True
                else:
                    if not self.config.persistent_retry:
                        logger.error(f"Failed to connect to WLED controller at {self.config.host}:{self.config.port}")
                        self.disconnect()
                        return False

                    # Persistent retry mode
                    elapsed = time.time() - start_time
                    if attempt == 1:
                        logger.info(
                            f"Attempting to connect to WLED controller at "
                            f"{self.config.host}:{self.config.port} "
                            f"(persistent retry enabled, interval: {self.config.retry_interval}s)"
                        )
                    elif elapsed < 60.0:  # Only log retry messages for first minute
                        logger.info(
                            f"Connection attempt {attempt} failed after {elapsed:.1f}s, "
                            f"retrying in {self.config.retry_interval}s..."
                        )

                    self.disconnect()
                    time.sleep(self.config.retry_interval)

            except KeyboardInterrupt:
                logger.info("Connection interrupted by user")
                self.disconnect()
                return False
            except Exception as e:
                if not self.config.persistent_retry:
                    logger.error(f"Connection error: {e}")
                    self.disconnect()
                    return False

                # Persistent retry mode
                elapsed = time.time() - start_time
                if elapsed < 60.0:  # Only log retry messages for first minute
                    logger.warning(
                        f"Connection error after {elapsed:.1f}s: {e}, retrying in {self.config.retry_interval}s..."
                    )
                self.disconnect()
                time.sleep(self.config.retry_interval)

    def disconnect(self) -> None:
        """Disconnect from WLED controller."""
        # Stop keepalive thread first
        self._stop_keepalive_thread()

        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                logger.warning(f"Error closing socket: {e}")
            finally:
                self.socket = None

        self.is_connected = False
        logger.info("Disconnected from WLED controller")

    def _send_query(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Query WLED device via HTTP to test connectivity and get status.

        WLED does not respond to DDP query packets, so we use the HTTP API instead.
        Fetches device information from http://wled.local/json/info

        Returns:
            Tuple of (success, status_dict) where status_dict contains WLED info
        """
        try:
            # Construct HTTP URL - support both hostname and IP address
            if self.config.host in ["wled.local", "WLED.local"]:
                base_url = f"http://{self.config.host}"
            else:
                # For IP addresses, use the IP directly
                base_url = f"http://{self.config.host}"

            info_url = f"{base_url}/json/info"

            # Send HTTP GET request with timeout
            response = requests.get(
                info_url,
                timeout=self.config.timeout,
                headers={"Accept": "application/json"},
            )

            # Check if request was successful
            if response.status_code == 200:
                try:
                    status = response.json()

                    # Validate this is a WLED device by checking the brand field
                    if status.get("brand") == "WLED":
                        logger.info(
                            f"WLED device detected: {status.get('name', 'Unknown')} "
                            f"v{status.get('ver', 'Unknown')} at {base_url}"
                        )

                        # Log additional useful info
                        leds_info = status.get("leds", {})
                        if isinstance(leds_info, dict) and "count" in leds_info:
                            logger.info(f"LED count: {leds_info['count']}")

                        # Test UDP connectivity by sending a simple packet
                        # (without expecting a response since WLED doesn't reply to DDP queries)
                        udp_success = self._test_udp_connectivity()
                        if not udp_success:
                            logger.warning("UDP connectivity test failed, but HTTP works")

                        return True, status
                    else:
                        logger.warning(
                            f"Device at {base_url} responded but is not WLED (brand: {status.get('brand', 'Unknown')})"
                        )
                        return False, None

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON response from {info_url}: {e}")
                    return False, None

            else:
                logger.warning(f"HTTP request to {info_url} failed with status {response.status_code}")
                return False, None

        except requests.exceptions.Timeout:
            logger.warning(f"HTTP request to WLED timed out after {self.config.timeout}s")
            return False, None
        except requests.exceptions.ConnectionError:
            logger.warning(f"Failed to connect to WLED at {self.config.host}")
            return False, None
        except Exception as e:
            logger.error(f"WLED HTTP query error: {e}")
            return False, None

    def _test_udp_connectivity(self) -> bool:
        """
        Test UDP connectivity by sending a simple DDP packet.

        WLED doesn't respond to DDP queries, but we can verify that
        we can send UDP packets to the device.

        Returns:
            True if UDP packet was sent successfully, False otherwise
        """
        try:
            if self.socket is None:
                return False

            # Send a minimal DDP packet with no data (just header)
            flags = DDPFlags.VER1
            packet = struct.pack(
                ">BBBBLH",
                flags,  # Flags
                0,  # Sequence
                DDPDataType.RGB24,  # Data type
                0,  # Destination ID
                0,  # Data offset (4 bytes)
                0,  # Data length
            )

            # Send the packet - don't wait for response since WLED doesn't reply to queries
            self.socket.sendto(packet, (self.config.host, self.config.port))
            return True

        except Exception as e:
            logger.warning(f"UDP connectivity test failed: {e}")
            return False

    def send_led_data(self, led_data: Union[bytes, np.ndarray]) -> TransmissionResult:
        """
        Send LED data to WLED controller.

        Args:
            led_data: RGB data for all LEDs - either bytes (led_count * 3)
                     or numpy array (led_count, 3)

        Returns:
            TransmissionResult with transmission metrics
        """
        start_time = time.time()
        errors = []
        if not self.is_connected or not self.socket:
            error_msg = "Not connected to WLED controller"
            logger.error(error_msg)
            return TransmissionResult(
                success=False,
                packets_sent=0,
                bytes_sent=0,
                transmission_time=time.time() - start_time,
                errors=[error_msg],
            )

        # Convert numpy array to bytes if needed
        if isinstance(led_data, np.ndarray):
            if led_data.shape != (self.config.led_count, 3):
                error_msg = f"Invalid LED data shape: expected {(self.config.led_count, 3)}, got {led_data.shape}"
                logger.error(error_msg)
                return TransmissionResult(
                    success=False,
                    packets_sent=0,
                    bytes_sent=0,
                    transmission_time=time.time() - start_time,
                    errors=[error_msg],
                )
            led_data_bytes = np.clip(led_data, 0, 255).astype(np.uint8).flatten().tobytes()
        else:
            if len(led_data) != self.config.led_count * 3:
                error_msg = f"Invalid LED data size: expected {self.config.led_count * 3}, got {len(led_data)}"
                logger.error(error_msg)
                return TransmissionResult(
                    success=False,
                    packets_sent=0,
                    bytes_sent=0,
                    transmission_time=time.time() - start_time,
                    errors=[error_msg],
                )
            led_data_bytes = led_data

        # Flow control - enforce minimum frame interval
        current_time = time.time()
        with self._lock:
            time_since_last = current_time - self.last_frame_time
            min_interval = 1.0 / self.config.max_fps

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)
                current_time = time.time()

            self.last_frame_time = current_time

        # Send data in fragments
        result = self._send_fragmented_data(led_data_bytes)

        # Store last LED data for keepalive if transmission was successful
        if result.success:
            with self._lock:
                self._last_led_data = led_data_bytes
                self._last_data_time = current_time

        # Calculate frame rate
        if result.success and self.last_frame_time > 0:
            frame_interval = current_time - self.last_frame_time
            result.frame_rate = 1.0 / frame_interval if frame_interval > 0 else 0.0

        # Update statistics
        if result.success:
            with self._lock:
                self.frames_sent += 1
        else:
            with self._lock:
                self.transmission_errors += 1

        return result

    def _send_fragmented_data(self, led_data: bytes) -> TransmissionResult:
        """
        Send LED data fragmented into multiple DDP packets.

        Args:
            led_data: Complete LED data to fragment and send

        Returns:
            TransmissionResult with transmission metrics
        """
        start_time = time.time()
        packets_sent = 0
        bytes_sent = 0
        errors = []
        try:
            data_offset = 0

            for packet_index in range(self.packets_per_frame):
                # Determine data size for this packet
                if packet_index == self.packets_per_frame - 1:
                    # Last packet
                    packet_data_size = self.last_packet_size
                else:
                    packet_data_size = self.data_per_packet

                # Extract data for this packet
                packet_data = led_data[data_offset : data_offset + packet_data_size]

                # Create DDP packet
                packet_result = self._send_ddp_packet(
                    data=packet_data,
                    data_offset=data_offset,
                    is_last_packet=(packet_index == self.packets_per_frame - 1),
                )

                if not packet_result:
                    error_msg = f"Failed to send packet {packet_index + 1}/{self.packets_per_frame}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                else:
                    packets_sent += 1
                    bytes_sent += len(packet_data) + DDP_HEADER_SIZE

                data_offset += packet_data_size

                with self._lock:
                    if packet_result:
                        self.packets_sent += 1

            transmission_time = time.time() - start_time
            success = len(errors) == 0

            return TransmissionResult(
                success=success,
                packets_sent=packets_sent,
                bytes_sent=bytes_sent,
                transmission_time=transmission_time,
                errors=errors,
            )

        except Exception as e:
            error_msg = f"Fragmented transmission error: {e}"
            logger.error(error_msg)
            return TransmissionResult(
                success=False,
                packets_sent=packets_sent,
                bytes_sent=bytes_sent,
                transmission_time=time.time() - start_time,
                errors=[error_msg],
            )

    def _send_ddp_packet(self, data: bytes, data_offset: int, is_last_packet: bool = False) -> bool:
        """
        Send a single DDP packet.

        Args:
            data: Packet data payload
            data_offset: Offset in the complete LED data array
            is_last_packet: Whether this is the last packet in the frame

        Returns:
            True if packet sent successfully, False otherwise
        """
        try:
            # Generate sequence number
            with self._lock:
                sequence = self.sequence_number
                self.sequence_number = (self.sequence_number + 1) % 256

            # Set flags
            flags = int(DDPFlags.VER1)
            if is_last_packet:
                flags |= int(DDPFlags.PUSH)  # Display data after last packet

            # Create DDP header
            header = struct.pack(
                ">BBBBLH",
                flags,  # Flags (1 byte)
                sequence,  # Sequence (1 byte)
                DDPDataType.RGB24,  # Data type (1 byte)
                0,  # Destination ID (1 byte)
                data_offset,  # Data offset (4 bytes)
                len(data),
            )  # Data length (2 bytes)

            # Combine header and data
            packet = header + data

            # Send packet with retries
            for attempt in range(self.config.retry_count):
                try:
                    if self.socket is None:
                        return False
                    self.socket.sendto(packet, (self.config.host, self.config.port))
                    return True
                except socket.timeout:
                    logger.warning(f"Packet send timeout, attempt {attempt + 1}/{self.config.retry_count}")
                except Exception as e:
                    logger.warning(f"Packet send error: {e}, attempt {attempt + 1}/{self.config.retry_count}")

                if attempt < self.config.retry_count - 1:
                    time.sleep(0.01)  # Brief delay before retry

            logger.error(f"Failed to send packet after {self.config.retry_count} attempts")
            return False

        except Exception as e:
            logger.error(f"DDP packet creation error: {e}")
            return False

    def set_solid_color(self, r: int, g: int, b: int) -> TransmissionResult:
        """
        Set all LEDs to a solid color.

        Args:
            r, g, b: RGB color values (0-255)

        Returns:
            TransmissionResult with transmission metrics
        """
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            error_msg = f"Invalid RGB values: ({r}, {g}, {b})"
            logger.error(error_msg)
            return TransmissionResult(
                success=False,
                packets_sent=0,
                bytes_sent=0,
                transmission_time=0.0,
                errors=[error_msg],
            )

        # Create solid color data
        led_data = bytes([r, g, b] * self.config.led_count)
        return self.send_led_data(led_data)

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
            result = self.send_led_data(led_data)

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
        led_data = np.zeros((self.config.led_count, 3), dtype=np.uint8)

        if pattern == "rainbow":
            # Rainbow pattern
            for i in range(self.config.led_count):
                hue = (i / self.config.led_count) * 360
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

    def get_statistics(self) -> dict:
        """
        Get transmission statistics.

        Returns:
            Dictionary with transmission statistics
        """
        with self._lock:
            # Calculate throughput if we have transmission data
            total_bytes = self.packets_sent * DDP_HEADER_SIZE  # Approximate
            if self.last_frame_time > 0 and self.frames_sent > 0:
                total_time = time.time() - (self.last_frame_time - (self.frames_sent / self.config.max_fps))
                throughput_mbps = (total_bytes * 8) / (total_time * 1_000_000) if total_time > 0 else 0.0
                avg_fps = self.frames_sent / total_time if total_time > 0 else 0.0
            else:
                throughput_mbps = 0.0
                avg_fps = 0.0

            return {
                "frames_sent": self.frames_sent,
                "packets_sent": self.packets_sent,
                "transmission_errors": self.transmission_errors,
                "packets_per_frame": self.packets_per_frame,
                "is_connected": self.is_connected,
                "host": self.config.host,
                "port": self.config.port,
                "led_count": self.config.led_count,
                "wled_status": self.wled_status,
                "max_fps": self.config.max_fps,
                "estimated_throughput_mbps": throughput_mbps,
                "average_fps": avg_fps,
                "keepalive_enabled": self._enable_keepalive,
                "keepalive_active": self._keepalive_thread is not None and self._keepalive_thread.is_alive(),
                "keepalive_interval": self._keepalive_interval,
                "last_data_time": self._last_data_time,
            }

    def reset_statistics(self) -> None:
        """Reset transmission statistics."""
        with self._lock:
            self.frames_sent = 0
            self.packets_sent = 0
            self.transmission_errors = 0

    def get_wled_status(self) -> Optional[Dict[str, Any]]:
        """
        Get cached WLED status from last connection.

        Returns:
            WLED status dictionary or None if not available
        """
        return self.wled_status

    def refresh_wled_status(self) -> bool:
        """
        Refresh WLED status by sending a new query.

        Returns:
            True if status refreshed successfully, False otherwise
        """
        if not self.is_connected or not self.socket:
            return False

        try:
            success, status = self._send_query()
            if success:
                self.wled_status = status
                return True
        except Exception as e:
            logger.warning(f"Failed to refresh WLED status: {e}")

        return False

    def _start_keepalive_thread(self) -> None:
        """Start the keepalive thread to repeat last LED pattern."""
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            return  # Thread already running

        self._keepalive_stop_event.clear()
        self._keepalive_thread = threading.Thread(target=self._keepalive_worker, name="WLEDKeepalive", daemon=True)
        self._keepalive_thread.start()
        logger.info("Started WLED keepalive thread")

    def _stop_keepalive_thread(self) -> None:
        """Stop the keepalive thread."""
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            self._keepalive_stop_event.set()
            self._keepalive_thread.join(timeout=2.0)
            if self._keepalive_thread.is_alive():
                logger.warning("Keepalive thread did not stop cleanly")
            else:
                logger.info("Stopped WLED keepalive thread")

        # Always clear the thread reference
        self._keepalive_thread = None

    def _keepalive_worker(self) -> None:
        """
        Background thread worker that sends keepalive packets.

        Repeats the last LED pattern every second if no new data has been sent
        to prevent WLED from reverting to local patterns.
        """
        logger.debug("WLED keepalive worker thread started")

        while not self._keepalive_stop_event.is_set():
            try:
                # Wait for the keepalive interval or stop event
                if self._keepalive_stop_event.wait(self._keepalive_interval):
                    break  # Stop event was set

                # Check if we're still connected
                if not self.is_connected or not self.socket:
                    continue

                current_time = time.time()

                # Check if we have data to repeat and if enough time has passed
                with self._lock:
                    last_data = self._last_led_data
                    last_time = self._last_data_time

                # Only send keepalive if:
                # 1. We have LED data to repeat
                # 2. It's been more than keepalive_interval since last data
                if last_data is not None and current_time - last_time >= self._keepalive_interval:
                    try:
                        # Send the same data again (bypass flow control for keepalive)
                        result = self._send_fragmented_data(last_data)
                        if result.success:
                            logger.debug("Sent WLED keepalive packet")
                        else:
                            logger.warning(f"Keepalive packet failed: {result.errors}")

                    except Exception as e:
                        logger.warning(f"Keepalive transmission error: {e}")

            except Exception as e:
                logger.error(f"Keepalive worker error: {e}")
                # Continue running despite errors

        logger.debug("WLED keepalive worker thread stopped")

    def set_keepalive_interval(self, interval: float) -> None:
        """
        Set the keepalive interval.

        Args:
            interval: Keepalive interval in seconds
        """
        if interval < 0.1:
            raise ValueError("Keepalive interval must be at least 0.1 seconds")

        with self._lock:
            self._keepalive_interval = interval

        logger.info(f"Set WLED keepalive interval to {interval}s")

    def set_keepalive_enabled(self, enabled: bool) -> None:
        """
        Enable or disable keepalive functionality.

        Args:
            enabled: Whether to enable keepalive
        """
        with self._lock:
            was_enabled = self._enable_keepalive
            self._enable_keepalive = enabled

        if enabled and not was_enabled and self.is_connected:
            # Start keepalive if we're connected and it wasn't running
            self._start_keepalive_thread()
        elif not enabled and was_enabled:
            # Stop keepalive if it was running
            self._stop_keepalive_thread()

        logger.info(f"WLED keepalive {'enabled' if enabled else 'disabled'}")

    def __enter__(self):
        """Context manager entry."""
        if self.connect():
            return self
        else:
            raise ConnectionError(f"Failed to connect to WLED at {self.config.host}:{self.config.port}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
