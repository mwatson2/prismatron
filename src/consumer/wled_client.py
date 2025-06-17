"""
WLED UDP Communication Client.

This module implements a client for communicating with WLED controllers over UDP
using the DDP (Distributed Display Protocol) for efficient LED data transmission.
Includes flow control, packet fragmentation, and error handling.
"""

import socket
import struct
import time
import logging
import threading
import json
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

from ..const import (
    LED_COUNT, LED_DATA_SIZE, WLED_DEFAULT_HOST, WLED_DEFAULT_PORT,
    DDP_HEADER_SIZE, DDP_MAX_DATA_PER_PACKET, WLED_MIN_FRAME_INTERVAL,
    WLED_TIMEOUT_SECONDS, WLED_RETRY_COUNT
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
            
        logger.info(f"LED array fragmentation: {self.packets_per_frame} packets, "
                   f"{self.data_per_packet} bytes per packet, "
                   f"last packet {self.last_packet_size} bytes")
    
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
                        logger.info(f"Connected to WLED '{status.get('name', 'Unknown')}' "
                                  f"v{status.get('ver', 'Unknown')} at {self.config.host}:{self.config.port}")
                        
                        # Validate LED count if available
                        wled_led_count = status.get('leds', {}).get('count', 0) if isinstance(status.get('leds'), dict) else 0
                        if wled_led_count > 0 and wled_led_count != self.config.led_count:
                            logger.warning(f"LED count mismatch: configured {self.config.led_count}, "
                                         f"WLED reports {wled_led_count}")
                    else:
                        logger.info(f"Connected to WLED controller at {self.config.host}:{self.config.port} "
                                  f"(no status response)")
                    
                    return True
                else:
                    if not self.config.persistent_retry:
                        logger.error(f"Failed to connect to WLED controller at {self.config.host}:{self.config.port}")
                        self.disconnect()
                        return False
                    
                    # Persistent retry mode
                    elapsed = time.time() - start_time
                    if attempt == 1:
                        logger.info(f"Attempting to connect to WLED controller at {self.config.host}:{self.config.port} "
                                  f"(persistent retry enabled, interval: {self.config.retry_interval}s)")
                    else:
                        logger.info(f"Connection attempt {attempt} failed after {elapsed:.1f}s, "
                                  f"retrying in {self.config.retry_interval}s...")
                    
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
                logger.warning(f"Connection error after {elapsed:.1f}s: {e}, "
                              f"retrying in {self.config.retry_interval}s...")
                self.disconnect()
                time.sleep(self.config.retry_interval)
    
    def disconnect(self) -> None:
        """Disconnect from WLED controller."""
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
        Send a query packet to test connectivity and get WLED status.
        
        Returns:
            Tuple of (success, status_dict) where status_dict contains WLED info
        """
        try:
            # Create DDP query packet for JSON status (Destination ID 251)
            flags = DDPFlags.VER1 | DDPFlags.QUERY
            packet = struct.pack('>BBBBLH', 
                               flags,           # Flags
                               0,               # Sequence (not used for query)
                               DDPDataType.RGB24,  # Data type
                               251,             # Destination ID 251 for JSON status
                               0,               # Data offset (4 bytes)
                               0)               # Data length
            
            # Send query packet
            self.socket.sendto(packet, (self.config.host, self.config.port))
            
            # Wait for response
            try:
                # Use a larger buffer to accommodate JSON responses
                # WLED JSON status can be quite large (1KB+ with all info)
                max_response_size = 8192
                response_data, addr = self.socket.recvfrom(max_response_size)
                
                # Validate minimum packet size for DDP header
                if len(response_data) < DDP_HEADER_SIZE:
                    logger.warning(f"Received truncated packet: {len(response_data)} bytes, "
                                 f"expected at least {DDP_HEADER_SIZE}")
                    return True, None  # Connection works, but invalid packet
                
                # Parse DDP response header
                flags, sequence, data_type, dest_id, offset, data_length = struct.unpack(
                    '>BBBBLH', response_data[:DDP_HEADER_SIZE])
                
                # Check if this is a valid DDP reply
                if not ((flags & DDPFlags.VER1) and (flags & DDPFlags.REPLY)):
                    logger.warning(f"Received non-DDP reply packet (flags=0x{flags:02x})")
                    return True, None  # Connection works, but unexpected response
                
                # Validate packet completeness
                expected_total_size = DDP_HEADER_SIZE + data_length
                if len(response_data) < expected_total_size:
                    logger.warning(f"Received incomplete packet: {len(response_data)} bytes, "
                                 f"expected {expected_total_size} (header + {data_length} data)")
                    return True, None  # Connection works, but truncated data
                
                # Check for reasonable data length (prevent excessive memory usage)
                if data_length > max_response_size - DDP_HEADER_SIZE:
                    logger.warning(f"Response data length too large: {data_length} bytes")
                    return True, None  # Connection works, but data too large
                
                # Extract JSON payload using actual data length from header
                if data_length > 0:
                    json_data = response_data[DDP_HEADER_SIZE:DDP_HEADER_SIZE + data_length]
                    
                    try:
                        status = json.loads(json_data.decode('utf-8'))
                        logger.info(f"WLED status received: {status.get('name', 'Unknown')} "
                                  f"v{status.get('ver', 'Unknown')} ({data_length} bytes)")
                        return True, status
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.warning(f"Failed to parse WLED JSON response: {e}")
                        logger.debug(f"Raw JSON data: {json_data[:100]}...")  # Log first 100 chars
                        return True, None  # Connection works, but no valid JSON
                else:
                    logger.warning("Received DDP reply with no data payload")
                    return True, None  # Connection works, but no data
                    
            except socket.timeout:
                logger.warning("No response to query packet (timeout)")
                return False, None  # No response - connection failed
                
        except Exception as e:
            logger.error(f"Query packet error: {e}")
            return False, None
    
    def send_led_data(self, led_data: bytes) -> bool:
        """
        Send LED data to WLED controller.
        
        Args:
            led_data: RGB data for all LEDs (led_count * 3 bytes)
            
        Returns:
            True if transmission successful, False otherwise
        """
        if not self.is_connected or not self.socket:
            logger.error("Not connected to WLED controller")
            return False
            
        if len(led_data) != self.config.led_count * 3:
            logger.error(f"Invalid LED data size: expected {self.config.led_count * 3}, got {len(led_data)}")
            return False
        
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
        success = self._send_fragmented_data(led_data)
        
        if success:
            with self._lock:
                self.frames_sent += 1
        else:
            with self._lock:
                self.transmission_errors += 1
                
        return success
    
    def _send_fragmented_data(self, led_data: bytes) -> bool:
        """
        Send LED data fragmented into multiple DDP packets.
        
        Args:
            led_data: Complete LED data to fragment and send
            
        Returns:
            True if all packets sent successfully, False otherwise
        """
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
                packet_data = led_data[data_offset:data_offset + packet_data_size]
                
                # Create DDP packet
                success = self._send_ddp_packet(
                    data=packet_data,
                    data_offset=data_offset,
                    is_last_packet=(packet_index == self.packets_per_frame - 1)
                )
                
                if not success:
                    logger.error(f"Failed to send packet {packet_index + 1}/{self.packets_per_frame}")
                    return False
                
                data_offset += packet_data_size
                
                with self._lock:
                    self.packets_sent += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Fragmented transmission error: {e}")
            return False
    
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
            flags = DDPFlags.VER1
            if is_last_packet:
                flags |= DDPFlags.PUSH  # Display data after last packet
            
            # Create DDP header
            header = struct.pack('>BBBBLH',
                               flags,                    # Flags (1 byte)
                               sequence,                 # Sequence (1 byte)
                               DDPDataType.RGB24,        # Data type (1 byte) 
                               0,                        # Destination ID (1 byte)
                               data_offset,              # Data offset (4 bytes)
                               len(data))                # Data length (2 bytes)
            
            # Combine header and data
            packet = header + data
            
            # Send packet with retries
            for attempt in range(self.config.retry_count):
                try:
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
    
    def set_solid_color(self, r: int, g: int, b: int) -> bool:
        """
        Set all LEDs to a solid color.
        
        Args:
            r, g, b: RGB color values (0-255)
            
        Returns:
            True if successful, False otherwise
        """
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            logger.error(f"Invalid RGB values: ({r}, {g}, {b})")
            return False
        
        # Create solid color data
        led_data = bytes([r, g, b] * self.config.led_count)
        return self.send_led_data(led_data)
    
    def get_statistics(self) -> dict:
        """
        Get transmission statistics.
        
        Returns:
            Dictionary with transmission statistics
        """
        with self._lock:
            return {
                'frames_sent': self.frames_sent,
                'packets_sent': self.packets_sent,
                'transmission_errors': self.transmission_errors,
                'packets_per_frame': self.packets_per_frame,
                'is_connected': self.is_connected,
                'host': self.config.host,
                'port': self.config.port,
                'led_count': self.config.led_count,
                'wled_status': self.wled_status
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
    
    def __enter__(self):
        """Context manager entry."""
        if self.connect():
            return self
        else:
            raise ConnectionError(f"Failed to connect to WLED at {self.config.host}:{self.config.port}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()