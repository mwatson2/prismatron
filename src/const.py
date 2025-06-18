"""
Global constants for the Prismatron LED Display System.

This module contains all system-wide constants used across different components
of the Prismatron software stack.
"""

import numpy as np

# Frame dimensions for 5:3 aspect ratio RGB
# Using dimensions that are efficient for LED optimization
FRAME_WIDTH = 1000  # 5:3 ratio, divisible by common factors
FRAME_HEIGHT = 600  # 5:3 ratio, divisible by common factors
FRAME_CHANNELS = 3  # RGB
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS
BUFFER_COUNT = 3  # Triple buffering

# Shared metadata numpy dtype - used by both producer and consumer
METADATA_DTYPE: np.dtype = np.dtype(
    [
        ("presentation_timestamp", np.float64),
        ("source_width", np.int32),
        ("source_height", np.int32),
        ("capture_timestamp", np.float64),
    ]
)

# LED Hardware Configuration
LED_COUNT = 3200  # Total number of RGB LEDs in the display
LED_DATA_SIZE = LED_COUNT * 3  # RGB data size in bytes

# WLED Communication Configuration
WLED_DEFAULT_HOST = "wled.local"  # Default WLED controller hostname
WLED_DEFAULT_PORT = 21324  # Default port for DDP protocol
WLED_DDP_PORT = 21324  # DDP (Distributed Display Protocol) port
WLED_UDP_RAW_PORT = 19446  # UDP Raw protocol port
WLED_E131_PORT = 5568  # E1.31/sACN protocol port

# DDP Protocol Constants
DDP_HEADER_SIZE = 10  # DDP header is 10 bytes
DDP_MAX_PACKET_SIZE = 1440  # Maximum DDP packet size (safe for most networks)
DDP_MAX_DATA_PER_PACKET = DDP_MAX_PACKET_SIZE - DDP_HEADER_SIZE  # Data payload size

# Flow Control and Timing
WLED_MAX_FPS = 60  # Maximum frame rate for WLED updates
WLED_MIN_FRAME_INTERVAL = 1.0 / WLED_MAX_FPS  # Minimum time between frames
WLED_TIMEOUT_SECONDS = 5.0  # Network timeout for WLED communication
WLED_RETRY_COUNT = 3  # Number of retries for failed transmissions
