"""
Global constants for the Prismatron LED Display System.

This module contains all system-wide constants used across different components
of the Prismatron software stack.
"""

import numpy as np

# Frame dimensions for 5:3 aspect ratio RGB
# Using smaller dimensions for memory efficiency and speed
FRAME_WIDTH = 800  # 5:3 ratio, reduced for performance
FRAME_HEIGHT = 480  # 5:3 ratio, reduced for performance
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
        ("playlist_item_index", np.int32),  # Current playlist item index for renderer sync
        ("is_first_frame_of_item", np.bool_),  # True if this is the first frame of a new playlist item
        # Timing data fields for performance analysis
        ("frame_index", np.int32),
        ("plugin_timestamp", np.float64),
        ("producer_timestamp", np.float64),
        ("item_duration", np.float64),
        ("write_to_buffer_time", np.float64),
        ("read_from_buffer_time", np.float64),
        # Transition data fields for playlist transitions
        ("transition_in_type", "U16"),  # Transition in type (e.g., "fade", "none")
        ("transition_in_duration", np.float64),  # Transition in duration in seconds
        ("transition_out_type", "U16"),  # Transition out type (e.g., "fade", "none")
        ("transition_out_duration", np.float64),  # Transition out duration in seconds
        ("item_timestamp", np.float64),  # Time within current item (for transition calculations)
    ]
)

# LED Hardware Configuration
LED_COUNT = 2624  # Total number of RGB LEDs in the display (updated to match diffusion patterns)
LED_DATA_SIZE = LED_COUNT * 3  # RGB data size in bytes

# WLED Communication Configuration
WLED_DEFAULT_HOST = "wled.local"  # Default WLED controller hostname
WLED_DEFAULT_PORT = 4048  # Default port for DDP protocol
WLED_DDP_PORT = 4048  # DDP (Distributed Display Protocol) port
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

# Content Source Frame Rate Configuration
DEFAULT_CONTENT_FPS = 24.0  # Default frame rate for static content (images, text effects)

# Log Management Configuration
LOG_MAX_SIZE_MB = 100  # Maximum log file size in MB before rotation
