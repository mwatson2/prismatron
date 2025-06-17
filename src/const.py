"""
Global constants for the Prismatron LED Display System.

This module contains all system-wide constants used across different components
of the Prismatron software stack.
"""

import numpy as np

# Frame dimensions for 1080p RGB
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_CHANNELS = 3  # RGB
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS
BUFFER_COUNT = 3  # Triple buffering

# Shared metadata numpy dtype - used by both producer and consumer
METADATA_DTYPE = np.dtype([
    ('presentation_timestamp', np.float64),
    ('source_width', np.int32),
    ('source_height', np.int32), 
    ('capture_timestamp', np.float64)
])