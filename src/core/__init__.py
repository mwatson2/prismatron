"""
Core Infrastructure Components.

This module contains the fundamental building blocks for the Prismatron system:
- Shared memory ring buffer for inter-process communication
- Control state manager for process coordination
- Common utilities and base classes
"""

from .control_state import ControlState, PlayState, SystemState, SystemStatus
from .shared_buffer import (
    SHARED_MEMORY_AVAILABLE,
    FrameConsumer,
    FrameProducer,
    FrameRingBuffer,
)

__all__ = [
    "FrameRingBuffer",
    "FrameConsumer",
    "FrameProducer",
    "SHARED_MEMORY_AVAILABLE",
    "ControlState",
    "PlayState",
    "SystemState",
    "SystemStatus",
]
