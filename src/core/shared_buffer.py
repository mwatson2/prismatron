"""
Shared Memory Ring Buffer for zero-copy frame sharing between processes.

This module implements a triple-buffered ring buffer using multiprocessing
shared memory for efficient frame data transfer between producer and consumer
processes without copying data.
"""

import contextlib
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..const import (
    BUFFER_COUNT,
    FRAME_CHANNELS,
    FRAME_HEIGHT,
    FRAME_SIZE,
    FRAME_WIDTH,
    METADATA_DTYPE,
)
from ..utils.frame_timing import FrameTimingData

# Check for shared_memory availability (Python 3.8+)
try:
    from multiprocessing import shared_memory

    SHARED_MEMORY_AVAILABLE = True
except ImportError:
    # Fallback for older Python versions
    SHARED_MEMORY_AVAILABLE = False
    shared_memory = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class FrameMetadata:
    """Metadata for a frame in the ring buffer."""

    presentation_timestamp: float  # When this frame should be displayed
    source_width: int  # Actual content width within buffer
    source_height: int  # Actual content height within buffer
    capture_timestamp: float  # When this frame was captured/created
    playlist_item_index: int = -1  # Current playlist item index for renderer sync
    is_first_frame_of_item: bool = False  # True if this is the first frame of a new playlist item
    timing_data: Optional[FrameTimingData] = None  # Detailed timing information for performance analysis
    # Transition fields for playlist item transitions
    transition_in_type: str = "none"  # Transition in type (e.g., "fade", "none")
    transition_in_duration: float = 0.0  # Transition in duration in seconds
    transition_out_type: str = "none"  # Transition out type (e.g., "fade", "none")
    transition_out_duration: float = 0.0  # Transition out duration in seconds
    item_timestamp: float = 0.0  # Time within current item (for transition calculations)
    item_duration: float = 0.0  # Total duration of current item


@dataclass
class BufferInfo:
    """Information about a shared memory buffer."""

    buffer: memoryview  # Raw buffer access
    timestamp: float
    frame_id: int
    buffer_index: int  # Actual buffer index in circular buffer
    metadata: Optional[FrameMetadata] = None

    def get_array(self, width: int, height: int, channels: int = FRAME_CHANNELS) -> np.ndarray:
        """Create numpy array view of the buffer with specified dimensions in PLANAR format."""
        # Buffer stores data in planar format (C, H, W)
        return np.ndarray((channels, height, width), dtype=np.uint8, buffer=self.buffer)

    def get_array_interleaved(self, width: int, height: int, channels: int = FRAME_CHANNELS) -> np.ndarray:
        """Create numpy array view of the buffer in interleaved format (H, W, C) for legacy support."""
        # Get planar array and convert to interleaved
        planar_array = self.get_array(width, height, channels)
        return np.transpose(planar_array, (1, 2, 0))  # (C, H, W) -> (H, W, C)


class FrameRingBuffer:
    """
    Triple-buffered ring buffer for sharing 1080p RGB frames between processes.

    Uses multiprocessing shared memory for zero-copy data transfer. Provides
    event-based synchronization between producer and consumer processes.
    """

    def __init__(self, name: str = "prismatron_buffer"):
        """
        Initialize the ring buffer.

        The ring buffer uses a triple-buffering scheme where:
        - Producer writes to the current write buffer
        - Consumer reads from the most recent complete buffer
        - One buffer may be transitioning between read/write states

        Args:
            name: Unique name for the shared memory segment (must be unique across system)
        """
        self.name = name
        self.buffer_size = FRAME_SIZE
        self.buffer_count = BUFFER_COUNT

        # Shared memory storage for frame data (one per buffer)
        self._shared_memory: List[Any] = []

        # Shared memory for frame metadata (one record per buffer)
        self._metadata_memory: Optional[Any] = None
        self._metadata_array: Optional[np.ndarray] = None

        # Control variables stored in shared memory for cross-process coordination
        self._control_memory: Optional[Any] = None
        self._control_array: Optional[np.ndarray] = None

        # Multiprocessing synchronization primitives
        # Note: Events don't work reliably across processes, we use polling instead
        self._lock = mp.Lock()  # Protects critical sections

        # Local state (not shared across processes)
        self._local_read_index = 0  # Last frame we've read
        self._local_read_buffer = 0  # Next buffer index to read from
        self._local_frame_counter = 0

        self._initialized = False

    def _create_array_views_from_shared_memory(self) -> bool:
        """
        Create metadata and control array views from existing shared memory segments.

        This helper method is used by both initialize() and connect() methods
        after the shared memory segments have been created or connected to.
        Frame buffers are accessed directly as raw memory.

        Returns:
            True if array views were created successfully, False otherwise
        """
        try:
            # Create structured array view of metadata
            if self._metadata_memory:
                self._metadata_array = np.ndarray(
                    (self.buffer_count,),
                    dtype=METADATA_DTYPE,
                    buffer=self._metadata_memory.buf,
                )

            # Create control array view
            if self._control_memory:
                self._control_array = np.ndarray(
                    (4 + self.buffer_count,),
                    dtype=np.float64,
                    buffer=self._control_memory.buf,
                )

            return True

        except Exception as e:
            logger.error(f"Failed to create array views: {e}")
            return False

    # Control structure helper methods
    def _get_write_index(self) -> int:
        """Get current write buffer index."""
        if not self._initialized or self._control_array is None:
            raise RuntimeError("Buffer must be initialized and have control array")
        return int(self._control_array[0])

    def _set_write_index(self, index: int) -> None:
        """Set current write buffer index."""
        if not self._initialized or self._control_array is None:
            raise RuntimeError("Buffer must be initialized and have control array")
        self._control_array[0] = index

    def _get_read_index(self) -> int:
        """Get current read buffer index."""
        if not self._initialized or self._control_array is None:
            raise RuntimeError("Buffer must be initialized and have control array")
        return int(self._control_array[1])

    def _set_read_index(self, index: int) -> None:
        """Set current read buffer index."""
        if not self._initialized or self._control_array is None:
            raise RuntimeError("Buffer must be initialized and have control array")
        self._control_array[1] = index

    def _get_frame_counter(self) -> int:
        """Get global frame counter."""
        if not self._initialized or self._control_array is None:
            raise RuntimeError("Buffer must be initialized and have control array")
        return int(self._control_array[2])

    def _increment_frame_counter(self) -> int:
        """Increment and return global frame counter."""
        if not self._initialized or self._control_array is None:
            raise RuntimeError("Buffer must be initialized and have control array")
        self._control_array[2] += 1
        return int(self._control_array[2])

    def _get_min_consumed_frame(self) -> int:
        """Get minimum consumed frame counter."""
        if not self._initialized or self._control_array is None:
            raise RuntimeError("Buffer must be initialized and have control array")
        return int(self._control_array[3])

    def _set_min_consumed_frame(self, frame: int) -> None:
        """Set minimum consumed frame counter."""
        if not self._initialized or self._control_array is None:
            raise RuntimeError("Buffer must be initialized and have control array")
        self._control_array[3] = frame

    def _get_buffer_timestamp(self, buffer_index: int) -> float:
        """Get timestamp for specific buffer."""
        if not self._initialized or self._control_array is None:
            raise RuntimeError("Buffer must be initialized and have control array")
        if not (0 <= buffer_index < self.buffer_count):
            raise IndexError(f"Buffer index {buffer_index} out of range [0, {self.buffer_count})")
        return float(self._control_array[4 + buffer_index])

    def _set_buffer_timestamp(self, buffer_index: int, timestamp: float) -> None:
        """Set timestamp for specific buffer."""
        if not self._initialized or self._control_array is None:
            raise RuntimeError("Buffer must be initialized and have control array")
        if not (0 <= buffer_index < self.buffer_count):
            raise IndexError(f"Buffer index {buffer_index} out of range [0, {self.buffer_count})")
        self._control_array[4 + buffer_index] = timestamp

    def _is_write_buffer_available(self, write_idx: int, current_frame: int) -> bool:
        """
        Check if the write buffer is available for writing.

        A buffer is available if the frame that would be overwritten has been consumed.
        With triple buffering, we can write up to 3 frames ahead of consumption.

        Args:
            write_idx: The write buffer index to check
            current_frame: The current frame counter

        Returns:
            True if buffer is available for writing
        """
        # Get the minimum consumed frame from shared memory
        min_consumed_frame = self._get_min_consumed_frame()

        # We can write if the frame we're about to write is not more than
        # buffer_count frames ahead of consumption
        next_frame = current_frame + 1
        frames_ahead = next_frame - min_consumed_frame

        # For triple buffering, we should only allow buffer_count frames ahead
        # This ensures we don't overwrite unconsumed frames
        max_frames_ahead = self.buffer_count

        # Special case: if no consumer has started yet (min_consumed_frame == 0),
        # only allow filling the initial buffers once, then block
        if min_consumed_frame == 0:
            # Only allow buffer_count frames total before blocking for consumer
            return current_frame < self.buffer_count

        return frames_ahead <= max_frames_ahead

    def get_status(self) -> Dict:
        """
        Get current status of the ring buffer.

        Returns:
            Dictionary with status information including buffer states and timing
        """
        if not self._initialized:
            return {"initialized": False}

        try:
            with self._lock:
                # Lock assertion removed - multiprocessing.Lock doesn't support blocking parameter
                return {
                    "initialized": True,
                    "write_index": self._get_write_index(),
                    "read_index": self._get_read_index(),
                    "frame_counter": self._get_frame_counter(),
                    "min_consumed_frame": self._get_min_consumed_frame(),
                    "local_read_index": self._local_read_index,
                    "buffer_count": self.buffer_count,
                    "buffer_size": self.buffer_size,
                    "buffer_timestamps": [self._get_buffer_timestamp(i) for i in range(self.buffer_count)],
                    "frames_behind": self._get_frame_counter() - self._local_read_index,
                }
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}

    def get_write_buffer(
        self,
        timeout: float = 1.0,
        presentation_timestamp: Optional[float] = None,
        source_width: Optional[int] = None,
        source_height: Optional[int] = None,
    ) -> Optional[BufferInfo]:
        """
        Get write buffer (fallback for base class usage).

        Note: This is provided for backwards compatibility. For new code,
        use FrameProducer.get_write_buffer() instead.
        """
        logger.warning("get_write_buffer called on base class - consider using FrameProducer")
        return None

    def advance_write(self) -> bool:
        """
        Advance write (fallback for base class usage).

        Note: This is provided for backwards compatibility. For new code,
        use FrameProducer.advance_write() instead.
        """
        logger.warning("advance_write called on base class - consider using FrameProducer")
        return False

    def wait_for_ready_buffer(self, timeout: float = 1.0) -> Optional[BufferInfo]:
        """
        Wait for ready buffer (fallback for base class usage).

        Note: This is provided for backwards compatibility. For new code,
        use FrameConsumer.wait_for_ready_buffer() instead.
        """
        logger.warning("wait_for_ready_buffer called on base class - consider using FrameConsumer")
        return None

    def release_read_buffer(self) -> bool:
        """
        Release read buffer (fallback for base class usage).

        Note: This is provided for backwards compatibility. For new code,
        use FrameConsumer.release_read_buffer() instead.
        """
        logger.warning("release_read_buffer called on base class - consider using FrameConsumer")
        return False

    def cleanup(self) -> None:
        """
        Clean up shared memory resources.

        This method safely releases all shared memory segments and should be
        called when the ring buffer is no longer needed. Only the process that
        created the buffer (called initialize()) should unlink the memory.
        """
        try:
            # Clean up frame buffers
            for i, shm in enumerate(self._shared_memory):
                try:
                    shm.close()  # Close this process's connection

                    # Only unlink if we created it (avoid errors from other processes)
                    with contextlib.suppress(FileNotFoundError):
                        shm.unlink()  # Remove from system (may fail if already removed)

                except Exception as e:
                    logger.warning(f"Error cleaning up shared memory buffer {i}: {e}")

            # Clean up metadata memory
            if self._metadata_memory:
                try:
                    self._metadata_memory.close()
                    with contextlib.suppress(FileNotFoundError):
                        self._metadata_memory.unlink()
                except Exception as e:
                    logger.warning(f"Error cleaning up metadata memory: {e}")

            # Clean up control memory
            if self._control_memory:
                try:
                    self._control_memory.close()
                    with contextlib.suppress(FileNotFoundError):
                        self._control_memory.unlink()
                except Exception as e:
                    logger.warning(f"Error cleaning up control memory: {e}")

            # Reset local state
            self._shared_memory.clear()
            self._metadata_memory = None
            self._metadata_array = None
            self._control_memory = None
            self._control_array = None
            self._initialized = False

            logger.info(f"Cleaned up ring buffer '{self.name}'")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, "_initialized") and self._initialized:
            self.cleanup()


class FrameProducer(FrameRingBuffer):
    """
    Producer-specific interface for the ring buffer.

    Provides methods optimized for frame production including writing frames
    with metadata, checking write availability, and monitoring buffer state.
    """

    def __init__(self, name: str = "prismatron_buffer"):
        """Initialize the producer."""
        super().__init__(name)
        self._frames_written = 0

    def initialize(self) -> bool:
        """Initialize the ring buffer for producer use."""
        if not SHARED_MEMORY_AVAILABLE:
            logger.error("Shared memory not available (requires Python 3.8+)")
            return False

        try:
            # Create shared memory for frame buffers
            for i in range(self.buffer_count):
                shm_name = f"{self.name}_buffer_{i}"
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.buffer_size)
                # Set group read/write permissions for shared memory
                import os
                import stat

                try:
                    os.chmod(f"/dev/shm/{shm_name}", stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
                except (OSError, PermissionError):
                    pass  # If we can't change permissions, continue anyway
                self._shared_memory.append(shm)

            # Create shared memory for frame metadata
            # Each metadata record: [presentation_timestamp,
            # source_width, source_height, capture_timestamp]
            # Types: [float64, int32, int32, float64] = 8 + 4 + 4 + 8 = 24 bytes per record
            metadata_size = METADATA_DTYPE.itemsize * self.buffer_count

            self._metadata_memory = shared_memory.SharedMemory(
                name=f"{self.name}_metadata", create=True, size=metadata_size
            )
            # Set group read/write permissions for metadata
            try:
                os.chmod(f"/dev/shm/{self.name}_metadata", stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
            except (OSError, PermissionError):
                pass  # If we can't change permissions, continue anyway

            # Create control structure for cross-process coordination
            # Layout: [write_idx, read_idx, frame_counter,
            # min_consumed_frame, timestamp0, timestamp1, timestamp2]
            control_size = 8 * (4 + self.buffer_count)  # 8 bytes per float64 value
            self._control_memory = shared_memory.SharedMemory(
                name=f"{self.name}_control", create=True, size=control_size
            )
            # Set group read/write permissions for control
            try:
                os.chmod(f"/dev/shm/{self.name}_control", stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
            except (OSError, PermissionError):
                pass  # If we can't change permissions, continue anyway

            # Create array views from shared memory
            if not self._create_array_views_from_shared_memory():
                return False

            # Frame buffers are left uninitialized - content will be written by producer

            # Initialize metadata to zeros
            if self._metadata_array is not None:
                self._metadata_array["presentation_timestamp"] = 0.0
                self._metadata_array["source_width"] = FRAME_WIDTH
                self._metadata_array["source_height"] = FRAME_HEIGHT
                self._metadata_array["capture_timestamp"] = 0.0
                # Initialize timing data fields
                self._metadata_array["frame_index"] = 0
                self._metadata_array["plugin_timestamp"] = 0.0
                self._metadata_array["producer_timestamp"] = 0.0
                self._metadata_array["item_duration"] = 0.0
                self._metadata_array["write_to_buffer_time"] = 0.0
                self._metadata_array["read_from_buffer_time"] = 0.0

            # Control array structure:
            # [0] = current write buffer index (0, 1, or 2)
            # [1] = current read buffer index (0, 1, or 2)
            # [2] = global frame counter (increments with each new frame)
            # [3] = minimum consumed frame (highest frame number that has been consumed)
            # [4] = timestamp when buffer 0 was last written
            # [5] = timestamp when buffer 1 was last written
            # [6] = timestamp when buffer 2 was last written
            if self._control_array is not None:
                self._control_array.fill(0)
                # Initialize read index to -1 to indicate no buffer is being read
                self._control_array[1] = -1

            self._initialized = True

            logger.info(f"Initialized ring buffer '{self.name}' with {self.buffer_count} buffers")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ring buffer: {e}")
            self.cleanup()
            return False

    def get_write_buffer(
        self,
        timeout: float = 1.0,
        presentation_timestamp: Optional[float] = None,
        source_width: Optional[int] = None,
        source_height: Optional[int] = None,
        playlist_item_index: Optional[int] = None,
        is_first_frame_of_item: bool = False,
    ) -> Optional[BufferInfo]:
        """
        Get the next buffer available for writing (producer process).

        This method waits for a buffer to become available for writing, ensuring
        the producer doesn't overwrite data that hasn't been consumed yet.

        Args:
            timeout: Maximum time to wait for available buffer in seconds
            presentation_timestamp: When this frame should be displayed (defaults to current time)
            source_width: Actual content width within buffer (defaults to full buffer width)
            source_height: Actual content height within buffer (defaults to full buffer height)
            playlist_item_index: Current playlist item index (for renderer synchronization)
            is_first_frame_of_item: True if this is the first frame of a new playlist item

        Returns:
            Dictionary with buffer info or None if not available within timeout
        """
        if not self._initialized:
            return None

        start_time = time.time()

        try:
            while (time.time() - start_time) < timeout:
                with self._lock:  # Protect against concurrent access
                    write_idx = self._get_write_index()
                    read_idx = self._get_read_index()
                    current_frame = self._get_frame_counter()

                    # Check if current write buffer is safe to write to
                    # We can't write to a buffer that the consumer is currently reading
                    # read_idx == -1 means no buffer is currently being read
                    # Also check if we've filled all buffers without any consumption
                    if (read_idx == -1 or write_idx != read_idx) and self._is_write_buffer_available(
                        write_idx, current_frame
                    ):
                        # Safe to write to current buffer
                        current_time = time.time()

                        # Set defaults for metadata
                        if presentation_timestamp is None:
                            presentation_timestamp = current_time
                        if source_width is None:
                            source_width = FRAME_WIDTH
                        if source_height is None:
                            source_height = FRAME_HEIGHT

                        # Store metadata for this buffer
                        if self._metadata_array is not None:
                            self._metadata_array[write_idx]["presentation_timestamp"] = presentation_timestamp
                            self._metadata_array[write_idx]["source_width"] = source_width
                            self._metadata_array[write_idx]["source_height"] = source_height
                            self._metadata_array[write_idx]["capture_timestamp"] = current_time
                            self._metadata_array[write_idx]["playlist_item_index"] = (
                                playlist_item_index if playlist_item_index is not None else -1
                            )
                            self._metadata_array[write_idx]["is_first_frame_of_item"] = is_first_frame_of_item

                        # Create metadata object for return
                        metadata = FrameMetadata(
                            presentation_timestamp=presentation_timestamp,
                            source_width=source_width,
                            source_height=source_height,
                            capture_timestamp=current_time,
                        )

                        buffer_info = BufferInfo(
                            buffer=memoryview(self._shared_memory[write_idx].buf),
                            timestamp=current_time,
                            frame_id=current_frame + 1,  # Next frame to be written
                            buffer_index=write_idx,  # Actual buffer index allocated
                            metadata=metadata,
                        )

                        return buffer_info

                # Buffer not available, wait briefly and retry
                time.sleep(0.001)  # 1ms polling interval

            # Timeout reached
            logger.warning(f"Write buffer timeout after {timeout}s")
            return None

        except Exception as e:
            logger.error(f"Failed to get write buffer: {e}")
            return None

    def advance_write(self) -> bool:
        """
        Signal that the current write buffer is complete and advance to next buffer.

        This atomically:
        1. Records the timestamp when this buffer was completed
        2. Increments the global frame counter
        3. Advances to the next write buffer

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False

        try:
            with self._lock:  # Atomic operation across all updates
                # Lock assertion removed - multiprocessing.Lock doesn't support blocking parameter
                current_time = time.time()
                write_idx = self._get_write_index()

                # Update timestamp for the buffer we just finished writing
                self._set_buffer_timestamp(write_idx, current_time)

                # Increment global frame counter (signals new data available)
                new_frame_count = self._increment_frame_counter()

                # Advance to next write buffer (circular)
                next_write_idx = (write_idx + 1) % self.buffer_count
                self._set_write_index(next_write_idx)

                # Update local counter for logging
                self._local_frame_counter += 1

                # Update frames written counter for producer stats
                if hasattr(self, "_frames_written"):
                    self._frames_written += 1

                return True

        except Exception as e:
            logger.error(f"Failed to advance write buffer: {e}")
            return False

    def get_producer_stats(self) -> Dict[str, Any]:
        """Get producer-specific statistics."""
        base_status = self.get_status()
        base_status.update(
            {
                "frames_written": self._frames_written,
                "is_producer": True,
                "write_buffer_available": (
                    self._is_write_buffer_available(int(self._control_array[0]), int(self._control_array[2]))
                    if self._initialized and self._control_array is not None
                    else False
                ),
            }
        )
        return base_status

    def can_write_frame(self) -> bool:
        """Check if a frame can be written immediately without waiting."""
        if not self._initialized:
            return False

        try:
            with self._lock:
                # Lock assertion removed - multiprocessing.Lock doesn't support blocking parameter
                if self._control_array is None:
                    return False
                write_idx = int(self._control_array[0])
                read_idx = int(self._control_array[1])
                current_frame = int(self._control_array[2])

                return (read_idx == -1 or write_idx != read_idx) and self._is_write_buffer_available(
                    write_idx, current_frame
                )
        except Exception:
            return False


class FrameConsumer(FrameRingBuffer):
    """
    Consumer-specific interface for the ring buffer.

    Provides methods optimized for frame consumption including frame timing,
    aspect ratio calculations, and consumption tracking.
    """

    def __init__(self, name: str = "prismatron_buffer"):
        """Initialize the consumer."""
        super().__init__(name)
        self._frames_consumed = 0
        self._last_frame_time = 0.0

    def connect(self) -> bool:
        """Connect to existing ring buffer for consumer use."""
        if not SHARED_MEMORY_AVAILABLE:
            logger.error("Shared memory not available (requires Python 3.8+)")
            return False

        try:
            # Connect to existing shared memory for frame buffers
            for i in range(self.buffer_count):
                shm_name = f"{self.name}_buffer_{i}"
                shm = shared_memory.SharedMemory(name=shm_name)
                self._shared_memory.append(shm)

            # Connect to existing metadata shared memory
            self._metadata_memory = shared_memory.SharedMemory(name=f"{self.name}_metadata")

            # Connect to existing control structure
            self._control_memory = shared_memory.SharedMemory(name=f"{self.name}_control")

            # Create array views from shared memory
            if not self._create_array_views_from_shared_memory():
                return False

            self._initialized = True
            logger.info(f"Connected to ring buffer '{self.name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ring buffer: {e}")
            return False

    def wait_for_ready_buffer(self, timeout: float = 1.0) -> Optional[BufferInfo]:
        """
        Wait for a buffer with new frame data (consumer process).

        This method polls for new frames by checking if the global frame counter
        has increased since our last read. Each consumer maintains its own read
        position independently.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Dictionary with buffer info or None if timeout/error
        """
        if not self._initialized:
            return None

        start_time = time.time()

        try:
            while (time.time() - start_time) < timeout:
                with self._lock:  # Quick atomic check
                    current_frame = self._get_frame_counter()

                    # Check if there's a new frame available at our next expected buffer
                    if current_frame > self._local_read_index:
                        # Calculate which frame we should read next
                        next_frame = self._local_read_index + 1

                        # Check if this frame exists (hasn't been overwritten)
                        if next_frame <= current_frame and (current_frame - next_frame) < self.buffer_count:
                            # The frame is available - calculate which buffer it's in
                            # For frame N, buffer index is (N-1) % buffer_count
                            read_idx = (next_frame - 1) % self.buffer_count
                            timestamp = self._get_buffer_timestamp(read_idx)

                            # Update the shared read index to indicate which buffer we're reading
                            self._set_read_index(read_idx)

                            # Update the minimum consumed frame counter to this frame
                            self._set_min_consumed_frame(next_frame)

                            # Update our local read position
                            self._local_read_index = next_frame
                            self._local_read_buffer = (self._local_read_buffer + 1) % self.buffer_count

                            # Read metadata for this buffer
                            metadata = None
                            if self._metadata_array is not None:
                                metadata_record = self._metadata_array[read_idx]

                                # Reconstruct timing data from shared memory fields
                                timing_data = None
                                if metadata_record["frame_index"] > 0:  # Valid timing data present
                                    # Mark read time in shared memory
                                    current_read_time = time.time()
                                    metadata_record["read_from_buffer_time"] = current_read_time

                                    timing_data = FrameTimingData(
                                        frame_index=int(metadata_record["frame_index"]),
                                        plugin_timestamp=float(metadata_record["plugin_timestamp"]),
                                        producer_timestamp=float(metadata_record["producer_timestamp"]),
                                        item_duration=float(metadata_record["item_duration"]),
                                        write_to_buffer_time=(
                                            float(metadata_record["write_to_buffer_time"])
                                            if metadata_record["write_to_buffer_time"] > 0
                                            else None
                                        ),
                                        read_from_buffer_time=current_read_time,
                                    )

                                metadata = FrameMetadata(
                                    presentation_timestamp=float(metadata_record["presentation_timestamp"]),
                                    source_width=int(metadata_record["source_width"]),
                                    source_height=int(metadata_record["source_height"]),
                                    capture_timestamp=float(metadata_record["capture_timestamp"]),
                                    playlist_item_index=int(metadata_record["playlist_item_index"]),
                                    is_first_frame_of_item=bool(metadata_record["is_first_frame_of_item"]),
                                    timing_data=timing_data,
                                    # Include transition fields from shared memory
                                    transition_in_type=str(metadata_record["transition_in_type"]),
                                    transition_in_duration=float(metadata_record["transition_in_duration"]),
                                    transition_out_type=str(metadata_record["transition_out_type"]),
                                    transition_out_duration=float(metadata_record["transition_out_duration"]),
                                    item_timestamp=float(metadata_record["item_timestamp"]),
                                    item_duration=float(metadata_record["item_duration"]),
                                )

                                buffer_info = BufferInfo(
                                    buffer=memoryview(self._shared_memory[read_idx].buf),
                                    timestamp=timestamp,
                                    frame_id=next_frame,
                                    buffer_index=read_idx,  # Actual buffer index being read
                                    metadata=metadata,
                                )

                                # Removed debug logs for cleaner output
                                return buffer_info
                        else:
                            # Frame was overwritten before we could read it
                            # Skip to the oldest available frame
                            oldest_available = max(1, current_frame - self.buffer_count + 1)
                            self._local_read_index = oldest_available - 1
                            continue

                # Brief sleep to avoid busy-waiting
                time.sleep(0.001)  # 1ms polling interval

            # Timeout reached
            return None

        except Exception as e:
            logger.error(f"Failed to wait for ready buffer: {e}")
            return None

    def release_read_buffer(self) -> bool:
        """
        Signal that the consumer is done reading the current buffer.

        This allows the producer to reuse the buffer for writing.
        Optional method - the consumer can also just call wait_for_ready_buffer()
        again which will automatically release the previous buffer.

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            return False

        try:
            with self._lock:
                # Lock assertion removed - multiprocessing.Lock doesn't support blocking parameter
                # Set read index to -1 to indicate no buffer is currently being read
                # This makes all buffers available for writing (except current write buffer)
                self._set_read_index(-1)
                logger.debug("Consumer released read buffer")
                return True

        except Exception as e:
            logger.error(f"Failed to release read buffer: {e}")
            return False

    def read_frame(self, timeout: float = 1.0) -> Optional[BufferInfo]:
        """
        Read a frame from the buffer.

        Args:
            timeout: Maximum time to wait for frame data

        Returns:
            Frame info dictionary or None if no frame available
        """
        buffer_info = self.wait_for_ready_buffer(timeout=timeout)
        if buffer_info is not None:
            self._frames_consumed += 1
            self._last_frame_time = time.time()
        return buffer_info

    def get_consumer_stats(self) -> Dict[str, Any]:
        """Get consumer-specific statistics."""
        base_status = self.get_status()

        # Calculate frame rate
        current_time = time.time()
        frame_rate = 0.0
        if self._frames_consumed > 0 and self._last_frame_time > 0:
            elapsed = current_time - (self._last_frame_time - (self._frames_consumed * 0.033))  # Estimate
            if elapsed > 0:
                frame_rate = self._frames_consumed / elapsed

        base_status.update(
            {
                "frames_consumed": self._frames_consumed,
                "estimated_fps": frame_rate,
                "is_consumer": True,
                "has_new_frame": base_status.get("frames_behind", 0) > 0,
            }
        )
        return base_status

    def finish_frame(self):
        """Signal that the current frame processing is complete."""
        self.release_read_buffer()


# Control state getter functions
def get_shared_control_state(ring_buffer: FrameRingBuffer) -> Dict[str, Any]:
    """
    Get comprehensive shared control state information.

    Args:
        ring_buffer: Any FrameRingBuffer instance (producer, consumer, or base)

    Returns:
        Dictionary with detailed control state information
    """
    if not ring_buffer._initialized:
        return {"initialized": False, "error": "Buffer not initialized"}

    try:
        with ring_buffer._lock:
            # Lock assertion removed - multiprocessing.Lock doesn't support blocking parameter
            control_array = ring_buffer._control_array
            if control_array is None:
                return {"initialized": True, "error": "Control array not available"}

            return {
                "initialized": True,
                "buffer_info": {
                    "write_index": int(control_array[0]),
                    "read_index": int(control_array[1]),
                    "frame_counter": int(control_array[2]),
                    "min_consumed_frame": int(control_array[3]),
                    "buffer_count": ring_buffer.buffer_count,
                    "buffer_size": ring_buffer.buffer_size,
                },
                "timing_info": {
                    "buffer_timestamps": [control_array[4 + i] for i in range(ring_buffer.buffer_count)],
                    "last_write_time": max([control_array[4 + i] for i in range(ring_buffer.buffer_count)]),
                    "current_time": time.time(),
                },
                "flow_control": {
                    "frames_ahead": int(control_array[2]) - int(control_array[3]),
                    "buffers_full": int(control_array[2]) - int(control_array[3]) >= ring_buffer.buffer_count,
                    "producer_blocked": int(control_array[2]) - int(control_array[3]) >= ring_buffer.buffer_count * 2,
                    "consumer_waiting": int(control_array[2]) == int(control_array[3]),
                },
            }
    except Exception as e:
        return {"initialized": True, "error": f"Failed to read control state: {e}"}


def get_buffer_utilization(ring_buffer: FrameRingBuffer) -> Dict[str, Any]:
    """
    Get buffer utilization metrics.

    Args:
        ring_buffer: Any FrameRingBuffer instance

    Returns:
        Dictionary with utilization percentages and rates
    """
    control_state = get_shared_control_state(ring_buffer)
    if "error" in control_state:
        return control_state

    buffer_info = control_state["buffer_info"]
    flow_control = control_state["flow_control"]

    # Calculate utilization metrics
    frames_buffered = flow_control["frames_ahead"]
    max_buffered = buffer_info["buffer_count"]

    utilization = {
        "buffer_fill_percentage": min(100.0, (frames_buffered / max_buffered) * 100.0),
        "producer_pressure": min(100.0, (frames_buffered / (max_buffered * 2)) * 100.0),
        "consumer_lag": max(0.0, frames_buffered - 1),  # Frames behind optimal
        "is_healthy": 0 < frames_buffered < max_buffered,
        "needs_attention": frames_buffered >= max_buffered or frames_buffered == 0,
    }

    return utilization


def get_frame_timing_info(ring_buffer: FrameRingBuffer) -> Dict[str, Any]:
    """
    Get frame timing and rate information.

    Args:
        ring_buffer: Any FrameRingBuffer instance

    Returns:
        Dictionary with timing analysis
    """
    control_state = get_shared_control_state(ring_buffer)
    if "error" in control_state:
        return control_state

    timing_info = control_state["timing_info"]
    buffer_timestamps = timing_info["buffer_timestamps"]

    # Calculate frame rate from recent timestamps
    valid_timestamps = [ts for ts in buffer_timestamps if ts > 0]

    frame_timing = {
        "recent_timestamps": valid_timestamps,
        "estimated_frame_rate": 0.0,
        "frame_interval_ms": 0.0,
        "timing_consistency": 0.0,
    }

    if len(valid_timestamps) >= 2:
        # Sort timestamps to get intervals
        sorted_times = sorted(valid_timestamps)
        intervals = [sorted_times[i] - sorted_times[i - 1] for i in range(1, len(sorted_times))]

        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            frame_timing["estimated_frame_rate"] = 1.0 / avg_interval if avg_interval > 0 else 0.0
            frame_timing["frame_interval_ms"] = avg_interval * 1000.0

            # Calculate consistency (lower std dev = more consistent)
            if len(intervals) > 1:
                variance = sum((interval - avg_interval) ** 2 for interval in intervals) / len(intervals)
                std_dev = variance**0.5
                frame_timing["timing_consistency"] = max(0.0, 1.0 - (std_dev / avg_interval))

    return frame_timing
