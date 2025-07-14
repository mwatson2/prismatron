"""
Preview Sink for LED Display System.

This module implements a sink that receives optimized LED values from the frame renderer
and makes them available to the web server process via shared memory IPC. The LED values
are converted from spatial order to physical order for display preview purposes.
"""

import logging
import mmap
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..const import LED_COUNT, LED_DATA_SIZE
from ..utils.spatial_ordering import reorder_block_values

logger = logging.getLogger(__name__)


@dataclass
class PreviewSinkConfig:
    """Configuration for the preview sink."""

    shared_memory_name: str = "prismatron_preview"
    update_ewma_alpha: float = 0.1  # EWMA alpha for statistics
    max_stats_history: int = 1000  # Maximum number of frames to track for statistics
    late_frame_threshold_ms: float = 16.67  # Threshold for considering frame late (60fps = 16.67ms)


class PreviewSinkStatistics:
    """Statistics tracking for preview sink using EWMA."""

    def __init__(self, alpha: float = 0.1):
        """
        Initialize statistics tracker.

        Args:
            alpha: EWMA alpha parameter (0 < alpha <= 1, smaller = more smoothing)
        """
        self.alpha = alpha

        # EWMA values
        self.ewma_fps = 0.0
        self.ewma_late_fraction = 0.0
        self.ewma_dropped_fraction = 0.0

        # Raw counters
        self.frames_processed = 0
        self.frames_late = 0
        self.frames_dropped = 0

        # Timing tracking
        self.last_frame_time = 0.0
        self.frame_intervals = []
        self.max_interval_history = 100

        # Thread safety
        self._lock = threading.Lock()

    def update_frame_timing(self, is_late: bool = False, is_dropped: bool = False) -> None:
        """
        Update frame timing statistics.

        Args:
            is_late: Whether this frame was late
            is_dropped: Whether this frame was dropped
        """
        current_time = time.time()

        with self._lock:
            self.frames_processed += 1

            if is_late:
                self.frames_late += 1
            if is_dropped:
                self.frames_dropped += 1

            # Calculate frame interval for FPS
            if self.last_frame_time > 0:
                interval = current_time - self.last_frame_time
                self.frame_intervals.append(interval)

                # Keep only recent intervals
                if len(self.frame_intervals) > self.max_interval_history:
                    self.frame_intervals = self.frame_intervals[-self.max_interval_history :]

                # Update EWMA FPS
                instant_fps = 1.0 / interval if interval > 0 else 0.0
                if self.ewma_fps == 0.0:
                    self.ewma_fps = instant_fps
                else:
                    self.ewma_fps = (1 - self.alpha) * self.ewma_fps + self.alpha * instant_fps

            self.last_frame_time = current_time

            # Update EWMA fractions
            late_fraction = self.frames_late / max(1, self.frames_processed)
            dropped_fraction = self.frames_dropped / max(1, self.frames_processed)

            if self.frames_processed == 1:
                # First frame, initialize EWMA
                self.ewma_late_fraction = late_fraction
                self.ewma_dropped_fraction = dropped_fraction
            else:
                # Update EWMA
                self.ewma_late_fraction = (1 - self.alpha) * self.ewma_late_fraction + self.alpha * late_fraction
                self.ewma_dropped_fraction = (
                    1 - self.alpha
                ) * self.ewma_dropped_fraction + self.alpha * dropped_fraction

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self._lock:
            avg_fps = 0.0
            if self.frame_intervals:
                avg_interval = np.mean(self.frame_intervals)
                avg_fps = 1.0 / avg_interval if avg_interval > 0 else 0.0

            return {
                "frames_processed": self.frames_processed,
                "frames_late": self.frames_late,
                "frames_dropped": self.frames_dropped,
                "ewma_fps": self.ewma_fps,
                "ewma_late_fraction": self.ewma_late_fraction,
                "ewma_dropped_fraction": self.ewma_dropped_fraction,
                "average_fps": avg_fps,
                "late_percentage": (self.frames_late / max(1, self.frames_processed)) * 100,
                "dropped_percentage": (self.frames_dropped / max(1, self.frames_processed)) * 100,
                "ewma_alpha": self.alpha,
            }

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.ewma_fps = 0.0
            self.ewma_late_fraction = 0.0
            self.ewma_dropped_fraction = 0.0
            self.frames_processed = 0
            self.frames_late = 0
            self.frames_dropped = 0
            self.last_frame_time = 0.0
            self.frame_intervals.clear()


class PreviewSink:
    """
    Preview sink that receives LED values and shares them with web server via shared memory.

    This sink converts LED values from spatial order (used by optimization) to physical order
    (needed for display preview) and stores them in shared memory for the web server to read.
    """

    def __init__(self, config: Optional[PreviewSinkConfig] = None, spatial_mapping: Optional[Dict[int, int]] = None):
        """
        Initialize preview sink.

        Args:
            config: Preview sink configuration
            spatial_mapping: Mapping from physical LED ID to spatial order index
        """
        self.config = config or PreviewSinkConfig()
        self.spatial_mapping = spatial_mapping or {}

        # Create reverse mapping if spatial mapping provided
        self.reverse_spatial_mapping = {}
        if self.spatial_mapping:
            self.reverse_spatial_mapping = {v: k for k, v in self.spatial_mapping.items()}

        # Statistics tracking
        self.stats = PreviewSinkStatistics(alpha=self.config.update_ewma_alpha)

        # Shared memory setup
        self.shared_memory_size = self._calculate_shared_memory_size()
        self.shared_memory_fd = None
        self.shared_memory_map = None

        # Thread safety
        self._lock = threading.Lock()

        # State
        self.is_running = False
        self.last_update_time = 0.0

        logger.info(f"Preview sink initialized with shared memory size: {self.shared_memory_size} bytes")

    def _calculate_shared_memory_size(self) -> int:
        """Calculate required shared memory size."""
        # Memory layout:
        # - Header (64 bytes): timestamp, frame_counter, led_count, padding
        # - LED data (LED_COUNT * 3 bytes): RGB values in physical order
        # - Statistics (128 bytes): JSON-encoded statistics
        header_size = 64
        led_data_size = LED_DATA_SIZE  # LED_COUNT * 3
        stats_size = 128

        total_size = header_size + led_data_size + stats_size

        # Round up to page boundary for efficiency
        page_size = 4096
        return ((total_size + page_size - 1) // page_size) * page_size

    def start(self) -> bool:
        """
        Start the preview sink by creating shared memory.

        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.is_running:
                logger.warning("Preview sink already running")
                return True

            # Create shared memory
            if not self._create_shared_memory():
                return False

            self.is_running = True
            logger.info("Preview sink started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start preview sink: {e}")
            return False

    def stop(self) -> None:
        """Stop the preview sink and cleanup shared memory."""
        try:
            if not self.is_running:
                return

            logger.info("Stopping preview sink...")
            self.is_running = False

            # Cleanup shared memory
            self._cleanup_shared_memory()

            logger.info("Preview sink stopped")

        except Exception as e:
            logger.error(f"Error stopping preview sink: {e}")

    def _create_shared_memory(self) -> bool:
        """Create and initialize shared memory."""
        try:
            import os
            import tempfile

            # Create a temporary file for shared memory
            temp_fd, temp_path = tempfile.mkstemp(prefix=f"{self.config.shared_memory_name}_")

            # Resize file to required size
            os.ftruncate(temp_fd, self.shared_memory_size)

            # Create memory map
            self.shared_memory_map = mmap.mmap(temp_fd, self.shared_memory_size)
            self.shared_memory_fd = temp_fd

            # Initialize header
            self._write_header(timestamp=time.time(), frame_counter=0, led_count=LED_COUNT)

            # Zero out LED data section
            led_data_offset = 64
            self.shared_memory_map[led_data_offset : led_data_offset + LED_DATA_SIZE] = b"\x00" * LED_DATA_SIZE

            # Initialize statistics
            self._write_statistics()

            logger.info(f"Created shared memory: {self.shared_memory_size} bytes")
            return True

        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            return False

    def _cleanup_shared_memory(self) -> None:
        """Cleanup shared memory resources."""
        try:
            if self.shared_memory_map:
                self.shared_memory_map.close()
                self.shared_memory_map = None

            if self.shared_memory_fd:
                import os

                os.close(self.shared_memory_fd)
                self.shared_memory_fd = None

        except Exception as e:
            logger.warning(f"Error cleaning up shared memory: {e}")

    def _write_header(self, timestamp: float, frame_counter: int, led_count: int) -> None:
        """Write header to shared memory."""
        if not self.shared_memory_map:
            return

        # Header format: timestamp(8) + frame_counter(8) + led_count(4) + padding(44)
        header_data = struct.pack("<ddI44x", timestamp, frame_counter, led_count)
        self.shared_memory_map[0:64] = header_data

    def _write_statistics(self) -> None:
        """Write statistics to shared memory."""
        if not self.shared_memory_map:
            return

        try:
            import json

            stats = self.stats.get_statistics()
            stats_json = json.dumps(stats).encode("utf-8")

            # Truncate if too long
            if len(stats_json) > 127:  # Leave 1 byte for null terminator
                stats_json = stats_json[:127]

            # Write to shared memory (last 128 bytes)
            stats_offset = self.shared_memory_size - 128
            self.shared_memory_map[stats_offset : stats_offset + len(stats_json)] = stats_json
            self.shared_memory_map[stats_offset + len(stats_json)] = 0  # Null terminator

        except Exception as e:
            logger.warning(f"Failed to write statistics to shared memory: {e}")

    def render_led_values(self, led_values: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Receive LED values from frame renderer and update shared memory.

        Args:
            led_values: LED RGB values, shape (led_count, 3) in spatial order
            metadata: Optional frame metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.is_running or not self.shared_memory_map:
            return False

        try:
            start_time = time.time()

            # Validate input
            if led_values.ndim != 2 or led_values.shape[1] != 3:
                logger.error(f"Invalid LED values shape: {led_values.shape}, expected (led_count, 3)")
                return False

            # Check if frame is late (simple heuristic based on time since last update)
            is_late = False
            if self.last_update_time > 0:
                time_since_last = (start_time - self.last_update_time) * 1000  # ms
                is_late = time_since_last > self.config.late_frame_threshold_ms

            # Convert from spatial order to physical order
            physical_led_values = self._convert_to_physical_order(led_values)

            # Update shared memory with new LED data
            with self._lock:
                # Update header
                frame_counter = getattr(self, "_frame_counter", 0) + 1
                self._frame_counter = frame_counter
                self._write_header(start_time, frame_counter, LED_COUNT)

                # Write LED data (convert to uint8 and flatten)
                led_data_uint8 = np.clip(physical_led_values, 0, 255).astype(np.uint8)
                led_data_bytes = led_data_uint8.flatten().tobytes()

                led_data_offset = 64
                self.shared_memory_map[led_data_offset : led_data_offset + len(led_data_bytes)] = led_data_bytes

                # Update statistics
                self.stats.update_frame_timing(is_late=is_late, is_dropped=False)
                self._write_statistics()

                self.last_update_time = start_time

            logger.debug(f"Updated preview data: frame {frame_counter}, {len(led_data_bytes)} bytes")
            return True

        except Exception as e:
            logger.error(f"Error updating preview data: {e}")
            # Update statistics with error
            self.stats.update_frame_timing(is_late=False, is_dropped=True)
            return False

    def _convert_to_physical_order(self, spatial_led_values: np.ndarray) -> np.ndarray:
        """
        Convert LED values from spatial order to physical order.

        Args:
            spatial_led_values: LED values in spatial order, shape (led_count, 3)

        Returns:
            LED values in physical order, shape (led_count, 3)
        """
        if not self.reverse_spatial_mapping:
            # No spatial mapping available, assume already in physical order
            logger.debug("No spatial mapping available, assuming physical order")
            return spatial_led_values

        led_count = spatial_led_values.shape[0]
        physical_led_values = np.zeros_like(spatial_led_values)

        # Convert each spatial index to physical index
        for spatial_idx in range(led_count):
            physical_idx = self.reverse_spatial_mapping.get(spatial_idx, spatial_idx)
            # Ensure physical_idx is within bounds
            if 0 <= physical_idx < led_count:
                physical_led_values[physical_idx] = spatial_led_values[spatial_idx]

        return physical_led_values

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get preview sink statistics.

        Returns:
            Dictionary with statistics and configuration
        """
        stats = self.stats.get_statistics()
        stats.update(
            {
                "is_running": self.is_running,
                "shared_memory_size": self.shared_memory_size,
                "led_count": LED_COUNT,
                "config": {
                    "shared_memory_name": self.config.shared_memory_name,
                    "update_ewma_alpha": self.config.update_ewma_alpha,
                    "late_frame_threshold_ms": self.config.late_frame_threshold_ms,
                },
                "has_spatial_mapping": bool(self.spatial_mapping),
                "spatial_mapping_size": len(self.spatial_mapping),
            }
        )
        return stats

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self.stats.reset()
        logger.info("Preview sink statistics reset")

    def get_shared_memory_info(self) -> Dict[str, Any]:
        """
        Get information about shared memory for web server access.

        Returns:
            Dictionary with shared memory access information
        """
        return {
            "shared_memory_name": self.config.shared_memory_name,
            "shared_memory_size": self.shared_memory_size,
            "led_data_offset": 64,
            "led_data_size": LED_DATA_SIZE,
            "stats_offset": self.shared_memory_size - 128,
            "stats_size": 128,
            "header_format": "<ddI44x",  # timestamp, frame_counter, led_count, padding
            "led_count": LED_COUNT,
        }

    def set_spatial_mapping(self, spatial_mapping: Dict[int, int]) -> None:
        """
        Update the spatial mapping.

        Args:
            spatial_mapping: Mapping from physical LED ID to spatial order index
        """
        self.spatial_mapping = spatial_mapping.copy()
        self.reverse_spatial_mapping = {v: k for k, v in self.spatial_mapping.items()}
        logger.info(f"Updated spatial mapping: {len(self.spatial_mapping)} LEDs")

    def __enter__(self):
        """Context manager entry."""
        if self.start():
            return self
        else:
            raise RuntimeError("Failed to start preview sink")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_preview_sink_from_pattern(
    pattern_file_path: str, config: Optional[PreviewSinkConfig] = None
) -> Optional[PreviewSink]:
    """
    Create a preview sink from a diffusion pattern file.

    Args:
        pattern_file_path: Path to the .npz pattern file
        config: Preview sink configuration (uses defaults if None)

    Returns:
        PreviewSink instance or None if failed to load
    """
    try:
        # Load pattern data
        data = np.load(pattern_file_path, allow_pickle=True)

        # Extract spatial mapping if available
        spatial_mapping = None
        if "led_spatial_mapping" in data:
            spatial_mapping = data["led_spatial_mapping"].item()
            logger.info(f"Loaded spatial mapping with {len(spatial_mapping)} LEDs from pattern")
        else:
            logger.warning("No spatial mapping found in pattern file")

        # Create preview sink
        sink = PreviewSink(config, spatial_mapping)

        logger.info(f"Created preview sink from pattern: {pattern_file_path}")
        return sink

    except Exception as e:
        logger.error(f"Failed to create preview sink from pattern: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    import argparse
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Test preview sink")
    parser.add_argument("--pattern-file", help="Path to diffusion pattern file")
    parser.add_argument("--test-duration", type=float, default=10.0, help="Test duration in seconds")

    args = parser.parse_args()

    # Create preview sink
    if args.pattern_file:
        sink = create_preview_sink_from_pattern(args.pattern_file)
    else:
        sink = PreviewSink()

    if not sink:
        logger.error("Failed to create preview sink")
        sys.exit(1)

    try:
        # Start sink
        if not sink.start():
            logger.error("Failed to start preview sink")
            sys.exit(1)

        logger.info(f"Preview sink started. Testing for {args.test_duration} seconds...")

        # Test with some dummy LED data
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < args.test_duration:
            # Generate test LED data (rainbow pattern)
            led_data = np.zeros((LED_COUNT, 3), dtype=np.float32)
            for i in range(LED_COUNT):
                hue = (i / LED_COUNT + (time.time() - start_time) * 0.1) % 1.0
                # Simple HSV to RGB conversion
                led_data[i] = [
                    (
                        255 * max(0, min(1, abs(hue * 6 - 3) - 1))
                        if 0 <= hue < 0.5
                        else 255 * max(0, min(1, 3 - abs(hue * 6 - 3)))
                    ),
                    255 * max(0, min(1, 2 - abs(hue * 6 - 2))),
                    255 * max(0, min(1, 2 - abs(hue * 6 - 4))),
                ]

            # Send to sink
            sink.render_led_values(led_data)
            frame_count += 1

            # Log statistics periodically
            if frame_count % 60 == 0:
                stats = sink.get_statistics()
                logger.info(
                    f"Frame {frame_count}: FPS={stats['ewma_fps']:.1f}, Late={stats['ewma_late_fraction']*100:.1f}%"
                )

            time.sleep(1 / 60)  # ~60 FPS

        # Final statistics
        stats = sink.get_statistics()
        logger.info(f"Test completed: {frame_count} frames, {stats['ewma_fps']:.1f} FPS average")
        logger.info(
            f"Late frames: {stats['ewma_late_fraction']*100:.1f}%, Dropped: {stats['ewma_dropped_fraction']*100:.1f}%"
        )

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        sink.stop()
        logger.info("Preview sink test completed")
