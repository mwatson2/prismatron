"""
Video Content Source Plugin.

This module implements a content source for video files with support
for various video formats, hardware acceleration, and FFmpeg integration.
"""

import contextlib
import logging
import os
import queue
import subprocess  # nosec B404 - needed for FFmpeg integration
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import ffmpeg

    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    ffmpeg = None  # type: ignore

from .base import (
    ContentSource,
    ContentSourceRegistry,
    ContentStatus,
    ContentType,
    FrameData,
)
from .hardware_acceleration_cache import get_hardware_acceleration_cache

logger = logging.getLogger(__name__)


class VideoSource(ContentSource):
    """
    Content source for video files.

    Supports various video formats with hardware acceleration detection
    and efficient frame decoding using FFmpeg.
    """

    def __init__(self, filepath: str):
        """
        Initialize video source.

        Args:
            filepath: Path to video file
        """
        super().__init__(filepath)
        self.content_info.content_type = ContentType.VIDEO

        # FFmpeg process and streams
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._frame_reader_thread: Optional[threading.Thread] = None
        self._frame_queue: Optional[Any] = None  # Will be a queue.Queue
        self._stop_reading = threading.Event()

        # Video properties
        self._frame_width: int = 0
        self._frame_height: int = 0
        self._frame_rate: float = 0.0
        self._total_frames: int = 0
        self._frame_size_bytes: int = 0

        # Current state
        self._current_frame_number: int = 0
        self._start_time: float = 0.0
        self._hardware_acceleration: Optional[str] = None
        self._hw_decoder: Optional[str] = None  # Specific hardware decoder (e.g., h264_nvmpi)

        # Threading for frame reading
        self._lock = threading.Lock()

    def setup(self) -> bool:
        """
        Initialize the video source and prepare for playback.

        Returns:
            True if setup successful, False otherwise
        """
        setup_start_time = time.time()

        if not FFMPEG_AVAILABLE:
            self.set_error("FFmpeg not available")
            return False

        if not os.path.exists(self.filepath):
            self.set_error(f"Video file not found: {self.filepath}")
            return False

        try:
            # Probe video file to get metadata
            probe_start = time.time()
            if not self._probe_video_metadata():
                return False
            probe_duration = time.time() - probe_start

            # Detect hardware acceleration capabilities
            hw_start = time.time()
            self._detect_hardware_acceleration()
            hw_duration = time.time() - hw_start

            # Initialize frame queue
            self._frame_queue = queue.Queue(maxsize=10)  # Buffer up to 10 frames

            # Start FFmpeg process
            ffmpeg_start = time.time()
            if not self._start_ffmpeg_process():
                return False
            ffmpeg_duration = time.time() - ffmpeg_start

            # Start frame reader thread
            self._start_frame_reader_thread()

            self.status = ContentStatus.READY
            total_setup_time = time.time() - setup_start_time

            logger.info(
                f"🎬 Video source initialized: {os.path.basename(self.filepath)} in {total_setup_time*1000:.1f}ms"
            )
            logger.info(
                f"   ⏱️  Timing breakdown: probe={probe_duration*1000:.1f}ms, hw_detect={hw_duration*1000:.1f}ms, ffmpeg={ffmpeg_duration*1000:.1f}ms"
            )
            logger.info(
                f"   📐 Resolution: {self._frame_width}x{self._frame_height}, {self._frame_rate} fps, {self.content_info.duration:.2f}s"
            )
            logger.info(f"   🚀 Hardware acceleration: {self._hardware_acceleration or 'None'}")

            return True

        except Exception as e:
            self.set_error(f"Video setup failed: {e}")
            return False

    def _probe_video_metadata(self) -> bool:
        """
        Probe video file to extract metadata.

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        try:
            # Use ffprobe to get video information
            probe_start = time.time()
            probe = ffmpeg.probe(self.filepath)
            probe_time = time.time() - probe_start
            logger.debug(f"ffprobe took {probe_time*1000:.1f}ms for {self.filepath}")

            # Find video stream
            video_stream = None
            for stream in probe["streams"]:
                if stream["codec_type"] == "video":
                    video_stream = stream
                    break

            if not video_stream:
                self.set_error("No video stream found in file")
                return False

            # Extract basic properties
            self._frame_width = int(video_stream["width"])
            self._frame_height = int(video_stream["height"])

            # Parse frame rate
            frame_rate_str = video_stream.get("r_frame_rate", "0/1")
            if "/" in frame_rate_str:
                num, den = map(int, frame_rate_str.split("/"))
                self._frame_rate = num / den if den != 0 else 0.0
            else:
                self._frame_rate = float(frame_rate_str)

            # Duration
            duration = float(video_stream.get("duration", 0))
            if duration == 0:
                # Try format duration
                duration = float(probe.get("format", {}).get("duration", 0))

            self._total_frames = int(self._frame_rate * duration) if duration > 0 else 0

            # Update content info
            self.content_info.width = self._frame_width
            self.content_info.height = self._frame_height
            self.content_info.fps = self._frame_rate
            self.content_info.duration = duration
            self.content_info.frame_count = self._total_frames

            # Calculate frame size in bytes (RGB format)
            self._frame_size_bytes = self._frame_width * self._frame_height * 3

            # Store additional metadata
            self.content_info.metadata = {
                "codec_name": video_stream.get("codec_name"),
                "codec_long_name": video_stream.get("codec_long_name"),
                "bit_rate": video_stream.get("bit_rate"),
                "format_name": probe.get("format", {}).get("format_name"),
                "file_size": probe.get("format", {}).get("size"),
            }

            return True

        except Exception as e:
            self.set_error(f"Failed to probe video metadata: {e}")
            return False

    def _detect_hardware_acceleration(self) -> None:
        """
        Get hardware acceleration options from global cache.

        Uses cached results to avoid subprocess calls during video initialization.
        """
        start_time = time.time()

        try:
            # Get cached hardware acceleration info
            cache = get_hardware_acceleration_cache()
            self._hardware_acceleration = cache.get_hardware_acceleration()

            if self._hardware_acceleration:
                # Get codec-specific decoder from cache
                codec_name = self.content_info.metadata.get("codec_name", "")
                self._hw_decoder = cache.get_decoder_for_codec(codec_name)

                if self._hw_decoder:
                    logger.info(f"Using cached hardware decoder: {self._hw_decoder} for {codec_name}")
                else:
                    logger.info(
                        f"Hardware acceleration available ({self._hardware_acceleration}) but no decoder for codec: {codec_name}"
                    )
                    self._hardware_acceleration = None
            else:
                logger.debug("No hardware acceleration available (cached result)")

            detection_time = time.time() - start_time
            logger.debug(f"Hardware acceleration detection from cache: {detection_time*1000:.1f}ms")

        except Exception as e:
            logger.warning(f"Failed to get hardware acceleration from cache: {e}")
            self._hardware_acceleration = None
            self._hw_decoder = None

    def _start_ffmpeg_process(self) -> bool:
        """
        Start FFmpeg process for video decoding.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build FFmpeg command
            input_stream = ffmpeg.input(self.filepath)

            # Apply hardware acceleration if available
            if self._hardware_acceleration in ["nvmpi", "nvv4l2dec"]:
                # Use Jetson hardware decoder
                if hasattr(self, "_hw_decoder"):
                    input_stream = ffmpeg.input(self.filepath, vcodec=self._hw_decoder)
                    logger.info(f"Using Jetson hardware decoder: {self._hw_decoder}")
                else:
                    # Fallback to auto-detect based on codec
                    codec_name = self.content_info.metadata.get("codec_name", "").lower()
                    decoder_prefix = "nvv4l2dec" if self._hardware_acceleration == "nvv4l2dec" else "nvmpi"
                    if "h264" in codec_name or "avc" in codec_name:
                        input_stream = ffmpeg.input(self.filepath, vcodec=f"h264_{decoder_prefix}")
                        logger.info(f"Using h264_{decoder_prefix} decoder")
                    elif "hevc" in codec_name or "h265" in codec_name:
                        input_stream = ffmpeg.input(self.filepath, vcodec=f"hevc_{decoder_prefix}")
                        logger.info(f"Using hevc_{decoder_prefix} decoder")
                    else:
                        logger.warning(f"Unknown codec {codec_name}, falling back to software decoding")
            elif self._hardware_acceleration == "cuda":
                input_stream = ffmpeg.input(self.filepath, hwaccel="cuda")
            elif self._hardware_acceleration == "nvdec":
                input_stream = ffmpeg.input(self.filepath, hwaccel="nvdec")

            # Configure output: scale to exact dimensions and convert to RGB24
            output_stream = ffmpeg.output(
                input_stream,
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{self._frame_width}x{self._frame_height}",
                r=self._frame_rate,
            )

            # Create process
            self._ffmpeg_process = ffmpeg.run_async(output_stream, pipe_stdout=True, pipe_stderr=True, quiet=True)

            logger.debug(f"FFmpeg process started with PID: {self._ffmpeg_process.pid}")
            return True

        except Exception as e:
            self.set_error(f"Failed to start FFmpeg process: {e}")
            return False

    def _start_frame_reader_thread(self) -> None:
        """Start background thread to read frames from FFmpeg."""
        self._frame_reader_thread = threading.Thread(target=self._frame_reader_worker, daemon=True)
        self._frame_reader_thread.start()

    def _frame_reader_worker(self) -> None:
        """
        Background worker that reads frames from FFmpeg and queues them.
        """
        frame_number = 0

        try:
            while not self._stop_reading.is_set() and self._ffmpeg_process:
                # Read one frame
                frame_bytes = self._ffmpeg_process.stdout.read(self._frame_size_bytes)

                if len(frame_bytes) != self._frame_size_bytes:
                    # End of video or error
                    logger.info(
                        f"End of video stream reached for {os.path.basename(self.filepath)} (read {len(frame_bytes)} bytes, expected {self._frame_size_bytes})"
                    )
                    break

                # Convert to numpy array in interleaved format
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame_array_interleaved = frame_array.reshape((self._frame_height, self._frame_width, 3))

                # Convert to planar format (3, H, W) for system consistency
                frame_array_planar = FrameData.convert_interleaved_to_planar(frame_array_interleaved)

                # Calculate local presentation timestamp (from zero)
                presentation_timestamp = frame_number / self._frame_rate if self._frame_rate > 0 else 0

                # Create frame data with planar format and duration metadata
                frame_data = FrameData(
                    array=frame_array_planar,  # Now in planar format (3, H, W)
                    width=self._frame_width,
                    height=self._frame_height,
                    channels=3,
                    presentation_timestamp=presentation_timestamp,  # Local timestamp from zero
                    duration=self.content_info.duration,  # Total duration of this video item
                )

                # Queue frame (with timeout to avoid blocking)
                try:
                    if self._frame_queue:
                        self._frame_queue.put(frame_data, timeout=1.0)
                        frame_number += 1
                except queue.Full:
                    # Queue might be full, continue
                    pass

        except Exception as e:
            logger.error(f"Frame reader error: {e}")

        finally:
            # Signal end of stream
            if self._frame_queue:
                with contextlib.suppress(queue.Full):
                    self._frame_queue.put(None, timeout=0.1)  # None signals end

    def get_next_frame(self) -> Optional[FrameData]:
        """
        Get the next frame from the video source.

        Returns:
            FrameData object with frame information, or None if end/error
        """
        if self.status == ContentStatus.ERROR:
            return None

        if self.status == ContentStatus.ENDED:
            return None

        if not self._frame_queue:
            self.set_error("Frame queue not initialized")
            return None

        try:
            # Get frame from queue (non-blocking)
            frame_data = self._frame_queue.get_nowait()

            if frame_data is None:
                # End of stream signal
                self.status = ContentStatus.ENDED
                logger.debug(f"Video source {os.path.basename(self.filepath)} status set to ENDED")
                return None

            # Update current state
            with self._lock:
                self._current_frame_number += 1
                self.current_frame = self._current_frame_number
                self.current_time = frame_data.presentation_timestamp or 0.0

            # Mark as playing if this is the first frame
            if self.status == ContentStatus.READY:
                self.status = ContentStatus.PLAYING
                self._start_time = time.time()

            return frame_data

        except queue.Empty:
            # No frame available yet, return None
            return None

    def get_duration(self) -> float:
        """
        Get total duration of video in seconds.

        Returns:
            Duration in seconds
        """
        return self.content_info.duration

    def seek(self, timestamp: float) -> bool:
        """
        Seek to specific timestamp in video.

        Args:
            timestamp: Target timestamp in seconds

        Returns:
            True if seek successful, False otherwise
        """
        try:
            # For now, seeking requires restarting the FFmpeg process
            # This is a simplified implementation

            if timestamp < 0 or timestamp >= self.content_info.duration:
                return False

            # Stop current playback
            self._stop_current_playback()

            # Restart with seek offset
            self._start_ffmpeg_with_seek(timestamp)

            # Update current time
            with self._lock:
                self.current_time = timestamp
                self._current_frame_number = int(timestamp * self._frame_rate)

            return True

        except Exception as e:
            self.set_error(f"Seek failed: {e}")
            return False

    def _stop_current_playback(self) -> None:
        """Stop current FFmpeg process and frame reading."""
        # Signal stop
        self._stop_reading.set()

        # Wait for reader thread
        if self._frame_reader_thread and self._frame_reader_thread.is_alive():
            self._frame_reader_thread.join(timeout=2.0)

        # Terminate FFmpeg process
        if self._ffmpeg_process:
            try:
                self._ffmpeg_process.terminate()
                self._ffmpeg_process.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                # Force kill if needed
                with contextlib.suppress(ProcessLookupError, OSError):
                    self._ffmpeg_process.kill()
            self._ffmpeg_process = None

        # Clear queue
        if self._frame_queue:
            while True:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break

        # Reset stop flag
        self._stop_reading.clear()

    def _start_ffmpeg_with_seek(self, start_time: float) -> bool:
        """
        Start FFmpeg process with seek to specific time.

        Args:
            start_time: Start time in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build FFmpeg command with seek
            input_stream = ffmpeg.input(self.filepath, ss=start_time)

            # Apply hardware acceleration if available
            if self._hardware_acceleration in ["nvmpi", "nvv4l2dec"]:
                # Use Jetson hardware decoder
                if hasattr(self, "_hw_decoder"):
                    input_stream = ffmpeg.input(self.filepath, ss=start_time, vcodec=self._hw_decoder)
                    logger.info(f"Using Jetson hardware decoder with seek: {self._hw_decoder}")
                else:
                    # Fallback to auto-detect based on codec
                    codec_name = self.content_info.metadata.get("codec_name", "").lower()
                    decoder_prefix = "nvv4l2dec" if self._hardware_acceleration == "nvv4l2dec" else "nvmpi"
                    if "h264" in codec_name or "avc" in codec_name:
                        input_stream = ffmpeg.input(self.filepath, ss=start_time, vcodec=f"h264_{decoder_prefix}")
                    elif "hevc" in codec_name or "h265" in codec_name:
                        input_stream = ffmpeg.input(self.filepath, ss=start_time, vcodec=f"hevc_{decoder_prefix}")
                    else:
                        logger.warning(f"Unknown codec {codec_name} for seek, falling back to software decoding")
            elif self._hardware_acceleration == "cuda":
                input_stream = ffmpeg.input(self.filepath, ss=start_time, hwaccel="cuda")
            elif self._hardware_acceleration == "nvdec":
                input_stream = ffmpeg.input(self.filepath, ss=start_time, hwaccel="nvdec")

            # Configure output
            output_stream = ffmpeg.output(
                input_stream,
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{self._frame_width}x{self._frame_height}",
                r=self._frame_rate,
            )

            # Create process
            self._ffmpeg_process = ffmpeg.run_async(output_stream, pipe_stdout=True, pipe_stderr=True, quiet=True)

            # Restart frame reader
            self._start_frame_reader_thread()

            return True

        except Exception as e:
            self.set_error(f"Failed to start FFmpeg with seek: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up resources and close video source."""
        try:
            # Stop playback
            self._stop_current_playback()

            # Reset state
            self._current_frame_number = 0
            self.current_time = 0.0
            self.status = ContentStatus.UNINITIALIZED

            logger.debug(f"Video source cleaned up: {self.filepath}")

        except Exception as e:
            logger.error(f"Error during video source cleanup: {e}")

    def get_video_info(self) -> Dict[str, Any]:
        """
        Get detailed video information.

        Returns:
            Dictionary with video properties
        """
        return {
            "width": self._frame_width,
            "height": self._frame_height,
            "fps": self._frame_rate,
            "duration": self.content_info.duration,
            "total_frames": self._total_frames,
            "current_frame": self._current_frame_number,
            "hardware_acceleration": self._hardware_acceleration,
            "codec": self.content_info.metadata.get("codec_name"),
            "format": self.content_info.metadata.get("format_name"),
            "bitrate": self.content_info.metadata.get("bit_rate"),
        }


# Register the video source plugin
ContentSourceRegistry.register(ContentType.VIDEO, VideoSource)
