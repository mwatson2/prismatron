"""
Video Content Source Plugin.

This module implements a content source for video files with support
for various video formats, hardware acceleration, and FFmpeg integration.
"""

import contextlib
import json
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
        self._stderr_monitor_thread: Optional[threading.Thread] = None

    def setup(self) -> bool:
        """
        Initialize the video source and prepare for playback.

        Returns:
            True if setup successful, False otherwise
        """
        logger.info(f"VideoSource.setup() START for {os.path.basename(self.filepath)}")
        setup_start_time = time.time()

        if not FFMPEG_AVAILABLE:
            logger.error("FFmpeg not available - cannot setup video source")
            self.set_error("FFmpeg not available")
            return False

        logger.debug(f"Checking if file exists: {self.filepath}")
        if not os.path.exists(self.filepath):
            logger.error(f"Video file not found: {self.filepath}")
            self.set_error(f"Video file not found: {self.filepath}")
            return False
        logger.debug(f"File exists: {self.filepath}")

        try:
            # Probe video file to get metadata
            logger.debug(f"About to probe video metadata for {os.path.basename(self.filepath)}")
            probe_start = time.time()
            probe_result = self._probe_video_metadata()
            probe_duration = time.time() - probe_start
            logger.debug(f"Probe completed in {probe_duration*1000:.1f}ms, result={probe_result}")

            if not probe_result:
                logger.error(f"Failed to probe video metadata for {os.path.basename(self.filepath)}")
                return False

            # Detect hardware acceleration capabilities
            logger.debug("About to detect hardware acceleration")
            hw_start = time.time()
            self._detect_hardware_acceleration()
            hw_duration = time.time() - hw_start
            logger.debug(f"Hardware acceleration detection completed in {hw_duration*1000:.1f}ms")

            # Initialize frame queue
            logger.debug("Initializing frame queue")
            self._frame_queue = queue.Queue(maxsize=10)  # Buffer up to 10 frames

            # Start FFmpeg process
            logger.debug("About to start FFmpeg process")
            ffmpeg_start = time.time()
            ffmpeg_result = self._start_ffmpeg_process()
            ffmpeg_duration = time.time() - ffmpeg_start
            logger.debug(f"FFmpeg process start completed in {ffmpeg_duration*1000:.1f}ms, result={ffmpeg_result}")

            if not ffmpeg_result:
                logger.error(f"Failed to start FFmpeg process for {os.path.basename(self.filepath)}")
                return False

            # Start frame reader thread
            logger.debug("About to start frame reader thread")
            self._start_frame_reader_thread()
            logger.debug("Frame reader thread started")

            self.status = ContentStatus.READY
            total_setup_time = time.time() - setup_start_time
            logger.info(f"VideoSource.setup() COMPLETED in {total_setup_time*1000:.1f}ms")

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
            logger.error(f"Video setup failed with exception: {e}", exc_info=True)
            self.set_error(f"Video setup failed: {e}")
            return False

    def _probe_video_metadata(self) -> bool:
        """
        Probe video file to extract metadata.

        Returns:
            True if successful, False otherwise
        """
        logger.debug(f"_probe_video_metadata() START for {os.path.basename(self.filepath)}")
        start_time = time.time()
        try:
            # Use ffprobe to get video information
            logger.debug(f"About to call ffmpeg.probe() on {self.filepath}")
            probe_start = time.time()
            probe = ffmpeg.probe(self.filepath)
            probe_time = time.time() - probe_start
            logger.debug(f"ffmpeg.probe() completed in {probe_time*1000:.1f}ms for {self.filepath}")

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
                "pix_fmt": video_stream.get("pix_fmt"),
                "bits_per_raw_sample": video_stream.get("bits_per_raw_sample"),
            }

            logger.debug(f"_probe_video_metadata() COMPLETED successfully for {os.path.basename(self.filepath)}")
            return True

        except Exception as e:
            logger.error(f"Failed to probe video metadata: {e}", exc_info=True)
            self.set_error(f"Failed to probe video metadata: {e}")
            return False

    def _detect_hardware_acceleration(self) -> None:
        """
        Get hardware acceleration options from global cache.

        Uses cached results to avoid subprocess calls during video initialization.
        Disables hardware acceleration for unsupported formats (e.g., 10-bit video).
        """
        start_time = time.time()

        try:
            # Check if video format is compatible with hardware acceleration
            pix_fmt = self.content_info.metadata.get("pix_fmt", "")
            bits_per_sample = self.content_info.metadata.get("bits_per_raw_sample")
            codec_name = self.content_info.metadata.get("codec_name", "")

            # Check for 10-bit or unsupported formats
            is_10bit = (
                bits_per_sample == "10"
                or "10le" in pix_fmt
                or "p10" in pix_fmt
                or "422" in pix_fmt  # 4:2:2 chroma subsampling often indicates professional/10-bit content
            )

            if is_10bit:
                logger.warning(
                    f"10-bit video detected ({pix_fmt}, {bits_per_sample}-bit) - hardware decoders may not support this format"
                )
                logger.info(f"Forcing software decoding for {os.path.basename(self.filepath)}")
                self._hardware_acceleration = None
                self._hw_decoder = None
                return

            # Get cached hardware acceleration info
            cache = get_hardware_acceleration_cache()
            self._hardware_acceleration = cache.get_hardware_acceleration()

            if self._hardware_acceleration:
                # Get codec-specific decoder from cache
                self._hw_decoder = cache.get_decoder_for_codec(codec_name)

                if self._hw_decoder:
                    logger.info(f"Using cached hardware decoder: {self._hw_decoder} for {codec_name} ({pix_fmt})")
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
        logger.debug(f"_start_ffmpeg_process() START for {os.path.basename(self.filepath)}")
        try:
            # Build FFmpeg command
            logger.debug("Building FFmpeg input stream")
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
            logger.debug(
                f"Configuring FFmpeg output stream: {self._frame_width}x{self._frame_height} @ {self._frame_rate} fps"
            )
            output_stream = ffmpeg.output(
                input_stream,
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{self._frame_width}x{self._frame_height}",
                r=self._frame_rate,
            )

            # Create process with error capture
            logger.info(f"Starting FFmpeg process for {os.path.basename(self.filepath)} (this may take a moment)...")
            self._ffmpeg_process = ffmpeg.run_async(output_stream, pipe_stdout=True, pipe_stderr=True, quiet=False)
            logger.info("FFmpeg process started successfully")

            logger.info(f"FFmpeg process started with PID: {self._ffmpeg_process.pid}")

            # Start stderr monitoring thread to capture FFmpeg errors
            logger.debug("Starting stderr monitor thread")
            self._start_stderr_monitor()
            logger.debug("stderr monitor thread started")

            logger.debug(f"_start_ffmpeg_process() COMPLETED for {os.path.basename(self.filepath)}")
            return True

        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}", exc_info=True)
            self.set_error(f"Failed to start FFmpeg process: {e}")
            return False

    def _start_frame_reader_thread(self) -> None:
        """Start background thread to read frames from FFmpeg."""
        self._frame_reader_thread = threading.Thread(target=self._frame_reader_worker, daemon=True)
        self._frame_reader_thread.start()

    def _start_stderr_monitor(self) -> None:
        """Start background thread to monitor FFmpeg stderr for errors."""
        self._stderr_monitor_thread = threading.Thread(target=self._stderr_monitor_worker, daemon=True)
        self._stderr_monitor_thread.start()

    def _stderr_monitor_worker(self) -> None:
        """Background worker that monitors FFmpeg stderr for errors."""
        if not self._ffmpeg_process or not self._ffmpeg_process.stderr:
            return

        try:
            while not self._stop_reading.is_set() and self._ffmpeg_process:
                stderr_line = self._ffmpeg_process.stderr.readline()
                if not stderr_line:
                    break

                stderr_text = stderr_line.decode("utf-8", errors="ignore").strip()
                if stderr_text:
                    # Log FFmpeg errors and warnings
                    if "error" in stderr_text.lower() or "failed" in stderr_text.lower():
                        logger.error(f"FFmpeg error for {os.path.basename(self.filepath)}: {stderr_text}")
                    elif "warning" in stderr_text.lower():
                        logger.warning(f"FFmpeg warning for {os.path.basename(self.filepath)}: {stderr_text}")
                    else:
                        logger.debug(f"FFmpeg info for {os.path.basename(self.filepath)}: {stderr_text}")

        except Exception as e:
            logger.debug(f"FFmpeg stderr monitor error: {e}")

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
            logger.error(f"Frame reader error for {os.path.basename(self.filepath)}: {e}", exc_info=True)

        finally:
            # Signal end of stream
            logger.info(
                f"Frame reader worker finished for {os.path.basename(self.filepath)} - "
                f"read {frame_number} frames total"
            )
            # Try hard to deliver the end-of-stream signal (None)
            # This is critical - without it, the producer will never know the video ended
            if self._frame_queue:
                for attempt in range(10):  # Try for up to 5 seconds
                    try:
                        self._frame_queue.put(None, timeout=0.5)
                        logger.debug(f"End-of-stream signal (None) successfully queued after {attempt + 1} attempt(s)")
                        break
                    except queue.Full:
                        if attempt == 9:
                            logger.error(
                                f"CRITICAL: Failed to queue end-of-stream signal after 10 attempts for {os.path.basename(self.filepath)} - "
                                f"producer may not detect video end!"
                            )
                        else:
                            logger.debug(f"Queue full on attempt {attempt + 1}, retrying...")
                        continue

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
                logger.info(
                    f"Video source {os.path.basename(self.filepath)} has ENDED (received end-of-stream signal from frame reader)"
                )
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
            # No frame available yet - frame reader may be slow or blocked
            # Check if processes are dead (fallback end-of-stream detection)
            ffmpeg_exited = False
            reader_dead = False

            if self._ffmpeg_process:
                poll_result = self._ffmpeg_process.poll()
                ffmpeg_exited = poll_result is not None

            if self._frame_reader_thread:
                reader_dead = not self._frame_reader_thread.is_alive()

            # If FFmpeg exited AND reader thread is dead AND queue is empty,
            # the video has ended (even if we never got the None sentinel)
            if ffmpeg_exited and reader_dead:
                self.status = ContentStatus.ENDED
                logger.info(
                    f"Video source {os.path.basename(self.filepath)} has ENDED "
                    f"(detected via process exit: FFmpeg returncode={poll_result}, reader thread dead, queue empty)"
                )
                return None

            # Log occasionally to help debug stuck scenarios (rate limit to avoid spam)
            if not hasattr(self, "_last_empty_queue_log_time"):
                self._last_empty_queue_log_time = 0.0

            current_time = time.time()
            if current_time - self._last_empty_queue_log_time >= 2.0:  # Log every 2 seconds
                # Gather diagnostic information
                ffmpeg_running = "unknown"
                ffmpeg_returncode = "N/A"
                reader_alive = "unknown"
                stderr_alive = "unknown"

                if self._ffmpeg_process:
                    poll_result = self._ffmpeg_process.poll()
                    if poll_result is None:
                        ffmpeg_running = "yes"
                    else:
                        ffmpeg_running = "NO"
                        ffmpeg_returncode = poll_result

                if self._frame_reader_thread:
                    reader_alive = "yes" if self._frame_reader_thread.is_alive() else "NO"

                if self._stderr_monitor_thread:
                    stderr_alive = "yes" if self._stderr_monitor_thread.is_alive() else "NO"

                logger.info(
                    f"Frame queue empty for {os.path.basename(self.filepath)} - "
                    f"frame {self._current_frame_number}, FFmpeg running: {ffmpeg_running} (returncode: {ffmpeg_returncode}), "
                    f"frame_reader alive: {reader_alive}, stderr_monitor alive: {stderr_alive}"
                )
                self._last_empty_queue_log_time = current_time
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

        # Wait for stderr monitor thread
        if self._stderr_monitor_thread and self._stderr_monitor_thread.is_alive():
            self._stderr_monitor_thread.join(timeout=1.0)

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

            # Create process with error capture
            self._ffmpeg_process = ffmpeg.run_async(output_stream, pipe_stdout=True, pipe_stderr=True, quiet=False)

            # Start monitors
            self._start_stderr_monitor()
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
