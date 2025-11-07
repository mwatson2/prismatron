#!/usr/bin/env python3
"""
Audio Capture Utility for Prismatron LED Display System

Standalone utility for capturing raw audio from microphone devices.
Provides a clean interface for testing and debugging audio input.
"""

import logging
import queue
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio capture"""

    sample_rate: int = 44100  # Hz
    channels: int = 1  # Mono
    dtype: str = "float32"  # Audio data type
    chunk_size: int = 1024  # Samples per chunk (~23ms at 44.1kHz)
    device_name: Optional[str] = "USB Audio"  # Device name to search for
    file_path: Optional[str] = None  # Path to audio file (test mode)
    playback_speed: float = 1.0  # Playback speed multiplier for file mode (1.0 = realtime)


class AudioCapture:
    """
    Standalone audio capture utility.

    Captures raw audio from microphone devices with configurable parameters.
    Designed for testing, debugging, and as a building block for audio processing.
    """

    def __init__(self, config: Optional[AudioConfig] = None, audio_callback: Optional[Callable] = None):
        """
        Initialize audio capture.

        Args:
            config: Audio configuration (uses defaults if None)
            audio_callback: Optional callback for audio chunks (chunk, timestamp)
        """
        self.config = config or AudioConfig()
        self.audio_callback = audio_callback

        # Determine capture mode
        self.file_mode = self.config.file_path is not None

        # Device selection (only for live mode)
        self.device_index: Optional[int] = None
        if not self.file_mode:
            self._find_audio_device()

        # Capture state
        self.is_capturing = False
        self.capture_thread: Optional[threading.Thread] = None

        # Only create queue if no callback provided (queue-based consumption)
        # When using callbacks, the queue is unnecessary and wastes memory
        if audio_callback is None:
            self.audio_queue = queue.Queue(maxsize=5000)
        else:
            self.audio_queue = None  # Callback mode - no queue needed

        # Thread pool for non-blocking callback execution
        self.callback_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AudioCallback")

        # Statistics
        self.total_samples_captured = 0
        self.total_chunks_captured = 0
        self.overflow_count = 0
        self.start_time = 0.0
        self.dropped_callbacks = 0

        # File mode info
        if self.file_mode:
            logger.info(
                f"AudioCapture initialized (FILE MODE): {self.config.sample_rate}Hz, "
                f"{self.config.channels}ch, file={self.config.file_path}, "
                f"speed={self.config.playback_speed}x"
            )
        else:
            logger.info(
                f"AudioCapture initialized (LIVE MODE): {self.config.sample_rate}Hz, "
                f"{self.config.channels}ch, device={self.device_index}"
            )

    def _find_audio_device(self) -> bool:
        """
        Find audio device matching the configured device name.

        Returns:
            True if device found, False otherwise
        """
        if not self.config.device_name:
            logger.info("No device name specified, using system default")
            return True

        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if self.config.device_name in device["name"] and device["max_input_channels"] > 0:
                    self.device_index = i
                    logger.info(f"Found audio device at index {i}: {device['name']}")
                    logger.info(
                        f"  Input channels: {device['max_input_channels']}, "
                        f"Sample rate: {device['default_samplerate']}Hz"
                    )
                    return True

            logger.warning(f"Device '{self.config.device_name}' not found, will use default")
            return False
        except Exception as e:
            logger.error(f"Error finding audio device: {e}")
            return False

    def list_devices(self) -> list:
        """
        List all available audio devices.

        Returns:
            List of device info dictionaries
        """
        try:
            devices = sd.query_devices()
            return [
                {
                    "index": i,
                    "name": device["name"],
                    "channels": device["max_input_channels"],
                    "sample_rate": device["default_samplerate"],
                }
                for i, device in enumerate(devices)
                if device["max_input_channels"] > 0
            ]
        except Exception as e:
            logger.error(f"Error listing devices: {e}")
            return []

    def start_capture(self) -> bool:
        """
        Start audio capture in a background thread.

        Returns:
            True if capture started successfully
        """
        if self.is_capturing:
            logger.warning("Audio capture already running")
            return False

        self.is_capturing = True
        self.start_time = time.time()
        self.total_samples_captured = 0
        self.total_chunks_captured = 0
        self.overflow_count = 0

        # Choose worker based on mode
        if self.file_mode:
            worker_target = self._file_playback_worker
            thread_name = "AudioFilePlayback"
        else:
            worker_target = self._capture_worker
            thread_name = "AudioCapture"

        self.capture_thread = threading.Thread(target=worker_target, daemon=True, name=thread_name)
        self.capture_thread.start()

        logger.info(f"Audio capture started ({'FILE' if self.file_mode else 'LIVE'} mode)")
        return True

    def stop_capture(self):
        """Stop audio capture and clean up resources"""
        if not self.is_capturing:
            return

        self.is_capturing = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        # Shutdown callback thread pool
        if hasattr(self, "callback_executor"):
            self.callback_executor.shutdown(wait=False, cancel_futures=True)

        duration = time.time() - self.start_time
        logger.info(
            f"Audio capture stopped: {self.total_chunks_captured} chunks, "
            f"{self.total_samples_captured} samples, "
            f"{duration:.1f}s, "
            f"{self.overflow_count} overflows, "
            f"{self.dropped_callbacks} dropped callbacks"
        )

    def _capture_worker(self):
        """Worker thread for audio capture using sounddevice callback-based approach"""
        logger.info(f"Audio capture worker started (device={self.device_index})")

        last_overflow_log = [0.0]  # Use list for mutable access in callback
        last_status_log = [0.0]
        status_log_interval = 5.0

        def audio_stream_callback(indata, frames, time_info, status):
            """Callback called by sounddevice for each audio chunk"""
            if status and status.input_overflow:
                self.overflow_count += 1
                current_time = time.time()
                if current_time - last_overflow_log[0] > 5.0:
                    logger.warning(f"Audio overflow #{self.overflow_count} - processing may be too slow")
                    last_overflow_log[0] = current_time

            # Flatten to 1D array if needed
            audio_chunk = indata.copy()  # Make copy since indata will be reused
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.flatten()

            # Update statistics
            self.total_samples_captured += len(audio_chunk)
            self.total_chunks_captured += 1
            chunk_timestamp = time.time()

            # Add to queue (non-blocking) - only if queue exists
            if self.audio_queue is not None:
                try:
                    self.audio_queue.put_nowait((audio_chunk, chunk_timestamp))
                except queue.Full:
                    logger.warning("Audio queue full - dropping chunk")

            # Call audio callback asynchronously if provided
            if self.audio_callback:
                try:
                    # Submit callback to thread pool for async execution
                    self.callback_executor.submit(self.audio_callback, audio_chunk, chunk_timestamp)
                except Exception as e:
                    self.dropped_callbacks += 1

            # Periodic status logging
            current_time = time.time()
            if current_time - last_status_log[0] >= status_log_interval:
                duration = current_time - self.start_time
                rms = np.sqrt(np.mean(audio_chunk * audio_chunk))
                logger.info(
                    f"Audio capture status: {self.total_chunks_captured} chunks, "
                    f"{self.total_samples_captured} samples, "
                    f"{duration:.1f}s, "
                    f"RMS={rms:.6f}"
                )
                last_status_log[0] = current_time

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.chunk_size,
                device=self.device_index,
                callback=audio_stream_callback,
            ) as stream:
                logger.info(
                    f"Audio stream opened: {self.config.sample_rate}Hz, "
                    f"{self.config.channels}ch, "
                    f"chunk={self.config.chunk_size} samples"
                )

                # Keep thread alive while capturing
                while self.is_capturing:
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")

        logger.info("Audio capture worker stopped")

    def _file_playback_worker(self):
        """Worker thread for playing back audio from file"""
        logger.info(f"File playback worker started (file={self.config.file_path})")

        # Status logging variables
        last_status_log = 0.0
        status_log_interval = 5.0

        try:
            # Open WAV file
            with wave.open(str(self.config.file_path), "rb") as wav_file:
                # Get file parameters
                file_sample_rate = wav_file.getframerate()
                file_channels = wav_file.getnchannels()
                file_sample_width = wav_file.getsampwidth()
                total_frames = wav_file.getnframes()
                file_duration = total_frames / file_sample_rate

                logger.info(
                    f"WAV file opened: {file_sample_rate}Hz, {file_channels}ch, "
                    f"{file_sample_width} bytes/sample, duration={file_duration:.1f}s"
                )

                # Validate file parameters match config
                if file_sample_rate != self.config.sample_rate:
                    logger.warning(
                        f"File sample rate ({file_sample_rate}Hz) differs from config "
                        f"({self.config.sample_rate}Hz) - using file rate"
                    )

                if file_channels != self.config.channels:
                    logger.warning(f"File channels ({file_channels}) differs from config " f"({self.config.channels})")

                # Calculate chunk timing
                chunk_duration = self.config.chunk_size / file_sample_rate  # seconds per chunk
                playback_chunk_duration = chunk_duration / self.config.playback_speed  # adjusted for speed

                logger.info(
                    f"Playback: chunk_size={self.config.chunk_size}, "
                    f"chunk_duration={chunk_duration*1000:.1f}ms, "
                    f"playback_speed={self.config.playback_speed}x"
                )

                # Read and playback chunks
                chunk_start_time = time.time()
                chunks_played = 0
                loop_count = 0
                first_chunk_ever_logged = False  # Track first chunk across all loops

                while self.is_capturing:
                    # Read chunk from file
                    frames_data = wav_file.readframes(self.config.chunk_size)

                    if not frames_data:
                        # End of file - loop back to beginning or stop
                        loop_count += 1
                        loop_time = time.time()
                        logger.info(f"End of file reached - looping (loop #{loop_count})")
                        logger.info(
                            f"AUDIO_LOOP: wall_time={loop_time:.6f}, loop_count={loop_count}, "
                            f"file={self.config.file_path}"
                        )
                        wav_file.rewind()
                        chunks_played = 0
                        chunk_start_time = time.time()
                        # DON'T reset first_chunk_ever_logged - only log AUDIO_START once per session
                        continue

                    # Convert bytes to numpy array
                    if file_sample_width == 2:
                        # 16-bit PCM
                        audio_chunk = np.frombuffer(frames_data, dtype=np.int16).astype(np.float32) / 32768.0
                    elif file_sample_width == 4:
                        # 32-bit PCM
                        audio_chunk = np.frombuffer(frames_data, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        logger.error(f"Unsupported sample width: {file_sample_width}")
                        break

                    # Handle multi-channel if needed
                    if file_channels > 1:
                        # Reshape and average channels to mono
                        audio_chunk = audio_chunk.reshape(-1, file_channels)
                        audio_chunk = np.mean(audio_chunk, axis=1)

                    # Ensure we have the right chunk size
                    if len(audio_chunk) < self.config.chunk_size:
                        # Pad last chunk
                        audio_chunk = np.pad(audio_chunk, (0, self.config.chunk_size - len(audio_chunk)))

                    # Update statistics
                    self.total_samples_captured += len(audio_chunk)
                    self.total_chunks_captured += 1
                    chunk_timestamp = time.time()
                    chunks_played += 1

                    # Log first chunk for timeline alignment (only once, not per loop)
                    if not first_chunk_ever_logged:
                        first_chunk_ever_logged = True
                        # Backdate by chunk duration - first sample started chunk_duration ago
                        audio_start_time = chunk_timestamp - chunk_duration
                        logger.info(
                            f"AUDIO_START: wall_time={audio_start_time:.6f}, file={self.config.file_path}, "
                            f"sample_rate={file_sample_rate}, chunk_size={self.config.chunk_size}, "
                            f"chunk_duration_ms={chunk_duration*1000:.1f}"
                        )

                    # Add to queue (non-blocking) - only if queue exists
                    if self.audio_queue is not None:
                        try:
                            self.audio_queue.put_nowait((audio_chunk, chunk_timestamp))
                        except queue.Full:
                            logger.warning("Audio queue full - dropping chunk")

                    # Call audio callback if provided
                    if self.audio_callback:
                        try:
                            # Submit callback to thread pool for async execution
                            self.callback_executor.submit(self.audio_callback, audio_chunk, chunk_timestamp)
                        except Exception as e:
                            self.dropped_callbacks += 1

                    # Periodic status logging
                    current_time = time.time()
                    if current_time - last_status_log >= status_log_interval:
                        duration = current_time - self.start_time
                        rms = np.sqrt(np.mean(audio_chunk * audio_chunk))
                        logger.info(
                            f"File playback status: {self.total_chunks_captured} chunks, "
                            f"{self.total_samples_captured} samples, "
                            f"{duration:.1f}s, "
                            f"RMS={rms:.6f}"
                        )
                        last_status_log = current_time

                    # Timing control - sleep to maintain playback rate
                    expected_time = chunk_start_time + (chunks_played * playback_chunk_duration)
                    current_time = time.time()
                    sleep_time = expected_time - current_time

                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif sleep_time < -0.1:
                        # Falling behind significantly
                        logger.warning(f"Playback falling behind by {-sleep_time:.3f}s")

        except FileNotFoundError:
            logger.error(f"Audio file not found: {self.config.file_path}")
        except Exception as e:
            logger.error(f"File playback error: {e}", exc_info=True)

        logger.info("File playback worker stopped")

    def read_chunk(self, timeout: float = 1.0) -> Optional[tuple[np.ndarray, float]]:
        """
        Read next audio chunk from queue.

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (audio_chunk, timestamp) or None if timeout or queue disabled
        """
        if self.audio_queue is None:
            logger.warning("read_chunk called but queue is disabled (callback mode is active)")
            return None

        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_statistics(self) -> dict:
        """
        Get capture statistics.

        Returns:
            Dictionary of statistics
        """
        duration = time.time() - self.start_time if self.is_capturing else 0.0
        return {
            "is_capturing": self.is_capturing,
            "duration": duration,
            "total_chunks": self.total_chunks_captured,
            "total_samples": self.total_samples_captured,
            "overflow_count": self.overflow_count,
            "dropped_callbacks": self.dropped_callbacks,
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # List available devices
    config = AudioConfig()
    capture = AudioCapture(config)

    print("\nAvailable audio devices:")
    for device in capture.list_devices():
        print(f"  [{device['index']}] {device['name']} - {device['channels']}ch @ {device['sample_rate']}Hz")

    # Test capture for 5 seconds
    print("\nStarting 5-second audio capture test...")
    capture.start_capture()

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        capture.stop_capture()
        stats = capture.get_statistics()
        print("\nCapture statistics:")
        print(f"  Duration: {stats['duration']:.1f}s")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Samples: {stats['total_samples']}")
        print(f"  Overflows: {stats['overflow_count']}")
