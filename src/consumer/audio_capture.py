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
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


@dataclass
class AGCConfig:
    """Configuration for Automatic Gain Control"""

    enabled: bool = True  # Enable/disable AGC
    target_rms: float = 0.25  # Target RMS level (leaves headroom for peaks)
    rms_window_ms: float = 200.0  # RMS calculation window in milliseconds
    attack_time_ms: float = 50.0  # Fast attack when signal exceeds target
    release_time_s: float = 120.0  # Slow release (2 minutes) for gain increase
    max_gain_db: float = 20.0  # Maximum gain in dB (10x multiplier)
    min_gain_db: float = -20.0  # Minimum gain in dB (0.1x multiplier)
    noise_gate_threshold: float = 0.01  # Below this RMS, freeze gain increase


@dataclass
class BassBoostConfig:
    """Configuration for bass boost EQ filter.

    Compensates for poor low-frequency response in typical USB microphones.
    Uses a low-shelf filter to boost frequencies below the cutoff.
    """

    enabled: bool = True  # Enable/disable bass boost
    gain_db: float = 6.0  # Boost amount in dB (default +6dB based on mic analysis)
    cutoff_hz: float = 150.0  # Corner frequency in Hz
    q_factor: float = 0.707  # Q factor (0.707 = Butterworth, no resonance)


class BassBoostFilter:
    """
    Low-shelf filter for bass boost compensation.

    Compensates for the poor low-frequency response typical of USB microphones.
    Uses a biquad low-shelf filter design.
    """

    def __init__(self, config: BassBoostConfig, sample_rate: int):
        """
        Initialize bass boost filter.

        Args:
            config: Bass boost configuration
            sample_rate: Audio sample rate in Hz
        """
        self.config = config
        self.sample_rate = sample_rate

        # Design the low-shelf biquad filter
        self.b, self.a = self._design_low_shelf(config.gain_db, config.cutoff_hz, sample_rate, config.q_factor)

        # Filter state for continuity between chunks (zi for lfilter_zi)
        self.zi = scipy_signal.lfilter_zi(self.b, self.a)

        logger.info(
            f"BassBoost initialized: gain={config.gain_db:+.1f}dB, "
            f"cutoff={config.cutoff_hz:.0f}Hz, Q={config.q_factor:.2f}"
        )

    def _design_low_shelf(
        self, gain_db: float, cutoff_hz: float, sample_rate: int, Q: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Design a low-shelf biquad filter.

        Args:
            gain_db: Gain in dB (positive = boost, negative = cut)
            cutoff_hz: Corner frequency in Hz
            sample_rate: Sample rate in Hz
            Q: Q factor (controls slope steepness)

        Returns:
            Tuple of (b, a) filter coefficients
        """
        A = 10 ** (gain_db / 40)  # Amplitude
        w0 = 2 * np.pi * cutoff_hz / sample_rate
        alpha = np.sin(w0) / (2 * Q)

        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)

        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha

        b = np.array([b0 / a0, b1 / a0, b2 / a0])
        a = np.array([1.0, a1 / a0, a2 / a0])

        return b, a

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Apply bass boost filter to audio chunk.

        Args:
            audio_chunk: Input audio samples (float32)

        Returns:
            Filtered audio chunk
        """
        if not self.config.enabled:
            return audio_chunk

        # Apply filter with state continuity
        output, self.zi = scipy_signal.lfilter(self.b, self.a, audio_chunk, zi=self.zi * audio_chunk[0])

        return output.astype(np.float32)

    def reset(self):
        """Reset filter state."""
        self.zi = scipy_signal.lfilter_zi(self.b, self.a)


@dataclass
class HighpassConfig:
    """Configuration for highpass filter to remove rumble/sub-bass noise.

    Removes low-frequency rumble that can contaminate beat detection.
    Uses a 2nd-order Butterworth highpass filter.
    """

    enabled: bool = True  # Enable/disable highpass filter
    cutoff_hz: float = 35.0  # Cutoff frequency in Hz (removes rumble below this)
    order: int = 2  # Filter order (2 = 12dB/octave rolloff)


class HighpassFilter:
    """
    Highpass filter for rumble removal.

    Removes low-frequency noise (room rumble, HVAC, handling noise) that can
    contaminate bass detection and cause false positives in beat detection.
    Uses a Butterworth highpass design for flat passband response.
    """

    def __init__(self, config: HighpassConfig, sample_rate: int):
        """
        Initialize highpass filter.

        Args:
            config: Highpass configuration
            sample_rate: Audio sample rate in Hz
        """
        self.config = config
        self.sample_rate = sample_rate

        # Design Butterworth highpass filter
        # Use sos (second-order sections) for numerical stability
        self.sos = scipy_signal.butter(
            config.order,
            config.cutoff_hz,
            btype="highpass",
            fs=sample_rate,
            output="sos",
        )

        # Filter state for continuity between chunks
        self.zi = scipy_signal.sosfilt_zi(self.sos)

        logger.info(
            f"HighpassFilter initialized: cutoff={config.cutoff_hz:.0f}Hz, "
            f"order={config.order} ({config.order * 6}dB/octave rolloff)"
        )

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Apply highpass filter to audio chunk.

        Args:
            audio_chunk: Input audio samples (float32)

        Returns:
            Filtered audio chunk with rumble removed
        """
        if not self.config.enabled:
            return audio_chunk

        # Apply filter with state continuity
        output, self.zi = scipy_signal.sosfilt(self.sos, audio_chunk, zi=self.zi * audio_chunk[0])

        return output.astype(np.float32)

    def reset(self):
        """Reset filter state."""
        self.zi = scipy_signal.sosfilt_zi(self.sos)


class AutomaticGainControl:
    """
    Automatic Gain Control for microphone input normalization.

    Uses RMS-based level detection with asymmetric attack/release:
    - Fast attack: Quickly reduce gain when signal saturates
    - Slow release: Gradually increase gain over minutes for quiet signals
    - Noise gate: Prevents amplifying background noise during silence
    """

    def __init__(self, config: AGCConfig, sample_rate: int, chunk_size: int):
        """
        Initialize AGC.

        Args:
            config: AGC configuration parameters
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of samples per audio chunk
        """
        self.config = config
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Current gain (linear, starts at unity)
        self.current_gain = 1.0

        # Precompute gain limits (convert dB to linear)
        self.max_gain = 10 ** (config.max_gain_db / 20.0)  # 20dB = 10x
        self.min_gain = 10 ** (config.min_gain_db / 20.0)  # -20dB = 0.1x

        # Precompute smoothing coefficients
        # Attack: time constant in samples, then convert to per-chunk alpha
        chunk_duration_s = chunk_size / sample_rate
        attack_time_s = config.attack_time_ms / 1000.0
        release_time_s = config.release_time_s

        # Alpha = 1 - exp(-chunk_duration / time_constant)
        # Higher alpha = faster response
        self.attack_alpha = 1.0 - np.exp(-chunk_duration_s / attack_time_s)
        self.release_alpha = 1.0 - np.exp(-chunk_duration_s / release_time_s)

        # RMS smoothing for stable level detection
        rms_window_s = config.rms_window_ms / 1000.0
        self.rms_alpha = 1.0 - np.exp(-chunk_duration_s / rms_window_s)
        self.smoothed_rms = 0.0

        # Statistics
        self.peak_gain_applied = 1.0
        self.min_gain_applied = 1.0
        self.gain_adjustments = 0
        self.noise_gate_activations = 0

        logger.info(
            f"AGC initialized: target_rms={config.target_rms:.2f}, "
            f"gain_range=[{config.min_gain_db:.0f}dB, {config.max_gain_db:.0f}dB], "
            f"attack={config.attack_time_ms:.0f}ms, release={config.release_time_s:.0f}s"
        )

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Apply AGC to an audio chunk.

        Args:
            audio_chunk: Input audio samples (float32, -1.0 to 1.0 range)

        Returns:
            Gain-adjusted audio chunk
        """
        if not self.config.enabled:
            return audio_chunk

        # Calculate RMS of input chunk
        chunk_rms = np.sqrt(np.mean(audio_chunk * audio_chunk))

        # Smooth the RMS measurement
        self.smoothed_rms += self.rms_alpha * (chunk_rms - self.smoothed_rms)

        # Determine target gain based on smoothed RMS
        if self.smoothed_rms > 0.0001:  # Avoid division by near-zero
            desired_gain = self.config.target_rms / self.smoothed_rms
        else:
            desired_gain = self.current_gain  # Hold current gain

        # Clamp desired gain to limits
        desired_gain = np.clip(desired_gain, self.min_gain, self.max_gain)

        # Apply asymmetric smoothing
        if desired_gain < self.current_gain:
            # Need to reduce gain (signal too loud) - use fast attack
            alpha = self.attack_alpha
            self.gain_adjustments += 1
        elif self.smoothed_rms < self.config.noise_gate_threshold:
            # Signal below noise gate - freeze gain (don't increase)
            alpha = 0.0
            self.noise_gate_activations += 1
        else:
            # Need to increase gain (signal too quiet) - use slow release
            alpha = self.release_alpha
            if alpha > 0:
                self.gain_adjustments += 1

        # Update gain with smoothing
        self.current_gain += alpha * (desired_gain - self.current_gain)

        # Track statistics
        self.peak_gain_applied = max(self.peak_gain_applied, self.current_gain)
        self.min_gain_applied = min(self.min_gain_applied, self.current_gain)

        # Apply gain to audio
        output = audio_chunk * self.current_gain

        # Soft clip to prevent hard clipping (tanh-based limiting)
        # Only kicks in if signal exceeds ±1.0
        output = np.tanh(output)

        return output

    def get_gain_db(self) -> float:
        """Get current gain in dB."""
        return 20.0 * np.log10(max(self.current_gain, 1e-10))

    def get_statistics(self) -> dict:
        """Get AGC statistics."""
        return {
            "enabled": self.config.enabled,
            "current_gain": self.current_gain,
            "current_gain_db": self.get_gain_db(),
            "smoothed_rms": self.smoothed_rms,
            "peak_gain_applied": self.peak_gain_applied,
            "min_gain_applied": self.min_gain_applied,
            "gain_adjustments": self.gain_adjustments,
            "noise_gate_activations": self.noise_gate_activations,
        }

    def reset(self):
        """Reset AGC to initial state."""
        self.current_gain = 1.0
        self.smoothed_rms = 0.0
        self.peak_gain_applied = 1.0
        self.min_gain_applied = 1.0
        self.gain_adjustments = 0
        self.noise_gate_activations = 0


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
    agc: Optional[AGCConfig] = None  # AGC configuration (None = disabled, AGCConfig() = enabled with defaults)
    highpass: Optional[HighpassConfig] = None  # Highpass filter for rumble removal (None = disabled)
    bass_boost: Optional[BassBoostConfig] = None  # Bass boost EQ (None = disabled, BassBoostConfig() = enabled)


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

        # Recording state (for capturing processed audio to WAV file)
        self._recording_lock = threading.Lock()
        self._is_recording = False
        self._recording_buffer: list[np.ndarray] = []
        self._recording_start_time = 0.0
        self._recording_duration = 0.0
        self._recording_output_path: Optional[Path] = None
        self._recording_samples_target = 0
        self._recording_samples_collected = 0

        # Initialize AGC if configured (only for live mode)
        self.agc: Optional[AutomaticGainControl] = None
        if not self.file_mode and self.config.agc is not None:
            self.agc = AutomaticGainControl(
                config=self.config.agc,
                sample_rate=self.config.sample_rate,
                chunk_size=self.config.chunk_size,
            )

        # Initialize highpass filter if configured (only for live mode)
        # Applied after AGC but before bass boost to remove rumble
        self.highpass: Optional[HighpassFilter] = None
        if not self.file_mode and self.config.highpass is not None:
            self.highpass = HighpassFilter(
                config=self.config.highpass,
                sample_rate=self.config.sample_rate,
            )

        # Initialize bass boost if configured (only for live mode)
        self.bass_boost: Optional[BassBoostFilter] = None
        if not self.file_mode and self.config.bass_boost is not None:
            self.bass_boost = BassBoostFilter(
                config=self.config.bass_boost,
                sample_rate=self.config.sample_rate,
            )

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

            # Apply AGC if enabled
            if self.agc is not None:
                audio_chunk = self.agc.process(audio_chunk)

            # Apply highpass filter if enabled (after AGC, before bass boost)
            # Removes rumble/sub-bass noise that can contaminate beat detection
            if self.highpass is not None:
                audio_chunk = self.highpass.process(audio_chunk)

            # Apply bass boost EQ if enabled (after AGC and highpass to boost clean signal)
            if self.bass_boost is not None:
                audio_chunk = self.bass_boost.process(audio_chunk)

            # Collect sample for recording if active (after all processing)
            self._collect_recording_sample(audio_chunk)

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
                agc_info = ""
                if self.agc is not None:
                    agc_info = f", AGC_gain={self.agc.get_gain_db():.1f}dB"
                logger.debug(
                    f"Audio capture status: {self.total_chunks_captured} chunks, "
                    f"{self.total_samples_captured} samples, "
                    f"{duration:.1f}s, "
                    f"RMS={rms:.6f}{agc_info}"
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
        stats = {
            "is_capturing": self.is_capturing,
            "duration": duration,
            "total_chunks": self.total_chunks_captured,
            "total_samples": self.total_samples_captured,
            "overflow_count": self.overflow_count,
            "dropped_callbacks": self.dropped_callbacks,
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
        }

        # Include AGC statistics if enabled
        if self.agc is not None:
            stats["agc"] = self.agc.get_statistics()

        return stats

    def start_recording(self, duration_seconds: float, output_path: Path) -> bool:
        """
        Start recording processed audio to a WAV file.

        Records audio after AGC and bass boost processing, allowing offline
        analysis of the processed audio signal.

        Args:
            duration_seconds: Recording duration in seconds
            output_path: Path to output WAV file

        Returns:
            True if recording started successfully, False if already recording or not capturing
        """
        if not self.is_capturing:
            logger.error("Cannot start recording - audio capture not running")
            return False

        if self.file_mode:
            logger.error("Cannot record in file playback mode")
            return False

        with self._recording_lock:
            if self._is_recording:
                logger.warning("Recording already in progress")
                return False

            self._recording_buffer = []
            self._recording_start_time = time.time()
            self._recording_duration = duration_seconds
            self._recording_output_path = Path(output_path)
            self._recording_samples_target = int(duration_seconds * self.config.sample_rate)
            self._recording_samples_collected = 0
            self._is_recording = True

            logger.info(
                f"Started recording: duration={duration_seconds}s, "
                f"target_samples={self._recording_samples_target}, "
                f"output={output_path}"
            )
            return True

    def _collect_recording_sample(self, audio_chunk: np.ndarray) -> None:
        """
        Collect audio chunk for recording (called from capture callback).

        Args:
            audio_chunk: Processed audio chunk to record
        """
        with self._recording_lock:
            if not self._is_recording:
                return

            # Calculate how many samples we still need
            samples_needed = self._recording_samples_target - self._recording_samples_collected

            if samples_needed <= 0:
                # Recording complete - finalize in background
                self._is_recording = False
                # Use thread pool to avoid blocking the audio callback
                self.callback_executor.submit(self._finalize_recording)
                return

            # Collect the chunk (or partial chunk if near the end)
            if len(audio_chunk) <= samples_needed:
                self._recording_buffer.append(audio_chunk.copy())
                self._recording_samples_collected += len(audio_chunk)
            else:
                # Only take what we need
                self._recording_buffer.append(audio_chunk[:samples_needed].copy())
                self._recording_samples_collected += samples_needed
                self._is_recording = False
                self.callback_executor.submit(self._finalize_recording)

    def _finalize_recording(self) -> None:
        """Write recorded audio buffer to WAV file."""
        with self._recording_lock:
            if not self._recording_buffer:
                logger.warning("No audio data to write")
                return

            output_path = self._recording_output_path
            buffer = self._recording_buffer
            sample_rate = self.config.sample_rate

            # Clear buffer reference (but keep local copy for writing)
            self._recording_buffer = []

        try:
            # Concatenate all chunks
            audio_data = np.concatenate(buffer)

            # Convert float32 [-1, 1] to int16 for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write WAV file
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            duration = len(audio_data) / sample_rate
            logger.info(
                f"Recording saved: {output_path}, "
                f"duration={duration:.1f}s, "
                f"samples={len(audio_data)}, "
                f"size={output_path.stat().st_size / 1024:.1f}KB"
            )

        except Exception as e:
            logger.error(f"Failed to write recording: {e}", exc_info=True)

    def get_recording_status(self) -> dict:
        """
        Get current recording status.

        Returns:
            Dictionary with recording state and progress
        """
        with self._recording_lock:
            if self._is_recording:
                elapsed = time.time() - self._recording_start_time
                progress = min(1.0, self._recording_samples_collected / max(1, self._recording_samples_target))
                return {
                    "is_recording": True,
                    "elapsed_seconds": elapsed,
                    "duration_seconds": self._recording_duration,
                    "progress": progress,
                    "samples_collected": self._recording_samples_collected,
                    "samples_target": self._recording_samples_target,
                    "output_path": str(self._recording_output_path) if self._recording_output_path else None,
                }
            else:
                return {
                    "is_recording": False,
                    "elapsed_seconds": 0.0,
                    "duration_seconds": 0.0,
                    "progress": 0.0,
                    "samples_collected": 0,
                    "samples_target": 0,
                    "output_path": None,
                }

    @property
    def is_recording(self) -> bool:
        """Check if recording is in progress."""
        with self._recording_lock:
            return self._is_recording


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
