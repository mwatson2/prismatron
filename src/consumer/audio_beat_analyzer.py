#!/usr/bin/env python3
"""
Audio Beat Analyzer for Prismatron LED Display System

This module provides real-time beat detection, BPM tracking, and downbeat identification
using the BeatNet library for audioreactive LED effects.

Key Features:
- Real-time beat detection from microphone input
- BPM calculation with smoothing
- Downbeat identification and prediction
- Beat intensity analysis
- Asynchronous processing with event callbacks
- GPU acceleration support
"""

import logging
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.consumer.audio_capture import AudioCapture, AudioConfig

# Audio processing imports
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available - audio processing will be limited")

# Aubio import with fallback
try:
    import aubio

    AUBIO_AVAILABLE = True
    # Note: logger is defined later, so we'll log this in the class initialization
except ImportError as e:
    AUBIO_AVAILABLE = False
    logging.warning(f"Aubio not available: {e}")
except Exception as e:
    AUBIO_AVAILABLE = False
    logging.warning(f"Aubio import issue: {e}")

# CUDA/PyTorch imports (defer CUDA check to avoid initialization issues in subprocesses)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _check_cuda_available() -> bool:
    """Check if CUDA is available, only when needed to avoid initialization in fork."""
    if not TORCH_AVAILABLE:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


logger = logging.getLogger(__name__)


@dataclass
class BeatEvent:
    """Data structure for beat events"""

    timestamp: float  # Beat time in seconds (from audio start)
    system_time: float  # System time when detected
    is_downbeat: bool  # True for downbeats (measure boundaries)
    bpm: float  # Current BPM estimate
    intensity: float  # Beat intensity (0.0-1.0)
    confidence: float  # Detection confidence (0.0-1.0)
    beat_count: int  # Total beat count since start


@dataclass
class AudioState:
    """Current audio analysis state"""

    is_active: bool = False
    current_bpm: float = 120.0
    last_beat_time: float = 0.0
    last_downbeat_time: float = 0.0
    beat_count: int = 0
    downbeat_count: int = 0
    beats_per_measure: int = 4
    confidence: float = 0.0
    beat_intensity: float = 0.0  # Last beat's RMS intensity (0.0-1.0)


class MockAubio:
    """Mock Aubio implementation for testing when Aubio is unavailable"""

    def __init__(self, sample_rate=44100, hop_size=512):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.start_time = time.time()
        self.last_beat = 0.0
        self.bpm = 120.0  # Default BPM
        self.frame_count = 0
        logger.info(f"MockAubio initialized (sample_rate={sample_rate}, hop_size={hop_size})")

    def process_frame(self, audio_frame):
        """Mock beat detection - generates periodic beats for testing"""
        self.frame_count += 1
        current_time = self.frame_count * self.hop_size / self.sample_rate
        beat_interval = 60.0 / self.bpm

        # Generate beats if enough time has passed
        if current_time - self.last_beat >= beat_interval:
            self.last_beat = current_time
            return True  # Simple boolean return

        return False

    def get_bpm(self):
        return self.bpm

    def get_confidence(self):
        return 0.8  # Mock confidence


class BeatIntensityAnalyzer:
    """Analyzes beat intensity from audio data"""

    def __init__(self, sample_rate=22050, window_size=1024):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.intensity_history = deque(maxlen=10)

    def analyze_intensity(self, audio_buffer: np.ndarray, beat_timestamp: float) -> float:
        """
        Analyze beat intensity from audio buffer around beat timestamp.

        Args:
            audio_buffer: Audio data array
            beat_timestamp: Time of detected beat

        Returns:
            Intensity score (0.0-1.0)
        """
        if not LIBROSA_AVAILABLE or len(audio_buffer) == 0:
            # Return mock intensity if no audio processing available
            return np.random.uniform(0.3, 1.0)

        try:
            # Calculate RMS energy around beat time
            # This is a simplified implementation - could be enhanced with spectral analysis
            rms = librosa.feature.rms(y=audio_buffer, frame_length=self.window_size)[0]
            intensity = np.mean(rms) * 2.0  # Scale to approximate 0-1 range
            intensity = np.clip(intensity, 0.0, 1.0)

            # Smooth intensity with history
            self.intensity_history.append(intensity)
            smoothed_intensity = np.mean(self.intensity_history)

            return float(smoothed_intensity)

        except Exception as e:
            logger.warning(f"Intensity analysis failed: {e}")
            return 0.5  # Default intensity


class BPMCalculator:
    """Calculates and smooths BPM from beat timestamps"""

    def __init__(self, history_size=16, smoothing_alpha=0.3):
        self.beat_history = deque(maxlen=history_size)
        self.smoothing_alpha = smoothing_alpha
        self.current_bpm = 120.0
        self.bpm_confidence = 0.0

    def update_beat(self, timestamp: float) -> tuple[float, float]:
        """
        Update BPM calculation with new beat timestamp.

        Args:
            timestamp: Beat timestamp in seconds

        Returns:
            Tuple of (bpm, confidence)
        """
        self.beat_history.append(timestamp)

        if len(self.beat_history) < 2:
            return self.current_bpm, 0.0

        # Calculate instantaneous BPM from recent beats
        time_diffs = np.diff(list(self.beat_history))

        # Filter out unrealistic intervals (< 0.3s or > 2.0s)
        valid_diffs = time_diffs[(time_diffs > 0.3) & (time_diffs < 2.0)]

        if len(valid_diffs) == 0:
            return self.current_bpm, 0.0

        # Calculate BPM from mean interval
        mean_interval = np.mean(valid_diffs)
        instantaneous_bpm = 60.0 / mean_interval

        # Apply exponential smoothing
        self.current_bpm = self.smoothing_alpha * instantaneous_bpm + (1 - self.smoothing_alpha) * self.current_bpm

        # Calculate confidence based on consistency
        interval_std = np.std(valid_diffs)
        self.bpm_confidence = max(0.0, 1.0 - (interval_std / mean_interval) * 2.0)

        return self.current_bpm, self.bpm_confidence


class AudioBeatAnalyzer:
    """
    Main audio beat analysis class for real-time beat detection and BPM tracking.

    Integrates BeatNet for beat detection with custom BPM calculation and intensity analysis.
    Uses sounddevice for reliable audio capture with USB microphone support.
    Designed to run in a separate thread to avoid blocking LED processing.
    """

    def __init__(
        self,
        beat_callback: Optional[Callable[[BeatEvent], None]] = None,
        model: int = 1,
        device: str = "auto",
        sample_rate: int = 22050,
        buffer_size: int = 1024,
        audio_config: Optional[AudioConfig] = None,
    ):
        """
        Initialize audio beat analyzer.

        Args:
            beat_callback: Callback function for beat events
            model: BeatNet model selection (1-3)
            device: Processing device ('auto', 'cuda', 'cpu')
            sample_rate: Audio sample rate
            buffer_size: Audio buffer size
            audio_config: Optional AudioConfig for custom audio capture settings (file mode, etc)
        """
        self.beat_callback = beat_callback
        self.sample_rate = sample_rate  # Aubio can work with 44100Hz directly
        self.capture_rate = 44100  # USB device native rate
        self.buffer_size = buffer_size
        self.hop_size = 512  # Aubio hop size for processing (~11.6ms at 44.1kHz)
        self.chunk_size = 1024  # Audio chunk size for capture (~23ms at 44.1kHz)

        # Processing components
        self.bpm_calculator = BPMCalculator()
        self.intensity_analyzer = BeatIntensityAnalyzer(self.sample_rate, buffer_size)

        # Threading
        self.running = False
        self.beat_thread = None
        self.beat_queue = queue.Queue(maxsize=100)

        # Thread pool for non-blocking beat callback execution
        self.callback_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="BeatCallback")

        # State tracking
        self.audio_state = AudioState()
        self.start_time = time.time()

        # Audio processing statistics
        self.total_frames_processed = 0
        self.total_beats_detected = 0
        self.last_status_log = 0.0
        self.last_beat_audio_timestamp = 0.0  # Track last beat in audio time, not wall clock

        # Aubio doesn't use GPU - always CPU-based but very fast
        self.device = "cpu"  # Aubio is CPU-only but optimized

        # Initialize Aubio
        self._initialize_aubio()

        # Create AudioCapture instance with callback for audio processing
        if audio_config is None:
            # Default config for live microphone capture
            audio_config = AudioConfig(
                sample_rate=self.capture_rate, channels=1, chunk_size=self.chunk_size, device_name="USB Audio"
            )
        self.audio_capture = AudioCapture(audio_config, audio_callback=self._on_audio_chunk)

        # Log aubio availability
        if AUBIO_AVAILABLE:
            try:
                logger.info(f"Aubio available: version {aubio.version}")
            except Exception:
                logger.info("Aubio available (version unknown)")

        logger.info(
            f"AudioBeatAnalyzer initialized (aubio, hop_size={self.hop_size}, "
            f"USB device={self.audio_capture.device_index})"
        )

    def _initialize_aubio(self):
        """Initialize Aubio with fallback to mock implementation"""
        if AUBIO_AVAILABLE:
            try:
                # Create aubio tempo tracker
                win_s = self.hop_size * 2  # Window size
                self.aubio_tempo = aubio.tempo("specdiff", win_s, self.hop_size, self.capture_rate)

                # Aubio beat tracking
                self.beat_times = deque(maxlen=32)

                logger.info("Aubio initialized successfully")
                return
            except Exception as e:
                logger.error(f"Aubio initialization failed: {e}")

        # Fallback to mock implementation
        self.aubio_tempo = MockAubio(sample_rate=self.capture_rate, hop_size=self.hop_size)
        self.beat_times = deque(maxlen=32)
        logger.warning("Using MockAubio for testing (Aubio unavailable)")

    def start_analysis(self):
        """Start real-time audio analysis in separate threads"""
        if self.running:
            logger.warning("Audio analysis already running")
            return

        self.running = True
        self.start_time = time.time()
        self.audio_state.is_active = True

        # Start audio capture using AudioCapture class
        self.audio_capture.start_capture()

        # Start beat event processing thread
        self.beat_event_thread = threading.Thread(target=self._beat_worker, daemon=True)
        self.beat_event_thread.start()

        logger.info("Audio beat analysis started with AudioCapture")

    def stop_analysis(self):
        """Stop audio analysis and clean up resources"""
        if not self.running:
            return

        self.running = False
        self.audio_state.is_active = False

        # Stop audio capture
        self.audio_capture.stop_capture()

        # Wait for beat processing thread to finish
        if hasattr(self, "beat_event_thread") and self.beat_event_thread.is_alive():
            self.beat_event_thread.join(timeout=2.0)

        # Shutdown callback thread pool
        if hasattr(self, "callback_executor"):
            self.callback_executor.shutdown(wait=False, cancel_futures=True)

        logger.info("Audio beat analysis stopped")

    def _on_audio_chunk(self, audio_chunk: np.ndarray, timestamp: float):
        """
        Callback for processing audio chunks from AudioCapture.

        Args:
            audio_chunk: Audio data array
            timestamp: System timestamp when chunk was captured
        """
        # Ensure audio chunk is 1D
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()

        # Process audio in hop_size chunks
        frames_to_process = len(audio_chunk) // self.hop_size

        for frame_idx in range(frames_to_process):
            start_idx = frame_idx * self.hop_size
            end_idx = start_idx + self.hop_size
            audio_frame = audio_chunk[start_idx:end_idx].astype(np.float32)

            # Process with Aubio
            beat_detected = False
            if hasattr(self.aubio_tempo, "process_frame"):
                # Mock aubio
                beat_detected = self.aubio_tempo.process_frame(audio_frame)
            else:
                # Real aubio
                beat = self.aubio_tempo(audio_frame)
                beat_detected = bool(beat[0])

            self.total_frames_processed += 1

            if beat_detected:
                # Calculate timestamp relative to audio start (in audio time, not wall clock)
                audio_timestamp = (self.total_frames_processed * self.hop_size) / self.capture_rate

                # Prevent duplicate detections too close together (use audio time, not wall clock)
                if audio_timestamp - self.last_beat_audio_timestamp > 0.2:  # 200ms minimum in audio time
                    current_time = time.time()
                    detection_latency_ms = (current_time - self.start_time - audio_timestamp) * 1000
                    self.last_beat_audio_timestamp = audio_timestamp
                    self.beat_times.append(audio_timestamp)
                    self.total_beats_detected += 1

                    # Calculate beat intensity from RMS of audio frame
                    # RMS gives us the energy/strength of the beat
                    beat_intensity = np.sqrt(np.mean(audio_frame**2))
                    # Normalize to 0-1 range with 5.0x scaling for better dynamic range
                    # This gives: heavy bass ~0.5-1.0, light hi-hats ~0.1-0.3
                    beat_intensity = float(np.clip(beat_intensity * 5.0, 0.0, 1.0))

                    # Log beat detection timing (DEBUG level for latency tracking)
                    logger.debug(
                        f"Beat detected: audio_time={audio_timestamp:.3f}s, "
                        f"wall_time={current_time:.3f}, "
                        f"detection_latency={detection_latency_ms:.1f}ms, "
                        f"intensity={beat_intensity:.3f}"
                    )

                    # Calculate BPM from aubio and intervals
                    aubio_bpm = 120.0  # Default
                    confidence = 0.8

                    if hasattr(self.aubio_tempo, "get_bpm"):
                        aubio_bpm = self.aubio_tempo.get_bpm()
                        confidence = getattr(self.aubio_tempo, "get_confidence", lambda: 0.8)()
                    else:
                        # Real aubio
                        aubio_bpm = self.aubio_tempo.get_bpm()
                        confidence = self.aubio_tempo.get_confidence()

                    # Calculate from intervals as backup
                    if len(self.beat_times) >= 4:
                        intervals = np.diff(list(self.beat_times)[-4:])
                        valid_intervals = intervals[(intervals > 0.3) & (intervals < 1.5)]
                        if len(valid_intervals) > 0:
                            calculated_bpm = 60.0 / np.mean(valid_intervals)
                            # Use aubio BPM if reasonable, otherwise calculated
                            if aubio_bpm > 0 and 60 <= aubio_bpm <= 200:
                                current_bpm = aubio_bpm
                            else:
                                current_bpm = calculated_bpm
                        else:
                            current_bpm = aubio_bpm if aubio_bpm > 0 else 120.0
                    else:
                        current_bpm = aubio_bpm if aubio_bpm > 0 else 120.0

                    # Log occasionally
                    if self.total_beats_detected <= 10 or np.random.random() < 0.1:
                        logger.info(f"🎵 Aubio beat detected: BPM={current_bpm:.1f}, confidence={confidence:.3f}")

                    # Send to beat queue with intensity
                    try:
                        queue_time = time.time()
                        self.beat_queue.put_nowait(
                            (audio_timestamp, 0.1, current_time, beat_intensity)
                        )  # Non-blocking put
                        queue_latency_ms = (queue_time - current_time) * 1000
                        logger.debug(
                            f"Beat queued: beat_count={self.total_beats_detected}, "
                            f"queue_depth={self.beat_queue.qsize()}, "
                            f"queue_latency={queue_latency_ms:.2f}ms, "
                            f"intensity={beat_intensity:.3f}"
                        )
                    except queue.Full:
                        # Drop beat event if queue is full to avoid blocking audio thread
                        logger.warning("Beat queue full - dropping beat event to maintain audio performance")

        # Periodic status logging
        current_time = time.time()
        status_log_interval = 10.0
        if current_time - self.last_status_log >= status_log_interval:
            current_audio_rms = np.sqrt(np.mean(audio_chunk * audio_chunk))

            logger.info(
                f"Audio processing status: {self.total_frames_processed} frames processed, "
                f"{self.total_beats_detected} beats detected, "
                f"audio RMS: {current_audio_rms:.6f}, "
                f"current BPM: {self.audio_state.current_bpm:.1f}"
            )
            self.last_status_log = current_time
            self.total_beats_detected = 0

    def _beat_worker(self):
        """Beat event processing worker thread"""
        logger.info("Beat event processing thread started")

        while self.running:
            try:
                # Get beat data from queue
                beat_data = self.beat_queue.get(timeout=0.1)
                beat_timestamp, downbeat_prob, system_time, intensity = beat_data

                # Process beat event
                self._process_beat_event(beat_timestamp, downbeat_prob, system_time, intensity)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Beat processing error: {e}")

        logger.info("Beat event processing thread stopped")

    def _process_beat_event(self, beat_timestamp: float, downbeat_prob: float, system_time: float, intensity: float):
        """
        Process a detected beat event from Aubio.

        Args:
            beat_timestamp: Beat time in audio timeline (seconds)
            downbeat_prob: Probability of downbeat (>0.5 = downbeat)
            system_time: Wall-clock time when beat was detected
            intensity: Beat intensity from RMS energy (0.0-1.0)
        """
        # Check if this is a new beat (avoid duplicates with tolerance)
        if abs(beat_timestamp - self.audio_state.last_beat_time) < 0.1:
            return

        # Update beat count
        self.audio_state.beat_count += 1
        self.audio_state.last_beat_time = beat_timestamp

        # For Aubio, we simplified downbeat detection (downbeat_prob < 0.5 = regular beat)
        is_downbeat = downbeat_prob > 0.5
        if is_downbeat:
            self.audio_state.downbeat_count += 1
            self.audio_state.last_downbeat_time = beat_timestamp

        # Update BPM calculation
        bpm, confidence = self.bpm_calculator.update_beat(beat_timestamp)
        self.audio_state.current_bpm = bpm
        self.audio_state.confidence = confidence
        self.audio_state.beat_intensity = intensity  # Store intensity for renderer access

        # Use the RMS-based intensity passed from audio processing
        # (intensity is already calculated from the audio frame that triggered the beat)

        # Create beat event
        beat_event = BeatEvent(
            timestamp=beat_timestamp,
            system_time=system_time,
            is_downbeat=is_downbeat,
            bpm=bpm,
            intensity=intensity,
            confidence=confidence,
            beat_count=self.audio_state.beat_count,
        )

        # Log beat event
        beat_type = "DOWNBEAT" if beat_event.is_downbeat else "BEAT"
        logger.debug(
            f"🎵 {beat_type} #{beat_event.beat_count}: BPM={beat_event.bpm:.1f}, "
            f"Intensity={beat_event.intensity:.2f}, "
            f"Confidence={beat_event.confidence:.2f}"
        )

        # Structured logging for timeline reconstruction
        logger.debug(
            f"BEAT_DETECTED: wall_time={beat_event.system_time:.6f}, "
            f"audio_ts={beat_event.timestamp:.6f}, "
            f"intensity={beat_event.intensity:.4f}, "
            f"confidence={beat_event.confidence:.3f}, "
            f"bpm={beat_event.bpm:.1f}, "
            f"is_downbeat={beat_event.is_downbeat}"
        )

        # Call beat callback asynchronously to avoid blocking audio processing
        if self.beat_callback:
            try:
                # Submit callback to thread pool for non-blocking execution
                future = self.callback_executor.submit(self.beat_callback, beat_event)
                # Don't wait for completion - let it run asynchronously
            except Exception as e:
                logger.error(f"Beat callback submission error: {e}")

    def get_current_state(self) -> AudioState:
        """Get current audio analysis state"""
        return self.audio_state

    def predict_next_beat(self, current_time: Optional[float] = None) -> float:
        """
        Predict the timing of the next beat.

        Args:
            current_time: Current time (defaults to system time)

        Returns:
            Predicted next beat timestamp
        """
        if current_time is None:
            current_time = time.time() - self.start_time

        if self.audio_state.last_beat_time == 0 or self.audio_state.current_bpm == 0:
            return current_time + 0.5  # Default prediction

        beat_interval = 60.0 / self.audio_state.current_bpm
        time_since_last = current_time - self.audio_state.last_beat_time

        if time_since_last >= beat_interval:
            return current_time  # Beat should be happening now
        else:
            return self.audio_state.last_beat_time + beat_interval

    def predict_next_downbeat(self, current_time: Optional[float] = None) -> float:
        """
        Predict the timing of the next downbeat.

        Args:
            current_time: Current time (defaults to system time)

        Returns:
            Predicted next downbeat timestamp
        """
        if current_time is None:
            current_time = time.time() - self.start_time

        if self.audio_state.last_downbeat_time == 0 or self.audio_state.current_bpm == 0:
            return current_time + 2.0  # Default prediction

        measure_interval = (60.0 / self.audio_state.current_bpm) * self.audio_state.beats_per_measure
        time_since_last = current_time - self.audio_state.last_downbeat_time

        if time_since_last >= measure_interval:
            return current_time  # Downbeat should be happening now
        else:
            return self.audio_state.last_downbeat_time + measure_interval


# Example usage and testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def beat_event_handler(beat_event: BeatEvent):
        """Example beat event handler"""
        print(
            f"Beat #{beat_event.beat_count}: "
            f"BPM={beat_event.bpm:.1f}, "
            f"Downbeat={'YES' if beat_event.is_downbeat else 'no'}, "
            f"Intensity={beat_event.intensity:.2f}, "
            f"Confidence={beat_event.confidence:.2f}"
        )

    # Create and start beat analyzer
    analyzer = AudioBeatAnalyzer(beat_callback=beat_event_handler)

    try:
        analyzer.start_analysis()
        print("Beat analyzer started. Press Ctrl+C to stop...")

        # Run for testing
        start_time = time.time()
        while time.time() - start_time < 30:  # Run for 30 seconds
            time.sleep(1)

            # Print current state
            state = analyzer.get_current_state()
            print(
                f"State: BPM={state.current_bpm:.1f}, "
                f"Beats={state.beat_count}, "
                f"Downbeats={state.downbeat_count}, "
                f"Active={state.is_active}"
            )

    except KeyboardInterrupt:
        print("\nStopping beat analyzer...")

    finally:
        analyzer.stop_analysis()
        print("Beat analyzer stopped")
