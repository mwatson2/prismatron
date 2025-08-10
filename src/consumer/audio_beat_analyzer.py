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
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Audio processing imports
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available - audio processing will be limited")

# BeatNet import with fallback
try:
    # Compatibility fix for Python 3.10+
    import collections
    import collections.abc

    if not hasattr(collections, "MutableSequence"):
        collections.MutableSequence = collections.abc.MutableSequence

    # Compatibility fix for numpy/madmom issues
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "complex"):
        np.complex = complex
    if not hasattr(np, "bool"):
        np.bool = bool

    from BeatNet.BeatNet import BeatNet

    BEATNET_AVAILABLE = True
except ImportError as e:
    BEATNET_AVAILABLE = False
    logging.warning(f"BeatNet not available: {e}")
except Exception as e:
    BEATNET_AVAILABLE = False
    logging.warning(f"BeatNet compatibility issue: {e}")

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


class MockBeatNet:
    """Mock BeatNet implementation for testing when BeatNet is unavailable"""

    def __init__(self, model=1, mode="stream", inference_model="PF", plot=None, thread=False, device="cpu"):
        self.mode = mode
        self.device = device
        self.thread = thread
        self.start_time = time.time()
        self.last_beat = 0.0
        self.bpm = 120.0  # Default BPM
        logger.info(f"MockBeatNet initialized (mode={mode}, device={device})")

    def process(self, audio_data=None):
        """Mock beat detection - generates periodic beats for testing"""
        current_time = time.time() - self.start_time
        beat_interval = 60.0 / self.bpm

        # Generate beats if enough time has passed
        if current_time - self.last_beat >= beat_interval:
            self.last_beat = current_time
            # Every 4th beat is a downbeat
            is_downbeat = (int(current_time / beat_interval) % 4) == 0
            return np.array([[current_time, 1.0 if is_downbeat else 0.0]])

        return np.array([])


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
    Designed to run in a separate thread to avoid blocking LED processing.
    """

    def __init__(
        self,
        beat_callback: Optional[Callable[[BeatEvent], None]] = None,
        model: int = 1,
        device: str = "auto",
        sample_rate: int = 22050,
        buffer_size: int = 1024,
    ):
        """
        Initialize audio beat analyzer.

        Args:
            beat_callback: Callback function for beat events
            model: BeatNet model selection (1-3)
            device: Processing device ('auto', 'cuda', 'cpu')
            sample_rate: Audio sample rate
            buffer_size: Audio buffer size
        """
        self.beat_callback = beat_callback
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        # Processing components
        self.bpm_calculator = BPMCalculator()
        self.intensity_analyzer = BeatIntensityAnalyzer(sample_rate, buffer_size)

        # Threading
        self.running = False
        self.audio_thread = None
        self.beat_queue = queue.Queue(maxsize=100)

        # State tracking
        self.audio_state = AudioState()
        self.start_time = time.time()

        # Device selection
        if device == "auto":
            device = "cuda" if _check_cuda_available() else "cpu"
        self.device = device

        # Initialize BeatNet
        self._initialize_beatnet(model)

        logger.info(f"AudioBeatAnalyzer initialized (device={device}, model={model})")

    def _initialize_beatnet(self, model: int):
        """Initialize BeatNet with fallback to mock implementation"""
        if BEATNET_AVAILABLE:
            try:
                self.beatnet = BeatNet(
                    model=model,
                    mode="stream",  # Real-time microphone mode
                    inference_model="PF",  # Particle filtering for real-time
                    plot=[],  # No visualization for performance
                    thread=True,  # Use separate thread
                    device=self.device,
                )
                logger.info("BeatNet initialized successfully")
                return
            except Exception as e:
                logger.error(f"BeatNet initialization failed: {e}")

        # Fallback to mock implementation
        self.beatnet = MockBeatNet(model=model, mode="stream", device=self.device, thread=True)
        logger.warning("Using MockBeatNet for testing (BeatNet unavailable)")

    def start_analysis(self):
        """Start real-time audio analysis in separate thread"""
        if self.running:
            logger.warning("Audio analysis already running")
            return

        self.running = True
        self.start_time = time.time()
        self.audio_state.is_active = True

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()

        # Start beat event processing thread
        self.beat_thread = threading.Thread(target=self._beat_worker, daemon=True)
        self.beat_thread.start()

        logger.info("Audio beat analysis started")

    def stop_analysis(self):
        """Stop audio analysis and clean up resources"""
        if not self.running:
            return

        self.running = False
        self.audio_state.is_active = False

        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        if hasattr(self, "beat_thread") and self.beat_thread.is_alive():
            self.beat_thread.join(timeout=2.0)

        logger.info("Audio beat analysis stopped")

    def _audio_worker(self):
        """Audio processing worker thread"""
        logger.info("Audio processing thread started")

        while self.running:
            try:
                # Process audio with BeatNet
                beats = self.beatnet.process()

                if beats is not None and len(beats) > 0:
                    current_time = time.time()

                    # Process each detected beat
                    for beat_timestamp, downbeat_prob in beats:
                        self.beat_queue.put((beat_timestamp, downbeat_prob, current_time))

                time.sleep(0.01)  # Small delay to prevent CPU overload

            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                if not self.running:
                    break
                time.sleep(0.1)  # Brief pause on error

        logger.info("Audio processing thread stopped")

    def _beat_worker(self):
        """Beat event processing worker thread"""
        logger.info("Beat event processing thread started")

        while self.running:
            try:
                # Get beat data from queue
                beat_data = self.beat_queue.get(timeout=0.1)
                beat_timestamp, downbeat_prob, system_time = beat_data

                # Process beat event
                self._process_beat_event(beat_timestamp, downbeat_prob, system_time)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Beat processing error: {e}")

        logger.info("Beat event processing thread stopped")

    def _process_beat_event(self, beat_timestamp: float, downbeat_prob: float, system_time: float):
        """Process a detected beat event"""
        # Check if this is a new beat (avoid duplicates)
        if beat_timestamp <= self.audio_state.last_beat_time:
            return

        # Update beat count
        self.audio_state.beat_count += 1
        self.audio_state.last_beat_time = beat_timestamp

        # Determine if this is a downbeat
        is_downbeat = downbeat_prob > 0.5
        if is_downbeat:
            self.audio_state.downbeat_count += 1
            self.audio_state.last_downbeat_time = beat_timestamp

        # Update BPM calculation
        bpm, confidence = self.bpm_calculator.update_beat(beat_timestamp)
        self.audio_state.current_bpm = bpm
        self.audio_state.confidence = confidence

        # Analyze beat intensity (simplified for now)
        intensity = self.intensity_analyzer.analyze_intensity(np.array([]), beat_timestamp)

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

        # Log beat event (for debugging)
        logger.debug(
            f"Beat #{beat_event.beat_count}: "
            f"BPM={beat_event.bpm:.1f}, "
            f"Downbeat={beat_event.is_downbeat}, "
            f"Intensity={beat_event.intensity:.2f}"
        )

        # Call beat callback if provided
        if self.beat_callback:
            try:
                self.beat_callback(beat_event)
            except Exception as e:
                logger.error(f"Beat callback error: {e}")

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
