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
    last_beat_wallclock_time: float = 0.0  # Wall-clock time of last beat for effects
    last_downbeat_time: float = 0.0
    last_downbeat_wallclock_time: float = 0.0  # Wall-clock time of last downbeat for timing
    beat_count: int = 0
    downbeat_count: int = 0
    beats_per_measure: int = 4
    confidence: float = 0.0
    beat_intensity: float = 0.0  # Last beat's accumulated bass energy intensity (0.0-1.0)
    beat_intensity_ready: bool = True  # False while accumulating bass energy for current beat
    current_rms: float = 0.0  # Continuous RMS audio level (EWMA smoothed)
    # Build-up/drop detection state
    buildup_state: str = "NORMAL"  # NORMAL or BUILDUP
    buildup_intensity: float = 0.0  # Continuous build-up progression (can exceed 1.0)
    last_cut_time: float = 0.0  # Wall-clock time of last cut event
    last_drop_time: float = 0.0  # Wall-clock time of last drop event


@dataclass
class BuildDropEvent:
    """Data structure for build-up/drop events"""

    timestamp: float  # Event time in seconds
    system_time: float  # System time when detected
    event_type: str  # "BUILDUP_UPDATE", "CUT", "DROP"
    buildup_intensity: float  # Continuous build-up progression (can exceed 1.0)
    bass_energy: float  # Current bass energy level
    high_energy: float  # Current high-frequency energy level
    confidence: float  # Detection confidence (0.0-1.0)
    is_cut: bool = False  # True on cut frames
    is_drop: bool = False  # True on drop frames


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


@dataclass
class BuildDropConfig:
    """Configuration for build-up/drop detection using spectral analysis"""

    # Frequency band definitions (Hz)
    bass_range: tuple[float, float] = (20.0, 250.0)  # Kick drum, bass
    mid_range: tuple[float, float] = (250.0, 2000.0)  # Snare, vocals, melodic
    high_range: tuple[float, float] = (2000.0, 8000.0)  # Cymbals, general hi-freq
    air_range: tuple[float, float] = (8000.0, 16000.0)  # Hi-hats, transient clicks

    # Snare roll detection bands
    snare_body_range: tuple[float, float] = (150.0, 400.0)  # Snare body
    snare_crack_range: tuple[float, float] = (2000.0, 5000.0)  # Snare rattle/crack

    # EWMA half-lives (in frames at ~86 fps)
    # Centroid slope: 1s input EWMA -> 0.25s diff -> 2s output EWMA
    ewma_centroid_input: int = 86  # 1s - applied to raw centroid before diff
    # Flux slopes: 0.5s input EWMA -> 0.25s diff -> 1s output EWMA
    ewma_flux_input: int = 43  # 0.5s - applied to raw flux before diff
    deriv_interval: int = 22  # 0.25s interval for slope calculation
    ewma_centroid_output: int = 172  # 2s - smoothing applied to centroid slope
    ewma_flux_output: int = 86  # 1s - smoothing applied to flux slopes

    ewma_mid_energy_short: int = 22  # 0.25s for current mid energy
    ewma_mid_energy_long: int = 344  # 4s for baseline mid energy

    # Thresholds
    cut_threshold: float = 0.5  # Mid energy ratio for cut detection
    drop_bass_slope_threshold: float = 20.0  # Bass flux slope for drop
    snare_magnitude_threshold: float = 0.3  # Snare autocorrelation peak
    snare_multiplier_minimum: int = 4  # 4x or 8x BPM
    flux_slope_entry_threshold: float = 5.0  # High/air slope for buildup entry

    # Cooldowns (in frames)
    cut_cooldown_frames: int = 172  # ~2s between cuts
    max_buildup_decrease_frames: int = 86  # ~1s dips allowed in buildup

    # Performance optimization
    snare_detection_interval: int = 1  # Run snare detection every N frames (1 = every frame)
    snare_window_size: int = 86  # ~1 second window for autocorrelation

    # FFT parameters
    fft_size: int = 2048


class BuildDropDetector:
    """
    Detects house/trance build-up and drop patterns in real-time audio.

    Uses spectral analysis to detect:
    - Build-up phases: snare rolls + rising high/air flux + rising spectral centroid
    - Cut events: sudden drop in mid energy (ends buildup)
    - Drop events: bass flux spike after buildup or within 2 bars of cut

    Outputs:
    - buildup_intensity: Continuous value during buildup (can exceed 1.0)
    - is_cut: True on frames where cut is detected
    - is_drop: True on frames where drop is detected
    """

    # Indices for the slopes array: [centroid, bass_flux, mid_flux, high_flux, air_flux]
    IDX_CENTROID = 0
    IDX_BASS_FLUX = 1
    IDX_MID_FLUX = 2
    IDX_HIGH_FLUX = 3
    IDX_AIR_FLUX = 4
    NUM_SLOPE_SIGNALS = 5

    # Class-level type annotations for Optional attributes
    prev_spectrum: Optional[np.ndarray]

    def __init__(self, config: Optional[BuildDropConfig] = None, sample_rate: int = 44100, buf_size: int = 512):
        """
        Initialize build-up/drop detector.

        Args:
            config: Detection configuration (uses defaults if None)
            sample_rate: Audio sample rate
            buf_size: Audio buffer size (hop size)
        """
        self.config = config if config is not None else BuildDropConfig()
        self.sample_rate = sample_rate
        self.buf_size = buf_size
        self.frame_rate = sample_rate / buf_size  # ~86 fps at 44100/512

        # Calculate EWMA alphas from half-lives
        # Centroid: 1s input -> 2s output
        self.alpha_centroid_input = 1 - np.exp(-np.log(2) / self.config.ewma_centroid_input)
        self.alpha_centroid_output = 1 - np.exp(-np.log(2) / self.config.ewma_centroid_output)
        # Flux: 0.5s input -> 1s output
        self.alpha_flux_input = 1 - np.exp(-np.log(2) / self.config.ewma_flux_input)
        self.alpha_flux_output = 1 - np.exp(-np.log(2) / self.config.ewma_flux_output)
        # Mid energy
        self.alpha_mid_short = 1 - np.exp(-np.log(2) / self.config.ewma_mid_energy_short)
        self.alpha_mid_long = 1 - np.exp(-np.log(2) / self.config.ewma_mid_energy_long)

        # Frequency bin ranges
        self._calculate_frequency_bins()

        # Circular buffers for slope calculation (need deriv_interval + 1 frames)
        self.slope_buffer_size = self.config.deriv_interval + 1
        # Separate buffers for centroid (1 value) and flux (4 values)
        self.centroid_input_buffer = np.zeros(self.slope_buffer_size)
        self.flux_input_buffer = np.zeros((self.slope_buffer_size, 4))  # bass, mid, high, air
        self.slope_buffer_idx = 0
        self.slope_buffer_filled = False

        # EWMA state - separate for centroid and flux
        self.ewma_centroid_input = 0.0  # Centroid input EWMA (1s)
        self.ewma_centroid_output = 0.0  # Centroid slope output EWMA (2s)
        self.ewma_flux_input = np.zeros(4)  # Flux input EWMAs (0.5s) [bass, mid, high, air]
        self.ewma_flux_output = np.zeros(4)  # Flux slope output EWMAs (1s)
        self.ewma_mid_short = 0.0  # Short-term mid energy EWMA
        self.ewma_mid_long = 0.0  # Long-term mid energy EWMA

        # Raw intensity smoothing (0.5s EWMA for smooth buildup curve)
        self.ewma_raw_intensity = 0.0
        self.alpha_raw_intensity = 1 - np.exp(-np.log(2) / 43)  # 0.5s half-life

        # Snare roll detection state
        self.snare_flux_buffer: deque[float] = deque(maxlen=self.config.snare_window_size)
        self.snare_roll_multiplier = 0  # 0, 2, 4, or 8
        self.snare_roll_magnitude = 0.0

        # Buildup state
        self.in_buildup = False
        self.buildup_start_frame = 0
        self.buildup_start_offset = 0.0
        self.buildup_intensity = 0.0
        self.peak_smoothed_intensity = 0.0
        self.frames_since_peak = 0

        # Cut/drop state
        self.last_cut_frame = -1000
        self.last_drop_frame = -1000

        # Frame counter
        self.frame_count = 0
        self.first_active_frame = -1  # Frame when audio becomes non-silent

        # Current BPM (updated externally)
        self.current_bpm = 120.0

        # Previous spectrum for flux calculation
        self.prev_spectrum = None

        # Current energies (for output)
        self.current_bass_energy = 0.0
        self.current_high_energy = 0.0

        # Initialize aubio phase vocoder for spectrum computation
        # This gives us overlapping windows (fft_size window, buf_size hop)
        self.pvoc = None
        if AUBIO_AVAILABLE:
            try:
                self.pvoc = aubio.pvoc(self.config.fft_size, self.buf_size)
                logger.info(
                    f"BuildDropDetector using aubio pvoc (fft_size={self.config.fft_size}, "
                    f"hop_size={self.buf_size})"
                )
            except Exception as e:
                logger.warning(f"Failed to create aubio pvoc: {e}, falling back to numpy FFT")
                self.pvoc = None

        logger.info(
            f"BuildDropDetector initialized (sample_rate={sample_rate}, "
            f"buf_size={buf_size}, frame_rate={self.frame_rate:.1f} fps)"
        )

    def _calculate_frequency_bins(self):
        """Calculate FFT bin indices for each frequency band"""
        freq_resolution = self.sample_rate / self.config.fft_size

        def freq_to_bin(freq: float) -> int:
            return int(freq / freq_resolution)

        self.bass_bins = (freq_to_bin(self.config.bass_range[0]), freq_to_bin(self.config.bass_range[1]))
        self.mid_bins = (freq_to_bin(self.config.mid_range[0]), freq_to_bin(self.config.mid_range[1]))
        self.high_bins = (freq_to_bin(self.config.high_range[0]), freq_to_bin(self.config.high_range[1]))
        self.air_bins = (freq_to_bin(self.config.air_range[0]), freq_to_bin(self.config.air_range[1]))
        self.snare_body_bins = (
            freq_to_bin(self.config.snare_body_range[0]),
            freq_to_bin(self.config.snare_body_range[1]),
        )
        self.snare_crack_bins = (
            freq_to_bin(self.config.snare_crack_range[0]),
            freq_to_bin(self.config.snare_crack_range[1]),
        )

    def _extract_band_energy(self, spectrum: np.ndarray, bin_range: tuple[int, int]) -> float:
        """Extract energy (sum of squared magnitudes) from a frequency band."""
        start_bin, end_bin = bin_range
        start_bin = max(0, start_bin)
        end_bin = min(len(spectrum), end_bin)
        if start_bin >= end_bin:
            return 0.0
        return float(np.sum(spectrum[start_bin:end_bin] ** 2))

    def _extract_band_flux(self, curr: np.ndarray, prev: np.ndarray, bin_range: tuple[int, int]) -> float:
        """Extract spectral flux (half-wave rectified difference) from a frequency band."""
        start_bin, end_bin = bin_range
        start_bin = max(0, start_bin)
        end_bin = min(len(curr), end_bin)
        if start_bin >= end_bin:
            return 0.0
        diff = curr[start_bin:end_bin] - prev[start_bin:end_bin]
        return float(np.sum(np.maximum(0, diff)))

    def set_bpm(self, bpm: float):
        """Update current BPM (called from AudioBeatAnalyzer)."""
        if bpm > 0:
            self.current_bpm = bpm

    def process_frame(self, audio_buffer: np.ndarray) -> dict:
        """
        Process audio frame and detect build-up/drop patterns.

        Args:
            audio_buffer: Audio samples (float32, mono)

        Returns:
            Dictionary with detection results:
                - buildup_intensity: Continuous buildup intensity
                - bass_energy: Current bass energy
                - high_energy: Current high-frequency energy
                - is_cut: True if cut detected this frame
                - is_drop: True if drop detected this frame
                - confidence: Detection confidence
        """
        # Ensure buffer is correct type and shape
        if audio_buffer.ndim > 1:
            audio_buffer = audio_buffer.flatten()
        audio_buffer = audio_buffer.astype(np.float32)

        # Pad or truncate to expected size
        if len(audio_buffer) < self.buf_size:
            audio_buffer = np.pad(audio_buffer, (0, self.buf_size - len(audio_buffer)))
        elif len(audio_buffer) > self.buf_size:
            audio_buffer = audio_buffer[: self.buf_size]

        self.frame_count += 1

        # Check for first active frame (non-silent)
        if self.first_active_frame < 0:
            rms = np.sqrt(np.mean(audio_buffer**2))
            if rms > 0.01:
                # Skip warmup frames after first detecting audio
                self.first_active_frame = self.frame_count + 86  # ~1s warmup

        # Compute FFT spectrum using aubio pvoc (preferred) or numpy fallback
        if self.pvoc is not None:
            # Use aubio phase vocoder - provides overlapping windows internally
            # pvoc maintains state and gives us fft_size window with buf_size hop
            cvec = self.pvoc(audio_buffer)
            spectrum = np.array(cvec.norm)
        else:
            # Fallback: numpy FFT with zero-padding (less accurate for flux)
            windowed = audio_buffer * np.hanning(len(audio_buffer))
            padded = np.zeros(self.config.fft_size)
            padded[: len(windowed)] = windowed
            spectrum = np.abs(np.fft.rfft(padded))

        # Extract band energies
        bass_energy = self._extract_band_energy(spectrum, self.bass_bins)
        mid_energy = self._extract_band_energy(spectrum, self.mid_bins)
        high_energy = self._extract_band_energy(spectrum, self.high_bins)
        air_energy = self._extract_band_energy(spectrum, self.air_bins)

        self.current_bass_energy = bass_energy
        self.current_high_energy = high_energy + air_energy

        # Extract spectral flux (requires previous spectrum)
        bass_flux = 0.0
        mid_flux = 0.0
        high_flux = 0.0
        air_flux = 0.0
        snare_body_flux = 0.0
        snare_crack_flux = 0.0

        if self.prev_spectrum is not None:
            bass_flux = self._extract_band_flux(spectrum, self.prev_spectrum, self.bass_bins)
            mid_flux = self._extract_band_flux(spectrum, self.prev_spectrum, self.mid_bins)
            high_flux = self._extract_band_flux(spectrum, self.prev_spectrum, self.high_bins)
            air_flux = self._extract_band_flux(spectrum, self.prev_spectrum, self.air_bins)
            snare_body_flux = self._extract_band_flux(spectrum, self.prev_spectrum, self.snare_body_bins)
            snare_crack_flux = self._extract_band_flux(spectrum, self.prev_spectrum, self.snare_crack_bins)

        self.prev_spectrum = spectrum.copy()

        # Compute spectral centroid
        freqs = np.fft.rfftfreq(self.config.fft_size, 1.0 / self.sample_rate)
        total_magnitude = np.sum(spectrum)
        if total_magnitude > 0:
            spectral_centroid = np.sum(freqs * spectrum) / total_magnitude
        else:
            spectral_centroid = 0.0

        # Skip processing until first active frame + warmup
        # first_active_frame is -1 until audio is detected, then set to frame + 86
        if self.first_active_frame < 0 or self.frame_count < self.first_active_frame:
            return {
                "buildup_intensity": 0.0,
                "bass_energy": bass_energy,
                "high_energy": self.current_high_energy,
                "is_cut": False,
                "is_drop": False,
                "confidence": 0.0,
                # Extended signals (zeros during warmup)
                "mid_energy": mid_energy,
                "air_energy": air_energy,
                "spectral_centroid": spectral_centroid,
                "bass_flux": bass_flux,
                "mid_flux": mid_flux,
                "high_flux": high_flux,
                "air_flux": air_flux,
                "ewma_slope_input": np.zeros(5),  # [centroid, bass, mid, high, air]
                "ewma_slope_output": np.zeros(5),  # slopes
                "ewma_mid_short": 0.0,
                "ewma_mid_long": 0.0,
                "mid_energy_ratio": 1.0,
                "snare_roll_multiplier": 0,
                "snare_roll_magnitude": 0.0,
                "in_buildup": False,
            }

        # Initialize EWMAs on first active frame
        if self.frame_count == self.first_active_frame:
            # Centroid starts at current value
            self.ewma_centroid_input = spectral_centroid
            self.ewma_centroid_output = 0.0
            # Flux starts at current values
            self.ewma_flux_input = np.array([bass_flux, mid_flux, high_flux, air_flux])
            self.ewma_flux_output = np.zeros(4)
            # Mid energy - initialize both to current mid energy
            self.ewma_mid_short = mid_energy
            self.ewma_mid_long = mid_energy
            # Fill buffers with initial values
            for i in range(self.slope_buffer_size):
                self.centroid_input_buffer[i] = self.ewma_centroid_input
                self.flux_input_buffer[i] = self.ewma_flux_input.copy()
            self.slope_buffer_filled = True
            logger.info(f"BuildDropDetector: First active frame {self.frame_count}")

        # Update centroid EWMA (1s half-life input)
        self.ewma_centroid_input = (
            self.alpha_centroid_input * spectral_centroid + (1 - self.alpha_centroid_input) * self.ewma_centroid_input
        )

        # Update flux EWMAs (0.5s half-life input)
        raw_flux = np.array([bass_flux, mid_flux, high_flux, air_flux])
        self.ewma_flux_input = self.alpha_flux_input * raw_flux + (1 - self.alpha_flux_input) * self.ewma_flux_input

        # Store in circular buffers for derivative calculation
        self.centroid_input_buffer[self.slope_buffer_idx] = self.ewma_centroid_input
        self.flux_input_buffer[self.slope_buffer_idx] = self.ewma_flux_input.copy()
        prev_idx = (self.slope_buffer_idx - self.config.deriv_interval) % self.slope_buffer_size
        self.slope_buffer_idx = (self.slope_buffer_idx + 1) % self.slope_buffer_size

        # Compute derivatives (current - past) / interval_seconds
        interval_seconds = self.config.deriv_interval * self.buf_size / self.sample_rate
        centroid_slope_raw = (self.ewma_centroid_input - self.centroid_input_buffer[prev_idx]) / interval_seconds
        flux_slope_raw = (self.ewma_flux_input - self.flux_input_buffer[prev_idx]) / interval_seconds

        # Apply output EWMA smoothing to slopes
        # Centroid: 2s half-life output
        self.ewma_centroid_output = (
            self.alpha_centroid_output * centroid_slope_raw
            + (1 - self.alpha_centroid_output) * self.ewma_centroid_output
        )
        # Flux: 1s half-life output
        self.ewma_flux_output = (
            self.alpha_flux_output * flux_slope_raw + (1 - self.alpha_flux_output) * self.ewma_flux_output
        )

        # Update mid energy EWMAs
        self.ewma_mid_short = self.alpha_mid_short * mid_energy + (1 - self.alpha_mid_short) * self.ewma_mid_short
        self.ewma_mid_long = self.alpha_mid_long * mid_energy + (1 - self.alpha_mid_long) * self.ewma_mid_long

        # Mid energy ratio for cut detection
        mid_energy_ratio = self.ewma_mid_short / (self.ewma_mid_long + 1e-10)

        # Snare roll detection (every N frames for performance)
        self.snare_flux_buffer.append(snare_body_flux * snare_crack_flux)
        if self.frame_count % self.config.snare_detection_interval == 0:
            self._detect_snare_roll()

        # Get slopes for detection logic
        centroid_slope = self.ewma_centroid_output
        bass_slope = self.ewma_flux_output[0]  # bass
        mid_slope = self.ewma_flux_output[1]  # mid (not used in detection, but for output)
        high_slope = self.ewma_flux_output[2]  # high
        air_slope = self.ewma_flux_output[3]  # air

        # Detection logic
        is_cut = False
        is_drop = False

        # Calculate drop window based on current BPM (2 bars = 8 beats)
        if self.current_bpm > 0:
            beats_per_second = self.current_bpm / 60.0
            two_bars_seconds = 8.0 / beats_per_second
            drop_window_frames = int(two_bars_seconds * self.frame_rate)
        else:
            drop_window_frames = int(4.0 * self.frame_rate)  # Default ~4s

        # Check for CUT: mid energy drops below threshold of long-term average
        cut_debounce_ok = (self.frame_count - self.last_cut_frame) > self.config.cut_cooldown_frames
        if mid_energy_ratio < self.config.cut_threshold and cut_debounce_ok and self.in_buildup:
            is_cut = True
            self.last_cut_frame = self.frame_count
            self.in_buildup = False
            logger.info(f"🎵 CUT at frame {self.frame_count} - mid energy ratio {mid_energy_ratio:.2f}")

        # Check for DROP: bass flux slope > threshold
        in_drop_window = (self.frame_count - self.last_cut_frame) <= drop_window_frames
        if bass_slope > self.config.drop_bass_slope_threshold:
            if self.in_buildup:
                is_drop = True
                self.last_drop_frame = self.frame_count
                self.in_buildup = False
                logger.info(f"🎵🎵🎵 DROP at frame {self.frame_count} - bass return during buildup")
            elif in_drop_window and self.last_cut_frame >= 0:
                is_drop = True
                self.last_drop_frame = self.frame_count
                logger.info(f"🎵🎵🎵 DROP at frame {self.frame_count} - bass return after cut")
                self.last_cut_frame = -1000  # Reset to prevent multiple drops

        # Buildup entry: snare roll + rising high/air flux
        can_start_buildup = (self.frame_count - self.last_cut_frame) > self.config.cut_cooldown_frames
        if not self.in_buildup and can_start_buildup:
            snare_entry = (
                self.snare_roll_magnitude > self.config.snare_magnitude_threshold
                and self.snare_roll_multiplier >= self.config.snare_multiplier_minimum
            )
            slope_entry = (
                high_slope > self.config.flux_slope_entry_threshold
                or air_slope > self.config.flux_slope_entry_threshold
            )

            if snare_entry and slope_entry:
                self.in_buildup = True
                self.buildup_start_frame = self.frame_count
                self.buildup_start_offset = self.ewma_raw_intensity  # Use smoothed value
                self.peak_smoothed_intensity = 0.0
                self.frames_since_peak = 0
                logger.info(f"🎵 Buildup started at frame {self.frame_count}")

        # Compute and smooth raw intensity (always, for continuous EWMA)
        raw_intensity = self._compute_raw_intensity(high_slope, air_slope, centroid_slope)
        self.ewma_raw_intensity = (
            self.alpha_raw_intensity * raw_intensity + (1 - self.alpha_raw_intensity) * self.ewma_raw_intensity
        )

        # Update buildup intensity using smoothed value
        if self.in_buildup:
            current_intensity = self.ewma_raw_intensity - self.buildup_start_offset

            # Check for sustained decrease (exit condition)
            if current_intensity < self.peak_smoothed_intensity - 0.05:
                self.frames_since_peak += 1
                if self.frames_since_peak > self.config.max_buildup_decrease_frames:
                    self.in_buildup = False
                    self.buildup_intensity = 0.0
                    logger.info(f"🎵 Buildup ended (intensity decreased) at frame {self.frame_count}")
            else:
                if current_intensity > self.peak_smoothed_intensity:
                    self.peak_smoothed_intensity = current_intensity
                self.frames_since_peak = 0
                self.buildup_intensity = max(0, current_intensity)
        else:
            self.buildup_intensity = 0.0

        return {
            "buildup_intensity": self.buildup_intensity,
            "bass_energy": bass_energy,
            "high_energy": self.current_high_energy,
            "is_cut": is_cut,
            "is_drop": is_drop,
            "confidence": 0.8 if self.in_buildup else 0.5,
            # Extended signals for visualization/debugging
            "mid_energy": mid_energy,
            "air_energy": air_energy,
            "spectral_centroid": spectral_centroid,
            "bass_flux": bass_flux,
            "mid_flux": mid_flux,
            "high_flux": high_flux,
            "air_flux": air_flux,
            "ewma_slope_input": np.array(
                [self.ewma_centroid_input, *self.ewma_flux_input]
            ),  # [centroid, bass, mid, high, air]
            "ewma_slope_output": np.array(
                [self.ewma_centroid_output, *self.ewma_flux_output]
            ),  # slopes [centroid, bass, mid, high, air]
            "ewma_mid_short": self.ewma_mid_short,
            "ewma_mid_long": self.ewma_mid_long,
            "mid_energy_ratio": mid_energy_ratio,
            "snare_roll_multiplier": self.snare_roll_multiplier,
            "snare_roll_magnitude": self.snare_roll_magnitude,
            "in_buildup": self.in_buildup,
        }

    def _compute_raw_intensity(self, high_slope: float, air_slope: float, centroid_slope: float) -> float:
        """Compute raw combined intensity metric for buildup detection."""
        # Use max of high and air flux slopes
        high_air_slope = max(high_slope, air_slope)

        # Snare component
        snare_intensity = 0.0
        if self.snare_roll_multiplier >= 4:
            snare_intensity = self.snare_roll_magnitude

        # Combined raw intensity
        raw_intensity = (
            max(0, high_air_slope) / 20.0  # high/air slope contribution
            + max(0, centroid_slope) / 1000.0  # centroid slope contribution
            + snare_intensity * 0.5  # snare contribution
        )
        return raw_intensity

    def _detect_snare_roll(self):
        """Detect snare roll using autocorrelation of snare flux product."""
        if len(self.snare_flux_buffer) < self.config.snare_window_size:
            return

        # Normalize the buffer
        window = np.array(self.snare_flux_buffer)
        max_val = np.max(window)
        if max_val > 0:
            window = window / max_val

        window_mean = np.mean(window)
        window_centered = window - window_mean
        window_std = np.std(window_centered)

        if window_std <= 0:
            self.snare_roll_multiplier = 0
            self.snare_roll_magnitude = 0.0
            return

        window_norm = window_centered / window_std

        # Compute autocorrelation
        autocorr = np.correlate(window_norm, window_norm, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]  # Take positive lags
        autocorr = autocorr / len(window)

        # Calculate lag values for 2x, 4x, 8x BPM
        if self.current_bpm <= 0:
            self.snare_roll_multiplier = 0
            self.snare_roll_magnitude = 0.0
            return

        beat_period_frames = self.frame_rate * 60.0 / self.current_bpm

        # Check for peaks at 8x, 4x, 2x (fastest first)
        multipliers = [8, 4, 2]
        detected_multiplier = 0
        detected_magnitude = 0.0

        for mult in multipliers:
            lag = beat_period_frames / mult
            lag_int = int(round(lag))

            if lag_int < 2 or lag_int >= len(autocorr) - 1:
                continue

            val = autocorr[lag_int]
            val_before = autocorr[lag_int - 1]
            val_after = autocorr[lag_int + 1]

            if val > val_before and val > val_after and val > 0.1:
                detected_multiplier = mult
                detected_magnitude = val
                break

        self.snare_roll_multiplier = detected_multiplier
        self.snare_roll_magnitude = detected_magnitude

    def get_state(self) -> str:
        """Get current state (NORMAL or BUILDUP for compatibility)."""
        return "BUILDUP" if self.in_buildup else "NORMAL"

    def get_buildup_intensity(self) -> float:
        """Get current build-up intensity."""
        return self.buildup_intensity


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

    # Class-level type annotations for Optional attributes
    builddrop_detector: Optional[BuildDropDetector]

    def __init__(
        self,
        beat_callback: Optional[Callable[[BeatEvent], None]] = None,
        builddrop_callback: Optional[Callable[[BuildDropEvent], None]] = None,
        model: int = 1,
        device: str = "auto",
        sample_rate: int = 22050,
        buffer_size: int = 1024,
        audio_config: Optional[AudioConfig] = None,
        enable_builddrop_detection: bool = False,
        builddrop_config: Optional[BuildDropConfig] = None,
    ):
        """
        Initialize audio beat analyzer.

        Args:
            beat_callback: Callback function for beat events
            builddrop_callback: Callback function for build-up/drop events
            model: BeatNet model selection (1-3)
            device: Processing device ('auto', 'cuda', 'cpu')
            sample_rate: Audio sample rate
            buffer_size: Audio buffer size
            audio_config: Optional AudioConfig for custom audio capture settings (file mode, etc)
            enable_builddrop_detection: Enable build-up/drop detection (default: False)
            builddrop_config: Configuration for build-up/drop detection
        """
        self.beat_callback = beat_callback
        self.builddrop_callback = builddrop_callback
        self.sample_rate = sample_rate  # Aubio can work with 44100Hz directly
        self.capture_rate = 44100  # USB device native rate
        self.buffer_size = buffer_size
        self.hop_size = 512  # Aubio hop size for processing (~11.6ms at 44.1kHz)
        self.chunk_size = 1024  # Audio chunk size for capture (~23ms at 44.1kHz)

        # Processing components
        self.bpm_calculator = BPMCalculator()
        self.intensity_analyzer = BeatIntensityAnalyzer(self.sample_rate, buffer_size)

        # Build-up/drop detection (optional)
        self.enable_builddrop_detection = enable_builddrop_detection
        if self.enable_builddrop_detection:
            self.builddrop_detector = BuildDropDetector(
                config=builddrop_config, sample_rate=self.capture_rate, buf_size=self.hop_size
            )
            logger.info("Build-up/drop detection enabled")
        else:
            self.builddrop_detector = None

        # Threading
        self.running = False
        self.beat_thread = None
        self.beat_queue: queue.Queue[Any] = queue.Queue(maxsize=100)

        # Thread pool for non-blocking beat callback execution
        self.callback_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="BeatCallback")

        # State tracking
        self.audio_state = AudioState()
        self.start_time = time.time()

        # Audio processing statistics
        self.total_frames_processed = 0
        self.total_beats_detected = 0
        self.last_status_log = 0.0
        self.last_beat_audio_timestamp = 0.0  # Track last beat in audio time, not wall clock

        # Bass energy accumulator for beat intensity calculation
        # Instead of using RMS of the onset frame, we accumulate bass energy over 3 frames
        # (onset frame + 2 subsequent frames) for more accurate beat intensity
        self._pending_beat_timestamp: Optional[float] = None  # Audio timestamp of pending beat
        self._pending_beat_wallclock: Optional[float] = None  # Wall-clock time of pending beat
        self._pending_beat_bpm: float = 120.0  # BPM at time of pending beat
        self._pending_beat_confidence: float = 0.8  # Confidence at time of pending beat
        self._bass_energy_accumulator: float = 0.0  # Accumulated bass energy
        self._bass_energy_frames_remaining: int = 0  # Frames left to accumulate (0 = no pending beat)
        self._bass_energy_frames_total: int = 3  # Total frames to accumulate (onset + 2)

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

    def __del__(self):
        """Cleanup when garbage collected - ensure threads are stopped."""
        try:
            if hasattr(self, "running") and self.running:
                self.stop_analysis()
        except Exception:
            pass  # Suppress errors during GC

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

            # Update continuous RMS level (EWMA with alpha=0.1 for smooth display)
            frame_rms = float(np.sqrt(np.mean(audio_frame**2)))
            self.audio_state.current_rms = 0.1 * frame_rms + 0.9 * self.audio_state.current_rms

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

            # Process build-up/drop detection if enabled
            builddrop_result = None  # Initialize to None for bass energy extraction
            if self.builddrop_detector is not None:
                # Update BPM in detector from current audio state
                if self.audio_state.current_bpm > 0:
                    self.builddrop_detector.set_bpm(self.audio_state.current_bpm)

                builddrop_result = self.builddrop_detector.process_frame(audio_frame)

                # Log periodic statistics (every 500 frames ~= 5.8s @ 512 samples/44.1kHz)
                if not hasattr(self, "_builddrop_log_counter"):
                    self._builddrop_log_counter = 0
                    self._builddrop_max_bass = 0.0
                    self._builddrop_max_high = 0.0
                    self._builddrop_max_intensity = 0.0
                    self._builddrop_cut_count = 0
                    self._builddrop_drop_count = 0

                self._builddrop_log_counter += 1
                self._builddrop_max_bass = max(self._builddrop_max_bass, builddrop_result["bass_energy"])
                self._builddrop_max_high = max(self._builddrop_max_high, builddrop_result["high_energy"])
                self._builddrop_max_intensity = max(
                    self._builddrop_max_intensity, builddrop_result["buildup_intensity"]
                )
                if builddrop_result.get("is_cut", False):
                    self._builddrop_cut_count += 1
                if builddrop_result.get("is_drop", False):
                    self._builddrop_drop_count += 1

                if self._builddrop_log_counter >= 500:
                    logger.info(
                        f"BuildDrop Stats (~5.8s): "
                        f"intensity={self._builddrop_max_intensity:.2f}, "
                        f"bass={self._builddrop_max_bass:.4f}, "
                        f"high={self._builddrop_max_high:.4f}, "
                        f"cuts={self._builddrop_cut_count}, drops={self._builddrop_drop_count}"
                    )
                    # Reset counters
                    self._builddrop_log_counter = 0
                    self._builddrop_max_bass = 0.0
                    self._builddrop_max_high = 0.0
                    self._builddrop_max_intensity = 0.0
                    self._builddrop_cut_count = 0
                    self._builddrop_drop_count = 0

                # Update audio state with build-up/drop info
                current_time = time.time()
                self.audio_state.buildup_state = "BUILDUP" if builddrop_result["buildup_intensity"] > 0 else "NORMAL"
                self.audio_state.buildup_intensity = builddrop_result["buildup_intensity"]

                # Update cut/drop times and log events
                is_cut = builddrop_result.get("is_cut", False)
                is_drop = builddrop_result.get("is_drop", False)
                if is_cut:
                    self.audio_state.last_cut_time = current_time
                    logger.info(
                        f"CUT detected: intensity={builddrop_result['buildup_intensity']:.2f}, "
                        f"bass={builddrop_result['bass_energy']:.4f}"
                    )
                if is_drop:
                    self.audio_state.last_drop_time = current_time
                    # The drop occurs ON a downbeat - set downbeat to last beat time
                    if self.audio_state.last_beat_wallclock_time > 0:
                        self.audio_state.last_downbeat_wallclock_time = self.audio_state.last_beat_wallclock_time
                        self.audio_state.last_downbeat_time = self.audio_state.last_beat_time
                        self.audio_state.downbeat_count += 1
                        logger.info(
                            f"🥁 DOWNBEAT: Drop detected - syncing to last beat "
                            f"(beat_time={self.audio_state.last_beat_wallclock_time:.3f})"
                        )
                    logger.info(
                        f"DROP detected: intensity={builddrop_result['buildup_intensity']:.2f}, "
                        f"bass={builddrop_result['bass_energy']:.4f}"
                    )

                # Trigger callback on cut, drop, or during buildup (for intensity updates)
                should_trigger_callback = is_cut or is_drop or builddrop_result["buildup_intensity"] > 0
                if self.builddrop_callback and should_trigger_callback:
                    # Calculate timestamp relative to audio start
                    audio_timestamp = (self.total_frames_processed * self.hop_size) / self.capture_rate

                    # Determine event type
                    if is_cut:
                        event_type = "CUT"
                    elif is_drop:
                        event_type = "DROP"
                    else:
                        event_type = "BUILDUP_UPDATE"

                    # Create build-drop event
                    builddrop_event = BuildDropEvent(
                        timestamp=audio_timestamp,
                        system_time=current_time,
                        event_type=event_type,
                        buildup_intensity=builddrop_result["buildup_intensity"],
                        bass_energy=builddrop_result["bass_energy"],
                        high_energy=builddrop_result["high_energy"],
                        confidence=builddrop_result["confidence"],
                        is_cut=is_cut,
                        is_drop=is_drop,
                    )

                    # Call callback asynchronously
                    try:
                        self.callback_executor.submit(self.builddrop_callback, builddrop_event)
                    except Exception as e:
                        logger.error(f"Build-drop callback submission error: {e}")

            # Get bass energy from builddrop_result if available (for beat intensity)
            # This provides spectral bass energy which is more accurate than RMS
            # Fall back to RMS-based energy if builddrop detection is not enabled
            current_bass_energy = 0.0
            use_bass_energy = False
            if self.builddrop_detector is not None and builddrop_result is not None:
                current_bass_energy = builddrop_result.get("bass_energy", 0.0)
                use_bass_energy = True
            else:
                # Fallback: use RMS energy when builddrop detection is not available
                # Scale RMS (typically 0-0.5) to similar range as bass energy (0-200)
                current_bass_energy = float(np.sqrt(np.mean(audio_frame**2))) * 400.0

            # === Bass energy accumulation for pending beat ===
            # Accumulate bass energy over 3 frames after beat detection for more accurate intensity
            if self._bass_energy_frames_remaining > 0:
                self._bass_energy_accumulator += current_bass_energy
                self._bass_energy_frames_remaining -= 1

                # Check if accumulation is complete
                if self._bass_energy_frames_remaining == 0:
                    # Compute final beat intensity from accumulated bass energy
                    # Normalize by number of frames and scale to [0, 1] range
                    # Bass energy values are typically in range 0-1000+ depending on audio level
                    accumulated_bass = self._bass_energy_accumulator / self._bass_energy_frames_total
                    # Scale to approximately [0, 1] - bass energy of ~100 maps to 0.5
                    beat_intensity = min(1.0, accumulated_bass / 200.0)

                    logger.debug(
                        f"Beat intensity finalized: accumulated_bass={self._bass_energy_accumulator:.2f}, "
                        f"avg={accumulated_bass:.2f}, intensity={beat_intensity:.3f}, "
                        f"use_bass_energy={use_bass_energy}"
                    )

                    # Send the completed beat to the queue
                    try:
                        self.beat_queue.put_nowait(
                            (self._pending_beat_timestamp, 0.1, self._pending_beat_wallclock, beat_intensity)
                        )
                        logger.debug(
                            f"Beat queued (with accumulated intensity): beat_count={self.total_beats_detected}, "
                            f"intensity={beat_intensity:.3f}"
                        )
                    except queue.Full:
                        logger.warning("Beat queue full - dropping beat event")

                    # Clear pending beat state
                    self._pending_beat_timestamp = None
                    self._pending_beat_wallclock = None

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

                    # Log beat detection timing (DEBUG level for latency tracking)
                    logger.debug(
                        f"Beat detected: audio_time={audio_timestamp:.3f}s, "
                        f"wall_time={current_time:.3f}, "
                        f"detection_latency={detection_latency_ms:.1f}ms"
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
                        logger.debug(f"🎵 Aubio beat detected: BPM={current_bpm:.1f}, confidence={confidence:.3f}")

                    # Start bass energy accumulation instead of immediately sending beat
                    # Store pending beat info and start accumulating
                    self._pending_beat_timestamp = audio_timestamp
                    self._pending_beat_wallclock = current_time
                    self._pending_beat_bpm = current_bpm
                    self._pending_beat_confidence = confidence
                    self._bass_energy_accumulator = current_bass_energy  # Include onset frame
                    self._bass_energy_frames_remaining = self._bass_energy_frames_total - 1  # -1 for onset frame

                    # Update audio state to indicate beat detected but intensity pending
                    self.audio_state.beat_intensity_ready = False

                    logger.debug(
                        f"Beat pending: starting {self._bass_energy_frames_total}-frame bass energy accumulation, "
                        f"onset_bass={current_bass_energy:.2f}"
                    )

        # Periodic status logging
        current_time = time.time()
        status_log_interval = 10.0
        if current_time - self.last_status_log >= status_log_interval:
            current_audio_rms = np.sqrt(np.mean(audio_chunk * audio_chunk))

            logger.debug(
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
            downbeat_prob: Probability of downbeat (unused - we use timing-based detection)
            system_time: Wall-clock time when beat was detected
            intensity: Beat intensity from accumulated bass energy (0.0-1.0)
        """
        # Check if this is a new beat (avoid duplicates with tolerance)
        if abs(beat_timestamp - self.audio_state.last_beat_time) < 0.1:
            return

        # Update beat count
        self.audio_state.beat_count += 1
        self.audio_state.last_beat_time = beat_timestamp
        self.audio_state.last_beat_wallclock_time = system_time  # Store wall-clock time for effects

        # Timing-based downbeat detection:
        # (a) First beat detected
        # (b) Beat approximately 4 beats from last downbeat based on BPM
        # Note: Drop-synced downbeats are handled in _on_audio_chunk when drop is detected
        is_downbeat = False

        # (a) First beat ever
        if self.audio_state.downbeat_count == 0:
            is_downbeat = True
            logger.info("🥁 DOWNBEAT: First beat detected")

        # (b) Approximately 4 beats from last downbeat based on BPM
        if not is_downbeat and self.audio_state.current_bpm > 0:
            beat_duration = 60.0 / self.audio_state.current_bpm
            four_beat_duration = beat_duration * self.audio_state.beats_per_measure
            last_downbeat_wallclock = getattr(self.audio_state, "last_downbeat_wallclock_time", 0)

            if last_downbeat_wallclock > 0:
                time_since_downbeat = system_time - last_downbeat_wallclock
                # Check if we're close to a multiple of 4 beats (within 35% tolerance)
                # Higher tolerance accounts for BPM drift and beat detection jitter
                tolerance = beat_duration * 0.35
                remainder = time_since_downbeat % four_beat_duration
                # Check if remainder is close to 0 or close to four_beat_duration
                # and ensure at least 3 beats have passed to avoid false positives
                near_measure_boundary = remainder < tolerance or (four_beat_duration - remainder) < tolerance
                if near_measure_boundary and time_since_downbeat > beat_duration * 3:
                    is_downbeat = True
                    beats_elapsed = round(time_since_downbeat / beat_duration)
                    logger.debug(
                        f"🥁 DOWNBEAT: ~{beats_elapsed} beats since last "
                        f"({time_since_downbeat:.3f}s, 4-beat={four_beat_duration:.3f}s)"
                    )

        if is_downbeat:
            self.audio_state.downbeat_count += 1
            self.audio_state.last_downbeat_time = beat_timestamp
            self.audio_state.last_downbeat_wallclock_time = system_time

        # Update BPM calculation
        bpm, confidence = self.bpm_calculator.update_beat(beat_timestamp)
        self.audio_state.current_bpm = bpm
        self.audio_state.confidence = confidence
        self.audio_state.beat_intensity = intensity  # Store intensity for renderer access
        self.audio_state.beat_intensity_ready = True  # Mark intensity as ready (accumulation complete)

        # Note: intensity is now from accumulated bass energy over 3 frames, not single-frame RMS

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
        logger.info(
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

    def get_audio_stats(self) -> dict:
        """
        Get current audio statistics including level and AGC state.

        Returns:
            Dictionary with audio_level and agc_gain_db
        """
        stats = {
            "audio_level": self.audio_state.current_rms,  # Continuous RMS from aubio processing
            "agc_gain_db": 0.0,
        }

        # Get AGC gain if available
        if hasattr(self, "audio_capture") and self.audio_capture is not None:
            capture_stats = self.audio_capture.get_statistics()
            agc_stats = capture_stats.get("agc")
            if agc_stats:
                stats["agc_gain_db"] = agc_stats.get("current_gain_db", 0.0)

        return stats


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

    def builddrop_event_handler(builddrop_event: BuildDropEvent):
        """Example build-up/drop event handler"""
        # Create visual indicator based on event type
        indicators = {
            "NORMAL": "⚪",
            "BUILDUP": "🟡",
            "PREDROP": "🟠",
            "DROP": "🔴💥",
            "POSTDROP": "🟢",
        }
        indicator = indicators.get(builddrop_event.event_type, "⚪")

        print(
            f"\n{indicator} BUILD/DROP EVENT: {builddrop_event.event_type}\n"
            f"  Intensity: {builddrop_event.buildup_intensity:.2f}\n"
            f"  Bass Energy: {builddrop_event.bass_energy:.4f}\n"
            f"  High Energy: {builddrop_event.high_energy:.4f}\n"
            f"  Confidence: {builddrop_event.confidence:.2f}\n"
        )

        # Example: trigger different LED effects based on state
        if builddrop_event.event_type == "BUILDUP":
            # Gradually increase saturation/brightness based on intensity
            effect_intensity = builddrop_event.buildup_intensity
            print(f"  → Apply build-up effect at {effect_intensity*100:.0f}% intensity")

        elif builddrop_event.event_type == "DROP":
            # Trigger explosive visual effect
            print("  → 💥 TRIGGER DROP EFFECT - Maximum impact!")

        elif builddrop_event.event_type == "POSTDROP":
            # Maintain elevated bass reactivity
            print(f"  → Enhanced bass response (energy: {builddrop_event.bass_energy:.4f})")

    # Example 1: Basic beat detection only
    print("=" * 60)
    print("Example 1: Basic beat detection")
    print("=" * 60)

    analyzer_basic = AudioBeatAnalyzer(beat_callback=beat_event_handler)

    # Example 2: Beat detection + build-up/drop detection
    print("\n" + "=" * 60)
    print("Example 2: Beat detection + Build-up/Drop detection")
    print("=" * 60)

    # Create custom build-drop configuration
    custom_config = BuildDropConfig(
        flux_slope_entry_threshold=5.0,  # Sensitivity for buildup entry detection
        drop_bass_slope_threshold=20.0,  # Bass flux slope threshold for drop detection
        cut_threshold=0.5,  # Mid energy ratio for cut detection
    )

    analyzer_full = AudioBeatAnalyzer(
        beat_callback=beat_event_handler,
        builddrop_callback=builddrop_event_handler,
        enable_builddrop_detection=True,
        builddrop_config=custom_config,
    )

    # Choose which analyzer to run
    print("\nSelect mode:")
    print("1. Basic beat detection only")
    print("2. Beat detection + Build-up/Drop detection (recommended for house/trance)")
    choice = input("Enter choice (1 or 2, default=2): ").strip() or "2"

    analyzer = analyzer_full if choice == "2" else analyzer_basic

    try:
        analyzer.start_analysis()
        print("\n✓ Audio analyzer started. Press Ctrl+C to stop...\n")

        # Run for testing
        start_time = time.time()
        while time.time() - start_time < 60:  # Run for 60 seconds
            time.sleep(2)

            # Print current state
            state = analyzer.get_current_state()
            status_line = (
                f"State: BPM={state.current_bpm:.1f}, "
                f"Beats={state.beat_count}, "
                f"Intensity={state.beat_intensity:.2f}"
            )

            # Add build-up/drop state if enabled
            if analyzer.enable_builddrop_detection:
                status_line += f", Build/Drop={state.buildup_state}"
                if state.buildup_state == "BUILDUP":
                    status_line += f" ({state.buildup_intensity*100:.0f}%)"

            print(status_line)

    except KeyboardInterrupt:
        print("\n\nStopping audio analyzer...")

    finally:
        analyzer.stop_analysis()
        print("✓ Audio analyzer stopped")
