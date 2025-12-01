#!/usr/bin/env python3
"""
Audio Beat Analyzer Testing and Debugging Tool.

Standalone utility for testing beat detection functionality on audio files.
Processes an audio file and generates a timeline visualization showing:
- Audio waveform
- Detected beats
- BPM over time
- RMS envelope

Usage:
    python tools/audio_beat_analyzer_test.py [audio_file.wav] [--speed SPEED]

    If no audio file is provided, uses audio_capture_test.wav from project root.
"""

import argparse
import logging
import sys
import time
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consumer.audio_beat_analyzer import AudioBeatAnalyzer, BeatEvent, BuildDropEvent
from src.consumer.audio_capture import AudioConfig

logger = logging.getLogger(__name__)


class BeatAnalyzerTester:
    """Test harness for AudioBeatAnalyzer with file playback"""

    def __init__(self, audio_file: Path, playback_speed: float = 10.0):
        """
        Initialize beat analyzer tester.

        Args:
            audio_file: Path to WAV file to analyze
            playback_speed: Playback speed multiplier (default 10x for faster testing)
        """
        self.audio_file = audio_file
        self.playback_speed = playback_speed

        # Storage for beat events
        self.beat_events = []

        # Storage for build-up/drop events
        self.builddrop_events = []

        # Storage for per-frame energy data (computed directly from audio)
        self.energy_timeline = None  # Will be computed in _compute_energy_bands()

        self.audio_data = None
        self.sample_rate = None

        # Load audio file metadata
        self._load_audio_metadata()

        logger.info(f"Initialized BeatAnalyzerTester for {audio_file}")
        logger.info(f"  Duration: {self.duration:.1f}s")
        logger.info(f"  Sample rate: {self.sample_rate}Hz")
        logger.info(f"  Playback speed: {playback_speed}x")

    def _load_audio_metadata(self):
        """Load audio file metadata"""
        with wave.open(str(self.audio_file), "rb") as wav_file:
            self.sample_rate = wav_file.getframerate()
            self.channels = wav_file.getnchannels()
            self.total_frames = wav_file.getnframes()
            self.duration = self.total_frames / self.sample_rate

            # Read audio data for plotting
            frames_data = wav_file.readframes(self.total_frames)
            sample_width = wav_file.getsampwidth()

            if sample_width == 2:
                # 16-bit PCM
                audio_array = np.frombuffer(frames_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                # 32-bit PCM
                audio_array = np.frombuffer(frames_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            # Convert to mono if needed
            if self.channels > 1:
                audio_array = audio_array.reshape(-1, self.channels)
                audio_array = np.mean(audio_array, axis=1)

            self.audio_data = audio_array

    def _calculate_bpm_timeline_internal(self, timestamps: np.ndarray, hop_size: int):
        """
        Calculate BPM over time using beat events (internal version for use during energy computation).

        Args:
            timestamps: Frame timestamps array
            hop_size: FFT hop size in samples

        Returns:
            Tuple of (time_points, bpm_values)
        """
        if len(self.beat_events) < 2:
            # Return a default BPM of 120 if no beats detected yet
            return np.array([0.0, self.duration]), np.array([120.0, 120.0])

        window_size = 5.0  # seconds

        # Create time points every 0.5 seconds
        time_points = np.arange(0, self.duration, 0.5)
        bpm_values = np.zeros(len(time_points))

        for i, t in enumerate(time_points):
            # Find beats within window [t - window_size/2, t + window_size/2]
            window_beats = [
                b["timestamp"] for b in self.beat_events if t - window_size / 2 <= b["timestamp"] <= t + window_size / 2
            ]

            if len(window_beats) >= 2:
                # Calculate BPM from intervals
                intervals = np.diff(window_beats)
                if len(intervals) > 0:
                    avg_interval = np.mean(intervals)
                    if avg_interval > 0:
                        bpm_values[i] = 60.0 / avg_interval

        # Fill in zeros with interpolation or default
        if np.any(bpm_values > 0):
            # Find first and last non-zero values
            nonzero_idx = np.where(bpm_values > 0)[0]
            first_valid = nonzero_idx[0]
            last_valid = nonzero_idx[-1]

            # Fill leading zeros with first valid value
            bpm_values[:first_valid] = bpm_values[first_valid]
            # Fill trailing zeros with last valid value
            bpm_values[last_valid + 1 :] = bpm_values[last_valid]

            # Interpolate internal zeros
            for i in range(first_valid, last_valid):
                if bpm_values[i] == 0:
                    # Find next non-zero
                    for j in range(i + 1, last_valid + 1):
                        if bpm_values[j] > 0:
                            # Linear interpolate
                            bpm_values[i] = bpm_values[i - 1] + (bpm_values[j] - bpm_values[i - 1]) / (j - i + 1)
                            break
        else:
            # Default to 120 BPM if no valid beats
            bpm_values[:] = 120.0

        return time_points, bpm_values

    def _compute_energy_bands(self):
        """
        Compute frequency band energies for the entire audio file.

        Uses the same parameters as BuildDropDetector for consistency:
        - hop_size: 512 samples (~11.6ms at 44.1kHz)
        - fft_size: 2048 samples

        Returns dict with arrays for each band and timestamps.
        """
        logger.info("Computing frequency band energies...")

        hop_size = 512
        fft_size = 2048

        # Frequency band definitions (Hz) - matching BuildDropDetector
        bass_range = (20.0, 250.0)
        mid_range = (250.0, 2000.0)
        high_range = (2000.0, 8000.0)
        air_range = (8000.0, 16000.0)

        # Frequency resolution
        freq_resolution = self.sample_rate / fft_size

        def freq_to_bin(freq: float) -> int:
            return int(freq / freq_resolution)

        # Calculate bin ranges
        bass_bins = (freq_to_bin(bass_range[0]), freq_to_bin(bass_range[1]))
        mid_bins = (freq_to_bin(mid_range[0]), freq_to_bin(mid_range[1]))
        high_bins = (freq_to_bin(high_range[0]), freq_to_bin(high_range[1]))
        air_bins = (freq_to_bin(air_range[0]), freq_to_bin(air_range[1]))

        # Number of frames
        n_frames = (len(self.audio_data) - fft_size) // hop_size + 1

        # Allocate arrays
        timestamps = np.zeros(n_frames)
        bass_energy = np.zeros(n_frames)
        mid_energy = np.zeros(n_frames)
        high_energy = np.zeros(n_frames)
        air_energy = np.zeros(n_frames)
        spectral_centroid = np.zeros(n_frames)

        # Spectral flux arrays (onset detection)
        bass_flux = np.zeros(n_frames)
        mid_flux = np.zeros(n_frames)
        high_flux = np.zeros(n_frames)
        air_flux = np.zeros(n_frames)

        # Snare roll detection bands: 150-400 Hz (snare body) and 2-5kHz (snare rattle/crack)
        snare_body_range = (150.0, 400.0)
        snare_crack_range = (2000.0, 5000.0)
        snare_body_bins = (freq_to_bin(snare_body_range[0]), freq_to_bin(snare_body_range[1]))
        snare_crack_bins = (freq_to_bin(snare_crack_range[0]), freq_to_bin(snare_crack_range[1]))
        snare_body_flux = np.zeros(n_frames)
        snare_crack_flux = np.zeros(n_frames)

        # Frequency array for centroid calculation
        freqs = np.fft.rfftfreq(fft_size, 1.0 / self.sample_rate)

        # Previous spectrum for flux calculation
        prev_spectrum = None

        # Process each frame
        for i in range(n_frames):
            start = i * hop_size
            end = start + fft_size

            # Get frame and apply window
            frame = self.audio_data[start:end] * np.hanning(fft_size)

            # Compute FFT magnitude
            spectrum = np.abs(np.fft.rfft(frame))

            # Extract band energies (sum of squared magnitudes)
            def band_energy(bins):
                s, e = max(0, bins[0]), min(len(spectrum), bins[1])
                return np.sum(spectrum[s:e] ** 2) if s < e else 0.0

            # Extract spectral flux for a band: sum(max(0, current - previous))
            def band_flux(bins, curr, prev):
                s, e = max(0, bins[0]), min(len(curr), bins[1])
                if s >= e:
                    return 0.0
                diff = curr[s:e] - prev[s:e]
                return np.sum(np.maximum(0, diff))

            timestamps[i] = start / self.sample_rate
            bass_energy[i] = band_energy(bass_bins)
            mid_energy[i] = band_energy(mid_bins)
            high_energy[i] = band_energy(high_bins)
            air_energy[i] = band_energy(air_bins)

            # Compute spectral flux (half-wave rectified difference)
            if prev_spectrum is not None:
                bass_flux[i] = band_flux(bass_bins, spectrum, prev_spectrum)
                mid_flux[i] = band_flux(mid_bins, spectrum, prev_spectrum)
                high_flux[i] = band_flux(high_bins, spectrum, prev_spectrum)
                air_flux[i] = band_flux(air_bins, spectrum, prev_spectrum)
                # Snare roll detection bands
                snare_body_flux[i] = band_flux(snare_body_bins, spectrum, prev_spectrum)
                snare_crack_flux[i] = band_flux(snare_crack_bins, spectrum, prev_spectrum)

            prev_spectrum = spectrum.copy()

            # Compute spectral centroid (weighted mean frequency)
            total_magnitude = np.sum(spectrum)
            if total_magnitude > 0:
                spectral_centroid[i] = np.sum(freqs * spectrum) / total_magnitude
            else:
                spectral_centroid[i] = 0.0

        # Calculate trend indicators using EWMA
        # Slow EWMA: ~4 second half-life (344 frames) - used for derivative calculation
        # Longer half-life reduces noise in the derivative
        slow_half_life = 344
        alpha_slow = 1 - np.exp(-np.log(2) / slow_half_life)

        # Find first non-silent frame to initialize EWMAs
        # Silent frames have noise-dominated spectral centroid that biases the EWMA
        # We skip ahead by ~1 second (86 frames) after first detecting audio to ensure
        # we're in a stable region, not in a transition from silence
        rms_threshold = 0.01  # RMS threshold for "non-silent"
        warmup_frames = 86  # ~1 second at 86 fps - let the audio stabilize
        first_active_frame = 0
        for i in range(n_frames):
            start = i * hop_size
            end = start + fft_size
            frame_rms = np.sqrt(np.mean(self.audio_data[start:end] ** 2))
            if frame_rms > rms_threshold:
                first_active_frame = min(i + warmup_frames, n_frames - 1)
                break

        logger.info(
            f"  First active frame: {first_active_frame} ({timestamps[first_active_frame]:.2f}s) (after {warmup_frames} frame warmup)"
        )

        def compute_slow_ewma(data):
            """Compute slow EWMA for trend analysis.

            Silent frames (before first_active_frame) output zero.
            EWMA calculation only starts from first non-silent frame to avoid
            bias from noise-dominated values during silence.

            Returns ewma_slow_arr.
            """
            ewma_slow_arr = np.zeros(len(data))

            if first_active_frame >= len(data):
                return ewma_slow_arr

            # Initialize EWMA from first active frame
            ewma_slow = data[first_active_frame]

            # Only compute from first active frame onwards
            for i in range(first_active_frame, len(data)):
                ewma_slow = alpha_slow * data[i] + (1 - alpha_slow) * ewma_slow
                ewma_slow_arr[i] = ewma_slow

            return ewma_slow_arr

        # Compute multiple EWMAs for centroid visualization
        # Additional half-lives for comparison
        def compute_ewma_with_halflife(data, half_life_frames):
            """Compute EWMA with specified half-life."""
            alpha = 1 - np.exp(-np.log(2) / half_life_frames)
            ewma_arr = np.zeros(len(data))
            if first_active_frame >= len(data):
                return ewma_arr
            ewma = data[first_active_frame]
            for i in range(first_active_frame, len(data)):
                ewma = alpha * data[i] + (1 - alpha) * ewma
                ewma_arr[i] = ewma
            return ewma_arr

        # Half-lives: 0.5s=43, 1s=86, 2s=172, 4s=344, 8s=688 frames
        centroid_ewma_05s = compute_ewma_with_halflife(spectral_centroid, 43)
        centroid_ewma_1s = compute_ewma_with_halflife(spectral_centroid, 86)
        centroid_ewma_2s = compute_ewma_with_halflife(spectral_centroid, 172)
        centroid_ewma_slow = compute_slow_ewma(spectral_centroid)  # 4s (344 frames)
        centroid_ewma_8s = compute_ewma_with_halflife(spectral_centroid, 688)

        # Trend measure: Derivative of 1s EWMA centroid using 0.25s interval
        # Using 1s EWMA smooths out noise while preserving trend information
        # 0.25s interval (22 frames) provides additional smoothing vs frame-to-frame
        # Scale to Hz/s by dividing by interval duration
        deriv_interval = 22  # ~0.25s at 86 fps
        interval_seconds = deriv_interval * hop_size / self.sample_rate  # actual interval in seconds

        def compute_interval_derivative(ewma_arr, interval):
            """Compute derivative using interval-spaced samples, scaled to Hz/s."""
            deriv = np.zeros(len(ewma_arr))
            for i in range(first_active_frame + interval, len(ewma_arr)):
                # Difference over interval frames, scaled to Hz/s
                deriv[i] = (ewma_arr[i] - ewma_arr[i - interval]) / interval_seconds
            return deriv

        # Calculate slope from 1s EWMA centroid (0.25s interval difference)
        centroid_slope_raw = compute_interval_derivative(centroid_ewma_1s, deriv_interval)

        # Apply 2s and 4s EWMA smoothing to the slope
        centroid_slope_ewma_2s = compute_ewma_with_halflife(centroid_slope_raw, 172)  # 2s = 172 frames
        centroid_slope_ewma_4s = compute_ewma_with_halflife(centroid_slope_raw, 344)  # 4s = 344 frames

        logger.info(f"  Centroid slope: 0.25s interval diff of 1s EWMA, then 2s/4s EWMA smoothing")

        # Compute EWMAs of spectral flux for each band (0.5s and 1s half-lives)
        bass_flux_ewma_05s = compute_ewma_with_halflife(bass_flux, 43)
        bass_flux_ewma_1s = compute_ewma_with_halflife(bass_flux, 86)
        mid_flux_ewma_05s = compute_ewma_with_halflife(mid_flux, 43)
        mid_flux_ewma_1s = compute_ewma_with_halflife(mid_flux, 86)
        high_flux_ewma_05s = compute_ewma_with_halflife(high_flux, 43)
        high_flux_ewma_1s = compute_ewma_with_halflife(high_flux, 86)
        air_flux_ewma_05s = compute_ewma_with_halflife(air_flux, 43)
        air_flux_ewma_1s = compute_ewma_with_halflife(air_flux, 86)

        # Compute EWMAs of mid energy for cut detection
        # Use 0.25s half-life for smoothed current value (reduces noise)
        # Use 4s half-life as the "long-term average" baseline
        mid_energy_ewma_025s = compute_ewma_with_halflife(mid_energy, 22)  # 0.25s = 22 frames
        mid_energy_ewma_4s = compute_ewma_with_halflife(mid_energy, 344)  # 4s = 344 frames

        # Compute mid energy ratio: 0.25s EWMA / 4s EWMA baseline
        # Add small epsilon to avoid division by zero
        mid_energy_ratio = np.zeros(n_frames)
        for i in range(n_frames):
            if mid_energy_ewma_4s[i] > 1e-10:
                mid_energy_ratio[i] = mid_energy_ewma_025s[i] / mid_energy_ewma_4s[i]
            else:
                mid_energy_ratio[i] = 1.0  # Default to 1.0 if baseline is near zero

        logger.info(f"  Mid energy ratio: 0.25s EWMA / 4s EWMA baseline")

        # Compute flux slopes: 0.5s EWMA -> 0.25s interval diff -> 1s EWMA smoothing
        def compute_flux_slope(flux_data):
            """Compute slope of flux using same algorithm as centroid slope."""
            # 0.5s EWMA on raw flux (43 frames)
            flux_05s = compute_ewma_with_halflife(flux_data, 43)
            # 0.25s interval difference
            slope_raw = compute_interval_derivative(flux_05s, deriv_interval)
            # 1s EWMA smoothing (86 frames)
            slope_1s = compute_ewma_with_halflife(slope_raw, 86)
            return slope_1s

        bass_flux_slope = compute_flux_slope(bass_flux)
        mid_flux_slope = compute_flux_slope(mid_flux)
        high_flux_slope = compute_flux_slope(high_flux)
        air_flux_slope = compute_flux_slope(air_flux)

        # Snare roll detection
        # Compute product of snare body and crack flux
        snare_flux_product = snare_body_flux * snare_crack_flux

        # Normalize the product to prevent numerical issues
        max_product = np.max(snare_flux_product)
        if max_product > 0:
            snare_flux_product_norm = snare_flux_product / max_product
        else:
            snare_flux_product_norm = snare_flux_product

        # Compute snare roll detection using autocorrelation
        # Window size: 86 frames (~1 second at 86 fps)
        window_size = 86
        snare_roll_multiplier = np.zeros(n_frames)  # 0=none, 2=2x, 4=4x, 8=8x
        snare_roll_magnitude = np.zeros(n_frames)

        # Get BPM timeline for lag calculation
        bpm_time_points, bpm_values = self._calculate_bpm_timeline_internal(timestamps, hop_size)

        frame_rate = self.sample_rate / hop_size  # ~86 fps

        for i in range(window_size, n_frames):
            # Get the most recent window_size values
            window = snare_flux_product_norm[i - window_size + 1 : i + 1]

            # Get current BPM (interpolate from bpm timeline)
            current_time = timestamps[i]
            current_bpm = np.interp(current_time, bpm_time_points, bpm_values)

            if current_bpm <= 0:
                continue

            # Calculate lag values for 1x, 2x, 4x, 8x BPM
            # lag = frames_per_beat = frame_rate / (bpm / 60) = frame_rate * 60 / bpm
            beat_period_frames = frame_rate * 60.0 / current_bpm

            # Compute autocorrelation of the window
            # Normalize the window first
            window_mean = np.mean(window)
            window_centered = window - window_mean
            window_std = np.std(window_centered)
            if window_std > 0:
                window_norm = window_centered / window_std
            else:
                continue

            # Full autocorrelation
            autocorr = np.correlate(window_norm, window_norm, mode="full")
            # Take only positive lags (second half)
            autocorr = autocorr[len(autocorr) // 2 :]
            # Normalize by window size
            autocorr = autocorr / window_size

            # Check for peaks at 2x, 4x, 8x BPM lags
            # (shorter lags = faster repetition = higher multiplier)
            multipliers = [8, 4, 2]  # Check in order of fastest first
            lags = [beat_period_frames / m for m in multipliers]

            detected_multiplier = 0
            detected_magnitude = 0.0

            for mult, lag in zip(multipliers, lags):
                lag_int = int(round(lag))
                if lag_int < 2 or lag_int >= len(autocorr) - 1:
                    continue

                # Check if this lag is a local peak
                val = autocorr[lag_int]
                val_before = autocorr[lag_int - 1]
                val_after = autocorr[lag_int + 1]

                if val > val_before and val > val_after and val > 0.1:  # Threshold for significance
                    detected_multiplier = mult
                    detected_magnitude = val
                    break  # Take the first (fastest) peak found

            snare_roll_multiplier[i] = detected_multiplier
            snare_roll_magnitude[i] = detected_magnitude

        logger.info(f"  Snare roll detection: found {np.sum(snare_roll_multiplier > 0)} frames with rolls")

        # Buildup, Cut, and Drop detection
        # Buildup: snare roll + rising high/air flux + rising centroid
        # Cut: sudden drop in mid flux (slope < -10) - ends buildup
        # Drop: bass increase after buildup or within 2 bars after cut
        buildup_intensity = np.zeros(n_frames)
        buildup_raw_intensity = np.zeros(n_frames)  # For debugging - the raw combined metric
        cut_events = np.zeros(n_frames)  # 1.0 at cut frames
        drop_events = np.zeros(n_frames)  # 1.0 at drop frames

        # Get the slopes we need (already computed)
        high_slope = high_flux_slope
        air_slope = air_flux_slope
        bass_slope = bass_flux_slope
        mid_slope = mid_flux_slope
        centroid_slope_2s = centroid_slope_ewma_2s

        # First pass: compute raw intensity for all frames
        for i in range(n_frames):
            # Compute raw combined intensity metric
            # Use max of high and air flux slopes
            high_air_slope = max(high_slope[i], air_slope[i])

            # Snare component: magnitude scaled by whether it's 4x or 8x
            snare_intensity = 0.0
            if snare_roll_multiplier[i] >= 4:
                snare_intensity = snare_roll_magnitude[i]

            # Combined raw intensity (all positive contributions)
            # high/air slope: positive = rising, scale so 10 gives ~0.5 contribution
            # centroid slope: positive = rising, scale so 500 Hz/s gives ~0.5 contribution
            # snare: 0-1 range, already scaled
            raw_intensity = (
                max(0, high_air_slope) / 20.0  # high/air slope contribution
                + max(0, centroid_slope_2s[i]) / 1000.0  # centroid slope contribution
                + snare_intensity * 0.5  # snare contribution
            )
            buildup_raw_intensity[i] = raw_intensity

        # Apply 0.5s EWMA to raw intensity for smoothed growth detection
        raw_intensity_smoothed = compute_ewma_with_halflife(buildup_raw_intensity, 43)

        # Get BPM for calculating "2 bars" window for drop detection after cut
        bpm_time_points, bpm_values = self._calculate_bpm_timeline_internal(timestamps, hop_size)
        frame_rate = self.sample_rate / hop_size  # ~86 fps

        # State variables for buildup/cut/drop tracking
        in_buildup = False
        buildup_start_frame = 0
        buildup_start_offset = 0.0
        peak_smoothed_intensity = 0.0
        frames_since_peak = 0
        max_frames_decreasing = 86  # Allow ~1s dips

        # Cut/drop state
        last_cut_frame = -1000  # Frame of last cut event
        drop_window_frames = 0  # Will be calculated based on BPM (2 bars)
        cut_cooldown_frames = 172  # ~2 second cooldown between cuts (prevents cut-buildup-cut cycles)

        # Second pass: detect buildups, cuts, and drops
        for i in range(n_frames):
            # Calculate drop window based on current BPM (2 bars = 8 beats)
            current_time = timestamps[i]
            current_bpm = np.interp(current_time, bpm_time_points, bpm_values)
            if current_bpm > 0:
                beats_per_second = current_bpm / 60.0
                two_bars_seconds = 8.0 / beats_per_second  # 8 beats = 2 bars (assuming 4/4)
                drop_window_frames = int(two_bars_seconds * frame_rate)

            # Check for CUT: mid energy drops below threshold of long-term average
            # CUT only occurs when we are in a buildup (and ends the buildup)
            # Threshold: 0.25s EWMA mid energy < 50% of 4s EWMA baseline
            cut_threshold = 0.5
            is_cut = mid_energy_ratio[i] < cut_threshold
            cut_debounce_ok = (i - last_cut_frame) > cut_cooldown_frames

            if is_cut and cut_debounce_ok and in_buildup:
                # Register cut event - only during buildup
                cut_events[i] = 1.0
                last_cut_frame = i
                logger.info(
                    f"  CUT at frame {i} ({timestamps[i]:.2f}s) - mid energy ratio {mid_energy_ratio[i]:.2f} < {cut_threshold} - ending buildup"
                )
                in_buildup = False

            # Check for DROP: bass flux slope > 20 after buildup or within 2 bars of cut
            is_drop_candidate = bass_slope[i] > 20
            in_drop_window = (i - last_cut_frame) <= drop_window_frames

            if is_drop_candidate:
                if in_buildup:
                    # Drop during buildup (bass return)
                    drop_events[i] = 1.0
                    in_buildup = False
                    logger.info(f"  DROP at frame {i} ({timestamps[i]:.2f}s) - bass return during buildup")
                elif in_drop_window and last_cut_frame >= 0:
                    # Drop within 2 bars after cut
                    drop_events[i] = 1.0
                    logger.info(
                        f"  DROP at frame {i} ({timestamps[i]:.2f}s) - bass return after cut ({(i - last_cut_frame) / frame_rate:.2f}s after cut)"
                    )
                    # Reset cut window to prevent multiple drops from same cut
                    last_cut_frame = -1000

            # Check entry conditions for buildup (if not already in buildup)
            # Don't restart buildup within cut cooldown period (prevents immediate restart after cut)
            can_start_buildup = (i - last_cut_frame) > cut_cooldown_frames

            if not in_buildup and can_start_buildup:
                # Entry: snare magnitude > 0.3 AND (4x or 8x) AND (high OR air flux slope > 5)
                snare_entry = snare_roll_magnitude[i] > 0.3 and snare_roll_multiplier[i] >= 4
                slope_entry = high_slope[i] > 5 or air_slope[i] > 5

                if snare_entry and slope_entry:
                    in_buildup = True
                    buildup_start_frame = i
                    buildup_start_offset = raw_intensity_smoothed[i]
                    peak_smoothed_intensity = 0.0
                    frames_since_peak = 0
                    logger.info(f"  Buildup started at frame {i} ({timestamps[i]:.2f}s)")

            elif in_buildup:
                # Already in buildup - check exit conditions (cut and drop handled above)

                # Use smoothed intensity for growth check
                current_smoothed = raw_intensity_smoothed[i] - buildup_start_offset

                # Exit condition: smoothed intensity decreasing for too long
                if current_smoothed < peak_smoothed_intensity - 0.05:  # Small tolerance
                    frames_since_peak += 1
                    if frames_since_peak > max_frames_decreasing:
                        in_buildup = False
                        logger.info(f"  Buildup ended (intensity decreased) at frame {i} ({timestamps[i]:.2f}s)")
                        continue
                else:
                    if current_smoothed > peak_smoothed_intensity:
                        peak_smoothed_intensity = current_smoothed
                    frames_since_peak = 0

                # Still in buildup - set intensity based on smoothed value
                buildup_intensity[i] = max(0, current_smoothed)

        # Log summary
        buildup_frames = np.sum(buildup_intensity > 0)
        cut_count = int(np.sum(cut_events))
        drop_count = int(np.sum(drop_events))

        if buildup_frames > 0:
            buildup_start_time = timestamps[np.argmax(buildup_intensity > 0)]
            buildup_end_idx = len(buildup_intensity) - 1 - np.argmax(buildup_intensity[::-1] > 0)
            buildup_end_time = timestamps[buildup_end_idx]
            logger.info(
                f"  Buildup detection: {buildup_frames} frames ({buildup_start_time:.2f}s to {buildup_end_time:.2f}s)"
            )
        else:
            logger.info(f"  Buildup detection: no buildups detected")

        logger.info(f"  Cut events: {cut_count}")
        logger.info(f"  Drop events: {drop_count}")

        self.energy_timeline = {
            "timestamps": timestamps,
            "bass": bass_energy,
            "mid": mid_energy,
            "high": high_energy,
            "air": air_energy,
            "spectral_centroid": spectral_centroid,
            "bass_flux": bass_flux,
            "mid_flux": mid_flux,
            "high_flux": high_flux,
            "air_flux": air_flux,
            "centroid_ewma_05s": centroid_ewma_05s,
            "centroid_ewma_1s": centroid_ewma_1s,
            "centroid_ewma_2s": centroid_ewma_2s,
            "centroid_ewma_slow": centroid_ewma_slow,  # 4s
            "centroid_ewma_8s": centroid_ewma_8s,
            "centroid_slope_raw": centroid_slope_raw,
            "centroid_slope_ewma_2s": centroid_slope_ewma_2s,
            "centroid_slope_ewma_4s": centroid_slope_ewma_4s,
            "bass_flux_ewma_05s": bass_flux_ewma_05s,
            "bass_flux_ewma_1s": bass_flux_ewma_1s,
            "mid_flux_ewma_05s": mid_flux_ewma_05s,
            "mid_flux_ewma_1s": mid_flux_ewma_1s,
            "high_flux_ewma_05s": high_flux_ewma_05s,
            "high_flux_ewma_1s": high_flux_ewma_1s,
            "air_flux_ewma_05s": air_flux_ewma_05s,
            "air_flux_ewma_1s": air_flux_ewma_1s,
            "bass_flux_slope": bass_flux_slope,
            "mid_flux_slope": mid_flux_slope,
            "high_flux_slope": high_flux_slope,
            "air_flux_slope": air_flux_slope,
            "snare_body_flux": snare_body_flux,
            "snare_crack_flux": snare_crack_flux,
            "snare_flux_product": snare_flux_product,
            "snare_roll_multiplier": snare_roll_multiplier,
            "snare_roll_magnitude": snare_roll_magnitude,
            "buildup_intensity": buildup_intensity,
            "buildup_raw_intensity": buildup_raw_intensity,
            "cut_events": cut_events,
            "drop_events": drop_events,
            "mid_energy_ewma_025s": mid_energy_ewma_025s,
            "mid_energy_ewma_4s": mid_energy_ewma_4s,
            "mid_energy_ratio": mid_energy_ratio,
        }

        logger.info(f"  Computed {n_frames} frames of energy data")
        logger.info(
            f"  Slow EWMA half-life: {slow_half_life} frames (~{slow_half_life * hop_size / self.sample_rate:.1f}s)"
        )
        logger.info(
            f"  Frame rate: {1.0 / (hop_size / self.sample_rate):.1f} fps ({hop_size / self.sample_rate * 1000:.1f}ms per frame)"
        )

    def beat_callback(self, beat_event: BeatEvent):
        """
        Callback for beat events.

        Args:
            beat_event: Beat event from analyzer
        """
        self.beat_events.append(
            {
                "timestamp": beat_event.timestamp,
                "confidence": beat_event.confidence,
                "is_downbeat": beat_event.is_downbeat,
                "intensity": beat_event.intensity,
            }
        )

        # Log occasionally
        if len(self.beat_events) <= 10 or len(self.beat_events) % 10 == 0:
            logger.info(
                f"Beat #{len(self.beat_events)}: "
                f"time={beat_event.timestamp:.2f}s, "
                f"intensity={beat_event.intensity:.2f}, "
                f"confidence={beat_event.confidence:.3f}, "
                f"downbeat={beat_event.is_downbeat}"
            )

    def builddrop_callback(self, builddrop_event: BuildDropEvent):
        """
        Callback for build-up/drop events.

        Args:
            builddrop_event: Build-up/drop event from analyzer
        """
        self.builddrop_events.append(
            {
                "timestamp": builddrop_event.timestamp,
                "event_type": builddrop_event.event_type,
                "buildup_intensity": builddrop_event.buildup_intensity,
                "bass_energy": builddrop_event.bass_energy,
                "high_energy": builddrop_event.high_energy,
                "confidence": builddrop_event.confidence,
            }
        )

        # Log all state changes
        logger.info(
            f"BuildDrop Event #{len(self.builddrop_events)}: "
            f"state={builddrop_event.event_type}, "
            f"time={builddrop_event.timestamp:.2f}s, "
            f"intensity={builddrop_event.buildup_intensity:.2f}, "
            f"bass={builddrop_event.bass_energy:.4f}, "
            f"high={builddrop_event.high_energy:.4f}, "
            f"confidence={builddrop_event.confidence:.3f}"
        )

    def run_analysis(self):
        """Run beat analysis on the audio file"""
        logger.info("=" * 80)
        logger.info("Starting beat analysis")
        logger.info("=" * 80)

        # Create audio config for file playback
        audio_config = AudioConfig(
            sample_rate=self.sample_rate,
            channels=1,
            chunk_size=1024,
            file_path=str(self.audio_file),
            playback_speed=self.playback_speed,
        )

        # Create beat analyzer with build-up/drop detection enabled
        analyzer = AudioBeatAnalyzer(
            beat_callback=self.beat_callback,
            builddrop_callback=self.builddrop_callback,
            audio_config=audio_config,
            enable_builddrop_detection=True,
        )

        # Start analysis
        start_time = time.time()
        analyzer.start_analysis()

        # Wait for processing to complete
        # At playback_speed=10x, a 30s file takes ~3s to process
        estimated_time = self.duration / self.playback_speed
        logger.info(f"Processing... estimated time: {estimated_time:.1f}s")

        try:
            # Add a bit of buffer time
            time.sleep(estimated_time + 2.0)
        except KeyboardInterrupt:
            logger.warning("Analysis interrupted by user")
        finally:
            analyzer.stop_analysis()

        elapsed = time.time() - start_time

        logger.info("=" * 80)
        logger.info("Analysis complete")
        logger.info(f"  Wall time: {elapsed:.1f}s")
        logger.info(f"  Audio duration: {self.duration:.1f}s")
        logger.info(f"  Speed ratio: {self.duration/elapsed:.1f}x")
        logger.info(f"  Beats detected: {len(self.beat_events)}")
        logger.info("=" * 80)

        return self.beat_events

    def calculate_bpm_timeline(self, window_size: float = 5.0):
        """
        Calculate BPM over time using sliding window.

        Args:
            window_size: Window size in seconds for BPM calculation

        Returns:
            Tuple of (time_points, bpm_values)
        """
        if len(self.beat_events) < 2:
            return np.array([]), np.array([])

        # Create time points every 0.5 seconds
        time_points = np.arange(0, self.duration, 0.5)
        bpm_values = np.zeros(len(time_points))

        for i, t in enumerate(time_points):
            # Find beats within window [t - window_size/2, t + window_size/2]
            window_beats = [
                b["timestamp"] for b in self.beat_events if t - window_size / 2 <= b["timestamp"] <= t + window_size / 2
            ]

            if len(window_beats) >= 2:
                # Calculate BPM from intervals
                intervals = np.diff(window_beats)
                if len(intervals) > 0:
                    avg_interval = np.mean(intervals)
                    if avg_interval > 0:
                        bpm_values[i] = 60.0 / avg_interval

        return time_points, bpm_values

    def _setup_time_axis(self, ax, duration: float):
        """
        Configure x-axis with 1-second tick marks (unlabeled) and labels every 5 or 10 seconds.

        Args:
            ax: Matplotlib axis object
            duration: Total duration in seconds
        """
        from matplotlib.ticker import FuncFormatter, MultipleLocator

        # Set x-axis limits
        ax.set_xlim(0, duration)

        # Determine label interval based on duration
        if duration > 60:
            label_interval = 10.0  # Label every 10s for long audio
        elif duration > 20:
            label_interval = 5.0  # Label every 5s for medium audio
        else:
            label_interval = 2.0  # Label every 2s for short audio

        # Minor ticks every 1 second (unlabeled tick marks)
        ax.xaxis.set_minor_locator(MultipleLocator(1.0))

        # Major ticks at label intervals (these get labels)
        ax.xaxis.set_major_locator(MultipleLocator(label_interval))

        # Format labels as integers
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x)}"))

        # Enable grid lines - minor (1s) are subtle, major (labeled) are more visible
        ax.grid(True, which="major", alpha=0.5)
        ax.grid(True, which="minor", alpha=0.2, linestyle=":")

        # Set x-axis label
        ax.set_xlabel("Time (seconds)")

    def plot_results(self, output_path: Path):
        """
        Plot beat detection results.

        Args:
            output_path: Output path for plot (will output .svg for vector graphics)
        """
        logger.info("Generating visualization...")

        # Compute per-frame energy bands and spectral centroid
        self._compute_energy_bands()

        # Create figure with subplots (15 subplots total)
        fig, axes = plt.subplots(15, 1, figsize=(16, 45))

        # Plot 1: Spectrogram with spectral centroid overlay
        try:
            import librosa.display

            # Compute mel spectrogram for better visualization
            n_fft = 2048
            hop_length = 512
            S = librosa.feature.melspectrogram(
                y=self.audio_data, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=128
            )
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Plot spectrogram
            img = librosa.display.specshow(
                S_dB,
                x_axis="time",
                y_axis="mel",
                sr=self.sample_rate,
                hop_length=hop_length,
                ax=axes[0],
                cmap="magma",
            )
            axes[0].set_ylabel("Frequency (Hz)")
            axes[0].set_title(f"Mel Spectrogram with Spectral Centroid ({self.duration:.1f}s)")

        except ImportError:
            # Fallback to matplotlib spectrogram
            axes[0].specgram(
                self.audio_data,
                NFFT=2048,
                Fs=self.sample_rate,
                noverlap=1024,
                cmap="magma",
            )
            axes[0].set_ylabel("Frequency (Hz)")
            axes[0].set_title(f"Spectrogram with Spectral Centroid ({self.duration:.1f}s)")
            axes[0].set_ylim(0, 8000)

        self._setup_time_axis(axes[0], self.duration)

        # Overlay spectral centroid on spectrogram
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]
            centroid = self.energy_timeline["spectral_centroid"]
            # Plot centroid as a white/cyan line on top of spectrogram
            axes[0].plot(t, centroid, linewidth=0.5, color="cyan", alpha=0.8, label="Spectral Centroid")
            axes[0].legend(loc="upper left")

        # Plot 2: BPM over time
        time_points, bpm_values = self.calculate_bpm_timeline()
        if len(time_points) > 0:
            valid_mask = bpm_values > 0
            axes[1].plot(time_points[valid_mask], bpm_values[valid_mask], linewidth=2, color="purple", marker="o")
            axes[1].set_ylabel("BPM")
            axes[1].set_title("Tempo (BPM) Over Time")
            self._setup_time_axis(axes[1], self.duration)
            if np.any(valid_mask):
                axes[1].set_ylim(0, max(200, np.max(bpm_values[valid_mask]) * 1.1))
        else:
            axes[1].text(
                0.5,
                0.5,
                "Insufficient beats for BPM calculation",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_ylabel("BPM")
            axes[1].set_title("Tempo (BPM) Over Time")
            self._setup_time_axis(axes[1], self.duration)

        # Plot 3: Beat intensity and confidence over time
        if self.beat_events:
            beat_times = [b["timestamp"] for b in self.beat_events]
            beat_intensities = [b["intensity"] for b in self.beat_events]
            beat_confidences = [b["confidence"] for b in self.beat_events]

            axes[2].scatter(
                beat_times,
                beat_intensities,
                c="red",
                s=50,
                alpha=0.7,
                edgecolors="darkred",
                linewidth=0.5,
                label="Intensity (RMS)",
            )
            axes[2].plot(beat_times, beat_intensities, linewidth=1.5, color="red", alpha=0.4, linestyle="-")
            axes[2].scatter(
                beat_times,
                beat_confidences,
                c="blue",
                s=50,
                alpha=0.7,
                edgecolors="darkblue",
                linewidth=0.5,
                label="Confidence",
            )
            axes[2].plot(beat_times, beat_confidences, linewidth=1.5, color="blue", alpha=0.4, linestyle="-")

            axes[2].set_ylabel("Value (0.0 - 1.0)")
            axes[2].set_title("Beat Intensity (RMS) and Confidence Over Time")
            self._setup_time_axis(axes[2], self.duration)
            axes[2].set_ylim(0, 1.1)
            axes[2].legend(loc="upper left")
        else:
            axes[2].text(0.5, 0.5, "No beats detected", ha="center", va="center", transform=axes[2].transAxes)
            axes[2].set_ylabel("Value (0.0 - 1.0)")
            axes[2].set_title("Beat Intensity and Confidence Over Time")
            self._setup_time_axis(axes[2], self.duration)

        # Plot 4: RMS envelope with beat markers (thin lines)
        window_size = int(self.sample_rate * 0.05)  # 50ms windows
        rms_envelope = self._calculate_rms_envelope(self.audio_data, window_size)
        rms_time = np.linspace(0, self.duration, len(rms_envelope))
        axes[3].plot(rms_time, rms_envelope, linewidth=0.5, color="orange")
        axes[3].set_ylabel("RMS Amplitude")
        axes[3].set_title("RMS Envelope with Beat Markers")
        self._setup_time_axis(axes[3], self.duration)

        if self.beat_events:
            beat_times = [b["timestamp"] for b in self.beat_events]
            axes[3].vlines(beat_times, 0, np.max(rms_envelope), colors="red", alpha=0.4, linewidth=0.5)

        # Plot 5: Bass Energy (separate scale)
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]
            axes[4].plot(t, self.energy_timeline["bass"], linewidth=0.5, color="red", alpha=0.9)
            axes[4].set_ylabel("Bass Energy (20-250 Hz)")
            axes[4].set_title("Bass Energy Over Time (~86 frames/sec, 11.6ms per frame)")
            self._setup_time_axis(axes[4], self.duration)
        else:
            axes[4].text(0.5, 0.5, "No energy data", ha="center", va="center", transform=axes[4].transAxes)
            axes[4].set_ylabel("Energy")
            axes[4].set_title("Bass Energy Over Time")
            self._setup_time_axis(axes[4], self.duration)

        # Plot 6: Mid/High/Air Energy (shared scale)
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]
            axes[5].plot(
                t, self.energy_timeline["mid"], linewidth=0.5, color="orange", label="Mid (250-2000 Hz)", alpha=0.9
            )
            axes[5].plot(
                t, self.energy_timeline["high"], linewidth=0.5, color="blue", label="High (2000-8000 Hz)", alpha=0.9
            )
            axes[5].plot(
                t, self.energy_timeline["air"], linewidth=0.5, color="cyan", label="Air (8000-16000 Hz)", alpha=0.9
            )
            axes[5].set_ylabel("Energy")
            axes[5].set_title("Mid/High/Air Energy Over Time (~86 frames/sec, 11.6ms per frame)")
            self._setup_time_axis(axes[5], self.duration)
            axes[5].legend(loc="upper left")
        else:
            axes[5].text(0.5, 0.5, "No energy data", ha="center", va="center", transform=axes[5].transAxes)
            axes[5].set_ylabel("Energy")
            axes[5].set_title("Mid/High/Air Energy Over Time")
            self._setup_time_axis(axes[5], self.duration)

        # Plot 7: Spectral Flux (onset detection per band)
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]
            axes[6].plot(
                t,
                self.energy_timeline["bass_flux"],
                linewidth=0.5,
                color="red",
                label="Bass Flux (20-250 Hz)",
                alpha=0.9,
            )
            axes[6].plot(
                t,
                self.energy_timeline["mid_flux"],
                linewidth=0.5,
                color="orange",
                label="Mid Flux (250-2000 Hz)",
                alpha=0.9,
            )
            axes[6].plot(
                t,
                self.energy_timeline["high_flux"],
                linewidth=0.5,
                color="blue",
                label="High Flux (2000-8000 Hz)",
                alpha=0.9,
            )
            axes[6].plot(
                t,
                self.energy_timeline["air_flux"],
                linewidth=0.5,
                color="cyan",
                label="Air Flux (8000-16000 Hz)",
                alpha=0.9,
            )
            axes[6].set_ylabel("Spectral Flux")
            axes[6].set_title("Spectral Flux (Half-Wave Rectified Difference) - Onset Detection")
            self._setup_time_axis(axes[6], self.duration)
            axes[6].legend(loc="upper left")
        else:
            axes[6].text(0.5, 0.5, "No flux data", ha="center", va="center", transform=axes[6].transAxes)
            axes[6].set_ylabel("Spectral Flux")
            axes[6].set_title("Spectral Flux - Onset Detection")
            self._setup_time_axis(axes[6], self.duration)

        # Plot 8: Centroid with multiple EWMAs for comparison
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]

            # Raw centroid and EWMAs with different half-lives
            axes[7].plot(
                t, self.energy_timeline["spectral_centroid"], linewidth=0.3, color="black", alpha=0.4, label="Raw"
            )
            axes[7].plot(
                t,
                self.energy_timeline["centroid_ewma_05s"],
                linewidth=0.5,
                color="magenta",
                alpha=0.7,
                label="EWMA 0.5s",
            )
            axes[7].plot(
                t, self.energy_timeline["centroid_ewma_1s"], linewidth=0.6, color="red", alpha=0.8, label="EWMA 1s"
            )
            axes[7].plot(
                t, self.energy_timeline["centroid_ewma_2s"], linewidth=0.6, color="orange", alpha=0.8, label="EWMA 2s"
            )
            axes[7].plot(
                t, self.energy_timeline["centroid_ewma_slow"], linewidth=0.8, color="blue", alpha=0.9, label="EWMA 4s"
            )
            axes[7].plot(
                t, self.energy_timeline["centroid_ewma_8s"], linewidth=0.8, color="purple", alpha=0.9, label="EWMA 8s"
            )
            axes[7].set_ylabel("Spectral Centroid (Hz)")
            axes[7].legend(loc="upper left")
            axes[7].set_title("Spectral Centroid with EWMAs (0.5s, 1s, 2s, 4s, 8s half-lives)")
            self._setup_time_axis(axes[7], self.duration)
        else:
            axes[7].text(0.5, 0.5, "No centroid data", ha="center", va="center", transform=axes[7].transAxes)
            axes[7].set_ylabel("Centroid")
            axes[7].set_title("Spectral Centroid")
            self._setup_time_axis(axes[7], self.duration)

        # Plot 9: Spectral Flux EWMAs (0.5s and 1s) for each band
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]

            # 0.5s EWMAs (dashed lines)
            axes[8].plot(
                t,
                self.energy_timeline["bass_flux_ewma_05s"],
                linewidth=0.5,
                color="red",
                alpha=0.6,
                linestyle="--",
                label="Bass 0.5s",
            )
            axes[8].plot(
                t,
                self.energy_timeline["mid_flux_ewma_05s"],
                linewidth=0.5,
                color="orange",
                alpha=0.6,
                linestyle="--",
                label="Mid 0.5s",
            )
            axes[8].plot(
                t,
                self.energy_timeline["high_flux_ewma_05s"],
                linewidth=0.5,
                color="blue",
                alpha=0.6,
                linestyle="--",
                label="High 0.5s",
            )
            axes[8].plot(
                t,
                self.energy_timeline["air_flux_ewma_05s"],
                linewidth=0.5,
                color="cyan",
                alpha=0.6,
                linestyle="--",
                label="Air 0.5s",
            )
            # 1s EWMAs (solid lines)
            axes[8].plot(
                t, self.energy_timeline["bass_flux_ewma_1s"], linewidth=0.8, color="red", alpha=0.9, label="Bass 1s"
            )
            axes[8].plot(
                t, self.energy_timeline["mid_flux_ewma_1s"], linewidth=0.8, color="orange", alpha=0.9, label="Mid 1s"
            )
            axes[8].plot(
                t, self.energy_timeline["high_flux_ewma_1s"], linewidth=0.8, color="blue", alpha=0.9, label="High 1s"
            )
            axes[8].plot(
                t, self.energy_timeline["air_flux_ewma_1s"], linewidth=0.8, color="cyan", alpha=0.9, label="Air 1s"
            )
            axes[8].set_ylabel("Flux EWMA")
            axes[8].legend(loc="upper left", ncol=2, fontsize=8)
            axes[8].set_title("Spectral Flux EWMAs (0.5s dashed, 1s solid) - Bass/Mid/High/Air")
            self._setup_time_axis(axes[8], self.duration)
        else:
            axes[8].text(0.5, 0.5, "No flux EWMA data", ha="center", va="center", transform=axes[8].transAxes)
            axes[8].set_ylabel("Flux EWMA")
            axes[8].set_title("Spectral Flux EWMAs")
            self._setup_time_axis(axes[8], self.duration)

        # Plot 10: Centroid slope (0.25s interval diff of 1s EWMA, then 2s/4s EWMA smoothing)
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]

            # Raw slope from 1s EWMA (already scaled to Hz/s)
            slope_raw = self.energy_timeline["centroid_slope_raw"]
            slope_2s = self.energy_timeline["centroid_slope_ewma_2s"]
            slope_4s = self.energy_timeline["centroid_slope_ewma_4s"]

            # Plot smoothed slopes on left y-axis
            axes[9].plot(t, slope_2s, linewidth=0.8, color="orange", alpha=0.9, label="2s EWMA of slope")
            axes[9].plot(t, slope_4s, linewidth=0.8, color="purple", alpha=0.9, label="4s EWMA of slope")
            axes[9].axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
            axes[9].set_ylabel("Smoothed Slope (Hz/s)")

            # Plot raw slope on right y-axis with separate scale
            ax9_right = axes[9].twinx()
            ax9_right.plot(t, slope_raw, linewidth=0.3, color="gray", alpha=0.4, label="Raw (1s EWMA, 0.25s diff)")
            ax9_right.set_ylabel("Raw Slope (Hz/s)", color="gray")
            ax9_right.tick_params(axis="y", labelcolor="gray")

            # Combine legends from both axes
            lines1, labels1 = axes[9].get_legend_handles_labels()
            lines2, labels2 = ax9_right.get_legend_handles_labels()
            axes[9].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            axes[9].set_title("Centroid Slope: 1s EWMA, 0.25s interval diff, with 2s/4s EWMA smoothing (Hz/s)")
            self._setup_time_axis(axes[9], self.duration)
        else:
            axes[9].text(0.5, 0.5, "No trend data", ha="center", va="center", transform=axes[9].transAxes)
            axes[9].set_ylabel("Trend")
            axes[9].set_title("Centroid Trend")
            self._setup_time_axis(axes[9], self.duration)

        # Plot 11: Spectral Flux Slopes (0.5s EWMA -> 0.25s diff -> 1s EWMA)
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]

            axes[10].plot(
                t,
                self.energy_timeline["bass_flux_slope"],
                linewidth=0.8,
                color="red",
                alpha=0.9,
                label="Bass (20-250 Hz)",
            )
            axes[10].plot(
                t,
                self.energy_timeline["mid_flux_slope"],
                linewidth=0.8,
                color="orange",
                alpha=0.9,
                label="Mid (250-2000 Hz)",
            )
            axes[10].plot(
                t,
                self.energy_timeline["high_flux_slope"],
                linewidth=0.8,
                color="blue",
                alpha=0.9,
                label="High (2000-8000 Hz)",
            )
            axes[10].plot(
                t,
                self.energy_timeline["air_flux_slope"],
                linewidth=0.8,
                color="cyan",
                alpha=0.9,
                label="Air (8000-16000 Hz)",
            )
            axes[10].axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
            axes[10].set_ylabel("Flux Slope (per second)")
            axes[10].set_title("Spectral Flux Slopes: 0.5s EWMA, 0.25s interval diff, 1s EWMA smoothing")
            self._setup_time_axis(axes[10], self.duration)
            axes[10].legend(loc="upper left")
        else:
            axes[10].text(0.5, 0.5, "No flux slope data", ha="center", va="center", transform=axes[10].transAxes)
            axes[10].set_ylabel("Flux Slope")
            axes[10].set_title("Spectral Flux Slopes")
            self._setup_time_axis(axes[10], self.duration)

        # Plot 12: Snare Roll Multiplier (2x, 4x, 8x or none)
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]
            multiplier = self.energy_timeline["snare_roll_multiplier"]

            # Plot as a step function with colored regions
            # Use scatter for detected rolls, colored by multiplier
            roll_mask = multiplier > 0
            if np.any(roll_mask):
                t_rolls = t[roll_mask]
                m_rolls = multiplier[roll_mask]

                # Color by multiplier: 2x=blue, 4x=orange, 8x=red
                colors = []
                for m in m_rolls:
                    if m == 2:
                        colors.append("blue")
                    elif m == 4:
                        colors.append("orange")
                    elif m == 8:
                        colors.append("red")
                    else:
                        colors.append("gray")

                axes[11].scatter(t_rolls, m_rolls, c=colors, s=10, alpha=0.7)

            # Also plot as a line for continuity
            axes[11].plot(t, multiplier, linewidth=0.5, color="gray", alpha=0.3)

            axes[11].set_ylabel("Roll Multiplier")
            axes[11].set_yticks([0, 2, 4, 8])
            axes[11].set_yticklabels(["none", "2x", "4x", "8x"])
            axes[11].set_ylim(-0.5, 9)
            axes[11].set_title("Snare Roll Detection: Multiplier (autocorrelation at BPM-related lags)")
            self._setup_time_axis(axes[11], self.duration)

            # Add legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=8, label="2x BPM"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=8, label="4x BPM"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="8x BPM"),
            ]
            axes[11].legend(handles=legend_elements, loc="upper left")
        else:
            axes[11].text(0.5, 0.5, "No snare roll data", ha="center", va="center", transform=axes[11].transAxes)
            axes[11].set_ylabel("Roll Multiplier")
            axes[11].set_title("Snare Roll Detection: Multiplier")
            self._setup_time_axis(axes[11], self.duration)

        # Plot 13: Snare Roll Magnitude (autocorrelation peak value)
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]
            magnitude = self.energy_timeline["snare_roll_magnitude"]
            multiplier = self.energy_timeline["snare_roll_multiplier"]

            # Plot magnitude colored by multiplier
            # Background: all magnitude values in gray
            axes[12].fill_between(t, 0, magnitude, color="lightgray", alpha=0.5)

            # Overlay colored points where rolls are detected
            roll_mask = multiplier > 0
            if np.any(roll_mask):
                t_rolls = t[roll_mask]
                mag_rolls = magnitude[roll_mask]
                m_rolls = multiplier[roll_mask]

                colors = []
                for m in m_rolls:
                    if m == 2:
                        colors.append("blue")
                    elif m == 4:
                        colors.append("orange")
                    elif m == 8:
                        colors.append("red")
                    else:
                        colors.append("gray")

                axes[12].scatter(t_rolls, mag_rolls, c=colors, s=10, alpha=0.7)

            axes[12].axhline(0.1, color="gray", linewidth=0.5, linestyle="--", alpha=0.5, label="Detection threshold")
            axes[12].set_ylabel("Autocorr Peak Magnitude")
            axes[12].set_ylim(0, 1.0)
            axes[12].set_title("Snare Roll Detection: Autocorrelation Peak Magnitude (snare body × crack flux)")
            self._setup_time_axis(axes[12], self.duration)
            axes[12].legend(loc="upper left")
        else:
            axes[12].text(0.5, 0.5, "No snare roll data", ha="center", va="center", transform=axes[12].transAxes)
            axes[12].set_ylabel("Magnitude")
            axes[12].set_title("Snare Roll Detection: Magnitude")
            self._setup_time_axis(axes[12], self.duration)

        # Plot 14: Buildup Intensity with Cut and Drop markers
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]
            buildup_intensity = self.energy_timeline["buildup_intensity"]
            buildup_raw = self.energy_timeline["buildup_raw_intensity"]
            cut_events = self.energy_timeline["cut_events"]
            drop_events = self.energy_timeline["drop_events"]

            # Plot raw intensity as background (light gray area)
            axes[13].fill_between(t, 0, buildup_raw, color="lightgray", alpha=0.4, label="Raw combined metric")

            # Plot buildup intensity as a filled area (only non-zero during buildup)
            buildup_mask = buildup_intensity > 0
            if np.any(buildup_mask):
                axes[13].fill_between(t, 0, buildup_intensity, color="red", alpha=0.6, label="Buildup intensity")
                axes[13].plot(t, buildup_intensity, linewidth=1.0, color="darkred", alpha=0.9)

            # Add threshold lines and annotations
            axes[13].axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

            # Get y-axis limit for marker height
            max_raw = np.max(buildup_raw)
            if max_raw > 0:
                marker_height = max_raw * 1.1
            else:
                marker_height = 1.0

            # Plot CUT events as vertical cyan lines
            cut_times = t[cut_events > 0]
            if len(cut_times) > 0:
                for ct in cut_times:
                    axes[13].axvline(ct, color="cyan", linewidth=2, alpha=0.8, linestyle="-")
                # Add single legend entry
                axes[13].axvline(cut_times[0], color="cyan", linewidth=2, alpha=0.8, label=f"CUT ({len(cut_times)})")

            # Plot DROP events as vertical green lines
            drop_times = t[drop_events > 0]
            if len(drop_times) > 0:
                for dt in drop_times:
                    axes[13].axvline(dt, color="lime", linewidth=3, alpha=0.9, linestyle="-")
                # Add single legend entry
                axes[13].axvline(drop_times[0], color="lime", linewidth=3, alpha=0.9, label=f"DROP ({len(drop_times)})")

            axes[13].set_ylabel("Buildup Intensity")
            axes[13].set_title("Buildup/Cut/Drop Detection: buildup (red), cut (cyan), drop (green)")
            self._setup_time_axis(axes[13], self.duration)
            axes[13].legend(loc="upper left")

            # Set reasonable y-axis limits
            if max_raw > 0:
                axes[13].set_ylim(0, marker_height)
        else:
            axes[13].text(0.5, 0.5, "No buildup data", ha="center", va="center", transform=axes[13].transAxes)
            axes[13].set_ylabel("Buildup Intensity")
            axes[13].set_title("Buildup Detection")
            self._setup_time_axis(axes[13], self.duration)

        # Plot 15: Mid Energy Ratio (for cut detection)
        if self.energy_timeline is not None:
            t = self.energy_timeline["timestamps"]
            mid_ratio = self.energy_timeline["mid_energy_ratio"]
            cut_events = self.energy_timeline["cut_events"]

            # Plot the ratio
            axes[14].plot(t, mid_ratio, linewidth=0.5, color="orange", alpha=0.9, label="Mid Energy Ratio")

            # Add threshold line at 0.5
            axes[14].axhline(0.5, color="red", linewidth=1.5, linestyle="--", alpha=0.7, label="Cut threshold (0.5)")
            axes[14].axhline(1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5, label="Baseline (1.0)")

            # Mark cut events
            cut_times = t[cut_events > 0]
            if len(cut_times) > 0:
                for ct in cut_times:
                    axes[14].axvline(ct, color="cyan", linewidth=2, alpha=0.8)
                axes[14].axvline(cut_times[0], color="cyan", linewidth=2, alpha=0.8, label=f"CUT ({len(cut_times)})")

            axes[14].set_ylabel("Mid Energy Ratio")
            axes[14].set_title("Mid Energy Ratio (0.25s EWMA / 4s EWMA baseline) - Cut Detection")
            self._setup_time_axis(axes[14], self.duration)
            axes[14].set_ylim(0, max(2.0, np.max(mid_ratio) * 1.1))
            axes[14].legend(loc="upper left")
        else:
            axes[14].text(0.5, 0.5, "No mid energy ratio data", ha="center", va="center", transform=axes[14].transAxes)
            axes[14].set_ylabel("Ratio")
            axes[14].set_title("Mid Energy Ratio")
            self._setup_time_axis(axes[14], self.duration)

        # Add statistics text
        stats_text = f"Beats detected: {len(self.beat_events)}"
        if len(self.beat_events) >= 2:
            intervals = np.diff([b["timestamp"] for b in self.beat_events])
            avg_interval = np.mean(intervals)
            avg_bpm = 60.0 / avg_interval if avg_interval > 0 else 0
            stats_text += f"  |  Average BPM: {avg_bpm:.1f}"

            # Add intensity statistics
            intensities = [b["intensity"] for b in self.beat_events]
            avg_intensity = np.mean(intensities)
            max_intensity = np.max(intensities)
            min_intensity = np.min(intensities)
            stats_text += f"  |  Intensity: avg={avg_intensity:.2f}, min={min_intensity:.2f}, max={max_intensity:.2f}"

            # Add confidence statistics
            confidences = [b["confidence"] for b in self.beat_events]
            avg_confidence = np.mean(confidences)
            min_confidence = np.min(confidences)
            stats_text += f"  |  Confidence: avg={avg_confidence:.2f}, min={min_confidence:.2f}"

        fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, bbox={"boxstyle": "round", "facecolor": "wheat"})

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualization saved: {output_path}")

    def _calculate_rms_envelope(self, audio_data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Calculate RMS envelope of audio data.

        Args:
            audio_data: Audio data array
            window_size: Window size in samples

        Returns:
            RMS envelope array
        """
        n_windows = len(audio_data) // window_size
        rms_envelope = np.zeros(n_windows)

        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window = audio_data[start:end]
            rms_envelope[i] = np.sqrt(np.mean(window**2))

        return rms_envelope


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test audio beat analyzer on a WAV file")
    parser.add_argument("audio_file", nargs="?", help="Path to WAV file (default: whereyouare.wav)")
    parser.add_argument(
        "--speed", type=float, default=10.0, help="Playback speed multiplier (default: 10.0 for faster testing)"
    )
    parser.add_argument("--output", type=str, help="Output path for plot (default: auto-generated)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Determine audio file
    if args.audio_file:
        audio_file = Path(args.audio_file)
    else:
        # Default to audio_capture_test.wav in project root
        project_root = Path(__file__).parent.parent
        audio_file = project_root / "whereyouare.wav"

    if not audio_file.exists():
        logger.error(f"Audio file not found: {audio_file}")
        logger.info("Please provide a WAV file or run tools/audio_capture_test.py to generate one")
        return 1

    # Determine output path (default to SVG for vector graphics)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = audio_file.parent / f"{audio_file.stem}_beat_analysis.svg"

    # Create tester and run analysis
    tester = BeatAnalyzerTester(audio_file, playback_speed=args.speed)
    beat_events = tester.run_analysis()

    # Generate visualization
    tester.plot_results(output_path)

    logger.info("\n" + "=" * 80)
    logger.info("Beat analysis test completed successfully")
    logger.info(f"  Input: {audio_file}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Beats detected: {len(beat_events)}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
