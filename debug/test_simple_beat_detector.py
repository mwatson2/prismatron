#!/usr/bin/env python3
"""
Simple Real-Time Beat Detector using Spectral Flux
Optimized for EDM/House music with clear beats
Uses sounddevice for reliable audio capture
"""

import queue
import threading
import time
from collections import deque

import librosa
import numpy as np
import sounddevice as sd


class RealTimeBeatDetector:
    def __init__(self, sample_rate=44100, chunk_size=2048):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.hop_length = chunk_size // 4  # 75% overlap for smoothness

        # Rolling buffer for spectral analysis (sliding window)
        self.buffer_duration = 4  # Number of chunks (~93ms total)
        self.buffer = deque(maxlen=self.buffer_duration)
        self.prev_spectrum = None

        # Beat detection parameters (tuned for EDM/house main beats)
        self.onset_threshold = 0.5  # Higher threshold for stronger beats only
        self.min_beat_interval_ms = 400  # 400ms minimum (150 BPM max)
        self.min_beat_interval_chunks = int(self.min_beat_interval_ms / (chunk_size / sample_rate * 1000))
        self.last_beat_chunk = 0
        self.current_chunk = 0

        # Additional filtering for consistent beats
        self.min_flux_ratio = 3.0  # Beat must be 3x stronger than baseline

        # BPM tracking
        self.beat_times = deque(maxlen=16)
        self.current_bpm = 0

        # Statistics
        self.total_beats = 0
        self.total_chunks = 0

        # Baseline flux tracking for adaptive thresholding
        self.flux_history = deque(maxlen=100)  # Track recent flux values
        self.baseline_flux = 0.1

    def detect_beat(self, audio_chunk):
        """
        Detect beat in audio chunk using spectral flux method
        Returns True if beat detected, False otherwise
        """
        # Convert to mono if stereo
        if len(audio_chunk.shape) > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)

        # Ensure correct size
        if len(audio_chunk) != self.chunk_size:
            # Pad or truncate to chunk_size
            if len(audio_chunk) < self.chunk_size:
                audio_chunk = np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[: self.chunk_size]

        self.buffer.append(audio_chunk)
        self.current_chunk += 1
        self.total_chunks += 1

        # Need full buffer for analysis
        if len(self.buffer) < self.buffer_duration:
            return False

        # Concatenate buffer for analysis (sliding window)
        signal = np.concatenate(list(self.buffer))

        # Apply window to reduce spectral leakage
        window = np.hanning(len(signal))
        signal = signal * window

        # Compute STFT and get magnitude spectrum
        try:
            stft = librosa.stft(signal, hop_length=self.hop_length, n_fft=1024)
            spectrum = np.mean(np.abs(stft), axis=1)  # Average magnitude across time

            if self.prev_spectrum is not None:
                # Spectral flux: sum of positive differences
                # Focus on low frequencies where kicks are prominent (20-150 Hz bins)
                low_freq_bins = int(20 * len(spectrum) / (self.sample_rate / 2))
                kick_bins = int(150 * len(spectrum) / (self.sample_rate / 2))
                flux = np.sum(
                    np.maximum(0, spectrum[low_freq_bins:kick_bins] - self.prev_spectrum[low_freq_bins:kick_bins])
                )

                # Track flux history for adaptive baseline
                self.flux_history.append(flux)
                if len(self.flux_history) > 10:
                    self.baseline_flux = np.percentile(self.flux_history, 70)  # 70th percentile as baseline

                # More selective beat detection
                chunks_since_last = self.current_chunk - self.last_beat_chunk

                # Beat must be:
                # 1. Above absolute threshold
                # 2. Above adaptive baseline threshold
                # 3. After minimum interval
                # 4. Strong relative to recent average
                adaptive_threshold = max(self.onset_threshold, self.baseline_flux * self.min_flux_ratio)

                is_beat = (
                    flux > self.onset_threshold
                    and flux > adaptive_threshold
                    and chunks_since_last > self.min_beat_interval_chunks
                )

                if is_beat:
                    self.last_beat_chunk = self.current_chunk
                    self.total_beats += 1

                    # Track timing for BPM
                    current_time = time.time()
                    self.beat_times.append(current_time)

                    # Calculate BPM from recent beats with stricter filtering
                    if len(self.beat_times) >= 3:
                        intervals = np.diff(list(self.beat_times))
                        # Filter for more consistent intervals (90-140 BPM typical for house/EDM)
                        valid_intervals = intervals[(intervals > 0.43) & (intervals < 0.67)]  # 90-140 BPM
                        if len(valid_intervals) >= 2:
                            # Check consistency - reject if intervals vary too much
                            std_dev = np.std(valid_intervals)
                            mean_interval = np.mean(valid_intervals)
                            if std_dev < mean_interval * 0.2:  # Less than 20% variation
                                self.current_bpm = 60.0 / mean_interval

                    return True

            self.prev_spectrum = spectrum

        except Exception as e:
            print(f"Error in spectral analysis: {e}")

        return False

    def get_stats(self):
        """Get detection statistics"""
        processing_time = self.total_chunks * self.chunk_size / self.sample_rate
        return {
            "total_beats": self.total_beats,
            "total_chunks": self.total_chunks,
            "processing_time": processing_time,
            "beats_per_minute": self.current_bpm,
            "detection_rate": self.total_beats / processing_time * 60 if processing_time > 0 else 0,
        }


def find_usb_device():
    """Find USB Audio Device"""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if "USB Audio" in device["name"] and device["max_input_channels"] > 0:
            print(f"Found USB Audio Device at index {i}: {device['name']}")
            return i
    print("USB Audio Device not found, will use default")
    return None


def test_beat_detector():
    """Test the simple beat detector"""
    print("=" * 60)
    print("SIMPLE REAL-TIME BEAT DETECTOR")
    print("Optimized for EDM/House music")
    print("=" * 60)

    # Find USB device
    usb_device = find_usb_device()

    # Initialize detector
    sample_rate = 44100
    chunk_size = 2048  # ~46ms chunks
    detector = RealTimeBeatDetector(sample_rate=sample_rate, chunk_size=chunk_size)

    print(f"Sample rate: {sample_rate}Hz")
    print(f"Chunk size: {chunk_size} samples (~{chunk_size/sample_rate*1000:.1f}ms)")
    print(f"Buffer: {detector.buffer_duration} chunks (~{detector.buffer_duration*chunk_size/sample_rate*1000:.0f}ms)")
    print(f"Min beat interval: {detector.min_beat_interval_ms}ms")
    print()
    print("Play some EDM/House music and watch for beat detection!")
    print("Press Ctrl+C to stop")
    print()
    print("-" * 60)
    print("Time     | BPM   | Beats | AvgBPM| Status")
    print("-" * 60)
    print("(BPM=current tempo, AvgBPM=session average)")

    start_time = time.time()
    last_stats_time = start_time
    stats_interval = 2.0  # Print stats every 2 seconds

    overflow_count = 0

    def audio_callback(indata, frames, time_info, status):
        """Audio callback function"""
        nonlocal overflow_count

        if status:
            overflow_count += 1
            # Only print overflow occasionally to avoid spam
            if overflow_count % 50 == 0:
                print(f"⚠️  Audio overflow #{overflow_count}")

        # Convert from sounddevice format
        audio_data = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()

        # Process in chunks if we got a larger block
        for i in range(0, len(audio_data), chunk_size):
            audio_chunk = audio_data[i : i + chunk_size]
            if len(audio_chunk) < chunk_size:
                continue  # Skip incomplete chunks

            # Detect beat
            beat_detected = detector.detect_beat(audio_chunk)

            if beat_detected:
                current_time = time.time()
                elapsed = current_time - start_time
                stats = detector.get_stats()

                # Print beat detection
                print(
                    f"{elapsed:8.2f} | {stats['beats_per_minute']:5.1f} | {stats['total_beats']:5d} | "
                    f"{stats['detection_rate']:5.1f} | 🎵 BEAT!"
                )

    try:
        # Open audio stream with larger buffer to prevent overflow
        with sd.InputStream(
            device=usb_device,
            channels=1,
            samplerate=sample_rate,
            blocksize=chunk_size * 4,  # Larger blocksize to prevent overflow
            dtype="float32",
            callback=audio_callback,
            latency="low",
        ):
            print("🎤 Audio stream started...")

            while True:
                # Print periodic stats
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    elapsed = current_time - start_time
                    stats = detector.get_stats()

                    if stats["total_beats"] == 0:
                        print(f"{elapsed:8.1f} | Listening for beats... (try louder music)")

                    last_stats_time = current_time

                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("Beat detection stopped")

        # Final statistics
        stats = detector.get_stats()
        print("\nFinal Statistics:")
        print(f"  Total beats detected: {stats['total_beats']}")
        print(f"  Processing time: {stats['processing_time']:.1f}s")
        print(f"  Final BPM: {stats['beats_per_minute']:.1f}")
        print(f"  Average detection rate: {stats['detection_rate']:.1f} BPM")
        print("=" * 60)


if __name__ == "__main__":
    test_beat_detector()
