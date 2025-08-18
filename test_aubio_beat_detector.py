#!/usr/bin/env python3
"""
Real-Time Beat Detector using Aubio
Aubio is a mature audio analysis library with proven beat detection algorithms
"""

import time
from collections import deque

import aubio
import numpy as np
import sounddevice as sd


class AubioRealTimeBeatDetector:
    def __init__(self, sample_rate=44100, hop_size=512):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.win_s = hop_size * 2  # Window size

        # Initialize aubio tempo tracker (best for EDM/house)
        # Available methods: "specdiff", "energy", "hfc", "complex", "phase", "wphase", "mkl", "kl", "specflux"
        self.tempo = aubio.tempo("specdiff", self.win_s, self.hop_size, sample_rate)

        # Optional: separate onset detector for more control
        self.onset = aubio.onset("energy", self.win_s, self.hop_size, sample_rate)

        # Beat tracking
        self.beat_times = deque(maxlen=32)
        self.current_bpm = 0
        self.total_beats = 0
        self.confidence = 0

        # Statistics
        self.total_frames = 0
        self.last_beat_time = 0

        print("Aubio initialized:")
        print(f"  Sample rate: {sample_rate}Hz")
        print(f"  Hop size: {hop_size} samples ({hop_size/sample_rate*1000:.1f}ms)")
        print(f"  Window size: {self.win_s} samples ({self.win_s/sample_rate*1000:.1f}ms)")
        print("  Tempo method: specdiff (good for EDM/house)")

    def process_frame(self, audio_frame):
        """
        Process audio frame and return True if beat detected
        """
        # Ensure correct data type and size
        if len(audio_frame) != self.hop_size:
            if len(audio_frame) < self.hop_size:
                audio_frame = np.pad(audio_frame, (0, self.hop_size - len(audio_frame)))
            else:
                audio_frame = audio_frame[: self.hop_size]

        audio_frame = audio_frame.astype(np.float32)
        self.total_frames += 1

        # Aubio tempo detection
        beat = self.tempo(audio_frame)
        is_beat = bool(beat[0])

        if is_beat:
            current_time = time.time()

            # Prevent duplicate detections too close together
            if current_time - self.last_beat_time > 0.2:  # 200ms minimum
                self.last_beat_time = current_time
                self.total_beats += 1
                self.beat_times.append(current_time)

                # Calculate BPM from aubio
                aubio_bpm = self.tempo.get_bpm()

                # Also calculate from intervals for comparison
                if len(self.beat_times) >= 4:
                    intervals = np.diff(list(self.beat_times)[-4:])  # Last 4 beats
                    if len(intervals) > 0:
                        valid_intervals = intervals[(intervals > 0.3) & (intervals < 1.5)]
                        if len(valid_intervals) > 0:
                            avg_interval = np.mean(valid_intervals)
                            calculated_bpm = 60.0 / avg_interval

                            # Use aubio BPM if available, otherwise calculated
                            self.current_bpm = aubio_bpm if aubio_bpm > 0 else calculated_bpm
                        else:
                            self.current_bpm = aubio_bpm
                else:
                    self.current_bpm = aubio_bpm

                # Get confidence from aubio
                self.confidence = self.tempo.get_confidence()

                return True

        return False

    def get_stats(self):
        """Get detection statistics"""
        processing_time = self.total_frames * self.hop_size / self.sample_rate
        return {
            "total_beats": self.total_beats,
            "total_frames": self.total_frames,
            "processing_time": processing_time,
            "current_bpm": self.current_bpm,
            "confidence": self.confidence,
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


def test_aubio_beat_detector():
    """Test aubio beat detector"""
    print("=" * 70)
    print("AUBIO REAL-TIME BEAT DETECTOR")
    print("Professional audio analysis library")
    print("=" * 70)

    # Find USB device
    usb_device = find_usb_device()

    # Initialize detector
    sample_rate = 44100
    hop_size = 512  # ~11.6ms per frame
    detector = AubioRealTimeBeatDetector(sample_rate=sample_rate, hop_size=hop_size)

    print("\nPlay EDM/House music and watch for beat detection!")
    print("Aubio should provide more accurate and consistent beat detection.")
    print("Press Ctrl+C to stop\n")

    print("-" * 70)
    print("Time     | BPM   | Conf  | Beats | AvgBPM| Status")
    print("-" * 70)
    print("(BPM=aubio tempo, Conf=confidence, AvgBPM=session average)")

    start_time = time.time()
    last_stats_time = start_time
    stats_interval = 3.0
    overflow_count = 0

    def audio_callback(indata, frames, time_info, status):
        """Audio callback function"""
        nonlocal overflow_count

        if status:
            overflow_count += 1
            if overflow_count % 100 == 0:
                print(f"⚠️  Audio overflow #{overflow_count}")

        # Convert from sounddevice format
        audio_data = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()

        # Process in hop_size chunks
        for i in range(0, len(audio_data), hop_size):
            audio_frame = audio_data[i : i + hop_size]
            if len(audio_frame) < hop_size:
                continue

            # Detect beat
            beat_detected = detector.process_frame(audio_frame)

            if beat_detected:
                current_time = time.time()
                elapsed = current_time - start_time
                stats = detector.get_stats()

                # Print beat detection with aubio confidence
                print(
                    f"{elapsed:8.2f} | {stats['current_bpm']:5.1f} | "
                    f"{stats['confidence']:5.3f} | {stats['total_beats']:5d} | "
                    f"{stats['detection_rate']:5.1f} | 🎵 BEAT!"
                )

    try:
        # Open audio stream
        with sd.InputStream(
            device=usb_device,
            channels=1,
            samplerate=sample_rate,
            blocksize=hop_size * 8,  # Larger buffer to prevent overflow
            dtype="float32",
            callback=audio_callback,
            latency="low",
        ):
            print("🎤 Aubio audio stream started...")

            while True:
                # Print periodic stats
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    elapsed = current_time - start_time
                    stats = detector.get_stats()

                    if stats["total_beats"] == 0:
                        print(f"{elapsed:8.1f} | Listening with aubio... (try music with clear beats)")
                    else:
                        print(
                            f"📊 {elapsed:6.1f}s | Current BPM: {stats['current_bpm']:.1f}, "
                            f"Confidence: {stats['confidence']:.3f}, "
                            f"Total beats: {stats['total_beats']}"
                        )

                    last_stats_time = current_time

                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n" + "-" * 70)
        print("Aubio beat detection stopped")

        # Final statistics
        stats = detector.get_stats()
        print("\nFinal Aubio Statistics:")
        print(f"  Total beats detected: {stats['total_beats']}")
        print(f"  Processing time: {stats['processing_time']:.1f}s")
        print(f"  Final BPM: {stats['current_bpm']:.1f}")
        print(f"  Final confidence: {stats['confidence']:.3f}")
        print(f"  Average detection rate: {stats['detection_rate']:.1f} BPM")
        if overflow_count > 0:
            print(f"  Audio overflows: {overflow_count}")
        print("=" * 70)


if __name__ == "__main__":
    test_aubio_beat_detector()
