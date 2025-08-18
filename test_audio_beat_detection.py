#!/usr/bin/env python3
"""
Real-time Audio Beat Detection Test Program

This program captures live microphone audio and performs real-time beat detection
using BeatNet, displaying beat information, BPM, and audio levels.

Features:
- Live microphone capture using sounddevice (more reliable than PyAudio)
- Captures at 44100Hz and downsamples to 22050Hz for BeatNet
- Real-time beat detection with BeatNet
- Visual display of beat events, BPM, and audio levels
- Handles both beats and downbeats
- Shows audio input levels to verify microphone is working
- Comprehensive error handling and diagnostics
"""

import logging
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# BeatNet compatibility fixes and import
def setup_beatnet_compatibility():
    """Apply Python 3.10+ compatibility fixes for BeatNet/madmom"""
    import collections
    import collections.abc
    import sys

    # Fix collections.MutableSequence issue
    if not hasattr(collections, "MutableSequence"):
        collections.MutableSequence = collections.abc.MutableSequence
    if not hasattr(collections, "Mapping"):
        collections.Mapping = collections.abc.Mapping
    if not hasattr(collections, "Sequence"):
        collections.Sequence = collections.abc.Sequence

    # Monkey patch sys.modules
    sys.modules["collections"].MutableSequence = collections.abc.MutableSequence
    sys.modules["collections"].Mapping = collections.abc.Mapping
    sys.modules["collections"].Sequence = collections.abc.Sequence

    # Fix numpy compatibility
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "complex"):
        np.complex = complex
    if not hasattr(np, "bool"):
        np.bool = bool


@dataclass
class BeatEvent:
    """Beat event information"""

    timestamp: float  # Beat timestamp (seconds)
    system_time: float  # System time when detected
    is_downbeat: bool  # True for downbeats
    confidence: float  # Detection confidence
    audio_level: float  # Current audio level


@dataclass
class AudioStats:
    """Real-time audio statistics"""

    rms_level: float = 0.0
    peak_level: float = 0.0
    total_beats: int = 0
    total_downbeats: int = 0
    current_bpm: float = 120.0
    last_beat_time: float = 0.0


class AudioLevelMonitor:
    """Monitor audio input levels to verify microphone is working"""

    def __init__(self, history_size=50):
        self.rms_history = deque(maxlen=history_size)
        self.peak_history = deque(maxlen=history_size)

    def update(self, audio_data):
        """Update audio levels with new data"""
        if len(audio_data) == 0:
            return 0.0, 0.0

        # Calculate RMS (average level)
        rms = np.sqrt(np.mean(audio_data**2))
        self.rms_history.append(rms)

        # Calculate peak level
        peak = np.max(np.abs(audio_data))
        self.peak_history.append(peak)

        return rms, peak

    def get_average_levels(self):
        """Get smoothed average levels"""
        if not self.rms_history:
            return 0.0, 0.0
        return np.mean(self.rms_history), np.mean(self.peak_history)


class BPMCalculator:
    """Calculate BPM from beat timestamps"""

    def __init__(self, history_size=16):
        self.beat_times = deque(maxlen=history_size)
        self.current_bpm = 120.0

    def add_beat(self, timestamp):
        """Add a new beat timestamp and calculate BPM"""
        self.beat_times.append(timestamp)

        if len(self.beat_times) < 2:
            return self.current_bpm

        # Calculate intervals between beats
        intervals = []
        for i in range(1, len(self.beat_times)):
            interval = self.beat_times[i] - self.beat_times[i - 1]
            # Filter realistic beat intervals (0.3-2.0 seconds)
            if 0.3 <= interval <= 2.0:
                intervals.append(interval)

        if intervals:
            avg_interval = np.mean(intervals)
            self.current_bpm = 60.0 / avg_interval

        return self.current_bpm


class RealTimeAudioBeatDetector:
    """Real-time audio beat detection system using sounddevice"""

    def __init__(self):
        self.capture_rate = 44100  # USB device native rate
        self.beatnet_rate = 22050  # BeatNet expected rate
        self.chunk_duration = 0.1  # 100ms chunks
        self.chunk_size = int(self.capture_rate * self.chunk_duration)
        self.running = False

        # Audio processing
        self.audio_stream = None
        self.audio_buffer = deque(maxlen=self.capture_rate * 2)  # 2 second buffer at 44.1kHz
        self.audio_buffer_22k = deque(maxlen=self.beatnet_rate * 2)  # 2 second buffer at 22.05kHz
        self.level_monitor = AudioLevelMonitor()
        self.bpm_calculator = BPMCalculator()

        # Beat detection
        self.beatnet = None
        self.stats = AudioStats()
        self.beat_events = deque(maxlen=100)  # Recent beat history

        # Threading
        self.audio_thread = None
        self.beat_thread = None
        self.display_thread = None

        # USB device index
        self.usb_device_index = None

        # Initialize components
        self._find_usb_device()
        self._initialize_beatnet()

    def _find_usb_device(self):
        """Find USB Audio Device"""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if "USB Audio" in device["name"] and device["max_input_channels"] > 0:
                self.usb_device_index = i
                logger.info(f"✅ Found USB Audio Device at index {i}")
                logger.info(f"   Name: {device['name']}")
                logger.info(f"   Sample Rate: {device['default_samplerate']}Hz")
                return True

        logger.warning("USB Audio Device not found, will use default")
        return False

    def _initialize_beatnet(self):
        """Initialize BeatNet with compatibility fixes"""
        try:
            logger.info("Setting up BeatNet compatibility...")
            setup_beatnet_compatibility()

            logger.info("Importing BeatNet...")
            from BeatNet.BeatNet import BeatNet

            logger.info("Initializing BeatNet for online processing...")
            self.beatnet = BeatNet(
                model=1,  # GTZAN model
                mode="online",  # Online mode for chunk processing
                inference_model="PF",  # Particle filtering
                plot=[],  # No visualization
                thread=False,  # No threading (we handle it)
                device="cpu",  # CPU processing
            )
            logger.info("✅ BeatNet initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ BeatNet initialization failed: {e}")
            self.beatnet = None
            return False

    def _audio_capture_worker(self):
        """Worker thread for audio capture using blocking reads"""
        logger.info("Audio capture worker started")

        # Set device if USB found
        device_param = None
        if self.usb_device_index is not None:
            device_param = self.usb_device_index
            logger.info(f"Using USB Audio Device {self.usb_device_index} for capture")

        try:
            with sd.InputStream(
                samplerate=self.capture_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                device=device_param,
            ) as stream:
                logger.info(f"Audio stream opened at {self.capture_rate}Hz")

                while self.running:
                    try:
                        # Read audio chunk (blocking)
                        audio_chunk, overflowed = stream.read(self.chunk_size)

                        if overflowed:
                            logger.warning("Audio overflow detected")

                        # Flatten to 1D array
                        audio_chunk = audio_chunk.flatten()

                        # Add to 44.1kHz buffer
                        self.audio_buffer.extend(audio_chunk)

                        # Downsample to 22.05kHz for BeatNet
                        audio_chunk_22k = audio_chunk[::2]  # Take every other sample
                        self.audio_buffer_22k.extend(audio_chunk_22k)

                        # Update audio level monitoring
                        rms, peak = self.level_monitor.update(audio_chunk)
                        self.stats.rms_level = rms
                        self.stats.peak_level = peak

                        # Log occasionally to show we're getting data
                        if np.random.random() < 0.02:  # Log 2% of the time
                            logger.debug(
                                f"Audio: RMS={rms:.6f}, Peak={peak:.6f}, "
                                f"Buffer={len(self.audio_buffer_22k)} samples"
                            )

                    except Exception as e:
                        if self.running:  # Only log if we're still supposed to be running
                            logger.error(f"Audio capture error: {e}")
                        time.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")

        logger.info("Audio capture worker stopped")

    def _beat_detection_worker(self):
        """Worker thread for beat detection"""
        logger.info("Beat detection worker started")

        # Wait for audio buffer to fill
        while self.running and len(self.audio_buffer_22k) < self.beatnet_rate:
            time.sleep(0.1)

        process_count = 0

        while self.running:
            try:
                # Process if we have enough audio (at least 1 second)
                if len(self.audio_buffer_22k) >= self.beatnet_rate and self.beatnet is not None:
                    # Convert buffer to numpy array
                    audio_array = np.array(list(self.audio_buffer_22k), dtype=np.float32)

                    # Process with BeatNet
                    beats = self.beatnet.process(audio_array)

                    process_count += 1
                    if process_count % 50 == 0:  # Log every 50 processes
                        logger.debug(f"BeatNet processed {process_count} chunks")

                    if beats is not None and len(beats) > 0:
                        current_time = time.time()
                        for beat_timestamp, downbeat_prob in beats:
                            self._handle_beat_detection(beat_timestamp, downbeat_prob, current_time)

                time.sleep(0.05)  # Process at ~20Hz

            except Exception as e:
                logger.error(f"Beat detection error: {e}")
                if not self.running:
                    break
                time.sleep(0.1)

        logger.info("Beat detection worker stopped")

    def _handle_beat_detection(self, beat_timestamp, downbeat_prob, system_time):
        """Handle a detected beat event"""
        # Avoid duplicate beats
        if self.stats.last_beat_time > 0 and abs(beat_timestamp - self.stats.last_beat_time) < 0.1:
            return

        # Update statistics
        self.stats.last_beat_time = beat_timestamp
        self.stats.total_beats += 1

        is_downbeat = downbeat_prob > 0.5
        if is_downbeat:
            self.stats.total_downbeats += 1

        # Update BPM calculation
        self.stats.current_bpm = self.bpm_calculator.add_beat(beat_timestamp)

        # Create beat event
        beat_event = BeatEvent(
            timestamp=beat_timestamp,
            system_time=system_time,
            is_downbeat=is_downbeat,
            confidence=downbeat_prob,
            audio_level=self.stats.rms_level,
        )

        self.beat_events.append(beat_event)

        # Log beat detection
        beat_type = "DOWNBEAT" if is_downbeat else "BEAT"
        logger.info(
            f"🎵 {beat_type}: #{self.stats.total_beats} at {beat_timestamp:.2f}s, "
            f"BPM={self.stats.current_bpm:.1f}, conf={downbeat_prob:.2f}"
        )

    def _display_worker(self):
        """Worker thread for real-time display updates"""
        logger.info("Display worker started")

        while self.running:
            try:
                # Clear screen and show current status
                print("\033[2J\033[H")  # Clear screen and move cursor to top
                print("=" * 80)
                print("🎵 REAL-TIME AUDIO BEAT DETECTION TEST 🎵")
                print("=" * 80)
                print()

                # Audio input status
                avg_rms, avg_peak = self.level_monitor.get_average_levels()
                print("🎤 AUDIO INPUT:")
                print(f"   Current Level: RMS={self.stats.rms_level:.6f}, Peak={self.stats.peak_level:.6f}")
                print(f"   Average Level: RMS={avg_rms:.6f}, Peak={avg_peak:.6f}")

                # Audio level bar visualization
                level_bar_length = 50
                # Scale up for visibility (audio is quiet in silent environment)
                rms_bar = "█" * min(int(avg_rms * level_bar_length * 200), level_bar_length)
                peak_bar = "▓" * min(int(avg_peak * level_bar_length * 200), level_bar_length)
                print(f"   RMS:  [{rms_bar:<{level_bar_length}}]")
                print(f"   Peak: [{peak_bar:<{level_bar_length}}]")

                if avg_rms > 0:
                    print("   ✅ Audio capture working! (seeing noise floor values)")
                else:
                    print("   ⚠️ No audio data detected")
                print()

                # Beat detection status
                print("🎵 BEAT DETECTION:")
                print(f"   Total Beats: {self.stats.total_beats}")
                print(f"   Total Downbeats: {self.stats.total_downbeats}")
                print(f"   Current BPM: {self.stats.current_bpm:.1f}")

                if self.stats.last_beat_time > 0:
                    time_since_beat = time.time() - self.stats.last_beat_time
                    print(f"   Last Beat: {time_since_beat:.1f}s ago")
                else:
                    print("   No beats detected yet (normal in silence)")
                print()

                # Recent beat events
                print("🎼 RECENT BEATS (last 10):")
                recent_beats = list(self.beat_events)[-10:]
                for i, beat in enumerate(recent_beats):
                    beat_type = "DOWNBEAT" if beat.is_downbeat else "beat    "
                    time_ago = time.time() - beat.system_time
                    print(
                        f"   {len(recent_beats)-i:2d}. {beat_type} - {time_ago:.1f}s ago - "
                        f"conf:{beat.confidence:.2f} - level:{beat.audio_level:.6f}"
                    )

                if not recent_beats:
                    print("   (No beats detected yet - this is normal in a silent environment)")
                print()

                # System status
                beatnet_status = "✅ Active" if self.beatnet is not None else "❌ Failed"
                audio_status = "✅ Active" if self.running else "❌ Failed"
                device_status = f"USB Device {self.usb_device_index}" if self.usb_device_index else "Default"

                print("🔧 SYSTEM STATUS:")
                print(f"   BeatNet: {beatnet_status}")
                print(f"   Audio Input: {audio_status} ({device_status})")
                print(f"   Capture Rate: {self.capture_rate}Hz → BeatNet Rate: {self.beatnet_rate}Hz")
                print(f"   Buffer Size: {len(self.audio_buffer_22k)} samples")
                print()

                print("Press Ctrl+C to stop...")
                print("=" * 80)

                time.sleep(1.0)  # Update display every second

            except Exception as e:
                logger.error(f"Display error: {e}")
                time.sleep(1.0)

        logger.info("Display worker stopped")

    def start(self):
        """Start real-time beat detection"""
        if self.running:
            logger.warning("Already running")
            return False

        if self.beatnet is None:
            logger.error("Cannot start - BeatNet not available")
            return False

        logger.info("Starting real-time beat detection...")
        self.running = True

        # Start worker threads
        self.audio_thread = threading.Thread(target=self._audio_capture_worker, daemon=True)
        self.beat_thread = threading.Thread(target=self._beat_detection_worker, daemon=True)
        self.display_thread = threading.Thread(target=self._display_worker, daemon=True)

        self.audio_thread.start()
        self.beat_thread.start()
        self.display_thread.start()

        logger.info("✅ Real-time beat detection started")
        return True

    def stop(self):
        """Stop beat detection and clean up"""
        if not self.running:
            return

        logger.info("Stopping real-time beat detection...")
        self.running = False

        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        if self.beat_thread and self.beat_thread.is_alive():
            self.beat_thread.join(timeout=2.0)
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)

        logger.info("✅ Beat detection stopped")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nReceived interrupt signal...")
    global detector
    if "detector" in globals():
        detector.stop()
    sys.exit(0)


def main():
    """Main function"""
    global detector

    print("🎵 Real-Time Audio Beat Detection Test Program 🎵")
    print("=" * 60)
    print("Using sounddevice for audio capture (more reliable)")
    print("=" * 60)

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create detector
    try:
        detector = RealTimeAudioBeatDetector()
    except Exception as e:
        logger.error(f"Failed to create beat detector: {e}")
        return 1

    # Start detection
    if not detector.start():
        logger.error("Failed to start beat detection")
        return 1

    try:
        print("\n🎤 Monitoring microphone input...")
        print("In a silent environment, you'll see low but non-zero audio levels")
        print("(this is the USB microphone's noise floor)")
        print("\nBeatNet may occasionally detect patterns even in noise.")
        print("\nPress Ctrl+C to stop...\n")

        # Keep main thread alive
        while detector.running:
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()

    print("\n✅ Beat detection test completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
