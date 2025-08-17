#!/usr/bin/env python3
"""
Audio File Beat Detection Test Program

This program loads an audio file and performs beat detection using BeatNet,
displaying beat information, BPM, and processing progress. This version doesn't
rely on microphone input and can test with known audio files.

Features:
- Load audio files (mp3, wav, etc.) using librosa
- Process audio through BeatNet in real-time simulation
- Display detected beats, BPM, and timing information
- Visual progress indicator
- Comprehensive beat analysis and statistics
"""

import logging
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Audio processing imports
try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.error("librosa not available - cannot load audio files")


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
    is_downbeat: bool  # True for downbeats
    confidence: float  # Detection confidence
    audio_level: float  # Audio level at beat time


@dataclass
class ProcessingStats:
    """Audio processing statistics"""

    total_duration: float = 0.0
    processed_duration: float = 0.0
    total_beats: int = 0
    total_downbeats: int = 0
    current_bpm: float = 120.0
    average_bpm: float = 120.0
    processing_fps: float = 0.0


class BPMCalculator:
    """Calculate BPM from beat timestamps"""

    def __init__(self, history_size=16):
        self.beat_times = deque(maxlen=history_size)
        self.current_bpm = 120.0
        self.all_bpms = []

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
            self.all_bpms.append(self.current_bpm)

        return self.current_bpm

    def get_average_bpm(self):
        """Get overall average BPM"""
        return np.mean(self.all_bpms) if self.all_bpms else 120.0


class AudioLevelAnalyzer:
    """Analyze audio levels for beat context"""

    def __init__(self, sample_rate=22050, window_size=1024):
        self.sample_rate = sample_rate
        self.window_size = window_size

    def get_level_at_time(self, audio_data, timestamp, duration=0.1):
        """Get audio level around a specific timestamp"""
        if len(audio_data) == 0:
            return 0.0

        # Convert timestamp to sample index
        start_sample = int((timestamp - duration / 2) * self.sample_rate)
        end_sample = int((timestamp + duration / 2) * self.sample_rate)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)

        if start_sample >= end_sample:
            return 0.0

        # Calculate RMS level in the window
        window_audio = audio_data[start_sample:end_sample]
        rms = np.sqrt(np.mean(window_audio**2))

        return float(rms)


class AudioFileBeatDetector:
    """Audio file beat detection system"""

    def __init__(self, audio_file_path: str, sample_rate=22050):
        self.audio_file_path = Path(audio_file_path)
        self.sample_rate = sample_rate
        self.running = False

        # Audio data
        self.audio_data = None
        self.audio_duration = 0.0

        # Processing
        self.beatnet = None
        self.level_analyzer = AudioLevelAnalyzer(sample_rate)
        self.bpm_calculator = BPMCalculator()
        self.stats = ProcessingStats()

        # Beat results
        self.beat_events = []

        # Initialize BeatNet
        self._initialize_beatnet()

        # Load audio file
        if LIBROSA_AVAILABLE:
            self._load_audio_file()

    def _initialize_beatnet(self):
        """Initialize BeatNet with compatibility fixes"""
        try:
            logger.info("Setting up BeatNet compatibility...")
            setup_beatnet_compatibility()

            logger.info("Importing BeatNet...")
            from BeatNet.BeatNet import BeatNet

            logger.info("Initializing BeatNet for offline processing...")
            self.beatnet = BeatNet(
                model=1,  # GTZAN model
                mode="online",  # Online mode for file processing
                inference_model="PF",  # Particle filtering
                plot=[],  # No visualization
                thread=False,  # No threading for file processing
                device="cpu",  # CPU processing
            )
            logger.info("✅ BeatNet initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ BeatNet initialization failed: {e}")
            self.beatnet = None
            return False

    def _load_audio_file(self):
        """Load audio file using librosa"""
        try:
            if not self.audio_file_path.exists():
                logger.error(f"Audio file not found: {self.audio_file_path}")
                return False

            logger.info(f"Loading audio file: {self.audio_file_path}")

            # Load audio with librosa
            self.audio_data, actual_sr = librosa.load(str(self.audio_file_path), sr=self.sample_rate, mono=True)

            self.audio_duration = len(self.audio_data) / self.sample_rate
            self.stats.total_duration = self.audio_duration

            logger.info(
                f"✅ Audio loaded: {self.audio_duration:.1f}s, " f"{len(self.audio_data)} samples, {actual_sr}Hz"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load audio file: {e}")
            return False

    def process_audio_file(self):
        """Process the entire audio file for beat detection"""
        if self.beatnet is None:
            logger.error("Cannot process - BeatNet not available")
            return False

        if self.audio_data is None:
            logger.error("Cannot process - no audio data loaded")
            return False

        logger.info("Starting beat detection processing...")

        try:
            # Process entire audio file with BeatNet
            start_time = time.time()

            logger.info("Calling BeatNet.process() on entire audio file...")
            beats = self.beatnet.process(self.audio_data)

            processing_time = time.time() - start_time
            self.stats.processing_fps = self.audio_duration / processing_time

            logger.info(
                f"✅ Processing completed in {processing_time:.2f}s " f"({self.stats.processing_fps:.1f}x real-time)"
            )

            if beats is None or len(beats) == 0:
                logger.warning("No beats detected in audio file")
                return True

            logger.info(f"Found {len(beats)} beat events")

            # Process detected beats
            for beat_timestamp, downbeat_prob in beats:
                self._process_beat_event(beat_timestamp, downbeat_prob)

            # Calculate final statistics
            self.stats.average_bpm = self.bpm_calculator.get_average_bpm()

            logger.info(
                f"✅ Beat detection completed: {self.stats.total_beats} beats, "
                f"{self.stats.total_downbeats} downbeats, "
                f"average BPM: {self.stats.average_bpm:.1f}"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Audio processing failed: {e}", exc_info=True)
            return False

    def _process_beat_event(self, beat_timestamp: float, downbeat_prob: float):
        """Process a detected beat event"""
        # Update statistics
        self.stats.total_beats += 1

        is_downbeat = downbeat_prob > 0.5
        if is_downbeat:
            self.stats.total_downbeats += 1

        # Update BPM calculation
        self.stats.current_bpm = self.bpm_calculator.add_beat(beat_timestamp)

        # Get audio level at beat time
        audio_level = self.level_analyzer.get_level_at_time(self.audio_data, beat_timestamp)

        # Create beat event
        beat_event = BeatEvent(
            timestamp=beat_timestamp, is_downbeat=is_downbeat, confidence=downbeat_prob, audio_level=audio_level
        )

        self.beat_events.append(beat_event)

        # Log beat detection
        beat_type = "DOWNBEAT" if is_downbeat else "beat"
        logger.info(
            f"🎵 {beat_type.upper()}: t={beat_timestamp:.2f}s, "
            f"BPM={self.stats.current_bpm:.1f}, "
            f"conf={downbeat_prob:.2f}, "
            f"level={audio_level:.3f}"
        )

    def display_results(self):
        """Display comprehensive beat detection results"""
        print("\n" + "=" * 80)
        print("🎵 AUDIO FILE BEAT DETECTION RESULTS 🎵")
        print("=" * 80)

        # File information
        print(f"\n📁 FILE INFORMATION:")
        print(f"   File: {self.audio_file_path}")
        print(f"   Duration: {self.stats.total_duration:.1f} seconds")
        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Samples: {len(self.audio_data) if self.audio_data is not None else 0:,}")

        # Processing statistics
        print(f"\n⚡ PROCESSING STATISTICS:")
        print(f"   Processing Speed: {self.stats.processing_fps:.1f}x real-time")
        print(f"   Total Beats: {self.stats.total_beats}")
        print(f"   Total Downbeats: {self.stats.total_downbeats}")
        print(f"   Beat Density: {self.stats.total_beats / self.stats.total_duration:.1f} beats/second")

        # BPM information
        print(f"\n🎼 BPM ANALYSIS:")
        print(f"   Average BPM: {self.stats.average_bpm:.1f}")
        print(f"   Final BPM: {self.stats.current_bpm:.1f}")
        if self.bpm_calculator.all_bpms:
            print(f"   BPM Range: {min(self.bpm_calculator.all_bpms):.1f} - {max(self.bpm_calculator.all_bpms):.1f}")
            print(f"   BPM Std Dev: {np.std(self.bpm_calculator.all_bpms):.1f}")

        # Beat timeline (show first 20 beats)
        print(f"\n🎵 BEAT TIMELINE (first 20 beats):")
        for i, beat in enumerate(self.beat_events[:20]):
            beat_type = "DOWNBEAT" if beat.is_downbeat else "beat    "
            print(
                f"   {i+1:2d}. {beat_type} @ {beat.timestamp:6.2f}s - "
                f"conf:{beat.confidence:.2f} - level:{beat.audio_level:.3f}"
            )

        if len(self.beat_events) > 20:
            print(f"   ... and {len(self.beat_events) - 20} more beats")

        # Timing analysis
        if len(self.beat_events) > 1:
            print(f"\n⏱️  TIMING ANALYSIS:")
            beat_intervals = []
            for i in range(1, len(self.beat_events)):
                interval = self.beat_events[i].timestamp - self.beat_events[i - 1].timestamp
                beat_intervals.append(interval)

            print(f"   Average Beat Interval: {np.mean(beat_intervals):.3f}s")
            print(f"   Beat Interval Range: {min(beat_intervals):.3f}s - {max(beat_intervals):.3f}s")
            print(f"   Beat Timing Consistency: {1.0 - (np.std(beat_intervals) / np.mean(beat_intervals)):.2f}")

        # Downbeat analysis
        downbeats = [beat for beat in self.beat_events if beat.is_downbeat]
        if len(downbeats) > 1:
            print(f"\n🎼 DOWNBEAT ANALYSIS:")
            downbeat_intervals = []
            for i in range(1, len(downbeats)):
                interval = downbeats[i].timestamp - downbeats[i - 1].timestamp
                downbeat_intervals.append(interval)

            print(f"   Average Measure Length: {np.mean(downbeat_intervals):.2f}s")
            print(
                f"   Estimated Time Signature: {np.mean(downbeat_intervals) / (60.0 / self.stats.average_bpm):.1f} beats/measure"
            )

        print("\n" + "=" * 80)


def create_test_audio():
    """Create a simple test audio file with a clear beat pattern"""
    if not LIBROSA_AVAILABLE:
        return None

    import soundfile as sf

    duration = 10.0  # 10 seconds
    sample_rate = 22050
    bpm = 120
    beat_interval = 60.0 / bpm

    # Generate time array
    t = np.linspace(0, duration, int(duration * sample_rate), False)

    # Generate beat clicks
    audio = np.zeros_like(t)
    for beat_time in np.arange(0, duration, beat_interval):
        beat_sample = int(beat_time * sample_rate)
        if beat_sample < len(audio):
            # Add a short click/beep
            click_duration = 0.1
            click_samples = int(click_duration * sample_rate)
            end_sample = min(beat_sample + click_samples, len(audio))

            # Generate sine wave click
            click_t = np.linspace(0, click_duration, end_sample - beat_sample)
            click = np.sin(2 * np.pi * 1000 * click_t) * np.exp(-click_t * 10)
            audio[beat_sample:end_sample] = click

    # Save test file
    test_file = "/tmp/test_beat_pattern.wav"
    sf.write(test_file, audio, sample_rate)
    logger.info(f"Created test audio file: {test_file}")
    return test_file


def main():
    """Main function"""
    print("🎵 Audio File Beat Detection Test Program 🎵")
    print("=" * 60)

    # Check dependencies
    if not LIBROSA_AVAILABLE:
        print("❌ librosa not available - cannot load audio files")
        print("Install with: pip install librosa")
        return 1

    # Get audio file path from command line or create test file
    audio_file = None
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if not Path(audio_file).exists():
            print(f"❌ Audio file not found: {audio_file}")
            return 1
    else:
        try:
            print("No audio file specified, creating test audio...")
            audio_file = create_test_audio()
            if audio_file is None:
                print("❌ Failed to create test audio file")
                return 1
        except Exception as e:
            print(f"❌ Failed to create test audio: {e}")
            return 1

    print(f"🎵 Processing audio file: {audio_file}")

    # Create detector
    try:
        detector = AudioFileBeatDetector(audio_file)
    except Exception as e:
        logger.error(f"Failed to create beat detector: {e}")
        return 1

    # Process audio file
    success = detector.process_audio_file()
    if not success:
        logger.error("Failed to process audio file")
        return 1

    # Display results
    detector.display_results()

    print("\n✅ Audio file beat detection test completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
