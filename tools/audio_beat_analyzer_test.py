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

from src.consumer.audio_beat_analyzer import AudioBeatAnalyzer, BeatEvent
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

        # Create beat analyzer
        analyzer = AudioBeatAnalyzer(beat_callback=self.beat_callback, audio_config=audio_config)

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

    def plot_results(self, output_path: Path):
        """
        Plot beat detection results.

        Args:
            output_path: Output path for plot
        """
        logger.info("Generating visualization...")

        # Create figure with subplots (added 5th subplot for intensity)
        fig, axes = plt.subplots(5, 1, figsize=(16, 14))

        # Time axis for audio
        time_axis = np.linspace(0, self.duration, len(self.audio_data))

        # Plot 1: Full waveform with beat markers
        axes[0].plot(time_axis, self.audio_data, linewidth=0.3, color="blue", alpha=0.5)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(f"Audio Waveform with Beat Markers ({self.duration:.1f}s)")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, self.duration)

        # Add beat markers with intensity-based coloring
        if self.beat_events:
            beat_times = [b["timestamp"] for b in self.beat_events]
            beat_intensities = [b["intensity"] for b in self.beat_events]
            downbeat_mask = [b["is_downbeat"] for b in self.beat_events]

            # Create a colormap for intensity (low=yellow, high=red)
            import matplotlib.cm as cm

            # Plot regular beats with intensity-based color
            for i, (time, intensity, is_downbeat) in enumerate(zip(beat_times, beat_intensities, downbeat_mask)):
                if is_downbeat:
                    # Downbeats in green with intensity-based alpha
                    axes[0].vlines(time, -1, 1, colors="green", alpha=0.5 + intensity * 0.5, linewidth=2.0)
                else:
                    # Regular beats with intensity-based color (yellow to red)
                    color = cm.YlOrRd(intensity)
                    axes[0].vlines(time, -1, 1, colors=color, alpha=0.7, linewidth=1.0)

            # Add legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color="red", linewidth=1.5, label="Beats (color = intensity)"),
                Line2D([0], [0], color="green", linewidth=2.0, label="Downbeats"),
            ]
            axes[0].legend(handles=legend_elements, loc="upper right")

        # Plot 2: Zoomed waveform (first 10 seconds)
        zoom_duration = min(10.0, self.duration)
        zoom_samples = int(zoom_duration * self.sample_rate)
        axes[1].plot(time_axis[:zoom_samples], self.audio_data[:zoom_samples], linewidth=0.5, color="blue")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_title(f"Audio Waveform - First {zoom_duration:.1f}s (Detail)")
        axes[1].grid(True, alpha=0.3)

        # Add beat markers to zoom with intensity
        if self.beat_events:
            import matplotlib.cm as cm

            zoom_beats = [b for b in self.beat_events if b["timestamp"] <= zoom_duration]

            for beat in zoom_beats:
                if beat["is_downbeat"]:
                    axes[1].vlines(
                        beat["timestamp"],
                        -1,
                        1,
                        colors="green",
                        alpha=0.5 + beat["intensity"] * 0.5,
                        linewidth=2.5,
                    )
                else:
                    color = cm.YlOrRd(beat["intensity"])
                    axes[1].vlines(beat["timestamp"], -1, 1, colors=color, alpha=0.8, linewidth=1.5)

        # Plot 3: BPM over time
        time_points, bpm_values = self.calculate_bpm_timeline()
        if len(time_points) > 0:
            # Filter out zero values for cleaner plot
            valid_mask = bpm_values > 0
            axes[2].plot(time_points[valid_mask], bpm_values[valid_mask], linewidth=2, color="purple", marker="o")
            axes[2].set_ylabel("BPM")
            axes[2].set_title("Tempo (BPM) Over Time")
            axes[2].grid(True, alpha=0.3)
            axes[2].set_xlim(0, self.duration)
            if np.any(valid_mask):
                axes[2].set_ylim(0, max(200, np.max(bpm_values[valid_mask]) * 1.1))
        else:
            axes[2].text(
                0.5,
                0.5,
                "Insufficient beats for BPM calculation",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )
            axes[2].set_ylabel("BPM")
            axes[2].set_title("Tempo (BPM) Over Time")

        # Plot 4: Beat intensity over time
        if self.beat_events:
            beat_times = [b["timestamp"] for b in self.beat_events]
            beat_intensities = [b["intensity"] for b in self.beat_events]

            # Create scatter plot with intensity-based colors
            import matplotlib.cm as cm

            colors = [cm.YlOrRd(intensity) for intensity in beat_intensities]
            axes[3].scatter(beat_times, beat_intensities, c=colors, s=50, alpha=0.8, edgecolors="black", linewidth=0.5)
            axes[3].plot(beat_times, beat_intensities, linewidth=1, color="gray", alpha=0.5, linestyle="--")
            axes[3].set_ylabel("Beat Intensity")
            axes[3].set_title("Beat Intensity Over Time")
            axes[3].grid(True, alpha=0.3)
            axes[3].set_xlim(0, self.duration)
            axes[3].set_ylim(0, 1.1)
        else:
            axes[3].text(0.5, 0.5, "No beats detected", ha="center", va="center", transform=axes[3].transAxes)
            axes[3].set_ylabel("Beat Intensity")
            axes[3].set_title("Beat Intensity Over Time")

        # Plot 5: RMS envelope with beat markers
        window_size = int(self.sample_rate * 0.05)  # 50ms windows
        rms_envelope = self._calculate_rms_envelope(self.audio_data, window_size)
        rms_time = np.linspace(0, self.duration, len(rms_envelope))
        axes[4].plot(rms_time, rms_envelope, linewidth=1.5, color="orange")
        axes[4].set_xlabel("Time (seconds)")
        axes[4].set_ylabel("RMS Amplitude")
        axes[4].set_title("RMS Envelope with Beat Markers")
        axes[4].grid(True, alpha=0.3)
        axes[4].set_xlim(0, self.duration)

        # Add beat markers to RMS
        if self.beat_events:
            beat_times = [b["timestamp"] for b in self.beat_events]
            axes[4].vlines(beat_times, 0, np.max(rms_envelope), colors="red", alpha=0.4, linewidth=0.8)

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

        fig.text(0.5, 0.02, stats_text, ha="center", fontsize=11, bbox={"boxstyle": "round", "facecolor": "wheat"})

        plt.tight_layout(rect=[0, 0.03, 1, 1])
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
    parser.add_argument("audio_file", nargs="?", help="Path to WAV file (default: audio_capture_test.wav)")
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
        audio_file = project_root / "audio_capture_test.wav"

    if not audio_file.exists():
        logger.error(f"Audio file not found: {audio_file}")
        logger.info("Please provide a WAV file or run tools/audio_capture_test.py to generate one")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = audio_file.parent / f"{audio_file.stem}_beat_analysis.png"

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
