#!/usr/bin/env python3
"""
BuildDrop Detector Testing and Visualization Tool.

Standalone utility for testing the production BuildDropDetector on audio files.
Processes an audio file frame-by-frame using the same detector as production,
and generates a timeline visualization showing all internal signals:
- Spectral centroid and EWMAs
- Spectral flux per band (bass/mid/high/air)
- Flux slopes
- Mid energy ratio (for cut detection)
- Snare roll detection (multiplier and magnitude)
- Buildup intensity, cut events, drop events

Usage:
    python tools/builddrop_detector_test.py [audio_file.wav]

    If no audio file is provided, uses whereyouare.wav from project root.
"""

import argparse
import logging
import sys
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consumer.audio_beat_analyzer import BuildDropConfig, BuildDropDetector

logger = logging.getLogger(__name__)


class BuildDropDetectorTester:
    """Test harness for BuildDropDetector with file processing and visualization."""

    def __init__(self, audio_file: Path):
        """
        Initialize detector tester.

        Args:
            audio_file: Path to WAV file to analyze
        """
        self.audio_file = audio_file
        self.audio_data = None
        self.sample_rate = None

        # Load audio file
        self._load_audio()

        # Create detector with same parameters as production
        self.hop_size = 512  # ~11.6ms at 44.1kHz
        self.detector = BuildDropDetector(
            config=BuildDropConfig(),
            sample_rate=self.sample_rate,
            buf_size=self.hop_size,
        )

        # Storage for frame-by-frame results
        self.results = []

        logger.info(f"Initialized BuildDropDetectorTester for {audio_file}")
        logger.info(f"  Duration: {self.duration:.1f}s")
        logger.info(f"  Sample rate: {self.sample_rate}Hz")
        logger.info(f"  Hop size: {self.hop_size} samples ({self.hop_size / self.sample_rate * 1000:.1f}ms)")

    def _load_audio(self):
        """Load audio file."""
        with wave.open(str(self.audio_file), "rb") as wav_file:
            self.sample_rate = wav_file.getframerate()
            self.channels = wav_file.getnchannels()
            self.total_frames = wav_file.getnframes()
            self.duration = self.total_frames / self.sample_rate

            # Read audio data
            frames_data = wav_file.readframes(self.total_frames)
            sample_width = wav_file.getsampwidth()

            if sample_width == 2:
                audio_array = np.frombuffer(frames_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio_array = np.frombuffer(frames_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            # Convert to mono if needed
            if self.channels > 1:
                audio_array = audio_array.reshape(-1, self.channels)
                audio_array = np.mean(audio_array, axis=1)

            self.audio_data = audio_array

    def run_analysis(self):
        """Process audio through the detector frame by frame."""
        logger.info("=" * 80)
        logger.info("Starting BuildDropDetector analysis")
        logger.info("=" * 80)

        n_frames = len(self.audio_data) // self.hop_size

        # Set a reasonable BPM for snare detection (will be overridden if beats are detected)
        self.detector.set_bpm(128.0)  # Common house/trance BPM

        for i in range(n_frames):
            start = i * self.hop_size
            end = start + self.hop_size
            frame = self.audio_data[start:end]

            # Process frame
            result = self.detector.process_frame(frame)

            # Add timestamp
            result["timestamp"] = start / self.sample_rate
            result["frame_idx"] = i

            self.results.append(result)

            # Log progress every 10 seconds
            if i > 0 and i % int(10 * self.sample_rate / self.hop_size) == 0:
                logger.info(f"  Processed {result['timestamp']:.1f}s / {self.duration:.1f}s")

        # Count events
        cut_count = sum(1 for r in self.results if r["is_cut"])
        drop_count = sum(1 for r in self.results if r["is_drop"])
        buildup_frames = sum(1 for r in self.results if r["in_buildup"])

        logger.info("=" * 80)
        logger.info("Analysis complete")
        logger.info(f"  Frames processed: {len(self.results)}")
        logger.info(f"  Buildup frames: {buildup_frames}")
        logger.info(f"  Cut events: {cut_count}")
        logger.info(f"  Drop events: {drop_count}")
        logger.info("=" * 80)

    def _setup_time_axis(self, ax, duration: float):
        """Configure x-axis with appropriate tick marks."""
        ax.set_xlim(0, duration)

        if duration > 60:
            label_interval = 10.0
        elif duration > 20:
            label_interval = 5.0
        else:
            label_interval = 2.0

        ax.xaxis.set_minor_locator(MultipleLocator(1.0))
        ax.xaxis.set_major_locator(MultipleLocator(label_interval))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x)}"))
        ax.grid(True, which="major", alpha=0.5)
        ax.grid(True, which="minor", alpha=0.2, linestyle=":")
        ax.set_xlabel("Time (seconds)")

    def plot_results(self, output_path: Path):
        """Generate visualization of all detector signals."""
        logger.info("Generating visualization...")

        if not self.results:
            logger.error("No results to plot. Run analysis first.")
            return

        # Extract arrays from results
        timestamps = np.array([r["timestamp"] for r in self.results])

        # Energies
        bass_energy = np.array([r["bass_energy"] for r in self.results])
        mid_energy = np.array([r["mid_energy"] for r in self.results])
        high_energy = np.array([r["high_energy"] for r in self.results])
        air_energy = np.array([r["air_energy"] for r in self.results])

        # Spectral centroid
        spectral_centroid = np.array([r["spectral_centroid"] for r in self.results])

        # Flux
        bass_flux = np.array([r["bass_flux"] for r in self.results])
        mid_flux = np.array([r["mid_flux"] for r in self.results])
        high_flux = np.array([r["high_flux"] for r in self.results])
        air_flux = np.array([r["air_flux"] for r in self.results])

        # EWMA slope input (centroid and flux values after input EWMA)
        ewma_centroid = np.array([r["ewma_slope_input"][0] for r in self.results])
        ewma_bass_flux = np.array([r["ewma_slope_input"][1] for r in self.results])
        ewma_mid_flux = np.array([r["ewma_slope_input"][2] for r in self.results])
        ewma_high_flux = np.array([r["ewma_slope_input"][3] for r in self.results])
        ewma_air_flux = np.array([r["ewma_slope_input"][4] for r in self.results])

        # Slopes (after output EWMA smoothing)
        centroid_slope = np.array([r["ewma_slope_output"][0] for r in self.results])
        bass_slope = np.array([r["ewma_slope_output"][1] for r in self.results])
        mid_slope = np.array([r["ewma_slope_output"][2] for r in self.results])
        high_slope = np.array([r["ewma_slope_output"][3] for r in self.results])
        air_slope = np.array([r["ewma_slope_output"][4] for r in self.results])

        # Mid energy ratio
        mid_energy_ratio = np.array([r["mid_energy_ratio"] for r in self.results])
        ewma_mid_short = np.array([r["ewma_mid_short"] for r in self.results])
        ewma_mid_long = np.array([r["ewma_mid_long"] for r in self.results])

        # Snare detection
        snare_multiplier = np.array([r["snare_roll_multiplier"] for r in self.results])
        snare_magnitude = np.array([r["snare_roll_magnitude"] for r in self.results])

        # Buildup/Cut/Drop
        buildup_intensity = np.array([r["buildup_intensity"] for r in self.results])
        in_buildup = np.array([r["in_buildup"] for r in self.results])
        is_cut = np.array([r["is_cut"] for r in self.results])
        is_drop = np.array([r["is_drop"] for r in self.results])

        # Create figure with subplots
        fig, axes = plt.subplots(12, 1, figsize=(16, 36))

        # Plot 1: Spectrogram (computed separately for reference)
        try:
            import librosa
            import librosa.display

            S = librosa.feature.melspectrogram(
                y=self.audio_data, sr=self.sample_rate, n_fft=2048, hop_length=512, n_mels=128
            )
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(
                S_dB, x_axis="time", y_axis="mel", sr=self.sample_rate, hop_length=512, ax=axes[0], cmap="magma"
            )
            axes[0].set_ylabel("Frequency (Hz)")
            axes[0].set_title(f"Mel Spectrogram ({self.duration:.1f}s)")
        except ImportError:
            axes[0].specgram(self.audio_data, NFFT=2048, Fs=self.sample_rate, noverlap=1024, cmap="magma")
            axes[0].set_ylabel("Frequency (Hz)")
            axes[0].set_title(f"Spectrogram ({self.duration:.1f}s)")
            axes[0].set_ylim(0, 8000)
        self._setup_time_axis(axes[0], self.duration)

        # Plot 2: Band Energies
        axes[1].plot(timestamps, bass_energy, linewidth=0.5, color="red", alpha=0.9, label="Bass (20-250 Hz)")
        axes[1].plot(timestamps, mid_energy, linewidth=0.5, color="orange", alpha=0.9, label="Mid (250-2000 Hz)")
        axes[1].plot(timestamps, high_energy, linewidth=0.5, color="blue", alpha=0.9, label="High (2000-8000 Hz)")
        axes[1].plot(timestamps, air_energy, linewidth=0.5, color="cyan", alpha=0.9, label="Air (8000-16000 Hz)")
        axes[1].set_ylabel("Energy")
        axes[1].set_title("Band Energies (from BuildDropDetector)")
        axes[1].legend(loc="upper left")
        self._setup_time_axis(axes[1], self.duration)

        # Plot 3: Spectral Flux (raw)
        axes[2].plot(timestamps, bass_flux, linewidth=0.5, color="red", alpha=0.9, label="Bass Flux")
        axes[2].plot(timestamps, mid_flux, linewidth=0.5, color="orange", alpha=0.9, label="Mid Flux")
        axes[2].plot(timestamps, high_flux, linewidth=0.5, color="blue", alpha=0.9, label="High Flux")
        axes[2].plot(timestamps, air_flux, linewidth=0.5, color="cyan", alpha=0.9, label="Air Flux")
        axes[2].set_ylabel("Spectral Flux")
        axes[2].set_title("Spectral Flux (half-wave rectified difference)")
        axes[2].legend(loc="upper left")
        self._setup_time_axis(axes[2], self.duration)

        # Plot 4: Spectral Centroid with EWMA
        axes[3].plot(timestamps, spectral_centroid, linewidth=0.3, color="gray", alpha=0.5, label="Raw Centroid")
        axes[3].plot(timestamps, ewma_centroid, linewidth=0.8, color="blue", alpha=0.9, label="EWMA Centroid (1s)")
        axes[3].set_ylabel("Spectral Centroid (Hz)")
        axes[3].set_title("Spectral Centroid: Raw vs 1s EWMA (ewma_slope_input[0])")
        axes[3].legend(loc="upper left")
        self._setup_time_axis(axes[3], self.duration)

        # Plot 5: Flux EWMAs (ewma_slope_input)
        axes[4].plot(timestamps, ewma_bass_flux, linewidth=0.8, color="red", alpha=0.9, label="Bass EWMA")
        axes[4].plot(timestamps, ewma_mid_flux, linewidth=0.8, color="orange", alpha=0.9, label="Mid EWMA")
        axes[4].plot(timestamps, ewma_high_flux, linewidth=0.8, color="blue", alpha=0.9, label="High EWMA")
        axes[4].plot(timestamps, ewma_air_flux, linewidth=0.8, color="cyan", alpha=0.9, label="Air EWMA")
        axes[4].set_ylabel("Flux EWMA")
        axes[4].set_title("Flux EWMAs (ewma_slope_input[1:5], 0.5s half-life)")
        axes[4].legend(loc="upper left")
        self._setup_time_axis(axes[4], self.duration)

        # Plot 6: Centroid Slope
        axes[5].plot(timestamps, centroid_slope, linewidth=0.8, color="purple", alpha=0.9, label="Centroid Slope")
        axes[5].axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        axes[5].set_ylabel("Slope (Hz/s)")
        axes[5].set_title("Centroid Slope (ewma_slope_output[0], 2s EWMA smoothing)")
        axes[5].legend(loc="upper left")
        self._setup_time_axis(axes[5], self.duration)

        # Plot 7: Flux Slopes (ewma_slope_output)
        axes[6].plot(timestamps, bass_slope, linewidth=0.8, color="red", alpha=0.9, label="Bass Slope")
        axes[6].plot(timestamps, mid_slope, linewidth=0.8, color="orange", alpha=0.9, label="Mid Slope")
        axes[6].plot(timestamps, high_slope, linewidth=0.8, color="blue", alpha=0.9, label="High Slope")
        axes[6].plot(timestamps, air_slope, linewidth=0.8, color="cyan", alpha=0.9, label="Air Slope")
        axes[6].axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        axes[6].axhline(20, color="red", linewidth=1, linestyle="--", alpha=0.7, label="Drop threshold (bass>20)")
        axes[6].axhline(5, color="blue", linewidth=1, linestyle="--", alpha=0.5, label="Entry threshold (high/air>5)")
        axes[6].set_ylabel("Flux Slope")
        axes[6].set_title("Flux Slopes (ewma_slope_output[1:5], 1s EWMA smoothing)")
        axes[6].legend(loc="upper left", ncol=2, fontsize=8)
        self._setup_time_axis(axes[6], self.duration)

        # Plot 8: Mid Energy Ratio (cut detection)
        axes[7].plot(
            timestamps, mid_energy_ratio, linewidth=0.8, color="orange", alpha=0.9, label="Mid Ratio (0.25s/4s)"
        )
        axes[7].axhline(0.5, color="cyan", linewidth=2, linestyle="--", alpha=0.7, label="Cut threshold (0.5)")
        axes[7].axhline(1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5, label="Baseline (1.0)")

        # Mark cut events
        cut_times = timestamps[is_cut]
        for ct in cut_times:
            axes[7].axvline(ct, color="cyan", linewidth=2, alpha=0.8)
        if len(cut_times) > 0:
            axes[7].axvline(cut_times[0], color="cyan", linewidth=2, alpha=0.8, label=f"CUT ({len(cut_times)})")

        axes[7].set_ylabel("Mid Energy Ratio")
        axes[7].set_title("Mid Energy Ratio (ewma_mid_short / ewma_mid_long) - Cut Detection")
        axes[7].set_ylim(0, max(2.0, np.max(mid_energy_ratio) * 1.1))
        axes[7].legend(loc="upper left")
        self._setup_time_axis(axes[7], self.duration)

        # Plot 9: Mid Energy EWMAs
        axes[8].plot(timestamps, ewma_mid_short, linewidth=0.8, color="orange", alpha=0.9, label="Mid 0.25s EWMA")
        axes[8].plot(timestamps, ewma_mid_long, linewidth=0.8, color="purple", alpha=0.9, label="Mid 4s EWMA")
        axes[8].set_ylabel("Mid Energy EWMA")
        axes[8].set_title("Mid Energy EWMAs (short=0.25s, long=4s half-life)")
        axes[8].legend(loc="upper left")
        self._setup_time_axis(axes[8], self.duration)

        # Plot 10: Snare Roll Detection
        roll_mask = snare_multiplier > 0
        if np.any(roll_mask):
            t_rolls = timestamps[roll_mask]
            m_rolls = snare_multiplier[roll_mask]
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
            axes[9].scatter(t_rolls, m_rolls, c=colors, s=10, alpha=0.7)

        axes[9].plot(timestamps, snare_multiplier, linewidth=0.5, color="gray", alpha=0.3)
        axes[9].set_ylabel("Roll Multiplier")
        axes[9].set_yticks([0, 2, 4, 8])
        axes[9].set_yticklabels(["none", "2x", "4x", "8x"])
        axes[9].set_ylim(-0.5, 9)
        axes[9].set_title("Snare Roll Detection: Multiplier (autocorrelation at BPM lags)")
        self._setup_time_axis(axes[9], self.duration)

        # Plot 11: Snare Roll Magnitude
        axes[10].fill_between(timestamps, 0, snare_magnitude, color="lightgray", alpha=0.5)
        if np.any(roll_mask):
            mag_rolls = snare_magnitude[roll_mask]
            axes[10].scatter(t_rolls, mag_rolls, c=colors, s=10, alpha=0.7)
        axes[10].axhline(0.3, color="red", linewidth=1.5, linestyle="--", alpha=0.7, label="Entry threshold (0.3)")
        axes[10].axhline(0.1, color="gray", linewidth=0.5, linestyle="--", alpha=0.5, label="Detection threshold (0.1)")
        axes[10].set_ylabel("Autocorr Magnitude")
        axes[10].set_ylim(0, 1.0)
        axes[10].set_title("Snare Roll Magnitude (autocorrelation peak value)")
        axes[10].legend(loc="upper left")
        self._setup_time_axis(axes[10], self.duration)

        # Plot 12: Buildup Intensity with Cut/Drop markers
        # Background: buildup intensity as filled area
        buildup_mask = buildup_intensity > 0
        if np.any(buildup_mask):
            axes[11].fill_between(timestamps, 0, buildup_intensity, color="red", alpha=0.6, label="Buildup Intensity")
            axes[11].plot(timestamps, buildup_intensity, linewidth=1.0, color="darkred", alpha=0.9)

        axes[11].axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

        # Get y-axis limit for marker height
        max_intensity = np.max(buildup_intensity) if np.any(buildup_mask) else 1.0
        marker_height = max(1.0, max_intensity * 1.1)

        # Plot CUT events as vertical cyan lines
        cut_times = timestamps[is_cut]
        if len(cut_times) > 0:
            for ct in cut_times:
                axes[11].axvline(ct, color="cyan", linewidth=2, alpha=0.8)
            axes[11].axvline(cut_times[0], color="cyan", linewidth=2, alpha=0.8, label=f"CUT ({len(cut_times)})")

        # Plot DROP events as vertical green lines
        drop_times = timestamps[is_drop]
        if len(drop_times) > 0:
            for dt in drop_times:
                axes[11].axvline(dt, color="lime", linewidth=3, alpha=0.9)
            axes[11].axvline(drop_times[0], color="lime", linewidth=3, alpha=0.9, label=f"DROP ({len(drop_times)})")

        axes[11].set_ylabel("Buildup Intensity")
        axes[11].set_title("Buildup/Cut/Drop Detection (production BuildDropDetector output)")
        axes[11].set_ylim(0, marker_height)
        axes[11].legend(loc="upper left")
        self._setup_time_axis(axes[11], self.duration)

        # Add statistics text
        cut_count = np.sum(is_cut)
        drop_count = np.sum(is_drop)
        buildup_frames = np.sum(buildup_intensity > 0)
        stats_text = (
            f"Frames: {len(self.results)} | "
            f"Buildup frames: {buildup_frames} | "
            f"Cuts: {cut_count} | "
            f"Drops: {drop_count}"
        )
        fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, bbox={"boxstyle": "round", "facecolor": "wheat"})

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualization saved: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test BuildDropDetector on a WAV file")
    parser.add_argument("audio_file", nargs="?", help="Path to WAV file (default: whereyouare.wav)")
    parser.add_argument("--output", type=str, help="Output path for plot (default: auto-generated)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Determine audio file
    if args.audio_file:
        audio_file = Path(args.audio_file)
    else:
        project_root = Path(__file__).parent.parent
        audio_file = project_root / "whereyouare.wav"

    if not audio_file.exists():
        logger.error(f"Audio file not found: {audio_file}")
        logger.info("Please provide a WAV file path")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = audio_file.parent / f"{audio_file.stem}_builddrop_test.svg"

    # Create tester and run analysis
    tester = BuildDropDetectorTester(audio_file)
    tester.run_analysis()

    # Generate visualization
    tester.plot_results(output_path)

    logger.info("\n" + "=" * 80)
    logger.info("BuildDropDetector test completed successfully")
    logger.info(f"  Input: {audio_file}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
