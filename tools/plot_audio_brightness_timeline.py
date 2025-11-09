#!/usr/bin/env python3
"""
Audio-Reactive Brightness Timeline Visualization.

This tool parses system logs to visualize the relationship between:
- Audio waveform (from file)
- Detected beats (from beat analyzer)
- Brightness boost applied to LEDs (from frame renderer)

The goal is to verify that LED brightness pulses align with detected beats
in real-time, showing the effectiveness of the audio-reactive system.

Usage:
    python tools/plot_audio_brightness_timeline.py logs/prismatron.log \\
        --audio-file audio_capture_test.wav \\
        -o brightness_timeline.png
"""

import argparse
import logging
import re
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioStartEvent:
    """Timeline start marker from audio capture."""

    wall_time: float
    file: str
    sample_rate: int
    chunk_size: int
    chunk_duration_ms: float


@dataclass
class AudioLoopEvent:
    """Audio file loop marker."""

    wall_time: float
    loop_count: int
    file: str


@dataclass
class BeatEvent:
    """Beat detection event."""

    wall_time: float
    audio_ts: float
    intensity: float
    confidence: float
    bpm: float
    is_downbeat: bool


@dataclass
class BrightnessEvent:
    """Brightness boost event."""

    wall_time: float
    multiplier: float


@dataclass
class RendererStateChange:
    """Renderer state change event."""

    wall_time: float
    from_state: str
    to_state: str


class LogParser:
    """Parses structured log events for timeline reconstruction."""

    # Regex patterns for structured log events
    AUDIO_START_PATTERN = re.compile(
        r"AUDIO_START: wall_time=(?P<wall_time>[\d.]+), file=(?P<file>[^,]+), "
        r"sample_rate=(?P<sample_rate>\d+), chunk_size=(?P<chunk_size>\d+), "
        r"chunk_duration_ms=(?P<chunk_duration_ms>[\d.]+)"
    )

    AUDIO_LOOP_PATTERN = re.compile(
        r"AUDIO_LOOP: wall_time=(?P<wall_time>[\d.]+), loop_count=(?P<loop_count>\d+), " r"file=(?P<file>.+)"
    )

    BEAT_DETECTED_PATTERN = re.compile(
        r"BEAT_DETECTED: wall_time=(?P<wall_time>[\d.]+), "
        r"audio_ts=(?P<audio_ts>[\d.]+), "
        r"intensity=(?P<intensity>[\d.]+), "
        r"confidence=(?P<confidence>[\d.]+), "
        r"bpm=(?P<bpm>[\d.]+), "
        r"is_downbeat=(?P<is_downbeat>True|False)"
    )

    BRIGHTNESS_BOOST_PATTERN = re.compile(
        r"BRIGHTNESS_BOOST: wall_time=(?P<wall_time>[\d.]+), " r"multiplier=(?P<multiplier>[\d.]+)"
    )

    RENDERER_STATE_PATTERN = re.compile(
        r"🔄 Renderer state change: RendererState\.(?P<from_state>\w+) -> RendererState\.(?P<to_state>\w+)"
    )

    def __init__(self):
        self.audio_start: Optional[AudioStartEvent] = None
        self.audio_loops: List[AudioLoopEvent] = []
        self.beats: List[BeatEvent] = []
        self.brightness: List[BrightnessEvent] = []
        self.renderer_state_changes: List[RendererStateChange] = []

    def parse_log_file(self, log_path: Path) -> None:
        """Parse log file and extract structured events."""
        logger.info(f"Parsing log file: {log_path}")

        with open(log_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    self._parse_line(line)
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue

        logger.info(
            f"Parsed events: audio_start={self.audio_start is not None}, "
            f"loops={len(self.audio_loops)}, beats={len(self.beats)}, "
            f"brightness_samples={len(self.brightness)}, "
            f"renderer_states={len(self.renderer_state_changes)}"
        )

    def _parse_line(self, line: str) -> None:
        """Parse a single log line for structured events."""
        # Try AUDIO_START (only keep the first one, ignore subsequent loops)
        match = self.AUDIO_START_PATTERN.search(line)
        if match:
            if self.audio_start is None:  # Only use the first AUDIO_START event
                self.audio_start = AudioStartEvent(
                    wall_time=float(match.group("wall_time")),
                    file=match.group("file"),
                    sample_rate=int(match.group("sample_rate")),
                    chunk_size=int(match.group("chunk_size")),
                    chunk_duration_ms=float(match.group("chunk_duration_ms")),
                )
            return

        # Try AUDIO_LOOP
        match = self.AUDIO_LOOP_PATTERN.search(line)
        if match:
            self.audio_loops.append(
                AudioLoopEvent(
                    wall_time=float(match.group("wall_time")),
                    loop_count=int(match.group("loop_count")),
                    file=match.group("file"),
                )
            )
            return

        # Try BEAT_DETECTED
        match = self.BEAT_DETECTED_PATTERN.search(line)
        if match:
            self.beats.append(
                BeatEvent(
                    wall_time=float(match.group("wall_time")),
                    audio_ts=float(match.group("audio_ts")),
                    intensity=float(match.group("intensity")),
                    confidence=float(match.group("confidence")),
                    bpm=float(match.group("bpm")),
                    is_downbeat=match.group("is_downbeat") == "True",
                )
            )
            return

        # Try BRIGHTNESS_BOOST
        match = self.BRIGHTNESS_BOOST_PATTERN.search(line)
        if match:
            self.brightness.append(
                BrightnessEvent(wall_time=float(match.group("wall_time")), multiplier=float(match.group("multiplier")))
            )
            return

        # Try RENDERER_STATE (need to extract timestamp from log line prefix)
        match = self.RENDERER_STATE_PATTERN.search(line)
        if match:
            # Extract timestamp from log line format: "2025-11-06 12:50:38,591 - ..."
            timestamp_match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)", line)
            if timestamp_match:
                from datetime import datetime

                timestamp_str = timestamp_match.group(1)
                millis = int(timestamp_match.group(2))
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                wall_time = dt.timestamp() + millis / 1000.0

                self.renderer_state_changes.append(
                    RendererStateChange(
                        wall_time=wall_time, from_state=match.group("from_state"), to_state=match.group("to_state")
                    )
                )
            return

    def get_timeline_zero(self) -> float:
        """Get wall clock time to use as timeline zero."""
        if self.audio_start:
            return self.audio_start.wall_time
        logger.warning("No AUDIO_START event found, using first event time")
        # Fallback: use earliest event
        all_times = []
        if self.beats:
            all_times.append(min(b.wall_time for b in self.beats))
        if self.brightness:
            all_times.append(min(b.wall_time for b in self.brightness))
        return min(all_times) if all_times else 0.0

    def to_audio_timeline(self, wall_time: float) -> float:
        """Convert wall clock time to audio timeline (seconds from start)."""
        timeline_zero = self.get_timeline_zero()
        return wall_time - timeline_zero

    def get_rendering_window(self) -> Optional[tuple[float, float]]:
        """
        Get the time window when renderer was actively rendering frames.

        Returns:
            Tuple of (start_time, end_time) in audio timeline, or None if no rendering period found
        """
        if not self.renderer_state_changes:
            logger.warning("No renderer state changes found - showing full timeline")
            return None

        # Find when renderer entered PLAYING state
        playing_start = None
        playing_end = None

        for change in self.renderer_state_changes:
            if change.to_state == "PLAYING":
                playing_start = self.to_audio_timeline(change.wall_time)
                logger.info(f"Renderer entered PLAYING state at audio time {playing_start:.1f}s")

            elif change.from_state == "PLAYING" and change.to_state != "PLAYING":
                playing_end = self.to_audio_timeline(change.wall_time)
                logger.info(f"Renderer left PLAYING state at audio time {playing_end:.1f}s")
                break

        # If we found a start but no explicit end, use the last brightness event
        # (brightness logs stop when rendering stops)
        if playing_start is not None and playing_end is None:
            if self.brightness:
                playing_end = self.to_audio_timeline(self.brightness[-1].wall_time)
                logger.info(f"No state change to end rendering, using last brightness event at {playing_end:.1f}s")
            else:
                logger.warning("No brightness events found - cannot determine rendering end")
                return None

        if playing_start is not None and playing_end is not None:
            duration = playing_end - playing_start
            logger.info(f"Rendering window: {playing_start:.1f}s to {playing_end:.1f}s (duration: {duration:.1f}s)")
            return (playing_start, playing_end)

        return None


class AudioFileLoader:
    """Loads audio file for waveform visualization."""

    def __init__(self, audio_path: Path):
        self.audio_path = audio_path
        self.waveform: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.duration: Optional[float] = None

    def load(self) -> None:
        """Load audio file and extract waveform."""
        logger.info(f"Loading audio file: {self.audio_path}")

        with wave.open(str(self.audio_path), "rb") as wav_file:
            self.sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            total_frames = wav_file.getnframes()
            self.duration = total_frames / self.sample_rate

            # Read all frames
            frames_data = wav_file.readframes(total_frames)

            # Convert to numpy array
            if sample_width == 2:
                # 16-bit PCM
                audio_array = np.frombuffer(frames_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                # 32-bit PCM
                audio_array = np.frombuffer(frames_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            # Convert to mono if needed
            if channels > 1:
                audio_array = audio_array.reshape(-1, channels)
                audio_array = np.mean(audio_array, axis=1)

            self.waveform = audio_array

        logger.info(
            f"Audio loaded: {self.sample_rate}Hz, {self.duration:.1f}s, " f"{len(self.waveform)} samples, {channels}ch"
        )


class TimelinePlotter:
    """Generates timeline visualization plot."""

    def __init__(self, parser: LogParser, audio_loader: AudioFileLoader):
        self.parser = parser
        self.audio_loader = audio_loader

    def plot(self, output_path: Path, max_duration: Optional[float] = None) -> None:
        """Generate timeline plot."""
        logger.info(f"Generating timeline plot: {output_path}")

        # Create figure with 4 subplots (very high resolution for detailed analysis)
        fig, axes = plt.subplots(4, 1, figsize=(96, 12))

        # Determine time range - prioritize rendering window if available
        time_start = 0.0
        time_end = None

        if max_duration:
            time_end = max_duration
        else:
            # Try to get rendering window from renderer state changes
            rendering_window = self.parser.get_rendering_window()

            if rendering_window:
                # Use rendering window with some padding
                time_start = max(0.0, rendering_window[0] - 2.0)  # 2s padding before
                time_end = rendering_window[1] + 2.0  # 2s padding after
                logger.info(
                    f"Using rendering window: {rendering_window[0]:.1f}s to {rendering_window[1]:.1f}s "
                    f"(with 2s padding: {time_start:.1f}s to {time_end:.1f}s)"
                )
            else:
                # Fallback: Calculate extent from all logged events
                all_times = []
                if self.parser.brightness:
                    all_times.extend([self.parser.to_audio_timeline(b.wall_time) for b in self.parser.brightness])
                if self.parser.beats:
                    all_times.extend([self.parser.to_audio_timeline(b.wall_time) for b in self.parser.beats])

                if all_times:
                    # Use extent of all events, with padding
                    time_start = max(0.0, min(all_times) - 1.0)
                    time_end = max(all_times) + 1.0
                    logger.info(
                        f"Auto-detected time range: {time_start:.1f} to {time_end:.1f}s "
                        f"based on {len(all_times)} events"
                    )
                else:
                    # Fallback to audio duration
                    time_end = self.audio_loader.duration if self.audio_loader.duration else 30.0
                    logger.warning(f"No events found, using audio duration: {time_start:.1f}s to {time_end:.1f}s")

        # Plot 1: Audio waveform with beat markers
        self._plot_waveform_with_beats(axes[0], time_start, time_end)

        # Plot 2: Brightness multiplier over time
        self._plot_brightness_timeline(axes[1], time_start, time_end)

        # Plot 3: Beat intensity over time
        self._plot_beat_intensity(axes[2], time_start, time_end)

        # Plot 4: Audio loops and timeline events
        self._plot_timeline_events(axes[3], time_start, time_end)

        # Add statistics
        self._add_statistics(fig)

        # Format and save
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Plot saved: {output_path}")

    def _plot_waveform_with_beats(self, ax, time_start: float, time_end: float) -> None:
        """Plot audio spectrogram with beat markers (tiled horizontally to show loops side-by-side)."""
        if self.audio_loader.waveform is None:
            ax.text(0.5, 0.5, "No audio waveform available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Audio Spectrogram with Beat Markers")
            return

        audio_duration = self.audio_loader.duration

        # Calculate how many tiles we need to cover the visible range
        num_tiles = int(np.ceil(time_end / audio_duration))
        tiled_duration = num_tiles * audio_duration

        logger.info(
            f"Creating spectrogram: audio_duration={audio_duration:.1f}s, "
            f"visible_range={time_start:.1f}s-{time_end:.1f}s, tiles={num_tiles}"
        )

        # Create spectrogram ONCE for the original audio
        # Use smaller NFFT for better time resolution to see individual beats
        NFFT = 512  # FFT window size - balance between time and frequency resolution
        noverlap = NFFT * 3 // 4  # 75% overlap for smooth visualization

        from matplotlib import mlab

        spectrum, freqs, times = mlab.specgram(
            self.audio_loader.waveform,
            NFFT=NFFT,
            Fs=self.audio_loader.sample_rate,
            noverlap=noverlap,
            scale_by_freq=False,
        )

        # Calculate percentile-based scaling for better contrast
        spectrum_flat = spectrum.flatten()
        vmin_percentile = np.percentile(spectrum_flat, 10)  # 10th percentile
        vmax_percentile = np.percentile(spectrum_flat, 99.5)  # 99.5th percentile

        logger.info(
            f"Spectrogram: time_range=[{times[0]:.1f}, {times[-1]:.1f}]s, "
            f"freq_range=[{freqs[0]:.1f}, {freqs[-1]:.1f}]Hz, "
            f"scaling: vmin={vmin_percentile:.6f}, vmax={vmax_percentile:.6f}, "
            f"shape={spectrum.shape}"
        )

        # Tile the spectrogram horizontally by plotting it multiple times
        for tile_idx in range(num_tiles):
            offset = tile_idx * audio_duration
            # Shift times for this tile
            tile_times = times + offset

            ax.pcolormesh(
                tile_times,
                freqs,
                spectrum,
                cmap="hot",
                vmin=vmin_percentile,
                vmax=vmax_percentile,
                shading="gouraud",  # Smooth interpolated shading (not blocky)
            )

        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Audio Spectrogram with Beat Markers ({num_tiles} × {audio_duration:.1f}s loops)")
        ax.set_xlim(time_start, time_end)  # Crop to visible range
        ax.set_ylim(0, 2000)  # Focus on lower frequencies where bass/beats are (0-2kHz)
        ax.grid(True, alpha=0.2, color="white", linewidth=0.5)

        # Add fine time ticks (major every 1s, minor every 0.1s)
        from matplotlib.ticker import MultipleLocator

        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis="x", which="minor", length=3, color="white")

        # Add beat markers at their actual timeline positions (no wrapping)
        if self.parser.beats:
            beat_times = [b.audio_ts for b in self.parser.beats]
            beat_intensities = [b.intensity for b in self.parser.beats]
            downbeat_mask = [b.is_downbeat for b in self.parser.beats]

            # Filter to visible range
            visible_beats = [
                (t, i, d) for t, i, d in zip(beat_times, beat_intensities, downbeat_mask) if time_start <= t <= time_end
            ]

            for time, intensity, is_downbeat in visible_beats:
                if is_downbeat:
                    # Lime with black outline for visibility
                    ax.axvline(time, color="black", alpha=0.8, linewidth=3.0, linestyle="-")
                    ax.axvline(time, color="lime", alpha=0.9, linewidth=2.0, linestyle="-")
                else:
                    # White/yellow for regular beats with black outline
                    ax.axvline(time, color="black", alpha=0.6, linewidth=2.5, linestyle="-")
                    ax.axvline(time, color="yellow", alpha=0.5 + intensity * 0.4, linewidth=1.5, linestyle="-")

            # Legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color="yellow", linewidth=1.5, alpha=0.8, label="Beat occurrence (audio_ts)"),
                Line2D([0], [0], color="lime", linewidth=2.0, label="Downbeats"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", facecolor="black", edgecolor="white", framealpha=0.7)

    def _plot_brightness_timeline(self, ax, time_start: float, time_end: float) -> None:
        """Plot brightness multiplier over time."""
        if not self.parser.brightness:
            ax.text(0.5, 0.5, "No brightness data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Brightness Multiplier Over Time")
            return

        # Convert to audio timeline (no wrapping)
        times = [self.parser.to_audio_timeline(b.wall_time) for b in self.parser.brightness]
        multipliers = [b.multiplier for b in self.parser.brightness]

        # Plot as line with markers
        ax.plot(times, multipliers, linewidth=1.5, color="orange", marker=".", markersize=2, label="Brightness")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Baseline (1.0x)")

        # Add beat markers - show BOTH occurrence and detection times
        if self.parser.beats:
            # Beat occurrence time (audio_ts from aubio frame counting)
            beat_times_occurred = [b.audio_ts for b in self.parser.beats]
            visible_occurred = [t for t in beat_times_occurred if time_start <= t <= time_end]

            # Beat detection time (wall_time converted to audio timeline)
            beat_times_detected = [self.parser.to_audio_timeline(b.wall_time) for b in self.parser.beats]
            visible_detected = [t for t in beat_times_detected if time_start <= t <= time_end]

            # Plot occurrence time (yellow, solid) - aligns with audio stream
            for beat_time in visible_occurred:
                ax.axvline(beat_time, color="yellow", alpha=0.5, linewidth=1.5, linestyle="-")

            # Plot detection time (red, dashed) - aligns with wall clock timeline
            for beat_time in visible_detected:
                ax.axvline(beat_time, color="red", alpha=0.4, linewidth=1, linestyle="--")

            # Add to legend
            from matplotlib.lines import Line2D

            beat_occurred_line = Line2D(
                [0], [0], color="yellow", linestyle="-", linewidth=1.5, alpha=0.5, label="Beat occurrence (audio_ts)"
            )
            beat_detected_line = Line2D(
                [0], [0], color="red", linestyle="--", linewidth=1, alpha=0.4, label="Beat detection (wall_time)"
            )
            handles, labels = ax.get_legend_handles_labels()
            handles.extend([beat_occurred_line, beat_detected_line])
            labels.extend(["Beat occurrence (audio_ts)", "Beat detection (wall_time)"])
            ax.legend(handles=handles, labels=labels, loc="upper right")
        else:
            ax.legend(loc="upper right")

        ax.set_ylabel("Brightness Multiplier")
        ax.set_title("LED Brightness Boost Over Time (with beat markers)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_start, time_end)  # Crop to rendering window

        # Add fine time ticks (major every 1s, minor every 0.1s)
        from matplotlib.ticker import MultipleLocator

        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))

        # Set y-axis to show boost range
        if multipliers:
            max_mult = max(multipliers)
            ax.set_ylim(0.9, max(2.0, max_mult * 1.1))

    def _plot_beat_intensity(self, ax, time_start: float, time_end: float) -> None:
        """Plot beat intensity and confidence over time."""
        if not self.parser.beats:
            ax.text(0.5, 0.5, "No beat data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Beat Intensity Over Time")
            return

        # Use audio_ts (when beat occurred) not wall_time (when detected)
        times = [b.audio_ts for b in self.parser.beats]
        intensities = [b.intensity for b in self.parser.beats]
        confidences = [b.confidence for b in self.parser.beats]

        # Plot intensity
        ax.scatter(
            times,
            intensities,
            c="red",
            s=50,
            alpha=0.7,
            edgecolors="darkred",
            linewidth=0.5,
            label="Intensity (RMS energy)",
        )
        ax.plot(times, intensities, linewidth=1.5, color="red", alpha=0.4)

        # Plot confidence
        ax.scatter(
            times,
            confidences,
            c="blue",
            s=50,
            alpha=0.7,
            edgecolors="darkblue",
            linewidth=0.5,
            label="Confidence (timing)",
        )
        ax.plot(times, confidences, linewidth=1.5, color="blue", alpha=0.4)

        ax.set_ylabel("Value (0.0 - 1.0)")
        ax.set_title(f"Beat Intensity (energy) and Confidence (timing) Over Time ({time_end - time_start:.1f}s window)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_start, time_end)  # Crop to rendering window
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right")

        # Add fine time ticks (major every 1s, minor every 0.1s)
        from matplotlib.ticker import MultipleLocator

        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    def _plot_timeline_events(self, ax, time_start: float, time_end: float) -> None:
        """Plot timeline events (loops, start markers)."""
        audio_duration = self.audio_loader.duration

        ax.set_xlabel("Time (seconds)")
        ax.set_title("Timeline Events (Audio Start and Loop Markers)")
        ax.set_xlim(time_start, time_end)  # Crop to rendering window
        ax.set_ylim(0, 1)
        ax.set_yticks([])  # Remove y-axis ticks
        ax.spines["left"].set_visible(False)  # Hide left spine

        # Add fine time ticks (major every 1s, minor every 0.1s)
        from matplotlib.ticker import MultipleLocator

        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))

        # Plot audio start (only if within visible range)
        if self.parser.audio_start and time_start <= 0 <= time_end:
            ax.axvline(0, color="green", linewidth=2, label="Audio Start")
            ax.text(0.1, 0.5, "Audio Start", va="center", ha="left", fontsize=9, color="green", fontweight="bold")

        # Plot loop boundaries at every audio_duration interval
        num_visible_loops = int(np.ceil(time_end / audio_duration))
        for loop_idx in range(num_visible_loops + 1):
            loop_time = loop_idx * audio_duration
            if time_start <= loop_time <= time_end:
                ax.axvline(loop_time, color="purple", linewidth=1.5, alpha=0.7, linestyle="--")
                # Alternate label positions to avoid overlap
                y_pos = 0.7 if loop_idx % 2 == 0 else 0.3
                ax.text(loop_time + 0.2, y_pos, f"Loop #{loop_idx}", va="center", ha="left", fontsize=8, color="purple")

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = []
        if self.parser.audio_start:
            legend_elements.append(Line2D([0], [0], color="green", linewidth=2, label="Audio Start"))
        legend_elements.append(Line2D([0], [0], color="purple", linewidth=1.5, linestyle="--", label="Loop Boundary"))
        ax.legend(handles=legend_elements, loc="upper right")

        ax.grid(True, alpha=0.3, axis="x")

    def _add_statistics(self, fig) -> None:
        """Add statistics text to figure."""
        stats_parts = []

        # Beat statistics
        if self.parser.beats:
            stats_parts.append(f"Beats detected: {len(self.parser.beats)}")
            avg_intensity = np.mean([b.intensity for b in self.parser.beats])
            avg_confidence = np.mean([b.confidence for b in self.parser.beats])
            stats_parts.append(f"Avg intensity: {avg_intensity:.2f}")
            stats_parts.append(f"Avg confidence: {avg_confidence:.2f}")

        # Brightness statistics
        if self.parser.brightness:
            multipliers = [b.multiplier for b in self.parser.brightness]
            boosted_count = sum(1 for m in multipliers if m > 1.01)
            boost_percentage = (boosted_count / len(multipliers)) * 100
            avg_boost = np.mean([m for m in multipliers if m > 1.01]) if boosted_count > 0 else 1.0
            stats_parts.append(f"Brightness samples: {len(self.parser.brightness)}")
            stats_parts.append(f"Boosted frames: {boosted_count} ({boost_percentage:.1f}%)")
            stats_parts.append(f"Avg boost: {avg_boost:.3f}x")

        # Audio loop statistics
        if self.parser.audio_loops:
            stats_parts.append(f"Audio loops: {len(self.parser.audio_loops)}")

        stats_text = "  |  ".join(stats_parts)
        fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, bbox={"boxstyle": "round", "facecolor": "wheat"})


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize audio-reactive brightness timeline from system logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python tools/plot_audio_brightness_timeline.py logs/prismatron.log \\
        --audio-file audio_capture_test.wav \\
        -o brightness_timeline.png
        """,
    )

    parser.add_argument("log_file", type=Path, help="Path to log file with structured events")
    parser.add_argument("--audio-file", type=Path, required=True, help="Path to audio file (WAV) used during testing")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output plot path (default: auto-generated)")
    parser.add_argument(
        "--max-duration", type=float, default=None, help="Maximum duration to plot in seconds (default: auto)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Validate inputs
    if not args.log_file.exists():
        logger.error(f"Log file not found: {args.log_file}")
        return 1

    if not args.audio_file.exists():
        logger.error(f"Audio file not found: {args.audio_file}")
        return 1

    # Determine output path
    if args.output is None:
        args.output = args.log_file.parent / f"{args.log_file.stem}_brightness_timeline.png"

    try:
        # Parse logs
        log_parser = LogParser()
        log_parser.parse_log_file(args.log_file)

        # Validate we have data
        if log_parser.audio_start is None:
            logger.warning("No AUDIO_START event found in logs - timeline alignment may be inaccurate")

        if not log_parser.beats:
            logger.warning("No BEAT_DETECTED events found in logs")

        if not log_parser.brightness:
            logger.error("No BRIGHTNESS_BOOST events found in logs - cannot generate timeline")
            return 1

        # Load audio file
        audio_loader = AudioFileLoader(args.audio_file)
        audio_loader.load()

        # Generate plot
        plotter = TimelinePlotter(log_parser, audio_loader)
        plotter.plot(args.output, max_duration=args.max_duration)

        logger.info("=" * 80)
        logger.info("Timeline visualization completed successfully")
        logger.info(f"  Log file: {args.log_file}")
        logger.info(f"  Audio file: {args.audio_file}")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Events: {len(log_parser.beats)} beats, {len(log_parser.brightness)} brightness samples")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Failed to generate timeline plot: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
