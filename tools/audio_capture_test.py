#!/usr/bin/env python3
"""
Audio Capture Testing and Debugging Tool.

Standalone utility for testing audio capture functionality including:
- Device detection and listing
- Live audio recording from USB microphone
- File-based audio playback (WAV files)
- PCM file writing
- Waveform visualization

Usage:
    python tools/audio_capture_test.py
"""

import logging
import os
import sys
import time
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.consumer.audio_capture import AudioCapture, AudioConfig

logger = logging.getLogger(__name__)


class TestAudioCapture:
    """Test suite for AudioCapture utility"""

    def test_initialization(self):
        """Test AudioCapture initialization"""
        config = AudioConfig(sample_rate=44100, channels=1, chunk_size=2048)
        capture = AudioCapture(config)

        assert capture.config.sample_rate == 44100
        assert capture.config.channels == 1
        assert capture.config.chunk_size == 2048
        assert not capture.is_capturing

    def test_list_devices(self):
        """Test device listing"""
        capture = AudioCapture()
        devices = capture.list_devices()

        logger.info(f"Found {len(devices)} audio input devices")
        for device in devices:
            logger.info(
                f"  [{device['index']}] {device['name']} - " f"{device['channels']}ch @ {device['sample_rate']}Hz"
            )

        assert isinstance(devices, list)

    def test_start_stop_capture(self):
        """Test starting and stopping audio capture"""
        capture = AudioCapture()

        # Start capture
        assert capture.start_capture()
        assert capture.is_capturing

        # Wait a bit for some data
        time.sleep(1.0)

        # Stop capture
        capture.stop_capture()
        assert not capture.is_capturing

        # Check statistics
        stats = capture.get_statistics()
        assert stats["total_chunks"] > 0
        assert stats["total_samples"] > 0

    def test_audio_callback(self):
        """Test audio callback functionality"""
        callback_count = [0]
        chunks_received = []

        def audio_callback(chunk, timestamp):
            callback_count[0] += 1
            chunks_received.append((chunk.copy(), timestamp))

        config = AudioConfig(chunk_size=1024)
        capture = AudioCapture(config, audio_callback=audio_callback)

        capture.start_capture()
        time.sleep(2.0)
        capture.stop_capture()

        assert callback_count[0] > 0
        assert len(chunks_received) > 0
        logger.info(f"Received {callback_count[0]} audio callbacks")

    def test_read_chunk(self):
        """Test reading chunks from queue"""
        capture = AudioCapture()
        capture.start_capture()

        # Read several chunks
        chunks = []
        for _ in range(5):
            chunk_data = capture.read_chunk(timeout=1.0)
            if chunk_data:
                audio_chunk, timestamp = chunk_data
                chunks.append(audio_chunk)
                assert isinstance(audio_chunk, np.ndarray)
                assert len(audio_chunk) > 0

        capture.stop_capture()

        assert len(chunks) > 0
        logger.info(f"Successfully read {len(chunks)} chunks")


class TestFileModeAudioCapture:
    """Tests for file-based audio playback mode"""

    def test_file_mode_playback(self, tmp_path):
        """Test audio capture in file mode (playback from WAV file)"""
        logger.info("=" * 80)
        logger.info("Testing file mode audio capture")
        logger.info("=" * 80)

        # First, create a test WAV file using live capture
        test_wav = tmp_path / "test_audio.wav"

        # Use the previously generated test file if it exists
        project_root = Path(__file__).parent.parent
        existing_wav = project_root / "audio_capture_test.wav"

        if existing_wav.exists():
            logger.info(f"Using existing test file: {existing_wav}")
            test_wav = existing_wav
        else:
            logger.info("Existing test file not found, skipping file mode test")
            pytest.skip("No test audio file available")

        # Storage for captured audio chunks
        audio_chunks_received = []

        def file_callback(chunk, timestamp):
            audio_chunks_received.append(chunk.copy())
            if len(audio_chunks_received) % 50 == 0:
                logger.info(f"  Received {len(audio_chunks_received)} chunks from file")

        # Create AudioCapture in file mode
        config = AudioConfig(
            sample_rate=44100, channels=1, chunk_size=1024, file_path=str(test_wav), playback_speed=10.0
        )  # 10x speed for faster testing

        capture = AudioCapture(config, audio_callback=file_callback)

        logger.info("\nStarting file playback (10x speed)...")
        capture.start_capture()

        # Run for 5 seconds (should play ~50 seconds of audio at 10x speed)
        time.sleep(5)

        capture.stop_capture()

        # Get statistics
        stats = capture.get_statistics()
        logger.info("\n" + "=" * 80)
        logger.info("File Playback Statistics:")
        logger.info(f"  Duration: {stats['duration']:.1f}s")
        logger.info(f"  Chunks processed: {stats['total_chunks']}")
        logger.info(f"  Samples processed: {stats['total_samples']}")
        logger.info(f"  Chunks in callback: {len(audio_chunks_received)}")
        logger.info("=" * 80)

        # Verify we got data
        assert len(audio_chunks_received) > 0, "No audio chunks received from file"
        assert stats["total_samples"] > 0, "No audio samples processed from file"

        logger.info("\n✓ File mode audio capture test completed successfully")

    def test_file_mode_realtime_playback(self, tmp_path):
        """Test audio capture in file mode with realtime playback speed"""
        logger.info("=" * 80)
        logger.info("Testing file mode audio capture (realtime speed)")
        logger.info("=" * 80)

        # Use the previously generated test file
        project_root = Path(__file__).parent.parent
        test_wav = project_root / "audio_capture_test.wav"

        if not test_wav.exists():
            logger.info("Test audio file not found, skipping realtime test")
            pytest.skip("No test audio file available")

        # Storage for captured audio chunks
        audio_chunks_received = []

        def file_callback(chunk, timestamp):
            audio_chunks_received.append(chunk.copy())

        # Create AudioCapture in file mode with realtime playback
        config = AudioConfig(
            sample_rate=44100, channels=1, chunk_size=1024, file_path=str(test_wav), playback_speed=1.0  # Realtime
        )

        capture = AudioCapture(config, audio_callback=file_callback)

        logger.info("\nStarting file playback (realtime speed)...")
        start_time = time.time()
        capture.start_capture()

        # Run for 3 seconds
        time.sleep(3)

        capture.stop_capture()
        elapsed = time.time() - start_time

        # Get statistics
        stats = capture.get_statistics()
        logger.info("\n" + "=" * 80)
        logger.info("Realtime File Playback Statistics:")
        logger.info(f"  Wall clock time: {elapsed:.1f}s")
        logger.info(f"  Audio duration processed: {stats['total_samples']/44100:.1f}s")
        logger.info(f"  Chunks processed: {stats['total_chunks']}")
        logger.info(f"  Samples processed: {stats['total_samples']}")
        logger.info("=" * 80)

        # Verify timing is approximately correct (should process ~3 seconds of audio in ~3 seconds)
        expected_samples = int(44100 * 3)  # 3 seconds at 44.1kHz
        sample_tolerance = 44100 * 0.5  # 0.5 second tolerance

        assert len(audio_chunks_received) > 0, "No audio chunks received from file"
        assert (
            abs(stats["total_samples"] - expected_samples) < sample_tolerance
        ), f"Timing mismatch: expected ~{expected_samples} samples, got {stats['total_samples']}"

        logger.info("\n✓ Realtime file mode audio capture test completed successfully")


class TestAudioRecordingAndVisualization:
    """Tests for 30-second audio recording with PCM writing and waveform plotting"""

    @pytest.mark.slow
    def test_30s_recording_with_pcm_and_plot(self, tmp_path):
        """
        Test 30-second audio recording with PCM file writing and waveform plotting.

        This test:
        1. Captures 30 seconds of audio from the microphone
        2. Writes the raw PCM data to a WAV file
        3. Plots the audio waveform
        4. Saves the plot to a file
        """
        logger.info("=" * 80)
        logger.info("Starting 30-second audio capture test")
        logger.info("=" * 80)

        # Configuration
        config = AudioConfig(sample_rate=44100, channels=1, chunk_size=1024, dtype="float32", device_name="USB Audio")

        # Storage for captured audio
        audio_chunks = []
        chunk_count = [0]

        def audio_callback(chunk, timestamp):
            audio_chunks.append(chunk.copy())
            chunk_count[0] += 1
            if chunk_count[0] % 50 == 0:  # Log every 50 chunks (~2.3s)
                duration = len(audio_chunks) * len(chunk) / config.sample_rate
                rms = np.sqrt(np.mean(chunk**2))
                logger.info(f"  Captured {chunk_count[0]} chunks ({duration:.1f}s), current RMS: {rms:.6f}")

        # Create capture instance
        capture = AudioCapture(config, audio_callback=audio_callback)

        # List devices
        logger.info("\nAvailable audio devices:")
        devices = capture.list_devices()
        for device in devices:
            marker = " <-- SELECTED" if device["index"] == capture.device_index else ""
            logger.info(f"  [{device['index']}] {device['name']} - {device['channels']}ch{marker}")

        # Start capture
        logger.info(f"\nStarting 30-second capture (device={capture.device_index})...")
        capture.start_capture()

        # Capture for 30 seconds
        capture_duration = 30.0
        start_time = time.time()

        try:
            while time.time() - start_time < capture_duration:
                time.sleep(1.0)
                elapsed = time.time() - start_time
                remaining = capture_duration - elapsed
                logger.info(f"  Recording... {elapsed:.0f}s / {capture_duration:.0f}s ({remaining:.0f}s remaining)")

        except KeyboardInterrupt:
            logger.warning("Recording interrupted by user")
        finally:
            capture.stop_capture()

        # Get statistics
        stats = capture.get_statistics()
        logger.info("\n" + "=" * 80)
        logger.info("Capture Statistics:")
        logger.info(f"  Duration: {stats['duration']:.1f}s")
        logger.info(f"  Chunks captured: {stats['total_chunks']}")
        logger.info(f"  Samples captured: {stats['total_samples']}")
        logger.info(f"  Overflows: {stats['overflow_count']}")
        logger.info(f"  Chunks in callback: {len(audio_chunks)}")
        logger.info("=" * 80)

        # Verify we got data
        assert len(audio_chunks) > 0, "No audio chunks captured"
        assert stats["total_samples"] > 0, "No audio samples captured"

        # Concatenate all chunks into single array
        audio_data = np.concatenate(audio_chunks)
        logger.info(f"\nCombined audio data: {len(audio_data)} samples, {len(audio_data)/config.sample_rate:.2f}s")

        # Calculate audio statistics
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        logger.info(f"Audio statistics: RMS={rms:.6f}, Peak={peak:.6f}")

        # Write to WAV file
        wav_path = tmp_path / "audio_capture_30s.wav"
        self._write_wav_file(wav_path, audio_data, config.sample_rate)
        logger.info(f"\nWAV file written: {wav_path}")
        logger.info(f"  File size: {os.path.getsize(wav_path) / 1024 / 1024:.2f} MB")

        # Plot waveform
        plot_path = tmp_path / "audio_waveform_30s.png"
        self._plot_waveform(audio_data, config.sample_rate, plot_path)
        logger.info(f"Waveform plot saved: {plot_path}")

        # Also save in project root for easy access
        project_root = Path(__file__).parent.parent
        wav_output = project_root / "audio_capture_test.wav"
        plot_output = project_root / "audio_waveform_test.png"

        self._write_wav_file(wav_output, audio_data, config.sample_rate)
        self._plot_waveform(audio_data, config.sample_rate, plot_output)

        logger.info("\n✓ Test outputs also saved to project root:")
        logger.info(f"  {wav_output}")
        logger.info(f"  {plot_output}")

        # Verify files were created
        assert wav_path.exists()
        assert plot_path.exists()
        assert wav_output.exists()
        assert plot_output.exists()

        logger.info("\n" + "=" * 80)
        logger.info("✓ 30-second audio capture test completed successfully")
        logger.info("=" * 80)

    def _write_wav_file(self, filepath: Path, audio_data: np.ndarray, sample_rate: int):
        """
        Write audio data to WAV file.

        Args:
            filepath: Output WAV file path
            audio_data: Audio data as float32 array (-1.0 to 1.0)
            sample_rate: Sample rate in Hz
        """
        # Convert float32 to int16 for WAV file
        # WAV format uses int16 range: -32768 to 32767
        audio_int16 = np.int16(audio_data * 32767)

        with wave.open(str(filepath), "w") as wav_file:
            # Set WAV file parameters
            n_channels = 1
            sample_width = 2  # 2 bytes for int16
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)

            # Write audio data
            wav_file.writeframes(audio_int16.tobytes())

        logger.info(f"Wrote {len(audio_data)} samples to {filepath}")

    def _plot_waveform(self, audio_data: np.ndarray, sample_rate: int, output_path: Path):
        """
        Plot audio waveform and save to file.

        Args:
            audio_data: Audio data as float32 array
            sample_rate: Sample rate in Hz
            output_path: Output plot file path
        """
        # Create time axis
        duration = len(audio_data) / sample_rate
        time_axis = np.linspace(0, duration, len(audio_data))

        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: Full waveform
        axes[0].plot(time_axis, audio_data, linewidth=0.5, color="blue", alpha=0.7)
        axes[0].set_xlabel("Time (seconds)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(f"Audio Waveform - Full Recording ({duration:.1f}s)")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, duration)

        # Plot 2: First 5 seconds (zoomed in)
        zoom_duration = min(5.0, duration)
        zoom_samples = int(zoom_duration * sample_rate)
        axes[1].plot(time_axis[:zoom_samples], audio_data[:zoom_samples], linewidth=0.5, color="green")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_title(f"Audio Waveform - First {zoom_duration:.1f}s (Detail)")
        axes[1].grid(True, alpha=0.3)

        # Plot 3: RMS envelope (smoothed)
        window_size = int(sample_rate * 0.1)  # 100ms windows
        rms_envelope = self._calculate_rms_envelope(audio_data, window_size)
        rms_time = np.linspace(0, duration, len(rms_envelope))
        axes[2].plot(rms_time, rms_envelope, linewidth=1.5, color="red")
        axes[2].set_xlabel("Time (seconds)")
        axes[2].set_ylabel("RMS Amplitude")
        axes[2].set_title("RMS Envelope (100ms windows)")
        axes[2].grid(True, alpha=0.3)

        # Add statistics text
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        stats_text = f"RMS: {rms:.6f}  |  Peak: {peak:.6f}  |  Sample Rate: {sample_rate}Hz"
        fig.text(0.5, 0.02, stats_text, ha="center", fontsize=10, bbox={"boxstyle": "round", "facecolor": "wheat"})

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Waveform plot saved with {len(audio_data)} samples")

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


# For manual testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Run the 30-second recording test
    test_instance = TestAudioRecordingAndVisualization()
    tmp_dir = Path("/tmp/audio_test")  # nosec B108 - test output
    tmp_dir.mkdir(exist_ok=True)

    logger.info("Running 30-second audio capture test...")
    test_instance.test_30s_recording_with_pcm_and_plot(tmp_dir)

    logger.info(f"\nTest files saved to: {tmp_dir}")
    logger.info("Test completed!")
