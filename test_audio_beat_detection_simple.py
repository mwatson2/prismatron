#!/usr/bin/env python3
"""
Simplified Audio Beat Detection Test - Uses blocking reads instead of callbacks
"""

import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd


# BeatNet compatibility fixes
def setup_beatnet_compatibility():
    """Apply Python 3.10+ compatibility fixes for BeatNet/madmom"""
    import collections
    import collections.abc

    if not hasattr(collections, "MutableSequence"):
        collections.MutableSequence = collections.abc.MutableSequence
    if not hasattr(collections, "Mapping"):
        collections.Mapping = collections.abc.Mapping
    if not hasattr(collections, "Sequence"):
        collections.Sequence = collections.abc.Sequence

    import numpy as np

    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "complex"):
        np.complex = complex
    if not hasattr(np, "bool"):
        np.bool = bool


def main():
    print("=" * 60)
    print("🎵 SIMPLIFIED AUDIO BEAT DETECTION TEST 🎵")
    print("=" * 60)

    # Setup BeatNet compatibility
    print("\nSetting up BeatNet compatibility...")
    setup_beatnet_compatibility()

    # Import BeatNet
    try:
        from BeatNet.BeatNet import BeatNet

        print("✅ BeatNet imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import BeatNet: {e}")
        return 1

    # Find USB Audio Device
    print("\n📋 Finding USB Audio Device...")
    devices = sd.query_devices()
    usb_device_id = None

    for i, device in enumerate(devices):
        if "USB Audio" in device["name"] and device["max_input_channels"] > 0:
            usb_device_id = i
            print(f"✅ Found USB Audio Device at index {i}")
            print(f"   Name: {device['name']}")
            print(f"   Sample Rate: {device['default_samplerate']}Hz")
            break

    if usb_device_id is None:
        print("❌ USB Audio Device not found!")
        return 1

    # Set default device
    sd.default.device = usb_device_id

    # Initialize BeatNet for offline processing (we'll feed it chunks)
    print("\n🎵 Initializing BeatNet...")
    try:
        beatnet = BeatNet(
            model=1,  # GTZAN model
            mode="online",  # Online mode for chunk processing
            inference_model="PF",  # Particle filtering
            plot=[],  # No visualization
            thread=False,  # No threading
            device="cpu",  # CPU processing
        )
        print("✅ BeatNet initialized")
    except Exception as e:
        print(f"❌ Failed to initialize BeatNet: {e}")
        return 1

    # Setup audio parameters
    capture_rate = 44100  # USB device native rate
    beatnet_rate = 22050  # BeatNet expects 22050Hz
    chunk_duration = 0.1  # 100ms chunks
    chunk_size = int(capture_rate * chunk_duration)

    print(f"\n🎤 Starting audio capture...")
    print(f"   Capture Rate: {capture_rate}Hz (USB device)")
    print(f"   BeatNet Rate: {beatnet_rate}Hz (will downsample)")
    print(f"   Chunk Size: {chunk_size} samples ({chunk_duration*1000:.0f}ms)")
    print("\nNote: In a silent environment, we expect low audio levels")
    print("but BeatNet may still detect patterns in the noise.\n")
    print("Press Ctrl+C to stop...\n")

    # Statistics
    beat_count = 0
    downbeat_count = 0
    beat_times = deque(maxlen=10)
    audio_buffer = []
    audio_buffer_22k = []  # Downsampled buffer for BeatNet
    last_display_time = time.time()

    try:
        # Create input stream at device's native rate
        with sd.InputStream(samplerate=capture_rate, channels=1, dtype="float32") as stream:
            print("Audio stream started. Listening for beats...\n")

            while True:
                # Read audio chunk
                audio_chunk, overflowed = stream.read(chunk_size)
                if overflowed:
                    print("⚠️ Audio overflow detected")

                # Flatten to 1D array
                audio_chunk = audio_chunk.flatten()

                # Add to 44.1kHz buffer
                audio_buffer.extend(audio_chunk)

                # Downsample to 22.05kHz for BeatNet (simple decimation)
                audio_chunk_22k = audio_chunk[::2]  # Take every other sample
                audio_buffer_22k.extend(audio_chunk_22k)

                # Keep buffers to 2 seconds max
                max_buffer_size_44k = capture_rate * 2
                max_buffer_size_22k = beatnet_rate * 2
                if len(audio_buffer) > max_buffer_size_44k:
                    audio_buffer = audio_buffer[-max_buffer_size_44k:]
                if len(audio_buffer_22k) > max_buffer_size_22k:
                    audio_buffer_22k = audio_buffer_22k[-max_buffer_size_22k:]

                # Calculate audio levels (from original rate)
                rms = np.sqrt(np.mean(audio_chunk**2))
                peak = np.max(np.abs(audio_chunk))

                # Process with BeatNet (feed the downsampled buffer)
                if len(audio_buffer_22k) >= beatnet_rate:  # At least 1 second of 22kHz audio
                    try:
                        audio_array = np.array(audio_buffer_22k, dtype=np.float32)
                        beats = beatnet.process(audio_array)

                        if beats is not None and len(beats) > 0:
                            for beat_time, downbeat_prob in beats:
                                # Check if this is a new beat (not a duplicate)
                                if not beat_times or beat_time > beat_times[-1] + 0.1:
                                    beat_count += 1
                                    beat_times.append(beat_time)

                                    is_downbeat = downbeat_prob > 0.5
                                    if is_downbeat:
                                        downbeat_count += 1

                                    # Calculate BPM
                                    bpm = 120
                                    if len(beat_times) >= 2:
                                        intervals = [
                                            beat_times[i] - beat_times[i - 1] for i in range(1, len(beat_times))
                                        ]
                                        if intervals:
                                            avg_interval = np.mean(intervals)
                                            if avg_interval > 0:
                                                bpm = 60.0 / avg_interval

                                    beat_type = "DOWNBEAT" if is_downbeat else "BEAT"
                                    print(
                                        f"🎵 {beat_type}: #{beat_count} at {beat_time:.2f}s, "
                                        f"BPM={bpm:.1f}, conf={downbeat_prob:.2f}"
                                    )

                    except Exception as e:
                        pass  # Silently ignore BeatNet processing errors

                # Display status every 2 seconds
                current_time = time.time()
                if current_time - last_display_time > 2.0:
                    print(
                        f"\n📊 Status: RMS={rms:.6f}, Peak={peak:.6f}, "
                        f"Beats={beat_count}, Downbeats={downbeat_count}, "
                        f"Buffer={len(audio_buffer)} samples\n"
                    )
                    last_display_time = current_time

                    if beat_count == 0:
                        print("   (No beats detected yet - this is normal in silence)")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"   Total Beats Detected: {beat_count}")
    print(f"   Total Downbeats: {downbeat_count}")
    if beat_count > 0:
        print(f"   Downbeat Ratio: {downbeat_count/beat_count*100:.1f}%")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
