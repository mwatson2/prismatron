#!/usr/bin/env python3
"""
Simple microphone level test - verifies we're getting audio data
"""

import sys
import time

import numpy as np
import sounddevice as sd


def main():
    print("=" * 60)
    print("🎤 MICROPHONE LEVEL TEST 🎤")
    print("=" * 60)

    # Find USB Audio Device
    devices = sd.query_devices()
    usb_device_id = None

    for i, device in enumerate(devices):
        if "USB Audio" in device["name"] and device["max_input_channels"] > 0:
            usb_device_id = i
            print(f"\n✅ Found USB Audio Device at index {i}")
            print(f"   Name: {device['name']}")
            print(f"   Sample Rate: {device['default_samplerate']}Hz")
            break

    if usb_device_id is None:
        print("❌ USB Audio Device not found!")
        return 1

    # Set default device
    sd.default.device = usb_device_id

    print("\n🎤 Starting audio capture...")
    print("Press Ctrl+C to stop\n")

    sample_rate = 44100
    chunk_duration = 0.1  # 100ms
    chunk_size = int(sample_rate * chunk_duration)

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            print("Audio stream started. Monitoring levels...\n")

            sample_count = 0
            while True:
                # Read audio chunk
                audio_chunk, overflowed = stream.read(chunk_size)
                audio_chunk = audio_chunk.flatten()

                # Calculate levels
                rms = np.sqrt(np.mean(audio_chunk**2))
                peak = np.max(np.abs(audio_chunk))
                mean = np.mean(np.abs(audio_chunk))

                # Create level meter
                bar_length = 50
                rms_bar = "█" * int(rms * bar_length * 100)

                # Display
                sample_count += 1
                print(f"[{sample_count:4d}] RMS: {rms:.6f} Peak: {peak:.6f} [{rms_bar:<{bar_length}}]")

                # Show raw values occasionally
                if sample_count % 10 == 0:
                    print(f"       Raw samples (first 5): {audio_chunk[:5]}")

                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    print("\n✅ Test completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
