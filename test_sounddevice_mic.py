#!/usr/bin/env python3
"""
Test USB microphone using sounddevice library
"""

import sys
import time

import numpy as np
import sounddevice as sd


def list_audio_devices():
    """List all available audio devices"""
    print("=" * 60)
    print("Available Audio Devices")
    print("=" * 60)
    devices = sd.query_devices()

    usb_device_id = None
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            marker = " 👉" if "USB Audio" in device["name"] else ""
            print(
                f"{i}: {device['name']} (in:{device['max_input_channels']}, "
                f"sr:{device['default_samplerate']}Hz){marker}"
            )
            if "USB Audio" in device["name"]:
                usb_device_id = i

    return usb_device_id


def test_recording(device_id=None, duration=3, samplerate=44100):
    """Test recording from a specific device"""
    print(f"\n{'='*60}")
    print(f"Testing Recording (Device: {device_id}, Duration: {duration}s)")
    print(f"{'='*60}")

    if device_id is not None:
        sd.default.device = device_id

    print(f"Recording for {duration} seconds...")
    print("Note: In a silent environment, we expect very low but non-zero values\n")

    try:
        # Record audio
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")

        # Show progress
        for i in range(duration):
            time.sleep(1)
            print(f"  Recording... {i+1}/{duration}s")

        sd.wait()  # Wait for recording to finish

        # Analyze the recording
        print("\n📊 Recording Analysis:")
        print(f"  Shape: {recording.shape}")
        print(f"  Sample rate: {samplerate} Hz")
        print(f"  Duration: {len(recording)/samplerate:.2f} seconds")

        # Calculate statistics
        rms = np.sqrt(np.mean(recording**2))
        peak = np.max(np.abs(recording))
        mean = np.mean(np.abs(recording))

        print("\n  Audio Levels:")
        print(f"    RMS:  {rms:.9f}")
        print(f"    Peak: {peak:.9f}")
        print(f"    Mean: {mean:.9f}")

        # Check if we got any data
        if rms > 0 or peak > 0:
            print("\n✅ SUCCESS! Microphone is capturing data!")
            print("   Even in silence, we see noise floor values")

            # Show a sample of the raw data
            print("\n  Sample of raw audio data (first 10 values):")
            for i in range(min(10, len(recording))):
                print(f"    [{i}]: {recording[i][0]:.9f}")

            return True
        else:
            print("\n❌ All zeros - microphone may not be working properly")
            return False

    except Exception as e:
        print(f"\n❌ Recording failed: {e}")
        return False


def test_continuous_monitoring(device_id=None, duration=5):
    """Monitor audio input continuously"""
    print(f"\n{'='*60}")
    print(f"Continuous Audio Monitoring (Device: {device_id})")
    print(f"{'='*60}")

    if device_id is not None:
        sd.default.device = device_id

    print("Monitoring audio levels...")
    print("Press Ctrl+C to stop\n")

    def audio_callback(indata, frames, time, status):
        """Callback for continuous monitoring"""
        if status:
            print(f"Status: {status}")

        # Calculate current levels
        rms = np.sqrt(np.mean(indata**2))
        peak = np.max(np.abs(indata))

        # Create a simple level meter
        bar_length = 50
        rms_bar = "█" * int(rms * bar_length * 1000)  # Scale up for visibility

        print(f"\rRMS: {rms:.9f} [{rms_bar:<{bar_length}}]", end="", flush=True)

    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100):
            for i in range(duration):
                time.sleep(1)
        print("\n")
        return True
    except Exception as e:
        print(f"\n❌ Monitoring failed: {e}")
        return False


def main():
    print("🎤 USB Microphone Test with Sounddevice 🎤\n")

    # List devices
    usb_device_id = list_audio_devices()

    if usb_device_id is None:
        print("\n⚠️ No USB Audio Device found!")
        print("Testing with default device...")
        device_to_test = None
    else:
        print(f"\n✅ Found USB Audio Device at index {usb_device_id}")
        device_to_test = usb_device_id

    # Test recording
    success = test_recording(device_to_test, duration=3)

    if success:
        # Test continuous monitoring
        print("\nNow testing continuous monitoring...")
        test_continuous_monitoring(device_to_test, duration=5)

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print(f"USB microphone at device {device_to_test} is working properly")
        print("\nTo use this device in your code:")
        print(f"  sounddevice: sd.default.device = {device_to_test}")
        print(f"  pyaudio: input_device_index={device_to_test}")
        return 0
    else:
        print("\n❌ Microphone test failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
