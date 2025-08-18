#!/usr/bin/env python3
"""
PyAudio Device Test - Find and test the working microphone
"""

import sys
import time

import numpy as np
import pyaudio


def test_audio_device(device_index, sample_rate=None, duration=2):
    """Test a specific audio device"""
    p = pyaudio.PyAudio()

    try:
        # Get device info
        info = p.get_device_info_by_index(device_index)
        print(f"\nTesting device {device_index}: {info['name']}")
        print(f"  Max input channels: {info['maxInputChannels']}")
        print(f"  Default sample rate: {info['defaultSampleRate']}")

        if info["maxInputChannels"] == 0:
            print("  ❌ No input channels - skipping")
            return False

        # Use device's default sample rate if not specified
        if sample_rate is None:
            sample_rate = int(info["defaultSampleRate"])

        # Try to open stream
        chunk_size = 1024
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
        )

        print("  ✅ Stream opened successfully")
        print(f"  Recording {duration} seconds...")

        # Record audio
        all_audio = []
        num_chunks = int(sample_rate * duration / chunk_size)

        for i in range(num_chunks):
            try:
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                all_audio.append(audio_chunk)

                # Calculate current levels
                rms = np.sqrt(np.mean(audio_chunk**2))
                peak = np.max(np.abs(audio_chunk))

                # Show progress
                if i % 10 == 0:
                    print(f"    Chunk {i}/{num_chunks}: RMS={rms:.6f}, Peak={peak:.6f}")

            except Exception as e:
                print(f"    ⚠️ Read error: {e}")

        # Close stream
        stream.stop_stream()
        stream.close()

        # Analyze full recording
        if all_audio:
            full_audio = np.concatenate(all_audio)
            overall_rms = np.sqrt(np.mean(full_audio**2))
            overall_peak = np.max(np.abs(full_audio))
            overall_mean = np.mean(np.abs(full_audio))

            print("\n  📊 Overall Statistics:")
            print(f"    RMS Level: {overall_rms:.6f}")
            print(f"    Peak Level: {overall_peak:.6f}")
            print(f"    Mean Level: {overall_mean:.6f}")
            print(f"    Total samples: {len(full_audio)}")

            # Check if we got actual audio (very low threshold for silent environment)
            # Even a silent USB mic should show some noise floor
            if overall_rms > 0.000001 or overall_peak > 0.000001:
                print(f"  ✅ AUDIO DEVICE WORKING! Device {device_index} is capturing data!")
                print("     (Even in silence, USB mics show tiny noise values)")
                return True
            else:
                print("  ⚠️ No audio data detected (all zeros - device may not be working)")
                return False

    except Exception as e:
        print(f"  ❌ Failed to test device: {e}")
        return False

    finally:
        p.terminate()

    return False


def find_working_microphone():
    """Find and test all audio devices"""
    p = pyaudio.PyAudio()

    print("=" * 60)
    print("PyAudio Microphone Device Tester")
    print("=" * 60)

    # List all devices
    print("\n📋 Available Audio Devices:")
    device_count = p.get_device_count()
    usb_devices = []

    for i in range(device_count):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  Device {i}: {info['name']} (Input channels: {info['maxInputChannels']})")
            # Focus on USB Audio Device only
            if "USB Audio" in info["name"]:
                usb_devices.append(i)
                print("    👉 Found USB Audio Device!")

    p.terminate()

    # Test only USB devices (skip problematic NVIDIA APE devices)
    print("\n🎤 Testing USB Audio Devices:")
    print("Note: It's currently silent, so we expect to see very low but non-zero values")
    print("(ambient noise from the USB microphone circuitry)\n")

    working_devices = []
    devices_to_test = usb_devices if usb_devices else []

    for i in devices_to_test:
        p = pyaudio.PyAudio()
        info = p.get_device_info_by_index(i)
        p.terminate()

        if test_audio_device(i):
            working_devices.append((i, info["name"]))
        time.sleep(0.5)  # Small delay between tests

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    if working_devices:
        print(f"✅ Found {len(working_devices)} working microphone(s):")
        for device_id, name in working_devices:
            print(f"   Device {device_id}: {name}")
        print(f"\n💡 To use in your code, add: input_device_index={working_devices[0][0]}")
    else:
        print("❌ No working microphones detected!")
        print("   - Check if microphone is connected")
        print("   - Check system audio permissions")
        print("   - Try speaking louder during the test")

    return working_devices


if __name__ == "__main__":
    working = find_working_microphone()
    sys.exit(0 if working else 1)
