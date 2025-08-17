#!/usr/bin/env python3
"""
Debug script to check why brightness boost isn't working
"""

import logging
import time

from src.core.control_state import ControlState


def debug_brightness_boost():
    """Debug the brightness boost control state settings"""
    print("🔧 Debugging Brightness Boost Configuration")
    print("=" * 60)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    try:
        # Connect to control state
        control_state = ControlState("prismatron_control")
        if not control_state.connect():
            print("❌ Failed to connect to control state")
            return

        print("✅ Connected to control state")

        # Get current status
        status = control_state.get_status()
        if not status:
            print("❌ Failed to get control state status")
            return

        print("\n📊 Current Control State Settings:")
        print(f"  audio_reactive_enabled: {getattr(status, 'audio_reactive_enabled', 'NOT_SET')}")
        print(f"  audio_enabled: {getattr(status, 'audio_enabled', 'NOT_SET')}")
        print(f"  beat_brightness_enabled: {getattr(status, 'beat_brightness_enabled', 'NOT_SET')}")
        print(f"  beat_brightness_intensity: {getattr(status, 'beat_brightness_intensity', 'NOT_SET')}")
        print(f"  beat_brightness_duration: {getattr(status, 'beat_brightness_duration', 'NOT_SET')}")
        print(f"  current_bpm: {getattr(status, 'current_bpm', 'NOT_SET')}")
        print(f"  last_beat_time: {getattr(status, 'last_beat_time', 'NOT_SET')}")
        print(f"  beat_count: {getattr(status, 'beat_count', 'NOT_SET')}")

        # Check if brightness boost should work
        print("\n🔍 Brightness Boost Analysis:")

        audio_reactive_enabled = getattr(status, "audio_reactive_enabled", False)
        audio_enabled = getattr(status, "audio_enabled", False)
        beat_brightness_enabled = getattr(status, "beat_brightness_enabled", False)

        if not audio_reactive_enabled:
            print("❌ Audio reactive is NOT enabled")
            print("   Solution: Enable audio reactive in the web interface")
        else:
            print("✅ Audio reactive is enabled")

        if not audio_enabled:
            print("❌ Audio processing is NOT enabled")
            print("   Solution: Audio analyzer needs to be running")
        else:
            print("✅ Audio processing is enabled")

        if not beat_brightness_enabled:
            print("❌ Beat brightness boost is NOT enabled")
            print("   Solution: Enable beat brightness boost in the web interface")
        else:
            print("✅ Beat brightness boost is enabled")

        current_bpm = getattr(status, "current_bpm", 0)
        beat_count = getattr(status, "beat_count", 0)

        if current_bpm <= 0:
            print("❌ No BPM detected (BPM = 0)")
            print("   Solution: Ensure music is playing and audio analyzer is receiving beats")
        else:
            print(f"✅ BPM detected: {current_bpm:.1f}")

        if beat_count <= 0:
            print("❌ No beats detected (beat count = 0)")
            print("   Solution: Check audio input, music volume, and beat detection settings")
        else:
            print(f"✅ Beats detected: {beat_count} total beats")

        # Provide recommendations
        print("\n💡 Recommendations:")
        if not audio_reactive_enabled:
            print("1. Enable audio reactive mode in the web interface")
        if not beat_brightness_enabled:
            print("2. Enable beat brightness boost in the audio settings")
        if current_bpm <= 0 or beat_count <= 0:
            print("3. Ensure music is playing with clear beats")
            print("4. Check microphone input levels and gain")
            print("5. Verify audio analyzer is running and detecting beats")

        if audio_reactive_enabled and beat_brightness_enabled and current_bpm > 0 and beat_count > 0:
            print("✅ All settings look correct - brightness boost should be working!")
            print("   If you still don't see brightness pulsing, check:")
            print("   - Beat brightness intensity setting (may be too low)")
            print("   - Beat brightness duration setting")
            print("   - LED content brightness (boost may be too small to notice on dim content)")

    except Exception as e:
        print(f"❌ Error during debugging: {e}")


if __name__ == "__main__":
    debug_brightness_boost()
