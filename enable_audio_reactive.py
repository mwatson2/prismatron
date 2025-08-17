#!/usr/bin/env python3
"""
Script to enable audio reactive settings and verify brightness boost functionality
"""

import logging
import time

from src.core.control_state import ControlState


def enable_audio_reactive():
    """Enable audio reactive settings in the control state"""
    print("🔧 Enabling Audio Reactive Settings")
    print("=" * 50)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    try:
        # Connect to control state (pass empty string to use default name)
        control_state = ControlState()
        if not control_state.connect():
            print("❌ Failed to connect to control state")
            return

        print("✅ Connected to control state")

        # Get current status
        status = control_state.get_status()
        if not status:
            print("❌ Failed to get control state status")
            return

        print("\n📊 Current Settings (Before):")
        print(f"  audio_reactive_enabled: {getattr(status, 'audio_reactive_enabled', 'NOT_SET')}")
        print(f"  audio_enabled: {getattr(status, 'audio_enabled', 'NOT_SET')}")
        print(f"  beat_brightness_enabled: {getattr(status, 'beat_brightness_enabled', 'NOT_SET')}")
        print(f"  beat_count: {getattr(status, 'beat_count', 'NOT_SET')}")

        # Enable audio reactive settings
        print("\n🔄 Enabling audio reactive settings...")

        # Enable all audio reactive settings using update_status
        control_state.update_status(
            audio_reactive_enabled=True,
            audio_enabled=True,
            beat_brightness_enabled=True,
            beat_brightness_intensity=0.5,  # 50% boost for more visible effect
            beat_brightness_duration=0.3,  # 30% of beat duration
        )
        print("✅ Updated audio reactive settings:")

        # Wait a moment for settings to propagate
        time.sleep(1)

        # Verify settings
        status = control_state.get_status()
        print("\n📊 Updated Settings (After):")
        print(f"  audio_reactive_enabled: {getattr(status, 'audio_reactive_enabled', 'NOT_SET')}")
        print(f"  audio_enabled: {getattr(status, 'audio_enabled', 'NOT_SET')}")
        print(f"  beat_brightness_enabled: {getattr(status, 'beat_brightness_enabled', 'NOT_SET')}")
        print(f"  beat_brightness_intensity: {getattr(status, 'beat_brightness_intensity', 'NOT_SET')}")
        print(f"  current_bpm: {getattr(status, 'current_bpm', 'NOT_SET')}")
        print(f"  beat_count: {getattr(status, 'beat_count', 'NOT_SET')}")

        print("\n✅ Audio reactive settings enabled!")
        print("💡 The brightness boost should now activate on detected beats.")
        print("🎵 Watch the logs for 'Beat brightness boost' messages from frame_renderer.py")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    enable_audio_reactive()
