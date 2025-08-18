#!/usr/bin/env python3
"""
Test the complete beat detection -> brightness pulsing integration
"""

import logging
import os
import sys
import time

import numpy as np

# Set environment to reduce resource usage
os.environ["MADMOM_SKIP_GPU"] = "True"

# Add path
sys.path.insert(0, "/mnt/dev/prismatron")

# Import directly from the files to avoid package issues
import importlib.util

# Import audio beat analyzer
spec = importlib.util.spec_from_file_location(
    "audio_beat_analyzer", "/mnt/dev/prismatron/src/consumer/audio_beat_analyzer.py"
)
audio_beat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_beat_module)

# Import frame renderer
spec = importlib.util.spec_from_file_location("frame_renderer", "/mnt/dev/prismatron/src/consumer/frame_renderer.py")
frame_renderer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frame_renderer_module)

# Import control state
spec = importlib.util.spec_from_file_location("control_state", "/mnt/dev/prismatron/src/control/control_state.py")
control_state_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control_state_module)

AudioBeatAnalyzer = audio_beat_module.AudioBeatAnalyzer
BeatEvent = audio_beat_module.BeatEvent
FrameRenderer = frame_renderer_module.FrameRenderer
ControlState = control_state_module.ControlState

# Reduce logging to see performance
logging.basicConfig(level=logging.INFO)

beat_count = 0
brightness_values = []


def beat_callback(beat_event: BeatEvent):
    """Handle beat events"""
    global beat_count
    beat_count += 1
    print(f"🎵 Beat #{beat_count}: BPM={beat_event.bpm:.1f}, intensity={beat_event.intensity:.2f}")


class MockControlState:
    """Mock control state for testing brightness boost"""

    def __init__(self):
        self.audio_reactive_enabled = True
        self.audio_enabled = True
        self.beat_brightness_enabled = True
        self.beat_brightness_intensity = 0.5  # 50% boost
        self.beat_brightness_duration = 0.3  # 30% of beat duration

    def get_status(self):
        return self


def test_brightness_calculation():
    """Test the brightness boost calculation directly"""
    print("Testing brightness boost calculation...")

    # Create mock control state
    control_state = MockControlState()

    # Create audio beat analyzer
    analyzer = AudioBeatAnalyzer(beat_callback=beat_callback, device="cpu")

    # Create frame renderer with audio analyzer
    renderer = FrameRenderer(
        control_state=control_state, audio_beat_analyzer=analyzer, first_frame_delay_ms=0.0, timing_tolerance_ms=5.0
    )

    # Start audio analysis
    analyzer.start_analysis()
    print("Audio analysis started...")

    try:
        # Wait for some beats and test brightness calculation
        start_time = time.time()
        test_duration = 8.0  # Test for 8 seconds

        while time.time() - start_time < test_duration:
            current_time = time.time()

            # Calculate brightness boost
            brightness_multiplier = renderer._calculate_beat_brightness_boost(current_time)
            brightness_values.append(brightness_multiplier)

            # Log occasionally
            if len(brightness_values) % 20 == 0:  # Every 20th calculation
                print(f"Time: {current_time - start_time:.1f}s, Brightness: {brightness_multiplier:.3f}")

            time.sleep(0.05)  # 20Hz sampling

        print("\nTest completed:")
        print(f"- Total beats detected: {beat_count}")
        print(f"- Brightness samples: {len(brightness_values)}")
        print(f"- Min brightness: {min(brightness_values):.3f}")
        print(f"- Max brightness: {max(brightness_values):.3f}")
        print(f"- Average brightness: {np.mean(brightness_values):.3f}")

        # Count samples with boost (>1.0)
        boosted_samples = [b for b in brightness_values if b > 1.001]
        print(f"- Samples with boost: {len(boosted_samples)} ({len(boosted_samples)/len(brightness_values)*100:.1f}%)")

        if len(boosted_samples) > 0:
            print(f"- Max boost: {max(boosted_samples):.3f}")
            print("✅ Brightness boost system is working!")
        else:
            print("⚠️  No brightness boosts detected - may need actual audio input")

    except Exception as e:
        print(f"Error during test: {e}")

    finally:
        analyzer.stop_analysis()
        print("Audio analysis stopped")


def main():
    print("Testing Beat Detection -> Brightness Pulsing Integration")
    print("=" * 60)

    test_brightness_calculation()


if __name__ == "__main__":
    main()
