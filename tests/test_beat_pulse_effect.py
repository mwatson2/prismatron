"""
Test script for beat pulse effect integration.

This script tests the audio-reactive brightness pulse effect by:
1. Creating a frame renderer with audio beat analyzer
2. Enabling audio reactive effects in control state
3. Simulating beat events and testing brightness calculations
4. Verifying that LED values are modified correctly during beats
"""

import sys
import time
from pathlib import Path

# Add src to path for tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from src.consumer.audio_beat_analyzer import AudioBeatAnalyzer, AudioState
from src.consumer.frame_renderer import FrameRenderer
from src.core.control_state import ControlState, SystemStatus


def test_beat_pulse_brightness_calculation():
    """Test the brightness boost calculation directly."""
    print("=== Testing Beat Pulse Brightness Calculation ===")

    # Create mock components
    control_state = ControlState("test_control")
    control_state.initialize()

    # Enable audio reactive effects
    control_state.update_status(audio_reactive_enabled=True, audio_enabled=True)

    # Create audio beat analyzer
    audio_analyzer = AudioBeatAnalyzer()

    # Set up beat state
    audio_analyzer.audio_state.is_active = True
    audio_analyzer.audio_state.current_bpm = 120.0  # 120 BPM = 0.5s per beat
    audio_analyzer.audio_state.last_beat_time = 0.0
    audio_analyzer.start_time = time.time()

    # Create frame renderer with LED ordering (dummy ordering for test)
    led_count = 100
    led_ordering = np.arange(led_count)  # Simple 1:1 mapping

    renderer = FrameRenderer(led_ordering=led_ordering, control_state=control_state, audio_beat_analyzer=audio_analyzer)

    print(f"Created frame renderer with {led_count} LEDs")
    print(f"Audio BPM: {audio_analyzer.audio_state.current_bpm}")
    print(f"Beat interval: {60.0 / audio_analyzer.audio_state.current_bpm:.3f}s")
    print(f"Boost window: {0.25 * (60.0 / audio_analyzer.audio_state.current_bpm):.3f}s")

    # Test brightness calculation at different times
    current_time = time.time()
    test_times = [
        0.0,  # At beat start
        0.125,  # Quarter way through boost window
        0.25,  # Half way through boost window
        0.375,  # Three quarters through boost window
        0.5,  # End of boost window (should start dropping off)
        0.75,  # Outside boost window
        1.0,  # Well outside boost window
    ]

    print("\nBrightness boost at different times after beat:")
    for t in test_times:
        test_time = current_time + t
        boost = renderer._calculate_beat_brightness_boost(test_time)
        boost_percent = (boost - 1.0) * 100
        print(f"  t={t:.3f}s: boost={boost:.4f} ({boost_percent:+.1f}%)")

    print()


def test_led_value_processing():
    """Test that LED values are correctly modified with brightness boost."""
    print("=== Testing LED Value Processing ===")

    # Create mock components
    control_state = ControlState("test_control2")
    control_state.initialize()

    # Enable audio reactive effects
    control_state.update_status(audio_reactive_enabled=True, audio_enabled=True)

    # Create audio beat analyzer
    audio_analyzer = AudioBeatAnalyzer()
    audio_analyzer.audio_state.is_active = True
    audio_analyzer.audio_state.current_bpm = 120.0
    audio_analyzer.audio_state.last_beat_time = 0.0
    audio_analyzer.start_time = time.time()

    # Create frame renderer
    led_count = 10
    led_ordering = np.arange(led_count)

    renderer = FrameRenderer(led_ordering=led_ordering, control_state=control_state, audio_beat_analyzer=audio_analyzer)

    # Create test LED values (moderate brightness to see boost effect)
    test_led_values = np.array(
        [
            [100, 0, 0],  # Red
            [0, 100, 0],  # Green
            [0, 0, 100],  # Blue
            [100, 100, 0],  # Yellow
            [100, 0, 100],  # Magenta
            [0, 100, 100],  # Cyan
            [50, 50, 50],  # Gray
            [200, 200, 200],  # Bright white
            [10, 10, 10],  # Dark
            [255, 255, 255],  # Max brightness
        ]
    )

    print(f"Testing with {led_count} LEDs")
    print("Original LED values (first 5 LEDs):")
    for i in range(5):
        r, g, b = test_led_values[i]
        print(f"  LED {i}: ({r:3d}, {g:3d}, {b:3d})")

    # Temporarily modify the _send_to_outputs method to capture the processed values
    original_send_to_outputs = renderer._send_to_outputs
    processed_values = None

    def capture_send_to_outputs(led_values, metadata=None):
        nonlocal processed_values
        # Apply the same processing as the original method but capture values
        physical_led_values = renderer._convert_spatial_to_physical(led_values)

        # Apply audio-reactive brightness boost if enabled
        brightness_multiplier = renderer._calculate_beat_brightness_boost(time.time())
        if brightness_multiplier != 1.0:
            physical_led_values = (physical_led_values * brightness_multiplier).astype(np.uint8)
            physical_led_values = np.clip(physical_led_values, 0, 255)

        processed_values = physical_led_values
        print(f"Brightness multiplier: {brightness_multiplier:.4f}")

    renderer._send_to_outputs = capture_send_to_outputs

    # Test processing
    renderer._send_to_outputs(test_led_values)

    if processed_values is not None:
        print("Processed LED values (first 5 LEDs):")
        for i in range(5):
            r, g, b = processed_values[i]
            orig_r, orig_g, orig_b = test_led_values[i]
            print(f"  LED {i}: ({r:3d}, {g:3d}, {b:3d}) [was ({orig_r:3d}, {orig_g:3d}, {orig_b:3d})]")

    # Restore original method
    renderer._send_to_outputs = original_send_to_outputs
    print()


def test_configuration_states():
    """Test different configuration states for audio reactive effects."""
    print("=== Testing Configuration States ===")

    control_state = ControlState("test_control3")
    control_state.initialize()

    audio_analyzer = AudioBeatAnalyzer()
    audio_analyzer.audio_state.is_active = True
    audio_analyzer.audio_state.current_bpm = 120.0

    led_ordering = np.arange(10)
    renderer = FrameRenderer(led_ordering=led_ordering, control_state=control_state, audio_beat_analyzer=audio_analyzer)

    current_time = time.time()

    # Test 1: Audio reactive disabled
    control_state.update_status(audio_reactive_enabled=False, audio_enabled=True)
    boost1 = renderer._calculate_beat_brightness_boost(current_time)
    print(f"Audio reactive disabled: boost={boost1:.4f} (should be 1.0)")

    # Test 2: Audio disabled
    control_state.update_status(audio_reactive_enabled=True, audio_enabled=False)
    boost2 = renderer._calculate_beat_brightness_boost(current_time)
    print(f"Audio disabled: boost={boost2:.4f} (should be 1.0)")

    # Test 3: Both enabled
    control_state.update_status(audio_reactive_enabled=True, audio_enabled=True)
    boost3 = renderer._calculate_beat_brightness_boost(current_time)
    print(f"Both enabled: boost={boost3:.4f} (should vary with beat timing)")

    # Test 4: No audio analyzer
    renderer_no_audio = FrameRenderer(led_ordering=led_ordering, control_state=control_state, audio_beat_analyzer=None)
    boost4 = renderer_no_audio._calculate_beat_brightness_boost(current_time)
    print(f"No audio analyzer: boost={boost4:.4f} (should be 1.0)")

    print()


# Tests can be run with pytest
