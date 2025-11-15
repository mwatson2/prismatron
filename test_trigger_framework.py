#!/usr/bin/env python3
"""
Test script for the trigger framework.

Tests:
1. Creating trigger configurations
2. Evaluating beat triggers with different conditions
3. Evaluating test triggers
4. Effect creation from triggers
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.consumer.frame_renderer import EffectTriggerConfig, EffectTriggerManager
from src.consumer.led_effect import LedEffectManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MockBeatState:
    """Mock beat state for testing."""

    def __init__(self, bpm=120, intensity=0.8, confidence=0.9):
        self.current_bpm = bpm
        self.beat_intensity = intensity
        self.confidence = confidence
        self.is_active = True
        self.last_beat_time = 0.0
        self.last_beat_wallclock_time = 0.0


def test_trigger_config_validation():
    """Test trigger configuration validation."""
    logger.info("\n=== Test 1: Trigger Config Validation ===")

    # Valid beat trigger
    trigger1 = EffectTriggerConfig(
        trigger_type="beat",
        effect_class="BeatBrightnessEffect",
        effect_params={"boost_intensity": 4.0, "duration_fraction": 0.4},
        confidence_min=0.5,
        intensity_min=0.3,
    )
    logger.info(f"✓ Created beat trigger: {trigger1.effect_class}")

    # Valid test trigger
    trigger2 = EffectTriggerConfig(
        trigger_type="test",
        effect_class="TemplateEffect",
        effect_params={
            "template_path": "templates/ring_800x480_leds.npy",
            "duration": 1.0,
            "blend_mode": "addboost",
            "intensity": 2.0,
        },
    )
    logger.info(f"✓ Created test trigger: {trigger2.effect_class}")

    # Invalid trigger type
    try:
        trigger3 = EffectTriggerConfig(trigger_type="invalid", effect_class="BeatBrightnessEffect")
        logger.error("✗ Should have raised ValueError for invalid trigger type")
    except ValueError as e:
        logger.info(f"✓ Correctly rejected invalid trigger type: {e}")


def test_beat_trigger_evaluation():
    """Test beat trigger evaluation with different conditions."""
    logger.info("\n=== Test 2: Beat Trigger Evaluation ===")

    # Create effect manager and trigger manager
    effect_manager = LedEffectManager()
    trigger_manager = EffectTriggerManager(effect_manager)

    # Configure triggers with different thresholds
    triggers = [
        # High-confidence trigger
        EffectTriggerConfig(
            trigger_type="beat",
            effect_class="BeatBrightnessEffect",
            effect_params={"boost_intensity": 5.0, "duration_fraction": 0.3},
            confidence_min=0.8,
            intensity_min=0.7,
        ),
        # Medium-confidence trigger
        EffectTriggerConfig(
            trigger_type="beat",
            effect_class="BeatBrightnessEffect",
            effect_params={"boost_intensity": 3.0, "duration_fraction": 0.4},
            confidence_min=0.5,
            intensity_min=0.3,
        ),
        # BPM-filtered trigger (120-140 BPM)
        EffectTriggerConfig(
            trigger_type="beat",
            effect_class="BeatBrightnessEffect",
            effect_params={"boost_intensity": 4.0, "duration_fraction": 0.5},
            confidence_min=0.3,
            bpm_min=120.0,
            bpm_max=140.0,
        ),
    ]
    trigger_manager.set_triggers(triggers)

    # Test Case 1: High confidence beat (should match first trigger)
    logger.info("\nTest Case 1: High confidence beat")
    beat_state = MockBeatState(bpm=130, intensity=0.9, confidence=0.85)
    beat_state.last_beat_wallclock_time = 1.0

    trigger_manager.evaluate_beat_triggers(
        frame_timeline_time=0.5, beat_state=beat_state, last_beat_wallclock_time=1.0, wallclock_delta=0.5
    )

    active_count = effect_manager.get_active_count()
    logger.info(f"Active effects after high-confidence beat: {active_count}")
    assert active_count == 1, f"Expected 1 effect, got {active_count}"

    # Clear effects for next test
    effect_manager.clear_effects()
    trigger_manager._last_beat_time_processed = -1.0

    # Test Case 2: Medium confidence beat (should match second trigger)
    logger.info("\nTest Case 2: Medium confidence beat")
    beat_state = MockBeatState(bpm=130, intensity=0.5, confidence=0.6)
    beat_state.last_beat_wallclock_time = 2.0

    trigger_manager.evaluate_beat_triggers(
        frame_timeline_time=1.5, beat_state=beat_state, last_beat_wallclock_time=2.0, wallclock_delta=0.5
    )

    active_count = effect_manager.get_active_count()
    logger.info(f"Active effects after medium-confidence beat: {active_count}")
    assert active_count == 1, f"Expected 1 effect, got {active_count}"

    # Clear effects for next test
    effect_manager.clear_effects()
    trigger_manager._last_beat_time_processed = -1.0

    # Test Case 3: Low confidence beat (should not match any trigger)
    logger.info("\nTest Case 3: Low confidence beat (below thresholds)")
    beat_state = MockBeatState(bpm=130, intensity=0.2, confidence=0.2)
    beat_state.last_beat_wallclock_time = 3.0

    trigger_manager.evaluate_beat_triggers(
        frame_timeline_time=2.5, beat_state=beat_state, last_beat_wallclock_time=3.0, wallclock_delta=0.5
    )

    active_count = effect_manager.get_active_count()
    logger.info(f"Active effects after low-confidence beat: {active_count}")
    assert active_count == 0, f"Expected 0 effects, got {active_count}"

    # Test Case 4: BPM out of range (should not match BPM-filtered trigger)
    logger.info("\nTest Case 4: BPM out of range")
    beat_state = MockBeatState(bpm=80, intensity=0.5, confidence=0.5)
    beat_state.last_beat_wallclock_time = 4.0

    # Clear processed beat
    trigger_manager._last_beat_time_processed = -1.0

    trigger_manager.evaluate_beat_triggers(
        frame_timeline_time=3.5, beat_state=beat_state, last_beat_wallclock_time=4.0, wallclock_delta=0.5
    )

    active_count = effect_manager.get_active_count()
    logger.info(f"Active effects after out-of-range BPM: {active_count}")
    # Should match second trigger (medium confidence, no BPM filter)
    assert active_count == 1, f"Expected 1 effect, got {active_count}"

    logger.info("✓ All beat trigger tests passed")


def test_test_trigger_evaluation():
    """Test periodic test trigger evaluation."""
    logger.info("\n=== Test 3: Test Trigger Evaluation ===")

    # Create effect manager and trigger manager
    effect_manager = LedEffectManager()
    trigger_manager = EffectTriggerManager(effect_manager)

    # Configure test trigger
    triggers = [
        EffectTriggerConfig(
            trigger_type="test",
            effect_class="TemplateEffect",
            effect_params={
                "template_path": "templates/ring_800x480_leds.npy",
                "duration": 1.0,
                "blend_mode": "addboost",
                "intensity": 2.0,
            },
        )
    ]
    trigger_manager.set_triggers(triggers)
    trigger_manager.set_test_interval(2.0)

    # Check if template exists, if not create a dummy one
    template_path = Path("templates/ring_800x480_leds.npy")
    if not template_path.exists():
        logger.info("Creating dummy template for testing")
        template_path.parent.mkdir(exist_ok=True)
        # Create a simple test template (10 frames, 100 LEDs)
        dummy_template = np.random.randint(0, 255, size=(10, 100), dtype=np.uint8)
        np.save(template_path, dummy_template)

    # Test Case 1: First trigger (should fire immediately)
    logger.info("\nTest Case 1: First test trigger")
    trigger_manager.evaluate_test_triggers(frame_timeline_time=0.0)

    active_count = effect_manager.get_active_count()
    logger.info(f"Active effects after first test trigger: {active_count}")
    assert active_count == 1, f"Expected 1 effect, got {active_count}"

    # Test Case 2: Too soon (should not fire)
    logger.info("\nTest Case 2: Test trigger too soon")
    trigger_manager.evaluate_test_triggers(frame_timeline_time=1.0)

    active_count = effect_manager.get_active_count()
    logger.info(f"Active effects (should be same): {active_count}")
    assert active_count == 1, f"Expected 1 effect, got {active_count}"

    # Test Case 3: After interval (should fire)
    logger.info("\nTest Case 3: Test trigger after interval")
    trigger_manager.evaluate_test_triggers(frame_timeline_time=2.5)

    active_count = effect_manager.get_active_count()
    logger.info(f"Active effects after second test trigger: {active_count}")
    assert active_count == 2, f"Expected 2 effects, got {active_count}"

    logger.info("✓ All test trigger tests passed")


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Trigger Framework Test Suite")
    logger.info("=" * 60)

    try:
        test_trigger_config_validation()
        test_beat_trigger_evaluation()
        test_test_trigger_evaluation()

        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests passed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
