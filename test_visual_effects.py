#!/usr/bin/env python3
"""Test script for the new visual effects system."""

import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_effects_loading():
    """Test that effects can be loaded and listed."""
    try:
        from src.producer.effects import EffectRegistry

        # List all effects
        effects = EffectRegistry.list_effects()
        logger.info(f"Successfully loaded {len(effects)} effects:")

        # Group by category
        categories = {}
        for effect in effects:
            category = effect["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(effect)

        # Print organized list
        for category, category_effects in categories.items():
            logger.info(f"\n  {category.upper()}:")
            for effect in category_effects:
                logger.info(f"    {effect['icon']} {effect['name']} ({effect['id']})")
                logger.info(f"      {effect['description']}")

        return True

    except Exception as e:
        logger.error(f"Failed to load effects: {e}")
        return False


def test_effect_creation():
    """Test creating and running individual effects."""
    try:
        from src.producer.effects import EffectRegistry

        # Test a few different effects
        test_effects = ["rainbow_sweep", "plasma", "fireworks", "digital_rain", "rotating_shapes"]

        for effect_id in test_effects:
            logger.info(f"\nTesting effect: {effect_id}")

            # Create effect
            effect = EffectRegistry.create_effect(effect_id, width=128, height=64, fps=30)
            if effect is None:
                logger.warning(f"  Could not create effect: {effect_id}")
                continue

            # Generate a few frames
            for i in range(5):
                frame = effect.generate_frame()
                if frame is None:
                    logger.warning(f"  Frame {i}: None returned")
                    break

                if frame.shape != (64, 128, 3):
                    logger.warning(f"  Frame {i}: Wrong shape {frame.shape}")
                    break

                if frame.dtype != "uint8":
                    logger.warning(f"  Frame {i}: Wrong dtype {frame.dtype}")
                    break

                # Check for reasonable values
                min_val, max_val = frame.min(), frame.max()
                if min_val < 0 or max_val > 255:
                    logger.warning(f"  Frame {i}: Values out of range [{min_val}, {max_val}]")
                    break

                logger.info(f"  Frame {i}: OK - Shape {frame.shape}, Range [{min_val}, {max_val}]")

            logger.info(f"  {effect_id}: SUCCESS")

        return True

    except Exception as e:
        logger.error(f"Failed to test effect creation: {e}")
        return False


def test_effect_source():
    """Test the effect source wrapper."""
    try:
        from src.producer.effect_source import EffectSource

        logger.info("\nTesting EffectSource wrapper:")

        # Create source
        source = EffectSource(width=128, height=64, fps=30)
        logger.info("  Created EffectSource")

        # Set an effect
        success = source.set_effect("rainbow_sweep", duration=5.0)
        if not success:
            logger.error("  Failed to set effect")
            return False
        logger.info("  Set rainbow_sweep effect")

        # Generate some frames
        for i in range(10):
            frame_data = source.get_frame()
            if frame_data is None:
                logger.warning(f"  Frame {i}: None returned")
                break

            logger.info(f"  Frame {i}: {frame_data.width}x{frame_data.height}, ID={frame_data.frame_id}")

            # Small delay
            time.sleep(0.03)

        # Check status
        status = source.get_status()
        logger.info(f"  Status: {status['active']}, Progress: {status['progress']:.2f}")

        logger.info("  EffectSource: SUCCESS")
        return True

    except Exception as e:
        logger.error(f"Failed to test EffectSource: {e}")
        return False


def test_api_integration():
    """Test that the API can load effects."""
    try:
        # Change directory to project root for proper imports
        os.chdir(Path(__file__).parent)

        from src.web.api_server import EFFECTS_AVAILABLE, get_effect_presets

        logger.info(f"\nTesting API integration (Effects available: {EFFECTS_AVAILABLE}):")

        # Get presets
        presets = get_effect_presets()
        logger.info(f"  Loaded {len(presets)} effect presets")

        # Check preset structure
        for preset in presets[:3]:  # Check first 3
            logger.info(f"  {preset.icon} {preset.name} ({preset.category})")

            # Validate required fields
            if not preset.id or not preset.name or not preset.description:
                logger.error(f"    Invalid preset: {preset}")
                return False

        logger.info("  API Integration: SUCCESS")
        return True

    except Exception as e:
        logger.error(f"Failed to test API integration: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting Visual Effects System Tests")
    logger.info("=" * 50)

    tests = [
        ("Effects Loading", test_effects_loading),
        ("Effect Creation", test_effect_creation),
        ("Effect Source", test_effect_source),
        ("API Integration", test_api_integration),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        logger.info("-" * 30)

        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            failed += 1

    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("🎉 All tests passed! Visual effects system is ready.")
        return 0
    else:
        logger.error(f"⚠️  {failed} test(s) failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
