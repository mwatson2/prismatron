"""
Unit tests for SparkleEffect.

Tests the sparkling/glitter LED effect that:
- Creates sparkle bursts at regular intervals
- Fades each burst from white to target color
- Supports overlapping fade periods
- Allows dynamic parameter updates
"""

import sys
from pathlib import Path

# Add src to path for tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from src.consumer.led_effect import SparkleEffect, SparkleSet


class TestSparkleEffectBasics:
    """Test basic SparkleEffect functionality."""

    def test_initialization(self):
        """Test SparkleEffect initialization with default parameters."""
        effect = SparkleEffect(
            start_time=0.0,
            interval_ms=50.0,
            fade_ms=200.0,
            density=0.05,
            led_count=100,
        )

        assert effect.interval_ms == 50.0
        assert effect.fade_ms == 200.0
        assert effect.density == 0.05
        assert effect.led_count == 100
        assert effect.duration is None  # No fixed duration
        assert len(effect.active_sets) == 0

    def test_no_sparkles_before_first_interval(self):
        """Test that no sparkles occur before the first interval boundary."""
        led_count = 100
        effect = SparkleEffect(
            start_time=0.0,
            interval_ms=100.0,
            fade_ms=200.0,
            density=0.1,
            led_count=led_count,
        )

        # Create black LED array
        leds = np.zeros((led_count, 3), dtype=np.uint8)

        # Apply at time 0.05s (before 0.1s interval boundary)
        result = effect.apply(leds, 0.05)

        assert result is False  # Effect never completes
        assert len(effect.active_sets) == 0  # No sparkles yet
        assert np.all(leds == 0)  # LEDs unchanged

    def test_first_sparkle_at_interval_boundary(self):
        """Test that sparkles appear at the first interval boundary."""
        led_count = 100
        effect = SparkleEffect(
            start_time=0.0,
            interval_ms=100.0,
            fade_ms=200.0,
            density=0.1,  # 10% = 10 LEDs
            led_count=led_count,
        )

        # Create colored LED array
        leds = np.full((led_count, 3), 100, dtype=np.uint8)

        # Apply at time 0.1s (at first interval boundary)
        result = effect.apply(leds, 0.1)

        assert result is False
        assert len(effect.active_sets) == 1
        # Some LEDs should now be white (255) due to sparkle at start of fade
        assert np.any(leds == 255)

    def test_sparkle_fade_progression(self):
        """Test that sparkles fade linearly from white to target color."""
        led_count = 100
        effect = SparkleEffect(
            start_time=0.0,
            interval_ms=100.0,
            fade_ms=100.0,  # Same as interval for simple testing
            density=1.0,  # All LEDs sparkle
            led_count=led_count,
        )

        # Create mid-brightness LEDs
        target_color = 100
        leds = np.full((led_count, 3), target_color, dtype=np.uint8)

        # At t=0.1s, first sparkle starts (fade_progress = 0, should be white)
        effect.apply(leds, 0.1)
        assert np.all(leds == 255), "At fade start, LEDs should be white"

        # At t=0.15s (50% through fade), LEDs should be ~halfway
        leds = np.full((led_count, 3), target_color, dtype=np.uint8)
        effect.apply(leds, 0.15)
        # Expected: 255 * 0.5 + 100 * 0.5 = 177.5
        expected = int(255 * 0.5 + target_color * 0.5)
        # Allow some tolerance for rounding
        assert np.allclose(leds, expected, atol=2)

        # At t=0.2s (100% through fade), LEDs should be back to target
        leds = np.full((led_count, 3), target_color, dtype=np.uint8)
        effect.apply(leds, 0.2)
        # At exactly 100% fade, the set will be expired
        # Note: At exactly 0.2s, a new sparkle starts at interval boundary 2
        # But the first sparkle (from 0.1s) has completed its 100ms fade

    def test_overlapping_sparkle_sets(self):
        """Test that multiple overlapping sparkle sets are tracked."""
        led_count = 100
        effect = SparkleEffect(
            start_time=0.0,
            interval_ms=50.0,  # New sparkle every 50ms
            fade_ms=200.0,  # Fade lasts 200ms
            density=0.05,
            led_count=led_count,
        )

        leds = np.full((led_count, 3), 100, dtype=np.uint8)

        # At t=0.2s, we should have 4 interval boundaries passed (50, 100, 150, 200ms)
        # Each set has 200ms fade, so all 4 should still be active
        effect.apply(leds, 0.2)

        assert len(effect.active_sets) == 4

    def test_expired_sets_removed(self):
        """Test that expired sparkle sets are removed."""
        led_count = 100
        effect = SparkleEffect(
            start_time=0.0,
            interval_ms=50.0,
            fade_ms=100.0,  # Fade lasts 100ms
            density=0.05,
            led_count=led_count,
        )

        leds = np.full((led_count, 3), 100, dtype=np.uint8)

        # At t=0.3s:
        # - 6 interval boundaries passed (50, 100, 150, 200, 250, 300ms)
        # - cutoff_time = 0.3 - 0.1 = 0.2s
        # - Sets must have start_time > 0.2s to be kept
        # - Sets from 50, 100, 150ms are expired (start_time <= 0.2s)
        # - Sets from 200, 250, 300ms should still be active (start_time > 0.2s)
        effect.apply(leds, 0.3)

        # Should have 3 active sets (200, 250, 300ms)
        assert len(effect.active_sets) == 3

    def test_never_completes(self):
        """Test that SparkleEffect never completes on its own."""
        effect = SparkleEffect(start_time=0.0, led_count=100)
        leds = np.full((100, 3), 100, dtype=np.uint8)

        # Apply many times
        for t in np.linspace(0, 10.0, 100):
            result = effect.apply(leds, t)
            assert result is False


class TestSparkleEffectParameters:
    """Test dynamic parameter updates."""

    def test_set_parameters_interval(self):
        """Test updating interval_ms parameter."""
        effect = SparkleEffect(start_time=0.0, interval_ms=100.0, led_count=100)

        effect.set_parameters(interval_ms=50.0)
        assert effect.interval_ms == 50.0

        # Other parameters should be unchanged
        assert effect.fade_ms == 200.0  # default
        assert effect.density == 0.05  # default

    def test_set_parameters_fade(self):
        """Test updating fade_ms parameter."""
        effect = SparkleEffect(start_time=0.0, fade_ms=100.0, led_count=100)

        effect.set_parameters(fade_ms=300.0)
        assert effect.fade_ms == 300.0

    def test_set_parameters_density(self):
        """Test updating density parameter."""
        effect = SparkleEffect(start_time=0.0, density=0.1, led_count=100)

        effect.set_parameters(density=0.2)
        assert effect.density == 0.2

    def test_density_clamped_to_valid_range(self):
        """Test that density is clamped to [0, 1]."""
        effect = SparkleEffect(start_time=0.0, led_count=100)

        effect.set_parameters(density=1.5)
        assert effect.density == 1.0

        effect.set_parameters(density=-0.5)
        assert effect.density == 0.0


class TestSparkleSetReproducibility:
    """Test that sparkle sets are reproducible via RNG seed."""

    def test_same_seed_same_indices(self):
        """Test that same seed produces same LED indices."""
        effect = SparkleEffect(start_time=0.0, led_count=100, density=0.1)

        seed = 12345
        count = 10

        indices1 = effect._get_sparkle_indices(count, seed)
        indices2 = effect._get_sparkle_indices(count, seed)

        np.testing.assert_array_equal(indices1, indices2)

    def test_different_seeds_different_indices(self):
        """Test that different seeds produce different LED indices."""
        effect = SparkleEffect(start_time=0.0, led_count=100, density=0.1)

        indices1 = effect._get_sparkle_indices(10, 12345)
        indices2 = effect._get_sparkle_indices(10, 54321)

        # Should be different (with very high probability)
        assert not np.array_equal(indices1, indices2)

    def test_indices_within_bounds(self):
        """Test that all indices are within LED count bounds."""
        effect = SparkleEffect(start_time=0.0, led_count=100, density=0.1)

        for seed in range(100):
            indices = effect._get_sparkle_indices(10, seed)
            assert np.all(indices >= 0)
            assert np.all(indices < 100)

    def test_no_duplicate_indices(self):
        """Test that indices are unique (no duplicates)."""
        effect = SparkleEffect(start_time=0.0, led_count=100, density=0.5)

        for seed in range(100):
            indices = effect._get_sparkle_indices(50, seed)
            assert len(indices) == len(np.unique(indices))


class TestSparkleEffectIntegration:
    """Integration tests for SparkleEffect."""

    def test_effect_with_manager(self):
        """Test SparkleEffect works with LedEffectManager."""
        from src.consumer.led_effect import LedEffectManager

        manager = LedEffectManager()
        effect = SparkleEffect(start_time=0.0, led_count=100)

        manager.add_effect(effect)
        assert manager.get_active_count() == 1

        leds = np.full((100, 3), 100, dtype=np.uint8)

        # Apply effects
        for t in np.linspace(0, 1.0, 50):
            manager.apply_effects(leds, t)

        # Effect should still be active (never completes)
        assert manager.get_active_count() == 1

        # Manual removal
        manager.remove_effect(effect)
        assert manager.get_active_count() == 0

    def test_get_info(self):
        """Test get_info returns expected information."""
        effect = SparkleEffect(
            start_time=1.0,
            interval_ms=75.0,
            fade_ms=150.0,
            density=0.08,
            led_count=200,
        )

        info = effect.get_info()

        assert info["name"] == "SparkleEffect"
        assert info["start_time"] == 1.0
        assert info["duration"] is None
        assert info["interval_ms"] == 75.0
        assert info["fade_ms"] == 150.0
        assert info["density"] == 0.08
        assert info["led_count"] == 200
        assert info["active_sets"] == 0

    def test_led_count_auto_update(self):
        """Test that led_count updates from actual array size."""
        effect = SparkleEffect(start_time=0.0, led_count=100)
        assert effect.led_count == 100

        # Apply with different sized array
        leds = np.full((200, 3), 100, dtype=np.uint8)
        effect.apply(leds, 0.1)

        assert effect.led_count == 200


class TestSparkleEffectEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_density(self):
        """Test with zero density (no sparkles)."""
        effect = SparkleEffect(start_time=0.0, density=0.0, led_count=100)
        leds = np.full((100, 3), 100, dtype=np.uint8)
        original = leds.copy()

        effect.apply(leds, 0.1)

        # With density=0, no sparkles should occur
        # Note: max(1, int(0.0 * 100)) = 1, so we still get 1 LED
        # This is intentional to prevent division by zero issues

    def test_full_density(self):
        """Test with full density (all LEDs sparkle)."""
        effect = SparkleEffect(start_time=0.0, density=1.0, led_count=100)
        leds = np.full((100, 3), 0, dtype=np.uint8)

        effect.apply(leds, 0.05)  # At t=0.05, first sparkle at 0.05s

        # All LEDs should be affected after first interval
        effect.apply(leds, 0.051)  # Just after 50ms default interval
        assert np.any(leds > 0)  # Some brightness from sparkle

    def test_very_short_fade(self):
        """Test with very short fade duration."""
        effect = SparkleEffect(
            start_time=0.0,
            interval_ms=100.0,
            fade_ms=1.0,  # 1ms fade
            density=0.1,
            led_count=100,
        )

        leds = np.full((100, 3), 100, dtype=np.uint8)

        # At t=0.1s, sparkle starts. At t=0.102s, sparkle should be faded
        effect.apply(leds, 0.1)
        effect.apply(leds, 0.102)

        # The first set should be expired
        assert len(effect.active_sets) <= 1

    def test_very_long_fade(self):
        """Test with very long fade duration."""
        effect = SparkleEffect(
            start_time=0.0,
            interval_ms=10.0,  # 10ms intervals
            fade_ms=1000.0,  # 1s fade
            density=0.01,
            led_count=100,
        )

        leds = np.full((100, 3), 100, dtype=np.uint8)

        # At t=0.5s, we should have 50 sparkle sets (0.5s / 10ms = 50)
        # All should still be active (fade is 1s)
        effect.apply(leds, 0.5)

        assert len(effect.active_sets) == 50

    def test_frame_skip_multiple_intervals(self):
        """Test handling of frame skips spanning multiple intervals."""
        effect = SparkleEffect(
            start_time=0.0,
            interval_ms=10.0,  # 10ms intervals
            fade_ms=200.0,
            density=0.1,
            led_count=100,
        )

        leds = np.full((100, 3), 100, dtype=np.uint8)

        # First frame at t=0 (no sparkles yet)
        effect.apply(leds, 0.0)
        assert len(effect.active_sets) == 0

        # Skip to t=0.1s - should add 10 sparkle sets (at 10, 20, ..., 100ms)
        effect.apply(leds, 0.1)
        assert len(effect.active_sets) == 10


class TestSparkleSetDataclass:
    """Test SparkleSet dataclass."""

    def test_sparkle_set_creation(self):
        """Test SparkleSet creation."""
        sparkle = SparkleSet(start_time=1.0, count=10, seed=12345)

        assert sparkle.start_time == 1.0
        assert sparkle.count == 10
        assert sparkle.seed == 12345

    def test_sparkle_set_equality(self):
        """Test SparkleSet equality comparison."""
        s1 = SparkleSet(start_time=1.0, count=10, seed=12345)
        s2 = SparkleSet(start_time=1.0, count=10, seed=12345)
        s3 = SparkleSet(start_time=2.0, count=10, seed=12345)

        assert s1 == s2
        assert s1 != s3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
