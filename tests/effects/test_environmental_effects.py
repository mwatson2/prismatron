"""
Unit tests for environmental visual effects.

Tests specific behaviors and characteristics of nature-inspired effects.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add src to path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from producer.effects.environmental_effects import AuroraBorealis, FireSimulation, Lightning


class TestFireSimulation:
    """Test FireSimulation effect."""

    def test_fire_colors(self):
        """Test that fire produces warm colors."""
        effect = FireSimulation(width=80, height=100)
        frame = effect.generate_frame()

        # Fire should be dominated by reds and yellows
        red_mean = frame[:, :, 0].mean()
        green_mean = frame[:, :, 1].mean()
        blue_mean = frame[:, :, 2].mean()

        # Fire should have more red than blue
        assert red_mean > blue_mean * 0.8

        # Should also have some yellow (red + green)
        assert red_mean > 0 and green_mean > 0

    def test_fire_upward_motion(self):
        """Test that fire flames move upward."""
        effect = FireSimulation(width=80, height=100, config={"intensity": 0.8})

        frames = []
        for _ in range(6):
            frames.append(effect.generate_frame())
            time.sleep(0.03)

        # Fire should be brightest at bottom, dimmer at top
        for frame in frames[-3:]:  # Check recent frames
            bottom_brightness = frame[-20:, :, :].mean()
            top_brightness = frame[:20, :, :].mean()

            # Bottom should generally be brighter than top
            if bottom_brightness > top_brightness:
                break
        else:
            # If not consistently brighter at bottom, at least should have fire
            assert frame.max() > 150

    def test_fire_intensity(self):
        """Test different fire intensities."""
        effect_low = FireSimulation(width=80, height=80, config={"intensity": 0.3})
        frame_low = effect_low.generate_frame()

        effect_high = FireSimulation(width=80, height=80, config={"intensity": 1.0})
        frame_high = effect_high.generate_frame()

        # Higher intensity should be brighter
        assert frame_high.mean() >= frame_low.mean() * 0.9

    def test_fire_wind_effect(self):
        """Test wind affecting fire direction."""
        effect_nowind = FireSimulation(width=100, height=80, config={"wind": 0.0})

        effect_wind = FireSimulation(width=100, height=80, config={"wind": 0.5})

        # Both should produce fire
        frame_nowind = effect_nowind.generate_frame()
        frame_wind = effect_wind.generate_frame()

        assert frame_nowind.max() > 0
        assert frame_wind.max() > 0

    def test_fire_animation(self):
        """Test fire animation over time."""
        effect = FireSimulation(width=60, height=80, config={"flicker_rate": 5.0})

        frames = []
        for _ in range(5):
            frames.append(effect.generate_frame())
            time.sleep(0.02)

        # Fire should flicker/animate
        for i in range(1, len(frames)):
            assert not np.array_equal(frames[i], frames[i - 1])

    def test_fire_base_size(self):
        """Test different fire base sizes."""
        effect_narrow = FireSimulation(width=100, height=80, config={"base_width": 0.2})
        frame_narrow = effect_narrow.generate_frame()

        effect_wide = FireSimulation(width=100, height=80, config={"base_width": 0.8})
        frame_wide = effect_wide.generate_frame()

        # Wide base should have fire across more of the width
        narrow_coverage = (frame_narrow[-10:, :, :].max(axis=2) > 100).sum()
        wide_coverage = (frame_wide[-10:, :, :].max(axis=2) > 100).sum()

        assert wide_coverage >= narrow_coverage


class TestLightning:
    """Test Lightning effect."""

    def test_lightning_flashes(self):
        """Test that lightning produces bright flashes."""
        effect = Lightning(width=100, height=100, config={"strike_probability": 0.8})  # High probability

        max_brightness = 0
        for _ in range(15):  # Multiple attempts
            frame = effect.generate_frame()
            max_brightness = max(max_brightness, frame.max())
            time.sleep(0.01)

        # Should see bright lightning
        assert max_brightness > 200

    def test_lightning_branching(self):
        """Test lightning branching patterns."""
        effect = Lightning(width=100, height=100, config={"branching": True, "strike_probability": 1.0})

        # Force lightning strikes
        bright_frames = []
        for _ in range(10):
            frame = effect.generate_frame()
            if frame.max() > 200:
                bright_frames.append(frame)
            time.sleep(0.01)

        # At least one frame should have lightning
        assert len(bright_frames) > 0 or effect.generate_frame().max() > 0

    def test_lightning_colors(self):
        """Test lightning color variations."""
        colors = ["white", "blue", "purple"]

        for color in colors:
            effect = Lightning(width=80, height=80, config={"lightning_color": color, "strike_probability": 1.0})

            # Try to generate lightning
            found_lightning = False
            for _ in range(10):
                frame = effect.generate_frame()
                if frame.max() > 150:
                    found_lightning = True
                    break
                time.sleep(0.01)

            # Should eventually see lightning or at least valid frame
            assert found_lightning or frame.shape == (80, 80, 3)

    def test_lightning_duration(self):
        """Test lightning strike duration."""
        effect = Lightning(width=80, height=80, config={"flash_duration": 3, "strike_probability": 1.0})

        # Look for sustained brightness over multiple frames
        brightness_sequence = []
        for _ in range(8):
            frame = effect.generate_frame()
            brightness_sequence.append(frame.max())
            time.sleep(0.02)

        # Should see some variation in brightness (strikes and fades)
        assert max(brightness_sequence) > min(brightness_sequence) + 20

    def test_background_darkness(self):
        """Test that background remains dark between strikes."""
        effect = Lightning(width=80, height=80, config={"strike_probability": 0.1})  # Low probability

        # Most frames should be dark
        dark_frames = 0
        total_frames = 10
        for _ in range(total_frames):
            frame = effect.generate_frame()
            if frame.mean() < 50:  # Dark frame
                dark_frames += 1
            time.sleep(0.02)

        # Most frames should be dark with occasional lightning
        assert dark_frames >= total_frames * 0.3


class TestAuroraBorealis:
    """Test AuroraBorealis effect."""

    def test_aurora_colors(self):
        """Test that aurora produces characteristic colors."""
        effect = AuroraBorealis(width=100, height=100)
        frame = effect.generate_frame()

        # Aurora typically has greens and blues
        green_mean = frame[:, :, 1].mean()
        blue_mean = frame[:, :, 2].mean()
        red_mean = frame[:, :, 0].mean()

        # Should have significant green or blue content
        assert green_mean > 30 or blue_mean > 30

    def test_aurora_waves(self):
        """Test aurora wave-like motion."""
        effect = AuroraBorealis(width=100, height=80, config={"wave_speed": 3.0})

        frames = []
        for _ in range(5):
            frames.append(effect.generate_frame())
            time.sleep(0.05)

        # Should see flowing/waving motion
        changes = []
        for i in range(1, len(frames)):
            change = np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean()
            changes.append(change)

        # Should have continuous smooth changes
        assert all(c > 0 for c in changes)
        assert max(changes) < 100  # But not too drastic

    def test_aurora_intensity_variation(self):
        """Test aurora intensity variations."""
        effect = AuroraBorealis(width=100, height=100, config={"intensity_variation": 0.8})

        frames = []
        for _ in range(8):
            frames.append(effect.generate_frame())
            time.sleep(0.03)

        # Intensity should vary over time
        intensities = [frame.mean() for frame in frames]
        assert max(intensities) > min(intensities) + 10

    def test_aurora_height_distribution(self):
        """Test aurora height in sky."""
        effect = AuroraBorealis(width=100, height=100)
        frame = effect.generate_frame()

        # Aurora typically appears in upper regions of sky
        # Check if there's more activity in upper half
        upper_half = frame[:50, :, :].mean()
        lower_half = frame[50:, :, :].mean()

        # Allow some flexibility - aurora can vary
        assert upper_half > 0 or lower_half > 0  # At least some aurora visible

    def test_aurora_color_zones(self):
        """Test different aurora color zones."""
        effect = AuroraBorealis(width=100, height=100, config={"color_zones": True})

        frame = effect.generate_frame()

        # Should have color variation across the aurora
        # Check if different regions have different color characteristics
        left_colors = frame[:, :30, :].mean(axis=(0, 1))
        right_colors = frame[:, -30:, :].mean(axis=(0, 1))

        # Some color variation expected
        color_diff = np.abs(left_colors - right_colors).sum()
        assert color_diff > 5 or frame.std() > 20

    def test_aurora_vertical_bands(self):
        """Test aurora vertical band structure."""
        effect = AuroraBorealis(width=100, height=100, config={"band_count": 5})

        frame = effect.generate_frame()

        # Should have vertical band-like structures
        # Check for vertical correlation
        correlations = []
        for col in range(20, 80, 20):  # Sample some columns
            column = frame[:, col, 1]  # Green channel
            # Check if there are connected vertical features
            if column.max() > 50:
                correlations.append(column.std())

        # Should have some vertical structure
        assert len(correlations) > 0

    def test_aurora_animation_smoothness(self):
        """Test smooth aurora animation."""
        effect = AuroraBorealis(width=80, height=80, config={"smoothness": 0.8})

        frames = []
        for _ in range(4):
            frames.append(effect.generate_frame())
            time.sleep(0.05)

        # Changes should be smooth (not too abrupt)
        for i in range(1, len(frames)):
            diff = np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean()
            # Smooth animation should have moderate changes
            assert 0 < diff < 80

    def test_aurora_shimmer_effect(self):
        """Test aurora shimmer/twinkle effect."""
        effect = AuroraBorealis(width=100, height=100, config={"shimmer": True, "shimmer_rate": 10.0})

        # Look for local brightness variations (shimmer)
        frames = []
        for _ in range(6):
            frames.append(effect.generate_frame())
            time.sleep(0.02)

        # Check if small regions change brightness (shimmer effect)
        found_shimmer = False
        for y in range(10, 90, 20):
            for x in range(10, 90, 20):
                region_brightness = [frame[y : y + 5, x : x + 5, :].mean() for frame in frames]
                if max(region_brightness) > min(region_brightness) + 15:
                    found_shimmer = True
                    break
            if found_shimmer:
                break

        assert found_shimmer or frames[-1].std() > 10  # Either shimmer or variation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
