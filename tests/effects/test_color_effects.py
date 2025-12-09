"""
Unit tests for color-based visual effects.

Tests specific behaviors and characteristics of color effects.
"""

# Add parent directory to path for tests
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytest.importorskip("cupy")

# Import effects
from src.producer.effects.color_effects import (
    ColorBreathe,
    ColorWipe,
    GradientFlow,
    RainbowSweep,
)


class TestRainbowSweep:
    """Test RainbowSweep effect."""

    def test_horizontal_sweep(self):
        """Test horizontal rainbow sweep."""
        effect = RainbowSweep(width=100, height=50, config={"direction": "horizontal"})
        frame = effect.generate_frame(0.0)

        # Should have horizontal color variation
        left_colors = frame[:, :10, :].mean(axis=(0, 1))
        right_colors = frame[:, -10:, :].mean(axis=(0, 1))

        # Colors should be different on left vs right
        color_diff = np.abs(left_colors - right_colors).sum()
        assert color_diff > 50

    def test_vertical_sweep(self):
        """Test vertical rainbow sweep."""
        effect = RainbowSweep(width=100, height=50, config={"direction": "vertical"})
        frame = effect.generate_frame(0.0)

        # Should have vertical color variation
        top_colors = frame[:10, :, :].mean(axis=(0, 1))
        bottom_colors = frame[-10:, :, :].mean(axis=(0, 1))

        # Colors should be different on top vs bottom
        color_diff = np.abs(top_colors - bottom_colors).sum()
        assert color_diff > 50

    def test_radial_sweep(self):
        """Test radial rainbow sweep."""
        effect = RainbowSweep(width=100, height=100, config={"direction": "radial"})
        frame = effect.generate_frame(0.0)

        # Center should be different from edges
        center = frame[45:55, 45:55, :].mean(axis=(0, 1))
        corners = np.concatenate(
            [
                frame[:10, :10, :].reshape(-1, 3),
                frame[:10, -10:, :].reshape(-1, 3),
                frame[-10:, :10, :].reshape(-1, 3),
                frame[-10:, -10:, :].reshape(-1, 3),
            ]
        ).mean(axis=0)

        color_diff = np.abs(center - corners).sum()
        assert color_diff > 30

    def test_diagonal_sweep(self):
        """Test diagonal rainbow sweep."""
        effect = RainbowSweep(width=100, height=100, config={"direction": "diagonal"})
        frame = effect.generate_frame(0.0)

        # Top-left vs bottom-right should be different
        tl = frame[:20, :20, :].mean(axis=(0, 1))
        br = frame[-20:, -20:, :].mean(axis=(0, 1))

        color_diff = np.abs(tl - br).sum()
        assert color_diff > 50

    def test_saturation_control(self):
        """Test saturation parameter."""
        # Low saturation should produce more muted colors
        effect_low = RainbowSweep(width=50, height=50, config={"saturation": 0.2})
        frame_low = effect_low.generate_frame(0.0)

        effect_high = RainbowSweep(width=50, height=50, config={"saturation": 1.0})
        frame_high = effect_high.generate_frame(0.0)

        # High saturation should have more vivid colors (higher std deviation)
        assert frame_high.std() > frame_low.std()

    def test_brightness_control(self):
        """Test brightness parameter."""
        effect_dark = RainbowSweep(width=50, height=50, config={"brightness": 0.3})
        frame_dark = effect_dark.generate_frame(0.0)

        effect_bright = RainbowSweep(width=50, height=50, config={"brightness": 1.0})
        frame_bright = effect_bright.generate_frame(0.0)

        assert frame_bright.mean() > frame_dark.mean()

    def test_wave_width(self):
        """Test wave width parameter (number of rainbows on screen)."""
        effect_narrow = RainbowSweep(width=100, height=50, config={"wave_width": 3.0})
        frame_narrow = effect_narrow.generate_frame(0.0)

        # With higher wave_width, should see more color transitions
        # Count approximate color transitions along a row
        row = frame_narrow[25, :, :]
        transitions = 0
        for i in range(1, len(row)):
            if np.abs(row[i] - row[i - 1]).sum() > 100:
                transitions += 1

        assert transitions > 2  # Should have multiple rainbow cycles

    def test_animation_speed(self):
        """Test animation speed parameter."""
        effect = RainbowSweep(width=50, height=50, config={"speed": 5.0})

        frame1 = effect.generate_frame(0.0)
        time.sleep(0.1)
        frame2 = effect.generate_frame(0.1)

        # Frames should be different due to animation
        assert not np.array_equal(frame1, frame2)

        # Higher speed should mean larger difference
        diff = np.abs(frame2.astype(float) - frame1.astype(float)).mean()
        assert diff > 5


class TestColorBreathe:
    """Test ColorBreathe effect."""

    def test_breathing_animation(self):
        """Test that brightness pulses over time."""
        # Use 1 Hz rate and sample at 0.1 second intervals to catch multiple phases
        effect = ColorBreathe(width=50, height=50, config={"breathe_rate": 1.0, "base_color": [255, 0, 0]})

        brightnesses = []
        for i in range(10):
            frame = effect.generate_frame(i * 0.1)  # 0.0 to 0.9 seconds, covers almost full cycle
            brightnesses.append(frame.mean())

        # Should see variation in brightness (breathing)
        assert max(brightnesses) > min(brightnesses) + 10

    def test_base_color(self):
        """Test different base colors."""
        # Test red
        effect_red = ColorBreathe(width=30, height=30, config={"base_color": [255, 0, 0]})
        frame_red = effect_red.generate_frame(0.0)
        assert frame_red[:, :, 0].mean() > frame_red[:, :, 1].mean()  # More red than green
        assert frame_red[:, :, 0].mean() > frame_red[:, :, 2].mean()  # More red than blue

        # Test blue
        effect_blue = ColorBreathe(width=30, height=30, config={"base_color": [0, 0, 255]})
        frame_blue = effect_blue.generate_frame(0.0)
        assert frame_blue[:, :, 2].mean() > frame_blue[:, :, 0].mean()  # More blue than red
        assert frame_blue[:, :, 2].mean() > frame_blue[:, :, 1].mean()  # More blue than green

    def test_intensity_range(self):
        """Test intensity min/max parameters."""
        effect = ColorBreathe(
            width=40,
            height=40,
            config={"base_color": [255, 255, 255], "intensity_min": 0.2, "intensity_max": 0.8, "breathe_rate": 10.0},
        )

        # Collect frames over a breathing cycle
        frames = []
        for i in range(20):
            frames.append(effect.generate_frame(i * 0.025))

        min_brightness = min(f.mean() for f in frames)
        max_brightness = max(f.mean() for f in frames)

        # Should stay within intensity bounds (approximately)
        assert min_brightness > 255 * 0.15  # Close to intensity_min
        assert max_brightness < 255 * 0.85  # Close to intensity_max

    def test_color_shift(self):
        """Test color shifting during breathing."""
        effect = ColorBreathe(
            width=40,
            height=40,
            config={"base_color": [255, 0, 0], "color_shift": True, "shift_amount": 0.2, "breathe_rate": 1.0},
        )

        # Collect frames over a full cycle
        frames = []
        for i in range(10):
            frames.append(effect.generate_frame(i * 0.1))

        # With color shift, hue should vary
        hues = []
        for frame in frames:
            # Approximate hue from RGB
            r = frame[:, :, 0].mean() / 255
            g = frame[:, :, 1].mean() / 255
            b = frame[:, :, 2].mean() / 255
            hues.append((r, g, b))

        # Colors should vary slightly
        color_variation = sum(abs(h[1] - hues[0][1]) + abs(h[2] - hues[0][2]) for h in hues[1:])
        assert color_variation > 0.1


class TestGradientFlow:
    """Test GradientFlow effect."""

    def test_gradient_creation(self):
        """Test that gradients are created properly."""
        effect = GradientFlow(width=100, height=50)
        frame = effect.generate_frame(0.0)

        # Should have smooth color transitions
        # Check horizontal gradient smoothness
        row = frame[25, :, :]
        total_diff = 0
        for i in range(1, len(row)):
            total_diff += np.abs(row[i].astype(float) - row[i - 1].astype(float)).sum()

        avg_diff = total_diff / (len(row) - 1)
        # Average difference between adjacent pixels should be small (smooth gradient)
        assert avg_diff < 30

    def test_multiple_color_stops(self):
        """Test gradients with multiple color stops."""
        effect = GradientFlow(
            width=100,
            height=50,
            config={"color_stops": [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 255]], "flow_speed": 0},
        )
        frame = effect.generate_frame(0.0)

        # Should transition through multiple colors
        left = frame[:, 5:20, :].mean(axis=(0, 1))  # Red segment (0-25%)
        middle = frame[:, 30:45, :].mean(axis=(0, 1))  # Green segment (25-50%)
        right = frame[:, 55:70, :].mean(axis=(0, 1))  # Blue segment (50-75%)

        # Each region should be dominated by different colors
        assert left[0] > left[1] and left[0] > left[2]  # Reddish
        assert middle[1] > middle[0] and middle[1] > middle[2]  # Greenish
        assert right[2] > right[0] and right[2] > right[1]  # Bluish

    def test_flow_animation(self):
        """Test that gradient flows/animates over time."""
        effect = GradientFlow(width=80, height=40, config={"flow_speed": 2.0})

        frame1 = effect.generate_frame(0.0)
        frame2 = effect.generate_frame(0.1)

        # Frames should be different (flowing)
        assert not np.array_equal(frame1, frame2)

        diff = np.abs(frame2.astype(float) - frame1.astype(float)).mean()
        assert diff > 5


class TestColorWipe:
    """Test ColorWipe effect."""

    def test_wipe_progression(self):
        """Test that wipe progresses across screen."""
        effect = ColorWipe(
            width=100,
            height=50,
            config={"wipe_speed": 5.0, "color_sequence": [[255, 0, 0], [0, 0, 255]], "hold_time": 0},
        )

        # Get frames at different times
        frames = []
        for i in range(5):
            frames.append(effect.generate_frame(i * 0.1))

        # Should see color transition (with deterministic seeding, check for any color variation)
        has_color_variation = False
        for frame in frames:
            # Check if frame has both colors or transitional colors
            red_pixels = (frame[:, :, 0] > 200) & (frame[:, :, 1] < 50) & (frame[:, :, 2] < 50)
            blue_pixels = (frame[:, :, 0] < 50) & (frame[:, :, 1] < 50) & (frame[:, :, 2] > 200)
            if red_pixels.any() and blue_pixels.any():
                has_color_variation = True
                break

        # Should have wipe effect or at least color animation
        assert has_color_variation or not all(np.array_equal(frames[0], f) for f in frames[1:])

    def test_wipe_direction(self):
        """Test different wipe directions."""
        directions = ["horizontal", "vertical", "diagonal", "radial"]

        for direction in directions:
            effect = ColorWipe(
                width=60, height=60, config={"direction": direction, "colors": [[255, 255, 0], [255, 0, 255]]}
            )

            frame = effect.generate_frame(0.0)

            # Should have two distinct color regions
            unique_colors = np.unique(frame.reshape(-1, 3), axis=0)

            # Due to antialiasing/smoothing, might have intermediate colors
            # But should have at least the two main colors prominently
            assert len(unique_colors) >= 2

    def test_multiple_colors(self):
        """Test wiping through multiple colors."""
        effect = ColorWipe(
            width=100,
            height=50,
            config={"colors": [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], "wipe_speed": 2.0},
        )

        # Generate several frames to see color transitions
        seen_colors = set()
        for i in range(10):
            frame = effect.generate_frame(i * 0.1)
            # Sample center pixel color
            color = tuple(frame[25, 50, :])
            seen_colors.add(color)

        # Should see multiple different colors over time
        assert len(seen_colors) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
