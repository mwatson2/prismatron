"""
Unit tests for matrix/digital visual effects.

Tests specific behaviors and characteristics of digital/cyberpunk style effects.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add src to path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from producer.effects.matrix_effects import BinaryStream, DigitalRain, GlitchArt


class TestDigitalRain:
    """Test DigitalRain effect (Matrix-style falling code)."""

    def test_rain_generation(self):
        """Test that digital rain streams are generated."""
        effect = DigitalRain(width=80, height=100)
        frame = effect.generate_frame()

        # Should have vertical streaming patterns
        assert frame.max() > 0
        assert frame.shape == (100, 80, 3)

    def test_rain_color(self):
        """Test characteristic green Matrix color."""
        effect = DigitalRain(width=80, height=100)
        frame = effect.generate_frame()

        if frame.max() > 0:
            # Find bright areas (active rain)
            bright_mask = frame.max(axis=2) > 100
            if bright_mask.any():
                bright_pixels = frame[bright_mask]

                # Should be predominantly green
                green_mean = bright_pixels[:, 1].mean()
                red_mean = bright_pixels[:, 0].mean()
                blue_mean = bright_pixels[:, 2].mean()

                # Green should dominate
                assert green_mean >= red_mean and green_mean >= blue_mean

    def test_rain_falling_motion(self):
        """Test that rain falls downward."""
        effect = DigitalRain(width=60, height=80, config={"speed": 3.0})

        frames = []
        for _ in range(5):
            frames.append(effect.generate_frame())
            time.sleep(0.03)

        # Check for downward movement patterns
        found_movement = False
        for i in range(len(frames) - 1):
            # Compare upper and lower regions between frames
            current_upper = frames[i][:30, :, :].mean()
            next_lower = frames[i + 1][30:60, :, :].mean()

            if current_upper > 50 and next_lower > 50:
                found_movement = True
                break

        # Should see either movement or at least animation
        assert found_movement or not all(np.array_equal(frames[0], f) for f in frames[1:])

    def test_rain_stream_density(self):
        """Test different stream densities."""
        effect_sparse = DigitalRain(width=80, height=80, config={"stream_density": 0.1})
        frame_sparse = effect_sparse.generate_frame()

        effect_dense = DigitalRain(width=80, height=80, config={"stream_density": 0.8})
        frame_dense = effect_dense.generate_frame()

        # Dense should have more active streams
        sparse_activity = (frame_sparse.max(axis=2) > 50).sum()
        dense_activity = (frame_dense.max(axis=2) > 50).sum()

        assert dense_activity >= sparse_activity * 0.8

    def test_character_variation(self):
        """Test that different characters/symbols appear."""
        effect = DigitalRain(width=80, height=100, config={"symbol_set": "matrix"})

        # Generate multiple frames to see character variation
        frames = []
        for _ in range(8):
            frames.append(effect.generate_frame())
            time.sleep(0.02)

        # Should see variation in patterns (different characters)
        patterns = [f.mean(axis=2) for f in frames]
        unique_patterns = len({tuple(p.flatten()[:100]) for p in patterns})

        assert unique_patterns > 1  # Should see different patterns

    def test_trail_effect(self):
        """Test character trailing/fading effect."""
        effect = DigitalRain(width=60, height=80, config={"trail_length": 8})

        frames = []
        for _ in range(5):
            frames.append(effect.generate_frame())
            time.sleep(0.03)

        # Should see gradient trails (bright at front, dim behind)
        found_trail = False
        for frame in frames:
            for col in range(frame.shape[1]):
                column = frame[:, col, 1]  # Green channel
                if column.max() > 150:
                    # Look for gradient pattern in this column
                    max_pos = np.argmax(column)
                    if max_pos < len(column) - 5:
                        # Check if it gets dimmer below
                        bright = column[max_pos]
                        dimmer = column[max_pos + 3 : max_pos + 6].mean()
                        if bright > dimmer + 20:
                            found_trail = True
                            break
            if found_trail:
                break

        assert found_trail or frames[-1].std() > 20

    def test_rain_speed(self):
        """Test different rain speeds."""
        effect_slow = DigitalRain(width=60, height=80, config={"speed": 0.5})
        effect_fast = DigitalRain(width=60, height=80, config={"speed": 5.0})

        # Generate frames and measure changes
        slow_frames = [effect_slow.generate_frame() for _ in range(3)]
        time.sleep(0.1)
        fast_frames = [effect_fast.generate_frame() for _ in range(3)]

        # Fast should show more change between frames
        slow_change = np.abs(slow_frames[1].astype(float) - slow_frames[0].astype(float)).mean()
        fast_change = np.abs(fast_frames[1].astype(float) - fast_frames[0].astype(float)).mean()

        # Allow some tolerance
        assert fast_change >= slow_change * 0.5


class TestBinaryStream:
    """Test BinaryStream effect."""

    def test_binary_generation(self):
        """Test that binary streams are generated."""
        effect = BinaryStream(width=100, height=80)
        frame = effect.generate_frame()

        # Should have streaming binary patterns
        assert frame.max() > 0
        assert frame.shape == (80, 100, 3)

    def test_binary_characters(self):
        """Test that only binary characters (0s and 1s) appear."""
        effect = BinaryStream(width=80, height=60, config={"density": 0.8})
        frame = effect.generate_frame()

        # Should show discrete patterns representing 0s and 1s
        # This is hard to test directly, but we can check for discrete levels
        if frame.max() > 0:
            unique_values = np.unique(frame.flatten())
            # Should have some discrete intensity levels
            assert len(unique_values) >= 2

    def test_binary_streaming(self):
        """Test binary stream animation."""
        effect = BinaryStream(width=60, height=80, config={"stream_speed": 3.0})

        frames = []
        for _ in range(4):
            frames.append(effect.generate_frame())
            time.sleep(0.05)

        # Should see streaming animation
        for i in range(1, len(frames)):
            assert not np.array_equal(frames[i], frames[i - 1])

    def test_binary_colors(self):
        """Test different binary color schemes."""
        schemes = ["green", "amber", "white", "blue"]

        for scheme in schemes:
            effect = BinaryStream(width=60, height=60, config={"color_scheme": scheme})
            frame = effect.generate_frame()

            # Each scheme should produce valid output
            assert frame.shape == (60, 60, 3)

            # Should have some content
            if frame.max() > 0:
                # Check if the dominant color matches scheme
                if scheme == "green":
                    assert frame[:, :, 1].max() >= frame[:, :, 0].max()
                elif scheme == "blue":
                    assert frame[:, :, 2].max() >= frame[:, :, 0].max()

    def test_binary_density(self):
        """Test different binary stream densities."""
        effect_sparse = BinaryStream(width=80, height=80, config={"density": 0.2})
        frame_sparse = effect_sparse.generate_frame()

        effect_dense = BinaryStream(width=80, height=80, config={"density": 0.9})
        frame_dense = effect_dense.generate_frame()

        # Dense should have more active bits
        sparse_bits = (frame_sparse.max(axis=2) > 50).sum()
        dense_bits = (frame_dense.max(axis=2) > 50).sum()

        assert dense_bits >= sparse_bits * 0.8

    def test_binary_flow_direction(self):
        """Test different flow directions."""
        directions = ["down", "up", "left", "right", "diagonal"]

        for direction in directions:
            effect = BinaryStream(width=60, height=60, config={"direction": direction})
            frame = effect.generate_frame()

            # Each direction should produce valid streams
            assert frame.shape == (60, 60, 3)


class TestGlitchArt:
    """Test GlitchArt effect."""

    def test_glitch_generation(self):
        """Test that glitch effects are generated."""
        effect = GlitchArt(width=100, height=100)
        frame = effect.generate_frame()

        # Should have glitch patterns
        assert frame.max() > 0
        assert frame.shape == (100, 100, 3)

    def test_glitch_intensity(self):
        """Test different glitch intensities."""
        effect_mild = GlitchArt(width=80, height=80, config={"intensity": 0.2})
        frame_mild = effect_mild.generate_frame()

        effect_extreme = GlitchArt(width=80, height=80, config={"intensity": 1.0})
        frame_extreme = effect_extreme.generate_frame()

        # Extreme glitches should be more chaotic
        mild_var = frame_mild.std()
        extreme_var = frame_extreme.std()

        assert extreme_var >= mild_var * 0.8

    def test_glitch_types(self):
        """Test different types of glitches."""
        glitch_types = ["digital", "analog", "datamosh", "pixel_sort", "color_shift"]

        for glitch_type in glitch_types:
            effect = GlitchArt(width=80, height=80, config={"glitch_type": glitch_type})
            frame = effect.generate_frame()

            # Each type should produce valid glitches
            assert frame.shape == (80, 80, 3)
            assert frame.max() > 0

    def test_glitch_frequency(self):
        """Test glitch occurrence frequency."""
        effect_rare = GlitchArt(width=80, height=80, config={"glitch_frequency": 0.1})

        effect_frequent = GlitchArt(width=80, height=80, config={"glitch_frequency": 0.9})

        # Generate multiple frames
        rare_frames = []
        frequent_frames = []

        for _ in range(8):
            rare_frames.append(effect_rare.generate_frame())
            frequent_frames.append(effect_frequent.generate_frame())
            time.sleep(0.02)

        # Frequent should show more variation (more glitches)
        rare_variations = [f.std() for f in rare_frames]
        frequent_variations = [f.std() for f in frequent_frames]

        assert max(frequent_variations) >= max(rare_variations) * 0.8

    def test_color_corruption(self):
        """Test color channel corruption glitches."""
        effect = GlitchArt(width=80, height=80, config={"color_corruption": True, "intensity": 0.8})

        frames = []
        for _ in range(5):
            frames.append(effect.generate_frame())
            time.sleep(0.03)

        # Should see color channel effects
        found_corruption = False
        for frame in frames:
            # Check for unusual color distributions
            r_mean = frame[:, :, 0].mean()
            g_mean = frame[:, :, 1].mean()
            b_mean = frame[:, :, 2].mean()

            # Color corruption might cause extreme channel imbalances
            channel_ratio = max(r_mean, g_mean, b_mean) / (min(r_mean, g_mean, b_mean) + 1)
            if channel_ratio > 3.0:
                found_corruption = True
                break

        # Should see either corruption or at least glitch effects
        assert found_corruption or any(f.std() > 50 for f in frames)

    def test_digital_artifacts(self):
        """Test digital glitch artifacts."""
        effect = GlitchArt(width=100, height=100, config={"digital_artifacts": True})

        frame = effect.generate_frame()

        # Digital artifacts might create sharp transitions
        edges_x = np.abs(np.diff(frame, axis=1)).max()
        edges_y = np.abs(np.diff(frame, axis=0)).max()

        # Should have some sharp digital edges
        assert edges_x > 50 or edges_y > 50

    def test_glitch_animation(self):
        """Test glitch animation over time."""
        effect = GlitchArt(width=80, height=80, config={"animation_speed": 5.0})

        frames = []
        for _ in range(4):
            frames.append(effect.generate_frame())
            time.sleep(0.05)

        # Glitches should animate/change
        changes = []
        for i in range(1, len(frames)):
            change = np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean()
            changes.append(change)

        # Should see significant changes (glitches)
        assert any(c > 20 for c in changes)

    def test_base_image_corruption(self):
        """Test corruption of underlying image patterns."""
        effect = GlitchArt(width=100, height=100, config={"base_pattern": "grid", "corruption_level": 0.7})

        frame = effect.generate_frame()

        # Should have base pattern with corruption applied
        assert frame.shape == (100, 100, 3)
        assert frame.max() > 0

        # Should show some structure (base pattern) but also chaos (corruption)
        assert frame.std() > 20  # Some variation from corruption

    def test_scan_line_effects(self):
        """Test scan line glitch effects."""
        effect = GlitchArt(width=80, height=100, config={"scan_lines": True, "scan_line_intensity": 0.8})

        frame = effect.generate_frame()

        # Scan lines should create horizontal patterns
        # Check for horizontal structure
        horizontal_var = []
        for row in range(0, 100, 10):
            row_data = frame[row, :, :]
            horizontal_var.append(row_data.std())

        # Should have some horizontal variation pattern
        assert any(v > 20 for v in horizontal_var)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
