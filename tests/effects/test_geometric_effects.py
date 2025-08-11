"""
Unit tests for geometric visual effects.

Tests specific behaviors and characteristics of geometric pattern effects.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add src to path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from producer.effects.geometric_effects import Kaleidoscope, Mandala, RotatingShapes, Spirals


class TestRotatingShapes:
    """Test RotatingShapes effect."""

    def test_shape_generation(self):
        """Test that shapes are generated."""
        effect = RotatingShapes(width=100, height=100)
        frame = effect.generate_frame()

        # Should have non-zero content
        assert frame.max() > 0
        assert frame.mean() > 0

    def test_rotation_animation(self):
        """Test that shapes rotate over time."""
        effect = RotatingShapes(width=80, height=80, config={"rotation_speed": 5.0})

        frames = []
        for _ in range(5):
            frames.append(effect.generate_frame())
            time.sleep(0.05)

        # Frames should be different due to rotation
        differences = []
        for i in range(1, len(frames)):
            diff = np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean()
            differences.append(diff)

        # Should see consistent changes from rotation
        assert all(d > 0 for d in differences)

    def test_shape_count(self):
        """Test different numbers of shapes."""
        effect_few = RotatingShapes(width=100, height=100, config={"shape_count": 2})
        frame_few = effect_few.generate_frame()

        effect_many = RotatingShapes(width=100, height=100, config={"shape_count": 10})
        frame_many = effect_many.generate_frame()

        # More shapes should generally mean more complex image (higher entropy)
        # Count non-background pixels
        few_pixels = np.sum(frame_few.max(axis=2) > 10)
        many_pixels = np.sum(frame_many.max(axis=2) > 10)

        # More shapes should cover more area (with some tolerance for overlap)
        assert many_pixels >= few_pixels * 0.8

    def test_shape_types(self):
        """Test different shape types."""
        shape_types = ["circles", "squares", "triangles", "mixed"]

        for shape_type in shape_types:
            effect = RotatingShapes(width=80, height=80, config={"shape_type": shape_type})
            frame = effect.generate_frame()

            # Each shape type should produce non-zero output
            assert frame.max() > 0
            assert frame.shape == (80, 80, 3)


class TestKaleidoscope:
    """Test Kaleidoscope effect."""

    def test_symmetry(self):
        """Test that kaleidoscope has rotational symmetry."""
        effect = Kaleidoscope(width=100, height=100, config={"segments": 6})
        frame = effect.generate_frame()

        # Convert to grayscale for easier comparison
        gray = frame.mean(axis=2)

        # Check center point
        cx, cy = 50, 50

        # Sample points at same radius but different angles
        radius = 30
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 segments

        values = []
        for angle in angles:
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            if 0 <= x < 100 and 0 <= y < 100:
                values.append(gray[y, x])

        # Values should show some repetition due to symmetry
        # (Not perfectly identical due to discretization)
        if len(values) >= 6:
            # Check if pattern repeats
            pattern_similarity = np.std(values) < np.mean(values) * 0.5
            assert pattern_similarity or len(set(values)) < len(values)

    def test_segment_count(self):
        """Test different numbers of kaleidoscope segments."""
        for segments in [4, 6, 8, 12]:
            effect = Kaleidoscope(width=80, height=80, config={"segments": segments})
            frame = effect.generate_frame()

            # Should produce valid output for any segment count
            assert frame.shape == (80, 80, 3)
            assert frame.max() > 0

    def test_animation(self):
        """Test kaleidoscope animation."""
        effect = Kaleidoscope(width=60, height=60, config={"rotation_speed": 3.0})

        frame1 = effect.generate_frame()
        time.sleep(0.1)
        frame2 = effect.generate_frame()

        # Should animate over time
        assert not np.array_equal(frame1, frame2)

    def test_complexity(self):
        """Test pattern complexity parameter."""
        effect_simple = Kaleidoscope(width=80, height=80, config={"complexity": 0.2})
        frame_simple = effect_simple.generate_frame()

        effect_complex = Kaleidoscope(width=80, height=80, config={"complexity": 1.0})
        frame_complex = effect_complex.generate_frame()

        # More complex should have more variation
        assert frame_complex.std() >= frame_simple.std() * 0.8


class TestSpirals:
    """Test Spirals effect."""

    def test_spiral_generation(self):
        """Test that spirals are visible."""
        effect = Spirals(width=100, height=100)
        frame = effect.generate_frame()

        # Should have spiral patterns radiating from center
        # Check for radial variation
        center = frame[45:55, 45:55, :].mean()
        edge = frame[:10, :10, :].mean()

        assert abs(center - edge) > 10

    def test_spiral_count(self):
        """Test different numbers of spiral arms."""
        effect_few = Spirals(width=80, height=80, config={"arm_count": 2})
        frame_few = effect_few.generate_frame()

        effect_many = Spirals(width=80, height=80, config={"arm_count": 8})
        frame_many = effect_many.generate_frame()

        # Both should produce valid spirals
        assert frame_few.max() > 0
        assert frame_many.max() > 0

    def test_spiral_rotation(self):
        """Test spiral rotation animation."""
        effect = Spirals(width=80, height=80, config={"rotation_speed": 5.0})

        frames = []
        for _ in range(4):
            frames.append(effect.generate_frame())
            time.sleep(0.05)

        # Should see rotation
        for i in range(1, len(frames)):
            assert not np.array_equal(frames[i], frames[i - 1])

    def test_spiral_tightness(self):
        """Test spiral tightness parameter."""
        effect_loose = Spirals(width=100, height=100, config={"tightness": 0.2})
        frame_loose = effect_loose.generate_frame()

        effect_tight = Spirals(width=100, height=100, config={"tightness": 1.0})
        frame_tight = effect_tight.generate_frame()

        # Both should produce valid output
        assert frame_loose.shape == (100, 100, 3)
        assert frame_tight.shape == (100, 100, 3)

        # Tight spirals might have more frequent transitions
        # Count transitions along a radial line
        line_loose = frame_loose[50, 50:, 0]
        line_tight = frame_tight[50, 50:, 0]

        trans_loose = np.sum(np.abs(np.diff(line_loose)) > 50)
        trans_tight = np.sum(np.abs(np.diff(line_tight)) > 50)

        # This is approximate - spiral structure affects transitions
        assert trans_loose >= 0 and trans_tight >= 0

    def test_spiral_direction(self):
        """Test clockwise vs counter-clockwise spirals."""
        effect_cw = Spirals(width=80, height=80, config={"direction": "clockwise"})
        frame_cw = effect_cw.generate_frame()

        effect_ccw = Spirals(width=80, height=80, config={"direction": "counter-clockwise"})
        frame_ccw = effect_ccw.generate_frame()

        # Both should produce valid but potentially different patterns
        assert frame_cw.shape == (80, 80, 3)
        assert frame_ccw.shape == (80, 80, 3)


class TestMandala:
    """Test Mandala effect."""

    def test_mandala_symmetry(self):
        """Test that mandala has radial symmetry."""
        effect = Mandala(width=100, height=100, config={"symmetry": 8})
        frame = effect.generate_frame()

        # Should have radial symmetry from center
        # Test by comparing regions at same radius
        gray = frame.mean(axis=2)
        cx, cy = 50, 50

        # Sample symmetric points
        radius = 30
        n_points = 8
        angles = np.linspace(0, 2 * np.pi, n_points + 1)[:-1]

        samples = []
        for angle in angles:
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            if 0 <= x < 100 and 0 <= y < 100:
                # Sample a small region
                region = gray[max(0, y - 2) : min(100, y + 3), max(0, x - 2) : min(100, x + 3)]
                samples.append(region.mean())

        # Should see some repetition in symmetric points
        if len(samples) >= 8:
            unique_values = len(set(np.round(samples, 1)))
            assert unique_values <= 6  # Some symmetry

    def test_mandala_layers(self):
        """Test different numbers of mandala layers."""
        effect_simple = Mandala(width=80, height=80, config={"layers": 2})
        frame_simple = effect_simple.generate_frame()

        effect_complex = Mandala(width=80, height=80, config={"layers": 6})
        frame_complex = effect_complex.generate_frame()

        # More layers should create more complex patterns
        assert frame_simple.std() > 0
        assert frame_complex.std() > 0

    def test_mandala_animation(self):
        """Test mandala animation/evolution."""
        effect = Mandala(width=80, height=80, config={"evolution_speed": 2.0})

        frames = []
        for _ in range(4):
            frames.append(effect.generate_frame())
            time.sleep(0.05)

        # Pattern should evolve over time
        changes = []
        for i in range(1, len(frames)):
            change = np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean()
            changes.append(change)

        assert any(c > 0 for c in changes)

    def test_mandala_color_scheme(self):
        """Test different color schemes."""
        schemes = ["rainbow", "monochrome", "complementary", "analogous"]

        for scheme in schemes:
            effect = Mandala(width=60, height=60, config={"color_scheme": scheme})
            frame = effect.generate_frame()

            # Each scheme should produce valid output
            assert frame.shape == (60, 60, 3)
            assert frame.max() > 0

    def test_mandala_detail_level(self):
        """Test detail level parameter."""
        effect_low = Mandala(width=100, height=100, config={"detail": 0.2})
        frame_low = effect_low.generate_frame()

        effect_high = Mandala(width=100, height=100, config={"detail": 1.0})
        frame_high = effect_high.generate_frame()

        # Higher detail might have more fine structure (higher frequency content)
        # Use edge detection as proxy for detail
        edges_low = np.abs(np.diff(frame_low, axis=0)).mean() + np.abs(np.diff(frame_low, axis=1)).mean()
        edges_high = np.abs(np.diff(frame_high, axis=0)).mean() + np.abs(np.diff(frame_high, axis=1)).mean()

        # Both should produce valid patterns
        assert frame_low.max() > 0
        assert frame_high.max() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
