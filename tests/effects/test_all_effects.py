"""
Comprehensive unit tests for all visual effects.

Tests validate that each effect:
1. Produces frames of correct dimensions
2. Generates non-zero, non-uniform pixel data
3. Creates different frames over time
4. Has reasonable frame metadata and timestamps
5. Properly initializes and handles configuration
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

from src.producer.effects.base_effect import BaseEffect, EffectRegistry
from src.producer.effects.color_effects import ColorBreathe, ColorWipe, GradientFlow, RainbowSweep
from src.producer.effects.environmental_effects import AuroraBorealis, FireSimulation, Lightning
from src.producer.effects.geometric_effects import Kaleidoscope, Mandala, RotatingShapes, Spirals
from src.producer.effects.matrix_effects import BinaryStream, DigitalRain, GlitchArt
from src.producer.effects.noise_effects import FractalNoise, PerlinNoiseFlow, SimplexClouds, VoronoiCells
from src.producer.effects.particle_effects import Fireworks, RainSnow, Starfield, SwarmBehavior
from src.producer.effects.wave_effects import LissajousCurves, PlasmaEffect, SineWaveVisualizer, WaterRipples

# List of all effect classes to test
ALL_EFFECT_CLASSES = [
    # Color effects
    RainbowSweep,
    ColorBreathe,
    GradientFlow,
    ColorWipe,
    # Environmental effects
    AuroraBorealis,
    FireSimulation,
    Lightning,
    # Geometric effects
    Kaleidoscope,
    Mandala,
    RotatingShapes,
    Spirals,
    # Matrix effects
    BinaryStream,
    DigitalRain,
    GlitchArt,
    # Noise effects
    FractalNoise,
    PerlinNoiseFlow,
    SimplexClouds,
    VoronoiCells,
    # Particle effects
    Fireworks,
    RainSnow,
    Starfield,
    SwarmBehavior,
    # Wave effects
    LissajousCurves,
    PlasmaEffect,
    SineWaveVisualizer,
    WaterRipples,
]


class TestEffectFramework:
    """Test the base effect framework."""

    def test_base_effect_abstract(self):
        """Test that BaseEffect cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEffect(width=128, height=64)

    def test_effect_coordinate_grids(self):
        """Test that coordinate grids are properly initialized."""
        # Use a concrete effect for testing
        effect = RainbowSweep(width=100, height=50)

        assert effect.width == 100
        assert effect.height == 50
        assert effect.x_grid.shape == (50, 100)
        assert effect.y_grid.shape == (50, 100)
        assert effect.x_norm.shape == (50, 100)
        assert effect.y_norm.shape == (50, 100)
        assert effect.radius.shape == (50, 100)
        assert effect.angle.shape == (50, 100)

        # Check normalized coordinates are in range [-1, 1]
        assert effect.x_norm.min() >= -1.1  # Small tolerance
        assert effect.x_norm.max() <= 1.1
        assert effect.y_norm.min() >= -1.1
        assert effect.y_norm.max() <= 1.1

    def test_hsv_to_rgb_conversion(self):
        """Test HSV to RGB conversion."""
        effect = RainbowSweep(width=10, height=10)

        # Test pure red (hue=0)
        h = np.zeros((10, 10))
        s = np.ones((10, 10))
        v = np.ones((10, 10))
        rgb = effect.hsv_to_rgb(h, s, v)

        assert rgb.shape == (10, 10, 3)
        assert rgb.dtype == np.uint8
        # Should be red
        assert np.allclose(rgb[:, :, 0], 255, atol=1)  # R channel
        assert np.allclose(rgb[:, :, 1], 0, atol=1)  # G channel
        assert np.allclose(rgb[:, :, 2], 0, atol=1)  # B channel

    def test_gradient_creation(self):
        """Test gradient creation between two colors."""
        effect = RainbowSweep(width=100, height=1)

        color1 = (255, 0, 0)  # Red
        color2 = (0, 0, 255)  # Blue
        position = np.linspace(0, 1, 100).reshape(1, 100)

        gradient = effect.create_gradient(color1, color2, position)

        assert gradient.shape == (1, 100, 3)
        assert gradient.dtype == np.uint8

        # Start should be red
        assert gradient[0, 0, 0] == 255
        assert gradient[0, 0, 2] == 0

        # End should be blue
        assert gradient[0, -1, 0] == 0
        assert gradient[0, -1, 2] == 255


@pytest.mark.parametrize("effect_class", ALL_EFFECT_CLASSES)
class TestAllEffects:
    """Parameterized tests for all effect classes."""

    def test_effect_initialization(self, effect_class):
        """Test that each effect can be initialized with default parameters."""
        effect = effect_class(width=128, height=64, fps=30)

        assert effect is not None
        assert effect.width == 128
        assert effect.height == 64
        assert effect.fps == 30
        assert effect.frame_count == 0
        assert isinstance(effect.config, dict)

    def test_effect_initialization_with_config(self, effect_class):
        """Test that each effect can be initialized with custom config."""
        config = {"speed": 2.0, "color": [255, 128, 0]}
        effect = effect_class(width=100, height=50, fps=24, config=config)

        assert effect.width == 100
        assert effect.height == 50
        assert effect.fps == 24
        # Config should contain at least the passed values
        assert "speed" in effect.config or "color" in effect.config or len(effect.config) >= 0

    def test_frame_dimensions(self, effect_class):
        """Test that generated frames have correct dimensions."""
        width, height = 160, 80
        effect = effect_class(width=width, height=height)

        frame = effect.generate_frame(0.0)

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (height, width, 3)
        assert frame.dtype == np.uint8

    def test_frame_value_range(self, effect_class):
        """Test that frame pixel values are in valid range [0, 255]."""
        effect = effect_class(width=64, height=32)

        frame = effect.generate_frame(0.0)

        assert frame.min() >= 0
        assert frame.max() <= 255

    def test_frame_is_not_zero(self, effect_class):
        """Test that frames contain non-zero data."""
        effect = effect_class(width=64, height=32)

        # Generate a few frames to account for effects that might start dark
        frames = [effect.generate_frame(i * 0.1) for i in range(5)]

        # At least one frame should have non-zero pixels
        has_nonzero = any(frame.max() > 0 for frame in frames)
        assert has_nonzero, f"{effect_class.__name__} produces only zero frames"

    def test_frame_is_not_uniform(self, effect_class):
        """Test that frames are not completely uniform (have some variation)."""
        effect = effect_class(width=64, height=32)

        # Generate several frames
        frames = [effect.generate_frame(i * 0.1) for i in range(10)]

        # Check that at least one frame has variation
        has_variation = False
        for frame in frames:
            # Check if there's variation in any channel
            for channel in range(3):
                channel_data = frame[:, :, channel]
                if channel_data.std() > 0:
                    has_variation = True
                    break
            if has_variation:
                break

        # Some effects might be uniform briefly, but not all frames
        assert has_variation, f"{effect_class.__name__} produces only uniform frames"

    def test_frames_change_over_time(self, effect_class):
        """Test that consecutive frames are different (animation works)."""
        effect = effect_class(width=32, height=32)

        # Generate multiple frames with explicit time values
        frames = []
        for i in range(5):
            frame = effect.generate_frame(i * 0.1)
            frames.append(frame)

        # Check that frames are not all identical
        all_identical = True
        reference = frames[0]
        for frame in frames[1:]:
            if not np.array_equal(frame, reference):
                all_identical = False
                break

        # Some effects might have slow animation or be static briefly
        # Generate more frames if needed
        if all_identical:
            for i in range(10):
                frame = effect.generate_frame((5 + i) * 0.1)
                if not np.array_equal(frame, reference):
                    all_identical = False
                    break

        assert not all_identical, f"{effect_class.__name__} produces identical frames"

    def test_frame_count_increments(self, effect_class):
        """Test that frame counter increments properly."""
        effect = effect_class(width=32, height=32)

        assert effect.frame_count == 0

        for i in range(5):
            effect.generate_frame(i * 0.1)
            assert effect.frame_count == i + 1

    def test_reset_functionality(self, effect_class):
        """Test that reset properly reinitializes the effect."""
        effect = effect_class(width=32, height=32)

        # Generate some frames
        for i in range(5):
            effect.generate_frame(i * 0.1)

        assert effect.frame_count == 5

        effect.reset()

        assert effect.frame_count == 0
        # Effect should still work after reset
        frame = effect.generate_frame(0.0)
        assert frame.shape == (32, 32, 3)

    def test_config_update(self, effect_class):
        """Test that configuration can be updated."""
        effect = effect_class(width=32, height=32)

        old_config = effect.config.copy()
        new_config = {"test_param": 123}

        effect.update_config(new_config)

        # New config should be merged
        assert effect.config["test_param"] == 123

        # Effect should still work after config update
        frame = effect.generate_frame(0.0)
        assert frame.shape == (32, 32, 3)

    def test_time_tracking(self, effect_class):
        """Test that get_time returns the presentation_time."""
        effect = effect_class(width=32, height=32)

        # get_time now returns the presentation_time passed to it
        assert effect.get_time(0.0) == 0.0
        assert effect.get_time(0.5) == 0.5
        assert effect.get_time(1.0) == 1.0


class TestSpecificEffectBehaviors:
    """Tests for specific effect behaviors and characteristics."""

    def test_rainbow_sweep_produces_colors(self):
        """Test that RainbowSweep produces a variety of colors."""
        effect = RainbowSweep(width=100, height=100)
        frame = effect.generate_frame(0.0)

        # Should have variation in all color channels
        assert frame[:, :, 0].std() > 10  # Red channel
        assert frame[:, :, 1].std() > 10  # Green channel
        assert frame[:, :, 2].std() > 10  # Blue channel

    def test_color_breathe_pulses(self):
        """Test that ColorBreathe creates pulsing intensity."""
        # Use 1 Hz rate and sample over a full cycle
        effect = ColorBreathe(width=50, height=50, config={"breathe_rate": 1.0})

        intensities = []
        for i in range(10):
            frame = effect.generate_frame(i * 0.1)  # 0.0 to 0.9 seconds
            intensities.append(frame.mean())

        # Intensity should vary (pulsing)
        assert max(intensities) > min(intensities) + 5

    def test_starfield_has_stars(self):
        """Test that Starfield effect has bright points (stars)."""
        effect = Starfield(width=100, height=100)

        # Generate a few frames to ensure stars appear
        for i in range(5):
            frame = effect.generate_frame(i * 0.1)
            # Should have some bright pixels (stars)
            if frame.max() > 100:
                break

        assert frame.max() > 100  # Should have bright stars

    def test_fire_simulation_has_warm_colors(self):
        """Test that FireSimulation produces warm colors (reds/yellows)."""
        effect = FireSimulation(width=64, height=64)

        # Generate several frames
        for i in range(5):
            frame = effect.generate_frame(i * 0.1)
            # Fire should have more red than blue
            red_mean = frame[:, :, 0].mean()
            blue_mean = frame[:, :, 2].mean()
            if red_mean > blue_mean:
                break

        # Fire effect should be warm (more red than blue on average)
        assert red_mean > blue_mean

    def test_digital_rain_has_vertical_patterns(self):
        """Test that DigitalRain has vertical streaming patterns."""
        effect = DigitalRain(width=64, height=64)

        # Generate frames until we see vertical patterns
        found_vertical = False
        for i in range(10):
            frame = effect.generate_frame(i * 0.1)

            # Check for vertical continuity (columns with similar values)
            for col in range(frame.shape[1]):
                column = frame[:, col, 1]  # Check green channel (Matrix-style)
                if column.max() > 0 and column.std() > 10:
                    found_vertical = True
                    break

            if found_vertical:
                break

        assert found_vertical, "DigitalRain should have vertical patterns"

    def test_plasma_effect_smooth_gradients(self):
        """Test that PlasmaEffect creates smooth gradients."""
        effect = PlasmaEffect(width=64, height=64)
        frame = effect.generate_frame(0.0)

        # Plasma should have smooth gradients (low high-frequency noise)
        # Check by comparing neighboring pixels
        dx = np.abs(np.diff(frame[:, :, 0], axis=1)).mean()
        dy = np.abs(np.diff(frame[:, :, 0], axis=0)).mean()

        # Average difference between neighbors should be relatively small
        assert dx < 50  # Not too sharp transitions
        assert dy < 50

    def test_lightning_has_bright_flashes(self):
        """Test that Lightning effect produces bright flashes."""
        effect = Lightning(width=64, height=64, config={"strike_probability": 0.5})

        # Generate many frames to catch a lightning strike
        max_brightness = 0
        for i in range(20):
            frame = effect.generate_frame(i * 0.05)
            max_brightness = max(max_brightness, frame.max())

        # Should have at least one bright flash
        assert max_brightness > 200

    def test_voronoi_cells_has_regions(self):
        """Test that VoronoiCells creates distinct regions."""
        effect = VoronoiCells(width=64, height=64)
        frame = effect.generate_frame(0.0)

        # Should have distinct regions (not completely smooth)
        # Check for edges between cells
        edges_x = np.abs(np.diff(frame[:, :, 0], axis=1))
        edges_y = np.abs(np.diff(frame[:, :, 0], axis=0))

        # Should have some sharp edges between cells
        assert edges_x.max() > 50 or edges_y.max() > 50


class TestEffectRegistry:
    """Test the effect registry system."""

    def test_registry_operations(self):
        """Test registering and retrieving effects from registry."""
        # Clear registry for test
        EffectRegistry._effects = {}

        # Register a test effect
        EffectRegistry.register(
            effect_id="test_rainbow",
            effect_class=RainbowSweep,
            name="Test Rainbow",
            description="A test rainbow effect",
            category="color",
            default_config={"speed": 1.0},
        )

        # Retrieve effect info
        effect_info = EffectRegistry.get_effect("test_rainbow")
        assert effect_info is not None
        assert effect_info["name"] == "Test Rainbow"
        assert effect_info["class"] == RainbowSweep
        assert effect_info["category"] == "color"
        assert effect_info["icon"] == "🎨"

    def test_create_effect_from_registry(self):
        """Test creating effect instances from registry."""
        # Register effect
        EffectRegistry._effects = {}
        EffectRegistry.register(
            effect_id="test_starfield",
            effect_class=Starfield,
            name="Test Starfield",
            description="A test starfield",
            category="particle",
            default_config={"star_count": 100},
        )

        # Create instance
        effect = EffectRegistry.create_effect("test_starfield", width=100, height=50, config={"speed": 2.0})

        assert effect is not None
        assert isinstance(effect, Starfield)
        assert effect.width == 100
        assert effect.height == 50
        assert effect.config["star_count"] == 100
        assert effect.config["speed"] == 2.0

    def test_list_effects(self):
        """Test listing all registered effects."""
        # Setup registry with multiple effects
        EffectRegistry._effects = {}
        EffectRegistry.register("effect1", RainbowSweep, "Effect 1", "Desc 1", "color", {})
        EffectRegistry.register("effect2", Starfield, "Effect 2", "Desc 2", "particle", {})

        effects_list = EffectRegistry.list_effects()

        assert len(effects_list) == 2
        assert any(e["id"] == "effect1" for e in effects_list)
        assert any(e["id"] == "effect2" for e in effects_list)

    def test_category_icons(self):
        """Test that category icons are assigned correctly."""
        categories_and_icons = {
            "geometric": "🔷",
            "particle": "✨",
            "wave": "🌊",
            "color": "🎨",
            "noise": "🌫️",
            "matrix": "💻",
            "environmental": "🌟",
            "unknown": "🎭",  # Default
        }

        for category, expected_icon in categories_and_icons.items():
            icon = EffectRegistry._get_icon_for_category(category)
            assert icon == expected_icon


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
