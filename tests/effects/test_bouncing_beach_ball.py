"""
Standalone tests for the BouncingBeachBall effect.

This is a standalone test file that doesn't depend on the other effect tests
which may need API updates.
"""

import os
import sys

import numpy as np
import pytest

# Add parent directory to path for tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.producer.effects.geometric_effects import BouncingBeachBall


class TestBouncingBeachBall:
    """Test BouncingBeachBall effect."""

    def test_effect_creation(self):
        """Test that the effect can be created with default config."""
        effect = BouncingBeachBall(width=128, height=64, fps=30)

        # Check that initialization worked
        assert effect.width == 128
        assert effect.height == 64
        assert effect.fps == 30
        assert hasattr(effect, "x")
        assert hasattr(effect, "y")
        assert hasattr(effect, "vx")
        assert hasattr(effect, "vy")
        assert hasattr(effect, "radius")

    def test_config_parameters(self):
        """Test that config parameters are applied correctly."""
        config = {"speed": 2.0, "ball_size": 0.3, "stripe_count": 8, "spin_speed": 3.0}
        effect = BouncingBeachBall(width=128, height=64, config=config)

        assert effect.speed == 2.0
        assert effect.ball_size == 0.3
        assert effect.stripe_count == 8
        assert effect.spin_speed == 3.0

        # Ball radius should be 30% of height
        expected_radius = int(64 * 0.3)
        assert effect.radius == expected_radius

    def test_frame_generation(self):
        """Test that frames are generated with correct shape and content."""
        effect = BouncingBeachBall(width=128, height=64)
        frame = effect.generate_frame(0.0)

        # Check frame properties
        assert frame.shape == (64, 128, 3)
        assert frame.dtype == np.uint8

        # Frame should have some non-zero content (the beach ball)
        assert frame.max() > 0
        assert np.any(frame > 0)

    def test_ball_position_within_bounds(self):
        """Test that ball position stays within screen bounds."""
        effect = BouncingBeachBall(width=128, height=64)

        # Initial position should be within bounds
        assert effect.radius <= effect.x <= effect.width - effect.radius
        assert effect.radius <= effect.y <= effect.height - effect.radius

    def test_ball_movement(self):
        """Test that ball moves between frames."""
        effect = BouncingBeachBall(width=128, height=64, config={"speed": 1.0})

        initial_x = effect.x
        initial_y = effect.y

        # Generate a few frames
        for i in range(5):
            effect.generate_frame(i * (1.0 / 30))  # 30 FPS timing

        # Position should have changed
        assert effect.x != initial_x or effect.y != initial_y

    def test_bouncing_behavior(self):
        """Test that ball bounces off walls correctly."""
        effect = BouncingBeachBall(width=100, height=100, config={"speed": 5.0})

        # Force ball near edge and simulate movement
        effect.x = effect.width - effect.radius - 1
        effect.vx = 100  # Moving right toward wall
        initial_vx = effect.vx

        # Generate frame to trigger bounce
        effect.generate_frame(0.0)

        # Velocity should have reversed
        assert effect.vx == -initial_vx

    def test_spinning_animation(self):
        """Test that ball spins (spin angle changes)."""
        effect = BouncingBeachBall(width=128, height=64, config={"spin_speed": 2.0})

        initial_spin = effect.spin_angle

        # Generate a frame
        effect.generate_frame(0.1)  # 0.1 seconds

        # Spin angle should have changed
        assert effect.spin_angle != initial_spin

    def test_different_stripe_counts(self):
        """Test that different stripe counts work."""
        for stripe_count in [4, 6, 8, 10]:
            effect = BouncingBeachBall(width=128, height=64, config={"stripe_count": stripe_count})
            frame = effect.generate_frame(0.0)

            # Should generate valid frame
            assert frame.shape == (64, 128, 3)
            assert frame.max() > 0

    def test_red_and_blue_colors_present(self):
        """Test that the beach ball contains both red and blue colors."""
        effect = BouncingBeachBall(width=128, height=64)
        frame = effect.generate_frame(0.0)

        # Check for red pixels (high red channel, low green/blue)
        red_pixels = (frame[:, :, 0] > 200) & (frame[:, :, 1] < 50) & (frame[:, :, 2] < 50)

        # Check for blue pixels (high blue channel, low red/green)
        blue_pixels = (frame[:, :, 2] > 200) & (frame[:, :, 0] < 50) & (frame[:, :, 1] < 50)

        # Both colors should be present
        assert np.any(red_pixels), "No red stripes found"
        assert np.any(blue_pixels), "No blue stripes found"

    def test_random_starting_direction(self):
        """Test that different instances start with different directions."""
        # Create multiple effects and check they have different velocities
        effects = [BouncingBeachBall(width=128, height=64) for _ in range(5)]
        velocities = [(e.vx, e.vy) for e in effects]

        # Should have some variation in starting velocities
        unique_velocities = set(velocities)
        assert len(unique_velocities) > 1, "All effects have same starting velocity"
