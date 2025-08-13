"""
Unit tests for noise-based visual effects.

Tests specific behaviors and characteristics of procedural noise effects.
"""

# Add parent directory to path for tests
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.producer.effects.noise_effects import (
    FractalNoise,
    PerlinNoiseFlow,
    SimplexClouds,
    VoronoiCells,
)


class TestPerlinNoiseFlow:
    """Test PerlinNoiseFlow effect."""

    def test_noise_generation(self):
        """Test that Perlin noise is generated."""
        effect = PerlinNoiseFlow(width=100, height=100)
        frame = effect.generate_frame()

        # Should have smooth noise patterns
        assert frame.max() > 0
        assert frame.shape == (100, 100, 3)

    def test_noise_smoothness(self):
        """Test that Perlin noise is smooth and continuous."""
        effect = PerlinNoiseFlow(width=80, height=80)
        frame = effect.generate_frame()

        # Check smoothness by measuring neighbor differences
        dx = np.abs(np.diff(frame[:, :, 0], axis=1))
        dy = np.abs(np.diff(frame[:, :, 0], axis=0))

        # Perlin noise should be smooth (small differences)
        assert dx.mean() < 100  # Adjust based on implementation
        assert dy.mean() < 100

    def test_noise_scale(self):
        """Test different noise scales."""
        effect_fine = PerlinNoiseFlow(width=100, height=100, config={"scale": 0.1})
        frame_fine = effect_fine.generate_frame()

        effect_coarse = PerlinNoiseFlow(width=100, height=100, config={"scale": 1.0})
        frame_coarse = effect_coarse.generate_frame()

        # Different scales should produce different patterns
        assert frame_fine.shape == frame_coarse.shape == (100, 100, 3)
        assert frame_fine.max() > 0 and frame_coarse.max() > 0

        # Fine scale might have more high-frequency details
        fine_edges = np.abs(np.diff(frame_fine[:, :, 0])).mean()
        coarse_edges = np.abs(np.diff(frame_coarse[:, :, 0])).mean()

        # At least they should be different
        assert abs(fine_edges - coarse_edges) > 0

    def test_noise_flow_animation(self):
        """Test noise flow animation."""
        effect = PerlinNoiseFlow(width=80, height=80, config={"flow_speed": 3.0})

        frames = []
        for _ in range(4):
            frames.append(effect.generate_frame())
            time.sleep(0.05)

        # Noise should flow/animate smoothly
        for i in range(1, len(frames)):
            assert not np.array_equal(frames[i], frames[i - 1])

            # Changes should be smooth
            diff = np.abs(frames[i].astype(float) - frames[i - 1].astype(float)).mean()
            assert 0 < diff < 100

    def test_noise_octaves(self):
        """Test different numbers of noise octaves."""
        effect_simple = PerlinNoiseFlow(width=80, height=80, config={"octaves": 1})
        frame_simple = effect_simple.generate_frame()

        effect_complex = PerlinNoiseFlow(width=80, height=80, config={"octaves": 6})
        frame_complex = effect_complex.generate_frame()

        # More octaves should create more complex patterns
        assert frame_complex.std() >= frame_simple.std() * 0.8

    def test_noise_color_mapping(self):
        """Test different color mapping modes."""
        modes = ["grayscale", "rainbow", "fire", "ocean"]

        for mode in modes:
            effect = PerlinNoiseFlow(width=60, height=60, config={"color_mode": mode})
            frame = effect.generate_frame()

            # Each mode should produce valid output
            assert frame.shape == (60, 60, 3)
            assert frame.max() > 0


class TestSimplexClouds:
    """Test SimplexClouds effect."""

    def test_cloud_generation(self):
        """Test that cloud patterns are generated."""
        effect = SimplexClouds(width=100, height=100)
        frame = effect.generate_frame()

        # Should have cloud-like patterns
        assert frame.max() > 0
        assert frame.shape == (100, 100, 3)

    def test_cloud_density(self):
        """Test different cloud densities."""
        effect_sparse = SimplexClouds(width=100, height=100, config={"density": 0.2})
        frame_sparse = effect_sparse.generate_frame()

        effect_dense = SimplexClouds(width=100, height=100, config={"density": 0.8})
        frame_dense = effect_dense.generate_frame()

        # Dense clouds should have more coverage
        sparse_coverage = (frame_sparse.max(axis=2) > 100).mean()
        dense_coverage = (frame_dense.max(axis=2) > 100).mean()

        assert dense_coverage >= sparse_coverage * 0.8

    def test_cloud_movement(self):
        """Test cloud movement/drift."""
        effect = SimplexClouds(width=100, height=80, config={"drift_speed": 2.0})

        frames = []
        for _ in range(5):
            frames.append(effect.generate_frame())
            time.sleep(0.04)

        # Clouds should drift
        for i in range(1, len(frames)):
            assert not np.array_equal(frames[i], frames[i - 1])

    def test_cloud_softness(self):
        """Test cloud edge softness."""
        effect_soft = SimplexClouds(width=80, height=80, config={"softness": 0.8})
        frame_soft = effect_soft.generate_frame()

        effect_sharp = SimplexClouds(width=80, height=80, config={"softness": 0.1})
        frame_sharp = effect_sharp.generate_frame()

        # Soft clouds should have more gradual edges
        soft_edges = np.abs(np.diff(frame_soft[:, :, 0])).mean()
        sharp_edges = np.abs(np.diff(frame_sharp[:, :, 0])).mean()

        # Both should produce clouds
        assert frame_soft.max() > 0
        assert frame_sharp.max() > 0

    def test_cloud_layers(self):
        """Test multiple cloud layers."""
        effect = SimplexClouds(width=100, height=100, config={"layers": 3})
        frame = effect.generate_frame()

        # Multiple layers should create more complex cloud patterns
        assert frame.shape == (100, 100, 3)
        assert frame.max() > 0

    def test_cloud_colors(self):
        """Test different cloud color schemes."""
        effect_white = SimplexClouds(width=80, height=80, config={"cloud_color": "white"})
        frame_white = effect_white.generate_frame()

        effect_storm = SimplexClouds(width=80, height=80, config={"cloud_color": "storm"})
        frame_storm = effect_storm.generate_frame()

        # Both should produce valid clouds
        assert frame_white.max() > 0
        assert frame_storm.max() > 0


class TestVoronoiCells:
    """Test VoronoiCells effect."""

    def test_cell_generation(self):
        """Test that Voronoi cells are generated."""
        effect = VoronoiCells(width=100, height=100)
        frame = effect.generate_frame()

        # Should have cellular patterns
        assert frame.max() > 0
        assert frame.shape == (100, 100, 3)

    def test_cell_count(self):
        """Test different numbers of cells."""
        effect_few = VoronoiCells(width=100, height=100, config={"cell_count": 10})
        frame_few = effect_few.generate_frame()

        effect_many = VoronoiCells(width=100, height=100, config={"cell_count": 100})
        frame_many = effect_many.generate_frame()

        # Both should produce cellular patterns
        assert frame_few.max() > 0
        assert frame_many.max() > 0

    def test_cell_edges(self):
        """Test that cells have distinct edges."""
        effect = VoronoiCells(width=80, height=80)
        frame = effect.generate_frame()

        # Should have sharp boundaries between cells
        edges_x = np.abs(np.diff(frame[:, :, 0], axis=1))
        edges_y = np.abs(np.diff(frame[:, :, 0], axis=0))

        # Should have some sharp edges
        assert edges_x.max() > 50 or edges_y.max() > 50

    def test_cell_animation(self):
        """Test cell animation/morphing."""
        effect = VoronoiCells(width=80, height=80, config={"morph_speed": 2.0})

        frames = []
        for _ in range(4):
            frames.append(effect.generate_frame())
            time.sleep(0.05)

        # Cells should morph/animate
        for i in range(1, len(frames)):
            assert not np.array_equal(frames[i], frames[i - 1])

    def test_distance_metrics(self):
        """Test different distance metrics for cells."""
        metrics = ["euclidean", "manhattan", "chebyshev"]

        for metric in metrics:
            effect = VoronoiCells(width=60, height=60, config={"distance_metric": metric})
            frame = effect.generate_frame()

            # Each metric should produce valid cells
            assert frame.shape == (60, 60, 3)
            assert frame.max() > 0

    def test_cell_colors(self):
        """Test different cell coloring modes."""
        effect_solid = VoronoiCells(width=80, height=80, config={"color_mode": "solid"})
        frame_solid = effect_solid.generate_frame()

        effect_gradient = VoronoiCells(width=80, height=80, config={"color_mode": "gradient"})
        frame_gradient = effect_gradient.generate_frame()

        # Both should produce colored cells
        assert frame_solid.max() > 0
        assert frame_gradient.max() > 0


class TestFractalNoise:
    """Test FractalNoise effect."""

    def test_fractal_generation(self):
        """Test that fractal noise is generated."""
        effect = FractalNoise(width=100, height=100)
        frame = effect.generate_frame()

        # Should have fractal patterns
        assert frame.max() > 0
        assert frame.shape == (100, 100, 3)

    def test_fractal_depth(self):
        """Test different fractal iteration depths."""
        effect_shallow = FractalNoise(width=80, height=80, config={"iterations": 3})
        frame_shallow = effect_shallow.generate_frame()

        effect_deep = FractalNoise(width=80, height=80, config={"iterations": 10})
        frame_deep = effect_deep.generate_frame()

        # More iterations should create more complex patterns
        assert frame_deep.std() >= frame_shallow.std() * 0.8

    def test_fractal_zoom(self):
        """Test fractal zoom animation."""
        effect = FractalNoise(width=80, height=80, config={"zoom_speed": 2.0})

        frames = []
        for _ in range(4):
            frames.append(effect.generate_frame())
            time.sleep(0.05)

        # Should see zoom animation
        for i in range(1, len(frames)):
            assert not np.array_equal(frames[i], frames[i - 1])

    def test_fractal_complexity(self):
        """Test fractal complexity parameter."""
        effect_simple = FractalNoise(width=100, height=100, config={"complexity": 0.2})
        frame_simple = effect_simple.generate_frame()

        effect_complex = FractalNoise(width=100, height=100, config={"complexity": 1.0})
        frame_complex = effect_complex.generate_frame()

        # More complex should have more variation
        assert frame_complex.std() >= frame_simple.std() * 0.8

    def test_fractal_color_schemes(self):
        """Test different fractal color schemes."""
        schemes = ["mandelbrot", "julia", "burning_ship", "custom"]

        for scheme in schemes:
            effect = FractalNoise(width=60, height=60, config={"color_scheme": scheme})
            frame = effect.generate_frame()

            # Each scheme should produce valid fractals
            assert frame.shape == (60, 60, 3)
            assert frame.max() > 0

    def test_fractal_parameters(self):
        """Test custom fractal parameters."""
        effect = FractalNoise(
            width=80, height=80, config={"center_x": -0.5, "center_y": 0.0, "zoom": 2.0, "escape_radius": 4.0}
        )
        frame = effect.generate_frame()

        # Custom parameters should still produce valid fractals
        assert frame.shape == (80, 80, 3)
        assert frame.max() > 0

    def test_fractal_animation_modes(self):
        """Test different fractal animation modes."""
        modes = ["zoom", "rotation", "parameter_sweep", "color_cycle"]

        for mode in modes:
            effect = FractalNoise(width=60, height=60, config={"animation_mode": mode})

            # Generate a few frames
            frames = []
            for _ in range(3):
                frames.append(effect.generate_frame())
                time.sleep(0.02)

            # Should animate in the chosen mode
            assert not all(np.array_equal(frames[0], frame) for frame in frames[1:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
