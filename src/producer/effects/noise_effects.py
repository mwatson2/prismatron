"""Noise-based visual effects optimized for LED display"""

import numpy as np

from .base_effect import BaseEffect, EffectRegistry


class PerlinNoiseFlow(BaseEffect):
    """Organic flowing patterns using Perlin-like noise."""

    def initialize(self):
        self.noise_scale = self.config.get("noise_scale", 3.0)  # Large scale for LED visibility
        self.animation_speed = self.config.get("animation_speed", 0.5)
        self.color_mapping = self.config.get("color_mapping", "rainbow")  # rainbow, fire, ocean
        self.octaves = self.config.get("octaves", 2)  # Fewer octaves for simpler patterns
        self.persistence = self.config.get("persistence", 0.5)

    def _noise_2d(self, x, y):
        """Simple 2D noise function."""
        # Using sine-based pseudo-noise for simplicity
        return (np.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1.0

    def _smooth_noise(self, x, y):
        """Smoothed noise using bilinear interpolation."""
        int_x = int(x)
        int_y = int(y)
        frac_x = x - int_x
        frac_y = y - int_y

        # Get corner values
        a = self._noise_2d(int_x, int_y)
        b = self._noise_2d(int_x + 1, int_y)
        c = self._noise_2d(int_x, int_y + 1)
        d = self._noise_2d(int_x + 1, int_y + 1)

        # Bilinear interpolation
        ab = a * (1 - frac_x) + b * frac_x
        cd = c * (1 - frac_x) + d * frac_x
        return ab * (1 - frac_y) + cd * frac_y

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time) * self.animation_speed

        # Generate multi-octave noise
        noise = np.zeros((self.height, self.width))
        amplitude = 1.0
        frequency = self.noise_scale / min(self.width, self.height)

        for octave in range(self.octaves):
            # Calculate noise for this octave
            x_coords = self.x_grid * frequency + t
            y_coords = self.y_grid * frequency

            # Simple noise calculation
            octave_noise = np.sin(x_coords) * np.cos(y_coords) + np.sin(x_coords * 1.7 + t) * np.cos(y_coords * 1.3)
            octave_noise = (octave_noise + 2) / 4  # Normalize to 0-1

            noise += octave_noise * amplitude
            amplitude *= self.persistence
            frequency *= 2

        # Normalize noise to 0-1
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

        # Apply color mapping
        if self.color_mapping == "rainbow":
            hue = noise
            saturation = np.ones_like(noise)
            value = np.ones_like(noise)
        elif self.color_mapping == "fire":
            # Fire colors: black -> red -> orange -> yellow
            hue = noise * 0.15  # Red to yellow range
            saturation = np.ones_like(noise)
            value = noise
        else:  # ocean
            # Ocean colors: dark blue -> light blue -> cyan
            hue = 0.5 + noise * 0.2
            saturation = 1.0 - noise * 0.3
            value = 0.3 + noise * 0.7

        frame = self.hsv_to_rgb(hue, saturation, value)
        self.frame_count += 1
        return frame


class SimplexClouds(BaseEffect):
    """Cloud-like formations using simplified noise."""

    def initialize(self):
        self.scale = self.config.get("scale", 5.0)
        self.movement_speed = self.config.get("movement_speed", 0.3)
        self.threshold = self.config.get("threshold", 0.5)  # Cloud density
        self.color_mode = self.config.get("color_mode", "white")  # white, sunset, storm
        self.layers = self.config.get("layers", 2)  # Cloud layers

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time) * self.movement_speed
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Generate cloud layers
        for layer in range(self.layers):
            # Different speed and scale for each layer
            layer_speed = t * (1 + layer * 0.3)
            layer_scale = self.scale * (1 + layer * 0.5)

            # Simple cloud pattern using sine waves
            pattern = (
                np.sin(self.x_grid / layer_scale + layer_speed) * np.cos(self.y_grid / layer_scale * 0.7)
                + np.sin((self.x_grid + self.y_grid) / (layer_scale * 1.5)) * 0.5
            )

            # Normalize and threshold
            pattern = (pattern + 1.5) / 3
            cloud_mask = pattern > self.threshold

            # Apply color based on mode
            if self.color_mode == "white":
                color = np.array([255, 255, 255]) * (0.8 - layer * 0.2)
            elif self.color_mode == "sunset":
                # Orange to pink gradient
                hue = 0.05 + layer * 0.03
                color_hsv = np.array([[[hue * 180, 200, 255]]], dtype=np.uint8)
                color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0] * (0.9 - layer * 0.1)
            else:  # storm
                # Dark gray to purple
                color = np.array([100 - layer * 20, 80 - layer * 20, 120 - layer * 10])

            # Add cloud layer to frame
            frame[cloud_mask] = np.maximum(frame[cloud_mask], color)

        self.frame_count += 1
        return frame


class VoronoiCells(BaseEffect):
    """Animated Voronoi diagram with moving seed points."""

    def initialize(self):
        self.cell_count = self.config.get("cell_count", 10)  # Fewer cells for LED display
        self.movement_pattern = self.config.get("movement_pattern", "circular")  # circular, random, flow
        self.color_mode = self.config.get("color_mode", "distinct")  # distinct, gradient, monochrome
        self.movement_speed = self.config.get("movement_speed", 0.3)
        self.show_seeds = self.config.get("show_seeds", False)

        # Initialize seed points
        self.seeds = []
        for i in range(self.cell_count):
            self.seeds.append(
                {
                    "x": np.random.uniform(0, 1),
                    "y": np.random.uniform(0, 1),
                    "base_x": np.random.uniform(0, 1),
                    "base_y": np.random.uniform(0, 1),
                    "phase": np.random.random() * 2 * np.pi,
                    "color": self._get_cell_color(i),
                }
            )

    def _get_cell_color(self, index):
        """Get color for a Voronoi cell."""
        if self.color_mode == "distinct":
            hue = index / self.cell_count
            color_hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
            return cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()
        elif self.color_mode == "gradient":
            return [255 * (index % 2), 128, 255 * ((index + 1) % 2)]
        else:  # monochrome
            gray = 50 + (index / self.cell_count) * 200
            return [gray, gray, gray]

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time) * self.movement_speed

        # Update seed positions
        for seed in self.seeds:
            if self.movement_pattern == "circular":
                # Circular motion around base position
                radius = 0.1
                seed["x"] = seed["base_x"] + radius * np.cos(t + seed["phase"])
                seed["y"] = seed["base_y"] + radius * np.sin(t + seed["phase"])
            elif self.movement_pattern == "random":
                # Random walk
                seed["x"] += np.random.uniform(-0.01, 0.01)
                seed["y"] += np.random.uniform(-0.01, 0.01)
                seed["x"] = np.clip(seed["x"], 0, 1)
                seed["y"] = np.clip(seed["y"], 0, 1)
            else:  # flow
                # Flowing movement
                seed["x"] = seed["base_x"] + 0.1 * np.sin(t + seed["phase"])
                seed["y"] = seed["base_y"] + 0.1 * np.cos(t * 0.7 + seed["phase"])

            # Keep seeds in bounds
            seed["x"] = seed["x"] % 1.0
            seed["y"] = seed["y"] % 1.0

        # Create Voronoi diagram
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # For each pixel, find nearest seed
        x_norm = self.x_grid / self.width
        y_norm = self.y_grid / self.height

        for y in range(0, self.height, 2):  # Skip pixels for performance
            for x in range(0, self.width, 2):
                px = x / self.width
                py = y / self.height

                # Find nearest seed
                min_dist = float("inf")
                nearest_seed = None

                for seed in self.seeds:
                    dist = (px - seed["x"]) ** 2 + (py - seed["y"]) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        nearest_seed = seed

                # Color the cell (2x2 block for performance)
                if nearest_seed:
                    frame[y : y + 2, x : x + 2] = nearest_seed["color"]

        # Optionally show seed points
        if self.show_seeds:
            for seed in self.seeds:
                px = int(seed["x"] * self.width)
                py = int(seed["y"] * self.height)
                cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)

        self.frame_count += 1
        return frame


class FractalNoise(BaseEffect):
    """Multi-octave fractal noise creating organic patterns."""

    def initialize(self):
        self.octaves = self.config.get("octaves", 3)
        self.lacunarity = self.config.get("lacunarity", 2.0)  # Frequency multiplier
        self.gain = self.config.get("gain", 0.5)  # Amplitude multiplier
        self.animation_speed = self.config.get("animation_speed", 0.3)
        self.color_scheme = self.config.get("color_scheme", "terrain")  # terrain, lava, ice
        self.threshold_mode = self.config.get("threshold_mode", False)  # Binary threshold

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time) * self.animation_speed

        # Generate fractal noise (simplified FBM)
        noise = np.zeros((self.height, self.width))
        frequency = 1.0 / min(self.width, self.height) * 5
        amplitude = 1.0
        max_amplitude = 0

        for octave in range(self.octaves):
            # Generate noise for this octave
            x_freq = self.x_grid * frequency
            y_freq = self.y_grid * frequency

            # Simple noise pattern
            octave_noise = (
                np.sin(x_freq + t) * np.cos(y_freq)
                + np.sin(x_freq * 1.3) * np.cos(y_freq * 1.7 + t)
                + np.sin((x_freq + y_freq) * 0.7 - t * 0.5)
            )

            noise += octave_noise * amplitude
            max_amplitude += amplitude

            frequency *= self.lacunarity
            amplitude *= self.gain

        # Normalize
        noise = (noise / max_amplitude + 1) / 2

        # Apply threshold if enabled
        if self.threshold_mode:
            noise = np.where(noise > 0.5, 1.0, 0.0)

        # Apply color scheme
        if self.color_scheme == "terrain":
            # Green to brown gradient based on height
            hue = 0.25 - noise * 0.15  # Green to brown
            saturation = 0.7 + noise * 0.3
            value = 0.4 + noise * 0.6
        elif self.color_scheme == "lava":
            # Black to red to yellow
            hue = noise * 0.15  # Red to yellow
            saturation = np.ones_like(noise)
            value = noise
        else:  # ice
            # White to light blue to dark blue
            hue = 0.55 + noise * 0.05  # Blue range
            saturation = noise * 0.5
            value = 1.0 - noise * 0.3

        frame = self.hsv_to_rgb(hue, saturation, value)
        self.frame_count += 1
        return frame


# Need cv2 for some effects
import cv2

# Register effects
EffectRegistry.register(
    "perlin_flow",
    PerlinNoiseFlow,
    "Perlin Flow",
    "Organic flowing noise patterns",
    "noise",
    {"noise_scale": 3.0, "animation_speed": 0.5, "color_mapping": "rainbow", "octaves": 2, "persistence": 0.5},
)

EffectRegistry.register(
    "simplex_clouds",
    SimplexClouds,
    "Cloud Formation",
    "Cloud-like patterns",
    "noise",
    {"scale": 5.0, "movement_speed": 0.3, "threshold": 0.5, "color_mode": "white", "layers": 2},
)

EffectRegistry.register(
    "voronoi_cells",
    VoronoiCells,
    "Voronoi Cells",
    "Animated cellular patterns",
    "noise",
    {
        "cell_count": 10,
        "movement_pattern": "circular",
        "color_mode": "distinct",
        "movement_speed": 0.3,
        "show_seeds": False,
    },
)

EffectRegistry.register(
    "fractal_noise",
    FractalNoise,
    "Fractal Noise",
    "Multi-octave fractal patterns",
    "noise",
    {
        "octaves": 3,
        "lacunarity": 2.0,
        "gain": 0.5,
        "animation_speed": 0.3,
        "color_scheme": "terrain",
        "threshold_mode": False,
    },
)
