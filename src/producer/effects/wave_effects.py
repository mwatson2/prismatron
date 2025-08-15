"""Wave-based visual effects optimized for LED display"""

import numpy as np

from .base_effect import BaseEffect, EffectRegistry


class SineWaveVisualizer(BaseEffect):
    """Multiple overlapping sine waves creating interference patterns."""

    def initialize(self):
        self.wave_count = self.config.get("wave_count", 3)
        self.frequencies = self.config.get("frequencies", [1.0, 1.5, 2.0])
        self.amplitudes = self.config.get("amplitudes", [0.3, 0.3, 0.3])
        self.phase_speeds = self.config.get("phase_speeds", [0.5, -0.3, 0.7])
        self.direction = self.config.get("direction", "horizontal")  # horizontal, vertical, radial
        self.color_mode = self.config.get("color_mode", "height")  # height, phase, rainbow
        self.thickness = self.config.get("thickness", 0.1)  # Wave thickness

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Calculate wave position based on direction
        if self.direction == "horizontal":
            position = self.x_norm
        elif self.direction == "vertical":
            position = self.y_norm
        else:  # radial
            position = self.radius

        # Sum multiple waves
        wave_sum = np.zeros_like(position)
        for i in range(min(self.wave_count, len(self.frequencies))):
            freq = self.frequencies[i] if i < len(self.frequencies) else 1.0
            amp = self.amplitudes[i] if i < len(self.amplitudes) else 0.3
            phase_speed = self.phase_speeds[i] if i < len(self.phase_speeds) else 0.5

            phase = 2 * np.pi * (freq * position + phase_speed * t)
            wave_sum += amp * np.sin(phase)

        # Normalize wave sum to -1 to 1
        wave_sum = np.clip(wave_sum / np.sum(self.amplitudes[: self.wave_count]), -1, 1)

        # Create visualization based on wave height
        if self.direction == "horizontal":
            # Horizontal waves going up and down
            wave_y = self.y_norm - wave_sum
            mask = np.abs(wave_y) < self.thickness
        elif self.direction == "vertical":
            # Vertical waves going left and right
            wave_x = self.x_norm - wave_sum
            mask = np.abs(wave_x) < self.thickness
        else:  # radial
            # Radial waves
            expected_radius = (wave_sum + 1) / 2  # Map to 0-1
            mask = np.abs(self.radius - expected_radius) < self.thickness

        # Apply color based on mode
        if self.color_mode == "height":
            # Color based on wave height
            hue = (wave_sum + 1) / 2  # Map -1 to 1 -> 0 to 1
            hue = (hue + t * 0.1) % 1.0
        elif self.color_mode == "phase":
            # Color based on phase
            hue = (position + t * 0.2) % 1.0
        else:  # rainbow
            # Rainbow gradient
            hue = (position * 0.5 + t * 0.3) % 1.0

        # Convert to RGB
        hue_array = hue
        saturation = np.ones_like(hue)
        value = np.ones_like(hue)
        color_frame = self.hsv_to_rgb(hue_array, saturation, value)

        # Apply mask
        frame[mask] = color_frame[mask]

        self.frame_count += 1
        return frame


class PlasmaEffect(BaseEffect):
    """Classic plasma effect using sine functions - perfect for LEDs."""

    def initialize(self):
        self.color_palette = self.config.get("color_palette", "rainbow")  # rainbow, fire, ocean, neon
        self.frequency = self.config.get("frequency", 2.0)
        self.animation_speed = self.config.get("animation_speed", 0.5)
        self.complexity = self.config.get("complexity", 3)  # Number of sine components

        # Define color palettes
        self.palettes = {
            "rainbow": lambda h: self.hsv_to_rgb(h, np.ones_like(h), np.ones_like(h)),
            "fire": lambda h: self.hsv_to_rgb(h * 0.15, np.ones_like(h), h),  # Red to yellow
            "ocean": lambda h: self.hsv_to_rgb(0.5 + h * 0.2, np.ones_like(h), 0.5 + h * 0.5),  # Blues
            "neon": lambda h: self.hsv_to_rgb(h, np.ones_like(h), np.where(h > 0.5, 1.0, 0.0)),  # High contrast
        }

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time) * self.animation_speed

        # Classic plasma algorithm with multiple sine components
        plasma = np.zeros((self.height, self.width))

        # Component 1: Diagonal sine
        plasma += np.sin((self.x_norm + self.y_norm) * self.frequency + t)

        # Component 2: Horizontal sine
        plasma += np.sin(self.x_norm * self.frequency * 2 + t * 1.5)

        if self.complexity >= 3:
            # Component 3: Circular sine
            plasma += np.sin(self.radius * self.frequency * 3 + t * 0.7)

        if self.complexity >= 4:
            # Component 4: Another diagonal
            plasma += np.sin((self.x_norm - self.y_norm) * self.frequency * 1.5 - t)

        # Normalize to 0-1
        plasma = (plasma + self.complexity) / (2 * self.complexity)

        # Apply color palette
        palette_func = self.palettes.get(self.color_palette, self.palettes["rainbow"])
        frame = palette_func(plasma)

        self.frame_count += 1
        return frame


class WaterRipples(BaseEffect):
    """Concentric ripples like water drops."""

    def initialize(self):
        self.ripple_speed = self.config.get("ripple_speed", 0.5)
        self.ripple_frequency = self.config.get("ripple_frequency", 1.0)  # New ripples per second
        self.max_ripples = self.config.get("max_ripples", 3)
        self.damping = self.config.get("damping", 0.1)  # How quickly ripples fade
        self.interference = self.config.get("interference", True)  # Allow wave interference
        self.color_mode = self.config.get("color_mode", "blue")  # blue, rainbow, monochrome

        self.ripples = []  # List of active ripples (x, y, start_time)
        self.last_ripple_time = -10  # Start with negative time to trigger immediate ripple

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Add new ripple if it's time
        if t - self.last_ripple_time > 1.0 / self.ripple_frequency and len(self.ripples) < self.max_ripples:
            # Random position for new ripple
            x = np.random.uniform(-0.8, 0.8)
            y = np.random.uniform(-0.8, 0.8)
            self.ripples.append((x, y, t))
            self.last_ripple_time = t

        # Remove old ripples
        self.ripples = [
            (x, y, st) for x, y, st in self.ripples if t - st < 3.0 / self.ripple_speed
        ]  # Ripples last 3 seconds

        # Calculate wave height at each pixel
        wave_height = np.zeros((self.height, self.width))

        for ripple_x, ripple_y, start_time in self.ripples:
            # Calculate distance from ripple center
            dist_x = self.x_norm - ripple_x
            dist_y = self.y_norm - ripple_y
            distance = np.sqrt(dist_x**2 + dist_y**2)

            # Calculate ripple wave
            age = t - start_time
            wave_position = age * self.ripple_speed

            # Create ring wave
            wave = np.sin(2 * np.pi * (distance - wave_position) * 3)  # 3 waves

            # Apply damping
            amplitude = np.exp(-age * self.damping) * np.exp(-distance * 0.3)

            # Only show wave near the wave front (creates rings instead of filled circles)
            wave_mask = np.abs(distance - wave_position) < 0.3
            wave = wave * wave_mask * amplitude

            if self.interference:
                wave_height += wave
            else:
                wave_height = np.maximum(wave_height, wave)

        # Normalize and convert to color
        wave_height = np.clip((wave_height + 1) / 2, 0, 1)

        if self.color_mode == "blue":
            # Blue water colors
            hue = 0.55 + wave_height * 0.1  # Blue to cyan
            saturation = 0.8 + wave_height * 0.2
            value = 0.3 + wave_height * 0.7
        elif self.color_mode == "rainbow":
            # Rainbow based on height
            hue = wave_height
            saturation = np.ones_like(hue)
            value = 0.5 + wave_height * 0.5
        else:  # monochrome
            # Simple grayscale
            frame[:, :] = (wave_height * 255).astype(np.uint8)[:, :, np.newaxis]
            self.frame_count += 1
            return frame

        frame = self.hsv_to_rgb(hue, saturation, value)

        self.frame_count += 1
        return frame


class LissajousCurves(BaseEffect):
    """Animated Lissajous curves creating beautiful patterns."""

    def initialize(self):
        self.x_frequency = self.config.get("x_frequency", 3)
        self.y_frequency = self.config.get("y_frequency", 2)
        self.phase_shift = self.config.get("phase_shift", 0.5)  # Initial phase difference
        self.animation_speed = self.config.get("animation_speed", 0.2)
        self.trail_length = self.config.get("trail_length", 0.5)  # How long trails persist
        self.line_thickness = self.config.get("line_thickness", 0.02)
        self.color_cycle = self.config.get("color_cycle", True)

        # Trail buffer for persistence effect
        self.trail_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time)

        # Fade trail buffer
        self.trail_buffer *= 1 - self.trail_length * 0.05

        # Animate phase shift for morphing patterns
        phase = self.phase_shift + t * self.animation_speed

        # Generate Lissajous curve points
        num_points = 500  # Reduced for LED display
        curve_t = np.linspace(0, 2 * np.pi, num_points)

        # Calculate curve positions
        x = np.sin(self.x_frequency * curve_t + phase)
        y = np.sin(self.y_frequency * curve_t)

        # Convert to pixel coordinates
        x_pixels = ((x + 1) / 2 * self.width).astype(int)
        y_pixels = ((y + 1) / 2 * self.height).astype(int)

        # Draw thick curve
        for i in range(len(x_pixels) - 1):
            if self.color_cycle:
                # Color changes along curve
                hue = (i / num_points + t * 0.2) % 1.0
            else:
                # Single color
                hue = 0.6  # Blue

            color_hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
            color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]

            # Draw line segment with thickness
            x1, y1 = x_pixels[i], y_pixels[i]
            x2, y2 = x_pixels[i + 1], y_pixels[i + 1]

            # Simple line drawing (could use cv2.line but doing it manually for control)
            # Create a mask for the line segment
            xx, yy = np.meshgrid(range(self.width), range(self.height))

            # Distance from point to line segment
            line_vec = np.array([x2 - x1, y2 - y1])
            line_len = np.linalg.norm(line_vec)

            if line_len > 0:
                line_vec = line_vec / line_len
                point_vec = np.stack([xx - x1, yy - y1], axis=-1)
                proj_len = np.clip(np.sum(point_vec * line_vec, axis=-1), 0, line_len)
                proj_point = np.stack([x1, y1]) + proj_len[:, :, np.newaxis] * line_vec

                dist = np.linalg.norm(np.stack([xx, yy], axis=-1) - proj_point, axis=-1)
                mask = dist < self.line_thickness * min(self.width, self.height)

                # Add to trail buffer
                self.trail_buffer[mask] = color_rgb

        # Convert trail buffer to output frame
        frame = np.clip(self.trail_buffer, 0, 255).astype(np.uint8)

        self.frame_count += 1
        return frame


# Note: cv2 import needed for LissajousCurves
import cv2

# Register effects
EffectRegistry.register(
    "sine_waves",
    SineWaveVisualizer,
    "Sine Waves",
    "Multiple overlapping sine waves",
    "wave",
    {
        "wave_count": 3,
        "frequencies": [1.0, 1.5, 2.0],
        "amplitudes": [0.3, 0.3, 0.3],
        "phase_speeds": [0.5, -0.3, 0.7],
        "direction": "horizontal",
        "color_mode": "height",
        "thickness": 0.1,
    },
)

EffectRegistry.register(
    "plasma",
    PlasmaEffect,
    "Plasma",
    "Classic plasma effect",
    "wave",
    {"color_palette": "rainbow", "frequency": 2.0, "animation_speed": 0.5, "complexity": 3},
)

EffectRegistry.register(
    "water_ripples",
    WaterRipples,
    "Water Ripples",
    "Concentric ripples with interference",
    "wave",
    {
        "ripple_speed": 0.5,
        "ripple_frequency": 1.0,
        "max_ripples": 3,
        "damping": 0.1,
        "interference": True,
        "color_mode": "blue",
    },
)

EffectRegistry.register(
    "lissajous",
    LissajousCurves,
    "Lissajous Curves",
    "Animated parametric curves",
    "wave",
    {
        "x_frequency": 3,
        "y_frequency": 2,
        "phase_shift": 0.5,
        "animation_speed": 0.2,
        "trail_length": 0.5,
        "line_thickness": 0.02,
        "color_cycle": True,
    },
)
