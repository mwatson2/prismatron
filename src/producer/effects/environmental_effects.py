"""Environmental simulation effects optimized for LED display"""

import cv2
import numpy as np

from .base_effect import BaseEffect, EffectRegistry


class FireSimulation(BaseEffect):
    """Simple fire effect with large flames visible on LEDs."""

    def initialize(self):
        self.intensity = self.config.get("intensity", 0.8)
        self.wind_strength = self.config.get("wind_strength", 0.1)
        self.fuel_rate = self.config.get("fuel_rate", 0.7)
        self.cooling_rate = self.config.get("cooling_rate", 0.05)
        self.spark_probability = self.config.get("spark_probability", 0.02)

        # Initialize temperature field
        self.temperature = np.zeros((self.height, self.width))

        # Add heat sources at bottom
        self.temperature[-3:, :] = self.fuel_rate

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        # Update temperature field
        new_temp = self.temperature.copy()

        # Convection - heat rises
        for y in range(self.height - 1, 0, -1):
            for x in range(self.width):
                # Heat rises with some randomness
                rise_amount = self.temperature[y, x] * 0.8
                new_temp[y - 1, x] += rise_amount * (0.8 + np.random.random() * 0.4)
                new_temp[y, x] -= rise_amount

        # Add wind effect
        if self.wind_strength > 0:
            wind_offset = int(self.wind_strength * 10 * (np.random.random() - 0.5))
            if wind_offset != 0:
                if wind_offset > 0:
                    new_temp[:, wind_offset:] = new_temp[:, :-wind_offset]
                    new_temp[:, :wind_offset] = 0
                else:
                    new_temp[:, :wind_offset] = new_temp[:, -wind_offset:]
                    new_temp[:, wind_offset:] = 0

        # Cooling
        new_temp *= 1 - self.cooling_rate

        # Add random sparks
        spark_mask = np.random.random((self.height, self.width)) < self.spark_probability
        new_temp[spark_mask] += np.random.random(np.sum(spark_mask)) * 0.3

        # Maintain heat sources at bottom
        new_temp[-3:, :] = np.maximum(new_temp[-3:, :], self.fuel_rate * self.intensity)

        # Clamp temperature
        new_temp = np.clip(new_temp, 0, 1)
        self.temperature = new_temp

        # Convert temperature to fire colors
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Fire color mapping: black -> red -> orange -> yellow -> white
        temp_normalized = self.temperature

        # Red channel
        frame[:, :, 0] = np.clip(temp_normalized * 255, 0, 255)

        # Green channel (starts later for orange/yellow)
        green_threshold = 0.4
        green_temp = np.maximum(0, temp_normalized - green_threshold) / (1 - green_threshold)
        frame[:, :, 1] = np.clip(green_temp * 255, 0, 255)

        # Blue channel (only for very hot regions -> white)
        blue_threshold = 0.8
        blue_temp = np.maximum(0, temp_normalized - blue_threshold) / (1 - blue_threshold)
        frame[:, :, 2] = np.clip(blue_temp * 255, 0, 255)

        self.frame_count += 1
        return frame


class Lightning(BaseEffect):
    """Lightning bolt generation with branching."""

    def initialize(self):
        # Support both strike_frequency and strike_probability for compatibility
        self.strike_frequency = self.config.get(
            "strike_frequency", self.config.get("strike_probability", 0.5)
        )  # Strikes per second
        self.branch_probability = self.config.get("branch_probability", 0.3)
        self.fade_time = self.config.get("fade_time", 0.3)  # How long bolts remain visible
        self.color = self.config.get("color", [255, 255, 255])  # Pure white for maximum brightness
        self.background_flash = self.config.get("background_flash", True)

        self.active_bolts = []
        self.last_strike_time = -10  # Start with negative time to trigger immediate strike
        self.flash_frame = np.zeros((self.height, self.width, 3), dtype=np.float32)

    def _generate_lightning_bolt(self, start_x, start_y, end_x, end_y, generation=0):
        """Generate a lightning bolt path with branching."""
        points = []

        # Create main path with random deviation
        num_segments = max(5, int(np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2) / 10))

        for i in range(num_segments + 1):
            t = i / num_segments

            # Linear interpolation with random deviation
            x = int(start_x + (end_x - start_x) * t)
            y = int(start_y + (end_y - start_y) * t)

            # Add random deviation (less for deeper generations)
            deviation = 20 / (generation + 1)
            x += int(np.random.uniform(-deviation, deviation))
            y += int(np.random.uniform(-deviation, deviation))

            # Keep in bounds
            x = np.clip(x, 0, self.width - 1)
            y = np.clip(y, 0, self.height - 1)

            points.append((x, y))

            # Branch probability decreases with generation
            if generation < 2 and i > 2 and np.random.random() < self.branch_probability / (generation + 1):
                # Create branch
                branch_length = np.random.uniform(0.3, 0.7) * (num_segments - i)
                branch_end_x = x + int(np.random.uniform(-30, 30))
                branch_end_y = min(self.height - 1, y + int(branch_length))

                branch_points = self._generate_lightning_bolt(x, y, branch_end_x, branch_end_y, generation + 1)
                points.extend(branch_points)

        return points

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time)

        # Fade existing flash
        self.flash_frame *= 0.9

        # Generate new lightning strike
        if t - self.last_strike_time > 1.0 / self.strike_frequency:
            # Random strike position
            start_x = int(np.random.uniform(0.2, 0.8) * self.width)
            start_y = 0
            end_x = int(np.random.uniform(0.1, 0.9) * self.width)
            end_y = self.height - 1

            # Generate bolt
            bolt_points = self._generate_lightning_bolt(start_x, start_y, end_x, end_y)

            self.active_bolts.append({"points": bolt_points, "start_time": t, "duration": self.fade_time})

            # Background flash
            if self.background_flash:
                flash_intensity = np.random.uniform(0.8, 1.0)  # Brighter minimum for tests
                self.flash_frame += flash_intensity * 255  # Bright white flash

            self.last_strike_time = t

        # Remove old bolts
        self.active_bolts = [bolt for bolt in self.active_bolts if t - bolt["start_time"] < bolt["duration"]]

        # Start with flash frame
        frame = np.clip(self.flash_frame, 0, 255).astype(np.uint8)

        # Draw active lightning bolts
        for bolt in self.active_bolts:
            age = t - bolt["start_time"]
            fade = max(0, 1 - age / bolt["duration"])

            # Draw bolt with fading
            bolt_color = [int(c * fade) for c in self.color]

            # Draw thick lines for LED visibility
            points = bolt["points"]
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], bolt_color, 3)

            # Add glow effect
            glow_color = [int(c * fade * 0.3) for c in self.color]
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], glow_color, 8)

        self.frame_count += 1
        return frame


class AuroraBorealis(BaseEffect):
    """Aurora Borealis/Northern Lights simulation."""

    def initialize(self):
        self.wave_speed = self.config.get("wave_speed", 0.3)
        self.color_palette = self.config.get("color_palette", "classic")  # classic, purple, blue
        self.intensity = self.config.get("intensity", 0.8)
        self.wave_count = self.config.get("wave_count", 2)  # Reduced from 3 to 2
        self.curtain_height = self.config.get("curtain_height", 0.7)  # How much of screen

        # Define color palettes
        self.palettes = {
            "classic": [(0, 255, 150), (0, 200, 255), (100, 255, 100)],  # Green-blue
            "purple": [(150, 50, 255), (255, 100, 200), (100, 150, 255)],  # Purple-pink
            "blue": [(0, 150, 255), (50, 200, 255), (100, 100, 255)],  # Blue variations
        }

        self.colors = self.palettes.get(self.color_palette, self.palettes["classic"])

        # Pre-compute wave pattern arrays for efficiency
        self.x_positions = np.arange(self.width)
        self.wave_cache_time = -1
        self.wave_cache_interval = 0.016  # Update every ~60fps
        self.cached_wave_y = np.zeros((self.wave_count, self.width))

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time) * self.wave_speed
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Create multiple aurora curtains
        curtain_base_y = int(self.height * (1 - self.curtain_height))

        # Update wave cache if needed
        if t - self.wave_cache_time > self.wave_cache_interval:
            for wave in range(self.wave_count):
                wave_offset = wave * 2 * np.pi / self.wave_count
                frequency = 1 + wave * 0.5
                amplitude = 30 + wave * 10

                # Vectorized wave calculation
                x_norm = self.x_positions / self.width * 2 * np.pi
                wave_height = amplitude * (
                    np.sin(frequency * x_norm + t + wave_offset) * 0.6
                    + np.sin(frequency * 1.7 * x_norm + t * 1.3 + wave_offset) * 0.3
                    + np.sin(frequency * 2.3 * x_norm - t * 0.7 + wave_offset) * 0.1
                )
                self.cached_wave_y[wave] = curtain_base_y + wave_height
            self.wave_cache_time = t

        for wave in range(self.wave_count):
            color = self.colors[wave % len(self.colors)]
            wave_y = self.cached_wave_y[wave]

            # Draw aurora curtain with vertical streaks
            for x in range(self.width):
                center_y = int(wave_y[x])

                # Draw vertical streak from wave position downward
                streak_length = int(40 + 20 * np.sin(t + x * 0.1))

                for y_offset in range(streak_length):
                    y = center_y + y_offset
                    if 0 <= y < self.height:
                        # Fade intensity with distance from center
                        fade = max(0, 1 - y_offset / streak_length)
                        fade *= self.intensity

                        # Add some horizontal spread for glow (reduced from ±2 to ±1)
                        for x_spread in range(-1, 2):
                            x_pos = x + x_spread
                            if 0 <= x_pos < self.width:
                                spread_fade = fade * max(0, 1 - abs(x_spread) / 2)

                                # Blend color
                                for c in range(3):
                                    frame[y, x_pos, c] = max(frame[y, x_pos, c], color[c] * spread_fade)

        # Simplified shimmer effect (optional - can be removed for more performance)
        if hasattr(self, "x_grid"):
            shimmer = (np.sin(t * 3 + self.x_grid * 0.1) + 1) * 0.1 + 0.9
            frame *= shimmer[:, :, np.newaxis]

        # Convert to uint8
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        self.frame_count += 1
        return frame


# Register effects
EffectRegistry.register(
    "fire_simulation",
    FireSimulation,
    "Fire",
    "Realistic fire simulation",
    "environmental",
    {"intensity": 0.8, "wind_strength": 0.1, "fuel_rate": 0.7, "cooling_rate": 0.05, "spark_probability": 0.02},
)

EffectRegistry.register(
    "lightning",
    Lightning,
    "Lightning",
    "Lightning bolts with branching",
    "environmental",
    {
        "strike_frequency": 0.5,
        "branch_probability": 0.3,
        "fade_time": 0.3,
        "color": [200, 200, 255],
        "background_flash": True,
    },
)

EffectRegistry.register(
    "aurora_borealis",
    AuroraBorealis,
    "Aurora",
    "Northern Lights simulation",
    "environmental",
    {"wave_speed": 0.3, "color_palette": "classic", "intensity": 0.8, "wave_count": 2, "curtain_height": 0.7},
)
