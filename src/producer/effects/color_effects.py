"""Color-based visual effects optimized for LED display"""

import numpy as np

from .base_effect import BaseEffect, EffectRegistry


class RainbowSweep(BaseEffect):
    """Smooth rainbow gradient sweeping across display."""

    def initialize(self):
        self.speed = self.config.get("speed", 0.5)
        self.direction = self.config.get("direction", "horizontal")  # horizontal, vertical, diagonal, radial
        self.saturation = self.config.get("saturation", 1.0)
        self.brightness = self.config.get("brightness", 1.0)
        self.wave_width = self.config.get("wave_width", 1.0)  # How many rainbows fit on screen

    def generate_frame(self) -> np.ndarray:
        t = self.get_time() * self.speed

        if self.direction == "horizontal":
            position = self.x_norm * self.wave_width + t
        elif self.direction == "vertical":
            position = self.y_norm * self.wave_width + t
        elif self.direction == "diagonal":
            position = (self.x_norm + self.y_norm) * self.wave_width * 0.5 + t
        else:  # radial
            position = self.radius * self.wave_width + t

        hue = position % 1.0
        saturation = np.ones_like(hue) * self.saturation
        value = np.ones_like(hue) * self.brightness

        self.frame_count += 1
        return self.hsv_to_rgb(hue, saturation, value)


class ColorBreathe(BaseEffect):
    """Pulsing color intensity like breathing."""

    def initialize(self):
        self.base_color = self.config.get("base_color", [255, 0, 128])  # RGB
        self.breathe_rate = self.config.get("breathe_rate", 0.5)  # Hz
        self.intensity_min = self.config.get("intensity_min", 0.2)
        self.intensity_max = self.config.get("intensity_max", 1.0)
        self.color_shift = self.config.get("color_shift", False)  # Shift hue while breathing
        self.shift_amount = self.config.get("shift_amount", 0.1)

        # Convert base color to HSV for easier manipulation
        r, g, b = [c / 255.0 for c in self.base_color]
        self.base_hue = self._rgb_to_hue(r, g, b)

    def _rgb_to_hue(self, r, g, b):
        """Convert RGB to hue value."""
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        diff = max_c - min_c

        if diff == 0:
            return 0

        if max_c == r:
            hue = ((g - b) / diff) % 6
        elif max_c == g:
            hue = (b - r) / diff + 2
        else:
            hue = (r - g) / diff + 4

        return hue / 6

    def generate_frame(self) -> np.ndarray:
        t = self.get_time()

        # Breathing intensity using sine wave
        breathe = np.sin(2 * np.pi * self.breathe_rate * t)
        intensity = self.intensity_min + (self.intensity_max - self.intensity_min) * (breathe + 1) / 2

        # Create full frame with solid color
        frame = np.ones((self.height, self.width, 3)) * intensity

        if self.color_shift:
            # Shift hue slightly with breathing
            hue = (self.base_hue + self.shift_amount * breathe) % 1.0
            hue_array = np.ones((self.height, self.width)) * hue
            saturation = np.ones((self.height, self.width))
            value = np.ones((self.height, self.width)) * intensity
            frame = self.hsv_to_rgb(hue_array, saturation, value)
        else:
            # Just modulate intensity
            for i in range(3):
                frame[:, :, i] = self.base_color[i] * intensity
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        self.frame_count += 1
        return frame


class GradientFlow(BaseEffect):
    """Flowing gradients with multiple color stops."""

    def initialize(self):
        self.color_stops = self.config.get(
            "color_stops",
            [
                [255, 0, 0],  # Red
                [255, 255, 0],  # Yellow
                [0, 255, 0],  # Green
                [0, 255, 255],  # Cyan
                [0, 0, 255],  # Blue
                [255, 0, 255],  # Magenta
            ],
        )
        self.flow_speed = self.config.get("flow_speed", 0.3)
        self.flow_direction = self.config.get("flow_direction", "horizontal")
        self.gradient_type = self.config.get("gradient_type", "linear")  # linear, radial
        self.blend_width = self.config.get("blend_width", 0.5)  # Smoothness of transitions

    def generate_frame(self) -> np.ndarray:
        t = self.get_time() * self.flow_speed

        # Get position based on direction
        if self.gradient_type == "radial":
            position = self.radius
        elif self.flow_direction == "horizontal":
            position = (self.x_norm + 1) / 2  # Normalize to 0-1
        elif self.flow_direction == "vertical":
            position = (self.y_norm + 1) / 2
        else:  # diagonal
            position = ((self.x_norm + self.y_norm) / 2 + 1) / 2

        # Animate the gradient
        animated_position = (position + t) % 1.0

        # Create gradient with multiple stops
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)
        num_stops = len(self.color_stops)

        for i in range(num_stops):
            color1 = np.array(self.color_stops[i])
            color2 = np.array(self.color_stops[(i + 1) % num_stops])

            # Calculate position range for this color segment
            start = i / num_stops
            end = (i + 1) / num_stops

            # Create smooth transition
            mask = np.logical_and(animated_position >= start, animated_position < end)
            local_pos = (animated_position - start) / (end - start)
            local_pos = np.clip(local_pos, 0, 1)

            # Apply smoothstep for smoother transitions
            local_pos = local_pos * local_pos * (3 - 2 * local_pos)

            for c in range(3):
                frame[:, :, c] += mask * (color1[c] * (1 - local_pos) + color2[c] * local_pos)

        # Handle wrap-around
        wrap_mask = animated_position >= (num_stops - 1) / num_stops
        color1 = np.array(self.color_stops[-1])
        color2 = np.array(self.color_stops[0])
        local_pos = (animated_position - (num_stops - 1) / num_stops) * num_stops
        local_pos = np.clip(local_pos, 0, 1)
        local_pos = local_pos * local_pos * (3 - 2 * local_pos)

        for c in range(3):
            frame[:, :, c] += wrap_mask * (color1[c] * (1 - local_pos) + color2[c] * local_pos)

        self.frame_count += 1
        return np.clip(frame, 0, 255).astype(np.uint8)


class ColorWipe(BaseEffect):
    """Progressive color fill across the display."""

    def initialize(self):
        self.color_sequence = self.config.get(
            "color_sequence",
            [
                [255, 0, 0],  # Red
                [0, 255, 0],  # Green
                [0, 0, 255],  # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
            ],
        )
        self.wipe_speed = self.config.get("wipe_speed", 0.5)
        self.wipe_direction = self.config.get("wipe_direction", "left_right")
        self.wipe_width = self.config.get("wipe_width", 0.1)  # Transition zone width
        self.hold_time = self.config.get("hold_time", 0.5)  # Time to hold each color

        self.current_color_index = 0
        self.wipe_position = 0
        self.last_transition_time = 0

    def generate_frame(self) -> np.ndarray:
        t = self.get_time()

        # Calculate wipe cycle time
        cycle_time = 1.0 / self.wipe_speed + self.hold_time
        cycle_progress = (t % cycle_time) / cycle_time

        # Check if we should advance to next color
        if cycle_progress < self.hold_time / cycle_time:
            # Holding current color
            wipe_progress = 1.0
        else:
            # Wiping
            wipe_progress = (cycle_progress - self.hold_time / cycle_time) / (1 - self.hold_time / cycle_time)

        # Get current and next color
        current_color = np.array(self.color_sequence[self.current_color_index])
        next_color_index = (self.current_color_index + 1) % len(self.color_sequence)
        next_color = np.array(self.color_sequence[next_color_index])

        # Update color index when wipe completes
        if t - self.last_transition_time > cycle_time:
            self.current_color_index = next_color_index
            self.last_transition_time = t

        # Calculate wipe position based on direction
        if self.wipe_direction == "left_right":
            position = (self.x_norm + 1) / 2
        elif self.wipe_direction == "right_left":
            position = 1 - (self.x_norm + 1) / 2
        elif self.wipe_direction == "top_bottom":
            position = (self.y_norm + 1) / 2
        elif self.wipe_direction == "bottom_top":
            position = 1 - (self.y_norm + 1) / 2
        elif self.wipe_direction == "center_out":
            position = self.radius / np.sqrt(2)
        else:  # 'edge_in'
            position = 1 - self.radius / np.sqrt(2)

        # Create smooth transition
        transition = np.clip((position - wipe_progress + self.wipe_width) / self.wipe_width, 0, 1)
        transition = transition * transition * (3 - 2 * transition)  # Smoothstep

        # Blend colors
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for c in range(3):
            frame[:, :, c] = current_color[c] * (1 - transition) + next_color[c] * transition

        self.frame_count += 1
        return frame


# Register effects
EffectRegistry.register(
    "rainbow_sweep",
    RainbowSweep,
    "Rainbow Sweep",
    "Smooth rainbow gradient sweeping across display",
    "color",
    {"speed": 0.5, "direction": "horizontal", "saturation": 1.0, "brightness": 1.0, "wave_width": 1.0},
)

EffectRegistry.register(
    "color_breathe",
    ColorBreathe,
    "Color Breathe",
    "Pulsing color intensity like breathing",
    "color",
    {
        "base_color": [255, 0, 128],
        "breathe_rate": 0.5,
        "intensity_min": 0.2,
        "intensity_max": 1.0,
        "color_shift": False,
        "shift_amount": 0.1,
    },
)

EffectRegistry.register(
    "gradient_flow",
    GradientFlow,
    "Gradient Flow",
    "Flowing gradients with multiple color stops",
    "color",
    {
        "color_stops": [[255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255]],
        "flow_speed": 0.3,
        "flow_direction": "horizontal",
        "gradient_type": "linear",
        "blend_width": 0.5,
    },
)

EffectRegistry.register(
    "color_wipe",
    ColorWipe,
    "Color Wipe",
    "Progressive color fill across the display",
    "color",
    {
        "color_sequence": [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]],
        "wipe_speed": 0.5,
        "wipe_direction": "left_right",
        "wipe_width": 0.1,
        "hold_time": 0.5,
    },
)
