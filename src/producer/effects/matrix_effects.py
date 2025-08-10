"""Matrix/digital visual effects optimized for LED display"""

import cv2
import numpy as np

from .base_effect import BaseEffect, EffectRegistry


class DigitalRain(BaseEffect):
    """Matrix-style falling characters - simplified for LED display."""

    def initialize(self):
        self.column_count = self.config.get("column_count", 20)  # Fewer columns for visibility
        self.fall_speed = self.config.get("fall_speed", 0.5)
        self.color = self.config.get("color", [0, 255, 0])  # Classic matrix green
        self.fade_length = self.config.get("fade_length", 10)  # Trail length
        self.spawn_rate = self.config.get("spawn_rate", 0.1)  # New drops per frame

        # Initialize columns
        self.columns = []
        column_width = self.width / self.column_count

        for i in range(self.column_count):
            self.columns.append(
                {
                    "x": int((i + 0.5) * column_width),
                    "y": np.random.randint(-self.height, 0),
                    "speed": 0.5 + np.random.random() * 0.5,
                    "trail": [],
                }
            )

    def generate_frame(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for column in self.columns:
            # Update column position
            column["y"] += column["speed"] * self.fall_speed * 2

            # Add current position to trail
            if column["y"] >= 0 and column["y"] < self.height:
                column["trail"].append(int(column["y"]))

            # Limit trail length
            if len(column["trail"]) > self.fade_length:
                column["trail"] = column["trail"][-self.fade_length :]

            # Draw trail with fading
            for i, y in enumerate(column["trail"]):
                if 0 <= y < self.height:
                    fade = (i + 1) / len(column["trail"])
                    color = [int(c * fade) for c in self.color]

                    # Draw a thick line for LED visibility
                    x = column["x"]
                    thickness = max(2, self.width // self.column_count // 2)
                    cv2.line(frame, (x, y), (x, min(y + 2, self.height - 1)), color, thickness)

            # Reset column when it goes off screen
            if column["y"] > self.height + self.fade_length and np.random.random() < self.spawn_rate:
                column["y"] = np.random.randint(-self.height // 2, 0)
                column["speed"] = 0.5 + np.random.random() * 0.5
                column["trail"] = []

        self.frame_count += 1
        return frame


class BinaryStream(BaseEffect):
    """Flowing binary numbers - simplified for LED readability."""

    def initialize(self):
        self.flow_direction = self.config.get("flow_direction", "vertical")  # vertical, horizontal
        self.flow_speed = self.config.get("flow_speed", 0.3)
        self.density = self.config.get("density", 0.3)  # Bit density
        self.color_0 = self.config.get("color_0", [0, 100, 0])  # Color for 0
        self.color_1 = self.config.get("color_1", [0, 255, 0])  # Color for 1
        self.block_size = self.config.get("block_size", 8)  # Size of each bit block

        # Initialize binary grid
        grid_width = self.width // self.block_size
        grid_height = self.height // self.block_size
        self.binary_grid = np.random.choice(
            [0, 1], size=(grid_height * 2, grid_width), p=[1 - self.density, self.density]
        )
        self.offset = 0

    def generate_frame(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Update offset for scrolling
        self.offset += self.flow_speed

        if self.flow_direction == "vertical":
            # Vertical scrolling
            pixel_offset = int(self.offset * self.block_size) % (self.binary_grid.shape[0] * self.block_size)

            for y in range(self.height // self.block_size + 1):
                for x in range(self.width // self.block_size):
                    # Get bit value with wrapping
                    grid_y = (y + pixel_offset // self.block_size) % self.binary_grid.shape[0]
                    bit = self.binary_grid[grid_y, x]

                    # Draw block
                    block_y = y * self.block_size - (pixel_offset % self.block_size)
                    block_x = x * self.block_size

                    if block_y < self.height and block_y + self.block_size > 0:
                        color = self.color_1 if bit else self.color_0
                        cv2.rectangle(
                            frame,
                            (block_x, max(0, block_y)),
                            (min(self.width, block_x + self.block_size), min(self.height, block_y + self.block_size)),
                            color,
                            -1,
                        )
        else:
            # Horizontal scrolling
            pixel_offset = int(self.offset * self.block_size) % (self.binary_grid.shape[1] * self.block_size)

            for y in range(self.height // self.block_size):
                for x in range(self.width // self.block_size + 1):
                    # Get bit value with wrapping
                    grid_x = (x + pixel_offset // self.block_size) % self.binary_grid.shape[1]
                    grid_y = y % self.binary_grid.shape[0]
                    bit = self.binary_grid[grid_y, grid_x]

                    # Draw block
                    block_x = x * self.block_size - (pixel_offset % self.block_size)
                    block_y = y * self.block_size

                    if block_x < self.width and block_x + self.block_size > 0:
                        color = self.color_1 if bit else self.color_0
                        cv2.rectangle(
                            frame,
                            (max(0, block_x), block_y),
                            (min(self.width, block_x + self.block_size), min(self.height, block_y + self.block_size)),
                            color,
                            -1,
                        )

        # Regenerate bits that scroll off screen
        if self.offset * self.block_size > self.binary_grid.shape[0] * self.block_size:
            self.offset = 0
            self.binary_grid = np.random.choice([0, 1], size=self.binary_grid.shape, p=[1 - self.density, self.density])

        self.frame_count += 1
        return frame


class GlitchArt(BaseEffect):
    """Digital glitch effects - bold patterns for LED display."""

    def initialize(self):
        self.glitch_intensity = self.config.get("glitch_intensity", 0.5)
        self.glitch_types = self.config.get("glitch_types", ["shift", "corrupt", "tear"])
        self.base_pattern = self.config.get("base_pattern", "bars")  # bars, blocks, gradient
        self.color_shift = self.config.get("color_shift", True)
        self.glitch_frequency = self.config.get("glitch_frequency", 2.0)  # Glitches per second

        self.last_glitch_time = 0
        self.current_glitch = None
        self.base_frame = self._create_base_pattern()

    def _create_base_pattern(self):
        """Create the base pattern to glitch."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if self.base_pattern == "bars":
            # Horizontal color bars
            bar_height = self.height // 6
            colors = [[255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255]]
            for i, color in enumerate(colors):
                y_start = i * bar_height
                y_end = min((i + 1) * bar_height, self.height)
                frame[y_start:y_end, :] = color

        elif self.base_pattern == "blocks":
            # Checkerboard blocks
            block_size = 16
            for y in range(0, self.height, block_size):
                for x in range(0, self.width, block_size):
                    if ((x // block_size) + (y // block_size)) % 2:
                        color = [255, 0, 128]
                    else:
                        color = [0, 128, 255]
                    frame[y : y + block_size, x : x + block_size] = color

        else:  # gradient
            # Color gradient
            for y in range(self.height):
                hue = y / self.height
                color_hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
                color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]
                frame[y, :] = color_rgb

        return frame

    def generate_frame(self) -> np.ndarray:
        t = self.get_time()
        frame = self.base_frame.copy()

        # Trigger new glitch
        if t - self.last_glitch_time > 1.0 / self.glitch_frequency:
            self.current_glitch = {
                "type": np.random.choice(self.glitch_types),
                "start_time": t,
                "duration": 0.1 + np.random.random() * 0.2,
                "params": {
                    "offset": np.random.randint(-20, 20),
                    "rows": np.random.randint(5, self.height // 4),
                    "position": np.random.randint(0, self.height),
                },
            }
            self.last_glitch_time = t

        # Apply current glitch
        if self.current_glitch and t - self.current_glitch["start_time"] < self.current_glitch["duration"]:
            params = self.current_glitch["params"]

            if self.current_glitch["type"] == "shift":
                # Horizontal shift glitch
                start_y = params["position"]
                end_y = min(start_y + params["rows"], self.height)
                shift = int(params["offset"] * self.glitch_intensity)

                if shift > 0:
                    frame[start_y:end_y, shift:] = frame[start_y:end_y, :-shift]
                elif shift < 0:
                    frame[start_y:end_y, :shift] = frame[start_y:end_y, -shift:]

            elif self.current_glitch["type"] == "corrupt":
                # Data corruption glitch
                start_y = params["position"]
                end_y = min(start_y + params["rows"], self.height)

                # Random pixel corruption
                corruption_mask = np.random.random((end_y - start_y, self.width)) < self.glitch_intensity * 0.1
                random_colors = np.random.randint(0, 255, (end_y - start_y, self.width, 3))
                frame[start_y:end_y][corruption_mask] = random_colors[corruption_mask]

            elif self.current_glitch["type"] == "tear":
                # Vertical tear glitch
                tear_x = int(self.width * np.random.random())
                tear_width = int(self.width * 0.1 * self.glitch_intensity)

                # Duplicate and shift section
                if tear_x + tear_width < self.width:
                    section = frame[:, tear_x : tear_x + tear_width].copy()
                    offset_y = params["offset"]
                    if offset_y > 0:
                        frame[offset_y:, tear_x : tear_x + tear_width] = section[:-offset_y]
                    elif offset_y < 0:
                        frame[:offset_y, tear_x : tear_x + tear_width] = section[-offset_y:]

        # Color channel shift
        if self.color_shift and np.random.random() < self.glitch_intensity * 0.3:
            # Shift color channels
            shift_amount = int(self.glitch_intensity * 5)
            temp = frame.copy()
            frame[:, shift_amount:, 0] = temp[:, :-shift_amount, 0]  # Shift red channel
            frame[:, :-shift_amount, 2] = temp[:, shift_amount:, 2]  # Shift blue channel

        self.frame_count += 1
        return frame


# Register effects
EffectRegistry.register(
    "digital_rain",
    DigitalRain,
    "Digital Rain",
    "Matrix-style falling code",
    "matrix",
    {"column_count": 20, "fall_speed": 0.5, "color": [0, 255, 0], "fade_length": 10, "spawn_rate": 0.1},
)

EffectRegistry.register(
    "binary_stream",
    BinaryStream,
    "Binary Stream",
    "Flowing binary digits",
    "matrix",
    {
        "flow_direction": "vertical",
        "flow_speed": 0.3,
        "density": 0.3,
        "color_0": [0, 100, 0],
        "color_1": [0, 255, 0],
        "block_size": 8,
    },
)

EffectRegistry.register(
    "glitch_art",
    GlitchArt,
    "Glitch Art",
    "Digital glitch effects",
    "matrix",
    {
        "glitch_intensity": 0.5,
        "glitch_types": ["shift", "corrupt", "tear"],
        "base_pattern": "bars",
        "color_shift": True,
        "glitch_frequency": 2.0,
    },
)
