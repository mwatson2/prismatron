"""Geometric pattern effects optimized for LED display"""

import cv2
import numpy as np

from .base_effect import BaseEffect, EffectRegistry


class RotatingShapes(BaseEffect):
    """Large rotating geometric shapes."""

    def initialize(self):
        self.shape_type = self.config.get("shape_type", "triangle")  # triangle, square, hexagon, star
        self.rotation_speed = self.config.get("rotation_speed", 0.5)  # rotations per second
        self.size = self.config.get("size", 0.6)  # 0-1 relative to frame
        self.color = self.config.get("color", [0, 255, 255])  # RGB
        self.outline_thickness = self.config.get("outline_thickness", 0)  # 0 for filled
        self.num_shapes = self.config.get("num_shapes", 1)  # Multiple shapes
        self.rainbow_mode = self.config.get("rainbow_mode", False)

    def _get_shape_points(self, shape_type, center, radius, angle):
        """Get points for a shape."""
        if shape_type == "triangle":
            n_points = 3
        elif shape_type == "square":
            n_points = 4
            angle += np.pi / 4  # Rotate square to be axis-aligned
        elif shape_type == "hexagon":
            n_points = 6
        elif shape_type == "star":
            n_points = 10  # 5-pointed star (alternating inner/outer points)
        else:
            n_points = 5  # Pentagon default

        points = []
        for i in range(n_points):
            if shape_type == "star":
                # Alternate between outer and inner radius for star
                r = radius if i % 2 == 0 else radius * 0.4
            else:
                r = radius

            theta = angle + (2 * np.pi * i / n_points)
            x = int(center[0] + r * np.cos(theta))
            y = int(center[1] + r * np.sin(theta))
            points.append([x, y])

        return np.array(points, dtype=np.int32)

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        angle = 2 * np.pi * self.rotation_speed * t

        for i in range(self.num_shapes):
            if self.num_shapes > 1:
                # Distribute shapes evenly
                shape_angle = angle + (2 * np.pi * i / self.num_shapes)
                if self.rainbow_mode:
                    hue = i / self.num_shapes
                    color_hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
                    color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]
                    color = color_rgb.tolist()
                else:
                    color = self.color
            else:
                shape_angle = angle
                color = self.color

            center = (int(self.center_x), int(self.center_y))
            radius = min(self.width, self.height) * self.size * 0.5

            points = self._get_shape_points(self.shape_type, center, radius, shape_angle)

            if self.outline_thickness == 0:
                cv2.fillPoly(frame, [points], color)
            else:
                # Scale thickness for visibility on LED display
                thickness = max(2, int(self.outline_thickness * min(self.width, self.height) / 64))
                cv2.polylines(frame, [points], True, color, thickness)

        self.frame_count += 1
        return frame


class Kaleidoscope(BaseEffect):
    """Simple kaleidoscope with large segments."""

    def initialize(self):
        self.segments = self.config.get("segments", 6)  # Number of mirror segments
        self.rotation_speed = self.config.get("rotation_speed", 0.2)
        self.pattern_type = self.config.get("pattern_type", "dots")  # dots, lines, gradient
        self.color_speed = self.config.get("color_speed", 0.5)
        self.pattern_scale = self.config.get("pattern_scale", 0.3)

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Create base pattern in one segment
        segment_angle = 2 * np.pi / self.segments
        rotation = self.rotation_speed * t * 2 * np.pi

        # Create pattern based on type
        if self.pattern_type == "dots":
            # Large dots that will be visible on LEDs
            num_dots = 3
            for i in range(num_dots):
                dot_r = (i + 1) * self.pattern_scale * min(self.width, self.height) / (num_dots + 1)
                dot_angle = rotation + i * 0.5
                dot_x = int(self.center_x + dot_r * np.cos(dot_angle))
                dot_y = int(self.center_y + dot_r * np.sin(dot_angle))

                # Color based on position and time
                hue = (i / num_dots + t * self.color_speed) % 1.0
                color_hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
                color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]

                radius = int(min(self.width, self.height) * 0.05)
                cv2.circle(frame, (dot_x, dot_y), radius, color_rgb.tolist(), -1)

        elif self.pattern_type == "lines":
            # Radial lines
            for i in range(3):
                angle = rotation + i * segment_angle / 3
                end_x = int(self.center_x + self.width * np.cos(angle))
                end_y = int(self.center_y + self.height * np.sin(angle))

                hue = (i / 3 + t * self.color_speed) % 1.0
                color_hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
                color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]

                thickness = max(2, int(min(self.width, self.height) / 32))
                cv2.line(frame, (int(self.center_x), int(self.center_y)), (end_x, end_y), color_rgb.tolist(), thickness)

        else:  # gradient
            # Radial gradient segment
            hue = (t * self.color_speed) % 1.0
            hue_array = np.ones((self.height, self.width)) * hue + self.radius * 0.2
            saturation = np.ones((self.height, self.width))
            value = 1.0 - np.clip(self.radius / 2, 0, 1)
            frame = self.hsv_to_rgb(hue_array % 1.0, saturation, value)

        # Mirror the pattern for kaleidoscope effect
        result = np.zeros_like(frame)
        for seg in range(self.segments):
            angle = seg * segment_angle

            # Create rotation matrix
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Transform coordinates
            x_centered = self.x_grid - self.center_x
            y_centered = self.y_grid - self.center_y

            x_rot = x_centered * cos_a - y_centered * sin_a + self.center_x
            y_rot = x_centered * sin_a + y_centered * cos_a + self.center_y

            # Get rotated pattern
            x_rot = np.clip(x_rot.astype(int), 0, self.width - 1)
            y_rot = np.clip(y_rot.astype(int), 0, self.height - 1)

            # Add mirrored segment
            if seg % 2 == 0:
                result = np.maximum(result, frame[y_rot, x_rot])
            else:
                # Flip for mirror effect
                result = np.maximum(result, frame[y_rot, self.width - 1 - x_rot])

        self.frame_count += 1
        return result


class Spirals(BaseEffect):
    """Bold spiral patterns visible on LED display."""

    def initialize(self):
        self.spiral_type = self.config.get("spiral_type", "archimedean")  # archimedean, logarithmic
        self.rotation_speed = self.config.get("rotation_speed", 0.3)
        self.line_thickness = self.config.get("line_thickness", 0.15)  # Relative thickness
        self.num_arms = self.config.get("num_arms", 2)
        self.color_mode = self.config.get("color_mode", "rainbow")  # rainbow, solid, gradient
        self.tightness = self.config.get("tightness", 0.5)  # How tight the spiral is

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        rotation = self.rotation_speed * t * 2 * np.pi

        # Calculate spiral distance for each pixel
        if self.spiral_type == "archimedean":
            # r = a + b*theta
            spiral_param = self.angle + rotation
            expected_radius = (spiral_param % (2 * np.pi)) / (2 * np.pi) * self.tightness
        else:  # logarithmic
            # r = a * e^(b*theta)
            spiral_param = self.angle + rotation
            expected_radius = 0.1 * np.exp(self.tightness * (spiral_param % (2 * np.pi)))
            expected_radius = expected_radius % 1.0

        # Create multiple spiral arms
        for arm in range(self.num_arms):
            arm_offset = 2 * np.pi * arm / self.num_arms
            arm_expected_radius = (expected_radius + arm_offset / (2 * np.pi)) % 1.0

            # Calculate distance from spiral
            distance = np.abs(self.radius - arm_expected_radius)

            # Create thick line
            mask = distance < self.line_thickness

            if self.color_mode == "rainbow":
                hue = (arm / self.num_arms + t * 0.2) % 1.0
                color = cv2.cvtColor(np.array([[[hue * 180, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
            elif self.color_mode == "gradient":
                # Gradient along the spiral
                hue = (self.radius + t * 0.2) % 1.0
                hue_array = np.ones((self.height, self.width)) * hue
                saturation = np.ones((self.height, self.width))
                value = np.ones((self.height, self.width))
                gradient_frame = self.hsv_to_rgb(hue_array, saturation, value)
                frame[mask] = gradient_frame[mask]
                continue
            else:  # solid
                color = [255, 100, 0]  # Orange default

            frame[mask] = color

        self.frame_count += 1
        return frame


class Mandala(BaseEffect):
    """Simple mandala patterns with radial symmetry."""

    def initialize(self):
        self.complexity = self.config.get("complexity", 3)  # Number of ring layers
        self.symmetry = self.config.get("symmetry", 8)  # Rotational symmetry order
        self.rotation_speed = self.config.get("rotation_speed", 0.1)
        self.color_palette = self.config.get("color_palette", "rainbow")  # rainbow, warm, cool
        self.pulse_speed = self.config.get("pulse_speed", 0.5)  # Pulsing animation

    def generate_frame(self, presentation_time: float) -> np.ndarray:
        t = self.get_time(presentation_time)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        rotation = self.rotation_speed * t * 2 * np.pi
        pulse = (np.sin(t * self.pulse_speed * 2 * np.pi) + 1) / 2

        # Create concentric rings
        for ring in range(self.complexity):
            ring_radius = (ring + 1) / (self.complexity + 1)
            ring_radius *= 0.8 + 0.2 * pulse  # Pulse effect

            # Create pattern for this ring
            for sym in range(self.symmetry):
                angle = rotation + 2 * np.pi * sym / self.symmetry

                # Add different elements based on ring
                if ring % 2 == 0:
                    # Circles
                    element_angle = angle + ring * np.pi / 6
                    x = int(self.center_x + ring_radius * self.width / 2 * np.cos(element_angle))
                    y = int(self.center_y + ring_radius * self.height / 2 * np.sin(element_angle))

                    radius = int(min(self.width, self.height) * 0.03 * (1 + ring * 0.3))

                    # Color based on palette
                    if self.color_palette == "rainbow":
                        hue = (sym / self.symmetry + ring * 0.2) % 1.0
                    elif self.color_palette == "warm":
                        hue = 0.1 * (sym / self.symmetry)  # Red to yellow range
                    else:  # cool
                        hue = 0.5 + 0.2 * (sym / self.symmetry)  # Blue to green range

                    color_hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
                    color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]

                    cv2.circle(frame, (x, y), radius, color_rgb.tolist(), -1)
                else:
                    # Lines
                    start_angle = angle
                    end_angle = angle + 2 * np.pi / self.symmetry * 0.7

                    x1 = int(self.center_x + ring_radius * self.width / 2 * 0.8 * np.cos(start_angle))
                    y1 = int(self.center_y + ring_radius * self.height / 2 * 0.8 * np.sin(start_angle))
                    x2 = int(self.center_x + ring_radius * self.width / 2 * np.cos(end_angle))
                    y2 = int(self.center_y + ring_radius * self.height / 2 * np.sin(end_angle))

                    if self.color_palette == "rainbow":
                        hue = (sym / self.symmetry + ring * 0.2 + 0.5) % 1.0
                    elif self.color_palette == "warm":
                        hue = 0.05 + 0.1 * (sym / self.symmetry)
                    else:  # cool
                        hue = 0.6 + 0.15 * (sym / self.symmetry)

                    color_hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
                    color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]

                    thickness = max(2, int(min(self.width, self.height) / 32))
                    cv2.line(frame, (x1, y1), (x2, y2), color_rgb.tolist(), thickness)

        self.frame_count += 1
        return frame


# Register effects
EffectRegistry.register(
    "rotating_shapes",
    RotatingShapes,
    "Rotating Shapes",
    "Large rotating geometric shapes",
    "geometric",
    {
        "shape_type": "triangle",
        "rotation_speed": 0.5,
        "size": 0.6,
        "color": [0, 255, 255],
        "outline_thickness": 0,
        "num_shapes": 1,
        "rainbow_mode": False,
    },
)

EffectRegistry.register(
    "kaleidoscope",
    Kaleidoscope,
    "Kaleidoscope",
    "Simple kaleidoscope with large segments",
    "geometric",
    {"segments": 6, "rotation_speed": 0.2, "pattern_type": "dots", "color_speed": 0.5, "pattern_scale": 0.3},
)

EffectRegistry.register(
    "spirals",
    Spirals,
    "Spirals",
    "Bold spiral patterns",
    "geometric",
    {
        "spiral_type": "archimedean",
        "rotation_speed": 0.3,
        "line_thickness": 0.15,
        "num_arms": 2,
        "color_mode": "rainbow",
        "tightness": 0.5,
    },
)

EffectRegistry.register(
    "mandala",
    Mandala,
    "Mandala",
    "Simple mandala with radial symmetry",
    "geometric",
    {"complexity": 3, "symmetry": 8, "rotation_speed": 0.1, "color_palette": "rainbow", "pulse_speed": 0.5},
)
