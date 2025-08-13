"""Particle system effects optimized for LED display"""

import cv2
import numpy as np

from .base_effect import BaseEffect, EffectRegistry


class Particle:
    """Simple particle for effects."""

    def __init__(self, x, y, vx, vy, color, life=1.0, size=1.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size


class Fireworks(BaseEffect):
    """Fireworks explosions with large, visible particles."""

    def initialize(self):
        self.explosion_frequency = self.config.get("explosion_frequency", 1.0)  # Explosions per second
        self.particle_count = self.config.get("particle_count", 30)  # Particles per explosion
        self.gravity = self.config.get("gravity", 0.1)
        self.colors = self.config.get("colors", "rainbow")  # rainbow, warm, cool, random
        self.trail_length = self.config.get("trail_length", 0.3)
        self.explosion_size = self.config.get("explosion_size", 0.3)

        self.particles = []
        self.last_explosion_time = -10  # Start with negative time to trigger immediate explosion
        self.trail_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)

    def create_explosion(self, x, y):
        """Create a new firework explosion."""
        # Choose color scheme
        if self.colors == "rainbow":
            base_hue = np.random.random()
            colors = []
            for i in range(self.particle_count):
                hue = (base_hue + i / self.particle_count) % 1.0
                color_hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
                color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()
                colors.append(color_rgb)
        elif self.colors == "warm":
            colors = [[255, np.random.randint(0, 128), 0] for _ in range(self.particle_count)]
        elif self.colors == "cool":
            colors = [[0, np.random.randint(0, 128), 255] for _ in range(self.particle_count)]
        else:  # random
            colors = [
                [np.random.randint(128, 255), np.random.randint(128, 255), np.random.randint(128, 255)]
                for _ in range(self.particle_count)
            ]

        # Create particles exploding outward
        for i in range(self.particle_count):
            angle = 2 * np.pi * i / self.particle_count + np.random.random() * 0.2
            speed = self.explosion_size * (0.5 + np.random.random() * 0.5)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)

            particle = Particle(x, y, vx, vy, colors[i], life=2.0, size=3.0)
            self.particles.append(particle)

    def generate_frame(self) -> np.ndarray:
        t = self.get_time()

        # Fade trail buffer
        self.trail_buffer *= 1 - self.trail_length

        # Create new explosion if it's time
        if t - self.last_explosion_time > 1.0 / self.explosion_frequency:
            # Random position for explosion
            x = np.random.uniform(0.2, 0.8)
            y = np.random.uniform(0.2, 0.8)
            self.create_explosion(x, y)
            self.last_explosion_time = t

        # Update and draw particles
        frame = self.trail_buffer.copy()
        surviving_particles = []

        for particle in self.particles:
            # Update physics
            particle.vy += self.gravity * 0.01
            particle.x += particle.vx * 0.01
            particle.y += particle.vy * 0.01
            particle.life -= 0.02

            if particle.life > 0 and 0 <= particle.x <= 1 and 0 <= particle.y <= 1:
                surviving_particles.append(particle)

                # Draw particle (larger for LED visibility)
                px = int(particle.x * self.width)
                py = int(particle.y * self.height)

                # Fade based on life
                alpha = particle.life / particle.max_life
                color = [c * alpha for c in particle.color]

                # Draw as a larger circle for visibility
                radius = int(particle.size * (1 + (1 - alpha) * 0.5))  # Expand as it fades
                cv2.circle(frame, (px, py), radius, color, -1)

        self.particles = surviving_particles
        self.trail_buffer = np.clip(frame, 0, 255)

        self.frame_count += 1
        return frame.astype(np.uint8)


class Starfield(BaseEffect):
    """3D starfield with large, visible stars."""

    def initialize(self):
        self.star_count = self.config.get("star_count", 50)  # Fewer stars, but larger
        self.speed = self.config.get("speed", 0.5)
        self.direction = self.config.get("direction", "forward")  # forward, backward, left, right
        self.star_size = self.config.get("star_size", 3)  # Minimum star size
        self.twinkle = self.config.get("twinkle", True)
        self.color_mode = self.config.get("color_mode", "white")  # white, colored, blue

        # Initialize stars with 3D positions
        self.stars = []
        for i in range(self.star_count):
            # Ensure some stars are very close (bright)
            if i < 10:  # First 10 stars are close for guaranteed brightness
                z = 0.01 + i * 0.02  # z from 0.01 to 0.19
                # Place bright stars near center to avoid going off-screen
                x = np.random.uniform(-0.5, 0.5)
                y = np.random.uniform(-0.5, 0.5)
            else:
                z = np.random.uniform(0.2, 1)
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
            self.stars.append(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "twinkle_phase": np.random.random() * 2 * np.pi,
                    "color": self._get_star_color(),
                }
            )

    def _get_star_color(self):
        """Get star color based on mode."""
        if self.color_mode == "white":
            return [255, 255, 255]
        elif self.color_mode == "blue":
            return [200, 200, 255]
        else:  # colored
            hue = np.random.random()
            color_hsv = np.array([[[hue * 180, 128, 255]]], dtype=np.uint8)
            return cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()

    def generate_frame(self) -> np.ndarray:
        t = self.get_time()
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for star in self.stars:
            # Update star position based on direction
            if self.direction == "forward":
                star["z"] -= self.speed * 0.01
                if star["z"] <= 0.01:  # Keep minimum distance
                    star["z"] = 1
                    star["x"] = np.random.uniform(-1, 1)
                    star["y"] = np.random.uniform(-1, 1)
            elif self.direction == "backward":
                star["z"] += self.speed * 0.01
                if star["z"] > 1:
                    star["z"] = 0.1
                    star["x"] = np.random.uniform(-1, 1)
                    star["y"] = np.random.uniform(-1, 1)
            elif self.direction == "left":
                star["x"] -= self.speed * 0.02
                if star["x"] < -1:
                    star["x"] = 1
                    star["y"] = np.random.uniform(-1, 1)
                    star["z"] = np.random.uniform(0.1, 1)
            else:  # right
                star["x"] += self.speed * 0.02
                if star["x"] > 1:
                    star["x"] = -1
                    star["y"] = np.random.uniform(-1, 1)
                    star["z"] = np.random.uniform(0.1, 1)

            # Project 3D position to 2D
            if self.direction in ["forward", "backward"]:
                screen_x = star["x"] / star["z"]
                screen_y = star["y"] / star["z"]
            else:
                screen_x = star["x"]
                screen_y = star["y"]

            # Convert to pixel coordinates
            px = int((screen_x + 1) / 2 * self.width)
            py = int((screen_y + 1) / 2 * self.height)

            if 0 <= px < self.width and 0 <= py < self.height:
                # Calculate star brightness and size based on Z depth
                brightness = 1.0 - star["z"]

                # Add twinkle effect (subtle - only 20% variation)
                if self.twinkle:
                    twinkle = (np.sin(star["twinkle_phase"] + t * 3) + 1) / 2
                    brightness *= 0.8 + 0.2 * twinkle  # 80% to 100% brightness

                # Calculate size (larger stars for LED visibility)
                size = int(self.star_size * (1 + brightness))

                # Draw star
                color = [int(c * brightness) for c in star["color"]]
                cv2.circle(frame, (px, py), size, color, -1)

        self.frame_count += 1
        return frame


class RainSnow(BaseEffect):
    """Rain or snow falling effect with large particles."""

    def initialize(self):
        self.particle_type = self.config.get("particle_type", "rain")  # rain, snow
        self.particle_density = self.config.get("particle_density", 30)  # Number of particles
        self.fall_speed = self.config.get("fall_speed", 0.5)
        self.wind_strength = self.config.get("wind_strength", 0.1)
        self.wind_variation = self.config.get("wind_variation", True)

        # Initialize particles
        self.particles = []
        for _ in range(self.particle_density):
            self.particles.append(
                {
                    "x": np.random.uniform(0, 1),
                    "y": np.random.uniform(0, 1),
                    "speed": 0.5 + np.random.random() * 0.5,
                    "size": np.random.uniform(0.5, 1.5) if self.particle_type == "snow" else 1.0,
                    "wind_phase": np.random.random() * 2 * np.pi,
                }
            )

    def generate_frame(self) -> np.ndarray:
        t = self.get_time()
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Calculate wind
        if self.wind_variation:
            wind = self.wind_strength * np.sin(t * 0.5)
        else:
            wind = self.wind_strength

        for particle in self.particles:
            # Update particle position
            particle["y"] += self.fall_speed * particle["speed"] * 0.02

            if self.particle_type == "snow":
                # Snow has more horizontal movement
                particle["x"] += wind * 0.01 + 0.005 * np.sin(particle["wind_phase"] + t * 2)
            else:
                # Rain falls more straight
                particle["x"] += wind * 0.005

            # Wrap around edges
            if particle["y"] > 1:
                particle["y"] = 0
                particle["x"] = np.random.uniform(0, 1)
            if particle["x"] < 0:
                particle["x"] = 1
            elif particle["x"] > 1:
                particle["x"] = 0

            # Draw particle
            px = int(particle["x"] * self.width)
            py = int(particle["y"] * self.height)

            if self.particle_type == "snow":
                # White snowflakes
                size = int(particle["size"] * 3)
                cv2.circle(frame, (px, py), size, (255, 255, 255), -1)
            else:
                # Blue rain streaks
                streak_length = int(self.fall_speed * 10)
                py_end = min(py + streak_length, self.height - 1)
                cv2.line(frame, (px, py), (px, py_end), (100, 150, 255), 2)

        self.frame_count += 1
        return frame


class SwarmBehavior(BaseEffect):
    """Flocking/swarm simulation with boid-like behavior."""

    def initialize(self):
        self.swarm_size = self.config.get("swarm_size", 20)  # Fewer but larger boids
        self.cohesion = self.config.get("cohesion", 0.5)
        self.separation = self.config.get("separation", 0.5)
        self.alignment = self.config.get("alignment", 0.5)
        self.max_speed = self.config.get("max_speed", 0.02)
        self.color_mode = self.config.get("color_mode", "gradient")  # gradient, uniform, rainbow
        self.trail = self.config.get("trail", True)

        # Initialize boids
        self.boids = []
        for i in range(self.swarm_size):
            self.boids.append(
                {
                    "x": np.random.uniform(0.2, 0.8),
                    "y": np.random.uniform(0.2, 0.8),
                    "vx": np.random.uniform(-0.01, 0.01),
                    "vy": np.random.uniform(-0.01, 0.01),
                    "color": self._get_boid_color(i),
                }
            )

        self.trail_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)

    def _get_boid_color(self, index):
        """Get color for a boid."""
        if self.color_mode == "gradient":
            hue = index / self.swarm_size
            color_hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
            return cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()
        elif self.color_mode == "rainbow":
            hue = (index / self.swarm_size) % 1.0
            color_hsv = np.array([[[hue * 180, 255, 255]]], dtype=np.uint8)
            return cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()
        else:  # uniform
            return [255, 200, 0]  # Yellow

    def generate_frame(self) -> np.ndarray:
        # Fade trail
        if self.trail:
            self.trail_buffer *= 0.9
        else:
            self.trail_buffer.fill(0)

        # Update boid positions using flocking rules
        for i, boid in enumerate(self.boids):
            # Find nearby boids
            neighbors = []
            for j, other in enumerate(self.boids):
                if i != j:
                    dist = np.sqrt((boid["x"] - other["x"]) ** 2 + (boid["y"] - other["y"]) ** 2)
                    if dist < 0.2:  # Neighbor radius
                        neighbors.append(other)

            if neighbors:
                # Cohesion - move toward center of neighbors
                center_x = np.mean([n["x"] for n in neighbors])
                center_y = np.mean([n["y"] for n in neighbors])
                cohesion_x = (center_x - boid["x"]) * self.cohesion * 0.01
                cohesion_y = (center_y - boid["y"]) * self.cohesion * 0.01

                # Separation - avoid crowding
                sep_x = sep_y = 0
                for n in neighbors:
                    dist = np.sqrt((boid["x"] - n["x"]) ** 2 + (boid["y"] - n["y"]) ** 2)
                    if dist < 0.05 and dist > 0:  # Too close
                        sep_x -= (n["x"] - boid["x"]) / dist
                        sep_y -= (n["y"] - boid["y"]) / dist
                sep_x *= self.separation * 0.01
                sep_y *= self.separation * 0.01

                # Alignment - match velocity of neighbors
                avg_vx = np.mean([n["vx"] for n in neighbors])
                avg_vy = np.mean([n["vy"] for n in neighbors])
                align_x = (avg_vx - boid["vx"]) * self.alignment * 0.01
                align_y = (avg_vy - boid["vy"]) * self.alignment * 0.01

                # Update velocity
                boid["vx"] += cohesion_x + sep_x + align_x
                boid["vy"] += cohesion_y + sep_y + align_y

            # Limit speed
            speed = np.sqrt(boid["vx"] ** 2 + boid["vy"] ** 2)
            if speed > self.max_speed:
                boid["vx"] = boid["vx"] / speed * self.max_speed
                boid["vy"] = boid["vy"] / speed * self.max_speed

            # Update position
            boid["x"] += boid["vx"]
            boid["y"] += boid["vy"]

            # Wrap around edges
            if boid["x"] < 0:
                boid["x"] = 1
            elif boid["x"] > 1:
                boid["x"] = 0
            if boid["y"] < 0:
                boid["y"] = 1
            elif boid["y"] > 1:
                boid["y"] = 0

            # Draw boid
            px = int(boid["x"] * self.width)
            py = int(boid["y"] * self.height)
            cv2.circle(self.trail_buffer, (px, py), 3, boid["color"], -1)

        frame = np.clip(self.trail_buffer, 0, 255).astype(np.uint8)
        self.frame_count += 1
        return frame


# Register effects
EffectRegistry.register(
    "fireworks",
    Fireworks,
    "Fireworks",
    "Colorful firework explosions",
    "particle",
    {
        "explosion_frequency": 1.0,
        "particle_count": 30,
        "gravity": 0.1,
        "colors": "rainbow",
        "trail_length": 0.3,
        "explosion_size": 0.3,
    },
)

EffectRegistry.register(
    "starfield",
    Starfield,
    "Starfield",
    "3D starfield effect",
    "particle",
    {"star_count": 50, "speed": 0.5, "direction": "forward", "star_size": 3, "twinkle": True, "color_mode": "white"},
)

EffectRegistry.register(
    "rain_snow",
    RainSnow,
    "Rain/Snow",
    "Falling rain or snow particles",
    "particle",
    {"particle_type": "rain", "particle_density": 30, "fall_speed": 0.5, "wind_strength": 0.1, "wind_variation": True},
)

EffectRegistry.register(
    "swarm",
    SwarmBehavior,
    "Swarm",
    "Flocking swarm behavior",
    "particle",
    {
        "swarm_size": 20,
        "cohesion": 0.5,
        "separation": 0.5,
        "alignment": 0.5,
        "max_speed": 0.02,
        "color_mode": "gradient",
        "trail": True,
    },
)
