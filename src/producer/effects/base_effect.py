"""Base class for visual effects"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseEffect(ABC):
    """Base class for all visual effects.

    Designed for low-detail LED display - focuses on bold, large-scale patterns
    that will be visible when downsampled to ~2600 LEDs.
    """

    def __init__(self, width: int = 128, height: int = 64, fps: int = 30, config: Optional[Dict[str, Any]] = None):
        """Initialize effect.

        Args:
            width: Frame width (will be downsampled for LEDs)
            height: Frame height (will be downsampled for LEDs)
            fps: Target frames per second
            config: Effect-specific configuration
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.config = config or {}
        self.frame_count = 0

        # Seed random number generator for consistent test behavior
        # Use a deterministic seed based on effect instance parameters
        seed = hash((width, height, fps, str(sorted(self.config.items())))) % 2**32
        np.random.seed(seed)

        # Create coordinate grids for efficient calculations
        self.y_grid, self.x_grid = np.mgrid[0:height, 0:width]
        self.center_x = width / 2
        self.center_y = height / 2

        # Normalized coordinates (-1 to 1)
        self.x_norm = (self.x_grid - self.center_x) / (width / 2)
        self.y_norm = (self.y_grid - self.center_y) / (height / 2)

        # Polar coordinates for radial effects
        self.radius = np.sqrt(self.x_norm**2 + self.y_norm**2)
        self.angle = np.arctan2(self.y_norm, self.x_norm)

        self.initialize()

    @abstractmethod
    def initialize(self):
        """Initialize effect-specific parameters."""

    @abstractmethod
    def generate_frame(self, presentation_time: float) -> np.ndarray:
        """Generate the next frame.

        Args:
            presentation_time: Time in seconds for this frame (for consistent animation timing)

        Returns:
            RGB frame as numpy array of shape (height, width, 3) with values 0-255
        """

    def update_config(self, new_config: Dict[str, Any]):
        """Update effect configuration."""
        self.config.update(new_config)
        self.initialize()  # Reinitialize with new config

    def get_time(self, presentation_time: float) -> float:
        """Get elapsed time for animation.

        Args:
            presentation_time: Presentation timestamp for frame-based animation timing

        Returns:
            Time in seconds for animation calculations
        """
        return presentation_time

    def reset(self):
        """Reset effect to initial state."""
        self.frame_count = 0
        self.initialize()

    def hsv_to_rgb(self, h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB.

        Args:
            h: Hue (0-1)
            s: Saturation (0-1)
            v: Value (0-1)

        Returns:
            RGB array with values 0-255
        """
        h = h % 1.0  # Wrap hue

        i = np.floor(h * 6).astype(int)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        i = i % 6

        rgb = np.zeros((*h.shape, 3))

        idx = i == 0
        rgb[idx] = np.stack([v[idx], t[idx], p[idx]], axis=-1)

        idx = i == 1
        rgb[idx] = np.stack([q[idx], v[idx], p[idx]], axis=-1)

        idx = i == 2
        rgb[idx] = np.stack([p[idx], v[idx], t[idx]], axis=-1)

        idx = i == 3
        rgb[idx] = np.stack([p[idx], q[idx], v[idx]], axis=-1)

        idx = i == 4
        rgb[idx] = np.stack([t[idx], p[idx], v[idx]], axis=-1)

        idx = i == 5
        rgb[idx] = np.stack([v[idx], p[idx], q[idx]], axis=-1)

        return (rgb * 255).astype(np.uint8)

    def create_gradient(
        self, color1: Tuple[int, int, int], color2: Tuple[int, int, int], position: np.ndarray
    ) -> np.ndarray:
        """Create a gradient between two colors.

        Args:
            color1: RGB tuple for start color
            color2: RGB tuple for end color
            position: Position array (0-1) for gradient mapping

        Returns:
            RGB array
        """
        position = np.clip(position, 0, 1)
        gradient = np.zeros((*position.shape, 3), dtype=np.uint8)

        for i in range(3):
            gradient[..., i] = (color1[i] * (1 - position) + color2[i] * position).astype(np.uint8)

        return gradient


class EffectRegistry:
    """Registry for available effects."""

    _effects = {}

    @classmethod
    def register(
        cls,
        effect_id: str,
        effect_class: type,
        name: str,
        description: str,
        category: str,
        default_config: Dict[str, Any],
    ):
        """Register an effect."""
        cls._effects[effect_id] = {
            "class": effect_class,
            "name": name,
            "description": description,
            "category": category,
            "config": default_config,
            "icon": cls._get_icon_for_category(category),
        }

    @classmethod
    def get_effect(cls, effect_id: str) -> Optional[Dict[str, Any]]:
        """Get effect info by ID."""
        return cls._effects.get(effect_id)

    @classmethod
    def list_effects(cls) -> list:
        """List all registered effects."""
        return [{"id": effect_id, **info} for effect_id, info in cls._effects.items()]

    @classmethod
    def create_effect(
        cls, effect_id: str, width: int = 128, height: int = 64, fps: int = 30, config: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseEffect]:
        """Create an effect instance."""
        effect_info = cls._effects.get(effect_id)
        if not effect_info:
            return None

        effect_class = effect_info["class"]
        default_config = effect_info["config"].copy()
        if config:
            default_config.update(config)

        return effect_class(width, height, fps, default_config)

    @staticmethod
    def _get_icon_for_category(category: str) -> str:
        """Get emoji icon for category."""
        icons = {
            "geometric": "🔷",
            "particle": "✨",
            "wave": "🌊",
            "color": "🎨",
            "noise": "🌫️",
            "matrix": "💻",
            "environmental": "🌟",
        }
        return icons.get(category, "🎭")
