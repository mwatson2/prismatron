"""
LED Position Mapping System.

This module implements LED position mapping, coordinate transformations,
and spatial indexing for the 3,200 randomly positioned RGB LEDs.
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT

logger = logging.getLogger(__name__)


@dataclass
class LEDPosition:
    """Represents the position and properties of a single LED."""

    led_id: int
    x: float  # X coordinate in display space (0.0 to 1.0)
    y: float  # Y coordinate in display space (0.0 to 1.0)
    pixel_x: int = (
        0  # X coordinate in frame buffer pixels (calculated in __post_init__)
    )
    pixel_y: int = (
        0  # Y coordinate in frame buffer pixels (calculated in __post_init__)
    )
    physical_x: float = 0.0  # Physical X position in mm (optional)
    physical_y: float = 0.0  # Physical Y position in mm (optional)
    calibration_data: Optional[Dict[str, Any]] = None  # LED-specific calibration

    def __post_init__(self):
        """Calculate pixel coordinates from normalized coordinates."""
        self.pixel_x = int(self.x * FRAME_WIDTH)
        self.pixel_y = int(self.y * FRAME_HEIGHT)

        # Clamp to valid frame bounds
        self.pixel_x = max(0, min(self.pixel_x, FRAME_WIDTH - 1))
        self.pixel_y = max(0, min(self.pixel_y, FRAME_HEIGHT - 1))

    def distance_to(self, other: "LEDPosition") -> float:
        """Calculate Euclidean distance to another LED position."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "led_id": self.led_id,
            "x": self.x,
            "y": self.y,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "physical_x": self.physical_x,
            "physical_y": self.physical_y,
            "calibration_data": self.calibration_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LEDPosition":
        """Create from dictionary."""
        return cls(
            led_id=data["led_id"],
            x=data["x"],
            y=data["y"],
            pixel_x=data.get("pixel_x", 0),
            pixel_y=data.get("pixel_y", 0),
            physical_x=data.get("physical_x", 0.0),
            physical_y=data.get("physical_y", 0.0),
            calibration_data=data.get("calibration_data"),
        )


class LEDMapper:
    """
    LED position mapping and spatial indexing system.

    Manages the positions of 3,200 randomly distributed LEDs and provides
    efficient spatial queries, coordinate transformations, and position
    management.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LED mapper.

        Args:
            config_path: Path to LED position configuration file
        """
        self.config_path = config_path or "config/led_positions.json"
        self.led_positions: List[LEDPosition] = []
        self._position_lookup: Dict[int, LEDPosition] = {}
        self._spatial_grid: Optional[np.ndarray] = None
        self._grid_resolution = 50  # Grid cells per dimension for spatial indexing

    def initialize(self) -> bool:
        """
        Initialize the LED mapper.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Try to load existing LED positions
            if os.path.exists(self.config_path):
                if self.load_led_positions():
                    logger.info(f"Loaded {len(self.led_positions)} LED positions")
                    self._build_spatial_index()
                    return True
                else:
                    logger.warning("Failed to load LED positions, generating new ones")

            # Generate new random positions if no config exists
            if self.generate_random_positions():
                self.save_led_positions()
                self._build_spatial_index()
                logger.info(f"Generated {len(self.led_positions)} random LED positions")
                return True

            return False

        except Exception as e:
            logger.error(f"LED mapper initialization failed: {e}")
            return False

    def generate_random_positions(self, seed: Optional[int] = 42) -> bool:
        """
        Generate random LED positions.

        Args:
            seed: Random seed for reproducible positions

        Returns:
            True if generation successful, False otherwise
        """
        try:
            if seed is not None:
                np.random.seed(seed)

            positions = []

            for led_id in range(LED_COUNT):
                # Generate random position in normalized coordinates
                x = np.random.random()
                y = np.random.random()

                # Create LED position
                led_pos = LEDPosition(
                    led_id=led_id,
                    x=x,
                    y=y,
                    pixel_x=0,  # Will be calculated in __post_init__
                    pixel_y=0,
                )

                positions.append(led_pos)

            self.led_positions = positions
            self._build_position_lookup()

            logger.info(f"Generated {len(positions)} random LED positions")
            return True

        except Exception as e:
            logger.error(f"Failed to generate random positions: {e}")
            return False

    def load_led_positions(self) -> bool:
        """
        Load LED positions from configuration file.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)

            positions = []
            led_data = data.get("led_positions", [])

            for led_dict in led_data:
                led_pos = LEDPosition.from_dict(led_dict)
                positions.append(led_pos)

            if len(positions) != LED_COUNT:
                logger.error(
                    f"Expected {LED_COUNT} LEDs, got {len(positions)} from config"
                )
                return False

            self.led_positions = positions
            self._build_position_lookup()

            logger.info(
                f"Loaded {len(positions)} LED positions from {self.config_path}"
            )
            return True

        except FileNotFoundError:
            logger.info(f"LED position file not found: {self.config_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load LED positions: {e}")
            return False

    def save_led_positions(self) -> bool:
        """
        Save LED positions to configuration file.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure config directory exists
            config_dir = Path(self.config_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data for serialization
            data = {
                "version": "1.0",
                "led_count": len(self.led_positions),
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "generated_timestamp": "",
                "led_positions": [led.to_dict() for led in self.led_positions],
            }

            # Save to file
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(
                f"Saved {len(self.led_positions)} LED positions to {self.config_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save LED positions: {e}")
            return False

    def _build_position_lookup(self) -> None:
        """Build LED ID to position lookup table."""
        self._position_lookup = {led.led_id: led for led in self.led_positions}

    def _build_spatial_index(self) -> None:
        """Build spatial grid index for efficient neighbor queries."""
        try:
            # Create grid to store LED IDs
            self._spatial_grid = {}

            for led in self.led_positions:
                # Calculate grid cell
                grid_x = int(led.x * self._grid_resolution)
                grid_y = int(led.y * self._grid_resolution)

                # Clamp to valid grid bounds
                grid_x = max(0, min(grid_x, self._grid_resolution - 1))
                grid_y = max(0, min(grid_y, self._grid_resolution - 1))

                # Add LED to grid cell
                grid_key = (grid_x, grid_y)
                if grid_key not in self._spatial_grid:
                    self._spatial_grid[grid_key] = []
                self._spatial_grid[grid_key].append(led.led_id)

            logger.debug(
                f"Built spatial index with {len(self._spatial_grid)} grid cells"
            )

        except Exception as e:
            logger.error(f"Failed to build spatial index: {e}")

    def get_led_position(self, led_id: int) -> Optional[LEDPosition]:
        """
        Get LED position by ID.

        Args:
            led_id: LED identifier

        Returns:
            LEDPosition object or None if not found
        """
        return self._position_lookup.get(led_id)

    def get_all_positions(self) -> List[LEDPosition]:
        """
        Get all LED positions.

        Returns:
            List of all LED positions
        """
        return self.led_positions.copy()

    def get_positions_in_region(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> List[LEDPosition]:
        """
        Get LED positions within a rectangular region.

        Args:
            x_min, y_min, x_max, y_max: Region bounds in normalized coordinates

        Returns:
            List of LED positions within the region
        """
        positions = []

        for led in self.led_positions:
            if x_min <= led.x <= x_max and y_min <= led.y <= y_max:
                positions.append(led)

        return positions

    def get_nearest_leds(
        self, x: float, y: float, count: int = 1, max_distance: float = 1.0
    ) -> List[Tuple[LEDPosition, float]]:
        """
        Find nearest LEDs to a given position.

        Args:
            x, y: Target position in normalized coordinates
            count: Number of nearest LEDs to return
            max_distance: Maximum search distance

        Returns:
            List of (LEDPosition, distance) tuples, sorted by distance
        """
        distances = []

        for led in self.led_positions:
            dx = led.x - x
            dy = led.y - y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance <= max_distance:
                distances.append((led, distance))

        # Sort by distance and return top count
        distances.sort(key=lambda item: item[1])
        return distances[:count]

    def sample_frame_at_leds(self, frame_array: np.ndarray) -> np.ndarray:
        """
        Sample frame colors at LED positions.

        Args:
            frame_array: Frame data array (height, width, channels)

        Returns:
            Array of RGB colors for each LED (led_count, 3)
        """
        try:
            if frame_array.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
                raise ValueError(
                    f"Frame shape {frame_array.shape[:2]} doesn't match "
                    f"expected {(FRAME_HEIGHT, FRAME_WIDTH)}"
                )

            colors = np.zeros((len(self.led_positions), 3), dtype=np.uint8)

            for i, led in enumerate(self.led_positions):
                # Sample color at LED pixel position
                colors[i] = frame_array[led.pixel_y, led.pixel_x, :3]

            return colors

        except Exception as e:
            logger.error(f"Failed to sample frame at LEDs: {e}")
            return np.zeros((len(self.led_positions), 3), dtype=np.uint8)

    def get_led_count(self) -> int:
        """Get total number of LEDs."""
        return len(self.led_positions)

    def get_position_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get LED positions as numpy arrays.

        Returns:
            Tuple of (x_positions, y_positions) arrays
        """
        x_positions = np.array([led.x for led in self.led_positions])
        y_positions = np.array([led.y for led in self.led_positions])
        return x_positions, y_positions

    def get_pixel_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get LED pixel positions as numpy arrays.

        Returns:
            Tuple of (pixel_x, pixel_y) arrays
        """
        pixel_x = np.array([led.pixel_x for led in self.led_positions])
        pixel_y = np.array([led.pixel_y for led in self.led_positions])
        return pixel_x, pixel_y

    def validate_positions(self) -> bool:
        """
        Validate LED positions for consistency.

        Returns:
            True if all positions are valid, False otherwise
        """
        try:
            if len(self.led_positions) != LED_COUNT:
                logger.error(
                    f"Expected {LED_COUNT} LEDs, got {len(self.led_positions)}"
                )
                return False

            for led in self.led_positions:
                if not (0.0 <= led.x <= 1.0 and 0.0 <= led.y <= 1.0):
                    logger.error(
                        f"LED {led.led_id} position out of bounds: ({led.x}, {led.y})"
                    )
                    return False

                if not (
                    0 <= led.pixel_x < FRAME_WIDTH and 0 <= led.pixel_y < FRAME_HEIGHT
                ):
                    logger.error(
                        f"LED {led.led_id} pixel position out of bounds: "
                        f"({led.pixel_x}, {led.pixel_y})"
                    )
                    return False

            # Check for duplicate IDs
            led_ids = [led.led_id for led in self.led_positions]
            if len(set(led_ids)) != len(led_ids):
                logger.error("Duplicate LED IDs found")
                return False

            logger.info("LED position validation passed")
            return True

        except Exception as e:
            logger.error(f"LED position validation failed: {e}")
            return False

    def get_mapper_stats(self) -> Dict[str, Any]:
        """
        Get LED mapper statistics.

        Returns:
            Dictionary with mapper statistics
        """
        if not self.led_positions:
            return {"led_count": 0, "status": "uninitialized"}

        x_positions, y_positions = self.get_position_arrays()

        return {
            "led_count": len(self.led_positions),
            "frame_dimensions": (FRAME_WIDTH, FRAME_HEIGHT),
            "position_bounds": {
                "x_min": float(np.min(x_positions)),
                "x_max": float(np.max(x_positions)),
                "y_min": float(np.min(y_positions)),
                "y_max": float(np.max(y_positions)),
            },
            "spatial_grid_cells": len(self._spatial_grid) if self._spatial_grid else 0,
            "config_path": self.config_path,
            "status": "initialized",
        }
