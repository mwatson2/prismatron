"""
Unit tests for the LED Position Mapping System.

Tests LED position generation, spatial indexing, coordinate transformations,
and frame sampling functionality.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from src.consumer.led_mapper import LEDMapper, LEDPosition


class TestLEDPosition(unittest.TestCase):
    """Test cases for LEDPosition class."""

    def test_initialization(self):
        """Test LED position initialization."""
        led = LEDPosition(led_id=0, x=0.5, y=0.3)

        self.assertEqual(led.led_id, 0)
        self.assertEqual(led.x, 0.5)
        self.assertEqual(led.y, 0.3)
        self.assertEqual(led.pixel_x, int(0.5 * FRAME_WIDTH))
        self.assertEqual(led.pixel_y, int(0.3 * FRAME_HEIGHT))

    def test_pixel_coordinate_calculation(self):
        """Test pixel coordinate calculation from normalized coordinates."""
        led = LEDPosition(led_id=1, x=0.0, y=0.0)
        self.assertEqual(led.pixel_x, 0)
        self.assertEqual(led.pixel_y, 0)

        led = LEDPosition(led_id=2, x=1.0, y=1.0)
        self.assertEqual(led.pixel_x, FRAME_WIDTH - 1)  # Clamped to bounds
        self.assertEqual(led.pixel_y, FRAME_HEIGHT - 1)

    def test_distance_calculation(self):
        """Test distance calculation between LED positions."""
        led1 = LEDPosition(led_id=0, x=0.0, y=0.0)
        led2 = LEDPosition(led_id=1, x=0.3, y=0.4)

        distance = led1.distance_to(led2)
        expected = 0.5  # 3-4-5 triangle
        self.assertAlmostEqual(distance, expected, places=5)

    def test_serialization(self):
        """Test LED position serialization and deserialization."""
        original = LEDPosition(
            led_id=42,
            x=0.7,
            y=0.2,
            physical_x=100.0,
            physical_y=50.0,
            calibration_data={"brightness": 0.8},
        )

        # Test to_dict
        data = original.to_dict()
        self.assertEqual(data["led_id"], 42)
        self.assertEqual(data["x"], 0.7)
        self.assertEqual(data["calibration_data"]["brightness"], 0.8)

        # Test from_dict
        restored = LEDPosition.from_dict(data)
        self.assertEqual(restored.led_id, original.led_id)
        self.assertEqual(restored.x, original.x)
        self.assertEqual(restored.y, original.y)
        self.assertEqual(restored.physical_x, original.physical_x)
        self.assertEqual(restored.calibration_data, original.calibration_data)


class TestLEDMapper(unittest.TestCase):
    """Test cases for LEDMapper class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_led_positions.json")
        self.mapper = LEDMapper(config_path=self.config_path)

    def tearDown(self):
        """Clean up after tests."""
        # Clean up temp files
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        os.rmdir(self.temp_dir)

    def test_initialization_with_generated_positions(self):
        """Test mapper initialization with generated positions."""
        result = self.mapper.initialize()

        self.assertTrue(result)
        self.assertEqual(len(self.mapper.led_positions), LED_COUNT)
        self.assertTrue(os.path.exists(self.config_path))

    def test_random_position_generation(self):
        """Test random LED position generation."""
        result = self.mapper.generate_random_positions(seed=123)

        self.assertTrue(result)
        self.assertEqual(len(self.mapper.led_positions), LED_COUNT)

        # Test that positions are within bounds
        for led in self.mapper.led_positions:
            self.assertGreaterEqual(led.x, 0.0)
            self.assertLessEqual(led.x, 1.0)
            self.assertGreaterEqual(led.y, 0.0)
            self.assertLessEqual(led.y, 1.0)

    def test_position_determinism(self):
        """Test that same seed produces same positions."""
        mapper1 = LEDMapper()
        mapper1.generate_random_positions(seed=42)

        mapper2 = LEDMapper()
        mapper2.generate_random_positions(seed=42)

        # Compare first few positions
        for i in range(min(10, len(mapper1.led_positions))):
            led1 = mapper1.led_positions[i]
            led2 = mapper2.led_positions[i]
            self.assertAlmostEqual(led1.x, led2.x, places=10)
            self.assertAlmostEqual(led1.y, led2.y, places=10)

    def test_save_and_load_positions(self):
        """Test saving and loading LED positions."""
        # Generate and save positions
        self.mapper.generate_random_positions(seed=456)
        original_positions = self.mapper.led_positions.copy()

        result = self.mapper.save_led_positions()
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.config_path))

        # Create new mapper and load positions
        new_mapper = LEDMapper(config_path=self.config_path)
        result = new_mapper.load_led_positions()

        self.assertTrue(result)
        self.assertEqual(len(new_mapper.led_positions), len(original_positions))

        # Compare positions
        for orig, loaded in zip(original_positions, new_mapper.led_positions):
            self.assertEqual(orig.led_id, loaded.led_id)
            self.assertAlmostEqual(orig.x, loaded.x, places=10)
            self.assertAlmostEqual(orig.y, loaded.y, places=10)

    def test_position_lookup(self):
        """Test LED position lookup by ID."""
        self.mapper.generate_random_positions(seed=789)

        # Test valid lookup
        led = self.mapper.get_led_position(100)
        self.assertIsNotNone(led)
        self.assertEqual(led.led_id, 100)

        # Test invalid lookup
        led = self.mapper.get_led_position(-1)
        self.assertIsNone(led)

        led = self.mapper.get_led_position(LED_COUNT + 1)
        self.assertIsNone(led)

    def test_region_queries(self):
        """Test getting LEDs within a region."""
        self.mapper.generate_random_positions(seed=101)

        # Query a small region
        region_leds = self.mapper.get_positions_in_region(0.4, 0.4, 0.6, 0.6)

        # All returned LEDs should be within the region
        for led in region_leds:
            self.assertGreaterEqual(led.x, 0.4)
            self.assertLessEqual(led.x, 0.6)
            self.assertGreaterEqual(led.y, 0.4)
            self.assertLessEqual(led.y, 0.6)

    def test_nearest_led_search(self):
        """Test finding nearest LEDs to a position."""
        self.mapper.generate_random_positions(seed=202)

        # Find nearest LEDs to center
        nearest = self.mapper.get_nearest_leds(0.5, 0.5, count=5)

        self.assertLessEqual(len(nearest), 5)

        # Check that results are sorted by distance
        distances = [distance for _, distance in nearest]
        self.assertEqual(distances, sorted(distances))

        # Check that all distances are reasonable
        for led, distance in nearest:
            expected_distance = led.distance_to(LEDPosition(0, 0.5, 0.5))
            self.assertAlmostEqual(distance, expected_distance, places=5)

    def test_frame_sampling(self):
        """Test sampling frame colors at LED positions."""
        self.mapper.generate_random_positions(seed=303)

        # Create test frame with gradient
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        for y in range(FRAME_HEIGHT):
            for x in range(FRAME_WIDTH):
                frame[y, x] = [x % 256, y % 256, 128]

        # Sample colors at LED positions
        colors = self.mapper.sample_frame_at_leds(frame)

        self.assertEqual(colors.shape, (LED_COUNT, 3))
        self.assertEqual(colors.dtype, np.uint8)

        # Verify some sampled colors match expected values
        for i, led in enumerate(self.mapper.led_positions[:10]):  # Check first 10
            expected_color = frame[led.pixel_y, led.pixel_x]
            np.testing.assert_array_equal(colors[i], expected_color)

    def test_frame_sampling_wrong_shape(self):
        """Test frame sampling with wrong frame shape."""
        self.mapper.generate_random_positions(seed=404)

        # Create frame with wrong dimensions
        wrong_frame = np.zeros((100, 200, 3), dtype=np.uint8)

        # Should return zeros for safety
        colors = self.mapper.sample_frame_at_leds(wrong_frame)

        self.assertEqual(colors.shape, (LED_COUNT, 3))
        np.testing.assert_array_equal(colors, np.zeros((LED_COUNT, 3), dtype=np.uint8))

    def test_position_arrays(self):
        """Test getting positions as numpy arrays."""
        self.mapper.generate_random_positions(seed=505)

        x_positions, y_positions = self.mapper.get_position_arrays()
        pixel_x, pixel_y = self.mapper.get_pixel_arrays()

        self.assertEqual(len(x_positions), LED_COUNT)
        self.assertEqual(len(y_positions), LED_COUNT)
        self.assertEqual(len(pixel_x), LED_COUNT)
        self.assertEqual(len(pixel_y), LED_COUNT)

        # Verify consistency with individual LED positions
        for i, led in enumerate(self.mapper.led_positions):
            self.assertEqual(x_positions[i], led.x)
            self.assertEqual(y_positions[i], led.y)
            self.assertEqual(pixel_x[i], led.pixel_x)
            self.assertEqual(pixel_y[i], led.pixel_y)

    def test_position_validation(self):
        """Test LED position validation."""
        self.mapper.generate_random_positions(seed=606)

        # Valid positions should pass
        result = self.mapper.validate_positions()
        self.assertTrue(result)

        # Test invalid position (out of bounds)
        self.mapper.led_positions[0].x = -0.1
        result = self.mapper.validate_positions()
        self.assertFalse(result)

        # Fix and test duplicate ID
        self.mapper.led_positions[0].x = 0.5
        self.mapper.led_positions[1].led_id = self.mapper.led_positions[0].led_id
        result = self.mapper.validate_positions()
        self.assertFalse(result)

    def test_mapper_statistics(self):
        """Test mapper statistics generation."""
        # Test uninitialized mapper
        stats = self.mapper.get_mapper_stats()
        self.assertEqual(stats["led_count"], 0)
        self.assertEqual(stats["status"], "uninitialized")

        # Test initialized mapper
        self.mapper.generate_random_positions(seed=707)
        stats = self.mapper.get_mapper_stats()

        self.assertEqual(stats["led_count"], LED_COUNT)
        self.assertEqual(stats["status"], "initialized")
        self.assertEqual(stats["frame_dimensions"], (FRAME_WIDTH, FRAME_HEIGHT))
        self.assertIn("position_bounds", stats)
        self.assertIn("x_min", stats["position_bounds"])
        self.assertIn("spatial_grid_cells", stats)

    def test_spatial_index_building(self):
        """Test spatial index construction."""
        self.mapper.generate_random_positions(seed=808)
        self.mapper._build_spatial_index()

        self.assertIsNotNone(self.mapper._spatial_grid)
        self.assertGreater(len(self.mapper._spatial_grid), 0)

    def test_config_file_format(self):
        """Test configuration file format."""
        self.mapper.generate_random_positions(seed=909)
        self.mapper.save_led_positions()

        # Load and verify JSON structure
        with open(self.config_path, "r") as f:
            data = json.load(f)

        self.assertIn("version", data)
        self.assertIn("led_count", data)
        self.assertIn("frame_width", data)
        self.assertIn("frame_height", data)
        self.assertIn("led_positions", data)

        self.assertEqual(data["led_count"], LED_COUNT)
        self.assertEqual(data["frame_width"], FRAME_WIDTH)
        self.assertEqual(data["frame_height"], FRAME_HEIGHT)
        self.assertEqual(len(data["led_positions"]), LED_COUNT)

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        nonexistent_path = "/tmp/nonexistent_led_config.json"
        mapper = LEDMapper(config_path=nonexistent_path)

        result = mapper.load_led_positions()
        self.assertFalse(result)

    def test_load_invalid_json(self):
        """Test loading from invalid JSON file."""
        # Create invalid JSON file
        with open(self.config_path, "w") as f:
            f.write("{ invalid json")

        result = self.mapper.load_led_positions()
        self.assertFalse(result)

    def test_initialization_from_existing_config(self):
        """Test initialization when config file already exists."""
        # First create and save a configuration
        self.mapper.generate_random_positions(seed=1010)
        self.mapper.save_led_positions()

        # Create new mapper and initialize from existing config
        new_mapper = LEDMapper(config_path=self.config_path)
        result = new_mapper.initialize()

        self.assertTrue(result)
        self.assertEqual(len(new_mapper.led_positions), LED_COUNT)


if __name__ == "__main__":
    unittest.main()
