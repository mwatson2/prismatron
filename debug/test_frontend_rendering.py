#!/usr/bin/env python3
"""
Frontend Rendering Test Tool.

This tool connects to the WebSocket API and renders LED data using the exact same
approach as the frontend HomePage.jsx component. This helps diagnose rendering
issues by comparing Python and JavaScript rendering results.

The tool replicates:
1. Canvas size: 800x480 pixels
2. LED circle radius: 8 pixels
3. Glow effect radius: 12 pixels for bright LEDs (any channel > 200)
4. Additive blending using PIL composite operations
5. Coordinate scaling from frame dimensions to canvas size
6. Dark LED skipping (RGB 0,0,0)

Usage:
    python test_frontend_rendering.py --patterns patterns.npz --api-url http://localhost:8000
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import requests
import websockets

try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Error: PIL not available - this tool requires Pillow")
    sys.exit(1)

logger = logging.getLogger(__name__)


class FrontendRenderingTester:
    """Test tool that replicates the exact frontend rendering approach."""

    def __init__(self, patterns_file: str, api_url: str = "http://localhost:8080"):
        """
        Initialize the tester.

        Args:
            patterns_file: Path to diffusion patterns NPZ file
            api_url: Base URL for the web API
        """
        self.patterns_file = patterns_file
        self.api_url = api_url.rstrip("/")
        self.websocket_url = api_url.replace("http", "ws") + "/ws"

        # Output directory
        self.output_dir = Path("/tmp/prismatron_frontend_test")
        self.output_dir.mkdir(exist_ok=True)

        # LED data from API and patterns
        self.api_led_positions: Optional[List] = None
        self.api_frame_dimensions: Optional[Dict] = None

        # Canvas settings (match frontend exactly)
        self.canvas_width = 800
        self.canvas_height = 480
        self.led_radius = 8  # Main LED circle radius
        self.glow_radius = 12  # Glow effect radius for bright LEDs
        self.bright_threshold = 200  # Channel value threshold for glow effect

        # Frame tracking
        self.frames_rendered = 0
        self.target_frames = 10  # Render 10 different frames
        self.previous_led_values: Optional[List] = None
        self.frame_hashes: Set[str] = set()

        logger.info(f"Initialized frontend rendering tester: patterns={patterns_file}, api={api_url}")
        logger.info(f"Canvas: {self.canvas_width}x{self.canvas_height}, LED radius: {self.led_radius}px")

    async def run_test_session(self) -> None:
        """Run the complete frontend rendering test session."""
        logger.info("Starting frontend rendering test session")

        # Step 1: Try to retrieve LED positions from API, fallback to patterns file
        try:
            await self.retrieve_api_led_positions()
        except Exception as e:
            logger.warning(f"API connection failed: {e}")
            logger.info("Falling back to loading LED positions from patterns file")
            self.load_led_positions_from_patterns()

        # Step 2: Connect to websocket and render LED frames (or use test data if API unavailable)
        try:
            await self.test_websocket_rendering()
        except Exception as e:
            logger.warning(f"WebSocket connection failed: {e}")
            logger.info("Generating test frames with synthetic LED data")
            self.generate_test_frames()

        logger.info("Frontend rendering test session complete")

    async def retrieve_api_led_positions(self) -> None:
        """Retrieve LED positions from the web API (exactly like frontend)."""
        logger.info("Retrieving LED positions from web API")

        try:
            response = requests.get(f"{self.api_url}/api/led-positions", timeout=10)
            response.raise_for_status()

            data = response.json()
            logger.info(f"API Response keys: {list(data.keys())}")

            # Extract LED positions and frame dimensions (matching frontend)
            self.api_led_positions = data.get("positions", [])
            self.api_frame_dimensions = data.get("frame_dimensions", {})

            logger.info(f"Retrieved {len(self.api_led_positions)} LED positions from API")
            logger.info(f"Frame dimensions: {self.api_frame_dimensions}")

            # Save API data for inspection
            api_data_file = self.output_dir / "api_led_positions.json"
            with open(api_data_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved API LED positions to {api_data_file}")

        except Exception as e:
            logger.error(f"Failed to retrieve LED positions from API: {e}")
            raise

    async def test_websocket_rendering(self) -> None:
        """Connect to websocket and render LED frames (matching frontend)."""
        logger.info(f"Connecting to websocket: {self.websocket_url}")

        try:
            async with websockets.connect(self.websocket_url) as websocket:
                logger.info("Connected to websocket")

                # Process messages until we have rendered target number of frames
                while self.frames_rendered < self.target_frames:
                    try:
                        # Receive message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        data = json.loads(message)

                        # Process the message
                        await self.process_websocket_message(data)

                    except asyncio.TimeoutError:
                        logger.warning("Websocket receive timeout - continuing...")
                        continue
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode websocket message: {e}")
                        continue

                logger.info(f"Successfully rendered {self.frames_rendered} different images")

        except Exception as e:
            logger.error(f"Websocket connection failed: {e}")
            raise

    async def process_websocket_message(self, data: Dict) -> None:
        """Process a websocket message and render if it contains LED data."""
        try:
            # Look for LED frame data in the message (matching frontend logic)
            if not (data.get("has_frame", False) and "frame_data" in data):
                return  # Not a frame data message, skip

            frame_data = data["frame_data"]
            if not frame_data or len(frame_data) == 0:
                logger.debug("No LED data in message")
                return

            # Check if this frame is different from the previous one
            if not self.is_frame_different(frame_data):
                logger.debug("Frame is identical to previous frame, skipping")
                return

            logger.info(f"Processing LED frame: {len(frame_data)} LEDs")

            # Render using frontend style technique
            rendered_image = self.render_frontend_style(frame_data)

            if rendered_image is not None:
                # Save the rendered image
                frame_file = self.output_dir / f"frontend_render_{self.frames_rendered:03d}.png"
                rendered_image.save(frame_file)
                logger.info(f"Saved frontend-style render {self.frames_rendered} to {frame_file}")

                # Save the LED values for inspection
                led_file = self.output_dir / f"led_values_{self.frames_rendered:03d}.json"
                with open(led_file, "w") as f:
                    json.dump(frame_data, f, indent=2)
                logger.info(f"Saved LED values {self.frames_rendered} to {led_file}")

                self.frames_rendered += 1

                # Update previous frame data
                self.previous_led_values = frame_data

        except Exception as e:
            logger.error(f"Failed to process websocket message: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def is_frame_different(self, frame_data: List) -> bool:
        """Check if the current frame is different from the previous one."""
        if self.previous_led_values is None:
            return True

        # Create a simple hash of the frame data to check for differences
        frame_str = str(frame_data)
        frame_hash = hash(frame_str)

        if frame_hash in self.frame_hashes:
            return False

        self.frame_hashes.add(frame_hash)
        return True

    def render_frontend_style(self, frame_data: List) -> Optional[Image.Image]:
        """
        Render LED values exactly like the frontend HomePage.jsx.

        This replicates the exact JavaScript canvas rendering logic:
        1. Create black 800x480 canvas
        2. Set additive blending mode
        3. Calculate scaling from frame dimensions to canvas
        4. For each LED position:
           - Scale coordinates to canvas size
           - Skip dark LEDs (RGB 0,0,0)
           - Draw 8px radius circle with LED color
           - Add 12px glow effect for bright LEDs (any channel > 200)

        Args:
            frame_data: LED values as received from websocket

        Returns:
            PIL Image with frontend-style rendering, or None if failed
        """
        try:
            if not self.api_led_positions or not self.api_frame_dimensions:
                logger.warning("Missing LED positions or frame dimensions - cannot render")
                return None

            # Get frame dimensions (matching frontend)
            frame_width = self.api_frame_dimensions.get("width", 800)
            frame_height = self.api_frame_dimensions.get("height", 480)

            # Calculate scaling to fit LED coordinate space into canvas (matching frontend)
            scale_x = self.canvas_width / frame_width
            scale_y = self.canvas_height / frame_height

            logger.debug(
                f"Canvas rendering: {self.canvas_width}x{self.canvas_height}, " f"scale: {scale_x:.3f}x{scale_y:.3f}"
            )

            # Create canvas with black background (matching frontend)
            canvas = Image.new("RGB", (self.canvas_width, self.canvas_height), (0, 0, 0))

            # We'll implement additive blending manually since PIL doesn't have 'lighter' mode
            # Create a working array for additive blending
            canvas_array = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.float32)

            leds_drawn = 0

            # Draw each LED (matching frontend forEach loop)
            for i, position in enumerate(self.api_led_positions):
                if i >= len(frame_data):
                    break

                x, y = position

                # Scale coordinates to canvas size (matching frontend)
                canvas_x = x * scale_x
                canvas_y = y * scale_y

                # Log first few LEDs for debugging (matching frontend)
                if i < 5:
                    logger.debug(f"LED {i}: raw pos [{x}, {y}] -> canvas [{canvas_x:.1f}, {canvas_y:.1f}]")

                # Get LED color data (matching frontend)
                color_data = frame_data[i]
                if not (color_data and isinstance(color_data, list) and len(color_data) >= 3):
                    continue

                r, g, b = color_data[:3]

                # Skip completely dark LEDs (matching frontend)
                if r == 0 and g == 0 and b == 0:
                    continue

                # Debug log for first few LEDs (matching frontend)
                if i < 5 and (r > 0 or g > 0 or b > 0):
                    logger.debug(f"LED {i} color: rgb({r}, {g}, {b})")

                # Draw LED circle with additive blending (matching frontend arc drawing)
                self._add_circle_to_canvas(canvas_array, int(canvas_x), int(canvas_y), self.led_radius, r, g, b)

                # Add glow effect for bright LEDs (matching frontend glow logic)
                if r > self.bright_threshold or g > self.bright_threshold or b > self.bright_threshold:
                    # Glow with 30% opacity (matching frontend rgba(r,g,b,0.3))
                    glow_r, glow_g, glow_b = r * 0.3, g * 0.3, b * 0.3
                    self._add_circle_to_canvas(
                        canvas_array, int(canvas_x), int(canvas_y), self.glow_radius, glow_r, glow_g, glow_b
                    )

                leds_drawn += 1

            # Convert additive array back to PIL Image
            # Clamp values to [0, 255] and convert to uint8
            canvas_array = np.clip(canvas_array, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(canvas_array)

            logger.debug(f"Frontend-style rendering complete: drew {leds_drawn} LED circles")
            return canvas

        except Exception as e:
            logger.error(f"Failed to render frontend style: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def _add_circle_to_canvas(
        self, canvas: np.ndarray, cx: int, cy: int, radius: int, r: float, g: float, b: float
    ) -> None:
        """
        Add a colored circle to the canvas using additive blending.

        This replicates the JavaScript canvas arc() and fill() operations.
        """
        # Calculate bounds
        y1 = max(0, cy - radius)
        y2 = min(canvas.shape[0], cy + radius + 1)
        x1 = max(0, cx - radius)
        x2 = min(canvas.shape[1], cx + radius + 1)

        # Create circle mask
        for y in range(y1, y2):
            for x in range(x1, x2):
                # Check if pixel is inside circle
                dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                if dist_sq <= radius**2:
                    # Add color values (additive blending)
                    canvas[y, x, 0] += r
                    canvas[y, x, 1] += g
                    canvas[y, x, 2] += b

    def load_led_positions_from_patterns(self) -> None:
        """Load LED positions directly from patterns file when API is unavailable."""
        logger.info(f"Loading LED positions from patterns file: {self.patterns_file}")

        try:
            data = np.load(self.patterns_file, allow_pickle=True)

            # Load LED positions (should be in physical order)
            if "led_positions" in data:
                led_positions_array = data["led_positions"]
                # Convert numpy array to list of lists for compatibility with API format
                self.api_led_positions = led_positions_array.tolist()
                logger.info(f"Loaded LED positions from patterns: {len(self.api_led_positions)} LEDs")
            else:
                logger.error("No LED positions found in patterns file")
                return

            # Set default frame dimensions
            self.api_frame_dimensions = {"width": 800, "height": 480}

            # Save patterns data for inspection
            patterns_data_file = self.output_dir / "patterns_led_positions.json"
            with open(patterns_data_file, "w") as f:
                json.dump(
                    {
                        "positions": self.api_led_positions,
                        "frame_dimensions": self.api_frame_dimensions,
                        "led_count": len(self.api_led_positions),
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved patterns LED positions to {patterns_data_file}")

        except Exception as e:
            logger.error(f"Failed to load LED positions from patterns: {e}")
            raise

    def generate_test_frames(self) -> None:
        """Generate test frames with synthetic LED data when WebSocket is unavailable."""
        logger.info("Generating test frames with synthetic LED data")

        if not self.api_led_positions:
            logger.error("No LED positions available for test frame generation")
            return

        led_count = len(self.api_led_positions)

        # Generate several test frames with different patterns
        test_patterns = [
            self._generate_rainbow_pattern(led_count),
            self._generate_gradient_pattern(led_count),
            self._generate_random_pattern(led_count),
            self._generate_hot_spots_pattern(led_count),
            self._generate_strobe_pattern(led_count),
        ]

        for pattern_idx, frame_data in enumerate(test_patterns):
            logger.info(f"Rendering test pattern {pattern_idx + 1}/{len(test_patterns)}")

            # Render using frontend style technique
            rendered_image = self.render_frontend_style(frame_data)

            if rendered_image is not None:
                # Save the rendered image
                frame_file = self.output_dir / f"test_pattern_{pattern_idx:03d}.png"
                rendered_image.save(frame_file)
                logger.info(f"Saved test pattern {pattern_idx} to {frame_file}")

                # Save the LED values for inspection
                led_file = self.output_dir / f"test_led_values_{pattern_idx:03d}.json"
                with open(led_file, "w") as f:
                    json.dump(frame_data, f, indent=2)

                self.frames_rendered += 1

    def _generate_rainbow_pattern(self, led_count: int) -> List:
        """Generate a rainbow color pattern across LEDs."""
        frame_data = []
        for i in range(led_count):
            # Simple HSV to RGB rainbow
            hue = (i / led_count) * 360
            saturation = 1.0
            value = 1.0

            # Convert HSV to RGB
            c = value * saturation
            x = c * (1 - abs(((hue / 60) % 2) - 1))
            m = value - c

            if 0 <= hue < 60:
                r, g, b = c, x, 0
            elif 60 <= hue < 120:
                r, g, b = x, c, 0
            elif 120 <= hue < 180:
                r, g, b = 0, c, x
            elif 180 <= hue < 240:
                r, g, b = 0, x, c
            elif 240 <= hue < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)

            frame_data.append([r, g, b])

        return frame_data

    def _generate_gradient_pattern(self, led_count: int) -> List:
        """Generate a red-to-blue gradient pattern."""
        frame_data = []
        for i in range(led_count):
            ratio = i / max(1, led_count - 1)
            r = int(255 * (1 - ratio))
            g = 0
            b = int(255 * ratio)
            frame_data.append([r, g, b])

        return frame_data

    def _generate_random_pattern(self, led_count: int) -> List:
        """Generate a random color pattern."""
        np.random.seed(42)  # Reproducible random pattern
        frame_data = []
        for i in range(led_count):
            r = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            b = np.random.randint(0, 256)
            frame_data.append([r, g, b])

        return frame_data

    def _generate_hot_spots_pattern(self, led_count: int) -> List:
        """Generate a pattern with a few bright hot spots."""
        frame_data = [[0, 0, 0] for _ in range(led_count)]

        # Add some bright spots
        hot_spots = [
            (led_count // 4, [255, 0, 0]),  # Red
            (led_count // 2, [0, 255, 0]),  # Green
            (3 * led_count // 4, [0, 0, 255]),  # Blue
            (led_count // 8, [255, 255, 0]),  # Yellow
            (7 * led_count // 8, [255, 0, 255]),  # Magenta
        ]

        for led_idx, color in hot_spots:
            if led_idx < led_count:
                frame_data[led_idx] = color

        return frame_data

    def _generate_strobe_pattern(self, led_count: int) -> List:
        """Generate a strobe pattern with alternating bright/dark."""
        frame_data = []
        for i in range(led_count):
            if i % 3 == 0:
                frame_data.append([255, 255, 255])  # Bright white
            else:
                frame_data.append([0, 0, 0])  # Dark

        return frame_data

    def generate_report(self) -> None:
        """Generate a summary report of the frontend rendering test."""
        report_file = self.output_dir / "frontend_rendering_report.txt"

        with open(report_file, "w") as f:
            f.write("Prismatron Frontend Rendering Test Report\n")
            f.write("=" * 45 + "\n\n")

            f.write(f"Patterns file: {self.patterns_file}\n")
            f.write(f"API URL: {self.api_url}\n")
            f.write(f"WebSocket URL: {self.websocket_url}\n\n")

            # Canvas settings
            f.write("Canvas Settings (matching frontend):\n")
            f.write(f"  Canvas size: {self.canvas_width}x{self.canvas_height}px\n")
            f.write(f"  LED circle radius: {self.led_radius}px\n")
            f.write(f"  Glow effect radius: {self.glow_radius}px\n")
            f.write(f"  Bright LED threshold: {self.bright_threshold}\n\n")

            # LED position data
            f.write("LED Position Data:\n")
            if self.api_led_positions is not None:
                f.write(f"  API LED positions: {len(self.api_led_positions)} LEDs\n")
            else:
                f.write("  API LED positions: Not loaded\n")

            f.write(f"  Frame dimensions: {self.api_frame_dimensions}\n\n")

            # Rendering results
            f.write("Rendering Results:\n")
            f.write(f"  Target frames: {self.target_frames}\n")
            f.write(f"  Frames rendered: {self.frames_rendered}\n")
            f.write(f"  Unique frame hashes: {len(self.frame_hashes)}\n\n")

            # Generated files
            f.write("Generated Files:\n")
            for file in sorted(self.output_dir.glob("*")):
                if file.is_file() and file.name != "frontend_rendering_report.txt":
                    f.write(f"  {file.name}\n")

        logger.info(f"Generated frontend rendering report: {report_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Test frontend LED rendering approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default API URL
  python test_frontend_rendering.py --patterns diffusion_patterns/synthetic_2624_uint8_fp32.npz

  # Test with custom API URL
  python test_frontend_rendering.py --patterns patterns.npz --api-url http://localhost:8000

  # Enable verbose logging
  python test_frontend_rendering.py --patterns patterns.npz --verbose
        """,
    )

    parser.add_argument("--patterns", required=True, type=Path, help="Diffusion patterns NPZ file")
    parser.add_argument(
        "--api-url", default="http://localhost:8080", help="Base URL for the web API (default: http://localhost:8080)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if not args.patterns.exists():
        logger.error(f"Patterns file not found: {args.patterns}")
        sys.exit(1)

    try:
        # Create tester and run session
        tester = FrontendRenderingTester(str(args.patterns), args.api_url)
        await tester.run_test_session()
        tester.generate_report()

        logger.info("Frontend rendering test complete!")
        logger.info(f"Check output files in: {tester.output_dir}")

    except Exception as e:
        logger.error(f"Frontend rendering test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
