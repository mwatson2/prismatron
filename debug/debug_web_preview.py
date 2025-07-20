#!/usr/bin/env python3
"""
Debug Web Preview Tool.

This tool helps debug the web interface preview rendering issue by:
1. Retrieving LED positions from the web API
2. Comparing LED positions from API with diffusion patterns file
3. Receiving LED values from web API websocket
4. Rendering LED values using the _render_preview_style technique
5. Continuing until 10 different images are rendered

Usage:
    python debug_web_preview.py --patterns patterns.npz --api-url http://localhost:8000
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
    print("Warning: PIL not available - image output will be limited")

logger = logging.getLogger(__name__)


class WebPreviewDebugger:
    """Debug tool for web interface preview rendering issues."""

    def __init__(self, patterns_file: str, api_url: str = "http://localhost:8000"):
        """
        Initialize the debugger.

        Args:
            patterns_file: Path to diffusion patterns NPZ file
            api_url: Base URL for the web API
        """
        self.patterns_file = patterns_file
        self.api_url = api_url.rstrip("/")
        self.websocket_url = api_url.replace("http", "ws") + "/ws"

        # Output directory
        self.output_dir = Path("/tmp/prismatron_web_debug")
        self.output_dir.mkdir(exist_ok=True)

        # LED data from API and patterns
        self.api_led_positions: Optional[np.ndarray] = None
        self.api_frame_dimensions: Optional[Dict] = None
        self.patterns_led_positions: Optional[np.ndarray] = None
        self.patterns_led_ordering: Optional[np.ndarray] = None

        # Frame tracking
        self.frames_rendered = 0
        self.target_frames = 4  # Capture more frames
        self.previous_led_values: Optional[List] = None
        self.frame_hashes: Set[str] = set()

        # LED values collection for comparison
        self.collected_api_led_values: List[np.ndarray] = []  # Store LED values from API

        logger.info(f"Initialized debugger: patterns={patterns_file}, api={api_url}")

    async def run_debug_session(self) -> None:
        """Run the complete debugging session."""
        logger.info("Starting web preview debugging session")

        # Step 1: Retrieve LED positions from API
        await self.retrieve_api_led_positions()

        # Step 2: Load and compare with patterns file
        self.load_patterns_led_positions()
        self.compare_led_positions()

        # Step 3: Connect to websocket and receive LED values
        await self.debug_websocket_led_values()

        # Step 4: Compare API LED values with logged LED files
        self.compare_api_with_logged_leds()

        logger.info("Web preview debugging session complete")

    async def retrieve_api_led_positions(self) -> None:
        """Retrieve LED positions from the web API."""
        logger.info("Retrieving LED positions from web API")

        try:
            response = requests.get(f"{self.api_url}/api/led-positions", timeout=10)
            response.raise_for_status()

            data = response.json()
            logger.info(f"API Response keys: {list(data.keys())}")

            # Extract LED positions and frame dimensions
            positions = data.get("positions", [])
            self.api_led_positions = np.array(positions) if positions else None
            self.api_frame_dimensions = data.get("frame_dimensions", {})

            logger.info(f"Retrieved {len(positions)} LED positions from API")
            logger.info(f"Frame dimensions: {self.api_frame_dimensions}")
            logger.info(f"Debug info: {data.get('debug', {})}")

            # Save API data for inspection
            api_data_file = self.output_dir / "api_led_positions.json"
            with open(api_data_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved API LED positions to {api_data_file}")

        except Exception as e:
            logger.error(f"Failed to retrieve LED positions from API: {e}")
            raise

    def load_patterns_led_positions(self) -> None:
        """Load LED positions and ordering from patterns file."""
        logger.info(f"Loading LED positions from patterns file: {self.patterns_file}")

        try:
            data = np.load(self.patterns_file, allow_pickle=True)

            # Load LED positions (in physical order)
            if "led_positions" in data:
                self.patterns_led_positions = data["led_positions"]
                logger.info(f"Loaded LED positions from patterns: {self.patterns_led_positions.shape}")
            else:
                logger.warning("No LED positions found in patterns file")

            # Load LED ordering (spatial_index -> physical_led_id)
            if "led_ordering" in data:
                self.patterns_led_ordering = data["led_ordering"]
                logger.info(f"Loaded LED ordering from patterns: {self.patterns_led_ordering.shape}")
            else:
                logger.warning("No LED ordering found in patterns file")

            # Save patterns data for inspection
            if self.patterns_led_positions is not None:
                patterns_csv = self.output_dir / "patterns_led_positions.csv"
                np.savetxt(
                    patterns_csv, self.patterns_led_positions, delimiter=",", header="x,y", comments="", fmt="%.6f"
                )
                logger.info(f"Saved patterns LED positions to {patterns_csv}")

            if self.patterns_led_ordering is not None:
                ordering_csv = self.output_dir / "patterns_led_ordering.csv"
                np.savetxt(
                    ordering_csv,
                    self.patterns_led_ordering,
                    delimiter=",",
                    header="physical_led_id",
                    comments="",
                    fmt="%d",
                )
                logger.info(f"Saved patterns LED ordering to {ordering_csv}")

        except Exception as e:
            logger.error(f"Failed to load patterns LED positions: {e}")
            raise

    def compare_led_positions(self) -> None:
        """Compare LED positions from API and patterns file."""
        logger.info("Comparing LED positions from API and patterns file")

        if self.api_led_positions is None:
            logger.error("No API LED positions to compare")
            return

        if self.patterns_led_positions is None:
            logger.error("No patterns LED positions to compare")
            return

        # Compare shapes
        api_shape = self.api_led_positions.shape
        patterns_shape = self.patterns_led_positions.shape

        logger.info(f"API LED positions shape: {api_shape}")
        logger.info(f"Patterns LED positions shape: {patterns_shape}")

        if api_shape != patterns_shape:
            logger.warning(f"Shape mismatch: API {api_shape} vs Patterns {patterns_shape}")
        else:
            logger.info("LED position shapes match")

        # Compare actual values (for overlapping range)
        min_count = min(len(self.api_led_positions), len(self.patterns_led_positions))

        if min_count > 0:
            api_subset = self.api_led_positions[:min_count]
            patterns_subset = self.patterns_led_positions[:min_count]

            # Calculate differences
            diff = np.abs(api_subset - patterns_subset)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            logger.info(f"Position comparison (first {min_count} LEDs):")
            logger.info(f"  Max difference: {max_diff:.6f}")
            logger.info(f"  Mean difference: {mean_diff:.6f}")

            # Save comparison data
            comparison_file = self.output_dir / "led_positions_comparison.csv"
            comparison_data = np.column_stack(
                [
                    api_subset[:, 0],
                    api_subset[:, 1],
                    patterns_subset[:, 0],
                    patterns_subset[:, 1],
                    diff[:, 0],
                    diff[:, 1],
                ]
            )
            header = "api_x,api_y,patterns_x,patterns_y,diff_x,diff_y"
            np.savetxt(comparison_file, comparison_data, delimiter=",", header=header, comments="", fmt="%.6f")
            logger.info(f"Saved position comparison to {comparison_file}")

            # Check if positions are essentially identical
            if max_diff < 1e-6:
                logger.info("LED positions are essentially identical")
            elif max_diff < 1.0:
                logger.warning(f"LED positions have small differences (max: {max_diff:.6f})")
            else:
                logger.error(f"LED positions have significant differences (max: {max_diff:.6f})")

    async def debug_websocket_led_values(self) -> None:
        """Connect to websocket and receive LED values for debugging."""
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
            # Look for LED frame data in the message
            frame_data = None
            total_leds = None

            # Check if this is a preview data message
            if "frame_data" in data and data.get("has_frame", False):
                frame_data = data["frame_data"]
                total_leds = data.get("total_leds", len(frame_data) if frame_data else 0)
                logger.debug(f"Received LED frame data: {total_leds} LEDs")
            else:
                # Not a frame data message, skip
                return

            if frame_data is None or len(frame_data) == 0:
                logger.debug("No LED data in message")
                return

            # Check if this frame is different from the previous one
            if not self.is_frame_different(frame_data):
                logger.debug("Frame is identical to previous frame, skipping")
                return

            # Convert to numpy array for rendering
            led_values = np.array(frame_data, dtype=np.uint8)
            logger.info(
                f"Processing LED frame: shape={led_values.shape}, range=[{led_values.min()}, {led_values.max()}]"
            )

            # Store LED values for comparison
            self.collected_api_led_values.append(led_values.copy())

            # Render using preview style technique
            rendered_image = self.render_preview_style(led_values)

            if rendered_image is not None:
                # Save the rendered image
                frame_file = self.output_dir / f"rendered_frame_{self.frames_rendered:03d}.png"
                rendered_image.save(frame_file)
                logger.info(f"Saved rendered frame {self.frames_rendered} to {frame_file}")

                # Save the LED values for inspection
                led_file = self.output_dir / f"led_values_{self.frames_rendered:03d}.csv"
                np.savetxt(led_file, led_values, delimiter=",", fmt="%d", header="R,G,B", comments="")
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

    def render_preview_style(self, led_values: np.ndarray) -> Optional[Image.Image]:
        """
        Render LED values as circles at spatial positions with additive color blending.

        This uses additive blending to avoid dark LEDs overwriting bright colors beneath.

        Args:
            led_values: LED values in spatial order, shape (led_count, 3), range [0, 255]

        Returns:
            PIL Image with LED values rendered as additive circles, or None if failed
        """
        try:
            if not PIL_AVAILABLE:
                logger.warning("PIL not available - cannot render preview style")
                return None

            if self.api_led_positions is None or self.patterns_led_ordering is None:
                logger.warning("Missing LED positions or ordering - cannot render preview")
                return None

            # Get frame dimensions
            frame_width = self.api_frame_dimensions.get("width", 800)
            frame_height = self.api_frame_dimensions.get("height", 480)

            # Create numpy array for additive rendering (float for precision)
            canvas_array = np.zeros((frame_height, frame_width, 3), dtype=np.float32)

            # Circle radius (32-pixel diameter = 16-pixel radius)
            radius = 16

            logger.debug(f"Rendering {len(led_values)} LEDs as additive circles on {frame_width}x{frame_height} canvas")

            # Pre-compute circle mask for efficiency
            circle_mask = self._create_circle_mask(radius)

            # LED values are in spatial order, positions are in physical order
            # We need to map from spatial index to physical LED position
            leds_drawn = 0
            for spatial_idx, (r, g, b) in enumerate(led_values):
                # Skip dark LEDs to avoid unnecessary computation
                if r == 0 and g == 0 and b == 0:
                    continue

                # Skip if this spatial index doesn't exist in ordering
                if spatial_idx >= len(self.patterns_led_ordering):
                    continue

                # Get physical LED ID for this spatial index
                physical_led_id = self.patterns_led_ordering[spatial_idx]

                # Skip if physical LED ID is out of range
                if physical_led_id >= len(self.api_led_positions):
                    continue

                # Get position for this physical LED
                x, y = self.api_led_positions[physical_led_id]

                # Convert to integer coordinates
                x, y = int(round(x)), int(round(y))

                # Skip if position is outside frame bounds
                if x < radius or y < radius or x >= frame_width - radius or y >= frame_height - radius:
                    continue

                # Apply additive blending using the circle mask
                self._add_circle_to_canvas(canvas_array, x, y, r, g, b, circle_mask)
                leds_drawn += 1

            # Convert to PIL Image
            # Clamp values to [0, 255] and convert to uint8
            canvas_array = np.clip(canvas_array, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(canvas_array)

            logger.debug(f"Additive rendering complete: drew {leds_drawn} LED circles")
            return canvas

        except Exception as e:
            logger.error(f"Failed to render preview style: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def _create_circle_mask(self, radius: int) -> np.ndarray:
        """Create a circular mask for additive blending."""
        size = 2 * radius + 1
        y, x = np.ogrid[:size, :size]
        center = radius
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
        return mask.astype(np.float32)

    def _add_circle_to_canvas(
        self, canvas: np.ndarray, cx: int, cy: int, r: int, g: int, b: int, mask: np.ndarray
    ) -> None:
        """Add a colored circle to the canvas using additive blending."""
        radius = mask.shape[0] // 2

        # Calculate bounds
        y1 = max(0, cy - radius)
        y2 = min(canvas.shape[0], cy + radius + 1)
        x1 = max(0, cx - radius)
        x2 = min(canvas.shape[1], cx + radius + 1)

        # Calculate mask bounds
        mask_y1 = max(0, radius - cy)
        mask_y2 = mask_y1 + (y2 - y1)
        mask_x1 = max(0, radius - cx)
        mask_x2 = mask_x1 + (x2 - x1)

        # Get the relevant portion of the mask
        circle_portion = mask[mask_y1:mask_y2, mask_x1:mask_x2]

        if circle_portion.size == 0:
            return

        # Apply additive blending
        canvas[y1:y2, x1:x2, 0] += circle_portion * r
        canvas[y1:y2, x1:x2, 1] += circle_portion * g
        canvas[y1:y2, x1:x2, 2] += circle_portion * b

    def compare_api_with_logged_leds(self) -> None:
        """Compare LED values from API with logged LED files from frame_renderer."""
        logger.info("Comparing API LED values with logged LED files")

        # Path to frame_renderer debug LED files
        debug_led_dir = Path("/tmp/prismatron_debug_leds")

        if not debug_led_dir.exists():
            logger.warning(f"Debug LED directory not found: {debug_led_dir}")
            return

        # Find all led_physical_*.npy and led_spatial_*.npy files
        physical_files = list(debug_led_dir.glob("led_physical_*.npy"))
        spatial_files = list(debug_led_dir.glob("led_spatial_*.npy"))

        if not physical_files and not spatial_files:
            logger.warning(f"No led_physical_*.npy or led_spatial_*.npy files found in {debug_led_dir}")
            return

        logger.info(f"Found {len(physical_files)} physical LED files: {[f.name for f in physical_files]}")
        logger.info(f"Found {len(spatial_files)} spatial LED files: {[f.name for f in spatial_files]}")
        logger.info(f"Collected {len(self.collected_api_led_values)} LED values from API")

        # Try to match each API LED value with logged LED files
        matches_found = 0
        comparison_results = []

        for api_idx, api_led_values in enumerate(self.collected_api_led_values):
            logger.info(f"Comparing API LED values {api_idx} with logged files...")

            best_match_file = None
            best_match_diff = float("inf")
            best_match_type = None

            # Try matching with spatial files (need to convert API values from physical to spatial order)
            for led_file in spatial_files:
                try:
                    # Load logged LED values (should be in spatial order)
                    logged_led_values = np.load(led_file)

                    # API LED values are in physical order, convert to spatial order for comparison
                    if self.patterns_led_ordering is not None:
                        # Convert API values from physical to spatial order
                        api_spatial_values = np.zeros_like(api_led_values)
                        api_spatial_values[np.arange(len(api_led_values))] = api_led_values[self.patterns_led_ordering]
                        api_comparison_values = api_spatial_values
                    else:
                        logger.warning("No LED ordering available, skipping spatial comparison")
                        continue

                    # Compare shapes
                    if api_comparison_values.shape != logged_led_values.shape:
                        logger.debug(
                            f"Shape mismatch with {led_file.name}: API {api_comparison_values.shape} vs logged {logged_led_values.shape}"
                        )
                        continue

                    # Calculate difference
                    diff = np.abs(api_comparison_values.astype(np.float32) - logged_led_values.astype(np.float32))
                    mean_diff = np.mean(diff)
                    max_diff = np.max(diff)

                    logger.debug(f"  {led_file.name} (spatial): mean_diff={mean_diff:.3f}, max_diff={max_diff:.3f}")

                    # Check if this is a better match
                    if mean_diff < best_match_diff:
                        best_match_diff = mean_diff
                        best_match_file = led_file
                        best_match_type = "spatial"

                except Exception as e:
                    logger.warning(f"Error comparing with {led_file.name}: {e}")
                    continue

            # Try matching with physical files (API values are already in physical order)
            for led_file in physical_files:
                try:
                    # Load logged LED values (should be in physical order)
                    logged_led_values = np.load(led_file)

                    # API LED values are already in physical order, compare directly
                    api_comparison_values = api_led_values

                    # Compare shapes
                    if api_comparison_values.shape != logged_led_values.shape:
                        logger.debug(
                            f"Shape mismatch with {led_file.name}: API {api_comparison_values.shape} vs logged {logged_led_values.shape}"
                        )
                        continue

                    # Calculate difference
                    diff = np.abs(api_comparison_values.astype(np.float32) - logged_led_values.astype(np.float32))
                    mean_diff = np.mean(diff)
                    max_diff = np.max(diff)

                    logger.debug(f"  {led_file.name} (physical): mean_diff={mean_diff:.3f}, max_diff={max_diff:.3f}")

                    # Check if this is a better match
                    if mean_diff < best_match_diff:
                        best_match_diff = mean_diff
                        best_match_file = led_file
                        best_match_type = "physical"

                except Exception as e:
                    logger.warning(f"Error comparing with {led_file.name}: {e}")
                    continue

            # Record the best match
            if best_match_file is not None:
                result = {
                    "api_index": api_idx,
                    "best_match_file": best_match_file.name,
                    "match_type": best_match_type,
                    "mean_difference": best_match_diff,
                    "is_exact_match": best_match_diff < 1e-6,
                    "is_close_match": best_match_diff < 1.0,
                }
                comparison_results.append(result)

                if result["is_exact_match"]:
                    matches_found += 1
                    logger.info(
                        f"  ✅ EXACT MATCH: API LED {api_idx} matches {best_match_file.name} ({best_match_type})"
                    )
                elif result["is_close_match"]:
                    logger.info(
                        f"  ⚠️  CLOSE MATCH: API LED {api_idx} matches {best_match_file.name} ({best_match_type}) (diff={best_match_diff:.3f})"
                    )
                else:
                    logger.warning(
                        f"  ❌ NO MATCH: API LED {api_idx} closest to {best_match_file.name} ({best_match_type}) (diff={best_match_diff:.3f})"
                    )
            else:
                logger.warning(f"  ❌ NO MATCH: API LED {api_idx} has no comparable logged file")

        # Save comparison results
        comparison_file = self.output_dir / "led_values_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison_results, f, indent=2)
        logger.info(f"Saved LED values comparison to {comparison_file}")

        # Summary
        logger.info(f"LED Values Comparison Summary:")
        logger.info(f"  API LED values collected: {len(self.collected_api_led_values)}")
        logger.info(f"  Logged spatial LED files found: {len(spatial_files)}")
        logger.info(f"  Logged physical LED files found: {len(physical_files)}")
        logger.info(f"  Exact matches found: {matches_found}/{len(self.collected_api_led_values)}")

        # Count matches by type
        spatial_matches = sum(
            1 for r in comparison_results if r.get("match_type") == "spatial" and r.get("is_exact_match")
        )
        physical_matches = sum(
            1 for r in comparison_results if r.get("match_type") == "physical" and r.get("is_exact_match")
        )

        if spatial_matches > 0:
            logger.info(f"  Spatial matches: {spatial_matches}")
        if physical_matches > 0:
            logger.info(f"  Physical matches: {physical_matches}")

        if matches_found == len(self.collected_api_led_values) and len(self.collected_api_led_values) > 0:
            logger.info("✅ All API LED values have exact matches in logged files!")
        elif matches_found > 0:
            logger.info(f"⚠️  Partial matches: {matches_found}/{len(self.collected_api_led_values)} exact matches")
        else:
            logger.warning("❌ No exact matches found between API and logged LED values")

    def generate_report(self) -> None:
        """Generate a summary report of the debugging session."""
        report_file = self.output_dir / "web_debug_report.txt"

        with open(report_file, "w") as f:
            f.write("Prismatron Web Preview Debug Report\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Patterns file: {self.patterns_file}\n")
            f.write(f"API URL: {self.api_url}\n")
            f.write(f"WebSocket URL: {self.websocket_url}\n\n")

            # LED position comparison
            f.write("LED Position Analysis:\n")
            if self.api_led_positions is not None:
                f.write(f"  API LED positions: {self.api_led_positions.shape}\n")
            else:
                f.write("  API LED positions: Not loaded\n")

            if self.patterns_led_positions is not None:
                f.write(f"  Patterns LED positions: {self.patterns_led_positions.shape}\n")
            else:
                f.write("  Patterns LED positions: Not loaded\n")

            if self.patterns_led_ordering is not None:
                f.write(f"  Patterns LED ordering: {self.patterns_led_ordering.shape}\n")
            else:
                f.write("  Patterns LED ordering: Not loaded\n")

            f.write(f"\nFrame dimensions: {self.api_frame_dimensions}\n")

            # Rendering results
            f.write(f"\nRendering Results:\n")
            f.write(f"  Target frames: {self.target_frames}\n")
            f.write(f"  Frames rendered: {self.frames_rendered}\n")
            f.write(f"  Unique frame hashes: {len(self.frame_hashes)}\n")
            f.write(f"  API LED values collected: {len(self.collected_api_led_values)}\n")

            # Generated files
            f.write(f"\nGenerated Files:\n")
            for file in sorted(self.output_dir.glob("*")):
                if file.is_file() and file.name != "web_debug_report.txt":
                    f.write(f"  {file.name}\n")

        logger.info(f"Generated debug report: {report_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Debug web interface preview rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Debug with default API URL
  python debug_web_preview.py --patterns diffusion_patterns/synthetic_2624_uint8_fp32.npz

  # Debug with custom API URL
  python debug_web_preview.py --patterns patterns.npz --api-url http://localhost:8000

  # Enable verbose logging
  python debug_web_preview.py --patterns patterns.npz --verbose
        """,
    )

    parser.add_argument("--patterns", required=True, type=Path, help="Diffusion patterns NPZ file")
    parser.add_argument(
        "--api-url", default="http://localhost:8000", help="Base URL for the web API (default: http://localhost:8000)"
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
        # Create debugger and run session
        debugger = WebPreviewDebugger(str(args.patterns), args.api_url)
        await debugger.run_debug_session()
        debugger.generate_report()

        logger.info("Web preview debugging complete!")
        logger.info(f"Check output files in: {debugger.output_dir}")

    except Exception as e:
        logger.error(f"Debug session failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
