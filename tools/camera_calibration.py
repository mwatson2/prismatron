#!/usr/bin/env python3
"""
Camera Calibration Tool.

This tool helps calibrate the camera for diffusion pattern capture by:
1. Showing live camera feed
2. Allowing user to select a 5:3 aspect ratio region
3. Providing visual feedback and guides
4. Saving calibration parameters for use with capture tool

Usage:
    python camera_calibration.py --camera-device 0 --output-config calibration.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from const import FRAME_HEIGHT, FRAME_WIDTH

logger = logging.getLogger(__name__)


class CameraCalibration:
    """Interactive camera calibration tool."""

    def __init__(self, camera_device: int = 0):
        """
        Initialize camera calibration.

        Args:
            camera_device: Camera device ID
        """
        self.camera_device = camera_device
        self.cap: Optional[cv2.VideoCapture] = None

        # Camera properties
        self.camera_width = 0
        self.camera_height = 0

        # Calibration state
        self.selection_mode = False
        self.selection_start: Optional[Tuple[int, int]] = None
        self.selection_end: Optional[Tuple[int, int]] = None
        self.crop_region: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h

        # Target aspect ratio (5:3)
        self.target_aspect = FRAME_WIDTH / FRAME_HEIGHT

        # UI state
        self.show_help = True
        self.show_grid = True
        self.show_guides = True

    def initialize(self) -> bool:
        """Initialize camera connection."""
        try:
            # Use working GStreamer pipeline for NVIDIA Jetson cameras
            gstreamer_pipeline = (
                f"nvarguscamerasrc sensor-id={self.camera_device} ! " "nvvidconv ! video/x-raw, format=BGR ! appsink"
            )

            logger.info(f"Using GStreamer pipeline: {gstreamer_pipeline}")
            self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

            if not self.cap.isOpened():
                logger.error("GStreamer pipeline failed to open")
                logger.error("Make sure nvarguscamerasrc is available and camera is not in use")
                return False

            logger.info("GStreamer pipeline opened successfully")

            # Get camera resolution
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Camera initialized: {self.camera_width}x{self.camera_height}")

            # Set up mouse callback
            cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Camera Calibration", self._mouse_callback)

            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection_start = (x, y)
            self.selection_mode = True
            self.selection_end = None

        elif event == cv2.EVENT_MOUSEMOVE and self.selection_mode:
            self.selection_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.selection_mode:
            self.selection_end = (x, y)
            self.selection_mode = False
            self._finalize_selection()

    def _finalize_selection(self):
        """Finalize the region selection and enforce 5:3 aspect ratio."""
        if not self.selection_start or not self.selection_end:
            return

        x1, y1 = self.selection_start
        x2, y2 = self.selection_end

        # Ensure top-left and bottom-right ordering
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        if w < 50 or h < 50:  # Minimum size
            logger.warning("Selection too small, ignoring")
            return

        # Enforce 5:3 aspect ratio
        current_aspect = w / h

        if current_aspect > self.target_aspect:
            # Selection is too wide, reduce width
            new_w = int(h * self.target_aspect)
            x_offset = (w - new_w) // 2
            x += x_offset
            w = new_w
        else:
            # Selection is too tall, reduce height
            new_h = int(w / self.target_aspect)
            y_offset = (h - new_h) // 2
            y += y_offset
            h = new_h

        # Ensure bounds are within camera frame
        x = max(0, min(x, self.camera_width - w))
        y = max(0, min(y, self.camera_height - h))
        w = min(w, self.camera_width - x)
        h = min(h, self.camera_height - y)

        self.crop_region = (x, y, w, h)
        logger.info(f"Crop region set: {self.crop_region} (aspect ratio: {w / h:.3f})")

    def _draw_guides(self, frame: np.ndarray) -> np.ndarray:
        """Draw guides and overlays on the frame."""
        overlay = frame.copy()

        # Draw grid if enabled
        if self.show_grid:
            self._draw_grid(overlay)

        # Draw aspect ratio guides if enabled
        if self.show_guides:
            self._draw_aspect_guides(overlay)

        # Draw current selection
        if self.selection_mode and self.selection_start and self.selection_end:
            self._draw_current_selection(overlay)

        # Draw finalized crop region
        if self.crop_region:
            self._draw_crop_region(overlay)

        # Draw help text
        if self.show_help:
            self._draw_help_text(overlay)

        return overlay

    def _draw_grid(self, frame: np.ndarray):
        """Draw grid lines for alignment."""
        h, w = frame.shape[:2]

        # Vertical lines (rule of thirds)
        for i in range(1, 3):
            x = w * i // 3
            cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)

        # Horizontal lines (rule of thirds)
        for i in range(1, 3):
            y = h * i // 3
            cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)

    def _draw_aspect_guides(self, frame: np.ndarray):
        """Draw 5:3 aspect ratio guide rectangles."""
        h, w = frame.shape[:2]

        # Calculate maximum 5:3 rectangle that fits in the frame
        if w / h > self.target_aspect:
            # Frame is wider than 5:3, fit by height
            guide_h = h
            guide_w = int(h * self.target_aspect)
        else:
            # Frame is taller than 5:3, fit by width
            guide_w = w
            guide_h = int(w / self.target_aspect)

        # Center the guide rectangle
        guide_x = (w - guide_w) // 2
        guide_y = (h - guide_h) // 2

        # Draw guide rectangle
        cv2.rectangle(
            frame,
            (guide_x, guide_y),
            (guide_x + guide_w, guide_y + guide_h),
            (0, 255, 255),  # Yellow
            2,
        )

        # Add label
        cv2.putText(
            frame,
            "5:3 Guide",
            (guide_x, guide_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    def _draw_current_selection(self, frame: np.ndarray):
        """Draw the current selection being made."""
        if not self.selection_start or not self.selection_end:
            return

        x1, y1 = self.selection_start
        x2, y2 = self.selection_end

        # Draw selection rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show dimensions and aspect ratio
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if h > 0:
            aspect = w / h
            text = f"{w}x{h} (aspect: {aspect:.3f})"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    def _draw_crop_region(self, frame: np.ndarray):
        """Draw the finalized crop region."""
        if not self.crop_region:
            return

        x, y, w, h = self.crop_region

        # Draw crop rectangle with thick border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Add corner markers
        corner_size = 20
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

        for corner_x, corner_y in corners:
            cv2.circle(frame, (corner_x, corner_y), corner_size // 2, (0, 0, 255), -1)
            cv2.circle(frame, (corner_x, corner_y), corner_size // 2, (255, 255, 255), 2)

        # Add crop region info
        aspect = w / h
        info_text = f"Crop: {w}x{h} (aspect: {aspect:.3f})"
        cv2.putText(frame, info_text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show preview of cropped region in corner
        self._draw_crop_preview(frame)

    def _draw_crop_preview(self, frame: np.ndarray):
        """Draw a small preview of the cropped region."""
        if not self.crop_region:
            return

        x, y, w, h = self.crop_region

        # Extract crop region
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            return

        # Scale preview to fit in corner
        preview_size = 150
        preview_aspect = w / h

        if preview_aspect > 1:
            preview_w = preview_size
            preview_h = int(preview_size / preview_aspect)
        else:
            preview_h = preview_size
            preview_w = int(preview_size * preview_aspect)

        preview = cv2.resize(crop, (preview_w, preview_h))

        # Position in top-right corner
        frame_h, frame_w = frame.shape[:2]
        preview_x = frame_w - preview_w - 10
        preview_y = 10

        # Add preview with border
        cv2.rectangle(
            frame,
            (preview_x - 2, preview_y - 2),
            (preview_x + preview_w + 2, preview_y + preview_h + 2),
            (255, 255, 255),
            2,
        )

        frame[preview_y : preview_y + preview_h, preview_x : preview_x + preview_w] = preview

        # Add label
        cv2.putText(
            frame,
            "Preview",
            (preview_x, preview_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

    def _draw_help_text(self, frame: np.ndarray):
        """Draw help text overlay."""
        help_lines = [
            "CAMERA CALIBRATION",
            "",
            "Controls:",
            "  Click & drag: Select crop region",
            "  g: Toggle grid",
            "  a: Toggle aspect guides",
            "  h: Toggle help",
            "  r: Reset selection",
            "  s: Save configuration",
            "  q: Quit",
            "",
            f"Target: {FRAME_WIDTH}x{FRAME_HEIGHT} (5:3)",
        ]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 320), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Help text
        for i, line in enumerate(help_lines):
            y = 30 + i * 20
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            thickness = 2 if i == 0 else 1

            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    def run_calibration(self) -> Optional[Dict]:
        """Run the interactive calibration process."""
        if not self.cap:
            logger.error("Camera not initialized")
            return None

        logger.info("Starting camera calibration - press 'h' for help")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break

                # Draw guides and overlays
                display_frame = self._draw_guides(frame)

                # Show frame
                cv2.imshow("Camera Calibration", display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    logger.info("Calibration quit by user")
                    break

                elif key == ord("h"):
                    self.show_help = not self.show_help

                elif key == ord("g"):
                    self.show_grid = not self.show_grid
                    logger.info(f"Grid {'enabled' if self.show_grid else 'disabled'}")

                elif key == ord("a"):
                    self.show_guides = not self.show_guides
                    logger.info(f"Aspect guides {'enabled' if self.show_guides else 'disabled'}")

                elif key == ord("r"):
                    self.crop_region = None
                    self.selection_start = None
                    self.selection_end = None
                    logger.info("Selection reset")

                elif key == ord("s"):
                    if self.crop_region:
                        return self._create_calibration_config()
                    else:
                        logger.warning("No crop region selected")

            return None

        except KeyboardInterrupt:
            logger.info("Calibration interrupted")
            return None

        finally:
            cv2.destroyAllWindows()

    def _create_calibration_config(self) -> Dict:
        """Create calibration configuration."""
        if not self.crop_region:
            return {}

        x, y, w, h = self.crop_region

        config = {
            "camera_device": self.camera_device,
            "camera_resolution": {
                "width": self.camera_width,
                "height": self.camera_height,
            },
            "crop_region": {"x": x, "y": y, "width": w, "height": h},
            "target_resolution": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
            "aspect_ratio": {"target": self.target_aspect, "actual": w / h},
            "calibration_timestamp": time.time(),
        }

        return config

    def cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Camera calibration for diffusion pattern capture")
    parser.add_argument("--camera-device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--output-config", help="Output configuration file path (.json)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create calibration tool
    calibration = CameraCalibration(camera_device=args.camera_device)

    try:
        # Initialize camera
        if not calibration.initialize():
            logger.error("Failed to initialize camera")
            return 1

        # Run calibration
        config = calibration.run_calibration()

        if config:
            # Save configuration if requested
            if args.output_config:
                output_path = Path(args.output_config)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w") as f:
                    json.dump(config, f, indent=2)

                logger.info(f"Calibration saved to {output_path}")

            # Print configuration
            print("\nCalibration Configuration:")
            print(json.dumps(config, indent=2))

            return 0
        else:
            logger.info("Calibration cancelled")
            return 1

    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return 1

    finally:
        calibration.cleanup()


if __name__ == "__main__":
    sys.exit(main())
