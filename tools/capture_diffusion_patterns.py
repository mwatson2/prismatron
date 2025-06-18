#!/usr/bin/env python3
"""
Diffusion Pattern Capture Tool.

This tool captures the diffusion patterns for each LED and color channel by:
1. Connecting to WLED controller
2. Setting each LED/channel to full brightness
3. Capturing camera image (800x480)
4. Storing the patterns in a numpy array
5. Saving the complete diffusion pattern dataset

Usage:
    python capture_diffusion_patterns.py --wled-host 192.168.1.100 --camera-device 0 --output patterns.npz --preview
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from consumer.wled_client import WLEDClient, WLEDConfig

logger = logging.getLogger(__name__)


class CameraCapture:
    """Handles camera capture with proper configuration."""

    def __init__(
        self,
        device_id: int = 0,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Initialize camera capture.

        Args:
            device_id: Camera device ID (usually 0 for default camera)
            crop_region: Optional crop region (x, y, width, height) for prismatron area
        """
        self.device_id = device_id
        self.crop_region = crop_region
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_width = 0
        self.camera_height = 0

    def initialize(self) -> bool:
        """Initialize camera and configure settings."""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera device {self.device_id}")
                return False

            # Get camera resolution
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Camera initialized: {self.camera_width}x{self.camera_height}")

            # Set camera properties for consistent capture
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Lower exposure for LED capture
            self.cap.set(cv2.CAP_PROP_GAIN, 1)  # Minimal gain
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            self.cap.set(cv2.CAP_PROP_CONTRAST, 1.0)

            # Warm up camera
            for _ in range(5):
                ret, _ = self.cap.read()
                if not ret:
                    logger.error("Failed to read from camera during warmup")
                    return False

            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame and process it.

        Returns:
            Processed frame as 800x480 RGB array, or None if failed
        """
        if not self.cap:
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                return None

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply crop region if specified
            if self.crop_region:
                x, y, w, h = self.crop_region
                frame = frame[y : y + h, x : x + w]

            # Scale to target resolution (800x480)
            frame = cv2.resize(
                frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR
            )

            return frame

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None

    def cleanup(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            self.cap = None


class DiffusionPatternCapture:
    """Main capture tool for diffusion patterns."""

    def __init__(
        self,
        wled_host: str,
        wled_port: int = 21324,
        camera_device: int = 0,
        capture_fps: float = 10.0,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Initialize diffusion pattern capture.

        Args:
            wled_host: WLED controller hostname/IP
            wled_port: WLED controller port
            camera_device: Camera device ID
            capture_fps: Target capture rate (captures per second)
            crop_region: Optional crop region for camera
        """
        self.wled_host = wled_host
        self.wled_port = wled_port
        self.capture_fps = capture_fps
        self.capture_interval = 1.0 / capture_fps

        # Initialize WLED client
        wled_config = WLEDConfig(
            host=wled_host, port=wled_port, led_count=LED_COUNT, max_fps=60.0
        )
        self.wled_client = WLEDClient(wled_config)

        # Initialize camera
        self.camera = CameraCapture(camera_device, crop_region)

        # Storage for diffusion patterns
        # Shape: (LED_COUNT, 3 channels, FRAME_HEIGHT, FRAME_WIDTH)
        # Using uint8 to save memory: 3200×3×480×800×1 = ~3.5GB vs 14GB for float32
        self.diffusion_patterns = np.zeros(
            (LED_COUNT, 3, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8
        )

    def initialize(self) -> bool:
        """Initialize WLED and camera connections."""
        # Connect to WLED
        if not self.wled_client.connect():
            logger.error("Failed to connect to WLED controller")
            return False

        logger.info(f"Connected to WLED at {self.wled_host}:{self.wled_port}")

        # Initialize camera
        if not self.camera.initialize():
            logger.error("Failed to initialize camera")
            return False

        return True

    def capture_patterns(self, preview: bool = False) -> bool:
        """
        Capture diffusion patterns for all LEDs and channels.

        Args:
            preview: Show live preview during capture

        Returns:
            True if capture successful
        """
        try:
            total_captures = LED_COUNT * 3  # 3 channels per LED
            logger.info(f"Starting capture of {total_captures} diffusion patterns")

            # Turn off all LEDs initially
            self.wled_client.set_solid_color(0, 0, 0)
            time.sleep(0.5)  # Allow LEDs to turn off

            for led_idx in range(LED_COUNT):
                for channel_idx in range(3):  # R, G, B channels
                    capture_num = led_idx * 3 + channel_idx + 1

                    logger.info(
                        f"Capturing LED {led_idx}, Channel {channel_idx} ({capture_num}/{total_captures})"
                    )

                    # Create LED data array (all off except current LED/channel)
                    led_data = np.zeros((LED_COUNT, 3), dtype=np.uint8)
                    led_data[
                        led_idx, channel_idx
                    ] = 255  # Full brightness for this LED/channel

                    # Send to WLED
                    result = self.wled_client.send_led_data(led_data)
                    if not result.success:
                        logger.warning(f"Failed to send LED data: {result.errors}")
                        continue

                    # Wait for LED to update and stabilize
                    time.sleep(self.capture_interval)

                    # Capture frame
                    frame = self.camera.capture_frame()
                    if frame is None:
                        logger.warning(
                            f"Failed to capture frame for LED {led_idx}, channel {channel_idx}"
                        )
                        continue

                    # Store diffusion pattern (keep as uint8 to save memory)
                    self.diffusion_patterns[led_idx, channel_idx] = frame.astype(
                        np.uint8
                    )

                    # Show preview if requested
                    if preview:
                        self._show_preview(
                            frame, led_idx, channel_idx, capture_num, total_captures
                        )

                    # Progress update
                    if capture_num % 100 == 0:
                        progress = (capture_num / total_captures) * 100
                        logger.info(
                            f"Progress: {progress:.1f}% ({capture_num}/{total_captures})"
                        )

            # Turn off all LEDs
            self.wled_client.set_solid_color(0, 0, 0)

            logger.info("Diffusion pattern capture completed successfully")
            return True

        except KeyboardInterrupt:
            logger.info("Capture interrupted by user")
            self.wled_client.set_solid_color(0, 0, 0)  # Turn off LEDs
            return False

        except Exception as e:
            logger.error(f"Capture failed: {e}")
            self.wled_client.set_solid_color(0, 0, 0)  # Turn off LEDs
            return False

    def _show_preview(
        self,
        frame: np.ndarray,
        led_idx: int,
        channel_idx: int,
        capture_num: int,
        total_captures: int,
    ):
        """Show live preview of capture."""
        try:
            # Convert back to BGR for OpenCV display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Add text overlay
            channel_names = ["Red", "Green", "Blue"]
            text = f"LED {led_idx} {channel_names[channel_idx]} ({capture_num}/{total_captures})"
            cv2.putText(
                display_frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Resize for display if needed
            display_frame = cv2.resize(display_frame, (800, 480))

            cv2.imshow("Diffusion Pattern Capture", display_frame)

            # Allow user to quit with 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("User requested quit")

        except Exception as e:
            logger.warning(f"Preview display failed: {e}")

    def save_patterns(self, output_path: str) -> bool:
        """
        Save captured diffusion patterns to file.

        Args:
            output_path: Path to save diffusion patterns (.npz format)

        Returns:
            True if save successful
        """
        try:
            # Create metadata
            metadata = {
                "led_count": LED_COUNT,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "channels": 3,
                "capture_fps": self.capture_fps,
                "wled_host": self.wled_host,
                "wled_port": self.wled_port,
                "capture_timestamp": time.time(),
                "data_shape": self.diffusion_patterns.shape,
                "data_dtype": str(self.diffusion_patterns.dtype),
            }

            # Save patterns and metadata
            np.savez_compressed(
                output_path,
                diffusion_patterns=self.diffusion_patterns,
                metadata=metadata,
            )

            file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(
                f"Diffusion patterns saved to {output_path} ({file_size_mb:.1f} MB)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
            return False

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, "wled_client"):
            self.wled_client.set_solid_color(0, 0, 0)  # Turn off LEDs
            self.wled_client.disconnect()

        if hasattr(self, "camera"):
            self.camera.cleanup()

        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Capture LED diffusion patterns")
    parser.add_argument(
        "--wled-host", required=True, help="WLED controller hostname/IP"
    )
    parser.add_argument(
        "--wled-port", type=int, default=21324, help="WLED controller port"
    )
    parser.add_argument("--camera-device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--output", required=True, help="Output file path (.npz)")
    parser.add_argument(
        "--capture-fps", type=float, default=10.0, help="Capture rate (fps)"
    )
    parser.add_argument("--preview", action="store_true", help="Show live preview")
    parser.add_argument(
        "--crop-region",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Camera crop region (x y width height)",
    )
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
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"capture_diffusion_{int(time.time())}.log"),
        ],
    )

    # Validate output path
    output_path = Path(args.output)
    if output_path.suffix != ".npz":
        logger.error("Output file must have .npz extension")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create crop region tuple if provided
    crop_region = None
    if args.crop_region:
        crop_region = tuple(args.crop_region)

    # Create capture tool
    capture_tool = DiffusionPatternCapture(
        wled_host=args.wled_host,
        wled_port=args.wled_port,
        camera_device=args.camera_device,
        capture_fps=args.capture_fps,
        crop_region=crop_region,
    )

    try:
        # Initialize
        if not capture_tool.initialize():
            logger.error("Failed to initialize capture tool")
            return 1

        # Estimate capture time
        total_captures = LED_COUNT * 3
        estimated_time_minutes = (total_captures * (1.0 / args.capture_fps)) / 60
        logger.info(f"Estimated capture time: {estimated_time_minutes:.1f} minutes")

        # Start capture
        if not capture_tool.capture_patterns(preview=args.preview):
            logger.error("Capture failed")
            return 1

        # Save patterns
        if not capture_tool.save_patterns(str(output_path)):
            logger.error("Failed to save patterns")
            return 1

        logger.info("Diffusion pattern capture completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Capture interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    finally:
        capture_tool.cleanup()


if __name__ == "__main__":
    sys.exit(main())
