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
import scipy.sparse as sp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
    from consumer.wled_client import WLEDClient, WLEDConfig
except ImportError:
    # Fallback to hardcoded constants for testing
    FRAME_HEIGHT = 480
    FRAME_WIDTH = 800
    LED_COUNT = 3200

    # Create mock classes for testing
    class WLEDConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class WLEDClient:
        def __init__(self, config):
            self.config = config

        def connect(self):
            return True

        def set_solid_color(self, r, g, b):
            return True

        def send_led_data(self, data):
            class Result:
                success = True
                errors = []

            return Result()


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

    def estimate_led_positions(self) -> np.ndarray:
        """
        Estimate LED positions from captured diffusion patterns using centroid analysis.

        Returns:
            Array of LED positions (led_count, 2) with [x, y] coordinates
        """
        logger.info("Estimating LED positions from diffusion patterns...")
        led_positions = np.zeros((LED_COUNT, 2), dtype=np.float32)

        for led_idx in range(LED_COUNT):
            # Combine all three color channels for position estimation
            combined_pattern = np.max(self.diffusion_patterns[led_idx], axis=0)

            # Calculate intensity-weighted centroid
            total_intensity = 0
            weighted_x = 0
            weighted_y = 0

            height, width = combined_pattern.shape
            for y in range(height):
                for x in range(width):
                    intensity = float(combined_pattern[y, x])
                    if intensity > 10:  # Threshold to ignore noise
                        total_intensity += intensity
                        weighted_x += x * intensity
                        weighted_y += y * intensity

            if total_intensity > 0:
                centroid_x = weighted_x / total_intensity
                centroid_y = weighted_y / total_intensity
                led_positions[led_idx] = [centroid_x, centroid_y]
            else:
                # Fallback for failed patterns
                logger.warning(f"Failed to estimate position for LED {led_idx}")
                led_positions[led_idx] = [width // 2, height // 2]

        logger.info(f"Estimated positions for {LED_COUNT} LEDs")
        return led_positions

    def morton_encode(self, x: float, y: float) -> int:
        """
        Convert 2D coordinates to Z-order (Morton) index for spatial locality.

        Args:
            x: X coordinate (normalized to frame width)
            y: Y coordinate (normalized to frame height)

        Returns:
            Morton-encoded integer for spatial ordering
        """
        # Normalize coordinates to [0, 1] and scale for precision
        x_norm = x / FRAME_WIDTH
        y_norm = y / FRAME_HEIGHT
        x_int = int(x_norm * 65535)  # 16-bit precision
        y_int = int(y_norm * 65535)

        result = 0
        for i in range(16):  # 16-bit precision
            result |= (x_int & (1 << i)) << i | (y_int & (1 << i)) << (i + 1)
        return result

    def create_led_spatial_ordering(self, led_positions: np.ndarray) -> dict:
        """
        Create LED ordering based on spatial proximity using Z-order curve.

        Args:
            led_positions: Array of LED positions (led_count, 2)

        Returns:
            Dictionary mapping physical_led_id -> spatially_ordered_matrix_index
        """
        # Create list of (led_id, x, y, morton_code)
        led_list = []
        for led_id, (x, y) in enumerate(led_positions):
            morton_code = self.morton_encode(float(x), float(y))
            led_list.append((led_id, x, y, morton_code))

        # Sort by Morton code for spatial locality
        led_list.sort(key=lambda item: item[3])

        # Create mapping: physical_led_id -> spatially_ordered_matrix_index
        spatial_mapping = {
            led_id: matrix_idx for matrix_idx, (led_id, _, _, _) in enumerate(led_list)
        }

        logger.info(f"Created spatial ordering for {len(spatial_mapping)} LEDs")
        return spatial_mapping

    def generate_sparse_csc_matrix(
        self,
        led_positions: np.ndarray,
        led_spatial_mapping: dict,
        sparsity_threshold: float = 0.01,
    ) -> sp.csc_matrix:
        """
        Generate sparse CSC matrix from captured diffusion patterns.

        Args:
            led_positions: LED position array
            led_spatial_mapping: Spatial ordering mapping
            sparsity_threshold: Threshold below which pixels are considered zero

        Returns:
            Sparse CSC matrix for optimization
        """
        logger.info(f"Generating sparse CSC matrix with threshold {sparsity_threshold}")

        # Prepare sparse matrix data structures
        rows = []
        cols = []
        values = []

        # Total number of pixels (single channel)
        pixels_per_channel = FRAME_HEIGHT * FRAME_WIDTH
        threshold_uint8 = int(sparsity_threshold * 255)  # Convert to uint8 scale

        for physical_led_id in range(LED_COUNT):
            # Get spatially-ordered matrix column index
            matrix_led_idx = led_spatial_mapping[physical_led_id]

            # Process all three color channels for this LED
            for channel in range(3):
                pattern = self.diffusion_patterns[physical_led_id, channel]

                # Extract significant pixels above threshold
                significant_pixels = np.where(pattern > threshold_uint8)
                pixel_rows, pixel_cols = significant_pixels

                for idx in range(len(pixel_rows)):
                    pixel_row = pixel_rows[idx]
                    pixel_col = pixel_cols[idx]
                    intensity = (
                        float(pattern[pixel_row, pixel_col]) / 255.0
                    )  # Normalize to [0,1]

                    # Calculate flattened pixel index (single channel format)
                    pixel_idx = pixel_row * FRAME_WIDTH + pixel_col

                    # Each LED has 3 columns (R, G, B) - treat as independent monochrome LEDs
                    matrix_column_idx = matrix_led_idx * 3 + channel

                    rows.append(pixel_idx)
                    cols.append(matrix_column_idx)
                    values.append(intensity)

            # Progress reporting
            if (physical_led_id + 1) % 500 == 0:
                sparsity = (
                    len(values) / ((physical_led_id + 1) * pixels_per_channel * 3) * 100
                )
                logger.info(
                    f"Processed {physical_led_id + 1}/{LED_COUNT} LEDs... "
                    f"Sparsity: {sparsity:.2f}%"
                )

        # Create CSC matrix (optimal for A^T operations in LSQR)
        logger.info(f"Creating CSC matrix from {len(values)} non-zero entries...")
        A_sparse_csc = sp.csc_matrix(
            (values, (rows, cols)),
            shape=(pixels_per_channel, LED_COUNT * 3),
            dtype=np.float32,
        )

        # Eliminate duplicate entries and compress
        A_sparse_csc.eliminate_zeros()
        A_sparse_csc = A_sparse_csc.tocsc()  # Ensure proper CSC format

        actual_sparsity = (
            A_sparse_csc.nnz / (A_sparse_csc.shape[0] * A_sparse_csc.shape[1]) * 100
        )
        memory_mb = A_sparse_csc.data.nbytes / (1024 * 1024)

        logger.info(f"Generated sparse CSC matrix")
        logger.info(f"Matrix shape: {A_sparse_csc.shape}")
        logger.info(f"Non-zero entries: {A_sparse_csc.nnz:,}")
        logger.info(f"Actual sparsity: {actual_sparsity:.3f}%")
        logger.info(f"Memory usage: {memory_mb:.1f} MB")

        return A_sparse_csc

    def save_sparse_matrix(
        self,
        sparse_matrix: sp.csc_matrix,
        led_positions: np.ndarray,
        led_spatial_mapping: dict,
        output_path: str,
        sparsity_threshold: float = 0.01,
    ) -> bool:
        """
        Save sparse CSC matrix and spatial mapping for optimization.

        Args:
            sparse_matrix: Sparse CSC matrix to save
            led_positions: LED position array
            led_spatial_mapping: LED spatial ordering mapping
            output_path: Output file path
            sparsity_threshold: Sparsity threshold used

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare metadata
            save_metadata = {
                "generator": "DiffusionPatternCapture",
                "format": "sparse_csc",
                "led_count": sparse_matrix.shape[1] // 3,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "matrix_shape": list(sparse_matrix.shape),
                "nnz": sparse_matrix.nnz,
                "sparsity_percent": sparse_matrix.nnz
                / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
                * 100,
                "sparsity_threshold": sparsity_threshold,
                "capture_timestamp": time.time(),
                "wled_host": self.wled_host,
                "wled_port": self.wled_port,
                "capture_fps": self.capture_fps,
            }

            # Save everything in a single NPZ file
            np.savez_compressed(
                output_path,
                # Sparse matrix components
                matrix_data=sparse_matrix.data,
                matrix_indices=sparse_matrix.indices,
                matrix_indptr=sparse_matrix.indptr,
                matrix_shape=sparse_matrix.shape,
                # LED information
                led_positions=led_positions,
                led_spatial_mapping=led_spatial_mapping,
                # Metadata
                metadata=save_metadata,
            )

            # Log file info
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB

            logger.info(f"Saved sparse matrix and mapping to {output_path}")
            logger.info(f"File size: {file_size:.1f} MB")

            return True

        except Exception as e:
            logger.error(f"Failed to save sparse matrix: {e}")
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
        "--sparse", "-s", action="store_true", help="Generate sparse CSC matrix format"
    )
    parser.add_argument(
        "--sparsity-threshold",
        type=float,
        default=0.01,
        help="Threshold for sparse matrix (default: 0.01)",
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

        # Save patterns (always save dense format as backup)
        if not capture_tool.save_patterns(str(output_path)):
            logger.error("Failed to save patterns")
            return 1

        # Generate sparse matrix if requested
        if args.sparse:
            logger.info("Generating sparse matrix format...")

            # Estimate LED positions from captured patterns
            led_positions = capture_tool.estimate_led_positions()

            # Create spatial ordering
            led_spatial_mapping = capture_tool.create_led_spatial_ordering(
                led_positions
            )

            # Generate sparse CSC matrix
            sparse_matrix = capture_tool.generate_sparse_csc_matrix(
                led_positions, led_spatial_mapping, args.sparsity_threshold
            )

            # Save sparse matrix
            if not capture_tool.save_sparse_matrix(
                sparse_matrix,
                led_positions,
                led_spatial_mapping,
                str(output_path),
                args.sparsity_threshold,
            ):
                logger.error("Failed to save sparse matrix")
                return 1

            logger.info("Sparse matrix generation completed successfully!")

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
