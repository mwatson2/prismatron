#!/usr/bin/env python3
"""
Diffusion Pattern Capture Tool.

This tool captures the diffusion patterns for each LED and color channel by:
1. Connecting to WLED controller
2. Setting each LED/channel to full brightness
3. Capturing camera image (800x480)
4. Analyzing optimal block positions with 4-pixel alignment
5. Storing patterns in SingleBlockMixedSparseTensor format
6. Applying RCM spatial ordering for optimal matrix bandwidth
7. Saving mixed tensor with LED positions and ordering
8. Use tools/compute_matrices.py to generate ATA matrices afterward

Features:
- Configurable block size (default: 64x64, supports 32-256)
- Precision control (fp16/fp32) for memory optimization
- uint8 storage format for memory efficiency and CUDA vectorization
- Automatic block position detection and alignment
- RCM spatial ordering for bandwidth optimization
- Modern mixed tensor storage format
- Compatible with visualization and optimization tools

Usage:
    python capture_diffusion_patterns.py --wled-host 192.168.7.140 --camera-device 0 --output patterns.npz --preview
    python capture_diffusion_patterns.py --wled-host 192.168.7.140 --output patterns.npz --block-size 64 --precision fp16 --uint8
    python capture_diffusion_patterns.py --wled-host 192.168.7.140 --camera-config config/camera.json --output patterns.npz --flip-image
    python capture_diffusion_patterns.py --wled-host 192.168.7.140 --output patterns.npz --gain 4.063512 --preview
    python tools/compute_matrices.py patterns.npz  # Generate ATA matrices
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
except ImportError:
    # Fallback for systems without CUDA
    import numpy as cp

# Import LED position utilities (add project root to path)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from tools.led_position_utils import calculate_block_positions

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.const import FRAME_HEIGHT, FRAME_WIDTH
from src.consumer.wled_client import WLEDClient, WLEDConfig
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from src.utils.spatial_ordering import compute_rcm_ordering

logger = logging.getLogger(__name__)


class CameraCapture:
    """Handles camera capture with proper configuration."""

    def __init__(
        self,
        device_id: int = 0,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
        flip_image: bool = False,
        manual_gain: Optional[float] = None,
        use_usb: bool = False,
        resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize camera capture.

        Args:
            device_id: Camera device ID (usually 0 for default camera)
            crop_region: Optional crop region (x, y, width, height) for prismatron area
            flip_image: Whether to flip the image 180 degrees (for upside-down camera mounting)
            manual_gain: Optional manual gain value for consistent LED capture (e.g., from led_gain_calibrator.py)
            use_usb: Use USB camera instead of CSI camera
            resolution: Optional camera resolution (width, height) from calibration
        """
        self.device_id = device_id
        self.crop_region = crop_region
        self.flip_image = flip_image
        self.manual_gain = manual_gain
        self.use_usb = use_usb
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_width = 0
        self.camera_height = 0
        self.gst_process = None
        self.temp_fifo = None

    def initialize(self) -> bool:
        """Initialize camera using OpenCV with GStreamer support."""
        try:
            if self.use_usb:
                # Use V4L2 for USB cameras
                logger.info(f"Using USB camera at /dev/video{self.device_id}")

                # Configure camera with v4l2-ctl BEFORE opening with OpenCV
                # This ensures consistent settings and disables all auto-adaptation
                self._configure_usb_camera_v4l2()

                # Try V4L2 backend first for USB cameras
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)

                if not self.cap.isOpened():
                    logger.warning("V4L2 backend failed, trying default backend")
                    self.cap = cv2.VideoCapture(self.device_id)

                if not self.cap.isOpened():
                    # Try using GStreamer with v4l2src as fallback
                    logger.warning("Default backend failed, trying GStreamer with v4l2src")
                    gstreamer_pipeline = (
                        f"v4l2src device=/dev/video{self.device_id} ! "
                        "videoconvert ! video/x-raw, format=BGR ! appsink"
                    )
                    logger.info(f"Using GStreamer pipeline: {gstreamer_pipeline}")
                    self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

                if not self.cap.isOpened():
                    logger.error(f"Failed to open USB camera at /dev/video{self.device_id}")
                    return False

                logger.info("USB camera opened successfully")

                # Set camera to 30 FPS for consistent timing
                try:
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    logger.info("Set USB camera to 30 FPS")
                except:
                    logger.warning("Could not set FPS on USB camera")

                # Set camera resolution if specified from calibration config
                if self.resolution is not None:
                    width, height = self.resolution
                    try:
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        logger.info(f"Set USB camera resolution to {width}x{height}")
                    except:
                        logger.warning(f"Could not set USB camera resolution to {width}x{height}")
            else:
                logger.info(f"Using OpenCV with GStreamer support for CSI camera {self.device_id}")

                # Build GStreamer pipeline with optional gain control for CSI camera
                pipeline_parts = [f"nvarguscamerasrc sensor-id={self.device_id}"]

                # Add manual gain control if specified
                if self.manual_gain is not None:
                    pipeline_parts.append(f'gainrange="{self.manual_gain} {self.manual_gain}"')
                    pipeline_parts.append("aelock=true")  # Lock auto-exposure for consistent capture
                    pipeline_parts.append("awblock=true")  # Lock auto-white-balance
                    logger.info(f"Setting manual gain: {self.manual_gain} with locked exposure and white balance")

                # Complete the pipeline
                # Use max-buffers=1 to reduce latency and get most recent frame
                gstreamer_pipeline = (
                    " ".join(pipeline_parts) + " ! "
                    "nvvidconv ! video/x-raw,format=I420 ! "
                    "videoconvert ! video/x-raw,format=BGR ! "
                    "appsink drop=1 max-buffers=1"
                )

                logger.info(f"Pipeline: {gstreamer_pipeline}")
                self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

                if not self.cap.isOpened():
                    logger.error("GStreamer pipeline failed to open")
                    return False

                logger.info("GStreamer pipeline opened successfully")

            # Get camera resolution
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Camera initialized: {self.camera_width}x{self.camera_height}")

            # Skip camera properties for now to avoid segfault
            logger.info("Skipping camera property configuration to avoid segfault")

            # Warm up camera and flush old frames
            logger.info("Warming up camera and flushing buffer...")
            # Read multiple frames to flush any buffered frames from before initialization
            for i in range(10):  # Increased from 3 to flush more frames
                try:
                    if i < 3:
                        logger.info(f"Reading warmup frame {i}...")
                    ret, frame = self.cap.read()
                    if not ret:
                        if i < 3:
                            logger.warning(f"Warmup frame {i} failed to read")
                    else:
                        if i < 3:
                            logger.info(
                                f"Warmup frame {i} successful, shape: {frame.shape if frame is not None else 'None'}"
                            )
                    # Small delay between reads to allow new frames
                    time.sleep(0.05)
                except Exception as e:
                    logger.error(f"Exception during warmup frame {i}: {e}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame using OpenCV VideoCapture.

        Returns:
            Processed frame as 800x480 RGB array, or None if failed
        """
        if not self.cap:
            logger.error("Camera not initialized")
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                return None

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply image flip if requested
            if self.flip_image:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Apply crop region if specified
            if self.crop_region:
                x, y, w, h = self.crop_region
                frame = frame[y : y + h, x : x + w]

            # Scale to target resolution (800x480)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

            return frame

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None

    def _configure_usb_camera_v4l2(self):
        """
        Configure USB camera with v4l2-ctl to disable all automatic features and set consistent manual settings.

        This method is called before opening the camera with OpenCV to ensure consistent capture
        settings throughout the diffusion pattern capture process.
        """
        device_path = f"/dev/video{self.device_id}"
        logger.info(f"Configuring USB camera {device_path} with v4l2-ctl for consistent capture...")

        # Commands to run - disable all auto features and set manual values
        commands = [
            # Disable auto white balance
            ["v4l2-ctl", "-d", device_path, "-c", "white_balance_automatic=0"],
            # Set manual exposure mode (1 = Manual Mode)
            ["v4l2-ctl", "-d", device_path, "-c", "auto_exposure=1"],
            # Disable continuous autofocus
            ["v4l2-ctl", "-d", device_path, "-c", "focus_automatic_continuous=0"],
            # Set fixed white balance temperature (daylight)
            ["v4l2-ctl", "-d", device_path, "-c", "white_balance_temperature=4600"],
            # Set backlight compensation to 0
            ["v4l2-ctl", "-d", device_path, "-c", "backlight_compensation=0"],
            # Set exposure time to proven working value
            ["v4l2-ctl", "-d", device_path, "-c", "exposure_time_absolute=100"],
            # Set brightness to keep background black while showing diffusion
            ["v4l2-ctl", "-d", device_path, "-c", "brightness=128"],
            # Set contrast to moderate level to avoid over-enhancement
            ["v4l2-ctl", "-d", device_path, "-c", "contrast=64"],
        ]

        # Set gain - use manual gain if specified, otherwise set moderate value
        if self.manual_gain is not None:
            # Map manual gain to v4l2 gain range (0-255)
            v4l2_gain = max(0, min(255, int(self.manual_gain)))
            commands.append(["v4l2-ctl", "-d", device_path, "-c", f"gain={v4l2_gain}"])
            logger.info(f"Using manual gain: {self.manual_gain} -> v4l2 gain: {v4l2_gain}")
        else:
            # Set moderate gain for consistent capture without saturation
            commands.append(["v4l2-ctl", "-d", device_path, "-c", "gain=20"])
            logger.info("Using default gain: 20 for consistent capture")

        # Execute commands
        success_count = 0
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    setting = cmd[4]  # The control=value part
                    logger.debug(f"✓ Set {setting}")
                    success_count += 1
                else:
                    setting = cmd[4]
                    logger.warning(f"✗ Failed to set {setting}: {result.stderr.strip()}")
            except subprocess.TimeoutExpired:
                setting = cmd[4]
                logger.warning(f"✗ Timeout setting {setting}")
            except Exception as e:
                setting = cmd[4]
                logger.warning(f"✗ Exception setting {setting}: {e}")

        logger.info(f"v4l2-ctl configuration: {success_count}/{len(commands)} settings applied")

        # Verify critical settings
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", device_path, "--list-ctrls"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info("Current camera settings verification:")
                for line in result.stdout.split("\n"):
                    if any(
                        ctrl in line
                        for ctrl in [
                            "auto_exposure",
                            "focus_automatic",
                            "white_balance_automatic",
                            "gain",
                            "brightness",
                            "contrast",
                        ]
                    ):
                        logger.info(f"  {line.strip()}")
            else:
                logger.warning("Could not verify camera settings")
        except:
            logger.warning("Could not verify camera settings - continuing with capture")

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
        capture_fps: float = 4.0,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
        block_size: int = 64,
        precision: str = "fp32",
        use_uint8: bool = True,
        flip_image: bool = False,
        total_led_count: int = 2600,  # Total LEDs in system (for data packets)
        capture_led_count: Optional[int] = None,  # LEDs to capture (for testing subset)
        debug_mode: bool = False,
        manual_gain: Optional[float] = None,
        use_usb: bool = False,
        camera_resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize diffusion pattern capture.

        Args:
            wled_host: WLED controller hostname/IP
            wled_port: WLED controller port
            camera_device: Camera device ID
            capture_fps: Target capture rate (captures per second)
            crop_region: Optional crop region for camera
            block_size: Block size for mixed tensor storage
            precision: Precision for mixed tensor storage ("fp16" or "fp32")
            use_uint8: Whether to use uint8 format for memory efficiency
            flip_image: Whether to flip the image 180 degrees (for upside-down camera mounting)
            led_count: Number of LEDs to capture
            debug_mode: Debug mode - only capture LEDs with physical index multiples of 100
            manual_gain: Manual camera gain value for consistent LED capture (from led_gain_calibrator.py)
            use_usb: Use USB camera instead of CSI camera
            camera_resolution: Optional camera resolution (width, height) from calibration config
        """
        self.wled_host = wled_host
        self.wled_port = wled_port
        self.capture_fps = capture_fps
        self.capture_interval = 1.0 / capture_fps
        self.block_size = block_size
        self.precision = precision
        self.use_uint8 = use_uint8
        self.debug_mode = debug_mode

        # Set LED counts
        self.total_led_count = total_led_count  # Total LEDs in system (for data packets)
        self.led_count = capture_led_count if capture_led_count is not None else total_led_count  # LEDs to capture

        logger.info(
            f"LED configuration: Total LEDs in system: {self.total_led_count}, LEDs to capture: {self.led_count}"
        )

        # Ensure capture count doesn't exceed total count
        if self.led_count > self.total_led_count:
            logger.warning(
                f"Capture LED count ({self.led_count}) exceeds total LED count ({self.total_led_count}), limiting to {self.total_led_count}"
            )
            self.led_count = self.total_led_count

        # Create debug LED list if in debug mode (multiples of 100)
        if self.debug_mode:
            self.debug_leds = [i for i in range(0, self.led_count, 100)]
            logger.info(f"Debug mode: will capture {len(self.debug_leds)} LEDs: {self.debug_leds}")
        else:
            self.debug_leds = None

        # Initialize WLED client with TOTAL LED count (so we send data for all LEDs)
        wled_config = WLEDConfig(host=wled_host, port=wled_port, led_count=self.total_led_count, max_fps=60.0)
        self.wled_client = WLEDClient(wled_config)

        # Initialize camera
        self.camera = CameraCapture(camera_device, crop_region, flip_image, manual_gain, use_usb, camera_resolution)

        # Determine output dtype based on precision and format
        if use_uint8:
            # Use uint8 for storage, float32 for computation
            tensor_dtype = cp.uint8
            output_dtype = cp.float32
        elif precision == "fp16":
            tensor_dtype = cp.float16
            output_dtype = cp.float16
        else:
            tensor_dtype = cp.float32
            output_dtype = cp.float32

        # Initialize mixed tensor for pattern storage
        self.mixed_tensor = SingleBlockMixedSparseTensor(
            batch_size=self.led_count,
            channels=3,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            block_size=block_size,
            device="cpu",  # Use CPU for capture
            dtype=tensor_dtype,
            output_dtype=output_dtype,
        )

        # Storage for LED positions and block positions
        self.led_positions = np.zeros((self.led_count, 2), dtype=np.float32)
        self.block_positions = np.zeros((self.led_count, 2), dtype=np.int32)  # Top-left corner of each block
        self.led_spatial_mapping = None  # Will be set after RCM reordering
        self.failed_leds = []  # Track LEDs that failed to capture
        self.failed_led_details = {}  # Track detailed failure information

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
            # Determine which LEDs to capture
            if self.debug_mode:
                led_list = self.debug_leds
                total_captures = len(led_list) * 3  # 3 channels per LED
                logger.info(f"Debug mode: capturing {total_captures} diffusion patterns for LEDs {led_list}")
            else:
                led_list = range(self.led_count)
                total_captures = self.led_count * 3  # 3 channels per LED
                logger.info(f"Starting capture of {total_captures} diffusion patterns")

            # Turn off all LEDs initially
            logger.info("Turning off all LEDs and waiting for camera to stabilize...")
            self.wled_client.set_solid_color(0, 0, 0)
            time.sleep(1.0)  # Wait for LEDs to turn off and camera to catch up

            # Flush any frames that might show the previous LED state
            logger.info("Flushing camera buffer...")
            for _ in range(5):
                _ = self.camera.capture_frame()
                time.sleep(0.033)  # ~1 frame at 30fps

            capture_count = 0
            for led_idx in led_list:
                logger.info(f"=" * 60)
                logger.info(f"STARTING LED {led_idx} - will capture Red, Green, Blue channels")
                logger.info(f"=" * 60)

                led_has_valid_block = False  # Track if we found a valid block for this LED
                block_position_determined = False  # Track if we've determined block position

                for channel_idx in range(3):  # R, G, B channels
                    channel_names = ["Red", "Green", "Blue"]  # Define channel names here
                    capture_count += 1

                    logger.info(
                        f"Capturing LED {led_idx}, Channel {channel_idx} ({channel_names[channel_idx]}) ({capture_count}/{total_captures})"
                    )

                    capture_successful = False

                    # Ensure all LEDs are off before lighting the next one
                    self.wled_client.set_solid_color(0, 0, 0)
                    time.sleep(0.2)  # 200ms gap to ensure LEDs are off

                    # Create LED data array for ALL LEDs (all off except current LED/channel)
                    led_data = np.zeros((self.total_led_count, 3), dtype=np.uint8)
                    led_data[led_idx, channel_idx] = 255  # Full brightness for this LED/channel

                    # Verify LED data is correct before sending
                    non_zero_leds = np.nonzero(led_data)
                    if len(non_zero_leds[0]) > 0:
                        logger.info(f"LED data verification: {len(non_zero_leds[0])} non-zero values")
                        for i in range(len(non_zero_leds[0])):
                            led_num = non_zero_leds[0][i]
                            channel_num = non_zero_leds[1][i]
                            value = led_data[led_num, channel_num]
                            channel_name = ["R", "G", "B"][channel_num]
                            logger.info(f"  LED {led_num} {channel_name} = {value}")
                    else:
                        logger.error("ERROR: LED data array is all zeros!")
                        continue

                    # Send to WLED with validation
                    logger.info(
                        f"Sending LED data - LED {led_idx}, channel {channel_idx} ({channel_names[channel_idx]}) = 255"
                    )
                    result = self.wled_client.send_led_data(led_data)
                    if not result.success:
                        logger.error(f"Failed to send LED data: {result.errors}")
                        continue
                    logger.debug(f"LED data sent successfully")

                    # Send a second time immediately to ensure reception
                    logger.debug(f"Sending LED data again for reliability")
                    result2 = self.wled_client.send_led_data(led_data)
                    if not result2.success:
                        logger.warning(f"Second LED data send failed: {result2.errors}")

                    # Brief wait for LED to start lighting
                    time.sleep(0.3)

                    # Fast polling for expected color detection
                    max_poll_attempts = 60  # Poll up to 60 times (about 12 seconds total)
                    poll_interval = 0.2  # Check every 200ms
                    color_detected = False
                    last_frame_hash = None  # Track frame changes to detect stale captures

                    channel_names = ["Red", "Green", "Blue"]
                    logger.info(
                        f"Fast polling for {channel_names[channel_idx]} color on LED {led_idx} (up to {max_poll_attempts} attempts)"
                    )

                    for poll_attempt in range(max_poll_attempts):
                        # Keep LED lit by resending data very frequently to prevent timeout
                        if poll_attempt % 2 == 0:  # Every 400ms (2x per second minimum)
                            result = self.wled_client.send_led_data(led_data)
                            if not result.success:
                                logger.warning(
                                    f"Failed to resend LED data during polling attempt {poll_attempt}: {result.errors}"
                                )
                            else:
                                logger.debug(f"LED data resent successfully (attempt {poll_attempt})")

                        # Flush camera buffer aggressively - read and discard multiple frames
                        # This ensures we get the most recent frame with current LED state
                        for flush_attempt in range(3):  # Flush 3 frames
                            flush_frame = self.camera.capture_frame()
                            if flush_frame is None:
                                break

                        # Small delay to let LED update after data send
                        time.sleep(0.05)

                        # Now capture the frame we want to analyze
                        frame = self.camera.capture_frame()
                        if frame is None:
                            logger.warning(f"Failed to capture frame during polling attempt {poll_attempt}")
                            time.sleep(poll_interval)
                            continue

                        # Check for stale frame detection using a simple hash
                        frame_hash = hash(frame.data.tobytes())
                        if frame_hash == last_frame_hash:
                            logger.warning(
                                f"Poll attempt {poll_attempt}: Same frame detected as previous attempt - possible stale camera buffer"
                            )
                        last_frame_hash = frame_hash

                        # Log frame stats for first few attempts to detect issues
                        if poll_attempt < 5:
                            frame_mean = np.mean(frame)
                            frame_max = np.max(frame)
                            red_mean = np.mean(frame[:, :, 0])
                            logger.debug(
                                f"Poll attempt {poll_attempt}: frame_mean={frame_mean:.1f}, frame_max={frame_max}, red_mean={red_mean:.1f}, hash={frame_hash}"
                            )

                        # For red channel (first), determine/re-determine block position on each attempt
                        # For other channels, use the position found during red channel
                        if channel_idx == 0:
                            # For red channel, keep trying to find the best position until color detected
                            current_top, current_left = self._find_optimal_block_position(frame, led_idx)

                            # Update position if this is the first attempt or if we haven't found color yet
                            if not block_position_determined or not color_detected:
                                top_row, left_col = current_top, current_left
                                self.block_positions[led_idx] = [top_row, left_col]
                                if not block_position_determined:
                                    block_position_determined = True
                                    logger.debug(f"  Initial block position for LED {led_idx}: ({top_row}, {left_col})")
                                else:
                                    logger.debug(
                                        f"  Updated block position for LED {led_idx}: ({top_row}, {left_col}) (attempt {poll_attempt+1})"
                                    )
                        else:
                            # Use the position determined during red channel
                            top_row, left_col = self.block_positions[led_idx]

                        # Check if we can see the expected color in this frame
                        if self._check_expected_color_luminance(frame, top_row, left_col, channel_idx, led_idx):
                            logger.debug(
                                f"  Expected color detected after {poll_attempt+1} polling attempts at position ({top_row}, {left_col})"
                            )
                            color_detected = True
                            break

                        time.sleep(poll_interval)

                    if not color_detected:
                        logger.warning(
                            f"Expected {channel_names[channel_idx]} color not detected after {max_poll_attempts} attempts"
                        )

                        # Record detailed failure information
                        failure_key = f"{led_idx}_{channel_idx}"
                        self.failed_led_details[failure_key] = {
                            "led_idx": led_idx,
                            "channel": channel_idx,
                            "channel_name": channel_names[channel_idx],
                            "failure_type": "color_detection",
                            "description": f"Expected {channel_names[channel_idx]} color not detected after {max_poll_attempts} attempts",
                            "poll_attempts": max_poll_attempts,
                        }

                        # Save debug image for failed LED if we have a frame (always save for failures)
                        if frame is not None:
                            # Determine block position for debug image
                            if block_position_determined:
                                top_row, left_col = self.block_positions[led_idx]
                            else:
                                # Try to find optimal position even for failed LED
                                try:
                                    top_row, left_col = self._find_optimal_block_position(frame, led_idx)
                                    self.block_positions[led_idx] = [top_row, left_col]
                                    block_position_determined = True
                                except:
                                    # Use center as final fallback
                                    top_row = (FRAME_HEIGHT - self.block_size) // 2
                                    left_col = (FRAME_WIDTH - self.block_size) // 2
                                    self.block_positions[led_idx] = [top_row, left_col]

                            # Save debug image with failure annotation
                            self._save_debug_image_with_failure(
                                frame,
                                led_idx,
                                top_row,
                                left_col,
                                f"FAILED_{channel_names[channel_idx]}_COLOR_DETECTION",
                            )

                            # Add debug image info to failure details
                            self.failed_led_details[failure_key]["debug_image"] = f"FAILED_led_{led_idx:04d}.jpg"
                            self.failed_led_details[failure_key][
                                "debug_heatmap"
                            ] = f"FAILED_led_{led_idx:04d}_heatmap.jpg"

                        # Store zero block as placeholder
                        if not block_position_determined:
                            top_row = (FRAME_HEIGHT - self.block_size) // 2
                            left_col = (FRAME_WIDTH - self.block_size) // 2
                            self.block_positions[led_idx] = [top_row, left_col]
                        else:
                            top_row, left_col = self.block_positions[led_idx]

                        zero_block = cp.zeros(
                            (self.block_size, self.block_size), dtype=cp.float32 if not self.use_uint8 else cp.uint8
                        )
                        self.mixed_tensor.set_block(led_idx, channel_idx, top_row, left_col, zero_block)
                        continue

                    # Color detected - capture 3 additional frames to ensure LED is fully lit
                    logger.debug(f"Capturing 3 confirmation frames for LED {led_idx}, channel {channel_idx}")
                    confirmation_frames = []

                    for conf_frame_idx in range(3):
                        # Keep LED lit
                        self.wled_client.send_led_data(led_data)
                        time.sleep(0.1)  # Brief wait between frames

                        conf_frame = self.camera.capture_frame()
                        if conf_frame is not None:
                            confirmation_frames.append(conf_frame)

                    if not confirmation_frames:
                        logger.warning(
                            f"Failed to capture any confirmation frames for LED {led_idx}, channel {channel_idx}"
                        )
                        continue

                    # Use the best confirmation frame (highest luminance in the expected channel)
                    best_frame = None
                    best_luminance = -1

                    for conf_frame in confirmation_frames:
                        block_rgb = conf_frame[
                            top_row : top_row + self.block_size, left_col : left_col + self.block_size, :
                        ]
                        channel_luminance = np.mean(block_rgb[:, :, channel_idx])

                        if channel_luminance > best_luminance:
                            best_luminance = channel_luminance
                            best_frame = conf_frame

                    # Position already optimized during polling for red channel
                    top_row, left_col = self.block_positions[led_idx]

                    # Convert frame to appropriate format
                    if self.use_uint8:
                        pattern_data = best_frame.astype(np.uint8)
                    elif self.precision == "fp16":
                        pattern_data = best_frame.astype(np.float16) / 255.0
                    else:
                        pattern_data = best_frame.astype(np.float32) / 255.0

                    # Extract block from the best frame
                    block = pattern_data[
                        top_row : top_row + self.block_size, left_col : left_col + self.block_size, channel_idx
                    ]

                    # Ensure block is the right size (pad with zeros if needed)
                    if block.shape != (self.block_size, self.block_size):
                        padded_block = np.zeros((self.block_size, self.block_size), dtype=pattern_data.dtype)
                        h, w = min(block.shape[0], self.block_size), min(block.shape[1], self.block_size)
                        padded_block[:h, :w] = block[:h, :w]
                        block = padded_block

                    # Final luminance check
                    if self._check_block_luminance(block, channel_idx, led_idx):
                        # Store block in mixed tensor
                        block_cupy = cp.asarray(block)
                        self.mixed_tensor.set_block(led_idx, channel_idx, top_row, left_col, block_cupy)
                        capture_successful = True
                        led_has_valid_block = True
                        logger.debug(f"  Successfully captured LED {led_idx}, channel {channel_idx}")

                        # Show preview if requested
                        if preview:
                            self._show_preview(best_frame, led_idx, channel_idx, capture_count, total_captures)
                    else:
                        logger.warning(f"Final luminance check failed for LED {led_idx}, channel {channel_idx}")

                        # Record detailed failure information
                        failure_key = f"{led_idx}_{channel_idx}"
                        self.failed_led_details[failure_key] = {
                            "led_idx": led_idx,
                            "channel": channel_idx,
                            "channel_name": channel_names[channel_idx],
                            "failure_type": "luminance_check",
                            "description": f"Final {channel_names[channel_idx]} luminance check failed after successful color detection",
                            "confirmation_frames": len(confirmation_frames),
                            "best_luminance": best_luminance,
                        }

                        # Save debug image for failed final luminance check (always save for failures)
                        if best_frame is not None:
                            self._save_debug_image_with_failure(
                                best_frame,
                                led_idx,
                                top_row,
                                left_col,
                                f"FAILED_{channel_names[channel_idx]}_LUMINANCE_CHECK",
                            )

                            # Add debug image info to failure details
                            self.failed_led_details[failure_key]["debug_image"] = f"FAILED_led_{led_idx:04d}.jpg"
                            self.failed_led_details[failure_key][
                                "debug_heatmap"
                            ] = f"FAILED_led_{led_idx:04d}_heatmap.jpg"

                        # Store zero block as fallback
                        zero_block = cp.zeros(
                            (self.block_size, self.block_size), dtype=cp.float32 if not self.use_uint8 else cp.uint8
                        )
                        self.mixed_tensor.set_block(led_idx, channel_idx, top_row, left_col, zero_block)

                    # Progress update
                    if capture_count % 100 == 0:
                        progress = (capture_count / total_captures) * 100
                        logger.info(f"Progress: {progress:.1f}% ({capture_count}/{total_captures})")

                # After all channels for this LED, check if we failed to capture it
                if not led_has_valid_block:
                    self.failed_leds.append(led_idx)
                    logger.error(f"LED {led_idx} completely failed to capture")

            # Turn off all LEDs
            self.wled_client.set_solid_color(0, 0, 0)

            # Report on failed LEDs
            if self.failed_leds:
                logger.warning(f"Failed to capture {len(self.failed_leds)} LEDs: {self.failed_leds}")
                logger.warning("These LEDs will have zero patterns in the output")
            else:
                logger.info("All LEDs captured successfully")

            logger.info("Diffusion pattern capture completed")

            # Reorder patterns using RCM spatial ordering
            self.mixed_tensor, self.led_spatial_mapping = self._reorder_to_rcm_spatial_ordering()

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

    def _save_debug_image(self, frame: np.ndarray, led_id: int, top_row: int, left_col: int):
        """
        Save debug image showing the detected block region for an LED.

        Args:
            frame: Full camera frame (RGB)
            led_id: LED physical index
            top_row: Top row of detected block
            left_col: Left column of detected block
        """
        try:
            # Create debug output directory
            debug_dir = Path("debug_led_blocks")
            debug_dir.mkdir(exist_ok=True)

            # Convert frame to BGR for OpenCV
            debug_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw the 64x64 block rectangle
            bottom_row = min(top_row + self.block_size, frame.shape[0])
            right_col = min(left_col + self.block_size, frame.shape[1])

            # Draw rectangle (BGR color: green)
            cv2.rectangle(debug_frame, (left_col, top_row), (right_col, bottom_row), (0, 255, 0), 2)

            # Add text label with LED ID and position
            label = f"LED {led_id}: ({top_row},{left_col})"
            cv2.putText(debug_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Add centroid position if available
            if led_id < len(self.led_positions):
                centroid_x, centroid_y = self.led_positions[led_id]
                centroid_label = f"Centroid: ({centroid_x:.1f}, {centroid_y:.1f})"
                cv2.putText(debug_frame, centroid_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw crosshair at centroid
                cx, cy = int(centroid_x), int(centroid_y)
                cv2.line(debug_frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 255), 1)
                cv2.line(debug_frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 255), 1)

            # Save main debug image
            filename = debug_dir / f"led_{led_id:04d}.jpg"
            cv2.imwrite(str(filename), debug_frame)
            logger.info(f"Saved debug image for LED {led_id}: {filename}")

            # Also save a heatmap showing the intensity pattern
            combined_pattern = np.max(frame, axis=2).astype(np.uint8)
            heatmap = cv2.applyColorMap(combined_pattern, cv2.COLORMAP_JET)

            # Draw the selected block on the heatmap
            cv2.rectangle(heatmap, (left_col, top_row), (right_col, bottom_row), (255, 255, 255), 2)

            # Save heatmap
            heatmap_filename = debug_dir / f"led_{led_id:04d}_heatmap.jpg"
            cv2.imwrite(str(heatmap_filename), heatmap)
            logger.info(f"Saved heatmap for LED {led_id}: {heatmap_filename}")

        except Exception as e:
            logger.warning(f"Failed to save debug image for LED {led_id}: {e}")

    def _save_debug_image_with_failure(
        self, frame: np.ndarray, led_id: int, top_row: int, left_col: int, failure_reason: str
    ):
        """
        Save debug image for a failed LED capture with failure annotation.

        Args:
            frame: The captured frame (RGB)
            led_id: LED index
            top_row, left_col: Block position
            failure_reason: Description of the failure
        """
        try:
            debug_dir = Path("debug_led_blocks")
            debug_dir.mkdir(exist_ok=True)

            # Validate input
            if frame is None:
                logger.warning(f"Cannot save failed debug image for LED {led_id}: frame is None")
                return
            if frame.shape[2] != 3:
                logger.warning(f"Cannot save failed debug image for LED {led_id}: invalid frame shape {frame.shape}")
                return

            logger.info(f"Saving failed LED {led_id} debug image to: {debug_dir} (reason: {failure_reason})")

            # Convert RGB to BGR for OpenCV
            debug_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

            # Draw the 64x64 block rectangle in RED for failures
            bottom_row = min(top_row + self.block_size, frame.shape[0])
            right_col = min(left_col + self.block_size, frame.shape[1])

            # Draw rectangle (BGR color: red for failure)
            cv2.rectangle(debug_frame, (left_col, top_row), (right_col, bottom_row), (0, 0, 255), 3)

            # Add failure text labels
            failure_label = f"LED {led_id}: FAILED"
            cv2.putText(debug_frame, failure_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            reason_label = failure_reason
            cv2.putText(debug_frame, reason_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            position_label = f"Block: ({top_row},{left_col})"
            cv2.putText(debug_frame, position_label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Add centroid position if available
            if led_id < len(self.led_positions):
                centroid_x, centroid_y = self.led_positions[led_id]
                centroid_label = f"Centroid: ({centroid_x:.1f}, {centroid_y:.1f})"
                cv2.putText(debug_frame, centroid_label, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw crosshair at centroid
                cx, cy = int(centroid_x), int(centroid_y)
                cv2.line(debug_frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 255), 1)
                cv2.line(debug_frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 255), 1)

            # Save failure debug image with FAILED prefix
            filename = debug_dir / f"FAILED_led_{led_id:04d}.jpg"
            cv2.imwrite(str(filename), debug_frame)
            logger.warning(f"Saved FAILED debug image for LED {led_id}: {filename}")

            # Also save a heatmap showing the intensity pattern
            combined_pattern = np.max(frame, axis=2).astype(np.uint8)
            heatmap = cv2.applyColorMap(combined_pattern, cv2.COLORMAP_JET)

            # Draw the selected block on the heatmap in red for failure
            cv2.rectangle(heatmap, (left_col, top_row), (right_col, bottom_row), (0, 0, 255), 3)

            # Add failure text on heatmap
            cv2.putText(heatmap, f"FAILED LED {led_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Save failure heatmap
            heatmap_filename = debug_dir / f"FAILED_led_{led_id:04d}_heatmap.jpg"
            cv2.imwrite(str(heatmap_filename), heatmap)
            logger.warning(f"Saved FAILED heatmap for LED {led_id}: {heatmap_filename}")

        except Exception as e:
            logger.warning(f"Failed to save FAILED debug image for LED {led_id}: {e}")

    def _check_expected_color_luminance(
        self, frame: np.ndarray, top_row: int, left_col: int, channel_idx: int, led_idx: int
    ) -> bool:
        """
        Check if the captured frame shows the expected color luminance in the LED block.
        This validates we're seeing the correct LED state, not the previous LED.

        Args:
            frame: The full captured frame (RGB)
            top_row, left_col: Block position
            channel_idx: Which color channel (0=R, 1=G, 2=B)
            led_idx: LED index for logging

        Returns:
            True if the expected color dominates in the block
        """
        # Extract the block for all channels
        block_rgb = frame[top_row : top_row + self.block_size, left_col : left_col + self.block_size, :]

        # Get corner background for each channel
        corner_size = 8
        corners = [
            block_rgb[:corner_size, :corner_size, :],
            block_rgb[:corner_size, -corner_size:, :],
            block_rgb[-corner_size:, :corner_size, :],
            block_rgb[-corner_size:, -corner_size:, :],
        ]
        background_rgb = np.mean([np.mean(corner, axis=(0, 1)) for corner in corners], axis=0)

        # Calculate mean luminance for each channel in the block
        block_means = np.mean(block_rgb, axis=(0, 1))
        above_background = block_means - background_rgb

        channel_names = ["Red", "Green", "Blue"]

        # Use WS2811 color space conversion for better LED color detection
        # The camera sees Rec.709 RGB, but LEDs emit WS2811 primaries
        # Converting back to WS2811 space should give better color separation

        min_absolute_threshold = 3.0 if self.use_uint8 else 3.0 / 255.0  # Basic noise floor

        # Check basic signal strength
        expected_channel_bright = above_background[channel_idx] > min_absolute_threshold

        r, g, b = above_background[0], above_background[1], above_background[2]
        total_signal = r + g + b

        if total_signal < min_absolute_threshold:
            ratio_test_passes = False
            logger.info(
                f"  WS2811 color detection: Total signal too low ({total_signal:.1f} < {min_absolute_threshold})"
            )
        else:
            # Normalize RGB values (work in linear space)
            max_val = max(r, g, b)
            if max_val > 0:
                r_norm = r / max_val
                g_norm = g / max_val
                b_norm = b / max_val
            else:
                r_norm = g_norm = b_norm = 0

            # Rec.709 → WS2811 transformation matrix (inverse of WS2811→Rec.709)
            # Pre-computed from your provided code:
            # M_ws2811_to_rec709 ≈ [[ 0.856,  0.144,  0.000],
            #                        [ 0.016,  0.844,  0.140],
            #                        [ 0.000,  0.066,  0.934]]
            # So inverse is:
            M_rec709_to_ws2811 = np.array([[1.169, -0.200, 0.001], [-0.022, 1.187, -0.177], [0.001, -0.084, 1.071]])

            # Convert Rec.709 RGB to WS2811 RGB
            rec709_rgb = np.array([r_norm, g_norm, b_norm])
            ws2811_rgb = M_rec709_to_ws2811 @ rec709_rgb

            # Clip negative values (can happen due to out-of-gamut colors)
            ws2811_rgb = np.maximum(ws2811_rgb, 0)

            # Renormalize after conversion
            ws_max = np.max(ws2811_rgb)
            if ws_max > 0:
                ws2811_rgb = ws2811_rgb / ws_max

            ws_r, ws_g, ws_b = ws2811_rgb

            # In WS2811 space, colors should be much more separated
            # Use simple dominance test with threshold
            dominance_threshold = 0.6  # Expected channel should be >60% of signal

            if channel_idx == 0:  # Red
                ratio_test_passes = ws_r > dominance_threshold
                logger.info(f"  WS2811 Red detection: WS_R={ws_r:.2f} > {dominance_threshold}: {ratio_test_passes}")
            elif channel_idx == 1:  # Green
                ratio_test_passes = ws_g > dominance_threshold
                logger.info(f"  WS2811 Green detection: WS_G={ws_g:.2f} > {dominance_threshold}: {ratio_test_passes}")
            else:  # Blue
                ratio_test_passes = ws_b > dominance_threshold
                logger.info(f"  WS2811 Blue detection: WS_B={ws_b:.2f} > {dominance_threshold}: {ratio_test_passes}")

            logger.info(f"  Rec.709 RGB (normalized): R={r_norm:.2f}, G={g_norm:.2f}, B={b_norm:.2f}")
            logger.info(f"  WS2811 RGB (converted): R={ws_r:.2f}, G={ws_g:.2f}, B={ws_b:.2f}")

            # Also show which channel dominates in WS2811 space
            ws_channels = [ws_r, ws_g, ws_b]
            dominant_channel = np.argmax(ws_channels)
            color_names = ["RED", "GREEN", "BLUE"]
            logger.info(
                f"  Dominant in WS2811 space: {color_names[dominant_channel]} ({ws_channels[dominant_channel]:.2f})"
            )

        logger.info(
            f"LED {led_idx} {channel_names[channel_idx]} color check: "
            f"R={above_background[0]:.1f}, G={above_background[1]:.1f}, B={above_background[2]:.1f}"
        )
        logger.info(
            f"  Expected {channel_names[channel_idx]} > {min_absolute_threshold:.1f}: {expected_channel_bright}"
        )

        has_expected_color = expected_channel_bright and ratio_test_passes

        if has_expected_color:
            logger.info(f"  ✓ Expected {channel_names[channel_idx]} color detected")
        else:
            logger.warning(
                f"  ✗ Expected {channel_names[channel_idx]} color not detected "
                f"(bright_enough={expected_channel_bright}, ratio_test_passes={ratio_test_passes})"
            )

        return has_expected_color

    def _check_block_luminance(self, block: np.ndarray, channel_idx: int, led_idx: int) -> bool:
        """
        Check if a captured block has sufficient luminance above background.

        Args:
            block: The captured block data (64x64 for single channel)
            channel_idx: Which color channel (0=R, 1=G, 2=B)
            led_idx: LED index for logging

        Returns:
            True if block has sufficient luminance, False otherwise
        """
        # Calculate mean and max of the block
        block_mean = np.mean(block)
        block_max = np.max(block)

        # Get a baseline from the corners (likely background)
        corner_size = 8
        corners = [
            block[:corner_size, :corner_size],
            block[:corner_size, -corner_size:],
            block[-corner_size:, :corner_size],
            block[-corner_size:, -corner_size:],
        ]
        background_mean = np.mean([np.mean(c) for c in corners])

        # Check if there's significant luminance above background
        # For uint8: expect at least 10 above background mean and max > 20
        # For float: scale these thresholds accordingly
        if self.use_uint8:
            min_above_background = 10
            min_max_value = 20
        else:
            min_above_background = 10 / 255.0
            min_max_value = 20 / 255.0

        above_background = block_mean - background_mean

        channel_names = ["Red", "Green", "Blue"]
        logger.debug(
            f"LED {led_idx} {channel_names[channel_idx]}: "
            f"mean={block_mean:.2f}, max={block_max:.2f}, "
            f"background={background_mean:.2f}, above_bg={above_background:.2f}"
        )

        # Check conditions
        has_luminance = (above_background > min_above_background) or (block_max > min_max_value)

        if not has_luminance:
            logger.warning(
                f"LED {led_idx} {channel_names[channel_idx]}: "
                f"Insufficient luminance detected (mean={block_mean:.2f}, max={block_max:.2f})"
            )

        return has_luminance

    def _align_to_pixel_boundary(self, x_coord: int) -> int:
        """
        Align x-coordinate to 4-pixel boundary.

        Args:
            x_coord: Original x-coordinate

        Returns:
            Aligned x-coordinate (rounded down to multiple of 4)
        """
        return (x_coord // 4) * 4

    def _find_optimal_block_position(self, pattern: np.ndarray, led_id: int) -> Tuple[int, int]:
        """
        Find the 64x64 block position with maximum total intensity using tiled search.

        Args:
            pattern: Captured pattern (height, width, 3)
            led_id: LED ID for position estimation

        Returns:
            Tuple of (top_row, left_col) for brightest block position (aligned to 4-pixel boundary)
        """
        try:
            logger.info(f"=== Finding optimal block position for LED {led_id} ===")

            # Combine all three color channels for intensity analysis
            combined_pattern = np.max(pattern, axis=2).astype(np.float32)
            height, width = combined_pattern.shape

            # Log overall image statistics
            img_mean = np.mean(combined_pattern)
            img_max = np.max(combined_pattern)
            img_min = np.min(combined_pattern)
            logger.info(
                f"Image stats - shape: {height}x{width}, mean: {img_mean:.1f}, max: {img_max:.1f}, min: {img_min:.1f}"
            )

            # Find brightest pixels in entire image
            bright_threshold = np.percentile(combined_pattern, 99)
            bright_pixels = combined_pattern > bright_threshold
            num_bright = np.sum(bright_pixels)
            if num_bright > 0:
                bright_y, bright_x = np.where(bright_pixels)
                logger.info(f"Found {num_bright} pixels above 99th percentile ({bright_threshold:.1f})")
                logger.info(f"Brightest pixel regions (first 10): {list(zip(bright_x[:10], bright_y[:10]))}")

            # Step 1: Tile the image into non-overlapping 64x64 blocks
            logger.info(f"Step 1: Tiling image into {self.block_size}x{self.block_size} blocks")
            tiles = []
            tile_intensities = []

            for tile_row in range(0, height - self.block_size + 1, self.block_size):
                for tile_col in range(0, width - self.block_size + 1, self.block_size):
                    # Extract 64x64 tile
                    tile = combined_pattern[
                        tile_row : tile_row + self.block_size, tile_col : tile_col + self.block_size
                    ]

                    # Calculate total intensity
                    tile_intensity = np.sum(tile)
                    tiles.append((tile_row, tile_col))
                    tile_intensities.append(tile_intensity)

            # Log top 5 brightest tiles
            if tile_intensities:
                sorted_tiles = sorted(zip(tile_intensities, tiles), reverse=True)
                logger.info(f"Top 5 brightest tiles:")
                for i, (intensity, (row, col)) in enumerate(sorted_tiles[:5]):
                    logger.info(f"  {i+1}. Tile at ({row}, {col}): intensity = {intensity:.1f}")

            # Step 2: Find the brightest tile first, then search around it
            logger.info(f"Step 2: Finding brightest region to focus search")

            # Find the brightest tile from Step 1
            if tile_intensities:
                max_tile_idx = np.argmax(tile_intensities)
                brightest_tile_row, brightest_tile_col = tiles[max_tile_idx]
                brightest_intensity = tile_intensities[max_tile_idx]
                logger.info(
                    f"Brightest tile at ({brightest_tile_row}, {brightest_tile_col}) with intensity {brightest_intensity:.1f}"
                )
            else:
                # Fallback to center if no tiles found
                brightest_tile_row = height // 2
                brightest_tile_col = width // 2
                logger.warning("No tiles found, using center as fallback")

            # Now search for the best 128x128 region centered around the brightest tile
            region_size = 2 * self.block_size  # 128x128 region

            # Start with the region containing the brightest tile
            best_region_top = max(0, brightest_tile_row - self.block_size // 2)
            best_region_left = max(0, brightest_tile_col - self.block_size // 2)

            # Ensure the region fits within the image
            if best_region_top + region_size > height:
                best_region_top = max(0, height - region_size)
            if best_region_left + region_size > width:
                best_region_left = max(0, width - region_size)

            logger.info(f"Using 128x128 region around brightest area: ({best_region_top}, {best_region_left})")

            # Skip the contrast analysis - we'll just use the brightest region
            max_contrast_score = brightest_intensity  # Use intensity as the score

            logger.info(
                f"Best 128x128 region selected at ({best_region_top}, {best_region_left}) based on brightest tile"
            )

            # Step 3: Within the best 128x128 region, find the optimal 64x64 block
            logger.info(f"Step 3: Finding optimal 64x64 block within best region")
            # Expand search to include all blocks whose centers fall within the best region
            half_block = self.block_size // 2
            search_top = max(0, best_region_top - half_block)
            search_left = max(0, best_region_left - half_block)
            search_bottom = min(height, best_region_top + region_size + half_block)
            search_right = min(width, best_region_left + region_size + half_block)

            logger.info(f"Search area: ({search_top}, {search_left}) to ({search_bottom}, {search_right})")

            # Step 4: Find the brightest 64x64 block in the search area
            logger.info(f"Step 4: Finding brightest 64x64 block in search area")
            max_intensity = 0
            max_intensity_score = 0
            best_top_row = search_top
            best_left_col = search_left

            block_scores = []  # Track top blocks for debugging

            for top_row in range(search_top, search_bottom - self.block_size + 1, 4):
                for left_col in range(search_left, search_right - self.block_size + 1, 4):
                    # Ensure block stays within image bounds
                    if top_row + self.block_size > height or left_col + self.block_size > width:
                        continue

                    # Extract the current 64x64 block
                    block = combined_pattern[top_row : top_row + self.block_size, left_col : left_col + self.block_size]

                    # Use simple sum of intensities as the primary metric
                    intensity = np.sum(block)

                    # Also calculate max pixel value in block for debugging
                    block_max = np.max(block)

                    # Track for debugging
                    if intensity > 0:
                        block_scores.append((intensity, block_max, top_row, left_col))

                    # Keep track of the block with highest intensity
                    if intensity > max_intensity:
                        max_intensity = intensity
                        max_intensity_score = np.sum(np.square(block))  # Keep for logging
                        best_top_row = top_row
                        best_left_col = left_col

            # Log top blocks found
            if block_scores:
                block_scores.sort(reverse=True)
                logger.info(f"Top 3 blocks in search area:")
                for i, (intensity, block_max, row, col) in enumerate(block_scores[:3]):
                    logger.info(
                        f"  {i+1}. Block at ({row}, {col}): intensity={intensity:.1f}, max_pixel={block_max:.1f}"
                    )

            # Align to 4-pixel boundary
            best_left_col = self._align_to_pixel_boundary(best_left_col)

            # Calculate centroid within the brightest block for reference
            best_block = combined_pattern[
                best_top_row : best_top_row + self.block_size, best_left_col : best_left_col + self.block_size
            ]

            # Find centroid within the block
            block_total = np.sum(best_block)
            if block_total > 0:
                y_indices, x_indices = np.indices(best_block.shape)
                centroid_x_in_block = np.sum(x_indices * best_block) / block_total
                centroid_y_in_block = np.sum(y_indices * best_block) / block_total

                # Convert to global coordinates
                global_centroid_x = best_left_col + centroid_x_in_block
                global_centroid_y = best_top_row + centroid_y_in_block
            else:
                global_centroid_x = best_left_col + self.block_size // 2
                global_centroid_y = best_top_row + self.block_size // 2

            # Store estimated LED position
            self.led_positions[led_id] = [global_centroid_x, global_centroid_y]

            logger.info(f"Step 4: Final block selected at ({best_top_row}, {best_left_col})")
            logger.info(f"  - Squared intensity score: {max_intensity_score:.1f}")
            logger.info(f"  - Regular intensity: {max_intensity:.1f}")
            logger.info(f"  - Aligned left column: {best_left_col}")
            logger.info(f"  - LED centroid position: ({global_centroid_x:.1f}, {global_centroid_y:.1f})")

            # Sample some pixel values from the selected block for debugging
            sample_block = combined_pattern[
                best_top_row : min(best_top_row + 10, best_top_row + self.block_size),
                best_left_col : min(best_left_col + 10, best_left_col + self.block_size),
            ]
            logger.info(f"  - Sample pixel values (top-left 10x10):")
            logger.info(f"    Max: {np.max(sample_block):.1f}, Mean: {np.mean(sample_block):.1f}")

            logger.info(f"=== Position detection complete for LED {led_id} ===")

            return best_top_row, best_left_col

        except Exception as e:
            logger.error(f"Failed to find brightest block for LED {led_id}: {e}")
            # Fallback to center
            top_row = max(0, (height - self.block_size) // 2)
            left_col_candidate = max(0, (width - self.block_size) // 2)
            left_col = self._align_to_pixel_boundary(left_col_candidate)

            # Store fallback position
            self.led_positions[led_id] = [width // 2, height // 2]

            return top_row, left_col

    def _reorder_to_rcm_spatial_ordering(self) -> Tuple[SingleBlockMixedSparseTensor, dict]:
        """
        Reorder captured patterns using RCM spatial ordering.

        Returns:
            Tuple of (reordered_mixed_tensor, led_spatial_mapping)
        """
        logger.info("Computing RCM spatial ordering for captured patterns...")

        # Compute RCM ordering using block positions
        rcm_order, inverse_order, expected_ata_diagonals = compute_rcm_ordering(self.block_positions, self.block_size)
        logger.info(f"Expected A^T A diagonals (from adjacency): {expected_ata_diagonals}")

        # Create mapping: physical_led_id -> rcm_ordered_matrix_index
        led_spatial_mapping = {original_id: rcm_pos for rcm_pos, original_id in enumerate(rcm_order)}

        # Create new mixed tensor with same configuration
        if self.use_uint8:
            tensor_dtype = cp.uint8
            output_dtype = cp.float32
        elif self.precision == "fp16":
            tensor_dtype = cp.float16
            output_dtype = cp.float16
        else:
            tensor_dtype = cp.float32
            output_dtype = cp.float32

        reordered_tensor = SingleBlockMixedSparseTensor(
            batch_size=self.led_count,
            channels=3,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            block_size=self.block_size,
            device="cpu",
            dtype=tensor_dtype,
            output_dtype=output_dtype,
        )

        # Copy all blocks from original tensor to RCM-ordered positions
        logger.info("Copying patterns to RCM-ordered tensor...")
        for original_led_id in range(self.led_count):
            rcm_led_id = led_spatial_mapping[original_led_id]

            for channel in range(3):
                # Get block from original tensor
                top_row, left_col = self.block_positions[original_led_id]

                # Get the block from the original mixed tensor
                try:
                    # Use get_block_info to retrieve the block data and position
                    block_info = self.mixed_tensor.get_block_info(original_led_id, channel)
                    block = block_info["values"]
                    stored_position = block_info["position"]

                    # Use the stored position (should match top_row, left_col)
                    stored_top, stored_left = stored_position

                    # Set in the reordered tensor at the RCM position using stored position
                    reordered_tensor.set_block(rcm_led_id, channel, stored_top, stored_left, block)

                except Exception as e:
                    logger.warning(f"Failed to copy block for LED {original_led_id}, channel {channel}: {e}")
                    # Create a zero block as fallback
                    zero_block = cp.zeros((self.block_size, self.block_size), dtype=tensor_dtype)
                    reordered_tensor.set_block(rcm_led_id, channel, top_row, left_col, zero_block)

        logger.info("RCM reordering completed")
        return reordered_tensor, led_spatial_mapping

    def save_patterns(self, output_path: str) -> bool:
        """
        Save captured diffusion patterns in mixed tensor format.
        Use tools/compute_matrices.py to generate ATA matrices afterward.

        Args:
            output_path: Path to save diffusion patterns (.npz format)

        Returns:
            True if save successful
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare metadata
            save_metadata = {
                "generator": "DiffusionPatternCapture",
                "format": "mixed_sparse_tensor",
                "led_count": self.led_count,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "channels": 3,
                "generation_timestamp": time.time(),
                "capture_fps": self.capture_fps,
                "wled_host": self.wled_host,
                "wled_port": self.wled_port,
                "block_size": self.block_size,
                "precision": self.precision,
                "use_uint8": self.use_uint8,
                "pattern_type": "captured_real",
                "intensity_variation": True,  # Real LEDs have natural variation
                "led_size_scaling": False,  # Not applicable for real capture
            }

            # Create led_ordering array: spatial_index -> physical_led_id
            # This is what the frame renderer will use to convert from spatial to physical order
            # Invert the spatial mapping: original spatial_mapping[physical_id] = spatial_index
            # We need: led_ordering[spatial_index] = physical_id
            led_ordering = np.zeros(self.led_count, dtype=np.int32)
            for physical_id, spatial_index in self.led_spatial_mapping.items():
                led_ordering[spatial_index] = physical_id

            # Save only essential data for matrix computation
            save_dict = {
                # LED information
                "led_positions": self.led_positions,
                "led_spatial_mapping": self.led_spatial_mapping,
                "led_ordering": led_ordering,  # spatial_index -> physical_led_id
                # Metadata
                "metadata": save_metadata,
                # Mixed tensor stored as nested element using to_dict()
                "mixed_tensor": self.mixed_tensor.to_dict(),
                # Failed LED capture tracking
                "failed_leds": self.failed_leds,
                "failed_led_details": self.failed_led_details,
            }

            np.savez_compressed(output_path, **save_dict)

            # Log file info
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB

            logger.info(f"Saved mixed tensor format to {output_path}")
            logger.info(f"File size: {file_size:.1f} MB")
            logger.info(
                f"Mixed tensor: {self.mixed_tensor.batch_size} LEDs, "
                f"{self.mixed_tensor.height}x{self.mixed_tensor.width}, "
                f"{self.mixed_tensor.block_size}x{self.mixed_tensor.block_size} blocks"
            )
            if self.failed_leds:
                logger.info(f"Failed LED captures saved: {len(self.failed_leds)} LEDs ({self.failed_leds})")

                # Log detailed failure breakdown
                color_failures = sum(
                    1 for details in self.failed_led_details.values() if details["failure_type"] == "color_detection"
                )
                luminance_failures = sum(
                    1 for details in self.failed_led_details.values() if details["failure_type"] == "luminance_check"
                )

                logger.info(
                    f"Failure breakdown: {color_failures} color detection failures, {luminance_failures} luminance check failures"
                )

                logger.info(f"Debug images saved for failed LEDs in debug_led_blocks/ with FAILED_ prefix")

                # Log individual LED failure details for analysis
                failed_led_set = set()
                for details in self.failed_led_details.values():
                    led_idx = details["led_idx"]
                    failed_led_set.add(led_idx)

                logger.info(f"Unique LEDs with failures: {sorted(failed_led_set)}")
                logger.info(f"Total failure instances (LED×channel): {len(self.failed_led_details)}")
            else:
                logger.info("All LEDs captured successfully")
            logger.info("Use tools/compute_matrices.py to generate ATA matrices for optimization")

            return True

        except Exception as e:
            logger.error(f"Failed to save mixed tensor format: {e}")
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
    parser.add_argument("--wled-host", required=False, default="wled.local", help="WLED controller hostname/IP")
    parser.add_argument("--wled-port", type=int, default=4048, help="WLED controller port")
    parser.add_argument("--camera-device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--usb", action="store_true", help="Use USB camera instead of CSI camera")
    parser.add_argument("--output", required=True, help="Output file path (.npz)")
    parser.add_argument("--capture-fps", type=float, default=10.0, help="Capture rate (fps)")
    parser.add_argument("--preview", action="store_true", help="Show live preview")
    parser.add_argument(
        "--crop-region",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Camera crop region (x y width height)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp32",
        help="Precision for mixed tensor storage (default: fp32)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Block size for mixed tensor storage (default: 64)",
    )
    parser.add_argument(
        "--uint8",
        action="store_true",
        help="Use uint8 format for memory efficiency and CUDA vectorization (recommended)",
    )
    parser.add_argument(
        "--flip-image",
        action="store_true",
        default=False,
        help="Flip image 180 degrees (for upside-down camera mounting)",
    )
    parser.add_argument(
        "--camera-config",
        help="Camera calibration JSON file from camera_calibration.py",
    )
    parser.add_argument(
        "--total-led-count",
        type=int,
        default=2600,
        help="Total number of LEDs in the system (for data packets, default: 2600)",
    )
    parser.add_argument(
        "--capture-led-count",
        type=int,
        help="Number of LEDs to capture (can be less than total for testing, default: same as total)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Debug mode: only capture LEDs with physical index multiples of 100 and save debug images",
    )
    parser.add_argument(
        "--gain",
        type=float,
        help="Manual camera gain value for consistent LED capture (recommended: use output from led_gain_calibrator.py)",
    )

    args = parser.parse_args()

    # Setup logging first, before any logger calls
    # Remove any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create new handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(f"capture_diffusion_{int(time.time())}.log")

    # Set formatting
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Set level on root logger - this will affect all loggers
    root_logger.setLevel(getattr(logging, args.log_level))

    # Re-get our module logger
    global logger
    logger = logging.getLogger(__name__)

    # Test that logging is working
    logger.info(f"Logging initialized at {args.log_level} level")

    # Validate block size
    if args.block_size < 32 or args.block_size > 256 or (args.block_size & (args.block_size - 1)) != 0:
        logger.error("Block size must be a power of 2 between 32 and 256")
        return 1

    # Validate output path
    output_path = Path(args.output)
    if output_path.suffix != ".npz":
        logger.error("Output file must have .npz extension")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load camera configuration if provided
    camera_device = args.camera_device
    crop_region = None
    flip_image = args.flip_image  # Start with command line value (default False unless --flip-image passed)
    total_led_count = args.total_led_count  # Total LEDs in system
    capture_led_count = args.capture_led_count  # LEDs to capture (can be subset)
    use_usb = args.usb  # Default from command line
    camera_resolution = None  # Will be set from config if available

    if args.camera_config:
        try:
            with open(args.camera_config) as f:
                camera_config = json.load(f)

            # Override camera device from config if not explicitly provided
            if hasattr(args, "camera_device") and args.camera_device == 0:  # Default value
                camera_device = camera_config.get("camera_device", 0)

            # Extract crop region from config
            crop_config = camera_config.get("crop_region")
            if crop_config:
                crop_region = (crop_config["x"], crop_config["y"], crop_config["width"], crop_config["height"])
                logger.info(f"Using crop region from config: {crop_region}")

            # Extract camera resolution from config
            resolution_config = camera_config.get("camera_resolution")
            if resolution_config:
                camera_resolution = (resolution_config["width"], resolution_config["height"])
                logger.info(f"Using camera resolution from config: {camera_resolution}")

            # Check if config specifies USB camera
            if camera_config.get("use_usb", False):
                use_usb = True
                logger.info("Using USB camera mode from config")

            # Check if config specifies flip image
            # Priority: command line flag > config file > default (False)
            # If --flip-image was NOT passed on command line, use config value if present
            if not args.flip_image and "flip_image" in camera_config:
                flip_image = camera_config.get("flip_image", False)
                logger.info(f"Using flip image mode from config: {flip_image}")
            elif args.flip_image:
                logger.info("Using flip image mode from command line flag")

            logger.info(f"Loaded camera configuration from {args.camera_config}")

        except Exception as e:
            logger.error(f"Failed to load camera config {args.camera_config}: {e}")
            return 1

    # Override crop region if provided via command line
    if args.crop_region:
        crop_region = tuple(args.crop_region)
        logger.info(f"Using crop region from command line: {crop_region}")

    # Create capture tool
    capture_tool = DiffusionPatternCapture(
        wled_host=args.wled_host,
        wled_port=args.wled_port,
        camera_device=camera_device,
        capture_fps=args.capture_fps,
        crop_region=crop_region,
        block_size=args.block_size,
        precision=args.precision,
        use_uint8=args.uint8,
        flip_image=flip_image,
        total_led_count=total_led_count,
        capture_led_count=capture_led_count,
        debug_mode=args.debug_mode,
        manual_gain=args.gain,
        use_usb=use_usb,
        camera_resolution=camera_resolution,
    )

    try:
        # Initialize
        if not capture_tool.initialize():
            logger.error("Failed to initialize capture tool")
            return 1

        # Estimate capture time
        actual_led_count = capture_tool.led_count
        total_captures = actual_led_count * 3
        estimated_time_minutes = (total_captures * (1.0 / args.capture_fps)) / 60
        logger.info(f"Capturing {actual_led_count} LEDs")
        logger.info(f"Estimated capture time: {estimated_time_minutes:.1f} minutes")

        # Start capture
        if not capture_tool.capture_patterns(preview=args.preview):
            logger.error("Capture failed")
            return 1

        # Save patterns in modern mixed tensor format
        if not capture_tool.save_patterns(str(output_path)):
            logger.error("Failed to save patterns")
            return 1

        logger.info("Diffusion pattern capture completed successfully!")
        logger.info("Use tools/compute_matrices.py to generate ATA matrices for optimization")
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
