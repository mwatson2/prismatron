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
7. Generating DiagonalATAMatrix for optimization
8. Saving in modern mixed tensor format compatible with the optimization engine

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
    python capture_diffusion_patterns.py --wled-host 192.168.7.140 --camera-config camera.json --output patterns.npz --flip-image
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

from src.const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from src.consumer.wled_client import WLEDClient, WLEDConfig
from src.utils.dense_ata_matrix import DenseATAMatrix
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix
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
    ):
        """
        Initialize camera capture.

        Args:
            device_id: Camera device ID (usually 0 for default camera)
            crop_region: Optional crop region (x, y, width, height) for prismatron area
            flip_image: Whether to flip the image 180 degrees (for upside-down camera mounting)
        """
        self.device_id = device_id
        self.crop_region = crop_region
        self.flip_image = flip_image
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_width = 0
        self.camera_height = 0
        self.gst_process = None
        self.temp_fifo = None

    def initialize(self) -> bool:
        """Initialize camera using OpenCV with GStreamer support."""
        try:
            logger.info(f"Using OpenCV with GStreamer support for camera {self.device_id}")

            # Use the working GStreamer pipeline
            gstreamer_pipeline = (
                f"nvarguscamerasrc sensor-id={self.device_id} ! "
                "nvvidconv ! video/x-raw,format=I420 ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=1"
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

            # Warm up camera with safer approach
            logger.info("Warming up camera...")
            for i in range(3):
                try:
                    logger.info(f"Reading warmup frame {i}...")
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning(f"Warmup frame {i} failed to read")
                    else:
                        logger.info(
                            f"Warmup frame {i} successful, shape: {frame.shape if frame is not None else 'None'}"
                        )
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
        use_uint8: bool = False,
        flip_image: bool = False,
        led_count: Optional[int] = None,
        debug_mode: bool = False,
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
            led_count: Number of LEDs to capture (default: LED_COUNT from config)
            debug_mode: Debug mode - only capture LEDs with physical index multiples of 100
        """
        self.wled_host = wled_host
        self.wled_port = wled_port
        self.capture_fps = capture_fps
        self.capture_interval = 1.0 / capture_fps
        self.block_size = block_size
        self.precision = precision
        self.use_uint8 = use_uint8
        self.debug_mode = debug_mode

        # Set LED count (use override or default from config)
        self.led_count = led_count if led_count is not None else LED_COUNT

        # Create debug LED list if in debug mode (multiples of 100)
        if self.debug_mode:
            self.debug_leds = [i for i in range(0, self.led_count, 100)]
            logger.info(f"Debug mode: will capture {len(self.debug_leds)} LEDs: {self.debug_leds}")
        else:
            self.debug_leds = None

        # Initialize WLED client
        wled_config = WLEDConfig(host=wled_host, port=wled_port, led_count=self.led_count, max_fps=60.0)
        self.wled_client = WLEDClient(wled_config)

        # Initialize camera
        self.camera = CameraCapture(camera_device, crop_region, flip_image)

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
            self.wled_client.set_solid_color(0, 0, 0)
            time.sleep(0.5)  # Allow LEDs to turn off

            capture_count = 0
            for led_idx in led_list:
                for channel_idx in range(3):  # R, G, B channels
                    capture_count += 1

                    logger.info(f"Capturing LED {led_idx}, Channel {channel_idx} ({capture_count}/{total_captures})")

                    # Create LED data array (all off except current LED/channel)
                    led_data = np.zeros((self.led_count, 3), dtype=np.uint8)
                    led_data[led_idx, channel_idx] = 255  # Full brightness for this LED/channel

                    # Send to WLED
                    result = self.wled_client.send_led_data(led_data)
                    if not result.success:
                        logger.warning(f"Failed to send LED data: {result.errors}")
                        continue

                    # Wait for LED to update and stabilize (longer for first LED)
                    if led_idx == 0 and channel_idx == 0:
                        time.sleep(1.0)  # Extra delay for first capture
                    else:
                        time.sleep(self.capture_interval)

                    # Capture frame
                    frame = self.camera.capture_frame()
                    if frame is None:
                        logger.warning(f"Failed to capture frame for LED {led_idx}, channel {channel_idx}")
                        continue

                    # Convert frame to appropriate format
                    if self.use_uint8:
                        # Keep as uint8 [0-255] for memory efficiency
                        pattern_data = frame.astype(np.uint8)
                    elif self.precision == "fp16":
                        # Convert to fp16 [0-1] range
                        pattern_data = frame.astype(np.float16) / 255.0
                    else:
                        # Convert to fp32 [0-1] range
                        pattern_data = frame.astype(np.float32) / 255.0

                    # For first channel of each LED, determine optimal block position
                    if channel_idx == 0:
                        top_row, left_col = self._find_optimal_block_position(frame, led_idx)
                        self.block_positions[led_idx] = [top_row, left_col]

                        # Save debug image if in debug mode
                        if self.debug_mode:
                            self._save_debug_image(frame, led_idx, top_row, left_col)
                    else:
                        # Use same block position for other channels of the same LED
                        top_row, left_col = self.block_positions[led_idx]

                    # Extract block from full pattern
                    block = pattern_data[
                        top_row : top_row + self.block_size, left_col : left_col + self.block_size, channel_idx
                    ]

                    # Ensure block is the right size (pad with zeros if needed)
                    if block.shape != (self.block_size, self.block_size):
                        padded_block = np.zeros((self.block_size, self.block_size), dtype=pattern_data.dtype)
                        h, w = min(block.shape[0], self.block_size), min(block.shape[1], self.block_size)
                        padded_block[:h, :w] = block[:h, :w]
                        block = padded_block

                    # Store block in mixed tensor
                    block_cupy = cp.asarray(block)
                    self.mixed_tensor.set_block(led_idx, channel_idx, top_row, left_col, block_cupy)

                    # Show preview if requested
                    if preview:
                        self._show_preview(frame, led_idx, channel_idx, capture_count, total_captures)

                    # Progress update
                    if capture_count % 100 == 0:
                        progress = (capture_count / total_captures) * 100
                        logger.info(f"Progress: {progress:.1f}% ({capture_count}/{total_captures})")

            # Turn off all LEDs
            self.wled_client.set_solid_color(0, 0, 0)

            logger.info("Diffusion pattern capture completed successfully")

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

            # Save image
            filename = debug_dir / f"led_{led_id:04d}.jpg"
            cv2.imwrite(str(filename), debug_frame)
            logger.info(f"Saved debug image for LED {led_id}: {filename}")

        except Exception as e:
            logger.warning(f"Failed to save debug image for LED {led_id}: {e}")

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
            # Combine all three color channels for intensity analysis
            combined_pattern = np.max(pattern, axis=2).astype(np.float32)
            height, width = combined_pattern.shape

            # Step 1: Tile the image into non-overlapping 64x64 blocks
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

            # Step 2: Find the 128x128 region with maximum local contrast for LED detection
            max_contrast_score = 0
            best_region_top = 0
            best_region_left = 0
            region_size = 2 * self.block_size  # 128x128 region

            # Search all possible 128x128 regions for local contrast
            for region_top in range(0, height - region_size + 1, self.block_size):
                for region_left in range(0, width - region_size + 1, self.block_size):
                    # Ensure region stays within image bounds
                    if region_top + region_size > height or region_left + region_size > width:
                        continue

                    # Extract 128x128 region
                    region = combined_pattern[
                        region_top : region_top + region_size, region_left : region_left + region_size
                    ]

                    # Calculate local contrast within this large region
                    region_max = np.max(region)
                    region_mean = np.mean(region)

                    # Skip very dark regions
                    if region_max < 25:
                        continue

                    # Find the peak region (top 10% of pixels in 128x128 region)
                    top_10_percent_threshold = np.percentile(region, 90)
                    peak_pixels = region >= top_10_percent_threshold
                    peak_count = np.sum(peak_pixels)

                    if peak_count == 0:
                        continue

                    # Calculate peak intensity
                    peak_intensity = np.mean(region[peak_pixels])

                    # Local contrast: how much does the peak stand out from region background
                    background_pixels = region < top_10_percent_threshold
                    if np.sum(background_pixels) > 0:
                        background_intensity = np.mean(region[background_pixels])
                        local_contrast = peak_intensity - background_intensity
                    else:
                        local_contrast = peak_intensity - region_mean

                    # Score: favor high contrast with reasonable peak size
                    # LEDs can saturate large areas, so don't penalize too much for size
                    max_reasonable_peak = region_size * region_size * 0.3  # Up to 30% of region
                    if local_contrast > 15 and peak_count <= max_reasonable_peak:
                        # Compactness bonus: smaller peaks get slight bonus, but not too much
                        compactness_factor = min(2.0, max_reasonable_peak / (peak_count + 1))
                        contrast_score = local_contrast * peak_intensity * compactness_factor
                    else:
                        contrast_score = 0

                    if contrast_score > max_contrast_score:
                        max_contrast_score = contrast_score
                        best_region_top = region_top
                        best_region_left = region_left

            # Step 3: Within the best 128x128 region, find the optimal 64x64 block
            # Expand search to include all blocks whose centers fall within the best region
            half_block = self.block_size // 2
            search_top = max(0, best_region_top - half_block)
            search_left = max(0, best_region_left - half_block)
            search_bottom = min(height, best_region_top + region_size + half_block)
            search_right = min(width, best_region_left + region_size + half_block)

            # Step 4: Exhaustive search for brightest 64x64 block using squared intensity bias
            max_intensity_score = 0
            best_top_row = search_top
            best_left_col = search_left

            for top_row in range(search_top, search_bottom - self.block_size + 1, 4):
                for left_col in range(search_left, search_right - self.block_size + 1, 4):
                    # Ensure block stays within image bounds
                    if top_row + self.block_size > height or left_col + self.block_size > width:
                        continue

                    # Extract the current 64x64 block
                    block = combined_pattern[top_row : top_row + self.block_size, left_col : left_col + self.block_size]

                    # Calculate squared intensity sum to bias towards brighter pixels and suppress noise
                    # This heavily favors bright LED pixels over accumulated dim noise
                    squared_intensities = np.square(block)
                    intensity_score = np.sum(squared_intensities)

                    # Keep track of the block with highest squared intensity score
                    if intensity_score > max_intensity_score:
                        max_intensity_score = intensity_score
                        best_top_row = top_row
                        best_left_col = left_col
                        # Also calculate regular intensity for logging
                        max_intensity = np.sum(block)

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

            logger.info(
                f"LED {led_id}: best contrast region at ({best_region_top}, {best_region_left}) score={max_contrast_score:.1f}, "
                f"brightest block at ({best_top_row}, {best_left_col}) intensity={max_intensity:.1f}, "
                f"search_region=({search_top},{search_left})-({search_bottom},{search_right})"
            )

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

    def _generate_dia_matrix(self) -> DiagonalATAMatrix:
        """
        Generate DiagonalATAMatrix from captured mixed tensor.

        Returns:
            DiagonalATAMatrix object with 3D DIA format
        """
        logger.info("Building DiagonalATAMatrix from captured patterns...")

        # Determine output dtype based on precision
        if self.use_uint8:
            output_dtype = cp.float32  # Computation in fp32, storage in uint8
        elif self.precision == "fp16":
            output_dtype = cp.float16
        else:
            output_dtype = cp.float32

        # Create DiagonalATAMatrix instance
        dia_matrix = DiagonalATAMatrix(self.led_count, crop_size=self.block_size, output_dtype=output_dtype)

        # For captured data, we need to compute A^T @ A from actual diffusion patterns
        # This requires converting the mixed tensor to a sparse matrix first
        logger.info("Converting mixed tensor to sparse matrix for DIA matrix computation...")

        # Convert mixed tensor to equivalent sparse CSC matrix
        sparse_matrix = self._mixed_tensor_to_sparse_matrix()

        # Build DIA matrix from sparse matrix - this now uses proper diagonal filtering
        dia_matrix.build_from_diffusion_matrix(sparse_matrix)

        logger.info(
            f"DiagonalATAMatrix built: {self.led_count} LEDs, bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k} diagonals"
        )

        return dia_matrix

    def _generate_dense_ata_matrix(self) -> DenseATAMatrix:
        """
        Generate DenseATAMatrix from captured mixed tensor.

        Returns:
            DenseATAMatrix object
        """
        logger.info("Building DenseATAMatrix from captured patterns...")

        # Determine dtypes based on precision
        if self.precision == "fp16":
            # Mixed precision: FP16 storage, FP32 computation
            storage_dtype = cp.float16
            output_dtype = cp.float32
            logger.info("Using mixed precision: FP16 storage with FP32 computation")
        else:
            # Standard FP32 for both storage and computation
            storage_dtype = cp.float32
            output_dtype = cp.float32
            logger.info("Using FP32 storage and computation")

        # Create DenseATAMatrix instance
        dense_matrix = DenseATAMatrix(
            led_count=self.led_count, channels=3, storage_dtype=storage_dtype, output_dtype=output_dtype
        )

        # Convert mixed tensor to sparse matrix for computation
        logger.info("Converting mixed tensor to sparse matrix for dense ATA computation...")
        sparse_matrix = self._mixed_tensor_to_sparse_matrix()

        # Build dense matrix from sparse matrix
        dense_matrix.build_from_diffusion_matrix(sparse_matrix)

        logger.info(f"DenseATAMatrix built: {self.led_count} LEDs, {dense_matrix.memory_mb:.1f}MB")
        return dense_matrix

    def _choose_ata_format(self) -> str:
        """
        Choose optimal ATA matrix format based on captured pattern characteristics.

        Returns:
            "dia" for diagonal format, "dense" for dense format
        """
        # For format selection, we need to estimate the bandwidth from LED positions
        # Import the function to calculate block positions
        from tools.led_position_utils import calculate_block_positions

        # Calculate block positions for captured LEDs
        block_positions = calculate_block_positions(self.led_positions, self.block_size, FRAME_WIDTH, FRAME_HEIGHT)

        # Estimate bandwidth using RCM ordering
        _, _, expected_diagonals = compute_rcm_ordering(block_positions, self.block_size)

        # Heuristics for format selection (similar to synthetic generator):
        # - Use dense if bandwidth > 80% of matrix size
        # - Use dense if expected diagonals > LED count (very dense)
        # - For captured patterns, be more conservative since they tend to be denser

        bandwidth_ratio = expected_diagonals / (2 * self.led_count - 1) if self.led_count > 1 else 1.0

        # Memory estimates (rough)
        dia_memory_mb = (3 * expected_diagonals * self.led_count * 4) / (1024 * 1024)  # FP32
        dense_memory_mb = (3 * self.led_count * self.led_count * 4) / (1024 * 1024)  # FP32

        logger.info(f"ATA format selection analysis for captured patterns:")
        logger.info(f"  LED count: {self.led_count}")
        logger.info(f"  Expected diagonals: {expected_diagonals}")
        logger.info(f"  Bandwidth ratio: {bandwidth_ratio:.3f}")
        logger.info(f"  Estimated DIA memory: {dia_memory_mb:.1f}MB")
        logger.info(f"  Estimated dense memory: {dense_memory_mb:.1f}MB")

        # Decision logic - captured patterns tend to be denser, so use lower thresholds
        if bandwidth_ratio > 0.6:  # Lower threshold for captured data
            format_choice = "dense"
            reason = f"High bandwidth ratio ({bandwidth_ratio:.3f} > 0.6)"
        elif expected_diagonals > self.led_count * 0.8:  # More conservative threshold
            format_choice = "dense"
            reason = f"Many diagonals ({expected_diagonals} > {self.led_count * 0.8:.0f})"
        elif self.led_count < 1000:  # Lower threshold for small matrices
            format_choice = "dense"
            reason = f"Small matrix ({self.led_count} LEDs)"
        elif dense_memory_mb < dia_memory_mb * 1.5:  # Dense is only 50% more memory
            format_choice = "dense"
            reason = f"Dense memory overhead acceptable ({dense_memory_mb:.1f}MB vs {dia_memory_mb:.1f}MB)"
        else:
            format_choice = "dia"
            reason = f"Sparse matrix benefits from DIA format"

        logger.info(f"  Selected format: {format_choice.upper()} ({reason})")
        return format_choice

    def _mixed_tensor_to_sparse_matrix(self) -> sp.csc_matrix:
        """
        Convert mixed tensor to equivalent sparse CSC matrix for DIA matrix computation.

        Returns:
            Sparse CSC matrix (pixels, leds*3) equivalent to the mixed tensor
        """
        logger.info("Converting mixed tensor to sparse CSC matrix...")

        # Prepare sparse matrix data structures
        rows = []
        cols = []
        values = []

        pixels_per_channel = FRAME_HEIGHT * FRAME_WIDTH

        for led_id in range(self.led_count):
            for channel in range(3):
                try:
                    # Get block position
                    top_row, left_col = self.block_positions[led_id]

                    # Get block from mixed tensor using get_block_info
                    block_info = self.mixed_tensor.get_block_info(led_id, channel)
                    block = block_info["values"]

                    # Convert to numpy if needed
                    if hasattr(block, "get"):  # CuPy array
                        block_np = cp.asnumpy(block)
                    else:
                        block_np = block

                    # Find non-zero elements in the block
                    block_rows, block_cols = np.nonzero(block_np)

                    # Convert block coordinates to global pixel coordinates
                    for br, bc in zip(block_rows, block_cols):
                        global_row = top_row + br
                        global_col = left_col + bc

                        # Check bounds
                        if global_row < FRAME_HEIGHT and global_col < FRAME_WIDTH:
                            # Flatten pixel index
                            pixel_idx = global_row * FRAME_WIDTH + global_col

                            # Column index for this LED/channel
                            matrix_col_idx = led_id * 3 + channel

                            # Value - normalize uint8 to [0,1] range if needed
                            raw_value = float(block_np[br, bc])
                            if self.use_uint8:
                                # Mixed tensor stores uint8 [0,255], normalize to [0,1] for diffusion matrix
                                value = raw_value / 255.0
                            else:
                                # Mixed tensor stores float32 [0,1], use directly
                                value = raw_value

                            rows.append(pixel_idx)
                            cols.append(matrix_col_idx)
                            values.append(value)

                except Exception as e:
                    logger.warning(f"Failed to process LED {led_id}, channel {channel}: {e}")
                    continue

        # Create CSC matrix
        sparse_matrix = sp.csc_matrix(
            (values, (rows, cols)),
            shape=(pixels_per_channel, self.led_count * 3),
            dtype=np.float32,
        )

        # Clean up
        sparse_matrix.eliminate_zeros()
        sparse_matrix = sparse_matrix.tocsc()

        logger.info(f"Created sparse matrix: shape {sparse_matrix.shape}, nnz {sparse_matrix.nnz}")
        return sparse_matrix

    def save_patterns(self, output_path: str) -> bool:
        """
        Save captured diffusion patterns to file in modern mixed tensor format.

        Args:
            output_path: Path to save diffusion patterns (.npz format)

        Returns:
            True if save successful
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Choose optimal ATA matrix format based on captured pattern characteristics
            ata_format = self._choose_ata_format()

            if ata_format == "dense":
                # Generate DenseATAMatrix
                logger.info("Generating DenseATAMatrix (dense format)...")
                ata_matrix = self._generate_dense_ata_matrix()
                save_dict_key = "dense_ata_matrix"
            else:
                # Generate DiagonalATAMatrix (DIA format)
                logger.info("Generating DiagonalATAMatrix (DIA format)...")
                ata_matrix = self._generate_dia_matrix()
                save_dict_key = "dia_matrix"

            # Prepare metadata matching synthetic generation tool format
            save_metadata = {
                "generator": "DiffusionPatternCapture",
                "format": "led_diffusion_csc_with_mixed_tensor",
                "led_count": self.led_count,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "channels": 3,
                "matrix_shape": [FRAME_HEIGHT * FRAME_WIDTH, self.led_count * 3],  # Equivalent sparse matrix shape
                "nnz": 0,  # Will be calculated if needed
                "sparsity_percent": 0.0,  # Will be calculated if needed
                "sparsity_threshold": 0.0,  # Not applicable for captured data
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

            # Save everything in a single NPZ file (matching synthetic tool format)
            save_dict = {
                # LED information
                "led_positions": self.led_positions,
                "led_spatial_mapping": self.led_spatial_mapping,
                "led_ordering": led_ordering,  # New: spatial_index -> physical_led_id
                # Metadata
                "metadata": save_metadata,
                # Mixed tensor stored as nested element using to_dict()
                "mixed_tensor": self.mixed_tensor.to_dict(),
                # ATA matrix in chosen format
                save_dict_key: ata_matrix.to_dict(),
            }

            np.savez_compressed(output_path, **save_dict)

            # Log file info
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB

            logger.info(f"Saved mixed tensor and {ata_format.upper()} ATA matrix to {output_path}")
            logger.info(f"File size: {file_size:.1f} MB")
            logger.info("Mixed tensor format: SingleBlockMixedSparseTensor")

            if ata_format == "dense":
                logger.info(f"Dense ATA matrix: {ata_matrix.led_count} LEDs, {ata_matrix.memory_mb:.1f}MB")
            else:
                logger.info(
                    f"DIA matrix: {ata_matrix.led_count} LEDs, bandwidth={ata_matrix.bandwidth}, k={ata_matrix.k} diagonals"
                )
            logger.info(
                f"Mixed tensor: {self.mixed_tensor.batch_size} LEDs, "
                f"{self.mixed_tensor.height}x{self.mixed_tensor.width}, "
                f"{self.mixed_tensor.block_size}x{self.mixed_tensor.block_size} blocks"
            )
            logger.info(
                f"DIA matrix: {dia_matrix.led_count} LEDs, bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k} diagonals"
            )
            storage_shape = dia_matrix.dia_data_cpu.shape if dia_matrix.dia_data_cpu is not None else "None"
            logger.info(f"DIA matrix storage shape: {storage_shape}")
            logger.info("Use compute_ata_inverse.py tool to add ATA inverse matrices for optimization")

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
    parser.add_argument("--wled-host", required=True, help="WLED controller hostname/IP")
    parser.add_argument("--wled-port", type=int, default=21324, help="WLED controller port")
    parser.add_argument("--camera-device", type=int, default=0, help="Camera device ID")
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
        help="Flip image 180 degrees (for upside-down camera mounting)",
    )
    parser.add_argument(
        "--camera-config",
        help="Camera calibration JSON file from camera_calibration.py",
    )
    parser.add_argument(
        "--led-count",
        type=int,
        help=f"Number of LEDs to capture (default: {LED_COUNT} from config)",
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

    args = parser.parse_args()

    # Validate block size
    if args.block_size < 32 or args.block_size > 256 or (args.block_size & (args.block_size - 1)) != 0:
        logger.error("Block size must be a power of 2 between 32 and 256")
        return 1

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

    # Load camera configuration if provided
    camera_device = args.camera_device
    crop_region = None
    flip_image = args.flip_image
    led_count = args.led_count  # Use command line override or None for default

    if args.camera_config:
        try:
            with open(args.camera_config, "r") as f:
                camera_config = json.load(f)

            # Override camera device from config if not explicitly provided
            if hasattr(args, "camera_device") and args.camera_device == 0:  # Default value
                camera_device = camera_config.get("camera_device", 0)

            # Extract crop region from config
            crop_config = camera_config.get("crop_region")
            if crop_config:
                crop_region = (crop_config["x"], crop_config["y"], crop_config["width"], crop_config["height"])
                logger.info(f"Using crop region from config: {crop_region}")

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
        led_count=led_count,
        debug_mode=args.debug_mode,
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
        logger.info("Use compute_ata_inverse.py tool to add ATA inverse matrices for optimization")
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
