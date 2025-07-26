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
    python capture_diffusion_patterns.py --wled-host 192.168.1.100 --camera-device 0 --output patterns.npz --preview
    python capture_diffusion_patterns.py --wled-host 192.168.1.100 --output patterns.npz --block-size 64 --precision fp16 --uint8
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

try:
    import cupy as cp
except ImportError:
    # Fallback for systems without CUDA
    import numpy as cp

# Import LED position utilities (add project root to path)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from tools.led_position_utils import calculate_block_positions

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
    from consumer.wled_client import WLEDClient, WLEDConfig
    from utils.diagonal_ata_matrix import DiagonalATAMatrix
    from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
    from utils.spatial_ordering import compute_rcm_ordering
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
        capture_fps: float = 10.0,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
        block_size: int = 64,
        precision: str = "fp32",
        use_uint8: bool = False,
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
        """
        self.wled_host = wled_host
        self.wled_port = wled_port
        self.capture_fps = capture_fps
        self.capture_interval = 1.0 / capture_fps
        self.block_size = block_size
        self.precision = precision
        self.use_uint8 = use_uint8

        # Initialize WLED client
        wled_config = WLEDConfig(host=wled_host, port=wled_port, led_count=LED_COUNT, max_fps=60.0)
        self.wled_client = WLEDClient(wled_config)

        # Initialize camera
        self.camera = CameraCapture(camera_device, crop_region)

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
            batch_size=LED_COUNT,
            channels=3,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            block_size=block_size,
            device="cpu",  # Use CPU for capture
            dtype=tensor_dtype,
            output_dtype=output_dtype,
        )

        # Storage for LED positions and block positions
        self.led_positions = np.zeros((LED_COUNT, 2), dtype=np.float32)
        self.block_positions = np.zeros((LED_COUNT, 2), dtype=np.int32)  # Top-left corner of each block
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
            total_captures = LED_COUNT * 3  # 3 channels per LED
            logger.info(f"Starting capture of {total_captures} diffusion patterns")

            # Turn off all LEDs initially
            self.wled_client.set_solid_color(0, 0, 0)
            time.sleep(0.5)  # Allow LEDs to turn off

            for led_idx in range(LED_COUNT):
                for channel_idx in range(3):  # R, G, B channels
                    capture_num = led_idx * 3 + channel_idx + 1

                    logger.info(f"Capturing LED {led_idx}, Channel {channel_idx} ({capture_num}/{total_captures})")

                    # Create LED data array (all off except current LED/channel)
                    led_data = np.zeros((LED_COUNT, 3), dtype=np.uint8)
                    led_data[led_idx, channel_idx] = 255  # Full brightness for this LED/channel

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
                        self._show_preview(frame, led_idx, channel_idx, capture_num, total_captures)

                    # Progress update
                    if capture_num % 100 == 0:
                        progress = (capture_num / total_captures) * 100
                        logger.info(f"Progress: {progress:.1f}% ({capture_num}/{total_captures})")

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
        Find optimal block position for a captured LED pattern.

        Args:
            pattern: Captured pattern (height, width, 3)
            led_id: LED ID for position estimation

        Returns:
            Tuple of (top_row, left_col) for block position (aligned to 4-pixel boundary)
        """
        try:
            # Combine all three color channels for intensity analysis
            combined_pattern = np.max(pattern, axis=2).astype(np.float32)

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

                # Store estimated LED position
                self.led_positions[led_id] = [centroid_x, centroid_y]

                # Calculate block position centered on LED
                top_row = max(0, min(height - self.block_size, int(centroid_y - self.block_size // 2)))
                left_col_candidate = max(0, min(width - self.block_size, int(centroid_x - self.block_size // 2)))

                # Align to 4-pixel boundary
                left_col = self._align_to_pixel_boundary(left_col_candidate)

                return top_row, left_col
            else:
                # Fallback for failed patterns - use center of frame
                logger.warning(f"Failed to find pattern for LED {led_id}, using center")
                self.led_positions[led_id] = [width // 2, height // 2]

                top_row = max(0, (height - self.block_size) // 2)
                left_col_candidate = max(0, (width - self.block_size) // 2)
                left_col = self._align_to_pixel_boundary(left_col_candidate)

                return top_row, left_col

        except Exception as e:
            logger.error(f"Failed to find block position for LED {led_id}: {e}")
            # Fallback to center
            top_row = max(0, (FRAME_HEIGHT - self.block_size) // 2)
            left_col_candidate = max(0, (FRAME_WIDTH - self.block_size) // 2)
            left_col = self._align_to_pixel_boundary(left_col_candidate)

            # Store fallback position
            self.led_positions[led_id] = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]

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
            batch_size=LED_COUNT,
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
        for original_led_id in range(LED_COUNT):
            rcm_led_id = led_spatial_mapping[original_led_id]

            for channel in range(3):
                # Get block from original tensor
                top_row, left_col = self.block_positions[original_led_id]

                # Get the block from the original mixed tensor
                try:
                    # Use the mixed tensor's internal method to get the block
                    block = self.mixed_tensor.get_block(original_led_id, channel, top_row, left_col)

                    # Set in the reordered tensor at the RCM position
                    reordered_tensor.set_block(rcm_led_id, channel, top_row, left_col, block)

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
        dia_matrix = DiagonalATAMatrix(LED_COUNT, crop_size=self.block_size, output_dtype=output_dtype)

        # For captured data, we need to compute A^T @ A from actual diffusion patterns
        # This requires converting the mixed tensor to a sparse matrix first
        logger.info("Converting mixed tensor to sparse matrix for DIA matrix computation...")

        # Convert mixed tensor to equivalent sparse CSC matrix
        sparse_matrix = self._mixed_tensor_to_sparse_matrix()

        # Build DIA matrix from sparse matrix
        dia_matrix.build_from_diffusion_matrix(sparse_matrix)

        logger.info(
            f"DiagonalATAMatrix built: {LED_COUNT} LEDs, bandwidth={dia_matrix.bandwidth}, k={dia_matrix.k} diagonals"
        )

        return dia_matrix

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

        for led_id in range(LED_COUNT):
            for channel in range(3):
                try:
                    # Get block position
                    top_row, left_col = self.block_positions[led_id]

                    # Get block from mixed tensor
                    block = self.mixed_tensor.get_block(led_id, channel, top_row, left_col)

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

                            # Value
                            value = float(block_np[br, bc])

                            rows.append(pixel_idx)
                            cols.append(matrix_col_idx)
                            values.append(value)

                except Exception as e:
                    logger.warning(f"Failed to process LED {led_id}, channel {channel}: {e}")
                    continue

        # Create CSC matrix
        sparse_matrix = sp.csc_matrix(
            (values, (rows, cols)),
            shape=(pixels_per_channel, LED_COUNT * 3),
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

            # Generate DiagonalATAMatrix (DIA format) from mixed tensor
            logger.info("Generating DiagonalATAMatrix (DIA format)...")
            dia_matrix = self._generate_dia_matrix()

            # Prepare metadata matching synthetic generation tool format
            save_metadata = {
                "generator": "DiffusionPatternCapture",
                "format": "led_diffusion_csc_with_mixed_tensor",
                "led_count": LED_COUNT,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "channels": 3,
                "matrix_shape": [FRAME_HEIGHT * FRAME_WIDTH, LED_COUNT * 3],  # Equivalent sparse matrix shape
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
            led_ordering = np.zeros(LED_COUNT, dtype=np.int32)
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
                # DIA format A^T @ A matrix
                "dia_matrix": dia_matrix.to_dict(),
            }

            np.savez_compressed(output_path, **save_dict)

            # Log file info
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB

            logger.info(f"Saved mixed tensor, DIA matrix, and metadata to {output_path}")
            logger.info(f"File size: {file_size:.1f} MB")
            logger.info("Mixed tensor format: SingleBlockMixedSparseTensor")
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
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
        block_size=args.block_size,
        precision=args.precision,
        use_uint8=args.uint8,
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
