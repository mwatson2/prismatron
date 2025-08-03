#!/usr/bin/env python3
"""
Example of enhanced camera controls for capture_diffusion_patterns.py

This shows how to add manual exposure, gain, and other camera controls
to improve LED pattern capture quality.
"""

import logging
from typing import Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


class EnhancedCameraCapture:
    """Enhanced camera capture with manual exposure and gain controls."""

    def __init__(
        self,
        device_id: int = 0,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
        flip_image: bool = False,
        # New camera control parameters
        manual_exposure: Optional[float] = None,  # Exposure time in seconds (e.g., 0.01 = 10ms)
        manual_gain: Optional[float] = None,  # Gain value (e.g., 1.0-16.0)
        isp_digital_gain: Optional[float] = None,  # ISP digital gain (1.0-8.0)
        exposure_compensation: float = 0.0,  # Exposure compensation (-2.0 to 2.0)
        saturation: float = 1.0,  # Color saturation (0.0 to 2.0)
        ae_lock: bool = False,  # Auto-exposure lock
        awb_lock: bool = False,  # Auto white balance lock
        antibanding_mode: int = 1,  # 0=off, 1=auto, 2=50Hz, 3=60Hz
    ):
        """
        Initialize enhanced camera capture with manual controls.

        Args:
            device_id: Camera device ID (usually 0 for default camera)
            crop_region: Optional crop region (x, y, width, height) for prismatron area
            flip_image: Whether to flip the image 180 degrees
            manual_exposure: Manual exposure time in seconds (None for auto)
            manual_gain: Manual gain value (None for auto)
            isp_digital_gain: ISP digital gain multiplier
            exposure_compensation: Exposure compensation adjustment
            saturation: Color saturation multiplier
            ae_lock: Lock auto-exposure
            awb_lock: Lock auto white balance
            antibanding_mode: Anti-banding frequency mode
        """
        self.device_id = device_id
        self.crop_region = crop_region
        self.flip_image = flip_image

        # Camera control parameters
        self.manual_exposure = manual_exposure
        self.manual_gain = manual_gain
        self.isp_digital_gain = isp_digital_gain
        self.exposure_compensation = exposure_compensation
        self.saturation = saturation
        self.ae_lock = ae_lock
        self.awb_lock = awb_lock
        self.antibanding_mode = antibanding_mode

        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_width = 0
        self.camera_height = 0

    def _build_gstreamer_pipeline(self) -> str:
        """Build GStreamer pipeline with camera controls."""
        # Base pipeline
        pipeline_parts = [f"nvarguscamerasrc sensor-id={self.device_id}"]

        # Add manual exposure control
        if self.manual_exposure is not None:
            # Convert seconds to nanoseconds
            exposure_ns = int(self.manual_exposure * 1_000_000_000)
            pipeline_parts.append(f'exposuretimerange="{exposure_ns} {exposure_ns}"')
            logger.info(f"Setting manual exposure: {self.manual_exposure}s ({exposure_ns}ns)")

        # Add manual gain control
        if self.manual_gain is not None:
            pipeline_parts.append(f'gainrange="{self.manual_gain} {self.manual_gain}"')
            logger.info(f"Setting manual gain: {self.manual_gain}")

        # Add ISP digital gain
        if self.isp_digital_gain is not None:
            pipeline_parts.append(f'ispdigitalgainrange="{self.isp_digital_gain} {self.isp_digital_gain}"')
            logger.info(f"Setting ISP digital gain: {self.isp_digital_gain}")

        # Add exposure compensation
        if self.exposure_compensation != 0.0:
            pipeline_parts.append(f"exposurecompensation={self.exposure_compensation}")
            logger.info(f"Setting exposure compensation: {self.exposure_compensation}")

        # Add saturation
        if self.saturation != 1.0:
            pipeline_parts.append(f"saturation={self.saturation}")
            logger.info(f"Setting saturation: {self.saturation}")

        # Add auto-exposure lock
        if self.ae_lock:
            pipeline_parts.append("aelock=true")
            logger.info("Enabling auto-exposure lock")

        # Add auto white balance lock
        if self.awb_lock:
            pipeline_parts.append("awblock=true")
            logger.info("Enabling auto white balance lock")

        # Add anti-banding mode
        if self.antibanding_mode != 1:  # 1 is default (auto)
            pipeline_parts.append(f"aeantibanding={self.antibanding_mode}")
            logger.info(f"Setting anti-banding mode: {self.antibanding_mode}")

        # Complete pipeline
        pipeline = (
            " ".join(pipeline_parts) + " ! "
            "nvvidconv ! video/x-raw,format=I420 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1"
        )

        return pipeline

    def initialize(self) -> bool:
        """Initialize camera with enhanced controls."""
        try:
            logger.info(f"Initializing enhanced camera {self.device_id} with manual controls")

            # Build GStreamer pipeline with camera controls
            gstreamer_pipeline = self._build_gstreamer_pipeline()
            logger.info(f"GStreamer pipeline: {gstreamer_pipeline}")

            self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

            if not self.cap.isOpened():
                logger.error("Enhanced GStreamer pipeline failed to open")
                return False

            logger.info("Enhanced GStreamer pipeline opened successfully")

            # Get camera resolution
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Camera initialized: {self.camera_width}x{self.camera_height}")

            # Warm up camera
            logger.info("Warming up camera with enhanced controls...")
            for i in range(5):  # More warmup frames for manual exposure
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"Warmup frame {i} failed")
                else:
                    logger.info(f"Warmup frame {i} successful, shape: {frame.shape}")

            return True

        except Exception as e:
            logger.error(f"Enhanced camera initialization failed: {e}")
            return False

    def capture_frame(self) -> Optional:
        """Capture frame with enhanced controls."""
        # Same as original capture_frame method
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
            frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_LINEAR)

            return frame

        except Exception as e:
            logger.error(f"Enhanced frame capture failed: {e}")
            return None

    def cleanup(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            self.cap = None


# Example usage configurations for LED pattern capture
CAPTURE_CONFIGS = {
    # Fast capture with higher gain for quick testing
    "fast_capture": {
        "manual_exposure": 0.005,  # 5ms exposure
        "manual_gain": 4.0,  # 4x gain
        "isp_digital_gain": 2.0,  # 2x digital gain
        "saturation": 1.2,  # Slightly boost colors
        "ae_lock": True,  # Lock exposure
        "awb_lock": True,  # Lock white balance
    },
    # High quality capture with longer exposure
    "high_quality": {
        "manual_exposure": 0.020,  # 20ms exposure (good for dim LEDs)
        "manual_gain": 2.0,  # Moderate gain
        "isp_digital_gain": 1.0,  # No digital gain (preserve quality)
        "saturation": 1.0,  # Natural colors
        "ae_lock": True,  # Lock exposure
        "awb_lock": True,  # Lock white balance
    },
    # Bright environment with fast exposure
    "bright_environment": {
        "manual_exposure": 0.002,  # 2ms exposure (fast for bright LEDs)
        "manual_gain": 1.0,  # Minimal gain
        "exposure_compensation": -0.5,  # Slightly underexpose
        "saturation": 1.1,  # Boost saturation slightly
        "ae_lock": True,  # Lock exposure
    },
    # Automatic with locked settings (good baseline)
    "auto_locked": {
        "ae_lock": True,  # Lock auto-exposure after initial setup
        "awb_lock": True,  # Lock auto white balance
        "saturation": 1.1,  # Slight saturation boost
        "antibanding_mode": 3,  # 60Hz anti-banding
    },
}


def get_recommended_config(led_brightness: str = "medium") -> dict:
    """
    Get recommended camera configuration based on LED brightness.

    Args:
        led_brightness: "dim", "medium", or "bright"

    Returns:
        Camera configuration dictionary
    """
    if led_brightness == "dim":
        return CAPTURE_CONFIGS["high_quality"]
    elif led_brightness == "bright":
        return CAPTURE_CONFIGS["bright_environment"]
    else:
        return CAPTURE_CONFIGS["auto_locked"]


if __name__ == "__main__":
    # Example usage
    config = get_recommended_config("medium")
    camera = EnhancedCameraCapture(device_id=0, **config)

    if camera.initialize():
        frame = camera.capture_frame()
        if frame is not None:
            print(f"Captured frame with shape: {frame.shape}")
        camera.cleanup()
