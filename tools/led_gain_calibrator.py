#!/usr/bin/env python3
"""
LED Gain Calibration Tool for Diffusion Pattern Capture.

This tool performs a two-phase LED gain calibration process to determine optimal
camera gain settings for LED diffusion pattern capture while avoiding pixel saturation.

Phase 1: Saturation Elimination Phase
- Tests a representative sample of LEDs (every ~26th LED for 2600 total)
- Reduces gain until no test LEDs produce saturated pixels (value = 255)
- Establishes a safe gain baseline

Phase 2: Maximum Dynamic Range Optimization
- Identifies the brightest LED from the test sample
- Calibrates gain so the brightest LED produces max pixel value of 254
- Maximizes dynamic range while maintaining 1-digit safety margin

Features:
- Two-phase calibration algorithm for optimal dynamic range
- Representative LED sampling across the entire array
- Automatic gain adjustment with safety margins
- Comprehensive brightness and saturation analysis
- Fixed camera settings for consistent measurements
- Validation and fine-tuning of final gain settings

Usage:
    python led_gain_calibrator.py --wled-host 192.168.7.140 --camera-device 0
    python led_gain_calibrator.py --wled-host 192.168.7.140 --total-leds 2600 --sample-size 100
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.consumer.wled_client import WLEDClient, WLEDConfig

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Results from LED gain calibration process."""

    final_gain: float
    safe_gain: float
    brightest_led_index: int
    brightest_led_max_pixel: int
    test_led_count: int
    saturated_leds_phase1: int
    validation_passed: bool
    calibration_time: float


@dataclass
class LEDTestResult:
    """Results from testing a single LED."""

    led_index: int
    max_pixel_value: int
    saturated_pixel_count: int
    has_saturation: bool
    mean_brightness: float
    brightest_position: Tuple[int, int]
    gain_setting: float


class GainControlledCamera:
    """Camera capture with gain control for calibration."""

    def __init__(
        self,
        device_id: int = 0,
        use_usb: bool = False,
        camera_config_path: Optional[str] = None,
        resolution: Tuple[int, int] = (640, 480),
    ):
        """
        Initialize camera with gain control.

        Args:
            device_id: Camera device ID (usually 0 for default camera)
            use_usb: Use USB camera instead of CSI camera
            camera_config_path: Path to camera configuration JSON file with crop region
            resolution: Camera resolution as (width, height) tuple
        """
        self.device_id = device_id
        self.use_usb = use_usb
        self.resolution = resolution
        self.current_gain = 2.0
        self.cap: Optional[cv2.VideoCapture] = None
        self.crop_region: Optional[Dict[str, int]] = None
        self.frame_count = 0  # Track frames for periodic logging

        # Load camera configuration if provided
        if camera_config_path and Path(camera_config_path).exists():
            try:
                with open(camera_config_path, "r") as f:
                    config = json.load(f)
                    if "crop_region" in config:
                        self.crop_region = config["crop_region"]
                        logger.info(f"Loaded crop region from {camera_config_path}: {self.crop_region}")
                    # Override device settings from config
                    if "camera_device" in config:
                        self.device_id = config["camera_device"]
                    if "use_usb" in config:
                        self.use_usb = config["use_usb"]
            except Exception as e:
                logger.warning(f"Failed to load camera config from {camera_config_path}: {e}")

    def initialize(self) -> bool:
        """Initialize camera with fixed exposure and gain control."""
        try:
            if self.use_usb:
                width, height = self.resolution
                logger.info(f"Initializing USB camera at /dev/video{self.device_id} with gain control")
                logger.info(f"Target resolution: {width}x{height}")

                # Use GStreamer with v4l2src for better control over resolution and format
                gstreamer_pipeline = (
                    f"v4l2src device=/dev/video{self.device_id} ! "
                    f"image/jpeg, width={width}, height={height}, framerate=30/1 ! "
                    "jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink"
                )
                logger.info(f"Using GStreamer pipeline: {gstreamer_pipeline}")
                self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

                if not self.cap.isOpened():
                    logger.warning("GStreamer MJPEG pipeline failed, trying V4L2 backend")
                    # Fallback to V4L2 with resolution setting
                    self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

                if not self.cap.isOpened():
                    logger.warning("V4L2 backend failed, trying default backend")
                    self.cap = cv2.VideoCapture(self.device_id)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                if not self.cap.isOpened():
                    logger.error(f"Failed to open USB camera at /dev/video{self.device_id}")
                    return False

                # Verify actual resolution
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"USB camera opened: {actual_width}x{actual_height}")

                # Try to set gain if supported by USB camera
                try:
                    self.cap.set(cv2.CAP_PROP_GAIN, self.current_gain)
                    logger.info(f"Set USB camera gain to {self.current_gain}")
                except:
                    logger.warning("Could not set gain on USB camera")
            else:
                logger.info(f"Initializing CSI camera {self.device_id} with gain control")

                # Build initial pipeline with conservative gain for CSI camera
                pipeline = self._build_calibration_pipeline(self.current_gain)

                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

                if not self.cap.isOpened():
                    logger.error("Failed to open camera pipeline")
                    return False

            # Warm up camera
            logger.info("Warming up camera...")
            for i in range(5):
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"Warmup frame {i} failed")
                else:
                    logger.debug(f"Warmup frame {i} successful: {frame.shape}")
                time.sleep(0.2)

            logger.info("Camera initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def _build_calibration_pipeline(self, gain_value: float) -> str:
        """
        Build GStreamer pipeline with specified gain for calibration.

        Args:
            gain_value: Camera gain value to set

        Returns:
            GStreamer pipeline string
        """
        return (
            f"nvarguscamerasrc sensor-id={self.device_id} "
            f'gainrange="{gain_value} {gain_value}" '
            f'exposuretimerange="10000000 10000000" '  # Fixed 10ms exposure
            "aelock=true "  # Lock auto-exposure
            "awblock=true "  # Lock auto-white-balance
            "saturation=1.0 "  # No saturation boost
            "! nvvidconv ! video/x-raw,format=I420 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1"
        )

    def set_gain(self, gain_value: float) -> bool:
        """
        Set camera gain by reinitializing with new pipeline (CSI) or setting property (USB).

        Args:
            gain_value: New gain value to set

        Returns:
            True if gain set successfully, False otherwise
        """
        try:
            logger.info(f"Setting camera gain to {gain_value}")

            if self.use_usb:
                # For USB cameras, try to set gain directly
                if self.cap:
                    try:
                        self.cap.set(cv2.CAP_PROP_GAIN, gain_value)
                        self.current_gain = gain_value
                        logger.info(f"Set USB camera gain to {gain_value}")
                    except:
                        logger.warning(f"Could not set gain {gain_value} on USB camera")
                        return False
            else:
                # For CSI cameras, reinitialize with new pipeline
                # Close existing capture
                if self.cap:
                    self.cap.release()

                # Create new pipeline with updated gain
                pipeline = self._build_calibration_pipeline(gain_value)
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

                if not self.cap.isOpened():
                    logger.error(f"Failed to set gain {gain_value}")
                    return False

                self.current_gain = gain_value

            # Allow camera to stabilize
            time.sleep(0.5)

            # Clear buffer with a few reads
            for _ in range(3):
                ret, _ = self.cap.read()
                if not ret:
                    logger.warning("Buffer clear read failed")

            logger.info(f"Camera gain set to {gain_value}")
            return True

        except Exception as e:
            logger.error(f"Failed to set gain {gain_value}: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera.

        Returns:
            Frame as RGB array (cropped if crop_region is configured), or None if failed
        """
        if not self.cap:
            logger.error("Camera not initialized")
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                return None

            # Log frame info periodically (every 50th frame)
            self.frame_count += 1
            if self.frame_count % 50 == 0:
                logger.debug(
                    f"Captured frame #{self.frame_count}: shape={frame.shape}, dtype={frame.dtype}, "
                    f"min={np.min(frame)}, max={np.max(frame)}"
                )

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply crop region if configured
            if self.crop_region:
                x = self.crop_region["x"]
                y = self.crop_region["y"]
                w = self.crop_region["width"]
                h = self.crop_region["height"]

                # Ensure crop region is within frame bounds
                frame_h, frame_w = frame.shape[:2]
                x = max(0, min(x, frame_w - 1))
                y = max(0, min(y, frame_h - 1))
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)

                # Apply crop
                frame = frame[y : y + h, x : x + w]
                if self.frame_count % 50 == 0:
                    logger.debug(f"Applied crop region: {x},{y} -> {x+w},{y+h}, " f"cropped shape={frame.shape}")

            return frame

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None

    def cleanup(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            self.cap = None


def analyze_frame_brightness(frame: np.ndarray) -> Dict:
    """
    Analyze captured frame for brightness characteristics.

    Args:
        frame: RGB frame array

    Returns:
        Dictionary with brightness analysis results
    """
    # Convert to grayscale for overall brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Find maximum pixel value across all channels
    max_pixel = np.max(frame)

    # Count saturated pixels
    saturated_pixels = np.sum(frame == 255)

    # Find brightest region (for LED localization)
    max_intensity_pos = np.unravel_index(np.argmax(gray), gray.shape)

    # Calculate additional statistics for LED detection
    mean_brightness = float(np.mean(frame))
    std_brightness = float(np.std(frame))

    # Calculate what percentage of the image is very bright (potential LED area)
    bright_threshold = max(200, max_pixel * 0.8)  # 80% of max or 200, whichever is higher
    bright_pixels = np.sum(frame > bright_threshold)
    bright_percentage = (bright_pixels / frame.size) * 100

    # Calculate what percentage is very dark (background)
    dark_threshold = 50
    dark_pixels = np.sum(frame < dark_threshold)
    dark_percentage = (dark_pixels / frame.size) * 100

    return {
        "max_pixel_value": int(max_pixel),
        "saturated_pixel_count": int(saturated_pixels),
        "brightest_position": max_intensity_pos,
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "bright_percentage": bright_percentage,
        "dark_percentage": dark_percentage,
        "has_saturation": saturated_pixels > 0,
        "brightness_range": int(max_pixel - np.min(frame)),
    }


def validate_basic_image(analysis: Dict, min_brightness: int = 20) -> Tuple[bool, str]:
    """
    Basic validation that we're seeing some image data with range.

    Args:
        analysis: Dictionary from analyze_frame_brightness()
        min_brightness: Minimum max pixel value to accept

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if image has any range at all (camera working)
    if analysis["brightness_range"] < 5:  # Very low threshold - just need some variation
        return False, f"No brightness range (range={analysis['brightness_range']}). Camera may be blocked or broken."

    # Check if we see any bright pixels at all (LEDs may be on)
    if analysis["max_pixel_value"] < min_brightness:
        return False, f"No bright pixels detected (max={analysis['max_pixel_value']}). LEDs may not be on or visible."

    # Basic validation passed
    return True, "Basic image validation passed"


def validate_led_image(analysis: Dict, strict: bool = True) -> Tuple[bool, str]:
    """
    Validate if the analyzed image looks like it contains an LED.

    Args:
        analysis: Dictionary from analyze_frame_brightness()
        strict: If False, use more lenient thresholds for low-gain testing

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if image is mostly dark (LED off or background)
    if analysis["max_pixel_value"] < 30:
        return (
            False,
            f"Image too dark (max={analysis['max_pixel_value']}). LED may not be on or camera not seeing LEDs.",
        )

    # Check if image has reasonable contrast (LED should create bright spot)
    min_contrast = 50 if strict else 20  # Lower contrast threshold for low gain
    if analysis["brightness_range"] < min_contrast:
        return False, f"Low contrast (range={analysis['brightness_range']}). Camera may not be seeing LED properly."

    # Check if image has too much overall brightness (wrong scene, room lights on)
    max_mean = 100 if strict else 150  # Higher threshold for high gain testing
    if analysis["mean_brightness"] > max_mean:
        return (
            False,
            f"Image too bright overall (mean={analysis['mean_brightness']:.1f}). Turn off room lights or check scene.",
        )

    # Check if bright area is too large (wrong scene, not pointing at LEDs)
    if analysis["bright_percentage"] > 10:
        return (
            False,
            f"Too much bright area ({analysis['bright_percentage']:.1f}%). Camera may not be pointed at LED display.",
        )

    # Image looks reasonable for LED calibration
    return True, "Image validation passed"


class LEDGainCalibrator:
    """LED gain calibration system with two-phase algorithm."""

    def __init__(self, wled_config: WLEDConfig, camera: GainControlledCamera, total_leds: int):
        """
        Initialize LED gain calibrator.

        Args:
            wled_config: WLED configuration
            camera: Gain-controlled camera instance
            total_leds: Total number of LEDs in the system
        """
        self.wled_client = WLEDClient(wled_config)
        self.camera = camera
        self.total_leds = total_leds

        # Calculate test LED indices - sample every ~26th LED for representative coverage
        sample_interval = max(1, total_leds // 100)  # ~100 test LEDs
        self.test_led_indices = list(range(0, total_leds, sample_interval))

        logger.info(f"Calibrator initialized: {total_leds} total LEDs, {len(self.test_led_indices)} test LEDs")

    def run_calibration(self) -> CalibrationResult:
        """
        Run complete two-phase LED gain calibration process.

        Returns:
            CalibrationResult with final gain and validation results
        """
        start_time = time.time()

        try:
            # Connect to WLED
            if not self.wled_client.connect():
                raise RuntimeError("Failed to connect to WLED controller")

            logger.info("Starting LED gain calibration process")

            # Test capture to verify camera is working
            logger.info("Testing camera capture...")
            test_frame = self.camera.capture_frame()
            if test_frame is not None:
                logger.info(
                    f"Camera test successful: frame shape={test_frame.shape}, "
                    f"mean brightness={np.mean(test_frame):.1f}"
                )
            else:
                raise RuntimeError("Camera test failed - unable to capture frames")

            # Verify gain control is working
            logger.info("Testing gain control functionality...")
            gain_test_passed = self._test_gain_control()
            if not gain_test_passed:
                raise RuntimeError("Gain control test failed - camera gain adjustment not working")

            # Phase 1: Eliminate saturation
            logger.info("Phase 1: Eliminating saturation...")
            safe_gain, test_results = self._phase1_eliminate_saturation()
            logger.info(f"Safe gain found: {safe_gain:.6f}")

            # Phase 2: Optimize for maximum dynamic range
            logger.info("Phase 2: Optimizing dynamic range...")
            final_gain = self._phase2_optimize_range(safe_gain, test_results)
            logger.info(f"Final optimized gain: {final_gain:.6f}")

            # Validation
            logger.info("Phase 3: Validating calibration...")
            validation_passed = self._validate_calibration(final_gain, test_results)

            # Find brightest LED info
            brightest_result = max(test_results, key=lambda x: x.max_pixel_value)

            calibration_time = time.time() - start_time

            result = CalibrationResult(
                final_gain=final_gain,
                safe_gain=safe_gain,
                brightest_led_index=brightest_result.led_index,
                brightest_led_max_pixel=brightest_result.max_pixel_value,
                test_led_count=len(self.test_led_indices),
                saturated_leds_phase1=sum(1 for r in test_results if r.has_saturation),
                validation_passed=validation_passed,
                calibration_time=calibration_time,
            )

            logger.info(f"Calibration completed in {calibration_time:.1f}s")
            return result

        finally:
            # Always turn off LEDs and disconnect
            self._turn_off_all_leds()
            self.wled_client.disconnect()

    def _phase1_eliminate_saturation(self) -> Tuple[float, List[LEDTestResult]]:
        """
        Phase 1: Progressive per-LED gain reduction until no saturation.
        For each LED, reduce gain until max pixel < 255, then move to next LED.

        Returns:
            Tuple of (final_safe_gain, test_results_list)
        """
        print(f"\nStarting progressive per-LED gain calibration...", flush=True)
        logger.info("Starting progressive per-LED gain calibration")

        current_gain = 4.0  # Start with high gain
        test_results = []
        brightest_led_max = 0

        for i, led_idx in enumerate(self.test_led_indices):
            print(
                f"\nTesting LED {led_idx} ({i+1}/{len(self.test_led_indices)}) at gain {current_gain:.6f}...",
                flush=True,
            )

            # Test this LED and reduce gain until not saturated
            led_gain, led_result = self._calibrate_single_led(led_idx, current_gain)

            # Update gain for next LED (keep the reduced gain)
            current_gain = led_gain
            test_results.append(led_result)

            # Track brightest LED
            if led_result.max_pixel_value > brightest_led_max:
                brightest_led_max = led_result.max_pixel_value

            # Progress reporting every 10 LEDs
            if i % 10 == 0 or led_result.has_saturation:
                print(
                    f"  LED {led_idx}: max_pixel={led_result.max_pixel_value}, "
                    f"final_gain={led_gain:.6f}, saturated={led_result.has_saturation}",
                    flush=True,
                )

            # Early validation on first few LEDs
            if i < 3:
                analysis = {
                    "max_pixel_value": led_result.max_pixel_value,
                    "brightness_range": led_result.max_pixel_value,  # Simplified
                }
                is_valid, validation_msg = validate_basic_image(analysis, min_brightness=5)
                if not is_valid:
                    if i == 0:  # First LED failure is more concerning
                        raise RuntimeError(f"LED calibration failed on first LED: {validation_msg}")
                    else:
                        print(f"⚠️  LED {led_idx} validation failed: {validation_msg}", flush=True)
                        logger.warning(f"LED {led_idx} validation failed: {validation_msg}")

        # Final summary
        saturated_leds = sum(1 for r in test_results if r.has_saturation)
        print(f"\n✅ Progressive calibration complete:", flush=True)
        print(f"  Final gain: {current_gain:.6f}", flush=True)
        print(f"  Brightest LED max pixel: {brightest_led_max}", flush=True)
        print(f"  Saturated LEDs remaining: {saturated_leds}/{len(test_results)}", flush=True)

        logger.info(f"Progressive calibration complete: gain={current_gain:.6f}, brightest={brightest_led_max}")

        return current_gain, test_results

    def _calibrate_single_led(self, led_idx: int, starting_gain: float) -> Tuple[float, LEDTestResult]:
        """
        Calibrate gain for a single LED until max pixel < 255.

        Args:
            led_idx: LED index to calibrate
            starting_gain: Initial gain value to try

        Returns:
            Tuple of (final_gain, test_result)
        """
        current_gain = starting_gain
        max_iterations = 10

        for iteration in range(max_iterations):
            # Set camera gain
            if not self.camera.set_gain(current_gain):
                raise RuntimeError(f"Failed to set camera gain to {current_gain}")

            # Give camera extra time to stabilize after gain change (prevents indicator flashing)
            time.sleep(0.3)

            # Turn on single LED
            self._set_single_led(led_idx, brightness=255)
            time.sleep(0.5)  # Give LED plenty of time to turn on and stabilize

            # Capture and analyze
            frame = self.camera.capture_frame()
            if frame is None:
                self._turn_off_all_leds()
                raise RuntimeError(f"Failed to capture frame for LED {led_idx}")

            analysis = analyze_frame_brightness(frame)

            # Debug: Always print what we measured
            print(
                f"      Iteration {iteration+1}: LED {led_idx} max_pixel={analysis['max_pixel_value']}, "
                f"saturated_pixels={analysis['saturated_pixel_count']}, "
                f"gain={current_gain:.6f}",
                flush=True,
            )

            # Turn off LED
            self._turn_off_all_leds()

            # Create result
            result = LEDTestResult(
                led_index=led_idx,
                max_pixel_value=analysis["max_pixel_value"],
                saturated_pixel_count=analysis["saturated_pixel_count"],
                has_saturation=analysis["has_saturation"],
                mean_brightness=analysis["mean_brightness"],
                brightest_position=analysis["brightest_position"],
                gain_setting=current_gain,
            )

            # Check if we're below saturation
            if analysis["max_pixel_value"] < 255:
                # Success! This LED is not saturated
                if iteration > 0:  # Only log if we had to reduce gain
                    print(
                        f"    LED {led_idx} calibrated: {starting_gain:.6f} → {current_gain:.6f} "
                        f"(max_pixel: {analysis['max_pixel_value']})",
                        flush=True,
                    )
                return current_gain, result

            # Still saturated, reduce gain
            old_gain = current_gain
            current_gain *= 0.9  # 10% reduction per iteration

            print(
                f"    LED {led_idx} STILL SATURATED (max={analysis['max_pixel_value']}), "
                f"reducing gain: {old_gain:.6f} → {current_gain:.6f}",
                flush=True,
            )

            # Check if gain reduction is too small to matter
            if abs(old_gain - current_gain) < 0.01:
                print(
                    f"⚠️  Gain reduction too small ({abs(old_gain - current_gain):.6f}) - may indicate gain control issues",
                    flush=True,
                )
                logger.warning(f"Very small gain reduction for LED {led_idx}: {abs(old_gain - current_gain):.6f}")

        # If we get here, we couldn't eliminate saturation
        print(
            f"⚠️  LED {led_idx} still saturated after {max_iterations} iterations "
            f"(final gain: {current_gain:.6f}, max_pixel: {analysis['max_pixel_value']})",
            flush=True,
        )
        logger.warning(f"LED {led_idx} still saturated after maximum iterations")

        return current_gain, result

    def _phase2_optimize_range(self, safe_gain: float, test_results: List[LEDTestResult]) -> float:
        """
        Phase 2: Optimize gain for brightest LED to reach 254.

        Args:
            safe_gain: Safe gain from Phase 1
            test_results: Test results from Phase 1

        Returns:
            Optimized final gain value
        """
        # Find brightest LED from test results
        brightest_result = max(test_results, key=lambda x: x.max_pixel_value)
        brightest_led = brightest_result.led_index
        brightest_max_pixel = brightest_result.max_pixel_value

        logger.info(f"Brightest test LED: {brightest_led} (max pixel: {brightest_max_pixel})")

        # Calculate target gain
        target_pixel = 254
        gain_multiplier = target_pixel / brightest_max_pixel
        target_gain = safe_gain * gain_multiplier

        logger.info(f"Target gain calculation: {safe_gain:.6f} × {gain_multiplier:.6f} = {target_gain:.6f}")

        # Set target gain and measure actual result
        if not self.camera.set_gain(target_gain):
            raise RuntimeError(f"Failed to set target gain {target_gain}")

        # Test with brightest LED
        self._set_single_led(brightest_led, brightness=255)
        time.sleep(0.2)  # Extra stabilization time for final measurement

        frame = self.camera.capture_frame()
        if frame is None:
            raise RuntimeError("Failed to capture frame for optimization")

        analysis = analyze_frame_brightness(frame)
        actual_max = analysis["max_pixel_value"]

        logger.info(f"Actual max pixel at target gain: {actual_max}")

        # Fine adjustment if needed
        if abs(actual_max - target_pixel) > 1:
            fine_multiplier = target_pixel / actual_max
            final_gain = target_gain * fine_multiplier
            logger.info(f"Fine adjustment: {target_gain:.6f} × {fine_multiplier:.6f} = {final_gain:.6f}")
        else:
            final_gain = target_gain

        # Turn off LED
        self._turn_off_all_leds()

        return final_gain

    def _test_gain_control(self) -> bool:
        """
        Test if gain control is working by testing different gain values.
        Focus on whether LED pixel brightness changes with gain adjustment.

        Returns:
            True if gain control is working, False otherwise
        """
        print("  Testing gain control with multiple LEDs...", flush=True)
        logger.info("Testing gain control functionality")

        try:
            # Use first 10 test LEDs for gain control test
            test_leds = self.test_led_indices[:10]
            print(f"  Using LEDs {test_leds} for gain test", flush=True)

            # Turn on multiple test LEDs
            led_data = np.zeros((self.total_leds, 3), dtype=np.uint8)
            for led_idx in test_leds:
                led_data[led_idx] = [255, 255, 255]  # White at full brightness

            result = self.wled_client.send_led_data(led_data)
            if not result.success:
                logger.warning(f"Failed to set test LEDs: {result.errors}")

            time.sleep(0.3)  # Let LEDs stabilize

            # Test multiple gain values to see if LED brightness changes
            gain_values = [1.0, 2.5, 4.0]  # Low, medium, high
            measurements = []

            for gain in gain_values:
                print(f"  Testing gain {gain}...", flush=True)

                if not self.camera.set_gain(gain):
                    print(f"  ❌ Failed to set gain {gain}", flush=True)
                    self._turn_off_all_leds()
                    return False

                time.sleep(0.5)  # Let camera stabilize
                frame = self.camera.capture_frame()
                if frame is None:
                    print(f"  ❌ Failed to capture frame at gain {gain}", flush=True)
                    self._turn_off_all_leds()
                    return False

                analysis = analyze_frame_brightness(frame)

                # Basic validation - just check we have some image data
                is_valid, validation_msg = validate_basic_image(analysis)
                if not is_valid:
                    print(f"  ❌ Basic image validation failed at gain {gain}: {validation_msg}", flush=True)
                    self._turn_off_all_leds()
                    return False

                measurements.append(
                    {
                        "gain": gain,
                        "max_pixel": analysis["max_pixel_value"],
                        "mean_brightness": analysis["mean_brightness"],
                        "brightness_range": analysis["brightness_range"],
                    }
                )

                print(
                    f"    Gain {gain}: max={analysis['max_pixel_value']}, "
                    f"mean={analysis['mean_brightness']:.1f}, "
                    f"range={analysis['brightness_range']}",
                    flush=True,
                )

            # Turn off LEDs
            self._turn_off_all_leds()

            # Analyze if gain control is working by comparing measurements
            low_meas = measurements[0]  # gain 1.0
            high_meas = measurements[2]  # gain 4.0

            # Check if max pixel values increase with gain
            max_diff = high_meas["max_pixel"] - low_meas["max_pixel"]
            max_ratio = high_meas["max_pixel"] / max(low_meas["max_pixel"], 1)

            # Check if mean brightness increases with gain
            mean_diff = high_meas["mean_brightness"] - low_meas["mean_brightness"]
            mean_ratio = high_meas["mean_brightness"] / max(low_meas["mean_brightness"], 1)

            print(f"  Gain effect analysis:", flush=True)
            print(
                f"    Max pixel: {low_meas['max_pixel']} -> {high_meas['max_pixel']} "
                f"(+{max_diff}, {max_ratio:.2f}x)",
                flush=True,
            )
            print(
                f"    Mean brightness: {low_meas['mean_brightness']:.1f} -> {high_meas['mean_brightness']:.1f} "
                f"(+{mean_diff:.1f}, {mean_ratio:.2f}x)",
                flush=True,
            )

            # Gain control is working if we see meaningful changes in LED brightness
            max_changed = max_diff >= 15 or max_ratio >= 1.15  # 15% increase minimum
            mean_changed = mean_diff >= 2 or mean_ratio >= 1.15  # Small absolute change OK for mean

            if not max_changed and not mean_changed:
                print(f"  ❌ Gain control not working: LED brightness unchanged", flush=True)
                print(f"     Max pixel change too small: +{max_diff} ({max_ratio:.2f}x)", flush=True)
                print(f"     Mean brightness change too small: +{mean_diff:.1f} ({mean_ratio:.2f}x)", flush=True)
                logger.error(f"Gain control test failed: max_diff={max_diff}, mean_diff={mean_diff:.1f}")
                return False

            # Check for monotonic increase (medium gain should be between low and high)
            med_meas = measurements[1]  # gain 2.5
            max_monotonic = low_meas["max_pixel"] <= med_meas["max_pixel"] <= high_meas["max_pixel"]
            mean_monotonic = low_meas["mean_brightness"] <= med_meas["mean_brightness"] <= high_meas["mean_brightness"]

            if not max_monotonic and not mean_monotonic:
                print(f"  ⚠️  Non-monotonic gain response - may indicate camera issues", flush=True)
                logger.warning("Non-monotonic gain response detected")
                # Continue anyway, basic gain control seems to work

            print(f"  ✅ Gain control working: LED brightness increases with gain", flush=True)
            logger.info(f"Gain control test passed: max {max_ratio:.2f}x, mean {mean_ratio:.2f}x increase")
            return True

        except Exception as e:
            print(f"  ❌ Gain control test failed with error: {e}", flush=True)
            logger.error(f"Gain control test failed with exception: {e}")
            self._turn_off_all_leds()
            return False

    def _validate_calibration(self, final_gain: float, test_results: List[LEDTestResult]) -> bool:
        """
        Validate the final calibration by testing random LEDs.

        Args:
            final_gain: Final calibrated gain value
            test_results: Original test results for reference

        Returns:
            True if validation passed, False otherwise
        """
        logger.info(f"Validating calibration with gain {final_gain:.6f}")

        # Set final gain
        if not self.camera.set_gain(final_gain):
            logger.error("Failed to set final gain for validation")
            return False

        # Test 10 random LEDs from the test set
        import random

        validation_leds = random.sample(
            self.test_led_indices, min(10, len(self.test_led_indices))
        )  # nosec B311 - not crypto

        saturated_count = 0
        max_pixel_values = []

        for led_idx in validation_leds:
            self._set_single_led(led_idx, brightness=255)
            time.sleep(0.1)

            frame = self.camera.capture_frame()
            if frame is None:
                logger.warning(f"Failed to capture validation frame for LED {led_idx}")
                continue

            analysis = analyze_frame_brightness(frame)
            max_pixel_values.append(analysis["max_pixel_value"])

            if analysis["has_saturation"]:
                saturated_count += 1
                logger.warning(f"LED {led_idx} still saturated at final gain")

            self._turn_off_all_leds()
            time.sleep(0.05)

        # Validation criteria
        max_observed = max(max_pixel_values) if max_pixel_values else 0
        validation_passed = (
            saturated_count == 0  # No saturation
            and max_observed >= 240  # Good dynamic range
            and max_observed <= 255  # Within valid range
        )

        logger.info("Validation results:")
        logger.info(f"  Saturated LEDs: {saturated_count}/{len(validation_leds)}")
        logger.info(f"  Max observed pixel: {max_observed}")
        logger.info(f"  Validation {'PASSED' if validation_passed else 'FAILED'}")

        return validation_passed

    def _set_single_led(self, led_index: int, brightness: int = 255):
        """
        Turn on single LED at specified brightness, all others off.

        Args:
            led_index: Index of LED to turn on
            brightness: Brightness level (0-255)
        """
        led_data = np.zeros((self.total_leds, 3), dtype=np.uint8)
        led_data[led_index] = [brightness, brightness, brightness]  # White at full brightness

        result = self.wled_client.send_led_data(led_data)
        if not result.success:
            logger.warning(f"Failed to set LED {led_index}: {result.errors}")

    def _turn_off_all_leds(self):
        """Turn off all LEDs."""
        result = self.wled_client.set_solid_color(0, 0, 0)
        if not result.success:
            logger.warning(f"Failed to turn off LEDs: {result.errors}")


def main():
    """Main calibration entry point."""
    parser = argparse.ArgumentParser(description="LED Gain Calibration Tool")

    # WLED configuration
    parser.add_argument("--wled-host", default="wled.local", help="WLED host address")
    parser.add_argument("--wled-port", type=int, default=4048, help="WLED DDP port")
    parser.add_argument("--leds", type=int, help="Total number of LEDs")

    # Camera configuration
    parser.add_argument("--camera-device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--usb", action="store_true", help="Use USB camera instead of CSI camera")
    parser.add_argument(
        "--camera-config", default="config/camera.json", help="Camera configuration JSON file with crop region"
    )
    parser.add_argument("--resolution", default="1920x1080", help="Camera resolution (WxH), e.g. 1920x1080, 1280x720")
    parser.add_argument("--list-resolutions", action="store_true", help="List supported camera resolutions and exit")

    # Output options
    parser.add_argument("--output", help="Output file to save calibration results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging - force reconfiguration
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Force reconfiguration
        handlers=[logging.StreamHandler()],  # Explicitly use console handler
    )

    # Disable WLED client debug logging as it's too verbose
    logging.getLogger("src.consumer.wled_client").setLevel(logging.INFO)

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split("x"))
        resolution = (width, height)
    except ValueError:
        logger.error(f"Invalid resolution format: {args.resolution}. Use WxH format (e.g., 1280x720)")
        return 1

    # List resolutions if requested
    if args.list_resolutions:
        if args.usb:
            print("Supported USB camera resolutions:")
            print("MJPEG format (recommended for high resolution):")
            print("  1920x1080 @ 30fps")
            print("  1280x720 @ 30fps")
            print("  1024x576 @ 30fps")
            print("  800x600 @ 30fps")
            print("  640x480 @ 30fps")
            print("\nYUYV format (lower resolution but uncompressed):")
            print("  1280x720 @ 10fps")
            print("  800x600 @ 24fps")
            print("  640x480 @ 30fps")
        else:
            print("CSI camera resolutions depend on your camera sensor.")
            print("Common resolutions: 1920x1080, 1280x720, 640x480")
        return 0

    try:
        # Also print to ensure we see output
        print("LED Gain Calibration Tool Starting", flush=True)
        logger.info("LED Gain Calibration Tool Starting")
        print(f"Target: {args.wled_host}:{args.wled_port} ({args.leds} LEDs)", flush=True)
        logger.info(f"Target: {args.wled_host}:{args.wled_port} ({args.leds} LEDs)")
        print(f"Camera: Device {args.camera_device}, USB={args.usb}", flush=True)
        logger.info(f"Camera: Device {args.camera_device}")

        # Initialize WLED configuration
        wled_config = WLEDConfig(
            host=args.wled_host, port=args.wled_port, led_count=args.leds, timeout=5.0, persistent_retry=False
        )

        # Initialize camera with config if available
        camera = GainControlledCamera(
            device_id=args.camera_device, use_usb=args.usb, camera_config_path=args.camera_config, resolution=resolution
        )
        if not camera.initialize():
            logger.error("Failed to initialize camera")
            return 1

        try:
            # Initialize calibrator
            calibrator = LEDGainCalibrator(wled_config, camera, args.leds)

            # Run calibration
            result = calibrator.run_calibration()

            # Display results
            logger.info("=" * 60)
            logger.info("CALIBRATION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Final optimized gain: {result.final_gain:.6f}")
            logger.info(f"Safe baseline gain: {result.safe_gain:.6f}")
            logger.info(f"Brightest LED: {result.brightest_led_index} (max pixel: {result.brightest_led_max_pixel})")
            logger.info(f"Test LEDs sampled: {result.test_led_count}")
            logger.info(f"Validation: {'PASSED' if result.validation_passed else 'FAILED'}")
            logger.info(f"Calibration time: {result.calibration_time:.1f} seconds")

            # Save results if requested
            if args.output:
                output_data = {
                    "final_gain": result.final_gain,
                    "safe_gain": result.safe_gain,
                    "brightest_led_index": result.brightest_led_index,
                    "brightest_led_max_pixel": result.brightest_led_max_pixel,
                    "test_led_count": result.test_led_count,
                    "validation_passed": result.validation_passed,
                    "calibration_time": result.calibration_time,
                    "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "wled_host": args.wled_host,
                    "wled_port": args.wled_port,
                    "total_leds": args.leds,
                    "camera_device": args.camera_device,
                }

                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2)
                logger.info(f"Results saved to {args.output}")

            return 0 if result.validation_passed else 1

        finally:
            camera.cleanup()

    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        if args.verbose:
            import traceback

            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
