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

from src.const import LED_COUNT
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

    def __init__(self, device_id: int = 0):
        """
        Initialize camera with gain control.

        Args:
            device_id: Camera device ID (usually 0 for default camera)
        """
        self.device_id = device_id
        self.current_gain = 2.0
        self.cap: Optional[cv2.VideoCapture] = None

    def initialize(self) -> bool:
        """Initialize camera with fixed exposure and gain control."""
        try:
            logger.info(f"Initializing camera {self.device_id} with gain control")

            # Build initial pipeline with conservative gain
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
        Set camera gain by reinitializing with new pipeline.

        Args:
            gain_value: New gain value to set

        Returns:
            True if gain set successfully, False otherwise
        """
        try:
            logger.info(f"Setting camera gain to {gain_value}")

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
            Frame as RGB array, or None if failed
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

    return {
        "max_pixel_value": int(max_pixel),
        "saturated_pixel_count": int(saturated_pixels),
        "brightest_position": max_intensity_pos,
        "mean_brightness": float(np.mean(frame)),
        "has_saturation": saturated_pixels > 0,
    }


class LEDGainCalibrator:
    """LED gain calibration system with two-phase algorithm."""

    def __init__(self, wled_config: WLEDConfig, camera: GainControlledCamera, total_leds: int = LED_COUNT):
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
        Phase 1: Find gain where no test LEDs saturate.

        Returns:
            Tuple of (safe_gain, test_results_list)
        """
        current_gain = 4.0  # Start conservatively
        max_iterations = 10

        for iteration in range(max_iterations):
            logger.info(f"Testing gain {current_gain:.6f} (iteration {iteration + 1}/{max_iterations})")

            # Set camera gain
            if not self.camera.set_gain(current_gain):
                raise RuntimeError(f"Failed to set camera gain to {current_gain}")

            # Test all sample LEDs at current gain
            test_results = []
            saturated_count = 0

            for led_idx in self.test_led_indices:
                # Turn on single LED at full brightness
                self._set_single_led(led_idx, brightness=255)

                # Wait for LED and camera to stabilize
                time.sleep(0.1)

                # Capture and analyze frame
                frame = self.camera.capture_frame()
                if frame is None:
                    raise RuntimeError(f"Failed to capture frame for LED {led_idx}")

                analysis = analyze_frame_brightness(frame)

                result = LEDTestResult(
                    led_index=led_idx,
                    max_pixel_value=analysis["max_pixel_value"],
                    saturated_pixel_count=analysis["saturated_pixel_count"],
                    has_saturation=analysis["has_saturation"],
                    mean_brightness=analysis["mean_brightness"],
                    brightest_position=analysis["brightest_position"],
                    gain_setting=current_gain,
                )

                test_results.append(result)

                if result.has_saturation:
                    saturated_count += 1

                # Turn off LED
                self._turn_off_all_leds()
                time.sleep(0.05)

            logger.info(f"  Saturated LEDs: {saturated_count}/{len(self.test_led_indices)}")

            if saturated_count == 0:
                logger.info(f"  ✓ No saturation found at gain {current_gain:.6f}")
                return current_gain, test_results
            else:
                # Reduce gain and try again
                current_gain *= 0.85
                logger.info(f"  → Reducing gain to {current_gain:.6f}")

        raise RuntimeError("Could not find safe gain after maximum iterations")

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

        validation_leds = random.sample(self.test_led_indices, min(10, len(self.test_led_indices)))

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

        logger.info(f"Validation results:")
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
    parser.add_argument("--wled-port", type=int, default=21324, help="WLED DDP port")
    parser.add_argument("--total-leds", type=int, default=LED_COUNT, help="Total number of LEDs")

    # Camera configuration
    parser.add_argument("--camera-device", type=int, default=0, help="Camera device ID")

    # Output options
    parser.add_argument("--output", help="Output file to save calibration results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    try:
        logger.info("LED Gain Calibration Tool Starting")
        logger.info(f"Target: {args.wled_host}:{args.wled_port} ({args.total_leds} LEDs)")
        logger.info(f"Camera: Device {args.camera_device}")

        # Initialize WLED configuration
        wled_config = WLEDConfig(
            host=args.wled_host, port=args.wled_port, led_count=args.total_leds, timeout=5.0, persistent_retry=False
        )

        # Initialize camera
        camera = GainControlledCamera(device_id=args.camera_device)
        if not camera.initialize():
            logger.error("Failed to initialize camera")
            return 1

        try:
            # Initialize calibrator
            calibrator = LEDGainCalibrator(wled_config, camera, args.total_leds)

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
                    "total_leds": args.total_leds,
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
