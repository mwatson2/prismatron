#!/usr/bin/env python3
"""
WLED Test Patterns

Interactive test program for sending LED patterns to WLED controllers.
Supports solid colors, rainbow cycles, and animated rainbow effects.

Usage:
    python wled_test_patterns.py --help
    python wled_test_patterns.py solid --color 255 0 0
    python wled_test_patterns.py rainbow-cycle --speed 2.0
    python wled_test_patterns.py animated-rainbow --speed 1.0 --width 0.2
"""

import argparse
import colorsys
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.const import WLED_DEFAULT_HOST, WLED_DEFAULT_PORT
from src.consumer.wled_client import WLEDClient, WLEDConfig


def configure_camera_with_v4l2(
    camera_device: int,
    manual_gain: Optional[float] = None,
    exposure_time: Optional[int] = None,
    brightness: Optional[int] = None,
    contrast: Optional[int] = None,
):
    """
    Configure camera using v4l2-ctl for more reliable control than OpenCV.

    Args:
        camera_device: Camera device number (e.g., 0 for /dev/video0)
        manual_gain: Optional manual gain value
        exposure_time: Optional exposure time (3-2047, lower = darker)
        brightness: Optional brightness value (0-255, default 128)
        contrast: Optional contrast value (0-255, default 128)
    """
    device_path = f"/dev/video{camera_device}"
    print(f"Configuring camera {device_path} with v4l2-ctl...")

    # Commands to run - based on user's v4l2-ctl --list-ctrls output
    commands = [
        # Disable auto white balance (if not already disabled)
        ["v4l2-ctl", "-d", device_path, "-c", "white_balance_automatic=0"],
        # Set manual exposure mode (1 = Manual Mode, not 3 = Aperture Priority)
        ["v4l2-ctl", "-d", device_path, "-c", "auto_exposure=1"],
        # Disable continuous autofocus
        ["v4l2-ctl", "-d", device_path, "-c", "focus_automatic_continuous=0"],
        # Set fixed white balance temperature
        ["v4l2-ctl", "-d", device_path, "-c", "white_balance_temperature=4600"],
        # Set backlight compensation to 0 (already at default)
        ["v4l2-ctl", "-d", device_path, "-c", "backlight_compensation=0"],
        # Set exposure time to a fixed value (now that auto_exposure is manual)
        [
            "v4l2-ctl",
            "-d",
            device_path,
            "-c",
            f"exposure_time_absolute={exposure_time if exposure_time is not None else 250}",
        ],
    ]

    # Add brightness and contrast settings if specified
    if brightness is not None:
        commands.append(["v4l2-ctl", "-d", device_path, "-c", f"brightness={brightness}"])
    if contrast is not None:
        commands.append(["v4l2-ctl", "-d", device_path, "-c", f"contrast={contrast}"])

    # Set gain
    if manual_gain is not None:
        # Map our gain to v4l2 gain range (0-255)
        v4l2_gain = max(0, min(255, int(manual_gain)))
        commands.append(["v4l2-ctl", "-d", device_path, "-c", f"gain={v4l2_gain}"])
    else:
        # Set low gain for consistency (not max like current 255)
        commands.append(["v4l2-ctl", "-d", device_path, "-c", "gain=50"])

    # Execute commands
    success_count = 0
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                setting = cmd[4]  # The control=value part
                print(f"✓ Set {setting}")
                success_count += 1
            else:
                setting = cmd[4]
                print(f"✗ Failed to set {setting}: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout setting {cmd[4]}")
        except FileNotFoundError:
            print("✗ v4l2-ctl not found - install v4l-utils package")
            break
        except Exception as e:
            print(f"✗ Error setting {cmd[4]}: {e}")

    print(f"v4l2-ctl configuration: {success_count}/{len(commands)} settings applied")

    # Verify settings
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--list-ctrls"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("\nCurrent camera settings:")
            for line in result.stdout.split("\n"):
                if any(
                    ctrl in line for ctrl in ["auto_exposure", "focus_automatic", "white_balance_automatic", "gain"]
                ):
                    print(f"  {line.strip()}")
        else:
            print("Could not verify camera settings")
    except:
        print("Could not verify camera settings")

    print("v4l2-ctl configuration complete\n")


class LEDPatternGenerator:
    """Generator for various LED patterns and effects."""

    def __init__(self, led_count: int):
        self.led_count = led_count
        self.time_offset = 0.0

    def solid_color(self, r: int, g: int, b: int) -> bytes:
        """
        Generate solid color pattern.

        Args:
            r, g, b: RGB color values (0-255)

        Returns:
            LED data bytes
        """
        return bytes([r, g, b] * self.led_count)

    def single_led(self, led_index: int, r: int, g: int, b: int) -> bytes:
        """
        Generate pattern with single LED lit.

        Args:
            led_index: Index of LED to light (0-based)
            r, g, b: RGB color values (0-255)

        Returns:
            LED data bytes
        """
        led_data = bytearray([0, 0, 0] * self.led_count)
        if 0 <= led_index < self.led_count:
            led_data[led_index * 3] = r
            led_data[led_index * 3 + 1] = g
            led_data[led_index * 3 + 2] = b
        return bytes(led_data)

    def rainbow_cycle(self, speed: float = 1.0) -> bytes:
        """
        Generate rainbow cycle pattern where all LEDs show the same color,
        cycling through the rainbow spectrum.

        Args:
            speed: Cycle speed (cycles per second)

        Returns:
            LED data bytes
        """
        # Calculate hue based on time
        hue = (self.time_offset * speed) % 1.0

        # Convert HSV to RGB (full saturation and value)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

        # Convert to 0-255 range
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        return bytes([r, g, b] * self.led_count)

    def animated_rainbow(self, speed: float = 1.0, width: float = 1.0) -> bytes:
        """
        Generate animated rainbow pattern where different LEDs show different colors,
        creating a moving rainbow effect across the LED array.

        Args:
            speed: Animation speed (cycles per second)
            width: Rainbow width as fraction of LED array (0.1 = 10% of array)

        Returns:
            LED data bytes
        """
        led_data = []

        for led_index in range(self.led_count):
            # Calculate position-based hue offset
            position_offset = (led_index / self.led_count) * width

            # Add time-based animation
            time_offset = self.time_offset * speed

            # Calculate final hue
            hue = (position_offset + time_offset) % 1.0

            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

            # Convert to 0-255 range and add to data
            led_data.extend([int(r * 255), int(g * 255), int(b * 255)])

        return bytes(led_data)

    def wave_pattern(self, speed: float = 1.0, frequency: float = 2.0) -> bytes:
        """
        Generate sine wave pattern across the LED array.

        Args:
            speed: Wave animation speed
            frequency: Number of waves across the array

        Returns:
            LED data bytes
        """
        led_data = []

        for led_index in range(self.led_count):
            # Calculate wave position
            position = (led_index / self.led_count) * frequency * 2 * math.pi
            wave_value = (math.sin(position + self.time_offset * speed * 2 * math.pi) + 1) / 2

            # Convert to brightness
            brightness = int(wave_value * 255)

            # Create color (can be customized)
            led_data.extend([brightness, 0, brightness])  # Purple wave

        return bytes(led_data)

    def update_time(self, delta_time: float):
        """Update internal time for animations."""
        self.time_offset += delta_time


class WLEDTestRunner:
    """Test runner for WLED patterns."""

    def __init__(self, config: WLEDConfig):
        self.config = config
        self.client = WLEDClient(config)
        self.generator = LEDPatternGenerator(config.led_count)
        self.running = False
        self.start_time = 0.0

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\nShutdown requested...")
        self.running = False

    def run_solid_color(self, r: int, g: int, b: int, duration: Optional[float] = None):
        """
        Run solid color pattern.

        Args:
            r, g, b: RGB color values
            duration: Run duration in seconds (None for infinite)
        """
        print(f"Setting all {self.config.led_count} LEDs to RGB({r}, {g}, {b})")

        if not self.client.connect():
            print("Error: Failed to connect to WLED controller")
            return False

        try:
            success = self.client.set_solid_color(r, g, b)
            if success:
                print("✓ Solid color pattern sent successfully")
                if duration:
                    print(f"Holding for {duration} seconds...")
                    time.sleep(duration)
            else:
                print("✗ Failed to send solid color pattern")
                return False

        finally:
            self.client.disconnect()

        return True

    def run_single_led(self, led_index: int, r: int, g: int, b: int, duration: Optional[float] = None):
        """
        Light a single LED and keep it lit by continuously sending packets.

        Args:
            led_index: Index of LED to light (0-based)
            r, g, b: RGB color values
            duration: Run duration in seconds (None for infinite)
        """
        print(f"Lighting LED {led_index} with RGB({r}, {g}, {b})")

        if led_index < 0 or led_index >= self.config.led_count:
            print(f"Error: LED index {led_index} out of range (0-{self.config.led_count-1})")
            return False

        if not self.client.connect():
            print("Error: Failed to connect to WLED controller")
            return False

        try:
            self.running = True
            self.start_time = time.time()
            frame_count = 0
            resend_interval = 0.2  # Send packet every 200ms to keep LED lit

            # Generate the LED data once
            led_data = self.generator.single_led(led_index, r, g, b)

            print(f"Sending packets every {resend_interval*1000:.0f}ms to maintain LED state")
            if duration:
                print(f"Running for {duration} seconds...")
            else:
                print("Running until interrupted (Ctrl+C to stop)...")

            last_send = 0
            while self.running:
                current_time = time.time()

                # Check duration
                if duration and (current_time - self.start_time) >= duration:
                    break

                # Send packet at regular intervals
                if current_time - last_send >= resend_interval:
                    success = self.client.send_led_data(led_data)
                    if not success:
                        print(f"\nWarning: Failed to send frame {frame_count}")
                    else:
                        frame_count += 1
                        if frame_count % 25 == 0:  # Status every ~5 seconds
                            elapsed = current_time - self.start_time
                            print(f"Status: {frame_count} packets sent, {elapsed:.1f}s elapsed")
                    last_send = current_time

                # Small sleep to avoid busy waiting
                time.sleep(0.01)

            # Turn off the LED when done
            print("\nTurning off LED...")
            self.client.set_solid_color(0, 0, 0)

            # Final statistics
            elapsed = time.time() - self.start_time
            print(f"\nCompleted: {frame_count} packets sent over {elapsed:.1f} seconds")

        except Exception as e:
            print(f"\nError during single LED execution: {e}")
            return False
        finally:
            self.client.disconnect()

        return True

    def run_brightness_contrast_test(
        self,
        led_index: int,
        r: int,
        g: int,
        b: int,
        camera_device: int = 0,
        output_dir: str = "brightness_contrast",
        camera_config: dict = None,
        brightness_values: List[int] = None,
        contrast_values: List[int] = None,
    ):
        """
        Test brightness and contrast settings effect on LED capture.

        Args:
            led_index: Index of LED to light (0-based)
            r, g, b: RGB color values
            camera_device: Camera device ID for capture
            output_dir: Directory to save captured images
            camera_config: Camera configuration dict (from JSON file)
            brightness_values: List of brightness values to test (default: [64, 128, 192])
            contrast_values: List of contrast values to test (default: [64, 128, 192])
        """
        if brightness_values is None:
            brightness_values = [64, 128, 192]  # Low, medium, high
        if contrast_values is None:
            contrast_values = [64, 128, 192]  # Low, medium, high

        # Process camera configuration
        crop_region = None
        flip_image = False
        use_usb = False
        camera_resolution = None

        if camera_config:
            if camera_device == 0:
                camera_device = camera_config.get("camera_device", 0)
            crop_config = camera_config.get("crop_region")
            if crop_config:
                crop_region = (crop_config["x"], crop_config["y"], crop_config["width"], crop_config["height"])
                print(f"Using crop region from config: {crop_region}")
            resolution_config = camera_config.get("camera_resolution")
            if resolution_config:
                camera_resolution = (resolution_config["width"], resolution_config["height"])
                print(f"Using camera resolution from config: {camera_resolution}")
            if camera_config.get("use_usb", False):
                use_usb = True
                print("Using USB camera mode from config")
            flip_image = camera_config.get("flip_image", False)
            if flip_image:
                print("Using flip image mode from config")

        # Determine color name
        if r == 255 and g == 0 and b == 0:
            color_name = "RED"
        elif r == 0 and g == 255 and b == 0:
            color_name = "GREEN"
        elif r == 0 and g == 0 and b == 255:
            color_name = "BLUE"
        else:
            color_name = f"RGB_{r}_{g}_{b}"

        print(f"=== Brightness/Contrast Test ===")
        print(f"LED Index: {led_index}")
        print(f"LED RGB: ({r}, {g}, {b}) - {color_name}")
        print(f"Brightness values: {brightness_values}")
        print(f"Contrast values: {contrast_values}")
        print(f"Output directory: {output_dir}")
        print()

        if led_index < 0 or led_index >= self.config.led_count:
            print(f"Error: LED index {led_index} out of range (0-{self.config.led_count-1})")
            return False

        if not self.client.connect():
            print("Error: Failed to connect to WLED controller")
            return False

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Storage for results
        test_results = []

        try:
            # Generate the LED data once
            led_data = self.generator.single_led(led_index, r, g, b)

            # Send LED command and wait for it to take effect
            print("Setting LED and waiting for WLED command to take effect...")
            self.client.send_led_data(led_data)
            time.sleep(2.0)  # Wait for WLED latency

            # Fixed gain and exposure for brightness/contrast testing
            fixed_gain = 200  # High gain
            fixed_exposure = 100  # Moderate exposure

            print(f"Using fixed settings: Gain={fixed_gain}, Exposure={fixed_exposure}")
            print("=" * 60)

            for brightness in brightness_values:
                for contrast in contrast_values:
                    print(f"\nTesting Brightness={brightness}, Contrast={contrast}")
                    print("-" * 40)

                    if use_usb:
                        print(f"Configuring USB camera at /dev/video{camera_device}")
                        cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
                        if not cap.isOpened():
                            cap = cv2.VideoCapture(camera_device)
                        if not cap.isOpened():
                            print("Error: Failed to open USB camera")
                            continue

                        # Configure camera with v4l2-ctl
                        configure_camera_with_v4l2(camera_device, fixed_gain, fixed_exposure, brightness, contrast)

                        # Set camera properties
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        if camera_resolution:
                            width, height = camera_resolution
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        else:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    else:
                        print("Error: Brightness/contrast test only supports USB cameras")
                        return False

                    # Warm up camera
                    print("Warming up camera...")
                    for _ in range(10):
                        cap.read()
                        time.sleep(0.05)

                    # Resend LED data to ensure it's still lit
                    self.client.send_led_data(led_data)
                    time.sleep(0.5)

                    # Capture frame
                    print("Capturing image...")

                    # Flush buffer and capture
                    for _ in range(3):
                        cap.read()
                        time.sleep(0.05)

                    ret, frame = cap.read()
                    if ret:
                        # Process frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        if flip_image:
                            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)

                        if crop_region:
                            x, y, w, h = crop_region
                            frame_rgb = frame_rgb[y : y + h, x : x + w]

                        frame_rgb = cv2.resize(frame_rgb, (800, 480), interpolation=cv2.INTER_LINEAR)

                        # Find LED region
                        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

                        # Calculate visible area (pixels above threshold)
                        thresh_binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
                        visible_pixels = np.sum(thresh_binary > 0)

                        # Find brightest point
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

                        # Extract ROI around brightest point
                        roi_size = 32
                        x, y = max_loc[0], max_loc[1]
                        y1 = max(0, y - roi_size // 2)
                        y2 = min(frame_rgb.shape[0], y + roi_size // 2)
                        x1 = max(0, x - roi_size // 2)
                        x2 = min(frame_rgb.shape[1], x + roi_size // 2)

                        if y2 > y1 and x2 > x1:
                            roi = frame_rgb[y1:y2, x1:x2]
                            roi_mean_r = np.mean(roi[:, :, 0])
                            roi_mean_g = np.mean(roi[:, :, 1])
                            roi_mean_b = np.mean(roi[:, :, 2])
                            roi_peak_r = np.max(roi[:, :, 0])
                            roi_peak_g = np.max(roi[:, :, 1])
                            roi_peak_b = np.max(roi[:, :, 2])
                        else:
                            roi_mean_r = roi_mean_g = roi_mean_b = 0
                            roi_peak_r = roi_peak_g = roi_peak_b = 0

                        # Store results
                        result = {
                            "brightness": brightness,
                            "contrast": contrast,
                            "roi_mean": (roi_mean_r, roi_mean_g, roi_mean_b),
                            "roi_peak": (roi_peak_r, roi_peak_g, roi_peak_b),
                            "visible_pixels": visible_pixels,
                            "max_brightness": max_val,
                            "saturated": max_val >= 254,
                        }
                        test_results.append(result)

                        # Create annotated image
                        annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        font = cv2.FONT_HERSHEY_SIMPLEX

                        # Add annotations
                        cv2.putText(
                            annotated_frame, f"B={brightness} C={contrast}", (10, 30), font, 0.8, (255, 255, 255), 2
                        )
                        cv2.putText(
                            annotated_frame, f"Visible: {visible_pixels} px", (10, 60), font, 0.6, (255, 255, 255), 2
                        )
                        cv2.putText(
                            annotated_frame,
                            f"ROI Mean: ({roi_mean_r:.1f},{roi_mean_g:.1f},{roi_mean_b:.1f})",
                            (10, 90),
                            font,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            annotated_frame,
                            f"ROI Peak: ({roi_peak_r:.0f},{roi_peak_g:.0f},{roi_peak_b:.0f})",
                            (10, 120),
                            font,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                        # Add saturation warning if needed
                        if max_val >= 254:
                            cv2.putText(annotated_frame, "SATURATED!", (10, 150), font, 0.8, (0, 0, 255), 2)

                        # Draw rectangle around LED region
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Save image
                        filename = f"B{brightness}_C{contrast}.jpg"
                        filepath = Path(output_dir) / filename
                        cv2.imwrite(str(filepath), annotated_frame)

                        print(f"  Visible pixels: {visible_pixels}")
                        print(f"  ROI Mean RGB: ({roi_mean_r:.1f}, {roi_mean_g:.1f}, {roi_mean_b:.1f})")
                        print(f"  ROI Peak RGB: ({roi_peak_r:.0f}, {roi_peak_g:.0f}, {roi_peak_b:.0f})")
                        print(f"  Max brightness: {max_val:.0f} {'[SATURATED]' if max_val >= 254 else '[OK]'}")
                        print(f"  Saved: {filename}")

                    else:
                        print("  Failed to capture image")

                    # Cleanup camera
                    cap.release()
                    time.sleep(0.5)

            # Generate summary report
            self._generate_brightness_contrast_report(test_results, led_index, color_name, output_dir)

        except Exception as e:
            print(f"\nError during brightness/contrast test: {e}")
            return False
        finally:
            # Turn off LED
            print("\nTurning off LED...")
            self.client.set_solid_color(0, 0, 0)
            self.client.disconnect()

        return True

    def run_exposure_test(
        self,
        led_index: int,
        r: int,
        g: int,
        b: int,
        camera_device: int = 0,
        output_dir: str = "exposure_test",
        camera_config: dict = None,
        exposure_values: List[int] = None,
        gain_values: List[int] = None,
    ):
        """
        Test exposure_time_absolute and gain combinations effect on LED capture.

        Args:
            led_index: Index of LED to light (0-based)
            r, g, b: RGB color values
            camera_device: Camera device ID for capture
            output_dir: Directory to save captured images
            camera_config: Camera configuration dict (from JSON file)
            exposure_values: List of exposure values to test (default: [50, 100, 200, 400, 800])
            gain_values: List of gain values to test (default: [0, 10, 20, 30, 40, 50])
        """
        if exposure_values is None:
            exposure_values = [50, 100, 200, 400, 800]  # Safe range within camera limits (3-2047)
        if gain_values is None:
            gain_values = [0, 10, 20, 30, 40, 50]  # Range from no gain to moderate gain

        # Process camera configuration
        crop_region = None
        flip_image = False
        use_usb = False
        camera_resolution = None

        if camera_config:
            if camera_device == 0:
                camera_device = camera_config.get("camera_device", 0)
            crop_config = camera_config.get("crop_region")
            if crop_config:
                crop_region = (crop_config["x"], crop_config["y"], crop_config["width"], crop_config["height"])
                print(f"Using crop region from config: {crop_region}")
            resolution_config = camera_config.get("camera_resolution")
            if resolution_config:
                camera_resolution = (resolution_config["width"], resolution_config["height"])
                print(f"Using camera resolution from config: {camera_resolution}")
            if camera_config.get("use_usb", False):
                use_usb = True
                print("Using USB camera mode from config")
            flip_image = camera_config.get("flip_image", False)
            if flip_image:
                print("Using flip image mode from config")

        # Determine color name
        if r == 255 and g == 0 and b == 0:
            color_name = "RED"
        elif r == 0 and g == 255 and b == 0:
            color_name = "GREEN"
        elif r == 0 and g == 0 and b == 255:
            color_name = "BLUE"
        else:
            color_name = f"RGB_{r}_{g}_{b}"

        print(f"=== Exposure/Gain Test ===")
        print(f"LED Index: {led_index}")
        print(f"LED RGB: ({r}, {g}, {b}) - {color_name}")
        print(f"Exposure values: {exposure_values}")
        print(f"Gain values: {gain_values}")
        print(f"Output directory: {output_dir}")
        print()

        if led_index < 0 or led_index >= self.config.led_count:
            print(f"Error: LED index {led_index} out of range (0-{self.config.led_count-1})")
            return False

        if not self.client.connect():
            print("Error: Failed to connect to WLED controller")
            return False

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Storage for results
        test_results = []

        try:
            # Generate the LED data once
            led_data = self.generator.single_led(led_index, r, g, b)

            # Send LED command and wait for it to take effect
            print("Setting LED and waiting for WLED command to take effect...")
            self.client.send_led_data(led_data)
            time.sleep(2.0)  # Wait for WLED latency

            # Fixed settings for exposure/gain testing
            fixed_brightness = 128  # Keep background black
            fixed_contrast = 64  # Lower contrast to avoid over-enhancement

            print(f"Using fixed settings: Brightness={fixed_brightness}, Contrast={fixed_contrast}")
            print("=" * 60)

            for gain in gain_values:
                for exposure in exposure_values:
                    print(f"\nTesting Gain={gain}, Exposure={exposure}")
                    print("-" * 40)

                    if use_usb:
                        print(f"Configuring USB camera at /dev/video{camera_device}")
                        cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
                        if not cap.isOpened():
                            cap = cv2.VideoCapture(camera_device)
                        if not cap.isOpened():
                            print("Error: Failed to open USB camera")
                            continue

                        # Configure camera with v4l2-ctl
                        configure_camera_with_v4l2(camera_device, gain, exposure, fixed_brightness, fixed_contrast)

                        # Set camera properties
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        if camera_resolution:
                            width, height = camera_resolution
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        else:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    else:
                        print("Error: Exposure test only supports USB cameras")
                        continue

                    # Warm up camera
                    print("Warming up camera...")
                    for _ in range(10):
                        ret, frame = cap.read()
                        if not ret:
                            print("Error: Failed to capture frame during warmup")
                            break

                    # Capture image
                    print("Capturing image...")
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture frame")
                        cap.release()
                        continue

                    # Apply crop if specified
                    if crop_region:
                        x, y, w, h = crop_region
                        frame = frame[y : y + h, x : x + w]

                    # Flip image if specified
                    if flip_image:
                        frame = cv2.flip(frame, -1)  # Flip both horizontally and vertically

                    # Analyze the captured image
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Count visible pixels (threshold at 30 to detect dim diffusion)
                    visible_pixels = np.sum(gray > 30)

                    # Calculate ROI statistics
                    roi_mean_bgr = np.mean(frame, axis=(0, 1))
                    roi_peak_bgr = np.max(frame, axis=(0, 1))
                    max_brightness = np.max(frame)

                    # Check for saturation
                    is_saturated = max_brightness >= 254
                    saturation_status = "SATURATED" if is_saturated else "OK"

                    print(f"  Visible pixels: {visible_pixels}")
                    print(f"  ROI Mean RGB: ({roi_mean_bgr[2]:.1f}, {roi_mean_bgr[1]:.1f}, {roi_mean_bgr[0]:.1f})")
                    print(f"  ROI Peak RGB: ({roi_peak_bgr[2]}, {roi_peak_bgr[1]}, {roi_peak_bgr[0]})")
                    print(f"  Max brightness: {max_brightness} [{saturation_status}]")

                    # Save annotated image
                    output_image = frame.copy()

                    # Add text annotations
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2

                    # Settings label
                    settings_text = f"G={gain}, E={exposure}"
                    cv2.putText(output_image, settings_text, (10, 30), font, font_scale, (255, 255, 255), thickness)

                    # Visible pixels count
                    pixels_text = f"Visible: {visible_pixels}"
                    cv2.putText(output_image, pixels_text, (10, 60), font, font_scale, (255, 255, 255), thickness)

                    # ROI mean values
                    mean_text = f"Mean: ({roi_mean_bgr[2]:.1f}, {roi_mean_bgr[1]:.1f}, {roi_mean_bgr[0]:.1f})"
                    cv2.putText(output_image, mean_text, (10, 90), font, font_scale, (255, 255, 255), thickness)

                    # ROI peak values
                    peak_text = f"Peak: ({roi_peak_bgr[2]}, {roi_peak_bgr[1]}, {roi_peak_bgr[0]})"
                    cv2.putText(output_image, peak_text, (10, 120), font, font_scale, (255, 255, 255), thickness)

                    # Saturation warning
                    if is_saturated:
                        cv2.putText(output_image, "SATURATED", (10, 150), font, font_scale, (0, 0, 255), thickness)

                    # Draw ROI rectangle (approximate LED region)
                    if crop_region:
                        h, w = output_image.shape[:2]
                        roi_x, roi_y = int(w * 0.4), int(h * 0.3)
                        roi_w, roi_h = int(w * 0.2), int(h * 0.4)
                        cv2.rectangle(output_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

                    # Save image
                    filename = f"G{gain}_E{exposure}.jpg"
                    filepath = Path(output_dir) / filename
                    cv2.imwrite(str(filepath), output_image)
                    print(f"  Saved: {filename}")

                    # Store results for analysis
                    test_results.append(
                        {
                            "gain": gain,
                            "exposure": exposure,
                            "visible_pixels": visible_pixels,
                            "mean_green": roi_mean_bgr[1],
                            "peak_green": roi_peak_bgr[1],
                            "is_saturated": is_saturated,
                            "filename": filename,
                        }
                    )

                    cap.release()

                    # Small delay to allow camera to reset between tests
                    time.sleep(0.5)

            # Generate report
            self._generate_exposure_report(test_results, led_index, color_name, output_dir)

        except Exception as e:
            print(f"\nError during exposure test: {e}")
            return False
        finally:
            # Turn off LED
            print("\nTurning off LED...")
            self.client.set_solid_color(0, 0, 0)
            self.client.disconnect()

        return True

    def _generate_exposure_report(self, results: List[dict], led_index: int, color_name: str, output_dir: str):
        """Generate exposure/gain test report."""
        print(f"\n{'='*80}")
        print(f"EXPOSURE/GAIN ANALYSIS - LED {led_index} {color_name}")
        print(f"{'='*80}")
        print()
        print(f"{'Gain':<6} {'Exposure':<10} {'Visible Pixels':<15} {'Mean Green':<12} {'Peak Green':<12} {'Status'}")
        print(f"{'-'*6} {'-'*10} {'-'*15} {'-'*12} {'-'*12} {'-'*10}")

        for result in results:
            status = "SATURATED" if result["is_saturated"] else "OK"
            print(
                f"{result['gain']:<6} {result['exposure']:<10} {result['visible_pixels']:<15} {result['mean_green']:<12.1f} {result['peak_green']:<12} {status}"
            )

        print()
        print(f"{'='*80}")
        print("KEY FINDINGS:")
        print(f"{'='*80}")

        # Find best non-saturated configuration
        non_saturated = [r for r in results if not r["is_saturated"]]
        if non_saturated:
            best = max(non_saturated, key=lambda x: x["visible_pixels"])
            print(
                f"Best diffusion visibility (no saturation): Gain={best['gain']}, Exposure={best['exposure']} ({best['visible_pixels']} pixels)"
            )

        # Count configurations without saturation
        total_configs = len(results)
        non_saturated_count = len(non_saturated)
        print(f"Configurations without saturation: {non_saturated_count}/{total_configs}")

        # Show range of visible pixels
        if results:
            min_pixels = min(r["visible_pixels"] for r in results)
            max_pixels = max(r["visible_pixels"] for r in results)
            print(f"Visible pixels range: {min_pixels}-{max_pixels}")

        # Show configurations grouped by gain level
        gain_groups = {}
        for result in results:
            gain = result["gain"]
            if gain not in gain_groups:
                gain_groups[gain] = []
            gain_groups[gain].append(result)

        print()
        print("ANALYSIS BY GAIN LEVEL:")
        for gain in sorted(gain_groups.keys()):
            group = gain_groups[gain]
            non_sat_count = len([r for r in group if not r["is_saturated"]])
            if non_sat_count > 0:
                best_in_group = max([r for r in group if not r["is_saturated"]], key=lambda x: x["visible_pixels"])
                print(
                    f"  Gain {gain}: {non_sat_count}/{len(group)} non-saturated, best: E={best_in_group['exposure']} ({best_in_group['visible_pixels']} pixels)"
                )
            else:
                print(f"  Gain {gain}: All {len(group)} configurations saturated")

        print(f"{'='*80}")

    def _generate_brightness_contrast_report(
        self, results: List[dict], led_index: int, color_name: str, output_dir: str
    ):
        """Generate brightness/contrast test report."""
        print(f"\n{'='*80}")
        print(f"BRIGHTNESS/CONTRAST ANALYSIS - LED {led_index} {color_name}")
        print(f"{'='*80}")

        # Table of results
        print(
            f"\n{'Brightness':<12} {'Contrast':<12} {'Visible Pixels':<15} {'Mean Green':<12} {'Peak Green':<12} {'Status'}"
        )
        print(f"{'-'*12} {'-'*12} {'-'*15} {'-'*12} {'-'*12} {'-'*10}")

        for r in results:
            mean_g = r["roi_mean"][1]
            peak_g = r["roi_peak"][1]
            status = "SATURATED" if r["saturated"] else "OK"
            print(
                f"{r['brightness']:<12} {r['contrast']:<12} {r['visible_pixels']:<15} "
                f"{mean_g:<12.1f} {peak_g:<12.0f} {status}"
            )

        # Analysis
        print(f"\n{'='*80}")
        print("KEY FINDINGS:")
        print(f"{'='*80}")

        # Find configuration with largest visible area
        if results:
            best_area = max(results, key=lambda x: x["visible_pixels"] if not x["saturated"] else 0)
            print(
                f"Best diffusion visibility (no saturation): Brightness={best_area['brightness']}, "
                f"Contrast={best_area['contrast']} ({best_area['visible_pixels']} pixels)"
            )

            # Find configurations without saturation
            no_sat = [r for r in results if not r["saturated"]]
            if no_sat:
                print(f"Configurations without saturation: {len(no_sat)}/{len(results)}")

            # Analyze effect of brightness
            for contrast in [64, 128, 192]:
                contrast_results = [r for r in results if r["contrast"] == contrast]
                if contrast_results:
                    pixels = [r["visible_pixels"] for r in contrast_results]
                    print(f"Contrast={contrast}: Visible pixels range {min(pixels)}-{max(pixels)}")

        print(f"{'='*80}")

    def run_gain_calibration(
        self,
        led_index: int,
        r: int,
        g: int,
        b: int,
        camera_device: int = 0,
        output_dir: str = "gain_calibration",
        camera_config: dict = None,
        gain_values: List[int] = None,
    ):
        """
        Capture the same LED at multiple gain values for gain calibration.

        Args:
            led_index: Index of LED to light (0-based)
            r, g, b: RGB color values
            camera_device: Camera device ID for capture
            output_dir: Directory to save captured images
            camera_config: Camera configuration dict (from JSON file)
            gain_values: List of gain values to test (default: [10, 25, 50, 75, 100, 150, 200])
        """
        if gain_values is None:
            gain_values = [10, 25, 50, 75, 100, 150, 200]  # Range of gain values to test

        # Process camera configuration if provided (same as capture_test)
        crop_region = None
        flip_image = False
        use_usb = False
        camera_resolution = None

        if camera_config:
            if camera_device == 0:
                camera_device = camera_config.get("camera_device", 0)
            crop_config = camera_config.get("crop_region")
            if crop_config:
                crop_region = (crop_config["x"], crop_config["y"], crop_config["width"], crop_config["height"])
                print(f"Using crop region from config: {crop_region}")
            resolution_config = camera_config.get("camera_resolution")
            if resolution_config:
                camera_resolution = (resolution_config["width"], resolution_config["height"])
                print(f"Using camera resolution from config: {camera_resolution}")
            if camera_config.get("use_usb", False):
                use_usb = True
                print("Using USB camera mode from config")
            flip_image = camera_config.get("flip_image", False)
            if flip_image:
                print("Using flip image mode from config")

        # Determine color name
        if r == 255 and g == 0 and b == 0:
            color_name = "RED"
        elif r == 0 and g == 255 and b == 0:
            color_name = "GREEN"
        elif r == 0 and g == 0 and b == 255:
            color_name = "BLUE"
        else:
            color_name = f"RGB_{r}_{g}_{b}"

        print(f"=== LED Gain Calibration Test ===")
        print(f"LED Index: {led_index}")
        print(f"LED RGB: ({r}, {g}, {b}) - {color_name}")
        print(f"Gain values to test: {gain_values}")
        print(f"Output directory: {output_dir}")
        print()

        if led_index < 0 or led_index >= self.config.led_count:
            print(f"Error: LED index {led_index} out of range (0-{self.config.led_count-1})")
            return False

        if not self.client.connect():
            print("Error: Failed to connect to WLED controller")
            return False

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Storage for gain calibration results
        gain_results = []

        try:
            # Generate the LED data once
            led_data = self.generator.single_led(led_index, r, g, b)

            # Send LED command and wait for it to take effect
            print("Setting LED and waiting for WLED command to take effect...")
            self.client.send_led_data(led_data)
            time.sleep(2.0)  # Wait for WLED latency

            for i, gain_value in enumerate(gain_values):
                print(f"\n{'='*60}")
                print(f"Testing Gain {gain_value} ({i+1}/{len(gain_values)})")
                print(f"{'='*60}")

                # Initialize camera with current gain value
                print(f"Initializing camera with gain {gain_value}...")

                if use_usb:
                    print(f"Using USB camera at /dev/video{camera_device}")
                    cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
                    if not cap.isOpened():
                        print("V4L2 backend failed, trying default backend")
                        cap = cv2.VideoCapture(camera_device)
                    if not cap.isOpened():
                        print("Error: Failed to open USB camera")
                        continue

                    # Configure camera with v4l2-ctl using current gain
                    configure_camera_with_v4l2(camera_device, gain_value)

                    # Set camera properties
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    if camera_resolution:
                        width, height = camera_resolution
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    else:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                else:
                    print("Error: Gain calibration only supports USB cameras")
                    return False

                # Warm up camera
                print("Warming up camera...")
                for _ in range(10):
                    cap.read()
                    time.sleep(0.05)

                # Resend LED data to ensure it's still lit
                self.client.send_led_data(led_data)
                time.sleep(0.5)

                # Capture frame
                print(f"Capturing image with gain {gain_value}...")

                # Flush buffer and capture
                for _ in range(3):
                    cap.read()
                    time.sleep(0.05)

                ret, frame = cap.read()
                if ret:
                    # Process frame same as capture_test
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if flip_image:
                        frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)

                    if crop_region:
                        x, y, w, h = crop_region
                        frame_rgb = frame_rgb[y : y + h, x : x + w]

                    frame_rgb = cv2.resize(frame_rgb, (800, 480), interpolation=cv2.INTER_LINEAR)

                    # Find LED region
                    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

                    roi_size = 32
                    x, y = max_loc[0], max_loc[1]
                    y1 = max(0, y - roi_size // 2)
                    y2 = min(frame_rgb.shape[0], y + roi_size // 2)
                    x1 = max(0, x - roi_size // 2)
                    x2 = min(frame_rgb.shape[1], x + roi_size // 2)

                    if y2 > y1 and x2 > x1:
                        roi = frame_rgb[y1:y2, x1:x2]
                        roi_mean_r = np.mean(roi[:, :, 0])
                        roi_mean_g = np.mean(roi[:, :, 1])
                        roi_mean_b = np.mean(roi[:, :, 2])
                        roi_peak_r = np.max(roi[:, :, 0])
                        roi_peak_g = np.max(roi[:, :, 1])
                        roi_peak_b = np.max(roi[:, :, 2])
                    else:
                        roi_mean_r = roi_mean_g = roi_mean_b = 0
                        roi_peak_r = roi_peak_g = roi_peak_b = 0

                    # Calculate frame mean
                    frame_mean_r = np.mean(frame_rgb[:, :, 0])
                    frame_mean_g = np.mean(frame_rgb[:, :, 1])
                    frame_mean_b = np.mean(frame_rgb[:, :, 2])

                    # Store results
                    result = {
                        "gain": gain_value,
                        "frame_mean": (frame_mean_r, frame_mean_g, frame_mean_b),
                        "roi_mean": (roi_mean_r, roi_mean_g, roi_mean_b),
                        "roi_peak": (roi_peak_r, roi_peak_g, roi_peak_b),
                        "brightest_point": (x, y),
                        "max_brightness": max_val,
                        "saturated": max_val >= 254,
                    }
                    gain_results.append(result)

                    # Create annotated image
                    annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Add comprehensive annotations
                    cv2.putText(
                        annotated_frame,
                        f"LED {led_index} {color_name} - Gain {gain_value}",
                        (10, 30),
                        font,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(annotated_frame, f"LED RGB: ({r}, {g}, {b})", (10, 60), font, 0.6, (255, 255, 255), 2)
                    cv2.putText(
                        annotated_frame,
                        f"Frame Mean: ({frame_mean_r:.1f}, {frame_mean_g:.1f}, {frame_mean_b:.1f})",
                        (10, 90),
                        font,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"ROI Mean: ({roi_mean_r:.1f}, {roi_mean_g:.1f}, {roi_mean_b:.1f})",
                        (10, 120),
                        font,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"ROI Peak: ({roi_peak_r:.0f}, {roi_peak_g:.0f}, {roi_peak_b:.0f})",
                        (10, 150),
                        font,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    # Add saturation warning if needed
                    if max_val >= 254:
                        cv2.putText(annotated_frame, "SATURATED!", (10, 180), font, 0.8, (0, 0, 255), 2)

                    # Draw rectangle around LED region
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "LED", (x1, y1 - 5), font, 0.5, (0, 255, 0), 1)

                    # Save image
                    filename = f"LED_{led_index:04d}_{color_name}_Gain_{gain_value:03d}.jpg"
                    filepath = Path(output_dir) / filename
                    cv2.imwrite(str(filepath), annotated_frame)

                    print(
                        f"✓ Gain {gain_value}: Mean=({roi_mean_r:.1f},{roi_mean_g:.1f},{roi_mean_b:.1f}) Peak=({roi_peak_r:.0f},{roi_peak_g:.0f},{roi_peak_b:.0f}) {'SATURATED' if max_val >= 254 else 'OK'}"
                    )

                else:
                    print(f"✗ Failed to capture image with gain {gain_value}")

                # Cleanup camera for this gain
                cap.release()
                time.sleep(0.5)  # Brief pause between gain changes

            # Generate summary report
            self._generate_gain_calibration_report(gain_results, led_index, color_name, output_dir)

        except Exception as e:
            print(f"\nError during gain calibration: {e}")
            return False
        finally:
            # Turn off LED
            print("\nTurning off LED...")
            self.client.set_solid_color(0, 0, 0)
            self.client.disconnect()

        return True

    def _generate_gain_calibration_report(
        self, gain_results: List[dict], led_index: int, color_name: str, output_dir: str
    ):
        """Generate a comprehensive gain calibration report."""
        print(f"\n{'='*80}")
        print(f"GAIN CALIBRATION REPORT - LED {led_index} {color_name}")
        print(f"{'='*80}")

        # Table header
        print(f"{'Gain':<6} {'ROI Mean RGB':<20} {'ROI Peak RGB':<20} {'Peak Bright':<12} {'Status'}")
        print(f"{'-'*6} {'-'*20} {'-'*20} {'-'*12} {'-'*10}")

        # Data rows
        for result in gain_results:
            gain = result["gain"]
            roi_mean = result["roi_mean"]
            roi_peak = result["roi_peak"]
            max_bright = result["max_brightness"]
            status = "SATURATED" if result["saturated"] else "OK"

            mean_str = f"({roi_mean[0]:.1f},{roi_mean[1]:.1f},{roi_mean[2]:.1f})"
            peak_str = f"({roi_peak[0]:.0f},{roi_peak[1]:.0f},{roi_peak[2]:.0f})"

            print(f"{gain:<6} {mean_str:<20} {peak_str:<20} {max_bright:<12.0f} {status}")

        # Analysis
        print(f"\n{'='*80}")
        print("GAIN ANALYSIS")
        print(f"{'='*80}")

        # Find optimal gain suggestions
        non_saturated = [r for r in gain_results if not r["saturated"]]
        saturated = [r for r in gain_results if r["saturated"]]

        if non_saturated:
            # Find highest gain without saturation
            best_non_saturated = max(non_saturated, key=lambda x: x["gain"])
            print(f"Highest non-saturated gain: {best_non_saturated['gain']}")

            # Find gain with best dynamic range (highest peak values without saturation)
            if color_name == "RED":
                best_dynamic = max(non_saturated, key=lambda x: x["roi_peak"][0])
                print(f"Best red dynamic range: Gain {best_dynamic['gain']} (peak R={best_dynamic['roi_peak'][0]:.0f})")
            elif color_name == "GREEN":
                best_dynamic = max(non_saturated, key=lambda x: x["roi_peak"][1])
                print(
                    f"Best green dynamic range: Gain {best_dynamic['gain']} (peak G={best_dynamic['roi_peak'][1]:.0f})"
                )
            elif color_name == "BLUE":
                best_dynamic = max(non_saturated, key=lambda x: x["roi_peak"][2])
                print(
                    f"Best blue dynamic range: Gain {best_dynamic['gain']} (peak B={best_dynamic['roi_peak'][2]:.0f})"
                )

        if saturated:
            print(f"Saturated gains: {[r['gain'] for r in saturated]}")

        # Recommendations
        print(f"\nRECOMMENDations:")
        if non_saturated:
            if len(non_saturated) >= 2:
                # Suggest a gain in the upper range of non-saturated values
                high_gains = sorted([r["gain"] for r in non_saturated])[-2:]
                print(f"• Recommended gain range: {high_gains[0]}-{high_gains[1]} (good dynamic range, no saturation)")
            else:
                print(f"• Recommended gain: {best_non_saturated['gain']} (highest without saturation)")
        else:
            print("• All gains cause saturation - consider lower LED brightness or darker environment")

        # Save report to file
        report_file = Path(output_dir) / f"gain_calibration_report_LED_{led_index:04d}_{color_name}.txt"
        with open(report_file, "w") as f:
            f.write(f"Gain Calibration Report - LED {led_index} {color_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write("Gain  ROI_Mean_RGB        ROI_Peak_RGB        Peak_Bright  Status\n")
            f.write("-" * 60 + "\n")
            for result in gain_results:
                mean_str = f"({result['roi_mean'][0]:.1f},{result['roi_mean'][1]:.1f},{result['roi_mean'][2]:.1f})"
                peak_str = f"({result['roi_peak'][0]:.0f},{result['roi_peak'][1]:.0f},{result['roi_peak'][2]:.0f})"
                status = "SATURATED" if result["saturated"] else "OK"
                f.write(
                    f"{result['gain']:<5} {mean_str:<19} {peak_str:<19} {result['max_brightness']:<12.0f} {status}\n"
                )

        print(f"\n✓ Report saved: {report_file}")
        print(f"✓ Images saved in: {output_dir}/")
        print(f"{'='*80}")

    def run_capture_test(
        self,
        led_index: int,
        r: int,
        g: int,
        b: int,
        camera_device: int = 0,
        output_dir: str = "led_color_tests",
        camera_config: dict = None,
    ):
        """
        Light a single LED in R, G, or B and capture an image after 10 seconds.
        Shows the LED for 30 seconds total so you can visually inspect it.

        Args:
            led_index: Index of LED to light (0-based)
            r, g, b: RGB color values (only one should be 255, others 0)
            camera_device: Camera device ID for capture
            output_dir: Directory to save captured images
            camera_config: Camera configuration dict (from JSON file)
        """
        # Process camera configuration if provided
        crop_region = None
        flip_image = False
        use_usb = False
        camera_resolution = None
        manual_gain = None

        if camera_config:
            # Override camera device from config if not explicitly provided
            if camera_device == 0:  # Default value
                camera_device = camera_config.get("camera_device", 0)

            # Extract crop region from config
            crop_config = camera_config.get("crop_region")
            if crop_config:
                crop_region = (crop_config["x"], crop_config["y"], crop_config["width"], crop_config["height"])
                print(f"Using crop region from config: {crop_region}")

            # Extract camera resolution from config
            resolution_config = camera_config.get("camera_resolution")
            if resolution_config:
                camera_resolution = (resolution_config["width"], resolution_config["height"])
                print(f"Using camera resolution from config: {camera_resolution}")

            # Check if config specifies USB camera
            if camera_config.get("use_usb", False):
                use_usb = True
                print("Using USB camera mode from config")

            # Check if config specifies flip image
            flip_image = camera_config.get("flip_image", False)
            if flip_image:
                print("Using flip image mode from config")

            # Get manual gain from config
            manual_gain = camera_config.get("manual_gain")
            if manual_gain:
                print(f"Using manual gain from config: {manual_gain}")

        # Determine color name
        if r == 255 and g == 0 and b == 0:
            color_name = "RED"
        elif r == 0 and g == 255 and b == 0:
            color_name = "GREEN"
        elif r == 0 and g == 0 and b == 255:
            color_name = "BLUE"
        else:
            color_name = f"RGB_{r}_{g}_{b}"

        print(f"=== LED Color Capture Test ===")
        print(f"LED Index: {led_index}")
        print(f"LED RGB: ({r}, {g}, {b}) - {color_name}")
        print(f"Duration: 30 seconds (capture at 10s)")
        print(f"Output directory: {output_dir}")
        print()

        if led_index < 0 or led_index >= self.config.led_count:
            print(f"Error: LED index {led_index} out of range (0-{self.config.led_count-1})")
            return False

        if not self.client.connect():
            print("Error: Failed to connect to WLED controller")
            return False

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        try:
            self.running = True
            self.start_time = time.time()
            resend_interval = 0.2  # Send packet every 200ms
            capture_done = False

            # Generate the LED data once
            led_data = self.generator.single_led(led_index, r, g, b)

            # Setup camera using the same approach as capture_diffusion_patterns.py
            print("Initializing camera...")

            if use_usb:
                print(f"Using USB camera at /dev/video{camera_device}")

                # Try V4L2 backend first for USB cameras
                cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)

                if not cap.isOpened():
                    print("V4L2 backend failed, trying default backend")
                    cap = cv2.VideoCapture(camera_device)

                if not cap.isOpened():
                    # Try using GStreamer with v4l2src as fallback
                    print("Default backend failed, trying GStreamer with v4l2src")
                    gstreamer_pipeline = (
                        f"v4l2src device=/dev/video{camera_device} ! "
                        "videoconvert ! video/x-raw, format=BGR ! appsink"
                    )
                    print(f"Using GStreamer pipeline: {gstreamer_pipeline}")
                    cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

                if not cap.isOpened():
                    print(f"Error: Failed to open USB camera at /dev/video{camera_device}")
                    return False

                print("USB camera opened successfully")

                # Configure camera with v4l2-ctl for more reliable control
                configure_camera_with_v4l2(camera_device, manual_gain)

                # Set camera to 30 FPS for consistent timing
                try:
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    print("Set USB camera to 30 FPS")
                except:
                    print("Could not set FPS on USB camera")

                # Also try OpenCV settings as backup (less reliable but worth trying)
                print("Applying additional OpenCV camera settings...")

                # Disable auto exposure
                try:
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode
                    print("Disabled auto exposure")
                except:
                    print("Could not disable auto exposure")

                # Set manual exposure if gain is provided (use gain as proxy for exposure)
                if manual_gain is not None:
                    try:
                        # Set exposure time (negative values often work for manual exposure)
                        cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # Often works for USB cameras
                        print("Set manual exposure time")
                    except:
                        print("Could not set manual exposure time")

                # Disable auto white balance
                try:
                    cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # 0 = disable auto white balance
                    print("Disabled auto white balance")
                except:
                    print("Could not disable auto white balance")

                # Set fixed white balance temperature
                try:
                    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)  # Daylight ~4600K
                    print("Set white balance to 4600K")
                except:
                    print("Could not set white balance temperature")

                # Disable backlight compensation
                try:
                    cap.set(cv2.CAP_PROP_BACKLIGHT, 0)
                    print("Disabled backlight compensation")
                except:
                    print("Could not disable backlight compensation")

                # Disable autofocus
                try:
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    print("Disabled autofocus")
                except:
                    print("Could not disable autofocus")

                # Set manual camera properties if needed for USB camera
                if manual_gain is not None:
                    try:
                        cap.set(cv2.CAP_PROP_GAIN, manual_gain)
                        print(f"Set USB camera gain to {manual_gain}")
                    except:
                        print("Could not set gain on USB camera")
                else:
                    # Set a fixed gain value to prevent auto-gain changes
                    try:
                        cap.set(cv2.CAP_PROP_GAIN, 0)  # Minimum gain
                        print("Set USB camera gain to minimum (0)")
                    except:
                        print("Could not set USB camera gain")

                # Set camera resolution if specified from calibration config
                if camera_resolution is not None:
                    width, height = camera_resolution
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        print(f"Set USB camera resolution to {width}x{height}")
                    except:
                        print(f"Could not set USB camera resolution to {width}x{height}")
                else:
                    # Default resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            else:
                print(f"Using CSI camera {camera_device} with GStreamer")

                # Build GStreamer pipeline with optional gain control for CSI camera
                pipeline_parts = [f"nvarguscamerasrc sensor-id={camera_device}"]

                # Add manual gain control if specified
                if manual_gain is not None:
                    pipeline_parts.append(f'gainrange="{manual_gain} {manual_gain}"')
                    pipeline_parts.append("aelock=true")  # Lock auto-exposure for consistent capture
                    pipeline_parts.append("awblock=true")  # Lock auto-white-balance
                    print(f"Setting manual gain: {manual_gain} with locked exposure and white balance")

                # Complete the pipeline
                # Use max-buffers=1 to reduce latency and get most recent frame
                gstreamer_pipeline = (
                    " ".join(pipeline_parts) + " ! "
                    "nvvidconv ! video/x-raw,format=I420 ! "
                    "videoconvert ! video/x-raw,format=BGR ! "
                    "appsink drop=1 max-buffers=1"
                )

                print(f"GStreamer pipeline: {gstreamer_pipeline}")
                cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

                if not cap.isOpened():
                    print("GStreamer pipeline failed to open")
                    return False

                print("GStreamer pipeline opened successfully")

            # Warm up camera and flush old frames
            print("Warming up camera and flushing buffer...")
            for i in range(10):  # Match capture_diffusion_patterns.py
                ret, frame = cap.read()
                if not ret:
                    if i < 3:
                        print(f"Warmup frame {i} failed to read")
                else:
                    if i < 3:
                        print(f"Warmup frame {i} successful, shape: {frame.shape if frame is not None else 'None'}")
                time.sleep(0.05)

            print(f"Lighting LED {led_index} with {color_name}...")
            print("LED will be shown for 25 seconds. Focus on 0-5s transition period.")
            print()

            last_send = 0
            capture_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10, 20]  # Focus on 0-5s transition
            captured_times = set()  # Track which times we've captured
            time_series_data = []  # Store RGB data over time

            while self.running:
                current_time = time.time()
                elapsed = current_time - self.start_time

                # Check if we've reached 25 seconds
                if elapsed >= 25.0:
                    print("\n25 seconds elapsed. Turning off LED.")
                    break

                # Capture images at multiple time points to track adaptation
                should_capture = False
                capture_label = ""
                for capture_time in capture_times:
                    if elapsed >= capture_time and capture_time not in captured_times:
                        should_capture = True
                        if capture_time == int(capture_time):
                            capture_label = f"{int(capture_time)}s"
                        else:
                            capture_label = f"{capture_time}s"
                        captured_times.add(capture_time)
                        break

                if should_capture:
                    print(f"\n{capture_label} elapsed. Capturing image...")

                    # Flush camera buffer aggressively like in capture_diffusion_patterns.py
                    print("Flushing camera buffer...")
                    for flush_attempt in range(3):
                        flush_frame = cap.read()
                        if flush_frame[0]:  # ret value
                            print(f"Flush {flush_attempt+1}/3 successful")
                        else:
                            print(f"Flush {flush_attempt+1}/3 failed")
                        time.sleep(0.05)

                    # Small delay to let camera stabilize
                    time.sleep(0.1)

                    # Now capture the frame we want to analyze
                    print(f"Capturing image at {capture_label}...")
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB for analysis
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Apply image flip if requested from config
                        if flip_image:
                            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)

                        # Apply crop region if specified from config
                        if crop_region:
                            x, y, w, h = crop_region
                            frame_rgb = frame_rgb[y : y + h, x : x + w]

                        # Scale to target resolution (800x480) to match capture tool
                        frame_rgb = cv2.resize(frame_rgb, (800, 480), interpolation=cv2.INTER_LINEAR)

                        # Calculate mean RGB values of the entire frame
                        mean_r = np.mean(frame_rgb[:, :, 0])
                        mean_g = np.mean(frame_rgb[:, :, 1])
                        mean_b = np.mean(frame_rgb[:, :, 2])

                        # Find brightest region (likely the LED) using processed RGB frame
                        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

                        print(f"Brightest point: ({max_loc[0]}, {max_loc[1]}) with value {max_val}")

                        # Extract a region around the brightest point
                        roi_size = 32
                        x, y = max_loc[0], max_loc[1]  # max_loc is (x, y)
                        y1 = max(0, y - roi_size // 2)
                        y2 = min(frame_rgb.shape[0], y + roi_size // 2)
                        x1 = max(0, x - roi_size // 2)
                        x2 = min(frame_rgb.shape[1], x + roi_size // 2)

                        print(f"ROI region: ({x1}, {y1}) to ({x2}, {y2}), size: {x2-x1}x{y2-y1}")

                        # Ensure ROI is valid
                        if y2 > y1 and x2 > x1:
                            roi = frame_rgb[y1:y2, x1:x2]
                            roi_r = np.mean(roi[:, :, 0])
                            roi_g = np.mean(roi[:, :, 1])
                            roi_b = np.mean(roi[:, :, 2])
                        else:
                            print("Warning: Invalid ROI region, using center region instead")
                            # Fallback to center region
                            center_y, center_x = frame_rgb.shape[0] // 2, frame_rgb.shape[1] // 2
                            y1 = max(0, center_y - roi_size // 2)
                            y2 = min(frame_rgb.shape[0], center_y + roi_size // 2)
                            x1 = max(0, center_x - roi_size // 2)
                            x2 = min(frame_rgb.shape[1], center_x + roi_size // 2)
                            roi = frame_rgb[y1:y2, x1:x2]
                            roi_r = np.mean(roi[:, :, 0])
                            roi_g = np.mean(roi[:, :, 1])
                            roi_b = np.mean(roi[:, :, 2])

                        # Convert RGB frame back to BGR for OpenCV annotation and saving
                        annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                        # Annotate the image
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(
                            annotated_frame, f"LED {led_index}: {color_name}", (10, 30), font, 0.8, (255, 255, 255), 2
                        )
                        cv2.putText(
                            annotated_frame, f"LED RGB: ({r}, {g}, {b})", (10, 60), font, 0.7, (255, 255, 255), 2
                        )
                        cv2.putText(
                            annotated_frame,
                            f"Frame Mean RGB: ({mean_r:.1f}, {mean_g:.1f}, {mean_b:.1f})",
                            (10, 90),
                            font,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.putText(
                            annotated_frame,
                            f"LED Region RGB: ({roi_r:.1f}, {roi_g:.1f}, {roi_b:.1f})",
                            (10, 120),
                            font,
                            0.7,
                            (255, 255, 255),
                            2,
                        )

                        # Draw rectangle around brightest region
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, "LED", (x1, y1 - 5), font, 0.5, (0, 255, 0), 1)

                        # Save the image with time stamp
                        filename = f"LED_{led_index:04d}_{color_name}_R{r}_G{g}_B{b}_{capture_label}.jpg"
                        filepath = Path(output_dir) / filename
                        cv2.imwrite(str(filepath), annotated_frame)

                        # Store time series data
                        time_point_data = {
                            "time": capture_label,
                            "elapsed_seconds": elapsed,
                            "frame_mean_rgb": (mean_r, mean_g, mean_b),
                            "led_region_rgb": (roi_r, roi_g, roi_b),
                            "brightest_point": (x, y),
                            "max_brightness": max_val,
                            "filename": filename,
                        }
                        time_series_data.append(time_point_data)

                        print(f"Image saved: {filepath}")
                        print(f"Time: {capture_label} | LED RGB (sent): ({r}, {g}, {b})")
                        print(f"Frame mean RGB: ({mean_r:.1f}, {mean_g:.1f}, {mean_b:.1f})")
                        print(f"LED region RGB: ({roi_r:.1f}, {roi_g:.1f}, {roi_b:.1f})")
                        print(f"Brightest point: ({x}, {y}) with value {max_val:.1f}")
                        print()
                    else:
                        print(f"Warning: Failed to capture image at {capture_label}")
                        # Still mark this time as captured to avoid retrying
                        time_val = float(capture_label.rstrip("s"))
                        captured_times.add(time_val)

                # Send packet at regular intervals to keep LED lit
                if current_time - last_send >= resend_interval:
                    success = self.client.send_led_data(led_data)
                    if not success:
                        print(f"\nWarning: Failed to send LED data")
                    last_send = current_time

                # Print countdown
                remaining = 25.0 - elapsed
                if int(elapsed) % 5 == 0 and int(elapsed * 10) % 50 == 0:  # Every 5 seconds
                    print(f"Time remaining: {remaining:.0f} seconds...")

                # Small sleep to avoid busy waiting
                time.sleep(0.01)

            # Turn off the LED when done
            print("\nTurning off LED...")
            self.client.set_solid_color(0, 0, 0)

            # Cleanup camera
            cap.release()

            # Generate time series analysis report
            if time_series_data:
                print(f"\n" + "=" * 80)
                print(f"TIME SERIES ANALYSIS - LED {led_index} {color_name}")
                print(f"=" * 80)

                print(f"{'Time':<8} {'Frame Mean RGB':<20} {'LED Region RGB':<20} {'Brightest':<12} {'Position'}")
                print(f"{'-'*8} {'-'*20} {'-'*20} {'-'*12} {'-'*15}")

                for data in time_series_data:
                    frame_rgb_str = f"({data['frame_mean_rgb'][0]:.1f},{data['frame_mean_rgb'][1]:.1f},{data['frame_mean_rgb'][2]:.1f})"
                    led_rgb_str = f"({data['led_region_rgb'][0]:.1f},{data['led_region_rgb'][1]:.1f},{data['led_region_rgb'][2]:.1f})"
                    brightness_str = f"{data['max_brightness']:.0f}"
                    position_str = f"({data['brightest_point'][0]},{data['brightest_point'][1]})"

                    print(
                        f"{data['time']:<8} {frame_rgb_str:<20} {led_rgb_str:<20} {brightness_str:<12} {position_str}"
                    )

                # Analyze changes over time
                print(f"\n" + "=" * 80)
                print("STABILITY ANALYSIS")
                print(f"=" * 80)

                if len(time_series_data) >= 2:
                    first = time_series_data[0]
                    last = time_series_data[-1]

                    # Calculate changes in LED region RGB
                    first_led_rgb = first["led_region_rgb"]
                    last_led_rgb = last["led_region_rgb"]

                    rgb_change = [abs(last_led_rgb[i] - first_led_rgb[i]) for i in range(3)]
                    rgb_change_pct = [100 * rgb_change[i] / max(first_led_rgb[i], 1) for i in range(3)]

                    print(f"LED Region RGB Change from {first['time']} to {last['time']}:")
                    print(
                        f"  Red:   {first_led_rgb[0]:.1f} → {last_led_rgb[0]:.1f} (Δ{rgb_change[0]:+.1f}, {rgb_change_pct[0]:+.1f}%)"
                    )
                    print(
                        f"  Green: {first_led_rgb[1]:.1f} → {last_led_rgb[1]:.1f} (Δ{rgb_change[1]:+.1f}, {rgb_change_pct[1]:+.1f}%)"
                    )
                    print(
                        f"  Blue:  {first_led_rgb[2]:.1f} → {last_led_rgb[2]:.1f} (Δ{rgb_change[2]:+.1f}, {rgb_change_pct[2]:+.1f}%)"
                    )

                    # Brightness stability
                    brightness_change = abs(last["max_brightness"] - first["max_brightness"])
                    brightness_change_pct = 100 * brightness_change / max(first["max_brightness"], 1)
                    print(
                        f"Brightness: {first['max_brightness']:.0f} → {last['max_brightness']:.0f} (Δ{brightness_change:+.0f}, {brightness_change_pct:+.1f}%)"
                    )

                    # Position stability
                    pos_change = (
                        (last["brightest_point"][0] - first["brightest_point"][0]) ** 2
                        + (last["brightest_point"][1] - first["brightest_point"][1]) ** 2
                    ) ** 0.5
                    print(f"Position drift: {pos_change:.1f} pixels")

                    # Determine stability
                    max_rgb_change = max(rgb_change_pct)
                    if max_rgb_change < 5:
                        stability = "STABLE"
                    elif max_rgb_change < 15:
                        stability = "MODERATE DRIFT"
                    else:
                        stability = "SIGNIFICANT DRIFT"

                    print(f"\nOverall LED Stability: {stability}")
                    print(f"Maximum RGB change: {max_rgb_change:.1f}%")

                print(f"\n" + "=" * 80)

            print(f"\nTest completed!")

        except Exception as e:
            print(f"\nError during capture test: {e}")
            return False
        finally:
            self.client.disconnect()

        return True

    def run_animated_pattern(
        self,
        pattern_name: str,
        speed: float = 1.0,
        duration: Optional[float] = None,
        **kwargs,
    ):
        """
        Run animated LED pattern.

        Args:
            pattern_name: Name of pattern function
            speed: Animation speed
            duration: Run duration in seconds (None for infinite)
            **kwargs: Additional pattern parameters
        """
        pattern_func = getattr(self.generator, pattern_name, None)
        if not pattern_func:
            print(f"Error: Unknown pattern '{pattern_name}'")
            return False

        print(f"Starting {pattern_name} pattern (speed: {speed:.1f})")
        if duration:
            print(f"Running for {duration} seconds...")
        else:
            print("Running until interrupted (Ctrl+C to stop)...")

        if not self.client.connect():
            print("Error: Failed to connect to WLED controller")
            return False

        try:
            self.running = True
            self.start_time = time.time()
            last_time = self.start_time
            frame_count = 0

            while self.running:
                current_time = time.time()
                delta_time = current_time - last_time
                last_time = current_time

                # Check duration
                if duration and (current_time - self.start_time) >= duration:
                    break

                # Update pattern time
                self.generator.update_time(delta_time)

                # Generate pattern data
                led_data = pattern_func(speed=speed, **kwargs)

                # Send to WLED
                success = self.client.send_led_data(led_data)
                if not success:
                    print(f"\nWarning: Failed to send frame {frame_count}")

                frame_count += 1

                # Print status every few seconds
                if frame_count % (30 * 5) == 0:  # Every ~5 seconds at 30 FPS
                    stats = self.client.get_statistics()
                    elapsed = current_time - self.start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Status: {frame_count} frames, {fps:.1f} FPS, {stats['transmission_errors']} errors")

                # Target ~30 FPS
                time.sleep(1.0 / 30.0)

            # Final statistics
            elapsed = time.time() - self.start_time
            stats = self.client.get_statistics()
            avg_fps = frame_count / elapsed if elapsed > 0 else 0

            print("\nPattern completed:")
            print(f"  Duration: {elapsed:.1f} seconds")
            print(f"  Frames sent: {frame_count}")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Transmission errors: {stats['transmission_errors']}")

        except Exception as e:
            print(f"\nError during pattern execution: {e}")
            return False
        finally:
            self.client.disconnect()

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WLED Test Patterns - Send test patterns to WLED controllers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s solid --color 255 0 0 --duration 5
  %(prog)s single --index 0 --color 255 0 0 --duration 10
  %(prog)s rainbow-cycle --speed 2.0
  %(prog)s animated-rainbow --speed 1.0 --width 0.3 --duration 30
  %(prog)s wave --speed 0.5 --frequency 3.0
        """,
    )

    # Global options
    parser.add_argument(
        "--host",
        default=WLED_DEFAULT_HOST,
        help=f"WLED controller hostname/IP (default: {WLED_DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=WLED_DEFAULT_PORT,
        help=f"WLED controller port (default: {WLED_DEFAULT_PORT})",
    )
    parser.add_argument(
        "--led-count",
        type=int,
        help="Number of LEDs",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Pattern duration in seconds (default: run until interrupted)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--persistent-retry",
        action="store_true",
        help="Keep retrying connection until successful (useful for startup)",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=10.0,
        help="Seconds between connection retries (default: 10.0)",
    )

    # Subcommands for different patterns
    subparsers = parser.add_subparsers(dest="pattern", help="Pattern type")

    # Solid color pattern
    solid_parser = subparsers.add_parser("solid", help="Solid color pattern")
    solid_parser.add_argument(
        "--color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        default=[255, 255, 255],
        help="RGB color values 0-255 (default: white)",
    )

    # Rainbow cycle pattern
    rainbow_parser = subparsers.add_parser("rainbow-cycle", help="Rainbow cycle pattern")
    rainbow_parser.add_argument("--speed", type=float, default=1.0, help="Cycle speed in Hz (default: 1.0)")

    # Animated rainbow pattern
    animated_parser = subparsers.add_parser("animated-rainbow", help="Animated rainbow pattern")
    animated_parser.add_argument("--speed", type=float, default=1.0, help="Animation speed in Hz (default: 1.0)")
    animated_parser.add_argument(
        "--width",
        type=float,
        default=1.0,
        help="Rainbow width as fraction of array (default: 1.0)",
    )

    # Wave pattern
    wave_parser = subparsers.add_parser("wave", help="Sine wave pattern")
    wave_parser.add_argument("--speed", type=float, default=1.0, help="Wave speed in Hz (default: 1.0)")
    wave_parser.add_argument(
        "--frequency",
        type=float,
        default=2.0,
        help="Number of waves across array (default: 2.0)",
    )

    # Single LED pattern
    single_parser = subparsers.add_parser("single", help="Light a single LED")
    single_parser.add_argument("--index", type=int, required=True, help="LED index to light (0-based)")
    single_parser.add_argument(
        "--color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        default=[255, 255, 255],
        help="RGB color values 0-255 (default: white)",
    )

    # Test connection
    test_parser = subparsers.add_parser("test", help="Test WLED connection")

    # Capture test for color analysis
    capture_parser = subparsers.add_parser("capture-test", help="Light LED and capture image for color analysis")
    capture_parser.add_argument("--index", type=int, required=True, help="LED index to light (0-based)")
    capture_parser.add_argument(
        "--color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        required=True,
        help="RGB color values 0-255 (use 255 0 0 for red, 0 255 0 for green, or 0 0 255 for blue)",
    )
    capture_parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    capture_parser.add_argument(
        "--output-dir",
        type=str,
        default="led_color_tests",
        help="Directory to save captured images (default: led_color_tests)",
    )
    capture_parser.add_argument(
        "--camera-config",
        help="Camera calibration JSON file from camera_calibration.py (same format as capture_diffusion_patterns.py)",
    )

    # Gain calibration test for finding optimal camera gain
    gain_parser = subparsers.add_parser(
        "gain-calibration", help="Test multiple gain values on same LED for calibration"
    )
    gain_parser.add_argument("--index", type=int, required=True, help="LED index to light (0-based)")
    gain_parser.add_argument(
        "--color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        required=True,
        help="RGB color values 0-255 (use 255 0 0 for red, 0 255 0 for green, or 0 0 255 for blue)",
    )
    gain_parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    gain_parser.add_argument(
        "--output-dir",
        type=str,
        default="gain_calibration",
        help="Directory to save captured images and report (default: gain_calibration)",
    )
    gain_parser.add_argument("--camera-config", help="Camera calibration JSON file from camera_calibration.py")
    gain_parser.add_argument(
        "--gain-values",
        nargs="+",
        type=int,
        default=[10, 25, 50, 75, 100, 150, 200],
        help="Gain values to test (default: 10 25 50 75 100 150 200)",
    )

    # Brightness/contrast test for understanding diffusion pattern visibility
    bc_parser = subparsers.add_parser(
        "brightness-contrast", help="Test brightness and contrast effect on LED visibility"
    )
    bc_parser.add_argument("--index", type=int, required=True, help="LED index to light (0-based)")
    bc_parser.add_argument(
        "--color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        required=True,
        help="RGB color values 0-255 (use 0 100 0 for moderate green to avoid saturation)",
    )
    bc_parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    bc_parser.add_argument(
        "--output-dir",
        type=str,
        default="brightness_contrast",
        help="Directory to save captured images and report (default: brightness_contrast)",
    )
    bc_parser.add_argument("--camera-config", help="Camera calibration JSON file from camera_calibration.py")
    bc_parser.add_argument(
        "--brightness-values",
        nargs="+",
        type=int,
        default=[64, 128, 192],
        help="Brightness values to test (default: 64 128 192)",
    )
    bc_parser.add_argument(
        "--contrast-values",
        nargs="+",
        type=int,
        default=[64, 128, 192],
        help="Contrast values to test (default: 64 128 192)",
    )

    # Exposure test for finding optimal exposure settings
    exp_parser = subparsers.add_parser(
        "exposure-test", help="Test exposure_time_absolute effect on LED diffusion visibility"
    )
    exp_parser.add_argument("--index", type=int, required=True, help="LED index to light (0-based)")
    exp_parser.add_argument(
        "--color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        required=True,
        help="RGB color values 0-255 (use 0 100 0 for moderate green to avoid saturation)",
    )
    exp_parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    exp_parser.add_argument(
        "--output-dir",
        type=str,
        default="exposure_test",
        help="Directory to save captured images and report (default: exposure_test)",
    )
    exp_parser.add_argument("--camera-config", help="Camera calibration JSON file from camera_calibration.py")
    exp_parser.add_argument(
        "--exposure-values",
        nargs="+",
        type=int,
        default=[50, 100, 200, 400, 800],
        help="Exposure values to test (default: 50 100 200 400 800)",
    )
    exp_parser.add_argument(
        "--gain-values",
        nargs="+",
        type=int,
        default=[0, 10, 20, 30, 40, 50],
        help="Gain values to test (default: 0 10 20 30 40 50)",
    )

    args = parser.parse_args()

    if not args.pattern:
        parser.print_help()
        return 1

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create WLED configuration
    config = WLEDConfig(
        host=args.host,
        port=args.port,
        led_count=args.led_count,
        timeout=5.0,
        retry_count=3,
        max_fps=30.0,
        persistent_retry=args.persistent_retry,
        retry_interval=args.retry_interval,
    )

    if args.verbose:
        print("WLED Configuration:")
        print(f"  Host: {config.host}")
        print(f"  Port: {config.port}")
        print(f"  LED Count: {config.led_count}")
        if config.persistent_retry:
            print(f"  Persistent Retry: enabled (interval: {config.retry_interval}s)")
        print()

    # Create test runner
    runner = WLEDTestRunner(config)

    try:
        if args.pattern == "test":
            # Test connection only
            print(f"Testing connection to WLED controller at {config.host}:{config.port}...")
            if runner.client.connect():
                print("✓ Connection successful")
                stats = runner.client.get_statistics()
                print(f"✓ Controller ready for {stats['led_count']} LEDs")

                # Show WLED status if available
                wled_status = runner.client.get_wled_status()
                if wled_status:
                    print(f"✓ WLED '{wled_status.get('name', 'Unknown')}' v{wled_status.get('ver', 'Unknown')}")
                    if "leds" in wled_status and isinstance(wled_status["leds"], dict):
                        led_info = wled_status["leds"]
                        print(f"✓ Hardware: {led_info.get('count', 0)} LEDs, {led_info.get('fps', 0)} FPS")

                runner.client.disconnect()
                return 0
            else:
                print("✗ Connection failed")
                return 1

        elif args.pattern == "solid":
            r, g, b = args.color
            if not all(0 <= c <= 255 for c in [r, g, b]):
                print("Error: RGB values must be between 0 and 255")
                return 1
            success = runner.run_solid_color(r, g, b, args.duration)

        elif args.pattern == "rainbow-cycle":
            success = runner.run_animated_pattern("rainbow_cycle", speed=args.speed, duration=args.duration)

        elif args.pattern == "animated-rainbow":
            success = runner.run_animated_pattern(
                "animated_rainbow",
                speed=args.speed,
                duration=args.duration,
                width=args.width,
            )

        elif args.pattern == "wave":
            success = runner.run_animated_pattern(
                "wave_pattern",
                speed=args.speed,
                duration=args.duration,
                frequency=args.frequency,
            )

        elif args.pattern == "single":
            r, g, b = args.color
            if not all(0 <= c <= 255 for c in [r, g, b]):
                print("Error: RGB values must be between 0 and 255")
                return 1
            success = runner.run_single_led(args.index, r, g, b, args.duration)

        elif args.pattern == "capture-test":
            r, g, b = args.color
            if not all(0 <= c <= 255 for c in [r, g, b]):
                print("Error: RGB values must be between 0 and 255")
                return 1

            # Load camera configuration if provided
            camera_config = None
            if args.camera_config:
                try:
                    with open(args.camera_config) as f:
                        camera_config = json.load(f)
                    print(f"Loaded camera configuration from {args.camera_config}")
                except Exception as e:
                    print(f"Error: Failed to load camera config {args.camera_config}: {e}")
                    return 1

            success = runner.run_capture_test(args.index, r, g, b, args.camera, args.output_dir, camera_config)

        elif args.pattern == "gain-calibration":
            r, g, b = args.color
            if not all(0 <= c <= 255 for c in [r, g, b]):
                print("Error: RGB values must be between 0 and 255")
                return 1

            # Validate gain values
            if not all(0 <= g <= 255 for g in args.gain_values):
                print("Error: Gain values must be between 0 and 255")
                return 1

            # Load camera configuration if provided
            camera_config = None
            if args.camera_config:
                try:
                    with open(args.camera_config) as f:
                        camera_config = json.load(f)
                    print(f"Loaded camera configuration from {args.camera_config}")
                except Exception as e:
                    print(f"Error: Failed to load camera config {args.camera_config}: {e}")
                    return 1

            success = runner.run_gain_calibration(
                args.index, r, g, b, args.camera, args.output_dir, camera_config, args.gain_values
            )

        elif args.pattern == "brightness-contrast":
            r, g, b = args.color
            if not all(0 <= c <= 255 for c in [r, g, b]):
                print("Error: RGB values must be between 0 and 255")
                return 1

            # Validate brightness and contrast values
            if not all(0 <= v <= 255 for v in args.brightness_values):
                print("Error: Brightness values must be between 0 and 255")
                return 1
            if not all(0 <= v <= 255 for v in args.contrast_values):
                print("Error: Contrast values must be between 0 and 255")
                return 1

            # Load camera configuration if provided
            camera_config = None
            if args.camera_config:
                try:
                    with open(args.camera_config) as f:
                        camera_config = json.load(f)
                    print(f"Loaded camera configuration from {args.camera_config}")
                except Exception as e:
                    print(f"Error: Failed to load camera config {args.camera_config}: {e}")
                    return 1

            success = runner.run_brightness_contrast_test(
                args.index,
                r,
                g,
                b,
                args.camera,
                args.output_dir,
                camera_config,
                args.brightness_values,
                args.contrast_values,
            )

        elif args.pattern == "exposure-test":
            r, g, b = args.color
            if not all(0 <= c <= 255 for c in [r, g, b]):
                print("Error: RGB values must be between 0 and 255")
                return 1

            # Validate exposure values (3-2047 based on camera guide)
            if not all(3 <= v <= 2047 for v in args.exposure_values):
                print("Error: Exposure values must be between 3 and 2047")
                return 1

            # Load camera configuration if provided
            camera_config = None
            if args.camera_config:
                try:
                    with open(args.camera_config) as f:
                        camera_config = json.load(f)
                    print(f"Loaded camera configuration from {args.camera_config}")
                except Exception as e:
                    print(f"Error: Failed to load camera config {args.camera_config}: {e}")
                    return 1

            success = runner.run_exposure_test(
                args.index, r, g, b, args.camera, args.output_dir, camera_config, args.exposure_values, args.gain_values
            )

        else:
            print(f"Error: Unknown pattern '{args.pattern}'")
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
