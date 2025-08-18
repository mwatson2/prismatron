#!/usr/bin/env python3
"""Test ultra-low exposure times with minimal gain to avoid saturation."""

import os
import subprocess
import sys
import time

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import the WLED client directly
from src.consumer.wled_client import WLEDClient, WLEDConfig


def configure_camera_minimal(camera_device: int, gain: int = 0, exposure: int = 3):
    """Configure camera with minimal sensitivity settings."""
    device_path = f"/dev/video{camera_device}"
    print(f"Configuring {device_path} with gain={gain}, exposure={exposure}...")

    commands = [
        ["v4l2-ctl", "-d", device_path, "-c", "white_balance_automatic=0"],
        ["v4l2-ctl", "-d", device_path, "-c", "auto_exposure=1"],  # Manual mode
        ["v4l2-ctl", "-d", device_path, "-c", "focus_automatic_continuous=0"],
        ["v4l2-ctl", "-d", device_path, "-c", "white_balance_temperature=4600"],
        ["v4l2-ctl", "-d", device_path, "-c", "backlight_compensation=0"],
        ["v4l2-ctl", "-d", device_path, "-c", f"exposure_time_absolute={exposure}"],
        ["v4l2-ctl", "-d", device_path, "-c", f"gain={gain}"],
    ]

    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Set {cmd[4]}")
        else:
            print(f"✗ Failed to set {cmd[4]}: {result.stderr.strip()}")


def test_led_with_settings(led_index=1, r=0, g=255, b=0):
    """Test LED capture with various exposure settings."""

    # Setup WLED
    config = WLEDConfig(host="192.168.1.188", port=21324, led_count=3000)
    client = WLEDClient(config)

    print(f"Testing LED {led_index} with RGB({r},{g},{b})")

    # Light the LED
    led_data = bytearray(3000 * 3)  # All LEDs off
    led_data[led_index * 3 : (led_index + 1) * 3] = [r, g, b]

    result = client.send_led_data(bytes(led_data))
    success = result.success
    if not success:
        print("Failed to send LED data")
        return

    print("Waiting for LED to light...")
    time.sleep(2)  # Wait for WLED command to take effect

    # Test different exposure times with gain=0
    exposures = [3, 10, 25, 50, 100]  # Ultra-low to normal

    for exposure in exposures:
        print(f"\n=== Testing exposure {exposure} (gain=0) ===")

        # Configure camera
        configure_camera_minimal(0, gain=0, exposure=exposure)

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open camera")
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Warm up camera
        print("Warming up...")
        for _ in range(5):
            cap.read()
        time.sleep(0.5)

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            cap.release()
            continue

        # Crop to region of interest (from camera config)
        crop_region = (598, 200, 946, 568)
        x, y, x2, y2 = crop_region
        cropped = frame[y:y2, x:x2]

        # Find LED region (brightest area)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(largest)
            led_region = cropped[y_roi : y_roi + h_roi, x_roi : x_roi + w_roi]

            # Calculate stats
            mean_bgr = np.mean(led_region, axis=(0, 1))
            peak_bgr = np.max(led_region, axis=(0, 1))
            mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])  # BGR to RGB
            peak_rgb = (peak_bgr[2], peak_bgr[1], peak_bgr[0])
            peak_brightness = max(peak_rgb)

            status = "SATURATED" if peak_brightness >= 254 else "OK"
            print(f"Mean RGB: ({mean_rgb[0]:.1f},{mean_rgb[1]:.1f},{mean_rgb[2]:.1f})")
            print(f"Peak RGB: ({peak_rgb[0]:.0f},{peak_rgb[1]:.0f},{peak_rgb[2]:.0f})")
            print(f"Peak brightness: {peak_brightness:.0f} - {status}")

            # Save image
            filename = f"exposure_test_exp{exposure}_gain0.jpg"
            cv2.imwrite(filename, cropped)
            print(f"Saved: {filename}")

        else:
            print("No LED region detected")

        cap.release()

    # Turn off LED
    print("\nTurning off LED...")
    led_data = bytearray(3000 * 3)  # All LEDs off
    client.send_led_data(bytes(led_data))


if __name__ == "__main__":
    test_led_with_settings()
