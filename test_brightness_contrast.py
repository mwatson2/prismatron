#!/usr/bin/env python3
"""Test brightness and contrast settings effect on LED capture."""

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Add src to path
sys.path.append("/mnt/dev/prismatron")
from src.consumer.wled_client import WLEDClient, WLEDConfig


def configure_camera_v4l2(device: int, gain: int, exposure: int, brightness: int, contrast: int):
    """Configure camera using v4l2-ctl with all manual settings."""
    device_path = f"/dev/video{device}"

    print("\nConfiguring camera with:")
    print(f"  Gain: {gain}")
    print(f"  Exposure: {exposure}")
    print(f"  Brightness: {brightness}")
    print(f"  Contrast: {contrast}")

    # Commands to disable all auto features and set manual values
    commands = [
        # Disable all automatic features
        ["v4l2-ctl", "-d", device_path, "-c", "white_balance_automatic=0"],
        ["v4l2-ctl", "-d", device_path, "-c", "auto_exposure=1"],  # Manual mode
        ["v4l2-ctl", "-d", device_path, "-c", "focus_automatic_continuous=0"],
        ["v4l2-ctl", "-d", device_path, "-c", "exposure_dynamic_framerate=0"],  # Disable dynamic framerate
        ["v4l2-ctl", "-d", device_path, "-c", "backlight_compensation=0"],
        # Set manual values
        ["v4l2-ctl", "-d", device_path, "-c", f"exposure_time_absolute={exposure}"],
        ["v4l2-ctl", "-d", device_path, "-c", f"gain={gain}"],
        ["v4l2-ctl", "-d", device_path, "-c", f"brightness={brightness}"],
        ["v4l2-ctl", "-d", device_path, "-c", f"contrast={contrast}"],
        ["v4l2-ctl", "-d", device_path, "-c", "white_balance_temperature=4600"],
        ["v4l2-ctl", "-d", device_path, "-c", "saturation=128"],  # Keep saturation at default
    ]

    success_count = 0
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            success_count += 1
        else:
            print(f"  Warning: Failed {cmd[4]}: {result.stderr.strip()}")

    print(f"  Applied {success_count}/{len(commands)} settings")
    return success_count == len(commands)


def capture_led_frame(camera_device: int = 0) -> Tuple[np.ndarray, Dict]:
    """Capture a single frame and analyze LED region."""
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened():
        return None, {"error": "Failed to open camera"}

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Warm up camera (important for stability)
    for _ in range(10):
        ret, _ = cap.read()
        if not ret:
            cap.release()
            return None, {"error": "Failed to warm up camera"}

    # Small delay for stability
    time.sleep(0.5)

    # Capture actual frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, {"error": "Failed to capture frame"}

    # Crop to region of interest (from camera-0810.json)
    crop_region = (598, 200, 946, 568)
    x, y, x2, y2 = crop_region
    cropped = frame[y:y2, x:x2]

    # Find bright regions (potential LED areas)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to find bright spots
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stats = {
        "frame_mean": np.mean(cropped, axis=(0, 1)),
        "frame_max": np.max(cropped, axis=(0, 1)),
        "frame_std": np.std(cropped, axis=(0, 1)),
    }

    if contours:
        # Find largest contour (likely the LED)
        largest = max(contours, key=cv2.contourArea)
        x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(largest)

        # Extract LED region
        led_region = cropped[y_roi : y_roi + h_roi, x_roi : x_roi + w_roi]

        # Calculate LED region statistics
        led_mean_bgr = np.mean(led_region, axis=(0, 1))
        led_max_bgr = np.max(led_region, axis=(0, 1))
        led_std_bgr = np.std(led_region, axis=(0, 1))

        stats.update(
            {
                "led_found": True,
                "led_area": w_roi * h_roi,
                "led_mean_rgb": (led_mean_bgr[2], led_mean_bgr[1], led_mean_bgr[0]),
                "led_max_rgb": (led_max_bgr[2], led_max_bgr[1], led_max_bgr[0]),
                "led_std_rgb": (led_std_bgr[2], led_std_bgr[1], led_std_bgr[0]),
                "led_bbox": (x_roi, y_roi, w_roi, h_roi),
                "saturated": max(led_max_bgr) >= 254,
            }
        )

        # Draw bbox on image for visualization
        cv2.rectangle(cropped, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 0), 2)
    else:
        stats.update({"led_found": False, "led_area": 0, "saturated": False})

    return cropped, stats


def test_brightness_contrast_grid():
    """Test a grid of brightness and contrast values."""

    # Setup WLED
    config = WLEDConfig(host="192.168.1.188", port=21324, led_count=3000)
    client = WLEDClient(config)

    # Light single green LED
    led_index = 1
    led_brightness = 100  # Use medium brightness to avoid saturation
    led_data = bytearray(3000 * 3)
    led_data[led_index * 3 : (led_index + 1) * 3] = [0, led_brightness, 0]  # Green

    print("Lighting LED...")
    result = client.send_led_data(bytes(led_data))
    if not result.success:
        print("Failed to send LED data!")
        return

    print("Waiting for LED to stabilize...")
    time.sleep(3)

    # Test parameters
    gain = 20  # Fixed moderate gain
    exposure = 100  # Fixed moderate exposure

    # Grid of brightness and contrast values to test
    brightness_values = [64, 128, 192]  # Low, medium, high
    contrast_values = [64, 128, 192]  # Low, medium, high

    results = []

    print("\n" + "=" * 60)
    print("BRIGHTNESS/CONTRAST GRID TEST")
    print(f"LED: Index {led_index}, RGB(0,{led_brightness},0)")
    print(f"Fixed: Gain={gain}, Exposure={exposure}")
    print("=" * 60)

    for brightness in brightness_values:
        for contrast in contrast_values:
            print(f"\n--- Testing Brightness={brightness}, Contrast={contrast} ---")

            # Configure camera
            if not configure_camera_v4l2(0, gain, exposure, brightness, contrast):
                print("Failed to configure camera, skipping...")
                continue

            # Wait for settings to take effect
            time.sleep(1)

            # Capture frame
            frame, stats = capture_led_frame(0)

            if frame is None:
                print(f"Failed to capture: {stats.get('error', 'Unknown error')}")
                continue

            # Save image
            filename = f"brightness_contrast/B{brightness}_C{contrast}.jpg"
            os.makedirs("brightness_contrast", exist_ok=True)
            cv2.imwrite(filename, frame)

            # Store results
            result = {"brightness": brightness, "contrast": contrast, "filename": filename, **stats}
            results.append(result)

            # Print summary
            if stats.get("led_found"):
                mean_rgb = stats["led_mean_rgb"]
                max_rgb = stats["led_max_rgb"]
                area = stats["led_area"]
                saturated = "SATURATED" if stats["saturated"] else "OK"

                print(f"  LED Mean RGB: ({mean_rgb[0]:.1f}, {mean_rgb[1]:.1f}, {mean_rgb[2]:.1f})")
                print(f"  LED Max RGB: ({max_rgb[0]:.0f}, {max_rgb[1]:.0f}, {max_rgb[2]:.0f}) - {saturated}")
                print(f"  LED Area: {area} pixels")
                print(f"  Saved: {filename}")
            else:
                print("  LED not detected!")

    # Turn off LED
    print("\nTurning off LED...")
    led_data = bytearray(3000 * 3)
    client.send_led_data(bytes(led_data))

    # Generate report
    print("\n" + "=" * 60)
    print("BRIGHTNESS/CONTRAST ANALYSIS REPORT")
    print("=" * 60)

    print("\nEffect on LED Detection Area:")
    print("-" * 40)
    print("Brightness  Contrast   LED Area (pixels)")
    print("-" * 40)
    for r in results:
        if r.get("led_found"):
            print(f"{r['brightness']:^10} {r['contrast']:^10} {r['led_area']:^15}")

    print("\nEffect on LED Brightness:")
    print("-" * 50)
    print("Brightness  Contrast   Mean Green   Max Green   Status")
    print("-" * 50)
    for r in results:
        if r.get("led_found"):
            mean_g = r["led_mean_rgb"][1]
            max_g = r["led_max_rgb"][1]
            status = "SAT" if r["saturated"] else "OK"
            print(f"{r['brightness']:^10} {r['contrast']:^10} {mean_g:^10.1f} {max_g:^10.0f} {status:^8}")

    # Save results to JSON
    with open("brightness_contrast/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to brightness_contrast/results.json")

    # Analysis summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("-" * 60)

    # Find configuration with largest LED area (best diffusion visibility)
    if results:
        best_area = max(results, key=lambda x: x.get("led_area", 0))
        print(
            f"Largest LED area: Brightness={best_area['brightness']}, "
            f"Contrast={best_area['contrast']} ({best_area['led_area']} pixels)"
        )

    # Find configurations without saturation
    no_sat = [r for r in results if r.get("led_found") and not r.get("saturated")]
    if no_sat:
        print(f"Configurations without saturation: {len(no_sat)}/{len(results)}")

    print("=" * 60)


if __name__ == "__main__":
    test_brightness_contrast_grid()
