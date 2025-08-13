#!/usr/bin/env python3
"""Simple brightness/contrast test with manual camera capture."""

import os
import subprocess
import time

import cv2
import numpy as np


def configure_camera(brightness, contrast, gain=20, exposure=100):
    """Configure camera with v4l2-ctl."""
    device = "/dev/video0"

    commands = [
        f"v4l2-ctl -d {device} -c white_balance_automatic=0",
        f"v4l2-ctl -d {device} -c auto_exposure=1",
        f"v4l2-ctl -d {device} -c focus_automatic_continuous=0",
        f"v4l2-ctl -d {device} -c exposure_dynamic_framerate=0",
        f"v4l2-ctl -d {device} -c backlight_compensation=0",
        f"v4l2-ctl -d {device} -c exposure_time_absolute={exposure}",
        f"v4l2-ctl -d {device} -c gain={gain}",
        f"v4l2-ctl -d {device} -c brightness={brightness}",
        f"v4l2-ctl -d {device} -c contrast={contrast}",
        f"v4l2-ctl -d {device} -c saturation=128",
    ]

    print(f"\nSetting Brightness={brightness}, Contrast={contrast}")
    for cmd in commands:
        subprocess.run(cmd, shell=True, capture_output=True)

    # Verify settings
    result = subprocess.run(
        f"v4l2-ctl -d {device} --list-ctrls | grep -E '(brightness|contrast|gain|exposure_time)' | grep value",
        shell=True,
        capture_output=True,
        text=True,
    )
    print("Current settings:")
    print(result.stdout)


def capture_and_analyze():
    """Capture frame and analyze."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Warm up
    print("Warming up camera...")
    for i in range(10):
        ret, _ = cap.read()
        if not ret:
            print(f"  Frame {i}: Failed")
            break

    time.sleep(1)

    # Capture
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture")
        return

    # Crop to LED region
    crop = frame[200:568, 598:946]

    # Find brightest region
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

    # Get 20x20 region around brightest point
    x, y = max_loc
    roi_size = 20
    x1 = max(0, x - roi_size // 2)
    y1 = max(0, y - roi_size // 2)
    x2 = min(crop.shape[1], x + roi_size // 2)
    y2 = min(crop.shape[0], y + roi_size // 2)

    led_roi = crop[y1:y2, x1:x2]

    # Calculate stats
    mean_bgr = np.mean(led_roi, axis=(0, 1))
    max_bgr = np.max(led_roi, axis=(0, 1))

    mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])
    max_rgb = (max_bgr[2], max_bgr[1], max_bgr[0])

    # Find actual visible area (pixels above threshold)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    visible_pixels = np.sum(thresh > 0)

    print(f"LED ROI Mean RGB: ({mean_rgb[0]:.1f}, {mean_rgb[1]:.1f}, {mean_rgb[2]:.1f})")
    print(f"LED ROI Max RGB: ({max_rgb[0]:.0f}, {max_rgb[1]:.0f}, {max_rgb[2]:.0f})")
    print(f"Visible pixels (>30): {visible_pixels}")
    print(f"Max brightness: {max(max_rgb):.0f} {'[SATURATED]' if max(max_rgb) >= 254 else '[OK]'}")

    return crop, {
        "mean_rgb": mean_rgb,
        "max_rgb": max_rgb,
        "visible_pixels": visible_pixels,
        "saturated": max(max_rgb) >= 254,
    }


def main():
    """Test brightness and contrast grid."""
    print("=" * 60)
    print("BRIGHTNESS/CONTRAST TEST")
    print(
        "Make sure LED is already lit with: python tools/wled_test_patterns.py --led-count 3000 single --index 1 --color 0 100 0"
    )
    print("=" * 60)

    os.makedirs("brightness_contrast", exist_ok=True)

    # Test grid
    brightness_values = [64, 128, 192]
    contrast_values = [64, 128, 192]

    results = []

    for brightness in brightness_values:
        for contrast in contrast_values:
            # Configure camera
            configure_camera(brightness, contrast)
            time.sleep(2)  # Let settings stabilize

            # Capture and analyze
            image, stats = capture_and_analyze()

            if image is not None and stats:
                # Save image
                filename = f"brightness_contrast/B{brightness}_C{contrast}.jpg"
                cv2.imwrite(filename, image)
                print(f"Saved: {filename}")

                results.append({"brightness": brightness, "contrast": contrast, **stats})

            print("-" * 40)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Effect on Visible Diffusion Area:")
    print("=" * 60)
    print("Brightness  Contrast  Visible Pixels  Max Green  Status")
    print("-" * 60)

    for r in results:
        status = "SAT" if r["saturated"] else "OK"
        print(
            f"{r['brightness']:^10} {r['contrast']:^10} {r['visible_pixels']:^14} "
            f"{r['max_rgb'][1]:^10.0f} {status:^6}"
        )

    # Find best configuration
    if results:
        best = max(results, key=lambda x: x["visible_pixels"] if not x["saturated"] else 0)
        print(
            f"\nBest (most visible, no saturation): "
            f"Brightness={best['brightness']}, Contrast={best['contrast']} "
            f"({best['visible_pixels']} pixels)"
        )


if __name__ == "__main__":
    main()
