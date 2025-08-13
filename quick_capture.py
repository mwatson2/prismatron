#!/usr/bin/env python3
"""Quick capture with current camera settings."""

import cv2
import numpy as np


def capture_with_current_settings():
    """Capture image with current v4l2-ctl settings."""
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Warm up
    print("Warming up...")
    for i in range(10):
        ret, frame = cap.read()
        print(f"Frame {i+1}: {ret}")

    print("Capturing...")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture")
        return

    # Crop to LED region (from camera config)
    crop_region = (598, 200, 946, 568)
    x, y, x2, y2 = crop_region
    cropped = frame[y:y2, x:x2]

    # Get overall stats
    mean_bgr = np.mean(cropped, axis=(0, 1))
    max_bgr = np.max(cropped, axis=(0, 1))
    mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])
    max_rgb = (max_bgr[2], max_bgr[1], max_bgr[0])

    print(f"Image mean RGB: ({mean_rgb[0]:.1f}, {mean_rgb[1]:.1f}, {mean_rgb[2]:.1f})")
    print(f"Image max RGB: ({max_rgb[0]:.0f}, {max_rgb[1]:.0f}, {max_rgb[2]:.0f})")
    print(f"Max brightness: {max(max_rgb):.0f}")

    if max(max_rgb) >= 254:
        print("⚠️  STILL SATURATED!")
    else:
        print("✅ No saturation!")

    # Save image
    filename = "minimal_settings_test.jpg"
    cv2.imwrite(filename, cropped)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    capture_with_current_settings()
