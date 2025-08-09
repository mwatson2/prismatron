#!/usr/bin/env python3
"""
Simple USB camera configuration tool for headless systems.
Creates a camera configuration file for use with capture_diffusion_patterns.py
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Configure USB camera for diffusion pattern capture")
    parser.add_argument("--camera-device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--output", default="camera_usb.json", help="Output configuration file")
    parser.add_argument("--test-image", help="Save a test capture to this file")

    args = parser.parse_args()

    # Try to open the camera
    print(f"Opening USB camera at /dev/video{args.camera_device}...")

    cap = cv2.VideoCapture(args.camera_device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_device)

    if not cap.isOpened():
        print(f"Failed to open camera at /dev/video{args.camera_device}")
        return 1

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera opened successfully: {width}x{height}")

    # Capture a test frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        cap.release()
        return 1

    # Save test image if requested
    if args.test_image:
        cv2.imwrite(args.test_image, frame)
        print(f"Test image saved to {args.test_image}")

    # Calculate crop region for 5:3 aspect ratio (800x480)
    target_aspect = 800 / 480  # 5:3 aspect ratio
    current_aspect = width / height

    if current_aspect > target_aspect:
        # Image is wider than target, crop width
        new_width = int(height * target_aspect)
        x_offset = (width - new_width) // 2
        crop_region = {"x": x_offset, "y": 0, "width": new_width, "height": height}
    else:
        # Image is taller than target, crop height
        new_height = int(width / target_aspect)
        y_offset = (height - new_height) // 2
        crop_region = {"x": 0, "y": y_offset, "width": width, "height": new_height}

    print(f"Calculated crop region: {crop_region}")

    # Create configuration
    config = {
        "camera_device": args.camera_device,
        "use_usb": True,
        "camera_resolution": {"width": width, "height": height},
        "crop_region": crop_region,
        "target_resolution": {"width": 800, "height": 480},
        "aspect_ratio": {"target": target_aspect, "actual": crop_region["width"] / crop_region["height"]},
    }

    # Save configuration
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfiguration saved to {args.output}")
    print("\nYou can now use this configuration with:")
    print(
        f"  python tools/capture_diffusion_patterns.py --wled-host <HOST> --camera-config {args.output} --usb --output patterns.npz"
    )

    cap.release()
    return 0


if __name__ == "__main__":
    exit(main())
