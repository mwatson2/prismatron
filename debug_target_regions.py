#!/usr/bin/env python3
"""
Debug target region overlap to understand the discrepancy.
"""

import cupy as cp

# Create the same target as in debug script
target = cp.zeros((16, 16), dtype=cp.float32)
target[2:6, 3:7] = 10.0  # Overlaps with LED 0, Channel 0 at (2,3)
target[8:12, 9:13] = 5.0  # Overlaps with LED 0, Channel 1 at (8,9)
target[0:4, 0:4] = 2.0  # Overlaps with LED 1, Channel 0 at (0,0)
target[10:14, 5:9] = 1.0  # Overlaps with LED 1, Channel 1 at (10,5)

print("Target image:")
print(target)

print("\n=== Checking Block Overlaps ===")

# LED 0, Channel 0: block at (2,3), size 4x4
print("LED 0, Channel 0 block region (2:6, 3:7):")
region = target[2:6, 3:7]
print(region)
print(f"Expected all 10.0, but some overlap with other regions!")

# Check if there are overlaps
print(f"\nOverlap check:")
print(
    f"Region [2:6, 3:7] overlaps with [0:4, 0:4]? {not (2 >= 4 or 6 <= 0 or 3 >= 4 or 7 <= 0)}"
)
print(f"Overlap area: rows {max(2,0)}:{min(6,4)}, cols {max(3,0)}:{min(7,4)}")

# The overlap is [2:4, 3:4] = 2 pixels that have value 2.0 instead of 10.0
overlap_region = target[2:4, 3:4]
print(f"Overlap region [2:4, 3:4]: {overlap_region}")

# Calculate correct expected value
# Block is 4x4 = 16 pixels
# 14 pixels have value 10.0, 2 pixels have value 2.0
correct_expected = 14 * 1.0 * 10.0 + 2 * 1.0 * 2.0
print(f"\nCorrect expected for LED 0, Channel 0: {correct_expected}")
print(f"Actual result was: 144.0")
print(f"Match? {abs(correct_expected - 144.0) < 1e-5}")
