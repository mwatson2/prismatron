#!/usr/bin/env python3
"""
Create a test checkerboard image 800x480 with 5x3 pattern.
Black and white squares except center three are red, green, blue.
"""

from PIL import Image, ImageDraw

# Image dimensions
width = 800
height = 480

# Grid dimensions
cols = 5
rows = 3

# Calculate square dimensions
square_width = width // cols
square_height = height // rows

# Create image
img = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(img)

# Define colors
colors = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}

# Draw checkerboard pattern
for row in range(rows):
    for col in range(cols):
        # Calculate square position
        x1 = col * square_width
        y1 = row * square_height
        x2 = x1 + square_width
        y2 = y1 + square_height

        # Determine if this is a center square (middle row, columns 1, 2, 3)
        if row == 1:  # Middle row
            if col == 1:  # Left center
                color = colors["red"]
            elif col == 2:  # Center center
                color = colors["green"]
            elif col == 3:  # Right center
                color = colors["blue"]
            else:
                # Regular checkerboard pattern
                is_black = (row + col) % 2 == 0
                color = colors["black"] if is_black else colors["white"]
        else:
            # Regular checkerboard pattern
            is_black = (row + col) % 2 == 0
            color = colors["black"] if is_black else colors["white"]

        # Draw the square
        draw.rectangle([x1, y1, x2, y2], fill=color)

# Save the image
img.save("/mnt/dev/prismatron/test_checkerboard_800x480.png")
print("Created test_checkerboard_800x480.png")
print(f"Image dimensions: {width}x{height}")
print(f"Grid: {cols}x{rows}")
print(f"Square size: {square_width}x{square_height}")
print("Center squares: red, green, blue (left to right)")
