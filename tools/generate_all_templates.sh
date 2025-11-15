#!/bin/bash
# Generate all LED effect templates and optimize them

set -e  # Exit on error

echo "=== Generating LED Effect Templates ==="
echo ""

# Activate virtual environment
source env/bin/activate

# Common parameters
FRAMES=30
WIDTH=800
HEIGHT=480

echo "1. Generating wide ring template (80px width)..."
python tools/generate_wide_ring_template.py \
    --frames $FRAMES \
    --width $WIDTH \
    --height $HEIGHT \
    --ring-width 80.0 \
    --output "wide_ring_800x480.npy" \
    --output-dir templates
echo ""

echo "2. Generating heart template (40px outline)..."
python tools/generate_heart_template.py \
    --frames $FRAMES \
    --width $WIDTH \
    --height $HEIGHT \
    --outline-width 40.0 \
    --output "heart_800x480.npy" \
    --output-dir templates
echo ""

echo "3. Generating 7-pointed star template (40px outline)..."
python tools/generate_star_template.py \
    --frames $FRAMES \
    --width $WIDTH \
    --height $HEIGHT \
    --outline-width 40.0 \
    --num-points 7 \
    --output "star7_800x480.npy" \
    --output-dir templates
echo ""

echo "=== Optimizing Templates to LED Patterns ==="
echo ""

echo "4. Optimizing wide ring to LEDs..."
python tools/optimize_template_to_leds.py \
    templates/wide_ring_800x480.npy \
    --output templates/wide_ring_800x480_leds.npy \
    --max-iterations 10
echo ""

echo "5. Optimizing heart to LEDs..."
python tools/optimize_template_to_leds.py \
    templates/heart_800x480.npy \
    --output templates/heart_800x480_leds.npy \
    --max-iterations 10
echo ""

echo "6. Optimizing star to LEDs..."
python tools/optimize_template_to_leds.py \
    templates/star7_800x480.npy \
    --output templates/star7_800x480_leds.npy \
    --max-iterations 10
echo ""

echo "=== All templates generated and optimized! ==="
ls -lh templates/*_800x480*.npy
