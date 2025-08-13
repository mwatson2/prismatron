#!/bin/bash

# Prismatron Headless Environment Setup
# This script sets up the dummy display for headless operation on Jetson devices

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUMMY_DISPLAY_SCRIPT="$SCRIPT_DIR/start-dummy-display.sh"

echo "=== Prismatron Headless Environment Setup ==="

# Check if dummy display script exists
if [ ! -f "$DUMMY_DISPLAY_SCRIPT" ]; then
    echo "ERROR: Dummy display script not found at $DUMMY_DISPLAY_SCRIPT"
    exit 1
fi

# Start dummy display if not already running
echo "Checking dummy display status..."
if ! $DUMMY_DISPLAY_SCRIPT status | grep -q "Dummy display running"; then
    echo "Starting dummy display..."
    $DUMMY_DISPLAY_SCRIPT start

    if [ $? -eq 0 ]; then
        echo "✓ Dummy display started successfully"
    else
        echo "✗ Failed to start dummy display"
        exit 1
    fi
else
    echo "✓ Dummy display already running"
fi

# Set environment variables for headless operation
export DISPLAY=:99
export GST_DEBUG=2

echo ""
echo "=== Environment configured for headless operation ==="
echo "DISPLAY=$DISPLAY"
echo "GST_DEBUG=$GST_DEBUG"
echo ""
echo "To use this environment in your shell, run:"
echo "  source $0"
echo ""
echo "To test GStreamer with NVIDIA elements:"
echo "  gst-launch-1.0 videotestsrc num-buffers=10 ! nvvidconv ! fakesink"
echo ""

# If sourced, export the variables for the current shell
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Environment variables exported to current shell"
fi
