#!/bin/bash

# Script to start a dummy display for headless Jetson operation
# This resolves EGL display connection errors in GStreamer/FFmpeg

DISPLAY_NUM=99
XORG_CONF="/etc/X11/xorg-dummy.conf"
PID_FILE="/tmp/Xorg-dummy.pid"

# Function to start dummy display
start_dummy_display() {
    echo "Starting dummy display on :$DISPLAY_NUM"

    # Kill existing dummy display if running
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "Killing existing dummy display (PID: $OLD_PID)"
            kill "$OLD_PID"
            sleep 1
        fi
        rm -f "$PID_FILE"
    fi

    # Start new dummy display
    echo "prism" | sudo -S Xorg :$DISPLAY_NUM -config "$XORG_CONF" -logfile /tmp/Xorg-dummy.log &
    XORG_PID=$!

    # Save PID
    echo "$XORG_PID" > "$PID_FILE"

    # Wait for X server to start
    sleep 2

    # Check if X server started successfully
    if kill -0 "$XORG_PID" 2>/dev/null; then
        echo "Dummy display started successfully (PID: $XORG_PID)"
        echo "Set DISPLAY=:$DISPLAY_NUM to use this display"
        return 0
    else
        echo "Failed to start dummy display"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Function to stop dummy display
stop_dummy_display() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping dummy display (PID: $PID)"
            kill "$PID"
            rm -f "$PID_FILE"
        else
            echo "Dummy display not running"
            rm -f "$PID_FILE"
        fi
    else
        echo "No dummy display PID file found"
    fi
}

# Function to check status
check_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Dummy display running (PID: $PID) on :$DISPLAY_NUM"
        else
            echo "Dummy display PID file exists but process not running"
            rm -f "$PID_FILE"
        fi
    else
        echo "Dummy display not running"
    fi
}

# Main script logic
case "$1" in
    start)
        start_dummy_display
        ;;
    stop)
        stop_dummy_display
        ;;
    restart)
        stop_dummy_display
        sleep 1
        start_dummy_display
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "This script manages a dummy X11 display server for headless operation"
        echo "on Jetson devices to resolve EGL display connection errors."
        echo ""
        echo "After starting, set DISPLAY=:$DISPLAY_NUM in your environment"
        exit 1
        ;;
esac
