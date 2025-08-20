#!/bin/bash

# Setup script for persistent X11 dummy display (headless operation)
# This creates a system service that starts a dummy X server on :99 at boot

echo "=== Setting up Persistent X11 Dummy Display Service ==="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script needs sudo privileges to install the system service"
    echo "Please run: sudo $0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Function to safely stop and disable a service if it exists
safely_stop_service() {
    local service_name=$1
    if systemctl list-units --full -all | grep -Fq "$service_name"; then
        echo "  - Stopping existing $service_name..."
        systemctl stop "$service_name" 2>/dev/null || true
        systemctl disable "$service_name" 2>/dev/null || true
    fi
}

# 1. Check if xorg-dummy.conf exists
if [ ! -f "$PROJECT_DIR/xorg-dummy.conf" ]; then
    echo "ERROR: xorg-dummy.conf not found in $PROJECT_DIR"
    echo "Please ensure xorg-dummy.conf exists in the project directory"
    exit 1
fi

# 2. Copy xorg configuration to system directory if needed
if [ ! -f "/etc/X11/xorg-dummy.conf" ] || ! diff -q "$PROJECT_DIR/xorg-dummy.conf" "/etc/X11/xorg-dummy.conf" > /dev/null 2>&1; then
    echo "Installing/updating X11 dummy configuration..."
    cp "$PROJECT_DIR/xorg-dummy.conf" /etc/X11/
    echo "  - Copied xorg-dummy.conf to /etc/X11/"
else
    echo "  - X11 dummy configuration already up to date"
fi

# 3. Handle existing dummy display service
echo "Checking for existing dummy display service..."
safely_stop_service "xorg-dummy.service"

# Kill any existing Xorg :99 processes
if pgrep -f "Xorg :99" > /dev/null; then
    echo "  - Killing existing Xorg :99 processes..."
    pkill -f "Xorg :99" || true
    sleep 2
fi

# 4. Install the dummy display system service
echo "Installing dummy display system service..."
if [ ! -f "$SCRIPT_DIR/xorg-dummy.service" ]; then
    echo "ERROR: xorg-dummy.service not found in $SCRIPT_DIR"
    echo "Please ensure the service file exists"
    exit 1
fi

cp "$SCRIPT_DIR/xorg-dummy.service" /etc/systemd/system/
echo "  - Service file copied to /etc/systemd/system/"

# 5. Reload systemd and enable the service
systemctl daemon-reload
systemctl enable xorg-dummy.service
echo "  - Service enabled to start at boot"

# 6. Start the service
echo "Starting dummy display service..."
systemctl start xorg-dummy.service

# Wait for the service to initialize
echo "Waiting for X server to initialize..."
for i in {1..10}; do
    if [ -S "/tmp/.X11-unix/X99" ]; then
        break
    fi
    sleep 1
done

# 7. Check if the service started successfully
if systemctl is-active --quiet xorg-dummy.service; then
    echo "✓ Dummy display service started successfully"
    echo ""
    systemctl status xorg-dummy.service --no-pager | head -n 5
else
    echo "✗ Failed to start dummy display service"
    echo "Check logs with: journalctl -u xorg-dummy.service -n 50"
    exit 1
fi

# 8. Verify X server is actually running
if [ -S "/tmp/.X11-unix/X99" ]; then
    echo ""
    echo "✓ X11 socket confirmed at /tmp/.X11-unix/X99"
else
    echo ""
    echo "⚠ Warning: X11 socket not found at /tmp/.X11-unix/X99"
    echo "The service may still be starting. Check with:"
    echo "  ls -la /tmp/.X11-unix/"
    echo "  journalctl -u xorg-dummy.service -f"
fi

# 9. Test the display connection
echo ""
echo "Testing display connection..."
if DISPLAY=:99 xset q > /dev/null 2>&1; then
    echo "✓ Display :99 is accessible"
else
    echo "⚠ Could not connect to display :99 (this may be normal if xset is not installed)"
fi

echo ""
echo "=== Dummy Display Setup Complete ==="
echo ""
echo "Display Information:"
echo "  • Display number: :99"
echo "  • Config file: /etc/X11/xorg-dummy.conf"
echo "  • Service name: xorg-dummy.service"
echo "  • Status: Running and enabled at boot"
echo ""
echo "Service Management:"
echo "  • Status: systemctl status xorg-dummy.service"
echo "  • Logs: journalctl -u xorg-dummy.service -f"
echo "  • Restart: systemctl restart xorg-dummy.service"
echo ""
echo "Next Steps:"
echo "  1. The Prismatron user service already has DISPLAY=:99 configured"
echo "  2. Run the user service setup if needed:"
echo "     $SCRIPT_DIR/setup_user_service.sh"
echo ""
echo "The dummy display will now start automatically at every boot."
