#!/bin/bash

# Prismatron User Service Setup Script
# Installs prismatron as a systemd user service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Prismatron User Service Installation ==="
echo ""
echo "This will install Prismatron as a user service for: $USER"
echo "Project directory: $PROJECT_DIR"
echo ""

# Create user systemd directory if it doesn't exist
mkdir -p ~/.config/systemd/user/

# Copy service file to user systemd directory
echo "Installing service file..."
cp "$SCRIPT_DIR/prismatron-user.service" ~/.config/systemd/user/prismatron.service

# Update paths in service file to match current user and directory
sed -i "s|/home/mark/dev/prismatron|$PROJECT_DIR|g" ~/.config/systemd/user/prismatron.service

# Reload systemd daemon to recognize new service
echo "Reloading systemd daemon..."
systemctl --user daemon-reload

# Enable service to start on boot
echo "Enabling service for automatic startup..."
systemctl --user enable prismatron.service

# Enable lingering so service runs without user logged in
echo "Enabling user lingering (service runs when not logged in)..."
sudo loginctl enable-linger $USER

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Service Management Commands:"
echo "  Start service:   systemctl --user start prismatron"
echo "  Stop service:    systemctl --user stop prismatron"
echo "  Restart:         systemctl --user restart prismatron"
echo "  View status:     systemctl --user status prismatron"
echo ""
echo "Log Commands:"
echo "  View logs:       journalctl --user -u prismatron -f"
echo "  Last 100 lines:  journalctl --user -u prismatron -n 100"
echo ""
echo "To start the service now, run:"
echo "  systemctl --user start prismatron"
echo ""
