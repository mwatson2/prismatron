#!/bin/bash

# Prismatron User Service Uninstall Script

set -e

echo "=== Prismatron User Service Uninstallation ==="
echo ""
echo "This will:"
echo "- Stop and disable the prismatron user service"
echo "- Remove the service file"
echo "- Keep application files and logs intact"
echo ""

read -p "Continue? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstallation cancelled"
    exit 0
fi

# Stop service if running
if systemctl --user is-active --quiet prismatron; then
    echo "Stopping prismatron service..."
    systemctl --user stop prismatron
    echo "✓ Service stopped"
else
    echo "Service is not running"
fi

# Disable service if enabled
if systemctl --user is-enabled --quiet prismatron 2>/dev/null; then
    echo "Disabling prismatron service..."
    systemctl --user disable prismatron
    echo "✓ Service disabled"
else
    echo "Service is not enabled"
fi

# Remove service file
SERVICE_FILE="$HOME/.config/systemd/user/prismatron.service"
if [ -f "$SERVICE_FILE" ]; then
    echo "Removing service file..."
    rm "$SERVICE_FILE"
    echo "✓ Service file removed"
fi

# Reload systemd
echo "Reloading systemd daemon..."
systemctl --user daemon-reload
echo "✓ Systemd reloaded"

echo ""
echo "=== Uninstallation Complete ==="
echo ""
echo "Application files remain in place."
echo "To run manually:"
echo "  cd $(dirname "$(dirname "${BASH_SOURCE[0]}")")"
echo "  source env/bin/activate"
echo "  python main.py"
echo ""
