#!/bin/bash
set -e

SERVICE_USER="prismatron"
SERVICE_FILE="/etc/systemd/system/prismatron.service"

echo "=== Prismatron Service Uninstallation ==="
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root or with sudo"
   exit 1
fi

echo "This will:"
echo "- Stop and disable the prismatron service"
echo "- Remove service files and configurations"
echo "- Keep the prismatron user and application files"
echo ""
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstallation cancelled"
    exit 0
fi

# 1. Stop and disable service
echo ""
if systemctl is-active --quiet prismatron; then
    echo "Stopping prismatron service..."
    systemctl stop prismatron
    echo "✓ Service stopped"
else
    echo "Service is not running"
fi

if systemctl is-enabled --quiet prismatron; then
    echo "Disabling prismatron service..."
    systemctl disable prismatron
    echo "✓ Service disabled"
else
    echo "Service is not enabled"
fi

# 2. Remove service files
echo ""
echo "Removing service files..."
[ -f "$SERVICE_FILE" ] && rm "$SERVICE_FILE" && echo "✓ Removed service file"
[ -f "/etc/sudoers.d/prismatron" ] && rm "/etc/sudoers.d/prismatron" && echo "✓ Removed sudoers configuration"
[ -f "/etc/tmpfiles.d/prismatron.conf" ] && rm "/etc/tmpfiles.d/prismatron.conf" && echo "✓ Removed tmpfiles configuration"
[ -f "/etc/logrotate.d/prismatron" ] && rm "/etc/logrotate.d/prismatron" && echo "✓ Removed logrotate configuration"

# 3. Remove runtime directory
echo ""
echo "Removing runtime directory..."
[ -d "/run/prismatron" ] && rm -rf "/run/prismatron" && echo "✓ Removed /run/prismatron"

# 4. Reload systemd
echo ""
echo "Reloading systemd configuration..."
systemctl daemon-reload
echo "✓ Systemd reloaded"

echo ""
echo "=== Uninstallation Complete ==="
echo ""
echo "Note: The following were preserved:"
echo "- Prismatron user account ($SERVICE_USER)"
echo "- Application files in /mnt/dev/prismatron"
echo "- Log files and uploaded content"
echo ""
echo "To completely remove everything, run:"
echo "  sudo userdel -r $SERVICE_USER"
echo "  sudo rm -rf /mnt/dev/prismatron"
echo ""
echo "To run manually again:"
echo "  cd /mnt/dev/prismatron"
echo "  source env/bin/activate"
echo "  python main.py"
echo ""
