#!/bin/bash
set -e

INSTALL_DIR="/mnt/dev/prismatron"
SERVICE_USER="prismatron"
SERVICE_FILE="/etc/systemd/system/prismatron.service"

echo "=== Prismatron Service Installation ==="
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root or with sudo"
   exit 1
fi

# 1. Create system user
if ! id "$SERVICE_USER" &>/dev/null; then
    echo "Creating system user: $SERVICE_USER"
    useradd -r -s /bin/false -d "$INSTALL_DIR" -M "$SERVICE_USER"
    usermod -a -G video,gpio,i2c,dialout,audio "$SERVICE_USER"
    echo "✓ User created and added to necessary groups"
else
    echo "User $SERVICE_USER already exists"
    usermod -a -G video,gpio,i2c,dialout,audio "$SERVICE_USER"
    echo "✓ User groups updated"
fi

# 2. Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/uploads"
mkdir -p "$INSTALL_DIR/playlists"
mkdir -p "$INSTALL_DIR/media"
mkdir -p "$INSTALL_DIR/thumbnails"
echo "✓ Directories created"

# 3. Set directory permissions
echo ""
echo "Setting directory permissions..."
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
chmod 755 "$INSTALL_DIR"
chmod 775 "$INSTALL_DIR"/logs 2>/dev/null || true
chmod 775 "$INSTALL_DIR"/uploads 2>/dev/null || true
chmod 775 "$INSTALL_DIR"/playlists 2>/dev/null || true
chmod 775 "$INSTALL_DIR"/media 2>/dev/null || true
chmod 775 "$INSTALL_DIR"/thumbnails 2>/dev/null || true
echo "✓ Permissions set"

# 4. Create runtime directories
echo ""
echo "Creating runtime directories..."
mkdir -p /run/prismatron
chown "$SERVICE_USER:$SERVICE_USER" /run/prismatron
echo "✓ Runtime directory created"

# 5. Install systemd service
echo ""
echo "Installing systemd service..."
cp "$INSTALL_DIR/scripts/prismatron.service" "$SERVICE_FILE"
echo "✓ Service file installed"

# 6. Install sudoers configuration
echo ""
echo "Configuring sudo permissions..."
cat <<EOF > /etc/sudoers.d/prismatron
# Allow prismatron user to restart its own service and reboot
$SERVICE_USER ALL=(root) NOPASSWD: /bin/systemctl restart prismatron.service
$SERVICE_USER ALL=(root) NOPASSWD: /bin/systemctl reboot
$SERVICE_USER ALL=(root) NOPASSWD: /bin/systemctl poweroff
EOF
chmod 440 /etc/sudoers.d/prismatron
echo "✓ Sudo permissions configured"

# 7. Create tmpfiles configuration
echo ""
echo "Creating tmpfiles configuration..."
echo "d /run/prismatron 0755 $SERVICE_USER $SERVICE_USER -" > /etc/tmpfiles.d/prismatron.conf
systemd-tmpfiles --create /etc/tmpfiles.d/prismatron.conf
echo "✓ Tmpfiles configuration created"

# 8. Create log rotation configuration
echo ""
echo "Setting up log rotation..."
cat <<EOF > /etc/logrotate.d/prismatron
$INSTALL_DIR/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 $SERVICE_USER $SERVICE_USER
    sharedscripts
    postrotate
        systemctl reload prismatron 2>/dev/null || true
    endscript
}
EOF
echo "✓ Log rotation configured"

# 9. Reload systemd
echo ""
echo "Reloading systemd configuration..."
systemctl daemon-reload
echo "✓ Systemd reloaded"

# 10. Enable service
echo ""
echo "Enabling service for automatic startup..."
systemctl enable prismatron.service
echo "✓ Service enabled"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Service Management Commands:"
echo "  Start service:   sudo systemctl start prismatron"
echo "  Stop service:    sudo systemctl stop prismatron"
echo "  Restart service: sudo systemctl restart prismatron"
echo "  View status:     sudo systemctl status prismatron"
echo ""
echo "Log Commands:"
echo "  View logs:       sudo journalctl -u prismatron -f"
echo "  View app logs:   tail -f $INSTALL_DIR/logs/prismatron.log"
echo ""
echo "To start the service now, run:"
echo "  sudo systemctl start prismatron"
echo ""
