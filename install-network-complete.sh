#!/bin/bash
#
# Complete Prismatron Network Management Installation Script
# Run with: sudo bash install-network-complete.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

log_info "Starting Prismatron Network Management Installation..."

# Step 1: Install Required Packages
log_info "Installing required packages..."
apt-get update
apt-get install -y hostapd dnsmasq python3-pip python3-dev python3-setuptools

# Fix dnsmasq service conflicts - we'll let NetworkManager handle DHCP
log_info "Configuring services to avoid conflicts..."
# Stop and disable dnsmasq system service - NetworkManager will handle DHCP for AP mode
systemctl stop dnsmasq || true
systemctl disable dnsmasq || true

# Stop and disable hostapd system service - NetworkManager will manage AP mode
systemctl stop hostapd || true
systemctl disable hostapd || true

log_success "Required packages installed and configured"

# Step 2: Install Python Dependencies
log_info "Installing Python dependencies..."
pip3 install python-networkmanager
log_success "Python dependencies installed"

# Step 3: Create Required Directories
log_info "Creating required directories..."
mkdir -p /etc/prismatron
mkdir -p /opt/prismatron/scripts
mkdir -p /var/log
chmod 755 /etc/prismatron /opt/prismatron /opt/prismatron/scripts
log_success "Directories created"

# Step 4: Copy Service Files
log_info "Installing service files..."
cp /mnt/dev/prismatron/scripts/prismatron-network.service /etc/systemd/system/
cp /mnt/dev/prismatron/scripts/network-startup.py /opt/prismatron/scripts/
cp /mnt/dev/prismatron/scripts/network-shutdown.py /opt/prismatron/scripts/
chmod +x /opt/prismatron/scripts/network-startup.py
chmod +x /opt/prismatron/scripts/network-shutdown.py
log_success "Service files installed"

# Step 5: Configure NetworkManager
log_info "Configuring NetworkManager..."
cat > /etc/NetworkManager/conf.d/10-globally-managed-devices.conf << 'EOF'
[keyfile]
unmanaged-devices=none
EOF

cat > /etc/NetworkManager/conf.d/99-prismatron.conf << 'EOF'
[main]
dhcp=internal

[device]
wifi.scan-rand-mac-address=no

[connection]
wifi.powersave=2
EOF
log_success "NetworkManager configured"

# Step 6: Create Initial Network Configuration
log_info "Creating initial network configuration..."
cat > /etc/prismatron/network-config.json << 'EOF'
{
  "mode": "ap",
  "ap_config": {
    "ssid": "prismatron",
    "password": null,
    "ip_address": "192.168.4.1",
    "netmask": "255.255.255.0",
    "dhcp_start": "192.168.4.2",
    "dhcp_end": "192.168.4.100",
    "channel": 6
  },
  "client_config": null,
  "startup_mode": "ap"
}
EOF
chmod 644 /etc/prismatron/network-config.json
log_success "Initial network configuration created"

# Step 7: Set Up Log Rotation
log_info "Configuring log rotation..."
cat > /etc/logrotate.d/prismatron-network << 'EOF'
/var/log/prismatron-network.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 644 root root
}
EOF
log_success "Log rotation configured"

# Step 8: Enable and Start the Service
log_info "Enabling and starting services..."
systemctl daemon-reload
systemctl enable prismatron-network.service
systemctl restart NetworkManager

# Wait a moment for NetworkManager to restart
sleep 3

systemctl start prismatron-network.service
log_success "Services enabled and started"

# Step 9: Verify Installation
log_info "Verifying installation..."

# Check if required commands are available
COMMANDS=("nmcli" "hostapd" "dnsmasq")
for cmd in "${COMMANDS[@]}"; do
    if command -v "$cmd" >/dev/null 2>&1; then
        log_info "✓ $cmd is available"
    else
        log_error "✗ $cmd is not available"
        exit 1
    fi
done

# Check if service is enabled
if systemctl is-enabled prismatron-network.service >/dev/null 2>&1; then
    log_info "✓ prismatron-network service is enabled"
else
    log_error "✗ prismatron-network service is not enabled"
    exit 1
fi

# Check if service is running
if systemctl is-active prismatron-network.service >/dev/null 2>&1; then
    log_info "✓ prismatron-network service is running"
else
    log_warning "⚠ prismatron-network service is not running - checking logs..."
    journalctl -u prismatron-network --no-pager --lines=10
fi

log_success "Installation verification completed"

log_success "Prismatron Network Management Installation Completed!"
log_info ""
log_info "The system will now start in AP mode by default with SSID 'prismatron' (no password)."
log_info "You can configure network settings through the web interface at:"
log_info "  - AP mode: http://192.168.4.1:8080"
log_info "  - Current IP: http://$(hostname -I | awk '{print $1}'):8080"
log_info ""
log_info "To check service status: sudo systemctl status prismatron-network"
log_info "To view logs: sudo journalctl -u prismatron-network -f"
log_info "To restart network: sudo systemctl restart prismatron-network"

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Update installation script paths for current directory structure", "status": "completed", "id": "1"}, {"content": "Install network management system dependencies", "status": "completed", "id": "2"}, {"content": "Configure NetworkManager for AP mode support", "status": "pending", "id": "3"}, {"content": "Install prismatron-network.service as system service", "status": "pending", "id": "4"}, {"content": "Test network management functionality", "status": "pending", "id": "5"}, {"content": "Provide manual installation commands for user to run", "status": "completed", "id": "6"}]
