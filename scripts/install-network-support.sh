#!/bin/bash
#
# Prismatron Network Support Installation Script
# Installs and configures WiFi AP mode support on Jetson Orin Nano
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
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Install required packages
install_packages() {
    log_info "Installing required packages..."

    apt-get update

    # Install network management packages
    PACKAGES=(
        "hostapd"           # Access Point daemon
        "dnsmasq"           # DHCP and DNS server
        "python3-pip"       # Python package manager
        "python3-dev"       # Python development headers
        "python3-setuptools" # Python setuptools
    )

    for package in "${PACKAGES[@]}"; do
        if dpkg -l | grep -q "^ii  $package "; then
            log_info "$package is already installed"
        else
            log_info "Installing $package..."
            apt-get install -y "$package"
        fi
    done

    log_success "Required packages installed"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."

    # Install in the virtual environment if it exists, otherwise system-wide
    if [[ -f "/mnt/dev/prismatron/env/bin/activate" ]]; then
        log_info "Installing in virtual environment..."
        source /mnt/dev/prismatron/env/bin/activate
        pip install python-networkmanager asyncio-subprocess
    else
        log_info "Installing system-wide..."
        pip3 install python-networkmanager asyncio-subprocess
    fi

    log_success "Python dependencies installed"
}

# Create required directories
create_directories() {
    log_info "Creating required directories..."

    mkdir -p /etc/prismatron
    mkdir -p /opt/prismatron/scripts
    mkdir -p /var/log

    # Set permissions
    chmod 755 /etc/prismatron
    chmod 755 /opt/prismatron
    chmod 755 /opt/prismatron/scripts

    log_success "Directories created"
}

# Install service files
install_service() {
    log_info "Installing network management service..."

    # Copy service file
    cp /mnt/dev/prismatron/scripts/prismatron-network.service /etc/systemd/system/

    # Copy startup/shutdown scripts
    cp /mnt/dev/prismatron/scripts/network-startup.py /opt/prismatron/scripts/
    cp /mnt/dev/prismatron/scripts/network-shutdown.py /opt/prismatron/scripts/

    # Make scripts executable
    chmod +x /opt/prismatron/scripts/network-startup.py
    chmod +x /opt/prismatron/scripts/network-shutdown.py

    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable prismatron-network.service

    log_success "Network management service installed and enabled"
}

# Configure NetworkManager
configure_networkmanager() {
    log_info "Configuring NetworkManager..."

    # Ensure NetworkManager is managing the wireless interface
    if [[ ! -f "/etc/NetworkManager/conf.d/10-globally-managed-devices.conf" ]]; then
        cat > /etc/NetworkManager/conf.d/10-globally-managed-devices.conf << EOF
[keyfile]
unmanaged-devices=none
EOF
    fi

    # Create NetworkManager configuration for Prismatron
    cat > /etc/NetworkManager/conf.d/99-prismatron.conf << EOF
[main]
dhcp=internal

[device]
wifi.scan-rand-mac-address=no

[connection]
wifi.powersave=2
EOF

    log_success "NetworkManager configured"
}

# Set up log rotation
configure_logging() {
    log_info "Configuring log rotation..."

    cat > /etc/logrotate.d/prismatron-network << EOF
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
}

# Create initial network configuration
create_initial_config() {
    log_info "Creating initial network configuration..."

    cat > /etc/prismatron/network-config.json << EOF
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
}

# Restart required services
restart_services() {
    log_info "Restarting NetworkManager..."
    systemctl restart NetworkManager

    log_info "Starting Prismatron network service..."
    systemctl start prismatron-network.service

    log_success "Services restarted"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    # Check if required commands are available
    COMMANDS=("nmcli" "hostapd" "dnsmasq")
    for cmd in "${COMMANDS[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            log_info "✓ $cmd is available"
        else
            log_error "✗ $cmd is not available"
            return 1
        fi
    done

    # Check if service is enabled
    if systemctl is-enabled prismatron-network.service >/dev/null 2>&1; then
        log_info "✓ prismatron-network service is enabled"
    else
        log_error "✗ prismatron-network service is not enabled"
        return 1
    fi

    # Check if service is running
    if systemctl is-active prismatron-network.service >/dev/null 2>&1; then
        log_info "✓ prismatron-network service is running"
    else
        log_warning "⚠ prismatron-network service is not running (this is OK, it will start on boot)"
    fi

    log_success "Installation verification completed"
}

# Main installation function
main() {
    log_info "Starting Prismatron network support installation..."

    check_root
    install_packages
    install_python_deps
    create_directories
    configure_networkmanager
    create_initial_config
    install_service
    configure_logging
    restart_services
    verify_installation

    log_success "Prismatron network support installation completed!"
    log_info ""
    log_info "The system will now start in AP mode by default with SSID 'prismatron' (no password)."
    log_info "You can configure network settings through the web interface at:"
    log_info "  - AP mode: http://192.168.4.1"
    log_info "  - Client mode: http://<device-ip>"
    log_info ""
    log_info "To check service status: sudo systemctl status prismatron-network"
    log_info "To view logs: sudo journalctl -u prismatron-network -f"
}

# Run main function
main "$@"
