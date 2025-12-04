#!/bin/bash

# WLED Network Setup Script
# Sets up ethernet interface with static IP and NAT to wifi

echo "Setting up WLED network configuration..."

# Configure ethernet interface with static IP
echo "Configuring ethernet interface enP8p1s0 with IP 10.0.10.1/24..."
sudo ip addr add 10.0.10.1/24 dev enP8p1s0
sudo ip link set enP8p1s0 up

# Verify ethernet interface is up
echo "Ethernet interface status:"
ip addr show enP8p1s0

# Enable IP forwarding
echo "Enabling IP forwarding..."
echo 'net.ipv4.ip_forward=1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Set up NAT/masquerading with iptables
echo "Setting up NAT rules..."
sudo iptables -t nat -A POSTROUTING -o wlP1p1s0 -j MASQUERADE
sudo iptables -A FORWARD -i enP8p1s0 -o wlP1p1s0 -j ACCEPT
sudo iptables -A FORWARD -i wlP1p1s0 -o enP8p1s0 -m state --state RELATED,ESTABLISHED -j ACCEPT

# Display current iptables NAT rules
echo "Current NAT rules:"
sudo iptables -t nat -L POSTROUTING -v
echo "Current FORWARD rules:"
sudo iptables -L FORWARD -v

echo ""
echo "Network setup complete!"
echo ""
echo "Configure your WLED device with:"
echo "  IP Address: 10.0.10.100 (or any IP in 10.0.10.2-254 range)"
echo "  Subnet Mask: 255.255.255.0"
echo "  Gateway: 10.0.10.1"
echo "  DNS: 8.8.8.8 (or your preferred DNS)"
echo ""
echo "To make configuration persistent across reboots:"
echo "1. Install iptables-persistent: sudo apt install iptables-persistent"
echo "2. Save rules: sudo iptables-save | sudo tee /etc/iptables/rules.v4"
echo "3. Create systemd service for network config (see setup_wled_network_persistent.sh)"
