#!/bin/bash

# WLED Network Test Script
# Tests the network configuration and connectivity

echo "Testing WLED network configuration..."
echo ""

# Check ethernet interface
echo "=== Ethernet Interface Status ==="
ip addr show enP8p1s0
echo ""

# Check IP forwarding
echo "=== IP Forwarding Status ==="
sysctl net.ipv4.ip_forward
echo ""

# Check NAT rules
echo "=== NAT Rules ==="
sudo iptables -t nat -L POSTROUTING -v --line-numbers
echo ""

# Check FORWARD rules
echo "=== FORWARD Rules ==="
sudo iptables -L FORWARD -v --line-numbers
echo ""

# Check routing table
echo "=== Routing Table ==="
ip route show
echo ""

# Test connectivity to common DNS servers
echo "=== Connectivity Test ==="
echo "Testing connectivity to 8.8.8.8..."
ping -c 3 8.8.8.8

echo ""
echo "=== Network Configuration Summary ==="
echo "This device ethernet: $(ip -4 addr show enP8p1s0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo 'Not configured')"
echo "WLED should be configured with:"
echo "  IP: 10.0.10.100 (or 10.0.10.2-254)"
echo "  Gateway: 10.0.10.1"
echo "  DNS: 8.8.8.8"
