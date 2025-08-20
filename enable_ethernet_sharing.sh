#!/bin/bash
# Enable internet sharing from WiFi to Ethernet for QuinLED

echo "Setting up internet sharing from WiFi (wlP1p1s0) to Ethernet (enP8p1s0)..."

# Method 1: Using NetworkManager connection sharing
echo "Configuring NetworkManager connection sharing..."
sudo nmcli connection modify "Wired connection 1" ipv4.method shared
sudo nmcli connection down "Wired connection 1"
sudo nmcli connection up "Wired connection 1"

echo "Ethernet sharing enabled!"
echo ""
echo "The Jetson will now act as a DHCP server on the ethernet interface."
echo "The QuinLED should get an IP in the 10.42.0.x range automatically."
echo ""
echo "To use static IP instead, configure the QuinLED with:"
echo "  IP: 192.168.1.2"
echo "  Subnet: 255.255.255.0"
echo "  Gateway: 192.168.1.1"
echo "  DNS: 8.8.8.8 or 192.168.1.1"
