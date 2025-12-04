#!/bin/bash
# Enable NAT-based internet sharing while keeping static IPs

echo "Setting up NAT-based internet sharing..."

# Enable IP forwarding
echo "Enabling IP forwarding..."
sudo sysctl -w net.ipv4.ip_forward=1

# Add NAT rules
echo "Adding NAT rules..."
sudo iptables -t nat -A POSTROUTING -o wlP1p1s0 -j MASQUERADE
sudo iptables -A FORWARD -i enP8p1s0 -o wlP1p1s0 -j ACCEPT
sudo iptables -A FORWARD -i wlP1p1s0 -o enP8p1s0 -m state --state RELATED,ESTABLISHED -j ACCEPT

echo "NAT sharing enabled!"
echo ""
echo "Now configure the QuinLED with:"
echo "  IP Address: 192.168.1.2"
echo "  Subnet Mask: 255.255.255.0"
echo "  Gateway: 192.168.1.1  (this Jetson)"
echo "  DNS: 8.8.8.8 or 1.1.1.1"
echo ""
echo "The QuinLED should now have internet access through the Jetson!"
