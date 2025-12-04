#!/bin/bash

# WLED Network Persistent Setup Script
# Creates systemd service for persistent network configuration

echo "Creating persistent WLED network configuration..."

# Create systemd service file
sudo tee /etc/systemd/system/wled-network.service > /dev/null << 'EOF'
[Unit]
Description=WLED Network Configuration
After=network.target
Wants=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c '\
    ip addr add 10.0.10.1/24 dev enP8p1s0 2>/dev/null || true; \
    ip link set enP8p1s0 up; \
    iptables -t nat -C POSTROUTING -o wlP1p1s0 -j MASQUERADE 2>/dev/null || iptables -t nat -A POSTROUTING -o wlP1p1s0 -j MASQUERADE; \
    iptables -C FORWARD -i enP8p1s0 -o wlP1p1s0 -j ACCEPT 2>/dev/null || iptables -A FORWARD -i enP8p1s0 -o wlP1p1s0 -j ACCEPT; \
    iptables -C FORWARD -i wlP1p1s0 -o enP8p1s0 -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || iptables -A FORWARD -i wlP1p1s0 -o enP8p1s0 -m state --state RELATED,ESTABLISHED -j ACCEPT'
ExecStop=/bin/bash -c '\
    iptables -t nat -D POSTROUTING -o wlP1p1s0 -j MASQUERADE 2>/dev/null || true; \
    iptables -D FORWARD -i enP8p1s0 -o wlP1p1s0 -j ACCEPT 2>/dev/null || true; \
    iptables -D FORWARD -i wlP1p1s0 -o enP8p1s0 -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || true; \
    ip addr del 10.0.10.1/24 dev enP8p1s0 2>/dev/null || true'

[Install]
WantedBy=multi-user.target
EOF

# Create sysctl configuration for IP forwarding
echo 'net.ipv4.ip_forward=1' | sudo tee /etc/sysctl.d/99-wled-forwarding.conf

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable wled-network.service

echo ""
echo "Persistent configuration created!"
echo ""
echo "To start the network configuration now, run:"
echo "  sudo systemctl start wled-network.service"
echo ""
echo "To check status:"
echo "  sudo systemctl status wled-network.service"
echo ""
echo "The configuration will automatically start on boot."
