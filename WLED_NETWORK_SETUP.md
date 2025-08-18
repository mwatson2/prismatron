# WLED Network Setup Guide

This guide explains how to configure the Prismatron device to provide a dedicated ethernet network for WLED devices with NAT routing to the internet.

## Overview

The setup creates a dedicated subnet for WLED communication while providing internet access through NAT routing via the wifi interface.

### Network Architecture
```
Internet
    |
    | (wifi)
[Prismatron Device]
    |
    | (ethernet - 10.0.10.0/24)
[WLED Device]
```

## Network Configuration

| Device | Interface | IP Address | Role |
|--------|-----------|------------|------|
| Prismatron | WiFi (wlP1p1s0) | DHCP (192.168.x.x) | Internet connection |
| Prismatron | Ethernet (enP8p1s0) | 10.0.10.1/24 | Gateway for WLED |
| WLED | Ethernet | 10.0.10.100/24 | LED controller |

## Setup Scripts

Three scripts are provided for different setup phases:

### 1. Immediate Setup (`setup_wled_network.sh`)
Configures the network immediately (temporary until reboot):
```bash
./setup_wled_network.sh
```

**What it does:**
- Configures ethernet interface with static IP 10.0.10.1/24
- Enables IP forwarding
- Sets up NAT/masquerading rules with iptables
- Displays current configuration

### 2. Persistent Setup (`setup_wled_network_persistent.sh`)
Creates systemd service for automatic configuration on boot:
```bash
./setup_wled_network_persistent.sh
sudo systemctl start wled-network.service
```

**What it does:**
- Creates `/etc/systemd/system/wled-network.service`
- Configures IP forwarding in `/etc/sysctl.d/99-wled-forwarding.conf`
- Enables automatic startup on boot
- Handles duplicate rule prevention

### 3. Test Configuration (`test_wled_network.sh`)
Verifies the network setup is working correctly:
```bash
./test_wled_network.sh
```

**What it tests:**
- Ethernet interface status
- IP forwarding configuration
- NAT and firewall rules
- Internet connectivity
- Displays configuration summary

## Step-by-Step Setup

### Step 1: Run Immediate Setup
```bash
cd /mnt/dev/prismatron
./setup_wled_network.sh
```

### Step 2: Make Configuration Persistent
```bash
./setup_wled_network_persistent.sh
sudo systemctl start wled-network.service
```

### Step 3: Verify Setup
```bash
./test_wled_network.sh
```

### Step 4: Configure WLED Device
Connect your WLED device via ethernet and configure it with:

- **IP Address:** `10.0.10.100`
- **Subnet Mask:** `255.255.255.0`
- **Gateway:** `10.0.10.1`
- **DNS:** `8.8.8.8`

## WLED Device Configuration

### Option 1: Web Interface
1. Connect WLED to ethernet
2. Access WLED web interface (may need to scan for IP initially)
3. Go to Config → WiFi Setup → Ethernet Settings
4. Configure static IP settings as shown above

### Option 2: JSON API
```bash
curl -X POST http://10.0.10.100/json/cfg \
  -H "Content-Type: application/json" \
  -d '{
    "eth": {
      "type": 1,
      "ip": [10,0,10,100],
      "gw": [10,0,10,1],
      "sn": [255,255,255,0],
      "dns": [8,8,8,8]
    }
  }'
```

## Troubleshooting

### Check Service Status
```bash
sudo systemctl status wled-network.service
```

### View Service Logs
```bash
sudo journalctl -u wled-network.service -f
```

### Manual Network Reset
```bash
# Stop service
sudo systemctl stop wled-network.service

# Clear iptables rules
sudo iptables -t nat -F POSTROUTING
sudo iptables -F FORWARD

# Remove IP from interface
sudo ip addr del 10.0.10.1/24 dev enP8p1s0

# Restart service
sudo systemctl start wled-network.service
```

### Check Ethernet Cable Connection
```bash
# Check if cable is connected
ip link show enP8p1s0
# Should show "UP" and no "NO-CARRIER"

# Check for link activity
ethtool enP8p1s0
```

### Test WLED Connectivity
```bash
# Ping WLED device
ping 10.0.10.100

# Test WLED HTTP interface
curl http://10.0.10.100/json/info

# Check if WLED can reach internet
# (configure WLED first, then check its connectivity)
```

## Advanced Configuration

### Custom IP Range
To use a different IP range, modify the scripts:
1. Change `10.0.10.1/24` to your desired gateway IP
2. Update iptables rules to match new interface
3. Configure WLED with corresponding subnet

### Multiple WLED Devices
The current setup supports multiple WLED devices:
- Use IPs: 10.0.10.100, 10.0.10.101, 10.0.10.102, etc.
- All use gateway: 10.0.10.1
- DHCP can be added if needed for automatic IP assignment

### Firewall Rules
Current setup allows all traffic. For enhanced security, add specific rules:
```bash
# Allow only HTTP/HTTPS for WLED updates
sudo iptables -A FORWARD -i enP8p1s0 -p tcp --dport 80 -j ACCEPT
sudo iptables -A FORWARD -i enP8p1s0 -p tcp --dport 443 -j ACCEPT
sudo iptables -A FORWARD -i enP8p1s0 -j DROP
```

## Maintenance

### Disable WLED Network
```bash
sudo systemctl stop wled-network.service
sudo systemctl disable wled-network.service
```

### Remove Configuration
```bash
sudo systemctl stop wled-network.service
sudo systemctl disable wled-network.service
sudo rm /etc/systemd/system/wled-network.service
sudo rm /etc/sysctl.d/99-wled-forwarding.conf
sudo systemctl daemon-reload
```

## Integration with Prismatron

The WLED network setup is designed to work seamlessly with the Prismatron LED system:

1. **Consumer Integration:** Update WLED client configuration to use `10.0.10.100`
2. **Reliable Communication:** Static IP eliminates network discovery issues
3. **Isolated Network:** WLED traffic doesn't interfere with main system network
4. **Internet Access:** WLED can still receive firmware updates and sync time

### Update Prismatron Configuration
In your Prismatron configuration files, update WLED client settings:
```python
WLED_HOST = "10.0.10.100"
WLED_PORT = 21324  # DDP protocol port
```

This dedicated network setup ensures reliable, high-performance communication between Prismatron and WLED devices while maintaining internet connectivity for both systems.
