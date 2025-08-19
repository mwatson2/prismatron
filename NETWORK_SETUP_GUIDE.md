# Prismatron Network Setup Guide

This guide documents the complete network setup process for the Prismatron LED Display System on a fresh Jetson Orin Nano installation.

## Prerequisites

- Jetson Orin Nano with Ubuntu 20.04/22.04
- Python 3.10 or higher
- NetworkManager installed (default on Jetson)
- sudo access for system configuration

## Network Architecture

The Prismatron system supports two network modes:

1. **Client Mode**: Connects to existing WiFi networks (default after setup)
2. **AP Mode**: Creates its own WiFi hotspot (SSID: 'prismatron', IP: 192.168.4.1)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/prismatron.git
cd prismatron
```

### 2. Create Python Virtual Environment

```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install pre-commit  # For development
```

### 4. Install Pre-commit Hooks (Development Only)

```bash
pre-commit install
```

### 5. Install Network Support

Run the network installation script as root:

```bash
sudo scripts/install-network-support.sh
```

This script will:
- Install required packages (hostapd, dnsmasq)
- Configure NetworkManager for AP mode support
- Create systemd service for network management
- Set up initial configuration files
- Configure system to start in AP mode by default

### 6. Fix NetworkManager Permissions

Run the permission fix script to allow the application user to manage networks:

```bash
sudo ./fix-network-permissions.sh
```

This script will:
- Add user to `netdev` group
- Create PolicyKit rule for NetworkManager access
- Enable WiFi scanning and network management from the web interface

**Note**: After running this script, you may need to log out and back in for group membership to take effect.

### 7. Configure Network Interface

The system auto-detects the WiFi interface, but if you need to override it:

Edit `/mnt/dev/prismatron/src/network/manager.py` and modify the interface detection order:

```python
# Line 50: Modify the fallback interface list
for interface in ["wlP1p1s0", "wlan0", "wlo1"]:  # Add your interface name
```

### 8. Verify Installation

Check the network service status:

```bash
sudo systemctl status prismatron-network
```

Test network detection:

```bash
python3 -c "
from src.network.manager import NetworkManager
import asyncio
async def test():
    nm = NetworkManager()
    print(f'Interface: {nm.interface}')
    status = await nm.get_status()
    print(f'Status: {status}')
asyncio.run(test())
"
```

## Network Configuration Files

### Main Configuration
- `/etc/prismatron/network-config.json` - Network mode and settings

### Service Files
- `/etc/systemd/system/prismatron-network.service` - systemd service
- `/opt/prismatron/scripts/network-startup.py` - Network initialization
- `/opt/prismatron/scripts/network-shutdown.py` - Network cleanup

### PolicyKit Rules
- `/etc/polkit-1/rules.d/10-prismatron-networkmanager.rules` - NetworkManager permissions

## Common Issues and Solutions

### Issue: "Device 'wlan0' not found"

**Solution**: The system auto-detects WiFi interfaces. If detection fails:
1. Check available interfaces: `nmcli device status`
2. Verify WiFi interface name (usually `wlP1p1s0` on Jetson)
3. System will automatically use the correct interface

### Issue: "WiFi scan request failed: not authorized"

**Solution**: Run the permission fix script:
```bash
sudo ./fix-network-permissions.sh
```

Then restart the application or log out/in for group membership to take effect.

### Issue: "DISCONNECTED" shown in web interface

**Solution**:
1. Restart the main application to load updated network code
2. Verify NetworkManager is running: `systemctl status NetworkManager`
3. Check network interface status: `nmcli device status`

### Issue: AP mode fails to start

**Solution**:
1. Ensure hostapd is installed: `sudo apt install hostapd`
2. Check for conflicting connections: `nmcli connection show`
3. Delete old AP connections: `nmcli connection delete prismatron-hotspot`
4. Retry AP mode activation

## Network API Endpoints

The web interface provides these network management endpoints:

- `GET /api/network/status` - Get current network status
- `GET /api/network/scan` - Scan for available WiFi networks
- `POST /api/network/connect` - Connect to WiFi network
- `POST /api/network/disconnect` - Disconnect from current network
- `POST /api/network/ap/enable` - Enable AP mode
- `POST /api/network/ap/disable` - Disable AP mode

## Testing Network Functionality

### Test Client Mode
```bash
# Scan for networks (may use cached results)
nmcli device wifi list

# Connect to a network
nmcli device wifi connect "SSID" password "PASSWORD"
```

### Test AP Mode
```bash
# Enable AP mode
nmcli connection up prismatron-hotspot

# Verify AP is running
nmcli device status
ip addr show wlP1p1s0
```

## Security Considerations

1. **AP Mode Security**: By default, AP mode has no password. To add security:
   - Edit `/etc/prismatron/network-config.json`
   - Add password to `ap_config` section
   - Restart network service

2. **NetworkManager Permissions**: The PolicyKit rule grants full NetworkManager access to the specified user. Restrict as needed for production.

3. **Web Interface**: Ensure proper authentication is implemented before exposing network controls in production.

## Troubleshooting Commands

```bash
# View network service logs
sudo journalctl -u prismatron-network -f

# Check NetworkManager status
systemctl status NetworkManager

# List network devices
nmcli device status

# Show WiFi networks (cached)
nmcli device wifi list

# View active connections
nmcli connection show --active

# Restart network service
sudo systemctl restart prismatron-network

# Check group membership
groups $USER

# Verify PolicyKit rules
ls -la /etc/polkit-1/rules.d/
```

## Development Notes

- The system gracefully handles permission issues by using cached WiFi scan results
- Interface auto-detection eliminates hardcoded interface names
- All network operations are async for non-blocking performance
- Logging provides detailed debugging information

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs: `journalctl -u prismatron-network`
3. Verify all installation steps were completed
4. Ensure NetworkManager is running and configured correctly

---

Last Updated: 2025-08-18
