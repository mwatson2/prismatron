# Network AP Fallback and Reboot Configuration

## Overview

This document explains how the Prismatron device manages network connectivity and system reboot functionality.

## Problem Statement

1. **AP Mode Priority Issue**: The device was entering WiFi AP mode on boot even when a known WiFi network (Mendacious) was available. This happened because NetworkManager's `autoconnect=yes` on the AP connection caused it to activate before client connections were attempted, despite having lower priority.

2. **Reboot Button Non-functional**: The web UI reboot button wasn't working because:
   - The frontend `handleReboot` was a placeholder showing an alert
   - Even after fixing the frontend, `sudo reboot` requires a password

## Solution

### 1. AP Fallback via NetworkManager Dispatcher

Instead of having the Prismatron application manage AP fallback (which could conflict with system-level network management), we delegate this to a NetworkManager dispatcher script.

**Script Location**: `/etc/NetworkManager/dispatcher.d/99-prismatron-ap-fallback`
**Source**: `/mnt/dev/prismatron/tools/99-prismatron-ap-fallback`

**How it works**:
1. NetworkManager runs dispatcher scripts on network events
2. On boot, when the loopback interface comes up (`lo` + `up`), the script triggers
3. It waits 30 seconds for NetworkManager to establish a WiFi client connection
4. If no client connection exists after 30s, it activates `prismatron-ap` connection
5. Runs completely independently of Prismatron service - provides network access even if the main service fails

**Installation**:
```bash
sudo cp /mnt/dev/prismatron/tools/99-prismatron-ap-fallback /etc/NetworkManager/dispatcher.d/
sudo chmod 755 /etc/NetworkManager/dispatcher.d/99-prismatron-ap-fallback
```

**Testing**:
```bash
# Simulate boot event
sudo /etc/NetworkManager/dispatcher.d/99-prismatron-ap-fallback lo up

# View logs
journalctl -t prismatron-ap-fallback
```

### 2. NetworkManager Connection Configuration

**AP Connection** (`prismatron-ap`):
- `autoconnect=no` - Prevents AP from activating before client connections are tried
- `autoconnect-priority=90` - Lower priority than client connections
- AP mode is only activated by the dispatcher script when no client connection exists

**Client Connection** (e.g., `Mendacious`):
- `autoconnect=yes` - Automatically connect on boot
- `autoconnect-priority=100` - Higher priority than AP

**View current settings**:
```bash
nmcli -t -f NAME,AUTOCONNECT,AUTOCONNECT-PRIORITY connection show
```

### 3. Code Changes in NetworkManager (`src/network/manager.py`)

- `enable_ap_mode(persist=False)` - Default to not setting autoconnect
- `ensure_connectivity()` renamed to `get_connectivity_status()` - Now just reports status, no fallback logic
- AP fallback responsibility moved entirely to system-level dispatcher script

### 4. Reboot Button Configuration

**Sudoers Rule** (`/etc/sudoers.d/prismatron-reboot`):
```
mark ALL=(ALL) NOPASSWD: /sbin/reboot, /sbin/shutdown
```

This allows the `mark` user to run `/sbin/reboot` without a password, which is required for the web UI reboot button to work.

**Installation**:
```bash
echo 'mark ALL=(ALL) NOPASSWD: /sbin/reboot, /sbin/shutdown' | sudo tee /etc/sudoers.d/prismatron-reboot
sudo chmod 440 /etc/sudoers.d/prismatron-reboot
```

**Code Path**:
1. User clicks "REBOOT" button in web UI (`SettingsPage.jsx`)
2. Frontend calls `POST /api/system/reboot`
3. API endpoint (`api_server.py`) calls `control_state.signal_reboot()`
4. Main process (`main.py`) detects reboot signal in monitoring loop
5. Main process calls `reboot_system()` which runs `sudo /sbin/reboot`

## Boot Sequence

1. System boots
2. NetworkManager starts and tries to connect to saved WiFi networks (Mendacious has highest priority)
3. Loopback interface comes up, triggering `99-prismatron-ap-fallback`
4. Script waits 30 seconds in background
5. After 30s:
   - If WiFi connected: script exits, no action needed
   - If no WiFi: script activates `prismatron-ap` connection
6. Prismatron service starts (independent of network fallback)
7. Service logs network status but does not manage AP fallback

## Key Files

| File | Purpose |
|------|---------|
| `/etc/NetworkManager/dispatcher.d/99-prismatron-ap-fallback` | System-level AP fallback script |
| `/etc/sudoers.d/prismatron-reboot` | Passwordless reboot permission |
| `tools/99-prismatron-ap-fallback` | Source copy of dispatcher script |
| `src/network/manager.py` | NetworkManager wrapper (simplified) |
| `src/web/api_server.py` | Reboot API endpoint |
| `src/web/frontend/src/pages/SettingsPage.jsx` | Reboot button UI |
| `main.py` | Reboot signal handling |
