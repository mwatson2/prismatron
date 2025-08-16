# Prismatron Systemd Service Implementation Plan

## Overview
This document outlines the complete plan for running Prismatron as a systemd service with proper security, automatic startup, and managed restart/reboot capabilities.

## Goals
- ✅ Automatic startup on system boot
- ✅ Run with minimal required permissions (principle of least privilege)
- ✅ Clean application restart without sudo
- ✅ Controlled system reboot capability
- ✅ Proper process management and logging
- ✅ Graceful shutdown and cleanup

## Architecture Design

### 1. Service Structure

```
prismatron.service (main service)
├── Manages all Prismatron processes
├── Runs as dedicated 'prismatron' user
├── Auto-restarts on failure
└── Starts after network is ready

prismatron-admin.socket (optional)
└── Privileged operations endpoint
    └── Handles system reboot requests
```

### 2. User and Permissions Model

#### System User Creation
```bash
# Create dedicated system user (no login shell)
sudo useradd -r -s /bin/false -d /mnt/dev/prismatron -m prismatron

# Add to necessary hardware groups (Jetson-specific)
sudo usermod -a -G video,gpio,i2c,dialout prismatron
```

#### Directory Permissions
```
/mnt/dev/prismatron/
├── [prismatron:prismatron 755] src/           # Application code (read/execute)
├── [prismatron:prismatron 755] env/           # Python virtual environment
├── [prismatron:prismatron 775] uploads/       # User uploads (read/write)
├── [prismatron:prismatron 775] playlists/     # Playlist files (read/write)
├── [prismatron:prismatron 775] media/         # Media files (read/write)
├── [prismatron:prismatron 775] logs/          # Application logs
└── [prismatron:prismatron 644] config/        # Configuration files
```

### 3. Systemd Service Configuration

#### Main Service File: `/etc/systemd/system/prismatron.service`

```ini
[Unit]
Description=Prismatron LED Display System
Documentation=https://github.com/yourusername/prismatron
After=network-online.target
Wants=network-online.target
# If using WLED, ensure network is truly ready
After=systemd-networkd-wait-online.service

[Service]
Type=forking
PIDFile=/run/prismatron/prismatron.pid
User=prismatron
Group=prismatron
WorkingDirectory=/mnt/dev/prismatron

# Environment setup
Environment="PYTHONPATH=/mnt/dev/prismatron"
Environment="PATH=/mnt/dev/prismatron/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="DISPLAY=:0"  # If display output needed

# Pre-start cleanup (remove stale shared memory)
ExecStartPre=/bin/bash -c 'rm -f /dev/shm/prismatron_*'

# Main execution
ExecStart=/mnt/dev/prismatron/env/bin/python /mnt/dev/prismatron/main.py --daemon

# Graceful shutdown
ExecStop=/bin/kill -TERM $MAINPID
TimeoutStopSec=30
KillMode=mixed

# Restart policy
Restart=on-failure
RestartSec=10
StartLimitInterval=600
StartLimitBurst=3

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

# Allow writing to necessary directories
ReadWritePaths=/mnt/dev/prismatron/uploads
ReadWritePaths=/mnt/dev/prismatron/playlists
ReadWritePaths=/mnt/dev/prismatron/media
ReadWritePaths=/mnt/dev/prismatron/logs
ReadWritePaths=/mnt/dev/prismatron/thumbnails
ReadWritePaths=/run/prismatron
ReadWritePaths=/dev/shm

# Hardware access (for Jetson/LED control)
DeviceAllow=/dev/i2c-* rw
DeviceAllow=/dev/gpiochip* rw
DeviceAllow=/dev/video* rw
SupplementaryGroups=video gpio i2c

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096
MemoryMax=4G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
```

#### Runtime Directory Configuration: `/etc/tmpfiles.d/prismatron.conf`

```
d /run/prismatron 0755 prismatron prismatron -
```

### 4. Restart and Reboot Implementation

#### Option A: Sudo Configuration (`/etc/sudoers.d/prismatron`)

```bash
# Allow prismatron user to restart its own service and reboot
prismatron ALL=(root) NOPASSWD: /bin/systemctl restart prismatron.service
prismatron ALL=(root) NOPASSWD: /bin/systemctl reboot
prismatron ALL=(root) NOPASSWD: /bin/systemctl poweroff
```

#### Option B: Polkit Rules (`/etc/polkit-1/rules.d/50-prismatron.rules`)

```javascript
polkit.addRule(function(action, subject) {
    if (action.id == "org.freedesktop.systemd1.manage-units" &&
        action.lookup("unit") == "prismatron.service" &&
        subject.user == "prismatron") {
        return polkit.Result.YES;
    }
});

polkit.addRule(function(action, subject) {
    if ((action.id == "org.freedesktop.login1.reboot" ||
         action.id == "org.freedesktop.login1.power-off") &&
        subject.user == "prismatron") {
        return polkit.Result.YES;
    }
});
```

### 5. Application Code Changes

#### Main Application Entry Point (`main.py`)

```python
#!/usr/bin/env python3
"""
Prismatron main entry point with systemd integration.
"""
import os
import sys
import signal
import asyncio
import argparse
import logging
from pathlib import Path

# Add proper PID file handling for systemd
def write_pid_file(pid_file: Path):
    """Write PID file for systemd tracking."""
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

def cleanup_pid_file(pid_file: Path):
    """Remove PID file on exit."""
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass

def setup_signal_handlers():
    """Setup graceful shutdown handlers."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # Trigger graceful shutdown of all processes
        asyncio.create_task(shutdown_application())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

async def shutdown_application():
    """Gracefully shutdown all components."""
    # Stop producer
    # Stop consumer  
    # Save state
    # Cleanup resources
    # Exit
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--daemon', action='store_true',
                        help='Run as daemon (for systemd)')
    args = parser.parse_args()

    if args.daemon:
        # Daemonize for systemd Type=forking
        pid = os.fork()
        if pid > 0:
            sys.exit(0)

        # Setup new session
        os.setsid()

        # Write PID file
        pid_file = Path('/run/prismatron/prismatron.pid')
        write_pid_file(pid_file)

        # Setup cleanup
        import atexit
        atexit.register(cleanup_pid_file, pid_file)

    # Continue with normal startup
    # ... existing startup code ...
```

#### API Endpoints (`src/web/api_server.py`)

```python
import subprocess
import asyncio
from fastapi import HTTPException

@app.post("/api/system/restart")
async def restart_application():
    """
    Restart the Prismatron application via systemd.
    Requires prismatron user to have sudo permission for this command.
    """
    try:
        # Log who requested the restart
        logger.warning(f"Application restart requested via API")

        # Schedule restart after response is sent
        asyncio.create_task(perform_restart())

        return {
            "status": "accepted",
            "message": "Application restart initiated. System will be back online in approximately 15 seconds."
        }
    except Exception as e:
        logger.error(f"Failed to initiate restart: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate restart")

async def perform_restart():
    """Execute restart after delay to allow response to be sent."""
    await asyncio.sleep(2)

    try:
        # Method 1: If we have sudo permission (configured in sudoers)
        result = subprocess.run(
            ["sudo", "/bin/systemctl", "restart", "prismatron.service"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            logger.error(f"Restart command failed: {result.stderr}")
            # Method 2: Try to exit with special code that systemd will restart
            os._exit(123)  # Special exit code for restart
    except Exception as e:
        logger.error(f"Restart failed: {e}")
        # Last resort: exit and let systemd restart us
        os._exit(1)

@app.post("/api/system/reboot")
async def reboot_system():
    """
    Reboot the entire system.
    Requires prismatron user to have sudo permission for reboot.
    """
    try:
        # Log who requested the reboot
        logger.warning(f"System reboot requested via API")

        # Schedule reboot after response is sent
        asyncio.create_task(perform_reboot())

        return {
            "status": "accepted",
            "message": "System reboot initiated. The system will restart in 5 seconds."
        }
    except Exception as e:
        logger.error(f"Failed to initiate reboot: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate reboot")

async def perform_reboot():
    """Execute system reboot after delay."""
    # Give time for response and any cleanup
    await asyncio.sleep(5)

    try:
        # Save any critical state
        logger.info("Saving state before reboot...")
        # ... save state code ...

        # Execute reboot
        subprocess.run(
            ["sudo", "/bin/systemctl", "reboot"],
            timeout=5
        )
    except Exception as e:
        logger.error(f"Reboot command failed: {e}")
```

### 6. Installation Process

#### Installation Script (`scripts/install_service.sh`)

```bash
#!/bin/bash
set -e

INSTALL_DIR="/mnt/dev/prismatron"
SERVICE_USER="prismatron"
SERVICE_FILE="/etc/systemd/system/prismatron.service"

echo "=== Prismatron Service Installation ==="

# 1. Create system user
if ! id "$SERVICE_USER" &>/dev/null; then
    echo "Creating system user: $SERVICE_USER"
    sudo useradd -r -s /bin/false -d "$INSTALL_DIR" -m "$SERVICE_USER"
    sudo usermod -a -G video,gpio,i2c,dialout "$SERVICE_USER"
else
    echo "User $SERVICE_USER already exists"
fi

# 2. Set directory permissions
echo "Setting directory permissions..."
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
sudo chmod 755 "$INSTALL_DIR"
sudo chmod -R 775 "$INSTALL_DIR"/{uploads,playlists,media,logs,thumbnails} 2>/dev/null || true

# 3. Create required directories
echo "Creating runtime directories..."
sudo mkdir -p /run/prismatron
sudo chown "$SERVICE_USER:$SERVICE_USER" /run/prismatron

# 4. Install systemd service
echo "Installing systemd service..."
sudo cp "$INSTALL_DIR/scripts/prismatron.service" "$SERVICE_FILE"

# 5. Install sudoers configuration
echo "Configuring sudo permissions..."
cat <<EOF | sudo tee /etc/sudoers.d/prismatron
$SERVICE_USER ALL=(root) NOPASSWD: /bin/systemctl restart prismatron.service
$SERVICE_USER ALL=(root) NOPASSWD: /bin/systemctl reboot
EOF
sudo chmod 440 /etc/sudoers.d/prismatron

# 6. Create tmpfiles configuration
echo "Creating tmpfiles configuration..."
echo "d /run/prismatron 0755 $SERVICE_USER $SERVICE_USER -" | \
    sudo tee /etc/tmpfiles.d/prismatron.conf

# 7. Reload systemd
echo "Reloading systemd configuration..."
sudo systemctl daemon-reload

# 8. Enable service
echo "Enabling service for automatic startup..."
sudo systemctl enable prismatron.service

echo "=== Installation Complete ==="
echo ""
echo "To start the service now:"
echo "  sudo systemctl start prismatron"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u prismatron -f"
echo ""
echo "To check status:"
echo "  sudo systemctl status prismatron"
```

### 7. Frontend Integration

#### Settings Page Updates (`src/web/frontend/src/pages/SettingsPage.jsx`)

```javascript
// Add confirmation dialogs for dangerous operations
const handleRestart = async () => {
  if (!confirm('Are you sure you want to restart the application? The display will be temporarily interrupted.')) {
    return;
  }

  try {
    const response = await fetch('/api/system/restart', { method: 'POST' });
    const data = await response.json();

    // Show notification
    showNotification(data.message, 'warning');

    // Start polling for reconnection
    setTimeout(() => {
      pollForReconnection();
    }, 5000);
  } catch (error) {
    showNotification('Failed to restart application', 'error');
  }
};

const handleReboot = async () => {
  if (!confirm('Are you sure you want to reboot the system? This will take several minutes.')) {
    return;
  }

  if (!confirm('This will REBOOT THE ENTIRE SYSTEM. Are you absolutely sure?')) {
    return;
  }

  try {
    const response = await fetch('/api/system/reboot', { method: 'POST' });
    const data = await response.json();

    // Show notification
    showNotification(data.message, 'warning');

    // Disable UI
    setSystemRebooting(true);
  } catch (error) {
    showNotification('Failed to reboot system', 'error');
  }
};
```

### 8. Monitoring and Logging

#### Log Rotation (`/etc/logrotate.d/prismatron`)

```
/mnt/dev/prismatron/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 prismatron prismatron
    sharedscripts
    postrotate
        systemctl reload prismatron 2>/dev/null || true
    endscript
}
```

#### Health Check Script (`scripts/health_check.sh`)

```bash
#!/bin/bash
# Health check for monitoring systems

# Check if service is active
if ! systemctl is-active --quiet prismatron; then
    echo "ERROR: Prismatron service is not running"
    exit 1
fi

# Check API endpoint
if ! curl -sf http://localhost:8080/api/status > /dev/null; then
    echo "ERROR: API is not responding"
    exit 2
fi

echo "OK: Prismatron is healthy"
exit 0
```

### 9. Testing Plan

#### Pre-deployment Testing
1. Test service start/stop/restart
2. Test automatic restart on crash
3. Test reboot functionality
4. Verify permissions and security restrictions
5. Test resource limits
6. Verify logging

#### Test Commands
```bash
# Start service
sudo systemctl start prismatron

# Check status
sudo systemctl status prismatron

# View logs
sudo journalctl -u prismatron -f

# Test restart
sudo systemctl restart prismatron

# Simulate crash
sudo kill -9 $(cat /run/prismatron/prismatron.pid)
# Should auto-restart within 10 seconds

# Test as prismatron user
sudo -u prismatron /bin/systemctl restart prismatron.service

# Check resource usage
systemctl show prismatron -p MemoryCurrent
systemctl show prismatron -p CPUUsageNSec
```

### 10. Rollback Plan

If issues occur:

```bash
# Stop and disable service
sudo systemctl stop prismatron
sudo systemctl disable prismatron

# Remove service files
sudo rm /etc/systemd/system/prismatron.service
sudo rm /etc/sudoers.d/prismatron
sudo rm /etc/tmpfiles.d/prismatron.conf

# Reload systemd
sudo systemctl daemon-reload

# Return to manual operation
cd /mnt/dev/prismatron
source env/bin/activate
python main.py
```

## Security Considerations

### Implemented Security Measures
- ✅ Runs as non-root user
- ✅ Minimal sudo permissions (only restart/reboot)
- ✅ Protected system directories
- ✅ Private /tmp
- ✅ Resource limits
- ✅ No new privileges
- ✅ Kernel protection

### Remaining Risks
- ⚠️ Reboot capability could be abused
- ⚠️ No authentication on restart/reboot endpoints (to be added)
- ⚠️ Service user has hardware access (required for functionality)

## Future Enhancements

1. **Authentication/Authorization**
   - Add JWT tokens for API access
   - Require admin role for system operations
   - Add rate limiting on dangerous endpoints

2. **Watchdog Integration**
   - Implement systemd watchdog
   - Periodic health checks
   - Automatic recovery actions

3. **Multi-instance Support**
   - Template service for multiple displays
   - Instance-specific configurations
   - Centralized management

4. **Backup and Recovery**
   - Automatic state backups
   - Configuration snapshots
   - Restore points before updates

## Implementation Checklist

- [ ] Create prismatron system user
- [ ] Set up directory permissions
- [ ] Create systemd service file
- [ ] Configure sudo permissions
- [ ] Update main.py for daemon mode
- [ ] Add restart/reboot API endpoints
- [ ] Update frontend with system controls
- [ ] Create installation script
- [ ] Test service operation
- [ ] Test restart functionality
- [ ] Test reboot functionality
- [ ] Document for operations team
- [ ] Create monitoring alerts
- [ ] Deploy to production

## Notes

- The service will need adjustment based on actual hardware (Jetson Orin Nano specifics)
- Network startup timing may need tuning for WLED communication
- Consider adding a "safe mode" that starts without auto-playing content
- May need to adjust memory/CPU limits based on actual usage patterns
