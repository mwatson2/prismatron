# EGL Display Issues Debug & Fix Guide

## Problem Description

After system reboots, FFmpeg fails to probe video metadata with errors like:
```
libEGL warning: DRI2: failed to authenticate
ffprobe error (see stderr output for detail)
Failed to probe video metadata: ffprobe error
```

This prevents the Prismatron system from loading video content sources.

## Root Cause

The issue occurs when the xorg-dummy service is configured to use the `dummy` driver instead of the `nvidia` driver. The dummy driver does not provide proper EGL/OpenGL support needed for FFmpeg's hardware acceleration features.

## Diagnosis Steps

### 1. Check Service Status
```bash
sudo systemctl status xorg-dummy
```
Should show "active (running)".

### 2. Check Current Driver Configuration
```bash
cat /etc/X11/xorg-dummy.conf | grep -A 5 "Section \"Device\""
```
Look for the `Driver` line - if it shows `"dummy"`, that's the problem.

### 3. Test FFmpeg with Display Set
```bash
export DISPLAY=:99
ffprobe media/countdown.mp4 2>&1 | head -10
```
If you see "libEGL warning: DRI2: failed to authenticate", the dummy driver is the issue.

### 4. Verify NVIDIA GPU Available
```bash
nvidia-smi
```
Should show the Orin GPU is detected.

### 5. Check Prismatron Service Logs
```bash
journalctl --user -u prismatron -n 50
```
Look for video source errors.

## Fix Procedure

### Step 1: Update Xorg Configuration
Edit the xorg dummy configuration file:
```bash
sudo nano /etc/X11/xorg-dummy.conf
```

Change the Device section from:
```
Section "Device"
    Identifier     "Card0"
    Driver         "dummy"
    VendorName     "NVIDIA Corporation"
    VideoRam       256000
EndSection
```

To:
```
Section "Device"
    Identifier     "Card0"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    VideoRam       256000
    Option         "UseDisplayDevice" "none"
EndSection
```

### Step 2: Restart Xorg Service
```bash
sudo systemctl restart xorg-dummy
```

### Step 3: Verify Fix
```bash
# Check service is running
sudo systemctl status xorg-dummy

# Test FFmpeg
export DISPLAY=:99
ffprobe media/countdown.mp4 2>&1 | head -5
```

Should no longer show EGL authentication errors.

### Step 4: Restart Prismatron
```bash
systemctl --user restart prismatron
systemctl --user status prismatron
```

## Why This Fixes It

- **nvidia driver**: Provides proper EGL/OpenGL context for hardware acceleration
- **UseDisplayDevice "none"**: Maintains headless operation without requiring physical display
- **Hardware acceleration**: FFmpeg can properly probe video metadata and decode content

## Prevention

This issue typically occurs when:
1. System updates modify driver configurations
2. Someone manually changes the driver to "dummy" thinking it's more appropriate for headless
3. GPU driver updates reset configurations

To prevent recurrence:
1. Document this configuration requirement in system setup notes
2. Consider adding a systemd service check that validates the driver setting
3. Monitor system logs after updates for EGL-related errors

## Alternative Debugging Commands

If the above doesn't work, try these additional checks:

```bash
# Check X11 logs
tail -50 /var/log/Xorg-dummy.log

# Check for conflicting display processes
ps aux | grep -i xorg

# Verify EGL libraries
ls -la /usr/lib/aarch64-linux-gnu/dri/

# Test OpenGL context (if glxinfo available)
DISPLAY=:99 glxinfo | head -20
```

## Notes

- This is specific to NVIDIA Jetson systems with headless operation requirements
- The dummy driver is insufficient for multimedia applications requiring hardware acceleration
- Always use nvidia driver with "UseDisplayDevice none" for headless GPU access
- Password for sudo operations: `prism`
