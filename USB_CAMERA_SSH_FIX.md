# USB Camera SSH Access Fix

## Problem
USB camera captures black/empty images when running tools from SSH terminal, but works correctly from the directly connected display terminal.

## Root Cause
The SSH session is missing critical environment variables, particularly:
- `DISPLAY=:1` - Required for X11 display server access
- `XDG_SESSION_TYPE=x11` - Session type indicator
- Desktop environment variables

Many camera libraries (OpenCV, GStreamer) require X11 display server access for proper initialization, even when not displaying GUI windows.

## Solution

### Step 1: On the Physical Display Terminal
Run this command to allow X11 connections from local users:
```bash
xhost +local:
```

Or for better security, allow only specific user:
```bash
xhost +SI:localuser:mark
```

**Note**: This needs to be done while physically at the device, logged into the desktop session.

### Step 2: In Your SSH Session
Set the required environment variables:
```bash
export DISPLAY=:1
export XDG_SESSION_TYPE=x11
```

You can add these to your `~/.bashrc` or `~/.profile` for persistence:
```bash
echo 'export DISPLAY=:1' >> ~/.bashrc
echo 'export XDG_SESSION_TYPE=x11' >> ~/.bashrc
```

### Step 3: Test Camera Access
Test if the camera now works properly:
```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); print(f'Capture success: {ret}, Frame shape: {frame.shape if ret else None}'); cap.release()"
```

## Alternative Solutions

### Option 1: SSH with X11 Forwarding
Connect with X11 forwarding enabled:
```bash
ssh -X user@prismatron
# or for trusted forwarding:
ssh -Y user@prismatron
```

### Option 2: Virtual Display (Headless Operation)
Install and use Xvfb (X Virtual Framebuffer):
```bash
sudo apt-get install xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

### Option 3: Direct V4L2 Access
For pure capture without display requirements, use V4L2 directly:
```python
# Use V4L2 backend explicitly in OpenCV
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
```

## Important Notes

1. **Security**: `xhost +local:` allows any local user to access your X display. Use specific user permissions for better security.

2. **GUI Windows**: With `DISPLAY=:1` set in SSH, any GUI windows (like camera previews) will appear on the physical display, not in your SSH terminal.

3. **Interaction**: You cannot interact with windows opened on the remote display from SSH - they're display-only.

4. **Persistence**: The `xhost` command needs to be run each time the X server restarts (after reboot or re-login).

## Verification
After applying the fix, verify the environment:
```bash
echo $DISPLAY  # Should show :1
echo $XDG_SESSION_TYPE  # Should show x11 (optional but helpful)
```

Then test your camera tools:
```bash
cd /mnt/dev/prismatron
source env/bin/activate
python tools/led_gain_calibrator.py  # Or your specific tool
```

## Troubleshooting

If still experiencing issues:
1. Check X server is running: `ps aux | grep Xorg`
2. Verify display number: `ls /tmp/.X11-unix/`
3. Check camera permissions: `ls -l /dev/video*`
4. Test with simple capture: `v4l2-ctl --device=/dev/video0 --stream-mmap --stream-to=test.jpg --stream-count=1`
