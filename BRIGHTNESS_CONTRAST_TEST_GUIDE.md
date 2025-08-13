# Brightness/Contrast Camera Test Guide

## Overview
This test helps determine optimal camera brightness and contrast settings for capturing LED diffusion patterns. It systematically tests different brightness/contrast combinations while keeping gain and exposure fixed, measuring how each setting affects the visible diffusion area.

## Quick Start

### Basic Test Command
```bash
python tools/wled_test_patterns.py --led-count 3000 brightness-contrast \
  --index 1 --color 0 100 0 \
  --camera-config camera-0810.json
```

### Full Test with Custom Values
```bash
python tools/wled_test_patterns.py --led-count 3000 brightness-contrast \
  --index 1 --color 0 100 0 \
  --camera-config camera-0810.json \
  --brightness-values 50 100 150 200 \
  --contrast-values 50 100 150 200
```

## Parameters

- `--index`: LED index to light (0-based)
- `--color R G B`: RGB values (use 0 100 0 for moderate green to avoid saturation)
- `--camera-config`: Camera configuration JSON file
- `--brightness-values`: List of brightness values to test (0-255, default: 64 128 192)
- `--contrast-values`: List of contrast values to test (0-255, default: 64 128 192)
- `--output-dir`: Directory for results (default: brightness_contrast)

## Output

### Image Files
- **Location**: `/mnt/dev/prismatron/brightness_contrast/`
- **Naming**: `B{brightness}_C{contrast}.jpg` (e.g., `B150_C200.jpg`)
- **Grid**: Creates N×M images for N brightness × M contrast values

### Each Image Shows
- Settings label (B=brightness, C=contrast)
- Visible pixel count (diffusion area measurement)
- ROI mean RGB values
- ROI peak RGB values
- Green rectangle marking LED region
- SATURATED warning if peak ≥ 254

### Console Report
The tool prints a summary table showing:
- Visible pixels for each setting combination
- Mean/peak green channel values
- Saturation status
- Optimal configuration recommendation

## Key Findings from Investigation

### Camera Control Ranges
- **Gain**: 0-255 (not just 10-200)
- **Exposure**: 3-2047 (minimum is 3)
- **Brightness**: 0-255 (default 128)
- **Contrast**: 0-255 (default 128)

### Why Brightness/Contrast Matter
- **Brightness**: Shifts overall image luminance, reveals faint diffusion edges
- **Contrast**: Amplifies differences between LED and background
- These settings explain the "larger diffusion patterns" seen with auto-adaptation

### Recommended Settings
Based on testing, optimal settings for maximum diffusion visibility:
- **Brightness**: 150-200 (reveals faint edges)
- **Contrast**: 150-200 (enhances edge detection)
- **Gain**: 20-40 (moderate, avoids noise)
- **Exposure**: 100-250 (moderate sensitivity)
- **LED brightness**: 80-120 (not 255, to avoid saturation)

## Important Notes

1. **Camera Requirements**: Requires USB camera at `/dev/video0`
2. **v4l2-ctl**: Uses v4l2-ctl for reliable camera control
3. **All Auto Features Disabled**: Disables auto-exposure, auto-white-balance, auto-focus
4. **Fixed Base Settings**: Keeps gain=20, exposure=100 constant during test
5. **LED Intensity**: Use moderate LED brightness (100/255) to avoid saturation

## Troubleshooting

### Camera Not Found
- Check USB camera is connected
- Verify camera appears at `/dev/video0`
- May need to reboot if camera was disconnected

### Saturation Issues
- Reduce LED brightness (use 80-100 instead of 255)
- Lower brightness setting below 150
- Reduce gain if still saturating

### No Diffusion Visible
- Increase brightness to 150-200
- Increase contrast to 150-200
- Check LED is actually lit
- Verify camera focus and positioning

## Example Workflow

1. Light a single LED at moderate brightness:
   ```bash
   python tools/wled_test_patterns.py --led-count 3000 single --index 1 --color 0 100 0
   ```

2. Run brightness/contrast test:
   ```bash
   python tools/wled_test_patterns.py --led-count 3000 brightness-contrast \
     --index 1 --color 0 100 0 --camera-config camera-0810.json
   ```

3. Review images in `brightness_contrast/` directory

4. Identify configuration with:
   - Largest visible pixel count
   - No saturation (peak < 254)
   - Good contrast between LED and background

5. Use optimal settings for diffusion pattern capture

## Implementation Details

The test is implemented in `/mnt/dev/prismatron/tools/wled_test_patterns.py`:
- `run_brightness_contrast_test()`: Main test method
- `configure_camera_with_v4l2()`: Camera configuration with v4l2-ctl
- `_generate_brightness_contrast_report()`: Analysis and reporting

This functionality was added to the existing WLED test patterns tool to leverage its proven camera capture infrastructure.
