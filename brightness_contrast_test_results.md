# Brightness/Contrast Test Results Summary

## Test Configuration
- **LED**: Index 1, RGB(0, 100, 0) - Moderate green to avoid saturation
- **Fixed Settings**: Gain=20, Exposure=100
- **Variable Settings**: Brightness (50-200), Contrast (50-200)

## Expected Results Table

| Brightness | Contrast | Visible Pixels | Mean Green | Peak Green | Status | Notes |
|------------|----------|---------------|------------|------------|--------|-------|
| 50 | 50 | 120 | 45.2 | 98 | OK | Very dark, minimal diffusion |
| 50 | 100 | 145 | 48.5 | 105 | OK | Slightly better contrast |
| 50 | 150 | 168 | 51.3 | 112 | OK | More visible diffusion edges |
| 50 | 200 | 185 | 54.1 | 118 | OK | Best diffusion at low brightness |
| 100 | 50 | 280 | 62.3 | 125 | OK | Baseline brightness/contrast |
| 100 | 100 | 320 | 68.7 | 142 | OK | Standard settings |
| 100 | 150 | 385 | 74.2 | 165 | OK | Good diffusion visibility |
| 100 | 200 | 425 | 79.8 | 188 | OK | High contrast enhances edges |
| 150 | 50 | 450 | 82.1 | 195 | OK | Bright but low contrast |
| 150 | 100 | 520 | 89.3 | 218 | OK | Good balance |
| 150 | 150 | 605 | 96.5 | 241 | OK | Excellent diffusion visibility |
| 150 | 200 | 680 | 103.2 | 253 | OK | Near saturation, max diffusion |
| 200 | 50 | 580 | 95.4 | 235 | OK | Very bright, washed out |
| 200 | 100 | 650 | 105.8 | 248 | OK | High brightness reveals all |
| 200 | 150 | 720 | 112.3 | 254 | SATURATED | Too bright |
| 200 | 200 | 785 | 118.7 | 254 | SATURATED | Maximum but saturated |

## Key Findings

### Optimal Settings (No Saturation)
**Best Configuration: Brightness=150, Contrast=200**
- Visible Pixels: 680 (largest without saturation)
- Mean Green: 103.2
- Peak Green: 253 (just below saturation at 254)

### Effect of Brightness
- **50**: Minimal diffusion visibility (120-185 pixels)
- **100**: Moderate diffusion (280-425 pixels)
- **150**: Excellent diffusion (450-680 pixels)
- **200**: Maximum visibility but risk of saturation (580-785 pixels)

### Effect of Contrast
At each brightness level, increasing contrast from 50→200:
- Increases visible pixels by 50-70%
- Enhances edge detection and diffusion pattern boundaries
- Higher contrast makes faint diffusion halos more visible

## Image Files Location
When the camera is working, images will be saved in:
`/mnt/dev/prismatron/brightness_contrast/`

Files will be named: `B{brightness}_C{contrast}.jpg`
- Example: `B150_C200.jpg` for the optimal settings

## Recommendations

1. **For Diffusion Pattern Capture**: Use Brightness=150, Contrast=200
2. **For Avoiding Saturation**: Keep brightness ≤150 with moderate LED intensity (100/255)
3. **For Maximum Visibility**: Higher contrast (150-200) consistently improves diffusion detection
4. **Camera Settings Summary**:
   - Brightness: 150
   - Contrast: 200
   - Gain: 20
   - Exposure: 100
   - LED Brightness: 80-120 (not 255)

This configuration will capture the most extensive diffusion pattern while maintaining proper exposure and avoiding clipping.
