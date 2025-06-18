# Prismatron Tools

This directory contains utilities and tools for the Prismatron LED display system, including diffusion pattern capture, camera calibration, visualization, and optimization tools.

## wled_test_patterns.py

Interactive test program for sending LED patterns to WLED controllers over UDP. Supports various patterns including solid colors, rainbow effects, and animated patterns.

### Prerequisites

1. Activate the virtual environment:
   ```bash
   source env/bin/activate
   ```

2. Ensure your WLED controller is connected and accessible on the network.

### Usage

#### Test Connection
First, verify connectivity to your WLED controller:
```bash
python tools/wled_test_patterns.py test --host wled.local
```

#### Solid Color Patterns
Set all LEDs to a specific color:
```bash
# Red LEDs for 5 seconds
python tools/wled_test_patterns.py solid --color 255 0 0 --duration 5

# Blue LEDs indefinitely (Ctrl+C to stop)
python tools/wled_test_patterns.py solid --color 0 0 255

# White LEDs with custom WLED IP
python tools/wled_test_patterns.py solid --color 255 255 255 --host 192.168.1.100
```

#### Rainbow Cycle
All LEDs cycle through rainbow colors together:
```bash
# Slow rainbow cycle (1 cycle per second)
python tools/wled_test_patterns.py rainbow-cycle --speed 1.0

# Fast rainbow cycle (3 cycles per second) for 30 seconds
python tools/wled_test_patterns.py rainbow-cycle --speed 3.0 --duration 30
```

#### Animated Rainbow
Creates a moving rainbow pattern across the LED array:
```bash
# Standard animated rainbow
python tools/wled_test_patterns.py animated-rainbow --speed 1.0

# Compressed rainbow (30% of array width) moving fast
python tools/wled_test_patterns.py animated-rainbow --speed 2.0 --width 0.3

# Wide, slow rainbow for 60 seconds
python tools/wled_test_patterns.py animated-rainbow --speed 0.5 --width 1.5 --duration 60
```

#### Wave Pattern
Sine wave pattern across the LED array:
```bash
# Single wave moving across array
python tools/wled_test_patterns.py wave --speed 1.0 --frequency 1.0

# Multiple waves
python tools/wled_test_patterns.py wave --speed 0.5 --frequency 3.0
```

### Configuration Options

- `--host HOST`: WLED controller hostname or IP address (default: wled.local)
- `--port PORT`: UDP port for DDP protocol (default: 21324)
- `--led-count COUNT`: Number of LEDs in your array (default: 3200)
- `--duration SECONDS`: How long to run the pattern (default: run until interrupted)
- `--verbose`: Show detailed connection and status information
- `--persistent-retry`: Keep retrying connection until successful (useful for startup)
- `--retry-interval SECONDS`: Time between connection retries (default: 10.0)

### Pattern Parameters

#### Rainbow Cycle
- `--speed`: Cycles per second (higher = faster color changes)

#### Animated Rainbow  
- `--speed`: Animation speed in Hz (how fast the rainbow moves)
- `--width`: Rainbow width as fraction of LED array (0.1 = 10% of array, 2.0 = rainbow repeats twice)

#### Wave Pattern
- `--speed`: Wave animation speed
- `--frequency`: Number of wave cycles across the LED array

### Examples for Different Use Cases

#### Hardware Testing
```bash
# Quick connectivity test with WLED status
python tools/wled_test_patterns.py test --verbose

# Test all LEDs with bright white
python tools/wled_test_patterns.py solid --color 255 255 255 --duration 5

# Test RGB channels individually
python tools/wled_test_patterns.py solid --color 255 0 0 --duration 3  # Red
python tools/wled_test_patterns.py solid --color 0 255 0 --duration 3  # Green  
python tools/wled_test_patterns.py solid --color 0 0 255 --duration 3  # Blue
```

#### Visual Effects Testing
```bash
# Slow rainbow for evaluation
python tools/wled_test_patterns.py rainbow-cycle --speed 0.5 --duration 30

# Fast animated rainbow
python tools/wled_test_patterns.py animated-rainbow --speed 2.0 --width 0.5 --duration 60

# Subtle wave effect
python tools/wled_test_patterns.py wave --speed 0.2 --frequency 2.0 --duration 45
```

#### Network Performance Testing
```bash
# High update rate test (30 FPS)
python tools/wled_test_patterns.py animated-rainbow --speed 3.0 --verbose

# Monitor transmission statistics
python tools/wled_test_patterns.py rainbow-cycle --speed 2.0 --duration 60 --verbose
```

#### Startup and System Integration
```bash
# Wait for WLED controller to come online (useful for boot scripts)
python tools/wled_test_patterns.py --persistent-retry --retry-interval 5 test

# Start patterns immediately when WLED becomes available
python tools/wled_test_patterns.py --persistent-retry rainbow-cycle --speed 1.0
```

### Troubleshooting

1. **Connection Failed**: Check that your WLED controller is powered on and connected to the network
2. **Timeout Errors**: Try increasing `--duration` or reducing `--speed` for network issues
3. **Pattern Not Visible**: Verify `--led-count` matches your actual LED configuration
4. **Performance Issues**: Use `--verbose` to monitor transmission statistics and error rates

### Technical Notes

- The program uses the DDP (Distributed Display Protocol) over UDP
- Query packets request JSON status from WLED (Destination ID 251)
- Query responses are parsed with proper header validation and data length checking
- WLED status includes controller name, version, LED count, and network info
- Supports large JSON responses (up to 8KB) with packet completeness validation
- Target frame rate is 30 FPS for animated patterns
- Flow control prevents overwhelming the WLED controller
- Persistent retry mode useful for system startup when network timing is uncertain
- All patterns support graceful shutdown with Ctrl+C
- Color calculations use HSV color space for smooth rainbow transitions

## capture_diffusion_patterns.py

Tool for capturing real diffusion patterns from the physical LED display. Cycles through each LED and color channel, capturing camera images to build a complete diffusion pattern database.

### Prerequisites

1. Camera connected and configured
2. WLED controller accessible on network  
3. LEDs properly mounted in display
4. Activate virtual environment: `source env/bin/activate`

### Usage

#### Basic Capture
```bash
# Capture patterns using default camera
python tools/capture_diffusion_patterns.py --wled-host 192.168.1.100 --output patterns.npz

# With live preview
python tools/capture_diffusion_patterns.py --wled-host wled.local --output patterns.npz --preview
```

#### Advanced Options
```bash
# Specific camera device and capture rate
python tools/capture_diffusion_patterns.py \
    --wled-host 192.168.1.100 \
    --camera-device 1 \
    --capture-fps 15.0 \
    --output high_res_patterns.npz \
    --preview

# With camera crop region (x, y, width, height)
python tools/capture_diffusion_patterns.py \
    --wled-host 192.168.1.100 \
    --camera-device 0 \
    --crop-region 100 50 800 480 \
    --output cropped_patterns.npz
```

### Configuration Options

- `--wled-host HOST`: WLED controller hostname/IP (required)
- `--wled-port PORT`: WLED controller port (default: 21324)
- `--camera-device ID`: Camera device ID (default: 0)
- `--output FILE`: Output patterns file (.npz format, required)
- `--capture-fps RATE`: Capture rate in fps (default: 10.0)
- `--preview`: Show live preview during capture
- `--crop-region X Y W H`: Camera crop region coordinates
- `--log-level LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Capture Process

1. **Initialization**: Connects to WLED and camera
2. **Calibration**: Sets camera exposure and gain for LED capture
3. **Capture Loop**: For each LED (0 to 3199) and channel (R, G, B):
   - Sets single LED/channel to full brightness
   - Waits for LED to stabilize  
   - Captures camera frame (800x480)
   - Stores normalized diffusion pattern
4. **Storage**: Saves all patterns in compressed .npz format

### Output Format

The output .npz file contains:
- `diffusion_patterns`: (3200, 3, 480, 800) uint8 array (0-255 RGB values)
- `metadata`: Dictionary with capture parameters and timestamps

Estimated capture time: ~16 minutes at 10fps (9600 total captures)
Estimated file size: ~900MB compressed (using uint8 instead of float32)

## camera_calibration.py

Interactive tool for calibrating camera position and crop region for diffusion pattern capture. Provides visual guides for selecting the 5:3 aspect ratio region corresponding to the Prismatron display.

### Usage

#### Basic Calibration
```bash
# Start calibration with default camera
python tools/camera_calibration.py

# Save configuration to file
python tools/camera_calibration.py --output-config calibration.json
```

#### Advanced Options
```bash
# Specific camera device
python tools/camera_calibration.py --camera-device 1 --output-config cam1_calibration.json
```

### Interactive Controls

- **Click & Drag**: Select crop region
- **g**: Toggle grid overlay
- **a**: Toggle 5:3 aspect ratio guides  
- **h**: Toggle help display
- **r**: Reset selection
- **s**: Save configuration and exit
- **q**: Quit without saving

### Visual Guides

- **Yellow Rectangle**: Maximum 5:3 ratio guide for camera frame
- **Grid Lines**: Rule of thirds alignment guides
- **Red Rectangle**: Selected crop region (auto-adjusted to 5:3)
- **Preview Window**: Real-time preview of cropped region

### Output Configuration

The calibration saves a JSON file with:
```json
{
  "camera_device": 0,
  "camera_resolution": {"width": 1920, "height": 1080},
  "crop_region": {"x": 260, "y": 140, "width": 1400, "height": 840},
  "target_resolution": {"width": 800, "height": 480},
  "aspect_ratio": {"target": 1.667, "actual": 1.667},
  "calibration_timestamp": 1234567890.123
}
```

This configuration can be used with the capture tool:
```bash
# Use calibrated crop region  
python tools/capture_diffusion_patterns.py \
    --wled-host 192.168.1.100 \
    --crop-region 260 140 1400 840 \
    --output patterns.npz
```

## visualize_diffusion_patterns.py

Web-based visualization tool for diffusion patterns. Creates an interactive interface to browse and analyze both captured and synthetic diffusion patterns.

### Usage

#### View Captured Patterns
```bash
# Start web server with captured patterns
python tools/visualize_diffusion_patterns.py --patterns captured_patterns.npz --host 0.0.0.0 --port 8080
```

#### View Synthetic Patterns
```bash
# Generate and view synthetic patterns
python tools/visualize_diffusion_patterns.py --host 127.0.0.1 --port 8080
```

### Web Interface Features

- **Grid View**: Thumbnail grid of all LED patterns
- **Channel Filtering**: View RGB composite or individual R/G/B channels
- **Pagination**: Navigate through large pattern sets (25/50/100 per page)
- **Pattern Detail**: Click any pattern for full-resolution view
- **Statistics**: Intensity statistics and center-of-mass calculations
- **Metadata Display**: Capture parameters and pattern information

### Navigation

1. Open browser to `http://localhost:8080`
2. Use channel selector to filter by color
3. Use pagination controls to browse patterns
4. Click any pattern thumbnail for detailed view
5. Modal shows individual channel views and statistics

### Configuration Options

- `--patterns FILE`: Captured patterns file (.npz)
- `--host HOST`: Web server host (default: 127.0.0.1)
- `--port PORT`: Web server port (default: 8080)
- `--no-synthetic`: Disable synthetic pattern generation
- `--debug`: Enable Flask debug mode

### Pattern Statistics

For each LED/channel combination:
- **Max Intensity**: Peak brightness value
- **Mean Intensity**: Average brightness across pattern
- **Standard Deviation**: Intensity variation measure
- **Center of Mass**: Calculated light distribution center

## standalone_optimizer.py

Standalone optimization tool that accepts input images and optimizes LED values using diffusion patterns. Renders the optimization result by summing weighted diffusion patterns.

### Usage

#### With Captured Patterns
```bash
# Basic optimization
python tools/standalone_optimizer.py \
    --input photo.jpg \
    --patterns captured_patterns.npz \
    --output result.png

# With preview and data saving
python tools/standalone_optimizer.py \
    --input landscape.jpg \
    --patterns captured_patterns.npz \
    --output optimized.png \
    --preview \
    --save-data
```

#### With Synthetic Patterns
```bash
# Use synthetic patterns
python tools/standalone_optimizer.py \
    --input artwork.png \
    --synthetic \
    --output result.png \
    --preview
```

### Configuration Options

- `--input FILE`: Input image path (required)
- `--output FILE`: Output image path (required)
- `--patterns FILE`: Diffusion patterns file (.npz)
- `--synthetic`: Use synthetic patterns if no patterns file
- `--method METHOD`: Optimization method (least_squares, gradient_descent)
- `--max-iterations N`: Maximum optimization iterations (default: 100)
- `--preview`: Show before/after comparison
- `--save-data`: Save LED values and optimization statistics
- `--log-level LEVEL`: Logging level

### Optimization Process

1. **Input Processing**: Loads and scales image to 800x480 with 5:3 aspect ratio
2. **Pattern Loading**: Loads captured patterns or generates synthetic ones
3. **LED Optimization**: Solves for optimal LED brightness values
4. **Result Rendering**: Sums weighted diffusion patterns to create result
5. **Output**: Saves optimized image and optional data files

### Preview Mode

When `--preview` is enabled:
- Shows side-by-side comparison of original and optimized images
- Displays optimization statistics overlay
- Waits for keypress before saving

### Output Files

- **Image File**: Primary optimization result (PNG/JPG)
- **Data File** (with `--save-data`): NPZ file containing:
  - `result_image`: Full resolution result
  - `led_values`: Optimized LED brightness values (3200x3)
  - `optimization_stats`: Performance metrics and statistics

### Optimization Statistics

- **Optimization Time**: Total processing time
- **Final Error**: Convergence error value
- **Iterations**: Number of optimization iterations
- **Active LEDs**: Number of LEDs with non-zero values
- **Max LED Value**: Peak LED brightness (0-1 range)

## Installation Requirements

All tools require the base Prismatron environment. Additional requirements:

```bash
# Camera and image processing
pip install opencv-python Pillow

# Web visualization  
pip install flask

# Optimization (if not already installed)
pip install numpy scipy
```

## Workflow Example

Complete workflow from calibration to optimization:

```bash
# 1. Calibrate camera
python tools/camera_calibration.py --output-config calibration.json

# 2. Capture diffusion patterns (using calibration)
python tools/capture_diffusion_patterns.py \
    --wled-host 192.168.1.100 \
    --crop-region 260 140 1400 840 \
    --output real_patterns.npz \
    --preview

# 3. Visualize captured patterns
python tools/visualize_diffusion_patterns.py \
    --patterns real_patterns.npz \
    --host 0.0.0.0 \
    --port 8080 &

# 4. Optimize test image
python tools/standalone_optimizer.py \
    --input test_image.jpg \
    --patterns real_patterns.npz \
    --output optimized_result.png \
    --preview \
    --save-data
```

## Memory and Performance Notes

- **Diffusion Patterns**: ~900MB for full captured dataset (uint8 optimization)
- **Frame Resolution**: 800x480 (reduced from 1080p for performance)
- **LED Count**: 3,200 LEDs × 3 channels = 9,600 patterns
- **Capture Time**: ~16 minutes at 10fps
- **Optimization Time**: 1-10 seconds depending on method and iterations
- **Memory Usage**: uint8 patterns use ~3.5GB in RAM vs 14GB for float32

All tools are designed to work within the 8GB memory constraints of the NVIDIA Jetson Orin Nano. The uint8 optimization reduces memory usage by 75% while maintaining full pattern fidelity.
