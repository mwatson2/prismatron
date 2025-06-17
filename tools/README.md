# WLED Test Patterns

This directory contains testing utilities for the Prismatron LED display system.

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