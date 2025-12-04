# LED Position Access Guide for Prismatron

This guide explains how LED positions and coordinates are stored, accessed, and used in the Prismatron LED Display System.

## Overview

LED positions in Prismatron represent the physical (x,y) coordinates of each LED in the display, measured in pixels within the target frame coordinate system (typically 800x480 pixels). These positions are crucial for the diffusion pattern optimization process.

## Storage Format

LED positions are stored in the diffusion pattern files (`.npz` format) with the following structure:

### 1. LED Positions Array
- **Key**: `led_positions`
- **Format**: `numpy.ndarray` with shape `(led_count, 2)`
- **Data Type**: `int64`
- **Content**: Physical (x,y) coordinates in pixels
- **Example**: `[[32, 38], [351, 172], [247, 72], ...]`

### 2. LED Spatial Mapping
- **Key**: `led_spatial_mapping`
- **Format**: `dict` mapping original LED indices to optimized indices
- **Purpose**: Maps between original LED IDs and spatially-optimized ordering
- **Example**: `{1620: 0, 1132: 1, 1166: 2, ...}`

## Coordinate System

- **Origin**: Top-left corner (0, 0)
- **X-axis**: Increases rightward (0 to frame_width-1)
- **Y-axis**: Increases downward (0 to frame_height-1)
- **Frame Size**: Typically 800×480 pixels
- **LED Range**: X: 20-779, Y: 20-459 (with 20-pixel margin)

## Access Methods

### Method 1: Direct File Access (Recommended)

```python
import numpy as np

def load_led_positions(pattern_path):
    """Load LED positions directly from pattern file."""
    data = np.load(pattern_path, allow_pickle=True)

    # Get LED positions array
    led_positions = data.get("led_positions", None)  # Shape: (led_count, 2)

    # Get spatial mapping
    led_spatial_mapping = data.get("led_spatial_mapping", None)
    if hasattr(led_spatial_mapping, 'item'):
        led_spatial_mapping = led_spatial_mapping.item()

    return led_positions, led_spatial_mapping

# Usage
pattern_file = "diffusion_patterns/synthetic_2624_uint8.npz"
positions, mapping = load_led_positions(pattern_file)
print(f"LED 0 position: {positions[0]}")  # Physical coordinates
```

### Method 2: Through LED Optimizer (Internal)

```python
from consumer.led_optimizer import LEDOptimizer

optimizer = LEDOptimizer(diffusion_patterns_path="diffusion_patterns/synthetic_2624_uint8.npz")
optimizer.initialize()

# Access internal attributes (private - not recommended for production)
led_positions = optimizer._led_positions        # (led_count, 2) array
led_mapping = optimizer._led_spatial_mapping    # dict mapping
```

### Method 3: Using Utility Functions

```python
from tools.led_position_utils import generate_random_led_positions

# Generate new random positions
positions = generate_random_led_positions(
    led_count=2624,
    frame_width=800,
    frame_height=480,
    seed=42
)
```

## Coordinate Mapping Examples

### Get Physical Position by LED Index

```python
# For original LED ID
led_id = 0
position = led_positions[led_id]
print(f"LED {led_id} at position ({position[0]}, {position[1]})")

# For optimized LED index (after spatial reordering)
optimized_index = 0
# Find original LED ID that maps to this optimized index
original_id = None
for orig_id, opt_id in led_spatial_mapping.items():
    if opt_id == optimized_index:
        original_id = orig_id
        break

if original_id is not None:
    position = led_positions[original_id]
    print(f"Optimized index {optimized_index} -> LED {original_id} -> Position {position}")
```

### Convert Between Index Types

```python
def get_optimized_index(original_led_id, led_spatial_mapping):
    """Get optimized index for an original LED ID."""
    return led_spatial_mapping.get(original_led_id, None)

def get_original_id(optimized_index, led_spatial_mapping):
    """Get original LED ID for an optimized index."""
    for orig_id, opt_id in led_spatial_mapping.items():
        if opt_id == optimized_index:
            return orig_id
    return None
```

## File Locations

### Main Diffusion Pattern Files
- `diffusion_patterns/synthetic_2624_uint8.npz` - Main 2624 LED pattern
- `diffusion_patterns/synthetic_2624_fp32.npz` - FP32 version
- `diffusion_patterns/synthetic_*_dia_*.npz` - Variants with different DIA factors

### Related Code Files
- `src/consumer/led_optimizer.py` - LED optimizer (loads positions internally)
- `src/utils/diffusion_pattern_manager.py` - Pattern management utilities
- `tools/led_position_utils.py` - Position generation utilities
- `tools/generate_synthetic_patterns.py` - Pattern generation tool

## Position Data Characteristics

Based on the current `synthetic_2624_uint8.npz` file:

- **LED Count**: 2624 LEDs
- **Position Range**: X: 20-779 pixels, Y: 20-459 pixels  
- **Distribution**: Uniform random within frame bounds
- **Margin**: 20-pixel border to avoid edge effects
- **Data Type**: int64 coordinates
- **Spatial Optimization**: Reordered using RCM (Reverse Cuthill-McKee) algorithm

## Usage in Optimization

LED positions are used to:

1. **Generate Diffusion Patterns**: Each LED's position determines its diffusion footprint
2. **Spatial Optimization**: LEDs are reordered for better cache locality
3. **Block Positioning**: Determine optimal diffusion block placement
4. **Adjacency Calculations**: Compute which LEDs have overlapping diffusion patterns

## Pattern Generation Context

LED positions are generated during pattern creation with:

- **Random Seed**: For reproducible layouts
- **Block Size**: Typically 64x64 pixel diffusion blocks
- **Alignment**: Block positions aligned to 4-pixel boundaries for CUDA efficiency
- **Spatial Ordering**: Z-order curve for improved memory locality

## Key Points

1. **Immutable**: LED positions are fixed per pattern file and don't change during optimization
2. **Physical Coordinates**: Represent actual pixel locations in the target frame
3. **Optimization Mapping**: Spatial reordering improves performance without changing coordinates
4. **Direct Access**: Best accessed directly from pattern files rather than through optimizer
5. **Integer Coordinates**: All positions are integer pixel coordinates
