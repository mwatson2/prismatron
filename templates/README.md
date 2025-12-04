# LED Effect Templates

This directory contains pre-generated LED effect templates for use with the TemplateEffect class in `src/consumer/led_effect.py`.

## Available Templates

### 1. Wide Ring (80px width)
- **Files**:
  - `wide_ring_800x480.npy` - Template frames (30 frames, 800x480)
  - `wide_ring_800x480_leds.npy` - Optimized LED patterns (30 frames, 3200 LEDs)
- **Description**: Growing ring animation with 80px outline width (double the standard 40px)
- **Animation**: Ring expands from center to screen edges with sine-wave intensity falloff

### 2. Heart Shape
- **Files**:
  - `heart_800x480.npy` - Template frames (30 frames, 800x480)
  - `heart_800x480_leds.npy` - Optimized LED patterns (30 frames, 3200 LEDs)
- **Description**: Growing heart outline with point-down orientation
- **Animation**: Heart grows from center using parametric heart equation with 40px outline
- **Equation**: `x = 16*sin³(t)`, `y = -(13*cos(t) - 5*cos(2t) - 2*cos(3t) - cos(4t))`

### 3. Seven-Pointed Star
- **Files**:
  - `star7_800x480.npy` - Template frames (30 frames, 800x480)
  - `star7_800x480_leds.npy` - Optimized LED patterns (30 frames, 3200 LEDs)
- **Description**: Growing 7-pointed star with 40px outline
- **Animation**: Star expands until fully off-screen
- **Geometry**: Outer points at 2x distance of inner points from center

## Template Specifications

All templates share these specifications:
- **Resolution**: 800x480 pixels
- **Frames**: 30 frames per animation
- **Outline Width**: 40px (except wide_ring at 80px)
- **Intensity Falloff**: Sine curve for smooth gradients
- **Frame Format**: float16 precision, shape (frames, height, width), range [0, 1]
- **LED Format**: float32 precision, shape (frames, led_count), range [0, 255]

## Usage Example

```python
from src.consumer.led_effect import TemplateEffectFactory, LedEffectManager

# Pre-load templates during system startup (one-time operation)
TemplateEffectFactory.preload_templates([
    "templates/wide_ring_800x480_leds.npy",
    "templates/heart_800x480_leds.npy",
    "templates/star7_800x480_leds.npy",
])

# Create effect manager
effect_manager = LedEffectManager()

# Create and add effects during runtime (e.g., on beat detection)
effect = TemplateEffectFactory.create_effect(
    template_path="templates/heart_800x480_leds.npy",
    start_time=current_frame_time,
    duration=0.5,  # Play over 0.5 seconds
    blend_mode="addboost",
    intensity=3.0,
    loop=False,
)
effect_manager.add_effect(effect)

# Apply effects to LED frame
effect_manager.apply_effects(led_values, frame_timestamp)
```

## Blend Modes

Templates support multiple blend modes:
- **alpha**: Classic alpha blending (fade in/out)
- **add**: Additive blending (brightens)
- **multiply**: Multiplicative blending (darkens/modulates)
- **replace**: Direct replacement
- **boost**: Multiplicative boost (brightens proportionally)
- **addboost**: Combined boost + add (recommended for beats)

## Generating New Templates

Use the provided tools to generate custom templates:

```bash
# Generate wide ring template
./tools/generate_wide_ring_template.py --frames 30 --width 800 --height 480 --ring-width 80.0

# Generate heart template
./tools/generate_heart_template.py --frames 30 --width 800 --height 480 --outline-width 40.0

# Generate star template
./tools/generate_star_template.py --frames 30 --width 800 --height 480 --num-points 7

# Optimize template to LED patterns
./tools/optimize_template_to_leds.py <template>.npy --max-iterations 10
```

## File Sizes

- Template frames: ~22 MB each (30 frames × 800×480 × 2 bytes fp16)
- LED patterns: ~376 KB each (30 frames × 3200 LEDs × 4 bytes fp32)

LED patterns are 58x smaller than template frames and optimized for real-time playback.
