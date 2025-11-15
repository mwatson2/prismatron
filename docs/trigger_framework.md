# LED Effect Trigger Framework

## Overview

The trigger framework provides a flexible, configuration-driven way to create LED effects in response to various events (beats, periodic timers, etc.). This replaces the previous hardcoded approach with a declarative configuration system.

## Architecture

### Components

1. **EffectTriggerConfig** - Configuration dataclass defining:
   - Trigger type (`"beat"` or `"test"`)
   - Effect class to instantiate (e.g., `"BeatBrightnessEffect"`, `"TemplateEffect"`)
   - Effect parameters
   - Trigger conditions (thresholds, BPM ranges, etc.)

2. **EffectTriggerManager** - Evaluates triggers and creates effects:
   - Maintains list of configured triggers
   - Evaluates beat triggers when new beats are detected
   - Evaluates test triggers on periodic intervals
   - Uses first-match semantics (stops at first matching trigger)

3. **LedEffectManager** - Manages active effects (existing component)

## Trigger Types

### Beat Triggers

Beat triggers fire when audio beats are detected. They support multiple conditions (ALL must be satisfied):

- **confidence_min**: Minimum beat confidence [0, 1]
- **intensity_min**: Minimum beat intensity [0, 1]
- **bpm_min**: Minimum BPM threshold
- **bpm_max**: Maximum BPM threshold

**Example:**
```python
trigger = EffectTriggerConfig(
    trigger_type="beat",
    effect_class="BeatBrightnessEffect",
    effect_params={
        "boost_intensity": 4.0,
        "duration_fraction": 0.4
    },
    confidence_min=0.5,     # Ignore weak beats
    bpm_min=120.0,          # Only for fast tempos
    bpm_max=140.0
)
```

### Test Triggers

Test triggers fire periodically for testing/debugging. They have no conditions and fire at a global interval.

**Example:**
```python
trigger = EffectTriggerConfig(
    trigger_type="test",
    effect_class="TemplateEffect",
    effect_params={
        "template_path": "templates/ring_800x480_leds.npy",
        "duration": 1.0,
        "blend_mode": "addboost",
        "intensity": 2.0
    }
)
```

## Available Effects

### BeatBrightnessEffect

Applies a sine wave brightness boost synchronized to beats.

**Parameters:**
- `boost_intensity`: Base boost intensity [0, 5.0]
- `duration_fraction`: Fraction of beat interval for boost [0.1, 1.0]

**Auto-provided by trigger:**
- `bpm`: Current BPM from beat state
- `beat_intensity`: Beat intensity from beat detection
- `beat_confidence`: Beat confidence from beat detection

### TemplateEffect

Plays pre-optimized LED patterns from template files.

**Parameters:**
- `template_path`: Path to template .npy file (required)
- `duration`: Effect duration in seconds
- `blend_mode`: How to blend template ("alpha", "add", "multiply", "replace", "boost", "addboost")
- `intensity`: Effect intensity [0, 1+]
- `loop`: Whether to loop the template (default: False)
- `add_multiplier`: For "addboost" mode [0, 1+]

## Configuration

### Backward Compatibility (Current)

The framework auto-generates triggers from existing `ControlState` settings in `_initialize_triggers_from_control_state()`:

1. **Test trigger** - if template effects are enabled
2. **Beat trigger** - if audio reactive and beat brightness are enabled

This maintains compatibility with existing UI controls.

### Future: Explicit Configuration

Later, triggers will be configured through a UI and stored in config files:

```python
# Example: Multiple beat triggers with different effects
triggers = [
    # High-energy beats -> strong brightness boost
    EffectTriggerConfig(
        trigger_type="beat",
        effect_class="BeatBrightnessEffect",
        effect_params={"boost_intensity": 5.0, "duration_fraction": 0.3},
        confidence_min=0.8,
        intensity_min=0.7
    ),

    # Medium beats -> template overlay
    EffectTriggerConfig(
        trigger_type="beat",
        effect_class="TemplateEffect",
        effect_params={
            "template_path": "templates/star7_800x480_leds.npy",
            "duration": 0.5,
            "blend_mode": "add",
            "intensity": 1.5
        },
        confidence_min=0.5,
        bpm_min=100.0,
        bpm_max=140.0
    ),

    # Periodic test effect
    EffectTriggerConfig(
        trigger_type="test",
        effect_class="TemplateEffect",
        effect_params={
            "template_path": "templates/ring_800x480_leds.npy",
            "duration": 1.0,
            "blend_mode": "addboost",
            "intensity": 2.0
        }
    )
]

# Apply configuration
renderer.trigger_manager.set_triggers(triggers)
renderer.trigger_manager.set_test_interval(2.0)
```

## First-Match Semantics

Triggers are evaluated in order. When a trigger matches, its effect is created and evaluation stops.

This allows **cascading thresholds**:

```python
triggers = [
    # Try high-intensity effect first
    EffectTriggerConfig(
        trigger_type="beat",
        effect_class="BeatBrightnessEffect",
        effect_params={"boost_intensity": 5.0, ...},
        confidence_min=0.8
    ),

    # Fall back to medium-intensity effect
    EffectTriggerConfig(
        trigger_type="beat",
        effect_class="BeatBrightnessEffect",
        effect_params={"boost_intensity": 3.0, ...},
        confidence_min=0.5
    ),

    # Catch-all for any beat
    EffectTriggerConfig(
        trigger_type="beat",
        effect_class="BeatBrightnessEffect",
        effect_params={"boost_intensity": 1.0, ...},
        confidence_min=0.0
    )
]
```

## Template Caching

`TemplateEffect` instances are created using `TemplateEffectFactory`, which caches template data to avoid repeated file I/O. This is critical for beat-triggered effects that may fire multiple times per second.

**Pre-loading templates:**
```python
from src.consumer.led_effect import TemplateEffectFactory

# Pre-load templates during initialization
TemplateEffectFactory.preload_templates([
    "templates/ring_800x480_leds.npy",
    "templates/star7_800x480_leds.npy",
    "templates/heart_800x480_leds.npy"
])
```

## Testing

Run the test suite:
```bash
python test_trigger_framework.py
```

Tests cover:
- Trigger configuration validation
- Beat trigger evaluation with various conditions
- Test trigger periodic firing
- Effect creation and template caching

## Migration Notes

### Old Approach (Deprecated)
```python
# Hardcoded in _check_and_create_beat_brightness_effect()
if beat_confidence >= threshold:
    effect = BeatBrightnessEffect(...)
    self.effect_manager.add_effect(effect)
```

### New Approach
```python
# Configured via triggers
trigger = EffectTriggerConfig(
    trigger_type="beat",
    effect_class="BeatBrightnessEffect",
    effect_params={...},
    confidence_min=0.5
)

# Framework handles evaluation and creation
trigger_manager.evaluate_beat_triggers(...)
```

## Adding New Effects

To add a new effect class:

1. Create effect class in `src/consumer/led_effect.py`:
```python
class MyCustomEffect(LedEffect):
    def __init__(self, start_time, my_param1, my_param2, **kwargs):
        super().__init__(start_time=start_time, **kwargs)
        self.my_param1 = my_param1
        self.my_param2 = my_param2

    def apply(self, led_values, frame_timestamp):
        # Modify led_values in-place
        ...
        return self.is_complete(frame_timestamp)
```

2. Configure trigger:
```python
trigger = EffectTriggerConfig(
    trigger_type="beat",  # or "test"
    effect_class="MyCustomEffect",
    effect_params={
        "my_param1": value1,
        "my_param2": value2
    },
    confidence_min=0.5  # beat trigger conditions
)
```

The framework will instantiate your effect with `MyCustomEffect(start_time=..., my_param1=..., my_param2=...)`.

## Future Enhancements

Potential future trigger types:
- **Time triggers** - Fire at specific times of day
- **Manual triggers** - User-initiated via UI
- **Playlist triggers** - Fire on track changes
- **Condition triggers** - Complex boolean logic
