# Build-Up/Drop Detection for Audio-Reactive LED Effects

## Overview

The `BuildDropDetector` class provides real-time detection of house/trance music build-up and drop patterns for triggering dramatic LED effects. It uses Aubio's phase vocoder for spectral analysis to detect:

- **Build-up phases**: Progressive increase in hi-hat density and energy
- **Pre-drop breaks**: Brief energy dips after build-ups
- **Drop moments**: Sudden bass return with high energy (the main impact!)
- **Post-drop phases**: Sustained high energy after drops

## State Machine

```
NORMAL ──→ BUILDUP ──→ PREDROP ──→ DROP ──→ POSTDROP ──→ NORMAL
   ↑          ↓                                    ↓
   └──────────┘                                    ↓
   (if confidence drops)                           ↓
                                                   ↓
              ←────────────────────────────────────┘
```

### State Descriptions

- **NORMAL**: Regular playback, no build/drop pattern detected
- **BUILDUP**: Build-up pattern detected - hi-hat density increasing, energy rising
- **PREDROP**: Brief energy dip detected after build-up (anticipation moment)
- **DROP**: Drop moment detected - sudden bass spike! (single-frame event)
- **POSTDROP**: Post-drop sustained energy phase (~2-3 seconds)

## Quick Start

### Basic Usage

```python
from src.consumer.audio_beat_analyzer import (
    AudioBeatAnalyzer,
    BuildDropConfig,
    BuildDropEvent,
)

def builddrop_callback(event: BuildDropEvent):
    """Handle build-up/drop events"""
    if event.event_type == "BUILDUP":
        # Gradually increase effect intensity
        apply_buildup_effect(event.buildup_intensity)

    elif event.event_type == "DROP":
        # Trigger explosive visual effect
        trigger_drop_effect()

    elif event.event_type == "POSTDROP":
        # Maintain elevated reactivity
        apply_enhanced_bass_response(event.bass_energy)

# Create analyzer with build-drop detection enabled
analyzer = AudioBeatAnalyzer(
    builddrop_callback=builddrop_callback,
    enable_builddrop_detection=True,
)

analyzer.start_analysis()
```

### With Custom Configuration

```python
# Customize detection parameters
config = BuildDropConfig(
    hihat_transient_threshold=0.15,    # Hi-hat detection sensitivity
    buildup_entry_threshold=8,         # Consecutive hi-hats to enter BUILDUP
    drop_bass_multiplier=3.0,          # Bass spike multiplier for drops
    energy_drop_threshold=0.5,         # Energy drop ratio for PREDROP
)

analyzer = AudioBeatAnalyzer(
    builddrop_callback=builddrop_callback,
    enable_builddrop_detection=True,
    builddrop_config=config,
)
```

## Configuration Parameters

### Frequency Bands

These define which frequencies are analyzed for each component:

```python
bass_range: (20.0, 250.0)      # Kick drum, bass
mid_range: (250.0, 2000.0)     # Snare, vocals, melodic
high_range: (2000.0, 8000.0)   # Cymbals, general hi-freq
air_range: (8000.0, 16000.0)   # Hi-hats, transient clicks
```

### Detection Thresholds

Fine-tune detection sensitivity:

```python
hihat_transient_threshold: 0.25   # Energy level for hi-hat detection
                                  # Lower = more sensitive (more false positives)
                                  # Higher = less sensitive (may miss buildups)
                                  # Default 0.25 is conservative to avoid false positives

buildup_entry_threshold: 20       # Consecutive hi-hat frames to enter BUILDUP
                                  # (~0.23s at default settings)
                                  # Increased from 8 to reduce false positives

drop_bass_multiplier: 4.0         # Bass spike multiplier for drop detection
                                  # Drop detected when bass > avg * this value
                                  # Increased from 3.0 to require stronger drops

energy_drop_threshold: 0.35       # Energy drop ratio for PREDROP
                                  # 0.35 = 65% energy drop triggers PREDROP
                                  # Decreased from 0.5 to require more dramatic drop
```

### State Duration Parameters

Control timing of state transitions (in frames, ~11.6ms per frame at 44.1kHz):

```python
min_buildup_frames: 86            # ~1.0s minimum before PREDROP allowed
                                  # Increased from 20 to avoid false positives
                                  # Real buildups are typically 4-8+ seconds
max_predrop_frames: 172           # ~2s maximum in PREDROP before timeout
postdrop_duration_frames: 200     # ~2.3s duration of POSTDROP state
```

### Energy History Tracking

```python
energy_history_size: 100          # Frames of total energy history
bass_history_size: 50             # Frames of bass energy history
```

## BuildDropEvent Structure

When a state change occurs, the callback receives a `BuildDropEvent`:

```python
@dataclass
class BuildDropEvent:
    timestamp: float           # Event time in seconds (from audio start)
    system_time: float         # System time when detected
    event_type: str            # NORMAL, BUILDUP, PREDROP, DROP, POSTDROP
    buildup_intensity: float   # 0.0-1.0 build-up progression
    bass_energy: float         # Current bass energy level
    high_energy: float         # Current high-frequency energy level
    confidence: float          # Detection confidence (0.0-1.0)
```

## LED Effect Examples

### Build-up Effect

Gradually increase visual intensity as the build-up progresses:

```python
def apply_buildup_effect(intensity: float):
    """
    Apply progressive build-up effect.
    intensity: 0.0 (start) to 1.0 (peak)
    """
    # Increase saturation
    saturation = 0.3 + (intensity * 0.7)  # 30% → 100%

    # Increase pulse rate
    pulse_rate = 1.0 + (intensity * 3.0)  # 1x → 4x

    # Shift colors toward white
    white_mix = intensity * 0.5  # 0% → 50% white

    # Apply to LED renderer
    renderer.set_saturation_scale(saturation)
    renderer.set_pulse_rate(pulse_rate)
    renderer.set_white_mix(white_mix)
```

### Drop Effect

Explosive visual impact at the drop moment:

```python
def trigger_drop_effect():
    """Trigger explosive drop effect"""
    # Flash to full white
    renderer.flash_white(duration=0.1)

    # Trigger expanding circle animation
    renderer.trigger_circle_wave(
        center=(32, 32),
        max_radius=50,
        duration=1.0,
        color=(255, 255, 255),
    )

    # Maximize saturation
    renderer.set_saturation_scale(1.0)

    # Trigger bass pump effect
    renderer.enable_bass_pump(intensity=1.0, duration=2.0)
```

### Post-drop Effect

Maintain elevated reactivity after drop:

```python
def apply_postdrop_effect(bass_energy: float):
    """Enhanced bass response during post-drop"""
    # Scale bass reactivity
    bass_scale = 1.5 + (bass_energy * 2.0)
    renderer.set_bass_scale(bass_scale)

    # Maintain high saturation
    renderer.set_saturation_scale(0.9)

    # Stronger beat flash
    renderer.set_beat_flash_intensity(0.8)
```

## Testing

### Using the Test Script

Run the included test script to see detection in action:

```bash
# Activate virtual environment
source env/bin/activate

# Run test script
python examples/test_builddrop_detection.py
```

Play some house/trance music and watch the console output for state transitions!

### Using the Main Module

You can also run the main audio_beat_analyzer module directly:

```bash
python -m src.consumer.audio_beat_analyzer
```

Select option 2 for build-up/drop detection.

## Tuning Tips

### Too Many False Positives?

**Note**: Default values have been tuned to minimize false positives. If you're still getting too many:

1. **Increase `hihat_transient_threshold`**: Requires stronger hi-hats to trigger
   ```python
   config.hihat_transient_threshold = 0.30  # Even more conservative
   ```

2. **Increase `buildup_entry_threshold`**: Requires longer hi-hat pattern
   ```python
   config.buildup_entry_threshold = 30  # ~0.35s - longer pattern required
   ```

3. **Increase `drop_bass_multiplier`**: Requires stronger bass spike
   ```python
   config.drop_bass_multiplier = 5.0  # More dramatic drop required
   ```

4. **Increase `min_buildup_frames`**: Require longer buildup duration
   ```python
   config.min_buildup_frames = 172  # ~2s minimum buildup duration
   ```

### Missing Real Build-ups?

If legitimate build-ups aren't detected:

1. **Decrease `hihat_transient_threshold`**: More sensitive to hi-hats
   ```python
   config.hihat_transient_threshold = 0.20  # More sensitive
   ```

2. **Decrease `buildup_entry_threshold`**: Shorter pattern needed
   ```python
   config.buildup_entry_threshold = 12  # ~0.14s - shorter pattern
   ```

3. **Decrease `min_buildup_frames`**: Allow shorter buildups
   ```python
   config.min_buildup_frames = 43  # ~0.5s minimum
   ```

4. **Adjust frequency ranges**: Some tracks use different frequency ranges
   ```python
   config.air_range = (6000.0, 14000.0)  # Adjust hi-hat range
   ```

### Drops Not Detected?

If drops are missed:

1. **Decrease `drop_bass_multiplier`**: Less dramatic spike needed
   ```python
   config.drop_bass_multiplier = 3.0  # More sensitive
   ```

2. **Decrease `energy_drop_threshold`**: Allow smaller energy dips
   ```python
   config.energy_drop_threshold = 0.45  # 55% drop instead of 65%
   ```

3. **Check bass frequency range**: Some tracks have sub-bass drops
   ```python
   config.bass_range = (20.0, 300.0)  # Include more sub-bass
   ```

## Performance

### Computational Cost

- **Phase Vocoder**: ~2048-point FFT per frame (~11.6ms intervals)
- **State Machine**: Minimal overhead (simple comparisons)
- **Total overhead**: ~5-10% CPU on Jetson Orin Nano

### Memory Usage

- **Energy history buffers**: ~1KB per detector
- **FFT buffers**: ~8KB (for 2048-point complex FFT)
- **Total**: ~10KB per detector instance

### Real-time Performance

Detection latency breakdown:
- Audio capture: ~11.6ms (one frame)
- FFT processing: ~1-2ms
- State machine: <0.1ms
- **Total latency**: ~13-15ms (excellent for real-time use!)

## Technical Details

### Frequency Band Analysis

The detector uses the Aubio phase vocoder (`aubio.pvoc`) to extract frequency spectra with emphasis on transients:

```python
# Initialize phase vocoder (2048-point window, 512-sample hop)
pvoc = aubio.pvoc(win_size=2048, hop_size=512)

# Process frame
pvoc.do(audio_frame, fft_grain)
spectrum = fft_grain.norm  # Get magnitude spectrum

# Extract band energy
bass_energy = sum(spectrum[bass_bins]^2)
air_energy = sum(spectrum[air_bins]^2)
```

### Hi-hat Detection

Hi-hats are detected in the "air" band (8-16 kHz):

```python
hihat_detected = air_energy > threshold

if hihat_detected:
    hihat_count += 1
else:
    hihat_count = 0  # Reset on gap

# Enter BUILDUP when sustained hi-hats detected
if hihat_count >= entry_threshold:
    state = "BUILDUP"
```

### Drop Detection

Drops are detected by sudden bass spikes after build-ups:

```python
recent_bass_avg = mean(bass_history[-10:])

if bass_energy > recent_bass_avg * drop_multiplier:
    state = "DROP"  # Sudden bass return!
```

## Integration with LED Renderer

### Accessing State from Renderer

The audio state is available via `AudioState`:

```python
# In your LED renderer
state = audio_analyzer.get_current_state()

if state.buildup_state == "BUILDUP":
    # Apply build-up effect based on intensity
    apply_effect(state.buildup_intensity)

elif state.buildup_state == "DROP":
    # This will only be true for one frame!
    trigger_drop()
```

### State-based Effect Modulation

Use the state to modulate existing effects:

```python
# In render loop
state = audio_analyzer.get_current_state()

# Base saturation from beat intensity
base_saturation = state.beat_intensity

# Modulate based on build/drop state
if state.buildup_state == "BUILDUP":
    # Progressively increase saturation during build-up
    saturation = base_saturation * (0.5 + 0.5 * state.buildup_intensity)

elif state.buildup_state in ["DROP", "POSTDROP"]:
    # Maximum saturation during/after drop
    saturation = 1.0

else:
    # Normal saturation
    saturation = base_saturation * 0.7

renderer.apply_saturation(saturation)
```

## Troubleshooting

### "Aubio not available" Warning

If you see this warning, install Aubio:

```bash
pip install aubio
```

The detector will fall back to simple FFT if Aubio is unavailable, but phase vocoder analysis is recommended.

### Audio Device Not Found

Check available audio devices:

```python
import sounddevice as sd
print(sd.query_devices())
```

Specify device in AudioConfig:

```python
from src.consumer.audio_capture import AudioConfig

audio_config = AudioConfig(
    sample_rate=44100,
    channels=1,
    chunk_size=1024,
    device_name="Your Device Name",
)

analyzer = AudioBeatAnalyzer(audio_config=audio_config, ...)
```

### High CPU Usage

If CPU usage is too high:

1. Reduce FFT window size (less frequency resolution):
   ```python
   # In BuildDropDetector.__init__
   self.pvoc_win_size = 1024  # Instead of 2048
   ```

2. Increase hop size (less frequent updates):
   ```python
   self.pvoc_hop_size = 1024  # Instead of 512
   ```

## Future Enhancements

Potential improvements for future versions:

1. **Machine Learning**: Train on labeled house/trance tracks for better detection
2. **Genre Adaptation**: Auto-tune parameters based on detected genre
3. **Multi-drop Detection**: Handle multiple consecutive drops (rare but possible)
4. **Build-up Types**: Distinguish between hi-hat builds and synth builds
5. **Breakdown Detection**: Detect breakdown sections (opposite of build-ups)

## References

- [Aubio Documentation](https://aubio.org/doc/latest/)
- [House Music Production Guide](https://www.attackmagazine.com/technique/tutorials/classic-house-buildups-and-breakdowns/)
- [Digital Signal Processing Basics](https://en.wikipedia.org/wiki/Phase_vocoder)

## Support

For issues or questions:
1. Check logs for detailed state transition information
2. Enable DEBUG logging: `logging.getLogger("src.consumer.audio_beat_analyzer").setLevel(logging.DEBUG)`
3. Use the test script to validate detection behavior
4. Adjust configuration parameters based on your specific music

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0
