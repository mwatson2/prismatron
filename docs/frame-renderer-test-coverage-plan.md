# Frame Renderer Test Coverage Plan

This document tracks the test coverage improvements needed for `src/consumer/frame_renderer.py`.

## Current Status

Current metrics (see Git history for last update):
- **Unit tests**: 81 tests in `tests/consumer/test_frame_renderer_unit.py`
- **File size**: 2106 lines containing 2 classes and many methods
- **Factory tests**: `TestCreateFrameRendererWithPattern` in `tests/utils/test_pattern_loader.py` (3 tests)

## Classes to Test

### 1. EffectTriggerConfig (lines 27-66) ✅ COMPLETED
A dataclass for effect trigger configuration with validation.

Methods/Properties to test:
- `__post_init__` validation:
  - Invalid trigger_type raises ValueError ✅
  - Beat trigger without conditions logs warning ✅
  - Valid configurations pass validation ✅

### 2. EffectTriggerManager (lines 68-388) ✅ COMPLETED
Manages effect triggers and creates effect instances.

Methods to test:
- `__init__` - Initialize manager with effect manager ✅
- `set_triggers` - Backward compatibility configuration ✅
- `set_common_and_carousel_triggers` - New common + carousel structure ✅
- `set_test_interval` - Set test trigger interval ✅
- `evaluate_beat_triggers` - Beat trigger evaluation ✅
- `_check_beat_conditions` - Beat condition checking ✅
- `_create_and_add_beat_effect` - Effect creation from beat ✅
- `_check_carousel_rotation` - Carousel rotation logic ✅
- `evaluate_test_triggers` - Test trigger evaluation ✅
- `_create_effect` - Effect factory method (partial)

### 3. FrameRenderer (lines 390-2106)
Main timestamp-based frame renderer.

## Priority 1: Core Timing and Rendering ✅ COMPLETED

### `establish_wallclock_delta` (lines 1009-1037) ✅
- Test: First frame establishes delta correctly ✅
- Test: Warning logged if delta already established ✅
- Test: Control state updated with delta (partial)

### `render_frame_at_timestamp` (lines 1039-1162) ✅
- Test: First frame calls establish_wallclock_delta ✅
- Test: Statistics updated correctly ✅
- Test: Returns False on error ✅
- Test: Discontinuity detection and timeline adjustment (not tested - complex)
- Test: First frame of new playlist item handling (not tested - complex)

### `is_frame_late` (lines 1996-2027) ✅
- Test: Returns False when not initialized ✅
- Test: Returns False for first frame of new item ✅
- Test: Returns True when frame exceeds late threshold ✅
- Test: Returns False when frame within threshold ✅

### `is_initialized` (lines 1992-1994) ✅
- Test: Returns False initially ✅
- Test: Returns True after first frame received ✅

## Priority 2: Pause/Resume and Timing Adjustments ✅ COMPLETED

### `pause_renderer` (lines 2029-2036) ✅
- Test: Sets is_paused flag ✅
- Test: Records pause_start_time ✅
- Test: No-op if already paused ✅

### `resume_renderer` (lines 2038-2050) ✅
- Test: Clears is_paused flag ✅
- Test: Adds pause duration to total_pause_time ✅
- Test: No-op if not paused ✅

### `get_adjusted_wallclock_delta` (lines 2051-2069) ✅
- Test: Returns 0 if wallclock_delta not set ✅
- Test: Returns delta + total_pause_time ✅
- Test: Adds current pause duration if currently paused ✅

## Priority 3: Sink Management ✅ COMPLETED

### `register_sink` / `unregister_sink` / `set_sink_enabled` ✅
- Test: Register sink with send_led_data method ✅
- Test: Register sink with render_led_values method ✅
- Test: Raises error for sink without compatible method ✅
- Test: Unregister removes sink ✅
- Test: Enable/disable toggles sink state ✅

### `set_output_targets` (lines 961-995) ✅
- Test: Clears existing sinks (implicit) ✅
- Test: Registers WLED sink if provided ✅
- Test: Registers test sink if provided ✅
- Test: Registers preview sink if provided ✅
- Test: Maintains legacy references ✅

### `_send_to_outputs` (lines 1547-1683)
- Test: Sends to all enabled sinks (not tested - complex)
- Test: Skips disabled sinks (not tested - complex)
- Test: Handles send_led_data interface (not tested - complex)
- Test: Handles render_led_values interface (not tested - complex)
- Test: Logs errors for failing sinks (not tested - complex)

## Priority 4: LED Conversion ✅ COMPLETED

### `_convert_spatial_to_physical` (lines 1786-1811) ✅
- Test: Returns unchanged if led_ordering is None ✅
- Test: Correctly reorders LED values according to led_ordering array ✅

## Priority 5: Statistics and Configuration ✅ COMPLETED

### `reset_stats` (lines 1886-1916) ✅
- Test: Resets all counters to zero ✅
- Test: Clears timing_errors list ✅
- Test: Resets EWMA values ✅
- Test: Resets output FPS tracking ✅
- Test: Clears pause state ✅

### `set_timing_parameters` (lines 1918-1945) ✅
- Test: Updates first_frame_delay ✅
- Test: Updates timing_tolerance ✅
- Test: Updates late_frame_log_threshold ✅
- Test: Handles None values (no change) ✅

### `set_ewma_alpha` (lines 1947-1958) ✅
- Test: Sets valid alpha value ✅
- Test: Raises error for invalid alpha (<=0 or >1) ✅

### `get_timing_stats` / `get_renderer_stats` (lines 1813-1884) ✅
- Test: Returns comprehensive stats dictionary ✅
- Test: Calculates percentages correctly ✅
- Test: Includes all expected keys ✅
- Test: get_renderer_stats is alias ✅

### `get_recent_performance_summary` (lines 1972-1990) ✅
- Test: Returns EWMA-based performance metrics ✅
- Test: is_performing_well logic correct ✅

## Priority 6: Beat Detection and Effects (Partial)

### `_check_and_create_beat_brightness_effect` (lines 837-915) ✅
- Test: Returns early if control state missing ✅
- Test: Returns early if audio analyzer missing ✅
- Test: Returns early if audio reactive disabled ✅
- Test: Calls trigger manager on valid beat ✅
- Test: Returns early if audio not enabled (partial)
- Test: Handles exceptions gracefully (not tested)

### `_manage_sparkle_effect_for_buildup` (lines 1211-1317)
- Test: Returns if sparkle disabled in config (not tested)
- Test: Returns if audio analyzer missing (not tested)
- Test: Removes sparkle on cut/drop events (not tested)
- Test: Removes sparkle when intensity drops to 0 (not tested)
- Test: Creates sparkle effect when intensity > 0 (not tested)
- Test: Updates existing sparkle parameters (not tested)

### `_manage_cut_fade_effect` / `_manage_drop_fade_effect`
- Test: Returns if audio analyzer missing (not tested)
- Test: Returns if effect disabled (not tested)
- Test: Creates effect on new cut/drop event (not tested)
- Test: Removes existing effect before creating new (not tested)
- Test: Cleans up reference when effect completes (not tested)

## Priority 7: EffectTriggerManager Tests ✅ COMPLETED

### `set_common_and_carousel_triggers` ✅
- Test: Sets common triggers list ✅
- Test: Sets carousel rule sets ✅
- Test: Sets carousel beat interval ✅
- Test: Resets carousel indices ✅

### `evaluate_beat_triggers` ✅
- Test: Skips already-processed beats ✅
- Test: Waits for beat intensity to be ready ✅
- Test: Checks common rules first (implicit)
- Test: Falls back to carousel rules (not tested)
- Test: Rotates carousel after interval ✅

### `_check_beat_conditions` ✅
- Test: Returns False if confidence below minimum ✅
- Test: Returns False if intensity below minimum ✅
- Test: Returns False if BPM outside range ✅
- Test: Returns True when all conditions met ✅

### `_create_effect`
- Test: Creates TemplateEffect with factory (not tested)
- Test: Creates other effect types directly (not tested)
- Test: Raises error for unknown effect class (not tested)
- Test: Multiplies intensity by beat_intensity for beat triggers (not tested)

## Priority 8: Helper Methods ✅ COMPLETED

### `_calculate_sparkle_param` (lines 1180-1209) ✅
- Test: Linear curve interpolation ✅
- Test: Ease-in curve interpolation ✅
- Test: Ease-out curve interpolation ✅
- Test: Inverse curve interpolation ✅
- Test: Clamps intensity to 0-10 range ✅
- Test: Works with custom min/max range ✅

### `_create_event_effect` (lines 1319-1404) ✅
- Test: Returns None for "none" effect class ✅
- Test: Creates FadeInEffect ✅
- Test: Creates FadeOutEffect ✅
- Test: Creates RandomInEffect/RandomOutEffect (not tested)
- Test: Creates InverseFadeIn/Out (not tested)
- Test: Creates TemplateEffect (not tested)
- Test: Returns None for unknown class ✅
- Test: Handles exceptions (not tested)

### `_track_timing_error` (lines 1685-1696) ✅
- Test: Appends error to list ✅
- Test: Trims list to max_timing_history ✅

### `_update_ewma_statistics` (lines 1698-1747)
- Test: Initializes EWMA on first frame (not tested)
- Test: Updates EWMA on subsequent frames (not tested)
- Test: Logs large timestamp gaps (not tested)
- Test: Derives FPS from interval EWMA (not tested)

### `mark_frame_dropped` (lines 1960-1970) ✅
- Test: Increments dropped_frames counter ✅

## Priority 9: LED Effects Interface ✅ COMPLETED

### `add_led_effect` / `clear_led_effects` / `get_active_effects_count` ✅
- Test: Delegates to effect_manager.add_effect ✅
- Test: Delegates to effect_manager.clear_effects ✅
- Test: Returns effect_manager.get_active_count() ✅

### `get_led_effects_stats` ✅
- Test: Returns effect_manager.get_stats() ✅

## Notes

- The FrameRenderer has complex interactions with audio beat analyzer and control state
- Many methods require mocking multiple dependencies
- Focus on testing the logic within each method, not integration
- Effect creation tests may need mocking of led_effect module classes
- Per CLAUDE.md: Focus on happy path and failure logging, not fallbacks
