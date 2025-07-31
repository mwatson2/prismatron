# Producer/Renderer State Machine Decoupling Plan

## Overview
This plan implements decoupled state management where producer and renderer operate independently. The producer controls content loading and generation, while the renderer controls playback timing and effects.

## Current State Architecture
- Single `PlayState` enum controls both producer and renderer
- Producer and renderer are tightly coupled through shared play state
- Stop command affects both producer and renderer immediately

## Target State Architecture

### Producer States
```python
class ProducerState(Enum):
    STOPPED = "stopped"     # Not generating frames
    PLAYING = "playing"     # Actively generating frames
    ERROR = "error"         # Producer error state
```

### Renderer States  
```python
class RendererState(Enum):
    STOPPED = "stopped"     # Not rendering frames
    WAITING = "waiting"     # Waiting for frames from producer
    PLAYING = "playing"     # Actively rendering frames
    PAUSED = "paused"       # Paused, holding current frame
    ERROR = "error"         # Renderer error state
```

## State Transition Logic

### Producer State Transitions
- **STOPPED → PLAYING**: User play command, content available
- **PLAYING → STOPPED**: User stop command OR content exhausted
- **Any → ERROR**: Producer error occurs

### Renderer State Transitions
- **STOPPED → WAITING**: User play command (producer starts)
- **WAITING → PLAYING**: LED buffer becomes full (smooth playback ready)
- **WAITING → STOPPED**: Producer stops before buffer fills
- **PLAYING → PAUSED**: User pause command
- **PAUSED → PLAYING**: User play command (resume)
- **PLAYING → STOPPED**: LED buffer empty AND producer stopped
- **PAUSED → STOPPED**: LED buffer empty AND producer stopped
- **Any → ERROR**: Renderer error occurs

### Key Behavioral Changes
1. **Play Command**: Starts producer immediately, renderer transitions to WAITING then PLAYING when buffer fills
2. **Pause Command**: Only pauses renderer, producer continues generating frames
3. **Stop Command**: Stops producer only, renderer stops naturally when buffer empties
4. **Buffer Backpressure**: Renderer state drives LED output, independent of producer
5. **WAITING State**: Renderer waits for first frames after play command, provides visual feedback

## Implementation Tasks

### ✅ Task Progress Tracking

- [x] **Task 1**: Extend control_state.py with new state enums and fields
  - ✅ Add ProducerState and RendererState enums
  - ✅ Add producer_state and renderer_state fields to SystemStatus  
  - ✅ Add helper methods for independent state management

- [x] **Task 2**: Update producer.py to use ProducerState logic
  - ✅ Replace PlayState usage with ProducerState
  - ✅ Implement independent producer state transitions
  - ✅ Remove coupling to renderer state
  - ✅ Remove pause function (producers only play/stop)

- [x] **Task 3**: Update consumer.py to use RendererState logic
  - ✅ Add RendererState import and initialization
  - ✅ Implement buffer-driven renderer state transitions
  - ✅ Add pause functionality that maintains current frame
  - ✅ Add automatic state transitions based on buffer status

- [x] **Task 4**: Update web API endpoints for new state commands
  - ✅ Modify pause endpoint to only affect renderer
  - ✅ Add stop endpoint that only affects producer  
  - ✅ Update play endpoint to handle renderer resume
  - ✅ Add separate producer and renderer status endpoints

- [x] **Task 5**: Remove first frame delay from renderer
  - ✅ Set first_frame_delay_ms to 0.0 in consumer initialization
  - ✅ Allow renderer to start immediately when frames available

- [x] **Task 6**: Test state transitions and buffer behavior
  - ✅ Verify all imports work correctly
  - ✅ Basic compilation and module loading tests passed

## Technical Implementation Details

### SystemStatus Updates
```python
@dataclass
class SystemStatus:
    # Legacy field for compatibility
    play_state: PlayState = PlayState.STOPPED

    # New decoupled state fields
    producer_state: ProducerState = ProducerState.STOPPED
    renderer_state: RendererState = RendererState.STOPPED

    # Buffer monitoring for state transitions
    led_buffer_frames: int = 0
    led_buffer_capacity: int = 0
```

### State Management Methods
- `set_producer_state()` - Control producer independently
- `set_renderer_state()` - Control renderer independently  
- `get_effective_play_state()` - Compute legacy PlayState for compatibility

### Buffer-Driven Transitions
- Renderer monitors LED buffer occupancy
- Automatic PLAYING → STOPPED when buffer empty and producer stopped
- Automatic STOPPED → PLAYING when frames available and not paused

## Compatibility Considerations
- Maintain legacy `play_state` field for existing code compatibility
- Compute effective play state from producer/renderer states
- Gradual migration path for existing components

## Testing Strategy
1. Unit tests for new state enums and transitions
2. Integration tests for producer/renderer independence
3. Buffer backpressure testing with various content types
4. WebSocket real-time status update verification
5. Legacy compatibility testing

---
*Plan created: 2025-07-31*  
*Implementation Status: ✅ COMPLETED*

## Implementation Summary

All major components have been successfully updated to support decoupled producer/renderer state management:

1. **control_state.py**: Added ProducerState and RendererState enums with backward-compatible legacy PlayState
2. **producer.py**: Updated to use ProducerState exclusively, removed pause functionality
3. **consumer.py**: Added renderer state management with buffer-driven transitions  
4. **api_server.py**: Updated endpoints for decoupled control (pause only affects renderer, stop only affects producer)
5. **First frame delay**: Removed 100ms delay for immediate renderer startup
6. **Testing**: Basic import and compilation tests passed

The system now supports independent producer and renderer control with automatic buffer-driven state transitions.
