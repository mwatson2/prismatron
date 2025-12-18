# Consumer Test Coverage Improvement Plan

This document tracks the remaining test coverage improvements needed for `src/consumer/consumer.py`.

## Current Status

As of the last update:
- **Total consumer tests**: 168
- **Unit tests in test_consumer_unit.py**: 131

## Priority 1: Core Frame Processing

These methods are central to the consumer's functionality:

### `_process_frame_optimization` (lines ~1623-1848)
- The main optimization loop
- Tests needed for:
  - Successful optimization with valid frame
  - Handling of optimization failures
  - Timing tracking behavior
  - LED buffer writing after optimization

### `_process_frame_for_batch` (lines ~1850-1917)
- Prepares frames for batch optimization
- Tests needed for:
  - Frame preparation and queuing
  - Batch threshold triggering
  - Metadata handling

### `_process_frame_batch` (line ~2160+)
- Batch processing of multiple frames
- Tests needed for:
  - Multi-frame batch optimization
  - Error handling for batch failures

## Priority 2: Initialization and Configuration

### `initialize()` (lines ~683-793)
- Complex initialization with many components
- Tests needed for:
  - Successful full initialization
  - Partial failure scenarios (one component fails)
  - Configuration validation

### `_check_audio_recording_request` (lines ~531-614)
- Audio recording debug feature
- Tests needed for:
  - Recording start/stop handling
  - File writing behavior

## Priority 3: Main Loop and Control Flow

### `run()` method
- Main event loop
- Tests needed for:
  - Loop iteration behavior
  - Shutdown signal handling
  - Error recovery

### `_run_optimization_loop` / `_run_rendering_loop`
- Core loop implementations
- Tests needed for:
  - Loop state management
  - Frame timing synchronization

## Priority 4: WLED Communication

### WLED reconnection logic
- Tests needed for:
  - Connection loss detection
  - Reconnection attempts
  - Fallback behavior during disconnection

## Priority 5: Edge Cases and Error Handling

### Error recovery paths
- Tests for exception handling in various methods
- Graceful degradation scenarios

### Timing edge cases
- Frame timing edge cases
- Buffer overflow/underflow handling

## Completed Test Coverage

### State Transitions
- RendererState transitions (STOPPED, WAITING, PLAYING, PAUSED)
- ProducerState handling
- State change callbacks

### Audio Beat Detection
- BeatEvent callbacks
- BuildDropEvent callbacks
- Audio level updates

### Frame Processing Basics
- Single frame processing flow
- Frame gap tracking
- LED transition effects

### Lifecycle Management
- Consumer start/stop
- Resource cleanup
- Component initialization order

## Notes

- The AdaptiveFrameDropper was removed (it was not successful) - no tests needed for it
- Focus on testing the happy path and failure logging, not fallbacks (per CLAUDE.md guidelines)
