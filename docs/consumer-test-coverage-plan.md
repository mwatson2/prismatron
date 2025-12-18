# Consumer Test Coverage Improvement Plan

This document tracks the remaining test coverage improvements needed for `src/consumer/consumer.py`.

## Current Status

As of the last update:
- **Total consumer tests**: 193
- **Unit tests in test_consumer_unit.py**: 156

## Priority 1: Core Frame Processing ✅ COMPLETED

These methods are central to the consumer's functionality:

### `_process_frame_optimization` (lines ~1623-1848) ✅
- 12 tests added in TestProcessFrameOptimization
- Covers: successful processing, late frame dropping, first frame handling,
  cascade detection, batch mode routing, brightness scaling

### `_process_frame_for_batch` (lines ~1850-1917) ✅
- 4 tests added in TestProcessFrameForBatch
- Covers: frame accumulation, batch triggering, metadata preservation

### `_process_frame_batch` (line ~2160+) ✅
- 4 tests added in TestProcessFrameBatch
- Covers: batch optimization, batch clearing, stats updates, exception handling

## Priority 2: Initialization and Configuration

### `initialize()` (lines ~683-793) - Partially covered
- 5 tests added in TestInitializeComponents
- Complex method with many dependencies - basic component tests added
- Full integration testing deferred due to complexity

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

### Core Frame Processing (NEW)
- `_process_frame_optimization`: 12 tests
- `_process_frame_for_batch`: 4 tests
- `_process_frame_batch`: 4 tests
- `_process_single_frame`: 4 tests (existing)

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

### Component Initialization (NEW)
- Frame consumer connection setup
- Control state initialization
- LED optimizer initialization
- Frame renderer initialization checks

## Notes

- The AdaptiveFrameDropper was removed (it was not successful) - no tests needed for it
- Focus on testing the happy path and failure logging, not fallbacks (per CLAUDE.md guidelines)
