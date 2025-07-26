# Playlist Transitions Implementation Plan

This document describes the implementation plan for adding general-purpose transitions between playlist items in the Prismatron LED Display System.

## Status Tracking

**Current Phase**: Phase 4 Complete - UI and API Implemented  
**Last Updated**: 2025-01-25  
**Overall Progress**: 80% (Phases 1-4 complete)

### Phase Status
- [x] **Phase 1: Core Infrastructure** (100%) - ✅ **COMPLETE**
  - [x] Base transition interface (`src/transitions/base_transition.py`)
  - [x] Fade transition implementation (`src/transitions/fade_transition.py`)
  - [x] Transition factory and registry (`src/transitions/transition_factory.py`)
  - [x] Playlist data model updates (`src/core/playlist_sync.py`)
- [x] **Phase 2: Producer Integration** (100%) - ✅ **COMPLETE**
  - [x] Updated shared buffer metadata structure (`src/const.py`)
  - [x] Enhanced PlaylistItem with transition configurations (`src/producer/producer.py`)
  - [x] Updated playlist synchronization to preserve transitions (`src/producer/producer.py`)
  - [x] Added transition metadata to frame data during rendering (`src/producer/producer.py`)
- [x] **Phase 3: Consumer Integration** (100%) - ✅ **COMPLETE**
  - [x] Created transition processor component (`src/consumer/transition_processor.py`)
  - [x] Integrated transition processing in consumer pipeline (`src/consumer/consumer.py`)
  - [x] Added metadata extraction for transition configuration (`src/consumer/consumer.py`)
  - [x] Applied transitions before LED optimization in frame processing loop
- [x] **Phase 4: UI and API** (100%) - ✅ **COMPLETE**
  - [x] Updated playlist API models with transition configuration (`src/web/api_server.py`)
  - [x] Added transition validation and schema support to API (`src/web/api_server.py`)
  - [x] Created transition types endpoint (`/api/transitions`)
  - [x] Added transition update endpoint (`/api/playlist/{item_id}/transitions`)
  - [x] Created TransitionConfig React component (`src/web/frontend/src/components/TransitionConfig.jsx`)
  - [x] Integrated transition UI into PlaylistPage with configuration modal
- [ ] **Phase 5: Testing and Polish** (0%)

### Current Task
- **COMPLETED**: Phase 4 - UI and API  
- **NEXT**: Phase 5 - Testing and Polish

### Work Interruption Notes
*Instructions: When work is interrupted, update this section with:*
- *What was being worked on*
- *Current file/function being modified*
- *Next specific steps to take*
- *Any blockers or design decisions pending*
- *Test status and any failing tests*

**Last Work Session**: Phase 4 Implementation (2025-01-25)  
**Files Modified**: 
- **Phase 1 (Core Infrastructure)**:
  - Created: `src/transitions/__init__.py`
  - Created: `src/transitions/base_transition.py` 
  - Created: `src/transitions/fade_transition.py`
  - Created: `src/transitions/transition_factory.py`
  - Modified: `src/core/playlist_sync.py` (added TransitionConfig and playlist item transitions)
- **Phase 2 (Producer Integration)**:
  - Modified: `src/const.py` (added transition fields to METADATA_DTYPE)
  - Modified: `src/producer/producer.py` (enhanced PlaylistItem, sync handling, frame metadata)
- **Phase 3 (Consumer Integration)**:
  - Created: `src/consumer/transition_processor.py` (transition processing component)
  - Modified: `src/consumer/consumer.py` (integrated transitions in frame processing pipeline)
- **Phase 4 (UI and API)**:
  - Modified: `src/web/api_server.py` (added TransitionConfig model, validation, endpoints)
  - Created: `src/web/frontend/src/components/TransitionConfig.jsx` (transition configuration modal)
  - Modified: `src/web/frontend/src/pages/PlaylistPage.jsx` (integrated transition UI controls)

**Next Steps**: 
1. Write comprehensive unit tests for transition components
2. Add integration tests for end-to-end transition functionality
3. Performance testing and optimization
4. Documentation and user guides

**Pending Decisions**: None  
**Blockers**: None  
**Tests**: All modules compile successfully, frontend builds without errors  

---

## Overview

The transitions feature allows configurable visual transitions between adjacent playlist items. The system supports:
- **Transition Types**: "none" (default), "fade", with extensibility for future types
- **Per-Item Configuration**: Each playlist item has `transition_in` and `transition_out` settings
- **Parameter Support**: Each transition type can have configurable parameters (e.g., fade duration)
- **Frame-Level Processing**: Transitions are applied to individual frames in the consumer pipeline

## Data Flow Architecture

### Transition Configuration Propagation

```
Playlist Definition → Producer → Shared Buffer → Consumer → Transition Processing
```

#### 1. **Playlist Storage** (`src/core/playlist.py`)
- Playlist items store transition configuration in JSON format
- Each item has `transition_in` and `transition_out` objects
- Configuration includes `type` and `parameters` fields

#### 2. **Producer Processing** (`src/producer/`)
- Producer reads playlist and calculates transition timing
- For each frame, determines:
  - Current item's transition configuration
  - Frame position within item (timestamp relative to item start)
  - Item duration for transition calculations
  - Whether frame is in transition region

#### 3. **Frame Metadata Enhancement**
- Producer adds transition data to frame metadata before writing to shared buffer
- Metadata includes:
  ```python
  frame_metadata.transition_config = {
      "transition_in": {"type": "fade", "parameters": {"duration": 1.0}},
      "transition_out": {"type": "none", "parameters": {}},
      "item_timestamp": 2.5,  # Time within current item
      "item_duration": 10.0,  # Total item duration
      "item_index": 3         # For debugging/logging
  }
  ```

#### 4. **Shared Buffer Transport**
- Transition configuration travels with frame data through existing shared memory buffer
- No changes needed to buffer structure - uses existing metadata system
- Consumer receives complete transition context for each frame

#### 5. **Consumer Processing** (`src/consumer/`)
- Consumer extracts transition config from frame metadata
- Applies transition processing before LED optimization
- Uses timestamp and duration to determine transition strength/progress

## Implementation Plan

### Phase 1: Core Infrastructure

#### **1.1 Data Model Changes**
- **File**: `src/core/playlist.py`
- **Task**: Add transition fields to playlist item structure
- **Details**:
  ```python
  class PlaylistItem:
      # ... existing fields ...
      transition_in: TransitionConfig = field(default_factory=lambda: TransitionConfig())
      transition_out: TransitionConfig = field(default_factory=lambda: TransitionConfig())
  
  @dataclass
  class TransitionConfig:
      type: str = "none"
      parameters: Dict[str, Any] = field(default_factory=dict)
  ```
- **Status**: Not started

#### **1.2 Base Transition Interface**
- **File**: `src/transitions/base_transition.py` (new)
- **Task**: Create abstract base class for transitions
- **Details**:
  ```python
  class BaseTransition:
      def apply_transition(self, frame: np.ndarray, timestamp: float, 
                          item_duration: float, transition_config: dict, 
                          direction: str) -> np.ndarray
      def get_transition_region(self, item_duration: float, 
                              transition_config: dict, direction: str) -> Tuple[float, float]
      def validate_parameters(self, parameters: dict) -> bool
  ```
- **Status**: Not started

#### **1.3 Fade Transition Implementation**
- **File**: `src/transitions/fade_transition.py` (new)
- **Task**: Implement fade in/out transition
- **Details**:
  - Calculate fade factor based on timestamp and duration
  - Apply brightness scaling to RGB frame data
  - Support linear and eased interpolation curves
  - Handle edge cases (zero duration, invalid timestamps)
- **Status**: Not started

#### **1.4 Transition Factory**
- **File**: `src/transitions/transition_factory.py` (new)
- **Task**: Registry and factory for transition types
- **Details**:
  - Register available transition implementations
  - Create instances based on type string
  - Validate parameters against transition schemas
  - Handle "none" transition as pass-through
- **Status**: Not started

### Phase 2: Producer Integration

#### **2.1 Playlist Manager Updates**
- **File**: `src/producer/playlist_manager.py`
- **Task**: Add transition configuration loading and validation
- **Details**:
  - Load transition config from playlist files  
  - Validate transition parameters during playlist load
  - Handle backward compatibility with existing playlists
  - Add default "none" transitions to items without config
- **Status**: Not started

#### **2.2 Frame Metadata Enhancement**
- **File**: Producer frame generation code
- **Task**: Add transition metadata to frame data
- **Details**:
  - Calculate item timestamp and duration for each frame
  - Determine current item's transition configuration
  - Add transition context to frame metadata
  - Ensure metadata survives shared buffer transport
- **Status**: Not started

### Phase 3: Consumer Integration

#### **3.1 Transition Processor Component**
- **File**: `src/consumer/transition_processor.py` (new)
- **Task**: Core transition processing engine
- **Details**:
  - Extract transition config from frame metadata
  - Determine if frame is in transition region
  - Apply appropriate transition using factory
  - Handle both in and out transitions
  - Optimize for performance (avoid unnecessary processing)
- **Status**: Not started

#### **3.2 Consumer Pipeline Integration**
- **File**: `src/consumer/consumer.py`
- **Task**: Integrate transition processing in optimization loop
- **Details**:
  - Add transition processor to consumer initialization
  - Call transition processing in `_process_frame_optimization()`
  - Apply transitions after frame validation, before LED optimization
  - Handle transition errors gracefully
  - Add transition timing to performance metrics
- **Status**: Not started

### Phase 4: UI and API

#### **4.1 Playlist Data API Updates**
- **File**: `src/web/api_server.py`
- **Task**: Add transition configuration to playlist APIs
- **Details**:
  - Update playlist item endpoints to handle transition config
  - Add validation for transition parameters
  - Return available transition types and schemas
  - Handle migration of existing playlists
- **Status**: Not started

#### **4.2 Web Interface Updates**
- **Files**: Web frontend playlist components
- **Task**: Add transition configuration UI
- **Details**:
  - Transition configuration panel for each playlist item
  - Dropdown selector for transition types
  - Dynamic parameter inputs based on transition type
  - Preview/visualization capabilities
  - Form validation and error display
- **Status**: Not started

### Phase 5: Testing and Polish

#### **5.1 Unit Tests**
- **Files**: `tests/transitions/`
- **Task**: Comprehensive test coverage
- **Details**:
  - Test individual transition implementations
  - Test transition factory and registry
  - Test parameter validation and edge cases
  - Performance benchmarks
- **Status**: Not started

#### **5.2 Integration Tests**
- **Files**: `tests/consumer/`, `tests/producer/`
- **Task**: End-to-end transition testing
- **Details**:
  - Test metadata propagation through pipeline
  - Visual output validation
  - Performance impact measurement
  - Error handling and recovery
- **Status**: Not started

#### **5.3 Documentation**
- **Task**: User and developer documentation
- **Details**:
  - Transition configuration guide
  - Available transition types reference
  - Developer guide for new transition types
  - Performance best practices
- **Status**: Not started

## Technical Details

### Frame Metadata Structure

The producer will enhance frame metadata with transition information:

```python
# Added to existing frame metadata
transition_context = {
    "transition_in": {
        "type": "fade",
        "parameters": {"duration": 1.0}
    },
    "transition_out": {
        "type": "none", 
        "parameters": {}
    },
    "item_timestamp": 2.5,    # Seconds from item start
    "item_duration": 10.0,    # Total item duration in seconds
    "item_index": 3,          # For debugging
    "item_id": "video_001"    # For debugging
}
```

### Transition Processing Flow

```python
# In consumer _process_frame_optimization()
def process_frame_with_transitions(frame, metadata):
    # Extract transition context
    transition_ctx = metadata.get('transition_context')
    if not transition_ctx:
        return frame  # No transition processing needed
    
    # Apply transition_in if in region
    if is_in_transition_in_region(transition_ctx):
        frame = apply_transition(frame, transition_ctx, 'in')
    
    # Apply transition_out if in region  
    if is_in_transition_out_region(transition_ctx):
        frame = apply_transition(frame, transition_ctx, 'out')
    
    return frame
```

### Performance Considerations

- **Conditional Processing**: Skip transition processing when not in transition regions
- **Memory Efficiency**: Modify frames in-place when possible
- **Caching**: Pre-calculate transition parameters where beneficial
- **Monitoring**: Track transition processing time in performance metrics

### Future Extensions

- **Additional Transition Types**: wipe, cross-fade, custom effects
- **Advanced Parameters**: curve types, multi-dimensional parameters
- **Audio-Reactive**: Transitions synchronized to audio analysis
- **Multi-Item**: Cross-fade between adjacent items (major enhancement)

## Configuration Examples

### Playlist JSON Structure
```json
{
  "items": [
    {
      "type": "video",
      "path": "video1.mp4",
      "duration": 30.0,
      "transition_in": {
        "type": "fade",
        "parameters": {"duration": 2.0}
      },
      "transition_out": {
        "type": "fade", 
        "parameters": {"duration": 1.5}
      }
    },
    {
      "type": "image",
      "path": "image1.jpg", 
      "duration": 10.0,
      "transition_in": {
        "type": "none",
        "parameters": {}
      },
      "transition_out": {
        "type": "none",
        "parameters": {}
      }
    }
  ]
}
```

### Fade Transition Parameters
```json
{
  "type": "fade",
  "parameters": {
    "duration": 2.0,        # Duration in seconds
    "curve": "linear",      # "linear", "ease-in", "ease-out", "ease-in-out"
    "min_brightness": 0.0   # Minimum brightness (0.0 = full fade to black)
  }
}
```

## Migration Strategy

1. **Backward Compatibility**: Existing playlists load with default "none" transitions
2. **Graceful Degradation**: Invalid transition configs fall back to "none"
3. **Progressive Enhancement**: UI shows transition options only for supported items
4. **Version Detection**: Playlist format version indicates transition support

## Error Handling

- **Invalid Transition Types**: Fall back to "none" with warning
- **Invalid Parameters**: Use defaults with validation error logging
- **Processing Errors**: Skip transition processing, log error, continue pipeline
- **Performance Issues**: Monitor transition processing time, alert if excessive

---

*Last Updated: 2025-01-24*  
*Document Version: 1.0*