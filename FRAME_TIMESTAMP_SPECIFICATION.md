# Frame Timestamp Handling Specification

## Overview

This specification details the implementation of precise frame timestamp handling to decouple frame production, LED optimization, and rendering processes. The goal is to achieve smooth, jitter-free LED display with proper temporal accuracy by introducing timestamp-based rendering and buffering between optimization and display output.

## Current Architecture Analysis

### Producer Process (`src/producer/producer.py`)

**Current Implementation:**
- Lines 314-316: Uses target FPS (`_target_fps = 30.0`) for frame timing
- Line 512-514: Frame rate limiting with `_frame_interval = 1.0 / _target_fps`
- Line 625: Sets `presentation_timestamp=frame_data.presentation_timestamp` in `get_write_buffer()`
- Lines 521-523: Updates `_frames_produced` and `_last_frame_time`

**Current Behavior:**
- Produces frames at fixed intervals based on content FPS
- Uses content source's `presentation_timestamp` from `FrameData` object
- Blocks on ring buffer when consumer can't keep up
- Free-running within FPS constraints

### Consumer Process (`src/consumer/consumer.py`)

**Current Implementation:**
- Lines 272-277: LED optimization using `_led_optimizer.optimize_frame()`
- Lines 294-299: Immediate WLED transmission after optimization
- Lines 236-239: Frame rate limiting with `target_frame_time = 1.0 / self.target_fps`
- Line 117: Default `target_fps = 15.0`

**Current Behavior:**
- Processes frames in tight loop with rate limiting
- Optimizes and immediately renders each frame
- No timestamp consideration for rendering timing
- No buffering between optimization and display

### Ring Buffer (`src/core/shared_buffer.py`)

**Current Implementation:**
- Supports `presentation_timestamp` metadata in buffer info
- Provides blocking/non-blocking read/write operations
- Handles frame metadata including source dimensions

## Proposed Architecture Changes

### 1. Timestamp-Based Rendering System

#### 1.1 Renderer Class Design

**New Component: `src/consumer/frame_renderer.py`**

```python
class FrameRenderer:
    """
    Timestamp-based frame renderer that handles precise timing for LED display.

    Features:
    - Establishes wallclock delta from first frame timestamp
    - Renders frames at their designated timestamps
    - Handles late/early frame timing
    - Supports multiple output targets (WLED, test renderer)
    """

    def __init__(self,
                 first_frame_delay_ms: float = 100.0,
                 timing_tolerance_ms: float = 5.0):
        self.first_frame_delay = first_frame_delay_ms / 1000.0
        self.timing_tolerance = timing_tolerance_ms / 1000.0
        self.wallclock_delta = None  # Established from first frame
        self.first_frame_received = False

        # Output targets
        self.wled_client = None
        self.test_renderer = None

        # Statistics
        self.frames_rendered = 0
        self.late_frames = 0
        self.early_frames = 0
```

#### 1.2 LED Values Buffer

**New Component: `src/consumer/led_buffer.py`**

```python
class LEDBuffer:
    """
    Ring buffer for optimized LED values with timestamp metadata.

    Stores small LED arrays (3 * 2624 bytes) instead of full frames,
    allowing deep buffering to absorb jitter in optimization process.
    """

    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.led_arrays = np.zeros((buffer_size, LED_COUNT, 3), dtype=np.uint8)
        self.timestamps = np.zeros(buffer_size, dtype=np.float64)
        self.metadata = [None] * buffer_size

        # Ring buffer state
        self.write_index = 0
        self.read_index = 0
        self.count = 0
        self.lock = threading.RLock()
```

### 2. Modified Consumer Architecture

#### 2.1 Consumer Process Threading Model

**Updated `src/consumer/consumer.py`:**

The consumer process will use two threads within a single process:

1. **Optimization Thread** (existing main loop, modified):
   - Reads frames from shared memory ring buffer
   - Performs LED optimization as fast as possible
   - Writes optimized LED values to LED buffer with timestamps
   - No frame rate limiting

2. **Renderer Thread** (new):
   - Reads LED values from LED buffer
   - Implements timestamp-based rendering logic
   - Outputs to WLED client and/or test renderer
   - Handles timing precision and jitter compensation

#### 2.2 Implementation Changes

**Modified Consumer Process Structure:**

```python
class ConsumerProcess:
    def __init__(self, ...):
        # Existing components
        self._led_optimizer = LEDOptimizer(...)
        self._wled_client = WLEDClient(...)

        # New components
        self._led_buffer = LEDBuffer(buffer_size=100)
        self._frame_renderer = FrameRenderer(
            first_frame_delay_ms=100.0,
            timing_tolerance_ms=5.0
        )

        # Threading
        self._optimization_thread = None
        self._renderer_thread = None

    def start(self):
        # Start both threads
        self._optimization_thread = threading.Thread(
            target=self._optimization_loop,
            name="OptimizationThread"
        )
        self._renderer_thread = threading.Thread(
            target=self._rendering_loop,
            name="RendererThread"
        )

        self._optimization_thread.start()
        self._renderer_thread.start()
```

### 3. Timestamp Handling Logic

#### 3.1 Wallclock Delta Establishment

**First Frame Processing:**

```python
def _establish_wallclock_delta(self, first_timestamp: float) -> None:
    """
    Establish fixed delta between frame timestamps and wallclock time.

    Args:
        first_timestamp: Presentation timestamp of first frame
    """
    current_wallclock = time.time()

    # Add default delay for buffering
    target_wallclock = current_wallclock + self.first_frame_delay

    # Calculate delta: wallclock_time = frame_timestamp + delta
    self.wallclock_delta = target_wallclock - first_timestamp

    self.first_frame_received = True

    logger.info(f"Established wallclock delta: {self.wallclock_delta:.3f}s "
                f"(first frame delay: {self.first_frame_delay:.3f}s)")
```

#### 3.2 Frame Rendering Timing

**Rendering Loop Logic:**

```python
def _render_frame_at_timestamp(self, led_values: np.ndarray,
                              frame_timestamp: float) -> None:
    """
    Render frame at its designated timestamp with timing logic.

    Args:
        led_values: Optimized LED values to display
        frame_timestamp: Original presentation timestamp from producer
    """
    # Calculate target wallclock time
    target_wallclock = frame_timestamp + self.wallclock_delta
    current_wallclock = time.time()

    # Time difference (negative = early, positive = late)
    time_diff = current_wallclock - target_wallclock

    if time_diff > self.timing_tolerance:
        # Late frame - render immediately
        self.late_frames += 1
        logger.debug(f"Late frame: {time_diff*1000:.1f}ms")
        self._send_to_outputs(led_values)

    elif time_diff < -self.timing_tolerance:
        # Early frame - wait until target time
        wait_time = -time_diff
        self.early_frames += 1
        logger.debug(f"Early frame: waiting {wait_time*1000:.1f}ms")
        time.sleep(wait_time)
        self._send_to_outputs(led_values)

    else:
        # On time - render immediately
        self._send_to_outputs(led_values)

    self.frames_rendered += 1
```

### 4. Integration Points

#### 4.1 Producer Changes (Minimal)

**No changes required** - producers already set `presentation_timestamp`:

- Line 625 in `producer.py`: `presentation_timestamp=frame_data.presentation_timestamp`
- Content sources provide timestamps through `FrameData.presentation_timestamp`

#### 4.2 Consumer Optimization Loop (Modified)

**Updated `_process_frame()` in `consumer.py`:**

```python
def _process_frame(self, buffer_info) -> None:
    """
    Process frame for LED optimization only - no rendering.
    Rendering handled by separate renderer thread.
    """
    try:
        # Extract frame and timestamp
        frame_array = buffer_info.get_array()
        timestamp = buffer_info.presentation_timestamp or time.time()

        # Validate and prepare frame
        rgb_frame = frame_array[:, :, :3].astype(np.uint8)
        if self.brightness_scale != 1.0:
            rgb_frame = (rgb_frame * self.brightness_scale).clip(0, 255).astype(np.uint8)

        # Optimize LED values (no timing constraints)
        result = self._led_optimizer.optimize_frame(rgb_frame, max_iterations=50)

        # Store in LED buffer with timestamp
        led_values_uint8 = result.led_values.astype(np.uint8)
        self._led_buffer.write_led_values(led_values_uint8, timestamp, {
            'optimization_time': time.time() - start_time,
            'converged': result.converged,
            'iterations': result.iterations
        })

        # Update statistics (optimization only)
        self._update_optimization_stats(result)

    except Exception as e:
        logger.error(f"Error in optimization loop: {e}")
```

#### 4.3 New Renderer Loop

**New `_rendering_loop()` in `consumer.py`:**

```python
def _rendering_loop(self) -> None:
    """
    Dedicated rendering thread that handles timestamp-based frame output.
    """
    logger.info("Renderer thread started")

    while self._running and not self._shutdown_requested:
        try:
            # Get next LED values with timeout
            led_data = self._led_buffer.read_led_values(timeout=0.1)

            if led_data is None:
                continue  # Timeout - no data available

            led_values, timestamp, metadata = led_data

            # Handle first frame timing establishment
            if not self._frame_renderer.first_frame_received:
                self._frame_renderer._establish_wallclock_delta(timestamp)

            # Render with timestamp-based timing
            self._frame_renderer._render_frame_at_timestamp(led_values, timestamp)

        except Exception as e:
            logger.error(f"Error in rendering loop: {e}")
            time.sleep(0.01)

    logger.info("Renderer thread ended")
```

### 5. Performance Characteristics

#### 5.1 Memory Usage

**LED Buffer Memory:**
- LED values: `100 frames × 2624 LEDs × 3 channels × 1 byte = 787KB`
- Timestamps: `100 frames × 8 bytes = 800 bytes`
- Total: ~800KB for 100-frame buffer (vs ~375MB for equivalent frame buffer)

#### 5.2 Timing Precision

**Expected Performance:**
- Frame timestamp precision: ±1ms (limited by content source timing)
- Rendering timing precision: ±5ms (configurable tolerance)
- Buffer depth: 100 frames = ~3.3s at 30fps for jitter absorption
- Maximum optimization jitter tolerance: ~3 seconds before frame drops

#### 5.3 Threading Benefits

**Optimization Thread:**
- Runs without timing constraints
- Can use full GPU/CPU resources for optimization
- No blocking on WLED transmission
- Batch optimization opportunities for future enhancements

**Renderer Thread:**
- Dedicated timing precision for display output
- Handles network jitter from WLED transmission
- Clean separation of concerns
- Consistent display timing regardless of optimization speed

### 6. Configuration Parameters

#### 6.1 Timing Configuration

```python
# Renderer timing settings
FIRST_FRAME_DELAY_MS = 100.0     # Default delay for first frame
TIMING_TOLERANCE_MS = 5.0        # Acceptable timing deviation
LATE_FRAME_LOG_THRESHOLD = 50.0  # Log late frames above this threshold

# Buffer sizes
LED_BUFFER_SIZE = 100            # Number of LED value frames to buffer
FRAME_BUFFER_SIZE = 8            # Existing shared memory buffer size
```

#### 6.2 Performance Tuning

```python
# Optimization settings (per frame)
MAX_OPTIMIZATION_ITERATIONS = 50  # No time constraints
CONVERGENCE_THRESHOLD = 1e-3

# Renderer thread priority (if supported)
RENDERER_THREAD_PRIORITY = "high"
```

### 7. Monitoring and Statistics

#### 7.1 New Statistics Tracking

**Renderer Statistics:**
- Frames rendered count
- Late frame count and average latency
- Early frame count and average wait time
- Timing accuracy distribution
- Buffer utilization metrics

**Optimization Statistics:**
- Optimization rate (fps) independent of display rate
- Average optimization time per frame
- GPU utilization during optimization

#### 7.2 Integration with Web Interface

**New WebSocket metrics:**
- `renderer_stats`: Timing accuracy, late/early frame counts
- `led_buffer_stats`: Buffer utilization, depth, overflow events
- `timestamp_delta`: Current wallclock delta value
- `timing_health`: Overall timing system health indicator

### 8. Migration Path

#### 8.1 Implementation Order

1. **Phase 1**: Implement `LEDBuffer` and `FrameRenderer` classes
2. **Phase 2**: Add renderer thread to `ConsumerProcess`
3. **Phase 3**: Modify optimization loop to write to LED buffer
4. **Phase 4**: Remove WLED transmission from optimization loop
5. **Phase 5**: Add statistics and monitoring integration
6. **Phase 6**: Performance tuning and optimization

#### 8.2 Backward Compatibility

Not required. This new mode will be the default frame timing mode.

### 9. Future Enhancements

#### 9.1 Batch Optimization

With separate LED buffer, future optimizations can include:
- Batch processing of multiple frames
- GPU kernel batching for improved utilization
- Temporal coherence optimization across frame sequences

#### 9.2 Adaptive Timing

- Dynamic adjustment of timing tolerance based on system performance
- Automatic buffer size adjustment based on optimization speed
- Frame drop/duplicate policies for extreme timing conditions

This specification provides a comprehensive framework for implementing precise timestamp-based rendering while maintaining compatibility with the existing system architecture.
