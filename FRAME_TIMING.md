# Frame Timing System

This document describes the frame timing system for debugging frame pipeline performance and timing issues.

## Overview

The frame timing system tracks detailed timestamps for each frame as it moves through the processing pipeline:

1. **Plugin Timestamp** - 0-based timestamp from content plugin (e.g., video file position)
2. **Producer Timestamp** - Global presentation timestamp (plugin timestamp + global offset)
3. **Write to Buffer Time** - When frame is written to shared memory buffer
4. **Read from Buffer Time** - When frame is read from shared memory buffer
5. **Write to LED Buffer Time** - When optimized LED data is written to LED buffer
6. **Read from LED Buffer Time** - When LED data is read from LED buffer
7. **Render Time** - When frame is actually sent to hardware
8. **Item Duration** - Duration of current playlist item

## Usage

### Enable Timing Logging

To enable timing logging, pass the `timing_log_path` parameter when creating a `ConsumerProcess`:

```python
consumer = ConsumerProcess(
    buffer_name="prismatron_buffer",
    control_name="prismatron_control",
    timing_log_path="/path/to/timing_log.csv"
)
```

This will create a CSV file with detailed timing information for each frame.

### CSV Format

The CSV file contains the following columns:

```csv
frame_index,plugin_timestamp,producer_timestamp,write_to_buffer_time,read_from_buffer_time,write_to_led_buffer_time,read_from_led_buffer_time,render_time,item_duration
```

### Visualization

Use the visualization script to create timing graphs:

```bash
# Basic usage
./scripts/visualize_frame_timing.py timing_log.csv

# With custom output directory
./scripts/visualize_frame_timing.py timing_log.csv -o /path/to/output/

# Limit number of frames for performance
./scripts/visualize_frame_timing.py timing_log.csv -m 500

# Print statistics only (no graphs)
./scripts/visualize_frame_timing.py timing_log.csv -s
```

### Generated Visualizations

The script creates two main visualizations:

1. **Timeline Graph** - Shows frame progression through pipeline
   - Top plot: Plugin and producer timestamps (content timeline)
   - Bottom plot: Wallclock processing times (pipeline timing)
   - Y-axis: Frame index (1 pixel per frame)
   - X-axis: Time (seconds)
   - Different colors for each pipeline stage

2. **Latency Analysis** - Histograms of processing delays
   - Shared Buffer Latency: Time in shared memory
   - Optimization Latency: LED optimization time
   - LED Buffer Latency: Time in LED buffer
   - Render Latency: Hardware communication time
   - End-to-End Latency: Total pipeline time

## Testing

Run the test script to verify the system works correctly:

```bash
./scripts/test_frame_timing.py
```

This creates sample data and tests both logging and visualization functionality.

## Performance Analysis

### Key Metrics

The system provides these timing metrics:

- **Buffer Latency** - Time between writing to and reading from shared buffer
- **Processing Latency** - Time for LED optimization (largest component)
- **LED Buffer Latency** - Time in LED buffer (should be minimal)
- **Render Latency** - Time to send data to hardware
- **End-to-End Latency** - Total time from buffer write to render

### Interpreting Results

- **High Buffer Latency** - Consumer not keeping up with producer
- **High Processing Latency** - LED optimization taking too long
- **High LED Buffer Latency** - Renderer not keeping up with optimizer
- **High Render Latency** - Hardware communication issues
- **Frame Rate Mismatch** - Compare plugin vs render frame rates

### Troubleshooting

Common issues and solutions:

1. **Frames dropping early** - Look for gaps in frame indices
2. **Buffer overflow** - High buffer latency, increase buffer size
3. **Optimization bottleneck** - High processing latency, optimize LED solver
4. **Renderer falling behind** - High LED buffer latency, check WLED communication
5. **Network issues** - High render latency, check WLED connectivity

## Implementation Details

The timing system uses:

- `FrameTimingData` class to store timestamps
- Timing data attached to frame metadata through pipeline
- `FrameTimingLogger` for CSV output
- Visualization script with matplotlib for analysis

Timing data flows:
1. Producer creates timing object and marks write time
2. Consumer reads timing object and marks read/optimization times  
3. Renderer marks final render time and logs to CSV
4. Visualization script analyzes CSV data

## Example Output

```
=== Frame Timing Statistics ===
Total frames analyzed: 1000
Frame index range: 1 - 1000
Content duration range: 0.000s - 33.333s

=== Processing Latencies (milliseconds) ===
Shared Buffer  : Mean=   2.1ms, Std=   1.2ms, P50=   1.8ms, P95=   4.2ms, P99=   6.1ms
Optimization   : Mean=  45.3ms, Std=  12.1ms, P50=  43.2ms, P95=  68.5ms, P99=  89.2ms
LED Buffer     : Mean=   0.8ms, Std=   0.4ms, P50=   0.7ms, P95=   1.5ms, P99=   2.1ms
Render         : Mean=   8.2ms, Std=   2.3ms, P50=   7.9ms, P95=  12.1ms, P99=  15.8ms
End-to-End     : Mean=  56.4ms, Std=  13.2ms, P50=  54.1ms, P95=  79.8ms, P99= 103.2ms

=== Frame Rates ===
Plugin frame rate: 30.0 fps (interval: 33.3ms)
Render frame rate: 28.5 fps (interval: 35.1ms)
```
