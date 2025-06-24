# PerformanceTiming proposal

This proposal describes a PerformanceTiming class with a simple API. The purpose of this class is to provide a lightweight way to annotate code for timing and performance measurements with minimal code "in-line". It should facilitate consistent millisecond timing of operations, calculation of memory bandwidth and FLOPS per second. It must support nested sections for measurement, repeated execution of the same sections and separate reporting of the first from subsequent occurrences of a section.

## PerformanceTiming usage

### Basic Usage
```python
timing = PerformanceTiming("LED Optimization Module")

# ... code ...

timing and timing.start("(A^T)b operation (mixed matrix)")
# ... code for the operation
timing and timing.stop("(A^T)b operation (mixed matrix)", read=1024000, written=234544, flops=4556334)

# ... more code ...

timing and timing.log(logger)
```

### Context Manager Usage
```python
timing = PerformanceTiming("LED Optimization Module")

with timing.section("matrix_multiply", flops=12345, read=1024000) as t:
    # ... operation code
    t.add_memory(written=234544)  # Add additional memory info during operation

# Alternative with automatic FLOPS calculation
with timing.section("einsum", operation_type="einsum", shape_info={"A": (1000,1000,3), "B": (1000,3)}) as t:
    # ... einsum operation
    pass  # FLOPS calculated automatically based on operation type and shapes
```

### Advanced Features
```python
# GPU timing with CUDA events for accuracy
timing.start("matrix_multiply", use_gpu_events=True, flops=4556334)
# ... CUDA operations
timing.stop("matrix_multiply")

# Memory transfer tracking
timing.start("data_transfer")
timing.record_memory_transfer(size_mb=50, direction="cpu_to_gpu")
timing.stop("data_transfer")

# Providing metrics up front (useful when known in advance)
timing.start("precomputed_operation", read=1024000, written=512000, flops=8888888)
# ... operation
timing.stop("precomputed_operation")  # Metrics already provided
```

## PerformanceTiming features

### Core Features
- **Sections** are identified by the same title text appearing in start() and stop() operations
- **Null-safe pattern**: Standard usage pattern is to write `timing and timing.method()` so that if `timing` is None this will be a noop
- **Flexible metrics**: Read bytes, written bytes and FLOPS can be provided on either start() or stop() operations
- **Nested sections**: Full support for nested timing sections with proper indentation in reports
- **GPU synchronization**: The start and stop operations will perform GPU sync when use_gpu_events=True
- **Repeat tracking**: Separate reporting of first vs. subsequent occurrences of the same section

### GPU Integration
- **CUDA event timing**: More accurate timing for GPU operations using `cp.cuda.Event()`
- **GPU memory bandwidth**: Distinguish between GPU memory operations vs. PCIe transfers
- **Memory access patterns**: Track sequential vs. random memory access patterns
- **Automatic GPU sync**: Optional GPU synchronization for accurate timing boundaries

### Memory and Performance Tracking
- **Memory bandwidth calculations**: 
  - GPU memory throughput (GB/s for on-device operations)
  - PCIe transfer rates (for CPU↔GPU transfers)
  - Memory access efficiency tracking
- **FLOPS calculations**:
  - Manual specification of FLOPS counts
  - Automatic calculation for common operations (einsum, matmul, etc.)
  - Per-operation type FLOPS estimation based on tensor shapes
- **Error handling**: Graceful handling of timing failures with optional fallback

### Reporting Features
The log method will report on each section, followed by indented reports on nested sections:

**For each section:**
- Section Name
- Number of occurrences  
- Total duration in section
- Percentage of total time (when include_percentages=True)
- Total read throughput (read bytes over total duration)
- Total write throughput (written bytes over total duration) 
- FLOPS per second (total FLOPS over total duration)
- GPU memory bandwidth (when applicable)
- PCIe transfer rates (when applicable)

**For repeated sections:**
- All of the above metrics for the first occurrence
- Averages of the above metrics across remaining occurrences
- Standard deviation of timing across occurrences

### Enhanced Reporting Options
```python
# Enhanced logging with sorting and percentages
timing.log(logger, include_percentages=True, sort_by='time')
timing.log(logger, include_percentages=True, sort_by='flops')

# Export to structured format
timing_data = timing.get_timing_data()  # Returns dict for further analysis
timing.export_csv("performance_report.csv")
```

### Error Handling and Robustness
```python
# Safe initialization pattern for optional profiling
timing = PerformanceTiming("Module") if ENABLE_PROFILING else None

# Automatic error recovery - timing failures don't crash the application
timing and timing.start("operation")  # No-op if timing is None or fails internally
```

## Implementation Notes

### GPU Support
- CUDA event timing will be conditionally compiled/imported
- CuPy-specific code will be commented out for systems without GPU support
- Automatic fallback to CPU timing when GPU timing unavailable

### Memory Bandwidth Calculation Types
- **Computational Memory Bandwidth**: For operations that are memory-bound (reads + writes / time)
- **Transfer Bandwidth**: For explicit data transfers between CPU/GPU
- **Effective Bandwidth**: Accounting for memory access patterns and cache efficiency

### Automatic FLOPS Calculation
Support for common operation types with automatic FLOPS estimation:
- `einsum`: Based on einsum equation and tensor shapes
- `matmul`: Based on matrix dimensions  
- `elementwise`: Based on number of elements and operation type
- `sparse_matmul`: Based on non-zero count and operation complexity 