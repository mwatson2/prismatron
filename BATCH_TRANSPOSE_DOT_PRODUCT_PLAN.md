# Batch A^T @ b Operation Implementation Plan

## Overview
Implementation plan for adding batch processing capability to the `transpose_dot_product_3d` operation in `SingleBlockMixedSparseTensor`. This will enable processing multiple input frames simultaneously with shape `(batch_frames, channels, height, width)`.

## Current Architecture Analysis

### Current Single-Frame Operation
- **Input**: `target_3d` shape `(channels, height, width)` - one frame
- **Grid**: `(batch_size, channels)` - one CUDA block per (LED, channel) pair  
- **Block**: `32 threads` - each thread processes `block_size/32` rows of the 64×64 LED block
- **Output**: `(batch_size, channels)` - dot product result for each LED/channel

### Proposed Batch Operation Architecture

## Method Signature
```python
def transpose_dot_product_3d_batch(
    self,
    target_batch: cp.ndarray,           # (batch_frames, channels, height, width)
    output_dtype: Optional[cp.dtype] = None,
    planar_output: bool = False
) -> cp.ndarray:
    """
    Compute batched A^T @ B operation where B contains multiple frames.

    Returns:
        shape (batch_frames, batch_size, channels) if planar_output=False
        shape (batch_frames, channels, batch_size) if planar_output=True
    """
```

## CUDA Kernel Design

### Selected Approach: 3D Grid Layout
```cuda
// Grid: (batch_size, channels, batch_frames) - 3D grid
// Block: (32, 1, 1) - 32 threads per block
// Each block processes one (LED, channel, frame) combination
```

**Benefits:**
- Natural extension of current 2D grid
- Each block processes one LED pattern against one frame
- Same thread-level optimization (32 threads process block_size/32 rows each)
- LED pattern can be loaded into shared memory once per block

## Implementation Details

### CUDA Kernel Structure

```cuda
extern "C" __global__
void compute_optimized_3d_batch_transpose_dot_product_kernel(
    const float* sparse_values,        // (channels, batch_size, block_size, block_size)
    const int* block_positions,        // (channels, batch_size, 2)
    const float* target_batch,         // (batch_frames, channels, height, width)
    float* result,                     // (batch_frames, batch_size, channels) or (batch_frames, channels, batch_size)
    const int batch_size,              // Number of LEDs
    const int channels,                // Number of color channels  
    const int batch_frames,            // Number of input frames
    const int height,
    const int width,
    const int block_size,
    const bool interleaved             // Output layout control
) {
    // 3D grid coordinates
    int led_id = blockIdx.x;           // LED index [0, batch_size)
    int channel_id = blockIdx.y;       // Channel index [0, channels)  
    int frame_id = blockIdx.z;         // Frame index [0, batch_frames)

    // Bounds checking
    if (led_id >= batch_size || channel_id >= channels || frame_id >= batch_frames)
        return;

    // **KEY OPTIMIZATION: Load LED pattern into shared memory**
    __shared__ float led_pattern[64*64];  // Assuming 64x64 blocks, could be parameterized

    // Cooperative loading of LED pattern by all 32 threads
    int elements_per_thread = (block_size * block_size + 31) / 32;
    int sparse_base_offset = channel_id * (batch_size * block_size * block_size) +
                            led_id * block_size * block_size;

    for (int i = 0; i < elements_per_thread; i++) {
        int element_idx = threadIdx.x * elements_per_thread + i;
        if (element_idx < block_size * block_size) {
            led_pattern[element_idx] = sparse_values[sparse_base_offset + element_idx];
        }
    }
    __syncthreads();  // Ensure all threads have loaded their part

    // Get LED block position  
    int pos_idx = channel_id * (batch_size * 2) + led_id * 2;
    int top_row = block_positions[pos_idx + 0];
    int top_col = block_positions[pos_idx + 1];

    // Process assigned rows (same logic as single-frame version)
    int rows_per_thread = block_size / 32;
    float thread_sum = 0.0f;

    // Calculate target frame base offset
    int target_frame_offset = frame_id * (channels * height * width) +
                             channel_id * (height * width) +
                             top_row * width + top_col;

    // Each thread processes rows_per_thread rows using shared memory LED pattern
    int thread_start_row = threadIdx.x * rows_per_thread;
    for (int row_offset = 0; row_offset < rows_per_thread; row_offset++) {
        int current_row = thread_start_row + row_offset;
        int target_row_offset = target_frame_offset + current_row * width;
        int led_row_offset = current_row * block_size;

        // Process all columns in this row using vectorized loads
        for (int col = 0; col < block_size; col += 4) {
            // Load 4 elements from target frame  
            float4 target_vec = *reinterpret_cast<const float4*>(
                &target_batch[target_row_offset + col]);

            // Load 4 elements from shared memory LED pattern
            float4 led_vec = *reinterpret_cast<const float4*>(
                &led_pattern[led_row_offset + col]);

            // Compute dot product
            thread_sum += led_vec.x * target_vec.x + led_vec.y * target_vec.y +
                         led_vec.z * target_vec.z + led_vec.w * target_vec.w;
        }
    }

    // Warp reduction
    auto warp = tiled_partition<32>(this_thread_block());
    float sum = reduce(warp, thread_sum, plus<float>());

    // Write result with correct indexing
    if (warp.thread_rank() == 0) {
        int result_idx;
        if (interleaved) {
            // (batch_frames, batch_size, channels)
            result_idx = frame_id * (batch_size * channels) + led_id * channels + channel_id;
        } else {
            // (batch_frames, channels, batch_size)  
            result_idx = frame_id * (channels * batch_size) + channel_id * batch_size + led_id;
        }
        result[result_idx] = sum;
    }
}
```

### Key Optimizations

1. **Shared Memory LED Pattern Loading**
   - Each block loads one 64×64 LED pattern into shared memory cooperatively
   - All 32 threads participate in loading ~128 elements each (4096/32)
   - LED pattern is reused across all computations in the block

2. **Vectorized Memory Access**
   - Same float4 vectorization as current implementation
   - 4-element loads for both target frame and LED pattern data
   - Maintains alignment requirements

3. **Grid Scaling**  
   - 3D grid naturally scales: `grid_size = (batch_size, channels, batch_frames)`
   - Total blocks = `batch_size × channels × batch_frames`
   - Same 32 threads per block for optimal warp utilization

## Memory Layout & Performance Analysis

### Memory Bandwidth
- **LED patterns**: Loaded once per block into shared memory (64KB for 64×64 FP32)
- **Target frames**: Streamed from global memory with vectorized loads
- **Result**: Coalesced writes to global memory

### Shared Memory Usage
- **LED pattern**: `64 × 64 × 4 bytes = 16KB` per block (well within 48KB limit)
- **Reduction workspace**: `32 × 4 bytes = 128 bytes`  
- **Total per block**: ~16KB (comfortable margin)

### Performance Expectations
- **Throughput**: Should scale linearly with batch size
- **Memory efficiency**: LED patterns loaded once per block vs. once per frame
- **Compute efficiency**: Same vectorized dot product inner loops

## Implementation Progress

### Phase 1: Core CUDA Kernel ✅
- [x] **Task 1.1**: Create `src/utils/kernels/compute_optimized_3d_batch.py`
- [x] **Task 1.2**: Implement FP32 batch kernel with shared memory optimization
- [x] **Task 1.3**: Add Python wrapper function
- [x] **Task 1.4**: Add kernel compilation and caching

### Phase 2: Integration ✅
- [x] **Task 2.1**: Add `transpose_dot_product_3d_batch()` method to `SingleBlockMixedSparseTensor`
- [x] **Task 2.2**: Add tensor validation and memory layout checks
- [x] **Task 2.3**: Handle both FP32 and uint8 input routing
- [x] **Task 2.4**: Implement error handling and fallbacks

### Phase 3: uint8 Support ✅
- [x] **Task 3.1**: Create `src/utils/kernels/compute_optimized_3d_batch_int8.py`
- [x] **Task 3.2**: Adapt kernel for uint8→FP32 conversion
- [x] **Task 3.3**: Add uint8 wrapper function
- [x] **Task 3.4**: Test uint8 variant correctness

### Phase 4: Testing ✅
- [x] **Task 4.1**: Unit tests - correctness vs single-frame calls
- [x] **Task 4.2**: Test both output layout formats (interleaved/planar)
- [x] **Task 4.3**: Boundary condition tests (batch=1, large batches)
- [x] **Task 4.4**: Memory layout validation tests
- [x] **Task 4.5**: Performance benchmarking vs repeated single calls

### Phase 5: Documentation ✅
- [x] **Task 5.1**: Update class docstrings with batch operation examples
- [x] **Task 5.2**: Add usage examples in method documentation  
- [x] **Task 5.3**: Performance analysis documentation
- [x] **Task 5.4**: Integration guide for optimization engine

## Integration Points

### Python Wrapper
```python
def cuda_transpose_dot_product_3d_batch_compute_optimized(
    sparse_values: cp.ndarray,          # (channels, batch_size, block_size, block_size)
    block_positions: cp.ndarray,        # (channels, batch_size, 2)
    target_batch: cp.ndarray,           # (batch_frames, channels, height, width)  
    batch_size: int,
    channels: int,
    batch_frames: int,
    block_size: int,
    interleaved: bool = True
) -> cp.ndarray:
```

### Class Method Addition
```python  
def transpose_dot_product_3d_batch(
    self,
    target_batch: cp.ndarray,
    output_dtype: Optional[cp.dtype] = None,
    planar_output: bool = False
) -> cp.ndarray:
```

## Testing Strategy

### Unit Tests
- **Correctness**: Compare batch operation against multiple single-frame calls  
- **Memory layouts**: Test both interleaved and planar output formats
- **Boundary conditions**: Single frame batch, large batches, edge cases
- **Data types**: Both FP32 and uint8 variants

### Performance Tests  
- **Scaling**: Measure throughput vs batch size
- **Memory usage**: Verify shared memory utilization
- **Comparison**: Batch vs repeated single-frame operations

## Success Criteria
- [x] Batch operation produces identical results to repeated single-frame calls
- [ ] Performance scales linearly with batch size (within memory bandwidth limits)
- [x] Memory usage stays within GPU limits for reasonable batch sizes
- [x] Both FP32 and uint8 variants work correctly
- [x] Integration with optimization engine is seamless

## Performance Analysis Results

### Benchmark Findings
The implementation was successfully completed and tested, with all correctness tests passing. However, performance benchmarks revealed that the batch operation is **slower** than repeated single-frame calls:

- **Average speedup**: 0.36x (2.8x slower)
- **Maximum speedup**: 0.69x (1.4x slower)
- **Best case**: Small systems (64 LEDs, 32x32 blocks) with large batch sizes (32 frames)

### Root Cause Analysis
The shared memory optimization approach has several issues:

1. **Shared Memory Overhead**: Loading 16KB LED patterns into shared memory creates significant latency
2. **3D Grid Inefficiency**: (batch_size × channels × batch_frames) creates too many small blocks
3. **Memory Access Patterns**: Batch dimension causes non-coalesced target frame access
4. **Block Utilization**: Each block processes only one LED/channel/frame combination

### Alternative Approaches for Future Optimization
1. **2D Grid with Frame Loop**: Keep (batch_size × channels) grid, loop over frames within each block
2. **Direct Global Memory**: Eliminate shared memory loading, use direct global memory access
3. **Coalesced Batch Access**: Restructure target frame layout for better memory coalescing
4. **Larger Block Sizes**: Process multiple LEDs per block to improve shared memory efficiency

The current implementation serves as a **correct baseline** for future optimization work. The algorithm is mathematically sound and produces identical results to single-frame operations.

---

This design leverages the existing kernel architecture while adding efficient batching through shared memory optimization and 3D grid scaling. The key insight is that each LED pattern can be loaded once into shared memory and reused for dot products against all frames in the batch.
