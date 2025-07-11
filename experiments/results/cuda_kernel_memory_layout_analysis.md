# CUDA Kernel Memory Layout Analysis for DiagonalATAMatrix

This document systematically examines all CUDA kernels used by the DiagonalATAMatrix class and documents their memory layout assumptions, input requirements, and potential areas where layout assertions should be added.

## Overview

The DiagonalATAMatrix class uses several types of CUDA kernels for matrix-vector operations:
1. **CustomDIA3DMatVec kernels** (basic and optimized) - FP32
2. **CustomDIA3DMatVecFP16 kernels** (basic and optimized) - Mixed precision (FP32→FP16)  
3. **PureFP16DIA3DKernel kernels** (basic and optimized) - Pure FP16
4. **Fallback implementations** - CuPy operations

## Kernel Analysis

### 1. CustomDIA3DMatVec Kernels (FP32)

**File:** `/mnt/dev/prismatron/src/utils/kernels/dia_matvec.py`

#### Basic Kernel: `dia_matvec_3d_kernel`
- **Lines:** 120-166
- **Function signature:**
  ```cuda
  void dia_matvec_3d_kernel(
      const float* data,     // (channels, num_bands, n)
      const int* offsets,    // (num_bands,)
      const float* x,        // (channels, n)
      float* y,              // (channels, n)
      int n, int num_bands, int channels
  )
  ```

- **Memory Layout Assumptions:**
  - **`data` tensor**: Expected layout `(channels, num_bands, n)` in **C-contiguous** format
    - Access pattern: `data[channel * num_bands * n + band * n + j]` (line 155)
    - **Critical**: Assumes row-major (C-contiguous) memory layout
  - **`x` input tensor**: Expected layout `(channels, n)` in **C-contiguous** format  
    - Access pattern: `x[channel * n + led]` (line 144)
    - **Critical**: Assumes channel-major, C-contiguous layout
  - **`y` output tensor**: Layout `(channels, n)` in **C-contiguous** format
    - Access pattern: `y[channel * n + led_idx]` (line 164)

- **Grid/Block Structure:**
  - 2D grid: `blockIdx.x` = LED indices, `blockIdx.y` = channel indices
  - Each thread processes one (channel, LED) pair
  - **Memory access pattern**: Sequential within each channel's data

#### Optimized Kernel: `dia_matvec_3d_optimized_kernel`  
- **Lines:** 169-247
- **Additional Memory Requirements:**
  - **Shared memory**: `extern __shared__ float shared_mem[]`
  - **Size**: `(block_size + 2 * max_band_offset) * sizeof(float)`
  - **Usage**: Caches input vector `x` for better memory bandwidth
  - **Same base layout assumptions** as basic kernel

- **Shared Memory Access Pattern:**
  - Cooperative loading of `x_channel[global_idx]` into `shared_x[i]`
  - **Critical assumption**: Input vector must be contiguous for efficient coalesced loads

### 2. CustomDIA3DMatVecFP16 Kernels (Mixed Precision)

**File:** `/mnt/dev/prismatron/src/utils/kernels/dia_matvec_fp16.py`

#### Basic Kernel: `dia_matvec_3d_fp16_kernel`
- **Lines:** 126-174  
- **Function signature:**
  ```cuda
  void dia_matvec_3d_fp16_kernel(
      const float* data,     // (channels, num_bands, n) - FP32 storage
      const int* offsets,    // (num_bands,)
      const float* x,        // (channels, n) - FP32 input
      __half* y,             // (channels, n) - FP16 output
      int n, int num_bands, int channels
  )
  ```

- **Memory Layout Assumptions:**
  - **Same as FP32 kernels** for input layout requirements
  - **`data` tensor**: FP32 storage, C-contiguous `(channels, num_bands, n)`
  - **`x` input**: FP32, C-contiguous `(channels, n)`
  - **`y` output**: FP16, C-contiguous `(channels, n)`
  - **Conversion**: FP32 computation with `__float2half(sum)` output (line 172)

#### Optimized Kernel: `dia_matvec_3d_optimized_fp16_kernel`
- **Lines:** 177-256
- **Same memory layout assumptions** with shared memory optimization
- **Shared memory**: Still FP32 for input vector caching

### 3. PureFP16DIA3DKernel Kernels (Pure FP16)

**File:** `/mnt/dev/prismatron/src/utils/kernels/pure_fp16_dia_kernel.py`

#### Basic Kernel: `pure_fp16_3d_dia_kernel`
- **Lines:** 20-68
- **Function signature:**
  ```cuda
  void pure_fp16_3d_dia_kernel(
      const __half* data,    // (channels, num_bands, n) - FP16 storage  
      const int* offsets,    // (num_bands,)
      const __half* x,       // (channels, n) - FP16 input
      __half* y,             // (channels, n) - FP16 output
      int n, int num_bands, int channels
  )
  ```

- **Memory Layout Assumptions:**
  - **`data` tensor**: FP16 storage, **C-contiguous** `(channels, num_bands, n)`
    - Access: `data_channel[band * n + j]` (line 56)
  - **`x` input**: FP16, **C-contiguous** `(channels, n)`
    - Access: `x_channel[j]` (line 60)  
  - **`y` output**: FP16, **C-contiguous** `(channels, n)`
  - **Computation**: Pure FP16 arithmetic with `__hadd`, `__hmul` (line 61)

#### Optimized Kernel: `pure_fp16_3d_dia_optimized_kernel`
- **Lines:** 72-152
- **Shared memory**: `extern __shared__ __half shared_x[]` - **FP16 shared memory**
- **Same base layout assumptions** with FP16 shared memory optimization

### 4. Fallback Implementation

**File:** `/mnt/dev/prismatron/src/utils/diagonal_ata_matrix.py`

#### Method: `_multiply_3d_fallback`
- **Lines:** 512-557
- **No explicit memory layout requirements** - uses CuPy array operations
- **Assumed layout**: Relies on CuPy's default C-contiguous behavior
- **Access patterns**: `self.dia_data_gpu[:, band_idx, :]` and `led_values_gpu[:, slice]`

## Current Layout Assumptions Summary

### Input Tensor Requirements
All kernels assume the following **critical** memory layouts:

1. **`dia_data_3d`**: Shape `(channels, num_bands, n)`
   - **Must be C-contiguous** for access pattern `data[c*nb*n + b*n + i]`
   - **Channel-major layout**: All data for channel 0, then channel 1, etc.

2. **`led_values`**: Shape `(channels, n)`
   - **Must be C-contiguous** for access pattern `x[c*n + i]`  
   - **Channel-major layout**: All LEDs for channel 0, then channel 1, etc.

3. **`dia_offsets`**: Shape `(num_bands,)`
   - **Must be C-contiguous** (standard for 1D arrays)

### Stride Patterns Expected
- **`dia_data_3d`** strides: `(num_bands*n*sizeof(dtype), n*sizeof(dtype), sizeof(dtype))`
- **`led_values`** strides: `(n*sizeof(dtype), sizeof(dtype))`

### Vectorized Memory Access
- **Coalesced loads**: Kernels assume contiguous memory for efficient GPU memory access
- **Shared memory**: Optimized kernels load chunks of input vectors assuming contiguous layout
- **Bank conflicts**: FP16 kernels use 2-byte aligned access patterns

## Missing Layout Assertions

### Critical Missing Checks in DiagonalATAMatrix

The following assertions should be added to **prevent incorrect memory layout issues**:

#### 1. In `multiply_3d()` method (line ~374)
```python
# Assert input tensor layout for optimal performance  
if not isinstance(led_values, cupy.ndarray):
    raise TypeError(f"led_values must be cupy.ndarray, got {type(led_values)}")

# Check contiguity - critical for kernel performance
if not led_values.flags.c_contiguous:
    raise ValueError(
        f"led_values must be C-contiguous for optimal kernel performance. "
        f"Current strides: {led_values.strides}, expected for (channels={self.channels}, n={self.led_count}): "
        f"({self.led_count * led_values.itemsize}, {led_values.itemsize}). "
        f"Use cupy.ascontiguousarray(led_values) to fix."
    )

# Check memory layout assumptions
expected_strides = (self.led_count * led_values.itemsize, led_values.itemsize)
if led_values.strides != expected_strides:
    raise ValueError(
        f"led_values has unexpected stride pattern. Expected {expected_strides}, got {led_values.strides}. "
        f"This indicates incorrect memory layout (not channel-major C-contiguous)."
    )
```

#### 2. In `build_from_diffusion_matrix()` method (line ~264)
```python
# Ensure DIA data is stored in optimal layout
if self.dia_data_cpu is not None:
    # Verify CPU data layout before GPU transfer
    expected_shape = (self.channels, self.k, self.led_count)
    if self.dia_data_cpu.shape != expected_shape:
        raise ValueError(f"DIA data shape mismatch: expected {expected_shape}, got {self.dia_data_cpu.shape}")

    # Ensure contiguous layout for GPU transfer efficiency
    if not self.dia_data_cpu.flags.c_contiguous:
        print("Warning: DIA data not C-contiguous, creating contiguous copy for optimal GPU performance")
        self.dia_data_cpu = np.ascontiguousarray(self.dia_data_cpu)

    # Create GPU version with layout verification
    self.dia_data_gpu = cupy.asarray(self.dia_data_cpu, dtype=self.storage_dtype)

    # Verify GPU layout
    if not self.dia_data_gpu.flags.c_contiguous:
        raise RuntimeError("GPU DIA data not C-contiguous after transfer - this should not happen")
```

#### 3. In kernel wrapper methods
```python
def _validate_kernel_inputs(self, dia_data_3d, dia_offsets, led_values):
    """Validate that all inputs have correct layout for kernel execution."""

    # Check shapes
    channels, num_bands, n = dia_data_3d.shape
    assert dia_offsets.shape == (num_bands,), f"Offset shape mismatch"
    assert led_values.shape == (channels, n), f"Input shape mismatch"

    # Check contiguity - CRITICAL for performance
    if not dia_data_3d.flags.c_contiguous:
        raise ValueError("dia_data_3d must be C-contiguous for kernel")
    if not led_values.flags.c_contiguous:
        raise ValueError("led_values must be C-contiguous for kernel")
    if not dia_offsets.flags.c_contiguous:
        raise ValueError("dia_offsets must be C-contiguous for kernel")

    # Check memory layout (channel-major)
    expected_data_strides = (num_bands * n * dia_data_3d.itemsize,
                           n * dia_data_3d.itemsize,
                           dia_data_3d.itemsize)
    if dia_data_3d.strides != expected_data_strides:
        raise ValueError(f"dia_data_3d incorrect memory layout. Expected strides {expected_data_strides}, got {dia_data_3d.strides}")

    expected_input_strides = (n * led_values.itemsize, led_values.itemsize)
    if led_values.strides != expected_input_strides:
        raise ValueError(f"led_values incorrect memory layout. Expected strides {expected_input_strides}, got {led_values.strides}")
```

## Recommended Implementation

### Priority 1: Critical Layout Validation
Add the above assertions to `multiply_3d()` and `g_ata_g_3d()` methods immediately, as these are called during optimization and incorrect layouts will cause:
- **Silent correctness errors** (wrong results)
- **Performance degradation** (non-coalesced memory access)
- **Potential GPU crashes** (memory access violations)

### Priority 2: Diagnostic Information
Add tensor layout analysis to debug methods:
```python
def _analyze_tensor_layout(self, tensor, name):
    """Analyze and report tensor memory layout for debugging."""
    print(f"Tensor {name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Strides: {tensor.strides}")
    print(f"  Itemsize: {tensor.itemsize}")
    print(f"  C-contiguous: {tensor.flags.c_contiguous}")
    print(f"  F-contiguous: {tensor.flags.f_contiguous}")

    # Calculate expected strides for channel-major layout
    if len(tensor.shape) == 2:  # (channels, n)
        expected_strides = (tensor.shape[1] * tensor.itemsize, tensor.itemsize)
        print(f"  Expected strides (channel-major): {expected_strides}")
        print(f"  Layout correct: {tensor.strides == expected_strides}")
```

### Priority 3: Automatic Layout Correction
For performance-critical paths, automatically fix layout issues:
```python
def _ensure_optimal_layout(self, tensor, expected_shape):
    """Ensure tensor has optimal layout for kernel execution."""
    if not tensor.flags.c_contiguous:
        print(f"Warning: Creating contiguous copy of tensor for optimal performance")
        return cupy.ascontiguousarray(tensor)
    return tensor
```

## Performance Impact of Layout Issues

### Coalesced vs Non-Coalesced Memory Access
- **Coalesced** (correct layout): ~900 GB/s memory bandwidth on A100
- **Non-coalesced** (wrong layout): ~100 GB/s memory bandwidth (9x slower)

### Shared Memory Bank Conflicts
- **Conflict-free** (correct FP16 alignment): Full shared memory bandwidth
- **Bank conflicts** (misaligned): 32x slower shared memory access

This analysis shows that **memory layout validation is critical** for both correctness and performance of the DiagonalATAMatrix CUDA kernels.
