# CUDA Kernels Memory Layout Documentation

## Overview

This document systematically examines all CUDA kernels used by the `SingleBlockMixedSparseTensor` class and documents their memory layout assumptions, stride requirements, and contiguity expectations.

## SingleBlockMixedSparseTensor Storage Format

The mixed tensor uses a **channels-first** memory layout throughout:

```
sparse_values:     (channels, batch_size, block_size, block_size)
block_positions:   (channels, batch_size, 2)
```

All data is stored in **C-contiguous** format by default.

## Kernel Categories

### 1. Transpose Dot Product Kernels (A^T @ b)

#### 1.1 FP32 Kernels

**File**: `src/utils/kernels/compute_optimized_3d.py`

**Kernel**: `compute_optimized_3d_transpose_dot_product_kernel`

**Memory Layout Requirements**:
- **Sparse Values**: `(channels, batch_size, block_size, block_size)` - **C-contiguous FP32**
- **Block Positions**: `(channels, batch_size, 2)` - **C-contiguous INT32**  
- **Target Image**: `(channels, height, width)` - **C-contiguous FP32 in planar format**
- **Output**: `(batch_size, channels)` - **C-contiguous FP32**

**Memory Access Patterns**:
- **Vectorized**: Uses `float4` loads for 4-element vectorization
- **Alignment**: Sparse values must be 16-byte aligned for optimal `float4` access
- **Stride Assumptions**:
  - Sparse data: `channel_stride = batch_size * block_size * block_size`
  - Target data: `channel_stride = height * width`
  - Row stride: `width` for target image access
- **Contiguity**: Requires C-contiguous arrays for optimal memory coalescing

**Critical Implementation Details**:
```c
// Sparse offset calculation assumes C-contiguous layout
int sparse_offset = channel_id * (batch_size * block_elements) + led_id * block_elements;

// Target offset calculation assumes planar C-contiguous layout  
int target_offset = channel_id * height * width + top_row * width + top_col;

// Vectorized access requires 16-byte alignment
float4 sparse_vec = *reinterpret_cast<const float4*>(&sparse_values[sparse_idx]);
float4 target_vec = load_float4_unaligned(&target_3d[target_idx]);
```

#### 1.2 UINT8 Kernels

**Files**: `src/utils/kernels/compute_optimized_3d_int8.py`, `src/utils/kernels/compute_optimized_3d_int8_fp16.py`

**Kernels**:
- `compute_optimized_3d_transpose_dot_product_int8_kernel`
- `compute_optimized_3d_transpose_dot_product_int8_experimental_kernel`

**Memory Layout Requirements**:
- **Sparse Values**: `(channels, batch_size, block_size, block_size)` - **C-contiguous UINT8**
- **Block Positions**: `(channels, batch_size, 2)` - **C-contiguous INT32**
- **Target Image**: `(channels, height, width)` - **C-contiguous UINT8 in planar format**
- **Output**: `(batch_size, channels)` - **C-contiguous FP32 or FP16**

**Memory Access Patterns**:
- **Vectorized**: Uses `uchar4` loads for 4-element vectorization
- **Alignment**:
  - **Standard version**: Handles unaligned access gracefully
  - **Experimental version**: Requires 4-byte alignment for optimal performance
- **Block size constraint**: `block_size % 4 == 0` for vectorization
- **Position constraints**: LED x-positions should be multiples of 4 for experimental version

**Critical Implementation Details**:
```c
// Identical memory layout calculations as FP32 version
int sparse_offset = channel_id * (batch_size * block_elements) + led_id * block_elements;
int target_offset = channel_id * height * width + top_row * width + top_col;

// UINT8 vectorized access  
uchar4 sparse_vec = *reinterpret_cast<const uchar4*>(&sparse_values[sparse_idx]);
uchar4 target_vec = *reinterpret_cast<const uchar4*>(&target_3d[target_idx]);

// Scaling for UINT8 -> FP32 conversion
result[idx] = sum / (255.0f * 255.0f);
```

### 2. Forward Pass Method (A @ x)

**Implementation**: CPU-based nested loops in `forward_pass_3d()` method

**Memory Layout Requirements**:
- **LED Values**: `(batch_size, channels)` - **C-contiguous FP32**
- **Output Frame**: `(channels, height, width)` - **C-contiguous FP32 in planar format**

**Memory Access Patterns**:
- **No vectorization**: Element-wise access
- **Scaling behavior**:
  - UINT8 matrices: `led_brightness / 255.0` scaling
  - FP32 matrices: Direct `led_brightness` scaling
- **Boundary clipping**: Handles block boundaries correctly

**Critical Implementation Details**:
```python
# Planar output format
output_frame = cp.zeros((self.channels, self.height, self.width), dtype=cp.float32)

# Channels-first indexing throughout
pattern_block = self.sparse_values[channel, led_idx]
top_row = int(self.block_positions[channel, led_idx, 0])
```

### 3. DIA Matrix-Vector Kernels (A^T A @ x)

**File**: `src/utils/kernels/dia_matvec.py`

**Kernels**:
- `dia_matvec_kernel` - Basic version
- `dia_matvec_optimized_kernel` - Shared memory optimization
- `dia_matvec_3d_kernel` - Multi-channel version

**Memory Layout Requirements**:

#### 3.1 2D DIA Kernels
- **DIA Data**: `(num_bands, n)` - **C-contiguous FP32**
- **Offsets**: `(num_bands,)` - **C-contiguous INT32**
- **Input Vector**: `(n,)` - **C-contiguous FP32**
- **Output Vector**: `(n,)` - **C-contiguous FP32**

#### 3.2 3D DIA Kernels  
- **DIA Data**: `(channels, num_bands, n)` - **C-contiguous FP32**
- **Offsets**: `(num_bands,)` - **C-contiguous INT32**
- **Input Vectors**: `(channels, n)` - **C-contiguous FP32 in channel-major layout**
- **Output Vectors**: `(channels, n)` - **C-contiguous FP32 in channel-major layout**

**Memory Access Patterns**:
- **Band-based access**: Iterates through diagonal bands
- **Shared memory**: Optimized versions cache vector elements
- **Bandwidth considerations**: Designed for matrices with limited bandwidth

**Critical Implementation Details**:
```c
// 2D DIA: A[i,j] stored at data[band * n + j]
const float matrix_val = data[band * n + j];

// 3D DIA: A[channel,i,j] stored at data_channel[band * n + j]
const float* data_channel = data + channel * num_bands * n;
const float matrix_val = data_channel[band * n + j];
```

## Memory Layout Issues and Solutions

### 1. Non-Contiguous Memory Detection

**Problem**: Mixed tensor operations can produce non-contiguous outputs that break kernel assumptions.

**Detection**:
```python
# Check tensor contiguity
is_c_contiguous = tensor.flags.c_contiguous
is_f_contiguous = tensor.flags.f_contiguous
```

**Solution**:
```python
# Force C-contiguous layout
tensor_contiguous = cp.ascontiguousarray(tensor)
```

### 2. Alignment Requirements

**FP32 Kernels**:
- 16-byte alignment for `float4` vectorization
- Checked in constructor: `assert sparse_values.data.ptr % 16 == 0`

**UINT8 Kernels**:
- 4-byte alignment for `uchar4` vectorization
- Block size must be multiple of 4: `block_size % 4 == 0`

### 3. Planar vs Interleaved Formats

**CUDA Kernels Expect**:
- **Planar format**: `(channels, height, width)` - channels separated
- **C-contiguous**: Memory layout `RRR...GGG...BBB...`

**Common Format Conversions**:
```python
# HWC to CHW planar
target_planar = np.ascontiguousarray(target_hwc.transpose(2, 0, 1))

# Ensure true planar layout (not just transpose view)
target_planar_uint8 = np.ascontiguousarray(target_frame.astype(np.uint8).transpose(2, 0, 1))
```

### 4. Stride Assumptions

**All kernels assume standard C-contiguous strides**:
- `sparse_values`: `[C*B*H*W, B*H*W, H*W, W]`
- `target_3d`: `[C*H*W, H*W, W]`
- `block_positions`: `[C*B*2, B*2, 2]`

**Non-standard strides will cause incorrect memory access patterns.**

## Best Practices

### 1. Memory Preparation
```python
# Always ensure C-contiguous arrays
sparse_values = cp.ascontiguousarray(sparse_values)
target_3d = cp.ascontiguousarray(target_3d)
block_positions = cp.ascontiguousarray(block_positions)
```

### 2. Format Validation
```python
# Check expected shapes and dtypes
assert sparse_values.shape == (channels, batch_size, block_size, block_size)
assert target_3d.shape == (channels, height, width)
assert sparse_values.dtype == expected_dtype
assert target_3d.flags.c_contiguous
```

### 3. Alignment Verification
```python
# Check alignment for FP32 kernels
if dtype == cp.float32:
    assert sparse_values.data.ptr % 16 == 0

# Check block size constraints for UINT8 kernels
if dtype == cp.uint8:
    assert block_size % 4 == 0
```

### 4. Debugging Memory Issues
```python
def analyze_tensor_properties(tensor, name):
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Strides: {tensor.strides}")
    print(f"  C-contiguous: {tensor.flags.c_contiguous}")
    print(f"  F-contiguous: {tensor.flags.f_contiguous}")
    if hasattr(tensor, 'data'):
        print(f"  Memory ptr: {tensor.data.ptr}")
        print(f"  Alignment: {tensor.data.ptr % 16}")
```

## Summary

The CUDA kernels in the SingleBlockMixedSparseTensor system have strict memory layout requirements:

1. **All arrays must be C-contiguous** for optimal performance
2. **Planar format required** for multi-channel data: `(channels, height, width)`
3. **Alignment constraints** vary by kernel type (16-byte for FP32, 4-byte for UINT8)
4. **Vectorization requirements** impose block size constraints
5. **Non-contiguous memory** from tensor operations can break kernel assumptions

The most critical issue is ensuring that intermediate tensor operations (like `transpose_dot_product_3d`) produce C-contiguous outputs, or explicitly making them contiguous before passing to subsequent kernels.
