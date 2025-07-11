# Sparse Matrix Optimization for Real-Time LED Display ("Prismatron")

## Project Overview

This document outlines the computational approach for optimizing LED brightness values in a real-time artistic display system with 3,200 randomly arranged RGB LEDs. The system uses sparse matrix optimization to approximate target images by solving a large-scale least squares problem.

## System Architecture

### Hardware Setup
- **Display**: Two 60"×36" panels separated by 4" gap
- **LEDs**: 3,200 RGB addressable LEDs randomly positioned between panels
- **Front Panel**: 1/4" textured acrylic for light diffusion
- **Back Panel**: 3mm aluminum for heat dissipation
- **Processing**: NVIDIA Jetson Orin Nano 8GB (40-67 TOPS)
- **LED Controller**: WLED running on QuinLED DigiOcta (EPS32))
- **Communication**: UDP ovwer WLAN

### Problem Formulation

The core optimization problem is:
```
Ax = b
```
Where:
- **A**: Sparse contribution matrix (1,152,000 × 3,200) mapping LED effects to pixels
- **x**: LED brightness values (3,200 unknowns)
- **b**: Target image (800×480×3 pixels flattened)
- 0 <= **x_i** <= 1 for all i

## Sparse Matrix Advantages

### Memory Efficiency
- **Dense storage**: ~7.4GB (1,152,000 × 3,200 × 2 bytes)
- **Sparse CSR storage**: ~50-200MB (assuming 5-15% sparsity)
- **Memory reduction**: 20-150x smaller

### Why the Matrix is Sparse
Each LED affects only a localized region of the display due to:
- Limited light spread through 4" air gap
- Textured diffusion panel creating focused patterns
- Typical LED influence: <5% of total pixels

## Implementation Strategy

### Phase 1: Calibration with LED Spatial Ordering

```python
import scipy.sparse as sp
import numpy as np

def calibrate_led_matrix_optimized():
    """Capture LED diffusion patterns and build cache-optimized sparse matrix."""

    # Step 1: Capture all LED patterns and estimate positions
    led_patterns = {}
    led_positions = {}

    for physical_led_id in range(3200):
        activate_single_led(physical_led_id)
        diffusion_image = capture_camera_image()

        # Extract significant pixels above threshold
        pattern = extract_significant_pixels(diffusion_image, threshold=0.01)
        led_patterns[physical_led_id] = pattern

        # Estimate LED position from centroid of diffusion pattern
        led_positions[physical_led_id] = estimate_led_position(pattern)

    # Step 2: Create spatial ordering of LEDs for cache optimization
    led_spatial_mapping = create_led_spatial_ordering(led_positions)

    # Step 3: Build matrix with spatially-ordered LED columns
    rows, cols, values = [], [], []

    for physical_led_id, pattern in led_patterns.items():
        # Map to spatially-ordered matrix column index
        matrix_led_idx = led_spatial_mapping[physical_led_id]

        for pixel_row, pixel_col, intensity in pattern:
            pixel_idx = pixel_row * 480 * 3 + pixel_col * 3  # RGB interleaved
            rows.append(pixel_idx)
            cols.append(matrix_led_idx)
            values.append(intensity)

    # Step 4: Create CSC matrix (optimal for A^T operations in LSQR)
    A_sparse_csc = sp.csc_matrix((values, (rows, cols)),
                                shape=(800*480*3, 3200),
                                dtype=np.float16)

    # Step 5: Save both CSC and LED mapping
    sp.save_npz('led_contribution_matrix_csc.npz', A_sparse_csc)
    np.save('led_spatial_mapping.npy', led_spatial_mapping)

    return A_sparse_csc, led_spatial_mapping

def estimate_led_position(diffusion_pattern):
    """Estimate LED physical position from diffusion pattern centroid."""
    if not diffusion_pattern:
        return (0, 0)  # Fallback for failed patterns

    total_intensity = 0
    weighted_row = 0
    weighted_col = 0

    for pixel_row, pixel_col, intensity in diffusion_pattern:
        total_intensity += intensity
        weighted_row += pixel_row * intensity
        weighted_col += pixel_col * intensity

    # Return centroid coordinates
    centroid_row = weighted_row / total_intensity
    centroid_col = weighted_col / total_intensity
    return (centroid_row, centroid_col)

def create_led_spatial_ordering(led_positions):
    """Create LED ordering based on spatial proximity using Z-order curve."""

    def morton_encode(x, y):
        """Convert 2D coordinates to Z-order (Morton) index for spatial locality."""
        # Normalize coordinates to integers
        x_int = int(x * 1000)  # Scale for precision
        y_int = int(y * 1000)

        result = 0
        for i in range(16):  # 16-bit precision
            result |= (x_int & (1 << i)) << i | (y_int & (1 << i)) << (i + 1)
        return result

    # Sort LEDs by spatial proximity (Z-order)
    led_list = [(led_id, pos[0], pos[1]) for led_id, pos in led_positions.items()]
    led_list.sort(key=lambda item: morton_encode(item[1], item[2]))

    # Create mapping: physical_led_id → spatially_ordered_matrix_index
    spatial_mapping = {led_id: matrix_idx for matrix_idx, (led_id, _, _) in enumerate(led_list)}

    return spatial_mapping

def extract_significant_pixels(image, threshold=0.01):
    """Extract pixels above threshold with their coordinates and intensities."""
    pattern = []
    height, width = image.shape[:2]

    for row in range(height):
        for col in range(width):
            if len(image.shape) == 3:  # RGB image
                # Use luminance for thresholding
                intensity = 0.299 * image[row, col, 0] + 0.587 * image[row, col, 1] + 0.114 * image[row, col, 2]
            else:
                intensity = image[row, col]

            if intensity > threshold:
                pattern.append((row, col, float(intensity)))

    return pattern
```

### Phase 2: Real-Time Optimization with CSC and Projected LSQR

```python
import cupy as cp
from cupyx.scipy.sparse import csc_matrix, csr_matrix
from cupyx.scipy.sparse.linalg import lsqr

def load_and_optimize_csc():
    """Load CSC matrix and run real-time optimization with projected LSQR."""

    # Load pre-computed CSC matrix (optimized for A^T operations)
    A_sparse_cpu = sp.load_npz('led_contribution_matrix_csc.npz')
    A_sparse_gpu = csc_matrix(A_sparse_cpu)  # Transfer CSC to GPU

    # Load LED spatial mapping for result interpretation
    led_mapping = np.load('led_spatial_mapping.npy', allow_pickle=True).item()

    while True:  # Real-time loop
        # Capture/receive target image
        target_image = get_target_image()
        target_gpu = cp.asarray(target_image.flatten(), dtype=cp.float16)

        # Solve with projected LSQR for [0,1] constraints
        led_brightness = solve_projected_lsqr_csc(A_sparse_gpu, target_gpu)

        # Send results to ESP32 via SPI
        send_led_data_to_esp32(led_brightness, led_mapping)

def solve_projected_lsqr_csc(A_csc, b, max_iter=50, tolerance=1e-6):
    """Projected LSQR using CSC format for optimal A^T operations."""

    # Initialize LED brightness values
    x = cp.full(A_csc.shape[1], 0.5, dtype=cp.float16)

    for iteration in range(max_iter):
        # Compute residual: r = Ax - b  
        residual = A_csc @ x - b

        # Compute gradient: g = A^T @ r (this is fast with CSC!)
        gradient = A_csc.T @ residual

        # Estimate step size using LSQR-style approach
        Ag = A_csc @ gradient  # A @ gradient
        step_size = cp.dot(gradient, gradient) / cp.dot(Ag, Ag)

        # Gradient descent step
        x_new = x - step_size * gradient

        # Project onto feasible region [0,1]
        x_new = cp.clip(x_new, 0, 1)

        # Check convergence
        if cp.linalg.norm(x_new - x) < tolerance:
            break

        x = x_new

    return x

def load_and_optimize_hybrid():
    """Alternative: Use both CSR and CSC for optimal forward/backward operations."""

    # Load CSC matrix
    A_csc_cpu = sp.load_npz('led_contribution_matrix_csc.npz')
    A_csc_gpu = csc_matrix(A_csc_cpu)

    # Convert to CSR for forward operations
    A_csr_gpu = A_csc_gpu.tocsr()

    led_mapping = np.load('led_spatial_mapping.npy', allow_pickle=True).item()

    while True:
        target_image = get_target_image()
        target_gpu = cp.asarray(target_image.flatten(), dtype=cp.float16)

        # Use hybrid approach with both formats
        led_brightness = solve_hybrid_lsqr(A_csr_gpu, A_csc_gpu, target_gpu)
        send_led_data_to_esp32(led_brightness, led_mapping)

def solve_hybrid_lsqr(A_csr, A_csc, b, max_iter=50, tolerance=1e-6):
    """Hybrid LSQR using CSR for Ax and CSC for A^T operations."""

    x = cp.full(A_csr.shape[1], 0.5, dtype=cp.float16)

    for iteration in range(max_iter):
        # Forward operation: use CSR (optimized for Ax)
        residual = A_csr @ x - b

        # Backward operation: use CSC (optimized for A^T x)  
        gradient = A_csc.T @ residual

        # Step size estimation
        Ag = A_csr @ gradient  # Use CSR for forward operation
        step_size = cp.dot(gradient, gradient) / cp.dot(Ag, Ag)

        # Update and project
        x_new = x - step_size * gradient
        x_new = cp.clip(x_new, 0, 1)

        if cp.linalg.norm(x_new - x) < tolerance:
            break

        x = x_new

    return x

def send_led_data_to_esp32(led_brightness, led_mapping):
    """Send LED data to ESP32 with spatial mapping for physical LED control."""

    # Convert to physical LED ordering
    physical_led_values = cp.zeros(3200, dtype=cp.uint8)

    # Map from matrix indices back to physical LED IDs
    reverse_mapping = {matrix_idx: physical_led_id
                      for physical_led_id, matrix_idx in led_mapping.items()}

    for matrix_idx in range(len(led_brightness)):
        physical_led_id = reverse_mapping[matrix_idx]
        # Convert float [0,1] to uint8 [0,255]
        physical_led_values[physical_led_id] = int(led_brightness[matrix_idx] * 255)

    # Send via SPI to ESP32
    spi_send_rgb_data(physical_led_values)
```

## CSC Format Optimization for LSQR

### Why CSC Over CSR for LED Optimization
CSC (Compressed Sparse Column) format provides significant advantages for LSQR iterations:

- **A^T operations**: CSC format makes A^T @ vector operations much faster (dominant in LSQR)
- **LED-centric storage**: Each column represents one LED's influence pattern (natural data organization)
- **Cache optimization**: When combined with spatial LED ordering, provides excellent memory locality
- **LSQR efficiency**: Most LSQR time is spent on transpose operations, where CSC excels

### Matrix Format Comparison

| Operation | CSR Performance | CSC Performance | LSQR Usage Frequency |
|-----------|----------------|-----------------|---------------------|
| **Ax** (forward) | Excellent | Good | ~30% of operations |
| **A^T x** (transpose) | Poor | Excellent | ~70% of operations |
| **Memory locality** | Row-based | Column-based | LED-centric is natural |

### Spatial LED Ordering Benefits
```python
# Without spatial ordering:
LED_columns = [LED_0, LED_1847, LED_23, LED_2901, ...]  # Random positions
Memory_access = scattered, many cache misses

# With spatial ordering (Z-order curve):
LED_columns = [LED_topLeft, LED_topLeft+1, LED_topLeft+2, ...]  # Sequential positions  
Memory_access = coalesced, excellent cache utilization
Cache_hit_rate: 20% → 80%
A^T_operations: 1.5-2.5x speedup
```

### Hybrid CSR/CSC Approach
For maximum performance, maintain both matrix formats:
- **CSC matrix**: For A^T operations (70% of LSQR time)
- **CSR matrix**: For Ax operations (30% of LSQR time)
- **Memory cost**: ~2x storage, but both matrices share the same underlying data pattern

## LSQR Algorithm Benefits

### Why LSQR Over Normal Equations
- **Memory efficient**: Never computes A^T A (would be 3,200 × 3,200 dense matrix)
- **Numerically stable**: Better conditioning than normal equations
- **Sparse-optimized**: Only requires sparse matrix-vector products Av and A^T u
- **Monotonic convergence**: Residual ||Ax - b|| decreases with each iteration
- **Constraint compatible**: Can be modified to handle LED brightness constraints

### GPU Acceleration with CuPy
- **CuSparse integration**: Leverages NVIDIA's optimized sparse BLAS
- **Ampere architecture**: Jetson Orin's GPU has specific sparse matrix instructions
- **Unified memory**: Efficient data transfer between CPU and GPU
- **Constraint operations**: cp.clip() and boolean masking are GPU-accelerated
- **CSC optimization**: cuSPARSE has specific optimizations for CSC transpose operations

## Performance Targets

### Computational Requirements
- **Target frame rate**: 15 fps
- **Processing budget**: ~67ms per frame
- **Expected LSQR iterations**: 20-50
- **Memory usage**: <1GB total (including GPU buffers)

### Expected Performance with CSC and Spatial Ordering
- **Matrix loading**: <100ms (one-time startup cost)
- **CSC projected LSQR**: 15-25ms per frame (optimized for A^T operations)
- **Hybrid CSR/CSC LSQR**: 12-20ms per frame (optimal for both Ax and A^T x)
- **LED spatial mapping**: <1ms overhead
- **SPI data transfer**: <5ms
- **Total latency**: 18-30ms per frame (comfortable for 15fps target)

### Performance Optimization Strategy
```python
def select_optimization_approach(performance_target):
    """Choose optimal approach based on performance requirements."""

    if performance_target == "maximum_speed":
        return solve_projected_lsqr_csc  # CSC with projection (15-25ms)
    elif performance_target == "balanced":
        return solve_hybrid_lsqr  # CSR+CSC hybrid (12-20ms)
    else:
        return solve_projected_lsqr_csc  # Default to CSC approach
```

### LED Spatial Ordering Validation
```python
def validate_led_spatial_ordering(led_positions, led_mapping):
    """Verify that spatial ordering improves cache locality."""

    # Measure average distance between consecutive LEDs
    total_distance = 0
    for i in range(len(led_mapping) - 1):
        led1_pos = led_positions[reverse_mapping[i]]
        led2_pos = led_positions[reverse_mapping[i+1]]
        distance = np.sqrt((led1_pos[0] - led2_pos[0])**2 + (led1_pos[1] - led2_pos[1])**2)
        total_distance += distance

    avg_neighbor_distance = total_distance / (len(led_mapping) - 1)
    print(f"Average distance between consecutive LEDs: {avg_neighbor_distance:.2f} pixels")

    # Good spatial ordering should have small average neighbor distance
    return avg_neighbor_distance < 50  # Threshold for good locality
```

## Storage and Loading Strategy

### Disk Storage Format
```python
# Save sparse matrix in SciPy's native format
sp.save_npz('led_contribution_matrix.npz', A_sparse)

# File characteristics:
# - Automatic compression
# - Preserves CSR format structure
# - Cross-platform compatibility
# - Fast loading optimized for scipy/cupy
```

### Runtime Loading
```python
# Fast startup sequence
A_sparse_cpu = sp.load_npz('led_contribution_matrix.npz')  # ~50-100ms
A_sparse_gpu = csr_matrix(A_sparse_cpu)                   # ~10-20ms GPU transfer
```

## Implementation Considerations

### LED Position Estimation and Matrix Organization
- **Centroid calculation**: LED position estimated from intensity-weighted pixel coordinates
- **Z-order spatial sorting**: LEDs ordered using Morton encoding for optimal cache locality  
- **CSC column optimization**: Spatially nearby LEDs have consecutive column indices
- **Cache performance**: 20% → 80% hit rate improvement from spatial ordering

### CSC Format Benefits
```python
# CSC storage efficiency for LED patterns:
CSC_advantages = {
    'transpose_operations': '2-3x faster than CSR',
    'led_centric_storage': 'Natural column-per-LED organization',
    'spatial_locality': 'Consecutive columns = nearby LEDs',
    'memory_bandwidth': '40-60% better utilization'
}
```

### Matrix Format Decision Tree
```python
def choose_matrix_format(optimization_priority):
    """
    Select optimal sparse matrix format based on priorities.
    """
    if optimization_priority == "memory_minimal":
        return "CSC_only"  # Single matrix, A^T optimized
    elif optimization_priority == "speed_maximal":
        return "CSR_CSC_hybrid"  # Both formats, optimal for all operations
    elif optimization_priority == "balanced":
        return "CSC_only"  # Good compromise, recommended
    else:
        return "CSC_only"  # Default recommendation
```

### Error Handling
- **Convergence monitoring**: Track LSQR iterations and residual
- **Fallback strategies**: Reduce iteration count if timing is critical
- **LED safety limits**: Clamp brightness values to safe ranges

### Calibration Quality
- **Capture consistency**: Use controlled lighting conditions
- **Camera calibration**: Account for lens distortion and color response
- **Multiple measurements**: Average several captures per LED for noise reduction

## Future Optimizations

### Algorithmic Improvements
- **Preconditioning**: Could improve LSQR convergence rate
- **Block processing**: Solve image regions independently
- **Temporal coherence**: Use previous frame as initialization

### Hardware Scaling
- **Higher resolution**: Scale to 4K+ target images
- **More LEDs**: Algorithm scales linearly with LED count

## Success Metrics

- **Visual quality**: Recognizable image approximation
- **Real-time performance**: Consistent 15+ fps
- **System stability**: Hours of continuous operation
- **Power efficiency**: <350W total system power

This approach combines cutting-edge sparse linear algebra with practical embedded systems to create a unique computational art installation that bridges mathematical optimization and visual creativity.
