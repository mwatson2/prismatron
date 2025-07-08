# Prismatron Pattern File Format

This document describes the format used for storing LED diffusion patterns in the Prismatron system.

## File Format Overview

Pattern files are stored as compressed NumPy archives (`.npz` files) containing multiple data arrays and metadata. The format supports sparse matrix storage and includes multiple representations optimized for different use cases.

## File Structure

All pattern files contain the following top-level keys:

### Core Data
- `led_positions` - LED physical positions in 2D space
- `led_spatial_mapping` - Mapping from physical LED IDs to spatially-ordered matrix indices
- `metadata` - Generation metadata and configuration parameters
- `mixed_tensor` - SingleBlockMixedSparseTensor representation (blocked sparse format)
- `dia_matrix` - DiagonalATAMatrix representation (DIA format for A^T A)

### Optional Data
- `ata_inverse` - Precomputed A^T A inverse matrices (if available)
- `ata_inverse_metadata` - Metadata for ATA inverse computation

## Data Specifications

### LED Positions
```
led_positions: ndarray, shape (led_count, 2), dtype=float
```
Physical LED positions in 2D space as (x, y) coordinates within the frame bounds.

### LED Spatial Mapping
```
led_spatial_mapping: dict
```
Maps physical LED IDs to spatially-ordered matrix column indices. Used for RCM (Reverse Cuthill-McKee) ordering to optimize matrix bandwidth.

### Metadata
```
metadata: dict
```
Contains generation parameters and system information:
- `generator`: Always "SyntheticPatternGenerator"
- `format`: Matrix format ("sparse_csc", "led_diffusion_csc_with_mixed_tensor")
- `led_count`: Number of LEDs
- `frame_width`, `frame_height`: Frame dimensions in pixels
- `channels`: Number of color channels (always 3 for RGB)
- `matrix_shape`: Shape of the diffusion matrix [pixels, led_count * channels]
- `nnz`: Number of non-zero elements in sparse matrix
- `sparsity_percent`: Percentage of non-zero elements
- `sparsity_threshold`: Threshold below which values are considered zero
- `generation_timestamp`: Unix timestamp of generation
- `pattern_type`: Pattern generation method ("gaussian_multi", "gaussian_simple", "exponential")
- `seed`: Random seed used for reproducible generation
- `intensity_variation`: Whether LED intensity varies between LEDs
- `block_size`: Size of diffusion blocks (typically 64 or 96)
- `generation_method`: Always "chunked" for synthetic patterns

### Mixed Tensor Format
```
mixed_tensor: dict (serialized SingleBlockMixedSparseTensor)
```
Blocked sparse tensor representation optimized for GPU computation. Contains:
- `batch_size`: Number of LEDs
- `channels`: Number of color channels
- `height`, `width`: Frame dimensions
- `block_size`: Block dimensions (e.g., 64x64, 96x96)
- `device`: Target device ("cpu" or "cuda")
- Block data stored as sparse coordinate format

### DIA Matrix Format
```
dia_matrix: dict (serialized DiagonalATAMatrix)
```
Diagonal storage format for A^T A matrix optimized for iterative solvers:
- `led_count`: Number of LEDs
- `bandwidth`: Matrix bandwidth
- `k`: Number of diagonals
- `crop_size`: Block size used for cropping
- `dia_data_cpu`: Diagonal data array
- `dia_offsets_cpu`: Diagonal offset array

## Pattern Generation Details

### Block-Based Organization
- Patterns are organized into blocks of size `block_size` × `block_size` pixels
- Block positions are aligned to multiples of 4 pixels for memory efficiency
- Each LED's diffusion pattern is cropped to its containing block

### Spatial Ordering
- LEDs are reordered using RCM ordering to minimize matrix bandwidth
- Original physical LED IDs are preserved in `led_spatial_mapping`
- This optimization improves cache performance and reduces memory bandwidth

### Sparsity Optimization
- Values below `sparsity_threshold` are eliminated to reduce memory usage
- Typical sparsity levels: 97-99% of matrix elements are zero
- Sparse CSC format is used for efficient storage and computation

## Usage Examples

### Loading Pattern Files
```python
import numpy as np

# Load pattern file
data = np.load('pattern_file.npz', allow_pickle=True)

# Access basic information
led_positions = data['led_positions']
metadata = data['metadata'].item()
led_count = metadata['led_count']

# Access mixed tensor format
mixed_tensor_dict = data['mixed_tensor'].item()
```

### Reconstruction
```python
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor
from src.utils.diagonal_ata_matrix import DiagonalATAMatrix

# Reconstruct mixed tensor
mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

# Reconstruct DIA matrix
dia_matrix_dict = data['dia_matrix'].item()
dia_matrix = DiagonalATAMatrix.from_dict(dia_matrix_dict)
```

## File Size Expectations

Typical file sizes for different LED counts:
- 100 LEDs: ~5-10 MB
- 1000 LEDs: ~50-100 MB  
- 2624 LEDs: ~200-400 MB
- 3200 LEDs: ~300-500 MB

Sizes depend on pattern complexity, sparsity threshold, and whether ATA inverse matrices are included.

## Validation

Pattern files can be validated using the visualization tools:
```bash
python tools/visualize_diffusion_patterns.py --pattern-file pattern_file.npz
```

This will generate visualizations showing:
- LED positions and spatial ordering
- Sample diffusion patterns
- Matrix structure and sparsity
- Block organization
