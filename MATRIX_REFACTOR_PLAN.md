# Matrix Computation Refactoring Plan

## Overview
Refactor the diffusion pattern tools to separate pattern generation/capture from matrix computation. The current issue is that the DIA matrix construction from the capture tool is inefficient, creating too many diagonals. We will create a clean pipeline where:

1. **Pattern Generation/Capture** → Mixed Sparse Tensor (A matrix in uint8)
2. **Matrix Computation** → Dense ATA (fp32) from Mixed Tensor
3. **Matrix Storage** → Symmetric DIA and Dense ATA Inverse

## Current Problems
- Capture tool creates DIA with 4729 diagonals for 3008 LEDs (inefficient - more than dense!)
- Conversion through CSC sparse matrix loses block structure information
- No direct method to compute ATA from Mixed Sparse Tensor blocks
- Inconsistent handling of uint8 vs fp32 conversions

## Proposed Architecture

### Phase 1: Pattern Generation/Capture
**Tools:** `generate_synthetic_patterns.py`, `capture_diffusion_patterns.py`
- **Output:** Mixed Sparse Tensor (uint8) + LED positions + RCM ordering
- **NO ATA computation** - just raw pattern data
- Remove all ATA matrix generation code from these tools

### Phase 2: Matrix Computation
**New Tool:** `compute_matrices.py`
- **Input:** Pattern file with Mixed Sparse Tensor
- **Process:**
  1. Load Mixed Sparse Tensor (uint8)
  2. Compute Dense ATA using new `compute_ata_dense()` method
  3. Create Symmetric DIA from Dense ATA
  4. Compute Dense ATA Inverse
- **Output:** Updated pattern file with all matrices

### Phase 3: Matrix Storage Classes
- **Mixed Sparse Tensor:** A matrix in uint8 format
- **Dense ATA Matrix:** Full ATA in fp32
- **Symmetric Diagonal ATA Matrix:** Optimized DIA storage (fp32)
- **Dense ATA Inverse:** For optimization (fp32)

## Implementation Steps

### Step 1: Add `compute_ata_dense()` to SingleBlockMixedSparseTensor ✓
**File:** `src/utils/single_block_sparse_tensor.py`
**Status:** COMPLETED - Method already exists at line 822-911

The method computes A^T A directly from block overlaps:
- For each pair of LEDs (i,j)
- For each channel
- Find overlapping regions of their blocks
- Compute dot product of overlap
- Handle uint8 scaling (divide by 255²)
- Output: (3, led_count, led_count) fp32 array

### Step 2: Create `compute_matrices.py` Tool
**Location:** `tools/compute_matrices.py`
**Tasks:**
- [ ] Load pattern file with Mixed Sparse Tensor
- [ ] Call `compute_ata_dense()` on the tensor
- [ ] Create DenseATAMatrix from the result
- [ ] Create SymmetricDiagonalATAMatrix from DenseATAMatrix
- [ ] Compute ATA inverse from DenseATAMatrix
- [ ] Save all matrices back to file

### Step 3: Refactor Pattern Generation Tools
**Files:** `tools/generate_synthetic_patterns.py`, `tools/capture_diffusion_patterns.py`
**Tasks:**
- [ ] Remove `_generate_dia_matrix()` methods
- [ ] Remove `_generate_dense_ata_matrix()` methods
- [ ] Remove `_choose_ata_format()` methods
- [ ] Remove ATA-related imports
- [ ] Update save methods to only save Mixed Tensor + metadata
- [ ] Update documentation

### Step 4: Update Repair/Conversion Tools
**Files:** Various tools in `tools/`
**Tasks:**
- [ ] Update `compute_ata_inverse.py` to work with new structure
- [ ] Update `convert_dia_to_dense.py` if needed
- [ ] Remove obsolete repair tools
- [ ] Create migration tool for old pattern files

### Step 5: Fix uint8 Scaling in compute_ata_dense()
**File:** `src/utils/single_block_sparse_tensor.py`
**Current Issue:** Method doesn't handle uint8 → fp32 conversion
**Fix needed at line 891-892:**
```python
# Current (incorrect for uint8):
dot_product = np.sum(overlap_i * overlap_j)

# Should be:
if self.dtype == cp.uint8:
    # Convert uint8 [0,255] to float [0,1] range
    overlap_i_float = overlap_i.astype(np.float32) / 255.0
    overlap_j_float = overlap_j.astype(np.float32) / 255.0
    dot_product = np.sum(overlap_i_float * overlap_j_float)
else:
    dot_product = np.sum(overlap_i * overlap_j)
```

## Task Tracking

### Immediate Tasks
- [ ] Fix uint8 scaling in `compute_ata_dense()` method
- [ ] Create `compute_matrices.py` tool
- [ ] Test with a small synthetic pattern
- [ ] Verify diagonal count is reasonable

### Pattern Generation Refactor
- [ ] Update `generate_synthetic_patterns.py`
- [ ] Update `capture_diffusion_patterns.py`
- [ ] Update save format documentation

### Tool Updates
- [ ] Update `compute_ata_inverse.py`
- [ ] Remove obsolete conversion tools
- [ ] Create migration tool

### Testing & Validation
- [ ] Test synthetic pattern generation
- [ ] Test real pattern capture
- [ ] Verify ATA matrix correctness
- [ ] Benchmark performance

## Benefits of Refactor

1. **Correctness:** Direct computation from blocks preserves true sparsity pattern
2. **Efficiency:** Avoids CSC conversion that creates artificial diagonals
3. **Clarity:** Clean separation of concerns
4. **Memory:** uint8 storage for patterns, fp32 only for matrices
5. **Flexibility:** Easy to add new matrix formats or optimizations

## File Format After Refactor

```python
{
    # Pattern data (from generation/capture)
    'mixed_tensor': {...},           # uint8 A matrix
    'led_positions': array,          # LED physical positions
    'led_spatial_mapping': dict,     # RCM ordering
    'led_ordering': array,           # Spatial to physical mapping
    'metadata': {...},               # Generation metadata

    # Computed matrices (from compute_matrices.py)
    'dense_ata_matrix': {...},       # fp32 dense ATA
    'symmetric_dia_matrix': {...},   # fp32 symmetric DIA
    'ata_inverse': array,            # fp32 inverse matrix
    'ata_inverse_metadata': {...}    # Computation metadata
}
```

## Notes

- Mixed Sparse Tensor always stores patterns in uint8 for memory efficiency
- ATA matrices are always fp32 for numerical stability  
- The `compute_ata_dense()` method already exists but needs uint8 scaling fix
- This refactor will make the capture tool produce correct, efficient DIA matrices

## Progress Log

**2024-08-08:**
- Identified root cause: CSC conversion creates artificial diagonals
- Found existing `compute_ata_dense()` method in SingleBlockMixedSparseTensor
- Created refactoring plan
- Next: Implement uint8 scaling fix and create compute_matrices.py tool
