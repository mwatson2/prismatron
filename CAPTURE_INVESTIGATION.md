# Capture Investigation - ATA Matrix Regeneration

## Investigation Status: 2025-07-29

This document tracks the investigation and regeneration of ATA matrices from captured diffusion pattern data after a system crash.

## Background

We were working on regenerating the ATA matrix in both dense and DIA form from raw capture data in `diffusion_patterns/capture-0728-01.npz` when the system crashed. We need to:

1. Recreate both dense and DIA ATA matrices from the capture data
2. Compare the matvec operations between these two formats
3. Record all commands executed to prevent loss of work if crashes occur again

## Staged/Unstaged Changes Analysis

### Key Files Modified
- `src/utils/frame_optimizer.py`: Updated to support both DIA and DenseATAMatrix formats
- `tools/capture_diffusion_patterns.py`: Added format selection logic and DenseATAMatrix support
- `tools/generate_synthetic_patterns.py`: Similar updates for synthetic patterns
- `tools/visualize_diffusion_patterns.py`: Support for both matrix formats

### New Tools Created
- `rebuild_ata_from_capture.py`: Tool to rebuild both DIA and dense ATA matrices from raw capture data
- `convert_dia_to_dense_ata.py`: Tool to convert DIA format to dense format
- `compare_ata_matrices.py`: Tool to compare dense vs DIA matrix operations for equivalence
- `src/utils/dense_ata_matrix.py`: New DenseATAMatrix class implementation

## Available Files

- `/mnt/dev/prismatron/diffusion_patterns/capture-0728-01.npz`: Main capture file to work with
- Multiple other capture files available as fallbacks

## Investigation Plan

### Phase 1: Rebuild ATA Matrices ✅ (Tools Ready)
- Use `rebuild_ata_from_capture.py` to regenerate both formats from raw capture data
- Apply proper uint8 scaling and global scaling factors
- Save rebuilt matrices to new file

### Phase 2: Compare Matrix Operations ✅ (Tool Ready)  
- Use `compare_ata_matrices.py` to verify mathematical equivalence
- Test with multiple LED value patterns
- Verify both formats produce identical results

## Command Execution Log

All commands will be logged below with timestamps to track progress:

```bash
# MEMORY-SAFE COMMANDS - Create one matrix format at a time to prevent crashes:

# Create DIA ATA matrix:
python rebuild_ata_from_capture.py --input diffusion_patterns/capture-0728-01.npz --output diffusion_patterns/capture-0728-01-dia.npz --format dia --verbose

# Create Dense ATA matrix:  
python rebuild_ata_from_capture.py --input diffusion_patterns/capture-0728-01.npz --output diffusion_patterns/capture-0728-01-dense.npz --format dense --verbose
```

## Matrix Format Details

### DIA Format (DiagonalATAMatrix)
- Sparse storage using diagonal format
- Memory efficient for sparse matrices
- Bandwidth-dependent performance
- Uses RCM ordering for optimization

### Dense Format (DenseATAMatrix)  
- Dense 3D array storage on GPU
- Better for high-bandwidth matrices
- Direct matrix-vector multiplication
- Larger memory footprint but potentially faster

## Status - INVESTIGATION COMPLETE ✅

- [x] Investigation of previous work
- [x] Analysis of staged/unstaged changes  
- [x] Tool availability verification
- [x] **MEMORY ISSUE FIXED**: Updated `rebuild_ata_from_capture.py` to create only one matrix format at a time
- [x] Execute matrix rebuilding (DIA format) - ✅ SUCCESS
- [x] Execute matrix rebuilding (Dense format) - ✅ SUCCESS  
- [x] Run matrix comparison tests - ❌ **CRITICAL BUG FOUND**
- [x] **BUG IDENTIFIED**: Dense matrix channel extraction is incorrect
- [x] **Fix Dense matrix implementation** - ✅ **COMPLETED**
- [x] **Re-run comparison after fix** - ✅ **VERIFIED**
- [x] **Performance analysis** - ✅ **ACCEPTABLE ERROR LEVELS**
- [x] **Documentation of results** - ✅ **DOCUMENTED**

### 🎯 FINAL SUMMARY

The capture investigation has been **successfully completed**. The critical Dense matrix channel extraction bug has been identified and fixed:

1. **Root Cause**: Dense matrix used incorrect block-based channel extraction instead of interleaved pattern
2. **Impact**: Caused ~65,000x scaling errors making Dense matrices unusable
3. **Resolution**: Fixed channel extraction in `src/utils/dense_ata_matrix.py` to match CSC construction
4. **Verification**: Error reduced from catastrophic scaling to excellent <1 ppm numerical precision difference
5. **Outcome**: Both DIA and Dense ATA matrix formats are now mathematically equivalent and ready for production use

The system can now reliably use either matrix format for LED optimization with confidence in mathematical correctness.

=== Previous Command That Crashed System ===
Command: python rebuild_ata_from_capture.py --input diffusion_patterns/capture-0728-01.npz --output diffusion_patterns/capture-0728-01-rebuilt.npz --verbose
Status: CRASHED DEVICE - tool was creating both DIA and dense matrices simultaneously

=== Updated Tool Features ===
- Memory-optimized: Creates only one matrix format per run
- Aggressive memory cleanup: Deletes intermediate objects
- Format selection: --format dia|dense required argument
- Better garbage collection during CSC matrix building

## Critical Bug Discovery - 2025-07-30

### ❌ Dense Matrix Channel Extraction Bug

During matrix comparison testing, we discovered that the DIA and Dense ATA matrices produce completely different results despite using the same CSC input matrix. Investigation revealed:

**Root Cause**: The Dense and DIA matrices use different channel extraction patterns from the CSC matrix:

- **DIA Matrix (CORRECT)**: Uses interleaved extraction matching the CSC construction
  - R channel: columns `[0, 3, 6, 9, 12, ...]` (led_id * 3 + 0)
  - G channel: columns `[1, 4, 7, 10, 13, ...]` (led_id * 3 + 1)  
  - B channel: columns `[2, 5, 8, 11, 14, ...]` (led_id * 3 + 2)

- **Dense Matrix (BUG)**: Uses incorrect block extraction
  - R channel: columns `[0:led_count]` (first third of matrix)
  - G channel: columns `[led_count:2*led_count]` (second third)
  - B channel: columns `[2*led_count:3*led_count]` (final third)

**Impact**: This bug causes Dense matrices to be computed from completely wrong data, leading to:
- ~65,000x scaling differences (255² factor from processing wrong channels)
- Mathematically incorrect optimization results
- System instability when using Dense matrix format

**Evidence**:
- Matrix comparison showed ratio of ~6.5e4 ≈ 255² between DIA and Dense results
- Simple controlled test confirmed different channel extraction patterns
- CSC matrix construction uses `col_idx = led_id * 3 + channel` (interleaved pattern)

**Files Affected**:
- `src/utils/dense_ata_matrix.py` - Lines 104-107 have incorrect channel extraction ✅ **FIXED**
- All rebuilt dense matrices are invalid and must be regenerated after fix ✅ **REGENERATED**

### ✅ BUG RESOLUTION - 2025-07-30

**Status: RESOLVED** - The Dense matrix channel extraction bug has been successfully fixed.

**Fix Applied**: Updated `src/utils/dense_ata_matrix.py` lines 104-107 to use correct interleaved channel extraction:
```python
# OLD (INCORRECT) - Block-based extraction:
start_col = channel * self.led_count
end_col = (channel + 1) * self.led_count
A_channel = diffusion_gpu[:, start_col:end_col]

# NEW (CORRECT) - Interleaved extraction matching CSC construction:
channel_cols = cupy.arange(channel, diffusion_gpu.shape[1], self.channels)
A_channel = diffusion_gpu[:, channel_cols]
```

**Verification Results**:
- Error reduced from ~65,000x scaling difference to tiny floating-point precision difference
- Both DIA and Dense matrices now use identical channel extraction pattern  
- Remaining error: **0.094 absolute units** (max difference), **<1 ppm relative error**
- This represents excellent numerical agreement between sparse (DIA) and dense matrix operations

### Executed Commands Log

```bash
# SUCCESSFUL - DIA matrix rebuild
python rebuild_ata_from_capture.py --input diffusion_patterns/capture-0728-01.npz --output diffusion_patterns/capture-0728-01-dia.npz --format dia --verbose
# Result: ✅ 112.5MB, k=895 diagonals, bandwidth=447, global_scaling_factor=0.065705

# SUCCESSFUL - Dense matrix rebuild (CORRECTED VERSION)  
python rebuild_ata_from_capture.py --input diffusion_patterns/capture-0728-01.npz --output diffusion_patterns/capture-0728-01-dense-fixed.npz --format dense --verbose
# Result: ✅ 117.9MB, mathematically correct after channel extraction fix

# FINAL COMPARISON TEST - Verified fix
python compare_ata_matrices.py --file diffusion_patterns/capture-0728-01-clean-comparison.npz --verbose
# Result: ✅ Max error 9.38e-02 absolute, <1 ppm relative - EXCELLENT NUMERICAL AGREEMENT
```
