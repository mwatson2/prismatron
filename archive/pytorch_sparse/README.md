# PyTorch Sparse Matrix Optimization - Archived

This directory contains the abandoned PyTorch 4D COO tensor approach for LED optimization.

## Summary

**Attempted approach**: Convert CuPy/SciPy sparse CSC matrices to PyTorch 4D COO tensors for GPU acceleration with einsum operations.

**Result**: Memory inefficient - 4.5x larger than CSC format (1,440 MB vs 320 MB).

## Memory Analysis

- **CSC (current)**: 320 MB
- **2D COO**: 480 MB (1.5x larger)  
- **4D COO**: 1,440 MB (4.5x larger)

**Root cause**: COO format stores 4 coordinates per non-zero value (height, width, led, channel) using int64, creating 8x coordinate overhead vs actual data.

## Files

- `led_optimizer_pytorch.py` - 4D COO tensor LED optimizer implementation
- `compare_optimizers.py` - Performance comparison tool (crashes due to memory)
- `debug_comparison_instructions.md` - Debug instructions from investigation
- `compare_sparse_formats.py` - Storage format efficiency analysis
- `analyze_tensor_memory.py` - Memory usage analysis tool

## Lesson Learned

Sparse matrix formats (CSC/CSR) exist specifically to minimize memory overhead. Converting to dense tensor representations defeats their purpose. The current CuPy/SciPy approach is already optimal for this use case.

## Recommendation

Continue with CuPy sparse CSC matrices - they provide the best balance of memory efficiency and GPU acceleration for the LED optimization problem.
