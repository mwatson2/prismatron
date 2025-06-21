# CuPy vs PyTorch Comparison Debug Instructions

## Current Status
- Script: `tools/compare_optimizers.py` (PyTorch vs CuPy comparison)
- Previous attempts have crashed
- Need to run in controlled way with detailed logging

## Command to Run
```bash
cd /mnt/dev/prismatron
source env/bin/activate
python tools/compare_optimizers.py
```

## Intention
- Compare performance between CuPy sparse matrix optimization and PyTorch dense matrix optimization
- Measure memory usage, execution time, and optimization quality
- Identify which approach is better for real-time LED optimization

## Expected Behavior
- Load test image
- Run both optimization methods
- Compare results and performance metrics
- Generate comparison report

## If Crash Occurs
1. Check error message for specific failure point
2. Look for memory issues (OOM errors)
3. Check GPU availability and CUDA setup
4. Verify input files exist and are accessible
5. Check for import errors or missing dependencies

## Environment Check
- Python virtual environment: `env/` (should be activated)
- CuPy availability: Check with `python -c "import cupy; print(cupy.__version__)"`
- PyTorch availability: Check with `python -c "import torch; print(torch.__version__)"`
- CUDA availability: Check with `nvidia-smi`

## Debug Steps if Crash
1. Run with reduced problem size first
2. Add try-catch blocks around major operations
3. Monitor memory usage with `htop` or `nvidia-smi`
4. Check logs for specific error location
