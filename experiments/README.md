# Experiments Directory

This directory contains experimental scripts, analysis tools, and research code that were used during development but are not part of the main production codebase.

## Directory Structure

### `dia_ata_inverse_analysis/`
Analysis of DIA (Diagonal) format ATA inverse matrices:
- `mse_convergence_dia_ata_inverse.py` - Main analysis script testing diagonal factors 2.0-4.0
- `mse_convergence_dia_ata_inverse.png` - Convergence plot results
- `mse_convergence_dia_analysis_results.md` - Analysis results documentation

**Key Findings**: Factor 3.0+ achieves near-dense quality, but dense ATA inverse preferred due to memory not being limiting factor and performance advantages only appearing at high quality cost.

### `mse_convergence_analysis/`
General MSE convergence analysis scripts:
- `mse_convergence_analysis.py` - Original convergence analysis
- `mse_convergence_analysis_with_dia.py` - Analysis including DIA formats
- `mse_convergence_dia_focus.py` - Focused DIA analysis
- `mse_comparison_table.py` - Comprehensive comparison tables
- `mse_table_simple.py` - Simplified comparison tables

### `debug_scripts/`
Debug and development scripts:
- Various debug_*.py scripts used during development
- `simple_mse_debug.py` - Simple MSE debugging tools

### `test_scripts/`
Experimental test scripts:
- `test_dia_ata_inverse.py` - DIA ATA inverse testing
- `test_dia_factor_sweep.py` - Factor sweep testing
- `test_mse_tracking.py` - MSE tracking tests
- `mse_convergence_test.py` - Convergence testing

### `precision_analysis/`
Precision and numerical analysis:
- `simple_mse_precision_test.py` - Simple precision tests
- `mse_precision_convergence_test.py` - Precision convergence analysis
- `mse_precision_convergence_real.py` - Real-world precision tests

### `results/`
Generated plots, images, and analysis results from experiments.

## Production Code

The main production code supports both dense and DIA format ATA inverse matrices through the `frame_optimizer.py` API. While experiments showed dense format is preferred for performance, DIA support is maintained for future research and larger scale scenarios.

## Usage

These scripts are preserved for reference and future research but should not be used in production. They may have dependencies on specific pattern files or configurations that were used during development.
