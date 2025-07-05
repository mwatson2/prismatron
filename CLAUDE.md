# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This is a Python project that uses a virtual environment located in `env/`.

To activate the virtual environment:
```bash
source env/bin/activate
```

## Installed Dependencies

The project has the following key scientific computing and AI packages installed:
- **CuPy** - GPU-accelerated computing (replaces PyTorch for optimization)
- **SciPy** (1.15.3) - Scientific computing with sparse matrix support
- **OpenCV** (4.11.0) - Computer vision library  
- **NumPy** (2.2.6) - Numerical computing
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning
- **NetworkX** (3.4.2) - Graph/network analysis
- **SymPy** (1.13.1) - Symbolic mathematics
- **Pillow** (11.2.1) - Image processing
- **PySerial** (3.5) - Serial communication

## Project Status

The Prismatron LED Display System is **95% complete** with all major components implemented and tested. Recent accomplishments:

### ✅ Completed Major Components (Phases 1-5):
- **Core Infrastructure**: Shared memory ring buffer, control state management
- **Producer Components**: Video/image sources, FFmpeg integration, content management
- **Consumer Components**: Sparse matrix LED optimization engine, WLED communication
- **Web Interface**: React frontend, FastAPI backend, real-time WebSocket updates
- **Multi-Process System**: Full orchestration, error handling, resource management

### 🔧 Recent Major Updates:
- **Sparse Matrix Optimization**: Migrated from PyTorch dense matrices to CuPy/SciPy sparse CSC matrices for 50x memory efficiency
- **LSQR Solver**: Implemented iterative LSQR optimization for real-time performance
- **RGB Channel Separation**: Optimized color handling for accurate LED reproduction
- **Comprehensive Testing**: 190 unit tests covering all components
- **Regression Testing**: Pixel-perfect optimization validation with PSNR fallback

## Development Context and Performance Expectations

**IMPORTANT**: This project is in early-stage optimization development. The focus should be on improving individual components and algorithms, not speculating about overall system performance.

### Current Development Phase
- **Target Scale**: System must ultimately support **2600 LEDs** (not the current 500-1000 LED test patterns)
- **Component Focus**: Optimization work targets individual pieces (matrix operations, kernels, algorithms)
- **Performance Context**: Overall FPS will be affected by multiple system processes (producer, consumer, web server, OS overhead)
- **Techniques Pipeline**: Many optimization techniques remain unexplored and will be implemented iteratively

### Performance Guidelines for Development
1. **Do not speculate about frames-per-second** for the complete system
2. **Focus on algorithmic improvements** for the specific component being worked on
3. **Measure component-level performance** (e.g., matrix multiply time, kernel execution time)
4. **Scale considerations** should focus on algorithmic complexity, not raw performance numbers
5. **Benchmark against baseline** implementations to measure improvement, not absolute performance

### Development Priorities
- Optimize individual matrix operations and CUDA kernels
- Improve memory usage and bandwidth characteristics  
- Enhance algorithmic efficiency for larger LED counts
- Maintain code quality and comprehensive testing
- **Defer overall system performance evaluation** until explicitly requested

When overall system performance evaluation is needed, it will be explicitly requested. Until then, focus on making the current optimization component as efficient as possible.

## Project Development Guidance

### Memories and Guidelines
- **Never try to calculate fps** for the overall system before it's explicitly requested
- Focus on component-level optimizations and algorithmic improvements
- Measure performance at the individual kernel and matrix operation level
- Defer speculative performance analysis until a complete request is made

### Deprecated Methods
- CSC natrix for A is deprecated - we are using the mixed tensor now
- Dense ATA is deprecated. We are using our custom DIA class for the ATA matrix

### Pattern Design
- We have decided on 64x64 blocks for the patterns
