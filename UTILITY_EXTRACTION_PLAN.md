# Utility Extraction Plan for Prismatron

This document outlines standalone utility classes that can be extracted from the current codebase and developed without CuPy dependencies.

## High Priority Utilities (Ready for Development)

### 1. FileIOManager
**Current Location**: Scattered across multiple files  
**Dependencies**: NumPy, PIL/OpenCV (optional), SciPy  
**Purpose**: Unified file handling for images, patterns, and data formats

**Functionality to Extract**:
- Image loading with multiple backend support (PIL/OpenCV fallback)
- Image saving with automatic format detection
- NPZ file handling for sparse/dense matrices
- Pattern file validation and metadata preservation
- Error handling and format conversion utilities

**Benefits**: 
- Single interface for all file operations
- Backend abstraction (PIL vs OpenCV)
- Consistent error handling across tools

**Current Code Locations**:
- `optimization_utils.py:190-226` - Image loading with backend fallback
- `generate_synthetic_patterns.py:366-437` - Pattern file saving
- `visualize_diffusion_patterns.py:81-281` - Sparse matrix loading

---

### 2. ConfigurationManager
**Current Location**: `wled_client.py`, scattered config logic  
**Dependencies**: Standard library (dataclasses, typing)  
**Purpose**: Centralized configuration management

**Functionality to Extract**:
- WLED connection configuration (`WLEDConfig` dataclass)
- Configuration validation and defaults
- Environment variable integration
- JSON/YAML configuration file parsing
- Configuration serialization/deserialization

**Benefits**:
- Consistent configuration across all components
- Testable configuration validation
- Environment-specific configurations

**Current Code Locations**:
- `wled_client.py:76-89` - WLEDConfig dataclass
- Various hardcoded configurations throughout tools

---

### 3. DataValidator
**Current Location**: Input validation scattered throughout  
**Dependencies**: NumPy only  
**Purpose**: Consistent data validation and preprocessing

**Functionality to Extract**:
- Image shape and format validation
- LED data validation (count, format, range checking)
- Pattern file structure validation
- Data type conversion and normalization utilities
- Range checking and clamping functions

**Benefits**:
- Consistent validation logic
- Better error messages
- Reusable validation functions

**Current Code Locations**:
- `led_optimizer_dense.py:380-384` - Image shape validation
- `wled_client.py:413-444` - LED data validation
- Pattern validation logic in tools

---

### 4. ImageProcessor
**Current Location**: Image utilities across tools  
**Dependencies**: NumPy, PIL/OpenCV (optional)  
**Purpose**: CuPy-free image processing utilities

**Functionality to Extract**:
- Image resizing and resampling algorithms
- Format conversions (RGB/BGR/grayscale)
- Quality metrics (PSNR, SSIM, MSE) - CPU implementations
- Thumbnail generation for web interfaces
- Base64 encoding/decoding for web APIs
- Color space conversions

**Benefits**:
- CPU-based image processing (no GPU required)
- Quality assessment tools
- Web interface support utilities

**Current Code Locations**:
- `optimization_utils.py:37-111` - Image comparison and quality metrics
- `visualize_diffusion_patterns.py:558-652` - Thumbnail generation
- Image conversion utilities scattered across files

---

## Medium Priority Utilities

### 5. WLEDCommunicator
**Current Location**: `wled_client.py`  
**Dependencies**: Requests, socket  
**Purpose**: WLED protocol abstraction

**Functionality to Extract**:
- DDP packet construction and validation
- HTTP API communication with WLED devices
- Connection management and retry logic
- Device discovery and status monitoring
- Protocol version handling

**Benefits**:
- Protocol abstraction layer
- Network error handling
- Testable communication layer

**Current Code Locations**:
- `wled_client.py:91-926` - Most of the WLED client implementation

---

### 6. PatternGenerator
**Current Location**: Pattern generation tools  
**Dependencies**: NumPy, SciPy (sparse matrices)  
**Purpose**: Mathematical pattern generation utilities

**Functionality to Extract**:
- Gaussian pattern generation algorithms
- LED test patterns (rainbow, wave, solid colors)
- Spatial ordering algorithms (Morton encoding)
- Color space conversions (HSV/RGB)
- Pattern scaling and normalization

**Benefits**:
- Reusable pattern algorithms
- Mathematical consistency
- Test pattern generation

**Current Code Locations**:
- `generate_synthetic_patterns.py:32-438` - SyntheticPatternGenerator
- `wled_test_patterns.py:32-138` - LED pattern generation

---

### 7. PerformanceProfiler
**Current Location**: Timing code throughout  
**Dependencies**: Standard library (time, threading)  
**Purpose**: Performance measurement and profiling

**Functionality to Extract**:
- High-precision timing utilities
- Memory usage tracking (Python objects)
- FPS and throughput calculations
- Performance statistics collection
- Benchmark comparison utilities

**Benefits**:
- Consistent performance measurement
- Optimization insights
- Debugging support

**Current Code Locations**:
- Various manual timing throughout consumer and tools
- Statistics tracking in multiple files

---

## Implementation Strategy

### Phase 1: Core Utilities (Week 1)
1. **ConfigurationManager** - Simple, wide impact
2. **DataValidator** - Essential for robustness
3. **FileIOManager** - Critical for all tools

### Phase 2: Processing Utilities (Week 2)  
4. **ImageProcessor** - Image handling abstraction
5. **PatternGenerator** - Mathematical utilities

### Phase 3: Advanced Utilities (Week 3)
6. **WLEDCommunicator** - Protocol abstraction
7. **PerformanceProfiler** - Performance tools

## Development Guidelines

### Design Principles
- **No CuPy dependencies** - All utilities must work without GPU
- **Testable design** - Each utility should be unit testable
- **Minimal dependencies** - Use standard library when possible
- **Error handling** - Comprehensive error handling and logging
- **Documentation** - Full docstrings and usage examples

### Directory Structure
```
src/utils/
├── __init__.py
├── config_manager.py      # ConfigurationManager
├── data_validator.py      # DataValidator  
├── file_io_manager.py     # FileIOManager
├── image_processor.py     # ImageProcessor
├── pattern_generator.py   # PatternGenerator
├── wled_communicator.py   # WLEDCommunicator
├── performance_profiler.py # PerformanceProfiler
└── performance_timing.py  # Already implemented
```

### Testing Strategy
- Unit tests for each utility class
- Integration tests for utility combinations
- Mock dependencies where needed
- Performance benchmarks for critical paths

### Benefits of This Approach
1. **Immediate Development** - Can work without GPU access
2. **Better Architecture** - Separation of concerns
3. **Improved Testing** - Isolated, testable components
4. **Code Reuse** - Utilities usable across tools and consumer
5. **Future Integration** - Easy to integrate when main server returns

This plan provides a clear roadmap for extracting reusable utilities that enhance the codebase while staying within current development constraints.