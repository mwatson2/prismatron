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

## Project description

# Prismatron LED Display Software Specification

## Project Overview

The Prismatron is a computational LED art display with 3,200 RGB LEDs randomly arranged between two 60"×36" panels. The system captures the unique diffusion pattern of each LED and uses optimization algorithms to approximate images by controlling individual LED brightness values. The software consists of three main processes: Producer (content rendering), Consumer (LED optimization), and Web Server (management interface).

## Architecture Decisions

### Hardware Platform
- **Processing Unit**: NVIDIA Jetson Orin Nano 8GB (40-67 TOPS)
- **LED Controller**: DigiOcta board running WLED firmware
- **Communication**: UDP protocol to WLED over WiFi
- **Display**: 800 x 640 images as universal interface between producer/consumer (display aspect ratio is 5:3, not 4:3 or 16:9)

### Software Stack
- **Language**: Python for all components
- **Video Processing**: FFmpeg via python-ffmpeg bindings
- **Optimization**: CuPy/SciPy sparse matrices with LSQR solver for GPU acceleration
- **Web Framework**: FastAPI for backend API
- **Frontend**: React + Vite + Tailwind CSS (API-first architecture)
- **Process Communication**: Shared memory buffers with lightweight IPC

### Data Flow
```
Content Sources → Producer Process → Shared Memory Ring Buffer → Consumer Process → UDP → WLED
                                                                      ↓
Web Interface ←→ FastAPI Server ←→ Control IPC ←→ All Processes
```

## Core Components

### 1. Shared Memory Ring Buffer (`shared_buffer.py`)

**Purpose**: Zero-copy frame sharing between producer and consumer processes.

**Key Features**:
- Triple-buffered ring buffer (3 × 800 x 640 RGB frames)
- Multiprocessing shared memory implementation
- Event-based synchronization between processes
- Automatic buffer management and cleanup

**Interface**:
```python
class FrameRingBuffer:
    def get_write_buffer() -> dict  # Returns buffer for producer
    def advance_write() -> None     # Signal frame complete
    def wait_for_ready_buffer(timeout) -> dict  # Consumer waits for data
    def cleanup() -> None          # Resource cleanup
```

### 2. Control State Manager (`control_state.py`)

**Purpose**: Lightweight IPC for process coordination and configuration.

**Shared State**:
- Play/pause control
- Current content file path
- System brightness setting
- Shutdown coordination
- Frame rate monitoring
- Error state reporting

**Interface**:
```python
class ControlState:
    def set_current_file(filepath) -> None
    def set_brightness(value) -> None
    def signal_shutdown() -> None
    def get_status() -> dict
```

### 3. Content Source Plugins (`content_sources/`)

**Base Plugin Architecture**:
```python
class ContentSource:
    def setup() -> bool
    def get_next_frame(shared_array) -> bool
    def get_duration() -> float
    def seek(timestamp) -> bool
    def cleanup() -> None
```

**Plugin Implementations**:
- `VideoSource`: FFmpeg-based video decoding with hardware acceleration
- `ImageSource`: Static images with duration/transition support
- `AnimationSource`: GIF and animated format support
- `LiveSource`: Real-time generative content (future)

### 4. Producer Process (`producer.py`)

**Purpose**: Renders content sources to shared memory texture buffer.

**Key Responsibilities**:
- Load and manage content playlist
- Hardware-accelerated video decoding via FFmpeg
- Frame rate management (source-appropriate rates)
- Direct rendering to shared memory buffers
- Content transition handling

**FFmpeg Integration**:
- Hardware decode: `hwaccel='nvdec'`
- GPU scaling: `scale_npp` filter
- Output format: 1920×1080 RGBA rawvideo
- Async subprocess management

### 5. LED Optimization Engine (`led_optimizer.py`)

**Purpose**: Core sparse matrix optimization algorithm that maps texture data to LED brightness values.

**Key Components**:
- Sparse CSC matrix storage for diffusion patterns (50x memory reduction)
- LSQR iterative solver for real-time optimization
- RGB channel separation for accurate color reproduction
- GPU acceleration via CuPy (falls back to CPU/SciPy)
- Real-time performance targeting 15fps with 3,200 LEDs

**Algorithm Flow**:
1. Load sparse diffusion matrices from preprocessed files
2. Flatten input texture to target vector (800x640x3 → 1.5M elements)
3. Solve: minimize ||A×x - target||² using LSQR where A=diffusion matrix, x=LED RGB values
4. Process RGB channels separately for color accuracy
5. Clamp and format output as LED brightness values [0,255]

### 6. Consumer Process (`consumer.py`)

**Purpose**: Processes shared memory frames and controls LED display.

**Key Responsibilities**:
- Monitor shared memory for new frames
- Upload texture data to GPU
- Execute LED optimization algorithm
- Format and send UDP packets to WLED
- Performance monitoring and error handling

**WLED Communication**:
- UDP protocol to DigiOcta IP address
- WLED real-time protocol format
- 3,200 RGB values (9,600 bytes total per frame)
- Error handling and reconnection logic

### 7. Web API Server (`api_server.py`)

**Purpose**: FastAPI-based REST API and static file server.

**API Endpoints**:
```
POST /api/upload          # File upload handling
GET  /api/playlist        # Content management
POST /api/control/play    # Playback control
POST /api/control/pause
POST /api/settings        # System configuration
GET  /api/status          # Real-time system status
WebSocket /api/live       # Live updates to frontend
```

**Features**:
- File upload with validation and preprocessing
- Playlist management and scheduling
- Real-time system monitoring
- Static file serving for React frontend
- CORS handling for development

### 8. React Frontend (`frontend/`)

**Modern SPA Architecture**:
- React 18 with functional components and hooks
- Vite build system for fast development
- Tailwind CSS for responsive design
- PWA capabilities for mobile installation

**Key Pages/Components**:
- Dashboard: System status and current playback
- Upload: Drag-and-drop file upload with preview
- Playlist: Content management and scheduling
- Settings: System configuration
- LED Preview: Real-time visualization of LED array

**Real-time Features**:
- WebSocket connection for live updates
- Live LED array preview
- System performance monitoring
- Upload progress tracking

## Implementation Status

### ✅ Phase 1-5: Complete (95% of project)
All core functionality implemented and tested:
- **Infrastructure**: Shared memory ring buffer, control state management
- **Producer**: Video/image sources, FFmpeg integration, playlist management
- **Consumer**: Sparse matrix LED optimization, WLED communication with keepalive
- **Web Interface**: React frontend with retro-futurism design, FastAPI backend
- **Multi-Process**: Full orchestration, error handling, graceful shutdown
- **Testing**: 190 comprehensive unit tests with regression validation

### 🔄 Phase 6: Final Integration (5% remaining)
Ready for hardware deployment and production optimization:
1. **Hardware Integration**: Test on NVIDIA Jetson Orin Nano with actual WLED controller
2. **Performance Tuning**: Optimize sparse matrix operations for 3,200 LEDs at 15fps
3. **Production Config**: Headless operation, WiFi hotspot, auto-startup services
4. **Real Diffusion Patterns**: Capture actual LED diffusion patterns to replace synthetic ones

## Technical Requirements

### Performance Targets
- Consumer process: ≥15fps LED optimization
- Memory usage: <2GB total system RAM
- Network: <100ms latency for web interface responses
- Storage: Efficient content management and cleanup

### Hardware Acceleration
- FFmpeg nvdec for video decoding
- CuPy GPU acceleration for sparse matrix optimization (auto-fallback to CPU/SciPy)
- OpenGL texture operations where applicable

### Error Handling
- Graceful degradation on hardware failures
- Automatic process restart on crashes
- Comprehensive logging and debugging support
- Network reconnection and retry logic

### Development Environment
- Python 3.8+ virtual environment
- Requirements.txt with pinned versions
- Docker containerization for easy deployment
- Comprehensive unit and integration test suite

## File Structure
```
prismatron/
├── requirements.txt
├── main.py                    # Process orchestration
├── src/                      # Main source code
│   ├── core/                 # Core infrastructure
│   │   ├── shared_buffer.py  # Ring buffer implementation  
│   │   └── control_state.py  # IPC management
│   ├── producer/             # Producer process components
│   │   ├── producer.py       # Producer process
│   │   └── content_sources/  # Plugin directory
│   │       ├── base.py
│   │       ├── video_source.py
│   │       └── image_source.py
│   ├── consumer/             # Consumer process components
│   │   ├── consumer.py       # Consumer process
│   │   ├── led_optimizer.py  # Sparse matrix optimization engine
│   │   └── wled_client.py    # WLED UDP communication
│   ├── utils/                # Shared utilities
│   │   └── optimization_utils.py  # Testing and pipeline utilities
│   └── web/                  # Web interface components
│       ├── api_server.py     # FastAPI backend
│       └── frontend/         # React application
│           ├── src/
│           ├── package.json
│           └── vite.config.js
├── tests/                    # Unit and integration tests
│   ├── fixtures/             # Test data and regression fixtures
│   └── test_*.py             # 190 comprehensive unit tests
├── tools/                    # Development and diagnostic tools
│   ├── standalone_optimizer.py   # Standalone optimization testing
│   ├── generate_synthetic_patterns.py  # Pattern generation
│   └── visualize_diffusion_patterns.py # Pattern visualization
├── diffusion_patterns/      # LED diffusion pattern storage
│   ├── *.npz                # Dense format pattern files
│   ├── *_matrix.npz         # Sparse CSC matrix files
│   └── *_mapping.npz        # LED spatial mapping files
└── config/                   # Configuration files
    ├── led_positions.json
    └── system_config.yaml
```

## Diffusion Patterns Directory

The `diffusion_patterns/` directory stores LED diffusion pattern data for sparse matrix optimization:

### Current Pattern Files
- **`synthetic_1000.npz`**: Sparse CSC matrix for 1000 LEDs (primary development pattern)
- **`patterns_1000_chunked.npz`**: Alternative sparse format with chunked data
- **`config/diffusion_patterns.npz`**: Default pattern file for system operation

### Sparse Format Structure
Each pattern file contains:
```python
{
    'matrix_data': sparse.data,        # Non-zero values
    'matrix_indices': sparse.indices,  # Row indices  
    'matrix_indptr': sparse.indptr,    # Column pointers (CSC format)
    'matrix_shape': (pixels, leds*3),  # Matrix dimensions
    'led_spatial_mapping': dict,       # LED ID mappings
    'led_positions': array            # 2D LED coordinates
}
```

### Pattern Generation Tools
```bash
# Generate sparse patterns (recommended)
python tools/generate_synthetic_patterns.py --sparse --output diffusion_patterns/patterns_1000.npz --led-count 1000

# Visualize patterns
python tools/visualize_diffusion_patterns.py --patterns diffusion_patterns/synthetic_1000.npz

# Test optimization with patterns
python tools/standalone_optimizer.py --input test_image.jpg --patterns diffusion_patterns/synthetic_1000 --output result.png
```

## Code Management Principles

### Archive Directory Policy
**IMPORTANT**: The `archive/` directory contains deprecated code for reference only.
- **Never modify files in `archive/`** - they are read-only references
- Archive contains older implementations (PyTorch 4D COO, older sparse implementations)
- Current active code is in `src/` directory only
- If archived code is needed, copy it to `src/` and modify the copy

### Current Active Components
- **Dense LED Optimizer**: `src/consumer/led_optimizer_dense.py` (primary optimizer)
- **Sparse LED Optimizer**: Uses archived sparse code via import in `src/utils/optimization_utils.py`
- **Main Consumer**: `src/consumer/consumer.py`
- **Standalone Tools**: `tools/standalone_optimizer.py` (production testing tool)

## Development Commands

```bash
# Activate virtual environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full test suite (190 tests, ~48s)
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_led_optimizer.py -v         # LED optimization tests
python -m pytest tests/test_optimization_regression.py -v  # Regression tests
python -m pytest tests/test_shared_buffer.py -v        # Shared memory tests

# Run quick test subset (skip slow multiprocess tests)
python -m pytest tests/ -v -k "not multiprocess"

# Test LED optimization performance
python test_optimization_performance.py

# Standalone optimization testing
python tools/standalone_optimizer.py --input test_image.jpg --patterns diffusion_patterns/synthetic_1000 --output result.png --verbose

# Run system components
python main.py                    # Full multi-process system
python -m src.web.api_server     # Web interface only
python -m src.consumer.consumer  # Consumer process only
```

### Test Results Summary
- **Total Tests**: 190 passed, 4 skipped (multiprocess timing tests)
- **Coverage**: All major components with comprehensive edge case testing
- **Regression Tests**: Pixel-perfect optimization validation with PSNR fallback
- **Performance**: Sparse matrix optimization achieves 15+ FPS target

## Remaining Tasks for Production Deployment

### Phase 6: System Integration (Final 5%)
1. **Hardware Integration Testing**: Test with actual NVIDIA Jetson Orin Nano and WLED controller
2. **Performance Profiling**: Optimize for 15fps with 3,200 LEDs on target hardware
3. **Production Configuration**: Headless operation, WiFi hotspot, auto-startup
4. **Diffusion Pattern Capture**: Replace synthetic patterns with real LED diffusion captures

## WLED Keepalive Feature

The WLED client now includes automatic keepalive functionality to prevent WLED controllers from reverting to local patterns when no new data is received for a few seconds.

### How it works:
- When connected, the WLED client automatically starts a background thread
- The thread repeats the last sent LED pattern every second (configurable)
- Only activates when no new data has been sent for the keepalive interval
- Automatically stops when disconnected or disabled

### Configuration:
```python
config = WLEDConfig(
    host="wled.local",
    led_count=2600,
    enable_keepalive=True,      # Enable/disable keepalive (default: True)
    keepalive_interval=1.0      # Interval in seconds (default: 1.0)
)
```

### Runtime control:
```python
client.set_keepalive_enabled(True)      # Enable keepalive
client.set_keepalive_enabled(False)     # Disable keepalive
client.set_keepalive_interval(2.0)      # Change interval to 2 seconds
```

### Monitoring:
```python
stats = client.get_statistics()
print(f"Keepalive enabled: {stats['keepalive_enabled']}")
print(f"Keepalive active: {stats['keepalive_active']}")
print(f"Keepalive interval: {stats['keepalive_interval']}s")
```

This ensures that LED patterns remain stable and don't flicker back to WLED's default patterns during playback.

## Recent Architectural Changes

### Sparse Matrix Migration (2024-12)
- **From**: PyTorch dense matrix optimization with 50GB+ memory requirements
- **To**: CuPy/SciPy sparse CSC matrices with <1GB memory usage (50x reduction)
- **Solver**: LSQR iterative optimization for real-time performance
- **Benefits**: Scalable to 3,200 LEDs, GPU acceleration with CPU fallback

### RGB Channel Optimization
- **Approach**: Separate R, G, B channel optimization for color accuracy
- **Result**: Improved color reproduction and optimization convergence
- **Performance**: Maintains real-time requirements with channel separation

### Comprehensive Testing Framework
- **Unit Tests**: 190 tests covering all components with edge cases
- **Regression Tests**: Pixel-perfect validation with PSNR quality fallback
- **Fixtures**: Small test patterns (100 LEDs) for CI/CD integration
- **Coverage**: Memory management, multiprocess communication, optimization accuracy

### Tools and Utilities
- **`standalone_optimizer.py`**: Production-ready optimization testing tool
- **`optimization_utils.py`**: Shared utilities for testing and validation
- **Pattern Generation**: Synthetic sparse matrix creation for development
- **Visualization**: Diffusion pattern analysis and debugging tools

This architecture provides a robust foundation for the final hardware integration phase.

## Development Guidelines

### Code Management Principles
- **Never create unnecessary duplicate code or test files**
- **Use existing standard tools and utilities** (e.g., `standalone_optimizer.py`, comprehensive test suite)
- **Prefer editing existing files over creating new ones**
- **Only create new files when absolutely necessary for the specific goal**
