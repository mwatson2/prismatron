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
- **PyTorch** (2.5.0) - Deep learning framework
- **OpenCV** (4.11.0) - Computer vision library  
- **NumPy** (2.2.6) - Numerical computing
- **SciPy** (1.15.3) - Scientific computing
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning
- **NetworkX** (3.4.2) - Graph/network analysis
- **SymPy** (1.13.1) - Symbolic mathematics
- **Pillow** (11.2.1) - Image processing
- **PySerial** (3.5) - Serial communication

## Project Structure

This appears to be a new project with minimal setup. The repository currently contains:
- `env/` - Python virtual environment
- `LICENSE` - MIT license
- `.gitignore` - Standard Python gitignore

## Project description

# Prismatron LED Display Software Specification

## Project Overview

The Prismatron is a computational LED art display with 3,200 RGB LEDs randomly arranged between two 60"×36" panels. The system captures the unique diffusion pattern of each LED and uses optimization algorithms to approximate images by controlling individual LED brightness values. The software consists of three main processes: Producer (content rendering), Consumer (LED optimization), and Web Server (management interface).

## Architecture Decisions

### Hardware Platform
- **Processing Unit**: NVIDIA Jetson Orin Nano 8GB (40-67 TOPS)
- **LED Controller**: DigiOcta board running WLED firmware
- **Communication**: UDP protocol to WLED over WiFi
- **Display**: 1080p texture as universal interface between producer/consumer

### Software Stack
- **Language**: Python for all components
- **Video Processing**: FFmpeg via python-ffmpeg bindings
- **Optimization**: PyTorch for GPU-accelerated matrix operations
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
- Triple-buffered ring buffer (3× 1920×1080×4 RGBA frames)
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

**Purpose**: Core optimization algorithm that maps texture data to LED brightness values.

**Key Components**:
- LED position mapping (3,200 random positions)
- Diffusion pattern database (captured light patterns per LED)
- TensorFlow/PyTorch optimization solver
- GPU-accelerated matrix operations
- Real-time performance targeting 15fps

**Algorithm Flow**:
1. Sample input texture at LED positions
2. Load corresponding diffusion patterns
3. Solve optimization: minimize ||A×x - target||² where x = LED brightness values
4. Apply brightness/color corrections
5. Output RGB values for each LED

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

## Implementation Tasks (In Order)

### Phase 1: Core Infrastructure

#### Task 1.1: Shared Memory Ring Buffer
**Test**: Create buffer, write test data, read from another process, verify data integrity and synchronization.

#### Task 1.2: Control State Manager  
**Test**: Set/get values across processes, verify event synchronization, test cleanup.

#### Task 1.3: Basic Content Source Plugin Architecture
**Test**: Load a single static image, verify plugin interface, test resource cleanup.

### Phase 2: Producer Components

#### Task 2.1: Video Source Plugin with FFmpeg
**Test**: Decode a short test video, verify frame output, test hardware acceleration detection.

#### Task 2.2: Image Source Plugin
**Test**: Load various image formats, verify scaling/format conversion, test memory usage.

#### Task 2.3: Producer Process Core
**Test**: Load content, render to shared memory, verify frame timing and buffer management.

### Phase 3: Consumer Components  

#### Task 3.1: LED Position Mapping System
**Test**: Capture LED position images (1 image per LED), manage storage of multiple capture versions

#### Task 3.2: Basic LED Optimization Engine
**Test**: Simple optimization with mock diffusion patterns, verify output format, benchmark performance.

#### Task 3.3: WLED UDP Communication
**Test**: Send test patterns to WLED, verify protocol compliance, test error handling.

#### Task 3.4: Consumer Process Core
**Test**: Read from shared memory, run optimization, send to WLED, measure end-to-end latency.

### Phase 4: Web Interface

#### Task 4.1: FastAPI Backend with Basic Endpoints
**Test**: API endpoints respond correctly, file upload works, WebSocket connections establish.

#### Task 4.2: React Frontend Core Components
**Test**: UI renders correctly, API calls work, responsive design functions on mobile.

#### Task 4.3: File Upload and Management System
**Test**: Upload various file types, verify preprocessing, test content validation.

### Phase 5: Integration and Advanced Features

#### Task 5.1: Multi-Process Orchestration
**Test**: All processes start/stop correctly, error handling works, resource cleanup is complete.

#### Task 5.2: Real-time Web Interface Features
**Test**: Live LED preview updates, system status monitoring, playlist control responsiveness.

#### Task 5.3: Advanced LED Optimization
**Test**: Load real diffusion pattern data, optimize complex images, verify output quality.

### Phase 6: System Integration

#### Task 6.1: Hardware Integration Testing
**Test**: Full system with actual Jetson hardware, WLED controller, network communication.

#### Task 6.2: Performance Optimization and Profiling
**Test**: Achieve target 15fps performance, optimize memory usage, minimize latency.

#### Task 6.3: Production Deployment and Configuration
**Test**: Headless operation, WiFi hotspot mode, system startup/shutdown procedures.

## Technical Requirements

### Performance Targets
- Consumer process: ≥15fps LED optimization
- Memory usage: <2GB total system RAM
- Network: <100ms latency for web interface responses
- Storage: Efficient content management and cleanup

### Hardware Acceleration
- FFmpeg nvdec for video decoding
- TensorFlow GPU acceleration for optimization
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
│   │       ├── image_source.py
│   │       └── animation_source.py
│   ├── consumer/             # Consumer process components
│   │   ├── consumer.py       # Consumer process
│   │   └── led_optimizer.py  # Optimization engine
│   └── web/                  # Web interface components
│       ├── api_server.py     # FastAPI backend
│       └── frontend/         # React application
│           ├── src/
│           ├── package.json
│           └── vite.config.js
├── tests/                    # Unit and integration tests
└── config/                   # Configuration files
    ├── led_positions.json
    ├── diffusion_patterns.npz
    └── system_config.yaml
```

## Development Commands

```bash
# Activate virtual environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_shared_buffer.py -v

# Run individual test
python -m pytest tests/test_shared_buffer.py::TestFrameRingBuffer::test_initialization -v

# Import modules from new structure
python -c "from src.core import FrameRingBuffer, ControlState"
python -c "from src.producer import ContentSourceRegistry, ImageSource"

# Run modules directly (when implemented)
python -m src.producer.producer
python -m src.consumer.consumer
python -m src.web.api_server
```