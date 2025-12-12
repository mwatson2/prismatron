[![CI](https://github.com/mwatson2/prismatron/workflows/CI/badge.svg)](https://github.com/mwatson2/prismatron/actions?query=workflow%3ACI)
[![Coverage](https://codecov.io/github/mwatson2/prismatron/branch/main/graph/badge.svg?token=1YYVB2YHFG)](https://codecov.io/github/mwatson2/prismatron)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Prismatron LED Display System

A real-time LED display optimization system that captures diffusion patterns from physical LEDs and uses sparse matrix optimization to reproduce video content on diffused LED arrays.

## Hardware Requirements

- **NVIDIA Jetson Orin Nano** (8GB) - Primary compute platform
- **WLED Controller** - LED controller with DDP protocol support (e.g. WLED platforms such as QuinLED)
- **USB Camera** - For diffusion pattern capture and calibration
- **LED Array** - Tested to 3200 with Jetson Orin Nano

## Project Structure

```
prismatron/
├── src/                    # Source code
│   ├── consumer/           # LED optimization and WLED communication
│   ├── producer/           # Video/image processing, FFmpeg integration
│   ├── web/                # FastAPI backend + React frontend
│   └── utils/              # Sparse matrix utilities, shared memory
├── tools/                  # Calibration and capture utilities
├── scripts/                # System setup and service scripts
├── tests/                  # Unit tests (190+ tests)
└── local/                  # Temporary outputs (gitignored)
```

## Environment Setup

### 1. Clone and Create Virtual Environment

```bash
git clone <repository-url> prismatron
cd prismatron
python3 -m venv env
source env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- **CuPy** - GPU-accelerated computing
- **SciPy** - Sparse matrix operations
- **OpenCV** - Camera capture and image processing
- **FastAPI** - Web API server
- **NumPy** - Numerical computing

### 3. Setup Data Directories

```bash
# Default location: /mnt/prismatron
make setup

# Custom location
make setup DATA_DIR=/path/to/data
```

This creates the runtime data directory with:
- `config/` - Configuration files
- `patterns/` - Captured diffusion patterns
- `media/` - Video/image content
- `playlists/` - Content playlists
- `uploads/` - User uploads
- `logs/` - Runtime logs

The `DATA_DIR` variable can be passed to any make command that uses data paths (e.g., `capture`, `matrices`, `run`).

## Make Commands

Run `make help` to see all available commands:

### Setup & Installation

| Command | Description |
|---------|-------------|
| `make setup [DATA_DIR=path]` | Create data directories and initialize config |
| `make frontend` | Build the React web frontend |
| `make install-service` | Install systemd user service (includes dummy display) |
| `make uninstall-service` | Remove systemd user service |

### Calibration Pipeline

The calibration pipeline captures LED diffusion patterns for the optimization engine:

```bash
# Step 1: Calibrate camera region of interest
make calibrate

# Step 2: Capture LED diffusion patterns
make capture

# Step 3: Compute optimization matrices
make matrices PATTERN=your-patterns.npz
```

| Command | Description |
|---------|-------------|
| `make calibrate` | Interactive camera calibration |
| `make capture` | Capture LED diffusion patterns (prompts for WLED host) |
| `make matrices PATTERN=file.npz` | Compute ATA matrices from patterns |

### Development

| Command | Description |
|---------|-------------|
| `make test` | Run pytest test suite |
| `make pre-commit` | Run all pre-commit hooks |
| `make dev-server` | Start API server with hot reload |
| `make run` | Run prismatron directly (not as service) |
| `make clean` | Clean build artifacts and cache |

### Service Management

| Command | Description |
|---------|-------------|
| `make service-status` | Show systemd service status |
| `make service-logs` | Follow service logs (journalctl) |

## Calibration Workflow

### 1. Camera Calibration (`make calibrate`)

Opens an interactive window to select the display region:
- Click and drag to select crop region
- Press `g` for grid overlay, `a` for aspect ratio guides
- Press `s` to save, `q` to quit

### 2. Pattern Capture (`make capture`)

Captures diffusion patterns by illuminating each LED individually:
- Prompts for WLED controller IP address
- Prompts for output filename
- Shows live preview during capture
- Captures ~9600 patterns (3200 LEDs x 3 channels)
- Takes approximately 16 minutes at default settings

### 3. Matrix Computation (`make matrices`)

Computes optimization matrices from captured patterns:
```bash
make matrices PATTERN=capture-1207-linear.npz
```

This generates:
- ATA matrices for LSQR optimization
- Symmetric diagonal matrices for efficient computation

## Service Installation

### Install as User Service

```bash
make install-service
```

This will:
1. Install the X11 dummy display config (for headless GPU access)
2. Start the dummy display on `:99`
3. Install and enable the prismatron systemd user service

### Service Commands

```bash
# Start the service
systemctl --user start prismatron

# Stop the service
systemctl --user stop prismatron

# View status
systemctl --user status prismatron

# View logs
journalctl --user -u prismatron -f
```

## Configuration

The main configuration file is at `/mnt/prismatron/config/config.json`:

```json
{
  "debug": false,
  "web_host": "0.0.0.0",
  "web_port": 8080,
  "wled_hosts": ["192.168.4.174"],
  "wled_port": 4048,
  "diffusion_patterns": "capture-0813-linear.npz",
  "optimization_iterations": 10
}
```

Key settings:
- `wled_hosts` - List of WLED controller IPs to try (in order)
- `diffusion_patterns` - Pattern file to use for optimization
- `optimization_iterations` - LSQR iterations (0 = pseudo-inverse only)

## Web Interface

Access the web interface at `http://<jetson-ip>:8080`

Features:
- Real-time playback control
- Content upload and management
- Playlist creation and editing
- System status monitoring

## Development

### Running Tests

```bash
source env/bin/activate
make test
```

### Pre-commit Hooks

Always run pre-commit before committing:

```bash
make pre-commit
```

### Frontend Development

```bash
# Development server with hot reload
make frontend-dev

# Production build
make frontend
```

## Technical Details

### Optimization Pipeline

1. **Frame Capture**: FFmpeg decodes video to 800x480 frames
2. **Shared Memory**: Ring buffer passes frames between processes
3. **Sparse Optimization**: CuPy/SciPy LSQR solver computes LED values
4. **DDP Output**: LED values sent to WLED controllers via UDP

### Memory Optimization

- Uses uint8 pattern storage (75% memory reduction vs float32)
- Custom mixed sparse tensor for diffusion patterns
- Custom symmetric block diagonal for ATA matrices
- Designed for 8GB Jetson Orin Nano constraints

## License

See LICENSE.txt
