"""
Shared pytest fixtures for Prismatron test suite.

Provides common fixtures for mocking system components, creating test data,
and managing test resources across all test modules.
"""

import json
import os

# Add src to path for imports
import sys
import tempfile
from pathlib import Path
from typing import Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Frame and LED Data Fixtures
# =============================================================================


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Generate a sample RGB frame (64x64x3)."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_led_data() -> np.ndarray:
    """Generate sample LED color data for 2624 LEDs."""
    return np.random.randint(0, 256, (2624, 3), dtype=np.uint8)


@pytest.fixture
def black_frame() -> np.ndarray:
    """Generate a black (all zeros) frame."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture
def white_frame() -> np.ndarray:
    """Generate a white (all 255) frame."""
    return np.full((64, 64, 3), 255, dtype=np.uint8)


@pytest.fixture
def gradient_frame() -> np.ndarray:
    """Generate a horizontal gradient frame."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for x in range(64):
        frame[:, x, :] = int(x * 255 / 63)
    return frame


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_uploads_dir(temp_dir: Path) -> Path:
    """Create a temporary uploads directory."""
    uploads = temp_dir / "uploads"
    uploads.mkdir(parents=True)
    return uploads


@pytest.fixture
def temp_media_dir(temp_dir: Path) -> Path:
    """Create a temporary media directory."""
    media = temp_dir / "media"
    media.mkdir(parents=True)
    return media


@pytest.fixture
def temp_playlists_dir(temp_dir: Path) -> Path:
    """Create a temporary playlists directory."""
    playlists = temp_dir / "playlists"
    playlists.mkdir(parents=True)
    return playlists


# =============================================================================
# Playlist Fixtures
# =============================================================================


@pytest.fixture
def sample_playlist_data() -> Dict:
    """Create sample playlist data structure."""
    return {
        "version": "1.0",
        "name": "Test Playlist",
        "description": "A test playlist",
        "created_at": "2024-01-01T00:00:00",
        "modified_at": "2024-01-01T00:00:00",
        "auto_repeat": True,
        "shuffle": False,
        "items": [
            {
                "id": "item-1",
                "name": "test_video.mp4",
                "type": "video",
                "file_path": "/path/to/test_video.mp4",
                "duration": 30.0,
                "order": 0,
                "transition_in": {"type": "fade", "parameters": {"duration": 1.0}},
                "transition_out": {"type": "fade", "parameters": {"duration": 1.0}},
            },
            {
                "id": "item-2",
                "name": "test_image.png",
                "type": "image",
                "file_path": "/path/to/test_image.png",
                "duration": 10.0,
                "order": 1,
                "transition_in": {"type": "none", "parameters": {}},
                "transition_out": {"type": "none", "parameters": {}},
            },
        ],
    }


@pytest.fixture
def sample_playlist_file(temp_playlists_dir: Path, sample_playlist_data: Dict) -> Path:
    """Create a sample playlist JSON file."""
    playlist_path = temp_playlists_dir / "test_playlist.json"
    with open(playlist_path, "w") as f:
        json.dump(sample_playlist_data, f)
    return playlist_path


# =============================================================================
# Mock Component Fixtures
# =============================================================================


@pytest.fixture
def mock_control_state() -> MagicMock:
    """Create a mock ControlState object."""
    mock = MagicMock()
    mock.get_brightness.return_value = 1.0
    mock.get_producer_state.return_value = "PLAYING"
    mock.get_renderer_state.return_value = "PLAYING"
    mock.get_rendering_index.return_value = 0
    mock.is_playing.return_value = True
    mock.is_paused.return_value = False
    mock.get_fps.return_value = 30.0
    mock.get_input_fps.return_value = 30.0
    mock.get_output_fps.return_value = 30.0
    mock.get_dropped_frames_percentage.return_value = 0.0
    mock.get_late_frame_percentage.return_value = 0.0
    mock.get_optimization_iterations.return_value = 10
    return mock


@pytest.fixture
def mock_playlist_sync_client() -> MagicMock:
    """Create a mock PlaylistSyncClient."""
    mock = MagicMock()
    mock.connected = True
    mock.play.return_value = True
    mock.pause.return_value = True
    mock.stop.return_value = True
    mock.next.return_value = True
    mock.previous.return_value = True
    mock.add_item.return_value = True
    mock.remove_item.return_value = True
    mock.clear_playlist.return_value = True
    mock.get_state.return_value = {
        "items": [],
        "current_index": 0,
        "is_playing": False,
    }
    return mock


@pytest.fixture
def mock_network_manager() -> MagicMock:
    """Create a mock NetworkManager."""
    from src.network.models import APConfig, NetworkMode, NetworkStatus

    mock = MagicMock()
    mock.interface = "wlan0"
    mock.ethernet_interface = "eth0"

    # Mock get_status
    mock.get_status = MagicMock(
        return_value=NetworkStatus(
            mode=NetworkMode.CLIENT,
            connected=True,
            interface="wlan0",
            ip_address="192.168.1.100",
            ssid="TestNetwork",
            signal_strength=80,
            gateway="192.168.1.1",
            dns_servers=["8.8.8.8"],
        )
    )

    # Mock scan_networks - async method
    async def mock_scan():
        from src.network.models import WiFiNetwork, WiFiSecurity

        return [
            WiFiNetwork(
                ssid="TestNetwork1",
                bssid="00:11:22:33:44:55",
                signal_strength=80,
                frequency=2437,
                security=WiFiSecurity.WPA2,
                connected=False,
            ),
            WiFiNetwork(
                ssid="TestNetwork2",
                bssid="00:11:22:33:44:66",
                signal_strength=60,
                frequency=5180,
                security=WiFiSecurity.WPA3,
                connected=False,
            ),
        ]

    mock.scan_networks = mock_scan

    # Mock connect_to_network - async method
    async def mock_connect(ssid, password=None):
        return True

    mock.connect_to_network = mock_connect

    # Mock disconnect - async method
    async def mock_disconnect():
        return True

    mock.disconnect = mock_disconnect

    # Mock enable_ap - async method
    async def mock_enable_ap():
        return True

    mock.enable_ap = mock_enable_ap

    # Mock disable_ap - async method
    async def mock_disable_ap():
        return True

    mock.disable_ap = mock_disable_ap

    return mock


@pytest.fixture
def mock_wled_client() -> MagicMock:
    """Create a mock WLEDClient."""
    mock = MagicMock()
    mock.connect.return_value = True
    mock.disconnect.return_value = True
    mock.send_frame.return_value = True
    mock.get_statistics.return_value = {
        "packets_sent": 100,
        "bytes_sent": 10000,
        "errors": 0,
    }
    return mock


@pytest.fixture
def mock_shared_buffer() -> MagicMock:
    """Create a mock SharedBuffer for testing."""
    mock = MagicMock()
    mock.connect.return_value = True
    mock.read_frame.return_value = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    mock.write_frame.return_value = True
    mock.get_stats.return_value = {
        "frames_written": 100,
        "frames_read": 100,
        "buffer_size": 10,
    }
    return mock


@pytest.fixture
def mock_content_source() -> MagicMock:
    """Create a mock ContentSource."""
    mock = MagicMock()
    mock.get_frame.return_value = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    mock.is_ready.return_value = True
    mock.get_duration.return_value = 10.0
    mock.get_progress.return_value = 0.5
    return mock


# =============================================================================
# System Info Fixtures
# =============================================================================


@pytest.fixture
def mock_psutil() -> Generator[MagicMock, None, None]:
    """Mock psutil for system stats."""
    with patch("psutil.virtual_memory") as mock_mem, patch("psutil.cpu_percent") as mock_cpu:
        mock_mem.return_value = MagicMock(percent=50.0, used=4 * 1024**3, total=8 * 1024**3)
        mock_cpu.return_value = 25.0
        yield {"memory": mock_mem, "cpu": mock_cpu}


@pytest.fixture
def mock_tegrastats() -> MagicMock:
    """Create a mock TegrastatsMonitor."""
    mock = MagicMock()
    mock.get_latest_stats.return_value = MagicMock(
        gpu_usage=30.0,
        cpu_usage=25.0,
        memory_used=4000,
        memory_total=8000,
    )
    return mock


# =============================================================================
# Audio Fixtures
# =============================================================================


@pytest.fixture
def sample_audio_buffer() -> np.ndarray:
    """Generate a sample audio buffer (1 second at 44100 Hz)."""
    # Generate a simple sine wave
    sample_rate = 44100
    duration = 1.0
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    return audio


@pytest.fixture
def sample_audio_config() -> Dict:
    """Create sample audio reactive configuration."""
    return {
        "enabled": True,
        "source": "system",
        "beat_brightness": True,
        "rules": [
            {
                "id": "rule-1",
                "trigger": "beat",
                "action": "brightness_pulse",
                "parameters": {"intensity": 1.5, "decay": 0.5},
            }
        ],
    }


# =============================================================================
# Effect Fixtures
# =============================================================================


@pytest.fixture
def sample_effect_config() -> Dict:
    """Create sample effect configuration."""
    return {
        "id": "rainbow_cycle",
        "name": "Rainbow Cycle",
        "description": "Smooth rainbow color cycling",
        "config": {
            "speed": 1.0,
            "brightness": 1.0,
            "hue_shift": 0.0,
        },
        "category": "color",
        "icon": "rainbow",
    }


# =============================================================================
# Test Image/Video File Fixtures
# =============================================================================


@pytest.fixture
def sample_image_file(temp_uploads_dir: Path) -> Path:
    """Create a sample PNG image file."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), color="red")
    img_path = temp_uploads_dir / "test_image.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_video_metadata() -> Dict:
    """Sample video metadata from ffprobe."""
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "duration": "30.0",
                "bit_rate": "5000000",
            }
        ],
        "format": {
            "filename": "test_video.mp4",
            "duration": "30.0",
            "size": "15000000",
            "bit_rate": "4000000",
        },
    }


# =============================================================================
# Subprocess Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_subprocess_run() -> Generator[MagicMock, None, None]:
    """Mock subprocess.run for command execution tests."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="success",
            stderr="",
        )
        yield mock_run


@pytest.fixture
def mock_asyncio_subprocess() -> Generator[MagicMock, None, None]:
    """Mock asyncio.create_subprocess_exec for async command tests."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = MagicMock(return_value=(b"success", b""))
        mock_exec.return_value = mock_process
        yield mock_exec


# =============================================================================
# FastAPI Test Client Fixtures
# =============================================================================


@pytest.fixture
def api_test_client():
    """Create a FastAPI TestClient for API testing.

    Note: This fixture patches global state to avoid import side effects.
    """
    from unittest.mock import MagicMock, patch

    # Patch all the global state and external dependencies before importing
    patches = [
        patch("src.web.api_server.control_state", MagicMock()),
        patch("src.web.api_server.playlist_sync_client", None),
        patch("src.web.api_server.network_manager", None),
        patch("src.web.api_server.tegrastats_monitor", None),
        patch("src.web.api_server.system_settings", MagicMock(led_count=2624)),
    ]

    for p in patches:
        p.start()

    try:
        from fastapi.testclient import TestClient

        from src.web.api_server import app

        client = TestClient(app)
        yield client
    finally:
        for p in patches:
            p.stop()


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_environment():
    """Clean up environment after each test."""
    yield
    # Any cleanup code here


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "hardware: marks tests that require hardware")
    config.addinivalue_line("markers", "network: marks tests that require network access")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
