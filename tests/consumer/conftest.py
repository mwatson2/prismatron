"""
Shared fixtures for consumer module tests.

Provides common test fixtures for testing ConsumerProcess and related components
with varying levels of mocking - from pure unit tests to integration tests.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

# Skip all tests in this directory if cupy is not available
pytest.importorskip("cupy")

import cupy as cp

from src.const import FRAME_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def real_pattern_path():
    """Path to real pattern file for integration tests."""
    return "/mnt/prismatron/patterns/capture-0813-linear.npz"


@pytest.fixture
def test_frame_numpy():
    """Generate test frame with gradient content (numpy)."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, FRAME_WIDTH).astype(np.uint8)  # Red gradient
    frame[:, :, 1] = 128  # Green mid
    frame[:, :, 2] = 64  # Blue low
    return frame


@pytest.fixture
def test_frame_cupy(test_frame_numpy):
    """Generate test frame with gradient content (cupy)."""
    return cp.asarray(test_frame_numpy)


@pytest.fixture
def test_frame_batch_cupy(test_frame_numpy):
    """Generate batch of 8 test frames (cupy)."""
    frames = np.stack([test_frame_numpy] * 8)
    return cp.asarray(frames)


@pytest.fixture
def black_frame_numpy():
    """Generate black test frame (numpy)."""
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8)


@pytest.fixture
def black_frame_cupy(black_frame_numpy):
    """Generate black test frame (cupy)."""
    return cp.asarray(black_frame_numpy)


# =============================================================================
# Mock Component Fixtures
# =============================================================================


@pytest.fixture
def mock_frame_consumer():
    """Mock shared memory frame consumer."""
    consumer = MagicMock()
    consumer.connect.return_value = True
    consumer.cleanup = Mock()
    return consumer


@pytest.fixture
def mock_control_state():
    """Mock control state for IPC."""
    from src.core import ControlState
    from src.core.control_state import ProducerState, RendererState

    control = Mock(spec=ControlState)
    control.initialize.return_value = True

    # Create a mock status object
    mock_status = Mock()
    mock_status.audio_reactive_enabled = False
    mock_status.audio_enabled = False
    mock_status.beat_brightness_enabled = False
    mock_status.optimization_iterations = 10
    mock_status.renderer_state = RendererState.STOPPED
    mock_status.producer_state = ProducerState.STOPPED
    mock_status.current_playlist_index = 0
    mock_status.rendering_index = -1

    control.get_status.return_value = mock_status
    control.update_status = Mock()
    control.set_renderer_state = Mock()

    return control


@pytest.fixture
def mock_wled_sink():
    """Mock WLED network sink."""
    sink = MagicMock()
    sink.connect.return_value = True
    sink.disconnect = Mock()
    sink.send_led_values = Mock(return_value=True)
    sink.get_statistics.return_value = {"packets_sent": 0, "errors": 0}
    sink.is_connected.return_value = True
    return sink


@pytest.fixture
def mock_led_buffer():
    """Mock LED buffer."""
    buffer = MagicMock()
    buffer.write_led_values.return_value = True
    buffer.read_latest_led_values.return_value = None
    buffer.get_buffer_stats.return_value = {"current_count": 0, "capacity": 20}
    buffer.clear = Mock()
    return buffer


@pytest.fixture
def mock_frame_renderer():
    """Mock frame renderer."""
    renderer = MagicMock()
    renderer.is_initialized.return_value = True
    renderer.get_renderer_stats.return_value = {"frames_rendered": 0}
    renderer.render_frame = Mock()
    renderer.start = Mock()
    renderer.stop = Mock()
    return renderer


@pytest.fixture
def mock_led_optimizer():
    """Mock LED optimizer."""
    optimizer = MagicMock()
    optimizer.initialize.return_value = True
    optimizer._actual_led_count = 3200
    optimizer._matrix_loaded = True
    optimizer.supports_batch_optimization.return_value = True
    optimizer.get_optimizer_stats.return_value = {"optimization_count": 0}

    # Create mock optimization result
    mock_result = Mock()
    mock_result.converged = True
    mock_result.iterations = 10
    mock_result.led_values = np.random.randint(0, 255, (3200, 3), dtype=np.uint8)
    mock_result.error_metrics = {"mse": 0.01}

    optimizer.optimize_frame.return_value = mock_result
    return optimizer


# =============================================================================
# Mock Metadata Fixtures
# =============================================================================


@dataclass
class MockBufferMetadata:
    """Mock buffer metadata for testing."""

    presentation_timestamp: float = 0.0
    playlist_item_index: int = 0
    is_first_frame_of_item: bool = False
    timing_data: Optional[Any] = None
    has_presentation_timestamp: bool = True


@pytest.fixture
def mock_buffer_info():
    """Create mock buffer info object."""

    class MockBufferInfo:
        def __init__(self):
            self.metadata = MockBufferMetadata()
            self.data = None

    return MockBufferInfo()


@pytest.fixture
def mock_metadata_dict():
    """Create standard metadata dictionary."""
    return {
        "timestamp": 1.0,
        "playlist_item_index": 0,
        "is_first_frame_of_item": False,
        "rendering_index": 0,
    }


# =============================================================================
# Real Component Fixtures (for integration tests)
# =============================================================================


@pytest.fixture
def real_led_optimizer(real_pattern_path):
    """Create real LED optimizer with actual pattern file."""
    from src.consumer.led_optimizer import LEDOptimizer

    optimizer = LEDOptimizer(
        diffusion_patterns_path=real_pattern_path,
        enable_batch_mode=True,
    )
    if not optimizer.initialize():
        pytest.skip("Failed to initialize LED optimizer with real pattern")
    return optimizer


@pytest.fixture
def real_led_buffer():
    """Create real LED buffer for testing."""
    from src.consumer.led_buffer import LEDBuffer

    buffer = LEDBuffer(led_count=3200, buffer_size=10)
    yield buffer
    buffer.clear()
