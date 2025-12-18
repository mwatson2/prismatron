# Test Coverage Improvement Plan

**Generated:** 2025-12-07
**Current Coverage:** 38.78% (12,114 of 19,788 statements missing)
**Target Coverage:** 70%+

---

## Executive Summary

The Prismatron codebase has significant coverage gaps, with several critical modules at 0% coverage. This plan prioritizes modules by:
1. **Statement count** - High-impact modules with many uncovered statements
2. **Criticality** - Core system components vs utilities
3. **Testability** - Modules that can be unit tested vs those requiring integration tests

---

## Priority 1: Critical Zero-Coverage Modules (0%)

These modules have **zero test coverage** and represent significant technical debt.

### 1.1 Web Layer (2,726 statements)

| Module | Statements | Current | Priority |
|--------|------------|---------|----------|
| `src/web/api_server.py` | 2,345 | 0% | HIGH |
| `src/web/video_converter.py` | 381 | 0% | MEDIUM |

**Test Strategy for `api_server.py`:**
```python
# tests/web/test_api_server.py
import pytest
from fastapi.testclient import TestClient

class TestAPIEndpoints:
    """Test FastAPI endpoints with TestClient"""

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_endpoint(self, client):
        response = client.get("/api/status")
        assert response.status_code == 200

    # Additional endpoint tests...

class TestWebSocketHandlers:
    """Test WebSocket connections and message handling"""

    async def test_websocket_connect(self, client):
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert "status" in data
```

**Test Strategy for `video_converter.py`:**
```python
# tests/web/test_video_converter.py
class TestVideoConverter:
    """Test video conversion functionality"""

    def test_probe_video_metadata(self, sample_video):
        converter = VideoConverter()
        metadata = converter.probe(sample_video)
        assert "duration" in metadata

    def test_convert_to_web_format(self, sample_video, tmp_path):
        converter = VideoConverter()
        output = tmp_path / "output.mp4"
        converter.convert(sample_video, output, format="web")
        assert output.exists()
```

---

### 1.2 Network Layer (435 statements)

| Module | Statements | Current | Priority |
|--------|------------|---------|----------|
| `src/network/manager.py` | 382 | 0% | HIGH |
| `src/network/models.py` | 50 | 0% | LOW |
| `src/network/__init__.py` | 3 | 0% | LOW |

**Test Strategy for `network/manager.py`:**
```python
# tests/network/test_manager.py
class TestNetworkManager:
    """Test network configuration management"""

    def test_get_current_connection(self, mock_nmcli):
        manager = NetworkManager()
        connection = manager.get_current_connection()
        assert connection is not None

    def test_scan_wifi_networks(self, mock_nmcli):
        manager = NetworkManager()
        networks = manager.scan_networks()
        assert isinstance(networks, list)

    def test_connect_to_network(self, mock_nmcli):
        manager = NetworkManager()
        result = manager.connect("TestNetwork", "password123")
        assert result.success
```

**Note:** Network tests should mock `nmcli` and system calls to avoid requiring actual network hardware.

---

### 1.3 Transition System (0% modules: 577 statements)

| Module | Statements | Current | Priority |
|--------|------------|---------|----------|
| `src/consumer/led_effect_transitions.py` | 118 | 0% | MEDIUM |
| `src/consumer/led_transition_processor.py` | 100 | 0% | MEDIUM |

**Test Strategy:**
```python
# tests/consumer/test_led_effect_transitions.py
class TestLEDEffectTransitions:
    """Test LED effect transition calculations"""

    def test_fade_transition_progress(self):
        transition = LEDEffectTransition(duration=1.0, type="fade")
        assert transition.get_progress(0.5) == 0.5

    def test_blend_frames(self):
        frame1 = np.zeros((100, 3), dtype=np.uint8)
        frame2 = np.full((100, 3), 255, dtype=np.uint8)
        result = transition.blend(frame1, frame2, 0.5)
        assert np.allclose(result, 127, atol=1)
```

---

### 1.4 Utility Modules (0% coverage)

| Module | Statements | Current | Priority |
|--------|------------|---------|----------|
| `src/utils/optimization_utils.py` | 170 | 0% | MEDIUM |
| `src/utils/log_rotation.py` | 144 | 0% | LOW |
| `src/utils/logging_utils.py` | 25 | 0% | LOW |
| `src/paths.py` | 25 | 0% | LOW |

**Test Strategy for `optimization_utils.py`:**
```python
# tests/utils/test_optimization_utils.py
class TestOptimizationUtils:
    """Test optimization utility functions"""

    def test_compute_gradient(self):
        # Test gradient computation
        pass

    def test_line_search(self):
        # Test line search algorithm
        pass
```

---

### 1.5 Deprecated/Unused Kernel Variants (173 statements)

| Module | Statements | Current | Notes |
|--------|------------|---------|-------|
| `src/utils/kernels/compute_optimized_3d_batch.py` | 40 | 0% | Deprecated |
| `src/utils/kernels/compute_optimized_3d_batch_int8.py` | 46 | 0% | Deprecated |
| `src/utils/kernels/compute_optimized_3d_batch_v3.py` | 56 | 0% | Deprecated |
| `src/utils/kernels/compute_optimized_3d_batch_v4_int8.py` | 31 | 0% | Deprecated |

**Recommendation:** Consider removing these deprecated kernel files from the codebase or excluding them from coverage metrics if they're kept for reference.

---

## Priority 2: Very Low Coverage Modules (<25%)

### 2.1 Core Consumer Components

| Module | Statements | Missing | Current | Priority |
|--------|------------|---------|---------|----------|
| `src/consumer/consumer.py` | 1,150 | 758 | 34% | CRITICAL |
| `src/consumer/frame_renderer.py` | 900 | 655 | 27% | HIGH |
| `src/consumer/wled_client.py` | 416 | 347 | 17% | HIGH |
| `src/consumer/led_buffer.py` | 131 | 113 | 14% | MEDIUM |
| `src/consumer/transition_processor.py` | 86 | 69 | 20% | MEDIUM |

**Test Strategy for `consumer.py`:**
```python
# tests/consumer/test_consumer.py
class TestConsumer:
    """Test main consumer process"""

    @pytest.fixture
    def mock_shared_buffer(self):
        """Create mock shared buffer for testing"""
        return MockSharedBuffer()

    @pytest.fixture
    def mock_wled_sink(self):
        """Create mock WLED sink"""
        return MockWLEDSink()

    def test_consumer_initialization(self, mock_shared_buffer, mock_wled_sink):
        consumer = Consumer(
            shared_buffer=mock_shared_buffer,
            wled_sink=mock_wled_sink
        )
        assert consumer.is_initialized

    def test_process_frame(self, consumer, sample_frame):
        result = consumer.process_frame(sample_frame)
        assert result.shape == (NUM_LEDS, 3)

    def test_frame_optimization(self, consumer):
        # Test frame optimization pipeline
        pass
```

**Test Strategy for `wled_client.py`:**
```python
# tests/consumer/test_wled_client.py
class TestWLEDClient:
    """Test WLED UDP client"""

    @pytest.fixture
    def mock_socket(self, mocker):
        return mocker.patch("socket.socket")

    def test_connect(self, mock_socket):
        client = WLEDClient("192.168.1.100")
        client.connect()
        mock_socket.return_value.connect.assert_called()

    def test_send_frame(self, mock_socket, sample_led_data):
        client = WLEDClient("192.168.1.100")
        client.send_frame(sample_led_data)
        mock_socket.return_value.send.assert_called()

    def test_reconnect_on_error(self, mock_socket):
        # Test automatic reconnection
        pass
```

---

### 2.2 Core Producer Components

| Module | Statements | Missing | Current | Priority |
|--------|------------|---------|---------|----------|
| `src/producer/producer.py` | 854 | 570 | 33% | CRITICAL |
| `src/producer/content_sources/text_source.py` | 302 | 271 | 10% | MEDIUM |
| `src/producer/effect_source.py` | 239 | 205 | 14% | MEDIUM |
| `src/producer/content_preparer.py` | 101 | 78 | 23% | MEDIUM |

**Test Strategy for `producer.py`:**
```python
# tests/producer/test_producer.py
class TestProducer:
    """Test main producer process"""

    @pytest.fixture
    def mock_content_source(self):
        return MockContentSource()

    @pytest.fixture
    def mock_shared_buffer(self):
        return MockSharedBuffer()

    def test_producer_initialization(self, mock_shared_buffer):
        producer = Producer(shared_buffer=mock_shared_buffer)
        assert producer.is_initialized

    def test_load_playlist(self, producer, sample_playlist):
        producer.load_playlist(sample_playlist)
        assert len(producer.playlist) > 0

    def test_produce_frame(self, producer):
        frame = producer.produce_frame()
        assert frame.shape == (HEIGHT, WIDTH, 3)
```

**Test Strategy for `text_source.py`:**
```python
# tests/producer/content_sources/test_text_source.py
class TestTextSource:
    """Test text rendering source"""

    def test_render_text(self):
        source = TextSource(text="Hello", font_size=24)
        frame = source.get_frame()
        assert frame is not None

    def test_scrolling_text(self):
        source = TextSource(text="Scrolling", scroll=True)
        frame1 = source.get_frame()
        frame2 = source.get_frame()
        assert not np.array_equal(frame1, frame2)
```

---

### 2.3 Transition System (Low Coverage)

| Module | Statements | Missing | Current | Priority |
|--------|------------|---------|---------|----------|
| `src/transitions/blur_transition.py` | 315 | 284 | 10% | LOW |
| `src/transitions/fade_transition.py` | 95 | 81 | 15% | LOW |
| `src/transitions/led_blur_transition.py` | 112 | 97 | 13% | LOW |
| `src/transitions/led_fade_transition.py` | 98 | 84 | 14% | LOW |
| `src/transitions/led_random_transition.py` | 139 | 123 | 12% | LOW |
| `src/transitions/led_transition_factory.py` | 113 | 87 | 23% | LOW |
| `src/transitions/transition_factory.py` | 130 | 89 | 32% | LOW |

**Test Strategy:**
```python
# tests/transitions/test_transitions.py
class TestFadeTransition:
    def test_fade_progress_calculation(self):
        transition = FadeTransition(duration=1.0)
        assert transition.calculate_alpha(0.0) == 0.0
        assert transition.calculate_alpha(0.5) == 0.5
        assert transition.calculate_alpha(1.0) == 1.0

    def test_fade_blend(self, frame_a, frame_b):
        transition = FadeTransition(duration=1.0)
        result = transition.blend(frame_a, frame_b, 0.5)
        expected = (frame_a + frame_b) / 2
        assert np.allclose(result, expected, atol=1)

class TestBlurTransition:
    def test_blur_kernel_generation(self):
        transition = BlurTransition(blur_size=5)
        kernel = transition.get_kernel(0.5)
        assert kernel.shape == (5, 5)

class TestTransitionFactory:
    def test_create_fade(self):
        transition = TransitionFactory.create("fade", duration=1.0)
        assert isinstance(transition, FadeTransition)

    def test_create_blur(self):
        transition = TransitionFactory.create("blur", duration=1.0)
        assert isinstance(transition, BlurTransition)
```

---

### 2.4 Utility Classes (Low Coverage)

| Module | Statements | Missing | Current | Priority |
|--------|------------|---------|---------|----------|
| `src/utils/led_diffusion_csc_matrix.py` | 211 | 174 | 18% | MEDIUM |
| `src/utils/batch_frame_optimizer.py` | 128 | 107 | 16% | MEDIUM |
| `src/utils/dense_ata_matrix.py` | 99 | 82 | 17% | LOW (deprecated) |
| `src/utils/pattern_loader.py` | 61 | 52 | 15% | MEDIUM |
| `src/utils/tegrastats.py` | 130 | 99 | 24% | LOW |

---

## Priority 3: Moderate Coverage Modules (25-50%)

### 3.1 Audio System

| Module | Statements | Missing | Current | Target |
|--------|------------|---------|---------|--------|
| `src/consumer/audio_beat_analyzer.py` | 708 | 343 | 52% | 70% |
| `src/consumer/audio_capture.py` | 349 | 184 | 47% | 70% |

**Test Strategy:**
```python
# tests/consumer/test_audio_beat_analyzer.py
class TestAudioBeatAnalyzer:
    def test_beat_detection(self, sample_audio_with_beats):
        analyzer = AudioBeatAnalyzer()
        beats = analyzer.detect_beats(sample_audio_with_beats)
        assert len(beats) > 0

    def test_frequency_analysis(self, sample_audio):
        analyzer = AudioBeatAnalyzer()
        spectrum = analyzer.get_frequency_spectrum(sample_audio)
        assert spectrum.shape[0] > 0
```

---

### 3.2 Core Infrastructure

| Module | Statements | Missing | Current | Target |
|--------|------------|---------|---------|--------|
| `src/core/playlist_sync.py` | 467 | 348 | 25% | 60% |
| `src/core/control_state.py` | 488 | 172 | 65% | 80% |

---

### 3.3 LED Optimization

| Module | Statements | Missing | Current | Target |
|--------|------------|---------|---------|--------|
| `src/consumer/led_optimizer.py` | 625 | 361 | 42% | 70% |
| `src/consumer/led_effect.py` | 336 | 184 | 45% | 70% |
| `src/consumer/preview_sink.py` | 246 | 131 | 47% | 70% |

---

### 3.4 Kernel Implementations

| Module | Statements | Missing | Current | Target |
|--------|------------|---------|---------|--------|
| `src/utils/kernels/dia_matvec.py` | 229 | 119 | 48% | 70% |
| `src/utils/kernels/precompiled_mma_kernel.py` | 288 | 169 | 41% | 60% |
| `src/utils/kernels/pure_fp16_dia_kernel.py` | 42 | 29 | 31% | 60% |

---

## Implementation Plan

### Phase 1: Foundation Tests (Week 1-2)

**Goal:** Establish test infrastructure and cover zero-coverage critical modules

1. **Create test fixtures and mocks:**
   - `conftest.py` with shared fixtures
   - Mock classes for SharedBuffer, WLEDSink, ContentSource
   - Sample data generators (frames, LED data, audio)

2. **Web Layer Tests:**
   - FastAPI TestClient setup
   - Basic endpoint tests (health, status, config)
   - WebSocket connection tests

3. **Network Layer Tests:**
   - Mock nmcli commands
   - Test network scanning and connection

**Expected Impact:** +8-10% coverage

---

### Phase 2: Core Component Tests (Week 3-4)

**Goal:** Cover main producer and consumer functionality

1. **Producer Tests:**
   - `producer.py` initialization and frame production
   - Content source loading and switching
   - Playlist management

2. **Consumer Tests:**
   - `consumer.py` frame processing pipeline
   - `wled_client.py` connection and data sending
   - `frame_renderer.py` rendering logic

**Expected Impact:** +12-15% coverage

---

### Phase 3: Transition & Effect Tests (Week 5)

**Goal:** Cover visual effect and transition systems

1. **Transition Tests:**
   - All transition types (fade, blur, random)
   - Factory pattern tests
   - LED-specific transitions

2. **Effect Tests:**
   - Remaining effect edge cases
   - Effect transitions

**Expected Impact:** +5-7% coverage

---

### Phase 4: Audio & Utility Tests (Week 6)

**Goal:** Complete coverage for remaining modules

1. **Audio System:**
   - Beat detection accuracy
   - Frequency analysis
   - Audio capture edge cases

2. **Utilities:**
   - Pattern loader
   - Frame timing
   - Optimization utilities

**Expected Impact:** +5-7% coverage

---

## Test Infrastructure Requirements

### Required Test Fixtures

```python
# tests/conftest.py

@pytest.fixture
def sample_frame():
    """Generate a sample RGB frame"""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

@pytest.fixture
def sample_led_data():
    """Generate sample LED color data"""
    return np.random.randint(0, 256, (2600, 3), dtype=np.uint8)

@pytest.fixture
def mock_shared_buffer():
    """Create a mock shared buffer for testing"""
    class MockSharedBuffer:
        def __init__(self):
            self._frames = []
        def write(self, frame):
            self._frames.append(frame)
        def read(self):
            return self._frames[-1] if self._frames else None
    return MockSharedBuffer()

@pytest.fixture
def mock_wled_client(mocker):
    """Mock WLED UDP client"""
    return mocker.MagicMock()

@pytest.fixture
def temp_playlist(tmp_path):
    """Create a temporary playlist file"""
    playlist = {
        "items": [
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "image", "path": "/path/to/image.png"}
        ]
    }
    path = tmp_path / "playlist.json"
    path.write_text(json.dumps(playlist))
    return path
```

### Mock Classes Needed

1. **MockSharedBuffer** - Simulates ring buffer without multiprocessing
2. **MockWLEDSink** - Captures LED data without network
3. **MockContentSource** - Provides deterministic test frames
4. **MockAudioCapture** - Provides sample audio data
5. **MockNMCLI** - Simulates network management commands

---

## Coverage Exclusions

Consider excluding these from coverage metrics:

```ini
# pyproject.toml or .coveragerc
[coverage:run]
omit =
    # Deprecated kernel variants
    src/utils/kernels/compute_optimized_3d_batch.py
    src/utils/kernels/compute_optimized_3d_batch_int8.py
    src/utils/kernels/compute_optimized_3d_batch_v3.py
    src/utils/kernels/compute_optimized_3d_batch_v4_int8.py
    # Dense ATA (deprecated per CLAUDE.md)
    src/utils/dense_ata_matrix.py
```

---

## Metrics & Milestones

| Milestone | Coverage Target | Statements Covered |
|-----------|-----------------|-------------------|
| Current | 38.78% | 7,674 |
| Phase 1 Complete | 48% | ~9,500 |
| Phase 2 Complete | 60% | ~11,900 |
| Phase 3 Complete | 67% | ~13,250 |
| Phase 4 Complete | 72% | ~14,250 |

---

## Files by Coverage (Quick Reference)

### 0% Coverage (Immediate Action)
- `src/web/api_server.py` (2,345 statements)
- `src/network/manager.py` (382 statements)
- `src/web/video_converter.py` (381 statements)
- `src/utils/optimization_utils.py` (170 statements)
- `src/utils/log_rotation.py` (144 statements)
- `src/consumer/led_effect_transitions.py` (118 statements)
- `src/consumer/led_transition_processor.py` (100 statements)

### <25% Coverage (High Priority)
- `src/consumer/consumer.py` (758 missing)
- `src/consumer/frame_renderer.py` (655 missing)
- `src/producer/producer.py` (570 missing)
- `src/consumer/wled_client.py` (347 missing)
- `src/core/playlist_sync.py` (348 missing)
- `src/transitions/blur_transition.py` (284 missing)
- `src/producer/content_sources/text_source.py` (271 missing)
- `src/producer/effect_source.py` (205 missing)

### Well-Covered Modules (80%+)
- `src/const.py` (100%)
- `src/core/__init__.py` (100%)
- `src/producer/effects/base_effect.py` (99%)
- `src/utils/spatial_ordering.py` (98%)
- `src/producer/effects/environmental_effects.py` (97%)
- `src/utils/cuda_kernels.py` (100%)
- `src/utils/kernels/symmetric_dia_kernel.py` (100%)
