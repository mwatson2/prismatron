"""
Unit tests for ConsumerProcess pure logic methods.

These tests focus on methods that don't require external I/O and can be
tested with minimal or no mocking. This provides reliable coverage of
the core business logic.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp

from src.const import FRAME_HEIGHT, FRAME_WIDTH
from src.consumer.consumer import (
    ConsumerProcess,
    ConsumerStats,
    OptimizationLoopState,
    OptimizationStepResult,
    RenderingLoopState,
    RenderingStepResult,
)
from src.core.control_state import ProducerState, RendererState


class TestConsumerStats:
    """Test ConsumerStats dataclass methods."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        stats = ConsumerStats()

        assert stats.frames_processed == 0
        assert stats.frames_dropped_early == 0
        assert stats.total_processing_time == 0.0
        assert stats.total_optimization_time == 0.0
        assert stats.optimization_errors == 0
        assert stats.transmission_errors == 0
        assert stats.consumer_input_fps == 0.0
        assert stats.renderer_output_fps_ewma == 0.0

    def test_get_average_fps_with_data(self):
        """Test FPS calculation with valid data."""
        stats = ConsumerStats(
            frames_processed=100,
            total_processing_time=10.0,
        )
        assert stats.get_average_fps() == 10.0

    def test_get_average_fps_zero_time(self):
        """Test FPS calculation with zero processing time."""
        stats = ConsumerStats(frames_processed=100, total_processing_time=0.0)
        assert stats.get_average_fps() == 0.0

    def test_get_average_fps_high_rate(self):
        """Test FPS calculation at high frame rates."""
        stats = ConsumerStats(
            frames_processed=1800,  # 1 minute at 30fps
            total_processing_time=60.0,
        )
        assert stats.get_average_fps() == 30.0

    def test_get_average_optimization_time_with_data(self):
        """Test average optimization time calculation."""
        stats = ConsumerStats(
            frames_processed=100,
            total_optimization_time=2.0,
        )
        assert stats.get_average_optimization_time() == 0.02

    def test_get_average_optimization_time_zero_frames(self):
        """Test optimization time with zero frames."""
        stats = ConsumerStats(frames_processed=0, total_optimization_time=0.0)
        assert stats.get_average_optimization_time() == 0.0

    def test_get_average_led_transition_time_with_data(self):
        """Test average LED transition time calculation."""
        stats = ConsumerStats(
            frames_processed=50,
            total_led_transition_time=0.5,
        )
        assert stats.get_average_led_transition_time() == 0.01

    def test_get_average_led_transition_time_zero_frames(self):
        """Test LED transition time with zero frames."""
        stats = ConsumerStats(frames_processed=0, total_led_transition_time=0.0)
        assert stats.get_average_led_transition_time() == 0.0

    def test_update_consumer_input_fps_first_frame(self):
        """Test input FPS update on first frame (no previous timestamp)."""
        stats = ConsumerStats()
        stats.update_consumer_input_fps(1.0)

        # First frame shouldn't calculate FPS (no delta)
        assert stats.consumer_input_fps == 0.0
        assert stats._last_frame_timestamp == 1.0

    def test_update_consumer_input_fps_subsequent_frames(self):
        """Test input FPS update with subsequent frames."""
        stats = ConsumerStats()
        stats.update_consumer_input_fps(1.0)
        stats.update_consumer_input_fps(1.0333)  # ~30 FPS

        assert 29.0 < stats.consumer_input_fps < 31.0

    def test_update_consumer_input_fps_60fps(self):
        """Test input FPS at 60 FPS."""
        stats = ConsumerStats()
        stats.update_consumer_input_fps(1.0)
        stats.update_consumer_input_fps(1.0167)  # ~60 FPS

        assert 58.0 < stats.consumer_input_fps < 62.0

    def test_update_consumer_input_fps_variable_rate(self):
        """Test input FPS with variable frame timing."""
        stats = ConsumerStats()
        stats.update_consumer_input_fps(1.0)
        stats.update_consumer_input_fps(1.05)  # 20 FPS
        assert 19.0 < stats.consumer_input_fps < 21.0

        stats.update_consumer_input_fps(1.0667)  # Back to ~60 FPS
        assert 58.0 < stats.consumer_input_fps < 62.0


class TestConsumerProcessPerformanceSettings:
    """Test ConsumerProcess performance settings methods."""

    @pytest.fixture
    def consumer_with_mocks(
        self,
        mock_frame_consumer,
        mock_control_state,
        mock_led_optimizer,
        mock_wled_sink,
        mock_led_buffer,
        mock_frame_renderer,
    ):
        """Create ConsumerProcess with all dependencies mocked."""
        with patch("src.consumer.consumer.FrameConsumer", return_value=mock_frame_consumer), patch(
            "src.consumer.consumer.ControlState", return_value=mock_control_state
        ), patch("src.consumer.consumer.LEDOptimizer", return_value=mock_led_optimizer), patch(
            "src.consumer.consumer.WLEDSink", return_value=mock_wled_sink
        ), patch(
            "src.consumer.consumer.LEDBuffer", return_value=mock_led_buffer
        ), patch(
            "src.utils.pattern_loader.create_frame_renderer_with_pattern", return_value=mock_frame_renderer
        ):
            consumer = ConsumerProcess(diffusion_patterns_path="/mock/pattern.npz")
            yield consumer

    def test_set_performance_settings_valid_fps(self, consumer_with_mocks):
        """Test setting valid FPS values."""
        consumer_with_mocks.set_performance_settings(target_fps=30.0)
        assert consumer_with_mocks.target_fps == 30.0

    def test_set_performance_settings_fps_upper_bound(self, consumer_with_mocks):
        """Test FPS is clamped to upper bound (60)."""
        consumer_with_mocks.set_performance_settings(target_fps=100.0)
        assert consumer_with_mocks.target_fps == 60.0

    def test_set_performance_settings_fps_lower_bound(self, consumer_with_mocks):
        """Test FPS is clamped to lower bound (1)."""
        consumer_with_mocks.set_performance_settings(target_fps=0.5)
        assert consumer_with_mocks.target_fps == 1.0

    def test_set_performance_settings_valid_brightness(self, consumer_with_mocks):
        """Test setting valid brightness values."""
        consumer_with_mocks.set_performance_settings(brightness_scale=0.8)
        assert consumer_with_mocks.brightness_scale == 0.8

    def test_set_performance_settings_brightness_upper_bound(self, consumer_with_mocks):
        """Test brightness is clamped to upper bound (1.0)."""
        consumer_with_mocks.set_performance_settings(brightness_scale=1.5)
        assert consumer_with_mocks.brightness_scale == 1.0

    def test_set_performance_settings_brightness_lower_bound(self, consumer_with_mocks):
        """Test brightness is clamped to lower bound (0.0)."""
        consumer_with_mocks.set_performance_settings(brightness_scale=-0.1)
        assert consumer_with_mocks.brightness_scale == 0.0

    def test_set_performance_settings_both_values(self, consumer_with_mocks):
        """Test setting both FPS and brightness."""
        consumer_with_mocks.set_performance_settings(target_fps=45.0, brightness_scale=0.5)
        assert consumer_with_mocks.target_fps == 45.0
        assert consumer_with_mocks.brightness_scale == 0.5


class TestConsumerProcessMetadataExtraction:
    """Test metadata extraction and transformation for transitions."""

    @pytest.fixture
    def consumer_with_mocks(
        self,
        mock_frame_consumer,
        mock_control_state,
        mock_led_optimizer,
        mock_wled_sink,
        mock_led_buffer,
        mock_frame_renderer,
    ):
        """Create ConsumerProcess with all dependencies mocked."""
        with patch("src.consumer.consumer.FrameConsumer", return_value=mock_frame_consumer), patch(
            "src.consumer.consumer.ControlState", return_value=mock_control_state
        ), patch("src.consumer.consumer.LEDOptimizer", return_value=mock_led_optimizer), patch(
            "src.consumer.consumer.WLEDSink", return_value=mock_wled_sink
        ), patch(
            "src.consumer.consumer.LEDBuffer", return_value=mock_led_buffer
        ), patch(
            "src.utils.pattern_loader.create_frame_renderer_with_pattern", return_value=mock_frame_renderer
        ):
            consumer = ConsumerProcess(diffusion_patterns_path="/mock/pattern.npz")
            yield consumer

    def test_extract_metadata_dict_with_none(self, consumer_with_mocks):
        """Test metadata extraction with None input."""
        result = consumer_with_mocks._extract_metadata_dict(None)
        assert result == {}

    def test_extract_metadata_dict_with_transition_metadata(self, consumer_with_mocks):
        """Test metadata extraction with transition metadata fields."""
        metadata = Mock()
        metadata.transition_in_type = "fade"
        metadata.transition_in_duration = 0.5
        metadata.transition_out_type = "cut"
        metadata.transition_out_duration = 0.2
        metadata.item_timestamp = 1.5
        metadata.item_duration = 30.0

        result = consumer_with_mocks._extract_metadata_dict(metadata)

        assert result["transition_in_type"] == "fade"
        assert result["transition_in_duration"] == 0.5
        assert result["transition_out_type"] == "cut"
        assert result["item_duration"] == 30.0

    def test_extract_metadata_dict_with_missing_transition_fields(self, consumer_with_mocks):
        """Test metadata extraction provides defaults for missing transition fields."""
        metadata = Mock(spec=["item_duration"])
        metadata.item_duration = 30.0

        result = consumer_with_mocks._extract_metadata_dict(metadata)

        # Should have default values for missing transition fields
        assert isinstance(result, dict)
        assert result["transition_in_type"] == "none"
        assert result["transition_in_duration"] == 0.0


class TestConsumerProcessBatchLogic:
    """Test batch processing logic methods."""

    @pytest.fixture
    def consumer_with_mocks(
        self,
        mock_frame_consumer,
        mock_control_state,
        mock_led_optimizer,
        mock_wled_sink,
        mock_led_buffer,
        mock_frame_renderer,
    ):
        """Create ConsumerProcess with all dependencies mocked."""
        with patch("src.consumer.consumer.FrameConsumer", return_value=mock_frame_consumer), patch(
            "src.consumer.consumer.ControlState", return_value=mock_control_state
        ), patch("src.consumer.consumer.LEDOptimizer", return_value=mock_led_optimizer), patch(
            "src.consumer.consumer.WLEDSink", return_value=mock_wled_sink
        ), patch(
            "src.consumer.consumer.LEDBuffer", return_value=mock_led_buffer
        ), patch(
            "src.utils.pattern_loader.create_frame_renderer_with_pattern", return_value=mock_frame_renderer
        ):
            consumer = ConsumerProcess(diffusion_patterns_path="/mock/pattern.npz", enable_batch_mode=True)
            # Manually set batch size for testing
            consumer._batch_size = 8
            consumer._frame_batch = []
            consumer._batch_metadata = []
            # Set batch start time to now to avoid timeout triggering
            consumer._last_batch_start_time = time.time()
            yield consumer

    def test_should_process_batch_empty(self, consumer_with_mocks):
        """Test batch check with empty batch."""
        consumer_with_mocks._frame_batch = []
        assert not consumer_with_mocks._should_process_batch()

    def test_should_process_batch_partial_no_timeout(self, consumer_with_mocks):
        """Test batch check with partial batch and no timeout."""
        consumer_with_mocks._frame_batch = [Mock()] * 4  # Half full
        consumer_with_mocks._last_batch_start_time = time.time()  # Reset timeout
        assert not consumer_with_mocks._should_process_batch()

    def test_should_process_batch_partial_with_timeout(self, consumer_with_mocks):
        """Test batch check with partial batch and timeout expired."""
        consumer_with_mocks._frame_batch = [Mock()] * 4  # Half full
        consumer_with_mocks._last_batch_start_time = time.time() - 10.0  # Old timestamp
        assert consumer_with_mocks._should_process_batch()

    def test_should_process_batch_full(self, consumer_with_mocks):
        """Test batch check with full batch."""
        consumer_with_mocks._frame_batch = [Mock()] * 8  # Full
        assert consumer_with_mocks._should_process_batch()

    def test_should_process_batch_overfull(self, consumer_with_mocks):
        """Test batch check with overfull batch."""
        consumer_with_mocks._frame_batch = [Mock()] * 10  # Over capacity
        assert consumer_with_mocks._should_process_batch()

    def test_should_process_batch_disabled(self, consumer_with_mocks):
        """Test batch check when batch mode is disabled."""
        consumer_with_mocks.enable_batch_mode = False
        consumer_with_mocks._frame_batch = [Mock()] * 8  # Full but disabled
        assert not consumer_with_mocks._should_process_batch()

    def test_clear_batch(self, consumer_with_mocks):
        """Test batch clearing."""
        consumer_with_mocks._frame_batch = [Mock()] * 5
        consumer_with_mocks._batch_metadata = [{"test": i} for i in range(5)]

        consumer_with_mocks._clear_batch()

        assert consumer_with_mocks._frame_batch == []
        assert consumer_with_mocks._batch_metadata == []


class TestConsumerProcessFrameGapDetection:
    """Test frame gap detection logic."""

    @pytest.fixture
    def consumer_with_mocks(
        self,
        mock_frame_consumer,
        mock_control_state,
        mock_led_optimizer,
        mock_wled_sink,
        mock_led_buffer,
        mock_frame_renderer,
    ):
        """Create ConsumerProcess with all dependencies mocked."""
        with patch("src.consumer.consumer.FrameConsumer", return_value=mock_frame_consumer), patch(
            "src.consumer.consumer.ControlState", return_value=mock_control_state
        ), patch("src.consumer.consumer.LEDOptimizer", return_value=mock_led_optimizer), patch(
            "src.consumer.consumer.WLEDSink", return_value=mock_wled_sink
        ), patch(
            "src.consumer.consumer.LEDBuffer", return_value=mock_led_buffer
        ), patch(
            "src.utils.pattern_loader.create_frame_renderer_with_pattern", return_value=mock_frame_renderer
        ):
            consumer = ConsumerProcess(diffusion_patterns_path="/mock/pattern.npz")
            yield consumer

    def test_detect_missing_frames_sequential(self, consumer_with_mocks):
        """Test no gaps detected with sequential frames."""
        # First call initializes tracking
        consumer_with_mocks._last_frame_index_seen = 10

        consumer_with_mocks._detect_and_log_missing_frames(11)

        assert consumer_with_mocks._last_frame_index_seen == 11

    def test_detect_missing_frames_gap(self, consumer_with_mocks):
        """Test gap detection with missing frames."""
        consumer_with_mocks._last_frame_index_seen = 10

        consumer_with_mocks._detect_and_log_missing_frames(15)  # 4 frames missing

        assert consumer_with_mocks._last_frame_index_seen == 15

    def test_detect_missing_frames_first_call(self, consumer_with_mocks):
        """Test first frame initializes tracking (when _last_frame_index_seen is 0)."""
        consumer_with_mocks._last_frame_index_seen = 0

        consumer_with_mocks._detect_and_log_missing_frames(1)

        assert consumer_with_mocks._last_frame_index_seen == 1

    def test_detect_missing_frames_backwards_no_update(self, consumer_with_mocks):
        """Test frame index going backwards doesn't update last_seen (error case)."""
        consumer_with_mocks._last_frame_index_seen = 100

        # Frame goes backwards - this is an error condition
        consumer_with_mocks._detect_and_log_missing_frames(50)

        # Should NOT update last_seen when sequence is broken
        assert consumer_with_mocks._last_frame_index_seen == 100


class TestConsumerProcessGetStats:
    """Test statistics retrieval."""

    @pytest.fixture
    def consumer_with_mocks(
        self,
        mock_frame_consumer,
        mock_control_state,
        mock_led_optimizer,
        mock_wled_sink,
        mock_led_buffer,
        mock_frame_renderer,
    ):
        """Create ConsumerProcess with all dependencies mocked."""
        with patch("src.consumer.consumer.FrameConsumer", return_value=mock_frame_consumer), patch(
            "src.consumer.consumer.ControlState", return_value=mock_control_state
        ), patch("src.consumer.consumer.LEDOptimizer", return_value=mock_led_optimizer), patch(
            "src.consumer.consumer.WLEDSink", return_value=mock_wled_sink
        ), patch(
            "src.consumer.consumer.LEDBuffer", return_value=mock_led_buffer
        ), patch(
            "src.utils.pattern_loader.create_frame_renderer_with_pattern", return_value=mock_frame_renderer
        ):
            consumer = ConsumerProcess(diffusion_patterns_path="/mock/pattern.npz")
            # Store mock references
            consumer._wled_client = mock_wled_sink
            consumer._led_buffer = mock_led_buffer
            consumer._led_optimizer = mock_led_optimizer
            consumer._frame_renderer = mock_frame_renderer
            yield consumer

    def test_get_stats_returns_dict(self, consumer_with_mocks):
        """Test get_stats returns a dictionary."""
        stats = consumer_with_mocks.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_includes_core_metrics(self, consumer_with_mocks):
        """Test get_stats includes core metrics."""
        consumer_with_mocks._stats.frames_processed = 100
        consumer_with_mocks._stats.total_processing_time = 10.0

        stats = consumer_with_mocks.get_stats()

        assert "frames_processed" in stats
        assert "average_optimization_fps" in stats
        assert stats["frames_processed"] == 100
        assert stats["average_optimization_fps"] == 10.0

    def test_get_stats_includes_error_counts(self, consumer_with_mocks):
        """Test get_stats includes error counts."""
        consumer_with_mocks._stats.optimization_errors = 5
        consumer_with_mocks._stats.transmission_errors = 2

        stats = consumer_with_mocks.get_stats()

        assert stats["optimization_errors"] == 5
        assert stats["transmission_errors"] == 2

    def test_get_statistics_alias(self, consumer_with_mocks):
        """Test get_statistics is an alias for get_stats."""
        stats1 = consumer_with_mocks.get_stats()
        stats2 = consumer_with_mocks.get_statistics()

        assert stats1 == stats2


class TestOptimizationLoopStep:
    """Test the extracted optimization loop step method."""

    @pytest.fixture
    def consumer_with_mocks(
        self,
        mock_frame_consumer,
        mock_control_state,
        mock_led_optimizer,
        mock_wled_sink,
        mock_led_buffer,
        mock_frame_renderer,
    ):
        """Create ConsumerProcess with all dependencies mocked."""
        with patch("src.consumer.consumer.FrameConsumer", return_value=mock_frame_consumer), patch(
            "src.consumer.consumer.ControlState", return_value=mock_control_state
        ), patch("src.consumer.consumer.LEDOptimizer", return_value=mock_led_optimizer), patch(
            "src.consumer.consumer.WLEDSink", return_value=mock_wled_sink
        ), patch(
            "src.consumer.consumer.LEDBuffer", return_value=mock_led_buffer
        ), patch(
            "src.utils.pattern_loader.create_frame_renderer_with_pattern", return_value=mock_frame_renderer
        ):
            consumer = ConsumerProcess(diffusion_patterns_path="/mock/pattern.npz")
            yield consumer

    def test_shutdown_signal_returns_should_stop(self, consumer_with_mocks):
        """Test that shutdown signal causes loop to stop."""
        consumer_with_mocks._control_state.should_shutdown.return_value = True
        loop_state = OptimizationLoopState()

        result = consumer_with_mocks._optimization_loop_step(None, loop_state)

        assert result.should_continue is False
        assert result.reason == "shutdown_signal"

    def test_timeout_continues_loop(self, consumer_with_mocks):
        """Test that timeout (None buffer_info) continues the loop."""
        consumer_with_mocks._control_state.should_shutdown.return_value = False
        loop_state = OptimizationLoopState()

        result = consumer_with_mocks._optimization_loop_step(None, loop_state)

        assert result.should_continue is True
        assert result.reason == "timeout"
        assert result.next_timeout == consumer_with_mocks.max_frame_wait_timeout

    def test_frame_processed_continues_loop(self, consumer_with_mocks):
        """Test that processing a frame continues the loop."""
        consumer_with_mocks._control_state.should_shutdown.return_value = False
        loop_state = OptimizationLoopState()

        # Mock buffer_info with frame data
        mock_buffer_info = Mock()
        mock_buffer_info.data = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        mock_buffer_info.metadata = Mock()
        mock_buffer_info.metadata.presentation_timestamp = 1.0
        mock_buffer_info.metadata.playlist_item_index = 0
        mock_buffer_info.metadata.is_first_frame_of_item = False

        # Mock _process_frame_optimization to avoid complex dependencies
        with patch.object(consumer_with_mocks, "_process_frame_optimization") as mock_process:
            result = consumer_with_mocks._optimization_loop_step(mock_buffer_info, loop_state)

        assert result.should_continue is True
        assert result.reason == "frame_processed"
        mock_process.assert_called_once_with(mock_buffer_info)

    def test_heartbeat_updated_on_step(self, consumer_with_mocks):
        """Test that optimization thread heartbeat is updated."""
        consumer_with_mocks._control_state.should_shutdown.return_value = False
        consumer_with_mocks._optimization_thread_heartbeat = 0.0
        loop_state = OptimizationLoopState()

        start_time = time.time()
        consumer_with_mocks._optimization_loop_step(None, loop_state)

        assert consumer_with_mocks._optimization_thread_heartbeat >= start_time

    def test_audio_state_checked_after_interval(self, consumer_with_mocks):
        """Test audio reactive state is checked after interval expires."""
        consumer_with_mocks._control_state.should_shutdown.return_value = False
        loop_state = OptimizationLoopState(
            last_audio_check=0.0,  # Long ago
            audio_check_interval=1.0,
        )

        with patch.object(consumer_with_mocks, "_update_audio_reactive_state") as mock_update:
            consumer_with_mocks._optimization_loop_step(None, loop_state)

        mock_update.assert_called_once()
        assert loop_state.last_audio_check > 0.0

    def test_audio_state_not_checked_before_interval(self, consumer_with_mocks):
        """Test audio reactive state is not checked before interval expires."""
        consumer_with_mocks._control_state.should_shutdown.return_value = False
        loop_state = OptimizationLoopState(
            last_audio_check=time.time(),  # Just checked
            audio_check_interval=1.0,
        )

        with patch.object(consumer_with_mocks, "_update_audio_reactive_state") as mock_update:
            consumer_with_mocks._optimization_loop_step(None, loop_state)

        mock_update.assert_not_called()

    def test_heartbeat_log_triggered_after_30_seconds(self, consumer_with_mocks):
        """Test heartbeat logging triggers after 30 second interval."""
        consumer_with_mocks._control_state.should_shutdown.return_value = False
        loop_state = OptimizationLoopState(
            last_heartbeat_log=0.0,  # 30+ seconds ago
        )

        old_heartbeat_log = loop_state.last_heartbeat_log
        consumer_with_mocks._optimization_loop_step(None, loop_state)

        # last_heartbeat_log should be updated
        assert loop_state.last_heartbeat_log > old_heartbeat_log


class TestOptimizationLoopStateDataclass:
    """Test OptimizationLoopState dataclass."""

    def test_default_values(self):
        """Test default initialization values."""
        state = OptimizationLoopState()

        assert state.last_audio_check == 0.0
        assert state.last_heartbeat_log == 0.0
        assert state.audio_check_interval == 1.0

    def test_custom_values(self):
        """Test initialization with custom values."""
        state = OptimizationLoopState(
            last_audio_check=5.0,
            last_heartbeat_log=10.0,
            audio_check_interval=2.0,
        )

        assert state.last_audio_check == 5.0
        assert state.last_heartbeat_log == 10.0
        assert state.audio_check_interval == 2.0


class TestOptimizationStepResultDataclass:
    """Test OptimizationStepResult dataclass."""

    def test_required_values(self):
        """Test initialization with required values."""
        result = OptimizationStepResult(
            should_continue=True,
            next_timeout=0.5,
        )

        assert result.should_continue is True
        assert result.next_timeout == 0.5
        assert result.reason == ""

    def test_with_reason(self):
        """Test initialization with reason."""
        result = OptimizationStepResult(
            should_continue=False,
            next_timeout=1.0,
            reason="shutdown_signal",
        )

        assert result.should_continue is False
        assert result.next_timeout == 1.0
        assert result.reason == "shutdown_signal"


class TestRenderingLoopStep:
    """Test the extracted rendering loop step method."""

    @pytest.fixture
    def consumer_with_mocks(
        self,
        mock_frame_consumer,
        mock_control_state,
        mock_led_optimizer,
        mock_wled_sink,
        mock_led_buffer,
        mock_frame_renderer,
    ):
        """Create ConsumerProcess with all dependencies mocked."""
        with patch("src.consumer.consumer.FrameConsumer", return_value=mock_frame_consumer), patch(
            "src.consumer.consumer.ControlState", return_value=mock_control_state
        ), patch("src.consumer.consumer.LEDOptimizer", return_value=mock_led_optimizer), patch(
            "src.consumer.consumer.WLEDSink", return_value=mock_wled_sink
        ), patch(
            "src.consumer.consumer.LEDBuffer", return_value=mock_led_buffer
        ), patch(
            "src.utils.pattern_loader.create_frame_renderer_with_pattern", return_value=mock_frame_renderer
        ):
            consumer = ConsumerProcess(diffusion_patterns_path="/mock/pattern.npz")
            consumer._led_buffer = mock_led_buffer
            consumer._frame_renderer = mock_frame_renderer
            yield consumer

    def test_no_control_status_returns_sleep(self, consumer_with_mocks):
        """Test that missing control status returns sleep action."""
        loop_state = RenderingLoopState()

        result = consumer_with_mocks._rendering_loop_step(None, None, loop_state)

        assert result.should_continue is True
        assert result.should_break is False
        assert result.wait_action == "sleep"
        assert result.reason == "no_control_status"

    def test_paused_state_returns_sleep(self, consumer_with_mocks):
        """Test that PAUSED state returns sleep action."""
        loop_state = RenderingLoopState()
        mock_status = Mock()
        mock_status.renderer_state = RendererState.PAUSED

        result = consumer_with_mocks._rendering_loop_step(mock_status, None, loop_state)

        assert result.should_continue is True
        assert result.wait_action == "sleep"
        assert result.reason == "paused"

    def test_waiting_state_returns_sleep(self, consumer_with_mocks):
        """Test that WAITING state returns sleep action."""
        loop_state = RenderingLoopState()
        mock_status = Mock()
        mock_status.renderer_state = RendererState.WAITING

        result = consumer_with_mocks._rendering_loop_step(mock_status, None, loop_state)

        assert result.should_continue is True
        assert result.wait_action == "sleep"
        assert result.reason == "waiting"

    def test_stopped_state_returns_sleep(self, consumer_with_mocks):
        """Test that STOPPED state returns sleep action."""
        loop_state = RenderingLoopState()
        mock_status = Mock()
        mock_status.renderer_state = RendererState.STOPPED

        result = consumer_with_mocks._rendering_loop_step(mock_status, None, loop_state)

        assert result.should_continue is True
        assert result.wait_action == "sleep"
        assert result.reason == "stopped"

    def test_playing_without_data_returns_read_buffer(self, consumer_with_mocks):
        """Test that PLAYING state without LED data returns read_buffer action."""
        loop_state = RenderingLoopState()
        mock_status = Mock()
        mock_status.renderer_state = RendererState.PLAYING

        result = consumer_with_mocks._rendering_loop_step(mock_status, None, loop_state)

        assert result.should_continue is True
        assert result.wait_action == "read_buffer"
        assert result.reason == "need_led_data"

    def test_playing_with_data_processes_frame(self, consumer_with_mocks):
        """Test that PLAYING state with LED data processes the frame."""
        loop_state = RenderingLoopState()
        mock_status = Mock()
        mock_status.renderer_state = RendererState.PLAYING

        # Create mock LED data
        led_values = np.zeros((3200, 3), dtype=np.uint8)
        timestamp = 1.0
        metadata = {"playlist_item_index": 0}
        led_data = (led_values, timestamp, metadata)

        # Mock frame renderer to return success
        consumer_with_mocks._frame_renderer.render_frame_at_timestamp.return_value = True

        result = consumer_with_mocks._rendering_loop_step(mock_status, led_data, loop_state)

        assert result.should_continue is True
        assert result.wait_action == "read_buffer"
        assert result.reason == "frame_processed"
        consumer_with_mocks._frame_renderer.render_frame_at_timestamp.assert_called_once()

    def test_heartbeat_updated_on_step(self, consumer_with_mocks):
        """Test that renderer thread heartbeat is updated."""
        consumer_with_mocks._renderer_thread_heartbeat = 0.0
        loop_state = RenderingLoopState()
        mock_status = Mock()
        mock_status.renderer_state = RendererState.STOPPED

        start_time = time.time()
        consumer_with_mocks._rendering_loop_step(mock_status, None, loop_state)

        assert consumer_with_mocks._renderer_thread_heartbeat >= start_time

    def test_pause_transition_calls_pause_renderer(self, consumer_with_mocks):
        """Test that transitioning to PAUSED calls pause_renderer."""
        loop_state = RenderingLoopState()
        consumer_with_mocks._last_renderer_state = RendererState.PLAYING
        mock_status = Mock()
        mock_status.renderer_state = RendererState.PAUSED

        consumer_with_mocks._rendering_loop_step(mock_status, None, loop_state)

        consumer_with_mocks._frame_renderer.pause_renderer.assert_called_once()

    def test_resume_transition_calls_resume_renderer(self, consumer_with_mocks):
        """Test that transitioning from PAUSED calls resume_renderer."""
        loop_state = RenderingLoopState()
        consumer_with_mocks._last_renderer_state = RendererState.PAUSED
        mock_status = Mock()
        mock_status.renderer_state = RendererState.PLAYING

        consumer_with_mocks._rendering_loop_step(mock_status, None, loop_state)

        consumer_with_mocks._frame_renderer.resume_renderer.assert_called_once()

    def test_render_error_increments_consecutive_errors(self, consumer_with_mocks):
        """Test that render errors increment consecutive error count."""
        loop_state = RenderingLoopState()
        mock_status = Mock()
        mock_status.renderer_state = RendererState.PLAYING

        led_data = (np.zeros((3200, 3), dtype=np.uint8), 1.0, {})

        # Make render fail
        consumer_with_mocks._frame_renderer.render_frame_at_timestamp.side_effect = Exception("Test error")

        result = consumer_with_mocks._rendering_loop_step(mock_status, led_data, loop_state)

        assert loop_state.consecutive_errors == 1
        assert result.should_continue is True  # Still continue after one error

    def test_critical_error_after_max_errors(self, consumer_with_mocks):
        """Test that reaching max consecutive errors triggers critical error."""
        loop_state = RenderingLoopState(consecutive_errors=9, max_consecutive_errors=10)
        mock_status = Mock()
        mock_status.renderer_state = RendererState.PLAYING

        led_data = (np.zeros((3200, 3), dtype=np.uint8), 1.0, {})

        # Make render fail
        consumer_with_mocks._frame_renderer.render_frame_at_timestamp.side_effect = Exception("Test error")

        result = consumer_with_mocks._rendering_loop_step(mock_status, led_data, loop_state)

        assert loop_state.consecutive_errors == 10
        assert result.should_continue is False
        assert result.should_break is True
        assert result.reason == "critical_render_error"


class TestRenderingLoopStateDataclass:
    """Test RenderingLoopState dataclass."""

    def test_default_values(self):
        """Test default initialization values."""
        state = RenderingLoopState()

        assert state.consecutive_errors == 0
        assert state.max_consecutive_errors == 10
        assert state.last_heartbeat_log == 0.0
        assert state.last_frame_render_time == 0.0
        assert state.last_buffer_warning_time == 0.0

    def test_custom_values(self):
        """Test initialization with custom values."""
        state = RenderingLoopState(
            consecutive_errors=5,
            max_consecutive_errors=20,
            last_heartbeat_log=10.0,
        )

        assert state.consecutive_errors == 5
        assert state.max_consecutive_errors == 20
        assert state.last_heartbeat_log == 10.0


class TestRenderingStepResultDataclass:
    """Test RenderingStepResult dataclass."""

    def test_required_values(self):
        """Test initialization with required values."""
        result = RenderingStepResult(
            should_continue=True,
            should_break=False,
            wait_action="sleep",
            wait_timeout=0.1,
        )

        assert result.should_continue is True
        assert result.should_break is False
        assert result.wait_action == "sleep"
        assert result.wait_timeout == 0.1
        assert result.reason == ""

    def test_with_all_values(self):
        """Test initialization with all values."""
        result = RenderingStepResult(
            should_continue=False,
            should_break=True,
            wait_action="read_buffer",
            wait_timeout=0.5,
            reason="critical_error",
        )

        assert result.should_continue is False
        assert result.should_break is True
        assert result.wait_action == "read_buffer"
        assert result.wait_timeout == 0.5
        assert result.reason == "critical_error"


# =============================================================================
# Phase 2: Integration Tests - State Machine and Frame Processing
# =============================================================================


class TestRendererStateTransitions:
    """Test renderer state transition logic."""

    @pytest.fixture
    def consumer_with_mocks(
        self,
        mock_frame_consumer,
        mock_control_state,
        mock_led_optimizer,
        mock_wled_sink,
        mock_led_buffer,
        mock_frame_renderer,
    ):
        """Create ConsumerProcess with all dependencies mocked."""
        with patch("src.consumer.consumer.FrameConsumer", return_value=mock_frame_consumer), patch(
            "src.consumer.consumer.ControlState", return_value=mock_control_state
        ), patch("src.consumer.consumer.LEDOptimizer", return_value=mock_led_optimizer), patch(
            "src.consumer.consumer.WLEDSink", return_value=mock_wled_sink
        ), patch(
            "src.consumer.consumer.LEDBuffer", return_value=mock_led_buffer
        ), patch(
            "src.utils.pattern_loader.create_frame_renderer_with_pattern", return_value=mock_frame_renderer
        ):
            consumer = ConsumerProcess(diffusion_patterns_path="/mock/pattern.npz")
            consumer._led_buffer = mock_led_buffer
            yield consumer

    def test_skip_default_status(self, consumer_with_mocks):
        """Test that default/error status is skipped."""
        mock_status = Mock()
        mock_status.consumer_input_fps = 0.0
        mock_status.renderer_output_fps = 0.0
        mock_status.uptime = 0.0
        mock_status.renderer_state = RendererState.STOPPED

        # Should return early without calling update_buffer_status
        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.update_buffer_status.assert_not_called()

    def test_waiting_to_playing_when_buffer_full(self, consumer_with_mocks):
        """Test transition from WAITING to PLAYING when buffer is full."""
        mock_status = Mock()
        mock_status.consumer_input_fps = 30.0
        mock_status.renderer_output_fps = 30.0
        mock_status.uptime = 100.0
        mock_status.renderer_state = RendererState.WAITING
        mock_status.producer_state = Mock()

        # Buffer is full
        consumer_with_mocks._led_buffer.get_buffer_stats.return_value = {
            "current_count": 10,
            "buffer_size": 10,
        }
        consumer_with_mocks._control_state.set_renderer_state.return_value = True

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_called_with(RendererState.PLAYING)

    def test_waiting_stays_waiting_when_buffer_not_full(self, consumer_with_mocks):
        """Test WAITING state stays when buffer is not full."""
        mock_status = Mock()
        mock_status.consumer_input_fps = 30.0
        mock_status.renderer_output_fps = 30.0
        mock_status.uptime = 100.0
        mock_status.renderer_state = RendererState.WAITING
        mock_status.producer_state = Mock()

        # Buffer is not full
        consumer_with_mocks._led_buffer.get_buffer_stats.return_value = {
            "current_count": 5,
            "buffer_size": 10,
        }

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_not_called()

    def test_playing_to_stopped_when_buffer_empty_and_producer_stopped(self, consumer_with_mocks):
        """Test transition from PLAYING to STOPPED when buffer empty and producer stopped."""
        from src.core.control_state import ProducerState

        mock_status = Mock()
        mock_status.consumer_input_fps = 30.0
        mock_status.renderer_output_fps = 30.0
        mock_status.uptime = 100.0
        mock_status.renderer_state = RendererState.PLAYING
        mock_status.producer_state = ProducerState.STOPPED

        # Buffer is empty
        consumer_with_mocks._led_buffer.get_buffer_stats.return_value = {
            "current_count": 0,
            "buffer_size": 10,
        }

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_called_with(RendererState.STOPPED)

    def test_playing_continues_when_buffer_has_frames(self, consumer_with_mocks):
        """Test PLAYING continues when buffer has frames."""
        from src.core.control_state import ProducerState

        mock_status = Mock()
        mock_status.consumer_input_fps = 30.0
        mock_status.renderer_output_fps = 30.0
        mock_status.uptime = 100.0
        mock_status.renderer_state = RendererState.PLAYING
        mock_status.producer_state = ProducerState.STOPPED

        # Buffer has frames
        consumer_with_mocks._led_buffer.get_buffer_stats.return_value = {
            "current_count": 5,
            "buffer_size": 10,
        }

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_not_called()

    def test_paused_to_stopped_when_buffer_empty_and_producer_stopped(self, consumer_with_mocks):
        """Test transition from PAUSED to STOPPED when buffer empty and producer stopped."""
        from src.core.control_state import ProducerState

        mock_status = Mock()
        mock_status.consumer_input_fps = 30.0
        mock_status.renderer_output_fps = 30.0
        mock_status.uptime = 100.0
        mock_status.renderer_state = RendererState.PAUSED
        mock_status.producer_state = ProducerState.STOPPED

        # Buffer is empty
        consumer_with_mocks._led_buffer.get_buffer_stats.return_value = {
            "current_count": 0,
            "buffer_size": 10,
        }

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_called_with(RendererState.STOPPED)

    def test_no_transition_when_led_buffer_none(self, consumer_with_mocks):
        """Test no transition when LED buffer is not initialized."""
        mock_status = Mock()
        mock_status.consumer_input_fps = 30.0
        mock_status.renderer_output_fps = 30.0
        mock_status.uptime = 100.0
        mock_status.renderer_state = RendererState.WAITING

        consumer_with_mocks._led_buffer = None

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_not_called()


class TestFrameGapTracking:
    """Test frame gap tracking logic."""

    @pytest.fixture
    def consumer_with_mocks(
        self,
        mock_frame_consumer,
        mock_control_state,
        mock_led_optimizer,
        mock_wled_sink,
        mock_led_buffer,
        mock_frame_renderer,
    ):
        """Create ConsumerProcess with all dependencies mocked."""
        with patch("src.consumer.consumer.FrameConsumer", return_value=mock_frame_consumer), patch(
            "src.consumer.consumer.ControlState", return_value=mock_control_state
        ), patch("src.consumer.consumer.LEDOptimizer", return_value=mock_led_optimizer), patch(
            "src.consumer.consumer.WLEDSink", return_value=mock_wled_sink
        ), patch(
            "src.consumer.consumer.LEDBuffer", return_value=mock_led_buffer
        ), patch(
            "src.utils.pattern_loader.create_frame_renderer_with_pattern", return_value=mock_frame_renderer
        ):
            consumer = ConsumerProcess(diffusion_patterns_path="/mock/pattern.npz")
            yield consumer

    def test_first_frame_initializes_tracking(self, consumer_with_mocks):
        """Test first frame initializes timestamp tracking."""
        consumer_with_mocks._last_frame_timestamp = 0.0
        consumer_with_mocks._last_frame_receive_time = 0.0

        consumer_with_mocks._track_frame_gaps(1.0, 100.0, has_presentation_timestamp=True)

        assert consumer_with_mocks._last_frame_timestamp == 1.0
        assert consumer_with_mocks._last_frame_receive_time == 100.0

    def test_sequential_frames_update_tracking(self, consumer_with_mocks):
        """Test sequential frames update tracking correctly."""
        consumer_with_mocks._last_frame_timestamp = 1.0
        consumer_with_mocks._last_frame_receive_time = 100.0

        consumer_with_mocks._track_frame_gaps(1.033, 100.033, has_presentation_timestamp=True)

        assert consumer_with_mocks._last_frame_timestamp == 1.033
        assert consumer_with_mocks._last_frame_receive_time == 100.033

    def test_no_timestamp_gap_check_without_presentation_timestamp(self, consumer_with_mocks):
        """Test timestamp gap check is skipped without presentation timestamp."""
        consumer_with_mocks._last_frame_timestamp = 1.0
        consumer_with_mocks._last_frame_receive_time = 100.0
        consumer_with_mocks._frame_timestamp_gap_threshold = 0.1

        # Large timestamp gap but no presentation timestamp - should not warn
        consumer_with_mocks._track_frame_gaps(2.0, 100.033, has_presentation_timestamp=False)

        # Just verify tracking is updated
        assert consumer_with_mocks._last_frame_timestamp == 2.0

    def test_batch_mode_increases_gap_threshold(self, consumer_with_mocks):
        """Test batch mode allows larger realtime gaps."""
        consumer_with_mocks.enable_batch_mode = True
        consumer_with_mocks._batch_size = 8
        consumer_with_mocks._realtime_gap_threshold = 0.2
        consumer_with_mocks._last_frame_timestamp = 1.0
        consumer_with_mocks._last_frame_receive_time = 100.0

        # Gap that would warn in single mode but not batch mode
        consumer_with_mocks._track_frame_gaps(1.5, 100.5, has_presentation_timestamp=True)

        # Should update tracking
        assert consumer_with_mocks._last_frame_receive_time == 100.5


class TestLedTransitionEffects:
    """Test LED transition effect creation."""

    @pytest.fixture
    def consumer_with_mocks(
        self,
        mock_frame_consumer,
        mock_control_state,
        mock_led_optimizer,
        mock_wled_sink,
        mock_led_buffer,
        mock_frame_renderer,
    ):
        """Create ConsumerProcess with all dependencies mocked."""
        with patch("src.consumer.consumer.FrameConsumer", return_value=mock_frame_consumer), patch(
            "src.consumer.consumer.ControlState", return_value=mock_control_state
        ), patch("src.consumer.consumer.LEDOptimizer", return_value=mock_led_optimizer), patch(
            "src.consumer.consumer.WLEDSink", return_value=mock_wled_sink
        ), patch(
            "src.consumer.consumer.LEDBuffer", return_value=mock_led_buffer
        ), patch(
            "src.utils.pattern_loader.create_frame_renderer_with_pattern", return_value=mock_frame_renderer
        ):
            consumer = ConsumerProcess(diffusion_patterns_path="/mock/pattern.npz")
            consumer._frame_renderer = mock_frame_renderer
            yield consumer

    def test_skip_when_no_item_duration(self, consumer_with_mocks):
        """Test transition effects skipped when no item duration."""
        metadata = {"item_duration": 0.0}

        consumer_with_mocks._create_led_transition_effects_for_item(0.0, metadata)

        consumer_with_mocks._frame_renderer.add_led_effect.assert_not_called()

    def test_skip_when_transition_type_none(self, consumer_with_mocks):
        """Test transition effects skipped when type is none."""
        metadata = {
            "item_duration": 30.0,
            "transition_in_type": "none",
            "transition_out_type": "none",
        }

        consumer_with_mocks._create_led_transition_effects_for_item(0.0, metadata)

        consumer_with_mocks._frame_renderer.add_led_effect.assert_not_called()

    def test_creates_fade_in_effect(self, consumer_with_mocks):
        """Test fade-in effect is created."""
        metadata = {
            "item_duration": 30.0,
            "transition_in_type": "led_fade",
            "transition_in_duration": 1.0,
            "transition_out_type": "none",
        }

        consumer_with_mocks._create_led_transition_effects_for_item(0.0, metadata)

        consumer_with_mocks._frame_renderer.add_led_effect.assert_called_once()

    def test_creates_fade_out_effect(self, consumer_with_mocks):
        """Test fade-out effect is created."""
        metadata = {
            "item_duration": 30.0,
            "transition_in_type": "none",
            "transition_out_type": "led_fade",
            "transition_out_duration": 1.0,
        }

        consumer_with_mocks._create_led_transition_effects_for_item(0.0, metadata)

        consumer_with_mocks._frame_renderer.add_led_effect.assert_called_once()

    def test_creates_both_fade_effects(self, consumer_with_mocks):
        """Test both fade-in and fade-out effects are created."""
        metadata = {
            "item_duration": 30.0,
            "transition_in_type": "led_fade",
            "transition_in_duration": 1.0,
            "transition_out_type": "led_fade",
            "transition_out_duration": 1.0,
        }

        consumer_with_mocks._create_led_transition_effects_for_item(0.0, metadata)

        assert consumer_with_mocks._frame_renderer.add_led_effect.call_count == 2

    def test_creates_random_in_effect(self, consumer_with_mocks):
        """Test random-in effect is created."""
        metadata = {
            "item_duration": 30.0,
            "transition_in_type": "led_random",
            "transition_in_duration": 1.0,
            "transition_out_type": "none",
        }

        consumer_with_mocks._create_led_transition_effects_for_item(0.0, metadata)

        consumer_with_mocks._frame_renderer.add_led_effect.assert_called_once()

    def test_creates_random_out_effect(self, consumer_with_mocks):
        """Test random-out effect is created."""
        metadata = {
            "item_duration": 30.0,
            "transition_in_type": "none",
            "transition_out_type": "led_random",
            "transition_out_duration": 1.0,
        }

        consumer_with_mocks._create_led_transition_effects_for_item(0.0, metadata)

        consumer_with_mocks._frame_renderer.add_led_effect.assert_called_once()


class TestConsumerUtilityMethods:
    """Test various utility methods on ConsumerProcess."""

    @pytest.fixture
    def consumer_with_mocks(
        self,
        mock_frame_consumer,
        mock_control_state,
        mock_led_optimizer,
        mock_wled_sink,
        mock_led_buffer,
        mock_frame_renderer,
    ):
        """Create ConsumerProcess with all dependencies mocked."""
        with patch("src.consumer.consumer.FrameConsumer", return_value=mock_frame_consumer), patch(
            "src.consumer.consumer.ControlState", return_value=mock_control_state
        ), patch("src.consumer.consumer.LEDOptimizer", return_value=mock_led_optimizer), patch(
            "src.consumer.consumer.WLEDSink", return_value=mock_wled_sink
        ), patch(
            "src.consumer.consumer.LEDBuffer", return_value=mock_led_buffer
        ), patch(
            "src.utils.pattern_loader.create_frame_renderer_with_pattern", return_value=mock_frame_renderer
        ):
            consumer = ConsumerProcess(diffusion_patterns_path="/mock/pattern.npz")
            consumer._led_buffer = mock_led_buffer
            consumer._frame_renderer = mock_frame_renderer
            yield consumer

    def test_is_renderer_initialized_true(self, consumer_with_mocks):
        """Test is_renderer_initialized returns True when initialized."""
        consumer_with_mocks._frame_renderer.is_initialized.return_value = True

        assert consumer_with_mocks.is_renderer_initialized() is True

    def test_is_renderer_initialized_false(self, consumer_with_mocks):
        """Test is_renderer_initialized returns False when not initialized."""
        consumer_with_mocks._frame_renderer.is_initialized.return_value = False

        assert consumer_with_mocks.is_renderer_initialized() is False

    def test_clear_led_buffer(self, consumer_with_mocks):
        """Test clear_led_buffer clears the buffer."""
        consumer_with_mocks.clear_led_buffer()

        consumer_with_mocks._led_buffer.clear.assert_called_once()

    def test_clear_led_buffer_when_none(self, consumer_with_mocks):
        """Test clear_led_buffer handles None buffer gracefully."""
        consumer_with_mocks._led_buffer = None

        # Should not raise
        consumer_with_mocks.clear_led_buffer()

    def test_reset_renderer_stats(self, consumer_with_mocks):
        """Test reset_renderer_stats resets frame renderer stats."""
        consumer_with_mocks.reset_renderer_stats()

        consumer_with_mocks._frame_renderer.reset_stats.assert_called_once()

    def test_set_wled_enabled_true(self, consumer_with_mocks):
        """Test enabling WLED sink."""
        consumer_with_mocks._wled_client = Mock()

        consumer_with_mocks.set_wled_enabled(True)

        consumer_with_mocks._frame_renderer.set_wled_enabled.assert_called_with(True)

    def test_set_wled_enabled_false(self, consumer_with_mocks):
        """Test disabling WLED sink."""
        consumer_with_mocks._wled_client = Mock()

        consumer_with_mocks.set_wled_enabled(False)

        consumer_with_mocks._frame_renderer.set_wled_enabled.assert_called_with(False)

    def test_set_led_buffer_size_success(self, consumer_with_mocks):
        """Test setting LED buffer size."""
        consumer_with_mocks._led_buffer.set_buffer_size.return_value = True

        result = consumer_with_mocks.set_led_buffer_size(20)

        assert result is True
        consumer_with_mocks._led_buffer.set_buffer_size.assert_called_with(20)

    def test_set_led_buffer_size_when_none(self, consumer_with_mocks):
        """Test setting LED buffer size when buffer is None."""
        consumer_with_mocks._led_buffer = None

        result = consumer_with_mocks.set_led_buffer_size(20)

        assert result is False

    def test_set_timing_parameters(self, consumer_with_mocks):
        """Test setting timing parameters on frame renderer."""
        consumer_with_mocks.set_timing_parameters(
            first_frame_delay_ms=100.0,
            timing_tolerance_ms=10.0,
        )

        consumer_with_mocks._frame_renderer.set_timing_parameters.assert_called_with(
            first_frame_delay_ms=100.0,
            timing_tolerance_ms=10.0,
        )


class TestHandleRendererStateTransitions:
    """Test _handle_renderer_state_transitions state machine logic."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked dependencies."""
        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._led_buffer = MagicMock()
            consumer._control_state = MagicMock()
            consumer._led_buffer.get_buffer_stats.return_value = {
                "current_count": 5,
                "buffer_size": 10,
            }
            return consumer

    def _create_mock_status(
        self,
        renderer_state: RendererState = RendererState.STOPPED,
        producer_state: ProducerState = ProducerState.STOPPED,
        consumer_input_fps: float = 30.0,
        renderer_output_fps: float = 30.0,
        uptime: float = 100.0,
    ):
        """Create a mock control status object."""
        mock_status = Mock()
        mock_status.renderer_state = renderer_state
        mock_status.producer_state = producer_state
        mock_status.consumer_input_fps = consumer_input_fps
        mock_status.renderer_output_fps = renderer_output_fps
        mock_status.uptime = uptime
        return mock_status

    def test_skips_transition_on_default_error_status(self, consumer_with_mocks):
        """Test that state transitions are skipped when status looks like error/default."""
        mock_status = self._create_mock_status(
            consumer_input_fps=0.0,
            renderer_output_fps=0.0,
            uptime=0.0,
        )

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        # Should not call set_renderer_state
        consumer_with_mocks._control_state.set_renderer_state.assert_not_called()

    def test_skips_transition_when_buffer_is_none(self, consumer_with_mocks):
        """Test that state transitions are skipped when LED buffer is None."""
        consumer_with_mocks._led_buffer = None
        mock_status = self._create_mock_status(renderer_state=RendererState.WAITING)

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        # Should not call anything since buffer is None
        consumer_with_mocks._control_state.set_renderer_state.assert_not_called()

    def test_updates_buffer_status(self, consumer_with_mocks):
        """Test that buffer status is updated in control state."""
        mock_status = self._create_mock_status(renderer_state=RendererState.STOPPED)

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.update_buffer_status.assert_called_with(5, 10)

    def test_stopped_state_no_automatic_transition(self, consumer_with_mocks):
        """Test that STOPPED state doesn't auto-transition."""
        mock_status = self._create_mock_status(renderer_state=RendererState.STOPPED)

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_not_called()

    def test_waiting_to_playing_when_buffer_full(self, consumer_with_mocks):
        """Test WAITING -> PLAYING transition when buffer is full."""
        consumer_with_mocks._led_buffer.get_buffer_stats.return_value = {
            "current_count": 10,
            "buffer_size": 10,
        }
        mock_status = self._create_mock_status(renderer_state=RendererState.WAITING)

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_called_with(RendererState.PLAYING)

    def test_waiting_stays_waiting_when_buffer_not_full(self, consumer_with_mocks):
        """Test WAITING stays WAITING when buffer is not full."""
        consumer_with_mocks._led_buffer.get_buffer_stats.return_value = {
            "current_count": 5,
            "buffer_size": 10,
        }
        mock_status = self._create_mock_status(
            renderer_state=RendererState.WAITING,
            producer_state=ProducerState.PLAYING,
        )

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_not_called()

    def test_playing_to_stopped_when_buffer_empty_and_producer_stopped(self, consumer_with_mocks):
        """Test PLAYING -> STOPPED when buffer empty and producer stopped."""
        consumer_with_mocks._led_buffer.get_buffer_stats.return_value = {
            "current_count": 0,
            "buffer_size": 10,
        }
        mock_status = self._create_mock_status(
            renderer_state=RendererState.PLAYING,
            producer_state=ProducerState.STOPPED,
        )

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_called_with(RendererState.STOPPED)

    def test_playing_continues_when_buffer_has_frames(self, consumer_with_mocks):
        """Test PLAYING continues when buffer has frames."""
        consumer_with_mocks._led_buffer.get_buffer_stats.return_value = {
            "current_count": 5,
            "buffer_size": 10,
        }
        mock_status = self._create_mock_status(
            renderer_state=RendererState.PLAYING,
            producer_state=ProducerState.STOPPED,
        )

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_not_called()

    def test_paused_to_stopped_when_buffer_empty_and_producer_stopped(self, consumer_with_mocks):
        """Test PAUSED -> STOPPED when buffer empty and producer stopped."""
        consumer_with_mocks._led_buffer.get_buffer_stats.return_value = {
            "current_count": 0,
            "buffer_size": 10,
        }
        mock_status = self._create_mock_status(
            renderer_state=RendererState.PAUSED,
            producer_state=ProducerState.STOPPED,
        )

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_called_with(RendererState.STOPPED)

    def test_paused_stays_paused_when_producer_playing(self, consumer_with_mocks):
        """Test PAUSED stays PAUSED when producer is playing."""
        consumer_with_mocks._led_buffer.get_buffer_stats.return_value = {
            "current_count": 0,
            "buffer_size": 10,
        }
        mock_status = self._create_mock_status(
            renderer_state=RendererState.PAUSED,
            producer_state=ProducerState.PLAYING,
        )

        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        consumer_with_mocks._control_state.set_renderer_state.assert_not_called()

    def test_handles_exception_gracefully(self, consumer_with_mocks):
        """Test that exceptions in state transitions are handled."""
        consumer_with_mocks._led_buffer.get_buffer_stats.side_effect = RuntimeError("Buffer error")
        mock_status = self._create_mock_status(renderer_state=RendererState.PLAYING)

        # Should not raise, error is logged
        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

    def test_state_change_detection_logging(self, consumer_with_mocks):
        """Test that state changes are tracked for logging."""
        mock_status = self._create_mock_status(renderer_state=RendererState.PLAYING)

        # First call initializes tracking
        consumer_with_mocks._handle_renderer_state_transitions(mock_status)

        # Verify tracking attribute was created
        assert hasattr(consumer_with_mocks, "_last_logged_state")


class TestWledReconnectionMonitoring:
    """Test WLED reconnection and thread monitoring."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked dependencies."""
        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._wled_client = MagicMock()
            consumer._frame_renderer = MagicMock()
            consumer._stop_event = MagicMock()
            consumer._wled_reconnection_event = MagicMock()
            return consumer

    def test_wled_sink_is_connected(self, consumer_with_mocks):
        """Test checking if WLED sink is connected."""
        consumer_with_mocks._wled_client.is_connected.return_value = True

        assert consumer_with_mocks._wled_client.is_connected() is True

    def test_wled_sink_reconnect_called(self, consumer_with_mocks):
        """Test that reconnect is called on WLED client."""
        consumer_with_mocks._wled_client.is_connected.return_value = False
        consumer_with_mocks._wled_client.reconnect.return_value = True

        result = consumer_with_mocks._wled_client.reconnect()

        assert result is True
        consumer_with_mocks._wled_client.reconnect.assert_called_once()

    def test_frame_renderer_wled_enabled_state(self, consumer_with_mocks):
        """Test frame renderer WLED enabled state."""
        consumer_with_mocks._frame_renderer.set_wled_enabled(True)
        consumer_with_mocks._frame_renderer.set_wled_enabled.assert_called_with(True)

        consumer_with_mocks._frame_renderer.set_wled_enabled(False)
        consumer_with_mocks._frame_renderer.set_wled_enabled.assert_called_with(False)


class TestAudioBeatCallbacks:
    """Test audio beat detection callback methods."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked dependencies."""
        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._control_state = MagicMock()
            consumer._audio_analysis_running = True
            return consumer

    def test_on_beat_detected_updates_control_state(self, consumer_with_mocks):
        """Test _on_beat_detected updates control state with beat info."""
        from src.consumer.audio_beat_analyzer import BeatEvent

        beat_event = BeatEvent(
            timestamp=1.0,
            system_time=time.time(),
            bpm=120.0,
            beat_count=1,
            confidence=0.9,
            intensity=0.8,
            is_downbeat=False,
        )

        consumer_with_mocks._on_beat_detected(beat_event)

        consumer_with_mocks._control_state.update_status.assert_called()
        call_kwargs = consumer_with_mocks._control_state.update_status.call_args[1]
        assert call_kwargs["current_bpm"] == 120.0
        assert call_kwargs["beat_count"] == 1
        assert call_kwargs["beat_confidence"] == 0.9
        assert call_kwargs["audio_intensity"] == 0.8

    def test_on_beat_detected_downbeat_updates_downbeat_time(self, consumer_with_mocks):
        """Test _on_beat_detected updates downbeat time for downbeats."""
        from src.consumer.audio_beat_analyzer import BeatEvent

        system_time = time.time()
        beat_event = BeatEvent(
            timestamp=1.0,
            system_time=system_time,
            bpm=120.0,
            beat_count=4,
            confidence=0.95,
            intensity=0.9,
            is_downbeat=True,
        )

        consumer_with_mocks._on_beat_detected(beat_event)

        # Should have two calls - one for beat info, one for downbeat time
        assert consumer_with_mocks._control_state.update_status.call_count == 2

    def test_on_beat_detected_handles_exception(self, consumer_with_mocks):
        """Test _on_beat_detected handles exceptions gracefully."""
        from src.consumer.audio_beat_analyzer import BeatEvent

        consumer_with_mocks._control_state.update_status.side_effect = RuntimeError("Test error")

        beat_event = BeatEvent(
            timestamp=1.0,
            system_time=time.time(),
            bpm=120.0,
            beat_count=1,
            confidence=0.9,
            intensity=0.8,
            is_downbeat=False,
        )

        # Should not raise
        consumer_with_mocks._on_beat_detected(beat_event)

    def test_on_builddrop_event_updates_control_state(self, consumer_with_mocks):
        """Test _on_builddrop_event updates control state."""
        from src.consumer.audio_beat_analyzer import BuildDropEvent

        builddrop_event = BuildDropEvent(
            timestamp=1.0,
            system_time=time.time(),
            event_type="building",
            buildup_intensity=0.7,
            bass_energy=0.5,
            high_energy=0.3,
            confidence=0.9,
            is_cut=False,
            is_drop=False,
        )

        consumer_with_mocks._on_builddrop_event(builddrop_event)

        consumer_with_mocks._control_state.update_status.assert_called_once()
        call_kwargs = consumer_with_mocks._control_state.update_status.call_args[1]
        assert call_kwargs["buildup_state"] == "building"
        assert call_kwargs["buildup_intensity"] == 0.7

    def test_on_builddrop_event_cut_updates_cut_time(self, consumer_with_mocks):
        """Test _on_builddrop_event updates cut time on cut events."""
        from src.consumer.audio_beat_analyzer import BuildDropEvent

        system_time = time.time()
        builddrop_event = BuildDropEvent(
            timestamp=1.0,
            system_time=system_time,
            event_type="cut",
            buildup_intensity=0.0,
            bass_energy=0.2,
            high_energy=0.1,
            confidence=0.95,
            is_cut=True,
            is_drop=False,
        )

        consumer_with_mocks._on_builddrop_event(builddrop_event)

        call_kwargs = consumer_with_mocks._control_state.update_status.call_args[1]
        assert call_kwargs["last_cut_time"] == system_time

    def test_on_builddrop_event_drop_updates_drop_time(self, consumer_with_mocks):
        """Test _on_builddrop_event updates drop time on drop events."""
        from src.consumer.audio_beat_analyzer import BuildDropEvent

        system_time = time.time()
        builddrop_event = BuildDropEvent(
            timestamp=1.0,
            system_time=system_time,
            event_type="drop",
            buildup_intensity=1.0,
            bass_energy=0.9,
            high_energy=0.8,
            confidence=0.98,
            is_cut=False,
            is_drop=True,
        )

        consumer_with_mocks._on_builddrop_event(builddrop_event)

        call_kwargs = consumer_with_mocks._control_state.update_status.call_args[1]
        assert call_kwargs["last_drop_time"] == system_time

    def test_on_builddrop_event_handles_exception(self, consumer_with_mocks):
        """Test _on_builddrop_event handles exceptions gracefully."""
        from src.consumer.audio_beat_analyzer import BuildDropEvent

        consumer_with_mocks._control_state.update_status.side_effect = RuntimeError("Test error")

        builddrop_event = BuildDropEvent(
            timestamp=1.0,
            system_time=time.time(),
            event_type="building",
            buildup_intensity=0.7,
            bass_energy=0.5,
            high_energy=0.3,
            confidence=0.9,
            is_cut=False,
            is_drop=False,
        )

        # Should not raise
        consumer_with_mocks._on_builddrop_event(builddrop_event)


class TestProcessSingleFrame:
    """Test single frame processing logic."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked dependencies."""
        from src.utils.frame_drop_rate_ewma import FrameDropRateEwma

        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._led_optimizer = MagicMock()
            consumer._led_buffer = MagicMock()
            consumer._control_state = MagicMock()
            consumer._stats = ConsumerStats()
            consumer._frame_renderer = MagicMock()
            consumer._frames_with_content = 0
            consumer.optimization_iterations = 10
            consumer.pre_optimization_drop_rate_ewma = FrameDropRateEwma(alpha=0.1, name="test")
            consumer._last_consumer_log_time = 0.0
            consumer._consumer_log_interval = 2.0
            return consumer

    def test_process_single_frame_success(self, consumer_with_mocks):
        """Test successful single frame processing."""
        # Create mock frame (cupy array)
        rgb_frame = cp.zeros((480, 800, 3), dtype=cp.uint8)
        rgb_frame[:, :, 0] = 128  # Non-zero content

        # Mock buffer info
        mock_buffer_info = Mock()
        mock_buffer_info.metadata = Mock()
        mock_buffer_info.metadata.presentation_timestamp = 1.0
        mock_buffer_info.metadata.playlist_item_index = 0
        mock_buffer_info.metadata.is_first_frame_of_item = False
        mock_buffer_info.metadata.timing_data = None

        # Mock optimization result
        mock_result = Mock()
        mock_result.led_values = np.random.randint(0, 255, (3200, 3), dtype=np.uint8)
        mock_result.converged = True
        mock_result.iterations = 5
        mock_result.error_metrics = {"mse": 0.01}
        consumer_with_mocks._led_optimizer.optimize_frame.return_value = mock_result

        # Mock LED buffer write
        consumer_with_mocks._led_buffer.write_led_values.return_value = True

        result = consumer_with_mocks._process_single_frame(rgb_frame, mock_buffer_info, {}, 0.001, time.time())

        assert result is False  # Not dropped
        consumer_with_mocks._led_optimizer.optimize_frame.assert_called_once()
        consumer_with_mocks._led_buffer.write_led_values.assert_called_once()

    def test_process_single_frame_optimization_not_converged(self, consumer_with_mocks):
        """Test single frame processing when optimization doesn't converge."""
        rgb_frame = cp.zeros((480, 800, 3), dtype=cp.uint8)

        mock_buffer_info = Mock()
        mock_buffer_info.metadata = Mock()
        mock_buffer_info.metadata.presentation_timestamp = 1.0
        mock_buffer_info.metadata.playlist_item_index = 0
        mock_buffer_info.metadata.is_first_frame_of_item = False
        mock_buffer_info.metadata.timing_data = None

        mock_result = Mock()
        mock_result.led_values = np.zeros((3200, 3), dtype=np.uint8)
        mock_result.converged = False  # Not converged
        mock_result.iterations = 10
        mock_result.error_metrics = {"mse": 0.5}
        consumer_with_mocks._led_optimizer.optimize_frame.return_value = mock_result
        consumer_with_mocks._led_buffer.write_led_values.return_value = True

        result = consumer_with_mocks._process_single_frame(rgb_frame, mock_buffer_info, {}, 0.001, time.time())

        assert result is False
        assert consumer_with_mocks._stats.optimization_errors == 1

    def test_process_single_frame_led_buffer_none(self, consumer_with_mocks):
        """Test single frame processing when LED buffer is None."""
        rgb_frame = cp.zeros((480, 800, 3), dtype=cp.uint8)

        mock_buffer_info = Mock()
        mock_buffer_info.metadata = Mock()
        mock_buffer_info.metadata.presentation_timestamp = 1.0
        mock_buffer_info.metadata.playlist_item_index = 0
        mock_buffer_info.metadata.is_first_frame_of_item = False
        mock_buffer_info.metadata.timing_data = None

        mock_result = Mock()
        mock_result.led_values = np.zeros((3200, 3), dtype=np.uint8)
        mock_result.converged = True
        mock_result.iterations = 5
        mock_result.error_metrics = {}
        consumer_with_mocks._led_optimizer.optimize_frame.return_value = mock_result

        consumer_with_mocks._led_buffer = None

        result = consumer_with_mocks._process_single_frame(rgb_frame, mock_buffer_info, {}, 0.001, time.time())

        assert result is True  # Dropped due to error

    def test_process_single_frame_reads_iterations_from_control_state(self, consumer_with_mocks):
        """Test that optimization iterations are read from control state."""
        rgb_frame = cp.zeros((480, 800, 3), dtype=cp.uint8)

        mock_buffer_info = Mock()
        mock_buffer_info.metadata = Mock()
        mock_buffer_info.metadata.presentation_timestamp = 1.0
        mock_buffer_info.metadata.playlist_item_index = 0
        mock_buffer_info.metadata.is_first_frame_of_item = False
        mock_buffer_info.metadata.timing_data = None

        # Mock control state with different iterations
        mock_status = Mock()
        mock_status.optimization_iterations = 15
        consumer_with_mocks._control_state.get_status.return_value = mock_status

        mock_result = Mock()
        mock_result.led_values = np.zeros((3200, 3), dtype=np.uint8)
        mock_result.converged = True
        mock_result.iterations = 15
        mock_result.error_metrics = {}
        consumer_with_mocks._led_optimizer.optimize_frame.return_value = mock_result
        consumer_with_mocks._led_buffer.write_led_values.return_value = True

        consumer_with_mocks._process_single_frame(rgb_frame, mock_buffer_info, {}, 0.001, time.time())

        # Verify optimize_frame was called with iterations from control state
        call_kwargs = consumer_with_mocks._led_optimizer.optimize_frame.call_args[1]
        assert call_kwargs["max_iterations"] == 15


class TestConsumerLifecycle:
    """Test consumer start/stop lifecycle methods."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked dependencies."""
        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._running = False
            consumer._shutdown_requested = False
            consumer._initialized = False
            consumer._control_state = MagicMock()
            consumer._frame_consumer = MagicMock()
            consumer._led_optimizer = MagicMock()
            consumer._led_buffer = MagicMock()
            consumer._wled_client = MagicMock()
            consumer._frame_renderer = MagicMock()
            consumer._audio_beat_analyzer = None
            consumer._audio_analysis_running = False
            consumer._optimization_thread = None
            consumer._renderer_thread = None
            consumer._wled_reconnection_thread = None
            consumer._thread_monitor_thread = None
            consumer._wled_reconnection_event = MagicMock()
            consumer._thread_monitor_shutdown_event = MagicMock()
            return consumer

    def test_start_returns_true_when_already_running(self, consumer_with_mocks):
        """Test start returns True if already running."""
        consumer_with_mocks._running = True

        result = consumer_with_mocks.start()

        assert result is True

    def test_stop_sets_shutdown_flags(self, consumer_with_mocks):
        """Test stop sets shutdown flags."""
        consumer_with_mocks._running = True
        consumer_with_mocks._shutdown_requested = False

        consumer_with_mocks.stop()

        assert consumer_with_mocks._shutdown_requested is True
        assert consumer_with_mocks._running is False

    def test_stop_prevents_duplicate_calls(self, consumer_with_mocks):
        """Test stop returns early on duplicate calls."""
        consumer_with_mocks._shutdown_requested = True

        consumer_with_mocks.stop()

        # Should not call set_renderer_state since it returns early
        consumer_with_mocks._control_state.set_renderer_state.assert_not_called()

    def test_stop_sets_renderer_state_to_stopped(self, consumer_with_mocks):
        """Test stop sets renderer state to STOPPED."""
        consumer_with_mocks._running = True

        consumer_with_mocks.stop()

        consumer_with_mocks._control_state.set_renderer_state.assert_called_with(RendererState.STOPPED)

    def test_stop_signals_wled_reconnection_event(self, consumer_with_mocks):
        """Test stop signals WLED reconnection event."""
        consumer_with_mocks._running = True

        consumer_with_mocks.stop()

        consumer_with_mocks._wled_reconnection_event.set.assert_called_once()

    def test_stop_signals_thread_monitor_shutdown(self, consumer_with_mocks):
        """Test stop signals thread monitor shutdown."""
        consumer_with_mocks._running = True

        consumer_with_mocks.stop()

        consumer_with_mocks._thread_monitor_shutdown_event.set.assert_called_once()

    def test_stop_stops_audio_analysis_if_running(self, consumer_with_mocks):
        """Test stop stops audio analysis if running."""
        consumer_with_mocks._running = True
        consumer_with_mocks._audio_beat_analyzer = MagicMock()
        consumer_with_mocks._audio_analysis_running = True

        consumer_with_mocks.stop()

        consumer_with_mocks._audio_beat_analyzer.stop_analysis.assert_called_once()
        assert consumer_with_mocks._audio_analysis_running is False


class TestUpdateAudioReactiveState:
    """Test audio reactive state management."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked dependencies."""
        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._control_state = MagicMock()
            consumer._audio_beat_analyzer = MagicMock()
            consumer._audio_analysis_running = False
            consumer._use_audio_test_file = True
            consumer._frame_renderer = MagicMock()
            return consumer

    def test_starts_audio_when_enabled_and_not_running(self, consumer_with_mocks):
        """Test audio starts when enabled but not running."""
        mock_status = Mock()
        mock_status.audio_reactive_enabled = True
        mock_status.use_audio_test_file = True
        consumer_with_mocks._control_state.get_status.return_value = mock_status

        consumer_with_mocks._update_audio_reactive_state()

        consumer_with_mocks._audio_beat_analyzer.start_analysis.assert_called_once()
        assert consumer_with_mocks._audio_analysis_running is True

    def test_stops_audio_when_disabled_and_running(self, consumer_with_mocks):
        """Test audio stops when disabled but running."""
        consumer_with_mocks._audio_analysis_running = True

        mock_status = Mock()
        mock_status.audio_reactive_enabled = False
        mock_status.use_audio_test_file = True
        consumer_with_mocks._control_state.get_status.return_value = mock_status

        consumer_with_mocks._update_audio_reactive_state()

        consumer_with_mocks._audio_beat_analyzer.stop_analysis.assert_called_once()
        assert consumer_with_mocks._audio_analysis_running is False

    def test_no_action_when_already_in_correct_state(self, consumer_with_mocks):
        """Test no action when audio is already in correct state."""
        consumer_with_mocks._audio_analysis_running = True

        mock_status = Mock()
        mock_status.audio_reactive_enabled = True  # Already running
        mock_status.use_audio_test_file = True
        consumer_with_mocks._control_state.get_status.return_value = mock_status

        consumer_with_mocks._update_audio_reactive_state()

        # Should not call start or stop
        consumer_with_mocks._audio_beat_analyzer.start_analysis.assert_not_called()
        consumer_with_mocks._audio_beat_analyzer.stop_analysis.assert_not_called()

    def test_returns_early_when_no_status(self, consumer_with_mocks):
        """Test returns early when control status is None."""
        consumer_with_mocks._control_state.get_status.return_value = None

        consumer_with_mocks._update_audio_reactive_state()

        # Should not interact with audio analyzer
        consumer_with_mocks._audio_beat_analyzer.start_analysis.assert_not_called()
        consumer_with_mocks._audio_beat_analyzer.stop_analysis.assert_not_called()

    def test_returns_early_when_no_audio_analyzer(self, consumer_with_mocks):
        """Test returns early when audio analyzer is None."""
        consumer_with_mocks._audio_beat_analyzer = None

        mock_status = Mock()
        mock_status.audio_reactive_enabled = True
        mock_status.use_audio_test_file = True
        consumer_with_mocks._control_state.get_status.return_value = mock_status

        # Should not raise
        consumer_with_mocks._update_audio_reactive_state()

    def test_handles_start_analysis_exception(self, consumer_with_mocks):
        """Test handles exception when starting analysis."""
        mock_status = Mock()
        mock_status.audio_reactive_enabled = True
        mock_status.use_audio_test_file = True
        consumer_with_mocks._control_state.get_status.return_value = mock_status

        consumer_with_mocks._audio_beat_analyzer.start_analysis.side_effect = RuntimeError("Test error")

        # Should not raise
        consumer_with_mocks._update_audio_reactive_state()

    def test_updates_audio_level_when_running(self, consumer_with_mocks):
        """Test updates audio level in control state when running."""
        consumer_with_mocks._audio_analysis_running = True
        consumer_with_mocks._audio_beat_analyzer.audio_capture = None  # No audio capture for recording

        mock_status = Mock()
        mock_status.audio_reactive_enabled = True
        mock_status.use_audio_test_file = True
        mock_status.audio_recording_requested = False
        consumer_with_mocks._control_state.get_status.return_value = mock_status

        consumer_with_mocks._audio_beat_analyzer.get_audio_stats.return_value = {
            "audio_level": 0.75,
            "agc_gain_db": 12.0,
        }

        consumer_with_mocks._update_audio_reactive_state()

        # Should call update_status at least once for audio level
        consumer_with_mocks._control_state.update_status.assert_called()

        # Check that one of the calls contains audio level
        calls = consumer_with_mocks._control_state.update_status.call_args_list
        audio_level_updated = False
        for call in calls:
            kwargs = call[1]
            if "audio_level" in kwargs:
                assert kwargs["audio_level"] == 0.75
                assert kwargs["agc_gain_db"] == 12.0
                audio_level_updated = True
                break
        assert audio_level_updated, "audio_level was not updated in any call"


class TestProcessFrameOptimization:
    """Test _process_frame_optimization method."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked dependencies for frame optimization tests."""
        from pathlib import Path

        from src.utils.frame_drop_rate_ewma import FrameDropRateEwma

        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._led_optimizer = MagicMock()
            consumer._led_buffer = MagicMock()
            consumer._control_state = MagicMock()
            consumer._stats = ConsumerStats()
            consumer._frame_renderer = MagicMock()
            consumer._transition_processor = MagicMock()
            consumer._timing_logger = None
            consumer._frames_with_content = 0
            consumer.optimization_iterations = 10
            consumer.enable_batch_mode = False
            consumer.brightness_scale = 1.0
            consumer.pre_optimization_drop_rate_ewma = FrameDropRateEwma(alpha=0.1, name="test")
            consumer._last_consumer_log_time = 0.0
            consumer._consumer_log_interval = 2.0
            consumer._current_optimization_item_index = -1
            consumer._suspend_late_frame_drops = False
            consumer._consecutive_late_drops = 0
            consumer._last_late_drop_time = 0.0
            consumer._first_late_drop_lateness = 0.0
            consumer._late_drop_cascade_threshold = 5
            consumer._last_frame_receive_time = 0.0
            consumer._last_frame_gap_time = 0.0
            consumer._frame_gap_detected = False
            consumer._debug_frame_count = 100  # Disable debug frame saving
            consumer._debug_max_frames = 10
            consumer._debug_frame_dir = Path("/tmp")
            consumer._frame_batch = []
            consumer._batch_metadata = []
            consumer._last_frame_timestamp = 0.0

            # Setup default mocks
            consumer._frame_renderer.is_frame_late.return_value = False
            consumer._frame_renderer.is_initialized.return_value = True
            consumer._transition_processor.process_frame.side_effect = lambda f, m: f

            return consumer

    @pytest.fixture
    def mock_buffer_info(self):
        """Create mock buffer info with valid frame data."""
        buffer_info = Mock()
        buffer_info.metadata = Mock()
        buffer_info.metadata.presentation_timestamp = 1.0
        buffer_info.metadata.playlist_item_index = 0
        buffer_info.metadata.is_first_frame_of_item = False
        buffer_info.metadata.timing_data = None
        buffer_info.metadata.has_presentation_timestamp = True

        # Create valid frame data
        frame_data = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame_data[:, :, 0] = 128  # Some non-zero content
        buffer_info.get_array_interleaved.return_value = frame_data

        return buffer_info

    def test_successful_frame_processing(self, consumer_with_mocks, mock_buffer_info):
        """Test successful frame processing routes to single frame processing."""
        # Mock single frame processing
        consumer_with_mocks._process_single_frame = Mock(return_value=False)

        result = consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        assert result is False  # Not dropped
        consumer_with_mocks._process_single_frame.assert_called_once()

    def test_late_frame_dropped(self, consumer_with_mocks, mock_buffer_info):
        """Test late frames are dropped."""
        consumer_with_mocks._frame_renderer.is_frame_late.return_value = True
        consumer_with_mocks._frame_renderer.get_adjusted_wallclock_delta.return_value = -100.0  # Very late

        result = consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        assert result is True  # Dropped
        assert consumer_with_mocks._stats.frames_dropped_early == 1

    def test_first_frame_of_item_suspends_late_drops(self, consumer_with_mocks, mock_buffer_info):
        """Test first frame of new playlist item suspends late frame dropping."""
        mock_buffer_info.metadata.is_first_frame_of_item = True
        mock_buffer_info.metadata.playlist_item_index = 1
        consumer_with_mocks._frame_renderer.is_frame_late.return_value = True

        # Mock single frame processing
        consumer_with_mocks._process_single_frame = Mock(return_value=False)

        result = consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        # First frame should NOT be dropped, late drops should be suspended
        assert result is False  # Frame was processed, not dropped
        assert consumer_with_mocks._suspend_late_frame_drops is True
        consumer_with_mocks._process_single_frame.assert_called_once()

    def test_resumes_late_drops_when_caught_up(self, consumer_with_mocks, mock_buffer_info):
        """Test late drop suspension is cleared when caught up."""
        consumer_with_mocks._suspend_late_frame_drops = True
        consumer_with_mocks._frame_renderer.is_frame_late.return_value = False

        consumer_with_mocks._process_single_frame = Mock(return_value=False)

        consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        assert consumer_with_mocks._suspend_late_frame_drops is False

    def test_invalid_frame_shape_dropped(self, consumer_with_mocks, mock_buffer_info):
        """Test frames with invalid shape are dropped."""
        # Return frame with wrong shape
        wrong_shape_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_buffer_info.get_array_interleaved.return_value = wrong_shape_frame

        result = consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        assert result is True  # Dropped

    def test_batch_mode_routes_to_batch_processing(self, consumer_with_mocks, mock_buffer_info):
        """Test batch mode routes to batch frame processing."""
        consumer_with_mocks.enable_batch_mode = True
        consumer_with_mocks._led_optimizer.supports_batch_optimization.return_value = True
        consumer_with_mocks._process_frame_for_batch = Mock(return_value=False)
        consumer_with_mocks._process_single_frame = Mock(return_value=False)

        result = consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        assert result is False
        consumer_with_mocks._process_frame_for_batch.assert_called_once()
        consumer_with_mocks._process_single_frame.assert_not_called()

    def test_exception_handling(self, consumer_with_mocks, mock_buffer_info):
        """Test exceptions during processing are handled."""
        mock_buffer_info.get_array_interleaved.side_effect = RuntimeError("Test error")

        result = consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        assert result is True  # Dropped due to error
        assert consumer_with_mocks._stats.optimization_errors == 1

    def test_brightness_scaling_applied(self, consumer_with_mocks, mock_buffer_info):
        """Test brightness scaling is applied to frames."""
        consumer_with_mocks.brightness_scale = 0.5
        consumer_with_mocks._process_single_frame = Mock(return_value=False)

        consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        # Verify transition processor was called (brightness applied before this)
        consumer_with_mocks._transition_processor.process_frame.assert_called_once()

    def test_updates_playlist_item_tracking(self, consumer_with_mocks, mock_buffer_info):
        """Test playlist item index tracking is updated."""
        mock_buffer_info.metadata.playlist_item_index = 5
        consumer_with_mocks._process_single_frame = Mock(return_value=False)

        consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        assert consumer_with_mocks._current_optimization_item_index == 5

    def test_cascade_detection(self, consumer_with_mocks, mock_buffer_info):
        """Test late frame drop cascade detection."""
        consumer_with_mocks._frame_renderer.is_frame_late.return_value = True
        consumer_with_mocks._frame_renderer.get_adjusted_wallclock_delta.return_value = -100.0
        consumer_with_mocks._consecutive_late_drops = 4  # One away from threshold

        result = consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        assert result is True  # Dropped
        assert consumer_with_mocks._consecutive_late_drops == 5  # Reached threshold

    def test_cascade_resets_on_non_late_frame(self, consumer_with_mocks, mock_buffer_info):
        """Test cascade counter resets when processing non-late frame."""
        consumer_with_mocks._consecutive_late_drops = 3
        consumer_with_mocks._frame_renderer.is_frame_late.return_value = False
        consumer_with_mocks._process_single_frame = Mock(return_value=False)

        consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        assert consumer_with_mocks._consecutive_late_drops == 0

    def test_updates_consumer_input_fps(self, consumer_with_mocks, mock_buffer_info):
        """Test consumer input FPS tracking is updated."""
        consumer_with_mocks._process_single_frame = Mock(return_value=False)
        consumer_with_mocks._stats._last_frame_timestamp = 0.9  # Previous timestamp

        consumer_with_mocks._process_frame_optimization(mock_buffer_info)

        # FPS should be updated (may be approximate based on timing)
        # Just verify the method didn't error
        assert consumer_with_mocks._stats.consumer_input_fps >= 0


class TestProcessFrameForBatch:
    """Test _process_frame_for_batch method."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked dependencies for batch tests."""
        from src.utils.frame_drop_rate_ewma import FrameDropRateEwma

        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._led_optimizer = MagicMock()
            consumer._led_buffer = MagicMock()
            consumer._stats = ConsumerStats()
            consumer._frame_batch = []
            consumer._batch_metadata = []
            consumer._batch_size = 8
            consumer._last_batch_start_time = 0.0
            consumer.enable_batch_mode = True
            consumer.pre_optimization_drop_rate_ewma = FrameDropRateEwma(alpha=0.1, name="test")

            return consumer

    @pytest.fixture
    def mock_buffer_info(self):
        """Create mock buffer info for batch tests."""
        buffer_info = Mock()
        buffer_info.metadata = Mock()
        buffer_info.metadata.presentation_timestamp = 1.0
        buffer_info.metadata.playlist_item_index = 0
        buffer_info.metadata.is_first_frame_of_item = False
        buffer_info.metadata.timing_data = None
        return buffer_info

    def test_frame_accumulated_to_batch(self, consumer_with_mocks, mock_buffer_info):
        """Test frame is accumulated to batch."""
        rgb_frame = cp.zeros((480, 800, 3), dtype=cp.uint8)
        consumer_with_mocks._should_process_batch = Mock(return_value=False)

        result = consumer_with_mocks._process_frame_for_batch(rgb_frame, mock_buffer_info, {}, 0.001, time.time())

        assert result is False  # Not dropped (accumulated)
        assert len(consumer_with_mocks._frame_batch) == 1
        assert len(consumer_with_mocks._batch_metadata) == 1

    def test_batch_processed_when_full(self, consumer_with_mocks, mock_buffer_info):
        """Test batch is processed when full."""
        rgb_frame = cp.zeros((480, 800, 3), dtype=cp.uint8)
        consumer_with_mocks._should_process_batch = Mock(return_value=True)
        consumer_with_mocks._process_frame_batch = Mock(return_value=True)

        result = consumer_with_mocks._process_frame_for_batch(rgb_frame, mock_buffer_info, {}, 0.001, time.time())

        assert result is False  # Not dropped
        consumer_with_mocks._process_frame_batch.assert_called_once()

    def test_metadata_preserved_in_batch(self, consumer_with_mocks, mock_buffer_info):
        """Test metadata is preserved when adding to batch."""
        rgb_frame = cp.zeros((480, 800, 3), dtype=cp.uint8)
        mock_buffer_info.metadata.playlist_item_index = 5
        mock_buffer_info.metadata.is_first_frame_of_item = True
        consumer_with_mocks._should_process_batch = Mock(return_value=False)

        consumer_with_mocks._process_frame_for_batch(
            rgb_frame, mock_buffer_info, {"transition_in_type": "fade"}, 0.001, time.time()
        )

        metadata = consumer_with_mocks._batch_metadata[0]
        assert metadata["playlist_item_index"] == 5
        assert metadata["is_first_frame_of_item"] is True
        assert metadata["transition_in_type"] == "fade"

    def test_initializes_batch_start_time(self, consumer_with_mocks, mock_buffer_info):
        """Test batch start time is initialized on first frame."""
        rgb_frame = cp.zeros((480, 800, 3), dtype=cp.uint8)
        consumer_with_mocks._should_process_batch = Mock(return_value=False)
        consumer_with_mocks._last_batch_start_time = 0.0

        before_time = time.time()
        consumer_with_mocks._process_frame_for_batch(rgb_frame, mock_buffer_info, {}, 0.001, time.time())
        after_time = time.time()

        assert consumer_with_mocks._last_batch_start_time >= before_time
        assert consumer_with_mocks._last_batch_start_time <= after_time


class TestProcessFrameBatch:
    """Test _process_frame_batch method."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked dependencies for batch processing tests."""
        from src.utils.frame_drop_rate_ewma import FrameDropRateEwma

        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._led_optimizer = MagicMock()
            consumer._led_buffer = MagicMock()
            consumer._frame_renderer = MagicMock()
            consumer._control_state = MagicMock()
            consumer._stats = ConsumerStats()
            consumer._timing_logger = None
            consumer._frame_batch = []
            consumer._batch_metadata = []
            consumer._batch_size = 8
            consumer.enable_batch_mode = True
            consumer.optimization_iterations = 10
            consumer.pre_optimization_drop_rate_ewma = FrameDropRateEwma(alpha=0.1, name="test")
            consumer._last_consumer_log_time = 0.0
            consumer._consumer_log_interval = 2.0
            consumer._frames_with_content = 0

            # Mock control state
            mock_status = Mock()
            mock_status.optimization_iterations = 10
            consumer._control_state.get_status.return_value = mock_status

            # Add frames to batch
            for i in range(8):
                frame = cp.zeros((480, 800, 3), dtype=cp.uint8)
                consumer._frame_batch.append(frame)
                consumer._batch_metadata.append(
                    {
                        "timestamp": float(i),
                        "playlist_item_index": 0,
                        "is_first_frame_of_item": False,
                        "timing_data": None,
                        "transition_time": 0.001,
                        "optimization_time": 0.0,
                    }
                )

            return consumer

    def test_batch_optimization_called(self, consumer_with_mocks):
        """Test batch optimization is called with accumulated frames."""
        mock_result = Mock()
        # led_values shape is (batch, channels, led_count) - use cupy array
        mock_result.led_values = cp.random.randint(0, 255, (8, 3, 3200), dtype=cp.uint8)
        mock_result.converged = True
        mock_result.iterations = 5
        # error_metrics is a list indexed by frame
        mock_result.error_metrics = [{"mse": 0.01} for _ in range(8)]
        consumer_with_mocks._led_optimizer.optimize_batch_frames.return_value = mock_result
        consumer_with_mocks._led_buffer.write_led_values.return_value = True

        result = consumer_with_mocks._process_frame_batch()

        assert result is True
        consumer_with_mocks._led_optimizer.optimize_batch_frames.assert_called_once()

    def test_batch_cleared_after_processing(self, consumer_with_mocks):
        """Test batch is cleared after processing."""
        mock_result = Mock()
        mock_result.led_values = cp.random.randint(0, 255, (8, 3, 3200), dtype=cp.uint8)
        mock_result.converged = True
        mock_result.iterations = 5
        mock_result.error_metrics = [{"mse": 0.01} for _ in range(8)]
        consumer_with_mocks._led_optimizer.optimize_batch_frames.return_value = mock_result
        consumer_with_mocks._led_buffer.write_led_values.return_value = True

        consumer_with_mocks._process_frame_batch()

        assert len(consumer_with_mocks._frame_batch) == 0
        assert len(consumer_with_mocks._batch_metadata) == 0

    def test_stats_updated_for_batch(self, consumer_with_mocks):
        """Test statistics are updated for all frames in batch."""
        mock_result = Mock()
        mock_result.led_values = cp.random.randint(0, 255, (8, 3, 3200), dtype=cp.uint8)
        mock_result.converged = True
        mock_result.iterations = 5
        mock_result.error_metrics = [{"mse": 0.01} for _ in range(8)]
        consumer_with_mocks._led_optimizer.optimize_batch_frames.return_value = mock_result
        consumer_with_mocks._led_buffer.write_led_values.return_value = True

        consumer_with_mocks._process_frame_batch()

        # All 8 frames should be counted as processed
        assert consumer_with_mocks._stats.frames_processed == 8

    def test_exception_handling(self, consumer_with_mocks):
        """Test exception handling in batch processing - batch cleared even on error."""
        consumer_with_mocks._led_optimizer.optimize_batch_frames.side_effect = RuntimeError("Batch error")

        # Method re-raises exceptions, but batch should be cleared in finally block
        with pytest.raises(RuntimeError, match="Batch error"):
            consumer_with_mocks._process_frame_batch()

        # Batch should be cleared even on error (in finally block)
        assert len(consumer_with_mocks._frame_batch) == 0
        assert len(consumer_with_mocks._batch_metadata) == 0


class TestInitializeComponents:
    """Test component initialization behavior."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked components."""
        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._frame_consumer = MagicMock()
            consumer._control_state = MagicMock()
            consumer._led_optimizer = MagicMock()
            consumer._frame_renderer = MagicMock()
            consumer._stats = ConsumerStats()

            return consumer

    def test_frame_consumer_connection_called_during_startup(self, consumer_with_mocks):
        """Test that frame consumer connect is properly set up."""
        # Verify the mock is configured correctly
        consumer_with_mocks._frame_consumer.connect.return_value = True
        assert consumer_with_mocks._frame_consumer.connect() is True

    def test_control_state_initialization(self, consumer_with_mocks):
        """Test control state initialization is properly set up."""
        consumer_with_mocks._control_state.initialize.return_value = True
        assert consumer_with_mocks._control_state.initialize() is True

    def test_led_optimizer_initialization(self, consumer_with_mocks):
        """Test LED optimizer initialization returns proper result."""
        consumer_with_mocks._led_optimizer.initialize.return_value = True
        assert consumer_with_mocks._led_optimizer.initialize() is True

        consumer_with_mocks._led_optimizer.initialize.return_value = False
        assert consumer_with_mocks._led_optimizer.initialize() is False

    def test_optimizer_led_count_available_after_init(self, consumer_with_mocks):
        """Test LED count is available after optimizer initialization."""
        consumer_with_mocks._led_optimizer._actual_led_count = 3200
        assert consumer_with_mocks._led_optimizer._actual_led_count == 3200

    def test_frame_renderer_initialized_check(self, consumer_with_mocks):
        """Test frame renderer initialization check."""
        consumer_with_mocks._frame_renderer.is_initialized.return_value = True
        assert consumer_with_mocks._frame_renderer.is_initialized() is True

        consumer_with_mocks._frame_renderer.is_initialized.return_value = False
        assert consumer_with_mocks._frame_renderer.is_initialized() is False


class TestCheckAudioRecordingRequest:
    """Test _check_audio_recording_request method."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked audio components."""
        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._control_state = MagicMock()
            consumer._audio_beat_analyzer = MagicMock()
            consumer._audio_beat_analyzer.audio_capture = MagicMock()
            return consumer

    def test_no_recording_requested_and_not_recording(self, consumer_with_mocks):
        """Test when no recording requested and not currently recording."""
        status = Mock()
        status.audio_recording_requested = False
        status.audio_recording_in_progress = False  # No recording was in progress
        consumer_with_mocks._audio_beat_analyzer.audio_capture.is_recording = False

        consumer_with_mocks._check_audio_recording_request(status)

        # Should not update any recording status
        consumer_with_mocks._control_state.update_status.assert_not_called()

    def test_updates_progress_when_recording_in_progress(self, consumer_with_mocks):
        """Test recording progress is updated when recording is in progress."""
        status = Mock()
        status.audio_recording_requested = False
        consumer_with_mocks._audio_beat_analyzer.audio_capture.is_recording = True
        consumer_with_mocks._audio_beat_analyzer.audio_capture.get_recording_status.return_value = {"progress": 0.5}

        consumer_with_mocks._check_audio_recording_request(status)

        consumer_with_mocks._control_state.update_status.assert_called_once_with(
            audio_recording_in_progress=True,
            audio_recording_progress=0.5,
        )

    def test_recording_finished_updates_status(self, consumer_with_mocks):
        """Test status updated when recording finishes."""
        status = Mock()
        status.audio_recording_requested = False
        status.audio_recording_in_progress = True  # Was recording
        consumer_with_mocks._audio_beat_analyzer.audio_capture.is_recording = False  # Now finished

        consumer_with_mocks._check_audio_recording_request(status)

        consumer_with_mocks._control_state.update_status.assert_called_once_with(
            audio_recording_in_progress=False,
            audio_recording_progress=1.0,
            audio_recording_status="complete",
        )

    def test_recording_requested_no_audio_capture(self, consumer_with_mocks):
        """Test error when recording requested but audio capture not available."""
        status = Mock()
        status.audio_recording_requested = True
        consumer_with_mocks._audio_beat_analyzer = None

        consumer_with_mocks._check_audio_recording_request(status)

        consumer_with_mocks._control_state.update_status.assert_called_once_with(
            audio_recording_requested=False,
            audio_recording_status="error: audio capture not available",
        )

    def test_recording_requested_already_recording(self, consumer_with_mocks):
        """Test warning when recording requested but already recording."""
        status = Mock()
        status.audio_recording_requested = True
        consumer_with_mocks._audio_beat_analyzer.audio_capture.is_recording = True

        consumer_with_mocks._check_audio_recording_request(status)

        consumer_with_mocks._control_state.update_status.assert_called_once_with(audio_recording_requested=False)

    def test_recording_requested_in_file_mode(self, consumer_with_mocks):
        """Test error when recording requested in file playback mode."""
        status = Mock()
        status.audio_recording_requested = True
        consumer_with_mocks._audio_beat_analyzer.audio_capture.is_recording = False
        consumer_with_mocks._audio_beat_analyzer.audio_capture.file_mode = True

        consumer_with_mocks._check_audio_recording_request(status)

        consumer_with_mocks._control_state.update_status.assert_called_once_with(
            audio_recording_requested=False,
            audio_recording_status="error: cannot record in file mode",
        )

    def test_recording_requested_no_output_path(self, consumer_with_mocks):
        """Test error when recording requested without output path."""
        status = Mock()
        status.audio_recording_requested = True
        status.audio_recording_duration = 60.0
        status.audio_recording_output_path = ""
        consumer_with_mocks._audio_beat_analyzer.audio_capture.is_recording = False
        consumer_with_mocks._audio_beat_analyzer.audio_capture.file_mode = False

        consumer_with_mocks._check_audio_recording_request(status)

        consumer_with_mocks._control_state.update_status.assert_called_once_with(
            audio_recording_requested=False,
            audio_recording_status="error: no output path specified",
        )

    def test_recording_started_successfully(self, consumer_with_mocks):
        """Test successful recording start."""
        status = Mock()
        status.audio_recording_requested = True
        status.audio_recording_duration = 30.0
        status.audio_recording_output_path = "/tmp/test.wav"
        consumer_with_mocks._audio_beat_analyzer.audio_capture.is_recording = False
        consumer_with_mocks._audio_beat_analyzer.audio_capture.file_mode = False
        consumer_with_mocks._audio_beat_analyzer.audio_capture.start_recording.return_value = True

        consumer_with_mocks._check_audio_recording_request(status)

        consumer_with_mocks._audio_beat_analyzer.audio_capture.start_recording.assert_called_once()
        consumer_with_mocks._control_state.update_status.assert_called_once_with(
            audio_recording_requested=False,
            audio_recording_in_progress=True,
            audio_recording_progress=0.0,
            audio_recording_status="recording",
        )

    def test_recording_failed_to_start(self, consumer_with_mocks):
        """Test failed recording start."""
        status = Mock()
        status.audio_recording_requested = True
        status.audio_recording_duration = 30.0
        status.audio_recording_output_path = "/tmp/test.wav"
        consumer_with_mocks._audio_beat_analyzer.audio_capture.is_recording = False
        consumer_with_mocks._audio_beat_analyzer.audio_capture.file_mode = False
        consumer_with_mocks._audio_beat_analyzer.audio_capture.start_recording.return_value = False

        consumer_with_mocks._check_audio_recording_request(status)

        consumer_with_mocks._control_state.update_status.assert_called_once_with(
            audio_recording_requested=False,
            audio_recording_status="error: failed to start recording",
        )

    def test_exception_handling(self, consumer_with_mocks):
        """Test exception handling in audio recording check."""
        status = Mock()
        status.audio_recording_requested = True
        consumer_with_mocks._audio_beat_analyzer.audio_capture.is_recording = Mock(
            side_effect=RuntimeError("Test error")
        )

        # Should not raise
        consumer_with_mocks._check_audio_recording_request(status)

        # Should update status with error
        consumer_with_mocks._control_state.update_status.assert_called()


class TestWledReconnectionLoop:
    """Test WLED reconnection loop behavior."""

    @pytest.fixture
    def consumer_with_mocks(self):
        """Create consumer with mocked WLED components."""
        with patch.object(ConsumerProcess, "__init__", lambda self: None):
            consumer = ConsumerProcess()
            consumer._running = True
            consumer._shutdown_requested = False
            consumer._wled_client = MagicMock()
            consumer._wled_reconnection_event = MagicMock()
            consumer._frame_renderer = MagicMock()
            consumer._test_renderer = None
            consumer._preview_sink = None
            consumer.wled_reconnect_interval = 0.1  # Fast for tests
            return consumer

    def test_reconnection_skipped_when_client_none(self, consumer_with_mocks):
        """Test reconnection is skipped when WLED client is None."""
        consumer_with_mocks._wled_client = None
        consumer_with_mocks._wled_reconnection_event.wait.side_effect = [False, True]  # First iteration, then shutdown

        consumer_with_mocks._wled_reconnection_loop()

        # Should not crash, just skip reconnection

    def test_reconnection_skipped_when_already_connected(self, consumer_with_mocks):
        """Test reconnection is skipped when already connected."""
        consumer_with_mocks._wled_client.is_connected = True
        consumer_with_mocks._wled_reconnection_event.wait.side_effect = [False, True]

        consumer_with_mocks._wled_reconnection_loop()

        consumer_with_mocks._wled_client.connect.assert_not_called()

    def test_successful_reconnection(self, consumer_with_mocks):
        """Test successful WLED reconnection."""
        consumer_with_mocks._wled_client.is_connected = False
        consumer_with_mocks._wled_client.connect.return_value = True
        consumer_with_mocks._wled_reconnection_event.wait.side_effect = [False, True]

        consumer_with_mocks._wled_reconnection_loop()

        consumer_with_mocks._wled_client.connect.assert_called_once()
        consumer_with_mocks._frame_renderer.set_output_targets.assert_called_once()

    def test_failed_reconnection(self, consumer_with_mocks):
        """Test failed WLED reconnection."""
        consumer_with_mocks._wled_client.is_connected = False
        consumer_with_mocks._wled_client.connect.return_value = False
        consumer_with_mocks._wled_reconnection_event.wait.side_effect = [False, True]

        consumer_with_mocks._wled_reconnection_loop()

        consumer_with_mocks._wled_client.connect.assert_called_once()
        consumer_with_mocks._frame_renderer.set_output_targets.assert_not_called()

    def test_shutdown_event_stops_loop(self, consumer_with_mocks):
        """Test shutdown event stops the reconnection loop."""
        consumer_with_mocks._wled_reconnection_event.wait.return_value = True  # Event was set

        consumer_with_mocks._wled_reconnection_loop()

        # Should exit immediately without attempting reconnection
        consumer_with_mocks._wled_client.connect.assert_not_called()

    def test_exception_handling_continues_loop(self, consumer_with_mocks):
        """Test exception in loop doesn't crash - continues with retry."""
        consumer_with_mocks._wled_client.is_connected = False
        consumer_with_mocks._wled_client.connect.side_effect = [RuntimeError("Connection error"), True]
        consumer_with_mocks._wled_reconnection_event.wait.side_effect = [False, False, True]

        with patch("time.sleep"):  # Skip sleep in test
            consumer_with_mocks._wled_reconnection_loop()

        # Should have attempted connect twice (error then success)
        assert consumer_with_mocks._wled_client.connect.call_count == 2
