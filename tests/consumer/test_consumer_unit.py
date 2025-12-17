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
from src.core.control_state import RendererState


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
