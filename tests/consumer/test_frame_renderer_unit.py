"""
Unit tests for FrameRenderer class and related components.

Tests the timestamp-based frame renderer, effect trigger management,
and LED output handling.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.consumer.frame_renderer import (
    EffectTriggerConfig,
    EffectTriggerManager,
    FrameRenderer,
)

# =============================================================================
# EffectTriggerConfig Tests
# =============================================================================


class TestEffectTriggerConfig:
    """Test EffectTriggerConfig dataclass validation."""

    def test_valid_beat_trigger_with_conditions(self):
        """Test valid beat trigger with conditions."""
        config = EffectTriggerConfig(
            trigger_type="beat",
            effect_class="BeatBrightnessEffect",
            effect_params={"boost_intensity": 2.0},
            confidence_min=0.5,
        )
        assert config.trigger_type == "beat"
        assert config.confidence_min == 0.5

    def test_valid_test_trigger(self):
        """Test valid test trigger."""
        config = EffectTriggerConfig(
            trigger_type="test",
            effect_class="TemplateEffect",
            effect_params={"template_path": "test.npy"},
        )
        assert config.trigger_type == "test"
        assert config.effect_class == "TemplateEffect"

    def test_invalid_trigger_type_raises_error(self):
        """Test that invalid trigger type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid trigger_type"):
            EffectTriggerConfig(
                trigger_type="invalid",
                effect_class="BeatBrightnessEffect",
            )

    def test_beat_trigger_without_conditions_logs_warning(self, caplog):
        """Test that beat trigger without conditions logs warning."""
        with caplog.at_level("WARNING"):
            EffectTriggerConfig(
                trigger_type="beat",
                effect_class="BeatBrightnessEffect",
            )
        assert "Beat trigger has no conditions" in caplog.text

    def test_beat_trigger_with_bpm_range(self):
        """Test beat trigger with BPM range conditions."""
        config = EffectTriggerConfig(
            trigger_type="beat",
            effect_class="BeatBrightnessEffect",
            bpm_min=100.0,
            bpm_max=140.0,
        )
        assert config.bpm_min == 100.0
        assert config.bpm_max == 140.0

    def test_beat_trigger_with_intensity_min(self):
        """Test beat trigger with intensity minimum."""
        config = EffectTriggerConfig(
            trigger_type="beat",
            effect_class="BeatBrightnessEffect",
            intensity_min=0.7,
        )
        assert config.intensity_min == 0.7


# =============================================================================
# EffectTriggerManager Tests
# =============================================================================


class TestEffectTriggerManager:
    """Test EffectTriggerManager class."""

    @pytest.fixture
    def mock_effect_manager(self):
        """Create mock LedEffectManager."""
        manager = MagicMock()
        manager.add_effect = Mock()
        return manager

    @pytest.fixture
    def trigger_manager(self, mock_effect_manager):
        """Create EffectTriggerManager with mock effect manager."""
        return EffectTriggerManager(mock_effect_manager)

    def test_init(self, trigger_manager, mock_effect_manager):
        """Test EffectTriggerManager initialization."""
        assert trigger_manager.effect_manager == mock_effect_manager
        assert trigger_manager.common_triggers == []
        assert trigger_manager.carousel_rule_sets == []
        assert trigger_manager.carousel_beat_interval == 4
        assert trigger_manager.test_trigger_interval == 2.0

    def test_set_triggers(self, trigger_manager):
        """Test set_triggers for backward compatibility."""
        triggers = [EffectTriggerConfig(trigger_type="beat", effect_class="BeatBrightnessEffect", confidence_min=0.5)]
        trigger_manager.set_triggers(triggers)

        assert trigger_manager.common_triggers == triggers
        assert trigger_manager.carousel_rule_sets == []

    def test_set_common_and_carousel_triggers(self, trigger_manager):
        """Test set_common_and_carousel_triggers."""
        common = [EffectTriggerConfig(trigger_type="beat", effect_class="Effect1", confidence_min=0.5)]
        carousel = [
            [EffectTriggerConfig(trigger_type="beat", effect_class="Effect2", intensity_min=0.3)],
            [EffectTriggerConfig(trigger_type="beat", effect_class="Effect3", intensity_min=0.5)],
        ]

        trigger_manager.set_common_and_carousel_triggers(common, carousel, carousel_beat_interval=8)

        assert trigger_manager.common_triggers == common
        assert trigger_manager.carousel_rule_sets == carousel
        assert trigger_manager.carousel_beat_interval == 8
        assert trigger_manager.carousel_current_index == 0
        assert trigger_manager.carousel_beat_count == 0

    def test_set_test_interval(self, trigger_manager):
        """Test set_test_interval."""
        trigger_manager.set_test_interval(5.0)
        assert trigger_manager.test_trigger_interval == 5.0

    def test_check_beat_conditions_confidence_min(self, trigger_manager):
        """Test _check_beat_conditions with confidence_min."""
        trigger = EffectTriggerConfig(trigger_type="beat", effect_class="Test", confidence_min=0.8)

        # Below threshold
        assert not trigger_manager._check_beat_conditions(trigger, 1.0, 0.5, 120.0)
        # Above threshold
        assert trigger_manager._check_beat_conditions(trigger, 1.0, 0.9, 120.0)

    def test_check_beat_conditions_intensity_min(self, trigger_manager):
        """Test _check_beat_conditions with intensity_min."""
        trigger = EffectTriggerConfig(trigger_type="beat", effect_class="Test", intensity_min=0.5)

        # Below threshold
        assert not trigger_manager._check_beat_conditions(trigger, 0.3, 1.0, 120.0)
        # Above threshold
        assert trigger_manager._check_beat_conditions(trigger, 0.7, 1.0, 120.0)

    def test_check_beat_conditions_bpm_range(self, trigger_manager):
        """Test _check_beat_conditions with BPM range."""
        trigger = EffectTriggerConfig(trigger_type="beat", effect_class="Test", bpm_min=100.0, bpm_max=140.0)

        # Below range
        assert not trigger_manager._check_beat_conditions(trigger, 1.0, 1.0, 80.0)
        # In range
        assert trigger_manager._check_beat_conditions(trigger, 1.0, 1.0, 120.0)
        # Above range
        assert not trigger_manager._check_beat_conditions(trigger, 1.0, 1.0, 160.0)

    def test_check_beat_conditions_all_conditions(self, trigger_manager):
        """Test _check_beat_conditions with all conditions."""
        trigger = EffectTriggerConfig(
            trigger_type="beat",
            effect_class="Test",
            confidence_min=0.5,
            intensity_min=0.3,
            bpm_min=100.0,
            bpm_max=140.0,
        )

        # All conditions met
        assert trigger_manager._check_beat_conditions(trigger, 0.5, 0.7, 120.0)
        # One condition not met: beat_confidence (0.3) < confidence_min (0.5)
        assert not trigger_manager._check_beat_conditions(trigger, 0.5, 0.3, 120.0)

    def test_carousel_rotation(self, trigger_manager):
        """Test _check_carousel_rotation rotates carousel."""
        carousel = [
            [EffectTriggerConfig(trigger_type="beat", effect_class="Effect1", confidence_min=0.5)],
            [EffectTriggerConfig(trigger_type="beat", effect_class="Effect2", confidence_min=0.5)],
        ]
        trigger_manager.set_common_and_carousel_triggers([], carousel, carousel_beat_interval=2)

        # Not at interval yet
        trigger_manager.carousel_beat_count = 1
        trigger_manager._check_carousel_rotation()
        assert trigger_manager.carousel_current_index == 0

        # At interval
        trigger_manager.carousel_beat_count = 2
        trigger_manager._check_carousel_rotation()
        assert trigger_manager.carousel_current_index == 1
        assert trigger_manager.carousel_beat_count == 0

    def test_carousel_rotation_wraps_around(self, trigger_manager):
        """Test _check_carousel_rotation wraps around to beginning."""
        carousel = [
            [EffectTriggerConfig(trigger_type="beat", effect_class="Effect1", confidence_min=0.5)],
            [EffectTriggerConfig(trigger_type="beat", effect_class="Effect2", confidence_min=0.5)],
        ]
        trigger_manager.set_common_and_carousel_triggers([], carousel, carousel_beat_interval=1)

        # Start at index 1 (last index)
        trigger_manager.carousel_current_index = 1
        trigger_manager.carousel_beat_count = 1

        trigger_manager._check_carousel_rotation()
        assert trigger_manager.carousel_current_index == 0  # Wrapped around

    def test_evaluate_beat_triggers_skips_already_processed(self, trigger_manager, mock_effect_manager):
        """Test evaluate_beat_triggers skips already processed beats."""
        trigger_manager.set_triggers(
            [EffectTriggerConfig(trigger_type="beat", effect_class="BeatBrightnessEffect", confidence_min=0.1)]
        )

        beat_state = Mock()
        beat_state.beat_intensity_ready = True
        beat_state.beat_intensity = 1.0
        beat_state.confidence = 1.0
        beat_state.current_bpm = 120.0

        # First call at beat time 1.0
        trigger_manager.evaluate_beat_triggers(0.0, beat_state, 1.0, 0.0)
        assert mock_effect_manager.add_effect.call_count == 1

        # Second call at same beat time - should be skipped
        trigger_manager.evaluate_beat_triggers(0.0, beat_state, 1.0, 0.0)
        assert mock_effect_manager.add_effect.call_count == 1  # No new call

    def test_evaluate_beat_triggers_waits_for_intensity_ready(self, trigger_manager, mock_effect_manager):
        """Test evaluate_beat_triggers waits for beat intensity to be ready."""
        trigger_manager.set_triggers(
            [EffectTriggerConfig(trigger_type="beat", effect_class="BeatBrightnessEffect", confidence_min=0.1)]
        )

        beat_state = Mock()
        beat_state.beat_intensity_ready = False  # Not ready
        beat_state.beat_intensity = 1.0
        beat_state.confidence = 1.0
        beat_state.current_bpm = 120.0

        trigger_manager.evaluate_beat_triggers(0.0, beat_state, 1.0, 0.0)
        mock_effect_manager.add_effect.assert_not_called()

    def test_evaluate_test_triggers(self, trigger_manager, mock_effect_manager):
        """Test evaluate_test_triggers creates effects."""
        # Use an effect that doesn't require special parameters
        trigger_manager.set_triggers(
            [
                EffectTriggerConfig(
                    trigger_type="test",
                    effect_class="BeatBrightnessEffect",
                    effect_params={"boost_intensity": 2.0, "duration_fraction": 0.5, "bpm": 120.0},
                )
            ]
        )
        trigger_manager.set_test_interval(1.0)

        # First call - should create effect
        trigger_manager.evaluate_test_triggers(0.0)
        assert mock_effect_manager.add_effect.call_count == 1

        # Second call too soon - should not create effect
        trigger_manager.evaluate_test_triggers(0.5)
        assert mock_effect_manager.add_effect.call_count == 1

        # Third call after interval - should create effect
        trigger_manager.evaluate_test_triggers(1.5)
        assert mock_effect_manager.add_effect.call_count == 2


# =============================================================================
# FrameRenderer Tests - Initialization
# =============================================================================


class TestFrameRendererInit:
    """Test FrameRenderer initialization."""

    def test_init_default_parameters(self):
        """Test FrameRenderer initialization with default parameters."""
        renderer = FrameRenderer(led_ordering=None)

        assert renderer.first_frame_delay == 0.1  # 100ms
        assert renderer.timing_tolerance == 0.005  # 5ms
        assert renderer.late_frame_log_threshold == 0.05  # 50ms
        assert renderer.led_ordering is None
        assert not renderer.first_frame_received
        assert renderer.wallclock_delta is None
        assert renderer.sinks == []

    def test_init_custom_parameters(self):
        """Test FrameRenderer initialization with custom parameters."""
        led_ordering = np.array([2, 0, 1])
        renderer = FrameRenderer(
            led_ordering=led_ordering,
            first_frame_delay_ms=200.0,
            timing_tolerance_ms=10.0,
            late_frame_log_threshold_ms=100.0,
        )

        assert renderer.first_frame_delay == 0.2  # 200ms
        assert renderer.timing_tolerance == 0.01  # 10ms
        assert renderer.late_frame_log_threshold == 0.1  # 100ms
        np.testing.assert_array_equal(renderer.led_ordering, led_ordering)

    def test_init_with_control_state(self):
        """Test FrameRenderer initialization with control state."""
        control_state = MagicMock()
        control_state.get_status.return_value = None

        renderer = FrameRenderer(led_ordering=None, control_state=control_state)

        assert renderer._control_state == control_state

    def test_init_with_audio_analyzer(self):
        """Test FrameRenderer initialization with audio analyzer."""
        audio_analyzer = MagicMock()

        renderer = FrameRenderer(led_ordering=None, audio_beat_analyzer=audio_analyzer)

        assert renderer._audio_beat_analyzer == audio_analyzer


# =============================================================================
# FrameRenderer Tests - Timing
# =============================================================================


class TestFrameRendererTiming:
    """Test FrameRenderer timing functionality."""

    @pytest.fixture
    def renderer(self):
        """Create FrameRenderer instance for testing."""
        return FrameRenderer(led_ordering=None)

    def test_establish_wallclock_delta(self, renderer):
        """Test establish_wallclock_delta sets delta correctly."""
        before_time = time.time()
        renderer.establish_wallclock_delta(first_timestamp=0.0)
        after_time = time.time()

        assert renderer.first_frame_received
        assert renderer.first_frame_timestamp == 0.0
        # Delta should be approximately current time + delay
        assert renderer.wallclock_delta >= before_time + renderer.first_frame_delay
        assert renderer.wallclock_delta <= after_time + renderer.first_frame_delay

    def test_establish_wallclock_delta_already_established(self, renderer, caplog):
        """Test establish_wallclock_delta logs warning if already established."""
        renderer.establish_wallclock_delta(0.0)
        first_delta = renderer.wallclock_delta

        with caplog.at_level("WARNING"):
            renderer.establish_wallclock_delta(1.0)

        assert "already established" in caplog.text
        assert renderer.wallclock_delta == first_delta  # Unchanged

    def test_is_initialized(self, renderer):
        """Test is_initialized returns correct state."""
        assert not renderer.is_initialized()

        renderer.establish_wallclock_delta(0.0)

        assert renderer.is_initialized()

    def test_get_adjusted_wallclock_delta_not_set(self, renderer):
        """Test get_adjusted_wallclock_delta returns 0 if not set."""
        assert renderer.get_adjusted_wallclock_delta() == 0.0

    def test_get_adjusted_wallclock_delta_with_pause(self, renderer):
        """Test get_adjusted_wallclock_delta includes pause time."""
        renderer.establish_wallclock_delta(0.0)
        base_delta = renderer.wallclock_delta

        renderer.total_pause_time = 1.0

        assert renderer.get_adjusted_wallclock_delta() == base_delta + 1.0

    def test_get_adjusted_wallclock_delta_while_paused(self, renderer):
        """Test get_adjusted_wallclock_delta includes current pause."""
        renderer.establish_wallclock_delta(0.0)
        base_delta = renderer.wallclock_delta

        renderer.is_paused = True
        renderer.pause_start_time = time.time() - 0.5  # Paused 0.5s ago

        adjusted = renderer.get_adjusted_wallclock_delta()
        assert adjusted >= base_delta + 0.5

    def test_is_frame_late_not_initialized(self, renderer):
        """Test is_frame_late returns False when not initialized."""
        assert not renderer.is_frame_late(0.0)

    def test_is_frame_late_first_frame_of_item(self, renderer):
        """Test is_frame_late returns False for first frame of new item."""
        renderer.establish_wallclock_delta(0.0)

        # Even with a very old timestamp, first frame of item should not be late
        metadata = {"is_first_frame_of_item": True}
        assert not renderer.is_frame_late(-10.0, metadata=metadata)

    def test_is_frame_late_within_threshold(self, renderer):
        """Test is_frame_late returns False when within threshold."""
        renderer.establish_wallclock_delta(0.0)

        # Frame timestamp that would be on time
        current = time.time()
        frame_ts = current - renderer.wallclock_delta

        assert not renderer.is_frame_late(frame_ts, late_threshold_ms=50.0)

    def test_is_frame_late_exceeds_threshold(self, renderer):
        """Test is_frame_late returns True when exceeding threshold."""
        renderer.establish_wallclock_delta(0.0)

        # Frame timestamp that is very late (from 1 second ago)
        current = time.time()
        frame_ts = current - renderer.wallclock_delta - 1.0

        assert renderer.is_frame_late(frame_ts, late_threshold_ms=50.0)


# =============================================================================
# FrameRenderer Tests - Pause/Resume
# =============================================================================


class TestFrameRendererPauseResume:
    """Test FrameRenderer pause and resume functionality."""

    @pytest.fixture
    def renderer(self):
        """Create FrameRenderer instance for testing."""
        return FrameRenderer(led_ordering=None)

    def test_pause_renderer(self, renderer):
        """Test pause_renderer sets pause state."""
        assert not renderer.is_paused
        assert renderer.pause_start_time is None

        renderer.pause_renderer()

        assert renderer.is_paused
        assert renderer.pause_start_time is not None

    def test_pause_renderer_already_paused(self, renderer):
        """Test pause_renderer is no-op if already paused."""
        renderer.pause_renderer()
        original_pause_time = renderer.pause_start_time

        time.sleep(0.01)
        renderer.pause_renderer()

        assert renderer.pause_start_time == original_pause_time

    def test_resume_renderer(self, renderer):
        """Test resume_renderer clears pause state and updates total."""
        renderer.pause_renderer()
        time.sleep(0.05)  # Pause for 50ms

        renderer.resume_renderer()

        assert not renderer.is_paused
        assert renderer.pause_start_time is None
        assert renderer.total_pause_time >= 0.05

    def test_resume_renderer_not_paused(self, renderer):
        """Test resume_renderer is no-op if not paused."""
        renderer.resume_renderer()

        assert not renderer.is_paused
        assert renderer.total_pause_time == 0.0


# =============================================================================
# FrameRenderer Tests - Sink Management
# =============================================================================


class TestFrameRendererSinks:
    """Test FrameRenderer sink management."""

    @pytest.fixture
    def renderer(self):
        """Create FrameRenderer instance for testing."""
        return FrameRenderer(led_ordering=None)

    def test_register_sink_with_send_led_data(self, renderer):
        """Test register_sink with send_led_data method."""
        sink = Mock()
        sink.send_led_data = Mock()

        renderer.register_sink(sink, "TestSink", enabled=True)

        assert len(renderer.sinks) == 1
        assert renderer.sinks[0]["name"] == "TestSink"
        assert renderer.sinks[0]["enabled"]

    def test_register_sink_with_render_led_values(self, renderer):
        """Test register_sink with render_led_values method."""
        sink = Mock()
        sink.render_led_values = Mock()
        del sink.send_led_data  # Remove the auto-mocked method

        renderer.register_sink(sink, "RenderSink", enabled=False)

        assert len(renderer.sinks) == 1
        assert renderer.sinks[0]["name"] == "RenderSink"
        assert not renderer.sinks[0]["enabled"]

    def test_register_sink_invalid_interface(self, renderer):
        """Test register_sink raises error for invalid interface."""
        sink = Mock(spec=[])  # No methods

        with pytest.raises(ValueError, match="must have"):
            renderer.register_sink(sink, "BadSink")

    def test_unregister_sink(self, renderer):
        """Test unregister_sink removes sink."""
        sink = Mock()
        sink.send_led_data = Mock()

        renderer.register_sink(sink, "TestSink")
        assert len(renderer.sinks) == 1

        renderer.unregister_sink(sink)
        assert len(renderer.sinks) == 0

    def test_set_sink_enabled(self, renderer):
        """Test set_sink_enabled toggles sink state."""
        sink = Mock()
        sink.send_led_data = Mock()

        renderer.register_sink(sink, "TestSink", enabled=True)
        assert renderer.sinks[0]["enabled"]

        renderer.set_sink_enabled(sink, False)
        assert not renderer.sinks[0]["enabled"]

        renderer.set_sink_enabled(sink, True)
        assert renderer.sinks[0]["enabled"]

    def test_set_output_targets(self, renderer):
        """Test set_output_targets configures sinks."""
        wled = Mock()
        wled.send_led_data = Mock()
        test = Mock()
        test.render_led_values = Mock()
        preview = Mock()
        preview.render_led_values = Mock()

        renderer.set_output_targets(wled_sink=wled, test_sink=test, preview_sink=preview)

        assert len(renderer.sinks) == 3
        assert renderer.wled_sink == wled
        assert renderer.test_sink == test
        assert renderer.enable_wled

    def test_set_wled_enabled(self, renderer):
        """Test set_wled_enabled controls WLED output."""
        wled = Mock()
        wled.send_led_data = Mock()
        renderer.set_output_targets(wled_sink=wled)

        renderer.set_wled_enabled(False)
        assert not renderer.enable_wled

        renderer.set_wled_enabled(True)
        assert renderer.enable_wled


# =============================================================================
# FrameRenderer Tests - LED Conversion
# =============================================================================


class TestFrameRendererLEDConversion:
    """Test FrameRenderer LED value conversion."""

    def test_convert_spatial_to_physical_no_ordering(self):
        """Test _convert_spatial_to_physical returns unchanged without ordering."""
        renderer = FrameRenderer(led_ordering=None)

        led_values = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        result = renderer._convert_spatial_to_physical(led_values)

        np.testing.assert_array_equal(result, led_values)

    def test_convert_spatial_to_physical_with_ordering(self):
        """Test _convert_spatial_to_physical reorders correctly."""
        led_ordering = np.array([2, 0, 1])  # spatial[0] -> physical[2], etc.
        renderer = FrameRenderer(led_ordering=led_ordering)

        led_values = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        result = renderer._convert_spatial_to_physical(led_values)

        # spatial[0] (red) -> physical[2]
        # spatial[1] (green) -> physical[0]
        # spatial[2] (blue) -> physical[1]
        expected = np.array([[0, 255, 0], [0, 0, 255], [255, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# FrameRenderer Tests - Statistics
# =============================================================================


class TestFrameRendererStatistics:
    """Test FrameRenderer statistics functionality."""

    @pytest.fixture
    def renderer(self):
        """Create FrameRenderer instance for testing."""
        return FrameRenderer(led_ordering=None)

    def test_reset_stats(self, renderer):
        """Test reset_stats clears all counters."""
        # Set some values
        renderer.frames_rendered = 100
        renderer.late_frames = 10
        renderer.early_frames = 5
        renderer.on_time_frames = 85
        renderer.dropped_frames = 2
        renderer.total_wait_time = 1.5
        renderer.total_late_time = 0.5
        renderer.timing_errors = [0.01, 0.02]
        renderer.ewma_fps = 30.0
        renderer.is_paused = True
        renderer.total_pause_time = 1.0

        renderer.reset_stats()

        assert renderer.frames_rendered == 0
        assert renderer.late_frames == 0
        assert renderer.early_frames == 0
        assert renderer.on_time_frames == 0
        assert renderer.dropped_frames == 0
        assert renderer.total_wait_time == 0.0
        assert renderer.total_late_time == 0.0
        assert len(renderer.timing_errors) == 0
        assert renderer.ewma_fps == 0.0
        assert not renderer.is_paused
        assert renderer.total_pause_time == 0.0

    def test_set_timing_parameters(self, renderer):
        """Test set_timing_parameters updates values."""
        renderer.set_timing_parameters(
            first_frame_delay_ms=200.0,
            timing_tolerance_ms=20.0,
            late_frame_log_threshold_ms=100.0,
        )

        assert renderer.first_frame_delay == 0.2
        assert renderer.timing_tolerance == 0.02
        assert renderer.late_frame_log_threshold == 0.1

    def test_set_timing_parameters_partial(self, renderer):
        """Test set_timing_parameters with partial values."""
        original_delay = renderer.first_frame_delay

        renderer.set_timing_parameters(timing_tolerance_ms=15.0)

        assert renderer.first_frame_delay == original_delay  # Unchanged
        assert renderer.timing_tolerance == 0.015

    def test_set_ewma_alpha_valid(self, renderer):
        """Test set_ewma_alpha with valid value."""
        renderer.set_ewma_alpha(0.2)
        assert renderer.ewma_alpha == 0.2

    def test_set_ewma_alpha_invalid_low(self, renderer):
        """Test set_ewma_alpha raises error for value <= 0."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            renderer.set_ewma_alpha(0.0)

    def test_set_ewma_alpha_invalid_high(self, renderer):
        """Test set_ewma_alpha raises error for value > 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            renderer.set_ewma_alpha(1.5)

    def test_mark_frame_dropped(self, renderer):
        """Test mark_frame_dropped increments counter."""
        assert renderer.dropped_frames == 0

        renderer.mark_frame_dropped()
        assert renderer.dropped_frames == 1

        renderer.mark_frame_dropped()
        assert renderer.dropped_frames == 2

    def test_get_timing_stats(self, renderer):
        """Test get_timing_stats returns comprehensive stats."""
        renderer.frames_rendered = 100
        renderer.late_frames = 10
        renderer.early_frames = 20
        renderer.on_time_frames = 70

        stats = renderer.get_timing_stats()

        assert stats["frames_rendered"] == 100
        assert stats["late_frames"] == 10
        assert stats["early_frames"] == 20
        assert stats["on_time_frames"] == 70
        assert stats["late_frame_percentage"] == 10.0
        assert stats["early_frame_percentage"] == 20.0
        assert stats["on_time_percentage"] == 70.0
        assert "ewma_fps" in stats
        assert "first_frame_delay_ms" in stats
        assert "registered_sinks" in stats

    def test_get_renderer_stats_alias(self, renderer):
        """Test get_renderer_stats is alias for get_timing_stats."""
        timing_stats = renderer.get_timing_stats()
        renderer_stats = renderer.get_renderer_stats()

        assert timing_stats == renderer_stats

    def test_get_recent_performance_summary(self, renderer):
        """Test get_recent_performance_summary returns EWMA metrics."""
        renderer.ewma_fps = 30.0
        renderer.ewma_late_fraction = 0.05
        renderer.ewma_dropped_fraction = 0.02
        renderer.frames_rendered = 100

        summary = renderer.get_recent_performance_summary()

        assert summary["recent_fps"] == 30.0
        assert summary["recent_late_percentage"] == 5.0
        assert summary["recent_dropped_percentage"] == 2.0
        assert summary["frames_rendered"] == 100
        assert summary["is_performing_well"]  # Good performance

    def test_get_recent_performance_summary_poor(self, renderer):
        """Test get_recent_performance_summary detects poor performance."""
        renderer.ewma_fps = 15.0  # Below 25 threshold
        renderer.ewma_late_fraction = 0.05
        renderer.ewma_dropped_fraction = 0.02

        summary = renderer.get_recent_performance_summary()

        assert not summary["is_performing_well"]

    def test_track_timing_error(self, renderer):
        """Test _track_timing_error adds to list."""
        renderer._track_timing_error(0.001)
        renderer._track_timing_error(-0.002)

        assert len(renderer.timing_errors) == 2
        assert renderer.timing_errors[0] == 0.001
        assert renderer.timing_errors[1] == -0.002

    def test_track_timing_error_trims_history(self, renderer):
        """Test _track_timing_error trims to max history."""
        renderer.max_timing_history = 5

        for i in range(10):
            renderer._track_timing_error(float(i))

        assert len(renderer.timing_errors) == 5
        assert renderer.timing_errors[0] == 5.0  # Oldest retained

    def test_get_output_fps(self, renderer):
        """Test get_output_fps returns EWMA FPS."""
        renderer.output_fps_ewma = 30.0
        assert renderer.get_output_fps() == 30.0


# =============================================================================
# FrameRenderer Tests - LED Effects Interface
# =============================================================================


class TestFrameRendererEffects:
    """Test FrameRenderer LED effects interface."""

    @pytest.fixture
    def renderer(self):
        """Create FrameRenderer instance for testing."""
        return FrameRenderer(led_ordering=None)

    def test_add_led_effect(self, renderer):
        """Test add_led_effect delegates to effect_manager."""
        effect = Mock()
        renderer.effect_manager = Mock()

        renderer.add_led_effect(effect)

        renderer.effect_manager.add_effect.assert_called_once_with(effect)

    def test_clear_led_effects(self, renderer):
        """Test clear_led_effects delegates to effect_manager."""
        renderer.effect_manager = Mock()

        renderer.clear_led_effects()

        renderer.effect_manager.clear_effects.assert_called_once()

    def test_get_active_effects_count(self, renderer):
        """Test get_active_effects_count delegates to effect_manager."""
        renderer.effect_manager = Mock()
        renderer.effect_manager.get_active_count.return_value = 3

        count = renderer.get_active_effects_count()

        assert count == 3
        renderer.effect_manager.get_active_count.assert_called_once()

    def test_get_led_effects_stats(self, renderer):
        """Test get_led_effects_stats delegates to effect_manager."""
        expected_stats = {"active_effects": 2, "total_created": 10}
        renderer.effect_manager = Mock()
        renderer.effect_manager.get_stats.return_value = expected_stats

        stats = renderer.get_led_effects_stats()

        assert stats == expected_stats
        renderer.effect_manager.get_stats.assert_called_once()


# =============================================================================
# FrameRenderer Tests - Render Frame
# =============================================================================


class TestFrameRendererRender:
    """Test FrameRenderer render functionality."""

    @pytest.fixture
    def renderer(self):
        """Create FrameRenderer with mocked components."""
        renderer = FrameRenderer(led_ordering=None)
        renderer.effect_manager = Mock()
        renderer.effect_manager.apply_effects = Mock()
        renderer.effect_manager.get_stats = Mock(return_value={})
        return renderer

    def test_render_frame_at_timestamp_first_frame(self, renderer):
        """Test render_frame_at_timestamp establishes delta on first frame."""
        led_values = np.zeros((10, 3), dtype=np.uint8)

        assert not renderer.first_frame_received

        result = renderer.render_frame_at_timestamp(led_values, 0.0)

        assert result
        assert renderer.first_frame_received
        assert renderer.wallclock_delta is not None
        assert renderer.frames_rendered == 1

    def test_render_frame_at_timestamp_increments_stats(self, renderer):
        """Test render_frame_at_timestamp increments frame counter."""
        led_values = np.zeros((10, 3), dtype=np.uint8)
        renderer.establish_wallclock_delta(0.0)

        # Render at a time that will be on time
        frame_ts = time.time() - renderer.wallclock_delta

        renderer.render_frame_at_timestamp(led_values, frame_ts)

        assert renderer.frames_rendered == 1

    def test_render_frame_at_timestamp_returns_false_on_error(self, renderer):
        """Test render_frame_at_timestamp returns False on error."""
        renderer.establish_wallclock_delta(0.0)

        # Make _send_to_outputs raise an error
        renderer._send_to_outputs = Mock(side_effect=Exception("Test error"))

        led_values = np.zeros((10, 3), dtype=np.uint8)
        frame_ts = time.time() - renderer.wallclock_delta

        result = renderer.render_frame_at_timestamp(led_values, frame_ts)

        assert not result


# =============================================================================
# FrameRenderer Tests - Helper Methods
# =============================================================================


class TestFrameRendererHelpers:
    """Test FrameRenderer helper methods."""

    @pytest.fixture
    def renderer(self):
        """Create FrameRenderer instance for testing."""
        return FrameRenderer(led_ordering=None)

    def test_calculate_sparkle_param_linear(self, renderer):
        """Test _calculate_sparkle_param with linear curve."""
        config = {"min": 0.0, "max": 1.0, "curve": "linear"}

        assert renderer._calculate_sparkle_param(0.0, config) == 0.0
        assert renderer._calculate_sparkle_param(5.0, config) == 0.5
        assert renderer._calculate_sparkle_param(10.0, config) == 1.0

    def test_calculate_sparkle_param_ease_in(self, renderer):
        """Test _calculate_sparkle_param with ease-in curve."""
        config = {"min": 0.0, "max": 1.0, "curve": "ease-in"}

        # t^2 curve
        assert renderer._calculate_sparkle_param(0.0, config) == 0.0
        assert renderer._calculate_sparkle_param(5.0, config) == 0.25  # 0.5^2
        assert renderer._calculate_sparkle_param(10.0, config) == 1.0

    def test_calculate_sparkle_param_ease_out(self, renderer):
        """Test _calculate_sparkle_param with ease-out curve."""
        config = {"min": 0.0, "max": 1.0, "curve": "ease-out"}

        # 1 - (1-t)^2 curve
        assert renderer._calculate_sparkle_param(0.0, config) == 0.0
        assert renderer._calculate_sparkle_param(5.0, config) == 0.75  # 1 - 0.5^2
        assert renderer._calculate_sparkle_param(10.0, config) == 1.0

    def test_calculate_sparkle_param_inverse(self, renderer):
        """Test _calculate_sparkle_param with inverse curve."""
        config = {"min": 0.0, "max": 1.0, "curve": "inverse"}

        # 1 - t curve
        assert renderer._calculate_sparkle_param(0.0, config) == 1.0
        assert renderer._calculate_sparkle_param(5.0, config) == 0.5
        assert renderer._calculate_sparkle_param(10.0, config) == 0.0

    def test_calculate_sparkle_param_clamps_intensity(self, renderer):
        """Test _calculate_sparkle_param clamps intensity to 0-10."""
        config = {"min": 0.0, "max": 100.0, "curve": "linear"}

        # Negative intensity clamped to 0
        assert renderer._calculate_sparkle_param(-5.0, config) == 0.0
        # Above 10 clamped to 10
        assert renderer._calculate_sparkle_param(15.0, config) == 100.0

    def test_calculate_sparkle_param_with_range(self, renderer):
        """Test _calculate_sparkle_param with custom min/max."""
        config = {"min": 30.0, "max": 300.0, "curve": "linear"}

        assert renderer._calculate_sparkle_param(0.0, config) == 30.0
        assert renderer._calculate_sparkle_param(5.0, config) == 165.0  # (30 + 300) / 2
        assert renderer._calculate_sparkle_param(10.0, config) == 300.0


# =============================================================================
# FrameRenderer Tests - Beat and Effect Management
# =============================================================================


class TestFrameRendererBeatEffects:
    """Test FrameRenderer beat detection and effect management."""

    @pytest.fixture
    def renderer_with_audio(self):
        """Create FrameRenderer with mocked audio components."""
        control_state = Mock()
        status = Mock()
        status.audio_reactive_enabled = True
        status.audio_enabled = True
        status.audio_reactive_trigger_config = None
        control_state.get_status.return_value = status

        audio_analyzer = Mock()
        beat_state = Mock()
        beat_state.is_active = True
        beat_state.current_bpm = 120.0
        beat_state.last_beat_wallclock_time = 0.0
        beat_state.beat_intensity = 1.0
        beat_state.confidence = 1.0
        beat_state.beat_intensity_ready = True
        audio_analyzer.get_current_state.return_value = beat_state

        renderer = FrameRenderer(
            led_ordering=None,
            control_state=control_state,
            audio_beat_analyzer=audio_analyzer,
        )
        renderer.effect_manager = Mock()
        renderer.effect_manager.apply_effects = Mock()
        renderer.effect_manager.get_stats = Mock(return_value={})
        renderer.establish_wallclock_delta(0.0)

        return renderer

    def test_check_beat_brightness_no_control_state(self):
        """Test _check_and_create_beat_brightness_effect returns if no control state."""
        renderer = FrameRenderer(led_ordering=None)
        renderer.trigger_manager = Mock()

        renderer._check_and_create_beat_brightness_effect(0.0)

        renderer.trigger_manager.evaluate_beat_triggers.assert_not_called()

    def test_check_beat_brightness_no_audio_analyzer(self):
        """Test _check_and_create_beat_brightness_effect returns if no audio analyzer."""
        control_state = Mock()
        # Set up status with None trigger config to avoid JSON serialization
        status = Mock()
        status.audio_reactive_trigger_config = None
        status.beat_brightness_intensity = 4.0
        status.beat_brightness_duration = 0.4
        status.beat_confidence_threshold = 0.5
        control_state.get_status.return_value = status

        renderer = FrameRenderer(led_ordering=None, control_state=control_state)
        renderer.trigger_manager = Mock()

        renderer._check_and_create_beat_brightness_effect(0.0)

        renderer.trigger_manager.evaluate_beat_triggers.assert_not_called()

    def test_check_beat_brightness_audio_disabled(self):
        """Test _check_and_create_beat_brightness_effect returns if audio disabled."""
        control_state = Mock()
        status = Mock()
        status.audio_reactive_enabled = False
        status.audio_reactive_trigger_config = None
        status.beat_brightness_intensity = 4.0
        status.beat_brightness_duration = 0.4
        status.beat_confidence_threshold = 0.5
        control_state.get_status.return_value = status

        audio_analyzer = Mock()

        renderer = FrameRenderer(
            led_ordering=None,
            control_state=control_state,
            audio_beat_analyzer=audio_analyzer,
        )
        renderer.trigger_manager = Mock()

        renderer._check_and_create_beat_brightness_effect(0.0)

        renderer.trigger_manager.evaluate_beat_triggers.assert_not_called()

    def test_check_beat_brightness_calls_trigger_manager(self, renderer_with_audio):
        """Test _check_and_create_beat_brightness_effect calls trigger manager."""
        renderer_with_audio.trigger_manager = Mock()

        renderer_with_audio._check_and_create_beat_brightness_effect(0.0)

        renderer_with_audio.trigger_manager.evaluate_beat_triggers.assert_called_once()


# =============================================================================
# FrameRenderer Tests - Event Effects
# =============================================================================


class TestFrameRendererEventEffects:
    """Test FrameRenderer event effect creation."""

    @pytest.fixture
    def renderer(self):
        """Create FrameRenderer instance for testing."""
        return FrameRenderer(led_ordering=None)

    def test_create_event_effect_none(self, renderer):
        """Test _create_event_effect returns None for 'none' class."""
        config = {"effect_class": "none", "params": {}}
        result = renderer._create_event_effect(config, 0.0)
        assert result is None

    def test_create_event_effect_unknown_class(self, renderer):
        """Test _create_event_effect returns None for unknown class."""
        config = {"effect_class": "UnknownEffect", "params": {}}
        result = renderer._create_event_effect(config, 0.0)
        assert result is None

    @patch("src.consumer.led_effect_transitions.FadeInEffect")
    def test_create_event_effect_fade_in(self, mock_fade_in, renderer):
        """Test _create_event_effect creates FadeInEffect."""
        mock_effect = Mock()
        mock_fade_in.return_value = mock_effect

        config = {
            "effect_class": "FadeInEffect",
            "params": {"duration": 1.5, "curve": "ease-in"},
        }
        result = renderer._create_event_effect(config, 0.0)

        assert result == mock_effect
        mock_fade_in.assert_called_once()

    @patch("src.consumer.led_effect_transitions.FadeOutEffect")
    def test_create_event_effect_fade_out(self, mock_fade_out, renderer):
        """Test _create_event_effect creates FadeOutEffect."""
        mock_effect = Mock()
        mock_fade_out.return_value = mock_effect

        config = {
            "effect_class": "FadeOutEffect",
            "params": {"duration": 2.0},
        }
        result = renderer._create_event_effect(config, 0.0)

        assert result == mock_effect
        mock_fade_out.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
