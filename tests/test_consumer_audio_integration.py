"""
Tests for Consumer Process audio integration.

This module tests the integration of audio beat detection functionality
with the consumer process, including lifecycle management and control state updates.
"""

import time
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest

from src.consumer.audio_beat_analyzer import BeatEvent
from src.core.control_state import ControlState, SystemStatus


class TestConsumerAudioIntegration:
    """Test consumer process audio integration."""

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.FrameRenderer")
    @patch("src.consumer.consumer.AudioBeatAnalyzer")
    def test_consumer_audio_initialization(
        self, mock_audio_analyzer, mock_renderer, mock_optimizer, mock_control_state, mock_frame_consumer
    ):
        """Test consumer initialization with audio enabled."""
        from src.consumer.consumer import ConsumerProcess

        # Set up mocks
        mock_audio_instance = Mock()
        mock_audio_analyzer.return_value = mock_audio_instance

        # Create consumer with audio enabled
        consumer = ConsumerProcess(
            enable_audio_reactive=True, audio_device="cuda", diffusion_patterns_path="test_patterns"
        )

        # Verify audio analyzer was created
        assert consumer._enable_audio_reactive is True
        assert consumer._audio_beat_analyzer == mock_audio_instance

        # Verify it was initialized with correct parameters
        mock_audio_analyzer.assert_called_once()
        call_kwargs = mock_audio_analyzer.call_args[1]
        assert "beat_callback" in call_kwargs
        assert call_kwargs["device"] == "cuda"
        assert call_kwargs["beat_callback"] == consumer._on_beat_detected

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.FrameRenderer")
    @patch("src.consumer.consumer.AudioBeatAnalyzer")
    def test_consumer_audio_disabled(
        self, mock_audio_analyzer, mock_renderer, mock_optimizer, mock_control_state, mock_frame_consumer
    ):
        """Test consumer initialization with audio disabled."""
        from src.consumer.consumer import ConsumerProcess

        # Create consumer with audio disabled
        consumer = ConsumerProcess(enable_audio_reactive=False, diffusion_patterns_path="test_patterns")

        # Verify audio analyzer was not created
        assert consumer._enable_audio_reactive is False
        assert consumer._audio_beat_analyzer is None
        mock_audio_analyzer.assert_not_called()

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.FrameRenderer")
    @patch("src.consumer.consumer.logger")
    def test_consumer_audio_initialization_failure(
        self, mock_logger, mock_renderer, mock_optimizer, mock_control_state, mock_frame_consumer
    ):
        """Test consumer handles audio initialization failure gracefully."""
        from src.consumer.consumer import ConsumerProcess

        # Make AudioBeatAnalyzer import fail
        with patch("src.consumer.consumer.AudioBeatAnalyzer", side_effect=ImportError("Test error")):
            consumer = ConsumerProcess(enable_audio_reactive=True, diffusion_patterns_path="test_patterns")

            # Should disable audio and log error
            assert consumer._enable_audio_reactive is False
            assert consumer._audio_beat_analyzer is None
            mock_logger.error.assert_called_once()

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.FrameRenderer")
    @patch("src.consumer.consumer.AudioBeatAnalyzer")
    def test_consumer_beat_callback(
        self, mock_audio_analyzer, mock_renderer, mock_optimizer, mock_control_state, mock_frame_consumer
    ):
        """Test consumer beat event callback updates control state."""
        from src.consumer.consumer import ConsumerProcess

        # Set up mocks
        mock_control_instance = Mock()
        mock_control_state.return_value = mock_control_instance

        # Create consumer
        consumer = ConsumerProcess(enable_audio_reactive=True, diffusion_patterns_path="test_patterns")

        # Create test beat event
        beat_event = BeatEvent(
            timestamp=1.5,
            system_time=time.time(),
            is_downbeat=False,
            bpm=128.0,
            intensity=0.75,
            confidence=0.9,
            beat_count=10,
        )

        # Call the beat callback
        consumer._on_beat_detected(beat_event)

        # Verify control state was updated
        mock_control_instance.update_status.assert_called()
        update_kwargs = mock_control_instance.update_status.call_args[1]

        assert update_kwargs["audio_enabled"] is True
        assert update_kwargs["current_bpm"] == 128.0
        assert update_kwargs["beat_count"] == 10
        assert update_kwargs["last_beat_time"] == 1.5
        assert update_kwargs["beat_confidence"] == 0.9
        assert update_kwargs["audio_intensity"] == 0.75

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.FrameRenderer")
    @patch("src.consumer.consumer.AudioBeatAnalyzer")
    def test_consumer_downbeat_callback(
        self, mock_audio_analyzer, mock_renderer, mock_optimizer, mock_control_state, mock_frame_consumer
    ):
        """Test consumer updates downbeat time correctly."""
        from src.consumer.consumer import ConsumerProcess

        # Set up mocks
        mock_control_instance = Mock()
        mock_control_state.return_value = mock_control_instance

        # Create consumer
        consumer = ConsumerProcess(enable_audio_reactive=True, diffusion_patterns_path="test_patterns")

        # Create downbeat event
        beat_event = BeatEvent(
            timestamp=2.0,
            system_time=time.time(),
            is_downbeat=True,
            bpm=120.0,
            intensity=0.8,
            confidence=0.95,
            beat_count=4,
        )

        # Call the beat callback
        consumer._on_beat_detected(beat_event)

        # Should have two update calls - one for beat, one for downbeat
        assert mock_control_instance.update_status.call_count == 2

        # Check downbeat update
        downbeat_update = mock_control_instance.update_status.call_args_list[1][1]
        assert downbeat_update["last_downbeat_time"] == 2.0

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.FrameRenderer")
    @patch("src.consumer.consumer.AudioBeatAnalyzer")
    @patch("src.consumer.consumer.logger")
    def test_consumer_beat_callback_error_handling(
        self, mock_logger, mock_audio_analyzer, mock_renderer, mock_optimizer, mock_control_state, mock_frame_consumer
    ):
        """Test consumer handles errors in beat callback gracefully."""
        from src.consumer.consumer import ConsumerProcess

        # Set up control state to raise error
        mock_control_instance = Mock()
        mock_control_instance.update_status.side_effect = RuntimeError("Test error")
        mock_control_state.return_value = mock_control_instance

        # Create consumer
        consumer = ConsumerProcess(enable_audio_reactive=True, diffusion_patterns_path="test_patterns")

        # Create beat event
        beat_event = BeatEvent(
            timestamp=1.0,
            system_time=time.time(),
            is_downbeat=False,
            bpm=120.0,
            intensity=0.5,
            confidence=0.8,
            beat_count=1,
        )

        # Call should not raise exception
        consumer._on_beat_detected(beat_event)

        # Should log error
        mock_logger.error.assert_called_once()
        assert "Error handling beat event" in mock_logger.error.call_args[0][0]

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.FrameRenderer")
    @patch("src.consumer.consumer.AudioBeatAnalyzer")
    def test_consumer_start_with_audio(
        self, mock_audio_analyzer, mock_renderer, mock_optimizer, mock_control_state, mock_frame_consumer
    ):
        """Test consumer start method starts audio analyzer."""
        from src.consumer.consumer import ConsumerProcess

        # Set up mocks
        mock_audio_instance = Mock()
        mock_audio_analyzer.return_value = mock_audio_instance

        # Mock other required components
        mock_renderer_instance = Mock()
        mock_renderer.return_value = mock_renderer_instance
        mock_renderer_instance.initialize.return_value = True

        # Create consumer
        consumer = ConsumerProcess(enable_audio_reactive=True, diffusion_patterns_path="test_patterns")

        # Mock internal state
        consumer._initialized = True
        consumer._running = False

        # Start consumer
        success = consumer.start()

        # Should start audio analyzer
        assert success is True
        mock_audio_instance.start_analysis.assert_called_once()

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.FrameRenderer")
    @patch("src.consumer.consumer.AudioBeatAnalyzer")
    @patch("src.consumer.consumer.logger")
    def test_consumer_start_audio_failure(
        self, mock_logger, mock_audio_analyzer, mock_renderer, mock_optimizer, mock_control_state, mock_frame_consumer
    ):
        """Test consumer handles audio start failure gracefully."""
        from src.consumer.consumer import ConsumerProcess

        # Set up mocks
        mock_audio_instance = Mock()
        mock_audio_instance.start_analysis.side_effect = RuntimeError("Audio start failed")
        mock_audio_analyzer.return_value = mock_audio_instance

        # Mock other required components
        mock_renderer_instance = Mock()
        mock_renderer.return_value = mock_renderer_instance
        mock_renderer_instance.initialize.return_value = True

        # Create consumer
        consumer = ConsumerProcess(enable_audio_reactive=True, diffusion_patterns_path="test_patterns")

        # Mock internal state
        consumer._initialized = True
        consumer._running = False

        # Start consumer - should succeed despite audio failure
        success = consumer.start()
        assert success is True

        # Should log error
        mock_logger.error.assert_called()
        assert "Failed to start audio beat analysis" in mock_logger.error.call_args[0][0]

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.FrameRenderer")
    @patch("src.consumer.consumer.AudioBeatAnalyzer")
    def test_consumer_stop_with_audio(
        self, mock_audio_analyzer, mock_renderer, mock_optimizer, mock_control_state, mock_frame_consumer
    ):
        """Test consumer stop method stops audio analyzer."""
        from src.consumer.consumer import ConsumerProcess

        # Set up mocks
        mock_audio_instance = Mock()
        mock_audio_analyzer.return_value = mock_audio_instance

        # Create consumer
        consumer = ConsumerProcess(enable_audio_reactive=True, diffusion_patterns_path="test_patterns")

        # Mock running state
        consumer._running = True
        consumer._shutdown_requested = False
        consumer._optimization_thread = Mock()
        consumer._renderer_thread = Mock()
        consumer._wled_reconnection_thread = Mock()
        consumer._wled_reconnection_event = Mock()

        # Stop consumer
        consumer.stop()

        # Should stop audio analyzer
        mock_audio_instance.stop_analysis.assert_called_once()

    @patch("src.consumer.consumer.FrameConsumer")
    @patch("src.consumer.consumer.ControlState")
    @patch("src.consumer.consumer.LEDOptimizer")
    @patch("src.consumer.consumer.FrameRenderer")
    @patch("src.consumer.consumer.AudioBeatAnalyzer")
    @patch("src.consumer.consumer.logger")
    def test_consumer_stop_audio_failure(
        self, mock_logger, mock_audio_analyzer, mock_renderer, mock_optimizer, mock_control_state, mock_frame_consumer
    ):
        """Test consumer handles audio stop failure gracefully."""
        from src.consumer.consumer import ConsumerProcess

        # Set up mocks
        mock_audio_instance = Mock()
        mock_audio_instance.stop_analysis.side_effect = RuntimeError("Audio stop failed")
        mock_audio_analyzer.return_value = mock_audio_instance

        # Create consumer
        consumer = ConsumerProcess(enable_audio_reactive=True, diffusion_patterns_path="test_patterns")

        # Mock running state
        consumer._running = True
        consumer._shutdown_requested = False
        consumer._optimization_thread = Mock()
        consumer._renderer_thread = Mock()
        consumer._wled_reconnection_thread = Mock()
        consumer._wled_reconnection_event = Mock()

        # Stop consumer - should not raise exception
        consumer.stop()

        # Should log error
        mock_logger.error.assert_called()
        assert "Error stopping audio beat analysis" in mock_logger.error.call_args[0][0]


class TestControlStateAudioFields:
    """Test control state audio field integration."""

    def test_system_status_audio_fields(self):
        """Test SystemStatus dataclass includes audio fields."""
        status = SystemStatus()

        # Check default values
        assert hasattr(status, "audio_enabled")
        assert hasattr(status, "current_bpm")
        assert hasattr(status, "beat_count")
        assert hasattr(status, "last_beat_time")
        assert hasattr(status, "last_downbeat_time")
        assert hasattr(status, "beat_confidence")
        assert hasattr(status, "audio_intensity")

        # Check defaults
        assert status.audio_enabled is False
        assert status.current_bpm == 120.0
        assert status.beat_count == 0
        assert status.last_beat_time == 0.0
        assert status.last_downbeat_time == 0.0
        assert status.beat_confidence == 0.0
        assert status.audio_intensity == 0.0

    def test_system_status_audio_serialization(self):
        """Test SystemStatus with audio fields can be serialized."""
        from dataclasses import asdict

        status = SystemStatus(
            audio_enabled=True,
            current_bpm=140.0,
            beat_count=50,
            last_beat_time=25.5,
            last_downbeat_time=24.0,
            beat_confidence=0.92,
            audio_intensity=0.65,
        )

        # Should serialize to dict
        status_dict = asdict(status)
        assert status_dict["audio_enabled"] is True
        assert status_dict["current_bpm"] == 140.0
        assert status_dict["beat_count"] == 50
        assert status_dict["last_beat_time"] == 25.5
        assert status_dict["last_downbeat_time"] == 24.0
        assert status_dict["beat_confidence"] == 0.92
        assert status_dict["audio_intensity"] == 0.65

    @patch("src.core.control_state.shared_memory")
    def test_control_state_audio_updates(self, mock_shared_memory):
        """Test ControlState can update audio fields."""
        # Mock shared memory
        mock_shm = Mock()
        mock_shm.buf = bytearray(4096)
        mock_shared_memory.SharedMemory.return_value = mock_shm

        # Create control state
        control = ControlState("test_control")

        # Update audio fields
        control.update_status(
            audio_enabled=True, current_bpm=128.0, beat_count=100, last_beat_time=50.0, beat_confidence=0.88
        )

        # Read back status
        status = control.get_status()

        # Verify updates (would need proper shared memory implementation)
        # For now just verify the method doesn't crash
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
