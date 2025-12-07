"""
Unit tests for LED Transition Processor.

Tests LEDTransitionProcessor class that handles LED transition processing
in the consumer pipeline.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_led_values():
    """Create sample LED values as a mock cupy array."""
    mock_array = MagicMock()
    mock_array.copy.return_value = mock_array
    mock_array.shape = (100, 3)
    return mock_array


@pytest.fixture
def mock_factory():
    """Create mock LED transition factory."""
    mock = MagicMock()
    mock.create_led_transition.return_value = MagicMock()
    mock.get_available_led_transitions.return_value = ["ledfade", "ledrandom", "ledblur"]
    mock.apply_led_transition_to_values.return_value = MagicMock()
    return mock


@pytest.fixture
def valid_metadata():
    """Create valid transition metadata."""
    return {
        "transition_in_type": "ledfade",
        "transition_in_duration": 1.0,
        "transition_out_type": "ledfade",
        "transition_out_duration": 1.0,
        "item_timestamp": 0.5,
        "item_duration": 10.0,
    }


@pytest.fixture
def metadata_no_led_transition():
    """Create metadata with non-LED transition types."""
    return {
        "transition_in_type": "fade",  # Not an LED transition
        "transition_in_duration": 1.0,
        "transition_out_type": "fade",
        "transition_out_duration": 1.0,
        "item_timestamp": 0.5,
        "item_duration": 10.0,
    }


# =============================================================================
# LEDTransitionProcessor Tests
# =============================================================================


class TestLEDTransitionProcessor:
    """Test LEDTransitionProcessor class."""

    def test_initialization(self, mock_factory):
        """Test LEDTransitionProcessor initialization."""
        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            assert processor._frame_count == 0
            assert processor._factory is mock_factory

    def test_process_led_values_no_metadata(self, sample_led_values, mock_factory):
        """Test processing when no transition metadata present."""
        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            # Empty metadata - no transitions
            result = processor.process_led_values(sample_led_values, {})

            # Should return original values unchanged
            assert result is sample_led_values
            assert processor._frame_count == 1

    def test_process_led_values_missing_fields(self, sample_led_values, mock_factory):
        """Test processing when metadata has missing fields."""
        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            # Partial metadata - missing required fields
            partial_metadata = {
                "transition_in_type": "ledfade",
                # Missing other required fields
            }

            result = processor.process_led_values(sample_led_values, partial_metadata)

            # Should return original values
            assert result is sample_led_values

    def test_process_led_values_with_valid_metadata(self, sample_led_values, valid_metadata, mock_factory):
        """Test processing with valid transition metadata."""
        # Configure mock to not apply transitions (not in transition region)
        mock_transition = MagicMock()
        mock_transition.is_in_transition_region.return_value = False
        mock_factory.create_led_transition.return_value = mock_transition

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            result = processor.process_led_values(sample_led_values, valid_metadata)

            # Should have copied the led_values
            sample_led_values.copy.assert_called_once()
            assert processor._frame_count == 1

    def test_process_led_values_applies_transition_in(self, sample_led_values, mock_factory):
        """Test that transition_in is applied when in transition region."""
        # Configure mock to apply transition_in
        mock_transition = MagicMock()
        mock_transition.is_in_transition_region.side_effect = lambda t, d, c, direction: direction == "in"
        mock_factory.create_led_transition.return_value = mock_transition

        # Return the modified values
        modified_values = MagicMock()
        mock_factory.apply_led_transition_to_values.return_value = modified_values

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            # Metadata indicating we're at the start (transition_in region)
            metadata = {
                "transition_in_type": "ledfade",
                "transition_in_duration": 1.0,
                "transition_out_type": "ledfade",
                "transition_out_duration": 1.0,
                "item_timestamp": 0.5,  # Early in item
                "item_duration": 10.0,
            }

            processor.process_led_values(sample_led_values, metadata)

            # Verify apply_led_transition_to_values was called
            mock_factory.apply_led_transition_to_values.assert_called()

    def test_process_led_values_applies_transition_out(self, sample_led_values, mock_factory):
        """Test that transition_out is applied when in transition region."""
        # Configure mock to apply transition_out
        mock_transition = MagicMock()
        mock_transition.is_in_transition_region.side_effect = lambda t, d, c, direction: direction == "out"
        mock_factory.create_led_transition.return_value = mock_transition

        modified_values = MagicMock()
        mock_factory.apply_led_transition_to_values.return_value = modified_values

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            # Metadata indicating we're near the end (transition_out region)
            metadata = {
                "transition_in_type": "ledfade",
                "transition_in_duration": 1.0,
                "transition_out_type": "ledfade",
                "transition_out_duration": 1.0,
                "item_timestamp": 9.5,  # Near end of item
                "item_duration": 10.0,
            }

            processor.process_led_values(sample_led_values, metadata)

            # Verify apply_led_transition_to_values was called
            mock_factory.apply_led_transition_to_values.assert_called()

    def test_process_led_values_non_led_transition(self, sample_led_values, metadata_no_led_transition, mock_factory):
        """Test processing with non-LED transition types."""
        mock_transition = MagicMock()
        mock_factory.create_led_transition.return_value = mock_transition

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            result = processor.process_led_values(sample_led_values, metadata_no_led_transition)

            # Verify apply was NOT called (non-LED transitions are skipped)
            mock_factory.apply_led_transition_to_values.assert_not_called()

    def test_process_led_values_handles_exception(self, sample_led_values, valid_metadata, mock_factory):
        """Test that exceptions are handled gracefully."""
        # Configure factory to raise exception
        mock_factory.create_led_transition.side_effect = Exception("Test error")

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            # Should not raise, should return original values
            result = processor.process_led_values(sample_led_values, valid_metadata)

            # Returns a copy (from the copy before exception)
            assert result is not None

    def test_get_statistics(self, mock_factory):
        """Test get_statistics returns correct values."""
        mock_factory.get_available_led_transitions.return_value = [
            "ledfade",
            "ledrandom",
        ]

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()
            processor._frame_count = 100

            stats = processor.get_statistics()

            assert stats["frames_processed"] == 100
            assert stats["available_led_transitions"] == ["ledfade", "ledrandom"]


# =============================================================================
# Extract Transition Context Tests
# =============================================================================


class TestExtractTransitionContext:
    """Test _extract_transition_context method."""

    def test_extract_with_valid_metadata(self, valid_metadata, mock_factory):
        """Test extraction with complete valid metadata."""
        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()
            context = processor._extract_transition_context(valid_metadata)

            assert context is not None
            assert context["timestamp"] == 0.5
            assert context["item_duration"] == 10.0
            assert context["transition_in"]["type"] == "ledfade"
            assert context["transition_out"]["type"] == "ledfade"

    def test_extract_with_missing_field(self, mock_factory):
        """Test extraction returns None when fields are missing."""
        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            # Missing item_timestamp
            incomplete_metadata = {
                "transition_in_type": "ledfade",
                "transition_in_duration": 1.0,
                "transition_out_type": "ledfade",
                "transition_out_duration": 1.0,
                "item_duration": 10.0,
                # Missing item_timestamp
            }

            context = processor._extract_transition_context(incomplete_metadata)

            assert context is None


# =============================================================================
# Should Apply Transition Tests
# =============================================================================


class TestShouldApplyTransition:
    """Test _should_apply_led_transition_in/out methods."""

    def test_should_apply_led_transition_in_true(self, mock_factory):
        """Test returns True when in transition_in region."""
        mock_transition = MagicMock()
        mock_transition.is_in_transition_region.return_value = True
        mock_factory.create_led_transition.return_value = mock_transition

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            context = {
                "timestamp": 0.5,
                "item_duration": 10.0,
                "transition_in": {"type": "ledfade", "parameters": {"duration": 1.0}},
                "transition_out": {"type": "ledfade", "parameters": {"duration": 1.0}},
            }

            result = processor._should_apply_led_transition_in(context)

            assert result is True

    def test_should_apply_led_transition_in_non_led_type(self, mock_factory):
        """Test returns False for non-LED transition type."""
        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            context = {
                "timestamp": 0.5,
                "item_duration": 10.0,
                "transition_in": {"type": "fade", "parameters": {}},  # Not LED type
                "transition_out": {"type": "fade", "parameters": {}},
            }

            result = processor._should_apply_led_transition_in(context)

            assert result is False

    def test_should_apply_led_transition_out_true(self, mock_factory):
        """Test returns True when in transition_out region."""
        mock_transition = MagicMock()
        mock_transition.is_in_transition_region.return_value = True
        mock_factory.create_led_transition.return_value = mock_transition

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            context = {
                "timestamp": 9.5,
                "item_duration": 10.0,
                "transition_in": {"type": "ledfade", "parameters": {"duration": 1.0}},
                "transition_out": {"type": "ledfade", "parameters": {"duration": 1.0}},
            }

            result = processor._should_apply_led_transition_out(context)

            assert result is True

    def test_should_apply_handles_factory_returning_none(self, mock_factory):
        """Test handles factory returning None for unknown transition."""
        mock_factory.create_led_transition.return_value = None

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            context = {
                "timestamp": 0.5,
                "item_duration": 10.0,
                "transition_in": {"type": "ledunknown", "parameters": {}},
                "transition_out": {"type": "ledunknown", "parameters": {}},
            }

            result = processor._should_apply_led_transition_in(context)

            assert result is False


# =============================================================================
# Apply LED Transition Tests
# =============================================================================


class TestApplyLEDTransition:
    """Test _apply_led_transition method."""

    def test_apply_led_transition_in(self, sample_led_values, mock_factory):
        """Test applying LED transition in direction."""
        expected_result = MagicMock()
        mock_factory.apply_led_transition_to_values.return_value = expected_result

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            context = {
                "timestamp": 0.5,
                "item_duration": 10.0,
                "transition_in": {"type": "ledfade", "parameters": {"duration": 1.0}},
                "transition_out": {"type": "ledfade", "parameters": {"duration": 1.0}},
            }

            result = processor._apply_led_transition(sample_led_values, context, "in")

            assert result is expected_result
            mock_factory.apply_led_transition_to_values.assert_called_once()

    def test_apply_led_transition_out(self, sample_led_values, mock_factory):
        """Test applying LED transition out direction."""
        expected_result = MagicMock()
        mock_factory.apply_led_transition_to_values.return_value = expected_result

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            context = {
                "timestamp": 9.5,
                "item_duration": 10.0,
                "transition_in": {"type": "ledfade", "parameters": {"duration": 1.0}},
                "transition_out": {"type": "ledfade", "parameters": {"duration": 1.0}},
            }

            result = processor._apply_led_transition(sample_led_values, context, "out")

            assert result is expected_result

    def test_apply_led_transition_handles_exception(self, sample_led_values, mock_factory):
        """Test that exceptions return original values."""
        mock_factory.apply_led_transition_to_values.side_effect = Exception("Test error")

        with patch(
            "src.consumer.led_transition_processor.get_led_transition_factory",
            return_value=mock_factory,
        ):
            from src.consumer.led_transition_processor import LEDTransitionProcessor

            processor = LEDTransitionProcessor()

            context = {
                "timestamp": 0.5,
                "item_duration": 10.0,
                "transition_in": {"type": "ledfade", "parameters": {"duration": 1.0}},
                "transition_out": {"type": "ledfade", "parameters": {"duration": 1.0}},
            }

            result = processor._apply_led_transition(sample_led_values, context, "in")

            # Should return original values on error
            assert result is sample_led_values
