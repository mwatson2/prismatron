"""
Pattern Loader Utilities.

This module provides utility functions for loading diffusion patterns and
extracting LED ordering information for the frame renderer.

For quick inspection of pattern files, use the inspection tool:
    python tools/inspect_patterns.py pattern_file.npz
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_led_ordering_from_pattern(pattern_file_path: str) -> Optional[np.ndarray]:
    """
    Load and validate LED ordering array from a diffusion pattern file.

    Args:
        pattern_file_path: Path to the .npz pattern file

    Returns:
        Validated LED ordering array mapping spatial indices to physical LED IDs,
        or None if not available or invalid
    """
    try:
        data = np.load(pattern_file_path, allow_pickle=True)

        if "led_ordering" in data:
            led_ordering = data["led_ordering"]

            # Validate the LED ordering
            is_valid, error_msg = validate_led_ordering(led_ordering)
            if not is_valid:
                logger.error(f"Invalid LED ordering in {pattern_file_path}: {error_msg}")
                return None

            logger.info(f"Loaded and validated LED ordering: {len(led_ordering)} LEDs from {pattern_file_path}")
            return led_ordering
        else:
            logger.warning(f"No led_ordering found in {pattern_file_path}")
            return None

    except Exception as e:
        logger.error(f"Failed to load LED ordering from {pattern_file_path}: {e}")
        return None


def create_frame_renderer_with_pattern(
    pattern_file_path: str,
    first_frame_delay_ms: float = 100.0,
    timing_tolerance_ms: float = 5.0,
    late_frame_log_threshold_ms: float = 50.0,
    control_state=None,
    audio_beat_analyzer=None,
    enable_position_shifting: bool = False,
    max_shift_distance: int = 3,
    shift_direction: str = "alternating",
):
    """
    Create a frame renderer with LED ordering loaded from a pattern file.

    Args:
        pattern_file_path: Path to the .npz pattern file
        first_frame_delay_ms: Default delay for first frame buffering
        timing_tolerance_ms: Acceptable timing deviation
        late_frame_log_threshold_ms: Log late frames above this threshold
        control_state: ControlState instance for audio reactive settings
        audio_beat_analyzer: AudioBeatAnalyzer instance for beat state access
        enable_position_shifting: Enable audio-reactive position shifting effects
        max_shift_distance: Maximum LED positions to shift (3-4 typical)
        shift_direction: Shift direction ("left", "right", "alternating")

    Returns:
        FrameRenderer instance configured with LED ordering from pattern
    """
    try:
        from ..consumer.frame_renderer import FrameRenderer
    except ImportError:
        # Handle case when running as standalone script
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.consumer.frame_renderer import FrameRenderer

    # Load LED ordering from pattern
    led_ordering = load_led_ordering_from_pattern(pattern_file_path)

    # Create frame renderer
    renderer = FrameRenderer(
        led_ordering=led_ordering,
        first_frame_delay_ms=first_frame_delay_ms,
        timing_tolerance_ms=timing_tolerance_ms,
        late_frame_log_threshold_ms=late_frame_log_threshold_ms,
        control_state=control_state,
        audio_beat_analyzer=audio_beat_analyzer,
        enable_position_shifting=enable_position_shifting,
        max_shift_distance=max_shift_distance,
        shift_direction=shift_direction,
    )

    if led_ordering is not None:
        logger.info(f"Frame renderer configured with LED ordering for {len(led_ordering)} LEDs")
    else:
        logger.warning("Frame renderer created without LED ordering - spatial conversion will be skipped")

    return renderer


def load_pattern_info(pattern_file_path: str) -> dict:
    """
    Load basic information about a pattern file.

    Args:
        pattern_file_path: Path to the .npz pattern file

    Returns:
        Dictionary with pattern information
    """
    try:
        data = np.load(pattern_file_path, allow_pickle=True)

        info = {
            "file_path": pattern_file_path,
            "file_exists": True,
            "keys": list(data.keys()),
            "has_led_ordering": "led_ordering" in data,
            "has_mixed_tensor": "mixed_tensor" in data,
            "has_dia_matrix": "dia_matrix" in data,
            "has_ata_inverse": "ata_inverse" in data,
        }

        # Extract metadata if available
        if "metadata" in data:
            metadata = data["metadata"].item() if hasattr(data["metadata"], "item") else data["metadata"]
            info.update(
                {
                    "led_count": metadata.get("led_count"),
                    "frame_width": metadata.get("frame_width"),
                    "frame_height": metadata.get("frame_height"),
                    "block_size": metadata.get("block_size"),
                    "pattern_type": metadata.get("pattern_type"),
                    "sparsity_threshold": metadata.get("sparsity_threshold"),
                }
            )

        # Extract LED ordering info if available
        if "led_ordering" in data:
            led_ordering = data["led_ordering"]
            info.update(
                {
                    "led_ordering_shape": led_ordering.shape,
                    "led_ordering_dtype": str(led_ordering.dtype),
                }
            )

        return info

    except Exception as e:
        return {
            "file_path": pattern_file_path,
            "file_exists": False,
            "error": str(e),
        }


def validate_led_ordering(led_ordering: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that LED ordering is a proper permutation.

    Args:
        led_ordering: LED ordering array to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if it's a 1D array
        if led_ordering.ndim != 1:
            return False, f"LED ordering must be 1D, got {led_ordering.ndim}D"

        # Check if it contains unique values
        unique_vals = np.unique(led_ordering)
        if len(unique_vals) != len(led_ordering):
            return False, f"LED ordering contains duplicates: {len(led_ordering)} values, {len(unique_vals)} unique"

        # Check if it's a permutation of 0 to len-1
        expected_range = np.arange(len(led_ordering))
        if not np.array_equal(np.sort(unique_vals), expected_range):
            return False, f"LED ordering is not a permutation of 0 to {len(led_ordering)-1}"

        return True, "Valid LED ordering"

    except Exception as e:
        return False, f"Error validating LED ordering: {e}"


# Example usage and testing
if __name__ == "__main__":
    import argparse
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Test pattern loading utilities")
    parser.add_argument("pattern_file", help="Path to diffusion pattern file")
    parser.add_argument("--test-renderer", action="store_true", help="Test frame renderer creation")

    args = parser.parse_args()

    # Test pattern info loading
    info = load_pattern_info(args.pattern_file)
    print("Pattern Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test LED ordering loading
    led_ordering = load_led_ordering_from_pattern(args.pattern_file)
    if led_ordering is not None:
        is_valid, message = validate_led_ordering(led_ordering)
        print(f"\nLED Ordering Validation: {message}")

        if args.test_renderer:
            # Test frame renderer creation
            renderer = create_frame_renderer_with_pattern(args.pattern_file)
            print(f"Frame renderer created: {renderer}")
    else:
        print("No LED ordering found in pattern file")
