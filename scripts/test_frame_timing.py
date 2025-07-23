#!/usr/bin/env python3
"""
Test script for frame timing system.

This script creates sample timing data and tests the logging and visualization functionality.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.frame_timing import FrameTimingData, FrameTimingLogger, get_next_frame_index, reset_frame_counter


def create_sample_timing_data(num_frames: int = 50) -> list[FrameTimingData]:
    """
    Create sample timing data for testing.

    Args:
        num_frames: Number of sample frames to create

    Returns:
        List of FrameTimingData objects
    """
    reset_frame_counter()
    timing_data_list = []

    # Simulate realistic timing
    base_time = time.time()
    plugin_time = 0.0

    for i in range(num_frames):
        timing_data = FrameTimingData(
            frame_index=get_next_frame_index(),
            plugin_timestamp=plugin_time,
            producer_timestamp=plugin_time + 1000.0,  # Producer adds offset
            item_duration=10.0,  # 10-second item
        )

        # Simulate pipeline timing with realistic delays
        current_time = base_time + plugin_time

        # Write to buffer
        timing_data.write_to_buffer_time = current_time + 0.001  # 1ms write delay

        # Read from buffer (some processing delay)
        timing_data.read_from_buffer_time = timing_data.write_to_buffer_time + 0.005  # 5ms

        # Write to LED buffer (optimization takes time)
        timing_data.write_to_led_buffer_time = timing_data.read_from_buffer_time + 0.050  # 50ms optimization

        # Read from LED buffer (quick)
        timing_data.read_from_led_buffer_time = timing_data.write_to_led_buffer_time + 0.001  # 1ms

        # Render (hardware communication)
        timing_data.render_time = timing_data.read_from_led_buffer_time + 0.010  # 10ms render

        timing_data_list.append(timing_data)

        # Advance plugin time (simulate 30 FPS)
        plugin_time += 1.0 / 30.0

    return timing_data_list


def test_timing_logger():
    """Test the timing logger functionality."""
    print("Testing frame timing logger...")

    # Create output directory
    output_dir = Path("timing_test_output")
    output_dir.mkdir(exist_ok=True)

    csv_file = output_dir / "test_timing.csv"

    # Create sample data
    timing_data_list = create_sample_timing_data(50)

    # Test logging
    with FrameTimingLogger(str(csv_file)) as logger:
        for timing_data in timing_data_list:
            logger.log_frame(timing_data)

    # Verify file was created and has correct content
    if csv_file.exists():
        print(f"✓ CSV file created: {csv_file}")

        # Read and verify content
        with open(csv_file, "r") as f:
            lines = f.readlines()
            print(f"✓ CSV has {len(lines)} lines (including header)")

        if len(lines) == len(timing_data_list) + 1:  # +1 for header
            print("✓ CSV has correct number of data rows")
        else:
            print(f"✗ Expected {len(timing_data_list) + 1} lines, got {len(lines)}")

    else:
        print("✗ CSV file was not created")

    return csv_file


def test_visualization(csv_file: Path):
    """Test the visualization functionality."""
    print("\nTesting visualization system...")

    try:
        # Import visualization module
        visualizer_path = Path(__file__).parent / "visualize_frame_timing.py"
        sys.path.insert(0, str(visualizer_path.parent))

        # Create visualizer and test statistics
        from visualize_frame_timing import FrameTimingVisualizer

        visualizer = FrameTimingVisualizer(str(csv_file))

        print("✓ Timing data loaded successfully")

        # Test statistics
        visualizer.print_statistics()
        print("✓ Statistics calculated successfully")

        # Test visualizations (save to files, don't display)
        output_dir = csv_file.parent
        timeline_path = output_dir / "test_timeline.png"
        latency_path = output_dir / "test_latency.png"

        visualizer.create_timeline_visualization(str(timeline_path), max_frames=50)
        print(f"✓ Timeline visualization saved: {timeline_path}")

        visualizer.create_latency_analysis(str(latency_path))
        print(f"✓ Latency analysis saved: {latency_path}")

        return True

    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def main():
    """Main test function."""
    print("=== Frame Timing System Test ===\n")

    try:
        # Test CSV logging
        csv_file = test_timing_logger()

        # Test visualization
        visualization_success = test_visualization(csv_file)

        print(f"\n=== Test Results ===")
        print(f"CSV Logging: ✓ PASS")
        print(f"Visualization: {'✓ PASS' if visualization_success else '✗ FAIL'}")

        if visualization_success:
            print(f"\nAll tests passed! Check the 'timing_test_output' directory for results.")
            return 0
        else:
            print(f"\nSome tests failed. Check error messages above.")
            return 1

    except Exception as e:
        print(f"Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
