#!/usr/bin/env python3
"""
Analyze timing in the simulation to understand rapid oscillations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def analyze_timing():
    """Analyze expected timing in the system."""
    print("🔍 Timing Analysis")
    print("=" * 60)

    producer_fps = 24.0
    renderer_fps = 15.0

    producer_interval = 1.0 / producer_fps  # Time between productions
    optimization_time = 1.0 / renderer_fps  # Time to optimize
    render_interval = 1.0 / 30.0  # Target render interval (30fps)

    print(f"Producer interval: {producer_interval*1000:.1f}ms (every {producer_interval:.4f}s)")
    print(f"Optimization time: {optimization_time*1000:.1f}ms (takes {optimization_time:.4f}s)")
    print(f"Render interval: {render_interval*1000:.1f}ms (every {render_interval:.4f}s)")
    print()

    # Calculate steady state behavior
    frames_produced_per_second = producer_fps
    frames_optimized_per_second = renderer_fps  # Can only optimize 15fps
    frames_rendered_per_second = 30.0  # Target render rate

    print("Steady State Analysis:")
    print(f"Frames produced/sec: {frames_produced_per_second}")
    print(f"Frames optimized/sec: {frames_optimized_per_second} (bottleneck)")
    print(f"Frames rendered/sec: up to {frames_rendered_per_second}")
    print()

    # Net flow
    net_flow = frames_optimized_per_second - frames_rendered_per_second
    print(f"Net flow into buffer: {net_flow} frames/sec")
    print("This means buffer should drain!")
    print()

    # But wait - we're rendering at frame timestamps, not 30fps!
    print("IMPORTANT: Renderer doesn't actually render at 30fps!")
    print("It renders frames at their target timestamps (1/30s apart)")
    print("But frames only arrive every 1/15s from optimization")
    print()

    # Real timing
    print("Real timing in steady state:")
    print("- Frame produced every 41.7ms")
    print("- Frame optimized 66.7ms later")
    print("- Frame rendered immediately (if buffer empty)")
    print("- Total pipeline: 108.3ms per frame")
    print()
    print("This means buffer stays nearly empty (0-1 frames)!")
    print("Oscillations are frame arrivals, not control oscillations!")


if __name__ == "__main__":
    analyze_timing()
