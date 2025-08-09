#!/usr/bin/env python3
"""
Analyze memory access patterns for single vs batch operations.
"""

import numpy as np


def analyze_memory_patterns():
    """Calculate theoretical memory access for different approaches."""

    # Configuration
    n_leds = 3008
    n_channels = 3
    n_frames = 8
    block_size = 64
    frame_height = 800
    frame_width = 480

    # Data sizes
    bytes_per_float = 4
    bytes_per_uint8 = 1

    # LED pattern size (assuming uint8 storage)
    led_pattern_size = block_size * block_size * bytes_per_uint8  # 64x64 = 4KB per pattern
    total_patterns_size = n_leds * n_channels * led_pattern_size  # 3008 * 3 * 4KB = 36MB

    # Frame data size (uint8 for images)
    frame_size = frame_height * frame_width * n_channels * bytes_per_uint8  # 800x480x3 = 1.15MB
    total_frames_size = n_frames * frame_size  # 8 * 1.15MB = 9.2MB

    print("=" * 80)
    print("MEMORY ACCESS ANALYSIS: 3008 LEDs, 8 frames")
    print("=" * 80)

    print("\nData Sizes:")
    print(f"  Single LED pattern: {led_pattern_size / 1024:.1f} KB")
    print(f"  All LED patterns: {total_patterns_size / (1024*1024):.1f} MB")
    print(f"  Single frame: {frame_size / (1024*1024):.2f} MB")
    print(f"  All frames: {total_frames_size / (1024*1024):.1f} MB")

    print("\n" + "-" * 80)
    print("SINGLE-FRAME APPROACH (8 separate kernel calls):")
    print("-" * 80)

    # Single-frame: Each kernel call processes 1 frame against all LEDs
    single_led_reads = n_frames * total_patterns_size  # Read all patterns for each frame
    single_frame_reads = n_frames * frame_size  # Each frame read once
    single_total_reads = single_led_reads + single_frame_reads

    print(
        f"  LED pattern reads: {n_frames} frames × {total_patterns_size/(1024*1024):.1f} MB = {single_led_reads/(1024*1024):.1f} MB"
    )
    print(f"  Frame reads: {n_frames} × {frame_size/(1024*1024):.2f} MB = {single_frame_reads/(1024*1024):.1f} MB")
    print(f"  Total memory reads: {single_total_reads/(1024*1024):.1f} MB")

    # Cache considerations
    print("\n  Cache behavior:")
    print(f"    - L2 cache (6MB on A100): Can't hold all {total_patterns_size/(1024*1024):.1f} MB patterns")
    print(f"    - Patterns evicted between frames → high DRAM traffic")
    print(f"    - Each frame region (64x64) accessed by multiple LEDs → good locality")

    print("\n" + "-" * 80)
    print("BATCH APPROACH (1 kernel call for 8 frames):")
    print("-" * 80)

    # Batch: Each LED pattern loaded once, all frames loaded once
    batch_led_reads = total_patterns_size  # Each pattern read once
    batch_frame_reads = total_frames_size  # All frames read once
    batch_total_reads = batch_led_reads + batch_frame_reads

    print(f"  LED pattern reads: {total_patterns_size/(1024*1024):.1f} MB (once)")
    print(f"  Frame reads: {total_frames_size/(1024*1024):.1f} MB (once)")
    print(f"  Total memory reads: {batch_total_reads/(1024*1024):.1f} MB")

    print("\n  Cache behavior:")
    print(f"    - Each block loads 1 LED pattern (4KB) to shared memory")
    print(f"    - Pattern reused for all {n_frames} frames")
    print(f"    - Frame regions accessed multiple times by different blocks")

    print("\n" + "-" * 80)
    print("THEORETICAL ADVANTAGE:")
    print("-" * 80)

    reduction = single_total_reads / batch_total_reads
    print(f"  Memory read reduction: {reduction:.1f}x")
    print(f"  Single approach reads: {single_total_reads/(1024*1024):.1f} MB")
    print(f"  Batch approach reads: {batch_total_reads/(1024*1024):.1f} MB")
    print(f"  Savings: {(single_total_reads - batch_total_reads)/(1024*1024):.1f} MB")

    print("\n" + "=" * 80)
    print("WHY ISN'T BATCH FASTER?")
    print("=" * 80)

    print("\n1. KERNEL LAUNCH OVERHEAD:")
    grid_size_single = n_leds * n_channels  # Per frame
    grid_size_batch = n_leds * n_channels * n_frames  # All frames
    print(f"   Single: {grid_size_single:,} blocks × {n_frames} launches = {grid_size_single * n_frames:,} total")
    print(f"   Batch (original): {grid_size_batch:,} blocks × 1 launch")
    print(f"   Batch (V2): {n_leds * n_channels:,} blocks × 1 launch")

    print("\n2. OCCUPANCY ISSUES:")
    print(f"   Single: 32 threads/block → poor occupancy")
    print(f"   Batch V2: 256 threads/block → better occupancy")
    print(f"   But: {n_leds * n_channels:,} blocks still causes scheduling pressure")

    print("\n3. MEMORY ACCESS PATTERNS:")
    print("   Issue: Frame data not aligned for coalesced access")
    print("   - LED positions are random within frame")
    print("   - Multiple blocks access overlapping frame regions")
    print("   - No spatial ordering enforced in kernel")

    print("\n4. SHARED MEMORY LIMITATIONS:")
    print(f"   - Can only cache 1 LED pattern (4KB) per block")
    print(f"   - Can't cache frame data (too large)")
    print(f"   - Limited benefit for large LED counts")


if __name__ == "__main__":
    analyze_memory_patterns()
