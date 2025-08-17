#!/usr/bin/env python3
"""
Quick performance test of optimized beat analyzer
"""

import logging
import os
import sys
import time

# Set environment to reduce resource usage
os.environ["MADMOM_SKIP_GPU"] = "True"

# Add path
sys.path.insert(0, "/mnt/dev/prismatron")

# Import directly
import importlib.util

spec = importlib.util.spec_from_file_location(
    "audio_beat_analyzer", "/mnt/dev/prismatron/src/consumer/audio_beat_analyzer.py"
)
audio_beat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_beat_module)

AudioBeatAnalyzer = audio_beat_module.AudioBeatAnalyzer
BeatEvent = audio_beat_module.BeatEvent

# Reduce logging to see performance
logging.basicConfig(level=logging.WARNING)

beat_count = 0


def beat_callback(beat_event: BeatEvent):
    global beat_count
    beat_count += 1
    if beat_count % 5 == 0:  # Only print every 5th beat
        print(f"Beat #{beat_count}: BPM={beat_event.bpm:.1f}")


def main():
    print("Testing optimized AudioBeatAnalyzer performance...")

    analyzer = AudioBeatAnalyzer(beat_callback=beat_callback, device="cpu")

    try:
        start_time = time.time()
        analyzer.start_analysis()
        print("Started. Running for 5 seconds...")

        time.sleep(5)

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.1f}s, detected {beat_count} beats")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        analyzer.stop_analysis()


if __name__ == "__main__":
    main()
