#!/usr/bin/env python3
"""
Quick test of the integrated audio beat analyzer
"""

import logging
import sys
import time

sys.path.insert(0, "/mnt/dev/prismatron")

# Import directly from the file to avoid package issues
import importlib.util

spec = importlib.util.spec_from_file_location(
    "audio_beat_analyzer", "/mnt/dev/prismatron/src/consumer/audio_beat_analyzer.py"
)
audio_beat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_beat_module)

AudioBeatAnalyzer = audio_beat_module.AudioBeatAnalyzer
BeatEvent = audio_beat_module.BeatEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def beat_callback(beat_event: BeatEvent):
    """Handle beat events"""
    beat_type = "DOWNBEAT" if beat_event.is_downbeat else "BEAT"
    print(
        f"🎵 {beat_type} #{beat_event.beat_count}: BPM={beat_event.bpm:.1f}, "
        f"intensity={beat_event.intensity:.2f}, conf={beat_event.confidence:.2f}"
    )


def main():
    print("Testing integrated AudioBeatAnalyzer...")

    # Create analyzer
    analyzer = AudioBeatAnalyzer(beat_callback=beat_callback)

    try:
        # Start analysis
        analyzer.start_analysis()
        print("Beat analyzer started. Running for 10 seconds...")

        # Run for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            time.sleep(1)

            # Show current state
            state = analyzer.get_current_state()
            print(
                f"Status: BPM={state.current_bpm:.1f}, Beats={state.beat_count}, "
                f"Downbeats={state.downbeat_count}, Active={state.is_active}"
            )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        analyzer.stop_analysis()
        print("Test completed!")


if __name__ == "__main__":
    main()
