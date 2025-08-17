#!/usr/bin/env python3
"""
Test the combined audio capture and processing worker to verify
audio overflow elimination with 2048 sample chunks and blocking reads
"""

import logging
import sys
import time

from src.consumer.audio_beat_analyzer import AudioBeatAnalyzer, BeatEvent


def beat_callback(beat_event: BeatEvent):
    """Beat event callback for testing"""
    print(
        f"🎵 Beat #{beat_event.beat_count}: BPM={beat_event.bpm:.1f}, "
        f"Intensity={beat_event.intensity:.2f}, Confidence={beat_event.confidence:.2f}"
    )


def test_combined_worker():
    """Test the combined audio worker implementation"""
    print("=" * 70)
    print("TESTING COMBINED AUDIO CAPTURE AND PROCESSING WORKER")
    print("Testing 2048 sample chunks with blocking reads")
    print("Should eliminate audio overflows")
    print("=" * 70)

    # Configure logging to see overflow warnings
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # Create analyzer with beat callback
    analyzer = AudioBeatAnalyzer(beat_callback=beat_callback)

    try:
        print("Starting combined audio worker...")
        analyzer.start_analysis()

        print("Running for 30 seconds to test for audio overflows...")
        print("Play music with clear beats and watch for:")
        print("- Beat detection events")
        print("- Audio overflow warnings (should be minimal/none)")
        print("- Processing status updates every 10 seconds")
        print()
        print("Press Ctrl+C to stop early...")
        print()

        start_time = time.time()
        last_status = 0

        while time.time() - start_time < 30:
            current_time = time.time()

            # Print status every 5 seconds
            if current_time - last_status >= 5:
                elapsed = current_time - start_time
                state = analyzer.get_current_state()

                print(
                    f"Test status ({elapsed:.1f}s): "
                    f"BPM={state.current_bpm:.1f}, "
                    f"Beats={state.beat_count}, "
                    f"Overflows={analyzer.overflow_count}"
                )

                last_status = current_time

            time.sleep(0.5)

        print("\n" + "=" * 70)
        print("30-second test completed!")

        # Final statistics
        state = analyzer.get_current_state()
        print("Final results:")
        print(f"  Total beats detected: {state.beat_count}")
        print(f"  Final BPM: {state.current_bpm:.1f}")
        print(f"  Audio overflows: {analyzer.overflow_count}")

        if analyzer.overflow_count == 0:
            print("✅ SUCCESS: No audio overflows detected!")
        elif analyzer.overflow_count < 5:
            print("⚠️  ACCEPTABLE: Few audio overflows detected")
        else:
            print("❌ ISSUE: Multiple audio overflows detected")

        print("=" * 70)

    except KeyboardInterrupt:
        print("\nStopping test early...")

    finally:
        analyzer.stop_analysis()
        print("Audio analyzer stopped")


if __name__ == "__main__":
    test_combined_worker()
