#!/usr/bin/env python3
"""
WLED Test Patterns

Interactive test program for sending LED patterns to WLED controllers.
Supports solid colors, rainbow cycles, and animated rainbow effects.

Usage:
    python wled_test_patterns.py --help
    python wled_test_patterns.py solid --color 255 0 0
    python wled_test_patterns.py rainbow-cycle --speed 2.0
    python wled_test_patterns.py animated-rainbow --speed 1.0 --width 0.2
"""

import argparse
import colorsys
import logging
import math
import os
import signal
import sys
import time
from typing import List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.const import LED_COUNT, WLED_DEFAULT_HOST, WLED_DEFAULT_PORT
from src.consumer.wled_client import WLEDClient, WLEDConfig


class LEDPatternGenerator:
    """Generator for various LED patterns and effects."""

    def __init__(self, led_count: int = LED_COUNT):
        self.led_count = led_count
        self.time_offset = 0.0

    def solid_color(self, r: int, g: int, b: int) -> bytes:
        """
        Generate solid color pattern.

        Args:
            r, g, b: RGB color values (0-255)

        Returns:
            LED data bytes
        """
        return bytes([r, g, b] * self.led_count)

    def rainbow_cycle(self, speed: float = 1.0) -> bytes:
        """
        Generate rainbow cycle pattern where all LEDs show the same color,
        cycling through the rainbow spectrum.

        Args:
            speed: Cycle speed (cycles per second)

        Returns:
            LED data bytes
        """
        # Calculate hue based on time
        hue = (self.time_offset * speed) % 1.0

        # Convert HSV to RGB (full saturation and value)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

        # Convert to 0-255 range
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        return bytes([r, g, b] * self.led_count)

    def animated_rainbow(self, speed: float = 1.0, width: float = 1.0) -> bytes:
        """
        Generate animated rainbow pattern where different LEDs show different colors,
        creating a moving rainbow effect across the LED array.

        Args:
            speed: Animation speed (cycles per second)
            width: Rainbow width as fraction of LED array (0.1 = 10% of array)

        Returns:
            LED data bytes
        """
        led_data = []

        for led_index in range(self.led_count):
            # Calculate position-based hue offset
            position_offset = (led_index / self.led_count) * width

            # Add time-based animation
            time_offset = self.time_offset * speed

            # Calculate final hue
            hue = (position_offset + time_offset) % 1.0

            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

            # Convert to 0-255 range and add to data
            led_data.extend([int(r * 255), int(g * 255), int(b * 255)])

        return bytes(led_data)

    def wave_pattern(self, speed: float = 1.0, frequency: float = 2.0) -> bytes:
        """
        Generate sine wave pattern across the LED array.

        Args:
            speed: Wave animation speed
            frequency: Number of waves across the array

        Returns:
            LED data bytes
        """
        led_data = []

        for led_index in range(self.led_count):
            # Calculate wave position
            position = (led_index / self.led_count) * frequency * 2 * math.pi
            wave_value = (math.sin(position + self.time_offset * speed * 2 * math.pi) + 1) / 2

            # Convert to brightness
            brightness = int(wave_value * 255)

            # Create color (can be customized)
            led_data.extend([brightness, 0, brightness])  # Purple wave

        return bytes(led_data)

    def update_time(self, delta_time: float):
        """Update internal time for animations."""
        self.time_offset += delta_time


class WLEDTestRunner:
    """Test runner for WLED patterns."""

    def __init__(self, config: WLEDConfig):
        self.config = config
        self.client = WLEDClient(config)
        self.generator = LEDPatternGenerator(config.led_count)
        self.running = False
        self.start_time = 0.0

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\nShutdown requested...")
        self.running = False

    def run_solid_color(self, r: int, g: int, b: int, duration: Optional[float] = None):
        """
        Run solid color pattern.

        Args:
            r, g, b: RGB color values
            duration: Run duration in seconds (None for infinite)
        """
        print(f"Setting all {self.config.led_count} LEDs to RGB({r}, {g}, {b})")

        if not self.client.connect():
            print("Error: Failed to connect to WLED controller")
            return False

        try:
            success = self.client.set_solid_color(r, g, b)
            if success:
                print("✓ Solid color pattern sent successfully")
                if duration:
                    print(f"Holding for {duration} seconds...")
                    time.sleep(duration)
            else:
                print("✗ Failed to send solid color pattern")
                return False

        finally:
            self.client.disconnect()

        return True

    def run_animated_pattern(
        self,
        pattern_name: str,
        speed: float = 1.0,
        duration: Optional[float] = None,
        **kwargs,
    ):
        """
        Run animated LED pattern.

        Args:
            pattern_name: Name of pattern function
            speed: Animation speed
            duration: Run duration in seconds (None for infinite)
            **kwargs: Additional pattern parameters
        """
        pattern_func = getattr(self.generator, pattern_name, None)
        if not pattern_func:
            print(f"Error: Unknown pattern '{pattern_name}'")
            return False

        print(f"Starting {pattern_name} pattern (speed: {speed:.1f})")
        if duration:
            print(f"Running for {duration} seconds...")
        else:
            print("Running until interrupted (Ctrl+C to stop)...")

        if not self.client.connect():
            print("Error: Failed to connect to WLED controller")
            return False

        try:
            self.running = True
            self.start_time = time.time()
            last_time = self.start_time
            frame_count = 0

            while self.running:
                current_time = time.time()
                delta_time = current_time - last_time
                last_time = current_time

                # Check duration
                if duration and (current_time - self.start_time) >= duration:
                    break

                # Update pattern time
                self.generator.update_time(delta_time)

                # Generate pattern data
                led_data = pattern_func(speed=speed, **kwargs)

                # Send to WLED
                success = self.client.send_led_data(led_data)
                if not success:
                    print(f"\nWarning: Failed to send frame {frame_count}")

                frame_count += 1

                # Print status every few seconds
                if frame_count % (30 * 5) == 0:  # Every ~5 seconds at 30 FPS
                    stats = self.client.get_statistics()
                    elapsed = current_time - self.start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Status: {frame_count} frames, {fps:.1f} FPS, {stats['transmission_errors']} errors")

                # Target ~30 FPS
                time.sleep(1.0 / 30.0)

            # Final statistics
            elapsed = time.time() - self.start_time
            stats = self.client.get_statistics()
            avg_fps = frame_count / elapsed if elapsed > 0 else 0

            print("\nPattern completed:")
            print(f"  Duration: {elapsed:.1f} seconds")
            print(f"  Frames sent: {frame_count}")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Transmission errors: {stats['transmission_errors']}")

        except Exception as e:
            print(f"\nError during pattern execution: {e}")
            return False
        finally:
            self.client.disconnect()

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WLED Test Patterns - Send test patterns to WLED controllers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s solid --color 255 0 0 --duration 5
  %(prog)s rainbow-cycle --speed 2.0
  %(prog)s animated-rainbow --speed 1.0 --width 0.3 --duration 30
  %(prog)s wave --speed 0.5 --frequency 3.0
        """,
    )

    # Global options
    parser.add_argument(
        "--host",
        default=WLED_DEFAULT_HOST,
        help=f"WLED controller hostname/IP (default: {WLED_DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=WLED_DEFAULT_PORT,
        help=f"WLED controller port (default: {WLED_DEFAULT_PORT})",
    )
    parser.add_argument(
        "--led-count",
        type=int,
        default=LED_COUNT,
        help=f"Number of LEDs (default: {LED_COUNT})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Pattern duration in seconds (default: run until interrupted)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--persistent-retry",
        action="store_true",
        help="Keep retrying connection until successful (useful for startup)",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=10.0,
        help="Seconds between connection retries (default: 10.0)",
    )

    # Subcommands for different patterns
    subparsers = parser.add_subparsers(dest="pattern", help="Pattern type")

    # Solid color pattern
    solid_parser = subparsers.add_parser("solid", help="Solid color pattern")
    solid_parser.add_argument(
        "--color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        default=[255, 255, 255],
        help="RGB color values 0-255 (default: white)",
    )

    # Rainbow cycle pattern
    rainbow_parser = subparsers.add_parser("rainbow-cycle", help="Rainbow cycle pattern")
    rainbow_parser.add_argument("--speed", type=float, default=1.0, help="Cycle speed in Hz (default: 1.0)")

    # Animated rainbow pattern
    animated_parser = subparsers.add_parser("animated-rainbow", help="Animated rainbow pattern")
    animated_parser.add_argument("--speed", type=float, default=1.0, help="Animation speed in Hz (default: 1.0)")
    animated_parser.add_argument(
        "--width",
        type=float,
        default=1.0,
        help="Rainbow width as fraction of array (default: 1.0)",
    )

    # Wave pattern
    wave_parser = subparsers.add_parser("wave", help="Sine wave pattern")
    wave_parser.add_argument("--speed", type=float, default=1.0, help="Wave speed in Hz (default: 1.0)")
    wave_parser.add_argument(
        "--frequency",
        type=float,
        default=2.0,
        help="Number of waves across array (default: 2.0)",
    )

    # Test connection
    test_parser = subparsers.add_parser("test", help="Test WLED connection")

    args = parser.parse_args()

    if not args.pattern:
        parser.print_help()
        return 1

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create WLED configuration
    config = WLEDConfig(
        host=args.host,
        port=args.port,
        led_count=args.led_count,
        timeout=5.0,
        retry_count=3,
        max_fps=30.0,
        persistent_retry=args.persistent_retry,
        retry_interval=args.retry_interval,
    )

    if args.verbose:
        print("WLED Configuration:")
        print(f"  Host: {config.host}")
        print(f"  Port: {config.port}")
        print(f"  LED Count: {config.led_count}")
        if config.persistent_retry:
            print(f"  Persistent Retry: enabled (interval: {config.retry_interval}s)")
        print()

    # Create test runner
    runner = WLEDTestRunner(config)

    try:
        if args.pattern == "test":
            # Test connection only
            print(f"Testing connection to WLED controller at {config.host}:{config.port}...")
            if runner.client.connect():
                print("✓ Connection successful")
                stats = runner.client.get_statistics()
                print(f"✓ Controller ready for {stats['led_count']} LEDs")

                # Show WLED status if available
                wled_status = runner.client.get_wled_status()
                if wled_status:
                    print(f"✓ WLED '{wled_status.get('name', 'Unknown')}' v{wled_status.get('ver', 'Unknown')}")
                    if "leds" in wled_status and isinstance(wled_status["leds"], dict):
                        led_info = wled_status["leds"]
                        print(f"✓ Hardware: {led_info.get('count', 0)} LEDs, {led_info.get('fps', 0)} FPS")

                runner.client.disconnect()
                return 0
            else:
                print("✗ Connection failed")
                return 1

        elif args.pattern == "solid":
            r, g, b = args.color
            if not all(0 <= c <= 255 for c in [r, g, b]):
                print("Error: RGB values must be between 0 and 255")
                return 1
            success = runner.run_solid_color(r, g, b, args.duration)

        elif args.pattern == "rainbow-cycle":
            success = runner.run_animated_pattern("rainbow_cycle", speed=args.speed, duration=args.duration)

        elif args.pattern == "animated-rainbow":
            success = runner.run_animated_pattern(
                "animated_rainbow",
                speed=args.speed,
                duration=args.duration,
                width=args.width,
            )

        elif args.pattern == "wave":
            success = runner.run_animated_pattern(
                "wave_pattern",
                speed=args.speed,
                duration=args.duration,
                frequency=args.frequency,
            )
        else:
            print(f"Error: Unknown pattern '{args.pattern}'")
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
