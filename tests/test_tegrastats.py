#!/usr/bin/env python3
"""
Test script for TegrastatsMonitor wrapper.

Demonstrates usage of the tegrastats monitoring utility and displays
live system statistics including GPU usage, CPU usage, temperatures, and power.
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.tegrastats import TegrastatsData, TegrastatsMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def format_stats(stats: TegrastatsData) -> str:
    """Format stats for display."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Timestamp: {stats.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("-" * 80)

    # Memory
    ram_percent = (stats.ram_used_mb / stats.ram_total_mb * 100) if stats.ram_total_mb > 0 else 0
    lines.append(f"RAM:  {stats.ram_used_mb:5d} / {stats.ram_total_mb:5d} MB  ({ram_percent:5.1f}%)")

    if stats.swap_total_mb > 0:
        swap_percent = (stats.swap_used_mb / stats.swap_total_mb * 100) if stats.swap_total_mb > 0 else 0
        lines.append(f"SWAP: {stats.swap_used_mb:5d} / {stats.swap_total_mb:5d} MB  ({swap_percent:5.1f}%)")

    # GPU
    lines.append("-" * 80)
    lines.append(f"GPU Usage: {stats.gpu_usage:5.1f}%")

    # CPU
    if stats.cpu_usage:
        cpu_str = "CPU Usage: " + ", ".join([f"Core{i}: {usage:5.1f}%" for i, usage in enumerate(stats.cpu_usage)])
        lines.append(cpu_str)
        avg_cpu = sum(stats.cpu_usage) / len(stats.cpu_usage)
        lines.append(f"CPU Avg:   {avg_cpu:5.1f}%")

    # Temperatures
    if stats.temperatures:
        lines.append("-" * 80)
        lines.append("Temperatures:")
        for sensor, temp in sorted(stats.temperatures.items()):
            lines.append(f"  {sensor:10s}: {temp:6.2f}°C")

    # Power
    if stats.power:
        lines.append("-" * 80)
        lines.append("Power Draw:")
        for rail, power_mw in sorted(stats.power.items()):
            power_w = power_mw / 1000.0
            lines.append(f"  {rail:20s}: {power_w:6.2f} W  ({power_mw:7.1f} mW)")

    lines.append("=" * 80)
    return "\n".join(lines)


def stats_callback(stats: TegrastatsData):
    """Callback function called on each stats update."""
    print("\n" + format_stats(stats))


def main():
    """Test the tegrastats monitor."""
    print("Tegrastats Monitor Test")
    print("=" * 80)
    print("This test will run for 15 seconds and display live system statistics.")
    print("Press Ctrl+C to stop early.")
    print("=" * 80)

    # Create monitor with 1 second interval
    monitor = TegrastatsMonitor(interval_ms=1000)

    # Register callback
    monitor.add_callback(stats_callback)

    try:
        # Start monitoring
        logger.info("Starting tegrastats monitor...")
        monitor.start()

        # Run for 15 seconds
        logger.info("Monitoring for 15 seconds...")
        time.sleep(15)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        # Stop monitoring
        logger.info("Stopping tegrastats monitor...")
        monitor.stop()

        # Display final stats
        final_stats = monitor.get_latest_stats()
        if final_stats:
            print("\n" + "=" * 80)
            print("FINAL STATS:")
            print(format_stats(final_stats))
            print("\nStats as dictionary:")
            print(final_stats.to_dict())
        else:
            logger.warning("No stats collected")

    logger.info("Test complete")


if __name__ == "__main__":
    main()
