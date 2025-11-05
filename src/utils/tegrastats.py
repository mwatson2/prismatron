"""
Tegrastats Monitor Wrapper for Jetson Platforms.

Provides a Python interface to the tegrastats command-line tool for monitoring
GPU usage, CPU usage, temperatures, memory, and power consumption.
"""

import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TegrastatsData:
    """Parsed tegrastats data."""

    timestamp: datetime = field(default_factory=datetime.now)
    ram_used_mb: int = 0
    ram_total_mb: int = 0
    swap_used_mb: int = 0
    swap_total_mb: int = 0
    cpu_usage: List[float] = field(default_factory=list)  # Per-core percentages
    gpu_usage: float = 0.0  # GR3D_FREQ percentage
    temperatures: Dict[str, float] = field(default_factory=dict)  # sensor_name -> temp_celsius
    power: Dict[str, float] = field(default_factory=dict)  # power_rail -> milliwatts

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "ram_used_mb": self.ram_used_mb,
            "ram_total_mb": self.ram_total_mb,
            "swap_used_mb": self.swap_used_mb,
            "swap_total_mb": self.swap_total_mb,
            "cpu_usage": self.cpu_usage,
            "gpu_usage": self.gpu_usage,
            "temperatures": self.temperatures,
            "power": self.power,
        }


class TegrastatsMonitor:
    """Monitor system stats using tegrastats command-line tool."""

    def __init__(self, interval_ms: int = 1000):
        """
        Initialize tegrastats monitor.

        Args:
            interval_ms: Sampling interval in milliseconds (default 1000ms = 1 second)
        """
        self.interval_ms = interval_ms
        self.callbacks: List[Callable[[TegrastatsData], None]] = []
        self.worker_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_stats: Optional[TegrastatsData] = None
        self.process: Optional[subprocess.Popen] = None

        logger.info(f"TegrastatsMonitor initialized with {interval_ms}ms interval")

    def start(self):
        """Start the tegrastats monitoring thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.shutdown_event.clear()
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Tegrastats monitor thread started")

    def stop(self):
        """Stop the tegrastats monitoring thread."""
        self.shutdown_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

        # Ensure process is terminated
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                logger.warning(f"Error terminating tegrastats process: {e}")

        logger.info("Tegrastats monitor thread stopped")

    def add_callback(self, callback: Callable[[TegrastatsData], None]):
        """
        Add a callback function to be called on each stats update.

        Args:
            callback: Function that takes TegrastatsData as argument
        """
        self.callbacks.append(callback)

    def get_latest_stats(self) -> Optional[TegrastatsData]:
        """
        Get the most recent stats (thread-safe).

        Returns:
            Latest TegrastatsData or None if no stats available yet
        """
        with self.lock:
            return self.latest_stats

    def _worker_loop(self):
        """Background worker loop that runs tegrastats and parses output."""
        logger.info("Tegrastats worker loop started")

        while not self.shutdown_event.is_set():
            try:
                # Start tegrastats process
                cmd = ["tegrastats", "--interval", str(self.interval_ms)]
                logger.info(f"Starting tegrastats: {' '.join(cmd)}")

                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,  # Suppress warnings about inaccessible debug interfaces
                    text=True,
                    bufsize=1,  # Line buffered
                )

                # Read output line by line
                while not self.shutdown_event.is_set():
                    line = self.process.stdout.readline()

                    # If empty string, process has terminated
                    if not line:
                        logger.warning("Tegrastats process terminated unexpectedly")
                        break

                    line = line.strip()
                    if line:
                        # Parse the line
                        stats = self._parse_line(line)
                        if stats:
                            # Update latest stats (thread-safe)
                            with self.lock:
                                self.latest_stats = stats

                            # Notify callbacks
                            self._notify_callbacks(stats)

                # Clean up process
                if self.process:
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=2)
                    except Exception as e:
                        logger.warning(f"Error cleaning up tegrastats process: {e}")

                # If we exited the loop due to process termination (not shutdown), restart
                if not self.shutdown_event.is_set():
                    logger.info("Restarting tegrastats in 5 seconds...")
                    time.sleep(5)

            except Exception as e:
                logger.error(f"Error in tegrastats worker loop: {e}")
                if not self.shutdown_event.is_set():
                    time.sleep(5)

        logger.info("Tegrastats worker loop stopped")

    def _parse_line(self, line: str) -> Optional[TegrastatsData]:
        """
        Parse a single line of tegrastats output.

        Example line:
        11-04-2025 21:20:48 RAM 5242/7620MB (lfb 27x1MB) SWAP 263/3810MB (cached 0MB)
        CPU [44%@1344,47%@1344,39%@1344,36%@1344,35%@729,39%@729] GR3D_FREQ 0%
        cpu@54.281C soc2@52.593C soc0@54.093C gpu@55.437C tj@55.437C soc1@54.218C
        VDD_IN 6239mW/6239mW VDD_CPU_GPU_CV 1547mW/1547mW VDD_SOC 1589mW/1589mW

        Args:
            line: Single line of tegrastats output

        Returns:
            TegrastatsData object or None if parsing failed
        """
        try:
            stats = TegrastatsData()

            # Parse timestamp (MM-DD-YYYY HH:MM:SS format)
            timestamp_match = re.search(r"(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})", line)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                stats.timestamp = datetime.strptime(timestamp_str, "%m-%d-%Y %H:%M:%S")

            # Parse RAM (e.g., "RAM 5242/7620MB")
            ram_match = re.search(r"RAM (\d+)/(\d+)MB", line)
            if ram_match:
                stats.ram_used_mb = int(ram_match.group(1))
                stats.ram_total_mb = int(ram_match.group(2))

            # Parse SWAP (e.g., "SWAP 263/3810MB")
            swap_match = re.search(r"SWAP (\d+)/(\d+)MB", line)
            if swap_match:
                stats.swap_used_mb = int(swap_match.group(1))
                stats.swap_total_mb = int(swap_match.group(2))

            # Parse CPU usage (e.g., "CPU [44%@1344,47%@1344,...]")
            cpu_match = re.search(r"CPU \[([\d%@,]+)\]", line)
            if cpu_match:
                cpu_str = cpu_match.group(1)
                # Extract percentages from format "44%@1344,47%@1344,..."
                cpu_percentages = re.findall(r"(\d+)%@\d+", cpu_str)
                stats.cpu_usage = [float(p) for p in cpu_percentages]

            # Parse GPU usage (e.g., "GR3D_FREQ 0%" or "GR3D_FREQ 45%")
            gpu_match = re.search(r"GR3D_FREQ (\d+)%", line)
            if gpu_match:
                stats.gpu_usage = float(gpu_match.group(1))

            # Parse temperatures (e.g., "cpu@54.281C", "gpu@55.437C")
            temp_matches = re.findall(r"(\w+)@([\d.]+)C", line)
            for sensor, temp in temp_matches:
                stats.temperatures[sensor] = float(temp)

            # Parse power rails (e.g., "VDD_IN 6239mW/6239mW")
            power_matches = re.findall(r"(VDD_\w+) (\d+)mW/(\d+)mW", line)
            for rail, instant, avg in power_matches:
                # Use average power (second value)
                stats.power[rail] = float(avg)

            return stats

        except Exception as e:
            logger.warning(f"Failed to parse tegrastats line: {e}")
            logger.debug(f"Problematic line: {line}")
            return None

    def _notify_callbacks(self, stats: TegrastatsData):
        """Notify all registered callbacks of stats update."""
        for callback in self.callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.warning(f"Callback failed: {e}")
