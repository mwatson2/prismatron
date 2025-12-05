#!/usr/bin/env python3
"""
Prismatron System Orchestrator.

Main entry point that coordinates startup and management of all system processes:
- Playlist Synchronization Service (Unix domain sockets)
- Web API Server (FastAPI)
- Producer Process (Content rendering)
- Consumer Process (LED optimization and WLED communication)

This orchestrator ensures proper startup sequencing, monitors process health,
and handles graceful shutdown coordination.
"""

import argparse
import atexit
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Don't import ConsumerProcess here - it imports cupy which breaks fork()
# from src.consumer.consumer import ConsumerProcess
from src.core.control_state import ControlState, PlayState, SystemState
from src.core.playlist_sync import PlaylistSyncService
from src.producer.producer import ProducerProcess
from src.web.api_server import run_server

logger = logging.getLogger(__name__)


class ProcessManager:
    """Manages all system processes and coordinates their lifecycle."""

    def __init__(self, config: Dict):
        """
        Initialize process manager.

        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.processes: Dict[str, multiprocessing.Process] = {}

        # Set umask to allow group access for shared memory
        import os

        os.umask(0o002)

        # Create a shared lock for inter-process synchronization of ControlState
        self.shared_control_lock = multiprocessing.Lock()

        # Create shared events for inter-process coordination
        self.shared_shutdown_event = multiprocessing.Event()
        self.shared_restart_event = multiprocessing.Event()
        self.shared_reboot_event = multiprocessing.Event()
        self.shared_config_updated_event = multiprocessing.Event()
        self.shared_status_updated_event = multiprocessing.Event()

        # Set the shared lock and events in ControlState before creating instances
        ControlState.set_shared_lock(self.shared_control_lock)
        ControlState.set_shared_events(
            self.shared_shutdown_event,
            self.shared_restart_event,
            self.shared_reboot_event,
            self.shared_config_updated_event,
            self.shared_status_updated_event,
        )

        self.control_state = ControlState()
        self.shutdown_requested = False

        # Clean up any orphaned shared memory from previous runs
        self._cleanup_orphaned_shared_memory()

        # Clean up any orphaned Unix domain socket from previous runs
        self._cleanup_orphaned_socket()

        # Process startup coordination
        self.playlist_sync_ready = multiprocessing.Event()
        self.web_server_ready = multiprocessing.Event()
        self.producer_ready = multiprocessing.Event()
        self.consumer_ready = multiprocessing.Event()

    def start_all_processes(self) -> bool:
        """
        Start all system processes in proper sequence.

        Returns:
            True if all processes started successfully
        """
        try:
            logger.info("Starting Prismatron system processes...")

            # Initialize control state
            if not self.control_state.initialize():
                logger.error("Failed to initialize control state")
                return False

            self.control_state.update_system_state(SystemState.STARTING)

            # Start playlist sync service first (required by web server and producer)
            if not self._start_playlist_sync():
                return False

            # Start web server (API needed for producer control)
            if not self._start_web_server():
                return False

            # Start consumer process (waits for WLED)
            if not self._start_consumer():
                return False

            # Start producer process (waits for API play command)
            if not self._start_producer():
                return False

            # Wait for all processes to be ready
            logger.info("Waiting for all processes to initialize...")

            # Playlist sync service should be ready quickly
            if not self.playlist_sync_ready.wait(timeout=15):
                logger.error("Playlist sync service startup timeout")
                return False

            # Web server should be ready quickly
            if not self.web_server_ready.wait(timeout=30):
                logger.error("Web server startup timeout")
                return False

            # Consumer needs time to connect to WLED
            if not self.consumer_ready.wait(timeout=60):
                logger.error("Consumer startup timeout")
                return False

            # Producer should be ready quickly
            if not self.producer_ready.wait(timeout=30):
                logger.error("Producer startup timeout")
                return False

            self.control_state.update_system_state(SystemState.RUNNING)
            logger.info("All processes started successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to start processes: {e}")
            self.stop_all_processes()
            return False

    def _start_playlist_sync(self) -> bool:
        """Start the playlist synchronization service."""
        try:
            logger.info("Starting playlist synchronization service...")

            def playlist_sync_worker():
                """Playlist sync service worker."""
                try:
                    # Ensure the shared lock and events are set in the child process
                    # (they should already be inherited, but this makes it explicit)
                    ControlState.set_shared_lock(self.shared_control_lock)
                    ControlState.set_shared_events(
                        self.shared_shutdown_event,
                        self.shared_restart_event,
                        self.shared_reboot_event,
                        self.shared_config_updated_event,
                        self.shared_status_updated_event,
                    )

                    # Setup logging for subprocess
                    import os

                    from src.utils.logging_utils import create_app_time_formatter

                    # Clear any existing handlers
                    root_logger = logging.getLogger()
                    root_logger.handlers = []

                    formatter = create_app_time_formatter()

                    # Create both console and file handlers
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)

                    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "prismatron.log")
                    file_handler = logging.FileHandler(log_file_path, mode="a")
                    file_handler.setFormatter(formatter)

                    root_logger.setLevel(logging.INFO if not self.config.get("debug") else logging.DEBUG)
                    root_logger.addHandler(console_handler)
                    root_logger.addHandler(file_handler)

                    # Enable DEBUG for consumer and producer components to debug thread issues
                    logging.getLogger("src.consumer.consumer").setLevel(logging.INFO)
                    logging.getLogger("src.consumer.frame_renderer").setLevel(logging.INFO)
                    logging.getLogger("src.consumer.led_buffer").setLevel(logging.INFO)
                    logging.getLogger("src.producer.producer").setLevel(logging.INFO)

                    # Create and start playlist sync service
                    service = PlaylistSyncService()
                    if not service.start():
                        logger.error("Failed to start playlist sync service")
                        return

                    # Signal ready
                    self.playlist_sync_ready.set()

                    # Keep running until shutdown
                    while service.running:
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Playlist sync service error: {e}")

            process = multiprocessing.Process(target=playlist_sync_worker, name="PlaylistSync")
            process.start()
            self.processes["playlist_sync"] = process

            return True

        except Exception as e:
            logger.error(f"Failed to start playlist sync service: {e}")
            return False

    def _start_web_server(self) -> bool:
        """Start the web API server process."""
        try:
            logger.info("Starting web server...")

            def web_server_worker():
                """Web server process worker."""
                try:
                    # Ensure the shared lock and events are set in the child process
                    # (they should already be inherited, but this makes it explicit)
                    ControlState.set_shared_lock(self.shared_control_lock)
                    ControlState.set_shared_events(
                        self.shared_shutdown_event,
                        self.shared_restart_event,
                        self.shared_reboot_event,
                        self.shared_config_updated_event,
                        self.shared_status_updated_event,
                    )

                    # Setup logging for subprocess
                    import os

                    from src.utils.logging_utils import create_app_time_formatter

                    # Clear any existing handlers
                    root_logger = logging.getLogger()
                    root_logger.handlers = []

                    formatter = create_app_time_formatter()

                    # Create both console and file handlers
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)

                    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "prismatron.log")
                    file_handler = logging.FileHandler(log_file_path, mode="a")
                    file_handler.setFormatter(formatter)

                    root_logger.setLevel(logging.INFO if not self.config.get("debug") else logging.DEBUG)
                    root_logger.addHandler(console_handler)
                    root_logger.addHandler(file_handler)

                    # Enable DEBUG for consumer and producer components to debug thread issues
                    logging.getLogger("src.consumer.consumer").setLevel(logging.INFO)
                    logging.getLogger("src.consumer.frame_renderer").setLevel(logging.INFO)
                    logging.getLogger("src.consumer.led_buffer").setLevel(logging.INFO)
                    logging.getLogger("src.producer.producer").setLevel(logging.INFO)

                    # Signal ready
                    self.web_server_ready.set()

                    # Start server (blocks until shutdown)
                    run_server(
                        host=self.config.get("web_host", "0.0.0.0"),
                        port=self.config.get("web_port", 8000),
                        debug=self.config.get("debug", False),
                        patterns_path=self.config.get("diffusion_patterns_path"),
                        led_count=self.config.get("led_count"),
                        config=self.config,
                    )

                except Exception as e:
                    logger.error(f"Web server error: {e}")

            process = multiprocessing.Process(target=web_server_worker, name="WebServer")
            process.start()
            self.processes["web_server"] = process

            return True

        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            return False

    def _start_consumer(self) -> bool:
        """Start the consumer process."""
        try:
            logger.info("Starting consumer process...")

            def consumer_worker():
                """Consumer process worker."""
                try:
                    # Ensure the shared lock and events are set in the child process
                    # (they should already be inherited, but this makes it explicit)
                    ControlState.set_shared_lock(self.shared_control_lock)
                    ControlState.set_shared_events(
                        self.shared_shutdown_event,
                        self.shared_restart_event,
                        self.shared_reboot_event,
                        self.shared_config_updated_event,
                        self.shared_status_updated_event,
                    )

                    # Setup logging for subprocess
                    import os

                    from src.utils.logging_utils import create_app_time_formatter

                    # Clear any existing handlers
                    root_logger = logging.getLogger()
                    root_logger.handlers = []

                    formatter = create_app_time_formatter()

                    # Create both console and file handlers
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)

                    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "prismatron.log")
                    file_handler = logging.FileHandler(log_file_path, mode="a")
                    file_handler.setFormatter(formatter)

                    root_logger.setLevel(logging.INFO if not self.config.get("debug") else logging.DEBUG)
                    root_logger.addHandler(console_handler)
                    root_logger.addHandler(file_handler)

                    # Enable DEBUG for consumer and producer components to debug thread issues
                    logging.getLogger("src.consumer.consumer").setLevel(logging.INFO)
                    logging.getLogger("src.consumer.frame_renderer").setLevel(logging.INFO)
                    logging.getLogger("src.consumer.led_buffer").setLevel(logging.INFO)
                    logging.getLogger("src.producer.producer").setLevel(logging.INFO)

                    # Debug environment variables
                    logger.info(f"Consumer worker environment:")
                    logger.info(f"  PATH: {os.environ.get('PATH', 'NOT SET')}")
                    logger.info(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')}")
                    logger.info(f"  CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")
                    logger.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
                    logger.info(f"  NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'NOT SET')}")

                    # Import ConsumerProcess
                    from src.consumer.consumer import ConsumerProcess

                    # Create consumer with configuration
                    # Support both single host (legacy) and list of hosts
                    wled_hosts = self.config.get("wled_hosts")
                    if wled_hosts is None:
                        # Fallback to single host for backward compatibility
                        wled_hosts = self.config.get("wled_host", "192.168.7.140")
                    consumer = ConsumerProcess(
                        wled_hosts=wled_hosts,
                        wled_port=self.config.get("wled_port", 4048),
                        diffusion_patterns_path=self.config.get("diffusion_patterns_path"),
                        timing_log_path=self.config.get("timing_log_path"),
                        enable_batch_mode=self.config.get("enable_batch_mode", False),
                        enable_adaptive_frame_dropping=self.config.get("enable_adaptive_frame_dropping", True),
                        enable_audio_reactive=self.config.get("enable_audio_reactive", False),
                        audio_device=self.config.get("audio_device"),
                        optimization_iterations=self.config.get("optimization_iterations", 10),
                    )

                    # Initialize consumer (WLED connection not required for startup)
                    logger.info("Initializing consumer process...")
                    if not consumer.initialize():
                        logger.error("Failed to initialize consumer process")
                        return

                    logger.info("Consumer initialized successfully (WLED connection will be retried automatically)")

                    # Signal ready
                    self.consumer_ready.set()

                    # Start consumer (blocks until shutdown)
                    if not consumer.start():
                        logger.error("Failed to start consumer")
                        return

                    # Keep running until shutdown
                    while consumer._running:
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Consumer process error: {e}")

            process = multiprocessing.Process(target=consumer_worker, name="Consumer")
            process.start()
            self.processes["consumer"] = process

            return True

        except Exception as e:
            logger.error(f"Failed to start consumer: {e}")
            return False

    def _start_producer(self) -> bool:
        """Start the producer process."""
        try:
            logger.info("Starting producer process...")

            def producer_worker():
                """Producer process worker."""
                try:
                    # Ensure the shared lock and events are set in the child process
                    # (they should already be inherited, but this makes it explicit)
                    ControlState.set_shared_lock(self.shared_control_lock)
                    ControlState.set_shared_events(
                        self.shared_shutdown_event,
                        self.shared_restart_event,
                        self.shared_reboot_event,
                        self.shared_config_updated_event,
                        self.shared_status_updated_event,
                    )

                    # Setup logging for subprocess
                    import os

                    from src.utils.logging_utils import create_app_time_formatter

                    # Clear any existing handlers
                    root_logger = logging.getLogger()
                    root_logger.handlers = []

                    formatter = create_app_time_formatter()

                    # Create both console and file handlers
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)

                    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "prismatron.log")
                    file_handler = logging.FileHandler(log_file_path, mode="a")
                    file_handler.setFormatter(formatter)

                    root_logger.setLevel(logging.INFO if not self.config.get("debug") else logging.DEBUG)
                    root_logger.addHandler(console_handler)
                    root_logger.addHandler(file_handler)

                    # Enable DEBUG for consumer and producer components to debug thread issues
                    logging.getLogger("src.consumer.consumer").setLevel(logging.INFO)
                    logging.getLogger("src.consumer.frame_renderer").setLevel(logging.INFO)
                    logging.getLogger("src.consumer.led_buffer").setLevel(logging.INFO)
                    logging.getLogger("src.producer.producer").setLevel(logging.INFO)

                    # Create producer
                    producer = ProducerProcess()

                    if not producer.initialize():
                        logger.error("Failed to initialize producer")
                        return

                    # Load default content if specified
                    content_dir = self.config.get("default_content_dir")
                    if content_dir and os.path.exists(content_dir):
                        count = producer.load_playlist_from_directory(content_dir)
                        logger.info(f"Loaded {count} content files from {content_dir}")

                    # Signal ready
                    self.producer_ready.set()

                    # Start producer (blocks until shutdown)
                    if not producer.start():
                        logger.error("Failed to start producer")
                        return

                    # Producer will wait for API play command before starting
                    logger.info("Producer ready - waiting for API play command")

                    # Keep running until shutdown
                    while producer._running:
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Producer process error: {e}")

            process = multiprocessing.Process(target=producer_worker, name="Producer")
            process.start()
            self.processes["producer"] = process

            return True

        except Exception as e:
            logger.error(f"Failed to start producer: {e}")
            return False

    def stop_all_processes(self) -> None:
        """Stop all processes gracefully."""
        try:
            logger.info("Stopping all processes...")
            self.shutdown_requested = True

            # Update control state
            self.control_state.update_system_state(SystemState.STOPPING)
            self.control_state.signal_shutdown()

            # Stop processes in reverse order
            process_order = ["producer", "consumer", "web_server", "playlist_sync"]

            for process_name in process_order:
                if process_name in self.processes:
                    process = self.processes[process_name]

                    if process.is_alive():
                        logger.info(f"Stopping {process_name}...")
                        process.terminate()

                        # Wait for graceful shutdown
                        process.join(timeout=10)

                        if process.is_alive():
                            logger.warning(f"Force killing {process_name}")
                            process.kill()
                            process.join(timeout=5)

            # Cleanup control state
            self.control_state.cleanup()

            logger.info("All processes stopped")

        except Exception as e:
            logger.error(f"Error stopping processes: {e}")

    def monitor_processes(self) -> None:
        """Monitor process health and handle control signals."""
        logger.info("Starting process monitoring loop...")
        start_time = time.time()
        last_log_time = start_time
        while not self.shutdown_requested:
            try:
                current_time = time.time()
                # Log every 10 minutes (600 seconds)
                if current_time - last_log_time >= 600:
                    uptime_seconds = int(current_time - start_time)
                    hours = uptime_seconds // 3600
                    minutes = (uptime_seconds % 3600) // 60
                    seconds = uptime_seconds % 60
                    logger.info(f"Monitor loop - Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
                    last_log_time = current_time

                # Check for control signals
                restart_requested = self.control_state.is_restart_requested()
                if restart_requested:
                    logger.info("Restart signal detected - shutting down for systemd restart")
                    self.stop_all_processes()
                    logger.info("All processes stopped, exiting for restart...")
                    # Exit with code 1 to trigger systemd restart (Restart=on-failure)
                    sys.exit(1)

                if self.control_state.is_reboot_requested():
                    logger.info("Reboot signal detected")
                    self.reboot_system()
                    return  # reboot_system will not return

                if self.control_state.is_shutdown_requested():
                    logger.info("Shutdown signal detected")
                    self.shutdown_requested = True
                    break

                # Check each process health
                for name, process in self.processes.items():
                    if not process.is_alive():
                        logger.error(f"Process {name} has died unexpectedly")
                        # TODO: Implement automatic restart logic if desired

                time.sleep(2)  # Check every 2 seconds

            except Exception as e:
                logger.error(f"Error monitoring processes: {e}")
                time.sleep(2)

    def restart_system(self) -> bool:
        """Restart all system processes."""
        logger.info("Restarting system...")
        self.stop_all_processes()
        time.sleep(2)  # Brief pause
        return self.start_all_processes()

    def reboot_system(self) -> None:
        """Reboot the entire device (Linux only)."""
        logger.info("Rebooting system...")
        self.stop_all_processes()

        try:
            # Use full path for reboot command
            # Requires sudoers rule: mark ALL=(ALL) NOPASSWD: /sbin/reboot
            logger.info("Executing sudo /sbin/reboot...")
            os.system("sudo /sbin/reboot")
        except Exception as e:
            logger.error(f"Failed to reboot system: {e}")

    def _cleanup_orphaned_shared_memory(self) -> None:
        """Clean up any orphaned shared memory from previous runs."""
        try:
            from multiprocessing import shared_memory

            # Try to connect to existing memory and clean it up
            try:
                orphaned_memory = shared_memory.SharedMemory(name="prismatron_control")
                orphaned_memory.close()
                orphaned_memory.unlink()
                logger.info("Cleaned up orphaned shared memory 'prismatron_control'")
            except FileNotFoundError:
                # No orphaned memory exists, which is good
                pass
        except Exception as e:
            logger.warning(f"Error checking for orphaned shared memory: {e}")

    def _cleanup_orphaned_socket(self) -> None:
        """Clean up any orphaned Unix domain socket from previous runs."""
        try:
            import os

            socket_path = "/tmp/prismatron_playlist.sock"
            if os.path.exists(socket_path):
                os.unlink(socket_path)
                logger.info("Cleaned up orphaned playlist socket")
        except Exception as e:
            logger.warning(f"Error cleaning up orphaned socket: {e}")

    def cleanup_all_resources(self) -> None:
        """Comprehensive cleanup of all system resources."""
        if self.shutdown_requested:
            return  # Avoid duplicate cleanup

        try:
            self.shutdown_requested = True
            logger.info("Performing comprehensive cleanup...")
            self.stop_all_processes()

            # Stop log rotation service
            try:
                from src.utils.log_rotation import stop_log_rotation

                if stop_log_rotation(timeout=2.0):
                    logger.info("Log rotation service stopped")
            except Exception as e:
                logger.warning(f"Error stopping log rotation: {e}")

            # Additional cleanup for any remaining shared memory
            self._cleanup_orphaned_shared_memory()

            # Additional cleanup for any remaining sockets
            self._cleanup_orphaned_socket()

        except Exception as e:
            logger.error(f"Error during comprehensive cleanup: {e}")


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration with automatic rotation."""
    import time

    from src.utils.log_rotation import start_log_rotation
    from src.utils.logging_utils import create_app_time_formatter, set_app_start_time

    # Set application start time
    set_app_start_time(time.time())

    level = logging.DEBUG if debug else logging.INFO

    # Clear the log file on startup
    log_file = "logs/prismatron.log"
    try:
        with open(log_file, "w") as f:
            f.write("")  # Clear the file
    except Exception:
        pass  # If we can't clear it, continue anyway

    # Create custom formatter with app time
    formatter = create_app_time_formatter()

    # Get the root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []

    # Create handlers
    # Console handler: WARNING+ only (these go to journalctl when running as systemd service)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)

    # File handler: full logging level (DEBUG or INFO based on --debug flag)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Set up the root logger
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Start log rotation service (checks every 5 minutes)
    if start_log_rotation(log_file, check_interval=300):
        logger.info("Log rotation service started (max 100MB per file, 200MB total)")
    else:
        logger.warning("Failed to start log rotation service")

    # Reduce noise from some modules
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.protocols.websockets").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.protocols.websockets.impl").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("websockets.protocol").setLevel(logging.WARNING)
    logging.getLogger("websockets.server").setLevel(logging.WARNING)


def signal_handler(signum, frame, process_manager: ProcessManager) -> None:
    """Handle shutdown signals."""
    print(f"Prismatron: Received signal {signum}, shutting down...")
    try:
        process_manager.cleanup_all_resources()
    except Exception as e:
        print(f"Prismatron: Error during cleanup: {e}")
    sys.exit(0)


def emergency_cleanup() -> None:
    """Emergency cleanup function for atexit."""
    try:
        from multiprocessing import shared_memory

        orphaned_memory = shared_memory.SharedMemory(name="prismatron_control")
        orphaned_memory.close()
        orphaned_memory.unlink()
        # Removed print to make emergency cleanup silent
    except:
        pass  # Silently handle any errors during emergency cleanup

    # Clean up socket
    try:
        import os

        socket_path = "/tmp/prismatron_playlist.sock"
        if os.path.exists(socket_path):
            os.unlink(socket_path)
    except:
        pass  # Silently handle any errors


def load_led_count_from_patterns(patterns_path: str) -> int:
    """Load LED count from diffusion patterns file."""
    import numpy as np

    if not patterns_path or not os.path.exists(patterns_path):
        raise ValueError(f"Pattern file not found: {patterns_path}")

    try:
        data = np.load(patterns_path, allow_pickle=True)
        metadata = data.get("metadata")

        if metadata is None:
            raise ValueError("Pattern file is missing metadata")

        metadata_item = metadata.item() if hasattr(metadata, "item") else metadata

        if "led_count" not in metadata_item:
            raise ValueError("Pattern file is missing required 'led_count' metadata")

        led_count = int(metadata_item["led_count"])
        logger.info(f"Loaded LED count from patterns: {led_count}")
        return led_count

    except Exception as e:
        raise ValueError(f"Failed to load LED count from patterns file: {e}")


def write_pid_file(pid_file: Path):
    """Write PID file for systemd tracking."""
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    logger.info(f"Written PID {os.getpid()} to {pid_file}")


def cleanup_pid_file(pid_file: Path):
    """Remove PID file on exit."""
    try:
        pid_file.unlink()
        logger.info(f"Removed PID file {pid_file}")
    except FileNotFoundError:
        pass


def daemonize():
    """Daemonize the process for systemd Type=forking."""
    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent process exits
        sys.exit(0)

    # Decouple from parent environment
    os.setsid()
    os.umask(0)

    # Second fork
    pid = os.fork()
    if pid > 0:
        # Parent process exits
        sys.exit(0)

    # We're now in the daemon process
    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    # Close file descriptors
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, sys.stdin.fileno())
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)


# Get the project root directory (where main.py is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")


def load_config_file(config_path: str = None) -> Dict:
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                # Filter out comments and null values
                config = {k: v for k, v in loaded_config.items() if k != "comments" and v is not None}
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config file {config_path}: {e}")
    else:
        logger.warning(f"Config file not found: {config_path}")
    return config


def main():
    """Main entry point."""

    # Parse just the config argument first to know which config file to load
    parser_config = argparse.ArgumentParser(add_help=False)
    parser_config.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to configuration file")
    config_args, remaining = parser_config.parse_known_args()

    # Load default configuration from specified file
    file_config = load_config_file(config_args.config)

    parser = argparse.ArgumentParser(description="Prismatron LED Display System")
    parser.add_argument("--config", default=config_args.config, help="Path to configuration file")
    parser.add_argument(
        "--debug", action="store_true", default=file_config.get("debug", False), help="Enable debug logging"
    )
    parser.add_argument("--web-host", default=file_config.get("web_host", "0.0.0.0"), help="Web server host")
    parser.add_argument("--web-port", type=int, default=file_config.get("web_port", 8080), help="Web server port")
    # Handle both wled_hosts (list) and wled_host (single) for backward compatibility
    wled_default = file_config.get("wled_hosts")
    if wled_default is None:
        wled_default = file_config.get("wled_host", "192.168.7.140")
    parser.add_argument(
        "--wled-hosts",
        nargs="+",
        default=wled_default if isinstance(wled_default, list) else [wled_default],
        help="WLED controller IPs or hostnames (tries each in order)",
    )
    parser.add_argument(
        "--wled-port", type=int, default=file_config.get("wled_port", 4048), help="WLED controller port"
    )
    parser.add_argument(
        "--content-dir", default=file_config.get("content_dir"), help="Default content directory to load"
    )
    parser.add_argument(
        "--diffusion-patterns", default=file_config.get("diffusion_patterns"), help="Path to diffusion patterns file"
    )
    parser.add_argument(
        "--timing-log", default=file_config.get("timing_log"), help="Path to CSV file for frame timing data logging"
    )

    # Handle batch-mode with config file default
    batch_mode_default = file_config.get("batch_mode", True)
    if batch_mode_default:
        parser.add_argument(
            "--batch-mode",
            action="store_true",
            default=True,
            help="Enable batch processing (8 frames at once) for improved performance",
        )
        parser.add_argument("--no-batch-mode", dest="batch_mode", action="store_false", help="Disable batch processing")
    else:
        parser.add_argument(
            "--batch-mode",
            action="store_true",
            default=False,
            help="Enable batch processing (8 frames at once) for improved performance",
        )

    # Handle adaptive-dropping with config file default
    adaptive_dropping_default = file_config.get("adaptive_dropping", False)
    if adaptive_dropping_default:
        parser.add_argument(
            "--adaptive-dropping",
            dest="no_adaptive_dropping",
            action="store_false",
            default=False,
            help="Enable adaptive frame dropping for LED buffer management",
        )
        parser.add_argument(
            "--no-adaptive-dropping", action="store_true", default=False, help="Disable adaptive frame dropping"
        )
    else:
        parser.add_argument(
            "--no-adaptive-dropping",
            action="store_true",
            default=True,
            help="Disable adaptive frame dropping for LED buffer management",
        )
        parser.add_argument(
            "--adaptive-dropping",
            dest="no_adaptive_dropping",
            action="store_false",
            help="Enable adaptive frame dropping",
        )

    # Audio reactive settings
    parser.add_argument(
        "--audio-reactive",
        action="store_true",
        default=file_config.get("audio_reactive", False),
        help="Enable audio reactive effects",
    )
    parser.add_argument(
        "--audio-device", default=file_config.get("audio_device"), help="Audio device index or name for beat detection"
    )

    # Daemon mode for systemd
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (for systemd service)")

    args = parser.parse_args()

    # Handle daemon mode before setting up logging
    pid_file = None
    if args.daemon:
        # Daemonize the process
        daemonize()

        # Write PID file
        pid_file = Path("/run/prismatron/prismatron.pid")
        write_pid_file(pid_file)

        # Register cleanup for PID file
        atexit.register(cleanup_pid_file, pid_file)

    # Setup logging (after daemonization so logs go to the right place)
    setup_logging(args.debug)

    # Register emergency cleanup for all exit scenarios
    atexit.register(emergency_cleanup)

    # Load LED count from patterns file if provided
    led_count = None
    if args.diffusion_patterns:
        try:
            led_count = load_led_count_from_patterns(args.diffusion_patterns)
        except ValueError as e:
            logger.error(f"Failed to load LED count: {e}")
            sys.exit(1)

    # Create configuration
    config = {
        "debug": args.debug,
        "web_host": args.web_host,
        "web_port": args.web_port,
        "wled_hosts": args.wled_hosts,
        "wled_port": args.wled_port,
        "default_content_dir": args.content_dir,
        "diffusion_patterns_path": args.diffusion_patterns,
        "led_count": led_count,  # Add LED count to config
        "timing_log_path": args.timing_log,
        "enable_batch_mode": args.batch_mode,
        "enable_adaptive_frame_dropping": not args.no_adaptive_dropping,  # Invert the flag
        "enable_audio_reactive": args.audio_reactive,
        "audio_device": args.audio_device,
    }

    # Merge in any additional configuration from the JSON file that isn't covered by command line args
    for key, value in file_config.items():
        if key not in config and key != "comments":
            config[key] = value

    logger.info("Starting Prismatron LED Display System")
    logger.info(f"Configuration: {config}")

    try:
        # Create process manager
        manager = ProcessManager(config)

        # Setup signal handlers
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, manager))
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, manager))

        # Start all processes
        if not manager.start_all_processes():
            logger.error("Failed to start system")
            sys.exit(1)

        logger.info("System started successfully!")
        logger.info(f"Web interface available at http://{args.web_host}:{args.web_port}")

        # Ensure network connectivity (run in background)
        import asyncio

        from src.network.manager import NetworkManager

        async def setup_network():
            """Log network connectivity status.

            Note: AP fallback is handled by system-level NetworkManager dispatcher script.
            """
            try:
                nm = NetworkManager()
                mode = await nm.get_connectivity_status()
                if mode == "client":
                    logger.info("Network status: WiFi client mode active")
                elif mode == "ap":
                    logger.info("Network status: AP mode active")
                else:
                    logger.info("Network status: disconnected (AP fallback handled by system)")
            except Exception as e:
                logger.error(f"Network status check error: {e}")

        # Start network setup in background (don't block main process)
        import threading

        def run_network_setup():
            asyncio.run(setup_network())

        network_thread = threading.Thread(target=run_network_setup, daemon=True)
        network_thread.start()

        # Monitor processes
        manager.monitor_processes()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
    finally:
        if "manager" in locals():
            manager.cleanup_all_resources()


if __name__ == "__main__":
    main()
