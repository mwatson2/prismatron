#!/usr/bin/env python3
"""
Prismatron System Orchestrator.

Main entry point that coordinates startup and management of all system processes:
- Web API Server (FastAPI)
- Producer Process (Content rendering)
- Consumer Process (LED optimization and WLED communication)

This orchestrator ensures proper startup sequencing, monitors process health,
and handles graceful shutdown coordination.
"""

import argparse
import atexit
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

from src.consumer.consumer import ConsumerProcess
from src.core.control_state import ControlState, PlayState, SystemState
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
        self.control_state = ControlState()
        self.shutdown_requested = False
        
        # Clean up any orphaned shared memory from previous runs
        self._cleanup_orphaned_shared_memory()

        # Process startup coordination
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

            # Start web server first (API needed for producer control)
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

    def _start_web_server(self) -> bool:
        """Start the web API server process."""
        try:
            logger.info("Starting web server...")

            def web_server_worker():
                """Web server process worker."""
                try:
                    # Setup logging for subprocess
                    logging.basicConfig(
                        level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    )

                    # Signal ready
                    self.web_server_ready.set()

                    # Start server (blocks until shutdown)
                    run_server(
                        host=self.config.get("web_host", "0.0.0.0"),
                        port=self.config.get("web_port", 8000),
                        debug=self.config.get("debug", False),
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
                    # Setup logging for subprocess
                    logging.basicConfig(
                        level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    )

                    # Create consumer with configuration
                    consumer = ConsumerProcess(
                        wled_host=self.config.get("wled_host", "192.168.1.100"),
                        wled_port=self.config.get("wled_port", 4048),
                        diffusion_patterns_path=self.config.get("diffusion_patterns_path"),
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
                    # Setup logging for subprocess
                    logging.basicConfig(
                        level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    )

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
            process_order = ["producer", "consumer", "web_server"]

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
        while not self.shutdown_requested:
            try:
                # Check for control signals
                if self.control_state.is_restart_requested():
                    logger.info("Restart signal detected")
                    if self.restart_system():
                        logger.info("System restart completed successfully")
                    else:
                        logger.error("System restart failed")
                    continue

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
            # Only works on Linux with appropriate permissions
            os.system("sudo reboot")
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

    def cleanup_all_resources(self) -> None:
        """Comprehensive cleanup of all system resources."""
        try:
            logger.info("Performing comprehensive cleanup...")
            self.stop_all_processes()
            
            # Additional cleanup for any remaining shared memory
            self._cleanup_orphaned_shared_memory()
            
        except Exception as e:
            logger.error(f"Error during comprehensive cleanup: {e}")


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO

    # Clear the log file on startup
    log_file = "prismatron.log"
    try:
        with open(log_file, "w") as f:
            f.write("")  # Clear the file
    except Exception:
        pass  # If we can't clear it, continue anyway

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

    # Reduce noise from some modules
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def signal_handler(signum, frame, process_manager: ProcessManager) -> None:
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    process_manager.cleanup_all_resources()
    sys.exit(0)

def emergency_cleanup() -> None:
    """Emergency cleanup function for atexit."""
    try:
        from multiprocessing import shared_memory
        orphaned_memory = shared_memory.SharedMemory(name="prismatron_control")
        orphaned_memory.close()
        orphaned_memory.unlink()
        print("Emergency cleanup: Removed orphaned shared memory")
    except:
        pass  # Silently handle any errors during emergency cleanup


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prismatron LED Display System")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--web-host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--web-port", type=int, default=8000, help="Web server port")
    parser.add_argument("--wled-host", default="192.168.1.100", help="WLED controller IP")
    parser.add_argument("--wled-port", type=int, default=4048, help="WLED controller port")
    parser.add_argument("--content-dir", help="Default content directory to load")
    parser.add_argument("--diffusion-patterns", help="Path to diffusion patterns file")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)
    
    # Register emergency cleanup for all exit scenarios
    atexit.register(emergency_cleanup)

    # Create configuration
    config = {
        "debug": args.debug,
        "web_host": args.web_host,
        "web_port": args.web_port,
        "wled_host": args.wled_host,
        "wled_port": args.wled_port,
        "default_content_dir": args.content_dir,
        "diffusion_patterns_path": args.diffusion_patterns,
    }

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
