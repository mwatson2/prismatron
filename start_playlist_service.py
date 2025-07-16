#!/usr/bin/env python3
"""
Start the Playlist Synchronization Service.

This script starts the playlist synchronization service that manages
playlist state across all processes using Unix domain sockets.
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.playlist_sync import PlaylistSyncService

logger = logging.getLogger(__name__)

# Global service instance for signal handling
service_instance: PlaylistSyncService = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global service_instance
    logger.info(f"Received signal {signum}, shutting down playlist service...")
    if service_instance:
        service_instance.stop()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Start the Prismatron playlist synchronization service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default socket path
  python start_playlist_service.py

  # Start with custom socket path
  python start_playlist_service.py --socket /tmp/custom_playlist.sock

  # Enable verbose logging
  python start_playlist_service.py --verbose
        """,
    )

    parser.add_argument(
        "--socket",
        default="/tmp/prismatron_playlist.sock",
        help="Unix domain socket path (default: /tmp/prismatron_playlist.sock)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        global service_instance
        
        # Create and start the service
        service_instance = PlaylistSyncService(socket_path=args.socket)
        
        logger.info(f"Starting playlist synchronization service on {args.socket}")
        if service_instance.start():
            logger.info("Playlist synchronization service started successfully")
            
            # Keep the service running
            try:
                while service_instance.running:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
        else:
            logger.error("Failed to start playlist synchronization service")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error running playlist synchronization service: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if service_instance:
            service_instance.stop()
            logger.info("Playlist synchronization service stopped")


if __name__ == "__main__":
    main()