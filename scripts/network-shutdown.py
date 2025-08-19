#!/usr/bin/env python3
"""
Network shutdown script for Prismatron system.
Cleans up network configuration on system shutdown.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, "/mnt/dev/prismatron/src")

from src.network import NetworkManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("/var/log/prismatron-network.log")],
)
logger = logging.getLogger(__name__)


async def shutdown_network():
    """Clean up network configuration on shutdown."""
    logger.info("Shutting down Prismatron network management...")

    try:
        # Initialize network manager
        network_manager = NetworkManager()

        # Get current status
        status = await network_manager.get_status()
        logger.info(f"Current network status: {status.mode}, connected: {status.connected}")

        # If in AP mode, disable it cleanly
        if status.mode.value == "ap":
            logger.info("Disabling AP mode...")
            success = await network_manager.disable_ap_mode()
            if success:
                logger.info("AP mode disabled successfully")
            else:
                logger.warning("Failed to disable AP mode cleanly")

        logger.info("Network shutdown completed")

    except Exception as e:
        logger.error(f"Network shutdown error: {e}")
        # Don't exit with error on shutdown - just log it


if __name__ == "__main__":
    try:
        asyncio.run(shutdown_network())
    except KeyboardInterrupt:
        logger.info("Network shutdown interrupted")
    except Exception as e:
        logger.error(f"Network shutdown error: {e}")
