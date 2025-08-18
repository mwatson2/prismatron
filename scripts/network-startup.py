#!/usr/bin/env python3
"""
Network startup script for Prismatron system.
Manages network configuration on system boot.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.network import NetworkManager
from src.network.models import NetworkMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("/var/log/prismatron-network.log")],
)
logger = logging.getLogger(__name__)


async def startup_network():
    """Initialize network configuration on startup."""
    logger.info("Starting Prismatron network management...")

    try:
        # Initialize network manager
        network_manager = NetworkManager()

        # Get current status
        status = await network_manager.get_status()
        logger.info(f"Current network status: {status.mode}, connected: {status.connected}")

        # Get saved configuration
        config = network_manager.get_config()
        logger.info(f"Configured startup mode: {config.startup_mode}")

        # Apply startup configuration based on saved settings
        if config.startup_mode == NetworkMode.AP:
            logger.info("Enabling AP mode on startup...")
            success = await network_manager.enable_ap_mode()
            if success:
                logger.info("AP mode enabled successfully")
            else:
                logger.error("Failed to enable AP mode")

        elif config.startup_mode == NetworkMode.CLIENT and config.client_config:
            logger.info(f"Attempting to connect to saved network: {config.client_config.ssid}")
            success = await network_manager.connect_wifi(config.client_config.ssid, config.client_config.password)
            if success:
                logger.info(f"Connected to {config.client_config.ssid} successfully")
            else:
                logger.warning(f"Failed to connect to {config.client_config.ssid}, falling back to AP mode")
                # Fallback to AP mode if client connection fails
                await network_manager.enable_ap_mode()
                logger.info("Fallback AP mode enabled")
        else:
            logger.info("No specific startup configuration, using current state")

        # Final status check
        final_status = await network_manager.get_status()
        logger.info(f"Final network status: {final_status.mode}, connected: {final_status.connected}")
        if final_status.ip_address:
            logger.info(f"IP address: {final_status.ip_address}")

    except Exception as e:
        logger.error(f"Network startup failed: {e}")
        # Try to enable AP mode as last resort
        try:
            network_manager = NetworkManager()
            await network_manager.enable_ap_mode()
            logger.info("Emergency AP mode enabled")
        except Exception as emergency_error:
            logger.error(f"Emergency AP mode failed: {emergency_error}")
            sys.exit(1)


if __name__ == "__main__":
    # Wait a bit for NetworkManager to be fully ready
    time.sleep(2)

    try:
        asyncio.run(startup_network())
        logger.info("Network startup completed successfully")
    except KeyboardInterrupt:
        logger.info("Network startup interrupted")
    except Exception as e:
        logger.error(f"Network startup error: {e}")
        sys.exit(1)
