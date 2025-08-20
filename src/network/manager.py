"""
NetworkManager wrapper for WiFi AP and client mode management.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import APConfig, ClientConfig, NetworkConfig, NetworkMode, NetworkStatus, WiFiNetwork, WiFiSecurity

logger = logging.getLogger(__name__)


class NetworkManagerError(Exception):
    """Base exception for network management errors."""


class NetworkManager:
    """Manages WiFi connections and AP mode using NetworkManager."""

    CONFIG_FILE = "/etc/prismatron/network-config.json"
    AP_CONNECTION_NAME = "prismatron-ap"  # Consistent connection name for AP mode

    def __init__(self):
        self.config = self._load_config()
        self.interface = self._detect_wifi_interface()

    def _detect_wifi_interface(self) -> str:
        """Detect the WiFi interface name."""
        try:
            result = subprocess.run(
                ["nmcli", "--terse", "--fields", "DEVICE,TYPE", "device", "status"],
                capture_output=True,
                text=True,
                check=True,
            )

            for line in result.stdout.strip().split("\n"):
                if line:
                    device, device_type = line.split(":")
                    if device_type == "wifi":
                        logger.info(f"Detected WiFi interface: {device}")
                        return device

            # Fallback to common names
            for interface in ["wlP1p1s0", "wlan0", "wlo1"]:
                try:
                    subprocess.run(["nmcli", "device", "show", interface], capture_output=True, check=True)
                    logger.info(f"Found WiFi interface: {interface}")
                    return interface
                except subprocess.CalledProcessError:
                    continue

            raise NetworkManagerError("No WiFi interface found")

        except Exception as e:
            logger.warning(f"Failed to detect WiFi interface: {e}, using fallback wlP1p1s0")
            return "wlP1p1s0"

    def _load_config(self) -> NetworkConfig:
        """Load network preferences from file."""
        try:
            if Path(self.CONFIG_FILE).exists():
                with open(self.CONFIG_FILE) as f:
                    data = json.load(f)
                    return NetworkConfig(
                        ap_config=APConfig(**data.get("ap_config", {})),
                        client_config=ClientConfig(**data["client_config"]) if data.get("client_config") else None,
                    )
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")

        return NetworkConfig(ap_config=APConfig())

    def _save_config(self) -> None:
        """Save network preferences to file."""
        try:
            Path(self.CONFIG_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(asdict(self.config), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            # Don't raise - preferences saving is not critical for operation
            logger.warning("Continuing without saving preferences")

    async def _run_command(
        self, cmd: List[str], check: bool = True, try_sudo: bool = False
    ) -> subprocess.CompletedProcess:
        """Run a shell command asynchronously.

        Args:
            cmd: Command and arguments to run
            check: If True, raise exception on non-zero exit code
            try_sudo: If True, try with sudo if regular command fails due to permissions
        """
        try:
            logger.debug(f"Running command: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)

            if check and result.returncode != 0:
                error_msg = stderr.decode().strip() or stdout.decode().strip()

                # Try sudo if requested and permission error detected
                if try_sudo and ("Insufficient privileges" in error_msg or "not authorized" in error_msg.lower()):
                    logger.debug(f"Trying command with sudo: {' '.join(cmd)}")
                    sudo_cmd = ["sudo"] + cmd
                    return await self._run_command(sudo_cmd, check=check, try_sudo=False)  # Avoid infinite recursion

                logger.error(f"Command failed: {' '.join(cmd)}, error: {error_msg}")
                raise NetworkManagerError(f"Command failed: {error_msg}")

            return result
        except Exception as e:
            if isinstance(e, NetworkManagerError):
                raise
            logger.error(f"Failed to run command {' '.join(cmd)}: {e}")
            raise NetworkManagerError(f"Failed to execute command: {e}") from e

    async def _restart_avahi(self) -> None:
        """Restart avahi-daemon to update .local address advertising."""
        try:
            logger.debug("Restarting avahi-daemon for network change")
            await self._run_command(["sudo", "systemctl", "restart", "avahi-daemon"], try_sudo=False)
            logger.debug("Avahi daemon restarted successfully")
        except Exception as e:
            logger.warning(f"Failed to restart avahi-daemon: {e}")
            # Don't raise - avahi restart is not critical for network operation

    async def get_status(self) -> NetworkStatus:
        """Get current network status."""
        try:
            # Get interface status
            result = await self._run_command(["nmcli", "device", "show", self.interface])
            output = result.stdout.decode()

            # Parse status
            connected = "connected" in output.lower()
            ip_address = None
            ssid = None
            signal_strength = None
            gateway = None
            dns_servers = []

            for line in output.split("\n"):
                line = line.strip()
                if "IP4.ADDRESS" in line and ip_address is None:
                    ip_address = line.split(":", 1)[1].strip().split("/")[0]
                elif "GENERAL.CONNECTION" in line and ssid is None:
                    ssid = line.split(":", 1)[1].strip()
                elif "IP4.GATEWAY" in line:
                    gateway = line.split(":", 1)[1].strip()
                elif "IP4.DNS" in line:
                    dns = line.split(":", 1)[1].strip()
                    if dns:
                        dns_servers.append(dns)

            # Determine mode
            if connected and ssid == self.config.ap_config.ssid:
                mode = NetworkMode.AP
            elif connected:
                mode = NetworkMode.CLIENT
            else:
                mode = NetworkMode.DISCONNECTED

            return NetworkStatus(
                mode=mode,
                connected=connected,
                interface=self.interface,
                ip_address=ip_address,
                ssid=ssid,
                signal_strength=signal_strength,
                gateway=gateway,
                dns_servers=dns_servers or None,
            )

        except Exception as e:
            logger.error(f"Failed to get network status: {e}")
            return NetworkStatus(mode=NetworkMode.DISCONNECTED, connected=False, interface=self.interface)

    async def scan_wifi(self) -> List[WiFiNetwork]:
        """Scan for available WiFi networks."""
        try:
            # Try to trigger scan, but continue even if it fails due to permissions
            try:
                await self._run_command(["nmcli", "device", "wifi", "rescan"], check=False, try_sudo=True)
                logger.debug("WiFi rescan completed")
            except Exception:
                logger.debug("WiFi rescan not available (using cached results)")

            # Get scan results
            result = await self._run_command(
                ["nmcli", "--terse", "--fields", "SSID,BSSID,SIGNAL,FREQ,SECURITY,ACTIVE", "device", "wifi", "list"]
            )

            networks = []
            seen_ssids = set()

            for line in result.stdout.decode().strip().split("\n"):
                if not line:
                    continue

                # Handle escaped colons in BSSID by replacing them temporarily
                line = line.replace(r"\:", "COLON_PLACEHOLDER")
                parts = line.split(":")

                if len(parts) >= 6:
                    # Restore colons in BSSID
                    ssid = parts[0].strip()
                    bssid = parts[1].replace("COLON_PLACEHOLDER", ":").strip()
                    signal = int(parts[2]) if parts[2].isdigit() else 0

                    # Parse frequency field (e.g., "2437 MHz" or "5280 MHz")
                    freq_str = parts[3].strip()
                    frequency = 0
                    if freq_str:
                        freq_parts = freq_str.split()
                        if freq_parts and freq_parts[0].isdigit():
                            frequency = int(freq_parts[0])

                    security_str = parts[4].strip().lower()
                    active = parts[5].strip().lower() == "yes"

                    # Skip duplicates and empty SSIDs
                    if not ssid or ssid in seen_ssids:
                        continue
                    seen_ssids.add(ssid)

                    # Determine security type
                    if not security_str or security_str == "--":
                        security = WiFiSecurity.OPEN
                    elif "wpa3" in security_str:
                        security = WiFiSecurity.WPA3
                    elif "wpa2" in security_str or "wpa" in security_str:
                        security = WiFiSecurity.WPA2
                    elif "wep" in security_str:
                        security = WiFiSecurity.WEP
                    else:
                        security = WiFiSecurity.WPA2  # Default fallback

                    networks.append(
                        WiFiNetwork(
                            ssid=ssid,
                            bssid=bssid,
                            signal_strength=signal,
                            frequency=frequency,
                            security=security,
                            connected=active,
                        )
                    )

            # Sort by signal strength
            networks.sort(key=lambda n: n.signal_strength, reverse=True)
            return networks

        except Exception as e:
            logger.error(f"Failed to scan WiFi: {e}")
            raise NetworkManagerError(f"WiFi scan failed: {e}") from e

    async def connect_wifi(self, ssid: str, password: Optional[str] = None, persist: bool = True) -> bool:
        """Connect to a WiFi network.

        Args:
            ssid: Network SSID to connect to
            password: Network password (if required)
            persist: If True, connection will auto-connect on boot (default: True)
        """
        try:
            # Disconnect current connections first
            await self.disconnect()

            # Check if a connection profile already exists for this SSID
            check_result = await self._run_command(["nmcli", "connection", "show", ssid], check=False)

            if check_result.returncode == 0:
                # Connection profile exists, use it directly
                logger.info(f"Using existing connection profile for {ssid}")
                cmd = ["nmcli", "connection", "up", ssid]

                # If a new password is provided, update the profile first
                if password:
                    logger.info("Updating connection password")
                    await self._run_command(
                        [
                            "nmcli",
                            "connection",
                            "modify",
                            ssid,
                            "802-11-wireless-security.psk",
                            password,
                        ],
                        try_sudo=True,
                    )
            else:
                # No existing profile, create new connection
                logger.info(f"Creating new connection profile for {ssid}")
                cmd = ["nmcli", "device", "wifi", "connect", ssid]
                if password:
                    cmd.extend(["password", password])
                cmd.extend(["ifname", self.interface])

            await self._run_command(cmd, try_sudo=True)

            # Set autoconnect and priority if needed
            if persist:
                await self._run_command(
                    [
                        "nmcli",
                        "connection",
                        "modify",
                        ssid,
                        "connection.autoconnect",
                        "yes",
                        "connection.autoconnect-priority",
                        "100",  # Higher priority than AP mode
                    ],
                    try_sudo=True,
                )
                logger.info(f"Set {ssid} to auto-connect on boot")

            # Wait a moment and verify connection is established
            await asyncio.sleep(2)  # Give NetworkManager time to establish connection

            # Verify we're actually connected to the target network
            status = await self.get_status()
            if not status.connected or status.ssid != ssid:
                raise NetworkManagerError(f"Connection verification failed - not connected to {ssid}")

            # Save client credentials for future use
            self.config.client_config = ClientConfig(ssid=ssid, password=password)
            self._save_config()

            # Restart avahi to update .local address advertising on new network
            await self._restart_avahi()

            logger.info(f"Successfully connected to WiFi: {ssid} (persist={persist})")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to WiFi {ssid}: {e}")
            raise NetworkManagerError(f"WiFi connection failed: {e}") from e

    async def disconnect(self) -> bool:
        """Disconnect from current network."""
        try:
            # Get current connection
            result = await self._run_command(
                ["nmcli", "--terse", "--fields", "NAME", "connection", "show", "--active"], check=False
            )

            connections = result.stdout.decode().strip().split("\n")
            for connection in connections:
                if connection.strip():
                    await self._run_command(["nmcli", "connection", "down", connection.strip()], try_sudo=True)

            logger.info("Disconnected from network")
            return True

        except Exception as e:
            logger.error(f"Failed to disconnect: {e}")
            return False

    async def enable_ap_mode(self, persist: bool = True) -> bool:
        """Enable WiFi access point mode.

        Args:
            persist: If True, AP mode will auto-start on boot (default: True)
        """
        try:
            # Disconnect from any current connections
            await self.disconnect()

            # Create hostapd configuration
            ap_config = self.config.ap_config

            # Check if AP connection already exists
            check_result = await self._run_command(
                ["nmcli", "connection", "show", self.AP_CONNECTION_NAME], check=False
            )

            if check_result.returncode == 0:
                # Connection exists, update it and activate
                logger.info("AP connection exists, updating settings...")

                # Update autoconnect setting
                await self._run_command(
                    [
                        "nmcli",
                        "connection",
                        "modify",
                        self.AP_CONNECTION_NAME,
                        "connection.autoconnect",
                        "yes" if persist else "no",
                        "connection.autoconnect-priority",
                        "50",  # Lower priority than client connections
                    ],
                    try_sudo=True,
                )
            else:
                # Create new connection with nmcli
                cmd = [
                    "nmcli",
                    "connection",
                    "add",
                    "type",
                    "wifi",
                    "ifname",
                    self.interface,
                    "con-name",
                    self.AP_CONNECTION_NAME,
                    "autoconnect",
                    "yes" if persist else "no",  # KEY CHANGE: Enable persistence by default
                    "connection.autoconnect-priority",
                    "50",  # Lower priority so client connections are preferred
                    "ssid",
                    ap_config.ssid,
                    "mode",
                    "ap",
                    "802-11-wireless.band",
                    "bg",
                    "802-11-wireless.channel",
                    str(ap_config.channel),
                    "ipv4.method",
                    "shared",
                    "ipv4.address",
                    f"{ap_config.ip_address}/24",
                ]

                # Add password if configured
                if ap_config.password:
                    cmd.extend(
                        [
                            "802-11-wireless-security.key-mgmt",
                            "wpa-psk",
                            "802-11-wireless-security.psk",
                            ap_config.password,
                        ]
                    )

                await self._run_command(cmd, try_sudo=True)

            # Activate the connection
            await self._run_command(["nmcli", "connection", "up", self.AP_CONNECTION_NAME], try_sudo=True)

            # Save preferences (AP config might have been updated)
            self._save_config()

            # Restart avahi to advertise .local address on AP network
            await self._restart_avahi()

            logger.info(f"AP mode enabled: {ap_config.ssid} (persist={persist})")
            return True

        except Exception as e:
            logger.error(f"Failed to enable AP mode: {e}")
            raise NetworkManagerError(f"AP mode activation failed: {e}") from e

    async def disable_ap_mode(self) -> bool:
        """Disable WiFi access point mode and remove autoconnect."""
        try:
            # Disconnect AP connection
            await self._run_command(
                ["nmcli", "connection", "down", self.AP_CONNECTION_NAME], check=False, try_sudo=True
            )

            # Disable autoconnect instead of deleting (preserves settings)
            await self._run_command(
                ["nmcli", "connection", "modify", self.AP_CONNECTION_NAME, "connection.autoconnect", "no"],
                check=False,
                try_sudo=True,
            )

            logger.info("AP mode disabled")
            return True

        except Exception as e:
            logger.error(f"Failed to disable AP mode: {e}")
            return False

    async def ensure_connectivity(self, startup_delay: int = 10) -> str:
        """Ensure network connectivity on startup.

        Waits for system to establish connections, then checks if connected.
        If no connection exists, enables AP mode as fallback.

        Args:
            startup_delay: Seconds to wait for system to establish connections

        Returns:
            "client" if connected to WiFi, "ap" if AP mode was enabled
        """
        logger.info(f"Checking network connectivity in {startup_delay} seconds...")
        await asyncio.sleep(startup_delay)

        try:
            # Check current network status
            status = await self.get_status()

            if status.connected and status.mode == NetworkMode.CLIENT:
                logger.info(f"Network connectivity OK - connected to {status.ssid} ({status.ip_address})")
                return "client"

            elif status.mode == NetworkMode.AP:
                logger.info(f"Already in AP mode - network available as {self.config.ap_config.ssid}")
                return "ap"

            else:
                logger.info("No network connectivity detected - enabling AP mode as fallback")
                success = await self.enable_ap_mode(persist=True)

                if success:
                    logger.info(f"AP mode enabled - network available as {self.config.ap_config.ssid}")
                    return "ap"
                else:
                    logger.error("Failed to enable AP mode - no network connectivity available")
                    return "disconnected"

        except Exception as e:
            logger.error(f"Error checking network connectivity: {e}")
            try:
                # Try to enable AP mode as last resort
                logger.info("Attempting AP mode as emergency fallback")
                await self.enable_ap_mode(persist=True)
                return "ap"
            except Exception:
                logger.error("Emergency AP mode failed - no network available")
                return "disconnected"

    async def switch_to_client_mode(self, ssid: str, password: Optional[str] = None) -> bool:
        """Switch from AP mode to client mode."""
        try:
            # Disable AP mode first
            await self.disable_ap_mode()

            # Connect to client network
            return await self.connect_wifi(ssid, password)

        except Exception as e:
            logger.error(f"Failed to switch to client mode: {e}")
            raise NetworkManagerError(f"Failed to switch to client mode: {e}") from e

    async def switch_to_ap_mode(self) -> bool:
        """Switch from client mode to AP mode."""
        try:
            # Disconnect from client network
            await self.disconnect()

            # Enable AP mode
            return await self.enable_ap_mode()

        except Exception as e:
            logger.error(f"Failed to switch to AP mode: {e}")
            raise NetworkManagerError(f"Failed to switch to AP mode: {e}") from e

    def get_config(self) -> NetworkConfig:
        """Get current network configuration."""
        return self.config

    def update_ap_config(self, **kwargs) -> None:
        """Update AP configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config.ap_config, key):
                setattr(self.config.ap_config, key, value)
        self._save_config()

    async def set_startup_preference(self, prefer_ap: bool = False) -> bool:
        """Set network startup preference.

        Args:
            prefer_ap: If True, prefer AP mode on boot. If False, prefer saved WiFi.

        Returns:
            True if preference was set successfully
        """
        try:
            if prefer_ap:
                # Set AP mode to higher priority
                await self._run_command(
                    [
                        "nmcli",
                        "connection",
                        "modify",
                        self.AP_CONNECTION_NAME,
                        "connection.autoconnect-priority",
                        "200",
                    ],
                    check=False,
                )

                # Lower priority for all client connections
                result = await self._run_command(
                    ["nmcli", "--terse", "--fields", "NAME,TYPE", "connection", "show"], check=False
                )

                for line in result.stdout.decode().strip().split("\n"):
                    if "802-11-wireless" in line and self.AP_CONNECTION_NAME not in line:
                        conn_name = line.split(":")[0]
                        await self._run_command(
                            ["nmcli", "connection", "modify", conn_name, "connection.autoconnect-priority", "50"],
                            check=False,
                        )

                logger.info("Set startup preference to AP mode")
            else:
                # Set AP mode to lower priority
                await self._run_command(
                    ["nmcli", "connection", "modify", self.AP_CONNECTION_NAME, "connection.autoconnect-priority", "50"],
                    check=False,
                )

                # Higher priority for client connections
                if self.config.client_config and self.config.client_config.ssid:
                    await self._run_command(
                        [
                            "nmcli",
                            "connection",
                            "modify",
                            self.config.client_config.ssid,
                            "connection.autoconnect-priority",
                            "100",
                        ],
                        check=False,
                    )

                logger.info("Set startup preference to client mode")

            return True

        except Exception as e:
            logger.error(f"Failed to set startup preference: {e}")
            return False

    async def get_saved_connections(self) -> List[str]:
        """Get list of saved WiFi connections.

        Returns:
            List of saved connection names (SSIDs)
        """
        try:
            result = await self._run_command(
                ["nmcli", "--terse", "--fields", "NAME,TYPE", "connection", "show"], check=False
            )

            connections = []
            for line in result.stdout.decode().strip().split("\n"):
                if "802-11-wireless" in line:
                    conn_name = line.split(":")[0]
                    if conn_name != self.AP_CONNECTION_NAME:
                        connections.append(conn_name)

            return connections

        except Exception as e:
            logger.error(f"Failed to get saved connections: {e}")
            return []
