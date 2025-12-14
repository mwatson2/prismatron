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
    ETHERNET_CLIENT_NAME = "prismatron-ethernet-client"  # Ethernet connection for client mode
    BRIDGE_NAME = "prismatron-bridge"  # Bridge connection for AP mode

    def __init__(self):
        self.config = self._load_config()
        self.interface = self._detect_wifi_interface()
        self.ethernet_interface = self._detect_ethernet_interface()

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

    def _detect_ethernet_interface(self) -> str:
        """Detect the Ethernet interface name."""
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
                    if device_type == "ethernet":
                        logger.info(f"Detected Ethernet interface: {device}")
                        return device

            # Fallback to common names
            for interface in ["enP8p1s0", "eth0", "enp0s1"]:
                try:
                    subprocess.run(["nmcli", "device", "show", interface], capture_output=True, check=True)
                    logger.info(f"Found Ethernet interface: {interface}")
                    return interface
                except subprocess.CalledProcessError:
                    continue

            raise NetworkManagerError("No Ethernet interface found")

        except Exception as e:
            logger.warning(f"Failed to detect Ethernet interface: {e}, using fallback enP8p1s0")
            return "enP8p1s0"

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

            result = subprocess.CompletedProcess(
                cmd, process.returncode if process.returncode is not None else -1, stdout, stderr
            )

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

    async def _enable_ip_forwarding(self) -> None:
        """Enable IP forwarding for routing between WiFi and Ethernet."""
        try:
            logger.info("Enabling IP forwarding for WiFi-Ethernet routing")
            await self._run_command(["sudo", "sysctl", "-w", "net.ipv4.ip_forward=1"], try_sudo=False)
            logger.debug("IP forwarding enabled")
        except Exception as e:
            logger.warning(f"Failed to enable IP forwarding: {e}")

    async def _configure_ethernet_client_mode(self) -> None:
        """Configure Ethernet with DHCP server for WiFi client mode."""
        try:
            logger.info(f"Configuring Ethernet ({self.ethernet_interface}) for client mode with DHCP server")

            # Check if connection exists
            check_result = await self._run_command(
                ["nmcli", "connection", "show", self.ETHERNET_CLIENT_NAME], check=False
            )

            if check_result.returncode == 0:
                # Connection exists, just activate it
                logger.info("Ethernet client mode connection exists, activating...")
                await self._run_command(["nmcli", "connection", "up", self.ETHERNET_CLIENT_NAME], try_sudo=True)
            else:
                # Create new connection with shared IPv4 (provides DHCP)
                logger.info("Creating new Ethernet client mode connection...")
                cmd = [
                    "nmcli",
                    "connection",
                    "add",
                    "type",
                    "ethernet",
                    "ifname",
                    self.ethernet_interface,
                    "con-name",
                    self.ETHERNET_CLIENT_NAME,
                    "ipv4.method",
                    "shared",  # This enables DHCP server
                    "ipv4.addresses",
                    "192.168.100.1/24",
                    "connection.autoconnect",
                    "yes",
                ]
                await self._run_command(cmd, try_sudo=True)
                logger.info("Ethernet configured for client mode with DHCP at 192.168.100.1/24")

            # Enable IP forwarding so WiFi clients can reach Ethernet devices
            await self._enable_ip_forwarding()

        except Exception as e:
            logger.error(f"Failed to configure Ethernet for client mode: {e}")
            raise NetworkManagerError(f"Ethernet client mode configuration failed: {e}") from e

    async def _configure_ethernet_ap_mode(self, ap_subnet_ip: str) -> None:
        """Configure Ethernet to share the same subnet as WiFi AP.

        Since most WiFi drivers don't support true bridging, we configure Ethernet
        on the same subnet as the AP and enable routing between them.

        Args:
            ap_subnet_ip: IP address from AP config (e.g., "192.168.4.1")
        """
        try:
            logger.info(f"Configuring Ethernet ({self.ethernet_interface}) for AP mode on same subnet")

            # Use a different IP in the same subnet for Ethernet (e.g., 192.168.4.2)
            # Parse the AP IP and increment it
            ip_parts = ap_subnet_ip.split(".")
            ethernet_ip = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.{int(ip_parts[3]) + 1}"

            ethernet_ap_name = f"{self.ETHERNET_CLIENT_NAME}-ap"

            # Check if connection exists
            check_result = await self._run_command(["nmcli", "connection", "show", ethernet_ap_name], check=False)

            if check_result.returncode == 0:
                # Connection exists, just activate it
                logger.info("Ethernet AP mode connection exists, activating...")
                await self._run_command(["nmcli", "connection", "up", ethernet_ap_name], try_sudo=True)
            else:
                # Create new connection on same subnet as AP
                logger.info(f"Creating Ethernet AP mode connection at {ethernet_ip}/24...")
                cmd = [
                    "nmcli",
                    "connection",
                    "add",
                    "type",
                    "ethernet",
                    "ifname",
                    self.ethernet_interface,
                    "con-name",
                    ethernet_ap_name,
                    "ipv4.method",
                    "shared",  # Provides DHCP server
                    "ipv4.addresses",
                    f"{ethernet_ip}/24",
                    "connection.autoconnect",
                    "no",  # Only activate when in AP mode
                ]
                await self._run_command(cmd, try_sudo=True)
                logger.info(f"Ethernet configured for AP mode at {ethernet_ip}/24")

            # Enable IP forwarding for routing between WiFi and Ethernet
            await self._enable_ip_forwarding()

        except Exception as e:
            logger.error(f"Failed to configure Ethernet for AP mode: {e}")
            raise NetworkManagerError(f"Ethernet AP mode configuration failed: {e}") from e

    async def _disable_ethernet_client_mode(self) -> None:
        """Disable Ethernet client mode connection."""
        try:
            logger.info("Disabling Ethernet client mode")
            await self._run_command(
                ["nmcli", "connection", "down", self.ETHERNET_CLIENT_NAME], check=False, try_sudo=True
            )
        except Exception as e:
            logger.warning(f"Failed to disable Ethernet client mode: {e}")

    async def _disable_ethernet_ap_mode(self) -> None:
        """Disable Ethernet AP mode connection."""
        try:
            logger.info("Disabling Ethernet AP mode")
            ethernet_ap_name = f"{self.ETHERNET_CLIENT_NAME}-ap"
            await self._run_command(["nmcli", "connection", "down", ethernet_ap_name], check=False, try_sudo=True)
        except Exception as e:
            logger.warning(f"Failed to disable Ethernet AP mode: {e}")

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

            # Configure Ethernet for client mode (with DHCP server for WLED)
            try:
                await self._configure_ethernet_client_mode()
            except Exception as e:
                logger.warning(f"Failed to configure Ethernet for client mode: {e}")
                # Don't fail the WiFi connection if Ethernet setup fails

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
            # Disable Ethernet connections first
            await self._disable_ethernet_client_mode()
            await self._disable_ethernet_ap_mode()

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

    async def enable_ap_mode(self, persist: bool = False) -> bool:
        """Enable WiFi access point mode.

        Args:
            persist: If True, AP mode will auto-start on boot (default: False).
                     Note: We default to False because AP autoconnect can prevent
                     the system from connecting to saved client networks on boot.
                     AP fallback is handled by the system-level NetworkManager
                     dispatcher script at /etc/NetworkManager/dispatcher.d/99-prismatron-ap-fallback
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

            # Configure Ethernet for AP mode (same subnet as WiFi AP for WLED access)
            try:
                await self._configure_ethernet_ap_mode(ap_config.ip_address)
            except Exception as e:
                logger.warning(f"Failed to configure Ethernet for AP mode: {e}")
                # Don't fail AP mode if Ethernet setup fails

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
            # Disable Ethernet AP mode first
            await self._disable_ethernet_ap_mode()

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

    async def get_connectivity_status(self) -> str:
        """Report current network connectivity status.

        Note: AP fallback on boot is handled by the system-level NetworkManager
        dispatcher script at /etc/NetworkManager/dispatcher.d/99-prismatron-ap-fallback

        Returns:
            "client" if connected to WiFi, "ap" if in AP mode, "disconnected" otherwise
        """
        try:
            status = await self.get_status()

            if status.connected and status.mode == NetworkMode.CLIENT:
                logger.info(f"Network status: client mode - {status.ssid} ({status.ip_address})")
                return "client"

            elif status.mode == NetworkMode.AP:
                logger.info(f"Network status: AP mode - {self.config.ap_config.ssid}")
                return "ap"

            else:
                logger.info("Network status: disconnected")
                return "disconnected"

        except Exception as e:
            logger.error(f"Error checking network status: {e}")
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

    async def switch_to_ap_mode(self, persist: bool = False) -> bool:
        """Switch from client mode to AP mode.

        Args:
            persist: If True, AP mode will auto-start on boot (default: False)
        """
        try:
            # Disconnect from client network
            await self.disconnect()

            # Enable AP mode
            return await self.enable_ap_mode(persist=persist)

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
