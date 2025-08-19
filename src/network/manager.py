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
        """Load network configuration from file."""
        try:
            if Path(self.CONFIG_FILE).exists():
                with open(self.CONFIG_FILE) as f:
                    data = json.load(f)
                    return NetworkConfig(
                        mode=NetworkMode(data.get("mode", "ap")),
                        ap_config=APConfig(**data.get("ap_config", {})),
                        client_config=ClientConfig(**data["client_config"]) if data.get("client_config") else None,
                        startup_mode=NetworkMode(data.get("startup_mode", "ap")),
                    )
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")

        return NetworkConfig(mode=NetworkMode.AP, ap_config=APConfig(), startup_mode=NetworkMode.AP)

    def _save_config(self) -> None:
        """Save network configuration to file."""
        try:
            Path(self.CONFIG_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(asdict(self.config), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise NetworkManagerError(f"Failed to save configuration: {e}") from e

    async def _run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command asynchronously."""
        try:
            logger.debug(f"Running command: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)

            if check and result.returncode != 0:
                error_msg = stderr.decode().strip() or stdout.decode().strip()
                logger.error(f"Command failed: {' '.join(cmd)}, error: {error_msg}")
                raise NetworkManagerError(f"Command failed: {error_msg}")

            return result
        except Exception as e:
            if isinstance(e, NetworkManagerError):
                raise
            logger.error(f"Failed to run command {' '.join(cmd)}: {e}")
            raise NetworkManagerError(f"Failed to execute command: {e}") from e

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
                await self._run_command(["nmcli", "device", "wifi", "rescan"])
                logger.debug("WiFi rescan completed")
            except NetworkManagerError as e:
                logger.warning(f"WiFi rescan failed (using cached results): {e}")

            # Get scan results
            result = await self._run_command(
                ["nmcli", "--terse", "--fields", "SSID,BSSID,SIGNAL,FREQ,SECURITY,ACTIVE", "device", "wifi", "list"]
            )

            networks = []
            seen_ssids = set()

            for line in result.stdout.decode().strip().split("\n"):
                if not line:
                    continue

                parts = line.split(":")
                if len(parts) >= 6:
                    ssid = parts[0].strip()
                    bssid = parts[1].strip()
                    signal = int(parts[2]) if parts[2].isdigit() else 0
                    frequency = int(parts[3]) if parts[3].isdigit() else 0
                    security_str = parts[4].strip().lower()
                    active = parts[5].strip() == "yes"

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

    async def connect_wifi(self, ssid: str, password: Optional[str] = None) -> bool:
        """Connect to a WiFi network."""
        try:
            # Disconnect current connections first
            await self.disconnect()

            # Build connection command
            cmd = ["nmcli", "device", "wifi", "connect", ssid]
            if password:
                cmd.extend(["password", password])
            cmd.extend(["ifname", self.interface])

            await self._run_command(cmd)

            # Update configuration
            self.config.client_config = ClientConfig(ssid=ssid, password=password)
            self.config.mode = NetworkMode.CLIENT
            self._save_config()

            logger.info(f"Successfully connected to WiFi: {ssid}")
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
                    await self._run_command(["nmcli", "connection", "down", connection.strip()])

            logger.info("Disconnected from network")
            return True

        except Exception as e:
            logger.error(f"Failed to disconnect: {e}")
            return False

    async def enable_ap_mode(self) -> bool:
        """Enable WiFi access point mode."""
        try:
            # Disconnect from any current connections
            await self.disconnect()

            # Create hostapd configuration
            ap_config = self.config.ap_config

            # Create connection with nmcli
            cmd = [
                "nmcli",
                "connection",
                "add",
                "type",
                "wifi",
                "ifname",
                self.interface,
                "con-name",
                f"{ap_config.ssid}-hotspot",
                "autoconnect",
                "no",
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
                    ["802-11-wireless-security.key-mgmt", "wpa-psk", "802-11-wireless-security.psk", ap_config.password]
                )

            await self._run_command(cmd)

            # Activate the connection
            await self._run_command(["nmcli", "connection", "up", f"{ap_config.ssid}-hotspot"])

            # Update configuration
            self.config.mode = NetworkMode.AP
            self._save_config()

            logger.info(f"AP mode enabled: {ap_config.ssid}")
            return True

        except Exception as e:
            logger.error(f"Failed to enable AP mode: {e}")
            raise NetworkManagerError(f"AP mode activation failed: {e}") from e

    async def disable_ap_mode(self) -> bool:
        """Disable WiFi access point mode."""
        try:
            # Disconnect AP connection
            ap_name = f"{self.config.ap_config.ssid}-hotspot"
            await self._run_command(["nmcli", "connection", "down", ap_name], check=False)

            # Delete AP connection
            await self._run_command(["nmcli", "connection", "delete", ap_name], check=False)

            logger.info("AP mode disabled")
            return True

        except Exception as e:
            logger.error(f"Failed to disable AP mode: {e}")
            return False

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
