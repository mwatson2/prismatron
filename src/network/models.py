"""
Data models for network management.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class NetworkMode(str, Enum):
    """Network operation modes."""

    AP = "ap"
    CLIENT = "client"
    DISCONNECTED = "disconnected"


class WiFiSecurity(str, Enum):
    """WiFi security types."""

    OPEN = "open"
    WEP = "wep"
    WPA = "wpa"
    WPA2 = "wpa2"
    WPA3 = "wpa3"


@dataclass
class WiFiNetwork:
    """Represents a WiFi network from scan results."""

    ssid: str
    bssid: str
    signal_strength: int  # 0-100 percentage
    frequency: int  # MHz
    security: WiFiSecurity
    connected: bool = False


@dataclass
class NetworkStatus:
    """Current network status information."""

    mode: NetworkMode
    connected: bool
    interface: str
    ip_address: Optional[str] = None
    ssid: Optional[str] = None
    signal_strength: Optional[int] = None
    gateway: Optional[str] = None
    dns_servers: Optional[List[str]] = None


@dataclass
class APConfig:
    """Access Point configuration."""

    ssid: str = "prismatron"
    password: Optional[str] = None
    ip_address: str = "192.168.4.1"
    netmask: str = "255.255.255.0"
    dhcp_start: str = "192.168.4.2"
    dhcp_end: str = "192.168.4.100"
    channel: int = 6


@dataclass
class ClientConfig:
    """WiFi client configuration."""

    ssid: str
    password: Optional[str] = None
    auto_connect: bool = True
    hidden: bool = False


@dataclass
class NetworkConfig:
    """Network preferences and credentials (not current state)."""

    ap_config: APConfig
    client_config: Optional[ClientConfig] = None
