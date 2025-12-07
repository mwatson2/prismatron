"""
Unit tests for the NetworkManager.

Tests network status retrieval, WiFi scanning, connection management,
AP mode enabling/disabling, and configuration persistence.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.network.manager import NetworkManager, NetworkManagerError
from src.network.models import (
    APConfig,
    ClientConfig,
    NetworkConfig,
    NetworkMode,
    NetworkStatus,
    WiFiNetwork,
    WiFiSecurity,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for synchronous calls."""
    with patch("subprocess.run") as mock_run:
        # Default: successful wifi interface detection
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="wlan0:wifi\neth0:ethernet",
            stderr="",
        )
        yield mock_run


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    config_file = tmp_path / "network-config.json"
    return config_file


@pytest.fixture
def sample_network_config():
    """Create sample network configuration."""
    return {
        "ap_config": {
            "ssid": "prismatron",
            "password": None,
            "ip_address": "192.168.4.1",
            "netmask": "255.255.255.0",
            "dhcp_start": "192.168.4.2",
            "dhcp_end": "192.168.4.100",
            "channel": 6,
        },
        "client_config": None,
    }


@pytest.fixture
def network_manager_with_mocks(mock_subprocess_run):
    """Create a NetworkManager with mocked subprocess calls."""
    with patch.object(NetworkManager, "_load_config") as mock_load:
        mock_load.return_value = NetworkConfig(ap_config=APConfig())
        manager = NetworkManager()
        manager.interface = "wlan0"
        manager.ethernet_interface = "eth0"
        yield manager


# =============================================================================
# Model Tests
# =============================================================================


class TestNetworkModels:
    """Test network data models."""

    def test_network_mode_enum(self):
        """Test NetworkMode enum values."""
        assert NetworkMode.AP.value == "ap"
        assert NetworkMode.CLIENT.value == "client"
        assert NetworkMode.DISCONNECTED.value == "disconnected"

    def test_wifi_security_enum(self):
        """Test WiFiSecurity enum values."""
        assert WiFiSecurity.OPEN.value == "open"
        assert WiFiSecurity.WEP.value == "wep"
        assert WiFiSecurity.WPA.value == "wpa"
        assert WiFiSecurity.WPA2.value == "wpa2"
        assert WiFiSecurity.WPA3.value == "wpa3"

    def test_wifi_network_dataclass(self):
        """Test WiFiNetwork dataclass."""
        network = WiFiNetwork(
            ssid="TestNetwork",
            bssid="00:11:22:33:44:55",
            signal_strength=80,
            frequency=2437,
            security=WiFiSecurity.WPA2,
            connected=False,
        )

        assert network.ssid == "TestNetwork"
        assert network.bssid == "00:11:22:33:44:55"
        assert network.signal_strength == 80
        assert network.frequency == 2437
        assert network.security == WiFiSecurity.WPA2
        assert network.connected is False

    def test_network_status_dataclass(self):
        """Test NetworkStatus dataclass."""
        status = NetworkStatus(
            mode=NetworkMode.CLIENT,
            connected=True,
            interface="wlan0",
            ip_address="192.168.1.100",
            ssid="TestNetwork",
            signal_strength=80,
            gateway="192.168.1.1",
            dns_servers=["8.8.8.8", "8.8.4.4"],
        )

        assert status.mode == NetworkMode.CLIENT
        assert status.connected is True
        assert status.interface == "wlan0"
        assert status.ip_address == "192.168.1.100"

    def test_ap_config_defaults(self):
        """Test APConfig default values."""
        config = APConfig()

        assert config.ssid == "prismatron"
        assert config.password is None
        assert config.ip_address == "192.168.4.1"
        assert config.netmask == "255.255.255.0"
        assert config.channel == 6

    def test_client_config(self):
        """Test ClientConfig dataclass."""
        config = ClientConfig(
            ssid="TestNetwork",
            password="testpassword",
            auto_connect=True,
            hidden=False,
        )

        assert config.ssid == "TestNetwork"
        assert config.password == "testpassword"
        assert config.auto_connect is True
        assert config.hidden is False

    def test_network_config(self):
        """Test NetworkConfig dataclass."""
        ap_config = APConfig(ssid="test-ap")
        client_config = ClientConfig(ssid="test-client")
        config = NetworkConfig(ap_config=ap_config, client_config=client_config)

        assert config.ap_config.ssid == "test-ap"
        assert config.client_config.ssid == "test-client"


# =============================================================================
# NetworkManager Initialization Tests
# =============================================================================


class TestNetworkManagerInit:
    """Test NetworkManager initialization."""

    def test_init_creates_config(self, mock_subprocess_run):
        """Test that initialization creates config."""
        with patch.object(NetworkManager, "_load_config") as mock_load:
            mock_load.return_value = NetworkConfig(ap_config=APConfig())

            manager = NetworkManager()

            assert manager.config is not None
            assert manager.config.ap_config is not None

    def test_init_detects_wifi_interface(self, mock_subprocess_run):
        """Test WiFi interface detection during init."""
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="wlan0:wifi\neth0:ethernet",
            stderr="",
        )

        with patch.object(NetworkManager, "_load_config") as mock_load:
            mock_load.return_value = NetworkConfig(ap_config=APConfig())

            manager = NetworkManager()

            # Should have detected interface
            assert manager.interface is not None

    def test_init_fallback_interface(self, mock_subprocess_run):
        """Test fallback interface when detection fails."""
        mock_subprocess_run.side_effect = Exception("nmcli failed")

        with patch.object(NetworkManager, "_load_config") as mock_load:
            mock_load.return_value = NetworkConfig(ap_config=APConfig())

            manager = NetworkManager()

            # Should use fallback
            assert manager.interface == "wlP1p1s0"


# =============================================================================
# Configuration Tests
# =============================================================================


class TestNetworkManagerConfig:
    """Test configuration loading and saving."""

    def test_load_config_from_file(self, tmp_path, sample_network_config):
        """Test loading config from file."""
        config_file = tmp_path / "network-config.json"
        config_file.write_text(json.dumps(sample_network_config))

        with patch.object(NetworkManager, "CONFIG_FILE", str(config_file)), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="wlan0:wifi",
                stderr="",
            )

            manager = NetworkManager()

            assert manager.config.ap_config.ssid == "prismatron"

    def test_load_config_missing_file(self, tmp_path, mock_subprocess_run):
        """Test loading config when file doesn't exist."""
        config_file = tmp_path / "nonexistent.json"

        with patch.object(NetworkManager, "CONFIG_FILE", str(config_file)), patch.object(
            NetworkManager, "_load_config"
        ) as mock_load:
            mock_load.return_value = NetworkConfig(ap_config=APConfig())

            manager = NetworkManager()

            # Should use defaults
            assert manager.config.ap_config.ssid == "prismatron"

    def test_save_config(self, tmp_path, network_manager_with_mocks):
        """Test saving configuration."""
        config_file = tmp_path / "network-config.json"

        with patch.object(NetworkManager, "CONFIG_FILE", str(config_file)):
            network_manager_with_mocks._save_config()

            # File should exist
            assert config_file.exists()


# =============================================================================
# Status Tests
# =============================================================================


class TestNetworkStatus:
    """Test network status retrieval."""

    @pytest.mark.asyncio
    async def test_get_status_connected(self, network_manager_with_mocks):
        """Test getting status when connected."""
        nmcli_output = """
GENERAL.DEVICE:                         wlan0
GENERAL.TYPE:                           wifi
GENERAL.STATE:                          100 (connected)
GENERAL.CONNECTION:                     TestNetwork
IP4.ADDRESS[1]:                         192.168.1.100/24
IP4.GATEWAY:                            192.168.1.1
IP4.DNS[1]:                             8.8.8.8
"""

        async def mock_run_command(cmd, **kwargs):
            return MagicMock(stdout=nmcli_output.encode(), returncode=0)

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command):
            status = await network_manager_with_mocks.get_status()

            assert status.connected is True
            assert status.ip_address == "192.168.1.100"
            assert status.ssid == "TestNetwork"
            assert status.gateway == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_get_status_disconnected(self, network_manager_with_mocks):
        """Test getting status when disconnected."""
        # Note: The get_status() method checks if "connected" is in the output
        # So we need to make sure "connected" is NOT in the output for disconnected state
        nmcli_output = """
GENERAL.DEVICE:                         wlan0
GENERAL.TYPE:                           wifi
GENERAL.STATE:                          30 (unavailable)
"""

        async def mock_run_command(cmd, **kwargs):
            return MagicMock(stdout=nmcli_output.encode(), returncode=0)

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command):
            status = await network_manager_with_mocks.get_status()

            assert status.connected is False
            assert status.mode == NetworkMode.DISCONNECTED

    @pytest.mark.asyncio
    async def test_get_status_error_handling(self, network_manager_with_mocks):
        """Test status retrieval error handling."""

        async def mock_run_command(cmd, **kwargs):
            raise NetworkManagerError("Command failed")

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command):
            status = await network_manager_with_mocks.get_status()

            # Should return disconnected status on error
            assert status.mode == NetworkMode.DISCONNECTED
            assert status.connected is False


# =============================================================================
# WiFi Scanning Tests
# =============================================================================


class TestWiFiScanning:
    """Test WiFi network scanning."""

    @pytest.mark.asyncio
    async def test_scan_wifi_returns_networks(self, network_manager_with_mocks):
        """Test scanning for WiFi networks."""
        scan_output = """TestNetwork1:00\\:11\\:22\\:33\\:44\\:55:80:2437 MHz:WPA2:no
TestNetwork2:00\\:11\\:22\\:33\\:44\\:66:60:5180 MHz:WPA3:yes
OpenNetwork:00\\:11\\:22\\:33\\:44\\:77:40:2412 MHz:--:no"""

        call_count = 0

        async def mock_run_command(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if "rescan" in cmd:
                return MagicMock(stdout=b"", returncode=0)
            return MagicMock(stdout=scan_output.encode(), returncode=0)

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command):
            networks = await network_manager_with_mocks.scan_wifi()

            assert len(networks) == 3
            # Should be sorted by signal strength
            assert networks[0].ssid == "TestNetwork1"
            assert networks[0].signal_strength == 80

    @pytest.mark.asyncio
    async def test_scan_wifi_parses_security(self, network_manager_with_mocks):
        """Test that security types are parsed correctly."""
        scan_output = """OpenNet:00\\:11\\:22\\:33\\:44\\:55:80:2437 MHz:--:no
WPA2Net:00\\:11\\:22\\:33\\:44\\:66:70:2437 MHz:WPA2:no
WPA3Net:00\\:11\\:22\\:33\\:44\\:77:60:5180 MHz:WPA3:no"""

        async def mock_run_command(cmd, **kwargs):
            if "rescan" in cmd:
                return MagicMock(stdout=b"", returncode=0)
            return MagicMock(stdout=scan_output.encode(), returncode=0)

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command):
            networks = await network_manager_with_mocks.scan_wifi()

            # Find each network by SSID
            open_net = next(n for n in networks if n.ssid == "OpenNet")
            wpa2_net = next(n for n in networks if n.ssid == "WPA2Net")
            wpa3_net = next(n for n in networks if n.ssid == "WPA3Net")

            assert open_net.security == WiFiSecurity.OPEN
            assert wpa2_net.security == WiFiSecurity.WPA2
            assert wpa3_net.security == WiFiSecurity.WPA3

    @pytest.mark.asyncio
    async def test_scan_wifi_error_handling(self, network_manager_with_mocks):
        """Test WiFi scan error handling."""

        async def mock_run_command(cmd, **kwargs):
            raise NetworkManagerError("Scan failed")

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command):  # noqa: SIM117
            with pytest.raises(NetworkManagerError):
                await network_manager_with_mocks.scan_wifi()

    @pytest.mark.asyncio
    async def test_scan_wifi_deduplicates_ssids(self, network_manager_with_mocks):
        """Test that duplicate SSIDs are filtered."""
        scan_output = """TestNetwork:00\\:11\\:22\\:33\\:44\\:55:80:2437 MHz:WPA2:no
TestNetwork:00\\:11\\:22\\:33\\:44\\:66:70:2437 MHz:WPA2:no"""

        async def mock_run_command(cmd, **kwargs):
            if "rescan" in cmd:
                return MagicMock(stdout=b"", returncode=0)
            return MagicMock(stdout=scan_output.encode(), returncode=0)

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command):
            networks = await network_manager_with_mocks.scan_wifi()

            # Should only have one network
            assert len(networks) == 1


# =============================================================================
# WiFi Connection Tests
# =============================================================================


class TestWiFiConnection:
    """Test WiFi connection management."""

    @pytest.mark.asyncio
    async def test_connect_wifi_success(self, network_manager_with_mocks):
        """Test successful WiFi connection."""
        call_count = 0

        async def mock_run_command(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock(stdout=b"", returncode=0)

        async def mock_get_status():
            return NetworkStatus(
                mode=NetworkMode.CLIENT,
                connected=True,
                interface="wlan0",
                ssid="TestNetwork",
            )

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command), patch.object(
            network_manager_with_mocks, "disconnect", new_callable=AsyncMock
        ), patch.object(network_manager_with_mocks, "get_status", side_effect=mock_get_status), patch.object(
            network_manager_with_mocks, "_configure_ethernet_client_mode", new_callable=AsyncMock
        ), patch.object(
            network_manager_with_mocks, "_restart_avahi", new_callable=AsyncMock
        ):
            result = await network_manager_with_mocks.connect_wifi("TestNetwork", "password123")

            assert result is True

    @pytest.mark.asyncio
    async def test_connect_wifi_with_existing_profile(self, network_manager_with_mocks):
        """Test connecting with existing connection profile."""
        calls = []

        async def mock_run_command(cmd, **kwargs):
            calls.append(cmd)
            # Connection profile exists
            if "connection" in cmd and "show" in cmd:
                return MagicMock(stdout=b"", returncode=0)
            return MagicMock(stdout=b"", returncode=0)

        async def mock_get_status():
            return NetworkStatus(
                mode=NetworkMode.CLIENT,
                connected=True,
                interface="wlan0",
                ssid="TestNetwork",
            )

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command), patch.object(
            network_manager_with_mocks, "disconnect", new_callable=AsyncMock
        ), patch.object(network_manager_with_mocks, "get_status", side_effect=mock_get_status), patch.object(
            network_manager_with_mocks, "_configure_ethernet_client_mode", new_callable=AsyncMock
        ), patch.object(
            network_manager_with_mocks, "_restart_avahi", new_callable=AsyncMock
        ):
            await network_manager_with_mocks.connect_wifi("TestNetwork", "password123")

            # Should have used "connection up" for existing profile
            connection_up_calls = [c for c in calls if "connection" in c and "up" in c]
            assert len(connection_up_calls) > 0


# =============================================================================
# Disconnect Tests
# =============================================================================


class TestDisconnect:
    """Test network disconnection."""

    @pytest.mark.asyncio
    async def test_disconnect(self, network_manager_with_mocks):
        """Test disconnecting from network."""

        async def mock_run_command(cmd, **kwargs):
            return MagicMock(stdout=b"", returncode=0)

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command), patch.object(
            network_manager_with_mocks, "_disable_ethernet_client_mode", new_callable=AsyncMock
        ):
            await network_manager_with_mocks.disconnect()


# =============================================================================
# AP Mode Tests
# =============================================================================


class TestAPMode:
    """Test Access Point mode management."""

    @pytest.mark.asyncio
    async def test_enable_ap_mode(self, network_manager_with_mocks):
        """Test enabling AP mode."""
        calls = []

        async def mock_run_command(cmd, **kwargs):
            calls.append(cmd)
            return MagicMock(stdout=b"", returncode=0)

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command), patch.object(
            network_manager_with_mocks, "disconnect", new_callable=AsyncMock
        ), patch.object(
            network_manager_with_mocks, "_configure_ethernet_ap_mode", new_callable=AsyncMock
        ), patch.object(
            network_manager_with_mocks, "_restart_avahi", new_callable=AsyncMock
        ):
            await network_manager_with_mocks.enable_ap_mode()

            # Should have created or activated AP connection
            assert len(calls) > 0

    @pytest.mark.asyncio
    async def test_disable_ap_mode(self, network_manager_with_mocks):
        """Test disabling AP mode."""

        async def mock_run_command(cmd, **kwargs):
            return MagicMock(stdout=b"", returncode=0)

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command), patch.object(
            network_manager_with_mocks, "_disable_ethernet_ap_mode", new_callable=AsyncMock
        ):
            await network_manager_with_mocks.disable_ap_mode()


# =============================================================================
# Command Execution Tests
# =============================================================================


class TestCommandExecution:
    """Test command execution helper."""

    @pytest.mark.asyncio
    async def test_run_command_success(self, network_manager_with_mocks):
        """Test successful command execution."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b"success", b""))
            mock_exec.return_value = mock_process

            result = await network_manager_with_mocks._run_command(["nmcli", "device", "status"])

            assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_run_command_failure(self, network_manager_with_mocks):
        """Test command execution failure."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = MagicMock()
            mock_process.returncode = 1
            mock_process.communicate = AsyncMock(return_value=(b"", b"error"))
            mock_exec.return_value = mock_process

            with pytest.raises(NetworkManagerError):
                await network_manager_with_mocks._run_command(["nmcli", "invalid", "command"])

    @pytest.mark.asyncio
    async def test_run_command_with_sudo_retry(self, network_manager_with_mocks):
        """Test command retry with sudo on permission error."""
        call_count = 0

        async def mock_communicate():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (b"", b"Insufficient privileges")
            return (b"success", b"")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = MagicMock()
            mock_process.returncode = 1
            mock_process.communicate = mock_communicate

            mock_process_sudo = MagicMock()
            mock_process_sudo.returncode = 0
            mock_process_sudo.communicate = AsyncMock(return_value=(b"success", b""))

            mock_exec.side_effect = [mock_process, mock_process_sudo]

            result = await network_manager_with_mocks._run_command(["nmcli", "device", "wifi", "rescan"], try_sudo=True)

            # Should have retried with sudo
            assert call_count >= 1


# =============================================================================
# Ethernet Configuration Tests
# =============================================================================


class TestEthernetConfiguration:
    """Test Ethernet configuration for different modes."""

    @pytest.mark.asyncio
    async def test_configure_ethernet_client_mode(self, network_manager_with_mocks):
        """Test configuring Ethernet for client mode."""
        calls = []

        async def mock_run_command(cmd, **kwargs):
            calls.append(cmd)
            # Return failure for connection check to force creation
            if "show" in cmd:
                return MagicMock(stdout=b"", returncode=10)
            return MagicMock(stdout=b"", returncode=0)

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command), patch.object(
            network_manager_with_mocks, "_enable_ip_forwarding", new_callable=AsyncMock
        ):
            await network_manager_with_mocks._configure_ethernet_client_mode()

            # Should have created connection
            add_calls = [c for c in calls if "add" in c]
            assert len(add_calls) > 0

    @pytest.mark.asyncio
    async def test_configure_ethernet_ap_mode(self, network_manager_with_mocks):
        """Test configuring Ethernet for AP mode."""
        calls = []

        async def mock_run_command(cmd, **kwargs):
            calls.append(cmd)
            # Return failure for connection check to force creation
            if "show" in cmd:
                return MagicMock(stdout=b"", returncode=10)
            return MagicMock(stdout=b"", returncode=0)

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command), patch.object(
            network_manager_with_mocks, "_enable_ip_forwarding", new_callable=AsyncMock
        ):
            await network_manager_with_mocks._configure_ethernet_ap_mode("192.168.4.1")

            # Should have created connection
            add_calls = [c for c in calls if "add" in c]
            assert len(add_calls) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in NetworkManager."""

    def test_network_manager_error_exception(self):
        """Test NetworkManagerError exception."""
        error = NetworkManagerError("Test error message")
        assert str(error) == "Test error message"

    @pytest.mark.asyncio
    async def test_graceful_handling_of_nmcli_failure(self, network_manager_with_mocks):
        """Test graceful handling when nmcli fails."""

        async def mock_run_command(cmd, **kwargs):
            raise Exception("nmcli not available")

        with patch.object(network_manager_with_mocks, "_run_command", side_effect=mock_run_command):
            # get_status should handle errors gracefully
            status = await network_manager_with_mocks.get_status()

            assert status.mode == NetworkMode.DISCONNECTED
            assert status.connected is False
