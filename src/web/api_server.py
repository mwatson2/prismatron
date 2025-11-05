"""
Prismatron Web API Server.

FastAPI-based backend for the Prismatron web interface with retro-futurism design.
Provides endpoints for home, upload, effects, playlist management, and settings.
"""

import asyncio
import json
import logging
import os
import random
import shutil
import subprocess

# Add src to path for imports
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import psutil
import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import jtop for GPU usage monitoring on Jetson platforms
try:
    from jtop import jtop

    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False

from src.const import FRAME_HEIGHT, FRAME_WIDTH
from src.core.control_state import ControlState, ProducerState, RendererState
from src.core.playlist_sync import PlaylistItem as SyncPlaylistItem
from src.core.playlist_sync import PlaylistState as SyncPlaylistState
from src.core.playlist_sync import (
    PlaylistSyncClient,
)
from src.core.playlist_sync import TransitionConfig as SyncTransitionConfig
from src.network import NetworkManager
from src.network.models import APConfig, ClientConfig, NetworkConfig, NetworkMode, NetworkStatus, WiFiNetwork

logger = logging.getLogger(__name__)


def get_cpu_temperature():
    """Get CPU temperature in Celsius from thermal zone."""
    try:
        with open("/sys/devices/virtual/thermal/thermal_zone0/temp") as f:
            temp_microcelsius = int(f.read().strip())
            return temp_microcelsius / 1000.0  # Convert microcelsius to celsius
    except Exception as e:
        logger.warning(f"Failed to read CPU temperature: {e}")
        return 0.0


def get_gpu_temperature():
    """Get GPU temperature in Celsius from thermal zone."""
    try:
        with open("/sys/devices/virtual/thermal/thermal_zone1/temp") as f:
            temp_microcelsius = int(f.read().strip())
            return temp_microcelsius / 1000.0  # Convert microcelsius to celsius
    except Exception as e:
        logger.warning(f"Failed to read GPU temperature: {e}")
        return 0.0


def get_gpu_usage():
    """Get GPU usage percentage using jtop."""
    if not JTOP_AVAILABLE:
        return 0.0

    try:
        with jtop() as jetson:
            if jetson.ok():
                # Access GPU usage via the correct path: gpu.status.load
                return float(jetson.gpu["gpu"]["status"]["load"])
        return 0.0
    except Exception as e:
        # Only warn once about jtop service issues to avoid log spam
        if not hasattr(get_gpu_usage, "_jtop_warning_shown"):
            logger.warning(f"Failed to read GPU usage: {e}")
            if "jtop.service" in str(e):
                logger.warning("Please run: sudo systemctl restart jtop.service")
            get_gpu_usage._jtop_warning_shown = True
        return 0.0


# Pydantic models for API requests/responses
class TransitionConfig(BaseModel):
    """Transition configuration model."""

    type: str = Field("none", description="Transition type: none, fade, blur")
    parameters: Dict = Field(default_factory=dict, description="Transition parameters")

    def dict_serializable(self):
        """Return a dictionary for serialization."""
        return self.dict()


class PlaylistItem(BaseModel):
    """Playlist item model."""

    id: str = Field(..., description="Unique item ID")
    name: str = Field(..., description="Display name")
    type: str = Field(..., description="Item type: image, video, effect")
    file_path: Optional[str] = Field(None, description="File path for media items")
    effect_config: Optional[Dict] = Field(None, description="Effect configuration")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    thumbnail: Optional[str] = Field(None, description="Base64 thumbnail")
    created_at: datetime = Field(default_factory=datetime.now)
    order: int = Field(0, description="Display order")
    transition_in: TransitionConfig = Field(default_factory=TransitionConfig, description="Transition in configuration")
    transition_out: TransitionConfig = Field(
        default_factory=TransitionConfig, description="Transition out configuration"
    )

    def dict_serializable(self):
        """Return a dictionary with datetime objects converted to ISO strings."""
        data = self.dict()
        data["created_at"] = self.created_at.isoformat()
        data["transition_in"] = self.transition_in.dict_serializable()
        data["transition_out"] = self.transition_out.dict_serializable()
        return data


class PlaylistState(BaseModel):
    """Current playlist state."""

    items: List[PlaylistItem] = Field(default_factory=list)
    current_index: int = Field(0, description="Currently playing item index")
    is_playing: bool = Field(False, description="Whether playback is active")
    auto_repeat: bool = Field(True, description="Auto-repeat playlist")
    shuffle: bool = Field(False, description="Shuffle mode")

    def dict_serializable(self):
        """Return a dictionary with datetime objects converted to ISO strings."""
        data = self.dict()
        data["items"] = [item.dict_serializable() for item in self.items]
        return data


class SystemSettings(BaseModel):
    """System settings model."""

    brightness: float = Field(1.0, ge=0.0, le=1.0, description="Global brightness (0-1)")
    frame_rate: float = Field(30.0, ge=1.0, le=60.0, description="Target frame rate")
    led_count: int = Field(description="Number of LEDs (read from pattern file)")
    display_resolution: Dict[str, int] = Field(default_factory=lambda: {"width": FRAME_WIDTH, "height": FRAME_HEIGHT})
    auto_start_playlist: bool = Field(True, description="Auto-start playlist on boot")
    preview_enabled: bool = Field(True, description="Enable live preview")
    audio_reactive_enabled: bool = Field(False, description="Enable audio reactive effects")


def get_system_settings() -> SystemSettings:
    """Get system settings."""
    global system_settings
    if system_settings is None:
        raise RuntimeError("SystemSettings not initialized - LED count not available from startup")
    return system_settings


def get_actual_led_count() -> int:
    """Get the actual LED count from system settings."""
    try:
        return get_system_settings().led_count
    except RuntimeError:
        # Fallback to a reasonable default if settings not available
        return 2624


def generate_random_led_transition() -> TransitionConfig:
    """Generate a random LED transition with 1s duration and default parameters."""
    led_transitions = ["ledblur", "ledfade", "ledrandom"]
    transition_type = random.choice(led_transitions)

    # LED transitions use default parameters for the specified duration
    return TransitionConfig(type=transition_type, parameters={"duration": 1.0})  # 1 second duration as requested


def load_uploads_playlist() -> Dict:
    """Load the uploads playlist from disk, or create a new one if it doesn't exist."""
    uploads_playlist_path = PLAYLISTS_DIR / UPLOADS_PLAYLIST_NAME

    if uploads_playlist_path.exists():
        try:
            with open(uploads_playlist_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load uploads playlist: {e}")

    # Return empty playlist structure
    return {
        "version": "1.0",
        "name": "Uploads Playlist",
        "description": "Automatically managed playlist for uploaded files",
        "created_at": datetime.now().isoformat(),
        "modified_at": datetime.now().isoformat(),
        "auto_repeat": True,
        "shuffle": False,
        "items": [],
    }


def save_uploads_playlist(playlist_data: Dict) -> None:
    """Save the uploads playlist to disk."""
    uploads_playlist_path = PLAYLISTS_DIR / UPLOADS_PLAYLIST_NAME
    playlist_data["modified_at"] = datetime.now().isoformat()

    try:
        with open(uploads_playlist_path, "w") as f:
            json.dump(playlist_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save uploads playlist: {e}")


async def periodic_upload_cleanup_task():
    """Background task to periodically clean up old upload files every 10 minutes."""
    while True:
        try:
            # Wait 10 minutes between cleanups
            await asyncio.sleep(10 * 60)  # 10 minutes = 600 seconds

            # Run the cleanup
            cleanup_old_uploads()

        except Exception as e:
            logger.error(f"Error in periodic upload cleanup task: {e}")


def cleanup_old_uploads() -> None:
    """
    Remove files from uploads folder that are older than 24 hours.
    Also removes them from uploads playlist and syncs with live playlist.
    """
    try:
        current_time = time.time()
        cutoff_time = current_time - (24 * 60 * 60)  # 24 hours ago

        items_to_remove = []
        files_removed_count = 0

        # Check files in uploads directory
        if UPLOAD_DIR.exists():
            for file_path in UPLOAD_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_stats = file_path.stat()
                        # Check if file is older than 24 hours
                        if file_stats.st_mtime < cutoff_time:
                            # Find corresponding playlist item
                            playlist_data = load_uploads_playlist()
                            for item in playlist_data["items"]:
                                if item.get("file_path") == str(file_path):
                                    items_to_remove.append(item)
                                    break

                            # Delete the file
                            file_path.unlink()
                            files_removed_count += 1
                            logger.info(f"Cleaned up old upload file (24h+): {file_path.name}")

                    except Exception as e:
                        logger.warning(f"Failed to process upload file {file_path}: {e}")

        # Remove items from playlist and sync
        if items_to_remove:
            remove_files_from_uploads_playlist_and_sync(items_to_remove)
            logger.info(
                f"Periodic cleanup: removed {files_removed_count} old files and {len(items_to_remove)} playlist items"
            )
        elif files_removed_count > 0:
            logger.info(f"Periodic cleanup: removed {files_removed_count} old files (no playlist items)")

    except Exception as e:
        logger.error(f"Failed to perform periodic upload cleanup: {e}")


def remove_files_from_uploads_playlist_and_sync(items_to_remove: List[Dict]) -> None:
    """
    Remove items from uploads playlist and notify playlist sync service if active.

    Args:
        items_to_remove: List of playlist items to remove
    """
    if not items_to_remove:
        return

    # Remove items from uploads playlist file
    playlist_data = load_uploads_playlist()

    # Remove items by ID
    item_ids_to_remove = {item["id"] for item in items_to_remove}
    playlist_data["items"] = [item for item in playlist_data["items"] if item["id"] not in item_ids_to_remove]

    # Update order indices
    for i, item in enumerate(playlist_data["items"]):
        item["order"] = i

    # Save the updated playlist
    save_uploads_playlist(playlist_data)

    # If uploads playlist is currently active, remove from live playlist
    global current_playlist_file
    if current_playlist_file == UPLOADS_PLAYLIST_NAME and playlist_sync_client and playlist_sync_client.connected:
        logger.info("Uploads playlist is active - removing items from live playlist")

        for old_item in items_to_remove:
            try:
                if playlist_sync_client.remove_item(old_item["id"]):
                    logger.info(f"Removed item {old_item['name']} from live playlist")
            except Exception as e:
                logger.warning(f"Failed to remove item from live playlist: {e}")


def manage_uploads_playlist(new_item: PlaylistItem) -> None:
    """
    Manage the uploads playlist by adding new item and maintaining max 8 items.
    When exceeding max items, delete oldest item from both playlist and uploads directory.
    If uploads playlist is currently active, notify playlist sync service of changes.
    """
    playlist_data = load_uploads_playlist()

    # Create new playlist item with random LED transitions
    transition_in = generate_random_led_transition()
    transition_out = generate_random_led_transition()

    playlist_item = {
        "id": new_item.id,
        "name": new_item.name,
        "type": new_item.type,
        "duration": new_item.duration or (10.0 if new_item.type == "image" else None),
        "order": 0,  # New items go to the beginning
        "transition_in": {"type": transition_in.type, "parameters": transition_in.parameters},
        "transition_out": {"type": transition_out.type, "parameters": transition_out.parameters},
        "file_path": new_item.file_path,
        "created_at": new_item.created_at.isoformat(),
    }

    # Track items that will be removed for live update
    items_to_remove = []

    # Add new item to beginning of playlist (most recent uploads first)
    playlist_data["items"].insert(0, playlist_item)

    # Remove oldest items if exceeding max count (oldest are now at the end)
    while len(playlist_data["items"]) > UPLOADS_PLAYLIST_MAX_ITEMS:
        oldest_item = playlist_data["items"].pop(-1)  # Remove from end (oldest)
        items_to_remove.append(oldest_item)

        # Delete the oldest file from uploads directory
        try:
            if oldest_item.get("file_path"):
                oldest_file_path = Path(oldest_item["file_path"])
                if oldest_file_path.exists():
                    oldest_file_path.unlink()
                    logger.info(f"Deleted oldest upload file: {oldest_file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete oldest upload file: {e}")

    # Update order indices and save playlist
    for i, item in enumerate(playlist_data["items"]):
        item["order"] = i
    save_uploads_playlist(playlist_data)

    # Remove old items from uploads playlist and sync with live playlist
    remove_files_from_uploads_playlist_and_sync(items_to_remove)

    # If uploads playlist is currently active, add new item to live playlist
    global current_playlist_file
    if current_playlist_file == UPLOADS_PLAYLIST_NAME and playlist_sync_client and playlist_sync_client.connected:

        logger.info("Uploads playlist is active - adding new item to live playlist")

        # Add new item to live playlist
        try:
            # Convert to PlaylistItem for sync
            live_item = PlaylistItem(
                id=playlist_item["id"],
                name=playlist_item["name"],
                type=playlist_item["type"],
                file_path=playlist_item["file_path"],
                duration=playlist_item["duration"],
                order=playlist_item["order"],
                transition_in=TransitionConfig(**playlist_item["transition_in"]),
                transition_out=TransitionConfig(**playlist_item["transition_out"]),
            )

            sync_item = api_item_to_sync_item(live_item)
            if playlist_sync_client.add_item(sync_item):
                logger.info(f"Added new item {playlist_item['name']} to live playlist")
            else:
                logger.warning("Failed to add new item to live playlist")

        except Exception as e:
            logger.warning(f"Failed to add new item to live playlist: {e}")

    # If no playlist is currently playing, auto-load and start the Uploads Playlist
    elif current_playlist_file is None and playlist_sync_client and playlist_sync_client.connected:
        logger.info("No playlist currently playing - auto-loading and starting Uploads Playlist")
        try:
            # Load the uploads playlist
            current_playlist_file = UPLOADS_PLAYLIST_NAME

            # Clear current playlist and add all uploads playlist items
            playlist_sync_client.clear_playlist()

            # Add all items from uploads playlist to live playlist
            for item_data in playlist_data["items"]:
                try:
                    # Convert to PlaylistItem for sync
                    live_item = PlaylistItem(
                        id=item_data["id"],
                        name=item_data["name"],
                        type=item_data["type"],
                        file_path=item_data["file_path"],
                        duration=item_data["duration"],
                        order=item_data["order"],
                        transition_in=TransitionConfig(**item_data["transition_in"]),
                        transition_out=TransitionConfig(**item_data["transition_out"]),
                    )

                    sync_item = api_item_to_sync_item(live_item)
                    playlist_sync_client.add_item(sync_item)

                except Exception as e:
                    logger.warning(f"Failed to add uploads playlist item to live playlist: {e}")

            # Start playing the uploads playlist
            if playlist_sync_client.play():
                logger.info("Successfully auto-loaded and started Uploads Playlist")
            else:
                logger.warning("Failed to start Uploads Playlist after auto-loading")

        except Exception as e:
            logger.warning(f"Failed to auto-load and start Uploads Playlist: {e}")
            current_playlist_file = None


def update_playlists_for_file_operation(
    old_filename: str, new_filename: Optional[str] = None, operation: str = "rename"
) -> Dict[str, Union[int, List[str]]]:
    """
    Update all playlists when a file is renamed or deleted.

    Args:
        old_filename: Original filename (e.g., "video.mp4")
        new_filename: New filename for rename operations
        operation: "rename" or "delete"

    Returns:
        Dict with update statistics and affected playlists
    """
    if operation not in ["rename", "delete"]:
        logger.error(f"Invalid operation: {operation}")
        return {"affected_playlists": 0, "items_modified": 0, "playlists": []}

    if operation == "rename" and not new_filename:
        logger.error("new_filename required for rename operation")
        return {"affected_playlists": 0, "items_modified": 0, "playlists": []}

    affected_playlists = []
    total_items_modified = 0
    current_playlist_affected = False

    try:
        # Iterate through all playlist files
        if not PLAYLISTS_DIR.exists():
            logger.warning(f"Playlists directory does not exist: {PLAYLISTS_DIR}")
            return {"affected_playlists": 0, "items_modified": 0, "playlists": []}

        for playlist_path in PLAYLISTS_DIR.iterdir():
            if not playlist_path.is_file() or playlist_path.suffix != ".json":
                continue

            try:
                # Load playlist
                with open(playlist_path) as f:
                    playlist_data = json.load(f)

                # Track if this playlist was modified
                playlist_modified = False
                items_modified_in_playlist = 0

                # Process items based on operation
                if operation == "rename":
                    # Update file_path and name for matching items
                    for item in playlist_data.get("items", []):
                        if item.get("type") in ["image", "video"] and item.get("file_path") == old_filename:
                            old_name = item.get("name", old_filename)
                            item["file_path"] = new_filename
                            item["name"] = new_filename  # Update display name to match new filename
                            items_modified_in_playlist += 1
                            playlist_modified = True
                            logger.info(
                                f"Renamed file reference in playlist {playlist_path.name}: name={old_name} -> {new_filename}, file_path={old_filename} -> {new_filename}"
                            )

                elif operation == "delete":
                    # Remove items with matching file_path
                    original_count = len(playlist_data.get("items", []))
                    playlist_data["items"] = [
                        item
                        for item in playlist_data.get("items", [])
                        if not (item.get("type") in ["image", "video"] and item.get("file_path") == old_filename)
                    ]
                    items_removed = original_count - len(playlist_data["items"])

                    if items_removed > 0:
                        items_modified_in_playlist = items_removed
                        playlist_modified = True

                        # Update order indices
                        for i, item in enumerate(playlist_data["items"]):
                            item["order"] = i

                        logger.info(
                            f"Removed {items_removed} item(s) from playlist {playlist_path.name} referencing {old_filename}"
                        )

                # Save playlist if modified
                if playlist_modified:
                    playlist_data["modified_at"] = datetime.now().isoformat()
                    with open(playlist_path, "w") as f:
                        json.dump(playlist_data, f, indent=2)

                    affected_playlists.append(playlist_path.name)
                    total_items_modified += items_modified_in_playlist

                    # Check if current playlist is affected
                    global current_playlist_file
                    if current_playlist_file == playlist_path.name:
                        current_playlist_affected = True

            except Exception as e:
                logger.error(f"Failed to process playlist {playlist_path}: {e}")
                continue

        # Always check and sync with live playlist (not just when saved playlists are affected)
        # Live playlist may contain unsaved items that reference the file
        if playlist_sync_client and playlist_sync_client.connected:
            try:
                # Use global playlist_state which is kept in sync via callback
                global playlist_state
                if playlist_state and playlist_state.items:
                    live_items_affected = 0

                    if operation == "rename":
                        # Update file_path and name in live playlist items
                        for item in playlist_state.items:
                            if item.type in ["image", "video"] and item.file_path:
                                # Extract just the filename from the absolute path
                                item_filename = Path(item.file_path).name
                                if item_filename == old_filename:
                                    # Update to new filename - keep directory part
                                    new_path = Path(item.file_path).parent / new_filename
                                    logger.info(
                                        f"Updating live playlist item: name={item.name} -> {new_filename}, file_path={item.file_path} -> {new_path}"
                                    )

                                    # Remember position for re-adding
                                    position = item.order

                                    # Remove old item
                                    playlist_sync_client.remove_item(item.id)

                                    # Update both file_path AND name
                                    item.file_path = str(new_path)
                                    item.name = new_filename  # ← Update the display name too!

                                    # Convert to sync item and add back at same position
                                    sync_item = api_item_to_sync_item(item)
                                    playlist_sync_client.add_item(sync_item, position=position)

                                    live_items_affected += 1

                    elif operation == "delete":
                        # Remove items from live playlist
                        items_to_remove = []
                        for item in playlist_state.items:
                            if item.type in ["image", "video"] and item.file_path:
                                item_filename = Path(item.file_path).name
                                if item_filename == old_filename:
                                    items_to_remove.append(item.id)

                        for item_id in items_to_remove:
                            logger.info(f"Removing item {item_id} from live playlist")
                            playlist_sync_client.remove_item(item_id)
                            live_items_affected += 1

                    if live_items_affected > 0:
                        logger.info(f"Updated {live_items_affected} items in live playlist")

            except Exception as e:
                logger.error(f"Failed to sync {operation} operation with live playlist: {e}")

        logger.info(
            f"File {operation} operation complete: {total_items_modified} items in {len(affected_playlists)} playlists"
        )

        return {
            "affected_playlists": len(affected_playlists),
            "items_modified": total_items_modified,
            "playlists": affected_playlists,
        }

    except Exception as e:
        logger.error(f"Failed to update playlists for file {operation}: {e}")
        return {"affected_playlists": 0, "items_modified": 0, "playlists": [], "error": str(e)}


class EffectPreset(BaseModel):
    """Effect preset model."""

    id: str = Field(..., description="Unique effect ID")
    name: str = Field(..., description="Effect name")
    description: str = Field("", description="Effect description")
    config: Dict = Field(default_factory=dict, description="Effect parameters")
    category: str = Field("general", description="Effect category")
    icon: str = Field("✨", description="Effect icon/emoji")


class SystemStatus(BaseModel):
    """Current system status."""

    is_online: bool = Field(True, description="System online status")
    current_file: Optional[str] = Field(None, description="Currently playing file")
    playlist_position: int = Field(0, description="Current playlist position")
    rendering_index: int = Field(-1, description="Currently rendering playlist index")
    renderer_state: str = Field("STOPPED", description="Current renderer state (STOPPED/WAITING/PLAYING/PAUSED)")
    brightness: float = Field(1.0, description="Current brightness")
    frame_rate: float = Field(0.0, description="Current frame rate")
    uptime: float = Field(0.0, description="System uptime in seconds")
    memory_usage: float = Field(0.0, description="Memory usage percentage")
    memory_usage_gb: float = Field(0.0, description="Memory usage in GB")
    cpu_usage: float = Field(0.0, description="CPU usage percentage")
    gpu_usage: float = Field(0.0, description="GPU usage percentage")
    led_panel_connected: bool = Field(False, description="LED panel connection status")
    led_panel_status: str = Field("disconnected", description="LED panel status (connected/connecting/disconnected)")

    # Temperature metrics
    cpu_temperature: float = Field(0.0, description="CPU temperature in Celsius")
    gpu_temperature: float = Field(0.0, description="GPU temperature in Celsius")

    # New FPS and frame dropping metrics
    consumer_input_fps: float = Field(0.0, description="Consumer input FPS from producer")
    renderer_output_fps: float = Field(0.0, description="Renderer output FPS (EWMA)")
    dropped_frames_percentage: float = Field(0.0, description="Percentage of frames dropped early")
    late_frame_percentage: float = Field(0.0, description="Percentage of late frames")


# Network Management Models
class WiFiConnectRequest(BaseModel):
    """Request model for connecting to WiFi."""

    ssid: str = Field(..., description="WiFi network SSID")
    password: Optional[str] = Field(None, description="WiFi password (if required)")


class NetworkStatusResponse(BaseModel):
    """Response model for network status."""

    mode: str = Field(..., description="Current network mode: ap, client, disconnected")
    connected: bool = Field(..., description="Whether currently connected")
    interface: str = Field(..., description="Network interface name")
    ip_address: Optional[str] = Field(None, description="Current IP address")
    ssid: Optional[str] = Field(None, description="Connected network SSID")
    signal_strength: Optional[int] = Field(None, description="Signal strength (0-100)")
    gateway: Optional[str] = Field(None, description="Gateway IP address")
    dns_servers: Optional[List[str]] = Field(None, description="DNS server addresses")


class WiFiNetworkResponse(BaseModel):
    """Response model for WiFi network."""

    ssid: str = Field(..., description="Network SSID")
    bssid: str = Field(..., description="Network BSSID")
    signal_strength: int = Field(..., description="Signal strength (0-100)")
    frequency: int = Field(..., description="Frequency in MHz")
    security: str = Field(..., description="Security type")
    connected: bool = Field(..., description="Whether currently connected")


# Global state
playlist_state = PlaylistState()
system_settings: Optional[SystemSettings] = None  # Will be initialized after consumer provides LED count
control_state: Optional[ControlState] = None
consumer_process: Optional[object] = None  # Will be set by main process
producer_process: Optional[object] = None  # Will be set by main process
diffusion_patterns_path: Optional[str] = None  # Will be set by main process
network_manager: Optional[NetworkManager] = None  # Network management instance

# Track currently loaded playlist file for real-time updates
current_playlist_file: Optional[str] = None

# Playlist synchronization client
playlist_sync_client: Optional[PlaylistSyncClient] = None

# Uploads playlist configuration
UPLOADS_PLAYLIST_NAME = "Uploads Playlist.json"
UPLOADS_PLAYLIST_MAX_ITEMS = 8

# File storage paths
UPLOAD_DIR = Path("uploads")
MEDIA_DIR = Path("media")
THUMBNAILS_DIR = Path("thumbnails")
PLAYLISTS_DIR = Path("playlists")

# Ensure directories exist
for dir_path in [UPLOAD_DIR, MEDIA_DIR, THUMBNAILS_DIR, PLAYLISTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Import the visual effects system
try:
    from src.producer.effects import EffectRegistry

    EFFECTS_AVAILABLE = True
    logger.info("Visual effects system loaded successfully")
except ImportError as e:
    logger.warning(f"Visual effects system not available: {e}")
    EFFECTS_AVAILABLE = False

# Import video conversion system
try:
    from src.web.video_converter import (
        ConversionJob,
        ConversionStatus,
        get_conversion_manager,
        initialize_conversion_manager,
        shutdown_conversion_manager,
    )

    VIDEO_CONVERSION_AVAILABLE = True
    logger.info("Video conversion system loaded successfully")
except ImportError as e:
    logger.warning(f"Video conversion system not available: {e}")
    VIDEO_CONVERSION_AVAILABLE = False


def get_effect_presets():
    """Get effect presets from the registry or fallback to hardcoded presets."""
    if EFFECTS_AVAILABLE:
        # Load effects from registry
        registry_effects = EffectRegistry.list_effects()
        presets = []

        for effect in registry_effects:
            preset = EffectPreset(
                id=effect["id"],
                name=effect["name"],
                description=effect["description"],
                config=effect["config"],
                category=effect["category"],
                icon=effect["icon"],
            )
            presets.append(preset)

        # Add text display effect (handled specially by existing system)
        presets.append(
            EffectPreset(
                id="text_display",
                name="Text Display",
                description="Display custom text with configurable fonts and colors",
                config={
                    "text": "Hello World",
                    "font_family": "DejaVu Sans",
                    "font_style": "normal",
                    "font_size": "auto",
                    "fg_color": "#FFFFFF",
                    "bg_color": "#000000",
                    "animation": "static",
                    "alignment": "center",
                    "vertical_alignment": "center",
                    "duration": 10.0,
                    "fps": 30.0,  # Note: This will be overridden by DEFAULT_CONTENT_FPS in text_source.py
                },
                category="text",
                icon="📝",
            )
        )

        return presets
    else:
        # Fallback to hardcoded presets
        return [
            EffectPreset(
                id="rainbow_cycle",
                name="Rainbow Cycle",
                description="Smooth rainbow color cycling across all LEDs",
                config={"speed": 1.0, "brightness": 1.0, "hue_shift": 0.0},
                category="color",
                icon="🌈",
            ),
            EffectPreset(
                id="color_wave",
                name="Color Wave",
                description="Animated wave of color flowing across the display",
                config={"color": "#ff0066", "speed": 2.0, "wavelength": 100},
                category="animation",
                icon="🌊",
            ),
            EffectPreset(
                id="text_display",
                name="Text Display",
                description="Display custom text with configurable fonts and colors",
                config={
                    "text": "Hello World",
                    "font_family": "DejaVu Sans",
                    "font_style": "normal",
                    "font_size": "auto",
                    "fg_color": "#FFFFFF",
                    "bg_color": "#000000",
                    "animation": "static",
                    "alignment": "center",
                    "vertical_alignment": "center",
                    "duration": 10.0,
                    "fps": 30.0,
                },
                category="text",
                icon="📝",
            ),
        ]


# Get effect presets (loaded dynamically or fallback)
EFFECT_PRESETS = get_effect_presets()

# Initialize FastAPI app
app = FastAPI(
    title="Prismatron Control Interface",
    description="Retro-futurism web interface for the Prismatron LED Display",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background task for preview data broadcasting
preview_task: Optional[asyncio.Task] = None
upload_cleanup_task: Optional[asyncio.Task] = None


def sync_item_to_api_item(sync_item: SyncPlaylistItem) -> PlaylistItem:
    """Convert sync playlist item to API playlist item."""
    # Convert transition configurations
    transition_in = TransitionConfig(
        type=sync_item.transition_in.type if sync_item.transition_in else "none",
        parameters=sync_item.transition_in.parameters if sync_item.transition_in else {},
    )
    transition_out = TransitionConfig(
        type=sync_item.transition_out.type if sync_item.transition_out else "none",
        parameters=sync_item.transition_out.parameters if sync_item.transition_out else {},
    )

    return PlaylistItem(
        id=sync_item.id,
        name=sync_item.name,
        type=sync_item.type,
        file_path=sync_item.file_path,
        effect_config=sync_item.effect_config,
        duration=sync_item.duration,
        thumbnail=sync_item.thumbnail,
        created_at=datetime.fromtimestamp(sync_item.created_at),
        order=sync_item.order,
        transition_in=transition_in,
        transition_out=transition_out,
    )


def api_item_to_sync_item(api_item: PlaylistItem) -> SyncPlaylistItem:
    """Convert API playlist item to sync playlist item."""
    # Convert transition configurations
    transition_in = SyncTransitionConfig(type=api_item.transition_in.type, parameters=api_item.transition_in.parameters)
    transition_out = SyncTransitionConfig(
        type=api_item.transition_out.type, parameters=api_item.transition_out.parameters
    )

    return SyncPlaylistItem(
        id=api_item.id,
        name=api_item.name,
        type=api_item.type,
        file_path=api_item.file_path,
        effect_config=api_item.effect_config,
        duration=api_item.duration,
        thumbnail=api_item.thumbnail,
        created_at=api_item.created_at.timestamp(),
        order=api_item.order,
        transition_in=transition_in,
        transition_out=transition_out,
    )


def validate_transition_config(transition_config: TransitionConfig) -> Dict[str, str]:
    """
    Validate transition configuration and return any errors.

    Returns:
        Dictionary of errors, empty if valid
    """
    errors = {}

    try:
        # Get available transition types from factory (includes LED transitions)
        from ..transitions.transition_factory import get_transition_factory

        factory = get_transition_factory()
        valid_types = factory.get_available_transitions_with_led()

        # Validate transition type
        if transition_config.type not in valid_types:
            errors["type"] = (
                f"Invalid transition type '{transition_config.type}'. Must be one of: {', '.join(valid_types)}"
            )
            return errors

        # Use factory validation for parameters
        transition_dict = {"type": transition_config.type, "parameters": transition_config.parameters}

        # Check if it's an LED transition
        if transition_config.type.startswith("led"):
            # Use LED transition factory for validation
            from ..transitions.led_transition_factory import get_led_transition_factory

            led_factory = get_led_transition_factory()
            if not led_factory.validate_led_transition_config(transition_dict):
                errors["parameters"] = f"Invalid parameters for LED transition '{transition_config.type}'"
        else:
            # Use regular transition factory for validation
            if not factory.validate_transition_config(transition_dict):
                errors["parameters"] = f"Invalid parameters for transition '{transition_config.type}'"

        return errors

    except Exception as e:
        # Fallback to hardcoded validation if factory fails
        logger.warning(f"Transition factory validation failed, using fallback: {e}")
        valid_types = ["none", "fade", "blur", "ledfade", "ledblur", "ledrandom"]
        if transition_config.type not in valid_types:
            errors["type"] = (
                f"Invalid transition type '{transition_config.type}'. Must be one of: {', '.join(valid_types)}"
            )

        # Basic parameter validation for fallback
        if transition_config.type in ["fade", "ledfade", "ledblur", "ledrandom"]:
            duration = transition_config.parameters.get("duration")
            if duration is not None:
                if not isinstance(duration, (int, float)):
                    errors["parameters.duration"] = "Duration must be a number"
                elif duration < 0.1 or duration > 60.0:
                    errors["parameters.duration"] = "Duration must be between 0.1 and 60.0 seconds"

    return errors


def validate_playlist_item(item: PlaylistItem) -> Dict[str, str]:
    """
    Validate playlist item including transition configurations.

    Returns:
        Dictionary of errors, empty if valid
    """
    errors = {}

    # Validate transition configurations
    transition_in_errors = validate_transition_config(item.transition_in)
    for key, error in transition_in_errors.items():
        errors[f"transition_in.{key}"] = error

    transition_out_errors = validate_transition_config(item.transition_out)
    for key, error in transition_out_errors.items():
        errors[f"transition_out.{key}"] = error

    return errors


def sync_state_to_api_state(sync_state: SyncPlaylistState) -> PlaylistState:
    """Convert sync playlist state to API playlist state."""
    return PlaylistState(
        items=[sync_item_to_api_item(item) for item in sync_state.items],
        current_index=sync_state.current_index,
        is_playing=sync_state.is_playing,
        auto_repeat=sync_state.auto_repeat,
        shuffle=sync_state.shuffle,
    )


async def on_playlist_sync_update(sync_state: SyncPlaylistState) -> None:
    """Handle playlist updates from synchronization service."""
    global playlist_state

    try:
        # Convert sync state to API state
        playlist_state = sync_state_to_api_state(sync_state)

        logger.info(
            f"Playlist sync update received: {len(playlist_state.items)} items, broadcasting to {len(manager.active_connections)} WebSocket clients"
        )

        # Broadcast update to connected WebSocket clients
        await manager.broadcast(
            {
                "type": "playlist_updated",
                "items": [item.dict_serializable() for item in playlist_state.items],
                "current_index": playlist_state.current_index,
                "is_playing": playlist_state.is_playing,
                "auto_repeat": playlist_state.auto_repeat,
                "shuffle": playlist_state.shuffle,
                "current_playlist_file": current_playlist_file,
            }
        )

        logger.debug("Playlist update broadcast complete")

    except Exception as e:
        logger.error(f"Error in async playlist sync update: {e}")


def sync_playlist_update_handler(sync_state: SyncPlaylistState) -> None:
    """Sync wrapper for playlist update handler."""
    try:
        logger.debug(f"Sync playlist update handler called with {len(sync_state.items)} items")

        # Create a task for the async function
        import asyncio

        try:
            # Try to get the running event loop
            loop = asyncio.get_running_loop()
            # If we get here, there's a running loop - schedule as task
            loop.create_task(on_playlist_sync_update(sync_state))
            logger.debug("Scheduled playlist update on running event loop")
        except RuntimeError:
            # No running loop - we can run one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # This shouldn't happen if get_running_loop() failed, but just in case
                    loop.create_task(on_playlist_sync_update(sync_state))
                    logger.debug("Scheduled playlist update on existing event loop")
                else:
                    loop.run_until_complete(on_playlist_sync_update(sync_state))
                    logger.debug("Ran playlist update to completion")
            except RuntimeError:
                # No event loop exists, create one
                asyncio.run(on_playlist_sync_update(sync_state))
                logger.debug("Created new event loop and ran playlist update")
    except Exception as e:
        logger.error(f"Error in playlist sync update handler: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    global preview_task, upload_cleanup_task, playlist_sync_client, network_manager

    # Initialize network manager
    try:
        network_manager = NetworkManager()
        logger.info("Network manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize network manager: {e}")

    # Start playlist synchronization client
    playlist_sync_client = PlaylistSyncClient(client_name="web_interface")
    playlist_sync_client.on_playlist_update = sync_playlist_update_handler
    if playlist_sync_client.connect():
        logger.info("Connected to playlist synchronization service")
    else:
        logger.warning("Failed to connect to playlist synchronization service")

    preview_task = asyncio.create_task(preview_broadcast_task())
    logger.info("Started preview data broadcasting task")

    upload_cleanup_task = asyncio.create_task(periodic_upload_cleanup_task())
    logger.info("Started periodic upload cleanup task (10 minute intervals)")

    # Register WebSocket broadcast callback for conversion updates
    if VIDEO_CONVERSION_AVAILABLE:
        conversion_manager = get_conversion_manager()
        if conversion_manager:
            # Get the main event loop to schedule tasks from worker thread
            main_loop = asyncio.get_event_loop()

            def on_conversion_status_update(job):
                """Broadcast conversion status updates via WebSocket (called from worker thread)."""
                try:
                    # Use run_coroutine_threadsafe to schedule on main event loop from worker thread
                    asyncio.run_coroutine_threadsafe(
                        manager.broadcast({"type": "conversion_update", "conversion": job.to_dict()}), main_loop
                    )
                    logger.debug(
                        f"Broadcasted conversion update for job {job.id}: {job.status.value} {job.progress:.1f}%"
                    )
                except Exception as e:
                    logger.error(f"Failed to broadcast conversion update: {e}")

            conversion_manager.add_status_callback(on_conversion_status_update)
            logger.info("Registered WebSocket broadcast callback for conversion updates")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background tasks on application shutdown."""
    import contextlib

    global preview_task, upload_cleanup_task, playlist_sync_client

    # Disconnect playlist sync client
    if playlist_sync_client:
        playlist_sync_client.disconnect()
        logger.info("Disconnected from playlist synchronization service")

    if preview_task:
        preview_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await preview_task
        logger.info("Stopped preview data broadcasting task")

    if upload_cleanup_task:
        upload_cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await upload_cleanup_task
        logger.info("Stopped periodic upload cleanup task")

    # Shutdown video conversion manager
    if VIDEO_CONVERSION_AVAILABLE:
        try:
            shutdown_conversion_manager()
        except Exception as e:
            logger.error(f"Failed to shutdown video conversion manager: {e}")


# Add no-cache middleware for static files
@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)

    # Add no-cache headers for static files (js, css, html)
    if request.url.path.startswith("/static/") or request.url.path.endswith((".js", ".css", ".html", ".map")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response


# WebSocket connections for live updates
class ConnectionManager:
    """Manages WebSocket connections for live updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {len(self.active_connections)} total connections")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {len(self.active_connections)} total connections")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


# Preview data broadcasting task
async def preview_broadcast_task():
    """Background task to broadcast preview data and system status at 5fps via WebSocket."""
    while True:
        try:
            if manager.active_connections:
                current_time = time.time()

                # Get LED panel connection status and new metrics from ControlState (IPC)
                led_panel_connected = False
                led_panel_status = "disconnected"
                consumer_input_fps = 0.0
                renderer_output_fps = 0.0
                dropped_frames_percentage = 0.0
                late_frame_percentage = 0.0

                # Get consumer statistics from ControlState (shared memory IPC)
                try:
                    if control_state:
                        system_status = control_state.get_status_dict()

                        # Extract consumer statistics from ControlState
                        consumer_input_fps = system_status.get("consumer_input_fps", 0.0)
                        renderer_output_fps = system_status.get("renderer_output_fps", 0.0)
                        dropped_frames_percentage = system_status.get("dropped_frames_percentage", 0.0)
                        late_frame_percentage = system_status.get("late_frame_percentage", 0.0)

                        # Debug log every 5 seconds to verify values more frequently
                        if hasattr(preview_broadcast_task, "_last_fps_debug_log"):
                            if current_time - preview_broadcast_task._last_fps_debug_log > 5.0:
                                logger.debug(
                                    f"WEBSOCKET FPS DEBUG: Read from ControlState - input_fps={consumer_input_fps:.2f}, output_fps={renderer_output_fps:.2f}, dropped={dropped_frames_percentage:.1f}%"
                                )
                                preview_broadcast_task._last_fps_debug_log = current_time
                        else:
                            preview_broadcast_task._last_fps_debug_log = current_time
                    else:
                        logger.debug("No control state available")

                    # Try to get LED panel connection status from consumer process if available (fallback)
                    if consumer_process and hasattr(consumer_process, "get_statistics"):
                        consumer_stats = consumer_process.get_statistics()
                        wled_stats = consumer_stats.get("wled_stats", {})
                        led_panel_connected = wled_stats.get("is_connected", False)

                        # Determine status string based on connection state
                        if led_panel_connected:
                            led_panel_status = "connected"
                        elif consumer_process.is_running and not led_panel_connected:
                            led_panel_status = "connecting"
                        else:
                            led_panel_status = "disconnected"
                    else:
                        # Default status when no consumer process reference
                        led_panel_status = "unknown"

                except Exception as e:
                    logger.warning(f"Failed to get consumer statistics from ControlState: {e}")

                # Get real system resource usage
                try:
                    # CPU usage (non-blocking, use cached value or 0.1s interval)
                    cpu_percent = psutil.cpu_percent(interval=0.1)

                    # Memory usage
                    mem_info = psutil.virtual_memory()
                    mem_percent = mem_info.percent
                    mem_used_gb = round(mem_info.used / 1e9, 2)  # RAM used in GB

                    # TODO: Get actual uptime from system start
                    uptime = current_time  # Placeholder
                except Exception as e:
                    logger.warning(f"Failed to get system resources: {e}")
                    cpu_percent = 0.0
                    mem_percent = 0.0
                    mem_used_gb = 0.0
                    uptime = current_time

                # Get temperature and usage readings
                cpu_temp = get_cpu_temperature()
                gpu_temp = get_gpu_temperature()
                gpu_usage = get_gpu_usage()

                # Get rendering_index and renderer state from control state
                rendering_index_for_status = -1
                renderer_state_value = "STOPPED"  # Default fallback
                if control_state:
                    system_status_for_index = control_state.get_status_dict()
                    rendering_index_for_status = system_status_for_index.get("rendering_index", -1)

                    # Get renderer state
                    status_obj = control_state.get_status()
                    if status_obj and hasattr(status_obj, "renderer_state"):
                        renderer_state_value = status_obj.renderer_state.value

                # Broadcast system status
                status_data = {
                    "type": "system_status",
                    "is_online": True,
                    "current_file": (
                        playlist_state.items[rendering_index_for_status].file_path
                        if playlist_state.items and 0 <= rendering_index_for_status < len(playlist_state.items)
                        else None
                    ),
                    "playlist_position": playlist_state.current_index,
                    "rendering_index": rendering_index_for_status,  # Add rendering_index to status broadcast
                    "renderer_state": renderer_state_value,  # Add renderer state to status broadcast
                    "brightness": (get_system_settings().brightness if consumer_process else 1.0),
                    "frame_rate": 30.0,  # Will be updated from shared memory below
                    "uptime": uptime,
                    "memory_usage": mem_percent,
                    "memory_usage_gb": mem_used_gb,
                    "cpu_usage": cpu_percent,
                    "gpu_usage": gpu_usage,
                    "cpu_temperature": cpu_temp,
                    "gpu_temperature": gpu_temp,
                    "led_panel_connected": led_panel_connected,
                    "led_panel_status": led_panel_status,
                    "consumer_input_fps": consumer_input_fps,
                    "renderer_output_fps": renderer_output_fps,
                    "dropped_frames_percentage": dropped_frames_percentage,
                    "late_frame_percentage": late_frame_percentage,
                    "timestamp": current_time,
                }

                # Get preview data (same logic as /api/preview endpoint)
                preview_data = {
                    "type": "preview_data",
                    "timestamp": current_time,
                    "is_active": playlist_state.is_playing,
                    "current_item": None,
                    "has_frame": False,
                    "frame_data": None,
                }

                # Use rendering_index from control state to show currently rendered item
                rendering_index = -1
                if control_state:
                    system_status = control_state.get_status_dict()
                    rendering_index = system_status.get("rendering_index", -1)

                if playlist_state.items and 0 <= rendering_index < len(playlist_state.items):
                    current_item = playlist_state.items[rendering_index]
                    preview_data["current_item"] = {"name": current_item.name, "type": current_item.type}

                # Try to get real LED data from shared memory (PreviewSink)
                try:
                    import mmap
                    import os

                    # Try to read from shared memory created by PreviewSink
                    try:
                        shm_fd = os.open("/dev/shm/prismatron_preview", os.O_RDONLY)

                        # Read full header (64 bytes): timestamp(8) + frame_counter(8) + led_count(4) + padding(44)
                        header_data = os.read(shm_fd, 64)
                        if len(header_data) == 64:
                            import struct

                            # Unpack header according to PreviewSink format: "<ddiid32x"
                            timestamp, frame_counter, led_count, shm_rendering_index, shm_playback_position = (
                                struct.unpack("<ddiid32x", header_data)
                            )

                            if led_count > 0:
                                # Read LED data starting at offset 64: led_count * 3 bytes (RGB)
                                os.lseek(shm_fd, 64, os.SEEK_SET)  # Seek to LED data section
                                led_data = os.read(shm_fd, led_count * 3)

                                if len(led_data) == led_count * 3:
                                    # Convert to list of [r, g, b] arrays with brightness factor for preview
                                    frame_data = []
                                    brightness_factor = 0.5  # Reduce saturation due to overlapping LEDs
                                    for i in range(0, len(led_data), 3):
                                        r, g, b = led_data[i], led_data[i + 1], led_data[i + 2]
                                        # Apply brightness factor to reduce saturation
                                        r = int(r * brightness_factor)
                                        g = int(g * brightness_factor)
                                        b = int(b * brightness_factor)
                                        frame_data.append([r, g, b])

                                    preview_data["has_frame"] = True
                                    preview_data["frame_data"] = frame_data
                                    preview_data["total_leds"] = led_count
                                    preview_data["frame_counter"] = frame_counter
                                    # Handle potential invalid rendering_index values
                                    preview_data["shm_rendering_index"] = (
                                        shm_rendering_index if shm_rendering_index < 999999 else -1
                                    )
                                    # Add playback position (handle invalid values)
                                    preview_data["playback_position"] = (
                                        shm_playback_position if shm_playback_position >= 0 else 0.0
                                    )
                                    # Removed spammy playback position log

                                    # Debug: Compare rendering_index from different sources (reduced frequency)
                                    if hasattr(preview_broadcast_task, "_last_debug_time"):
                                        if current_time - preview_broadcast_task._last_debug_time > 10.0:
                                            logger.debug(
                                                f"Rendering index - ControlState={rendering_index}, SharedMemory={shm_rendering_index}"
                                            )
                                            preview_broadcast_task._last_debug_time = current_time
                                    else:
                                        preview_broadcast_task._last_debug_time = current_time

                        # Try to read statistics from shared memory
                        try:
                            import json

                            # Read statistics from shared memory (last 1024 bytes)
                            file_size = os.lseek(shm_fd, 0, os.SEEK_END)
                            stats_offset = file_size - 1024
                            os.lseek(shm_fd, stats_offset, os.SEEK_SET)
                            stats_data = os.read(shm_fd, 1024)

                            # Find null terminator
                            null_pos = stats_data.find(b"\x00")
                            if null_pos > 0:
                                stats_json = stats_data[:null_pos].decode("utf-8")
                                stats = json.loads(stats_json)

                                # Extract FPS and frame statistics
                                # Try renderer_fps first (if available), otherwise use ewma_fps (preview sink FPS)
                                frame_rate = stats.get("renderer_fps", stats.get("ewma_fps", 30.0))
                                status_data.update(
                                    {
                                        "frame_rate": frame_rate,
                                        "late_frame_percentage": stats.get(
                                            "renderer_late_percentage", stats.get("ewma_late_fraction", 0.0) * 100
                                        ),
                                        "dropped_frame_percentage": stats.get(
                                            "renderer_dropped_percentage", stats.get("ewma_dropped_fraction", 0.0) * 100
                                        ),
                                        "frames_processed": stats.get(
                                            "total_frames_rendered", stats.get("frames_processed", 0)
                                        ),
                                    }
                                )
                            else:
                                logger.debug(f"No null terminator found in stats data, length={len(stats_data)}")
                        except Exception as e:
                            logger.warning(f"Failed to read statistics from shared memory: {e}")

                        os.close(shm_fd)
                    except FileNotFoundError:
                        pass  # PreviewSink shared memory not found
                    except Exception:
                        pass  # Other errors accessing shared memory

                except Exception:
                    pass  # Failed to access preview shared memory

                # Broadcast system status with updated statistics
                await manager.broadcast(status_data)

                # Fallback to test pattern if no real data available
                if not preview_data["has_frame"]:
                    logger.debug("WebSocket using fallback rainbow test pattern - no shared memory data available")
                    preview_data["has_frame"] = True
                    preview_colors = []
                    brightness_factor = 0.5  # Reduce saturation due to overlapping LEDs
                    # Use the actual LED count for test pattern
                    test_led_count = get_actual_led_count()
                    for i in range(test_led_count):
                        # Simple rainbow pattern
                        hue = (i / test_led_count) * 360  # Full rainbow across all LEDs
                        r = int(255 * max(0, min(1, abs((hue / 60) % 6 - 3) - 1)) * brightness_factor)
                        g = int(255 * max(0, min(1, 2 - abs((hue / 60) % 6 - 2))) * brightness_factor)
                        b = int(255 * max(0, min(1, 2 - abs((hue / 60) % 6 - 4))) * brightness_factor)
                        preview_colors.append([r, g, b])
                    preview_data["frame_data"] = preview_colors
                    preview_data["total_leds"] = test_led_count
                    preview_data["shm_rendering_index"] = -1  # No shared memory data

                # Broadcast preview data to all connected clients
                await manager.broadcast(preview_data)

        except Exception as e:
            logger.warning(f"Preview broadcast error: {e}")

        # Wait for 200ms (5fps)
        await asyncio.sleep(0.2)


# API Routes


@app.get("/")
async def root():
    """Serve the main React application."""
    frontend_dir = Path(__file__).parent / "frontend" / "dist"
    index_file = frontend_dir / "index.html"

    if index_file.exists():
        response = FileResponse(str(index_file))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    else:
        return JSONResponse(
            {
                "message": "Prismatron API Server",
                "version": "1.0.0",
                "endpoints": {
                    "docs": "/api/docs",
                    "status": "/api/status",
                    "websocket": "/ws",
                },
            }
        )


# Captive Portal Detection Endpoints
# These endpoints handle various OS captive portal detection mechanisms
# They redirect to the main Prismatron interface when detected


@app.get("/generate_204")
async def android_captive_portal():
    """Android captive portal detection - expects 204 No Content normally."""
    # Redirect to main interface to trigger captive portal popup
    return RedirectResponse(url="/", status_code=302)


@app.get("/connectivitycheck.gstatic.com/generate_204")
async def chrome_captive_portal():
    """Chrome/Android captive portal detection."""
    return RedirectResponse(url="/", status_code=302)


@app.get("/hotspot-detect.html")
async def apple_captive_portal():
    """Apple iOS/macOS captive portal detection."""
    # Apple expects an HTML response
    return RedirectResponse(url="/", status_code=302)


@app.get("/captive.apple.com/hotspot-detect.html")
async def apple_captive_portal_alt():
    """Alternative Apple captive portal detection endpoint."""
    return RedirectResponse(url="/", status_code=302)


@app.get("/ncsi.txt")
async def windows_captive_portal():
    """Windows Network Connectivity Status Indicator."""
    # Windows expects specific text content
    return PlainTextResponse("Microsoft NCSI", status_code=200)


@app.get("/connecttest.txt")
async def windows_captive_portal_alt():
    """Windows 10 connectivity test."""
    return PlainTextResponse("Microsoft Connect Test", status_code=200)


@app.get("/www.msftconnecttest.com/connecttest.txt")
async def windows_captive_portal_full():
    """Windows 10 full URL connectivity test."""
    return PlainTextResponse("Microsoft Connect Test", status_code=200)


@app.get("/success.txt")
async def firefox_captive_portal():
    """Firefox captive portal detection."""
    # Firefox expects "success" response normally
    return RedirectResponse(url="/", status_code=302)


@app.get("/kindle-wifi/wifistub.html")
async def kindle_captive_portal():
    """Amazon Kindle captive portal detection."""
    return RedirectResponse(url="/", status_code=302)


@app.get("/library/test/success.html")
async def apple_library_test():
    """Apple library test endpoint."""
    return RedirectResponse(url="/", status_code=302)


# Home/Status endpoints
@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status."""
    # Get LED panel connection status and new metrics from consumer process
    led_panel_connected = False
    led_panel_status = "disconnected"
    consumer_input_fps = 0.0
    renderer_output_fps = 0.0
    dropped_frames_percentage = 0.0
    late_frame_percentage = 0.0

    # Get consumer statistics from ControlState (shared memory IPC)
    try:
        if control_state:
            system_status = control_state.get_status_dict()

            # Extract consumer statistics from ControlState
            consumer_input_fps = system_status.get("consumer_input_fps", 0.0)
            renderer_output_fps = system_status.get("renderer_output_fps", 0.0)
            dropped_frames_percentage = system_status.get("dropped_frames_percentage", 0.0)
            late_frame_percentage = system_status.get("late_frame_percentage", 0.0)

            # Debug log every 10 seconds to verify values
            current_time = time.time()
            if hasattr(get_system_status, "_last_status_debug_log"):
                if current_time - get_system_status._last_status_debug_log > 10.0:
                    logger.info(
                        f"STATUS API DEBUG (ControlState): input_fps={consumer_input_fps:.1f}, output_fps={renderer_output_fps:.1f}, dropped={dropped_frames_percentage:.1f}%"
                    )
                    get_system_status._last_status_debug_log = current_time
            else:
                get_system_status._last_status_debug_log = current_time

    except Exception as e:
        logger.warning(f"Failed to get consumer statistics from ControlState: {e}")

    # Try to get LED panel connection status from consumer process if available (fallback)
    if consumer_process and hasattr(consumer_process, "get_statistics"):
        try:
            consumer_stats = consumer_process.get_statistics()
            wled_stats = consumer_stats.get("wled_stats", {})
            led_panel_connected = wled_stats.get("is_connected", False)

            # Determine status string based on connection state
            if led_panel_connected:
                led_panel_status = "connected"
            elif consumer_process.is_running and not led_panel_connected:
                led_panel_status = "connecting"
            else:
                led_panel_status = "disconnected"
        except Exception as e:
            logger.warning(f"Failed to get LED panel status: {e}")
    else:
        # Default status when no consumer process reference
        led_panel_status = "unknown"

    # Get real system resource usage
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem_info = psutil.virtual_memory()
        mem_percent = mem_info.percent
        mem_used_gb = round(mem_info.used / 1e9, 2)  # RAM used in GB
        uptime = time.time()  # TODO: Get actual uptime
    except Exception as e:
        logger.warning(f"Failed to get system resources: {e}")
        cpu_percent = 0.0
        mem_percent = 0.0
        mem_used_gb = 0.0
        uptime = time.time()

    # Get temperature and usage readings
    cpu_temp = get_cpu_temperature()
    gpu_temp = get_gpu_temperature()
    gpu_usage = get_gpu_usage()

    # Get actual frame rate from shared memory
    frame_rate = 30.0  # Default fallback
    try:
        import json
        import os
        import struct

        shm_fd = os.open("/dev/shm/prismatron_preview", os.O_RDONLY)
        # Read statistics from shared memory (last 1024 bytes)
        file_size = os.lseek(shm_fd, 0, os.SEEK_END)
        stats_offset = file_size - 1024
        os.lseek(shm_fd, stats_offset, os.SEEK_SET)
        stats_data = os.read(shm_fd, 1024)

        # Find null terminator
        null_pos = stats_data.find(b"\x00")
        if null_pos > 0:
            stats_json = stats_data[:null_pos].decode("utf-8")
            stats = json.loads(stats_json)

            # Try renderer_fps first (if available), otherwise use ewma_fps (preview sink FPS)
            frame_rate = stats.get("renderer_fps", stats.get("ewma_fps", 30.0))

        os.close(shm_fd)
    except Exception:
        pass  # Use default frame_rate if shared memory not available

    # Get rendering_index and renderer state from control state to show currently rendered item
    rendering_index = -1
    renderer_state_value = "STOPPED"  # Default fallback
    if control_state:
        system_status_dict = control_state.get_status_dict()
        rendering_index = system_status_dict.get("rendering_index", -1)

        # Get renderer state
        status_obj = control_state.get_status()
        if status_obj and hasattr(status_obj, "renderer_state"):
            renderer_state_value = status_obj.renderer_state.value

    return SystemStatus(
        is_online=True,
        current_file=(
            playlist_state.items[rendering_index].file_path
            if playlist_state.items and 0 <= rendering_index < len(playlist_state.items)
            else None
        ),
        playlist_position=playlist_state.current_index,
        rendering_index=rendering_index,
        renderer_state=renderer_state_value,
        brightness=(get_system_settings().brightness if consumer_process else 1.0),
        frame_rate=frame_rate,
        uptime=uptime,
        memory_usage=mem_percent,
        memory_usage_gb=mem_used_gb,
        cpu_usage=cpu_percent,
        gpu_usage=gpu_usage,
        cpu_temperature=cpu_temp,
        gpu_temperature=gpu_temp,
        led_panel_connected=led_panel_connected,
        led_panel_status=led_panel_status,
        consumer_input_fps=consumer_input_fps,
        renderer_output_fps=renderer_output_fps,
        dropped_frames_percentage=dropped_frames_percentage,
        late_frame_percentage=late_frame_percentage,
    )


@app.post("/api/control/play")
async def play_content():
    """Start producer and resume renderer if paused."""
    try:
        logger.info("API PLAY REQUEST: Starting producer and resuming renderer")

        # Start producer via playlist sync service
        producer_started = False
        if playlist_sync_client and playlist_sync_client.connected:
            success = playlist_sync_client.play()
            if success:
                producer_started = True
            else:
                logger.error("Failed to send play command via sync service")
        else:
            logger.error("Playlist sync service not connected")

        # Handle renderer state based on current state
        renderer_handled = False
        if control_state:
            status = control_state.get_status()
            if status:
                if status.renderer_state == RendererState.PAUSED:
                    # Resume from pause directly to playing
                    renderer_handled = control_state.set_renderer_state(RendererState.PLAYING)
                elif status.renderer_state == RendererState.STOPPED:
                    # Start from stopped - go to waiting for frames
                    renderer_handled = control_state.set_renderer_state(RendererState.WAITING)
                    logger.info("Renderer set to WAITING for frames from producer")

        if producer_started:
            return {"status": "playing"}
        else:
            return {"status": "error", "message": "Failed to start producer"}

    except Exception as e:
        logger.error(f"Failed to start playback: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/control/pause")
async def pause_content():
    """Pause renderer only - producer continues generating frames."""
    try:
        logger.info("API PAUSE REQUEST: Pausing renderer only")

        # Pause renderer directly via control state
        if control_state and control_state.set_renderer_state(RendererState.PAUSED):
            return {"status": "paused"}
        else:
            logger.error("Failed to pause renderer")
            return {"status": "error", "message": "Control state unavailable"}

    except Exception as e:
        logger.error(f"Failed to pause renderer: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/control/stop")
async def stop_content():
    """Stop producer only - renderer will stop when buffer empties."""
    try:
        logger.info("API STOP REQUEST: Stopping producer only")

        # Stop producer via playlist sync service
        if playlist_sync_client and playlist_sync_client.connected:
            success = playlist_sync_client.pause()  # This stops the producer
            if success:
                return {"status": "stopped"}
            else:
                logger.error("Failed to send stop command via sync service")
                return {"status": "error", "message": "Playlist sync service unavailable"}
        else:
            logger.error("Playlist sync service not connected")
            return {"status": "error", "message": "Playlist sync service not connected"}

    except Exception as e:
        logger.error(f"Failed to stop producer: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/control/producer_state")
async def get_producer_state():
    """Get current producer state."""
    try:
        if control_state:
            status = control_state.get_status()
            if status:
                return {"producer_state": status.producer_state.value}
        return {"status": "error", "message": "Control state unavailable"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/control/renderer_state")
async def get_renderer_state():
    """Get current renderer state."""
    try:
        if control_state:
            status = control_state.get_status()
            if status:
                return {
                    "renderer_state": status.renderer_state.value,
                    "buffer_frames": status.led_buffer_frames,
                    "buffer_capacity": status.led_buffer_capacity,
                }
        return {"status": "error", "message": "Control state unavailable"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/control/next")
async def next_item():
    """Skip to next playlist item."""
    try:
        # Send next command to playlist sync service
        if playlist_sync_client and playlist_sync_client.connected:
            success = playlist_sync_client.next_item()
            if success:
                return {"current_index": playlist_state.current_index}
            else:
                logger.error("Failed to send next command via sync service")
                return {"status": "error", "message": "Playlist sync service unavailable"}
        else:
            logger.error("Playlist sync service not connected")
            return {"status": "error", "message": "Playlist sync service not connected"}

    except Exception as e:
        logger.error(f"Failed to skip to next item: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/control/previous")
async def previous_item():
    """Skip to previous playlist item."""
    try:
        # Send previous command to playlist sync service
        if playlist_sync_client and playlist_sync_client.connected:
            success = playlist_sync_client.previous_item()
            if success:
                return {"current_index": playlist_state.current_index}
            else:
                logger.error("Failed to send previous command via sync service")
                return {"status": "error", "message": "Playlist sync service unavailable"}
        else:
            logger.error("Playlist sync service not connected")
            return {"status": "error", "message": "Playlist sync service not connected"}

    except Exception as e:
        logger.error(f"Failed to skip to previous item: {e}")
        return {"status": "error", "message": str(e)}


# Upload endpoints
@app.get("/api/uploads")
async def list_uploads():
    """List existing files in the uploads directory."""
    try:
        uploads = []
        if UPLOAD_DIR.exists():
            for file_path in UPLOAD_DIR.iterdir():
                if file_path.is_file():
                    # Determine file type
                    file_ext = file_path.suffix.lower().lstrip(".")
                    allowed_types = {
                        "image": ["jpg", "jpeg", "png", "gif", "bmp", "webp"],
                        "video": ["mp4", "avi", "mov", "mkv", "webm", "m4v"],
                    }

                    content_type = None
                    for type_name, extensions in allowed_types.items():
                        if file_ext in extensions:
                            content_type = type_name
                            break

                    if content_type:
                        file_stats = file_path.stat()
                        uploads.append(
                            {
                                "id": file_path.stem,  # filename without extension
                                "name": file_path.name,
                                "type": content_type,
                                "size": file_stats.st_size,
                                "modified": file_stats.st_mtime,
                                "path": str(file_path),
                            }
                        )

        # Sort by modification time (newest first)
        uploads.sort(key=lambda x: x["modified"], reverse=True)
        return {"files": uploads}

    except Exception as e:
        logger.error(f"Failed to list uploads: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/uploads/{file_id}/add")
async def add_existing_file_to_playlist(file_id: str, name: Optional[str] = None, duration: Optional[float] = None):
    """Add an existing file from uploads to the playlist."""
    try:
        # Find the file in uploads directory
        file_path = None
        for candidate in UPLOAD_DIR.iterdir():
            if candidate.is_file() and candidate.stem == file_id:
                file_path = candidate
                break

        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")

        # Determine file type
        file_ext = file_path.suffix.lower().lstrip(".")
        allowed_types = {
            "image": ["jpg", "jpeg", "png", "gif", "bmp", "webp"],
            "video": ["mp4", "avi", "mov", "mkv", "webm", "m4v"],
        }

        content_type = None
        for type_name, extensions in allowed_types.items():
            if file_ext in extensions:
                content_type = type_name
                break

        if not content_type:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

        # Create playlist item
        item = PlaylistItem(
            id=str(uuid.uuid4()),
            name=name or file_path.name,
            type=content_type,
            file_path=str(file_path),
            duration=duration,
            order=len(playlist_state.items),
        )

        # Send to playlist sync service
        if playlist_sync_client and playlist_sync_client.connected:
            sync_item = api_item_to_sync_item(item)
            success = playlist_sync_client.add_item(sync_item)
            if not success:
                logger.error("Failed to add item to playlist sync service")
                raise HTTPException(status_code=500, detail="Failed to add item to playlist")
        else:
            logger.error("Playlist sync service not available")
            raise HTTPException(status_code=503, detail="Playlist sync service not available")

        # Note: WebSocket broadcast will happen via sync service callback
        return {"status": "added", "item": item.dict_serializable()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add existing file to playlist: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),  # noqa: B008
    name: Optional[str] = Form(None),
    duration: Optional[float] = Form(None),
):
    """Upload image or video file and add to playlist."""
    try:
        # Validate file type
        allowed_types = {
            "image": ["jpg", "jpeg", "png", "gif", "bmp", "webp"],
            "video": ["mp4", "avi", "mov", "mkv", "webm", "m4v"],
        }

        file_ext = file.filename.split(".")[-1].lower() if file.filename else ""

        content_type = None
        for type_name, extensions in allowed_types.items():
            if file_ext in extensions:
                content_type = type_name
                break

        if not content_type:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

        # Generate unique ID for playlist item
        file_id = str(uuid.uuid4())

        # Use original filename and save to temp_conversions with _upload suffix
        original_name = name or file.filename or f"uploaded_file.{file_ext}"
        base_name = Path(original_name).stem
        temp_file_path = Path("temp_conversions") / f"{base_name}_upload.{file_ext}"

        # Ensure temp_conversions directory exists
        temp_file_path.parent.mkdir(exist_ok=True)

        # Save uploaded file to temp location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Handle video vs image differently
        if content_type == "video":
            # Queue video for conversion instead of immediate playlist addition
            if VIDEO_CONVERSION_AVAILABLE:
                conversion_manager = get_conversion_manager()
                if conversion_manager:
                    job = conversion_manager.queue_conversion(temp_file_path, original_name)

                    # Register callback to add to playlist when conversion completes
                    def on_conversion_complete(completed_job):
                        if completed_job.id == job.id and completed_job.status == ConversionStatus.COMPLETED:
                            # Create playlist item with converted file
                            converted_item = PlaylistItem(
                                id=file_id,
                                name=original_name,
                                type="video",
                                file_path=str(completed_job.output_path),
                                duration=duration,
                                order=len(playlist_state.items),
                                transition_in=generate_random_led_transition(),
                                transition_out=generate_random_led_transition(),
                            )

                            # Add to uploads playlist
                            # manage_uploads_playlist will also add to live playlist if uploads playlist is active
                            manage_uploads_playlist(converted_item)

                            logger.info(f"Added converted video to uploads playlist: {original_name}")

                    conversion_manager.add_status_callback(on_conversion_complete)

                    return {
                        "status": "queued_for_conversion",
                        "job_id": job.id,
                        "message": "Video queued for conversion to H.264/800x480",
                    }
                else:
                    raise HTTPException(status_code=503, detail="Video conversion manager not available")
            else:
                raise HTTPException(status_code=503, detail="Video conversion not available")
        else:
            # Handle images - move from temp to uploads directory
            final_image_path = UPLOAD_DIR / original_name

            try:
                shutil.move(str(temp_file_path), str(final_image_path))
                logger.info(f"Moved image from temp to uploads: {final_image_path}")
            except Exception as e:
                logger.error(f"Failed to move image file: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to process image file: {e}") from e

            # Set default duration for images to 10s as requested
            if duration is None:
                duration = 10.0

            # Create playlist item with random LED transitions
            transition_in = generate_random_led_transition()
            transition_out = generate_random_led_transition()

            item = PlaylistItem(
                id=file_id,
                name=original_name,
                type=content_type,
                file_path=str(final_image_path),
                duration=duration,
                order=len(playlist_state.items),
                transition_in=transition_in,
                transition_out=transition_out,
            )

            # Validate the item (including transitions)
            validation_errors = validate_playlist_item(item)
            if validation_errors:
                raise HTTPException(status_code=400, detail={"errors": validation_errors})

            # Automatically manage uploads playlist (separate from main playlist)
            manage_uploads_playlist(item)

            # Send to main playlist sync service
            if playlist_sync_client and playlist_sync_client.connected:
                sync_item = api_item_to_sync_item(item)
                success = playlist_sync_client.add_item(sync_item)
                if not success:
                    logger.warning("Failed to add item to playlist sync service, adding locally")
                    playlist_state.items.append(item)
            else:
                logger.warning("Playlist sync service not available, adding locally")
                playlist_state.items.append(item)

            # Note: WebSocket broadcast will happen via sync service callback
            return {"status": "uploaded", "item": item.dict_serializable()}

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/uploads/{file_id}/move-to-media")
async def move_upload_to_media(file_id: str):
    """Move a file from uploads folder to media folder."""
    try:
        # Find the file in uploads directory
        source_path = None
        for candidate in UPLOAD_DIR.iterdir():
            if candidate.is_file() and candidate.stem == file_id:
                source_path = candidate
                break

        if not source_path:
            raise HTTPException(status_code=404, detail="File not found in uploads")

        # Create destination path in media directory
        dest_path = MEDIA_DIR / source_path.name

        # Check if file already exists in media directory
        if dest_path.exists():
            # Generate unique filename
            name_parts = source_path.stem, source_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = MEDIA_DIR / f"{name_parts[0]}_{counter}{name_parts[1]}"
                counter += 1

        # Move the file
        shutil.move(str(source_path), str(dest_path))

        # Determine file type
        file_ext = dest_path.suffix.lower().lstrip(".")
        allowed_types = {
            "image": ["jpg", "jpeg", "png", "gif", "bmp", "webp"],
            "video": ["mp4", "avi", "mov", "mkv", "webm", "m4v"],
        }

        content_type = None
        for type_name, extensions in allowed_types.items():
            if file_ext in extensions:
                content_type = type_name
                break

        if not content_type:
            # Move back if unsupported type
            shutil.move(str(dest_path), str(source_path))
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

        # Get file stats
        file_stats = dest_path.stat()

        logger.info(f"Moved file from uploads to media: {source_path.name} -> {dest_path.name}")

        # Remove from uploads playlist and sync with live playlist if active
        playlist_data = load_uploads_playlist()
        items_to_remove = []

        # Find all playlist items that reference the moved file (using original source_path)
        for item in playlist_data["items"]:
            if item.get("file_path") == str(source_path):
                items_to_remove.append(item)
                logger.info(f"Found playlist item to remove: {item['name']} (id={item['id']})")

        # Remove items from uploads playlist and sync
        if items_to_remove:
            remove_files_from_uploads_playlist_and_sync(items_to_remove)
            logger.info(f"Removed {len(items_to_remove)} items from uploads playlist after moving to media")

        return {
            "status": "moved",
            "file": {
                "id": dest_path.stem,
                "name": dest_path.name,
                "type": content_type,
                "size": file_stats.st_size,
                "modified": file_stats.st_mtime,
                "path": str(dest_path),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to move file from uploads to media: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Video Conversion endpoints
@app.get("/api/conversions")
async def list_conversions():
    """List active and recent video conversion jobs."""
    try:
        if not VIDEO_CONVERSION_AVAILABLE:
            raise HTTPException(status_code=503, detail="Video conversion not available")

        conversion_manager = get_conversion_manager()
        if not conversion_manager:
            raise HTTPException(status_code=503, detail="Video conversion manager not initialized")

        jobs = conversion_manager.get_all_jobs()
        return {"conversions": [job.to_dict() for job in jobs]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list conversions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/conversions/{job_id}")
async def get_conversion(job_id: str):
    """Get details of a specific conversion job."""
    try:
        if not VIDEO_CONVERSION_AVAILABLE:
            raise HTTPException(status_code=503, detail="Video conversion not available")

        conversion_manager = get_conversion_manager()
        if not conversion_manager:
            raise HTTPException(status_code=503, detail="Video conversion manager not initialized")

        job = conversion_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Conversion job not found")

        return {"conversion": job.to_dict()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversion {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/conversions/{job_id}/cancel")
async def cancel_conversion(job_id: str):
    """Cancel a video conversion job."""
    try:
        if not VIDEO_CONVERSION_AVAILABLE:
            raise HTTPException(status_code=503, detail="Video conversion not available")

        conversion_manager = get_conversion_manager()
        if not conversion_manager:
            raise HTTPException(status_code=503, detail="Video conversion manager not initialized")

        success = conversion_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversion job not found or cannot be cancelled")

        return {"status": "cancelled", "job_id": job_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel conversion {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/api/conversions/{job_id}")
async def remove_conversion(job_id: str):
    """Remove a completed conversion job from the list."""
    try:
        if not VIDEO_CONVERSION_AVAILABLE:
            raise HTTPException(status_code=503, detail="Video conversion not available")

        conversion_manager = get_conversion_manager()
        if not conversion_manager:
            raise HTTPException(status_code=503, detail="Video conversion manager not initialized")

        job = conversion_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Conversion job not found")

        if job.status not in [ConversionStatus.COMPLETED, ConversionStatus.FAILED, ConversionStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail="Can only remove completed, failed, or cancelled jobs")

        # Remove from queue (this is a simple implementation)
        with conversion_manager.lock:
            conversion_manager.conversion_queue = [j for j in conversion_manager.conversion_queue if j.id != job_id]

        return {"status": "removed", "job_id": job_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove conversion {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Media endpoints
@app.get("/api/media")
async def list_media():
    """List existing files in the media directory."""
    try:
        media_files = []
        if MEDIA_DIR.exists():
            for file_path in MEDIA_DIR.iterdir():
                if file_path.is_file():
                    # Determine file type
                    file_ext = file_path.suffix.lower().lstrip(".")
                    allowed_types = {
                        "image": ["jpg", "jpeg", "png", "gif", "bmp", "webp"],
                        "video": ["mp4", "avi", "mov", "mkv", "webm", "m4v"],
                    }

                    content_type = None
                    for type_name, extensions in allowed_types.items():
                        if file_ext in extensions:
                            content_type = type_name
                            break

                    if content_type:
                        file_stats = file_path.stat()
                        media_files.append(
                            {
                                "id": file_path.stem,  # filename without extension
                                "name": file_path.name,
                                "type": content_type,
                                "size": file_stats.st_size,
                                "modified": file_stats.st_mtime,
                                "path": str(file_path),
                            }
                        )

        # Sort by modification time (newest first)
        media_files.sort(key=lambda x: x["modified"], reverse=True)
        return {"files": media_files}

    except Exception as e:
        logger.error(f"Failed to list media files: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/media/{file_id}/add")
async def add_existing_media_file_to_playlist(
    file_id: str, name: Optional[str] = None, duration: Optional[float] = None
):
    """Add an existing file from media directory to the playlist."""
    try:
        # Find the file in media directory
        file_path = None
        for candidate in MEDIA_DIR.iterdir():
            if candidate.is_file() and candidate.stem == file_id:
                file_path = candidate
                break

        if not file_path:
            raise HTTPException(status_code=404, detail="File not found")

        # Determine file type
        file_ext = file_path.suffix.lower().lstrip(".")
        allowed_types = {
            "image": ["jpg", "jpeg", "png", "gif", "bmp", "webp"],
            "video": ["mp4", "avi", "mov", "mkv", "webm", "m4v"],
        }

        content_type = None
        for type_name, extensions in allowed_types.items():
            if file_ext in extensions:
                content_type = type_name
                break

        if not content_type:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

        # Create playlist item
        item = PlaylistItem(
            id=str(uuid.uuid4()),
            name=name or file_path.name,
            type=content_type,
            file_path=str(file_path),
            duration=duration,
            order=len(playlist_state.items),
        )

        # Send to playlist sync service
        if playlist_sync_client and playlist_sync_client.connected:
            sync_item = api_item_to_sync_item(item)
            success = playlist_sync_client.add_item(sync_item)
            if not success:
                logger.error("Failed to add item to playlist sync service")
                raise HTTPException(status_code=500, detail="Failed to add item to playlist")
        else:
            logger.error("Playlist sync service not available")
            raise HTTPException(status_code=503, detail="Playlist sync service not available")

        # Note: WebSocket broadcast will happen via sync service callback
        return {"status": "added", "item": item.dict_serializable()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add existing media file to playlist: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class RenameFileRequest(BaseModel):
    """Request model for renaming a file."""

    new_name: str = Field(..., description="New filename")


@app.post("/api/media/{file_id}/rename")
async def rename_media_file(file_id: str, request: RenameFileRequest):
    """Rename a file in the media directory and update all playlists."""
    try:
        # Sanitize new name - remove path separators and dangerous characters
        new_name = request.new_name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="New filename cannot be empty")

        # Remove path separators
        new_name = new_name.replace("/", "").replace("\\", "")

        # Find the file in media directory
        old_file_path = None
        for candidate in MEDIA_DIR.iterdir():
            if candidate.is_file() and candidate.stem == file_id:
                old_file_path = candidate
                break

        if not old_file_path:
            raise HTTPException(status_code=404, detail="File not found in media directory")

        old_filename = old_file_path.name
        old_extension = old_file_path.suffix

        # If new_name doesn't have extension, use the old one
        if not Path(new_name).suffix:
            new_name = new_name + old_extension
        else:
            # Validate that extension matches
            new_extension = Path(new_name).suffix
            if new_extension.lower() != old_extension.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot change file extension from {old_extension} to {new_extension}",
                )

        # Check if destination already exists
        new_file_path = MEDIA_DIR / new_name
        if new_file_path.exists():
            raise HTTPException(status_code=409, detail=f"File with name '{new_name}' already exists")

        # Rename the physical file
        old_file_path.rename(new_file_path)
        logger.info(f"Renamed file: {old_file_path} -> {new_file_path}")

        # Update all playlists
        update_result = update_playlists_for_file_operation(old_filename, new_name, operation="rename")

        return {
            "status": "renamed",
            "old_name": old_filename,
            "new_name": new_name,
            "affected_playlists": update_result["affected_playlists"],
            "items_modified": update_result["items_modified"],
            "playlists": update_result["playlists"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rename media file: {e}")
        # Try to rename back if possible
        if "new_file_path" in locals() and new_file_path.exists() and "old_file_path" in locals():
            try:
                new_file_path.rename(old_file_path)
                logger.info(f"Rolled back rename: {new_file_path} -> {old_file_path}")
            except Exception as rollback_error:
                logger.error(f"Failed to rollback rename: {rollback_error}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/api/media/{file_id}")
async def delete_media_file(file_id: str):
    """Delete a file from the media directory and remove from all playlists."""
    try:
        # Find the file in media directory
        file_path = None
        for candidate in MEDIA_DIR.iterdir():
            if candidate.is_file() and candidate.stem == file_id:
                file_path = candidate
                break

        if not file_path:
            raise HTTPException(status_code=404, detail="File not found in media directory")

        filename = file_path.name

        # Update all playlists first (before deleting file)
        update_result = update_playlists_for_file_operation(filename, operation="delete")

        # Delete the physical file
        file_path.unlink()
        logger.info(f"Deleted file: {file_path}")

        return {
            "status": "deleted",
            "filename": filename,
            "affected_playlists": update_result["affected_playlists"],
            "items_removed": update_result["items_modified"],
            "playlists": update_result["playlists"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete media file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Effects endpoints
@app.get("/api/effects", response_model=List[EffectPreset])
async def get_effects():
    """Get available effect presets."""
    return EFFECT_PRESETS


class AddEffectRequest(BaseModel):
    """Request model for adding an effect to playlist."""

    name: Optional[str] = None
    duration: Optional[float] = None
    config: Optional[Dict] = None


@app.post("/api/effects/{effect_id}/add")
async def add_effect_to_playlist(
    effect_id: str,
    request: AddEffectRequest,
):
    """Add an effect to the playlist."""
    logger.info(f"Adding effect {effect_id} with request: name='{request.name}', config={request.config}")

    # Find effect preset
    effect_preset = next((e for e in EFFECT_PRESETS if e.id == effect_id), None)
    if not effect_preset:
        raise HTTPException(status_code=404, detail="Effect not found")

    # Merge custom config with preset
    final_config = effect_preset.config.copy()
    if request.config:
        final_config.update(request.config)

    logger.info(f"Final config for {effect_id}: {final_config}")

    # Handle text effects specially
    if effect_id == "text_display":
        # For text effects, we create a JSON config string as the "file_path"
        text_config = json.dumps(final_config)

        # Create playlist item for text content
        final_name = request.name or f"Text: {final_config.get('text', 'Hello World')}"
        logger.info(
            f"Creating text effect with name: '{final_name}' (request.name='{request.name}', text='{final_config.get('text')}')"
        )

        # Use duration from config or request, with fallback to 10.0
        text_duration = request.duration or final_config.get("duration", 10.0)

        item = PlaylistItem(
            id=str(uuid.uuid4()),
            name=final_name,
            type="text",
            file_path=text_config,  # JSON config as file path for text content
            duration=text_duration,
            order=len(playlist_state.items),
        )
    else:
        # Create standard effect playlist item
        item = PlaylistItem(
            id=str(uuid.uuid4()),
            name=request.name or effect_preset.name,
            type="effect",
            effect_config={"effect_id": effect_id, "parameters": final_config},
            duration=request.duration or 30.0,  # Default 30 seconds for effects
            order=len(playlist_state.items),
        )

    # Send to playlist sync service
    if playlist_sync_client and playlist_sync_client.connected:
        sync_item = api_item_to_sync_item(item)
        success = playlist_sync_client.add_item(sync_item)
        if not success:
            logger.warning("Failed to add item to playlist sync service, adding locally")
            playlist_state.items.append(item)
    else:
        logger.warning("Playlist sync service not available, adding locally")
        playlist_state.items.append(item)

    # Note: WebSocket broadcast will happen via sync service callback
    return {"status": "added", "item": item.dict_serializable()}


# Playlist endpoints
@app.get("/api/playlist", response_model=PlaylistState)
async def get_playlist():
    """Get current playlist state."""
    return playlist_state


@app.post("/api/playlist/reorder")
async def reorder_playlist(item_ids: List[str]):
    """Reorder playlist items."""
    # Send reorder command to playlist sync service
    if playlist_sync_client and playlist_sync_client.connected:
        success = playlist_sync_client.reorder_items(item_ids)
        if not success:
            logger.warning("Failed to reorder items via sync service, applying locally")
            # Fallback to local reordering
            item_map = {item.id: item for item in playlist_state.items}
            reordered_items = []
            for i, item_id in enumerate(item_ids):
                if item_id in item_map:
                    item = item_map[item_id]
                    item.order = i
                    reordered_items.append(item)
            playlist_state.items = reordered_items
    else:
        logger.warning("Playlist sync service not available, applying reorder locally")
        # Fallback to local reordering
        item_map = {item.id: item for item in playlist_state.items}
        reordered_items = []
        for i, item_id in enumerate(item_ids):
            if item_id in item_map:
                item = item_map[item_id]
                item.order = i
                reordered_items.append(item)
        playlist_state.items = reordered_items

    # Note: WebSocket broadcast will happen via sync service callback
    return {"status": "reordered"}


@app.delete("/api/playlist/{item_id}")
async def remove_playlist_item(item_id: str):
    """Remove item from playlist."""
    # Send remove command to playlist sync service
    if playlist_sync_client and playlist_sync_client.connected:
        success = playlist_sync_client.remove_item(item_id)
        if success:
            return {"status": "removed"}
        else:
            logger.warning("Failed to remove item via sync service")
            raise HTTPException(status_code=500, detail="Failed to remove item")
    else:
        logger.warning("Playlist sync service not available")

        # Fallback to local removal
        original_length = len(playlist_state.items)
        removed_item = None

        # Find the item to remove
        for item in playlist_state.items:
            if item.id == item_id:
                removed_item = item
                break

        playlist_state.items = [item for item in playlist_state.items if item.id != item_id]

        if len(playlist_state.items) < original_length and removed_item:
            # Adjust current index if needed
            if playlist_state.current_index >= len(playlist_state.items):
                playlist_state.current_index = max(0, len(playlist_state.items) - 1)

            return {"status": "removed"}
        else:
            raise HTTPException(status_code=404, detail="Item not found")


@app.post("/api/playlist/clear")
async def clear_playlist():
    """Clear all playlist items."""
    global current_playlist_file

    # Send clear command to playlist sync service
    if playlist_sync_client and playlist_sync_client.connected:
        success = playlist_sync_client.clear_playlist()
        if success:
            # Clear the current playlist file tracking
            current_playlist_file = None
            return {"status": "cleared"}
        else:
            logger.warning("Failed to clear playlist via sync service")
            raise HTTPException(status_code=500, detail="Failed to clear playlist")
    else:
        logger.warning("Playlist sync service not available")

        # Fallback to local clear
        playlist_state.items.clear()
        playlist_state.current_index = 0
        playlist_state.is_playing = False

        # Clear the current playlist file tracking
        current_playlist_file = None

        return {"status": "cleared"}


@app.post("/api/playlist/shuffle")
async def toggle_shuffle():
    """Toggle shuffle mode."""
    playlist_state.shuffle = not playlist_state.shuffle

    await manager.broadcast(
        {
            "type": "playlist_state",
            "shuffle": playlist_state.shuffle,
            "auto_repeat": playlist_state.auto_repeat,
        }
    )

    return {"shuffle": playlist_state.shuffle}


@app.post("/api/playlist/repeat")
async def toggle_repeat():
    """Toggle auto-repeat mode."""
    playlist_state.auto_repeat = not playlist_state.auto_repeat

    await manager.broadcast(
        {
            "type": "playlist_state",
            "shuffle": playlist_state.shuffle,
            "auto_repeat": playlist_state.auto_repeat,
        }
    )

    return {"auto_repeat": playlist_state.auto_repeat}


# Transition endpoints
@app.get("/api/transitions")
async def get_transition_types():
    """Get available transition types and their schemas."""
    from ..transitions.transition_factory import get_transition_factory

    try:
        # Get all transition schemas (image + LED)
        factory = get_transition_factory()
        all_schemas = factory.get_all_schemas_with_led()

        # Convert to API format with user-friendly names
        types = []

        # Define display names and descriptions for each transition type
        transition_info = {
            "none": {"name": "None", "description": "No transition"},
            "fade": {"name": "Fade", "description": "Fade in/out transition"},
            "blur": {"name": "Blur", "description": "Gaussian blur in/out transition"},
            "ledfade": {"name": "LED Fade", "description": "Direct LED brightness fade (more efficient)"},
            "ledblur": {"name": "LED Blur", "description": "1D spatial blur effect on LED array"},
            "ledrandom": {"name": "LED Random", "description": "Random sparkle/lighting pattern effect"},
        }

        for trans_type, schema in all_schemas.items():
            info = transition_info.get(
                trans_type, {"name": trans_type.title(), "description": f"{trans_type} transition"}
            )

            # Convert schema properties to API format
            api_parameters = {}
            if "properties" in schema:
                for param_name, param_schema in schema["properties"].items():
                    api_param = {
                        "type": param_schema.get("type", "string"),
                        "default": param_schema.get("default"),
                        "description": param_schema.get("description", ""),
                    }

                    # Add constraints if present
                    if "minimum" in param_schema:
                        api_param["min"] = param_schema["minimum"]
                    if "maximum" in param_schema:
                        api_param["max"] = param_schema["maximum"]
                    if "enum" in param_schema:
                        api_param["options"] = param_schema["enum"]

                    api_parameters[param_name] = api_param

            types.append(
                {
                    "type": trans_type,
                    "name": info["name"],
                    "description": info["description"],
                    "parameters": api_parameters,
                }
            )

        return {"types": types}

    except Exception as e:
        logger.error(f"Error getting transition types: {e}")
        # Fallback to basic types
        return {
            "types": [
                {"type": "none", "name": "None", "description": "No transition", "parameters": {}},
                {"type": "fade", "name": "Fade", "description": "Fade in/out transition", "parameters": {}},
                {"type": "ledfade", "name": "LED Fade", "description": "Direct LED brightness fade", "parameters": {}},
                {"type": "ledblur", "name": "LED Blur", "description": "1D spatial blur effect", "parameters": {}},
                {"type": "ledrandom", "name": "LED Random", "description": "Random sparkle effect", "parameters": {}},
            ]
        }


@app.put("/api/playlist/{item_id}/transitions")
async def update_item_transitions(item_id: str, transition_in: TransitionConfig, transition_out: TransitionConfig):
    """Update transition configurations for a playlist item."""
    try:
        # Validate transition configurations
        validation_errors = {}

        transition_in_errors = validate_transition_config(transition_in)
        for key, error in transition_in_errors.items():
            validation_errors[f"transition_in.{key}"] = error

        transition_out_errors = validate_transition_config(transition_out)
        for key, error in transition_out_errors.items():
            validation_errors[f"transition_out.{key}"] = error

        if validation_errors:
            raise HTTPException(status_code=400, detail={"errors": validation_errors})

        # Log transition configuration update for debugging
        logger.info(
            f"API: Updating transitions for item {item_id}: "
            f"in={transition_in.type}({transition_in.parameters}), "
            f"out={transition_out.type}({transition_out.parameters})"
        )

        # Send update to playlist sync service
        if playlist_sync_client and playlist_sync_client.connected:
            # Convert API transition configs to sync transition configs
            sync_transition_in = SyncTransitionConfig(type=transition_in.type, parameters=transition_in.parameters)
            sync_transition_out = SyncTransitionConfig(type=transition_out.type, parameters=transition_out.parameters)

            # Send update via sync service
            success = playlist_sync_client.update_item_transitions(item_id, sync_transition_in, sync_transition_out)

            if success:
                # Log successful transmission to consumer
                logger.info(f"API: Successfully sent transition update for item {item_id} to consumer via sync service")
                # Note: WebSocket broadcast will happen via sync service callback when the update comes back
                return {
                    "status": "updated",
                    "item_id": item_id,
                    "transition_in": transition_in.dict_serializable(),
                    "transition_out": transition_out.dict_serializable(),
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to update transitions via sync service")
        else:
            # Update local playlist state if sync service not available
            item_found = False
            for item in playlist_state.items:
                if item.id == item_id:
                    item.transition_in = transition_in
                    item.transition_out = transition_out
                    item_found = True
                    break

            if not item_found:
                raise HTTPException(status_code=404, detail="Playlist item not found")

            return {
                "status": "updated",
                "item_id": item_id,
                "transition_in": transition_in.dict_serializable(),
                "transition_out": transition_out.dict_serializable(),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating item transitions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Settings endpoints
@app.get("/api/settings", response_model=SystemSettings)
async def get_settings():
    """Get current system settings."""
    # Get current audio reactive setting from control state
    audio_reactive_enabled = False
    if control_state:
        try:
            status = control_state.get_status()
            if status:
                audio_reactive_enabled = status.audio_reactive_enabled
        except Exception as e:
            logger.warning(f"Failed to get audio reactive status: {e}")

    # Update system settings with current control state values
    current_settings = get_system_settings().model_copy()
    current_settings.audio_reactive_enabled = audio_reactive_enabled

    return current_settings


@app.post("/api/settings")
async def update_settings(settings: SystemSettings):
    """Update system settings."""
    global system_settings
    system_settings = settings

    # TODO: Apply settings to actual system

    await manager.broadcast({"type": "settings_updated", "settings": settings.dict()})

    return {"status": "updated", "settings": settings.dict()}


@app.post("/api/settings/brightness")
async def set_brightness(brightness: float):
    """Set global brightness."""
    if not 0.0 <= brightness <= 1.0:
        raise HTTPException(status_code=400, detail="Brightness must be between 0.0 and 1.0")

    system_settings.brightness = brightness

    # TODO: Apply brightness to actual system

    await manager.broadcast({"type": "brightness_changed", "brightness": brightness})

    return {"brightness": brightness}


class AudioReactiveRequest(BaseModel):
    """Request model for audio reactive setting."""

    enabled: bool = Field(..., description="Whether audio reactive effects are enabled")


@app.get("/api/settings/audio-reactive")
async def get_audio_reactive_settings():
    """Get current audio reactive settings."""
    try:
        # Get current settings from control state
        audio_reactive_enabled = False
        position_shifting_enabled = False
        max_shift_distance = 3
        shift_direction = "alternating"

        # Beat brightness boost settings
        beat_brightness_enabled = True
        beat_brightness_intensity = 0.25
        beat_brightness_duration = 0.25

        if control_state:
            try:
                status = control_state.get_status()
                if status:
                    audio_reactive_enabled = status.audio_reactive_enabled
                    position_shifting_enabled = status.position_shifting_enabled
                    max_shift_distance = status.max_shift_distance
                    shift_direction = status.shift_direction
                    beat_brightness_enabled = getattr(status, "beat_brightness_enabled", True)
                    beat_brightness_intensity = getattr(status, "beat_brightness_intensity", 0.25)
                    beat_brightness_duration = getattr(status, "beat_brightness_duration", 0.25)
            except Exception as e:
                logger.warning(f"Failed to get audio reactive status: {e}")

        return {
            "enabled": audio_reactive_enabled,
            "position_shifting_enabled": position_shifting_enabled,
            "max_shift_distance": max_shift_distance,
            "shift_direction": shift_direction,
            "beat_brightness_enabled": beat_brightness_enabled,
            "beat_brightness_intensity": beat_brightness_intensity,
            "beat_brightness_duration": beat_brightness_duration,
        }

    except Exception as e:
        logger.error(f"Failed to get audio reactive settings: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/settings/audio-reactive")
async def set_audio_reactive_enabled(request: AudioReactiveRequest):
    """Set audio reactive effects enabled/disabled."""
    try:
        # Update in control state if available
        if control_state:
            control_state.update_status(audio_reactive_enabled=request.enabled)
            logger.info(f"Updated audio reactive enabled to {request.enabled}")
        else:
            logger.warning("Control state not available - audio reactive setting not updated")

        await manager.broadcast({"type": "audio_reactive_changed", "enabled": request.enabled})

        return {"enabled": request.enabled, "status": "updated"}

    except Exception as e:
        logger.error(f"Failed to set audio reactive enabled: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class PositionShiftingRequest(BaseModel):
    """Request model for position shifting settings."""

    enabled: bool = Field(..., description="Whether position shifting is enabled")
    max_shift_distance: int = Field(3, ge=1, le=10, description="Maximum shift distance (1-10)")
    shift_direction: str = Field("alternating", description="Shift direction: left, right, alternating")


class BeatBrightnessRequest(BaseModel):
    """Request model for beat brightness boost settings."""

    enabled: bool = Field(..., description="Whether beat brightness boost is enabled")
    intensity: float = Field(0.25, ge=0.0, le=1.0, description="Brightness boost intensity (0.0-1.0)")
    duration: float = Field(0.25, ge=0.1, le=1.0, description="Boost duration as fraction of beat (0.1-1.0)")


@app.post("/api/settings/position-shifting")
async def set_position_shifting_settings(request: PositionShiftingRequest):
    """Set position shifting settings."""
    try:
        # Validate shift direction
        valid_directions = ["left", "right", "alternating"]
        if request.shift_direction not in valid_directions:
            raise HTTPException(
                status_code=400, detail=f"Invalid shift direction. Must be one of: {', '.join(valid_directions)}"
            )

        # Update in control state if available
        if control_state:
            control_state.update_status(
                position_shifting_enabled=request.enabled,
                max_shift_distance=request.max_shift_distance,
                shift_direction=request.shift_direction,
            )
            logger.info(
                f"Updated position shifting: enabled={request.enabled}, distance={request.max_shift_distance}, direction={request.shift_direction}"
            )
        else:
            logger.warning("Control state not available - position shifting settings not updated")

        await manager.broadcast(
            {
                "type": "position_shifting_changed",
                "enabled": request.enabled,
                "max_shift_distance": request.max_shift_distance,
                "shift_direction": request.shift_direction,
            }
        )

        return {
            "enabled": request.enabled,
            "max_shift_distance": request.max_shift_distance,
            "shift_direction": request.shift_direction,
            "status": "updated",
        }

    except Exception as e:
        logger.error(f"Failed to set position shifting settings: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/settings/beat-brightness")
async def set_beat_brightness_settings(request: BeatBrightnessRequest):
    """Set beat brightness boost settings."""
    try:
        # Update in control state if available
        if control_state:
            control_state.update_status(
                beat_brightness_enabled=request.enabled,
                beat_brightness_intensity=request.intensity,
                beat_brightness_duration=request.duration,
            )
            logger.info(
                f"Updated beat brightness: enabled={request.enabled}, intensity={request.intensity}, duration={request.duration}"
            )
        else:
            logger.warning("Control state not available - beat brightness settings not updated")

        await manager.broadcast(
            {
                "type": "beat_brightness_changed",
                "enabled": request.enabled,
                "intensity": request.intensity,
                "duration": request.duration,
            }
        )

        return {
            "enabled": request.enabled,
            "intensity": request.intensity,
            "duration": request.duration,
            "status": "updated",
        }

    except Exception as e:
        logger.error(f"Failed to set beat brightness settings: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class OptimizationIterationsRequest(BaseModel):
    """Request model for optimization iterations."""

    iterations: int = Field(
        ..., ge=0, le=20, description="Number of optimization iterations (0-20, 0 = pseudo inverse only)"
    )


@app.post("/api/settings/optimization-iterations")
async def set_optimization_iterations(request: OptimizationIterationsRequest):
    """Set optimization iterations."""
    try:
        # Update in control state if available
        if control_state:
            control_state.update_status(optimization_iterations=request.iterations)
            logger.info(f"Updated optimization iterations to {request.iterations}")
        else:
            logger.warning("Control state not available - optimization iterations not updated")

        await manager.broadcast({"type": "optimization_iterations_changed", "iterations": request.iterations})

        return {"iterations": request.iterations, "status": "updated"}

    except Exception as e:
        logger.error(f"Failed to set optimization iterations: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# WebSocket endpoint for live updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    try:
        # Send initial state
        settings_dict = get_system_settings().dict()

        await websocket.send_json(
            {
                "type": "initial_state",
                "playlist": playlist_state.dict_serializable(),
                "settings": settings_dict,
                "timestamp": time.time(),
            }
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                message = await websocket.receive_json()
                # Handle client messages if needed
                logger.debug(f"Received WebSocket message: {message}")
            except Exception as e:
                logger.warning(f"WebSocket message error: {e}")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


# System control endpoints
@app.post("/api/system/restart")
async def restart_system():
    """Restart the Prismatron application via control state signal."""
    try:
        logger.warning("Application restart requested via API")

        await manager.broadcast({"type": "system_restart", "timestamp": time.time()})

        # Signal restart through control state - main process will handle graceful shutdown
        if control_state:
            control_state.signal_restart()
            return {
                "status": "restart_initiated",
                "message": "Application restarting. System will be back online in approximately 15 seconds.",
            }
        else:
            # Fallback to direct exit if control state unavailable
            asyncio.create_task(perform_restart())
            return {
                "status": "restart_initiated",
                "message": "Application restarting. System will be back online in approximately 15 seconds.",
            }

    except Exception as e:
        logger.error(f"Failed to restart system: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def perform_restart():
    """Fallback restart by exiting after delay to allow response to be sent."""
    await asyncio.sleep(2)  # Give time for HTTP response to be sent

    logger.info("Exiting application for restart (fallback)...")
    # Exit with code 1 to trigger systemd restart (Restart=on-failure)
    os._exit(1)


@app.post("/api/system/reboot")
async def reboot_system():
    """Reboot the entire device."""
    try:
        # Send reboot signal via control state
        if control_state:
            control_state.signal_reboot()

        await manager.broadcast({"type": "system_reboot", "timestamp": time.time()})

        return {"status": "reboot_initiated", "message": "Device reboot initiated"}

    except Exception as e:
        logger.error(f"Failed to reboot system: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Network Management Endpoints
@app.get("/api/network/status", response_model=NetworkStatusResponse)
async def get_network_status():
    """Get current network status."""
    try:
        if not network_manager:
            raise HTTPException(status_code=503, detail="Network manager not available")

        status = await network_manager.get_status()
        return NetworkStatusResponse(
            mode=status.mode.value,
            connected=status.connected,
            interface=status.interface,
            ip_address=status.ip_address,
            ssid=status.ssid,
            signal_strength=status.signal_strength,
            gateway=status.gateway,
            dns_servers=status.dns_servers,
        )
    except Exception as e:
        logger.error(f"Failed to get network status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/network/scan", response_model=List[WiFiNetworkResponse])
async def scan_wifi_networks():
    """Scan for available WiFi networks."""
    try:
        if not network_manager:
            raise HTTPException(status_code=503, detail="Network manager not available")

        networks = await network_manager.scan_wifi()
        return [
            WiFiNetworkResponse(
                ssid=network.ssid,
                bssid=network.bssid,
                signal_strength=network.signal_strength,
                frequency=network.frequency,
                security=network.security.value,
                connected=network.connected,
            )
            for network in networks
        ]
    except Exception as e:
        logger.error(f"Failed to scan WiFi networks: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/network/connect")
async def connect_wifi(request: WiFiConnectRequest):
    """Connect to a WiFi network."""
    try:
        if not network_manager:
            raise HTTPException(status_code=503, detail="Network manager not available")

        success = await network_manager.connect_wifi(request.ssid, request.password)
        if success:
            # Broadcast network status change
            await manager.broadcast({"type": "network_status_changed", "mode": "client", "ssid": request.ssid})
            return {"status": "connected", "ssid": request.ssid}
        else:
            raise HTTPException(status_code=500, detail="Failed to connect to WiFi")
    except Exception as e:
        logger.error(f"Failed to connect to WiFi {request.ssid}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/network/disconnect")
async def disconnect_wifi():
    """Disconnect from current network."""
    try:
        if not network_manager:
            raise HTTPException(status_code=503, detail="Network manager not available")

        success = await network_manager.disconnect()
        if success:
            # Broadcast network status change
            await manager.broadcast({"type": "network_status_changed", "mode": "disconnected"})
            return {"status": "disconnected"}
        else:
            raise HTTPException(status_code=500, detail="Failed to disconnect")
    except Exception as e:
        logger.error(f"Failed to disconnect: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/network/ap/enable")
async def enable_ap_mode():
    """Enable WiFi access point mode."""
    try:
        if not network_manager:
            raise HTTPException(status_code=503, detail="Network manager not available")

        success = await network_manager.enable_ap_mode()
        if success:
            # Broadcast network status change
            await manager.broadcast({"type": "network_status_changed", "mode": "ap", "ssid": "prismatron"})
            return {"status": "ap_enabled", "ssid": "prismatron"}
        else:
            raise HTTPException(status_code=500, detail="Failed to enable AP mode")
    except Exception as e:
        logger.error(f"Failed to enable AP mode: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/network/ap/disable")
async def disable_ap_mode():
    """Disable WiFi access point mode."""
    try:
        if not network_manager:
            raise HTTPException(status_code=503, detail="Network manager not available")

        success = await network_manager.disable_ap_mode()
        if success:
            # Broadcast network status change
            await manager.broadcast({"type": "network_status_changed", "mode": "disconnected"})
            return {"status": "ap_disabled"}
        else:
            raise HTTPException(status_code=500, detail="Failed to disable AP mode")
    except Exception as e:
        logger.error(f"Failed to disable AP mode: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Preview endpoint for LED display
@app.get("/api/preview")
async def get_led_preview():
    """Get current LED preview data for the home page display."""
    try:
        preview_data = {
            "timestamp": time.time(),
            "is_active": playlist_state.is_playing,
            "current_item": None,
            "has_frame": False,
            "frame_data": None,
        }

        # Use rendering_index from control state to show currently rendered item
        rendering_index = -1
        if control_state:
            system_status = control_state.get_status_dict()
            rendering_index = system_status.get("rendering_index", -1)

        if playlist_state.items and 0 <= rendering_index < len(playlist_state.items):
            current_item = playlist_state.items[rendering_index]
            preview_data["current_item"] = {"name": current_item.name, "type": current_item.type}

        # Try to get real LED data from shared memory (PreviewSink)
        try:
            import mmap
            import os

            # Try to read from shared memory created by PreviewSink
            try:
                shm_fd = os.open("/dev/shm/prismatron_preview", os.O_RDONLY)

                # Read full header (64 bytes): timestamp(8) + frame_counter(8) + led_count(4) + padding(44)
                header_data = os.read(shm_fd, 64)
                if len(header_data) == 64:
                    import struct

                    # Unpack header according to PreviewSink format: "<ddiid32x"
                    timestamp, frame_counter, led_count, shm_rendering_index, shm_playback_position = struct.unpack(
                        "<ddiid32x", header_data
                    )

                    if led_count > 0:
                        # Read LED data starting at offset 64: led_count * 3 bytes (RGB)
                        os.lseek(shm_fd, 64, os.SEEK_SET)  # Seek to LED data section
                        led_data = os.read(shm_fd, led_count * 3)

                        if len(led_data) == led_count * 3:
                            # Convert to list of [r, g, b] arrays with brightness factor for preview
                            frame_data = []
                            brightness_factor = 0.5  # Reduce saturation due to overlapping LEDs
                            for i in range(0, len(led_data), 3):
                                r, g, b = led_data[i], led_data[i + 1], led_data[i + 2]
                                # Apply brightness factor to reduce saturation
                                r = int(r * brightness_factor)
                                g = int(g * brightness_factor)
                                b = int(b * brightness_factor)
                                frame_data.append([r, g, b])

                            preview_data["has_frame"] = True
                            preview_data["frame_data"] = frame_data
                            preview_data["total_leds"] = led_count
                            preview_data["frame_counter"] = frame_counter
                            # Handle potential invalid rendering_index values
                            preview_data["shm_rendering_index"] = (
                                shm_rendering_index if shm_rendering_index < 999999 else -1
                            )
                            # Add playback position (handle invalid values)
                            preview_data["playback_position"] = (
                                shm_playback_position if shm_playback_position >= 0 else 0.0
                            )
                            # Removed spammy playback position log
                        else:
                            logger.debug(
                                f"Incomplete LED data in shared memory: {len(led_data)} bytes for {led_count} LEDs"
                            )
                    else:
                        logger.debug("No LED count in shared memory header")
                else:
                    logger.debug(f"Invalid header size in shared memory: {len(header_data)} bytes, expected 64")

                os.close(shm_fd)
            except FileNotFoundError:
                logger.debug("PreviewSink shared memory not found - consumer may not be running with preview sink")
            except Exception as e:
                logger.debug(f"Error reading from shared memory: {e}")

        except Exception as e:
            logger.warning(f"Failed to access preview shared memory: {e}")

        # Fallback to test pattern if no real data available
        if not preview_data["has_frame"]:
            logger.debug("Using fallback rainbow test pattern - no shared memory data available")
            preview_data["has_frame"] = True
            preview_colors = []
            brightness_factor = 0.5  # Reduce saturation due to overlapping LEDs
            # Use the actual LED count for test pattern
            test_led_count = get_actual_led_count()
            for i in range(test_led_count):
                # Simple rainbow pattern
                hue = (i / test_led_count) * 360  # Full rainbow across all LEDs
                r = int(255 * max(0, min(1, abs((hue / 60) % 6 - 3) - 1)) * brightness_factor)
                g = int(255 * max(0, min(1, 2 - abs((hue / 60) % 6 - 2))) * brightness_factor)
                b = int(255 * max(0, min(1, 2 - abs((hue / 60) % 6 - 4))) * brightness_factor)
                preview_colors.append([r, g, b])
            preview_data["frame_data"] = preview_colors
            preview_data["total_leds"] = test_led_count
            preview_data["shm_rendering_index"] = -1  # No shared memory data

        return preview_data

    except Exception as e:
        logger.error(f"Failed to get LED preview: {e}")
        return {
            "timestamp": time.time(),
            "is_active": False,
            "has_frame": False,
            "frame_data": None,
            "current_item": None,
        }


@app.get("/api/led-positions")
async def get_led_positions():
    """Get LED positions for preview rendering."""
    try:
        # Load LED positions from diffusion patterns file
        if diffusion_patterns_path is None:
            raise HTTPException(status_code=500, detail="No diffusion patterns file configured")

        patterns_path = Path(diffusion_patterns_path)

        if not patterns_path.exists():
            raise HTTPException(status_code=404, detail=f"LED positions data not found at {patterns_path}")

        # Load the diffusion patterns data
        data = np.load(patterns_path, allow_pickle=True)

        if "led_positions" not in data:
            raise HTTPException(status_code=404, detail="LED positions not found in patterns data")

        led_positions = data["led_positions"]  # Shape: (led_count, 2) - in physical order

        # LED positions are already in physical order, do not modify
        positions_list = led_positions.tolist()
        logger.info(f"Loaded LED positions in physical order: {len(positions_list)} LEDs")

        # Get frame dimensions for scaling/normalization
        frame_width = FRAME_WIDTH  # 800
        frame_height = FRAME_HEIGHT  # 480

        # Add debugging information
        y_coords = [pos[1] for pos in positions_list[:100]]  # First 100 Y coordinates
        debug_stats = {
            "min_y": min(y_coords),
            "max_y": max(y_coords),
            "sample_positions": positions_list[:5],  # First 5 positions for debugging
        }

        return {
            "led_count": len(positions_list),
            "positions": positions_list,  # Array of [x, y] coordinates
            "frame_dimensions": {"width": frame_width, "height": frame_height},
            "coordinate_system": {
                "origin": "top-left",
                "description": "Pixel coordinates where (0,0) is top-left corner",
            },
            "debug": debug_stats,  # Debug information to help with frontend troubleshooting
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get LED positions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load LED positions: {str(e)}") from e


# System fonts endpoint
@app.get("/api/system-fonts")
async def get_system_fonts():
    """Get list of available system fonts."""
    try:
        import matplotlib.font_manager as fm

        # Get all TTF font files
        font_files = fm.findSystemFonts(fontpaths=None, fontext="ttf")

        # Extract font names and organize them
        fonts = []
        seen_names = set()

        for font_path in font_files:
            try:
                # Get font properties
                font_prop = fm.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                font_family = font_prop.get_family()[0] if font_prop.get_family() else font_name
                font_style = font_prop.get_style()
                font_weight = font_prop.get_weight()

                # Use font name as the unique identifier since family is always 'sans-serif'
                font_key = font_name

                logger.debug(
                    f"Found font: {font_name}, family: {font_family}, style: {font_style}, weight: {font_weight}, path: {font_path}"
                )

                if font_key not in seen_names:
                    fonts.append(
                        {
                            "name": font_name,
                            "family": font_family,
                            "style": font_style,
                            "weight": font_weight,
                            "path": font_path,
                        }
                    )
                    seen_names.add(font_key)

            except Exception as e:
                # Skip fonts that can't be processed
                logger.warning(f"Skipping font {font_path}: {e}")
                continue

        # Sort fonts by name
        fonts.sort(key=lambda x: x["name"].lower())

        # Group by name for easier frontend usage (since family is always 'sans-serif')
        font_families = {}
        for font in fonts:
            name = font["name"]
            font_families[name] = [font]  # Each font name gets its own entry

        return {"fonts": fonts, "families": font_families, "count": len(fonts)}

    except ImportError:
        # Fallback to basic font list if matplotlib not available
        logger.warning("matplotlib not available for font detection, using fallback list")
        basic_fonts = [
            {"name": "Arial", "family": "Arial", "style": "normal", "weight": "normal", "path": "arial.ttf"},
            {
                "name": "Helvetica",
                "family": "Helvetica",
                "style": "normal",
                "weight": "normal",
                "path": "helvetica.ttf",
            },
            {"name": "Times New Roman", "family": "Times", "style": "normal", "weight": "normal", "path": "times.ttf"},
            {"name": "Courier New", "family": "Courier", "style": "normal", "weight": "normal", "path": "courier.ttf"},
            {
                "name": "DejaVu Sans",
                "family": "DejaVu Sans",
                "style": "normal",
                "weight": "normal",
                "path": "DejaVuSans.ttf",
            },
        ]
        return {
            "fonts": basic_fonts,
            "families": {font["family"]: [font] for font in basic_fonts},
            "count": len(basic_fonts),
        }

    except Exception as e:
        logger.error(f"Failed to get system fonts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system fonts: {str(e)}") from e


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "active_connections": len(manager.active_connections),
    }


def set_consumer_process(consumer):
    """Set the consumer process reference for accessing LED data."""
    global consumer_process
    consumer_process = consumer


def set_producer_process(producer):
    """Set the producer process reference for playlist management."""
    global producer_process
    producer_process = producer


def set_control_state(control):
    """Set the control state reference for process communication."""
    global control_state
    control_state = control


def create_app():
    """Create and configure the FastAPI application."""
    return app


# Serve static files (for production)
frontend_dir = Path(__file__).parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir / "static")), name="static")


# Playlist Save/Load endpoints
@app.get("/api/playlists")
async def list_saved_playlists():
    """List all saved playlists with metadata."""
    try:
        playlists = []
        if PLAYLISTS_DIR.exists():
            for file_path in PLAYLISTS_DIR.glob("*.json"):
                try:
                    with open(file_path) as f:
                        data = json.load(f)

                    # Calculate total duration (handle None values)
                    total_duration = sum(item.get("duration") or 0 for item in data.get("items", []))

                    playlists.append(
                        {
                            "filename": file_path.name,
                            "name": data.get("name", file_path.stem),
                            "description": data.get("description", ""),
                            "item_count": len(data.get("items", [])),
                            "total_duration": total_duration,
                            "created_at": data.get("created_at"),
                            "modified_at": data.get("modified_at"),
                            "auto_repeat": data.get("auto_repeat", True),
                            "shuffle": data.get("shuffle", False),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to read playlist file {file_path}: {e}")
                    continue

        # Sort by modification time (newest first)
        playlists.sort(key=lambda x: x.get("modified_at", ""), reverse=True)
        return {"playlists": playlists}

    except Exception as e:
        logger.error(f"Failed to list playlists: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/playlists/{filename}")
async def load_playlist(filename: str):
    """Load a saved playlist and replace current playlist."""
    global current_playlist_file

    try:
        # Validate filename
        if not filename.endswith(".json"):
            filename += ".json"

        file_path = PLAYLISTS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Playlist not found")

        # Load playlist data
        with open(file_path) as f:
            data = json.load(f)

        # Clear current playlist via sync service
        if playlist_sync_client and playlist_sync_client.connected:
            playlist_sync_client.clear_playlist()

        # Add items to playlist
        items_added = 0
        for item_data in data.get("items", []):
            try:
                # Handle file paths - convert relative to absolute
                if item_data.get("file_path"):
                    # For media files, ensure they're absolute paths
                    if item_data["type"] in ["image", "video"]:
                        # If it's not already an absolute path
                        if not item_data["file_path"].startswith("/"):
                            file_path = item_data["file_path"]
                            # If path already includes uploads/ or media/, use as-is
                            if file_path.startswith(("uploads/", "media/")):
                                item_data["file_path"] = str(Path(file_path))
                            else:
                                # If it's just a filename, default to uploads directory
                                item_data["file_path"] = str(UPLOAD_DIR / file_path)
                    # For text type, file_path contains JSON config string
                    elif item_data["type"] == "text":
                        # Extract duration from text config if available
                        try:
                            text_config = json.loads(item_data["file_path"])
                            # Override duration with the one from text config
                            if "duration" in text_config:
                                item_data["duration"] = text_config["duration"]
                        except (json.JSONDecodeError, KeyError):
                            pass  # Keep original duration if parsing fails

                # Create PlaylistItem with transitions
                item = PlaylistItem(
                    id=item_data.get("id", str(uuid.uuid4())),
                    name=item_data["name"],
                    type=item_data["type"],
                    file_path=item_data.get("file_path"),
                    effect_config=item_data.get("effect_config"),
                    duration=item_data.get("duration"),
                    order=item_data.get("order", items_added),
                    transition_in=TransitionConfig(
                        **item_data.get("transition_in", {"type": "none", "parameters": {}})
                    ),
                    transition_out=TransitionConfig(
                        **item_data.get("transition_out", {"type": "none", "parameters": {}})
                    ),
                )

                # Send to playlist sync service
                if playlist_sync_client and playlist_sync_client.connected:
                    sync_item = api_item_to_sync_item(item)
                    if playlist_sync_client.add_item(sync_item):
                        items_added += 1
                    else:
                        logger.warning(f"Failed to add item {item.name} to playlist")
                else:
                    # Fallback to local playlist if sync service not available
                    playlist_state.items.append(item)
                    items_added += 1

            except Exception as e:
                logger.warning(f"Failed to add playlist item: {e}")
                continue

        # Update playlist settings
        playlist_state.auto_repeat = data.get("auto_repeat", True)
        playlist_state.shuffle = data.get("shuffle", False)

        # Track the currently loaded playlist file
        current_playlist_file = filename

        logger.info(f"Loaded playlist {filename} with {items_added} items")

        return {
            "status": "loaded",
            "filename": filename,
            "name": data.get("name", filename),
            "items_loaded": items_added,
            "total_items": len(data.get("items", [])),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load playlist: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class SavePlaylistRequest(BaseModel):
    """Request model for saving a playlist."""

    name: str = Field(..., description="Playlist name")
    description: Optional[str] = Field(None, description="Playlist description")
    overwrite: bool = Field(False, description="Whether to overwrite existing file")


@app.post("/api/playlists/save")
async def save_playlist(request: SavePlaylistRequest):
    """Save current playlist to a new file."""
    try:
        # Sanitize filename from name
        import re

        safe_name = re.sub(r"[^\w\s-]", "", request.name)
        safe_name = re.sub(r"[-\s]+", "_", safe_name)
        filename = f"{safe_name}.json"

        file_path = PLAYLISTS_DIR / filename

        # Check if file exists and overwrite is not allowed
        if file_path.exists() and not request.overwrite:
            raise HTTPException(status_code=409, detail=f"Playlist '{filename}' already exists")

        # Prepare playlist data
        now = datetime.now().isoformat()
        playlist_data = {
            "version": "1.0",
            "name": request.name,
            "description": request.description or "",
            "created_at": now if not file_path.exists() else None,  # Keep original if updating
            "modified_at": now,
            "auto_repeat": playlist_state.auto_repeat,
            "shuffle": playlist_state.shuffle,
            "items": [],
        }

        # If updating, preserve created_at
        if file_path.exists():
            try:
                with open(file_path) as f:
                    existing_data = json.load(f)
                    playlist_data["created_at"] = existing_data.get("created_at", now)
            except Exception:
                playlist_data["created_at"] = now
        else:
            playlist_data["created_at"] = now

        # Convert items to save format
        for item in playlist_state.items:
            # For text items, extract the actual duration from the config
            actual_duration = item.duration
            if item.type == "text" and item.file_path:
                try:
                    text_config = json.loads(item.file_path)
                    # Use the duration from the text config if available
                    actual_duration = text_config.get("duration", item.duration)
                except (json.JSONDecodeError, AttributeError):
                    pass  # Keep original duration if parsing fails

            item_data = {
                "id": item.id,
                "name": item.name,
                "type": item.type,
                "duration": actual_duration,
                "order": item.order,
                "transition_in": {"type": item.transition_in.type, "parameters": item.transition_in.parameters},
                "transition_out": {"type": item.transition_out.type, "parameters": item.transition_out.parameters},
            }

            # Handle file paths - convert to relative for portability
            if item.file_path:
                if item.type in ["image", "video"]:
                    # Convert absolute path to relative (just filename)
                    file_path = Path(item.file_path)
                    if file_path.is_absolute():
                        item_data["file_path"] = file_path.name
                    else:
                        item_data["file_path"] = item.file_path
                else:
                    # For text type, file_path contains JSON config
                    item_data["file_path"] = item.file_path

            # Add effect config if present
            if item.effect_config:
                item_data["effect_config"] = item.effect_config

            playlist_data["items"].append(item_data)

        # Save to file
        with open(file_path, "w") as f:
            json.dump(playlist_data, f, indent=2)

        logger.info(f"Saved playlist to {filename}")

        return {
            "status": "saved",
            "filename": filename,
            "name": request.name,
            "item_count": len(playlist_data["items"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save playlist: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/api/playlists/{filename}")
async def delete_playlist(filename: str):
    """Delete a saved playlist file."""
    try:
        # Validate filename
        if not filename.endswith(".json"):
            filename += ".json"

        file_path = PLAYLISTS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Playlist not found")

        # Delete the file
        file_path.unlink()

        logger.info(f"Deleted playlist {filename}")

        return {"status": "deleted", "filename": filename}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete playlist: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Serve service worker files from root
@app.get("/registerSW.js")
async def serve_register_sw():
    """Serve the service worker registration file."""
    frontend_dir = Path(__file__).parent / "frontend" / "dist"
    sw_file = frontend_dir / "registerSW.js"
    if sw_file.exists():
        return FileResponse(str(sw_file), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service worker not found")


@app.get("/sw.js")
async def serve_sw():
    """Serve the service worker file."""
    frontend_dir = Path(__file__).parent / "frontend" / "dist"
    sw_file = frontend_dir / "sw.js"
    if sw_file.exists():
        return FileResponse(str(sw_file), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service worker not found")


# Catch-all route for SPA routing - must be after all API routes
@app.get("/{path:path}")
async def catch_all(path: str):
    """Catch-all route to serve React app for SPA routing."""
    # Don't interfere with API routes
    if path.startswith(("api/", "ws")) or path in ("docs", "redoc"):
        raise HTTPException(status_code=404, detail="Not found")

    # Serve the React app for all other routes
    frontend_dir = Path(__file__).parent / "frontend" / "dist"
    index_file = frontend_dir / "index.html"

    if index_file.exists():
        response = FileResponse(str(index_file))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")


# Duplicate endpoint removed - using the earlier restart_system() implementation


# Duplicate reboot endpoint removed - using the earlier reboot_system() implementation


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
    patterns_path: Optional[str] = None,
    led_count: Optional[int] = None,
    config: Optional[Dict] = None,
):
    """Run the API server."""
    global diffusion_patterns_path, system_settings
    diffusion_patterns_path = patterns_path

    # Initialize video conversion manager if available and config provided
    if VIDEO_CONVERSION_AVAILABLE and config:
        try:
            initialize_conversion_manager(config)

            # Set up WebSocket broadcasting for conversion updates
            conversion_manager = get_conversion_manager()
            if conversion_manager:

                def broadcast_conversion_update(job):
                    try:
                        # Create an async task to broadcast the update
                        import asyncio

                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(
                                manager.broadcast(
                                    {"type": "conversion_update", "conversion": job.to_dict(), "timestamp": time.time()}
                                )
                            )
                        except RuntimeError:
                            # No running loop, ignore for now
                            pass
                    except Exception as e:
                        logger.warning(f"Failed to broadcast conversion update: {e}")

                conversion_manager.add_status_callback(broadcast_conversion_update)
                logger.info("Conversion WebSocket broadcasting enabled")
        except Exception as e:
            logger.error(f"Failed to initialize video conversion manager: {e}")

    # Initialize SystemSettings with LED count from startup
    if led_count is not None:
        system_settings = SystemSettings(led_count=led_count)
        logger.info(f"Initialized SystemSettings with LED count: {led_count}")
    else:
        logger.warning(
            "No LED count provided to API server - SystemSettings will not be available until consumer starts"
        )

    # Create and connect to control state (shared memory created by main process)
    api_control_state = ControlState()
    if api_control_state.connect():
        set_control_state(api_control_state)
        logger.info("API server connected to control state")
    else:
        logger.warning("API server failed to connect to control state - FPS metrics will be unavailable")

    # For a multi-process LED system, disable auto-reload to avoid file watching noise
    uvicorn.run(
        "src.web.api_server:app",
        host=host,
        port=port,
        reload=False,  # Disabled to prevent watchfiles logging noise
        log_level="info",  # Keep at info level to avoid verbose WebSocket debugging
        access_log=False,  # Disable access logging to reduce noise
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prismatron Web API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Setup logging
    from src.utils.logging_utils import create_app_time_formatter

    formatter = create_app_time_formatter()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        handlers=[handler],
    )

    logger.info(f"Starting Prismatron API server on {args.host}:{args.port}")
    run_server(args.host, args.port, args.debug)
