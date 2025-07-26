"""
Prismatron Web API Server.

FastAPI-based backend for the Prismatron web interface with retro-futurism design.
Provides endpoints for home, upload, effects, playlist management, and settings.
"""

import asyncio
import json
import logging
import os
import shutil

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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT, LED_DATA_SIZE
from src.core.control_state import ControlState
from src.core.playlist_sync import PlaylistItem as SyncPlaylistItem
from src.core.playlist_sync import PlaylistState as SyncPlaylistState
from src.core.playlist_sync import TransitionConfig as SyncTransitionConfig
from src.core.playlist_sync import (
    PlaylistSyncClient,
)

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


# Pydantic models for API requests/responses
class TransitionConfig(BaseModel):
    """Transition configuration model."""
    
    type: str = Field("none", description="Transition type: none, fade")
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
    transition_out: TransitionConfig = Field(default_factory=TransitionConfig, description="Transition out configuration")

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
    led_count: int = Field(LED_COUNT, description="Number of LEDs")
    display_resolution: Dict[str, int] = Field(default_factory=lambda: {"width": FRAME_WIDTH, "height": FRAME_HEIGHT})
    auto_start_playlist: bool = Field(True, description="Auto-start playlist on boot")
    preview_enabled: bool = Field(True, description="Enable live preview")


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
    brightness: float = Field(1.0, description="Current brightness")
    frame_rate: float = Field(0.0, description="Current frame rate")
    uptime: float = Field(0.0, description="System uptime in seconds")
    memory_usage: float = Field(0.0, description="Memory usage percentage")
    memory_usage_gb: float = Field(0.0, description="Memory usage in GB")
    cpu_usage: float = Field(0.0, description="CPU usage percentage")
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


# Global state
playlist_state = PlaylistState()
system_settings = SystemSettings()
control_state: Optional[ControlState] = None
consumer_process: Optional[object] = None  # Will be set by main process
producer_process: Optional[object] = None  # Will be set by main process
diffusion_patterns_path: Optional[str] = None  # Will be set by main process

# Playlist synchronization client
playlist_sync_client: Optional[PlaylistSyncClient] = None

# File storage paths
UPLOAD_DIR = Path("uploads")
THUMBNAILS_DIR = Path("thumbnails")
PLAYLISTS_DIR = Path("playlists")

# Ensure directories exist
for dir_path in [UPLOAD_DIR, THUMBNAILS_DIR, PLAYLISTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Built-in effect presets
EFFECT_PRESETS = [
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
        id="sparkle",
        name="Sparkle",
        description="Random twinkling sparkles across the display",
        config={"density": 0.1, "color": "#ffffff", "fade_speed": 0.95},
        category="particle",
        icon="✨",
    ),
    EffectPreset(
        id="plasma",
        name="Plasma Field",
        description="Smooth plasma-like color gradients",
        config={"speed": 1.5, "scale": 50, "complexity": 3},
        category="generated",
        icon="🔮",
    ),
    EffectPreset(
        id="fire",
        name="Fire Effect",
        description="Realistic fire simulation",
        config={"intensity": 0.8, "height": 0.6, "cooling": 55},
        category="simulation",
        icon="🔥",
    ),
    EffectPreset(
        id="matrix_rain",
        name="Matrix Rain",
        description="Digital rain effect like in The Matrix",
        config={"speed": 3.0, "density": 0.3, "color": "#00ff41"},
        category="retro",
        icon="💚",
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
            "fps": 30.0,  # Note: This will be overridden by DEFAULT_CONTENT_FPS in text_source.py
        },
        category="text",
        icon="📝",
    ),
]

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


def sync_item_to_api_item(sync_item: SyncPlaylistItem) -> PlaylistItem:
    """Convert sync playlist item to API playlist item."""
    # Convert transition configurations
    transition_in = TransitionConfig(
        type=sync_item.transition_in.type if sync_item.transition_in else "none",
        parameters=sync_item.transition_in.parameters if sync_item.transition_in else {}
    )
    transition_out = TransitionConfig(
        type=sync_item.transition_out.type if sync_item.transition_out else "none",
        parameters=sync_item.transition_out.parameters if sync_item.transition_out else {}
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
    transition_in = SyncTransitionConfig(
        type=api_item.transition_in.type,
        parameters=api_item.transition_in.parameters
    )
    transition_out = SyncTransitionConfig(
        type=api_item.transition_out.type,
        parameters=api_item.transition_out.parameters
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
    
    # Validate transition type
    valid_types = ["none", "fade"]
    if transition_config.type not in valid_types:
        errors["type"] = f"Invalid transition type '{transition_config.type}'. Must be one of: {', '.join(valid_types)}"
    
    # Validate parameters based on type
    if transition_config.type == "fade":
        duration = transition_config.parameters.get("duration")
        if duration is None:
            errors["parameters.duration"] = "Duration parameter is required for fade transitions"
        elif not isinstance(duration, (int, float)):
            errors["parameters.duration"] = "Duration must be a number"
        elif duration < 0.1 or duration > 10.0:
            errors["parameters.duration"] = "Duration must be between 0.1 and 10.0 seconds"
    
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

        # Broadcast update to connected WebSocket clients
        await manager.broadcast(
            {
                "type": "playlist_updated",
                "items": [item.dict_serializable() for item in playlist_state.items],
                "current_index": playlist_state.current_index,
                "is_playing": playlist_state.is_playing,
                "auto_repeat": playlist_state.auto_repeat,
                "shuffle": playlist_state.shuffle,
            }
        )

    except Exception as e:
        logger.error(f"Error in async playlist sync update: {e}")


def sync_playlist_update_handler(sync_state: SyncPlaylistState) -> None:
    """Sync wrapper for playlist update handler."""
    try:
        # Create a task for the async function
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If there's already a running loop, schedule as a task
                loop.create_task(on_playlist_sync_update(sync_state))
            else:
                # If no running loop, run the async function
                loop.run_until_complete(on_playlist_sync_update(sync_state))
        except RuntimeError:
            # No event loop in current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(on_playlist_sync_update(sync_state))
            finally:
                loop.close()
    except Exception as e:
        logger.error(f"Error in playlist sync update handler: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    global preview_task, playlist_sync_client

    # Start playlist synchronization client
    playlist_sync_client = PlaylistSyncClient(client_name="web_interface")
    playlist_sync_client.on_playlist_update = sync_playlist_update_handler
    if playlist_sync_client.connect():
        logger.info("Connected to playlist synchronization service")
    else:
        logger.warning("Failed to connect to playlist synchronization service")

    preview_task = asyncio.create_task(preview_broadcast_task())
    logger.info("Started preview data broadcasting task")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background tasks on application shutdown."""
    import contextlib

    global preview_task, playlist_sync_client

    # Disconnect playlist sync client
    if playlist_sync_client:
        playlist_sync_client.disconnect()
        logger.info("Disconnected from playlist synchronization service")

    if preview_task:
        preview_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await preview_task
        logger.info("Stopped preview data broadcasting task")


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

                        # Debug log every 10 seconds to verify values
                        if hasattr(get_system_status, "_last_debug_log"):
                            if current_time - get_system_status._last_debug_log > 10.0:
                                logger.info(
                                    f"API DEBUG (ControlState): input_fps={consumer_input_fps:.1f}, output_fps={renderer_output_fps:.1f}, dropped={dropped_frames_percentage:.1f}%"
                                )
                                get_system_status._last_debug_log = current_time
                        else:
                            get_system_status._last_debug_log = current_time
                    else:
                        logger.warning("API DEBUG (ControlState): no control state available")

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

                # Get temperature readings
                cpu_temp = get_cpu_temperature()
                gpu_temp = get_gpu_temperature()

                # Get rendering_index from control state 
                rendering_index_for_status = -1
                if control_state:
                    system_status_for_index = control_state.get_status_dict()
                    rendering_index_for_status = system_status_for_index.get("rendering_index", -1)

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
                    "brightness": system_settings.brightness,
                    "frame_rate": 30.0,  # Will be updated from shared memory below
                    "uptime": uptime,
                    "memory_usage": mem_percent,
                    "memory_usage_gb": mem_used_gb,
                    "cpu_usage": cpu_percent,
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

                            # Unpack header according to PreviewSink format: "<ddii40x"
                            timestamp, frame_counter, led_count, shm_rendering_index = struct.unpack("<ddii40x", header_data)
                            logger.debug(f"API SHM READ: timestamp={timestamp:.3f}, frame={frame_counter}, leds={led_count}, rendering_index={shm_rendering_index}")

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
                                    preview_data["shm_rendering_index"] = shm_rendering_index if shm_rendering_index < 999999 else -1
                                    
                                    # Debug: Compare rendering_index from different sources
                                    if hasattr(preview_broadcast_task, "_last_debug_time"):
                                        if current_time - preview_broadcast_task._last_debug_time > 2.0:
                                            logger.info(f"RENDERING INDEX DEBUG: ControlState={rendering_index}, SharedMemory={shm_rendering_index}, Frame={frame_counter}")
                                            logger.info(f"SHM TIMESTAMP DEBUG: shm_timestamp={timestamp:.3f}, current_time={current_time:.3f}, age={(current_time-timestamp):.3f}s")
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

                            logger.debug(f"Read stats data: first 50 bytes = {stats_data[:50]}")

                            # Find null terminator
                            null_pos = stats_data.find(b"\x00")
                            if null_pos > 0:
                                stats_json = stats_data[:null_pos].decode("utf-8")
                                stats = json.loads(stats_json)

                                logger.debug(f"Parsed stats: renderer_fps={stats.get('renderer_fps', 'missing')}")

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
                    test_led_count = LED_COUNT  # 2624
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

    # Get temperature readings
    cpu_temp = get_cpu_temperature()
    gpu_temp = get_gpu_temperature()

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

    # Get rendering_index from control state to show currently rendered item
    rendering_index = -1
    if control_state:
        system_status_dict = control_state.get_status_dict()
        rendering_index = system_status_dict.get("rendering_index", -1)

    return SystemStatus(
        is_online=True,
        current_file=(
            playlist_state.items[rendering_index].file_path
            if playlist_state.items and 0 <= rendering_index < len(playlist_state.items)
            else None
        ),
        playlist_position=playlist_state.current_index,
        rendering_index=rendering_index,
        brightness=system_settings.brightness,
        frame_rate=frame_rate,
        uptime=uptime,
        memory_usage=mem_percent,
        memory_usage_gb=mem_used_gb,
        cpu_usage=cpu_percent,
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
    """Start playback."""
    try:
        logger.info("API PLAY REQUEST: Received play command from web interface")

        # Send play command to playlist sync service
        if playlist_sync_client and playlist_sync_client.connected:
            success = playlist_sync_client.play()
            if success:
                return {"status": "playing"}
            else:
                logger.error("Failed to send play command via sync service")
                return {"status": "error", "message": "Playlist sync service unavailable"}
        else:
            logger.error("Playlist sync service not connected")
            return {"status": "error", "message": "Playlist sync service not connected"}

    except Exception as e:
        logger.error(f"Failed to start playback: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/control/pause")
async def pause_content():
    """Pause playback."""
    try:
        # Send pause command to playlist sync service
        if playlist_sync_client and playlist_sync_client.connected:
            success = playlist_sync_client.pause()
            if success:
                return {"status": "paused"}
            else:
                logger.error("Failed to send pause command via sync service")
                return {"status": "error", "message": "Playlist sync service unavailable"}
        else:
            logger.error("Playlist sync service not connected")
            return {"status": "error", "message": "Playlist sync service not connected"}

    except Exception as e:
        logger.error(f"Failed to pause playback: {e}")
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

        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}.{file_ext}"

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Create playlist item
        item = PlaylistItem(
            id=file_id,
            name=name or file.filename or f"Uploaded {content_type}",
            type=content_type,
            file_path=str(file_path),
            duration=duration,
            order=len(playlist_state.items),
        )

        # Validate the item (including default transitions)
        validation_errors = validate_playlist_item(item)
        if validation_errors:
            raise HTTPException(status_code=400, detail={"errors": validation_errors})

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
        return {"status": "uploaded", "item": item.dict_serializable()}

    except Exception as e:
        logger.error(f"Upload failed: {e}")
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
        import json

        text_config = json.dumps(final_config)

        # Create playlist item for text content
        final_name = request.name or f"Text: {final_config.get('text', 'Hello World')}"
        logger.info(
            f"Creating text effect with name: '{final_name}' (request.name='{request.name}', text='{final_config.get('text')}')"
        )

        item = PlaylistItem(
            id=str(uuid.uuid4()),
            name=final_name,
            type="text",
            file_path=text_config,  # JSON config as file path for text content
            duration=request.duration or final_config.get("duration", 10.0),
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
    # Send clear command to playlist sync service
    if playlist_sync_client and playlist_sync_client.connected:
        success = playlist_sync_client.clear_playlist()
        if success:
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
    return {
        "types": [
            {
                "type": "none",
                "name": "None",
                "description": "No transition",
                "parameters": {}
            },
            {
                "type": "fade",
                "name": "Fade",
                "description": "Fade in/out transition",
                "parameters": {
                    "duration": {
                        "type": "number",
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "description": "Fade duration in seconds"
                    }
                }
            }
        ]
    }


@app.put("/api/playlist/{item_id}/transitions")
async def update_item_transitions(
    item_id: str,
    transition_in: TransitionConfig,
    transition_out: TransitionConfig
):
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
        
        # Find and update the item
        if playlist_sync_client and playlist_sync_client.connected:
            # Get current state from sync service
            sync_state = playlist_sync_client.get_state()
            if sync_state:
                # Find the item to update
                item_found = False
                for sync_item in sync_state.items:
                    if sync_item.id == item_id:
                        # Update transitions
                        sync_item.transition_in = SyncTransitionConfig(
                            type=transition_in.type,
                            parameters=transition_in.parameters
                        )
                        sync_item.transition_out = SyncTransitionConfig(
                            type=transition_out.type,
                            parameters=transition_out.parameters
                        )
                        item_found = True
                        break
                
                if not item_found:
                    raise HTTPException(status_code=404, detail="Playlist item not found")
                
                # Update the sync service
                success = playlist_sync_client.update_item(sync_item)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to update item in sync service")
                
                return {
                    "status": "updated",
                    "item_id": item_id,
                    "transition_in": transition_in.dict_serializable(),
                    "transition_out": transition_out.dict_serializable()
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to get playlist state from sync service")
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
                "transition_out": transition_out.dict_serializable()
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
    return system_settings


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


# WebSocket endpoint for live updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    try:
        # Send initial state
        await websocket.send_json(
            {
                "type": "initial_state",
                "playlist": playlist_state.dict_serializable(),
                "settings": system_settings.dict(),
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
    """Restart the Prismatron system processes."""
    try:
        # Send restart signal via control state
        if control_state:
            control_state.signal_restart()

        await manager.broadcast({"type": "system_restart", "timestamp": time.time()})

        return {"status": "restart_initiated", "message": "System restart initiated"}

    except Exception as e:
        logger.error(f"Failed to restart system: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


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

                    # Unpack header according to PreviewSink format: "<ddii40x"
                    timestamp, frame_counter, led_count, shm_rendering_index = struct.unpack("<ddii40x", header_data)
                    logger.debug(f"API PREVIEW READ: timestamp={timestamp:.3f}, frame={frame_counter}, leds={led_count}, rendering_index={shm_rendering_index}")

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
                            preview_data["shm_rendering_index"] = shm_rendering_index if shm_rendering_index < 999999 else -1
                            logger.debug(
                                f"Using real LED data from shared memory: {len(frame_data)} LEDs, frame {frame_counter}, shm_rendering_index={shm_rendering_index}"
                            )
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
            test_led_count = LED_COUNT  # 2624
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
            logger.debug(f"Using fallback test pattern with {test_led_count} LEDs")

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


def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False, patterns_path: Optional[str] = None):
    """Run the API server."""
    global diffusion_patterns_path
    diffusion_patterns_path = patterns_path

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
        log_level="debug" if debug else "info",
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
