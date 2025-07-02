"""
Prismatron Web API Server.

FastAPI-based backend for the Prismatron web interface with retro-futurism design.
Provides endpoints for home, upload, effects, playlist management, and settings.
"""

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

from const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT
from core.control_state import ControlState

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
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


class PlaylistState(BaseModel):
    """Current playlist state."""

    items: List[PlaylistItem] = Field(default_factory=list)
    current_index: int = Field(0, description="Currently playing item index")
    is_playing: bool = Field(False, description="Whether playback is active")
    auto_repeat: bool = Field(True, description="Auto-repeat playlist")
    shuffle: bool = Field(False, description="Shuffle mode")


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
    brightness: float = Field(1.0, description="Current brightness")
    frame_rate: float = Field(0.0, description="Current frame rate")
    uptime: float = Field(0.0, description="System uptime in seconds")
    memory_usage: float = Field(0.0, description="Memory usage percentage")
    cpu_usage: float = Field(0.0, description="CPU usage percentage")


# Global state
playlist_state = PlaylistState()
system_settings = SystemSettings()
control_state: Optional[ControlState] = None

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

# API Routes


@app.get("/")
async def root():
    """Serve the main React application."""
    frontend_dir = Path(__file__).parent / "frontend" / "dist"
    index_file = frontend_dir / "index.html"

    if index_file.exists():
        return FileResponse(str(index_file))
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
    # TODO: Integrate with actual system monitoring
    return SystemStatus(
        is_online=True,
        current_file=(
            playlist_state.items[playlist_state.current_index].file_path
            if playlist_state.items and 0 <= playlist_state.current_index < len(playlist_state.items)
            else None
        ),
        playlist_position=playlist_state.current_index,
        brightness=system_settings.brightness,
        frame_rate=30.0,  # TODO: Get actual frame rate
        uptime=time.time(),  # TODO: Get actual uptime
        memory_usage=45.2,  # TODO: Get actual memory usage
        cpu_usage=23.1,  # TODO: Get actual CPU usage
    )


@app.post("/api/control/play")
async def play_content():
    """Start playback."""
    playlist_state.is_playing = True
    await manager.broadcast(
        {
            "type": "playback_state",
            "is_playing": True,
            "current_index": playlist_state.current_index,
        }
    )
    return {"status": "playing"}


@app.post("/api/control/pause")
async def pause_content():
    """Pause playback."""
    playlist_state.is_playing = False
    await manager.broadcast(
        {
            "type": "playback_state",
            "is_playing": False,
            "current_index": playlist_state.current_index,
        }
    )
    return {"status": "paused"}


@app.post("/api/control/next")
async def next_item():
    """Skip to next playlist item."""
    if playlist_state.items:
        playlist_state.current_index = (playlist_state.current_index + 1) % len(playlist_state.items)
        await manager.broadcast({"type": "playlist_position", "current_index": playlist_state.current_index})
    return {"current_index": playlist_state.current_index}


@app.post("/api/control/previous")
async def previous_item():
    """Skip to previous playlist item."""
    if playlist_state.items:
        playlist_state.current_index = (playlist_state.current_index - 1) % len(playlist_state.items)
        await manager.broadcast({"type": "playlist_position", "current_index": playlist_state.current_index})
    return {"current_index": playlist_state.current_index}


# Upload endpoints
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
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

        playlist_state.items.append(item)

        # Broadcast playlist update
        await manager.broadcast(
            {
                "type": "playlist_updated",
                "items": [item.dict() for item in playlist_state.items],
            }
        )

        return {"status": "uploaded", "item": item.dict()}

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Effects endpoints
@app.get("/api/effects", response_model=List[EffectPreset])
async def get_effects():
    """Get available effect presets."""
    return EFFECT_PRESETS


@app.post("/api/effects/{effect_id}/add")
async def add_effect_to_playlist(
    effect_id: str,
    name: Optional[str] = None,
    duration: Optional[float] = None,
    config: Optional[Dict] = None,
):
    """Add an effect to the playlist."""
    # Find effect preset
    effect_preset = next((e for e in EFFECT_PRESETS if e.id == effect_id), None)
    if not effect_preset:
        raise HTTPException(status_code=404, detail="Effect not found")

    # Merge custom config with preset
    final_config = effect_preset.config.copy()
    if config:
        final_config.update(config)

    # Create playlist item
    item = PlaylistItem(
        id=str(uuid.uuid4()),
        name=name or effect_preset.name,
        type="effect",
        effect_config={"effect_id": effect_id, "parameters": final_config},
        duration=duration or 30.0,  # Default 30 seconds for effects
        order=len(playlist_state.items),
    )

    playlist_state.items.append(item)

    # Broadcast playlist update
    await manager.broadcast(
        {
            "type": "playlist_updated",
            "items": [item.dict() for item in playlist_state.items],
        }
    )

    return {"status": "added", "item": item.dict()}


# Playlist endpoints
@app.get("/api/playlist", response_model=PlaylistState)
async def get_playlist():
    """Get current playlist state."""
    return playlist_state


@app.post("/api/playlist/reorder")
async def reorder_playlist(item_ids: List[str]):
    """Reorder playlist items."""
    # Create mapping of id to item
    item_map = {item.id: item for item in playlist_state.items}

    # Reorder items
    reordered_items = []
    for i, item_id in enumerate(item_ids):
        if item_id in item_map:
            item = item_map[item_id]
            item.order = i
            reordered_items.append(item)

    playlist_state.items = reordered_items

    # Broadcast update
    await manager.broadcast(
        {
            "type": "playlist_updated",
            "items": [item.dict() for item in playlist_state.items],
        }
    )

    return {"status": "reordered"}


@app.delete("/api/playlist/{item_id}")
async def remove_playlist_item(item_id: str):
    """Remove item from playlist."""
    original_length = len(playlist_state.items)
    playlist_state.items = [item for item in playlist_state.items if item.id != item_id]

    if len(playlist_state.items) < original_length:
        # Adjust current index if needed
        if playlist_state.current_index >= len(playlist_state.items):
            playlist_state.current_index = max(0, len(playlist_state.items) - 1)

        # Broadcast update
        await manager.broadcast(
            {
                "type": "playlist_updated",
                "items": [item.dict() for item in playlist_state.items],
            }
        )

        return {"status": "removed"}
    else:
        raise HTTPException(status_code=404, detail="Item not found")


@app.post("/api/playlist/clear")
async def clear_playlist():
    """Clear all playlist items."""
    playlist_state.items.clear()
    playlist_state.current_index = 0
    playlist_state.is_playing = False

    await manager.broadcast({"type": "playlist_updated", "items": []})

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
                "playlist": playlist_state.dict(),
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


# Serve static files (for production)
@app.mount("/static", StaticFiles(directory="static"), name="static")
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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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


def create_app():
    """Create and configure the FastAPI application."""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run the API server."""
    uvicorn.run(
        "src.web.api_server:app",
        host=host,
        port=port,
        reload=debug,
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
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting Prismatron API server on {args.host}:{args.port}")
    run_server(args.host, args.port, args.debug)
