#!/usr/bin/env python3
"""
Test script for the Prismatron web interface.
Runs a simplified FastAPI server for testing the frontend.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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

# Simple mock constants
FRAME_WIDTH = 800
FRAME_HEIGHT = 480
LED_COUNT = 3200

logger = logging.getLogger(__name__)


# Pydantic models (simplified versions)
class PlaylistItem(BaseModel):
    id: str
    name: str
    type: str
    file_path: Optional[str] = None
    effect_config: Optional[Dict] = None
    duration: Optional[float] = None
    thumbnail: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    order: int = 0


class PlaylistState(BaseModel):
    items: List[PlaylistItem] = Field(default_factory=list)
    current_index: int = 0
    is_playing: bool = False
    auto_repeat: bool = True
    shuffle: bool = False


class SystemSettings(BaseModel):
    brightness: float = 1.0
    frame_rate: float = 30.0
    led_count: int = LED_COUNT
    display_resolution: Dict[str, int] = Field(
        default_factory=lambda: {"width": FRAME_WIDTH, "height": FRAME_HEIGHT}
    )
    auto_start_playlist: bool = True
    preview_enabled: bool = True


class EffectPreset(BaseModel):
    id: str
    name: str
    description: str = ""
    config: Dict = Field(default_factory=dict)
    category: str = "general"
    icon: str = "✨"


class SystemStatus(BaseModel):
    is_online: bool = True
    current_file: Optional[str] = None
    playlist_position: int = 0
    brightness: float = 1.0
    frame_rate: float = 30.0
    uptime: float = 0.0
    memory_usage: float = 45.2
    cpu_usage: float = 23.1


# Global state
playlist_state = PlaylistState()
system_settings = SystemSettings()

# File storage paths
UPLOAD_DIR = Path("test_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
]

# Initialize FastAPI app
app = FastAPI(
    title="Prismatron Control Interface (Test)",
    description="Test server for the Prismatron web interface",
    version="1.0.0-test",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {len(self.active_connections)} total")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {len(self.active_connections)} total")

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


# API Routes
@app.get("/")
async def root():
    """Serve the main React application."""
    frontend_dir = Path("src/web/frontend/dist")
    index_file = frontend_dir / "index.html"

    if index_file.exists():
        return FileResponse(str(index_file))
    else:
        return JSONResponse(
            {
                "message": "Prismatron Test API Server",
                "version": "1.0.0-test",
                "frontend_status": "Build required - run 'npm run build' in src/web/frontend/",
                "endpoints": {
                    "docs": "/docs",
                    "status": "/api/status",
                    "websocket": "/ws",
                },
            }
        )


@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status."""
    return SystemStatus(
        current_file=playlist_state.items[playlist_state.current_index].file_path
        if playlist_state.items
        and 0 <= playlist_state.current_index < len(playlist_state.items)
        else None,
        playlist_position=playlist_state.current_index,
        brightness=system_settings.brightness,
        uptime=time.time(),
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
        playlist_state.current_index = (playlist_state.current_index + 1) % len(
            playlist_state.items
        )
        await manager.broadcast(
            {"type": "playlist_position", "current_index": playlist_state.current_index}
        )
    return {"current_index": playlist_state.current_index}


@app.post("/api/control/previous")
async def previous_item():
    """Skip to previous playlist item."""
    if playlist_state.items:
        playlist_state.current_index = (playlist_state.current_index - 1) % len(
            playlist_state.items
        )
        await manager.broadcast(
            {"type": "playlist_position", "current_index": playlist_state.current_index}
        )
    return {"current_index": playlist_state.current_index}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    duration: Optional[float] = Form(None),
):
    """Upload image or video file and add to playlist."""
    try:
        # Simple file validation
        allowed_exts = ["jpg", "jpeg", "png", "gif", "mp4", "mov", "webm"]
        file_ext = file.filename.split(".")[-1].lower() if file.filename else ""

        if file_ext not in allowed_exts:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file_ext}"
            )

        # Save file (mock)
        file_path = UPLOAD_DIR / file.filename
        content_type = "image" if file_ext in ["jpg", "jpeg", "png", "gif"] else "video"

        # Create playlist item
        item = PlaylistItem(
            id=f"test_{len(playlist_state.items)}",
            name=name or file.filename or f"Uploaded {content_type}",
            type=content_type,
            file_path=str(file_path),
            duration=duration or (10.0 if content_type == "image" else 30.0),
            order=len(playlist_state.items),
        )

        playlist_state.items.append(item)

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


@app.get("/api/effects", response_model=List[EffectPreset])
async def get_effects():
    """Get available effect presets."""
    return EFFECT_PRESETS


@app.post("/api/effects/{effect_id}/add")
async def add_effect_to_playlist(
    effect_id: str, name: Optional[str] = None, duration: Optional[float] = None
):
    """Add an effect to the playlist."""
    effect_preset = next((e for e in EFFECT_PRESETS if e.id == effect_id), None)
    if not effect_preset:
        raise HTTPException(status_code=404, detail="Effect not found")

    item = PlaylistItem(
        id=f"effect_{len(playlist_state.items)}",
        name=name or effect_preset.name,
        type="effect",
        effect_config={"effect_id": effect_id, "parameters": effect_preset.config},
        duration=duration or 30.0,
        order=len(playlist_state.items),
    )

    playlist_state.items.append(item)

    await manager.broadcast(
        {
            "type": "playlist_updated",
            "items": [item.dict() for item in playlist_state.items],
        }
    )

    return {"status": "added", "item": item.dict()}


@app.get("/api/playlist", response_model=PlaylistState)
async def get_playlist():
    """Get current playlist state."""
    return playlist_state


@app.delete("/api/playlist/{item_id}")
async def remove_playlist_item(item_id: str):
    """Remove item from playlist."""
    original_length = len(playlist_state.items)
    playlist_state.items = [item for item in playlist_state.items if item.id != item_id]

    if len(playlist_state.items) < original_length:
        if playlist_state.current_index >= len(playlist_state.items):
            playlist_state.current_index = max(0, len(playlist_state.items) - 1)

        await manager.broadcast(
            {
                "type": "playlist_updated",
                "items": [item.dict() for item in playlist_state.items],
            }
        )

        return {"status": "removed"}
    else:
        raise HTTPException(status_code=404, detail="Item not found")


@app.get("/api/settings", response_model=SystemSettings)
async def get_settings():
    """Get current system settings."""
    return system_settings


@app.post("/api/settings")
async def update_settings(settings: SystemSettings):
    """Update system settings."""
    global system_settings
    system_settings = settings

    await manager.broadcast({"type": "settings_updated", "settings": settings.dict()})

    return {"status": "updated", "settings": settings.dict()}


@app.post("/api/settings/brightness")
async def set_brightness(brightness: float):
    """Set global brightness."""
    if not 0.0 <= brightness <= 1.0:
        raise HTTPException(
            status_code=400, detail="Brightness must be between 0.0 and 1.0"
        )

    system_settings.brightness = brightness

    await manager.broadcast({"type": "brightness_changed", "brightness": brightness})

    return {"brightness": brightness}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    try:
        await websocket.send_json(
            {
                "type": "initial_state",
                "playlist": playlist_state.dict(),
                "settings": system_settings.dict(),
                "timestamp": time.time(),
            }
        )

        while True:
            try:
                message = await websocket.receive_json()
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


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0-test",
        "active_connections": len(manager.active_connections),
    }


# Serve static files
try:
    frontend_dist = Path("src/web/frontend/dist")
    if frontend_dist.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(frontend_dist / "static")),
            name="static",
        )
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🚀 Starting Prismatron Test Web Interface")
    print("📍 Backend API: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Frontend: http://localhost:3000 (if running dev server)")
    print()

    uvicorn.run(
        "test_web_interface:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
