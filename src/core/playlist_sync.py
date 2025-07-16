"""
Playlist Synchronization Service.

This service manages playlist state across all processes using message passing
via Unix domain sockets. It provides atomic operations, bidirectional updates,
and ensures consistency between web interface and producer processes.
"""

import asyncio
import json
import logging
import os
import socket
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class PlaylistItem:
    """Playlist item data structure."""

    id: str
    name: str
    type: str  # "image", "video", "effect", "text"
    file_path: Optional[str] = None
    effect_config: Optional[Dict] = None
    duration: Optional[float] = None
    thumbnail: Optional[str] = None
    created_at: float = 0.0
    order: int = 0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlaylistItem":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PlaylistState:
    """Complete playlist state."""

    items: List[PlaylistItem]
    current_index: int = 0
    is_playing: bool = False
    auto_repeat: bool = True
    shuffle: bool = False
    last_updated: float = 0.0

    def __post_init__(self):
        if self.last_updated == 0.0:
            self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "items": [item.to_dict() for item in self.items],
            "current_index": self.current_index,
            "is_playing": self.is_playing,
            "auto_repeat": self.auto_repeat,
            "shuffle": self.shuffle,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlaylistState":
        """Create from dictionary."""
        items = [PlaylistItem.from_dict(item_data) for item_data in data.get("items", [])]
        return cls(
            items=items,
            current_index=data.get("current_index", 0),
            is_playing=data.get("is_playing", False),
            auto_repeat=data.get("auto_repeat", True),
            shuffle=data.get("shuffle", False),
            last_updated=data.get("last_updated", time.time()),
        )


@dataclass
class PlaylistMessage:
    """Message structure for playlist operations."""

    type: str  # "update", "add", "remove", "reorder", "clear", "play", "pause", "next", "prev"
    operation: Optional[str] = None  # Specific operation for update type
    data: Optional[Dict[str, Any]] = None
    client_id: Optional[str] = None
    timestamp: float = 0.0
    request_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.request_id is None:
            self.request_id = str(uuid4())

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "PlaylistMessage":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class PlaylistSyncService:
    """
    Playlist synchronization service using Unix domain sockets.

    This service acts as the single source of truth for playlist state,
    managing updates from multiple clients (web interface, producer, etc.)
    and broadcasting changes to all connected clients.
    """

    def __init__(self, socket_path: str = "/tmp/prismatron_playlist.sock"):
        self.socket_path = socket_path
        self.playlist_state = PlaylistState(items=[])
        self.clients: Dict[str, Dict[str, Any]] = {}  # client_id -> client_info
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.server_thread: Optional[threading.Thread] = None

        # Event callbacks
        self.on_playlist_changed: Optional[Callable[[PlaylistState], None]] = None
        self.on_playback_changed: Optional[Callable[[bool, int], None]] = None

        # Thread safety
        self._lock = threading.Lock()

        logger.info(f"Playlist sync service initialized with socket: {self.socket_path}")

    def start(self) -> bool:
        """Start the playlist synchronization service."""
        try:
            if self.running:
                logger.warning("Playlist sync service already running")
                return True

            # Remove existing socket file
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

            # Create Unix domain socket
            self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_socket.bind(self.socket_path)
            self.server_socket.listen(5)

            # Set socket permissions
            os.chmod(self.socket_path, 0o666)

            self.running = True
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()

            logger.info(f"Playlist sync service started on {self.socket_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to start playlist sync service: {e}")
            return False

    def stop(self):
        """Stop the playlist synchronization service."""
        try:
            self.running = False

            if self.server_socket:
                self.server_socket.close()

            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5.0)

            # Clean up socket file
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

            logger.info("Playlist sync service stopped")

        except Exception as e:
            logger.error(f"Error stopping playlist sync service: {e}")

    def _server_loop(self):
        """Main server loop handling client connections."""
        logger.info("Playlist sync server loop started")

        while self.running:
            try:
                if not self.server_socket:
                    break

                # Accept new client connections
                client_socket, _ = self.server_socket.accept()
                client_id = str(uuid4())

                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client, args=(client_socket, client_id), daemon=True
                )
                client_thread.start()

            except OSError:
                # Socket closed or other OS error
                break
            except Exception as e:
                logger.error(f"Error in server loop: {e}")

        logger.info("Playlist sync server loop ended")

    def _handle_client(self, client_socket: socket.socket, client_id: str):
        """Handle individual client connection."""
        logger.info(f"New playlist sync client connected: {client_id}")

        # Message buffering for proper framing
        message_buffer = ""

        with self._lock:
            self.clients[client_id] = {
                "socket": client_socket,
                "connected_at": time.time(),
                "last_seen": time.time(),
            }

        try:
            # Send current playlist state to new client
            self._send_to_client(
                client_id, PlaylistMessage(type="full_state", data=self.playlist_state.to_dict(), client_id="server")
            )

            # Handle client messages
            while self.running:
                try:
                    # Receive message
                    data = client_socket.recv(4096)
                    if not data:
                        break

                    try:
                        # Add new data to buffer
                        message_buffer += data.decode("utf-8")

                        # Process complete messages (delimited by newlines)
                        while "\n" in message_buffer:
                            message_str, message_buffer = message_buffer.split("\n", 1)

                            if message_str.strip():  # Skip empty messages
                                message = PlaylistMessage.from_json(message_str)
                                message.client_id = client_id

                                # Update last seen
                                with self._lock:
                                    if client_id in self.clients:
                                        self.clients[client_id]["last_seen"] = time.time()

                                # Process message
                                self._process_message(message)

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from client {client_id}: {e}")
                        # Clear buffer on JSON error to prevent cascade failures
                        message_buffer = ""
                    except Exception as e:
                        logger.error(f"Error handling client {client_id} message: {e}")

                except Exception as e:
                    logger.error(f"Error in client {client_id} message loop: {e}")

        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up client
            with self._lock:
                if client_id in self.clients:
                    del self.clients[client_id]

            import contextlib

            with contextlib.suppress(Exception):
                client_socket.close()

            logger.info(f"Playlist sync client disconnected: {client_id}")

    def _process_message(self, message: PlaylistMessage):
        """Process incoming message and update playlist state."""
        try:
            logger.debug(f"Processing message: {message.type} from {message.client_id}")

            with self._lock:
                if message.type == "add_item":
                    self._handle_add_item(message)
                elif message.type == "remove_item":
                    self._handle_remove_item(message)
                elif message.type == "reorder":
                    self._handle_reorder(message)
                elif message.type == "clear":
                    self._handle_clear(message)
                elif message.type == "play":
                    self._handle_play(message)
                elif message.type == "pause":
                    self._handle_pause(message)
                elif message.type == "next":
                    self._handle_next(message)
                elif message.type == "previous":
                    self._handle_previous(message)
                elif message.type == "set_position":
                    self._handle_set_position(message)
                else:
                    logger.warning(f"Unknown message type: {message.type}")
                    return

                # Update timestamp
                self.playlist_state.last_updated = time.time()

                # Broadcast update to all clients (including sender for consistency)
                self._broadcast_update(exclude_client=None)

                # Trigger callbacks
                if message.type in ["add_item", "remove_item", "reorder", "clear"] and self.on_playlist_changed:
                    self.on_playlist_changed(self.playlist_state)

                if message.type in ["play", "pause", "next", "previous", "set_position"] and self.on_playback_changed:
                    self.on_playback_changed(self.playlist_state.is_playing, self.playlist_state.current_index)

        except Exception as e:
            logger.error(f"Error processing message {message.type}: {e}")

    def _handle_add_item(self, message: PlaylistMessage):
        """Handle add item message."""
        if not message.data or "item" not in message.data:
            logger.error("Add item message missing item data")
            return

        item_data = message.data["item"]
        item = PlaylistItem.from_dict(item_data)

        # Add to end or specific position
        position = message.data.get("position")
        if position is not None and 0 <= position <= len(self.playlist_state.items):
            self.playlist_state.items.insert(position, item)
            # Update order for items after insertion point
            for i in range(position, len(self.playlist_state.items)):
                self.playlist_state.items[i].order = i
        else:
            item.order = len(self.playlist_state.items)
            self.playlist_state.items.append(item)

        logger.info(f"Added playlist item: {item.name} at position {item.order}")

    def _handle_remove_item(self, message: PlaylistMessage):
        """Handle remove item message."""
        if not message.data or "item_id" not in message.data:
            logger.error("Remove item message missing item_id")
            return

        item_id = message.data["item_id"]
        original_length = len(self.playlist_state.items)

        # Remove item
        self.playlist_state.items = [item for item in self.playlist_state.items if item.id != item_id]

        if len(self.playlist_state.items) < original_length:
            # Update order for remaining items
            for i, item in enumerate(self.playlist_state.items):
                item.order = i

            # Adjust current index if needed
            if self.playlist_state.current_index >= len(self.playlist_state.items):
                self.playlist_state.current_index = max(0, len(self.playlist_state.items) - 1)

            logger.info(f"Removed playlist item: {item_id}")
        else:
            logger.warning(f"Item not found for removal: {item_id}")

    def _handle_reorder(self, message: PlaylistMessage):
        """Handle reorder message."""
        if not message.data or "item_ids" not in message.data:
            logger.error("Reorder message missing item_ids")
            return

        item_ids = message.data["item_ids"]

        # Create mapping of id to item
        item_map = {item.id: item for item in self.playlist_state.items}

        # Reorder items
        reordered_items = []
        for i, item_id in enumerate(item_ids):
            if item_id in item_map:
                item = item_map[item_id]
                item.order = i
                reordered_items.append(item)

        self.playlist_state.items = reordered_items
        logger.info(f"Reordered playlist: {len(reordered_items)} items")

    def _handle_clear(self, message: PlaylistMessage):
        """Handle clear playlist message."""
        self.playlist_state.items.clear()
        self.playlist_state.current_index = 0
        self.playlist_state.is_playing = False
        logger.info("Cleared playlist")

    def _handle_play(self, message: PlaylistMessage):
        """Handle play message."""
        self.playlist_state.is_playing = True
        logger.info("Playlist playing")

    def _handle_pause(self, message: PlaylistMessage):
        """Handle pause message."""
        self.playlist_state.is_playing = False
        logger.info("Playlist paused")

    def _handle_next(self, message: PlaylistMessage):
        """Handle next item message."""
        if self.playlist_state.items:
            self.playlist_state.current_index = (self.playlist_state.current_index + 1) % len(self.playlist_state.items)
            logger.info(f"Next item: index {self.playlist_state.current_index}")

    def _handle_previous(self, message: PlaylistMessage):
        """Handle previous item message."""
        if self.playlist_state.items:
            self.playlist_state.current_index = (self.playlist_state.current_index - 1) % len(self.playlist_state.items)
            logger.info(f"Previous item: index {self.playlist_state.current_index}")

    def _handle_set_position(self, message: PlaylistMessage):
        """Handle set position message."""
        if not message.data or "index" not in message.data:
            logger.error("Set position message missing index")
            return

        index = message.data["index"]
        if 0 <= index < len(self.playlist_state.items):
            self.playlist_state.current_index = index
            logger.info(f"Set playlist position: index {index}")
        else:
            logger.warning(f"Invalid playlist index: {index}")

    def _broadcast_update(self, exclude_client: Optional[str] = None):
        """Broadcast full playlist state to all clients. Must be called with lock held."""
        state_message = PlaylistMessage(type="full_state", data=self.playlist_state.to_dict(), client_id="server")

        logger.debug(f"Broadcasting to {len(self.clients)} clients (exclude: {exclude_client})")
        clients_to_remove = []
        for client_id in list(self.clients.keys()):
            if exclude_client == client_id:
                continue

            logger.debug(f"Sending update to client {client_id}")
            if not self._send_to_client(client_id, state_message):
                logger.warning(f"Failed to send to client {client_id}")
                clients_to_remove.append(client_id)
            else:
                logger.debug(f"Successfully sent update to client {client_id}")

        # Remove disconnected clients
        for client_id in clients_to_remove:
            if client_id in self.clients:
                del self.clients[client_id]

    def _send_to_client(self, client_id: str, message: PlaylistMessage) -> bool:
        """Send message to specific client with proper framing."""
        try:
            if client_id not in self.clients:
                return False

            client_socket = self.clients[client_id]["socket"]
            message_json = message.to_json()

            # Add newline delimiter for proper message framing
            message_data = (message_json + "\n").encode("utf-8")
            client_socket.send(message_data)
            return True

        except Exception as e:
            logger.warning(f"Failed to send message to client {client_id}: {e}")
            return False

    def get_playlist_state(self) -> PlaylistState:
        """Get current playlist state."""
        with self._lock:
            return PlaylistState.from_dict(self.playlist_state.to_dict())

    def get_client_count(self) -> int:
        """Get number of connected clients."""
        with self._lock:
            return len(self.clients)


class PlaylistSyncClient:
    """
    Client for connecting to the playlist synchronization service.

    Used by web interface and producer process to send/receive playlist updates.
    """

    def __init__(self, socket_path: str = "/tmp/prismatron_playlist.sock", client_name: str = "unknown"):
        self.socket_path = socket_path
        self.client_name = client_name
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.listener_thread: Optional[threading.Thread] = None
        self.running = False

        # Message buffering for proper framing
        self._message_buffer = ""

        # Event callbacks
        self.on_playlist_update: Optional[Callable[[PlaylistState], None]] = None
        self.on_connection_lost: Optional[Callable[[], None]] = None

        logger.info(f"Playlist sync client '{client_name}' initialized")

    def connect(self) -> bool:
        """Connect to the playlist synchronization service."""
        try:
            if self.connected:
                logger.warning(f"Client '{self.client_name}' already connected")
                return True

            # Create Unix domain socket
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.connect(self.socket_path)

            self.connected = True
            self.running = True

            # Start listener thread
            self.listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
            self.listener_thread.start()

            logger.info(f"Playlist sync client '{self.client_name}' connected")
            return True

        except Exception as e:
            logger.error(f"Failed to connect playlist sync client '{self.client_name}': {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the playlist synchronization service."""
        try:
            self.running = False
            self.connected = False

            if self.socket:
                self.socket.close()
                self.socket = None

            if self.listener_thread and self.listener_thread.is_alive():
                self.listener_thread.join(timeout=5.0)

            logger.info(f"Playlist sync client '{self.client_name}' disconnected")

        except Exception as e:
            logger.error(f"Error disconnecting client '{self.client_name}': {e}")

    def _listener_loop(self):
        """Listen for messages from the server."""
        logger.info(f"Playlist sync client '{self.client_name}' listener started")

        try:
            while self.running and self.connected:
                if not self.socket:
                    break

                # Receive message
                data = self.socket.recv(4096)
                if not data:
                    break

                try:
                    # Add new data to buffer
                    self._message_buffer += data.decode("utf-8")

                    # Process complete messages (delimited by newlines)
                    while "\n" in self._message_buffer:
                        message_str, self._message_buffer = self._message_buffer.split("\n", 1)

                        if message_str.strip():  # Skip empty messages
                            message = PlaylistMessage.from_json(message_str)
                            self._handle_message(message)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received by client '{self.client_name}': {e}")
                    # Clear buffer on JSON error to prevent cascade failures
                    self._message_buffer = ""
                except Exception as e:
                    logger.error(f"Error handling message in client '{self.client_name}': {e}")

        except Exception as e:
            logger.error(f"Error in listener loop for client '{self.client_name}': {e}")
        finally:
            self.connected = False
            if self.on_connection_lost:
                self.on_connection_lost()

        logger.info(f"Playlist sync client '{self.client_name}' listener ended")

    def _handle_message(self, message: PlaylistMessage):
        """Handle incoming message from server."""
        try:
            if message.type == "full_state" and message.data and self.on_playlist_update:
                playlist_state = PlaylistState.from_dict(message.data)
                self.on_playlist_update(playlist_state)

        except Exception as e:
            logger.error(f"Error handling message in client '{self.client_name}': {e}")

    def send_message(self, message: PlaylistMessage) -> bool:
        """Send message to server."""
        try:
            if not self.connected or not self.socket:
                logger.warning(f"Client '{self.client_name}' not connected")
                return False

            message_json = message.to_json()
            # Add newline delimiter for proper message framing
            message_data = (message_json + "\n").encode("utf-8")
            self.socket.send(message_data)
            return True

        except Exception as e:
            logger.error(f"Failed to send message from client '{self.client_name}': {e}")
            return False

    # Convenience methods for common operations
    def add_item(self, item: PlaylistItem, position: Optional[int] = None) -> bool:
        """Add item to playlist."""
        data = {"item": item.to_dict()}
        if position is not None:
            data["position"] = position

        return self.send_message(PlaylistMessage(type="add_item", data=data))

    def remove_item(self, item_id: str) -> bool:
        """Remove item from playlist."""
        return self.send_message(PlaylistMessage(type="remove_item", data={"item_id": item_id}))

    def reorder_items(self, item_ids: List[str]) -> bool:
        """Reorder playlist items."""
        return self.send_message(PlaylistMessage(type="reorder", data={"item_ids": item_ids}))

    def clear_playlist(self) -> bool:
        """Clear playlist."""
        return self.send_message(PlaylistMessage(type="clear"))

    def play(self) -> bool:
        """Start playback."""
        return self.send_message(PlaylistMessage(type="play"))

    def pause(self) -> bool:
        """Pause playback."""
        return self.send_message(PlaylistMessage(type="pause"))

    def next_item(self) -> bool:
        """Go to next item."""
        return self.send_message(PlaylistMessage(type="next"))

    def previous_item(self) -> bool:
        """Go to previous item."""
        return self.send_message(PlaylistMessage(type="previous"))

    def set_position(self, index: int) -> bool:
        """Set current playlist position."""
        return self.send_message(PlaylistMessage(type="set_position", data={"index": index}))
