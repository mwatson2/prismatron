"""Visual effects source for the producer pipeline"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from ..shared.frame_data import FrameData
from .effects import EffectRegistry


class EffectSource:
    """Source that generates visual effects for LED display."""

    def __init__(self, width: int = 128, height: int = 64, fps: int = 30):
        """Initialize effect source.

        Args:
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps

        self.current_effect = None
        self.effect_start_time = 0
        self.effect_duration = 30.0  # Default duration
        self.frame_count = 0

        self.logger = logging.getLogger(__name__)

        # Auto-rotation settings
        self.auto_rotate = False
        self.rotation_effects = []
        self.rotation_index = 0

        self.logger.info(f"Effect source initialized: {width}x{height}@{fps}fps")

    def set_effect(self, effect_id: str, config: Optional[Dict[str, Any]] = None, duration: float = 30.0) -> bool:
        """Set the current effect.

        Args:
            effect_id: ID of effect to load
            config: Effect configuration
            duration: How long to run effect (seconds)

        Returns:
            True if effect was set successfully
        """
        try:
            effect = EffectRegistry.create_effect(effect_id, self.width, self.height, self.fps, config)
            if effect is None:
                self.logger.error(f"Failed to create effect: {effect_id}")
                return False

            self.current_effect = effect
            self.effect_start_time = time.time()
            self.effect_duration = duration
            self.frame_count = 0

            effect_info = EffectRegistry.get_effect(effect_id)
            self.logger.info(f"Set effect: {effect_info['name']} ({effect_id}) for {duration}s")
            return True

        except Exception as e:
            self.logger.error(f"Error setting effect {effect_id}: {e}")
            return False

    def set_auto_rotation(self, enabled: bool, effect_list: Optional[list] = None):
        """Enable/disable automatic effect rotation.

        Args:
            enabled: Whether to enable auto-rotation
            effect_list: List of effect IDs to rotate through (if None, uses all effects)
        """
        self.auto_rotate = enabled

        if effect_list:
            self.rotation_effects = effect_list
        else:
            # Use all registered effects
            all_effects = EffectRegistry.list_effects()
            self.rotation_effects = [effect["id"] for effect in all_effects]

        self.rotation_index = 0

        self.logger.info(
            f"Auto-rotation {'enabled' if enabled else 'disabled'} " f"with {len(self.rotation_effects)} effects"
        )

    def get_frame(self) -> Optional[FrameData]:
        """Get the next frame from the current effect.

        Returns:
            FrameData with the generated frame, or None if no effect active
        """
        if self.current_effect is None:
            if self.auto_rotate and self.rotation_effects:
                # Start first effect in rotation
                self._rotate_to_next_effect()
            else:
                return None

        try:
            # Check if current effect should end
            elapsed = time.time() - self.effect_start_time
            if elapsed >= self.effect_duration:
                if self.auto_rotate and self.rotation_effects:
                    self._rotate_to_next_effect()
                else:
                    self.logger.info("Effect duration expired, stopping")
                    self.current_effect = None
                    return None

            # Generate frame
            frame_data = self.current_effect.generate_frame()

            if frame_data is None or frame_data.shape != (self.height, self.width, 3):
                self.logger.error(f"Invalid frame shape: {frame_data.shape if frame_data is not None else None}")
                return None

            # Create FrameData
            timestamp = time.time()
            frame = FrameData(
                data=frame_data, timestamp=timestamp, frame_id=self.frame_count, width=self.width, height=self.height
            )

            self.frame_count += 1
            return frame

        except Exception as e:
            self.logger.error(f"Error generating frame: {e}")
            return None

    def _rotate_to_next_effect(self):
        """Rotate to the next effect in the list."""
        if not self.rotation_effects:
            return

        effect_id = self.rotation_effects[self.rotation_index]
        self.rotation_index = (self.rotation_index + 1) % len(self.rotation_effects)

        # Set effect with default configuration
        self.set_effect(effect_id, duration=self.effect_duration)

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the effect source.

        Returns:
            Status dictionary
        """
        if self.current_effect is None:
            return {
                "active": False,
                "effect": None,
                "elapsed": 0,
                "duration": 0,
                "progress": 0,
                "auto_rotate": self.auto_rotate,
                "frame_count": self.frame_count,
            }

        elapsed = time.time() - self.effect_start_time
        progress = min(elapsed / self.effect_duration, 1.0) if self.effect_duration > 0 else 0

        # Get current effect info
        current_effect_info = None
        for effect_info in EffectRegistry.list_effects():
            if isinstance(self.current_effect, effect_info["class"]):
                current_effect_info = effect_info
                break

        return {
            "active": True,
            "effect": current_effect_info,
            "elapsed": elapsed,
            "duration": self.effect_duration,
            "progress": progress,
            "auto_rotate": self.auto_rotate,
            "frame_count": self.frame_count,
        }

    def list_available_effects(self) -> list:
        """Get list of all available effects.

        Returns:
            List of effect information dictionaries
        """
        return EffectRegistry.list_effects()

    def stop(self):
        """Stop the current effect."""
        if self.current_effect:
            effect_name = getattr(self.current_effect, "__class__", {}).get("__name__", "Unknown")
            self.logger.info(f"Stopping effect: {effect_name}")

        self.current_effect = None
        self.auto_rotate = False
        self.frame_count = 0


class EffectSourceManager:
    """Manager for multiple effect sources and playlists."""

    def __init__(self):
        self.sources = {}
        self.playlist = []
        self.playlist_index = 0
        self.current_source_id = None
        self.logger = logging.getLogger(__name__)

    def create_source(self, source_id: str, width: int = 128, height: int = 64, fps: int = 30) -> EffectSource:
        """Create a new effect source.

        Args:
            source_id: Unique identifier for the source
            width: Frame width
            height: Frame height
            fps: Target FPS

        Returns:
            Created EffectSource
        """
        source = EffectSource(width, height, fps)
        self.sources[source_id] = source

        if self.current_source_id is None:
            self.current_source_id = source_id

        self.logger.info(f"Created effect source: {source_id}")
        return source

    def get_source(self, source_id: str) -> Optional[EffectSource]:
        """Get an effect source by ID."""
        return self.sources.get(source_id)

    def set_active_source(self, source_id: str) -> bool:
        """Set the active effect source."""
        if source_id in self.sources:
            self.current_source_id = source_id
            self.logger.info(f"Set active source: {source_id}")
            return True
        return False

    def get_active_source(self) -> Optional[EffectSource]:
        """Get the currently active source."""
        if self.current_source_id:
            return self.sources.get(self.current_source_id)
        return None

    def add_to_playlist(
        self,
        effect_id: str,
        config: Optional[Dict[str, Any]] = None,
        duration: float = 30.0,
        name: Optional[str] = None,
    ) -> bool:
        """Add an effect to the playlist.

        Args:
            effect_id: Effect to add
            config: Effect configuration
            duration: Duration in seconds
            name: Custom name for playlist entry

        Returns:
            True if added successfully
        """
        try:
            # Validate effect exists
            if EffectRegistry.get_effect(effect_id) is None:
                self.logger.error(f"Effect not found: {effect_id}")
                return False

            playlist_entry = {
                "effect_id": effect_id,
                "config": config or {},
                "duration": duration,
                "name": name or EffectRegistry.get_effect(effect_id)["name"],
            }

            self.playlist.append(playlist_entry)
            self.logger.info(f"Added to playlist: {playlist_entry['name']}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding to playlist: {e}")
            return False

    def start_playlist(self, source_id: Optional[str] = None) -> bool:
        """Start playing the playlist.

        Args:
            source_id: Source to play playlist on (uses active source if None)

        Returns:
            True if playlist started
        """
        if not self.playlist:
            self.logger.warning("Cannot start playlist: empty")
            return False

        source = self.get_source(source_id) if source_id else self.get_active_source()
        if not source:
            self.logger.error("No source available for playlist")
            return False

        # Set up auto-rotation through playlist
        effect_ids = [entry["effect_id"] for entry in self.playlist]
        source.set_auto_rotation(True, effect_ids)

        # Start with first effect
        first_entry = self.playlist[0]
        source.set_effect(first_entry["effect_id"], first_entry["config"], first_entry["duration"])

        self.playlist_index = 0
        self.logger.info(f"Started playlist with {len(self.playlist)} effects")
        return True

    def clear_playlist(self):
        """Clear the current playlist."""
        self.playlist.clear()
        self.playlist_index = 0
        self.logger.info("Playlist cleared")

    def get_playlist_status(self) -> Dict[str, Any]:
        """Get playlist status.

        Returns:
            Playlist status information
        """
        return {"count": len(self.playlist), "current_index": self.playlist_index, "entries": self.playlist}
