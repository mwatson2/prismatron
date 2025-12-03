"""
Content Pre-Preparation for Seamless Transitions.

This module provides proactive content source preparation to eliminate
transition delays between playlist items. Uses event-driven timing
rather than polling.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

from .content_sources import ContentSource, ContentSourceRegistry

if TYPE_CHECKING:
    from .producer import ContentPlaylist

logger = logging.getLogger(__name__)


class ContentPreparer:
    """
    Handles pre-preparation of upcoming content sources.

    Uses timer-based scheduling triggered when content starts playing.
    Preparation runs in a background thread to avoid blocking playback.
    """

    def __init__(self, playlist: "ContentPlaylist", lookahead_seconds: float = 2.0):
        """
        Initialize content preparer.

        Args:
            playlist: Reference to the content playlist
            lookahead_seconds: How far ahead to start preparation (default 2.0s)
        """
        self._playlist = playlist
        self._lookahead_seconds = lookahead_seconds

        # Pending source state
        self._pending_source: Optional[ContentSource] = None
        self._pending_item_index: int = -1
        self._pending_item_filepath: str = ""
        self._pending_lock = threading.Lock()

        # Timer for scheduling preparation
        self._prep_timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()

        # Track current item to avoid duplicate timers
        self._current_scheduled_index: int = -1

    def schedule_preparation(self, current_index: int, item_duration: float) -> None:
        """
        Schedule pre-preparation of the next item.

        Called when a new playlist item starts playing.

        Args:
            current_index: Index of the item that just started
            item_duration: Duration of the current item in seconds
        """
        with self._timer_lock:
            # Cancel any existing timer
            if self._prep_timer:
                self._prep_timer.cancel()
                self._prep_timer = None

            # Don't schedule if duration is too short
            if item_duration <= self._lookahead_seconds:
                logger.debug(
                    f"Item duration ({item_duration:.1f}s) <= lookahead "
                    f"({self._lookahead_seconds}s), skipping pre-preparation"
                )
                return

            # Calculate delay before starting preparation
            delay = item_duration - self._lookahead_seconds
            self._current_scheduled_index = current_index

            # Schedule preparation
            self._prep_timer = threading.Timer(delay, self._do_preparation, args=(current_index,))
            self._prep_timer.daemon = True
            self._prep_timer.start()

            logger.info(
                f"Scheduled pre-preparation in {delay:.1f}s (item {current_index}, duration {item_duration:.1f}s)"
            )

    def _do_preparation(self, triggered_from_index: int) -> None:
        """
        Execute preparation of the next content source.

        Runs in timer thread.

        Args:
            triggered_from_index: The index that was playing when timer was set
        """
        # Verify we haven't moved to a different item since scheduling
        with self._timer_lock:
            if self._current_scheduled_index != triggered_from_index:
                logger.debug(
                    f"Skipping preparation: index changed from "
                    f"{triggered_from_index} to {self._current_scheduled_index}"
                )
                return

        # Determine next item (with loop handling)
        next_index = triggered_from_index + 1
        next_item = self._playlist.get_item_at_index(next_index)

        if next_item is None and self._playlist._loop_playlist:
            next_index = 0
            next_item = self._playlist.get_item_at_index(0)

        if next_item is None:
            logger.debug("No next item to pre-prepare")
            return

        filepath = next_item.filepath
        # Truncate filepath for logging (may be JSON for text/effect content)
        display_path = filepath[:100] + "..." if len(filepath) > 100 else filepath
        logger.info(f"Pre-preparing next content: index={next_index}, path={display_path}")

        try:
            # Create a NEW source (don't use item's cached source)
            content_type = ContentSourceRegistry.detect_content_type(filepath)
            source = ContentSourceRegistry.create_source(filepath, content_type)

            if source is None:
                logger.warning(f"Failed to create source for pre-preparation: {display_path}")
                return

            # Run setup (this is the slow part we're moving off main thread)
            start_time = time.time()
            setup_success = source.setup()
            setup_duration = time.time() - start_time

            if setup_success:
                with self._pending_lock:
                    # Clean up any existing pending source
                    if self._pending_source:
                        self._pending_source.cleanup()

                    self._pending_source = source
                    self._pending_item_index = next_index
                    self._pending_item_filepath = filepath

                logger.info(f"Pre-prepared content ready: index={next_index}, setup took {setup_duration*1000:.1f}ms")
            else:
                logger.warning(f"Pre-preparation setup failed: {display_path}")
                source.cleanup()

        except Exception as e:
            logger.error(f"Error pre-preparing content: {e}")

    def get_pending_source(self, expected_index: int, expected_filepath: str) -> Optional[ContentSource]:
        """
        Get pending source if it matches expectations.

        Returns None if:
        - No pending source ready
        - Index doesn't match expected next item
        - Filepath doesn't match expected next item

        The pending source is cleared after being returned (ownership transfer).

        Args:
            expected_index: Expected playlist index for the pending source
            expected_filepath: Expected filepath for the pending source

        Returns:
            ContentSource if valid match, None otherwise
        """
        with self._pending_lock:
            if self._pending_source is None:
                return None

            if self._pending_item_index != expected_index or self._pending_item_filepath != expected_filepath:
                # Truncate for logging
                expected_display = expected_filepath[:50] + "..." if len(expected_filepath) > 50 else expected_filepath
                pending_display = (
                    self._pending_item_filepath[:50] + "..."
                    if len(self._pending_item_filepath) > 50
                    else self._pending_item_filepath
                )
                logger.debug(
                    f"Pending source mismatch: expected ({expected_index}, {expected_display}), "
                    f"have ({self._pending_item_index}, {pending_display})"
                )
                return None

            # Transfer ownership
            source = self._pending_source
            self._pending_source = None
            self._pending_item_index = -1
            self._pending_item_filepath = ""

            logger.info(f"Returning pre-prepared source for index {expected_index}")
            return source

    def invalidate(self) -> None:
        """
        Cancel pending timer and clear pending source.

        Called on playlist sync updates to handle skip/reorder/etc.
        """
        with self._timer_lock:
            if self._prep_timer:
                self._prep_timer.cancel()
                self._prep_timer = None
            self._current_scheduled_index = -1

        with self._pending_lock:
            if self._pending_source:
                logger.info("Invalidating pending content source due to playlist change")
                self._pending_source.cleanup()
                self._pending_source = None
                self._pending_item_index = -1
                self._pending_item_filepath = ""

    def cleanup(self) -> None:
        """Clean up all resources."""
        self.invalidate()
        logger.info("Content preparer cleaned up")
