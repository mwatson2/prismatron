"""
LED Effect Framework.

Provides base classes and infrastructure for applying effects to LED frames.
Effects are applied in sequence, modifying LED values in-place before they
are sent to output sinks.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LedEffect(ABC):
    """
    Base class for LED effects.

    An effect instance is created when the effect starts and destroyed when it ends.
    Effects modify LED values in-place and signal completion via their return value.

    Lifecycle:
    1. Effect is instantiated with start time and parameters
    2. apply() is called once per frame with current LED values and timestamp
    3. Effect modifies LED values in-place
    4. Effect returns True when it should be removed from the active effects list

    Subclasses should implement:
    - apply(): Main effect logic
    - Optionally override __init__() for custom initialization
    """

    def __init__(self, start_time: float, duration: Optional[float] = None, **kwargs):
        """
        Initialize LED effect.

        Args:
            start_time: Wall-clock time when effect starts (from time.time())
            duration: Effect duration in seconds (None = infinite/manual control)
            **kwargs: Additional effect-specific parameters
        """
        self.start_time = start_time
        self.duration = duration
        self.frame_count = 0  # Number of frames processed

        # Effect metadata
        self.effect_name = self.__class__.__name__
        self.effect_params = kwargs

        logger.debug(f"Created {self.effect_name} effect at t={start_time:.3f}s, duration={duration}")

    @abstractmethod
    def apply(self, led_values: np.ndarray, current_time: float) -> bool:
        """
        Apply effect to LED values in-place.

        Args:
            led_values: LED values to modify, shape (led_count, 3) or (led_count,)
                       Values in range [0, 255] for uint8 or [0, 1] for float
            current_time: Current wall-clock time (from time.time())

        Returns:
            True if this is the last frame and effect should be removed, False to continue

        Note:
            This method MUST modify led_values in-place. Do not return a new array.
        """

    def get_elapsed_time(self, current_time: float) -> float:
        """
        Get elapsed time since effect start.

        Args:
            current_time: Current wall-clock time

        Returns:
            Elapsed time in seconds
        """
        return current_time - self.start_time

    def get_progress(self, current_time: float) -> float:
        """
        Get effect progress as a fraction [0, 1].

        Args:
            current_time: Current wall-clock time

        Returns:
            Progress fraction (0=start, 1=end), or 0.0 if duration is None
        """
        if self.duration is None or self.duration <= 0:
            return 0.0

        elapsed = self.get_elapsed_time(current_time)
        return min(1.0, elapsed / self.duration)

    def is_complete(self, current_time: float) -> bool:
        """
        Check if effect has completed based on duration.

        Args:
            current_time: Current wall-clock time

        Returns:
            True if effect duration has elapsed, False otherwise
        """
        if self.duration is None:
            return False

        return self.get_elapsed_time(current_time) >= self.duration

    def get_info(self) -> Dict[str, Any]:
        """
        Get effect information for debugging/logging.

        Returns:
            Dictionary with effect metadata
        """
        return {
            "name": self.effect_name,
            "start_time": self.start_time,
            "duration": self.duration,
            "frame_count": self.frame_count,
            "params": self.effect_params,
        }


class TemplateEffect(LedEffect):
    """
    LED effect that applies pre-optimized LED patterns from a template.

    Template is an array of shape (frames, led_count) containing LED values
    for each frame of the animation. The effect progresses through frames
    based on elapsed time, making it independent of input frame rate.

    The template frames are distributed evenly across the specified duration.
    On each apply() call, the frame closest in time to the current moment is selected.
    """

    def __init__(
        self,
        start_time: float,
        template: np.ndarray,
        duration: float,
        blend_mode: str = "alpha",
        intensity: float = 1.0,
        loop: bool = False,
        **kwargs,
    ):
        """
        Initialize template-based effect.

        Args:
            start_time: Wall-clock time when effect starts
            template: LED pattern array, shape (frames, led_count)
            duration: Effect duration in seconds (template will span this duration)
            blend_mode: How to apply template ("alpha", "add", "multiply", "replace")
            intensity: Effect intensity/opacity [0, 1]
            loop: Whether to loop the template when it reaches the end
            **kwargs: Additional parameters passed to base class
        """
        # Set duration (None if looping infinitely)
        effect_duration = None if loop else duration

        super().__init__(start_time=start_time, duration=effect_duration, **kwargs)

        self.template = template.astype(np.float32)
        self.template_duration = duration  # Duration for one complete template playback
        self.blend_mode = blend_mode
        self.intensity = intensity
        self.loop = loop
        self.num_frames = template.shape[0]

        # Validate template shape
        if self.template.ndim != 2:
            raise ValueError(f"Template must be 2D (frames, leds), got shape {self.template.shape}")

        logger.info(
            f"Created TemplateEffect: {self.num_frames} frames over {duration:.2f}s, "
            f"blend={blend_mode}, intensity={intensity:.2f}, loop={loop}"
        )

    def apply(self, led_values: np.ndarray, current_time: float) -> bool:
        """
        Apply template effect to LED values.

        Selects the template frame closest in time based on the elapsed time
        and template duration. This makes the effect independent of input frame rate.

        Args:
            led_values: LED values to modify (led_count, 3) in range [0, 255]
            current_time: Current wall-clock time

        Returns:
            True if effect is complete (no more frames and not looping)
        """
        self.frame_count += 1

        # Calculate elapsed time since effect start
        elapsed = self.get_elapsed_time(current_time)

        # Handle looping: wrap elapsed time to [0, template_duration)
        if self.loop:
            elapsed = elapsed % self.template_duration

        # Check if effect is complete (only for non-looping effects)
        if not self.loop and elapsed >= self.template_duration:
            return True

        # Map elapsed time to template frame index
        # Frames are distributed evenly across template_duration
        # progress goes from 0.0 to 1.0 over the duration
        progress = min(1.0, elapsed / self.template_duration)

        # Calculate frame index (round to nearest frame)
        # Use (num_frames - 1) to ensure last frame is reached at progress=1.0
        frame_idx = int(round(progress * (self.num_frames - 1)))

        # Clamp to valid range [0, num_frames-1]
        frame_idx = max(0, min(self.num_frames - 1, frame_idx))

        # Get template frame for this LED position
        template_frame = self.template[frame_idx]  # Shape: (led_count,)

        # Validate LED count matches
        if template_frame.shape[0] != led_values.shape[0]:
            logger.error(f"Template LED count {template_frame.shape[0]} != frame LED count {led_values.shape[0]}")
            return True  # Remove effect on error

        # Apply effect based on blend mode
        # Template is single-channel, broadcast to RGB if needed
        if led_values.ndim == 2 and led_values.shape[1] == 3:
            # RGB format: broadcast template to all channels
            template_rgb = template_frame[:, np.newaxis]  # (led_count, 1)

            if self.blend_mode == "alpha":
                # Alpha blend: led = led * (1 - alpha) + template * alpha
                alpha = self.intensity
                led_values[:] = led_values * (1 - alpha) + template_rgb * alpha

            elif self.blend_mode == "add":
                # Additive blend: led = led + template * intensity
                led_values[:] = np.clip(led_values + template_rgb * self.intensity, 0, 255)

            elif self.blend_mode == "multiply":
                # Multiplicative blend: led = led * (template * intensity)
                # Normalize template to [0, 1] for multiplication
                template_normalized = template_rgb / 255.0
                led_values[:] = led_values * (template_normalized * self.intensity)

            elif self.blend_mode == "replace":
                # Direct replacement: led = template * intensity
                led_values[:] = template_rgb * self.intensity

            else:
                logger.warning(f"Unknown blend mode: {self.blend_mode}, using alpha")
                alpha = self.intensity
                led_values[:] = led_values * (1 - alpha) + template_rgb * alpha

        else:
            # Single-channel format or unsupported shape
            logger.warning(f"Unsupported LED values shape: {led_values.shape}, skipping effect")
            return True

        return False  # Continue effect

    def get_info(self) -> Dict[str, Any]:
        """Get template effect information."""
        info = super().get_info()
        info.update(
            {
                "num_frames": self.num_frames,
                "template_duration": self.template_duration,
                "blend_mode": self.blend_mode,
                "intensity": self.intensity,
                "loop": self.loop,
            }
        )
        return info


class LedEffectManager:
    """
    Manages a collection of active LED effects.

    Applies effects in sequence to LED values before they are sent to output sinks.
    Automatically removes completed effects from the active list.
    """

    def __init__(self):
        """Initialize effect manager."""
        self.active_effects = []  # List of active LedEffect instances
        self._effects_applied = 0  # Total number of effects applied
        self._effects_completed = 0  # Total number of effects completed

    def add_effect(self, effect: LedEffect) -> None:
        """
        Add a new effect to the active list.

        Args:
            effect: LedEffect instance to add
        """
        self.active_effects.append(effect)
        logger.info(f"Added effect: {effect.effect_name}, total active: {len(self.active_effects)}")

    def remove_effect(self, effect: LedEffect) -> None:
        """
        Remove an effect from the active list.

        Args:
            effect: LedEffect instance to remove
        """
        if effect in self.active_effects:
            self.active_effects.remove(effect)
            self._effects_completed += 1
            logger.debug(f"Removed effect: {effect.effect_name}, total active: {len(self.active_effects)}")

    def clear_effects(self) -> None:
        """Remove all active effects."""
        count = len(self.active_effects)
        self.active_effects.clear()
        logger.info(f"Cleared {count} active effects")

    def apply_effects(self, led_values: np.ndarray, current_time: float) -> None:
        """
        Apply all active effects to LED values.

        Effects are applied in sequence. Completed effects are automatically removed.

        Args:
            led_values: LED values to modify in-place (led_count, 3)
            current_time: Current wall-clock time
        """
        if not self.active_effects:
            return

        # Apply each effect in sequence (use copy of list for safe removal)
        for effect in self.active_effects[:]:
            try:
                is_complete = effect.apply(led_values, current_time)
                self._effects_applied += 1

                if is_complete:
                    self.remove_effect(effect)

            except Exception as e:
                logger.error(f"Error applying effect {effect.effect_name}: {e}")
                # Remove failed effect
                self.remove_effect(effect)

    def get_active_count(self) -> int:
        """Get number of active effects."""
        return len(self.active_effects)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get effect manager statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "active_effects": len(self.active_effects),
            "effects_applied": self._effects_applied,
            "effects_completed": self._effects_completed,
            "active_effect_names": [e.effect_name for e in self.active_effects],
        }
