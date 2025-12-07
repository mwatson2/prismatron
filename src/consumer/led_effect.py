"""
LED Effect Framework.

Provides base classes and infrastructure for applying effects to LED frames.
Effects are applied in sequence, modifying LED values in-place before they
are sent to output sinks.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TemplateEffectFactory:
    """
    Factory for creating TemplateEffect instances with cached template data.

    Loads template files once and reuses them across multiple effect instances,
    avoiding repeated file I/O operations (e.g., on every beat).
    """

    _template_cache: Dict[str, np.ndarray] = {}

    @classmethod
    def preload_templates(cls, template_paths: list[str]) -> None:
        """
        Pre-load templates into cache during initialization.

        This should be called during system startup, before any effects are triggered,
        to ensure zero file I/O during render time.

        Args:
            template_paths: List of template file paths to pre-load
        """
        logger.info(f"Pre-loading {len(template_paths)} templates...")

        for path in template_paths:
            try:
                cls.load_template(path)
            except Exception as e:
                logger.error(f"Failed to pre-load template {path}: {e}")
                # Continue loading other templates even if one fails

        cache_info = cls.get_cache_info()
        logger.info(
            f"Pre-loaded {cache_info['cached_templates']} templates, "
            f"total memory: {cache_info['total_memory_bytes'] / 1024 / 1024:.2f} MB"
        )

    @classmethod
    def load_template(cls, template_path: str) -> np.ndarray:
        """
        Load template from file, using cache if available.

        Args:
            template_path: Path to template file (.npy format)

        Returns:
            Template array of shape (frames, led_count)

        Raises:
            FileNotFoundError: If template file doesn't exist
            ValueError: If template format is invalid
        """
        # Check cache first
        if template_path in cls._template_cache:
            logger.debug(f"Using cached template: {template_path}")
            return cls._template_cache[template_path]

        # Load template from file
        logger.info(f"Loading template from file: {template_path}")

        try:
            template = np.load(template_path)

            # Validate template shape
            if template.ndim != 2:
                raise ValueError(f"Template must be 2D (frames, leds), got shape {template.shape}")

            # Cache the template
            cls._template_cache[template_path] = template
            logger.info(f"Cached template: {template_path}, shape={template.shape}")

            return template

        except FileNotFoundError:
            logger.error(f"Template file not found: {template_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading template {template_path}: {e}")
            raise

    @classmethod
    def create_effect(
        cls,
        template_path: str,
        start_time: float,
        duration: float,
        blend_mode: str = "alpha",
        intensity: float = 1.0,
        loop: bool = False,
        add_multiplier: float = 0.4,
        color_thieving: bool = False,
        **kwargs,
    ) -> "TemplateEffect":
        """
        Create a TemplateEffect instance using cached template data.

        Args:
            template_path: Path to template file (.npy format)
            start_time: Frame timestamp when effect starts
            duration: Effect duration in seconds
            blend_mode: How to apply template ("alpha", "add", "multiply", "replace", "boost", "addboost")
            intensity: Effect intensity/opacity [0, 1+]
            loop: Whether to loop the template when it reaches the end
            add_multiplier: Multiplier for additive component in 'addboost' mode [0, 1+]
            color_thieving: Whether to extract color from input LEDs using first template frame
            **kwargs: Additional parameters passed to TemplateEffect

        Returns:
            New TemplateEffect instance with cached template
        """
        template = cls.load_template(template_path)

        return TemplateEffect(
            start_time=start_time,
            template=template,
            duration=duration,
            blend_mode=blend_mode,
            intensity=intensity,
            loop=loop,
            add_multiplier=add_multiplier,
            color_thieving=color_thieving,
            **kwargs,
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached templates to free memory."""
        count = len(cls._template_cache)
        cls._template_cache.clear()
        logger.info(f"Cleared {count} cached templates")

    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """
        Get information about cached templates.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_templates": len(cls._template_cache),
            "template_paths": list(cls._template_cache.keys()),
            "total_memory_bytes": sum(template.nbytes for template in cls._template_cache.values()),
        }


class LedEffect(ABC):
    """
    Base class for LED effects.

    An effect instance is created when the effect starts and destroyed when it ends.
    Effects modify LED values in-place and signal completion via their return value.

    IMPORTANT: All timing is based on the frame timeline (starting from zero),
    NOT wall-clock time. Frame timestamps come from the video/content producer
    and represent the presentation time of each frame.

    Lifecycle:
    1. Effect is instantiated with start frame timestamp and parameters
    2. apply() is called once per frame with current LED values and frame timestamp
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
            start_time: Frame timestamp when effect starts (from frame timeline, starts at 0)
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
    def apply(self, led_values: np.ndarray, frame_timestamp: float) -> bool:
        """
        Apply effect to LED values in-place.

        Args:
            led_values: LED values to modify, shape (led_count, 3) or (led_count,)
                       Values in range [0, 255] for uint8 or [0, 1] for float
            frame_timestamp: Current frame timestamp (from frame timeline, starts at 0)

        Returns:
            True if this is the last frame and effect should be removed, False to continue

        Note:
            This method MUST modify led_values in-place. Do not return a new array.
        """

    def get_elapsed_time(self, frame_timestamp: float) -> float:
        """
        Get elapsed time since effect start.

        Args:
            frame_timestamp: Current frame timestamp

        Returns:
            Elapsed time in seconds
        """
        return frame_timestamp - self.start_time

    def get_progress(self, frame_timestamp: float) -> float:
        """
        Get effect progress as a fraction [0, 1].

        Args:
            frame_timestamp: Current frame timestamp

        Returns:
            Progress fraction (0=start, 1=end), or 0.0 if duration is None
        """
        if self.duration is None or self.duration <= 0:
            return 0.0

        elapsed = self.get_elapsed_time(frame_timestamp)
        return min(1.0, elapsed / self.duration)

    def is_complete(self, frame_timestamp: float) -> bool:
        """
        Check if effect has completed based on duration.

        Args:
            frame_timestamp: Current frame timestamp

        Returns:
            True if effect duration has elapsed, False otherwise
        """
        if self.duration is None:
            return False

        return self.get_elapsed_time(frame_timestamp) >= self.duration

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
        add_multiplier: float = 0.4,
        color_thieving: bool = False,
        **kwargs,
    ):
        """
        Initialize template-based effect.

        Args:
            start_time: Wall-clock time when effect starts
            template: LED pattern array, shape (frames, led_count)
            duration: Effect duration in seconds (template will span this duration)
            blend_mode: How to apply template:
                - "alpha": led = led * (1 - a) + template * a
                - "add": led = led + template * a
                - "multiply": led = led * (template/255 * a)
                - "replace": led = template * a
                - "boost": led = led * (1 + a * template/255)
                - "addboost": led = led * (1 + a * template/255) + template * add_multiplier
            intensity: Effect intensity/opacity parameter 'a' [0, 1+]
            loop: Whether to loop the template when it reaches the end
            add_multiplier: Multiplier for additive component in 'addboost' mode [0, 1+]
            color_thieving: Whether to extract color from input LEDs using first template frame
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
        self.add_multiplier = add_multiplier
        self.color_thieving = color_thieving
        self.num_frames = template.shape[0]

        # Color thieving state
        self.thieved_color = None  # Will be set on first frame if color_thieving is enabled

        # Validate template shape
        if self.template.ndim != 2:
            raise ValueError(f"Template must be 2D (frames, leds), got shape {self.template.shape}")

        logger.info(
            f"Created TemplateEffect: {self.num_frames} frames over {duration:.2f}s, "
            f"blend={blend_mode}, intensity={intensity:.2f}, loop={loop}, color_thieving={color_thieving}"
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

        # Color thieving: extract weighted average color from first frame
        if self.color_thieving and self.thieved_color is None and led_values.ndim == 2 and led_values.shape[1] == 3:
            # Use first template frame as weights
            first_frame = self.template[0]  # Shape: (led_count,)

            # Calculate weighted average RGB color
            # weights shape: (led_count,), led_values shape: (led_count, 3)
            total_weight = np.sum(first_frame)
            if total_weight > 0:
                # Weighted average for each RGB channel
                weighted_rgb = np.sum(led_values * first_frame[:, np.newaxis], axis=0) / total_weight

                # Convert RGB to XYZ colorspace
                # Using sRGB to XYZ conversion matrix (D65 illuminant)
                rgb_normalized = weighted_rgb / 255.0
                xyz_matrix = np.array(
                    [
                        [0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041],
                    ]
                )
                xyz = xyz_matrix @ rgb_normalized

                # Clamp Y (luminance) to 1.0
                xyz[1] = min(xyz[1], 1.0)

                # Convert XYZ back to RGB
                xyz_to_rgb_matrix = np.array(
                    [
                        [3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252],
                    ]
                )
                rgb_normalized_clamped = xyz_to_rgb_matrix @ xyz

                # Clamp to valid RGB range and convert back to [0, 255]
                rgb_normalized_clamped = np.clip(rgb_normalized_clamped, 0.0, 1.0)
                self.thieved_color = rgb_normalized_clamped * 255.0

                logger.info(
                    f"Color thieving: extracted color RGB=({self.thieved_color[0]:.1f}, "
                    f"{self.thieved_color[1]:.1f}, {self.thieved_color[2]:.1f})"
                )
            else:
                # No weight in first frame, use white
                self.thieved_color = np.array([255.0, 255.0, 255.0])
                logger.warning("Color thieving: first template frame has zero weight, using white")

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
                # With color thieving: led = led + template * intensity * color
                if self.color_thieving and self.thieved_color is not None:
                    # Multiply template by thieved color (broadcast across LEDs)
                    colored_template = template_rgb * self.thieved_color[np.newaxis, :] / 255.0
                    led_values[:] = np.clip(led_values + colored_template * self.intensity, 0, 255)
                else:
                    led_values[:] = np.clip(led_values + template_rgb * self.intensity, 0, 255)

            elif self.blend_mode == "multiply":
                # Multiplicative blend: led = led * (template * intensity)
                # Normalize template to [0, 1] for multiplication
                template_normalized = template_rgb / 255.0
                led_values[:] = led_values * (template_normalized * self.intensity)

            elif self.blend_mode == "replace":
                # Direct replacement: led = template * intensity
                led_values[:] = template_rgb * self.intensity

            elif self.blend_mode == "boost":
                # Boost blend: led = led * (1 + intensity * template_normalized)
                # Where template_normalized is template / 255.0
                # This boosts the LED value proportionally to the template value
                # intensity parameter 'a' controls the boost strength
                template_normalized = template_rgb / 255.0
                led_values[:] = np.clip(led_values * (1.0 + self.intensity * template_normalized), 0, 255)

            elif self.blend_mode == "addboost":
                # Combined boost and add blend:
                # First apply boost: led = led * (1 + intensity * template_normalized)
                # Then apply add: led = led + template * add_multiplier
                # With color thieving: add component uses thieved color
                template_normalized = template_rgb / 255.0
                boosted = led_values * (1.0 + self.intensity * template_normalized)

                if self.color_thieving and self.thieved_color is not None:
                    # Multiply template by thieved color for additive component
                    colored_template = template_rgb * self.thieved_color[np.newaxis, :] / 255.0
                    led_values[:] = np.clip(boosted + colored_template * self.add_multiplier, 0, 255)
                else:
                    led_values[:] = np.clip(boosted + template_rgb * self.add_multiplier, 0, 255)

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
                "color_thieving": self.color_thieving,
                "thieved_color": self.thieved_color.tolist() if self.thieved_color is not None else None,
            }
        )
        return info


class BeatBrightnessEffect(LedEffect):
    """
    Audio-reactive beat brightness boost effect.

    Applies a sine wave brightness boost for one beat. The effect is created
    when a beat is detected and automatically expires after the boost duration.

    Formula: multiplier = 1.0 + intensity * scaled_intensity * sin(t * pi / boost_duration)
    where:
    - t = time since beat start
    - intensity = base boost intensity (0.0 to 5.0)
    - scaled_intensity = sqrt(beat_intensity) for dynamic scaling
    - boost_duration = duration_fraction * (60.0 / BPM)
    """

    def __init__(
        self,
        start_time: float,
        bpm: float,
        beat_intensity: float = 1.0,
        boost_intensity: float = 4.0,
        duration_fraction: float = 0.4,
        **kwargs,
    ):
        """
        Initialize beat brightness boost effect.

        Args:
            start_time: Wall-clock time when beat occurs
            bpm: Current BPM (beats per minute)
            beat_intensity: Beat intensity value from beat detection [0, 1]
            boost_intensity: Base boost intensity multiplier [0, 5.0]
            duration_fraction: Fraction of beat interval for boost [0.1, 1.0]
            **kwargs: Additional parameters passed to base class
        """
        # Calculate beat duration and boost window
        beat_duration = 60.0 / max(1.0, bpm)  # Prevent division by zero
        boost_duration = duration_fraction * beat_duration

        super().__init__(start_time=start_time, duration=boost_duration, **kwargs)

        self.bpm = bpm
        self.beat_intensity = beat_intensity
        self.boost_intensity = boost_intensity
        self.duration_fraction = duration_fraction
        self.boost_duration = boost_duration

        # Calculate scaled intensity using sqrt for better dynamic range
        self.intensity_scaled = np.sqrt(beat_intensity)
        self.dynamic_boost_factor = boost_intensity * self.intensity_scaled

        logger.debug(
            f"Created BeatBrightnessEffect: BPM={bpm:.1f}, "
            f"beat_intensity={beat_intensity:.2f}, "
            f"boost_duration={boost_duration*1000:.1f}ms, "
            f"dynamic_factor={self.dynamic_boost_factor:.2f}"
        )

    def apply(self, led_values: np.ndarray, current_time: float) -> bool:
        """
        Apply beat brightness boost to LED values.

        Args:
            led_values: LED values to modify (led_count, 3) in range [0, 255]
            current_time: Current wall-clock time

        Returns:
            True if boost window is complete
        """
        self.frame_count += 1

        # Get elapsed time since beat
        elapsed = self.get_elapsed_time(current_time)

        # Check if boost window is complete
        if elapsed >= self.boost_duration:
            return True

        # Calculate sine wave boost: sin(t * pi / boost_duration)
        # This goes from 0 -> 1 -> 0 over the boost_duration
        boost = self.dynamic_boost_factor * np.sin(elapsed * np.pi / self.boost_duration)
        multiplier = 1.0 + boost

        # Apply boost as brightness multiplier
        # Clamp to [0, 255] for uint8, or [0, 1] for float
        if led_values.dtype == np.uint8 or led_values.max() > 1.0:
            # uint8 format [0, 255]
            led_values[:] = np.clip(led_values * multiplier, 0, 255).astype(led_values.dtype)
        else:
            # float format [0, 1]
            led_values[:] = np.clip(led_values * multiplier, 0, 1)

        return False  # Continue effect

    def get_info(self) -> Dict[str, Any]:
        """Get beat brightness effect information."""
        info = super().get_info()
        info.update(
            {
                "bpm": self.bpm,
                "beat_intensity": self.beat_intensity,
                "boost_intensity": self.boost_intensity,
                "duration_fraction": self.duration_fraction,
                "boost_duration": self.boost_duration,
                "dynamic_boost_factor": self.dynamic_boost_factor,
            }
        )
        return info


class InverseFadeEffect(LedEffect):
    """
    LED effect that fades between the current LED values and full white (255).

    Unlike a regular fade which goes to/from black (0), this inverse fade
    transitions to/from full white.

    - "out" direction: Animates from current LED values (x) to white (255)
    - "in" direction: Animates from white (255) to current LED values (x)

    Uses linear interpolation over the specified duration.
    """

    def __init__(
        self,
        start_time: float,
        duration: float,
        direction: str = "out",
        curve: str = "linear",
        **kwargs,
    ):
        """
        Initialize inverse fade effect.

        Args:
            start_time: Frame timestamp when effect starts
            duration: Effect duration in seconds
            direction: "out" to fade from current to white, "in" to fade from white to current
            curve: Interpolation curve ("linear", "ease-in", "ease-out", "ease-in-out")
            **kwargs: Additional parameters passed to base class
        """
        super().__init__(start_time=start_time, duration=duration, **kwargs)

        if direction not in ("in", "out"):
            raise ValueError(f"direction must be 'in' or 'out', got '{direction}'")

        self.direction = direction
        self.curve = curve

        # Store the initial LED values for "in" direction (will be captured on first frame)
        self.target_values = None

        logger.info(f"Created InverseFadeEffect: direction={direction}, duration={duration:.2f}s, curve={curve}")

    def _apply_curve(self, progress: float) -> float:
        """
        Apply interpolation curve to linear progress.

        Args:
            progress: Linear progress value between 0.0 and 1.0

        Returns:
            Transformed progress value with curve applied
        """
        progress = max(0.0, min(1.0, progress))

        if self.curve == "linear":
            return progress
        elif self.curve == "ease-in":
            # Quadratic ease-in (slow start, fast end)
            return progress * progress
        elif self.curve == "ease-out":
            # Quadratic ease-out (fast start, slow end)
            return 1.0 - (1.0 - progress) * (1.0 - progress)
        elif self.curve == "ease-in-out":
            # Cubic ease-in-out (slow start and end, fast middle)
            if progress < 0.5:
                return 2.0 * progress * progress
            else:
                return 1.0 - 2.0 * (1.0 - progress) * (1.0 - progress)
        else:
            logger.warning(f"Unknown curve type '{self.curve}', using linear")
            return progress

    def apply(self, led_values: np.ndarray, frame_timestamp: float) -> bool:
        """
        Apply inverse fade effect to LED values.

        Args:
            led_values: LED values to modify (led_count, 3) in range [0, 255]
            frame_timestamp: Current frame timestamp

        Returns:
            True if effect is complete
        """
        self.frame_count += 1

        # Check if effect is complete
        if self.is_complete(frame_timestamp):
            # Apply final state
            if self.direction == "out":
                # Final state is white
                led_values[:] = 255
            # For "in" direction, final state is the target values (already in led_values)
            return True

        # Calculate progress
        progress = self.get_progress(frame_timestamp)
        curved_progress = self._apply_curve(progress)

        if self.direction == "out":
            # Fade from current values to white
            # led = current * (1 - progress) + 255 * progress
            # led = current + progress * (255 - current)
            led_values[:] = np.clip(
                led_values + curved_progress * (255.0 - led_values.astype(np.float32)),
                0,
                255,
            ).astype(led_values.dtype)

        else:  # direction == "in"
            # Fade from white to target values
            # On first frame, capture target values and set to white
            if self.target_values is None:
                self.target_values = led_values.copy()
                led_values[:] = 255

            # led = 255 * (1 - progress) + target * progress
            # led = 255 + progress * (target - 255)
            led_values[:] = np.clip(
                255.0 + curved_progress * (self.target_values.astype(np.float32) - 255.0),
                0,
                255,
            ).astype(led_values.dtype)

        return False

    def get_info(self) -> Dict[str, Any]:
        """Get inverse fade effect information."""
        info = super().get_info()
        info.update(
            {
                "direction": self.direction,
                "curve": self.curve,
                "has_target_values": self.target_values is not None,
            }
        )
        return info


@dataclass
class SparkleSet:
    """Represents a set of LEDs that were sparkled at a specific time."""

    start_time: float  # When this set started sparkling
    count: int  # Number of LEDs in this set
    seed: int  # RNG seed to reproduce the LED selection and colors


class SparkleEffect(LedEffect):
    """
    LED effect that creates a sparkling/glitter effect.

    Every `interval_ms` milliseconds, a fraction `density` of LEDs are set to full
    brightness (white) and then fade linearly over `fade_ms` milliseconds back to
    their target color.

    Since fade_ms can be greater than interval_ms, multiple overlapping sparkle sets
    can be active at once. Each set is tracked by its start time, count, and RNG seed
    for compact storage.

    This effect has no fixed duration - it continues until explicitly removed.
    Parameters (interval_ms, fade_ms, density) can be modified at any time.
    """

    # XYZ to RGB conversion matrix (D65 illuminant)
    _XYZ_TO_RGB = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )

    def __init__(
        self,
        start_time: float,
        interval_ms: float = 50.0,
        fade_ms: float = 200.0,
        density: float = 0.05,
        led_count: int = 2600,
        random_colors: bool = False,
        **kwargs,
    ):
        """
        Initialize sparkle effect.

        Args:
            start_time: Frame timestamp when effect starts
            interval_ms: Interval between sparkle bursts in milliseconds
            fade_ms: Duration of fade from white to target color in milliseconds
            density: Fraction of LEDs to sparkle each interval [0, 1]
            led_count: Total number of LEDs in the system
            random_colors: If True, sparkles use random colors instead of white
            **kwargs: Additional parameters passed to base class
        """
        # No fixed duration - effect continues until removed
        super().__init__(start_time=start_time, duration=None, **kwargs)

        self.interval_ms = interval_ms
        self.fade_ms = fade_ms
        self.density = density
        self.led_count = led_count
        self.random_colors = random_colors

        # Track active sparkle sets
        self.active_sets: list[SparkleSet] = []

        # Track the last interval boundary we processed
        self.last_interval_boundary = 0

        # RNG for generating seeds (use start_time as initial seed for reproducibility)
        self._seed_rng = np.random.default_rng(int(start_time * 1000) % (2**31))

        logger.info(
            f"Created SparkleEffect: interval={interval_ms}ms, fade={fade_ms}ms, "
            f"density={density:.1%}, led_count={led_count}, random_colors={random_colors}"
        )

    def set_parameters(
        self,
        interval_ms: Optional[float] = None,
        fade_ms: Optional[float] = None,
        density: Optional[float] = None,
        random_colors: Optional[bool] = None,
    ) -> None:
        """
        Update effect parameters dynamically.

        Args:
            interval_ms: New interval between sparkle bursts (or None to keep current)
            fade_ms: New fade duration (or None to keep current)
            density: New LED density fraction (or None to keep current)
            random_colors: New random colors setting (or None to keep current)
        """
        if interval_ms is not None:
            self.interval_ms = interval_ms
        if fade_ms is not None:
            self.fade_ms = fade_ms
        if density is not None:
            self.density = max(0.0, min(1.0, density))
        if random_colors is not None:
            self.random_colors = random_colors

        logger.debug(
            f"SparkleEffect parameters updated: interval={self.interval_ms}ms, "
            f"fade={self.fade_ms}ms, density={self.density:.1%}, random_colors={self.random_colors}"
        )

    def _get_sparkle_data(self, count: int, seed: int) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the LED indices and optional random colors for a sparkle set using its seed.

        Args:
            count: Number of LEDs to select
            seed: RNG seed for reproducible selection

        Returns:
            Tuple of (indices array, colors array or None)
            Colors array is shape (count, 3) RGB values if random_colors is enabled
        """
        rng = np.random.default_rng(seed)
        actual_count = min(count, self.led_count)
        indices = rng.choice(self.led_count, size=actual_count, replace=False)

        if not self.random_colors:
            return indices, None

        # Generate random colors in XYZ space, then convert to RGB
        # We pick random X and Z (chromaticity), with Y=1.0 (full luminance)
        # X and Z typically range from 0 to ~1.1 for visible colors
        # Using a simpler approach: generate random hues in HSV then convert to RGB
        # This ensures vibrant, saturated colors

        # Generate random hues (0-1), full saturation, full value
        hues = rng.uniform(0, 1, size=actual_count)

        # HSV to RGB conversion (S=1, V=1)
        # H is in [0,1], maps to [0,360) degrees
        h6 = hues * 6.0
        sector = h6.astype(int) % 6
        f = h6 - sector

        # For S=1, V=1: p=0, q=1-f, t=f
        colors = np.zeros((actual_count, 3), dtype=np.float32)

        # Red channel
        colors[:, 0] = np.where(
            sector == 0,
            1.0,
            np.where(
                sector == 1,
                1.0 - f,
                np.where(sector == 2, 0.0, np.where(sector == 3, 0.0, np.where(sector == 4, f, 1.0))),
            ),
        )
        # Green channel
        colors[:, 1] = np.where(
            sector == 0,
            f,
            np.where(
                sector == 1,
                1.0,
                np.where(sector == 2, 1.0, np.where(sector == 3, 1.0 - f, np.where(sector == 4, 0.0, 0.0))),
            ),
        )
        # Blue channel
        colors[:, 2] = np.where(
            sector == 0,
            0.0,
            np.where(
                sector == 1,
                0.0,
                np.where(sector == 2, f, np.where(sector == 3, 1.0, np.where(sector == 4, 1.0, 1.0 - f))),
            ),
        )

        # Scale to 0-255
        colors = (colors * 255.0).astype(np.float32)

        return indices, colors

    def apply(self, led_values: np.ndarray, frame_timestamp: float) -> bool:
        """
        Apply sparkle effect to LED values.

        Args:
            led_values: LED values to modify (led_count, 3) in range [0, 255]
            frame_timestamp: Current frame timestamp

        Returns:
            Always False (effect never completes on its own)
        """
        self.frame_count += 1

        # Update led_count from actual array if different
        actual_led_count = led_values.shape[0]
        if actual_led_count != self.led_count:
            self.led_count = actual_led_count

        # Calculate elapsed time in milliseconds
        elapsed_ms = self.get_elapsed_time(frame_timestamp) * 1000.0
        fade_ms = self.fade_ms
        interval_ms = self.interval_ms

        # Step 1: Remove expired sparkle sets (older than fade_ms)
        cutoff_time = frame_timestamp - (fade_ms / 1000.0)
        self.active_sets = [s for s in self.active_sets if s.start_time > cutoff_time]

        # Step 2: Determine how many new sparkle sets need to be added
        # Calculate which interval boundaries have passed since last frame
        current_interval = int(elapsed_ms / interval_ms)

        # Add new sets for each interval boundary that has passed
        while self.last_interval_boundary < current_interval:
            self.last_interval_boundary += 1

            # Calculate the actual start time for this interval boundary
            boundary_elapsed_ms = self.last_interval_boundary * interval_ms
            set_start_time = self.start_time + (boundary_elapsed_ms / 1000.0)

            # Only add if the set hasn't already expired
            if set_start_time > cutoff_time:
                # Calculate number of LEDs to sparkle
                count = max(1, int(self.led_count * self.density))

                # Generate a new seed for this set
                seed = int(self._seed_rng.integers(0, 2**31))

                self.active_sets.append(SparkleSet(start_time=set_start_time, count=count, seed=seed))

        # Step 3: Apply fades for all active sparkle sets
        # We need to blend sparkle color towards target based on how far through the fade we are
        # fade_progress = 0: full sparkle color, fade_progress = 1: target color

        for sparkle_set in self.active_sets:
            # Calculate fade progress for this set
            set_age_ms = (frame_timestamp - sparkle_set.start_time) * 1000.0
            fade_progress = min(1.0, max(0.0, set_age_ms / fade_ms))

            # Get the LED indices and optional random colors for this set
            indices, sparkle_colors = self._get_sparkle_data(sparkle_set.count, sparkle_set.seed)

            # Apply the fade: lerp from sparkle color to target color
            # At fade_progress=0: full sparkle color (white or random)
            # At fade_progress=1: original target color

            if led_values.ndim == 2 and led_values.shape[1] == 3:
                # RGB format
                target_colors = led_values[indices].astype(np.float32)

                if sparkle_colors is not None:
                    # Random colors mode: fade from random color to target
                    # Scale random color by (1 - fade_progress) to fade out
                    sparkle_blend = sparkle_colors * (1.0 - fade_progress)
                    led_values[indices] = np.clip(sparkle_blend + target_colors * fade_progress, 0, 255).astype(
                        led_values.dtype
                    )
                else:
                    # White sparkle mode: fade from white to target
                    white_blend = 255.0 * (1.0 - fade_progress)
                    led_values[indices] = np.clip(white_blend + target_colors * fade_progress, 0, 255).astype(
                        led_values.dtype
                    )
            else:
                # Single channel - random colors not supported, use white
                target_values = led_values[indices].astype(np.float32)
                white_blend = 255.0 * (1.0 - fade_progress)
                led_values[indices] = np.clip(white_blend + target_values * fade_progress, 0, 255).astype(
                    led_values.dtype
                )

        return False  # Effect never completes on its own

    def get_info(self) -> Dict[str, Any]:
        """Get sparkle effect information."""
        info = super().get_info()
        info.update(
            {
                "interval_ms": self.interval_ms,
                "fade_ms": self.fade_ms,
                "density": self.density,
                "led_count": self.led_count,
                "active_sets": len(self.active_sets),
                "random_colors": self.random_colors,
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

    def apply_effects(self, led_values: np.ndarray, frame_timestamp: float) -> None:
        """
        Apply all active effects to LED values.

        Effects are applied in sequence. Completed effects are automatically removed.

        Args:
            led_values: LED values to modify in-place (led_count, 3)
            frame_timestamp: Current frame timestamp (from frame timeline, starts at 0)
        """
        if not self.active_effects:
            return

        # Apply each effect in sequence (use copy of list for safe removal)
        for effect in self.active_effects[:]:
            try:
                is_complete = effect.apply(led_values, frame_timestamp)
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
