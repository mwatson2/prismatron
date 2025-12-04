"""
Timestamp-Based Frame Renderer.

This module implements precise timestamp-based rendering for LED displays.
Handles wallclock timing establishment, late/early frame logic, and output
to multiple targets (WLED, test renderer).
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .led_effect import LedEffectManager
from .test_sink import TestSink
from .wled_sink import WLEDSink

logger = logging.getLogger(__name__)


@dataclass
class EffectTriggerConfig:
    """
    Configuration for an effect trigger.

    A trigger defines:
    1. When to fire (beat detection, periodic test, etc.)
    2. What conditions must be met (thresholds, BPM ranges)
    3. Which effect to create and with what parameters
    """

    trigger_type: str  # "beat" or "test"
    effect_class: str  # Name of effect class from led_effect module
    effect_params: Dict[str, Any] = field(default_factory=dict)

    # Beat trigger conditions (all must be satisfied)
    confidence_min: Optional[float] = None
    intensity_min: Optional[float] = None
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None

    # Test trigger timing (global interval set at manager level)

    def __post_init__(self):
        """Validate configuration."""
        if self.trigger_type not in ("beat", "test"):
            raise ValueError(f"Invalid trigger_type: {self.trigger_type}")

        # Validate beat trigger has at least one condition
        if self.trigger_type == "beat":
            has_condition = any(
                [
                    self.confidence_min is not None,
                    self.intensity_min is not None,
                    self.bpm_min is not None,
                    self.bpm_max is not None,
                ]
            )
            if not has_condition:
                logger.warning("Beat trigger has no conditions - will match all beats")


class EffectTriggerManager:
    """
    Manages effect triggers and creates effect instances based on configured rules.

    Evaluates common rules first (always active), then carousel rule sets (rotating).
    Uses first-match semantics within each rule list.
    Supports beat triggers (audio-reactive) and test triggers (periodic testing).
    """

    def __init__(self, effect_manager: LedEffectManager):
        """
        Initialize trigger manager.

        Args:
            effect_manager: LedEffectManager instance to add created effects to
        """
        self.effect_manager = effect_manager

        # Common rules (always active, checked first)
        self.common_triggers: List[EffectTriggerConfig] = []

        # Carousel rule sets (rotates every N beats)
        self.carousel_rule_sets: List[List[EffectTriggerConfig]] = []
        self.carousel_beat_interval = 4  # Number of beats between carousel rotations
        self.carousel_current_index = 0  # Current active carousel rule set index
        self.carousel_beat_count = 0  # Beat counter for carousel rotation

        # Test trigger state
        self.test_trigger_interval = 2.0  # Global interval for test triggers (seconds)
        self._last_test_trigger_time = -float("inf")  # Allow first trigger to fire immediately

        # Beat trigger state
        self._last_beat_time_processed = -1.0  # Track last beat we created an effect for

        logger.info("EffectTriggerManager initialized with common + carousel structure")

    def set_triggers(self, triggers: List[EffectTriggerConfig]) -> None:
        """
        Set trigger configuration (backward compatibility - converts to common rules).

        Args:
            triggers: List of trigger configurations
        """
        self.common_triggers = triggers
        self.carousel_rule_sets = []
        logger.info(f"[Backward Compatibility] Configured {len(triggers)} effect triggers as common rules")

    def set_common_and_carousel_triggers(
        self,
        common_triggers: List[EffectTriggerConfig],
        carousel_rule_sets: List[List[EffectTriggerConfig]],
        carousel_beat_interval: int = 4,
    ) -> None:
        """
        Set common rules and carousel rule sets.

        Args:
            common_triggers: Common rules (always active, checked first)
            carousel_rule_sets: List of carousel rule sets (each set is a list of triggers)
            carousel_beat_interval: Number of beats between carousel rotations
        """
        self.common_triggers = common_triggers
        self.carousel_rule_sets = carousel_rule_sets
        self.carousel_beat_interval = carousel_beat_interval
        self.carousel_current_index = 0
        self.carousel_beat_count = 0

        total_carousel_rules = sum(len(rule_set) for rule_set in carousel_rule_sets)
        logger.info(
            f"Configured audio reactive triggers: "
            f"{len(common_triggers)} common rules, "
            f"{len(carousel_rule_sets)} carousel sets ({total_carousel_rules} total carousel rules), "
            f"carousel interval={carousel_beat_interval} beats"
        )

    def set_test_interval(self, interval: float) -> None:
        """Set global test trigger interval."""
        self.test_trigger_interval = interval
        logger.info(f"Test trigger interval set to {interval}s")

    def evaluate_beat_triggers(
        self, frame_timeline_time: float, beat_state: Any, last_beat_wallclock_time: float, wallclock_delta: float
    ) -> None:
        """
        Evaluate beat triggers and create effects for new beats.

        Checks common rules first (always active), then carousel rules (rotating).
        Uses first-match semantics - stops after first matching rule.

        Note: Beat intensity is accumulated over 3 audio frames (~35ms) for accuracy.
        If beat_intensity_ready is False, we skip this frame and wait for intensity
        to be finalized. The original beat timestamp is still used for effect timing.

        Args:
            frame_timeline_time: Current time on frame timeline
            beat_state: AudioBeatState from beat analyzer
            last_beat_wallclock_time: Wall-clock time of last beat
            wallclock_delta: Delta between wall-clock and frame timeline
        """
        # Check if this is a new beat we haven't processed yet
        if last_beat_wallclock_time <= self._last_beat_time_processed:
            return  # Already processed this beat

        # Check if beat intensity is ready (accumulated over 3 frames)
        # If not ready, skip this frame - we'll process on the next frame when intensity is available
        beat_intensity_ready = getattr(beat_state, "beat_intensity_ready", True)
        if not beat_intensity_ready:
            # Beat detected but intensity still accumulating - wait for next frame
            # This adds ~20-35ms latency but gives much more accurate intensity
            return

        # Convert beat wall-clock time to frame timeline
        beat_frame_timeline_time = last_beat_wallclock_time - wallclock_delta

        # Get beat properties
        beat_intensity = getattr(beat_state, "beat_intensity", 1.0)
        beat_confidence = getattr(beat_state, "confidence", 1.0)
        current_bpm = beat_state.current_bpm

        # === STEP 1: Check common rules first (always active) ===
        for trigger in self.common_triggers:
            if trigger.trigger_type != "beat":
                continue

            if self._check_beat_conditions(
                trigger, beat_intensity, beat_confidence, current_bpm
            ) and self._create_and_add_beat_effect(
                trigger, beat_frame_timeline_time, current_bpm, beat_intensity, beat_confidence, "COMMON"
            ):
                # Effect created successfully - mark beat as processed and stop
                self._last_beat_time_processed = last_beat_wallclock_time
                self.carousel_beat_count += 1
                self._check_carousel_rotation()
                return

        # === STEP 2: Check carousel rules (fallback, rotating) ===
        if self.carousel_rule_sets and len(self.carousel_rule_sets) > 0:
            # Get current active carousel rule set
            current_rule_set = self.carousel_rule_sets[self.carousel_current_index]

            for trigger in current_rule_set:
                if trigger.trigger_type != "beat":
                    continue

                carousel_name = f"CAROUSEL[{self.carousel_current_index}]"
                if self._check_beat_conditions(
                    trigger, beat_intensity, beat_confidence, current_bpm
                ) and self._create_and_add_beat_effect(
                    trigger, beat_frame_timeline_time, current_bpm, beat_intensity, beat_confidence, carousel_name
                ):
                    # Effect created successfully - mark beat as processed and stop
                    self._last_beat_time_processed = last_beat_wallclock_time
                    self.carousel_beat_count += 1
                    self._check_carousel_rotation()
                    return

        # No trigger matched - mark beat as processed anyway
        self._last_beat_time_processed = last_beat_wallclock_time
        self.carousel_beat_count += 1
        self._check_carousel_rotation()

    def _check_beat_conditions(
        self, trigger: EffectTriggerConfig, beat_intensity: float, beat_confidence: float, current_bpm: float
    ) -> bool:
        """Check if beat trigger conditions are met."""
        if trigger.confidence_min is not None and beat_confidence < trigger.confidence_min:
            return False
        if trigger.intensity_min is not None and beat_intensity < trigger.intensity_min:
            return False
        if trigger.bpm_min is not None and current_bpm < trigger.bpm_min:
            return False
        return not (trigger.bpm_max is not None and current_bpm > trigger.bpm_max)

    def _create_and_add_beat_effect(
        self,
        trigger: EffectTriggerConfig,
        beat_frame_timeline_time: float,
        current_bpm: float,
        beat_intensity: float,
        beat_confidence: float,
        source_label: str,
    ) -> bool:
        """Create and add beat effect. Returns True if successful, False otherwise."""
        try:
            effect = self._create_effect(
                trigger=trigger,
                start_time=beat_frame_timeline_time,
                bpm=current_bpm,
                beat_intensity=beat_intensity,
                beat_confidence=beat_confidence,
            )

            self.effect_manager.add_effect(effect)

            logger.info(
                f"🎵 Beat trigger matched ({source_label}): {trigger.effect_class}, "
                f"BPM={current_bpm:.1f}, intensity={beat_intensity:.2f}, "
                f"confidence={beat_confidence:.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create effect from beat trigger: {e}")
            return False

    def _check_carousel_rotation(self) -> None:
        """Check if it's time to rotate to next carousel rule set."""
        if not self.carousel_rule_sets or len(self.carousel_rule_sets) == 0:
            return

        if self.carousel_beat_count >= self.carousel_beat_interval:
            # Rotate to next rule set
            self.carousel_current_index = (self.carousel_current_index + 1) % len(self.carousel_rule_sets)
            self.carousel_beat_count = 0
            logger.info(
                f"🔄 Carousel rotated to rule set {self.carousel_current_index} / {len(self.carousel_rule_sets) - 1}"
            )

    def evaluate_test_triggers(self, frame_timeline_time: float) -> None:
        """
        Evaluate test triggers and create periodic effects.

        Args:
            frame_timeline_time: Current time on frame timeline
        """
        # Check if it's time to create a test effect
        time_since_last = frame_timeline_time - self._last_test_trigger_time

        if time_since_last < self.test_trigger_interval:
            return

        # Evaluate test triggers from common triggers (first match wins)
        for trigger in self.common_triggers:
            if trigger.trigger_type != "test":
                continue

            # Test triggers have no conditions - just create effect
            try:
                effect = self._create_effect(trigger=trigger, start_time=frame_timeline_time)

                self.effect_manager.add_effect(effect)

                logger.info(f"🧪 Test trigger fired: {trigger.effect_class}, " f"params={trigger.effect_params}")

                # Update last trigger time and stop (first match)
                self._last_test_trigger_time = frame_timeline_time
                return

            except Exception as e:
                logger.error(f"Failed to create effect from test trigger: {e}")
                continue

    def _create_effect(self, trigger: EffectTriggerConfig, start_time: float, **extra_params):
        """
        Create an effect instance from trigger configuration.

        Args:
            trigger: Trigger configuration
            start_time: Effect start time on frame timeline
            **extra_params: Additional parameters (BPM, beat_intensity, etc.)

        Returns:
            Effect instance
        """
        # Import led_effect module to get effect classes
        from . import led_effect

        # Merge trigger params with extra params (extra params override)
        params = {**trigger.effect_params, **extra_params}

        # Special handling for TemplateEffect - use factory for caching
        if trigger.effect_class == "TemplateEffect":
            # Extract template_path from params
            template_path = params.pop("template_path", None)
            if template_path is None:
                raise ValueError("TemplateEffect requires 'template_path' parameter")

            # For beat-triggered template effects, multiply intensity multipliers by beat_intensity
            # Check if this is a beat trigger by looking for beat_intensity in extra_params
            beat_intensity = extra_params.get("beat_intensity")

            if beat_intensity is not None:
                # Beat-triggered effect: convert multipliers to final values
                if "intensity_multiplier" in params:
                    # Convert intensity_multiplier to final intensity value
                    intensity_multiplier = params.pop("intensity_multiplier")
                    params["intensity"] = intensity_multiplier * beat_intensity
                    logger.debug(
                        f"TemplateEffect: intensity_multiplier={intensity_multiplier:.2f} * beat_intensity={beat_intensity:.2f} = {params['intensity']:.2f}"
                    )

                if "add_multiplier_factor" in params:
                    # Convert add_multiplier_factor to final add_multiplier value
                    add_multiplier_factor = params.pop("add_multiplier_factor")
                    params["add_multiplier"] = add_multiplier_factor * beat_intensity
                    logger.debug(
                        f"TemplateEffect: add_multiplier_factor={add_multiplier_factor:.2f} * beat_intensity={beat_intensity:.2f} = {params['add_multiplier']:.2f}"
                    )
            else:
                # Test trigger or other: use intensity/add_multiplier directly
                # Convert multiplier names to standard names if present (for consistency)
                if "intensity_multiplier" in params:
                    params["intensity"] = params.pop("intensity_multiplier")
                if "add_multiplier_factor" in params:
                    params["add_multiplier"] = params.pop("add_multiplier_factor")

            # Create using factory (with caching)
            effect = led_effect.TemplateEffectFactory.create_effect(
                template_path=template_path, start_time=start_time, **params
            )
        else:
            # Get effect class for other effect types
            effect_class = getattr(led_effect, trigger.effect_class, None)
            if effect_class is None:
                raise ValueError(f"Unknown effect class: {trigger.effect_class}")

            # Create effect instance directly
            effect = effect_class(start_time=start_time, **params)

        return effect


class FrameRenderer:
    """
    Timestamp-based frame renderer that handles precise timing for LED display.

    Features:
    - Establishes wallclock delta from first frame timestamp
    - Renders frames at their designated timestamps
    - Handles late/early frame timing
    - Supports multiple output targets (WLED, test renderer)
    - Converts LED values from spatial to physical order before output
    """

    def __init__(
        self,
        led_ordering: np.ndarray,
        first_frame_delay_ms: float = 100.0,
        timing_tolerance_ms: float = 5.0,
        late_frame_log_threshold_ms: float = 50.0,
        control_state=None,
        audio_beat_analyzer=None,
    ):
        """
        Initialize frame renderer.

        Args:
            first_frame_delay_ms: Default delay for first frame buffering
            timing_tolerance_ms: Acceptable timing deviation
            late_frame_log_threshold_ms: Log late frames above this threshold
            led_ordering: Array mapping spatial indices to physical LED IDs
            control_state: ControlState instance for audio reactive settings
            audio_beat_analyzer: AudioBeatAnalyzer instance for beat state access
        """
        self.first_frame_delay = first_frame_delay_ms / 1000.0
        self.timing_tolerance = timing_tolerance_ms / 1000.0
        self.late_frame_log_threshold = late_frame_log_threshold_ms / 1000.0

        # LED ordering for spatial to physical conversion
        self.led_ordering = led_ordering

        # Audio reactive components
        self._control_state = control_state
        self._audio_beat_analyzer = audio_beat_analyzer

        # Timing state
        self.wallclock_delta = None  # Established from first frame
        self.first_frame_received = False
        self.first_frame_timestamp = None

        # Pause time tracking
        self.is_paused = False
        self.pause_start_time = None
        self.total_pause_time = 0.0

        # Output sinks (multiple sink support)
        self.sinks = []  # List of registered sinks
        self.sink_names = {}  # Map sink instances to names for logging

        # Legacy compatibility - maintain individual references
        self.wled_sink: Optional[WLEDSink] = None
        self.test_sink: Optional[TestSink] = None
        self.enable_wled = True
        self.enable_test_sink = False

        # Statistics
        self.frames_rendered = 0
        self.late_frames = 0
        self.early_frames = 0
        self.on_time_frames = 0
        self.dropped_frames = 0  # For future frame dropping policy
        self.total_wait_time = 0.0
        self.total_late_time = 0.0
        self.start_time = time.time()

        # Beat detection state for creating brightness boost effects
        self._last_beat_time_processed = -1.0  # Track last beat we created an effect for
        self._beat_boost_logged = False

        # EWMA statistics for recent performance tracking
        self.ewma_alpha = 0.1  # EWMA smoothing factor
        self.ewma_frame_interval = 0.0  # EWMA of inter-frame interval (seconds)
        self.ewma_fps = 0.0  # Derived from ewma_frame_interval
        self.ewma_late_fraction = 0.0
        self.ewma_dropped_fraction = 0.0
        self.last_ewma_update = 0.0
        self.last_frame_timestamp = 0.0  # Last frame timestamp for interval calculation
        self.large_timestamp_gap_threshold = 2.0  # Log gaps larger than 2 seconds

        # Output FPS tracking (frames sent to sinks per second)
        self.output_fps_interval_ewma = 0.0  # EWMA of inter-frame interval for output FPS
        self.output_fps_ewma = 0.0  # Derived from output_fps_interval_ewma
        self._last_sink_call_time = 0.0  # Track sink call completion time for FPS calculation

        # Timing distribution tracking
        self.timing_errors = []  # Track last 100 timing errors for analysis
        self.max_timing_history = 100

        # Debug LED value writing (first 10 different frames)
        self._debug_led_count = 0
        self._debug_max_leds = 0
        self._debug_led_dir = Path("/tmp/prismatron_debug_leds")
        self._debug_led_dir.mkdir(exist_ok=True)
        self._debug_previous_led_values = None  # Track previous frame for uniqueness

        # Track error message timing to silence after 1 minute
        self._error_message_start_time = time.time()
        self._silent_after_minutes = 1.0

        # Track first frame timestamp for current playlist item (for playback position)
        self.current_item_first_frame_timestamp = None
        self.current_rendering_index = -1

        # Track if we need to adjust timeline for a new playlist item
        self._pending_timeline_adjustment = False

        # LED effects manager
        self.effect_manager = LedEffectManager()

        # Effect trigger manager (new framework)
        self.trigger_manager = EffectTriggerManager(self.effect_manager)

        # Template effect testing configuration - ENABLED BY DEFAULT
        self._test_template_effects = True  # Enable/disable template effect testing
        self._test_template_path = "templates/ring_800x480_leds.npy"
        self._test_template_interval = 2.0  # Create effect every 2 seconds
        self._test_template_duration = 1.0  # Effect duration in seconds
        self._test_template_blend_mode = "addboost"  # Blend mode for testing
        self._test_template_intensity = 2.0  # Effect intensity
        self._last_template_effect_time = 0.0  # Last time effect was created

        # Track last known trigger configuration hash to detect changes
        self._last_trigger_config_hash = None

        # Sparkle effect state for buildup integration
        self._sparkle_effect = None  # SparkleEffect instance or None
        self._last_cut_time_processed: float = 0.0
        self._last_drop_time_processed: float = 0.0

        # Cut fade effect state (separate tracking from sparkle's cut detection)
        self._last_cut_time_for_fade: float = 0.0
        self._cut_fade_effect = None  # Effect instance or None

        # Drop fade effect state (inverse fade from white back to content)
        self._last_drop_time_for_fade: float = 0.0
        self._drop_fade_effect = None  # Effect instance or None

        # Cut/drop effect configuration (loaded from control state)
        # Default configurations if not set in control state
        self._cut_effect_config: Optional[Dict[str, Any]] = {
            "enabled": True,
            "effect_class": "FadeInEffect",
            "params": {"duration": 2.0, "curve": "ease-in", "min_brightness": 0.0},
        }
        self._drop_effect_config: Optional[Dict[str, Any]] = {
            "enabled": True,
            "effect_class": "InverseFadeIn",
            "params": {"duration": 2.0, "curve": "ease-out"},
        }

        # Initialize triggers from control state (if available)
        self._initialize_triggers_from_control_state()

        logger.info(
            f"FrameRenderer initialized: delay={first_frame_delay_ms}ms, " f"tolerance=±{timing_tolerance_ms}ms"
        )

    def _initialize_triggers_from_control_state(self) -> None:
        """
        Initialize trigger configuration from ControlState.

        Loads trigger configuration from the new audio_reactive_trigger_config in ControlState.
        Falls back to legacy configuration for backward compatibility.
        """
        triggers = []
        test_interval = 2.0

        logger.info(f"Initializing triggers from control state (control_state={self._control_state is not None})")

        if self._control_state:
            try:
                status = self._control_state.get_status()
                if not status:
                    logger.warning("No control state status available")
                    # Create default trigger even without control state
                    triggers.append(
                        EffectTriggerConfig(
                            trigger_type="beat",
                            effect_class="BeatBrightnessEffect",
                            effect_params={
                                "boost_intensity": 4.0,
                                "duration_fraction": 0.4,
                            },
                            confidence_min=0.5,
                            intensity_min=None,
                            bpm_min=None,
                            bpm_max=None,
                        )
                    )
                    logger.info("Created default beat trigger (no control state)")
                    self.trigger_manager.set_triggers(triggers)
                    self.trigger_manager.set_test_interval(test_interval)
                    return

                # Try to load from new trigger configuration
                trigger_config = getattr(status, "audio_reactive_trigger_config", None)
                if trigger_config:
                    # Load from new configuration
                    test_interval = trigger_config.get("test_interval", 2.0)

                    # Check if we have new common + carousel structure
                    if "common_rules" in trigger_config or "carousel_rule_sets" in trigger_config:
                        # New structure: common rules + carousel rule sets
                        common_rules_data = trigger_config.get("common_rules", [])
                        carousel_sets_data = trigger_config.get("carousel_rule_sets", [])
                        carousel_beat_interval = trigger_config.get("carousel_beat_interval", 4)

                        # Parse common rules
                        common_triggers = []
                        for rule in common_rules_data:
                            try:
                                trigger = rule.get("trigger", {})
                                effect = rule.get("effect", {})

                                trigger_config_obj = EffectTriggerConfig(
                                    trigger_type=trigger.get("type", "beat"),
                                    effect_class=effect.get("class", "BeatBrightnessEffect"),
                                    effect_params=effect.get("params", {}),
                                    confidence_min=trigger.get("params", {}).get("confidence_min"),
                                    intensity_min=trigger.get("params", {}).get("intensity_min"),
                                    bpm_min=trigger.get("params", {}).get("bpm_min"),
                                    bpm_max=trigger.get("params", {}).get("bpm_max"),
                                )
                                common_triggers.append(trigger_config_obj)
                            except Exception as e:
                                logger.error(f"Failed to parse common rule: {e}")
                                continue

                        # Parse carousel rule sets
                        carousel_rule_sets = []
                        for rule_set_data in carousel_sets_data:
                            rule_set = []
                            for rule in rule_set_data.get("rules", []):
                                try:
                                    trigger = rule.get("trigger", {})
                                    effect = rule.get("effect", {})

                                    trigger_config_obj = EffectTriggerConfig(
                                        trigger_type=trigger.get("type", "beat"),
                                        effect_class=effect.get("class", "TemplateEffect"),
                                        effect_params=effect.get("params", {}),
                                        confidence_min=trigger.get("params", {}).get("confidence_min"),
                                        intensity_min=trigger.get("params", {}).get("intensity_min"),
                                        bpm_min=trigger.get("params", {}).get("bpm_min"),
                                        bpm_max=trigger.get("params", {}).get("bpm_max"),
                                    )
                                    rule_set.append(trigger_config_obj)
                                except Exception as e:
                                    logger.error(f"Failed to parse carousel rule: {e}")
                                    continue

                            if rule_set:  # Only add non-empty rule sets
                                carousel_rule_sets.append(rule_set)

                        # Use new common + carousel API
                        self.trigger_manager.set_common_and_carousel_triggers(
                            common_triggers, carousel_rule_sets, carousel_beat_interval
                        )
                        self.trigger_manager.set_test_interval(test_interval)

                        # Load cut/drop effect configuration
                        cut_effect = trigger_config.get("cut_effect")
                        drop_effect = trigger_config.get("drop_effect")

                        if cut_effect:
                            self._cut_effect_config = cut_effect
                            logger.info(
                                f"Loaded cut effect config: class={cut_effect.get('effect_class')}, "
                                f"enabled={cut_effect.get('enabled', True)}"
                            )

                        if drop_effect:
                            self._drop_effect_config = drop_effect
                            logger.info(
                                f"Loaded drop effect config: class={drop_effect.get('effect_class')}, "
                                f"enabled={drop_effect.get('enabled', True)}"
                            )

                        logger.info(
                            f"Loaded {len(common_triggers)} common rules, {len(carousel_rule_sets)} carousel sets"
                        )
                        return

                    # Backward compatibility: old flat rules list
                    rules = trigger_config.get("rules", [])
                    for rule in rules:
                        try:
                            # Extract trigger and effect configuration
                            trigger = rule.get("trigger", {})
                            effect = rule.get("effect", {})

                            # Create EffectTriggerConfig from rule
                            trigger_config_obj = EffectTriggerConfig(
                                trigger_type=trigger.get("type", "beat"),
                                effect_class=effect.get("class", "BeatBrightnessEffect"),
                                effect_params=effect.get("params", {}),
                                # Beat trigger conditions
                                confidence_min=trigger.get("params", {}).get("confidence_min"),
                                intensity_min=trigger.get("params", {}).get("intensity_min"),
                                bpm_min=trigger.get("params", {}).get("bpm_min"),
                                bpm_max=trigger.get("params", {}).get("bpm_max"),
                            )
                            triggers.append(trigger_config_obj)

                        except Exception as e:
                            logger.error(f"Failed to parse trigger rule: {e}, rule={rule}")
                            continue

                    logger.info(f"[Backward Compatibility] Loaded {len(triggers)} triggers from flat rules list")

                else:
                    # Fall back to legacy configuration for backward compatibility
                    logger.info("No new trigger configuration found, using legacy configuration")

                    # Add test trigger if template effects are enabled (legacy)
                    if self._test_template_effects:
                        triggers.append(
                            EffectTriggerConfig(
                                trigger_type="test",
                                effect_class="TemplateEffect",
                                effect_params={
                                    "template_path": self._test_template_path,
                                    "duration": self._test_template_duration,
                                    "blend_mode": self._test_template_blend_mode,
                                    "intensity": self._test_template_intensity,  # For test triggers, use intensity directly (no beat)
                                    "add_multiplier": 0.4,  # For test triggers, use add_multiplier directly
                                },
                            )
                        )

                    # Add beat trigger - always create default even if not explicitly enabled
                    # This allows testing before UI configuration is set
                    boost_intensity = getattr(status, "beat_brightness_intensity", 4.0)
                    boost_duration_fraction = getattr(status, "beat_brightness_duration", 0.4)
                    confidence_threshold = getattr(status, "beat_confidence_threshold", 0.5)

                    triggers.append(
                        EffectTriggerConfig(
                            trigger_type="beat",
                            effect_class="BeatBrightnessEffect",
                            effect_params={
                                "boost_intensity": boost_intensity,
                                "duration_fraction": boost_duration_fraction,
                            },
                            confidence_min=confidence_threshold,
                            intensity_min=None,
                            bpm_min=None,
                            bpm_max=None,
                        )
                    )

                    logger.info(
                        f"Auto-configured beat trigger (legacy/default): "
                        f"boost_intensity={boost_intensity:.2f}, "
                        f"confidence_min={confidence_threshold:.2f}"
                    )

                    test_interval = self._test_template_interval

            except Exception as e:
                logger.warning(f"Failed to load trigger configuration: {e}", exc_info=True)
        else:
            logger.warning("No control state available - cannot load triggers")

        # Set triggers in manager
        logger.info(f"Setting {len(triggers)} triggers in trigger manager")
        self.trigger_manager.set_triggers(triggers)
        self.trigger_manager.set_test_interval(test_interval)

        # Update configuration hash to track changes
        if self._control_state:
            status = self._control_state.get_status()
            if status:
                trigger_config = getattr(status, "audio_reactive_trigger_config", None)
                if trigger_config:
                    # Use JSON string hash for comparison
                    self._last_trigger_config_hash = hash(json.dumps(trigger_config, sort_keys=True))
                else:
                    self._last_trigger_config_hash = None

    def _check_for_trigger_config_updates(self) -> None:
        """
        Check if the trigger configuration has changed and reload if needed.

        This allows the renderer to pick up configuration changes from the UI
        without requiring a restart.
        """
        if not self._control_state:
            return

        try:
            status = self._control_state.get_status()
            if not status:
                return

            trigger_config = getattr(status, "audio_reactive_trigger_config", None)

            # Calculate current config hash
            current_hash = None
            if trigger_config:
                current_hash = hash(json.dumps(trigger_config, sort_keys=True))

            # Check if configuration has changed
            if current_hash != self._last_trigger_config_hash:
                logger.info(
                    f"Trigger configuration changed, reloading "
                    f"(old_hash={self._last_trigger_config_hash}, new_hash={current_hash})"
                )
                self._initialize_triggers_from_control_state()

        except Exception as e:
            logger.warning(f"Failed to check for trigger config updates: {e}")

    def _check_and_create_beat_brightness_effect(self, frame_timeline_time: float) -> None:
        """
        Check for new beats and create effects using trigger manager.

        This method uses the new trigger framework to evaluate beat conditions
        and create effects based on configured triggers.

        Args:
            frame_timeline_time: Current time on the frame timeline (from get_adjusted_wallclock_delta)
        """
        # DEBUG: Log once that method is being called
        if not hasattr(self, "_logged_beat_check_called"):
            self._logged_beat_check_called = True
            logger.info("_check_and_create_beat_brightness_effect is being called")

        # Check if audio reactive effects are enabled
        if not self._control_state or not self._audio_beat_analyzer:
            # Log once to help debug
            if not hasattr(self, "_logged_no_audio_components"):
                self._logged_no_audio_components = True
                logger.info(
                    f"Beat detection not active: control_state={self._control_state is not None}, "
                    f"audio_analyzer={self._audio_beat_analyzer is not None}"
                )
            return

        try:
            # Get current control state to check if audio reactive is enabled
            status = self._control_state.get_status()
            if not status:
                return

            if not status.audio_reactive_enabled:
                # Log once
                if not hasattr(self, "_logged_audio_reactive_disabled"):
                    self._logged_audio_reactive_disabled = True
                    logger.info("Beat detection: audio_reactive_enabled=False")
                return

            if not status.audio_enabled:
                # Log once
                if not hasattr(self, "_logged_audio_not_enabled"):
                    self._logged_audio_not_enabled = True
                    logger.info("Beat detection: audio_enabled=False (analyzer not running)")
                return

            # Get audio beat state
            beat_state = self._audio_beat_analyzer.get_current_state()
            if not beat_state or not beat_state.is_active:
                # Log once
                if not hasattr(self, "_logged_beat_state_inactive"):
                    self._logged_beat_state_inactive = True
                    logger.info(f"Beat detection: beat_state inactive (state={beat_state})")
                return

            if beat_state.current_bpm <= 0:
                # Log once per session
                if not hasattr(self, "_logged_no_bpm"):
                    self._logged_no_bpm = True
                    logger.info("Beat detection: No BPM detected yet")
                return

            # Clear the "no BPM" flag so we can log again if BPM drops to 0
            if hasattr(self, "_logged_no_bpm"):
                delattr(self, "_logged_no_bpm")

            # Get the most recent beat time from beat_state (wall-clock time)
            last_beat_wallclock_time = getattr(beat_state, "last_beat_wallclock_time", 0.0)

            # Use trigger manager to evaluate beat triggers
            self.trigger_manager.evaluate_beat_triggers(
                frame_timeline_time=frame_timeline_time,
                beat_state=beat_state,
                last_beat_wallclock_time=last_beat_wallclock_time,
                wallclock_delta=self.get_adjusted_wallclock_delta(),
            )

        except Exception as e:
            logger.warning(f"Error checking for beat brightness effects: {e}")

    def _calculate_beat_brightness_boost(self, current_time: float) -> float:
        """
        DEPRECATED: Legacy inline brightness boost implementation.

        This method is no longer used. Beat brightness boost is now handled by
        the BeatBrightnessEffect class through the LED effects framework.
        See _check_and_create_beat_brightness_effect() for the new implementation.

        Calculate brightness boost based on beat timing for audio-reactive effects.

        NOTE: This is the legacy inline implementation. Consider migrating to the new
        BeatBrightnessEffect class which creates effect instances per beat:

            from .led_effect import BeatBrightnessEffect

            # When beat detected, create effect using beat timestamp from audio timeline
            # (converted to frame timeline)
            effect = BeatBrightnessEffect(
                start_time=beat_frame_timestamp,  # Beat time on frame timeline
                bpm=beat_state.current_bpm,
                beat_intensity=beat_state.beat_intensity,
                boost_intensity=4.0,
                duration_fraction=0.4
            )
            renderer.add_led_effect(effect)

        Implements a configurable sine wave brightness boost during a portion of each beat interval.
        Formula: 1.0 + intensity * sin(t * pi / (duration * d)) where:
        - t = time since beat start
        - d = inter-beat duration (60.0 / BPM)
        - intensity = configurable boost intensity (0.0 to 1.0)
        - duration = configurable fraction of beat duration (0.1 to 1.0)

        Args:
            current_time: Current system time in seconds

        Returns:
            Brightness multiplier (1.0 = no boost, up to 2.0 = 100% boost)
        """
        # Check if audio reactive effects are enabled
        if not self._control_state or not self._audio_beat_analyzer:
            logger.debug("Beat brightness boost: No control state or audio analyzer")
            return 1.0

        try:
            # Get current control state to check if audio reactive is enabled
            status = self._control_state.get_status()
            if not status:
                logger.debug("Beat brightness boost: No status available")
                return 1.0

            if not status.audio_reactive_enabled:
                logger.debug("Beat brightness boost: Audio reactive not enabled")
                return 1.0

            if not status.audio_enabled:
                logger.debug("Beat brightness boost: Audio not enabled")
                return 1.0

            # Check if beat brightness boost is specifically enabled
            if not status.beat_brightness_enabled:
                logger.debug("Beat brightness boost: Beat brightness not enabled")
                return 1.0

            # Get configurable parameters with fallbacks
            boost_intensity = getattr(status, "beat_brightness_intensity", 4.0)  # Strong brightness boost (max 5.0)
            boost_duration_fraction = getattr(status, "beat_brightness_duration", 0.4)  # 400ms at 60 BPM
            confidence_threshold = getattr(status, "beat_confidence_threshold", 0.5)  # Ignore weak beats

            # Clamp parameters to safe ranges (intensity can now go up to 5x)
            boost_intensity = max(0.0, min(5.0, boost_intensity))
            boost_duration_fraction = max(0.1, min(1.0, boost_duration_fraction))
            confidence_threshold = max(0.0, min(1.0, confidence_threshold))

            # Log beat boost configuration once
            if not self._beat_boost_logged:
                logger.info(
                    f"🎵 Beat brightness boost enabled: intensity={boost_intensity:.2f}, "
                    f"duration={boost_duration_fraction:.2f}, confidence_threshold={confidence_threshold:.2f}"
                )
                self._beat_boost_logged = True

            # Get audio beat state
            beat_state = self._audio_beat_analyzer.get_current_state()
            if not beat_state or not beat_state.is_active:
                return 1.0

            # Calculate inter-beat duration from current BPM
            if beat_state.current_bpm <= 0:
                return 1.0
            beat_duration = 60.0 / beat_state.current_bpm

            # Calculate audio timeline time
            audio_time = current_time - self._audio_beat_analyzer.start_time

            # Get the most recent beat time from beat_state
            last_beat_time = beat_state.last_beat_time

            # Calculate time since the most recent beat
            t = audio_time - last_beat_time

            # Log beat state age and phase (DEBUG level)
            beat_state_age_ms = (current_time - self._audio_beat_analyzer.start_time - last_beat_time) * 1000
            beat_phase = (t / beat_duration) if beat_duration > 0 else 0
            logger.debug(
                f"Beat state read: BPM={beat_state.current_bpm:.1f}, "
                f"state_age={beat_state_age_ms:.1f}ms, "
                f"beat_phase={beat_phase:.2f}, "
                f"t={t:.3f}s"
            )

            # If we're past one full beat interval, we need to find which beat we're in
            if t > beat_duration:
                # Calculate how many beats have passed since last detected beat
                beats_passed = int(t / beat_duration)
                # Find the start of the current beat interval
                current_beat_start = last_beat_time + (beats_passed * beat_duration)
                # Recalculate t as time since current beat start
                t = audio_time - current_beat_start

            # Apply sine wave boost for configured duration of beat interval
            boost_duration = boost_duration_fraction * beat_duration
            if 0 <= t <= boost_duration:
                # Get beat intensity and confidence for dynamic boost
                # Use the beat_intensity and confidence from the beat state
                beat_intensity_value = getattr(beat_state, "beat_intensity", 1.0)
                beat_confidence = getattr(beat_state, "confidence", 1.0)

                # Apply confidence threshold - ignore weak beats
                if beat_confidence < confidence_threshold:
                    logger.debug(
                        f"Beat ignored: confidence {beat_confidence:.2f} < threshold {confidence_threshold:.2f}"
                    )
                    return 1.0

                # Calculate dynamic boost with improved intensity scaling
                # Use sqrt to expand the intensity range (0.1 -> 0.32, 0.4 -> 0.63)
                intensity_scaled = math.sqrt(beat_intensity_value)
                # Boost is: base * scaled_intensity * sine_wave (confidence already filtered)
                dynamic_boost_factor = boost_intensity * intensity_scaled
                boost = dynamic_boost_factor * math.sin(t * math.pi / boost_duration)
                multiplier = 1.0 + boost

                # Log boost calculation at DEBUG level
                logger.debug(
                    f"BRIGHTNESS_BOOST: multiplier={multiplier:.3f}, t={t:.3f}s, "
                    f"boost_intensity={boost_intensity:.2f}, intensity_raw={beat_intensity_value:.2f}, "
                    f"intensity_scaled={intensity_scaled:.3f}, confidence={beat_confidence:.2f}, "
                    f"dynamic_factor={dynamic_boost_factor:.3f}, boost={boost:.3f}"
                )
                return multiplier
            else:
                if np.random.random() < 0.01:  # Log 1% of non-boost events to reduce spam
                    logger.debug(
                        f"🎵 Beat brightness boost: No boost (t={t:.3f}s > duration={boost_duration:.3f}s), BPM={beat_state.current_bpm:.1f}"
                    )
                return 1.0  # No boost outside beat window

        except Exception as e:
            logger.warning(f"Error calculating beat brightness boost: {e}")
            return 1.0

    def register_sink(self, sink, name: str, enabled: bool = True) -> None:
        """
        Register a new output sink.

        Args:
            sink: Sink instance that must have a method to receive LED data
            name: Human-readable name for the sink
            enabled: Whether the sink is initially enabled
        """
        if hasattr(sink, "send_led_data") or hasattr(sink, "render_led_values"):
            self.sinks.append({"sink": sink, "name": name, "enabled": enabled, "failing": False})
            self.sink_names[sink] = name
            logger.info(f"Registered sink: {name} (enabled={enabled})")
        else:
            raise ValueError(f"Sink {name} must have 'send_led_data' or 'render_led_values' method")

    def unregister_sink(self, sink) -> None:
        """
        Unregister an output sink.

        Args:
            sink: Sink instance to remove
        """
        name = self.sink_names.get(sink, "Unknown")
        self.sinks = [s for s in self.sinks if s["sink"] != sink]
        if sink in self.sink_names:
            del self.sink_names[sink]
        logger.info(f"Unregistered sink: {name}")

    def set_sink_enabled(self, sink, enabled: bool) -> None:
        """
        Enable or disable a specific sink.

        Args:
            sink: Sink instance
            enabled: Whether to enable the sink
        """
        for s in self.sinks:
            if s["sink"] == sink:
                s["enabled"] = enabled
                name = self.sink_names.get(sink, "Unknown")
                logger.info(f"Set sink {name} enabled={enabled}")
                break

    def set_output_targets(
        self, wled_sink: Optional[WLEDSink] = None, test_sink: Optional[TestSink] = None, preview_sink: Optional = None
    ) -> None:
        """
        Set output targets for rendering (legacy compatibility method).

        Args:
            wled_sink: WLED sink for LED output
            test_sink: Test sink for debugging
            preview_sink: Preview sink for web interface
        """
        # Clear existing sinks
        self.sinks.clear()
        self.sink_names.clear()

        # Register provided sinks
        if wled_sink is not None:
            self.register_sink(wled_sink, "WLED", enabled=True)
        if test_sink is not None:
            self.register_sink(test_sink, "TestSink", enabled=False)
        if preview_sink is not None:
            self.register_sink(preview_sink, "PreviewSink", enabled=True)

        # Maintain legacy references
        self.wled_sink = wled_sink
        self.test_sink = test_sink
        self.enable_wled = wled_sink is not None
        self.enable_test_sink = test_sink is not None

        logger.info(
            f"Output targets: WLED={self.enable_wled}, TestSink={self.enable_test_sink}, Preview={preview_sink is not None}"
        )

    def set_wled_enabled(self, enabled: bool) -> None:
        """Enable or disable WLED output (legacy compatibility)."""
        self.enable_wled = enabled and (self.wled_sink is not None)
        if self.wled_sink is not None:
            self.set_sink_enabled(self.wled_sink, enabled)

    def set_test_sink_enabled(self, enabled: bool) -> None:
        """Enable or disable test sink output (legacy compatibility)."""
        self.enable_test_sink = enabled and (self.test_sink is not None)
        if self.test_sink is not None:
            self.set_sink_enabled(self.test_sink, enabled)

    def establish_wallclock_delta(self, first_timestamp: float) -> None:
        """
        Establish fixed delta between frame timestamps and wallclock time.

        Args:
            first_timestamp: Presentation timestamp of first frame
        """
        if self.first_frame_received:
            logger.warning("Wallclock delta already established")
            return

        current_wallclock = time.time()

        # Add default delay for buffering
        target_wallclock = current_wallclock + self.first_frame_delay

        # Calculate delta: wallclock_time = frame_timestamp + delta
        self.wallclock_delta = target_wallclock - first_timestamp
        self.first_frame_timestamp = first_timestamp
        self.first_frame_received = True

        logger.info(
            f"Established wallclock delta: {self.wallclock_delta:.3f}s "
            f"(first frame delay: {self.first_frame_delay:.3f}s)"
        )

    def render_frame_at_timestamp(
        self, led_values: np.ndarray, frame_timestamp: float, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Render frame at its designated timestamp with timing logic.

        Args:
            led_values: Optimized LED values to display, shape (led_count, 3)
            frame_timestamp: Original presentation timestamp from producer
            metadata: Optional frame metadata

        Returns:
            True if rendered successfully, False otherwise
        """
        # Use the metadata from producer to check if this is the first frame of a new item
        is_first_frame_of_new_item = metadata.get("is_first_frame_of_item", False) if metadata else False
        playlist_item_index = metadata.get("playlist_item_index", -1) if metadata else -1

        if is_first_frame_of_new_item and playlist_item_index >= 0:
            logger.info(f"RENDERER: First frame of new playlist item {playlist_item_index} detected")

        if not self.first_frame_received:
            self.establish_wallclock_delta(frame_timestamp)
        elif is_first_frame_of_new_item:
            # First frame of a new item - check if we need to adjust timeline
            current_wallclock = time.time()
            test_target = frame_timestamp + self.get_adjusted_wallclock_delta()
            lateness = current_wallclock - test_target

            if lateness > 0.05:  # If more than 50ms late
                # Adjust the wallclock delta to make this frame on-time
                logger.warning(f"🎬 Adjusting timeline for new playlist item - frame was {lateness*1000:.1f}ms late")
                self.wallclock_delta = current_wallclock - frame_timestamp
                # Reset pause time since we're resetting the timeline
                self.total_pause_time = 0.0
                logger.info(f"New wallclock delta: {self.wallclock_delta:.3f}s")

        # Calculate target wallclock time with pause compensation
        target_wallclock = frame_timestamp + self.get_adjusted_wallclock_delta()
        current_wallclock = time.time()

        # Time difference (negative = early, positive = late)
        time_diff = current_wallclock - target_wallclock

        # Track timing error for statistics
        self._track_timing_error(time_diff)

        # Debug logging for high FPS investigation
        if self.frames_rendered % 1 == 0:  # Log every frame
            waiting = f", waiting {-time_diff*1000:.1f}ms" if time_diff < -self.timing_tolerance else ""
            late = f", late {time_diff*1000:.1f}ms" if time_diff > self.timing_tolerance else ""
            logger.debug(
                f"Frame {self.frames_rendered}: timestamp={frame_timestamp:.3f}, "
                f"target_wall={target_wallclock:.3f}, current_wall={current_wallclock:.3f}, "
                f"time_diff={time_diff*1000:.1f}ms, ewma_fps={self.ewma_fps:.1f}"
                f"{waiting}{late}"
            )

        try:
            # Store current frame timestamp for use in _send_to_outputs
            self._current_frame_timestamp = frame_timestamp

            # Capture first frame timestamp for current item if needed
            if metadata and "playlist_item_index" in metadata and self.current_item_first_frame_timestamp is None:
                self.current_item_first_frame_timestamp = frame_timestamp
                logger.debug(
                    f"Captured first frame timestamp {frame_timestamp:.3f} for item {metadata['playlist_item_index']}"
                )

            if time_diff > self.timing_tolerance:
                # Late frame - render immediately
                self.late_frames += 1
                self.total_late_time += time_diff

                self._send_to_outputs(led_values, metadata)

            elif time_diff < -self.timing_tolerance:
                # Early frame - wait until target time
                wait_time = -time_diff
                self.early_frames += 1
                self.total_wait_time += wait_time

                time.sleep(wait_time)
                self._send_to_outputs(led_values, metadata)

            else:
                # On time - render immediately
                self.on_time_frames += 1
                self._send_to_outputs(led_values, metadata)

            self.frames_rendered += 1
            self._update_ewma_statistics(frame_timestamp)
            return True

        except Exception as e:
            logger.error(f"Error rendering frame: {e}")
            return False

    def _create_periodic_template_effect(self, frame_timeline_time: float) -> None:
        """
        Create template effects periodically using trigger manager.

        Args:
            frame_timeline_time: Current time on the frame timeline
        """
        if not self._test_template_effects:
            return

        # Use trigger manager to evaluate test triggers
        try:
            self.trigger_manager.evaluate_test_triggers(frame_timeline_time)
        except Exception as e:
            logger.error(f"Failed to evaluate test triggers: {e}")

    def _manage_sparkle_effect_for_buildup(self, frame_timeline_time: float) -> None:
        """
        Manage sparkle effect based on buildup detection state.

        Creates, updates, or removes sparkle effects based on:
        - Buildup intensity (0 = no sparkle, >0 = sparkle with intensity-based parameters)
        - Cut/drop events (remove sparkle on cut or drop)

        Args:
            frame_timeline_time: Current time on the frame timeline
        """
        # Import led_effect locally to avoid circular imports
        from . import led_effect

        # Check if audio beat analyzer is available
        if not self._audio_beat_analyzer:
            return

        try:
            # Get current audio state
            audio_state = self._audio_beat_analyzer.get_current_state()
            if not audio_state or not audio_state.is_active:
                return

            buildup_intensity = audio_state.buildup_intensity
            last_cut_time = audio_state.last_cut_time
            last_drop_time = audio_state.last_drop_time

            # Check for cut or drop events since last frame
            cut_occurred = last_cut_time > self._last_cut_time_processed
            drop_occurred = last_drop_time > self._last_drop_time_processed

            # Update processed times
            self._last_cut_time_processed = last_cut_time
            self._last_drop_time_processed = last_drop_time

            # Remove sparkle effect on cut or drop
            if cut_occurred or drop_occurred:
                if self._sparkle_effect is not None:
                    self.effect_manager.remove_effect(self._sparkle_effect)
                    self._sparkle_effect = None
                    event_type = "CUT" if cut_occurred else "DROP"
                    logger.info(f"🎆 Removed sparkle effect on {event_type}")
                return  # Don't create new sparkle on the same frame as cut/drop

            # If buildup intensity is zero, remove any existing sparkle effect
            if buildup_intensity <= 0:
                if self._sparkle_effect is not None:
                    self.effect_manager.remove_effect(self._sparkle_effect)
                    self._sparkle_effect = None
                    logger.debug("Removed sparkle effect (buildup intensity = 0)")
                return

            # Calculate sparkle parameters based on buildup intensity
            x = buildup_intensity

            # LED fraction: min(x/10, 1)
            density = min(x / 10.0, 1.0)

            # Sparkle interval: min(300, max(30, 30/x)) in milliseconds
            interval_ms = min(300.0, max(30.0, 30.0 / x))

            # Fade interval: 2x the sparkle interval
            fade_ms = 2.0 * interval_ms

            # Create or update sparkle effect
            if self._sparkle_effect is None:
                # Create new sparkle effect
                # Get LED count from the LED ordering array
                led_count = len(self.led_ordering)

                self._sparkle_effect = led_effect.SparkleEffect(
                    start_time=frame_timeline_time,
                    interval_ms=interval_ms,
                    fade_ms=fade_ms,
                    density=density,
                    led_count=led_count,
                )
                self.effect_manager.add_effect(self._sparkle_effect)
                logger.info(
                    f"🎆 Created sparkle effect: intensity={x:.2f}, density={density:.2%}, "
                    f"interval={interval_ms:.1f}ms, fade={fade_ms:.1f}ms"
                )
            else:
                # Update existing sparkle effect parameters
                self._sparkle_effect.set_parameters(
                    interval_ms=interval_ms,
                    fade_ms=fade_ms,
                    density=density,
                )
                logger.debug(
                    f"🎆 Updated sparkle effect: intensity={x:.2f}, density={density:.2%}, "
                    f"interval={interval_ms:.1f}ms, fade={fade_ms:.1f}ms"
                )

        except Exception as e:
            logger.error(f"Error managing sparkle effect for buildup: {e}")

    def _create_event_effect(self, effect_config: Dict[str, Any], frame_timeline_time: float):
        """
        Create an effect instance from event configuration.

        Args:
            effect_config: Configuration dict with effect_class and params
            frame_timeline_time: Current time on the frame timeline

        Returns:
            Effect instance or None if creation fails
        """
        # Import effect classes locally to avoid circular imports
        from . import led_effect, led_effect_transitions

        effect_class = effect_config.get("effect_class", "none")
        params = effect_config.get("params", {})

        if effect_class == "none":
            return None

        try:
            # Map effect class names to actual classes
            # From led_effect_transitions.py
            if effect_class == "FadeInEffect":
                return led_effect_transitions.FadeInEffect(
                    start_time=frame_timeline_time,
                    duration=params.get("duration", 2.0),
                    curve=params.get("curve", "ease-in"),
                    min_brightness=params.get("min_brightness", 0.0),
                )
            elif effect_class == "FadeOutEffect":
                return led_effect_transitions.FadeOutEffect(
                    start_time=frame_timeline_time,
                    duration=params.get("duration", 2.0),
                    curve=params.get("curve", "ease-out"),
                    min_brightness=params.get("min_brightness", 0.0),
                )
            elif effect_class == "RandomInEffect":
                return led_effect_transitions.RandomInEffect(
                    start_time=frame_timeline_time,
                    duration=params.get("duration", 2.0),
                    leds_per_frame=params.get("leds_per_frame", 10),
                    fade_tail=params.get("fade_tail", True),
                )
            elif effect_class == "RandomOutEffect":
                return led_effect_transitions.RandomOutEffect(
                    start_time=frame_timeline_time,
                    duration=params.get("duration", 2.0),
                    leds_per_frame=params.get("leds_per_frame", 10),
                    fade_tail=params.get("fade_tail", True),
                )
            # From led_effect.py
            elif effect_class == "InverseFadeIn":
                return led_effect.InverseFadeEffect(
                    start_time=frame_timeline_time,
                    duration=params.get("duration", 2.0),
                    direction="in",
                    curve=params.get("curve", "ease-out"),
                )
            elif effect_class == "InverseFadeOut":
                return led_effect.InverseFadeEffect(
                    start_time=frame_timeline_time,
                    duration=params.get("duration", 2.0),
                    direction="out",
                    curve=params.get("curve", "ease-in"),
                )
            elif effect_class == "TemplateEffect":
                template_path = params.get("template_path", "templates/ring_800x480_leds.npy")
                return led_effect.TemplateEffectFactory.create_effect(
                    template_path=template_path,
                    start_time=frame_timeline_time,
                    duration=params.get("duration", 1.0),
                    blend_mode=params.get("blend_mode", "addboost"),
                    intensity=params.get("intensity", 2.0),
                    add_multiplier=params.get("add_multiplier", 0.4),
                    color_thieving=params.get("color_thieving", False),
                )
            else:
                logger.warning(f"Unknown event effect class: {effect_class}")
                return None

        except Exception as e:
            logger.error(f"Failed to create event effect {effect_class}: {e}")
            return None

    def _manage_cut_fade_effect(self, frame_timeline_time: float) -> None:
        """
        Manage effect on cut events using configurable effect types.

        When a cut is detected, creates an effect based on the current
        cut_effect configuration. The effect is added last in the effects
        list to ensure it runs after other effects.

        Args:
            frame_timeline_time: Current time on the frame timeline
        """
        # Check if audio beat analyzer is available
        if not self._audio_beat_analyzer:
            return

        # Check if cut effect is enabled
        if not self._cut_effect_config or not self._cut_effect_config.get("enabled", True):
            return

        try:
            # Get current audio state
            audio_state = self._audio_beat_analyzer.get_current_state()
            if not audio_state or not audio_state.is_active:
                return

            last_cut_time = audio_state.last_cut_time

            # Check if a cut occurred since last frame
            if last_cut_time > self._last_cut_time_for_fade:
                self._last_cut_time_for_fade = last_cut_time

                # Remove any existing cut fade effect
                if self._cut_fade_effect is not None:
                    self.effect_manager.remove_effect(self._cut_fade_effect)
                    self._cut_fade_effect = None

                # Create effect from configuration
                self._cut_fade_effect = self._create_event_effect(self._cut_effect_config, frame_timeline_time)

                if self._cut_fade_effect is not None:
                    # Add to effect manager - it will be applied last since we add it last
                    self.effect_manager.add_effect(self._cut_fade_effect)
                    effect_class = self._cut_effect_config.get("effect_class", "unknown")
                    logger.info(f"🎬 Created cut effect ({effect_class}) at t={frame_timeline_time:.3f}s")

            # Clean up reference when effect completes
            if self._cut_fade_effect is not None and self._cut_fade_effect.is_complete(frame_timeline_time):
                self._cut_fade_effect = None

        except Exception as e:
            logger.error(f"Error managing cut fade effect: {e}")

    def _manage_drop_fade_effect(self, frame_timeline_time: float) -> None:
        """
        Manage effect on drop events using configurable effect types.

        When a drop is detected, creates an effect based on the current
        drop_effect configuration. By default, creates an inverse fade-in
        effect that starts with the panel at full white and fades back to
        the playing video for a dramatic "flash to white" effect.

        Args:
            frame_timeline_time: Current time on the frame timeline
        """
        # Check if audio beat analyzer is available
        if not self._audio_beat_analyzer:
            return

        # Check if drop effect is enabled
        if not self._drop_effect_config or not self._drop_effect_config.get("enabled", True):
            return

        try:
            # Get current audio state
            audio_state = self._audio_beat_analyzer.get_current_state()
            if not audio_state or not audio_state.is_active:
                return

            last_drop_time = audio_state.last_drop_time

            # Check if a drop occurred since last frame
            if last_drop_time > self._last_drop_time_for_fade:
                self._last_drop_time_for_fade = last_drop_time

                # Remove any existing drop fade effect
                if self._drop_fade_effect is not None:
                    self.effect_manager.remove_effect(self._drop_fade_effect)
                    self._drop_fade_effect = None

                # Create effect from configuration
                self._drop_fade_effect = self._create_event_effect(self._drop_effect_config, frame_timeline_time)

                if self._drop_fade_effect is not None:
                    # Add to effect manager
                    self.effect_manager.add_effect(self._drop_fade_effect)
                    effect_class = self._drop_effect_config.get("effect_class", "unknown")
                    logger.info(f"💥 Created drop effect ({effect_class}) at t={frame_timeline_time:.3f}s")

            # Clean up reference when effect completes
            if self._drop_fade_effect is not None and self._drop_fade_effect.is_complete(frame_timeline_time):
                self._drop_fade_effect = None

        except Exception as e:
            logger.error(f"Error managing drop fade effect: {e}")

    def enable_template_effect_testing(
        self,
        enabled: bool = True,
        template_path: str = "templates/ring_800x480_leds.npy",
        interval: float = 2.0,
        duration: float = 1.0,
        blend_mode: str = "add",
        intensity: float = 1.0,
    ) -> None:
        """
        Enable or disable periodic template effect testing.

        Args:
            enabled: Whether to enable template effect testing
            template_path: Path to template file
            interval: Time between effects in seconds
            duration: Effect duration in seconds
            blend_mode: Blend mode ("add", "alpha", "multiply", "replace", "boost")
            intensity: Effect intensity [0, 1+]
        """
        self._test_template_effects = enabled
        self._test_template_path = template_path
        self._test_template_interval = interval
        self._test_template_duration = duration
        self._test_template_blend_mode = blend_mode
        self._test_template_intensity = intensity

        # Update trigger manager configuration
        self._initialize_triggers_from_control_state()

        logger.info(
            f"Template effect testing {'enabled' if enabled else 'disabled'}: "
            f"path={template_path}, interval={interval}s, duration={duration}s, "
            f"blend={blend_mode}, intensity={intensity}"
        )

    def _send_to_outputs(self, led_values: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Send LED values to all enabled output sinks.

        Args:
            led_values: LED RGB values in spatial order, shape (led_count, 3)
            metadata: Optional frame metadata
        """
        # Apply LED effects (templates, animations, etc.) in SPATIAL order
        # Templates are created using spatial LED positions, so apply them before
        # converting to physical order
        current_wall_clock = time.time()
        frame_timeline_time = current_wall_clock - self.get_adjusted_wallclock_delta()

        # Check for trigger configuration updates (from UI changes)
        # Only check every ~100 frames to avoid overhead
        if self.frames_rendered % 100 == 0:
            self._check_for_trigger_config_updates()

        # Check for new beats and create brightness boost effects
        self._check_and_create_beat_brightness_effect(frame_timeline_time)

        # Create periodic template effects for testing (if enabled)
        self._create_periodic_template_effect(frame_timeline_time)

        # Manage sparkle effect based on buildup detection
        self._manage_sparkle_effect_for_buildup(frame_timeline_time)

        # Manage cut fade effect - fade from black on cut
        self._manage_cut_fade_effect(frame_timeline_time)

        # Manage drop fade effect - flash to white on drop, fade back to content
        self._manage_drop_fade_effect(frame_timeline_time)

        # Apply all active effects to spatial LED values (including beat brightness boost)
        self.effect_manager.apply_effects(led_values, frame_timeline_time)

        # Convert from spatial to physical order after applying effects
        physical_led_values = self._convert_spatial_to_physical(led_values)

        # Add rendering_index to metadata for PreviewSink
        enhanced_metadata = metadata.copy() if metadata else {}
        if metadata and "playlist_item_index" in metadata:
            enhanced_metadata["rendering_index"] = metadata["playlist_item_index"]

            # Track first frame timestamp when rendering index changes (for playback position calculation)
            new_rendering_index = metadata["playlist_item_index"]
            if new_rendering_index != self.current_rendering_index:
                self.current_rendering_index = new_rendering_index
                # Reset first frame timestamp for the new item
                self.current_item_first_frame_timestamp = None
                logger.debug(f"Rendering index changed to {new_rendering_index}, resetting first frame timestamp")

            # Add playback position if we have the first frame timestamp
            if self.current_item_first_frame_timestamp is not None and hasattr(self, "_current_frame_timestamp"):
                # Calculate playback position as difference from first frame timestamp for this item
                # This gets the frame_timestamp that was passed to render_frame_at_timestamp
                # We need to access it from the current frame being rendered
                playback_position = self._current_frame_timestamp - self.current_item_first_frame_timestamp
                enhanced_metadata["playback_position"] = max(0.0, playback_position)  # Ensure non-negative
                # Removed spammy playback position log

        # Debug: Write first 10 different LED value sets to temporary files for analysis
        if self._debug_led_count < self._debug_max_leds:
            try:
                # Check if this frame is different from the previous one
                is_different = True
                if self._debug_previous_led_values is not None:
                    # Compare with previous frame (use spatial values for comparison)
                    diff = np.abs(led_values.astype(np.float32) - self._debug_previous_led_values.astype(np.float32))
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)

                    # Consider frames different if max difference > 1 or mean difference > 0.1
                    is_different = max_diff > 1.0 or mean_diff > 0.1

                if is_different:
                    # Save both spatial and physical LED values for comparison
                    debug_spatial_file = self._debug_led_dir / f"led_spatial_{self._debug_led_count:03d}.npy"
                    debug_physical_file = self._debug_led_dir / f"led_physical_{self._debug_led_count:03d}.npy"
                    np.save(debug_spatial_file, led_values)
                    np.save(debug_physical_file, physical_led_values)
                    logger.info(
                        f"DEBUG: Wrote unique LED values {self._debug_led_count} to {debug_spatial_file} and {debug_physical_file}"
                    )

                    # Update previous frame and increment counter
                    self._debug_previous_led_values = led_values.copy()
                    self._debug_led_count += 1

            except Exception as e:
                logger.warning(f"DEBUG: Failed to write LED values {self._debug_led_count}: {e}")

        # Send to all registered sinks
        for sink_info in self.sinks:
            if not sink_info["enabled"]:
                continue

            sink = sink_info["sink"]
            name = sink_info["name"]

            try:
                # Try different sink interfaces
                if hasattr(sink, "send_led_data"):
                    # WLED-style sink
                    result = sink.send_led_data(physical_led_values)
                    if hasattr(result, "success") and not result.success:
                        if not sink_info["failing"]:
                            logger.warning(f"{name} transmission failed: {result.errors}")
                            sink_info["failing"] = True
                    else:
                        sink_info["failing"] = False
                elif hasattr(sink, "render_led_values"):
                    # Renderer-style sink
                    if hasattr(sink, "is_running") and not sink.is_running:
                        continue  # Skip if sink is not running
                    # Try to call with metadata (for preview sink), fall back to older signature if it fails
                    try:
                        # First try with metadata (preview sink needs this for playback position)
                        sink.render_led_values(physical_led_values.astype(np.uint8), enhanced_metadata)
                        if name == "PreviewSink" and "playback_position" in enhanced_metadata:
                            logger.debug(
                                f"Called PreviewSink with metadata containing playback_position={enhanced_metadata['playback_position']:.3f}"
                            )
                    except TypeError as e:
                        logger.debug(f"Sink {name} doesn't accept metadata parameter, trying basic signature: {e}")
                        # Fall back to old signature without metadata (for compatibility)
                        sink.render_led_values(physical_led_values.astype(np.uint8))
                else:
                    logger.warning(f"Sink {name} has no compatible interface")

            except Exception as e:
                logger.error(f"{name} sink error: {e}")

        # Measure time immediately after sending to all sinks for FPS calculation
        current_time = time.time()
        self._update_output_fps(current_time)

    def _track_timing_error(self, time_diff: float) -> None:
        """
        Track timing errors for statistical analysis.

        Args:
            time_diff: Timing difference in seconds (positive = late, negative = early)
        """
        self.timing_errors.append(time_diff)

        # Keep only recent history
        if len(self.timing_errors) > self.max_timing_history:
            self.timing_errors = self.timing_errors[-self.max_timing_history :]

    def _update_ewma_statistics(self, frame_timestamp: float) -> None:
        """
        Update EWMA-based statistics for recent performance tracking.

        Args:
            frame_timestamp: Current frame's presentation timestamp
        """
        current_time = time.time()

        # Log large timestamp gaps that might indicate transitions
        if self.last_frame_timestamp > 0:
            frame_interval = frame_timestamp - self.last_frame_timestamp
            if frame_interval > self.large_timestamp_gap_threshold:
                logger.warning(
                    f"Large frame timestamp gap detected: {frame_interval:.3f}s "
                    f"(previous: {self.last_frame_timestamp:.3f}, current: {frame_timestamp:.3f})"
                )

        # Calculate instantaneous values based on wall-clock render timing
        if self.last_ewma_update > 0:
            wall_clock_interval = current_time - self.last_ewma_update

            # Update EWMA of frame interval, then derive FPS
            if self.ewma_frame_interval == 0.0:
                self.ewma_frame_interval = wall_clock_interval
            else:
                self.ewma_frame_interval = (
                    1 - self.ewma_alpha
                ) * self.ewma_frame_interval + self.ewma_alpha * wall_clock_interval

            # Derive FPS from EWMA of interval
            self.ewma_fps = 1.0 / self.ewma_frame_interval if self.ewma_frame_interval > 0 else 0.0

        # Update EWMA fractions
        late_fraction = self.late_frames / max(1, self.frames_rendered)
        dropped_fraction = self.dropped_frames / max(1, self.frames_rendered)

        if self.frames_rendered == 1:
            # First frame, initialize EWMA
            self.ewma_late_fraction = late_fraction
            self.ewma_dropped_fraction = dropped_fraction
        else:
            # Update EWMA
            self.ewma_late_fraction = (1 - self.ewma_alpha) * self.ewma_late_fraction + self.ewma_alpha * late_fraction
            self.ewma_dropped_fraction = (
                1 - self.ewma_alpha
            ) * self.ewma_dropped_fraction + self.ewma_alpha * dropped_fraction

        self.last_ewma_update = current_time
        self.last_frame_timestamp = frame_timestamp

    def _update_output_fps(self, current_time: float, alpha: float = 0.1) -> None:
        """
        Update output FPS tracking based on wall-clock time between sink calls.
        This measures the actual rate at which frames are rendered to sinks.

        Uses EWMA of inter-frame intervals, then derives FPS as the reciprocal.

        Args:
            current_time: Time when sink calls completed (from time.time())
            alpha: EWMA smoothing factor
        """
        # Calculate interval since the PREVIOUS sink call completed
        if hasattr(self, "_last_sink_call_time") and self._last_sink_call_time > 0:
            time_diff = current_time - self._last_sink_call_time
            if time_diff > 0:
                # Update EWMA of inter-frame interval
                if not hasattr(self, "output_fps_interval_ewma") or self.output_fps_interval_ewma == 0.0:
                    self.output_fps_interval_ewma = time_diff
                else:
                    self.output_fps_interval_ewma = (1 - alpha) * self.output_fps_interval_ewma + alpha * time_diff

                # Derive FPS from EWMA of interval
                self.output_fps_ewma = 1.0 / self.output_fps_interval_ewma if self.output_fps_interval_ewma > 0 else 0.0

        # Store this time as the start of the next interval
        self._last_sink_call_time = current_time

        # Initialize if needed
        if not hasattr(self, "output_fps_ewma"):
            self.output_fps_ewma = 0.0
        if not hasattr(self, "output_fps_interval_ewma"):
            self.output_fps_interval_ewma = 0.0

    def get_output_fps(self) -> float:
        """Get the current output FPS (frames sent to sinks per second)."""
        return getattr(self, "output_fps_ewma", 0.0)

    def _convert_spatial_to_physical(self, led_values: np.ndarray) -> np.ndarray:
        """
        Convert LED values from spatial order to physical order using explicit element copying.

        Args:
            led_values: LED values in spatial order, shape (led_count, 3)

        Returns:
            LED values in physical order, shape (led_count, 3)
        """
        # LED ordering should have been validated at load time
        # led_ordering maps spatial_index -> physical_led_id
        # We want to place spatial_led_values[spatial_idx] at physical_led_values[physical_led_id]
        physical_led_values = np.zeros_like(led_values)

        # Use explicit element-by-element copy to ensure proper memory reordering
        # self.led_ordering[i] gives the physical LED ID for spatial index i
        for spatial_idx in range(len(led_values)):
            physical_led_id = self.led_ordering[spatial_idx]
            physical_led_values[physical_led_id] = led_values[spatial_idx].copy()

        return physical_led_values

    def get_timing_stats(self) -> Dict[str, Any]:
        """
        Get detailed timing statistics.

        Returns:
            Dictionary with timing statistics
        """
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frames_rendered / elapsed_time if elapsed_time > 0 else 0.0

        # Calculate timing error statistics
        timing_error_stats = {}
        if self.timing_errors:
            errors_ms = [err * 1000 for err in self.timing_errors]
            timing_error_stats = {
                "mean_error_ms": np.mean(errors_ms),
                "std_error_ms": np.std(errors_ms),
                "min_error_ms": np.min(errors_ms),
                "max_error_ms": np.max(errors_ms),
                "p95_error_ms": np.percentile(errors_ms, 95),
                "p99_error_ms": np.percentile(errors_ms, 99),
            }

        return {
            # Basic counts
            "frames_rendered": self.frames_rendered,
            "late_frames": self.late_frames,
            "early_frames": self.early_frames,
            "on_time_frames": self.on_time_frames,
            "dropped_frames": self.dropped_frames,
            # Timing statistics
            "avg_render_fps": avg_fps,
            "late_frame_percentage": (self.late_frames / max(1, self.frames_rendered)) * 100,
            "early_frame_percentage": (self.early_frames / max(1, self.frames_rendered)) * 100,
            "on_time_percentage": (self.on_time_frames / max(1, self.frames_rendered)) * 100,
            "dropped_frame_percentage": (self.dropped_frames / max(1, self.frames_rendered)) * 100,
            # EWMA statistics (recent performance)
            "ewma_fps": self.ewma_fps,
            "ewma_late_fraction": self.ewma_late_fraction,
            "ewma_dropped_fraction": self.ewma_dropped_fraction,
            "ewma_late_percentage": self.ewma_late_fraction * 100,
            "ewma_dropped_percentage": self.ewma_dropped_fraction * 100,
            "ewma_alpha": self.ewma_alpha,
            # Timing details
            "avg_wait_time_ms": (self.total_wait_time / max(1, self.early_frames)) * 1000,
            "avg_late_time_ms": (self.total_late_time / max(1, self.late_frames)) * 1000,
            "total_wait_time_s": self.total_wait_time,
            "total_late_time_s": self.total_late_time,
            # Configuration
            "first_frame_delay_ms": self.first_frame_delay * 1000,
            "timing_tolerance_ms": self.timing_tolerance * 1000,
            "wallclock_delta_s": self.wallclock_delta,
            "adjusted_wallclock_delta_s": self.get_adjusted_wallclock_delta(),
            "total_pause_time_s": self.total_pause_time,
            "is_paused": self.is_paused,
            "first_frame_received": self.first_frame_received,
            # Output sinks
            "registered_sinks": len(self.sinks),
            "enabled_sinks": sum(1 for s in self.sinks if s["enabled"]),
            "sink_names": [s["name"] for s in self.sinks if s["enabled"]],
            # Legacy compatibility
            "wled_enabled": self.enable_wled,
            "test_sink_enabled": self.enable_test_sink,
            # LED effects
            "led_effects": self.effect_manager.get_stats(),
            # Timing distribution
            **timing_error_stats,
        }

    def get_renderer_stats(self) -> Dict[str, Any]:
        """Get renderer statistics (alias for get_timing_stats)."""
        return self.get_timing_stats()

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.frames_rendered = 0
        self.late_frames = 0
        self.early_frames = 0
        self.on_time_frames = 0
        self.dropped_frames = 0
        self.total_wait_time = 0.0
        self.total_late_time = 0.0
        self.timing_errors.clear()
        self.start_time = time.time()

        # Reset EWMA statistics
        self.ewma_frame_interval = 0.0
        self.ewma_fps = 0.0
        self.ewma_late_fraction = 0.0
        self.ewma_dropped_fraction = 0.0
        self.last_ewma_update = 0.0
        self.last_frame_timestamp = 0.0

        # Reset output FPS tracking
        self.output_fps_interval_ewma = 0.0
        self.output_fps_ewma = 0.0
        self._last_sink_call_time = 0.0

        # Reset pause tracking
        self.is_paused = False
        self.pause_start_time = None
        self.total_pause_time = 0.0

        logger.debug("Renderer statistics reset")

    def set_timing_parameters(
        self,
        first_frame_delay_ms: Optional[float] = None,
        timing_tolerance_ms: Optional[float] = None,
        late_frame_log_threshold_ms: Optional[float] = None,
    ) -> None:
        """
        Update timing parameters.

        Args:
            first_frame_delay_ms: New first frame delay
            timing_tolerance_ms: New timing tolerance
            late_frame_log_threshold_ms: New late frame log threshold
        """
        if first_frame_delay_ms is not None:
            self.first_frame_delay = first_frame_delay_ms / 1000.0

        if timing_tolerance_ms is not None:
            self.timing_tolerance = timing_tolerance_ms / 1000.0

        if late_frame_log_threshold_ms is not None:
            self.late_frame_log_threshold = late_frame_log_threshold_ms / 1000.0

        logger.info(
            f"Timing parameters updated: delay={self.first_frame_delay*1000:.1f}ms, "
            f"tolerance=±{self.timing_tolerance*1000:.1f}ms, "
            f"log_threshold={self.late_frame_log_threshold*1000:.1f}ms"
        )

    def set_ewma_alpha(self, alpha: float) -> None:
        """
        Set the EWMA alpha parameter for recent statistics tracking.

        Args:
            alpha: EWMA alpha parameter (0 < alpha <= 1, smaller = more smoothing)
        """
        if not (0 < alpha <= 1):
            raise ValueError("EWMA alpha must be between 0 and 1")

        self.ewma_alpha = alpha
        logger.info(f"EWMA alpha set to {alpha:.3f}")

    def mark_frame_dropped(self) -> None:
        """
        Mark a frame as dropped (for future frame dropping policies).

        This method should be called when a frame is intentionally dropped
        due to timing constraints or buffer overruns.
        """
        self.dropped_frames += 1
        self._update_ewma_statistics()
        logger.debug(f"Frame marked as dropped (total: {self.dropped_frames})")

    def get_recent_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent performance using EWMA statistics.

        Returns:
            Dictionary with recent performance metrics
        """
        return {
            "recent_fps": self.ewma_fps,
            "recent_late_percentage": self.ewma_late_fraction * 100,
            "recent_dropped_percentage": self.ewma_dropped_fraction * 100,
            "ewma_alpha": self.ewma_alpha,
            "frames_rendered": self.frames_rendered,
            "is_performing_well": (
                self.ewma_fps > 25  # At least 25 FPS
                and self.ewma_late_fraction < 0.1  # Less than 10% late frames
                and self.ewma_dropped_fraction < 0.05  # Less than 5% dropped frames
            ),
        }

    def is_initialized(self) -> bool:
        """Check if renderer is initialized with timing delta."""
        return self.first_frame_received and self.wallclock_delta is not None

    def is_frame_late(
        self, frame_timestamp: float, late_threshold_ms: float = 50.0, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a frame is already late for rendering.

        Args:
            frame_timestamp: Presentation timestamp of the frame
            late_threshold_ms: Threshold in milliseconds for considering a frame late
            metadata: Optional frame metadata to check for first frame of new item

        Returns:
            True if frame is late and should be dropped, False if frame should be processed
        """
        if not self.is_initialized():
            # If renderer not initialized, we can't determine timing - process the frame
            return False

        # Use the metadata from producer to check if this is the first frame of a new item
        if metadata and metadata.get("is_first_frame_of_item", False):
            # First frame of new item - never consider it late (will adjust timeline instead)
            return False

        # Calculate target wallclock time with pause compensation
        target_wallclock = frame_timestamp + self.get_adjusted_wallclock_delta()
        current_wallclock = time.time()

        # Time difference (positive = late)
        time_diff = current_wallclock - target_wallclock
        late_threshold = late_threshold_ms / 1000.0

        return time_diff > late_threshold

    def pause_renderer(self) -> None:
        """
        Mark the renderer as paused and start tracking pause time.
        """
        if not self.is_paused:
            self.is_paused = True
            self.pause_start_time = time.time()
            logger.debug("Renderer paused, started tracking pause time")

    def resume_renderer(self) -> None:
        """
        Mark the renderer as resumed and add accumulated pause time to offset.
        """
        if self.is_paused and self.pause_start_time is not None:
            pause_duration = time.time() - self.pause_start_time
            self.total_pause_time += pause_duration
            self.is_paused = False
            self.pause_start_time = None
            logger.info(
                f"Renderer resumed, added {pause_duration:.3f}s pause time (total: {self.total_pause_time:.3f}s)"
            )

    def get_adjusted_wallclock_delta(self) -> float:
        """
        Get wallclock delta adjusted for pause time.

        Returns:
            Adjusted wallclock delta that accounts for time spent in pause
        """
        if self.wallclock_delta is None:
            return 0.0

        # Add total pause time to the delta to compensate for paused periods
        adjusted_delta = self.wallclock_delta + self.total_pause_time

        # If currently paused, also add the current pause duration
        if self.is_paused and self.pause_start_time is not None:
            current_pause_duration = time.time() - self.pause_start_time
            adjusted_delta += current_pause_duration

        return adjusted_delta

    def add_led_effect(self, effect) -> None:
        """
        Add a new LED effect to the active effects list.

        Args:
            effect: LedEffect instance to add

        Example:
            from .led_effect import TemplateEffect
            template = np.load('template_leds.npy')

            # Use frame timestamp (from current frame being rendered)
            # NOT wall-clock time!
            effect = TemplateEffect(
                start_time=current_frame_timestamp,  # From frame timeline
                template=template,
                duration=2.0,  # Effect plays over 2 seconds
                blend_mode='alpha',
                intensity=0.8
            )
            renderer.add_led_effect(effect)
        """
        self.effect_manager.add_effect(effect)

    def clear_led_effects(self) -> None:
        """Remove all active LED effects."""
        self.effect_manager.clear_effects()

    def get_active_effects_count(self) -> int:
        """Get the number of active LED effects."""
        return self.effect_manager.get_active_count()

    def get_led_effects_stats(self) -> Dict[str, Any]:
        """Get LED effects statistics."""
        return self.effect_manager.get_stats()
