"""
Consumer Process Components.

This module contains components for the consumer process:
- LED position mapping and calibration
- LED optimization engine for image approximation
- WLED communication protocols
- Consumer process implementation
"""

from .consumer import ConsumerProcess, ConsumerStats
from .frame_renderer import FrameRenderer
from .led_buffer import LEDBuffer
from .led_effect import BeatBrightnessEffect, LedEffect, LedEffectManager, TemplateEffect
from .led_optimizer import LEDOptimizer, OptimizationResult
from .wled_client import TransmissionResult, WLEDClient, WLEDConfig

__all__ = [
    "ConsumerProcess",
    "ConsumerStats",
    "FrameRenderer",
    "LEDBuffer",
    "LEDOptimizer",
    "OptimizationResult",
    "WLEDClient",
    "WLEDConfig",
    "TransmissionResult",
    "LedEffect",
    "TemplateEffect",
    "BeatBrightnessEffect",
    "LedEffectManager",
]
