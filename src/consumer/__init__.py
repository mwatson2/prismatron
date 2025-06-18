"""
Consumer Process Components.

This module contains components for the consumer process:
- LED position mapping and calibration
- LED optimization engine for image approximation
- WLED communication protocols
- Consumer process implementation
"""

from .led_mapper import LEDMapper, LEDPosition
from .led_optimizer import LEDOptimizer, OptimizationResult
from .wled_comm import TransmissionResult, WLEDCommunicator, WLEDProtocol

__all__ = [
    "LEDMapper",
    "LEDPosition",
    "LEDOptimizer",
    "OptimizationResult",
    "WLEDCommunicator",
    "WLEDProtocol",
    "TransmissionResult",
]
