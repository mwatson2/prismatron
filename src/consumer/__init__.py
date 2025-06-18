"""
Consumer Process Components.

This module contains components for the consumer process:
- LED position mapping and calibration
- LED optimization engine for image approximation
- WLED communication protocols
- Consumer process implementation
"""

from .consumer import ConsumerProcess, ConsumerStats
from .led_mapper import LEDMapper, LEDPosition
from .led_optimizer import LEDOptimizer, OptimizationResult
from .wled_comm import TransmissionResult, WLEDCommunicator, WLEDProtocol

__all__ = [
    "ConsumerProcess",
    "ConsumerStats",
    "LEDMapper",
    "LEDPosition",
    "LEDOptimizer",
    "OptimizationResult",
    "WLEDCommunicator",
    "WLEDProtocol",
    "TransmissionResult",
]
