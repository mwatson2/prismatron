"""
Consumer Process Components.

This module contains components for the consumer process:
- LED position mapping and calibration
- LED optimization engine for image approximation
- WLED communication protocols
- Consumer process implementation
"""

from .consumer import ConsumerProcess, ConsumerStats
from .led_optimizer_dense import DenseLEDOptimizer, DenseOptimizationResult
from .wled_client import TransmissionResult, WLEDClient, WLEDConfig

__all__ = [
    "ConsumerProcess",
    "ConsumerStats",
    "DenseLEDOptimizer",
    "DenseOptimizationResult",
    "WLEDClient",
    "WLEDConfig",
    "TransmissionResult",
]
