"""Visual effects system for LED display"""

from .base_effect import BaseEffect, EffectRegistry
from .color_effects import ColorBreathe, ColorWipe, GradientFlow, RainbowSweep
from .environmental_effects import AuroraBorealis, FireSimulation, Lightning
from .geometric_effects import Kaleidoscope, Mandala, RotatingShapes, Spirals
from .matrix_effects import BinaryStream, DigitalRain, GlitchArt
from .noise_effects import FractalNoise, PerlinNoiseFlow, SimplexClouds, VoronoiCells
from .particle_effects import Fireworks, RainSnow, Starfield, SwarmBehavior
from .wave_effects import LissajousCurves, PlasmaEffect, SineWaveVisualizer, WaterRipples

__all__ = [
    "BaseEffect",
    "EffectRegistry",
    # Geometric
    "RotatingShapes",
    "Kaleidoscope",
    "Spirals",
    "Mandala",
    # Particle
    "Fireworks",
    "Starfield",
    "RainSnow",
    "SwarmBehavior",
    # Wave
    "SineWaveVisualizer",
    "PlasmaEffect",
    "WaterRipples",
    "LissajousCurves",
    # Color
    "RainbowSweep",
    "ColorBreathe",
    "GradientFlow",
    "ColorWipe",
    # Noise
    "PerlinNoiseFlow",
    "SimplexClouds",
    "VoronoiCells",
    "FractalNoise",
    # Matrix
    "DigitalRain",
    "BinaryStream",
    "GlitchArt",
    # Environmental
    "FireSimulation",
    "Lightning",
    "AuroraBorealis",
]
