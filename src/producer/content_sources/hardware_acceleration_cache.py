"""
Global Hardware Acceleration Detection Cache.

This module provides a singleton cache for hardware acceleration capabilities
to avoid repeated subprocess calls to ffmpeg during video source initialization.
"""

import logging
import subprocess
import threading
import time
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class HardwareAccelerationCache:
    """
    Singleton cache for hardware acceleration detection results.

    This cache is populated once when the first video source is created,
    and then reused for all subsequent video sources to avoid startup delays.
    """

    _instance: Optional["HardwareAccelerationCache"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "HardwareAccelerationCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = True

        # Hardware acceleration detection results
        self.hardware_acceleration: Optional[str] = None
        self.available_decoders: Set[str] = set()
        self.available_hwaccels: Set[str] = set()
        self.detection_completed = False

        logger.info("Hardware acceleration cache initialized")

    def ensure_detection_complete(self) -> None:
        """
        Ensure hardware acceleration detection has been performed.
        This method is thread-safe and will only run detection once.
        """
        if self.detection_completed:
            return

        with self._lock:
            if self.detection_completed:
                return

            logger.info("🔍 Performing one-time hardware acceleration detection...")
            start_time = time.time()

            try:
                self._detect_hardware_acceleration()
                detection_time = time.time() - start_time
                logger.info(f"✅ Hardware acceleration detection completed in {detection_time:.3f}s")
                logger.info(f"Hardware acceleration: {self.hardware_acceleration or 'None'}")
                logger.info(f"Available decoders: {len(self.available_decoders)} found")

            except Exception as e:
                logger.error(f"❌ Hardware acceleration detection failed: {e}")
                # Set safe defaults
                self.hardware_acceleration = None
                self.available_decoders = set()
                self.available_hwaccels = set()

            self.detection_completed = True

    def _detect_hardware_acceleration(self) -> None:
        """
        Perform the actual hardware acceleration detection.
        This is called only once and cached for all video sources.
        """
        try:
            # First check for Jetson NVMPI decoders (Jetson Orin Nano, etc.)
            result = subprocess.run(  # nosec B607 - ffmpeg is a trusted tool
                ["ffmpeg", "-hide_banner", "-decoders"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                decoders = result.stdout.lower()
                self.available_decoders = {
                    line.split()[1]
                    for line in decoders.split("\n")
                    if line.strip() and not line.startswith("---") and " " in line
                }

                # Check for Jetson-specific hardware decoders (NVMPI or NVV4L2DEC)
                if (
                    "h264_nvmpi" in decoders
                    or "hevc_nvmpi" in decoders
                    or "h264_nvv4l2dec" in decoders
                    or "hevc_nvv4l2dec" in decoders
                ):
                    # Determine which decoder variant is available
                    if "h264_nvv4l2dec" in decoders or "hevc_nvv4l2dec" in decoders:
                        self.hardware_acceleration = "nvv4l2dec"
                        logger.info("NVIDIA Jetson NVV4L2DEC hardware acceleration available")
                    else:
                        self.hardware_acceleration = "nvmpi"
                        logger.info("NVIDIA Jetson NVMPI hardware acceleration available")
                    return

            # Fall back to checking standard hardware acceleration
            result = subprocess.run(  # nosec B607 - ffmpeg is a trusted tool
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                hwaccels = result.stdout.lower()
                self.available_hwaccels = {
                    line.strip() for line in hwaccels.split("\n") if line.strip() and not line.startswith("hardware")
                }

                if "cuda" in hwaccels:
                    self.hardware_acceleration = "cuda"
                    logger.info("NVIDIA CUDA hardware acceleration available")
                elif "nvdec" in hwaccels:
                    self.hardware_acceleration = "nvdec"
                    logger.info("NVIDIA NVDEC hardware acceleration available")
                else:
                    logger.info("No hardware acceleration available")
            else:
                logger.warning("Could not detect standard hardware acceleration")

        except subprocess.TimeoutExpired:
            logger.warning("Hardware acceleration detection timed out")
        except Exception as e:
            logger.warning(f"Hardware acceleration detection failed: {e}")

    def get_hardware_acceleration(self) -> Optional[str]:
        """
        Get the detected hardware acceleration type.

        Returns:
            Hardware acceleration type or None if not available
        """
        self.ensure_detection_complete()
        return self.hardware_acceleration

    def get_decoder_for_codec(self, codec_name: str) -> Optional[str]:
        """
        Get the appropriate hardware decoder for a specific codec.

        Args:
            codec_name: Codec name (e.g., 'h264', 'hevc')

        Returns:
            Hardware decoder name or None if not available
        """
        self.ensure_detection_complete()

        if not self.hardware_acceleration:
            return None

        codec_lower = codec_name.lower()

        if self.hardware_acceleration in ["nvmpi", "nvv4l2dec"]:
            if "h264" in codec_lower or "avc" in codec_lower:
                decoder = f"h264_{self.hardware_acceleration}"
                return decoder if decoder in self.available_decoders else None
            elif "hevc" in codec_lower or "h265" in codec_lower:
                decoder = f"hevc_{self.hardware_acceleration}"
                return decoder if decoder in self.available_decoders else None

        return None

    def supports_decoder(self, decoder_name: str) -> bool:
        """
        Check if a specific decoder is available.

        Args:
            decoder_name: Name of the decoder to check

        Returns:
            True if decoder is available, False otherwise
        """
        self.ensure_detection_complete()
        return decoder_name in self.available_decoders


# Global singleton instance
_hardware_cache = HardwareAccelerationCache()


def get_hardware_acceleration_cache() -> HardwareAccelerationCache:
    """
    Get the global hardware acceleration cache instance.

    Returns:
        The singleton hardware acceleration cache
    """
    return _hardware_cache


def initialize_hardware_acceleration() -> None:
    """
    Initialize hardware acceleration detection proactively.
    This can be called during system startup to avoid delays later.
    """
    cache = get_hardware_acceleration_cache()
    cache.ensure_detection_complete()
    logger.info("Hardware acceleration cache pre-initialized")
