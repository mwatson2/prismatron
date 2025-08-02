"""
Frame Drop Rate EWMA Tracker.

This module implements exponentially weighted moving average tracking for frame drop rates
using binary (0/1) samples, providing efficient recent performance monitoring.
"""

import logging

logger = logging.getLogger(__name__)


class FrameDropRateEwma:
    """
    Exponentially Weighted Moving Average tracker for frame drop rates.

    Tracks drop rates using binary samples (dropped=1, kept=0) with EWMA smoothing
    to provide recent performance indicators while being computationally efficient.

    Uses the same calculation pattern as the adaptive frame dropper for consistency.
    """

    def __init__(self, alpha: float = 0.1, name: str = "FrameDropRate"):
        """
        Initialize EWMA frame drop rate tracker.

        Args:
            alpha: EWMA smoothing factor (0 < alpha <= 1)
                  Smaller values = more smoothing, larger values = more responsive
            name: Human-readable name for logging purposes
        """
        if not (0 < alpha <= 1):
            raise ValueError("EWMA alpha must be between 0 and 1")

        self.alpha = alpha
        self.name = name

        # EWMA state
        self._ewma_rate = 0.0
        self._sample_count = 0

        # Statistics
        self._total_samples = 0
        self._total_drops = 0

        logger.debug(f"{self.name} EWMA tracker initialized with alpha={alpha:.3f}")

    def update(self, dropped: bool) -> None:
        """
        Update EWMA with a new frame drop sample.

        Args:
            dropped: True if frame was dropped, False if frame was kept
        """
        drop_sample = 1.0 if dropped else 0.0
        self._sample_count += 1
        self._total_samples += 1

        if dropped:
            self._total_drops += 1

        if self._sample_count == 1:
            # Initialize EWMA on first sample
            self._ewma_rate = drop_sample
        else:
            # Update EWMA using the same pattern as adaptive frame dropper
            self._ewma_rate = (1 - self.alpha) * self._ewma_rate + self.alpha * drop_sample

    def get_rate(self) -> float:
        """
        Get current EWMA drop rate.

        Returns:
            Current EWMA drop rate as a fraction (0.0 = no drops, 1.0 = all drops)
        """
        return self._ewma_rate

    def get_rate_percentage(self) -> float:
        """
        Get current EWMA drop rate as percentage.

        Returns:
            Current EWMA drop rate as percentage (0.0 = 0%, 100.0 = 100%)
        """
        return self._ewma_rate * 100.0

    def get_total_rate(self) -> float:
        """
        Get overall drop rate since initialization.

        Returns:
            Total drop rate as a fraction (0.0 = no drops, 1.0 = all drops)
        """
        if self._total_samples == 0:
            return 0.0
        return self._total_drops / self._total_samples

    def get_total_rate_percentage(self) -> float:
        """
        Get overall drop rate as percentage since initialization.

        Returns:
            Total drop rate as percentage (0.0 = 0%, 100.0 = 100%)
        """
        return self.get_total_rate() * 100.0

    def get_stats(self) -> dict:
        """
        Get comprehensive statistics.

        Returns:
            Dictionary with current statistics
        """
        return {
            "name": self.name,
            "ewma_rate": self._ewma_rate,
            "ewma_rate_percentage": self._ewma_rate * 100.0,
            "total_rate": self.get_total_rate(),
            "total_rate_percentage": self.get_total_rate_percentage(),
            "total_samples": self._total_samples,
            "total_drops": self._total_drops,
            "alpha": self.alpha,
        }

    def reset(self) -> None:
        """
        Reset all statistics and EWMA state.
        """
        old_total_samples = self._total_samples
        old_total_drops = self._total_drops

        self._ewma_rate = 0.0
        self._sample_count = 0
        self._total_samples = 0
        self._total_drops = 0

        logger.debug(
            f"{self.name} EWMA reset: was {old_total_drops}/{old_total_samples} "
            f"({(old_total_drops/max(1, old_total_samples))*100:.1f}%)"
        )

    def set_alpha(self, alpha: float) -> None:
        """
        Update EWMA alpha parameter.

        Args:
            alpha: New EWMA smoothing factor (0 < alpha <= 1)
        """
        if not (0 < alpha <= 1):
            raise ValueError("EWMA alpha must be between 0 and 1")

        old_alpha = self.alpha
        self.alpha = alpha

        logger.debug(f"{self.name} EWMA alpha updated: {old_alpha:.3f} -> {alpha:.3f}")

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.name}(ewma={self._ewma_rate:.3f}, "
            f"total={self._total_drops}/{self._total_samples}, "
            f"alpha={self.alpha:.3f})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"FrameDropRateEwma(name='{self.name}', alpha={self.alpha:.3f}, "
            f"ewma_rate={self._ewma_rate:.3f}, total_samples={self._total_samples}, "
            f"total_drops={self._total_drops})"
        )
