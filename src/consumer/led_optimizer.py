"""
LED Optimization Engine.

This module implements the core optimization algorithm that finds LED brightness
values to best approximate a target image using diffusion patterns. The optimization
solves: minimize ||Σ(weight_i * pattern_i) - target||² where weight_i = LED brightness.

Uses diffusion pattern database and GPU-accelerated matrix operations for
real-time performance targeting 15fps.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

from ..const import FRAME_HEIGHT, FRAME_WIDTH, LED_COUNT

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from LED optimization process."""

    led_values: np.ndarray  # RGB values for each LED (led_count, 3)
    error_metrics: Dict[str, float]  # Error metrics (mse, mae, etc.)
    optimization_time: float  # Time taken for optimization in seconds
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether optimization converged
    target_frame: Optional[np.ndarray] = None  # Original target frame (for debugging)

    def get_led_count(self) -> int:
        """Get number of LEDs in result."""
        return self.led_values.shape[0]

    def get_total_error(self) -> float:
        """Get total optimization error."""
        return self.error_metrics.get("mse", float("inf"))


class LEDOptimizer:
    """
    LED optimization engine for approximating images with LED array.

    Implements GPU-accelerated optimization to solve the inverse lighting
    problem: given a target image, find LED brightness values that best
    approximate the image as a weighted sum of diffusion patterns.

    Each LED has a diffusion pattern - a full-size image showing the light
    distribution it creates on the diffuser. The optimization finds the
    optimal weights (LED brightness) to minimize the difference between
    the weighted sum of patterns and the target image.
    """

    def __init__(
        self,
        diffusion_patterns_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize LED optimizer.

        Args:
            diffusion_patterns_path: Path to diffusion pattern database
            device: PyTorch device ('cpu', 'cuda', etc.)
        """
        self.diffusion_patterns_path = (
            diffusion_patterns_path or "config/diffusion_patterns.npz"
        )

        # Device selection
        if device is None:
            self.device = self._detect_device()
        else:
            self.device = device

        # Optimization parameters
        self.max_iterations = 100
        self.convergence_threshold = 1e-6
        self.learning_rate = 0.01
        self.regularization_weight = 0.001

        # State
        self._diffusion_patterns: Optional[
            torch.Tensor
        ] = None  # (led_count, height, width, 3)
        self._diffusion_patterns_loaded = False
        self._actual_led_count = (
            LED_COUNT  # Default to constant, updated when patterns loaded
        )

        # Statistics
        self._optimization_count = 0
        self._total_optimization_time = 0.0

    def _detect_device(self) -> str:
        """
        Detect best available compute device.

        Returns:
            Device string ('cuda', 'cpu', etc.)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using CPU-only optimization")
            return "cpu"

        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")

        return device

    def initialize(self) -> bool:
        """
        Initialize the LED optimizer.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if not TORCH_AVAILABLE:
                logger.error("PyTorch not available for optimization")
                return False

            # Load diffusion patterns
            if not self._load_diffusion_patterns():
                logger.error("Diffusion patterns not found")
                logger.error(
                    "Generate patterns first with: python tools/generate_synthetic_patterns.py"
                )
                return False

            logger.info(f"LED optimizer initialized with device: {self.device}")
            logger.info(f"Diffusion patterns shape: {self._diffusion_patterns.shape}")
            return True

        except Exception as e:
            logger.error(f"LED optimizer initialization failed: {e}")
            return False

    def _load_diffusion_patterns(self) -> bool:
        """
        Load diffusion pattern database.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not Path(self.diffusion_patterns_path).exists():
                return False

            # Load diffusion patterns from file
            data = np.load(self.diffusion_patterns_path)
            diffusion_patterns = data[
                "diffusion_patterns"
            ]  # Shape: (led_count, height, width, 3)

            # Use actual LED count from patterns file instead of hardcoded constant
            actual_led_count = diffusion_patterns.shape[0]
            if actual_led_count <= 0:
                logger.error(f"Invalid LED count in patterns: {actual_led_count}")
                return False

            logger.info(f"Using {actual_led_count} LEDs from patterns file")
            self._actual_led_count = actual_led_count  # Store actual LED count

            if diffusion_patterns.shape[1:3] != (FRAME_HEIGHT, FRAME_WIDTH):
                logger.error(
                    f"Diffusion pattern dimensions mismatch: "
                    f"{diffusion_patterns.shape[1:3]} != {(FRAME_HEIGHT, FRAME_WIDTH)}"
                )
                return False

            # Convert to tensor
            self._diffusion_patterns = torch.tensor(
                diffusion_patterns,
                dtype=torch.float32,
                device=self.device,
            )

            self._diffusion_patterns_loaded = True
            logger.info(
                f"Loaded diffusion patterns from {self.diffusion_patterns_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load diffusion patterns: {e}")
            return False

    def optimize_frame(
        self,
        target_frame: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Optimize LED values to approximate target frame using diffusion patterns.

        Args:
            target_frame: Target image (height, width, 3) in range [0, 255]
            initial_values: Initial LED values (led_count, 3), if None uses zeros
            max_iterations: Override default max iterations

        Returns:
            OptimizationResult with LED values and metrics
        """
        start_time = time.time()

        try:
            if not self._diffusion_patterns_loaded:
                raise RuntimeError("Diffusion patterns not loaded")

            # Validate input
            if target_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise ValueError(
                    f"Target frame shape {target_frame.shape} != {(FRAME_HEIGHT, FRAME_WIDTH, 3)}"
                )

            # Convert target to tensor
            target_tensor = torch.tensor(
                target_frame,
                dtype=torch.float32,
                device=self.device,
            )

            # Initialize LED values
            if initial_values is not None:
                led_values = torch.tensor(
                    initial_values,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=True,
                )
            else:
                led_values = torch.zeros(
                    (self._actual_led_count, 3),
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=True,
                )

            # Setup optimizer
            optimizer = torch.optim.Adam([led_values], lr=self.learning_rate)
            max_iters = max_iterations or self.max_iterations

            # Optimization loop
            converged = False
            prev_loss = float("inf")

            for iteration in range(max_iters):
                optimizer.zero_grad()

                # Forward pass: compute weighted sum of diffusion patterns
                # diffusion_patterns: (led_count, height, width, 3)
                # led_values: (led_count, 3)
                # We need to compute: Σ(led_values[i] * diffusion_patterns[i])

                # Reshape for broadcasting: led_values -> (led_count, 1, 1, 3)
                led_weights = led_values.unsqueeze(1).unsqueeze(1)

                # Element-wise multiply and sum over LED dimension
                reconstructed = torch.sum(self._diffusion_patterns * led_weights, dim=0)

                # Compute loss
                mse_loss = F.mse_loss(reconstructed, target_tensor)

                # Add regularization to prevent overly bright LEDs
                reg_loss = self.regularization_weight * torch.mean(led_values**2)

                total_loss = mse_loss + reg_loss

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Clamp LED values to valid range [0, 255]
                with torch.no_grad():
                    led_values.clamp_(0, 255)

                # Check convergence
                loss_change = abs(total_loss.item() - prev_loss)
                if loss_change < self.convergence_threshold:
                    converged = True
                    break

                prev_loss = total_loss.item()

            # Compute final metrics
            with torch.no_grad():
                led_weights = led_values.unsqueeze(1).unsqueeze(1)
                final_reconstructed = torch.sum(
                    self._diffusion_patterns * led_weights, dim=0
                )

                mse = F.mse_loss(final_reconstructed, target_tensor).item()
                mae = torch.mean(torch.abs(final_reconstructed - target_tensor)).item()
                max_error = torch.max(
                    torch.abs(final_reconstructed - target_tensor)
                ).item()

            # Create result
            optimization_time = time.time() - start_time

            result = OptimizationResult(
                led_values=led_values.detach().cpu().numpy(),
                error_metrics={
                    "mse": mse,
                    "mae": mae,
                    "max_error": max_error,
                    "rmse": np.sqrt(mse),
                },
                optimization_time=optimization_time,
                iterations=iteration + 1,
                converged=converged,
                target_frame=target_frame.copy(),
            )

            # Update statistics
            self._optimization_count += 1
            self._total_optimization_time += optimization_time

            logger.debug(
                f"Optimization completed in {optimization_time:.3f}s, "
                f"{iteration + 1} iterations, MSE: {mse:.2f}"
            )
            return result

        except Exception as e:
            optimization_time = time.time() - start_time
            logger.error(f"Optimization failed after {optimization_time:.3f}s: {e}")

            # Return error result
            return OptimizationResult(
                led_values=np.zeros((self._actual_led_count, 3), dtype=np.float32),
                error_metrics={
                    "mse": float("inf"),
                    "mae": float("inf"),
                    "max_error": float("inf"),
                },
                optimization_time=optimization_time,
                iterations=0,
                converged=False,
            )

    def set_optimization_parameters(
        self,
        max_iterations: Optional[int] = None,
        learning_rate: Optional[float] = None,
        convergence_threshold: Optional[float] = None,
        regularization_weight: Optional[float] = None,
    ) -> None:
        """
        Update optimization parameters.

        Args:
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization
            convergence_threshold: Convergence threshold
            regularization_weight: Regularization weight
        """
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if convergence_threshold is not None:
            self.convergence_threshold = convergence_threshold
        if regularization_weight is not None:
            self.regularization_weight = regularization_weight

        logger.info(
            f"Updated optimization parameters: max_iter={self.max_iterations}, "
            f"lr={self.learning_rate}"
        )

    def get_optimizer_stats(self) -> Dict[str, Any]:
        """
        Get optimizer statistics.

        Returns:
            Dictionary with optimizer statistics
        """
        avg_time = self._total_optimization_time / max(1, self._optimization_count)

        return {
            "device": self.device,
            "torch_available": TORCH_AVAILABLE,
            "diffusion_patterns_loaded": self._diffusion_patterns_loaded,
            "optimization_count": self._optimization_count,
            "total_optimization_time": self._total_optimization_time,
            "average_optimization_time": avg_time,
            "estimated_fps": 1.0 / avg_time if avg_time > 0 else 0.0,
            "parameters": {
                "max_iterations": self.max_iterations,
                "learning_rate": self.learning_rate,
                "convergence_threshold": self.convergence_threshold,
                "regularization_weight": self.regularization_weight,
            },
            "diffusion_patterns_shape": list(self._diffusion_patterns.shape)
            if self._diffusion_patterns is not None
            else None,
            "led_count": self._actual_led_count,
            "frame_dimensions": (FRAME_WIDTH, FRAME_HEIGHT),
        }

    def save_diffusion_patterns(
        self, patterns: np.ndarray, metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save diffusion patterns to file.

        Args:
            patterns: Diffusion patterns array (led_count, height, width, 3)
            metadata: Optional metadata dictionary

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure config directory exists
            config_dir = Path(self.diffusion_patterns_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data for saving
            save_data = {
                "diffusion_patterns": patterns,
                "metadata": metadata or {},
                "led_count": self._actual_led_count,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
            }

            # Save to file
            np.savez_compressed(self.diffusion_patterns_path, **save_data)

            logger.info(f"Saved diffusion patterns to {self.diffusion_patterns_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save diffusion patterns: {e}")
            return False
