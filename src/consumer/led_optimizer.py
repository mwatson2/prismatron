"""
LED Optimization Engine.

This module implements the core optimization algorithm that maps texture data
to LED brightness values by solving the optimization problem:
minimize ||A×x - target||² where x = LED brightness values

Uses diffusion pattern database and GPU-accelerated matrix operations for
real-time performance targeting 15fps.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from .led_mapper import LEDMapper

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
    approximate the image through the diffusion patterns.
    """

    def __init__(
        self,
        led_mapper: LEDMapper,
        diffusion_patterns_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize LED optimizer.

        Args:
            led_mapper: LED position mapper
            diffusion_patterns_path: Path to diffusion pattern database
            device: PyTorch device ('cpu', 'cuda', etc.)
        """
        self.led_mapper = led_mapper
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
        self._diffusion_matrix: Optional[torch.Tensor] = None
        self._diffusion_patterns_loaded = False
        self._led_positions_tensor: Optional[torch.Tensor] = None

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

            # Load LED positions
            if not self.led_mapper.led_positions:
                logger.error("LED mapper not initialized")
                return False

            # Convert LED positions to tensor
            self._prepare_led_positions()

            # Try to load diffusion patterns
            if not self._load_diffusion_patterns():
                logger.warning("Diffusion patterns not found, generating mock patterns")
                self._generate_mock_diffusion_patterns()

            logger.info(f"LED optimizer initialized with device: {self.device}")
            logger.info(f"Diffusion matrix shape: {self._diffusion_matrix.shape}")
            return True

        except Exception as e:
            logger.error(f"LED optimizer initialization failed: {e}")
            return False

    def _prepare_led_positions(self) -> None:
        """Prepare LED positions as tensors."""
        try:
            x_pos, y_pos = self.led_mapper.get_position_arrays()
            pixel_x, pixel_y = self.led_mapper.get_pixel_arrays()

            # Convert to tensors
            self._led_positions_tensor = torch.tensor(
                np.column_stack([x_pos, y_pos]), dtype=torch.float32, device=self.device
            )

            self._led_pixel_coords = torch.tensor(
                np.column_stack([pixel_x, pixel_y]),
                dtype=torch.long,
                device=self.device,
            )

            logger.debug(f"Prepared {len(x_pos)} LED positions as tensors")

        except Exception as e:
            logger.error(f"Failed to prepare LED positions: {e}")
            raise

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

            if diffusion_patterns.shape[0] != LED_COUNT:
                logger.error(
                    f"Diffusion patterns count mismatch: "
                    f"{diffusion_patterns.shape[0]} != {LED_COUNT}"
                )
                return False

            if diffusion_patterns.shape[1:3] != (FRAME_HEIGHT, FRAME_WIDTH):
                logger.error(
                    f"Diffusion pattern dimensions mismatch: "
                    f"{diffusion_patterns.shape[1:3]} != {(FRAME_HEIGHT, FRAME_WIDTH)}"
                )
                return False

            # Convert to tensor and reshape for matrix operations
            # Reshape to (num_pixels, led_count, 3) for matrix multiplication
            num_pixels = FRAME_HEIGHT * FRAME_WIDTH
            self._diffusion_matrix = torch.tensor(
                diffusion_patterns.transpose(1, 2, 0, 3).reshape(
                    num_pixels, LED_COUNT, 3
                ),
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

    def _generate_mock_diffusion_patterns(self) -> None:
        """Generate mock diffusion patterns for testing."""
        try:
            logger.info("Generating mock diffusion patterns...")

            num_pixels = FRAME_HEIGHT * FRAME_WIDTH

            # Create simple Gaussian diffusion patterns
            diffusion_matrix = np.zeros((num_pixels, LED_COUNT, 3), dtype=np.float32)

            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]
            y_coords = y_coords.flatten()
            x_coords = x_coords.flatten()

            # Generate patterns for each LED
            for led_idx, led in enumerate(self.led_mapper.led_positions):
                # Calculate Gaussian falloff from LED position
                dx = x_coords - led.pixel_x
                dy = y_coords - led.pixel_y
                distances_sq = dx * dx + dy * dy

                # Gaussian with different spread for each color channel
                sigma_base = 20.0  # Base spread in pixels

                for channel in range(3):
                    sigma = sigma_base * (0.8 + 0.4 * np.random.random())  # Vary spread
                    intensity = np.exp(-distances_sq / (2 * sigma * sigma))

                    # Normalize and scale
                    intensity = (
                        intensity / np.max(intensity)
                        if np.max(intensity) > 0
                        else intensity
                    )
                    diffusion_matrix[:, led_idx, channel] = intensity * 255.0

            # Convert to tensor
            self._diffusion_matrix = torch.tensor(
                diffusion_matrix, dtype=torch.float32, device=self.device
            )

            self._diffusion_patterns_loaded = True
            logger.info("Generated mock diffusion patterns successfully")

        except Exception as e:
            logger.error(f"Failed to generate mock diffusion patterns: {e}")
            raise

    def optimize_frame(
        self,
        target_frame: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Optimize LED values to approximate target frame.

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

            # Convert target to tensor and flatten
            target_tensor = torch.tensor(
                target_frame.reshape(-1, 3),  # (num_pixels, 3)
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
                    (LED_COUNT, 3),
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

                # Forward pass: compute reconstructed image
                # diffusion_matrix: (num_pixels, led_count, 3)
                # led_values: (led_count, 3)
                # reconstructed: (num_pixels, 3)
                reconstructed = torch.sum(
                    self._diffusion_matrix
                    * led_values.unsqueeze(0),  # Broadcast LED values
                    dim=1,
                )

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
                final_reconstructed = torch.sum(
                    self._diffusion_matrix * led_values.unsqueeze(0), dim=1
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
                led_values=np.zeros((LED_COUNT, 3), dtype=np.float32),
                error_metrics={
                    "mse": float("inf"),
                    "mae": float("inf"),
                    "max_error": float("inf"),
                },
                optimization_time=optimization_time,
                iterations=0,
                converged=False,
            )

    def sample_and_optimize(self, frame_array: np.ndarray) -> OptimizationResult:
        """
        Sample frame at LED positions and optimize (simple approach).

        This is a simplified approach that samples the frame at LED positions
        and uses those as target values, bypassing full diffusion modeling.

        Args:
            frame_array: Input frame (height, width, 3)

        Returns:
            OptimizationResult with sampled LED values
        """
        start_time = time.time()

        try:
            # Validate frame dimensions first
            if frame_array.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
                raise ValueError(
                    f"Frame shape {frame_array.shape[:2]} != expected {(FRAME_HEIGHT, FRAME_WIDTH)}"
                )

            # Sample colors at LED positions
            sampled_colors = self.led_mapper.sample_frame_at_leds(frame_array)

            # Check if sampling failed (all zeros might indicate error)
            if np.all(sampled_colors == 0):
                logger.warning("Frame sampling returned all zeros, possible error")

            # Simple metrics (no optimization error since we're just sampling)
            optimization_time = time.time() - start_time

            result = OptimizationResult(
                led_values=sampled_colors.astype(np.float32),
                error_metrics={
                    "mse": 0.0,  # No error for direct sampling
                    "mae": 0.0,
                    "max_error": 0.0,
                    "rmse": 0.0,
                },
                optimization_time=optimization_time,
                iterations=1,
                converged=True,
                target_frame=frame_array.copy(),
            )

            logger.debug(f"Frame sampling completed in {optimization_time:.3f}s")
            return result

        except Exception as e:
            optimization_time = time.time() - start_time
            logger.error(f"Frame sampling failed: {e}")

            return OptimizationResult(
                led_values=np.zeros((LED_COUNT, 3), dtype=np.float32),
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
            "diffusion_matrix_shape": list(self._diffusion_matrix.shape)
            if self._diffusion_matrix is not None
            else None,
            "led_count": LED_COUNT,
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
                "led_count": LED_COUNT,
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
