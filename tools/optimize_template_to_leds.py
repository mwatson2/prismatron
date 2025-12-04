#!/usr/bin/env python3
"""
Optimize LED effect templates to LED patterns using the LED optimizer.

Takes a template file (frames, height, width) and converts each frame to
LED values (frames, leds) by:
1. Broadcasting single-channel template to RGB
2. Optimizing with LEDOptimizer
3. Averaging RGB LED outputs to single channel

Output is saved as a .npy file with shape (frames, leds).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.const import FRAME_HEIGHT, FRAME_WIDTH
from src.consumer.led_optimizer import LEDOptimizer


def load_config(config_path: Path) -> dict:
    """Load configuration from config.json."""
    with open(config_path, "r") as f:
        return json.load(f)


def optimize_template_to_leds(
    template_path: Path,
    output_path: Path,
    diffusion_patterns_path: str,
    max_iterations: int = 10,
    verbose: bool = True,
) -> np.ndarray:
    """
    Optimize template frames to LED patterns.

    Args:
        template_path: Path to template .npy file (frames, height, width)
        output_path: Path to save optimized LED patterns
        diffusion_patterns_path: Path to diffusion patterns file
        max_iterations: Maximum optimization iterations per frame
        verbose: Print progress information

    Returns:
        Optimized LED patterns array (frames, leds)
    """
    # Load template
    if verbose:
        print(f"Loading template from: {template_path}")
    template = np.load(template_path)

    if template.ndim != 3:
        raise ValueError(f"Expected 3D template (frames, height, width), got shape {template.shape}")

    num_frames, height, width = template.shape

    if verbose:
        print(f"Template shape: {template.shape}")
        print(f"Template dtype: {template.dtype}")
        print(f"Template range: [{template.min():.6f}, {template.max():.6f}]")

    # Validate dimensions match expected frame size
    if height != FRAME_HEIGHT or width != FRAME_WIDTH:
        raise ValueError(
            f"Template dimensions ({width}x{height}) do not match "
            f"expected frame size ({FRAME_WIDTH}x{FRAME_HEIGHT})"
        )

    # Initialize LED optimizer
    if verbose:
        print(f"\nInitializing LED optimizer...")
        print(f"Diffusion patterns: {diffusion_patterns_path}")

    optimizer = LEDOptimizer(
        diffusion_patterns_path=diffusion_patterns_path,
        enable_performance_timing=False,
        enable_batch_mode=False,
    )

    if not optimizer.initialize():
        raise RuntimeError("Failed to initialize LED optimizer")

    led_count = optimizer._actual_led_count

    if verbose:
        print(f"LED count: {led_count}")
        print(f"Optimization iterations: {max_iterations}")

    # Allocate output array
    optimized_leds = np.zeros((num_frames, led_count), dtype=np.float32)

    # Process each frame
    if verbose:
        print(f"\nOptimizing {num_frames} frames...")

    for frame_idx in range(num_frames):
        # Get single-channel frame
        frame_single = template[frame_idx]  # Shape: (height, width)

        # Convert to [0, 255] range
        frame_scaled = (frame_single * 255.0).astype(np.float32)

        # Broadcast to RGB by repeating across channels
        frame_rgb = np.stack([frame_scaled, frame_scaled, frame_scaled], axis=-1)  # Shape: (height, width, 3)

        # Validate shape
        assert frame_rgb.shape == (
            FRAME_HEIGHT,
            FRAME_WIDTH,
            3,
        ), f"Frame shape {frame_rgb.shape} != expected ({FRAME_HEIGHT}, {FRAME_WIDTH}, 3)"

        # Convert to uint8 explicitly
        frame_uint8 = frame_rgb.astype(np.uint8)

        # Optimize frame
        result = optimizer.optimize_frame(
            target_frame=frame_uint8,
            max_iterations=max_iterations,
            debug=False,
        )

        # Get LED values (led_count, 3) and average across channels
        led_values_rgb = result.led_values  # Shape: (led_count, 3), range [0, 1]

        # Transfer from GPU to CPU if needed
        if hasattr(led_values_rgb, "get"):
            led_values_rgb = led_values_rgb.get()

        led_values_single = np.mean(led_values_rgb, axis=1)  # Shape: (led_count,)

        # Store in output array
        optimized_leds[frame_idx] = led_values_single

        if verbose and (frame_idx + 1) % 10 == 0:
            progress = (frame_idx + 1) / num_frames * 100
            print(
                f"Progress: {frame_idx + 1}/{num_frames} ({progress:.1f}%) - "
                f"Frame {frame_idx}: {result.iterations} iters, "
                f"LED range [{led_values_single.min():.3f}, {led_values_single.max():.3f}]"
            )

    # Save optimized LED patterns
    if verbose:
        print(f"\nSaving optimized LED patterns to: {output_path}")

    np.save(output_path, optimized_leds)

    if verbose:
        print(f"Output shape: {optimized_leds.shape}")
        print(f"Output dtype: {optimized_leds.dtype}")
        print(f"Output range: [{optimized_leds.min():.6f}, {optimized_leds.max():.6f}]")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    return optimized_leds


def main():
    parser = argparse.ArgumentParser(
        description="Optimize LED effect templates to LED patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "template",
        type=str,
        help="Path to template .npy file (frames, height, width)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for optimized LED patterns (default: template_name_leds.npy)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Path to config.json file",
    )

    parser.add_argument(
        "--diffusion-patterns",
        type=str,
        default=None,
        help="Path to diffusion patterns file (overrides config.json)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum optimization iterations per frame",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    config = load_config(config_path)

    # Get diffusion patterns path
    if args.diffusion_patterns:
        diffusion_patterns = args.diffusion_patterns
    else:
        diffusion_patterns = config.get("diffusion_patterns")
        if not diffusion_patterns:
            print("Error: No diffusion_patterns specified in config.json")
            return 1

    # Resolve diffusion patterns path
    diffusion_patterns_path = Path(diffusion_patterns)
    if not diffusion_patterns_path.exists():
        print(f"Error: Diffusion patterns file not found: {diffusion_patterns_path}")
        return 1

    # Parse template path
    template_path = Path(args.template)
    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: add "_leds" suffix before extension
        output_path = template_path.parent / f"{template_path.stem}_leds.npy"

    # Run optimization
    try:
        optimize_template_to_leds(
            template_path=template_path,
            output_path=output_path,
            diffusion_patterns_path=str(diffusion_patterns_path),
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
        )
        return 0
    except Exception as e:
        print(f"Error optimizing template: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
