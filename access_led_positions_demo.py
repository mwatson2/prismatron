#!/usr/bin/env python3
"""
Demonstrate how to access LED positions in the Prismatron codebase.

This script shows how to:
1. Load LED positions from diffusion pattern files
2. Access LED positions through the LED optimizer
3. Understand the format and mapping of LED positions
"""

import sys
from pathlib import Path

import numpy as np

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def load_led_positions_directly(pattern_path: str):
    """
    Load LED positions directly from a diffusion pattern file.

    Args:
        pattern_path: Path to the .npz pattern file

    Returns:
        tuple: (led_positions, led_spatial_mapping, metadata)
    """
    print(f"Loading LED positions from: {pattern_path}")

    # Load the pattern file
    data = np.load(pattern_path, allow_pickle=True)

    # Extract LED positions - these are the physical (x,y) coordinates
    led_positions = data.get("led_positions", None)

    # Extract spatial mapping - maps LED indices to optimized ordering
    led_spatial_mapping = data.get("led_spatial_mapping", None)
    if led_spatial_mapping is not None and hasattr(led_spatial_mapping, "item"):
        led_spatial_mapping = led_spatial_mapping.item()

    # Extract metadata
    metadata = data.get("metadata", {})
    if hasattr(metadata, "item"):
        metadata = metadata.item()

    return led_positions, led_spatial_mapping, metadata


def access_led_positions_via_optimizer(pattern_path: str):
    """
    Access LED positions through the LED optimizer interface.

    Args:
        pattern_path: Path to the .npz pattern file
    """
    try:
        from consumer.led_optimizer import LEDOptimizer

        print("Accessing LED positions via LEDOptimizer...")

        # Initialize the optimizer with the pattern file
        optimizer = LEDOptimizer(diffusion_patterns_path=pattern_path)

        # Initialize to load the patterns
        if not optimizer.initialize():
            print("Failed to initialize optimizer")
            return None, None

        # Access the internal LED positions and mapping
        # Note: These are private attributes, so this demonstrates the internal structure
        led_positions = getattr(optimizer, "_led_positions", None)
        led_spatial_mapping = getattr(optimizer, "_led_spatial_mapping", None)

        return led_positions, led_spatial_mapping

    except Exception as e:
        print(f"Error accessing via optimizer: {e}")
        return None, None


def analyze_led_positions(led_positions, led_spatial_mapping, title="LED Positions"):
    """
    Analyze and display information about LED positions.

    Args:
        led_positions: Array of LED positions (N, 2)
        led_spatial_mapping: Dictionary mapping LED indices
        title: Title for the analysis
    """
    print(f"\n=== {title} ===")

    if led_positions is None:
        print("No LED positions available")
        return

    print(f"LED positions shape: {led_positions.shape}")
    print(f"LED positions dtype: {led_positions.dtype}")
    print(f"Number of LEDs: {len(led_positions)}")

    # Position ranges
    x_min, x_max = led_positions[:, 0].min(), led_positions[:, 0].max()
    y_min, y_max = led_positions[:, 1].min(), led_positions[:, 1].max()
    print(f"X range: {x_min} to {x_max}")
    print(f"Y range: {y_min} to {y_max}")

    # Sample positions
    print("First 5 LED positions:")
    for i in range(min(5, len(led_positions))):
        x, y = led_positions[i]
        print(f"  LED {i}: ({x}, {y})")

    # Spatial mapping info
    if led_spatial_mapping:
        print(f"\nSpatial mapping type: {type(led_spatial_mapping)}")
        if isinstance(led_spatial_mapping, dict):
            print(f"Mapping size: {len(led_spatial_mapping)} entries")
            print("Sample mapping entries (original_id -> optimized_id):")
            for i, (orig_id, opt_id) in enumerate(list(led_spatial_mapping.items())[:5]):
                print(f"  {orig_id} -> {opt_id}")
    else:
        print("No spatial mapping available")


def main():
    """Main demonstration function."""
    print("LED Position Access Demonstration")
    print("=" * 50)

    # Default pattern file
    pattern_file = "diffusion_patterns/synthetic_2624_uint8.npz"

    if not Path(pattern_file).exists():
        print(f"Pattern file not found: {pattern_file}")
        print("Please make sure you have generated patterns first.")
        return

    # Method 1: Direct access from file
    print("\n1. Loading LED positions directly from pattern file:")
    led_positions_direct, mapping_direct, metadata = load_led_positions_directly(pattern_file)
    analyze_led_positions(led_positions_direct, mapping_direct, "Direct File Access")

    # Print metadata info
    if metadata:
        print(f"\nMetadata: {metadata}")

    # Method 2: Access via optimizer (if possible)
    print("\n2. Accessing LED positions via LEDOptimizer:")
    led_positions_opt, mapping_opt = access_led_positions_via_optimizer(pattern_file)
    if led_positions_opt is not None:
        analyze_led_positions(led_positions_opt, mapping_opt, "Optimizer Access")

    # Demonstrate coordinate mapping
    if led_positions_direct is not None and mapping_direct:
        print("\n3. Coordinate mapping example:")
        print("How to get the physical position of an LED by its optimized index:")

        # Example: Get position of LED at optimized index 0
        optimized_index = 0
        # Find original LED ID that maps to this optimized index
        original_id = None
        for orig_id, opt_id in mapping_direct.items():
            if opt_id == optimized_index:
                original_id = orig_id
                break

        if original_id is not None:
            position = led_positions_direct[original_id]
            print(
                f"  Optimized index {optimized_index} -> Original LED {original_id} -> "
                f"Position ({position[0]}, {position[1]})"
            )

        # Example: Get position of original LED ID 0
        original_id = 0
        if original_id < len(led_positions_direct):
            position = led_positions_direct[original_id]
            optimized_index = mapping_direct.get(original_id, "unknown")
            print(
                f"  Original LED {original_id} -> Position ({position[0]}, {position[1]}) -> "
                f"Optimized index {optimized_index}"
            )


if __name__ == "__main__":
    main()
