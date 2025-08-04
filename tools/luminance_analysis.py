#!/usr/bin/env python3
"""
Luminance Analysis Tool for LED Diffusion Patterns.

This tool loads diffusion patterns from a captured NPZ file and analyzes the luminance
distribution that would result from all LEDs at maximum brightness. It provides:

1. Constructs a full-brightness frame by summing all LED contributions
2. Converts RGB to luminance using standard photometric formula
3. Calculates and plots CDF of luminance values
4. Provides decile statistics for luminance distribution

The analysis helps understand the physical luminance characteristics and uniformity
of the LED display system when all LEDs are fully illuminated.

Usage:
    python luminance_analysis.py --input patterns.npz --output luminance_analysis.png
    python luminance_analysis.py --input patterns.npz --show-plot
    python luminance_analysis.py --input patterns.npz --save-to-npz
    python luminance_analysis.py --input patterns.npz --output plot.png --save-to-npz
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
except ImportError:
    # Fallback for systems without CUDA
    import numpy as cp

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.const import FRAME_HEIGHT, FRAME_WIDTH
from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


class LuminanceAnalyzer:
    """Analyzes luminance characteristics of LED diffusion patterns."""

    def __init__(self, patterns_file: str):
        """
        Initialize luminance analyzer.

        Args:
            patterns_file: Path to NPZ file containing diffusion patterns
        """
        self.patterns_file = patterns_file
        self.mixed_tensor = None
        self.led_count = 0
        self.metadata = None
        self.original_data = None  # Store original NPZ data for updating

    def load_patterns(self) -> bool:
        """
        Load diffusion patterns from NPZ file.

        Returns:
            True if loading successful, False otherwise
        """
        try:
            logger.info(f"Loading diffusion patterns from {self.patterns_file}")

            # Load NPZ file
            data = np.load(self.patterns_file, allow_pickle=True)
            self.original_data = dict(data)  # Store all original data

            # Extract metadata
            self.metadata = data["metadata"].item()
            self.led_count = self.metadata["led_count"]

            logger.info(f"Loaded patterns: {self.led_count} LEDs, {FRAME_WIDTH}x{FRAME_HEIGHT} frame")
            logger.info(f"Pattern type: {self.metadata.get('pattern_type', 'unknown')}")
            logger.info(f"Generator: {self.metadata.get('generator', 'unknown')}")

            # Load mixed tensor
            mixed_tensor_dict = data["mixed_tensor"].item()
            self.mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

            logger.info(
                f"Mixed tensor loaded: {self.mixed_tensor.batch_size} LEDs, "
                f"block_size={self.mixed_tensor.block_size}, "
                f"dtype={self.mixed_tensor.dtype}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            return False

    def construct_full_brightness_frame(self) -> np.ndarray:
        """
        Construct a frame with all LEDs at maximum brightness.

        Sums the contribution of each LED/channel block to create the combined
        illumination pattern that would result from all LEDs at full brightness.

        Returns:
            RGB frame (3, height, width) as fp32 with values scaled 0-1
        """
        logger.info("Constructing full-brightness frame by summing all LED contributions")

        # Initialize accumulator frame
        frame = np.zeros((3, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)

        processed_blocks = 0
        total_blocks = self.led_count * 3

        for led_id in range(self.led_count):
            for channel in range(3):
                try:
                    # Get block information
                    block_info = self.mixed_tensor.get_block_info(led_id, channel)
                    block = block_info["values"]
                    position = block_info["position"]

                    # Convert CuPy to NumPy if needed
                    if hasattr(block, "get"):  # CuPy array
                        block_np = cp.asnumpy(block)
                    else:
                        block_np = block

                    # Scale from uint8 [0,255] to float32 [0,1] if needed
                    if self.mixed_tensor.dtype == cp.uint8:
                        block_scaled = block_np.astype(np.float32) / 255.0
                    else:
                        block_scaled = block_np.astype(np.float32)

                    # Get position
                    top_row, left_col = position

                    # Ensure block fits within frame bounds
                    block_height, block_width = block_scaled.shape
                    end_row = min(top_row + block_height, FRAME_HEIGHT)
                    end_col = min(left_col + block_width, FRAME_WIDTH)

                    # Calculate actual region to copy
                    copy_height = end_row - top_row
                    copy_width = end_col - left_col

                    if copy_height > 0 and copy_width > 0:
                        # Add block contribution to the frame (no clipping)
                        frame[channel, top_row:end_row, left_col:end_col] += block_scaled[:copy_height, :copy_width]

                    processed_blocks += 1

                    if processed_blocks % 500 == 0:
                        progress = (processed_blocks / total_blocks) * 100
                        logger.info(f"Progress: {progress:.1f}% ({processed_blocks}/{total_blocks} blocks)")

                except Exception as e:
                    logger.warning(f"Failed to process LED {led_id}, channel {channel}: {e}")
                    continue

        logger.info(f"Completed frame construction: processed {processed_blocks}/{total_blocks} blocks")

        # Log frame statistics
        frame_stats = {"min": np.min(frame), "max": np.max(frame), "mean": np.mean(frame), "std": np.std(frame)}

        logger.info(
            f"Frame statistics: min={frame_stats['min']:.6f}, "
            f"max={frame_stats['max']:.6f}, "
            f"mean={frame_stats['mean']:.6f}, "
            f"std={frame_stats['std']:.6f}"
        )

        # Check for any extremely high values that might indicate overlapping
        high_value_count = np.sum(frame > 2.0)
        if high_value_count > 0:
            logger.warning(f"Found {high_value_count} pixels with values > 2.0 (possible LED overlap)")
            logger.warning(f"Peak value: {np.max(frame):.3f}")

        return frame

    def convert_to_luminance(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Convert RGB frame to luminance using standard photometric formula.

        Uses the ITU-R BT.709 standard luminance coefficients:
        Y = 0.2126*R + 0.7152*G + 0.0722*B

        Args:
            rgb_frame: RGB frame (3, height, width) as fp32

        Returns:
            Luminance frame (height, width) as fp32
        """
        logger.info("Converting RGB to luminance using ITU-R BT.709 coefficients")

        # ITU-R BT.709 luminance coefficients
        r_coeff = 0.2126
        g_coeff = 0.7152
        b_coeff = 0.0722

        # Extract RGB channels
        r_channel = rgb_frame[0]  # Red
        g_channel = rgb_frame[1]  # Green
        b_channel = rgb_frame[2]  # Blue

        # Calculate luminance
        luminance = r_coeff * r_channel + g_coeff * g_channel + b_coeff * b_channel

        # Log luminance statistics
        lum_stats = {
            "min": np.min(luminance),
            "max": np.max(luminance),
            "mean": np.mean(luminance),
            "std": np.std(luminance),
        }

        logger.info(
            f"Luminance statistics: min={lum_stats['min']:.6f}, "
            f"max={lum_stats['max']:.6f}, "
            f"mean={lum_stats['mean']:.6f}, "
            f"std={lum_stats['std']:.6f}"
        )

        return luminance

    def analyze_luminance_distribution(self, luminance: np.ndarray) -> dict:
        """
        Analyze luminance distribution and calculate statistics.

        Args:
            luminance: Luminance frame (height, width) as fp32

        Returns:
            Dictionary with distribution statistics
        """
        logger.info("Analyzing luminance distribution")

        # Flatten and sort luminance values
        flat_luminance = luminance.flatten()
        sorted_luminance = np.sort(flat_luminance)

        total_pixels = len(sorted_luminance)
        logger.info(f"Analyzing {total_pixels} pixels")

        # Calculate decile percentiles (10%, 20%, ..., 90%) plus peak
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        decile_values = np.percentile(sorted_luminance, percentiles)
        peak_value = np.max(sorted_luminance)

        # Prepare statistics dictionary
        stats = {
            "total_pixels": total_pixels,
            "min_luminance": np.min(sorted_luminance),
            "max_luminance": peak_value,
            "mean_luminance": np.mean(sorted_luminance),
            "std_luminance": np.std(sorted_luminance),
            "percentiles": percentiles,
            "decile_values": decile_values,
            "peak_value": peak_value,
            "sorted_luminance": sorted_luminance,
        }

        return stats

    def print_statistics_table(self, stats: dict):
        """
        Print formatted table of luminance statistics.

        Args:
            stats: Statistics dictionary from analyze_luminance_distribution
        """
        print("\n" + "=" * 60)
        print("LUMINANCE DISTRIBUTION ANALYSIS")
        print("=" * 60)
        print(f"Total pixels: {stats['total_pixels']:,}")
        print(f"LED count: {self.led_count}")
        print(f"Frame size: {FRAME_WIDTH}x{FRAME_HEIGHT}")
        print()

        print("Overall Statistics:")
        print(f"  Minimum:     {stats['min_luminance']:.6f}")
        print(f"  Maximum:     {stats['max_luminance']:.6f}")
        print(f"  Mean:        {stats['mean_luminance']:.6f}")
        print(f"  Std Dev:     {stats['std_luminance']:.6f}")
        print()

        print("Luminance Deciles:")
        print("  Percentile  |  Luminance Value")
        print("  ------------|----------------")

        for i, (percentile, value) in enumerate(zip(stats["percentiles"], stats["decile_values"])):
            print(f"      {percentile:2d}%    |    {value:.6f}")

        print(f"      Peak    |    {stats['peak_value']:.6f}")
        print()

        # Additional insights
        dynamic_range = stats["max_luminance"] / stats["min_luminance"] if stats["min_luminance"] > 0 else float("inf")
        print("Distribution Insights:")
        print(f"  Dynamic range:     {dynamic_range:.1f}:1")

        # Check uniformity (90th percentile / 10th percentile)
        if stats["decile_values"][0] > 0:  # 10th percentile
            uniformity_ratio = stats["decile_values"][-1] / stats["decile_values"][0]  # 90th / 10th
            print(f"  Uniformity ratio:  {uniformity_ratio:.1f}:1 (90th/10th percentile)")

        # Check how much of the image is "dark" (below 10% of peak)
        dark_threshold = stats["peak_value"] * 0.1
        dark_pixels = np.sum(stats["sorted_luminance"] < dark_threshold)
        dark_percentage = (dark_pixels / stats["total_pixels"]) * 100
        print(f"  Dark regions:      {dark_percentage:.1f}% (< 10% of peak)")

        print("=" * 60)

    def plot_luminance_cdf(self, stats: dict, output_path: str = None, show_plot: bool = False):
        """
        Plot cumulative distribution function of luminance values.

        Args:
            stats: Statistics dictionary from analyze_luminance_distribution
            output_path: Path to save plot (optional)
            show_plot: Whether to display plot interactively
        """
        logger.info("Plotting luminance CDF")

        sorted_luminance = stats["sorted_luminance"]
        total_pixels = len(sorted_luminance)

        # Create CDF: y-axis is cumulative probability (0-1)
        cdf_y = np.arange(1, total_pixels + 1) / total_pixels

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Full CDF
        ax1.plot(sorted_luminance, cdf_y, "b-", linewidth=1.5, alpha=0.8)
        ax1.set_xlabel("Luminance Value")
        ax1.set_ylabel("Cumulative Probability")
        ax1.set_title(f"Luminance CDF - Full Brightness Analysis ({self.led_count} LEDs)")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, stats["max_luminance"])
        ax1.set_ylim(0, 1)

        # Add decile markers
        for i, (percentile, value) in enumerate(zip(stats["percentiles"], stats["decile_values"])):
            ax1.axvline(x=value, color="red", linestyle="--", alpha=0.6, linewidth=1)
            ax1.text(value, 0.05 + (i % 3) * 0.1, f"{percentile}%", rotation=90, fontsize=8, ha="right", va="bottom")

        # Plot 2: Zoomed view (focus on main distribution, exclude extreme outliers)
        # Use 95th percentile as upper limit for zoom
        zoom_limit = np.percentile(sorted_luminance, 95)
        zoom_mask = sorted_luminance <= zoom_limit

        ax2.plot(sorted_luminance[zoom_mask], cdf_y[zoom_mask], "g-", linewidth=2, alpha=0.8)
        ax2.set_xlabel("Luminance Value")
        ax2.set_ylabel("Cumulative Probability")
        ax2.set_title("Luminance CDF - Zoomed View (up to 95th percentile)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, zoom_limit)
        ax2.set_ylim(0, 0.95)

        # Add decile markers to zoom plot
        for percentile, value in zip(stats["percentiles"][:8], stats["decile_values"][:8]):  # Up to 80%
            if value <= zoom_limit:
                ax2.axvline(x=value, color="red", linestyle="--", alpha=0.6, linewidth=1)
                ax2.text(value, percentile / 100 + 0.02, f"{percentile}%", fontsize=8, ha="center", va="bottom")

        # Add metadata text
        metadata_text = f"Pattern Type: {self.metadata.get('pattern_type', 'unknown')}\n"
        metadata_text += f"Generator: {self.metadata.get('generator', 'unknown')}\n"
        metadata_text += f"Frame: {FRAME_WIDTH}x{FRAME_HEIGHT}\n"
        metadata_text += f"Mean Luminance: {stats['mean_luminance']:.4f}\n"
        metadata_text += f"Peak Luminance: {stats['peak_value']:.4f}"

        ax1.text(
            0.02,
            0.98,
            metadata_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        # Save plot if requested
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved luminance CDF plot to {output_path}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    def save_analysis_to_npz(self, stats: dict, luminance_frame: np.ndarray, rgb_frame: np.ndarray) -> bool:
        """
        Save luminance analysis results back to the NPZ file.

        Args:
            stats: Statistics dictionary from analyze_luminance_distribution
            luminance_frame: Luminance frame (height, width) as fp32
            rgb_frame: Full-brightness RGB frame (3, height, width) as fp32

        Returns:
            True if save successful, False otherwise
        """
        try:
            logger.info(f"Saving luminance analysis results to {self.patterns_file}")

            # Create luminance analysis results dictionary
            luminance_analysis = {
                "analysis_timestamp": time.time(),
                "analysis_version": "1.0",
                "led_count": self.led_count,
                "frame_size": [FRAME_WIDTH, FRAME_HEIGHT],
                "total_pixels": stats["total_pixels"],
                # Overall statistics
                "min_luminance": float(stats["min_luminance"]),
                "max_luminance": float(stats["max_luminance"]),
                "mean_luminance": float(stats["mean_luminance"]),
                "std_luminance": float(stats["std_luminance"]),
                # Decile percentiles
                "percentiles": stats["percentiles"],
                "decile_values": stats["decile_values"].tolist(),
                "peak_value": float(stats["peak_value"]),
                # Distribution insights
                "dynamic_range": (
                    float(stats["max_luminance"] / stats["min_luminance"])
                    if stats["min_luminance"] > 0
                    else float("inf")
                ),
                "uniformity_ratio": (
                    float(stats["decile_values"][-1] / stats["decile_values"][0])
                    if stats["decile_values"][0] > 0
                    else float("inf")
                ),
                # Dark region analysis
                "dark_threshold": float(stats["peak_value"] * 0.1),
                "dark_pixels_count": int(np.sum(stats["sorted_luminance"] < stats["peak_value"] * 0.1)),
                "dark_pixels_percentage": float(
                    (np.sum(stats["sorted_luminance"] < stats["peak_value"] * 0.1) / stats["total_pixels"]) * 100
                ),
                # RGB frame statistics
                "rgb_frame_min": float(np.min(rgb_frame)),
                "rgb_frame_max": float(np.max(rgb_frame)),
                "rgb_frame_mean": float(np.mean(rgb_frame)),
                "rgb_frame_std": float(np.std(rgb_frame)),
                # Overlap detection
                "high_value_pixels_count": int(np.sum(rgb_frame > 2.0)),
                "high_value_pixels_percentage": float((np.sum(rgb_frame > 2.0) / (3 * stats["total_pixels"])) * 100),
                # Compressed data arrays (for storage efficiency)
                "luminance_percentiles_1_to_99": np.percentile(stats["sorted_luminance"], np.arange(1, 100)).astype(
                    np.float32
                ),
            }

            # Optional: Store downsampled luminance frame for visualization (every 4th pixel)
            downsampled_luminance = luminance_frame[::4, ::4].astype(np.float32)
            luminance_analysis["downsampled_luminance_frame"] = downsampled_luminance
            luminance_analysis["downsampled_frame_size"] = list(downsampled_luminance.shape)

            # Update the original data dictionary
            updated_data = dict(self.original_data)
            updated_data["luminance_analysis"] = luminance_analysis

            # Create backup of original file
            backup_path = Path(self.patterns_file).with_suffix(".npz.backup")
            if not backup_path.exists():
                import shutil

                shutil.copy2(self.patterns_file, backup_path)
                logger.info(f"Created backup: {backup_path}")

            # Save updated NPZ file
            np.savez_compressed(self.patterns_file, **updated_data)

            # Calculate file size change
            file_size_mb = Path(self.patterns_file).stat().st_size / (1024 * 1024)

            logger.info("Successfully updated NPZ file with luminance analysis")
            logger.info(f"Updated file size: {file_size_mb:.1f} MB")
            logger.info(
                f"Analysis includes: statistics, percentiles, and {downsampled_luminance.shape} downsampled frame"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to save analysis to NPZ file: {e}")
            return False

    def run_analysis(self, output_path: str = None, show_plot: bool = False, save_to_npz: bool = False) -> bool:
        """
        Run complete luminance analysis.

        Args:
            output_path: Path to save plot (optional)
            show_plot: Whether to display plot interactively
            save_to_npz: Whether to save analysis results back to NPZ file

        Returns:
            True if analysis successful, False otherwise
        """
        try:
            # Load patterns
            if not self.load_patterns():
                return False

            # Construct full brightness frame
            rgb_frame = self.construct_full_brightness_frame()

            # Convert to luminance
            luminance = self.convert_to_luminance(rgb_frame)

            # Analyze distribution
            stats = self.analyze_luminance_distribution(luminance)

            # Print statistics table
            self.print_statistics_table(stats)

            # Plot CDF
            self.plot_luminance_cdf(stats, output_path, show_plot)

            # Save analysis results to NPZ file if requested
            if save_to_npz:
                if not self.save_analysis_to_npz(stats, luminance, rgb_frame):
                    logger.warning("Failed to save analysis to NPZ file, but analysis completed")

            logger.info("Luminance analysis completed successfully")
            return True

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze luminance distribution of LED diffusion patterns")

    parser.add_argument("--input", required=True, help="Input NPZ file with diffusion patterns")
    parser.add_argument("--output", help="Output PNG file for luminance CDF plot")
    parser.add_argument("--show-plot", action="store_true", help="Display plot interactively")
    parser.add_argument(
        "--save-to-npz", action="store_true", help="Save analysis results back to input NPZ file (creates backup)"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return 1

    if input_path.suffix != ".npz":
        logger.error("Input file must have .npz extension")
        return 1

    # Validate output path if provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix not in [".png", ".jpg", ".pdf", ".svg"]:
            logger.warning("Output file should have image extension (.png, .jpg, .pdf, .svg)")

    try:
        # Create analyzer
        analyzer = LuminanceAnalyzer(str(input_path))

        # Run analysis
        if not analyzer.run_analysis(args.output, args.show_plot, args.save_to_npz):
            logger.error("Analysis failed")
            return 1

        logger.info("Analysis completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
