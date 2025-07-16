#!/usr/bin/env python3
"""
Debug Frame Analysis Tool.

This tool analyzes the debug frames and LED values saved by the consumer
and renderer to identify where the preview display issue occurs.

Features:
1. Convert source frames to PNG images for visual inspection
2. Use the diffusion pattern matrix A to reconstruct images from LED values
3. Compare source, optimized LED, and reconstructed images
4. Identify where in the pipeline the issue occurs

Usage:
    python debug_frame_analysis.py --patterns patterns.npz
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available - image output will be limited")

try:
    import cupy as cp
except ImportError:
    # Fallback for systems without CUDA
    import numpy as cp

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


class DebugFrameAnalyzer:
    """Analyzes debug frames and LED values to identify pipeline issues."""

    def __init__(self, patterns_file: str):
        """
        Initialize analyzer with diffusion patterns.

        Args:
            patterns_file: Path to diffusion patterns NPZ file
        """
        self.patterns_file = patterns_file
        self.mixed_tensor: Optional[SingleBlockMixedSparseTensor] = None
        self.metadata: Dict = {}

        # Debug directories
        self.frame_dir = Path("/tmp/prismatron_debug_frames")
        self.led_dir = Path("/tmp/prismatron_debug_leds")
        self.output_dir = Path("/tmp/prismatron_debug_analysis")
        self.output_dir.mkdir(exist_ok=True)

        # Load patterns
        self._load_patterns()

        # Frame dimensions (from metadata or default)
        self.frame_width = 800  # Default
        self.frame_height = 480  # Default
        if "frame_width" in self.metadata:
            self.frame_width = self.metadata["frame_width"]
        if "frame_height" in self.metadata:
            self.frame_height = self.metadata["frame_height"]

        # LED positions for preview rendering
        self.led_positions: Optional[np.ndarray] = None
        self.led_ordering: Optional[np.ndarray] = None

    def _load_patterns(self) -> None:
        """Load diffusion patterns from NPZ file."""
        try:
            logger.info(f"Loading patterns from {self.patterns_file}")
            data = np.load(self.patterns_file, allow_pickle=True)

            # Load metadata
            if "metadata" in data:
                self.metadata = data["metadata"].item()
                logger.info(f"Loaded metadata: {list(self.metadata.keys())}")

            # Load mixed tensor
            if "mixed_tensor" in data:
                mixed_tensor_dict = data["mixed_tensor"].item()
                self.mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
                logger.info(
                    f"Loaded mixed tensor: batch_size={self.mixed_tensor.batch_size}, block_size={self.mixed_tensor.block_size}"
                )
            else:
                raise ValueError("No mixed_tensor found in patterns file")

            logger.info("Patterns loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            raise

    def _load_led_positions(self) -> None:
        """Load LED positions and ordering from patterns file."""
        try:
            data = np.load(self.patterns_file, allow_pickle=True)

            # Load LED positions (in physical order)
            if "led_positions" in data:
                self.led_positions = data["led_positions"]
                logger.info(f"Loaded LED positions: {self.led_positions.shape}")

            # Load LED ordering (spatial_index -> physical_led_id)
            if "led_ordering" in data:
                self.led_ordering = data["led_ordering"]
                logger.info(f"Loaded LED ordering: {self.led_ordering.shape}")

            if self.led_positions is None:
                logger.warning("No LED positions found in patterns file - preview rendering will be limited")

        except Exception as e:
            logger.error(f"Failed to load LED positions: {e}")
            self.led_positions = None
            self.led_ordering = None

    def analyze_pipeline(self) -> None:
        """Analyze the complete debug pipeline."""
        logger.info("Starting debug pipeline analysis")

        # Check for debug files
        frame_files = sorted(self.frame_dir.glob("frame_*.npy"))
        led_spatial_files = sorted(self.led_dir.glob("led_spatial_*.npy"))
        led_physical_files = sorted(self.led_dir.glob("led_physical_*.npy"))

        logger.info(f"Found {len(frame_files)} frame files")
        logger.info(f"Found {len(led_spatial_files)} spatial LED files")
        logger.info(f"Found {len(led_physical_files)} physical LED files")

        if not frame_files:
            logger.error("No debug frame files found - run the system first to generate debug data")
            return

        if not led_spatial_files:
            logger.error("No debug LED files found - check if renderer is saving LED values")
            return

        # Load LED positions for preview rendering
        self._load_led_positions()

        # Analyze each frame/LED pair
        max_files = min(len(frame_files), len(led_spatial_files))
        for i in range(max_files):
            logger.info(f"Analyzing frame {i}")
            self._analyze_frame_pair(i, frame_files[i], led_spatial_files[i])

        logger.info("Debug pipeline analysis complete")

    def _analyze_frame_pair(self, index: int, frame_file: Path, led_file: Path) -> None:
        """
        Analyze a source frame and corresponding LED values.

        Args:
            index: Frame index
            frame_file: Path to source frame NPY file
            led_file: Path to LED values NPY file
        """
        try:
            # Load source frame and LED values
            source_frame = np.load(frame_file)  # Shape: (H, W, 3)
            led_values = np.load(led_file)  # Shape: (led_count, 3)

            logger.info(f"Frame {index}: source_frame.shape={source_frame.shape}, led_values.shape={led_values.shape}")
            logger.info(f"Frame {index}: source range=[{source_frame.min()}, {source_frame.max()}]")
            logger.info(f"Frame {index}: LED range=[{led_values.min()}, {led_values.max()}]")

            # Save source frame as PNG
            if PIL_AVAILABLE:
                source_png = self.output_dir / f"frame_{index:03d}_source.png"
                source_img = Image.fromarray(source_frame.astype(np.uint8))
                source_img.save(source_png)
                logger.info(f"Saved source frame to {source_png}")

            # Reconstruct image from LED values using mixed tensor
            reconstructed_frame = self._reconstruct_from_led_values(led_values)

            if reconstructed_frame is not None:
                logger.info(f"Frame {index}: reconstructed.shape={reconstructed_frame.shape}")
                logger.info(
                    f"Frame {index}: reconstructed range=[{reconstructed_frame.min()}, {reconstructed_frame.max()}]"
                )

                # Save reconstructed frame as PNG
                if PIL_AVAILABLE:
                    reconstructed_png = self.output_dir / f"frame_{index:03d}_reconstructed.png"
                    reconstructed_img = Image.fromarray(reconstructed_frame.astype(np.uint8))
                    reconstructed_img.save(reconstructed_png)
                    logger.info(f"Saved reconstructed frame to {reconstructed_png}")

                # Calculate difference
                diff_frame = np.abs(source_frame.astype(np.float32) - reconstructed_frame.astype(np.float32))
                mse = np.mean(diff_frame**2)
                logger.info(f"Frame {index}: MSE between source and reconstructed = {mse:.2f}")

                # Save difference frame
                if PIL_AVAILABLE:
                    diff_normalized = np.clip(diff_frame * 2, 0, 255).astype(np.uint8)  # Enhance difference visibility
                    diff_png = self.output_dir / f"frame_{index:03d}_difference.png"
                    diff_img = Image.fromarray(diff_normalized)
                    diff_img.save(diff_png)
                    logger.info(f"Saved difference frame to {diff_png}")

            # Render preview-style visualization (like website preview)
            if self.led_positions is not None and self.led_ordering is not None:
                preview_image = self._render_preview_style(led_values)
                if preview_image is not None and PIL_AVAILABLE:
                    preview_png = self.output_dir / f"frame_{index:03d}_preview.png"
                    preview_image.save(preview_png)
                    logger.info(f"Saved preview-style rendering to {preview_png}")

            # Save LED values as CSV for inspection
            led_csv = self.output_dir / f"frame_{index:03d}_led_values.csv"
            np.savetxt(led_csv, led_values, delimiter=",", fmt="%.1f", header="R,G,B", comments="")
            logger.info(f"Saved LED values to {led_csv}")

        except Exception as e:
            logger.error(f"Failed to analyze frame {index}: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _reconstruct_from_led_values(self, led_values: np.ndarray) -> Optional[np.ndarray]:
        """
        Reconstruct image from LED values using mixed tensor forward pass.

        Args:
            led_values: LED values in shape (led_count, 3), range [0, 255]

        Returns:
            Reconstructed frame in shape (H, W, 3), range [0, 255], or None if failed
        """
        try:
            # Convert LED values from [0, 255] to [0, 1] range for mixed tensor
            led_values_float = led_values.astype(np.float32) / 255.0

            logger.info(
                f"LED values for reconstruction: shape={led_values_float.shape}, range=[{led_values_float.min():.3f}, {led_values_float.max():.3f}]"
            )

            # Use forward_pass_3d to render all LEDs and channels at once
            # This matches the approach in visualize_diffusion_patterns.py
            output_frame = self.mixed_tensor.forward_pass_3d(led_values_float)

            # output_frame should be shape (3, height, width) in range [0, 1]
            logger.info(
                f"Forward pass output: shape={output_frame.shape}, range=[{output_frame.min():.3f}, {output_frame.max():.3f}]"
            )

            # Convert from planar (3, H, W) to interleaved (H, W, 3) format
            if output_frame.shape[0] == 3:
                output_frame = output_frame.transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)

            # Scale from [0, 1] to [0, 255] range
            output_frame = np.clip(output_frame, 0.0, 1.0)
            output_frame = (output_frame * 255.0).astype(np.uint8)

            # Ensure it's a NumPy array (CPU) for PIL compatibility
            if hasattr(output_frame, "get"):
                # It's a CuPy array - convert to NumPy
                output_frame = output_frame.get()

            logger.info(
                f"Final reconstructed frame: shape={output_frame.shape}, range=[{output_frame.min()}, {output_frame.max()}]"
            )
            return output_frame

        except Exception as e:
            logger.error(f"Failed to reconstruct frame from LED values: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def _render_preview_style(self, led_values: np.ndarray) -> Optional[Image.Image]:
        """
        Render LED values as circles at spatial positions (like website preview).

        Args:
            led_values: LED values in spatial order, shape (led_count, 3), range [0, 255]

        Returns:
            PIL Image with LED values rendered as 32-pixel circles, or None if failed
        """
        try:
            if not PIL_AVAILABLE:
                logger.warning("PIL not available - cannot render preview style")
                return None

            # Create black canvas
            canvas = Image.new("RGB", (self.frame_width, self.frame_height), color=(0, 0, 0))
            draw = ImageDraw.Draw(canvas)

            # Circle radius (32-pixel diameter = 16-pixel radius)
            radius = 16

            logger.info(f"Rendering {len(led_values)} LEDs as preview circles")

            # LED values are in spatial order, positions are in physical order
            # We need to map from spatial index to physical LED position
            for spatial_idx, (r, g, b) in enumerate(led_values):
                # Skip if this spatial index doesn't exist in ordering
                if spatial_idx >= len(self.led_ordering):
                    continue

                # Get physical LED ID for this spatial index
                physical_led_id = self.led_ordering[spatial_idx]

                # Skip if physical LED ID is out of range
                if physical_led_id >= len(self.led_positions):
                    continue

                # Get position for this physical LED
                x, y = self.led_positions[physical_led_id]

                # Convert to integer coordinates
                x, y = int(round(x)), int(round(y))

                # Skip if position is outside frame bounds
                if x < radius or y < radius or x >= self.frame_width - radius or y >= self.frame_height - radius:
                    continue

                # Convert LED values to integer RGB
                color = (int(r), int(g), int(b))

                # Draw filled circle at LED position
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

            logger.info("Preview-style rendering complete")
            return canvas

        except Exception as e:
            logger.error(f"Failed to render preview style: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def generate_report(self) -> None:
        """Generate a summary report of the analysis."""
        report_file = self.output_dir / "analysis_report.txt"

        with open(report_file, "w") as f:
            f.write("Prismatron Debug Frame Analysis Report\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Patterns file: {self.patterns_file}\n")
            f.write(f"Metadata keys: {list(self.metadata.keys())}\n")

            if self.mixed_tensor:
                f.write(f"Mixed tensor batch_size: {self.mixed_tensor.batch_size}\n")
                f.write(f"Mixed tensor block_size: {self.mixed_tensor.block_size}\n")

            f.write(f"\nFrame files: {len(list(self.frame_dir.glob('frame_*.npy')))}\n")
            f.write(f"LED files: {len(list(self.led_dir.glob('led_spatial_*.npy')))}\n")

            f.write(f"\nOutput files saved to: {self.output_dir}\n")
            f.write("\nFiles generated:\n")
            for png_file in sorted(self.output_dir.glob("*.png")):
                f.write(f"  {png_file.name}\n")
            for csv_file in sorted(self.output_dir.glob("*.csv")):
                f.write(f"  {csv_file.name}\n")

            # Report LED position loading status
            f.write(f"\nLED position data loaded: {self.led_positions is not None}\n")
            if self.led_positions is not None:
                f.write(f"LED count: {len(self.led_positions)}\n")
                f.write(f"LED ordering loaded: {self.led_ordering is not None}\n")
                if self.led_ordering is not None:
                    f.write(f"Ordering array length: {len(self.led_ordering)}\n")
                f.write("Preview-style rendering: Available\n")
            else:
                f.write("Preview-style rendering: Not available (no LED positions)\n")

        logger.info(f"Generated analysis report: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug frame analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze debug frames with diffusion patterns
  python debug_frame_analysis.py --patterns diffusion_patterns/synthetic_2624_uint8_fp32.npz

  # Enable verbose logging
  python debug_frame_analysis.py --patterns patterns.npz --verbose
        """,
    )

    parser.add_argument("--patterns", required=True, type=Path, help="Diffusion patterns NPZ file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if not args.patterns.exists():
        logger.error(f"Patterns file not found: {args.patterns}")
        sys.exit(1)

    try:
        # Create analyzer and run analysis
        analyzer = DebugFrameAnalyzer(str(args.patterns))
        analyzer.analyze_pipeline()
        analyzer.generate_report()

        logger.info("Analysis complete!")
        logger.info(f"Check output files in: {analyzer.output_dir}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
