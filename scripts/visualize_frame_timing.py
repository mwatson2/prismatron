#!/usr/bin/env python3
"""
Frame Timing Visualization Script - Updated for late dropped frames and out-of-order entries

This script reads frame timing CSV data and creates detailed visualizations
showing the timeline of frame processing through the pipeline, handling both
complete and incomplete (dropped) frames.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class FrameTimingVisualizer:
    """Visualizes frame timing data from CSV files."""

    def __init__(self, csv_file_path: str):
        """
        Initialize visualizer with CSV data.

        Args:
            csv_file_path: Path to CSV file containing timing data
        """
        self.csv_file_path = Path(csv_file_path)
        self.timing_data: Optional[pd.DataFrame] = None
        self.complete_frames: Optional[pd.DataFrame] = None
        self.incomplete_frames: Optional[pd.DataFrame] = None
        self.load_data()

    def load_data(self) -> None:
        """Load timing data from CSV file."""
        try:
            self.timing_data = pd.read_csv(self.csv_file_path)
            print(f"Loaded {len(self.timing_data)} frame timing records")

            # Validate required columns
            required_columns = [
                "frame_index",
                "plugin_timestamp",
                "producer_timestamp",
                "write_to_buffer_time",
                "read_from_buffer_time",
                "write_to_led_buffer_time",
                "read_from_led_buffer_time",
                "render_time",
                "item_duration",
            ]

            missing_columns = [col for col in required_columns if col not in self.timing_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Sort by frame_index to handle out-of-order entries
            self.timing_data = self.timing_data.sort_values("frame_index").reset_index(drop=True)
            print(f"Data sorted by frame_index")

            # Separate complete and incomplete frames (zeros indicate late drops)
            complete_mask = (
                (self.timing_data["write_to_buffer_time"] > 0)
                & (self.timing_data["read_from_buffer_time"] > 0)
                & (self.timing_data["write_to_led_buffer_time"] > 0)
                & (self.timing_data["read_from_led_buffer_time"] > 0)
                & (self.timing_data["render_time"] > 0)
            )

            # Store both complete and incomplete frames for analysis
            self.complete_frames = self.timing_data[complete_mask].copy()
            self.incomplete_frames = self.timing_data[~complete_mask].copy()

            complete_count = len(self.complete_frames)
            incomplete_count = len(self.incomplete_frames)
            total_count = len(self.timing_data)

            print(f"Complete frames: {complete_count} ({complete_count/total_count*100:.1f}%)")
            print(f"Incomplete/dropped frames: {incomplete_count} ({incomplete_count/total_count*100:.1f}%)")
            print(f"Total frames: {total_count}")

        except Exception as e:
            print(f"Error loading timing data: {e}")
            raise

    def create_timeline_visualization(self, output_path: Optional[str] = None, max_frames: int = 1000) -> None:
        """
        Create timeline visualization with complete and incomplete frames.

        Args:
            output_path: Path to save the visualization (optional)
            max_frames: Maximum number of frames to visualize (for performance)
        """
        if self.timing_data is None or len(self.timing_data) == 0:
            print("No valid timing data to visualize")
            return

        # Limit to max_frames for performance
        all_data = self.timing_data.head(max_frames).copy()
        complete_data = self.complete_frames.head(max_frames).copy()
        incomplete_data = self.incomplete_frames.head(max_frames).copy()

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16))
        fig.suptitle(
            f"Frame Timing Pipeline Analysis ({len(all_data)} total frames, {len(complete_data)} complete, {len(incomplete_data)} dropped)",
            fontsize=16,
        )

        # Define colors for different stages
        colors = {
            "plugin_timestamp": "#1f77b4",  # Blue
            "producer_timestamp": "#ff7f0e",  # Orange
            "write_to_buffer_time": "#2ca02c",  # Green
            "read_from_buffer_time": "#d62728",  # Red
            "write_to_led_buffer_time": "#9467bd",  # Purple
            "read_from_led_buffer_time": "#8c564b",  # Brown
            "render_time": "#e377c2",  # Pink
        }

        # Calculate wallclock time range from all available data
        all_wallclock_times = []

        # Collect all non-zero wallclock times from all frames
        if len(complete_data) > 0:
            all_wallclock_times.extend(
                [
                    complete_data["write_to_buffer_time"],
                    complete_data["read_from_buffer_time"],
                    complete_data["write_to_led_buffer_time"],
                    complete_data["read_from_led_buffer_time"],
                    complete_data["render_time"],
                ]
            )

        if len(incomplete_data) > 0:
            for col in [
                "write_to_buffer_time",
                "read_from_buffer_time",
                "write_to_led_buffer_time",
                "read_from_led_buffer_time",
                "render_time",
            ]:
                non_zero_times = incomplete_data[incomplete_data[col] > 0][col]
                if len(non_zero_times) > 0:
                    all_wallclock_times.append(non_zero_times)

        # Find the earliest wallclock time to use as reference for all plots
        if all_wallclock_times:
            min_wallclock = min(times.min() for times in all_wallclock_times if len(times) > 0)
        else:
            min_wallclock = 0

        # Plot 1: Content timestamps converted to wallclock time - All frames
        all_y_positions = all_data["frame_index"].values

        # Convert content timestamps to wallclock time for alignment
        content_start_wallclock = min_wallclock  # Assume content starts at the beginning of processing

        ax1.scatter(
            all_data["plugin_timestamp"] + (content_start_wallclock - min_wallclock),
            all_y_positions,
            c=colors["plugin_timestamp"],
            s=1,
            alpha=0.7,
            label="Plugin Timestamp",
        )
        ax1.scatter(
            all_data["producer_timestamp"] + (content_start_wallclock - min_wallclock),
            all_y_positions,
            c=colors["producer_timestamp"],
            s=1,
            alpha=0.7,
            label="Producer Timestamp",
        )

        ax1.set_xlabel("Time (seconds from processing start)")
        ax1.set_ylabel("Frame Index")
        ax1.set_title("Content Timeline (Plugin and Producer Timestamps) - All Frames")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Calculate x-axis limits for all plots to align them
        all_x_values = []
        all_x_values.extend(all_data["plugin_timestamp"])
        all_x_values.extend(all_data["producer_timestamp"])

        if len(complete_data) > 0:
            all_x_values.extend(complete_data["write_to_buffer_time"] - min_wallclock)
            all_x_values.extend(complete_data["read_from_buffer_time"] - min_wallclock)
            all_x_values.extend(complete_data["write_to_led_buffer_time"] - min_wallclock)
            all_x_values.extend(complete_data["read_from_led_buffer_time"] - min_wallclock)
            all_x_values.extend(complete_data["render_time"] - min_wallclock)

            # Plot 2: Complete frames pipeline (wallclock times)
            complete_y_positions = complete_data["frame_index"].values

            ax2.scatter(
                complete_data["write_to_buffer_time"] - min_wallclock,
                complete_y_positions,
                c=colors["write_to_buffer_time"],
                s=1,
                alpha=0.7,
                label="Write to Buffer",
            )
            ax2.scatter(
                complete_data["read_from_buffer_time"] - min_wallclock,
                complete_y_positions,
                c=colors["read_from_buffer_time"],
                s=1,
                alpha=0.7,
                label="Read from Buffer",
            )
            ax2.scatter(
                complete_data["write_to_led_buffer_time"] - min_wallclock,
                complete_y_positions,
                c=colors["write_to_led_buffer_time"],
                s=1,
                alpha=0.7,
                label="Write to LED Buffer",
            )
            ax2.scatter(
                complete_data["read_from_led_buffer_time"] - min_wallclock,
                complete_y_positions,
                c=colors["read_from_led_buffer_time"],
                s=1,
                alpha=0.7,
                label="Read from LED Buffer",
            )
            ax2.scatter(
                complete_data["render_time"] - min_wallclock,
                complete_y_positions,
                c=colors["render_time"],
                s=1,
                alpha=0.7,
                label="Render Time",
            )

            ax2.set_xlabel("Time (seconds from processing start)")
            ax2.set_ylabel("Frame Index")
            ax2.set_title("Complete Frames Pipeline Timeline (Wallclock Times)")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(
                0.5,
                0.5,
                "No complete frames to display",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
                fontsize=14,
            )
            ax2.set_title("Complete Frames Pipeline Timeline (No Data)")

        # Plot 3: Dropped/incomplete frames analysis
        if len(incomplete_data) > 0:
            incomplete_y_positions = incomplete_data["frame_index"].values

            # Show what stages incomplete frames reached
            stages = [
                ("write_to_buffer_time", "Write to Buffer", colors["write_to_buffer_time"]),
                ("read_from_buffer_time", "Read from Buffer", colors["read_from_buffer_time"]),
                ("write_to_led_buffer_time", "Write to LED Buffer", colors["write_to_led_buffer_time"]),
                ("read_from_led_buffer_time", "Read from LED Buffer", colors["read_from_led_buffer_time"]),
                ("render_time", "Render Time", colors["render_time"]),
            ]

            for stage_col, stage_name, color in stages:
                stage_mask = incomplete_data[stage_col] > 0
                if stage_mask.any():
                    stage_data = incomplete_data[stage_mask]
                    # Always use the same time reference for alignment
                    stage_times = stage_data[stage_col] - min_wallclock
                    all_x_values.extend(stage_times)
                    ax3.scatter(
                        stage_times,
                        stage_data["frame_index"],
                        c=color,
                        s=3,
                        alpha=0.8,
                        label=f"{stage_name} (reached)",
                        marker="x",
                    )

            ax3.set_xlabel("Time (seconds from processing start)")
            ax3.set_ylabel("Frame Index")
            ax3.set_title("Dropped/Incomplete Frames - Pipeline Stages Reached")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(
                0.5,
                0.5,
                "No dropped frames to display",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax3.transAxes,
                fontsize=14,
            )
            ax3.set_title("Dropped/Incomplete Frames (No Data)")

        # Align x-axis limits for all three plots
        if all_x_values:
            x_min = min(all_x_values)
            x_max = max(all_x_values)
            x_margin = (x_max - x_min) * 0.05  # 5% margin
            ax1.set_xlim(x_min - x_margin, x_max + x_margin)
            ax2.set_xlim(x_min - x_margin, x_max + x_margin)
            ax3.set_xlim(x_min - x_margin, x_max + x_margin)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Timeline visualization saved to {output_path}")
        else:
            plt.show()

    def create_latency_analysis(self, output_path: Optional[str] = None) -> None:
        """
        Create latency analysis showing time differences between pipeline stages (complete frames only).

        Args:
            output_path: Path to save the visualization (optional)
        """
        if self.complete_frames is None or len(self.complete_frames) == 0:
            print("No complete frames for latency analysis")
            return

        data = self.complete_frames.copy()

        # Calculate latencies between stages
        latencies = {
            "Buffer Latency": data["read_from_buffer_time"] - data["write_to_buffer_time"],
            "Processing Latency": data["write_to_led_buffer_time"] - data["read_from_buffer_time"],
            "LED Buffer Latency": data["read_from_led_buffer_time"] - data["write_to_led_buffer_time"],
            "Render Latency": data["render_time"] - data["read_from_led_buffer_time"],
            "End-to-End Latency": data["render_time"] - data["write_to_buffer_time"],
        }

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Frame Processing Latency Analysis", fontsize=16)

        # Plot histograms of latencies
        for idx, (name, latency) in enumerate(latencies.items()):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col] if len(latencies) > 3 else axes[col]

            # Remove outliers for better visualization (keep 99th percentile)
            p99 = np.percentile(latency, 99)
            filtered_latency = latency[latency <= p99]

            ax.hist(filtered_latency * 1000, bins=50, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Latency (milliseconds)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{name}\nMean: {latency.mean()*1000:.1f}ms, Std: {latency.std()*1000:.1f}ms")
            ax.grid(True, alpha=0.3)

        # Remove empty subplot if we have 5 plots
        if len(latencies) == 5:
            fig.delaxes(axes[1, 2])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Latency analysis saved to {output_path}")
        else:
            plt.show()

    def print_statistics(self) -> None:
        """Print detailed timing statistics."""
        if self.timing_data is None or len(self.timing_data) == 0:
            print("No valid timing data for statistics")
            return

        all_data = self.timing_data
        complete_data = self.complete_frames
        incomplete_data = self.incomplete_frames

        print("\n=== Frame Timing Statistics ===")
        print(f"Total frames analyzed: {len(all_data)}")
        print(f"Complete frames: {len(complete_data)} ({len(complete_data)/len(all_data)*100:.1f}%)")
        print(f"Incomplete/dropped frames: {len(incomplete_data)} ({len(incomplete_data)/len(all_data)*100:.1f}%)")
        print(f"Frame index range: {all_data['frame_index'].min()} - {all_data['frame_index'].max()}")
        print(
            f"Content duration range: {all_data['plugin_timestamp'].min():.3f}s - {all_data['plugin_timestamp'].max():.3f}s"
        )

        # Analyze dropped frame patterns
        if len(incomplete_data) > 0:
            print("\n=== Dropped Frame Analysis ===")

            # Count frames that reached each stage
            stages = [
                ("write_to_buffer_time", "Write to Buffer"),
                ("read_from_buffer_time", "Read from Buffer"),
                ("write_to_led_buffer_time", "Write to LED Buffer"),
                ("read_from_led_buffer_time", "Read from LED Buffer"),
                ("render_time", "Render Time"),
            ]

            for stage_col, stage_name in stages:
                reached_count = (incomplete_data[stage_col] > 0).sum()
                print(
                    f"Frames reaching {stage_name}: {reached_count}/{len(incomplete_data)} ({reached_count/len(incomplete_data)*100:.1f}%)"
                )

        # Calculate processing latencies (complete frames only)
        if len(complete_data) > 0:
            buffer_latency = (complete_data["read_from_buffer_time"] - complete_data["write_to_buffer_time"]) * 1000
            processing_latency = (
                complete_data["write_to_led_buffer_time"] - complete_data["read_from_buffer_time"]
            ) * 1000
            led_buffer_latency = (
                complete_data["read_from_led_buffer_time"] - complete_data["write_to_led_buffer_time"]
            ) * 1000
            render_latency = (complete_data["render_time"] - complete_data["read_from_led_buffer_time"]) * 1000
            end_to_end_latency = (complete_data["render_time"] - complete_data["write_to_buffer_time"]) * 1000

            print("\n=== Processing Latencies (milliseconds) - Complete Frames Only ====")
            latencies = {
                "Shared Buffer": buffer_latency,
                "Optimization": processing_latency,
                "LED Buffer": led_buffer_latency,
                "Render": render_latency,
                "End-to-End": end_to_end_latency,
            }

            for name, latency in latencies.items():
                print(
                    f"{name:15}: Mean={latency.mean():6.1f}ms, Std={latency.std():6.1f}ms, "
                    f"P50={latency.median():6.1f}ms, P95={np.percentile(latency, 95):6.1f}ms, "
                    f"P99={np.percentile(latency, 99):6.1f}ms"
                )
        else:
            print("\n=== Processing Latencies ====")
            print("No complete frames available for latency analysis")

        # Calculate frame rates
        if len(all_data) > 1:
            plugin_frame_interval = np.diff(all_data["plugin_timestamp"]).mean()

            print(f"\n=== Frame Rates ===")
            print(
                f"Plugin frame rate: {1.0/plugin_frame_interval:.1f} fps (interval: {plugin_frame_interval*1000:.1f}ms)"
            )

            if len(complete_data) > 1:
                render_frame_interval = np.diff(complete_data["render_time"]).mean()
                print(
                    f"Render frame rate: {1.0/render_frame_interval:.1f} fps (interval: {render_frame_interval*1000:.1f}ms)"
                )
                print(f"Frame completion rate: {len(complete_data)/len(all_data)*100:.1f}%")
            else:
                print("Render frame rate: N/A (no complete frames)")
                print(f"Frame completion rate: 0.0%")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize frame timing data from CSV file")
    parser.add_argument("csv_file", help="Path to CSV file with timing data")
    parser.add_argument("--output-dir", "-o", help="Output directory for visualizations")
    parser.add_argument(
        "--max-frames", "-m", type=int, default=1000, help="Maximum number of frames to visualize (default: 1000)"
    )
    parser.add_argument("--stats-only", "-s", action="store_true", help="Print statistics only, no visualizations")

    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    try:
        visualizer = FrameTimingVisualizer(str(csv_path))

        # Print statistics
        visualizer.print_statistics()

        if not args.stats_only:
            # Determine output paths
            output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            timeline_path = output_dir / f"{csv_path.stem}_timeline.png"
            latency_path = output_dir / f"{csv_path.stem}_latency.png"

            # Create visualizations
            print(f"\nCreating timeline visualization...")
            visualizer.create_timeline_visualization(str(timeline_path), args.max_frames)

            if len(visualizer.complete_frames) > 0:
                print(f"Creating latency analysis...")
                visualizer.create_latency_analysis(str(latency_path))
            else:
                print("Skipping latency analysis - no complete frames")

            print(f"\nVisualization complete! Files saved to {output_dir}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
