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
                "led_transition_time",
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
                & (self.timing_data["led_transition_time"] > 0)
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
        Create timeline visualization with horizontal lines showing frame states.

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

        # Create figure with 3 subplots (content timeline + 2 frame state plots)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18))
        fig.suptitle(
            f"Frame Timing Pipeline Analysis ({len(all_data)} total frames, {len(complete_data)} complete, {len(incomplete_data)} dropped)",
            fontsize=16,
        )

        # Define colors for different frame states
        state_colors = {
            "shared_buffer": "#2ca02c",  # Green - time in shared buffer
            "optimization": "#d62728",  # Red - optimization time
            "led_buffer": "#9467bd",  # Purple - time in LED buffer
            "led_transitions": "#ff69b4",  # Hot Pink - LED transitions time
            "rendered": "#ffff00",  # Bright Yellow - time in rendered state (high visibility)
            "led_queue": "#ff8c00",  # Orange - LED buffer queue length
            "plugin_timestamp": "#1f77b4",  # Blue - plugin timestamps
            "producer_timestamp": "#ff7f0e",  # Orange - producer timestamps
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
                    complete_data["led_transition_time"],
                    complete_data["render_time"],
                ]
            )

        if len(incomplete_data) > 0:
            for col in [
                "write_to_buffer_time",
                "read_from_buffer_time",
                "write_to_led_buffer_time",
                "read_from_led_buffer_time",
                "led_transition_time",
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

        def draw_frame_states(ax, data, title_suffix="", show_incomplete=False):
            """Draw frame states with 2px linewidth and 1px gap between frames."""
            ax.set_title(f"Frame States Combined{title_suffix}")

            # Plot complete frames
            for _, frame in data.iterrows():
                frame_y = frame["frame_index"] * 3  # 3px per frame (2px line + 1px gap)

                # Time in shared buffer (with 1px gap at start)
                if frame["write_to_buffer_time"] > 0 and frame["read_from_buffer_time"] > 0:
                    start_time = frame["write_to_buffer_time"] - min_wallclock + 0.001  # 1px gap
                    end_time = frame["read_from_buffer_time"] - min_wallclock
                    ax.plot(
                        [start_time, end_time],
                        [frame_y, frame_y],
                        color=state_colors["shared_buffer"],
                        linewidth=2,
                        alpha=0.8,
                        solid_capstyle="butt",
                        zorder=3,  # Higher z-order to draw on top of LED queue
                    )

                # Optimization time
                if frame["read_from_buffer_time"] > 0 and frame["write_to_led_buffer_time"] > 0:
                    start_time = frame["read_from_buffer_time"] - min_wallclock + 0.001
                    end_time = frame["write_to_led_buffer_time"] - min_wallclock
                    ax.plot(
                        [start_time, end_time],
                        [frame_y, frame_y],
                        color=state_colors["optimization"],
                        linewidth=2,
                        alpha=0.8,
                        solid_capstyle="butt",
                        zorder=3,  # Higher z-order to draw on top of LED queue
                    )

                # Time in LED buffer
                if frame["write_to_led_buffer_time"] > 0 and frame["read_from_led_buffer_time"] > 0:
                    start_time = frame["write_to_led_buffer_time"] - min_wallclock + 0.001
                    end_time = frame["read_from_led_buffer_time"] - min_wallclock
                    ax.plot(
                        [start_time, end_time],
                        [frame_y, frame_y],
                        color=state_colors["led_buffer"],
                        linewidth=2,
                        alpha=0.8,
                        solid_capstyle="butt",
                        zorder=3,  # Higher z-order to draw on top of LED queue
                    )

                # LED transitions time
                if frame["read_from_led_buffer_time"] > 0 and frame["led_transition_time"] > 0:
                    start_time = frame["read_from_led_buffer_time"] - min_wallclock + 0.001
                    end_time = frame["led_transition_time"] - min_wallclock
                    ax.plot(
                        [start_time, end_time],
                        [frame_y, frame_y],
                        color=state_colors["led_transitions"],
                        linewidth=2,
                        alpha=0.8,
                        solid_capstyle="butt",
                        zorder=3,  # Higher z-order to draw on top of LED queue
                    )

                # Time in rendered state
                if frame["led_transition_time"] > 0 and frame["render_time"] > 0:
                    start_time = frame["led_transition_time"] - min_wallclock + 0.001
                    end_time = frame["render_time"] - min_wallclock
                    ax.plot(
                        [start_time, end_time],
                        [frame_y, frame_y],
                        color=state_colors["rendered"],
                        linewidth=3,  # Make rendered segments thicker for better visibility
                        alpha=1.0,  # Make rendered segments fully opaque
                        solid_capstyle="butt",
                        zorder=4,  # Highest z-order for rendered segments
                    )

            # Plot incomplete frames if requested
            if show_incomplete and len(incomplete_data) > 0:
                for _, frame in incomplete_data.iterrows():
                    frame_y = frame["frame_index"] * 3  # 3px per frame (2px line + 1px gap)

                    # Show partial states for incomplete frames
                    if frame["write_to_buffer_time"] > 0 and frame["read_from_buffer_time"] > 0:
                        start_time = frame["write_to_buffer_time"] - min_wallclock + 0.001
                        end_time = frame["read_from_buffer_time"] - min_wallclock
                        ax.plot(
                            [start_time, end_time],
                            [frame_y, frame_y],
                            color=state_colors["shared_buffer"],
                            linewidth=2,
                            alpha=0.4,
                            solid_capstyle="butt",
                            zorder=2,  # Medium z-order for incomplete frames
                        )

                    if frame["read_from_buffer_time"] > 0 and frame["write_to_led_buffer_time"] > 0:
                        start_time = frame["read_from_buffer_time"] - min_wallclock + 0.001
                        end_time = frame["write_to_led_buffer_time"] - min_wallclock
                        ax.plot(
                            [start_time, end_time],
                            [frame_y, frame_y],
                            color=state_colors["optimization"],
                            linewidth=2,
                            alpha=0.4,
                            solid_capstyle="butt",
                            zorder=2,  # Medium z-order for incomplete frames
                        )

                    if frame["write_to_led_buffer_time"] > 0 and frame["read_from_led_buffer_time"] > 0:
                        start_time = frame["write_to_led_buffer_time"] - min_wallclock + 0.001
                        end_time = frame["read_from_led_buffer_time"] - min_wallclock
                        ax.plot(
                            [start_time, end_time],
                            [frame_y, frame_y],
                            color=state_colors["led_buffer"],
                            linewidth=2,
                            alpha=0.4,
                            solid_capstyle="butt",
                            zorder=2,  # Medium z-order for incomplete frames
                        )

                    # LED transitions time for incomplete frames
                    if frame["read_from_led_buffer_time"] > 0 and frame["led_transition_time"] > 0:
                        start_time = frame["read_from_led_buffer_time"] - min_wallclock + 0.001
                        end_time = frame["led_transition_time"] - min_wallclock
                        ax.plot(
                            [start_time, end_time],
                            [frame_y, frame_y],
                            color=state_colors["led_transitions"],
                            linewidth=2,
                            alpha=0.4,
                            solid_capstyle="butt",
                            zorder=2,  # Medium z-order for incomplete frames
                        )

        def calculate_led_queue_length_square_wave(data):
            """Calculate LED buffer queue length over time as square wave (only complete frames)."""
            events = []

            # Only add events for frames that have BOTH write_to_led_buffer_time AND led_transition_time
            # This excludes dropped frames that only have one of these timestamps
            for _, frame in data.iterrows():
                if frame["write_to_led_buffer_time"] > 0 and frame["led_transition_time"] > 0:
                    events.append((frame["write_to_led_buffer_time"], +1, frame["frame_index"]))
                    events.append((frame["led_transition_time"], -1, frame["frame_index"]))

            # Sort by time
            events.sort()

            # Calculate square wave points
            times = []
            queue_lengths = []
            current_length = 0

            for timestamp, delta, frame_idx in events:
                # Add point just before change (horizontal line)
                times.append(timestamp - min_wallclock)
                queue_lengths.append(current_length)

                # Update queue length
                current_length += delta

                # Add point just after change (vertical line)
                times.append(timestamp - min_wallclock)
                queue_lengths.append(current_length)

            return times, queue_lengths

        # Plot 1: Content Timeline (Plugin and Producer Timestamps)
        ax1.set_title("Content Timeline (Plugin and Producer Timestamps)")
        all_y_positions = all_data["frame_index"].values

        # Convert content timestamps to wallclock time for alignment
        content_start_wallclock = min_wallclock  # Assume content starts at the beginning of processing

        ax1.scatter(
            all_data["plugin_timestamp"] + (content_start_wallclock - min_wallclock),
            all_y_positions,
            c=state_colors["plugin_timestamp"],
            s=1,
            alpha=0.7,
            label="Plugin Timestamp (Item Timestamp)",
        )
        ax1.scatter(
            all_data["producer_timestamp"] + (content_start_wallclock - min_wallclock),
            all_y_positions,
            c=state_colors["producer_timestamp"],
            s=1,
            alpha=0.7,
            label="Producer Timestamp (Global Timestamp)",
        )

        ax1.set_ylabel("Frame Index")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Complete frames only
        # First, create and configure LED queue axis (behind frame states)
        if len(complete_data) > 0:
            queue_times, queue_lengths = calculate_led_queue_length_square_wave(complete_data)
            if queue_times:
                ax2_queue = ax2.twinx()
                ax2_queue.plot(
                    queue_times,
                    queue_lengths,
                    color=state_colors["led_queue"],
                    linewidth=1.5,
                    alpha=0.3,  # More transparent so it stays in background
                    label="LED Buffer Queue Length",
                    zorder=1,  # Lower z-order to draw behind frame states
                )
                # Set axis to go to at least 10
                max_queue = max(queue_lengths) if queue_lengths else 0
                ax2_queue.set_ylim(0, max(10, max_queue + 1))
                ax2_queue.set_ylabel("LED Buffer Queue Length", color=state_colors["led_queue"])
                ax2_queue.tick_params(axis="y", labelcolor=state_colors["led_queue"])
                ax2_queue.grid(True, alpha=0.2, axis="y")  # Add subtle grid for queue levels

        # Then draw frame states on top
        draw_frame_states(ax2, complete_data, " (Complete Frames Only)")

        # Add legends for plot 2
        ax2.plot([], [], color=state_colors["shared_buffer"], linewidth=2, label="Time in Shared Buffer")
        ax2.plot([], [], color=state_colors["optimization"], linewidth=2, label="Optimization Time")
        ax2.plot([], [], color=state_colors["led_buffer"], linewidth=2, label="Time in LED Buffer")
        ax2.plot([], [], color=state_colors["led_transitions"], linewidth=2, label="LED Transitions Time")
        ax2.plot([], [], color=state_colors["rendered"], linewidth=2, label="Time in Rendered State")
        ax2.set_ylabel("Frame Index")
        ax2.legend(loc="upper left")

        # Plot 3: All frames (complete + incomplete)
        # First, create and configure LED queue axis (behind frame states)
        if len(complete_data) > 0:
            queue_times, queue_lengths = calculate_led_queue_length_square_wave(complete_data)
            if queue_times:
                ax3_queue = ax3.twinx()
                ax3_queue.plot(
                    queue_times,
                    queue_lengths,
                    color=state_colors["led_queue"],
                    linewidth=1.5,
                    alpha=0.3,  # More transparent so it stays in background
                    label="LED Buffer Queue Length",
                    zorder=1,  # Lower z-order to draw behind frame states
                )
                # Set axis to go to at least 10
                max_queue = max(queue_lengths) if queue_lengths else 0
                ax3_queue.set_ylim(0, max(10, max_queue + 1))
                ax3_queue.set_ylabel("LED Buffer Queue Length", color=state_colors["led_queue"])
                ax3_queue.tick_params(axis="y", labelcolor=state_colors["led_queue"])
                ax3_queue.grid(True, alpha=0.2, axis="y")  # Add subtle grid for queue levels

        # Then draw frame states on top
        draw_frame_states(ax3, complete_data, " (All Frames)", show_incomplete=True)

        # Add legends for plot 3
        ax3.plot([], [], color=state_colors["shared_buffer"], linewidth=2, label="Complete: Time in Shared Buffer")
        ax3.plot([], [], color=state_colors["optimization"], linewidth=2, label="Complete: Optimization Time")
        ax3.plot([], [], color=state_colors["led_buffer"], linewidth=2, label="Complete: Time in LED Buffer")
        ax3.plot([], [], color=state_colors["led_transitions"], linewidth=2, label="Complete: LED Transitions Time")
        ax3.plot([], [], color=state_colors["rendered"], linewidth=2, label="Complete: Time in Rendered State")
        ax3.plot(
            [], [], color=state_colors["shared_buffer"], linewidth=2, alpha=0.4, label="Incomplete: Partial States"
        )
        ax3.set_xlabel("Time (seconds from processing start)")
        ax3.set_ylabel("Frame Index")
        ax3.legend(loc="upper left")

        # Calculate x-axis limits from all timing data
        all_x_values = []
        if len(all_data) > 0:
            for col in [
                "write_to_buffer_time",
                "read_from_buffer_time",
                "write_to_led_buffer_time",
                "read_from_led_buffer_time",
                "led_transition_time",
                "render_time",
            ]:
                non_zero_times = all_data[all_data[col] > 0][col]
                if len(non_zero_times) > 0:
                    all_x_values.extend(non_zero_times - min_wallclock)

        # Add content timestamps to x-axis calculation
        all_x_values.extend(all_data["plugin_timestamp"])
        all_x_values.extend(all_data["producer_timestamp"])

        # Set up axes for all plots with adaptive scaling
        if all_x_values:
            x_min = min(all_x_values)
            x_max = max(all_x_values)
            x_margin = (x_max - x_min) * 0.05
            x_range = x_max - x_min

            # Calculate adaptive Y-axis (frame index) scaling
            max_frame_index = max(all_data["frame_index"])
            max_frame_y = max_frame_index * 3  # Account for 3px per frame (2px line + 1px gap)

            # Target ~20 labels on Y-axis, interval must be multiple of 10
            target_y_labels = 20
            y_range = max_frame_y
            base_interval = max(15, int(y_range / target_y_labels / 15) * 15)  # Round to nearest 15 (5 frames * 3px)

            # Ensure we have at least some reasonable spacing
            y_interval = max(30, base_interval)  # Minimum 30 (10 frames * 3px)

            # Calculate adaptive X-axis (time) scaling
            # Target ~100 vertical grid lines
            target_x_lines = 100
            x_interval = x_range / target_x_lines

            # Round to nice intervals (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, etc.)
            magnitude = 10 ** np.floor(np.log10(x_interval))
            normalized = x_interval / magnitude
            if normalized <= 1:
                nice_interval = magnitude
            elif normalized <= 2:
                nice_interval = 2 * magnitude
            elif normalized <= 5:
                nice_interval = 5 * magnitude
            else:
                nice_interval = 10 * magnitude

            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(x_min - x_margin, x_max + x_margin)

                # Adaptive horizontal gridlines for frame indices
                h_grid_ticks = np.arange(0, max_frame_y + y_interval, y_interval)
                ax.set_yticks(h_grid_ticks, minor=False)

                # Convert y-tick positions back to frame numbers for labels
                ax.set_yticklabels([f"{int(y/3)}" for y in h_grid_ticks])

                ax.grid(True, which="major", axis="y", alpha=0.3, linestyle="-")

                # Adaptive vertical gridlines for time
                grid_start = np.floor(x_min / nice_interval) * nice_interval
                grid_end = np.ceil(x_max / nice_interval) * nice_interval
                v_grid_ticks = np.arange(grid_start, grid_end + nice_interval, nice_interval)
                ax.set_xticks(v_grid_ticks, minor=True)
                ax.grid(True, which="minor", axis="x", alpha=0.2, linestyle="-", linewidth=0.5)
                ax.grid(True, which="major", axis="x", alpha=0.3)

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
            "LED Transitions Latency": data["led_transition_time"] - data["read_from_led_buffer_time"],
            "Render Latency": data["render_time"] - data["led_transition_time"],
            "End-to-End Latency": data["render_time"] - data["write_to_buffer_time"],
        }

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Frame Processing Latency Analysis", fontsize=16)

        # Plot histograms of latencies
        for idx, (name, latency) in enumerate(latencies.items()):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            # Remove outliers for better visualization (keep 99th percentile)
            p99 = np.percentile(latency, 99)
            filtered_latency = latency[latency <= p99]

            ax.hist(filtered_latency * 1000, bins=50, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Latency (milliseconds)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{name}\nMean: {latency.mean()*1000:.1f}ms, Std: {latency.std()*1000:.1f}ms")
            ax.grid(True, alpha=0.3)

        # We now have 6 plots, so no need to remove any subplot

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

        # Analyze frame progression through pipeline stages
        print("\n=== Frame Pipeline Progression Analysis ===")

        # Count ALL frames that reached each stage (complete + incomplete)
        stages = [
            ("write_to_buffer_time", "Write to Buffer"),
            ("read_from_buffer_time", "Read from Buffer"),
            ("write_to_led_buffer_time", "Write to LED Buffer"),
            ("read_from_led_buffer_time", "Read from LED Buffer"),
            ("led_transition_time", "LED Transitions"),
            ("render_time", "Render Time"),
        ]

        total_frames = len(all_data)
        for stage_col, stage_name in stages:
            reached_count = (all_data[stage_col] > 0).sum()
            print(
                f"Frames reaching {stage_name}: {reached_count}/{total_frames} ({reached_count/total_frames*100:.1f}%)"
            )

        # Additional analysis for incomplete frames only
        if len(incomplete_data) > 0:
            print(f"\n=== Incomplete Frame Details ===")
            print(f"Total incomplete frames: {len(incomplete_data)}")

            for stage_col, stage_name in stages:
                incomplete_reached = (incomplete_data[stage_col] > 0).sum()
                print(
                    f"  Incomplete frames reaching {stage_name}: {incomplete_reached}/{len(incomplete_data)} ({incomplete_reached/len(incomplete_data)*100:.1f}%)"
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
            led_transitions_latency = (
                complete_data["led_transition_time"] - complete_data["read_from_led_buffer_time"]
            ) * 1000
            render_latency = (complete_data["render_time"] - complete_data["led_transition_time"]) * 1000
            end_to_end_latency = (complete_data["render_time"] - complete_data["write_to_buffer_time"]) * 1000

            print("\n=== Processing Latencies (milliseconds) - Complete Frames Only ====")
            latencies = {
                "Shared Buffer": buffer_latency,
                "Optimization": processing_latency,
                "LED Buffer": led_buffer_latency,
                "LED Transitions": led_transitions_latency,
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
