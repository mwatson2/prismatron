#!/usr/bin/env python3
"""
Test different Kp values for more aggressive PID response in 24→15fps single-frame case.
Focus on the 3rd row, left graph (24→15 Single - LED Buffer Level).
"""

import sys

sys.path.insert(0, "/mnt/dev/prismatron/src")
sys.path.insert(0, "/mnt/dev/prismatron")

import logging

import matplotlib.pyplot as plt
import numpy as np

from frame_drop_simulation import FrameDropSimulation

# Suppress warnings for cleaner output
logging.getLogger().setLevel(logging.WARNING)


def run_kp_test():
    """Test different Kp values and show all 8 cases with focus on 24→15 single."""
    print("🎯 PID Kp Tuning Analysis - Focus on 24→15fps Single-Frame Startup")
    print("=" * 70)

    # Test different Kp values for 24→15 single with P-only control
    kp_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # Test P-only control

    # All scenarios to always show complete picture
    scenarios = [
        {"producer_fps": 15.0, "renderer_fps": 15.0, "batch": False, "label": "15→15 Single"},
        {"producer_fps": 15.0, "renderer_fps": 15.0, "batch": True, "label": "15→15 Batch"},
        {"producer_fps": 24.0, "renderer_fps": 15.0, "batch": False, "label": "24→15 Single"},  # FOCUS CASE
        {"producer_fps": 24.0, "renderer_fps": 15.0, "batch": True, "label": "24→15 Batch"},
    ]

    for kp in kp_values:
        print(f"\n🔧 Testing Kp = {kp}")
        results = []

        for scenario in scenarios:
            print(f"  Running {scenario['label']}...")

            sim = FrameDropSimulation(
                producer_fps=scenario["producer_fps"],
                renderer_fps=scenario["renderer_fps"],
                buffer_capacity=12,
                target_buffer_level=8,
                use_pid_controller=True,
                batch_optimization=scenario["batch"],
                batch_size=8,
            )

            # Set PID settings with variable Kp and zero Ki/Kd for P-only control
            if scenario["batch"]:
                # Batch mode settings - P-only
                sim.frame_dropper.kp = kp * 0.5  # Scale down for batch mode
                sim.frame_dropper.ki = 0.0  # P-only control
                sim.frame_dropper.kd = 0.0  # P-only control
                sim.frame_dropper.led_buffer_ewma_alpha = 0.01
            else:
                # Single-frame settings - TEST Kp with P-only control
                sim.frame_dropper.kp = kp  # THIS IS WHAT WE'RE TESTING
                sim.frame_dropper.ki = 0.0  # P-only control - no integral suppression
                sim.frame_dropper.kd = 0.0  # P-only control - no derivative suppression
                sim.frame_dropper.led_buffer_ewma_alpha = 0.05

            # Run simulation
            sim.run_simulation(15.0)  # Shorter, focused on startup

            # Extract data
            times = np.array([t for t, _ in sim.buffer_size_history])
            buffer_sizes = np.array([s for _, s in sim.buffer_size_history])

            # Get EWMA data (approximate)
            ewma_alpha = sim.frame_dropper.led_buffer_ewma_alpha
            ewma_levels = []
            ewma_weighted_sum = 0.0
            ewma_total_weight = 0.0

            for buffer_size in buffer_sizes:
                ewma_weighted_sum = (1 - ewma_alpha) * ewma_weighted_sum + ewma_alpha * buffer_size
                ewma_total_weight = (1 - ewma_alpha) * ewma_total_weight + ewma_alpha

                if ewma_total_weight > 0:
                    ewma_value = ewma_weighted_sum / ewma_total_weight
                else:
                    ewma_value = buffer_size
                ewma_levels.append(ewma_value)

            ewma_levels = np.array(ewma_levels)

            # Get drop rates
            drop_rate_dict = dict(sim.drop_rate_history)
            drop_rates = []
            for time in times:
                closest_time = min(drop_rate_dict.keys(), key=lambda x: abs(x - time), default=time)
                drop_rates.append(drop_rate_dict.get(closest_time, 0.0) * 100)
            drop_rates = np.array(drop_rates)

            results.append(
                {
                    "times": times,
                    "buffer_sizes": buffer_sizes,
                    "ewma_levels": ewma_levels,
                    "drop_rates": drop_rates,
                    "label": scenario["label"],
                    "is_focus": scenario["producer_fps"] == 24.0 and not scenario["batch"],
                }
            )

        # Create 8-subplot figure (4 scenarios × 2 plots each)
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(
            f"P-Only Controller Kp Tuning (Kp = {kp}, Ki=0, Kd=0)\n🎯 FOCUS: Row 3, Left Graph (24→15 Single Buffer Level)",
            fontsize=16,
            fontweight="bold",
        )

        colors = ["blue", "green", "red", "orange"]

        for i, result in enumerate(results):
            # Buffer level plot (left column)
            ax_buffer = axes[i, 0]
            ax_buffer.plot(
                result["times"], result["buffer_sizes"], color=colors[i], alpha=0.7, linewidth=1, label="Actual Buffer"
            )
            ax_buffer.plot(
                result["times"], result["ewma_levels"], color="purple", alpha=0.8, linewidth=2, label="EWMA Buffer"
            )
            ax_buffer.axhline(y=8, color="green", linestyle="--", alpha=0.7, label="Target (8)")
            ax_buffer.axhline(y=12, color="red", linestyle="--", alpha=0.7, label="Capacity (12)")

            # Highlight focus case
            title_prefix = "🎯 FOCUS: " if result["is_focus"] else ""
            # Include PID gains in the title
            pid_text = f"(Kp={kp}, Ki=0, Kd=0)"
            ax_buffer.set_title(
                f'{title_prefix}{result["label"]} - Buffer Level {pid_text}',
                fontweight="bold" if result["is_focus"] else "normal",
            )

            if result["is_focus"]:
                ax_buffer.patch.set_facecolor("lightyellow")  # Highlight background

            ax_buffer.set_xlabel("Time (s)")
            ax_buffer.set_ylabel("Buffer Size (frames)")
            ax_buffer.grid(True, alpha=0.3)
            ax_buffer.legend()
            ax_buffer.set_ylim(-0.5, 13)
            ax_buffer.set_xlim(0, 15)  # Focus on startup

            # Drop rate plot (right column)
            ax_drop = axes[i, 1]
            # Include PID gains in the drop rate label
            ax_drop.plot(
                result["times"],
                result["drop_rates"],
                color=colors[i],
                alpha=0.7,
                linewidth=1,
                label=f"Drop Rate (Kp={kp})",
            )
            if result["is_focus"]:
                ax_drop.axhline(y=37.5, color="red", linestyle="--", alpha=0.7, label="Expected (37.5%)")
                ax_drop.patch.set_facecolor("lightyellow")  # Highlight background
            ax_drop.axhline(y=0, color="black", linestyle="-", alpha=0.5)

            ax_drop.set_title(
                f'{title_prefix}{result["label"]} - Drop Rate {pid_text}',
                fontweight="bold" if result["is_focus"] else "normal",
            )
            ax_drop.set_xlabel("Time (s)")
            ax_drop.set_ylabel("Drop Rate (%)")
            ax_drop.grid(True, alpha=0.3)
            ax_drop.legend()  # Always show legend with PID gains
            ax_drop.set_ylim(-5, 70)
            ax_drop.set_xlim(0, 15)  # Focus on startup

        # Add analysis text
        focus_result = next(r for r in results if r["is_focus"])
        startup_mask = focus_result["times"] <= 5.0
        if np.any(startup_mask):
            startup_drop = focus_result["drop_rates"][startup_mask]
            avg_startup_drop = np.mean(startup_drop) if len(startup_drop) > 0 else 0

            # Find time to reach target buffer (buffer > 7)
            target_mask = focus_result["buffer_sizes"] >= 7
            if np.any(target_mask):
                time_to_target = (
                    focus_result["times"][target_mask][0] if len(focus_result["times"][target_mask]) > 0 else 20
                )
            else:
                time_to_target = 20

            analysis_text = (
                f"P-Only Controller (Kp = {kp}, Ki=0, Kd=0):\n"
                f"• 24→15 Single startup drop rate: {avg_startup_drop:.1f}% (target: 37.5%)\n"
                f"• Time to reach buffer target: {time_to_target:.1f}s\n"
                f"• P term at buffer=0: {kp * (-8):.1f} → ~{kp * 33:.1f}% expected drop rate\n"
                f"• 🎯 Look at Row 3, Left Graph (highlighted yellow)"
            )
        else:
            analysis_text = f"Kp = {kp} - Analysis pending"

        fig.text(
            0.02,
            0.02,
            analysis_text,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgreen", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.15)
        plt.savefig(f"kp_tuning_kp_{kp:.0f}.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Kp={kp} analysis saved as 'kp_tuning_kp_{kp:.0f}.png'")

        # Print startup analysis for focus case
        if np.any(startup_mask):
            print(
                f"   24→15 Single: Avg startup drop rate: {avg_startup_drop:.1f}%, Time to target: {time_to_target:.1f}s"
            )


if __name__ == "__main__":
    run_kp_test()
