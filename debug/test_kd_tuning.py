#!/usr/bin/env python3
"""
Test different Kd values with fixed Kp=3 for PD control in 24→15fps single-frame case.
Focus on finding optimal derivative gain to reduce oscillations.
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


def run_kd_test():
    """Test different Kd values with fixed Kp=3 and Ki=0 (PD control)."""
    print("🎯 PID Kd Tuning Analysis - Fixed Kp=3.0, Ki=0 (PD Control)")
    print("=" * 70)

    # Test different Kd values with fixed Kp=3
    kp_value = 3.0
    kd_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]  # Range of derivative gains

    # All scenarios to always show complete picture
    scenarios = [
        {"producer_fps": 15.0, "renderer_fps": 15.0, "batch": False, "label": "15→15 Single"},
        {"producer_fps": 15.0, "renderer_fps": 15.0, "batch": True, "label": "15→15 Batch"},
        {"producer_fps": 24.0, "renderer_fps": 15.0, "batch": False, "label": "24→15 Single"},  # FOCUS CASE
        {"producer_fps": 24.0, "renderer_fps": 15.0, "batch": True, "label": "24→15 Batch"},
    ]

    for kd in kd_values:
        print(f"\n🔧 Testing Kd = {kd} (with Kp={kp_value})")
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

            # Set PID settings with fixed Kp=3 and variable Kd
            if scenario["batch"]:
                # Batch mode settings - scale down for batch mode
                sim.frame_dropper.kp = kp_value * 0.5  # Scale down for batch mode
                sim.frame_dropper.ki = 0.0  # No integral for PD control
                sim.frame_dropper.kd = kd * 0.5  # Scale down derivative for batch mode
                sim.frame_dropper.led_buffer_ewma_alpha = 0.01
            else:
                # Single-frame settings - TEST Kd with fixed Kp=3
                sim.frame_dropper.kp = kp_value  # Fixed at 3.0
                sim.frame_dropper.ki = 0.0  # PD control - no integral
                sim.frame_dropper.kd = kd  # THIS IS WHAT WE'RE TESTING
                sim.frame_dropper.led_buffer_ewma_alpha = 0.05

            # Run simulation for longer to see steady-state behavior
            sim.run_simulation(30.0)  # Longer to observe oscillations

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

            # Calculate oscillation metrics
            steady_state_start = int(0.5 * len(buffer_sizes))  # Last 50% for steady state
            if steady_state_start < len(buffer_sizes):
                steady_state_buffer = buffer_sizes[steady_state_start:]
                oscillation_std = np.std(steady_state_buffer)
                oscillation_range = np.max(steady_state_buffer) - np.min(steady_state_buffer)
                avg_buffer = np.mean(steady_state_buffer)
            else:
                oscillation_std = 0
                oscillation_range = 0
                avg_buffer = np.mean(buffer_sizes)

            results.append(
                {
                    "times": times,
                    "buffer_sizes": buffer_sizes,
                    "ewma_levels": ewma_levels,
                    "drop_rates": drop_rates,
                    "label": scenario["label"],
                    "is_focus": scenario["producer_fps"] == 24.0 and not scenario["batch"],
                    "oscillation_std": oscillation_std,
                    "oscillation_range": oscillation_range,
                    "avg_buffer": avg_buffer,
                }
            )

        # Create 8-subplot figure (4 scenarios × 2 plots each)
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(
            f"PD Controller Tuning (Kp={kp_value}, Ki=0, Kd={kd})\n🎯 FOCUS: Row 3 (24→15 Single)",
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

            # Add oscillation metrics to title
            title_prefix = "🎯 " if result["is_focus"] else ""
            osc_text = f" (σ={result['oscillation_std']:.2f}, R={result['oscillation_range']:.1f})"
            pid_text = f" Kp={kp_value}, Kd={kd}"
            ax_buffer.set_title(
                f'{title_prefix}{result["label"]} - Buffer{osc_text}',
                fontweight="bold" if result["is_focus"] else "normal",
            )

            if result["is_focus"]:
                ax_buffer.patch.set_facecolor("lightyellow")  # Highlight background

            ax_buffer.set_xlabel("Time (s)")
            ax_buffer.set_ylabel("Buffer Size (frames)")
            ax_buffer.grid(True, alpha=0.3)
            ax_buffer.legend(loc="upper right", fontsize=8)
            ax_buffer.set_ylim(-0.5, 13)
            ax_buffer.set_xlim(0, 30)

            # Drop rate plot (right column)
            ax_drop = axes[i, 1]
            ax_drop.plot(
                result["times"],
                result["drop_rates"],
                color=colors[i],
                alpha=0.7,
                linewidth=1,
                label=f"Drop Rate (Kd={kd})",
            )
            if result["is_focus"]:
                ax_drop.axhline(y=37.5, color="red", linestyle="--", alpha=0.7, label="Expected (37.5%)")
                ax_drop.patch.set_facecolor("lightyellow")  # Highlight background
            ax_drop.axhline(y=0, color="black", linestyle="-", alpha=0.5)

            ax_drop.set_title(
                f'{title_prefix}{result["label"]} - Drop Rate', fontweight="bold" if result["is_focus"] else "normal"
            )
            ax_drop.set_xlabel("Time (s)")
            ax_drop.set_ylabel("Drop Rate (%)")
            ax_drop.grid(True, alpha=0.3)
            ax_drop.legend(loc="upper right", fontsize=8)
            ax_drop.set_ylim(-5, 70)
            ax_drop.set_xlim(0, 30)

        # Add analysis text
        focus_result = next(r for r in results if r["is_focus"])

        # Calculate settling time (time to reach within 10% of target)
        target = 8.0
        tolerance = 1.0  # Within 1 frame of target
        settling_time = 30.0  # Default to max if never settles
        for idx, (t, size) in enumerate(zip(focus_result["times"], focus_result["buffer_sizes"])):
            if abs(size - target) <= tolerance:
                # Check if it stays within tolerance
                remaining = focus_result["buffer_sizes"][idx:]
                if len(remaining) > 10 and all(abs(s - target) <= tolerance * 2 for s in remaining[:10]):
                    settling_time = t
                    break

        analysis_text = (
            f"PD Controller (Kp={kp_value}, Ki=0, Kd={kd}):\n"
            f"• 24→15 Single oscillation std: {focus_result['oscillation_std']:.2f}\n"
            f"• Oscillation range: {focus_result['oscillation_range']:.1f} frames\n"
            f"• Average buffer level: {focus_result['avg_buffer']:.1f} frames\n"
            f"• Settling time: {settling_time:.1f}s\n"
            f"• D term effect at rate=1 frame/s: {kd * 1.0:.2f} → {kd * 33:.1f}% drop change"
        )

        fig.text(
            0.02,
            0.02,
            analysis_text,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgreen", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.15)

        # Format Kd value for filename (avoid dots in filename)
        kd_str = f"{kd:.2f}".replace(".", "_")
        plt.savefig(f"kd_tuning_kd_{kd_str}.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Kd={kd} analysis saved as 'kd_tuning_kd_{kd_str}.png'")

        # Print analysis for focus case
        print(
            f"   24→15 Single: Oscillation σ={focus_result['oscillation_std']:.2f}, Range={focus_result['oscillation_range']:.1f}, Settling={settling_time:.1f}s"
        )


if __name__ == "__main__":
    run_kd_test()
