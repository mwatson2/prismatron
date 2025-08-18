#!/usr/bin/env python3
"""
Test different Ki values with fixed Kp=3, Kd=0.2 for PID control in 24→15fps single-frame case.
Focus on eliminating steady-state error while avoiding integral windup.
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


def run_ki_test():
    """Test different Ki values with fixed Kp=3, Kd=0.2 (full PID control)."""
    print("🎯 PID Ki Tuning Analysis - Fixed Kp=3.0, Kd=0.2")
    print("=" * 70)

    # Test different Ki values with fixed Kp=3, Kd=0.2
    kp_value = 3.0
    kd_value = 0.2
    ki_values = [0.5]  # Final Ki value to test

    # All scenarios to always show complete picture
    scenarios = [
        {"producer_fps": 15.0, "renderer_fps": 15.0, "batch": False, "label": "15→15 Single"},
        {"producer_fps": 15.0, "renderer_fps": 15.0, "batch": True, "label": "15→15 Batch"},
        {"producer_fps": 24.0, "renderer_fps": 15.0, "batch": False, "label": "24→15 Single"},  # FOCUS CASE
        {"producer_fps": 24.0, "renderer_fps": 15.0, "batch": True, "label": "24→15 Batch"},
    ]

    for ki in ki_values:
        print(f"\n🔧 Testing Ki = {ki} (with Kp={kp_value}, Kd={kd_value})")
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

            # Set PID settings with fixed Kp=3, Kd=0.2 and variable Ki
            if scenario["batch"]:
                # Batch mode settings - scale down for batch mode
                sim.frame_dropper.kp = kp_value * 0.5  # Scale down for batch mode
                sim.frame_dropper.ki = ki * 0.5  # Scale down integral for batch mode
                sim.frame_dropper.kd = kd_value * 0.5  # Scale down derivative for batch mode
                sim.frame_dropper.led_buffer_ewma_alpha = 0.01
            else:
                # Single-frame settings - TEST Ki with fixed Kp=3, Kd=0.2
                sim.frame_dropper.kp = kp_value  # Fixed at 3.0
                sim.frame_dropper.ki = ki  # THIS IS WHAT WE'RE TESTING
                sim.frame_dropper.kd = kd_value  # Fixed at 0.2
                sim.frame_dropper.led_buffer_ewma_alpha = 0.05

            # Run simulation to see steady-state behavior and integral accumulation
            sim.run_simulation(20.0)  # Observe integral effects

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

            # Calculate steady-state error metrics
            steady_state_start = int(0.75 * len(buffer_sizes))  # Last 25% for steady state
            if steady_state_start < len(buffer_sizes):
                steady_state_buffer = buffer_sizes[steady_state_start:]
                steady_state_avg = np.mean(steady_state_buffer)
                steady_state_std = np.std(steady_state_buffer)
                steady_state_error = abs(steady_state_avg - 8.0)  # Error from target

                # Check for integral windup (high variance in late stage)
                last_quarter = buffer_sizes[-len(buffer_sizes) // 4 :]
                windup_indicator = np.std(last_quarter) > 2.0
            else:
                steady_state_avg = np.mean(buffer_sizes)
                steady_state_std = np.std(buffer_sizes)
                steady_state_error = abs(steady_state_avg - 8.0)
                windup_indicator = False

            # Get integral term value at end of simulation
            final_integral = sim.frame_dropper.error_integral if hasattr(sim.frame_dropper, "error_integral") else 0

            results.append(
                {
                    "times": times,
                    "buffer_sizes": buffer_sizes,
                    "ewma_levels": ewma_levels,
                    "drop_rates": drop_rates,
                    "label": scenario["label"],
                    "is_focus": scenario["producer_fps"] == 24.0 and not scenario["batch"],
                    "steady_state_avg": steady_state_avg,
                    "steady_state_std": steady_state_std,
                    "steady_state_error": steady_state_error,
                    "windup_indicator": windup_indicator,
                    "final_integral": final_integral,
                }
            )

        # Create 8-subplot figure (4 scenarios × 2 plots each)
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(
            f"PID Controller Tuning (Kp={kp_value}, Ki={ki}, Kd={kd_value})\n🎯 FOCUS: Row 3 (24→15 Single)",
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

            # Add steady-state average line
            ax_buffer.axhline(
                y=result["steady_state_avg"],
                color="orange",
                linestyle=":",
                alpha=0.8,
                label=f'SS Avg ({result["steady_state_avg"]:.1f})',
            )

            # Add title with metrics
            title_prefix = "🎯 " if result["is_focus"] else ""
            ss_text = f" (SS Err={result['steady_state_error']:.2f})"
            windup_text = " ⚠️ WINDUP" if result["windup_indicator"] else ""
            ax_buffer.set_title(
                f'{title_prefix}{result["label"]} - Buffer{ss_text}{windup_text}',
                fontweight="bold" if result["is_focus"] else "normal",
            )

            if result["is_focus"]:
                ax_buffer.patch.set_facecolor("lightyellow")  # Highlight background

            ax_buffer.set_xlabel("Time (s)")
            ax_buffer.set_ylabel("Buffer Size (frames)")
            ax_buffer.grid(True, alpha=0.3)
            ax_buffer.legend(loc="upper right", fontsize=7)
            ax_buffer.set_ylim(-0.5, 13)
            ax_buffer.set_xlim(0, 20)

            # Drop rate plot (right column)
            ax_drop = axes[i, 1]
            ax_drop.plot(
                result["times"],
                result["drop_rates"],
                color=colors[i],
                alpha=0.7,
                linewidth=1,
                label=f"Drop Rate (Ki={ki})",
            )
            if result["is_focus"]:
                ax_drop.axhline(y=37.5, color="red", linestyle="--", alpha=0.7, label="Expected (37.5%)")
                # Calculate actual average drop rate in steady state
                steady_state_start_idx = int(0.75 * len(result["drop_rates"]))
                if steady_state_start_idx < len(result["drop_rates"]):
                    ss_drop_avg = np.mean(result["drop_rates"][steady_state_start_idx:])
                    ax_drop.axhline(
                        y=ss_drop_avg, color="orange", linestyle=":", alpha=0.8, label=f"SS Avg ({ss_drop_avg:.1f}%)"
                    )
                ax_drop.patch.set_facecolor("lightyellow")  # Highlight background
            ax_drop.axhline(y=0, color="black", linestyle="-", alpha=0.5)

            # Add integral value to title
            integral_text = f" (∫={result['final_integral']:.1f})" if ki > 0 else ""
            ax_drop.set_title(
                f'{title_prefix}{result["label"]} - Drop Rate{integral_text}',
                fontweight="bold" if result["is_focus"] else "normal",
            )
            ax_drop.set_xlabel("Time (s)")
            ax_drop.set_ylabel("Drop Rate (%)")
            ax_drop.grid(True, alpha=0.3)
            ax_drop.legend(loc="upper right", fontsize=7)
            ax_drop.set_ylim(-5, 70)
            ax_drop.set_xlim(0, 20)

        # Add analysis text
        focus_result = next(r for r in results if r["is_focus"])

        # Calculate rise time (time to first reach near target)
        target = 8.0
        tolerance = 1.0
        rise_time = 20.0  # Default to max if never reaches
        for idx, (t, size) in enumerate(zip(focus_result["times"], focus_result["buffer_sizes"])):
            if abs(size - target) <= tolerance:
                rise_time = t
                break

        analysis_text = (
            f"Full PID Controller (Kp={kp_value}, Ki={ki}, Kd={kd_value}):\n"
            f"• 24→15 Single steady-state error: {focus_result['steady_state_error']:.2f} frames\n"
            f"• Steady-state average: {focus_result['steady_state_avg']:.1f} frames (target: 8.0)\n"
            f"• Steady-state std dev: {focus_result['steady_state_std']:.2f}\n"
            f"• Rise time to target: {rise_time:.1f}s\n"
            f"• Final integral term: {focus_result['final_integral']:.1f}\n"
            f"• Windup warning: {'YES ⚠️' if focus_result['windup_indicator'] else 'No'}"
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

        # Format Ki value for filename (avoid dots in filename)
        ki_str = f"{ki:.2f}".replace(".", "_")
        plt.savefig(f"ki_tuning_ki_{ki_str}.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Ki={ki} analysis saved as 'ki_tuning_ki_{ki_str}.png'")

        # Print analysis for focus case
        print(
            f"   24→15 Single: SS Error={focus_result['steady_state_error']:.2f}, "
            f"SS Avg={focus_result['steady_state_avg']:.1f}, Rise={rise_time:.1f}s, "
            f"Integral={focus_result['final_integral']:.1f}"
        )


if __name__ == "__main__":
    run_ki_test()
