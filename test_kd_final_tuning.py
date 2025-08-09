#!/usr/bin/env python3
"""
Final Kd tuning with fixed Kp=3, Ki=0.5 for optimal oscillation damping in 24→15fps single-frame case.
Focus on finding the Kd value that eliminates oscillations while maintaining good steady-state tracking.
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


def run_kd_final_test():
    """Test higher Kd values with fixed Kp=3, Ki=0.5 for oscillation damping."""
    print("🎯 Final Kd Tuning for Oscillation Damping - Fixed Kp=3.0, Ki=0.5")
    print("=" * 70)

    # Test higher Kd values with fixed Kp=3, Ki=0.5
    kp_value = 3.0
    ki_value = 0.5
    kd_values = [0.5, 1.0, 1.5, 2.0]  # Higher derivative gains for damping

    # All scenarios to always show complete picture
    scenarios = [
        {"producer_fps": 15.0, "renderer_fps": 15.0, "batch": False, "label": "15→15 Single"},
        {"producer_fps": 15.0, "renderer_fps": 15.0, "batch": True, "label": "15→15 Batch"},
        {"producer_fps": 24.0, "renderer_fps": 15.0, "batch": False, "label": "24→15 Single"},  # FOCUS CASE
        {"producer_fps": 24.0, "renderer_fps": 15.0, "batch": True, "label": "24→15 Batch"},
    ]

    for kd in kd_values:
        print(f"\n🔧 Testing Kd = {kd} (with Kp={kp_value}, Ki={ki_value})")
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

            # Set PID settings with fixed Kp=3, Ki=0.5 and variable Kd
            if scenario["batch"]:
                # Batch mode settings - scale down for batch mode
                sim.frame_dropper.kp = kp_value * 0.5  # Scale down for batch mode
                sim.frame_dropper.ki = ki_value * 0.5  # Scale down integral for batch mode
                sim.frame_dropper.kd = kd * 0.5  # Scale down derivative for batch mode
                sim.frame_dropper.led_buffer_ewma_alpha = 0.01
            else:
                # Single-frame settings - TEST Kd with fixed Kp=3, Ki=0.5
                sim.frame_dropper.kp = kp_value  # Fixed at 3.0
                sim.frame_dropper.ki = ki_value  # Fixed at 0.5
                sim.frame_dropper.kd = kd  # THIS IS WHAT WE'RE TESTING
                sim.frame_dropper.led_buffer_ewma_alpha = 0.05

            # Run simulation to observe oscillation damping
            sim.run_simulation(25.0)  # Long enough to see settling behavior

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

            # Calculate oscillation and stability metrics
            steady_state_start = int(0.6 * len(buffer_sizes))  # Last 40% for steady state analysis
            if steady_state_start < len(buffer_sizes):
                steady_state_buffer = buffer_sizes[steady_state_start:]
                steady_state_avg = np.mean(steady_state_buffer)
                steady_state_std = np.std(steady_state_buffer)
                steady_state_error = abs(steady_state_avg - 8.0)  # Error from target
                oscillation_range = np.max(steady_state_buffer) - np.min(steady_state_buffer)

                # Calculate overshoot (max value above target)
                max_buffer = np.max(buffer_sizes)
                overshoot = max(0, max_buffer - 8.0)

                # Check for good settling (low variance in final 25%)
                final_quarter_start = int(0.75 * len(buffer_sizes))
                final_quarter = buffer_sizes[final_quarter_start:]
                final_std = np.std(final_quarter) if len(final_quarter) > 0 else steady_state_std
                well_settled = final_std < 0.5  # Less than 0.5 frame variation

            else:
                steady_state_avg = np.mean(buffer_sizes)
                steady_state_std = np.std(buffer_sizes)
                steady_state_error = abs(steady_state_avg - 8.0)
                oscillation_range = np.max(buffer_sizes) - np.min(buffer_sizes)
                overshoot = max(0, np.max(buffer_sizes) - 8.0)
                final_std = steady_state_std
                well_settled = final_std < 0.5

            # Get final integral term value
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
                    "oscillation_range": oscillation_range,
                    "overshoot": overshoot,
                    "final_std": final_std,
                    "well_settled": well_settled,
                    "final_integral": final_integral,
                }
            )

        # Create 8-subplot figure (4 scenarios × 2 plots each)
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(
            f"Final PID Tuning (Kp={kp_value}, Ki={ki_value}, Kd={kd})\n🎯 FOCUS: Oscillation Damping (Row 3)",
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

            # Add steady-state average and bounds
            ax_buffer.axhline(
                y=result["steady_state_avg"],
                color="orange",
                linestyle=":",
                alpha=0.8,
                label=f'SS Avg ({result["steady_state_avg"]:.1f})',
            )

            # Highlight settling quality
            settle_color = "lightgreen" if result["well_settled"] else "lightcoral"
            settle_text = "✓" if result["well_settled"] else "⚠️"

            # Add title with key metrics
            title_prefix = "🎯 " if result["is_focus"] else ""
            metrics_text = f" (σ={result['steady_state_std']:.2f}, R={result['oscillation_range']:.1f}) {settle_text}"
            ax_buffer.set_title(
                f'{title_prefix}{result["label"]} - Buffer{metrics_text}',
                fontweight="bold" if result["is_focus"] else "normal",
            )

            if result["is_focus"]:
                ax_buffer.patch.set_facecolor("lightyellow")  # Highlight background

            ax_buffer.set_xlabel("Time (s)")
            ax_buffer.set_ylabel("Buffer Size (frames)")
            ax_buffer.grid(True, alpha=0.3)
            ax_buffer.legend(loc="upper right", fontsize=7)
            ax_buffer.set_ylim(-0.5, 13)
            ax_buffer.set_xlim(0, 25)

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
                # Add steady-state drop rate average
                steady_drop_start = int(0.6 * len(result["drop_rates"]))
                if steady_drop_start < len(result["drop_rates"]):
                    ss_drop_avg = np.mean(result["drop_rates"][steady_drop_start:])
                    ax_drop.axhline(
                        y=ss_drop_avg, color="orange", linestyle=":", alpha=0.8, label=f"SS Drop ({ss_drop_avg:.1f}%)"
                    )
                ax_drop.patch.set_facecolor("lightyellow")  # Highlight background
            ax_drop.axhline(y=0, color="black", linestyle="-", alpha=0.5)

            ax_drop.set_title(
                f'{title_prefix}{result["label"]} - Drop Rate', fontweight="bold" if result["is_focus"] else "normal"
            )
            ax_drop.set_xlabel("Time (s)")
            ax_drop.set_ylabel("Drop Rate (%)")
            ax_drop.grid(True, alpha=0.3)
            ax_drop.legend(loc="upper right", fontsize=7)
            ax_drop.set_ylim(-5, 70)
            ax_drop.set_xlim(0, 25)

        # Add comprehensive analysis text
        focus_result = next(r for r in results if r["is_focus"])

        # Calculate settling time (time when oscillations become < 0.5 frames)
        settling_time = 25.0
        for idx in range(len(focus_result["times"])):
            if idx > len(focus_result["times"]) * 0.2:  # After initial 20%
                # Check if remaining signal has low variance
                remaining = focus_result["buffer_sizes"][idx:]
                if len(remaining) > 10 and np.std(remaining[:10]) < 0.5:
                    settling_time = focus_result["times"][idx]
                    break

        analysis_text = (
            f"Oscillation Damping Analysis (Kp={kp_value}, Ki={ki_value}, Kd={kd}):\n"
            f"• 24→15 Single steady-state error: {focus_result['steady_state_error']:.2f} frames\n"
            f"• Oscillation std dev: {focus_result['steady_state_std']:.2f} frames\n"
            f"• Oscillation range: {focus_result['oscillation_range']:.1f} frames\n"
            f"• Max overshoot: {focus_result['overshoot']:.1f} frames above target\n"
            f"• Settling time: {settling_time:.1f}s\n"
            f"• Well settled: {'YES ✓' if focus_result['well_settled'] else 'NO ⚠️'}\n"
            f"• Final integral: {focus_result['final_integral']:.1f}"
        )

        fig.text(
            0.02,
            0.02,
            analysis_text,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightblue", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.15)

        # Format Kd value for filename (avoid dots in filename)
        kd_str = f"{kd:.1f}".replace(".", "_")
        plt.savefig(f"kd_final_tuning_kd_{kd_str}.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Kd={kd} analysis saved as 'kd_final_tuning_kd_{kd_str}.png'")

        # Print analysis for focus case
        settle_status = "WELL_SETTLED" if focus_result["well_settled"] else "OSCILLATING"
        print(
            f"   24→15 Single: SS_Err={focus_result['steady_state_error']:.2f}, "
            f"Osc_σ={focus_result['steady_state_std']:.2f}, "
            f"Range={focus_result['oscillation_range']:.1f}, "
            f"Overshoot={focus_result['overshoot']:.1f}, "
            f"Status={settle_status}"
        )


if __name__ == "__main__":
    run_kd_final_test()
