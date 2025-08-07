#!/usr/bin/env python3
"""
Frame Drop Simulation for Adaptive Frame Dropper Testing.

This simulation models the complete frame pipeline:
1. Producer generates frames at specified fps
2. Frames go through optimization (duration based on renderer fps)
3. Frames enter LED buffer and are rendered at target timestamps
4. Adaptive frame dropper controls drop rate based on buffer level
5. Both controllers wait for buffer to fill before activating
6. Tracks adaptive drops, late drops, and buffer dynamics
"""

import heapq
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.consumer.adaptive_frame_dropper import AdaptiveFrameDropper

# Set up logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of simulation events."""

    CONSUMER_LOOP = "consumer_loop"
    OPTIMIZATION_COMPLETE = "optimization_complete"
    RENDERER_LOOP = "renderer_loop"


@dataclass
class Frame:
    """Represents a frame in the simulation."""

    frame_id: int
    target_timestamp: float  # When frame should be presented
    produced_at: float
    optimization_started: Optional[float] = None
    optimization_completed: Optional[float] = None
    buffer_enter_time: Optional[float] = None
    render_scheduled_time: Optional[float] = None
    rendered_at: Optional[float] = None


@dataclass
class SimulationEvent:
    """Simulation event for priority queue."""

    timestamp: float
    event_type: EventType
    frame: Optional[Frame] = None
    data: Optional[dict] = None

    def __lt__(self, other):
        """Required for heapq comparison."""
        return self.timestamp < other.timestamp


class FrameDropSimulation:
    """
    Event-driven simulation of the frame dropping system.
    """

    def __init__(
        self,
        producer_fps: float = 24.0,
        renderer_fps: float = 15.0,
        buffer_capacity: int = 12,
        target_buffer_level: int = 10,
        use_pid_controller: bool = False,
    ):
        """
        Initialize the simulation.

        Args:
            producer_fps: Frames per second from producer
            renderer_fps: Target frames per second for optimization
            buffer_capacity: LED buffer capacity (max frames)
            target_buffer_level: Target buffer level for PID controller
            use_pid_controller: Whether to use PID controller (True) or legacy (False)
        """
        self.producer_fps = producer_fps
        self.renderer_fps = renderer_fps
        self.buffer_capacity = buffer_capacity
        self.target_buffer_level = target_buffer_level
        self.use_pid_controller = use_pid_controller

        # Event queue
        self.event_queue: List[SimulationEvent] = []

        # System state
        self.current_time = 0.0
        self.led_buffer: List[Frame] = []
        self.optimization_queue: List[Frame] = []
        self.shared_buffer: List[Frame] = []  # Producer-consumer shared buffer (3 frames max)
        self.shared_buffer_capacity = 3
        self.next_frame_id = 0

        # Renderer state
        self.renderer_state = "WAITING"
        self.renderer_clock_started = False
        self.renderer_clock_start_time = 0.0
        self.target_frame_interval = 1.0 / self.producer_fps  # Frame interval based on producer fps

        # Optimization timing
        self.optimization_duration = 1.0 / self.renderer_fps

        # Initialize adaptive frame dropper with balanced EWMA
        if use_pid_controller:
            self.frame_dropper = AdaptiveFrameDropper(
                led_buffer_capacity=buffer_capacity,
                led_buffer_ewma_alpha=0.03,  # Balanced EWMA - responsive but still smoothing
                max_drop_rate=0.66,
                use_pid_controller=True,
                kp=1.0,  # Start with Kp-only
                ki=0.0,  # Disable integral term
                kd=0.0,  # Disable derivative term
                target_buffer_level=target_buffer_level,  # Pass target buffer level to constructor
            )
        else:
            self.frame_dropper = AdaptiveFrameDropper(
                led_buffer_capacity=buffer_capacity,
                led_buffer_ewma_alpha=0.03,  # Balanced EWMA - responsive but still smoothing
                max_drop_rate=0.66,
                use_pid_controller=False,
                target_buffer_level=target_buffer_level,  # Pass target buffer level for consistency
            )

        # Statistics collection
        self.buffer_size_history: List[Tuple[float, int]] = []
        self.drop_rate_history: List[Tuple[float, float]] = []
        self.ewma_buffer_history: List[Tuple[float, float]] = []
        self.rendered_frames: List[Frame] = []
        self.dropped_frame_ids: List[int] = []
        self.late_dropped_frames: List[Frame] = []

        # Drop rate tracking
        self.actual_drop_rates: List[Tuple[float, float, float, float, float]] = (
            []
        )  # (time, adaptive_rate, pre_opt_rate, late_rate, total_rate)

        # Additional tracking
        self.frames_processed_by_dropper = 0
        self.frames_dropped_by_dropper = 0
        self.frames_dropped_pre_optimization = 0  # Dropped before optimization due to lateness

        # Utilization tracking
        self.consumer_wait_times: List[float] = []  # Track individual wait times
        self.total_wait_time = 0.0  # Total time spent waiting for LED buffer space
        self.wait_time_ewma = 0.0  # EWMA of wait time (utilization penalty indicator)
        self.wait_time_ewma_alpha = 0.05  # EWMA coefficient for wait time

        # Consumer state tracking
        self.consumer_state_log: List[Tuple[float, str, dict]] = []  # (time, state, data)
        self.last_logged_time = 0.0
        self.last_consumer_loop_time = 0.0

        logger.info(
            f"Simulation initialized: Producer={producer_fps}fps, "
            f"Renderer={renderer_fps}fps, Buffer={buffer_capacity}, "
            f"Controller={'PID' if use_pid_controller else 'Legacy'}"
        )

    def schedule_event(self, timestamp: float, event_type: EventType, frame: Frame = None):
        """Schedule a simulation event."""
        event = SimulationEvent(timestamp, event_type, frame)
        heapq.heappush(self.event_queue, event)

    def log_consumer_state(self, state: str, start_time: float, data: dict):
        """Log consumer loop state changes with timing and buffer information."""
        duration = self.current_time - start_time
        log_entry = (self.current_time, state, {**data, "duration_ms": duration * 1000})
        self.consumer_state_log.append(log_entry)

        # Log to console only in steady state (after t=50s) for PID controller
        if (
            self.use_pid_controller and self.current_time > 50.0 and self.current_time - self.last_logged_time > 0.1
        ):  # Log every 100ms in steady state
            print(f"t={self.current_time:.3f}s: {state} - {data} (duration: {duration*1000:.1f}ms)")
            self.last_logged_time = self.current_time

    def try_produce_frame(self):
        """Producer tries to produce a frame if shared buffer has space."""
        if len(self.shared_buffer) >= self.shared_buffer_capacity:
            return None  # Producer blocked

        # Producer creates frame with timestamp at target rate
        frame = Frame(
            frame_id=self.next_frame_id,
            target_timestamp=self.next_frame_id * self.target_frame_interval,
            produced_at=self.current_time,
        )
        self.next_frame_id += 1

        # Add to shared buffer
        self.shared_buffer.append(frame)
        logger.debug(
            f"Producer created frame {frame.frame_id} with timestamp {frame.target_timestamp:.3f}s "
            f"(shared buffer: {len(self.shared_buffer)}/{self.shared_buffer_capacity})"
        )
        return frame

    def handle_consumer_loop(self):
        """Consumer loop: pull → check drops → optimize → push to LED buffer → repeat."""
        loop_start_time = self.current_time

        # Calculate gap since last consumer loop call (before updating last_consumer_loop_time)
        gap_since_last_loop = (
            self.current_time - self.last_consumer_loop_time if self.last_consumer_loop_time > 0 else 0
        )

        # Log ALL consumer loop entries in steady state to debug missing calls
        if self.use_pid_controller and self.current_time > 50.0:
            print(
                f"t={self.current_time:.3f}s: CONSUMER_LOOP_START - gap_since_last={gap_since_last_loop*1000:.1f}ms, shared_buffer={len(self.shared_buffer)}, led_buffer={len(self.led_buffer)}, optimization_queue={len(self.optimization_queue)}"
            )

        # Update last consumer loop time AFTER logging
        self.last_consumer_loop_time = self.current_time

        # Producer tries to fill shared buffer (runs as fast as possible)
        while len(self.shared_buffer) < self.shared_buffer_capacity:
            produced_frame = self.try_produce_frame()
            if produced_frame is None:
                break  # Shared buffer full

        # Consumer pulls frame from shared buffer
        if not self.shared_buffer:
            # No frames available, try again soon
            self.log_consumer_state(
                "IDLE_NO_FRAMES", loop_start_time, {"shared_buffer": 0, "led_buffer": len(self.led_buffer)}
            )
            self.schedule_event(self.current_time + 0.001, EventType.CONSUMER_LOOP)
            return

        frame = self.shared_buffer.pop(0)
        logger.debug(f"Consumer pulled frame {frame.frame_id} (timestamp {frame.target_timestamp:.3f}s)")

        # Only run frame droppers if renderer is PLAYING
        should_drop_adaptive = False
        should_drop_late = False

        if self.renderer_state.upper() == "PLAYING":
            # Run adaptive frame dropper decision
            should_drop_adaptive = self.frame_dropper.should_drop_frame(
                self.current_time, len(self.led_buffer), self.renderer_state, self.wait_time_ewma
            )

            # Track dropper statistics
            self.frames_processed_by_dropper += 1

            if not should_drop_adaptive:
                # Check if frame will be late even after optimization completes
                target_render_time = self.renderer_clock_start_time + frame.target_timestamp
                late_threshold = 0.050  # 50ms threshold like production

                # Check if frame will be late AFTER optimization completes
                optimization_complete_time = self.current_time + self.optimization_duration

                if optimization_complete_time > target_render_time + late_threshold:
                    should_drop_late = True

        # Handle frame based on drop decisions
        if should_drop_adaptive:
            # Frame dropped by adaptive algorithm
            self.dropped_frame_ids.append(frame.frame_id)
            self.frames_dropped_by_dropper += 1
            self.log_consumer_state(
                "ADAPTIVE_DROP",
                loop_start_time,
                {
                    "frame_id": frame.frame_id,
                    "led_buffer": len(self.led_buffer),
                    "target_drop_rate": self.frame_dropper.target_drop_rate,
                },
            )

            # Log frame drop in steady state
            if self.use_pid_controller and self.current_time > 50.0:
                print(
                    f"t={self.current_time:.3f}s: ADAPTIVE_DROP - frame_id={frame.frame_id}, target_drop_rate={self.frame_dropper.target_drop_rate*100:.1f}%, led_buffer={len(self.led_buffer)}"
                )

            # Immediately schedule next consumer loop iteration
            self.schedule_event(self.current_time, EventType.CONSUMER_LOOP)

        elif should_drop_late:
            # Frame will be late even after optimization - drop it before optimization
            self.dropped_frame_ids.append(frame.frame_id)
            self.frames_dropped_pre_optimization += 1
            self.log_consumer_state(
                "LATE_DROP",
                loop_start_time,
                {
                    "frame_id": frame.frame_id,
                    "led_buffer": len(self.led_buffer),
                    "target_render_time": target_render_time,
                    "optimization_complete_time": optimization_complete_time,
                },
            )

            # Log late drop in steady state
            if self.use_pid_controller and self.current_time > 50.0:
                print(
                    f"t={self.current_time:.3f}s: LATE_DROP - frame_id={frame.frame_id}, led_buffer={len(self.led_buffer)}"
                )

            # Immediately schedule next consumer loop iteration
            self.schedule_event(self.current_time, EventType.CONSUMER_LOOP)

        else:
            # Start optimization task (blocks consumer until complete)
            frame.optimization_started = self.current_time
            optimization_complete_time = self.current_time + self.optimization_duration
            self.log_consumer_state(
                "START_OPTIMIZATION",
                loop_start_time,
                {
                    "frame_id": frame.frame_id,
                    "led_buffer": len(self.led_buffer),
                    "optimization_duration": self.optimization_duration,
                },
            )

            # Log optimization start in steady state
            if self.use_pid_controller and self.current_time > 50.0:
                print(
                    f"t={self.current_time:.3f}s: PUSHING_TO_OPTIMIZATION - frame_id={frame.frame_id}, will_complete_at={optimization_complete_time:.3f}s, led_buffer={len(self.led_buffer)}"
                )

            self.schedule_event(optimization_complete_time, EventType.OPTIMIZATION_COMPLETE, frame)
            self.optimization_queue.append(frame)
            logger.debug(
                f"Frame {frame.frame_id} started optimization (will complete at {optimization_complete_time:.3f}s)"
            )
            # Consumer is BLOCKED until optimization completes - don't schedule next loop iteration

    def handle_optimization_complete(self, frame: Frame):
        """Handle optimization task completion and resume consumer loop."""
        optimization_complete_start = self.current_time
        frame.optimization_completed = self.current_time

        # Remove from optimization queue
        if frame in self.optimization_queue:
            self.optimization_queue.remove(frame)

        # Try to add to LED buffer - block if no space (backpressure)
        if len(self.led_buffer) >= self.buffer_capacity:
            # LED buffer full - consumer must wait (utilization penalty)

            # Track wait time if this is the first wait attempt for this frame
            if not hasattr(frame, "wait_start_time"):
                frame.wait_start_time = self.current_time
                self.log_consumer_state(
                    "START_WAITING",
                    optimization_complete_start,
                    {
                        "frame_id": frame.frame_id,
                        "led_buffer": len(self.led_buffer),
                        "buffer_capacity": self.buffer_capacity,
                    },
                )
            else:
                self.log_consumer_state(
                    "CONTINUE_WAITING",
                    optimization_complete_start,
                    {
                        "frame_id": frame.frame_id,
                        "led_buffer": len(self.led_buffer),
                        "wait_duration": self.current_time - frame.wait_start_time,
                    },
                )

            # Reschedule optimization complete - consumer blocked
            self.schedule_event(self.current_time + 0.001, EventType.OPTIMIZATION_COMPLETE, frame)
            return

        # Buffer space available - frame can proceed
        if hasattr(frame, "wait_start_time"):
            # This frame was waiting - record wait time
            wait_duration = self.current_time - frame.wait_start_time
            self.consumer_wait_times.append(wait_duration)
            self.total_wait_time += wait_duration

            # Update wait time EWMA (utilization penalty indicator)
            if self.wait_time_ewma == 0.0:
                self.wait_time_ewma = wait_duration
            else:
                self.wait_time_ewma = (
                    1 - self.wait_time_ewma_alpha
                ) * self.wait_time_ewma + self.wait_time_ewma_alpha * wait_duration

            self.log_consumer_state(
                "END_WAITING",
                optimization_complete_start,
                {
                    "frame_id": frame.frame_id,
                    "led_buffer": len(self.led_buffer),
                    "wait_duration": wait_duration,
                    "wait_time_ewma": self.wait_time_ewma,
                },
            )
            delattr(frame, "wait_start_time")  # Clean up
        else:
            # No waiting required
            self.log_consumer_state(
                "BUFFER_AVAILABLE",
                optimization_complete_start,
                {"frame_id": frame.frame_id, "led_buffer": len(self.led_buffer)},
            )

        # Add to LED buffer
        frame.buffer_enter_time = self.current_time
        self.led_buffer.append(frame)

        # Always log buffer addition in steady state (not filtered by time)
        if self.use_pid_controller and self.current_time > 50.0:
            print(
                f"t={self.current_time:.3f}s: ADDED_TO_LED_BUFFER - frame_id={frame.frame_id}, buffer_size={len(self.led_buffer)}, timestamp={frame.target_timestamp:.3f}s"
            )

        logger.debug(
            f"Frame {frame.frame_id} added to LED buffer (size: {len(self.led_buffer)}) "
            f"with original timestamp {frame.target_timestamp:.3f}s"
        )

        # Check if buffer reached target level for the first time
        if len(self.led_buffer) >= self.target_buffer_level and not self.renderer_clock_started:
            self.start_renderer()

        # Consumer loop can continue - optimization complete unblocks consumer
        if self.use_pid_controller and self.current_time > 50.0:
            print(
                f"t={self.current_time:.3f}s: OPTIMIZATION_COMPLETE - frame_id={frame.frame_id}, scheduling_consumer_loop"
            )
        self.schedule_event(self.current_time, EventType.CONSUMER_LOOP)

    def start_renderer(self):
        """Start the renderer when buffer first becomes full."""
        self.renderer_state = "PLAYING"
        self.renderer_clock_started = True
        self.renderer_clock_start_time = self.current_time

        logger.info(
            f"Renderer STARTED at t={self.current_time:.3f}s (buffer reached target level {self.target_buffer_level})"
        )
        logger.info(
            f"Buffer size: {len(self.led_buffer)}, First frame timestamp: {self.led_buffer[0].target_timestamp:.3f}s"
        )

        # Start renderer loop immediately
        self.schedule_event(self.current_time, EventType.RENDERER_LOOP)

    def handle_renderer_loop(self):
        """Renderer loop: pull from LED buffer → check late → either drop+continue or wait+render → repeat."""
        if not self.renderer_clock_started:
            return

        if not self.led_buffer:
            # No frames to render, check again soon
            self.schedule_event(self.current_time + 0.001, EventType.RENDERER_LOOP)
            return

        # Get next frame to render
        frame = self.led_buffer[0]

        # Calculate actual render time: renderer_clock_start + original_timestamp
        actual_render_time = self.renderer_clock_start_time + frame.target_timestamp

        # Check if frame is late
        if self.current_time > actual_render_time + 0.005:  # 5ms tolerance like production
            # Frame is late - drop it and immediately continue loop
            # IMPORTANT: This late drop is NOT fed back to adaptive dropper (matches production)
            late_frame = self.led_buffer.pop(0)
            self.late_dropped_frames.append(late_frame)
            logger.debug(
                f"Frame {late_frame.frame_id} DROPPED (late in renderer) - "
                f"should render at {actual_render_time:.3f}s, current time {self.current_time:.3f}s"
            )

            # Immediately continue renderer loop (no wait)
            self.schedule_event(self.current_time, EventType.RENDERER_LOOP)

        else:
            # Frame is on time - wait until render time, then render
            if self.current_time < actual_render_time:
                # Wait until render time
                self.schedule_event(actual_render_time, EventType.RENDERER_LOOP)
                logger.debug(
                    f"Waiting to render frame {frame.frame_id} at {actual_render_time:.3f}s "
                    f"(current time {self.current_time:.3f}s)"
                )
            else:
                # Render now
                frame = self.led_buffer.pop(0)
                frame.rendered_at = self.current_time
                self.rendered_frames.append(frame)
                logger.debug(f"Rendered frame {frame.frame_id} at t={self.current_time:.3f}")

                # Immediately continue renderer loop
                self.schedule_event(self.current_time, EventType.RENDERER_LOOP)

    def record_system_state(self):
        """Record current system state for analysis."""
        # Record instantaneous buffer size
        self.buffer_size_history.append((self.current_time, len(self.led_buffer)))

        # Get frame dropper statistics
        stats = self.frame_dropper.get_stats()
        self.drop_rate_history.append((self.current_time, stats["target_drop_rate"]))
        self.ewma_buffer_history.append((self.current_time, stats["led_buffer_level_ewma"]))

        # Calculate actual drop rates
        total_produced = self.next_frame_id
        if total_produced > 0:
            adaptive_rate = self.frames_dropped_by_dropper / total_produced
            pre_opt_rate = self.frames_dropped_pre_optimization / total_produced
            late_rate = len(self.late_dropped_frames) / total_produced
            total_rate = adaptive_rate + pre_opt_rate + late_rate

            self.actual_drop_rates.append((self.current_time, adaptive_rate, pre_opt_rate, late_rate, total_rate))

            # Debug PID behavior in steady state (every 10 seconds)
            if int(self.current_time) % 10 == 0 and abs(self.current_time - int(self.current_time)) < 0.1:
                if self.use_pid_controller:
                    pid_stats = self.frame_dropper.get_stats()["pid_state"]
                    print(
                        f"t={self.current_time:.1f}s: Buffer={len(self.led_buffer)}, EWMA={stats['led_buffer_level_ewma']:.1f}, "
                        f"Target={stats['target_drop_rate']*100:.1f}%, Actual={total_rate*100:.1f}%, "
                        f"Integral={pid_stats['error_integral']:.2f}"
                    )

    def run_simulation(self, duration_seconds: float = 120.0):
        """Run the simulation for the specified duration."""
        controller_name = "PID" if self.use_pid_controller else "Legacy"
        logger.info(f"Starting {controller_name} simulation for {duration_seconds}s...")

        # Schedule initial consumer loop
        self.schedule_event(0.0, EventType.CONSUMER_LOOP)

        event_count = 0
        last_log_time = 0

        while self.current_time < duration_seconds and self.event_queue:
            # Get next chronological event
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp

            # Process event
            if event.event_type == EventType.CONSUMER_LOOP:
                self.handle_consumer_loop()
            elif event.event_type == EventType.OPTIMIZATION_COMPLETE:
                self.handle_optimization_complete(event.frame)
            elif event.event_type == EventType.RENDERER_LOOP:
                self.handle_renderer_loop()

            # Record system state
            self.record_system_state()

            event_count += 1

            # Progress logging every 10 seconds
            if self.current_time - last_log_time > 10:
                total_dropped = self.frames_dropped_by_dropper + self.frames_dropped_pre_optimization
                total_drop_rate = (total_dropped / max(1, self.next_frame_id)) * 100
                logger.info(
                    f"t={self.current_time:.0f}s: Produced={self.next_frame_id}, "
                    f"Adaptive={self.frames_dropped_by_dropper}, "
                    f"PreOpt={self.frames_dropped_pre_optimization}, "
                    f"Total={total_dropped} ({total_drop_rate:.1f}%), "
                    f"Buffer={len(self.led_buffer)}"
                )
                last_log_time = self.current_time

        logger.info(f"{controller_name} simulation completed: {event_count} events processed")
        self._print_summary()

    def _print_summary(self):
        """Print detailed simulation summary."""
        controller_name = "PID" if self.use_pid_controller else "Legacy"
        total_produced = self.next_frame_id
        total_adaptive_dropped = self.frames_dropped_by_dropper
        total_pre_opt_dropped = self.frames_dropped_pre_optimization
        total_late_dropped = len(self.late_dropped_frames)
        total_rendered = len(self.rendered_frames)

        adaptive_drop_rate = (total_adaptive_dropped / max(1, total_produced)) * 100
        pre_opt_drop_rate = (total_pre_opt_dropped / max(1, total_produced)) * 100
        late_drop_rate = (total_late_dropped / max(1, total_produced)) * 100
        total_drop_rate = (
            (total_adaptive_dropped + total_pre_opt_dropped + total_late_dropped) / max(1, total_produced) * 100
        )

        logger.info(f"\n=== {controller_name} SIMULATION SUMMARY ===")
        logger.info(f"Duration: {self.current_time:.1f}s")
        logger.info(f"Renderer state: {self.renderer_state}")
        logger.info(f"Frames produced: {total_produced}")
        logger.info(f"Adaptive drops: {total_adaptive_dropped} ({adaptive_drop_rate:.1f}%)")
        logger.info(f"Pre-optimization drops: {total_pre_opt_dropped} ({pre_opt_drop_rate:.1f}%)")
        logger.info(f"Late drops (renderer): {total_late_dropped} ({late_drop_rate:.1f}%)")
        logger.info(
            f"Total drops: {total_adaptive_dropped + total_pre_opt_dropped + total_late_dropped} ({total_drop_rate:.1f}%)"
        )
        logger.info(f"Rendered: {total_rendered}")

        if self.renderer_clock_started:
            logger.info(f"Renderer started at t={self.renderer_clock_start_time:.3f}s")

        # Buffer statistics
        if self.buffer_size_history:
            buffer_sizes = [size for _, size in self.buffer_size_history]
            logger.info(
                f"Buffer stats: avg={np.mean(buffer_sizes):.2f}, "
                f"std={np.std(buffer_sizes):.2f}, "
                f"min={min(buffer_sizes)}, max={max(buffer_sizes)}"
            )


def tune_kp_controller(kp_values=[0.5, 1.0, 2.0], producer_fps=24.0, renderer_fps=15.0, duration=60.0):
    """Test different Kp values for PID controller tuning."""
    print("🎯 PID Controller Kp Tuning")
    print(f"Producer: {producer_fps}fps, Renderer: {renderer_fps}fps, Duration: {duration}s")
    print(f"Buffer: capacity=12, target=10")
    print("=" * 70)

    results = {}

    for kp in kp_values:
        print(f"\nTesting Kp = {kp}...")
        sim = FrameDropSimulation(
            producer_fps=producer_fps,
            renderer_fps=renderer_fps,
            buffer_capacity=12,
            target_buffer_level=10,
            use_pid_controller=True,
        )
        # Update Kp value
        sim.frame_dropper.kp = kp
        sim.run_simulation(duration)
        results[kp] = sim

        # Quick summary
        buffer_sizes = [size for _, size in sim.buffer_size_history]
        if buffer_sizes:
            buffer_std = np.std(buffer_sizes)
            buffer_avg = np.mean(buffer_sizes)
            target_error = abs(buffer_avg - 10.0)
            print(f"Kp={kp}: Avg buffer={buffer_avg:.2f}, Std={buffer_std:.2f}, Target error={target_error:.2f}")

    # Generate comparison plots
    plot_kp_comparison(results, producer_fps, renderer_fps)

    return results


def tune_ki_controller(ki_values=[0.01, 0.02, 0.05, 0.1], kp=1.0, producer_fps=24.0, renderer_fps=15.0, duration=120.0):
    """Test different Ki values with fixed Kp for steady-state error elimination."""
    print("🎯 PID Controller Ki Tuning (Kp=1.0, Kd=0)")
    print(f"Producer: {producer_fps}fps, Renderer: {renderer_fps}fps, Duration: {duration}s")
    print(f"Buffer: capacity=12, target=10")
    print("=" * 70)

    results = {}

    for ki in ki_values:
        print(f"\nTesting Ki = {ki} (Kp={kp}, Kd=0)...")
        sim = FrameDropSimulation(
            producer_fps=producer_fps,
            renderer_fps=renderer_fps,
            buffer_capacity=12,
            target_buffer_level=10,
            use_pid_controller=True,
        )
        # Update PID gains
        sim.frame_dropper.kp = kp
        sim.frame_dropper.ki = ki
        sim.frame_dropper.kd = 0.0
        sim.run_simulation(duration)
        results[ki] = sim

        # Quick summary
        buffer_sizes = [size for _, size in sim.buffer_size_history]
        if buffer_sizes:
            buffer_std = np.std(buffer_sizes)
            buffer_avg = np.mean(buffer_sizes)
            target_error = abs(buffer_avg - 10.0)
            # Calculate steady-state error (last 25% of simulation)
            steady_state_start = int(0.75 * len(buffer_sizes))
            steady_state_sizes = buffer_sizes[steady_state_start:]
            steady_state_avg = np.mean(steady_state_sizes) if steady_state_sizes else buffer_avg
            steady_state_error = abs(steady_state_avg - 10.0)

            print(
                f"Ki={ki}: Avg buffer={buffer_avg:.2f}, Std={buffer_std:.2f}, "
                f"Target error={target_error:.2f}, SS error={steady_state_error:.2f}"
            )

    # Generate comparison plots
    plot_ki_comparison(results, kp, producer_fps, renderer_fps)

    return results


def tune_kd_controller(
    kd_values=[0.0, 0.01, 0.02, 0.05, 0.1], kp=1.0, ki=0.3, producer_fps=24.0, renderer_fps=15.0, duration=120.0
):
    """Test different Kd values with fixed Kp and Ki for oscillation reduction."""
    print(f"🎯 PID Controller Kd Tuning (Kp={kp}, Ki={ki})")
    print(f"Producer: {producer_fps}fps, Renderer: {renderer_fps}fps, Duration: {duration}s")
    print(f"Buffer: capacity=12, target=10")
    print("=" * 70)

    results = {}

    for kd in kd_values:
        print(f"\nTesting Kd = {kd} (Kp={kp}, Ki={ki})...")
        sim = FrameDropSimulation(
            producer_fps=producer_fps,
            renderer_fps=renderer_fps,
            buffer_capacity=12,
            target_buffer_level=10,
            use_pid_controller=True,
        )
        # Update PID gains
        sim.frame_dropper.kp = kp
        sim.frame_dropper.ki = ki
        sim.frame_dropper.kd = kd
        sim.run_simulation(duration)
        results[kd] = sim

        # Quick summary with oscillation analysis
        buffer_sizes = [size for _, size in sim.buffer_size_history]
        if buffer_sizes:
            buffer_std = np.std(buffer_sizes)
            buffer_avg = np.mean(buffer_sizes)
            target_error = abs(buffer_avg - 10.0)
            # Calculate steady-state error (last 25% of simulation)
            steady_state_start = int(0.75 * len(buffer_sizes))
            steady_state_sizes = buffer_sizes[steady_state_start:]
            steady_state_avg = np.mean(steady_state_sizes) if steady_state_sizes else buffer_avg
            steady_state_error = abs(steady_state_avg - 10.0)
            steady_state_std = np.std(steady_state_sizes) if steady_state_sizes else buffer_std

            print(
                f"Kd={kd}: Avg buffer={buffer_avg:.2f}, Std={buffer_std:.2f}, "
                f"Target error={target_error:.2f}, SS error={steady_state_error:.2f}, SS std={steady_state_std:.2f}"
            )

    # Generate comparison plots
    plot_kd_comparison(results, kp, ki, producer_fps, renderer_fps)

    return results


def run_comparison(producer_fps=24.0, renderer_fps=15.0, duration=120.0):
    """Run comparison between legacy and PID controllers."""
    print("🎯 Frame Drop Controller Comparison")
    print(f"Producer: {producer_fps}fps, Renderer: {renderer_fps}fps, Duration: {duration}s")
    print(f"Buffer: capacity=12, target=10")
    print("=" * 70)

    # Run legacy simulation
    print("\nRunning Legacy Controller...")
    legacy_sim = FrameDropSimulation(
        producer_fps=producer_fps,
        renderer_fps=renderer_fps,
        buffer_capacity=12,
        target_buffer_level=10,  # Not used by legacy, but for consistency
        use_pid_controller=False,
    )
    legacy_sim.run_simulation(duration)

    print("\n" + "=" * 70)

    # Run PID simulation
    print("\nRunning Optimally Tuned PID Controller (Kp=1.0, Ki=0.3, Kd=2.0)...")
    pid_sim = FrameDropSimulation(
        producer_fps=producer_fps,
        renderer_fps=renderer_fps,
        buffer_capacity=12,
        target_buffer_level=10,
        use_pid_controller=True,
    )
    # Set optimal PID gains
    pid_sim.frame_dropper.kp = 1.0
    pid_sim.frame_dropper.ki = 0.3
    pid_sim.frame_dropper.kd = 2.0
    pid_sim.run_simulation(duration)

    # Generate comparison plots
    plot_comparison(legacy_sim, pid_sim, producer_fps, renderer_fps)

    return legacy_sim, pid_sim


def plot_kp_comparison(results, producer_fps, renderer_fps):
    """Generate comparison plots for different Kp values."""
    kp_values = sorted(results.keys())
    n_plots = len(kp_values)

    fig, axes = plt.subplots(n_plots, 2, figsize=(16, 4 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    colors = ["blue", "green", "orange", "red", "purple", "brown"]

    for i, kp in enumerate(kp_values):
        sim = results[kp]
        color = colors[i % len(colors)]

        # Buffer levels plot
        if sim.buffer_size_history and sim.ewma_buffer_history:
            times_inst, sizes_inst = zip(*sim.buffer_size_history)
            times_ewma, sizes_ewma = zip(*sim.ewma_buffer_history)

            axes[i, 0].plot(times_inst, sizes_inst, color=color, linewidth=1, alpha=0.5, label="Instantaneous")
            axes[i, 0].plot(times_ewma, sizes_ewma, color=color, linewidth=2, label="EWMA")
            axes[i, 0].axhline(y=10, color="red", linestyle="--", alpha=0.7, label="Target (10)")
            axes[i, 0].axhline(y=12, color="gray", linestyle="--", alpha=0.7, label="Capacity (12)")
            axes[i, 0].set_title(f"Kp={kp}: Buffer Levels")
            axes[i, 0].set_ylabel("Buffer Size (frames)")
            axes[i, 0].set_xlabel("Time (s)")
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].legend()
            axes[i, 0].set_ylim(-0.5, 12.5)

        # Drop rate plot
        if sim.drop_rate_history:
            times, rates = zip(*sim.drop_rate_history)
            axes[i, 1].plot(times, [r * 100 for r in rates], color=color, linewidth=1)

            # Add expected drop rate line
            expected_drop = max(0, 1 - renderer_fps / producer_fps) * 100
            axes[i, 1].axhline(
                y=expected_drop, color="green", linestyle="--", alpha=0.7, label=f"Expected ({expected_drop:.1f}%)"
            )

            axes[i, 1].set_title(f"Kp={kp}: Target Drop Rate")
            axes[i, 1].set_ylabel("Drop Rate (%)")
            axes[i, 1].set_xlabel("Time (s)")
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend()
            axes[i, 1].set_ylim(-5, 70)

    plt.suptitle(
        f"Kp Value Comparison\\n"
        f"Producer: {producer_fps}fps, Renderer: {renderer_fps}fps, "
        f"Target: 10 frames, Capacity: 12 frames",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig("kp_tuning_comparison.png", dpi=150, bbox_inches="tight")
    print("\\nKp tuning plots saved to kp_tuning_comparison.png")
    plt.show()


def plot_ki_comparison(results, kp, producer_fps, renderer_fps):
    """Generate comparison plots for different Ki values with fixed Kp."""
    ki_values = sorted(results.keys())
    n_plots = len(ki_values)

    fig, axes = plt.subplots(n_plots, 2, figsize=(16, 4 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    colors = ["blue", "green", "orange", "red", "purple", "brown", "pink", "gray"]

    for i, ki in enumerate(ki_values):
        sim = results[ki]
        color = colors[i % len(colors)]

        # Buffer levels plot
        if sim.buffer_size_history and sim.ewma_buffer_history:
            times_inst, sizes_inst = zip(*sim.buffer_size_history)
            times_ewma, sizes_ewma = zip(*sim.ewma_buffer_history)

            axes[i, 0].plot(times_inst, sizes_inst, color=color, linewidth=1, alpha=0.5, label="Instantaneous")
            axes[i, 0].plot(times_ewma, sizes_ewma, color=color, linewidth=2, label="EWMA")
            axes[i, 0].axhline(y=10, color="red", linestyle="--", alpha=0.7, label="Target (10)")
            axes[i, 0].axhline(y=12, color="gray", linestyle="--", alpha=0.7, label="Capacity (12)")

            # Add steady-state analysis
            buffer_sizes = [size for _, size in sim.buffer_size_history]
            steady_state_start = int(0.75 * len(buffer_sizes))
            steady_state_avg = np.mean(buffer_sizes[steady_state_start:])
            axes[i, 0].axhline(
                y=steady_state_avg, color="orange", linestyle=":", alpha=0.8, label=f"SS Avg ({steady_state_avg:.1f})"
            )

            axes[i, 0].set_title(f"Kp={kp}, Ki={ki}: Buffer Levels")
            axes[i, 0].set_ylabel("Buffer Size (frames)")
            axes[i, 0].set_xlabel("Time (s)")
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].legend()
            axes[i, 0].set_ylim(-0.5, 12.5)

        # Drop rate plot (both target and actual)
        if sim.drop_rate_history:
            times, rates = zip(*sim.drop_rate_history)
            axes[i, 1].plot(times, [r * 100 for r in rates], color=color, linewidth=2, label="Target Drop Rate")

            # Add actual drop rates
            if sim.actual_drop_rates:
                times_actual, adaptive_rates, pre_opt_rates, late_rates, total_rates = zip(*sim.actual_drop_rates)
                axes[i, 1].plot(
                    times_actual,
                    [r * 100 for r in adaptive_rates],
                    color="blue",
                    linewidth=1,
                    alpha=0.7,
                    label="Actual Adaptive",
                )
                axes[i, 1].plot(
                    times_actual,
                    [r * 100 for r in pre_opt_rates],
                    color="orange",
                    linewidth=1,
                    alpha=0.7,
                    label="Pre-opt Late",
                )
                axes[i, 1].plot(
                    times_actual,
                    [r * 100 for r in late_rates],
                    color="red",
                    linewidth=1,
                    alpha=0.7,
                    label="Renderer Late",
                )
                axes[i, 1].plot(
                    times_actual,
                    [r * 100 for r in total_rates],
                    color="purple",
                    linewidth=2,
                    alpha=0.8,
                    label="Total Actual",
                )

            # Add expected drop rate line
            expected_drop = max(0, 1 - renderer_fps / producer_fps) * 100
            axes[i, 1].axhline(
                y=expected_drop, color="green", linestyle="--", alpha=0.7, label=f"Expected ({expected_drop:.1f}%)"
            )

            axes[i, 1].set_title(f"Kp={kp}, Ki={ki}: Drop Rates")
            axes[i, 1].set_ylabel("Drop Rate (%)")
            axes[i, 1].set_xlabel("Time (s)")
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend(fontsize=8)
            axes[i, 1].set_ylim(-5, 70)

    plt.suptitle(
        f"Ki Value Comparison (Kp={kp}, Kd=0)\\n"
        f"Producer: {producer_fps}fps, Renderer: {renderer_fps}fps, "
        f"Target: 10 frames, Capacity: 12 frames",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig("ki_tuning_comparison.png", dpi=150, bbox_inches="tight")
    print("\\nKi tuning plots saved to ki_tuning_comparison.png")
    plt.show()


def plot_kd_comparison(results, kp, ki, producer_fps, renderer_fps):
    """Generate comparison plots for different Kd values with fixed Kp and Ki."""
    kd_values = sorted(results.keys())
    n_plots = len(kd_values)

    fig, axes = plt.subplots(n_plots, 2, figsize=(16, 4 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    colors = ["blue", "green", "orange", "red", "purple", "brown", "pink", "gray"]

    for i, kd in enumerate(kd_values):
        sim = results[kd]
        color = colors[i % len(colors)]

        # Buffer levels plot
        if sim.buffer_size_history and sim.ewma_buffer_history:
            times_inst, sizes_inst = zip(*sim.buffer_size_history)
            times_ewma, sizes_ewma = zip(*sim.ewma_buffer_history)

            axes[i, 0].plot(times_inst, sizes_inst, color=color, linewidth=1, alpha=0.5, label="Instantaneous")
            axes[i, 0].plot(times_ewma, sizes_ewma, color=color, linewidth=2, label="EWMA")
            axes[i, 0].axhline(y=10, color="red", linestyle="--", alpha=0.7, label="Target (10)")
            axes[i, 0].axhline(y=12, color="gray", linestyle="--", alpha=0.7, label="Capacity (12)")

            # Add steady-state analysis (last 25% of simulation)
            buffer_sizes = [size for _, size in sim.buffer_size_history]
            steady_state_start = int(0.75 * len(buffer_sizes))
            steady_state_avg = np.mean(buffer_sizes[steady_state_start:])
            steady_state_std = np.std(buffer_sizes[steady_state_start:])
            axes[i, 0].axhline(
                y=steady_state_avg,
                color="orange",
                linestyle=":",
                alpha=0.8,
                label=f"SS Avg ({steady_state_avg:.1f}±{steady_state_std:.1f})",
            )

            axes[i, 0].set_title(f"Kp={kp}, Ki={ki}, Kd={kd}: Buffer Levels")
            axes[i, 0].set_ylabel("Buffer Size (frames)")
            axes[i, 0].set_xlabel("Time (s)")
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].legend()
            axes[i, 0].set_ylim(-0.5, 12.5)

        # Drop rate plot (both target and actual)
        if sim.drop_rate_history:
            times, rates = zip(*sim.drop_rate_history)
            axes[i, 1].plot(times, [r * 100 for r in rates], color=color, linewidth=2, label="Target Drop Rate")

            # Add actual drop rates
            if sim.actual_drop_rates:
                times_actual, adaptive_rates, pre_opt_rates, late_rates, total_rates = zip(*sim.actual_drop_rates)
                axes[i, 1].plot(
                    times_actual,
                    [r * 100 for r in adaptive_rates],
                    color="blue",
                    linewidth=1,
                    alpha=0.7,
                    label="Actual Adaptive",
                )
                axes[i, 1].plot(
                    times_actual,
                    [r * 100 for r in pre_opt_rates],
                    color="orange",
                    linewidth=1,
                    alpha=0.7,
                    label="Pre-opt Late",
                )
                axes[i, 1].plot(
                    times_actual,
                    [r * 100 for r in late_rates],
                    color="red",
                    linewidth=1,
                    alpha=0.7,
                    label="Renderer Late",
                )
                axes[i, 1].plot(
                    times_actual,
                    [r * 100 for r in total_rates],
                    color="purple",
                    linewidth=2,
                    alpha=0.8,
                    label="Total Actual",
                )

            # Add expected drop rate line
            expected_drop = max(0, 1 - renderer_fps / producer_fps) * 100
            axes[i, 1].axhline(
                y=expected_drop, color="green", linestyle="--", alpha=0.7, label=f"Expected ({expected_drop:.1f}%)"
            )

            axes[i, 1].set_title(f"Kp={kp}, Ki={ki}, Kd={kd}: Drop Rates")
            axes[i, 1].set_ylabel("Drop Rate (%)")
            axes[i, 1].set_xlabel("Time (s)")
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend(fontsize=8)
            axes[i, 1].set_ylim(-5, 70)

    plt.suptitle(
        f"Kd Value Comparison (Kp={kp}, Ki={ki})\\n"
        f"Producer: {producer_fps}fps, Renderer: {renderer_fps}fps, "
        f"Target: 10 frames, Capacity: 12 frames",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig("kd_tuning_comparison.png", dpi=150, bbox_inches="tight")
    print("\\nKd tuning plots saved to kd_tuning_comparison.png")
    plt.show()


def plot_comparison(legacy_sim, pid_sim, producer_fps, renderer_fps):
    """Generate comprehensive comparison plots."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # Plot legacy results (left column)
    plot_simulation_results(legacy_sim, axes[:, 0], "Legacy Controller", "red")

    # Plot PID results (right column)
    plot_simulation_results(pid_sim, axes[:, 1], "PID Controller", "blue")

    # Add overall title
    expected_drop = max(0, 1 - renderer_fps / producer_fps) * 100
    plt.suptitle(
        f"Frame Drop Controller Comparison\n"
        f"Producer: {producer_fps}fps, Renderer: {renderer_fps}fps, "
        f"Expected Drop Rate: {expected_drop:.1f}%",
        fontsize=16,
    )

    plt.tight_layout()
    plt.savefig("frame_drop_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to frame_drop_comparison.png")
    plt.show()


def plot_simulation_results(sim, axes, title, color):
    """Plot results for one simulation."""
    # Buffer levels (instantaneous and EWMA)
    if sim.buffer_size_history and sim.ewma_buffer_history:
        times_inst, sizes_inst = zip(*sim.buffer_size_history)
        times_ewma, sizes_ewma = zip(*sim.ewma_buffer_history)

        axes[0].plot(times_inst, sizes_inst, color=color, linewidth=1, alpha=0.5, label="Instantaneous")
        axes[0].plot(times_ewma, sizes_ewma, color=color, linewidth=2, label="EWMA")
        axes[0].axhline(y=sim.buffer_capacity, color="gray", linestyle="--", alpha=0.7, label="Capacity")
        axes[0].set_title(f"{title}: Buffer Levels")
        axes[0].set_ylabel("Buffer Size (frames)")
        axes[0].set_xlabel("Time (s)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim(-0.5, sim.buffer_capacity + 0.5)

    # Drop rate over time (both target and actual)
    if sim.drop_rate_history:
        times, rates = zip(*sim.drop_rate_history)
        axes[1].plot(times, [r * 100 for r in rates], color=color, linewidth=2, label="Target Drop Rate")

        # Add actual drop rates
        if sim.actual_drop_rates:
            times_actual, adaptive_rates, pre_opt_rates, late_rates, total_rates = zip(*sim.actual_drop_rates)
            axes[1].plot(
                times_actual,
                [r * 100 for r in adaptive_rates],
                color="blue",
                linewidth=1,
                alpha=0.7,
                label="Actual Adaptive",
            )
            axes[1].plot(
                times_actual,
                [r * 100 for r in pre_opt_rates],
                color="orange",
                linewidth=1,
                alpha=0.7,
                label="Pre-opt Late",
            )
            axes[1].plot(
                times_actual, [r * 100 for r in late_rates], color="red", linewidth=1, alpha=0.7, label="Renderer Late"
            )
            axes[1].plot(
                times_actual,
                [r * 100 for r in total_rates],
                color="purple",
                linewidth=2,
                alpha=0.8,
                label="Total Actual",
            )

        # Add expected drop rate line
        expected_drop = max(0, 1 - sim.renderer_fps / sim.producer_fps) * 100
        axes[1].axhline(
            y=expected_drop, color="green", linestyle="--", alpha=0.7, label=f"Expected ({expected_drop:.1f}%)"
        )

        axes[1].set_title(f"{title}: Drop Rates")
        axes[1].set_ylabel("Drop Rate (%)")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        axes[1].set_ylim(-5, 70)

    # Summary statistics
    total_produced = sim.next_frame_id
    adaptive_dropped = len(sim.dropped_frame_ids)
    late_dropped = len(sim.late_dropped_frames)
    rendered = len(sim.rendered_frames)

    expected_drop = max(0, 1 - sim.renderer_fps / sim.producer_fps) * 100
    actual_adaptive_drop = (adaptive_dropped / max(1, total_produced)) * 100
    actual_late_drop = (late_dropped / max(1, total_produced)) * 100
    total_drop = actual_adaptive_drop + actual_late_drop

    # Buffer statistics
    if sim.buffer_size_history:
        buffer_sizes = [size for _, size in sim.buffer_size_history]
        buffer_avg = np.mean(buffer_sizes)
        buffer_std = np.std(buffer_sizes)

        # Check for oscillations (high variance after steady state)
        if len(buffer_sizes) > 1000:  # After initial transient
            steady_state_sizes = buffer_sizes[1000:]
            steady_state_std = np.std(steady_state_sizes)
            oscillating = steady_state_std > 2.0
        else:
            steady_state_std = buffer_std
            oscillating = False
    else:
        buffer_avg = 0
        buffer_std = 0
        steady_state_std = 0
        oscillating = False

    summary = f"""{title}
Duration: {sim.current_time:.1f}s
Renderer: {sim.renderer_state}

Frames Produced: {total_produced}
Adaptive Drops: {adaptive_dropped} ({actual_adaptive_drop:.1f}%)
Late Drops: {late_dropped} ({actual_late_drop:.1f}%)
Total Drops: {adaptive_dropped + late_dropped} ({total_drop:.1f}%)
Rendered: {rendered}

Expected Drop Rate: {expected_drop:.1f}%
Actual Adaptive Drop: {actual_adaptive_drop:.1f}%
Error: {abs(actual_adaptive_drop - expected_drop):.1f}%

Buffer Stats:
- Average: {buffer_avg:.2f} frames
- Std Dev: {buffer_std:.2f}
- Steady State Std: {steady_state_std:.2f}
- Status: {'⚠️ OSCILLATING' if oscillating else '✅ Stable'}

Utilization Stats:
- Wait Events: {len(sim.consumer_wait_times)}
- Total Wait Time: {sim.total_wait_time:.3f}s
- Wait Time EWMA: {sim.wait_time_ewma:.4f}s
- Avg Wait Time: {np.mean(sim.consumer_wait_times) if sim.consumer_wait_times else 0:.4f}s"""

    axes[2].text(
        0.05, 0.95, summary, transform=axes[2].transAxes, verticalalignment="top", fontfamily="monospace", fontsize=9
    )
    axes[2].set_title(f"{title}: Summary")
    axes[2].axis("off")


def main():
    """Run the fixed simulation comparison."""
    # Test with 2 minutes simulation
    legacy_sim, pid_sim = run_comparison(producer_fps=24.0, renderer_fps=15.0, duration=120.0)  # 2 minutes

    return legacy_sim, pid_sim


if __name__ == "__main__":
    main()
