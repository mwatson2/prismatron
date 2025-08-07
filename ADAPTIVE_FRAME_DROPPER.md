Adaptive Frame Dropper Development Status

  Problem Statement

  The Prismatron LED Display System needs adaptive frame dropping to maintain healthy LED buffer levels while optimizing throughput. The system processes video
  frames through a pipeline: Producer → Shared Buffer → Consumer (Optimization) → LED Buffer → Renderer. When the LED buffer fills up, it creates backpressure that
  can destabilize the entire pipeline.

  The adaptive frame dropper uses a PID controller to maintain the LED buffer at a target level by selectively dropping frames before optimization. This reduces
  optimization workload and allows the buffer to maintain optimal levels.

  How the Simulation Works

  The frame_drop_simulation.py models the complete dual-loop system:

  1. Producer Loop: Generates frames at 24fps and adds them to the shared buffer
  2. Consumer Loop: Takes frames from shared buffer, runs optimization (1/15s duration), and adds results to LED buffer
  3. Renderer Loop: Consumes frames from LED buffer at 15fps based on frame timestamps
  4. Adaptive Dropper: Called before optimization, decides whether to drop frames based on LED buffer level

  Key behaviors:
  - Frame dropping only activates when renderer state is "PLAYING"
  - Renderer starts when LED buffer first reaches the target level (10 frames)
  - PID controller uses EWMA tracking of buffer levels with configurable gains
  - System models realistic optimization timing and buffer dynamics

  Current State

  PID Controller Implementation: ✅ Complete and working
  - Target buffer level parameter correctly implemented
  - PID control law with proper sign behavior
  - Integral term resets correctly when Ki=0
  - Utilization penalty integration for LED buffer wait times

  Latest Test Results (Kp=1.0, Ki=0, Kd=0, no utilization penalty):
  - Theoretical optimal drop rate: 37.5%
  - Actual drop rate achieved: 49.8%
  - Buffer EWMA settled at: 4.0 frames (target: 10.0)
  - System behavior: Stable, no oscillations
  - Startup behavior: Correct (no drops until renderer starts)

  Current Challenge

  The P-only controller (Kp=1.0) achieves stable operation but settles at a suboptimal equilibrium:
  - Buffer maintains level below target (4.0 vs 10.0)
  - Drop rate higher than theoretical optimum (49.8% vs 37.5%)
  - System is stable but not achieving desired buffer management goals

  Next Steps

  1. Investigate equilibrium behavior: Analyze why P-only control settles below target buffer level
  2. Test PI control: Add small integral gain (Ki=0.05-0.2) to eliminate steady-state error
  3. Parameter tuning: Systematically test different Kp values (0.1-2.0 range)
  4. Utilization penalty: Test impact of buffer wait time penalty on performance
  5. Production integration: Deploy tuned controller in actual system for validation

  Files Modified

  - src/consumer/adaptive_frame_dropper.py: Core PID controller implementation
  - frame_drop_simulation.py: Complete system simulation with dual-loop architecture
  - pid_controller_analysis.png: Latest test results visualization

  Key Parameters

  - Target buffer level: 10 frames
  - Buffer capacity: 12 frames
  - EWMA alpha: 0.03 (buffer level tracking)
  - Max drop rate: 66% (supports up to 2x input rate)

  The system is ready for parameter optimization and production testing.
