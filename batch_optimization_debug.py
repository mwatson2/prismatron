#!/usr/bin/env python3
"""
Debug script for batch frame optimization testing with comprehensive logging.
This script logs all operations to prevent data loss during OOM crashes.
"""

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cupy as cp
import numpy as np

# Add src to path
sys.path.insert(0, "src")

# Import the batch optimizer
from utils.batch_frame_optimizer import convert_ata_dia_to_dense, optimize_batch_frames_led_values
from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.frame_optimizer import load_ata_inverse_from_pattern, optimize_frame_led_values
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


# Set up comprehensive logging
class DebugLogger:
    def __init__(self, log_file="batch_optimization_debug.log"):
        self.log_file = log_file
        self.start_time = time.time()
        self.log(f"=== Debug Session Started at {datetime.now()} ===")

    def log(self, message):
        """Log message with timestamp and memory info."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        elapsed = time.time() - self.start_time

        # Memory info (simplified)
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            memory_mb = 0

        # GPU memory info
        try:
            gpu_memory = cp.get_default_memory_pool().used_bytes() / 1024 / 1024
            gpu_total = cp.get_default_memory_pool().total_bytes() / 1024 / 1024
        except:
            gpu_memory = 0
            gpu_total = 0

        log_message = f"[{timestamp}] +{elapsed:.3f}s | RAM: {memory_mb:.1f}MB | GPU: {gpu_memory:.1f}/{gpu_total:.1f}MB | {message}"

        print(log_message)
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
            f.flush()

    def log_error(self, error, context=""):
        """Log error with full traceback."""
        self.log(f"ERROR in {context}: {error}")
        self.log(f"Traceback: {traceback.format_exc()}")


# Global logger
logger = DebugLogger()


def load_test_matrices(pattern_path: str):
    """Load test matrices with debug logging."""
    logger.log(f"Loading matrices from {pattern_path}")

    try:
        # Load the pattern data
        data = np.load(pattern_path, allow_pickle=True)
        logger.log(f"Pattern data loaded successfully")

        led_count = int(data["led_count"])
        frame_height = int(data["frame_height"])
        frame_width = int(data["frame_width"])

        logger.log(f"Pattern specs: {led_count} LEDs, {frame_height}x{frame_width}")

        # Create SingleBlockMixedSparseTensor for AT matrix
        logger.log("Creating SingleBlockMixedSparseTensor...")
        at_matrix = SingleBlockMixedSparseTensor(
            led_count=led_count, frame_height=frame_height, frame_width=frame_width
        )

        # Load diffusion matrix
        if "diffusion_matrix" in data:
            diffusion_matrix = data["diffusion_matrix"]
            logger.log(f"Diffusion matrix shape: {diffusion_matrix.shape}")
            at_matrix.build_from_csc_matrix(diffusion_matrix)
            logger.log("AT matrix built from diffusion matrix")
        else:
            logger.log("ERROR: No diffusion matrix found in pattern file")
            return None, None, None

        # Create DiagonalATAMatrix
        logger.log("Creating DiagonalATAMatrix...")
        ata_matrix = DiagonalATAMatrix(led_count=led_count, frame_height=frame_height, frame_width=frame_width)

        # Build ATA matrix from the same diffusion matrix
        logger.log("Building ATA matrix from diffusion matrix...")
        ata_matrix.build_from_diffusion_matrix(diffusion_matrix)
        logger.log("ATA matrix built successfully")

        # Load ATA inverse
        logger.log("Loading ATA inverse...")
        ata_inverse = load_ata_inverse_from_pattern(pattern_path)
        if ata_inverse is None:
            logger.log("ERROR: No ATA inverse found in pattern file")
            return None, None, None

        logger.log(f"ATA inverse shape: {ata_inverse.shape}")
        logger.log("All matrices loaded successfully")

        return at_matrix, ata_matrix, ata_inverse, led_count, frame_height, frame_width

    except Exception as e:
        logger.log_error(e, "load_test_matrices")
        return None, None, None, None, None, None


def create_test_frames(batch_size: int, frame_height: int, frame_width: int) -> np.ndarray:
    """Create test frames with debug logging."""
    logger.log(f"Creating {batch_size} test frames ({frame_height}x{frame_width})")

    try:
        frames = []
        for i in range(batch_size):
            logger.log(f"Creating frame {i+1}/{batch_size}")

            # Create realistic frame patterns
            frame = np.zeros((3, frame_height, frame_width), dtype=np.uint8)

            # Add different patterns
            if i % 4 == 0:
                # Gradient
                x = np.linspace(0, 1, frame_width)
                y = np.linspace(0, 1, frame_height)
                X, Y = np.meshgrid(x, y)
                frame[0] = (X * 255).astype(np.uint8)
                frame[1] = (Y * 255).astype(np.uint8)
                frame[2] = ((X + Y) * 127).astype(np.uint8)
            elif i % 4 == 1:
                # Sinusoidal
                x = np.linspace(0, 4 * np.pi, frame_width)
                y = np.linspace(0, 4 * np.pi, frame_height)
                X, Y = np.meshgrid(x, y)
                frame[0] = ((np.sin(X) * np.cos(Y) + 1) * 127).astype(np.uint8)
                frame[1] = ((np.cos(X) * np.sin(Y) + 1) * 127).astype(np.uint8)
                frame[2] = ((np.sin(X + Y) + 1) * 127).astype(np.uint8)
            elif i % 4 == 2:
                # Circular
                center_y, center_x = frame_height // 2, frame_width // 2
                y, x = np.ogrid[:frame_height, :frame_width]
                for c in range(3):
                    radius = (c + 1) * min(frame_height, frame_width) // 6
                    mask = ((y - center_y) ** 2 + (x - center_x) ** 2) < radius**2
                    frame[c, mask] = 255
            else:
                # Random with structure
                frame = np.random.randint(0, 255, (3, frame_height, frame_width), dtype=np.uint8)

            frames.append(frame)

        batch_frames = np.stack(frames, axis=0)
        logger.log(f"Test frames created successfully: {batch_frames.shape}")
        logger.log(f"Test frames memory: {batch_frames.nbytes / 1024 / 1024:.1f} MB")

        return batch_frames

    except Exception as e:
        logger.log_error(e, "create_test_frames")
        return None


def test_batch_optimization(at_matrix, ata_matrix, ata_inverse, batch_frames, max_iterations=5):
    """Test batch optimization with comprehensive logging."""
    batch_size = batch_frames.shape[0]
    logger.log(f"Starting batch optimization test: {batch_size} frames, {max_iterations} iterations")

    try:
        # First, test the DIA to dense conversion
        logger.log("Testing ATA DIA to dense conversion...")
        conversion_start = time.time()
        ata_dense = convert_ata_dia_to_dense(ata_matrix)
        conversion_time = time.time() - conversion_start
        logger.log(f"ATA conversion completed in {conversion_time:.3f}s")
        logger.log(f"Dense ATA shape: {ata_dense.shape}")
        logger.log(f"Dense ATA memory: {ata_dense.nbytes / 1024 / 1024:.1f} MB")

        # Test batch optimization
        logger.log("Starting batch optimization...")
        batch_start = time.time()

        result = optimize_batch_frames_led_values(
            batch_frames,
            at_matrix,
            ata_matrix,
            ata_inverse,
            max_iterations=max_iterations,
            debug=True,
            enable_timing=True,
        )

        batch_time = time.time() - batch_start
        logger.log(f"Batch optimization completed in {batch_time:.3f}s")

        # Log results
        logger.log(f"Result LED values shape: {result.led_values.shape}")
        logger.log(f"Result LED values range: [{result.led_values.min()}, {result.led_values.max()}]")
        logger.log(f"Iterations: {result.iterations}")
        logger.log(f"Converged: {result.converged}")

        # Log timing breakdown
        if result.timing_data:
            logger.log("Timing breakdown:")
            total_time = sum(result.timing_data.values())
            for section, time_val in result.timing_data.items():
                percentage = (time_val / total_time) * 100
                logger.log(f"  {section}: {time_val:.3f}s ({percentage:.1f}%)")

        # Performance metrics
        per_frame_time = batch_time / batch_size
        fps = batch_size / batch_time
        logger.log(f"Per-frame time: {per_frame_time:.3f}s")
        logger.log(f"Effective FPS: {fps:.1f}")

        return result

    except Exception as e:
        logger.log_error(e, "test_batch_optimization")
        return None


def test_single_frame_comparison(at_matrix, ata_matrix, ata_inverse, batch_frames, max_iterations=5):
    """Test single frame processing for comparison."""
    batch_size = batch_frames.shape[0]
    logger.log(f"Starting single frame comparison test: {batch_size} frames")

    try:
        single_start = time.time()
        single_results = []

        for i in range(batch_size):
            logger.log(f"Processing single frame {i+1}/{batch_size}")
            frame = batch_frames[i]

            result = optimize_frame_led_values(
                frame, at_matrix, ata_matrix, ata_inverse, max_iterations=max_iterations, debug=False
            )
            single_results.append(result)

        single_time = time.time() - single_start
        logger.log(f"Single frame processing completed in {single_time:.3f}s")

        # Performance metrics
        per_frame_time = single_time / batch_size
        fps = batch_size / single_time
        logger.log(f"Single frame per-frame time: {per_frame_time:.3f}s")
        logger.log(f"Single frame effective FPS: {fps:.1f}")

        return single_results, single_time

    except Exception as e:
        logger.log_error(e, "test_single_frame_comparison")
        return None, None


def run_comprehensive_test():
    """Run comprehensive batch optimization test."""
    logger.log("Starting comprehensive batch optimization test")

    try:
        # Find pattern file
        pattern_candidates = [
            "patterns_2624_fp16.npz",
            "diffusion_patterns/synthetic_2624_fp16_64x64.npz",
            "diffusion_patterns/synthetic_2624_fp16.npz",
        ]

        pattern_path = None
        for candidate in pattern_candidates:
            if Path(candidate).exists():
                pattern_path = candidate
                break

        if not pattern_path:
            logger.log("No 2624 LED pattern file found. Looking for any pattern file...")
            pattern_files = list(Path(".").glob("*patterns*.npz"))
            if pattern_files:
                pattern_path = str(pattern_files[0])
            else:
                logger.log("ERROR: No pattern files found")
                return

        logger.log(f"Using pattern file: {pattern_path}")

        # Load matrices
        matrices = load_test_matrices(pattern_path)
        if matrices[0] is None:
            logger.log("ERROR: Failed to load matrices")
            return

        at_matrix, ata_matrix, ata_inverse, led_count, frame_height, frame_width = matrices

        # Test both batch sizes
        for batch_size in [8, 16]:
            logger.log(f"\n{'='*50}")
            logger.log(f"TESTING BATCH SIZE: {batch_size}")
            logger.log(f"{'='*50}")

            # Create test frames
            test_frames = create_test_frames(batch_size, frame_height, frame_width)
            if test_frames is None:
                logger.log("ERROR: Failed to create test frames")
                continue

            # Test batch optimization
            batch_result = test_batch_optimization(at_matrix, ata_matrix, ata_inverse, test_frames, max_iterations=5)

            if batch_result is None:
                logger.log("ERROR: Batch optimization failed")
                continue

            # Test single frame comparison
            single_results, single_time = test_single_frame_comparison(
                at_matrix, ata_matrix, ata_inverse, test_frames, max_iterations=5
            )

            if single_results is None:
                logger.log("ERROR: Single frame comparison failed")
                continue

            # Compare results
            logger.log("Comparing batch vs single frame results...")
            max_diff = 0
            for i in range(batch_size):
                single_led = single_results[i].led_values
                batch_led = batch_result.led_values[i]
                diff = np.abs(single_led.astype(np.float32) - batch_led.astype(np.float32))
                frame_max_diff = np.max(diff)
                max_diff = max(max_diff, frame_max_diff)

            logger.log(f"Maximum LED value difference: {max_diff:.2f} (out of 255)")

            if max_diff < 5:
                logger.log("✓ Results are consistent between single and batch processing")
            else:
                logger.log("⚠ Results have significant differences")

            # Calculate speedup
            if single_time and batch_result.timing_data:
                batch_time = sum(batch_result.timing_data.values())
                speedup = single_time / batch_time
                logger.log(f"Speedup: {speedup:.2f}x")

            # Cleanup
            del test_frames
            if batch_result:
                del batch_result
            if single_results:
                del single_results

            # Force garbage collection
            import gc

            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

            logger.log(f"Batch size {batch_size} test completed successfully")

        logger.log("All tests completed successfully!")

    except Exception as e:
        logger.log_error(e, "run_comprehensive_test")

    finally:
        logger.log("=== Test session completed ===")


if __name__ == "__main__":
    run_comprehensive_test()
