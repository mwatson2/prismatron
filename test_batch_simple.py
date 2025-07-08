#!/usr/bin/env python3
"""
Simple test script for batch frame optimization.
Uses the actual pattern file format and logs results.
"""

import time

import cupy as cp
import numpy as np


# Simple logging function
def log_with_timestamp(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def main():
    log_with_timestamp("Starting batch frame optimization test")

    try:
        # Load pattern data from the 64x64 pattern file which has ATA inverse
        pattern_file = "diffusion_patterns/synthetic_2624_fp16_64x64.npz"
        log_with_timestamp(f"Loading pattern data from {pattern_file}...")
        data = np.load(pattern_file, allow_pickle=True)

        metadata = data["metadata"].item()
        led_count = metadata["led_count"]
        frame_width = metadata["frame_width"]
        frame_height = metadata["frame_height"]

        log_with_timestamp(f"Pattern info: {led_count} LEDs, {frame_height}x{frame_width}")

        # Test both batch sizes
        for batch_size in [8, 16]:
            log_with_timestamp(f"\nTesting batch size: {batch_size}")

            # Create test frames
            log_with_timestamp("Creating test frames...")
            test_frames = np.random.randint(0, 255, (batch_size, 3, frame_height, frame_width), dtype=np.uint8)
            log_with_timestamp(f"Test frames shape: {test_frames.shape}")
            log_with_timestamp(f"Test frames memory: {test_frames.nbytes / 1024 / 1024:.1f} MB")

            # Import and test batch optimizer
            try:
                import sys

                sys.path.insert(0, "src")
                from utils.batch_frame_optimizer import optimize_batch_frames_led_values
                from utils.diagonal_ata_matrix import DiagonalATAMatrix
                from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

                log_with_timestamp("Successfully imported batch optimizer modules")

                # Create mock matrices for testing
                log_with_timestamp("Creating test matrices...")

                # Actually, we don't need to create the matrix - we can load it directly
                # Just load from the saved data

                # Load mixed tensor using from_dict()
                mixed_tensor_dict = data["mixed_tensor"].item()
                at_matrix = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)
                log_with_timestamp("Built AT matrix from saved tensor")

                # Load DIA matrix using from_dict()
                dia_dict = data["dia_matrix"].item()
                ata_matrix = DiagonalATAMatrix.from_dict(dia_dict)
                log_with_timestamp("Built ATA matrix from saved DIA data")

                # Try to load real ATA inverse if available, otherwise skip
                if "ata_inverse" in data:
                    ata_inverse = data["ata_inverse"]
                    log_with_timestamp(f"Loaded real ATA inverse: shape={ata_inverse.shape}")
                else:
                    log_with_timestamp("No real ATA inverse found, skipping optimization test")
                    continue

                log_with_timestamp("Starting batch optimization test...")
                start_time = time.time()

                result = optimize_batch_frames_led_values(
                    test_frames, at_matrix, ata_matrix, ata_inverse, max_iterations=3, debug=True, enable_timing=True
                )

                elapsed = time.time() - start_time
                log_with_timestamp(f"Batch optimization completed in {elapsed:.3f}s")

                # Log results
                log_with_timestamp(f"Result shape: {result.led_values.shape}")
                log_with_timestamp(f"LED values range: [{result.led_values.min()}, {result.led_values.max()}]")
                log_with_timestamp(f"Iterations: {result.iterations}")

                # Performance metrics
                per_frame_time = elapsed / batch_size
                fps = batch_size / elapsed
                log_with_timestamp(f"Per-frame time: {per_frame_time:.3f}s")
                log_with_timestamp(f"Effective FPS: {fps:.1f}")

                # Timing breakdown
                if result.timing_data:
                    log_with_timestamp("Timing breakdown:")
                    for section, time_val in result.timing_data.items():
                        log_with_timestamp(f"  {section}: {time_val:.3f}s")

                log_with_timestamp(f"Batch size {batch_size} test completed successfully!")

            except ImportError as e:
                log_with_timestamp(f"Import error: {e}")
                continue
            except Exception as e:
                log_with_timestamp(f"Test error: {e}")
                import traceback

                log_with_timestamp(f"Traceback: {traceback.format_exc()}")
                continue

            # Cleanup
            del test_frames
            import gc

            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

        log_with_timestamp("All tests completed!")

    except Exception as e:
        log_with_timestamp(f"Fatal error: {e}")
        import traceback

        log_with_timestamp(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
