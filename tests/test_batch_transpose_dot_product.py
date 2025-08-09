"""
Tests for batch transpose dot product operation in SingleBlockMixedSparseTensor.

This test suite verifies that the batched A^T @ B operation produces identical results
to repeated single-frame calls, tests both FP32 and uint8 variants, and validates
different output layout formats.
"""

import logging
import unittest

import cupy as cp
import numpy as np

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logger = logging.getLogger(__name__)


class TestBatchTransposeDotProduct(unittest.TestCase):
    """Test cases for batched transpose dot product operations."""

    def setUp(self):
        """Set up test fixtures with various tensor configurations."""
        # Test configurations for different sizes
        self.test_configs = [
            {
                "batch_size": 8,  # Small test - 8 LEDs
                "channels": 3,  # RGB
                "height": 128,
                "width": 128,
                "block_size": 32,  # 32x32 blocks
                "batch_frames": 4,  # Process 4 frames at once
                "name": "small_32x32",
            },
            {
                "batch_size": 16,  # Medium test - 16 LEDs
                "channels": 3,  # RGB
                "height": 256,
                "width": 256,
                "block_size": 64,  # 64x64 blocks
                "batch_frames": 8,  # Process 8 frames at once
                "name": "medium_64x64",
            },
        ]

        # Enable CUDA error checking for debugging
        cp.cuda.runtime.setDevice(0)

    def _create_test_tensor(self, config, dtype=cp.float32):
        """Create a test tensor with random LED patterns."""
        tensor = SingleBlockMixedSparseTensor(
            batch_size=config["batch_size"],
            channels=config["channels"],
            height=config["height"],
            width=config["width"],
            block_size=config["block_size"],
            dtype=dtype,
            output_dtype=cp.float32,
        )

        # Set random blocks at random positions
        np.random.seed(42)  # Reproducible tests
        for led_idx in range(config["batch_size"]):
            for channel_idx in range(config["channels"]):
                # Generate random position ensuring block fits in frame
                max_row = config["height"] - config["block_size"]
                max_col = config["width"] - config["block_size"]
                row = np.random.randint(0, max_row)
                col = np.random.randint(0, max_col)

                # Align positions for vectorization (multiple of 4)
                col = (col // 4) * 4

                # Generate random block data
                if dtype == cp.float32:
                    block_data = cp.random.rand(config["block_size"], config["block_size"], dtype=cp.float32)
                    # Scale to reasonable range [0, 1]
                    block_data = block_data * 0.5 + 0.1
                else:  # uint8
                    block_data = cp.random.randint(0, 256, (config["block_size"], config["block_size"]), dtype=cp.uint8)

                tensor.set_block(led_idx, channel_idx, row, col, block_data)

        return tensor

    def _create_test_target_batch(self, config, batch_frames, dtype=cp.float32):
        """Create a batch of target frames for testing."""
        np.random.seed(123)  # Different seed for target data

        if dtype == cp.float32:
            target_batch = cp.random.rand(
                batch_frames, config["channels"], config["height"], config["width"], dtype=cp.float32
            )
            # Scale to reasonable range [0, 1]
            target_batch = target_batch * 0.8 + 0.1
        else:  # uint8
            target_batch = cp.random.randint(
                0, 256, (batch_frames, config["channels"], config["height"], config["width"]), dtype=cp.uint8
            )

        return target_batch

    def test_fp32_batch_correctness_interleaved(self):
        """Test FP32 batch operation correctness vs repeated single calls (interleaved output)."""
        for config in self.test_configs:
            with self.subTest(config=config["name"]):
                logger.info(f"Testing FP32 batch correctness (interleaved) - {config['name']}")

                # Create test tensor and target batch
                tensor = self._create_test_tensor(config, dtype=cp.float32)
                target_batch = self._create_test_target_batch(config, config["batch_frames"], dtype=cp.float32)

                # Batch operation (interleaved output)
                batch_result = tensor.transpose_dot_product_3d_batch(
                    target_batch, planar_output=False  # (batch_frames, batch_size, channels)
                )

                # Individual frame operations for comparison
                individual_results = []
                for frame_idx in range(config["batch_frames"]):
                    frame_result = tensor.transpose_dot_product_3d(
                        target_batch[frame_idx], planar_output=False  # (batch_size, channels)
                    )
                    individual_results.append(frame_result)

                # Stack individual results to match batch output shape
                expected_result = cp.stack(individual_results, axis=0)  # (batch_frames, batch_size, channels)

                # Verify shapes match
                self.assertEqual(
                    batch_result.shape,
                    expected_result.shape,
                    f"Shape mismatch: batch {batch_result.shape} vs expected {expected_result.shape}",
                )

                # Verify results are close (account for floating point precision)
                max_diff = cp.max(cp.abs(batch_result - expected_result)).get()
                mean_diff = cp.mean(cp.abs(batch_result - expected_result)).get()
                relative_error = mean_diff / (cp.mean(cp.abs(expected_result)).get() + 1e-7)

                logger.info(
                    f"  Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}, Relative error: {relative_error:.2e}"
                )

                # Assert very close results (should be nearly identical for FP32)
                self.assertLess(max_diff, 1e-4, f"Max difference {max_diff} exceeds threshold for {config['name']}")
                self.assertLess(
                    relative_error, 1e-5, f"Relative error {relative_error} exceeds threshold for {config['name']}"
                )

    def test_fp32_batch_correctness_planar(self):
        """Test FP32 batch operation correctness vs repeated single calls (planar output)."""
        for config in self.test_configs:
            with self.subTest(config=config["name"]):
                logger.info(f"Testing FP32 batch correctness (planar) - {config['name']}")

                # Create test tensor and target batch
                tensor = self._create_test_tensor(config, dtype=cp.float32)
                target_batch = self._create_test_target_batch(config, config["batch_frames"], dtype=cp.float32)

                # Batch operation (planar output)
                batch_result = tensor.transpose_dot_product_3d_batch(
                    target_batch, planar_output=True  # (batch_frames, channels, batch_size)
                )

                # Individual frame operations for comparison
                individual_results = []
                for frame_idx in range(config["batch_frames"]):
                    frame_result = tensor.transpose_dot_product_3d(
                        target_batch[frame_idx], planar_output=True  # (channels, batch_size)
                    )
                    individual_results.append(frame_result)

                # Stack individual results to match batch output shape
                expected_result = cp.stack(individual_results, axis=0)  # (batch_frames, channels, batch_size)

                # Verify shapes match
                self.assertEqual(
                    batch_result.shape,
                    expected_result.shape,
                    f"Shape mismatch: batch {batch_result.shape} vs expected {expected_result.shape}",
                )

                # Verify results are close
                max_diff = cp.max(cp.abs(batch_result - expected_result)).get()
                mean_diff = cp.mean(cp.abs(batch_result - expected_result)).get()
                relative_error = mean_diff / (cp.mean(cp.abs(expected_result)).get() + 1e-7)

                logger.info(
                    f"  Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}, Relative error: {relative_error:.2e}"
                )

                # Assert very close results
                self.assertLess(max_diff, 1e-4, f"Max difference {max_diff} exceeds threshold for {config['name']}")
                self.assertLess(
                    relative_error, 1e-5, f"Relative error {relative_error} exceeds threshold for {config['name']}"
                )

    def test_uint8_batch_correctness_interleaved(self):
        """Test uint8 batch operation correctness vs repeated single calls (interleaved output)."""
        for config in self.test_configs:
            with self.subTest(config=config["name"]):
                logger.info(f"Testing uint8 batch correctness (interleaved) - {config['name']}")

                # Create test tensor and target batch
                tensor = self._create_test_tensor(config, dtype=cp.uint8)
                target_batch = self._create_test_target_batch(config, config["batch_frames"], dtype=cp.uint8)

                # Batch operation (interleaved output)
                batch_result = tensor.transpose_dot_product_3d_batch(
                    target_batch, planar_output=False  # (batch_frames, batch_size, channels)
                )

                # Individual frame operations for comparison
                individual_results = []
                for frame_idx in range(config["batch_frames"]):
                    frame_result = tensor.transpose_dot_product_3d(
                        target_batch[frame_idx], planar_output=False  # (batch_size, channels)
                    )
                    individual_results.append(frame_result)

                # Stack individual results to match batch output shape
                expected_result = cp.stack(individual_results, axis=0)  # (batch_frames, batch_size, channels)

                # Verify shapes match
                self.assertEqual(
                    batch_result.shape,
                    expected_result.shape,
                    f"Shape mismatch: batch {batch_result.shape} vs expected {expected_result.shape}",
                )

                # Verify results are close (uint8 arithmetic has different precision characteristics)
                max_diff = cp.max(cp.abs(batch_result - expected_result)).get()
                mean_diff = cp.mean(cp.abs(batch_result - expected_result)).get()
                relative_error = mean_diff / (cp.mean(cp.abs(expected_result)).get() + 1e-7)

                logger.info(
                    f"  Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}, Relative error: {relative_error:.2e}"
                )

                # uint8 operations should still be very close (using integer arithmetic)
                self.assertLess(max_diff, 1e-6, f"Max difference {max_diff} exceeds threshold for {config['name']}")
                self.assertLess(
                    relative_error, 1e-6, f"Relative error {relative_error} exceeds threshold for {config['name']}"
                )

    def test_uint8_batch_correctness_planar(self):
        """Test uint8 batch operation correctness vs repeated single calls (planar output)."""
        for config in self.test_configs:
            with self.subTest(config=config["name"]):
                logger.info(f"Testing uint8 batch correctness (planar) - {config['name']}")

                # Create test tensor and target batch
                tensor = self._create_test_tensor(config, dtype=cp.uint8)
                target_batch = self._create_test_target_batch(config, config["batch_frames"], dtype=cp.uint8)

                # Batch operation (planar output)
                batch_result = tensor.transpose_dot_product_3d_batch(
                    target_batch, planar_output=True  # (batch_frames, channels, batch_size)
                )

                # Individual frame operations for comparison
                individual_results = []
                for frame_idx in range(config["batch_frames"]):
                    frame_result = tensor.transpose_dot_product_3d(
                        target_batch[frame_idx], planar_output=True  # (channels, batch_size)
                    )
                    individual_results.append(frame_result)

                # Stack individual results to match batch output shape
                expected_result = cp.stack(individual_results, axis=0)  # (batch_frames, channels, batch_size)

                # Verify shapes match
                self.assertEqual(
                    batch_result.shape,
                    expected_result.shape,
                    f"Shape mismatch: batch {batch_result.shape} vs expected {expected_result.shape}",
                )

                # Verify results are close
                max_diff = cp.max(cp.abs(batch_result - expected_result)).get()
                mean_diff = cp.mean(cp.abs(batch_result - expected_result)).get()
                relative_error = mean_diff / (cp.mean(cp.abs(expected_result)).get() + 1e-7)

                logger.info(
                    f"  Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}, Relative error: {relative_error:.2e}"
                )

                # Assert very close results
                self.assertLess(max_diff, 1e-6, f"Max difference {max_diff} exceeds threshold for {config['name']}")
                self.assertLess(
                    relative_error, 1e-6, f"Relative error {relative_error} exceeds threshold for {config['name']}"
                )

    def test_single_frame_batch(self):
        """Test batch operation with single frame (edge case)."""
        config = self.test_configs[0]  # Use small config

        logger.info("Testing single frame batch operation")

        # Create test tensor and single-frame batch
        tensor = self._create_test_tensor(config, dtype=cp.float32)
        single_frame = self._create_test_target_batch(config, 1, dtype=cp.float32)  # batch_frames=1

        # Batch operation with single frame
        batch_result = tensor.transpose_dot_product_3d_batch(
            single_frame, planar_output=False  # (1, batch_size, channels)
        )

        # Single frame operation for comparison
        single_result = tensor.transpose_dot_product_3d(single_frame[0], planar_output=False)  # (batch_size, channels)

        # Expand single result to match batch shape
        expected_result = single_result[cp.newaxis, :, :]  # (1, batch_size, channels)

        # Verify shapes match
        self.assertEqual(batch_result.shape, expected_result.shape)
        self.assertEqual(batch_result.shape[0], 1, "Batch dimension should be 1")

        # Verify results are identical
        max_diff = cp.max(cp.abs(batch_result - expected_result)).get()
        self.assertLess(max_diff, 1e-6, "Single frame batch should be identical to single operation")

    def test_large_batch_frames(self):
        """Test batch operation with larger number of frames."""
        config = self.test_configs[0]  # Use small tensor config for speed
        large_batch_frames = 16  # Process 16 frames at once

        logger.info(f"Testing large batch operation ({large_batch_frames} frames)")

        # Create test tensor and large target batch
        tensor = self._create_test_tensor(config, dtype=cp.float32)
        target_batch = self._create_test_target_batch(config, large_batch_frames, dtype=cp.float32)

        # Batch operation
        batch_result = tensor.transpose_dot_product_3d_batch(
            target_batch, planar_output=False  # (16, batch_size, channels)
        )

        # Verify shape is correct
        expected_shape = (large_batch_frames, config["batch_size"], config["channels"])
        self.assertEqual(
            batch_result.shape,
            expected_shape,
            f"Large batch shape mismatch: got {batch_result.shape}, expected {expected_shape}",
        )

        # Sample a few frames for correctness check (too expensive to check all)
        test_frames = [0, large_batch_frames // 2, large_batch_frames - 1]
        for frame_idx in test_frames:
            single_result = tensor.transpose_dot_product_3d(target_batch[frame_idx], planar_output=False)

            frame_diff = cp.max(cp.abs(batch_result[frame_idx] - single_result)).get()
            self.assertLess(frame_diff, 1e-4, f"Frame {frame_idx} differs from single operation")

    def test_input_validation(self):
        """Test input validation for batch operations."""
        config = self.test_configs[0]
        tensor = self._create_test_tensor(config, dtype=cp.float32)

        # Test wrong height dimension
        wrong_height = cp.random.rand(
            config["batch_frames"], config["channels"], config["height"] + 1, config["width"], dtype=cp.float32
        )

        with self.assertRaises(ValueError) as cm:
            tensor.transpose_dot_product_3d_batch(wrong_height)
        self.assertIn("height", str(cm.exception))

        # Test wrong channels dimension
        wrong_channels = cp.random.rand(
            config["batch_frames"], config["channels"] + 1, config["height"], config["width"], dtype=cp.float32
        )

        with self.assertRaises(ValueError) as cm:
            tensor.transpose_dot_product_3d_batch(wrong_channels)
        self.assertIn("channels", str(cm.exception))

        # Test wrong dtype
        uint8_tensor = self._create_test_tensor(config, dtype=cp.uint8)
        fp32_batch = cp.random.rand(
            config["batch_frames"], config["channels"], config["height"], config["width"], dtype=cp.float32
        )

        with self.assertRaises(ValueError) as cm:
            uint8_tensor.transpose_dot_product_3d_batch(fp32_batch)
        self.assertIn("dtype", str(cm.exception))


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Run tests
    unittest.main()
