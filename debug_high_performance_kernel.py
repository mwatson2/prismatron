#!/usr/bin/env python3
"""
Debug High-Performance CUDA Kernel.

Debugging the high-performance kernel to identify and fix CUDA_ERROR_INVALID_VALUE.
"""

import logging
import sys
from pathlib import Path

import cupy as cp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_small_test_tensor(led_count: int = 10) -> SingleBlockMixedSparseTensor:
    """Create small test tensor for debugging."""
    logger.info(f"Creating debug tensor: {led_count} LEDs, 96x96 blocks...")

    height, width, channels = 480, 800, 3
    block_size = 96

    tensor = SingleBlockMixedSparseTensor(
        led_count, channels, height, width, block_size
    )

    # Simple test patterns
    for led_id in range(led_count):
        for channel in range(channels):
            # Fixed position to avoid issues - ensure it fits
            top_row = 50 + (led_id * 10) % (height - block_size - 50)
            top_col = 50 + (channel * 20) % (width - block_size - 50)

            # Simple pattern
            pattern = np.ones((block_size, block_size), dtype=np.float32) * 0.5
            pattern[block_size // 2, block_size // 2] = 1.0  # Center spike

            tensor.set_block(led_id, channel, top_row, top_col, cp.asarray(pattern))

    return tensor


def debug_kernel_launch():
    """Debug the high-performance kernel step by step."""
    logger.info("DEBUGGING HIGH-PERFORMANCE CUDA KERNEL")
    logger.info("=" * 50)

    # Start with very small test
    for led_count in [1, 5, 10, 50, 100]:
        logger.info(f"\\nTesting {led_count} LEDs...")

        try:
            # Create tensor
            tensor = create_small_test_tensor(led_count)
            logger.info(
                f"  Tensor created: {tensor.batch_size} LEDs, {tensor.channels} channels"
            )

            # Create test target
            target = cp.ones((tensor.height, tensor.width), dtype=cp.float32)
            logger.info(f"  Target created: {target.shape}")

            # Calculate expected parameters
            total_pairs = tensor.batch_size * tensor.channels
            shared_mem_size = (tensor.block_size * tensor.block_size + 512) * 4
            logger.info(f"  Total (LED, channel) pairs: {total_pairs}")
            logger.info(f"  Expected grid size: ({total_pairs},)")
            logger.info(f"  Expected block size: (512,)")
            logger.info(
                f"  Expected shared memory: {shared_mem_size} bytes ({shared_mem_size/1024:.1f}KB)"
            )

            # Test corrected kernel first (as reference)
            try:
                ref_result = tensor.transpose_dot_product_cuda_corrected(target)
                logger.info(
                    f"  ✅ Corrected kernel works: result shape {ref_result.shape}"
                )
                logger.info(
                    f"    Result range: [{float(cp.min(ref_result)):.3f}, {float(cp.max(ref_result)):.3f}]"
                )
            except Exception as e:
                logger.error(f"  ❌ Corrected kernel failed: {e}")
                continue

            # Test high-performance kernel
            try:
                hp_result = tensor.transpose_dot_product_cuda_high_performance(target)
                logger.info(
                    f"  ✅ High-performance kernel works: result shape {hp_result.shape}"
                )
                logger.info(
                    f"    Result range: [{float(cp.min(hp_result)):.3f}, {float(cp.max(hp_result)):.3f}]"
                )

                # Check correctness
                max_error = float(cp.max(cp.abs(ref_result - hp_result)))
                logger.info(f"    Max error vs corrected: {max_error:.2e}")

                if max_error < 1e-4:
                    logger.info(f"  ✅ Results match within tolerance")
                else:
                    logger.warning(f"  ⚠️  Results differ more than expected")

            except Exception as e:
                logger.error(f"  ❌ High-performance kernel failed: {e}")

                # Check GPU properties for debugging
                try:
                    device = cp.cuda.Device()
                    props = device.attributes
                    max_shared_mem = props["MaxSharedMemoryPerBlock"]
                    max_threads_per_block = props["MaxThreadsPerBlock"]
                    max_blocks_per_mp = props["MaxBlocksPerMultiprocessor"]

                    logger.error(
                        f"    GPU max shared memory per block: {max_shared_mem} bytes ({max_shared_mem/1024:.1f}KB)"
                    )
                    logger.error(
                        f"    GPU max threads per block: {max_threads_per_block}"
                    )
                    logger.error(f"    GPU max blocks per MP: {max_blocks_per_mp}")

                    if shared_mem_size > max_shared_mem:
                        logger.error(
                            f"    ❌ ISSUE: Requesting {shared_mem_size} bytes > max {max_shared_mem} bytes"
                        )

                    if 512 > max_threads_per_block:
                        logger.error(
                            f"    ❌ ISSUE: Requesting 512 threads > max {max_threads_per_block}"
                        )

                except Exception as prop_error:
                    logger.error(f"    Could not get GPU properties: {prop_error}")

                break  # Stop testing larger sizes if this one failed

        except Exception as e:
            logger.error(f"  ❌ Failed at tensor creation: {e}")
            break

    logger.info(f"\\nDebugging complete.")


if __name__ == "__main__":
    debug_kernel_launch()
