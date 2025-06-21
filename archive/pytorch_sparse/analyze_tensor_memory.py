#!/usr/bin/env python3
"""
Analyze memory usage of PyTorch 4D COO tensor for LED optimization.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_coo_tensor_memory():
    """Analyze memory usage of 4D COO tensor."""

    # Tensor dimensions from the logs
    height, width, led_count, channels = 480, 800, 1000, 3
    nnz = 41_945_614

    logger.info("=== PyTorch 4D COO Tensor Memory Analysis ===")
    logger.info(f"Tensor shape: ({height}, {width}, {led_count}, {channels})")
    logger.info(f"Non-zero values: {nnz:,}")

    # Memory for indices tensor (4 dimensions × nnz × 8 bytes for long)
    indices_bytes = 4 * nnz * 8
    indices_mb = indices_bytes / (1024 * 1024)
    indices_gb = indices_mb / 1024

    # Memory for values tensor (nnz × 4 bytes for float32)
    values_bytes = nnz * 4
    values_mb = values_bytes / (1024 * 1024)
    values_gb = values_mb / 1024

    # Total memory per tensor
    total_bytes = indices_bytes + values_bytes
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024

    logger.info(f"Indices tensor memory: {indices_mb:.1f} MB ({indices_gb:.2f} GB)")
    logger.info(f"Values tensor memory: {values_mb:.1f} MB ({values_gb:.2f} GB)")
    logger.info(f"Total per tensor: {total_mb:.1f} MB ({total_gb:.2f} GB)")

    # Compare with system memory
    system_memory_gb = 7.4
    memory_percentage = (total_gb / system_memory_gb) * 100

    logger.info(f"System memory: {system_memory_gb} GB")
    logger.info(f"Tensor uses: {memory_percentage:.1f}% of system memory")

    # Multiple tensors
    for num_tensors in [2, 3]:
        multi_gb = total_gb * num_tensors
        multi_percentage = (multi_gb / system_memory_gb) * 100
        logger.info(
            f"{num_tensors} tensors: {multi_gb:.2f} GB ({multi_percentage:.1f}% of system)"
        )

    # Compare with sparse CSC matrix (for reference)
    logger.info("\n=== Comparison with CuPy/SciPy Sparse CSC ===")

    # CSC matrix: data + indices + indptr
    csc_data_bytes = nnz * 4  # float32 values
    csc_indices_bytes = nnz * 4  # int32 indices
    csc_indptr_bytes = (led_count * 3 + 1) * 4  # int32 column pointers

    csc_total_bytes = csc_data_bytes + csc_indices_bytes + csc_indptr_bytes
    csc_total_mb = csc_total_bytes / (1024 * 1024)
    csc_total_gb = csc_total_mb / 1024

    logger.info(f"CSC matrix memory: {csc_total_mb:.1f} MB ({csc_total_gb:.3f} GB)")

    memory_ratio = total_gb / csc_total_gb
    logger.info(f"PyTorch COO is {memory_ratio:.1f}x larger than CSC matrix")

    logger.info("\n=== Recommendations ===")
    if total_gb > 1.0:
        logger.warning("4D COO tensor uses >1GB memory - unsustainable on 8GB system")
        logger.info("1. Use CPU device instead of CUDA to avoid GPU memory limits")
        logger.info("2. Implement chunked processing to reduce memory footprint")
        logger.info("3. Consider staying with CuPy/SciPy sparse matrices")
        logger.info("4. Implement proper tensor cleanup/garbage collection")


if __name__ == "__main__":
    analyze_coo_tensor_memory()
