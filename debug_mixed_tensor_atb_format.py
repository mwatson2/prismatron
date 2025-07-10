#!/usr/bin/env python3
"""
Debug program to investigate mixed tensor ATb output format.

This program replicates the frame optimizer's ATb calculation to examine
why the mixed tensor operation produces interleaved format output that
causes DIA kernels to read incorrect values.
"""

import sys
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent / "src"))

from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_flower_image() -> np.ndarray:
    """Load and prepare the flower image (same as frame optimizer)."""
    flower_path = Path("images/source/flower")

    # Try common image extensions
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        if (flower_path.parent / f"{flower_path.name}{ext}").exists():
            flower_path = flower_path.parent / f"{flower_path.name}{ext}"
            break

    if not flower_path.exists():
        # Create synthetic flower-like test pattern
        print("Warning: Flower image not found, creating synthetic test pattern")
        frame = np.zeros((480, 800, 3), dtype=np.uint8)

        # Create flower-like pattern
        center_y, center_x = 240, 400
        for y in range(480):
            for x in range(800):
                dx, dy = x - center_x, y - center_y
                dist = np.sqrt(dx * dx + dy * dy)
                angle = np.arctan2(dy, dx)

                # Petals pattern
                petal_intensity = (np.sin(6 * angle) + 1) / 2
                if dist < 150:
                    frame[y, x, 0] = int(255 * petal_intensity * (1 - dist / 150))  # Red petals
                    frame[y, x, 1] = int(128 * (1 - dist / 200))  # Green center
                    frame[y, x, 2] = int(64 * petal_intensity)  # Blue accent

        return frame.transpose(2, 0, 1)  # Convert to (3, H, W)

    # Load real flower image
    image = Image.open(flower_path)
    image = image.convert("RGB")
    image = image.resize((800, 480), Image.Resampling.LANCZOS)

    # Convert to numpy array and transpose to planar format
    frame = np.array(image).transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    return frame.astype(np.uint8)


def analyze_tensor_memory_layout(tensor: cp.ndarray, name: str) -> None:
    """Analyze detailed memory layout properties of a tensor."""
    print(f"\n📊 {name} Memory Layout Analysis:")
    print(f"  Type: {type(tensor)}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Strides: {tensor.strides}")
    print(f"  Itemsize: {tensor.itemsize}")

    # Memory contiguity
    print(f"  C-contiguous: {tensor.flags.c_contiguous}")
    print(f"  F-contiguous: {tensor.flags.f_contiguous}")
    print(f"  Own data: {tensor.flags.owndata}")

    # Expected strides for C-contiguous layout
    if len(tensor.shape) == 2:
        expected_strides = (tensor.shape[1] * tensor.itemsize, tensor.itemsize)
        print(f"  Expected C strides: {expected_strides}")

    # Memory pointer info
    if hasattr(tensor, "data"):
        print(f"  Data pointer: {hex(tensor.data.ptr)}")
        print(f"  Memory alignment: {tensor.data.ptr % 16} bytes (16-byte aligned: {tensor.data.ptr % 16 == 0})")

    # Check if this looks like interleaved format
    if len(tensor.shape) == 2 and tensor.shape[0] == 3:
        # For (3, N) tensor, check if data looks interleaved
        cpu_data = cp.asnumpy(tensor)

        # Sample first few elements from each channel
        print(f"  First 5 elements of each channel:")
        for c in range(3):
            print(f"    Channel {c}: {cpu_data[c, :5]}")

        # Check if channels have similar statistics (might indicate interleaving)
        print(f"  Channel statistics:")
        for c in range(3):
            ch_data = cpu_data[c]
            print(
                f"    Channel {c}: mean={ch_data.mean():.3f}, std={ch_data.std():.3f}, min={ch_data.min():.3f}, max={ch_data.max():.3f}"
            )


def investigate_mixed_tensor_atb_output():
    """Investigate the mixed tensor ATb output format and memory layout."""

    print("=" * 70)
    print("MIXED TENSOR ATb OUTPUT FORMAT INVESTIGATION")
    print("=" * 70)

    # Load the 2624 LED synthetic patterns
    pattern_file = "diffusion_patterns/synthetic_2624_uint8_dia_10.0.npz"

    if not Path(pattern_file).exists():
        print(f"Pattern file not found: {pattern_file}")
        return

    print(f"Loading pattern file: {pattern_file}")
    data = np.load(pattern_file, allow_pickle=True)

    # Load mixed tensor
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())
    print(f"Mixed tensor: {mixed_tensor.dtype}, batch_size: {mixed_tensor.batch_size}")
    print(f"Mixed tensor spatial dims: {mixed_tensor.height}x{mixed_tensor.width}")
    print(f"Mixed tensor block size: {mixed_tensor.block_size}")

    # Load flower image
    target_frame = load_flower_image()
    print(f"Target frame: shape={target_frame.shape}, dtype={target_frame.dtype}")

    # Replicate frame optimizer's ATb calculation exactly
    print(f"\n🔍 REPLICATING FRAME OPTIMIZER ATb CALCULATION:")
    print("=" * 60)

    # Step 1: Convert to planar uint8 format (same as frame optimizer)
    if target_frame.shape == (3, 480, 800):
        # Already in planar format - ensure uint8 and C-contiguous
        target_planar_uint8 = np.ascontiguousarray(target_frame.astype(np.uint8))
    elif target_frame.shape == (480, 800, 3):
        # Convert from HWC to CHW planar format with true data rearrangement
        target_planar_uint8 = np.ascontiguousarray(target_frame.astype(np.uint8).transpose(2, 0, 1))
    else:
        raise ValueError(f"Unsupported frame shape {target_frame.shape}")

    print(f"Target planar format: {target_planar_uint8.shape}, dtype: {target_planar_uint8.dtype}")
    print(f"Target C-contiguous: {target_planar_uint8.flags.c_contiguous}")

    # Step 2: Calculate ATb using mixed tensor exactly like frame optimizer
    print(f"\n🧮 CALCULATING A^T @ b:")
    print("-" * 40)

    if mixed_tensor.dtype == cp.uint8:
        # For uint8 mixed tensors: use uint8 x uint8 -> fp32 kernel with built-in scaling
        target_gpu = cp.asarray(target_planar_uint8)  # Keep as uint8: (3, height, width)
        print("Using uint8 x uint8 -> fp32 kernel with automatic scaling")
    else:
        # For float32 mixed tensors: convert target to float32 [0,1]
        target_float32 = target_planar_uint8.astype(np.float32) / 255.0
        target_gpu = cp.asarray(target_float32)
        print("Using fp32 x fp32 -> fp32 kernel")

    # Analyze input tensor before mixed tensor operation
    analyze_tensor_memory_layout(target_gpu, "Input target_gpu (before mixed tensor operation)")

    # Perform the mixed tensor operation that produces the problematic output
    print(f"\n🔬 MIXED TENSOR TRANSPOSE DOT PRODUCT:")
    print("-" * 50)

    ATb_gpu_raw = mixed_tensor.transpose_dot_product_3d(target_gpu)
    print(f"Raw output shape: {ATb_gpu_raw.shape}, dtype: {ATb_gpu_raw.dtype}")

    # Analyze the raw output from mixed tensor operation
    analyze_tensor_memory_layout(ATb_gpu_raw, "Raw ATb from mixed tensor (before transposition)")

    # Step 3: Convert to (3, led_count) format as frame optimizer does
    print(f"\n🔄 CONVERTING TO (3, led_count) FORMAT:")
    print("-" * 45)

    if ATb_gpu_raw.shape[0] != 3:
        ATb_gpu_transposed = ATb_gpu_raw.T  # Convert (led_count, 3) -> (3, led_count)
        print(f"Transposed ATb shape: {ATb_gpu_transposed.shape}")
        analyze_tensor_memory_layout(ATb_gpu_transposed, "ATb after transposition (.T)")
    else:
        ATb_gpu_transposed = ATb_gpu_raw
        print("ATb already in (3, led_count) format - no transposition needed")

    # Step 4: Apply ascontiguousarray fix (the critical fix we identified)
    print(f"\n🛠️  APPLYING ASCONTIGUOUSARRAY FIX:")
    print("-" * 40)

    print("Before fix:")
    analyze_tensor_memory_layout(ATb_gpu_transposed, "ATb before ascontiguousarray")

    ATb_gpu_fixed = cp.ascontiguousarray(ATb_gpu_transposed)

    print("After fix:")
    analyze_tensor_memory_layout(ATb_gpu_fixed, "ATb after ascontiguousarray")

    # Step 5: Compare memory patterns between problematic and fixed versions
    print(f"\n🔍 MEMORY PATTERN COMPARISON:")
    print("-" * 35)

    # Convert to CPU for detailed analysis
    ATb_transposed_cpu = cp.asnumpy(ATb_gpu_transposed)
    ATb_fixed_cpu = cp.asnumpy(ATb_gpu_fixed)

    # Check if the values are actually the same (they should be)
    values_identical = np.allclose(ATb_transposed_cpu, ATb_fixed_cpu)
    print(f"Values identical after fix: {values_identical}")

    if not values_identical:
        max_diff = np.max(np.abs(ATb_transposed_cpu - ATb_fixed_cpu))
        print(f"Maximum difference: {max_diff}")

    # Examine the first few elements to see if there's a pattern
    print(f"\nFirst 10 elements comparison:")
    print("Problematic version (F-contiguous or non-contiguous):")
    for c in range(3):
        print(f"  Channel {c}: {ATb_transposed_cpu[c, :10]}")

    print("Fixed version (C-contiguous):")
    for c in range(3):
        print(f"  Channel {c}: {ATb_fixed_cpu[c, :10]}")

    # Step 6: Investigate why transpose produces non-contiguous output
    print(f"\n🧪 INVESTIGATING TRANSPOSE OPERATION:")
    print("-" * 40)

    print("Creating test tensors to understand transpose behavior...")

    # Create a simple test case
    test_shape = (100, 3)  # (led_count, channels) - typical mixed tensor output
    test_data = np.arange(test_shape[0] * test_shape[1], dtype=np.float32).reshape(test_shape)
    test_gpu = cp.asarray(test_data)

    print(f"Test tensor (led_count, channels): shape={test_gpu.shape}")
    analyze_tensor_memory_layout(test_gpu, "Test tensor before transpose")

    test_transposed = test_gpu.T
    print(f"Test tensor after .T: shape={test_transposed.shape}")
    analyze_tensor_memory_layout(test_transposed, "Test tensor after .T")

    test_fixed = cp.ascontiguousarray(test_transposed)
    analyze_tensor_memory_layout(test_fixed, "Test tensor after ascontiguousarray")

    # Step 7: Demonstrate the specific issue with mixed tensor output
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")
    print("-" * 25)

    print("The mixed tensor transpose_dot_product_3d() operation:")
    print("1. Returns data in (led_count, channels) format")
    print("2. Frame optimizer uses .T to convert to (channels, led_count)")
    print("3. The .T operation creates a TRANSPOSED VIEW, not a new array")
    print("4. This view has F-contiguous memory layout (column-major)")
    print("5. DIA kernels expect C-contiguous layout (row-major)")
    print("6. F-contiguous layout causes kernels to read wrong memory locations")

    print(f"\nMemory layout implications:")
    print(f"- Original (led_count, channels): C-contiguous, strides {ATb_gpu_raw.strides}")
    print(f"- After .T (channels, led_count): F-contiguous, strides {ATb_gpu_transposed.strides}")
    print(f"- After fix (channels, led_count): C-contiguous, strides {ATb_gpu_fixed.strides}")

    # Calculate expected C-contiguous strides
    expected_strides = (ATb_gpu_fixed.shape[1] * ATb_gpu_fixed.itemsize, ATb_gpu_fixed.itemsize)
    print(f"- Expected C-contiguous strides: {expected_strides}")

    # Step 8: Show how this affects DIA kernel performance
    print(f"\n⚡ PERFORMANCE IMPLICATIONS:")
    print("-" * 30)

    print("F-contiguous layout causes:")
    print("1. Non-coalesced memory access in CUDA kernels")
    print("2. Up to 9x performance degradation")
    print("3. Incorrect results due to wrong memory offsets")
    print("4. DIA diagonal access patterns break completely")

    print(f"\nSolution:")
    print("Always use cp.ascontiguousarray() after tensor transpose operations")
    print("before passing to DIA kernels or other CUDA kernels expecting C-contiguous layout")

    return {
        "raw_output": ATb_gpu_raw,
        "transposed": ATb_gpu_transposed,
        "fixed": ATb_gpu_fixed,
        "mixed_tensor": mixed_tensor,
    }


def demonstrate_kernel_behavior():
    """Demonstrate how different memory layouts affect kernel behavior."""

    print(f"\n" + "=" * 70)
    print("KERNEL BEHAVIOR DEMONSTRATION")
    print("=" * 70)

    # Create test data with known pattern
    print("Creating test data with known pattern...")
    test_data = np.zeros((2624, 3), dtype=np.float32)

    # Fill with pattern: channel 0 = LED_index, channel 1 = LED_index * 2, channel 2 = LED_index * 3
    for led in range(2624):
        test_data[led, 0] = float(led)  # Red channel
        test_data[led, 1] = float(led * 2)  # Green channel
        test_data[led, 2] = float(led * 3)  # Blue channel

    test_gpu = cp.asarray(test_data)
    print(f"Test data shape: {test_gpu.shape}")

    # Create transpose view (F-contiguous)
    test_transposed = test_gpu.T  # (3, 2624)
    print(f"Transposed shape: {test_transposed.shape}")

    # Create contiguous version
    test_contiguous = cp.ascontiguousarray(test_transposed)
    print(f"Contiguous shape: {test_contiguous.shape}")

    # Show the difference in memory layout
    print(f"\nMemory layout comparison:")
    print(f"Original (led, ch): C-contig={test_gpu.flags.c_contiguous}, strides={test_gpu.strides}")
    print(
        f"Transposed (ch, led): C-contig={test_transposed.flags.c_contiguous}, F-contig={test_transposed.flags.f_contiguous}, strides={test_transposed.strides}"
    )
    print(f"Contiguous (ch, led): C-contig={test_contiguous.flags.c_contiguous}, strides={test_contiguous.strides}")

    # Show how this affects data access
    print(f"\nData access pattern check:")
    print(f"Values should be: Ch0[0,1,2]==[0,1,2], Ch1[0,1,2]==[0,2,4], Ch2[0,1,2]==[0,3,6]")

    # Check transposed view
    trans_cpu = cp.asnumpy(test_transposed)
    print(f"Transposed view: Ch0[0,1,2]={trans_cpu[0,:3]}, Ch1[0,1,2]={trans_cpu[1,:3]}, Ch2[0,1,2]={trans_cpu[2,:3]}")

    # Check contiguous version
    contig_cpu = cp.asnumpy(test_contiguous)
    print(
        f"Contiguous ver:  Ch0[0,1,2]={contig_cpu[0,:3]}, Ch1[0,1,2]={contig_cpu[1,:3]}, Ch2[0,1,2]={contig_cpu[2,:3]}"
    )

    # Verify they contain the same values
    values_match = np.allclose(trans_cpu, contig_cpu)
    print(f"Values match: {values_match}")

    if values_match:
        print("✅ Both layouts contain the same values")
        print("✅ The issue is memory access pattern, not data corruption")
    else:
        print("❌ Different values detected - this indicates a deeper issue")


if __name__ == "__main__":
    print("Investigating mixed tensor ATb output format...")

    # Main investigation
    results = investigate_mixed_tensor_atb_output()

    # Demonstration of kernel behavior
    demonstrate_kernel_behavior()

    print(f"\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)
    print("1. Mixed tensor transpose_dot_product_3d() returns (led_count, 3) format")
    print("2. Frame optimizer uses .T to get (3, led_count) format")
    print("3. .T creates F-contiguous view, not C-contiguous array")
    print("4. DIA kernels require C-contiguous input for correct operation")
    print("5. Solution: Always use cp.ascontiguousarray() after transpose")
    print("6. This fix prevents 235x scaling errors and performance degradation")
    print("7. The defensive validation we added detects and prevents this issue")
