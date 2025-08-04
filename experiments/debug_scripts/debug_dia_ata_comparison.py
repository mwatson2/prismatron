#!/usr/bin/env python3
"""
Debug script to compare DIA vs dense ATA inverse operations.
Tests einsum operation inv(ATA) @ x for both formats.
"""

import sys
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent / "src"))

from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_flower_image() -> np.ndarray:
    """Load and prepare the flower image for testing (same as frame optimizer)."""
    flower_path = Path("images/source/flower")

    # Try common image extensions
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        if (flower_path.parent / f"{flower_path.name}{ext}").exists():
            flower_path = flower_path.parent / f"{flower_path.name}{ext}"
            break

    if not flower_path.exists():
        # Fallback: create a synthetic flower-like test pattern (same as frame optimizer)
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


def calculate_atb_like_frame_optimizer(
    target_frame: np.ndarray, mixed_tensor: SingleBlockMixedSparseTensor
) -> cp.ndarray:
    """Calculate A^T @ b exactly the same way as frame optimizer."""
    # Same logic as frame optimizer _calculate_atb
    if mixed_tensor.dtype == cp.uint8:
        # For uint8 mixed tensors: use uint8 x uint8 -> fp32 kernel with built-in scaling
        target_gpu = cp.asarray(target_frame)  # Keep as uint8: (3, height, width)
        print("Using uint8 x uint8 -> fp32 kernel with automatic scaling")
    else:
        # For float32 mixed tensors: convert target to float32 [0,1] for fp32 x fp32 -> fp32 kernel
        target_float32 = target_frame.astype(np.float32) / 255.0
        target_gpu = cp.asarray(target_float32)
        print("Using fp32 x fp32 -> fp32 kernel")

    result = mixed_tensor.transpose_dot_product_3d(target_gpu)  # Shape: (led_count, 3), dtype: float32

    # Convert to (3, led_count) format as in frame optimizer
    if result.shape[0] != 3:
        result = result.T  # Convert (led_count, 3) -> (3, led_count)

    return result


def compare_dia_vs_dense_ata_inverse():
    """Compare DIA vs dense ATA inverse operations using real A^T @ b from frame optimizer."""

    # Test with the full diagonal version first
    pattern_file = "diffusion_patterns/synthetic_2624_uint8_dia_10.0.npz"

    if not Path(pattern_file).exists():
        print(f"Pattern file not found: {pattern_file}")
        return

    print(f"Loading pattern file: {pattern_file}")
    data = np.load(pattern_file, allow_pickle=True)

    print("Available keys:", list(data.keys()))

    # Load all components (same as frame optimizer)
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(data["mixed_tensor"].item())
    ata_inverse_dense = data["ata_inverse"]  # (3, 2624, 2624)
    ata_inverse_dia_dict = data["ata_inverse_dia"].item()
    ata_inverse_dia = DiagonalATAMatrix.from_dict(ata_inverse_dia_dict)

    print(f"Mixed tensor: {mixed_tensor.dtype}, batch_size: {mixed_tensor.batch_size}")
    print(f"Dense ATA inverse: shape={ata_inverse_dense.shape}, dtype={ata_inverse_dense.dtype}")
    print(f"DIA ATA inverse: bandwidth={ata_inverse_dia.bandwidth}, k={ata_inverse_dia.k}")

    # Load flower image and calculate A^T @ b exactly like frame optimizer
    target_frame = load_flower_image()
    print(f"Target frame: shape={target_frame.shape}, dtype={target_frame.dtype}")

    # Calculate A^T @ b using exact same method as frame optimizer
    ATb_gpu_real = calculate_atb_like_frame_optimizer(target_frame, mixed_tensor)

    # CRITICAL: Apply the same fix as frame optimizer - ensure C-contiguous layout
    # This simulates the fix we added to the frame optimizer
    ATb_gpu_real = cp.ascontiguousarray(ATb_gpu_real)

    ATb_cpu_real = cp.asnumpy(ATb_gpu_real)

    print("\nA^T @ b from frame optimizer method:")
    print(f"  Shape: {ATb_gpu_real.shape}")
    print(f"  Range: [{ATb_cpu_real.min():.2f}, {ATb_cpu_real.max():.2f}]")
    print(f"  RMS: {np.sqrt(np.mean(ATb_cpu_real**2)):.2f}")
    print(f"  Dtype: {ATb_gpu_real.dtype}")

    # NOW TEST: Replace with random values of same dtype and similar range
    print("\n🧪 TESTING 1: Using uniform random A^T @ b with same dtype/range...")
    np.random.seed(42)
    ATb_uniform = np.random.rand(3, 2624).astype(np.float32) * 2500.0  # Similar range [0, 2500]
    ATb_uniform_gpu = cp.asarray(ATb_uniform)
    ATb_uniform_cpu = cp.asnumpy(ATb_uniform_gpu)

    print("Uniform random A^T @ b:")
    print(f"  Shape: {ATb_uniform_gpu.shape}")
    print(f"  Range: [{ATb_uniform_cpu.min():.2f}, {ATb_uniform_cpu.max():.2f}]")
    print(f"  RMS: {np.sqrt(np.mean(ATb_uniform_cpu**2)):.2f}")
    print(f"  Dtype: {ATb_uniform_gpu.dtype}")

    # NEW TEST: Element-by-element copy from actual ATb values (no random sampling)
    print("\n🧪 TESTING 2: Using element-by-element copy from real A^T @ b values...")
    # Create a copy by explicitly copying each element (to test if storage layout matters)
    ATb_copied = np.zeros_like(ATb_cpu_real, dtype=np.float32)
    for c in range(3):
        for led in range(2624):
            ATb_copied[c, led] = ATb_cpu_real[c, led]  # Element-by-element copy

    ATb_copied_gpu = cp.asarray(ATb_copied)
    ATb_copied_cpu = cp.asnumpy(ATb_copied_gpu)

    print("Element-by-element copied from real A^T @ b:")
    print(f"  Shape: {ATb_copied_gpu.shape}")
    print(f"  Range: [{ATb_copied_cpu.min():.2f}, {ATb_copied_cpu.max():.2f}]")
    print(f"  RMS: {np.sqrt(np.mean(ATb_copied_cpu**2)):.2f}")
    print(f"  Dtype: {ATb_copied_gpu.dtype}")

    # NEW TEST: Random sampling from actual ATb values (create before analysis)
    print("\n🧪 TESTING 3: Using random sampling from real A^T @ b values...")
    np.random.seed(123)  # Different seed for variety
    # Flatten real ATb, sample with replacement, reshape
    ATb_real_flat = ATb_cpu_real.flatten()
    sampled_indices = np.random.choice(len(ATb_real_flat), size=(3, 2624))
    ATb_sampled = ATb_real_flat[sampled_indices].astype(np.float32)
    ATb_sampled_gpu = cp.asarray(ATb_sampled)
    ATb_sampled_cpu = cp.asnumpy(ATb_sampled_gpu)

    print("Sampled from real A^T @ b:")
    print(f"  Shape: {ATb_sampled_gpu.shape}")
    print(f"  Range: [{ATb_sampled_cpu.min():.2f}, {ATb_sampled_cpu.max():.2f}]")
    print(f"  RMS: {np.sqrt(np.mean(ATb_sampled_cpu**2)):.2f}")
    print(f"  Dtype: {ATb_sampled_gpu.dtype}")

    # 🔍 TENSOR PROPERTIES INVESTIGATION
    print("\n🔍 DETAILED TENSOR PROPERTIES INVESTIGATION:")
    print("=" * 60)

    def analyze_tensor_properties(tensor, name):
        print(f"\n📊 {name}:")
        print(f"  Type: {type(tensor)}")
        print(f"  Device: {tensor.device if hasattr(tensor, 'device') else 'N/A'}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Strides: {tensor.strides}")
        print(f"  Itemsize: {tensor.itemsize}")
        print(f"  Is contiguous: {tensor.flags.c_contiguous if hasattr(tensor, 'flags') else 'N/A'}")
        print(f"  Is F-contiguous: {tensor.flags.f_contiguous if hasattr(tensor, 'flags') else 'N/A'}")
        print(
            f"  Memory layout: {'C' if tensor.flags.c_contiguous else 'F' if tensor.flags.f_contiguous else 'Non-contiguous' if hasattr(tensor, 'flags') else 'Unknown'}"
        )

        # CuPy specific properties
        if hasattr(tensor, "data"):
            print(f"  Data pointer: {hex(tensor.data.ptr)}")
            print(f"  Data memory size: {tensor.data.mem.size if hasattr(tensor.data, 'mem') else 'N/A'}")
        if hasattr(tensor, "flags"):
            available_flags = [attr for attr in dir(tensor.flags) if not attr.startswith("_")]
            print(f"  Available flags: {available_flags}")
            print(f"  Flags.owndata: {tensor.flags.owndata}")
            print(f"  Flags.forc: {getattr(tensor.flags, 'forc', 'N/A')}")
            print(f"  Flags.fnc: {getattr(tensor.flags, 'fnc', 'N/A')}")

        # Memory footprint
        print(f"  Memory size (bytes): {tensor.nbytes}")
        print(f"  Size (elements): {tensor.size}")

        # Check for special CuPy array properties
        if hasattr(tensor, "__cuda_array_interface__"):
            print(f"  CUDA array interface: {tensor.__cuda_array_interface__}")

        # First few elements for validation
        if tensor.size > 0:
            flat_tensor = tensor.flatten()
            n_show = min(10, tensor.size)
            print(f"  First {n_show} elements: {flat_tensor[:n_show]}")

    # Analyze all ATb tensors
    analyze_tensor_properties(ATb_gpu_real, "Real ATb (ORIGINAL from mixed tensor)")
    analyze_tensor_properties(ATb_copied_gpu, "Element-by-element copied ATb")
    analyze_tensor_properties(ATb_uniform_gpu, "Uniform random ATb")
    analyze_tensor_properties(ATb_sampled_gpu, "Sampled from real ATb")

    # Additional deep inspection of the problematic tensor
    print("\n🔬 DEEP INSPECTION OF ORIGINAL ATb TENSOR:")
    print("=" * 60)
    print("Original ATb tensor creation path analysis:")
    print("1. Created by: mixed_tensor.transpose_dot_product_3d(target_gpu)")
    print(f"2. Mixed tensor dtype: {mixed_tensor.dtype}")
    print(f"3. Target dtype: {target_frame.dtype}")

    # Check if the original tensor has any unusual memory characteristics
    if hasattr(ATb_gpu_real, "__array_interface__"):
        print(f"Array interface: {ATb_gpu_real.__array_interface__}")

    # Check tensor base and parent relationships
    if hasattr(ATb_gpu_real, "base"):
        print(f"ATb_gpu_real.base: {ATb_gpu_real.base}")
    if hasattr(ATb_copied_gpu, "base"):
        print(f"ATb_copied_gpu.base: {ATb_copied_gpu.base}")

    # Calculate dense reference first for contiguity test
    ata_inverse_gpu = cp.asarray(ata_inverse_dense)
    result_dense_real = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_gpu_real)
    result_dense_real_cpu = cp.asnumpy(result_dense_real)
    dense_real_rms = np.sqrt(np.mean(result_dense_real_cpu**2))

    # Check if we can force contiguity and see if that fixes it
    print("\n🧪 TESTING CONTIGUITY FIX:")
    ATb_contiguous = cp.ascontiguousarray(ATb_gpu_real)
    analyze_tensor_properties(ATb_contiguous, "Contiguous copy of real ATb")

    # Test if the contiguous version works correctly
    print("\n9️⃣ DIA @ Contiguous Real ATb:")
    print("  🔍 Testing if making the tensor contiguous fixes the issue...")
    result_dia_contiguous = ata_inverse_dia.multiply_3d(ATb_contiguous, debug_logging=True)
    result_dia_contiguous_cpu = cp.asnumpy(result_dia_contiguous)
    dia_contiguous_rms = np.sqrt(np.mean(result_dia_contiguous_cpu**2))
    print(f"   Range: [{result_dia_contiguous_cpu.min():.6f}, {result_dia_contiguous_cpu.max():.6f}]")
    print(f"   RMS: {dia_contiguous_rms:.6f}")
    contiguous_scaling = dia_contiguous_rms / dense_real_rms
    print(f"   Scaling vs dense: {contiguous_scaling:.2f}x")

    if abs(contiguous_scaling - 1.0) < 0.1:
        print("   ✅ CONTIGUOUS COPY FIXES THE ISSUE!")
        print("   → The problem is NON-CONTIGUOUS memory layout from mixed tensor operation")
    else:
        print(f"   ❌ Contiguous copy still has issues ({contiguous_scaling:.2f}x)")
        print("   → The problem is deeper than just memory contiguity")

    # NEW TEST: Random sampling from actual ATb values (create before analysis)
    print("\n🧪 TESTING 3: Using random sampling from real A^T @ b values...")
    np.random.seed(123)  # Different seed for variety
    # Flatten real ATb, sample with replacement, reshape
    ATb_real_flat = ATb_cpu_real.flatten()
    sampled_indices = np.random.choice(len(ATb_real_flat), size=(3, 2624))
    ATb_sampled = ATb_real_flat[sampled_indices].astype(np.float32)
    ATb_sampled_gpu = cp.asarray(ATb_sampled)
    ATb_sampled_cpu = cp.asnumpy(ATb_sampled_gpu)

    print("Sampled from real A^T @ b:")
    print(f"  Shape: {ATb_sampled_gpu.shape}")
    print(f"  Range: [{ATb_sampled_cpu.min():.2f}, {ATb_sampled_cpu.max():.2f}]")
    print(f"  RMS: {np.sqrt(np.mean(ATb_sampled_cpu**2)):.2f}")
    print(f"  Dtype: {ATb_sampled_gpu.dtype}")

    # Dense reference was already calculated above

    print("\n" + "=" * 60)
    print("TESTING ALL EIGHT COMBINATIONS")
    print("=" * 60)

    # 1. Dense @ Real ATb
    print("\n1️⃣ Dense @ Real ATb:")
    print(f"   Range: [{result_dense_real_cpu.min():.6f}, {result_dense_real_cpu.max():.6f}]")
    print(f"   RMS: {dense_real_rms:.6f}")

    # 2. Dense @ Uniform Random
    print("\n2️⃣ Dense @ Uniform Random:")
    result_dense_uniform = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_uniform_gpu)
    result_dense_uniform_cpu = cp.asnumpy(result_dense_uniform)
    dense_uniform_rms = np.sqrt(np.mean(result_dense_uniform_cpu**2))
    print(f"   Range: [{result_dense_uniform_cpu.min():.6f}, {result_dense_uniform_cpu.max():.6f}]")
    print(f"   RMS: {dense_uniform_rms:.6f}")

    # 3. Dense @ Element-by-element Copied
    print("\n3️⃣ Dense @ Element-by-element Copied:")
    result_dense_copied = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_copied_gpu)
    result_dense_copied_cpu = cp.asnumpy(result_dense_copied)
    dense_copied_rms = np.sqrt(np.mean(result_dense_copied_cpu**2))
    print(f"   Range: [{result_dense_copied_cpu.min():.6f}, {result_dense_copied_cpu.max():.6f}]")
    print(f"   RMS: {dense_copied_rms:.6f}")

    # 4. Dense @ Sampled from Real
    print("\n4️⃣ Dense @ Sampled from Real:")
    result_dense_sampled = cp.einsum("ijk,ik->ij", ata_inverse_gpu, ATb_sampled_gpu)
    result_dense_sampled_cpu = cp.asnumpy(result_dense_sampled)
    dense_sampled_rms = np.sqrt(np.mean(result_dense_sampled_cpu**2))
    print(f"   Range: [{result_dense_sampled_cpu.min():.6f}, {result_dense_sampled_cpu.max():.6f}]")
    print(f"   RMS: {dense_sampled_rms:.6f}")

    # 5. DIA @ Real ATb (with logging!)
    print("\n5️⃣ DIA @ Real ATb:")
    print("  🔍 Enabling debug logging to see which kernel executes...")
    result_dia_real = ata_inverse_dia.multiply_3d(ATb_gpu_real, debug_logging=True)
    result_dia_real_cpu = cp.asnumpy(result_dia_real)
    dia_real_rms = np.sqrt(np.mean(result_dia_real_cpu**2))
    print(f"   Range: [{result_dia_real_cpu.min():.6f}, {result_dia_real_cpu.max():.6f}]")
    print(f"   RMS: {dia_real_rms:.6f}")

    # 6. DIA @ Uniform Random (with logging!)
    print("\n6️⃣ DIA @ Uniform Random:")
    print("  🔍 Enabling debug logging to see which kernel executes...")
    result_dia_uniform = ata_inverse_dia.multiply_3d(ATb_uniform_gpu, debug_logging=True)
    result_dia_uniform_cpu = cp.asnumpy(result_dia_uniform)
    dia_uniform_rms = np.sqrt(np.mean(result_dia_uniform_cpu**2))
    print(f"   Range: [{result_dia_uniform_cpu.min():.6f}, {result_dia_uniform_cpu.max():.6f}]")
    print(f"   RMS: {dia_uniform_rms:.6f}")

    # 7. DIA @ Element-by-element Copied (with logging!)
    print("\n7️⃣ DIA @ Element-by-element Copied:")
    print("  🔍 Enabling debug logging to see which kernel executes...")
    result_dia_copied = ata_inverse_dia.multiply_3d(ATb_copied_gpu, debug_logging=True)
    result_dia_copied_cpu = cp.asnumpy(result_dia_copied)
    dia_copied_rms = np.sqrt(np.mean(result_dia_copied_cpu**2))
    print(f"   Range: [{result_dia_copied_cpu.min():.6f}, {result_dia_copied_cpu.max():.6f}]")
    print(f"   RMS: {dia_copied_rms:.6f}")

    # 8. DIA @ Sampled from Real (with logging!)
    print("\n8️⃣ DIA @ Sampled from Real:")
    print("  🔍 Enabling debug logging to see which kernel executes...")
    result_dia_sampled = ata_inverse_dia.multiply_3d(ATb_sampled_gpu, debug_logging=True)
    result_dia_sampled_cpu = cp.asnumpy(result_dia_sampled)
    dia_sampled_rms = np.sqrt(np.mean(result_dia_sampled_cpu**2))
    print(f"   Range: [{result_dia_sampled_cpu.min():.6f}, {result_dia_sampled_cpu.max():.6f}]")
    print(f"   RMS: {dia_sampled_rms:.6f}")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Input RMS values
    atb_real_rms = np.sqrt(np.mean(ATb_cpu_real**2))
    atb_uniform_rms = np.sqrt(np.mean(ATb_uniform_cpu**2))
    atb_copied_rms = np.sqrt(np.mean(ATb_copied_cpu**2))
    atb_sampled_rms = np.sqrt(np.mean(ATb_sampled_cpu**2))

    print("\n📊 Input RMS values:")
    print(f"   Real ATb: {atb_real_rms:.2f}")
    print(f"   Uniform random: {atb_uniform_rms:.2f}")
    print(f"   Element-by-element copied: {atb_copied_rms:.2f}")
    print(f"   Sampled from real: {atb_sampled_rms:.2f}")

    # Dense behavior analysis
    print("\n📊 Dense matrix behavior:")
    print(f"   Real ATb → RMS: {dense_real_rms:.6f}")
    print(f"   Uniform → RMS: {dense_uniform_rms:.6f}")
    print(f"   Element-by-element copied → RMS: {dense_copied_rms:.6f}")
    print(f"   Sampled → RMS: {dense_sampled_rms:.6f}")
    print(f"   Scaling uniform/real: {dense_uniform_rms/dense_real_rms:.2f}x")
    print(f"   Scaling copied/real: {dense_copied_rms/dense_real_rms:.2f}x")
    print(f"   Scaling sampled/real: {dense_sampled_rms/dense_real_rms:.2f}x")

    # DIA behavior analysis
    print("\n📊 DIA matrix behavior:")
    print(f"   Real ATb → RMS: {dia_real_rms:.6f}")
    print(f"   Uniform → RMS: {dia_uniform_rms:.6f}")
    print(f"   Element-by-element copied → RMS: {dia_copied_rms:.6f}")
    print(f"   Sampled → RMS: {dia_sampled_rms:.6f}")
    print(f"   Scaling uniform/real: {dia_uniform_rms/dia_real_rms:.2f}x")
    print(f"   Scaling copied/real: {dia_copied_rms/dia_real_rms:.2f}x")
    print(f"   Scaling sampled/real: {dia_sampled_rms/dia_real_rms:.2f}x")

    # Cross comparisons: DIA vs Dense scaling
    print("\n🔍 DIA vs Dense scaling factors:")
    real_scaling = dia_real_rms / dense_real_rms
    uniform_scaling = dia_uniform_rms / dense_uniform_rms
    copied_scaling = dia_copied_rms / dense_copied_rms
    sampled_scaling = dia_sampled_rms / dense_sampled_rms

    print(f"   Real ATb: {real_scaling:.2f}x")
    print(f"   Uniform: {uniform_scaling:.2f}x")
    print(f"   Element-by-element copied: {copied_scaling:.2f}x")
    print(f"   Sampled: {sampled_scaling:.2f}x")

    # Key insight: does value distribution matter?
    print("\n🔍 Key insight - Value distribution vs memory layout impact:")
    print(f"   🔍 Element-by-element copy test: {copied_scaling:.2f}x")
    print(f"   🔍 Random sampling test: {sampled_scaling:.2f}x")
    print(f"   🔍 Real ATb test: {real_scaling:.2f}x")

    if abs(copied_scaling - real_scaling) < 1.0:
        print(f"   ❌ Element-by-element copy STILL has the same issue ({copied_scaling:.2f}x vs {real_scaling:.2f}x)")
        print("   → This proves it's NOT a memory layout or storage format issue")
        print("   → The issue is the SPATIAL ARRANGEMENT of the values themselves")
    else:
        print(f"   ✅ Element-by-element copy behaves differently ({copied_scaling:.2f}x vs {real_scaling:.2f}x)")
        print("   → This suggests a memory layout or storage format issue")

    if abs(sampled_scaling - real_scaling) < 1.0:
        print(f"   ❌ Sampled values behave like real ({sampled_scaling:.2f}x vs {real_scaling:.2f}x)")
        print("   → The issue is likely the VALUE DISTRIBUTION, not specific A^T @ b calculation")
    else:
        print(f"   ✅ Sampled values behave differently ({sampled_scaling:.2f}x vs {real_scaling:.2f}x)")
        print("   → The issue is likely the SPECIFIC A^T @ b calculation process")

    if abs(uniform_scaling - 1.0) < 0.1:
        print(f"   ✅ Uniform distribution works correctly ({uniform_scaling:.2f}x ≈ 1.0)")
        print("   → This confirms DIA matrix implementation itself is correct")
    else:
        print(f"   ❌ Even uniform distribution has issues ({uniform_scaling:.2f}x)")
        print("   → This suggests fundamental DIA matrix implementation issues")

    # Sample value comparison
    print("\n📋 Sample values comparison:")
    for i in range(min(3, result_dense_real_cpu.shape[1])):
        print(f"   LED {i}:")
        print(
            f"     Dense: real={result_dense_real_cpu[0,i]:.6f}, uniform={result_dense_uniform_cpu[0,i]:.6f}, copied={result_dense_copied_cpu[0,i]:.6f}, sampled={result_dense_sampled_cpu[0,i]:.6f}"
        )
        print(
            f"     DIA:   real={result_dia_real_cpu[0,i]:.6f}, uniform={result_dia_uniform_cpu[0,i]:.6f}, copied={result_dia_copied_cpu[0,i]:.6f}, sampled={result_dia_sampled_cpu[0,i]:.6f}"
        )

    # Visualize the input patterns
    print("\n📊 VISUALIZATION: Input tensor patterns")
    print("=" * 60)

    import matplotlib.pyplot as plt

    # Create visualization of the three input tensors
    fig, axes = plt.subplots(3, 3, figsize=(15, 8))
    fig.suptitle("Input Tensor Patterns: Real ATb vs Uniform vs Sampled", fontsize=14)

    # Normalize each tensor for visualization (0-1 range)
    def normalize_for_viz(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val > min_val:
            return (tensor - min_val) / (max_val - min_val)
        else:
            return tensor * 0

    # Real ATb
    real_norm = normalize_for_viz(ATb_cpu_real)
    for c in range(3):
        axes[0, c].imshow(real_norm[c : c + 1, :], aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axes[0, c].set_title(f"Real ATb - Channel {c}")
        axes[0, c].set_ylabel("Channel")
        if c == 1:
            axes[0, c].set_xlabel("LED Index")

    # Uniform random
    uniform_norm = normalize_for_viz(ATb_uniform_cpu)
    for c in range(3):
        axes[1, c].imshow(uniform_norm[c : c + 1, :], aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axes[1, c].set_title(f"Uniform Random - Channel {c}")
        axes[1, c].set_ylabel("Channel")
        if c == 1:
            axes[1, c].set_xlabel("LED Index")

    # Sampled from real
    sampled_norm = normalize_for_viz(ATb_sampled_cpu)
    for c in range(3):
        axes[2, c].imshow(sampled_norm[c : c + 1, :], aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axes[2, c].set_title(f"Sampled from Real - Channel {c}")
        axes[2, c].set_ylabel("Channel")
        if c == 1:
            axes[2, c].set_xlabel("LED Index")

    plt.tight_layout()
    plt.savefig("input_tensor_patterns.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print some statistics about spatial correlation
    print("\n📈 Spatial correlation analysis:")

    # Check channel correlations
    for i in range(3):
        for j in range(i + 1, 3):
            corr_real = np.corrcoef(ATb_cpu_real[i], ATb_cpu_real[j])[0, 1]
            corr_uniform = np.corrcoef(ATb_uniform_cpu[i], ATb_uniform_cpu[j])[0, 1]
            corr_sampled = np.corrcoef(ATb_sampled_cpu[i], ATb_sampled_cpu[j])[0, 1]
            print(
                f"   Channel {i}-{j} correlation: Real={corr_real:.3f}, Uniform={corr_uniform:.3f}, Sampled={corr_sampled:.3f}"
            )

    # Check spatial smoothness (adjacent LED correlation)
    def spatial_smoothness(tensor):
        # Correlation between adjacent LEDs across all channels
        adj_corr = []
        for c in range(3):
            if tensor.shape[1] > 1:
                corr = np.corrcoef(tensor[c, :-1], tensor[c, 1:])[0, 1]
                if not np.isnan(corr):
                    adj_corr.append(corr)
        return np.mean(adj_corr) if adj_corr else 0

    smooth_real = spatial_smoothness(ATb_cpu_real)
    smooth_uniform = spatial_smoothness(ATb_uniform_cpu)
    smooth_sampled = spatial_smoothness(ATb_sampled_cpu)

    print(
        f"   Adjacent LED correlation: Real={smooth_real:.3f}, Uniform={smooth_uniform:.3f}, Sampled={smooth_sampled:.3f}"
    )

    # Check for potential transposition issues by looking at value patterns
    print("\n🔍 Potential transposition check:")
    print(f"   Real ATb first 10 values Ch0: {ATb_cpu_real[0, :10]}")
    print(f"   Real ATb first 10 values Ch1: {ATb_cpu_real[1, :10]}")
    print(f"   Real ATb first 10 values Ch2: {ATb_cpu_real[2, :10]}")

    # Check if the channels have very different statistics (might indicate transposition)
    for c in range(3):
        print(
            f"   Channel {c}: mean={ATb_cpu_real[c].mean():.2f}, std={ATb_cpu_real[c].std():.2f}, min={ATb_cpu_real[c].min():.2f}, max={ATb_cpu_real[c].max():.2f}"
        )

    return {
        "dense_real": result_dense_real_cpu,
        "dense_uniform": result_dense_uniform_cpu,
        "dense_copied": result_dense_copied_cpu,
        "dense_sampled": result_dense_sampled_cpu,
        "dia_real": result_dia_real_cpu,
        "dia_uniform": result_dia_uniform_cpu,
        "dia_copied": result_dia_copied_cpu,
        "dia_sampled": result_dia_sampled_cpu,
    }


def test_diagonal_reduction_effect():
    """Test the effect of reducing diagonals (10.0 vs 1.0 factor)."""

    print("\n" + "=" * 60)
    print("TESTING DIAGONAL REDUCTION EFFECT")
    print("=" * 60)

    # Test with full diagonals (10.0 factor)
    pattern_file_full = "diffusion_patterns/synthetic_2624_uint8_dia_10.0.npz"
    pattern_file_reduced = "diffusion_patterns/synthetic_2624_uint8_dia_1.0.npz"

    if not Path(pattern_file_full).exists():
        print(f"Full diagonal file not found: {pattern_file_full}")
        return

    if not Path(pattern_file_reduced).exists():
        print(f"Reduced diagonal file not found: {pattern_file_reduced}")
        return

    # Load both files
    data_full = np.load(pattern_file_full, allow_pickle=True)
    data_reduced = np.load(pattern_file_reduced, allow_pickle=True)

    # Create DIA matrices
    ata_inv_full = DiagonalATAMatrix.from_dict(data_full["ata_inverse_dia"].item())
    ata_inv_reduced = DiagonalATAMatrix.from_dict(data_reduced["ata_inverse_dia"].item())

    print(f"Full diagonals (10.0): k={ata_inv_full.k}, bandwidth={ata_inv_full.bandwidth}")
    print(f"Reduced diagonals (1.0): k={ata_inv_reduced.k}, bandwidth={ata_inv_reduced.bandwidth}")

    # Test with same realistic vector
    np.random.seed(42)
    test_vector = np.random.rand(3, 2624).astype(np.float32) * 4096.0  # Realistic A^T @ b range
    test_vector_gpu = cp.asarray(test_vector)

    # Get results from both
    result_full = ata_inv_full.multiply_3d(test_vector_gpu)
    result_reduced = ata_inv_reduced.multiply_3d(test_vector_gpu)

    result_full_cpu = cp.asnumpy(result_full)
    result_reduced_cpu = cp.asnumpy(result_reduced)

    # Compare
    max_diff = np.max(np.abs(result_full_cpu - result_reduced_cpu))
    mean_diff = np.mean(np.abs(result_full_cpu - result_reduced_cpu))
    rms_diff = np.sqrt(np.mean((result_full_cpu - result_reduced_cpu) ** 2))

    full_rms = np.sqrt(np.mean(result_full_cpu**2))

    print("\nComparison between full (10.0) and reduced (1.0) diagonals:")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"RMS difference: {rms_diff:.6f}")
    if full_rms > 1e-10:
        print(f"Relative error: {rms_diff/full_rms*100:.2f}%")

    # Also compare with dense reference
    ata_inverse_dense = data_full["ata_inverse"]
    ata_inverse_gpu = cp.asarray(ata_inverse_dense)
    result_dense = cp.einsum("ijk,ik->ij", ata_inverse_gpu, test_vector_gpu)
    result_dense_cpu = cp.asnumpy(result_dense)

    # Compare reduced vs dense
    max_diff_dense = np.max(np.abs(result_dense_cpu - result_reduced_cpu))
    rms_diff_dense = np.sqrt(np.mean((result_dense_cpu - result_reduced_cpu) ** 2))
    dense_rms = np.sqrt(np.mean(result_dense_cpu**2))

    print("\nComparison between reduced (1.0) and dense reference:")
    print(f"Max difference: {max_diff_dense:.6f}")
    print(f"RMS difference: {rms_diff_dense:.6f}")
    if dense_rms > 1e-10:
        print(f"Relative error: {rms_diff_dense/dense_rms*100:.2f}%")


if __name__ == "__main__":
    # First test: compare DIA vs dense
    print("Testing DIA vs Dense ATA inverse operations...")
    compare_dia_vs_dense_ata_inverse()

    # Second test: effect of diagonal reduction
    test_diagonal_reduction_effect()
