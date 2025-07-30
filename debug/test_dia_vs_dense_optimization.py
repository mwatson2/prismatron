#!/usr/bin/env python3
"""
Test DIA vs Dense ATA Matrix Optimization Comparison.

This script tests that the frame optimization produces identical results
when using DIA vs Dense ATA matrix formats with synthetic pattern data.
"""

import sys
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image

sys.path.append("src")
from utils.dense_ata_matrix import DenseATAMatrix
from utils.diagonal_ata_matrix import DiagonalATAMatrix
from utils.frame_optimizer import optimize_frame_led_values
from utils.single_block_sparse_tensor import SingleBlockMixedSparseTensor


def load_synthetic_data():
    """Load synthetic pattern data with both DIA and Dense matrices."""
    print("Loading synthetic pattern data...")

    # Load original synthetic data with working DIA matrix
    original_data = np.load("diffusion_patterns/synthetic_2624_uint8.npz", allow_pickle=True)

    # Load our newly created Dense matrix
    dense_data = np.load("diffusion_patterns/synthetic-dense-test.npz", allow_pickle=True)

    # Extract components
    mixed_tensor_dict = original_data["mixed_tensor"].item()
    mixed_tensor = SingleBlockMixedSparseTensor.from_dict(mixed_tensor_dict)

    # Load both matrix formats
    dia_matrix_dict = original_data["dia_matrix"].item()
    dia_matrix = DiagonalATAMatrix.from_dict(dia_matrix_dict)

    dense_matrix_dict = dense_data["dense_ata_matrix_rebuilt"].item()
    dense_matrix = DenseATAMatrix.from_dict(dense_matrix_dict)

    # Load ATA inverse (same for both - we'll use the original one)
    ata_inverse = original_data["ata_inverse"]

    print(f"✅ Loaded synthetic data:")
    print(f"   Mixed tensor: {mixed_tensor.batch_size} LEDs")
    print(f"   DIA matrix: {dia_matrix.led_count} LEDs, k={dia_matrix.k} diagonals")
    print(f"   Dense matrix: {dense_matrix.led_count} LEDs, {dense_matrix.memory_mb:.1f}MB")
    print(f"   ATA inverse: {ata_inverse.shape}")

    return mixed_tensor, dia_matrix, dense_matrix, ata_inverse


def load_and_prepare_image(image_path, target_size=(800, 480)):
    """Load and prepare source image for optimization."""
    print(f"Loading image: {image_path}")

    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to target frame size
        img_resized = img.resize(target_size, Image.LANCZOS)

        # Convert to numpy array in HWC format, then to planar CHW
        img_array = np.array(img_resized, dtype=np.uint8)  # Shape: (480, 800, 3)
        img_planar = img_array.transpose(2, 0, 1)  # Shape: (3, 480, 800)

        # Convert to GPU array
        target_frame = cp.asarray(img_planar)

        print(f"✅ Image prepared: {target_frame.shape}, dtype={target_frame.dtype}")
        return target_frame


def run_optimization_comparison(mixed_tensor, dia_matrix, dense_matrix, ata_inverse, target_frame, image_name):
    """Run optimization with both DIA and Dense matrices and compare results."""
    print(f"\n=== OPTIMIZING {image_name.upper()} ===")

    # Optimization parameters
    opt_params = {
        "max_iterations": 10,
        "convergence_threshold": 0.3,
        "step_size_scaling": 0.9,
        "compute_error_metrics": True,
        "debug": True,
    }

    # IMPORTANT: Both optimizations must use the SAME ATA inverse for initialization
    # Only the ata_matrix parameter should differ (DIA vs Dense format)

    # Run optimization with DIA matrix (using original ATA inverse for initialization)
    print("\n🔵 Running optimization with DIA matrix...")
    dia_result = optimize_frame_led_values(
        target_frame=target_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dia_matrix,  # DIA format for A^T A @ x operations
        ata_inverse=ata_inverse,  # Original ATA inverse for initialization
        **opt_params,
    )

    # Run optimization with Dense matrix (using SAME ATA inverse for initialization)
    print("\n🟢 Running optimization with Dense matrix...")
    dense_result = optimize_frame_led_values(
        target_frame=target_frame,
        at_matrix=mixed_tensor,
        ata_matrix=dense_matrix,  # Dense format for A^T A @ x operations
        ata_inverse=ata_inverse,  # SAME ATA inverse for initialization
        **opt_params,
    )

    # Compare results
    print(f"\n=== COMPARISON RESULTS FOR {image_name.upper()} ===")

    # Convert to CPU for comparison
    dia_values = cp.asnumpy(dia_result.led_values)
    dense_values = cp.asnumpy(dense_result.led_values)

    print(f"DIA result shape: {dia_values.shape}, range: [{dia_values.min()}, {dia_values.max()}]")
    print(f"Dense result shape: {dense_values.shape}, range: [{dense_values.min()}, {dense_values.max()}]")

    # Calculate differences
    max_diff = np.max(np.abs(dia_values - dense_values))
    mean_diff = np.mean(np.abs(dia_values - dense_values))
    rms_diff = np.sqrt(np.mean((dia_values - dense_values) ** 2))

    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    print(f"RMS difference: {rms_diff}")

    # Check if results are essentially identical
    tolerance = 1.0  # Allow 1 intensity level difference on 0-255 scale
    if max_diff <= tolerance:
        print(f"✅ SUCCESS: Results are identical within tolerance ({tolerance})")
        success = True
    else:
        print(f"❌ FAILURE: Results differ by more than tolerance ({tolerance})")
        success = False

    # Compare error metrics
    print("\nError metrics comparison:")
    for metric in ["mse", "mae", "psnr"]:
        if metric in dia_result.error_metrics and metric in dense_result.error_metrics:
            dia_val = dia_result.error_metrics[metric]
            dense_val = dense_result.error_metrics[metric]
            diff = abs(dia_val - dense_val)
            print(f"  {metric}: DIA={dia_val:.6f}, Dense={dense_val:.6f}, diff={diff:.6f}")

    print(f"Iterations: DIA={dia_result.iterations}, Dense={dense_result.iterations}")
    print(f"Converged: DIA={dia_result.converged}, Dense={dense_result.converged}")

    return success, max_diff, mean_diff, rms_diff


def main():
    """Main test function."""
    print("🚀 DIA vs Dense ATA Matrix Optimization Comparison Test")
    print("=" * 60)

    try:
        # Load synthetic data
        mixed_tensor, dia_matrix, dense_matrix, ata_inverse = load_synthetic_data()

        # Test images
        test_images = ["images/source/flower.jpg", "images/source/pexels.jpg", "images/source/warhol_marilyn.jpg"]

        results = []
        all_success = True

        for image_path in test_images:
            if not Path(image_path).exists():
                print(f"⚠️  Skipping {image_path} - file not found")
                continue

            image_name = Path(image_path).stem

            # Load and prepare image
            target_frame = load_and_prepare_image(image_path)

            # Run comparison
            success, max_diff, mean_diff, rms_diff = run_optimization_comparison(
                mixed_tensor, dia_matrix, dense_matrix, ata_inverse, target_frame, image_name
            )

            results.append(
                {
                    "image": image_name,
                    "success": success,
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "rms_diff": rms_diff,
                }
            )

            if not success:
                all_success = False

        # Final summary
        print("\n" + "=" * 60)
        print("📊 FINAL SUMMARY")
        print("=" * 60)

        for result in results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"{result['image']:<15} {status} (max_diff: {result['max_diff']:.3f})")

        if all_success:
            print(f"\n🎉 ALL TESTS PASSED!")
            print("DIA and Dense ATA matrices produce identical optimization results!")
            return 0
        else:
            print(f"\n❌ SOME TESTS FAILED!")
            print("DIA and Dense ATA matrices produce different results!")
            return 1

    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
